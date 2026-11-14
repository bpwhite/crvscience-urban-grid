//! Coupled Fire-Following-Earthquake (`FFE`) Conflagration Perimeter Road Closures, Wetterberg/Intini
//! Koschmieder Smoke-Visibility Speed Degradation, Protected Emergency Ambulance/Hospital/`LZ`
//! Lifelines, Staged Neighborhood Evacuation, and Arterial Contraflow Lane Reversal
//!
//! Grounded in:
//! - FHWA *Disaster Recovery Reform Act Section 1209 (Contraflow Provision)*
//!   (`https://ops.fhwa.dot.gov/publications/fhwahop20026/fhwahop20026.pdf`), FHWA *Using Highways
//!   for No-Notice Evacuations* (`https://ops.fhwa.dot.gov/publications/evac_primer_nn/primer.pdf`),
//!   and Kim, Shekhar & Min (2008), *Contraflow Transportation Network Reconfiguration for Evacuation
//!   Route Planning* (`https://www.computer.org/csdl/journal/tk/2008/08/ttk2008081115/13rRUwdrdL6`).
//! - Li, Cova & Dennison, *Setting Wildfire Evacuation Triggers by Coupling Fire and Traffic
//!   Simulation Models* (`https://www.researchgate.net/publication/327476427_Setting_Wildfire_Evacuation_Triggers_by_Coupling_Fire_and_Traffic_Simulation_Models_A_Spatiotemporal_GIS_Approach`)
//!   and *Evaluating Traffic Operation Conditions during Wildfire Evacuation*
//!   (`https://www.mdpi.com/2571-6255/6/5/184`).
//! - Wetterberg et al. (2021) / Intini et al. (2022) / Ronchi et al. (2023), *The verification of
//!   wildland-urban interface fire evacuation models* (`https://pmc.ncbi.nlm.nih.gov/articles/PMC10220130/`).

use crate::aeromedical_medevac::{
    koschmieder_smoke_visibility_and_ceiling, AeromedicalMedevacConfig,
};
use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::ffe_conflagration::{
    hamada_hazus_spread_rates, scawthorn_hazus_ignition_rate_per_mmsqft,
};
use crate::hospital_surge::synthesize_hospital_surge_roster;
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi, BuriedPipeMaterialClass,
};
use crate::utilities::{MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const EVACUATION_PARCEL_CSV_HEADER: &str = "parcel_id,zone_code,evac_stage,release_delay_min,evac_vehicles,mean_smoke_vis_sm,mean_clearance_min,fire_overrun_prob_pct";
pub const EVACUATION_CORRIDOR_CSV_HEADER: &str = "segment_id,road_name,base_lanes,contraflow_added_lanes,protected_ems_lz_corridor,thermal_closure_prob_pct,mean_smoke_speed_factor,peak_vc_ratio";

const SQFT_PER_M2: f64 = 10.763_910_4;
const METERS_PER_STATUTE_MILE: f64 = 1_609.344;

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64_open01(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        u.clamp(1e-12, 1.0 - 1e-12)
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[inline]
fn sample_poisson_icdf(lambda: f64, u01: f64) -> usize {
    if lambda <= 1e-12 {
        return 0;
    }
    let lam = lambda.min(45.0);
    let mut p = (-lam).exp();
    let mut cdf = p;
    if u01 <= cdf {
        return 0;
    }
    for k in 1..=96 {
        p *= lam / (k as f64);
        cdf += p;
        if u01 <= cdf {
            return k;
        }
    }
    96
}

fn cholesky_lower(corr: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = corr[i * n + j];
            if i == j {
                s += 1e-9;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = s.max(1e-12).sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j].max(1e-12);
            }
        }
    }
    l
}

/// Neighborhood mobilization sequencing strategy under advancing FFE conflagrations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvacuationStagingStrategy {
    /// Simultaneous uncoordinated departure (`t = 0` mass loading, severe bottleneck gridlock).
    SimultaneousUnstaged,
    /// 3-Stage Fire-Proximity Trigger (`Stage 1 <= 45m fire front @ 0m`, `Stage 2 Buffer`, `Stage 3 Outer`).
    FireProximityThreeStage,
    /// Vulnerability + Fire-Proximity Priority (prioritizes fire front + high-density `R3`/`MU-4`/`CBD` blocks).
    VulnerabilityAndFireWeighted,
}

impl Default for EvacuationStagingStrategy {
    fn default() -> Self {
        Self::FireProximityThreeStage
    }
}

impl EvacuationStagingStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::SimultaneousUnstaged => "Simultaneous All-Zones Departure (Unstaged)",
            Self::FireProximityThreeStage => "Staged 3-Zone Fire-Proximity Release (Cova/Pel)",
            Self::VulnerabilityAndFireWeighted => "Staged Fire + High-Density Vulnerability Priority",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::SimultaneousUnstaged => "Unstaged",
            Self::FireProximityThreeStage => "Staged-3Zone",
            Self::VulnerabilityAndFireWeighted => "Staged-Vuln",
        }
    }
}

/// Arterial contraflow lane-reversal configuration & emergency responder corridor protection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContraflowReversalPolicy {
    /// Standard bidirectional lane operation (no inbound lanes reversed).
    NoContraflowBaseline,
    /// Aggressive full one-way-out reversal on outbound arterials (maximizes civilian outflow, severs inbound EMS!).
    FullUnprotectedOneWayOut,
    /// FHWA / Wolshon EMS-Protected Contraflow: preserves `>= 1` inbound lane on Ambulance / Hospital / `LZ` spines while reversing surplus outbound arterial lanes!
    EmsProtectedArterialContraflow,
}

impl Default for ContraflowReversalPolicy {
    fn default() -> Self {
        Self::EmsProtectedArterialContraflow
    }
}

impl ContraflowReversalPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::NoContraflowBaseline => "No Contraflow (Standard Two-Way Lanes)",
            Self::FullUnprotectedOneWayOut => "Full One-Way-Out Reversal (Blocks Inbound EMS)",
            Self::EmsProtectedArterialContraflow => {
                "EMS/LZ-Protected Arterial Contraflow (FHWA/Wolshon)"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::NoContraflowBaseline => "No-CF",
            Self::FullUnprotectedOneWayOut => "Full-1Way",
            Self::EmsProtectedArterialContraflow => "EMS-Prot-CF",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvacuationContraflowConfig {
    /// Coupled medevac (`LZ`s + ambient visibility), hospital surge, seismic bridge & FFE parameters.
    pub medevac: AeromedicalMedevacConfig,
    pub staging_strategy: EvacuationStagingStrategy,
    pub contraflow_policy: ContraflowReversalPolicy,
    /// Release offset (`minutes`, default `20.0 min`) for Stage-2 intermediate buffer neighborhoods.
    pub stage2_release_delay_min: f64,
    /// Release offset (`minutes`, default `45.0 min`) for Stage-3 outer-ring precautionary neighborhoods.
    pub stage3_release_delay_min: f64,
    /// Evacuating private vehicles generated per resident/commuter equivalent (default `0.58`).
    pub evac_vehicles_per_capita: f64,
    /// Radiant heat & ember road-closure standoff buffer around active burning parcels (`meters`, default `56.0 m`).
    pub fire_perimeter_road_buffer_m: f64,
}

impl Default for EvacuationContraflowConfig {
    fn default() -> Self {
        Self {
            medevac: AeromedicalMedevacConfig::default(),
            staging_strategy: EvacuationStagingStrategy::FireProximityThreeStage,
            contraflow_policy: ContraflowReversalPolicy::EmsProtectedArterialContraflow,
            stage2_release_delay_min: 20.0,
            stage3_release_delay_min: 45.0,
            evac_vehicles_per_capita: 0.58,
            fire_perimeter_road_buffer_m: 56.0,
        }
    }
}

/// Wetterberg et al. (2021) / Intini et al. (2022) WUI-NITY smoke-impaired driving speed & saturation-flow
/// multipliers `(speed_factor, capacity_factor)` as a function of Koschmieder visibility `vis_sm` (`statute miles`).
#[inline]
pub fn smoke_visibility_speed_and_capacity_factors(vis_sm: f64) -> (f64, f64) {
    let vis_m = (vis_sm.max(0.01)) * METERS_PER_STATUTE_MILE;
    if vis_m >= 1_200.0 {
        (1.0, 1.0)
    } else if vis_m >= 150.0 {
        let u = ((vis_m - 150.0) / 1_050.0).clamp(0.0, 1.0);
        let v_fac = 0.42 + 0.58 * u.powf(0.65);
        let q_fac = 0.54 + 0.46 * u.powf(0.70);
        (v_fac, q_fac)
    } else {
        let u = (vis_m / 150.0).clamp(0.0, 1.0);
        (0.22 + 0.20 * u, 0.34 + 0.20 * u)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContraflowCorridorSegment {
    pub segment_id: usize,
    pub twin_inbound_seg_id: Option<usize>,
    pub road_name: String,
    pub from_pos: Vec2,
    pub to_pos: Vec2,
    pub base_lanes: usize,
    pub contraflow_added_lanes: usize,
    pub remaining_inbound_lanes: usize,
    pub protected_ems_lz_corridor: bool,
    pub thermal_closure_prob_pct: f64,
    pub mean_smoke_speed_factor: f64,
    pub peak_vc_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelEvacuationStageRow {
    pub parcel_id: usize,
    pub zone_code: ZoneCode,
    pub center: Vec2,
    /// Assigned staging tier (`1 = Immediate Fire-Front`, `2 = Intermediate Buffer`, `3 = Precautionary Outer`).
    pub evac_stage: u8,
    pub release_delay_min: f64,
    pub evac_vehicles: f64,
    pub mean_smoke_vis_sm: f64,
    pub mean_clearance_min: f64,
    pub fire_overrun_prob_pct: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EvacuationWavePoint {
    pub minute: f64,
    pub evacuated_pct: f64,
    pub active_network_queue_vehs: f64,
    pub thermally_closed_segments: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvacuationContraflowReport {
    pub config: EvacuationContraflowConfig,
    pub total_evacuees: f64,
    pub total_evac_vehicles: f64,
    pub contraflow_reversed_segments_count: usize,
    pub contraflow_lane_km_added: f64,
    pub protected_ems_lz_segments_count: usize,
    pub expected_thermally_closed_segments: f64,
    pub mean_road_smoke_speed_factor: f64,
    /// Time (`minutes`) for 50% of evacuating households to clear into safe perimeter refuges.
    pub t50_clearance_min: f64,
    /// Time (`minutes`) for 90% of evacuating households to clear (`NFPA / FEMA` primary clearance metric).
    pub t90_clearance_min: f64,
    /// Time (`minutes`) for 95% of evacuating households to clear.
    pub t95_clearance_min: f64,
    pub peak_bottleneck_vc_ratio: f64,
    /// Expected number of residents whose evacuation route/queue is overtaken by advancing FFE flame fronts!
    pub expected_fire_overrun_residents: f64,
    /// Percentage (`0..=100%`) of Fire-Station, Trauma-Hospital & Park-`LZ` emergency corridors retaining inbound responder lanes.
    pub ems_lz_corridor_access_preserved_pct: f64,
    pub mean_ems_inbound_response_min: f64,
    pub timeline: Vec<EvacuationWavePoint>,
    pub corridor_rows: Vec<ContraflowCorridorSegment>,
    pub parcel_rows: Vec<ParcelEvacuationStageRow>,
    pub solve_time_ms: f64,
}

impl EvacuationContraflowReport {
    pub fn to_parcel_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.parcel_rows.len() * 96 + 256);
        out.push_str(EVACUATION_PARCEL_CSV_HEADER);
        out.push('\n');
        for r in &self.parcel_rows {
            out.push_str(&format!(
                "{},{:?},{},{:.1},{:.1},{:.2},{:.2},{:.2}\n",
                r.parcel_id,
                r.zone_code,
                r.evac_stage,
                r.release_delay_min,
                r.evac_vehicles,
                r.mean_smoke_vis_sm,
                r.mean_clearance_min,
                r.fire_overrun_prob_pct
            ));
        }
        out
    }

    pub fn to_corridor_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.corridor_rows.len() * 112 + 256);
        out.push_str(EVACUATION_CORRIDOR_CSV_HEADER);
        out.push('\n');
        for c in &self.corridor_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{},{},{:.1},{:.3},{:.2}\n",
                c.segment_id,
                c.road_name.replace('"', "'"),
                c.base_lanes,
                c.contraflow_added_lanes,
                c.protected_ems_lz_corridor,
                c.thermal_closure_prob_pct,
                c.mean_smoke_speed_factor,
                c.peak_vc_ratio
            ));
        }
        out
    }

    pub fn export_parcel_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_parcel_csv_string())?;
        Ok(self.parcel_rows.len())
    }
}

fn median_sa10_gmpe_g(mw: f64, r_rup_km: f64, vs30_mps: f64) -> f64 {
    let m = mw.clamp(5.0, 9.3);
    let r = r_rup_km.max(1.5);
    let h_eff = (2.5 + (0.48 * (m - 6.0)).exp().min(18.0)).max(3.5);
    let r_eff = (r * r + h_eff * h_eff).sqrt();
    let dm = m - 6.5;
    let mag_term = if dm <= 0.0 {
        0.92 * dm
    } else {
        0.74 * dm - 0.085 * dm * dm
    };
    let dist_term = -(1.08 + 0.055 * (m - 6.5)) * (r_eff / 10.0).ln() - 0.0032 * r_eff;
    let site_term = 0.38 * (760.0 / vs30_mps.clamp(140.0, 1100.0)).ln();
    ((0.24_f64).ln() + mag_term + dist_term + site_term)
        .exp()
        .clamp(0.005, 3.50)
}

#[derive(Copy, Clone, PartialEq)]
struct DijkNode {
    cost: f64,
    id: usize,
}
impl Eq for DijkNode {}
impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Identifies outbound arterial contraflow candidates and protected emergency Ambulance / Hospital /
/// Medevac-`LZ` spine segments across `net`.
pub fn plan_contraflow_and_protected_ems_corridors(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &EvacuationContraflowConfig,
) -> (Vec<Option<usize>>, Vec<bool>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let n_segs = net.segments.len();
    let b = net.bounds;
    let center = Vec2 {
        x: 0.5 * (b.min_x + b.max_x),
        y: 0.5 * (b.min_y + b.max_y),
    };

    // 1. Pair opposite-direction twin segments `(u -> v)` and `(v -> u)`
    let mut twin_seg: Vec<Option<usize>> = vec![None; n_segs];
    for s in &net.segments {
        if let Some(tw) = net
            .segments
            .iter()
            .find(|o| o.from_junc == s.to_junc && o.to_junc == s.from_junc && !o.closed)
        {
            twin_seg[s.id] = Some(tw.id);
        }
    }

    // 2. Identify Protected Emergency Ambulance / Fire / Hospital / Park-LZ Spine segments
    let hospitals = synthesize_hospital_surge_roster(net, facilities, &cfg.medevac.surge);
    let mut crit_anchors: Vec<Vec2> = facilities.iter().map(|f| f.world_pos).collect();
    for h in &hospitals {
        crit_anchors.push(h.world_pos);
    }
    // Include Park Emergency Helipad LZs
    for p in parcels.iter().filter(|p| p.zone == ZoneCode::PARK).take(3) {
        crit_anchors.push(p.center);
    }

    let mut protected_ems_lz = vec![false; n_segs];
    for s in &net.segments {
        if s.closed || s.base_pts.is_empty() {
            continue;
        }
        let mid = s.base_pts[s.base_pts.len() / 2];
        // Segments hosting or directly feeding within 165 m of any Hospital, Fire Station, or Park LZ
        // OR major primary cross-town hospital connectors form the Protected Emergency Spine
        let near_anchor = crit_anchors.iter().any(|a| a.dist(mid) <= 165.0);
        if near_anchor {
            protected_ems_lz[s.id] = true;
            if let Some(tw) = twin_seg[s.id] {
                protected_ems_lz[tw] = true;
            }
        }
    }

    // 3. Determine contraflow lane additions (`cf_add[sid]`) & remaining inbound lanes (`rem_in[sid]`)
    let mut cf_added = vec![0usize; n_segs];
    let mut rem_inbound: Vec<usize> = net.segments.iter().map(|s| s.lane_count).collect();
    let mut perimeter_shelter_segs: Vec<usize> = Vec::new();

    let span_x = (b.max_x - b.min_x).max(100.0);
    let span_y = (b.max_y - b.min_y).max(100.0);
    for s in &net.segments {
        if s.closed || s.base_pts.is_empty() {
            continue;
        }
        let p_from = net.junctions[s.from_junc].pos;
        let p_to = net.junctions[s.to_junc].pos;
        let d_from = p_from.dist(center);
        let d_to = p_to.dist(center);

        // Perimeter exit sink segments near map boundary heading outward
        let near_edge = p_to.x <= b.min_x + 0.14 * span_x
            || p_to.x >= b.max_x - 0.14 * span_x
            || p_to.y <= b.min_y + 0.14 * span_y
            || p_to.y >= b.max_y - 0.14 * span_y;
        if near_edge && d_to >= d_from {
            perimeter_shelter_segs.push(s.id);
        }

        // Candidate outbound radial evacuation arterial: `d_to > d_from + 6.0` & `priority >= 1`
        if d_to > d_from + 6.0 && (s.priority >= 1 || s.lane_count >= 2) {
            if let Some(in_sid) = twin_seg[s.id] {
                // Avoid double-flipping both directions
                if cf_added[in_sid] > 0 {
                    continue;
                }
                let in_lanes = net.segments[in_sid].lane_count;
                match cfg.contraflow_policy {
                    ContraflowReversalPolicy::NoContraflowBaseline => {}
                    ContraflowReversalPolicy::FullUnprotectedOneWayOut => {
                        // Flip ALL inbound lanes to outbound (even if single-lane or EMS corridor!)
                        if in_lanes >= 1 {
                            cf_added[s.id] = in_lanes;
                            rem_inbound[in_sid] = 0;
                        }
                    }
                    ContraflowReversalPolicy::EmsProtectedArterialContraflow => {
                        if protected_ems_lz[s.id] || protected_ems_lz[in_sid] {
                            // Strictly preserve >= 1 dedicated inbound Ambulance / Fire / LZ lane!
                            if in_lanes >= 2 {
                                cf_added[s.id] = in_lanes - 1;
                                rem_inbound[in_sid] = 1;
                            }
                        } else if in_lanes >= 2 {
                            cf_added[s.id] = in_lanes - 1;
                            rem_inbound[in_sid] = 1;
                        } else if s.lane_count >= 1 && in_lanes == 1 {
                            // Parallel non-EMS arterial can donate 1 lane while preserving twin EMS spine
                            cf_added[s.id] = 1;
                            rem_inbound[in_sid] = 0;
                        }
                    }
                }
            }
        }
    }

    if perimeter_shelter_segs.is_empty() && !net.segments.is_empty() {
        perimeter_shelter_segs.push(0);
    }

    (
        twin_seg,
        protected_ems_lz,
        cf_added,
        rem_inbound,
        perimeter_shelter_segs,
    )
}

/// Runs `Rayon`-parallelized post-earthquake + active `FFE` conflagration perimeter road-closure,
/// Koschmieder smoke speed-reduction, staged neighborhood evacuation, and arterial contraflow
/// Monte Carlo simulations.
pub fn run_evacuation_contraflow_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &EvacuationContraflowConfig,
) -> EvacuationContraflowReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let n_segs = net.segments.len();
    let n_junc = net.junctions.len();
    let trials = cfg.medevac.surge.ffe.seismic.monte_carlo_trials.clamp(20, 5_000);

    let (twin_seg, protected_ems_lz, cf_added, rem_inbound, shelter_segs) =
        plan_contraflow_and_protected_ems_corridors(net, parcels, facilities, cfg);

    let fire_station_segs: Vec<usize> = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation && f.segment_id < n_segs)
        .map(|f| f.segment_id)
        .collect();
    let n_engines = (fire_station_segs.len().max(1) * 2).max(2);

    let b = net.bounds;
    let west_x = b.min_x;
    let mut parcel_pop = Vec::with_capacity(n_parcels);
    let mut parcel_vehs = Vec::with_capacity(n_parcels);
    let mut parcel_tfa = Vec::with_capacity(n_parcels);
    let mut parcel_sa10 = Vec::with_capacity(n_parcels);
    let mut parcel_pga = Vec::with_capacity(n_parcels);
    let mut parcel_near_junc = Vec::with_capacity(n_parcels);
    let mut parcel_near_seg = Vec::with_capacity(n_parcels);
    let mut parcel_hamada = Vec::with_capacity(n_parcels);

    for p in parcels {
        let sp = p.zone.spec();
        let pop = if p.zone == ZoneCode::PARK {
            2.0
        } else {
            (p.residents() + 0.45 * p.jobs()).max(16.0)
        };
        let vehs = pop * cfg.evac_vehicles_per_capita.clamp(0.15, 1.2);
        let raw_tfa = (p.area_ha * 10_000.0 * sp.far * SQFT_PER_M2) / 1_000_000.0;
        let tfa = if p.zone == ZoneCode::PARK {
            0.001
        } else {
            raw_tfa.max(0.045)
        };
        let loc_km = ((p.center.x - west_x).max(0.0)) / 1000.0;
        let r_km = (cfg.medevac.surge.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(
            cfg.medevac.surge.ffe.seismic.moment_magnitude_mw,
            r_km,
            cfg.medevac.surge.ffe.seismic.vs30_m_per_s,
        );
        let pga = (0.86
            * sa10
            * (cfg.medevac.surge.ffe.seismic.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
        .clamp(0.01, 3.0);

        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                p.center
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&p.center.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        let b_sid = (0..n_segs)
            .filter(|&sid| !net.segments[sid].closed && !net.segments[sid].base_pts.is_empty())
            .min_by(|&a, &bs| {
                let ma = net.segments[a].base_pts[net.segments[a].base_pts.len() / 2];
                let mb = net.segments[bs].base_pts[net.segments[bs].base_pts.len() / 2];
                p.center
                    .dist(ma)
                    .partial_cmp(&p.center.dist(mb))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);

        parcel_pop.push(pop);
        parcel_vehs.push(vehs);
        parcel_tfa.push(tfa);
        parcel_sa10.push(sa10);
        parcel_pga.push(pga);
        parcel_near_junc.push(nj);
        parcel_near_seg.push(b_sid);
        parcel_hamada.push(hamada_hazus_spread_rates(
            p.zone,
            cfg.medevac.surge.ffe.wind_speed_mps,
        ));
    }

    // Segment midpoints & parcel proximity for thermal perimeter closures
    let seg_mids: Vec<Vec2> = net
        .segments
        .iter()
        .map(|s| {
            if s.base_pts.is_empty() {
                Vec2 { x: 0.0, y: 0.0 }
            } else {
                s.base_pts[s.base_pts.len() / 2]
            }
        })
        .collect();
    let fire_buf_m = cfg.fire_perimeter_road_buffer_m.clamp(20.0, 180.0);
    let mut seg_adj_parcels: Vec<Vec<usize>> = vec![Vec::new(); n_segs];
    for sid in 0..n_segs {
        for (pi, p) in parcels.iter().enumerate() {
            let dx = (seg_mids[sid].x - p.max.x)
                .max(p.min.x - seg_mids[sid].x)
                .max(0.0);
            let dy = (seg_mids[sid].y - p.max.y)
                .max(p.min.y - seg_mids[sid].y)
                .max(0.0);
            if dx.hypot(dy) <= fire_buf_m {
                seg_adj_parcels[sid].push(pi);
            }
        }
    }

    // Hamada fire-spread graph
    let wind_rad = cfg.medevac.surge.ffe.wind_direction_deg.to_radians();
    let wx_hat = wind_rad.cos();
    let wy_hat = wind_rad.sin();
    let horizon_m = cfg.medevac.surge.ffe.conflagration_horizon_mins.clamp(30.0, 360.0);
    let mut fire_spread_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_parcels];
    for i in 0..n_parcels {
        if parcels[i].zone == ZoneCode::PARK {
            continue;
        }
        let pi = &parcels[i];
        let rates = parcel_hamada[i];
        for k in 0..n_parcels {
            if i == k || parcels[k].zone == ZoneCode::PARK {
                continue;
            }
            let pk = &parcels[k];
            let dx = pk.center.x - pi.center.x;
            let dy = pk.center.y - pi.center.y;
            let center_dist = dx.hypot(dy);
            if !(1.0..=280.0).contains(&center_dist) {
                continue;
            }
            let gap_x = (pk.min.x - pi.max.x).max(pi.min.x - pk.max.x).max(0.0);
            let gap_y = (pk.min.y - pi.max.y).max(pi.min.y - pk.max.y).max(0.0);
            let edge_gap_m = gap_x.hypot(gap_y);
            let cos_th = (dx * wx_hat + dy * wy_hat) / center_dist;
            let sin_th = (-dx * wy_hat + dy * wx_hat).abs() / center_dist;
            let dir_jump_cap = if cos_th >= 0.25 {
                rates.max_jump_gap_downwind_m * cos_th + 14.0 * (1.0 - cos_th)
            } else if cos_th >= -0.25 {
                14.5 + 0.85 * cfg.medevac.surge.ffe.wind_speed_mps
            } else {
                10.5 + 0.25 * cfg.medevac.surge.ffe.wind_speed_mps
            };
            if edge_gap_m > dir_jump_cap {
                continue;
            }
            let a_semi = if cos_th >= 0.0 {
                rates.v_down_m_min
            } else {
                rates.v_up_m_min
            };
            let b_semi = rates.v_cross_m_min;
            let inv_r = ((cos_th / a_semi.max(0.05)).powi(2) + (sin_th / b_semi.max(0.05)).powi(2))
                .sqrt();
            let v_theta = (1.0 / inv_r.max(1e-6)).max(0.05);
            let step_min = center_dist / v_theta + 2.5 + 0.18 * edge_gap_m;
            if step_min <= horizon_m {
                fire_spread_adj[i].push((k, step_min));
            }
        }
    }

    let bridges = enumerate_seismic_bridge_inventory(net, facilities, &cfg.medevac.surge.ffe.seismic);
    let n_br = bridges.len();

    let mut u_corridors: Vec<(usize, f64, Vec2, bool)> = Vec::new();
    let mut junc_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_junc];
    for s in &net.segments {
        if s.from_junc < s.to_junc
            || !net
                .segments
                .iter()
                .any(|o| o.from_junc == s.to_junc && o.to_junc == s.from_junc)
        {
            let cid = u_corridors.len();
            u_corridors.push((cid, s.length.max(12.0), seg_mids[s.id], s.priority >= 2));
            junc_adj[s.from_junc].push((s.to_junc, cid));
            junc_adj[s.to_junc].push((s.from_junc, cid));
        }
    }
    let n_corr = u_corridors.len();
    let n_rep_corr = ((n_corr as f64)
        * (cfg
            .medevac
            .surge
            .ffe
            .seismic
            .seismic_pipe_replacement_pct
            .clamp(0.0, 100.0)
            / 100.0))
        .round() as usize;
    let wtp_junc = (0..n_junc)
        .min_by(|&a, &bj| {
            let tgt = Vec2 {
                x: b.min_x,
                y: 0.5 * (b.min_y + b.max_y),
            };
            net.junctions[a]
                .pos
                .dist(tgt)
                .partial_cmp(&net.junctions[bj].pos.dist(tgt))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);

    // Base road adjacency (outbound reverse Dijkstra from `shelter_segs`)
    let mut out_neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    let mut in_neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let sig_pen = match net.junctions[seg.to_junc].control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => 6.5,
            ControlMode::Roundabout => 3.2,
            ControlMode::PermissiveYield => 2.0,
        };
        for &lid in &seg.lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let conn = &net.connectors[cid];
                let out_sid = conn.to_seg_id;
                if net.segments[out_sid].closed {
                    continue;
                }
                let t_pen = match conn.turn_type {
                    TurnType::Through => 0.4,
                    TurnType::Right => 1.8,
                    TurnType::Left => 2.8,
                };
                if !out_neighbors[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    out_neighbors[seg.id].push((out_sid, sig_pen + t_pen));
                    in_neighbors[out_sid].push((seg.id, sig_pen + t_pen));
                }
            }
        }
    }

    // Effective urban signalized lane capacities (`vehicles / 10-min wave`):
    // Arterial intersection saturation flow ~1,050 veh/hr = 175 veh/10m; Wolshon contraflow ~0.82 efficiency = 145 veh/10m!
    let eff_evac_cap_10m: Vec<f64> = (0..n_segs)
        .map(|sid| {
            let base_l = rem_inbound[sid] as f64;
            let cf_l = cf_added[sid] as f64;
            (base_l * 175.0 + cf_l * 145.0).max(0.0)
        })
        .collect();

    let n_sites = n_parcels + n_br;
    let mut site_pts = Vec::with_capacity(n_sites);
    for p in parcels {
        site_pts.push(p.center);
    }
    for br in &bridges {
        site_pts.push(seg_mids[br.segment_ids[0]]);
    }
    let range_km = if cfg.medevac.surge.ffe.seismic.vs30_m_per_s < 270.0 {
        29.5
    } else {
        25.0
    };
    let mut corr_mat = vec![0.0; n_sites * n_sites];
    for i in 0..n_sites {
        for j in 0..n_sites {
            let d_km = site_pts[i].dist(site_pts[j]) / 1000.0;
            corr_mat[i * n_sites + j] = (-3.0 * d_km / range_km).exp();
        }
    }
    let chol = cholesky_lower(&corr_mat, n_sites);

    // 18 waves of 10 minutes = 180 minutes evacuation horizon
    let n_waves = 18usize;
    let dt_min = 10.0;
    let total_evacuees: f64 = parcel_pop.iter().sum();
    let total_evac_vehs: f64 = parcel_vehs.iter().sum();

    let ems_prot_indices: Vec<usize> = (0..n_segs).filter(|&sid| protected_ems_lz[sid]).collect();
    let ems_preserved_cnt = ems_prot_indices
        .iter()
        .filter(|&&sid| rem_inbound[sid] >= 1)
        .count();
    let ems_preserved_pct = if ems_prot_indices.is_empty() {
        100.0
    } else {
        (ems_preserved_cnt as f64 / ems_prot_indices.len() as f64) * 100.0
    };

    #[derive(Clone)]
    struct EvacTrialOut {
        t50_min: f64,
        t90_min: f64,
        t95_min: f64,
        peak_vc: f64,
        overrun_res: f64,
        therm_closed_cnt: f64,
        mean_smoke_v_fac: f64,
        mean_ems_in_min: f64,
        seg_therm_closed: Vec<f64>,
        seg_smoke_v_fac: Vec<f64>,
        seg_peak_vc: Vec<f64>,
        parcel_stage: Vec<u8>,
        parcel_rel_min: Vec<f64>,
        parcel_vis_sm: Vec<f64>,
        parcel_clear_min: Vec<f64>,
        parcel_overrun: Vec<f64>,
        wave_evac_pct: Vec<f64>,
        wave_queue_vehs: Vec<f64>,
        wave_therm_closed: Vec<f64>,
    }

    let seed_base = cfg.medevac.surge.ffe.seismic.random_seed ^ 0xE7AC_C04F_3500_B21D;
    let evp_boost = if evp_cfg.enabled { 1.25 } else { 1.0 };

    let trial_outs: Vec<EvacTrialOut> = (0..trials)
        .into_par_iter()
        .map(|trial_idx| {
            let mut rng = SplitMix64::new(
                seed_base.wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let eta_inter = rng.next_standard_normal();
            let mut z_ind = vec![0.0; n_sites];
            for z in &mut z_ind {
                *z = rng.next_standard_normal();
            }
            let mut eps_intra = vec![0.0; n_sites];
            for i in 0..n_sites {
                let mut s = 0.0;
                for k in 0..=i {
                    s += chol[i * n_sites + k] * z_ind[k];
                }
                eps_intra[i] = s;
            }
            let tau_b = cfg.medevac.surge.ffe.seismic.inter_event_sigma_ln;
            let phi_w = cfg.medevac.surge.ffe.seismic.intra_event_sigma_ln;

            // 1. Seismic bridge closures
            let mut br_closed = vec![false; n_segs];
            for (bi, br) in bridges.iter().enumerate() {
                let sa_r = (br.median_site_sa10_g.ln()
                    + tau_b * eta_inter
                    + phi_w * eps_intra[n_parcels + bi])
                    .exp();
                let probs = br.effective_archetype.exceedance_probs(sa_r);
                if rng.next_f64_open01() < probs[2] {
                    for &sid in &br.segment_ids {
                        if sid < n_segs {
                            br_closed[sid] = true;
                        }
                    }
                }
            }

            // 2. Water mains + FFE ignitions + Hamada fire arrival time `burn_time_min[p]`
            let mut pipe_open = vec![true; n_corr];
            let mut leak_lps = 0.0;
            for &(cid, len_m, mid, is_art) in &u_corridors {
                let loc_km = ((mid.x - west_x).max(0.0)) / 1000.0;
                let r_km = (cfg.medevac.surge.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
                let sa_c = (median_sa10_gmpe_g(
                    cfg.medevac.surge.ffe.seismic.moment_magnitude_mw,
                    r_km,
                    cfg.medevac.surge.ffe.seismic.vs30_m_per_s,
                )
                .ln()
                    + tau_b * eta_inter
                    + phi_w * rng.next_standard_normal())
                .exp();
                let pgv =
                    median_pgv_cm_s_from_sa10(sa_c, cfg.medevac.surge.ffe.seismic.vs30_m_per_s)
                        .clamp(1.0, 350.0);
                let mat = if cid < n_rep_corr {
                    BuriedPipeMaterialClass::SeismicErdipHdpe
                } else {
                    cfg.medevac.surge.ffe.seismic.baseline_water_pipe_material
                };
                let rr = ala_hazus_pipe_repair_rate_per_km(
                    pgv,
                    cfg.medevac.surge.ffe.seismic.pipe_fragility_law,
                    mat,
                    if is_art { 300.0 } else { 200.0 },
                    false,
                );
                let n_rep = sample_poisson_icdf(rr * (len_m / 1000.0), rng.next_f64_open01());
                if n_rep > 0 {
                    let mut brk = 0usize;
                    for _ in 0..n_rep {
                        if rng.next_f64_open01() < 0.25 {
                            brk += 1;
                        }
                    }
                    if brk > 0 {
                        pipe_open[cid] = false;
                    }
                    leak_lps += ((n_rep - brk) as f64) * 6.5 + (brk as f64) * 24.0;
                }
            }

            let sub_sa =
                (parcel_sa10.first().copied().unwrap_or(0.35).ln() + tau_b * eta_inter).exp();
            let sub_ok =
                rng.next_f64_open01() >= normal_cdf_phi((sub_sa.ln() - (0.58_f64).ln()) / 0.55);
            let pump_head_m =
                if sub_ok || cfg.medevac.surge.ffe.seismic.critical_lifeline_backup_power {
                    55.0
                } else {
                    23.0
                };
            let mut w_dist = vec![usize::MAX; n_junc];
            let mut q = VecDeque::new();
            if wtp_junc < n_junc {
                w_dist[wtp_junc] = 0;
                q.push_back(wtp_junc);
            }
            while let Some(u) = q.pop_front() {
                for &(v, cid) in &junc_adj[u] {
                    if pipe_open[cid] && w_dist[v] == usize::MAX {
                        w_dist[v] = w_dist[u] + 1;
                        q.push_back(v);
                    }
                }
            }

            let mut ignitions = Vec::new();
            for pi in 0..n_parcels {
                let pga_r = (parcel_pga[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.5);
                let ig_rate = scawthorn_hazus_ignition_rate_per_mmsqft(
                    pga_r,
                    parcels[pi].zone,
                    cfg.medevac.surge.ffe.ignition_model,
                    cfg.medevac.surge.ffe.seismic_gas_shutoff_pct,
                );
                if sample_poisson_icdf(ig_rate * parcel_tfa[pi], rng.next_f64_open01()) > 0 {
                    ignitions.push(pi);
                }
            }
            let drawdown_m =
                ((leak_lps + (ignitions.len() as f64) * 58.0) / 145.0).powf(1.852) * 5.8;
            let junc_psi: Vec<f64> = (0..n_junc)
                .map(|ji| {
                    if w_dist[ji] == usize::MAX {
                        0.0
                    } else {
                        ((pump_head_m - ((w_dist[ji] as f64) * 1.45 + drawdown_m)).max(0.0))
                            * PSI_PER_METER_H2O
                    }
                })
                .collect();

            let cistern_p =
                (cfg.medevac.surge.ffe.auxiliary_cistern_coverage_pct.clamp(0.0, 100.0)) / 100.0;
            let mut rem_eng = n_engines;
            let mut escaped_seeds = Vec::new();
            for &pi in &ignitions {
                let has_w = junc_psi[parcel_near_junc[pi]] >= MIN_FIRE_FLOW_RESIDUAL_PSI
                    || rng.next_f64_open01() < cistern_p;
                if rem_eng > 0 && has_w {
                    rem_eng -= 1;
                } else {
                    escaped_seeds.push(pi);
                }
            }

            let mut burn_time_min = vec![f64::INFINITY; n_parcels];
            let mut fheap = BinaryHeap::new();
            for &sp in &escaped_seeds {
                burn_time_min[sp] = 0.0;
                fheap.push(DijkNode { cost: 0.0, id: sp });
            }
            while let Some(DijkNode { cost, id }) = fheap.pop() {
                if cost > burn_time_min[id] {
                    continue;
                }
                for &(nxt, step_m) in &fire_spread_adj[id] {
                    let nt = cost + step_m;
                    if nt <= horizon_m && nt < burn_time_min[nxt] {
                        burn_time_min[nxt] = nt;
                        fheap.push(DijkNode { cost: nt, id: nxt });
                    }
                }
            }

            // Smoke helper at world position `pt` at `t_min`
            let u_adv = cfg.medevac.surge.ffe.wind_speed_mps.max(1.4);
            let smoke_vis_at = |pt: Vec2, t_min: f64| -> f64 {
                let mut pm = 0.0;
                for qi in 0..n_parcels {
                    if !burn_time_min[qi].is_finite() || burn_time_min[qi] > t_min {
                        continue;
                    }
                    let age = (t_min - burn_time_min[qi]).max(0.0);
                    let inten = (-age / 150.0).exp().max(0.25);
                    let dx = pt.x - parcels[qi].center.x;
                    let dy = pt.y - parcels[qi].center.y;
                    let x_down = dx * wx_hat + dy * wy_hat;
                    let y_cross = (-dx * wy_hat + dy * wx_hat).abs();
                    if x_down >= -45.0 {
                        let sig_y = 55.0 + 0.24 * x_down.max(25.0);
                        pm += (parcel_tfa[qi] * 46_000.0 * inten / (u_adv * sig_y))
                            * (-0.5 * (y_cross / sig_y).powi(2)).exp();
                    }
                }
                koschmieder_smoke_visibility_and_ceiling(
                    cfg.medevac.ambient_visibility_sm,
                    cfg.medevac.ambient_ceiling_ft,
                    pm,
                )
                .0
            };

            // 3. Assign Staged Neighborhood Evacuation Tiers (`Stage 1`, `Stage 2`, `Stage 3`)
            let mut parcel_stage = vec![1u8; n_parcels];
            let mut parcel_rel_min = vec![0.0; n_parcels];
            let mut parcel_vis_60 = vec![cfg.medevac.ambient_visibility_sm; n_parcels];
            for pi in 0..n_parcels {
                let vis60 = smoke_vis_at(parcels[pi].center, 60.0);
                parcel_vis_60[pi] = vis60;
                let bt = burn_time_min[pi];
                let high_vuln = matches!(
                    parcels[pi].zone,
                    ZoneCode::R3 | ZoneCode::MU4 | ZoneCode::CBD
                );
                let stg = match cfg.staging_strategy {
                    EvacuationStagingStrategy::SimultaneousUnstaged => 1,
                    EvacuationStagingStrategy::FireProximityThreeStage => {
                        let s2_min_bt = cfg.stage2_release_delay_min.clamp(5.0, 90.0) + 110.0;
                        let s3_min_bt = cfg.stage3_release_delay_min.clamp(10.0, 140.0) + 125.0;
                        if bt <= s2_min_bt || vis60 < 1.35 {
                            1
                        } else if bt <= s3_min_bt || vis60 < 2.6 {
                            2
                        } else {
                            3
                        }
                    }
                    EvacuationStagingStrategy::VulnerabilityAndFireWeighted => {
                        let s2_min_bt = cfg.stage2_release_delay_min.clamp(5.0, 90.0) + 115.0;
                        let s3_min_bt = cfg.stage3_release_delay_min.clamp(10.0, 140.0) + 135.0;
                        if bt <= s2_min_bt || (high_vuln && (bt <= s3_min_bt || vis60 < 2.2)) {
                            1
                        } else if bt <= s3_min_bt || high_vuln || vis60 < 3.2 {
                            2
                        } else {
                            3
                        }
                    }
                };
                parcel_stage[pi] = stg;
                parcel_rel_min[pi] = match stg {
                    1 => 0.0,
                    2 => cfg.stage2_release_delay_min.clamp(5.0, 90.0),
                    _ => cfg.stage3_release_delay_min.clamp(10.0, 140.0),
                };
            }

            // 4. Simulate 18 x 10-min waves of coupled smoke speed drop, thermal closures, contraflow routing & BPR queues
            let mut seg_queue_vehs = vec![0.0; n_segs];
            let mut seg_peak_vc = vec![0.0f64; n_segs];
            let mut seg_therm_ever = vec![false; n_segs];
            let mut seg_smoke_v_acc = vec![0.0; n_segs];

            let mut wave_evac_pct = vec![0.0; n_waves];
            let mut wave_queue_vehs = vec![0.0; n_waves];
            let mut wave_therm_closed = vec![0.0; n_waves];

            let mut parcel_cleared_vehs = vec![0.0; n_parcels];
            let mut parcel_clear_min = vec![180.0; n_parcels];
            let mut parcel_done = vec![false; n_parcels];
            let mut cumul_cleared_vehs = 0.0;
            let mut peak_vc_global = 0.0f64;

            for wi in 0..n_waves {
                let t_now = (wi as f64 + 0.5) * dt_min;

                // Evaluate dynamic thermal road closures & Wetterberg/Intini smoke speed factors
                let mut seg_open = vec![true; n_segs];
                let mut seg_trav_s = vec![f64::INFINITY; n_segs];
                let mut seg_cap_w = vec![1.0; n_segs];
                let mut therm_cnt = 0.0;

                for sid in 0..n_segs {
                    if net.segments[sid].closed || br_closed[sid] || eff_evac_cap_10m[sid] < 1.0 {
                        seg_open[sid] = false;
                        continue;
                    }
                    let thermally_overrun = seg_adj_parcels[sid]
                        .iter()
                        .any(|&pi| burn_time_min[pi] <= t_now);
                    if thermally_overrun {
                        seg_open[sid] = false;
                        seg_therm_ever[sid] = true;
                        therm_cnt += 1.0;
                        continue;
                    }
                    let vis_sm = smoke_vis_at(seg_mids[sid], t_now);
                    let (v_fac, q_fac) = smoke_visibility_speed_and_capacity_factors(vis_sm);
                    seg_smoke_v_acc[sid] += v_fac;
                    let eff_v = (net.segments[sid].speed_mps * v_fac).clamp(2.2, 31.0);
                    let cap = (eff_evac_cap_10m[sid] * q_fac).max(25.0);
                    let q_vc = (seg_queue_vehs[sid] / cap).clamp(0.0, 3.5);
                    let bpr_mult = 1.0 + 0.34 * q_vc.powi(3);
                    seg_trav_s[sid] = (net.segments[sid].length / eff_v) * bpr_mult;
                    seg_cap_w[sid] = cap;
                }
                wave_therm_closed[wi] = therm_cnt;

                // Reverse Dijkstra from open `shelter_segs` to find dynamic outbound evacuation tree `next_out_seg[sid]`
                let mut dist_to_shelter_s = vec![f64::INFINITY; n_segs];
                let mut next_out_seg: Vec<Option<usize>> = vec![None; n_segs];
                let mut heap = BinaryHeap::new();
                let shelter_set: HashSet<usize> = shelter_segs.iter().copied().collect();

                for &sh_sid in &shelter_segs {
                    if seg_open[sh_sid] {
                        dist_to_shelter_s[sh_sid] = seg_trav_s[sh_sid];
                        heap.push(DijkNode {
                            cost: seg_trav_s[sh_sid],
                            id: sh_sid,
                        });
                    }
                }
                while let Some(DijkNode { cost, id }) = heap.pop() {
                    if cost > dist_to_shelter_s[id] {
                        continue;
                    }
                    for &(prev_sid, turn_pen_s) in &in_neighbors[id] {
                        if !seg_open[prev_sid] {
                            continue;
                        }
                        let nc = cost + turn_pen_s + seg_trav_s[prev_sid];
                        if nc < dist_to_shelter_s[prev_sid] {
                            dist_to_shelter_s[prev_sid] = nc;
                            next_out_seg[prev_sid] = Some(id);
                            heap.push(DijkNode {
                                cost: nc,
                                id: prev_sid,
                            });
                        }
                    }
                }

                // Mobilize vehicles per parcel using Rayleigh/S-curve departure (`tau = 16 min` post-stage-release)
                let mut seg_demand = vec![0.0; n_segs];
                for pi in 0..n_parcels {
                    let rel = parcel_rel_min[pi];
                    let t0_w = (wi as f64) * dt_min;
                    let t1_w = t0_w + dt_min;
                    if t1_w <= rel {
                        continue;
                    }
                    let cdf0 = if t0_w <= rel {
                        0.0
                    } else {
                        1.0 - (-((t0_w - rel) / 18.0).powi(2)).exp()
                    };
                    let cdf1 = 1.0 - (-((t1_w - rel) / 18.0).powi(2)).exp();
                    let dep_vehs = parcel_vehs[pi] * (cdf1 - cdf0).max(0.0);
                    if dep_vehs <= 1e-6 {
                        continue;
                    }

                    // Propagate departing vehicles up to 8 hops along their outbound shortest-path chain
                    let mut cur = parcel_near_seg[pi];
                    let mut hops = 0usize;
                    while hops < 8 {
                        seg_demand[cur] += dep_vehs;
                        if shelter_set.contains(&cur) {
                            break;
                        }
                        if let Some(nxt) = next_out_seg[cur] {
                            cur = nxt;
                            hops += 1;
                        } else {
                            break;
                        }
                    }
                }

                // Update segment BPR V/C & residual queues, and compute network clearance throughput
                let mut max_wave_vc = 0.0f64;
                let mut tot_q_rem = 0.0;
                for sid in 0..n_segs {
                    if !seg_open[sid] {
                        continue;
                    }
                    let tot_load = seg_queue_vehs[sid] + seg_demand[sid];
                    let vc = tot_load / seg_cap_w[sid].max(1.0);
                    if vc > seg_peak_vc[sid] {
                        seg_peak_vc[sid] = vc;
                    }
                    if vc > max_wave_vc {
                        max_wave_vc = vc;
                    }
                    if vc > peak_vc_global {
                        peak_vc_global = vc;
                    }
                    let discharged = tot_load.min(seg_cap_w[sid]);
                    seg_queue_vehs[sid] = (tot_load - discharged).max(0.0);
                    tot_q_rem += seg_queue_vehs[sid];
                }

                // Update parcel-level effective clearance progress accounting for downstream bottleneck delay
                let net_cong_delay_min = if max_wave_vc > 0.55 {
                    ((max_wave_vc - 0.55).powf(1.35)) * 18.0
                } else {
                    0.0
                };
                cumul_cleared_vehs = 0.0;
                for pi in 0..n_parcels {
                    let rel = parcel_rel_min[pi];
                    let sid = parcel_near_seg[pi];
                    let route_min = if dist_to_shelter_s[sid].is_finite() {
                        dist_to_shelter_s[sid] / 60.0
                    } else {
                        38.0 // surface detour when primary road thermally severed
                    };
                    let eff_elapsed = ((wi as f64 + 1.0) * dt_min - rel - route_min - net_cong_delay_min)
                        .max(0.0);
                    let cleared_frac = (1.0 - (-(eff_elapsed / 18.0).powi(2)).exp()).clamp(0.0, 1.0);
                    parcel_cleared_vehs[pi] = parcel_vehs[pi] * cleared_frac;
                    cumul_cleared_vehs += parcel_cleared_vehs[pi];
                    if !parcel_done[pi] && cleared_frac >= 0.90 {
                        parcel_done[pi] = true;
                        parcel_clear_min[pi] = (wi as f64 + 1.0) * dt_min;
                    }
                }

                wave_evac_pct[wi] = ((cumul_cleared_vehs / total_evac_vehs.max(1.0)) * 100.0)
                    .clamp(0.0, 100.0);
                wave_queue_vehs[wi] = tot_q_rem;
            }

            // Interpolate T50, T90, T95 network clearance times
            let interp_t = |target_pct: f64| -> f64 {
                for (wi, &pct) in wave_evac_pct.iter().enumerate() {
                    if pct >= target_pct {
                        let prev_pct = if wi == 0 { 0.0 } else { wave_evac_pct[wi - 1] };
                        let prev_t = (wi as f64) * dt_min;
                        let frac = ((target_pct - prev_pct) / (pct - prev_pct).max(1e-6))
                            .clamp(0.0, 1.0);
                        return prev_t + frac * dt_min;
                    }
                }
                180.0 + (target_pct - wave_evac_pct.last().copied().unwrap_or(80.0)).max(0.0) * 2.2
            };
            let t50_min = interp_t(50.0);
            let t90_min = interp_t(90.0);
            let t95_min = interp_t(95.0);

            // Evaluate Fire-Overrun Threatened Residents (`parcel_clear_min[pi] > burn_time_min[pi]`)
            let mut overrun_res = 0.0;
            let mut parcel_overrun = vec![0.0; n_parcels];
            for pi in 0..n_parcels {
                if !parcel_done[pi] {
                    parcel_clear_min[pi] = t95_min.max(180.0);
                }
                if burn_time_min[pi].is_finite() {
                    // Probability of fire front overtaking tail evacuees before 90% block clearance
                    let margin = parcel_clear_min[pi] - burn_time_min[pi];
                    if margin > -12.0 {
                        let p_ovr = 1.0 / (1.0 + (-margin / 9.0).exp());
                        parcel_overrun[pi] = p_ovr * 100.0;
                        overrun_res += parcel_pop[pi] * 0.35 * p_ovr;
                    }
                }
            }

            // Evaluate Inbound EMS / Fire / LZ Responder Travel Time under active contraflow (`rem_inbound`)
            let mut in_ems_s = vec![f64::INFINITY; n_segs];
            let mut eheap = BinaryHeap::new();
            for &fs_sid in &fire_station_segs {
                if rem_inbound[fs_sid] >= 1 && !br_closed[fs_sid] {
                    let s0 = 45.0 + net.segments[fs_sid].length / (net.segments[fs_sid].speed_mps * evp_boost).max(4.0);
                    if s0 < in_ems_s[fs_sid] {
                        in_ems_s[fs_sid] = s0;
                        eheap.push(DijkNode { cost: s0, id: fs_sid });
                    }
                }
            }
            while let Some(DijkNode { cost, id }) = eheap.pop() {
                if cost > in_ems_s[id] {
                    continue;
                }
                for &(nxt, t_pen) in &out_neighbors[id] {
                    if rem_inbound[nxt] >= 1 && !br_closed[nxt] && !seg_therm_ever[nxt] {
                        let step = t_pen
                            + net.segments[nxt].length
                                / (net.segments[nxt].speed_mps * evp_boost).max(4.0);
                        if cost + step < in_ems_s[nxt] {
                            in_ems_s[nxt] = cost + step;
                            eheap.push(DijkNode {
                                cost: cost + step,
                                id: nxt,
                            });
                        }
                    }
                }
            }
            let mean_ems_in_min = {
                let mut sum_m = 0.0;
                for pi in 0..n_parcels {
                    let sid = parcel_near_seg[pi];
                    let m = if in_ems_s[sid].is_finite() {
                        in_ems_s[sid] / 60.0 + 1.2
                    } else {
                        // Inbound lane reversed (`FullUnprotectedOneWayOut`) -> responder blocked / forced onto shoulder crawl!
                        24.5
                    };
                    sum_m += m;
                }
                sum_m / (n_parcels.max(1) as f64)
            };

            let nw_f = n_waves as f64;
            let seg_smoke_v_fac: Vec<f64> = seg_smoke_v_acc
                .into_iter()
                .map(|acc| if acc > 0.0 { (acc / nw_f).clamp(0.22, 1.0) } else { 0.85 })
                .collect();
            let mean_smoke_v_fac =
                seg_smoke_v_fac.iter().sum::<f64>() / (n_segs.max(1) as f64);
            let therm_closed_cnt = seg_therm_ever.iter().filter(|&&b| b).count() as f64;

            EvacTrialOut {
                t50_min,
                t90_min,
                t95_min,
                peak_vc: peak_vc_global,
                overrun_res,
                therm_closed_cnt,
                mean_smoke_v_fac,
                mean_ems_in_min,
                seg_therm_closed: seg_therm_ever
                    .into_iter()
                    .map(|b| if b { 100.0 } else { 0.0 })
                    .collect(),
                seg_smoke_v_fac,
                seg_peak_vc,
                parcel_stage,
                parcel_rel_min,
                parcel_vis_sm: parcel_vis_60,
                parcel_clear_min,
                parcel_overrun,
                wave_evac_pct,
                wave_queue_vehs,
                wave_therm_closed,
            }
        })
        .collect();

    let tf = trials as f64;
    let cf_seg_cnt = cf_added.iter().filter(|&&c| c > 0).count();
    let cf_lane_km: f64 = (0..n_segs)
        .map(|sid| (cf_added[sid] as f64) * (net.segments[sid].length / 1000.0))
        .sum();

    let mut corridor_rows = Vec::with_capacity(n_segs);
    for (sid, s) in net.segments.iter().enumerate() {
        corridor_rows.push(ContraflowCorridorSegment {
            segment_id: sid,
            twin_inbound_seg_id: twin_seg[sid],
            road_name: s.name.clone(),
            from_pos: net.junctions[s.from_junc].pos,
            to_pos: net.junctions[s.to_junc].pos,
            base_lanes: s.lane_count,
            contraflow_added_lanes: cf_added[sid],
            remaining_inbound_lanes: rem_inbound[sid],
            protected_ems_lz_corridor: protected_ems_lz[sid],
            thermal_closure_prob_pct: trial_outs
                .iter()
                .map(|t| t.seg_therm_closed[sid])
                .sum::<f64>()
                / tf,
            mean_smoke_speed_factor: trial_outs
                .iter()
                .map(|t| t.seg_smoke_v_fac[sid])
                .sum::<f64>()
                / tf,
            peak_vc_ratio: trial_outs.iter().map(|t| t.seg_peak_vc[sid]).sum::<f64>() / tf,
        });
    }

    let mut parcel_rows = Vec::with_capacity(n_parcels);
    for (pi, p) in parcels.iter().enumerate() {
        let mut stg_counts = [0usize; 4];
        for t in &trial_outs {
            let st = (t.parcel_stage[pi] as usize).clamp(1, 3);
            stg_counts[st] += 1;
        }
        let modal_stage = (1..=3)
            .max_by_key(|&st| stg_counts[st])
            .unwrap_or(1) as u8;

        parcel_rows.push(ParcelEvacuationStageRow {
            parcel_id: pi,
            zone_code: p.zone,
            center: p.center,
            evac_stage: modal_stage,
            release_delay_min: trial_outs.iter().map(|t| t.parcel_rel_min[pi]).sum::<f64>() / tf,
            evac_vehicles: parcel_vehs[pi],
            mean_smoke_vis_sm: trial_outs.iter().map(|t| t.parcel_vis_sm[pi]).sum::<f64>() / tf,
            mean_clearance_min: trial_outs
                .iter()
                .map(|t| t.parcel_clear_min[pi])
                .sum::<f64>()
                / tf,
            fire_overrun_prob_pct: trial_outs.iter().map(|t| t.parcel_overrun[pi]).sum::<f64>()
                / tf,
        });
    }

    let mut timeline = Vec::with_capacity(n_waves);
    for wi in 0..n_waves {
        timeline.push(EvacuationWavePoint {
            minute: (wi as f64 + 1.0) * dt_min,
            evacuated_pct: trial_outs.iter().map(|t| t.wave_evac_pct[wi]).sum::<f64>() / tf,
            active_network_queue_vehs: trial_outs
                .iter()
                .map(|t| t.wave_queue_vehs[wi])
                .sum::<f64>()
                / tf,
            thermally_closed_segments: trial_outs
                .iter()
                .map(|t| t.wave_therm_closed[wi])
                .sum::<f64>()
                / tf,
        });
    }

    EvacuationContraflowReport {
        config: cfg.clone(),
        total_evacuees,
        total_evac_vehicles: total_evac_vehs,
        contraflow_reversed_segments_count: cf_seg_cnt,
        contraflow_lane_km_added: cf_lane_km,
        protected_ems_lz_segments_count: ems_prot_indices.len(),
        expected_thermally_closed_segments: trial_outs
            .iter()
            .map(|t| t.therm_closed_cnt)
            .sum::<f64>()
            / tf,
        mean_road_smoke_speed_factor: trial_outs
            .iter()
            .map(|t| t.mean_smoke_v_fac)
            .sum::<f64>()
            / tf,
        t50_clearance_min: trial_outs.iter().map(|t| t.t50_min).sum::<f64>() / tf,
        t90_clearance_min: trial_outs.iter().map(|t| t.t90_min).sum::<f64>() / tf,
        t95_clearance_min: trial_outs.iter().map(|t| t.t95_min).sum::<f64>() / tf,
        peak_bottleneck_vc_ratio: trial_outs.iter().map(|t| t.peak_vc).sum::<f64>() / tf,
        expected_fire_overrun_residents: trial_outs.iter().map(|t| t.overrun_res).sum::<f64>() / tf,
        ems_lz_corridor_access_preserved_pct: ems_preserved_pct,
        mean_ems_inbound_response_min: trial_outs
            .iter()
            .map(|t| t.mean_ems_in_min)
            .sum::<f64>()
            / tf,
        timeline,
        corridor_rows,
        parcel_rows,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

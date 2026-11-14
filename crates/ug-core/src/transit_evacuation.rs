//! Coupled GTFS Transit-Bus & ADA Paratransit Evacuation Fleets, Neighborhood Pickup Depots (`PP`),
//! and Dedicated Bus-Plus-EMS Priority Contraflow Lanes into Clean-Air (`MERV-13 + HEPA`) Shelters
//!
//! Grounded in:
//! - Transportation Research Board (TRB) Special Report 294, *The Role of Transit in Emergency
//!   Evacuation*, National Academies Press (`https://onlinepubs.trb.org/onlinepubs/sr/sr294.pdf`).
//! - US DOT Federal Highway Administration (FHWA), *Evacuating Populations with Special Needs —
//!   Routes to Effective Evacuation Planning Primer Series*
//!   (`https://ops.fhwa.dot.gov/publications/fhwahop09022/sn3_planning.htm`;
//!   `https://rosap.ntl.bts.gov/view/dot/16633/dot_16633_DS1.pdf`) & Federal Transit Administration
//!   (FTA), *Coordinated Transit Response Planning and Operations Support Tools for Mitigating Impacts
//!   of All-Hazard Emergency Events* (`https://digitalcommons.mtu.edu/context/michigantech-p/article/36234/viewcontent/dot_64115_DS1.pdf`).
//! - Zhao, L., Chien, S., Meegoda, J., Luo, Z., & Liu, X. (2015), *Bus Contraflow Lane: Improved
//!   Contraflow Approach in Freeway Evacuation*
//!   (`https://www.researchgate.net/publication/272717305_Bus_Contraflow_Lane_Improved_Contraflow_Approach_in_Freeway_Evacuation`).
//! - Bish, D. R. (2011) & Dhamala, T. N., Pyakurel, U., & Dempe, S. (2018), *A Critical Survey on
//!   Transit-Based Evacuation Planning Models (`TBEP`)*, *International Journal of Operations Research*
//!   15(1):29–47 (`http://www.orstw.org.tw/ijor/vol15no1/IJOR2018_vol15_no1_p29_p47.pdf`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind};
use crate::emergency_shelters::{
    compute_indoor_shelter_pm25_ug_m3, enumerate_candidate_shelters, EmergencySheltersConfig,
    ShelterFacilityArchetype,
};
use crate::evacuation_contraflow::smoke_visibility_speed_and_capacity_factors;
use crate::ffe_conflagration::{
    hamada_hazus_spread_rates, scawthorn_hazus_ignition_rate_per_mmsqft,
};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi,
};
use crate::transit_gtfs::{BoardingFareMode, BusCoachSpec, TransitGtfsSystem};
use crate::utilities::{MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const TRANSIT_EVAC_DEPOT_CSV_HEADER: &str = "depot_id,gtfs_stop_id,name,segment_id,serving_gtfs_routes,mean_walk_m,carless_demand_pax,evacuated_clean_air_pax,stranded_pax,mean_shuttle_cycle_min,mean_wait_min,ffe_closure_prob_pct";
pub const TRANSIT_EVAC_WAVE_CSV_HEADER: &str = "minute,cumul_carless_evacuated_pct,cumul_ada_evacuated_pct,cumul_combined_clean_air_pct,active_bus_trips_completed,active_ada_van_trips_completed";

const SQFT_PER_M2: f64 = 10.763_910_4;

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

/// Priority lane & contraflow regime for evacuation transit coaches, ADA paratransit vans & EMS units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusContraflowLaneMode {
    /// Transit buses and ADA lift vans sit inside general private-auto BPR queues (`1 + 0.34*(V/C)^3` gridlock).
    MixedAutoQueueNoPriority,
    /// Uses only pre-existing curbside `has_bus_lane` segments without opposing-lane contraflow reversal.
    CurbsideBusOnlyNoContraflow,
    /// Zhao et al. (2015) / TRB SR 294 Dedicated Bus + EMS Priority Contraflow Lane: reverses `1` opposing
    /// arterial lane for exclusive Bus + ADA + EMS shuttle flow while keeping bidirectional EMS hospital/LZ access!
    DedicatedBusPlusEmsContraflow,
}

impl Default for BusContraflowLaneMode {
    fn default() -> Self {
        Self::DedicatedBusPlusEmsContraflow
    }
}

impl BusContraflowLaneMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MixedAutoQueueNoPriority => {
                "Mixed Auto Queue (No Transit Priority — Severe BPR Delay)"
            }
            Self::CurbsideBusOnlyNoContraflow => {
                "Curbside Bus-Only Lanes (No Contraflow Reversal)"
            }
            Self::DedicatedBusPlusEmsContraflow => {
                "Dedicated Bus + EMS Priority Contraflow Lane (Zhao / TRB SR 294)"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::MixedAutoQueueNoPriority => "Mixed-Queue",
            Self::CurbsideBusOnlyNoContraflow => "Curb-BusOnly",
            Self::DedicatedBusPlusEmsContraflow => "Bus+EMS-Contraflow",
        }
    }
}

/// Transit-Based Evacuation Problem (`TBEP`) fleet scheduling & clean-air shelter routing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitDispatchStrategy {
    /// Buses stay on static fixed-route GTFS loops and unload at geographically nearest nominal shelters.
    FixedGtfsHeadwayLoop,
    /// Allocates bus trips proportional to pickup-depot headcount without FFE fire urgency or strict HEPA filtering.
    DemandProportionalShuttle,
    /// Bish (2011) Rolling-Horizon `TBEP` + ADA Curb-to-Clean-Air-Shelter Priority: dispatches coaches &
    /// lift vans by `FFE` fire/smoke urgency exclusively into ATC-20 green-tagged `MERV-13 + HEPA` shelters!
    FireUrgencyCleanAirTbep,
}

impl Default for TransitDispatchStrategy {
    fn default() -> Self {
        Self::FireUrgencyCleanAirTbep
    }
}

impl TransitDispatchStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::FixedGtfsHeadwayLoop => "Static GTFS Fixed-Route Loop (Smoke-Blind Unload)",
            Self::DemandProportionalShuttle => "Demand-Proportional Depot Shuttle (Smoke-Blind)",
            Self::FireUrgencyCleanAirTbep => {
                "Dynamic Bish TBEP + ADA Paratransit -> Clean-Air HEPA Shelters"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::FixedGtfsHeadwayLoop => "GTFS-Static",
            Self::DemandProportionalShuttle => "Demand-Prop",
            Self::FireUrgencyCleanAirTbep => "Urgency-CleanAir-TBEP",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitEvacuationConfig {
    /// Coupled clean-air shelters (`MERV-13 + HEPA`), evacuation contraflow, FFE & seismic MC config.
    pub shelters: EmergencySheltersConfig,
    pub lane_mode: BusContraflowLaneMode,
    pub dispatch_strategy: TransitDispatchStrategy,
    pub coach_spec: BusCoachSpec,
    pub fare_mode: BoardingFareMode,
    /// Active fixed-route GTFS transit buses mobilized for neighborhood pickup shuttle waves (`2..=36`).
    pub transit_bus_fleet: usize,
    /// Active wheelchair-lift ADA paratransit vans mobilized for curb-to-clean-air-shelter runs (`1..=24`).
    pub ada_paratransit_vans: usize,
    /// Maximum Neighborhood Pickup Depots (`[GTFS-PP]`) activated from GTFS stops (`4..=16`, default `8`).
    pub max_pickup_depots: usize,
    /// Maximum recommended pedestrian walk radius (`m`) for carless ambulatory evacuees (TRB SR 294 = `400 m`).
    pub walk_catchment_radius_m: f64,
    /// Maximum evacuation clearance window (`minutes`) before FFE conflagration & dense smoke cutoff (default `120 min`).
    pub clearance_window_min: f64,
    /// Citywide carless household multiplier (`0.5..=2.0`, `1.0` = empirical urban zoning baseline).
    pub carless_demand_multiplier: f64,
}

impl Default for TransitEvacuationConfig {
    fn default() -> Self {
        Self {
            shelters: EmergencySheltersConfig {
                shelter_backup_power_water_sanitation: true,
                ..EmergencySheltersConfig::default()
            },
            lane_mode: BusContraflowLaneMode::DedicatedBusPlusEmsContraflow,
            dispatch_strategy: TransitDispatchStrategy::FireUrgencyCleanAirTbep,
            coach_spec: BusCoachSpec::Articulated60FtBrt,
            fare_mode: BoardingFareMode::AllDoorFareFreeBrt,
            transit_bus_fleet: 10,
            ada_paratransit_vans: 6,
            max_pickup_depots: 8,
            walk_catchment_radius_m: 400.0,
            clearance_window_min: 120.0,
            carless_demand_multiplier: 1.0,
        }
    }
}

/// Computes evacuation bus boarding + shelter alighting dwell (`minutes`) per coach trip under
/// TCQSM 3rd Ed. fare-collection physics plus emergency carry-on baggage/pet factor (`1.30x`),
/// and evacuation coach capacity (`pax/trip` accounting for carry-on bags: `50` on `40-ft`, `80` on `60-ft`).
#[inline]
pub fn compute_evac_coach_capacity_and_dwell_min(
    coach_spec: BusCoachSpec,
    fare_mode: BoardingFareMode,
    boarding_pax: f64,
) -> (usize, f64) {
    let evac_cap = match coach_spec {
        BusCoachSpec::Standard40Ft => 50usize,
        BusCoachSpec::Articulated60FtBrt => 80usize,
    };
    let pax = boarding_pax.clamp(1.0, evac_cap as f64);
    let (clearance_s, board_s_per_pax, alight_s_per_pax) = match fare_mode {
        BoardingFareMode::CashBoxFrontDoor => (4.5, 4.15 * 1.30, 2.20),
        BoardingFareMode::SmartCardFrontDoor => (3.8, 2.75 * 1.25, 1.95),
        BoardingFareMode::AllDoorFareFreeBrt => (2.8, 1.75 * 1.22, 1.35),
    };
    let total_dwell_s = 2.0 * clearance_s + pax * (board_s_per_pax + alight_s_per_pax);
    (evac_cap, (total_dwell_s / 60.0).clamp(0.4, 18.0))
}

/// Computes ADA paratransit van capacity (`mixed wheelchair + companion/elderly pax`) and hydraulic
/// wheelchair-lift + 4-point Q'Straint tie-down boarding/alighting dwell (`minutes`).
#[inline]
pub fn compute_ada_van_capacity_and_dwell_min(boarding_ada_pax: f64) -> (usize, f64) {
    let van_cap = 8usize; // 4 wheelchair tie-down positions + 4 frail-elderly/caregiver seats
    let pax = boarding_ada_pax.clamp(1.0, van_cap as f64);
    // ~50% require hydraulic ramp/lift deployment + 4-point securement (~3.4 min), ~50% assisted ambulatory (~1.4 min)
    let dwell_min = 1.2 + pax * 2.35;
    (van_cap, dwell_min.clamp(2.0, 24.0))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PickupDepotTelemetryRow {
    pub depot_id: usize,
    pub gtfs_stop_id: String,
    pub name: String,
    pub segment_id: usize,
    pub world_pos: Vec2,
    pub serving_gtfs_routes: usize,
    pub mean_walk_dist_m: f64,
    pub carless_demand_pax: f64,
    pub evacuated_clean_air_pax: f64,
    pub evacuated_any_shelter_pax: f64,
    pub stranded_pax: f64,
    pub mean_shuttle_cycle_min: f64,
    pub mean_wait_min: f64,
    pub ffe_closure_prob_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusShuttleTourRow {
    pub vehicle_label: String,
    pub is_ada_paratransit: bool,
    pub origin_depot_or_zone: String,
    pub dest_shelter_id: usize,
    pub dest_shelter_name: String,
    pub origin_pos: Vec2,
    pub shelter_pos: Vec2,
    pub trips_completed: f64,
    pub cycle_time_min: f64,
    pub total_pax_delivered: f64,
    pub shelter_clean_air_ok: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransitEvacWavePoint {
    pub minute: f64,
    pub cumul_carless_evacuated_pct: f64,
    pub cumul_ada_evacuated_pct: f64,
    pub cumul_combined_clean_air_pct: f64,
    pub active_bus_trips_completed: f64,
    pub active_ada_van_trips_completed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitContraflowSegmentRow {
    pub segment_id: usize,
    pub road_name: String,
    pub base_lanes: usize,
    pub bus_ems_priority_lane: bool,
    pub contraflow_added_lanes: usize,
    pub remaining_auto_lanes: usize,
    pub ems_hospital_spine: bool,
    pub ffe_thermal_closure_prob_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitEvacuationReport {
    pub config: TransitEvacuationConfig,
    pub active_pickup_depots_count: usize,
    pub open_clean_air_shelters_count: usize,
    pub bus_ems_contraflow_segments_count: usize,
    pub total_carless_demand_pax: f64,
    pub total_ada_mobility_limited_pax: f64,
    pub evacuated_carless_pct: f64,
    pub evacuated_ada_pct: f64,
    pub combined_clean_air_sheltered_pct: f64,
    pub stranded_vulnerable_pax: f64,
    pub p90_stranded_vulnerable_pax: f64,
    pub mean_bus_shuttle_cycle_min: f64,
    pub mean_trips_per_bus: f64,
    pub mean_trips_per_ada_van: f64,
    pub mean_carless_wait_min: f64,
    pub mean_ada_curb_wait_min: f64,
    pub ems_inbound_access_preserved_pct: f64,
    pub depot_rows: Vec<PickupDepotTelemetryRow>,
    pub representative_shuttle_tours: Vec<BusShuttleTourRow>,
    pub wave_timeline: Vec<TransitEvacWavePoint>,
    pub contraflow_segments: Vec<TransitContraflowSegmentRow>,
    pub parcel_cleared_pct: Vec<f64>,
    pub parcel_wait_min: Vec<f64>,
    pub solve_time_ms: f64,
}

impl TransitEvacuationReport {
    pub fn to_depot_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.depot_rows.len() * 140);
        out.push_str(TRANSIT_EVAC_DEPOT_CSV_HEADER);
        out.push('\n');
        for d in &self.depot_rows {
            out.push_str(&format!(
                "{},{},\"{}\",{},{},{:.1},{:.1},{:.1},{:.1},{:.2},{:.2},{:.2}\n",
                d.depot_id,
                d.gtfs_stop_id,
                d.name.replace('"', "'"),
                d.segment_id,
                d.serving_gtfs_routes,
                d.mean_walk_dist_m,
                d.carless_demand_pax,
                d.evacuated_clean_air_pax,
                d.stranded_pax,
                d.mean_shuttle_cycle_min,
                d.mean_wait_min,
                d.ffe_closure_prob_pct,
            ));
        }
        out
    }

    pub fn to_wave_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.wave_timeline.len() * 96);
        out.push_str(TRANSIT_EVAC_WAVE_CSV_HEADER);
        out.push('\n');
        for w in &self.wave_timeline {
            out.push_str(&format!(
                "{:.1},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                w.minute,
                w.cumul_carless_evacuated_pct,
                w.cumul_ada_evacuated_pct,
                w.cumul_combined_clean_air_pct,
                w.active_bus_trips_completed,
                w.active_ada_van_trips_completed,
            ));
        }
        out
    }

    pub fn export_depot_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_depot_csv_string())?;
        Ok(self.depot_rows.len())
    }
}

#[derive(Clone)]
struct CandidatePickupDepot {
    depot_id: usize,
    gtfs_stop_id: String,
    name: String,
    segment_id: usize,
    world_pos: Vec2,
    serving_gtfs_routes: usize,
}

/// Extracts & spatially selects up to `max_pickup_depots` Neighborhood Pickup Depots (`[GTFS-PP]`)
/// directly from `TransitGtfsSystem` stops (or synthesizes arterial stops if empty), maximizing
/// `400 m` walkable coverage of carless urban households.
fn select_gtfs_pickup_depots(
    net: &RoadNetwork,
    parcels: &[Parcel],
    gtfs: &TransitGtfsSystem,
    cfg: &TransitEvacuationConfig,
) -> Vec<CandidatePickupDepot> {
    let max_pp = cfg.max_pickup_depots.clamp(2, 16);
    let mut pool: Vec<(String, String, usize, Vec2, usize)> = Vec::new();

    if !gtfs.stops.is_empty() {
        for st in &gtfs.stops {
            if st.segment_id < net.segments.len() && !net.segments[st.segment_id].closed {
                let r_cnt = gtfs
                    .routes
                    .iter()
                    .filter(|r| r.stop_ids.contains(&st.id))
                    .count()
                    .max(1);
                pool.push((
                    st.gtfs_stop_id.clone(),
                    st.name.clone(),
                    st.segment_id,
                    st.world_pos,
                    r_cnt,
                ));
            }
        }
    }

    if pool.len() < 4 {
        let synth = TransitGtfsSystem::synthesize_for_network(net, parcels, 10.0);
        for st in &synth.stops {
            if st.segment_id < net.segments.len()
                && !net.segments[st.segment_id].closed
                && pool.iter().all(|p| p.2 != st.segment_id)
            {
                let r_cnt = synth
                    .routes
                    .iter()
                    .filter(|r| r.stop_ids.contains(&st.id))
                    .count()
                    .max(1);
                pool.push((
                    st.gtfs_stop_id.clone(),
                    st.name.clone(),
                    st.segment_id,
                    st.world_pos,
                    r_cnt,
                ));
            }
        }
    }

    if pool.is_empty() {
        for (idx, seg) in net
            .segments
            .iter()
            .filter(|s| !s.closed && !s.base_pts.is_empty())
            .take(max_pp)
            .enumerate()
        {
            let mid = seg.base_pts[seg.base_pts.len() / 2];
            pool.push((
                format!("PP-{:02}", idx + 1),
                format!("{} Evac Stop", seg.name),
                seg.id,
                mid,
                1,
            ));
        }
    }

    // Compute baseline carless parcel weights to greedily cover walkable catchments (`p-median` / `MCLP`)
    let parcel_weights: Vec<(Vec2, f64)> = parcels
        .iter()
        .map(|p| {
            let (carless_sh, _) = zone_carless_and_ada_shares(p.zone);
            let hh_pop = ((p.trip_weight * 1.45).max(3.5)) * carless_sh;
            (p.center, hh_pop)
        })
        .collect();

    let walk_r = cfg.walk_catchment_radius_m.clamp(150.0, 1200.0);
    let mut chosen_indices: Vec<usize> = Vec::with_capacity(max_pp.min(pool.len()));
    let mut covered_parcel = vec![false; parcel_weights.len()];

    while chosen_indices.len() < max_pp && chosen_indices.len() < pool.len() {
        let mut best_idx = None;
        let mut best_gain = -1.0f64;

        for (cand_i, cand) in pool.iter().enumerate() {
            if chosen_indices.contains(&cand_i) {
                continue;
            }
            // Enforce minimum 140 m spacing between neighborhood pickup depots when alternatives exist
            let min_sep = chosen_indices
                .iter()
                .map(|&j| pool[j].3.dist(cand.3))
                .fold(f64::INFINITY, f64::min);
            let sep_pen = if min_sep < 140.0 { 0.25 } else { 1.0 };

            let mut gain = (cand.4 as f64) * 1.5;
            for (pi, &(ppos, w)) in parcel_weights.iter().enumerate() {
                let d = ppos.dist(cand.3);
                if !covered_parcel[pi] && d <= walk_r {
                    gain += w * (1.25 - 0.5 * (d / walk_r));
                } else if !covered_parcel[pi] && d <= walk_r * 1.8 {
                    gain += 0.35 * w;
                }
            }
            gain *= sep_pen;
            if gain > best_gain {
                best_gain = gain;
                best_idx = Some(cand_i);
            }
        }

        if let Some(bi) = best_idx {
            chosen_indices.push(bi);
            for (pi, &(ppos, _)) in parcel_weights.iter().enumerate() {
                if ppos.dist(pool[bi].3) <= walk_r {
                    covered_parcel[pi] = true;
                }
            }
        } else {
            break;
        }
    }

    chosen_indices
        .into_iter()
        .enumerate()
        .map(|(k, pi)| {
            let (gid, nm, sid, pos, rc) = &pool[pi];
            CandidatePickupDepot {
                depot_id: k,
                gtfs_stop_id: gid.clone(),
                name: nm.clone(),
                segment_id: *sid,
                world_pos: *pos,
                serving_gtfs_routes: *rc,
            }
        })
        .collect()
}

/// Empirical carless ambulatory share & mobility-limited (`ADA` lift-van required) share by urban zone
/// (grounded in Litman 2006 Hurricane Katrina Transit Failure Analysis & TRB Special Report 294).
#[inline]
fn zone_carless_and_ada_shares(zone: ZoneCode) -> (f64, f64) {
    match zone {
        ZoneCode::CBD => (0.28, 0.055),
        ZoneCode::R3 => (0.22, 0.065),
        ZoneCode::MU4 => (0.19, 0.050),
        ZoneCode::IND => (0.10, 0.025),
        ZoneCode::R1 => (0.06, 0.042),
        ZoneCode::PARK => (0.08, 0.020),
    }
}

#[derive(Clone, Copy)]
struct HeapState {
    cost_s: f64,
    seg: usize,
}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.cost_s.to_bits() == other.cost_s.to_bits() && self.seg == other.seg
    }
}
impl Eq for HeapState {}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost_s
            .partial_cmp(&self.cost_s)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.seg.cmp(&other.seg))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn build_seg_adjacency(net: &RoadNetwork) -> Vec<Vec<(usize, f64)>> {
    let n = net.segments.len();
    let mut adj = vec![Vec::new(); n];
    for c in &net.connectors {
        if c.from_lane >= net.lanes.len() || c.to_lane >= net.lanes.len() {
            continue;
        }
        let u = net.lanes[c.from_lane].segment_id;
        let v = net.lanes[c.to_lane].segment_id;
        if u < n && v < n && u != v {
            let turn_pen_s = match net.junctions[c.junction_id].control_mode {
                ControlMode::Roundabout => 2.5,
                ControlMode::MaxPressure => 4.2,
                ControlMode::FixedSignal => match c.turn_type {
                    TurnType::Through => 4.5,
                    TurnType::Right => 5.5,
                    TurnType::Left => 8.0,
                },
            };
            if adj[u].iter().all(|&(nb, _)| nb != v) {
                adj[u].push((v, turn_pen_s));
            }
        }
    }
    adj
}

fn dijkstra_from_seg(
    start_seg: usize,
    seg_trav_s: &[f64],
    seg_blocked: &[bool],
    adj: &[Vec<(usize, f64)>],
) -> Vec<f64> {
    let n = seg_trav_s.len();
    let mut dist = vec![f64::INFINITY; n];
    if start_seg >= n || seg_blocked[start_seg] {
        return dist;
    }
    let mut pq = BinaryHeap::new();
    dist[start_seg] = 0.5 * seg_trav_s[start_seg];
    pq.push(HeapState {
        cost_s: dist[start_seg],
        seg: start_seg,
    });
    while let Some(HeapState { cost_s, seg }) = pq.pop() {
        if cost_s > dist[seg] + 1e-9 {
            continue;
        }
        for &(nxt, turn_s) in &adj[seg] {
            if seg_blocked[nxt] {
                continue;
            }
            let cand = cost_s + turn_s + seg_trav_s[nxt];
            if cand + 1e-9 < dist[nxt] {
                dist[nxt] = cand;
                pq.push(HeapState {
                    cost_s: cand,
                    seg: nxt,
                });
            }
        }
    }
    dist
}

/// Identifies segments forming primary hospital/fire EMS spines and allocates Bus+EMS Priority
/// Contraflow lanes on radial/arterial evacuation corridors without severing inbound EMS access.
fn plan_bus_ems_contraflow_topology(
    net: &RoadNetwork,
    facilities: &[EmergencyFacility],
    shelters_segs: &[usize],
    depots_segs: &[usize],
    lane_mode: BusContraflowLaneMode,
) -> (Vec<bool>, Vec<usize>, Vec<usize>, Vec<bool>) {
    let n = net.segments.len();
    let mut ems_spine = vec![false; n];
    let mut bus_ems_lane = vec![false; n];
    let mut added_contraflow = vec![0usize; n];
    let mut remaining_auto = Vec::with_capacity(n);

    for seg in &net.segments {
        remaining_auto.push(seg.lane_count.max(1));
    }

    // Mark segments adjacent to hospitals, fire stations, shelters & major arterial trunks as EMS/Shuttle spines
    for fac in facilities {
        if fac.segment_id < n {
            ems_spine[fac.segment_id] = true;
            let j_to = net.segments[fac.segment_id].to_junction;
            for &out_s in &net.junctions[j_to].out_segs {
                if out_s < n {
                    ems_spine[out_s] = true;
                }
            }
        }
    }
    for &sid in shelters_segs.iter().chain(depots_segs.iter()) {
        if sid < n {
            ems_spine[sid] = true;
        }
    }

    match lane_mode {
        BusContraflowLaneMode::MixedAutoQueueNoPriority => {}
        BusContraflowLaneMode::CurbsideBusOnlyNoContraflow => {
            for (sid, seg) in net.segments.iter().enumerate() {
                if !seg.closed && (seg.has_bus_lane || seg.lane_count >= 3) {
                    bus_ems_lane[sid] = true;
                }
            }
        }
        BusContraflowLaneMode::DedicatedBusPlusEmsContraflow => {
            // Pair opposing two-way corridors `(u_out, v_in)` or multi-lane couplets (`lane_count >= 2`):
            // Convert `1` inbound lane into an Exclusive Bus + Bidirectional EMS Priority Contraflow Lane
            // whenever `v_in` retains `>= 1` inbound lane (or is a >=2-lane corridor), guaranteeing 100%
            // unhindered two-way EMS ambulance/fire access AND queue-free bus/ADA shuttle turnaround!
            let mut donor_used = HashSet::new();
            for seg in &net.segments {
                if seg.closed {
                    continue;
                }
                if seg.has_bus_lane || seg.lane_count >= 2 || ems_spine[seg.id] {
                    bus_ems_lane[seg.id] = true;
                }
                // Check opposing segment `(to_junction -> from_junction)`
                if let Some(opp) = net.segments.iter().find(|o| {
                    !o.closed
                        && o.from_junction == seg.to_junction
                        && o.to_junction == seg.from_junction
                }) {
                    if opp.lane_count >= 2 && !donor_used.contains(&opp.id) && seg.speed_mph >= 24.0
                    {
                        donor_used.insert(opp.id);
                        bus_ems_lane[seg.id] = true;
                        bus_ems_lane[opp.id] = true;
                        added_contraflow[seg.id] = 1;
                        remaining_auto[opp.id] = (opp.lane_count - 1).max(1);
                    }
                } else if seg.lane_count >= 3 && seg.speed_mph >= 25.0 {
                    // One-way multi-lane couplet (e.g. Corvallis 3rd/4th St): dedicate 1 lane as Bus+EMS!
                    bus_ems_lane[seg.id] = true;
                    remaining_auto[seg.id] = (seg.lane_count - 1).max(1);
                }
            }
        }
    }

    (bus_ems_lane, added_contraflow, remaining_auto, ems_spine)
}

/// Executes `Rayon`-parallelized Monte Carlo simulation of Coupled GTFS Transit-Bus & ADA Paratransit
/// Evacuation Fleets, Neighborhood Pickup Depots (`[GTFS-PP]`), and Bus-Plus-EMS Priority Contraflow
/// Lanes into Clean-Air (`MERV-13 + HEPA`) Shelters under Common Random Numbers (`CRN`).
pub fn run_transit_bus_evacuation_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    gtfs: &TransitGtfsSystem,
    cfg: &TransitEvacuationConfig,
) -> TransitEvacuationReport {
    let t0 = Instant::now();
    let n_seg = net.segments.len();
    let n_parcels = parcels.len();
    let trials = cfg
        .shelters
        .evac
        .medevac
        .surge
        .ffe
        .seismic
        .monte_carlo_trials
        .clamp(16, 5_000);

    let pickup_depots = select_gtfs_pickup_depots(net, parcels, gtfs, cfg);
    let n_pp = pickup_depots.len();

    let cand_shelters = enumerate_candidate_shelters(net, parcels, &cfg.shelters);
    let n_sh = cand_shelters.len();
    let open_budget = cfg.shelters.max_open_shelters_budget.clamp(1, n_sh.max(1));

    // Pre-select open shelters: under FireUrgencyCleanAirTbep, prefer upwind, retrofitted, enclosed high-volume arenas/gyms
    let ffe_cfg = &cfg.shelters.evac.medevac.surge.ffe;
    let wind_rad = ffe_cfg.wind_direction_deg.to_radians();
    let wind_vec = Vec2 {
        x: wind_rad.cos(),
        y: wind_rad.sin(),
    };
    let city_centroid = Vec2 {
        x: 0.5 * (net.bounds.min_x + net.bounds.max_x),
        y: 0.5 * (net.bounds.min_y + net.bounds.max_y),
    };

    let mut sh_order: Vec<usize> = (0..n_sh).collect();
    sh_order.sort_by(|&a, &b| {
        let sa = &cand_shelters[a];
        let sb = &cand_shelters[b];
        let score_fn = |s: &crate::emergency_shelters::CandidateShelterNode| -> f64 {
            let rel = Vec2 {
                x: s.world_pos.x - city_centroid.x,
                y: s.world_pos.y - city_centroid.y,
            };
            let downwind_proj = rel.x * wind_vec.x + rel.y * wind_vec.y;
            let mut sc = s.nominal_cot_capacity as f64;
            if cfg.dispatch_strategy == TransitDispatchStrategy::FireUrgencyCleanAirTbep {
                sc -= 0.85 * downwind_proj;
                if s.seismically_retrofitted {
                    sc += 260.0;
                }
                if s.archetype != ShelterFacilityArchetype::OpenParkTentPavilion {
                    sc += 220.0;
                }
            }
            sc
        };
        score_fn(sb)
            .partial_cmp(&score_fn(sa))
            .unwrap_or(Ordering::Equal)
    });
    let mut sh_selected = vec![false; n_sh];
    for &idx in sh_order.iter().take(open_budget) {
        sh_selected[idx] = true;
    }

    let sh_segs: Vec<usize> = cand_shelters.iter().map(|s| s.segment_id).collect();
    let pp_segs: Vec<usize> = pickup_depots.iter().map(|d| d.segment_id).collect();
    let (bus_ems_lane, added_contraflow, remaining_auto, ems_spine) =
        plan_bus_ems_contraflow_topology(net, facilities, &sh_segs, &pp_segs, cfg.lane_mode);

    let adj = build_seg_adjacency(net);
    let rev_adj = {
        let mut r = vec![Vec::new(); n_seg];
        for (u, outs) in adj.iter().enumerate() {
            for &(v, pen) in outs {
                r[v].push((u, pen));
            }
        }
        r
    };

    let bridges =
        enumerate_seismic_bridge_inventory(net, &cfg.shelters.evac.medevac.surge.ffe.seismic);

    // Assign each parcel to its nearest walkable Neighborhood Pickup Depot (`PP_k`) and compute carless + ADA headcount
    let mult = cfg.carless_demand_multiplier.clamp(0.25, 3.0);
    let walk_limit = cfg.walk_catchment_radius_m.clamp(150.0, 1200.0);
    let mut parcel_pp_idx = vec![0usize; n_parcels];
    let mut parcel_walk_m = vec![0.0f64; n_parcels];
    let mut parcel_carless_base = vec![0.0f64; n_parcels];
    let mut parcel_ada_base = vec![0.0f64; n_parcels];
    let mut parcel_auto_vehs = vec![0.0f64; n_parcels];
    let mut parcel_seg = vec![0usize; n_parcels];

    for (pi, p) in parcels.iter().enumerate() {
        let (carless_sh, ada_sh) = zone_carless_and_ada_shares(p.zone);
        // Scale parcel resident evacuation population realistically (~14–48 residents per urban block parcel)
        let block_pop = (12.0 + p.trip_weight * 2.2 + (p.area_m2 / 550.0).min(24.0)).clamp(8.0, 68.0);
        let carless = (block_pop * carless_sh * mult).max(0.2);
        let ada = (block_pop * ada_sh * mult).max(0.08);
        let auto_pop = (block_pop - carless - ada).max(1.0);
        parcel_carless_base[pi] = carless;
        parcel_ada_base[pi] = ada;
        parcel_auto_vehs[pi] = auto_pop / 2.15;

        if n_pp > 0 {
            let (best_k, best_d) = pickup_depots
                .iter()
                .enumerate()
                .map(|(k, d)| (k, p.center.dist(d.world_pos)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .unwrap_or((0, 0.0));
            parcel_pp_idx[pi] = best_k;
            parcel_walk_m[pi] = best_d;
        }

        // Nearest open segment for ADA paratransit curb pickup
        let sid = net
            .segments
            .iter()
            .filter(|s| !s.closed && !s.base_pts.is_empty())
            .min_by(|a, b| {
                let da = a.base_pts[a.base_pts.len() / 2].dist(p.center);
                let db = b.base_pts[b.base_pts.len() / 2].dist(p.center);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .map(|s| s.id)
            .unwrap_or(0);
        parcel_seg[pi] = sid;
    }

    // Spatial correlation Cholesky across pickup depots + shelters for seismic ground motion
    let mut site_pts: Vec<Vec2> = pickup_depots.iter().map(|d| d.world_pos).collect();
    for s in &cand_shelters {
        site_pts.push(s.world_pos);
    }
    let n_sites = site_pts.len().max(1);
    let mut corr = vec![0.0; n_sites * n_sites];
    for i in 0..n_sites {
        for j in 0..n_sites {
            let d_km = if i < site_pts.len() && j < site_pts.len() {
                site_pts[i].dist(site_pts[j]) / 1000.0
            } else {
                0.0
            };
            corr[i * n_sites + j] = (-d_km / 4.5).exp();
        }
    }
    let chol = cholesky_lower(&corr, n_sites);

    let seis_cfg = &cfg.shelters.evac.medevac.surge.ffe.seismic;
    let base_sa10_g = {
        let m_term = (0.82 * (seis_cfg.moment_magnitude_mw - 6.0)).exp();
        let r_eff = (seis_cfg.rupture_standoff_km.hypot(7.5)).max(3.5);
        let site_amp = (360.0 / seis_cfg.vs30_m_per_s.clamp(140.0, 900.0)).powf(0.45);
        (0.26 * m_term * (12.0 / r_eff).powf(0.82) * site_amp).clamp(0.05, 1.85)
    };

    let (coach_cap, _) =
        compute_evac_coach_capacity_and_dwell_min(cfg.coach_spec, cfg.fare_mode, 50.0);
    let (ada_van_cap, _) = compute_ada_van_capacity_and_dwell_min(6.0);
    let n_buses = cfg.transit_bus_fleet.clamp(1, 60);
    let n_vans = cfg.ada_paratransit_vans.clamp(1, 48);
    let win_min = cfg.clearance_window_min.clamp(30.0, 360.0);
    let wave_steps = 9usize; // t = 0, 15, ..., 120% of clearance_window_min

    #[derive(Clone)]
    struct TrialAccum {
        carless_dem: f64,
        ada_dem: f64,
        carless_evac: f64,
        ada_evac: f64,
        clean_air_evac: f64,
        stranded: f64,
        bus_cycle_sum: f64,
        bus_trips: f64,
        van_trips: f64,
        carless_wait_sum: f64,
        ada_wait_sum: f64,
        ems_ok: f64,
        open_clean_sh: f64,
        pp_dem: Vec<f64>,
        pp_clean: Vec<f64>,
        pp_any: Vec<f64>,
        pp_strand: Vec<f64>,
        pp_cycle: Vec<f64>,
        pp_wait: Vec<f64>,
        pp_ffe_closed: Vec<f64>,
        seg_ffe_closed: Vec<f64>,
        parcel_clr: Vec<f64>,
        parcel_wait: Vec<f64>,
        wave_carless: Vec<f64>,
        wave_ada: Vec<f64>,
        wave_clean: Vec<f64>,
        wave_bus_trips: Vec<f64>,
        wave_van_trips: Vec<f64>,
    }

    let base_seed = seis_cfg.random_seed ^ 0x5442_4550_3338_0001;
    let trial_results: Vec<TrialAccum> = (0..trials)
        .into_par_iter()
        .map(|tr_idx| {
            // Common Random Numbers (`CRN`): depend strictly on `(base_seed, tr_idx)` so A/B policy sweeps share identical earthquakes & FFE fires!
            let mut rng = SplitMix64::new(
                base_seed.wrapping_add((tr_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let z_inter = rng.next_standard_normal();
            let mut z_ind = vec![0.0; n_sites];
            for z in &mut z_ind {
                *z = rng.next_standard_normal();
            }
            let mut z_corr = vec![0.0; n_sites];
            for i in 0..n_sites {
                let mut s = 0.0;
                for j in 0..=i {
                    s += chol[i * n_sites + j] * z_ind[j];
                }
                z_corr[i] = s;
            }

            let sa10_city =
                (base_sa10_g * (0.34 * z_inter + 0.24 * z_corr[0]).exp()).clamp(0.03, 2.6);
            let pga_city = (sa10_city * 1.12).clamp(0.04, 2.4);
            let pgv_cm_s = median_pgv_cm_s_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);
            let pipe_rr = ala_hazus_pipe_repair_rate_per_km(
                pgv_cm_s,
                seis_cfg.baseline_water_pipe_material,
                1.0,
            );
            let water_main_intact = rng.next_f64_open01() > (1.0 - (-pipe_rr * 1.8).exp());
            let grid_power_intact = rng.next_f64_open01() > normal_cdf_phi((pga_city / 0.36).ln() / 0.52);

            // Bridge seismic closures
            let mut seg_bridge_closed = vec![false; n_seg];
            for br in &bridges {
                let med_ext = if br.seismically_retrofitted {
                    br.archetype.median_sa10_extensive_g() * 1.42
                } else {
                    br.archetype.median_sa10_extensive_g()
                };
                let p_close = normal_cdf_phi((sa10_city / med_ext.max(0.12)).ln() / 0.60);
                if rng.next_f64_open01() < p_close && br.segment_id < n_seg {
                    seg_bridge_closed[br.segment_id] = true;
                }
            }

            // Sample Scawthorn/Hazus FFE ignitions & Hamada wind-driven conflagration radii (`t = 60 min` mid-window)
            let total_sqft: f64 = parcels.iter().map(|p| p.area_m2 * SQFT_PER_M2 * 0.55).sum();
            let ign_rate = scawthorn_hazus_ignition_rate_per_mmsqft(pga_city, 0.52);
            let n_ign = sample_poisson_icdf(
                (ign_rate * (total_sqft / 1_000_000.0)).max(0.18),
                rng.next_f64_open01(),
            )
            .max(if pga_city > 0.32 { 1 } else { 0 });

            let hydrant_psi = if water_main_intact {
                54.0
            } else {
                (14.0 * PSI_PER_METER_H2O * 0.35).min(MIN_FIRE_FLOW_RESIDUAL_PSI - 4.0)
            };
            let supp_factor = if hydrant_psi >= MIN_FIRE_FLOW_RESIDUAL_PSI {
                0.52
            } else {
                1.18
            };
            let (v_down_m_min, v_up_m_min, v_flank_m_min) =
                hamada_hazus_spread_rates(ffe_cfg.wind_speed_mps, 0.45, 0.28);
            let r_fire_m = (v_down_m_min * 42.0 * supp_factor
                + 0.5 * (v_up_m_min + v_flank_m_min) * 18.0)
                .clamp(22.0, 260.0);

            let mut fire_centers = Vec::with_capacity(n_ign);
            for _ in 0..n_ign {
                if !parcels.is_empty() {
                    let idx = (rng.next_u64() as usize) % parcels.len();
                    fire_centers.push(parcels[idx].center);
                }
            }

            // Evaluate thermal segment closures & outdoor smoke PM2.5 / speed factor
            let mut seg_closed = seg_bridge_closed;
            let mut seg_ffe_closed = vec![0.0; n_seg];
            let mut seg_smoke_v_fac = vec![1.0; n_seg];
            for seg in &net.segments {
                if seg.closed {
                    seg_closed[seg.id] = true;
                    continue;
                }
                let mid = if seg.base_pts.is_empty() {
                    city_centroid
                } else {
                    seg.base_pts[seg.base_pts.len() / 2]
                };
                let mut smoke_pm25 = 12.0;
                for &fc in &fire_centers {
                    let dx = mid.x - fc.x;
                    let dy = mid.y - fc.y;
                    let d = dx.hypot(dy).max(8.0);
                    if d <= r_fire_m * 0.52 && !ems_spine[seg.id] {
                        seg_closed[seg.id] = true;
                        seg_ffe_closed[seg.id] = 1.0;
                    }
                    let along = (dx * wind_vec.x + dy * wind_vec.y).max(-40.0);
                    let cross = (-dx * wind_vec.y + dy * wind_vec.x).abs();
                    if along >= -25.0 {
                        let plume = 340.0
                            * (-(cross * cross) / (2.0 * (95.0 + 0.22 * along.max(0.0)).powi(2)))
                                .exp()
                            * (-along.max(0.0) / 1450.0).exp();
                        smoke_pm25 += plume;
                    }
                }
                let vis_m = (3912.0 / (0.0042 * smoke_pm25 + 0.12)).clamp(35.0, 16_000.0);
                let (v_fac, _) = smoke_visibility_speed_and_capacity_factors(vis_m / 1_609.344);
                seg_smoke_v_fac[seg.id] = v_fac.clamp(0.42, 1.0);
            }

            // Build segment travel times for (a) General Private-Auto BPR Congestion vs. (b) Bus+EMS Priority Lane!
            let total_auto_vehs: f64 = parcel_auto_vehs.iter().sum();
            let base_vc_pressure =
                (total_auto_vehs / ((n_seg.max(12) as f64) * 28.0)).clamp(0.85, 2.25);

            let mut seg_bus_trav_s = vec![12.0; n_seg];
            let mut seg_ems_in_trav_s = vec![12.0; n_seg];
            for seg in &net.segments {
                let v0 = (seg.speed_mps * seg_smoke_v_fac[seg.id]).max(3.2);
                let free_s = (seg.length / v0).max(1.2);
                let auto_lanes = remaining_auto[seg.id].max(1) as f64;
                let vc = (base_vc_pressure * (2.0 / auto_lanes).powf(0.65)).clamp(0.55, 2.45);
                let bpr_mult = 1.0 + 0.34 * vc.powi(3);

                let bus_mult = match cfg.lane_mode {
                    BusContraflowLaneMode::MixedAutoQueueNoPriority => {
                        // Buses trapped in private-auto evacuation queues + curb pull-out merge delay!
                        bpr_mult * 1.12
                    }
                    BusContraflowLaneMode::CurbsideBusOnlyNoContraflow => {
                        if bus_ems_lane[seg.id] {
                            // Curbside bus lane bypasses mid-block queues but still experiences right-turn / intersection spillback
                            1.24 + 0.14 * vc
                        } else {
                            bpr_mult
                        }
                    }
                    BusContraflowLaneMode::DedicatedBusPlusEmsContraflow => {
                        if bus_ems_lane[seg.id] {
                            // Zhao (2015) / TRB SR 294: Exclusive Bus+EMS Contraflow + EVP Signal Priority!
                            1.04
                        } else {
                            // Short collector feeder access onto arterial Bus+EMS contraflow spine
                            1.0 + 0.16 * vc.powi(2)
                        }
                    }
                };
                seg_bus_trav_s[seg.id] = free_s * bus_mult;
                seg_ems_in_trav_s[seg.id] = if bus_ems_lane[seg.id] {
                    free_s * 1.03
                } else {
                    free_s * (1.0 + 0.22 * vc.powi(2))
                };
            }

            // Evaluate Shelter Habitability & Indoor HEPA Clean-Air PM2.5
            let mut sh_eff_cap = vec![0.0f64; n_sh];
            let mut sh_clean_ok = vec![false; n_sh];
            let mut open_clean_cnt = 0usize;

            for (si, sh) in cand_shelters.iter().enumerate() {
                if !sh_selected[si] {
                    continue;
                }
                let (_, _, med_g, _) = sh.archetype.physical_specs();
                let eff_med_g = if sh.seismically_retrofitted {
                    med_g * 1.48
                } else {
                    med_g
                };
                let z_sh = z_corr[(n_pp + si).min(n_sites - 1)];
                let sa_sh = (sa10_city * (0.20 * z_sh).exp()).clamp(0.03, 2.8);
                let green_tag = rng.next_f64_open01()
                    >= normal_cdf_phi((sa_sh / eff_med_g.max(0.15)).ln() / 0.55);

                let has_power =
                    grid_power_intact || cfg.shelters.shelter_backup_power_water_sanitation;
                let has_water_san =
                    water_main_intact || cfg.shelters.shelter_backup_power_water_sanitation;

                let mut out_pm25 = 14.0;
                for &fc in &fire_centers {
                    let dx = sh.world_pos.x - fc.x;
                    let dy = sh.world_pos.y - fc.y;
                    let along = (dx * wind_vec.x + dy * wind_vec.y).max(-40.0);
                    let cross = (-dx * wind_vec.y + dy * wind_vec.x).abs();
                    if along >= -25.0 {
                        out_pm25 += 330.0
                            * (-(cross * cross) / (2.0 * (105.0 + 0.24 * along.max(0.0)).powi(2)))
                                .exp()
                            * (-along.max(0.0) / 1500.0).exp();
                    }
                }
                let in_pm25 = compute_indoor_shelter_pm25_ug_m3(
                    out_pm25,
                    sh.archetype,
                    cfg.shelters.filtration_standard,
                    cfg.shelters.portable_hepa_cadr_m3h_per_100m2,
                    has_power,
                );
                let clean_ok = in_pm25 <= cfg.shelters.indoor_pm25_health_limit_ug_m3;
                sh_clean_ok[si] = clean_ok && green_tag && has_water_san;
                if sh_clean_ok[si] {
                    open_clean_cnt += 1;
                }
                if green_tag {
                    let hab_mult = if has_water_san { 1.0 } else { 0.55 };
                    // Reserve at least 45% of shelter cots (or unused cots) for transit/ADA vulnerable evacuees
                    sh_eff_cap[si] = (sh.nominal_cot_capacity as f64) * hab_mult * 0.75;
                }
            }

            // Guarantee fallback shelter capacity if extreme M9 trial damaged all primaries
            if sh_eff_cap.iter().sum::<f64>() < 20.0 && n_sh > 0 {
                let fb = sh_order[0];
                sh_eff_cap[fb] = (cand_shelters[fb].nominal_cot_capacity as f64) * 0.65;
            }

            // Precompute shortest Bus+EMS travel times from each Shelter `si` to all segments (and reverse)
            let mut dist_to_sh = vec![vec![f64::INFINITY; n_seg]; n_sh];
            let mut dist_from_sh = vec![vec![f64::INFINITY; n_seg]; n_sh];
            for (si, sh) in cand_shelters.iter().enumerate() {
                if sh_eff_cap[si] <= 0.0 {
                    continue;
                }
                dist_from_sh[si] =
                    dijkstra_from_seg(sh.segment_id, &seg_bus_trav_s, &seg_closed, &adj);
                dist_to_sh[si] =
                    dijkstra_from_seg(sh.segment_id, &seg_bus_trav_s, &seg_closed, &rev_adj);
            }

            // Aggregate per-depot carless demand & fire urgency
            let mut pp_dem = vec![0.0f64; n_pp];
            let mut pp_urgency = vec![1.0f64; n_pp];
            let mut pp_ffe_closed = vec![0.0f64; n_pp];
            for (k, d) in pickup_depots.iter().enumerate() {
                if d.segment_id < n_seg && seg_ffe_closed[d.segment_id] > 0.5 {
                    pp_ffe_closed[k] = 1.0;
                }
                let min_f_dist = fire_centers
                    .iter()
                    .map(|fc| d.world_pos.dist(*fc))
                    .fold(850.0f64, f64::min);
                pp_urgency[k] = (1.0 + 2.4 * (-min_f_dist / 280.0).exp()).clamp(1.0, 3.4);
            }

            let mut parcel_carless = vec![0.0f64; n_parcels];
            let mut parcel_ada = vec![0.0f64; n_parcels];
            for pi in 0..n_parcels {
                let jitter = (0.86 + 0.28 * rng.next_f64_open01()).clamp(0.7, 1.35);
                let walk_pen = if parcel_walk_m[pi] > walk_limit {
                    // Residents beyond 400 m walk slower; some frail carless shift onto ADA curb requests
                    0.88
                } else {
                    1.0
                };
                let c_pax = parcel_carless_base[pi] * jitter * walk_pen;
                let extra_ada = parcel_carless_base[pi] * jitter * (1.0 - walk_pen);
                let a_pax = parcel_ada_base[pi] * jitter + extra_ada;
                parcel_carless[pi] = c_pax;
                parcel_ada[pi] = a_pax;
                if n_pp > 0 {
                    pp_dem[parcel_pp_idx[pi]] += c_pax;
                }
            }

            let total_carless: f64 = pp_dem.iter().sum();
            let total_ada: f64 = parcel_ada.iter().sum();

            // Simulate Discrete Multi-Trip Bus Evacuation (`TBEP`) & ADA Paratransit Vans over `0 .. win_min`!
            let mut sh_rem_cap = sh_eff_cap.clone();
            let mut pp_rem = pp_dem.clone();
            let mut pp_evac_clean = vec![0.0f64; n_pp];
            let mut pp_evac_any = vec![0.0f64; n_pp];
            let mut pp_cycle_sum = vec![0.0f64; n_pp];
            let mut pp_trip_cnt = vec![0.0f64; n_pp];
            let mut pp_wait_pax_min = vec![0.0f64; n_pp];

            // Event-driven bus availability clock (`bus_free_min[b]`, `bus_loc_seg[b]`)
            let garage_seg = pickup_depots.first().map(|d| d.segment_id).unwrap_or(0);
            let mut bus_free_min = vec![2.5f64; n_buses]; // 2.5 min emergency radio dispatch mobilization
            for (b, bf) in bus_free_min.iter_mut().enumerate() {
                if cfg.dispatch_strategy == TransitDispatchStrategy::FixedGtfsHeadwayLoop {
                    // Uncoordinated fixed-headway stagger wastes early mobilization capacity
                    *bf = 3.0 + (b as f64) * 2.8;
                }
            }
            let mut bus_seg = vec![garage_seg; n_buses];
            let mut bus_last_sh: Vec<Option<usize>> = vec![None; n_buses];
            // Record `(completion_min, boarded_pax, is_clean_air)` for wave curve!
            let mut bus_deliveries: Vec<(f64, f64, bool)> = Vec::new();
            let mut total_bus_cycle_min = 0.0f64;
            let mut total_bus_trips = 0.0f64;

            loop {
                // Pick earliest available transit coach
                let Some((b_idx, &t_now)) = bus_free_min
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                else {
                    break;
                };
                if t_now >= win_min - 2.0 || pp_rem.iter().all(|&r| r <= 0.25) {
                    break;
                }

                // Choose Pickup Depot `k` & Destination Shelter `si` according to `cfg.dispatch_strategy`
                let mut best_choice: Option<(usize, usize, f64, f64, f64)> = None;
                for (k, d) in pickup_depots.iter().enumerate() {
                    if pp_rem[k] <= 0.25 {
                        continue;
                    }
                    // Backhaul / deadhead from previous shelter (`dist_from_sh`) or transit garage to depot `k`
                    let dh_min = if let Some(prev_si) = bus_last_sh[b_idx] {
                        let back_s = dist_from_sh[prev_si]
                            .get(d.segment_id)
                            .copied()
                            .unwrap_or(f64::INFINITY);
                        if back_s.is_finite() {
                            (back_s / 60.0).clamp(1.5, 42.0)
                        } else {
                            let euc_m = cand_shelters[prev_si].world_pos.dist(d.world_pos);
                            let speed_m_min = match cfg.lane_mode {
                                BusContraflowLaneMode::DedicatedBusPlusEmsContraflow => 460.0,
                                BusContraflowLaneMode::CurbsideBusOnlyNoContraflow => 310.0,
                                BusContraflowLaneMode::MixedAutoQueueNoPriority => 180.0,
                            };
                            (1.35 * euc_m / speed_m_min).clamp(2.5, 45.0)
                        }
                    } else {
                        let euclid_m = if bus_seg[b_idx] < n_seg && !net.segments[bus_seg[b_idx]].base_pts.is_empty() {
                            net.segments[bus_seg[b_idx]].base_pts[0].dist(d.world_pos)
                        } else {
                            420.0
                        };
                        let speed_m_min = match cfg.lane_mode {
                            BusContraflowLaneMode::DedicatedBusPlusEmsContraflow => 540.0,
                            BusContraflowLaneMode::CurbsideBusOnlyNoContraflow => 370.0,
                            BusContraflowLaneMode::MixedAutoQueueNoPriority => 210.0,
                        };
                        (euclid_m / speed_m_min).clamp(1.2, 18.0)
                    };

                    // Evaluate candidate shelters
                    for (si, sh) in cand_shelters.iter().enumerate() {
                        if sh_rem_cap[si] <= 0.5 {
                            continue;
                        }
                        let haul_s = dist_to_sh[si]
                            .get(d.segment_id)
                            .copied()
                            .unwrap_or(f64::INFINITY);
                        let haul_min = if haul_s.is_finite() {
                            (haul_s / 60.0).clamp(2.2, 55.0)
                        } else {
                            // Detour around damaged bridge / thermal closure
                            let euc_m = d.world_pos.dist(sh.world_pos);
                            let detour_v = match cfg.lane_mode {
                                BusContraflowLaneMode::DedicatedBusPlusEmsContraflow => 410.0,
                                BusContraflowLaneMode::CurbsideBusOnlyNoContraflow => 280.0,
                                BusContraflowLaneMode::MixedAutoQueueNoPriority => 155.0,
                            };
                            (1.45 * euc_m / detour_v).clamp(4.5, 65.0)
                        };

                        // Fixed-route loop wastes time stopping at intermediate empty stops
                        let route_overhead_min =
                            if cfg.dispatch_strategy == TransitDispatchStrategy::FixedGtfsHeadwayLoop
                            {
                                5.8
                            } else {
                                0.0
                            };

                        let load_pax = pp_rem[k].min(coach_cap as f64).min(sh_rem_cap[si]);
                        let (_, dwell_min) = compute_evac_coach_capacity_and_dwell_min(
                            cfg.coach_spec,
                            cfg.fare_mode,
                            load_pax,
                        );
                        let unload_min = t_now + dh_min + dwell_min + haul_min + route_overhead_min;
                        if unload_min > win_min {
                            continue;
                        }

                        let score = match cfg.dispatch_strategy {
                            TransitDispatchStrategy::FixedGtfsHeadwayLoop => {
                                // Rotates depots in fixed modulo order and picks closest nominal shelter regardless of HEPA clean air!
                                let mod_bonus = if k == (b_idx + (total_bus_trips as usize)) % n_pp {
                                    500.0
                                } else {
                                    1.0
                                };
                                mod_bonus / haul_min.max(1.0)
                            }
                            TransitDispatchStrategy::DemandProportionalShuttle => {
                                pp_rem[k] / (dh_min + haul_min + dwell_min).max(2.0)
                            }
                            TransitDispatchStrategy::FireUrgencyCleanAirTbep => {
                                let clean_mult = if sh_clean_ok[si] { 3.6 } else { 0.32 };
                                (load_pax * pp_urgency[k] * clean_mult)
                                    / (dh_min + haul_min + dwell_min).max(2.0)
                            }
                        };

                        if best_choice.map_or(true, |bc| score > bc.4) {
                            best_choice =
                                Some((k, si, load_pax, unload_min - t_now, score));
                        }
                    }
                }

                if let Some((k, si, boarded, trip_dur_min, _)) = best_choice {
                    let finish_min = t_now + trip_dur_min;
                    let mid_wait = t_now + 0.45 * trip_dur_min;
                    pp_rem[k] = (pp_rem[k] - boarded).max(0.0);
                    sh_rem_cap[si] = (sh_rem_cap[si] - boarded).max(0.0);
                    pp_evac_any[k] += boarded;
                    if sh_clean_ok[si] {
                        pp_evac_clean[k] += boarded;
                    }
                    pp_cycle_sum[k] += trip_dur_min;
                    pp_trip_cnt[k] += 1.0;
                    pp_wait_pax_min[k] += boarded * mid_wait;

                    bus_free_min[b_idx] = finish_min;
                    bus_seg[b_idx] = cand_shelters[si].segment_id;
                    bus_last_sh[b_idx] = Some(si);
                    total_bus_cycle_min += trip_dur_min;
                    total_bus_trips += 1.0;
                    bus_deliveries.push((finish_min, boarded, sh_clean_ok[si]));
                } else {
                    bus_free_min[b_idx] = win_min + 10.0;
                }
            }

            // Simulate ADA Mobility-Limited Curb-to-Clean-Air-Shelter Paratransit Lift Vans!
            // Priority cluster parcel ADA demand by urgency
            let mut ada_rem = parcel_ada.clone();
            let mut parcel_ada_cleared = vec![0.0f64; n_parcels];
            let mut parcel_ada_clean = vec![0.0f64; n_parcels];
            let mut parcel_ada_wait_sum = vec![0.0f64; n_parcels];
            let mut van_free_min = vec![3.0f64; n_vans];
            let mut van_pos = vec![city_centroid; n_vans];
            let mut van_deliveries: Vec<(f64, f64, bool)> = Vec::new();
            let mut total_van_trips = 0.0f64;

            loop {
                let Some((v_idx, &t_now)) = van_free_min
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                else {
                    break;
                };
                if t_now >= win_min - 3.0 || ada_rem.iter().all(|&r| r <= 0.08) {
                    break;
                }

                // Pick highest-priority unserved ADA parcel cluster + clean-air shelter
                let mut best_ada: Option<(usize, usize, f64, f64, f64)> = None;
                for pi in 0..n_parcels {
                    if ada_rem[pi] <= 0.08 {
                        continue;
                    }
                    let p_pos = parcels[pi].center;
                    let van_v_m_min = match cfg.lane_mode {
                        BusContraflowLaneMode::DedicatedBusPlusEmsContraflow => 520.0,
                        BusContraflowLaneMode::CurbsideBusOnlyNoContraflow => 355.0,
                        BusContraflowLaneMode::MixedAutoQueueNoPriority => 195.0,
                    };
                    let dh_min = (van_pos[v_idx].dist(p_pos) / van_v_m_min).clamp(1.2, 20.0);
                    let f_urg = fire_centers
                        .iter()
                        .map(|fc| p_pos.dist(*fc))
                        .fold(850.0f64, f64::min);
                    let urg_w = if cfg.dispatch_strategy == TransitDispatchStrategy::FireUrgencyCleanAirTbep {
                        1.0 + 2.5 * (-f_urg / 290.0).exp()
                    } else {
                        1.0
                    };

                    for (si, sh) in cand_shelters.iter().enumerate() {
                        if sh_rem_cap[si] <= 0.25 {
                            continue;
                        }
                        let p_seg = parcel_seg[pi];
                        let haul_s = dist_to_sh[si]
                            .get(p_seg)
                            .copied()
                            .unwrap_or(f64::INFINITY);
                        let haul_min = if haul_s.is_finite() {
                            (haul_s / 60.0).clamp(2.0, 50.0)
                        } else {
                            (1.42 * p_pos.dist(sh.world_pos) / van_v_m_min).clamp(4.0, 58.0)
                        };
                        let load_ada = ada_rem[pi].min(ada_van_cap as f64).min(sh_rem_cap[si]);
                        let (_, lift_dwell_min) = compute_ada_van_capacity_and_dwell_min(load_ada);
                        let dur_min = dh_min + lift_dwell_min + haul_min;
                        if t_now + dur_min > win_min {
                            continue;
                        }
                        let clean_mult = if cfg.dispatch_strategy
                            == TransitDispatchStrategy::FireUrgencyCleanAirTbep
                        {
                            if sh_clean_ok[si] {
                                4.2
                            } else {
                                0.25
                            }
                        } else {
                            1.0
                        };
                        let sc = (load_ada * urg_w * clean_mult) / dur_min.max(2.5);
                        if best_ada.map_or(true, |ba| sc > ba.4) {
                            best_ada = Some((pi, si, load_ada, dur_min, sc));
                        }
                    }
                }

                if let Some((pi, si, boarded, dur_min, _)) = best_ada {
                    // Also sweep up to remaining van capacity from immediately adjacent parcels (`<= 220 m`)!
                    let mut tot_boarded = boarded;
                    ada_rem[pi] = (ada_rem[pi] - boarded).max(0.0);
                    let wait_t = t_now + 0.5 * dur_min;
                    parcel_ada_cleared[pi] += boarded;
                    if sh_clean_ok[si] {
                        parcel_ada_clean[pi] += boarded;
                    }
                    parcel_ada_wait_sum[pi] += boarded * wait_t;

                    let rem_van_space = (ada_van_cap as f64 - tot_boarded).min(sh_rem_cap[si] - boarded);
                    if rem_van_space > 0.25 {
                        let mut left = rem_van_space;
                        for pj in 0..n_parcels {
                            if left <= 0.10 {
                                break;
                            }
                            if ada_rem[pj] > 0.08 && parcels[pj].center.dist(parcels[pi].center) <= 220.0 {
                                let take = ada_rem[pj].min(left);
                                ada_rem[pj] -= take;
                                left -= take;
                                tot_boarded += take;
                                parcel_ada_cleared[pj] += take;
                                if sh_clean_ok[si] {
                                    parcel_ada_clean[pj] += take;
                                }
                                parcel_ada_wait_sum[pj] += take * wait_t;
                            }
                        }
                    }

                    sh_rem_cap[si] = (sh_rem_cap[si] - tot_boarded).max(0.0);
                    let finish_min = t_now + dur_min + 1.2;
                    van_free_min[v_idx] = finish_min;
                    van_pos[v_idx] = cand_shelters[si].world_pos;
                    total_van_trips += 1.0;
                    van_deliveries.push((finish_min, tot_boarded, sh_clean_ok[si]));
                } else {
                    van_free_min[v_idx] = win_min + 10.0;
                }
            }

            // Verify inbound EMS hospital access preservation (% of fire/EMS facilities with finite <18 min inbound path to hospital/depots)
            let hosp_segs: Vec<usize> = facilities
                .iter()
                .filter(|f| f.kind == EmergencyFacilityKind::Hospital)
                .map(|f| f.segment_id)
                .collect();
            let ems_check_origin = hosp_segs
                .first()
                .copied()
                .or_else(|| sh_segs.first().copied())
                .unwrap_or(0);
            let ems_dist =
                dijkstra_from_seg(ems_check_origin, &seg_ems_in_trav_s, &seg_closed, &adj);
            let reachable_pp = pickup_depots
                .iter()
                .filter(|d| ems_dist.get(d.segment_id).copied().unwrap_or(f64::INFINITY) < 18.0 * 60.0)
                .count();
            let ems_ok_pct = if n_pp > 0 {
                100.0 * (reachable_pp as f64) / (n_pp as f64)
            } else {
                100.0
            };

            let carless_evac: f64 = pp_evac_any.iter().sum();
            let ada_evac: f64 = parcel_ada_cleared.iter().sum();
            let clean_evac: f64 =
                pp_evac_clean.iter().sum::<f64>() + parcel_ada_clean.iter().sum::<f64>();
            let tot_dem = (total_carless + total_ada).max(1e-6);
            let stranded = (tot_dem - (carless_evac + ada_evac)).max(0.0);

            let c_wait = if carless_evac > 1e-6 {
                pp_wait_pax_min.iter().sum::<f64>() / carless_evac
            } else {
                win_min
            };
            let a_wait = if ada_evac > 1e-6 {
                parcel_ada_wait_sum.iter().sum::<f64>() / ada_evac
            } else {
                win_min
            };

            // Per-parcel combined clearance % & wait min
            let mut p_clr = vec![100.0; n_parcels];
            let mut p_wt = vec![0.0; n_parcels];
            for pi in 0..n_parcels {
                let k = parcel_pp_idx[pi];
                let pp_frac = if n_pp > 0 && pp_dem[k] > 1e-6 {
                    (pp_evac_clean[k] / pp_dem[k]).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let ada_frac = if parcel_ada[pi] > 1e-6 {
                    (parcel_ada_clean[pi] / parcel_ada[pi]).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let walk_min = parcel_walk_m[pi] / 72.0; // 1.2 m/s pedestrian walk speed
                let dep_wt = if n_pp > 0 && pp_evac_any[k] > 1e-6 {
                    pp_wait_pax_min[k] / pp_evac_any[k]
                } else {
                    win_min
                };
                p_clr[pi] = 100.0
                    * ((parcel_carless[pi] * pp_frac + parcel_ada[pi] * ada_frac)
                        / (parcel_carless[pi] + parcel_ada[pi]).max(1e-6))
                    .clamp(0.0, 1.0);
                p_wt[pi] = walk_min + dep_wt;
            }

            // Build wave timeline points
            let mut w_car = vec![0.0; wave_steps];
            let mut w_ada = vec![0.0; wave_steps];
            let mut w_cln = vec![0.0; wave_steps];
            let mut w_bt = vec![0.0; wave_steps];
            let mut w_vt = vec![0.0; wave_steps];
            for ws in 0..wave_steps {
                let t_cut = (ws as f64 / (wave_steps - 1) as f64) * win_min;
                let bc: f64 = bus_deliveries
                    .iter()
                    .filter(|&&(tm, _, _)| tm <= t_cut + 1e-6)
                    .map(|&(_, px, _)| px)
                    .sum();
                let b_cln: f64 = bus_deliveries
                    .iter()
                    .filter(|&&(tm, _, cl)| tm <= t_cut + 1e-6 && cl)
                    .map(|&(_, px, _)| px)
                    .sum();
                let bt_n = bus_deliveries
                    .iter()
                    .filter(|&&(tm, _, _)| tm <= t_cut + 1e-6)
                    .count() as f64;

                let vc: f64 = van_deliveries
                    .iter()
                    .filter(|&&(tm, _, _)| tm <= t_cut + 1e-6)
                    .map(|&(_, px, _)| px)
                    .sum();
                let v_cln: f64 = van_deliveries
                    .iter()
                    .filter(|&&(tm, _, cl)| tm <= t_cut + 1e-6 && cl)
                    .map(|&(_, px, _)| px)
                    .sum();
                let vt_n = van_deliveries
                    .iter()
                    .filter(|&&(tm, _, _)| tm <= t_cut + 1e-6)
                    .count() as f64;

                w_car[ws] = 100.0 * (bc / total_carless.max(1e-6)).clamp(0.0, 1.0);
                w_ada[ws] = 100.0 * (vc / total_ada.max(1e-6)).clamp(0.0, 1.0);
                w_cln[ws] = 100.0 * ((b_cln + v_cln) / tot_dem).clamp(0.0, 1.0);
                w_bt[ws] = bt_n;
                w_vt[ws] = vt_n;
            }

            let mut pp_cyc = vec![0.0; n_pp];
            let mut pp_wt = vec![0.0; n_pp];
            let mut pp_str = vec![0.0; n_pp];
            for k in 0..n_pp {
                pp_cyc[k] = if pp_trip_cnt[k] > 0.0 {
                    pp_cycle_sum[k] / pp_trip_cnt[k]
                } else {
                    win_min
                };
                pp_wt[k] = if pp_evac_any[k] > 1e-6 {
                    pp_wait_pax_min[k] / pp_evac_any[k]
                } else {
                    win_min
                };
                pp_str[k] = pp_rem[k];
            }

            TrialAccum {
                carless_dem: total_carless,
                ada_dem: total_ada,
                carless_evac,
                ada_evac,
                clean_air_evac: clean_evac,
                stranded,
                bus_cycle_sum: if total_bus_trips > 0.0 {
                    total_bus_cycle_min / total_bus_trips
                } else {
                    win_min
                },
                bus_trips: total_bus_trips,
                van_trips: total_van_trips,
                carless_wait_sum: c_wait,
                ada_wait_sum: a_wait,
                ems_ok: ems_ok_pct,
                open_clean_sh: open_clean_cnt as f64,
                pp_dem,
                pp_clean: pp_evac_clean,
                pp_any: pp_evac_any,
                pp_strand: pp_str,
                pp_cycle: pp_cyc,
                pp_wait: pp_wt,
                pp_ffe_closed,
                seg_ffe_closed,
                parcel_clr: p_clr,
                parcel_wait: p_wt,
                wave_carless: w_car,
                wave_ada: w_ada,
                wave_clean: w_cln,
                wave_bus_trips: w_bt,
                wave_van_trips: w_vt,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let mean_carless_dem: f64 = trial_results.iter().map(|r| r.carless_dem).sum::<f64>() * inv_t;
    let mean_ada_dem: f64 = trial_results.iter().map(|r| r.ada_dem).sum::<f64>() * inv_t;
    let mean_carless_evac: f64 = trial_results.iter().map(|r| r.carless_evac).sum::<f64>() * inv_t;
    let mean_ada_evac: f64 = trial_results.iter().map(|r| r.ada_evac).sum::<f64>() * inv_t;
    let mean_clean_evac: f64 = trial_results.iter().map(|r| r.clean_air_evac).sum::<f64>() * inv_t;
    let mean_stranded: f64 = trial_results.iter().map(|r| r.stranded).sum::<f64>() * inv_t;

    let mut strand_sorted: Vec<f64> = trial_results.iter().map(|r| r.stranded).collect();
    strand_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p90_idx = (((trials as f64) * 0.90).floor() as usize).min(trials - 1);
    let p90_stranded = strand_sorted[p90_idx];

    let mean_bus_cycle = trial_results.iter().map(|r| r.bus_cycle_sum).sum::<f64>() * inv_t;
    let mean_bus_trips = trial_results.iter().map(|r| r.bus_trips).sum::<f64>() * inv_t;
    let mean_van_trips = trial_results.iter().map(|r| r.van_trips).sum::<f64>() * inv_t;
    let mean_c_wait = trial_results.iter().map(|r| r.carless_wait_sum).sum::<f64>() * inv_t;
    let mean_a_wait = trial_results.iter().map(|r| r.ada_wait_sum).sum::<f64>() * inv_t;
    let mean_ems_ok = trial_results.iter().map(|r| r.ems_ok).sum::<f64>() * inv_t;
    let mean_open_clean_sh =
        (trial_results.iter().map(|r| r.open_clean_sh).sum::<f64>() * inv_t).round() as usize;

    // Assemble Pickup Depot telemetry rows
    let mut depot_rows = Vec::with_capacity(n_pp);
    for (k, d) in pickup_depots.iter().enumerate() {
        let mut walk_sum = 0.0;
        let mut walk_w = 0.0;
        for pi in 0..n_parcels {
            if parcel_pp_idx[pi] == k {
                walk_sum += parcel_walk_m[pi] * parcel_carless_base[pi];
                walk_w += parcel_carless_base[pi];
            }
        }
        depot_rows.push(PickupDepotTelemetryRow {
            depot_id: k,
            gtfs_stop_id: d.gtfs_stop_id.clone(),
            name: d.name.clone(),
            segment_id: d.segment_id,
            world_pos: d.world_pos,
            serving_gtfs_routes: d.serving_gtfs_routes,
            mean_walk_dist_m: if walk_w > 0.0 {
                walk_sum / walk_w
            } else {
                180.0
            },
            carless_demand_pax: trial_results.iter().map(|r| r.pp_dem[k]).sum::<f64>() * inv_t,
            evacuated_clean_air_pax: trial_results.iter().map(|r| r.pp_clean[k]).sum::<f64>()
                * inv_t,
            evacuated_any_shelter_pax: trial_results.iter().map(|r| r.pp_any[k]).sum::<f64>()
                * inv_t,
            stranded_pax: trial_results.iter().map(|r| r.pp_strand[k]).sum::<f64>() * inv_t,
            mean_shuttle_cycle_min: trial_results.iter().map(|r| r.pp_cycle[k]).sum::<f64>()
                * inv_t,
            mean_wait_min: trial_results.iter().map(|r| r.pp_wait[k]).sum::<f64>() * inv_t,
            ffe_closure_prob_pct: 100.0
                * trial_results.iter().map(|r| r.pp_ffe_closed[k]).sum::<f64>()
                * inv_t,
        });
    }

    // Build representative Bus & ADA shuttle tours for viewport rendering
    let mut rep_tours = Vec::new();
    let open_sh_indices: Vec<usize> = (0..n_sh).filter(|&i| sh_selected[i]).collect();
    if !open_sh_indices.is_empty() {
        for (k, d_row) in depot_rows.iter().enumerate() {
            let best_si = open_sh_indices
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let da = cand_shelters[a].world_pos.dist(d_row.world_pos);
                    let db = cand_shelters[b].world_pos.dist(d_row.world_pos);
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(open_sh_indices[0]);
            let sh = &cand_shelters[best_si];
            rep_tours.push(BusShuttleTourRow {
                vehicle_label: format!("BUS-SHUTTLE-{:02}", k + 1),
                is_ada_paratransit: false,
                origin_depot_or_zone: d_row.name.clone(),
                dest_shelter_id: sh.shelter_id,
                dest_shelter_name: sh.name.clone(),
                origin_pos: d_row.world_pos,
                shelter_pos: sh.world_pos,
                trips_completed: (d_row.evacuated_any_shelter_pax / (coach_cap as f64)).max(0.5),
                cycle_time_min: d_row.mean_shuttle_cycle_min,
                total_pax_delivered: d_row.evacuated_any_shelter_pax,
                shelter_clean_air_ok: sh.archetype != ShelterFacilityArchetype::OpenParkTentPavilion
                    || cfg.shelters.shelter_backup_power_water_sanitation,
            });
        }
        // Add top ADA paratransit door-to-shelter arcs from high-density `R3`/`CBD` parcels
        let mut dense_parcels: Vec<usize> = (0..n_parcels).collect();
        dense_parcels.sort_by(|&a, &b| {
            parcel_ada_base[b]
                .partial_cmp(&parcel_ada_base[a])
                .unwrap_or(Ordering::Equal)
        });
        for (vi, &pi) in dense_parcels.iter().take(n_vans.min(4)).enumerate() {
            let p_pos = parcels[pi].center;
            let best_si = open_sh_indices
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    cand_shelters[a]
                        .world_pos
                        .dist(p_pos)
                        .partial_cmp(&cand_shelters[b].world_pos.dist(p_pos))
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap_or(open_sh_indices[0]);
            let sh = &cand_shelters[best_si];
            rep_tours.push(BusShuttleTourRow {
                vehicle_label: format!("ADA-VAN-{:02}", vi + 1),
                is_ada_paratransit: true,
                origin_depot_or_zone: format!("{:?} Block #{}", parcels[pi].zone, pi),
                dest_shelter_id: sh.shelter_id,
                dest_shelter_name: sh.name.clone(),
                origin_pos: p_pos,
                shelter_pos: sh.world_pos,
                trips_completed: (mean_van_trips / (n_vans as f64)).max(1.0),
                cycle_time_min: (mean_bus_cycle * 1.18).max(8.0),
                total_pax_delivered: mean_ada_evac / (n_vans.min(4) as f64),
                shelter_clean_air_ok: true,
            });
        }
    }

    let mut wave_timeline = Vec::with_capacity(wave_steps);
    for ws in 0..wave_steps {
        wave_timeline.push(TransitEvacWavePoint {
            minute: (ws as f64 / (wave_steps - 1) as f64) * win_min,
            cumul_carless_evacuated_pct: trial_results
                .iter()
                .map(|r| r.wave_carless[ws])
                .sum::<f64>()
                * inv_t,
            cumul_ada_evacuated_pct: trial_results.iter().map(|r| r.wave_ada[ws]).sum::<f64>()
                * inv_t,
            cumul_combined_clean_air_pct: trial_results
                .iter()
                .map(|r| r.wave_clean[ws])
                .sum::<f64>()
                * inv_t,
            active_bus_trips_completed: trial_results
                .iter()
                .map(|r| r.wave_bus_trips[ws])
                .sum::<f64>()
                * inv_t,
            active_ada_van_trips_completed: trial_results
                .iter()
                .map(|r| r.wave_van_trips[ws])
                .sum::<f64>()
                * inv_t,
        });
    }

    let mut parcel_cleared_pct = vec![100.0; n_parcels];
    let mut parcel_wait_min = vec![0.0; n_parcels];
    for pi in 0..n_parcels {
        parcel_cleared_pct[pi] =
            trial_results.iter().map(|r| r.parcel_clr[pi]).sum::<f64>() * inv_t;
        parcel_wait_min[pi] = trial_results.iter().map(|r| r.parcel_wait[pi]).sum::<f64>() * inv_t;
    }

    let mut contraflow_segments = Vec::with_capacity(n_seg);
    let mut bus_ems_cnt = 0usize;
    for seg in &net.segments {
        if bus_ems_lane[seg.id] {
            bus_ems_cnt += 1;
        }
        let ffe_prob =
            100.0 * trial_results.iter().map(|r| r.seg_ffe_closed[seg.id]).sum::<f64>() * inv_t;
        contraflow_segments.push(TransitContraflowSegmentRow {
            segment_id: seg.id,
            road_name: seg.name.clone(),
            base_lanes: seg.lane_count,
            bus_ems_priority_lane: bus_ems_lane[seg.id],
            contraflow_added_lanes: added_contraflow[seg.id],
            remaining_auto_lanes: remaining_auto[seg.id],
            ems_hospital_spine: ems_spine[seg.id],
            ffe_thermal_closure_prob_pct: ffe_prob,
        });
    }

    let tot_vuln = (mean_carless_dem + mean_ada_dem).max(1e-6);
    TransitEvacuationReport {
        config: cfg.clone(),
        active_pickup_depots_count: n_pp,
        open_clean_air_shelters_count: mean_open_clean_sh,
        bus_ems_contraflow_segments_count: bus_ems_cnt,
        total_carless_demand_pax: mean_carless_dem,
        total_ada_mobility_limited_pax: mean_ada_dem,
        evacuated_carless_pct: (100.0 * mean_carless_evac / mean_carless_dem.max(1e-6))
            .clamp(0.0, 100.0),
        evacuated_ada_pct: (100.0 * mean_ada_evac / mean_ada_dem.max(1e-6)).clamp(0.0, 100.0),
        combined_clean_air_sheltered_pct: (100.0 * mean_clean_evac / tot_vuln).clamp(0.0, 100.0),
        stranded_vulnerable_pax: mean_stranded,
        p90_stranded_vulnerable_pax: p90_stranded,
        mean_bus_shuttle_cycle_min: mean_bus_cycle,
        mean_trips_per_bus: mean_bus_trips / (n_buses as f64),
        mean_trips_per_ada_van: mean_van_trips / (n_vans as f64),
        mean_carless_wait_min: mean_c_wait,
        mean_ada_curb_wait_min: mean_a_wait,
        ems_inbound_access_preserved_pct: mean_ems_ok,
        depot_rows,
        representative_shuttle_tours: rep_tours,
        wave_timeline,
        contraflow_segments,
        parcel_cleared_pct,
        parcel_wait_min,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

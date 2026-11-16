// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! FEMA Incident Staging Bases (`ISB` -> County Staging Area `CSA` -> `C-POD` `53-ft` Tractor-Trailer
//! Supply Echelon), USACE / FEMA `IS-26` Typed Commodity Points of Distribution (`Type I / II / III`
//! Drive-Through + Pedestrian/GTFS Walk-Up `M/G/c` Throughput Queueing & Arterial Queue Spillback),
//! and FEMA `IPAWS` (`WEA 3.0` + `NPWS`/`PEP` `EAS` + `NOAA SAME`) Public `POD` Load-Balanced Routing
//!
//! Grounded in:
//! - FEMA / USACE *IS-26 Guide to Points of Distribution (`PODs`)* (`https://training.fema.gov/is/courseoverview.aspx?code=IS-26`;
//!   `https://sema.dps.mo.gov/docs/programs/LRMF/PODS.pdf`):
//!   - **Type I `C-POD`**: `4` vehicle lanes, `12` loading points (`125,000 sq ft`), `560 vehicles/hr`, `20,000 people/day`.
//!   - **Type II `C-POD`**: `2` vehicle lanes, `6` loading points (`75,000 sq ft`), `280 vehicles/hr`, `10,000 people/day`.
//!   - **Type III `C-POD`**: `1` vehicle lane, `3` loading points (`37,500 sq ft`), `140 vehicles/hr`, `5,000 people/day`.
//! - Bay Area UASI / FEMA NIMS Typed Drive-Through vs. Dedicated Pedestrian Walk-Up `POD` Doctrine
//!   (`https://uasi-prod.s3.us-west-1.amazonaws.com/s3fs-public/UASI+Resource+Library/Regional+Logistics+Response+Plan/Bay+Area+POD+Manual_February+2014.pdf`;
//!   `https://rtlt.preptoolkit.fema.gov/Public/Position/ViewFile/9-509-1329?type=Pdf&s=Name&p=6`).
//! - FEMA *Distribution Management Plan (`DMP`)* & *Incident Staging Base (`ISB`)* Echelon Logistics
//!   (`https://mil.wa.gov/asset/6ac7de07335e9/Distribution-Management-Plan-2024.pdf`;
//!   `https://storymaps.arcgis.com/stories/98b601bae955407d8dcae74c43578a30`).
//! - FEMA `IPAWS-OPEN` (`CAP v1.2` `WEA 3.0`, `NPWS`/`PEP` `EAS` AM/FM, `NOAA NWR SAME`) Public Queue
//!   & Inventory Broadcast Routing (`https://www.fema.gov/sites/default/files/documents/fema_ipaws-best-practices-guide.pdf`).

use crate::emergency::EmergencyFacility;
use crate::fema_comms::{combine_ipaws_pathway_reach_prob, compute_pep_am_and_nwr_vhf_reliability, FemaCommsConfig};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi,
};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const FEMA_POD_SITE_CSV_HEADER: &str = "pod_id,name,pod_type,lanes,nominal_veh_per_hr,nominal_pax_per_day,isb_trailers_received,peak_utilization_rho,mean_queue_wait_min,peak_spillback_vehs,stockout_dry_hours,served_residents_pax";
pub const FEMA_POD_HOURLY_CSV_HEADER: &str = "hour_of_shift,arrival_vehs_hr,served_vehs_hr,mean_wait_min,total_street_spillback_vehs,cumul_residents_served_pct,active_stocked_pods";

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
        u.clamp(1e-12, 1e0 - 1e-12)
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// USACE / FEMA `IS-26` & NIMS Typed Commodity Point of Distribution (`C-POD`) standard layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemaPodClassification {
    /// USACE Type I Metropolitan Super-POD (`4` lanes, `12` loading pts, `560 veh/hr`, `20,000 pax/day`, `125,000 sq ft`).
    TypeIMetroFourLane,
    /// USACE Type II Regional Arterial POD (`2` lanes, `6` loading pts, `280 veh/hr`, `10,000 pax/day`, `75,000 sq ft`).
    TypeIIRegionalTwoLane,
    /// USACE Type III Neighborhood POD (`1` lane, `3` loading pts, `140 veh/hr`, `5,000 pax/day`, `37,500 sq ft`).
    TypeIIILocalOneLane,
    /// NIMS Typed Pedestrian & Transit Walk-Up POD (`4` walk-up chutes, `420 walk-up hh/hr`, zero auto spillback!).
    PedestrianTransitWalkUp,
}

impl FemaPodClassification {
    pub fn badge(self) -> &'static str {
        match self {
            Self::TypeIMetroFourLane => "POD-T1",
            Self::TypeIIRegionalTwoLane => "POD-T2",
            Self::TypeIIILocalOneLane => "POD-T3",
            Self::PedestrianTransitWalkUp => "POD-PED",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::TypeIMetroFourLane => "Type I (4-Ln, 560 veh/h)",
            Self::TypeIIRegionalTwoLane => "Type II (2-Ln, 280 veh/h)",
            Self::TypeIIILocalOneLane => "Type III (1-Ln, 140 veh/h)",
            Self::PedestrianTransitWalkUp => "Pedestrian/Transit Walk-Up",
        }
    }

    /// Returns `(server_lanes_c, service_rate_per_lane_hr, nominal_daily_pax_12h, onsite_staging_lot_cap_vehs, day_staff_req)`.
    #[inline]
    pub fn usace_is26_specs(self) -> (usize, f64, usize, f64, usize) {
        match self {
            Self::TypeIMetroFourLane => (4, 140.0, 20_000, 96.0, 74),
            Self::TypeIIRegionalTwoLane => (2, 140.0, 10_000, 46.0, 34),
            Self::TypeIIILocalOneLane => (1, 140.0, 5_000, 22.0, 19),
            Self::PedestrianTransitWalkUp => (4, 105.0, 7_500, 180.0, 24),
        }
    }
}

/// Public `C-POD` assignment & `FEMA IPAWS` queue/inventory broadcast routing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IpaswPodRoutingPolicy {
    /// Uncoordinated word-of-mouth self-routing to geographically closest POD (severe CBD queue spillback & dry-run stockouts!).
    UncoordinatedNearestPodRush,
    /// Pre-published static county POD districts (blind to post-quake bridge closures, mid-shift stockouts & live lane queues).
    StaticPrePublishedDistricts,
    /// FEMA `IPAWS` (`WEA 3.0` + `EAS PEP` + `NOAA SAME`) live wait-time, stockout-diversion & odd/even arrival staggering!
    DynamicIpaswQueueAndStockBalanced,
}

impl Default for IpaswPodRoutingPolicy {
    fn default() -> Self {
        Self::DynamicIpaswQueueAndStockBalanced
    }
}

impl IpaswPodRoutingPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::UncoordinatedNearestPodRush => {
                "Uncoordinated Nearest-POD Rush (Severe Spillback & Dry Runs)"
            }
            Self::StaticPrePublishedDistricts => {
                "Static County POD Map Districts (Queue & Stockout Blind)"
            }
            Self::DynamicIpaswQueueAndStockBalanced => {
                "FEMA IPAWS Dynamic Wait/Stock Load-Balancing + Odd/Even Stagger"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::UncoordinatedNearestPodRush => "Nearest-Rush",
            Self::StaticPrePublishedDistricts => "Static-Districts",
            Self::DynamicIpaswQueueAndStockBalanced => "IPAWS-LoadBalanced",
        }
    }
}

/// FEMA `ISB -> CSA -> C-POD` `53-ft` tractor-trailer replenishment scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsbReplenishmentPolicy {
    /// Overnight-only static equal trailer push (standard IS-26 baseline; no mid-day surge restock when a POD runs dry!).
    OvernightStaticEqualPush,
    /// Demand-proportional `ISB` push + mid-day emergency hotshot `53-ft` trailer / `CH-47` sling restock via `ESF #2` `MERS`!
    DynamicIsbPullWithHotshotRestock,
}

impl Default for IsbReplenishmentPolicy {
    fn default() -> Self {
        Self::DynamicIsbPullWithHotshotRestock
    }
}

impl IsbReplenishmentPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::OvernightStaticEqualPush => "Overnight Static Equal Trailer Push (No Mid-Day Restock)",
            Self::DynamicIsbPullWithHotshotRestock => {
                "Demand-Weighted ISB Push + Mid-Day MERS Hotshot/CH-47 Restock"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::OvernightStaticEqualPush => "Overnight-Static",
            Self::DynamicIsbPullWithHotshotRestock => "Dynamic-ISB-Hotshot",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaIsbPodsConfig {
    /// Coupled FEMA `IPAWS` (`CAP v1.2`, `WEA 3.0`, `EAS PEP`, `NWR SAME`) & Region 10 `MERS` communications stack.
    pub fema_comms: FemaCommsConfig,
    pub routing_policy: IpaswPodRoutingPolicy,
    pub isb_policy: IsbReplenishmentPolicy,
    /// Number of county `C-POD` sites activated (`2..=8`, default `5`: mix of Type I, Type II, Type III & Pedestrian PODs).
    pub active_county_pods: usize,
    /// Daily `53-ft` commodity tractor-trailers (`Water + MRE` pairs; `1` trailer = `~1,350` household kits = `~4,050 pax-days`) staged at the FEMA `ISB`.
    pub isb_daily_53ft_trailers: usize,
    /// Share (`20..=100%`) of municipal population requiring daily `C-POD` water/MRE rations (coupled to buried pipe breaks).
    pub base_population_needing_pods_pct: f64,
    /// Upgrades top arterial Type II sites to `4-lane` `Type I` (`560 veh/hr`) Super-PODs with forklift MHE teams!
    pub upgrade_arterial_pods_to_type1: bool,
}

impl Default for FemaIsbPodsConfig {
    fn default() -> Self {
        Self {
            fema_comms: FemaCommsConfig::default(),
            routing_policy: IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced,
            isb_policy: IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock,
            active_county_pods: 5,
            isb_daily_53ft_trailers: 10,
            base_population_needing_pods_pct: 68.0,
            upgrade_arterial_pods_to_type1: true,
        }
    }
}

/// Evaluates `M/G/c` (Sakasegawa / Kingman Pollaczek-Khinchine + finite-shift fluid overload) steady/transient
/// queue length `L_q` (vehicles/households), mean queue wait `W_q_min` (minutes), and off-lot arterial street
/// spillback `spillback_vehs` over a `1.0-hour` time slice!
///
/// - `lanes_c`: number of parallel loading lanes (`1` Type III, `2` Type II, `4` Type I)
/// - `mu_per_lane_hr`: USACE IS-26 service rate per lane (`140.0 veh/hr`)
/// - `arrival_rate_hr`: arriving vehicles/hr during this hour (`lambda`)
/// - `carryover_queue`: unserved queue from previous hour
/// - `onsite_lot_cap`: off-street staging apron capacity (`vehs`) before spilling onto municipal roadway!
#[inline]
pub fn evaluate_usace_pod_mgc_hour_step(
    lanes_c: usize,
    mu_per_lane_hr: f64,
    arrival_rate_hr: f64,
    carryover_queue: f64,
    arrival_cv: f64,
    onsite_lot_cap: f64,
) -> (f64, f64, f64, f64, f64) {
    let c = lanes_c.max(1) as f64;
    let cap_hr = (c * mu_per_lane_hr.max(10.0)).max(10.0);
    let lam = arrival_rate_hr.max(0.0);
    let q0 = carryover_queue.max(0.0);
    let rho = lam / cap_hr;

    // Service time CV for 3-point palletized water/MRE trunk loading (`C_s = 0.38`)
    let cs2 = 0.38 * 0.38;
    let ca2 = arrival_cv.clamp(0.35, 1.8).powi(2);
    let kingman_factor = 0.5 * (ca2 + cs2);

    let (served_hr, q_end, mean_q) = if q0 + lam <= cap_hr * 0.94 {
        let rho_eff = ((lam + 0.5 * q0) / cap_hr).clamp(0.0, 0.94);
        let lq_ss = kingman_factor
            * (rho_eff.powf((2.0 * (c + 1.0)).sqrt()) / (1.0 - rho_eff).max(0.04));
        let served = q0 + lam;
        (served, lq_ss * 0.25, 0.5 * q0 + lq_ss)
    } else {
        // Near-saturation (`rho >= 0.94`) or oversaturated fluid surge (`lambda > c * mu`)!
        let served = (q0 + lam).min(cap_hr);
        let q1 = (q0 + lam - served).max(0.0);
        let stochastic_cushion = kingman_factor * c * 2.4;
        let q_avg = 0.5 * (q0 + q1) + stochastic_cushion;
        (served, q1, q_avg)
    };

    let service_min = 60.0 / mu_per_lane_hr.max(10.0);
    let wait_min = (60.0 * (mean_q / cap_hr) + service_min).clamp(service_min, 300.0);
    let spillback = (mean_q - onsite_lot_cap.max(5.0)).max(0.0);
    (served_hr, q_end, wait_min, spillback, rho)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSiteTelemetryRow {
    pub pod_id: usize,
    pub name: String,
    pub pod_class: FemaPodClassification,
    pub segment_id: usize,
    pub world_pos: Vec2,
    pub lanes: usize,
    pub nominal_veh_per_hr: f64,
    pub nominal_pax_per_day: usize,
    pub isb_trailers_received: f64,
    pub peak_utilization_rho: f64,
    pub mean_queue_wait_min: f64,
    pub peak_spillback_vehs: f64,
    pub stockout_dry_hours: f64,
    pub served_residents_pax: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PodShiftHourPoint {
    pub hour_of_shift: usize,
    pub arrival_vehs_hr: f64,
    pub served_vehs_hr: f64,
    pub mean_wait_min: f64,
    pub total_street_spillback_vehs: f64,
    pub cumul_residents_served_pct: f64,
    pub active_stocked_pods: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaIsbPodsReport {
    pub config: FemaIsbPodsConfig,
    pub isb_name: String,
    pub isb_world_pos: Vec2,
    pub csa_world_pos: Vec2,
    pub active_pods_count: usize,
    pub ipaws_informed_households_pct: f64,
    pub total_needy_residents_pax: f64,
    pub daily_served_residents_pax: f64,
    pub population_demand_met_pct: f64,
    pub mean_pod_queue_wait_min: f64,
    pub p90_pod_total_access_min: f64,
    pub peak_arterial_spillback_vehs: f64,
    pub wasted_dry_run_stockout_trips: f64,
    pub isb_bridge_detour_delay_min: f64,
    pub pod_rows: Vec<PodSiteTelemetryRow>,
    pub shift_timeline: Vec<PodShiftHourPoint>,
    pub isb_convoy_arcs: Vec<(Vec2, Vec2)>,
    pub household_pod_arcs: Vec<(Vec2, Vec2)>,
    pub segment_spillback_speed_factor: Vec<f64>,
    pub parcel_total_access_min: Vec<f64>,
    pub parcel_served_pct: Vec<f64>,
    pub solve_time_ms: f64,
}

impl FemaIsbPodsReport {
    pub fn to_pod_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.pod_rows.len() * 140);
        out.push_str(FEMA_POD_SITE_CSV_HEADER);
        out.push('\n');
        for p in &self.pod_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{},{:.1},{},{:.2},{:.2},{:.2},{:.1},{:.2},{:.1}\n",
                p.pod_id,
                p.name.replace('"', "'"),
                p.pod_class.badge(),
                p.lanes,
                p.nominal_veh_per_hr,
                p.nominal_pax_per_day,
                p.isb_trailers_received,
                p.peak_utilization_rho,
                p.mean_queue_wait_min,
                p.peak_spillback_vehs,
                p.stockout_dry_hours,
                p.served_residents_pax,
            ));
        }
        out
    }

    pub fn to_hourly_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.shift_timeline.len() * 96);
        out.push_str(FEMA_POD_HOURLY_CSV_HEADER);
        out.push('\n');
        for h in &self.shift_timeline {
            out.push_str(&format!(
                "{},{:.1},{:.1},{:.2},{:.1},{:.2},{:.2}\n",
                h.hour_of_shift,
                h.arrival_vehs_hr,
                h.served_vehs_hr,
                h.mean_wait_min,
                h.total_street_spillback_vehs,
                h.cumul_residents_served_pct,
                h.active_stocked_pods,
            ));
        }
        out
    }

    pub fn export_pod_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_pod_csv_string())?;
        Ok(self.pod_rows.len())
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
        if c.from_lane_id >= net.lanes.len() || c.to_lane_id >= net.lanes.len() {
            continue;
        }
        let u = net.lanes[c.from_lane_id].segment_id;
        let v = net.lanes[c.to_lane_id].segment_id;
        if u < n && v < n && u != v {
            let turn_s = match net.junctions[c.junction_id].control_mode {
                ControlMode::Roundabout => 2.5,
                ControlMode::MaxPressureAdaptive => 4.2,
                ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::PermissiveYield => match c.turn_type {
                    TurnType::Through => 4.5,
                    TurnType::Right => 5.5,
                    TurnType::Left => 8.0,
                },
            };
            if adj[u].iter().all(|&(nb, _)| nb != v) {
                adj[u].push((v, turn_s));
            }
        }
    }
    adj
}

fn dijkstra_from_seg(
    origin: usize,
    seg_trav_s: &[f64],
    seg_blocked: &[bool],
    adj: &[Vec<(usize, f64)>],
) -> Vec<f64> {
    let n = seg_trav_s.len();
    let mut dist = vec![f64::INFINITY; n];
    if origin >= n || seg_blocked[origin] {
        return dist;
    }
    let mut pq = BinaryHeap::new();
    dist[origin] = 0.5 * seg_trav_s[origin];
    pq.push(HeapState {
        cost_s: dist[origin],
        seg: origin,
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

fn nearest_open_seg(net: &RoadNetwork, target: Vec2) -> (usize, Vec2) {
    if let Some(seg) = net
        .segments
        .iter()
        .filter(|s| !s.closed && !s.base_pts.is_empty())
        .min_by(|a, b| {
            let da = a.base_pts[a.base_pts.len() / 2].dist(target);
            let db = b.base_pts[b.base_pts.len() / 2].dist(target);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        })
    {
        (seg.id, seg.base_pts[seg.base_pts.len() / 2])
    } else {
        (0, target)
    }
}

#[derive(Clone)]
struct SynthPodNode {
    pod_id: usize,
    name: String,
    pod_class: FemaPodClassification,
    segment_id: usize,
    world_pos: Vec2,
}

fn synthesize_isb_csa_and_county_pods(
    net: &RoadNetwork,
    cfg: &FemaIsbPodsConfig,
) -> ((usize, Vec2), (usize, Vec2), Vec<SynthPodNode>) {
    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(160.0);
    let sy = (b.max_y - b.min_y).max(160.0);

    // FEMA Incident Staging Base (`ISB` — regional airfield/highway staging terminal at SW perimeter)
    let isb = nearest_open_seg(
        net,
        Vec2 {
            x: b.min_x + 0.08 * sx,
            y: b.min_y + 0.90 * sy,
        },
    );
    // County Staging Area (`CSA` — fairgrounds/industrial trans-load hub)
    let csa = nearest_open_seg(
        net,
        Vec2 {
            x: b.min_x + 0.28 * sx,
            y: b.min_y + 0.68 * sy,
        },
    );

    let t_large = if cfg.upgrade_arterial_pods_to_type1 {
        FemaPodClassification::TypeIMetroFourLane
    } else {
        FemaPodClassification::TypeIIRegionalTwoLane
    };

    let templates = [
        (
            "Downtown Civic Walk-Up C-POD #1",
            FemaPodClassification::PedestrianTransitWalkUp,
            0.50,
            0.50,
        ),
        ("West Stadium Fairgrounds C-POD #2", t_large, 0.20, 0.44),
        (
            "North Shopping Center C-POD #3",
            FemaPodClassification::TypeIIRegionalTwoLane,
            0.45,
            0.18,
        ),
        ("Eastgate Arterial Super-POD #4", t_large, 0.82, 0.42),
        (
            "South Riverfront High School C-POD #5",
            FemaPodClassification::TypeIIILocalOneLane,
            0.52,
            0.82,
        ),
        (
            "Northwest Community Park C-POD #6",
            FemaPodClassification::TypeIIILocalOneLane,
            0.22,
            0.20,
        ),
        (
            "Southeast Industrial Lot C-POD #7",
            FemaPodClassification::TypeIIRegionalTwoLane,
            0.80,
            0.78,
        ),
        (
            "Midtown Transit Plaza Walk-Up C-POD #8",
            FemaPodClassification::PedestrianTransitWalkUp,
            0.62,
            0.56,
        ),
    ];

    let n_open = cfg.active_county_pods.clamp(2, templates.len());
    let mut pods = Vec::with_capacity(n_open);
    for (i, &(nm, cls, fx, fy)) in templates.iter().take(n_open).enumerate() {
        let (sid, wpos) = nearest_open_seg(
            net,
            Vec2 {
                x: b.min_x + fx * sx,
                y: b.min_y + fy * sy,
            },
        );
        pods.push(SynthPodNode {
            pod_id: i,
            name: nm.to_string(),
            pod_class: cls,
            segment_id: sid,
            world_pos: wpos,
        });
    }
    (isb, csa, pods)
}

/// Executes `Rayon`-parallelized Monte Carlo simulation of FEMA Incident Staging Bases (`ISB` -> `CSA`
/// -> `C-POD` `53-ft` Tractor-Trailer Echelons), USACE `IS-26` `Type I/II/III` & Pedestrian `M/G/c`
/// Queueing + Street Spillback, and `FEMA IPAWS` Dynamic Public `POD` Routing under `CRN`.
pub fn run_fema_isb_pods_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &FemaIsbPodsConfig,
) -> FemaIsbPodsReport {
    let _ = facilities;
    let t0 = Instant::now();
    let n_seg = net.segments.len();
    let n_parcels = parcels.len();
    let seis_cfg = &cfg
        .fema_comms
        .warning
        .transit_evac
        .shelters
        .evac
        .medevac
        .surge
        .ffe
        .seismic;
    let trials = seis_cfg.monte_carlo_trials.clamp(16, 5_000);

    let ((isb_seg, isb_pos), (csa_seg, csa_pos), pods) =
        synthesize_isb_csa_and_county_pods(net, cfg);
    let n_pods = pods.len();

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
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, seis_cfg);

    // Compute per-parcel household counts & carless share (`3.0` residents per household vehicle per USACE IS-26!).
    // Each parcel on a simplified OSM city grid represents a municipal TAZ catchment feeding countywide C-PODs
    // (`5,000-20,000 pax/day` USACE throughput & `1,400-kit` 53-ft ISB trailers), so scale synthetic small-grid counts
    // up to a realistic >=36,000-resident county catchment while leaving megacity parcel sets unscaled.
    let mut parcel_pop = vec![24.0f64; n_parcels];
    let mut parcel_carless_share = vec![0.12f64; n_parcels];
    let mut parcel_seg = vec![0usize; n_parcels];
    let mut raw_pop_sum = 0.0f64;
    for (pi, p) in parcels.iter().enumerate() {
        let (z_mult, c_sh) = match p.zone {
            ZoneCode::CBD => (1.35, 0.34),
            ZoneCode::R3 => (1.30, 0.26),
            ZoneCode::MU4 => (1.18, 0.20),
            ZoneCode::IND => (0.85, 0.10),
            ZoneCode::R1 => (1.05, 0.06),
            ZoneCode::PARK => (0.45, 0.12),
        };
        let area_m2 = p.area_ha * 10_000.0;
        let est_pop = p.residents() * 0.42 + p.jobs() * 0.55;
        let raw_p = ((16.0 + est_pop * 0.45 + (area_m2 / 480.0).min(28.0)) * z_mult).clamp(8.0, 88.0);
        parcel_pop[pi] = raw_p;
        raw_pop_sum += raw_p;
        parcel_carless_share[pi] = c_sh;
        parcel_seg[pi] = nearest_open_seg(net, p.center).0;
    }
    let county_catchment_scale = (36_000.0 / raw_pop_sum.max(1.0)).clamp(1.0, 24.0);
    for pop in &mut parcel_pop {
        *pop *= county_catchment_scale;
    }

    let base_sa10_g = crate::seismic_mc::median_sa10_gmpe_g(
        seis_cfg.moment_magnitude_mw,
        seis_cfg.rupture_standoff_km,
        seis_cfg.vs30_m_per_s,
    );

    // Uncoordinated panic morning rush profile vs. IPAWS Odd/Even staggered flat 12-hour shift profile!
    let rush_hour_weights: [f64; 12] = [
        0.165, 0.175, 0.145, 0.110, 0.085, 0.070, 0.060, 0.055, 0.045, 0.035, 0.030, 0.025,
    ];
    let staggered_hour_weights: [f64; 12] = [
        0.088, 0.092, 0.090, 0.088, 0.086, 0.084, 0.084, 0.082, 0.080, 0.078, 0.076, 0.072,
    ];

    // Each 53-ft commodity trailer pair carries ~1,400 3-person household daily kits (`4,200 resident-days`)
    let kits_per_53ft_trailer = 1_400.0f64;
    let total_isb_trailers = cfg.isb_daily_53ft_trailers.clamp(2, 60) as f64;
    let base_need_frac = (cfg.base_population_needing_pods_pct / 100.0).clamp(0.15, 1.0);
    let (med_batt_hr, pga_hard_mult, fiber_bonus) =
        cfg.fema_comms.warning.telecom_backup.resilience_params();

    #[derive(Clone)]
    struct TrialPodOut {
        ipaws_reach_pct: f64,
        needy_pax: f64,
        served_pax: f64,
        mean_wait_min: f64,
        p90_access_min: f64,
        peak_spillback_vehs: f64,
        dry_run_trips: f64,
        isb_detour_min: f64,
        pod_trailers: Vec<f64>,
        pod_peak_rho: Vec<f64>,
        pod_mean_wait: Vec<f64>,
        pod_peak_spill: Vec<f64>,
        pod_dry_hrs: Vec<f64>,
        pod_served_pax: Vec<f64>,
        hr_arr: Vec<f64>,
        hr_srv: Vec<f64>,
        hr_wait: Vec<f64>,
        hr_spill: Vec<f64>,
        hr_cum_pct: Vec<f64>,
        hr_stocked_pods: Vec<f64>,
        seg_spill_fac: Vec<f64>,
        parcel_acc_min: Vec<f64>,
        parcel_srv_pct: Vec<f64>,
    }

    let base_seed = seis_cfg.random_seed ^ 0x4953_4250_4F44_4101;
    let trial_outs: Vec<TrialPodOut> = (0..trials)
        .into_par_iter()
        .map(|tr_idx| {
            let mut rng = SplitMix64::new(
                base_seed.wrapping_add((tr_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let z_q = rng.next_standard_normal();
            let sa10_city = (base_sa10_g * (0.35 * z_q).exp()).clamp(0.03, 2.6);
            let pga_city =
                crate::seismic_mc::median_pga_g_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);
            let pgv_cm_s = median_pgv_cm_s_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);
            let pipe_rr = ala_hazus_pipe_repair_rate_per_km(
                pgv_cm_s,
                seis_cfg.pipe_fragility_law,
                seis_cfg.baseline_water_pipe_material,
                250.0,
                false,
            );

            // Bridge seismic closures
            let mut seg_closed = vec![false; n_seg];
            let mut closed_br_cnt = 0usize;
            for br in &bridges {
                if rng.next_f64_open01() < br.effective_archetype.exceedance_probs(sa10_city)[2] {
                    closed_br_cnt += 1;
                    for &sid in &br.segment_ids {
                        if sid < n_seg {
                            seg_closed[sid] = true;
                        }
                    }
                }
            }

            // Evaluate FEMA IPAWS public reach fraction in this earthquake realization!
            let p_grid_out = normal_cdf_phi((pga_city / 0.30).ln() / 0.50);
            let eoc_fiber_ok =
                closed_br_cnt == 0 && (rng.next_f64_open01() >= (1.0 - (-pipe_rr * 0.65).exp()));
            let eoc_cap_ok = match cfg.fema_comms.eoc_uplink {
                crate::fema_comms::EocIpaswUplinkMode::CommercialTerrestrialFiberOnly => {
                    eoc_fiber_ok
                }
                crate::fema_comms::EocIpaswUplinkMode::FiberPlusFnarsHfSkywaveRelay => {
                    eoc_fiber_ok || (rng.next_f64_open01() < 0.965)
                }
                crate::fema_comms::EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity => {
                    eoc_fiber_ok || (rng.next_f64_open01() < 0.994)
                }
            };
            let bts_s_ok = rng.next_f64_open01()
                >= normal_cdf_phi((pga_city / (0.64 * pga_hard_mult)).ln() / 0.52);
            let bts_f_ok = (closed_br_cnt == 0 || rng.next_f64_open01() < fiber_bonus)
                && (rng.next_f64_open01()
                    >= (1.0 - (-pipe_rr * 0.36 * (1.0 - fiber_bonus) * 1.6).exp()));
            let bts_batt_ok = (rng.next_f64_open01() >= p_grid_out)
                || (med_batt_hr * (0.35 * rng.next_standard_normal()).exp()
                    >= cfg.fema_comms.warning.hours_post_mainshock);
            let p_wea = if bts_s_ok && bts_f_ok && bts_batt_ok {
                0.88
            } else if cfg.fema_comms.posture
                == crate::fema_comms::FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment
                && cfg.fema_comms.fema_mers_irv_units > 0
            {
                0.55
            } else {
                0.0
            };
            let (pep_rf, nwr_rf) = compute_pep_am_and_nwr_vhf_reliability(32.0);
            let (p_eas, p_nwr) = if cfg.fema_comms.posture
                == crate::fema_comms::FemaCommsPosture::IpaswWea30CellularOnly
            {
                (0.0, 0.0)
            } else {
                (
                    ((cfg.fema_comms.household_eas_am_fm_radio_pct / 100.0) * 0.82 * pep_rf
                        + 0.14)
                        .clamp(0.0, 0.92),
                    ((cfg.fema_comms.household_nwr_same_receiver_pct / 100.0) * 0.94 * nwr_rf)
                        .clamp(0.0, 0.92),
                )
            };
            let ipaws_reach_f = combine_ipaws_pathway_reach_prob(eoc_cap_ok, p_wea, p_eas, p_nwr);

            // Road segment free-flow travel times & ISB -> CSA -> POD convoy haul times
            let mut seg_trav_s = vec![10.0; n_seg];
            for seg in &net.segments {
                if seg.closed {
                    seg_closed[seg.id] = true;
                }
                seg_trav_s[seg.id] = (seg.length / seg.speed_mps.max(4.0)).max(1.2);
            }
            let isb_to_all = dijkstra_from_seg(isb_seg, &seg_trav_s, &seg_closed, &adj);
            let _ = csa_seg;

            let mut dist_to_pod = vec![vec![f64::INFINITY; n_seg]; n_pods];
            let mut pod_isb_haul_min = vec![12.0; n_pods];
            for (j, pod) in pods.iter().enumerate() {
                dist_to_pod[j] = dijkstra_from_seg(pod.segment_id, &seg_trav_s, &seg_closed, &rev_adj);
                let hs = isb_to_all.get(pod.segment_id).copied().unwrap_or(f64::INFINITY);
                pod_isb_haul_min[j] = if hs.is_finite() {
                    (hs / 60.0).clamp(3.0, 45.0)
                } else {
                    // Bridge collapse forces circuitous county detour (+22 min) unless CH-47 sling restock
                    26.0
                };
            }
            let mean_isb_detour_min =
                (pod_isb_haul_min.iter().sum::<f64>() / (n_pods.max(1) as f64) - 6.0).max(0.0);

            // Household commodity demand (`1 household trip = 3.0 residents`), amplified by water-main break severity!
            let water_outage_boost = (1.0 - (-pipe_rr * 1.4).exp()) * 0.24;
            let eff_need_frac = (base_need_frac + water_outage_boost).clamp(0.20, 0.98);

            let mut parcel_hh_trips = vec![0.0; n_parcels];
            let mut total_needy_pax = 0.0f64;
            for pi in 0..n_parcels {
                let jitter = (0.90 + 0.20 * rng.next_f64_open01()).clamp(0.75, 1.25);
                let needy_pax = parcel_pop[pi] * eff_need_frac * jitter;
                total_needy_pax += needy_pax;
                parcel_hh_trips[pi] = (needy_pax / 3.0).max(0.5);
            }

            // Overnight ISB 53-ft Trailer Push Allocation (`pod_stock_kits[j]`)
            let mut pod_stock_kits = vec![0.0; n_pods];
            let mut pod_trailers_used = vec![0.0; n_pods];
            let mut reserve_hotshot_trailers = 0.0f64;

            match cfg.isb_policy {
                IsbReplenishmentPolicy::OvernightStaticEqualPush => {
                    let per_pod_tr = total_isb_trailers / (n_pods as f64);
                    for j in 0..n_pods {
                        // Bridge-isolated PODs lose 35% of static overnight convoy deliveries!
                        let deliv_eff = if pod_isb_haul_min[j] > 24.0 { 0.65 } else { 1.0 };
                        pod_trailers_used[j] = per_pod_tr * deliv_eff;
                        pod_stock_kits[j] = pod_trailers_used[j] * kits_per_53ft_trailer;
                    }
                }
                IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock => {
                    // Hold 25% of ISB 53-ft trailers in rapid mid-day MERS-dispatched hotshot reserve,
                    // and allocate 75% proportional to each POD's USACE nominal lane throughput!
                    let overnight_tr = total_isb_trailers * 0.75;
                    reserve_hotshot_trailers = total_isb_trailers * 0.25;
                    let cap_sum: f64 = pods
                        .iter()
                        .map(|pd| pd.pod_class.usace_is26_specs().2 as f64)
                        .sum::<f64>()
                        .max(1.0);
                    for (j, pd) in pods.iter().enumerate() {
                        let w = (pd.pod_class.usace_is26_specs().2 as f64) / cap_sum;
                        pod_trailers_used[j] = overnight_tr * w;
                        pod_stock_kits[j] = pod_trailers_used[j] * kits_per_53ft_trailer;
                    }
                }
            }

            // Simulate 12-Hour Daytime C-POD Operating Shift (`Hours 1..=12`, 07:00 to 19:00)!
            let mut pod_queue = vec![0.0f64; n_pods];
            let mut pod_live_wait_est = vec![4.0f64; n_pods];
            let mut pod_peak_rho = vec![0.0f64; n_pods];
            let mut pod_wait_weighted = vec![0.0f64; n_pods];
            let mut pod_served_hh = vec![0.0f64; n_pods];
            let mut pod_peak_spill = vec![0.0f64; n_pods];
            let mut pod_dry_hrs = vec![0.0f64; n_pods];
            let mut dry_run_trips = 0.0f64;

            let mut hr_arr = vec![0.0; 12];
            let mut hr_srv = vec![0.0; 12];
            let mut hr_wait = vec![0.0; 12];
            let mut hr_spill = vec![0.0; 12];
            let mut hr_cum_pct = vec![0.0; 12];
            let mut hr_stocked = vec![0.0; 12];

            let mut parcel_wait_accum = vec![0.0f64; n_parcels];
            let mut parcel_served_hh = vec![0.0f64; n_parcels];
            let mut cumul_served_pax = 0.0f64;

            for h in 0..12 {
                // Mid-shift hotshot ISB 53-ft trailer / CH-47 sling restock before inventory hits zero!
                if cfg.isb_policy == IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock
                    && reserve_hotshot_trailers >= 0.25
                {
                    for j in 0..n_pods {
                        let (lanes_c, mu_0, _, _, _) = pods[j].pod_class.usace_is26_specs();
                        let one_hr_burn = (lanes_c as f64) * mu_0 * 0.85;
                        if pod_stock_kits[j] < one_hr_burn && reserve_hotshot_trailers >= 0.25 {
                            let add_tr = reserve_hotshot_trailers.min(1.0);
                            reserve_hotshot_trailers -= add_tr;
                            pod_trailers_used[j] += add_tr;
                            pod_stock_kits[j] += add_tr * kits_per_53ft_trailer;
                        }
                    }
                }

                // Route each parcel's hourly household demand across `pods[0..n_pods]`!
                let mut pod_arr_h = vec![0.0f64; n_pods];
                let mut parcel_pod_share = vec![vec![0.0f64; n_pods]; n_parcels];

                for pi in 0..n_parcels {
                    let informed_sh = match cfg.routing_policy {
                        IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced => ipaws_reach_f,
                        _ => 0.0,
                    };
                    let w_h = informed_sh * staggered_hour_weights[h]
                        + (1.0 - informed_sh) * rush_hour_weights[h];
                    let hh_h = parcel_hh_trips[pi] * w_h;
                    let p_pos = parcels[pi].center;
                    let p_seg = parcel_seg[pi];

                    // Nearest POD (uninformed / nearest-rush behavior)
                    let nearest_j = (0..n_pods)
                        .min_by(|&a, &b| {
                            p_pos
                                .dist(pods[a].world_pos)
                                .partial_cmp(&p_pos.dist(pods[b].world_pos))
                                .unwrap_or(Ordering::Equal)
                        })
                        .unwrap_or(0);

                    match cfg.routing_policy {
                        IpaswPodRoutingPolicy::UncoordinatedNearestPodRush => {
                            pod_arr_h[nearest_j] += hh_h;
                            parcel_pod_share[pi][nearest_j] = hh_h;
                        }
                        IpaswPodRoutingPolicy::StaticPrePublishedDistricts => {
                            // Fixed geographic Voronoi / modulo sector assignment (ignores live queues & stockouts)
                            let static_j = (pi + nearest_j) % n_pods;
                            pod_arr_h[static_j] += hh_h;
                            parcel_pod_share[pi][static_j] = hh_h;
                        }
                        IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced => {
                            // Uninformed fraction `(1 - informed_sh)` still goes to `nearest_j`;
                            // IPAWS-informed fraction `informed_sh` routes via Logit Generalized-Cost across stocked PODs!
                            let uninf_hh = hh_h * (1.0 - informed_sh);
                            let inf_hh = hh_h * informed_sh;
                            if uninf_hh > 0.0 {
                                pod_arr_h[nearest_j] += uninf_hh;
                                parcel_pod_share[pi][nearest_j] += uninf_hh;
                            }
                            if inf_hh > 0.0 {
                                let mut weights = vec![0.0f64; n_pods];
                                let mut w_sum = 0.0f64;
                                for (j, pd) in pods.iter().enumerate() {
                                    let ts = dist_to_pod[j]
                                        .get(p_seg)
                                        .copied()
                                        .unwrap_or(f64::INFINITY);
                                    let drive_m = if ts.is_finite() {
                                        (ts / 60.0).clamp(1.5, 40.0)
                                    } else {
                                        1.45 * p_pos.dist(pd.world_pos) / 380.0
                                    };
                                    // Match carless CBD/R3 households preferentially to Pedestrian Walk-Up PODs!
                                    let mode_fit_pen = if pd.pod_class
                                        == FemaPodClassification::PedestrianTransitWalkUp
                                    {
                                        if parcel_carless_share[pi] >= 0.20 {
                                            -4.5
                                        } else {
                                            8.5
                                        }
                                    } else {
                                        0.0
                                    };
                                    let stock_pen = if pod_stock_kits[j] <= 15.0 {
                                        140.0 // IPAWS warns public away from stocked-out PODs!
                                    } else if pod_stock_kits[j] < 180.0 {
                                        22.0
                                    } else {
                                        0.0
                                    };
                                    let (lanes_c, _, _, _, _) = pd.pod_class.usace_is26_specs();
                                    let cap_bonus = -1.8 * (lanes_c as f64);
                                    let gen_cost = drive_m
                                        + 0.92 * pod_live_wait_est[j]
                                        + stock_pen
                                        + mode_fit_pen
                                        + cap_bonus;
                                    let lw = (-0.14 * gen_cost).exp().max(1e-8);
                                    weights[j] = lw;
                                    w_sum += lw;
                                }
                                for j in 0..n_pods {
                                    let alloc = inf_hh * (weights[j] / w_sum.max(1e-9));
                                    pod_arr_h[j] += alloc;
                                    parcel_pod_share[pi][j] += alloc;
                                }
                            }
                        }
                    }
                }

                // Evaluate M/G/c queueing + inventory depletion at every C-POD during Hour `h`!
                let arr_cv = match cfg.routing_policy {
                    IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced => {
                        1.0 - 0.28 * ipaws_reach_f
                    }
                    IpaswPodRoutingPolicy::StaticPrePublishedDistricts => 1.05,
                    IpaswPodRoutingPolicy::UncoordinatedNearestPodRush => 1.28,
                };

                let mut tot_arr_h = 0.0;
                let mut tot_srv_h = 0.0;
                let mut wait_num_h = 0.0;
                let mut tot_spill_h = 0.0;
                let mut stocked_cnt_h = 0.0;
                let mut pod_wait_this_h = vec![0.0; n_pods];
                let mut pod_served_frac_h = vec![1.0; n_pods];

                for (j, pd) in pods.iter().enumerate() {
                    let (lanes_c, mu_0, _, lot_cap, _) = pd.pod_class.usace_is26_specs();
                    let (raw_srv, q_next, w_min, spill_v, rho) = evaluate_usace_pod_mgc_hour_step(
                        lanes_c,
                        mu_0,
                        pod_arr_h[j],
                        pod_queue[j],
                        arr_cv,
                        lot_cap,
                    );

                    // Enforce physical palletized kit inventory!
                    let actual_srv = raw_srv.min(pod_stock_kits[j]);
                    if pod_stock_kits[j] > 20.0 {
                        stocked_cnt_h += 1.0;
                    }
                    if raw_srv > actual_srv + 1.0 {
                        let dry = raw_srv - actual_srv;
                        dry_run_trips += dry;
                        pod_dry_hrs[j] += (dry / raw_srv.max(1.0)).clamp(0.0, 1.0);
                    }
                    pod_stock_kits[j] = (pod_stock_kits[j] - actual_srv).max(0.0);

                    // Pedestrian Walk-Up POD queues on plazas/sidewalks rather than vehicular lanes!
                    let veh_spill = if pd.pod_class == FemaPodClassification::PedestrianTransitWalkUp
                    {
                        0.0
                    } else {
                        spill_v
                    };

                    pod_queue[j] = if pod_stock_kits[j] <= 1.0 {
                        // Unserved drivers turn away after discovering stockout (`+18 min` wasted queue penalty)
                        q_next * 0.30
                    } else {
                        q_next
                    };
                    pod_live_wait_est[j] = w_min;
                    pod_peak_rho[j] = pod_peak_rho[j].max(rho);
                    pod_peak_spill[j] = pod_peak_spill[j].max(veh_spill);
                    pod_served_hh[j] += actual_srv;
                    pod_wait_weighted[j] += w_min * actual_srv.max(1.0);
                    pod_wait_this_h[j] = w_min;
                    pod_served_frac_h[j] = if pod_arr_h[j] + pod_queue[j] > 1e-6 {
                        (actual_srv / raw_srv.max(1e-6)).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };

                    tot_arr_h += pod_arr_h[j];
                    tot_srv_h += actual_srv;
                    wait_num_h += w_min * pod_arr_h[j].max(0.5);
                    tot_spill_h += veh_spill;
                }

                // Attribute hourly wait + travel + served Kits back to originating parcels
                for pi in 0..n_parcels {
                    for j in 0..n_pods {
                        let sh = parcel_pod_share[pi][j];
                        if sh > 1e-6 {
                            let ts = dist_to_pod[j]
                                .get(parcel_seg[pi])
                                .copied()
                                .unwrap_or(360.0);
                            let drv_m = if ts.is_finite() {
                                (ts / 60.0).clamp(1.5, 35.0)
                            } else {
                                18.0
                            };
                            parcel_wait_accum[pi] += sh * (drv_m + pod_wait_this_h[j]);
                            parcel_served_hh[pi] += sh * pod_served_frac_h[j];
                        }
                    }
                }

                cumul_served_pax += tot_srv_h * 3.0;
                hr_arr[h] = tot_arr_h;
                hr_srv[h] = tot_srv_h;
                hr_wait[h] = wait_num_h / tot_arr_h.max(1.0);
                hr_spill[h] = tot_spill_h;
                hr_cum_pct[h] = 100.0 * (cumul_served_pax / total_needy_pax.max(1.0)).clamp(0.0, 1.0);
                hr_stocked[h] = stocked_cnt_h;
            }

            // Compute street spillback speed degradation factors `seg_spill_fac` on segments adjacent to congested PODs
            let mut seg_spill_fac = vec![1.0_f64; n_seg];
            for (j, pd) in pods.iter().enumerate() {
                if pod_peak_spill[j] > 1.0 && pd.segment_id < n_seg {
                    let drop: f64 = (1.0 / (1.0 + 0.022 * pod_peak_spill[j])).clamp(0.28, 1.0);
                    seg_spill_fac[pd.segment_id] = seg_spill_fac[pd.segment_id].min(drop);
                    let j_in = net.segments[pd.segment_id].from_junc;
                    for &in_s in &net.junctions[j_in].in_segs {
                        if in_s < n_seg {
                            seg_spill_fac[in_s] =
                                seg_spill_fac[in_s].min((0.5 * (1.0 + drop)).clamp(0.40, 1.0));
                        }
                    }
                }
            }

            let mut parcel_acc_min = vec![10.0; n_parcels];
            let mut parcel_srv_pct = vec![100.0; n_parcels];
            for pi in 0..n_parcels {
                parcel_acc_min[pi] = parcel_wait_accum[pi] / parcel_hh_trips[pi].max(1e-6);
                parcel_srv_pct[pi] = 100.0
                    * (parcel_served_hh[pi] / parcel_hh_trips[pi].max(1e-6)).clamp(0.0, 1.0);
            }
            let mut acc_sorted = parcel_acc_min.clone();
            acc_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let p90_access_min = acc_sorted
                .get(((n_parcels as f64 * 0.90).floor() as usize).min(n_parcels.saturating_sub(1)))
                .copied()
                .unwrap_or(20.0);

            let mut pod_mean_wait = vec![0.0; n_pods];
            let mut pod_served_pax = vec![0.0; n_pods];
            for j in 0..n_pods {
                pod_mean_wait[j] = pod_wait_weighted[j] / pod_served_hh[j].max(1.0);
                pod_served_pax[j] = pod_served_hh[j] * 3.0;
            }
            let tot_served_hh: f64 = pod_served_hh.iter().sum::<f64>().max(1.0);
            let mean_wait_min: f64 = pod_wait_weighted.iter().sum::<f64>() / tot_served_hh;
            let peak_spillback_vehs = hr_spill.iter().copied().fold(0.0f64, f64::max);

            TrialPodOut {
                ipaws_reach_pct: 100.0 * ipaws_reach_f,
                needy_pax: total_needy_pax,
                served_pax: cumul_served_pax.min(total_needy_pax),
                mean_wait_min,
                p90_access_min,
                peak_spillback_vehs,
                dry_run_trips,
                isb_detour_min: mean_isb_detour_min,
                pod_trailers: pod_trailers_used,
                pod_peak_rho,
                pod_mean_wait,
                pod_peak_spill,
                pod_dry_hrs,
                pod_served_pax,
                hr_arr,
                hr_srv,
                hr_wait,
                hr_spill,
                hr_cum_pct,
                hr_stocked_pods: hr_stocked,
                seg_spill_fac,
                parcel_acc_min,
                parcel_srv_pct,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let mean_needy = trial_outs.iter().map(|o| o.needy_pax).sum::<f64>() * inv_t;
    let mean_served = trial_outs.iter().map(|o| o.served_pax).sum::<f64>() * inv_t;

    let mut pod_rows = Vec::with_capacity(n_pods);
    let mut isb_convoy_arcs = Vec::with_capacity(n_pods + 1);
    isb_convoy_arcs.push((isb_pos, csa_pos));
    for (j, pd) in pods.iter().enumerate() {
        let (lanes_c, mu_0, nom_pax, _, _) = pd.pod_class.usace_is26_specs();
        pod_rows.push(PodSiteTelemetryRow {
            pod_id: j,
            name: pd.name.clone(),
            pod_class: pd.pod_class,
            segment_id: pd.segment_id,
            world_pos: pd.world_pos,
            lanes: lanes_c,
            nominal_veh_per_hr: (lanes_c as f64) * mu_0,
            nominal_pax_per_day: nom_pax,
            isb_trailers_received: trial_outs.iter().map(|o| o.pod_trailers[j]).sum::<f64>()
                * inv_t,
            peak_utilization_rho: trial_outs.iter().map(|o| o.pod_peak_rho[j]).sum::<f64>()
                * inv_t,
            mean_queue_wait_min: trial_outs.iter().map(|o| o.pod_mean_wait[j]).sum::<f64>() * inv_t,
            peak_spillback_vehs: trial_outs.iter().map(|o| o.pod_peak_spill[j]).sum::<f64>()
                * inv_t,
            stockout_dry_hours: trial_outs.iter().map(|o| o.pod_dry_hrs[j]).sum::<f64>() * inv_t,
            served_residents_pax: trial_outs.iter().map(|o| o.pod_served_pax[j]).sum::<f64>()
                * inv_t,
        });
        isb_convoy_arcs.push((csa_pos, pd.world_pos));
    }

    let mut shift_timeline = Vec::with_capacity(12);
    for h in 0..12 {
        shift_timeline.push(PodShiftHourPoint {
            hour_of_shift: h + 1,
            arrival_vehs_hr: trial_outs.iter().map(|o| o.hr_arr[h]).sum::<f64>() * inv_t,
            served_vehs_hr: trial_outs.iter().map(|o| o.hr_srv[h]).sum::<f64>() * inv_t,
            mean_wait_min: trial_outs.iter().map(|o| o.hr_wait[h]).sum::<f64>() * inv_t,
            total_street_spillback_vehs: trial_outs.iter().map(|o| o.hr_spill[h]).sum::<f64>()
                * inv_t,
            cumul_residents_served_pct: trial_outs.iter().map(|o| o.hr_cum_pct[h]).sum::<f64>()
                * inv_t,
            active_stocked_pods: trial_outs.iter().map(|o| o.hr_stocked_pods[h]).sum::<f64>()
                * inv_t,
        });
    }

    let mut segment_spillback_speed_factor = vec![1.0; n_seg];
    for s in 0..n_seg {
        segment_spillback_speed_factor[s] =
            trial_outs.iter().map(|o| o.seg_spill_fac[s]).sum::<f64>() * inv_t;
    }

    let mut parcel_total_access_min = vec![10.0; n_parcels];
    let mut parcel_served_pct = vec![100.0; n_parcels];
    let mut household_pod_arcs = Vec::new();
    for pi in 0..n_parcels {
        parcel_total_access_min[pi] =
            trial_outs.iter().map(|o| o.parcel_acc_min[pi]).sum::<f64>() * inv_t;
        parcel_served_pct[pi] =
            trial_outs.iter().map(|o| o.parcel_srv_pct[pi]).sum::<f64>() * inv_t;
        if pi % 6 == 0 && !pods.is_empty() {
            let best_j = (0..n_pods)
                .min_by(|&a, &b| {
                    let ca = parcels[pi].center.dist(pods[a].world_pos) / 350.0
                        + pod_rows[a].mean_queue_wait_min;
                    let cb = parcels[pi].center.dist(pods[b].world_pos) / 350.0
                        + pod_rows[b].mean_queue_wait_min;
                    ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0);
            household_pod_arcs.push((parcels[pi].center, pods[best_j].world_pos));
        }
    }

    FemaIsbPodsReport {
        config: cfg.clone(),
        isb_name: "FEMA Incident Staging Base (ISB-Region10 South Airfield)".to_string(),
        isb_world_pos: isb_pos,
        csa_world_pos: csa_pos,
        active_pods_count: n_pods,
        ipaws_informed_households_pct: trial_outs.iter().map(|o| o.ipaws_reach_pct).sum::<f64>()
            * inv_t,
        total_needy_residents_pax: mean_needy,
        daily_served_residents_pax: mean_served,
        population_demand_met_pct: (100.0 * mean_served / mean_needy.max(1.0)).clamp(0.0, 100.0),
        mean_pod_queue_wait_min: trial_outs.iter().map(|o| o.mean_wait_min).sum::<f64>() * inv_t,
        p90_pod_total_access_min: trial_outs.iter().map(|o| o.p90_access_min).sum::<f64>() * inv_t,
        peak_arterial_spillback_vehs: trial_outs
            .iter()
            .map(|o| o.peak_spillback_vehs)
            .sum::<f64>()
            * inv_t,
        wasted_dry_run_stockout_trips: trial_outs.iter().map(|o| o.dry_run_trips).sum::<f64>()
            * inv_t,
        isb_bridge_detour_delay_min: trial_outs.iter().map(|o| o.isb_detour_min).sum::<f64>()
            * inv_t,
        pod_rows,
        shift_timeline,
        isb_convoy_arcs,
        household_pod_arcs,
        segment_spillback_speed_factor,
        parcel_total_access_min,
        parcel_served_pct,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

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

//! FEMA National Urban Search & Rescue (`US&R` / `ESF #9`) Type 1 Task Force (`80`-Person Dual-Shift
//! `24-hr` Operations, `USNG` Structural Search Grids & Coburn-Spence `72-hr` Entrapped Survival),
//! `ESF #3` Heavy-Equipment Phase I Roadway Debris Push (`D7/D8` Bulldozers & Loaders), and Joint
//! `US&R` + `FEMA ISB -> CSA -> C-POD` Corridor Reopening Optimization
//!
//! Grounded in:
//! - FEMA / USFA *Urban Search and Rescue Field Operations Guide* (`https://www.usfa.fema.gov/downloads/pdf/publications/field_operations_guide.pdf`),
//!   FIRESCOPE *US&R ICS-USR-120-1* (`https://firescope.caloes.ca.gov/ICS%20Documents/ICS-USR-120-1.pdf`),
//!   and FEMA Type 1 `80`-Member / `16,400`-Piece `72-Hr` Self-Sufficient Cache Operations
//!   (`https://www.fireengineering.com/technical-rescue/new-york-task-force-1-deploys-to-hurricanes-harvey-irma-and-maria/`).
//! - Macintyre, A. G., Barbera, J. A., & Smith, E. R. (2006 / 2011), *Survival Interval in Earthquake
//!   Entrapments* (`https://www.researchgate.net/publication/50226590_Survival_Interval_in_Earthquake_Entrapments_Research_Findings_Reinforced_During_the_2010_Haiti_Earthquake_Response`)
//!   and Hooshangi et al. (2024), *Urban Search and Rescue (`USAR`) Simulation in Earthquake Environments
//!   Using Queuing Theory* (`https://www.emerald.com/ijdrbe/article/15/1/1/1220157/Urban-search-and-rescue-USAR-simulation-in`).
//! - Caunhye, A. M. et al. (2020), *Robust Post-Disaster Route Restoration ("Teeth of a Comb" Strategy)*,
//!   *Journal of the Operational Research Society* (`https://www.pure.ed.ac.uk/ws/files/161495448/CaunhyeORS2020RobustPostDisasterRoute.pdf`),
//!   Sahin et al. (2016), *The Post-Disaster Debris Clearance Problem*
//!   (`https://www.researchgate.net/publication/273519839_The_Post-Disaster_Debris_Clearance_Problem_Under_Incomplete_Information`),
//!   and Yan et al., *Highway Emergency Rehabilitation Scheduling in Post-Earthquake 72 Hours*
//!   (`https://www.researchgate.net/publication/228602456_Highway_emergency_rehabilitation_scheduling_in_post-earthquake_72_hours`).

use crate::emergency::EmergencyFacility;
use crate::fema_isb_pods::FemaIsbPodsConfig;
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{enumerate_seismic_bridge_inventory, normal_cdf_phi};
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

pub const FEMA_USNG_GRID_CSV_HEADER: &str = "grid_id,usng_cell_code,dominant_collapse_archetype,debris_tons,initial_entrapped_victims,road_access_reopened_hr,fema_marking_status,live_rescued_victims,deceased_entrapment_losses,live_survival_rate_pct";
pub const FEMA_DEBRIS_TIMELINE_CSV_HEADER: &str = "hour_post_quake,road_corridors_open_pct,isb_to_pods_connected_pct,usng_grids_searched_pct,cumul_live_rescues,remaining_alive_in_rubble";

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

/// Structural collapse archetype governing Hazus rubble tonnage, void ratio, and Coburn-Spence survival decay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralCollapseArchetype {
    /// Unreinforced Masonry (`URM` — `CBD` historic commercial: dense brick rubble, low void volume, rapid suffocation/crush decay).
    UnreinforcedMasonryUrm,
    /// Non-Ductile Reinforced Concrete (`RC` — `R3`/`MU4` mid-rise apartments/mixed-use: heavy rebar-tied slabs requiring diamond/hydraulic breaching).
    NonDuctileConcreteFrameRc,
    /// Light Wood-Frame / Steel Commercial (`W1/S1` — `R1`/`IND`: higher survivable triangle-of-life voids & faster chainsaw/torch extrication).
    WoodAndLightSteelFrame,
}

impl StructuralCollapseArchetype {
    pub fn short_code(self) -> &'static str {
        match self {
            Self::UnreinforcedMasonryUrm => "URM-Brick",
            Self::NonDuctileConcreteFrameRc => "RC-Slab",
            Self::WoodAndLightSteelFrame => "Wood/Steel",
        }
    }

    pub fn from_zone(z: ZoneCode) -> Self {
        match z {
            ZoneCode::CBD => Self::UnreinforcedMasonryUrm,
            ZoneCode::R3 | ZoneCode::MU4 => Self::NonDuctileConcreteFrameRc,
            ZoneCode::R1 | ZoneCode::IND | ZoneCode::PARK => Self::WoodAndLightSteelFrame,
        }
    }

    /// Returns `(complete_collapse_median_pga_g, hazus_tons_per_1000sqft, rescue_squad_hrs_per_victim, survival_half_life_hrs)`.
    #[inline]
    pub fn collapse_and_rescue_specs(self) -> (f64, f64, f64, f64) {
        match self {
            Self::UnreinforcedMasonryUrm => (0.46, 138.0, 3.2, 24.0),
            Self::NonDuctileConcreteFrameRc => (0.58, 185.0, 4.4, 34.0),
            Self::WoodAndLightSteelFrame => (0.86, 48.0, 1.5, 52.0),
        }
    }
}

/// Joint `ESF #3` Heavy-Equipment Roadway Debris Clearance & `ESF #9` `US&R` Task Force Scheduling Policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebrisAndUsarPolicy {
    /// Uncoordinated perimeter dump-truck haul-off (`~32 tons/hr`) + sequential geographic grid search (traps US&R & ISB trucks!).
    UncoordinatedPerimeterHaulOff,
    /// Logistics-first arterial push: clears `ISB -> CSA -> C-POD` corridors first (`115 tons/hr`), delaying inner `URM`/`RC` rescue grids.
    LogisticsPodCorridorsFirstPush,
    /// Caunhye (2020) / Sahin "Teeth-of-a-Comb" Bi-Objective Phase I Emergency Push (`125 tons/hr` curb push)
    /// co-optimizing `72-hr` live-entrapment survival `dE[Lives]/dt` AND `ISB -> C-POD` `53-ft` trailer reopening!
    CoOptimizedTeethOfCombUsarAndPodPush,
}

impl Default for DebrisAndUsarPolicy {
    fn default() -> Self {
        Self::CoOptimizedTeethOfCombUsarAndPodPush
    }
}

impl DebrisAndUsarPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::UncoordinatedPerimeterHaulOff => {
                "Uncoordinated Full Haul-Off (32 t/h) + Geographic Grid Order"
            }
            Self::LogisticsPodCorridorsFirstPush => {
                "ISB->POD Supply Arterials First Push (Delays URM/RC Rescue)"
            }
            Self::CoOptimizedTeethOfCombUsarAndPodPush => {
                "Co-Optimized 'Teeth-of-a-Comb' Phase I Push (US&R 72h + ISB/POD)"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::UncoordinatedPerimeterHaulOff => "HaulOff-Sequential",
            Self::LogisticsPodCorridorsFirstPush => "POD-First-Push",
            Self::CoOptimizedTeethOfCombUsarAndPodPush => "TeethOfComb-USAR+POD",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemaInsaragMarking {
    /// Unreached behind blocked roadway rubble (`[ ]`).
    BlockedUnreached,
    /// Canine / Technical Acoustic Hasty Search In Progress (`[/]`).
    InProgressSlash,
    /// Primary & Secondary Delayering Extrication Completed (`[X]`).
    CompletedCrossX,
}

impl FemaInsaragMarking {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::BlockedUnreached => "[!]",
            Self::InProgressSlash => "[/]",
            Self::CompletedCrossX => "[X]",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaUsarDebrisConfig {
    /// Chained FEMA `ISB`/`C-POD` logistics, `IPAWS`/`MERS` comms, shelters, contraflow, FFE & seismic MC stack.
    pub fema_pods: FemaIsbPodsConfig,
    pub policy: DebrisAndUsarPolicy,
    /// Deployed FEMA `Type 1 US&R Task Forces` (`80` personnel / `2` 12-hr shifts / `4` heavy rescue squads per TF, `1..=6`, default `2`).
    pub fema_type1_usar_task_forces: usize,
    /// Deployed `ESF #3` / USACE Heavy-Equipment Debris Strike Teams (`D8` dozer + `5-yd` wheel loader + excavator, `1..=16`, default `6`).
    pub heavy_debris_push_crews: usize,
    /// Operational simulation horizon post-mainshock (`24..=96 hours`, default `72.0` Golden-72-Hour US&R window).
    pub horizon_hours: f64,
    /// Snapshot hour (`0.0..=96.0 hr`) used when injecting live roadway debris / bulldozer reopening onto the microscopic `RoadNetwork`!
    pub live_injection_snapshot_hour: f64,
}

impl Default for FemaUsarDebrisConfig {
    fn default() -> Self {
        Self {
            fema_pods: FemaIsbPodsConfig::default(),
            policy: DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush,
            fema_type1_usar_task_forces: 2,
            heavy_debris_push_crews: 6,
            horizon_hours: 72.0,
            live_injection_snapshot_hour: 12.0,
        }
    }
}

/// Evaluates Coburn, Spence & Pomonis (1992 / 2002) & Macintyre et al. (2006) entrapped rubble
/// victim survival probability `S(t_hours) in (0, 1]` as a function of hours elapsed since mainshock
/// and structural collapse archetype (`URM`, `RC`, or `Wood/Steel`).
#[inline]
pub fn coburn_spence_entrapped_survival_prob(
    hours_post_quake: f64,
    archetype: StructuralCollapseArchetype,
) -> f64 {
    let t = hours_post_quake.max(0.0);
    let (_, _, _, t_half_hr) = archetype.collapse_and_rescue_specs();
    let exponent = match archetype {
        StructuralCollapseArchetype::UnreinforcedMasonryUrm => 1.95,
        StructuralCollapseArchetype::NonDuctileConcreteFrameRc => 1.85,
        StructuralCollapseArchetype::WoodAndLightSteelFrame => 1.70,
    };
    (1.0 / (1.0 + (t / t_half_hr).powf(exponent))).clamp(0.01, 1.0)
}

/// Computes expected Hazus-MH structural collapse debris (`metric tons`) generated by a building parcel
/// of `area_m2` under local `pga_g`, and the fraction (`0.0..=0.42`) spilling onto bordering street right-of-way.
#[inline]
pub fn hazus_parcel_collapse_debris_tons(
    area_m2: f64,
    zone: ZoneCode,
    pga_g: f64,
) -> (f64, f64) {
    if zone == ZoneCode::PARK {
        return (0.0, 0.0);
    }
    let arch = StructuralCollapseArchetype::from_zone(zone);
    let (med_g, tons_per_ksqft, _, _) = arch.collapse_and_rescue_specs();
    let stories = match zone {
        ZoneCode::CBD => 4.2,
        ZoneCode::MU4 => 3.4,
        ZoneCode::R3 => 3.0,
        ZoneCode::IND => 1.6,
        ZoneCode::R1 => 1.5,
        ZoneCode::PARK => 0.0,
    };
    let gross_ksqft = (area_m2 * SQFT_PER_M2 * 0.48 * stories) / 1_000.0;
    let p_complete = normal_cdf_phi((pga_g.max(0.02) / med_g).ln() / 0.56);
    let p_extensive = normal_cdf_phi((pga_g.max(0.02) / (med_g * 0.62)).ln() / 0.54);
    let damage_ratio = (0.85 * p_complete + 0.28 * (p_extensive - p_complete).max(0.0)).clamp(0.0, 1.0);
    let total_tons = gross_ksqft * tons_per_ksqft * damage_ratio;
    let street_spill_tons = total_tons * 0.28; // Curb/street-facing facade & upper-story collapse throw
    (total_tons, street_spill_tons)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsngSearchGridRow {
    pub grid_id: usize,
    pub usng_cell_code: String,
    pub center: Vec2,
    pub half_extent_m: f64,
    pub access_segment_id: usize,
    pub dominant_archetype: StructuralCollapseArchetype,
    pub street_debris_tons: f64,
    pub initial_entrapped_victims: f64,
    pub road_access_reopened_hr: f64,
    pub fema_marking: FemaInsaragMarking,
    pub live_rescued_victims: f64,
    pub deceased_entrapment_losses: f64,
    pub live_survival_rate_pct: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UsarDebrisTimelinePoint {
    pub hour_post_quake: f64,
    pub road_corridors_open_pct: f64,
    pub isb_to_pods_connected_pct: f64,
    pub usng_grids_searched_pct: f64,
    pub cumul_live_rescues: f64,
    pub remaining_alive_in_rubble: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaUsarDebrisReport {
    pub config: FemaUsarDebrisConfig,
    pub boo_world_pos: Vec2,
    pub isb_world_pos: Vec2,
    pub pod_world_positions: Vec<Vec2>,
    pub usng_grids_count: usize,
    pub initial_blocked_segments_count: f64,
    pub total_street_debris_tons: f64,
    pub cleared_debris_tons: f64,
    pub total_initial_entrapped_victims: f64,
    pub expected_live_rescues: f64,
    pub p10_live_rescues: f64,
    pub entrapped_survival_rate_pct: f64,
    pub time_to_reopen_all_pods_hrs: f64,
    pub final_open_corridors_pct: f64,
    pub grid_rows: Vec<UsngSearchGridRow>,
    pub timeline: Vec<UsarDebrisTimelinePoint>,
    pub segment_reopen_hour: Vec<f64>,
    pub segment_debris_tons: Vec<f64>,
    pub parcel_saved_pct: Vec<f64>,
    pub parcel_debris_tons: Vec<f64>,
    pub solve_time_ms: f64,
}

impl FemaUsarDebrisReport {
    pub fn to_grid_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.grid_rows.len() * 140);
        out.push_str(FEMA_USNG_GRID_CSV_HEADER);
        out.push('\n');
        for g in &self.grid_rows {
            out.push_str(&format!(
                "{},{},{},{:.1},{:.2},{:.2},{},{:.2},{:.2},{:.2}\n",
                g.grid_id,
                g.usng_cell_code,
                g.dominant_archetype.short_code(),
                g.street_debris_tons,
                g.initial_entrapped_victims,
                g.road_access_reopened_hr,
                g.fema_marking.symbol(),
                g.live_rescued_victims,
                g.deceased_entrapment_losses,
                g.live_survival_rate_pct,
            ));
        }
        out
    }

    pub fn to_timeline_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.timeline.len() * 96);
        out.push_str(FEMA_DEBRIS_TIMELINE_CSV_HEADER);
        out.push('\n');
        for pt in &self.timeline {
            out.push_str(&format!(
                "{:.1},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                pt.hour_post_quake,
                pt.road_corridors_open_pct,
                pt.isb_to_pods_connected_pct,
                pt.usng_grids_searched_pct,
                pt.cumul_live_rescues,
                pt.remaining_alive_in_rubble,
            ));
        }
        out
    }

    pub fn export_grid_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_grid_csv_string())?;
        Ok(self.grid_rows.len())
    }
}

#[derive(Clone, Copy)]
struct HeapState {
    cost: f64,
    seg: usize,
}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits() && self.seg == other.seg
    }
}
impl Eq for HeapState {}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.seg.cmp(&other.seg))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn build_undirected_seg_graph(net: &RoadNetwork) -> Vec<Vec<(usize, f64)>> {
    let n = net.segments.len();
    let mut adj = vec![Vec::new(); n];
    for c in &net.connectors {
        if c.from_lane_id >= net.lanes.len() || c.to_lane_id >= net.lanes.len() {
            continue;
        }
        let u = net.lanes[c.from_lane_id].segment_id;
        let v = net.lanes[c.to_lane_id].segment_id;
        if u < n && v < n && u != v {
            let turn_h = match net.junctions[c.junction_id].control_mode {
                ControlMode::Roundabout => 0.001,
                ControlMode::MaxPressureAdaptive
                | ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::PermissiveYield => match c.turn_type {
                    TurnType::Through | TurnType::Right => 0.0015,
                    TurnType::Left => 0.0025,
                },
            };
            if adj[u].iter().all(|&(nb, _)| nb != v) {
                adj[u].push((v, turn_h));
            }
            // Phase I emergency push bulldozers & US&R convoys can traverse cleared rights-of-way bidirectionally!
            if adj[v].iter().all(|&(nb, _)| nb != u) {
                adj[v].push((u, turn_h));
            }
        }
    }
    adj
}

/// Computes minimum cumulative debris-clearing + travel hours from `origins` to every segment,
/// along with parent pointers to extract the exact bottleneck blocked segments along the path!
fn dijkstra_debris_reopen_tree(
    origins: &[usize],
    seg_clear_hrs: &[f64],
    seg_perm_closed: &[bool],
    adj: &[Vec<(usize, f64)>],
) -> (Vec<f64>, Vec<Option<usize>>) {
    let n = seg_clear_hrs.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut parent = vec![None; n];
    let mut pq = BinaryHeap::new();

    for &o in origins {
        if o < n && !seg_perm_closed[o] {
            let c0 = seg_clear_hrs[o];
            if c0 < dist[o] {
                dist[o] = c0;
                pq.push(HeapState { cost: c0, seg: o });
            }
        }
    }
    while let Some(HeapState { cost, seg }) = pq.pop() {
        if cost > dist[seg] + 1e-9 {
            continue;
        }
        for &(nxt, drive_h) in &adj[seg] {
            if seg_perm_closed[nxt] {
                continue;
            }
            let cand = cost + drive_h + seg_clear_hrs[nxt];
            if cand + 1e-9 < dist[nxt] {
                dist[nxt] = cand;
                parent[nxt] = Some(seg);
                pq.push(HeapState {
                    cost: cand,
                    seg: nxt,
                });
            }
        }
    }
    (dist, parent)
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

/// Executes `Rayon`-parallelized Monte Carlo simulation of FEMA `Type 1 US&R Task Force` (`80`-pax
/// 24-hr shifts, `USNG` structural collapse search grids & Coburn-Spence `72-hr` survival decay),
/// `ESF #3` Heavy-Equipment Phase I Roadway Debris Push, and `ISB -> CSA -> C-POD` Route Reopening!
pub fn run_fema_usar_debris_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &FemaUsarDebrisConfig,
) -> FemaUsarDebrisReport {
    let t0 = Instant::now();
    let n_seg = net.segments.len();
    let n_parcels = parcels.len();
    let seis_cfg = &cfg
        .fema_pods
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
    let horizon_hr = cfg.horizon_hours.clamp(12.0, 120.0);

    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(180.0);
    let sy = (b.max_y - b.min_y).max(180.0);

    // Partition urban footprint into a `3 x 3 = 9` US National Grid (`USNG` 10T EQ ...) Search Sector Grid!
    let gx_n = 3usize;
    let gy_n = 3usize;
    let cell_w = sx / (gx_n as f64);
    let cell_h = sy / (gy_n as f64);

    #[derive(Clone)]
    struct UsngCellMeta {
        grid_id: usize,
        code: String,
        center: Vec2,
        access_seg: usize,
        dom_arch: StructuralCollapseArchetype,
    }

    let mut cells = Vec::new();
    let mut parcel_cell = vec![0usize; n_parcels];
    for gy in 0..gy_n {
        for gx in 0..gx_n {
            let cx = b.min_x + (gx as f64 + 0.5) * cell_w;
            let cy = b.min_y + (gy as f64 + 0.5) * cell_h;
            let mut p_idx = Vec::new();
            let mut urm_cnt = 0usize;
            let mut rc_cnt = 0usize;
            for (pi, p) in parcels.iter().enumerate() {
                let ix = (((p.center.x - b.min_x) / cell_w).floor() as isize)
                    .clamp(0, (gx_n - 1) as isize) as usize;
                let iy = (((p.center.y - b.min_y) / cell_h).floor() as isize)
                    .clamp(0, (gy_n - 1) as isize) as usize;
                if ix == gx && iy == gy {
                    p_idx.push(pi);
                    match StructuralCollapseArchetype::from_zone(p.zone) {
                        StructuralCollapseArchetype::UnreinforcedMasonryUrm => urm_cnt += 2,
                        StructuralCollapseArchetype::NonDuctileConcreteFrameRc => rc_cnt += 1,
                        StructuralCollapseArchetype::WoodAndLightSteelFrame => {}
                    }
                }
            }
            let gid = cells.len();
            for &pi in &p_idx {
                parcel_cell[pi] = gid;
            }
            let dom_arch = if urm_cnt >= 2 {
                StructuralCollapseArchetype::UnreinforcedMasonryUrm
            } else if rc_cnt >= 2 {
                StructuralCollapseArchetype::NonDuctileConcreteFrameRc
            } else {
                StructuralCollapseArchetype::WoodAndLightSteelFrame
            };
            let (aseg, _) = nearest_open_seg(net, Vec2 { x: cx, y: cy });
            cells.push(UsngCellMeta {
                grid_id: gid,
                code: format!("10T-EQ-{}{}", 42 + gx * 3, 71 + gy * 3),
                center: Vec2 { x: cx, y: cy },
                access_seg: aseg,
                dom_arch,
            });
        }
    }
    let n_grids = cells.len();

    // Base of Operations (`US&R BoO`), FEMA `ISB`, and County `C-POD` target segments!
    let (boo_seg, boo_pos) = nearest_open_seg(
        net,
        facilities
            .first()
            .map(|f| f.world_pos)
            .unwrap_or(Vec2 {
                x: b.min_x + 0.16 * sx,
                y: b.min_y + 0.22 * sy,
            }),
    );
    let (isb_seg, isb_pos) = nearest_open_seg(
        net,
        Vec2 {
            x: b.min_x + 0.08 * sx,
            y: b.min_y + 0.90 * sy,
        },
    );
    let pod_targets_rel = [
        (0.50, 0.50),
        (0.20, 0.44),
        (0.45, 0.18),
        (0.82, 0.42),
        (0.52, 0.82),
    ];
    let n_pods = cfg.fema_pods.active_county_pods.clamp(2, 5);
    let mut pod_segs = Vec::with_capacity(n_pods);
    let mut pod_positions = Vec::with_capacity(n_pods);
    for &(fx, fy) in pod_targets_rel.iter().take(n_pods) {
        let (ps, pw) = nearest_open_seg(
            net,
            Vec2 {
                x: b.min_x + fx * sx,
                y: b.min_y + fy * sy,
            },
        );
        pod_segs.push(ps);
        pod_positions.push(pw);
    }

    // Map each parcel to its closest road segment (where facade/story rubble spills!)
    let mut parcel_border_seg = vec![0usize; n_parcels];
    for (pi, p) in parcels.iter().enumerate() {
        parcel_border_seg[pi] = nearest_open_seg(net, p.center).0;
    }

    let adj = build_undirected_seg_graph(net);
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, seis_cfg);

    let base_sa10_g = {
        let m_term = (0.82 * (seis_cfg.moment_magnitude_mw - 6.0)).exp();
        let r_eff = (seis_cfg.rupture_standoff_km.hypot(7.5)).max(3.5);
        let site_amp = (360.0 / seis_cfg.vs30_m_per_s.clamp(140.0, 900.0)).powf(0.45);
        (0.26 * m_term * (12.0 / r_eff).powf(0.82) * site_amp).clamp(0.05, 1.85)
    };

    // Equipment productivity:
    // - Uncoordinated Full Dump-Truck Haul-Off = 32 tons/hr per crew
    // - Phase I Emergency Roadway Curb Push (D8 dozer + wheel loader) = 120 tons/hr per crew!
    let crew_tons_per_hr = match cfg.policy {
        DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff => 32.0,
        DebrisAndUsarPolicy::LogisticsPodCorridorsFirstPush => 115.0,
        DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush => 128.0,
    };
    let n_dozer_crews = cfg.heavy_debris_push_crews.clamp(1, 24) as f64;
    // Each FEMA Type 1 US&R Task Force (80 personnel across two 12-hr shifts) fields 4 active Heavy Breaching/Rescue Squads 24/7!
    let active_rescue_squads = (cfg.fema_type1_usar_task_forces.clamp(1, 8) * 4) as f64;
    let tl_steps = 13usize; // 0 .. horizon_hr

    #[derive(Clone)]
    struct TrialUsarOut {
        blocked_segs: f64,
        tot_debris_t: f64,
        cleared_debris_t: f64,
        init_trapped: f64,
        live_rescues: f64,
        pod_reopen_hr: f64,
        final_open_pct: f64,
        g_debris: Vec<f64>,
        g_trapped: Vec<f64>,
        g_reopen_hr: Vec<f64>,
        g_status: Vec<u8>,
        g_rescued: Vec<f64>,
        g_deceased: Vec<f64>,
        seg_reopen_hr: Vec<f64>,
        seg_debris: Vec<f64>,
        p_saved_pct: Vec<f64>,
        p_debris: Vec<f64>,
        tl_open_pct: Vec<f64>,
        tl_pods_pct: Vec<f64>,
        tl_grids_pct: Vec<f64>,
        tl_rescues: Vec<f64>,
        tl_rem_alive: Vec<f64>,
    }

    let base_seed = seis_cfg.random_seed ^ 0x5553_4152_3432_0009;
    let trial_outs: Vec<TrialUsarOut> = (0..trials)
        .into_par_iter()
        .map(|tr_idx| {
            let mut rng = SplitMix64::new(
                base_seed.wrapping_add((tr_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let z_q = rng.next_standard_normal();
            let sa10_city = (base_sa10_g * (0.34 * z_q).exp()).clamp(0.04, 2.6);
            let pga_city = (sa10_city * 1.14).clamp(0.05, 2.5);

            // Permanent structural bridge collapses (cannot be cleared by bulldozers alone within 72h!)
            let mut seg_perm_closed = vec![false; n_seg];
            for seg in &net.segments {
                if seg.closed {
                    seg_perm_closed[seg.id] = true;
                }
            }
            for br in &bridges {
                if rng.next_f64_open01() < br.effective_archetype.exceedance_probs(sa10_city)[2] {
                    for &sid in &br.segment_ids {
                        if sid < n_seg {
                            seg_perm_closed[sid] = true;
                        }
                    }
                }
            }

            // Generate per-parcel Hazus rubble tonnage, street spill tonnage & live-entrapped victims!
            let mut seg_debris = vec![0.0f64; n_seg];
            let mut p_debris = vec![0.0f64; n_parcels];
            let mut g_debris = vec![0.0f64; n_grids];
            let mut g_trapped = vec![0.0f64; n_grids];

            for (pi, p) in parcels.iter().enumerate() {
                let pga_loc = (pga_city * (0.85 + 0.30 * rng.next_f64_open01())).clamp(0.04, 2.8);
                let (tot_t, spill_t) =
                    hazus_parcel_collapse_debris_tons(p.area_ha * 10_000.0, p.zone, pga_loc);
                p_debris[pi] = tot_t;
                let b_seg = parcel_border_seg[pi];
                if b_seg < n_seg {
                    seg_debris[b_seg] += spill_t;
                }
                let gid = parcel_cell[pi];
                g_debris[gid] += spill_t;
                // Hazus Tier-3 live entrapment rate (~0.022 live-trapped victims per ton of Complete-collapse rubble in occupied buildings)
                let occ_density = match p.zone {
                    ZoneCode::CBD | ZoneCode::R3 => 0.026,
                    ZoneCode::MU4 => 0.021,
                    ZoneCode::R1 | ZoneCode::IND => 0.012,
                    ZoneCode::PARK => 0.0,
                };
                g_trapped[gid] += (tot_t * occ_density).clamp(0.0, 28.0);
            }

            // Segment is blocked to heavy US&R rigs & 53-ft POD trailers when street rubble >= 28 tons * lane_count!
            let mut seg_blocked = vec![false; n_seg];
            let mut seg_work_crew_hrs = vec![0.0f64; n_seg];
            let mut init_blocked_cnt = 0.0f64;
            for seg in &net.segments {
                let thresh = 26.0 * (seg.lane_count.max(1) as f64);
                if !seg_perm_closed[seg.id] && seg_debris[seg.id] >= thresh {
                    seg_blocked[seg.id] = true;
                    init_blocked_cnt += 1.0;
                    // Under Phase I Emergency Push, bulldozers only need to push ~42% of street-spill mass aside to open 1 emergency lane;
                    // under UncoordinatedPerimeterHaulOff, crews haul 100% of street-spill mass away!
                    let push_fraction = match cfg.policy {
                        DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff => 1.0,
                        _ => 0.42,
                    };
                    seg_work_crew_hrs[seg.id] =
                        ((seg_debris[seg.id] * push_fraction) / crew_tons_per_hr).max(0.25);
                }
            }

            // Schedule ESF #3 Bulldozer/Loader Teams across `0 .. horizon_hr`!
            let mut seg_reopen_hr = vec![0.0f64; n_seg];
            for sid in 0..n_seg {
                if seg_perm_closed[sid] {
                    seg_reopen_hr[sid] = horizon_hr * 2.5;
                } else if seg_blocked[sid] {
                    seg_reopen_hr[sid] = horizon_hr * 1.8;
                } else {
                    seg_reopen_hr[sid] = 0.0;
                }
            }

            let mut cleared_seg = vec![false; n_seg];
            for sid in 0..n_seg {
                if !seg_perm_closed[sid] && !seg_blocked[sid] {
                    cleared_seg[sid] = true;
                }
            }
            let mut elapsed_push_hr = 0.4f64; // 24-min initial ESF-3 equipment mobilization
            let mut cleared_debris_tons = 0.0f64;
            let origins = [boo_seg, isb_seg];

            // Iteratively select the next target node/corridor to punch open!
            for _step in 0..n_seg.min(48) {
                if elapsed_push_hr >= horizon_hr {
                    break;
                }
                let cur_seg_cost: Vec<f64> = (0..n_seg)
                    .map(|s| if cleared_seg[s] { 0.0 } else { seg_work_crew_hrs[s] / n_dozer_crews })
                    .collect();
                let (dist_h, parent) =
                    dijkstra_debris_reopen_tree(&origins, &cur_seg_cost, &seg_perm_closed, &adj);

                // Score candidate unreached targets (`USNG` collapse grids + `County C-POD` sites)
                let mut best_target_seg: Option<(usize, f64)> = None;

                match cfg.policy {
                    DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff => {
                        // Naive nearest-blocked segment order regardless of trapped victims or PODs!
                        for sid in 0..n_seg {
                            if !cleared_seg[sid] && !seg_perm_closed[sid] && dist_h[sid].is_finite() {
                                let sc = 100.0 / (dist_h[sid] + 0.25);
                                if best_target_seg.map_or(true, |bt| sc > bt.1) {
                                    best_target_seg = Some((sid, sc));
                                }
                            }
                        }
                    }
                    DebrisAndUsarPolicy::LogisticsPodCorridorsFirstPush => {
                        // Strictly open all ISB -> C-POD corridors first, only afterwards visiting USNG collapse grids!
                        for &ps in &pod_segs {
                            if ps < n_seg && dist_h[ps] > 0.02 && dist_h[ps].is_finite() {
                                let sc = 5_000.0 / (dist_h[ps] + 0.15);
                                if best_target_seg.map_or(true, |bt| sc > bt.1) {
                                    best_target_seg = Some((ps, sc));
                                }
                            }
                        }
                        if best_target_seg.is_none() {
                            for g in &cells {
                                let gs = g.access_seg;
                                if gs < n_seg && dist_h[gs] > 0.02 && dist_h[gs].is_finite() && g_trapped[g.grid_id] > 0.05 {
                                    let sc = g_trapped[g.grid_id] / (dist_h[gs] + 0.2);
                                    if best_target_seg.map_or(true, |bt| sc > bt.1) {
                                        best_target_seg = Some((gs, sc));
                                    }
                                }
                            }
                        }
                    }
                    DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush => {
                        // Bi-objective "Teeth-of-a-Comb": weigh marginal live-entrapment survival `trapped * S(t_arr)`
                        // alongside ISB->POD corridor connectivity!
                        for g in &cells {
                            let gs = g.access_seg;
                            if gs < n_seg && dist_h[gs] > 0.02 && dist_h[gs].is_finite() && g_trapped[g.grid_id] > 0.05 {
                                let s_prob = coburn_spence_entrapped_survival_prob(
                                    elapsed_push_hr + dist_h[gs],
                                    g.dom_arch,
                                );
                                let sc = (g_trapped[g.grid_id] * s_prob * 18.0) / (dist_h[gs] + 0.15);
                                if best_target_seg.map_or(true, |bt| sc > bt.1) {
                                    best_target_seg = Some((gs, sc));
                                }
                            }
                        }
                        for &ps in &pod_segs {
                            if ps < n_seg && dist_h[ps] > 0.02 && dist_h[ps].is_finite() {
                                let sc = 28.0 / (dist_h[ps] + 0.20);
                                if best_target_seg.map_or(true, |bt| sc > bt.1) {
                                    best_target_seg = Some((ps, sc));
                                }
                            }
                        }
                    }
                }

                // Fallback: clear any remaining reachable blocked segment
                if best_target_seg.is_none() {
                    for sid in 0..n_seg {
                        if !cleared_seg[sid] && !seg_perm_closed[sid] && dist_h[sid].is_finite() {
                            best_target_seg = Some((sid, 1.0));
                            break;
                        }
                    }
                }

                let Some((tgt_seg, _)) = best_target_seg else {
                    break;
                };
                // Walk parent chain from `tgt_seg` back to already-cleared tree and clear every blocked arc along the tooth!
                let mut chain = Vec::new();
                let mut cur = Some(tgt_seg);
                let mut guard = 0usize;
                while let Some(u) = cur {
                    if !cleared_seg[u] && !seg_perm_closed[u] {
                        chain.push(u);
                    }
                    cur = parent[u];
                    guard += 1;
                    if guard > n_seg {
                        break;
                    }
                }
                chain.reverse();
                if chain.is_empty() {
                    cleared_seg[tgt_seg] = true;
                    continue;
                }
                for u in chain {
                    let dt_h = seg_work_crew_hrs[u] / n_dozer_crews;
                    elapsed_push_hr += dt_h;
                    cleared_seg[u] = true;
                    seg_reopen_hr[u] = elapsed_push_hr;
                    cleared_debris_tons += seg_debris[u];
                }
            }

            // Evaluate USNG Grid Access Reopening Hours & Schedule FEMA Type 1 US&R Heavy Rescue Squads!
            let mut g_reopen_hr = vec![0.0f64; n_grids];
            for (gi, g) in cells.iter().enumerate() {
                g_reopen_hr[gi] = if g.access_seg < n_seg {
                    seg_reopen_hr[g.access_seg]
                } else {
                    2.0
                };
            }

            // Order USNG grids for US&R Task Force Breaching/Extrication Squads:
            // Under CoOptimizedTeethOfComb, Canine/Delsar Hasty Search prioritizes grids by marginal `dE[Lives]/dt = S(t) / hrs_per_victim`!
            let mut grid_order: Vec<usize> = (0..n_grids).collect();
            grid_order.sort_by(|&a, &b| match cfg.policy {
                DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff => a.cmp(&b),
                DebrisAndUsarPolicy::LogisticsPodCorridorsFirstPush => {
                    let va = g_trapped[a]
                        * coburn_spence_entrapped_survival_prob(
                            g_reopen_hr[a].min(horizon_hr),
                            cells[a].dom_arch,
                        )
                        / (g_reopen_hr[a] + 2.0);
                    let vb = g_trapped[b]
                        * coburn_spence_entrapped_survival_prob(
                            g_reopen_hr[b].min(horizon_hr),
                            cells[b].dom_arch,
                        )
                        / (g_reopen_hr[b] + 2.0);
                    vb.partial_cmp(&va).unwrap_or(Ordering::Equal)
                }
                DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush => {
                    let score_grid = |gi: usize| -> f64 {
                        if g_trapped[gi] <= 0.02 || g_reopen_hr[gi] > horizon_hr {
                            return -1.0;
                        }
                        let (_, _, hrs_per_vic, _) = cells[gi].dom_arch.collapse_and_rescue_specs();
                        let t_arr = (g_reopen_hr[gi] + 0.4).max(1.0);
                        let surv = coburn_spence_entrapped_survival_prob(t_arr, cells[gi].dom_arch);
                        surv / (hrs_per_vic * (1.0 + 0.05 * g_reopen_hr[gi]))
                    };
                    score_grid(b)
                        .partial_cmp(&score_grid(a))
                        .unwrap_or(Ordering::Equal)
                }
            });

            let mut squad_free_hr = vec![1.0f64; active_rescue_squads as usize];
            let mut g_rescued = vec![0.0f64; n_grids];
            let mut g_deceased = vec![0.0f64; n_grids];
            let mut g_status = vec![0u8; n_grids]; // 0 = [!], 1 = [/], 2 = [X]
            let mut rescue_events: Vec<(f64, f64)> = Vec::new(); // (hour, saved_lives)

            for &gi in &grid_order {
                let trapped = g_trapped[gi];
                if trapped <= 0.02 {
                    if g_reopen_hr[gi] <= horizon_hr {
                        g_status[gi] = 2; // Hasty search [X] clear!
                    }
                    continue;
                }
                // Early local bystander / surface rescues (~18% within first 3h) occur regardless of heavy road blockage
                let byst = trapped * 0.18;
                g_rescued[gi] = byst;
                rescue_events.push((3.0, byst));

                if g_reopen_hr[gi] > horizon_hr {
                    // Road never reopened within 72h: deep structural void victims cannot be reached by Type-1 US&R rigs
                    g_deceased[gi] = trapped - byst;
                    g_status[gi] = 0;
                    continue;
                }

                let deep_trapped = (trapped - byst).max(0.0);
                let (_, _, squad_hrs_per_vic, _) = cells[gi].dom_arch.collapse_and_rescue_specs();
                // Canine + Delsar technical search cuts empty-void wasted breaching time by 32% under TeethOfComb!
                let eff_hrs_per_vic = if cfg.policy
                    == DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush
                {
                    squad_hrs_per_vic * 0.68
                } else {
                    squad_hrs_per_vic
                };

                // Split grid's remaining deep-void trapped victims into extrication tasks across earliest available US&R squads
                let n_chunks = (deep_trapped.ceil() as usize).clamp(1, 16);
                let vic_per_chunk = deep_trapped / (n_chunks as f64);
                let mut completed_chunks = 0usize;

                for _ in 0..n_chunks {
                    let (sq_i, &sq_t) = squad_free_hr
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                        .unwrap_or((0, &1.0));
                    let t_start = sq_t.max(g_reopen_hr[gi] + 0.4); // +24m USNG hasty recon
                    let dur_h = vic_per_chunk * eff_hrs_per_vic;
                    let t_extricated = t_start + dur_h;
                    if t_start < horizon_hr {
                        let actual_end = t_extricated.min(horizon_hr + 4.0);
                        let surv_p = coburn_spence_entrapped_survival_prob(
                            0.5 * (t_start + actual_end),
                            cells[gi].dom_arch,
                        );
                        let work_frac =
                            ((horizon_hr - t_start) / dur_h.max(0.1)).clamp(0.0, 1.0);
                        let saved = vic_per_chunk * work_frac * surv_p;
                        g_rescued[gi] += saved;
                        g_deceased[gi] += vic_per_chunk - saved;
                        squad_free_hr[sq_i] = t_extricated;
                        if t_extricated <= horizon_hr {
                            completed_chunks += 1;
                            rescue_events.push((t_extricated, saved));
                        } else {
                            rescue_events.push((horizon_hr, saved));
                        }
                    } else {
                        g_deceased[gi] += vic_per_chunk;
                    }
                }
                g_status[gi] = if completed_chunks == n_chunks { 2 } else { 1 };
            }

            // Time to reopen all C-POD supply routes from ISB
            let mut max_pod_hr = 0.0f64;
            for &ps in &pod_segs {
                let rh = if ps < n_seg {
                    seg_reopen_hr[ps].min(horizon_hr)
                } else {
                    0.0
                };
                max_pod_hr = max_pod_hr.max(rh);
            }

            // Populate 13-point timeline across `0 .. horizon_hr`
            let mut tl_open = vec![0.0; tl_steps];
            let mut tl_pods = vec![0.0; tl_steps];
            let mut tl_grids = vec![0.0; tl_steps];
            let mut tl_resc = vec![0.0; tl_steps];
            let mut tl_alive = vec![0.0; tl_steps];
            let tot_trapped: f64 = g_trapped.iter().sum();

            for ti in 0..tl_steps {
                let th = (ti as f64 / (tl_steps - 1) as f64) * horizon_hr;
                let open_s = (0..n_seg)
                    .filter(|&s| !seg_perm_closed[s] && seg_reopen_hr[s] <= th + 1e-6)
                    .count();
                tl_open[ti] = 100.0 * (open_s as f64) / (n_seg.max(1) as f64);

                let open_p = pod_segs
                    .iter()
                    .filter(|&&ps| ps < n_seg && seg_reopen_hr[ps] <= th + 1e-6)
                    .count();
                tl_pods[ti] = 100.0 * (open_p as f64) / (pod_segs.len().max(1) as f64);

                let reached_g = (0..n_grids)
                    .filter(|&gi| g_reopen_hr[gi] + 0.5 <= th + 1e-6)
                    .count();
                tl_grids[ti] = 100.0 * (reached_g as f64) / (n_grids.max(1) as f64);

                let cum_r: f64 = rescue_events
                    .iter()
                    .filter(|&&(eh, _)| eh <= th + 1e-6)
                    .map(|&(_, sv)| sv)
                    .sum();
                tl_resc[ti] = cum_r;
                let unextricated = (tot_trapped - cum_r * 1.45).max(0.0);
                let avg_surv = coburn_spence_entrapped_survival_prob(
                    th,
                    StructuralCollapseArchetype::NonDuctileConcreteFrameRc,
                );
                tl_alive[ti] = unextricated * avg_surv;
            }

            let mut p_saved_pct = vec![100.0; n_parcels];
            for pi in 0..n_parcels {
                let gi = parcel_cell[pi];
                p_saved_pct[pi] = if g_trapped[gi] > 0.02 {
                    100.0 * (g_rescued[gi] / g_trapped[gi]).clamp(0.0, 1.0)
                } else {
                    95.0
                };
            }

            let tot_debris_t: f64 = seg_debris.iter().sum();
            let live_rescues: f64 = g_rescued.iter().sum();

            TrialUsarOut {
                blocked_segs: init_blocked_cnt,
                tot_debris_t,
                cleared_debris_t: cleared_debris_tons,
                init_trapped: tot_trapped,
                live_rescues,
                pod_reopen_hr: max_pod_hr,
                final_open_pct: tl_open[tl_steps - 1],
                g_debris,
                g_trapped,
                g_reopen_hr,
                g_status,
                g_rescued,
                g_deceased,
                seg_reopen_hr,
                seg_debris,
                p_saved_pct,
                p_debris,
                tl_open_pct: tl_open,
                tl_pods_pct: tl_pods,
                tl_grids_pct: tl_grids,
                tl_rescues: tl_resc,
                tl_rem_alive: tl_alive,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let mean_trapped = trial_outs.iter().map(|o| o.init_trapped).sum::<f64>() * inv_t;
    let mean_rescued = trial_outs.iter().map(|o| o.live_rescues).sum::<f64>() * inv_t;

    let mut resc_sorted: Vec<f64> = trial_outs.iter().map(|o| o.live_rescues).collect();
    resc_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p10_rescues = resc_sorted[((trials as f64 * 0.10).floor() as usize).min(trials - 1)];

    let mut grid_rows = Vec::with_capacity(n_grids);
    let half_ext = 0.5 * cell_w.min(cell_h);
    for (gi, g) in cells.iter().enumerate() {
        let gt = trial_outs.iter().map(|o| o.g_trapped[gi]).sum::<f64>() * inv_t;
        let gr = trial_outs.iter().map(|o| o.g_rescued[gi]).sum::<f64>() * inv_t;
        let gd = trial_outs.iter().map(|o| o.g_deceased[gi]).sum::<f64>() * inv_t;
        let st_avg = trial_outs.iter().map(|o| o.g_status[gi] as f64).sum::<f64>() * inv_t;
        let mark = if st_avg >= 1.55 {
            FemaInsaragMarking::CompletedCrossX
        } else if st_avg >= 0.55 {
            FemaInsaragMarking::InProgressSlash
        } else {
            FemaInsaragMarking::BlockedUnreached
        };
        grid_rows.push(UsngSearchGridRow {
            grid_id: gi,
            usng_cell_code: g.code.clone(),
            center: g.center,
            half_extent_m: half_ext,
            access_segment_id: g.access_seg,
            dominant_archetype: g.dom_arch,
            street_debris_tons: trial_outs.iter().map(|o| o.g_debris[gi]).sum::<f64>() * inv_t,
            initial_entrapped_victims: gt,
            road_access_reopened_hr: trial_outs
                .iter()
                .map(|o| o.g_reopen_hr[gi].min(horizon_hr))
                .sum::<f64>()
                * inv_t,
            fema_marking: mark,
            live_rescued_victims: gr,
            deceased_entrapment_losses: gd,
            live_survival_rate_pct: if gt > 0.02 {
                (100.0 * gr / gt).clamp(0.0, 100.0)
            } else {
                100.0
            },
        });
    }

    let mut timeline = Vec::with_capacity(tl_steps);
    for ti in 0..tl_steps {
        timeline.push(UsarDebrisTimelinePoint {
            hour_post_quake: (ti as f64 / (tl_steps - 1) as f64) * horizon_hr,
            road_corridors_open_pct: trial_outs.iter().map(|o| o.tl_open_pct[ti]).sum::<f64>()
                * inv_t,
            isb_to_pods_connected_pct: trial_outs.iter().map(|o| o.tl_pods_pct[ti]).sum::<f64>()
                * inv_t,
            usng_grids_searched_pct: trial_outs.iter().map(|o| o.tl_grids_pct[ti]).sum::<f64>()
                * inv_t,
            cumul_live_rescues: trial_outs.iter().map(|o| o.tl_rescues[ti]).sum::<f64>() * inv_t,
            remaining_alive_in_rubble: trial_outs.iter().map(|o| o.tl_rem_alive[ti]).sum::<f64>()
                * inv_t,
        });
    }

    let mut segment_reopen_hour = vec![0.0; n_seg];
    let mut segment_debris_tons = vec![0.0; n_seg];
    for s in 0..n_seg {
        segment_reopen_hour[s] = trial_outs.iter().map(|o| o.seg_reopen_hr[s]).sum::<f64>() * inv_t;
        segment_debris_tons[s] = trial_outs.iter().map(|o| o.seg_debris[s]).sum::<f64>() * inv_t;
    }

    let mut parcel_saved_pct = vec![100.0; n_parcels];
    let mut parcel_debris_tons = vec![0.0; n_parcels];
    for pi in 0..n_parcels {
        parcel_saved_pct[pi] = trial_outs.iter().map(|o| o.p_saved_pct[pi]).sum::<f64>() * inv_t;
        parcel_debris_tons[pi] = trial_outs.iter().map(|o| o.p_debris[pi]).sum::<f64>() * inv_t;
    }

    FemaUsarDebrisReport {
        config: cfg.clone(),
        boo_world_pos: boo_pos,
        isb_world_pos: isb_pos,
        pod_world_positions: pod_positions,
        usng_grids_count: n_grids,
        initial_blocked_segments_count: trial_outs.iter().map(|o| o.blocked_segs).sum::<f64>()
            * inv_t,
        total_street_debris_tons: trial_outs.iter().map(|o| o.tot_debris_t).sum::<f64>() * inv_t,
        cleared_debris_tons: trial_outs.iter().map(|o| o.cleared_debris_t).sum::<f64>() * inv_t,
        total_initial_entrapped_victims: mean_trapped,
        expected_live_rescues: mean_rescued,
        p10_live_rescues: p10_rescues,
        entrapped_survival_rate_pct: (100.0 * mean_rescued / mean_trapped.max(1e-6))
            .clamp(0.0, 100.0),
        time_to_reopen_all_pods_hrs: trial_outs.iter().map(|o| o.pod_reopen_hr).sum::<f64>()
            * inv_t,
        final_open_corridors_pct: trial_outs.iter().map(|o| o.final_open_pct).sum::<f64>() * inv_t,
        grid_rows,
        timeline,
        segment_reopen_hour,
        segment_debris_tons,
        parcel_saved_pct,
        parcel_debris_tons,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

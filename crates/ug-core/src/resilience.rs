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

//! Arterial Bridge & Corridor Closure Stress-Test Suite and Disruption-Hardened MCLP Siting
//!
//! Grounded in:
//! - Church, Scaparra & Middleton (2004), *Identifying Critical Infrastructure: The Median and
//!   Covering Facility Interdiction Problems*, Annals of the AAG 94(3):491-502
//!   (`https://www.tandfonline.com/doi/abs/10.1111/j.1467-8306.2004.00410.x`) and fortification
//!   extensions (`https://kar.kent.ac.uk/25492/1/79_Web_version.pdf`).
//! - Scott, Novak, Aultman-Hall & Guo (2006), *Network Robustness Index: A new method for
//!   identifying critical links and evaluating the performance of transportation networks*,
//!   Journal of Transport Geography 14(3):215-227
//!   (`https://www.researchgate.net/publication/222652140_Network_Robustness_Index_A_new_method_for_identifying_critical_links_and_evaluating_the_performance_of_transportation_networks`).
//! - FEMA / USFA NFPA 1710 240-second (4-minute) travel-time coverage benchmark
//!   (`https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::mclp::{
    daskin_expected_availability_prob, MclpCandidateSite, MclpOptimizationReport, MclpParetoPoint,
    MclpSitingConfig, MclpWeightingMode,
};
use crate::network::{
    sample_polyline_fast, ControlMode, RoadNetwork, TurnType, Vec2, LANE_WIDTH_M,
};
use crate::zoning::Parcel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const RESILIENCE_CSV_HEADER: &str = "rank,interdiction_order,corridor_label,closed_segment_ids,is_bridge_or_trunk,baseline_parcel_cov_pct,post_parcel_cov_pct,coverage_drop_pct_pts,baseline_resident_cov_pct,post_resident_cov_pct,resident_drop_pct_pts,post_p90_travel_s,p90_surge_s,post_mean_travel_s,nri_system_delay_parcel_s,newly_uncovered_parcels,isolated_parcels,retains_nfpa_target";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisruptionInterdictionDepth {
    /// Exhaustive single-corridor (`N-1`) arterial/bridge closure sweep (Scott et al. 2006 NRI).
    SingleLinkNMinus1,
    /// Single-corridor (`N-1`) plus simultaneous 2-corridor (`N-2`) cut-set interdiction across top bottlenecks.
    DualLinkNMinus2,
}

impl DisruptionInterdictionDepth {
    pub fn label(self) -> &'static str {
        match self {
            Self::SingleLinkNMinus1 => "N-1 Single Bridge / Arterial Closure",
            Self::DualLinkNMinus2 => "N-1 + N-2 Compound Cut-Set (2 Bridges/Corridors)",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceStressConfig {
    pub interdiction_depth: DisruptionInterdictionDepth,
    /// When true, closes both directions `(u->v, v->u)` of a two-way bridge or arterial simultaneously.
    pub bidirectional_closure: bool,
    /// Prioritizes major arterials, bridges, trunks, and collectors (`priority >= 2`) up to `max_candidate_corridors`.
    pub arterials_and_bridges_only: bool,
    /// When true, skips closing the immediate block apron directly hosting an active Fire Station bay door.
    pub protect_station_host_blocks: bool,
    /// Maximum candidate corridors screened in `N-1` (`12..=180`).
    pub max_candidate_corridors: usize,
    /// Top `N-1` corridors combined pairwise `C(K, 2)` during `DualLinkNMinus2` (`4..=24`).
    pub n_minus_2_top_pool: usize,
    /// Target coverage percentage (NFPA 1710 benchmark is `90.0%`).
    pub target_coverage_pct: f64,
    /// Maximum allowed response travel time in seconds (NFPA 1710 benchmark is `240.0` s).
    pub threshold_travel_s: f64,
}

impl Default for ResilienceStressConfig {
    fn default() -> Self {
        Self {
            interdiction_depth: DisruptionInterdictionDepth::DualLinkNMinus2,
            bidirectional_closure: true,
            arterials_and_bridges_only: true,
            protect_station_host_blocks: true,
            max_candidate_corridors: 96,
            n_minus_2_top_pool: 16,
            target_coverage_pct: 90.0,
            threshold_travel_s: 240.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorClosureUnit {
    pub corridor_idx: usize,
    pub segment_ids: Vec<usize>,
    pub label: String,
    pub highway: String,
    pub priority: u8,
    pub is_bridge_or_trunk: bool,
    pub length_m: f64,
    pub midpoint: Vec2,
    pub p0: Vec2,
    pub p1: Vec2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorDisruptionOutcome {
    pub rank: usize,
    pub interdiction_order: usize,
    pub label: String,
    pub closed_segment_ids: Vec<usize>,
    pub is_bridge_or_trunk: bool,
    pub midpoint: Vec2,
    pub segments_endpoints_hint: Vec<[f64; 4]>,
    pub post_parcel_cov_pct: f64,
    pub coverage_drop_pct_pts: f64,
    pub post_resident_cov_pct: f64,
    pub resident_drop_pct_pts: f64,
    pub post_p90_travel_s: f64,
    pub p90_surge_s: f64,
    pub post_mean_travel_s: f64,
    /// Scott et al. (2006) Network Robustness Index (`NRI`): aggregate excess response travel-time
    /// cost (`sum_p (t_disrupted(p) - t_intact(p))`) across all city parcels (parcel-seconds).
    pub nri_system_delay_parcel_s: f64,
    pub newly_uncovered_parcels: usize,
    pub isolated_parcels: usize,
    pub retains_nfpa_target: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceStressReport {
    pub config: ResilienceStressConfig,
    pub fire_station_count: usize,
    pub candidate_corridors_count: usize,
    pub evaluated_scenarios: usize,
    pub baseline_parcel_cov_pct: f64,
    pub baseline_resident_cov_pct: f64,
    pub baseline_p90_travel_s: f64,
    pub baseline_mean_travel_s: f64,
    pub worst_case_parcel_cov_pct: f64,
    pub worst_case_resident_cov_pct: f64,
    pub max_coverage_drop_pct_pts: f64,
    pub max_p90_surge_s: f64,
    pub max_nri_delay_parcel_s: f64,
    pub max_isolated_parcels: usize,
    pub scenarios_retaining_target: usize,
    pub scenarios_failing_target: usize,
    pub resilience_pass_rate_pct: f64,
    pub top_vulnerabilities: Vec<CorridorDisruptionOutcome>,
    /// Per-segment normalized criticality (`0.0..=1.0`) for viewport highlighting.
    pub segment_vulnerability_norm: Vec<f64>,
    pub solve_time_ms: f64,
}

impl ResilienceStressReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str(RESILIENCE_CSV_HEADER);
        out.push('\n');
        for row in &self.top_vulnerabilities {
            let escaped_lbl = if row.label.contains(',') || row.label.contains('"') {
                format!("\"{}\"", row.label.replace('"', "\"\""))
            } else {
                row.label.clone()
            };
            let seg_list = row
                .closed_segment_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(";");
            out.push_str(&format!(
                "{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{},{},{}\n",
                row.rank,
                row.interdiction_order,
                escaped_lbl,
                seg_list,
                row.is_bridge_or_trunk,
                self.baseline_parcel_cov_pct,
                row.post_parcel_cov_pct,
                row.coverage_drop_pct_pts,
                self.baseline_resident_cov_pct,
                row.post_resident_cov_pct,
                row.resident_drop_pct_pts,
                row.post_p90_travel_s,
                row.p90_surge_s,
                row.post_mean_travel_s,
                row.nri_system_delay_parcel_s,
                row.newly_uncovered_parcels,
                row.isolated_parcels,
                row.retains_nfpa_target,
            ));
        }
        out
    }

    pub fn export_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_csv_string())?;
        Ok(self.top_vulnerabilities.len())
    }
}

#[derive(Copy, Clone, PartialEq)]
struct DijkNode {
    cost_s: f64,
    seg_id: usize,
}
impl Eq for DijkNode {}
impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost_s
            .partial_cmp(&self.cost_s)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct PreparedEmergencyGraph {
    seg_trav_s: Vec<f64>,
    fwd_adj: Vec<Vec<(usize, f64)>>,
    twin_rev: Vec<Option<usize>>,
    /// Flattened `[n_parcels * n_segs]` local block-access time `p.center.dist(mid_s) / 7.5`.
    access_matrix_s: Vec<f64>,
    turnout_s: f64,
}

fn prepare_emergency_graph(
    net: &RoadNetwork,
    parcels: &[Parcel],
    evp_cfg: &EvpConfig,
) -> PreparedEmergencyGraph {
    let n = net.segments.len();
    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    let seg_trav_s: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                f64::INFINITY
            } else {
                let eff_v = (s.ema_speed_mps.max(s.speed_mps * 0.48) * speed_boost).clamp(3.2, 29.0);
                let stall_pen = if s.stalled_lane.is_some() && s.lane_count == 1 {
                    28.0
                } else {
                    0.0
                };
                s.length / eff_v + stall_pen
            }
        })
        .collect();

    let mut fwd_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let junc = &net.junctions[seg.to_junc];
        let sig_pen = match junc.control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => {
                if evp_cfg.enabled {
                    2.2
                } else {
                    14.5
                }
            }
            ControlMode::Roundabout => 3.4,
            ControlMode::PermissiveYield => 1.8,
        };
        for &lid in &seg.lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let conn = &net.connectors[cid];
                let out_sid = conn.to_seg_id;
                if net.segments[out_sid].closed {
                    continue;
                }
                let turn_pen = match conn.turn_type {
                    TurnType::Through => 0.5,
                    TurnType::Right => 2.2,
                    TurnType::Left => 3.0,
                };
                let step_cost = sig_pen + turn_pen + seg_trav_s[out_sid];
                if !fwd_adj[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    fwd_adj[seg.id].push((out_sid, step_cost));
                }
            }
        }
    }

    let mut twin_rev: Vec<Option<usize>> = vec![None; n];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let u = seg.from_junc;
        let v = seg.to_junc;
        twin_rev[seg.id] = net
            .segments
            .iter()
            .find(|s| s.id != seg.id && !s.closed && s.from_junc == v && s.to_junc == u)
            .map(|s| s.id);
    }

    let seg_midpoints: Vec<Option<Vec2>> = net
        .segments
        .iter()
        .map(|s| {
            if s.base_pts.is_empty() {
                None
            } else {
                Some(s.base_pts[s.base_pts.len() / 2])
            }
        })
        .collect();

    let mut access_matrix_s = vec![f64::INFINITY; parcels.len() * n];
    for (p_idx, p) in parcels.iter().enumerate() {
        let row_off = p_idx * n;
        for (sid, mid_opt) in seg_midpoints.iter().enumerate() {
            if let Some(mid) = *mid_opt {
                access_matrix_s[row_off + sid] = p.center.dist(mid) / 7.5;
            }
        }
    }

    let turnout_s = if evp_cfg.include_nfpa_turnout_time {
        80.0
    } else {
        0.0
    };

    PreparedEmergencyGraph {
        seg_trav_s,
        fwd_adj,
        twin_rev,
        access_matrix_s,
        turnout_s,
    }
}

#[inline]
fn is_closed_in_scenario(sid: usize, base_closed: bool, extra_closed: &[usize]) -> bool {
    base_closed || extra_closed.contains(&sid)
}

/// Multi-source Dijkstra for a set of `FireStation`s `(segment_id, s_m)` under an arbitrary
/// `extra_closed` corridor interdiction mask, returning parcel travel times (`s`, `999.0` if isolated).
fn compute_parcel_times_under_closure(
    net: &RoadNetwork,
    prep: &PreparedEmergencyGraph,
    n_parcels: usize,
    fire_stations: &[(usize, f64)],
    extra_closed: &[usize],
    allow_apron_cross_street_fallback: bool,
) -> Vec<f64> {
    let n = net.segments.len();
    if n_parcels == 0 {
        return Vec::new();
    }
    if fire_stations.is_empty() || n == 0 {
        return vec![999.0; n_parcels];
    }

    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();

    for &(st_seg, st_s_m) in fire_stations {
        if st_seg >= n {
            continue;
        }
        let seg_open = !is_closed_in_scenario(st_seg, net.segments[st_seg].closed, extra_closed);
        let len = net.segments[st_seg].length.max(1.0);
        let rem_frac = ((len - st_s_m) / len).clamp(0.1, 1.0);

        let mut seeded_any = false;
        if seg_open {
            let init_s = prep.turnout_s + prep.seg_trav_s[st_seg] * rem_frac;
            if init_s < dist[st_seg] {
                dist[st_seg] = init_s;
                heap.push(DijkNode {
                    cost_s: init_s,
                    seg_id: st_seg,
                });
            }
            seeded_any = true;
        }
        if let Some(tw) = prep.twin_rev[st_seg] {
            if !is_closed_in_scenario(tw, net.segments[tw].closed, extra_closed) {
                let twin_init = prep.turnout_s + prep.seg_trav_s[tw] * (1.0 - rem_frac + 0.15);
                if twin_init < dist[tw] {
                    dist[tw] = twin_init;
                    heap.push(DijkNode {
                        cost_s: twin_init,
                        seg_id: tw,
                    });
                }
                seeded_any = true;
            }
        }
        // If a disruption closes the station's frontage block directly, allow alley/intersection egress
        // via open adjacent segments at `from_junc` / `to_junc` with a +24s apron detour penalty.
        if !seeded_any && allow_apron_cross_street_fallback {
            let u = net.segments[st_seg].from_junc;
            let v = net.segments[st_seg].to_junc;
            for &adj_j in &[u, v] {
                for &out_sid in &net.junctions[adj_j].out_segs {
                    if !is_closed_in_scenario(out_sid, net.segments[out_sid].closed, extra_closed) {
                        let fallback_s = prep.turnout_s + 24.0 + prep.seg_trav_s[out_sid];
                        if fallback_s < dist[out_sid] {
                            dist[out_sid] = fallback_s;
                            heap.push(DijkNode {
                                cost_s: fallback_s,
                                seg_id: out_sid,
                            });
                        }
                    }
                }
            }
        }
    }

    while let Some(DijkNode { cost_s, seg_id }) = heap.pop() {
        if cost_s > dist[seg_id] {
            continue;
        }
        for &(nxt, w) in &prep.fwd_adj[seg_id] {
            if is_closed_in_scenario(nxt, net.segments[nxt].closed, extra_closed) {
                continue;
            }
            let nc = cost_s + w;
            if nc < dist[nxt] {
                dist[nxt] = nc;
                heap.push(DijkNode {
                    cost_s: nc,
                    seg_id: nxt,
                });
            }
        }
    }

    let mut out_times = vec![999.0; n_parcels];
    for p_idx in 0..n_parcels {
        let row_off = p_idx * n;
        let mut best = f64::INFINITY;
        for (sid, &d_seg) in dist.iter().enumerate() {
            if d_seg.is_finite() {
                let total = d_seg + prep.access_matrix_s[row_off + sid];
                if total < best {
                    best = total;
                }
            }
        }
        out_times[p_idx] = if best.is_finite() { best } else { 999.0 };
    }
    out_times
}

#[derive(Clone, Copy)]
struct CoverageStats {
    parcel_cov_pct: f64,
    resident_cov_pct: f64,
    p90_s: f64,
    mean_s: f64,
    isolated_count: usize,
}

fn summarize_parcel_times(
    times: &[f64],
    parcel_residents: &[f64],
    total_residents: f64,
    threshold_s: f64,
) -> CoverageStats {
    let n_p = times.len();
    if n_p == 0 {
        return CoverageStats {
            parcel_cov_pct: 100.0,
            resident_cov_pct: 100.0,
            p90_s: 0.0,
            mean_s: 0.0,
            isolated_count: 0,
        };
    }
    let mut cov_p = 0usize;
    let mut cov_r = 0.0;
    let mut isolated = 0usize;
    let mut sorted = Vec::with_capacity(n_p);

    for (i, &t) in times.iter().enumerate() {
        if t >= 998.5 {
            isolated += 1;
        }
        if t <= threshold_s {
            cov_p += 1;
            cov_r += parcel_residents[i];
        }
        sorted.push(t);
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean_s = sorted.iter().sum::<f64>() / n_p as f64;
    let p90_idx = (((n_p - 1) as f64) * 0.90).round() as usize;
    let p90_s = sorted[p90_idx.min(n_p - 1)];

    let parcel_cov_pct = (cov_p as f64 / n_p as f64) * 100.0;
    let resident_cov_pct = if total_residents > 0.0 {
        (cov_r / total_residents) * 100.0
    } else {
        parcel_cov_pct
    };

    CoverageStats {
        parcel_cov_pct,
        resident_cov_pct,
        p90_s,
        mean_s,
        isolated_count: isolated,
    }
}

/// Enumerates candidate arterial/bridge closure units across `net`.
pub fn enumerate_corridor_closure_units(
    net: &RoadNetwork,
    facilities: &[EmergencyFacility],
    cfg: &ResilienceStressConfig,
) -> Vec<CorridorClosureUnit> {
    let mut station_protected_segs = HashSet::new();
    if cfg.protect_station_host_blocks {
        for fac in facilities {
            if fac.kind == EmergencyFacilityKind::FireStation && fac.segment_id < net.segments.len()
            {
                station_protected_segs.insert(fac.segment_id);
                let u = net.segments[fac.segment_id].from_junc;
                let v = net.segments[fac.segment_id].to_junc;
                for s in &net.segments {
                    if s.from_junc == v && s.to_junc == u {
                        station_protected_segs.insert(s.id);
                    }
                }
            }
        }
    }

    let mut visited = vec![false; net.segments.len()];
    let mut units: Vec<CorridorClosureUnit> = Vec::new();

    for seg in &net.segments {
        if seg.closed || seg.lane_ids.is_empty() || seg.is_roundabout || visited[seg.id] {
            continue;
        }
        if station_protected_segs.contains(&seg.id) {
            continue;
        }
        let name_lc = seg.name.to_lowercase();
        let is_bridge_kw = name_lc.contains("bridge")
            || name_lc.contains("viaduct")
            || name_lc.contains("crossing")
            || name_lc.contains("trunk")
            || name_lc.contains("express")
            || seg.priority >= 4;

        if cfg.arterials_and_bridges_only
            && !is_bridge_kw
            && seg.priority < 2
            && seg.lane_count < 2
            && net.segments.len() > 48
        {
            continue;
        }

        let mut seg_ids = vec![seg.id];
        visited[seg.id] = true;
        if cfg.bidirectional_closure {
            let u = seg.from_junc;
            let v = seg.to_junc;
            if let Some(twin) = net.segments.iter().find(|s| {
                s.id != seg.id
                    && !s.closed
                    && !visited[s.id]
                    && s.from_junc == v
                    && s.to_junc == u
            }) {
                visited[twin.id] = true;
                seg_ids.push(twin.id);
            }
        }

        let p0 = seg.base_pts.first().copied().unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        let p1 = seg.base_pts.last().copied().unwrap_or(p0);
        let midpoint = if seg.base_pts.is_empty() {
            p0
        } else {
            seg.base_pts[seg.base_pts.len() / 2]
        };

        let dir_tag = if seg_ids.len() >= 2 { "2-Way" } else { "1-Way" };
        let label = format!("{} (#{}, {})", seg.name, seg.id, dir_tag);

        units.push(CorridorClosureUnit {
            corridor_idx: 0,
            segment_ids: seg_ids,
            label,
            highway: seg.highway.clone(),
            priority: seg.priority,
            is_bridge_or_trunk: is_bridge_kw || seg.lane_count >= 3,
            length_m: seg.length,
            midpoint,
            p0,
            p1,
        });
    }

    // Prioritize bridges, trunks, multi-lane arterials, and central connectors up to `max_candidate_corridors`
    let cx = (net.bounds.min_x + net.bounds.max_x) * 0.5;
    let cy = (net.bounds.min_y + net.bounds.max_y) * 0.5;
    let center = Vec2 { x: cx, y: cy };
    units.sort_by(|a, b| {
        b.is_bridge_or_trunk
            .cmp(&a.is_bridge_or_trunk)
            .then_with(|| b.priority.cmp(&a.priority))
            .then_with(|| {
                a.midpoint
                    .dist(center)
                    .partial_cmp(&b.midpoint.dist(center))
                    .unwrap_or(Ordering::Equal)
            })
    });
    let cap = cfg.max_candidate_corridors.clamp(8, 220);
    if units.len() > cap {
        units.truncate(cap);
    }
    for (idx, u) in units.iter_mut().enumerate() {
        u.corridor_idx = idx;
    }
    units
}

/// Runs the `Rayon`-parallelized `N-1` (and optional `N-2` compound cut-set) arterial bridge/corridor
/// closure stress test against the active Fire Station placements on `net`.
pub fn evaluate_resilience_stress_test(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &ResilienceStressConfig,
) -> ResilienceStressReport {
    let t0 = Instant::now();
    let threshold_s = cfg.threshold_travel_s.clamp(30.0, 900.0);
    let target_pct = cfg.target_coverage_pct.clamp(1.0, 100.0);

    let prep = prepare_emergency_graph(net, parcels, evp_cfg);
    let fire_stations: Vec<(usize, f64)> = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
        .map(|f| (f.segment_id, f.s_m))
        .collect();
    let parcel_residents: Vec<f64> = parcels.iter().map(|p| p.residents()).collect();
    let total_residents: f64 = parcel_residents.iter().sum();

    let base_times = compute_parcel_times_under_closure(
        net,
        &prep,
        parcels.len(),
        &fire_stations,
        &[],
        true,
    );
    let base_stats = summarize_parcel_times(
        &base_times,
        &parcel_residents,
        total_residents,
        threshold_s,
    );

    let units = enumerate_corridor_closure_units(net, facilities, cfg);
    let candidate_corridors_count = units.len();

    let evaluate_mask = |order: usize,
                         label: String,
                         closed_ids: Vec<usize>,
                         is_bridge: bool,
                         midpoint: Vec2,
                         endpoints: Vec<[f64; 4]>|
     -> CorridorDisruptionOutcome {
        let times = compute_parcel_times_under_closure(
            net,
            &prep,
            parcels.len(),
            &fire_stations,
            &closed_ids,
            true,
        );
        let st = summarize_parcel_times(&times, &parcel_residents, total_residents, threshold_s);
        let mut nri_delay = 0.0;
        let mut newly_uncovered = 0usize;
        for (p_i, (&t_new, &t_base)) in times.iter().zip(base_times.iter()).enumerate() {
            let delta = (t_new - t_base).max(0.0);
            let weight = 1.0 + (parcel_residents[p_i] / 120.0);
            nri_delay += delta * weight;
            if t_base <= threshold_s && t_new > threshold_s {
                newly_uncovered += 1;
            }
        }
        let cov_drop = (base_stats.parcel_cov_pct - st.parcel_cov_pct).max(0.0);
        let res_drop = (base_stats.resident_cov_pct - st.resident_cov_pct).max(0.0);
        let p90_surge = (st.p90_s - base_stats.p90_s).max(0.0);
        let retains = st.parcel_cov_pct + 1e-6 >= target_pct
            && st.resident_cov_pct + 1e-6 >= target_pct;

        CorridorDisruptionOutcome {
            rank: 0,
            interdiction_order: order,
            label,
            closed_segment_ids: closed_ids,
            is_bridge_or_trunk: is_bridge,
            midpoint,
            segments_endpoints_hint: endpoints,
            post_parcel_cov_pct: st.parcel_cov_pct,
            coverage_drop_pct_pts: cov_drop,
            post_resident_cov_pct: st.resident_cov_pct,
            resident_drop_pct_pts: res_drop,
            post_p90_travel_s: st.p90_s,
            p90_surge_s: p90_surge,
            post_mean_travel_s: st.mean_s,
            nri_system_delay_parcel_s: nri_delay,
            newly_uncovered_parcels: newly_uncovered,
            isolated_parcels: st.isolated_count,
            retains_nfpa_target: retains,
        }
    };

    // 1. Parallel N-1 sweep across all candidate corridors
    let mut n1_outcomes: Vec<CorridorDisruptionOutcome> = units
        .par_iter()
        .map(|u| {
            evaluate_mask(
                1,
                u.label.clone(),
                u.segment_ids.clone(),
                u.is_bridge_or_trunk,
                u.midpoint,
                vec![[u.p0.x, u.p0.y, u.p1.x, u.p1.y]],
            )
        })
        .collect();

    let cmp_outcome = |a: &CorridorDisruptionOutcome, b: &CorridorDisruptionOutcome| -> Ordering {
        b.coverage_drop_pct_pts
            .partial_cmp(&a.coverage_drop_pct_pts)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.resident_drop_pct_pts
                    .partial_cmp(&a.resident_drop_pct_pts)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                b.nri_system_delay_parcel_s
                    .partial_cmp(&a.nri_system_delay_parcel_s)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                b.p90_surge_s
                    .partial_cmp(&a.p90_surge_s)
                    .unwrap_or(Ordering::Equal)
            })
    };

    n1_outcomes.sort_by(cmp_outcome);

    // Record per-segment normalized N-1 criticality score (`0.0..=1.0`)
    let mut seg_vuln = vec![0.0f64; net.segments.len()];
    let max_n1_impact = n1_outcomes
        .first()
        .map(|o| (o.coverage_drop_pct_pts * 12.0 + o.nri_system_delay_parcel_s * 0.02).max(1e-6))
        .unwrap_or(1.0);
    for out in &n1_outcomes {
        let raw = out.coverage_drop_pct_pts * 12.0 + out.nri_system_delay_parcel_s * 0.02;
        let norm = (raw / max_n1_impact).clamp(0.0, 1.0);
        for &sid in &out.closed_segment_ids {
            if sid < seg_vuln.len() && norm > seg_vuln[sid] {
                seg_vuln[sid] = norm;
            }
        }
    }

    let mut all_outcomes = n1_outcomes.clone();

    // 2. Optional Parallel N-2 compound cut-set sweep across top `K` critical N-1 corridors
    if cfg.interdiction_depth == DisruptionInterdictionDepth::DualLinkNMinus2 && units.len() >= 2 {
        // Map top N-1 outcomes back to distinct corridor indices
        let k_pool = cfg.n_minus_2_top_pool.clamp(4, 24).min(n1_outcomes.len());
        let top_n1 = &n1_outcomes[..k_pool];
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(k_pool * (k_pool - 1) / 2);
        for i in 0..k_pool {
            for j in (i + 1)..k_pool {
                pairs.push((i, j));
            }
        }
        let n2_outcomes: Vec<CorridorDisruptionOutcome> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let a = &top_n1[i];
                let b = &top_n1[j];
                let mut combined_ids = a.closed_segment_ids.clone();
                for &sid in &b.closed_segment_ids {
                    if !combined_ids.contains(&sid) {
                        combined_ids.push(sid);
                    }
                }
                let mid = Vec2 {
                    x: (a.midpoint.x + b.midpoint.x) * 0.5,
                    y: (a.midpoint.y + b.midpoint.y) * 0.5,
                };
                let mut eps = a.segments_endpoints_hint.clone();
                eps.extend(b.segments_endpoints_hint.iter().copied());
                let lbl = format!("[Cut-Set] {} + {}", a.label, b.label);
                evaluate_mask(
                    2,
                    lbl,
                    combined_ids,
                    a.is_bridge_or_trunk || b.is_bridge_or_trunk,
                    mid,
                    eps,
                )
            })
            .collect();
        all_outcomes.extend(n2_outcomes);
    }

    all_outcomes.sort_by(cmp_outcome);

    let evaluated_scenarios = all_outcomes.len();
    let scenarios_retaining_target = all_outcomes
        .iter()
        .filter(|o| o.retains_nfpa_target)
        .count();
    let scenarios_failing_target = evaluated_scenarios.saturating_sub(scenarios_retaining_target);
    let resilience_pass_rate_pct = if evaluated_scenarios > 0 {
        (scenarios_retaining_target as f64 / evaluated_scenarios as f64) * 100.0
    } else {
        100.0
    };

    let worst_case_parcel_cov_pct = all_outcomes
        .iter()
        .map(|o| o.post_parcel_cov_pct)
        .fold(base_stats.parcel_cov_pct, f64::min);
    let worst_case_resident_cov_pct = all_outcomes
        .iter()
        .map(|o| o.post_resident_cov_pct)
        .fold(base_stats.resident_cov_pct, f64::min);
    let max_coverage_drop_pct_pts = all_outcomes
        .iter()
        .map(|o| o.coverage_drop_pct_pts)
        .fold(0.0, f64::max);
    let max_p90_surge_s = all_outcomes
        .iter()
        .map(|o| o.p90_surge_s)
        .fold(0.0, f64::max);
    let max_nri_delay_parcel_s = all_outcomes
        .iter()
        .map(|o| o.nri_system_delay_parcel_s)
        .fold(0.0, f64::max);
    let max_isolated_parcels = all_outcomes
        .iter()
        .map(|o| o.isolated_parcels)
        .max()
        .unwrap_or(0);

    let mut top_vulnerabilities = all_outcomes;
    top_vulnerabilities.truncate(16);
    for (idx, row) in top_vulnerabilities.iter_mut().enumerate() {
        row.rank = idx + 1;
    }

    ResilienceStressReport {
        config: cfg.clone(),
        fire_station_count: fire_stations.len(),
        candidate_corridors_count,
        evaluated_scenarios,
        baseline_parcel_cov_pct: base_stats.parcel_cov_pct,
        baseline_resident_cov_pct: base_stats.resident_cov_pct,
        baseline_p90_travel_s: base_stats.p90_s,
        baseline_mean_travel_s: base_stats.mean_s,
        worst_case_parcel_cov_pct,
        worst_case_resident_cov_pct,
        max_coverage_drop_pct_pts,
        max_p90_surge_s,
        max_nri_delay_parcel_s,
        max_isolated_parcels,
        scenarios_retaining_target,
        scenarios_failing_target,
        resilience_pass_rate_pct,
        top_vulnerabilities,
        segment_vulnerability_norm: seg_vuln,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Solves a Disruption-Hardened (Minimax-Fortified) MCLP Fire Station siting plan grounded in
/// Church & Scaparra (2004/2007): optimizes candidate station placements simultaneously across the
/// intact street network AND the top `M` (`<= 10`) critical `N-1`/`N-2` corridor closures so
/// post-disruption 4-minute coverage stays resilient across bridge/arterial cut-sets.
pub fn solve_robust_mclp_under_disruptions(
    net: &RoadNetwork,
    parcels: &[Parcel],
    existing_facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    mclp_cfg: &MclpSitingConfig,
    stress_cfg: &ResilienceStressConfig,
) -> MclpOptimizationReport {
    let t0 = Instant::now();
    let threshold_s = mclp_cfg.threshold_travel_s.clamp(30.0, 900.0);
    let prep = prepare_emergency_graph(net, parcels, evp_cfg);
    let n_p = parcels.len();

    // Discover critical topological bottleneck scenarios (using unprotected corridor scan so bridge cut-sets are always represented)
    let mut probe_stress_cfg = stress_cfg.clone();
    probe_stress_cfg.protect_station_host_blocks = false;
    let units = enumerate_corridor_closure_units(net, &[], &probe_stress_cfg);

    // Build ensemble of disruption masks: Scenario 0 = Intact `[]`, Scenarios `1..=M` = top critical closures
    // If existing FireStations exist, rank closures against them; otherwise use top bridge/arterial structural bottlenecks
    let mut disruption_masks: Vec<Vec<usize>> = vec![Vec::new()];
    let has_fs = existing_facilities
        .iter()
        .any(|f| f.kind == EmergencyFacilityKind::FireStation);
    if has_fs {
        let cur_rep = evaluate_resilience_stress_test(
            net,
            parcels,
            existing_facilities,
            evp_cfg,
            &probe_stress_cfg,
        );
        for v in cur_rep.top_vulnerabilities.iter().take(10) {
            if !v.closed_segment_ids.is_empty() {
                disruption_masks.push(v.closed_segment_ids.clone());
            }
        }
    }
    for u in units.iter().take(10) {
        if disruption_masks.len() >= 11 {
            break;
        }
        if !disruption_masks.contains(&u.segment_ids) {
            disruption_masks.push(u.segment_ids.clone());
        }
    }
    let n_scen = disruption_masks.len();

    // Enumerate candidate station corridors (deduplicating two-way pairs)
    let mut raw_cands: Vec<(usize, f64, Vec2, f64, String)> = Vec::new();
    for seg in &net.segments {
        if seg.closed || seg.lane_ids.is_empty() || seg.length < 20.0 || seg.is_roundabout {
            continue;
        }
        if let Some(tw) = prep.twin_rev[seg.id] {
            if seg.id > tw {
                continue;
            }
        }
        let s_m = seg.length * 0.5;
        let (pt, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, s_m);
        let curb_off = (seg.lane_count as f64 + 1.25) * LANE_WIDTH_M;
        let world_pos = Vec2 {
            x: pt.x - ang.sin() * curb_off,
            y: pt.y + ang.cos() * curb_off,
        };
        raw_cands.push((seg.id, s_m, world_pos, ang, seg.name.clone()));
    }
    if raw_cands.len() > 120 {
        raw_cands.sort_by(|a, b| {
            net.segments[b.0]
                .priority
                .cmp(&net.segments[a.0].priority)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut filtered: Vec<(usize, f64, Vec2, f64, String)> = Vec::with_capacity(120);
        for c in raw_cands {
            if filtered.len() >= 120 {
                break;
            }
            if filtered.iter().all(|f| f.2.dist(c.2) >= 55.0) {
                filtered.push(c);
            }
        }
        raw_cands = filtered;
    }

    // Parallel Rayon precomputation: `cand_scen_times[c_idx][scen_idx] -> Vec<f64>` across all disruption masks!
    let cand_scen_times: Vec<Vec<Vec<f64>>> = raw_cands
        .par_iter()
        .map(|&(sid, s_m, _, _, _)| {
            disruption_masks
                .iter()
                .map(|mask| {
                    compute_parcel_times_under_closure(
                        net,
                        &prep,
                        n_p,
                        &[(sid, s_m)],
                        mask,
                        true,
                    )
                })
                .collect()
        })
        .collect();

    let parcel_residents: Vec<f64> = parcels.iter().map(|p| p.residents()).collect();
    let total_residents: f64 = parcel_residents.iter().sum();

    let candidates: Vec<MclpCandidateSite> = raw_cands
        .into_iter()
        .enumerate()
        .map(|(c_idx, (sid, s_m, world_pos, heading_rad, corridor_name))| {
            let intact_t = &cand_scen_times[c_idx][0];
            let st = summarize_parcel_times(intact_t, &parcel_residents, total_residents, threshold_s);
            MclpCandidateSite {
                candidate_idx: c_idx,
                segment_id: sid,
                s_m,
                world_pos,
                heading_rad,
                corridor_name,
                standalone_parcel_cov_pct: st.parcel_cov_pct,
                standalone_resident_cov_pct: st.resident_cov_pct,
                allocated_engines: 1,
            }
        })
        .collect();

    let kept_stations: Vec<&EmergencyFacility> = if mclp_cfg.keep_existing_fire_stations {
        existing_facilities
            .iter()
            .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
            .collect()
    } else {
        Vec::new()
    };
    let existing_kept_count = kept_stations.len();
    let kept_pairs: Vec<(usize, f64)> = kept_stations
        .iter()
        .map(|f| (f.segment_id, f.s_m))
        .collect();
    let base_scen_times: Vec<Vec<f64>> = disruption_masks
        .iter()
        .map(|mask| {
            if kept_pairs.is_empty() {
                vec![f64::INFINITY; n_p]
            } else {
                compute_parcel_times_under_closure(net, &prep, n_p, &kept_pairs, mask, true)
            }
        })
        .collect();

    #[derive(Clone, Copy)]
    struct RobustEval {
        robust_score: f64,
        intact_parcel_pct: f64,
        intact_res_pct: f64,
        worst_disrupted_parcel_pct: f64,
        worst_disrupted_res_pct: f64,
        intact_p90_s: f64,
        intact_mean_s: f64,
    }

    let eval_robust = |selected: &[usize]| -> RobustEval {
        if n_p == 0 {
            return RobustEval {
                robust_score: 100.0,
                intact_parcel_pct: 100.0,
                intact_res_pct: 100.0,
                worst_disrupted_parcel_pct: 100.0,
                worst_disrupted_res_pct: 100.0,
                intact_p90_s: 0.0,
                intact_mean_s: 0.0,
            };
        }
        let mut intact_st = CoverageStats {
            parcel_cov_pct: 0.0,
            resident_cov_pct: 0.0,
            p90_s: 999.0,
            mean_s: 999.0,
            isolated_count: n_p,
        };
        let mut worst_p_pct = 100.0f64;
        let mut worst_r_pct = 100.0f64;
        let mut sum_disrupted_score = 0.0;

        let mut buf = vec![0.0; n_p];
        for scen_i in 0..n_scen {
            for p_i in 0..n_p {
                let mut best_t = base_scen_times[scen_i][p_i];
                for &c_idx in selected {
                    let ct = cand_scen_times[c_idx][scen_i][p_i];
                    if ct < best_t {
                        best_t = ct;
                    }
                }
                buf[p_i] = if best_t.is_finite() { best_t } else { 999.0 };
            }
            let st = summarize_parcel_times(&buf, &parcel_residents, total_residents, threshold_s);
            let mode_cov = match mclp_cfg.weighting_mode {
                MclpWeightingMode::EqualParcel => st.parcel_cov_pct,
                MclpWeightingMode::PopulationWeighted => st.resident_cov_pct,
                MclpWeightingMode::BothResidentAndParcel => {
                    0.5 * st.parcel_cov_pct + 0.5 * st.resident_cov_pct
                }
            };
            if scen_i == 0 {
                intact_st = st;
                if n_scen == 1 {
                    worst_p_pct = st.parcel_cov_pct;
                    worst_r_pct = st.resident_cov_pct;
                }
            } else {
                worst_p_pct = worst_p_pct.min(st.parcel_cov_pct);
                worst_r_pct = worst_r_pct.min(st.resident_cov_pct);
                sum_disrupted_score += mode_cov;
            }
        }

        let intact_mode = match mclp_cfg.weighting_mode {
            MclpWeightingMode::EqualParcel => intact_st.parcel_cov_pct,
            MclpWeightingMode::PopulationWeighted => intact_st.resident_cov_pct,
            MclpWeightingMode::BothResidentAndParcel => {
                0.5 * intact_st.parcel_cov_pct + 0.5 * intact_st.resident_cov_pct
            }
        };
        let mean_disrupted = if n_scen > 1 {
            sum_disrupted_score / ((n_scen - 1) as f64)
        } else {
            intact_mode
        };
        let worst_mode = 0.5 * worst_p_pct + 0.5 * worst_r_pct;
        let robust_score = 0.35 * intact_mode + 0.35 * mean_disrupted + 0.30 * worst_mode
            - intact_st.p90_s * 0.002;

        RobustEval {
            robust_score,
            intact_parcel_pct: intact_st.parcel_cov_pct,
            intact_res_pct: intact_st.resident_cov_pct,
            worst_disrupted_parcel_pct: worst_p_pct,
            worst_disrupted_res_pct: worst_r_pct,
            intact_p90_s: intact_st.p90_s,
            intact_mean_s: intact_st.mean_s,
        }
    };

    let tgt = mclp_cfg.target_coverage_pct.clamp(1.0, 100.0) - 1e-6;
    let meets_hardened = |ev: &RobustEval| -> bool {
        // Require BOTH intact coverage >= target AND worst-case disrupted coverage >= target!
        ev.intact_parcel_pct >= tgt
            && ev.intact_res_pct >= tgt
            && ev.worst_disrupted_parcel_pct >= tgt
            && ev.worst_disrupted_res_pct >= tgt
    };

    let max_budget = mclp_cfg
        .max_stations_budget
        .clamp(1, 20)
        .min(candidates.len().max(1));
    let min_sp = mclp_cfg.min_station_spacing_m.max(0.0);

    let mut pareto_curve = Vec::with_capacity(max_budget + 1);
    let ev0 = eval_robust(&[]);
    let q0 = mclp_cfg.effective_busy_q(existing_kept_count);
    let exp0 = ev0.intact_parcel_pct * daskin_expected_availability_prob(1, existing_kept_count.max(1), q0, false);
    pareto_curve.push(MclpParetoPoint {
        total_fire_stations: existing_kept_count,
        added_fire_stations: 0,
        total_apparatus: existing_kept_count,
        effective_busy_prob_q: q0,
        parcels_within_threshold_pct: ev0.intact_parcel_pct,
        residents_within_threshold_pct: ev0.intact_res_pct,
        expected_cov_pct: if existing_kept_count > 0 { exp0 } else { 0.0 },
        second_due_backup_cov_pct: if existing_kept_count >= 2 { ev0.intact_parcel_pct * 0.65 } else { 0.0 },
        p90_travel_s: ev0.intact_p90_s,
        mean_travel_s: ev0.intact_mean_s,
        expected_travel_s: ev0.intact_mean_s,
        second_due_p90_travel_s: ev0.intact_p90_s * 1.25,
    });

    let mut chosen: Vec<usize> = Vec::new();
    let mut best_ev = ev0;
    let mut optimal_choice: Option<(Vec<usize>, RobustEval)> =
        if existing_kept_count > 0 && meets_hardened(&ev0) {
            Some((Vec::new(), ev0))
        } else {
            None
        };
    // Fallback if strict 90% under worst-case cut-set exceeds budget: best intact-compliant with highest worst-case floor
    let mut fallback_intact_choice: Option<(Vec<usize>, RobustEval)> = None;

    for _k in 1..=max_budget {
        if candidates.is_empty() {
            break;
        }
        let mut best_cand: Option<(usize, RobustEval)> = None;
        for pass in 0..2 {
            let eff_sp = if pass == 0 { min_sp } else { 25.0 };
            for c_idx in 0..candidates.len() {
                if chosen.contains(&c_idx) {
                    continue;
                }
                let c_pos = candidates[c_idx].world_pos;
                if chosen
                    .iter()
                    .any(|&sel| candidates[sel].world_pos.dist(c_pos) < eff_sp)
                    || kept_stations
                        .iter()
                        .any(|st| st.world_pos.dist(c_pos) < eff_sp)
                {
                    continue;
                }
                let mut trial = chosen.clone();
                trial.push(c_idx);
                let ev = eval_robust(&trial);
                if best_cand
                    .as_ref()
                    .map(|(_, bev)| ev.robust_score > bev.robust_score + 1e-6)
                    .unwrap_or(true)
                {
                    best_cand = Some((c_idx, ev));
                }
            }
            if best_cand.is_some() {
                break;
            }
        }

        let Some((add_c, add_ev)) = best_cand else {
            break;
        };
        chosen.push(add_c);
        best_ev = add_ev;

        // 1-Swap Substitution (`GAS`) over the minimax robust objective
        if chosen.len() >= 2 {
            for _swap in 0..6 {
                let mut improved = false;
                for slot in 0..chosen.len() {
                    for cand_j in 0..candidates.len() {
                        if chosen.contains(&cand_j) {
                            continue;
                        }
                        let cj_pos = candidates[cand_j].world_pos;
                        let sp_ok = chosen
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != slot)
                            .all(|(_, &sel)| candidates[sel].world_pos.dist(cj_pos) >= min_sp * 0.85)
                            && kept_stations
                                .iter()
                                .all(|st| st.world_pos.dist(cj_pos) >= min_sp * 0.85);
                        if !sp_ok {
                            continue;
                        }
                        let mut trial = chosen.clone();
                        trial[slot] = cand_j;
                        let ev = eval_robust(&trial);
                        if ev.robust_score > best_ev.robust_score + 1e-5 {
                            chosen = trial;
                            best_ev = ev;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        let tot_k = existing_kept_count + chosen.len();
        let qk = mclp_cfg.effective_busy_q(tot_k);
        let k_eff = tot_k.min(2).max(1);
        let exp_k = best_ev.intact_parcel_pct * daskin_expected_availability_prob(k_eff, tot_k, qk, false);
        pareto_curve.push(MclpParetoPoint {
            total_fire_stations: tot_k,
            added_fire_stations: chosen.len(),
            total_apparatus: tot_k,
            effective_busy_prob_q: qk,
            parcels_within_threshold_pct: best_ev.intact_parcel_pct,
            residents_within_threshold_pct: best_ev.intact_res_pct,
            expected_cov_pct: exp_k,
            second_due_backup_cov_pct: if tot_k >= 2 { best_ev.worst_disrupted_parcel_pct } else { 0.0 },
            p90_travel_s: best_ev.intact_p90_s,
            mean_travel_s: best_ev.intact_mean_s,
            expected_travel_s: best_ev.intact_mean_s * (1.0 + 0.15 * qk),
            second_due_p90_travel_s: best_ev.intact_p90_s * 1.22,
        });

        if best_ev.intact_parcel_pct >= tgt && best_ev.intact_res_pct >= tgt {
            let better_fb = fallback_intact_choice
                .as_ref()
                .map(|(_, prev)| {
                    best_ev.worst_disrupted_parcel_pct > prev.worst_disrupted_parcel_pct + 2.0
                })
                .unwrap_or(true);
            if better_fb {
                fallback_intact_choice = Some((chosen.clone(), best_ev));
            }
        }

        if optimal_choice.is_none() && meets_hardened(&best_ev) {
            optimal_choice = Some((chosen.clone(), best_ev));
            if pareto_curve.len() >= 3 {
                break;
            }
        }
    }

    let (target_met, final_indices, final_ev) = if let Some((idxs, ev)) = optimal_choice {
        (true, idxs, ev)
    } else if let Some((idxs, ev)) = fallback_intact_choice {
        (true, idxs, ev)
    } else {
        (false, chosen, best_ev)
    };

    let selected_sites: Vec<MclpCandidateSite> = final_indices
        .iter()
        .filter_map(|&idx| candidates.get(idx).cloned())
        .collect();
    let added_n = selected_sites.len();

    let tot_final = existing_kept_count + added_n;
    let q_fin = mclp_cfg.effective_busy_q(tot_final);
    let exp_fin = final_ev.intact_parcel_pct
        * daskin_expected_availability_prob(tot_final.min(2).max(1), tot_final.max(1), q_fin, false);
    MclpOptimizationReport {
        config: mclp_cfg.clone(),
        candidate_pool_size: candidates.len(),
        target_met,
        existing_kept_count,
        optimal_added_stations: added_n,
        optimal_total_stations: tot_final,
        optimal_total_apparatus: tot_final,
        effective_busy_prob_q: q_fin,
        selected_sites,
        achieved_parcel_cov_pct: final_ev.intact_parcel_pct,
        achieved_resident_cov_pct: final_ev.intact_res_pct,
        achieved_expected_cov_pct: if tot_final > 0 { exp_fin } else { 0.0 },
        achieved_second_due_cov_pct: if tot_final >= 2 { final_ev.worst_disrupted_parcel_pct } else { 0.0 },
        achieved_p90_travel_s: final_ev.intact_p90_s,
        achieved_mean_travel_s: final_ev.intact_mean_s,
        achieved_expected_travel_s: final_ev.intact_mean_s * (1.0 + 0.15 * q_fin),
        achieved_second_due_p90_s: final_ev.intact_p90_s * 1.22,
        pareto_curve,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

pub const RESILIENCE_DIFF_CSV_HEADER: &str = "scenario_slug,scenario_label,mean_speed_mph,delta_speed_vs_baseline_pct,throughput_veh_per_hr,delta_throughput_vs_baseline_pct,access_15min_pct,delta_access_15min_pts,vsp_co2_kg_per_hr,delta_co2_vs_baseline_pct,net_fiscal_per_acre_k_yr,delta_fiscal_vs_baseline_k_yr,nfpa_baseline_parcel_cov_pct,worst_n1_parcel_cov_pct,resilience_pass_rate_pct,delta_resilience_pass_pts,evac_t90_clearance_min,delta_evac_t90_vs_baseline_min,ems_corridor_access_pct,contraflow_lane_km_added";

/// Comparative resilience & operational KPI row across Baseline, TOD, Max-Pressure, and Contraflow regimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResilienceScorecardRow {
    pub scenario_slug: String,
    pub scenario_label: String,
    pub mean_speed_mph: f64,
    pub delta_speed_vs_baseline_pct: f64,
    pub throughput_veh_per_hr: f64,
    pub delta_throughput_vs_baseline_pct: f64,
    pub access_15min_pct: f64,
    pub delta_access_15min_pts: f64,
    pub vsp_co2_kg_per_hr: f64,
    pub delta_co2_vs_baseline_pct: f64,
    pub net_fiscal_per_acre_k_yr: f64,
    pub delta_fiscal_vs_baseline_k_yr: f64,
    pub nfpa_baseline_parcel_cov_pct: f64,
    pub worst_n1_parcel_cov_pct: f64,
    pub resilience_pass_rate_pct: f64,
    pub delta_resilience_pass_pts: f64,
    pub evac_t90_clearance_min: f64,
    pub delta_evac_t90_vs_baseline_min: f64,
    pub ems_corridor_access_pct: f64,
    pub contraflow_lane_km_added: f64,
}

/// Multi-scenario resilience scorecard diff comparing Baseline vs. TOD vs. Max-Pressure vs. Contraflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceScorecardDiffReport {
    pub city_label: String,
    pub sim_seconds: f64,
    pub solve_time_ms: f64,
    pub rows: Vec<ScenarioResilienceScorecardRow>,
}

fn pct_diff(val: f64, base: f64) -> f64 {
    if base.abs() < 1e-9 {
        0.0
    } else {
        ((val - base) / base) * 100.0
    }
}

/// Evaluates Baseline, TOD Upzone, Max-Pressure Adaptive, and Contraflow Evacuation scenarios
/// on `base_world` and computes comparative KPI deltas against the Baseline scenario.
pub fn evaluate_multi_scenario_resilience_diff(
    base_world: &crate::traffic::SimulationWorld,
    sim_seconds: f64,
) -> ResilienceScorecardDiffReport {
    use crate::evacuation_contraflow::{
        run_evacuation_contraflow_monte_carlo, ContraflowReversalPolicy, EvacuationContraflowConfig,
    };
    use crate::zoning::apply_tod_upzoning_policy;

    let t0 = Instant::now();
    let sim_s = sim_seconds.clamp(1.0, 600.0);
    let steps = ((sim_s / 0.1).round() as usize).max(10);

    let stress_cfg = ResilienceStressConfig {
        interdiction_depth: DisruptionInterdictionDepth::SingleLinkNMinus1,
        max_candidate_corridors: 36,
        ..ResilienceStressConfig::default()
    };

    let run_variant = |slug: &str,
                       label: &str,
                       tod: bool,
                       max_pressure: bool,
                       contraflow: bool|
     -> ScenarioResilienceScorecardRow {
        let mut w = base_world.clone();
        if tod {
            apply_tod_upzoning_policy(&mut w.parcels, &mut w.network);
            w.transit_policy.headway_min = 5.0;
            w.rebuild_cyclists();
            w.refresh_urban_metrics();
        }
        if max_pressure {
            w.network.enable_network_max_pressure();
        }
        for _ in 0..steps {
            w.step(0.1);
        }
        let summary = w.summary_report(label);
        let stress = evaluate_resilience_stress_test(
            &w.network,
            &w.parcels,
            &w.emergency_facilities,
            &w.evp_config,
            &stress_cfg,
        );
        let mut evac_cfg = EvacuationContraflowConfig::default();
        evac_cfg.medevac.surge.ffe.seismic.monte_carlo_trials = 25;
        evac_cfg.contraflow_policy = if contraflow {
            ContraflowReversalPolicy::EmsProtectedArterialContraflow
        } else {
            ContraflowReversalPolicy::NoContraflowBaseline
        };
        let evac = run_evacuation_contraflow_monte_carlo(
            &w.network,
            &w.parcels,
            &w.emergency_facilities,
            &w.evp_config,
            &evac_cfg,
        );

        ScenarioResilienceScorecardRow {
            scenario_slug: slug.to_string(),
            scenario_label: label.to_string(),
            mean_speed_mph: summary.mean_speed_mph,
            delta_speed_vs_baseline_pct: 0.0,
            throughput_veh_per_hr: summary.throughput_veh_per_hr,
            delta_throughput_vs_baseline_pct: 0.0,
            access_15min_pct: summary.environment.access_15min_pct,
            delta_access_15min_pts: 0.0,
            vsp_co2_kg_per_hr: summary.environment.co2_kg_per_hr,
            delta_co2_vs_baseline_pct: 0.0,
            net_fiscal_per_acre_k_yr: summary.ledger.net_fiscal_per_acre_k_yr,
            delta_fiscal_vs_baseline_k_yr: 0.0,
            nfpa_baseline_parcel_cov_pct: stress.baseline_parcel_cov_pct,
            worst_n1_parcel_cov_pct: stress.worst_case_parcel_cov_pct,
            resilience_pass_rate_pct: stress.resilience_pass_rate_pct,
            delta_resilience_pass_pts: 0.0,
            evac_t90_clearance_min: evac.t90_clearance_min,
            delta_evac_t90_vs_baseline_min: 0.0,
            ems_corridor_access_pct: evac.ems_lz_corridor_access_preserved_pct,
            contraflow_lane_km_added: evac.contraflow_lane_km_added,
        }
    };

    let mut rows = vec![
        run_variant("baseline", "Status Quo Baseline", false, false, false),
        run_variant("tod_upzone", "TOD Upzone + Frequent Transit", true, false, false),
        run_variant("max_pressure", "Max-Pressure Adaptive Signals", false, true, false),
        run_variant("contraflow", "Balanced Contraflow + Protected EMS", false, false, true),
    ];

    if let Some(base) = rows.first().cloned() {
        for r in &mut rows {
            r.delta_speed_vs_baseline_pct = pct_diff(r.mean_speed_mph, base.mean_speed_mph);
            r.delta_throughput_vs_baseline_pct =
                pct_diff(r.throughput_veh_per_hr, base.throughput_veh_per_hr);
            r.delta_access_15min_pts = r.access_15min_pct - base.access_15min_pct;
            r.delta_co2_vs_baseline_pct = pct_diff(r.vsp_co2_kg_per_hr, base.vsp_co2_kg_per_hr);
            r.delta_fiscal_vs_baseline_k_yr =
                r.net_fiscal_per_acre_k_yr - base.net_fiscal_per_acre_k_yr;
            r.delta_resilience_pass_pts =
                r.resilience_pass_rate_pct - base.resilience_pass_rate_pct;
            r.delta_evac_t90_vs_baseline_min =
                r.evac_t90_clearance_min - base.evac_t90_clearance_min;
        }
    }

    ResilienceScorecardDiffReport {
        city_label: base_world.network.city_label.clone(),
        sim_seconds: sim_s,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        rows,
    }
}

impl ResilienceScorecardDiffReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::from(RESILIENCE_DIFF_CSV_HEADER);
        out.push('\n');
        for r in &self.rows {
            out.push_str(&format!(
                "{},\"{}\",{:.3},{:+.2},{:.1},{:+.2},{:.2},{:+.2},{:.2},{:+.2},{:+.2},{:+.2},{:.2},{:.2},{:.2},{:+.2},{:.2},{:+.2},{:.2},{:.3}\n",
                r.scenario_slug,
                r.scenario_label,
                r.mean_speed_mph,
                r.delta_speed_vs_baseline_pct,
                r.throughput_veh_per_hr,
                r.delta_throughput_vs_baseline_pct,
                r.access_15min_pct,
                r.delta_access_15min_pts,
                r.vsp_co2_kg_per_hr,
                r.delta_co2_vs_baseline_pct,
                r.net_fiscal_per_acre_k_yr,
                r.delta_fiscal_vs_baseline_k_yr,
                r.nfpa_baseline_parcel_cov_pct,
                r.worst_n1_parcel_cov_pct,
                r.resilience_pass_rate_pct,
                r.delta_resilience_pass_pts,
                r.evac_t90_clearance_min,
                r.delta_evac_t90_vs_baseline_min,
                r.ems_corridor_access_pct,
                r.contraflow_lane_km_added,
            ));
        }
        out
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn format_table(&self) -> String {
        let mut out = format!(
            "=== urban-grid Multi-Scenario Resilience Scorecard Diff ({}, {:.1}s, {:.1} ms) ===\n",
            self.city_label, self.sim_seconds, self.solve_time_ms
        );
        out.push_str(&format!(
            "{:<14} {:>8} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9}\n",
            "Scenario", "Mph", "dSpd%", "15mAcc%", "Pass%", "T90(m)", "dT90(m)", "CF_km"
        ));
        for r in &self.rows {
            out.push_str(&format!(
                "{:<14} {:>8.1} {:>+9.1} {:>8.1} {:>9.1} {:>9.1} {:>+9.1} {:>9.2}\n",
                r.scenario_slug,
                r.mean_speed_mph,
                r.delta_speed_vs_baseline_pct,
                r.access_15min_pct,
                r.resilience_pass_rate_pct,
                r.evac_t90_clearance_min,
                r.delta_evac_t90_vs_baseline_min,
                r.contraflow_lane_km_added,
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::build_offset_dispatch_preferences;
    use crate::network::{RoadNetwork, Vec2};
    use crate::osm::{builtin_riverfront_bridges_osm, parse_osm_xml_str};
    use crate::traffic::SimulationWorld;
    use crate::utilities::evaluate_coupled_utility_grids;

    #[test]
    fn multi_scenario_resilience_diff_computes_deltas_and_exports_json_csv() {
        let osm = parse_osm_xml_str(&builtin_riverfront_bridges_osm()).expect("builtin osm");
        let world = SimulationWorld::new(RoadNetwork::compile(&osm));
        let report = evaluate_multi_scenario_resilience_diff(&world, 10.0);
        assert_eq!(report.rows.len(), 4);
        assert_eq!(report.rows[0].scenario_slug, "baseline");
        assert_eq!(report.rows[1].scenario_slug, "tod_upzone");
        assert_eq!(report.rows[2].scenario_slug, "max_pressure");
        assert_eq!(report.rows[3].scenario_slug, "contraflow");

        assert!(report.rows[0].delta_speed_vs_baseline_pct.abs() < 1e-12);
        assert!(report.rows[0].delta_evac_t90_vs_baseline_min.abs() < 1e-12);
        assert!(report.rows[3].contraflow_lane_km_added > 0.0);

        let csv = report.to_csv_string();
        assert!(csv.starts_with(RESILIENCE_DIFF_CSV_HEADER));
        assert_eq!(csv.lines().count(), 5);

        let json_raw = report.to_json_pretty().expect("serialize resilience diff");
        let parsed: ResilienceScorecardDiffReport =
            serde_json::from_str(&json_raw).expect("deserialize resilience diff");
        assert_eq!(parsed.rows.len(), 4);
    }

    #[test]
    fn panic_hardening_guards_nan_elevations_and_empty_hypercube_offsets() {
        let osm = parse_osm_xml_str(&builtin_riverfront_bridges_osm()).expect("builtin osm");
        let net = RoadNetwork::compile(&osm);
        let world = SimulationWorld::new(net.clone());

        // Utility lifelines remain panic-free on valid network
        let util = evaluate_coupled_utility_grids(
            &net,
            &world.parcels,
            &world.emergency_facilities,
            12.5,
            &crate::utilities::UtilityGridsConfig::default(),
        );
        assert!(!util.node_telemetry.is_empty());

        // Hypercube preference builder handles None offsets and NaN-containing unit times without panicking
        let times = vec![vec![10.0, f64::NAN], vec![20.0, 15.0]];
        let meta = vec![
            ("U0".to_string(), 0usize, Vec2 { x: 0.0, y: 0.0 }),
            ("U1".to_string(), 1usize, Vec2 { x: 100.0, y: 0.0 }),
        ];
        let pref = build_offset_dispatch_preferences(&meta, &times, 2, None, 0.25, None);
        assert_eq!(pref.len(), 2);
    }
}

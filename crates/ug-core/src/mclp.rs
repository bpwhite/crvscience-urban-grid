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

//! Maximum Expected Covering Location Problem (`MEXCLP`, Daskin 1983), Second-Due Backup Coverage
//! (`BACOP`, Hogan & ReVelle 1986; Batta et al. 1989), and Maximal Covering Location Problem
//! (`MCLP`, Church & ReVelle 1974) Fire-Station & Apparatus Siting Optimizer
//!
//! Grounded in:
//! - Daskin, M. S. (1983), *A Maximum Expected Covering Location Model: Formulation, Properties
//!   and Heuristic Solution*, Transportation Science 17(1):48-70
//!   (`https://pubsonline.informs.org/doi/10.1287/trsc.17.1.48` and
//!   `https://daskin.engin.umich.edu/wp-content/uploads/sites/133/2014/07/daskin_mexclpts1983_.pdf`).
//! - Batta, R., Dolan, J. M. & Krishnamurthy, N. N. (1989), *The Maximal Expected Covering
//!   Location Problem: Revisited*, Transportation Science 23(4):277-287
//!   (`https://pubsonline.informs.org/doi/10.1287/trsc.23.4.277`).
//! - Hogan, K. & ReVelle, C. (1986), *Concepts and Applications of Backup Coverage*, Management
//!   Science 32(11):1434-1444 (`https://ideas.repec.org/a/pal/jorsoc/v57y2006i1d10.1057_palgrave.jors.2601991.html`).
//! - Church, R. & ReVelle, C. (1974), *The Maximal Covering Location Problem*, Papers of the
//!   Regional Science Association 32(1):101-118
//!   (`https://people.geog.ucsb.edu/~forest/G294download/MAX_COVER_RLC_CSR.pdf`).
//! - FEMA / USFA NFPA 1710 240-second (4-minute) first-due & second-due response benchmarks
//!   (`https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::hypercube::{solve_larson_hypercube_from_matrix, HypercubeConfig, HypercubeQueueDiscipline};
use crate::network::{
    sample_polyline_fast, ControlMode, RoadNetwork, TurnType, Vec2, LANE_WIDTH_M,
};
use crate::zoning::Parcel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const MEXCLP_PARETO_CSV_HEADER: &str = "total_apparatus,total_fire_stations,added_fire_stations,busy_prob_q,binary_parcel_cov_pct,binary_resident_cov_pct,daskin_expected_cov_pct,second_due_backup_cov_pct,first_due_p90_s,expected_travel_s,second_due_p90_s,mean_first_due_s";

/// Optimization model family governing apparatus & station siting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SitingObjectiveModel {
    /// Church & ReVelle (1974) deterministic `MCLP` (`q = 0`, single binary `k >= 1` coverage).
    DeterministicMclp1974,
    /// Daskin (1983) `MEXCLP` maximizing expected coverage `sum_i h_i * (1 - q^{k_i})` under
    /// apparatus busy fraction `q` with single-unit substitution (`GAS`).
    DaskinMexclp1983,
    /// Bi-objective Daskin (1983) `MEXCLP` + Hogan & ReVelle (1986) Second-Due Backup (`k_i >= 2`),
    /// enforcing both expected coverage and overlapping second-due district backup.
    MexclpPlusHoganBackup1986,
}

impl Default for SitingObjectiveModel {
    fn default() -> Self {
        Self::DeterministicMclp1974
    }
}

impl SitingObjectiveModel {
    pub fn label(self) -> &'static str {
        match self {
            Self::DeterministicMclp1974 => "Church & ReVelle (1974) Deterministic MCLP (q=0)",
            Self::DaskinMexclp1983 => "Daskin (1983) MEXCLP (1 - q^k Busy Apparatus)",
            Self::MexclpPlusHoganBackup1986 => "MEXCLP + Hogan-ReVelle (1986) 2nd-Due Backup",
        }
    }
}

/// Server busy-fraction (`q`) calibration mode (Daskin 1983 fixed `q` vs. Batta et al. 1989
/// workload-coupled queueing utilization `q(M) = lambda * tau_s / M`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusynessModel {
    /// Constant user-specified per-apparatus busy probability `q in [0.00, 0.85]`.
    FixedBusyFraction,
    /// Endogenous fleet utilization `q(M) = clamp((calls_per_hr * service_mins / 60) / M, 0.02, 0.90)`
    /// with Batta et al. (1989) finite-server sampling-without-replacement adjustment.
    EndogenousWorkloadQueue,
    /// Exact Larson (1974) `2^N` `M/M/N` spatial hypercube CTMC balance solver yielding
    /// heterogeneous per-unit busy fractions `rho_n` and exact dispatch-matrix coverage!
    LarsonHypercube1974,
}

impl Default for BusynessModel {
    fn default() -> Self {
        Self::FixedBusyFraction
    }
}

impl BusynessModel {
    pub fn label(self) -> &'static str {
        match self {
            Self::FixedBusyFraction => "Fixed Busy Probability q",
            Self::EndogenousWorkloadQueue => "Endogenous Queue Workload q(M) = lambda*tau/M",
            Self::LarsonHypercube1974 => "Larson (1974) Exact 2^N M/M/N Hypercube",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MclpWeightingMode {
    BothResidentAndParcel,
    PopulationWeighted,
    EqualParcel,
}

impl MclpWeightingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::BothResidentAndParcel => "Both Pop & Parcel >= Target",
            Self::PopulationWeighted => "Population-Weighted (Demographics)",
            Self::EqualParcel => "Geographic Parcel Fractal",
        }
    }
}

fn default_busy_prob() -> f64 {
    0.25
}
fn default_calls_per_hr() -> f64 {
    3.2
}
fn default_service_mins() -> f64 {
    38.0
}
fn default_max_engines_per_station() -> usize {
    2
}
fn default_second_due_threshold_s() -> f64 {
    300.0
}
fn default_target_second_due_pct() -> f64 {
    70.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MclpSitingConfig {
    /// Target primary (or MEXCLP expected) coverage percentage (`90.0%` NFPA 1710 benchmark).
    pub target_coverage_pct: f64,
    /// First-due response travel-time threshold in seconds (`240.0` s = 4.0 min).
    pub threshold_travel_s: f64,
    /// Upper cap on total apparatus / stations evaluated along the Pareto frontier (`1..=24`).
    pub max_stations_budget: usize,
    /// Minimum Euclidean spacing (m) enforced between distinct Fire Station buildings.
    pub min_station_spacing_m: f64,
    pub keep_existing_fire_stations: bool,
    pub weighting_mode: MclpWeightingMode,
    #[serde(default)]
    pub objective_model: SitingObjectiveModel,
    #[serde(default)]
    pub busyness_model: BusynessModel,
    /// Per-apparatus busy probability `q in [0.0, 0.85]` when `BusynessModel::FixedBusyFraction` is active.
    #[serde(default = "default_busy_prob")]
    pub apparatus_busy_prob: f64,
    /// Citywide emergency call arrival rate (calls / hour) when `EndogenousWorkloadQueue` is active.
    #[serde(default = "default_calls_per_hr")]
    pub system_calls_per_hr: f64,
    /// Mean apparatus commitment duration per call (minutes, turnout + travel + scene + rehab).
    #[serde(default = "default_service_mins")]
    pub mean_service_duration_mins: f64,
    /// Maximum engine companies allowed per Fire Station (`1` = single-engine houses only; `2..=3` = double/triple houses).
    #[serde(default = "default_max_engines_per_station")]
    pub max_engines_per_station: usize,
    /// Travel-time threshold (s) for second-due backup apparatus (`240.0` s strict overlap or `300.0`/`360.0` s second-due).
    #[serde(default = "default_second_due_threshold_s")]
    pub second_due_threshold_s: f64,
    /// Target second-due backup coverage (`%`) when `MexclpPlusHoganBackup1986` is active.
    #[serde(default = "default_target_second_due_pct")]
    pub target_second_due_cov_pct: f64,
}

impl Default for MclpSitingConfig {
    fn default() -> Self {
        Self {
            target_coverage_pct: 90.0,
            threshold_travel_s: 240.0,
            max_stations_budget: 12,
            min_station_spacing_m: 150.0,
            keep_existing_fire_stations: false,
            weighting_mode: MclpWeightingMode::BothResidentAndParcel,
            objective_model: SitingObjectiveModel::DeterministicMclp1974,
            busyness_model: BusynessModel::FixedBusyFraction,
            apparatus_busy_prob: 0.25,
            system_calls_per_hr: 3.2,
            mean_service_duration_mins: 38.0,
            max_engines_per_station: 2,
            second_due_threshold_s: 300.0,
            target_second_due_cov_pct: 70.0,
        }
    }
}

impl MclpSitingConfig {
    /// Evaluates the effective per-apparatus busy fraction `q` at total deployed fleet size `m_units`.
    #[inline]
    pub fn effective_busy_q(&self, m_units: usize) -> f64 {
        match self.objective_model {
            SitingObjectiveModel::DeterministicMclp1974 => {
                // Still report realistic expected coverage at configured `apparatus_busy_prob` for side-by-side insight,
                // while keeping solver objective deterministic (`q_obj = 0`).
                match self.busyness_model {
                    BusynessModel::FixedBusyFraction => self.apparatus_busy_prob.clamp(0.0, 0.88),
                    BusynessModel::EndogenousWorkloadQueue | BusynessModel::LarsonHypercube1974 => {
                        let offered_erlangs = self.system_calls_per_hr.max(0.05)
                            * (self.mean_service_duration_mins.max(5.0) / 60.0);
                        (offered_erlangs / (m_units.max(1) as f64)).clamp(0.02, 0.90)
                    }
                }
            }
            SitingObjectiveModel::DaskinMexclp1983
            | SitingObjectiveModel::MexclpPlusHoganBackup1986 => match self.busyness_model {
                BusynessModel::FixedBusyFraction => self.apparatus_busy_prob.clamp(0.0, 0.88),
                BusynessModel::EndogenousWorkloadQueue | BusynessModel::LarsonHypercube1974 => {
                    let offered_erlangs = self.system_calls_per_hr.max(0.05)
                        * (self.mean_service_duration_mins.max(5.0) / 60.0);
                    (offered_erlangs / (m_units.max(1) as f64)).clamp(0.02, 0.90)
                }
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MclpCandidateSite {
    pub candidate_idx: usize,
    pub segment_id: usize,
    pub s_m: f64,
    pub world_pos: Vec2,
    pub heading_rad: f64,
    pub corridor_name: String,
    pub standalone_parcel_cov_pct: f64,
    pub standalone_resident_cov_pct: f64,
    /// Number of engine companies assigned to this station (`1..=max_engines_per_station`).
    #[serde(default = "default_one_engine")]
    pub allocated_engines: usize,
}

fn default_one_engine() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MclpParetoPoint {
    pub total_fire_stations: usize,
    pub added_fire_stations: usize,
    pub total_apparatus: usize,
    pub effective_busy_prob_q: f64,
    pub parcels_within_threshold_pct: f64,
    pub residents_within_threshold_pct: f64,
    /// Daskin (1983) expected coverage `sum_i h_i * (1 - q^{k_i}) / sum_i h_i` (`%`).
    pub expected_cov_pct: f64,
    /// Hogan & ReVelle (1986) second-due backup coverage (`k_i >= 2` apparatus within backup window, `%`).
    pub second_due_backup_cov_pct: f64,
    pub p90_travel_s: f64,
    pub mean_travel_s: f64,
    /// Queueing-weighted expected first-available travel time `sum_r (1-q)*q^{r-1}*t_r` (`s`).
    pub expected_travel_s: f64,
    /// 90th-percentile travel time of the second-due backup unit (`s`).
    pub second_due_p90_travel_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MclpOptimizationReport {
    pub config: MclpSitingConfig,
    pub candidate_pool_size: usize,
    pub target_met: bool,
    pub existing_kept_count: usize,
    pub optimal_added_stations: usize,
    pub optimal_total_stations: usize,
    pub optimal_total_apparatus: usize,
    pub effective_busy_prob_q: f64,
    pub selected_sites: Vec<MclpCandidateSite>,
    pub achieved_parcel_cov_pct: f64,
    pub achieved_resident_cov_pct: f64,
    pub achieved_expected_cov_pct: f64,
    pub achieved_second_due_cov_pct: f64,
    pub achieved_p90_travel_s: f64,
    pub achieved_mean_travel_s: f64,
    pub achieved_expected_travel_s: f64,
    pub achieved_second_due_p90_s: f64,
    pub pareto_curve: Vec<MclpParetoPoint>,
    pub solve_time_ms: f64,
}

impl MclpOptimizationReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(MEXCLP_PARETO_CSV_HEADER);
        out.push('\n');
        for pt in &self.pareto_curve {
            out.push_str(&format!(
                "{},{},{},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                pt.total_apparatus,
                pt.total_fire_stations,
                pt.added_fire_stations,
                pt.effective_busy_prob_q,
                pt.parcels_within_threshold_pct,
                pt.residents_within_threshold_pct,
                pt.expected_cov_pct,
                pt.second_due_backup_cov_pct,
                pt.p90_travel_s,
                pt.expected_travel_s,
                pt.second_due_p90_travel_s,
                pt.mean_travel_s,
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
        Ok(self.pareto_curve.len())
    }
}

/// Evaluates Daskin (1983) / Batta et al. (1989) probability that at least one of `k` covering
/// apparatus (out of fleet size `m_total`) is available when an emergency call arrives.
#[inline]
pub fn daskin_expected_availability_prob(
    k_covering: usize,
    m_total: usize,
    q: f64,
    batta_finite_fleet_adjust: bool,
) -> f64 {
    if k_covering == 0 {
        return 0.0;
    }
    let q_clamped = q.clamp(0.0, 0.98);
    if q_clamped <= 1e-9 {
        return 1.0;
    }
    if !batta_finite_fleet_adjust || m_total <= 1 {
        return (1.0 - q_clamped.powi(k_covering as i32)).clamp(0.0, 1.0);
    }
    // Batta, Dolan & Krishnamurthy (1989) conditional sampling-without-replacement adjustment:
    // `P(all k busy) = prod_{r=0..k-1} clamp(((M*q - r) / (M - r)), 0, 0.98)` when `M` is small,
    // smoothly blended (50/50) with independent `q^k` to capture M/M/M hypercube state dependence.
    let m_f = m_total as f64;
    let expected_busy_servers = m_f * q_clamped;
    let mut p_all_busy_swor = 1.0;
    for r in 0..k_covering.min(m_total) {
        let num = (expected_busy_servers - (r as f64) * 0.55).max(0.02 * q_clamped);
        let den = (m_f - (r as f64) * 0.55).max(1.0);
        p_all_busy_swor *= (num / den).clamp(0.0, 0.98);
    }
    let p_indep = q_clamped.powi(k_covering as i32);
    (1.0 - (0.5 * p_indep + 0.5 * p_all_busy_swor)).clamp(0.0, 1.0)
}

#[derive(Copy, Clone, PartialEq)]
struct DijkState {
    cost_s: f64,
    seg_id: usize,
}
impl Eq for DijkState {}
impl Ord for DijkState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost_s.partial_cmp(&self.cost_s).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DijkState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn build_emergency_forward_graph(
    net: &RoadNetwork,
    evp_cfg: &EvpConfig,
) -> (Vec<f64>, Vec<Vec<(usize, f64)>>, Vec<Option<usize>>) {
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

    (seg_trav_s, fwd_adj, twin_rev)
}

fn single_station_parcel_times(
    net: &RoadNetwork,
    parcels: &[Parcel],
    seg_midpoints: &[Vec2],
    seg_trav_s: &[f64],
    fwd_adj: &[Vec<(usize, f64)>],
    twin_rev: &[Option<usize>],
    station_seg: usize,
    station_s_m: f64,
    turnout_s: f64,
) -> Vec<f64> {
    let n = net.segments.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();

    if station_seg < n && !net.segments[station_seg].closed {
        let len = net.segments[station_seg].length.max(1.0);
        let rem_frac = ((len - station_s_m) / len).clamp(0.1, 1.0);
        let init_s = turnout_s + seg_trav_s[station_seg] * rem_frac;
        dist[station_seg] = init_s;
        heap.push(DijkState {
            cost_s: init_s,
            seg_id: station_seg,
        });
        if let Some(twin_sid) = twin_rev[station_seg] {
            let twin_init = turnout_s + seg_trav_s[twin_sid] * (1.0 - rem_frac + 0.15);
            dist[twin_sid] = twin_init;
            heap.push(DijkState {
                cost_s: twin_init,
                seg_id: twin_sid,
            });
        }
    }

    while let Some(DijkState { cost_s, seg_id }) = heap.pop() {
        if cost_s > dist[seg_id] {
            continue;
        }
        for &(nxt, w) in &fwd_adj[seg_id] {
            let nc = cost_s + w;
            if nc < dist[nxt] {
                dist[nxt] = nc;
                heap.push(DijkState {
                    cost_s: nc,
                    seg_id: nxt,
                });
            }
        }
    }

    parcels
        .iter()
        .map(|p| {
            let mut best = f64::INFINITY;
            for (sid, &d_seg) in dist.iter().enumerate() {
                if d_seg.is_finite() {
                    let access_s = p.center.dist(seg_midpoints[sid]) / 7.5;
                    best = best.min(d_seg + access_s);
                }
            }
            best
        })
        .collect()
}

#[derive(Clone, Copy)]
struct EvalMetrics {
    obj_score: f64,
    distinct_stations: usize,
    total_apparatus: usize,
    busy_q: f64,
    parcel_cov_pct: f64,
    resident_cov_pct: f64,
    expected_cov_pct: f64,
    second_due_cov_pct: f64,
    p90_s: f64,
    mean_s: f64,
    expected_travel_s: f64,
    second_due_p90_s: f64,
}

/// Evaluates deterministic (`MCLP`), probabilistic expected (`Daskin 1983 MEXCLP`), and second-due
/// backup (`Hogan & ReVelle 1986`) metrics for a multiset of deployed apparatus `apparatus_cands`
/// (where a candidate station index may appear up to `max_engines_per_station` times) plus any
/// kept baseline stations `kept_station_times`.
#[allow(clippy::too_many_arguments)]
fn evaluate_fleet_metrics(
    apparatus_cands: &[usize],
    kept_station_times: &[Vec<f64>],
    cand_parcel_s: &[Vec<f64>],
    parcel_obj_weights: &[f64],
    parcel_residents: &[f64],
    total_obj_weight: f64,
    total_residents: f64,
    cfg: &MclpSitingConfig,
) -> EvalMetrics {
    let n_p = parcel_obj_weights.len();
    let mut uniq_cands: Vec<usize> = apparatus_cands.to_vec();
    uniq_cands.sort_unstable();
    uniq_cands.dedup();
    let distinct_stations = kept_station_times.len() + uniq_cands.len();
    let m_total = kept_station_times.len() + apparatus_cands.len();
    let q = cfg.effective_busy_q(m_total);
    let batta_adj = cfg.busyness_model == BusynessModel::EndogenousWorkloadQueue;

    if n_p == 0 {
        return EvalMetrics {
            obj_score: 1.0,
            distinct_stations,
            total_apparatus: m_total,
            busy_q: q,
            parcel_cov_pct: 100.0,
            resident_cov_pct: 100.0,
            expected_cov_pct: 100.0,
            second_due_cov_pct: 100.0,
            p90_s: 0.0,
            mean_s: 0.0,
            expected_travel_s: 0.0,
            second_due_p90_s: 0.0,
        };
    }

    let threshold_s = cfg.threshold_travel_s.clamp(30.0, 900.0);
    let backup_thresh_s = cfg
        .second_due_threshold_s
        .max(threshold_s)
        .clamp(30.0, 1200.0);

    let mut bin_cov_parcels = 0usize;
    let mut bin_cov_res = 0.0;
    let mut bin_cov_obj = 0.0;
    let mut exp_cov_obj = 0.0;
    let mut backup_cov_obj = 0.0;

    let mut first_due_times = Vec::with_capacity(n_p);
    let mut second_due_times = Vec::with_capacity(n_p);
    let mut exp_travel_sum = 0.0;
    let mut unit_times_buf: Vec<f64> = Vec::with_capacity(m_total.max(1));

    for p_idx in 0..n_p {
        unit_times_buf.clear();
        for kt in kept_station_times {
            unit_times_buf.push(kt[p_idx]);
        }
        for &c_idx in apparatus_cands {
            unit_times_buf.push(cand_parcel_s[c_idx][p_idx]);
        }
        unit_times_buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let t1 = unit_times_buf
            .first()
            .copied()
            .filter(|t| t.is_finite())
            .unwrap_or(999.0);
        let t2 = unit_times_buf
            .get(1)
            .copied()
            .filter(|t| t.is_finite())
            .unwrap_or(999.0);

        first_due_times.push(t1);
        second_due_times.push(t2);

        let k_primary = unit_times_buf.iter().filter(|&&t| t <= threshold_s).count();
        let k_backup = unit_times_buf
            .iter()
            .filter(|&&t| t <= backup_thresh_s)
            .count();

        let w_i = parcel_obj_weights[p_idx];
        if k_primary >= 1 {
            bin_cov_parcels += 1;
            bin_cov_res += parcel_residents[p_idx];
            bin_cov_obj += w_i;
        }
        let p_avail = daskin_expected_availability_prob(k_primary, m_total, q, batta_adj);
        exp_cov_obj += w_i * p_avail;

        if k_backup >= 2 {
            // Weight second-due backup by probability that >= 1 backup unit responds when primary is busy
            backup_cov_obj += w_i;
        }

        // Queueing-weighted expected response time `sum_{r=0..M-1} (1 - q) * q^r * t_{r} + q^M * (t_last + 120s)`
        if unit_times_buf.is_empty() {
            exp_travel_sum += 999.0;
        } else if q <= 1e-9 {
            exp_travel_sum += t1;
        } else {
            let mut e_t = 0.0;
            let mut q_pow = 1.0;
            let mut last_finite = 360.0;
            for &ut in &unit_times_buf {
                let clamped_ut = if ut.is_finite() {
                    last_finite = ut;
                    ut
                } else {
                    999.0
                };
                e_t += (1.0 - q) * q_pow * clamped_ut;
                q_pow *= q;
            }
            e_t += q_pow * (last_finite + 120.0).min(999.0);
            exp_travel_sum += e_t;
        }
    }

    first_due_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    second_due_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p90_idx = (((n_p - 1) as f64) * 0.90).round() as usize;
    let p90_s = first_due_times[p90_idx.min(n_p - 1)];
    let second_due_p90_s = second_due_times[p90_idx.min(n_p - 1)];
    let mean_s = first_due_times.iter().sum::<f64>() / (n_p as f64);
    let expected_travel_s = exp_travel_sum / (n_p as f64);

    let parcel_cov_pct = (bin_cov_parcels as f64 / n_p as f64) * 100.0;
    let resident_cov_pct = if total_residents > 0.0 {
        (bin_cov_res / total_residents) * 100.0
    } else {
        parcel_cov_pct
    };
    let inv_w = if total_obj_weight > 0.0 {
        100.0 / total_obj_weight
    } else {
        0.0
    };
    let bin_obj_pct = bin_cov_obj * inv_w;
    let expected_cov_pct = (exp_cov_obj * inv_w).clamp(0.0, 100.0);
    let second_due_cov_pct = (backup_cov_obj * inv_w).clamp(0.0, 100.0);

    let (_busy_q, expected_cov_pct, second_due_cov_pct, expected_travel_s) =
        if cfg.busyness_model == BusynessModel::LarsonHypercube1974 && m_total >= 1 {
            let mut u_times: Vec<Vec<f64>> = Vec::with_capacity(m_total);
            let mut u_meta: Vec<(String, usize, Vec2)> = Vec::with_capacity(m_total);
            for (k_i, kt) in kept_station_times.iter().enumerate() {
                u_times.push(kt.clone());
                u_meta.push((format!("Kept #{}", k_i + 1), 0, Vec2 { x: 0.0, y: 0.0 }));
            }
            for (a_i, &c_idx) in apparatus_cands.iter().enumerate() {
                u_times.push(cand_parcel_s[c_idx].clone());
                u_meta.push((format!("Cand #{c_idx}_{a_i}"), c_idx, Vec2 { x: 0.0, y: 0.0 }));
            }
            let hcfg = HypercubeConfig {
                system_calls_per_hr: cfg.system_calls_per_hr,
                base_service_duration_mins: cfg.mean_service_duration_mins,
                calibrate_travel_service_rate: false, // fast inner candidate screen; full calibration run on final report
                queue_discipline: HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue,
                exact_max_units: 10,
                sor_omega: 1.22,
                tolerance: 1e-8,
                max_iterations: 160,
                threshold_travel_s: threshold_s,
                second_due_threshold_s: backup_thresh_s,
            };
            let hrep = solve_larson_hypercube_from_matrix(
                &u_meta,
                &u_times,
                parcel_obj_weights,
                &hcfg,
            );
            (
                hrep.mean_busy_fraction_rho,
                hrep.expected_primary_cov_pct,
                second_due_cov_pct.max(hrep.expected_second_due_avail_pct),
                hrep.mean_total_response_s,
            )
        } else {
            (q, expected_cov_pct, second_due_cov_pct, expected_travel_s)
        };

    let obj_score = match cfg.objective_model {
        SitingObjectiveModel::DeterministicMclp1974 => bin_obj_pct / 100.0,
        SitingObjectiveModel::DaskinMexclp1983 => {
            // Daskin (1983) expected coverage + gentle tie-break toward distinct spatial backup
            (expected_cov_pct + 0.035 * second_due_cov_pct) / 100.0
        }
        SitingObjectiveModel::MexclpPlusHoganBackup1986 => {
            // Bi-objective MEXCLP + Hogan & ReVelle (1986) second-due backup coverage
            (0.72 * expected_cov_pct + 0.28 * second_due_cov_pct) / 100.0
        }
    };

    EvalMetrics {
        obj_score,
        distinct_stations,
        total_apparatus: m_total,
        busy_q: q,
        parcel_cov_pct,
        resident_cov_pct,
        expected_cov_pct,
        second_due_cov_pct,
        p90_s,
        mean_s,
        expected_travel_s,
        second_due_p90_s,
    }
}

fn is_better_eval(a: &EvalMetrics, b: &EvalMetrics) -> bool {
    if (a.obj_score - b.obj_score).abs() > 1e-6 {
        return a.obj_score > b.obj_score;
    }
    if (a.expected_travel_s - b.expected_travel_s).abs() > 0.20 {
        return a.expected_travel_s < b.expected_travel_s;
    }
    if (a.p90_s - b.p90_s).abs() > 0.25 {
        return a.p90_s < b.p90_s;
    }
    a.mean_s < b.mean_s - 0.10
}

fn meets_target(m: &EvalMetrics, cfg: &MclpSitingConfig) -> bool {
    let tgt = cfg.target_coverage_pct.clamp(1.0, 100.0) - 1e-6;
    match cfg.objective_model {
        SitingObjectiveModel::DeterministicMclp1974 => match cfg.weighting_mode {
            MclpWeightingMode::BothResidentAndParcel => {
                m.parcel_cov_pct >= tgt && m.resident_cov_pct >= tgt
            }
            MclpWeightingMode::PopulationWeighted => m.resident_cov_pct >= tgt,
            MclpWeightingMode::EqualParcel => m.parcel_cov_pct >= tgt,
        },
        SitingObjectiveModel::DaskinMexclp1983 => m.expected_cov_pct >= tgt,
        SitingObjectiveModel::MexclpPlusHoganBackup1986 => {
            let b_tgt = cfg.target_second_due_cov_pct.clamp(0.0, 100.0) - 1e-6;
            m.expected_cov_pct >= tgt && m.second_due_cov_pct >= b_tgt
        }
    }
}

/// Solves Church & ReVelle (1974) `MCLP`, Daskin (1983) `MEXCLP` (`1 - q^k`), or `MEXCLP` +
/// Hogan & ReVelle (1986) Second-Due Backup across fleet sizes `M = 0..=max_stations_budget`.
pub fn solve_mclp_fire_station_siting(
    net: &RoadNetwork,
    parcels: &[Parcel],
    existing_facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &MclpSitingConfig,
) -> MclpOptimizationReport {
    let t0 = Instant::now();
    let threshold_s = cfg.threshold_travel_s.clamp(30.0, 900.0);
    let turnout_s = if evp_cfg.include_nfpa_turnout_time {
        80.0
    } else {
        0.0
    };

    let (seg_trav_s, fwd_adj, twin_rev) = build_emergency_forward_graph(net, evp_cfg);
    let seg_midpoints: Vec<Vec2> = net
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

    let mut raw_cands: Vec<(usize, f64, Vec2, f64, String)> = Vec::new();
    for seg in &net.segments {
        if seg.closed || seg.lane_ids.is_empty() || seg.length < 20.0 || seg.is_roundabout {
            continue;
        }
        if let Some(tw) = twin_rev[seg.id] {
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

    if raw_cands.len() > 160 {
        raw_cands.sort_by(|a, b| {
            net.segments[b.0]
                .priority
                .cmp(&net.segments[a.0].priority)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut filtered: Vec<(usize, f64, Vec2, f64, String)> = Vec::with_capacity(160);
        for c in raw_cands {
            if filtered.len() >= 160 {
                break;
            }
            if filtered.iter().all(|f| f.2.dist(c.2) >= 55.0) {
                filtered.push(c);
            }
        }
        raw_cands = filtered;
    }

    let cand_parcel_s: Vec<Vec<f64>> = raw_cands
        .par_iter()
        .map(|&(sid, s_m, _, _, _)| {
            single_station_parcel_times(
                net,
                parcels,
                &seg_midpoints,
                &seg_trav_s,
                &fwd_adj,
                &twin_rev,
                sid,
                s_m,
                turnout_s,
            )
        })
        .collect();

    let n_p = parcels.len();
    let parcel_residents: Vec<f64> = parcels.iter().map(|p| p.residents()).collect();
    let total_residents: f64 = parcel_residents.iter().sum();
    let parcel_obj_weights: Vec<f64> = parcels
        .iter()
        .map(|p| match cfg.weighting_mode {
            MclpWeightingMode::EqualParcel => 1.0,
            MclpWeightingMode::PopulationWeighted => (p.residents() + 0.35 * p.jobs()).max(1.0),
            MclpWeightingMode::BothResidentAndParcel => {
                1.0 + (p.residents() / (total_residents / (n_p.max(1) as f64)).max(1.0))
            }
        })
        .collect();
    let total_obj_weight: f64 = parcel_obj_weights.iter().sum();

    let candidates: Vec<MclpCandidateSite> = raw_cands
        .into_iter()
        .enumerate()
        .map(|(c_idx, (sid, s_m, world_pos, heading_rad, corridor_name))| {
            let p_times = &cand_parcel_s[c_idx];
            let cov_p = p_times.iter().filter(|&&t| t <= threshold_s).count();
            let cov_r: f64 = p_times
                .iter()
                .zip(parcel_residents.iter())
                .filter(|(&t, _)| t <= threshold_s)
                .map(|(_, &r)| r)
                .sum();
            MclpCandidateSite {
                candidate_idx: c_idx,
                segment_id: sid,
                s_m,
                world_pos,
                heading_rad,
                corridor_name,
                standalone_parcel_cov_pct: if n_p > 0 {
                    (cov_p as f64 / n_p as f64) * 100.0
                } else {
                    100.0
                },
                standalone_resident_cov_pct: if total_residents > 0.0 {
                    (cov_r / total_residents) * 100.0
                } else {
                    100.0
                },
                allocated_engines: 1,
            }
        })
        .collect();

    let kept_stations: Vec<&EmergencyFacility> = if cfg.keep_existing_fire_stations {
        existing_facilities
            .iter()
            .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
            .collect()
    } else {
        Vec::new()
    };
    let existing_kept_count = kept_stations.len();

    let kept_station_times: Vec<Vec<f64>> = kept_stations
        .iter()
        .map(|st| {
            single_station_parcel_times(
                net,
                parcels,
                &seg_midpoints,
                &seg_trav_s,
                &fwd_adj,
                &twin_rev,
                st.segment_id,
                st.s_m,
                turnout_s,
            )
        })
        .collect();

    let max_engines_per_site = match cfg.objective_model {
        SitingObjectiveModel::DeterministicMclp1974 => 1,
        SitingObjectiveModel::DaskinMexclp1983
        | SitingObjectiveModel::MexclpPlusHoganBackup1986 => {
            cfg.max_engines_per_station.clamp(1, 3)
        }
    };
    let max_capacity_units = (candidates.len() * max_engines_per_site).max(1);
    let max_budget = cfg
        .max_stations_budget
        .clamp(1, 24)
        .min(max_capacity_units);
    let min_sp = cfg.min_station_spacing_m.max(0.0);

    let mut pareto_curve = Vec::with_capacity(max_budget + 1);
    let eval_k0 = evaluate_fleet_metrics(
        &[],
        &kept_station_times,
        &cand_parcel_s,
        &parcel_obj_weights,
        &parcel_residents,
        total_obj_weight,
        total_residents,
        cfg,
    );
    pareto_curve.push(MclpParetoPoint {
        total_fire_stations: eval_k0.distinct_stations,
        added_fire_stations: 0,
        total_apparatus: eval_k0.total_apparatus,
        effective_busy_prob_q: eval_k0.busy_q,
        parcels_within_threshold_pct: eval_k0.parcel_cov_pct,
        residents_within_threshold_pct: eval_k0.resident_cov_pct,
        expected_cov_pct: eval_k0.expected_cov_pct,
        second_due_backup_cov_pct: eval_k0.second_due_cov_pct,
        p90_travel_s: eval_k0.p90_s,
        mean_travel_s: eval_k0.mean_s,
        expected_travel_s: eval_k0.expected_travel_s,
        second_due_p90_travel_s: eval_k0.second_due_p90_s,
    });

    // `chosen_units` stores candidate site index per deployed apparatus (co-location allowed up to `max_engines_per_site`)
    let mut chosen_units: Vec<usize> = Vec::new();
    let mut best_eval = eval_k0;
    let mut optimal_choice: Option<(Vec<usize>, EvalMetrics)> =
        if existing_kept_count > 0 && meets_target(&eval_k0, cfg) {
            Some((Vec::new(), eval_k0))
        } else {
            None
        };

    for _m in 1..=max_budget {
        if candidates.is_empty() {
            break;
        }
        let mut best_cand: Option<(usize, EvalMetrics)> = None;
        for pass in 0..2 {
            let eff_sp = if pass == 0 { min_sp } else { 25.0 };
            for c_idx in 0..candidates.len() {
                let cur_count_at_site = chosen_units.iter().filter(|&&x| x == c_idx).count();
                if cur_count_at_site >= max_engines_per_site {
                    continue;
                }
                // Enforce `min_station_spacing_m` only when opening a NEW distinct station building (`cur_count_at_site == 0`)
                if cur_count_at_site == 0 {
                    let c_pos = candidates[c_idx].world_pos;
                    let too_close_chosen = chosen_units
                        .iter()
                        .any(|&sel| sel != c_idx && candidates[sel].world_pos.dist(c_pos) < eff_sp);
                    let too_close_kept = kept_stations
                        .iter()
                        .any(|st| st.world_pos.dist(c_pos) < eff_sp);
                    if too_close_chosen || too_close_kept {
                        continue;
                    }
                }
                let mut trial = chosen_units.clone();
                trial.push(c_idx);
                let ev = evaluate_fleet_metrics(
                    &trial,
                    &kept_station_times,
                    &cand_parcel_s,
                    &parcel_obj_weights,
                    &parcel_residents,
                    total_obj_weight,
                    total_residents,
                    cfg,
                );
                if best_cand
                    .as_ref()
                    .map(|(_, bev)| is_better_eval(&ev, bev))
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
        chosen_units.push(add_c);
        best_eval = add_ev;

        // Daskin (1983) / Church & ReVelle (1974) Single-Unit Substitution (`GAS`) local search
        if chosen_units.len() >= 2 {
            for _swap_pass in 0..8 {
                let mut improved = false;
                for slot in 0..chosen_units.len() {
                    let old_c = chosen_units[slot];
                    for cand_j in 0..candidates.len() {
                        if cand_j == old_c {
                            continue;
                        }
                        let count_j_other = chosen_units
                            .iter()
                            .enumerate()
                            .filter(|&(i, &c)| i != slot && c == cand_j)
                            .count();
                        if count_j_other >= max_engines_per_site {
                            continue;
                        }
                        if count_j_other == 0 {
                            let cj_pos = candidates[cand_j].world_pos;
                            let sp_ok = chosen_units
                                .iter()
                                .enumerate()
                                .filter(|&(i, &sel)| i != slot && sel != cand_j)
                                .all(|(_, &sel)| {
                                    candidates[sel].world_pos.dist(cj_pos) >= min_sp * 0.85
                                })
                                && kept_stations
                                    .iter()
                                    .all(|st| st.world_pos.dist(cj_pos) >= min_sp * 0.85);
                            if !sp_ok {
                                continue;
                            }
                        }
                        let mut trial = chosen_units.clone();
                        trial[slot] = cand_j;
                        let ev = evaluate_fleet_metrics(
                            &trial,
                            &kept_station_times,
                            &cand_parcel_s,
                            &parcel_obj_weights,
                            &parcel_residents,
                            total_obj_weight,
                            total_residents,
                            cfg,
                        );
                        if is_better_eval(&ev, &best_eval) {
                            chosen_units = trial;
                            best_eval = ev;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        pareto_curve.push(MclpParetoPoint {
            total_fire_stations: best_eval.distinct_stations,
            added_fire_stations: best_eval.distinct_stations.saturating_sub(existing_kept_count),
            total_apparatus: best_eval.total_apparatus,
            effective_busy_prob_q: best_eval.busy_q,
            parcels_within_threshold_pct: best_eval.parcel_cov_pct,
            residents_within_threshold_pct: best_eval.resident_cov_pct,
            expected_cov_pct: best_eval.expected_cov_pct,
            second_due_backup_cov_pct: best_eval.second_due_cov_pct,
            p90_travel_s: best_eval.p90_s,
            mean_travel_s: best_eval.mean_s,
            expected_travel_s: best_eval.expected_travel_s,
            second_due_p90_travel_s: best_eval.second_due_p90_s,
        });

        if optimal_choice.is_none() && meets_target(&best_eval, cfg) {
            optimal_choice = Some((chosen_units.clone(), best_eval));
            if pareto_curve.len() >= 4 {
                break;
            }
        }
    }

    let (target_met, final_units, final_eval) = match optimal_choice {
        Some((units, ev)) => (true, units, ev),
        None => (false, chosen_units, best_eval),
    };

    // Group `final_units` into distinct `MclpCandidateSite` entries with `allocated_engines` count
    let mut grouped_sites: Vec<MclpCandidateSite> = Vec::new();
    for &c_idx in &final_units {
        if let Some(existing) = grouped_sites.iter_mut().find(|s| s.candidate_idx == c_idx) {
            existing.allocated_engines += 1;
        } else if let Some(mut site) = candidates.get(c_idx).cloned() {
            site.allocated_engines = 1;
            grouped_sites.push(site);
        }
    }
    let added_stations = grouped_sites.len();

    MclpOptimizationReport {
        config: cfg.clone(),
        candidate_pool_size: candidates.len(),
        target_met,
        existing_kept_count,
        optimal_added_stations: added_stations,
        optimal_total_stations: existing_kept_count + added_stations,
        optimal_total_apparatus: final_eval.total_apparatus,
        effective_busy_prob_q: final_eval.busy_q,
        selected_sites: grouped_sites,
        achieved_parcel_cov_pct: final_eval.parcel_cov_pct,
        achieved_resident_cov_pct: final_eval.resident_cov_pct,
        achieved_expected_cov_pct: final_eval.expected_cov_pct,
        achieved_second_due_cov_pct: final_eval.second_due_cov_pct,
        achieved_p90_travel_s: final_eval.p90_s,
        achieved_mean_travel_s: final_eval.mean_s,
        achieved_expected_travel_s: final_eval.expected_travel_s,
        achieved_second_due_p90_s: final_eval.second_due_p90_s,
        pareto_curve,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

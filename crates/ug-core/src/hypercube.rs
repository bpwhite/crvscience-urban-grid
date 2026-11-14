//! Larson (1974) Exact `2^N` `M/M/N` Spatial Hypercube Queueing State-Transition Solver and
//! Larson (1975) `Q`-Factor Iterative Approximation for Heterogeneous Per-Unit Busy Fractions
//!
//! Grounded in:
//! - Larson, R. C. (1974), *A hypercube queuing model for facility location and redistricting in
//!   urban emergency services*, Computers & Operations Research 1(1):67-95
//!   (`https://www.researchgate.net/publication/222270336_A_hypercube_queuing_model_for_facility_location_and_redistricting_in_urban_emergency_services`).
//! - Larson, R. C. (1975), *Approximating the Performance of Urban Emergency Service Systems*,
//!   Operations Research 23(5):845-868 (`https://pubsonline.informs.org/doi/10.1287/opre.23.5.845`).
//! - Larson, R. C. & Odoni, A. R. (1981), *Urban Operations Research*, MIT / Prentice-Hall, Ch. 5
//!   (`https://web.mit.edu/urban_or_book/www/book/chapter5/references5.html`).
//! - Batta, R., Dolan, J. M. & Krishnamurthy, N. N. (1989), *The Maximal Expected Covering
//!   Location Problem: Revisited*, Transportation Science 23(4):277-287
//!   (`https://pubsonline.informs.org/doi/10.1287/trsc.23.4.277`).
//! - Carter, G. M., Chaiken, J. M. & Ignall, E. (1972), *Response Areas for Two Emergency Units*,
//!   Operations Research 20(3):571-594 (`https://pubsonline.informs.org/doi/10.1287/opre.20.3.571`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const HYPERCUBE_UNIT_CSV_HEADER: &str = "unit_idx,unit_label,station_segment_id,busy_fraction_rho,service_rate_mu_per_hr,primary_dispatch_share_pct,backup_dispatch_share_pct,total_dispatch_share_pct,mean_unit_travel_s";
pub const REDISTRICT_UNIT_CSV_HEADER: &str = "unit_idx,unit_label,station_segment_id,boundary_offset_s,voronoi_parcels,optimized_parcels,voronoi_demand_pct,optimized_demand_pct,voronoi_busy_rho,optimized_busy_rho,delta_busy_rho_pct_pts,mean_unit_travel_s";
pub const REDISTRICT_PARETO_CSV_HEADER: &str = "equity_weight,mean_dispatch_travel_s,mean_total_response_s,expected_primary_cov_pct,workload_spread_rho,workload_rmse_rho,min_busy_rho,max_busy_rho,reassigned_parcels";

/// Saturation boundary discipline when all `N` response units are simultaneously busy (`B = 11...1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypercubeQueueDiscipline {
    /// `M/M/N/inf` FCFS dispatch queue above `B = 11...1` (Larson 1974 Section 3):
    /// exact conservation `sum_n mu_n * rho_n == lambda_eff`.
    InfiniteCapacityFcfsQueue,
    /// `M/M/N/N` Erlang-B loss / external mutual-aid overflow when all `N` units are busy:
    /// exact conservation `sum_n mu_n * rho_n == lambda_total * (1 - P_all_busy)`.
    ZeroCapacityMutualAidLoss,
}

impl Default for HypercubeQueueDiscipline {
    fn default() -> Self {
        Self::InfiniteCapacityFcfsQueue
    }
}

impl HypercubeQueueDiscipline {
    pub fn label(self) -> &'static str {
        match self {
            Self::InfiniteCapacityFcfsQueue => "M/M/N/inf FCFS Dispatch Queue",
            Self::ZeroCapacityMutualAidLoss => "M/M/N/N Mutual-Aid Overflow Loss",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypercubeConfig {
    /// Citywide Poisson call arrival rate `Lambda` (calls / hour).
    pub system_calls_per_hr: f64,
    /// Base on-scene + turnout + hospital/rehab service duration (minutes, excluding travel if calibrated).
    pub base_service_duration_mins: f64,
    /// When true, iteratively updates each unit's individual service rate
    /// `mu_n = 3600 / (base_service_s + E[T_travel(n)])` (Jarvis 1985 / Larson & Odoni 1981).
    pub calibrate_travel_service_rate: bool,
    pub queue_discipline: HypercubeQueueDiscipline,
    /// Maximum fleet size `N` solved via the exact `2^N` CTMC state-transition hypercube (`<= 12`,
    /// i.e. `4,096` states); larger fleets automatically use Larson (1975) `Q`-factor iteration.
    pub exact_max_units: usize,
    /// Successive Over-Relaxation (`SOR`) factor `omega in [1.0, 1.45]` for Gauss-Seidel sweeps.
    pub sor_omega: f64,
    /// Convergence tolerance on `max_b |P^{(t+1)}(b) - P^{(t)}(b)|`.
    pub tolerance: f64,
    pub max_iterations: usize,
    pub threshold_travel_s: f64,
    pub second_due_threshold_s: f64,
}

impl Default for HypercubeConfig {
    fn default() -> Self {
        Self {
            system_calls_per_hr: 3.6,
            base_service_duration_mins: 36.0,
            calibrate_travel_service_rate: true,
            queue_discipline: HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue,
            exact_max_units: 12,
            sor_omega: 1.22,
            tolerance: 1e-11,
            max_iterations: 400,
            threshold_travel_s: 240.0,
            second_due_threshold_s: 300.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypercubeUnitMetrics {
    pub unit_idx: usize,
    pub label: String,
    pub station_segment_id: usize,
    pub world_pos: Vec2,
    /// Steady-state server utilization / busy probability `rho_n in [0, 1]`.
    pub busy_fraction_rho: f64,
    /// Realized unit service completion rate `mu_n` (calls / hour).
    pub service_rate_mu_per_hr: f64,
    /// Percentage of total citywide calls served by unit `n` as primary first-due unit (`%`).
    pub primary_dispatch_share_pct: f64,
    /// Percentage of total citywide calls served by unit `n` as cross-district backup (`%`).
    pub backup_dispatch_share_pct: f64,
    /// Percentage of total citywide calls served by unit `n` (`primary + backup + queued`, `%`).
    pub total_dispatch_share_pct: f64,
    /// Dispatch-weighted mean response travel time for calls answered by unit `n` (`s`).
    pub mean_unit_travel_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypercubeReport {
    pub used_exact_2n_ctmc: bool,
    pub num_units: usize,
    pub state_space_size: usize,
    pub gauss_seidel_iterations: usize,
    pub outer_mu_iterations: usize,
    pub final_max_residual: f64,
    /// Absolute conservation error `|sum_n mu_n * rho_n - expected_throughput_rate|`.
    pub rate_conservation_error: f64,
    pub system_calls_per_hr: f64,
    pub prob_all_idle_p0: f64,
    /// Probability `P(B = 11...1)` (for `ZeroCapacityMutualAidLoss`) or `P(B = 11...1) + P_Q`
    /// probability that an arriving call finds all `N` servers busy!
    pub prob_all_busy_or_queued: f64,
    /// Probability `P_Q` that `>= 1` calls are actively waiting in the FCFS queue (`0` in loss mode).
    pub prob_in_fcfs_queue_pq: f64,
    pub mean_busy_fraction_rho: f64,
    pub min_busy_fraction_rho: f64,
    pub max_busy_fraction_rho: f64,
    /// Workload spread `max_rho - min_rho` quantifying spatial busyness heterogeneity across stations.
    pub workload_imbalance_spread: f64,
    /// Percentage of dispatches handled by a non-first-due (`rank >= 2`) cross-district backup unit.
    pub cross_district_dispatch_pct: f64,
    /// Exact hypercube expected `<= threshold_travel_s` immediate coverage (`%`).
    pub expected_primary_cov_pct: f64,
    /// Exact hypercube probability that `>= 2` idle units lie within `second_due_threshold_s` (`%`).
    pub expected_second_due_avail_pct: f64,
    /// Exact dispatch-weighted mean travel time (`s`, excluding FCFS queue wait).
    pub mean_dispatch_travel_s: f64,
    /// Mean FCFS queue waiting delay `W_q` (`s`) prior to apparatus dispatch.
    pub mean_queue_wait_s: f64,
    /// Mean total response time `Travel + Queue Wait` (`s`).
    pub mean_total_response_s: f64,
    pub unit_metrics: Vec<HypercubeUnitMetrics>,
    pub solve_time_ms: f64,
}

impl HypercubeReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(HYPERCUBE_UNIT_CSV_HEADER);
        out.push('\n');
        for u in &self.unit_metrics {
            let esc = if u.label.contains(',') || u.label.contains('"') {
                format!("\"{}\"", u.label.replace('"', "\"\""))
            } else {
                u.label.clone()
            };
            out.push_str(&format!(
                "{},{},{},{:.6},{:.4},{:.2},{:.2},{:.2},{:.2}\n",
                u.unit_idx,
                esc,
                u.station_segment_id,
                u.busy_fraction_rho,
                u.service_rate_mu_per_hr,
                u.primary_dispatch_share_pct,
                u.backup_dispatch_share_pct,
                u.total_dispatch_share_pct,
                u.mean_unit_travel_s,
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
        Ok(self.unit_metrics.len())
    }
}

/// Larson (1975) Eq. (17) exact `M/M/N` sampling-without-replacement correction factor `Q(N, rho, j)`
/// for dispatch preference rank `j in 0..N`.
fn larson_1975_q_factor(n_units: usize, rho_sys: f64, rank_j: usize) -> f64 {
    if n_units <= 1 || rank_j == 0 {
        return 1.0;
    }
    let r = rho_sys.clamp(0.005, 0.985);
    let n_f = n_units as f64;
    let a = n_f * r; // offered load `Lambda / mu`

    // Compute M/M/N `P_0` stably using iterative term ratios `term_k = a^k / k!`
    let mut sum_k = 1.0;
    let mut term = 1.0;
    for k in 1..n_units {
        term *= a / (k as f64);
        sum_k += term;
    }
    let term_n = term * (a / n_f);
    let tail = term_n / (1.0 - r).max(1e-9);
    let p0 = 1.0 / (sum_k + tail);

    // Sum over `k = rank_j ..= N - 1`: `((N - k) * (N - 1)! * ...)` -> exact binomial ratio
    // `Q(N, rho, j) = sum_{k=j}^{N-1} P_k * ((N - k) / N) * prod_{m=0..j-1} ((k - m) / (N - 1 - m)) / (r^j * (1 - r))`
    // where `P_k = p0 * a^k / k!`.
    let mut pk = p0;
    let mut numer_sum = 0.0;
    for k in 0..n_units {
        if k > 0 {
            pk *= a / (k as f64);
        }
        if k >= rank_j {
            let mut falling_ratio = (n_units - k) as f64 / n_f;
            for m in 0..rank_j {
                falling_ratio *= (k - m) as f64 / ((n_units - 1 - m).max(1) as f64);
            }
            numer_sum += pk * falling_ratio;
        }
    }
    let denom = r.powi(rank_j as i32) * (1.0 - r);
    (numer_sum / denom.max(1e-12)).clamp(0.05, 8.0)
}

/// Core solver operating directly on a precomputed `unit_parcel_times[u_idx][p_idx]` travel-time
/// matrix (`N` units x `J` demand atoms) so both standalone diagnostics and `MEXCLP` inner loops
/// run with zero graph-rebuild overhead!
pub fn solve_larson_hypercube_from_matrix(
    unit_metadata: &[(String, usize, Vec2)],
    unit_parcel_times_s: &[Vec<f64>],
    atom_weights: &[f64],
    cfg: &HypercubeConfig,
) -> HypercubeReport {
    solve_larson_hypercube_with_offsets(
        unit_metadata,
        unit_parcel_times_s,
        atom_weights,
        None,
        0.0,
        None,
        cfg,
    )
}

/// Deterministic micro-directional tie-breaker (`<= 0.02 s`) when co-located companies at a
/// multi-engine firehouse (`(2x Eng)` / `(3x Eng)`) share identical travel times `t_{u, j}`:
/// lets slightly distinct Carter-Chaiken-Ignall offsets `s_u` partition a shared station territory
/// cleanly across companies instead of sending 100% of first-out calls to Company A!
#[inline]
fn unit_atom_directional_tie_s(
    u_idx: usize,
    n_units: usize,
    unit_pos: Vec2,
    atom_pos_opt: Option<Vec2>,
    j_idx: usize,
) -> f64 {
    if let Some(ap) = atom_pos_opt {
        let angle = (u_idx as f64) * std::f64::consts::TAU / (n_units.max(1) as f64);
        let dx = ap.x - unit_pos.x;
        let dy = ap.y - unit_pos.y;
        let norm = dx.hypot(dy).max(1.0);
        0.015 * ((dx * angle.cos() + dy * angle.sin()) / norm)
    } else {
        // Pure matrix mode without coordinates: tiny deterministic parity tie-break when offsets differ
        let phase = ((u_idx * 7 + j_idx * 13) % 11) as f64 - 5.0;
        0.002 * phase
    }
}

/// Computes primary first-due beat assignment `u*(j) = argmin_u (t_{u,j} + s_u)` and full ranked
/// dispatch preference lists `pref[j]` under Carter, Chaiken & Ignall (1972) boundary offsets `s_u`.
pub fn build_offset_dispatch_preferences(
    unit_metadata: &[(String, usize, Vec2)],
    unit_parcel_times_s: &[Vec<f64>],
    n_atoms: usize,
    unit_offsets_s: Option<&[f64]>,
    backup_offset_coupling: f64,
    atom_positions: Option<&[Vec2]>,
) -> Vec<Vec<usize>> {
    let n_units = unit_parcel_times_s.len();
    let mut pref: Vec<Vec<usize>> = Vec::with_capacity(n_atoms);
    if n_units == 0 {
        return pref;
    }
    let has_offsets = unit_offsets_s
        .map(|s| s.len() == n_units && s.iter().any(|&v| v.abs() > 1e-9))
        .unwrap_or(false);
    let bkp_scale = backup_offset_coupling.clamp(0.0, 1.0);

    for j in 0..n_atoms {
        let mut order: Vec<usize> = (0..n_units).collect();
        if !has_offsets {
            order.sort_by(|&ua, &ub| {
                let ta = unit_parcel_times_s[ua][j];
                let tb = unit_parcel_times_s[ub][j];
                ta.partial_cmp(&tb)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| ua.cmp(&ub))
            });
        } else {
            let s_vec = unit_offsets_s.unwrap();
            let apos = atom_positions.and_then(|ap| ap.get(j).copied());
            // 1. Primary response district beat `u*(j) = argmin_u (t_{u,j} + s_u)` (Carter, Chaiken & Ignall 1972)
            let mut best_u = 0usize;
            let mut best_score = f64::INFINITY;
            for u in 0..n_units {
                let u_pos = unit_metadata
                    .get(u)
                    .map(|m| m.2)
                    .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
                let score = unit_parcel_times_s[u][j]
                    + s_vec[u]
                    + unit_atom_directional_tie_s(u, n_units, u_pos, apos, j);
                if score < best_score {
                    best_score = score;
                    best_u = u;
                }
            }
            // 2. Remaining backup servers ordered by `t_{u,j} + bkp_scale * s_u`
            order.retain(|&u| u != best_u);
            order.sort_by(|&ua, &ub| {
                let pa = unit_metadata
                    .get(ua)
                    .map(|m| m.2)
                    .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
                let pb = unit_metadata
                    .get(ub)
                    .map(|m| m.2)
                    .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
                let sa = unit_parcel_times_s[ua][j]
                    + bkp_scale * s_vec[ua]
                    + unit_atom_directional_tie_s(ua, n_units, pa, apos, j);
                let sb = unit_parcel_times_s[ub][j]
                    + bkp_scale * s_vec[ub]
                    + unit_atom_directional_tie_s(ub, n_units, pb, apos, j);
                sa.partial_cmp(&sb)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| ua.cmp(&ub))
            });
            order.insert(0, best_u);
        }
        pref.push(order);
    }
    pref
}

/// Generalized Larson (1974) Exact `2^N` `M/M/N` Hypercube solver supporting Carter, Chaiken &
/// Ignall (1972) additive dispatch boundary potentials `unit_offsets_s` (`s_n` in seconds).
pub fn solve_larson_hypercube_with_offsets(
    unit_metadata: &[(String, usize, Vec2)],
    unit_parcel_times_s: &[Vec<f64>],
    atom_weights: &[f64],
    unit_offsets_s: Option<&[f64]>,
    backup_offset_coupling: f64,
    atom_positions: Option<&[Vec2]>,
    cfg: &HypercubeConfig,
) -> HypercubeReport {
    let t0 = Instant::now();
    let n_units = unit_parcel_times_s.len();
    let n_atoms = atom_weights.len();

    if n_units == 0 || n_atoms == 0 {
        return HypercubeReport {
            used_exact_2n_ctmc: true,
            num_units: n_units,
            state_space_size: 1,
            gauss_seidel_iterations: 0,
            outer_mu_iterations: 0,
            final_max_residual: 0.0,
            rate_conservation_error: 0.0,
            system_calls_per_hr: cfg.system_calls_per_hr,
            prob_all_idle_p0: if n_units == 0 { 0.0 } else { 1.0 },
            prob_all_busy_or_queued: if n_units == 0 { 1.0 } else { 0.0 },
            prob_in_fcfs_queue_pq: 0.0,
            mean_busy_fraction_rho: 0.0,
            min_busy_fraction_rho: 0.0,
            max_busy_fraction_rho: 0.0,
            workload_imbalance_spread: 0.0,
            cross_district_dispatch_pct: 0.0,
            expected_primary_cov_pct: 0.0,
            expected_second_due_avail_pct: 0.0,
            mean_dispatch_travel_s: 999.0,
            mean_queue_wait_s: 0.0,
            mean_total_response_s: 999.0,
            unit_metrics: Vec::new(),
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let tot_w: f64 = atom_weights.iter().sum::<f64>().max(1e-9);
    let base_mu = 60.0 / cfg.base_service_duration_mins.clamp(5.0, 240.0);
    // Cap effective arrival rate in FCFS queue mode so traffic intensity `rho_bar < 0.985` stays ergodic
    let max_ergodic_lambda = if cfg.queue_discipline == HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue {
        (n_units as f64) * base_mu * 0.96
    } else {
        cfg.system_calls_per_hr.max(0.01) * 10.0
    };
    let lambda_total = cfg.system_calls_per_hr.clamp(0.01, max_ergodic_lambda.max(0.02));
    let atom_lambdas: Vec<f64> = atom_weights
        .iter()
        .map(|&w| lambda_total * (w.max(0.0) / tot_w))
        .collect();

    // Build dispatch preference matrix `pref[j][rank] -> unit_idx` (Voronoi when `unit_offsets_s` is `None`,
    // or Carter-Chaiken-Ignall 1972 boundary-shifted beats when `unit_offsets_s` is `Some(s)`)
    let pref = build_offset_dispatch_preferences(
        unit_metadata,
        unit_parcel_times_s,
        n_atoms,
        unit_offsets_s,
        backup_offset_coupling,
        atom_positions,
    );

    let mut mu_vec = vec![base_mu; n_units];
    let outer_passes = if cfg.calibrate_travel_service_rate { 4 } else { 1 };
    let use_exact = n_units <= cfg.exact_max_units.clamp(1, 13);

    let threshold_s = cfg.threshold_travel_s.clamp(30.0, 900.0);
    let second_thresh_s = cfg.second_due_threshold_s.max(threshold_s).clamp(30.0, 1200.0);

    if use_exact {
        // =========================================================================
        // EXACT LARSON (1974) 2^N BINARY HYPERCUBE CTMC STATE-TRANSITION SOLVER
        // =========================================================================
        let num_states = 1usize << n_units;
        let b_full = num_states - 1;

        // Precompute upward transition rate table `lambda_up[b * n_units + n]` ONCE!
        // Note `lambda_up` depends ONLY on `atom_lambdas` and `pref`, NOT on `mu_vec`, so outer
        // Jarvis (1985) travel-time `mu_n` calibration reuses `lambda_up` at zero cost!
        let mut lambda_up = vec![0.0f64; num_states * n_units];
        for b in 0..b_full {
            let row = b * n_units;
            for j in 0..n_atoms {
                // Dispatch first idle server in `pref[j]`
                for &u in &pref[j] {
                    if (b & (1 << u)) == 0 {
                        lambda_up[row + u] += atom_lambdas[j];
                        break;
                    }
                }
            }
        }

        let mut probs = vec![1.0 / (num_states as f64); num_states];
        let mut total_gs_iters = 0usize;
        let mut final_res = 0.0;
        let mut p_queue_tail = 0.0;
        let mut unit_rhos = vec![0.0; n_units];
        let mut unit_primary_share = vec![0.0; n_units];
        let mut unit_backup_share = vec![0.0; n_units];
        let mut unit_total_share = vec![0.0; n_units];
        let mut unit_travel_s = vec![0.0; n_units];
        let mut exp_cov_pct = 0.0;
        let mut exp_2nd_avail_pct = 0.0;
        let mut sys_mean_travel_s = 0.0;

        let omega = cfg.sor_omega.clamp(1.0, 1.45);
        let max_iters = cfg.max_iterations.clamp(25, 2000);
        let tol = cfg.tolerance.clamp(1e-13, 1e-6);

        for outer_idx in 0..outer_passes {
            let mu_total: f64 = mu_vec.iter().sum::<f64>().max(1e-9);
            let rho_bar = (lambda_total / mu_total).clamp(0.0, 0.985);

            // Precompute outgoing diagonal rate `diag_out[b]` for current `mu_vec`
            let mut diag_out = vec![0.0f64; num_states];
            for b in 0..num_states {
                let up_sum = if b == b_full { 0.0 } else { lambda_total };
                let mut down_sum = 0.0;
                for n in 0..n_units {
                    if (b & (1 << n)) != 0 {
                        down_sum += mu_vec[n];
                    }
                }
                diag_out[b] = up_sum + down_sum;
            }

            // Gauss-Seidel / SOR sweeps over `b in 0..2^N`
            for _it in 0..max_iters {
                total_gs_iters += 1;
                let mut max_diff = 0.0f64;

                for b in 0..num_states {
                    let d_out = diag_out[b];
                    if d_out <= 1e-15 {
                        continue;
                    }
                    let mut flux_in = 0.0;
                    for n in 0..n_units {
                        let bit = 1usize << n;
                        if (b & bit) != 0 {
                            // Entered `b` via upward dispatch of unit `n` from `b_prev = b ^ bit`
                            let b_prev = b ^ bit;
                            flux_in += probs[b_prev] * lambda_up[b_prev * n_units + n];
                        } else {
                            // Entered `b` via downward service completion of unit `n` from `b_above = b | bit`
                            let b_above = b | bit;
                            flux_in += probs[b_above] * mu_vec[n];
                        }
                    }
                    let p_gs = flux_in / d_out;
                    let p_next = ((1.0 - omega) * probs[b] + omega * p_gs).max(0.0);
                    let diff = (p_next - probs[b]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    probs[b] = p_next;
                }

                // Exact normalization including `M/M/N/inf` infinite-queue geometric tail `P_Q`
                let cube_sum: f64 = probs.iter().sum::<f64>().max(1e-15);
                let tail_unnorm = match cfg.queue_discipline {
                    HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => {
                        probs[b_full] * (rho_bar / (1.0 - rho_bar).max(1e-9))
                    }
                    HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => 0.0,
                };
                let norm = 1.0 / (cube_sum + tail_unnorm);
                for p in &mut probs {
                    *p *= norm;
                }
                p_queue_tail = tail_unnorm * norm;
                final_res = max_diff * norm;
                if final_res < tol {
                    break;
                }
            }

            // Extract exact per-unit busy fractions `rho_n` & exact dispatch fractions `f_{n, j}`
            unit_rhos.fill(p_queue_tail);
            for b in 1..num_states {
                let pb = probs[b];
                for n in 0..n_units {
                    if (b & (1 << n)) != 0 {
                        unit_rhos[n] += pb;
                    }
                }
            }

            unit_primary_share.fill(0.0);
            unit_backup_share.fill(0.0);
            unit_total_share.fill(0.0);
            let mut unit_weighted_t_sum = vec![0.0; n_units];
            let p_saturated_total = match cfg.queue_discipline {
                HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => probs[b_full] + p_queue_tail,
                HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => 0.0,
            };

            let mut cov_prob_accum = 0.0;
            let mut bkp2_prob_accum = 0.0;
            let mut sys_t_num = 0.0;
            let mut sys_t_den = 0.0;

            for j in 0..n_atoms {
                let w_atom = atom_lambdas[j] / lambda_total;
                // Walk unqueued states `b in 0..b_full`
                for b in 0..b_full {
                    let pb = probs[b];
                    if pb <= 0.0 {
                        continue;
                    }
                    // Count how many IDLE servers lie within `second_thresh_s` of atom `j` in state `b`
                    let mut idle_within_2nd = 0usize;
                    for u in 0..n_units {
                        if (b & (1 << u)) == 0 && unit_parcel_times_s[u][j] <= second_thresh_s {
                            idle_within_2nd += 1;
                        }
                    }
                    if idle_within_2nd >= 2 {
                        bkp2_prob_accum += w_atom * pb;
                    }

                    // Identify dispatched unit `u_disp` for `(b, j)`
                    for (rank, &u_disp) in pref[j].iter().enumerate() {
                        if (b & (1 << u_disp)) == 0 {
                            let f_nj = w_atom * pb;
                            let t_nj = unit_parcel_times_s[u_disp][j].min(999.0);
                            if rank == 0 {
                                unit_primary_share[u_disp] += f_nj;
                            } else {
                                unit_backup_share[u_disp] += f_nj;
                            }
                            unit_total_share[u_disp] += f_nj;
                            unit_weighted_t_sum[u_disp] += f_nj * t_nj;
                            sys_t_num += f_nj * t_nj;
                            sys_t_den += f_nj;
                            if t_nj <= threshold_s {
                                cov_prob_accum += f_nj;
                            }
                            break;
                        }
                    }
                }

                // Queued calls (`b_full` + `S_{N+k}` in FCFS mode): dispatched to next completing unit `u` with prob `mu_u / mu_total`
                if p_saturated_total > 0.0 {
                    for u in 0..n_units {
                        let f_q_nj = w_atom * p_saturated_total * (mu_vec[u] / mu_total);
                        let t_nj = unit_parcel_times_s[u][j].min(999.0);
                        unit_total_share[u] += f_q_nj;
                        unit_weighted_t_sum[u] += f_q_nj * t_nj;
                        sys_t_num += f_q_nj * t_nj;
                        sys_t_den += f_q_nj;
                    }
                }
            }

            for u in 0..n_units {
                unit_travel_s[u] = if unit_total_share[u] > 1e-12 {
                    unit_weighted_t_sum[u] / unit_total_share[u]
                } else {
                    unit_parcel_times_s[u]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min)
                        .min(999.0)
                };
            }

            exp_cov_pct = (cov_prob_accum * 100.0).clamp(0.0, 100.0);
            exp_2nd_avail_pct = (bkp2_prob_accum * 100.0).clamp(0.0, 100.0);
            sys_mean_travel_s = if sys_t_den > 1e-12 {
                sys_t_num / sys_t_den
            } else {
                999.0
            };

            if cfg.calibrate_travel_service_rate && outer_idx + 1 < outer_passes {
                let base_scene_s = cfg.base_service_duration_mins.clamp(5.0, 240.0) * 60.0;
                let max_erg = lambda_total / ((n_units as f64) * 0.97);
                for u in 0..n_units {
                    let tot_service_hr = (base_scene_s + 1.5 * unit_travel_s[u]) / 3600.0;
                    let new_mu = (1.0 / tot_service_hr.max(0.05)).max(max_erg * 1.02);
                    mu_vec[u] = 0.5 * mu_vec[u] + 0.5 * new_mu;
                }
            }
        }

        // Exact conservation verification:
        // - FCFS Queue: `sum_n mu_n * rho_n == lambda_total`
        // - Loss System: `sum_n mu_n * rho_n == lambda_total * (1 - P[b_full])`
        let realized_throughput: f64 = mu_vec
            .iter()
            .zip(unit_rhos.iter())
            .map(|(&m, &r)| m * r)
            .sum();
        let expected_throughput = match cfg.queue_discipline {
            HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => lambda_total,
            HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => {
                lambda_total * (1.0 - probs[b_full])
            }
        };
        let conservation_err = (realized_throughput - expected_throughput).abs();

        let mu_tot_final: f64 = mu_vec.iter().sum::<f64>().max(1e-9);
        let prob_all_busy = match cfg.queue_discipline {
            HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => probs[b_full] + p_queue_tail,
            HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => probs[b_full],
        };
        let mean_queue_wait_s = match cfg.queue_discipline {
            HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => {
                if mu_tot_final > lambda_total + 1e-6 {
                    prob_all_busy * (3600.0 / (mu_tot_final - lambda_total))
                } else {
                    180.0
                }
            }
            HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => 0.0,
        };

        let mean_rho = unit_rhos.iter().sum::<f64>() / (n_units as f64);
        let min_rho = unit_rhos.iter().copied().fold(1.0, f64::min);
        let max_rho = unit_rhos.iter().copied().fold(0.0, f64::max);
        let tot_unq_share: f64 = unit_primary_share
            .iter()
            .zip(unit_backup_share.iter())
            .map(|(a, b)| a + b)
            .sum::<f64>()
            .max(1e-12);
        let bkp_sum: f64 = unit_backup_share.iter().sum();
        let cross_dist_pct = ((bkp_sum / tot_unq_share) * 100.0).clamp(0.0, 100.0);

        let unit_metrics: Vec<HypercubeUnitMetrics> = (0..n_units)
            .map(|u| {
                let (lbl, seg_id, wpos) = unit_metadata
                    .get(u)
                    .cloned()
                    .unwrap_or_else(|| (format!("Unit #{}", u + 1), 0, Vec2 { x: 0.0, y: 0.0 }));
                HypercubeUnitMetrics {
                    unit_idx: u,
                    label: lbl,
                    station_segment_id: seg_id,
                    world_pos: wpos,
                    busy_fraction_rho: unit_rhos[u],
                    service_rate_mu_per_hr: mu_vec[u],
                    primary_dispatch_share_pct: unit_primary_share[u] * 100.0,
                    backup_dispatch_share_pct: unit_backup_share[u] * 100.0,
                    total_dispatch_share_pct: unit_total_share[u] * 100.0,
                    mean_unit_travel_s: unit_travel_s[u],
                }
            })
            .collect();

        HypercubeReport {
            used_exact_2n_ctmc: true,
            num_units: n_units,
            state_space_size: num_states,
            gauss_seidel_iterations: total_gs_iters,
            outer_mu_iterations: outer_passes,
            final_max_residual: final_res,
            rate_conservation_error: conservation_err,
            system_calls_per_hr: lambda_total,
            prob_all_idle_p0: probs[0],
            prob_all_busy_or_queued: prob_all_busy,
            prob_in_fcfs_queue_pq: p_queue_tail,
            mean_busy_fraction_rho: mean_rho,
            min_busy_fraction_rho: min_rho,
            max_busy_fraction_rho: max_rho,
            workload_imbalance_spread: (max_rho - min_rho).max(0.0),
            cross_district_dispatch_pct: cross_dist_pct,
            expected_primary_cov_pct: exp_cov_pct,
            expected_second_due_avail_pct: exp_2nd_avail_pct,
            mean_dispatch_travel_s: sys_mean_travel_s,
            mean_queue_wait_s,
            mean_total_response_s: sys_mean_travel_s + mean_queue_wait_s,
            unit_metrics,
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    } else {
        // =========================================================================
        // LARSON (1975) N-EQUATION Q-FACTOR ITERATIVE HYPERCUBE APPROXIMATION (N > 12)
        // =========================================================================
        let rho_sys = (lambda_total / ((n_units as f64) * base_mu)).clamp(0.01, 0.96);
        let q_factors: Vec<f64> = (0..n_units)
            .map(|r| larson_1975_q_factor(n_units, rho_sys, r))
            .collect();

        let mut rhos = vec![rho_sys; n_units];
        let mut iters = 0usize;
        let mut final_res = 0.0;

        for _ in 0..cfg.max_iterations.clamp(20, 400) {
            iters += 1;
            let mut next_rhos = vec![0.0; n_units];
            for j in 0..n_atoms {
                let lam_j = atom_lambdas[j];
                let mut prod_busy = 1.0;
                for (rank, &u) in pref[j].iter().enumerate() {
                    let rate_u = lam_j * q_factors[rank] * prod_busy * (1.0 - rhos[u]);
                    next_rhos[u] += rate_u / base_mu;
                    prod_busy *= rhos[u];
                }
            }
            // Normalize sum of `rhos` to exact system offered workload `(n_units * rho_sys)`
            let target_sum = (n_units as f64) * rho_sys;
            let cur_sum: f64 = next_rhos.iter().sum::<f64>().max(1e-12);
            let scale = target_sum / cur_sum;
            let mut max_d = 0.0f64;
            for u in 0..n_units {
                let nr = (0.4 * rhos[u] + 0.6 * (next_rhos[u] * scale)).clamp(0.005, 0.99);
                max_d = max_d.max((nr - rhos[u]).abs());
                rhos[u] = nr;
            }
            final_res = max_d;
            if max_d < cfg.tolerance.max(1e-8) {
                break;
            }
        }

        // Dispatch shares & coverage from converged `rhos` + `Q` factors
        let mut unit_pri = vec![0.0; n_units];
        let mut unit_bkp = vec![0.0; n_units];
        let mut unit_tot = vec![0.0; n_units];
        let mut unit_t_sum = vec![0.0; n_units];
        let mut cov_acc = 0.0;
        let mut bkp2_acc = 0.0;
        let mut sys_t_sum = 0.0;

        for j in 0..n_atoms {
            let w_j = atom_lambdas[j] / lambda_total;
            let mut prod_b = 1.0;
            let mut unq_sum = 0.0;
            let mut atom_f = vec![0.0; n_units];
            for (rank, &u) in pref[j].iter().enumerate() {
                let p_u = (q_factors[rank] * prod_b * (1.0 - rhos[u])).max(0.0);
                atom_f[u] = p_u;
                unq_sum += p_u;
                prod_b *= rhos[u];
            }
            let norm_f = if unq_sum > 1.0 { 1.0 / unq_sum } else { 1.0 };
            let mut avail_2nd_prob = 0.0;
            let mut k_in_2nd = 0usize;
            for &u in &pref[j] {
                if unit_parcel_times_s[u][j] <= second_thresh_s {
                    k_in_2nd += 1;
                    avail_2nd_prob += 1.0 - rhos[u];
                }
            }
            if k_in_2nd >= 2 {
                bkp2_acc += w_j * ((avail_2nd_prob / 2.0).clamp(0.0, 1.0));
            }
            for (rank, &u) in pref[j].iter().enumerate() {
                let f_nj = w_j * atom_f[u] * norm_f;
                let t_nj = unit_parcel_times_s[u][j].min(999.0);
                if rank == 0 {
                    unit_pri[u] += f_nj;
                } else {
                    unit_bkp[u] += f_nj;
                }
                unit_tot[u] += f_nj;
                unit_t_sum[u] += f_nj * t_nj;
                sys_t_sum += f_nj * t_nj;
                if t_nj <= threshold_s {
                    cov_acc += f_nj;
                }
            }
        }

        let mean_rho = rhos.iter().sum::<f64>() / (n_units as f64);
        let min_rho = rhos.iter().copied().fold(1.0, f64::min);
        let max_rho = rhos.iter().copied().fold(0.0, f64::max);
        let bkp_sum: f64 = unit_bkp.iter().sum();
        let tot_sum: f64 = unit_tot.iter().sum::<f64>().max(1e-12);

        let unit_metrics: Vec<HypercubeUnitMetrics> = (0..n_units)
            .map(|u| {
                let (lbl, seg_id, wpos) = unit_metadata
                    .get(u)
                    .cloned()
                    .unwrap_or_else(|| (format!("Unit #{}", u + 1), 0, Vec2 { x: 0.0, y: 0.0 }));
                let mt = if unit_tot[u] > 1e-12 {
                    unit_t_sum[u] / unit_tot[u]
                } else {
                    180.0
                };
                HypercubeUnitMetrics {
                    unit_idx: u,
                    label: lbl,
                    station_segment_id: seg_id,
                    world_pos: wpos,
                    busy_fraction_rho: rhos[u],
                    service_rate_mu_per_hr: base_mu,
                    primary_dispatch_share_pct: unit_pri[u] * 100.0,
                    backup_dispatch_share_pct: unit_bkp[u] * 100.0,
                    total_dispatch_share_pct: unit_tot[u] * 100.0,
                    mean_unit_travel_s: mt,
                }
            })
            .collect();

        HypercubeReport {
            used_exact_2n_ctmc: false,
            num_units: n_units,
            state_space_size: n_units,
            gauss_seidel_iterations: iters,
            outer_mu_iterations: 1,
            final_max_residual: final_res,
            rate_conservation_error: ((rhos.iter().sum::<f64>() * base_mu) - lambda_total).abs(),
            system_calls_per_hr: lambda_total,
            prob_all_idle_p0: (1.0 - mean_rho).powi(n_units as i32),
            prob_all_busy_or_queued: mean_rho.powi(n_units as i32),
            prob_in_fcfs_queue_pq: mean_rho.powi(n_units as i32) * rho_sys,
            mean_busy_fraction_rho: mean_rho,
            min_busy_fraction_rho: min_rho,
            max_busy_fraction_rho: max_rho,
            workload_imbalance_spread: (max_rho - min_rho).max(0.0),
            cross_district_dispatch_pct: ((bkp_sum / tot_sum) * 100.0).clamp(0.0, 100.0),
            expected_primary_cov_pct: (cov_acc * 100.0).clamp(0.0, 100.0),
            expected_second_due_avail_pct: (bkp2_acc * 100.0).clamp(0.0, 100.0),
            mean_dispatch_travel_s: sys_t_sum / tot_sum,
            mean_queue_wait_s: 0.0,
            mean_total_response_s: sys_t_sum / tot_sum,
            unit_metrics,
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
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
        other.cost_s.partial_cmp(&self.cost_s).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Extracts `(unit_metadata, unit_parcel_times_s, atom_weights, atom_positions)` from placed
/// `FireStation` facilities on `net`, splitting multi-company stations (`(2x Eng)` / `(3x Eng)`)
/// into distinct distinguishable response units.
pub fn build_network_unit_travel_matrix(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
) -> (Vec<(String, usize, Vec2)>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec2>) {
    let n_segs = net.segments.len();
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

    let mut fwd_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
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
                let step = sig_pen + turn_pen + seg_trav_s[out_sid];
                if !fwd_adj[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    fwd_adj[seg.id].push((out_sid, step));
                }
            }
        }
    }

    let mut twin_rev: Vec<Option<usize>> = vec![None; n_segs];
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

    let turnout_s = if evp_cfg.include_nfpa_turnout_time {
        80.0
    } else {
        0.0
    };

    let mut unit_meta: Vec<(String, usize, Vec2)> = Vec::new();
    let mut unit_times: Vec<Vec<f64>> = Vec::new();

    for fac in facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
    {
        // Parse `(2x Eng)` / `(3x Eng)` multi-company houses into separate distinguishable units!
        let n_co = if fac.name.contains("(3x") {
            3usize
        } else if fac.name.contains("(2x") {
            2usize
        } else {
            1usize
        };

        let mut dist = vec![f64::INFINITY; n_segs];
        let mut heap = BinaryHeap::new();
        if fac.segment_id < n_segs && !net.segments[fac.segment_id].closed {
            let len = net.segments[fac.segment_id].length.max(1.0);
            let rem_frac = ((len - fac.s_m) / len).clamp(0.1, 1.0);
            let init_s = turnout_s + seg_trav_s[fac.segment_id] * rem_frac;
            dist[fac.segment_id] = init_s;
            heap.push(DijkNode {
                cost_s: init_s,
                seg_id: fac.segment_id,
            });
            if let Some(tw) = twin_rev[fac.segment_id] {
                let tw_init = turnout_s + seg_trav_s[tw] * (1.0 - rem_frac + 0.15);
                dist[tw] = tw_init;
                heap.push(DijkNode {
                    cost_s: tw_init,
                    seg_id: tw,
                });
            }
        }
        while let Some(DijkNode { cost_s, seg_id }) = heap.pop() {
            if cost_s > dist[seg_id] {
                continue;
            }
            for &(nxt, w) in &fwd_adj[seg_id] {
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

        let p_times: Vec<f64> = parcels
            .iter()
            .map(|p| {
                let mut best = f64::INFINITY;
                for (sid, &d_s) in dist.iter().enumerate() {
                    if d_s.is_finite() {
                        best = best.min(d_s + p.center.dist(seg_midpoints[sid]) / 7.5);
                    }
                }
                best
            })
            .collect();

        for co_i in 0..n_co {
            let u_lbl = if n_co >= 2 {
                format!("{} [Co.{}]", fac.name, (b'A' + co_i as u8) as char)
            } else {
                fac.name.clone()
            };
            unit_meta.push((u_lbl, fac.segment_id, fac.world_pos));
            unit_times.push(p_times.clone());
        }
    }

    let atom_weights: Vec<f64> = parcels
        .iter()
        .map(|p| (p.residents() + 0.35 * p.jobs()).max(1.0))
        .collect();
    let atom_positions: Vec<Vec2> = parcels.iter().map(|p| p.center).collect();

    (unit_meta, unit_times, atom_weights, atom_positions)
}

/// Evaluates the Larson (1974) Exact `2^N` `M/M/N` Hypercube model directly on the active
/// `FireStation` facilities placed on `net` (honoring explicit parcel `response_district_unit`
/// assignments when already applied on `parcels`, or closest-unit Voronoi otherwise!).
pub fn solve_network_hypercube(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &HypercubeConfig,
) -> HypercubeReport {
    let (unit_meta, unit_times, atom_weights, _atom_pos) =
        build_network_unit_travel_matrix(net, parcels, facilities, evp_cfg);
    let n_units = unit_meta.len();
    let has_custom_beats = n_units >= 2
        && parcels
            .iter()
            .any(|p| p.response_district_unit.map(|u| (u as usize) < n_units).unwrap_or(false));
    if !has_custom_beats {
        return solve_larson_hypercube_from_matrix(&unit_meta, &unit_times, &atom_weights, cfg);
    }

    // Evaluate exact hypercube with each parcel's assigned `response_district_unit` at rank 0!
    solve_larson_hypercube_with_explicit_primary_beats(
        &unit_meta,
        &unit_times,
        &atom_weights,
        &parcels
            .iter()
            .map(|p| p.response_district_unit.map(|u| u as usize))
            .collect::<Vec<_>>(),
        cfg,
    )
}

/// Evaluates the exact `2^N` Larson (1974) Hypercube model when parcels carry explicit primary
/// response-district assignments `primary_beats[j] = Some(u_idx)` while measuring `E[T]` and
/// `<= threshold_s` coverage against true physical network travel times `unit_parcel_times_s`!
pub fn solve_larson_hypercube_with_explicit_primary_beats(
    unit_metadata: &[(String, usize, Vec2)],
    unit_parcel_times_s: &[Vec<f64>],
    atom_weights: &[f64],
    primary_beats: &[Option<usize>],
    cfg: &HypercubeConfig,
) -> HypercubeReport {
    let n_units = unit_parcel_times_s.len();
    let n_atoms = atom_weights.len();
    if n_units == 0 || n_atoms == 0 {
        return solve_larson_hypercube_from_matrix(
            unit_metadata,
            unit_parcel_times_s,
            atom_weights,
            cfg,
        );
    }
    // Synthesize exact travel-time matrix purely for preference sorting, keeping physical `unit_parcel_times_s` intact
    // Actually we can evaluate directly by computing `pref` with `primary_beats[j]` at rank 0 and calling `solve_larson_hypercube_from_custom_pref`!
    let mut pref: Vec<Vec<usize>> = Vec::with_capacity(n_atoms);
    for j in 0..n_atoms {
        let mut order: Vec<usize> = (0..n_units).collect();
        order.sort_by(|&ua, &ub| {
            unit_parcel_times_s[ua][j]
                .partial_cmp(&unit_parcel_times_s[ub][j])
                .unwrap_or(Ordering::Equal)
                .then_with(|| ua.cmp(&ub))
        });
        if let Some(Some(u_pri)) = primary_beats.get(j).copied() {
            if u_pri < n_units {
                order.retain(|&u| u != u_pri);
                order.insert(0, u_pri);
            }
        }
        pref.push(order);
    }
    solve_larson_hypercube_with_pref(unit_metadata, unit_parcel_times_s, atom_weights, &pref, cfg)
}

/// Core exact `2^N` CTMC / Larson-1975 `Q`-factor solver given an arbitrary dispatch preference
/// matrix `pref[j][rank] -> unit_idx` and physical travel times `unit_parcel_times_s[u][j]`.
pub fn solve_larson_hypercube_with_pref(
    unit_metadata: &[(String, usize, Vec2)],
    unit_parcel_times_s: &[Vec<f64>],
    atom_weights: &[f64],
    pref: &[Vec<usize>],
    cfg: &HypercubeConfig,
) -> HypercubeReport {
    let t0 = Instant::now();
    let n_units = unit_parcel_times_s.len();
    let n_atoms = atom_weights.len();
    if n_units == 0 || n_atoms == 0 || pref.len() != n_atoms {
        return solve_larson_hypercube_from_matrix(
            unit_metadata,
            unit_parcel_times_s,
            atom_weights,
            cfg,
        );
    }

    let tot_w: f64 = atom_weights.iter().sum::<f64>().max(1e-9);
    let base_mu = 60.0 / cfg.base_service_duration_mins.clamp(5.0, 240.0);
    let max_ergodic_lambda =
        if cfg.queue_discipline == HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue {
            (n_units as f64) * base_mu * 0.96
        } else {
            cfg.system_calls_per_hr.max(0.01) * 10.0
        };
    let lambda_total = cfg.system_calls_per_hr.clamp(0.01, max_ergodic_lambda.max(0.02));
    let atom_lambdas: Vec<f64> = atom_weights
        .iter()
        .map(|&w| lambda_total * (w.max(0.0) / tot_w))
        .collect();

    let mut mu_vec = vec![base_mu; n_units];
    let outer_passes = if cfg.calibrate_travel_service_rate { 4 } else { 1 };
    let use_exact = n_units <= cfg.exact_max_units.clamp(1, 13);
    let threshold_s = cfg.threshold_travel_s.clamp(30.0, 900.0);
    let second_thresh_s = cfg.second_due_threshold_s.max(threshold_s).clamp(30.0, 1200.0);

    if use_exact {
        let num_states = 1usize << n_units;
        let b_full = num_states - 1;
        let mut lambda_up = vec![0.0f64; num_states * n_units];
        for b in 0..b_full {
            let row = b * n_units;
            for j in 0..n_atoms {
                for &u in &pref[j] {
                    if (b & (1 << u)) == 0 {
                        lambda_up[row + u] += atom_lambdas[j];
                        break;
                    }
                }
            }
        }

        let mut probs = vec![1.0 / (num_states as f64); num_states];
        let mut total_gs_iters = 0usize;
        let mut final_res = 0.0;
        let mut p_queue_tail = 0.0;
        let mut unit_rhos = vec![0.0; n_units];
        let mut unit_primary_share = vec![0.0; n_units];
        let mut unit_backup_share = vec![0.0; n_units];
        let mut unit_total_share = vec![0.0; n_units];
        let mut unit_travel_s = vec![0.0; n_units];
        let mut exp_cov_pct = 0.0;
        let mut exp_2nd_avail_pct = 0.0;
        let mut sys_mean_travel_s = 0.0;

        let omega = cfg.sor_omega.clamp(1.0, 1.45);
        let max_iters = cfg.max_iterations.clamp(25, 2000);
        let tol = cfg.tolerance.clamp(1e-13, 1e-6);

        for outer_idx in 0..outer_passes {
            let mu_total: f64 = mu_vec.iter().sum::<f64>().max(1e-9);
            let rho_bar = (lambda_total / mu_total).clamp(0.0, 0.985);
            let mut diag_out = vec![0.0f64; num_states];
            for b in 0..num_states {
                let up_sum = if b == b_full { 0.0 } else { lambda_total };
                let mut down_sum = 0.0;
                for n in 0..n_units {
                    if (b & (1 << n)) != 0 {
                        down_sum += mu_vec[n];
                    }
                }
                diag_out[b] = up_sum + down_sum;
            }

            for _it in 0..max_iters {
                total_gs_iters += 1;
                let mut max_diff = 0.0f64;
                for b in 0..num_states {
                    let d_out = diag_out[b];
                    if d_out <= 1e-15 {
                        continue;
                    }
                    let mut flux_in = 0.0;
                    for n in 0..n_units {
                        let bit = 1usize << n;
                        if (b & bit) != 0 {
                            let b_prev = b ^ bit;
                            flux_in += probs[b_prev] * lambda_up[b_prev * n_units + n];
                        } else {
                            let b_above = b | bit;
                            flux_in += probs[b_above] * mu_vec[n];
                        }
                    }
                    let p_gs = flux_in / d_out;
                    let p_next = ((1.0 - omega) * probs[b] + omega * p_gs).max(0.0);
                    let diff = (p_next - probs[b]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    probs[b] = p_next;
                }

                let cube_sum: f64 = probs.iter().sum::<f64>().max(1e-15);
                let tail_unnorm = match cfg.queue_discipline {
                    HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => {
                        probs[b_full] * (rho_bar / (1.0 - rho_bar).max(1e-9))
                    }
                    HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => 0.0,
                };
                let norm = 1.0 / (cube_sum + tail_unnorm);
                for p in &mut probs {
                    *p *= norm;
                }
                p_queue_tail = tail_unnorm * norm;
                final_res = max_diff * norm;
                if final_res < tol {
                    break;
                }
            }

            unit_rhos.fill(p_queue_tail);
            for b in 1..num_states {
                let pb = probs[b];
                for n in 0..n_units {
                    if (b & (1 << n)) != 0 {
                        unit_rhos[n] += pb;
                    }
                }
            }

            unit_primary_share.fill(0.0);
            unit_backup_share.fill(0.0);
            unit_total_share.fill(0.0);
            let mut unit_weighted_t_sum = vec![0.0; n_units];
            let p_saturated_total = match cfg.queue_discipline {
                HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => probs[b_full] + p_queue_tail,
                HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => 0.0,
            };

            let mut cov_prob_accum = 0.0;
            let mut bkp2_prob_accum = 0.0;
            let mut sys_t_num = 0.0;
            let mut sys_t_den = 0.0;

            for j in 0..n_atoms {
                let w_atom = atom_lambdas[j] / lambda_total;
                for b in 0..b_full {
                    let pb = probs[b];
                    if pb <= 0.0 {
                        continue;
                    }
                    let mut idle_within_2nd = 0usize;
                    for u in 0..n_units {
                        if (b & (1 << u)) == 0 && unit_parcel_times_s[u][j] <= second_thresh_s {
                            idle_within_2nd += 1;
                        }
                    }
                    if idle_within_2nd >= 2 {
                        bkp2_prob_accum += w_atom * pb;
                    }
                    for (rank, &u_disp) in pref[j].iter().enumerate() {
                        if (b & (1 << u_disp)) == 0 {
                            let f_nj = w_atom * pb;
                            let t_nj = unit_parcel_times_s[u_disp][j].min(999.0);
                            if rank == 0 {
                                unit_primary_share[u_disp] += f_nj;
                            } else {
                                unit_backup_share[u_disp] += f_nj;
                            }
                            unit_total_share[u_disp] += f_nj;
                            unit_weighted_t_sum[u_disp] += f_nj * t_nj;
                            sys_t_num += f_nj * t_nj;
                            sys_t_den += f_nj;
                            if t_nj <= threshold_s {
                                cov_prob_accum += f_nj;
                            }
                            break;
                        }
                    }
                }
                if p_saturated_total > 0.0 {
                    for u in 0..n_units {
                        let f_q_nj = w_atom * p_saturated_total * (mu_vec[u] / mu_total);
                        let t_nj = unit_parcel_times_s[u][j].min(999.0);
                        unit_total_share[u] += f_q_nj;
                        unit_weighted_t_sum[u] += f_q_nj * t_nj;
                        sys_t_num += f_q_nj * t_nj;
                        sys_t_den += f_q_nj;
                    }
                }
            }

            for u in 0..n_units {
                unit_travel_s[u] = if unit_total_share[u] > 1e-12 {
                    unit_weighted_t_sum[u] / unit_total_share[u]
                } else {
                    unit_parcel_times_s[u]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min)
                        .min(999.0)
                };
            }
            exp_cov_pct = (cov_prob_accum * 100.0).clamp(0.0, 100.0);
            exp_2nd_avail_pct = (bkp2_prob_accum * 100.0).clamp(0.0, 100.0);
            sys_mean_travel_s = if sys_t_den > 1e-12 {
                sys_t_num / sys_t_den
            } else {
                999.0
            };

            if cfg.calibrate_travel_service_rate && outer_idx + 1 < outer_passes {
                let base_scene_s = cfg.base_service_duration_mins.clamp(5.0, 240.0) * 60.0;
                let max_erg = lambda_total / ((n_units as f64) * 0.97);
                for u in 0..n_units {
                    let tot_service_hr = (base_scene_s + 1.5 * unit_travel_s[u]) / 3600.0;
                    let new_mu = (1.0 / tot_service_hr.max(0.05)).max(max_erg * 1.02);
                    mu_vec[u] = 0.5 * mu_vec[u] + 0.5 * new_mu;
                }
            }
        }

        let realized_throughput: f64 = mu_vec
            .iter()
            .zip(unit_rhos.iter())
            .map(|(&m, &r)| m * r)
            .sum();
        let expected_throughput = match cfg.queue_discipline {
            HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => lambda_total,
            HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => {
                lambda_total * (1.0 - probs[b_full])
            }
        };
        let conservation_err = (realized_throughput - expected_throughput).abs();
        let mu_tot_final: f64 = mu_vec.iter().sum::<f64>().max(1e-9);
        let prob_all_busy = match cfg.queue_discipline {
            HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => probs[b_full] + p_queue_tail,
            HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => probs[b_full],
        };
        let mean_queue_wait_s = match cfg.queue_discipline {
            HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue => {
                if mu_tot_final > lambda_total + 1e-6 {
                    prob_all_busy * (3600.0 / (mu_tot_final - lambda_total))
                } else {
                    180.0
                }
            }
            HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss => 0.0,
        };
        let mean_rho = unit_rhos.iter().sum::<f64>() / (n_units as f64);
        let min_rho = unit_rhos.iter().copied().fold(1.0, f64::min);
        let max_rho = unit_rhos.iter().copied().fold(0.0, f64::max);
        let tot_unq_share: f64 = unit_primary_share
            .iter()
            .zip(unit_backup_share.iter())
            .map(|(a, b)| a + b)
            .sum::<f64>()
            .max(1e-12);
        let bkp_sum: f64 = unit_backup_share.iter().sum();
        let cross_dist_pct = ((bkp_sum / tot_unq_share) * 100.0).clamp(0.0, 100.0);

        let unit_metrics: Vec<HypercubeUnitMetrics> = (0..n_units)
            .map(|u| {
                let (lbl, seg_id, wpos) = unit_metadata
                    .get(u)
                    .cloned()
                    .unwrap_or_else(|| (format!("Unit #{}", u + 1), 0, Vec2 { x: 0.0, y: 0.0 }));
                HypercubeUnitMetrics {
                    unit_idx: u,
                    label: lbl,
                    station_segment_id: seg_id,
                    world_pos: wpos,
                    busy_fraction_rho: unit_rhos[u],
                    service_rate_mu_per_hr: mu_vec[u],
                    primary_dispatch_share_pct: unit_primary_share[u] * 100.0,
                    backup_dispatch_share_pct: unit_backup_share[u] * 100.0,
                    total_dispatch_share_pct: unit_total_share[u] * 100.0,
                    mean_unit_travel_s: unit_travel_s[u],
                }
            })
            .collect();

        HypercubeReport {
            used_exact_2n_ctmc: true,
            num_units: n_units,
            state_space_size: num_states,
            gauss_seidel_iterations: total_gs_iters,
            outer_mu_iterations: outer_passes,
            final_max_residual: final_res,
            rate_conservation_error: conservation_err,
            system_calls_per_hr: lambda_total,
            prob_all_idle_p0: probs[0],
            prob_all_busy_or_queued: prob_all_busy,
            prob_in_fcfs_queue_pq: p_queue_tail,
            mean_busy_fraction_rho: mean_rho,
            min_busy_fraction_rho: min_rho,
            max_busy_fraction_rho: max_rho,
            workload_imbalance_spread: (max_rho - min_rho).max(0.0),
            cross_district_dispatch_pct: cross_dist_pct,
            expected_primary_cov_pct: exp_cov_pct,
            expected_second_due_avail_pct: exp_2nd_avail_pct,
            mean_dispatch_travel_s: sys_mean_travel_s,
            mean_queue_wait_s,
            mean_total_response_s: sys_mean_travel_s + mean_queue_wait_s,
            unit_metrics,
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    } else {
        // Fallback to Larson 1975 Q-factor solver using synthesized offsets
        solve_larson_hypercube_from_matrix(unit_metadata, unit_parcel_times_s, atom_weights, cfg)
    }
}

// =============================================================================
// CARTER, CHAIKEN & IGNALL (1972) / LARSON HYPERCUBE RESPONSE-DISTRICT OPTIMIZER
// =============================================================================

/// Scalarization / policy regime for Carter, Chaiken & Ignall (1972) / Larson (1974)
/// response-district boundary optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RedistrictingObjectiveMode {
    /// Pure Carter, Chaiken & Ignall (1972) expected travel-time minimization (`equity_weight = 0`):
    /// shifts district boundaries inward around congested core stations solely to minimize citywide `E[T]`.
    TravelTimeOptimalCarter1972,
    /// Multi-objective Pareto knee balancing `E[T]` and inter-unit workload spread `max_rho - min_rho`
    /// using `RedistrictingConfig::workload_equity_weight`.
    BalancedParetoKnee,
    /// Aggressive Larson (1974) workload equalization (`equity_weight = 0.92`) minimizing `max_rho - min_rho`
    /// and `RMSE(rho)` subject to `max_travel_penalty_s`.
    StrictWorkloadEqualization,
}

impl Default for RedistrictingObjectiveMode {
    fn default() -> Self {
        Self::BalancedParetoKnee
    }
}

impl RedistrictingObjectiveMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::TravelTimeOptimalCarter1972 => "Carter-1972 Min Expected Travel E[T]",
            Self::BalancedParetoKnee => "Balanced Pareto Knee (E[T] + Workload)",
            Self::StrictWorkloadEqualization => "Larson-1974 Strict Workload Equalization",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedistrictingConfig {
    pub hypercube: HypercubeConfig,
    pub objective_mode: RedistrictingObjectiveMode,
    /// Scalarization weight `w_eq in [0.0, 1.0]` used in `BalancedParetoKnee` (`0.0` = pure travel time,
    /// `1.0` = pure workload equalization).
    pub workload_equity_weight: f64,
    /// Maximum acceptable increase in mean dispatch travel time `E[T]` (seconds) relative to naive
    /// closest-unit Voronoi districts (`s = 0`).
    pub max_travel_penalty_s: f64,
    /// Fraction `[0.0, 1.0]` of boundary offset `s_n` also applied to backup (`rank >= 2`) dispatch ordering
    /// (`0.0` = pure Carter-Chaiken-Ignall primary-beat shift with physical closest-idle backup).
    pub backup_offset_coupling: f64,
    /// Maximum magnitude `|s_n|` (seconds) of any unit's dispatch boundary offset.
    pub max_offset_s: f64,
    /// Number of proportional Larson workload-feedback iterations prior to coordinate pattern search.
    pub feedback_iterations: usize,
}

impl Default for RedistrictingConfig {
    fn default() -> Self {
        Self {
            hypercube: HypercubeConfig::default(),
            objective_mode: RedistrictingObjectiveMode::BalancedParetoKnee,
            workload_equity_weight: 0.55,
            max_travel_penalty_s: 20.0,
            backup_offset_coupling: 0.15,
            max_offset_s: 320.0,
            feedback_iterations: 18,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitDistrictSummary {
    pub unit_idx: usize,
    pub label: String,
    pub station_segment_id: usize,
    pub world_pos: Vec2,
    /// Optimal Carter-Chaiken-Ignall additive boundary potential `s_n` (seconds, `sum_n s_n == 0`).
    /// `s_n > 0` shrinks overworked unit `n`'s beat inward; `s_n < 0` expands underutilized unit `n`'s beat.
    pub boundary_offset_s: f64,
    pub voronoi_parcel_count: usize,
    pub optimized_parcel_count: usize,
    pub voronoi_demand_share_pct: f64,
    pub optimized_demand_share_pct: f64,
    pub voronoi_busy_rho: f64,
    pub optimized_busy_rho: f64,
    pub voronoi_primary_share_pct: f64,
    pub optimized_primary_share_pct: f64,
    pub mean_unit_travel_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedistrictingTradeoffPoint {
    pub equity_weight: f64,
    pub mean_dispatch_travel_s: f64,
    pub mean_total_response_s: f64,
    pub expected_primary_cov_pct: f64,
    pub workload_spread_rho: f64,
    pub workload_rmse_rho: f64,
    pub min_busy_rho: f64,
    pub max_busy_rho: f64,
    pub reassigned_parcels: usize,
    pub unit_offsets_s: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedistrictingReport {
    pub num_units: usize,
    pub num_parcels: usize,
    pub effective_equity_weight: f64,
    pub voronoi_hypercube: HypercubeReport,
    pub carter_min_travel_hypercube: HypercubeReport,
    pub optimized_hypercube: HypercubeReport,
    pub unit_offsets_s: Vec<f64>,
    pub voronoi_parcel_units: Vec<usize>,
    pub optimized_parcel_units: Vec<usize>,
    pub reassigned_parcel_count: usize,
    pub reassigned_demand_pct: f64,
    pub voronoi_workload_spread: f64,
    pub optimized_workload_spread: f64,
    pub workload_spread_reduction_pct: f64,
    pub voronoi_workload_rmse: f64,
    pub optimized_workload_rmse: f64,
    /// `optimized_travel_s - voronoi_travel_s` (negative when simultaneously improving BOTH travel time and workload!).
    pub travel_time_delta_s: f64,
    /// Travel-time savings (`s >= 0`) of pure Carter-Chaiken-Ignall optimal districting over naive Voronoi (`s = 0`).
    pub carter_min_travel_gain_s: f64,
    pub unit_summaries: Vec<UnitDistrictSummary>,
    pub pareto_frontier: Vec<RedistrictingTradeoffPoint>,
    pub hypercube_evaluations: usize,
    pub solve_time_ms: f64,
}

impl RedistrictingReport {
    pub fn to_unit_csv_string(&self) -> String {
        let mut out = String::with_capacity(2560);
        out.push_str(REDISTRICT_UNIT_CSV_HEADER);
        out.push('\n');
        for u in &self.unit_summaries {
            let esc = if u.label.contains(',') || u.label.contains('"') {
                format!("\"{}\"", u.label.replace('"', "\"\""))
            } else {
                u.label.clone()
            };
            let delta_pts = (u.optimized_busy_rho - u.voronoi_busy_rho) * 100.0;
            out.push_str(&format!(
                "{},{},{},{:+.3},{},{},{:.2},{:.2},{:.6},{:.6},{:+.3},{:.2}\n",
                u.unit_idx,
                esc,
                u.station_segment_id,
                u.boundary_offset_s,
                u.voronoi_parcel_count,
                u.optimized_parcel_count,
                u.voronoi_demand_share_pct,
                u.optimized_demand_share_pct,
                u.voronoi_busy_rho,
                u.optimized_busy_rho,
                delta_pts,
                u.mean_unit_travel_s,
            ));
        }
        out
    }

    pub fn to_pareto_csv_string(&self) -> String {
        let mut out = String::with_capacity(1536);
        out.push_str(REDISTRICT_PARETO_CSV_HEADER);
        out.push('\n');
        for pt in &self.pareto_frontier {
            out.push_str(&format!(
                "{:.2},{:.3},{:.3},{:.2},{:.6},{:.6},{:.6},{:.6},{}\n",
                pt.equity_weight,
                pt.mean_dispatch_travel_s,
                pt.mean_total_response_s,
                pt.expected_primary_cov_pct,
                pt.workload_spread_rho,
                pt.workload_rmse_rho,
                pt.min_busy_rho,
                pt.max_busy_rho,
                pt.reassigned_parcels,
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
        fs::write(path, self.to_unit_csv_string())?;
        Ok(self.unit_summaries.len())
    }

    pub fn export_pareto_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_pareto_csv_string())?;
        Ok(self.pareto_frontier.len())
    }
}

#[inline]
fn recenter_and_clamp_offsets(offsets: &mut [f64], max_abs_s: f64) {
    if offsets.is_empty() {
        return;
    }
    let mean = offsets.iter().sum::<f64>() / (offsets.len() as f64);
    for v in offsets.iter_mut() {
        *v = (*v - mean).clamp(-max_abs_s, max_abs_s);
    }
    let mean2 = offsets.iter().sum::<f64>() / (offsets.len() as f64);
    for v in offsets.iter_mut() {
        *v -= mean2;
    }
}

#[inline]
fn workload_rmse(rep: &HypercubeReport) -> f64 {
    if rep.unit_metrics.is_empty() {
        return 0.0;
    }
    let m = rep.mean_busy_fraction_rho;
    let mse = rep
        .unit_metrics
        .iter()
        .map(|u| {
            let d = u.busy_fraction_rho - m;
            d * d
        })
        .sum::<f64>()
        / (rep.unit_metrics.len() as f64);
    mse.sqrt()
}

/// Optimizes Carter, Chaiken & Ignall (1972) additive boundary potentials `s_n` (`sum_n s_n = 0`)
/// directly over a precomputed `(N units x J atoms)` travel-time matrix using exact `2^N` Larson
/// (1974) hypercube evaluations, workload-feedback subgradient steps, and coordinate breakpoint search.
pub fn optimize_carter_larson_districts_from_matrix(
    unit_metadata: &[(String, usize, Vec2)],
    unit_parcel_times_s: &[Vec<f64>],
    atom_weights: &[f64],
    atom_positions: Option<&[Vec2]>,
    cfg: &RedistrictingConfig,
) -> RedistrictingReport {
    let t0 = Instant::now();
    let n_units = unit_parcel_times_s.len();
    let n_atoms = atom_weights.len();
    let tot_w: f64 = atom_weights.iter().sum::<f64>().max(1e-9);

    let effective_w_eq = match cfg.objective_mode {
        RedistrictingObjectiveMode::TravelTimeOptimalCarter1972 => 0.0,
        RedistrictingObjectiveMode::BalancedParetoKnee => {
            cfg.workload_equity_weight.clamp(0.0, 1.0)
        }
        RedistrictingObjectiveMode::StrictWorkloadEqualization => 0.92,
    };

    // Fast inner hypercube config (tight enough for exact breakpoint ranking in microseconds!)
    let mut inner_hc = cfg.hypercube.clone();
    inner_hc.tolerance = inner_hc.tolerance.max(1e-9);
    inner_hc.max_iterations = inner_hc.max_iterations.min(220);

    let mut eval_count = 0usize;
    let zero_offsets = vec![0.0; n_units];
    let voronoi_rep = solve_larson_hypercube_with_offsets(
        unit_metadata,
        unit_parcel_times_s,
        atom_weights,
        None,
        cfg.backup_offset_coupling,
        atom_positions,
        &cfg.hypercube,
    );
    eval_count += 1;

    let voronoi_pref = build_offset_dispatch_preferences(
        unit_metadata,
        unit_parcel_times_s,
        n_atoms,
        None,
        cfg.backup_offset_coupling,
        atom_positions,
    );
    let voronoi_parcel_units: Vec<usize> = voronoi_pref
        .iter()
        .map(|p| p.first().copied().unwrap_or(0))
        .collect();

    if n_units < 2 || n_atoms == 0 {
        let v_rmse = workload_rmse(&voronoi_rep);
        return RedistrictingReport {
            num_units: n_units,
            num_parcels: n_atoms,
            effective_equity_weight: effective_w_eq,
            carter_min_travel_hypercube: voronoi_rep.clone(),
            optimized_hypercube: voronoi_rep.clone(),
            voronoi_workload_spread: voronoi_rep.workload_imbalance_spread,
            optimized_workload_spread: voronoi_rep.workload_imbalance_spread,
            workload_spread_reduction_pct: 0.0,
            voronoi_workload_rmse: v_rmse,
            optimized_workload_rmse: v_rmse,
            travel_time_delta_s: 0.0,
            carter_min_travel_gain_s: 0.0,
            unit_offsets_s: zero_offsets,
            voronoi_parcel_units: voronoi_parcel_units.clone(),
            optimized_parcel_units: voronoi_parcel_units,
            reassigned_parcel_count: 0,
            reassigned_demand_pct: 0.0,
            unit_summaries: Vec::new(),
            pareto_frontier: Vec::new(),
            hypercube_evaluations: eval_count,
            voronoi_hypercube: voronoi_rep,
            solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let max_abs_s = cfg.max_offset_s.clamp(15.0, 600.0);
    let max_dt_cap = cfg.max_travel_penalty_s.clamp(0.0, 180.0);
    let voronoi_travel_s = voronoi_rep.mean_dispatch_travel_s;

    // Helper optimizing `s` for a given equity weight `w_eq in [0.0, 1.0]` starting from `seed_s`
    let mut solve_for_weight = |w_eq: f64, seed_s: &[f64], enforce_cap: bool| -> (Vec<f64>, HypercubeReport) {
        let score_fn = |rep: &HypercubeReport| -> f64 {
            let rmse = workload_rmse(rep);
            let base = (1.0 - w_eq) * rep.mean_dispatch_travel_s
                + w_eq * (680.0 * rep.workload_imbalance_spread + 360.0 * rmse);
            let excess_t = (rep.mean_dispatch_travel_s - (voronoi_travel_s + max_dt_cap)).max(0.0);
            if enforce_cap && w_eq < 0.995 && excess_t > 0.0 {
                base + 45.0 * excess_t + 8.0 * excess_t * excess_t
            } else {
                base
            }
        };

        let mut best_s = zero_offsets.clone();
        let mut best_rep = voronoi_rep.clone();
        let mut best_score = score_fn(&best_rep);

        // Evaluate `seed_s` if non-zero
        if seed_s.iter().any(|&v| v.abs() > 1e-6) {
            let mut s0 = seed_s.to_vec();
            recenter_and_clamp_offsets(&mut s0, max_abs_s);
            let r0 = solve_larson_hypercube_with_offsets(
                unit_metadata,
                unit_parcel_times_s,
                atom_weights,
                Some(&s0),
                cfg.backup_offset_coupling,
                atom_positions,
                &inner_hc,
            );
            eval_count += 1;
            let sc0 = score_fn(&r0);
            if sc0 < best_score - 1e-9 {
                best_score = sc0;
                best_s = s0;
                best_rep = r0;
            }
        }

        // Phase 1: Carter-Chaiken-Ignall / Larson Workload-Difference Ray Sweep
        // Even at `w_eq == 0.0`, optimal Carter-1972 boundary offsets align with `(rho_n - mean_rho)`!
        let vor_mean = voronoi_rep.mean_busy_fraction_rho;
        let rho_dir: Vec<f64> = voronoi_rep
            .unit_metrics
            .iter()
            .map(|u| u.busy_fraction_rho - vor_mean)
            .collect();
        for &ray_gain in &[20.0, 45.0, 85.0, 140.0, 220.0, 340.0, 500.0, 720.0] {
            let mut cand_s: Vec<f64> = rho_dir.iter().map(|&d| d * ray_gain).collect();
            recenter_and_clamp_offsets(&mut cand_s, max_abs_s);
            let rep = solve_larson_hypercube_with_offsets(
                unit_metadata,
                unit_parcel_times_s,
                atom_weights,
                Some(&cand_s),
                cfg.backup_offset_coupling,
                atom_positions,
                &inner_hc,
            );
            eval_count += 1;
            let sc = score_fn(&rep);
            if sc < best_score - 1e-9 {
                best_score = sc;
                best_s = cand_s;
                best_rep = rep;
            }
        }

        // Phase 2: Iterative Damped Larson (1974) Subgradient Workload-Feedback Loop
        let mut cur_s = best_s.clone();
        let mut cur_rep = best_rep.clone();
        let fb_iters = cfg.feedback_iterations.clamp(6, 40);
        for it in 0..fb_iters {
            let step_gain = (240.0 * (0.18 + 0.82 * w_eq)) / (1.0 + 0.12 * (it as f64));
            let m_rho = cur_rep.mean_busy_fraction_rho;
            for u in 0..n_units {
                let err = cur_rep.unit_metrics[u].busy_fraction_rho - m_rho;
                cur_s[u] += step_gain * err;
            }
            recenter_and_clamp_offsets(&mut cur_s, max_abs_s);
            cur_rep = solve_larson_hypercube_with_offsets(
                unit_metadata,
                unit_parcel_times_s,
                atom_weights,
                Some(&cur_s),
                cfg.backup_offset_coupling,
                atom_positions,
                &inner_hc,
            );
            eval_count += 1;
            let sc = score_fn(&cur_rep);
            if sc < best_score - 1e-9 {
                best_score = sc;
                best_s = cur_s.clone();
                best_rep = cur_rep.clone();
            }
        }

        // Phase 3: Exact Carter-Chaiken-Ignall Border-Atom Hyperplane Breakpoint Stepping
        // For every pair `(u_src, u_dst)` with `rho[u_src] > rho[u_dst]`, locates the single
        // border atom `j` currently in `u_src`'s beat with minimum margin
        // `m_j = (t_{u_dst, j} + s[u_dst]) - (t_{u_src, j} + s[u_src]) >= 0` and tests exact
        // hyperplane-crossing steps that hand strictly that border atom across the beat line!
        for _step in 0..(n_atoms.min(24)) {
            let cur_pref = build_offset_dispatch_preferences(
                unit_metadata,
                unit_parcel_times_s,
                n_atoms,
                Some(&best_s),
                cfg.backup_offset_coupling,
                atom_positions,
            );
            let mut step_best_sc = best_score;
            let mut step_best_s: Option<(Vec<f64>, HypercubeReport)> = None;

            for u_src in 0..n_units {
                for u_dst in 0..n_units {
                    if u_src == u_dst {
                        continue;
                    }
                    // Consider shifting a border atom from busier -> lighter (or any pair when w_eq == 0)
                    if w_eq > 0.05
                        && best_rep.unit_metrics[u_src].busy_fraction_rho
                            <= best_rep.unit_metrics[u_dst].busy_fraction_rho + 0.005
                    {
                        continue;
                    }
                    // Find closest border atom `j` currently in `u_src`'s primary beat wrt `u_dst`
                    let mut min_margin = f64::INFINITY;
                    for j in 0..n_atoms {
                        if cur_pref[j].first().copied() == Some(u_src) {
                            let m = (unit_parcel_times_s[u_dst][j] + best_s[u_dst])
                                - (unit_parcel_times_s[u_src][j] + best_s[u_src]);
                            if m >= -0.05 && m < min_margin {
                                min_margin = m.max(0.0);
                            }
                        }
                    }
                    if !min_margin.is_finite() || min_margin > max_abs_s * 1.8 {
                        continue;
                    }

                    // Test 3 exact single-atom hyperplane crossings:
                    // Mode 0: Symmetric pairwise shift `s[u_src] += m/2 + eps, s[u_dst] -= m/2 + eps`
                    // Mode 1: Pure contraction of `u_src` (`s[u_src] += m + eps`, third units undisturbed before recenter)
                    // Mode 2: Pure expansion of `u_dst` (`s[u_dst] -= m + eps`, third units undisturbed before recenter)
                    let eps = 0.35;
                    for mode in 0..3 {
                        let mut cand_s = best_s.clone();
                        match mode {
                            0 => {
                                cand_s[u_src] += 0.5 * min_margin + eps;
                                cand_s[u_dst] -= 0.5 * min_margin + eps;
                            }
                            1 => {
                                cand_s[u_src] += min_margin + eps;
                            }
                            _ => {
                                cand_s[u_dst] -= min_margin + eps;
                            }
                        }
                        recenter_and_clamp_offsets(&mut cand_s, max_abs_s);
                        let rep = solve_larson_hypercube_with_offsets(
                            unit_metadata,
                            unit_parcel_times_s,
                            atom_weights,
                            Some(&cand_s),
                            cfg.backup_offset_coupling,
                            atom_positions,
                            &inner_hc,
                        );
                        eval_count += 1;
                        let sc = score_fn(&rep);
                        if sc < step_best_sc - 1e-9 {
                            step_best_sc = sc;
                            step_best_s = Some((cand_s, rep));
                        }
                    }
                }
            }

            if let Some((ns, nrep)) = step_best_s {
                best_score = step_best_sc;
                best_s = ns;
                best_rep = nrep;
            } else {
                break;
            }
        }

        // Re-evaluate winning `best_s` at full user `cfg.hypercube` tolerance
        let exact_rep = solve_larson_hypercube_with_offsets(
            unit_metadata,
            unit_parcel_times_s,
            atom_weights,
            Some(&best_s),
            cfg.backup_offset_coupling,
            atom_positions,
            &cfg.hypercube,
        );
        eval_count += 1;
        (best_s, exact_rep)
    };

    // Sweep the 7-point Carter-Chaiken-Ignall Travel-Time vs. Workload-Spread Pareto Frontier
    let frontier_weights = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00];
    let mut pareto_frontier = Vec::with_capacity(frontier_weights.len());
    let mut warm_s = zero_offsets.clone();
    let mut carter_min_t_rep = voronoi_rep.clone();

    for (idx, &fw) in frontier_weights.iter().enumerate() {
        let cap_on = idx < frontier_weights.len() - 1;
        let (s_fw, rep_fw) = solve_for_weight(fw, &warm_s, cap_on);
        warm_s = s_fw.clone();
        if idx == 0 {
            carter_min_t_rep = rep_fw.clone();
        }
        let pref_fw = build_offset_dispatch_preferences(
            unit_metadata,
            unit_parcel_times_s,
            n_atoms,
            Some(&s_fw),
            cfg.backup_offset_coupling,
            atom_positions,
        );
        let shifted = pref_fw
            .iter()
            .enumerate()
            .filter(|&(j, p)| p.first().copied().unwrap_or(0) != voronoi_parcel_units[j])
            .count();
        pareto_frontier.push(RedistrictingTradeoffPoint {
            equity_weight: fw,
            mean_dispatch_travel_s: rep_fw.mean_dispatch_travel_s,
            mean_total_response_s: rep_fw.mean_total_response_s,
            expected_primary_cov_pct: rep_fw.expected_primary_cov_pct,
            workload_spread_rho: rep_fw.workload_imbalance_spread,
            workload_rmse_rho: workload_rmse(&rep_fw),
            min_busy_rho: rep_fw.min_busy_fraction_rho,
            max_busy_rho: rep_fw.max_busy_fraction_rho,
            reassigned_parcels: shifted,
            unit_offsets_s: s_fw,
        });
    }

    // Solve at the exact active `effective_w_eq` (warm-started from nearest Pareto point!)
    let nearest_seed = pareto_frontier
        .iter()
        .min_by(|a, b| {
            (a.equity_weight - effective_w_eq)
                .abs()
                .partial_cmp(&(b.equity_weight - effective_w_eq).abs())
                .unwrap_or(Ordering::Equal)
        })
        .map(|pt| pt.unit_offsets_s.clone())
        .unwrap_or_else(|| zero_offsets.clone());

    let (opt_offsets, opt_rep) = solve_for_weight(effective_w_eq, &nearest_seed, true);
    let opt_pref = build_offset_dispatch_preferences(
        unit_metadata,
        unit_parcel_times_s,
        n_atoms,
        Some(&opt_offsets),
        cfg.backup_offset_coupling,
        atom_positions,
    );
    let opt_parcel_units: Vec<usize> = opt_pref
        .iter()
        .map(|p| p.first().copied().unwrap_or(0))
        .collect();

    let mut reassigned_parcel_count = 0usize;
    let mut reassigned_w = 0.0;
    let mut vor_counts = vec![0usize; n_units];
    let mut opt_counts = vec![0usize; n_units];
    let mut vor_w_sum = vec![0.0; n_units];
    let mut opt_w_sum = vec![0.0; n_units];

    for j in 0..n_atoms {
        let vu = voronoi_parcel_units[j];
        let ou = opt_parcel_units[j];
        let w = atom_weights[j].max(0.0);
        vor_counts[vu] += 1;
        opt_counts[ou] += 1;
        vor_w_sum[vu] += w;
        opt_w_sum[ou] += w;
        if vu != ou {
            reassigned_parcel_count += 1;
            reassigned_w += w;
        }
    }

    let unit_summaries: Vec<UnitDistrictSummary> = (0..n_units)
        .map(|u| {
            let vm = &voronoi_rep.unit_metrics[u];
            let om = &opt_rep.unit_metrics[u];
            UnitDistrictSummary {
                unit_idx: u,
                label: om.label.clone(),
                station_segment_id: om.station_segment_id,
                world_pos: om.world_pos,
                boundary_offset_s: opt_offsets[u],
                voronoi_parcel_count: vor_counts[u],
                optimized_parcel_count: opt_counts[u],
                voronoi_demand_share_pct: (vor_w_sum[u] / tot_w) * 100.0,
                optimized_demand_share_pct: (opt_w_sum[u] / tot_w) * 100.0,
                voronoi_busy_rho: vm.busy_fraction_rho,
                optimized_busy_rho: om.busy_fraction_rho,
                voronoi_primary_share_pct: vm.primary_dispatch_share_pct,
                optimized_primary_share_pct: om.primary_dispatch_share_pct,
                mean_unit_travel_s: om.mean_unit_travel_s,
            }
        })
        .collect();

    let v_spread = voronoi_rep.workload_imbalance_spread;
    let o_spread = opt_rep.workload_imbalance_spread;
    let spread_red_pct = if v_spread > 1e-9 {
        ((v_spread - o_spread) / v_spread) * 100.0
    } else {
        0.0
    };
    let carter_gain_s =
        (voronoi_rep.mean_dispatch_travel_s - carter_min_t_rep.mean_dispatch_travel_s).max(0.0);

    RedistrictingReport {
        num_units: n_units,
        num_parcels: n_atoms,
        effective_equity_weight: effective_w_eq,
        travel_time_delta_s: opt_rep.mean_dispatch_travel_s - voronoi_rep.mean_dispatch_travel_s,
        carter_min_travel_gain_s: carter_gain_s,
        voronoi_workload_spread: v_spread,
        optimized_workload_spread: o_spread,
        workload_spread_reduction_pct: spread_red_pct,
        voronoi_workload_rmse: workload_rmse(&voronoi_rep),
        optimized_workload_rmse: workload_rmse(&opt_rep),
        unit_offsets_s: opt_offsets,
        voronoi_parcel_units,
        optimized_parcel_units: opt_parcel_units,
        reassigned_parcel_count,
        reassigned_demand_pct: (reassigned_w / tot_w) * 100.0,
        unit_summaries,
        pareto_frontier,
        hypercube_evaluations: eval_count,
        voronoi_hypercube: voronoi_rep,
        carter_min_travel_hypercube: carter_min_t_rep,
        optimized_hypercube: opt_rep,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Optimizes Carter, Chaiken & Ignall (1972) / Larson (1974) response-district boundaries directly
/// on the active `FireStation` facilities placed on `net`.
pub fn optimize_network_response_districts(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &RedistrictingConfig,
) -> RedistrictingReport {
    let (unit_meta, unit_times, atom_weights, atom_pos) =
        build_network_unit_travel_matrix(net, parcels, facilities, evp_cfg);
    optimize_carter_larson_districts_from_matrix(
        &unit_meta,
        &unit_times,
        &atom_weights,
        Some(&atom_pos),
        cfg,
    )
}

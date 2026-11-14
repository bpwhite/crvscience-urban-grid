//! Diurnal 24-Hour Non-Stationary Emergency Call Arrival (`Lambda(h)`), Spatial Land-Use Commuter
//! Shift (`w_j(h)`), Bureau of Public Roads (`BPR`, 1964) Rush-Hour Arterial Travel-Time Profiles,
//! Green & Kolesar (1991) Pointwise Stationary Approximation (`PSA`) 24-Hour Hypercube Trajectory,
//! and Separate **Peak vs. Off-Peak Carter-Chaiken-Ignall Response-District Dispatch Tables**.
//!
//! Grounded in:
//! - Green, L. & Kolesar, P. (1991), *The Pointwise Stationary Approximation for Queues with
//!   Nonstationary Arrivals*, Management Science 37(1):84-97; Whitt, W. (1991), Management Science
//!   37(3):307-314 (`https://pubsonline.informs.org/doi/10.1287/mnsc.37.3.307`).
//! - Bureau of Public Roads (1964), *Traffic Assignment Manual*, U.S. Dept. of Commerce (`t = t_0 *
//!   [1 + alpha * (V/C)^beta]`, `https://ascelibrary.org/doi/10.1061/JTEPBS.0000760`).
//! - U.S. Fire Administration / FEMA NFIRS 24-Hour Fire & EMS Run Profile
//!   (`https://www.usfa.fema.gov/downloads/pdf/statistics/v22i1-fire-department-run-profile.pdf`)
//!   and Cantwell et al. (2015) bimodal diurnal EMS demand (`https://pubmed.ncbi.nlm.nih.gov/25664379/`).
//! - Carter, G. M., Chaiken, J. M. & Ignall, E. (1972), *Response Areas for Two Emergency Units*,
//!   Operations Research 20(3):571-594 (`https://pubsonline.informs.org/doi/10.1287/opre.20.3.571`).
//! - Larson, R. C. (1974), *A hypercube queuing model for facility location and redistricting in
//!   urban emergency services*, Computers & Operations Research 1(1):67-95.

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::hypercube::{
    build_offset_dispatch_preferences, optimize_carter_larson_districts_from_matrix,
    solve_larson_hypercube_from_matrix, solve_larson_hypercube_with_pref, HypercubeConfig,
    RedistrictingConfig, RedistrictingReport,
};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::zoning::{Parcel, ZoneCode};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const DIURNAL_HOURLY_CSV_HEADER: &str = "hour_24,regime,calls_per_hr,arterial_vc_ratio,bpr_slowdown_pct,voronoi_mean_travel_s,static_opt_mean_travel_s,dual_table_mean_travel_s,dual_table_total_response_s,voronoi_spread_rho,static_opt_spread_rho,dual_table_spread_rho,dual_table_expected_cov_pct,prob_all_busy_or_queued";
pub const DIURNAL_DISPATCH_TABLE_CSV_HEADER: &str = "parcel_id,zone_code,offpeak_1st_due_unit,offpeak_2nd_due_unit,peak_1st_due_unit,peak_2nd_due_unit,primary_beat_switched,backup_beat_switched,offpeak_primary_travel_s,peak_primary_travel_s";

/// Empirical 24-hour urban Fire/EMS call-arrival multipliers (`h = 0..24`), normalized so
/// `sum_{h=0..23} USFA_DIURNAL_CALL_MULTIPLIERS[h] == 24.00` (exact unit daily mean `1.00`).
/// Captures the 03:00-04:00 overnight trough (`0.32x`) and 16:00-18:00 afternoon/evening peak (`1.54x`).
pub const USFA_DIURNAL_CALL_MULTIPLIERS: [f64; 24] = [
    0.48, 0.40, 0.35, 0.32, 0.34, 0.44, // 00:00 - 05:59 Overnight trough (sum = 2.33)
    0.66, 0.94, 1.18, 1.26, 1.32, 1.40, // 06:00 - 11:59 Morning commute & late-AM climb (sum = 6.76)
    1.39, 1.36, 1.38, 1.43, 1.51, 1.54, // 12:00 - 17:59 Midday plateau & 17:00 peak (sum = 8.61)
    1.48, 1.34, 1.14, 0.96, 0.78, 0.60, // 18:00 - 23:59 Evening taper (sum = 6.30 -> total = 24.00)
];

/// Baseline 24-hour arterial Volume-to-Capacity (`V/C`) commuter congestion profile (`h = 0..24`),
/// featuring bimodal AM Rush (`07:00-09:59`, peak `1.22` at `08:00`) and PM Rush (`15:00-18:59`,
/// peak `1.28` at `17:00`) separated by midday commercial traffic and free-flow overnight roads.
pub const BPR_HOURLY_ARTERIAL_VC: [f64; 24] = [
    0.18, 0.14, 0.12, 0.12, 0.16, 0.34, // 00:00 - 05:59 Free-flow overnight
    0.68, 1.06, 1.22, 0.96, 0.76, 0.78, // 06:00 - 11:59 AM Commute Rush (08:00 peak)
    0.82, 0.78, 0.84, 1.02, 1.20, 1.28, // 12:00 - 17:59 Midday & PM Commute Rush (17:00 peak)
    1.08, 0.82, 0.60, 0.46, 0.34, 0.24, // 18:00 - 23:59 Evening clearing
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiurnalWindowPreset {
    /// Bimodal AM + PM Rush-Hour Peak (`07:00-09:59` & `15:00-19:59`, 8 Peak hrs / 16 Off-Peak hrs).
    BimodalCommutePeak8Hr,
    /// Full Daytime Commercial/Commuter Shift (`07:00-18:59` Peak, `19:00-06:59` Night Off-Peak).
    DaytimeCommercial12Hr,
}

impl Default for DiurnalWindowPreset {
    fn default() -> Self {
        Self::BimodalCommutePeak8Hr
    }
}

impl DiurnalWindowPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::BimodalCommutePeak8Hr => "Bimodal Rush Peak (07-10 & 15-20)",
            Self::DaytimeCommercial12Hr => "Daytime Shift Peak (07:00-19:00)",
        }
    }

    #[inline]
    pub fn is_peak_hour(self, hour: u8) -> bool {
        let h = hour % 24;
        match self {
            Self::BimodalCommutePeak8Hr => (7..10).contains(&h) || (15..20).contains(&h),
            Self::DaytimeCommercial12Hr => (7..19).contains(&h),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiurnalDispatchRegime {
    PeakRush,
    OffPeakFreeFlow,
}

impl DiurnalDispatchRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::PeakRush => "Peak Rush Table",
            Self::OffPeakFreeFlow => "Off-Peak Free-Flow Table",
        }
    }

    pub fn short_tag(self) -> &'static str {
        match self {
            Self::PeakRush => "PEAK",
            Self::OffPeakFreeFlow => "OFF-PEAK",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiurnalProfileConfig {
    pub window_preset: DiurnalWindowPreset,
    /// 24-hour mean citywide emergency call arrival rate `Lambda_bar` (calls / hour).
    pub daily_mean_calls_per_hr: f64,
    /// Amplitude `[0.0, 1.35]` scaling the empirical USFA 24-hour call curve (`1.0` = standard USFA, `0.0` = flat).
    pub call_profile_amplitude: f64,
    /// Rush-hour arterial `V/C` multiplier (`1.0` = standard commuter peak, `1.35` = severe gridlock).
    pub rush_vc_scale: f64,
    /// Bureau of Public Roads (`BPR`) congestion coefficient `alpha` (`t = t_0 * [1 + alpha*(V/C)^beta]`).
    pub bpr_alpha: f64,
    /// Bureau of Public Roads (`BPR`) congestion exponent `beta` (classic `4.0`).
    pub bpr_beta: f64,
    /// Underlying Carter-Chaiken-Ignall / Larson hypercube redistricting configuration.
    pub redistrict: RedistrictingConfig,
}

impl Default for DiurnalProfileConfig {
    fn default() -> Self {
        Self {
            window_preset: DiurnalWindowPreset::BimodalCommutePeak8Hr,
            daily_mean_calls_per_hr: 3.6,
            call_profile_amplitude: 1.0,
            rush_vc_scale: 1.18,
            bpr_alpha: 0.24,
            bpr_beta: 4.0,
            redistrict: RedistrictingConfig::default(),
        }
    }
}

/// Returns exact normalized hourly call-rate multiplier `m(h)` at `hour in 0..24` such that
/// `mean_{h=0..23} m(h) == 1.0` for `amplitude in [0.0, 1.35]`.
#[inline]
pub fn diurnal_call_multiplier(hour: u8, amplitude: f64) -> f64 {
    let amp = amplitude.clamp(0.0, 1.35);
    let raw = USFA_DIURNAL_CALL_MULTIPLIERS[(hour as usize) % 24];
    (1.0 + amp * (raw - 1.0)).max(0.05)
}

/// Returns arterial baseline `V/C` ratio at `hour in 0..24` scaled by `rush_vc_scale`.
#[inline]
pub fn diurnal_arterial_vc_ratio(hour: u8, rush_vc_scale: f64) -> f64 {
    BPR_HOURLY_ARTERIAL_VC[(hour as usize) % 24] * rush_vc_scale.clamp(0.0, 2.5)
}

/// Bureau of Public Roads (`BPR`, 1964) link travel-time multiplier `1 + alpha * (V/C)^beta`.
#[inline]
pub fn bpr_travel_time_multiplier(vc_ratio: f64, alpha: f64, beta: f64) -> f64 {
    let vc = vc_ratio.max(0.0);
    1.0 + alpha.max(0.0) * vc.powf(beta.clamp(1.0, 8.0))
}

/// Computes hour-specific spatial emergency demand weights `w_j(h)` reflecting daytime population
/// inflows to `CBD`, `IND`, and `MU4` job centers versus evening/overnight returns to `R1`/`R3` homes.
pub fn compute_hourly_parcel_weights(parcels: &[Parcel], hour: u8) -> Vec<f64> {
    let h = hour % 24;
    let (home_mult, job_mult, park_bonus) = if (8..18).contains(&h) {
        // Daytime work/commercial surge in CBD, MU-4 & Industrial/Tech corridors
        (0.54, 1.42, 6.0)
    } else if h == 7 || (18..20).contains(&h) {
        // Commute shoulder transition
        (0.88, 0.82, 4.0)
    } else {
        // Evening & overnight residential concentration in R1 / R3 / MU-4 housing
        (1.34, 0.10, 0.5)
    };

    parcels
        .iter()
        .map(|p| {
            let park_term = if p.zone == ZoneCode::PARK {
                park_bonus * p.area_ha
            } else {
                0.0
            };
            (home_mult * p.residents() + job_mult * 0.68 * p.jobs() + park_term).max(0.5)
        })
        .collect()
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

/// Computes `(unit_metadata, unit_parcel_times_s[u][j], mean_bpr_slowdown_pct)` at a specific
/// clock `hour in 0..24` using directional AM/PM tidal commuter `V/C` loading and BPR link delays.
pub fn build_hourly_unit_travel_matrix(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    hour: u8,
    cfg: &DiurnalProfileConfig,
) -> (Vec<(String, usize, Vec2)>, Vec<Vec<f64>>, f64) {
    let n_segs = net.segments.len();
    let h = hour % 24;
    let base_vc = diurnal_arterial_vc_ratio(h, cfg.rush_vc_scale);
    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    // Even with siren EVP green-wave preemption, dense arterial platoon queues create residual impedance
    let eff_alpha = if evp_cfg.enabled {
        cfg.bpr_alpha * 0.72
    } else {
        cfg.bpr_alpha
    };

    let city_center = Vec2 {
        x: 0.5 * (net.bounds.min_x + net.bounds.max_x),
        y: 0.5 * (net.bounds.min_y + net.bounds.max_y),
    };

    let mut mult_sum = 0.0;
    let mut mult_cnt = 0usize;

    let seg_trav_s: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                return f64::INFINITY;
            }
            let free_v = (s.speed_mps * speed_boost).clamp(4.0, 30.0);
            let free_t = s.length / free_v;

            // Arterial / collector hierarchy factor: major multi-lane arterials & bridges absorb commuter rush
            let is_arterial = s.lane_count >= 2 || s.priority >= 2 || s.speed_mps >= 13.5;
            let hier_factor = if is_arterial { 1.18 } else { 0.56 };

            // Tidal AM inbound (`07..10`) vs. PM outbound (`15..19`) directional commuter asymmetry
            let p_start = net.junctions[s.from_junc].pos;
            let p_end = net.junctions[s.to_junc].pos;
            let d0 = p_start.dist(city_center);
            let d1 = p_end.dist(city_center);
            let inbound = d1 + 2.0 < d0;
            let tidal_factor = if (7..10).contains(&h) {
                if inbound { 1.15 } else { 0.88 }
            } else if (15..19).contains(&h) {
                if !inbound { 1.16 } else { 0.90 }
            } else {
                1.0
            };

            let vc_seg = base_vc * hier_factor * tidal_factor;
            let bpr_mult = bpr_travel_time_multiplier(vc_seg, eff_alpha, cfg.bpr_beta);
            if is_arterial {
                mult_sum += bpr_mult;
                mult_cnt += 1;
            }
            let stall_pen = if s.stalled_lane.is_some() && s.lane_count == 1 {
                28.0
            } else {
                0.0
            };
            free_t * bpr_mult + stall_pen
        })
        .collect();

    let mean_slowdown_pct = if mult_cnt > 0 {
        ((mult_sum / mult_cnt as f64) - 1.0).max(0.0) * 100.0
    } else {
        (bpr_travel_time_multiplier(base_vc, eff_alpha, cfg.bpr_beta) - 1.0).max(0.0) * 100.0
    };

    // Intersection queue clearance delay also rises with `base_vc` during rush hours
    let rush_sig_scale = 1.0 + 0.35 * base_vc.powi(2);
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
                    2.2 * rush_sig_scale
                } else {
                    14.5 * rush_sig_scale
                }
            }
            ControlMode::Roundabout => 3.4 * (1.0 + 0.18 * base_vc),
            ControlMode::PermissiveYield => 1.8 * (1.0 + 0.15 * base_vc),
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

    (unit_meta, unit_times, mean_slowdown_pct)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelDispatchRow {
    pub parcel_id: usize,
    pub zone_label: String,
    pub offpeak_first_due_unit: usize,
    pub offpeak_second_due_unit: usize,
    pub peak_first_due_unit: usize,
    pub peak_second_due_unit: usize,
    pub primary_beat_switched: bool,
    pub backup_beat_switched: bool,
    pub offpeak_primary_travel_s: f64,
    pub peak_primary_travel_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiurnalDispatchTable {
    pub regime: DiurnalDispatchRegime,
    pub active_hours: Vec<u8>,
    pub mean_calls_per_hr: f64,
    pub mean_arterial_vc: f64,
    pub mean_bpr_slowdown_pct: f64,
    pub unit_offsets_s: Vec<f64>,
    pub parcel_primary_units: Vec<usize>,
    pub parcel_second_due_units: Vec<usize>,
    pub parcel_preference_lists: Vec<Vec<usize>>,
    pub redistricting_report: RedistrictingReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyDiurnalMetrics {
    pub hour: u8,
    pub is_peak_regime: bool,
    pub calls_per_hr: f64,
    pub arterial_vc_ratio: f64,
    pub bpr_slowdown_pct: f64,
    pub voronoi_mean_travel_s: f64,
    pub static_opt_mean_travel_s: f64,
    pub dual_table_mean_travel_s: f64,
    pub dual_table_total_response_s: f64,
    pub voronoi_workload_spread: f64,
    pub static_opt_workload_spread: f64,
    pub dual_table_workload_spread: f64,
    pub dual_table_expected_cov_pct: f64,
    pub prob_all_busy_or_queued: f64,
    pub rate_conservation_error: f64,
    pub unit_busy_rhos: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiurnalRedistrictingReport {
    pub num_units: usize,
    pub num_parcels: usize,
    pub window_preset: DiurnalWindowPreset,
    pub peak_table: DiurnalDispatchTable,
    pub offpeak_table: DiurnalDispatchTable,
    pub parcel_run_cards: Vec<ParcelDispatchRow>,
    pub hourly_trajectory: Vec<HourlyDiurnalMetrics>,
    /// Number of parcels whose 1st-due primary unit switches between Off-Peak and Peak tables.
    pub primary_beat_switches_count: usize,
    /// Number of parcels whose 1st-due OR 2nd-due run-card assignment switches between Off-Peak and Peak.
    pub any_rank12_switches_count: usize,
    pub switched_demand_share_pct: f64,
    pub daily_voronoi_mean_travel_s: f64,
    pub daily_static_opt_mean_travel_s: f64,
    pub daily_dual_table_mean_travel_s: f64,
    pub daily_voronoi_workload_spread: f64,
    pub daily_static_opt_workload_spread: f64,
    pub daily_dual_table_workload_spread: f64,
    pub peak_window_spread_reduction_pct: f64,
    pub solve_time_ms: f64,
}

impl DiurnalRedistrictingReport {
    pub fn to_hourly_csv_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str(DIURNAL_HOURLY_CSV_HEADER);
        out.push('\n');
        for hm in &self.hourly_trajectory {
            let reg = if hm.is_peak_regime { "PEAK" } else { "OFF_PEAK" };
            out.push_str(&format!(
                "{:02}:00,{},{:.3},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.6},{:.6},{:.6},{:.2},{:.6}\n",
                hm.hour,
                reg,
                hm.calls_per_hr,
                hm.arterial_vc_ratio,
                hm.bpr_slowdown_pct,
                hm.voronoi_mean_travel_s,
                hm.static_opt_mean_travel_s,
                hm.dual_table_mean_travel_s,
                hm.dual_table_total_response_s,
                hm.voronoi_workload_spread,
                hm.static_opt_workload_spread,
                hm.dual_table_workload_spread,
                hm.dual_table_expected_cov_pct,
                hm.prob_all_busy_or_queued,
            ));
        }
        out
    }

    pub fn to_dispatch_table_csv_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str(DIURNAL_DISPATCH_TABLE_CSV_HEADER);
        out.push('\n');
        for rc in &self.parcel_run_cards {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.2},{:.2}\n",
                rc.parcel_id,
                rc.zone_label,
                rc.offpeak_first_due_unit,
                rc.offpeak_second_due_unit,
                rc.peak_first_due_unit,
                rc.peak_second_due_unit,
                rc.primary_beat_switched,
                rc.backup_beat_switched,
                rc.offpeak_primary_travel_s,
                rc.peak_primary_travel_s,
            ));
        }
        out
    }

    pub fn export_hourly_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_hourly_csv_string())?;
        Ok(self.hourly_trajectory.len())
    }

    pub fn export_dispatch_table_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_dispatch_table_csv_string())?;
        Ok(self.parcel_run_cards.len())
    }
}

/// Solves the complete 24-hour diurnal emergency profile, constructs separate **Peak Rush** and
/// **Off-Peak Free-Flow** Carter-Chaiken-Ignall / Larson response-district dispatch tables, and
/// evaluates the 24-hour Green & Kolesar (1991) Pointwise Stationary Approximation (`PSA`)
/// hypercube trajectory comparing Voronoi vs. Single All-Day Table vs. Dual Peak/Off-Peak Tables!
pub fn solve_diurnal_peak_offpeak_dispatch_tables(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &DiurnalProfileConfig,
) -> DiurnalRedistrictingReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let atom_pos: Vec<Vec2> = parcels.iter().map(|p| p.center).collect();

    // 1. Precompute hourly call rates `lambda(h)`, spatial parcel demand weights `w_j(h)`,
    //    and BPR rush-hour travel-time matrices `T_h[u][j]` across all 24 hours `0..24`.
    let mut unit_meta: Vec<(String, usize, Vec2)> = Vec::new();
    let mut hourly_times: Vec<Vec<Vec<f64>>> = Vec::with_capacity(24);
    let mut hourly_weights: Vec<Vec<f64>> = Vec::with_capacity(24);
    let mut hourly_calls: Vec<f64> = Vec::with_capacity(24);
    let mut hourly_vc: Vec<f64> = Vec::with_capacity(24);
    let mut hourly_slowdown: Vec<f64> = Vec::with_capacity(24);

    for h in 0..24u8 {
        let lam_h = (cfg.daily_mean_calls_per_hr.max(0.05)
            * diurnal_call_multiplier(h, cfg.call_profile_amplitude))
        .max(0.02);
        let vc_h = diurnal_arterial_vc_ratio(h, cfg.rush_vc_scale);
        let w_h = compute_hourly_parcel_weights(parcels, h);
        let (meta_h, times_h, slow_h) =
            build_hourly_unit_travel_matrix(net, parcels, facilities, evp_cfg, h, cfg);
        if h == 0 {
            unit_meta = meta_h;
        }
        hourly_calls.push(lam_h);
        hourly_vc.push(vc_h);
        hourly_weights.push(w_h);
        hourly_times.push(times_h);
        hourly_slowdown.push(slow_h);
    }

    let n_units = unit_meta.len();
    let mut peak_hours = Vec::new();
    let mut offpeak_hours = Vec::new();
    for h in 0..24u8 {
        if cfg.window_preset.is_peak_hour(h) {
            peak_hours.push(h);
        } else {
            offpeak_hours.push(h);
        }
    }

    // Helper aggregating call-weighted travel matrix `T_reg[u][j]` & spatial demand `w_reg[j]` over a set of hours
    let aggregate_regime = |hrs: &[u8]| -> (f64, f64, f64, Vec<Vec<f64>>, Vec<f64>) {
        let mut tot_calls = 0.0;
        let mut vc_sum = 0.0;
        let mut slow_sum = 0.0;
        let mut agg_t = vec![vec![0.0; n_parcels]; n_units];
        let mut agg_w = vec![0.0; n_parcels];

        for &h in hrs {
            let hi = h as usize;
            let lam = hourly_calls[hi];
            tot_calls += lam;
            vc_sum += hourly_vc[hi];
            slow_sum += hourly_slowdown[hi];

            let w_tot_h: f64 = hourly_weights[hi].iter().sum::<f64>().max(1e-9);
            for j in 0..n_parcels {
                let call_j = lam * (hourly_weights[hi][j] / w_tot_h);
                agg_w[j] += call_j;
            }
            for u in 0..n_units {
                for j in 0..n_parcels {
                    agg_t[u][j] += lam * hourly_times[hi][u][j];
                }
            }
        }
        let denom = tot_calls.max(1e-9);
        for u in 0..n_units {
            for j in 0..n_parcels {
                agg_t[u][j] /= denom;
            }
        }
        let n_h = (hrs.len().max(1)) as f64;
        (
            tot_calls / n_h,
            vc_sum / n_h,
            slow_sum / n_h,
            agg_t,
            agg_w,
        )
    };

    // 2. Optimize Separate Carter-Chaiken-Ignall / Larson Dispatch Tables for:
    //    (a) Peak Rush Window (`peak_hours`)
    //    (b) Off-Peak Free-Flow Window (`offpeak_hours`)
    //    (c) Single Compromise All-Day Baseline (`0..24`)
    let (peak_lam, peak_vc, peak_slow, peak_times, peak_weights) = aggregate_regime(&peak_hours);
    let (off_lam, off_vc, off_slow, off_times, off_weights) = aggregate_regime(&offpeak_hours);
    let all_hours: Vec<u8> = (0..24).collect();
    let (all_lam, _, _, all_times, all_weights) = aggregate_regime(&all_hours);

    let mut peak_rcfg = cfg.redistrict.clone();
    peak_rcfg.hypercube.system_calls_per_hr = peak_lam;
    let peak_rep = optimize_carter_larson_districts_from_matrix(
        &unit_meta,
        &peak_times,
        &peak_weights,
        Some(&atom_pos),
        &peak_rcfg,
    );

    let mut off_rcfg = cfg.redistrict.clone();
    off_rcfg.hypercube.system_calls_per_hr = off_lam;
    let off_rep = optimize_carter_larson_districts_from_matrix(
        &unit_meta,
        &off_times,
        &off_weights,
        Some(&atom_pos),
        &off_rcfg,
    );

    let mut all_rcfg = cfg.redistrict.clone();
    all_rcfg.hypercube.system_calls_per_hr = all_lam;
    let all_rep = optimize_carter_larson_districts_from_matrix(
        &unit_meta,
        &all_times,
        &all_weights,
        Some(&atom_pos),
        &all_rcfg,
    );

    let peak_pref = build_offset_dispatch_preferences(
        &unit_meta,
        &peak_times,
        n_parcels,
        Some(&peak_rep.unit_offsets_s),
        cfg.redistrict.backup_offset_coupling,
        Some(&atom_pos),
    );
    let off_pref = build_offset_dispatch_preferences(
        &unit_meta,
        &off_times,
        n_parcels,
        Some(&off_rep.unit_offsets_s),
        cfg.redistrict.backup_offset_coupling,
        Some(&atom_pos),
    );
    let static_all_pref = build_offset_dispatch_preferences(
        &unit_meta,
        &all_times,
        n_parcels,
        Some(&all_rep.unit_offsets_s),
        cfg.redistrict.backup_offset_coupling,
        Some(&atom_pos),
    );

    let extract_ranks = |pref: &[Vec<usize>]| -> (Vec<usize>, Vec<usize>) {
        let pri = pref
            .iter()
            .map(|r| r.first().copied().unwrap_or(0))
            .collect();
        let sec = pref
            .iter()
            .map(|r| r.get(1).copied().or_else(|| r.first().copied()).unwrap_or(0))
            .collect();
        (pri, sec)
    };
    let (peak_pri, peak_sec) = extract_ranks(&peak_pref);
    let (off_pri, off_sec) = extract_ranks(&off_pref);

    // 3. Build Parcel CAD Run-Card Comparison Table (`Off-Peak` vs. `Peak`)
    let mut parcel_run_cards = Vec::with_capacity(n_parcels);
    let mut primary_beat_switches_count = 0usize;
    let mut any_rank12_switches_count = 0usize;
    let mut switched_w = 0.0;
    let tot_all_w: f64 = all_weights.iter().sum::<f64>().max(1e-9);

    for (j, p) in parcels.iter().enumerate() {
        let op1 = off_pri.get(j).copied().unwrap_or(0);
        let op2 = off_sec.get(j).copied().unwrap_or(0);
        let pk1 = peak_pri.get(j).copied().unwrap_or(0);
        let pk2 = peak_sec.get(j).copied().unwrap_or(0);
        let pri_sw = op1 != pk1;
        let bkp_sw = op2 != pk2;
        if pri_sw {
            primary_beat_switches_count += 1;
            switched_w += all_weights[j];
        }
        if pri_sw || bkp_sw {
            any_rank12_switches_count += 1;
        }
        let op_t = if n_units > 0 {
            off_times[op1.min(n_units - 1)][j]
        } else {
            999.0
        };
        let pk_t = if n_units > 0 {
            peak_times[pk1.min(n_units - 1)][j]
        } else {
            999.0
        };
        parcel_run_cards.push(ParcelDispatchRow {
            parcel_id: p.id,
            zone_label: format!("{:?}", p.zone),
            offpeak_first_due_unit: op1,
            offpeak_second_due_unit: op2,
            peak_first_due_unit: pk1,
            peak_second_due_unit: pk2,
            primary_beat_switched: pri_sw,
            backup_beat_switched: bkp_sw,
            offpeak_primary_travel_s: op_t,
            peak_primary_travel_s: pk_t,
        });
    }

    // 4. Evaluate 24-Hour Green & Kolesar (1991) Pointwise Stationary Approximation (`PSA`)
    //    across all 24 hours comparing Voronoi vs. Static All-Day Table vs. Dual Peak/Off-Peak Tables!
    let mut hourly_trajectory = Vec::with_capacity(24);
    let mut tot_daily_calls = 0.0;
    let mut acc_vor_t = 0.0;
    let mut acc_sta_t = 0.0;
    let mut acc_dual_t = 0.0;
    let mut acc_vor_sp = 0.0;
    let mut acc_sta_sp = 0.0;
    let mut acc_dual_sp = 0.0;
    let mut peak_vor_sp_sum = 0.0;
    let mut peak_dual_sp_sum = 0.0;
    let mut peak_call_sum = 0.0;

    for h in 0..24u8 {
        let hi = h as usize;
        let is_pk = cfg.window_preset.is_peak_hour(h);
        let lam_h = hourly_calls[hi];
        let mut h_cfg: HypercubeConfig = cfg.redistrict.hypercube.clone();
        h_cfg.system_calls_per_hr = lam_h;

        let vor_h = solve_larson_hypercube_from_matrix(
            &unit_meta,
            &hourly_times[hi],
            &hourly_weights[hi],
            &h_cfg,
        );
        let sta_h = solve_larson_hypercube_with_pref(
            &unit_meta,
            &hourly_times[hi],
            &hourly_weights[hi],
            &static_all_pref,
            &h_cfg,
        );
        let active_pref = if is_pk { &peak_pref } else { &off_pref };
        let dual_h = solve_larson_hypercube_with_pref(
            &unit_meta,
            &hourly_times[hi],
            &hourly_weights[hi],
            active_pref,
            &h_cfg,
        );

        tot_daily_calls += lam_h;
        acc_vor_t += lam_h * vor_h.mean_dispatch_travel_s;
        acc_sta_t += lam_h * sta_h.mean_dispatch_travel_s;
        acc_dual_t += lam_h * dual_h.mean_dispatch_travel_s;
        acc_vor_sp += lam_h * vor_h.workload_imbalance_spread;
        acc_sta_sp += lam_h * sta_h.workload_imbalance_spread;
        acc_dual_sp += lam_h * dual_h.workload_imbalance_spread;

        if is_pk {
            peak_call_sum += lam_h;
            peak_vor_sp_sum += lam_h * vor_h.workload_imbalance_spread;
            peak_dual_sp_sum += lam_h * dual_h.workload_imbalance_spread;
        }

        hourly_trajectory.push(HourlyDiurnalMetrics {
            hour: h,
            is_peak_regime: is_pk,
            calls_per_hr: lam_h,
            arterial_vc_ratio: hourly_vc[hi],
            bpr_slowdown_pct: hourly_slowdown[hi],
            voronoi_mean_travel_s: vor_h.mean_dispatch_travel_s,
            static_opt_mean_travel_s: sta_h.mean_dispatch_travel_s,
            dual_table_mean_travel_s: dual_h.mean_dispatch_travel_s,
            dual_table_total_response_s: dual_h.mean_total_response_s,
            voronoi_workload_spread: vor_h.workload_imbalance_spread,
            static_opt_workload_spread: sta_h.workload_imbalance_spread,
            dual_table_workload_spread: dual_h.workload_imbalance_spread,
            dual_table_expected_cov_pct: dual_h.expected_primary_cov_pct,
            prob_all_busy_or_queued: dual_h.prob_all_busy_or_queued,
            rate_conservation_error: dual_h.rate_conservation_error,
            unit_busy_rhos: dual_h
                .unit_metrics
                .iter()
                .map(|u| u.busy_fraction_rho)
                .collect(),
        });
    }

    let c_den = tot_daily_calls.max(1e-9);
    let peak_sp_red_pct = if peak_vor_sp_sum > 1e-9 {
        ((peak_vor_sp_sum - peak_dual_sp_sum) / peak_vor_sp_sum) * 100.0
    } else {
        0.0
    };
    let _ = peak_call_sum;

    let peak_table = DiurnalDispatchTable {
        regime: DiurnalDispatchRegime::PeakRush,
        active_hours: peak_hours,
        mean_calls_per_hr: peak_lam,
        mean_arterial_vc: peak_vc,
        mean_bpr_slowdown_pct: peak_slow,
        unit_offsets_s: peak_rep.unit_offsets_s.clone(),
        parcel_primary_units: peak_pri,
        parcel_second_due_units: peak_sec,
        parcel_preference_lists: peak_pref,
        redistricting_report: peak_rep,
    };

    let offpeak_table = DiurnalDispatchTable {
        regime: DiurnalDispatchRegime::OffPeakFreeFlow,
        active_hours: offpeak_hours,
        mean_calls_per_hr: off_lam,
        mean_arterial_vc: off_vc,
        mean_bpr_slowdown_pct: off_slow,
        unit_offsets_s: off_rep.unit_offsets_s.clone(),
        parcel_primary_units: off_pri,
        parcel_second_due_units: off_sec,
        parcel_preference_lists: off_pref,
        redistricting_report: off_rep,
    };

    DiurnalRedistrictingReport {
        num_units: n_units,
        num_parcels: n_parcels,
        window_preset: cfg.window_preset,
        peak_table,
        offpeak_table,
        parcel_run_cards,
        hourly_trajectory,
        primary_beat_switches_count,
        any_rank12_switches_count,
        switched_demand_share_pct: (switched_w / tot_all_w) * 100.0,
        daily_voronoi_mean_travel_s: acc_vor_t / c_den,
        daily_static_opt_mean_travel_s: acc_sta_t / c_den,
        daily_dual_table_mean_travel_s: acc_dual_t / c_den,
        daily_voronoi_workload_spread: acc_vor_sp / c_den,
        daily_static_opt_workload_spread: acc_sta_sp / c_den,
        daily_dual_table_workload_spread: acc_dual_sp / c_den,
        peak_window_spread_reduction_pct: peak_sp_red_pct,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

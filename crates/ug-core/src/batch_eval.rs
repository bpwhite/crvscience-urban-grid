//! Headless `Rayon`-parallelized multi-regime experiment engine comparing **Fixed-Time**,
//! **Varaiya (2013) Max-Pressure Adaptive**, and **FHWA Modern Roundabout** scenarios across
//! configurable demand multipliers and Common Random Numbers (`CRN`) Monte Carlo replicates,
//! producing an RFC 4180 comparative `.csv` scorecard with baseline deltas and regime roll-ups.

use crate::network::Vec2;
use crate::network_edit::convert_junction_to_roundabout;
use crate::persistence::PersistenceError;
use crate::traffic::SimulationWorld;
use crate::zoning::apply_tod_upzoning_policy;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BatchControlRegime {
    FixedTime,
    MaxPressureAdaptive,
    ModernRoundabout,
}

impl BatchControlRegime {
    pub const ALL: [Self; 3] = [
        Self::FixedTime,
        Self::MaxPressureAdaptive,
        Self::ModernRoundabout,
    ];

    pub fn slug(self) -> &'static str {
        match self {
            Self::FixedTime => "fixed_time",
            Self::MaxPressureAdaptive => "max_pressure",
            Self::ModernRoundabout => "modern_roundabout",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::FixedTime => "Fixed-Time 2-Phase",
            Self::MaxPressureAdaptive => "Max-Pressure Adaptive",
            Self::ModernRoundabout => "Modern Roundabout",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExperimentConfig {
    pub duration_s: f64,
    pub warmup_s: f64,
    pub dt_s: f64,
    pub demand_multipliers: Vec<f64>,
    pub replicates: usize,
    pub roundabout_radius_m: f64,
    pub roundabout_lanes: usize,
    pub max_roundabouts: usize,
    pub include_tod_upzone: bool,
    pub base_seed: u64,
}

impl Default for BatchExperimentConfig {
    fn default() -> Self {
        Self {
            duration_s: 180.0,
            warmup_s: 30.0,
            dt_s: 0.1,
            demand_multipliers: vec![0.75, 1.00, 1.35],
            replicates: 1,
            roundabout_radius_m: 18.0,
            roundabout_lanes: 2,
            max_roundabouts: 6,
            include_tod_upzone: false,
            base_seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchScorecardRow {
    pub row_type: String,
    pub demand_label: String,
    pub demand_multiplier: f64,
    pub regime: BatchControlRegime,
    pub regime_title: String,
    pub replicates: usize,
    pub mean_speed_mph: f64,
    pub speed_std_mph: f64,
    pub avg_queued_vehicles: f64,
    pub throughput_veh_per_hr: f64,
    pub avg_trip_time_s: f64,
    pub hcm_los: char,
    pub mp_controlled_junctions: usize,
    pub mp_phase_switches: u64,
    pub roundabouts_converted: usize,
    pub gtfs_boardings: u64,
    pub gtfs_mean_dwell_s: f64,
    pub gtfs_load_factor_pct: f64,
    pub vsp_co2_kg_per_hr: f64,
    pub vsp_pm25_g_per_hr: f64,
    pub corridor_noise_dba: f64,
    pub access_15min_pct: f64,
    pub net_fiscal_per_acre_k_yr: f64,
    pub speed_delta_vs_fixed_pct: f64,
    pub queue_delta_vs_fixed_pct: f64,
    pub trip_time_delta_vs_fixed_pct: f64,
    pub throughput_delta_vs_fixed_pct: f64,
    pub co2_delta_vs_fixed_pct: f64,
}

pub const BATCH_SCORECARD_CSV_HEADER: &str = "row_type,demand_scale,demand_multiplier,regime_slug,regime_title,replicates,mean_speed_mph,speed_std_mph,avg_queued_vehicles,throughput_veh_per_hr,avg_trip_time_s,hcm_los,mp_controlled_junctions,mp_phase_switches,roundabouts_converted,gtfs_boardings,gtfs_mean_dwell_s,gtfs_load_factor_pct,vsp_co2_kg_per_hr,vsp_pm25_g_per_hr,corridor_noise_dba,access_15min_pct,net_fiscal_per_acre_k_yr,speed_delta_vs_fixed_pct,queue_delta_vs_fixed_pct,trip_time_delta_vs_fixed_pct,throughput_delta_vs_fixed_pct,co2_delta_vs_fixed_pct";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExperimentReport {
    pub config: BatchExperimentConfig,
    pub total_parallel_runs: usize,
    pub total_simulated_seconds: f64,
    pub wall_clock_ms: f64,
    pub realtime_speedup_x: f64,
    pub scenario_rows: Vec<BatchScorecardRow>,
    pub regime_summary_rows: Vec<BatchScorecardRow>,
}

/// Lock-free atomic progress & cooperative cancellation tracker shared between a background
/// `Rayon` worker pool and the 120 FPS `egui` UI thread (`Ordering::Relaxed` — zero mutexes).
#[derive(Debug, Default)]
pub struct BatchProgressTracker {
    pub total_runs: AtomicUsize,
    pub completed_runs: AtomicUsize,
    pub total_steps: AtomicUsize,
    pub completed_steps: AtomicUsize,
    pub cancel_requested: AtomicBool,
}

impl BatchProgressTracker {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Relaxed);
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Relaxed)
    }

    /// Returns `(completed_runs, total_runs, smooth_step_fraction_0_to_1)`.
    pub fn snapshot_progress(&self) -> (usize, usize, f32) {
        let done_runs = self.completed_runs.load(Ordering::Relaxed);
        let tot_runs = self.total_runs.load(Ordering::Relaxed);
        let done_steps = self.completed_steps.load(Ordering::Relaxed);
        let tot_steps = self.total_steps.load(Ordering::Relaxed);
        let frac = if tot_steps > 0 {
            (done_steps as f32 / tot_steps as f32).clamp(0.0, 1.0)
        } else if tot_runs > 0 {
            (done_runs as f32 / tot_runs as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (done_runs, tot_runs, frac)
    }
}

#[derive(Clone, Copy)]
struct SingleRunJob {
    demand_idx: usize,
    demand_mult: f64,
    replicate_idx: usize,
    regime: BatchControlRegime,
}

#[derive(Clone)]
struct SingleRunMetrics {
    demand_idx: usize,
    demand_mult: f64,
    regime: BatchControlRegime,
    mean_speed_mph: f64,
    avg_queued_vehicles: f64,
    throughput_veh_per_hr: f64,
    avg_trip_time_s: f64,
    mp_controlled_junctions: usize,
    mp_phase_switches: u64,
    roundabouts_converted: usize,
    gtfs_boardings: u64,
    gtfs_mean_dwell_s: f64,
    gtfs_load_factor_pct: f64,
    vsp_co2_kg_per_hr: f64,
    vsp_pm25_g_per_hr: f64,
    corridor_noise_dba: f64,
    access_15min_pct: f64,
    net_fiscal_per_acre_k_yr: f64,
}

fn hcm_los_from_speed(mean_speed_mph: f64) -> char {
    if mean_speed_mph >= 21.0 {
        'A'
    } else if mean_speed_mph >= 17.0 {
        'B'
    } else if mean_speed_mph >= 13.0 {
        'C'
    } else if mean_speed_mph >= 9.5 {
        'D'
    } else if mean_speed_mph >= 6.5 {
        'E'
    } else {
        'F'
    }
}

fn pct_delta(val: f64, base: f64) -> f64 {
    if base.abs() < 1e-9 {
        0.0
    } else {
        ((val - base) / base) * 100.0
    }
}

/// Converts up to `max_roundabouts` non-overlapping 3+ arm intersections (prioritized by arm count
/// and incident corridor hierarchy) into multi-lane curved FHWA modern roundabouts.
pub fn convert_network_candidate_roundabouts(
    world: &mut SimulationWorld,
    radius_m: f64,
    ring_lanes: usize,
    max_roundabouts: usize,
) -> usize {
    if max_roundabouts == 0 {
        return 0;
    }
    let saved_history = world.edit_history.clone();
    let min_spacing_m = (radius_m * 3.1).max(48.0);
    let mut candidates: Vec<(usize, u32, Vec2)> = world
        .network
        .junctions
        .iter()
        .filter(|j| !j.is_roundabout_node && j.arm_count >= 3)
        .filter_map(|j| {
            // Ensure all incident segments are long enough to accommodate approach flare trimming
            let min_len = j
                .in_segs
                .iter()
                .chain(j.out_segs.iter())
                .map(|&sid| world.network.segments[sid].length)
                .fold(f64::INFINITY, f64::min);
            if !min_len.is_finite() || min_len < (radius_m * 2.1).max(34.0) {
                return None;
            }
            let prio_sum: u32 = j
                .in_segs
                .iter()
                .map(|&sid| world.network.segments[sid].priority as u32)
                .sum();
            let score = (j.arm_count as u32) * 100 + prio_sum;
            Some((j.id, score, j.pos))
        })
        .collect();

    // Highest arm-count & arterial priority first, deterministic junction ID tie-breaker
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut chosen_centers: Vec<Vec2> = Vec::new();
    let mut converted = 0usize;
    for (jid, _, pos) in candidates {
        if converted >= max_roundabouts {
            break;
        }
        if chosen_centers.iter().any(|&c| c.dist(pos) < min_spacing_m) {
            continue;
        }
        if convert_junction_to_roundabout(world, jid, radius_m, ring_lanes).is_ok() {
            chosen_centers.push(pos);
            converted += 1;
        }
    }
    world.edit_history = saved_history;
    converted
}

fn execute_single_job(
    base_world: &SimulationWorld,
    cfg: &BatchExperimentConfig,
    job: SingleRunJob,
    tracker: Option<&BatchProgressTracker>,
) -> Option<SingleRunMetrics> {
    if tracker.map(|t| t.is_cancelled()).unwrap_or(false) {
        return None;
    }
    let mut world = base_world.clone();
    // Common Random Numbers (CRN): identical arrival seed across all 3 regimes for the same replicate!
    let seed = cfg
        .base_seed
        .wrapping_add((job.replicate_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95))
        .wrapping_add((job.demand_idx as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    world.set_rng_seed(seed);

    if cfg.include_tod_upzone {
        apply_tod_upzoning_policy(&mut world.parcels, &mut world.network);
        world.transit_policy.headway_min = 5.0;
        world.rebuild_cyclists();
    }

    let mut roundabouts_converted = 0usize;
    match job.regime {
        BatchControlRegime::FixedTime => {
            world.network.restore_fixed_time_signals();
        }
        BatchControlRegime::MaxPressureAdaptive => {
            world.network.enable_network_max_pressure();
        }
        BatchControlRegime::ModernRoundabout => {
            world.network.restore_fixed_time_signals();
            roundabouts_converted = convert_network_candidate_roundabouts(
                &mut world,
                cfg.roundabout_radius_m,
                cfg.roundabout_lanes,
                cfg.max_roundabouts,
            );
        }
    }

    world.base_demand_per_lane_hr =
        (world.base_demand_per_lane_hr * job.demand_mult.clamp(0.15, 4.0)).clamp(50.0, 1800.0);
    world.refresh_urban_metrics();
    world.telemetry_log.clear();

    let dt = cfg.dt_s.clamp(0.02, 0.25);
    let total_s = cfg.duration_s.max(10.0);
    let warmup_s = cfg.warmup_s.clamp(0.0, total_s * 0.6);
    let warmup_steps = (warmup_s / dt).round() as usize;
    let eval_steps = (((total_s - warmup_s).max(dt)) / dt).round() as usize;

    let mut pending_tick_report = 0usize;
    for w_i in 0..warmup_steps {
        if w_i % 20 == 0 && tracker.map(|t| t.is_cancelled()).unwrap_or(false) {
            return None;
        }
        world.step(dt);
        pending_tick_report += 1;
        if pending_tick_report >= 20 {
            if let Some(tr) = tracker {
                tr.completed_steps.fetch_add(pending_tick_report, Ordering::Relaxed);
            }
            pending_tick_report = 0;
        }
    }
    world.completed_trips.clear();
    world.telemetry_log.clear();

    let mut sum_speed_mph = 0.0;
    let mut sum_queued = 0.0;
    let mut sample_n = 0usize;

    for step_i in 0..eval_steps {
        if step_i % 20 == 0 && tracker.map(|t| t.is_cancelled()).unwrap_or(false) {
            return None;
        }
        world.step(dt);
        pending_tick_report += 1;
        if pending_tick_report >= 20 {
            if let Some(tr) = tracker {
                tr.completed_steps.fetch_add(pending_tick_report, Ordering::Relaxed);
            }
            pending_tick_report = 0;
        }
        if step_i % 10 == 0 || step_i + 1 == eval_steps {
            sum_speed_mph += world.mean_speed_mph();
            sum_queued += world.queued_vehicles() as f64;
            sample_n += 1;
        }
    }
    if let Some(tr) = tracker {
        if pending_tick_report > 0 {
            tr.completed_steps.fetch_add(pending_tick_report, Ordering::Relaxed);
        }
        tr.completed_runs.fetch_add(1, Ordering::Relaxed);
    }

    let eval_window_s = (eval_steps as f64 * dt).max(1.0);
    let throughput_veh_per_hr = (world.completed_trips.len() as f64 / eval_window_s) * 3600.0;
    let avg_trip_time_s = world.avg_trip_time_s();
    let mean_speed_mph = if sample_n > 0 {
        sum_speed_mph / sample_n as f64
    } else {
        world.mean_speed_mph()
    };
    let avg_queued_vehicles = if sample_n > 0 {
        sum_queued / sample_n as f64
    } else {
        world.queued_vehicles() as f64
    };

    Some(SingleRunMetrics {
        demand_idx: job.demand_idx,
        demand_mult: job.demand_mult,
        regime: job.regime,
        mean_speed_mph,
        avg_queued_vehicles,
        throughput_veh_per_hr,
        avg_trip_time_s,
        mp_controlled_junctions: world.mp_telemetry.controlled_junctions,
        mp_phase_switches: world.mp_telemetry.total_phase_switches,
        roundabouts_converted,
        gtfs_boardings: world.transit_gtfs.telemetry.cumulative_boardings,
        gtfs_mean_dwell_s: world.transit_gtfs.telemetry.mean_dwell_s,
        gtfs_load_factor_pct: world.transit_gtfs.telemetry.mean_load_factor_pct,
        vsp_co2_kg_per_hr: world.env_snapshot.co2_kg_per_hr,
        vsp_pm25_g_per_hr: world.env_snapshot.pm25_g_per_hr,
        corridor_noise_dba: world.env_snapshot.corridor_noise_dba,
        access_15min_pct: world.env_snapshot.access_15min_pct,
        net_fiscal_per_acre_k_yr: world.ledger.net_fiscal_per_acre_k_yr,
    })
}

/// Runs all `(demand_multiplier x regime x replicate)` cells concurrently using `Rayon` work-stealing
/// threads and compiles both per-demand rows and cross-demand `REGIME_MEAN_SUMMARY` roll-up rows.
pub fn run_parallel_batch_experiment(
    base_world: &SimulationWorld,
    cfg: &BatchExperimentConfig,
) -> BatchExperimentReport {
    run_parallel_batch_experiment_with_progress(base_world, cfg, None)
        .expect("uncancelled batch experiment always completes")
}

/// Executes the parallel `Rayon` experiment sweep while updating an optional lock-free
/// `BatchProgressTracker` (`None` returned immediately if `tracker.request_cancel()` is triggered).
pub fn run_parallel_batch_experiment_with_progress(
    base_world: &SimulationWorld,
    cfg: &BatchExperimentConfig,
    tracker: Option<&BatchProgressTracker>,
) -> Option<BatchExperimentReport> {
    let t0 = Instant::now();
    let demands = if cfg.demand_multipliers.is_empty() {
        vec![1.0]
    } else {
        cfg.demand_multipliers.clone()
    };
    let reps = cfg.replicates.max(1);

    let mut jobs = Vec::with_capacity(demands.len() * BatchControlRegime::ALL.len() * reps);
    for (d_idx, &d_mult) in demands.iter().enumerate() {
        for regime in BatchControlRegime::ALL {
            for rep in 0..reps {
                jobs.push(SingleRunJob {
                    demand_idx: d_idx,
                    demand_mult: d_mult,
                    replicate_idx: rep,
                    regime,
                });
            }
        }
    }

    let total_parallel_runs = jobs.len();
    let dt = cfg.dt_s.clamp(0.02, 0.25);
    let total_s = cfg.duration_s.max(10.0);
    let warmup_s = cfg.warmup_s.clamp(0.0, total_s * 0.6);
    let steps_per_run =
        ((warmup_s / dt).round() as usize) + ((((total_s - warmup_s).max(dt)) / dt).round() as usize);

    if let Some(tr) = tracker {
        tr.total_runs.store(total_parallel_runs, Ordering::Relaxed);
        tr.completed_runs.store(0, Ordering::Relaxed);
        tr.total_steps
            .store(total_parallel_runs * steps_per_run.max(1), Ordering::Relaxed);
        tr.completed_steps.store(0, Ordering::Relaxed);
    }

    let maybe_results: Vec<Option<SingleRunMetrics>> = jobs
        .into_par_iter()
        .map(|job| execute_single_job(base_world, cfg, job, tracker))
        .collect();

    if tracker.map(|t| t.is_cancelled()).unwrap_or(false) || maybe_results.iter().any(|r| r.is_none())
    {
        return None;
    }
    let results: Vec<SingleRunMetrics> = maybe_results.into_iter().flatten().collect();

    let mut scenario_rows: Vec<BatchScorecardRow> =
        Vec::with_capacity(demands.len() * BatchControlRegime::ALL.len());

    for (d_idx, &d_mult) in demands.iter().enumerate() {
        let mut group_rows: Vec<BatchScorecardRow> = Vec::with_capacity(3);
        for regime in BatchControlRegime::ALL {
            let matching: Vec<&SingleRunMetrics> = results
                .iter()
                .filter(|r| r.demand_idx == d_idx && r.regime == regime)
                .collect();
            let n = matching.len().max(1) as f64;
            let mean_speed = matching.iter().map(|r| r.mean_speed_mph).sum::<f64>() / n;
            let speed_var = if matching.len() > 1 {
                matching
                    .iter()
                    .map(|r| (r.mean_speed_mph - mean_speed).powi(2))
                    .sum::<f64>()
                    / ((matching.len() - 1) as f64)
            } else {
                0.0
            };
            let avg_q = matching.iter().map(|r| r.avg_queued_vehicles).sum::<f64>() / n;
            let thru = matching.iter().map(|r| r.throughput_veh_per_hr).sum::<f64>() / n;
            let trip_s = matching.iter().map(|r| r.avg_trip_time_s).sum::<f64>() / n;
            let mp_j = matching
                .first()
                .map(|r| r.mp_controlled_junctions)
                .unwrap_or(0);
            let mp_sw = (matching
                .iter()
                .map(|r| r.mp_phase_switches as f64)
                .sum::<f64>()
                / n)
                .round() as u64;
            let rb_conv = matching
                .first()
                .map(|r| r.roundabouts_converted)
                .unwrap_or(0);
            let gtfs_b = (matching
                .iter()
                .map(|r| r.gtfs_boardings as f64)
                .sum::<f64>()
                / n)
                .round() as u64;
            let gtfs_dw = matching.iter().map(|r| r.gtfs_mean_dwell_s).sum::<f64>() / n;
            let gtfs_lf = matching
                .iter()
                .map(|r| r.gtfs_load_factor_pct)
                .sum::<f64>()
                / n;
            let co2 = matching.iter().map(|r| r.vsp_co2_kg_per_hr).sum::<f64>() / n;
            let pm25 = matching.iter().map(|r| r.vsp_pm25_g_per_hr).sum::<f64>() / n;
            let noise = matching.iter().map(|r| r.corridor_noise_dba).sum::<f64>() / n;
            let acc15 = matching.iter().map(|r| r.access_15min_pct).sum::<f64>() / n;
            let fiscal = matching
                .iter()
                .map(|r| r.net_fiscal_per_acre_k_yr)
                .sum::<f64>()
                / n;

            group_rows.push(BatchScorecardRow {
                row_type: "SCENARIO_RUN".to_string(),
                demand_label: format!("{d_mult:.2}x"),
                demand_multiplier: d_mult,
                regime,
                regime_title: regime.title().to_string(),
                replicates: matching.len(),
                mean_speed_mph: mean_speed,
                speed_std_mph: speed_var.sqrt(),
                avg_queued_vehicles: avg_q,
                throughput_veh_per_hr: thru,
                avg_trip_time_s: trip_s,
                hcm_los: hcm_los_from_speed(mean_speed),
                mp_controlled_junctions: mp_j,
                mp_phase_switches: mp_sw,
                roundabouts_converted: rb_conv,
                gtfs_boardings: gtfs_b,
                gtfs_mean_dwell_s: gtfs_dw,
                gtfs_load_factor_pct: gtfs_lf,
                vsp_co2_kg_per_hr: co2,
                vsp_pm25_g_per_hr: pm25,
                corridor_noise_dba: noise,
                access_15min_pct: acc15,
                net_fiscal_per_acre_k_yr: fiscal,
                speed_delta_vs_fixed_pct: 0.0,
                queue_delta_vs_fixed_pct: 0.0,
                trip_time_delta_vs_fixed_pct: 0.0,
                throughput_delta_vs_fixed_pct: 0.0,
                co2_delta_vs_fixed_pct: 0.0,
            });
        }

        // Compute comparative deltas relative to Fixed-Time baseline at the same demand scale
        let base_speed = group_rows[0].mean_speed_mph;
        let base_q = group_rows[0].avg_queued_vehicles;
        let base_trip = group_rows[0].avg_trip_time_s;
        let base_thru = group_rows[0].throughput_veh_per_hr;
        let base_co2 = group_rows[0].vsp_co2_kg_per_hr;

        for row in &mut group_rows {
            row.speed_delta_vs_fixed_pct = pct_delta(row.mean_speed_mph, base_speed);
            row.queue_delta_vs_fixed_pct = pct_delta(row.avg_queued_vehicles, base_q);
            row.trip_time_delta_vs_fixed_pct = pct_delta(row.avg_trip_time_s, base_trip);
            row.throughput_delta_vs_fixed_pct = pct_delta(row.throughput_veh_per_hr, base_thru);
            row.co2_delta_vs_fixed_pct = pct_delta(row.vsp_co2_kg_per_hr, base_co2);
        }
        scenario_rows.extend(group_rows);
    }

    // Build cross-demand REGIME_MEAN_SUMMARY roll-up rows
    let mut regime_summary_rows: Vec<BatchScorecardRow> = Vec::with_capacity(3);
    for regime in BatchControlRegime::ALL {
        let sub: Vec<&BatchScorecardRow> = scenario_rows
            .iter()
            .filter(|r| r.regime == regime)
            .collect();
        let m = sub.len().max(1) as f64;
        let mean_speed = sub.iter().map(|r| r.mean_speed_mph).sum::<f64>() / m;
        regime_summary_rows.push(BatchScorecardRow {
            row_type: "REGIME_MEAN_SUMMARY".to_string(),
            demand_label: "ALL_SCALES_MEAN".to_string(),
            demand_multiplier: demands.iter().sum::<f64>() / demands.len().max(1) as f64,
            regime,
            regime_title: regime.title().to_string(),
            replicates: sub.iter().map(|r| r.replicates).sum(),
            mean_speed_mph: mean_speed,
            speed_std_mph: sub.iter().map(|r| r.speed_std_mph).sum::<f64>() / m,
            avg_queued_vehicles: sub.iter().map(|r| r.avg_queued_vehicles).sum::<f64>() / m,
            throughput_veh_per_hr: sub.iter().map(|r| r.throughput_veh_per_hr).sum::<f64>() / m,
            avg_trip_time_s: sub.iter().map(|r| r.avg_trip_time_s).sum::<f64>() / m,
            hcm_los: hcm_los_from_speed(mean_speed),
            mp_controlled_junctions: sub.first().map(|r| r.mp_controlled_junctions).unwrap_or(0),
            mp_phase_switches: (sub.iter().map(|r| r.mp_phase_switches as f64).sum::<f64>() / m)
                .round() as u64,
            roundabouts_converted: sub.first().map(|r| r.roundabouts_converted).unwrap_or(0),
            gtfs_boardings: (sub.iter().map(|r| r.gtfs_boardings as f64).sum::<f64>() / m).round()
                as u64,
            gtfs_mean_dwell_s: sub.iter().map(|r| r.gtfs_mean_dwell_s).sum::<f64>() / m,
            gtfs_load_factor_pct: sub.iter().map(|r| r.gtfs_load_factor_pct).sum::<f64>() / m,
            vsp_co2_kg_per_hr: sub.iter().map(|r| r.vsp_co2_kg_per_hr).sum::<f64>() / m,
            vsp_pm25_g_per_hr: sub.iter().map(|r| r.vsp_pm25_g_per_hr).sum::<f64>() / m,
            corridor_noise_dba: sub.iter().map(|r| r.corridor_noise_dba).sum::<f64>() / m,
            access_15min_pct: sub.iter().map(|r| r.access_15min_pct).sum::<f64>() / m,
            net_fiscal_per_acre_k_yr: sub.iter().map(|r| r.net_fiscal_per_acre_k_yr).sum::<f64>()
                / m,
            speed_delta_vs_fixed_pct: 0.0,
            queue_delta_vs_fixed_pct: 0.0,
            trip_time_delta_vs_fixed_pct: 0.0,
            throughput_delta_vs_fixed_pct: 0.0,
            co2_delta_vs_fixed_pct: 0.0,
        });
    }

    if !regime_summary_rows.is_empty() {
        let b_sp = regime_summary_rows[0].mean_speed_mph;
        let b_q = regime_summary_rows[0].avg_queued_vehicles;
        let b_tr = regime_summary_rows[0].avg_trip_time_s;
        let b_th = regime_summary_rows[0].throughput_veh_per_hr;
        let b_co2 = regime_summary_rows[0].vsp_co2_kg_per_hr;
        for row in &mut regime_summary_rows {
            row.speed_delta_vs_fixed_pct = pct_delta(row.mean_speed_mph, b_sp);
            row.queue_delta_vs_fixed_pct = pct_delta(row.avg_queued_vehicles, b_q);
            row.trip_time_delta_vs_fixed_pct = pct_delta(row.avg_trip_time_s, b_tr);
            row.throughput_delta_vs_fixed_pct = pct_delta(row.throughput_veh_per_hr, b_th);
            row.co2_delta_vs_fixed_pct = pct_delta(row.vsp_co2_kg_per_hr, b_co2);
        }
    }

    let elapsed_s = t0.elapsed().as_secs_f64().max(1e-6);
    let total_sim_s = total_parallel_runs as f64 * cfg.duration_s.max(10.0);

    Some(BatchExperimentReport {
        config: cfg.clone(),
        total_parallel_runs,
        total_simulated_seconds: total_sim_s,
        wall_clock_ms: elapsed_s * 1000.0,
        realtime_speedup_x: total_sim_s / elapsed_s,
        scenario_rows,
        regime_summary_rows,
    })
}

fn write_csv_row(out: &mut String, r: &BatchScorecardRow) {
    let _ = writeln!(
        out,
        "{},{},{:.2},{},\"{}\",{},{:.3},{:.3},{:.2},{:.1},{:.2},{},{},{},{},{},{:.2},{:.1},{:.2},{:.2},{:.2},{:.1},{:.2},{:+.2},{:+.2},{:+.2},{:+.2},{:+.2}",
        r.row_type,
        r.demand_label,
        r.demand_multiplier,
        r.regime.slug(),
        r.regime_title,
        r.replicates,
        r.mean_speed_mph,
        r.speed_std_mph,
        r.avg_queued_vehicles,
        r.throughput_veh_per_hr,
        r.avg_trip_time_s,
        r.hcm_los,
        r.mp_controlled_junctions,
        r.mp_phase_switches,
        r.roundabouts_converted,
        r.gtfs_boardings,
        r.gtfs_mean_dwell_s,
        r.gtfs_load_factor_pct,
        r.vsp_co2_kg_per_hr,
        r.vsp_pm25_g_per_hr,
        r.corridor_noise_dba,
        r.access_15min_pct,
        r.net_fiscal_per_acre_k_yr,
        r.speed_delta_vs_fixed_pct,
        r.queue_delta_vs_fixed_pct,
        r.trip_time_delta_vs_fixed_pct,
        r.throughput_delta_vs_fixed_pct,
        r.co2_delta_vs_fixed_pct
    );
}

impl BatchExperimentReport {
    /// Formats an RFC 4180 28-column `.csv` scorecard containing all per-demand scenario rows
    /// followed by `REGIME_MEAN_SUMMARY` cross-demand roll-up rows.
    pub fn to_csv_string(&self) -> String {
        let total_lines = 1 + self.scenario_rows.len() + self.regime_summary_rows.len();
        let mut out = String::with_capacity(total_lines * 180);
        let _ = writeln!(out, "{BATCH_SCORECARD_CSV_HEADER}");
        for r in &self.scenario_rows {
            write_csv_row(&mut out, r);
        }
        for r in &self.regime_summary_rows {
            write_csv_row(&mut out, r);
        }
        out
    }

    pub fn export_csv_path(&self, path: &Path) -> Result<usize, PersistenceError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let csv_body = self.to_csv_string();
        fs::write(path, csv_body)?;
        Ok(self.scenario_rows.len() + self.regime_summary_rows.len())
    }
}

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

//! CLI runner and formatted stdout scorecard printer for `urban-grid batch`, executing parallel
//! Fixed-Time, Max-Pressure Adaptive, and Modern Roundabout experiments on a configurable `Rayon`
//! thread pool and exporting RFC 4180 `.csv` + optional `.json` scorecards.

use rayon::ThreadPoolBuilder;
use std::fs;
use std::path::Path;
use ug_core::{
    run_parallel_batch_experiment, BatchExperimentConfig, BatchExperimentReport, SimulationWorld,
};

pub fn parse_demand_scales_arg(raw: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim().trim_end_matches('x');
        if trimmed.is_empty() {
            continue;
        }
        let val: f64 = trimmed
            .parse()
            .map_err(|_| format!("Invalid demand multiplier `{part}` (expected e.g. 0.75,1.0,1.35)"))?;
        if !(0.15..=4.0).contains(&val) {
            return Err(format!(
                "Demand multiplier {val} out of supported range [0.15, 4.0]"
            ));
        }
        out.push(val);
    }
    if out.is_empty() {
        return Err("At least one demand multiplier is required.".to_string());
    }
    Ok(out)
}

pub fn execute_batch_cli(
    base_world: &SimulationWorld,
    cfg: &BatchExperimentConfig,
    threads: Option<usize>,
    csv_out: &Path,
    json_out: Option<&Path>,
) -> Result<BatchExperimentReport, Box<dyn std::error::Error>> {
    let report = if let Some(n_threads) = threads.filter(|&n| n > 0) {
        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| format!("Rayon thread-pool error: {e}"))?;
        pool.install(|| run_parallel_batch_experiment(base_world, cfg))
    } else {
        run_parallel_batch_experiment(base_world, cfg)
    };

    let exported_rows = report.export_csv_path(csv_out)?;
    if let Some(jpath) = json_out {
        if let Some(parent) = jpath.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(jpath, serde_json::to_string_pretty(&report)?)?;
    }

    println!(
        "\n=== urban-grid v2.2 Parallel Batch Experiment Scorecard ({} runs | {:.1} ms | {:.0}x real-time) ===",
        report.total_parallel_runs, report.wall_clock_ms, report.realtime_speedup_x
    );
    println!(
        "{:<8} | {:<22} | {:>9} | {:>8} | {:>10} | {:>8} | {:>10} | {:>9}",
        "Demand",
        "Control Regime",
        "Speed mph",
        "Avg Q",
        "Thru veh/h",
        "Trip (s)",
        "CO2 kg/h",
        "vs Fixed"
    );
    println!("{}", "-".repeat(102));

    for r in &report.scenario_rows {
        println!(
            "{:<8} | {:<22} | {:>7.2} {} | {:>8.1} | {:>10.0} | {:>8.1} | {:>10.1} | {:>+8.1}%",
            r.demand_label,
            r.regime_title,
            r.mean_speed_mph,
            r.hcm_los,
            r.avg_queued_vehicles,
            r.throughput_veh_per_hr,
            r.avg_trip_time_s,
            r.vsp_co2_kg_per_hr,
            r.speed_delta_vs_fixed_pct
        );
    }

    println!("{}", "-".repeat(102));
    println!("Cross-Demand Roll-Up Summary (REGIME_MEAN_SUMMARY):");
    for r in &report.regime_summary_rows {
        println!(
            "  * {:<22} -> Mean {:.2} mph (LOS {}, {:+.1}% speed, {:+.1}% queue, {:+.1}% CO2 vs Fixed-Time)",
            r.regime_title,
            r.mean_speed_mph,
            r.hcm_los,
            r.speed_delta_vs_fixed_pct,
            r.queue_delta_vs_fixed_pct,
            r.co2_delta_vs_fixed_pct
        );
    }
    println!(
        "Exported {} scorecard rows (28 RFC 4180 columns) -> {}",
        exported_rows,
        csv_out.display()
    );

    Ok(report)
}

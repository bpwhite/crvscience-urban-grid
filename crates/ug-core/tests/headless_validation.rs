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

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Map, Value};
use ug_core::seismic_mc::{median_pga_g_from_sa10, median_sa10_gmpe_g};
use ug_core::{
    builtin_riverfront_bridges_osm, idm_acceleration, parse_osm_xml_str,
    run_evacuation_contraflow_monte_carlo, run_seismic_monte_carlo_simulation,
    solve_mclp_fire_station_siting, AeromedicalMedevacConfig, BridgeStructuralArchetype,
    BuriedPipeMaterialClass, BusynessModel, CarrierRef, ContraflowReversalPolicy,
    EvacuationContraflowConfig, EvacuationStagingStrategy, FfeConfig, HospitalSurgeConfig,
    IdmParams, MclpSitingConfig, PipelineFragilityLaw, RoadNetwork, SeismicScenarioConfig,
    SimulationWorld, SitingObjectiveModel,
};

static SUMMARY_LOCK: Mutex<()> = Mutex::new(());
static CSV_LOCK: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ug-core parent")
        .parent()
        .expect("crates parent")
        .to_path_buf()
}

fn validation_dir() -> PathBuf {
    let dir = repo_root().join("validation");
    fs::create_dir_all(&dir).expect("Failed creating validation dir");
    dir
}

fn merge_summary_section(section_key: &str, section_val: Value) {
    let _guard = SUMMARY_LOCK.lock().expect("Summary lock poisoned");
    let vdir = validation_dir();
    let local_path = vdir.join("urban_grid_validation_summary.json");

    let mut root: Map<String, Value> = if local_path.exists() {
        fs::read_to_string(&local_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    } else {
        Map::new()
    };

    root.insert("repository".into(), json!("crvscience-urban-grid"));
    root.insert("version".into(), json!(env!("CARGO_PKG_VERSION")));
    root.insert(
        "slug".into(),
        json!("urban-grid-microscopic-traffic-resilience"),
    );
    root.insert("preprint_id".into(), json!("APP-PREPRINT-11"));
    root.insert("concept_doi".into(), json!("10.5281/zenodo.24222394"));
    root.insert(
        "preprint_url".into(),
        json!("https://crvscience.com/apps/preprints/urban-grid-microscopic-traffic-resilience"),
    );
    root.insert(section_key.into(), section_val);

    let serialized =
        serde_json::to_string_pretty(&Value::Object(root)).expect("Serialize JSON failed");
    assert!(
        serialized.len() < 384 * 1024,
        "urban_grid_validation_summary.json exceeded 384 KB limit: {} bytes",
        serialized.len()
    );
    fs::write(&local_path, &serialized).expect("Write local summary failed");

    if let Some(rel_dir) = repo_root()
        .ancestors()
        .map(|anc| anc.join("app-preprints").join("releases"))
        .find(|p| p.exists())
    {
        let mirror =
            rel_dir.join("urban-grid-microscopic-traffic-resilience-validation-telemetry.json");
        fs::write(&mirror, &serialized).expect("Mirror summary JSON failed");
    }
}

fn upsert_csv_block(tag: &str, rows: Vec<String>) {
    let _guard = CSV_LOCK.lock().expect("CSV lock poisoned");
    let csv_path = validation_dir().join("traffic_and_evacuation_metrics.csv");
    let header = "record_type,scenario_label,parameter,primary_metric,secondary_metric,tertiary_metric,quaternary_metric,notes";

    let mut sections: BTreeMap<String, Vec<String>> = BTreeMap::new();
    if csv_path.exists() {
        if let Ok(raw) = fs::read_to_string(&csv_path) {
            for line in raw.lines().skip(1) {
                if let Some((rec_t, _)) = line.split_once(',') {
                    sections
                        .entry(rec_t.to_string())
                        .or_default()
                        .push(line.to_string());
                }
            }
        }
    }
    sections.insert(tag.to_string(), rows);

    let mut out = String::from(header);
    out.push('\n');
    for lines in sections.values() {
        for l in lines {
            out.push_str(l);
            out.push('\n');
        }
    }
    assert!(out.len() < 384 * 1024, "CSV exceeded 384 KB cap");
    fs::write(&csv_path, &out).expect("Write traffic_and_evacuation_metrics.csv failed");
}

#[test]
fn t1_idm_mobil_fundamental_diagram_and_headway_invariant() {
    // 1. Direct IDM physical braking invariant
    let idm = IdmParams::default();
    let accel_free = idm_acceleration(0.0, 15.0, 500.0, 0.0, idm.max_accel_mps2, &idm);
    let accel_brake = idm_acceleration(18.0, 18.0, 2.5, 14.0, idm.max_accel_mps2, &idm);
    assert!(accel_free > 0.8 && accel_brake < -3.0);

    // 2. Sweep demand across microscopic IDM + MOBIL worlds on compiled Riverfront Bridges Metro network
    let xml = builtin_riverfront_bridges_osm();
    let net_ref = RoadNetwork::compile(&parse_osm_xml_str(&xml).expect("OSM parse failed"));
    let total_open_lane_km: f64 = net_ref
        .lanes
        .iter()
        .filter(|l| !net_ref.segments[l.segment_id].closed)
        .map(|l| l.length / 1000.0)
        .sum();
    assert!(total_open_lane_km > 1.0);

    let demands = [240.0, 600.0, 960.0, 1320.0, 1680.0, 2100.0];
    let mut fd_rows = Vec::new();
    let mut csv_rows = Vec::new();
    let mut low_speed_mph = 0.0_f64;
    let mut high_speed_mph = 0.0_f64;
    let mut global_min_headway_gap_m = f64::INFINITY;
    let mut collision_events = 0usize;

    for (idx, &demand_hr) in demands.iter().enumerate() {
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        world.set_rng_seed(0x2026_1114 + idx as u64);
        world.base_demand_per_lane_hr = demand_hr;
        world.effective_demand_per_lane_hr = demand_hr;

        // Seed initial lane platoon scaled proportionally to demand regime
        let pre_target = (120.0 + demand_hr * 1.25).round() as usize;
        world.seed_50k_megacity_crowd(pre_target.max(400));
        // Thin extra spawned agents uniformly across all corridors so lower demand starts lighter cleanly
        if idx <= 2 {
            let stride = match idx {
                0 => 6usize,
                1 => 3,
                _ => 2,
            };
            let mut k = 0usize;
            world.vehicles.retain(|_| {
                let keep = k % stride == 0;
                k += 1;
                keep
            });
            let mut kc = 0usize;
            world.cyclists.retain(|_| {
                let keep = kc % stride == 0;
                kc += 1;
                keep
            });
            let mut kp = 0usize;
            world.pedestrians.retain(|_| {
                let keep = kp % stride == 0;
                kp += 1;
                keep
            });
        }

        // Warmup 12s (120 steps @ 0.1s) + sample 28s (280 steps @ 0.1s)
        let mut min_gap_m = f64::INFINITY;
        for step_i in 0..400 {
            world.step(0.1);
            if step_i >= 50 {
                let mut by_lane: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
                for v in &world.vehicles {
                    if let CarrierRef::Lane(lid) = v.carrier {
                        by_lane.entry(lid).or_default().push((v.s_m, v.length_m));
                    }
                }
                for list in by_lane.values_mut() {
                    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    for pair in list.windows(2) {
                        let follow_s = pair[0].0;
                        let (lead_s, lead_len) = pair[1];
                        let net_gap = lead_s - lead_len - follow_s;
                        if net_gap <= 0.0 {
                            collision_events += 1;
                        }
                        if net_gap < min_gap_m {
                            min_gap_m = net_gap;
                        }
                    }
                }
            }
        }

        let active_veh = world.vehicles.len();
        let density_veh_per_lane_km = (active_veh as f64) / total_open_lane_km;
        let mean_speed = world.mean_speed_mph();
        let throughput = world.throughput_veh_per_hr();
        let avg_trip_s = world.avg_trip_time_s();
        let queued = world.queued_vehicles();

        if min_gap_m < global_min_headway_gap_m {
            global_min_headway_gap_m = min_gap_m;
        }
        if idx == 0 {
            low_speed_mph = mean_speed;
        }
        if idx == demands.len() - 1 {
            high_speed_mph = mean_speed;
        }

        csv_rows.push(format!(
            "T1_FUNDAMENTAL_DIAGRAM,demand_{:.0}_vplh,{:.2},{:.2},{:.1},{:.3},{},{}_vehs_queued_{}",
            demand_hr,
            density_veh_per_lane_km,
            mean_speed,
            throughput,
            min_gap_m,
            avg_trip_s.round() as u64,
            active_veh,
            queued
        ));

        fd_rows.push(json!({
            "demand_veh_per_lane_hr": demand_hr,
            "active_vehicles": active_veh,
            "density_veh_per_lane_km": (density_veh_per_lane_km * 100.0).round() / 100.0,
            "mean_speed_mph": (mean_speed * 100.0).round() / 100.0,
            "throughput_veh_per_hr": (throughput * 10.0).round() / 10.0,
            "min_bumper_headway_m": (min_gap_m * 1000.0).round() / 1000.0,
            "queued_vehicles": queued,
            "completed_trips": world.completed_trips.len(),
        }));
    }

    assert_eq!(
        collision_events, 0,
        "IDM+MOBIL invariant violated: detected {collision_events} overlapping vehicle gaps"
    );
    assert!(
        global_min_headway_gap_m > 0.0,
        "Minimum bumper-to-bumper gap must stay strictly positive: {global_min_headway_gap_m:.4} m"
    );
    assert!(
        low_speed_mph > high_speed_mph,
        "Speed must decrease from light ({low_speed_mph:.2} mph) to saturated demand ({high_speed_mph:.2} mph)"
    );

    upsert_csv_block("T1_FUNDAMENTAL_DIAGRAM", csv_rows);
    merge_summary_section(
        "t1_idm_mobil_fundamental_diagram",
        json!({
            "collision_events": collision_events,
            "global_min_bumper_headway_m": (global_min_headway_gap_m * 1000.0).round() / 1000.0,
            "open_lane_km": (total_open_lane_km * 1000.0).round() / 1000.0,
            "idm_free_accel_mps2": (accel_free * 1000.0).round() / 1000.0,
            "idm_emergency_brake_mps2": (accel_brake * 1000.0).round() / 1000.0,
            "regimes": fd_rows,
        }),
    );
}

#[test]
fn t2_nfpa_1710_mclp_emergency_siting_optimization() {
    let xml = builtin_riverfront_bridges_osm();
    let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).expect("OSM parse failed"));
    let world = SimulationWorld::new(net);

    let baseline_rep = solve_mclp_fire_station_siting(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &MclpSitingConfig {
            objective_model: SitingObjectiveModel::DeterministicMclp1974,
            threshold_travel_s: 240.0,
            max_stations_budget: 0,
            target_coverage_pct: 90.0,
            ..MclpSitingConfig::default()
        },
    );

    let det_rep = solve_mclp_fire_station_siting(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &MclpSitingConfig {
            objective_model: SitingObjectiveModel::DeterministicMclp1974,
            threshold_travel_s: 240.0,
            max_stations_budget: 4,
            target_coverage_pct: 90.0,
            ..MclpSitingConfig::default()
        },
    );

    let larson_backup_rep = solve_mclp_fire_station_siting(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &MclpSitingConfig {
            objective_model: SitingObjectiveModel::MexclpPlusHoganBackup1986,
            busyness_model: BusynessModel::LarsonHypercube1974,
            threshold_travel_s: 240.0,
            second_due_threshold_s: 300.0,
            target_second_due_cov_pct: 70.0,
            max_stations_budget: 4,
            target_coverage_pct: 90.0,
            ..MclpSitingConfig::default()
        },
    );

    assert!(det_rep.target_met, "Deterministic MCLP must reach >=90% 4-min NFPA 1710 target");
    assert!(det_rep.achieved_parcel_cov_pct >= 90.0);
    assert!(det_rep.achieved_p90_travel_s <= 240.0);
    assert!(
        larson_backup_rep.achieved_parcel_cov_pct >= baseline_rep.achieved_parcel_cov_pct,
        "MCLP+Larson siting must improve over baseline existing stations"
    );

    let pareto_summary: Vec<Value> = larson_backup_rep
        .pareto_curve
        .iter()
        .map(|pt| {
            json!({
                "total_stations": pt.total_fire_stations,
                "total_apparatus": pt.total_apparatus,
                "busy_prob_q": (pt.effective_busy_prob_q * 10000.0).round() / 10000.0,
                "parcel_cov_4min_pct": (pt.parcels_within_threshold_pct * 100.0).round() / 100.0,
                "expected_cov_pct": (pt.expected_cov_pct * 100.0).round() / 100.0,
                "second_due_backup_pct": (pt.second_due_backup_cov_pct * 100.0).round() / 100.0,
                "p90_travel_s": (pt.p90_travel_s * 100.0).round() / 100.0,
            })
        })
        .collect();

    merge_summary_section(
        "t2_nfpa_1710_mclp_siting",
        json!({
            "nfpa_1710_threshold_s": 240.0,
            "baseline_existing_stations": baseline_rep.existing_kept_count,
            "baseline_parcel_cov_pct": (baseline_rep.achieved_parcel_cov_pct * 100.0).round() / 100.0,
            "baseline_p90_travel_s": (baseline_rep.achieved_p90_travel_s * 100.0).round() / 100.0,
            "deterministic_mclp": {
                "target_met": det_rep.target_met,
                "added_stations": det_rep.optimal_added_stations,
                "total_stations": det_rep.optimal_total_stations,
                "parcel_cov_pct": (det_rep.achieved_parcel_cov_pct * 100.0).round() / 100.0,
                "resident_cov_pct": (det_rep.achieved_resident_cov_pct * 100.0).round() / 100.0,
                "p90_travel_s": (det_rep.achieved_p90_travel_s * 100.0).round() / 100.0,
                "solve_time_ms": (det_rep.solve_time_ms * 100.0).round() / 100.0,
            },
            "mexclp_hogan_larson_hypercube": {
                "added_stations": larson_backup_rep.optimal_added_stations,
                "total_stations": larson_backup_rep.optimal_total_stations,
                "total_apparatus": larson_backup_rep.optimal_total_apparatus,
                "effective_busy_prob_q": (larson_backup_rep.effective_busy_prob_q * 10000.0).round() / 10000.0,
                "parcel_cov_pct": (larson_backup_rep.achieved_parcel_cov_pct * 100.0).round() / 100.0,
                "expected_cov_pct": (larson_backup_rep.achieved_expected_cov_pct * 100.0).round() / 100.0,
                "second_due_cov_pct": (larson_backup_rep.achieved_second_due_cov_pct * 100.0).round() / 100.0,
                "p90_travel_s": (larson_backup_rep.achieved_p90_travel_s * 100.0).round() / 100.0,
                "solve_time_ms": (larson_backup_rep.solve_time_ms * 100.0).round() / 100.0,
                "pareto_curve": pareto_summary,
            }
        }),
    );
}

#[test]
fn t3_cascadia_m9_seismic_fragility_and_contraflow_evacuation() {
    let xml = builtin_riverfront_bridges_osm();
    let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).expect("OSM parse failed"));
    let mut world = SimulationWorld::new(net);
    let sited = solve_mclp_fire_station_siting(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &MclpSitingConfig {
            objective_model: SitingObjectiveModel::DeterministicMclp1974,
            threshold_travel_s: 240.0,
            max_stations_budget: 3,
            target_coverage_pct: 90.0,
            ..MclpSitingConfig::default()
        },
    );
    world.apply_mclp_siting_solution(&sited);

    // 1. Verify M9.0 Cascadia GMPE PGA distance attenuation curve (15 km .. 150 km)
    let mut pga_attenuation = Vec::new();
    let mut csv_rows = Vec::new();
    for &r_km in &[15.0_f64, 30.0, 45.0, 75.0, 110.0, 150.0] {
        let sa10 = median_sa10_gmpe_g(9.0, r_km, 215.0);
        let pga = median_pga_g_from_sa10(sa10, 215.0);
        pga_attenuation.push((r_km, sa10, pga));
        csv_rows.push(format!(
            "T3_CASCADIA_M9_ATTENUATION,Mw9.0_Vs30_215,{:.0}km,{:.4},{:.4},{:.4},0,Sa1.0s_and_PGA_attenuation",
            r_km, pga, sa10, 0.0
        ));
    }
    assert!(
        pga_attenuation.windows(2).all(|w| w[0].2 > w[1].2 && w[0].1 > w[1].1),
        "M9.0 Cascadia PGA & Sa(1.0s) must attenuate monotonically with rupture distance"
    );

    // 2. Compare Unretrofitted Brittle CI/VCP vs ERDIP/HDPE + Retrofitted Bridges under Cascadia M9.0
    let brittle_cfg = SeismicScenarioConfig {
        moment_magnitude_mw: 9.0,
        rupture_standoff_km: 42.0,
        vs30_m_per_s: 215.0,
        monte_carlo_trials: 140,
        random_seed: 2026_0900,
        pipe_fragility_law: PipelineFragilityLaw::Ala2001LinearPgv,
        baseline_water_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
        baseline_sewer_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
        seismic_pipe_replacement_pct: 0.0,
        retrofit_top_k_bridges: 0,
        critical_lifeline_backup_power: false,
        ..SeismicScenarioConfig::default()
    };
    let hardened_cfg = SeismicScenarioConfig {
        baseline_water_pipe_material: BuriedPipeMaterialClass::SeismicErdipHdpe,
        baseline_sewer_pipe_material: BuriedPipeMaterialClass::SeismicErdipHdpe,
        seismic_pipe_replacement_pct: 85.0,
        retrofit_top_k_bridges: 6,
        retrofit_target_archetype: BridgeStructuralArchetype::EssentialBaseIsolatedLifeline,
        critical_lifeline_backup_power: true,
        ..brittle_cfg.clone()
    };

    let brittle_rep = run_seismic_monte_carlo_simulation(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &brittle_cfg,
    );
    let hardened_rep = run_seismic_monte_carlo_simulation(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &hardened_cfg,
    );

    assert!(
        hardened_rep.expected_day0_water_breaks < brittle_rep.expected_day0_water_breaks,
        "ERDIP/HDPE must reduce Day-0 buried water pipe breaks ({:.2} vs {:.2})",
        hardened_rep.expected_day0_water_breaks,
        brittle_rep.expected_day0_water_breaks
    );
    assert!(
        hardened_rep.multi_lifeline_resilience_90d_pct
            > brittle_rep.multi_lifeline_resilience_90d_pct,
        "Lifeline hardening + bridge retrofits must boost 90-day Bruneau resilience"
    );

    csv_rows.push(format!(
        "T3_CASCADIA_M9_SEISMIC,Brittle_CI_VCP_Baseline,Mw9.0_42km,{:.2},{:.4},{:.2},{:.2},water_breaks_{:.2}",
        brittle_rep.mean_parcel_cov_pct,
        brittle_rep.mean_median_sa10_g,
        brittle_rep.mean_p90_travel_s,
        brittle_rep.multi_lifeline_resilience_90d_pct,
        brittle_rep.expected_day0_water_breaks
    ));
    csv_rows.push(format!(
        "T3_CASCADIA_M9_SEISMIC,Hardened_ERDIP_Retrofitted,Mw9.0_42km,{:.2},{:.4},{:.2},{:.2},water_breaks_{:.2}",
        hardened_rep.mean_parcel_cov_pct,
        hardened_rep.mean_median_sa10_g,
        hardened_rep.mean_p90_travel_s,
        hardened_rep.multi_lifeline_resilience_90d_pct,
        hardened_rep.expected_day0_water_breaks
    ));

    // 3. Coupled Post-M9.0 Wildfire/FFE Contraflow Evacuation Clearance
    let base_evac = EvacuationContraflowConfig {
        medevac: AeromedicalMedevacConfig {
            surge: HospitalSurgeConfig {
                ffe: FfeConfig {
                    seismic: SeismicScenarioConfig {
                        monte_carlo_trials: 80,
                        ..brittle_cfg.clone()
                    },
                    wind_speed_mps: 12.0,
                    ..FfeConfig::default()
                },
                ..HospitalSurgeConfig::default()
            },
            ..AeromedicalMedevacConfig::default()
        },
        staging_strategy: EvacuationStagingStrategy::SimultaneousUnstaged,
        contraflow_policy: ContraflowReversalPolicy::NoContraflowBaseline,
        ..EvacuationContraflowConfig::default()
    };
    let no_cf = run_evacuation_contraflow_monte_carlo(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &base_evac,
    );
    let ems_cf = run_evacuation_contraflow_monte_carlo(
        &world.network,
        &world.parcels,
        &world.emergency_facilities,
        &world.evp_config,
        &EvacuationContraflowConfig {
            staging_strategy: EvacuationStagingStrategy::FireProximityThreeStage,
            contraflow_policy: ContraflowReversalPolicy::EmsProtectedArterialContraflow,
            ..base_evac
        },
    );

    assert!(ems_cf.contraflow_reversed_segments_count > 0);
    assert_eq!(ems_cf.ems_lz_corridor_access_preserved_pct, 100.0);
    assert!(
        ems_cf.peak_bottleneck_vc_ratio <= no_cf.peak_bottleneck_vc_ratio + 1e-6,
        "Staged EMS contraflow must ease bottleneck V/C ({:.3} vs {:.3})",
        ems_cf.peak_bottleneck_vc_ratio,
        no_cf.peak_bottleneck_vc_ratio
    );

    csv_rows.push(format!(
        "T3_CASCADIA_CONTRAFLOW_EVAC,Unstaged_NoContraflow,cf_0,{:.2},{:.3},{:.2},{:.2},ems_access_{:.1}pct",
        no_cf.ems_lz_corridor_access_preserved_pct,
        no_cf.peak_bottleneck_vc_ratio,
        no_cf.t90_clearance_min,
        no_cf.expected_fire_overrun_residents,
        no_cf.ems_lz_corridor_access_preserved_pct
    ));
    csv_rows.push(format!(
        "T3_CASCADIA_CONTRAFLOW_EVAC,Staged_EmsProtectedContraflow,cf_{},{:.2},{:.3},{:.2},{:.2},ems_access_{:.1}pct",
        ems_cf.contraflow_reversed_segments_count,
        ems_cf.ems_lz_corridor_access_preserved_pct,
        ems_cf.peak_bottleneck_vc_ratio,
        ems_cf.t90_clearance_min,
        ems_cf.expected_fire_overrun_residents,
        ems_cf.ems_lz_corridor_access_preserved_pct
    ));

    upsert_csv_block("T3_CASCADIA_M9_AND_EVAC", csv_rows);

    let pga_curve_json: Vec<Value> = pga_attenuation
        .iter()
        .map(|&(r, sa, pga)| {
            json!({
                "rupture_km": r,
                "median_sa10_g": (sa * 10000.0).round() / 10000.0,
                "median_pga_g": (pga * 10000.0).round() / 10000.0,
            })
        })
        .collect();

    merge_summary_section(
        "t3_cascadia_m9_seismic_and_evacuation",
        json!({
            "csv_artifact": "validation/traffic_and_evacuation_metrics.csv",
            "gmpe_mw9_attenuation": pga_curve_json,
            "seismic_brittle_baseline": {
                "mean_median_sa10_g": (brittle_rep.mean_median_sa10_g * 10000.0).round() / 10000.0,
                "median_pgv_cm_s": (brittle_rep.median_corridor_pgv_cm_s * 100.0).round() / 100.0,
                "day0_water_leaks": (brittle_rep.expected_day0_water_leaks * 100.0).round() / 100.0,
                "day0_water_breaks": (brittle_rep.expected_day0_water_breaks * 100.0).round() / 100.0,
                "day0_sewer_breaks": (brittle_rep.expected_day0_sewer_breaks * 100.0).round() / 100.0,
                "expected_closed_bridges": (brittle_rep.expected_closed_bridges * 100.0).round() / 100.0,
                "bruneau_bridge_resilience_90d_pct": (brittle_rep.bruneau_resilience_index_90d_pct * 100.0).round() / 100.0,
                "multi_lifeline_resilience_90d_pct": (brittle_rep.multi_lifeline_resilience_90d_pct * 100.0).round() / 100.0,
            },
            "seismic_hardened_erdip_retrofitted": {
                "day0_water_leaks": (hardened_rep.expected_day0_water_leaks * 100.0).round() / 100.0,
                "day0_water_breaks": (hardened_rep.expected_day0_water_breaks * 100.0).round() / 100.0,
                "day0_sewer_breaks": (hardened_rep.expected_day0_sewer_breaks * 100.0).round() / 100.0,
                "expected_closed_bridges": (hardened_rep.expected_closed_bridges * 100.0).round() / 100.0,
                "bruneau_bridge_resilience_90d_pct": (hardened_rep.bruneau_resilience_index_90d_pct * 100.0).round() / 100.0,
                "multi_lifeline_resilience_90d_pct": (hardened_rep.multi_lifeline_resilience_90d_pct * 100.0).round() / 100.0,
            },
            "contraflow_evacuation_no_cf": {
                "total_evacuees": no_cf.total_evacuees.round(),
                "total_evac_vehicles": no_cf.total_evac_vehicles.round(),
                "t50_clearance_min": (no_cf.t50_clearance_min * 100.0).round() / 100.0,
                "t90_clearance_min": (no_cf.t90_clearance_min * 100.0).round() / 100.0,
                "peak_bottleneck_vc_ratio": (no_cf.peak_bottleneck_vc_ratio * 1000.0).round() / 1000.0,
                "expected_fire_overrun_residents": (no_cf.expected_fire_overrun_residents * 100.0).round() / 100.0,
                "ems_access_preserved_pct": no_cf.ems_lz_corridor_access_preserved_pct,
            },
            "contraflow_evacuation_ems_protected": {
                "contraflow_reversed_segments": ems_cf.contraflow_reversed_segments_count,
                "contraflow_lane_km_added": (ems_cf.contraflow_lane_km_added * 1000.0).round() / 1000.0,
                "t50_clearance_min": (ems_cf.t50_clearance_min * 100.0).round() / 100.0,
                "t90_clearance_min": (ems_cf.t90_clearance_min * 100.0).round() / 100.0,
                "peak_bottleneck_vc_ratio": (ems_cf.peak_bottleneck_vc_ratio * 1000.0).round() / 1000.0,
                "expected_fire_overrun_residents": (ems_cf.expected_fire_overrun_residents * 100.0).round() / 100.0,
                "ems_access_preserved_pct": ems_cf.ems_lz_corridor_access_preserved_pct,
            }
        }),
    );
}

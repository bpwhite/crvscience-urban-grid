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

#![forbid(unsafe_code)]

pub mod aeromedical_medevac;
pub mod batch_eval;
pub mod burn_scar_debris_flow;
pub mod commodity_resupply;
pub mod diurnal;
pub mod economics;
pub mod emergency;
pub mod emergency_shelters;
pub mod environment;
pub mod evacuation_contraflow;
pub mod fema_comms;
pub mod fema_gsb_drc_tdms;
pub mod fema_isb_pods;
pub mod fema_usar_debris;
pub mod ffe_conflagration;
pub mod hospital_surge;
pub mod gpu_pack;
pub mod history;
pub mod hypercube;
pub mod max_pressure;
pub mod mclp;
pub mod multimodal;
pub mod network;
pub mod network_edit;
pub mod osm;
pub mod persistence;
pub mod resilience;
pub mod seismic_mc;
pub mod traffic;
pub mod transit_evacuation;
pub mod transit_gtfs;
pub mod utilities;
pub mod warning_mobilization;
pub mod wildfire_cameras;
pub mod wtp_turbidity_storage;
pub mod zoning;

pub use batch_eval::{
    convert_network_candidate_roundabouts, run_parallel_batch_experiment,
    run_parallel_batch_experiment_with_progress, BatchControlRegime,
    BatchExperimentConfig, BatchExperimentReport, BatchProgressTracker,
    BatchScorecardRow, BATCH_SCORECARD_CSV_HEADER,
};
pub use commodity_resupply::{
    holguin_veras_deprivation_cost, run_commodity_resupply_monte_carlo,
    solve_clarke_wright_2opt_tour, CommodityResupplyConfig, CommodityResupplyReport,
    DailyResupplyPoint, DemandNodeKind, ReliefCommodityKind, ResupplyDispatchPolicy,
    ScheduledConvoyTour, SiteResupplyTelemetryRow, RESUPPLY_DAILY_CSV_HEADER,
    RESUPPLY_SITE_CSV_HEADER,
};
pub use diurnal::{
    bpr_travel_time_multiplier, build_hourly_unit_travel_matrix, compute_hourly_parcel_weights,
    diurnal_arterial_vc_ratio, diurnal_call_multiplier, solve_diurnal_peak_offpeak_dispatch_tables,
    DiurnalDispatchRegime, DiurnalDispatchTable, DiurnalProfileConfig, DiurnalRedistrictingReport,
    DiurnalWindowPreset, HourlyDiurnalMetrics, ParcelDispatchRow, BPR_HOURLY_ARTERIAL_VC,
    DIURNAL_DISPATCH_TABLE_CSV_HEADER, DIURNAL_HOURLY_CSV_HEADER, USFA_DIURNAL_CALL_MULTIPLIERS,
};
pub use economics::{evaluate_municipal_ledger, MunicipalLedger, ScenarioSummaryReport};
pub use emergency::{
    evaluate_emergency_coverage, evp_override_connector_light, step_emergency_preemption,
    synthesize_default_facilities, EmergencyCoverageSummary, EmergencyFacility,
    EmergencyFacilityKind, EvpConfig, EvpTelemetry, JunctionEvpState,
};
pub use emergency_shelters::{
    compute_indoor_shelter_pm25_ug_m3, enumerate_candidate_shelters,
    run_emergency_shelters_monte_carlo, AirFiltrationStandard, CandidateShelterNode,
    EmergencySheltersConfig, EmergencySheltersReport, ParcelShelterAssignmentRow,
    ShelterFacilityArchetype, ShelterFiltrationCurvePoint, ShelterHabitabilityRow,
    ShelterSitingAndRoutingPolicy, SHELTER_FACILITY_CSV_HEADER, SHELTER_PARCEL_ASSIGN_CSV_HEADER,
};
pub use environment::{evaluate_environment_and_isochrones, EnvironmentalSnapshot};
pub use evacuation_contraflow::{
    plan_contraflow_and_protected_ems_corridors, run_evacuation_contraflow_monte_carlo,
    smoke_visibility_speed_and_capacity_factors, ContraflowCorridorSegment,
    ContraflowReversalPolicy, EvacuationContraflowConfig, EvacuationContraflowReport,
    EvacuationStagingStrategy, EvacuationWavePoint, ParcelEvacuationStageRow,
    EVACUATION_CORRIDOR_CSV_HEADER, EVACUATION_PARCEL_CSV_HEADER,
};
pub use fema_comms::{
    combine_ipaws_pathway_reach_prob, compute_pep_am_and_nwr_vhf_reliability,
    run_fema_comms_monte_carlo, EocIpaswUplinkMode, FemaAssetKind, FemaCommsAssetRow,
    FemaCommsConfig, FemaCommsPosture, FemaCommsReport, FemaIpaswTimelinePoint,
    FEMA_ASSET_CSV_HEADER, FEMA_TIMELINE_CSV_HEADER,
};
pub use fema_isb_pods::{
    evaluate_usace_pod_mgc_hour_step, run_fema_isb_pods_monte_carlo, FemaIsbPodsConfig,
    FemaIsbPodsReport, FemaPodClassification, IpaswPodRoutingPolicy, IsbReplenishmentPolicy,
    PodShiftHourPoint, PodSiteTelemetryRow, FEMA_POD_HOURLY_CSV_HEADER, FEMA_POD_SITE_CSV_HEADER,
};
pub use fema_usar_debris::{
    coburn_spence_entrapped_survival_prob, hazus_parcel_collapse_debris_tons,
    run_fema_usar_debris_monte_carlo, DebrisAndUsarPolicy, FemaInsaragMarking,
    FemaUsarDebrisConfig, FemaUsarDebrisReport, StructuralCollapseArchetype,
    UsarDebrisTimelinePoint, UsngSearchGridRow, FEMA_DEBRIS_TIMELINE_CSV_HEADER,
    FEMA_USNG_GRID_CSV_HEADER,
};
pub use fema_gsb_drc_tdms::{
    evaluate_drc_mric_mgc_wait_min, run_fema_gsb_drc_tdms_monte_carlo,
    size_usace_249th_prime_power_pack, FemaGsbDrcTdmsConfig, FemaGsbDrcTdmsReport,
    FemaRecoveryDailyPoint, FemaRecoveryExecutionPolicy, FemaRecoveryFacilityRow,
    FemaRecoverySiteKind, FEMA_RECOVERY_DAILY_CSV_HEADER, FEMA_RECOVERY_FACILITY_CSV_HEADER,
};
pub use wildfire_cameras::{
    builtin_western_wildfire_camera_feed_json, compute_rothermel_wui_fire_behavior,
    load_wildfire_camera_feed_path, nwcg_crew_and_dozer_line_rates_ft_hr,
    parse_wildfire_camera_feed_str, project_cameras_to_bounds, run_wildfire_alertwest_monte_carlo,
    triangulate_camera_bearings_wls, NwcgSuppressionThresholdClass, RothermelFireBehaviorMetrics,
    WildfireAlertWestConfig, WildfireAlertWestReport, WildfireCameraKind, WildfireCameraSite,
    WildfireDetectionAndAttackPosture, WildfireTimelinePoint, WuiFuelModel,
    WuiStructureTriageCategory, WILDFIRE_CAMERA_CSV_HEADER, WILDFIRE_TIMELINE_CSV_HEADER,
};
pub use burn_scar_debris_flow::{
    fhwa_hds5_culvert_headwater_ratio, nrcs_baer_hydrophobic_runoff_and_peak_q,
    run_burn_scar_debris_flow_monte_carlo, usgs_gartner_2014_debris_volume_m3,
    usgs_staley_m1_debris_flow_likelihood, usgs_staley_m1_i15_p50_threshold_mm_hr,
    BaerWatershedMitigationPolicy, BurnScarDebrisFlowConfig, BurnScarDebrisFlowReport,
    CulvertHydraulicState, IntensityThresholdCurvePoint, SoilBurnSeverityClass,
    StormwaterCulvertCrossingRow, UsgsCombinedHazardRating, UsgsDebrisFlowBasinRow,
    CULVERT_WASHOUT_CSV_HEADER, USGS_BASIN_CSV_HEADER,
};
pub use wtp_turbidity_storage::{
    compute_postfire_coagulant_jar_test, run_wtp_turbidity_storage_monte_carlo,
    CoagulantJarTestResult, WaterSupplyAssetKind, WaterSupplyFacilityRow,
    WtpCoagulantAndIntakePolicy, WtpHourlyTelemetryPoint, WtpTurbidityStorageConfig,
    WtpTurbidityStorageReport, WTP_FACILITY_CSV_HEADER, WTP_TIMELINE_CSV_HEADER,
};
pub use ffe_conflagration::{
    hamada_hazus_spread_rates, run_ffe_conflagration_monte_carlo,
    scawthorn_hazus_ignition_rate_per_mmsqft, FfeConfig, FfeIgnitionModel, FfeMonteCarloReport,
    HamadaDirectionalRates, ParcelFfeRiskRow, FFE_PARCEL_CSV_HEADER,
};
pub use aeromedical_medevac::{
    evaluate_helipad_flight_envelope, koschmieder_smoke_visibility_and_ceiling,
    run_aeromedical_medevac_monte_carlo, AeromedicalMedevacConfig, AeromedicalMedevacReport,
    HelipadStructuralType, HelipadTelemetryRow, InterHospitalTransferArc, MedevacAvionicsRegime,
    MedevacEpochPoint, RotorcraftAirframeClass, MEDEVAC_HELIPAD_CSV_HEADER,
    MEDEVAC_TRANSFER_ARC_CSV_HEADER,
};
pub use hospital_surge::{
    hazus_parcel_casualty_rates, hospital_fault_tree_functionality, run_hospital_surge_monte_carlo,
    synthesize_hospital_surge_roster, AmbulanceDiversionPolicy, CasualtyTriageAcuity,
    HospitalEdTelemetryRow, HospitalSurgeConfig, HospitalSurgeMonteCarloReport, HospitalSurgeNode,
    HospitalTraumaDesignation, ParcelCasualtyRoutingRow, SurgeEpochPoint,
    HOSPITAL_SURGE_FACILITY_CSV_HEADER, HOSPITAL_SURGE_PARCEL_CSV_HEADER,
};
pub use gpu_pack::{
    build_heatmap_rgba, pack_agents_into, pack_road_quads_into, response_time_4min_colormap,
    step_plume_advection_diffusion_cpu, GpuAgentInstance, GpuCameraUniform,
    GpuPlumeComputeUniform, GpuRoadQuadInstance, GpuStagingArena, HeatmapLayer,
    PlumePhysicsParams, AGENT_FLAG_BLINK_LEFT,
    AGENT_FLAG_BLINK_RIGHT, AGENT_FLAG_BRAKING, AGENT_FLAG_DWELLING, AGENT_KIND_BUS,
    AGENT_KIND_CYCLIST, AGENT_KIND_EMERGENCY, AGENT_KIND_PEDESTRIAN, AGENT_KIND_SEDAN,
    AGENT_KIND_TRUCK, HEATMAP_BYTES_PER_ROW, HEATMAP_BYTE_LEN, HEATMAP_DIM,
    ROAD_STYLE_BRIDGE_SPAN, ROAD_STYLE_BUS_STOP, ROAD_STYLE_EVP_PREEMPT, ROAD_STYLE_FIRE_STATION,
    ROAD_STYLE_HOSPITAL, ROAD_STYLE_WATER_CHANNEL,
};
pub use history::{
    apply_edit_snapshot_to_world, EditCategory, EditCommandRecord, EditHistoryStack,
    EditStateSnapshot, UndoRedoOutcome, DEFAULT_MAX_HISTORY_ENTRIES,
};
pub use max_pressure::{
    mp_connector_light, phase_for_connector, step_max_pressure_controllers, JunctionMpState,
    MaxPressureConfig, MaxPressureTelemetry, MpPhaseId, MpTransitionStage,
};
pub use hypercube::{
    build_network_unit_travel_matrix, build_offset_dispatch_preferences,
    optimize_carter_larson_districts_from_matrix, optimize_network_response_districts,
    solve_larson_hypercube_from_matrix, solve_larson_hypercube_with_explicit_primary_beats,
    solve_larson_hypercube_with_offsets, solve_larson_hypercube_with_pref, solve_network_hypercube,
    HypercubeConfig, HypercubeQueueDiscipline, HypercubeReport, HypercubeUnitMetrics,
    RedistrictingConfig, RedistrictingObjectiveMode, RedistrictingReport,
    RedistrictingTradeoffPoint, UnitDistrictSummary, HYPERCUBE_UNIT_CSV_HEADER,
    REDISTRICT_PARETO_CSV_HEADER, REDISTRICT_UNIT_CSV_HEADER,
};
pub use mclp::{
    daskin_expected_availability_prob, solve_mclp_fire_station_siting, BusynessModel,
    MclpCandidateSite, MclpOptimizationReport, MclpParetoPoint, MclpSitingConfig,
    MclpWeightingMode, SitingObjectiveModel, MEXCLP_PARETO_CSV_HEADER,
};
pub use multimodal::{evaluate_mode_split, ModeShares, TransitPolicy};
pub use network_edit::{
    adjust_segment_lane_count, convert_junction_to_roundabout, draw_road_segment,
    hit_test_junction, hit_test_segment, preview_draw_snap, rebuild_world_after_network_edit,
    set_segment_lane_count, NetworkEditError, NewRoadSpec, RoundaboutBuildSummary, SnapTarget,
};
pub use network::{
    sample_polyline, sample_polyline_fast, ControlMode, LightColor, RoadNetwork, TurnType, Vec2,
    WaterFeature, LANE_WIDTH_M,
};
pub use seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi, run_seismic_monte_carlo_simulation,
    sample_ala_hazus_pgv_curves, sample_hazus_restoration_curves, BridgeStructuralArchetype,
    BuriedPipeMaterialClass, MonteCarloTrialSummary, PipelineFragilityLaw, RecoveryEpochSummary,
    SeismicBridgeAsset, SeismicDamageState, SeismicMonteCarloReport, SeismicPipelineAssetSummary,
    SeismicScenarioConfig, HAZUS_BENCHMARK_DAYS, RECOVERY_TIMELINE_CSV_HEADER,
    RECOVERY_TIMELINE_DAYS, SEISMIC_BRIDGE_CSV_HEADER, SEISMIC_PIPELINE_CSV_HEADER,
};
pub use resilience::{
    enumerate_corridor_closure_units, evaluate_multi_scenario_resilience_diff,
    evaluate_resilience_stress_test, solve_robust_mclp_under_disruptions, CorridorClosureUnit,
    CorridorDisruptionOutcome, DisruptionInterdictionDepth, ResilienceScorecardDiffReport,
    ResilienceStressConfig, ResilienceStressReport, ScenarioResilienceScorecardRow,
    RESILIENCE_CSV_HEADER, RESILIENCE_DIFF_CSV_HEADER,
};
pub use persistence::{
    PersistenceError, ScenarioSnapshot, TelemetrySampleRow, TelemetryTimeSeriesRecorder,
    CSV_HEADER_COLUMNS, SNAPSHOT_SCHEMA_VERSION,
};
pub use osm::{
    builtin_braess_diamond_osm, builtin_foothill_canyon_wui_osm, builtin_harbor_peninsula_osm,
    builtin_metropolis_50k_osm, builtin_radial_ring_plazas_osm, builtin_riverfront_bridges_osm,
    builtin_roundabout_osm, load_osm_path, parse_osm_xml_str, parse_overpass_json_str, OsmError,
    RawOsmDataset,
};
pub use traffic::{
    idm_acceleration, CarrierRef, CyclistAgent, IdmParams, MobilParams, PedestrianAgent,
    SimulationWorld, Vehicle, VehicleClass,
};
pub use transit_evacuation::{
    compute_ada_van_capacity_and_dwell_min, compute_evac_coach_capacity_and_dwell_min,
    run_transit_bus_evacuation_monte_carlo, BusContraflowLaneMode, BusShuttleTourRow,
    PickupDepotTelemetryRow, TransitContraflowSegmentRow, TransitDispatchStrategy,
    TransitEvacWavePoint, TransitEvacuationConfig, TransitEvacuationReport,
    TRANSIT_EVAC_DEPOT_CSV_HEADER, TRANSIT_EVAC_WAVE_CSV_HEADER,
};
pub use transit_gtfs::{
    parse_gtfs_csv_table, shortest_segment_chain, BoardingFareMode, BusCoachSpec, DwellOutcome,
    GtfsError, TransitGtfsSystem, TransitRoute, TransitStop, TransitTelemetrySnapshot,
};
pub use utilities::{
    evaluate_coupled_utility_grids, hazen_williams_head_loss_m, manning_circular_pipe_d_over_d,
    manning_full_pipe_capacity_m3_s, ElectricalGridSummary, UtilityAssetKind,
    UtilityFacilityNode, UtilityGridsConfig, UtilityGridsReport, UtilityHeatmapSublayer,
    UtilityJunctionTelemetry, WastewaterGridSummary, WaterGridSummary,
    MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O, UTILITY_NODE_CSV_HEADER,
};
pub use warning_mobilization::{
    compute_siren_spl_dbc, evaluate_padm_mobilization_cdf, run_warning_mobilization_monte_carlo,
    AlertChannelMix, CellSiteMountArchetype, CellTowerTelemetryRow, DepartureStagingPolicy,
    MobilizationCurvePoint, SirenSiteRow, TelecomBackupPowerStandard, WarningMobilizationConfig,
    WarningMobilizationReport, WARNING_CURVE_CSV_HEADER, WARNING_TOWER_CSV_HEADER,
};
pub use zoning::{
    apply_tod_upzoning_policy, compute_gravity_od_matrix, generate_parcels_for_bounds,
    generate_parcels_for_network, Parcel, ZoneCode,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_gpu_pod_layouts_match_wgsl_byte_alignment() {
        assert_eq!(size_of::<GpuCameraUniform>(), 48);
        assert_eq!(size_of::<GpuPlumeComputeUniform>(), 48);
        assert_eq!(align_of::<GpuCameraUniform>(), 4);
        assert_eq!(size_of::<GpuRoadQuadInstance>(), 32);
        assert_eq!(size_of::<GpuAgentInstance>(), 32);
        assert_eq!(HEATMAP_BYTES_PER_ROW % 256, 0);
        assert_eq!(HEATMAP_BYTE_LEN, 256 * 256 * 4);
    }

    #[test]
    fn test_riverfront_bridges_osm_compiles_lanes_signals_and_water() {
        let xml = builtin_riverfront_bridges_osm();
        let osm = parse_osm_xml_str(&xml).expect("valid Riverfront Bridges OSM");
        let net = RoadNetwork::compile(&osm);
        assert!(net.junctions.len() >= 18);
        assert!(net.lanes.len() >= 50);
        assert!(!net.water_features.is_empty());
        assert!(net.segments.iter().any(|s| s.is_bridge));
        assert!(!net.sources.is_empty());
        assert!(!net.sinks.is_empty());
    }

    #[test]
    fn test_idm_braking_prevents_collision() {
        let idm = IdmParams::default();
        let free_acc = idm_acceleration(10.0, 15.0, 500.0, 0.0, 1.6, &idm);
        assert!(free_acc > 0.5);
        let brake_acc = idm_acceleration(15.0, 15.0, 3.0, 12.0, 1.6, &idm);
        assert!(brake_acc < -3.0);
    }

    #[test]
    fn test_wilson_gravity_matrix_row_conservation() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let parcels = generate_parcels_for_bounds(net.bounds);
        let od = compute_gravity_od_matrix(&parcels, 0.12);
        assert_eq!(od.len(), parcels.len());
        let total_trips: f64 = od.iter().flatten().sum();
        assert!(total_trips > 50.0);
    }

    #[test]
    fn test_tod_upzoning_shifts_mcfadden_mode_share_and_fiscal_yield() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let base_auto = world.mode_shares.auto;
        let base_fiscal = world.ledger.net_fiscal_per_acre_k_yr;

        apply_tod_upzoning_policy(&mut world.parcels, &mut world.network);
        world.transit_policy.headway_min = 5.0;
        world.refresh_urban_metrics();

        assert!(world.mode_shares.auto < base_auto);
        assert!(world.mode_shares.transit + world.mode_shares.bike > 0.25);
        assert!(world.ledger.net_fiscal_per_acre_k_yr > base_fiscal);
    }

    #[test]
    fn test_gtfs_csv_ingestion_dwell_times_and_live_crowding() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let parcels = generate_parcels_for_bounds(net.bounds);

        let stops_csv = "stop_id,stop_name,stop_lat,stop_lon\nS1,\"West Couplet North\",40.000950,-105.002500\nS2,\"West Couplet South\",39.999050,-105.002500\n";
        let routes_csv = "route_id,route_short_name,route_long_name,route_type,route_color\nR1,1,\"Metro Trunk Line 1\",3,EF4444\n";
        let trips_csv = "route_id,service_id,trip_id,direction_id\nR1,WK,T1,0\n";
        let st_csv = "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nT1,08:00:00,08:00:20,S1,1\nT1,08:02:00,08:02:20,S2,2\n";

        let mut gtfs = TransitGtfsSystem::from_gtfs_csv_strs(
            stops_csv, routes_csv, trips_csv, st_csv, &net, &parcels, 8.0,
        )
        .expect("GTFS CSV parses and snaps cleanly");
        assert_eq!(gtfs.stops.len(), 2);
        assert_eq!(gtfs.routes.len(), 1);

        // Verify front-door cash dwell exceeds all-door fare-free BRT dwell for identical queue
        gtfs.stops[0].waiting_pax = 14.0;
        gtfs.fare_mode = BoardingFareMode::AllDoorFareFreeBrt;
        let brt_dwell = gtfs.process_bus_stop_service(0, 0, 20, 10.0, 8.0);
        gtfs.stops[0].waiting_pax = 14.0;
        gtfs.fare_mode = BoardingFareMode::FrontDoorCash;
        let cash_dwell = gtfs.process_bus_stop_service(0, 0, 20, 30.0, 8.0);
        assert!(cash_dwell.dwell_s > brt_dwell.dwell_s + 15.0);
        assert_eq!(brt_dwell.boarded, 14);

        // Verify end-to-end microscopic GTFS bus execution inside SimulationWorld
        let mut world = SimulationWorld::new(net);
        assert!(!world.transit_gtfs.routes.is_empty());
        assert!(!world.transit_gtfs.stops.is_empty());
        for _ in 0..900 {
            world.step(0.1);
        }
        assert!(world.transit_gtfs.telemetry.cumulative_boardings > 0);
        assert!(world.transit_gtfs.telemetry.mean_dwell_s >= 2.4);
    }

    #[test]
    fn test_megacity_50k_agents_pack_and_heatmap_rasterize() {
        let xml = builtin_metropolis_50k_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).expect("valid Megacity OSM"));
        let mut world = SimulationWorld::new(net);
        world.seed_50k_megacity_crowd(52_000);
        assert!(world.total_active_agents() >= 50_000);

        world.step(1.0 / 60.0);
        let mut arena = GpuStagingArena::new();
        pack_road_quads_into(&world, &mut arena.road_quads);
        pack_agents_into(&world, 0.004, &mut arena.agents);
        build_heatmap_rgba(
            &world,
            HeatmapLayer::AgentDensity,
            &arena.agents.clone(),
            &mut arena,
        );

        assert_eq!(arena.agents.len(), world.total_active_agents());
        assert!(!arena.road_quads.is_empty());
        assert_eq!(arena.heatmap_rgba.len(), HEATMAP_BYTE_LEN);
        assert!(arena.heatmap_rgba.iter().any(|&b| b > 0));
    }


    #[test]
    fn test_semi_lagrangian_plume_advects_downwind_and_decays() {
        let n = HEATMAP_DIM * HEATMAP_DIM;
        let mut src = vec![0.0f32; n];
        let mut inject = vec![0.0f32; n];
        let mut dst = vec![0.0f32; n];

        let cy = 128usize;
        let cx = 100usize;
        inject[cy * HEATMAP_DIM + cx] = 4.0;

        let params = PlumePhysicsParams {
            wind_x_mps: 12.0, // Strong westerly wind blowing East (+X)
            wind_y_mps: 0.0,
            diffusivity: 0.35,
            decay_rate: 0.4,
        };
        step_plume_advection_diffusion_cpu(&src, &inject, &mut dst, [512.0, 512.0], params, 0.25);
        src.copy_from_slice(&dst);
        inject.fill(0.0);
        for _ in 0..8 {
            step_plume_advection_diffusion_cpu(
                &src,
                &inject,
                &mut dst,
                [512.0, 512.0],
                params,
                0.25,
            );
            src.copy_from_slice(&dst);
        }

        // Peak center of mass along row `cy` must have advected strictly East (`x > cx`)!
        let (peak_x, &peak_val) = src[cy * HEATMAP_DIM..(cy + 1) * HEATMAP_DIM]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(peak_x > cx + 4);
        assert!(peak_val > 0.01);
    }


    #[test]
    fn test_scenario_snapshot_json_roundtrip_and_csv_telemetry_export() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        apply_tod_upzoning_policy(&mut world.parcels, &mut world.network);
        world.network.junctions[0].ns_green_s = 37.5;
        for _ in 0..50 {
            world.step(0.1);
        }

        let plume = PlumePhysicsParams {
            wind_x_mps: 5.5,
            wind_y_mps: -2.0,
            diffusivity: 0.52,
            decay_rate: 0.70,
        };
        let snap = ScenarioSnapshot::capture(&world, plume, "Riverfront Bridges TOD Test", true);
        let json_str = snap.to_json_pretty().expect("JSON serializes cleanly without tuple-key error");
        assert!(json_str.contains(SNAPSHOT_SCHEMA_VERSION));

        let parsed = ScenarioSnapshot::from_json_str(&json_str).expect("JSON deserializes cleanly");
        let (restored, restored_plume) = parsed.into_world_and_plume();
        assert_eq!(restored.parcels.len(), world.parcels.len());
        assert_eq!(restored.vehicles.len(), world.vehicles.len());
        assert!((restored.network.junctions[0].ns_green_s - 37.5).abs() < 1e-6);
        assert!((restored_plume.wind_x_mps - 5.5).abs() < 1e-5);
        assert!(!restored.network.next_hop.is_empty());

        let csv_str = world.telemetry_log.to_csv_string();
        let lines: Vec<&str> = csv_str.lines().collect();
        assert!(lines.len() >= 5);
        let header_cols = lines[0].split(',').count();
        assert_eq!(header_cols, 29);
        for row_line in &lines[1..] {
            assert_eq!(row_line.split(',').count(), header_cols);
        }
    }


    #[test]
    fn test_max_pressure_adapts_green_to_asymmetric_queue_and_gates_spillback() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let upgraded = world.network.enable_network_max_pressure();
        assert!(upgraded >= 12);

        // Pick a signalized MaxPressureAdaptive junction with both Axis 0 and Axis 1 incoming segments
        let jid = world
            .network
            .junctions
            .iter()
            .find(|j| {
                j.control_mode == ControlMode::MaxPressureAdaptive
                    && j.in_segs.iter().any(|&s| world.network.segments[s].approach_axis == 0)
                    && j.in_segs.iter().any(|&s| world.network.segments[s].approach_axis == 1)
            })
            .map(|j| j.id)
            .expect("4-way signalized junction exists");

        // Force junction to start on Axis 0 Green, while injecting a heavy asymmetric queue on Axis 1!
        world.network.junctions[jid].mp_state.active_phase = MpPhaseId::Axis0ThroughRight;
        world.network.junctions[jid].mp_state.stage = MpTransitionStage::Green;
        world.network.junctions[jid].mp_state.stage_elapsed_s = 0.0;
        world.vehicles.clear();
        world.base_demand_per_lane_hr = 0.0;
        world.transit_policy.couple_gravity_demand = false;

        let axis1_seg = *world.network.junctions[jid]
            .in_segs
            .iter()
            .find(|&&s| world.network.segments[s].approach_axis == 1)
            .unwrap();
        let axis1_lid = world.network.segments[axis1_seg].lane_ids[0];
        let lane_len = world.network.lanes[axis1_lid].length;

        for k in 0..5 {
            world.vehicles.push(Vehicle {
                id: 9000 + k,
                v_class: VehicleClass::Sedan,
                length_m: 4.5,
                width_m: 1.9,
                accel_mult: 1.0,
                v_max_mps: 15.0,
                carrier: CarrierRef::Lane(axis1_lid),
                s_m: (lane_len - 2.0 - k as f64 * 6.8).max(1.0),
                v: 0.0,
                acc: 0.0,
                lat_visual_offset_m: 0.0,
                lc_cooldown_s: 0.0,
                dest_seg_id: None,
                planned_conn_id: None,
                turn_intent: TurnType::Through,
                spawn_time_s: 0.0,
                gtfs_route_idx: None,
                gtfs_route_seq: 0,
                gtfs_next_stop_order: 0,
                dwell_remaining_s: 0.0,
                onboard_pax: 0,
                crush_capacity: 0,
            });
        }

        // Advance past min_green (7s) + yellow (3.5s) + all_red (1.5s) = ~13.5s
        let mut saw_axis1_green = false;
        for _ in 0..160 {
            world.step(0.1);
            let st = &world.network.junctions[jid].mp_state;
            if st.stage == MpTransitionStage::Green
                && matches!(
                    st.active_phase,
                    MpPhaseId::Axis1ThroughRight | MpPhaseId::Axis1ProtectedLeft
                )
            {
                saw_axis1_green = true;
                break;
            }
        }
        assert!(
            saw_axis1_green,
            "Max-Pressure must switch green to the heavily queued Axis 1 approach"
        );
        assert!(world.mp_telemetry.total_phase_switches >= 1);
    }


    #[test]
    fn test_interactive_road_draw_lane_edit_and_roundabout_conversion_preserve_simulation() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        for _ in 0..80 {
            world.step(0.1);
        }
        assert!(!world.vehicles.is_empty());

        // 1. Lane-count editing (+1 widening & road diet) on segment 0 while vehicles are active
        let orig_lanes = world.network.segments[0].lane_count;
        let widened = set_segment_lane_count(&mut world, 0, orig_lanes + 1, true)
            .expect("lane widening succeeds");
        assert_eq!(widened, (orig_lanes + 1).min(5));

        // 2. Interactive road-segment drawing across blocks (snapping / T-splitting)
        let b = world.network.bounds;
        let p_start = Vec2 {
            x: b.min_x + 45.0,
            y: 0.5 * (b.min_y + b.max_y),
        };
        let p_end = Vec2 {
            x: b.max_x - 45.0,
            y: 0.5 * (b.min_y + b.max_y) + 35.0,
        };
        let created = draw_road_segment(&mut world, p_start, p_end, NewRoadSpec::default())
            .expect("road drawing creates bidirectional segments");
        assert_eq!(created.len(), 2);

        // 3. Modern Roundabout conversion on a 4-arm intersection
        let candidate_jid = world
            .network
            .junctions
            .iter()
            .find(|j| !j.is_roundabout_node && j.arm_count >= 4)
            .map(|j| j.id)
            .expect("4-arm intersection exists");
        let rb_summary = convert_junction_to_roundabout(&mut world, candidate_jid, 20.0, 2)
            .expect("roundabout conversion succeeds");
        assert!(rb_summary.arms_connected >= 4);
        assert_eq!(rb_summary.ring_lanes, 2);

        // Step world 100 ticks after all three live topology edits to verify zero index panics!
        for _ in 0..100 {
            world.step(0.1);
        }
        assert!(!world.vehicles.is_empty());
    }

    #[test]
    fn test_simulation_world_steps_cleanly_and_completes_trips() {
        let xml = builtin_roundabout_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        for _ in 0..800 {
            world.step(0.1);
        }
        assert!(world.vehicles.len() >= 10);
        assert!(!world.completed_trips.is_empty());
        assert!(world.env_snapshot.co2_kg_per_hr > 0.0);
    }

    #[test]
    fn test_undo_redo_command_stack_across_road_draw_lanes_roundabout_and_zoning() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        for _ in 0..40 {
            world.step(0.1);
        }

        let base_segs = world.network.segments.len();
        let base_juncs = world.network.junctions.len();
        let base_seg0_lanes = world.network.segments[0].lane_count;
        let base_parcel0_zone = world.parcels[0].zone;
        let target_zone = if base_parcel0_zone == ZoneCode::CBD {
            ZoneCode::PARK
        } else {
            ZoneCode::CBD
        };

        // 1. Zoning edit (lightweight ZoningAndPolicy memento)
        assert!(world.set_parcel_zone(0, target_zone));
        assert!(!world.set_parcel_zone(0, target_zone)); // idempotent repeat does not spam stack
        assert_eq!(world.edit_history.undo_len(), 1);

        // 2. Lane-count edit
        let new_lanes = set_segment_lane_count(&mut world, 0, base_seg0_lanes + 1, true)
            .expect("lane widening succeeds");
        assert_eq!(world.edit_history.undo_len(), 2);

        // 3. Road-segment drawing
        let b = world.network.bounds;
        let p_a = Vec2 {
            x: b.min_x + 42.0,
            y: 0.5 * (b.min_y + b.max_y) - 18.0,
        };
        let p_b = Vec2 {
            x: b.max_x - 42.0,
            y: 0.5 * (b.min_y + b.max_y) + 28.0,
        };
        draw_road_segment(&mut world, p_a, p_b, NewRoadSpec::default())
            .expect("road drawing succeeds");
        let segs_after_draw = world.network.segments.len();
        assert!(segs_after_draw > base_segs);
        assert_eq!(world.edit_history.undo_len(), 3);

        // 4. Roundabout conversion
        let rb_jid = world
            .network
            .junctions
            .iter()
            .find(|j| !j.is_roundabout_node && j.arm_count >= 4)
            .map(|j| j.id)
            .expect("4-arm intersection exists");
        convert_junction_to_roundabout(&mut world, rb_jid, 20.0, 2)
            .expect("roundabout conversion succeeds");
        let segs_after_rb = world.network.segments.len();
        assert!(segs_after_rb > segs_after_draw);
        assert_eq!(world.edit_history.undo_len(), 4);
        assert_eq!(world.edit_history.redo_len(), 0);

        // Advance live microscopic simulation while vehicles traverse edited topology
        for _ in 0..30 {
            world.step(0.1);
        }

        // Undo all 4 edits in reverse order (Ctrl+Z x4)
        let u1 = world.undo_edit().expect("undo roundabout");
        assert_eq!(u1.category, EditCategory::ConvertRoundabout);
        assert_eq!(world.network.segments.len(), segs_after_draw);

        let u2 = world.undo_edit().expect("undo road draw");
        assert_eq!(u2.category, EditCategory::DrawRoad);
        assert_eq!(world.network.segments.len(), base_segs);
        assert_eq!(world.network.junctions.len(), base_juncs);

        let u3 = world.undo_edit().expect("undo lane edit");
        assert_eq!(u3.category, EditCategory::EditLanes);
        assert_eq!(world.network.segments[0].lane_count, base_seg0_lanes);

        let u4 = world.undo_edit().expect("undo parcel zoning");
        assert_eq!(u4.category, EditCategory::PaintZoning);
        assert_eq!(world.parcels[0].zone, base_parcel0_zone);
        assert!(world.undo_edit().is_none());
        assert_eq!(world.edit_history.redo_len(), 4);

        for _ in 0..20 {
            world.step(0.1);
        }

        // Redo all 4 edits in forward order (Ctrl+Y x4)
        let r1 = world.redo_edit().expect("redo parcel zoning");
        assert_eq!(r1.category, EditCategory::PaintZoning);
        assert_eq!(world.parcels[0].zone, target_zone);

        let r2 = world.redo_edit().expect("redo lane edit");
        assert_eq!(r2.category, EditCategory::EditLanes);
        assert_eq!(world.network.segments[0].lane_count, new_lanes);

        let r3 = world.redo_edit().expect("redo road draw");
        assert_eq!(r3.category, EditCategory::DrawRoad);
        assert_eq!(world.network.segments.len(), segs_after_draw);

        let r4 = world.redo_edit().expect("redo roundabout");
        assert_eq!(r4.category, EditCategory::ConvertRoundabout);
        assert_eq!(world.network.segments.len(), segs_after_rb);
        assert!(world.redo_edit().is_none());

        // Branching edit after undo must clear redo stack
        world.undo_edit().expect("undo once before branching");
        assert_eq!(world.edit_history.redo_len(), 1);
        world.apply_tod_upzoning_tracked();
        assert_eq!(world.edit_history.redo_len(), 0);
    }

    #[test]
    fn test_rayon_parallel_batch_runner_compares_fixed_max_pressure_and_roundabouts() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let base_world = SimulationWorld::new(net);

        let cfg = BatchExperimentConfig {
            duration_s: 35.0,
            warmup_s: 10.0,
            dt_s: 0.1,
            demand_multipliers: vec![0.80, 1.25],
            replicates: 2,
            roundabout_radius_m: 18.0,
            roundabout_lanes: 2,
            max_roundabouts: 4,
            include_tod_upzone: false,
            base_seed: 0xCAFE_BABE_1234_5678,
        };

        let report = run_parallel_batch_experiment(&base_world, &cfg);
        assert_eq!(report.total_parallel_runs, 2 * 3 * 2);
        assert_eq!(report.scenario_rows.len(), 6);
        assert_eq!(report.regime_summary_rows.len(), 3);

        // Verify Max-Pressure controlled junctions and Roundabout conversions took effect
        let mp_row = report
            .regime_summary_rows
            .iter()
            .find(|r| r.regime == BatchControlRegime::MaxPressureAdaptive)
            .unwrap();
        assert!(mp_row.mp_controlled_junctions >= 10);

        let rb_row = report
            .regime_summary_rows
            .iter()
            .find(|r| r.regime == BatchControlRegime::ModernRoundabout)
            .unwrap();
        assert!(rb_row.roundabouts_converted >= 1);

        // Verify RFC 4180 CSV scorecard formatting (header + 6 scenario rows + 3 summary rows = 10 lines, 28 columns each)
        let csv_text = report.to_csv_string();
        let lines: Vec<&str> = csv_text.lines().collect();
        assert_eq!(lines.len(), 10);
        assert_eq!(lines[0].split(',').count(), 28);
        let parsed_rows = parse_gtfs_csv_table(&csv_text);
        assert_eq!(parsed_rows.len(), 9);
        for pr in &parsed_rows {
            assert_eq!(pr.len(), 28);
            assert!(pr.contains_key("speed_delta_vs_fixed_pct"));
        }
    }

    #[test]
    fn test_batch_progress_tracker_and_cooperative_cancellation() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let base_world = SimulationWorld::new(net);
        let cfg = BatchExperimentConfig {
            duration_s: 25.0,
            warmup_s: 5.0,
            dt_s: 0.1,
            demand_multipliers: vec![1.0],
            replicates: 1,
            ..BatchExperimentConfig::default()
        };

        // 1. Normal completion tracks 3/3 runs and 100% progress
        let tracker = BatchProgressTracker::new();
        let rep = run_parallel_batch_experiment_with_progress(&base_world, &cfg, Some(&tracker))
            .expect("sweep completes when uncancelled");
        let (done, tot, frac) = tracker.snapshot_progress();
        assert_eq!(done, 3);
        assert_eq!(tot, 3);
        assert!((frac - 1.0).abs() < 1e-4);
        assert_eq!(rep.regime_summary_rows.len(), 3);

        // 2. Pre-cancelled tracker aborts immediately without blocking
        let cancel_tracker = BatchProgressTracker::new();
        cancel_tracker.request_cancel();
        let aborted =
            run_parallel_batch_experiment_with_progress(&base_world, &cfg, Some(&cancel_tracker));
        assert!(aborted.is_none());
    }

    #[test]
    fn test_emergency_responder_corridor_green_wave_preemption_and_mobil_yielding() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        world.evp_config.auto_dispatch_per_min = 0.0;

        // Pick a multi-lane signalized corridor so both MOBIL right-lane changes and shoulder pull-overs occur
        let seg_id = world
            .network
            .segments
            .iter()
            .find(|s| {
                !s.closed
                    && s.lane_count >= 2
                    && s.length >= 75.0
                    && matches!(
                        world.network.junctions[s.to_junc].control_mode,
                        ControlMode::Signal2Phase | ControlMode::Signal4PhaseProtectedLeft
                    )
            })
            .map(|s| s.id)
            .expect("multi-lane signalized segment exists");

        let left_lid = world.network.segments[seg_id].lane_ids[0];
        world.vehicles.clear();
        world.base_demand_per_lane_hr = 0.0;
        world.transit_policy.couple_gravity_demand = false;

        // Place two slower civilian sedans on the left passing lane (`lane 0`) ahead of the emergency responder
        for k in 0..2 {
            world.vehicles.push(Vehicle {
                id: 700 + k,
                v_class: VehicleClass::Sedan,
                length_m: 4.5,
                width_m: 1.9,
                accel_mult: 1.0,
                v_max_mps: 8.5,
                carrier: CarrierRef::Lane(left_lid),
                s_m: 26.0 + (k as f64) * 14.0,
                v: 5.0,
                acc: 0.0,
                lat_visual_offset_m: 0.0,
                lc_cooldown_s: 0.0,
                dest_seg_id: None,
                planned_conn_id: None,
                turn_intent: TurnType::Through,
                spawn_time_s: 0.0,
                gtfs_route_idx: None,
                gtfs_route_seq: 0,
                gtfs_next_stop_order: 0,
                dwell_remaining_s: 0.0,
                onboard_pax: 0,
                crush_capacity: 0,
            });
        }

        let ev_id = world
            .dispatch_emergency_on_segment(Some(seg_id))
            .expect("emergency responder dispatched");

        let mut saw_preemption = false;
        let mut saw_civilian_pullover_or_right_lane = false;
        let mut arena = GpuStagingArena::new();

        for _ in 0..60 {
            world.step(0.1);
            if world.evp_telemetry.preempted_junctions >= 1 {
                saw_preemption = true;
            }
            for v in &world.vehicles {
                if v.id != ev_id {
                    let moved_right_lane = match v.carrier {
                        CarrierRef::Lane(lid) => world.network.lanes[lid].index > 0,
                        _ => false,
                    };
                    if moved_right_lane || v.lat_visual_offset_m >= 0.55 {
                        saw_civilian_pullover_or_right_lane = true;
                    }
                }
            }
        }

        pack_agents_into(&world, 0.0, &mut arena.agents);
        pack_road_quads_into(&world, &mut arena.road_quads);
        assert!(
            arena
                .agents
                .iter()
                .any(|a| (a.flags & 15) == AGENT_KIND_EMERGENCY),
            "EmergencyResponder must pack as AGENT_KIND_EMERGENCY (5u)"
        );
        assert!(
            saw_preemption,
            "Approaching EmergencyResponder must preempt downstream signalized junctions into green wave"
        );
        assert!(
            saw_civilian_pullover_or_right_lane,
            "Civilian sedans ahead of EmergencyResponder must yield right via MOBIL or shoulder pull-over"
        );
    }

    #[test]
    fn test_fire_station_and_hospital_placement_undo_and_4min_coverage_heatmap() {
        let xml = builtin_metropolis_50k_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);

        // Remove all default stations first to verify 0% baseline -> expansion when adding stations
        world.emergency_facilities.clear();
        world.refresh_urban_metrics();
        assert_eq!(world.emergency_coverage.fire_station_count, 0);
        assert_eq!(world.emergency_coverage.parcels_within_4min_pct, 0.0);

        let b = world.network.bounds;
        let fs1_id = world
            .place_emergency_facility_at(
                b.min_x + 0.25 * (b.max_x - b.min_x),
                b.min_y + 0.25 * (b.max_y - b.min_y),
                EmergencyFacilityKind::FireStation,
            )
            .expect("placed Fire Station #1");
        assert_eq!(fs1_id, 0);
        let cov_one = world.emergency_coverage.parcels_within_4min_pct;
        let p90_one = world.emergency_coverage.p90_fire_travel_s;
        assert!(cov_one > 0.0);

        // Place a second Fire Station in the opposite quadrant -> 4-min coverage rises & P90 drops!
        world
            .place_emergency_facility_at(
                b.min_x + 0.78 * (b.max_x - b.min_x),
                b.min_y + 0.78 * (b.max_y - b.min_y),
                EmergencyFacilityKind::FireStation,
            )
            .expect("placed Fire Station #2");
        assert!(world.emergency_coverage.parcels_within_4min_pct >= cov_one);
        assert!(world.emergency_coverage.p90_fire_travel_s < p90_one);

        // Place a Trauma Hospital and verify hospital transport time becomes finite
        world
            .place_emergency_facility_at(
                0.5 * (b.min_x + b.max_x),
                0.5 * (b.min_y + b.max_y),
                EmergencyFacilityKind::Hospital,
            )
            .expect("placed Trauma Hospital");
        assert_eq!(world.emergency_coverage.hospital_count, 1);
        assert!(world.emergency_coverage.mean_hospital_transport_s < 400.0);

        // Verify EVP Green-Wave enabled yields faster emergency coverage than un-preempted red delays
        let mean_with_evp = world.emergency_coverage.mean_fire_travel_s;
        world.evp_config.enabled = false;
        world.refresh_urban_metrics();
        let mean_without_evp = world.emergency_coverage.mean_fire_travel_s;
        assert!(mean_with_evp < mean_without_evp);
        world.evp_config.enabled = true;
        world.refresh_urban_metrics();

        // Verify 256x256 RGBA8 EmergencyResponse4Min GPU heatmap & facility pad quads
        let mut arena = GpuStagingArena::new();
        pack_road_quads_into(&world, &mut arena.road_quads);
        assert!(arena
            .road_quads
            .iter()
            .any(|q| q.style_kind == ROAD_STYLE_FIRE_STATION));
        assert!(arena
            .road_quads
            .iter()
            .any(|q| q.style_kind == ROAD_STYLE_HOSPITAL));

        build_heatmap_rgba(
            &world,
            HeatmapLayer::EmergencyResponse4Min,
            &arena.agents.clone(),
            &mut arena,
        );
        assert_eq!(arena.heatmap_rgba.len(), HEATMAP_BYTE_LEN);
        assert!(arena.heatmap_rgba.iter().any(|&v| v > 0));

        // Verify Ctrl+Z / Ctrl+Y undo & redo of Hospital placement
        let undone = world.undo_edit().expect("undo hospital placement");
        assert_eq!(undone.category, EditCategory::EmergencyFacility);
        assert_eq!(world.emergency_coverage.hospital_count, 0);
        world.redo_edit().expect("redo hospital placement");
        assert_eq!(world.emergency_coverage.hospital_count, 1);
    }

    #[test]
    fn test_mclp_station_siting_optimizer_reaches_90pct_nfpa1710_coverage_and_undoes() {
        // Use a large 12x12 Megacity supergrid with a tight 150s threshold so a single corner
        // station cannot cover 90% alone, requiring multi-station MCLP + GAS optimization!
        let xml = builtin_metropolis_50k_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);

        // Place 1 poorly sited corner Fire Station so initial coverage is well below 90%
        world
            .emergency_facilities
            .retain(|f| f.kind == EmergencyFacilityKind::Hospital);
        let b = world.network.bounds;
        world
            .place_emergency_facility_at(
                b.min_x + 25.0,
                b.min_y + 25.0,
                EmergencyFacilityKind::FireStation,
            )
            .expect("corner station placed");

        let cfg = MclpSitingConfig {
            target_coverage_pct: 90.0,
            threshold_travel_s: 150.0,
            max_stations_budget: 10,
            min_station_spacing_m: 140.0,
            keep_existing_fire_stations: false,
            weighting_mode: MclpWeightingMode::BothResidentAndParcel,
            ..MclpSitingConfig::default()
        };

        let report = world.solve_mclp_station_siting(&cfg);
        assert!(report.candidate_pool_size >= 15);
        assert!(
            report.target_met,
            "MCLP optimizer must find a station configuration reaching >= 90% coverage"
        );
        assert!(report.optimal_total_stations >= 1 && report.optimal_total_stations <= 8);
        assert!(report.achieved_parcel_cov_pct >= 90.0 - 1e-4);
        assert!(report.achieved_resident_cov_pct >= 90.0 - 1e-4);

        // Verify monotonic non-decreasing coverage along the Pareto curve
        for w in report.pareto_curve.windows(2) {
            assert!(
                w[1].parcels_within_threshold_pct + 1e-6 >= w[0].parcels_within_threshold_pct
                    || w[1].residents_within_threshold_pct + 1e-6
                        >= w[0].residents_within_threshold_pct
            );
        }

        // Apply with standard 240s NFPA 1710 threshold & verify live SimulationWorld coverage + Ctrl+Z undo
        let cfg_240 = MclpSitingConfig {
            target_coverage_pct: 90.0,
            threshold_travel_s: 240.0,
            ..cfg
        };
        let rep_240 = world.solve_mclp_station_siting(&cfg_240);
        assert!(rep_240.target_met);
        let prev_fs_count = world.emergency_coverage.fire_station_count;
        let applied_n = world.apply_mclp_siting_solution(&rep_240);
        assert_eq!(applied_n, rep_240.optimal_total_stations);
        assert!(world.emergency_coverage.parcels_within_4min_pct >= 90.0 - 1e-4);
        assert_eq!(world.emergency_coverage.hospital_count, 1); // Hospital preserved!

        // Undo restores exact prior station layout
        world.undo_edit().expect("undo MCLP siting");
        assert_eq!(world.emergency_coverage.fire_station_count, prev_fs_count);
    }

    #[test]
    fn test_arterial_bridge_closure_stress_test_and_robust_mclp_hardening() {
        // Build a synthetic two-island urban network linked by a central arterial bridge + longer
        // secondary bypass bridge so closing the central bridge stresses single-side MCLP siting!
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<osm version=\"0.6\">\n");
        // West bank 3x3 grid (lon -123.280 .. -123.272) and East bank 3x3 grid (lon -123.262 .. -123.254)
        let mut nid = 1usize;
        let mut west_nodes = [[0usize; 3]; 3];
        let mut east_nodes = [[0usize; 3]; 3];
        for r in 0..3 {
            let lat = 44.560 + (r as f64) * 0.0032;
            for c in 0..3 {
                let w_lon = -123.280 + (c as f64) * 0.0038;
                xml.push_str(&format!("  <node id=\"{nid}\" lat=\"{lat:.6}\" lon=\"{w_lon:.6}\"/>\n"));
                west_nodes[r][c] = nid;
                nid += 1;

                let e_lon = -123.260 + (c as f64) * 0.0038;
                xml.push_str(&format!("  <node id=\"{nid}\" lat=\"{lat:.6}\" lon=\"{e_lon:.6}\"/>\n"));
                east_nodes[r][c] = nid;
                nid += 1;
            }
        }
        let mut wid = 100usize;
        for bank in [&west_nodes, &east_nodes] {
            for r in 0..3 {
                for c in 0..2 {
                    let a = bank[r][c];
                    let b = bank[r][c + 1];
                    xml.push_str(&format!(
                        "  <way id=\"{wid}\"><nd ref=\"{a}\"/><nd ref=\"{b}\"/><tag k=\"highway\" v=\"secondary\"/><tag k=\"name\" v=\"Bank St {wid}\"/></way>\n"
                    ));
                    wid += 1;
                }
            }
            for c in 0..3 {
                for r in 0..2 {
                    let a = bank[r][c];
                    let b = bank[r + 1][c];
                    xml.push_str(&format!(
                        "  <way id=\"{wid}\"><nd ref=\"{a}\"/><nd ref=\"{b}\"/><tag k=\"highway\" v=\"secondary\"/><tag k=\"name\" v=\"Ave {wid}\"/></way>\n"
                    ));
                    wid += 1;
                }
            }
        }
        // Central fast arterial bridge connecting West(1,2) <-> East(1,0)
        xml.push_str(&format!(
            "  <way id=\"900\"><nd ref=\"{}\"/><nd ref=\"{}\"/><tag k=\"highway\" v=\"trunk\"/><tag k=\"lanes\" v=\"4\"/><tag k=\"maxspeed\" v=\"45 mph\"/><tag k=\"name\" v=\"Central River Bridge\"/></way>\n",
            west_nodes[1][2], east_nodes[1][0]
        ));
        // Secondary northern bridge connecting West(2,2) <-> East(2,0)
        xml.push_str(&format!(
            "  <way id=\"901\"><nd ref=\"{}\"/><nd ref=\"{}\"/><tag k=\"highway\" v=\"primary\"/><tag k=\"lanes\" v=\"2\"/><tag k=\"maxspeed\" v=\"25 mph\"/><tag k=\"name\" v=\"North Bypass Bridge\"/></way>\n",
            west_nodes[2][2], east_nodes[2][0]
        ));
        xml.push_str("</osm>\n");

        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);

        // Site Naive MCLP under a tight 135s threshold where 1 station near the Central River Bridge covers >= 90% intact
        let mclp_cfg = MclpSitingConfig {
            target_coverage_pct: 90.0,
            threshold_travel_s: 135.0,
            max_stations_budget: 6,
            min_station_spacing_m: 120.0,
            keep_existing_fire_stations: false,
            weighting_mode: MclpWeightingMode::BothResidentAndParcel,
            ..MclpSitingConfig::default()
        };
        let naive_rep = world.solve_mclp_station_siting(&mclp_cfg);
        assert!(naive_rep.target_met);
        world.apply_mclp_siting_solution(&naive_rep);

        let stress_cfg = ResilienceStressConfig {
            interdiction_depth: DisruptionInterdictionDepth::DualLinkNMinus2,
            bidirectional_closure: true,
            arterials_and_bridges_only: false,
            protect_station_host_blocks: true,
            max_candidate_corridors: 64,
            n_minus_2_top_pool: 10,
            target_coverage_pct: 90.0,
            threshold_travel_s: 135.0,
        };
        let naive_stress = world.run_resilience_stress_test(&stress_cfg);
        assert!(naive_stress.evaluated_scenarios >= 10);
        assert!(
            naive_stress.max_coverage_drop_pct_pts > 5.0
                || naive_stress.max_nri_delay_parcel_s > 50.0,
            "Bridge/corridor interdiction must detect coverage drop or NRI rerouting delay"
        );
        let csv = naive_stress.to_csv_string();
        assert!(csv.starts_with(RESILIENCE_CSV_HEADER));

        // Inject worst-case closure live & verify undo restores open network
        let worst = naive_stress.top_vulnerabilities[0].clone();
        let closed_n = world.set_segments_closed_tracked(
            &worst.closed_segment_ids,
            true,
            format!("Stress-test closure: {}", worst.label),
        );
        assert!(closed_n >= 1);
        assert_eq!(world.closed_segment_count(), closed_n);
        world.undo_edit().expect("undo bridge closure");
        assert_eq!(world.closed_segment_count(), 0);

        // Now solve Disruption-Hardened Robust MCLP and verify worst-case post-closure coverage improves!
        let robust_rep = world.solve_disruption_hardened_mclp(&mclp_cfg, &stress_cfg);
        assert!(robust_rep.target_met);
        world.apply_mclp_siting_solution(&robust_rep);
        let robust_stress = world.run_resilience_stress_test(&stress_cfg);
        assert!(
            robust_stress.worst_case_parcel_cov_pct + 1e-6 >= naive_stress.worst_case_parcel_cov_pct,
            "Disruption-hardened MCLP must maintain or improve worst-case post-closure coverage ({:.1}% vs {:.1}%)",
            robust_stress.worst_case_parcel_cov_pct,
            naive_stress.worst_case_parcel_cov_pct
        );
    }

    #[test]
    fn test_seismic_bridge_fragility_curves_and_monte_carlo_retrofit_crn() {
        // 1. Verify Abramowitz-Stegun normal CDF & Hazus lognormal fragility monotonicity + state conservation
        assert!((normal_cdf_phi(0.0) - 0.5).abs() < 1e-6);
        let arch = BridgeStructuralArchetype::Pre1975MultiSpanSimplySupported;
        let mut prev_ext = 0.0;
        for step in 1..=25 {
            let im_g = (step as f64) * 0.10;
            let ex = arch.exceedance_probs(im_g);
            assert!(ex[0] >= ex[1] && ex[1] >= ex[2] && ex[2] >= ex[3]);
            assert!(ex[2] >= prev_ext - 1e-9);
            prev_ext = ex[2];
            let st = arch.discrete_state_probs(im_g);
            let sum_p: f64 = st.iter().sum();
            assert!((sum_p - 1.0).abs() < 1e-6);
        }

        // 2. Build a two-bank city connected by two river bridges and site MCLP Fire Stations
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<osm version=\"0.6\">\n");
        let mut nid = 1usize;
        let mut west = [[0usize; 3]; 3];
        let mut east = [[0usize; 3]; 3];
        for r in 0..3 {
            let lat = 44.560 + (r as f64) * 0.0032;
            for c in 0..3 {
                xml.push_str(&format!(
                    "  <node id=\"{nid}\" lat=\"{lat:.6}\" lon=\"{:.6}\"/>\n",
                    -123.280 + (c as f64) * 0.0038
                ));
                west[r][c] = nid;
                nid += 1;
                xml.push_str(&format!(
                    "  <node id=\"{nid}\" lat=\"{lat:.6}\" lon=\"{:.6}\"/>\n",
                    -123.260 + (c as f64) * 0.0038
                ));
                east[r][c] = nid;
                nid += 1;
            }
        }
        let mut wid = 100usize;
        for bank in [&west, &east] {
            for r in 0..3 {
                for c in 0..2 {
                    xml.push_str(&format!(
                        "  <way id=\"{wid}\"><nd ref=\"{}\"/><nd ref=\"{}\"/><tag k=\"highway\" v=\"secondary\"/><tag k=\"name\" v=\"St {wid}\"/></way>\n",
                        bank[r][c], bank[r][c + 1]
                    ));
                    wid += 1;
                }
            }
            for c in 0..3 {
                for r in 0..2 {
                    xml.push_str(&format!(
                        "  <way id=\"{wid}\"><nd ref=\"{}\"/><nd ref=\"{}\"/><tag k=\"highway\" v=\"secondary\"/><tag k=\"name\" v=\"Ave {wid}\"/></way>\n",
                        bank[r][c], bank[r + 1][c]
                    ));
                    wid += 1;
                }
            }
        }
        xml.push_str(&format!(
            "  <way id=\"900\"><nd ref=\"{}\"/><nd ref=\"{}\"/><tag k=\"highway\" v=\"trunk\"/><tag k=\"lanes\" v=\"4\"/><tag k=\"name\" v=\"Van Buren River Bridge\"/></way>\n",
            west[1][2], east[1][0]
        ));
        xml.push_str(&format!(
            "  <way id=\"901\"><nd ref=\"{}\"/><nd ref=\"{}\"/><tag k=\"highway\" v=\"primary\"/><tag k=\"lanes\" v=\"2\"/><tag k=\"name\" v=\"Harrison River Bridge\"/></way>\n",
            west[2][2], east[2][0]
        ));
        xml.push_str("</osm>\n");

        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let mclp_rep = world.solve_mclp_station_siting(&MclpSitingConfig {
            target_coverage_pct: 90.0,
            threshold_travel_s: 160.0,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&mclp_rep);

        // 3. Compare unretrofitted vs. retrofitted bridges under identical CRN earthquake realizations!
        let base_cfg = SeismicScenarioConfig {
            moment_magnitude_mw: 7.6,
            rupture_standoff_km: 12.0,
            vs30_m_per_s: 240.0,
            monte_carlo_trials: 200,
            random_seed: 42_2026,
            retrofit_top_k_bridges: 0,
            retrofit_target_archetype: BridgeStructuralArchetype::EssentialBaseIsolatedLifeline,
            threshold_travel_s: 160.0,
            ..SeismicScenarioConfig::default()
        };
        let unretro_rep = world.run_seismic_monte_carlo(&base_cfg);
        assert!(unretro_rep.bridge_count >= 2);
        assert!(unretro_rep.expected_closed_bridges > 0.05);
        assert_eq!(unretro_rep.coverage_histogram_bins.iter().sum::<usize>(), 200);
        assert!(unretro_rep.to_csv_string().starts_with(SEISMIC_BRIDGE_CSV_HEADER));

        let retro_cfg = SeismicScenarioConfig {
            retrofit_top_k_bridges: unretro_rep.bridge_count,
            ..base_cfg
        };
        let retro_rep = world.run_seismic_monte_carlo(&retro_cfg);
        assert!(
            retro_rep.expected_closed_bridges <= unretro_rep.expected_closed_bridges + 1e-9,
            "CRN seismic retrofit must reduce expected bridge closures ({:.2} vs {:.2})",
            retro_rep.expected_closed_bridges,
            unretro_rep.expected_closed_bridges
        );
        assert!(
            retro_rep.mean_parcel_cov_pct + 1e-6 >= unretro_rep.mean_parcel_cov_pct,
            "CRN seismic retrofit must preserve or improve post-event mean coverage"
        );

        // 4. Inject worst seismic realization live and verify Ctrl+Z undo restores open bridges
        if !unretro_rep.worst_realization.closed_segment_ids.is_empty() {
            let n_c = world.inject_seismic_realization_tracked(
                &unretro_rep.worst_realization,
                "Inject Worst Seismic Realization",
            );
            assert!(n_c >= 1);
            world.undo_edit().expect("undo seismic realization");
            assert_eq!(world.closed_segment_count(), 0);
        }
    }

    #[test]
    fn test_hazus_bridge_restoration_functions_days_1_7_30_90_and_recovery_curves() {
        // 1. Verify exact Hazus damage-state restoration hierarchy across Days 1, 7, 30, and 90:
        //    - Slight (0.6 +- 0.6 d) recovers fastest (~>90% by Day 3)
        //    - Moderate (2.5 +- 2.7 d) recovers mostly by Day 7
        //    - Extensive (75 +- 42 d) recovers substantially between Day 30 and Day 90
        //    - Complete (230 +- 110 d) remains mostly unrecovered through Day 90
        for &d in &[1.0, 7.0, 30.0, 90.0] {
            let f_sl = SeismicDamageState::Slight.hazus_functionality_at_day(d, 1.0);
            let f_md = SeismicDamageState::Moderate.hazus_functionality_at_day(d, 1.0);
            let f_ex = SeismicDamageState::Extensive.hazus_functionality_at_day(d, 1.0);
            let f_cp = SeismicDamageState::Complete.hazus_functionality_at_day(d, 1.0);
            assert!(f_sl >= f_md - 1e-6 && f_md >= f_ex - 1e-6 && f_ex >= f_cp - 1e-6);
        }
        assert!(SeismicDamageState::Slight.hazus_functionality_at_day(7.0, 1.0) > 0.99);
        assert!(SeismicDamageState::Moderate.hazus_functionality_at_day(7.0, 1.0) > 0.92);
        assert!(
            SeismicDamageState::Extensive.hazus_functionality_at_day(90.0, 1.0)
                > SeismicDamageState::Extensive.hazus_functionality_at_day(7.0, 1.0) + 0.45
        );

        // 2. Run Monte Carlo recovery trajectory on a two-bank bridge network & verify monotonic recovery
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig::default());
        world.apply_mclp_siting_solution(&sited);

        let scfg = SeismicScenarioConfig {
            moment_magnitude_mw: 7.8,
            rupture_standoff_km: 10.0,
            vs30_m_per_s: 230.0,
            monte_carlo_trials: 160,
            random_seed: 77_2026,
            repair_crew_speedup: 1.0,
            ..SeismicScenarioConfig::default()
        };
        let rep_1x = world.run_seismic_monte_carlo(&scfg);
        assert_eq!(rep_1x.recovery_timeline.len(), RECOVERY_TIMELINE_DAYS.len());
        // Verify Days 1, 7, 30, and 90 are all explicitly present and flagged
        for &bd in &HAZUS_BENCHMARK_DAYS {
            assert!(rep_1x
                .recovery_timeline
                .iter()
                .any(|ep| (ep.day - bd).abs() < 1e-6 && ep.is_hazus_primary_benchmark));
        }
        // Coverage & bridge functionality must be monotonically non-decreasing from Day 0 -> Day 1 -> Day 7 -> Day 30 -> Day 90!
        for w in rep_1x.recovery_timeline.windows(2) {
            assert!(
                w[1].mean_parcel_cov_pct + 1e-6 >= w[0].mean_parcel_cov_pct,
                "Expected parcel coverage must recover monotonically ({:.2}% at {} vs {:.2}% at {})",
                w[1].mean_parcel_cov_pct,
                w[1].label,
                w[0].mean_parcel_cov_pct,
                w[0].label
            );
            assert!(
                w[1].mean_bridge_functionality_pct + 1e-6 >= w[0].mean_bridge_functionality_pct
            );
            assert!(w[1].expected_closed_bridges <= w[0].expected_closed_bridges + 1e-6);
        }
        assert!(
            rep_1x.bruneau_resilience_index_90d_pct > 0.0
                && rep_1x.bruneau_resilience_index_90d_pct <= 100.0
        );
        assert!(rep_1x
            .to_recovery_csv_string()
            .starts_with(RECOVERY_TIMELINE_CSV_HEADER));

        // 3. Accelerated emergency repair crews (2.0x) must improve or preserve the 90-day Bruneau Resilience Index under CRN!
        let rep_2x = world.run_seismic_monte_carlo(&SeismicScenarioConfig {
            repair_crew_speedup: 2.0,
            ..scfg
        });
        assert!(
            rep_2x.bruneau_resilience_index_90d_pct + 1e-6
                >= rep_1x.bruneau_resilience_index_90d_pct,
            "2.0x emergency repair crew speedup must raise or preserve Bruneau 90-day Resilience Index ({:.2}% vs {:.2}%)",
            rep_2x.bruneau_resilience_index_90d_pct,
            rep_1x.bruneau_resilience_index_90d_pct
        );
    }

    #[test]
    fn test_daskin_1983_mexclp_busy_probability_and_second_due_backup_coverage() {
        // 1. Verify Daskin (1983) 1 - q^k formula & marginal gain (1 - q)*q^(k-1) > 0
        let p1 = daskin_expected_availability_prob(1, 6, 0.30, false);
        let p2 = daskin_expected_availability_prob(2, 6, 0.30, false);
        let p3 = daskin_expected_availability_prob(3, 6, 0.30, false);
        assert!((p1 - 0.70).abs() < 1e-9);
        assert!((p2 - 0.91).abs() < 1e-9);
        assert!((p3 - 0.973).abs() < 1e-9);

        // 2. Compare Deterministic MCLP (q=0) vs. Daskin MEXCLP + Hogan-ReVelle Second-Due Backup (q=0.30)
        //    on the Riverfront network: when apparatus are busy 30% of the time, single-station binary 100% coverage
        //    only yields 70% expected coverage, so MEXCLP deploys overlapping backup apparatus to reach >= 90%!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let world = SimulationWorld::new(net);

        let det_cfg = MclpSitingConfig {
            objective_model: SitingObjectiveModel::DeterministicMclp1974,
            apparatus_busy_prob: 0.30,
            target_coverage_pct: 90.0,
            threshold_travel_s: 240.0,
            max_stations_budget: 10,
            ..MclpSitingConfig::default()
        };
        let det_rep = world.solve_mclp_station_siting(&det_cfg);
        assert!(det_rep.target_met);

        let mexclp_cfg = MclpSitingConfig {
            objective_model: SitingObjectiveModel::MexclpPlusHoganBackup1986,
            busyness_model: BusynessModel::FixedBusyFraction,
            apparatus_busy_prob: 0.30,
            target_coverage_pct: 90.0,
            target_second_due_cov_pct: 80.0,
            threshold_travel_s: 240.0,
            second_due_threshold_s: 260.0,
            max_engines_per_station: 2,
            max_stations_budget: 10,
            ..MclpSitingConfig::default()
        };
        let mex_rep = world.solve_mclp_station_siting(&mexclp_cfg);
        assert!(
            mex_rep.target_met,
            "MEXCLP + Backup must reach >= 90% expected coverage & >= 80% second-due backup"
        );
        assert!(mex_rep.optimal_total_apparatus > det_rep.optimal_total_apparatus);
        assert!(mex_rep.achieved_expected_cov_pct >= 90.0 - 1e-4);
        assert!(mex_rep.achieved_second_due_cov_pct >= 80.0 - 1e-4);
        assert!(
            mex_rep.achieved_expected_travel_s < det_rep.achieved_expected_travel_s,
            "Overlapping MEXCLP backup units must reduce queueing-weighted expected response time ({:.1}s vs {:.1}s)",
            mex_rep.achieved_expected_travel_s,
            det_rep.achieved_expected_travel_s
        );
        assert!(mex_rep.to_csv_string().starts_with(MEXCLP_PARETO_CSV_HEADER));

        // 3. Verify Endogenous Workload Queue mode (`q(M) = lambda*tau / M`) shrinks `q` monotonically as fleet grows
        let endog_cfg = MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            busyness_model: BusynessModel::EndogenousWorkloadQueue,
            system_calls_per_hr: 4.5,
            mean_service_duration_mins: 40.0,
            max_stations_budget: 8,
            ..mexclp_cfg
        };
        let endog_rep = world.solve_mclp_station_siting(&endog_cfg);
        for w in endog_rep.pareto_curve.iter().skip(1).collect::<Vec<_>>().windows(2) {
            assert!(w[1].effective_busy_prob_q <= w[0].effective_busy_prob_q + 1e-9);
            assert!(w[1].expected_cov_pct + 1e-6 >= w[0].expected_cov_pct);
        }
    }

    #[test]
    fn test_larson_1974_exact_mmn_hypercube_conservation_and_heterogeneous_workloads() {
        // 1. Construct a 4-unit spatial emergency system with asymmetric downtown vs. suburban demand:
        //    Atom 0 (Downtown core, weight 65) is closest to Unit 0 -> Unit 1 -> Unit 2 -> Unit 3
        //    Atom 1 (Midtown,       weight 20) is closest to Unit 1 -> Unit 0 -> Unit 2 -> Unit 3
        //    Atom 2 (East suburb,   weight 10) is closest to Unit 2 -> Unit 1 -> Unit 3 -> Unit 0
        //    Atom 3 (Outer fringe,  weight  5) is closest to Unit 3 -> Unit 2 -> Unit 1 -> Unit 0
        let meta = vec![
            ("Engine 1 (Downtown)".to_string(), 0, Vec2 { x: 0.0, y: 0.0 }),
            ("Engine 2 (Midtown)".to_string(), 1, Vec2 { x: 100.0, y: 0.0 }),
            ("Engine 3 (Suburb)".to_string(), 2, Vec2 { x: 250.0, y: 0.0 }),
            ("Engine 4 (Fringe)".to_string(), 3, Vec2 { x: 450.0, y: 0.0 }),
        ];
        let times_s = vec![
            vec![75.0, 145.0, 290.0, 410.0],
            vec![135.0, 80.0, 210.0, 330.0],
            vec![265.0, 195.0, 95.0, 220.0],
            vec![390.0, 310.0, 215.0, 105.0],
        ];
        let weights = vec![65.0, 20.0, 10.0, 5.0];

        // A. Exact M/M/4/inf FCFS Queue (homogeneous mu = 2.0 calls/hr, Lambda = 3.2 calls/hr -> sum(rho_n) == 1.60 exact!)
        let fcfs_cfg = HypercubeConfig {
            system_calls_per_hr: 3.2,
            base_service_duration_mins: 30.0, // mu = 2.0/hr
            calibrate_travel_service_rate: false,
            queue_discipline: HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue,
            tolerance: 1e-12,
            ..HypercubeConfig::default()
        };
        let fcfs_rep = solve_larson_hypercube_from_matrix(&meta, &times_s, &weights, &fcfs_cfg);
        assert!(fcfs_rep.used_exact_2n_ctmc);
        assert_eq!(fcfs_rep.state_space_size, 16);
        assert!(
            fcfs_rep.rate_conservation_error < 1e-8,
            "M/M/N/inf throughput conservation error must vanish (got {:e})",
            fcfs_rep.rate_conservation_error
        );
        let sum_rho: f64 = fcfs_rep.unit_metrics.iter().map(|u| u.busy_fraction_rho).sum();
        assert!((sum_rho - 1.60).abs() < 1e-8);
        // Downtown Engine 1 must be strictly busier than Outer Fringe Engine 4!
        assert!(
            fcfs_rep.unit_metrics[0].busy_fraction_rho
                > fcfs_rep.unit_metrics[3].busy_fraction_rho + 0.15,
            "Asymmetric spatial demand must produce heterogeneous per-unit busy fractions ({:.3} vs {:.3})",
            fcfs_rep.unit_metrics[0].busy_fraction_rho,
            fcfs_rep.unit_metrics[3].busy_fraction_rho
        );
        assert!(fcfs_rep.cross_district_dispatch_pct > 5.0);
        assert!(fcfs_rep.to_csv_string().starts_with(HYPERCUBE_UNIT_CSV_HEADER));

        // B. Exact M/M/4/4 Mutual-Aid Loss + Travel-Time Calibrated Heterogeneous `mu_n`
        let loss_cfg = HypercubeConfig {
            calibrate_travel_service_rate: true,
            queue_discipline: HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss,
            ..fcfs_cfg
        };
        let loss_rep = solve_larson_hypercube_from_matrix(&meta, &times_s, &weights, &loss_cfg);
        assert!(
            loss_rep.rate_conservation_error < 1e-8,
            "Heterogeneous-mu M/M/N/N loss conservation error must vanish (got {:e})",
            loss_rep.rate_conservation_error
        );

        // 2. Verify live `SimulationWorld` hypercube integration & `BusynessModel::LarsonHypercube1974` inside MEXCLP!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let hc_siting_cfg = MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            busyness_model: BusynessModel::LarsonHypercube1974,
            system_calls_per_hr: 3.6,
            mean_service_duration_mins: 36.0,
            target_coverage_pct: 88.0,
            max_stations_budget: 6,
            ..MclpSitingConfig::default()
        };
        let sited = world.solve_mclp_station_siting(&hc_siting_cfg);
        assert!(sited.optimal_total_apparatus >= 2);
        world.apply_mclp_siting_solution(&sited);

        let live_hc = world.solve_active_fleet_hypercube(&HypercubeConfig::default());
        assert_eq!(live_hc.num_units, sited.optimal_total_apparatus);
        assert!(live_hc.rate_conservation_error < 1e-7);
    }

    #[test]
    fn test_carter_chaiken_ignall_1972_and_larson_hypercube_redistricting() {
        // 1. Construct a 12-atom linear corridor served by 3 Fire Engines where heavy downtown
        //    demand clusters near Engine 1 while lighter suburban demand surrounds Engines 2 & 3:
        //    Under naive Voronoi (`s = 0`), Engine 1 is heavily overloaded (`rho_0 >> rho_2`),
        //    causing frequent cross-district dispatches and dominated travel time!
        let meta = vec![
            ("Engine 1 (Core)".to_string(), 0, Vec2 { x: 0.0, y: 0.0 }),
            ("Engine 2 (Midtown)".to_string(), 1, Vec2 { x: 180.0, y: 0.0 }),
            ("Engine 3 (Suburb)".to_string(), 2, Vec2 { x: 380.0, y: 0.0 }),
        ];
        let atom_xs = [
            -20.0_f64, 15.0, 45.0, 72.0, 86.0, 105.0, 145.0, 195.0, 245.0, 278.0, 325.0, 400.0,
        ];
        // Travel time `t_{u, j} = 40 + |x_u - x_j| * 0.85` seconds
        let unit_xs = [0.0_f64, 180.0, 380.0];
        let times_s: Vec<Vec<f64>> = unit_xs
            .iter()
            .map(|&ux| atom_xs.iter().map(|&ax| 40.0 + (ux - ax).abs() * 0.85).collect())
            .collect();
        let weights = vec![34.0, 28.0, 22.0, 16.0, 14.0, 9.0, 6.0, 5.0, 4.0, 4.0, 3.0, 2.0];

        let hc_cfg = HypercubeConfig {
            system_calls_per_hr: 3.3,
            base_service_duration_mins: 30.0, // mu = 2.0/hr -> mean rho = 3.3 / 6.0 = 0.55
            calibrate_travel_service_rate: false,
            queue_discipline: HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue,
            tolerance: 1e-11,
            ..HypercubeConfig::default()
        };

        // A. Carter, Chaiken & Ignall (1972) Pure Travel-Time Minimization (`equity_weight = 0`):
        //    Shrinking overloaded Engine 1's boundary (`s_0 > 0`) keeps Engine 1 available for dense core atoms,
        //    achieving `E[T] <= E[T_Voronoi]` while simultaneously lowering workload imbalance!
        let carter_cfg = RedistrictingConfig {
            hypercube: hc_cfg.clone(),
            objective_mode: RedistrictingObjectiveMode::TravelTimeOptimalCarter1972,
            ..RedistrictingConfig::default()
        };
        let carter_rep = optimize_carter_larson_districts_from_matrix(
            &meta,
            &times_s,
            &weights,
            None,
            &carter_cfg,
        );
        assert!(
            carter_rep.optimized_hypercube.mean_dispatch_travel_s
                <= carter_rep.voronoi_hypercube.mean_dispatch_travel_s + 1e-8,
            "Carter-1972 optimal boundary must beat or match Voronoi E[T] ({:.3}s vs {:.3}s)",
            carter_rep.optimized_hypercube.mean_dispatch_travel_s,
            carter_rep.voronoi_hypercube.mean_dispatch_travel_s
        );

        // Verify strict Carter-1972 travel-time improvement (> 1.0s faster than Voronoi on asymmetric corridor!)
        assert!(
            carter_rep.carter_min_travel_gain_s > 1.0,
            "Carter-1972 optimal boundary must strictly beat naive Voronoi E[T] by > 1.0s (got {:.2}s)",
            carter_rep.carter_min_travel_gain_s
        );

        // B. Balanced Pareto Knee (`Lambda = 2.1/hr`, `equity_weight = 0.80`, `max_travel_penalty_s = 65.0s`):
        //    verify zero-mean offsets (`sum s_n == 0`), `s_0 > 0 > s_2`, > 45% workload spread reduction,
        //    and exact hypercube rate conservation!
        let bal_cfg = RedistrictingConfig {
            hypercube: HypercubeConfig {
                system_calls_per_hr: 2.1,
                ..hc_cfg
            },
            objective_mode: RedistrictingObjectiveMode::BalancedParetoKnee,
            workload_equity_weight: 0.92,
            max_travel_penalty_s: 65.0,
            ..RedistrictingConfig::default()
        };
        let bal_rep = optimize_carter_larson_districts_from_matrix(
            &meta,
            &times_s,
            &weights,
            None,
            &bal_cfg,
        );
        let offset_sum: f64 = bal_rep.unit_offsets_s.iter().sum();
        assert!(offset_sum.abs() < 1e-8, "Boundary offsets must sum to zero");
        assert!(
            bal_rep.unit_offsets_s[0] > 5.0 && bal_rep.unit_offsets_s[2] < -5.0,
            "Overloaded core Engine 1 must get positive boundary penalty s_0 > 0 and peripheral Engine 3 s_2 < 0 ({:?})",
            bal_rep.unit_offsets_s
        );
        assert!(bal_rep.reassigned_parcel_count >= 1);
        assert!(
            bal_rep.optimized_workload_spread < bal_rep.voronoi_workload_spread * 0.60,
            "Carter-Larson redistricting must substantially compress workload imbalance ({:.3} -> {:.3})",
            bal_rep.voronoi_workload_spread,
            bal_rep.optimized_workload_spread
        );
        assert!(bal_rep.optimized_hypercube.rate_conservation_error < 1e-8);
        assert_eq!(bal_rep.pareto_frontier.len(), 7);
        assert!(bal_rep.to_unit_csv_string().starts_with(REDISTRICT_UNIT_CSV_HEADER));
        assert!(bal_rep.to_pareto_csv_string().starts_with(REDISTRICT_PARETO_CSV_HEADER));

        // 2. Full `SimulationWorld` Riverfront network redistricting + `HeatmapLayer::ResponseDistricts` + `Ctrl+Z` undo/redo!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        assert!(sited.optimal_total_apparatus >= 3);
        world.apply_mclp_siting_solution(&sited);

        let net_rep = world.optimize_response_districts(&RedistrictingConfig {
            objective_mode: RedistrictingObjectiveMode::StrictWorkloadEqualization,
            max_travel_penalty_s: 35.0,
            ..RedistrictingConfig::default()
        });
        assert!(
            net_rep.optimized_workload_spread <= net_rep.voronoi_workload_spread + 1e-9,
            "Network redistricting must equalize or preserve station workloads"
        );
        let _shifted = world.apply_redistricting_plan_tracked(&net_rep);
        assert!(world.parcels.iter().all(|p| p.response_district_unit.is_some()));

        // Verify GPU ResponseDistricts heatmap rasterizes categorical districts & boundary seams cleanly
        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::ResponseDistricts, &[], &mut arena);
        let non_zero_px = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(non_zero_px, HEATMAP_DIM * HEATMAP_DIM);

        // Verify `Ctrl+Z` undo restores `None` (Voronoi) and `Ctrl+Y` redo restores optimized beats!
        world.undo_edit().expect("undo redistricting");
        assert!(world.parcels.iter().all(|p| p.response_district_unit.is_none()));
        world.redo_edit().expect("redo redistricting");
        assert!(world.parcels.iter().all(|p| p.response_district_unit.is_some()));
    }

    #[test]
    fn test_diurnal_24hr_call_rush_profiles_and_peak_offpeak_dispatch_tables() {
        // 1. Verify exact 24-hour USFA call multiplier conservation (`sum == 24.0`) & bimodal peaks
        let sum_24: f64 = (0..24).map(|h| diurnal_call_multiplier(h, 1.0)).sum();
        assert!(
            (sum_24 - 24.0).abs() < 1e-9,
            "24-hour diurnal call multipliers must have exact mean 1.0 (sum={sum_24})"
        );
        assert!(diurnal_call_multiplier(17, 1.0) > diurnal_call_multiplier(3, 1.0) * 4.0);

        // 2. Verify BPR rush-hour travel-time inflation (`17:00 PM Rush > 08:00 AM Rush >> 03:00 Overnight`)
        let m_03 = bpr_travel_time_multiplier(diurnal_arterial_vc_ratio(3, 1.18), 0.24, 4.0);
        let m_08 = bpr_travel_time_multiplier(diurnal_arterial_vc_ratio(8, 1.18), 0.24, 4.0);
        let m_17 = bpr_travel_time_multiplier(diurnal_arterial_vc_ratio(17, 1.18), 0.24, 4.0);
        assert!(m_17 > m_08 && m_08 > m_03 + 0.50);

        // 3. Solve separate Peak vs. Off-Peak dispatch tables on the Riverfront network & verify distinct tables + PSA!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        assert!(sited.optimal_total_apparatus >= 3);
        world.apply_mclp_siting_solution(&sited);

        let dcfg = DiurnalProfileConfig {
            window_preset: DiurnalWindowPreset::DaytimeCommercial12Hr,
            daily_mean_calls_per_hr: 3.8,
            call_profile_amplitude: 1.0,
            rush_vc_scale: 1.30,
            bpr_alpha: 0.28,
            bpr_beta: 4.0,
            redistrict: RedistrictingConfig {
                objective_mode: RedistrictingObjectiveMode::StrictWorkloadEqualization,
                workload_equity_weight: 0.92,
                max_travel_penalty_s: 45.0,
                ..RedistrictingConfig::default()
            },
        };
        let drep = world.solve_diurnal_dispatch_tables(&dcfg);
        assert!(drep.num_units >= 3);
        assert_eq!(drep.hourly_trajectory.len(), 24);
        assert_eq!(drep.peak_table.active_hours.len(), 12);
        assert_eq!(drep.offpeak_table.active_hours.len(), 12);
        assert!(
            drep.peak_table.mean_calls_per_hr > drep.offpeak_table.mean_calls_per_hr * 1.35,
            "Peak window call rate must substantially exceed Off-Peak ({:.2} vs {:.2})",
            drep.peak_table.mean_calls_per_hr,
            drep.offpeak_table.mean_calls_per_hr
        );
        assert!(
            drep.peak_table.mean_bpr_slowdown_pct > drep.offpeak_table.mean_bpr_slowdown_pct + 15.0,
            "Peak rush BPR arterial slowdown must exceed Off-Peak ({:.1}% vs {:.1}%)",
            drep.peak_table.mean_bpr_slowdown_pct,
            drep.offpeak_table.mean_bpr_slowdown_pct
        );

        // Every hour `0..24` in the Green-Kolesar PSA trajectory must satisfy exact hypercube rate conservation!
        for hm in &drep.hourly_trajectory {
            assert!(
                hm.rate_conservation_error < 1e-7,
                "Hour {:02}:00 hypercube conservation error too large ({:e})",
                hm.hour,
                hm.rate_conservation_error
            );
        }
        assert!(
            drep.daily_dual_table_workload_spread <= drep.daily_voronoi_workload_spread + 1e-8,
            "24h Dual Peak/Off-Peak tables must improve or preserve daily workload balance ({:.4} vs {:.4})",
            drep.daily_dual_table_workload_spread,
            drep.daily_voronoi_workload_spread
        );

        // Verify Peak vs. Off-Peak tables differ in offsets or run-card assignments due to commuter shift + BPR congestion!
        let offset_l1_diff: f64 = drep
            .peak_table
            .unit_offsets_s
            .iter()
            .zip(drep.offpeak_table.unit_offsets_s.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            offset_l1_diff > 1.0 || drep.any_rank12_switches_count >= 1,
            "Peak and Off-Peak optimal dispatch tables must differentiate under rush-hour BPR + commuter shift"
        );
        assert!(drep.to_hourly_csv_string().starts_with(DIURNAL_HOURLY_CSV_HEADER));
        assert!(drep
            .to_dispatch_table_csv_string()
            .starts_with(DIURNAL_DISPATCH_TABLE_CSV_HEADER));

        // 4. Verify live Peak/Off-Peak table application, regime toggling, and `Ctrl+Z` undo/redo!
        let _sw = world.apply_diurnal_dispatch_tables_tracked(&drep, DiurnalDispatchRegime::PeakRush);
        assert!(world
            .parcels
            .iter()
            .all(|p| p.peak_district_unit.is_some() && p.offpeak_district_unit.is_some()));
        let _upd = world.activate_diurnal_regime_tracked(DiurnalDispatchRegime::OffPeakFreeFlow);
        for p in &world.parcels {
            assert_eq!(p.response_district_unit, p.offpeak_district_unit);
        }
        world.undo_edit().expect("undo regime switch back to Peak");
        for p in &world.parcels {
            assert_eq!(p.response_district_unit, p.peak_district_unit);
        }
        world.undo_edit().expect("undo diurnal table installation");
        assert!(world.parcels.iter().all(|p| p.peak_district_unit.is_none()));
    }

    #[test]
    fn test_coupled_electrical_water_wastewater_grids_and_cascading_outages() {
        // 1. Verify Hazen-Williams & Manning hydraulic laws analytically
        let hf_low = hazen_williams_head_loss_m(200.0, 0.030, 125.0, 0.25);
        let hf_high = hazen_williams_head_loss_m(200.0, 0.060, 125.0, 0.25);
        // Doubling flow Q increases Hazen-Williams head loss by `2^1.852 == 3.61x`!
        assert!(((hf_high / hf_low) - 2.0f64.powf(1.852)).abs() < 1e-4);

        let q_full = manning_full_pipe_capacity_m3_s(0.45, 0.013, 0.005);
        assert!(q_full > 0.15 && q_full < 0.40);
        assert!((manning_circular_pipe_d_over_d(0.5) - 0.5).abs() < 1e-9);
        assert!(manning_circular_pipe_d_over_d(1.25) > 1.0);

        // 2. Nominal Coupled 3-Grid Operation on the Riverfront network (`12:30` midday solar + dry weather)
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        world.diurnal_clock_hour = 12.5;

        let nom_cfg = UtilityGridsConfig {
            solar_pv_penetration_pct: 30.0,
            storm_rainfall_mm_hr: 2.0,
            active_fire_flow_demand_gpm: 1_500.0,
            ..UtilityGridsConfig::default()
        };
        let nom_rep = world.evaluate_utility_grids(&nom_cfg);
        assert_eq!(nom_rep.electrical.energized_nodes_pct, 100.0);
        assert!(nom_rep.electrical.solar_pv_generation_mw > 0.01);
        assert!(nom_rep.electrical.ansi_c84_1_compliance_pct >= 90.0);
        assert!(nom_rep.water.booster_pumps_energized);
        assert!(
            nom_rep.water.min_residual_pressure_psi >= MIN_FIRE_FLOW_RESIDUAL_PSI,
            "Nominal booster head must maintain >= 20 psi residual fire-hydrant pressure (got {:.1} psi)",
            nom_rep.water.min_residual_pressure_psi
        );
        assert_eq!(nom_rep.wastewater.surcharged_sso_nodes_count, 0);
        assert!(nom_rep.to_csv_string().starts_with(UTILITY_NODE_CSV_HEADER));

        // 3. Extreme Wet-Weather Cloudburst (`65 mm/hr`): verify EPA RDII surges WWTP influent & sewer d/D!
        let storm_rep = world.evaluate_utility_grids(&UtilityGridsConfig {
            storm_rainfall_mm_hr: 65.0,
            ..nom_cfg.clone()
        });
        assert!(
            storm_rep.wastewater.wet_weather_rdii_lps
                > nom_rep.wastewater.wet_weather_rdii_lps * 20.0
        );
        assert!(
            storm_rep.wastewater.total_wwtp_influent_lps
                > nom_rep.wastewater.total_wwtp_influent_lps * 1.5
        );
        assert!(storm_rep.wastewater.max_pipe_d_over_d > nom_rep.wastewater.max_pipe_d_over_d);

        // 4. Rinaldi et al. (2001) Total Electrical Blackout Cascade (`tripped = [0, 1]`, no backup)
        //    vs. Critical Microgrid / Diesel Backup Recovery!
        let blackout_cfg = UtilityGridsConfig {
            tripped_substation_indices: vec![0, 1],
            critical_backup_generators_enabled: false,
            ..nom_cfg.clone()
        };
        let blk_rep = world.evaluate_utility_grids(&blackout_cfg);
        assert_eq!(blk_rep.electrical.energized_nodes_pct, 0.0);
        assert!(!blk_rep.water.booster_pumps_energized);
        assert!(!blk_rep.wastewater.wwtp_energized);
        assert!(
            blk_rep.water.mean_residual_pressure_psi
                < nom_rep.water.mean_residual_pressure_psi - 30.0,
            "Loss of electric booster pumps must drop municipal water pressure ({:.1} vs {:.1} psi)",
            blk_rep.water.mean_residual_pressure_psi,
            nom_rep.water.mean_residual_pressure_psi
        );
        assert!(blk_rep.wastewater.surcharged_sso_nodes_count >= 1);
        assert!(blk_rep.electrical.darkened_signal_junctions_count >= 1);

        // Test undoable signal darkening cascade (`Signal -> 4-Way PermissiveYield`, then `Ctrl+Z`!)
        let degraded = world.apply_darkened_signal_fallback_tracked();
        assert!(degraded >= 1);
        world.undo_edit().expect("undo darkened signal cascade");

        // Enabling Critical Backup Generators restores Water Booster Pressure, Lift/WWTP & Signals during grid blackout!
        let bkp_rep = world.evaluate_utility_grids(&UtilityGridsConfig {
            critical_backup_generators_enabled: true,
            ..blackout_cfg
        });
        assert!(bkp_rep.water.booster_pumps_energized);
        assert!(bkp_rep.wastewater.wwtp_energized);
        assert_eq!(bkp_rep.electrical.darkened_signal_junctions_count, 0);
        assert!(
            bkp_rep.water.min_residual_pressure_psi >= MIN_FIRE_FLOW_RESIDUAL_PSI,
            "Backup generators on water booster stations must restore >= 20 psi fire-hydrant pressure!"
        );

        // Verify GPU UtilityLifelines heatmap rasterizes cleanly
        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::UtilityLifelines, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_ala_2001_hazus_pgv_pipeline_fragility_and_multilifeline_seismic_restoration() {
        // 1. Verify ALA (2001) & Hazus O'Rourke-Ayala (1993) PGV repair-rate hierarchy across pipe materials
        for &law in &[
            PipelineFragilityLaw::Ala2001LinearPgv,
            PipelineFragilityLaw::HazusORourkeAyala1993,
        ] {
            let rr_ci = ala_hazus_pipe_repair_rate_per_km(
                65.0,
                law,
                BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                200.0,
                false,
            );
            let rr_di = ala_hazus_pipe_repair_rate_per_km(
                65.0,
                law,
                BuriedPipeMaterialClass::SegmentedDuctileIronPvc,
                200.0,
                false,
            );
            let rr_st = ala_hazus_pipe_repair_rate_per_km(
                65.0,
                law,
                BuriedPipeMaterialClass::WeldedSteelRestrainedJoint,
                200.0,
                false,
            );
            let rr_er = ala_hazus_pipe_repair_rate_per_km(
                65.0,
                law,
                BuriedPipeMaterialClass::SeismicErdipHdpe,
                200.0,
                false,
            );
            assert!(rr_ci > rr_di && rr_di > rr_st && rr_st > rr_er && rr_er > 0.0);
            // ERDIP/HDPE (`K_1 = 0.08`) cuts repair rate by exactly `92%` relative to Brittle Cast Iron (`K_1 = 1.0`)!
            assert!(((rr_er / rr_ci) - 0.08).abs() < 1e-6);
        }

        // 2. Run Multi-Lifeline Seismic Monte Carlo on the Riverfront network under Mw 7.7 subduction/crustal event:
        //    compare Brittle Cast Iron / VCP (`0%` replaced) vs. `100%` Seismic ERDIP/HDPE under CRN!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let unretro_cfg = SeismicScenarioConfig {
            moment_magnitude_mw: 7.7,
            rupture_standoff_km: 10.0,
            vs30_m_per_s: 220.0,
            monte_carlo_trials: 180,
            random_seed: 91_2026,
            pipe_fragility_law: PipelineFragilityLaw::HazusORourkeAyala1993,
            baseline_water_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
            baseline_sewer_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
            seismic_pipe_replacement_pct: 0.0,
            water_sewer_repair_crews: 4,
            ..SeismicScenarioConfig::default()
        };
        let unretro = world.run_seismic_monte_carlo(&unretro_cfg);
        assert!(unretro.total_water_pipe_km > 0.5);
        assert!(unretro.expected_day0_water_leaks > unretro.expected_day0_water_breaks);
        assert!(unretro.expected_day0_water_breaks > 0.05);
        assert!(unretro.expected_day0_sewer_breaks > 0.05);
        assert!(unretro
            .to_pipeline_csv_string()
            .starts_with(SEISMIC_PIPELINE_CSV_HEADER));

        // Verify monotonic multi-lifeline restoration across Days 0 -> 1 -> 3 -> 7 -> 14 -> 30 -> 60 -> 90!
        for w in unretro.recovery_timeline.windows(2) {
            assert!(
                w[1].mean_electric_service_pct + 1e-6 >= w[0].mean_electric_service_pct,
                "Electric service must recover monotonically ({:.2}% vs {:.2}%)",
                w[1].mean_electric_service_pct,
                w[0].mean_electric_service_pct
            );
            assert!(
                w[1].mean_water_20psi_service_pct + 1e-6 >= w[0].mean_water_20psi_service_pct,
                "Potable water >=20psi service must recover monotonically ({:.2}% vs {:.2}%)",
                w[1].mean_water_20psi_service_pct,
                w[0].mean_water_20psi_service_pct
            );
            assert!(
                w[1].mean_wastewater_service_pct + 1e-6 >= w[0].mean_wastewater_service_pct,
                "Wastewater free-flow service must recover monotonically ({:.2}% vs {:.2}%)",
                w[1].mean_wastewater_service_pct,
                w[0].mean_wastewater_service_pct
            );
            assert!(
                w[1].expected_water_repairs_remaining
                    <= w[0].expected_water_repairs_remaining + 1e-9
            );
        }

        // 3. Replace 100% of brittle pipes with `SeismicErdipHdpe` + Backup Lifeline Power under CRN!
        let erdip = world.run_seismic_monte_carlo(&SeismicScenarioConfig {
            seismic_pipe_replacement_pct: 100.0,
            critical_lifeline_backup_power: true,
            ..unretro_cfg
        });
        assert!(
            erdip.expected_day0_water_breaks < unretro.expected_day0_water_breaks * 0.20,
            "ERDIP/HDPE replacement must slash expected water main breaks by >80% ({:.3} vs {:.3})",
            erdip.expected_day0_water_breaks,
            unretro.expected_day0_water_breaks
        );
        assert!(
            erdip.expected_day0_sewer_breaks < unretro.expected_day0_sewer_breaks * 0.20,
            "ERDIP/HDPE replacement must slash expected sewer main breaks by >80% ({:.3} vs {:.3})",
            erdip.expected_day0_sewer_breaks,
            unretro.expected_day0_sewer_breaks
        );
        assert!(
            erdip.water_resilience_index_90d_pct + 1e-6 >= unretro.water_resilience_index_90d_pct
        );
        assert!(
            erdip.wastewater_resilience_index_90d_pct + 1e-6
                >= unretro.wastewater_resilience_index_90d_pct
        );
    }

    #[test]
    fn test_scawthorn_hazus_ffe_ignition_hamada_wind_spread_and_hydrant_coupling() {
        // 1. Verify Scawthorn (2005) & FEMA Hazus Ch. 10 PGA ignition monotonicity & AGSV gas shutoff
        assert_eq!(
            scawthorn_hazus_ignition_rate_per_mmsqft(
                0.03,
                ZoneCode::R3,
                FfeIgnitionModel::Scawthorn2005QuadraticPga,
                0.0
            ),
            0.0
        );
        let ig_03 = scawthorn_hazus_ignition_rate_per_mmsqft(
            0.30,
            ZoneCode::R3,
            FfeIgnitionModel::Scawthorn2005QuadraticPga,
            0.0,
        );
        let ig_07 = scawthorn_hazus_ignition_rate_per_mmsqft(
            0.70,
            ZoneCode::R3,
            FfeIgnitionModel::Scawthorn2005QuadraticPga,
            0.0,
        );
        let ig_07_agsv = scawthorn_hazus_ignition_rate_per_mmsqft(
            0.70,
            ZoneCode::R3,
            FfeIgnitionModel::Scawthorn2005QuadraticPga,
            100.0,
        );
        assert!(ig_07 > ig_03 && ig_03 > 0.02);
        assert!(((ig_07_agsv / ig_07) - 0.52).abs() < 1e-6);

        // 2. Verify Hamada (1951) / FEMA (2020) Eqs. (8)-(10):
        //    - At `U = 0 m/s`, downwind == crosswind == upwind (exact isotropic circle!)
        //    - At `U = 12 m/s`, `v_down > v_cross > v_up` and wood-frame `R3` > fire-resistive `CBD`!
        let calm = hamada_hazus_spread_rates(ZoneCode::R3, 0.0);
        assert!((calm.v_down_m_min - calm.v_cross_m_min).abs() < 1e-9);
        assert!((calm.v_down_m_min - calm.v_up_m_min).abs() < 1e-9);

        let windy_wood = hamada_hazus_spread_rates(ZoneCode::R3, 12.0);
        let windy_cbd = hamada_hazus_spread_rates(ZoneCode::CBD, 12.0);
        assert!(
            windy_wood.v_down_m_min > windy_wood.v_cross_m_min
                && windy_wood.v_cross_m_min > windy_wood.v_up_m_min
        );
        assert!(windy_wood.v_down_m_min > windy_cbd.v_down_m_min * 1.5);

        // 3. Coupled Monte Carlo FFE on the Riverfront network under Common Random Numbers (`CRN`):
        //    (a) High wind (`15 m/s`) vs. calm wind (`1 m/s`) with brittle Cast Iron pipes
        //    (b) `100%` Seismic ERDIP water pipes + Auxiliary Cisterns + Backup Power slashing dry-hydrant failures & loss!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_ffe = FfeConfig {
            seismic: SeismicScenarioConfig {
                moment_magnitude_mw: 7.8,
                rupture_standoff_km: 9.0,
                vs30_m_per_s: 215.0,
                monte_carlo_trials: 180,
                random_seed: 2026_1906,
                pipe_fragility_law: PipelineFragilityLaw::HazusORourkeAyala1993,
                baseline_water_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                seismic_pipe_replacement_pct: 0.0,
                critical_lifeline_backup_power: false,
                ..SeismicScenarioConfig::default()
            },
            ignition_model: FfeIgnitionModel::Scawthorn2005QuadraticPga,
            wind_speed_mps: 15.0,
            wind_direction_deg: 0.0,
            conflagration_horizon_mins: 180.0,
            initial_attack_flashover_s: 360.0,
            seismic_gas_shutoff_pct: 0.0,
            auxiliary_cistern_coverage_pct: 0.0,
        };
        let high_wind_rep = world.run_ffe_monte_carlo(&base_ffe);
        assert!(high_wind_rep.expected_ignitions > 0.2);
        assert!(high_wind_rep.expected_dry_hydrant_failures > 0.05);
        assert!(high_wind_rep.to_csv_string().starts_with(FFE_PARCEL_CSV_HEADER));

        let calm_rep = world.run_ffe_monte_carlo(&FfeConfig {
            wind_speed_mps: 1.0,
            ..base_ffe.clone()
        });
        assert!(
            high_wind_rep.expected_burned_parcels + 1e-9 >= calm_rep.expected_burned_parcels,
            "15 m/s gale must spread fire across >= parcels than 1 m/s calm ({:.2} vs {:.2})",
            high_wind_rep.expected_burned_parcels,
            calm_rep.expected_burned_parcels
        );

        // Harden water supply (`100%` ERDIP pipes + `100%` Auxiliary Cisterns + Backup Booster Power)
        // under IDENTICAL wind (`15 m/s`) & IDENTICAL ignition draws (`seismic_gas_shutoff_pct: 0.0`)!
        let hardened_water_rep = world.run_ffe_monte_carlo(&FfeConfig {
            seismic: SeismicScenarioConfig {
                seismic_pipe_replacement_pct: 100.0,
                critical_lifeline_backup_power: true,
                ..base_ffe.seismic.clone()
            },
            auxiliary_cistern_coverage_pct: 100.0,
            ..base_ffe
        });
        assert_eq!(hardened_water_rep.expected_dry_hydrant_failures, 0.0);
        assert!(
            hardened_water_rep.expected_knockdowns_suppressed
                >= high_wind_rep.expected_knockdowns_suppressed,
            "Intact >=20 psi hydrants & cisterns must increase initial-attack fire knockdowns ({:.2} vs {:.2})",
            hardened_water_rep.expected_knockdowns_suppressed,
            high_wind_rep.expected_knockdowns_suppressed
        );
        assert!(
            hardened_water_rep.expected_fire_loss_musd
                <= high_wind_rep.expected_fire_loss_musd + 1e-6,
            "Eliminating dry-hydrant failures must reduce expected post-quake conflagration loss (${:.2}M vs ${:.2}M)",
            hardened_water_rep.expected_fire_loss_musd,
            high_wind_rep.expected_fire_loss_musd
        );

        // Verify GPU FireFollowingQuake heatmap rasterizes non-zero RGBA8 pixels
        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::FireFollowingQuake, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_post_earthquake_hospital_ed_surge_triage_queueing_and_dynamic_ambulance_diversion() {
        // 1. Verify FEMA P-58 / Hazus fault-tree hospital functionality monotonicity
        let q_unretro = hospital_fault_tree_functionality(
            HospitalTraumaDesignation::Level2CommunityTrauma,
            false,
            0.75,
            0.65,
            0.0,
            false,
            8.0,
            false,
        );
        let q_spc5 = hospital_fault_tree_functionality(
            HospitalTraumaDesignation::Level2CommunityTrauma,
            true,
            0.75,
            0.65,
            0.0,
            false,
            8.0,
            true,
        );
        assert!(q_spc5 > q_unretro * 1.8);

        // 2. Verify Hazus casualty rates scale with Sa(1.0s)
        let (r_lo, y_lo, g_lo) = hazus_parcel_casualty_rates(0.25, ZoneCode::CBD);
        let (r_hi, y_hi, g_hi) = hazus_parcel_casualty_rates(0.95, ZoneCode::CBD);
        assert!(r_hi > r_lo && y_hi > y_lo && g_hi > g_lo);

        // 3. Compare `ClosestHospitalNoDiversion` vs. `DynamicWaitPlusTravelTriage` under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_surge = HospitalSurgeConfig {
            ffe: FfeConfig {
                seismic: SeismicScenarioConfig {
                    moment_magnitude_mw: 7.9,
                    rupture_standoff_km: 8.5,
                    vs30_m_per_s: 215.0,
                    monte_carlo_trials: 140,
                    random_seed: 2026_3300,
                    baseline_water_pipe_material:
                        BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                    ..SeismicScenarioConfig::default()
                },
                wind_speed_mps: 13.0,
                wind_direction_deg: 0.0,
                conflagration_horizon_mins: 180.0,
                ..FfeConfig::default()
            },
            diversion_policy: AmbulanceDiversionPolicy::ClosestHospitalNoDiversion,
            surge_window_hours: 6.0,
            diversion_occupancy_threshold_pct: 90.0,
            ambulances_per_station: 3,
            min_receiving_hospitals: 3,
            hospital_seismic_retrofit_pct: 34.0,
            hospital_onsite_backup_lifelines: false,
            aba_burn_surge_expansion: false,
            deploy_mobile_field_hospitals: false,
        };

        let closest_rep = world.run_hospital_surge_monte_carlo(&base_surge);
        assert!(closest_rep.expected_total_casualties > 1.0);
        assert_eq!(closest_rep.expected_diverted_ambulances, 0.0);
        assert!(closest_rep
            .to_hospital_csv_string()
            .starts_with(HOSPITAL_SURGE_FACILITY_CSV_HEADER));
        assert!(closest_rep
            .to_parcel_csv_string()
            .starts_with(HOSPITAL_SURGE_PARCEL_CSV_HEADER));

        let dynamic_rep = world.run_hospital_surge_monte_carlo(&HospitalSurgeConfig {
            diversion_policy: AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage,
            ..base_surge.clone()
        });
        assert!(
            dynamic_rep.mean_red_burn_time_to_treatment_min
                <= closest_rep.mean_red_burn_time_to_treatment_min + 1e-6,
            "Dynamic MOCC diversion must reduce or match mean Red/Burn time-to-treatment ({:.2} vs {:.2} min)",
            dynamic_rep.mean_red_burn_time_to_treatment_min,
            closest_rep.mean_red_burn_time_to_treatment_min
        );
        assert!(
            dynamic_rep.burn_matched_to_burn_center_pct + 1e-6
                >= closest_rep.burn_matched_to_burn_center_pct
        );

        // 4. Full resilience package (SPC-5 Retrofit + On-Site Backup Lifelines + ABA +50% Burn Surge + Mobile Field Hospital)
        let hardened_rep = world.run_hospital_surge_monte_carlo(&HospitalSurgeConfig {
            diversion_policy: AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage,
            hospital_seismic_retrofit_pct: 100.0,
            hospital_onsite_backup_lifelines: true,
            aba_burn_surge_expansion: true,
            deploy_mobile_field_hospitals: true,
            ..base_surge
        });
        assert!(
            hardened_rep.mean_system_ed_functionality_pct
                > closest_rep.mean_system_ed_functionality_pct
        );
        assert!(
            hardened_rep.expected_preventable_surge_deaths
                <= closest_rep.expected_preventable_surge_deaths + 1e-6
        );

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::HospitalTraumaSurge, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_rotary_wing_medevac_helipads_smoke_flight_envelope_and_burn_transfers() {
        // 1. Verify Koschmieder visibility extinction & FAA 14 CFR 135.609 vs. WAAS PinS IFR gate
        let (vis_clear, ceil_clear) = koschmieder_smoke_visibility_and_ceiling(6.0, 1800.0, 0.0);
        assert!((vis_clear - 6.0).abs() < 1e-6 && (ceil_clear - 1800.0).abs() < 1e-6);

        let (vis_smoke, ceil_smoke) = koschmieder_smoke_visibility_and_ceiling(6.0, 1800.0, 450.0);
        assert!(vis_smoke < 1.5 && vis_smoke > 0.80 && ceil_smoke < 1250.0);
        // 450 ug/m^3 smoke grounds 14 CFR 135.609 Day VFR (requires >= 2.0 SM), while WAAS PinS IFR (0.75 SM / 350 ft) stays OPEN!
        assert!(!evaluate_helipad_flight_envelope(
            MedevacAvionicsRegime::DayVfrPart135,
            RotorcraftAirframeClass::MediumTwinIcuAw139,
            HelipadStructuralType::HospitalSurfaceGradePad,
            true,
            vis_smoke,
            ceil_smoke,
            10.0,
        ));
        assert!(evaluate_helipad_flight_envelope(
            MedevacAvionicsRegime::WaasPinSIfr,
            RotorcraftAirframeClass::MediumTwinIcuAw139,
            HelipadStructuralType::HospitalSurfaceGradePad,
            true,
            vis_smoke,
            ceil_smoke,
            10.0,
        ));

        // 2. Compare Unhardened Night-VFR Rooftop Pads vs. Hardened WAAS PinS IFR + Surface Pads + WRBDC Burn Pact under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_med = AeromedicalMedevacConfig {
            surge: HospitalSurgeConfig {
                ffe: FfeConfig {
                    seismic: SeismicScenarioConfig {
                        moment_magnitude_mw: 7.9,
                        rupture_standoff_km: 8.5,
                        vs30_m_per_s: 215.0,
                        monte_carlo_trials: 140,
                        random_seed: 2026_3400,
                        baseline_water_pipe_material:
                            BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                        ..SeismicScenarioConfig::default()
                    },
                    wind_speed_mps: 13.5,
                    wind_direction_deg: 0.0,
                    conflagration_horizon_mins: 180.0,
                    ..FfeConfig::default()
                },
                ..HospitalSurgeConfig::default()
            },
            avionics_regime: MedevacAvionicsRegime::NightUnassistedVfr,
            airframe_class: RotorcraftAirframeClass::LightTwinHemsEc135,
            rotorcraft_fleet_count: 3,
            ambient_ceiling_ft: 950.0, // Below 1,000 ft Night-VFR ceiling -> grounds Night-VFR!
            ambient_visibility_sm: 3.2,
            helipad_seismic_and_ground_lz_hardening: false,
            wrbdc_regional_burn_mutual_aid: true,
        };

        let vfr_grounded_rep = world.run_aeromedical_medevac_mc(&base_med);
        assert_eq!(vfr_grounded_rep.expected_air_secondary_burn_transfers, 0.0);
        assert!(vfr_grounded_rep
            .to_helipad_csv_string()
            .starts_with(MEDEVAC_HELIPAD_CSV_HEADER));
        assert!(vfr_grounded_rep
            .to_transfer_arc_csv_string()
            .starts_with(MEDEVAC_TRANSFER_ARC_CSV_HEADER));

        let pins_ifr_rep = world.run_aeromedical_medevac_mc(&AeromedicalMedevacConfig {
            avionics_regime: MedevacAvionicsRegime::WaasPinSIfr,
            airframe_class: RotorcraftAirframeClass::MediumTwinIcuAw139,
            helipad_seismic_and_ground_lz_hardening: true,
            wrbdc_regional_burn_mutual_aid: true,
            ..base_med
        });
        assert!(pins_ifr_rep.system_flight_envelope_open_pct > 50.0);
        assert!(
            pins_ifr_rep.expected_air_secondary_burn_transfers
                > vfr_grounded_rep.expected_air_secondary_burn_transfers
        );
        assert!(
            pins_ifr_rep.mean_secondary_burn_transfer_min
                < vfr_grounded_rep.mean_secondary_burn_transfer_min
        );
        assert!(
            pins_ifr_rep.expected_preventable_burn_transfer_deaths
                <= vfr_grounded_rep.expected_preventable_burn_transfer_deaths + 1e-6
        );

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::AeromedicalMedevac, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_staged_evacuation_arterial_contraflow_smoke_speed_and_ems_corridor_protection() {
        // 1. Verify Wetterberg / Intini smoke-visibility speed & capacity reduction monotonicity
        let (v_dense, q_dense) = smoke_visibility_speed_and_capacity_factors(0.06);
        let (v_haze, q_haze) = smoke_visibility_speed_and_capacity_factors(0.35);
        let (v_clear, q_clear) = smoke_visibility_speed_and_capacity_factors(2.5);
        assert!(v_dense < v_haze && v_haze < v_clear && (v_clear - 1.0).abs() < 1e-9);
        assert!(q_dense < q_haze && q_haze < q_clear && (q_clear - 1.0).abs() < 1e-9);

        // 2. Compare Unstaged + No-Contraflow vs. Full-OneWay (breaks EMS) vs. Staged + EMS-Protected Contraflow under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_evac = EvacuationContraflowConfig {
            medevac: AeromedicalMedevacConfig {
                surge: HospitalSurgeConfig {
                    ffe: FfeConfig {
                        seismic: SeismicScenarioConfig {
                            moment_magnitude_mw: 7.8,
                            rupture_standoff_km: 9.0,
                            vs30_m_per_s: 220.0,
                            monte_carlo_trials: 120,
                            random_seed: 2026_3500,
                            baseline_water_pipe_material:
                                BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                            ..SeismicScenarioConfig::default()
                        },
                        wind_speed_mps: 14.0,
                        wind_direction_deg: 0.0,
                        conflagration_horizon_mins: 180.0,
                        ..FfeConfig::default()
                    },
                    ..HospitalSurgeConfig::default()
                },
                ambient_visibility_sm: 4.5,
                ..AeromedicalMedevacConfig::default()
            },
            staging_strategy: EvacuationStagingStrategy::SimultaneousUnstaged,
            contraflow_policy: ContraflowReversalPolicy::NoContraflowBaseline,
            stage2_release_delay_min: 20.0,
            stage3_release_delay_min: 45.0,
            evac_vehicles_per_capita: 0.72,
            fire_perimeter_road_buffer_m: 58.0,
        };

        let unstaged_no_cf = world.run_evacuation_contraflow_mc(&base_evac);
        assert_eq!(unstaged_no_cf.contraflow_reversed_segments_count, 0);
        assert!(unstaged_no_cf
            .to_parcel_csv_string()
            .starts_with(EVACUATION_PARCEL_CSV_HEADER));
        assert!(unstaged_no_cf
            .to_corridor_csv_string()
            .starts_with(EVACUATION_CORRIDOR_CSV_HEADER));

        let ems_prot_cf = world.run_evacuation_contraflow_mc(&EvacuationContraflowConfig {
            staging_strategy: EvacuationStagingStrategy::FireProximityThreeStage,
            contraflow_policy: ContraflowReversalPolicy::EmsProtectedArterialContraflow,
            ..base_evac.clone()
        });
        assert_eq!(ems_prot_cf.ems_lz_corridor_access_preserved_pct, 100.0);
        assert!(ems_prot_cf.contraflow_reversed_segments_count > 0);
        assert!(
            ems_prot_cf.peak_bottleneck_vc_ratio <= unstaged_no_cf.peak_bottleneck_vc_ratio + 1e-6,
            "Staged + Contraflow must reduce peak bottleneck V/C ({:.2} vs {:.2})",
            ems_prot_cf.peak_bottleneck_vc_ratio,
            unstaged_no_cf.peak_bottleneck_vc_ratio
        );
        assert!(
            ems_prot_cf.expected_fire_overrun_residents
                <= unstaged_no_cf.expected_fire_overrun_residents + 1e-6,
            "ems_prot_cf overrun {:.4} (t90={:.2}, peak_vc={:.3}, cf={}) > unstaged_no_cf overrun {:.4} (t90={:.2}, peak_vc={:.3})",
            ems_prot_cf.expected_fire_overrun_residents,
            ems_prot_cf.t90_clearance_min,
            ems_prot_cf.peak_bottleneck_vc_ratio,
            ems_prot_cf.contraflow_reversed_segments_count,
            unstaged_no_cf.expected_fire_overrun_residents,
            unstaged_no_cf.t90_clearance_min,
            unstaged_no_cf.peak_bottleneck_vc_ratio
        );

        let full_oneway = world.run_evacuation_contraflow_mc(&EvacuationContraflowConfig {
            contraflow_policy: ContraflowReversalPolicy::FullUnprotectedOneWayOut,
            ..base_evac
        });
        assert!(
            ems_prot_cf.ems_lz_corridor_access_preserved_pct
                >= full_oneway.ems_lz_corridor_access_preserved_pct
        );

        // 3. Verify live undoable contraflow topology application (`Ctrl+Z` / `Ctrl+Y`)
        let orig_lanes: Vec<usize> = world.network.segments.iter().map(|s| s.lane_count).collect();
        let (rev_n, _) = world.apply_evacuation_contraflow_to_network(&ems_prot_cf);
        assert!(rev_n > 0);
        assert!(world.undo_edit().is_some());
        let restored_lanes: Vec<usize> =
            world.network.segments.iter().map(|s| s.lane_count).collect();
        assert_eq!(orig_lanes, restored_lanes);

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::EvacuationContraflow, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_post_earthquake_emergency_shelters_hepa_filtration_habitability_and_routing() {
        // 1. Verify EPA / ASHRAE Guideline 44 indoor PM2.5 filtration mass-balance & power dependence
        let pm_merv8 = compute_indoor_shelter_pm25_ug_m3(
            320.0,
            ShelterFacilityArchetype::PublicSchoolGymnasium,
            AirFiltrationStandard::BaselineMerv8Unsealed,
            0.0,
            true,
        );
        let pm_merv13 = compute_indoor_shelter_pm25_ug_m3(
            320.0,
            ShelterFacilityArchetype::PublicSchoolGymnasium,
            AirFiltrationStandard::RecirculatingMerv13Hvac,
            0.0,
            true,
        );
        let pm_hepa_powered = compute_indoor_shelter_pm25_ug_m3(
            320.0,
            ShelterFacilityArchetype::PublicSchoolGymnasium,
            AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
            450.0,
            true,
        );
        let pm_hepa_blackout = compute_indoor_shelter_pm25_ug_m3(
            320.0,
            ShelterFacilityArchetype::PublicSchoolGymnasium,
            AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
            450.0,
            false,
        );
        assert!(pm_hepa_powered < 35.0 && pm_hepa_powered < pm_merv13 && pm_merv13 < pm_merv8);
        assert!(pm_hepa_blackout > pm_hepa_powered * 3.0);

        // 2. Compare Uncoordinated Nearest MERV-8 vs. 2-Stage Clean-Air Capacitated HEPA + Backup Lifelines under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_sh = EmergencySheltersConfig {
            evac: EvacuationContraflowConfig {
                medevac: AeromedicalMedevacConfig {
                    surge: HospitalSurgeConfig {
                        ffe: FfeConfig {
                            seismic: SeismicScenarioConfig {
                                moment_magnitude_mw: 7.8,
                                rupture_standoff_km: 9.0,
                                vs30_m_per_s: 215.0,
                                monte_carlo_trials: 120,
                                random_seed: 2026_3600,
                                baseline_water_pipe_material:
                                    BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                baseline_sewer_pipe_material:
                                    BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                ..SeismicScenarioConfig::default()
                            },
                            wind_speed_mps: 13.5,
                            wind_direction_deg: 0.0,
                            conflagration_horizon_mins: 180.0,
                            ..FfeConfig::default()
                        },
                        ..HospitalSurgeConfig::default()
                    },
                    ..AeromedicalMedevacConfig::default()
                },
                ..EvacuationContraflowConfig::default()
            },
            filtration_standard: AirFiltrationStandard::BaselineMerv8Unsealed,
            siting_routing_policy: ShelterSitingAndRoutingPolicy::UncoordinatedNearestShelter,
            max_open_shelters_budget: 4,
            sqft_per_evacuee: 50.0,
            shelter_seismic_retrofit_pct: 20.0,
            shelter_backup_power_water_sanitation: false,
            portable_hepa_cadr_m3h_per_100m2: 450.0,
            indoor_pm25_health_limit_ug_m3: 35.0,
        };

        let uncoord_rep = world.run_emergency_shelters_mc(&base_sh);
        assert!(uncoord_rep.total_displaced_evacuees > 50.0);
        assert!(uncoord_rep
            .to_shelter_csv_string()
            .starts_with(SHELTER_FACILITY_CSV_HEADER));
        assert!(uncoord_rep
            .to_parcel_csv_string()
            .starts_with(SHELTER_PARCEL_ASSIGN_CSV_HEADER));

        let clean_cap_rep = world.run_emergency_shelters_mc(&EmergencySheltersConfig {
            filtration_standard: AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
            siting_routing_policy: ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated,
            shelter_seismic_retrofit_pct: 100.0,
            shelter_backup_power_water_sanitation: true,
            ..base_sh
        });
        assert!(
            clean_cap_rep.mean_assigned_indoor_pm25_ug_m3
                < uncoord_rep.mean_assigned_indoor_pm25_ug_m3
        );
        assert!(
            clean_cap_rep.sheltered_clean_air_pct + 1e-6 >= uncoord_rep.sheltered_clean_air_pct
        );
        assert!(
            clean_cap_rep.expected_overcrowded_or_unmet_evacuees
                <= uncoord_rep.expected_overcrowded_or_unmet_evacuees + 1e-6
        );

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::EmergencyShelters, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_multi_day_shelter_commodity_resupply_clarke_wright_vrp_and_deprivation_costs() {
        // 1. Verify Holguin-Veras (2013) strict convexity & medical-oxygen acuity ordering
        let c6 = holguin_veras_deprivation_cost(ReliefCommodityKind::PotableWaterLiters, 6.0, 100.0);
        let c12 =
            holguin_veras_deprivation_cost(ReliefCommodityKind::PotableWaterLiters, 12.0, 100.0);
        assert!(c12 - c6 > c6 && c6 > 0.0);
        let o2_c = holguin_veras_deprivation_cost(
            ReliefCommodityKind::MedicalOxygenKiloLiters,
            8.0,
            50.0,
        );
        let mre_c =
            holguin_veras_deprivation_cost(ReliefCommodityKind::MreRationsMeals, 8.0, 50.0);
        assert!(o2_c > mre_c * 10.0);

        // 2. Verify Clarke-Wright + 2-Opt tour beats radial out-and-back sum on 4-stop square
        let pts = [
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 10.0, y: 2.0 },
            Vec2 { x: 12.0, y: 10.0 },
            Vec2 { x: 2.0, y: 12.0 },
            Vec2 { x: 8.0, y: 8.0 },
        ];
        let mut dmat = vec![vec![0.0; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                dmat[i][j] = pts[i].dist(pts[j]);
            }
        }
        let (tour, tour_len) = solve_clarke_wright_2opt_tour(0, &[1, 4, 2, 3], &dmat);
        let radial_sum: f64 = [1, 2, 3, 4].iter().map(|&i| 2.0 * dmat[0][i]).sum();
        assert_eq!(tour.len(), 4);
        assert!(tour_len < radial_sum * 0.65);

        // 3. Compare Static Milk-Run vs. Deprivation-Aware Clarke-Wright IRP under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_resup = CommodityResupplyConfig {
            shelters: EmergencySheltersConfig {
                evac: EvacuationContraflowConfig {
                    medevac: AeromedicalMedevacConfig {
                        surge: HospitalSurgeConfig {
                            ffe: FfeConfig {
                                seismic: SeismicScenarioConfig {
                                    moment_magnitude_mw: 7.9,
                                    rupture_standoff_km: 8.5,
                                    vs30_m_per_s: 215.0,
                                    monte_carlo_trials: 100,
                                    random_seed: 2026_3700,
                                    baseline_water_pipe_material:
                                        BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                    ..SeismicScenarioConfig::default()
                                },
                                wind_speed_mps: 12.0,
                                ..FfeConfig::default()
                            },
                            ..HospitalSurgeConfig::default()
                        },
                        ..AeromedicalMedevacConfig::default()
                    },
                    ..EvacuationContraflowConfig::default()
                },
                max_open_shelters_budget: 4,
                ..EmergencySheltersConfig::default()
            },
            dispatch_policy: ResupplyDispatchPolicy::StaticEqualRotationMilkRun,
            horizon_days: 7,
            water_tanker_trucks: 2,
            diesel_fuel_trucks: 2,
            mre_ration_trucks: 2,
            medical_oxygen_trucks: 2,
            initial_onsite_reserve_hours: 24.0,
            enable_helicopter_sling_resupply: false,
        };

        let milk_rep = world.run_commodity_resupply_mc(&base_resup);
        assert!(milk_rep
            .to_site_csv_string()
            .starts_with(RESUPPLY_SITE_CSV_HEADER));
        assert!(milk_rep
            .to_daily_csv_string()
            .starts_with(RESUPPLY_DAILY_CSV_HEADER));

        let irp_rep = world.run_commodity_resupply_mc(&CommodityResupplyConfig {
            dispatch_policy: ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp,
            enable_helicopter_sling_resupply: true,
            ..base_resup.clone()
        });
        assert!(
            irp_rep.total_deprivation_social_cost <= milk_rep.total_deprivation_social_cost + 1e-6,
            "Deprivation-Aware IRP must reduce social deprivation cost ({:.1} vs {:.1})",
            irp_rep.total_deprivation_social_cost,
            milk_rep.total_deprivation_social_cost
        );
        assert!(irp_rep.system_supply_security_pct + 1e-6 >= milk_rep.system_supply_security_pct);

        // 4. Verify CMS / Joint Commission 96-Hour On-Site Reserve further slashes deprivation
        let irp_96h = world.run_commodity_resupply_mc(&CommodityResupplyConfig {
            dispatch_policy: ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp,
            initial_onsite_reserve_hours: 96.0,
            enable_helicopter_sling_resupply: true,
            ..base_resup
        });
        assert!(
            irp_96h.total_deprivation_social_cost <= irp_rep.total_deprivation_social_cost + 1e-6
        );

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::CommodityResupply, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_gtfs_transit_bus_and_ada_paratransit_contraflow_evacuation_into_clean_air_shelters() {
        // 1. Verify TCQSM emergency all-door boarding dwell & 60-ft articulated vs 40-ft coach throughput
        let (cap40, dw_cash) = compute_evac_coach_capacity_and_dwell_min(
            BusCoachSpec::Standard40Ft,
            BoardingFareMode::FrontDoorCash,
            50.0,
        );
        let (cap60, dw_brt) = compute_evac_coach_capacity_and_dwell_min(
            BusCoachSpec::Articulated60FtBrt,
            BoardingFareMode::AllDoorFareFreeBrt,
            50.0,
        );
        assert_eq!(cap40, 50);
        assert_eq!(cap60, 80);
        assert!(dw_brt < dw_cash * 0.55);

        let (ada_cap, ada_dw) = compute_ada_van_capacity_and_dwell_min(6.0);
        assert_eq!(ada_cap, 8);
        assert!(ada_dw > dw_brt * 2.5);

        // 2. Compare Mixed-Auto-Queue Fixed-GTFS vs. Dedicated Bus+EMS Contraflow + Clean-Air TBEP under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_tevac = TransitEvacuationConfig {
            shelters: EmergencySheltersConfig {
                evac: EvacuationContraflowConfig {
                    medevac: AeromedicalMedevacConfig {
                        surge: HospitalSurgeConfig {
                            ffe: FfeConfig {
                                seismic: SeismicScenarioConfig {
                                    moment_magnitude_mw: 7.8,
                                    rupture_standoff_km: 9.0,
                                    vs30_m_per_s: 220.0,
                                    monte_carlo_trials: 80,
                                    random_seed: 2026_3800,
                                    baseline_water_pipe_material:
                                        BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                    ..SeismicScenarioConfig::default()
                                },
                                wind_speed_mps: 11.5,
                                ..FfeConfig::default()
                            },
                            ..HospitalSurgeConfig::default()
                        },
                        ..AeromedicalMedevacConfig::default()
                    },
                    ..EvacuationContraflowConfig::default()
                },
                max_open_shelters_budget: 4,
                shelter_backup_power_water_sanitation: true,
                ..EmergencySheltersConfig::default()
            },
            lane_mode: BusContraflowLaneMode::MixedAutoQueueNoPriority,
            dispatch_strategy: TransitDispatchStrategy::FixedGtfsHeadwayLoop,
            coach_spec: BusCoachSpec::Standard40Ft,
            fare_mode: BoardingFareMode::FrontDoorCash,
            transit_bus_fleet: 6,
            ada_paratransit_vans: 4,
            max_pickup_depots: 8,
            walk_catchment_radius_m: 400.0,
            clearance_window_min: 110.0,
            carless_demand_multiplier: 1.2,
        };

        let mixed_rep = world.run_transit_bus_evacuation_mc(&base_tevac);
        assert!(mixed_rep
            .to_depot_csv_string()
            .starts_with(TRANSIT_EVAC_DEPOT_CSV_HEADER));
        assert!(mixed_rep
            .to_wave_csv_string()
            .starts_with(TRANSIT_EVAC_WAVE_CSV_HEADER));
        assert!(mixed_rep.active_pickup_depots_count >= 4);

        let priority_rep = world.run_transit_bus_evacuation_mc(&TransitEvacuationConfig {
            lane_mode: BusContraflowLaneMode::DedicatedBusPlusEmsContraflow,
            dispatch_strategy: TransitDispatchStrategy::FireUrgencyCleanAirTbep,
            coach_spec: BusCoachSpec::Articulated60FtBrt,
            fare_mode: BoardingFareMode::AllDoorFareFreeBrt,
            ..base_tevac
        });

        assert!(
            priority_rep.mean_bus_shuttle_cycle_min < mixed_rep.mean_bus_shuttle_cycle_min * 0.72,
            "Dedicated Bus+EMS contraflow + all-door BRT must cut shuttle cycle time ({:.1}m vs {:.1}m)",
            priority_rep.mean_bus_shuttle_cycle_min,
            mixed_rep.mean_bus_shuttle_cycle_min
        );
        assert!(
            priority_rep.combined_clean_air_sheltered_pct + 1e-6
                >= mixed_rep.combined_clean_air_sheltered_pct,
            "priority clean={:.4}% (carless={:.2}%, ada={:.2}%, clean_sh={:.2}, trips={:.1}) < mixed clean={:.4}% (carless={:.2}%, ada={:.2}%, clean_sh={:.2}, trips={:.1})",
            priority_rep.combined_clean_air_sheltered_pct,
            priority_rep.evacuated_carless_pct,
            priority_rep.evacuated_ada_pct,
            priority_rep.open_clean_air_shelters_count,
            priority_rep.mean_trips_per_bus,
            mixed_rep.combined_clean_air_sheltered_pct,
            mixed_rep.evacuated_carless_pct,
            mixed_rep.evacuated_ada_pct,
            mixed_rep.open_clean_air_shelters_count,
            mixed_rep.mean_trips_per_bus
        );
        assert!(priority_rep.stranded_vulnerable_pax <= mixed_rep.stranded_vulnerable_pax + 1e-6);
        assert!(
            priority_rep.ems_inbound_access_preserved_pct >= 80.0,
            "priority ems_ok={:.2}% < 80.0 (mixed ems_ok={:.2}%, cf_segs={}, depots={})",
            priority_rep.ems_inbound_access_preserved_pct,
            mixed_rep.ems_inbound_access_preserved_pct,
            priority_rep.bus_ems_contraflow_segments_count,
            priority_rep.active_pickup_depots_count,
        );

        // 3. Verify undoable live Bus+EMS contraflow network mutation + Ctrl+Z / Ctrl+Y
        let undo_before = world.edit_history.undo_len();
        let (bus_segs, _) = world.apply_bus_ems_contraflow_to_network(&priority_rep);
        assert!(bus_segs > 0);
        assert_eq!(world.edit_history.undo_len(), undo_before + 1);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::TransitBusEvacuation, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_cellular_blackout_fragility_wea_siren_d2d_mobilization_and_delayed_staging() {
        // 1. Verify NIST TN 1950 / NRC inverse-square siren attenuation & -22 dB indoor loss
        let spl_30_out = compute_siren_spl_dbc(30.48, 124.0, false);
        let spl_30_in = compute_siren_spl_dbc(30.48, 124.0, true);
        let spl_600_out = compute_siren_spl_dbc(600.0, 124.0, false);
        assert!((spl_30_out - 124.0).abs() < 0.05);
        assert!((spl_30_out - spl_30_in - 22.0).abs() < 0.05);
        assert!(spl_600_out > 70.0 && spl_600_out < 99.0);

        // 2. Verify Lindell-Perry PADM / EMBLEM2 Weibull warning + preparation convolution monotonicity
        let m15_fast = evaluate_padm_mobilization_cdf(15.0, 3.0, 1.8, 9.0, 0.96);
        let m45_fast = evaluate_padm_mobilization_cdf(45.0, 3.0, 1.8, 9.0, 0.96);
        let m45_dark = evaluate_padm_mobilization_cdf(45.0, 48.0, 1.3, 20.0, 0.55);
        assert!(m45_fast > m15_fast && m15_fast > 0.15);
        assert!(m45_fast > m45_dark * 2.4);

        // 3. Compare Legacy 3h Battery + WEA-Only Unstaged vs. Multi-Modal + Adaptive Blackout Staging at t = +5h nocturnal quake!
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_wcfg = WarningMobilizationConfig {
            transit_evac: TransitEvacuationConfig {
                shelters: EmergencySheltersConfig {
                    evac: EvacuationContraflowConfig {
                        medevac: AeromedicalMedevacConfig {
                            surge: HospitalSurgeConfig {
                                ffe: FfeConfig {
                                    seismic: SeismicScenarioConfig {
                                        moment_magnitude_mw: 7.9,
                                        rupture_standoff_km: 8.5,
                                        vs30_m_per_s: 215.0,
                                        monte_carlo_trials: 80,
                                        random_seed: 2026_3900,
                                        baseline_water_pipe_material:
                                            BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                        ..SeismicScenarioConfig::default()
                                    },
                                    wind_speed_mps: 12.0,
                                    ..FfeConfig::default()
                                },
                                ..HospitalSurgeConfig::default()
                            },
                            ..AeromedicalMedevacConfig::default()
                        },
                        ..EvacuationContraflowConfig::default()
                    },
                    ..EmergencySheltersConfig::default()
                },
                clearance_window_min: 110.0,
                ..TransitEvacuationConfig::default()
            },
            telecom_backup: TelecomBackupPowerStandard::LegacyCommercial2to4HrBattery,
            channel_mix: AlertChannelMix::WeaAndReverse911Only,
            staging_policy: DepartureStagingPolicy::UnstagedImmediateOnReceipt,
            hours_post_mainshock: 5.0,
            nocturnal_sleeping_event: true,
            outdoor_siren_sites: 6,
            canvass_patrol_units: 8,
        };

        let wea_only_rep = world.run_warning_mobilization_mc(&base_wcfg);
        assert!(wea_only_rep
            .to_tower_csv_string()
            .starts_with(WARNING_TOWER_CSV_HEADER));
        assert!(wea_only_rep
            .to_curve_csv_string()
            .starts_with(WARNING_CURVE_CSV_HEADER));
        assert!(wea_only_rep.mean_cell_towers_blackout_pct > 25.0);

        let multi_rep = world.run_warning_mobilization_mc(&WarningMobilizationConfig {
            channel_mix: AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor,
            staging_policy: DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging,
            ..base_wcfg.clone()
        });
        assert!(
            multi_rep.mean_warning_receipt_min < wea_only_rep.mean_warning_receipt_min * 0.65,
            "Multi-modal sirens + Hi-Lo/D2D must slash warning receipt delay under cell blackout ({:.1}m vs {:.1}m)",
            multi_rep.mean_warning_receipt_min,
            wea_only_rep.mean_warning_receipt_min
        );
        assert!(
            multi_rep.late_departure_fire_trapped_pax
                <= wea_only_rep.late_departure_fire_trapped_pax + 1e-6
        );

        // 4. Verify CPUC 72-Hr Hardened + LEO Satellite Backhaul slashes cellular blackout % at +5h
        let cpuc72_rep = world.run_warning_mobilization_mc(&WarningMobilizationConfig {
            telecom_backup: TelecomBackupPowerStandard::Cpuc72HrHardenedWithLeoBackhaul,
            ..base_wcfg
        });
        assert!(
            cpuc72_rep.mean_cell_towers_blackout_pct
                < wea_only_rep.mean_cell_towers_blackout_pct * 0.55
        );

        // 5. Verify Ctrl+Z / Ctrl+Y undoable parcel staging application
        let u0 = world.edit_history.undo_len();
        let n_upd = world.apply_warning_staging_to_parcels(&multi_rep);
        assert_eq!(n_upd, world.parcels.len());
        assert_eq!(world.edit_history.undo_len(), u0 + 1);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::WarningMobilization, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_fema_ipaws_pep_eas_noaa_same_fnars_and_mers_esf2_resilience() {
        // 1. Verify NPWS Primary Entry Point 50kW AM groundwave outlasts VHF at 90 km & IPAWS probability law
        let (pep_90, nwr_90) = compute_pep_am_and_nwr_vhf_reliability(90.0);
        assert!(pep_90 > 0.70 && pep_90 > nwr_90 * 2.0);
        assert_eq!(combine_ipaws_pathway_reach_prob(false, 0.9, 0.8, 0.7), 0.0);
        let p_tri = combine_ipaws_pathway_reach_prob(true, 0.0, 0.50, 0.60);
        assert!((p_tri - 0.80).abs() < 1e-6);

        // 2. Compare Fiber-Only EOC + WEA-3.0-Only vs. Full FEMA Continuity (FNARS+MERS VSAT + PEP EAS + NWR SAME + MERS ESF-2)
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_fema = FemaCommsConfig {
            warning: WarningMobilizationConfig {
                transit_evac: TransitEvacuationConfig {
                    shelters: EmergencySheltersConfig {
                        evac: EvacuationContraflowConfig {
                            medevac: AeromedicalMedevacConfig {
                                surge: HospitalSurgeConfig {
                                    ffe: FfeConfig {
                                        seismic: SeismicScenarioConfig {
                                            moment_magnitude_mw: 8.1,
                                            rupture_standoff_km: 9.0,
                                            vs30_m_per_s: 215.0,
                                            monte_carlo_trials: 80,
                                            random_seed: 2026_4000,
                                            baseline_water_pipe_material:
                                                BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                            ..SeismicScenarioConfig::default()
                                        },
                                        ..FfeConfig::default()
                                    },
                                    ..HospitalSurgeConfig::default()
                                },
                                ..AeromedicalMedevacConfig::default()
                            },
                            ..EvacuationContraflowConfig::default()
                        },
                        ..EmergencySheltersConfig::default()
                    },
                    ..TransitEvacuationConfig::default()
                },
                telecom_backup: TelecomBackupPowerStandard::LegacyCommercial2to4HrBattery,
                hours_post_mainshock: 5.5,
                nocturnal_sleeping_event: true,
                ..WarningMobilizationConfig::default()
            },
            eoc_uplink: EocIpaswUplinkMode::CommercialTerrestrialFiberOnly,
            posture: FemaCommsPosture::IpaswWea30CellularOnly,
            household_eas_am_fm_radio_pct: 48.0,
            household_nwr_same_receiver_pct: 36.0,
            fema_mers_irv_units: 3,
            airlift_mers_flyaway_kits: true,
        };

        let fragile_rep = world.run_fema_comms_mc(&base_fema);
        assert!(fragile_rep
            .to_asset_csv_string()
            .starts_with(FEMA_ASSET_CSV_HEADER));
        assert!(fragile_rep
            .to_timeline_csv_string()
            .starts_with(FEMA_TIMELINE_CSV_HEADER));

        let fema_full_rep = world.run_fema_comms_mc(&FemaCommsConfig {
            eoc_uplink: EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity,
            posture: FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment,
            ..base_fema
        });

        assert!(
            fema_full_rep.eoc_cap_origination_uptime_pct
                > fragile_rep.eoc_cap_origination_uptime_pct + 20.0,
            "FNARS HF + MERS VSAT must restore COG EOC CAP v1.2 origination ({:.1}% vs {:.1}%)",
            fema_full_rep.eoc_cap_origination_uptime_pct,
            fragile_rep.eoc_cap_origination_uptime_pct
        );
        assert!(fema_full_rep.npws_pep_broadcast_uptime_pct >= 80.0);
        assert!(
            fema_full_rep.public_ipaws_verified_reach_pct
                > fragile_rep.public_ipaws_verified_reach_pct + 25.0
        );
        assert!(
            fema_full_rep.esf2_nifog_responder_lmr_coverage_pct
                > fragile_rep.esf2_nifog_responder_lmr_coverage_pct
        );

        let u0 = world.edit_history.undo_len();
        let n_p = world.apply_fema_mers_posture_tracked(&fema_full_rep);
        assert_eq!(n_p, world.parcels.len());
        assert_eq!(world.edit_history.undo_len(), u0 + 1);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::FemaCommsArchitecture, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_fema_isb_usace_is26_pod_mgc_queueing_spillback_and_ipaws_routing() {
        // 1. Verify exact USACE IS-26 Type I / II / III throughput specs & M/G/c queue convexity
        let (c1, mu1, pax1, _, _) = FemaPodClassification::TypeIMetroFourLane.usace_is26_specs();
        let (c2, mu2, pax2, _, _) = FemaPodClassification::TypeIIRegionalTwoLane.usace_is26_specs();
        let (c3, mu3, pax3, _, _) = FemaPodClassification::TypeIIILocalOneLane.usace_is26_specs();
        assert_eq!(((c1 as f64 * mu1) as usize, pax1), (560, 20_000));
        assert_eq!(((c2 as f64 * mu2) as usize, pax2), (280, 10_000));
        assert_eq!(((c3 as f64 * mu3) as usize, pax3), (140, 5_000));

        let (_, _, w_low, sp_low, _) =
            evaluate_usace_pod_mgc_hour_step(2, 140.0, 160.0, 0.0, 1.0, 46.0);
        let (_, _, w_surge, sp_surge, _) =
            evaluate_usace_pod_mgc_hour_step(2, 140.0, 420.0, 30.0, 1.25, 46.0);
        assert!(w_surge > w_low * 4.0);
        assert!(sp_surge > sp_low + 25.0);

        // 2. Compare Uncoordinated Nearest-POD Rush vs. FEMA IPAWS Load-Balanced + Dynamic ISB Hotshot under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_pods = FemaIsbPodsConfig {
            fema_comms: FemaCommsConfig {
                warning: WarningMobilizationConfig {
                    transit_evac: TransitEvacuationConfig {
                        shelters: EmergencySheltersConfig {
                            evac: EvacuationContraflowConfig {
                                medevac: AeromedicalMedevacConfig {
                                    surge: HospitalSurgeConfig {
                                        ffe: FfeConfig {
                                            seismic: SeismicScenarioConfig {
                                                moment_magnitude_mw: 7.9,
                                                rupture_standoff_km: 9.0,
                                                vs30_m_per_s: 215.0,
                                                monte_carlo_trials: 80,
                                                random_seed: 2026_4100,
                                                baseline_water_pipe_material:
                                                    BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                                ..SeismicScenarioConfig::default()
                                            },
                                            ..FfeConfig::default()
                                        },
                                        ..HospitalSurgeConfig::default()
                                    },
                                    ..AeromedicalMedevacConfig::default()
                                },
                                ..EvacuationContraflowConfig::default()
                            },
                            ..EmergencySheltersConfig::default()
                        },
                        ..TransitEvacuationConfig::default()
                    },
                    ..WarningMobilizationConfig::default()
                },
                ..FemaCommsConfig::default()
            },
            routing_policy: IpaswPodRoutingPolicy::UncoordinatedNearestPodRush,
            isb_policy: IsbReplenishmentPolicy::OvernightStaticEqualPush,
            active_county_pods: 5,
            isb_daily_53ft_trailers: 8,
            base_population_needing_pods_pct: 75.0,
            upgrade_arterial_pods_to_type1: false,
        };

        let rush_rep = world.run_fema_isb_pods_mc(&base_pods);
        assert!(rush_rep
            .to_pod_csv_string()
            .starts_with(FEMA_POD_SITE_CSV_HEADER));
        assert!(rush_rep
            .to_hourly_csv_string()
            .starts_with(FEMA_POD_HOURLY_CSV_HEADER));

        let ipaws_rep = world.run_fema_isb_pods_mc(&FemaIsbPodsConfig {
            routing_policy: IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced,
            isb_policy: IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock,
            upgrade_arterial_pods_to_type1: true,
            ..base_pods
        });

        assert!(
            ipaws_rep.mean_pod_queue_wait_min < rush_rep.mean_pod_queue_wait_min * 0.72,
            "IPAWS POD load-balancing + Type I upgrade must cut queue wait ({:.1}m vs {:.1}m)",
            ipaws_rep.mean_pod_queue_wait_min,
            rush_rep.mean_pod_queue_wait_min
        );
        assert!(
            ipaws_rep.peak_arterial_spillback_vehs <= rush_rep.peak_arterial_spillback_vehs + 1e-6
        );
        assert!(ipaws_rep.population_demand_met_pct + 1e-6 >= rush_rep.population_demand_met_pct);

        // 3. Verify Ctrl+Z / Ctrl+Y undoable POD queue spillback injection onto RoadNetwork
        let u0 = world.edit_history.undo_len();
        let _ = world.apply_pod_spillback_to_network(&rush_rep);
        assert_eq!(world.edit_history.undo_len(), u0 + 1);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::FemaIsbPods, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn test_fema_type1_usar_usng_grids_heavy_debris_push_and_isb_pod_reopening() {
        // 1. Verify Coburn-Spence / Macintyre 72h survival ordering (Wood > RC > URM) & monotonic decay
        let s_urm_12 = coburn_spence_entrapped_survival_prob(
            12.0,
            StructuralCollapseArchetype::UnreinforcedMasonryUrm,
        );
        let s_urm_72 = coburn_spence_entrapped_survival_prob(
            72.0,
            StructuralCollapseArchetype::UnreinforcedMasonryUrm,
        );
        let s_wood_72 = coburn_spence_entrapped_survival_prob(
            72.0,
            StructuralCollapseArchetype::WoodAndLightSteelFrame,
        );
        assert!(s_urm_12 > 0.70 && s_urm_72 < 0.15);
        assert!(s_wood_72 > s_urm_72 * 2.5);

        let (cbd_tons, cbd_spill) = hazus_parcel_collapse_debris_tons(2_500.0, ZoneCode::CBD, 0.65);
        let (r1_tons, _) = hazus_parcel_collapse_debris_tons(2_500.0, ZoneCode::R1, 0.65);
        assert!(cbd_tons > r1_tons * 3.5 && cbd_spill > 20.0);

        // 2. Compare Uncoordinated Haul-Off vs. Co-Optimized Teeth-of-a-Comb US&R + ISB/POD Push under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
            objective_model: SitingObjectiveModel::DaskinMexclp1983,
            apparatus_busy_prob: 0.35,
            target_coverage_pct: 95.0,
            max_stations_budget: 4,
            ..MclpSitingConfig::default()
        });
        world.apply_mclp_siting_solution(&sited);

        let base_usar = FemaUsarDebrisConfig {
            fema_pods: FemaIsbPodsConfig {
                fema_comms: FemaCommsConfig {
                    warning: WarningMobilizationConfig {
                        transit_evac: TransitEvacuationConfig {
                            shelters: EmergencySheltersConfig {
                                evac: EvacuationContraflowConfig {
                                    medevac: AeromedicalMedevacConfig {
                                        surge: HospitalSurgeConfig {
                                            ffe: FfeConfig {
                                                seismic: SeismicScenarioConfig {
                                                    moment_magnitude_mw: 8.1,
                                                    rupture_standoff_km: 8.5,
                                                    vs30_m_per_s: 210.0,
                                                    monte_carlo_trials: 80,
                                                    random_seed: 2026_4200,
                                                    ..SeismicScenarioConfig::default()
                                                },
                                                ..FfeConfig::default()
                                            },
                                            ..HospitalSurgeConfig::default()
                                        },
                                        ..AeromedicalMedevacConfig::default()
                                    },
                                    ..EvacuationContraflowConfig::default()
                                },
                                ..EmergencySheltersConfig::default()
                            },
                            ..TransitEvacuationConfig::default()
                        },
                        ..WarningMobilizationConfig::default()
                    },
                    ..FemaCommsConfig::default()
                },
                ..FemaIsbPodsConfig::default()
            },
            policy: DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff,
            fema_type1_usar_task_forces: 2,
            heavy_debris_push_crews: 4,
            horizon_hours: 72.0,
            live_injection_snapshot_hour: 12.0,
        };

        let haul_rep = world.run_fema_usar_debris_mc(&base_usar);
        assert!(haul_rep
            .to_grid_csv_string()
            .starts_with(FEMA_USNG_GRID_CSV_HEADER));
        assert!(haul_rep
            .to_timeline_csv_string()
            .starts_with(FEMA_DEBRIS_TIMELINE_CSV_HEADER));
        assert_eq!(haul_rep.usng_grids_count, 9);

        let comb_rep = world.run_fema_usar_debris_mc(&FemaUsarDebrisConfig {
            policy: DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush,
            ..base_usar
        });

        assert!(
            comb_rep.expected_live_rescues >= haul_rep.expected_live_rescues - 1e-6,
            "Teeth-of-a-Comb Phase I Push must rescue >= entrapped victims ({:.2} vs {:.2})",
            comb_rep.expected_live_rescues,
            haul_rep.expected_live_rescues
        );
        assert!(
            comb_rep.time_to_reopen_all_pods_hrs <= haul_rep.time_to_reopen_all_pods_hrs + 1e-6
        );

        // 3. Verify Ctrl+Z / Ctrl+Y undoable Hour-H debris & bulldozer reopening injection
        let u0 = world.edit_history.undo_len();
        let _ = world.apply_debris_and_reopening_at_hour(&comb_rep, 12.0);
        assert_eq!(world.edit_history.undo_len(), u0 + 1);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::FemaUsarDebris, &[], &mut arena);
        let nz = arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count();
        assert_eq!(nz, HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn fema_gsb_tdms_drc_and_wildfire_cameras_work() {
        // 1. Verify USACE 249th Prime Power sizing & DRC/MRIC Allen-Cunneen M/G/c queueing physics
        let (hosp_kw, hosp_pre_h) = size_usace_249th_prime_power_pack(true, false, false, true);
        let (_, hosp_raw_h) = size_usace_249th_prime_power_pack(true, false, false, false);
        assert!(hosp_kw >= 1000.0 && hosp_pre_h < hosp_raw_h);
        let (w_light, _) = evaluate_drc_mric_mgc_wait_min(8.0, 12, 26.0, 0.55);
        let (w_heavy, _) = evaluate_drc_mric_mgc_wait_min(26.0, 12, 26.0, 0.55);
        assert!(w_heavy > w_light);

        // 2. Verify ALERTWest Camera Network JSON, GeoJSON & CSV feed parsers + WLS cross-azimuth triangulation
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let cams = parse_wildfire_camera_feed_str(
            &builtin_western_wildfire_camera_feed_json(),
            world.network.bounds,
        )
        .unwrap();
        assert!(cams.len() >= 5);
        assert!(cams.iter().any(|c| c.co_located_pnsn_shakealert));

        let geojson_str = r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[-105.0512,40.0244]},"properties":{"id":"Axis-WestRidge","provider":"ALERTWest Network","zoom_x":40.0,"shakealert":true}}]}"#;
        let gj_cams = parse_wildfire_camera_feed_str(geojson_str, world.network.bounds).unwrap();
        assert_eq!(gj_cams.len(), 1);

        let csv_str = "camera_id,provider,kind,lat,lon,elev_m,az_deg,zoom_x,shakealert
Axis-SummitLookout,ALERTWest,PTZ,40.018,-105.031,230,95,40,true
";
        let csv_cams = parse_wildfire_camera_feed_str(csv_str, world.network.bounds).unwrap();
        assert_eq!(csv_cams.len(), 1);

        // Exact 2-ray orthogonal intersection at (500, 400): Cam A at (100, 400) bearing 90° East, Cam B at (500, 800) bearing 0° North
        let (est_pt, cep_m) = triangulate_camera_bearings_wls(
            &[
                (Vec2 { x: 100.0, y: 400.0 }, 90.0, 0.25),
                (Vec2 { x: 500.0, y: 800.0 }, 0.0, 0.25),
            ],
            Vec2 { x: 0.0, y: 0.0 },
        );
        assert!((est_pt.x - 500.0).abs() < 1e-3 && (est_pt.y - 400.0).abs() < 1e-3 && cep_m < 15.0);

        // 3. Verify Rothermel flame length scaling & FEMA GSB/TDMS/DRC + ALERTWest Wildfire Monte Carlo
        let mild_fb = compute_rothermel_wui_fire_behavior(WuiFuelModel::BrushWuiFm5, 6.0, 72.0, 45.0, 8.0);
        let severe_fb = compute_rothermel_wui_fire_behavior(
            WuiFuelModel::ConiferTimberUnderstoryFm10,
            32.0,
            98.0,
            10.0,
            30.0,
        );
        assert!(severe_fb.flame_length_ft > mild_fb.flame_length_ft * 1.8);

        let mut rec_cfg = FemaGsbDrcTdmsConfig::default();
        rec_cfg.fema_usar.fema_pods.fema_comms.warning.transit_evac.shelters.evac.medevac.surge.ffe.seismic.monte_carlo_trials = 40;
        let adhoc_rec = world.run_fema_gsb_drc_tdms_mc(&FemaGsbDrcTdmsConfig {
            policy: FemaRecoveryExecutionPolicy::SequentialAdHoc,
            ..rec_cfg.clone()
        });
        let integ_rec = world.run_fema_gsb_drc_tdms_mc(&rec_cfg);
        assert!(integ_rec.tdms_landfill_volume_saved_yd3 > adhoc_rec.tdms_landfill_volume_saved_yd3 + 100.0);
        assert!(integ_rec.ia_registered_14d_pct >= adhoc_rec.ia_registered_14d_pct - 1e-6);
        assert!(integ_rec.to_facility_csv_string().starts_with(FEMA_RECOVERY_FACILITY_CSV_HEADER));

        let wf_911 = world.run_wildfire_alertwest_mc(&WildfireAlertWestConfig {
            posture: WildfireDetectionAndAttackPosture::Legacy911GroundOnly,
            monte_carlo_trials: 48,
            ..WildfireAlertWestConfig::default()
        });
        let wf_aw = world.run_wildfire_alertwest_mc(&WildfireAlertWestConfig {
            posture: WildfireDetectionAndAttackPosture::FullAlertWestMduAndAerialAttack,
            monte_carlo_trials: 48,
            ..WildfireAlertWestConfig::default()
        });
        assert!(wf_aw.mean_detection_to_dispatch_min < wf_911.mean_detection_to_dispatch_min * 0.35);
        assert!(wf_aw.final_burned_area_acres < wf_911.final_burned_area_acres);
        assert!(wf_aw.to_camera_csv_string().starts_with(WILDFIRE_CAMERA_CSV_HEADER));

        // 4. Verify Ctrl+Z undoable GSB power & WUI fuelbreak commands + both new GPU heatmaps
        let u0 = world.edit_history.undo_len();
        let _ = world.apply_fema_gsb_backup_power_tracked(&integ_rec);
        let _ = world.apply_wui_fuelbreak_and_fire_closures_tracked(&wf_aw);
        assert_eq!(world.edit_history.undo_len(), u0 + 2);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::FemaGsbDrcTdms, &[], &mut arena);
        assert_eq!(arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count(), HEATMAP_DIM * HEATMAP_DIM);
        build_heatmap_rgba(&world, HeatmapLayer::WildfireAlertWestResponse, &[], &mut arena);
        assert_eq!(arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count(), HEATMAP_DIM * HEATMAP_DIM);
    }

    #[test]
    fn burn_scar_hydrophobic_runoff_usgs_m1_and_culvert_washouts_work() {
        // 1. Exact USGS Staley et al. (2017) M1 P=0.50 threshold identity & Gartner (2014) volume scaling
        let i15_p50 = usgs_staley_m1_i15_p50_threshold_mm_hr(0.55, 560.0, 0.28);
        let p_at_thresh = usgs_staley_m1_debris_flow_likelihood(i15_p50 / 4.0, 0.55, 560.0, 0.28);
        assert!(
            (p_at_thresh - 0.50).abs() < 1e-9,
            "Exact M1 inversion must yield P=0.50 at I15_P50={i15_p50:.2} mm/hr"
        );
        let v_low = usgs_gartner_2014_debris_volume_m3(16.0, 0.4, 180.0);
        let v_high = usgs_gartner_2014_debris_volume_m3(48.0, 1.8, 420.0);
        assert!(v_high > v_low * 3.0);

        // 2. Hydrophobic CN runoff amplification & FHWA HDS-5 woody-plugging overtopping
        let (q_unburned_mm, q_unburned_pk) =
            nrcs_baer_hydrophobic_runoff_and_peak_q(65.0, 36.0, 1.6, 60.0, 0.04);
        let (q_hydro_mm, q_hydro_pk) =
            nrcs_baer_hydrophobic_runoff_and_peak_q(65.0, 36.0, 1.6, 94.0, 0.35);
        assert!(q_hydro_mm > q_unburned_mm * 3.0 && q_hydro_pk > q_unburned_pk * 4.0);

        let (hwd_clear, st_clear) = fhwa_hds5_culvert_headwater_ratio(2.2, 2, 1.4, 0.08);
        let (hwd_plug, st_plug) = fhwa_hds5_culvert_headwater_ratio(5.8, 1, 0.9, 0.72);
        assert!(hwd_plug > hwd_clear * 2.0);
        assert_eq!(st_clear, CulvertHydraulicState::PassableClear);
        assert_eq!(st_plug, CulvertHydraulicState::EmbankmentWashoutClosed);

        // 3. Compare Untreated Burn Scar vs Full BAER Mulch + Grizzly Racks + Box Culverts under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);
        let mut base_df = BurnScarDebrisFlowConfig {
            baer_policy: BaerWatershedMitigationPolicy::UntreatedBurnScar,
            peak_i15_intensity_mm_hr: 42.0,
            storm_6hr_depth_mm: 78.0,
            ..BurnScarDebrisFlowConfig::default()
        };
        base_df.wildfire.monte_carlo_trials = 48;

        let raw_rep = world.run_burn_scar_debris_flow_mc(&base_df);
        let baer_rep = world.run_burn_scar_debris_flow_mc(&BurnScarDebrisFlowConfig {
            baer_policy: BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing,
            ..base_df
        });
        assert!(
            baer_rep.expected_washed_out_culverts <= raw_rep.expected_washed_out_culverts + 1e-6
        );
        assert!(
            baer_rep.total_predicted_sediment_volume_m3 < raw_rep.total_predicted_sediment_volume_m3
        );
        assert!(baer_rep.to_basin_csv_string().starts_with(USGS_BASIN_CSV_HEADER));
        assert!(baer_rep.to_culvert_csv_string().starts_with(CULVERT_WASHOUT_CSV_HEADER));

        // 4. Verify Ctrl+Z undoable culvert washout + RDII sanitary sewer surcharge coupling & GPU heatmap
        let u0 = world.edit_history.undo_len();
        let _ = world.apply_culvert_washouts_and_rdii_tracked(&raw_rep);
        assert_eq!(world.edit_history.undo_len(), u0 + 1);
        assert!(world.utility_config.storm_rainfall_mm_hr > 10.0);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::BurnScarDebrisFlow, &[], &mut arena);
        assert_eq!(
            arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count(),
            HEATMAP_DIM * HEATMAP_DIM
        );
    }

    #[test]
    fn wtp_turbidity_shutoff_coagulant_and_storage_drawdown_work() {
        // 1. Verify post-fire jar-test coagulant stoichiometry & EPA IESWTR 0.30 NTU compliance physics
        let clean_jar = compute_postfire_coagulant_jar_test(4.0, 2.2, 7.5, false, false);
        let ash_alum_jar = compute_postfire_coagulant_jar_test(680.0, 18.0, 8.9, false, false);
        let ash_ferric_jar = compute_postfire_coagulant_jar_test(680.0, 18.0, 8.9, true, false);
        assert!(clean_jar.ieswtr_03ntu_compliant);
        assert!(ash_alum_jar.coagulant_dose_mg_l > clean_jar.coagulant_dose_mg_l * 3.0);
        assert!(!ash_alum_jar.ieswtr_03ntu_compliant);
        assert!(ash_ferric_jar.cfe_filtered_ntu < ash_alum_jar.cfe_filtered_ntu);
        assert!(ash_ferric_jar.pac_dose_mg_l > 10.0);

        // 2. Compare Run-to-Failure Alum vs Automated Shutoff vs Presed + FeCl3/PAC + Wells under CRN
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);

        let mut base_wtp = WtpTurbidityStorageConfig {
            policy: WtpCoagulantAndIntakePolicy::RunToFailureAlumBaseline,
            intake_shutoff_threshold_ntu: 320.0,
            finished_storage_capacity_mg: 9.5,
            ..WtpTurbidityStorageConfig::default()
        };
        base_wtp.debris_flow.peak_i15_intensity_mm_hr = 44.0;
        base_wtp.debris_flow.baer_policy = BaerWatershedMitigationPolicy::UntreatedBurnScar;
        base_wtp.debris_flow.wildfire.monte_carlo_trials = 40;

        let fail_rep = world.run_wtp_turbidity_storage_mc(&base_wtp);
        let shut_rep = world.run_wtp_turbidity_storage_mc(&WtpTurbidityStorageConfig {
            policy: WtpCoagulantAndIntakePolicy::AutomatedIntakeShutoffAlumOnly,
            ..base_wtp.clone()
        });
        let ferric_rep = world.run_wtp_turbidity_storage_mc(&WtpTurbidityStorageConfig {
            policy: WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie,
            ..base_wtp
        });

        assert!(shut_rep.intake_shutoff_duration_hrs > fail_rep.intake_shutoff_duration_hrs);
        assert!(shut_rep.ieswtr_03ntu_compliance_pct > fail_rep.ieswtr_03ntu_compliance_pct);
        assert!(ferric_rep.min_finished_storage_fill_pct > shut_rep.min_finished_storage_fill_pct);
        assert!(ferric_rep.boil_water_advisory_prob_pct <= shut_rep.boil_water_advisory_prob_pct + 1e-6);
        assert!(ferric_rep.to_timeline_csv_string().starts_with(WTP_TIMELINE_CSV_HEADER));
        assert!(ferric_rep.to_facility_csv_string().starts_with(WTP_FACILITY_CSV_HEADER));

        // 3. Verify Ctrl+Z undoable finished-water storage drawdown coupling into Hazen-Williams 20-psi grid
        let u0 = world.edit_history.undo_len();
        let _ = world.apply_wtp_storage_drawdown_tracked(&shut_rep);
        assert_eq!(world.edit_history.undo_len(), u0 + 1);
        assert!(world.undo_edit().is_some());
        assert!(world.redo_edit().is_some());

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::WtpTurbidityStorage, &[], &mut arena);
        assert_eq!(
            arena.heatmap_rgba.chunks_exact(4).filter(|px| px[3] > 0).count(),
            HEATMAP_DIM * HEATMAP_DIM
        );
    }

    #[test]
    fn test_five_distinct_city_morphologies_water_bridges_and_clipped_parcels() {
        let scenarios = [
            builtin_riverfront_bridges_osm(),
            builtin_harbor_peninsula_osm(),
            builtin_radial_ring_plazas_osm(),
            builtin_foothill_canyon_wui_osm(),
            builtin_metropolis_50k_osm(),
        ];
        let mut labels = Vec::new();
        let mut signatures = Vec::new();
        for xml in scenarios {
            let osm = parse_osm_xml_str(&xml).expect("morphology OSM compiles");
            let net = RoadNetwork::compile(&osm);
            assert!(!net.segments.is_empty());
            assert!(!net.junctions.is_empty());
            let parcels = generate_parcels_for_network(&net);
            assert!(!parcels.is_empty());
            let world = SimulationWorld::new(net);
            let mut arena = GpuStagingArena::new();
            pack_road_quads_into(&world, &mut arena.road_quads);
            let has_water_quad = arena
                .road_quads
                .iter()
                .any(|q| q.style_kind == ROAD_STYLE_WATER_CHANNEL);
            let has_bridge_quad = arena
                .road_quads
                .iter()
                .any(|q| q.style_kind == ROAD_STYLE_BRIDGE_SPAN);
            if !world.network.water_features.is_empty() {
                assert!(
                    has_water_quad,
                    "{} has water_features but packed no ROAD_STYLE_WATER_CHANNEL quads",
                    world.network.city_label
                );
            }
            if world.network.segments.iter().any(|s| s.is_bridge) {
                assert!(
                    has_bridge_quad,
                    "{} has bridge segments but packed no ROAD_STYLE_BRIDGE_SPAN quads",
                    world.network.city_label
                );
            }
            labels.push(world.network.city_label.clone());
            signatures.push((world.network.segments.len(), world.network.junctions.len()));
        }
        // Riverfront, Harbor, Foothill, and Metropolis must have bridges and water features
        for idx in [0usize, 1, 3, 4] {
            let net = RoadNetwork::compile(&parse_osm_xml_str(&match idx {
                0 => builtin_riverfront_bridges_osm(),
                1 => builtin_harbor_peninsula_osm(),
                3 => builtin_foothill_canyon_wui_osm(),
                _ => builtin_metropolis_50k_osm(),
            }).unwrap());
            assert!(!net.water_features.is_empty(), "scenario {idx} missing water");
            assert!(net.segments.iter().any(|s| s.is_bridge), "scenario {idx} missing bridge");
        }
        // All 5 labels are unique and non-empty
        for i in 0..labels.len() {
            assert!(!labels[i].is_empty());
            for j in (i + 1)..labels.len() {
                assert_ne!(labels[i], labels[j]);
            }
        }
        assert_ne!(signatures[0], signatures[2]);
    }

    #[test]
    fn test_staged_evacuation_earliest_populated_stage_departs_at_t0() {
        let xml = builtin_riverfront_bridges_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);

        let mut cfg = EvacuationContraflowConfig::default();
        cfg.staging_strategy = EvacuationStagingStrategy::FireProximityThreeStage;
        cfg.contraflow_policy = ContraflowReversalPolicy::EmsProtectedArterialContraflow;
        cfg.stage2_release_delay_min = 25.0;
        cfg.stage3_release_delay_min = 50.0;
        cfg.medevac.surge.ffe.seismic.moment_magnitude_mw = 6.2;
        cfg.medevac.surge.ffe.seismic.rupture_standoff_km = 35.0;
        cfg.medevac.surge.ffe.seismic.seismic_pipe_replacement_pct = 100.0;
        cfg.medevac.surge.ffe.seismic.monte_carlo_trials = 16;

        let report = world.run_evacuation_contraflow_mc(&cfg);
        let min_rel = report
            .parcel_rows
            .iter()
            .filter(|r| r.evac_vehicles > 0.0)
            .map(|r| r.release_delay_min)
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_rel < 1e-6,
            "earliest populated evacuation stage must depart at t=0, got min release_delay_min={min_rel}"
        );
        assert!(
            !report.timeline.is_empty() && report.timeline[0].evacuated_pct > 0.0,
            "wave 0 (t=10m) must show non-zero evacuation throughput instead of dead air"
        );
        assert!(
            report.t50_clearance_min < 35.0,
            "expected no +42..+45m dead-air inflation, got T50={:.2}m",
            report.t50_clearance_min
        );
    }

    #[test]
    fn test_staged_evacuation_bottleneck_adaptive_spacing_couples_to_capacity() {
        let xml = builtin_foothill_canyon_wui_osm();
        let net = RoadNetwork::compile(&parse_osm_xml_str(&xml).unwrap());
        let mut world = SimulationWorld::new(net);

        let mut base_cfg = EvacuationContraflowConfig::default();
        base_cfg.staging_strategy = EvacuationStagingStrategy::SimultaneousUnstaged;
        base_cfg.contraflow_policy = ContraflowReversalPolicy::NoContraflowBaseline;
        base_cfg.stage2_release_delay_min = 25.0;
        base_cfg.stage3_release_delay_min = 55.0;
        base_cfg.medevac.surge.ffe.seismic.moment_magnitude_mw = 8.2;
        base_cfg.medevac.surge.ffe.seismic.monte_carlo_trials = 24;

        let unstaged = world.run_evacuation_contraflow_mc(&base_cfg);

        let mut staged_cfg = base_cfg.clone();
        staged_cfg.staging_strategy = EvacuationStagingStrategy::FireProximityThreeStage;
        staged_cfg.contraflow_policy = ContraflowReversalPolicy::EmsProtectedArterialContraflow;
        let ems_cf = world.run_evacuation_contraflow_mc(&staged_cfg);

        assert!(
            !ems_cf.timeline.is_empty() && ems_cf.timeline[0].evacuated_pct > 0.0,
            "earliest populated stage must depart at t=0 across MC trials, yielding >0% wave-0 evacuation"
        );

        for row in &ems_cf.parcel_rows {
            assert!(
                row.release_delay_min <= staged_cfg.stage3_release_delay_min + 1e-6,
                "parcel release delay ({:.2}) exceeded stage3 cap ({:.2})",
                row.release_delay_min,
                staged_cfg.stage3_release_delay_min
            );
        }
        assert!(
            ems_cf.peak_bottleneck_vc_ratio <= unstaged.peak_bottleneck_vc_ratio + 0.05,
            "staged+CF peak V/C ({:.3}) should not degrade materially over unstaged ({:.3})",
            ems_cf.peak_bottleneck_vc_ratio,
            unstaged.peak_bottleneck_vc_ratio
        );
    }

    #[test]
    fn test_zoning_prunes_all_off_road_and_water_parcels_across_morphologies() {
        fn dist_pt_seg(p: Vec2, a: Vec2, b: Vec2) -> f64 {
            let abx = b.x - a.x;
            let aby = b.y - a.y;
            let len2 = abx * abx + aby * aby;
            if len2 <= 1e-9 {
                return p.dist(a);
            }
            let t = (((p.x - a.x) * abx + (p.y - a.y) * aby) / len2).clamp(0.0, 1.0);
            p.dist(Vec2 {
                x: a.x + t * abx,
                y: a.y + t * aby,
            })
        }
        fn pt_in_poly(p: Vec2, poly: &[Vec2]) -> bool {
            if poly.len() < 3 {
                return false;
            }
            let mut inside = false;
            let mut j = poly.len() - 1;
            for i in 0..poly.len() {
                let pi = poly[i];
                let pj = poly[j];
                let intersects = ((pi.y > p.y) != (pj.y > p.y))
                    && (p.x < (pj.x - pi.x) * (p.y - pi.y) / ((pj.y - pi.y) + 1e-12) + pi.x);
                if intersects {
                    inside = !inside;
                }
                j = i;
            }
            inside
        }

        let scenarios = [
            builtin_riverfront_bridges_osm(),
            builtin_harbor_peninsula_osm(),
            builtin_radial_ring_plazas_osm(),
            builtin_foothill_canyon_wui_osm(),
            builtin_metropolis_50k_osm(),
            builtin_roundabout_osm(),
            builtin_braess_diamond_osm(),
        ];
        for (idx, xml) in scenarios.iter().enumerate() {
            let osm = parse_osm_xml_str(xml).unwrap();
            let net = RoadNetwork::compile(&osm);
            let parcels = generate_parcels_for_network(&net);
            assert!(!parcels.is_empty(), "scenario {idx} generated 0 parcels");

            if net.city_label.to_ascii_lowercase().contains("harbor") {
                assert!(
                    parcels.iter().any(|p| p.center.x > 215.0 && p.center.y > 140.0),
                    "Harbor Port Peninsula must retain parcels"
                );
            }
            if net.city_label.to_ascii_lowercase().contains("radial") {
                assert!(
                    parcels.iter().any(|p| p.center.x.hypot(p.center.y) < 125.0),
                    "Radial central park must retain parcels"
                );
            }

            for p in &parcels {
                let min_road_d = net
                    .segments
                    .iter()
                    .filter(|s| !s.closed)
                    .flat_map(|s| s.base_pts.windows(2))
                    .map(|w| dist_pt_seg(p.center, w[0], w[1]))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    min_road_d <= 115.0 + 1e-6,
                    "scenario {idx} ({}) parcel #{} at ({:.1},{:.1}) is off-road ({min_road_d:.1} m)",
                    net.city_label,
                    p.id,
                    p.center.x,
                    p.center.y
                );

                let pts = [
                    p.center,
                    p.min,
                    Vec2 { x: p.max.x, y: p.min.y },
                    p.max,
                    Vec2 { x: p.min.x, y: p.max.y },
                    Vec2 { x: p.min.x, y: p.center.y },
                    Vec2 { x: p.max.x, y: p.center.y },
                    Vec2 { x: p.center.x, y: p.min.y },
                    Vec2 { x: p.center.x, y: p.max.y },
                ];
                for wf in &net.water_features {
                    if wf.is_polygon {
                        for &pt in &pts {
                            assert!(
                                !pt_in_poly(pt, &wf.pts),
                                "scenario {idx} parcel #{} sample point ({:.1},{:.1}) falls inside water polygon",
                                p.id, pt.x, pt.y
                            );
                        }
                    } else {
                        for &pt in &pts {
                            let d = wf
                                .pts
                                .windows(2)
                                .map(|w| dist_pt_seg(pt, w[0], w[1]))
                                .fold(f64::INFINITY, f64::min);
                            assert!(
                                d >= wf.half_width_m,
                                "scenario {idx} parcel #{} sample point overlaps linear waterway ({d:.1} < {:.1})",
                                p.id, wf.half_width_m
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_ems_response_4min_colormap_and_contours_harbor_peninsula() {
        let c0 = response_time_4min_colormap(0.0);
        let c120 = response_time_4min_colormap(120.0);
        let c240 = response_time_4min_colormap(240.0);
        let c360 = response_time_4min_colormap(360.0);
        assert_ne!(c0, c120);
        assert_ne!(c120, c240);
        assert_ne!(c240, c360);

        let osm = parse_osm_xml_str(&builtin_harbor_peninsula_osm()).unwrap();
        let mut world = SimulationWorld::new(RoadNetwork::compile(&osm));
        let mut cfg = MclpSitingConfig::default();
        cfg.max_stations_budget = 1;
        let mrep = world.solve_mclp_station_siting(&cfg);
        world.apply_mclp_siting_solution(&mrep);

        let mut arena = GpuStagingArena::new();
        build_heatmap_rgba(&world, HeatmapLayer::EmergencyResponse4Min, &[], &mut arena);

        let distinct_colors: std::collections::HashSet<[u8; 3]> = arena
            .heatmap_rgba
            .chunks_exact(4)
            .filter(|px| px[3] > 0)
            .map(|px| [px[0], px[1], px[2]])
            .collect();
        assert!(
            distinct_colors.len() > 16,
            "Harbor Peninsula EMS heatmap must have smooth continuous gradients/contours, got only {} distinct colors",
            distinct_colors.len()
        );
    }

    #[test]
    fn test_overpass_json_and_pbf_paths_offline_safe() {
        let overpass_json = r#"{
            "elements": [
                {"type":"node","id":1,"lat":40.000,"lon":-105.002},
                {"type":"node","id":2,"lat":40.000,"lon":-105.000},
                {"type":"node","id":3,"lat":40.002,"lon":-105.000},
                {"type":"way","id":10,"nodes":[1,2,3],"tags":{"highway":"primary","lanes":"2","name":"Offline Ave"}}
            ]
        }"#;
        let ds = parse_overpass_json_str(overpass_json).expect("offline Overpass JSON parses");
        let net = RoadNetwork::compile(&ds);
        assert!(!net.segments.is_empty());

        #[cfg(feature = "pbf")]
        {
            let missing_err = load_osm_path(std::path::Path::new("nonexistent_test_map_12345.pbf"));
            assert!(matches!(missing_err, Err(OsmError::Dataset(_))));

            let tmp_pbf = std::env::temp_dir().join("ug_invalid_offline_test.pbf");
            std::fs::write(&tmp_pbf, b"not_a_valid_osmpbf_file").unwrap();
            let bad_err = load_osm_path(&tmp_pbf);
            let _ = std::fs::remove_file(&tmp_pbf);
            assert!(matches!(bad_err, Err(OsmError::Dataset(_))));

            let empty_pbf = std::env::temp_dir().join("ug_empty_offline_test.pbf");
            std::fs::write(&empty_pbf, b"").unwrap();
            let empty_err = load_osm_path(&empty_pbf);
            let _ = std::fs::remove_file(&empty_pbf);
            assert!(matches!(empty_err, Err(OsmError::Dataset(_))));
        }
    }

    #[test]
    fn test_zoning_prunes_linear_waterway_quarter_edge_overlap() {
        fn dist_pt_seg(p: Vec2, a: Vec2, b: Vec2) -> f64 {
            let abx = b.x - a.x;
            let aby = b.y - a.y;
            let len2 = abx * abx + aby * aby;
            if len2 <= 1e-9 {
                return p.dist(a);
            }
            let t = (((p.x - a.x) * abx + (p.y - a.y) * aby) / len2).clamp(0.0, 1.0);
            p.dist(Vec2 {
                x: a.x + t * abx,
                y: a.y + t * aby,
            })
        }
        fn dist_pt_rect(p: Vec2, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> f64 {
            let dx = if p.x < min_x {
                min_x - p.x
            } else if p.x > max_x {
                p.x - max_x
            } else {
                0.0
            };
            let dy = if p.y < min_y {
                min_y - p.y
            } else if p.y > max_y {
                p.y - max_y
            } else {
                0.0
            };
            dx.hypot(dy)
        }
        fn seg_hits_rect_local(
            a: Vec2,
            b: Vec2,
            min_x: f64,
            max_x: f64,
            min_y: f64,
            max_y: f64,
        ) -> bool {
            let mut t0 = 0.0_f64;
            let mut t1 = 1.0_f64;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            for (p, q) in [
                (-dx, a.x - min_x),
                (dx, max_x - a.x),
                (-dy, a.y - min_y),
                (dy, max_y - a.y),
            ] {
                if p.abs() <= 1e-12 {
                    if q < 0.0 {
                        return false;
                    }
                } else {
                    let r = q / p;
                    if p < 0.0 {
                        t0 = t0.max(r);
                    } else {
                        t1 = t1.min(r);
                    }
                    if t0 > t1 {
                        return false;
                    }
                }
            }
            true
        }
        fn exact_seg_to_rect(a: Vec2, b: Vec2, p: &Parcel) -> f64 {
            if seg_hits_rect_local(a, b, p.min.x, p.max.x, p.min.y, p.max.y) {
                return 0.0;
            }
            let tl = p.min;
            let tr = Vec2 { x: p.max.x, y: p.min.y };
            let br = p.max;
            let bl = Vec2 { x: p.min.x, y: p.max.y };
            dist_pt_rect(a, p.min.x, p.max.x, p.min.y, p.max.y)
                .min(dist_pt_rect(b, p.min.x, p.max.x, p.min.y, p.max.y))
                .min(dist_pt_seg(tl, a, b))
                .min(dist_pt_seg(tr, a, b))
                .min(dist_pt_seg(br, a, b))
                .min(dist_pt_seg(bl, a, b))
        }

        // 1. Adversarial quarter-edge linear waterway tip (27.0 m outside edge, half_width_m = 30.0)
        let osm = parse_osm_xml_str(&builtin_roundabout_osm()).unwrap();
        let mut net = RoadNetwork::compile(&osm);
        let base_parcels = generate_parcels_for_network(&net);
        assert!(!base_parcels.is_empty());
        let p0 = &base_parcels[0];
        let pw = p0.max.x - p0.min.x;
        let tip_x = p0.min.x + 0.25 * pw;
        let tip_y = p0.min.y - 27.0;
        let wf = WaterFeature {
            name: "Quarter-Edge River Tip".to_string(),
            kind: "river".to_string(),
            pts: vec![
                Vec2 {
                    x: tip_x,
                    y: tip_y - 80.0,
                },
                Vec2 { x: tip_x, y: tip_y },
            ],
            half_width_m: 30.0,
            is_polygon: false,
        };
        net.water_features.push(wf);
        let pruned_parcels = generate_parcels_for_network(&net);
        assert!(!pruned_parcels.is_empty());
        assert!(
            pruned_parcels.len() < base_parcels.len(),
            "adversarial quarter-edge parcel must be pruned (base={}, pruned={})",
            base_parcels.len(),
            pruned_parcels.len()
        );
        for p in &pruned_parcels {
            for wf in &net.water_features {
                if !wf.is_polygon {
                    let d = wf
                        .pts
                        .windows(2)
                        .map(|w| exact_seg_to_rect(w[0], w[1], p))
                        .fold(f64::INFINITY, f64::min);
                    assert!(
                        d >= wf.half_width_m,
                        "parcel #{} overlaps linear waterway '{}' ({:.2}m < {:.2}m)",
                        p.id,
                        wf.name,
                        d,
                        wf.half_width_m
                    );
                }
            }
        }

        // 2. Verify all 7 built-in cities retain healthy non-empty parcels with 0 exact linear water overlaps
        let scenarios = [
            builtin_riverfront_bridges_osm(),
            builtin_harbor_peninsula_osm(),
            builtin_radial_ring_plazas_osm(),
            builtin_foothill_canyon_wui_osm(),
            builtin_metropolis_50k_osm(),
            builtin_roundabout_osm(),
            builtin_braess_diamond_osm(),
        ];
        for (idx, xml) in scenarios.iter().enumerate() {
            let net = RoadNetwork::compile(&parse_osm_xml_str(xml).unwrap());
            let parcels = generate_parcels_for_network(&net);
            assert!(!parcels.is_empty(), "scenario {idx} generated 0 parcels");
            if net.city_label.to_ascii_lowercase().contains("harbor") {
                assert!(
                    parcels.iter().any(|p| p.center.x > 215.0 && p.center.y > 140.0),
                    "Harbor Port Peninsula must retain parcels"
                );
            }
            if net.city_label.to_ascii_lowercase().contains("radial") {
                assert!(
                    parcels.iter().any(|p| p.center.x.hypot(p.center.y) < 125.0),
                    "Radial central park must retain parcels"
                );
            }
            for p in &parcels {
                for wf in &net.water_features {
                    if !wf.is_polygon {
                        let d = wf
                            .pts
                            .windows(2)
                            .map(|w| exact_seg_to_rect(w[0], w[1], p))
                            .fold(f64::INFINITY, f64::min);
                        assert!(
                            d >= wf.half_width_m,
                            "scenario {idx} parcel #{} overlaps linear waterway '{}' ({:.2}m < {:.2}m)",
                            p.id,
                            wf.name,
                            d,
                            wf.half_width_m
                        );
                    }
                }
            }
        }
    }
}

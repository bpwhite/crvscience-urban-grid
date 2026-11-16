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

#![cfg(feature = "gui")]

use crate::batch::parse_demand_scales_arg;
use crate::wgpu_pass::{
    UrbanViewportGpuResources, UrbanViewportPaintCallback, ViewportFramePayload,
};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2 as EguiVec2};
use eframe::egui_wgpu::{self, wgpu};
use rayon::ThreadPoolBuilder;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use ug_core::{
    build_heatmap_rgba, builtin_braess_diamond_osm, builtin_foothill_canyon_wui_osm,
    builtin_harbor_peninsula_osm, builtin_metropolis_50k_osm, builtin_radial_ring_plazas_osm,
    builtin_riverfront_bridges_osm, builtin_roundabout_osm, convert_network_candidate_roundabouts,
    run_parallel_batch_experiment_with_progress, BatchControlRegime, BatchExperimentConfig,
    BatchExperimentReport, BatchProgressTracker, EditCategory, EmergencyFacilityKind,
    BusynessModel, DisruptionInterdictionDepth, HypercubeConfig, HypercubeQueueDiscipline,
    AeromedicalMedevacConfig, AeromedicalMedevacReport, AirFiltrationStandard,
    AlertChannelMix, AmbulanceDiversionPolicy, BusContraflowLaneMode,
    CommodityResupplyConfig, CommodityResupplyReport, ContraflowReversalPolicy,
    DebrisAndUsarPolicy, DepartureStagingPolicy, DiurnalDispatchRegime, EocIpaswUplinkMode,
    FemaAssetKind, FemaCommsConfig, FemaCommsPosture, FemaCommsReport,
    FemaInsaragMarking, FemaIsbPodsConfig, FemaIsbPodsReport, FemaPodClassification,
    FemaGsbDrcTdmsConfig, FemaGsbDrcTdmsReport, FemaRecoveryExecutionPolicy,
    FemaRecoverySiteKind, FemaUsarDebrisConfig, FemaUsarDebrisReport, IpaswPodRoutingPolicy,
    builtin_western_wildfire_camera_feed_json, load_wildfire_camera_feed_path,
    BaerWatershedMitigationPolicy, BurnScarDebrisFlowConfig, BurnScarDebrisFlowReport,
    CulvertHydraulicState, UsgsCombinedHazardRating, WaterSupplyAssetKind,
    WtpCoagulantAndIntakePolicy, WtpTurbidityStorageConfig, WtpTurbidityStorageReport,
    WildfireAlertWestConfig, WildfireAlertWestReport, WildfireCameraKind,
    WildfireDetectionAndAttackPosture, WuiFuelModel, IsbReplenishmentPolicy,
    ReliefCommodityKind, ResupplyDispatchPolicy, TelecomBackupPowerStandard,
    TransitDispatchStrategy, TransitEvacuationConfig, TransitEvacuationReport,
    WarningMobilizationConfig, WarningMobilizationReport, DiurnalProfileConfig,
    DiurnalRedistrictingReport, DiurnalWindowPreset, EmergencySheltersConfig,
    EmergencySheltersReport, EvacuationContraflowConfig, EvacuationContraflowReport,
    EvacuationStagingStrategy, ShelterSitingAndRoutingPolicy, FfeConfig, FfeIgnitionModel,
    FfeMonteCarloReport, HelipadStructuralType, HospitalSurgeConfig,
    HospitalSurgeMonteCarloReport, HospitalTraumaDesignation, HypercubeReport,
    MclpOptimizationReport, MclpSitingConfig, MedevacAvionicsRegime, RedistrictingConfig,
    RotorcraftAirframeClass, RedistrictingObjectiveMode, RedistrictingReport, MclpWeightingMode,
    SitingObjectiveModel, BridgeStructuralArchetype, BuriedPipeMaterialClass,
    PipelineFragilityLaw, ResilienceStressConfig, ResilienceStressReport,
    sample_ala_hazus_pgv_curves, sample_hazus_restoration_curves, SeismicMonteCarloReport,
    SeismicScenarioConfig, pack_agents_into, pack_road_quads_into, parse_osm_xml_str, ControlMode,
    GpuCameraUniform, BoardingFareMode, BusCoachSpec, GpuPlumeComputeUniform, GpuStagingArena,
    HeatmapLayer, adjust_segment_lane_count, convert_junction_to_roundabout, draw_road_segment,
    hit_test_junction, hit_test_segment, preview_draw_snap, MpPhaseId, NewRoadSpec,
    PlumePhysicsParams, RoadNetwork, SimulationWorld, SnapTarget, UtilityAssetKind,
    UtilityHeatmapSublayer, Vec2 as WorldVec2, ZoneCode,
};

const FIXED_PHYSICS_DT: f64 = 1.0 / 60.0;

#[derive(PartialEq, Eq, Clone, Copy)]
enum ViewportTool {
    Inspect,
    DrawRoad,
    EditLanes,
    ConvertRoundabout,
    PaintZone,
    PlaceFacility,
    ToggleClosure,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum DockTab {
    SignalsAndRoads,
    ZoningAndTransit,
    IdmMobilAndCrowd,
    CityAndShockwave,
    BatchSweep,
    ResilienceStress,
    UtilityGrids,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum StressSubMode {
    DeterministicCutSets,
    SeismicMonteCarlo,
    FireFollowingQuake,
    HospitalTraumaSurge,
    AeromedicalMedevac,
    EvacuationContraflow,
    EmergencyShelters,
    CommodityResupply,
    TransitBusEvacuation,
    WarningMobilization,
    FemaCommsArchitecture,
    FemaIsbPods,
    FemaUsarDebris,
    FemaGsbDrcTdms,
    WildfireAlertWestResponse,
    BurnScarDebrisFlow,
    WtpTurbidityStorage,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum GuiScreen {
    MainEntranceHub,
    ActiveWorkbench,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum ResilienceCategoryFilter {
    All17,
    HazardAndMedical,
    EvacAndSustainment,
    FemaOperations,
    WildfireAndWui,
}

impl ResilienceCategoryFilter {
    fn contains(self, mode: StressSubMode) -> bool {
        match self {
            Self::All17 => true,
            Self::HazardAndMedical => matches!(
                mode,
                StressSubMode::DeterministicCutSets
                    | StressSubMode::SeismicMonteCarlo
                    | StressSubMode::FireFollowingQuake
                    | StressSubMode::BurnScarDebrisFlow
                    | StressSubMode::WtpTurbidityStorage
                    | StressSubMode::HospitalTraumaSurge
                    | StressSubMode::AeromedicalMedevac
            ),
            Self::EvacAndSustainment => matches!(
                mode,
                StressSubMode::EvacuationContraflow
                    | StressSubMode::TransitBusEvacuation
                    | StressSubMode::EmergencyShelters
                    | StressSubMode::CommodityResupply
            ),
            Self::FemaOperations => matches!(
                mode,
                StressSubMode::WarningMobilization
                    | StressSubMode::FemaCommsArchitecture
                    | StressSubMode::FemaIsbPods
                    | StressSubMode::FemaUsarDebris
                    | StressSubMode::FemaGsbDrcTdms
            ),
            Self::WildfireAndWui => matches!(
                mode,
                StressSubMode::WildfireAlertWestResponse
                    | StressSubMode::BurnScarDebrisFlow
                    | StressSubMode::WtpTurbidityStorage
                    | StressSubMode::FireFollowingQuake
                    | StressSubMode::EvacuationContraflow
                    | StressSubMode::EmergencyShelters
            ),
        }
    }

    fn for_mode(mode: StressSubMode) -> Self {
        if matches!(
            mode,
            StressSubMode::WildfireAlertWestResponse
                | StressSubMode::BurnScarDebrisFlow
                | StressSubMode::WtpTurbidityStorage
        ) {
            Self::WildfireAndWui
        } else if Self::HazardAndMedical.contains(mode) {
            Self::HazardAndMedical
        } else if Self::EvacAndSustainment.contains(mode) {
            Self::EvacAndSustainment
        } else {
            Self::FemaOperations
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum SeismicCurveView {
    FragilitySa10,
    HazusRestoration90d,
    AlaPgvPipeFragility,
}

struct ActiveGuiSweepHandle {
    rx: Receiver<Result<Option<BatchExperimentReport>, String>>,
    tracker: Arc<BatchProgressTracker>,
    worker_threads: usize,
}

pub struct UrbanGridGuiApp {
    world: SimulationWorld,
    active_city_name: String,
    gpu_ready: bool,
    staging: GpuStagingArena,
    paused: bool,
    speed_scale: f64,
    physics_accum_s: f64,
    plume_params: PlumePhysicsParams,
    plume_reset_pending: bool,
    heatmap_dirty: bool,
    layer: HeatmapLayer,
    tool: ViewportTool,
    draw_spec: NewRoadSpec,
    draw_start_world: Option<WorldVec2>,
    symmetric_lane_edit: bool,
    roundabout_radius_m: f64,
    roundabout_lanes: usize,
    active_facility_kind: EmergencyFacilityKind,
    coverage_refresh_accum_s: f64,
    mclp_cfg: MclpSitingConfig,
    latest_mclp_report: Option<MclpOptimizationReport>,
    mclp_csv_path: String,
    hypercube_cfg: HypercubeConfig,
    latest_hypercube_report: Option<HypercubeReport>,
    hypercube_csv_path: String,
    redistrict_cfg: RedistrictingConfig,
    latest_redistricting_report: Option<RedistrictingReport>,
    redistrict_csv_path: String,
    redistrict_pareto_csv_path: String,
    diurnal_cfg: DiurnalProfileConfig,
    latest_diurnal_report: Option<DiurnalRedistrictingReport>,
    active_diurnal_regime: DiurnalDispatchRegime,
    diurnal_hourly_csv_path: String,
    diurnal_table_csv_path: String,
    utility_csv_path: String,
    stress_sub_mode: StressSubMode,
    stress_cfg: ResilienceStressConfig,
    latest_stress_report: Option<ResilienceStressReport>,
    stress_csv_path: String,
    seismic_cfg: SeismicScenarioConfig,
    latest_seismic_report: Option<SeismicMonteCarloReport>,
    seismic_csv_path: String,
    recovery_csv_path: String,
    pipeline_csv_path: String,
    ffe_cfg: FfeConfig,
    latest_ffe_report: Option<FfeMonteCarloReport>,
    ffe_csv_path: String,
    surge_cfg: HospitalSurgeConfig,
    latest_surge_report: Option<HospitalSurgeMonteCarloReport>,
    surge_ab_summary: Option<String>,
    surge_hosp_csv_path: String,
    medevac_cfg: AeromedicalMedevacConfig,
    latest_medevac_report: Option<AeromedicalMedevacReport>,
    medevac_ab_summary: Option<String>,
    medevac_csv_path: String,
    evac_cfg: EvacuationContraflowConfig,
    latest_evac_report: Option<EvacuationContraflowReport>,
    evac_ab_summary: Option<String>,
    evac_csv_path: String,
    shelters_cfg: EmergencySheltersConfig,
    latest_shelters_report: Option<EmergencySheltersReport>,
    shelters_ab_summary: Option<String>,
    shelters_csv_path: String,
    resupply_cfg: CommodityResupplyConfig,
    latest_resupply_report: Option<CommodityResupplyReport>,
    resupply_ab_summary: Option<String>,
    resupply_csv_path: String,
    transit_evac_cfg: TransitEvacuationConfig,
    latest_transit_evac_report: Option<TransitEvacuationReport>,
    transit_evac_ab_summary: Option<String>,
    transit_evac_csv_path: String,
    warning_cfg: WarningMobilizationConfig,
    latest_warning_report: Option<WarningMobilizationReport>,
    warning_ab_summary: Option<String>,
    warning_csv_path: String,
    fema_comms_cfg: FemaCommsConfig,
    latest_fema_comms_report: Option<FemaCommsReport>,
    fema_comms_ab_summary: Option<String>,
    fema_comms_csv_path: String,
    fema_pods_cfg: FemaIsbPodsConfig,
    latest_fema_pods_report: Option<FemaIsbPodsReport>,
    fema_pods_ab_summary: Option<String>,
    fema_pods_csv_path: String,
    fema_usar_cfg: FemaUsarDebrisConfig,
    latest_fema_usar_report: Option<FemaUsarDebrisReport>,
    fema_usar_ab_summary: Option<String>,
    fema_usar_csv_path: String,
    fema_rec_cfg: FemaGsbDrcTdmsConfig,
    latest_fema_rec_report: Option<FemaGsbDrcTdmsReport>,
    fema_rec_ab_summary: Option<String>,
    fema_rec_csv_path: String,
    wildfire_cfg: WildfireAlertWestConfig,
    latest_wildfire_report: Option<WildfireAlertWestReport>,
    wildfire_ab_summary: Option<String>,
    wildfire_feed_path: String,
    wildfire_csv_path: String,
    debris_flow_cfg: BurnScarDebrisFlowConfig,
    latest_debris_flow_report: Option<BurnScarDebrisFlowReport>,
    debris_flow_ab_summary: Option<String>,
    debris_basin_csv_path: String,
    debris_culvert_csv_path: String,
    wtp_turb_cfg: WtpTurbidityStorageConfig,
    latest_wtp_turb_report: Option<WtpTurbidityStorageReport>,
    wtp_turb_ab_summary: Option<String>,
    wtp_turb_csv_path: String,
    fragility_plot_arch: BridgeStructuralArchetype,
    seismic_curve_view: SeismicCurveView,
    selected_recovery_epoch_idx: usize,
    tab: DockTab,
    active_brush: ZoneCode,
    selected_junction: usize,
    selected_segment: usize,
    cam_x: f64,
    cam_y: f64,
    cam_zoom: f64,
    target_crowd_slider: usize,
    ema_frame_ms: f64,
    snapshot_path: String,
    csv_export_path: String,
    save_include_agents: bool,
    io_status_msg: String,
    sweep_cfg: BatchExperimentConfig,
    sweep_scales_input: String,
    sweep_bg_threads: usize,
    sweep_max_threads: usize,
    sweep_csv_path: String,
    active_sweep: Option<ActiveGuiSweepHandle>,
    latest_sweep_report: Option<BatchExperimentReport>,
    screen: GuiScreen,
    resilience_cat_filter: ResilienceCategoryFilter,
    hub_search_query: String,
    command_palette_open: bool,
    command_palette_query: String,
    show_cad_toolbar: bool,
}

impl UrbanGridGuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>, world: SimulationWorld) -> Self {
        Self::new_with_wgpu_state(cc.wgpu_render_state.as_ref(), world)
    }

    pub fn new_with_wgpu_state(
        wgpu_render_state: Option<&egui_wgpu::RenderState>,
        mut world: SimulationWorld,
    ) -> Self {
        let mut gpu_ready = false;
        if let Some(wgpu_state) = wgpu_render_state {
            let resources = UrbanViewportGpuResources::new(
                &wgpu_state.device,
                &wgpu_state.queue,
                wgpu_state.target_format,
            );
            wgpu_state
                .renderer
                .write()
                .callback_resources
                .insert(resources);
            gpu_ready = true;
        }

        // Seed staged ready traffic cleanly at t = 0.0 s without running physics ahead of user launch.
        if world.total_active_agents() < 200 {
            world.seed_initial_traffic(280);
        }
        let b = world.network.bounds;
        let span = (b.max_x - b.min_x).max(b.max_y - b.min_y).max(200.0);
        let init_zoom = (760.0 / span).clamp(0.32, 2.2);
        let selected_segment = world.watched_segment_id;
        let target_crowd_slider = world.total_active_agents().max(52_000);
        let active_city_name = world.network.city_label.clone();

        Self {
            cam_x: 0.5 * (b.min_x + b.max_x),
            cam_y: 0.5 * (b.min_y + b.max_y),
            cam_zoom: init_zoom,
            world,
            active_city_name,
            gpu_ready,
            staging: GpuStagingArena::new(),
            paused: true,
            speed_scale: 1.0,
            physics_accum_s: 0.0,
            plume_params: PlumePhysicsParams::default(),
            plume_reset_pending: true,
            heatmap_dirty: true,
            layer: HeatmapLayer::Zoning,
            tool: ViewportTool::Inspect,
            draw_spec: NewRoadSpec::default(),
            draw_start_world: None,
            symmetric_lane_edit: true,
            roundabout_radius_m: 20.0,
            roundabout_lanes: 2,
            active_facility_kind: EmergencyFacilityKind::FireStation,
            coverage_refresh_accum_s: 0.0,
            mclp_cfg: MclpSitingConfig {
                objective_model: SitingObjectiveModel::DaskinMexclp1983,
                ..MclpSitingConfig::default()
            },
            latest_mclp_report: None,
            mclp_csv_path: "exports/mexclp_pareto_frontier.csv".to_string(),
            hypercube_cfg: HypercubeConfig::default(),
            latest_hypercube_report: None,
            hypercube_csv_path: "exports/larson_hypercube_units.csv".to_string(),
            redistrict_cfg: RedistrictingConfig::default(),
            latest_redistricting_report: None,
            redistrict_csv_path: "exports/carter_larson_districts.csv".to_string(),
            redistrict_pareto_csv_path: "exports/carter_larson_pareto.csv".to_string(),
            diurnal_cfg: DiurnalProfileConfig::default(),
            latest_diurnal_report: None,
            active_diurnal_regime: DiurnalDispatchRegime::PeakRush,
            diurnal_hourly_csv_path: "exports/diurnal_24hr_profile.csv".to_string(),
            diurnal_table_csv_path: "exports/diurnal_peak_offpeak_run_cards.csv".to_string(),
            utility_csv_path: "exports/utility_lifelines_telemetry.csv".to_string(),
            stress_sub_mode: StressSubMode::SeismicMonteCarlo,
            stress_cfg: ResilienceStressConfig::default(),
            latest_stress_report: None,
            stress_csv_path: "exports/resilience_stress_scorecard.csv".to_string(),
            seismic_cfg: SeismicScenarioConfig::default(),
            latest_seismic_report: None,
            seismic_csv_path: "exports/seismic_bridge_fragility_mc.csv".to_string(),
            recovery_csv_path: "exports/hazus_recovery_curve_90d.csv".to_string(),
            pipeline_csv_path: "exports/ala_pipeline_fragility_mc.csv".to_string(),
            ffe_cfg: FfeConfig::default(),
            latest_ffe_report: None,
            ffe_csv_path: "exports/ffe_conflagration_risk.csv".to_string(),
            surge_cfg: HospitalSurgeConfig::default(),
            latest_surge_report: None,
            surge_ab_summary: None,
            surge_hosp_csv_path: "exports/hospital_ed_surge_telemetry.csv".to_string(),
            medevac_cfg: AeromedicalMedevacConfig::default(),
            latest_medevac_report: None,
            medevac_ab_summary: None,
            medevac_csv_path: "exports/medevac_helipad_burn_transfers.csv".to_string(),
            evac_cfg: EvacuationContraflowConfig::default(),
            latest_evac_report: None,
            evac_ab_summary: None,
            evac_csv_path: "exports/staged_evacuation_contraflow.csv".to_string(),
            shelters_cfg: EmergencySheltersConfig::default(),
            latest_shelters_report: None,
            shelters_ab_summary: None,
            shelters_csv_path: "exports/emergency_shelters_hepa.csv".to_string(),
            resupply_cfg: CommodityResupplyConfig::default(),
            latest_resupply_report: None,
            resupply_ab_summary: None,
            resupply_csv_path: "exports/commodity_resupply_vrp.csv".to_string(),
            transit_evac_cfg: TransitEvacuationConfig::default(),
            latest_transit_evac_report: None,
            transit_evac_ab_summary: None,
            transit_evac_csv_path: "exports/transit_bus_evac_depots.csv".to_string(),
            warning_cfg: WarningMobilizationConfig::default(),
            latest_warning_report: None,
            warning_ab_summary: None,
            warning_csv_path: "exports/cellular_bts_warning_curves.csv".to_string(),
            fema_comms_cfg: FemaCommsConfig::default(),
            latest_fema_comms_report: None,
            fema_comms_ab_summary: None,
            fema_comms_csv_path: "exports/fema_ipaws_mers_assets.csv".to_string(),
            fema_pods_cfg: FemaIsbPodsConfig::default(),
            latest_fema_pods_report: None,
            fema_pods_ab_summary: None,
            fema_pods_csv_path: "exports/fema_isb_county_pods.csv".to_string(),
            fema_usar_cfg: FemaUsarDebrisConfig::default(),
            latest_fema_usar_report: None,
            fema_usar_ab_summary: None,
            fema_usar_csv_path: "exports/fema_usar_usng_debris.csv".to_string(),
            fema_rec_cfg: FemaGsbDrcTdmsConfig::default(),
            latest_fema_rec_report: None,
            fema_rec_ab_summary: None,
            fema_rec_csv_path: "exports/fema_gsb_tdms_drc_sites.csv".to_string(),
            wildfire_cfg: WildfireAlertWestConfig::default(),
            latest_wildfire_report: None,
            wildfire_ab_summary: None,
            wildfire_feed_path: "data/western_wildfire_cameras.json".to_string(),
            wildfire_csv_path: "exports/western_wildfire_cameras.csv".to_string(),
            debris_flow_cfg: BurnScarDebrisFlowConfig::default(),
            latest_debris_flow_report: None,
            debris_flow_ab_summary: None,
            debris_basin_csv_path: "exports/usgs_postfire_debris_basins.csv".to_string(),
            debris_culvert_csv_path: "exports/fhwa_hds5_culvert_washouts.csv".to_string(),
            wtp_turb_cfg: WtpTurbidityStorageConfig::default(),
            latest_wtp_turb_report: None,
            wtp_turb_ab_summary: None,
            wtp_turb_csv_path: "exports/wtp_turbidity_coagulant_drawdown_72h.csv".to_string(),
            fragility_plot_arch: BridgeStructuralArchetype::Pre1975MultiSpanSimplySupported,
            seismic_curve_view: SeismicCurveView::HazusRestoration90d,
            selected_recovery_epoch_idx: 0,
            tab: DockTab::SignalsAndRoads,
            active_brush: ZoneCode::MU4,
            selected_junction: 0,
            selected_segment,
            target_crowd_slider,
            ema_frame_ms: 8.33,
            snapshot_path: "exports/scenario_snapshot.json".to_string(),
            csv_export_path: "exports/telemetry_timeseries.csv".to_string(),
            save_include_agents: true,
            io_status_msg: String::new(),
            sweep_cfg: BatchExperimentConfig {
                duration_s: 120.0,
                warmup_s: 25.0,
                ..BatchExperimentConfig::default()
            },
            sweep_scales_input: "0.75, 1.00, 1.35".to_string(),
            sweep_bg_threads: thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2).max(1))
                .unwrap_or(2),
            sweep_max_threads: thread::available_parallelism()
                .map(|n| n.get().max(1))
                .unwrap_or(8),
            sweep_csv_path: "exports/gui_batch_scorecard.csv".to_string(),
            active_sweep: None,
            latest_sweep_report: None,
            screen: GuiScreen::MainEntranceHub,
            resilience_cat_filter: ResilienceCategoryFilter::All17,
            hub_search_query: String::new(),
            command_palette_open: false,
            command_palette_query: String::new(),
            show_cad_toolbar: true,
        }
    }

    fn launch_background_sweep(&mut self) {
        if self.active_sweep.is_some() {
            return;
        }
        let scales = match parse_demand_scales_arg(&self.sweep_scales_input) {
            Ok(v) => v,
            Err(e) => {
                self.io_status_msg = format!("Sweep config error: {e}");
                return;
            }
        };
        self.sweep_cfg.demand_multipliers = scales;

        // Clone only the clean topological, zoning, GTFS, and physics state (stripping transient
        // live crowd buffers so launching a sweep from 50k+ Megacity mode takes <0.2 ms on the UI thread!).
        let mut base_template = self.world.clone();
        base_template.vehicles.clear();
        base_template.completed_trips.clear();
        base_template.xt_history.clear();
        base_template.telemetry_log.clear();
        base_template.edit_history.clear();
        base_template.sim_time_s = 0.0;
        base_template.cyclists.truncate(400);
        base_template.pedestrians.truncate(600);
        base_template.max_vehicles = (base_template.network.lanes.len() * 16).clamp(500, 6_000);

        let cfg = self.sweep_cfg.clone();
        let bg_threads = self.sweep_bg_threads.clamp(1, self.sweep_max_threads);
        let tracker = Arc::new(BatchProgressTracker::new());
        let tracker_bg = Arc::clone(&tracker);
        let (tx, rx) = mpsc::channel();

        let spawn_res = thread::Builder::new()
            .name("ug-bg-sweep".to_string())
            .spawn(move || {
                let res = ThreadPoolBuilder::new()
                    .num_threads(bg_threads)
                    .thread_name(|idx| format!("ug-sweep-w{idx}"))
                    .build()
                    .map_err(|e| format!("Failed to build background Rayon pool: {e}"))
                    .map(|pool| {
                        pool.install(|| {
                            run_parallel_batch_experiment_with_progress(
                                &base_template,
                                &cfg,
                                Some(&tracker_bg),
                            )
                        })
                    });
                let _ = tx.send(res);
            });

        match spawn_res {
            Ok(_) => {
                self.active_sweep = Some(ActiveGuiSweepHandle {
                    rx,
                    tracker,
                    worker_threads: bg_threads,
                });
                self.io_status_msg = format!(
                    "Launched async Rayon sweep on {bg_threads} background threads..."
                );
            }
            Err(e) => {
                self.io_status_msg = format!("Thread spawn error: {e}");
            }
        }
    }

    fn poll_background_sweep(&mut self) {
        let Some(handle) = self.active_sweep.as_ref() else {
            return;
        };
        match handle.rx.try_recv() {
            Ok(Ok(Some(report))) => {
                self.io_status_msg = format!(
                    "Sweep finished: {} runs in {:.0} ms ({:.0}x real-time)",
                    report.total_parallel_runs, report.wall_clock_ms, report.realtime_speedup_x
                );
                self.latest_sweep_report = Some(report);
                self.active_sweep = None;
            }
            Ok(Ok(None)) => {
                self.io_status_msg = "Background Rayon sweep cancelled.".to_string();
                self.active_sweep = None;
            }
            Ok(Err(err_msg)) => {
                self.io_status_msg = format!("Sweep failed: {err_msg}");
                self.active_sweep = None;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.io_status_msg = "Background sweep worker disconnected.".to_string();
                self.active_sweep = None;
            }
        }
    }

    fn clamp_selection_indices(&mut self) {
        self.selected_junction = self
            .selected_junction
            .min(self.world.network.junctions.len().saturating_sub(1));
        self.selected_segment = self
            .selected_segment
            .min(self.world.network.segments.len().saturating_sub(1));
    }

    fn trigger_undo(&mut self) {
        if let Some(out) = self.world.undo_edit() {
            self.draw_start_world = None;
            self.clamp_selection_indices();
            self.heatmap_dirty = true;
            self.io_status_msg = format!(
                "Undid [{}]: {} ({} undo / {} redo)",
                out.category.short_tag(),
                out.label,
                out.undo_remaining,
                out.redo_remaining
            );
        } else {
            self.io_status_msg = "Undo stack is empty.".to_string();
        }
    }

    fn trigger_redo(&mut self) {
        if let Some(out) = self.world.redo_edit() {
            self.draw_start_world = None;
            self.clamp_selection_indices();
            self.heatmap_dirty = true;
            self.io_status_msg = format!(
                "Redid [{}]: {} ({} undo / {} redo)",
                out.category.short_tag(),
                out.label,
                out.undo_remaining,
                out.redo_remaining
            );
        } else {
            self.io_status_msg = "Redo stack is empty.".to_string();
        }
    }

    fn load_builtin(&mut self, xml: &str, megacity_crowd: Option<usize>) {
        if let Ok(osm) = parse_osm_xml_str(xml) {
            let net = RoadNetwork::compile(&osm);
            let mut w = SimulationWorld::new(net);
            if let Some(crowd_n) = megacity_crowd {
                w.seed_50k_megacity_crowd(crowd_n);
                self.layer = HeatmapLayer::AgentDensity;
            } else {
                w.seed_initial_traffic(280);
            }
            let b = w.network.bounds;
            let span = (b.max_x - b.min_x).max(b.max_y - b.min_y).max(200.0);
            self.cam_x = 0.5 * (b.min_x + b.max_x);
            self.cam_y = 0.5 * (b.min_y + b.max_y);
            self.cam_zoom = (760.0 / span).clamp(0.28, 2.2);
            self.selected_junction = 0;
            self.selected_segment = w.watched_segment_id;
            self.draw_start_world = None;
            self.latest_mclp_report = None;
            self.latest_hypercube_report = None;
            self.latest_redistricting_report = None;
            self.latest_diurnal_report = None;
            self.latest_ffe_report = None;
            self.latest_surge_report = None;
            self.surge_ab_summary = None;
            self.latest_medevac_report = None;
            self.medevac_ab_summary = None;
            self.latest_evac_report = None;
            self.evac_ab_summary = None;
            self.latest_shelters_report = None;
            self.shelters_ab_summary = None;
            self.latest_resupply_report = None;
            self.resupply_ab_summary = None;
            self.latest_transit_evac_report = None;
            self.transit_evac_ab_summary = None;
            self.latest_warning_report = None;
            self.warning_ab_summary = None;
            self.latest_fema_comms_report = None;
            self.fema_comms_ab_summary = None;
            self.latest_fema_pods_report = None;
            self.fema_pods_ab_summary = None;
            self.latest_fema_usar_report = None;
            self.fema_usar_ab_summary = None;
            self.latest_fema_rec_report = None;
            self.fema_rec_ab_summary = None;
            self.latest_wildfire_report = None;
            self.wildfire_ab_summary = None;
            self.latest_debris_flow_report = None;
            self.debris_flow_ab_summary = None;
            self.latest_wtp_turb_report = None;
            self.wtp_turb_ab_summary = None;
            self.latest_stress_report = None;
            self.latest_seismic_report = None;
            self.physics_accum_s = 0.0;
            self.plume_reset_pending = true;
            self.heatmap_dirty = true;
            self.paused = true;
            self.active_city_name = w.network.city_label.clone();
            self.world = w;
        }
    }

    fn enter_workbench(
        &mut self,
        tab: DockTab,
        sub_mode: Option<StressSubMode>,
        layer: HeatmapLayer,
        tool: ViewportTool,
    ) {
        self.tab = tab;
        if let Some(sm) = sub_mode {
            self.stress_sub_mode = sm;
            if !self.resilience_cat_filter.contains(sm) {
                self.resilience_cat_filter = ResilienceCategoryFilter::for_mode(sm);
            }
        }
        if self.layer != layer {
            self.layer = layer;
            self.heatmap_dirty = true;
            self.plume_reset_pending = true;
        }
        self.tool = tool;
        self.draw_start_world = None;
        self.paused = true;
        self.screen = GuiScreen::ActiveWorkbench;
        self.command_palette_open = false;
    }

    fn enter_workbench_running(
        &mut self,
        tab: DockTab,
        sub_mode: Option<StressSubMode>,
        layer: HeatmapLayer,
        tool: ViewportTool,
    ) {
        self.enter_workbench(tab, sub_mode, layer, tool);
        self.paused = false;
    }

    fn active_scenario_mode_label(&self) -> &'static str {
        if self.tab == DockTab::ResilienceStress {
            match self.stress_sub_mode {
                StressSubMode::DeterministicCutSets => "N-1 / N-2 Bridge & Corridor Cut-Sets",
                StressSubMode::SeismicMonteCarlo => "Seismic Bridge & Buried Pipe Monte Carlo",
                StressSubMode::FireFollowingQuake => "Fire-Following-Earthquake Conflagration",
                StressSubMode::HospitalTraumaSurge => "Hospital Trauma & Regional Burn Surge",
                StressSubMode::AeromedicalMedevac => "Rotary-Wing Medevac & Smoke Envelopes",
                StressSubMode::EvacuationContraflow => "Staged Evacuation & Arterial Contraflow",
                StressSubMode::TransitBusEvacuation => "GTFS Bus+EMS Contraflow Evacuation",
                StressSubMode::EmergencyShelters => "Clean-Air Shelters & HEPA Filtration",
                StressSubMode::CommodityResupply => "Multi-Day Water/Diesel/O2 Resupply VRP",
                StressSubMode::WarningMobilization => "Cell Blackout, WEA/Siren & Mobilization",
                StressSubMode::FemaCommsArchitecture => "FEMA IPAWS & MERS ESF-2 Comms",
                StressSubMode::FemaIsbPods => "FEMA ISB & C-POD Supply Queues",
                StressSubMode::FemaUsarDebris => "FEMA Type-1 US&R Grids & Debris Push",
                StressSubMode::FemaGsbDrcTdms => "USACE 249th GSB, TDMS & DRC Queues",
                StressSubMode::WildfireAlertWestResponse => "ALERTWest AI PTZ & WUI Suppression",
                StressSubMode::BurnScarDebrisFlow => "Post-Fire Debris Flow & Culverts",
                StressSubMode::WtpTurbidityStorage => "WTP Ash/Turbidity & Storage Drawdown",
            }
        } else {
            match self.layer {
                HeatmapLayer::Zoning => "Urban Land Use & Microsimulation",
                HeatmapLayer::Isochrone15Min => "15-Minute Multimodal Access Isochrone",
                HeatmapLayer::EmergencyResponse4Min => "4-Minute EMS Coverage (NFPA 1710)",
                HeatmapLayer::ResponseDistricts => "Carter-Larson Response Districts",
                HeatmapLayer::UtilityLifelines => "Coupled Electric / Water / Sewer Grids",
                HeatmapLayer::VspEmissions => "VSP Vehicle Emissions Advection Plume",
                HeatmapLayer::AgentDensity => "50k+ Agent Multimodal Density",
                HeatmapLayer::Off => "Clean Network Asphalt & Water Channels",
                _ => "Resilience & Hazard Analysis",
            }
        }
    }

    fn render_command_palette(&mut self, ctx: &egui::Context) {
        if !self.command_palette_open {
            return;
        }
        let mut keep_open = true;
        egui::Window::new("Quick-Jump Module & Scenario Palette (Ctrl+K)")
            .open(&mut keep_open)
            .collapsible(false)
            .resizable(false)
            .default_width(540.0)
            .anchor(egui::Align2::CENTER_TOP, EguiVec2::new(0.0, 56.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Search:").strong());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.command_palette_query)
                            .hint_text("Type module, FEMA system, lifeline, CAD tool, or scenario...")
                            .desired_width(420.0),
                    );
                    if self.command_palette_query.is_empty() {
                        resp.request_focus();
                    }
                    if ui.small_button("Clear").clicked() {
                        self.command_palette_query.clear();
                    }
                });
                ui.separator();

                let q = self.command_palette_query.trim().to_ascii_lowercase();
                let entries: [(
                    &str,
                    &str,
                    DockTab,
                    Option<StressSubMode>,
                    HeatmapLayer,
                    ViewportTool,
                ); 26] = [
                    (
                        "Traffic & Signals",
                        "Signal Control, Max-Pressure & EVP Green-Wave Preemption",
                        DockTab::SignalsAndRoads,
                        None,
                        HeatmapLayer::EmergencyResponse4Min,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Street CAD Studio",
                        "Draw Roads, Edit Lanes (+/-) & Convert Curved Roundabouts",
                        DockTab::SignalsAndRoads,
                        None,
                        HeatmapLayer::Zoning,
                        ViewportTool::DrawRoad,
                    ),
                    (
                        "Zoning & GTFS",
                        "Land-Use Parcel Zoning, Gravity OD & GTFS Scheduled Bus Fleet",
                        DockTab::ZoningAndTransit,
                        None,
                        HeatmapLayer::Zoning,
                        ViewportTool::PaintZone,
                    ),
                    (
                        "50k+ Agent Physics",
                        "IDM/MOBIL Microscopic Car-Following, Plumes & 50k+ GPU Crowd",
                        DockTab::IdmMobilAndCrowd,
                        None,
                        HeatmapLayer::AgentDensity,
                        ViewportTool::Inspect,
                    ),
                    (
                        "City Ledger & x-t",
                        "15-Minute Isochrones, Municipal Ledger & Space-Time Shockwaves",
                        DockTab::CityAndShockwave,
                        None,
                        HeatmapLayer::Isochrone15Min,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Rayon Batch Sweeps",
                        "Parallel Multi-Thread A/B Control Regime & Demand Sweeps",
                        DockTab::BatchSweep,
                        None,
                        HeatmapLayer::VspEmissions,
                        ViewportTool::Inspect,
                    ),
                    (
                        "MEXCLP & Hypercube",
                        "Daskin MEXCLP Station Siting & Larson 2^N Queueing",
                        DockTab::SignalsAndRoads,
                        None,
                        HeatmapLayer::EmergencyResponse4Min,
                        ViewportTool::PlaceFacility,
                    ),
                    (
                        "Diurnal Beats",
                        "Carter-Chaiken Response Districts & 24-Hr Peak/Off-Peak Dispatch",
                        DockTab::SignalsAndRoads,
                        None,
                        HeatmapLayer::ResponseDistricts,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Utility Lifelines",
                        "12.47 kV Electric DistFlow, 20-psi Water Hydrants & EPA RDII Sewers",
                        DockTab::UtilityGrids,
                        None,
                        HeatmapLayer::UtilityLifelines,
                        ViewportTool::Inspect,
                    ),
                    (
                        "N-1 / N-2 Cut-Sets",
                        "Deterministic Arterial & Bridge Closure Vulnerability Interdiction",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::DeterministicCutSets),
                        HeatmapLayer::EmergencyResponse4Min,
                        ViewportTool::ToggleClosure,
                    ),
                    (
                        "Seismic Monte Carlo",
                        "Hazus Bridge Fragility, ALA PGV Buried Pipes & 90-Day Restoration",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::SeismicMonteCarlo),
                        HeatmapLayer::EmergencyResponse4Min,
                        ViewportTool::Inspect,
                    ),
                    (
                        "FFE Conflagration",
                        "Scawthorn Post-Quake Ignitions, Hamada Wind Spread & Low-psi Hydrants",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::FireFollowingQuake),
                        HeatmapLayer::FireFollowingQuake,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Hospital ED Surge",
                        "FEMA P-58 Trauma/Burn Surge Queueing & MOCC Ambulance Diversion",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::HospitalTraumaSurge),
                        HeatmapLayer::HospitalTraumaSurge,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Rotary Medevac",
                        "FAA Helipads, Koschmieder Smoke Flight Envelopes & Burn Transfers",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::AeromedicalMedevac),
                        HeatmapLayer::AeromedicalMedevac,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Contraflow Evac",
                        "FFE Perimeter Road Closures, Smoke Speed & Arterial Lane Reversal",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::EvacuationContraflow),
                        HeatmapLayer::EvacuationContraflow,
                        ViewportTool::Inspect,
                    ),
                    (
                        "GTFS Bus+EMS Evac",
                        "GTFS Pickup Depots, 60-ft Coaches, ADA Lift Vans & Bus+EMS Contraflow",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::TransitBusEvacuation),
                        HeatmapLayer::TransitBusEvacuation,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Clean-Air Shelters",
                        "ATC-20 Seismic Habitability, MERV-13 + HEPA Filtration & Routing",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::EmergencyShelters),
                        HeatmapLayer::EmergencyShelters,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Commodity VRP",
                        "Multi-Day Water, Diesel, MRE & Oxygen Clarke-Wright Deprivation IRP",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::CommodityResupply),
                        HeatmapLayer::CommodityResupply,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Cell & Siren PADM",
                        "Cellular BTS Blackout Fragility, 124-dBC Sirens & PADM Mobilization",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::WarningMobilization),
                        HeatmapLayer::WarningMobilization,
                        ViewportTool::Inspect,
                    ),
                    (
                        "FEMA IPAWS & MERS",
                        "IPAWS-OPEN CAP v1.2, NPWS/PEP EAS, NOAA SAME, FNARS & Region 10 MERS",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::FemaCommsArchitecture),
                        HeatmapLayer::FemaCommsArchitecture,
                        ViewportTool::Inspect,
                    ),
                    (
                        "FEMA ISB & C-PODs",
                        "FEMA Incident Staging Bases, USACE IS-26 POD Queues & IPAWS Routing",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::FemaIsbPods),
                        HeatmapLayer::FemaIsbPods,
                        ViewportTool::Inspect,
                    ),
                    (
                        "FEMA Type-1 US&R",
                        "80-Pax 24h USNG Collapse Grids, 72h Survival & ESF-3 Debris Push",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::FemaUsarDebris),
                        HeatmapLayer::FemaUsarDebris,
                        ViewportTool::Inspect,
                    ),
                    (
                        "FEMA GSB/DRC/TDMS",
                        "USACE 249th Prime Power GSB, 75%-Red. TDMS Debris & DRC/MRIC IA Queues",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::FemaGsbDrcTdms),
                        HeatmapLayer::FemaGsbDrcTdms,
                        ViewportTool::Inspect,
                    ),
                    (
                        "ALERTWest Wildfire WUI",
                        "ALERTWest AI PTZ Cameras, Starlink MDUs & NWCG Wildfire Suppression",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::WildfireAlertWestResponse),
                        HeatmapLayer::WildfireAlertWestResponse,
                        ViewportTool::Inspect,
                    ),
                    (
                        "Burn-Scar & Culverts",
                        "Hydrophobic CN Runoff, USGS M1/Gartner Debris Basins & HDS-5 Culvert Washouts",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::BurnScarDebrisFlow),
                        HeatmapLayer::BurnScarDebrisFlow,
                        ViewportTool::Inspect,
                    ),
                    (
                        "WTP Ash & Turbidity",
                        "Burn-Scar Ash/NTU Spikes, Intake Shutoffs, FeCl3/PAC Coag & Tank Drawdown",
                        DockTab::ResilienceStress,
                        Some(StressSubMode::WtpTurbidityStorage),
                        HeatmapLayer::WtpTurbidityStorage,
                        ViewportTool::Inspect,
                    ),
                ];

                egui::ScrollArea::vertical()
                    .max_height(340.0)
                    .show(ui, |ui| {
                        for (badge, desc, tab, sm, lyr, tl) in entries {
                            if !q.is_empty()
                                && !badge.to_ascii_lowercase().contains(&q)
                                && !desc.to_ascii_lowercase().contains(&q)
                            {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new(format!("[{badge}]"))
                                            .monospace()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    self.enter_workbench(tab, sm, lyr, tl);
                                }
                                ui.label(desc);
                            });
                        }
                    });
            });
        if !keep_open {
            self.command_palette_open = false;
        }
    }

    fn render_main_entrance_hub(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::group(&ctx.style())
                    .fill(Color32::from_rgb(9, 13, 24))
                    .stroke(Stroke::NONE)
                    .inner_margin(18.0),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Hero Entrance Header Card
                        egui::Frame::group(ui.style())
                            .fill(Color32::from_rgb(15, 23, 42))
                            .stroke(Stroke::new(1.2, Color32::from_rgb(51, 65, 85)))
                            .rounding(8.0)
                            .inner_margin(16.0)
                            .show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label(
                                            egui::RichText::new(
                                                "URBAN-GRID v5.2  —  MULTI-HAZARD LIFELINE, FEMA, WILDFIRE & WTP TURBIDITY WORKBENCH",
                                            )
                                            .size(20.0)
                                            .strong()
                                            .color(Color32::from_rgb(226, 232, 240)),
                                        );
                                        ui.label(
                                            egui::RichText::new(
                                                "50k+ wgpu Traffic  |  Complete FEMA Suite (IPAWS/MERS, ISB/PODs, US&R, 249th GSB, TDMS, DRC/MRIC)  |  ALERTWest WUI Wildfire",
                                            )
                                            .color(Color32::from_rgb(148, 163, 184)),
                                        );
                                        ui.add_space(4.0);
                                        ui.monospace(format!(
                                            "Staged City: {} | {} road segs | {} junctions | {} parcels | {} staged agents | {}",
                                            self.active_city_name,
                                            self.world.network.segments.len(),
                                            self.world.network.junctions.len(),
                                            self.world.parcels.len(),
                                            self.world.total_active_agents(),
                                            if self.paused { "PAUSED — Ready (Zero Background CPU)" } else { "RUNNING" },
                                        ));
                                    });

                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui
                                                .button(
                                                    egui::RichText::new("  ▶ Launch & Run  ")
                                                        .size(14.5)
                                                        .strong()
                                                        .color(Color32::from_rgb(52, 211, 153)),
                                                )
                                                .on_hover_text(
                                                    "Open the wgpu map viewport and start simulation immediately",
                                                )
                                                .clicked()
                                            {
                                                self.paused = false;
                                                self.screen = GuiScreen::ActiveWorkbench;
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new(
                                                        "  Enter Workbench (Paused)  ",
                                                    )
                                                    .size(14.5)
                                                    .strong()
                                                    .color(Color32::from_rgb(250, 204, 21)),
                                                )
                                                .on_hover_text(
                                                    "Open the wgpu map viewport paused at t=0.0s (press Space or ▶ Run to start)",
                                                )
                                                .clicked()
                                            {
                                                self.paused = true;
                                                self.screen = GuiScreen::ActiveWorkbench;
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new("Command Palette (Ctrl+K)")
                                                        .color(Color32::from_rgb(125, 211, 252)),
                                                )
                                                .clicked()
                                            {
                                                self.command_palette_open = true;
                                            }
                                        },
                                    );
                                });
                            });

                        ui.add_space(10.0);

                        // Row 1: Built-in Urban Scenarios & Snapshot Loader
                        egui::Frame::group(ui.style())
                            .fill(Color32::from_rgb(13, 20, 36))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(38, 52, 78)))
                            .rounding(7.0)
                            .inner_margin(12.0)
                            .show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(
                                        egui::RichText::new("1. SELECT OR LOAD SCENARIO NETWORK")
                                            .strong()
                                            .color(Color32::from_rgb(186, 230, 253)),
                                    );
                                    ui.separator();
                                    ui.label("Snapshot .json:");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.snapshot_path)
                                            .desired_width(210.0),
                                    );
                                    if ui.button("Load Snapshot & Open Map").clicked() {
                                        if let Ok((loaded_world, loaded_plume)) =
                                            SimulationWorld::load_snapshot_json(Path::new(
                                                &self.snapshot_path,
                                            ))
                                        {
                                            let b = loaded_world.network.bounds;
                                            let span =
                                                (b.max_x - b.min_x).max(b.max_y - b.min_y).max(200.0);
                                            self.cam_x = 0.5 * (b.min_x + b.max_x);
                                            self.cam_y = 0.5 * (b.min_y + b.max_y);
                                            self.cam_zoom = (760.0 / span).clamp(0.28, 2.2);
                                            self.selected_junction = 0;
                                            self.selected_segment = loaded_world.watched_segment_id;
                                            self.plume_params = loaded_plume;
                                            self.active_city_name = loaded_world.network.city_label.clone();
                                            self.world = loaded_world;
                                            self.heatmap_dirty = true;
                                            self.plume_reset_pending = true;
                                            self.paused = true;
                                            self.screen = GuiScreen::ActiveWorkbench;
                                        }
                                    }
                                });
                                ui.add_space(6.0);
                                ui.columns(5, |cols| {
                                    cols[0].vertical(|ui| {
                                        ui.strong("1. Riverfront Bridges Metro");
                                        ui.small(
                                            "S-curve river channel, 4 bridges, 1-way couplets, West/East districts & riverside parks.",
                                        );
                                        ui.horizontal_wrapped(|ui| {
                                            if ui.button("Open (Paused)").clicked() {
                                                self.load_builtin(&builtin_riverfront_bridges_osm(), None);
                                                self.enter_workbench(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::Zoning,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new("▶ Launch & Run")
                                                        .color(Color32::from_rgb(16, 185, 129)),
                                                )
                                                .clicked()
                                            {
                                                self.load_builtin(&builtin_riverfront_bridges_osm(), None);
                                                self.enter_workbench_running(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::Zoning,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                        });
                                    });
                                    cols[1].vertical(|ui| {
                                        ui.strong("2. Harbor Bay & Peninsula");
                                        ui.small(
                                            "Curved bay shoreline, 35-deg coastal boulevard, port terminal & 2 bay bridges.",
                                        );
                                        ui.horizontal_wrapped(|ui| {
                                            if ui.button("Open (Paused)").clicked() {
                                                self.load_builtin(&builtin_harbor_peninsula_osm(), None);
                                                self.enter_workbench(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::Zoning,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new("▶ Launch & Run")
                                                        .color(Color32::from_rgb(16, 185, 129)),
                                                )
                                                .clicked()
                                            {
                                                self.load_builtin(&builtin_harbor_peninsula_osm(), None);
                                                self.enter_workbench_running(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::Zoning,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                        });
                                    });
                                    cols[2].vertical(|ui| {
                                        ui.strong("3. Radial Ring & Star Plazas");
                                        ui.small(
                                            "3 concentric beltways, 8 radial avenues, civic canal, central park & roundabout plazas.",
                                        );
                                        ui.horizontal_wrapped(|ui| {
                                            if ui.button("Open (Paused)").clicked() {
                                                self.load_builtin(&builtin_radial_ring_plazas_osm(), None);
                                                self.enter_workbench(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::Zoning,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new("▶ Launch & Run")
                                                        .color(Color32::from_rgb(16, 185, 129)),
                                                )
                                                .clicked()
                                            {
                                                self.load_builtin(&builtin_radial_ring_plazas_osm(), None);
                                                self.enter_workbench_running(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::Zoning,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                        });
                                    });
                                    cols[3].vertical(|ui| {
                                        ui.strong("4. Foothill Canyon WUI");
                                        ui.small(
                                            "Curved canyon roads, mountain creek bridges, WUI ridge loops & 2 evac highways.",
                                        );
                                        ui.horizontal_wrapped(|ui| {
                                            if ui.button("Open (Paused)").clicked() {
                                                self.load_builtin(&builtin_foothill_canyon_wui_osm(), None);
                                                self.enter_workbench(
                                                    DockTab::ResilienceStress,
                                                    Some(StressSubMode::WildfireAlertWestResponse),
                                                    HeatmapLayer::WildfireAlertWestResponse,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new("▶ Launch & Run")
                                                        .color(Color32::from_rgb(16, 185, 129)),
                                                )
                                                .clicked()
                                            {
                                                self.load_builtin(&builtin_foothill_canyon_wui_osm(), None);
                                                self.enter_workbench_running(
                                                    DockTab::ResilienceStress,
                                                    Some(StressSubMode::WildfireAlertWestResponse),
                                                    HeatmapLayer::WildfireAlertWestResponse,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                        });
                                    });
                                    cols[4].vertical(|ui| {
                                        ui.strong("5. Metropolis 50k+ Multimodal");
                                        ui.small(
                                            "Multi-district arterial hierarchy, diagonal BRT spine, river crossings & 52k+ agents.",
                                        );
                                        ui.horizontal_wrapped(|ui| {
                                            if ui.button("Open (Paused)").clicked() {
                                                self.load_builtin(
                                                    &builtin_metropolis_50k_osm(),
                                                    Some(self.target_crowd_slider.max(52_000)),
                                                );
                                                self.enter_workbench(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::AgentDensity,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                            if ui
                                                .button(
                                                    egui::RichText::new("▶ Launch & Run")
                                                        .color(Color32::from_rgb(56, 189, 248)),
                                                )
                                                .clicked()
                                            {
                                                self.load_builtin(
                                                    &builtin_metropolis_50k_osm(),
                                                    Some(self.target_crowd_slider.max(52_000)),
                                                );
                                                self.enter_workbench_running(
                                                    DockTab::SignalsAndRoads,
                                                    None,
                                                    HeatmapLayer::AgentDensity,
                                                    ViewportTool::Inspect,
                                                );
                                            }
                                        });
                                    });
                                });
                                ui.add_space(4.0);
                                ui.horizontal_wrapped(|ui| {
                                    ui.small("Compact Network Micro-Labs:");
                                    if ui.small_button("2-Lane Roundabout").clicked() {
                                        self.load_builtin(&builtin_roundabout_osm(), None);
                                        self.enter_workbench(
                                            DockTab::SignalsAndRoads,
                                            None,
                                            HeatmapLayer::Zoning,
                                            ViewportTool::Inspect,
                                        );
                                    }
                                    if ui.small_button("Braess Diamond").clicked() {
                                        self.load_builtin(&builtin_braess_diamond_osm(), None);
                                        self.enter_workbench(
                                            DockTab::SignalsAndRoads,
                                            None,
                                            HeatmapLayer::Zoning,
                                            ViewportTool::Inspect,
                                        );
                                    }
                                });
                            });

                        ui.add_space(10.0);

                        // Search Filter Bar for the 5-Domain Module Directory
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("2. LAUNCH SIMULATION OR RESILIENCE WORKBENCH MODULE")
                                    .strong()
                                    .color(Color32::from_rgb(167, 243, 208)),
                            );
                            ui.separator();
                            ui.add(
                                egui::TextEdit::singleline(&mut self.hub_search_query)
                                    .hint_text("Filter 24 modules (e.g. ALERTWest, Wildfire, GSB, DRC, TDMS, IPAWS, US&R, POD, GTFS)...")
                                    .desired_width(430.0),
                            );
                            if !self.hub_search_query.is_empty() && ui.small_button("Clear").clicked() {
                                self.hub_search_query.clear();
                            }
                        });
                        ui.add_space(6.0);

                        let fq = self.hub_search_query.trim().to_ascii_lowercase();
                        let domains: [(
                            &str,
                            Color32,
                            [(
                                &str,
                                &str,
                                DockTab,
                                Option<StressSubMode>,
                                HeatmapLayer,
                                ViewportTool,
                            ); 4],
                        ); 6] = [
                            (
                                "DOMAIN I — MICROSCOPIC TRAFFIC, STREET CAD & MULTIMODAL TRANSIT",
                                Color32::from_rgb(56, 189, 248),
                                [
                                    (
                                        "Signals, Max-Pressure & EVP Green-Wave",
                                        "2/4-phase signals, Varaiya Max-Pressure control & multi-hop emergency green waves.",
                                        DockTab::SignalsAndRoads,
                                        None,
                                        HeatmapLayer::EmergencyResponse4Min,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Interactive Street CAD & Roundabouts",
                                        "Draw snapped 2-way/BRT/bike streets, +/- lane road diets & curved roundabouts (Ctrl+Z).",
                                        DockTab::SignalsAndRoads,
                                        None,
                                        HeatmapLayer::Zoning,
                                        ViewportTool::DrawRoad,
                                    ),
                                    (
                                        "Parcel Zoning, Gravity OD & GTFS Buses",
                                        "Paint R1/R3/MU4/CBD/IND/PARK parcels, MNL mode split & scheduled GTFS bus routes.",
                                        DockTab::ZoningAndTransit,
                                        None,
                                        HeatmapLayer::Zoning,
                                        ViewportTool::PaintZone,
                                    ),
                                    (
                                        "50k+ IDM/MOBIL Physics & Rayon Sweeps",
                                        "Microscopic car-following, @compute wind plumes, x-t diagrams & parallel A/B sweeps.",
                                        DockTab::IdmMobilAndCrowd,
                                        None,
                                        HeatmapLayer::AgentDensity,
                                        ViewportTool::Inspect,
                                    ),
                                ],
                            ),
                            (
                                "DOMAIN II — EMERGENCY DISTRICTING & MUNICIPAL UTILITY GRIDS",
                                Color32::from_rgb(52, 211, 153),
                                [
                                    (
                                        "Daskin MEXCLP & Larson 2^N Hypercube",
                                        "Optimal fire/EMS station siting with busy-apparatus queueing & second-due backup.",
                                        DockTab::SignalsAndRoads,
                                        None,
                                        HeatmapLayer::EmergencyResponse4Min,
                                        ViewportTool::PlaceFacility,
                                    ),
                                    (
                                        "Carter-Chaiken Beats & 24-Hr Diurnal",
                                        "Workload-equalized response district boundaries & peak vs. off-peak dispatch tables.",
                                        DockTab::SignalsAndRoads,
                                        None,
                                        HeatmapLayer::ResponseDistricts,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Electric, 20-psi Water & Sewer Grids",
                                        "Coupled 12.47 kV DistFlow feeders, Hazen-Williams hydrants & Manning/RDII sewers.",
                                        DockTab::UtilityGrids,
                                        None,
                                        HeatmapLayer::UtilityLifelines,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "N-1 / N-2 Bridge & Arterial Cut-Sets",
                                        "Exhaustive single & compound bridge/corridor interdiction vulnerability ranking.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::DeterministicCutSets),
                                        HeatmapLayer::EmergencyResponse4Min,
                                        ViewportTool::ToggleClosure,
                                    ),
                                ],
                            ),
                            (
                                "DOMAIN III — SEISMIC FRAGILITY, CONFLAGRATION & MEDICAL SURGE",
                                Color32::from_rgb(251, 146, 60),
                                [
                                    (
                                        "Seismic Bridge & ALA PGV Pipe MC",
                                        "Spatially correlated Sa(1.0s) bridge fragility, buried pipe breaks & 90-day recovery.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::SeismicMonteCarlo),
                                        HeatmapLayer::EmergencyResponse4Min,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Fire-Following-Earthquake (FFE)",
                                        "Scawthorn post-quake ignitions, Hamada elliptical wind spread & <20 psi hydrants.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::FireFollowingQuake),
                                        HeatmapLayer::FireFollowingQuake,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Hospital ED Surge & MOCC Diversion",
                                        "FEMA P-58 acute bed fragility, ABA burn/trauma priority triage & ambulance diversion.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::HospitalTraumaSurge),
                                        HeatmapLayer::HospitalTraumaSurge,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Rotary-Wing Medevac & Smoke Envelopes",
                                        "Hospital/field helipads, FAA 14 CFR 135.609 smoke/wind ceilings & burn-center airlifts.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::AeromedicalMedevac),
                                        HeatmapLayer::AeromedicalMedevac,
                                        ViewportTool::Inspect,
                                    ),
                                ],
                            ),
                            (
                                "DOMAIN IV — EVACUATION, CLEAN-AIR SHELTERS & COMMODITY VRP",
                                Color32::from_rgb(45, 212, 191),
                                [
                                    (
                                        "Staged Evacuation & Arterial Contraflow",
                                        "Active FFE thermal closures, smoke visibility speeds & EMS-protected contraflow.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::EvacuationContraflow),
                                        HeatmapLayer::EvacuationContraflow,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "GTFS Bus + EMS Contraflow & ADA Vans",
                                        "400m GTFS pickup depots, fare-free 60-ft BRT coaches & ADA wheelchair-lift shuttles.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::TransitBusEvacuation),
                                        HeatmapLayer::TransitBusEvacuation,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "ATC-20 Clean-Air (MERV-13 + HEPA) Shelters",
                                        "Congregate shelter habitability fault trees, indoor PM2.5 mass-balance & routing.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::EmergencyShelters),
                                        HeatmapLayer::EmergencyShelters,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Multi-Day Water/Diesel/MRE/O2 VRP",
                                        "14-day Clarke-Wright 2-Opt convoy tours & Holguin-Veras deprivation-cost IRP.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::CommodityResupply),
                                        HeatmapLayer::CommodityResupply,
                                        ViewportTool::Inspect,
                                    ),
                                ],
                            ),
                            (
                                "DOMAIN V — FEMA COMMUNICATIONS (ESF-2), ISB/PODS & TYPE-1 US&R (ESF-9)",
                                Color32::from_rgb(250, 204, 21),
                                [
                                    (
                                        "Cell BTS Blackout, Sirens & PADM Curves",
                                        "FCC 8h vs. CPUC 72h BTS backup, 124-dBC sirens, Hi-Lo/D2D & delayed staging.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::WarningMobilization),
                                        HeatmapLayer::WarningMobilization,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "FEMA IPAWS-OPEN, PEP EAS, SAME & MERS",
                                        "COG EOC CAP v1.2, 50kW NPWS/PEP AM/FM, NOAA 162MHz SAME, FNARS HF & Region 10 MERS.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::FemaCommsArchitecture),
                                        HeatmapLayer::FemaCommsArchitecture,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "FEMA ISB & USACE IS-26 POD Queueing",
                                        "53-ft ISB trailer push, Type I/II/III + Walk-Up M/G/c spillback & IPAWS POD routing.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::FemaIsbPods),
                                        HeatmapLayer::FemaIsbPods,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "FEMA Type-1 US&R Grids & Debris Push",
                                        "80-pax 24h USNG collapse grids, 72h survival decay & ESF-3 'Teeth-of-Comb' push.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::FemaUsarDebris),
                                        HeatmapLayer::FemaUsarDebris,
                                        ViewportTool::Inspect,
                                    ),
                                ],
                            ),
                            (
                                "DOMAIN VI — FEMA ESF-3/IA RECOVERY & ALERTWEST WUI WILDFIRE RESPONSE",
                                Color32::from_rgb(248, 113, 113),
                                [
                                    (
                                        "FEMA GSB (249th), TDMS & DRC/MRIC",
                                        "USACE 249th Prime Power packs, 75%-red. TDMS debris yards & FEMA/SBA DRC + MRIC queues.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::FemaGsbDrcTdms),
                                        HeatmapLayer::FemaGsbDrcTdms,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "ALERTWest Wildfire AI Cameras",
                                        "Link JSON/GeoJSON/CSV feeds: 360° 40x NIR PTZ, 2-min AI scan, seismic telemetry & Starlink MDUs.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::WildfireAlertWestResponse),
                                        HeatmapLayer::WildfireAlertWestResponse,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "WTP Ash/Turbidity, Coagulant & Storage",
                                        "River ash/NTU shutoffs, FeCl3+PAC dosing, IESWTR 0.3 NTU & clearwell drawdown.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::WtpTurbidityStorage),
                                        HeatmapLayer::WtpTurbidityStorage,
                                        ViewportTool::Inspect,
                                    ),
                                    (
                                        "Burn-Scar Debris Flows & Culvert Washout",
                                        "Hydrophobic CN runoff, USGS M1/Gartner debris basins & HDS-5 culvert/RDII washouts.",
                                        DockTab::ResilienceStress,
                                        Some(StressSubMode::BurnScarDebrisFlow),
                                        HeatmapLayer::BurnScarDebrisFlow,
                                        ViewportTool::Inspect,
                                    ),
                                ],
                            ),
                        ];

                        for (dom_title, dom_col, cards) in domains {
                            let visible_cards: Vec<_> = cards
                                .into_iter()
                                .filter(|(t, d, _, _, _, _)| {
                                    fq.is_empty()
                                        || dom_title.to_ascii_lowercase().contains(&fq)
                                        || t.to_ascii_lowercase().contains(&fq)
                                        || d.to_ascii_lowercase().contains(&fq)
                                })
                                .collect();
                            if visible_cards.is_empty() {
                                continue;
                            }

                            egui::Frame::group(ui.style())
                                .fill(Color32::from_rgb(13, 20, 36))
                                .stroke(Stroke::new(1.0, Color32::from_rgb(38, 52, 78)))
                                .rounding(7.0)
                                .inner_margin(11.0)
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(dom_title).strong().color(dom_col),
                                    );
                                    ui.add_space(5.0);
                                    ui.columns(2, |cols| {
                                        for (idx, (title, desc, tab, sm, lyr, tl)) in
                                            visible_cards.into_iter().enumerate()
                                        {
                                            cols[idx % 2].vertical(|ui| {
                                                egui::Frame::group(ui.style())
                                                    .fill(Color32::from_rgb(19, 28, 48))
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        Color32::from_rgb(48, 64, 92),
                                                    ))
                                                    .rounding(5.0)
                                                    .inner_margin(8.0)
                                                    .show(ui, |ui| {
                                                        ui.horizontal(|ui| {
                                                            ui.label(
                                                                egui::RichText::new(title)
                                                                    .strong()
                                                                    .color(Color32::from_rgb(
                                                                        241, 245, 249,
                                                                    )),
                                                            );
                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    if ui
                                                                        .small_button(
                                                                            egui::RichText::new(
                                                                                "Open ->",
                                                                            )
                                                                            .strong()
                                                                            .color(dom_col),
                                                                        )
                                                                        .clicked()
                                                                    {
                                                                        self.enter_workbench(
                                                                            tab, sm, lyr, tl,
                                                                        );
                                                                    }
                                                                },
                                                            );
                                                        });
                                                        ui.small(desc);
                                                    });
                                                ui.add_space(4.0);
                                            });
                                        }
                                    });
                                });
                            ui.add_space(8.0);
                        }
                    });
            });
    }
}

impl UrbanGridGuiApp {
    pub fn ui_step(&mut self, ctx: &egui::Context) {
        self.poll_background_sweep();
        let dt_real = ctx.input(|i| i.stable_dt as f64).clamp(0.0005, 0.05);
        self.ema_frame_ms = self.ema_frame_ms * 0.92 + (dt_real * 1000.0) * 0.08;

        // Global CAD Undo / Redo keyboard shortcuts (`Ctrl+Z`, `Ctrl+Y`, and `Ctrl+Shift+Z`).
        if !ctx.wants_keyboard_input() {
            let (want_redo, want_undo) = ctx.input_mut(|i| {
                let redo_shift_z = egui::KeyboardShortcut::new(
                    egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                    egui::Key::Z,
                );
                let redo_y = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
                let undo_z = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
                let r = i.consume_shortcut(&redo_shift_z) || i.consume_shortcut(&redo_y);
                let u = if r { false } else { i.consume_shortcut(&undo_z) };
                (r, u)
            });
            if want_redo {
                self.trigger_redo();
            } else if want_undo {
                self.trigger_undo();
            }
        }

        // Global Navigation Shortcuts: `Ctrl+K` Command Palette & `F1` / `Home` Main Entrance Hub toggle
        let (toggle_palette, toggle_home, toggle_pause) = ctx.input_mut(|i| {
            let cmd_k = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);
            let tk = i.consume_shortcut(&cmd_k);
            let th = !tk && (i.key_pressed(egui::Key::F1) || i.key_pressed(egui::Key::Home));
            let tp = i.key_pressed(egui::Key::Space);
            (tk, th, tp)
        });
        if toggle_palette {
            self.command_palette_open = !self.command_palette_open;
        }
        if toggle_home && !ctx.wants_keyboard_input() {
            self.screen = match self.screen {
                GuiScreen::MainEntranceHub => GuiScreen::ActiveWorkbench,
                GuiScreen::ActiveWorkbench => GuiScreen::MainEntranceHub,
            };
        }

        self.render_command_palette(ctx);
        // Zero background simulation or continuous repaint while on MainEntranceHub!
        if self.screen == GuiScreen::MainEntranceHub {
            self.render_main_entrance_hub(ctx);
            return;
        }

        if toggle_pause && !ctx.wants_keyboard_input() {
            self.paused = !self.paused;
        }

        if !self.paused {
            if self.layer == HeatmapLayer::EmergencyResponse4Min
                || (self.layer == HeatmapLayer::ResponseDistricts
                    && self.world.diurnal_live_clock_enabled)
            {
                self.coverage_refresh_accum_s += dt_real;
                if self.coverage_refresh_accum_s >= 1.2 {
                    self.coverage_refresh_accum_s = 0.0;
                    self.heatmap_dirty = true;
                }
            }
            self.physics_accum_s += dt_real * self.speed_scale;
            let mut substeps = 0usize;
            while self.physics_accum_s >= FIXED_PHYSICS_DT && substeps < 2 {
                self.world.step(FIXED_PHYSICS_DT);
                self.physics_accum_s -= FIXED_PHYSICS_DT;
                substeps += 1;
            }
            if self.physics_accum_s > FIXED_PHYSICS_DT * 2.0 {
                self.physics_accum_s = 0.0;
            }
            // Request continuous repaint only while simulation is actively running
            ctx.request_repaint();
        }
        let interp_dt = if self.paused {
            0.0
        } else {
            self.physics_accum_s.clamp(0.0, FIXED_PHYSICS_DT)
        };

        egui::TopBottomPanel::top("top_toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button(
                        egui::RichText::new("Home Menu (F1)")
                            .strong()
                            .color(Color32::from_rgb(250, 204, 21)),
                    )
                    .on_hover_text("Return to the clean 5-Domain Main Entrance Menu Hub (F1 / Home)")
                    .clicked()
                {
                    self.screen = GuiScreen::MainEntranceHub;
                }
                ui.heading("urban-grid v5.2");
                if ui
                    .small_button(
                        egui::RichText::new("Ctrl+K")
                            .monospace()
                            .color(Color32::from_rgb(125, 211, 252)),
                    )
                    .on_hover_text("Open Quick-Jump Module & Scenario Command Palette (Ctrl+K)")
                    .clicked()
                {
                    self.command_palette_open = !self.command_palette_open;
                }
                ui.checkbox(&mut self.show_cad_toolbar, "CAD Bar");
                ui.separator();
                if ui.button("Riverfront Bridges").clicked() {
                    self.load_builtin(&builtin_riverfront_bridges_osm(), None);
                }
                if ui.button("Harbor Peninsula").clicked() {
                    self.load_builtin(&builtin_harbor_peninsula_osm(), None);
                }
                if ui.button("Radial Ring Plazas").clicked() {
                    self.load_builtin(&builtin_radial_ring_plazas_osm(), None);
                }
                if ui.button("Foothill Canyon").clicked() {
                    self.load_builtin(&builtin_foothill_canyon_wui_osm(), None);
                }
                if ui
                    .button(
                        egui::RichText::new("Metropolis 50k+")
                            .strong()
                            .color(Color32::from_rgb(56, 189, 248)),
                    )
                    .clicked()
                {
                    self.load_builtin(
                        &builtin_metropolis_50k_osm(),
                        Some(self.target_crowd_slider.max(52_000)),
                    );
                }
                if ui.small_button("Roundabout").clicked() {
                    self.load_builtin(&builtin_roundabout_osm(), None);
                }
                if ui.small_button("Braess").clicked() {
                    self.load_builtin(&builtin_braess_diamond_osm(), None);
                }
                ui.separator();
                if ui
                    .button(if self.paused {
                        egui::RichText::new("▶ Run (Space)")
                            .strong()
                            .color(Color32::from_rgb(16, 185, 129))
                    } else {
                        egui::RichText::new("⏸ Pause (Space)")
                            .strong()
                            .color(Color32::from_rgb(250, 204, 21))
                    })
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                if ui
                    .button(
                        egui::RichText::new("Dispatch Emergency (E)")
                            .strong()
                            .color(Color32::from_rgb(248, 113, 113)),
                    )
                    .on_hover_text(
                        "Dispatch a siren EmergencyResponder with multi-hop corridor green-wave preemption & MOBIL pull-over yielding",
                    )
                    .clicked()
                {
                    if let Some(ev_id) = self
                        .world
                        .dispatch_emergency_on_segment(Some(self.selected_segment))
                    {
                        self.io_status_msg = format!(
                            "Dispatched Emergency Unit #{ev_id} on corridor #{} (EVP Green-Wave active)",
                            self.selected_segment
                        );
                    }
                }

                let prev_layer = self.layer;
                egui::ComboBox::from_id_salt("layer_cb")
                    .selected_text(match self.layer {
                        HeatmapLayer::Zoning => "GPU Heatmap: Parcel Zoning",
                        HeatmapLayer::Isochrone15Min => "GPU Heatmap: 15-Min Isochrone",
                        HeatmapLayer::EmergencyResponse4Min => "GPU Heatmap: 4-Min Emergency Coverage",
                        HeatmapLayer::ResponseDistricts => "GPU Heatmap: Response Districts (Beats)",
                        HeatmapLayer::UtilityLifelines => "GPU Heatmap: Electric / Water / Sewer Grids",
                        HeatmapLayer::FireFollowingQuake => "GPU Heatmap: Fire-Following-Earthquake (FFE)",
                        HeatmapLayer::HospitalTraumaSurge => "GPU Heatmap: Hospital ED Surge & Diversion",
                        HeatmapLayer::AeromedicalMedevac => "GPU Heatmap: Medevac Helipads & Smoke Envelope",
                        HeatmapLayer::EvacuationContraflow => "GPU Heatmap: Staged Evacuation & Contraflow",
                        HeatmapLayer::EmergencyShelters => "GPU Heatmap: Clean-Air Shelters & HEPA Routing",
                        HeatmapLayer::CommodityResupply => "GPU Heatmap: Multi-Day Commodity Resupply VRP",
                        HeatmapLayer::TransitBusEvacuation => "GPU Heatmap: GTFS Bus+EMS Contraflow & ADA Evac",
                        HeatmapLayer::WarningMobilization => "GPU Heatmap: Cell Blackout, WEA/Siren & Mobilization",
                        HeatmapLayer::FemaCommsArchitecture => "GPU Heatmap: FEMA IPAWS (WEA/PEP/NWR) & MERS ESF-2",
                        HeatmapLayer::FemaIsbPods => "GPU Heatmap: FEMA ISB, USACE POD Queueing & IPAWS Routing",
                        HeatmapLayer::FemaUsarDebris => "GPU Heatmap: FEMA US&R Type-1 Grids, Debris Push & POD Reopen",
                        HeatmapLayer::FemaGsbDrcTdms => "GPU Heatmap: FEMA 249th GSB Power, TDMS Debris & DRC/MRIC",
                        HeatmapLayer::WildfireAlertWestResponse => "GPU Heatmap: ALERTWest Cameras & NWCG WUI Wildfire",
                        HeatmapLayer::BurnScarDebrisFlow => "GPU Heatmap: Burn-Scar Runoff, USGS Debris Basins & Culverts",
                        HeatmapLayer::WtpTurbidityStorage => "GPU Heatmap: WTP Ash/Turbidity Shutoff, Coagulant & Storage",
                        HeatmapLayer::VspEmissions => "GPU Heatmap: VSP Emissions Plume",
                        HeatmapLayer::AgentDensity => "GPU Heatmap: 50k Agent Density",
                        HeatmapLayer::Off => "GPU Heatmap: Off (Clean Asphalt)",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::Zoning,
                            "GPU Heatmap: Parcel Zoning",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::Isochrone15Min,
                            "GPU Heatmap: 15-Min Isochrone",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::EmergencyResponse4Min,
                            "GPU Heatmap: 4-Min Emergency Coverage (NFPA 1710)",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::ResponseDistricts,
                            "GPU Heatmap: Response Districts / Beats (Carter-Larson)",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::UtilityLifelines,
                            "GPU Heatmap: Coupled Electric / Water / Wastewater Grids",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::FireFollowingQuake,
                            "GPU Heatmap: Fire-Following-Earthquake (Scawthorn/Hamada)",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::HospitalTraumaSurge,
                            "GPU Heatmap: Hospital ED Surge & Ambulance Diversion",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::AeromedicalMedevac,
                            "GPU Heatmap: Rotary-Wing Medevac & Smoke Envelope",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::EvacuationContraflow,
                            "GPU Heatmap: Staged Evacuation & Arterial Contraflow",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::EmergencyShelters,
                            "GPU Heatmap: Emergency Shelters, Lifelines & HEPA Air",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::CommodityResupply,
                            "GPU Heatmap: Multi-Day Water/Diesel/MRE/O2 Resupply VRP",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::TransitBusEvacuation,
                            "GPU Heatmap: GTFS Bus+EMS Contraflow & ADA Clean-Air Evac",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::WarningMobilization,
                            "GPU Heatmap: Cell Tower Blackout, WEA/Siren & Delayed Staging",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::FemaCommsArchitecture,
                            "GPU Heatmap: FEMA IPAWS (CAP/PEP/NWR) & Region 10 MERS ESF-2",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::FemaIsbPods,
                            "GPU Heatmap: FEMA ISB, Type I-III POD Queues & IPAWS Routing",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::FemaUsarDebris,
                            "GPU Heatmap: FEMA US&R Type-1 USNG Grids & Road Debris Push",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::FemaGsbDrcTdms,
                            "GPU Heatmap: FEMA 249th GSB Prime Power, TDMS & DRC/MRIC",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::WildfireAlertWestResponse,
                            "GPU Heatmap: ALERTWest PTZ Cameras & NWCG WUI Wildfire",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::BurnScarDebrisFlow,
                            "GPU Heatmap: Burn-Scar Runoff, USGS M1 Basins & Culvert Washout",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::WtpTurbidityStorage,
                            "GPU Heatmap: WTP Ash/Turbidity Shutoff, FeCl3/PAC & Tank Drawdown",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::VspEmissions,
                            "GPU Heatmap: VSP Emissions Plume",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::AgentDensity,
                            "GPU Heatmap: 50k Agent Density",
                        );
                        ui.selectable_value(
                            &mut self.layer,
                            HeatmapLayer::Off,
                            "GPU Heatmap: Off (Clean Asphalt)",
                        );
                    });
                if self.layer != prev_layer {
                    self.heatmap_dirty = true;
                    self.plume_reset_pending = true;
                }

                ui.separator();
                if ui.button("Save .json").on_hover_text("Save scenario snapshot to JSON").clicked() {
                    let include_ag = self.save_include_agents && self.world.total_active_agents() <= 15_000;
                    match self.world.save_snapshot_json(
                        Path::new(&self.snapshot_path),
                        self.plume_params,
                        "Workbench Scenario Snapshot",
                        include_ag,
                    ) {
                        Ok(bytes) => {
                            self.io_status_msg = format!(
                                "Saved {} ({:.1} KB)",
                                self.snapshot_path,
                                bytes as f64 / 1024.0
                            );
                        }
                        Err(e) => {
                            self.io_status_msg = format!("Save failed: {e}");
                        }
                    }
                }
                if ui.button("Load .json").on_hover_text("Load scenario snapshot from JSON").clicked() {
                    match SimulationWorld::load_snapshot_json(Path::new(&self.snapshot_path)) {
                        Ok((loaded_world, loaded_plume)) => {
                            let b = loaded_world.network.bounds;
                            let span = (b.max_x - b.min_x).max(b.max_y - b.min_y).max(200.0);
                            self.cam_x = 0.5 * (b.min_x + b.max_x);
                            self.cam_y = 0.5 * (b.min_y + b.max_y);
                            self.cam_zoom = (760.0 / span).clamp(0.28, 2.2);
                            self.selected_junction = 0;
                            self.selected_segment = loaded_world.watched_segment_id;
                            self.plume_params = loaded_plume;
                            self.io_status_msg = format!(
                                "Loaded {} (t={:.1}s, {} agents)",
                                self.snapshot_path,
                                loaded_world.sim_time_s,
                                loaded_world.total_active_agents()
                            );
                            self.active_city_name = loaded_world.network.city_label.clone();
                            self.world = loaded_world;
                            self.physics_accum_s = 0.0;
                            self.heatmap_dirty = true;
                            self.plume_reset_pending = true;
                            self.paused = true;
                        }
                        Err(e) => {
                            self.io_status_msg = format!("Load failed: {e}");
                        }
                    }
                }
                if ui.button("Export .csv").on_hover_text("Export 1 Hz time-series telemetry to CSV").clicked() {
                    match self.world.export_telemetry_csv(Path::new(&self.csv_export_path)) {
                        Ok(rows) => {
                            self.io_status_msg =
                                format!("Exported {rows} rows -> {}", self.csv_export_path);
                        }
                        Err(e) => {
                            self.io_status_msg = format!("CSV export failed: {e}");
                        }
                    }
                }

                ui.separator();
                let fps = (1000.0 / self.ema_frame_ms.max(0.5)).min(999.0);
                ui.label(format!(
                    "Agents: {} ({} veh, {} bike, {} ped) | {:.0} FPS ({:.2} ms) | {:.1} mph | HCM LOS {}",
                    self.world.total_active_agents(),
                    self.world.vehicles.len(),
                    self.world.cyclists.len(),
                    self.world.pedestrians.len(),
                    fps,
                    self.ema_frame_ms,
                    self.world.mean_speed_mph(),
                    self.world.ledger.hcm_los
                ));
                if !self.io_status_msg.is_empty() {
                    ui.separator();
                    ui.colored_label(Color32::from_rgb(52, 211, 153), &self.io_status_msg);
                }
            });
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                let status_badge = if self.paused {
                    egui::RichText::new("⏸ PAUSED — Ready")
                        .strong()
                        .color(Color32::from_rgb(250, 204, 21))
                } else {
                    egui::RichText::new("▶ RUNNING")
                        .strong()
                        .color(Color32::from_rgb(52, 211, 153))
                };
                let br_cnt = self.world.network.segments.iter().filter(|s| s.is_bridge).count();
                let wt_cnt = self.world.network.water_features.len();
                ui.label(
                    egui::RichText::new(format!("[Active City Map: {}]", self.active_city_name))
                        .strong()
                        .color(Color32::from_rgb(125, 211, 252)),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new(format!(
                        "[Active Scenario / Hazard Mode: {}]",
                        self.active_scenario_mode_label()
                    ))
                    .color(Color32::from_rgb(167, 243, 208)),
                );
                ui.separator();
                ui.label(status_badge);
                ui.monospace(format!(
                    "| Sim Clock {:.1}s | Agents {} | Bridges {} | Water {}",
                    self.world.sim_time_s,
                    self.world.total_active_agents(),
                    br_cnt,
                    wt_cnt
                ));
                if ui
                    .small_button(if self.paused { "▶ Run" } else { "⏸ Pause" })
                    .clicked()
                {
                    self.paused = !self.paused;
                }
            });
            if self.show_cad_toolbar {
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label("Viewport Tool:");
                if ui
                    .selectable_label(self.tool == ViewportTool::Inspect, "1: Pan / Inspect")
                    .clicked()
                {
                    self.tool = ViewportTool::Inspect;
                    self.draw_start_world = None;
                }
                if ui
                    .selectable_label(self.tool == ViewportTool::DrawRoad, "2: Draw Road")
                    .clicked()
                {
                    self.tool = ViewportTool::DrawRoad;
                }
                if ui
                    .selectable_label(self.tool == ViewportTool::EditLanes, "3: Edit Lanes (+/-)")
                    .clicked()
                {
                    self.tool = ViewportTool::EditLanes;
                    self.draw_start_world = None;
                }
                if ui
                    .selectable_label(
                        self.tool == ViewportTool::ConvertRoundabout,
                        "4: Convert Roundabout",
                    )
                    .clicked()
                {
                    self.tool = ViewportTool::ConvertRoundabout;
                    self.draw_start_world = None;
                }
                if ui
                    .selectable_label(self.tool == ViewportTool::PaintZone, "5: Paint Zoning")
                    .clicked()
                {
                    self.tool = ViewportTool::PaintZone;
                    self.draw_start_world = None;
                }
                if ui
                    .selectable_label(
                        self.tool == ViewportTool::PlaceFacility,
                        "6: Fire / Hospital",
                    )
                    .clicked()
                {
                    self.tool = ViewportTool::PlaceFacility;
                    self.draw_start_world = None;
                    self.layer = HeatmapLayer::EmergencyResponse4Min;
                    self.heatmap_dirty = true;
                }
                if ui
                    .selectable_label(
                        self.tool == ViewportTool::ToggleClosure,
                        format!("7: Bridge/Road Closure ({})", self.world.closed_segment_count()),
                    )
                    .clicked()
                {
                    self.tool = ViewportTool::ToggleClosure;
                    self.draw_start_world = None;
                    self.layer = HeatmapLayer::EmergencyResponse4Min;
                    self.heatmap_dirty = true;
                }
                ui.separator();
                let undo_tip = self
                    .world
                    .edit_history
                    .peek_undo()
                    .map(|r| format!("Ctrl+Z — Undo: {}", r.label))
                    .unwrap_or_else(|| "Ctrl+Z — Nothing to undo".to_string());
                if ui
                    .add_enabled(
                        self.world.edit_history.can_undo(),
                        egui::Button::new(format!("Undo ({})", self.world.edit_history.undo_len())),
                    )
                    .on_hover_text(undo_tip)
                    .clicked()
                {
                    self.trigger_undo();
                }
                let redo_tip = self
                    .world
                    .edit_history
                    .peek_redo()
                    .map(|r| format!("Ctrl+Y / Ctrl+Shift+Z — Redo: {}", r.label))
                    .unwrap_or_else(|| "Ctrl+Y — Nothing to redo".to_string());
                if ui
                    .add_enabled(
                        self.world.edit_history.can_redo(),
                        egui::Button::new(format!("Redo ({})", self.world.edit_history.redo_len())),
                    )
                    .on_hover_text(redo_tip)
                    .clicked()
                {
                    self.trigger_redo();
                }

                match self.tool {
                    ViewportTool::DrawRoad => {
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.draw_spec.lanes_per_dir, 1..=4)
                                .text("Lanes/dir"),
                        );
                        ui.checkbox(&mut self.draw_spec.bidirectional, "Two-Way");
                        ui.checkbox(&mut self.draw_spec.has_bus_lane, "BRT Lane");
                        ui.checkbox(&mut self.draw_spec.has_bike_lane, "Bike Lane");
                        if self.draw_start_world.is_some() && ui.button("Cancel Origin (Esc)").clicked()
                        {
                            self.draw_start_world = None;
                        }
                    }
                    ViewportTool::EditLanes => {
                        ui.separator();
                        ui.checkbox(&mut self.symmetric_lane_edit, "Symmetric Two-Way Edit");
                        ui.label("Left-click road: +1 lane | Right-click road: -1 lane (road diet)");
                    }
                    ViewportTool::ConvertRoundabout => {
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.roundabout_radius_m, 14.0..=30.0)
                                .text("Radius (m)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.roundabout_lanes, 1..=3)
                                .text("Ring Lanes"),
                        );
                        ui.label("Click any 3+ arm intersection to build curved roundabout");
                    }
                    ViewportTool::PlaceFacility => {
                        ui.separator();
                        ui.selectable_value(
                            &mut self.active_facility_kind,
                            EmergencyFacilityKind::FireStation,
                            "Fire Station (NFPA 4-Min)",
                        );
                        ui.selectable_value(
                            &mut self.active_facility_kind,
                            EmergencyFacilityKind::Hospital,
                            "Trauma Hospital (+)",
                        );
                        if ui
                            .button(
                                egui::RichText::new("MEXCLP Auto-Site >=90%")
                                    .strong()
                                    .color(Color32::from_rgb(52, 211, 153)),
                            )
                            .on_hover_text(
                                "Solve Daskin (1983) MEXCLP + 1-Swap GAS accounting for busy apparatus & second-due backup coverage",
                            )
                            .clicked()
                        {
                            let rep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                            let n_applied = self.world.apply_mclp_siting_solution(&rep);
                            self.io_status_msg = format!(
                                "MEXCLP sited {n_applied} stations ({} units, q={:.2}) -> E[Cov]={:.1}%, 2nd-Due={:.1}% ({:.1} ms)",
                                rep.optimal_total_apparatus,
                                rep.effective_busy_prob_q,
                                rep.achieved_expected_cov_pct,
                                rep.achieved_second_due_cov_pct,
                                rep.solve_time_ms
                            );
                            self.latest_mclp_report = Some(rep);
                            self.layer = HeatmapLayer::EmergencyResponse4Min;
                            self.heatmap_dirty = true;
                        }
                        ui.label("L-click: Place | R-click: Remove");
                    }
                    ViewportTool::ToggleClosure => {
                        ui.separator();
                        ui.checkbox(&mut self.stress_cfg.bidirectional_closure, "2-Way Closure");
                        if ui
                            .button(
                                egui::RichText::new("Run Stress-Test")
                                    .strong()
                                    .color(Color32::from_rgb(251, 146, 60)),
                            )
                            .clicked()
                        {
                            let srep = self.world.run_resilience_stress_test(&self.stress_cfg);
                            self.io_status_msg = format!(
                                "Stress-tested {} disruptions ({:.1} ms): worst cov {:.1}% (-{:.1} pts), pass rate {:.0}%",
                                srep.evaluated_scenarios,
                                srep.solve_time_ms,
                                srep.worst_case_parcel_cov_pct,
                                srep.max_coverage_drop_pct_pts,
                                srep.resilience_pass_rate_pct
                            );
                            self.latest_stress_report = Some(srep);
                            self.tab = DockTab::ResilienceStress;
                            self.layer = HeatmapLayer::EmergencyResponse4Min;
                            self.heatmap_dirty = true;
                        }
                        if self.world.closed_segment_count() > 0
                            && ui.button("Reopen All Roads").clicked()
                        {
                            let n_open = self.world.clear_all_road_closures_tracked();
                            self.io_status_msg =
                                format!("Reopened {n_open} closed segments (Ctrl+Z to undo)");
                            self.heatmap_dirty = true;
                        }
                        ui.label("L-click road: Toggle Closure | R-click: Reopen");
                    }
                    ViewportTool::Inspect | ViewportTool::PaintZone => {}
                }
            });
            }
        });

        egui::SidePanel::right("inspector_dock")
            .default_width(405.0)
            .show(ctx, |ui| {
                let sweep_lbl = if self.active_sweep.is_some() {
                    "A/B Sweep *"
                } else {
                    "A/B Sweep"
                };
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.tab, DockTab::SignalsAndRoads, "Signals/St");
                    ui.selectable_value(&mut self.tab, DockTab::ZoningAndTransit, "Zone/GTFS");
                    ui.selectable_value(&mut self.tab, DockTab::IdmMobilAndCrowd, "50k+/Phys");
                    ui.selectable_value(&mut self.tab, DockTab::CityAndShockwave, "City/x-t");
                    ui.selectable_value(&mut self.tab, DockTab::BatchSweep, sweep_lbl);
                    ui.selectable_value(&mut self.tab, DockTab::ResilienceStress, "Stress-Test");
                    ui.selectable_value(&mut self.tab, DockTab::UtilityGrids, "Utilities");
                });
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                    DockTab::SignalsAndRoads => {
                        if let Some(junc) =
                            self.world.network.junctions.get_mut(self.selected_junction)
                        {
                            ui.label(format!(
                                "Selected Junction #{} (OSM {})",
                                junc.id, junc.osm_node_id
                            ));
                            egui::ComboBox::from_label("Control")
                                .selected_text(format!("{:?}", junc.control_mode))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut junc.control_mode,
                                        ControlMode::Signal2Phase,
                                        "2-Phase Signal",
                                    );
                                    ui.selectable_value(
                                        &mut junc.control_mode,
                                        ControlMode::Signal4PhaseProtectedLeft,
                                        "4-Phase Prot. Left",
                                    );
                                    ui.selectable_value(
                                        &mut junc.control_mode,
                                        ControlMode::MaxPressureAdaptive,
                                        "Max-Pressure Adaptive",
                                    );
                                    ui.selectable_value(
                                        &mut junc.control_mode,
                                        ControlMode::Roundabout,
                                        "Roundabout Yield",
                                    );
                                    ui.selectable_value(
                                        &mut junc.control_mode,
                                        ControlMode::PermissiveYield,
                                        "Permissive Yield",
                                    );
                                });
                            ui.add(
                                egui::Slider::new(&mut junc.ns_green_s, 6.0..=60.0)
                                    .text("Axis A Green (s)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut junc.ew_green_s, 6.0..=60.0)
                                    .text("Axis B Green (s)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut junc.offset_s, 0.0..=60.0)
                                    .text("Wave Offset (s)"),
                            );
                            if ui
                                .button(if junc.hold_phase {
                                    "Release Held Phase"
                                } else {
                                    "Hold Current Phase"
                                })
                                .clicked()
                            {
                                junc.hold_phase = !junc.hold_phase;
                                junc.held_at_s = self.world.sim_time_s;
                            }
                            if junc.control_mode == ControlMode::MaxPressureAdaptive {
                                ui.separator();
                                ui.label(format!(
                                    "Active MP Stage: {:?} ({}) — {:.1}s | Switches: {}",
                                    junc.mp_state.stage,
                                    junc.mp_state.active_phase.short_label(),
                                    junc.mp_state.stage_elapsed_s,
                                    junc.mp_state.phase_switches
                                ));
                                for ph in MpPhaseId::ALL {
                                    let pval = junc.mp_state.phase_pressures[ph.index()];
                                    let wait = junc.mp_state.phase_wait_s[ph.index()];
                                    ui.label(format!(
                                        "  {} -> P = {:.2} (wait {:.1}s)",
                                        ph.short_label(),
                                        pval,
                                        wait
                                    ));
                                }
                            }
                        }
                        ui.separator();
                        ui.label(
                            egui::RichText::new(
                                "Emergency Vehicle Preemption (EVP) & Siren Yielding:",
                            )
                            .strong(),
                        );
                        if ui
                            .checkbox(
                                &mut self.world.evp_config.enabled,
                                "Enable Corridor Green-Wave EVP + MOBIL Pull-Over",
                            )
                            .changed()
                        {
                            self.world.refresh_urban_metrics();
                            self.heatmap_dirty = true;
                        }
                        if ui
                            .checkbox(
                                &mut self.world.evp_config.include_nfpa_turnout_time,
                                "Include NFPA 1710 Turnout (80s Fire / 60s EMS)",
                            )
                            .changed()
                        {
                            self.world.refresh_urban_metrics();
                            self.heatmap_dirty = true;
                        }
                        ui.add(
                            egui::Slider::new(
                                &mut self.world.evp_config.lookahead_distance_m,
                                80.0..=500.0,
                            )
                            .text("EVP Lookahead (m)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.world.evp_config.max_corridor_hops, 1..=6)
                                .text("Green-Wave Hops"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.world.evp_config.siren_yield_radius_m,
                                25.0..=140.0,
                            )
                            .text("Siren Yield Radius (m)"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.world.evp_config.auto_dispatch_per_min,
                                0.0..=3.0,
                            )
                            .text("Auto EV / min"),
                        );
                        let evpt = self.world.evp_telemetry;
                        ui.colored_label(
                            Color32::from_rgb(103, 232, 249),
                            format!(
                                "EVP Active: {} units ({:.1} mph) | {} juncs green-waved | {} cars yielding",
                                evpt.active_responders,
                                evpt.mean_ev_speed_mph,
                                evpt.preempted_junctions,
                                evpt.yielding_vehicles
                            ),
                        );
                        let ecov = self.world.emergency_coverage;
                        let cov_col = if ecov.parcels_within_4min_pct >= 90.0 {
                            Color32::from_rgb(52, 211, 153)
                        } else if ecov.parcels_within_4min_pct >= 65.0 {
                            Color32::from_rgb(250, 204, 21)
                        } else {
                            Color32::from_rgb(248, 113, 113)
                        };
                        ui.colored_label(
                            cov_col,
                            format!(
                                "NFPA 1710 <=4m: {:.0}% parcels ({:.0}% pop) | P90 {:.0}s | <=8m {:.0}%",
                                ecov.parcels_within_4min_pct,
                                ecov.residents_within_4min_pct,
                                ecov.p90_fire_travel_s,
                                ecov.parcels_within_8min_als_pct
                            ),
                        );
                        ui.small(format!(
                            "Facilities: {} Fire Stations, {} Hospitals | Mean Hosp Transport: {:.0}s",
                            ecov.fire_station_count,
                            ecov.hospital_count,
                            ecov.mean_hospital_transport_s
                        ));
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(
                                "MEXCLP & Second-Due Backup Siting (Daskin 1983 / Hogan-ReVelle 1986):",
                            )
                            .strong(),
                        );
                        egui::ComboBox::from_id_salt("mexclp_obj_cb")
                            .selected_text(self.mclp_cfg.objective_model.label())
                            .show_ui(ui, |ui| {
                                for om in [
                                    SitingObjectiveModel::DaskinMexclp1983,
                                    SitingObjectiveModel::MexclpPlusHoganBackup1986,
                                    SitingObjectiveModel::DeterministicMclp1974,
                                ] {
                                    ui.selectable_value(
                                        &mut self.mclp_cfg.objective_model,
                                        om,
                                        om.label(),
                                    );
                                }
                            });
                        egui::ComboBox::from_id_salt("mexclp_busy_mode_cb")
                            .selected_text(self.mclp_cfg.busyness_model.label())
                            .show_ui(ui, |ui| {
                                for bm in [
                                    BusynessModel::LarsonHypercube1974,
                                    BusynessModel::FixedBusyFraction,
                                    BusynessModel::EndogenousWorkloadQueue,
                                ] {
                                    ui.selectable_value(
                                        &mut self.mclp_cfg.busyness_model,
                                        bm,
                                        bm.label(),
                                    );
                                }
                            });
                        match self.mclp_cfg.busyness_model {
                            BusynessModel::FixedBusyFraction => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.mclp_cfg.apparatus_busy_prob,
                                        0.00..=0.75,
                                    )
                                    .step_by(0.01)
                                    .text("Busy Prob q"),
                                );
                            }
                            BusynessModel::EndogenousWorkloadQueue
                            | BusynessModel::LarsonHypercube1974 => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.mclp_cfg.system_calls_per_hr,
                                        0.5..=14.0,
                                    )
                                    .text("Calls / hr"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.mclp_cfg.mean_service_duration_mins,
                                        15.0..=75.0,
                                    )
                                    .text("Service (min)"),
                                );
                            }
                        }
                        egui::ComboBox::from_id_salt("mclp_weight_cb")
                            .selected_text(self.mclp_cfg.weighting_mode.label())
                            .show_ui(ui, |ui| {
                                for m in [
                                    MclpWeightingMode::BothResidentAndParcel,
                                    MclpWeightingMode::PopulationWeighted,
                                    MclpWeightingMode::EqualParcel,
                                ] {
                                    ui.selectable_value(
                                        &mut self.mclp_cfg.weighting_mode,
                                        m,
                                        m.label(),
                                    );
                                }
                            });
                        ui.add(
                            egui::Slider::new(&mut self.mclp_cfg.target_coverage_pct, 50.0..=100.0)
                                .text("Target E[Cov] (%)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mclp_cfg.threshold_travel_s, 90.0..=360.0)
                                .step_by(15.0)
                                .text("1st-Due Max (s)"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.mclp_cfg.second_due_threshold_s,
                                120.0..=480.0,
                            )
                            .step_by(15.0)
                            .text("2nd-Due Backup (s)"),
                        );
                        if self.mclp_cfg.objective_model
                            == SitingObjectiveModel::MexclpPlusHoganBackup1986
                        {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.mclp_cfg.target_second_due_cov_pct,
                                    25.0..=95.0,
                                )
                                .text("2nd-Due Target (%)"),
                            );
                        }
                        ui.add(
                            egui::Slider::new(
                                &mut self.mclp_cfg.max_engines_per_station,
                                1..=3,
                            )
                            .text("Max Engines / Stn"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mclp_cfg.max_stations_budget, 2..=20)
                                .text("Max Fleet Budget"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.mclp_cfg.min_station_spacing_m,
                                60.0..=400.0,
                            )
                            .text("Min Spacing (m)"),
                        );
                        ui.checkbox(
                            &mut self.mclp_cfg.keep_existing_fire_stations,
                            "Keep Existing Fire Stations (Incremental Siting)",
                        );
                        ui.horizontal(|ui| {
                            if ui.button("Preview MEXCLP").clicked() {
                                let rep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                self.io_status_msg = format!(
                                    "MEXCLP ({:.1} ms): {} units across {} stations -> E[Cov]={:.1}%, 2nd-Due={:.1}% (q={:.2})",
                                    rep.solve_time_ms,
                                    rep.optimal_total_apparatus,
                                    rep.optimal_total_stations,
                                    rep.achieved_expected_cov_pct,
                                    rep.achieved_second_due_cov_pct,
                                    rep.effective_busy_prob_q
                                );
                                self.latest_mclp_report = Some(rep);
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                            if ui
                                .button(
                                    egui::RichText::new("Solve & Apply (Ctrl+Z)")
                                        .strong()
                                        .color(Color32::from_rgb(52, 211, 153)),
                                )
                                .clicked()
                            {
                                let rep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                let total_fs = self.world.apply_mclp_siting_solution(&rep);
                                self.io_status_msg = format!(
                                    "Applied MEXCLP: {total_fs} stations ({} units, q={:.2}) -> E[Cov]={:.1}%, 2nd-Due={:.1}%",
                                    rep.optimal_total_apparatus,
                                    rep.effective_busy_prob_q,
                                    rep.achieved_expected_cov_pct,
                                    rep.achieved_second_due_cov_pct
                                );
                                self.latest_mclp_report = Some(rep);
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                        });

                        if let Some(mrep) = self.latest_mclp_report.clone() {
                            let status_col = if mrep.target_met {
                                Color32::from_rgb(52, 211, 153)
                            } else {
                                Color32::from_rgb(250, 204, 21)
                            };
                            ui.colored_label(
                                status_col,
                                format!(
                                    "Fleet: {} units in {} stations (q={:.2}) | E[Cov]={:.1}% | 2nd-Due={:.1}% ({:.1} ms)",
                                    mrep.optimal_total_apparatus,
                                    mrep.optimal_total_stations,
                                    mrep.effective_busy_prob_q,
                                    mrep.achieved_expected_cov_pct,
                                    mrep.achieved_second_due_cov_pct,
                                    mrep.solve_time_ms
                                ),
                            );
                            ui.monospace(format!(
                                "Binary k>=1: {:.1}% pcl / {:.1}% pop | 1st P90: {:.0}s
E[Travel]  : {:.1}s | 2nd-Due Backup P90: {:.0}s",
                                mrep.achieved_parcel_cov_pct,
                                mrep.achieved_resident_cov_pct,
                                mrep.achieved_p90_travel_s,
                                mrep.achieved_expected_travel_s,
                                mrep.achieved_second_due_p90_s
                            ));

                            // 3-Series MEXCLP Pareto Frontier: Binary k>=1 (Sky), Daskin E[Cov] 1-q^k (Emerald), 2nd-Due k>=2 (Amber)
                            if mrep.pareto_curve.len() >= 2 {
                                let (p_rect, _) = ui.allocate_exact_size(
                                    EguiVec2::new(360.0, 98.0),
                                    Sense::hover(),
                                );
                                let pp = ui.painter_at(p_rect);
                                pp.rect_filled(p_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                let tgt_y = p_rect.bottom()
                                    - 13.0
                                    - ((mrep.config.target_coverage_pct as f32 / 100.0)
                                        .clamp(0.0, 1.0))
                                        * (p_rect.height() - 22.0);
                                pp.line_segment(
                                    [
                                        Pos2::new(p_rect.left() + 6.0, tgt_y),
                                        Pos2::new(p_rect.right() - 6.0, tgt_y),
                                    ],
                                    Stroke::new(
                                        1.0,
                                        Color32::from_rgba_unmultiplied(250, 204, 21, 130),
                                    ),
                                );
                                let n_pts = mrep.pareto_curve.len();
                                let mut poly_bin = Vec::with_capacity(n_pts);
                                let mut poly_exp = Vec::with_capacity(n_pts);
                                let mut poly_bkp = Vec::with_capacity(n_pts);

                                for (i, pt) in mrep.pareto_curve.iter().enumerate() {
                                    let x = p_rect.left()
                                        + 14.0
                                        + (i as f32 / (n_pts - 1).max(1) as f32)
                                            * (p_rect.width() - 28.0);
                                    let map_y = |pct: f64| -> f32 {
                                        p_rect.bottom()
                                            - 13.0
                                            - ((pct as f32 / 100.0).clamp(0.0, 1.0))
                                                * (p_rect.height() - 22.0)
                                    };
                                    poly_bin.push(Pos2::new(x, map_y(pt.residents_within_threshold_pct)));
                                    poly_exp.push(Pos2::new(x, map_y(pt.expected_cov_pct)));
                                    poly_bkp.push(Pos2::new(x, map_y(pt.second_due_backup_cov_pct)));

                                    pp.circle_filled(
                                        Pos2::new(x, map_y(pt.expected_cov_pct)),
                                        2.8,
                                        Color32::from_rgb(52, 211, 153),
                                    );
                                    if n_pts <= 10 || i % 2 == 0 || i == n_pts - 1 {
                                        pp.text(
                                            Pos2::new(x, p_rect.bottom() - 1.0),
                                            egui::Align2::CENTER_BOTTOM,
                                            format!("M={}", pt.total_apparatus),
                                            egui::FontId::monospace(8.8),
                                            Color32::GRAY,
                                        );
                                    }
                                }
                                pp.add(egui::Shape::line(
                                    poly_bin,
                                    Stroke::new(1.3, Color32::from_rgb(56, 189, 248)),
                                ));
                                pp.add(egui::Shape::line(
                                    poly_bkp,
                                    Stroke::new(1.6, Color32::from_rgb(251, 146, 60)),
                                ));
                                pp.add(egui::Shape::line(
                                    poly_exp,
                                    Stroke::new(2.3, Color32::from_rgb(52, 211, 153)),
                                ));
                                pp.text(
                                    Pos2::new(p_rect.left() + 8.0, p_rect.top() + 2.0),
                                    egui::Align2::LEFT_TOP,
                                    "k>=1(Sky)  E[Cov]=1-q^k(Grn)  2nd-Due k>=2(Org)",
                                    egui::FontId::monospace(8.4),
                                    Color32::GRAY,
                                );
                            }
                            ui.horizontal(|ui| {
                                ui.label("CSV:");
                                ui.text_edit_singleline(&mut self.mclp_csv_path);
                                if ui.small_button("Export .csv").clicked() {
                                    match mrep.export_csv_path(Path::new(&self.mclp_csv_path)) {
                                        Ok(n_r) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_r} MEXCLP frontier rows -> {}",
                                                self.mclp_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV failed: {e}");
                                        }
                                    }
                                }
                            });
                        }

                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(
                                "Larson (1974) Exact 2^N M/M/N Spatial Hypercube Queue:",
                            )
                            .strong(),
                        );
                        egui::ComboBox::from_id_salt("hc_disc_cb")
                            .selected_text(self.hypercube_cfg.queue_discipline.label())
                            .show_ui(ui, |ui| {
                                for qd in [
                                    HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue,
                                    HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss,
                                ] {
                                    ui.selectable_value(
                                        &mut self.hypercube_cfg.queue_discipline,
                                        qd,
                                        qd.label(),
                                    );
                                }
                            });
                        ui.checkbox(
                            &mut self.hypercube_cfg.calibrate_travel_service_rate,
                            "Travel-Time Calibrated Heterogeneous mu_n",
                        );
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.hypercube_cfg.system_calls_per_hr,
                                    0.5..=14.0,
                                )
                                .text("Lambda/hr"),
                            );
                        });
                        ui.horizontal(|ui| {
                            if ui
                                .button(
                                    egui::RichText::new("Solve Exact Hypercube P(B)")
                                        .strong()
                                        .color(Color32::from_rgb(56, 189, 248)),
                                )
                                .on_hover_text(
                                    "Solves Larson (1974) 2^N CTMC state-transition balance equations on active Fire Stations",
                                )
                                .clicked()
                            {
                                if self.world.emergency_coverage.fire_station_count == 0 {
                                    let sited = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                    self.world.apply_mclp_siting_solution(&sited);
                                    self.latest_mclp_report = Some(sited);
                                }
                                self.hypercube_cfg.threshold_travel_s =
                                    self.mclp_cfg.threshold_travel_s;
                                self.hypercube_cfg.second_due_threshold_s =
                                    self.mclp_cfg.second_due_threshold_s;
                                let hrep =
                                    self.world.solve_active_fleet_hypercube(&self.hypercube_cfg);
                                self.io_status_msg = format!(
                                    "Hypercube ({} states, {} iters, {:.2} ms): rho {:.1}%..{:.1}%, E[Cov]={:.1}%, x-dist={:.1}%",
                                    hrep.state_space_size,
                                    hrep.gauss_seidel_iterations,
                                    hrep.solve_time_ms,
                                    hrep.min_busy_fraction_rho * 100.0,
                                    hrep.max_busy_fraction_rho * 100.0,
                                    hrep.expected_primary_cov_pct,
                                    hrep.cross_district_dispatch_pct
                                );
                                self.latest_hypercube_report = Some(hrep);
                                self.heatmap_dirty = true;
                            }
                        });

                        if let Some(hrep) = self.latest_hypercube_report.clone() {
                            let mode_tag = if hrep.used_exact_2n_ctmc {
                                format!("Exact 2^{}={} CTMC", hrep.num_units, hrep.state_space_size)
                            } else {
                                format!("Larson-1975 Q ({} units)", hrep.num_units)
                            };
                            ui.monospace(format!(
                                "{} | {} GS sweeps | res={:.1e} | err={:.1e}
Busy rho: Mean {:.1}% (Min {:.1}% .. Max {:.1}% | Spread {:.1}pt)
P0 Idle : {:.1}% | P(Sat/Q)={:.2}% | W_q={:.1}s | E[Resp]={:.1}s
E[Cov]  : {:.1}% | 2nd-Avail={:.1}% | Cross-Dist={:.1}%",
                                mode_tag,
                                hrep.gauss_seidel_iterations,
                                hrep.final_max_residual,
                                hrep.rate_conservation_error,
                                hrep.mean_busy_fraction_rho * 100.0,
                                hrep.min_busy_fraction_rho * 100.0,
                                hrep.max_busy_fraction_rho * 100.0,
                                hrep.workload_imbalance_spread * 100.0,
                                hrep.prob_all_idle_p0 * 100.0,
                                hrep.prob_all_busy_or_queued * 100.0,
                                hrep.mean_queue_wait_s,
                                hrep.mean_total_response_s,
                                hrep.expected_primary_cov_pct,
                                hrep.expected_second_due_avail_pct,
                                hrep.cross_district_dispatch_pct
                            ));

                            // Per-Unit Heterogeneous Busy-Fraction (`rho_n`) Bar Chart
                            let n_bars = hrep.unit_metrics.len().min(8);
                            if n_bars > 0 {
                                let bh_tot = (n_bars as f32) * 17.0 + 8.0;
                                let (u_rect, _) = ui.allocate_exact_size(
                                    EguiVec2::new(360.0, bh_tot),
                                    Sense::hover(),
                                );
                                let up = ui.painter_at(u_rect);
                                up.rect_filled(u_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                for (i, um) in hrep.unit_metrics.iter().take(n_bars).enumerate() {
                                    let y_top = u_rect.top() + 4.0 + (i as f32) * 17.0;
                                    let rho_f = (um.busy_fraction_rho as f32).clamp(0.0, 1.0);
                                    let w_px = (rho_f * 165.0).max(2.0);
                                    let col = if rho_f >= 0.55 {
                                        Color32::from_rgb(244, 63, 94)
                                    } else if rho_f >= 0.35 {
                                        Color32::from_rgb(250, 204, 21)
                                    } else {
                                        Color32::from_rgb(52, 211, 153)
                                    };
                                    up.rect_filled(
                                        Rect::from_min_size(
                                            Pos2::new(u_rect.left() + 145.0, y_top + 2.0),
                                            EguiVec2::new(w_px, 11.5),
                                        ),
                                        2.0,
                                        col,
                                    );
                                    let u_short: String = um.label.chars().take(20).collect();
                                    up.text(
                                        Pos2::new(u_rect.left() + 5.0, y_top + 7.5),
                                        egui::Align2::LEFT_CENTER,
                                        u_short,
                                        egui::FontId::monospace(9.2),
                                        Color32::LIGHT_GRAY,
                                    );
                                    up.text(
                                        Pos2::new(u_rect.right() - 5.0, y_top + 7.5),
                                        egui::Align2::RIGHT_CENTER,
                                        format!(
                                            "rho={:.1}% ({:.0}s)",
                                            um.busy_fraction_rho * 100.0,
                                            um.mean_unit_travel_s
                                        ),
                                        egui::FontId::monospace(9.0),
                                        Color32::WHITE,
                                    );
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("HC CSV:");
                                ui.text_edit_singleline(&mut self.hypercube_csv_path);
                                if ui.small_button("Export .csv").clicked() {
                                    match hrep.export_csv_path(Path::new(&self.hypercube_csv_path)) {
                                        Ok(n_r) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_r} hypercube unit rows -> {}",
                                                self.hypercube_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV failed: {e}");
                                        }
                                    }
                                }
                            });
                        }

                        ui.add_space(5.0);
                        ui.label(
                            egui::RichText::new(
                                "Carter, Chaiken & Ignall (1972) / Larson Beat Redistricting:",
                            )
                            .strong(),
                        );
                        egui::ComboBox::from_id_salt("redist_mode_cb")
                            .selected_text(self.redistrict_cfg.objective_mode.label())
                            .show_ui(ui, |ui| {
                                for rm in [
                                    RedistrictingObjectiveMode::BalancedParetoKnee,
                                    RedistrictingObjectiveMode::TravelTimeOptimalCarter1972,
                                    RedistrictingObjectiveMode::StrictWorkloadEqualization,
                                ] {
                                    ui.selectable_value(
                                        &mut self.redistrict_cfg.objective_mode,
                                        rm,
                                        rm.label(),
                                    );
                                }
                            });
                        if self.redistrict_cfg.objective_mode
                            == RedistrictingObjectiveMode::BalancedParetoKnee
                        {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.redistrict_cfg.workload_equity_weight,
                                    0.0..=1.0,
                                )
                                .text("Equity Weight"),
                            );
                        }
                        ui.add(
                            egui::Slider::new(
                                &mut self.redistrict_cfg.max_travel_penalty_s,
                                0.0..=60.0,
                            )
                            .text("Max +E[T] Cap (s)"),
                        );
                        ui.horizontal(|ui| {
                            if ui
                                .button(
                                    egui::RichText::new("Optimize District Boundaries (Ctrl+Z)")
                                        .strong()
                                        .color(Color32::from_rgb(250, 204, 21)),
                                )
                                .on_hover_text(
                                    "Computes Carter-Chaiken-Ignall (1972) optimal dispatch offsets s_n over the exact 2^N hypercube and applies undoable beat boundaries",
                                )
                                .clicked()
                            {
                                if self.world.emergency_coverage.fire_station_count == 0 {
                                    let sited =
                                        self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                    self.world.apply_mclp_siting_solution(&sited);
                                    self.latest_mclp_report = Some(sited);
                                }
                                self.redistrict_cfg.hypercube = self.hypercube_cfg.clone();
                                self.redistrict_cfg.hypercube.threshold_travel_s =
                                    self.mclp_cfg.threshold_travel_s;
                                self.redistrict_cfg.hypercube.second_due_threshold_s =
                                    self.mclp_cfg.second_due_threshold_s;

                                // Optimize against clean baseline travel matrix, then apply undoably
                                let rrep =
                                    self.world.optimize_response_districts(&self.redistrict_cfg);
                                let shifted = self.world.apply_redistricting_plan_tracked(&rrep);
                                self.latest_hypercube_report =
                                    Some(rrep.optimized_hypercube.clone());
                                self.layer = HeatmapLayer::ResponseDistricts;
                                self.heatmap_dirty = true;
                                self.io_status_msg = format!(
                                    "Redistricted {} beats ({:.1}% demand, {:.1} ms): Spread {:.1}pt -> {:.1}pt ({:+.0}%), dE[T]={:+.1}s",
                                    shifted,
                                    rrep.reassigned_demand_pct,
                                    rrep.solve_time_ms,
                                    rrep.voronoi_workload_spread * 100.0,
                                    rrep.optimized_workload_spread * 100.0,
                                    -rrep.workload_spread_reduction_pct,
                                    rrep.travel_time_delta_s
                                );
                                self.latest_redistricting_report = Some(rrep);
                            }
                            if ui
                                .button("Voronoi Reset")
                                .on_hover_text("Reset all parcels to closest-unit Voronoi beats (s_n = 0)")
                                .clicked()
                            {
                                let cleared = self.world.clear_response_districts_tracked();
                                self.heatmap_dirty = true;
                                self.io_status_msg =
                                    format!("Reset {cleared} parcels to closest-unit Voronoi beats");
                            }
                        });

                        if let Some(rrep) = self.latest_redistricting_report.clone() {
                            ui.monospace(format!(
                                "Spread: {:.1}pt -> {:.1}pt ({:.1}% cut) | RMSE {:.2}% -> {:.2}%
E[T]  : {:.1}s -> {:.1}s ({:+.2}s) | Carter-72 Gain: -{:.2}s
Beats : {}/{} parcels shifted ({:.1}% demand) | {} HC evals",
                                rrep.voronoi_workload_spread * 100.0,
                                rrep.optimized_workload_spread * 100.0,
                                rrep.workload_spread_reduction_pct,
                                rrep.voronoi_workload_rmse * 100.0,
                                rrep.optimized_workload_rmse * 100.0,
                                rrep.voronoi_hypercube.mean_dispatch_travel_s,
                                rrep.optimized_hypercube.mean_dispatch_travel_s,
                                rrep.travel_time_delta_s,
                                rrep.carter_min_travel_gain_s,
                                rrep.reassigned_parcel_count,
                                rrep.num_parcels,
                                rrep.reassigned_demand_pct,
                                rrep.hypercube_evaluations,
                            ));

                            // Per-Unit Boundary Offset `s_n` & Busyness Equalization Readout
                            for us in rrep.unit_summaries.iter().take(6) {
                                let short_lbl: String = us.label.chars().take(18).collect();
                                ui.monospace(format!(
                                    "  {:<18} s={:+5.1}s | {:>2}->{:>2}p | rho {:.1}%->{:.1}%",
                                    short_lbl,
                                    us.boundary_offset_s,
                                    us.voronoi_parcel_count,
                                    us.optimized_parcel_count,
                                    us.voronoi_busy_rho * 100.0,
                                    us.optimized_busy_rho * 100.0,
                                ));
                            }

                            // Carter-Chaiken-Ignall Undominated Tradeoff Frontier Plot (X: Travel E[T], Y: Spread)
                            if rrep.pareto_frontier.len() >= 2 {
                                let (f_rect, _) = ui.allocate_exact_size(
                                    EguiVec2::new(360.0, 96.0),
                                    Sense::hover(),
                                );
                                let fp = ui.painter_at(f_rect);
                                fp.rect_filled(f_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                fp.rect_stroke(
                                    f_rect,
                                    4.0,
                                    Stroke::new(1.0, Color32::from_rgb(38, 52, 78)),
                                );

                                let mut min_t = rrep.voronoi_hypercube.mean_dispatch_travel_s;
                                let mut max_t = min_t;
                                let mut max_sp = rrep.voronoi_workload_spread;
                                for pt in &rrep.pareto_frontier {
                                    min_t = min_t.min(pt.mean_dispatch_travel_s);
                                    max_t = max_t.max(pt.mean_dispatch_travel_s);
                                    max_sp = max_sp.max(pt.workload_spread_rho);
                                }
                                let t_span = (max_t - min_t).max(1.5);
                                let sp_span = max_sp.max(0.02);

                                let map_pt = |t_s: f64, sp: f64| -> Pos2 {
                                    let nx = ((t_s - min_t) / t_span).clamp(0.0, 1.0) as f32;
                                    let ny = (sp / sp_span).clamp(0.0, 1.0) as f32;
                                    Pos2::new(
                                        f_rect.left() + 18.0 + nx * (f_rect.width() - 36.0),
                                        f_rect.bottom() - 15.0 - ny * (f_rect.height() - 30.0),
                                    )
                                };

                                let mut poly = Vec::with_capacity(rrep.pareto_frontier.len());
                                for pt in &rrep.pareto_frontier {
                                    let pos = map_pt(pt.mean_dispatch_travel_s, pt.workload_spread_rho);
                                    poly.push(pos);
                                    fp.circle_filled(pos, 2.8, Color32::from_rgb(56, 189, 248));
                                }
                                fp.add(egui::Shape::line(
                                    poly,
                                    Stroke::new(2.0, Color32::from_rgb(56, 189, 248)),
                                ));

                                // Mark Voronoi (`s = 0`) in crimson and active Optimal Plan in gold!
                                let vor_pos = map_pt(
                                    rrep.voronoi_hypercube.mean_dispatch_travel_s,
                                    rrep.voronoi_workload_spread,
                                );
                                fp.circle_stroke(
                                    vor_pos,
                                    4.5,
                                    Stroke::new(1.8, Color32::from_rgb(244, 63, 94)),
                                );
                                let opt_pos = map_pt(
                                    rrep.optimized_hypercube.mean_dispatch_travel_s,
                                    rrep.optimized_workload_spread,
                                );
                                fp.circle_filled(opt_pos, 4.4, Color32::from_rgb(250, 204, 21));

                                fp.text(
                                    Pos2::new(f_rect.left() + 8.0, f_rect.top() + 3.0),
                                    egui::Align2::LEFT_TOP,
                                    "Carter-1972 Frontier: E[T](X) vs Spread(Y) | Red=Voronoi Gold=Active",
                                    egui::FontId::monospace(8.3),
                                    Color32::GRAY,
                                );
                                fp.text(
                                    Pos2::new(f_rect.left() + 8.0, f_rect.bottom() - 2.0),
                                    egui::Align2::LEFT_BOTTOM,
                                    format!("E[T]: {:.1}s .. {:.1}s", min_t, max_t),
                                    egui::FontId::monospace(8.3),
                                    Color32::LIGHT_GRAY,
                                );
                            }

                            ui.horizontal(|ui| {
                                ui.label("Beat CSV:");
                                ui.text_edit_singleline(&mut self.redistrict_csv_path);
                                if ui.small_button("Units .csv").clicked() {
                                    match rrep.export_csv_path(Path::new(&self.redistrict_csv_path)) {
                                        Ok(n_r) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_r} redistricted unit rows -> {}",
                                                self.redistrict_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV failed: {e}");
                                        }
                                    }
                                }
                                if ui.small_button("Pareto .csv").clicked() {
                                    match rrep.export_pareto_csv_path(Path::new(
                                        &self.redistrict_pareto_csv_path,
                                    )) {
                                        Ok(n_r) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_r} Pareto tradeoff rows -> {}",
                                                self.redistrict_pareto_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV failed: {e}");
                                        }
                                    }
                                }
                            });
                        }

                        ui.add_space(5.0);
                        ui.label(
                            egui::RichText::new(
                                "Diurnal 24-Hr Call & Rush-Hour Peak/Off-Peak Dispatch Tables:",
                            )
                            .strong(),
                        );
                        egui::ComboBox::from_id_salt("diurnal_win_cb")
                            .selected_text(self.diurnal_cfg.window_preset.label())
                            .show_ui(ui, |ui| {
                                for wp in [
                                    DiurnalWindowPreset::BimodalCommutePeak8Hr,
                                    DiurnalWindowPreset::DaytimeCommercial12Hr,
                                ] {
                                    ui.selectable_value(
                                        &mut self.diurnal_cfg.window_preset,
                                        wp,
                                        wp.label(),
                                    );
                                }
                            });
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.diurnal_cfg.daily_mean_calls_per_hr,
                                    0.5..=12.0,
                                )
                                .text("24h Calls/hr"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(&mut self.diurnal_cfg.rush_vc_scale, 0.4..=1.8)
                                    .text("Rush V/C"),
                            );
                            ui.checkbox(
                                &mut self.world.diurnal_live_clock_enabled,
                                "Live 24h Clock",
                            );
                        });
                        ui.add(
                            egui::Slider::new(&mut self.world.diurnal_clock_hour, 0.0..=23.9)
                                .text("Clock Hr"),
                        );

                        ui.horizontal(|ui| {
                            if ui
                                .button(
                                    egui::RichText::new("Solve Peak/Off-Peak Tables (Ctrl+Z)")
                                        .strong()
                                        .color(Color32::from_rgb(52, 211, 153)),
                                )
                                .on_hover_text(
                                    "Solves 24-hour USFA call + BPR rush-hour profiles (Green-Kolesar PSA) and installs separate Peak & Off-Peak CAD dispatch tables",
                                )
                                .clicked()
                            {
                                if self.world.emergency_coverage.fire_station_count == 0 {
                                    let sited =
                                        self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                    self.world.apply_mclp_siting_solution(&sited);
                                    self.latest_mclp_report = Some(sited);
                                }
                                self.diurnal_cfg.redistrict = self.redistrict_cfg.clone();
                                self.diurnal_cfg.redistrict.hypercube = self.hypercube_cfg.clone();
                                let drep =
                                    self.world.solve_diurnal_dispatch_tables(&self.diurnal_cfg);
                                let cur_h = (self.world.diurnal_clock_hour.floor() as isize)
                                    .rem_euclid(24) as u8;
                                let init_reg =
                                    if self.diurnal_cfg.window_preset.is_peak_hour(cur_h) {
                                        DiurnalDispatchRegime::PeakRush
                                    } else {
                                        DiurnalDispatchRegime::OffPeakFreeFlow
                                    };
                                let sw = self
                                    .world
                                    .apply_diurnal_dispatch_tables_tracked(&drep, init_reg);
                                self.active_diurnal_regime = init_reg;
                                self.latest_redistricting_report = Some(match init_reg {
                                    DiurnalDispatchRegime::PeakRush => {
                                        drep.peak_table.redistricting_report.clone()
                                    }
                                    DiurnalDispatchRegime::OffPeakFreeFlow => {
                                        drep.offpeak_table.redistricting_report.clone()
                                    }
                                });
                                self.layer = HeatmapLayer::ResponseDistricts;
                                self.heatmap_dirty = true;
                                self.io_status_msg = format!(
                                    "Built 24h Peak/Off-Peak Tables ({:.1} ms): {} primary / {} 1st+2nd beat switches, 24h E[T] {:.1}s",
                                    drep.solve_time_ms,
                                    sw,
                                    drep.any_rank12_switches_count,
                                    drep.daily_dual_table_mean_travel_s
                                );
                                self.latest_diurnal_report = Some(drep);
                            }
                        });

                        if let Some(drep) = self.latest_diurnal_report.clone() {
                            ui.horizontal(|ui| {
                                let pk_sel =
                                    self.active_diurnal_regime == DiurnalDispatchRegime::PeakRush;
                                if ui
                                    .selectable_label(pk_sel, "Active: PEAK RUSH Table")
                                    .clicked()
                                {
                                    self.active_diurnal_regime = DiurnalDispatchRegime::PeakRush;
                                    self.world.diurnal_clock_hour = 17.0;
                                    let upd = self
                                        .world
                                        .activate_diurnal_regime_tracked(DiurnalDispatchRegime::PeakRush);
                                    self.latest_redistricting_report =
                                        Some(drep.peak_table.redistricting_report.clone());
                                    self.layer = HeatmapLayer::ResponseDistricts;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Activated PEAK RUSH Dispatch Table ({upd} parcels switched)"
                                    );
                                }
                                if ui
                                    .selectable_label(!pk_sel, "Active: OFF-PEAK Table")
                                    .clicked()
                                {
                                    self.active_diurnal_regime =
                                        DiurnalDispatchRegime::OffPeakFreeFlow;
                                    self.world.diurnal_clock_hour = 3.0;
                                    let upd = self.world.activate_diurnal_regime_tracked(
                                        DiurnalDispatchRegime::OffPeakFreeFlow,
                                    );
                                    self.latest_redistricting_report =
                                        Some(drep.offpeak_table.redistricting_report.clone());
                                    self.layer = HeatmapLayer::ResponseDistricts;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Activated OFF-PEAK Free-Flow Dispatch Table ({upd} parcels switched)"
                                    );
                                }
                            });

                            ui.monospace(format!(
                                "Peak  ({:>2}h): {:.2}/hr | V/C={:.2} (+{:.0}% BPR) | E[T]={:.1}s | Sprd={:.1}pt
OffPk ({:>2}h): {:.2}/hr | V/C={:.2} (+{:.0}% BPR) | E[T]={:.1}s | Sprd={:.1}pt
24h PSA    : Vor Sprd {:.1}pt -> Dual {:.1}pt | Flip: {} 1st / {} 1st+2nd",
                                drep.peak_table.active_hours.len(),
                                drep.peak_table.mean_calls_per_hr,
                                drep.peak_table.mean_arterial_vc,
                                drep.peak_table.mean_bpr_slowdown_pct,
                                drep.peak_table
                                    .redistricting_report
                                    .optimized_hypercube
                                    .mean_dispatch_travel_s,
                                drep.peak_table.redistricting_report.optimized_workload_spread
                                    * 100.0,
                                drep.offpeak_table.active_hours.len(),
                                drep.offpeak_table.mean_calls_per_hr,
                                drep.offpeak_table.mean_arterial_vc,
                                drep.offpeak_table.mean_bpr_slowdown_pct,
                                drep.offpeak_table
                                    .redistricting_report
                                    .optimized_hypercube
                                    .mean_dispatch_travel_s,
                                drep.offpeak_table
                                    .redistricting_report
                                    .optimized_workload_spread
                                    * 100.0,
                                drep.daily_voronoi_workload_spread * 100.0,
                                drep.daily_dual_table_workload_spread * 100.0,
                                drep.primary_beat_switches_count,
                                drep.any_rank12_switches_count,
                            ));

                            // 24-Hour Diurnal Call Bar + BPR Response Travel Time Curve (00:00..23:00)
                            let (c_rect, _) = ui.allocate_exact_size(
                                EguiVec2::new(360.0, 94.0),
                                Sense::hover(),
                            );
                            let cp = ui.painter_at(c_rect);
                            cp.rect_filled(c_rect, 4.0, Color32::from_rgb(10, 15, 28));
                            cp.rect_stroke(
                                c_rect,
                                4.0,
                                Stroke::new(1.0, Color32::from_rgb(38, 52, 78)),
                            );

                            let max_lam = drep
                                .hourly_trajectory
                                .iter()
                                .map(|h| h.calls_per_hr)
                                .fold(1.0, f64::max);
                            let mut min_et = f64::INFINITY;
                            let mut max_et = 1.0f64;
                            for hm in &drep.hourly_trajectory {
                                min_et = min_et
                                    .min(hm.voronoi_mean_travel_s)
                                    .min(hm.dual_table_mean_travel_s);
                                max_et = max_et
                                    .max(hm.voronoi_mean_travel_s)
                                    .max(hm.dual_table_mean_travel_s);
                            }
                            let et_span = (max_et - min_et).max(2.0);
                            let plot_w = c_rect.width() - 20.0;
                            let bar_step = plot_w / 24.0;

                            let mut vor_line = Vec::with_capacity(24);
                            let mut dual_line = Vec::with_capacity(24);
                            for hm in &drep.hourly_trajectory {
                                let x_mid =
                                    c_rect.left() + 10.0 + (hm.hour as f32 + 0.5) * bar_step;
                                let bh = ((hm.calls_per_hr / max_lam).clamp(0.0, 1.0) as f32)
                                    * (c_rect.height() - 26.0);
                                let bcol = if hm.is_peak_regime {
                                    Color32::from_rgba_unmultiplied(251, 146, 60, 92)
                                } else {
                                    Color32::from_rgba_unmultiplied(56, 189, 248, 56)
                                };
                                cp.rect_filled(
                                    Rect::from_min_max(
                                        Pos2::new(x_mid - bar_step * 0.38, c_rect.bottom() - 12.0 - bh),
                                        Pos2::new(x_mid + bar_step * 0.38, c_rect.bottom() - 12.0),
                                    ),
                                    1.5,
                                    bcol,
                                );

                                let map_et = |v: f64| -> f32 {
                                    c_rect.bottom()
                                        - 13.0
                                        - (((v - min_et) / et_span).clamp(0.0, 1.0) as f32)
                                            * (c_rect.height() - 28.0)
                                };
                                vor_line.push(Pos2::new(x_mid, map_et(hm.voronoi_mean_travel_s)));
                                dual_line
                                    .push(Pos2::new(x_mid, map_et(hm.dual_table_mean_travel_s)));
                            }
                            cp.add(egui::Shape::line(
                                vor_line,
                                Stroke::new(1.4, Color32::from_rgb(244, 63, 94)),
                            ));
                            cp.add(egui::Shape::line(
                                dual_line,
                                Stroke::new(2.1, Color32::from_rgb(52, 211, 153)),
                            ));

                            // Current simulated diurnal clock vertical cursor
                            let cur_x = c_rect.left()
                                + 10.0
                                + ((self.world.diurnal_clock_hour as f32 / 24.0).clamp(0.0, 1.0))
                                    * plot_w;
                            cp.line_segment(
                                [
                                    Pos2::new(cur_x, c_rect.top() + 12.0),
                                    Pos2::new(cur_x, c_rect.bottom() - 12.0),
                                ],
                                Stroke::new(1.4, Color32::from_rgb(250, 204, 21)),
                            );
                            cp.text(
                                Pos2::new(c_rect.left() + 6.0, c_rect.top() + 2.0),
                                egui::Align2::LEFT_TOP,
                                "24h Calls(Bars: Org=Peak) & BPR E[T] (Red=Voronoi Grn=Dual)",
                                egui::FontId::monospace(8.2),
                                Color32::GRAY,
                            );
                            cp.text(
                                Pos2::new(c_rect.left() + 6.0, c_rect.bottom() - 1.0),
                                egui::Align2::LEFT_BOTTOM,
                                format!(
                                    "00h      06h      12h      18h     23h | {:02}:00",
                                    (self.world.diurnal_clock_hour.floor() as usize) % 24
                                ),
                                egui::FontId::monospace(8.2),
                                Color32::LIGHT_GRAY,
                            );

                            ui.horizontal(|ui| {
                                ui.label("24h CSV:");
                                ui.text_edit_singleline(&mut self.diurnal_table_csv_path);
                                if ui.small_button("Tables .csv").clicked() {
                                    match drep.export_dispatch_table_csv_path(Path::new(
                                        &self.diurnal_table_csv_path,
                                    )) {
                                        Ok(n_r) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_r} Peak/Off-Peak run-card rows -> {}",
                                                self.diurnal_table_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV failed: {e}");
                                        }
                                    }
                                }
                                if ui.small_button("24h .csv").clicked() {
                                    match drep.export_hourly_csv_path(Path::new(
                                        &self.diurnal_hourly_csv_path,
                                    )) {
                                        Ok(n_r) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_r} diurnal hourly PSA rows -> {}",
                                                self.diurnal_hourly_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV failed: {e}");
                                        }
                                    }
                                }
                            });
                        }
                        ui.separator();
                        ui.label("Network-Wide Varaiya (2013) Max-Pressure Control:");
                        ui.horizontal(|ui| {
                            if ui.button("Enable Citywide Max-Pressure").clicked() {
                                let mut upgraded = 0usize;
                                self.world.execute_topology_edit(
                                    EditCategory::SignalControl,
                                    "Enable Citywide Max-Pressure Adaptive",
                                    |w| {
                                        upgraded = w.network.enable_network_max_pressure();
                                    },
                                );
                                self.io_status_msg = format!(
                                    "Upgraded {upgraded} junctions to Max-Pressure Adaptive"
                                );
                            }
                            if ui.button("Revert Fixed-Time").clicked() {
                                let mut reverted = 0usize;
                                self.world.execute_topology_edit(
                                    EditCategory::SignalControl,
                                    "Revert Signals to Fixed-Time 2-Phase",
                                    |w| {
                                        reverted = w.network.restore_fixed_time_signals();
                                    },
                                );
                                self.io_status_msg = format!(
                                    "Reverted {reverted} junctions to Fixed-Time 2-Phase"
                                );
                            }
                        });
                        ui.add(
                            egui::Slider::new(&mut self.world.mp_config.min_green_s, 4.0..=20.0)
                                .text("MP Min Green (s)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.world.mp_config.max_green_s, 20.0..=90.0)
                                .text("MP Max Green (s)"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.world.mp_config.hysteresis_ratio,
                                0.0..=0.50,
                            )
                            .text("Switching Hysteresis"),
                        );
                        ui.checkbox(
                            &mut self.world.mp_config.capacity_aware_spillback,
                            "Capacity-Aware Spillback Gate (Gregoire 2015)",
                        );
                        ui.checkbox(
                            &mut self.world.mp_config.passenger_weighted_transit,
                            "GTFS Bus Passenger-Weighted Pressure",
                        );
                        ui.checkbox(
                            &mut self.world.mp_config.allow_protected_left_phases,
                            "Protected-Left MP Phases (3+ Lane Approaches)",
                        );
                        let mpt = self.world.mp_telemetry;
                        if mpt.controlled_junctions > 0 {
                            ui.label(format!(
                                "MP Telemetry: {} juncs | Mean P {:.2} | Switches {} | Spillback-Gated {}",
                                mpt.controlled_junctions,
                                mpt.mean_junction_pressure,
                                mpt.total_phase_switches,
                                mpt.spillback_gated_movements
                            ));
                        }
                        ui.separator();
                        if let Some(seg_ro) = self.world.network.segments.get(self.selected_segment)
                        {
                            let sid = seg_ro.id;
                            let mut closed_val = seg_ro.closed;
                            let mut bus_val = seg_ro.has_bus_lane;
                            let mut bike_val = seg_ro.has_bike_lane;
                            ui.label(format!(
                                "Corridor #{}: {} ({}, {} ln/dir)",
                                sid, seg_ro.name, seg_ro.highway, seg_ro.lane_count
                            ));
                            let c1 = ui
                                .checkbox(&mut closed_val, "Close Roadway (Dynamic Reroute)")
                                .changed();
                            let c2 = ui
                                .checkbox(&mut bus_val, "Dedicated Red Bus/BRT Lane + TSP")
                                .changed();
                            let c3 = ui
                                .checkbox(&mut bike_val, "Protected Emerald Bike Lane")
                                .changed();
                            if c1 || c2 || c3 {
                                self.world.execute_topology_edit(
                                    EditCategory::CorridorPolicy,
                                    format!("Update Complete Streets flags on corridor #{sid}"),
                                    |w| {
                                        if let Some(s) = w.network.segments.get_mut(sid) {
                                            s.closed = closed_val;
                                            s.has_bus_lane = bus_val;
                                            s.has_bike_lane = bike_val;
                                        }
                                        w.network.recompute_dynamic_routes();
                                        w.rebuild_cyclists();
                                        w.refresh_urban_metrics();
                                    },
                                );
                                self.heatmap_dirty = true;
                            }
                        }
                        ui.horizontal(|ui| {
                            if ui.button("+1 Lane (Widen)").clicked() {
                                match adjust_segment_lane_count(
                                    &mut self.world,
                                    self.selected_segment,
                                    1,
                                    self.symmetric_lane_edit,
                                ) {
                                    Ok(nl) => {
                                        self.io_status_msg = format!(
                                            "Widened corridor #{} to {nl} lanes/dir",
                                            self.selected_segment
                                        );
                                        self.heatmap_dirty = true;
                                    }
                                    Err(e) => self.io_status_msg = format!("Edit error: {e}"),
                                }
                            }
                            if ui.button("-1 Lane (Road Diet)").clicked() {
                                match adjust_segment_lane_count(
                                    &mut self.world,
                                    self.selected_segment,
                                    -1,
                                    self.symmetric_lane_edit,
                                ) {
                                    Ok(nl) => {
                                        self.io_status_msg = format!(
                                            "Road diet on corridor #{} -> {nl} lanes/dir",
                                            self.selected_segment
                                        );
                                        self.heatmap_dirty = true;
                                    }
                                    Err(e) => self.io_status_msg = format!("Edit error: {e}"),
                                }
                            }
                        });
                        if ui
                            .button("Convert Selected Junction to Modern Roundabout")
                            .clicked()
                        {
                            match convert_junction_to_roundabout(
                                &mut self.world,
                                self.selected_junction,
                                self.roundabout_radius_m,
                                self.roundabout_lanes,
                            ) {
                                Ok(sum) => {
                                    self.io_status_msg = format!(
                                        "Built {}-lane roundabout ({}-arm, R={:.1}m)",
                                        sum.ring_lanes, sum.arms_connected, sum.inscribed_radius_m
                                    );
                                    self.heatmap_dirty = true;
                                }
                                Err(e) => self.io_status_msg = format!("Roundabout: {e}"),
                            }
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "History Stack ({} undo / {} redo):",
                                self.world.edit_history.undo_len(),
                                self.world.edit_history.redo_len()
                            ));
                            if (self.world.edit_history.can_undo()
                                || self.world.edit_history.can_redo())
                                && ui.small_button("Clear").clicked()
                            {
                                self.world.edit_history.clear();
                                self.io_status_msg = "Cleared command history stack.".to_string();
                            }
                        });
                        for rec in self.world.edit_history.undo_entries().iter().rev().take(6) {
                            ui.small(format!(
                                "• #{} [{}] {}",
                                rec.id,
                                rec.category.short_tag(),
                                rec.label
                            ));
                        }
                        if let Some(next_r) = self.world.edit_history.peek_redo() {
                            ui.colored_label(
                                Color32::from_rgb(125, 211, 252),
                                format!(
                                    "Redo ready: [{}] {}",
                                    next_r.category.short_tag(),
                                    next_r.label
                                ),
                            );
                        }
                    }
                    DockTab::ZoningAndTransit => {
                        ui.label("Click any city parcel block on map to paint selected zone:");
                        ui.horizontal_wrapped(|ui| {
                            for z in [
                                ZoneCode::R1,
                                ZoneCode::R3,
                                ZoneCode::MU4,
                                ZoneCode::CBD,
                                ZoneCode::IND,
                                ZoneCode::PARK,
                            ] {
                                ui.selectable_value(&mut self.active_brush, z, z.spec().label);
                            }
                        });
                        if ui
                            .button("Apply 15-Minute TOD Upzoning + BRT + Bike Grid")
                            .clicked()
                        {
                            self.world.apply_tod_upzoning_tracked();
                            self.io_status_msg =
                                "Applied 15-Min TOD Upzoning + BRT (Ctrl+Z to undo)".to_string();
                            self.heatmap_dirty = true;
                        }
                        ui.separator();
                        let ms = self.world.mode_shares;
                        ui.label(format!(
                            "Mode Split: Auto {:.0}% | Bus {:.0}% | Bike {:.0}% | Walk {:.0}%",
                            ms.auto * 100.0,
                            ms.transit * 100.0,
                            ms.bike * 100.0,
                            ms.walk * 100.0
                        ));
                        let mut pol_changed = false;
                        pol_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut self.world.transit_policy.headway_min,
                                    3.0..=30.0,
                                )
                                .text("Bus Headway (min)"),
                            )
                            .changed();
                        pol_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut self.world.transit_policy.fare_usd,
                                    0.0..=4.0,
                                )
                                .text("Transit Fare ($)"),
                            )
                            .changed();
                        pol_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut self.world.transit_policy.cordon_pricing_usd,
                                    0.0..=15.0,
                                )
                                .text("Cordon Price ($)"),
                            )
                            .changed();
                        if pol_changed {
                            self.world.refresh_urban_metrics();
                            self.heatmap_dirty = true;
                        }
                        ui.separator();
                        ui.label("GTFS Fixed-Route Operations & TCQSM Dwell/Crowding:");
                        egui::ComboBox::from_id_salt("gtfs_fare_mode")
                            .selected_text(self.world.transit_gtfs.fare_mode.label())
                            .show_ui(ui, |ui| {
                                for m in [
                                    BoardingFareMode::AllDoorFareFreeBrt,
                                    BoardingFareMode::SmartcardTap,
                                    BoardingFareMode::FrontDoorCash,
                                ] {
                                    ui.selectable_value(
                                        &mut self.world.transit_gtfs.fare_mode,
                                        m,
                                        m.label(),
                                    );
                                }
                            });
                        egui::ComboBox::from_id_salt("gtfs_coach_spec")
                            .selected_text(self.world.transit_gtfs.coach_spec.label())
                            .show_ui(ui, |ui| {
                                for c in [
                                    BusCoachSpec::Standard40Ft,
                                    BusCoachSpec::Articulated60FtBrt,
                                ] {
                                    ui.selectable_value(
                                        &mut self.world.transit_gtfs.coach_spec,
                                        c,
                                        c.label(),
                                    );
                                }
                            });
                        ui.checkbox(
                            &mut self.world.transit_gtfs.hold_early_buses,
                            "Anti-Bunching Headway Holding at Stops",
                        );
                        let tel = self.world.transit_gtfs.telemetry;
                        ui.label(format!(
                            "GTFS Network: {} routes | {} curbside stops | {} active buses",
                            tel.route_count, tel.stop_count, tel.active_scheduled_buses
                        ));
                        ui.label(format!(
                            "Ridership: {} onboard | {} waiting at curb | {} total boardings",
                            tel.onboard_passengers,
                            tel.waiting_passengers,
                            tel.cumulative_boardings
                        ));
                        ui.label(format!(
                            "Crowding: {:.0}% mean ({:.0}% peak) | Load LOS {} | Pass-ups: {}",
                            tel.mean_load_factor_pct,
                            tel.peak_load_factor_pct,
                            tel.tcqsm_crowding_los,
                            tel.denied_passups
                        ));
                        ui.label(format!(
                            "Mean Stop Dwell: {:.1} s | Bus Bunching: {:.0}%",
                            tel.mean_dwell_s, tel.bunching_pct
                        ));
                    }
                    DockTab::IdmMobilAndCrowd => {
                        ui.label("wgpu WGSL Instanced Crowd Scaler:");
                        ui.add(
                            egui::Slider::new(&mut self.target_crowd_slider, 5_000..=100_000)
                                .step_by(2_500.0)
                                .text("Target Agents"),
                        );
                        if ui
                            .button("Seed Crowd on Current Network (Vehicles + Bikes + Peds)")
                            .clicked()
                        {
                            self.world.seed_50k_megacity_crowd(self.target_crowd_slider);
                            self.heatmap_dirty = true;
                        }
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.speed_scale, 0.25..=4.0)
                                .text("Sim Speed Scale"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.world.base_demand_per_lane_hr,
                                80.0..=1100.0,
                            )
                            .text("Base Flow (veh/h/ln)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.world.idm.time_headway_s, 0.45..=2.8)
                                .text("IDM Headway T (s)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.world.idm.min_jam_gap_m, 1.0..=5.0)
                                .text("Jam Gap s0 (m)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.world.idm.max_accel_mps2, 0.6..=3.5)
                                .text("Max Accel a (m/s2)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.world.mobil.politeness, 0.0..=1.0)
                                .text("MOBIL Politeness p"),
                        );
                    }
                    DockTab::CityAndShockwave => {
                        let env = self.world.env_snapshot;
                        let led = &self.world.ledger;
                        ui.label(format!(
                            "Residents: {} | Jobs: {}",
                            led.total_residents, led.total_jobs
                        ));
                        ui.label(format!(
                            "15-Min Accessibility: {:.0}% of parcels",
                            env.access_15min_pct
                        ));
                        ui.label(format!(
                            "VSP Tailpipe: {:.1} kg CO2/h | {:.1} g PM2.5/h",
                            env.co2_kg_per_hr, env.pm25_g_per_hr
                        ));
                        ui.label(format!(
                            "Noise: {:.1} dBA | Heat Island: {:+.1} C",
                            env.corridor_noise_dba, env.heat_island_delta_c
                        ));
                        ui.label(format!(
                            "Stormwater Runoff: {:.0} L/s",
                            env.stormwater_runoff_lps
                        ));
                        ui.label(format!(
                            "Net Fiscal Balance: {:+.1}k USD/acre/yr",
                            led.net_fiscal_per_acre_k_yr
                        ));
                        ui.separator();
                        ui.label("wgpu @compute Advection-Diffusion Plume Controls:");
                        ui.add(
                            egui::Slider::new(&mut self.plume_params.wind_x_mps, -12.0..=12.0)
                                .text("Wind U East (m/s)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.plume_params.wind_y_mps, -12.0..=12.0)
                                .text("Wind V South (m/s)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.plume_params.diffusivity, 0.05..=0.90)
                                .text("Eddy Diffusivity"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.plume_params.decay_rate, 0.15..=2.50)
                                .text("Plume Decay (1/s)"),
                        );
                        ui.separator();
                        ui.label(format!(
                            "Persistence & 1 Hz Telemetry ({} CSV rows buffered):",
                            self.world.telemetry_log.len()
                        ));
                        ui.horizontal(|ui| {
                            ui.label("JSON:");
                            ui.text_edit_singleline(&mut self.snapshot_path);
                        });
                        ui.checkbox(
                            &mut self.save_include_agents,
                            "Include Microscopic Agents in .json (<=15k)",
                        );
                        ui.horizontal(|ui| {
                            ui.label("CSV: ");
                            ui.text_edit_singleline(&mut self.csv_export_path);
                        });
                        if ui.button("Clear Buffered CSV Rows").clicked() {
                            self.world.telemetry_log.clear();
                            self.io_status_msg = "Cleared CSV telemetry buffer.".to_string();
                        }
                        ui.separator();
                        ui.label("Space-Time (x-t) Corridor Shockwave Plot:");
                        let (rect, _) =
                            ui.allocate_exact_size(EguiVec2::new(320.0, 180.0), Sense::hover());
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 4.0, Color32::from_rgb(8, 12, 22));
                        let t_now = self.world.sim_time_s;
                        let t_min = t_now - 60.0;
                        for col in &self.world.xt_history {
                            if col.sim_time_s < t_min {
                                continue;
                            }
                            let x = rect.left()
                                + ((col.sim_time_s - t_min) / 60.0) as f32 * rect.width();
                            for &(pnorm, mph) in &col.samples {
                                let y = rect.top() + (1.0 - pnorm) * rect.height();
                                let c = if mph < 4.5 {
                                    Color32::from_rgb(244, 63, 94)
                                } else if mph < 14.0 {
                                    Color32::from_rgb(251, 146, 60)
                                } else {
                                    Color32::from_rgb(52, 211, 153)
                                };
                                painter.rect_filled(
                                    Rect::from_center_size(Pos2::new(x, y), EguiVec2::splat(2.5)),
                                    0.0,
                                    c,
                                );
                            }
                        }
                    }
                    DockTab::BatchSweep => {
                        ui.label(
                            egui::RichText::new(
                                "Async Rayon Regime Sweep (Fixed vs. Max-Pressure vs. Roundabout)",
                            )
                            .strong(),
                        );
                        ui.small(
                            "Clones your live edited street network & parcels onto an isolated background Rayon thread pool so the 120 FPS viewport never stalls.",
                        );
                        ui.separator();

                        ui.add_enabled_ui(self.active_sweep.is_none(), |ui| {
                            ui.add(
                                egui::Slider::new(&mut self.sweep_cfg.duration_s, 45.0..=480.0)
                                    .step_by(15.0)
                                    .text("Duration (s)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.sweep_cfg.warmup_s, 10.0..=90.0)
                                    .step_by(5.0)
                                    .text("Warmup (s)"),
                            );
                            ui.horizontal(|ui| {
                                ui.label("Demand Scales:");
                                ui.text_edit_singleline(&mut self.sweep_scales_input);
                            });
                            ui.add(
                                egui::Slider::new(&mut self.sweep_cfg.replicates, 1..=4)
                                    .text("CRN Replicates"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.sweep_cfg.max_roundabouts, 1..=12)
                                    .text("Max Roundabouts"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.sweep_cfg.roundabout_radius_m,
                                    14.0..=28.0,
                                )
                                .text("Roundabout R (m)"),
                            );
                            ui.checkbox(
                                &mut self.sweep_cfg.include_tod_upzone,
                                "Include 15-Min TOD Upzoning + BRT Across All Regimes",
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.sweep_bg_threads,
                                    1..=self.sweep_max_threads,
                                )
                                .text("Bg Rayon Threads"),
                            );
                        });

                        ui.horizontal(|ui| {
                            if self.active_sweep.is_none() {
                                if ui
                                    .button(
                                        egui::RichText::new("Launch Async Rayon Sweep")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    self.launch_background_sweep();
                                }
                            } else if let Some(active) = self.active_sweep.as_ref() {
                                let (done_r, tot_r, frac) = active.tracker.snapshot_progress();
                                ui.add(
                                    egui::ProgressBar::new(frac)
                                        .desired_width(245.0)
                                        .text(format!(
                                            "{done_r}/{tot_r} runs ({:.0}%) on {}t",
                                            frac * 100.0,
                                            active.worker_threads
                                        ))
                                        .animate(true),
                                );
                                if ui.button("Cancel").clicked() {
                                    active.tracker.request_cancel();
                                }
                            }
                        });

                        if let Some(rep) = self.latest_sweep_report.clone() {
                            ui.separator();
                            ui.colored_label(
                                Color32::from_rgb(125, 211, 252),
                                format!(
                                    "Completed {} parallel runs in {:.0} ms ({:.0}x real-time)",
                                    rep.total_parallel_runs, rep.wall_clock_ms, rep.realtime_speedup_x
                                ),
                            );

                            // Find highest mean speed winner in REGIME_MEAN_SUMMARY
                            let best_regime = rep
                                .regime_summary_rows
                                .iter()
                                .max_by(|a, b| {
                                    a.mean_speed_mph
                                        .partial_cmp(&b.mean_speed_mph)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });

                            if let Some(win) = best_regime {
                                ui.label(format!(
                                    "Top Mean Speed: {} ({:.2} mph, LOS {}, {:+.1}% vs Fixed)",
                                    win.regime_title,
                                    win.mean_speed_mph,
                                    win.hcm_los,
                                    win.speed_delta_vs_fixed_pct
                                ));
                                if ui
                                    .button(format!(
                                        "Apply `{}` to Live Viewport (Ctrl+Z Undoable)",
                                        win.regime_title
                                    ))
                                    .clicked()
                                {
                                    let win_reg = win.regime;
                                    let rb_r = rep.config.roundabout_radius_m;
                                    let rb_ln = rep.config.roundabout_lanes;
                                    let rb_max = rep.config.max_roundabouts;
                                    match win_reg {
                                        BatchControlRegime::FixedTime => {
                                            self.world.execute_topology_edit(
                                                EditCategory::SignalControl,
                                                "Apply Sweep Winner: Fixed-Time Signals",
                                                |w| {
                                                    w.network.restore_fixed_time_signals();
                                                },
                                            );
                                        }
                                        BatchControlRegime::MaxPressureAdaptive => {
                                            self.world.execute_topology_edit(
                                                EditCategory::SignalControl,
                                                "Apply Sweep Winner: Max-Pressure Adaptive",
                                                |w| {
                                                    w.network.enable_network_max_pressure();
                                                },
                                            );
                                        }
                                        BatchControlRegime::ModernRoundabout => {
                                            self.world.execute_topology_edit(
                                                EditCategory::ConvertRoundabout,
                                                "Apply Sweep Winner: Arterial Modern Roundabouts",
                                                |w| {
                                                    w.network.restore_fixed_time_signals();
                                                    convert_network_candidate_roundabouts(
                                                        w, rb_r, rb_ln, rb_max,
                                                    );
                                                },
                                            );
                                            self.clamp_selection_indices();
                                        }
                                    }
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Applied `{}` onto live map (Ctrl+Z to undo).",
                                        win.regime_title
                                    );
                                }
                            }

                            // Visual Regime Comparison Bar Chart (`REGIME_MEAN_SUMMARY`)
                            ui.add_space(4.0);
                            ui.small("Cross-Demand Regime Comparison (Speed mph & Avg Queue):");
                            let (chart_rect, _) = ui.allocate_exact_size(
                                EguiVec2::new(365.0, 108.0),
                                Sense::hover(),
                            );
                            let cp = ui.painter_at(chart_rect);
                            cp.rect_filled(chart_rect, 4.0, Color32::from_rgb(10, 15, 28));
                            let max_spd = rep
                                .regime_summary_rows
                                .iter()
                                .map(|r| r.mean_speed_mph)
                                .fold(1.0_f64, f64::max);
                            let palette = [
                                Color32::from_rgb(148, 163, 184),
                                Color32::from_rgb(56, 189, 248),
                                Color32::from_rgb(16, 185, 129),
                            ];
                            for (idx, r) in rep.regime_summary_rows.iter().enumerate() {
                                let y0 = chart_rect.top() + 10.0 + idx as f32 * 32.0;
                                let bar_max_w = 185.0_f32;
                                let w_px = ((r.mean_speed_mph / max_spd) as f32 * bar_max_w)
                                    .clamp(6.0, bar_max_w);
                                let col = palette[idx % palette.len()];
                                cp.text(
                                    Pos2::new(chart_rect.left() + 8.0, y0 + 8.0),
                                    egui::Align2::LEFT_CENTER,
                                    match r.regime {
                                        BatchControlRegime::FixedTime => "Fixed-Time",
                                        BatchControlRegime::MaxPressureAdaptive => "Max-Pressure",
                                        BatchControlRegime::ModernRoundabout => "Roundabout",
                                    },
                                    egui::FontId::monospace(11.0),
                                    Color32::LIGHT_GRAY,
                                );
                                let bar_rect = Rect::from_min_size(
                                    Pos2::new(chart_rect.left() + 98.0, y0 + 1.0),
                                    EguiVec2::new(w_px, 15.0),
                                );
                                cp.rect_filled(bar_rect, 3.0, col);
                                cp.text(
                                    Pos2::new(bar_rect.right() + 6.0, y0 + 8.0),
                                    egui::Align2::LEFT_CENTER,
                                    format!(
                                        "{:.1} mph | Q {:.1}",
                                        r.mean_speed_mph, r.avg_queued_vehicles
                                    ),
                                    egui::FontId::monospace(10.5),
                                    Color32::WHITE,
                                );
                            }

                            ui.add_space(4.0);
                            ui.small("Per-Demand & Summary Scorecard:");
                            egui::Grid::new("gui_batch_scorecard_grid")
                                .striped(true)
                                .spacing([8.0, 4.0])
                                .show(ui, |ui| {
                                    ui.strong("Scale");
                                    ui.strong("Regime");
                                    ui.strong("Mph");
                                    ui.strong("AvgQ");
                                    ui.strong("CO2");
                                    ui.strong("ΔSpd");
                                    ui.end_row();

                                    for row in rep
                                        .scenario_rows
                                        .iter()
                                        .chain(rep.regime_summary_rows.iter())
                                    {
                                        let is_sum = row.row_type == "REGIME_MEAN_SUMMARY";
                                        let scale_txt = if is_sum {
                                            "MEAN"
                                        } else {
                                            row.demand_label.as_str()
                                        };
                                        let short_reg = match row.regime {
                                            BatchControlRegime::FixedTime => "Fixed",
                                            BatchControlRegime::MaxPressureAdaptive => "MaxPres",
                                            BatchControlRegime::ModernRoundabout => "Roundabt",
                                        };
                                        ui.monospace(scale_txt);
                                        ui.monospace(short_reg);
                                        ui.monospace(format!(
                                            "{:.1}{}",
                                            row.mean_speed_mph, row.hcm_los
                                        ));
                                        ui.monospace(format!("{:.1}", row.avg_queued_vehicles));
                                        ui.monospace(format!("{:.0}", row.vsp_co2_kg_per_hr));
                                        let d_col = if row.speed_delta_vs_fixed_pct > 0.5 {
                                            Color32::from_rgb(52, 211, 153)
                                        } else if row.speed_delta_vs_fixed_pct < -0.5 {
                                            Color32::from_rgb(244, 63, 94)
                                        } else {
                                            Color32::GRAY
                                        };
                                        ui.colored_label(
                                            d_col,
                                            format!("{:+.1}%", row.speed_delta_vs_fixed_pct),
                                        );
                                        ui.end_row();
                                    }
                                });

                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("CSV:");
                                ui.text_edit_singleline(&mut self.sweep_csv_path);
                            });
                            if ui.button("Export Scorecard .csv").clicked() {
                                match rep.export_csv_path(Path::new(&self.sweep_csv_path)) {
                                    Ok(n_rows) => {
                                        self.io_status_msg = format!(
                                            "Exported {n_rows} scorecard rows -> {}",
                                            self.sweep_csv_path
                                        );
                                    }
                                    Err(e) => {
                                        self.io_status_msg = format!("CSV export failed: {e}");
                                    }
                                }
                            }
                        }
                    }
                    DockTab::ResilienceStress => {
                        ui.horizontal_wrapped(|ui| {
                            ui.small("Domain:");
                            ui.selectable_value(
                                &mut self.resilience_cat_filter,
                                ResilienceCategoryFilter::All17,
                                "All (17)",
                            );
                            ui.selectable_value(
                                &mut self.resilience_cat_filter,
                                ResilienceCategoryFilter::HazardAndMedical,
                                "Hazard & EMS",
                            );
                            ui.selectable_value(
                                &mut self.resilience_cat_filter,
                                ResilienceCategoryFilter::EvacAndSustainment,
                                "Evac & Shelters",
                            );
                            ui.selectable_value(
                                &mut self.resilience_cat_filter,
                                ResilienceCategoryFilter::FemaOperations,
                                "FEMA Suite",
                            );
                            ui.selectable_value(
                                &mut self.resilience_cat_filter,
                                ResilienceCategoryFilter::WildfireAndWui,
                                "Wildfire & WUI",
                            );
                        });
                        ui.horizontal_wrapped(|ui| {
                            let all_modes = [
                                (StressSubMode::SeismicMonteCarlo, "Seismic MC"),
                                (StressSubMode::FireFollowingQuake, "FFE Spread"),
                                (StressSubMode::HospitalTraumaSurge, "ED Surge"),
                                (StressSubMode::AeromedicalMedevac, "Medevac"),
                                (StressSubMode::DeterministicCutSets, "N-1/N-2"),
                                (StressSubMode::EvacuationContraflow, "Evac+CF"),
                                (StressSubMode::TransitBusEvacuation, "Bus+EMS Evac"),
                                (StressSubMode::EmergencyShelters, "Shelters+HEPA"),
                                (StressSubMode::CommodityResupply, "Supply VRP"),
                                (StressSubMode::WarningMobilization, "WEA & Sirens"),
                                (StressSubMode::FemaCommsArchitecture, "FEMA IPAWS/MERS"),
                                (StressSubMode::FemaIsbPods, "ISB & PODs"),
                                (StressSubMode::FemaUsarDebris, "US&R & Debris"),
                                (StressSubMode::FemaGsbDrcTdms, "GSB/TDMS/DRC"),
                                (StressSubMode::WildfireAlertWestResponse, "ALERTWest Fire"),
                                (StressSubMode::BurnScarDebrisFlow, "Burn-Scar & Culverts"),
                                (StressSubMode::WtpTurbidityStorage, "WTP Ash/Turbidity"),
                            ];
                            for (sm, lbl) in all_modes {
                                if self.resilience_cat_filter.contains(sm)
                                    || self.stress_sub_mode == sm
                                {
                                    ui.selectable_value(&mut self.stress_sub_mode, sm, lbl);
                                }
                            }
                        });
                        ui.separator();

                        match self.stress_sub_mode {
                        StressSubMode::DeterministicCutSets => {
                        ui.label(
                            egui::RichText::new(
                                "Arterial Bridge & Corridor Closure Stress-Test (N-1 / N-2)",
                            )
                            .strong(),
                        );
                        ui.small(
                            "Evaluates how current MCLP Fire Station placements hold up under single-corridor (N-1) & compound 2-bridge/corridor (N-2) closures (Church et al. 2004; Scott et al. 2006 NRI).",
                        );
                        ui.separator();

                        egui::ComboBox::from_id_salt("stress_depth_cb")
                            .selected_text(self.stress_cfg.interdiction_depth.label())
                            .show_ui(ui, |ui| {
                                for d in [
                                    DisruptionInterdictionDepth::SingleLinkNMinus1,
                                    DisruptionInterdictionDepth::DualLinkNMinus2,
                                ] {
                                    ui.selectable_value(
                                        &mut self.stress_cfg.interdiction_depth,
                                        d,
                                        d.label(),
                                    );
                                }
                            });

                        ui.checkbox(
                            &mut self.stress_cfg.bidirectional_closure,
                            "Bidirectional Bridge / Arterial Severance",
                        );
                        ui.checkbox(
                            &mut self.stress_cfg.arterials_and_bridges_only,
                            "Focus on Bridges, Trunks & Multi-Lane Arterials",
                        );
                        ui.checkbox(
                            &mut self.stress_cfg.protect_station_host_blocks,
                            "Keep Fire Station Host Apron Block Open",
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.stress_cfg.target_coverage_pct,
                                50.0..=100.0,
                            )
                            .text("Target Cov (%)"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.stress_cfg.threshold_travel_s,
                                90.0..=360.0,
                            )
                            .step_by(15.0)
                            .text("Max Travel (s)"),
                        );

                        ui.horizontal_wrapped(|ui| {
                            if self.world.emergency_coverage.fire_station_count == 0
                                && ui.button("1. Auto-Site MCLP >=90%").clicked()
                            {
                                let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                self.world.apply_mclp_siting_solution(&mrep);
                                self.latest_mclp_report = Some(mrep);
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                            if ui
                                .button(
                                    egui::RichText::new("Run Closure Stress-Test (Rayon)")
                                        .strong()
                                        .color(Color32::from_rgb(251, 146, 60)),
                                )
                                .clicked()
                            {
                                let srep = self.world.run_resilience_stress_test(&self.stress_cfg);
                                self.io_status_msg = format!(
                                    "Evaluated {} closures in {:.1} ms -> worst {:.1}% cov (-{:.1} pts)",
                                    srep.evaluated_scenarios,
                                    srep.solve_time_ms,
                                    srep.worst_case_parcel_cov_pct,
                                    srep.max_coverage_drop_pct_pts
                                );
                                self.latest_stress_report = Some(srep);
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                            if ui
                                .button(
                                    egui::RichText::new("Harden MCLP Siting (Ctrl+Z)")
                                        .strong()
                                        .color(Color32::from_rgb(52, 211, 153)),
                                )
                                .on_hover_text(
                                    "Re-solve MCLP across the ensemble of worst-case bridge/corridor disruptions (Church & Scaparra fortification)",
                                )
                                .clicked()
                            {
                                let mut r_mclp = self.mclp_cfg.clone();
                                r_mclp.target_coverage_pct = self.stress_cfg.target_coverage_pct;
                                r_mclp.threshold_travel_s = self.stress_cfg.threshold_travel_s;
                                let hardened = self
                                    .world
                                    .solve_disruption_hardened_mclp(&r_mclp, &self.stress_cfg);
                                let total_fs = self.world.apply_mclp_siting_solution(&hardened);
                                let post_stress =
                                    self.world.run_resilience_stress_test(&self.stress_cfg);
                                self.io_status_msg = format!(
                                    "Hardened MCLP ({total_fs} stations): worst-case post-closure cov {:.1}% | pass rate {:.0}%",
                                    post_stress.worst_case_parcel_cov_pct,
                                    post_stress.resilience_pass_rate_pct
                                );
                                self.latest_mclp_report = Some(hardened);
                                self.latest_stress_report = Some(post_stress);
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                        });

                        let active_closed = self.world.closed_segment_count();
                        if active_closed > 0 {
                            ui.horizontal(|ui| {
                                ui.colored_label(
                                    Color32::from_rgb(248, 113, 113),
                                    format!("Live Active Closures: {active_closed} segments"),
                                );
                                if ui.button("Reopen All (Ctrl+Z)").clicked() {
                                    self.world.clear_all_road_closures_tracked();
                                    self.heatmap_dirty = true;
                                }
                            });
                        }

                        if let Some(srep) = self.latest_stress_report.clone() {
                            ui.separator();
                            let pass_col = if srep.resilience_pass_rate_pct >= 90.0 {
                                Color32::from_rgb(52, 211, 153)
                            } else if srep.resilience_pass_rate_pct >= 60.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            ui.colored_label(
                                pass_col,
                                format!(
                                    "Resilience Pass Rate: {:.1}% ({}/{} scenarios >= {:.0}% cov | {:.1} ms)",
                                    srep.resilience_pass_rate_pct,
                                    srep.scenarios_retaining_target,
                                    srep.evaluated_scenarios,
                                    srep.config.target_coverage_pct,
                                    srep.solve_time_ms
                                ),
                            );
                            ui.monospace(format!(
                                "Intact: {:.1}% pcl / {:.1}% pop (P90 {:.0}s)
Worst : {:.1}% pcl (-{:.1} pts) | P90 +{:.0}s | Isolated: {} pcl",
                                srep.baseline_parcel_cov_pct,
                                srep.baseline_resident_cov_pct,
                                srep.baseline_p90_travel_s,
                                srep.worst_case_parcel_cov_pct,
                                srep.max_coverage_drop_pct_pts,
                                srep.max_p90_surge_s,
                                srep.max_isolated_parcels
                            ));

                            // Ranked Horizontal Vulnerability Bar Chart (`ui.painter_at`)
                            let n_bars = srep.top_vulnerabilities.len().min(6);
                            if n_bars > 0 {
                                let chart_h = (n_bars as f32) * 18.0 + 10.0;
                                let (c_rect, _) = ui.allocate_exact_size(
                                    EguiVec2::new(365.0, chart_h),
                                    Sense::hover(),
                                );
                                let cp = ui.painter_at(c_rect);
                                cp.rect_filled(c_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                let max_drop = srep
                                    .max_coverage_drop_pct_pts
                                    .max(srep.max_nri_delay_parcel_s * 0.01)
                                    .max(1.0) as f32;
                                for (idx, v) in srep.top_vulnerabilities.iter().take(n_bars).enumerate() {
                                    let y_top = c_rect.top() + 5.0 + (idx as f32) * 18.0;
                                    let score = (v.coverage_drop_pct_pts
                                        .max(v.nri_system_delay_parcel_s * 0.01))
                                        as f32;
                                    let frac = (score / max_drop).clamp(0.04, 1.0);
                                    let bar_w = frac * 175.0;
                                    let bar_col = if !v.retains_nfpa_target {
                                        Color32::from_rgb(244, 63, 94)
                                    } else {
                                        Color32::from_rgb(251, 146, 60)
                                    };
                                    cp.rect_filled(
                                        Rect::from_min_size(
                                            Pos2::new(c_rect.left() + 180.0, y_top + 2.0),
                                            EguiVec2::new(bar_w, 12.0),
                                        ),
                                        2.0,
                                        bar_col,
                                    );
                                    let short_lbl: String = v.label.chars().take(24).collect();
                                    cp.text(
                                        Pos2::new(c_rect.left() + 6.0, y_top + 8.0),
                                        egui::Align2::LEFT_CENTER,
                                        format!("#{} {}", v.rank, short_lbl),
                                        egui::FontId::monospace(9.5),
                                        Color32::LIGHT_GRAY,
                                    );
                                    cp.text(
                                        Pos2::new(c_rect.right() - 5.0, y_top + 8.0),
                                        egui::Align2::RIGHT_CENTER,
                                        format!("-{:.1}pt", v.coverage_drop_pct_pts),
                                        egui::FontId::monospace(9.5),
                                        Color32::WHITE,
                                    );
                                }
                            }

                            ui.separator();
                            ui.label("Top Vulnerable Corridors & Cut-Sets (Click to Inject Live):");
                            egui::Grid::new("stress_vuln_grid")
                                .striped(true)
                                .spacing([6.0, 4.0])
                                .show(ui, |ui| {
                                    ui.strong("#");
                                    ui.strong("Corridor / Cut-Set");
                                    ui.strong("Post%");
                                    ui.strong("NRI");
                                    ui.strong("Act");
                                    ui.end_row();

                                    for v in srep.top_vulnerabilities.iter().take(10) {
                                        ui.monospace(format!(
                                            "{} (N-{})",
                                            v.rank, v.interdiction_order
                                        ));
                                        let trunc: String = v.label.chars().take(22).collect();
                                        if ui
                                            .selectable_label(false, trunc)
                                            .on_hover_text(format!(
                                                "{}
Segments: {:?}
Post-Closure Cov: {:.1}% (-{:.1} pts)
P90: {:.0}s (+{:.0}s) | Isolated: {} parcels
Click name to pan camera",
                                                v.label,
                                                v.closed_segment_ids,
                                                v.post_parcel_cov_pct,
                                                v.coverage_drop_pct_pts,
                                                v.post_p90_travel_s,
                                                v.p90_surge_s,
                                                v.isolated_parcels
                                            ))
                                            .clicked()
                                        {
                                            self.cam_x = v.midpoint.x;
                                            self.cam_y = v.midpoint.y;
                                        }
                                        let c_col = if v.retains_nfpa_target {
                                            Color32::from_rgb(52, 211, 153)
                                        } else {
                                            Color32::from_rgb(244, 63, 94)
                                        };
                                        ui.colored_label(
                                            c_col,
                                            format!("{:.1}%", v.post_parcel_cov_pct),
                                        );
                                        ui.monospace(format!("{:.0}", v.nri_system_delay_parcel_s));
                                        if ui
                                            .small_button("Close")
                                            .on_hover_text(
                                                "Inject this road/bridge closure into the live simulation (Ctrl+Z to undo)",
                                            )
                                            .clicked()
                                        {
                                            self.world.clear_all_road_closures_tracked();
                                            let n_c = self.world.set_segments_closed_tracked(
                                                &v.closed_segment_ids,
                                                true,
                                                format!("Stress-Test Closure #{}: {}", v.rank, v.label),
                                            );
                                            self.cam_x = v.midpoint.x;
                                            self.cam_y = v.midpoint.y;
                                            self.layer = HeatmapLayer::EmergencyResponse4Min;
                                            self.heatmap_dirty = true;
                                            self.io_status_msg = format!(
                                                "Injected closure ({n_c} segs): {} -> live <=4m cov {:.1}%",
                                                v.label,
                                                self.world.emergency_coverage.parcels_within_4min_pct
                                            );
                                        }
                                        ui.end_row();
                                    }
                                });

                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("CSV:");
                                ui.text_edit_singleline(&mut self.stress_csv_path);
                            });
                            if ui.button("Export Stress Scorecard .csv").clicked() {
                                match srep.export_csv_path(Path::new(&self.stress_csv_path)) {
                                    Ok(n_rows) => {
                                        self.io_status_msg = format!(
                                            "Exported {n_rows} vulnerability rows -> {}",
                                            self.stress_csv_path
                                        );
                                    }
                                    Err(e) => {
                                        self.io_status_msg = format!("CSV export failed: {e}");
                                    }
                                }
                            }
                        }
                        }
                        StressSubMode::SeismicMonteCarlo => {
                            ui.label(
                                egui::RichText::new(
                                    "Probabilistic Seismic Bridge Fragility & Monte Carlo (Hazus / Jayaram-Baker)",
                                )
                                .strong(),
                            );
                            ui.small(
                                "Samples spatially correlated Sa(1.0s) ground-motion fields & lognormal bridge fragility curves (Slight/Moderate/Extensive/Complete) across N parallel Rayon realizations.",
                            );
                            ui.horizontal_wrapped(|ui| {
                                ui.small("Presets:");
                                if ui.small_button("M6.7 Crustal").clicked() {
                                    self.seismic_cfg.moment_magnitude_mw = 6.7;
                                    self.seismic_cfg.rupture_standoff_km = 10.0;
                                    self.seismic_cfg.vs30_m_per_s = 280.0;
                                }
                                if ui.small_button("M7.3 Valley Fault").clicked() {
                                    self.seismic_cfg.moment_magnitude_mw = 7.3;
                                    self.seismic_cfg.rupture_standoff_km = 14.0;
                                    self.seismic_cfg.vs30_m_per_s = 250.0;
                                }
                                if ui.small_button("M9.0 Cascadia CSZ").clicked() {
                                    self.seismic_cfg.moment_magnitude_mw = 9.0;
                                    self.seismic_cfg.rupture_standoff_km = 68.0;
                                    self.seismic_cfg.vs30_m_per_s = 240.0;
                                }
                            });
                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.moment_magnitude_mw, 5.5..=9.2)
                                    .step_by(0.1)
                                    .text("Magnitude (Mw)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.rupture_standoff_km, 2.0..=140.0)
                                    .text("Fault Standoff (km)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.vs30_m_per_s, 150.0..=800.0)
                                    .text("Site Vs30 (m/s)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.seismic_cfg.spatial_corr_range_km,
                                    8.0..=55.0,
                                )
                                .text("Corr Range b (km)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.monte_carlo_trials, 50..=2500)
                                    .step_by(50.0)
                                    .text("MC Trials (N)"),
                            );

                            ui.separator();
                            ui.label(
                                egui::RichText::new(
                                    "CRN Retrofit, Crew Speedup & Hazus S-Curves:",
                                )
                                .strong(),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.seismic_cfg.retrofit_top_k_bridges,
                                    0..=16,
                                )
                                .text("Retrofit Top-K Bridges"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.seismic_cfg.repair_crew_speedup,
                                    0.5..=3.0,
                                )
                                .step_by(0.1)
                                .text("Repair Crew Speed (x)"),
                            );
                            egui::ComboBox::from_id_salt("seismic_retro_arch_cb")
                                .selected_text(self.seismic_cfg.retrofit_target_archetype.short_label())
                                .show_ui(ui, |ui| {
                                    for a in [
                                        BridgeStructuralArchetype::ConventionalContinuousConcrete,
                                        BridgeStructuralArchetype::SeismicallyRetrofittedCollar,
                                        BridgeStructuralArchetype::EssentialBaseIsolatedLifeline,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.seismic_cfg.retrofit_target_archetype,
                                            a,
                                            a.short_label(),
                                        );
                                    }
                                });

                            ui.separator();
                            ui.label(
                                egui::RichText::new(
                                    "ALA (2001) / Hazus PGV Buried Water & Sewer Pipes:",
                                )
                                .strong(),
                            );
                            egui::ComboBox::from_id_salt("pipe_law_cb")
                                .selected_text(self.seismic_cfg.pipe_fragility_law.label())
                                .show_ui(ui, |ui| {
                                    for pl in [
                                        PipelineFragilityLaw::Ala2001LinearPgv,
                                        PipelineFragilityLaw::HazusORourkeAyala1993,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.seismic_cfg.pipe_fragility_law,
                                            pl,
                                            pl.label(),
                                        );
                                    }
                                });
                            egui::ComboBox::from_id_salt("pipe_mat_cb")
                                .selected_text(format!(
                                    "Base Pipe: {}",
                                    self.seismic_cfg.baseline_water_pipe_material.short_label()
                                ))
                                .show_ui(ui, |ui| {
                                    for pm in [
                                        BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
                                        BuriedPipeMaterialClass::SegmentedDuctileIronPvc,
                                        BuriedPipeMaterialClass::WeldedSteelRestrainedJoint,
                                        BuriedPipeMaterialClass::SeismicErdipHdpe,
                                    ] {
                                        if ui
                                            .selectable_value(
                                                &mut self.seismic_cfg.baseline_water_pipe_material,
                                                pm,
                                                pm.short_label(),
                                            )
                                            .clicked()
                                        {
                                            self.seismic_cfg.baseline_sewer_pipe_material = pm;
                                        }
                                    }
                                });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.seismic_cfg.seismic_pipe_replacement_pct,
                                        0.0..=100.0,
                                    )
                                    .text("ERDIP Replace %"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.seismic_cfg.water_sewer_repair_crews,
                                        1..=20,
                                    )
                                    .text("Pipe Crews"),
                                );
                                ui.checkbox(
                                    &mut self.seismic_cfg.critical_lifeline_backup_power,
                                    "Backup Gen",
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                ui.selectable_value(
                                    &mut self.seismic_curve_view,
                                    SeismicCurveView::HazusRestoration90d,
                                    "Hazus 0-90d",
                                );
                                ui.selectable_value(
                                    &mut self.seismic_curve_view,
                                    SeismicCurveView::FragilitySa10,
                                    "Bridge P(DS|Sa)",
                                );
                                ui.selectable_value(
                                    &mut self.seismic_curve_view,
                                    SeismicCurveView::AlaPgvPipeFragility,
                                    "ALA Pipe RR(PGV)",
                                );
                            });

                            match self.seismic_curve_view {
                                SeismicCurveView::FragilitySa10 => {
                                    egui::ComboBox::from_id_salt("frag_plot_arch_cb")
                                        .selected_text(format!(
                                            "Curve: {}",
                                            self.fragility_plot_arch.short_label()
                                        ))
                                        .show_ui(ui, |ui| {
                                            for a in [
                                                BridgeStructuralArchetype::Pre1975MultiSpanSimplySupported,
                                                BridgeStructuralArchetype::ConventionalContinuousConcrete,
                                                BridgeStructuralArchetype::SeismicallyRetrofittedCollar,
                                                BridgeStructuralArchetype::EssentialBaseIsolatedLifeline,
                                            ] {
                                                ui.selectable_value(
                                                    &mut self.fragility_plot_arch,
                                                    a,
                                                    a.short_label(),
                                                );
                                            }
                                        });

                                    let (f_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 96.0),
                                        Sense::hover(),
                                    );
                                    let fp = ui.painter_at(f_rect);
                                    fp.rect_filled(f_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let max_im = 2.0f64;
                                    let curve_pts =
                                        self.fragility_plot_arch.sample_curve_points(max_im, 40);
                                    let ds_colors = [
                                        Color32::from_rgb(52, 211, 153),
                                        Color32::from_rgb(250, 204, 21),
                                        Color32::from_rgb(251, 146, 60),
                                        Color32::from_rgb(244, 63, 94),
                                    ];
                                    for ds_i in 0..4 {
                                        let mut poly = Vec::with_capacity(curve_pts.len());
                                        for &(im, probs) in &curve_pts {
                                            let x = f_rect.left()
                                                + 10.0
                                                + ((im / max_im) as f32) * (f_rect.width() - 20.0);
                                            let y = f_rect.bottom()
                                                - 14.0
                                                - (probs[ds_i] as f32) * (f_rect.height() - 22.0);
                                            poly.push(Pos2::new(x, y));
                                        }
                                        fp.add(egui::Shape::line(
                                            poly,
                                            Stroke::new(1.8, ds_colors[ds_i]),
                                        ));
                                    }
                                    if let Some(mcrep) = self.latest_seismic_report.as_ref() {
                                        let med_x = f_rect.left()
                                            + 10.0
                                            + (((mcrep.mean_median_sa10_g / max_im).clamp(0.0, 1.0))
                                                as f32)
                                                * (f_rect.width() - 20.0);
                                        fp.line_segment(
                                            [
                                                Pos2::new(med_x, f_rect.top() + 6.0),
                                                Pos2::new(med_x, f_rect.bottom() - 14.0),
                                            ],
                                            Stroke::new(1.3, Color32::from_rgb(125, 211, 252)),
                                        );
                                        fp.text(
                                            Pos2::new(
                                                (med_x + 3.0).min(f_rect.right() - 62.0),
                                                f_rect.top() + 6.0,
                                            ),
                                            egui::Align2::LEFT_TOP,
                                            format!("Sa={:.2}g", mcrep.mean_median_sa10_g),
                                            egui::FontId::monospace(9.5),
                                            Color32::from_rgb(125, 211, 252),
                                        );
                                    }
                                    fp.text(
                                        Pos2::new(f_rect.left() + 8.0, f_rect.bottom() - 2.0),
                                        egui::Align2::LEFT_BOTTOM,
                                        "0.0g   Slight(Grn) Mod(Yel) Ext/Close(Org) Comp(Red)   2.0g",
                                        egui::FontId::monospace(8.8),
                                        Color32::GRAY,
                                    );
                                }
                                SeismicCurveView::HazusRestoration90d => {
                                    let (r_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 98.0),
                                        Sense::hover(),
                                    );
                                    let rp = ui.painter_at(r_rect);
                                    rp.rect_filled(r_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let max_d = 90.0f64;
                                    let rest_pts = sample_hazus_restoration_curves(
                                        max_d,
                                        self.seismic_cfg.repair_crew_speedup,
                                        46,
                                    );
                                    for &(bd, lbl) in &[
                                        (1.0, "1d"),
                                        (7.0, "7d"),
                                        (30.0, "30d"),
                                        (90.0, "90d"),
                                    ] {
                                        let bx = r_rect.left()
                                            + 12.0
                                            + ((bd / max_d) as f32) * (r_rect.width() - 24.0);
                                        rp.line_segment(
                                            [
                                                Pos2::new(bx, r_rect.top() + 6.0),
                                                Pos2::new(bx, r_rect.bottom() - 14.0),
                                            ],
                                            Stroke::new(
                                                1.0,
                                                Color32::from_rgba_unmultiplied(148, 163, 184, 70),
                                            ),
                                        );
                                        rp.text(
                                            Pos2::new(bx, r_rect.bottom() - 1.0),
                                            egui::Align2::CENTER_BOTTOM,
                                            lbl,
                                            egui::FontId::monospace(8.8),
                                            Color32::LIGHT_GRAY,
                                        );
                                    }
                                    let ds_colors = [
                                        Color32::from_rgb(52, 211, 153),
                                        Color32::from_rgb(250, 204, 21),
                                        Color32::from_rgb(251, 146, 60),
                                        Color32::from_rgb(244, 63, 94),
                                    ];
                                    for ds_i in 0..4 {
                                        let mut poly = Vec::with_capacity(rest_pts.len());
                                        for &(d, funcs) in &rest_pts {
                                            let x = r_rect.left()
                                                + 12.0
                                                + ((d / max_d) as f32) * (r_rect.width() - 24.0);
                                            let y = r_rect.bottom()
                                                - 15.0
                                                - (funcs[ds_i] as f32) * (r_rect.height() - 24.0);
                                            poly.push(Pos2::new(x, y));
                                        }
                                        rp.add(egui::Shape::line(
                                            poly,
                                            Stroke::new(1.85, ds_colors[ds_i]),
                                        ));
                                    }
                                    rp.text(
                                        Pos2::new(r_rect.left() + 8.0, r_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "Hazus Bridge Functionality Q(t): Sl(Grn) Mod(Yel) Ext(Org) Comp(Red)",
                                        egui::FontId::monospace(8.4),
                                        Color32::GRAY,
                                    );
                                }
                                SeismicCurveView::AlaPgvPipeFragility => {
                                    let (p_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 98.0),
                                        Sense::hover(),
                                    );
                                    let pp = ui.painter_at(p_rect);
                                    pp.rect_filled(p_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let max_pgv = 110.0f64;
                                    let rr_pts = sample_ala_hazus_pgv_curves(
                                        self.seismic_cfg.pipe_fragility_law,
                                        max_pgv,
                                        41,
                                    );
                                    let max_rr = rr_pts
                                        .last()
                                        .map(|x| x.1[0])
                                        .unwrap_or(1.0)
                                        .max(0.05);
                                    let mat_cols = [
                                        Color32::from_rgb(244, 63, 94),  // Brittle CI / VCP
                                        Color32::from_rgb(251, 146, 60), // Ductile Iron / PVC
                                        Color32::from_rgb(56, 189, 248), // Welded Steel
                                        Color32::from_rgb(52, 211, 153), // Seismic ERDIP / HDPE
                                    ];
                                    for m_i in 0..4 {
                                        let mut poly = Vec::with_capacity(rr_pts.len());
                                        for &(pgv, rrs) in &rr_pts {
                                            let x = p_rect.left()
                                                + 12.0
                                                + ((pgv / max_pgv) as f32) * (p_rect.width() - 24.0);
                                            let y = p_rect.bottom()
                                                - 15.0
                                                - ((rrs[m_i] / max_rr).clamp(0.0, 1.0) as f32)
                                                    * (p_rect.height() - 25.0);
                                            poly.push(Pos2::new(x, y));
                                        }
                                        pp.add(egui::Shape::line(
                                            poly,
                                            Stroke::new(1.85, mat_cols[m_i]),
                                        ));
                                    }
                                    if let Some(mcrep) = self.latest_seismic_report.as_ref() {
                                        let px = p_rect.left()
                                            + 12.0
                                            + (((mcrep.median_corridor_pgv_cm_s / max_pgv)
                                                .clamp(0.0, 1.0))
                                                as f32)
                                                * (p_rect.width() - 24.0);
                                        pp.line_segment(
                                            [
                                                Pos2::new(px, p_rect.top() + 8.0),
                                                Pos2::new(px, p_rect.bottom() - 15.0),
                                            ],
                                            Stroke::new(1.3, Color32::from_rgb(250, 204, 21)),
                                        );
                                        pp.text(
                                            Pos2::new(
                                                (px + 3.0).min(p_rect.right() - 78.0),
                                                p_rect.top() + 14.0,
                                            ),
                                            egui::Align2::LEFT_TOP,
                                            format!("PGV={:.0}cm/s", mcrep.median_corridor_pgv_cm_s),
                                            egui::FontId::monospace(8.8),
                                            Color32::from_rgb(250, 204, 21),
                                        );
                                    }
                                    pp.text(
                                        Pos2::new(p_rect.left() + 8.0, p_rect.top() + 2.0),
                                        egui::Align2::LEFT_TOP,
                                        "Pipe RR/km vs PGV: CI(Red) DI(Org) Steel(Sky) ERDIP(Grn)",
                                        egui::FontId::monospace(8.3),
                                        Color32::GRAY,
                                    );
                                    pp.text(
                                        Pos2::new(p_rect.left() + 8.0, p_rect.bottom() - 1.0),
                                        egui::Align2::LEFT_BOTTOM,
                                        format!("0 cm/s -> 110 cm/s (Max RR={:.2}/km)", max_rr),
                                        egui::FontId::monospace(8.3),
                                        Color32::LIGHT_GRAY,
                                    );
                                }
                            }

                            ui.horizontal_wrapped(|ui| {
                                if self.world.emergency_coverage.fire_station_count == 0
                                    && ui.button("1. Auto-Site MCLP").clicked()
                                {
                                    let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                    self.world.apply_mclp_siting_solution(&mrep);
                                    self.latest_mclp_report = Some(mrep);
                                    self.layer = HeatmapLayer::EmergencyResponse4Min;
                                    self.heatmap_dirty = true;
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("Run Seismic Monte Carlo (Rayon)")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    let mcrep = self.world.run_seismic_monte_carlo(&self.seismic_cfg);
                                    self.fragility_plot_arch = mcrep.reference_archetype;
                                    self.io_status_msg = format!(
                                        "Simulated {} seismic trials ({:.1} ms): E[cov]={:.1}%, P10={:.1}%, E[closed]={:.1} bridges",
                                        mcrep.config.monte_carlo_trials,
                                        mcrep.solve_time_ms,
                                        mcrep.mean_parcel_cov_pct,
                                        mcrep.p10_parcel_cov_pct,
                                        mcrep.expected_closed_bridges
                                    );
                                    self.latest_seismic_report = Some(mcrep);
                                    self.layer = HeatmapLayer::EmergencyResponse4Min;
                                    self.heatmap_dirty = true;
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("Harden MCLP (Ctrl+Z)")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    let mut r_mclp = self.mclp_cfg.clone();
                                    r_mclp.target_coverage_pct = self.seismic_cfg.target_coverage_pct;
                                    r_mclp.threshold_travel_s = self.seismic_cfg.threshold_travel_s;
                                    let hardened = self
                                        .world
                                        .solve_disruption_hardened_mclp(&r_mclp, &self.stress_cfg);
                                    let n_fs = self.world.apply_mclp_siting_solution(&hardened);
                                    let mcrep = self.world.run_seismic_monte_carlo(&self.seismic_cfg);
                                    self.io_status_msg = format!(
                                        "Hardened MCLP ({n_fs} stations): seismic P10 cov {:.1}% | >=90% surv {:.1}%",
                                        mcrep.p10_parcel_cov_pct,
                                        mcrep.prob_meeting_nfpa_target_pct
                                    );
                                    self.latest_mclp_report = Some(hardened);
                                    self.latest_seismic_report = Some(mcrep);
                                    self.layer = HeatmapLayer::EmergencyResponse4Min;
                                    self.heatmap_dirty = true;
                                }
                            });

                            if let Some(mcrep) = self.latest_seismic_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Intact: {:.1}% cov (P90 {:.0}s) | Median Sa(1s): {:.2}g ({} bridges)
MC Cov: Mean {:.1}% | P50 {:.1}% | P10 {:.1}% | P05 {:.1}%
MC P90: Mean {:.0}s | P90-tail {:.0}s | P95-tail {:.0}s
Closed: E[B]={:.2} (P90={} br) | P(Cov>={:.0}%) = {:.1}%",
                                    mcrep.baseline_parcel_cov_pct,
                                    mcrep.baseline_p90_travel_s,
                                    mcrep.mean_median_sa10_g,
                                    mcrep.bridge_count,
                                    mcrep.mean_parcel_cov_pct,
                                    mcrep.p50_parcel_cov_pct,
                                    mcrep.p10_parcel_cov_pct,
                                    mcrep.p05_parcel_cov_pct,
                                    mcrep.mean_p90_travel_s,
                                    mcrep.p90_p90_travel_s,
                                    mcrep.p95_p90_travel_s,
                                    mcrep.expected_closed_bridges,
                                    mcrep.p90_closed_bridges,
                                    mcrep.config.target_coverage_pct,
                                    mcrep.prob_meeting_nfpa_target_pct
                                ));

                                // 10-Bin Monte Carlo Post-Event Coverage Histogram (`0-10%` .. `90-100%`)
                                let (h_rect, _) = ui.allocate_exact_size(
                                    EguiVec2::new(365.0, 74.0),
                                    Sense::hover(),
                                );
                                let hp = ui.painter_at(h_rect);
                                hp.rect_filled(h_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                let max_bin = (*mcrep.coverage_histogram_bins.iter().max().unwrap_or(&1)).max(1) as f32;
                                let bw = (h_rect.width() - 24.0) / 10.0;
                                for (b_i, &cnt) in mcrep.coverage_histogram_bins.iter().enumerate() {
                                    let bh = ((cnt as f32) / max_bin) * (h_rect.height() - 22.0);
                                    let x0 = h_rect.left() + 12.0 + (b_i as f32) * bw;
                                    let col = if b_i == 9 {
                                        Color32::from_rgb(52, 211, 153)
                                    } else if b_i >= 7 {
                                        Color32::from_rgb(250, 204, 21)
                                    } else {
                                        Color32::from_rgb(244, 63, 94)
                                    };
                                    hp.rect_filled(
                                        Rect::from_min_size(
                                            Pos2::new(x0 + 1.5, h_rect.bottom() - 13.0 - bh),
                                            EguiVec2::new((bw - 3.0).max(2.0), bh.max(1.0)),
                                        ),
                                        2.0,
                                        col,
                                    );
                                }
                                hp.text(
                                    Pos2::new(h_rect.left() + 8.0, h_rect.bottom() - 1.0),
                                    egui::Align2::LEFT_BOTTOM,
                                    "MC 4-Min Coverage Histogram: 0%  ->  50%  ->  90-100%",
                                    egui::FontId::monospace(9.0),
                                    Color32::GRAY,
                                );

                                // Multi-Week Emergency Coverage Recovery Curve (Days 0, 1, 3, 7, 14, 30, 60, 90)
                                ui.separator();
                                let rec_days_txt = mcrep
                                    .days_to_90pct_mean_recovery
                                    .map(|d| format!("{d:.1} days"))
                                    .unwrap_or_else(|| ">90 days".to_string());
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Multi-Week Coverage Recovery | Bruneau R90 = {:.1}% | >=90% in {}",
                                        mcrep.bruneau_resilience_index_90d_pct, rec_days_txt
                                    ))
                                    .strong()
                                    .color(Color32::from_rgb(125, 211, 252)),
                                );

                                if mcrep.recovery_timeline.len() >= 2 {
                                    let (rc_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 104.0),
                                        Sense::hover(),
                                    );
                                    let rcp = ui.painter_at(rc_rect);
                                    rcp.rect_filled(rc_rect, 4.0, Color32::from_rgb(10, 15, 28));

                                    let day_to_x = |d: f64| -> f32 {
                                        let u = (d.clamp(0.0, 90.0) / 90.0).sqrt() as f32;
                                        rc_rect.left() + 14.0 + u * (rc_rect.width() - 28.0)
                                    };
                                    let cov_to_y = |cov_pct: f64| -> f32 {
                                        rc_rect.bottom()
                                            - 15.0
                                            - ((cov_pct.clamp(0.0, 100.0) / 100.0) as f32)
                                                * (rc_rect.height() - 23.0)
                                    };

                                    let y_tgt = cov_to_y(mcrep.config.target_coverage_pct);
                                    rcp.line_segment(
                                        [
                                            Pos2::new(rc_rect.left() + 8.0, y_tgt),
                                            Pos2::new(rc_rect.right() - 8.0, y_tgt),
                                        ],
                                        Stroke::new(
                                            1.0,
                                            Color32::from_rgba_unmultiplied(250, 204, 21, 140),
                                        ),
                                    );

                                    let mut poly_p90 = Vec::with_capacity(mcrep.recovery_timeline.len());
                                    let mut poly_mean = Vec::with_capacity(mcrep.recovery_timeline.len());
                                    let mut poly_p10 = Vec::with_capacity(mcrep.recovery_timeline.len());

                                    for ep in &mcrep.recovery_timeline {
                                        let x = day_to_x(ep.day);
                                        poly_p90.push(Pos2::new(x, cov_to_y(ep.p90_parcel_cov_pct)));
                                        poly_mean.push(Pos2::new(x, cov_to_y(ep.mean_parcel_cov_pct)));
                                        poly_p10.push(Pos2::new(x, cov_to_y(ep.p10_parcel_cov_pct)));

                                        if ep.day == 0.0 || ep.is_hazus_primary_benchmark {
                                            rcp.line_segment(
                                                [
                                                    Pos2::new(x, rc_rect.top() + 6.0),
                                                    Pos2::new(x, rc_rect.bottom() - 15.0),
                                                ],
                                                Stroke::new(
                                                    1.0,
                                                    Color32::from_rgba_unmultiplied(148, 163, 184, 55),
                                                ),
                                            );
                                            let tick = if ep.day == 0.0 {
                                                "0d".to_string()
                                            } else {
                                                format!("{:.0}d", ep.day)
                                            };
                                            rcp.text(
                                                Pos2::new(x, rc_rect.bottom() - 1.0),
                                                egui::Align2::CENTER_BOTTOM,
                                                tick,
                                                egui::FontId::monospace(8.8),
                                                Color32::LIGHT_GRAY,
                                            );
                                        }
                                    }

                                    // Overlay 90-Day Multi-Lifeline Restoration Curves (Elec=Gold, Water=Cyan, Sewer=Violet)
                                    let mut poly_elec = Vec::with_capacity(mcrep.recovery_timeline.len());
                                    let mut poly_watr = Vec::with_capacity(mcrep.recovery_timeline.len());
                                    let mut poly_sewr = Vec::with_capacity(mcrep.recovery_timeline.len());
                                    for ep in &mcrep.recovery_timeline {
                                        let x = day_to_x(ep.day);
                                        poly_elec.push(Pos2::new(x, cov_to_y(ep.mean_electric_service_pct)));
                                        poly_watr.push(Pos2::new(x, cov_to_y(ep.mean_water_20psi_service_pct)));
                                        poly_sewr.push(Pos2::new(x, cov_to_y(ep.mean_wastewater_service_pct)));
                                    }
                                    rcp.add(egui::Shape::line(
                                        poly_elec,
                                        Stroke::new(1.4, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    rcp.add(egui::Shape::line(
                                        poly_watr,
                                        Stroke::new(1.6, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    rcp.add(egui::Shape::line(
                                        poly_sewr,
                                        Stroke::new(1.4, Color32::from_rgb(192, 132, 252)),
                                    ));
                                    rcp.add(egui::Shape::line(
                                        poly_p90,
                                        Stroke::new(1.3, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    rcp.add(egui::Shape::line(
                                        poly_mean,
                                        Stroke::new(2.3, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    rcp.add(egui::Shape::line(
                                        poly_p10,
                                        Stroke::new(1.6, Color32::from_rgb(244, 63, 94)),
                                    ));

                                    let sel_idx = self
                                        .selected_recovery_epoch_idx
                                        .min(mcrep.recovery_timeline.len() - 1);
                                    let sel_ep = &mcrep.recovery_timeline[sel_idx];
                                    let sx = day_to_x(sel_ep.day);
                                    rcp.circle_filled(
                                        Pos2::new(sx, cov_to_y(sel_ep.mean_parcel_cov_pct)),
                                        4.0,
                                        Color32::from_rgb(250, 204, 21),
                                    );
                                    rcp.text(
                                        Pos2::new(rc_rect.left() + 8.0, rc_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "P90(Grn)  Mean(Cyn)  P10-Lower(Red)  vs  sqrt(Day 0..90)",
                                        egui::FontId::monospace(8.4),
                                        Color32::GRAY,
                                    );

                                    ui.horizontal_wrapped(|ui| {
                                        for (idx, ep) in mcrep.recovery_timeline.iter().enumerate() {
                                            if ep.day == 0.0 || ep.is_hazus_primary_benchmark {
                                                let short_d = if ep.day == 0.0 {
                                                    "Day 0".to_string()
                                                } else {
                                                    format!("Day {:.0}", ep.day)
                                                };
                                                if ui
                                                    .selectable_label(
                                                        self.selected_recovery_epoch_idx == idx,
                                                        short_d,
                                                    )
                                                    .clicked()
                                                {
                                                    self.selected_recovery_epoch_idx = idx;
                                                }
                                            }
                                        }
                                        if ui
                                            .small_button("Inject Selected Day (Ctrl+Z)")
                                            .on_hover_text(
                                                "Applies the selected recovery day's P10 remaining bridge closures onto the live street network",
                                            )
                                            .clicked()
                                        {
                                            let ep = &mcrep.recovery_timeline[sel_idx];
                                            let n_c = self.world.inject_recovery_epoch_tracked(ep);
                                            self.layer = HeatmapLayer::EmergencyResponse4Min;
                                            self.heatmap_dirty = true;
                                            self.io_status_msg = format!(
                                                "Injected {} P10 state ({n_c} closed segs) -> live <=4m cov {:.1}%",
                                                ep.label,
                                                self.world.emergency_coverage.parcels_within_4min_pct
                                            );
                                        }
                                    });

                                    egui::Grid::new("hazus_recovery_epochs_grid")
                                        .striped(true)
                                        .spacing([6.0, 3.0])
                                        .show(ui, |ui| {
                                            ui.strong("Epoch");
                                            ui.strong("Mean%");
                                            ui.strong("P10%");
                                            ui.strong("P90(s)");
                                            ui.strong("Func%");
                                            ui.end_row();
                                            for ep in &mcrep.recovery_timeline {
                                                if ep.day == 0.0 || ep.is_hazus_primary_benchmark {
                                                    ui.monospace(&ep.label);
                                                    ui.monospace(format!(
                                                        "{:.1}%",
                                                        ep.mean_parcel_cov_pct
                                                    ));
                                                    let p10_c = if ep.p10_parcel_cov_pct >= 90.0 {
                                                        Color32::from_rgb(52, 211, 153)
                                                    } else if ep.p10_parcel_cov_pct >= 75.0 {
                                                        Color32::from_rgb(250, 204, 21)
                                                    } else {
                                                        Color32::from_rgb(244, 63, 94)
                                                    };
                                                    ui.colored_label(
                                                        p10_c,
                                                        format!("{:.1}%", ep.p10_parcel_cov_pct),
                                                    );
                                                    ui.monospace(format!(
                                                        "{:.0}s",
                                                        ep.mean_p90_travel_s
                                                    ));
                                                    ui.monospace(format!(
                                                        "{:.0}%",
                                                        ep.mean_bridge_functionality_pct
                                                    ));
                                                    ui.end_row();
                                                }
                                            }
                                        });
                                }

                                ui.horizontal_wrapped(|ui| {
                                    if ui
                                        .small_button("Inject P10 Quake Live (Ctrl+Z)")
                                        .on_hover_text("Closes the bridges damaged in the 10th-percentile seismic realization")
                                        .clicked()
                                    {
                                        let n_c = self.world.inject_seismic_realization_tracked(
                                            &mcrep.p10_realization,
                                            format!(
                                                "Seismic Mw{:.1} P10 Realization ({} bridges closed)",
                                                mcrep.config.moment_magnitude_mw,
                                                mcrep.p10_realization.closed_bridges_count
                                            ),
                                        );
                                        self.layer = HeatmapLayer::EmergencyResponse4Min;
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Injected P10 quake ({n_c} closed segs) -> live <=4m cov {:.1}%",
                                            self.world.emergency_coverage.parcels_within_4min_pct
                                        );
                                    }
                                    if ui
                                        .small_button("Inject Worst Realization")
                                        .clicked()
                                    {
                                        let n_c = self.world.inject_seismic_realization_tracked(
                                            &mcrep.worst_realization,
                                            format!(
                                                "Seismic Mw{:.1} Worst Realization ({} bridges closed)",
                                                mcrep.config.moment_magnitude_mw,
                                                mcrep.worst_realization.closed_bridges_count
                                            ),
                                        );
                                        self.layer = HeatmapLayer::EmergencyResponse4Min;
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Injected worst quake ({n_c} closed segs) -> live <=4m cov {:.1}%",
                                            self.world.emergency_coverage.parcels_within_4min_pct
                                        );
                                    }
                                    if self.world.closed_segment_count() > 0
                                        && ui.small_button("Reopen All").clicked()
                                    {
                                        self.world.clear_all_road_closures_tracked();
                                        self.heatmap_dirty = true;
                                    }
                                });

                                ui.separator();
                                ui.label("Bridge Fragility Inventory (Sorted by MC Closure Risk):");
                                egui::Grid::new("seismic_bridge_grid")
                                    .striped(true)
                                    .spacing([6.0, 4.0])
                                    .show(ui, |ui| {
                                        ui.strong("Bridge");
                                        ui.strong("Sa(1s)");
                                        ui.strong("P(Cls)");
                                        ui.strong("State");
                                        ui.end_row();

                                        for b in mcrep.bridge_inventory.iter().take(10) {
                                            let short_b: String = b.label.chars().take(18).collect();
                                            if ui
                                                .selectable_label(false, short_b)
                                                .on_hover_text(format!(
                                                    "{}\nClass: {}{}\nNone {:.0}% | Slight {:.0}% | Mod {:.0}% | Ext {:.0}% | Comp {:.0}%\nHazus Func: Day1 {:.0}% | Day7 {:.0}% | Day30 {:.0}% | Day90 {:.0}% (E[Rep]={:.1}d)\nClick to pan camera",
                                                    b.label,
                                                    b.effective_archetype.short_label(),
                                                    if b.is_retrofitted { " [RETROFITTED]" } else { "" },
                                                    b.mc_state_pcts[0],
                                                    b.mc_state_pcts[1],
                                                    b.mc_state_pcts[2],
                                                    b.mc_state_pcts[3],
                                                    b.mc_state_pcts[4],
                                                    b.expected_func_day1_pct,
                                                    b.expected_func_day7_pct,
                                                    b.expected_func_day30_pct,
                                                    b.expected_func_day90_pct,
                                                    b.expected_repair_days
                                                ))
                                                .clicked()
                                            {
                                                self.cam_x = b.midpoint.x;
                                                self.cam_y = b.midpoint.y;
                                                self.fragility_plot_arch = b.effective_archetype;
                                            }
                                            ui.monospace(format!("{:.2}g", b.median_site_sa10_g));
                                            let r_col = if b.mc_closure_rate_pct >= 35.0 {
                                                Color32::from_rgb(244, 63, 94)
                                            } else if b.mc_closure_rate_pct >= 15.0 {
                                                Color32::from_rgb(250, 204, 21)
                                            } else {
                                                Color32::from_rgb(52, 211, 153)
                                            };
                                            ui.colored_label(
                                                r_col,
                                                format!("{:.1}%", b.mc_closure_rate_pct),
                                            );
                                            ui.monospace(if b.is_retrofitted {
                                                "Retro"
                                            } else {
                                                "Base"
                                            });
                                            ui.end_row();
                                        }
                                    });

                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.label("CSV:");
                                    ui.text_edit_singleline(&mut self.seismic_csv_path);
                                });
                                if ui.button("Export Bridge Fragility .csv").clicked() {
                                    match mcrep.export_csv_path(Path::new(&self.seismic_csv_path)) {
                                        Ok(n_rows) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_rows} bridge fragility rows -> {}",
                                                self.seismic_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("CSV export failed: {e}");
                                        }
                                    }
                                }
                                ui.horizontal(|ui| {
                                    ui.label("Rec CSV:");
                                    ui.text_edit_singleline(&mut self.recovery_csv_path);
                                });
                                ui.monospace(format!(
                                    "Pipes ({:.1}km, PGV {:.0}cm/s): W-Brk={:.2} W-Lk={:.2} | S-Brk={:.2}
R90 Lifelines: Elec {:.1}% | Water>=20psi {:.1}% | Sewer {:.1}%",
                                    mcrep.total_water_pipe_km,
                                    mcrep.median_corridor_pgv_cm_s,
                                    mcrep.expected_day0_water_breaks,
                                    mcrep.expected_day0_water_leaks,
                                    mcrep.expected_day0_sewer_breaks,
                                    mcrep.electric_resilience_index_90d_pct,
                                    mcrep.water_resilience_index_90d_pct,
                                    mcrep.wastewater_resilience_index_90d_pct,
                                ));
                                if ui.button("Export 90-Day Recovery Curve .csv").clicked() {
                                    match mcrep.export_recovery_csv_path(Path::new(&self.recovery_csv_path)) {
                                        Ok(n_rows) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_rows} Hazus recovery epochs -> {}",
                                                self.recovery_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("Recovery CSV failed: {e}");
                                        }
                                    }
                                }
                                ui.horizontal(|ui| {
                                    ui.label("Pipe CSV:");
                                    ui.text_edit_singleline(&mut self.pipeline_csv_path);
                                });
                                if ui.button("Export ALA Buried Pipeline Fragility .csv").clicked() {
                                    match mcrep.export_pipeline_csv_path(Path::new(&self.pipeline_csv_path)) {
                                        Ok(n_rows) => {
                                            self.io_status_msg = format!(
                                                "Exported {n_rows} buried pipeline fragility rows -> {}",
                                                self.pipeline_csv_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("Pipeline CSV failed: {e}");
                                        }
                                    }
                                }
                            }
                        }
                        StressSubMode::FireFollowingQuake => {
                            ui.label(
                                "Scawthorn (2005) / Hazus PGA floor-area ignitions + Hamada (1951) elliptical wind spread coupled to <20 psi broken water hydrants.",
                            );
                            egui::ComboBox::from_id_salt("ffe_ig_cb")
                                .selected_text(self.ffe_cfg.ignition_model.label())
                                .show_ui(ui, |ui| {
                                    for im in [
                                        FfeIgnitionModel::Scawthorn2005QuadraticPga,
                                        FfeIgnitionModel::HazusClassic1999Pga,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.ffe_cfg.ignition_model,
                                            im,
                                            im.label(),
                                        );
                                    }
                                });

                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.moment_magnitude_mw, 5.5..=9.0)
                                    .text("Quake Mw"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.rupture_standoff_km, 2.0..=80.0)
                                    .text("Rupture (km)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.ffe_cfg.wind_speed_mps, 0.0..=25.0)
                                    .text("Wind U (m/s)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.ffe_cfg.wind_direction_deg, -180.0..=180.0)
                                    .text("Wind Dir (deg)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.ffe_cfg.conflagration_horizon_mins,
                                    30.0..=360.0,
                                )
                                .text("Horizon (min)"),
                            );
                            ui.separator();
                            ui.label(
                                egui::RichText::new("Hydrant & Mitigation Retrofits:").strong(),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.seismic_cfg.seismic_pipe_replacement_pct,
                                    0.0..=100.0,
                                )
                                .text("ERDIP Water %"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.ffe_cfg.auxiliary_cistern_coverage_pct,
                                    0.0..=100.0,
                                )
                                .text("Cisterns/AWSS %"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.ffe_cfg.seismic_gas_shutoff_pct,
                                    0.0..=100.0,
                                )
                                .text("Gas Shutoff %"),
                            );
                            ui.checkbox(
                                &mut self.seismic_cfg.critical_lifeline_backup_power,
                                "Booster Pump Backup Power",
                            );

                            if ui
                                .button(
                                    egui::RichText::new("Run Scawthorn / Hazus FFE Monte Carlo")
                                        .strong()
                                        .color(Color32::from_rgb(251, 146, 60)),
                                )
                                .clicked()
                            {
                                if self.world.emergency_coverage.fire_station_count == 0 {
                                    let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                    self.world.apply_mclp_siting_solution(&mrep);
                                    self.latest_mclp_report = Some(mrep);
                                }
                                self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                let wr = self.ffe_cfg.wind_direction_deg.to_radians();
                                self.plume_params.wind_x_mps =
                                    (self.ffe_cfg.wind_speed_mps * wr.cos()) as f32;
                                self.plume_params.wind_y_mps =
                                    (self.ffe_cfg.wind_speed_mps * wr.sin()) as f32;

                                let frep = self.world.run_ffe_monte_carlo(&self.ffe_cfg);
                                self.layer = HeatmapLayer::FireFollowingQuake;
                                self.heatmap_dirty = true;
                                self.io_status_msg = format!(
                                    "FFE MC ({:.1} ms): E[Ign]={:.2}, DryHyd={:.2}, E[Burn]={:.1} parcels (${:.1}M loss)",
                                    frep.solve_time_ms,
                                    frep.expected_ignitions,
                                    frep.expected_dry_hydrant_failures,
                                    frep.expected_burned_parcels,
                                    frep.expected_fire_loss_musd
                                );
                                self.latest_ffe_report = Some(frep);
                            }

                            if let Some(frep) = self.latest_ffe_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Engines: {} | Hydrants: {:.1} psi ({:.0}% <20psi dry)
Ignit  : E[N]={:.2} (P90={}) | Knockdowns={:.2}
Escaped: {:.2} (Dry-Hydrant Fails={:.2})
Burned : {:.2} parcels (P90={}) | {:.3} MMsqft
Loss   : E[$]={:.1}M (P90=${:.1}M) | Displ={:.0} res",
                                    frep.total_fire_engines,
                                    frep.mean_post_quake_hydrant_psi,
                                    frep.dry_hydrant_parcels_pct,
                                    frep.expected_ignitions,
                                    frep.p90_ignitions,
                                    frep.expected_knockdowns_suppressed,
                                    frep.expected_escaped_conflagrations,
                                    frep.expected_dry_hydrant_failures,
                                    frep.expected_burned_parcels,
                                    frep.p90_burned_parcels,
                                    frep.expected_burned_floor_area_mmsqft,
                                    frep.expected_fire_loss_musd,
                                    frep.p90_fire_loss_musd,
                                    frep.expected_displaced_residents,
                                ));

                                let (e_rect, _) = ui.allocate_exact_size(
                                    EguiVec2::new(365.0, 112.0),
                                    Sense::hover(),
                                );
                                let ep = ui.painter_at(e_rect);
                                ep.rect_filled(e_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                let center_p = Pos2::new(
                                    e_rect.left() + 105.0,
                                    e_rect.center().y + 4.0,
                                );
                                let max_r_m = frep
                                    .polar_spread_envelope_60m
                                    .iter()
                                    .map(|x| x.1)
                                    .fold(20.0, f64::max);
                                let scale_px = 42.0 / (max_r_m as f32);

                                let mut poly_w = Vec::with_capacity(38);
                                let mut poly_c = Vec::with_capacity(38);
                                for &(ang_d, rw, rc) in &frep.polar_spread_envelope_60m {
                                    let ar = ang_d.to_radians() as f32;
                                    poly_w.push(Pos2::new(
                                        center_p.x + (rw as f32) * scale_px * ar.cos(),
                                        center_p.y + (rw as f32) * scale_px * ar.sin(),
                                    ));
                                    poly_c.push(Pos2::new(
                                        center_p.x + (rc as f32) * scale_px * ar.cos(),
                                        center_p.y + (rc as f32) * scale_px * ar.sin(),
                                    ));
                                }
                                ep.add(egui::Shape::line(
                                    poly_w,
                                    Stroke::new(2.0, Color32::from_rgb(251, 146, 60)),
                                ));
                                ep.add(egui::Shape::line(
                                    poly_c,
                                    Stroke::new(1.6, Color32::from_rgb(56, 189, 248)),
                                ));
                                ep.circle_filled(center_p, 2.5, Color32::from_rgb(244, 63, 94));
                                let wr = frep.config.wind_direction_deg.to_radians() as f32;
                                ep.line_segment(
                                    [
                                        center_p,
                                        Pos2::new(
                                            center_p.x + 36.0 * wr.cos(),
                                            center_p.y + 36.0 * wr.sin(),
                                        ),
                                    ],
                                    Stroke::new(1.5, Color32::from_rgb(250, 204, 21)),
                                );
                                ep.text(
                                    Pos2::new(e_rect.left() + 185.0, e_rect.top() + 16.0),
                                    egui::Align2::LEFT_TOP,
                                    format!(
                                        "Hamada 60m Ellipse
Wind: {:.1} m/s
Max R3: {:.0} m
Org = Wood R3
Sky = Fire-Res CBD",
                                        frep.config.wind_speed_mps, max_r_m
                                    ),
                                    egui::FontId::monospace(8.8),
                                    Color32::LIGHT_GRAY,
                                );

                                ui.horizontal(|ui| {
                                    ui.label("FFE CSV:");
                                    ui.text_edit_singleline(&mut self.ffe_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match frep.export_csv_path(Path::new(&self.ffe_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} FFE parcel risk rows -> {}",
                                                    self.ffe_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::HospitalTraumaSurge => {
                            ui.label(
                                "FEMA P-58 / Hazus Hospital ED Surge + START/ABA Burn Priority Queueing & Dynamic MOCC Ambulance Diversion.",
                            );
                            egui::ComboBox::from_id_salt("surge_div_cb")
                                .selected_text(self.surge_cfg.diversion_policy.label())
                                .show_ui(ui, |ui| {
                                    for pol in [
                                        AmbulanceDiversionPolicy::ClosestHospitalNoDiversion,
                                        AmbulanceDiversionPolicy::ThresholdSaturationDiversion,
                                        AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.surge_cfg.diversion_policy,
                                            pol,
                                            pol.label(),
                                        );
                                    }
                                });

                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.moment_magnitude_mw, 5.5..=9.0)
                                    .text("Quake Mw"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.seismic_cfg.rupture_standoff_km, 2.0..=80.0)
                                    .text("Rupture (km)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.ffe_cfg.wind_speed_mps, 0.0..=25.0)
                                    .text("FFE Wind (m/s)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.surge_cfg.surge_window_hours, 2.0..=16.0)
                                    .text("Surge Win (h)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.surge_cfg.diversion_occupancy_threshold_pct,
                                    60.0..=140.0,
                                )
                                .text("Divert Sat %"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.surge_cfg.ambulances_per_station, 1..=8)
                                    .text("Amb / Station"),
                            );
                            ui.separator();
                            ui.label(
                                egui::RichText::new("Hospital Surge & Seismic Hardening:").strong(),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.surge_cfg.hospital_seismic_retrofit_pct,
                                    0.0..=100.0,
                                )
                                .text("SPC-5 Hosp %"),
                            );
                            ui.checkbox(
                                &mut self.surge_cfg.hospital_onsite_backup_lifelines,
                                "72h Hospital Power & Water Tanks",
                            );
                            ui.checkbox(
                                &mut self.surge_cfg.aba_burn_surge_expansion,
                                "ABA +50% Crisis Burn ICU Surge",
                            );
                            ui.checkbox(
                                &mut self.surge_cfg.deploy_mobile_field_hospitals,
                                "Deploy DMAT Mobile Field Hospital (MFH)",
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run Hospital ED Surge MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    let srep = self.world.run_hospital_surge_monte_carlo(&self.surge_cfg);
                                    self.layer = HeatmapLayer::HospitalTraumaSurge;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "ED Surge MC ({:.1} ms): Red/Burn Treat={:.1}m | GoldenHr={:.1}% | Divert={:.1}%",
                                        srep.solve_time_ms,
                                        srep.mean_red_burn_time_to_treatment_min,
                                        srep.golden_hour_treated_pct,
                                        srep.ambulance_diversion_rate_pct
                                    );
                                    self.latest_surge_report = Some(srep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Closest vs. MOCC")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    let mut c_cfg = self.surge_cfg.clone();
                                    c_cfg.diversion_policy =
                                        AmbulanceDiversionPolicy::ClosestHospitalNoDiversion;
                                    let c_rep = self.world.run_hospital_surge_monte_carlo(&c_cfg);

                                    let mut d_cfg = self.surge_cfg.clone();
                                    d_cfg.diversion_policy =
                                        AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage;
                                    let d_rep = self.world.run_hospital_surge_monte_carlo(&d_cfg);
                                    self.surge_cfg.diversion_policy =
                                        AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage;
                                    self.surge_ab_summary = Some(format!(
                                        "A/B CRN: Closest Wait={:.1}m (GH {:.1}%, Mort {:.2}) -> MOCC Wait={:.1}m (GH {:.1}%, Mort {:.2})",
                                        c_rep.mean_red_burn_ed_wait_min,
                                        c_rep.golden_hour_treated_pct,
                                        c_rep.expected_preventable_surge_deaths,
                                        d_rep.mean_red_burn_ed_wait_min,
                                        d_rep.golden_hour_treated_pct,
                                        d_rep.expected_preventable_surge_deaths
                                    ));
                                    self.latest_surge_report = Some(d_rep);
                                    self.layer = HeatmapLayer::HospitalTraumaSurge;
                                    self.heatmap_dirty = true;
                                }
                            });

                            if let Some(ab_txt) = &self.surge_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(srep) = self.latest_surge_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Fleet : {} amb | ED Func: {:.1}% | BurnCtr Match: {:.1}%\nCas   : Red={:.1} (P90={}) | Burn={:.1} (P90={}) | Yel={:.1} | Grn={:.1}\nRoute : AmbTrav={:.1}m | Diverted={:.1} ({:.1}%) | BurnXfer={:.2}\nQueue : Red/Burn Wait={:.1}m (P90={:.1}m) | Treat={:.1}m\nSurviv: Golden-Hr (<=60m)={:.1}% | Prev. Surge Deaths={:.2}",
                                    srep.total_ambulances,
                                    srep.mean_system_ed_functionality_pct,
                                    srep.burn_matched_to_burn_center_pct,
                                    srep.expected_red_trauma,
                                    srep.p90_red_trauma,
                                    srep.expected_burn_critical,
                                    srep.p90_burn_critical,
                                    srep.expected_yellow_delayed,
                                    srep.expected_green_walkin,
                                    srep.mean_ambulance_transport_min,
                                    srep.expected_diverted_ambulances,
                                    srep.ambulance_diversion_rate_pct,
                                    srep.expected_secondary_burn_transfers,
                                    srep.mean_red_burn_ed_wait_min,
                                    srep.p90_red_burn_ed_wait_min,
                                    srep.mean_red_burn_time_to_treatment_min,
                                    srep.golden_hour_treated_pct,
                                    srep.expected_preventable_surge_deaths
                                ));

                                // Multi-Hour ED Queue Backlog & Mean Wait Trajectory (`egui` plot)
                                if srep.surge_timeline.len() >= 2 {
                                    let (q_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 96.0),
                                        Sense::hover(),
                                    );
                                    let qp = ui.painter_at(q_rect);
                                    qp.rect_filled(q_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let max_y = srep
                                        .surge_timeline
                                        .iter()
                                        .map(|pt| pt.mean_ed_wait_min.max(pt.active_ed_queue_patients))
                                        .fold(65.0, f64::max) as f32;
                                    let n_pts = srep.surge_timeline.len();
                                    let mut poly_w = Vec::with_capacity(n_pts);
                                    let mut poly_q = Vec::with_capacity(n_pts);
                                    for (idx, pt) in srep.surge_timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = q_rect.left() + 10.0 + u * (q_rect.width() - 20.0);
                                        let yw = q_rect.bottom()
                                            - 14.0
                                            - ((pt.mean_ed_wait_min as f32) / max_y).clamp(0.0, 1.0)
                                                * (q_rect.height() - 22.0);
                                        let yq = q_rect.bottom()
                                            - 14.0
                                            - ((pt.active_ed_queue_patients as f32) / max_y)
                                                .clamp(0.0, 1.0)
                                                * (q_rect.height() - 22.0);
                                        poly_w.push(Pos2::new(x, yw));
                                        poly_q.push(Pos2::new(x, yq));
                                    }
                                    qp.add(egui::Shape::line(
                                        poly_q,
                                        Stroke::new(1.7, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    qp.add(egui::Shape::line(
                                        poly_w,
                                        Stroke::new(2.1, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    qp.text(
                                        Pos2::new(q_rect.left() + 8.0, q_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "ED Surge Trajectory: Red=ED Wait(min)  Sky=Queued Patients",
                                        egui::FontId::monospace(8.4),
                                        Color32::GRAY,
                                    );
                                }

                                for h in &srep.hospital_rows {
                                    ui.small(format!(
                                        "[{}] {}: {:.1}/{:.0} bays ({:.0}% pk) | Wait {:.1}m | Div {:.0}%",
                                        h.designation.badge_prefix(),
                                        h.name,
                                        h.mean_functional_ed_bays,
                                        h.baseline_ed_bays,
                                        h.peak_ed_utilization_pct,
                                        h.mean_red_burn_wait_min,
                                        h.time_on_diversion_pct
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Surge CSV:");
                                    ui.text_edit_singleline(&mut self.surge_hosp_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match srep.export_hospital_csv_path(Path::new(&self.surge_hosp_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} hospital ED surge rows -> {}",
                                                    self.surge_hosp_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::AeromedicalMedevac => {
                            ui.label(
                                "FAA 14 CFR 135.609 + Koschmieder FFE Smoke/Wind Flight Envelopes, Rooftop/Ground Helipads & Secondary WRBDC Burn Transfers.",
                            );
                            egui::ComboBox::from_id_salt("medevac_av_cb")
                                .selected_text(self.medevac_cfg.avionics_regime.label())
                                .show_ui(ui, |ui| {
                                    for av in [
                                        MedevacAvionicsRegime::DayVfrPart135,
                                        MedevacAvionicsRegime::NightUnassistedVfr,
                                        MedevacAvionicsRegime::NightNvisHtawsVfr,
                                        MedevacAvionicsRegime::WaasPinSIfr,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.medevac_cfg.avionics_regime,
                                            av,
                                            av.label(),
                                        );
                                    }
                                });
                            egui::ComboBox::from_id_salt("medevac_ac_cb")
                                .selected_text(self.medevac_cfg.airframe_class.label())
                                .show_ui(ui, |ui| {
                                    for ac in [
                                        RotorcraftAirframeClass::LightTwinHemsEc135,
                                        RotorcraftAirframeClass::MediumTwinIcuAw139,
                                        RotorcraftAirframeClass::HeavyNationalGuardUh60,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.medevac_cfg.airframe_class,
                                            ac,
                                            ac.label(),
                                        );
                                    }
                                });

                            ui.add(
                                egui::Slider::new(&mut self.medevac_cfg.rotorcraft_fleet_count, 1..=10)
                                    .text("HEMS Fleet"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.medevac_cfg.ambient_ceiling_ft,
                                    200.0..=5000.0,
                                )
                                .text("Ceiling (ft)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.medevac_cfg.ambient_visibility_sm,
                                    0.25..=10.0,
                                )
                                .text("Ambient Vis (SM)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.ffe_cfg.wind_speed_mps, 0.0..=26.0)
                                    .text("Wind / Smoke U (m/s)"),
                            );
                            ui.separator();
                            ui.checkbox(
                                &mut self.medevac_cfg.helipad_seismic_and_ground_lz_hardening,
                                "SPC-5 Surface Helipads + Park Emergency LZs",
                            );
                            ui.checkbox(
                                &mut self.medevac_cfg.wrbdc_regional_burn_mutual_aid,
                                "WRBDC Interstate Burn Air Bridge (Portland/Seattle)",
                            );
                            ui.checkbox(
                                &mut self.surge_cfg.aba_burn_surge_expansion,
                                "ABA +50% Local Burn ICU Surge Beds",
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run Rotary-Wing Medevac MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    let mrep = self.world.run_aeromedical_medevac_mc(&self.medevac_cfg);
                                    self.layer = HeatmapLayer::AeromedicalMedevac;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Medevac MC ({:.1} ms): EnvOpen={:.1}% | AirXfer={:.2} | XferTime={:.1}m",
                                        mrep.solve_time_ms,
                                        mrep.system_flight_envelope_open_pct,
                                        mrep.expected_air_secondary_burn_transfers,
                                        mrep.mean_secondary_burn_transfer_min
                                    );
                                    self.latest_medevac_report = Some(mrep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B VFR vs. PinS IFR + Ground LZ")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();

                                    let mut vfr_c = self.medevac_cfg.clone();
                                    vfr_c.avionics_regime = MedevacAvionicsRegime::DayVfrPart135;
                                    vfr_c.helipad_seismic_and_ground_lz_hardening = false;
                                    let v_rep = self.world.run_aeromedical_medevac_mc(&vfr_c);

                                    let mut ifr_c = self.medevac_cfg.clone();
                                    ifr_c.avionics_regime = MedevacAvionicsRegime::WaasPinSIfr;
                                    ifr_c.helipad_seismic_and_ground_lz_hardening = true;
                                    let i_rep = self.world.run_aeromedical_medevac_mc(&ifr_c);

                                    self.medevac_cfg.avionics_regime =
                                        MedevacAvionicsRegime::WaasPinSIfr;
                                    self.medevac_cfg.helipad_seismic_and_ground_lz_hardening = true;
                                    self.medevac_ab_summary = Some(format!(
                                        "A/B CRN: Day-VFR Env={:.0}% (Xfer {:.0}m, Scrub {:.2}) -> PinS-IFR+SurfPad Env={:.0}% (Xfer {:.0}m, Scrub {:.2})",
                                        v_rep.system_flight_envelope_open_pct,
                                        v_rep.mean_secondary_burn_transfer_min,
                                        v_rep.expected_weather_smoke_scrubbed_sorties,
                                        i_rep.system_flight_envelope_open_pct,
                                        i_rep.mean_secondary_burn_transfer_min,
                                        i_rep.expected_weather_smoke_scrubbed_sorties
                                    ));
                                    self.latest_medevac_report = Some(i_rep);
                                    self.layer = HeatmapLayer::AeromedicalMedevac;
                                    self.heatmap_dirty = true;
                                }
                            });

                            if let Some(ab_txt) = &self.medevac_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(mrep) = self.latest_medevac_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Pads  : {} total | StructOK={:.1}% | SmokeVis={:.2} SM | EnvOpen={:.1}%\nBurns : Crit={:.2} | SecXfer Req={:.2} | SceneAirlift={:.2}\nModes : AirXfer={:.2} (WRBDC={:.2}) | CCT Ground={:.2} | Scrubs={:.2}\nSpeed : Air={:.1}m vs CCT={:.1}m | Mean Xfer={:.1}m\nSurviv: <=4hr Burn Ctr={:.1}% | Prev. Xfer Deaths={:.2}",
                                    mrep.total_helipads,
                                    mrep.mean_helipad_structural_ok_pct,
                                    mrep.mean_helipad_smoke_vis_sm,
                                    mrep.system_flight_envelope_open_pct,
                                    mrep.expected_burn_critical_casualties,
                                    mrep.expected_burn_requiring_secondary_transfer,
                                    mrep.expected_primary_scene_airlifts,
                                    mrep.expected_air_secondary_burn_transfers,
                                    mrep.expected_wrbdc_regional_air_transfers,
                                    mrep.expected_cct_ground_fallback_transfers,
                                    mrep.expected_weather_smoke_scrubbed_sorties,
                                    mrep.mean_air_burn_transfer_min,
                                    mrep.mean_cct_ground_transfer_min,
                                    mrep.mean_secondary_burn_transfer_min,
                                    mrep.burn_definitive_care_within_4hr_pct,
                                    mrep.expected_preventable_burn_transfer_deaths
                                ));

                                if mrep.timeline.len() >= 2 {
                                    let (m_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let mp = ui.painter_at(m_rect);
                                    mp.rect_filled(m_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = mrep.timeline.len();
                                    let max_xf = mrep
                                        .timeline
                                        .last()
                                        .map(|pt| {
                                            pt.cumulative_air_burn_transfers
                                                .max(pt.cumulative_cct_ground_fallbacks)
                                                .max(1.0)
                                        })
                                        .unwrap_or(1.0) as f32;
                                    let mut poly_env = Vec::with_capacity(n_pts);
                                    let mut poly_air = Vec::with_capacity(n_pts);
                                    let mut poly_cct = Vec::with_capacity(n_pts);
                                    for (idx, pt) in mrep.timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = m_rect.left() + 10.0 + u * (m_rect.width() - 20.0);
                                        let ye = m_rect.bottom()
                                            - 14.0
                                            - ((pt.envelope_open_pads_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (m_rect.height() - 22.0);
                                        let ya = m_rect.bottom()
                                            - 14.0
                                            - ((pt.cumulative_air_burn_transfers as f32) / max_xf)
                                                .clamp(0.0, 1.0)
                                                * (m_rect.height() - 22.0);
                                        let yc = m_rect.bottom()
                                            - 14.0
                                            - ((pt.cumulative_cct_ground_fallbacks as f32) / max_xf)
                                                .clamp(0.0, 1.0)
                                                * (m_rect.height() - 22.0);
                                        poly_env.push(Pos2::new(x, ye));
                                        poly_air.push(Pos2::new(x, ya));
                                        poly_cct.push(Pos2::new(x, yc));
                                    }
                                    mp.add(egui::Shape::line(
                                        poly_env,
                                        Stroke::new(1.5, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    mp.add(egui::Shape::line(
                                        poly_air,
                                        Stroke::new(2.1, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    mp.add(egui::Shape::line(
                                        poly_cct,
                                        Stroke::new(1.7, Color32::from_rgb(251, 146, 60)),
                                    ));
                                    mp.text(
                                        Pos2::new(m_rect.left() + 8.0, m_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "Grn=FlightEnv%  Sky=Cumul Air Burn Xfer  Org=CCT Fallback",
                                        egui::FontId::monospace(8.2),
                                        Color32::GRAY,
                                    );
                                }

                                for p in &mrep.helipad_rows {
                                    ui.small(format!(
                                        "[{}] {}: Env {:.0}% ({:.1} SM) | Out {:.2} / In {:.2}",
                                        p.pad_type.short_badge(),
                                        p.name,
                                        p.flight_envelope_open_pct,
                                        p.mean_smoke_vis_sm,
                                        p.expected_outbound_burn_transfers + p.expected_primary_scene_airlifts,
                                        p.expected_inbound_burn_transfers
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Pad CSV:");
                                    ui.text_edit_singleline(&mut self.medevac_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match mrep.export_helipad_csv_path(Path::new(&self.medevac_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} helipad rows -> {}",
                                                    self.medevac_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::EvacuationContraflow => {
                            ui.label(
                                "Couples FFE thermal road closures, Wetterberg/Intini smoke speed reduction & protected EMS/LZ spines to staged evacuation & contraflow.",
                            );
                            egui::ComboBox::from_id_salt("evac_stg_cb")
                                .selected_text(self.evac_cfg.staging_strategy.label())
                                .show_ui(ui, |ui| {
                                    for st in [
                                        EvacuationStagingStrategy::SimultaneousUnstaged,
                                        EvacuationStagingStrategy::FireProximityThreeStage,
                                        EvacuationStagingStrategy::VulnerabilityAndFireWeighted,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.evac_cfg.staging_strategy,
                                            st,
                                            st.label(),
                                        );
                                    }
                                });
                            egui::ComboBox::from_id_salt("evac_cf_cb")
                                .selected_text(self.evac_cfg.contraflow_policy.label())
                                .show_ui(ui, |ui| {
                                    for cp in [
                                        ContraflowReversalPolicy::NoContraflowBaseline,
                                        ContraflowReversalPolicy::FullUnprotectedOneWayOut,
                                        ContraflowReversalPolicy::EmsProtectedArterialContraflow,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.evac_cfg.contraflow_policy,
                                            cp,
                                            cp.label(),
                                        );
                                    }
                                });

                            ui.add(
                                egui::Slider::new(
                                    &mut self.evac_cfg.stage2_release_delay_min,
                                    5.0..=60.0,
                                )
                                .text("Stage-2 Delay (m)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.evac_cfg.stage3_release_delay_min,
                                    15.0..=110.0,
                                )
                                .text("Stage-3 Delay (m)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.evac_cfg.evac_vehicles_per_capita,
                                    0.25..=1.10,
                                )
                                .text("Veh / Capita"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.evac_cfg.fire_perimeter_road_buffer_m,
                                    25.0..=140.0,
                                )
                                .text("Fire Buffer (m)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.ffe_cfg.wind_speed_mps, 0.0..=26.0)
                                    .text("FFE Wind (m/s)"),
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run Evacuation & Contraflow MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    let erep = self.world.run_evacuation_contraflow_mc(&self.evac_cfg);
                                    self.layer = HeatmapLayer::EvacuationContraflow;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Evac MC ({:.1} ms): T90={:.1}m | Overrun={:.1} res | EMS-LZ Access={:.0}%",
                                        erep.solve_time_ms,
                                        erep.t90_clearance_min,
                                        erep.expected_fire_overrun_residents,
                                        erep.ems_lz_corridor_access_preserved_pct
                                    );
                                    self.latest_evac_report = Some(erep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Unstaged vs. Staged+CF")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();

                                    let mut u_cfg = self.evac_cfg.clone();
                                    u_cfg.staging_strategy =
                                        EvacuationStagingStrategy::SimultaneousUnstaged;
                                    u_cfg.contraflow_policy =
                                        ContraflowReversalPolicy::NoContraflowBaseline;
                                    let u_rep = self.world.run_evacuation_contraflow_mc(&u_cfg);

                                    let mut s_cfg = self.evac_cfg.clone();
                                    s_cfg.staging_strategy =
                                        EvacuationStagingStrategy::FireProximityThreeStage;
                                    s_cfg.contraflow_policy =
                                        ContraflowReversalPolicy::EmsProtectedArterialContraflow;
                                    let s_rep = self.world.run_evacuation_contraflow_mc(&s_cfg);

                                    self.evac_cfg.staging_strategy =
                                        EvacuationStagingStrategy::FireProximityThreeStage;
                                    self.evac_cfg.contraflow_policy =
                                        ContraflowReversalPolicy::EmsProtectedArterialContraflow;
                                    self.evac_ab_summary = Some(format!(
                                        "A/B CRN: Unstaged PeakVC={:.2} (Overrun {:.1} res) -> Staged+EMS-CF PeakVC={:.2} (Overrun {:.1} res, EMS {:.0}%)",
                                        u_rep.peak_bottleneck_vc_ratio,
                                        u_rep.expected_fire_overrun_residents,
                                        s_rep.peak_bottleneck_vc_ratio,
                                        s_rep.expected_fire_overrun_residents,
                                        s_rep.ems_lz_corridor_access_preserved_pct
                                    ));
                                    self.latest_evac_report = Some(s_rep);
                                    self.layer = HeatmapLayer::EvacuationContraflow;
                                    self.heatmap_dirty = true;
                                }
                            });

                            if let Some(ab_txt) = &self.evac_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(erep) = self.latest_evac_report.clone() {
                                if ui
                                    .button(
                                        egui::RichText::new(
                                            "Apply Contraflow & FFE Closures to Live Traffic (Ctrl+Z)",
                                        )
                                        .strong()
                                        .color(Color32::from_rgb(250, 204, 21)),
                                    )
                                    .clicked()
                                {
                                    let (rev_n, cl_n) =
                                        self.world.apply_evacuation_contraflow_to_network(&erep);
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Applied live contraflow ({rev_n} outbound arterials boosted, {cl_n} FFE blocks closed — Ctrl+Z to undo)"
                                    );
                                }

                                ui.separator();
                                ui.monospace(format!(
                                    "Pop/Veh: {:.0} res ({:.0} veh) | CF Reversed: {} segs ({:.2} ln-km)\nFFE/Smk: Closed={:.1} segs | SmokeSpeed={:.0}% | Peak V/C={:.2}\nClear  : T50={:.1}m | T90={:.1}m | T95={:.1}m\nSafety : Overrun Risk={:.1} res | EMS-LZ Spine={:.0}% ({:.1}m in)",
                                    erep.total_evacuees,
                                    erep.total_evac_vehicles,
                                    erep.contraflow_reversed_segments_count,
                                    erep.contraflow_lane_km_added,
                                    erep.expected_thermally_closed_segments,
                                    erep.mean_road_smoke_speed_factor * 100.0,
                                    erep.peak_bottleneck_vc_ratio,
                                    erep.t50_clearance_min,
                                    erep.t90_clearance_min,
                                    erep.t95_clearance_min,
                                    erep.expected_fire_overrun_residents,
                                    erep.ems_lz_corridor_access_preserved_pct,
                                    erep.mean_ems_inbound_response_min
                                ));

                                if erep.timeline.len() >= 2 {
                                    let (e_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let ep = ui.painter_at(e_rect);
                                    ep.rect_filled(e_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = erep.timeline.len();
                                    let max_q = erep
                                        .timeline
                                        .iter()
                                        .map(|pt| pt.active_network_queue_vehs)
                                        .fold(50.0, f64::max) as f32;
                                    let mut poly_c = Vec::with_capacity(n_pts);
                                    let mut poly_q = Vec::with_capacity(n_pts);
                                    for (idx, pt) in erep.timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = e_rect.left() + 10.0 + u * (e_rect.width() - 20.0);
                                        let yc = e_rect.bottom()
                                            - 14.0
                                            - ((pt.evacuated_pct as f32) / 100.0).clamp(0.0, 1.0)
                                                * (e_rect.height() - 22.0);
                                        let yq = e_rect.bottom()
                                            - 14.0
                                            - ((pt.active_network_queue_vehs as f32) / max_q)
                                                .clamp(0.0, 1.0)
                                                * (e_rect.height() - 22.0);
                                        poly_c.push(Pos2::new(x, yc));
                                        poly_q.push(Pos2::new(x, yq));
                                    }
                                    ep.add(egui::Shape::line(
                                        poly_c,
                                        Stroke::new(2.2, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    ep.add(egui::Shape::line(
                                        poly_q,
                                        Stroke::new(1.7, Color32::from_rgb(251, 146, 60)),
                                    ));
                                    ep.text(
                                        Pos2::new(e_rect.left() + 8.0, e_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "180m Evacuation Curve: Grn=Cleared%  Org=Active Queue Vehs",
                                        egui::FontId::monospace(8.2),
                                        Color32::GRAY,
                                    );
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Evac CSV:");
                                    ui.text_edit_singleline(&mut self.evac_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match erep.export_parcel_csv_path(Path::new(&self.evac_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} staged evacuation rows -> {}",
                                                    self.evac_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::EmergencyShelters => {
                            ui.label(
                                "FEMA P-361 / ARC 4496 Shelter Siting, ATC-20 + Water/Sewer/Power Habitability, ASHRAE44 HEPA Air Filtration & Evacuee Routing.",
                            );
                            egui::ComboBox::from_id_salt("shelt_pol_cb")
                                .selected_text(self.shelters_cfg.siting_routing_policy.label())
                                .show_ui(ui, |ui| {
                                    for sp in [
                                        ShelterSitingAndRoutingPolicy::UncoordinatedNearestShelter,
                                        ShelterSitingAndRoutingPolicy::CapacitatedDistanceOnly,
                                        ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.shelters_cfg.siting_routing_policy,
                                            sp,
                                            sp.label(),
                                        );
                                    }
                                });
                            egui::ComboBox::from_id_salt("shelt_filt_cb")
                                .selected_text(self.shelters_cfg.filtration_standard.label())
                                .show_ui(ui, |ui| {
                                    for fs in [
                                        AirFiltrationStandard::BaselineMerv8Unsealed,
                                        AirFiltrationStandard::RecirculatingMerv13Hvac,
                                        AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.shelters_cfg.filtration_standard,
                                            fs,
                                            fs.label(),
                                        );
                                    }
                                });

                            ui.add(
                                egui::Slider::new(
                                    &mut self.shelters_cfg.max_open_shelters_budget,
                                    2..=6,
                                )
                                .text("Open Shelters (K)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.shelters_cfg.sqft_per_evacuee,
                                    35.0..=80.0,
                                )
                                .text("SqFt / Evacuee"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.shelters_cfg.portable_hepa_cadr_m3h_per_100m2,
                                    0.0..=1000.0,
                                )
                                .text("HEPA CADR/100m2"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.shelters_cfg.shelter_seismic_retrofit_pct,
                                    0.0..=100.0,
                                )
                                .text("Seismic IO %"),
                            );
                            ui.checkbox(
                                &mut self.shelters_cfg.shelter_backup_power_water_sanitation,
                                "72h Backup Gen + Water Bladders + Sanitation Vaults",
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run Shelter & HEPA MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    let sh_rep =
                                        self.world.run_emergency_shelters_mc(&self.shelters_cfg);
                                    self.layer = HeatmapLayer::EmergencyShelters;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Shelters MC ({:.1} ms): CleanAir={:.1}% | InPM2.5={:.1} ug/m3 | Unmet={:.0}",
                                        sh_rep.solve_time_ms,
                                        sh_rep.sheltered_clean_air_pct,
                                        sh_rep.mean_assigned_indoor_pm25_ug_m3,
                                        sh_rep.expected_overcrowded_or_unmet_evacuees
                                    );
                                    self.latest_shelters_report = Some(sh_rep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Nearest MERV-8 vs. CleanAir HEPA")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();

                                    let mut b_cfg = self.shelters_cfg.clone();
                                    b_cfg.siting_routing_policy =
                                        ShelterSitingAndRoutingPolicy::UncoordinatedNearestShelter;
                                    b_cfg.filtration_standard =
                                        AirFiltrationStandard::BaselineMerv8Unsealed;
                                    b_cfg.shelter_backup_power_water_sanitation = false;
                                    let b_rep = self.world.run_emergency_shelters_mc(&b_cfg);

                                    let mut h_cfg = self.shelters_cfg.clone();
                                    h_cfg.siting_routing_policy =
                                        ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated;
                                    h_cfg.filtration_standard =
                                        AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa;
                                    h_cfg.shelter_backup_power_water_sanitation = true;
                                    let h_rep = self.world.run_emergency_shelters_mc(&h_cfg);

                                    self.shelters_cfg.siting_routing_policy =
                                        ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated;
                                    self.shelters_cfg.filtration_standard =
                                        AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa;
                                    self.shelters_cfg.shelter_backup_power_water_sanitation = true;
                                    self.shelters_ab_summary = Some(format!(
                                        "A/B CRN: MERV-8 CleanAir={:.0}% ({:.0} ug/m3, Unmet {:.0}) -> HEPA+Bkp CleanAir={:.0}% ({:.0} ug/m3, Unmet {:.0})",
                                        b_rep.sheltered_clean_air_pct,
                                        b_rep.mean_assigned_indoor_pm25_ug_m3,
                                        b_rep.expected_overcrowded_or_unmet_evacuees,
                                        h_rep.sheltered_clean_air_pct,
                                        h_rep.mean_assigned_indoor_pm25_ug_m3,
                                        h_rep.expected_overcrowded_or_unmet_evacuees
                                    ));
                                    self.latest_shelters_report = Some(h_rep);
                                    self.layer = HeatmapLayer::EmergencyShelters;
                                    self.heatmap_dirty = true;
                                }
                            });

                            if let Some(ab_txt) = &self.shelters_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(sh_rep) = self.latest_shelters_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Shelts : {}/{} open | NomCap={} cots | Usable={:.0} cots\nDispl  : E[N]={:.0} (P90={}) | CleanAir={:.0} ({:.1}%)\nDeficit: Unmet/Overcrowd={:.1} | Smoke >{:.0}ug={:.1}\nAccess : Mean Trav={:.1}m (P90={:.1}m) | Mean InPM2.5={:.1} ug/m3",
                                    sh_rep.selected_open_shelters_count,
                                    sh_rep.candidate_shelters_count,
                                    sh_rep.total_nominal_cot_capacity,
                                    sh_rep.expected_usable_cot_capacity,
                                    sh_rep.total_displaced_evacuees,
                                    sh_rep.p90_displaced_evacuees,
                                    sh_rep.expected_sheltered_in_clean_air,
                                    sh_rep.sheltered_clean_air_pct,
                                    sh_rep.expected_overcrowded_or_unmet_evacuees,
                                    sh_rep.config.indoor_pm25_health_limit_ug_m3,
                                    sh_rep.expected_smoke_exceedance_evacuees,
                                    sh_rep.mean_shelter_travel_min,
                                    sh_rep.p90_shelter_travel_min,
                                    sh_rep.mean_assigned_indoor_pm25_ug_m3
                                ));

                                if sh_rep.filtration_response_curve.len() >= 2 {
                                    let (s_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 90.0),
                                        Sense::hover(),
                                    );
                                    let sp = ui.painter_at(s_rect);
                                    sp.rect_filled(s_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = sh_rep.filtration_response_curve.len();
                                    let max_in = sh_rep
                                        .filtration_response_curve
                                        .last()
                                        .map(|pt| pt.merv8_indoor_pm25.max(80.0))
                                        .unwrap_or(150.0) as f32;
                                    let mut p8 = Vec::with_capacity(n_pts);
                                    let mut p13 = Vec::with_capacity(n_pts);
                                    let mut phepa = Vec::with_capacity(n_pts);
                                    for (idx, pt) in sh_rep.filtration_response_curve.iter().enumerate()
                                    {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = s_rect.left() + 10.0 + u * (s_rect.width() - 20.0);
                                        let y_of = |v: f64| -> f32 {
                                            s_rect.bottom()
                                                - 14.0
                                                - ((v as f32) / max_in).clamp(0.0, 1.0)
                                                    * (s_rect.height() - 22.0)
                                        };
                                        p8.push(Pos2::new(x, y_of(pt.merv8_indoor_pm25)));
                                        p13.push(Pos2::new(x, y_of(pt.merv13_indoor_pm25)));
                                        phepa.push(Pos2::new(x, y_of(pt.merv13_hepa_indoor_pm25)));
                                    }
                                    let y_epa = s_rect.bottom()
                                        - 14.0
                                        - ((sh_rep.config.indoor_pm25_health_limit_ug_m3 as f32)
                                            / max_in)
                                            .clamp(0.0, 1.0)
                                            * (s_rect.height() - 22.0);
                                    sp.line_segment(
                                        [
                                            Pos2::new(s_rect.left() + 8.0, y_epa),
                                            Pos2::new(s_rect.right() - 8.0, y_epa),
                                        ],
                                        Stroke::new(
                                            1.0,
                                            Color32::from_rgba_unmultiplied(250, 204, 21, 145),
                                        ),
                                    );
                                    sp.add(egui::Shape::line(
                                        p8,
                                        Stroke::new(1.7, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    sp.add(egui::Shape::line(
                                        p13,
                                        Stroke::new(1.7, Color32::from_rgb(251, 146, 60)),
                                    ));
                                    sp.add(egui::Shape::line(
                                        phepa,
                                        Stroke::new(2.2, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    sp.text(
                                        Pos2::new(s_rect.left() + 8.0, s_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "Indoor PM2.5 vs Outdoor(0-500): Red=MERV8 Org=MERV13 Grn=HEPA",
                                        egui::FontId::monospace(8.1),
                                        Color32::GRAY,
                                    );
                                }

                                for sr in &sh_rep.shelter_rows {
                                    let status = if sr.selected_open { "OPEN" } else { "STBY" };
                                    ui.small(format!(
                                        "[{} {}] {}: {:.0}/{:.0} cots | InPM {:.1}ug | ATC20 {:.0}%",
                                        sr.archetype.badge_prefix(),
                                        status,
                                        sr.name,
                                        sr.expected_assigned_evacuees,
                                        sr.mean_effective_capacity,
                                        sr.mean_indoor_pm25_ug_m3,
                                        sr.atc20_green_tag_pct
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Shelter CSV:");
                                    ui.text_edit_singleline(&mut self.shelters_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match sh_rep.export_shelter_csv_path(Path::new(&self.shelters_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} shelter habitability rows -> {}",
                                                    self.shelters_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::CommodityResupply => {
                            ui.label(
                                "Multi-Day Potable Water (18kL), Diesel Fuel (11.4kL), MRE & Medical O2 Clarke-Wright 2-Opt VRP + Holguin-Veras Deprivation.",
                            );
                            egui::ComboBox::from_id_salt("resup_pol_cb")
                                .selected_text(self.resupply_cfg.dispatch_policy.label())
                                .show_ui(ui, |ui| {
                                    for rp in [
                                        ResupplyDispatchPolicy::StaticEqualRotationMilkRun,
                                        ResupplyDispatchPolicy::GreedyMyopicReorderThreshold,
                                        ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.resupply_cfg.dispatch_policy,
                                            rp,
                                            rp.label(),
                                        );
                                    }
                                });

                            ui.add(
                                egui::Slider::new(&mut self.resupply_cfg.horizon_days, 3..=14)
                                    .text("Horizon (Days)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.resupply_cfg.initial_onsite_reserve_hours,
                                    12.0..=96.0,
                                )
                                .text("Init Reserve (h)"),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.resupply_cfg.water_tanker_trucks,
                                        1..=8,
                                    )
                                    .text("H2O Trk"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.resupply_cfg.diesel_fuel_trucks,
                                        1..=8,
                                    )
                                    .text("Diesel Trk"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.resupply_cfg.mre_ration_trucks,
                                        1..=8,
                                    )
                                    .text("MRE Trk"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.resupply_cfg.medical_oxygen_trucks,
                                        1..=8,
                                    )
                                    .text("O2 Trk"),
                                );
                            });
                            ui.checkbox(
                                &mut self.resupply_cfg.enable_helicopter_sling_resupply,
                                "UH-60 Helicopter Sling-Load Bridge Bypass",
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run Multi-Day Supply VRP MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.resupply_cfg.shelters = self.shelters_cfg.clone();

                                    let rrep =
                                        self.world.run_commodity_resupply_mc(&self.resupply_cfg);
                                    self.layer = HeatmapLayer::CommodityResupply;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Supply VRP ({:.1} ms): Security={:.1}% | CW Sav={:.1}% | Depriv={:.0}",
                                        rrep.solve_time_ms,
                                        rrep.system_supply_security_pct,
                                        rrep.mean_convoy_tour_savings_pct,
                                        rrep.total_deprivation_social_cost
                                    );
                                    self.latest_resupply_report = Some(rrep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Milk-Run vs. Clarke-Wright IRP")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep = self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.resupply_cfg.shelters = self.shelters_cfg.clone();

                                    let mut m_cfg = self.resupply_cfg.clone();
                                    m_cfg.dispatch_policy =
                                        ResupplyDispatchPolicy::StaticEqualRotationMilkRun;
                                    let m_rep = self.world.run_commodity_resupply_mc(&m_cfg);

                                    let mut c_cfg = self.resupply_cfg.clone();
                                    c_cfg.dispatch_policy =
                                        ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp;
                                    let c_rep = self.world.run_commodity_resupply_mc(&c_cfg);

                                    self.resupply_cfg.dispatch_policy =
                                        ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp;
                                    self.resupply_ab_summary = Some(format!(
                                        "A/B CRN: MilkRun Sec={:.1}% (Depriv {:.0}) -> CW-IRP Sec={:.1}% (Depriv {:.0}, Sav {:.0}%)",
                                        m_rep.system_supply_security_pct,
                                        m_rep.total_deprivation_social_cost,
                                        c_rep.system_supply_security_pct,
                                        c_rep.total_deprivation_social_cost,
                                        c_rep.mean_convoy_tour_savings_pct
                                    ));
                                    self.latest_resupply_report = Some(c_rep);
                                    self.layer = HeatmapLayer::CommodityResupply;
                                    self.heatmap_dirty = true;
                                }
                            });

                            if let Some(ab_txt) = &self.resupply_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(rrep) = self.latest_resupply_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Deliv : H2O={:.0}kL | Diesel={:.0}kL | MRE={:.1}k | O2={:.0}kL\nTours : CW Savings={:.1}% | Heli Sling Sorties={:.1}\nStock : H2O={:.0} p-h | Diesel={:.1} h | O2={:.0} pt-h\nScore : Supply Security={:.1}% | Deprivation={:.0}",
                                    rrep.total_water_delivered_kl,
                                    rrep.total_diesel_delivered_kl,
                                    rrep.total_mre_delivered_k,
                                    rrep.total_oxygen_delivered_kl,
                                    rrep.mean_convoy_tour_savings_pct,
                                    rrep.expected_helicopter_sling_sorties,
                                    rrep.expected_water_stockout_person_hrs,
                                    rrep.expected_diesel_blackout_site_hrs,
                                    rrep.expected_oxygen_shortfall_patient_hrs,
                                    rrep.system_supply_security_pct,
                                    rrep.total_deprivation_social_cost
                                ));

                                if rrep.daily_timeline.len() >= 2 {
                                    let (d_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 90.0),
                                        Sense::hover(),
                                    );
                                    let dp = ui.painter_at(d_rect);
                                    dp.rect_filled(d_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = rrep.daily_timeline.len();
                                    let max_dep = rrep
                                        .daily_timeline
                                        .last()
                                        .map(|pt| pt.cumulative_deprivation_cost.max(100.0))
                                        .unwrap_or(100.0) as f32;
                                    let mut poly_br = Vec::with_capacity(n_pts);
                                    let mut poly_wt = Vec::with_capacity(n_pts);
                                    let mut poly_dp = Vec::with_capacity(n_pts);
                                    for (idx, pt) in rrep.daily_timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = d_rect.left() + 10.0 + u * (d_rect.width() - 20.0);
                                        let yb = d_rect.bottom()
                                            - 14.0
                                            - ((pt.open_bridges_pct as f32) / 100.0).clamp(0.0, 1.0)
                                                * (d_rect.height() - 22.0);
                                        let yw = d_rect.bottom()
                                            - 14.0
                                            - ((pt.water_mains_restored_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (d_rect.height() - 22.0);
                                        let yd = d_rect.bottom()
                                            - 14.0
                                            - ((pt.cumulative_deprivation_cost as f32) / max_dep)
                                                .clamp(0.0, 1.0)
                                                * (d_rect.height() - 22.0);
                                        poly_br.push(Pos2::new(x, yb));
                                        poly_wt.push(Pos2::new(x, yw));
                                        poly_dp.push(Pos2::new(x, yd));
                                    }
                                    dp.add(egui::Shape::line(
                                        poly_br,
                                        Stroke::new(1.7, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    dp.add(egui::Shape::line(
                                        poly_wt,
                                        Stroke::new(1.7, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    dp.add(egui::Shape::line(
                                        poly_dp,
                                        Stroke::new(2.1, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    dp.text(
                                        Pos2::new(d_rect.left() + 8.0, d_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "Multi-Day: Grn=OpenBridges% Sky=WaterMains% Red=CumulDepriv",
                                        egui::FontId::monospace(8.1),
                                        Color32::GRAY,
                                    );
                                }

                                for sr in &rrep.site_rows {
                                    ui.small(format!(
                                        "[{}] {}: Sec {:.1}% | H2O {:.1}h | Dsl {:.1}h | O2 {:.1}h",
                                        sr.site_kind.badge(),
                                        sr.name,
                                        sr.service_security_pct,
                                        sr.mean_water_stockout_hrs,
                                        sr.mean_diesel_blackout_hrs,
                                        sr.mean_oxygen_stockout_hrs
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Supply CSV:");
                                    ui.text_edit_singleline(&mut self.resupply_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match rrep.export_site_csv_path(Path::new(&self.resupply_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} commodity resupply site rows -> {}",
                                                    self.resupply_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::TransitBusEvacuation => {
                            ui.label(
                                "GTFS Neighborhood Pickup Depots (400m walk), 40/60-ft Transit Shuttle Buses, ADA Wheelchair-Lift Vans & Bus+EMS Priority Contraflow into Clean-Air HEPA Shelters.",
                            );
                            egui::ComboBox::from_id_salt("tevac_lane_cb")
                                .selected_text(self.transit_evac_cfg.lane_mode.label())
                                .show_ui(ui, |ui| {
                                    for lm in [
                                        BusContraflowLaneMode::MixedAutoQueueNoPriority,
                                        BusContraflowLaneMode::CurbsideBusOnlyNoContraflow,
                                        BusContraflowLaneMode::DedicatedBusPlusEmsContraflow,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.transit_evac_cfg.lane_mode,
                                            lm,
                                            lm.label(),
                                        );
                                    }
                                });

                            egui::ComboBox::from_id_salt("tevac_strat_cb")
                                .selected_text(self.transit_evac_cfg.dispatch_strategy.label())
                                .show_ui(ui, |ui| {
                                    for ds in [
                                        TransitDispatchStrategy::FixedGtfsHeadwayLoop,
                                        TransitDispatchStrategy::DemandProportionalShuttle,
                                        TransitDispatchStrategy::FireUrgencyCleanAirTbep,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.transit_evac_cfg.dispatch_strategy,
                                            ds,
                                            ds.label(),
                                        );
                                    }
                                });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.transit_evac_cfg.transit_bus_fleet,
                                        2..=32,
                                    )
                                    .text("GTFS Buses"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.transit_evac_cfg.ada_paratransit_vans,
                                        1..=20,
                                    )
                                    .text("ADA Vans"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.transit_evac_cfg.max_pickup_depots,
                                        4..=14,
                                    )
                                    .text("Pickup Depots"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.transit_evac_cfg.clearance_window_min,
                                        45.0..=240.0,
                                    )
                                    .text("Window (m)"),
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run GTFS Bus+EMS Evac MC")
                                            .strong()
                                            .color(Color32::from_rgb(45, 212, 191)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.transit_evac_cfg.coach_spec =
                                        self.world.transit_gtfs.coach_spec;

                                    let trep = self
                                        .world
                                        .run_transit_bus_evacuation_mc(&self.transit_evac_cfg);
                                    self.layer = HeatmapLayer::TransitBusEvacuation;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Bus+EMS Evac ({:.1} ms): Clean-Air={:.1}% | Cycle={:.1}m | Stranded={:.0}",
                                        trep.solve_time_ms,
                                        trep.combined_clean_air_sheltered_pct,
                                        trep.mean_bus_shuttle_cycle_min,
                                        trep.stranded_vulnerable_pax
                                    );
                                    self.latest_transit_evac_report = Some(trep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Mixed-Queue vs. Bus+EMS Contraflow")
                                            .strong()
                                            .color(Color32::from_rgb(250, 204, 21)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();

                                    let mut m_cfg = self.transit_evac_cfg.clone();
                                    m_cfg.lane_mode =
                                        BusContraflowLaneMode::MixedAutoQueueNoPriority;
                                    m_cfg.dispatch_strategy =
                                        TransitDispatchStrategy::FixedGtfsHeadwayLoop;
                                    let m_rep = self.world.run_transit_bus_evacuation_mc(&m_cfg);

                                    let mut p_cfg = self.transit_evac_cfg.clone();
                                    p_cfg.lane_mode =
                                        BusContraflowLaneMode::DedicatedBusPlusEmsContraflow;
                                    p_cfg.dispatch_strategy =
                                        TransitDispatchStrategy::FireUrgencyCleanAirTbep;
                                    let p_rep = self.world.run_transit_bus_evacuation_mc(&p_cfg);

                                    self.transit_evac_cfg.lane_mode =
                                        BusContraflowLaneMode::DedicatedBusPlusEmsContraflow;
                                    self.transit_evac_cfg.dispatch_strategy =
                                        TransitDispatchStrategy::FireUrgencyCleanAirTbep;
                                    self.transit_evac_ab_summary = Some(format!(
                                        "A/B CRN: Mixed CleanAir={:.1}% (Cyc {:.1}m, Strand {:.0}) -> Bus+EMS CleanAir={:.1}% (Cyc {:.1}m, Strand {:.0})",
                                        m_rep.combined_clean_air_sheltered_pct,
                                        m_rep.mean_bus_shuttle_cycle_min,
                                        m_rep.stranded_vulnerable_pax,
                                        p_rep.combined_clean_air_sheltered_pct,
                                        p_rep.mean_bus_shuttle_cycle_min,
                                        p_rep.stranded_vulnerable_pax
                                    ));
                                    self.latest_transit_evac_report = Some(p_rep);
                                    self.layer = HeatmapLayer::TransitBusEvacuation;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(trep) = self.latest_transit_evac_report.clone() {
                                    if ui
                                        .small_button("Apply Bus+EMS Contraflow to Live Roadway (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let (n_b, n_c) =
                                            self.world.apply_bus_ems_contraflow_to_network(&trep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Committed {n_b} Bus+EMS priority segments & {n_c} contraflow lanes (Ctrl+Z to undo)"
                                        );
                                    }
                                }
                            });

                            if let Some(ab_txt) = &self.transit_evac_ab_summary {
                                ui.colored_label(Color32::from_rgb(253, 224, 71), ab_txt);
                            }

                            if let Some(trep) = self.latest_transit_evac_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Demand: Carless={:.0} pax ({:.1}% clr) | ADA={:.0} pax ({:.1}% clr)\nClean : {:.1}% into HEPA Shelters | Stranded={:.0} (P90={:.0})\nFleet : Bus Cyc={:.1}m ({:.1} trips/bus) | ADA={:.1} trips/van\nAccess: Carless Wait={:.1}m | ADA Wait={:.1}m | EMS Spine={:.0}%",
                                    trep.total_carless_demand_pax,
                                    trep.evacuated_carless_pct,
                                    trep.total_ada_mobility_limited_pax,
                                    trep.evacuated_ada_pct,
                                    trep.combined_clean_air_sheltered_pct,
                                    trep.stranded_vulnerable_pax,
                                    trep.p90_stranded_vulnerable_pax,
                                    trep.mean_bus_shuttle_cycle_min,
                                    trep.mean_trips_per_bus,
                                    trep.mean_trips_per_ada_van,
                                    trep.mean_carless_wait_min,
                                    trep.mean_ada_curb_wait_min,
                                    trep.ems_inbound_access_preserved_pct
                                ));

                                if trep.wave_timeline.len() >= 2 {
                                    let (w_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 88.0),
                                        Sense::hover(),
                                    );
                                    let wp = ui.painter_at(w_rect);
                                    wp.rect_filled(w_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = trep.wave_timeline.len();
                                    let mut poly_car = Vec::with_capacity(n_pts);
                                    let mut poly_ada = Vec::with_capacity(n_pts);
                                    let mut poly_cln = Vec::with_capacity(n_pts);
                                    for (idx, pt) in trep.wave_timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = w_rect.left() + 10.0 + u * (w_rect.width() - 20.0);
                                        let yc = w_rect.bottom()
                                            - 12.0
                                            - ((pt.cumul_carless_evacuated_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (w_rect.height() - 22.0);
                                        let ya = w_rect.bottom()
                                            - 12.0
                                            - ((pt.cumul_ada_evacuated_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (w_rect.height() - 22.0);
                                        let yk = w_rect.bottom()
                                            - 12.0
                                            - ((pt.cumul_combined_clean_air_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (w_rect.height() - 22.0);
                                        poly_car.push(Pos2::new(x, yc));
                                        poly_ada.push(Pos2::new(x, ya));
                                        poly_cln.push(Pos2::new(x, yk));
                                    }
                                    wp.add(egui::Shape::line(
                                        poly_car,
                                        Stroke::new(1.8, Color32::from_rgb(45, 212, 191)),
                                    ));
                                    wp.add(egui::Shape::line(
                                        poly_ada,
                                        Stroke::new(1.8, Color32::from_rgb(232, 121, 249)),
                                    ));
                                    wp.add(egui::Shape::line(
                                        poly_cln,
                                        Stroke::new(2.1, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    wp.text(
                                        Pos2::new(w_rect.left() + 8.0, w_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "Shuttle Waves: Teal=Carless% Mag=ADA-Lift% Gold=CleanAir%",
                                        egui::FontId::monospace(8.1),
                                        Color32::GRAY,
                                    );
                                }

                                for dr in &trep.depot_rows {
                                    ui.small(format!(
                                        "[PP-{}] {}: {:.0}/{:.0} clean | Walk {:.0}m | Cyc {:.1}m",
                                        dr.depot_id + 1,
                                        dr.name,
                                        dr.evacuated_clean_air_pax,
                                        dr.carless_demand_pax,
                                        dr.mean_walk_dist_m,
                                        dr.mean_shuttle_cycle_min
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Depots CSV:");
                                    ui.text_edit_singleline(&mut self.transit_evac_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match trep.export_depot_csv_path(Path::new(&self.transit_evac_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} GTFS evacuation pickup depot rows -> {}",
                                                    self.transit_evac_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::WarningMobilization => {
                            ui.label(
                                "Cellular BTS PGA/Fiber/Battery Blackout Fragility, IPAWS WEA vs. 124-dBC Outdoor Sirens & Hi-Lo/Door-to-Door PADM S-Curves + Delayed Departure Staging.",
                            );
                            egui::ComboBox::from_id_salt("warn_batt_cb")
                                .selected_text(self.warning_cfg.telecom_backup.label())
                                .show_ui(ui, |ui| {
                                    for tb in [
                                        TelecomBackupPowerStandard::LegacyCommercial2to4HrBattery,
                                        TelecomBackupPowerStandard::FccEightHourBatteryStandard,
                                        TelecomBackupPowerStandard::Cpuc72HrHardenedWithLeoBackhaul,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.warning_cfg.telecom_backup,
                                            tb,
                                            tb.label(),
                                        );
                                    }
                                });

                            egui::ComboBox::from_id_salt("warn_chan_cb")
                                .selected_text(self.warning_cfg.channel_mix.label())
                                .show_ui(ui, |ui| {
                                    for cm in [
                                        AlertChannelMix::WeaAndReverse911Only,
                                        AlertChannelMix::WeaPlusOutdoorAcousticSirens,
                                        AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.warning_cfg.channel_mix,
                                            cm,
                                            cm.label(),
                                        );
                                    }
                                });

                            egui::ComboBox::from_id_salt("warn_stage_cb")
                                .selected_text(self.warning_cfg.staging_policy.label())
                                .show_ui(ui, |ui| {
                                    for sp in [
                                        DepartureStagingPolicy::UnstagedImmediateOnReceipt,
                                        DepartureStagingPolicy::StaticGeoZoneWeaStaging,
                                        DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.warning_cfg.staging_policy,
                                            sp,
                                            sp.label(),
                                        );
                                    }
                                });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.warning_cfg.hours_post_mainshock,
                                        0.25..=24.0,
                                    )
                                    .text("Post-Quake (h)"),
                                );
                                ui.checkbox(
                                    &mut self.warning_cfg.nocturnal_sleeping_event,
                                    "02:00 AM Night",
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.warning_cfg.outdoor_siren_sites,
                                        0..=12,
                                    )
                                    .text("124dB Sirens"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.warning_cfg.canvass_patrol_units,
                                        1..=20,
                                    )
                                    .text("Hi-Lo/D2D"),
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run Cell Blackout & Mobilization MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();

                                    let wrep =
                                        self.world.run_warning_mobilization_mc(&self.warning_cfg);
                                    self.layer = HeatmapLayer::WarningMobilization;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "Warning MC ({:.1} ms): BTS Dark={:.1}% | Receipt={:.1}m | Trapped={:.0}",
                                        wrep.solve_time_ms,
                                        wrep.mean_cell_towers_blackout_pct,
                                        wrep.mean_warning_receipt_min,
                                        wrep.late_departure_fire_trapped_pax
                                    );
                                    self.latest_warning_report = Some(wrep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B WEA-Only vs. Multi-Modal + Adaptive Hold")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();

                                    let mut w0_cfg = self.warning_cfg.clone();
                                    w0_cfg.channel_mix = AlertChannelMix::WeaAndReverse911Only;
                                    w0_cfg.staging_policy =
                                        DepartureStagingPolicy::UnstagedImmediateOnReceipt;
                                    let r0 = self.world.run_warning_mobilization_mc(&w0_cfg);

                                    let mut w1_cfg = self.warning_cfg.clone();
                                    w1_cfg.channel_mix =
                                        AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor;
                                    w1_cfg.staging_policy =
                                        DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging;
                                    let r1 = self.world.run_warning_mobilization_mc(&w1_cfg);

                                    self.warning_cfg.channel_mix =
                                        AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor;
                                    self.warning_cfg.staging_policy =
                                        DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging;
                                    self.warning_ab_summary = Some(format!(
                                        "A/B CRN: WEA-Only Warn={:.1}m (Trapped {:.0}) -> MultiModal+Hold Warn={:.1}m (Trapped {:.0})",
                                        r0.mean_warning_receipt_min,
                                        r0.late_departure_fire_trapped_pax,
                                        r1.mean_warning_receipt_min,
                                        r1.late_departure_fire_trapped_pax
                                    ));
                                    self.latest_warning_report = Some(r1);
                                    self.layer = HeatmapLayer::WarningMobilization;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(wrep) = self.latest_warning_report.clone() {
                                    if ui
                                        .small_button("Commit Adaptive Staging to Parcels (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let n_p =
                                            self.world.apply_warning_staging_to_parcels(&wrep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Committed adaptive departure stages across {n_p} parcels (Ctrl+Z to undo)"
                                        );
                                    }
                                }
                            });

                            if let Some(ab_txt) = &self.warning_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(wrep) = self.latest_warning_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Telecom: BTS Blackout={:.1}% | Pop in WEA Dark={:.1}%\nTiming : Warn={:.1}m | P50 Dep={:.1}m | P90 Dep={:.1}m\nStaging: Inner Clr={:.1}m | Peak V/C={:.2} | Safe Clr={:.1}%\nHazard : Fire-Trapped={:.1} pax (P90={:.1})",
                                    wrep.mean_cell_towers_blackout_pct,
                                    wrep.population_in_wea_blackout_pct,
                                    wrep.mean_warning_receipt_min,
                                    wrep.median_mobilization_departure_min,
                                    wrep.p90_mobilization_departure_min,
                                    wrep.inner_fire_zone_clearance_min,
                                    wrep.network_peak_vc_ratio,
                                    wrep.safe_evacuated_within_window_pct,
                                    wrep.late_departure_fire_trapped_pax,
                                    wrep.p90_fire_trapped_pax
                                ));

                                if wrep.mobilization_curves.len() >= 2 {
                                    let (m_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let mp = ui.painter_at(m_rect);
                                    mp.rect_filled(m_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = wrep.mobilization_curves.len();
                                    let mut poly_wea = Vec::with_capacity(n_pts);
                                    let mut poly_sir = Vec::with_capacity(n_pts);
                                    let mut poly_mul = Vec::with_capacity(n_pts);
                                    let mut poly_clr = Vec::with_capacity(n_pts);
                                    for (idx, pt) in wrep.mobilization_curves.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = m_rect.left() + 10.0 + u * (m_rect.width() - 20.0);
                                        let y_fn = |pct: f64| -> f32 {
                                            m_rect.bottom()
                                                - 12.0
                                                - ((pct as f32) / 100.0).clamp(0.0, 1.0)
                                                    * (m_rect.height() - 22.0)
                                        };
                                        poly_wea.push(Pos2::new(x, y_fn(pt.wea_only_mobilized_pct)));
                                        poly_sir.push(Pos2::new(
                                            x,
                                            y_fn(pt.wea_plus_siren_mobilized_pct),
                                        ));
                                        poly_mul
                                            .push(Pos2::new(x, y_fn(pt.multimodal_mobilized_pct)));
                                        poly_clr
                                            .push(Pos2::new(x, y_fn(pt.safe_corridor_cleared_pct)));
                                    }
                                    mp.add(egui::Shape::line(
                                        poly_wea,
                                        Stroke::new(1.7, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    mp.add(egui::Shape::line(
                                        poly_sir,
                                        Stroke::new(1.7, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    mp.add(egui::Shape::line(
                                        poly_mul,
                                        Stroke::new(2.1, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    mp.add(egui::Shape::line(
                                        poly_clr,
                                        Stroke::new(1.8, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    mp.text(
                                        Pos2::new(m_rect.left() + 8.0, m_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "PADM S-Curves: Red=WEA Gold=+Siren Grn=MultiModal Sky=Cleared",
                                        egui::FontId::monospace(8.0),
                                        Color32::GRAY,
                                    );
                                }

                                for tr in &wrep.tower_rows {
                                    ui.small(format!(
                                        "[BTS-{}] {}: WEA {:.0}% (Str {:.0}% | Fib {:.0}% | Batt {:.0}%)",
                                        tr.tower_id + 1,
                                        tr.name,
                                        tr.wea_cell_online_pct,
                                        tr.structural_intact_pct,
                                        tr.fiber_backhaul_ok_pct,
                                        tr.battery_energized_pct
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("BTS CSV:");
                                    ui.text_edit_singleline(&mut self.warning_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match wrep.export_tower_csv_path(Path::new(&self.warning_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} cellular BTS fragility rows -> {}",
                                                    self.warning_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::FemaCommsArchitecture => {
                            ui.label(
                                "FEMA IPAWS-OPEN (CAP v1.2, WEA 3.0, NPWS/PEP 50kW AM/FM EAS, NOAA 162MHz SAME), FNARS HF Skywave & Region 10 MERS IRV/MEOV ESF-2 LMR.",
                            );
                            egui::ComboBox::from_id_salt("fema_eoc_cb")
                                .selected_text(self.fema_comms_cfg.eoc_uplink.label())
                                .show_ui(ui, |ui| {
                                    for ul in [
                                        EocIpaswUplinkMode::CommercialTerrestrialFiberOnly,
                                        EocIpaswUplinkMode::FiberPlusFnarsHfSkywaveRelay,
                                        EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.fema_comms_cfg.eoc_uplink,
                                            ul,
                                            ul.label(),
                                        );
                                    }
                                });

                            egui::ComboBox::from_id_salt("fema_posture_cb")
                                .selected_text(self.fema_comms_cfg.posture.label())
                                .show_ui(ui, |ui| {
                                    for fp in [
                                        FemaCommsPosture::IpaswWea30CellularOnly,
                                        FemaCommsPosture::IpaswTriPathWeaEasPepNwrSame,
                                        FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.fema_comms_cfg.posture,
                                            fp,
                                            fp.label(),
                                        );
                                    }
                                });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_comms_cfg.household_eas_am_fm_radio_pct,
                                        5.0..=90.0,
                                    )
                                    .text("EAS AM/FM %"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_comms_cfg.household_nwr_same_receiver_pct,
                                        5.0..=85.0,
                                    )
                                    .text("NWR SAME %"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_comms_cfg.fema_mers_irv_units,
                                        0..=6,
                                    )
                                    .text("MERS IRVs"),
                                );
                                ui.checkbox(
                                    &mut self.fema_comms_cfg.airlift_mers_flyaway_kits,
                                    "UH-60 Fly-Away VSAT",
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run FEMA IPAWS & MERS ESF-2 MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();
                                    self.fema_comms_cfg.warning = self.warning_cfg.clone();

                                    let frep =
                                        self.world.run_fema_comms_mc(&self.fema_comms_cfg);
                                    self.layer = HeatmapLayer::FemaCommsArchitecture;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "FEMA IPAWS ({:.1} ms): CAP={:.1}% | Public={:.1}% | ESF-2 LMR={:.1}%",
                                        frep.solve_time_ms,
                                        frep.eoc_cap_origination_uptime_pct,
                                        frep.public_ipaws_verified_reach_pct,
                                        frep.esf2_nifog_responder_lmr_coverage_pct
                                    );
                                    self.latest_fema_comms_report = Some(frep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Fiber+WEA vs. Full FEMA IPAWS+PEP+NWR+MERS")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();
                                    self.fema_comms_cfg.warning = self.warning_cfg.clone();

                                    let mut f0 = self.fema_comms_cfg.clone();
                                    f0.eoc_uplink =
                                        EocIpaswUplinkMode::CommercialTerrestrialFiberOnly;
                                    f0.posture = FemaCommsPosture::IpaswWea30CellularOnly;
                                    let r0 = self.world.run_fema_comms_mc(&f0);

                                    let mut f1 = self.fema_comms_cfg.clone();
                                    f1.eoc_uplink =
                                        EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity;
                                    f1.posture =
                                        FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment;
                                    let r1 = self.world.run_fema_comms_mc(&f1);

                                    self.fema_comms_cfg.eoc_uplink =
                                        EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity;
                                    self.fema_comms_cfg.posture =
                                        FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment;
                                    self.fema_comms_ab_summary = Some(format!(
                                        "A/B CRN: WEA+Fiber Reach={:.1}% (CAP {:.0}%, LMR {:.0}%) -> Full FEMA IPAWS+MERS Reach={:.1}% (CAP {:.0}%, LMR {:.0}%)",
                                        r0.public_ipaws_verified_reach_pct,
                                        r0.eoc_cap_origination_uptime_pct,
                                        r0.esf2_nifog_responder_lmr_coverage_pct,
                                        r1.public_ipaws_verified_reach_pct,
                                        r1.eoc_cap_origination_uptime_pct,
                                        r1.esf2_nifog_responder_lmr_coverage_pct
                                    ));
                                    self.latest_fema_comms_report = Some(r1);
                                    self.layer = HeatmapLayer::FemaCommsArchitecture;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(frep) = self.latest_fema_comms_report.clone() {
                                    if ui
                                        .small_button("Deploy MERS & IPAWS Posture Live (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let n_p =
                                            self.world.apply_fema_mers_posture_tracked(&frep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Deployed FEMA MERS/IPAWS posture across {n_p} parcels (Ctrl+Z to undo)"
                                        );
                                    }
                                }
                            });

                            if let Some(ab_txt) = &self.fema_comms_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(frep) = self.latest_fema_comms_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Uplink: COG EOC CAP={:.1}% | NPWS PEP={:.1}% | NWR SAME={:.1}%\nReach : WEA3.0={:.1}% -> Verified IPAWS Public={:.1}%\nESF #2: NIFOG Responder LMR={:.1}% | MERS Units={}\nSafety: Mean Mobilize={:.1}m | Unwarned={:.0} (P90={:.0})",
                                    frep.eoc_cap_origination_uptime_pct,
                                    frep.npws_pep_broadcast_uptime_pct,
                                    frep.noaa_nwr_same_uptime_pct,
                                    frep.commercial_wea30_uptime_pct,
                                    frep.public_ipaws_verified_reach_pct,
                                    frep.esf2_nifog_responder_lmr_coverage_pct,
                                    frep.deployed_mers_units_count,
                                    frep.mean_ipaws_mobilization_min,
                                    frep.unwarned_isolated_residents_pax,
                                    frep.p90_unwarned_residents_pax
                                ));

                                if frep.timeline.len() >= 2 {
                                    let (f_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let fp = ui.painter_at(f_rect);
                                    fp.rect_filled(f_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = frep.timeline.len();
                                    let mut poly_w = Vec::with_capacity(n_pts);
                                    let mut poly_p = Vec::with_capacity(n_pts);
                                    let mut poly_n = Vec::with_capacity(n_pts);
                                    let mut poly_m = Vec::with_capacity(n_pts);
                                    for (idx, pt) in frep.timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = f_rect.left() + 10.0 + u * (f_rect.width() - 20.0);
                                        let y_fn = |pct: f64| -> f32 {
                                            f_rect.bottom()
                                                - 12.0
                                                - ((pct as f32) / 100.0).clamp(0.0, 1.0)
                                                    * (f_rect.height() - 22.0)
                                        };
                                        poly_w.push(Pos2::new(x, y_fn(pt.wea30_only_reach_pct)));
                                        poly_p.push(Pos2::new(x, y_fn(pt.wea_plus_eas_pep_pct)));
                                        poly_n.push(Pos2::new(
                                            x,
                                            y_fn(pt.wea_eas_plus_nwr_same_pct),
                                        ));
                                        poly_m.push(Pos2::new(
                                            x,
                                            y_fn(pt.full_fema_ipaws_mers_pct),
                                        ));
                                    }
                                    fp.add(egui::Shape::line(
                                        poly_w,
                                        Stroke::new(1.6, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    fp.add(egui::Shape::line(
                                        poly_p,
                                        Stroke::new(1.7, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    fp.add(egui::Shape::line(
                                        poly_n,
                                        Stroke::new(1.8, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    fp.add(egui::Shape::line(
                                        poly_m,
                                        Stroke::new(2.2, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    fp.text(
                                        Pos2::new(f_rect.left() + 8.0, f_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "IPAWS Stack: Red=WEA Gold=+PEP-EAS Sky=+NWR-SAME Grn=+MERS",
                                        egui::FontId::monospace(8.0),
                                        Color32::GRAY,
                                    );
                                }

                                for ar in &frep.asset_rows {
                                    ui.small(format!(
                                        "[{}] {}: Online {:.0}% | CAP {:.0}% | Reach {:.0}%",
                                        ar.kind.badge(),
                                        ar.callsign_or_name,
                                        ar.post_quake_online_pct,
                                        ar.eoc_cap_link_ok_pct,
                                        ar.public_or_lmr_reach_pct
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("FEMA CSV:");
                                    ui.text_edit_singleline(&mut self.fema_comms_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match frep.export_asset_csv_path(Path::new(&self.fema_comms_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} FEMA IPAWS/MERS asset rows -> {}",
                                                    self.fema_comms_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::FemaIsbPods => {
                            ui.label(
                                "FEMA Incident Staging Base (ISB -> CSA -> C-POD 53-ft Trailers), USACE IS-26 Type I/II/III + Walk-Up M/G/c Queue Spillback & IPAWS Public POD Routing.",
                            );
                            egui::ComboBox::from_id_salt("pod_route_cb")
                                .selected_text(self.fema_pods_cfg.routing_policy.label())
                                .show_ui(ui, |ui| {
                                    for rp in [
                                        IpaswPodRoutingPolicy::UncoordinatedNearestPodRush,
                                        IpaswPodRoutingPolicy::StaticPrePublishedDistricts,
                                        IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.fema_pods_cfg.routing_policy,
                                            rp,
                                            rp.label(),
                                        );
                                    }
                                });

                            egui::ComboBox::from_id_salt("pod_isb_cb")
                                .selected_text(self.fema_pods_cfg.isb_policy.label())
                                .show_ui(ui, |ui| {
                                    for ip in [
                                        IsbReplenishmentPolicy::OvernightStaticEqualPush,
                                        IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.fema_pods_cfg.isb_policy,
                                            ip,
                                            ip.label(),
                                        );
                                    }
                                });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_pods_cfg.active_county_pods,
                                        2..=8,
                                    )
                                    .text("County PODs"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_pods_cfg.isb_daily_53ft_trailers,
                                        3..=28,
                                    )
                                    .text("ISB 53ft Trk"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_pods_cfg.base_population_needing_pods_pct,
                                        25.0..=95.0,
                                    )
                                    .text("Needy Pop %"),
                                );
                                ui.checkbox(
                                    &mut self.fema_pods_cfg.upgrade_arterial_pods_to_type1,
                                    "Type I (4-Ln) Super-PODs",
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run FEMA ISB & POD Queueing MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();
                                    self.fema_comms_cfg.warning = self.warning_cfg.clone();
                                    self.fema_pods_cfg.fema_comms = self.fema_comms_cfg.clone();

                                    let prep =
                                        self.world.run_fema_isb_pods_mc(&self.fema_pods_cfg);
                                    self.layer = HeatmapLayer::FemaIsbPods;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "FEMA PODs ({:.1} ms): Served={:.1}% | Wait={:.1}m | Spillback={:.0} vehs",
                                        prep.solve_time_ms,
                                        prep.population_demand_met_pct,
                                        prep.mean_pod_queue_wait_min,
                                        prep.peak_arterial_spillback_vehs
                                    );
                                    self.latest_fema_pods_report = Some(prep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Nearest Rush vs. IPAWS POD Load-Balancing")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();
                                    self.fema_comms_cfg.warning = self.warning_cfg.clone();
                                    self.fema_pods_cfg.fema_comms = self.fema_comms_cfg.clone();

                                    let mut p0 = self.fema_pods_cfg.clone();
                                    p0.routing_policy =
                                        IpaswPodRoutingPolicy::UncoordinatedNearestPodRush;
                                    p0.isb_policy = IsbReplenishmentPolicy::OvernightStaticEqualPush;
                                    let r0 = self.world.run_fema_isb_pods_mc(&p0);

                                    let mut p1 = self.fema_pods_cfg.clone();
                                    p1.routing_policy =
                                        IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced;
                                    p1.isb_policy =
                                        IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock;
                                    let r1 = self.world.run_fema_isb_pods_mc(&p1);

                                    self.fema_pods_cfg.routing_policy =
                                        IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced;
                                    self.fema_pods_cfg.isb_policy =
                                        IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock;
                                    self.fema_pods_ab_summary = Some(format!(
                                        "A/B CRN: Rush Wait={:.1}m (Spill {:.0}, Served {:.1}%) -> IPAWS Wait={:.1}m (Spill {:.0}, Served {:.1}%)",
                                        r0.mean_pod_queue_wait_min,
                                        r0.peak_arterial_spillback_vehs,
                                        r0.population_demand_met_pct,
                                        r1.mean_pod_queue_wait_min,
                                        r1.peak_arterial_spillback_vehs,
                                        r1.population_demand_met_pct
                                    ));
                                    self.latest_fema_pods_report = Some(r1);
                                    self.layer = HeatmapLayer::FemaIsbPods;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(prep) = self.latest_fema_pods_report.clone() {
                                    if ui
                                        .small_button("Inject POD Queue Spillback to Streets (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let n_s =
                                            self.world.apply_pod_spillback_to_network(&prep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Applied POD queue street spillback across {n_s} road segments (Ctrl+Z to undo)"
                                        );
                                    }
                                }
                            });

                            if let Some(ab_txt) = &self.fema_pods_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(prep) = self.latest_fema_pods_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Demand: Needy={:.0} pax | Served={:.0} ({:.1}%) | IPAWS={:.1}%\nQueues: Mean Wait={:.1}m | P90 Access={:.1}m | Spillback={:.0} veh\nSupply: Dry-Run Trips={:.0} | ISB Bridge Detour=+{:.1}m",
                                    prep.total_needy_residents_pax,
                                    prep.daily_served_residents_pax,
                                    prep.population_demand_met_pct,
                                    prep.ipaws_informed_households_pct,
                                    prep.mean_pod_queue_wait_min,
                                    prep.p90_pod_total_access_min,
                                    prep.peak_arterial_spillback_vehs,
                                    prep.wasted_dry_run_stockout_trips,
                                    prep.isb_bridge_detour_delay_min
                                ));

                                if prep.shift_timeline.len() >= 2 {
                                    let (p_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let pp = ui.painter_at(p_rect);
                                    pp.rect_filled(p_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = prep.shift_timeline.len();
                                    let max_w = prep
                                        .shift_timeline
                                        .iter()
                                        .map(|pt| pt.mean_wait_min)
                                        .fold(30.0f64, f64::max) as f32;
                                    let max_sp = prep
                                        .shift_timeline
                                        .iter()
                                        .map(|pt| pt.total_street_spillback_vehs)
                                        .fold(25.0f64, f64::max)
                                        as f32;
                                    let mut poly_srv = Vec::with_capacity(n_pts);
                                    let mut poly_wt = Vec::with_capacity(n_pts);
                                    let mut poly_sp = Vec::with_capacity(n_pts);
                                    for (idx, pt) in prep.shift_timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = p_rect.left() + 10.0 + u * (p_rect.width() - 20.0);
                                        let ys = p_rect.bottom()
                                            - 12.0
                                            - ((pt.cumul_residents_served_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (p_rect.height() - 22.0);
                                        let yw = p_rect.bottom()
                                            - 12.0
                                            - ((pt.mean_wait_min as f32) / max_w).clamp(0.0, 1.0)
                                                * (p_rect.height() - 22.0);
                                        let ysp = p_rect.bottom()
                                            - 12.0
                                            - ((pt.total_street_spillback_vehs as f32) / max_sp)
                                                .clamp(0.0, 1.0)
                                                * (p_rect.height() - 22.0);
                                        poly_srv.push(Pos2::new(x, ys));
                                        poly_wt.push(Pos2::new(x, yw));
                                        poly_sp.push(Pos2::new(x, ysp));
                                    }
                                    pp.add(egui::Shape::line(
                                        poly_srv,
                                        Stroke::new(2.0, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    pp.add(egui::Shape::line(
                                        poly_wt,
                                        Stroke::new(1.8, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    pp.add(egui::Shape::line(
                                        poly_sp,
                                        Stroke::new(1.9, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    pp.text(
                                        Pos2::new(p_rect.left() + 8.0, p_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "12h Shift: Grn=CumulServed% Gold=WaitMin Red=StreetSpillback",
                                        egui::FontId::monospace(8.0),
                                        Color32::GRAY,
                                    );
                                }

                                for pr in &prep.pod_rows {
                                    ui.small(format!(
                                        "[{}] {}: Wait {:.1}m | Spill {:.0} | Trk {:.1} | {:.0} pax",
                                        pr.pod_class.badge(),
                                        pr.name,
                                        pr.mean_queue_wait_min,
                                        pr.peak_spillback_vehs,
                                        pr.isb_trailers_received,
                                        pr.served_residents_pax
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("PODs CSV:");
                                    ui.text_edit_singleline(&mut self.fema_pods_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match prep.export_pod_csv_path(Path::new(&self.fema_pods_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} FEMA ISB/C-POD rows -> {}",
                                                    self.fema_pods_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::FemaUsarDebris => {
                            ui.label(
                                "FEMA Type 1 US&R Task Forces (80-pax 24h Shifts, USNG Collapse Grids & Coburn-Spence 72h Survival), ESF-3 Dozer/Loader Push & ISB->POD Reopening.",
                            );
                            egui::ComboBox::from_id_salt("usar_pol_cb")
                                .selected_text(self.fema_usar_cfg.policy.label())
                                .show_ui(ui, |ui| {
                                    for up in [
                                        DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff,
                                        DebrisAndUsarPolicy::LogisticsPodCorridorsFirstPush,
                                        DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.fema_usar_cfg.policy,
                                            up,
                                            up.label(),
                                        );
                                    }
                                });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_usar_cfg.fema_type1_usar_task_forces,
                                        1..=6,
                                    )
                                    .text("Type-1 US&R TFs"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_usar_cfg.heavy_debris_push_crews,
                                        1..=16,
                                    )
                                    .text("ESF-3 Dozers"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_usar_cfg.horizon_hours,
                                        24.0..=96.0,
                                    )
                                    .text("Window (h)"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.fema_usar_cfg.live_injection_snapshot_hour,
                                        0.0..=72.0,
                                    )
                                    .text("Snap H+"),
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run FEMA US&R & Debris Push MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();
                                    self.fema_comms_cfg.warning = self.warning_cfg.clone();
                                    self.fema_pods_cfg.fema_comms = self.fema_comms_cfg.clone();
                                    self.fema_usar_cfg.fema_pods = self.fema_pods_cfg.clone();

                                    let urep =
                                        self.world.run_fema_usar_debris_mc(&self.fema_usar_cfg);
                                    self.layer = HeatmapLayer::FemaUsarDebris;
                                    self.heatmap_dirty = true;
                                    self.io_status_msg = format!(
                                        "FEMA US&R ({:.1} ms): Rescued={:.1} ({:.1}%) | PODs Reopen={:.1}h",
                                        urep.solve_time_ms,
                                        urep.expected_live_rescues,
                                        urep.entrapped_survival_rate_pct,
                                        urep.time_to_reopen_all_pods_hrs
                                    );
                                    self.latest_fema_usar_report = Some(urep);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Haul-Off vs. Teeth-of-Comb US&R+POD Push")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    if self.world.emergency_coverage.fire_station_count == 0 {
                                        let mrep =
                                            self.world.solve_mclp_station_siting(&self.mclp_cfg);
                                        self.world.apply_mclp_siting_solution(&mrep);
                                        self.latest_mclp_report = Some(mrep);
                                    }
                                    self.ffe_cfg.seismic = self.seismic_cfg.clone();
                                    self.surge_cfg.ffe = self.ffe_cfg.clone();
                                    self.medevac_cfg.surge = self.surge_cfg.clone();
                                    self.evac_cfg.medevac = self.medevac_cfg.clone();
                                    self.shelters_cfg.evac = self.evac_cfg.clone();
                                    self.transit_evac_cfg.shelters = self.shelters_cfg.clone();
                                    self.warning_cfg.transit_evac = self.transit_evac_cfg.clone();
                                    self.fema_comms_cfg.warning = self.warning_cfg.clone();
                                    self.fema_pods_cfg.fema_comms = self.fema_comms_cfg.clone();
                                    self.fema_usar_cfg.fema_pods = self.fema_pods_cfg.clone();

                                    let mut u0 = self.fema_usar_cfg.clone();
                                    u0.policy = DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff;
                                    let r0 = self.world.run_fema_usar_debris_mc(&u0);

                                    let mut u1 = self.fema_usar_cfg.clone();
                                    u1.policy =
                                        DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush;
                                    let r1 = self.world.run_fema_usar_debris_mc(&u1);

                                    self.fema_usar_cfg.policy =
                                        DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush;
                                    self.fema_usar_ab_summary = Some(format!(
                                        "A/B CRN: HaulOff Saved={:.1} ({:.1}%, POD {:.1}h) -> TeethOfComb Saved={:.1} ({:.1}%, POD {:.1}h)",
                                        r0.expected_live_rescues,
                                        r0.entrapped_survival_rate_pct,
                                        r0.time_to_reopen_all_pods_hrs,
                                        r1.expected_live_rescues,
                                        r1.entrapped_survival_rate_pct,
                                        r1.time_to_reopen_all_pods_hrs
                                    ));
                                    self.latest_fema_usar_report = Some(r1);
                                    self.layer = HeatmapLayer::FemaUsarDebris;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(urep) = self.latest_fema_usar_report.clone() {
                                    if ui
                                        .small_button("Inject Debris/Reopen @ Snap H+ (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let snap_h = self.fema_usar_cfg.live_injection_snapshot_hour;
                                        let (n_open, n_blk) = self
                                            .world
                                            .apply_debris_and_reopening_at_hour(&urep, snap_h);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Applied H+{snap_h:.0}h debris state: {n_open} pushed open, {n_blk} still blocked (Ctrl+Z undo)"
                                        );
                                    }
                                }
                            });

                            if let Some(ab_txt) = &self.fema_usar_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab_txt);
                            }

                            if let Some(urep) = self.latest_fema_usar_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Debris: {:.0} t street rubble ({:.1} blocked segs -> {:.0} t cleared)\nUS&R  : Trapped={:.1} | Saved={:.1} ({:.1}%, P10={:.1})\nRoutes: All PODs Reopen={:.1}h | Final Open Roads={:.1}%",
                                    urep.total_street_debris_tons,
                                    urep.initial_blocked_segments_count,
                                    urep.cleared_debris_tons,
                                    urep.total_initial_entrapped_victims,
                                    urep.expected_live_rescues,
                                    urep.entrapped_survival_rate_pct,
                                    urep.p10_live_rescues,
                                    urep.time_to_reopen_all_pods_hrs,
                                    urep.final_open_corridors_pct
                                ));

                                if urep.timeline.len() >= 2 {
                                    let (u_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let up = ui.painter_at(u_rect);
                                    up.rect_filled(u_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = urep.timeline.len();
                                    let max_vic = urep.total_initial_entrapped_victims.max(5.0) as f32;
                                    let mut poly_pod = Vec::with_capacity(n_pts);
                                    let mut poly_grd = Vec::with_capacity(n_pts);
                                    let mut poly_sav = Vec::with_capacity(n_pts);
                                    for (idx, pt) in urep.timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = u_rect.left() + 10.0 + u * (u_rect.width() - 20.0);
                                        let yp = u_rect.bottom()
                                            - 12.0
                                            - ((pt.isb_to_pods_connected_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (u_rect.height() - 22.0);
                                        let yg = u_rect.bottom()
                                            - 12.0
                                            - ((pt.usng_grids_searched_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (u_rect.height() - 22.0);
                                        let ys = u_rect.bottom()
                                            - 12.0
                                            - ((pt.cumul_live_rescues as f32) / max_vic)
                                                .clamp(0.0, 1.0)
                                                * (u_rect.height() - 22.0);
                                        poly_pod.push(Pos2::new(x, yp));
                                        poly_grd.push(Pos2::new(x, yg));
                                        poly_sav.push(Pos2::new(x, ys));
                                    }
                                    up.add(egui::Shape::line(
                                        poly_pod,
                                        Stroke::new(1.8, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    up.add(egui::Shape::line(
                                        poly_grd,
                                        Stroke::new(1.8, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    up.add(egui::Shape::line(
                                        poly_sav,
                                        Stroke::new(2.2, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    up.text(
                                        Pos2::new(u_rect.left() + 8.0, u_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "72h Ops: Gold=ISB->PODs% Sky=USNGSearched% Grn=SavedLives",
                                        egui::FontId::monospace(8.0),
                                        Color32::GRAY,
                                    );
                                }

                                for gr in &urep.grid_rows {
                                    ui.small(format!(
                                        "{} {} ({}): {:.1}/{:.1} saved | Reopen {:.1}h | {:.0}t",
                                        gr.fema_marking.symbol(),
                                        gr.usng_cell_code,
                                        gr.dominant_archetype.short_code(),
                                        gr.live_rescued_victims,
                                        gr.initial_entrapped_victims,
                                        gr.road_access_reopened_hr,
                                        gr.street_debris_tons
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("USNG CSV:");
                                    ui.text_edit_singleline(&mut self.fema_usar_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match urep.export_grid_csv_path(Path::new(&self.fema_usar_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} FEMA USNG collapse search grid rows -> {}",
                                                    self.fema_usar_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::FemaGsbDrcTdms => {
                            ui.label(
                                egui::RichText::new(
                                    "FEMA 249th Prime Power GSB, TDMS Debris & DRC/MRIC IA Suite",
                                )
                                .strong(),
                            );
                            ui.small(
                                "USACE 249th Engineer Prime Power packs, EPA/FEMA-325 TDMS 75% vegetative grinding + HHW/Freon sorting, and FEMA/SBA DRC + MRIC M/G/c survivor queues.",
                            );
                            egui::ComboBox::from_id_salt("fema_rec_pol_cb")
                                .selected_text(self.fema_rec_cfg.policy.short_label())
                                .show_ui(ui, |ui| {
                                    for pol in [
                                        FemaRecoveryExecutionPolicy::SequentialAdHoc,
                                        FemaRecoveryExecutionPolicy::LifelinePowerFirst,
                                        FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.fema_rec_cfg.policy,
                                            pol,
                                            pol.short_label(),
                                        );
                                    }
                                });
                            ui.add(
                                egui::Slider::new(&mut self.fema_rec_cfg.usace_249th_prt_teams, 1..=10)
                                    .text("USACE 249th PRTs"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.fema_rec_cfg.gsb_generator_packs_inventory,
                                    2..=24,
                                )
                                .text("GSB Gen Packs"),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.fema_rec_cfg.active_tdms_sites, 1..=4)
                                    .text("TDMS Sort Yards"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.fema_rec_cfg.debris_haul_dump_trucks,
                                    6..=60,
                                )
                                .text("30-yd3 Dump Trucks"),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(&mut self.fema_rec_cfg.fixed_drc_centers, 1..=5)
                                        .text("Fixed DRCs"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut self.fema_rec_cfg.mobile_mric_units, 0..=6)
                                        .text("MRIC Vans"),
                                );
                            });
                            ui.add(
                                egui::Slider::new(&mut self.fema_rec_cfg.specialists_per_drc, 4..=24)
                                    .text("IA+SBA Staff/DRC"),
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run FEMA GSB/TDMS/DRC MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    self.fema_rec_cfg.fema_usar = self.fema_usar_cfg.clone();
                                    let rep = self.world.run_fema_gsb_drc_tdms_mc(&self.fema_rec_cfg);
                                    self.io_status_msg = format!(
                                        "FEMA Recovery ({:.1} ms): 72h Power={:.1}%, TDMS Haul={:.1}% ({:.0} yd3 saved), IA Reg={:.1}%",
                                        rep.solve_time_ms,
                                        rep.energized_lifelines_72h_pct,
                                        rep.curbside_haul_completed_14d_pct,
                                        rep.tdms_landfill_volume_saved_yd3,
                                        rep.ia_registered_14d_pct
                                    );
                                    self.latest_fema_rec_report = Some(rep);
                                    self.layer = HeatmapLayer::FemaGsbDrcTdms;
                                    self.heatmap_dirty = true;
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B AdHoc vs Integrated")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    self.fema_rec_cfg.fema_usar = self.fema_usar_cfg.clone();
                                    let mut c0 = self.fema_rec_cfg.clone();
                                    c0.policy = FemaRecoveryExecutionPolicy::SequentialAdHoc;
                                    let r0 = self.world.run_fema_gsb_drc_tdms_mc(&c0);
                                    let mut c1 = self.fema_rec_cfg.clone();
                                    c1.policy =
                                        FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite;
                                    let r1 = self.world.run_fema_gsb_drc_tdms_mc(&c1);
                                    self.fema_rec_cfg.policy =
                                        FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite;
                                    self.fema_rec_ab_summary = Some(format!(
                                        "A/B CRN: AdHoc Pwr72h={:.0}% | Saved={:.0} yd3 | Wait={:.0}m -> Suite Pwr72h={:.0}% | Saved={:.0} yd3 | Wait={:.0}m",
                                        r0.energized_lifelines_72h_pct,
                                        r0.tdms_landfill_volume_saved_yd3,
                                        r0.mean_survivor_drc_wait_min,
                                        r1.energized_lifelines_72h_pct,
                                        r1.tdms_landfill_volume_saved_yd3,
                                        r1.mean_survivor_drc_wait_min
                                    ));
                                    self.latest_fema_rec_report = Some(r1);
                                    self.layer = HeatmapLayer::FemaGsbDrcTdms;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(rrep) = self.latest_fema_rec_report.clone() {
                                    if ui
                                        .small_button("Energize GSB Backup Power (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let n_up =
                                            self.world.apply_fema_gsb_backup_power_tracked(&rrep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Energized {:.2} MW USACE 249th GSB prime power ({n_up} parcels updated, Ctrl+Z undo)",
                                            rrep.deployed_prime_power_mw
                                        );
                                    }
                                }
                            });

                            if let Some(ab) = &self.fema_rec_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab);
                            }

                            if let Some(rrep) = self.latest_fema_rec_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "GSB Power : {:.1}% lifelines <=72h (Mean {:.1}h | {:.2} MW)
TDMS Yards: {:.0}t debris | {:.1}% hauled | {:.0} yd3 saved | {} Freon
DRC & MRIC: {:.1}% IA reg ({:.0} HH) | Wait {:.1}m | ${:.2}M IA+SBA",
                                    rrep.energized_lifelines_72h_pct,
                                    rrep.mean_time_to_emergency_power_hrs,
                                    rrep.deployed_prime_power_mw,
                                    rrep.total_disaster_debris_tons,
                                    rrep.curbside_haul_completed_14d_pct,
                                    rrep.tdms_landfill_volume_saved_yd3,
                                    rrep.white_goods_freon_units,
                                    rrep.ia_registered_14d_pct,
                                    rrep.affected_ia_eligible_households,
                                    rrep.mean_survivor_drc_wait_min,
                                    rrep.projected_ia_and_sba_disbursement_musd
                                ));

                                if rrep.daily_timeline.len() >= 2 {
                                    let (d_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 90.0),
                                        Sense::hover(),
                                    );
                                    let dp = ui.painter_at(d_rect);
                                    dp.rect_filled(d_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = rrep.daily_timeline.len();
                                    let mut poly_pwr = Vec::with_capacity(n_pts);
                                    let mut poly_haul = Vec::with_capacity(n_pts);
                                    let mut poly_ia = Vec::with_capacity(n_pts);
                                    for (idx, pt) in rrep.daily_timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = d_rect.left() + 10.0 + u * (d_rect.width() - 20.0);
                                        let y_of = |pct: f64| -> f32 {
                                            d_rect.bottom()
                                                - 12.0
                                                - ((pct as f32) / 100.0).clamp(0.0, 1.0)
                                                    * (d_rect.height() - 22.0)
                                        };
                                        poly_pwr.push(Pos2::new(x, y_of(pt.critical_lifelines_energized_pct)));
                                        poly_haul.push(Pos2::new(x, y_of(pt.curbside_debris_hauled_pct)));
                                        poly_ia.push(Pos2::new(x, y_of(pt.ia_households_registered_pct)));
                                    }
                                    dp.add(egui::Shape::line(
                                        poly_pwr,
                                        Stroke::new(2.0, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    dp.add(egui::Shape::line(
                                        poly_haul,
                                        Stroke::new(1.8, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    dp.add(egui::Shape::line(
                                        poly_ia,
                                        Stroke::new(2.1, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    dp.text(
                                        Pos2::new(d_rect.left() + 8.0, d_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "14-Day Recovery: Gold=GSBPower%  Sky=TDMSHaul%  Grn=IAReg%",
                                        egui::FontId::monospace(8.0),
                                        Color32::GRAY,
                                    );
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Sites CSV:");
                                    ui.text_edit_singleline(&mut self.fema_rec_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match rrep.export_facility_csv_path(Path::new(&self.fema_rec_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} FEMA GSB/TDMS/DRC/MRIC rows -> {}",
                                                    self.fema_rec_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::WildfireAlertWestResponse => {
                            ui.label(
                                egui::RichText::new(
                                    "ALERTWest AI PTZ Cameras + NWCG WUI Suppression",
                                )
                                .strong(),
                            );
                            ui.small(
                                "Links alertwest.org PTZ camera feeds (360° 40x NIR PTZ, 2-min AI scan, co-located seismic telemetry, <50 lb Starlink MDUs) with Rothermel/NWCG VLAT Phos-Chek & Hotshot suppression.",
                            );

                            // Live Feed File Linker (.json / GeoJSON / .csv)
                            ui.horizontal(|ui| {
                                ui.label("Camera Feed:");
                                ui.text_edit_singleline(&mut self.wildfire_feed_path);
                            });
                            ui.horizontal_wrapped(|ui| {
                                if ui.small_button("Link .json / GeoJSON / .csv").clicked() {
                                    match load_wildfire_camera_feed_path(
                                        Path::new(&self.wildfire_feed_path),
                                        self.world.network.bounds,
                                    ) {
                                        Ok(cams) => {
                                            let n_c = cams.len();
                                            self.wildfire_cfg.custom_camera_sites = cams;
                                            self.io_status_msg = format!(
                                                "Linked {n_c} ALERTWest cameras from {}",
                                                self.wildfire_feed_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("Feed load error: {e}");
                                        }
                                    }
                                }
                                if ui.small_button("Write Sample Feed .json").clicked() {
                                    if let Some(parent) = Path::new(&self.wildfire_feed_path).parent()
                                    {
                                        let _ = std::fs::create_dir_all(parent);
                                    }
                                    match std::fs::write(
                                        &self.wildfire_feed_path,
                                        builtin_western_wildfire_camera_feed_json(),
                                    ) {
                                        Ok(()) => {
                                            self.io_status_msg = format!(
                                                "Wrote sample ALERTWest camera feed -> {}",
                                                self.wildfire_feed_path
                                            );
                                        }
                                        Err(e) => {
                                            self.io_status_msg = format!("Write error: {e}");
                                        }
                                    }
                                }
                                if !self.wildfire_cfg.custom_camera_sites.is_empty()
                                    && ui.small_button("Use Built-in Feed").clicked()
                                {
                                    self.wildfire_cfg.custom_camera_sites.clear();
                                }
                            });

                            egui::ComboBox::from_id_salt("wf_posture_cb")
                                .selected_text(self.wildfire_cfg.posture.short_label())
                                .show_ui(ui, |ui| {
                                    for pst in [
                                        WildfireDetectionAndAttackPosture::Legacy911GroundOnly,
                                        WildfireDetectionAndAttackPosture::AlertWestFixedPtzInitialAttack,
                                        WildfireDetectionAndAttackPosture::FullAlertWestMduAndAerialAttack,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.wildfire_cfg.posture,
                                            pst,
                                            pst.short_label(),
                                        );
                                    }
                                });
                            egui::ComboBox::from_id_salt("wf_fuel_cb")
                                .selected_text(self.wildfire_cfg.fuel_model.short_label())
                                .show_ui(ui, |ui| {
                                    for fm in [
                                        WuiFuelModel::ShortGrassFm1,
                                        WuiFuelModel::BrushWuiFm5,
                                        WuiFuelModel::HeavyChaparralSlashFm4,
                                        WuiFuelModel::ConiferTimberUnderstoryFm10,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.wildfire_cfg.fuel_model,
                                            fm,
                                            fm.short_label(),
                                        );
                                    }
                                });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(&mut self.wildfire_cfg.wind_speed_mph, 0.0..=60.0)
                                        .text("Wind mph"),
                                );
                                ui.checkbox(
                                    &mut self.wildfire_cfg.nocturnal_ignition,
                                    "40-mi NIR Night",
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wildfire_cfg.relative_humidity_pct,
                                        5.0..=75.0,
                                    )
                                    .text("RH %"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wildfire_cfg.terrain_slope_pct,
                                        0.0..=55.0,
                                    )
                                    .text("Slope %"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wildfire_cfg.starlink_mdu_units,
                                        0..=4,
                                    )
                                    .text("Starlink MDUs"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wildfire_cfg.type1_hotshot_crews_20pax,
                                        1..=8,
                                    )
                                    .text("Type-1 IHCs"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wildfire_cfg.fixed_wing_airtankers,
                                        0..=4,
                                    )
                                    .text("VLAT/LAT"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wildfire_cfg.wildland_bulldozers,
                                        0..=6,
                                    )
                                    .text("Dozers"),
                                );
                            });

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run ALERTWest Wildfire MC")
                                            .strong()
                                            .color(Color32::from_rgb(248, 113, 113)),
                                    )
                                    .clicked()
                                {
                                    let wrep = self.world.run_wildfire_alertwest_mc(&self.wildfire_cfg);
                                    self.io_status_msg = format!(
                                        "Wildfire MC ({:.1} ms): {} cams ({} LOS, CEP90={:.0}m, Det={:.1}m) -> {:.1} ac ({:.0}% cont)",
                                        wrep.solve_time_ms,
                                        wrep.active_camera_count,
                                        wrep.cameras_with_los_count,
                                        wrep.triangulation_cep90_m,
                                        wrep.mean_detection_to_dispatch_min,
                                        wrep.final_burned_area_acres,
                                        wrep.final_containment_pct
                                    );
                                    self.latest_wildfire_report = Some(wrep);
                                    self.layer = HeatmapLayer::WildfireAlertWestResponse;
                                    self.heatmap_dirty = true;
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B 911 vs ALERTWest+VLAT")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    let mut w0 = self.wildfire_cfg.clone();
                                    w0.posture = WildfireDetectionAndAttackPosture::Legacy911GroundOnly;
                                    let r0 = self.world.run_wildfire_alertwest_mc(&w0);
                                    let mut w1 = self.wildfire_cfg.clone();
                                    w1.posture =
                                        WildfireDetectionAndAttackPosture::FullAlertWestMduAndAerialAttack;
                                    let r1 = self.world.run_wildfire_alertwest_mc(&w1);
                                    self.wildfire_cfg.posture =
                                        WildfireDetectionAndAttackPosture::FullAlertWestMduAndAerialAttack;
                                    self.wildfire_ab_summary = Some(format!(
                                        "A/B CRN: 911 Det={:.1}m ({:.1} ac, Lost={:.1}) -> ALERTWest Det={:.1}m ({:.1} ac, Lost={:.1})",
                                        r0.mean_detection_to_dispatch_min,
                                        r0.final_burned_area_acres,
                                        r0.expected_wui_structures_lost,
                                        r1.mean_detection_to_dispatch_min,
                                        r1.final_burned_area_acres,
                                        r1.expected_wui_structures_lost
                                    ));
                                    self.latest_wildfire_report = Some(r1);
                                    self.layer = HeatmapLayer::WildfireAlertWestResponse;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(wrep) = self.latest_wildfire_report.clone() {
                                    if ui
                                        .small_button("Cut WUI Fuelbreak & Evac (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let (n_fb, n_cls) = self
                                            .world
                                            .apply_wui_fuelbreak_and_fire_closures_tracked(&wrep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Cut {n_fb} WUI fuelbreak parcels & closed {n_cls} fire-impinged roads (Ctrl+Z undo)"
                                        );
                                    }
                                }
                            });

                            if let Some(ab) = &self.wildfire_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab);
                            }

                            if let Some(wrep) = self.latest_wildfire_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Cameras  : {} active ({} LOS, {} ShakeAlert) | CEP90={:.1}m | Det={:.1}m
Rothermel: ROS={:.1} ch/h | Flame={:.1} ft ({})
Attack   : {:.1} ac (P90 {:.1}) | Cont={:.0}% | Phos-Chek={:.0} gal | GoEvac={:.0}",
                                    wrep.active_camera_count,
                                    wrep.cameras_with_los_count,
                                    wrep.pnsn_shakealert_colocated_count,
                                    wrep.triangulation_cep90_m,
                                    wrep.mean_detection_to_dispatch_min,
                                    wrep.fire_behavior.rate_of_spread_chains_per_hr,
                                    wrep.fire_behavior.flame_length_ft,
                                    wrep.fire_behavior.threshold_class.label(),
                                    wrep.final_burned_area_acres,
                                    wrep.p90_burned_area_acres,
                                    wrep.final_containment_pct,
                                    wrep.phos_chek_dropped_gallons,
                                    wrep.level3_go_evac_residents
                                ));

                                if wrep.timeline.len() >= 2 {
                                    let (w_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 90.0),
                                        Sense::hover(),
                                    );
                                    let wp = ui.painter_at(w_rect);
                                    wp.rect_filled(w_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = wrep.timeline.len();
                                    let max_ac = wrep
                                        .timeline
                                        .iter()
                                        .map(|t| t.active_fire_area_acres)
                                        .fold(5.0, f64::max) as f32;
                                    let mut poly_ac = Vec::with_capacity(n_pts);
                                    let mut poly_cnt = Vec::with_capacity(n_pts);
                                    for (idx, pt) in wrep.timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = w_rect.left() + 10.0 + u * (w_rect.width() - 20.0);
                                        let ya = w_rect.bottom()
                                            - 12.0
                                            - ((pt.active_fire_area_acres as f32) / max_ac)
                                                .clamp(0.0, 1.0)
                                                * (w_rect.height() - 22.0);
                                        let yc = w_rect.bottom()
                                            - 12.0
                                            - ((pt.containment_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (w_rect.height() - 22.0);
                                        poly_ac.push(Pos2::new(x, ya));
                                        poly_cnt.push(Pos2::new(x, yc));
                                    }
                                    wp.add(egui::Shape::line(
                                        poly_ac,
                                        Stroke::new(2.1, Color32::from_rgb(248, 113, 113)),
                                    ));
                                    wp.add(egui::Shape::line(
                                        poly_cnt,
                                        Stroke::new(2.0, Color32::from_rgb(52, 211, 153)),
                                    ));
                                    wp.text(
                                        Pos2::new(w_rect.left() + 8.0, w_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "WUI Progression: Red=FireAcres  Grn=NWCGContainment%",
                                        egui::FontId::monospace(8.0),
                                        Color32::GRAY,
                                    );
                                }

                                for cam in wrep.camera_sites.iter().take(6) {
                                    ui.small(format!(
                                        "[{}] {} ({:.0}x, {:.0}m) -> az {:.0} deg | conf {:.0}%",
                                        cam.kind.badge(),
                                        cam.camera_id,
                                        cam.optical_zoom_x,
                                        cam.elevation_m,
                                        cam.pan_azimuth_deg,
                                        cam.latest_ai_smoke_confidence.unwrap_or(0.0) * 100.0
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Cams CSV:");
                                    ui.text_edit_singleline(&mut self.wildfire_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match wrep.export_camera_csv_path(Path::new(&self.wildfire_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} ALERTWest camera rows -> {}",
                                                    self.wildfire_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::BurnScarDebrisFlow => {
                            ui.label(
                                egui::RichText::new(
                                    "Post-Fire Hydrophobic Runoff, USGS Debris Basins & Culvert Washouts",
                                )
                                .strong(),
                            );
                            ui.small(
                                "USFS BAER hydrophobic Curve Number (CN) runoff, USGS Staley (2017) M1 likelihood & Gartner (2014) sediment volume, and FHWA HDS-5 culvert HW/D overtopping + RDII sewer surcharge.",
                            );

                            egui::ComboBox::from_id_salt("baer_pol_cb")
                                .selected_text(self.debris_flow_cfg.baer_policy.short_label())
                                .show_ui(ui, |ui| {
                                    for bp in [
                                        BaerWatershedMitigationPolicy::UntreatedBurnScar,
                                        BaerWatershedMitigationPolicy::AerialWoodStrandMulchOnly,
                                        BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.debris_flow_cfg.baer_policy,
                                            bp,
                                            bp.short_label(),
                                        );
                                    }
                                });
                            ui.add(
                                egui::Slider::new(
                                    &mut self.debris_flow_cfg.peak_i15_intensity_mm_hr,
                                    8.0..=92.0,
                                )
                                .text("Peak I15 (mm/hr)"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.debris_flow_cfg.storm_6hr_depth_mm,
                                    15.0..=165.0,
                                )
                                .text("6-Hr Storm (mm)"),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.debris_flow_cfg.soil_kf_erodibility,
                                        0.10..=0.48,
                                    )
                                    .text("Soil KF"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.debris_flow_cfg.baseline_culvert_diameter_m,
                                        0.60..=2.20,
                                    )
                                    .text("CMP Diam (m)"),
                                );
                            });
                            ui.add(
                                egui::Slider::new(
                                    &mut self.debris_flow_cfg.delineated_basins_count,
                                    4..=12,
                                )
                                .text("USGS Sub-Basins"),
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run USGS Debris-Flow & Culvert MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    self.debris_flow_cfg.wildfire = self.wildfire_cfg.clone();
                                    let drep = self
                                        .world
                                        .run_burn_scar_debris_flow_mc(&self.debris_flow_cfg);
                                    self.io_status_msg = format!(
                                        "USGS Debris-Flow MC ({:.1} ms): CN={:.1} ({:.1}x runoff), I15_P50={:.1} mm/hr, E[Washout]={:.1}/{} culverts",
                                        drep.solve_time_ms,
                                        drep.mean_post_fire_curve_number,
                                        drep.runoff_amplification_ratio,
                                        drep.firewide_median_i15_p50_mm_hr,
                                        drep.expected_washed_out_culverts,
                                        drep.evaluated_culverts_count
                                    );
                                    self.latest_debris_flow_report = Some(drep);
                                    self.layer = HeatmapLayer::BurnScarDebrisFlow;
                                    self.heatmap_dirty = true;
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Untreated vs Full BAER")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    self.debris_flow_cfg.wildfire = self.wildfire_cfg.clone();
                                    let mut d0 = self.debris_flow_cfg.clone();
                                    d0.baer_policy = BaerWatershedMitigationPolicy::UntreatedBurnScar;
                                    let r0 = self.world.run_burn_scar_debris_flow_mc(&d0);
                                    let mut d1 = self.debris_flow_cfg.clone();
                                    d1.baer_policy =
                                        BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing;
                                    let r1 = self.world.run_burn_scar_debris_flow_mc(&d1);
                                    self.debris_flow_cfg.baer_policy =
                                        BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing;
                                    self.debris_flow_ab_summary = Some(format!(
                                        "A/B CRN: Untreated Washouts={:.1} (Sed {:.0} m3, 4mCov {:.0}%) -> Full BAER Washouts={:.1} (Sed {:.0} m3, 4mCov {:.0}%)",
                                        r0.expected_washed_out_culverts,
                                        r0.total_predicted_sediment_volume_m3,
                                        r0.post_washout_4min_fire_cov_pct,
                                        r1.expected_washed_out_culverts,
                                        r1.total_predicted_sediment_volume_m3,
                                        r1.post_washout_4min_fire_cov_pct
                                    ));
                                    self.latest_debris_flow_report = Some(r1);
                                    self.layer = HeatmapLayer::BurnScarDebrisFlow;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(drep) = self.latest_debris_flow_report.clone() {
                                    if ui
                                        .small_button("Inject Culvert Washouts + RDII (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let (n_w, n_f) = self
                                            .world
                                            .apply_culvert_washouts_and_rdii_tracked(&drep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Injected {n_w} culvert washouts & {n_f} flooded segments + {:.0} mm/hr RDII sewer surcharge (Ctrl+Z undo)",
                                            drep.coupled_rdii_sewer_inflow_mm_hr
                                        );
                                    }
                                }
                            });

                            if let Some(ab) = &self.debris_flow_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab);
                            }

                            if let Some(drep) = self.latest_debris_flow_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Hydrophobic: CN={:.1} | Runoff={:.1}mm ({:.1}x vs {:.1}mm unburned)
USGS M1+Vol: Med I15_P50={:.1} mm/h ({}/{} >P50) | Sed={:.0} m3
FHWA HDS-5 : Washout={:.1} (P90={}) / {} clv | <=4m Cov={:.1}% | RDII={:.0}mm/h",
                                    drep.mean_post_fire_curve_number,
                                    drep.mean_hydrophobic_runoff_depth_mm,
                                    drep.runoff_amplification_ratio,
                                    drep.unburned_baseline_runoff_mm,
                                    drep.firewide_median_i15_p50_mm_hr,
                                    drep.basins_exceeding_p50_count,
                                    drep.basin_rows.len(),
                                    drep.total_predicted_sediment_volume_m3,
                                    drep.expected_washed_out_culverts,
                                    drep.p90_washed_out_culverts,
                                    drep.evaluated_culverts_count,
                                    drep.post_washout_4min_fire_cov_pct,
                                    drep.coupled_rdii_sewer_inflow_mm_hr
                                ));

                                if drep.id_threshold_curve.len() >= 2 {
                                    let (id_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 88.0),
                                        Sense::hover(),
                                    );
                                    let ip = ui.painter_at(id_rect);
                                    ip.rect_filled(id_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = drep.id_threshold_curve.len();
                                    let mut poly_mean = Vec::with_capacity(n_pts);
                                    let mut poly_max = Vec::with_capacity(n_pts);
                                    for (idx, pt) in drep.id_threshold_curve.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = id_rect.left() + 10.0 + u * (id_rect.width() - 20.0);
                                        let ym = id_rect.bottom()
                                            - 12.0
                                            - ((pt.mean_basin_likelihood_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (id_rect.height() - 22.0);
                                        let yx = id_rect.bottom()
                                            - 12.0
                                            - ((pt.max_basin_likelihood_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (id_rect.height() - 22.0);
                                        poly_mean.push(Pos2::new(x, ym));
                                        poly_max.push(Pos2::new(x, yx));
                                    }
                                    ip.add(egui::Shape::line(
                                        poly_mean,
                                        Stroke::new(2.0, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    ip.add(egui::Shape::line(
                                        poly_max,
                                        Stroke::new(2.0, Color32::from_rgb(244, 63, 94)),
                                    ));
                                    ip.text(
                                        Pos2::new(id_rect.left() + 8.0, id_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "USGS M1 P(DebrisFlow | I15=8..80mm/h): Sky=Mean  Red=WorstBasin",
                                        egui::FontId::monospace(7.9),
                                        Color32::GRAY,
                                    );
                                }

                                for cr in drep.culvert_rows.iter().take(6) {
                                    ui.small(format!(
                                        "[{}] {} ({}x{:.2}m) | Plug={:.0}% | HW/D={:.2} | Wash={:.0}%",
                                        cr.hydraulic_state.badge(),
                                        cr.road_name,
                                        cr.barrels,
                                        cr.diameter_m,
                                        cr.debris_plug_pct,
                                        cr.peak_hw_over_d,
                                        cr.washout_prob_pct
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Basin CSV:");
                                    ui.text_edit_singleline(&mut self.debris_basin_csv_path);
                                    if ui.small_button("Export").clicked() {
                                        if let Ok(n_b) = drep.export_basin_csv_path(Path::new(&self.debris_basin_csv_path)) {
                                            let _ = drep.export_culvert_csv_path(Path::new(&self.debris_culvert_csv_path));
                                            self.io_status_msg = format!(
                                                "Exported {n_b} USGS basins -> {} (+ culverts -> {})",
                                                self.debris_basin_csv_path,
                                                self.debris_culvert_csv_path
                                            );
                                        }
                                    }
                                });
                            }
                        }
                        StressSubMode::WtpTurbidityStorage => {
                            ui.label(
                                egui::RichText::new(
                                    "Post-Fire Ash/Turbidity, WTP Intake Shutoffs & Storage Drawdown",
                                )
                                .strong(),
                            );
                            ui.small(
                                "Couples burn-scar sediment/ash pulses (NTU, DOC, alkaline pH) to SCADA surface-water intake shutoffs, Alum vs. FeCl3+PAC coagulant dosing, IESWTR 0.3 NTU filtration, and clearwell/tank drawdown.",
                            );

                            egui::ComboBox::from_id_salt("wtp_pol_cb")
                                .selected_text(self.wtp_turb_cfg.policy.short_label())
                                .show_ui(ui, |ui| {
                                    for wp in [
                                        WtpCoagulantAndIntakePolicy::RunToFailureAlumBaseline,
                                        WtpCoagulantAndIntakePolicy::AutomatedIntakeShutoffAlumOnly,
                                        WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.wtp_turb_cfg.policy,
                                            wp,
                                            wp.short_label(),
                                        );
                                    }
                                });
                            ui.add(
                                egui::Slider::new(
                                    &mut self.wtp_turb_cfg.intake_shutoff_threshold_ntu,
                                    100.0..=1_400.0,
                                )
                                .text("Shutoff Threshold (NTU)"),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wtp_turb_cfg.wtp_design_capacity_mgd,
                                        4.0..=45.0,
                                    )
                                    .text("WTP Cap (MGD)"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wtp_turb_cfg.finished_storage_capacity_mg,
                                        3.0..=35.0,
                                    )
                                    .text("Storage (MG)"),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wtp_turb_cfg.emergency_well_intertie_mgd,
                                        0.0..=16.0,
                                    )
                                    .text("Well/Tie (MGD)"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.wtp_turb_cfg.onsite_coagulant_inventory_tons,
                                        4.0..=55.0,
                                    )
                                    .text("Coag Tank (t)"),
                                );
                            });
                            ui.add(
                                egui::Slider::new(
                                    &mut self.wtp_turb_cfg.active_fire_hydrant_draft_gpm,
                                    0.0..=4_000.0,
                                )
                                .text("Hydrant Draft (GPM)"),
                            );

                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Run 72h WTP Turbidity & Storage MC")
                                            .strong()
                                            .color(Color32::from_rgb(56, 189, 248)),
                                    )
                                    .clicked()
                                {
                                    self.debris_flow_cfg.wildfire = self.wildfire_cfg.clone();
                                    self.wtp_turb_cfg.debris_flow = self.debris_flow_cfg.clone();
                                    let wrep =
                                        self.world.run_wtp_turbidity_storage_mc(&self.wtp_turb_cfg);
                                    self.io_status_msg = format!(
                                        "WTP Turbidity MC ({:.1} ms): Peak {:.0} NTU, Shutoff={:.1}h, MinStor={:.1}%, P(BWA)={:.1}%",
                                        wrep.solve_time_ms,
                                        wrep.peak_raw_river_ntu,
                                        wrep.intake_shutoff_duration_hrs,
                                        wrep.min_finished_storage_fill_pct,
                                        wrep.boil_water_advisory_prob_pct
                                    );
                                    self.latest_wtp_turb_report = Some(wrep);
                                    self.layer = HeatmapLayer::WtpTurbidityStorage;
                                    self.heatmap_dirty = true;
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("A/B Shutoff-Only vs FeCl3+Wells")
                                            .strong()
                                            .color(Color32::from_rgb(52, 211, 153)),
                                    )
                                    .clicked()
                                {
                                    self.debris_flow_cfg.wildfire = self.wildfire_cfg.clone();
                                    self.wtp_turb_cfg.debris_flow = self.debris_flow_cfg.clone();
                                    let mut c0 = self.wtp_turb_cfg.clone();
                                    c0.policy =
                                        WtpCoagulantAndIntakePolicy::AutomatedIntakeShutoffAlumOnly;
                                    let r0 = self.world.run_wtp_turbidity_storage_mc(&c0);
                                    let mut c1 = self.wtp_turb_cfg.clone();
                                    c1.policy =
                                        WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie;
                                    let r1 = self.world.run_wtp_turbidity_storage_mc(&c1);
                                    self.wtp_turb_cfg.policy =
                                        WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie;
                                    self.wtp_turb_ab_summary = Some(format!(
                                        "A/B CRN: ShutoffAlum MinStor={:.1}% (MinPsi={:.0}, BWA={:.0}%) -> FeCl3+Wells MinStor={:.1}% (MinPsi={:.0}, BWA={:.0}%)",
                                        r0.min_finished_storage_fill_pct,
                                        r0.min_high_zone_pressure_psi,
                                        r0.boil_water_advisory_prob_pct,
                                        r1.min_finished_storage_fill_pct,
                                        r1.min_high_zone_pressure_psi,
                                        r1.boil_water_advisory_prob_pct
                                    ));
                                    self.latest_wtp_turb_report = Some(r1);
                                    self.layer = HeatmapLayer::WtpTurbidityStorage;
                                    self.heatmap_dirty = true;
                                }
                                if let Some(wrep) = self.latest_wtp_turb_report.clone() {
                                    if ui
                                        .small_button("Inject Tank Drawdown -> 20-psi Grid (Ctrl+Z)")
                                        .clicked()
                                    {
                                        let ok_20psi =
                                            self.world.apply_wtp_storage_drawdown_tracked(&wrep);
                                        self.heatmap_dirty = true;
                                        self.io_status_msg = format!(
                                            "Injected {:.1}% min tank storage into Hazen-Williams water grid -> {:.1}% junctions >=20 psi (Ctrl+Z undo)",
                                            wrep.min_finished_storage_fill_pct,
                                            ok_20psi
                                        );
                                    }
                                }
                            });

                            if let Some(ab) = &self.wtp_turb_ab_summary {
                                ui.colored_label(Color32::from_rgb(110, 231, 183), ab);
                            }

                            if let Some(wrep) = self.latest_wtp_turb_report.clone() {
                                ui.separator();
                                ui.monospace(format!(
                                    "Raw River : Peak {:.0} NTU | DOC {:.1} mg/L | Ash pH {:.2} ({:.1}h > cutoff)
WTP & Coag: Shutoff {:.1}h | Peak {:.0} mg/L ({:.1}t used, {:.0}t sludge) | CFE {:.2} NTU
Storage   : Min {:.1}% (P10 {:.1}%) | High-Zone {:.1} psi | P(BWA)={:.1}% ({:.0} pax)",
                                    wrep.peak_raw_river_ntu,
                                    wrep.peak_raw_doc_mg_l,
                                    wrep.peak_raw_ash_ph,
                                    wrep.hours_exceeding_shutoff_ntu,
                                    wrep.intake_shutoff_duration_hrs,
                                    wrep.peak_coagulant_dose_mg_l,
                                    wrep.total_coagulant_consumed_tons,
                                    wrep.total_alum_ferric_sludge_tons,
                                    wrep.peak_cfe_filtered_ntu,
                                    wrep.min_finished_storage_fill_pct,
                                    wrep.p10_min_storage_fill_pct,
                                    wrep.min_high_zone_pressure_psi,
                                    wrep.boil_water_advisory_prob_pct,
                                    wrep.expected_residents_under_bwa
                                ));

                                if wrep.hourly_timeline.len() >= 2 {
                                    let (t_rect, _) = ui.allocate_exact_size(
                                        EguiVec2::new(365.0, 92.0),
                                        Sense::hover(),
                                    );
                                    let tp = ui.painter_at(t_rect);
                                    tp.rect_filled(t_rect, 4.0, Color32::from_rgb(10, 15, 28));
                                    let n_pts = wrep.hourly_timeline.len();
                                    let max_ntu = wrep.peak_raw_river_ntu.max(200.0) as f32;
                                    let mut poly_ntu = Vec::with_capacity(n_pts);
                                    let mut poly_stor = Vec::with_capacity(n_pts);
                                    let mut poly_coag = Vec::with_capacity(n_pts);
                                    for (idx, pt) in wrep.hourly_timeline.iter().enumerate() {
                                        let u = idx as f32 / ((n_pts - 1).max(1) as f32);
                                        let x = t_rect.left() + 10.0 + u * (t_rect.width() - 20.0);
                                        let yn = t_rect.bottom()
                                            - 12.0
                                            - ((pt.raw_river_ntu as f32) / max_ntu).clamp(0.0, 1.0)
                                                * (t_rect.height() - 22.0);
                                        let ys = t_rect.bottom()
                                            - 12.0
                                            - ((pt.storage_fill_pct as f32) / 100.0)
                                                .clamp(0.0, 1.0)
                                                * (t_rect.height() - 22.0);
                                        let yc = t_rect.bottom()
                                            - 12.0
                                            - ((pt.coagulant_dose_mg_l as f32) / 265.0)
                                                .clamp(0.0, 1.0)
                                                * (t_rect.height() - 22.0);
                                        poly_ntu.push(Pos2::new(x, yn));
                                        poly_stor.push(Pos2::new(x, ys));
                                        poly_coag.push(Pos2::new(x, yc));
                                    }
                                    tp.add(egui::Shape::line(
                                        poly_ntu,
                                        Stroke::new(1.9, Color32::from_rgb(251, 146, 60)),
                                    ));
                                    tp.add(egui::Shape::line(
                                        poly_coag,
                                        Stroke::new(1.7, Color32::from_rgb(250, 204, 21)),
                                    ));
                                    tp.add(egui::Shape::line(
                                        poly_stor,
                                        Stroke::new(2.2, Color32::from_rgb(56, 189, 248)),
                                    ));
                                    tp.text(
                                        Pos2::new(t_rect.left() + 8.0, t_rect.top() + 3.0),
                                        egui::Align2::LEFT_TOP,
                                        "72h WTP: Org=RawNTU  Gold=CoagDose  Sky=FinishedStorage%",
                                        egui::FontId::monospace(7.9),
                                        Color32::GRAY,
                                    );
                                }

                                ui.horizontal(|ui| {
                                    ui.label("72h CSV:");
                                    ui.text_edit_singleline(&mut self.wtp_turb_csv_path);
                                    if ui.small_button("Export .csv").clicked() {
                                        match wrep.export_timeline_csv_path(Path::new(&self.wtp_turb_csv_path)) {
                                            Ok(n_r) => {
                                                self.io_status_msg = format!(
                                                    "Exported {n_r} hourly WTP turbidity/coagulant/storage rows -> {}",
                                                    self.wtp_turb_csv_path
                                                );
                                            }
                                            Err(e) => {
                                                self.io_status_msg = format!("CSV failed: {e}");
                                            }
                                        }
                                    }
                                });
                            }
                        }
                        }
                    }
                    DockTab::UtilityGrids => {
                        ui.heading("Electrical, Potable Water & Wastewater Grids");
                        ui.label(
                            "Coupled 12.47 kV DistFlow power, Hazen-Williams 20-psi fire-hydrant water mains, and Manning + EPA RDII sanitary sewers.",
                        );
                        ui.separator();

                        let mut ucfg = self.world.utility_config.clone();
                        let mut changed = false;

                        egui::ComboBox::from_id_salt("util_sublayer_cb")
                            .selected_text(ucfg.heatmap_sublayer.label())
                            .show_ui(ui, |ui| {
                                for sl in [
                                    UtilityHeatmapSublayer::CompositeLifelineHealth,
                                    UtilityHeatmapSublayer::ElectricVoltagePu,
                                    UtilityHeatmapSublayer::WaterResidualPressurePsi,
                                    UtilityHeatmapSublayer::WastewaterSewerLoading,
                                ] {
                                    if ui
                                        .selectable_value(&mut ucfg.heatmap_sublayer, sl, sl.label())
                                        .changed()
                                    {
                                        changed = true;
                                        self.layer = HeatmapLayer::UtilityLifelines;
                                    }
                                }
                            });

                        if ui
                            .button(
                                egui::RichText::new("Show Utility Lifelines Heatmap on GPU")
                                    .strong()
                                    .color(Color32::from_rgb(56, 189, 248)),
                            )
                            .clicked()
                        {
                            self.layer = HeatmapLayer::UtilityLifelines;
                            self.heatmap_dirty = true;
                        }

                        ui.separator();
                        ui.label(egui::RichText::new("1. 12.47 kV Electrical Grid & DERs:").strong());
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut ucfg.substation_voltage_pu, 0.96..=1.05)
                                    .text("OLTC V0 (p.u.)"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut ucfg.ev_charging_penetration_pct, 0.0..=100.0)
                                    .text("EV Charging %"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut ucfg.solar_pv_penetration_pct, 0.0..=100.0)
                                    .text("Rooftop PV %"),
                            )
                            .changed();

                        let n_subs = self.world.utility_report.electrical.num_substations_total.max(1);
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Substations:");
                            for s_idx in 0..n_subs {
                                let tripped = ucfg.tripped_substation_indices.contains(&s_idx);
                                let lbl = if tripped {
                                    format!("Sub #{} [TRIPPED]", s_idx + 1)
                                } else {
                                    format!("Sub #{} [ONLINE]", s_idx + 1)
                                };
                                let col = if tripped {
                                    Color32::from_rgb(244, 63, 94)
                                } else {
                                    Color32::from_rgb(52, 211, 153)
                                };
                                if ui.button(egui::RichText::new(lbl).color(col)).clicked() {
                                    if tripped {
                                        ucfg.tripped_substation_indices.retain(|&x| x != s_idx);
                                    } else {
                                        ucfg.tripped_substation_indices.push(s_idx);
                                    }
                                    changed = true;
                                    self.layer = HeatmapLayer::UtilityLifelines;
                                }
                            }
                        });

                        ui.separator();
                        ui.label(
                            egui::RichText::new("2. Potable Water Mains (Hazen-Williams + Fire):")
                                .strong(),
                        );
                        changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut ucfg.active_fire_flow_demand_gpm,
                                    0.0..=4_500.0,
                                )
                                .text("Hydrant GPM"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut ucfg.hazen_williams_c, 75.0..=145.0)
                                    .text("Pipe C-Factor"),
                            )
                            .changed();

                        ui.separator();
                        ui.label(
                            egui::RichText::new("3. Sanitary Sewer (Manning + Storm RDII):").strong(),
                        );
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut ucfg.storm_rainfall_mm_hr, 0.0..=80.0)
                                    .text("Storm mm/hr"),
                            )
                            .changed();

                        ui.separator();
                        ui.label(
                            egui::RichText::new("4. Cross-Lifeline Interdependencies (Rinaldi 2001):")
                                .strong(),
                        );
                        changed |= ui
                            .checkbox(
                                &mut ucfg.couple_closed_roads_to_utility_mains,
                                "Closed Bridges Sever Utility Mains",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(
                                &mut ucfg.critical_backup_generators_enabled,
                                "Microgrid / Diesel Backup on Pumps, WWTP & Signals",
                            )
                            .changed();

                        if changed {
                            let _ = self.world.evaluate_utility_grids(&ucfg);
                            self.heatmap_dirty = true;
                        }

                        if self.world.utility_report.electrical.darkened_signal_junctions_count > 0
                            && ui
                                .button(
                                    egui::RichText::new(format!(
                                        "Cascade {} Darkened Signals -> 4-Way Yield (Ctrl+Z)",
                                        self.world
                                            .utility_report
                                            .electrical
                                            .darkened_signal_junctions_count
                                    ))
                                    .color(Color32::from_rgb(251, 146, 60)),
                                )
                                .clicked()
                        {
                            let deg = self.world.apply_darkened_signal_fallback_tracked();
                            self.io_status_msg = format!(
                                "Cascaded {deg} unpowered signalized junctions to 4-Way Yield (Ctrl+Z to restore)"
                            );
                        }

                        let urep = &self.world.utility_report;
                        ui.separator();
                        ui.monospace(format!(
                            "Lifeline Health : {:.1}% ({:.2} ms | {:02}:00)
[ELEC] Energized: {:.0}% | V={:.3}..{:.3} pu (ANSI {:.0}%)
       Load {:.2} MW - PV {:.2} MW + Loss {:.0} kW | MaxFdr {:.0}%
[WATR] Boosters : {} | P={:.1}..{:.1} psi (>=20psi: {:.0}%)
       Supply {:.1} L/s (Fire {:.1} L/s) | Pump {:.0} kW
[SEWR] Lift/WWTP: {} | DWF {:.1} + RDII {:.1} = {:.1} L/s
       WWTP {:.0}% cap | Max d/D={:.2} | SSO={} ({:.1} L/s)",
                            urep.composite_lifeline_score_pct,
                            urep.solve_time_ms,
                            (urep.clock_hour.floor() as usize) % 24,
                            urep.electrical.energized_nodes_pct,
                            urep.electrical.min_voltage_pu,
                            urep.electrical.max_voltage_pu,
                            urep.electrical.ansi_c84_1_compliance_pct,
                            urep.electrical.total_load_mw,
                            urep.electrical.solar_pv_generation_mw,
                            urep.electrical.line_resistive_losses_kw,
                            urep.electrical.max_feeder_loading_pct,
                            if urep.water.booster_pumps_energized { "ON" } else { "OFF (GRAVITY)" },
                            urep.water.min_residual_pressure_psi,
                            urep.water.max_residual_pressure_psi,
                            urep.water.nodes_meeting_20psi_fire_pct,
                            urep.water.total_supply_lps,
                            urep.water.fire_hydrant_demand_lps,
                            urep.water.pumping_power_kw,
                            if urep.wastewater.wwtp_energized { "ONLINE" } else { "TRIPPED" },
                            urep.wastewater.dry_weather_sanitary_lps,
                            urep.wastewater.wet_weather_rdii_lps,
                            urep.wastewater.total_wwtp_influent_lps,
                            urep.wastewater.wwtp_utilization_pct,
                            urep.wastewater.max_pipe_d_over_d,
                            urep.wastewater.surcharged_sso_nodes_count,
                            urep.wastewater.estimated_sso_overflow_lps,
                        ));

                        // 4-Bar Lifeline Compliance Gauge (`ANSI C84.1`, `Water >=20psi`, `Sewer Free-Flow`, `WWTP Headroom`)
                        let (g_rect, _) =
                            ui.allocate_exact_size(EguiVec2::new(365.0, 78.0), Sense::hover());
                        let gp = ui.painter_at(g_rect);
                        gp.rect_filled(g_rect, 4.0, Color32::from_rgb(10, 15, 28));
                        let n_j_tot = urep.node_telemetry.len().max(1) as f64;
                        let sewer_ok_pct = ((urep.node_telemetry.len()
                            - urep.wastewater.surcharged_sso_nodes_count)
                            as f64
                            / n_j_tot)
                            * 100.0;
                        let wwtp_headroom_pct =
                            (100.0 - (urep.wastewater.wwtp_utilization_pct - 60.0).max(0.0))
                                .clamp(0.0, 100.0);
                        let bars = [
                            ("Elec ANSI V", urep.electrical.ansi_c84_1_compliance_pct),
                            ("Water >=20psi", urep.water.nodes_meeting_20psi_fire_pct),
                            ("Sewer No-SSO", sewer_ok_pct),
                            ("WWTP Margin", wwtp_headroom_pct),
                        ];
                        for (bi, (blbl, bpct)) in bars.iter().enumerate() {
                            let y = g_rect.top() + 5.0 + (bi as f32) * 17.5;
                            let frac = (*bpct / 100.0).clamp(0.0, 1.0) as f32;
                            let bcol = if *bpct >= 90.0 {
                                Color32::from_rgb(52, 211, 153)
                            } else if *bpct >= 65.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            gp.text(
                                Pos2::new(g_rect.left() + 6.0, y + 6.5),
                                egui::Align2::LEFT_CENTER,
                                *blbl,
                                egui::FontId::monospace(9.0),
                                Color32::LIGHT_GRAY,
                            );
                            gp.rect_filled(
                                Rect::from_min_size(
                                    Pos2::new(g_rect.left() + 108.0, y + 2.0),
                                    EguiVec2::new((frac * 195.0).max(2.0), 10.5),
                                ),
                                2.0,
                                bcol,
                            );
                            gp.text(
                                Pos2::new(g_rect.right() - 6.0, y + 6.5),
                                egui::Align2::RIGHT_CENTER,
                                format!("{:.0}%", bpct),
                                egui::FontId::monospace(9.0),
                                Color32::WHITE,
                            );
                        }

                        ui.horizontal(|ui| {
                            ui.label("CSV:");
                            ui.text_edit_singleline(&mut self.utility_csv_path);
                            if ui.small_button("Export .csv").clicked() {
                                match self
                                    .world
                                    .utility_report
                                    .export_csv_path(Path::new(&self.utility_csv_path))
                                {
                                    Ok(n_r) => {
                                        self.io_status_msg = format!(
                                            "Exported {n_r} utility junction rows -> {}",
                                            self.utility_csv_path
                                        );
                                    }
                                    Err(e) => {
                                        self.io_status_msg = format!("CSV failed: {e}");
                                    }
                                }
                            }
                        });
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
            if response.dragged() {
                if self.tool == ViewportTool::PaintZone {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let wx = self.cam_x + (pos.x - rect.center().x) as f64 / self.cam_zoom;
                        let wy = self.cam_y + (pos.y - rect.center().y) as f64 / self.cam_zoom;
                        if let Some(pid) =
                            self.world.paint_parcel_zone_at(wx, wy, self.active_brush)
                        {
                            self.io_status_msg = format!(
                                "Painted parcel #{pid} -> {}",
                                self.active_brush.spec().label
                            );
                            self.heatmap_dirty = true;
                        }
                    }
                } else {
                    let d = response.drag_delta();
                    self.cam_x -= d.x as f64 / self.cam_zoom;
                    self.cam_y -= d.y as f64 / self.cam_zoom;
                }
            }
            if response.hovered() {
                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.1 {
                    let factor = if scroll > 0.0 { 1.12 } else { 1.0 / 1.12 };
                    self.cam_zoom = (self.cam_zoom * factor).clamp(0.18, 10.5);
                }
            }

            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.draw_start_world = None;
            }
            if !ctx.wants_keyboard_input() {
                ctx.input(|i| {
                    if i.key_pressed(egui::Key::Num1) {
                        self.tool = ViewportTool::Inspect;
                        self.draw_start_world = None;
                    } else if i.key_pressed(egui::Key::Num2) {
                        self.tool = ViewportTool::DrawRoad;
                    } else if i.key_pressed(egui::Key::Num3) {
                        self.tool = ViewportTool::EditLanes;
                        self.draw_start_world = None;
                    } else if i.key_pressed(egui::Key::Num4) {
                        self.tool = ViewportTool::ConvertRoundabout;
                        self.draw_start_world = None;
                    } else if i.key_pressed(egui::Key::Num5) {
                        self.tool = ViewportTool::PaintZone;
                        self.draw_start_world = None;
                    } else if i.key_pressed(egui::Key::Num6) {
                        self.tool = ViewportTool::PlaceFacility;
                        self.draw_start_world = None;
                        self.layer = HeatmapLayer::EmergencyResponse4Min;
                        self.heatmap_dirty = true;
                    } else if i.key_pressed(egui::Key::Num7) {
                        self.tool = ViewportTool::ToggleClosure;
                        self.draw_start_world = None;
                        self.layer = HeatmapLayer::EmergencyResponse4Min;
                        self.heatmap_dirty = true;
                    } else if i.key_pressed(egui::Key::E) {
                        if let Some(ev_id) = self
                            .world
                            .dispatch_emergency_on_segment(Some(self.selected_segment))
                        {
                            self.io_status_msg = format!(
                                "Dispatched Emergency Unit #{ev_id} on corridor #{}",
                                self.selected_segment
                            );
                        }
                    }
                });
            }

            let hover_world: Option<(Pos2, f64, f64)> = response.hover_pos().map(|pos| {
                let wx = self.cam_x + (pos.x - rect.center().x) as f64 / self.cam_zoom;
                let wy = self.cam_y + (pos.y - rect.center().y) as f64 / self.cam_zoom;
                (pos, wx, wy)
            });

            if response.secondary_clicked() {
                if self.tool == ViewportTool::DrawRoad && self.draw_start_world.is_some() {
                    self.draw_start_world = None;
                } else if self.tool == ViewportTool::ToggleClosure {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let wx = self.cam_x + (pos.x - rect.center().x) as f64 / self.cam_zoom;
                        let wy = self.cam_y + (pos.y - rect.center().y) as f64 / self.cam_zoom;
                        if let Some((sid, _, name)) = self.world.toggle_corridor_closure_at(
                            wx,
                            wy,
                            self.stress_cfg.bidirectional_closure,
                            Some(false),
                        ) {
                            self.io_status_msg = format!(
                                "Reopened `{name}` (#{sid}) -> <=4m cov {:.1}%",
                                self.world.emergency_coverage.parcels_within_4min_pct
                            );
                            self.heatmap_dirty = true;
                        }
                    }
                } else if self.tool == ViewportTool::PlaceFacility {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let wx = self.cam_x + (pos.x - rect.center().x) as f64 / self.cam_zoom;
                        let wy = self.cam_y + (pos.y - rect.center().y) as f64 / self.cam_zoom;
                        if let Some(rem) = self.world.remove_emergency_facility_near(wx, wy, 34.0) {
                            self.io_status_msg = format!(
                                "Removed {} (4-min coverage now {:.0}%)",
                                rem.name, self.world.emergency_coverage.parcels_within_4min_pct
                            );
                            self.heatmap_dirty = true;
                        }
                    }
                } else if self.tool == ViewportTool::EditLanes {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let wx = self.cam_x + (pos.x - rect.center().x) as f64 / self.cam_zoom;
                        let wy = self.cam_y + (pos.y - rect.center().y) as f64 / self.cam_zoom;
                        if let Some((sid, _, _)) =
                            hit_test_segment(&self.world.network, wx, wy, 18.0)
                        {
                            self.selected_segment = sid;
                            self.world.watched_segment_id = sid;
                            if let Ok(nl) = adjust_segment_lane_count(
                                &mut self.world,
                                sid,
                                -1,
                                self.symmetric_lane_edit,
                            ) {
                                self.io_status_msg =
                                    format!("Road diet on segment #{sid} -> {nl} lanes/dir");
                                self.heatmap_dirty = true;
                            }
                        }
                    }
                }
            }

            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let wx = self.cam_x + (pos.x - rect.center().x) as f64 / self.cam_zoom;
                    let wy = self.cam_y + (pos.y - rect.center().y) as f64 / self.cam_zoom;

                    match self.tool {
                        ViewportTool::DrawRoad => {
                            let snap = preview_draw_snap(&self.world.network, wx, wy);
                            let pt = snap.pos();
                            if let Some(origin) = self.draw_start_world {
                                match draw_road_segment(&mut self.world, origin, pt, self.draw_spec)
                                {
                                    Ok(new_sids) => {
                                        if let Some(&first_new) = new_sids.first() {
                                            self.selected_segment = first_new;
                                            self.world.watched_segment_id = first_new;
                                        }
                                        self.io_status_msg = format!(
                                            "Built new road ({:.0} m, {} ln/dir)",
                                            origin.dist(pt),
                                            self.draw_spec.lanes_per_dir
                                        );
                                        self.draw_start_world = None;
                                        self.heatmap_dirty = true;
                                    }
                                    Err(e) => {
                                        self.io_status_msg = format!("Draw road: {e}");
                                    }
                                }
                            } else {
                                self.draw_start_world = Some(pt);
                                self.io_status_msg =
                                    "Origin anchored — click destination point or existing street to connect."
                                        .to_string();
                            }
                        }
                        ViewportTool::EditLanes => {
                            if let Some((sid, _, _)) =
                                hit_test_segment(&self.world.network, wx, wy, 18.0)
                            {
                                self.selected_segment = sid;
                                self.world.watched_segment_id = sid;
                                if let Ok(nl) = adjust_segment_lane_count(
                                    &mut self.world,
                                    sid,
                                    1,
                                    self.symmetric_lane_edit,
                                ) {
                                    self.io_status_msg =
                                        format!("Widened segment #{sid} -> {nl} lanes/dir");
                                    self.heatmap_dirty = true;
                                }
                            }
                        }
                        ViewportTool::ConvertRoundabout => {
                            if let Some(jid) = hit_test_junction(&self.world.network, wx, wy, 24.0)
                            {
                                self.selected_junction = jid;
                                match convert_junction_to_roundabout(
                                    &mut self.world,
                                    jid,
                                    self.roundabout_radius_m,
                                    self.roundabout_lanes,
                                ) {
                                    Ok(sum) => {
                                        self.io_status_msg = format!(
                                            "Converted junction #{jid} -> {}-lane roundabout (R={:.1}m)",
                                            sum.ring_lanes, sum.inscribed_radius_m
                                        );
                                        self.heatmap_dirty = true;
                                    }
                                    Err(e) => {
                                        self.io_status_msg = format!("Roundabout: {e}");
                                    }
                                }
                            }
                        }
                        ViewportTool::ToggleClosure => {
                            if let Some((sid, is_closed, name)) =
                                self.world.toggle_corridor_closure_at(
                                    wx,
                                    wy,
                                    self.stress_cfg.bidirectional_closure,
                                    None,
                                )
                            {
                                let state_str = if is_closed { "CLOSED" } else { "REOPENED" };
                                self.io_status_msg = format!(
                                    "{state_str} `{name}` (#{sid}) -> NFPA <=4m cov {:.1}% (P90 {:.0}s)",
                                    self.world.emergency_coverage.parcels_within_4min_pct,
                                    self.world.emergency_coverage.p90_fire_travel_s
                                );
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                        }
                        ViewportTool::PlaceFacility => {
                            if let Some(fid) = self.world.place_emergency_facility_at(
                                wx,
                                wy,
                                self.active_facility_kind,
                            ) {
                                self.io_status_msg = format!(
                                    "Placed {} #{fid} — NFPA 1710 <=4m coverage: {:.0}% (P90 {:.0}s)",
                                    self.active_facility_kind.label(),
                                    self.world.emergency_coverage.parcels_within_4min_pct,
                                    self.world.emergency_coverage.p90_fire_travel_s
                                );
                                self.layer = HeatmapLayer::EmergencyResponse4Min;
                                self.heatmap_dirty = true;
                            }
                        }
                        ViewportTool::PaintZone => {
                            if let Some(pid) =
                                self.world.paint_parcel_zone_at(wx, wy, self.active_brush)
                            {
                                self.io_status_msg = format!(
                                    "Painted parcel #{pid} -> {}",
                                    self.active_brush.spec().label
                                );
                                self.heatmap_dirty = true;
                            }
                        }
                        ViewportTool::Inspect => {
                            if let Some(jid) = hit_test_junction(&self.world.network, wx, wy, 16.0)
                            {
                                self.selected_junction = jid;
                                if let Some(&sid) = self.world.network.junctions[jid].in_segs.first()
                                {
                                    self.selected_segment = sid;
                                    self.world.watched_segment_id = sid;
                                }
                            } else if let Some((sid, _, _)) =
                                hit_test_segment(&self.world.network, wx, wy, 14.0)
                            {
                                self.selected_segment = sid;
                                self.world.watched_segment_id = sid;
                                self.selected_junction = self.world.network.segments[sid].to_junc;
                            } else if self.tab == DockTab::ZoningAndTransit {
                                if let Some(pid) =
                                    self.world.paint_parcel_zone_at(wx, wy, self.active_brush)
                                {
                                    self.io_status_msg = format!(
                                        "Painted parcel #{pid} -> {}",
                                        self.active_brush.spec().label
                                    );
                                    self.heatmap_dirty = true;
                                }
                            }
                        }
                    }
                }
            }

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 12, 21));

            if self.gpu_ready {
                pack_road_quads_into(&self.world, &mut self.staging.road_quads);
                pack_agents_into(&self.world, interp_dt, &mut self.staging.agents);

                let is_gpu_compute_plume = matches!(
                    self.layer,
                    HeatmapLayer::VspEmissions | HeatmapLayer::AgentDensity
                );

                // Discrete parcel overlays (`Zoning`, `Isochrone15Min`) upload only when edited;
                // `VspEmissions` & `AgentDensity` run 100% on GPU via `plume_compute.wgsl`!
                let heatmap_rgba = if !is_gpu_compute_plume && self.heatmap_dirty {
                    self.heatmap_dirty = false;
                    let agents_snap = self.staging.agents.clone();
                    build_heatmap_rgba(
                        &self.world,
                        self.layer,
                        &agents_snap,
                        &mut self.staging,
                    );
                    Some(self.staging.heatmap_rgba.clone())
                } else {
                    if is_gpu_compute_plume {
                        self.heatmap_dirty = false;
                    }
                    None
                };

                let b = self.world.network.bounds;
                let span_x = (b.max_x - b.min_x).max(10.0) as f32;
                let span_y = (b.max_y - b.min_y).max(10.0) as f32;

                let plume_compute = if is_gpu_compute_plume {
                    let mut mode_bits = self.layer as u32;
                    if self.plume_reset_pending {
                        mode_bits |= 256;
                        self.plume_reset_pending = false;
                    }
                    let n_ag = self.staging.agents.len();
                    let source_gain = if self.layer == HeatmapLayer::VspEmissions {
                        if n_ag > 8_000 { 1.15 } else { 3.40 }
                    } else if n_ag > 10_000 {
                        0.85
                    } else {
                        2.90
                    };
                    Some(GpuPlumeComputeUniform {
                        world_min: [b.min_x as f32, b.min_y as f32],
                        world_span: [span_x, span_y],
                        wind_mps: [self.plume_params.wind_x_mps, self.plume_params.wind_y_mps],
                        dt_s: if self.paused { 0.0 } else { dt_real as f32 },
                        diffusivity: self.plume_params.diffusivity,
                        decay_rate: self.plume_params.decay_rate,
                        source_gain,
                        agent_count: n_ag as u32,
                        mode_and_reset: mode_bits,
                    })
                } else {
                    None
                };

                let camera = GpuCameraUniform {
                    cam_center: [self.cam_x as f32, self.cam_y as f32],
                    viewport_px: [rect.width().max(1.0), rect.height().max(1.0)],
                    world_min: [b.min_x as f32, b.min_y as f32],
                    world_span: [
                        (b.max_x - b.min_x).max(10.0) as f32,
                        (b.max_y - b.min_y).max(10.0) as f32,
                    ],
                    zoom_px_per_m: self.cam_zoom as f32,
                    sim_time_s: (self.world.sim_time_s + interp_dt) as f32,
                    heatmap_opacity: if self.layer == HeatmapLayer::Off {
                        0.0
                    } else {
                        0.82
                    },
                    heatmap_mode: self.layer as u32,
                };

                let payload = ViewportFramePayload {
                    camera,
                    plume_compute,
                    road_quads: self.staging.road_quads.clone(),
                    agents: self.staging.agents.clone(),
                    heatmap_rgba,
                };

                let cb = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    UrbanViewportPaintCallback {
                        payload: Arc::new(Mutex::new(Some(payload))),
                    },
                );
                painter.add(cb);

                let world_to_screen = |wx: f64, wy: f64| -> Pos2 {
                    Pos2::new(
                        rect.center().x + ((wx - self.cam_x) * self.cam_zoom) as f32,
                        rect.center().y + ((wy - self.cam_y) * self.cam_zoom) as f32,
                    )
                };

                // Seismic bridge fragility failure-probability halos when Seismic Monte Carlo mode is active
                if self.tab == DockTab::ResilienceStress
                    && self.stress_sub_mode == StressSubMode::SeismicMonteCarlo
                {
                    if let Some(mcrep) = self.latest_seismic_report.as_ref() {
                        for b in mcrep.bridge_inventory.iter().take(8) {
                            let p0_s = world_to_screen(b.p0.x, b.p0.y);
                            let p1_s = world_to_screen(b.p1.x, b.p1.y);
                            let col = if b.mc_closure_rate_pct >= 35.0 {
                                Color32::from_rgba_unmultiplied(244, 63, 94, 210)
                            } else if b.mc_closure_rate_pct >= 15.0 {
                                Color32::from_rgba_unmultiplied(250, 204, 21, 195)
                            } else {
                                Color32::from_rgba_unmultiplied(52, 211, 153, 185)
                            };
                            painter.line_segment([p0_s, p1_s], Stroke::new(3.4, col));
                            if self.cam_zoom >= 0.30 {
                                let m_scr = world_to_screen(b.midpoint.x, b.midpoint.y);
                                painter.text(
                                    Pos2::new(m_scr.x, m_scr.y - 8.0),
                                    egui::Align2::CENTER_BOTTOM,
                                    format!("P(Cls)={:.0}%", b.mc_closure_rate_pct),
                                    egui::FontId::monospace(10.5),
                                    col,
                                );
                            }
                        }
                    }
                }

                // Critical corridor / bridge vulnerability callouts when Stress-Test mode is active
                if (self.tab == DockTab::ResilienceStress || self.tool == ViewportTool::ToggleClosure)
                    && self.latest_stress_report.is_some()
                {
                    if let Some(srep) = self.latest_stress_report.as_ref() {
                        let pulse = 0.55 + 0.45 * ((self.world.sim_time_s * 5.2).sin() as f32);
                        for v in srep.top_vulnerabilities.iter().take(6) {
                            for &[x0, y0, x1, y1] in &v.segments_endpoints_hint {
                                let p0_s = world_to_screen(x0, y0);
                                let p1_s = world_to_screen(x1, y1);
                                painter.line_segment(
                                    [p0_s, p1_s],
                                    Stroke::new(
                                        3.2,
                                        Color32::from_rgba_unmultiplied(
                                            244,
                                            63,
                                            94,
                                            (130.0 + 105.0 * pulse) as u8,
                                        ),
                                    ),
                                );
                            }
                            if self.cam_zoom >= 0.30 {
                                let m_scr = world_to_screen(v.midpoint.x, v.midpoint.y);
                                painter.text(
                                    Pos2::new(m_scr.x, m_scr.y - 8.0),
                                    egui::Align2::CENTER_BOTTOM,
                                    format!("CRIT#{} (-{:.0}pt)", v.rank, v.coverage_drop_pct_pts),
                                    egui::FontId::monospace(10.5),
                                    Color32::from_rgb(253, 164, 175),
                                );
                            }
                        }
                    }
                }

                // Pulsing gold-emerald rings for MCLP candidate station sites
                if let Some(mrep) = self.latest_mclp_report.as_ref() {
                    let pulse = 0.55 + 0.45 * ((self.world.sim_time_s * 4.5).sin() as f32);
                    for (idx, site) in mrep.selected_sites.iter().enumerate() {
                        let s_scr = world_to_screen(site.world_pos.x, site.world_pos.y);
                        painter.circle_stroke(
                            s_scr,
                            (11.0 * self.cam_zoom as f32).clamp(8.0, 24.0),
                            Stroke::new(
                                2.2,
                                Color32::from_rgba_unmultiplied(
                                    250,
                                    204,
                                    21,
                                    (140.0 + 100.0 * pulse) as u8,
                                ),
                            ),
                        );
                        if site.allocated_engines >= 2 {
                            painter.circle_stroke(
                                s_scr,
                                (15.5 * self.cam_zoom as f32).clamp(11.5, 32.0),
                                Stroke::new(
                                    1.8,
                                    Color32::from_rgba_unmultiplied(56, 189, 248, 210),
                                ),
                            );
                        }
                        if self.cam_zoom >= 0.35 {
                            let tag = if site.allocated_engines >= 2 {
                                format!("MEXCLP#{} [{}x]", idx + 1, site.allocated_engines)
                            } else {
                                format!("MEXCLP#{}", idx + 1)
                            };
                            painter.text(
                                Pos2::new(s_scr.x, s_scr.y + (7.0 * self.cam_zoom as f32).max(9.0)),
                                egui::Align2::CENTER_TOP,
                                tag,
                                egui::FontId::monospace(10.5),
                                Color32::from_rgb(254, 240, 138),
                            );
                        }
                    }
                }

                // Carter-Chaiken-Ignall station offset badges & Peak/Off-Peak switched-beat outlines when ResponseDistricts is active
                if self.layer == HeatmapLayer::ResponseDistricts {
                    if let Some(drep) = self.latest_diurnal_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[CAD DISPATCH TABLE: {} | {:02}:00 | {} Peak/Off-Peak Beat Switches]",
                                self.active_diurnal_regime.short_tag(),
                                (self.world.diurnal_clock_hour.floor() as usize) % 24,
                                drep.primary_beat_switches_count
                            ),
                            egui::FontId::monospace(11.5),
                            Color32::from_rgb(167, 243, 208),
                        );
                        if self.cam_zoom >= 0.36 {
                            for (j, p) in self.world.parcels.iter().enumerate() {
                                if let Some(rc) = drep.parcel_run_cards.get(j) {
                                    if rc.primary_beat_switched {
                                        let p0 = world_to_screen(p.min.x, p.min.y);
                                        let p1 = world_to_screen(p.max.x, p.max.y);
                                        painter.rect_stroke(
                                            Rect::from_two_pos(p0, p1).shrink(2.0),
                                            2.0,
                                            Stroke::new(
                                                1.8,
                                                Color32::from_rgba_unmultiplied(232, 121, 249, 215),
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if let Some(rrep) = self.latest_redistricting_report.as_ref() {
                        if self.cam_zoom >= 0.36 {
                            for (j, p) in self.world.parcels.iter().enumerate() {
                                if j < rrep.voronoi_parcel_units.len()
                                    && j < rrep.optimized_parcel_units.len()
                                    && rrep.voronoi_parcel_units[j] != rrep.optimized_parcel_units[j]
                                {
                                    let p0 = world_to_screen(p.min.x, p.min.y);
                                    let p1 = world_to_screen(p.max.x, p.max.y);
                                    let r_scr = Rect::from_two_pos(p0, p1);
                                    painter.rect_stroke(
                                        r_scr,
                                        2.0,
                                        Stroke::new(
                                            1.4,
                                            Color32::from_rgba_unmultiplied(250, 204, 21, 185),
                                        ),
                                    );
                                }
                            }
                            for us in &rrep.unit_summaries {
                                let u_scr = world_to_screen(us.world_pos.x, us.world_pos.y);
                                painter.text(
                                    Pos2::new(
                                        u_scr.x,
                                        u_scr.y + (16.0 * self.cam_zoom as f32).max(18.0),
                                    ),
                                    egui::Align2::CENTER_TOP,
                                    format!(
                                        "s={:+.1}s | rho={:.1}%",
                                        us.boundary_offset_s,
                                        us.optimized_busy_rho * 100.0
                                    ),
                                    egui::FontId::monospace(9.8),
                                    Color32::from_rgb(254, 249, 195),
                                );
                            }
                        }
                    }
                }

                // FEMA [GSB], USACE [249th-GEN] Prime Power, [FEMA-TDMS] Sort Yards & [FEMA-DRC]/[FEMA-MRIC] Overlay
                if self.layer == HeatmapLayer::FemaGsbDrcTdms {
                    if let Some(rrep) = self.latest_fema_rec_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[FEMA 249th GSB, TDMS & DRC/MRIC ({}) | 72hPwr={:.0}% ({:.2}MW) | TDMS Saved={:.0} yd3 | IA Reg={:.1}%]",
                                rrep.config.policy.short_label(),
                                rrep.energized_lifelines_72h_pct,
                                rrep.deployed_prime_power_mw,
                                rrep.tdms_landfill_volume_saved_yd3,
                                rrep.ia_registered_14d_pct
                            ),
                            egui::FontId::monospace(10.8),
                            Color32::from_rgb(167, 243, 208),
                        );

                        for &(p0, p1) in &rrep.gsb_dispatch_arcs {
                            painter.line_segment(
                                [world_to_screen(p0.x, p0.y), world_to_screen(p1.x, p1.y)],
                                Stroke::new(
                                    2.0,
                                    Color32::from_rgba_unmultiplied(250, 204, 21, 190),
                                ),
                            );
                        }
                        for &(p0, p1) in &rrep.tdms_haul_arcs {
                            painter.line_segment(
                                [world_to_screen(p0.x, p0.y), world_to_screen(p1.x, p1.y)],
                                Stroke::new(
                                    1.3,
                                    Color32::from_rgba_unmultiplied(56, 189, 248, 120),
                                ),
                            );
                        }
                        for fr in &rrep.facility_rows {
                            let scr = world_to_screen(fr.world_pos.x, fr.world_pos.y);
                            let fcol = match fr.kind {
                                FemaRecoverySiteKind::GeneratorStagingBase => {
                                    Color32::from_rgb(250, 204, 21)
                                }
                                FemaRecoverySiteKind::CriticalLifelineGenTarget => {
                                    Color32::from_rgb(52, 211, 153)
                                }
                                FemaRecoverySiteKind::TemporaryDebrisSiteTdms => {
                                    Color32::from_rgb(56, 189, 248)
                                }
                                FemaRecoverySiteKind::FixedDisasterRecoveryCenterDrc => {
                                    Color32::from_rgb(244, 114, 182)
                                }
                                FemaRecoverySiteKind::MobileRegistrationIntakeMric => {
                                    Color32::from_rgb(167, 243, 208)
                                }
                            };
                            painter.circle_stroke(scr, 11.5, Stroke::new(2.3, fcol));
                            painter.text(
                                Pos2::new(scr.x, scr.y - 13.5),
                                egui::Align2::CENTER_BOTTOM,
                                format!("[{}]", fr.site_code),
                                egui::FontId::monospace(8.6),
                                fcol,
                            );
                        }
                    }
                }

                // Post-Wildfire River Intake Ash/Turbidity Shutoff, WTP Coagulant & Clearwell/Tank Drawdown Overlay
                if self.layer == HeatmapLayer::WtpTurbidityStorage {
                    if let Some(wrep) = self.latest_wtp_turb_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[WTP ASH/TURBIDITY & STORAGE ({} | Peak={:.0} NTU | Shutoff={:.1}h) | MinStor={:.1}% | HighZone={:.1} psi | BWA={:.0}%]",
                                wrep.config.policy.short_label(),
                                wrep.peak_raw_river_ntu,
                                wrep.intake_shutoff_duration_hrs,
                                wrep.min_finished_storage_fill_pct,
                                wrep.min_high_zone_pressure_psi,
                                wrep.boil_water_advisory_prob_pct
                            ),
                            egui::FontId::monospace(10.5),
                            Color32::from_rgb(165, 243, 252),
                        );

                        for &(p0, p1) in &wrep.raw_to_wtp_to_tank_arcs {
                            painter.line_segment(
                                [world_to_screen(p0.x, p0.y), world_to_screen(p1.x, p1.y)],
                                Stroke::new(
                                    2.1,
                                    Color32::from_rgba_unmultiplied(56, 189, 248, 185),
                                ),
                            );
                        }

                        for fr in &wrep.facility_rows {
                            let scr = world_to_screen(fr.world_pos.x, fr.world_pos.y);
                            let fcol = if fr.bwa_triggered {
                                Color32::from_rgb(244, 63, 94)
                            } else {
                                match fr.kind {
                                    WaterSupplyAssetKind::SurfaceWaterRiverIntake => {
                                        Color32::from_rgb(251, 146, 60)
                                    }
                                    WaterSupplyAssetKind::PresedimentationAndWtpPlant => {
                                        Color32::from_rgb(56, 189, 248)
                                    }
                                    WaterSupplyAssetKind::EmergencyGroundwaterWellfield => {
                                        Color32::from_rgb(52, 211, 153)
                                    }
                                    WaterSupplyAssetKind::ClearwellAndPressureZoneTank => {
                                        Color32::from_rgb(250, 204, 21)
                                    }
                                }
                            };
                            painter.circle_stroke(scr, 12.0, Stroke::new(2.5, fcol));
                            let bwa_tag = if fr.bwa_triggered { " BWA!" } else { "" };
                            painter.text(
                                Pos2::new(scr.x, scr.y - 14.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!(
                                    "[{}{} {:.0}%]",
                                    fr.kind.badge(),
                                    bwa_tag,
                                    fr.min_level_or_uptime_pct
                                ),
                                egui::FontId::monospace(8.6),
                                fcol,
                            );
                        }
                    }
                }

                // Post-Wildfire Burn-Scar Hydrophobic Runoff, USGS M1 Debris-Flow Runout Channels & FHWA HDS-5 Culvert Washouts
                if self.layer == HeatmapLayer::BurnScarDebrisFlow {
                    if let Some(drep) = self.latest_debris_flow_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[BURN-SCAR & USGS DEBRIS FLOW ({} | I15={:.0} mm/h vs P50={:.1} mm/h) | Sed={:.0} m3 | Washouts={:.1}/{} clv]",
                                drep.config.baer_policy.short_label(),
                                drep.config.peak_i15_intensity_mm_hr,
                                drep.firewide_median_i15_p50_mm_hr,
                                drep.total_predicted_sediment_volume_m3,
                                drep.expected_washed_out_culverts,
                                drep.evaluated_culverts_count
                            ),
                            egui::FontId::monospace(10.6),
                            Color32::from_rgb(253, 224, 71),
                        );

                        // USGS Basin Headwater -> Alluvial-Fan Outlet Debris-Flow Channels
                        for br in &drep.basin_rows {
                            let p0 = world_to_screen(br.headwater_pos.x, br.headwater_pos.y);
                            let p1 = world_to_screen(br.outlet_pos.x, br.outlet_pos.y);
                            let bcol = match br.combined_hazard {
                                UsgsCombinedHazardRating::Extreme => Color32::from_rgb(244, 63, 94),
                                UsgsCombinedHazardRating::High => Color32::from_rgb(251, 146, 60),
                                UsgsCombinedHazardRating::Moderate => {
                                    Color32::from_rgb(250, 204, 21)
                                }
                                UsgsCombinedHazardRating::Low => Color32::from_rgb(52, 211, 153),
                            };
                            painter.line_segment([p0, p1], Stroke::new(2.7, bcol));
                            painter.circle_stroke(p0, 9.5, Stroke::new(2.0, bcol));
                            painter.text(
                                Pos2::new(p0.x, p0.y - 11.5),
                                egui::Align2::CENTER_BOTTOM,
                                format!(
                                    "[{} P={:.0}% | {:.0}m3]",
                                    br.basin_code, br.m1_likelihood_pct, br.gartner_volume_m3
                                ),
                                egui::FontId::monospace(8.4),
                                bcol,
                            );
                        }

                        // FHWA HDS-5 Stormwater Culvert Crossing Badges (`HW/D` & Plugging)
                        for cr in &drep.culvert_rows {
                            let c_scr = world_to_screen(cr.world_pos.x, cr.world_pos.y);
                            let ccol = match cr.hydraulic_state {
                                CulvertHydraulicState::EmbankmentWashoutClosed => {
                                    Color32::from_rgb(244, 63, 94)
                                }
                                CulvertHydraulicState::OvertoppingFlooded => {
                                    Color32::from_rgb(251, 146, 60)
                                }
                                CulvertHydraulicState::SurchargedInlet => {
                                    Color32::from_rgb(250, 204, 21)
                                }
                                CulvertHydraulicState::PassableClear => {
                                    Color32::from_rgb(52, 211, 153)
                                }
                            };
                            painter.circle_stroke(c_scr, 11.5, Stroke::new(2.5, ccol));
                            painter.text(
                                Pos2::new(c_scr.x, c_scr.y - 13.5),
                                egui::Align2::CENTER_BOTTOM,
                                format!(
                                    "[{} HW/D={:.2}]",
                                    cr.hydraulic_state.badge(),
                                    cr.peak_hw_over_d
                                ),
                                egui::FontId::monospace(8.6),
                                ccol,
                            );
                        }
                    }
                }

                // ALERTWest PTZ Camera Azimuth Triangulation Rays, CEP90 Ignition Fix & VLAT Phos-Chek Retardant Lines
                if self.layer == HeatmapLayer::WildfireAlertWestResponse {
                    if let Some(wrep) = self.latest_wildfire_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[ALERTWEST WILDFIRE ({} | {} LOS Cams | CEP90={:.0}m | Det={:.1}m) | {:.1} ac | Flame={:.1}ft | Cont={:.0}%]",
                                wrep.config.posture.short_label(),
                                wrep.cameras_with_los_count,
                                wrep.triangulation_cep90_m,
                                wrep.mean_detection_to_dispatch_min,
                                wrep.final_burned_area_acres,
                                wrep.fire_behavior.flame_length_ft,
                                wrep.final_containment_pct
                            ),
                            egui::FontId::monospace(10.8),
                            Color32::from_rgb(253, 186, 116),
                        );

                        // Cross-azimuth camera smoke bearing lines
                        for &(c0, c1, conf) in &wrep.camera_bearing_rays {
                            let alpha = ((conf * 210.0) as u8).clamp(95, 225);
                            painter.line_segment(
                                [world_to_screen(c0.x, c0.y), world_to_screen(c1.x, c1.y)],
                                Stroke::new(
                                    1.9,
                                    Color32::from_rgba_unmultiplied(56, 189, 248, alpha),
                                ),
                            );
                        }

                        // VLAT/LAT Phos-Chek Long-Term Retardant Drop Lines (Crimson-Magenta)
                        for &(r0, r1) in &wrep.retardant_drop_segments {
                            painter.line_segment(
                                [world_to_screen(r0.x, r0.y), world_to_screen(r1.x, r1.y)],
                                Stroke::new(4.2, Color32::from_rgb(244, 63, 94)),
                            );
                        }

                        // Type-1 Hotshot & Wildland Dozer Flanking Lines (Emerald-Gold)
                        for &(h0, h1) in &wrep.handcrew_dozer_segments {
                            painter.line_segment(
                                [world_to_screen(h0.x, h0.y), world_to_screen(h1.x, h1.y)],
                                Stroke::new(3.0, Color32::from_rgb(52, 211, 153)),
                            );
                        }

                        // Triangulated Ignition Fix + CEP_90 Uncertainty Circle
                        let ign_scr = world_to_screen(
                            wrep.triangulated_ignition_pos.x,
                            wrep.triangulated_ignition_pos.y,
                        );
                        let cep_px = ((wrep.triangulation_cep90_m * self.cam_zoom) as f32)
                            .clamp(9.0, 95.0);
                        painter.circle_stroke(
                            ign_scr,
                            cep_px,
                            Stroke::new(2.2, Color32::from_rgb(250, 204, 21)),
                        );
                        painter.text(
                            Pos2::new(ign_scr.x, ign_scr.y - cep_px - 3.0),
                            egui::Align2::CENTER_BOTTOM,
                            format!("WLS FIX (CEP90={:.0}m)", wrep.triangulation_cep90_m),
                            egui::FontId::monospace(9.0),
                            Color32::from_rgb(254, 240, 138),
                        );

                        // ALERTWest & Starlink MDU Camera Badges
                        for cam in &wrep.camera_sites {
                            let c_scr = world_to_screen(cam.world_pos.x, cam.world_pos.y);
                            let ccol = match cam.kind {
                                WildfireCameraKind::MountaintopFixedPtz => {
                                    Color32::from_rgb(56, 189, 248)
                                }
                                WildfireCameraKind::WuiLookoutTower => {
                                    Color32::from_rgb(251, 146, 60)
                                }
                                WildfireCameraKind::StarlinkMobileDeploymentUnit => {
                                    Color32::from_rgb(52, 211, 153)
                                }
                            };
                            painter.circle_stroke(c_scr, 11.5, Stroke::new(2.4, ccol));
                            let shk = if cam.co_located_pnsn_shakealert { "+SA" } else { "" };
                            painter.text(
                                Pos2::new(c_scr.x, c_scr.y - 13.5),
                                egui::Align2::CENTER_BOTTOM,
                                format!("[{}{}] {}", cam.kind.badge(), shk, cam.camera_id),
                                egui::FontId::monospace(8.7),
                                ccol,
                            );
                        }
                    }
                }

                // FEMA [US&R-BoO], USNG Structural Collapse Grids ([!]/[/]/[X]), ESF-3 Debris Push & ISB->POD Reopening
                if self.layer == HeatmapLayer::FemaUsarDebris {
                    if let Some(urep) = self.latest_fema_usar_report.as_ref() {
                        let snap_h = self.fema_usar_cfg.live_injection_snapshot_hour;
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[FEMA TYPE-1 US&R & ESF-3 DEBRIS ({} | Snap H+{:.0}h) | Saved={:.1} ({:.1}%) | POD Reopen={:.1}h]",
                                urep.config.policy.short_label(),
                                snap_h,
                                urep.expected_live_rescues,
                                urep.entrapped_survival_rate_pct,
                                urep.time_to_reopen_all_pods_hrs
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(254, 240, 138),
                        );

                        // Highlight debris-blocked vs ESF-3 bulldozer-reopened road segments at `snap_h`
                        for (sid, &reopen_h) in urep.segment_reopen_hour.iter().enumerate() {
                            if sid >= self.world.network.segments.len()
                                || urep.segment_debris_tons[sid] < 18.0
                            {
                                continue;
                            }
                            let pts: Vec<Pos2> = self.world.network.segments[sid]
                                .base_pts
                                .iter()
                                .map(|p| world_to_screen(p.x, p.y))
                                .collect();
                            if pts.len() >= 2 {
                                let scol = if reopen_h <= snap_h + 1e-6 {
                                    Color32::from_rgba_unmultiplied(52, 211, 153, 200)
                                } else {
                                    Color32::from_rgba_unmultiplied(244, 63, 94, 215)
                                };
                                painter.add(egui::Shape::line(pts, Stroke::new(2.8, scol)));
                            }
                        }

                        // Draw 3x3 USNG Search Grid Bounding Boxes & FEMA INSARAG `[X]` / `[/]` / `[!]` Badges
                        for gr in &urep.grid_rows {
                            let p_min = world_to_screen(
                                gr.center.x - gr.half_extent_m,
                                gr.center.y - gr.half_extent_m,
                            );
                            let p_max = world_to_screen(
                                gr.center.x + gr.half_extent_m,
                                gr.center.y + gr.half_extent_m,
                            );
                            let rect = egui::Rect::from_two_pos(p_min, p_max);
                            let gcol = match gr.fema_marking {
                                FemaInsaragMarking::CompletedCrossX => {
                                    Color32::from_rgba_unmultiplied(52, 211, 153, 145)
                                }
                                FemaInsaragMarking::InProgressSlash => {
                                    Color32::from_rgba_unmultiplied(250, 204, 21, 160)
                                }
                                FemaInsaragMarking::BlockedUnreached => {
                                    Color32::from_rgba_unmultiplied(244, 63, 94, 175)
                                }
                            };
                            painter.rect_stroke(
                                rect,
                                2.0,
                                Stroke::new(1.3, gcol),
                            );
                            let c_scr = world_to_screen(gr.center.x, gr.center.y);
                            painter.text(
                                c_scr,
                                egui::Align2::CENTER_CENTER,
                                format!(
                                    "{} {}\n{:.1} saved",
                                    gr.fema_marking.symbol(),
                                    gr.usng_cell_code,
                                    gr.live_rescued_victims
                                ),
                                egui::FontId::monospace(8.9),
                                gcol,
                            );
                        }

                        // US&R Base of Operations [US&R-BoO] & Reopened PODs
                        let boo_scr = world_to_screen(urep.boo_world_pos.x, urep.boo_world_pos.y);
                        painter.circle_stroke(
                            boo_scr,
                            14.0,
                            Stroke::new(2.7, Color32::from_rgb(56, 189, 248)),
                        );
                        painter.text(
                            Pos2::new(boo_scr.x, boo_scr.y - 16.0),
                            egui::Align2::CENTER_BOTTOM,
                            "[US&R-BoO]",
                            egui::FontId::monospace(9.8),
                            Color32::from_rgb(186, 230, 253),
                        );

                        for (idx, ppos) in urep.pod_world_positions.iter().enumerate() {
                            let p_scr = world_to_screen(ppos.x, ppos.y);
                            painter.circle_stroke(
                                p_scr,
                                10.0,
                                Stroke::new(2.1, Color32::from_rgb(250, 204, 21)),
                            );
                            painter.text(
                                Pos2::new(p_scr.x, p_scr.y + 12.0),
                                egui::Align2::CENTER_TOP,
                                format!("[POD-{}]", idx + 1),
                                egui::FontId::monospace(8.6),
                                Color32::from_rgb(253, 224, 71),
                            );
                        }
                    }
                }

                // FEMA [ISB], [CSA], USACE Type I-III / Walk-Up [POD] Queue Spillback & IPAWS Routing Vectors
                if self.layer == HeatmapLayer::FemaIsbPods {
                    if let Some(prep) = self.latest_fema_pods_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[FEMA ISB & C-PODS ({} | {}) | Served={:.1}% | Mean Wait={:.1}m | Spillback={:.0} veh]",
                                prep.config.routing_policy.short_label(),
                                prep.config.isb_policy.short_label(),
                                prep.population_demand_met_pct,
                                prep.mean_pod_queue_wait_min,
                                prep.peak_arterial_spillback_vehs
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(167, 243, 208),
                        );

                        // Household -> IPAWS-assigned C-POD demand vectors
                        for &(p0, p1) in &prep.household_pod_arcs {
                            let s0 = world_to_screen(p0.x, p0.y);
                            let s1 = world_to_screen(p1.x, p1.y);
                            painter.line_segment(
                                [s0, s1],
                                Stroke::new(
                                    1.2,
                                    Color32::from_rgba_unmultiplied(56, 189, 248, 115),
                                ),
                            );
                        }

                        // ISB -> CSA -> C-POD 53-ft tractor-trailer echelon supply corridors
                        for &(p0, p1) in &prep.isb_convoy_arcs {
                            let s0 = world_to_screen(p0.x, p0.y);
                            let s1 = world_to_screen(p1.x, p1.y);
                            painter.line_segment(
                                [s0, s1],
                                Stroke::new(
                                    2.3,
                                    Color32::from_rgba_unmultiplied(250, 204, 21, 205),
                                ),
                            );
                        }

                        let isb_scr = world_to_screen(prep.isb_world_pos.x, prep.isb_world_pos.y);
                        painter.circle_stroke(
                            isb_scr,
                            15.0,
                            Stroke::new(2.8, Color32::from_rgb(250, 204, 21)),
                        );
                        painter.text(
                            Pos2::new(isb_scr.x, isb_scr.y - 17.0),
                            egui::Align2::CENTER_BOTTOM,
                            "[FEMA-ISB]",
                            egui::FontId::monospace(10.0),
                            Color32::from_rgb(254, 240, 138),
                        );

                        let csa_scr = world_to_screen(prep.csa_world_pos.x, prep.csa_world_pos.y);
                        painter.circle_stroke(
                            csa_scr,
                            12.0,
                            Stroke::new(2.3, Color32::from_rgb(56, 189, 248)),
                        );
                        painter.text(
                            Pos2::new(csa_scr.x, csa_scr.y - 14.0),
                            egui::Align2::CENTER_BOTTOM,
                            "[COUNTY-CSA]",
                            egui::FontId::monospace(9.2),
                            Color32::from_rgb(186, 230, 253),
                        );

                        for pr in &prep.pod_rows {
                            let scr = world_to_screen(pr.world_pos.x, pr.world_pos.y);
                            let pcol = if pr.mean_queue_wait_min <= 18.0 && pr.stockout_dry_hours < 0.5 {
                                Color32::from_rgb(52, 211, 153)
                            } else if pr.mean_queue_wait_min <= 42.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            if pr.peak_spillback_vehs > 5.0 {
                                painter.circle_stroke(
                                    scr,
                                    (14.0 + (pr.peak_spillback_vehs as f32) * 0.22).clamp(15.0, 42.0),
                                    Stroke::new(
                                        1.8,
                                        Color32::from_rgba_unmultiplied(244, 63, 94, 175),
                                    ),
                                );
                            }
                            let _ = FemaPodClassification::TypeIMetroFourLane;
                            painter.circle_stroke(scr, 12.0, Stroke::new(2.5, pcol));
                            painter.text(
                                Pos2::new(scr.x, scr.y - 14.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!(
                                    "[{} {:.0}m]",
                                    pr.pod_class.badge(),
                                    pr.mean_queue_wait_min
                                ),
                                egui::FontId::monospace(9.2),
                                pcol,
                            );
                        }
                    }
                }

                // FEMA [COG-EOC CAP], [NPWS-PEP AM/FM], [NWR-SAME 162MHz] & Region 10 [MERS-IRV] ESF-2 Overlay
                if self.layer == HeatmapLayer::FemaCommsArchitecture {
                    if let Some(frep) = self.latest_fema_comms_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[FEMA IPAWS & ESF-2 ({} | {}) | CAP={:.0}% | IPAWS Reach={:.1}% | NIFOG LMR={:.1}%]",
                                frep.config.eoc_uplink.short_label(),
                                frep.config.posture.short_label(),
                                frep.eoc_cap_origination_uptime_pct,
                                frep.public_ipaws_verified_reach_pct,
                                frep.esf2_nifog_responder_lmr_coverage_pct
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(165, 243, 252),
                        );

                        // EOC <-> FEMA MERS IRV Satellite/HF Tactical Uplink Vectors
                        for &(p0, p1) in &frep.mers_uplink_vectors {
                            let s0 = world_to_screen(p0.x, p0.y);
                            let s1 = world_to_screen(p1.x, p1.y);
                            painter.line_segment(
                                [s0, s1],
                                Stroke::new(
                                    2.2,
                                    Color32::from_rgba_unmultiplied(52, 211, 153, 210),
                                ),
                            );
                        }

                        for ar in &frep.asset_rows {
                            let scr = world_to_screen(ar.world_pos.x, ar.world_pos.y);
                            let acol = match ar.kind {
                                FemaAssetKind::CogEocCapOriginator => {
                                    Color32::from_rgb(250, 204, 21)
                                }
                                FemaAssetKind::NpwsPepBroadcastStation => {
                                    Color32::from_rgb(56, 189, 248)
                                }
                                FemaAssetKind::NoaaNwrSameTransmitter => {
                                    Color32::from_rgb(168, 85, 247)
                                }
                                FemaAssetKind::FemaMersIrvUnit => Color32::from_rgb(52, 211, 153),
                                FemaAssetKind::CommercialWeaCellSite => {
                                    if ar.post_quake_online_pct >= 65.0 {
                                        Color32::from_rgb(45, 212, 191)
                                    } else {
                                        Color32::from_rgb(244, 63, 94)
                                    }
                                }
                            };
                            if ar.kind == FemaAssetKind::FemaMersIrvUnit {
                                let edge = world_to_screen(
                                    ar.world_pos.x + ar.coverage_radius_m,
                                    ar.world_pos.y,
                                );
                                let r_px = (edge.x - scr.x).abs().clamp(18.0, 220.0);
                                painter.circle_stroke(
                                    scr,
                                    r_px,
                                    Stroke::new(
                                        1.6,
                                        Color32::from_rgba_unmultiplied(52, 211, 153, 145),
                                    ),
                                );
                            }
                            painter.circle_stroke(scr, 12.0, Stroke::new(2.5, acol));
                            painter.text(
                                Pos2::new(scr.x, scr.y - 14.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("[{} {:.0}%]", ar.kind.badge(), ar.post_quake_online_pct),
                                egui::FontId::monospace(9.2),
                                acol,
                            );
                        }
                    }
                }

                // Post-Earthquake Cellular [BTS] Blackout Sectors, [SIREN] 70-dBC Contours & [HI-LO/D2D] Canvass Sweeps
                if self.layer == HeatmapLayer::WarningMobilization {
                    if let Some(wrep) = self.latest_warning_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[CELL BLACKOUT & WEA/SIREN MOBILIZATION ({} | {}) | BTS Dark={:.1}% | Warn={:.1}m | Trapped={:.0}]",
                                wrep.config.telecom_backup.short_label(),
                                wrep.config.channel_mix.short_label(),
                                wrep.mean_cell_towers_blackout_pct,
                                wrep.mean_warning_receipt_min,
                                wrep.late_departure_fire_trapped_pax
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(186, 230, 253),
                        );

                        // Outdoor 124-dBC Acoustic Warning Sirens (62 dBC indoor / 70 dBC outdoor rings)
                        if wrep.config.channel_mix != AlertChannelMix::WeaAndReverse911Only {
                            for sr in &wrep.siren_rows {
                                let s_scr = world_to_screen(sr.world_pos.x, sr.world_pos.y);
                                let edge_scr = world_to_screen(
                                    sr.world_pos.x + sr.radius_62dbc_indoor_wake_m,
                                    sr.world_pos.y,
                                );
                                let r_px = (edge_scr.x - s_scr.x).abs().clamp(14.0, 220.0);
                                painter.circle_stroke(
                                    s_scr,
                                    r_px,
                                    Stroke::new(
                                        1.5,
                                        Color32::from_rgba_unmultiplied(250, 204, 21, 145),
                                    ),
                                );
                                painter.text(
                                    Pos2::new(s_scr.x, s_scr.y + 12.0),
                                    egui::Align2::CENTER_TOP,
                                    format!("[SIREN-{}]", sr.siren_id + 1),
                                    egui::FontId::monospace(8.8),
                                    Color32::from_rgb(253, 224, 71),
                                );
                            }
                        }

                        // Hi-Lo Patrol Cruiser & Door-to-Door Canvass Priority Vectors
                        for &(p0, p1) in &wrep.canvass_sweep_paths {
                            let s0 = world_to_screen(p0.x, p0.y);
                            let s1 = world_to_screen(p1.x, p1.y);
                            painter.line_segment(
                                [s0, s1],
                                Stroke::new(
                                    2.1,
                                    Color32::from_rgba_unmultiplied(52, 211, 153, 200),
                                ),
                            );
                        }

                        // Cellular Base Transceiver Stations [BTS]
                        for tr in &wrep.tower_rows {
                            let t_scr = world_to_screen(tr.world_pos.x, tr.world_pos.y);
                            let tcol = if tr.wea_cell_online_pct >= 75.0 {
                                Color32::from_rgb(56, 189, 248)
                            } else if tr.wea_cell_online_pct >= 40.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            painter.circle_stroke(t_scr, 12.0, Stroke::new(2.5, tcol));
                            painter.text(
                                Pos2::new(t_scr.x, t_scr.y - 14.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("[BTS{} {:.0}%]", tr.tower_id + 1, tr.wea_cell_online_pct),
                                egui::FontId::monospace(9.2),
                                tcol,
                            );
                        }
                    }
                }

                // GTFS Transit-Bus & ADA Paratransit Evacuation [GTFS-PP] pickup depots, Bus+EMS contraflow & shuttle loops
                if self.layer == HeatmapLayer::TransitBusEvacuation {
                    if let Some(trep) = self.latest_transit_evac_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[GTFS BUS+EMS EVAC ({} | {}) | Clean-Air={:.1}% | Cycle={:.1}m | EMS Spine={:.0}%]",
                                trep.config.lane_mode.short_label(),
                                trep.config.dispatch_strategy.short_label(),
                                trep.combined_clean_air_sheltered_pct,
                                trep.mean_bus_shuttle_cycle_min,
                                trep.ems_inbound_access_preserved_pct
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(153, 246, 228),
                        );

                        // Highlight Dedicated Bus+EMS Priority Contraflow arterial segments
                        for cs in &trep.contraflow_segments {
                            if !cs.bus_ems_priority_lane || cs.segment_id >= self.world.network.segments.len() {
                                continue;
                            }
                            let pts: Vec<Pos2> = self.world.network.segments[cs.segment_id]
                                .base_pts
                                .iter()
                                .map(|p| world_to_screen(p.x, p.y))
                                .collect();
                            if pts.len() >= 2 {
                                let lcol = if cs.contraflow_added_lanes > 0 {
                                    Color32::from_rgba_unmultiplied(250, 204, 21, 185)
                                } else {
                                    Color32::from_rgba_unmultiplied(45, 212, 191, 150)
                                };
                                painter.add(egui::Shape::line(pts, Stroke::new(2.6, lcol)));
                            }
                        }

                        // Shuttle tour vectors: Teal-Gold = GTFS Bus Shuttle, Magenta = ADA Wheelchair-Lift Van
                        for tr in &trep.representative_shuttle_tours {
                            let p0 = world_to_screen(tr.origin_pos.x, tr.origin_pos.y);
                            let p1 = world_to_screen(tr.shelter_pos.x, tr.shelter_pos.y);
                            let scol = if tr.is_ada_paratransit {
                                Color32::from_rgba_unmultiplied(232, 121, 249, 205)
                            } else {
                                Color32::from_rgba_unmultiplied(52, 211, 153, 205)
                            };
                            painter.line_segment([p0, p1], Stroke::new(2.0, scol));
                            painter.circle_stroke(
                                p1,
                                13.0,
                                Stroke::new(2.3, Color32::from_rgb(56, 189, 248)),
                            );
                        }

                        // Neighborhood GTFS Pickup Depots [GTFS-PP] + 400m catchment ring
                        for dr in &trep.depot_rows {
                            let d_scr = world_to_screen(dr.world_pos.x, dr.world_pos.y);
                            let clr_ratio =
                                (dr.evacuated_clean_air_pax / dr.carless_demand_pax.max(1.0))
                                    .clamp(0.0, 1.0);
                            let dcol = if clr_ratio >= 0.85 {
                                Color32::from_rgb(45, 212, 191)
                            } else if clr_ratio >= 0.55 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            painter.circle_stroke(d_scr, 11.0, Stroke::new(2.4, dcol));
                            painter.text(
                                Pos2::new(d_scr.x, d_scr.y - 13.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("[GTFS-PP{} {:.0}%]", dr.depot_id + 1, clr_ratio * 100.0),
                                egui::FontId::monospace(9.1),
                                dcol,
                            );
                        }
                    }
                }

                // Multi-Day Commodity Resupply [RSA-DEPOT], Clarke-Wright convoy tours & site sustainment gauges
                if self.layer == HeatmapLayer::CommodityResupply {
                    if let Some(rrep) = self.latest_resupply_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[MULTI-DAY RESUPPLY ({} | {}d) | Security={:.1}% | CW Tour Sav={:.1}% | Depriv={:.0}]",
                                rrep.config.dispatch_policy.short_label(),
                                rrep.config.horizon_days,
                                rrep.system_supply_security_pct,
                                rrep.mean_convoy_tour_savings_pct,
                                rrep.total_deprivation_social_cost
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(186, 230, 253),
                        );

                        // Multi-stop Clarke-Wright 2-Opt convoy tour polylines
                        for tour in &rrep.representative_day1_tours {
                            let t_col = match tour.commodity {
                                ReliefCommodityKind::PotableWaterLiters => {
                                    Color32::from_rgba_unmultiplied(56, 189, 248, 195)
                                }
                                ReliefCommodityKind::GeneratorDieselLiters => {
                                    Color32::from_rgba_unmultiplied(250, 204, 21, 195)
                                }
                                ReliefCommodityKind::MreRationsMeals => {
                                    Color32::from_rgba_unmultiplied(52, 211, 153, 180)
                                }
                                ReliefCommodityKind::MedicalOxygenKiloLiters => {
                                    Color32::from_rgba_unmultiplied(232, 121, 249, 205)
                                }
                            };
                            let pts: Vec<Pos2> = tour
                                .stop_positions
                                .iter()
                                .map(|p| world_to_screen(p.x, p.y))
                                .collect();
                            if pts.len() >= 2 {
                                painter.add(egui::Shape::line(
                                    pts,
                                    Stroke::new(
                                        if tour.via_helicopter_airdrop { 2.4 } else { 1.85 },
                                        t_col,
                                    ),
                                ));
                            }
                        }

                        // RSA Staging Depot marker
                        let dep_scr =
                            world_to_screen(rrep.depot_world_pos.x, rrep.depot_world_pos.y);
                        painter.circle_stroke(
                            dep_scr,
                            15.0,
                            Stroke::new(2.8, Color32::from_rgb(250, 204, 21)),
                        );
                        painter.text(
                            Pos2::new(dep_scr.x, dep_scr.y - 17.0),
                            egui::Align2::CENTER_BOTTOM,
                            "[RSA-DEPOT]",
                            egui::FontId::monospace(10.0),
                            Color32::from_rgb(254, 240, 138),
                        );

                        // Sustainment sites (Shelters & Trauma Hospitals)
                        for sr in &rrep.site_rows {
                            let s_scr = world_to_screen(sr.world_pos.x, sr.world_pos.y);
                            let scol = if sr.service_security_pct >= 92.0 {
                                Color32::from_rgb(52, 211, 153)
                            } else if sr.service_security_pct >= 75.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            painter.circle_stroke(s_scr, 12.0, Stroke::new(2.3, scol));
                            painter.text(
                                Pos2::new(s_scr.x, s_scr.y - 14.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("[{} {:.0}%]", sr.site_kind.badge(), sr.service_security_pct),
                                egui::FontId::monospace(9.2),
                                scol,
                            );
                        }
                    }
                }

                // Emergency Shelters [SH] habitability rings, HEPA indoor PM2.5 badges & evacuee routing vectors
                if self.layer == HeatmapLayer::EmergencyShelters {
                    if let Some(sh_rep) = self.latest_shelters_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[CLEAN-AIR SHELTERS ({} + {}) | CleanAir={:.1}% | InPM2.5={:.1} ug/m3 | Unmet={:.0}]",
                                sh_rep.config.siting_routing_policy.short_label(),
                                sh_rep.config.filtration_standard.short_label(),
                                sh_rep.sheltered_clean_air_pct,
                                sh_rep.mean_assigned_indoor_pm25_ug_m3,
                                sh_rep.expected_overcrowded_or_unmet_evacuees
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(167, 243, 208),
                        );

                        if self.cam_zoom >= 0.30 {
                            for pr in &sh_rep.parcel_rows {
                                if pr.displaced_evacuees >= 4.0
                                    && pr.primary_shelter_id < sh_rep.shelter_rows.len()
                                {
                                    let p_scr = world_to_screen(pr.center.x, pr.center.y);
                                    let s_pos = sh_rep.shelter_rows[pr.primary_shelter_id].world_pos;
                                    let s_scr = world_to_screen(s_pos.x, s_pos.y);
                                    let line_col = if pr.sheltered_clean_air_pct >= 75.0 {
                                        Color32::from_rgba_unmultiplied(52, 211, 153, 115)
                                    } else {
                                        Color32::from_rgba_unmultiplied(251, 146, 60, 130)
                                    };
                                    painter.line_segment([p_scr, s_scr], Stroke::new(1.25, line_col));
                                }
                            }
                        }

                        for sr in &sh_rep.shelter_rows {
                            let s_scr = world_to_screen(sr.world_pos.x, sr.world_pos.y);
                            let sh_col = if !sr.selected_open {
                                Color32::from_rgb(148, 163, 184)
                            } else if sr.clean_air_ok_pct >= 75.0 && sr.peak_occupancy_pct <= 105.0 {
                                Color32::from_rgb(52, 211, 153)
                            } else if sr.clean_air_ok_pct >= 40.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            painter.circle_stroke(s_scr, 13.5, Stroke::new(2.5, sh_col));
                            painter.text(
                                s_scr,
                                egui::Align2::CENTER_CENTER,
                                "SH",
                                egui::FontId::monospace(9.5),
                                sh_col,
                            );
                            if sr.selected_open {
                                painter.text(
                                    Pos2::new(s_scr.x, s_scr.y - 15.5),
                                    egui::Align2::CENTER_BOTTOM,
                                    format!(
                                        "[{} {:.0}% | {:.0}ug]",
                                        sr.archetype.badge_prefix(),
                                        sr.peak_occupancy_pct,
                                        sr.mean_indoor_pm25_ug_m3
                                    ),
                                    egui::FontId::monospace(9.2),
                                    sh_col,
                                );
                            }
                        }
                    }
                }

                // Staged Evacuation & Arterial Contraflow overlays (Outbound CF lanes, Protected EMS/LZ spines & FFE roadblocks)
                if self.layer == HeatmapLayer::EvacuationContraflow {
                    if let Some(erep) = self.latest_evac_report.as_ref() {
                        for cr in &erep.corridor_rows {
                            let p0 = world_to_screen(cr.from_pos.x, cr.from_pos.y);
                            let p1 = world_to_screen(cr.to_pos.x, cr.to_pos.y);
                            if cr.thermal_closure_prob_pct >= 35.0 {
                                painter.line_segment(
                                    [p0, p1],
                                    Stroke::new(3.0, Color32::from_rgb(244, 63, 94)),
                                );
                            } else if cr.contraflow_added_lanes > 0 {
                                painter.line_segment(
                                    [p0, p1],
                                    Stroke::new(2.8, Color32::from_rgb(56, 189, 248)),
                                );
                                if self.cam_zoom >= 0.42 {
                                    let mid = Pos2::new(0.5 * (p0.x + p1.x), 0.5 * (p0.y + p1.y));
                                    painter.text(
                                        mid,
                                        egui::Align2::CENTER_CENTER,
                                        format!("+{}CF", cr.contraflow_added_lanes),
                                        egui::FontId::monospace(8.8),
                                        Color32::from_rgb(254, 240, 138),
                                    );
                                }
                            } else if cr.protected_ems_lz_corridor && cr.remaining_inbound_lanes >= 1 {
                                painter.line_segment(
                                    [p0, p1],
                                    Stroke::new(
                                        1.9,
                                        Color32::from_rgba_unmultiplied(52, 211, 153, 185),
                                    ),
                                );
                            }
                        }

                        if self.cam_zoom >= 0.40 {
                            for pr in &erep.parcel_rows {
                                let p_scr = world_to_screen(pr.center.x, pr.center.y);
                                let st_col = match pr.evac_stage {
                                    1 => Color32::from_rgb(251, 113, 133),
                                    2 => Color32::from_rgb(250, 204, 21),
                                    _ => Color32::from_rgb(103, 232, 249),
                                };
                                painter.text(
                                    p_scr,
                                    egui::Align2::CENTER_CENTER,
                                    format!("S{}", pr.evac_stage),
                                    egui::FontId::monospace(9.0),
                                    st_col,
                                );
                            }
                        }
                    }
                }

                // Rotary-Wing Medevac [H] helipads, smoke-visibility advisories & curved burn-transfer flight arcs
                if self.layer == HeatmapLayer::AeromedicalMedevac {
                    if let Some(mrep) = self.latest_medevac_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[ROTARY-WING MEDEVAC ({}) | EnvOpen={:.0}% | SmokeVis={:.1} SM | Air Burn Xfer={:.2} ({:.0}m vs {:.0}m CCT)]",
                                mrep.config.avionics_regime.short_label(),
                                mrep.system_flight_envelope_open_pct,
                                mrep.mean_helipad_smoke_vis_sm,
                                mrep.expected_air_secondary_burn_transfers,
                                mrep.mean_air_burn_transfer_min,
                                mrep.mean_cct_ground_transfer_min
                            ),
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(125, 211, 252),
                        );

                        // Curved aeromedical flight arcs for secondary inter-hospital burn transfers
                        for arc in &mrep.transfer_arcs {
                            if arc.expected_air_transfers + arc.expected_cct_ground_fallbacks >= 0.02 {
                                let p0 = world_to_screen(arc.from_pos.x, arc.from_pos.y);
                                let p2 = world_to_screen(arc.to_pos.x, arc.to_pos.y);
                                let mid = Pos2::new(
                                    0.5 * (p0.x + p2.x) - 0.14 * (p2.y - p0.y),
                                    0.5 * (p0.y + p2.y) + 0.14 * (p2.x - p0.x) - 18.0,
                                );
                                let mut curve_pts = Vec::with_capacity(13);
                                for k in 0..=12 {
                                    let t = (k as f32) / 12.0;
                                    let omt = 1.0 - t;
                                    curve_pts.push(Pos2::new(
                                        omt * omt * p0.x + 2.0 * omt * t * mid.x + t * t * p2.x,
                                        omt * omt * p0.y + 2.0 * omt * t * mid.y + t * t * p2.y,
                                    ));
                                }
                                let arc_col = if arc.expected_air_transfers >= arc.expected_cct_ground_fallbacks {
                                    if arc.is_external_wrbdc_hub {
                                        Color32::from_rgba_unmultiplied(250, 204, 21, 210)
                                    } else {
                                        Color32::from_rgba_unmultiplied(56, 189, 248, 210)
                                    }
                                } else {
                                    Color32::from_rgba_unmultiplied(251, 146, 60, 175)
                                };
                                painter.add(egui::Shape::line(curve_pts, Stroke::new(2.0, arc_col)));
                            }
                        }

                        // [H] Helipad & Emergency LZ status rings
                        for pr in &mrep.helipad_rows {
                            let h_scr = world_to_screen(pr.world_pos.x, pr.world_pos.y);
                            let pad_col = if pr.pad_type == HelipadStructuralType::ExternalRegionalBurnHub {
                                Color32::from_rgb(250, 204, 21)
                            } else if pr.flight_envelope_open_pct >= 70.0 {
                                Color32::from_rgb(52, 211, 153)
                            } else if pr.flight_envelope_open_pct >= 30.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(244, 63, 94)
                            };
                            painter.circle_stroke(h_scr, 13.0, Stroke::new(2.4, pad_col));
                            painter.text(
                                h_scr,
                                egui::Align2::CENTER_CENTER,
                                "H",
                                egui::FontId::monospace(10.5),
                                pad_col,
                            );
                            painter.text(
                                Pos2::new(h_scr.x, h_scr.y - 15.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!(
                                    "[{} {:.0}% | {:.1}SM]",
                                    pr.pad_type.short_badge(),
                                    pr.flight_envelope_open_pct,
                                    pr.mean_smoke_vis_sm
                                ),
                                egui::FontId::monospace(9.2),
                                pad_col,
                            );
                        }
                    }
                }

                // Hospital ED Surge saturation rings, DIVERT alerts & dynamic ambulance routing vectors
                if self.layer == HeatmapLayer::HospitalTraumaSurge {
                    if let Some(srep) = self.latest_surge_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[HOSPITAL ED SURGE ({}) | Golden-Hr={:.1}% | Red/Burn Wait={:.1}m | Divert={:.1}%]",
                                srep.config.diversion_policy.short_label(),
                                srep.golden_hour_treated_pct,
                                srep.mean_red_burn_ed_wait_min,
                                srep.ambulance_diversion_rate_pct
                            ),
                            egui::FontId::monospace(11.2),
                            Color32::from_rgb(167, 243, 208),
                        );

                        if self.cam_zoom >= 0.30 {
                            for pr in &srep.parcel_rows {
                                if pr.diversion_prob_pct >= 22.0
                                    && (pr.expected_red_trauma + pr.expected_burn_critical) >= 0.04
                                    && pr.primary_hospital_id < srep.hospital_rows.len()
                                {
                                    let p_scr = world_to_screen(pr.center.x, pr.center.y);
                                    let h_pos = srep.hospital_rows[pr.primary_hospital_id].world_pos;
                                    let h_scr = world_to_screen(h_pos.x, h_pos.y);
                                    painter.line_segment(
                                        [p_scr, h_scr],
                                        Stroke::new(
                                            1.35,
                                            Color32::from_rgba_unmultiplied(56, 189, 248, 135),
                                        ),
                                    );
                                }
                            }
                        }

                        for hr in &srep.hospital_rows {
                            let h_scr = world_to_screen(hr.world_pos.x, hr.world_pos.y);
                            let ring_col = if hr.designation == HospitalTraumaDesignation::MobileFieldHospital {
                                Color32::from_rgb(56, 189, 248)
                            } else if hr.peak_ed_utilization_pct >= 95.0 {
                                Color32::from_rgb(244, 63, 94)
                            } else if hr.peak_ed_utilization_pct >= 75.0 {
                                Color32::from_rgb(250, 204, 21)
                            } else {
                                Color32::from_rgb(52, 211, 153)
                            };
                            painter.circle_stroke(h_scr, 14.0, Stroke::new(2.6, ring_col));
                            let div_tag = if hr.time_on_diversion_pct >= 25.0 {
                                " DIVERT"
                            } else {
                                ""
                            };
                            painter.text(
                                Pos2::new(h_scr.x, h_scr.y - 16.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!(
                                    "[{}{div_tag} {:.0}% | {:.0}m]",
                                    hr.designation.badge_prefix(),
                                    hr.peak_ed_utilization_pct,
                                    hr.mean_red_burn_wait_min
                                ),
                                egui::FontId::monospace(9.6),
                                ring_col,
                            );
                        }
                    }
                }

                // FFE Conflagration hotspots, dry-hydrant warnings & prevailing wind arrow on viewport
                if self.layer == HeatmapLayer::FireFollowingQuake {
                    if let Some(frep) = self.latest_ffe_report.as_ref() {
                        painter.text(
                            Pos2::new(rect.left() + 14.0, rect.top() + 36.0),
                            egui::Align2::LEFT_TOP,
                            format!(
                                "[FFE CONFLAGRATION | Wind {:.1} m/s @ {:.0} deg | E[Burn]={:.1} parcels (${:.1}M) | DryHyd={:.0}%]",
                                frep.config.wind_speed_mps,
                                frep.config.wind_direction_deg,
                                frep.expected_burned_parcels,
                                frep.expected_fire_loss_musd,
                                frep.dry_hydrant_parcels_pct
                            ),
                            egui::FontId::monospace(11.2),
                            Color32::from_rgb(253, 186, 116),
                        );
                        if self.cam_zoom >= 0.36 {
                            for r in &frep.parcel_risks {
                                if r.burn_prob_pct >= 8.0 {
                                    let p_scr = world_to_screen(r.center.x, r.center.y);
                                    let rad = (6.0 + (r.burn_prob_pct as f32) * 0.25).clamp(6.0, 18.0);
                                    painter.circle_stroke(
                                        p_scr,
                                        rad,
                                        Stroke::new(2.0, Color32::from_rgb(251, 146, 60)),
                                    );
                                    if r.dry_hydrant_prob_pct >= 40.0 {
                                        painter.text(
                                            Pos2::new(p_scr.x, p_scr.y - rad - 2.0),
                                            egui::Align2::CENTER_BOTTOM,
                                            format!("DRY {:.0}%", r.dry_hydrant_prob_pct),
                                            egui::FontId::monospace(8.8),
                                            Color32::from_rgb(253, 224, 71),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Substation, Water Plant, Lift Station & WWTP badges when UtilityLifelines is active
                if self.layer == HeatmapLayer::UtilityLifelines {
                    for asset in &self.world.utility_report.assets {
                        let a_scr = world_to_screen(asset.world_pos.x, asset.world_pos.y);
                        let ring_col = if !asset.online {
                            Color32::from_rgb(244, 63, 94)
                        } else if asset.on_backup_power {
                            Color32::from_rgb(250, 204, 21)
                        } else {
                            match asset.kind {
                                UtilityAssetKind::ElectricSubstation => {
                                    Color32::from_rgb(250, 204, 21)
                                }
                                UtilityAssetKind::WaterTreatmentBooster => {
                                    Color32::from_rgb(56, 189, 248)
                                }
                                UtilityAssetKind::SewageLiftStation => {
                                    Color32::from_rgb(192, 132, 252)
                                }
                                UtilityAssetKind::WastewaterTreatmentPlant => {
                                    Color32::from_rgb(52, 211, 153)
                                }
                            }
                        };
                        painter.circle_stroke(a_scr, 11.0, Stroke::new(2.3, ring_col));
                        let status_sfx = if !asset.online {
                            ":OFF"
                        } else if asset.on_backup_power {
                            ":BKP"
                        } else {
                            ""
                        };
                        painter.text(
                            Pos2::new(a_scr.x, a_scr.y - 13.0),
                            egui::Align2::CENTER_BOTTOM,
                            format!("[{}{}]", asset.kind.short_badge(), status_sfx),
                            egui::FontId::monospace(10.5),
                            ring_col,
                        );
                    }
                }

                // Crisp facility badges over placed Fire Stations & Trauma Hospitals
                if self.cam_zoom >= 0.42 {
                    for fac in &self.world.emergency_facilities {
                        let f_scr = world_to_screen(fac.world_pos.x, fac.world_pos.y);
                        let badge_col = match fac.kind {
                            EmergencyFacilityKind::FireStation => Color32::from_rgb(254, 205, 211),
                            EmergencyFacilityKind::Hospital => Color32::from_rgb(186, 230, 253),
                        };
                        painter.text(
                            Pos2::new(f_scr.x, f_scr.y - (5.5 * self.cam_zoom as f32).max(8.0)),
                            egui::Align2::CENTER_BOTTOM,
                            format!("{}{}", fac.kind.short_prefix(), fac.id + 1),
                            egui::FontId::monospace(11.0),
                            badge_col,
                        );
                    }
                }

                // Highlight selected junction ring
                if let Some(sel_j) = self.world.network.junctions.get(self.selected_junction) {
                    if !sel_j.in_segs.is_empty() || !sel_j.out_segs.is_empty() {
                        painter.circle_stroke(
                            world_to_screen(sel_j.pos.x, sel_j.pos.y),
                            (9.0 * self.cam_zoom as f32).clamp(6.0, 26.0),
                            Stroke::new(1.6, Color32::from_rgba_unmultiplied(56, 189, 248, 175)),
                        );
                    }
                }

                // Live interactive CAD tool cursor previews
                if let Some((_, hwx, hwy)) = hover_world {
                    match self.tool {
                        ViewportTool::DrawRoad => {
                            let snap = preview_draw_snap(&self.world.network, hwx, hwy);
                            let sp = snap.pos();
                            let sp_scr = world_to_screen(sp.x, sp.y);
                            let snap_col = match snap {
                                SnapTarget::ExistingJunction { .. } => Color32::from_rgb(16, 185, 129),
                                SnapTarget::SplitSegment { .. } => Color32::from_rgb(250, 204, 21),
                                SnapTarget::FreePoint { .. } => Color32::from_rgb(56, 189, 248),
                            };
                            painter.circle_stroke(sp_scr, 9.0, Stroke::new(2.2, snap_col));

                            if let Some(orig) = self.draw_start_world {
                                let o_scr = world_to_screen(orig.x, orig.y);
                                let total_lanes = if self.draw_spec.bidirectional {
                                    self.draw_spec.lanes_per_dir * 2
                                } else {
                                    self.draw_spec.lanes_per_dir
                                };
                                let width_px =
                                    ((total_lanes as f64 * 3.5) * self.cam_zoom).clamp(3.0, 48.0)
                                        as f32;
                                painter.line_segment(
                                    [o_scr, sp_scr],
                                    Stroke::new(
                                        width_px,
                                        Color32::from_rgba_unmultiplied(56, 189, 248, 70),
                                    ),
                                );
                                painter.line_segment(
                                    [o_scr, sp_scr],
                                    Stroke::new(2.0, snap_col),
                                );
                                let mid = Pos2::new(
                                    0.5 * (o_scr.x + sp_scr.x),
                                    0.5 * (o_scr.y + sp_scr.y) - 12.0,
                                );
                                painter.text(
                                    mid,
                                    egui::Align2::CENTER_BOTTOM,
                                    format!("{:.1} m ({} ln/dir)", orig.dist(sp), self.draw_spec.lanes_per_dir),
                                    egui::FontId::monospace(12.0),
                                    Color32::WHITE,
                                );
                            }
                        }
                        ViewportTool::EditLanes => {
                            if let Some((sid, _, proj)) =
                                hit_test_segment(&self.world.network, hwx, hwy, 18.0)
                            {
                                let seg = &self.world.network.segments[sid];
                                let pts: Vec<Pos2> = seg
                                    .base_pts
                                    .iter()
                                    .map(|p| world_to_screen(p.x, p.y))
                                    .collect();
                                painter.add(egui::Shape::line(
                                    pts,
                                    Stroke::new(
                                        3.2,
                                        Color32::from_rgba_unmultiplied(250, 204, 21, 225),
                                    ),
                                ));
                                let lbl_pos = world_to_screen(proj.x, proj.y);
                                painter.text(
                                    Pos2::new(lbl_pos.x, lbl_pos.y - 12.0),
                                    egui::Align2::CENTER_BOTTOM,
                                    format!("#{} {} ({} ln/dir)", seg.id, seg.name, seg.lane_count),
                                    egui::FontId::monospace(12.0),
                                    Color32::from_rgb(254, 240, 138),
                                );
                            }
                        }
                        ViewportTool::ConvertRoundabout => {
                            if let Some(jid) = hit_test_junction(&self.world.network, hwx, hwy, 24.0)
                            {
                                let j = &self.world.network.junctions[jid];
                                let ok = !j.is_roundabout_node && j.arm_count >= 3;
                                let c = if ok {
                                    Color32::from_rgb(16, 185, 129)
                                } else {
                                    Color32::from_rgb(244, 63, 94)
                                };
                                let c_scr = world_to_screen(j.pos.x, j.pos.y);
                                let r_px = (self.roundabout_radius_m * self.cam_zoom).max(10.0) as f32;
                                painter.circle_stroke(c_scr, r_px, Stroke::new(2.4, c));
                                painter.circle_stroke(
                                    c_scr,
                                    (r_px * 0.45).max(4.0),
                                    Stroke::new(1.4, Color32::from_rgba_unmultiplied(148, 163, 184, 180)),
                                );
                                painter.text(
                                    Pos2::new(c_scr.x, c_scr.y - r_px - 6.0),
                                    egui::Align2::CENTER_BOTTOM,
                                    format!(
                                        "Junc #{} ({}-arm, {} ring ln)",
                                        j.id, j.arm_count, self.roundabout_lanes
                                    ),
                                    egui::FontId::monospace(12.0),
                                    c,
                                );
                            }
                        }
                        ViewportTool::PlaceFacility => {
                            let c = match self.active_facility_kind {
                                EmergencyFacilityKind::FireStation => Color32::from_rgb(244, 63, 94),
                                EmergencyFacilityKind::Hospital => Color32::from_rgb(56, 189, 248),
                            };
                            let cur_scr = world_to_screen(hwx, hwy);
                            painter.circle_stroke(cur_scr, 12.0, Stroke::new(2.2, c));
                            painter.text(
                                Pos2::new(cur_scr.x, cur_scr.y - 15.0),
                                egui::Align2::CENTER_BOTTOM,
                                format!("+ Place {}", self.active_facility_kind.label()),
                                egui::FontId::monospace(12.0),
                                c,
                            );
                        }
                        ViewportTool::ToggleClosure => {
                            if let Some((sid, _, proj)) =
                                hit_test_segment(&self.world.network, hwx, hwy, 20.0)
                            {
                                let seg = &self.world.network.segments[sid];
                                let p_scr = world_to_screen(proj.x, proj.y);
                                let col = if seg.closed {
                                    Color32::from_rgb(52, 211, 153)
                                } else {
                                    Color32::from_rgb(244, 63, 94)
                                };
                                let act = if seg.closed { "Reopen" } else { "Close" };
                                painter.circle_stroke(p_scr, 13.0, Stroke::new(2.4, col));
                                painter.text(
                                    Pos2::new(p_scr.x, p_scr.y - 16.0),
                                    egui::Align2::CENTER_BOTTOM,
                                    format!("{act} `{}` (#{sid})", seg.name),
                                    egui::FontId::monospace(11.5),
                                    col,
                                );
                            }
                        }
                        ViewportTool::Inspect | ViewportTool::PaintZone => {}
                    }
                }

                // Crisp HUD overlay pill in top-left of viewport
                let hud_rect = Rect::from_min_size(
                    Pos2::new(rect.left() + 12.0, rect.top() + 12.0),
                    EguiVec2::new(430.0, 28.0),
                );
                painter.rect_filled(
                    hud_rect,
                    6.0,
                    Color32::from_rgba_unmultiplied(15, 23, 42, 215),
                );
                painter.text(
                    Pos2::new(hud_rect.left() + 10.0, hud_rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    format!(
                        "wgpu @compute + Instanced  |  {} quads + {} SDF agents  |  {:.0} FPS ({:.2} ms)",
                        self.staging.road_quads.len(),
                        self.staging.agents.len(),
                        (1000.0 / self.ema_frame_ms.max(0.5)).min(999.0),
                        self.ema_frame_ms
                    ),
                    egui::FontId::monospace(11.5),
                    Color32::from_rgb(186, 230, 253),
                );
                if self.layer == HeatmapLayer::EvacuationContraflow {
                    if let Some(erep) = self.latest_evac_report.as_ref() {
                        let evac_label = format!(
                            "[EVACUATION ({} + {}) | T90={:.1}m | Overrun={:.1} res | EMS-LZ Access={:.0}%]",
                            erep.config.staging_strategy.short_label(),
                            erep.config.contraflow_policy.short_label(),
                            erep.t90_clearance_min,
                            erep.expected_fire_overrun_residents,
                            erep.ems_lz_corridor_access_preserved_pct
                        );
                        let evac_bg = Rect::from_min_size(
                            Pos2::new(rect.left() + 12.0, rect.top() + 44.0),
                            EguiVec2::new((evac_label.len() as f32 * 6.8 + 26.0).min(rect.width() - 24.0), 22.0),
                        );
                        painter.rect_filled(
                            evac_bg,
                            5.0,
                            Color32::from_rgb(10, 15, 28),
                        );
                        painter.text(
                            Pos2::new(evac_bg.left() + 8.0, evac_bg.center().y),
                            egui::Align2::LEFT_CENTER,
                            evac_label,
                            egui::FontId::monospace(11.0),
                            Color32::from_rgb(167, 243, 208),
                        );
                    }
                }
            } else {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "wgpu render state unavailable — launch with default wgpu renderer enabled.",
                    egui::FontId::proportional(15.0),
                    Color32::LIGHT_GRAY,
                );
            }
        });
    }
}

impl eframe::App for UrbanGridGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui_step(ctx);
    }
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    fn noop_clone(_: *const ()) -> std::task::RawWaker {
        noop_raw_waker()
    }
    fn noop(_: *const ()) {}
    fn noop_raw_waker() -> std::task::RawWaker {
        static VTABLE: std::task::RawWakerVTable =
            std::task::RawWakerVTable::new(noop_clone, noop, noop, noop);
        std::task::RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { std::task::Waker::from_raw(noop_raw_waker()) };
    let mut cx = std::task::Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn write_rgba_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    let row_bytes = width as usize * 4;
    if rgba.len() != row_bytes * height as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Unexpected RGBA buffer length {}", rgba.len()),
        ));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in rgba.chunks_exact(row_bytes) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    let mut zlib = Vec::with_capacity(raw.len() + (raw.len() / 65535 + 1) * 5 + 6);
    zlib.extend_from_slice(&[0x78, 0x01]);
    let n_chunks = raw.len().div_ceil(65535).max(1);
    for (idx, chunk) in raw.chunks(65535).enumerate() {
        zlib.push(if idx + 1 == n_chunks { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(chunk);
    }
    zlib.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

    let push_chunk = |out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let mut crc = 0xFFFF_FFFFu32;
        for &b in tag.iter().chain(data.iter()) {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        out.extend_from_slice(&(crc ^ 0xFFFF_FFFF).to_be_bytes());
    };
    let mut png = Vec::with_capacity(zlib.len() + 64);
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = 6;
    push_chunk(&mut png, b"IHDR", &ihdr);
    push_chunk(&mut png, b"IDAT", &zlib);
    push_chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

pub fn render_headless_screenshot(
    world: SimulationWorld,
    out_path: &std::path::Path,
    view: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let width: u32 = 1600;
    let height: u32 = 1000;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("No GPU adapter available for offscreen rendering")?;
    let (device, queue) = block_on_future(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("urban-grid-offscreen-device"),
            ..Default::default()
        },
        None,
    ))?;

    let adapter = Arc::new(adapter);
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);
    let wgpu_state = egui_wgpu::RenderState {
        adapter: Arc::clone(&adapter),
        available_adapters: Arc::from(Vec::<wgpu::Adapter>::new()),
        device: Arc::clone(&device),
        queue: Arc::clone(&queue),
        target_format: format,
        renderer: Arc::new(egui::epaint::mutex::RwLock::new(renderer)),
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("urban-grid-offscreen-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bytes_per_row = width * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("urban-grid-offscreen-staging"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let egui_ctx = egui::Context::default();
    egui_ctx.set_pixels_per_point(1.0);
    let mut app = UrbanGridGuiApp::new_with_wgpu_state(Some(&wgpu_state), world);
    if view == 2 {
        app.load_builtin(&builtin_harbor_peninsula_osm(), None);
        app.mclp_cfg.max_stations_budget = 1;
        let mrep = app.world.solve_mclp_station_siting(&app.mclp_cfg);
        app.world.apply_mclp_siting_solution(&mrep);
        app.latest_mclp_report = Some(mrep);
        app.world.apply_tod_upzoning_tracked();
        app.world.transit_gtfs.fare_mode = ug_core::BoardingFareMode::AllDoorFareFreeBrt;
        app.world.transit_gtfs.coach_spec = ug_core::BusCoachSpec::Articulated60FtBrt;
        app.world.transit_gtfs.hold_early_buses = true;
        for _ in 0..180 {
            app.world.step(0.1);
        }
        app.enter_workbench(
            DockTab::ZoningAndTransit,
            None,
            HeatmapLayer::EmergencyResponse4Min,
            ViewportTool::Inspect,
        );
        app.cam_zoom = 0.62;
    } else if view == 3 {
        app.load_builtin(&builtin_foothill_canyon_wui_osm(), None);
        let mrep = app.world.solve_mclp_station_siting(&app.mclp_cfg);
        app.world.apply_mclp_siting_solution(&mrep);
        app.latest_mclp_report = Some(mrep);

        app.latest_stress_report = None;

        let mcrep = app.world.run_seismic_monte_carlo(&app.seismic_cfg);
        app.fragility_plot_arch = mcrep.reference_archetype;
        app.latest_seismic_report = Some(mcrep);

        app.medevac_cfg.surge = app.surge_cfg.clone();
        app.evac_cfg.medevac = app.medevac_cfg.clone();

        let mut b_cfg = app.evac_cfg.clone();
        b_cfg.staging_strategy = ug_core::EvacuationStagingStrategy::SimultaneousUnstaged;
        b_cfg.contraflow_policy = ug_core::ContraflowReversalPolicy::NoContraflowBaseline;
        let b_rep = app.world.run_evacuation_contraflow_mc(&b_cfg);

        let mut s_cfg = app.evac_cfg.clone();
        s_cfg.staging_strategy = ug_core::EvacuationStagingStrategy::FireProximityThreeStage;
        s_cfg.contraflow_policy = ug_core::ContraflowReversalPolicy::EmsProtectedArterialContraflow;
        let s_rep = app.world.run_evacuation_contraflow_mc(&s_cfg);

        app.world.apply_evacuation_contraflow_to_network(&s_rep);
        app.evac_cfg.staging_strategy = ug_core::EvacuationStagingStrategy::FireProximityThreeStage;
        app.evac_cfg.contraflow_policy =
            ug_core::ContraflowReversalPolicy::EmsProtectedArterialContraflow;
        app.evac_ab_summary = Some(format!(
            "A/B Evac: Unstaged+NoCF P90={:.0}m, Overrun={:.0} res, EMS={:.0}% -> 3-Stage+EMS-CF P90={:.0}m, Overrun={:.0} res, EMS={:.0}% (+{:.1} lane-km, {:.1} ms)",
            b_rep.t90_clearance_min,
            b_rep.expected_fire_overrun_residents,
            b_rep.ems_lz_corridor_access_preserved_pct,
            s_rep.t90_clearance_min,
            s_rep.expected_fire_overrun_residents,
            s_rep.ems_lz_corridor_access_preserved_pct,
            s_rep.contraflow_lane_km_added,
            s_rep.solve_time_ms,
        ));
        app.latest_evac_report = Some(s_rep);

        for _ in 0..30 {
            app.world.step(0.05);
        }
        app.enter_workbench(
            DockTab::ResilienceStress,
            Some(StressSubMode::EvacuationContraflow),
            HeatmapLayer::EvacuationContraflow,
            ViewportTool::Inspect,
        );
        app.cam_zoom = 0.62;
    } else {
        // View 1: Default Riverfront Bridges Metro shown paused/ready at t=0.0s
        app.enter_workbench(
            DockTab::SignalsAndRoads,
            None,
            HeatmapLayer::Zoning,
            ViewportTool::Inspect,
        );
        app.cam_zoom = 0.66;
    }

    for frame_idx in 0..4u32 {
        let raw_input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                EguiVec2::new(width as f32, height as f32),
            )),
            viewports: std::iter::once((
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(1.0),
                    ..Default::default()
                },
            ))
            .collect(),
            time: Some(frame_idx as f64 / 60.0),
            predicted_dt: 1.0 / 60.0,
            ..Default::default()
        };
        let full_output = egui_ctx.run(raw_input, |ctx| {
            app.ui_step(ctx);
        });
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 1.0,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("urban-grid-offscreen-encoder"),
        });
        let user_cmd_bufs = {
            let mut renderer = wgpu_state.renderer.write();
            for (id, image_delta) in &full_output.textures_delta.set {
                renderer.update_texture(&device, &queue, *id, image_delta);
            }
            renderer.update_buffers(
                &device,
                &queue,
                &mut encoder,
                &paint_jobs,
                &screen_descriptor,
            )
        };
        {
            let renderer = wgpu_state.renderer.read();
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("urban-grid-offscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let mut render_pass = render_pass.forget_lifetime();
            renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }
        if frame_idx == 3 {
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(user_cmd_bufs.into_iter().chain(std::iter::once(encoder.finish())));
        device.poll(wgpu::Maintain::Wait);
        let mut renderer = wgpu_state.renderer.write();
        for id in &full_output.textures_delta.free {
            renderer.free_texture(id);
        }
    }

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()??;
    let mapped = slice.get_mapped_range();
    write_rgba_png(out_path, width, height, &mapped)?;
    drop(mapped);
    staging.unmap();
    Ok(())
}

pub fn run_gui(world: SimulationWorld, vsync: bool) -> eframe::Result<()> {
    let present_mode = if vsync {
        wgpu::PresentMode::AutoVsync
    } else {
        wgpu::PresentMode::AutoNoVsync
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(
                "urban-grid v5.2 — FEMA Suite, ALERTWest Wildfire, WTP Turbidity & 50k+ wgpu Workbench",
            )
            .with_inner_size([1440.0, 880.0]),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            present_mode,
            desired_maximum_frame_latency: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "urban-grid",
        options,
        Box::new(|cc| Ok(Box::new(UrbanGridGuiApp::new(cc, world)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ug_core::{parse_osm_xml_str, RoadNetwork};

    #[test]
    fn test_paused_on_launch_and_zero_background_sim_while_on_hub_or_paused() {
        let osm = parse_osm_xml_str(&builtin_riverfront_bridges_osm()).unwrap();
        let world = SimulationWorld::new(RoadNetwork::compile(&osm));
        let mut app = UrbanGridGuiApp::new_with_wgpu_state(None, world);
        assert!(app.paused);
        assert_eq!(app.screen, GuiScreen::MainEntranceHub);
        assert_eq!(app.world.sim_time_s, 0.0);
        assert_eq!(app.active_city_name, "Riverfront Bridges Metro");

        let ctx = egui::Context::default();
        let make_input = |t_s: f64| egui::RawInput {
            time: Some(t_s),
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, EguiVec2::new(1280.0, 800.0))),
            ..Default::default()
        };
        let mut t_s = 0.0;
        for frame_idx in 0..8 {
            t_s += 0.5;
            let out = ctx.run(make_input(t_s), |c| {
                app.ui_step(c);
            });
            if frame_idx >= 5 {
                assert!(
                    !out.viewport_output.values().any(|v| v.repaint_delay.is_zero()),
                    "MainEntranceHub must not request continuous repaint once settled"
                );
            }
        }
        assert_eq!(app.world.sim_time_s, 0.0);

        // Entering ActiveWorkbench stays paused at t = 0.0s until user starts
        app.enter_workbench(
            DockTab::SignalsAndRoads,
            None,
            HeatmapLayer::Zoning,
            ViewportTool::Inspect,
        );
        assert!(app.paused);
        for frame_idx in 0..8 {
            t_s += 0.5;
            let out = ctx.run(make_input(t_s), |c| {
                app.ui_step(c);
            });
            if frame_idx >= 5 {
                assert!(
                    !out.viewport_output.values().any(|v| v.repaint_delay.is_zero()),
                    "Paused ActiveWorkbench must not spin continuous repaint"
                );
            }
        }
        assert_eq!(app.world.sim_time_s, 0.0);

        // Unpausing advances physics cleanly and requests continuous repaint
        app.paused = false;
        let mut requested_active_repaint = false;
        for _ in 0..4 {
            t_s += 0.02;
            let out = ctx.run(make_input(t_s), |c| {
                app.ui_step(c);
            });
            if out.viewport_output.values().any(|v| v.repaint_delay.is_zero()) {
                requested_active_repaint = true;
            }
        }
        assert!(requested_active_repaint);
        assert!(app.world.sim_time_s > 0.0);
    }
}

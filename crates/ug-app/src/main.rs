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

mod batch;
#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "tui")]
mod tui;
#[cfg(feature = "gui")]
mod wgpu_pass;

use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::PathBuf;
use ug_core::{
    apply_tod_upzoning_policy, builtin_braess_diamond_osm, builtin_foothill_canyon_wui_osm,
    builtin_harbor_peninsula_osm, builtin_metropolis_50k_osm, builtin_radial_ring_plazas_osm,
    builtin_riverfront_bridges_osm, builtin_roundabout_osm, load_osm_path, parse_osm_xml_str,
    BatchExperimentConfig, BusynessModel, DisruptionInterdictionDepth, DiurnalDispatchRegime,
    DiurnalProfileConfig, DiurnalWindowPreset, HypercubeConfig, HypercubeQueueDiscipline,
    MclpSitingConfig, RedistrictingConfig, RedistrictingObjectiveMode,
    AeromedicalMedevacConfig, AirFiltrationStandard, AlertChannelMix,
    AmbulanceDiversionPolicy, BoardingFareMode, BusCoachSpec, BusContraflowLaneMode,
    CommodityResupplyConfig, ContraflowReversalPolicy, DepartureStagingPolicy,
    DebrisAndUsarPolicy, EmergencySheltersConfig, EocIpaswUplinkMode,
    builtin_western_wildfire_camera_feed_json, load_wildfire_camera_feed_path,
    EvacuationContraflowConfig, FemaCommsConfig, FemaCommsPosture, FemaGsbDrcTdmsConfig,
    FemaIsbPodsConfig, FemaRecoveryExecutionPolicy, FemaUsarDebrisConfig,
    BaerWatershedMitigationPolicy, BurnScarDebrisFlowConfig,
    WtpCoagulantAndIntakePolicy, WtpTurbidityStorageConfig,
    IpaswPodRoutingPolicy, IsbReplenishmentPolicy, WildfireAlertWestConfig,
    WildfireDetectionAndAttackPosture, WuiFuelModel,
    ResupplyDispatchPolicy,
    TelecomBackupPowerStandard, TransitDispatchStrategy, TransitEvacuationConfig,
    WarningMobilizationConfig,
    EvacuationStagingStrategy, FfeConfig, FfeIgnitionModel, ShelterSitingAndRoutingPolicy,
    HospitalSurgeConfig, MedevacAvionicsRegime, PipelineFragilityLaw, PlumePhysicsParams,
    ResilienceStressConfig, RoadNetwork, RotorcraftAirframeClass,
    SeismicScenarioConfig, SimulationWorld, SitingObjectiveModel, UtilityGridsConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WtpPolicyCli {
    RunToFailure,
    IntakeShutoff,
    FerricPacWells,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BaerPolicyCli {
    Untreated,
    AerialMulch,
    FullBaer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FemaRecoveryPolicyCli {
    AdHoc,
    PowerFirst,
    IntegratedSuite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WildfirePostureCli {
    Legacy911,
    FixedPtzIa,
    FullAerial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WuiFuelCli {
    GrassFm1,
    BrushFm5,
    ChaparralFm4,
    TimberFm10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum UsarDebrisPolicyCli {
    HaulOff,
    PodsFirst,
    TeethOfComb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PodRoutingCli {
    NearestRush,
    StaticDistricts,
    IpaswBalanced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum IsbPolicyCli {
    OvernightStatic,
    DynamicHotshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EocUplinkCli {
    FiberOnly,
    FiberFnarsHf,
    FullMersFnars,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FemaPostureCli {
    WeaOnly,
    TriPathIpasw,
    FullIpaswMers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TelecomBackupCli {
    Legacy3h,
    Fcc8h,
    Cpuc72hLeo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum AlertChannelCli {
    WeaOnly,
    WeaSirens,
    MultiModal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DepartureStagingCli {
    Unstaged,
    StaticWea,
    AdaptiveBlackout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BusLaneModeCli {
    MixedQueue,
    CurbBusOnly,
    BusEmsContraflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TransitEvacPolicyCli {
    FixedGtfs,
    DemandProp,
    CleanAirTbep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ResupplyPolicyCli {
    MilkRun,
    MyopicSs,
    ClarkeWrightIrp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ShelterPolicyCli {
    NearestUncoord,
    CapDist,
    CleanAir2Stage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum AirFilterCli {
    Merv8,
    Merv13,
    Merv13Hepa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EvacStagingCli {
    Unstaged,
    Staged3Zone,
    StagedVuln,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ContraflowCli {
    None,
    FullOneWay,
    EmsProtected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum MedevacAvionicsCli {
    DayVfr,
    NightVfr,
    Nvis,
    PinsIfr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DiversionCliPolicy {
    Closest,
    Threshold,
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BuiltinScenario {
    RiverfrontBridges,
    HarborPeninsula,
    RadialRingPlazas,
    FoothillCanyonWui,
    Metropolis50k,
    Roundabout,
    Braess,
}

#[derive(Parser, Debug)]
#[command(
    name = "urban-grid",
    version,
    about = "Standalone OpenStreetMap Microscopic Traffic, GTFS Transit, 50k+ wgpu WGSL Crowd & Urban Planning Simulator"
)]
struct Cli {
    /// Render an offscreen 1600x1000 RGBA PNG screenshot without opening a GUI window
    #[arg(long, value_name = "PATH", global = true)]
    #[allow(dead_code)]
    screenshot: Option<std::path::PathBuf>,

    /// Select which built-in module showcase view to capture (1 = primary workbench, 2 = GTFS transit & 4-min EMS, 3 = seismic & contraflow evacuation)
    #[arg(long, default_value_t = 1, value_name = "VIEW", global = true)]
    #[allow(dead_code)]
    screenshot_view: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch interactive egui + custom wgpu @compute/instanced 120 FPS desktop workbench
    Gui {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::Metropolis50k)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        gtfs: Option<PathBuf>,
        /// Optional .json scenario snapshot to restore on startup
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long)]
        agents: Option<usize>,
        /// Enable decentralized Varaiya (2013) Max-Pressure adaptive signal control across all junctions
        #[arg(long)]
        max_pressure: bool,
        /// Optional number of EmergencyResponder units dispatched immediately at startup
        #[arg(long, default_value_t = 0)]
        evp_units: usize,
        #[arg(long, default_value_t = false)]
        vsync: bool,
    },
    /// Launch interactive Ratatui terminal dashboard
    Tui {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        gtfs: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long)]
        max_pressure: bool,
    },
    /// Run headless batch simulation and emit JSON/CSV outputs
    Simulate {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        gtfs: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 180.0)]
        seconds: f64,
        #[arg(long)]
        agents: Option<usize>,
        #[arg(long)]
        tod_upzone: bool,
        #[arg(long)]
        max_pressure: bool,
        #[arg(long)]
        json_out: Option<PathBuf>,
        /// Optional path to save a full .json ScenarioSnapshot at the end of the run
        #[arg(long)]
        save_snapshot: Option<PathBuf>,
        /// Optional path to export the 1 Hz 29-column time-series telemetry .csv
        #[arg(long)]
        csv_out: Option<PathBuf>,
    },
    /// Compare Baseline vs. 15-Minute TOD + 5-min All-Door BRT + Protected Bike Network
    Compare {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        gtfs: Option<PathBuf>,
        #[arg(long, default_value_t = 180.0)]
        seconds: f64,
        #[arg(long)]
        csv_baseline: Option<PathBuf>,
        #[arg(long)]
        csv_tod: Option<PathBuf>,
    },
    /// Compare Fixed-Time, Max-Pressure Adaptive, and Modern Roundabout scenarios in parallel with Rayon and export a summary CSV scorecard
    Batch {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        gtfs: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        /// Duration in simulated seconds per parallel run
        #[arg(long, default_value_t = 180.0)]
        seconds: f64,
        /// Warmup seconds prior to recording steady-state throughput & queue averages
        #[arg(long, default_value_t = 30.0)]
        warmup: f64,
        /// Comma-separated demand scale multipliers (e.g. `0.75,1.0,1.35`)
        #[arg(long, default_value = "0.75,1.00,1.35")]
        demand_scales: String,
        /// Monte Carlo Common Random Numbers (CRN) replicates per (demand, regime) cell
        #[arg(long, default_value_t = 1)]
        replicates: usize,
        /// Inscribed roundabout radius (m) for Modern Roundabout regime
        #[arg(long, default_value_t = 18.0)]
        roundabout_radius: f64,
        /// Circulatory lanes (`1..=3`) for Modern Roundabout regime
        #[arg(long, default_value_t = 2)]
        roundabout_lanes: usize,
        /// Maximum candidate 3+ arm intersections converted in Modern Roundabout regime
        #[arg(long, default_value_t = 6)]
        max_roundabouts: usize,
        /// Apply 15-Minute TOD upzoning + BRT + protected bike grid across all compared regimes
        #[arg(long)]
        tod_upzone: bool,
        /// Optional Rayon worker thread count (defaults to all logical CPU cores)
        #[arg(long)]
        threads: Option<usize>,
        /// Output path for RFC 4180 28-column comparative summary CSV scorecard
        #[arg(long, default_value = "exports/batch_scorecard.csv")]
        csv_out: PathBuf,
        /// Optional JSON output path for structured BatchExperimentReport
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Solve Daskin (1983) MEXCLP (busy apparatus & 2nd-due backup) or Church & ReVelle (1974) MCLP station/fleet siting
    Mclp {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::Metropolis50k)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 90.0)]
        target_pct: f64,
        #[arg(long, default_value_t = 240.0)]
        threshold_s: f64,
        #[arg(long, default_value_t = 14)]
        max_stations: usize,
        #[arg(long, default_value_t = true)]
        mexclp: bool,
        #[arg(long, default_value_t = 0.25)]
        busy_prob: f64,
        #[arg(long, default_value_t = false)]
        endogenous_queue: bool,
        #[arg(long, default_value_t = false)]
        hypercube: bool,
        #[arg(long, default_value_t = 2)]
        max_engines: usize,
        #[arg(long, default_value_t = 0.0)]
        backup_target_pct: f64,
        #[arg(long)]
        keep_existing: bool,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Run arterial bridge/corridor closure stress-test (N-1 & N-2 interdiction) on MCLP station placements
    StressTest {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::Metropolis50k)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 90.0)]
        target_pct: f64,
        #[arg(long, default_value_t = 240.0)]
        threshold_s: f64,
        #[arg(long, default_value_t = true)]
        auto_site_mclp: bool,
        #[arg(long, default_value_t = false)]
        harden_mclp: bool,
        #[arg(long, default_value_t = true)]
        n_minus_2: bool,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Run probabilistic Hazus seismic bridge fragility & spatially correlated Monte Carlo post-event travel-time sweep
    SeismicMc {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::Metropolis50k)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 7.2)]
        mw: f64,
        #[arg(long, default_value_t = 16.0)]
        distance_km: f64,
        #[arg(long, default_value_t = 270.0)]
        vs30: f64,
        #[arg(long, default_value_t = 500)]
        trials: usize,
        #[arg(long, default_value_t = 0)]
        retrofit_top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        crew_speedup: f64,
        #[arg(long, default_value_t = 0.0)]
        pipe_replace_pct: f64,
        #[arg(long, default_value_t = 6)]
        pipe_crews: usize,
        #[arg(long, default_value_t = false)]
        hazus_pgv_law: bool,
        #[arg(long, default_value_t = false)]
        backup_power: bool,
        #[arg(long, default_value_t = true)]
        auto_site_mclp: bool,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        recovery_csv_out: Option<PathBuf>,
        #[arg(long)]
        pipeline_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Solve Larson (1974) exact 2^N M/M/N spatial hypercube queueing equations for heterogeneous per-unit busy fractions
    Hypercube {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 3.6)]
        calls_per_hr: f64,
        #[arg(long, default_value_t = 36.0)]
        service_mins: f64,
        #[arg(long, default_value_t = false)]
        loss_mode: bool,
        #[arg(long, default_value_t = true)]
        calibrate_mu: bool,
        #[arg(long, default_value_t = 5)]
        stations_budget: usize,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Optimize Carter, Chaiken & Ignall (1972) / Larson hypercube response-district boundaries
    Redistrict {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 0.55)]
        equity_weight: f64,
        #[arg(long, default_value_t = 20.0)]
        max_travel_penalty_s: f64,
        #[arg(long, default_value_t = 3.6)]
        calls_per_hr: f64,
        #[arg(long, default_value_t = 36.0)]
        service_mins: f64,
        #[arg(long, default_value_t = 5)]
        stations_budget: usize,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        pareto_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Solve 24-hour diurnal emergency call & BPR rush-hour travel-time profiles with separate Peak vs. Off-Peak dispatch tables
    Diurnal {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 3.6)]
        daily_calls_per_hr: f64,
        #[arg(long, default_value_t = 1.18)]
        rush_vc_scale: f64,
        #[arg(long, default_value_t = 0.55)]
        equity_weight: f64,
        #[arg(long, default_value_t = false)]
        daytime_12h_window: bool,
        #[arg(long, default_value_t = 5)]
        stations_budget: usize,
        #[arg(long)]
        hourly_csv_out: Option<PathBuf>,
        #[arg(long)]
        table_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Evaluate coupled 12.47 kV Electrical, Hazen-Williams Potable Water, and Manning/RDII Wastewater lifeline grids
    Utilities {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 12.5)]
        clock_hour: f64,
        #[arg(long, default_value_t = 18.0)]
        ev_pct: f64,
        #[arg(long, default_value_t = 22.0)]
        solar_pct: f64,
        #[arg(long, default_value_t = 1500.0)]
        fire_flow_gpm: f64,
        #[arg(long, default_value_t = 4.0)]
        storm_mm_hr: f64,
        #[arg(long, default_value_t = false)]
        trip_all_substations: bool,
        #[arg(long, default_value_t = false)]
        backup_generators: bool,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate Scawthorn / Hazus Fire-Following-Earthquake (FFE) ignitions & wind-driven conflagration coupled to <20 psi hydrants
    Ffe {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, default_value_t = 7.5)]
        mw: f64,
        #[arg(long, default_value_t = 12.0)]
        distance_km: f64,
        #[arg(long, default_value_t = 240.0)]
        vs30: f64,
        #[arg(long, default_value_t = 300)]
        trials: usize,
        #[arg(long, default_value_t = 10.0)]
        wind_mps: f64,
        #[arg(long, default_value_t = 0.0)]
        wind_deg: f64,
        #[arg(long, default_value_t = 180.0)]
        horizon_mins: f64,
        #[arg(long, default_value_t = 0.0)]
        pipe_replace_pct: f64,
        #[arg(long, default_value_t = 0.0)]
        cistern_pct: f64,
        #[arg(long, default_value_t = 0.0)]
        gas_shutoff_pct: f64,
        #[arg(long, default_value_t = false)]
        backup_power: bool,
        #[arg(long)]
        csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate post-earthquake Hospital ED Surge Capacity, Burn/Trauma Priority Triage Queueing & Dynamic Ambulance Diversion
    Surge {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = DiversionCliPolicy::Dynamic)]
        policy: DiversionCliPolicy,
        #[arg(long, default_value_t = 7.6)]
        mw: f64,
        #[arg(long, default_value_t = 11.0)]
        distance_km: f64,
        #[arg(long, default_value_t = 235.0)]
        vs30: f64,
        #[arg(long, default_value_t = 250)]
        trials: usize,
        #[arg(long, default_value_t = 10.0)]
        wind_mps: f64,
        #[arg(long, default_value_t = 6.0)]
        surge_hours: f64,
        #[arg(long, default_value_t = 35.0)]
        hosp_retrofit_pct: f64,
        #[arg(long, default_value_t = false)]
        backup_lifelines: bool,
        #[arg(long, default_value_t = false)]
        aba_burn_surge: bool,
        #[arg(long, default_value_t = false)]
        field_hospitals: bool,
        #[arg(long)]
        hosp_csv_out: Option<PathBuf>,
        #[arg(long)]
        parcel_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate post-earthquake Rotary-Wing Medevac Helipads, Koschmieder Smoke/Weather Flight Envelopes & Secondary WRBDC Burn Transfers
    Medevac {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = MedevacAvionicsCli::DayVfr)]
        avionics: MedevacAvionicsCli,
        #[arg(long, default_value_t = 3)]
        fleet: usize,
        #[arg(long, default_value_t = 7.7)]
        mw: f64,
        #[arg(long, default_value_t = 10.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 230.0)]
        vs30: f64,
        #[arg(long, default_value_t = 250)]
        trials: usize,
        #[arg(long, default_value_t = 11.5)]
        wind_mps: f64,
        #[arg(long, default_value_t = 1500.0)]
        ceiling_ft: f64,
        #[arg(long, default_value_t = 5.0)]
        visibility_sm: f64,
        #[arg(long, default_value_t = false)]
        harden_pads: bool,
        #[arg(long, default_value_t = true)]
        wrbdc_mutual_aid: bool,
        #[arg(long)]
        helipad_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate coupled FFE perimeter road closures, smoke visibility speed drop, staged neighborhood evacuation & arterial contraflow
    Evacuate {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = EvacStagingCli::Staged3Zone)]
        staging: EvacStagingCli,
        #[arg(long, value_enum, default_value_t = ContraflowCli::EmsProtected)]
        contraflow: ContraflowCli,
        #[arg(long, default_value_t = 7.7)]
        mw: f64,
        #[arg(long, default_value_t = 10.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 250)]
        trials: usize,
        #[arg(long, default_value_t = 13.0)]
        wind_mps: f64,
        #[arg(long, default_value_t = 4.5)]
        visibility_sm: f64,
        #[arg(long, default_value_t = 20.0)]
        stage2_delay_min: f64,
        #[arg(long, default_value_t = 45.0)]
        stage3_delay_min: f64,
        #[arg(long)]
        parcel_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate post-earthquake Emergency Shelter Siting, ATC-20/Lifeline Habitability, HEPA Smoke Filtration & Capacitated Evacuee Routing
    Shelters {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ShelterPolicyCli::CleanAir2Stage)]
        policy: ShelterPolicyCli,
        #[arg(long, value_enum, default_value_t = AirFilterCli::Merv13Hepa)]
        filtration: AirFilterCli,
        #[arg(long, default_value_t = 4)]
        open_shelters: usize,
        #[arg(long, default_value_t = 50.0)]
        sqft_per_person: f64,
        #[arg(long, default_value_t = 450.0)]
        hepa_cadr: f64,
        #[arg(long, default_value_t = 7.7)]
        mw: f64,
        #[arg(long, default_value_t = 10.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 250)]
        trials: usize,
        #[arg(long, default_value_t = 13.0)]
        wind_mps: f64,
        #[arg(long, default_value_t = false)]
        backup_lifelines: bool,
        #[arg(long)]
        shelter_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate post-earthquake multi-day shelter & hospital commodity resupply VRP (Water, Diesel, MREs & Medical Oxygen)
    Resupply {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ResupplyPolicyCli::ClarkeWrightIrp)]
        policy: ResupplyPolicyCli,
        #[arg(long, default_value_t = 7)]
        days: usize,
        #[arg(long, default_value_t = 36.0)]
        init_reserve_hrs: f64,
        #[arg(long, default_value_t = 2)]
        water_trucks: usize,
        #[arg(long, default_value_t = 2)]
        diesel_trucks: usize,
        #[arg(long, default_value_t = 2)]
        mre_trucks: usize,
        #[arg(long, default_value_t = 2)]
        oxygen_trucks: usize,
        #[arg(long, default_value_t = true)]
        heli_sling: bool,
        #[arg(long, default_value_t = 7.8)]
        mw: f64,
        #[arg(long, default_value_t = 9.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 200)]
        trials: usize,
        #[arg(long)]
        site_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate coupled GTFS transit-bus & ADA paratransit evacuation via Bus+EMS contraflow lanes into clean-air HEPA shelters
    TransitEvac {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = BusLaneModeCli::BusEmsContraflow)]
        lane_mode: BusLaneModeCli,
        #[arg(long, value_enum, default_value_t = TransitEvacPolicyCli::CleanAirTbep)]
        policy: TransitEvacPolicyCli,
        #[arg(long, default_value_t = 10)]
        buses: usize,
        #[arg(long, default_value_t = 6)]
        ada_vans: usize,
        #[arg(long, default_value_t = 8)]
        pickup_depots: usize,
        #[arg(long, default_value_t = 120.0)]
        window_min: f64,
        #[arg(long, default_value_t = 7.8)]
        mw: f64,
        #[arg(long, default_value_t = 9.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 200)]
        trials: usize,
        #[arg(long)]
        depot_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate post-earthquake cellular tower blackout fragility, WEA vs. siren/door-to-door mobilization S-curves & delayed staging
    Warning {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = TelecomBackupCli::Legacy3h)]
        telecom_backup: TelecomBackupCli,
        #[arg(long, value_enum, default_value_t = AlertChannelCli::MultiModal)]
        channels: AlertChannelCli,
        #[arg(long, value_enum, default_value_t = DepartureStagingCli::AdaptiveBlackout)]
        staging: DepartureStagingCli,
        #[arg(long, default_value_t = 4.5)]
        hours_post_quake: f64,
        #[arg(long, default_value_t = true)]
        night_event: bool,
        #[arg(long, default_value_t = 6)]
        sirens: usize,
        #[arg(long, default_value_t = 8)]
        patrol_units: usize,
        #[arg(long, default_value_t = 7.8)]
        mw: f64,
        #[arg(long, default_value_t = 9.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 200)]
        trials: usize,
        #[arg(long)]
        tower_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate FEMA IPAWS-OPEN (CAP v1.2, WEA 3.0, NPWS/PEP EAS, NOAA SAME), FNARS HF & Region 10 MERS ESF-2 communications
    FemaComms {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = EocUplinkCli::FullMersFnars)]
        eoc_uplink: EocUplinkCli,
        #[arg(long, value_enum, default_value_t = FemaPostureCli::FullIpaswMers)]
        posture: FemaPostureCli,
        #[arg(long, default_value_t = 46.0)]
        eas_radio_pct: f64,
        #[arg(long, default_value_t = 32.0)]
        nwr_same_pct: f64,
        #[arg(long, default_value_t = 3)]
        mers_units: usize,
        #[arg(long, default_value_t = 5.0)]
        hours_post_quake: f64,
        #[arg(long, default_value_t = true)]
        night_event: bool,
        #[arg(long, default_value_t = 8.0)]
        mw: f64,
        #[arg(long, default_value_t = 9.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 200)]
        trials: usize,
        #[arg(long)]
        asset_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate FEMA Incident Staging Bases (ISB), USACE IS-26 Type I-III POD queueing & IPAWS public POD routing
    FemaPods {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = PodRoutingCli::IpaswBalanced)]
        routing: PodRoutingCli,
        #[arg(long, value_enum, default_value_t = IsbPolicyCli::DynamicHotshot)]
        isb_policy: IsbPolicyCli,
        #[arg(long, default_value_t = 5)]
        pods: usize,
        #[arg(long, default_value_t = 10)]
        trailers_53ft: usize,
        #[arg(long, default_value_t = 68.0)]
        needy_pop_pct: f64,
        #[arg(long, default_value_t = true)]
        type1_super_pods: bool,
        #[arg(long, default_value_t = 7.9)]
        mw: f64,
        #[arg(long, default_value_t = 9.5)]
        distance_km: f64,
        #[arg(long, default_value_t = 200)]
        trials: usize,
        #[arg(long)]
        pod_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate FEMA Type 1 US&R Task Force USNG structural search grids, ESF-3 roadway debris clearance & ISB/POD reopening
    FemaUsar {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = UsarDebrisPolicyCli::TeethOfComb)]
        policy: UsarDebrisPolicyCli,
        #[arg(long, default_value_t = 2)]
        type1_task_forces: usize,
        #[arg(long, default_value_t = 6)]
        dozer_crews: usize,
        #[arg(long, default_value_t = 72.0)]
        horizon_hrs: f64,
        #[arg(long, default_value_t = 8.0)]
        mw: f64,
        #[arg(long, default_value_t = 9.0)]
        distance_km: f64,
        #[arg(long, default_value_t = 200)]
        trials: usize,
        #[arg(long)]
        grid_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate FEMA ESF-3 Generator Staging Base (249th Prime Power), TDMS Debris Sorting & DRC/MRIC Individual Assistance
    FemaRecovery {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = FemaRecoveryPolicyCli::IntegratedSuite)]
        policy: FemaRecoveryPolicyCli,
        #[arg(long, default_value_t = 4)]
        prt_teams: usize,
        #[arg(long, default_value_t = 14)]
        gen_packs: usize,
        #[arg(long, default_value_t = 2)]
        tdms_sites: usize,
        #[arg(long, default_value_t = 28)]
        dump_trucks: usize,
        #[arg(long, default_value_t = 2)]
        drcs: usize,
        #[arg(long, default_value_t = 3)]
        mrics: usize,
        #[arg(long, default_value_t = 8.0)]
        mw: f64,
        #[arg(long, default_value_t = 160)]
        trials: usize,
        #[arg(long)]
        facility_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate ALERTWest (alertwest.org) AI PTZ Wildfire Triangulation & NWCG Suppression
    Wildfire {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::FoothillCanyonWui)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        /// Optional path to an ALERTWest / western wildfire camera feed (.json, GeoJSON, or .csv)
        #[arg(long)]
        camera_feed: Option<PathBuf>,
        /// Optional path to write the built-in western wildfire camera JSON template
        #[arg(long)]
        export_sample_feed: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = WildfirePostureCli::FullAerial)]
        posture: WildfirePostureCli,
        #[arg(long, value_enum, default_value_t = WuiFuelCli::TimberFm10)]
        fuel: WuiFuelCli,
        #[arg(long, default_value_t = 24.0)]
        wind_mph: f64,
        #[arg(long, default_value_t = 14.0)]
        rh_pct: f64,
        #[arg(long, default_value_t = 22.0)]
        slope_pct: f64,
        #[arg(long, default_value_t = false)]
        night_nir: bool,
        #[arg(long, default_value_t = 2)]
        starlink_mdus: usize,
        #[arg(long, default_value_t = 3)]
        hotshot_crews: usize,
        #[arg(long, default_value_t = 2)]
        airtankers: usize,
        #[arg(long, default_value_t = 160)]
        trials: usize,
        #[arg(long)]
        camera_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate Post-Wildfire Burn-Scar Hydrophobic Runoff, USGS M1/Gartner Debris-Flow Basins & FHWA HDS-5 Culvert Washouts
    DebrisFlow {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = BaerPolicyCli::FullBaer)]
        baer_policy: BaerPolicyCli,
        #[arg(long, default_value_t = 36.0)]
        i15_mm_hr: f64,
        #[arg(long, default_value_t = 70.0)]
        storm_mm: f64,
        #[arg(long, default_value_t = 0.28)]
        soil_kf: f64,
        #[arg(long, default_value_t = 8)]
        basins: usize,
        #[arg(long, default_value_t = 160)]
        trials: usize,
        #[arg(long)]
        basin_csv_out: Option<PathBuf>,
        #[arg(long)]
        culvert_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Simulate Post-Wildfire Burn-Scar Ash/Turbidity Spikes, WTP Intake Shutoffs, Coagulant Dosing & Finished-Water Drawdown
    WtpTurbidity {
        #[arg(long, value_enum, default_value_t = BuiltinScenario::RiverfrontBridges)]
        scenario: BuiltinScenario,
        #[arg(long)]
        osm: Option<PathBuf>,
        #[arg(long)]
        load_snapshot: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = WtpPolicyCli::FerricPacWells)]
        policy: WtpPolicyCli,
        #[arg(long, default_value_t = 350.0)]
        shutoff_ntu: f64,
        #[arg(long, default_value_t = 16.0)]
        wtp_mgd: f64,
        #[arg(long, default_value_t = 12.5)]
        storage_mg: f64,
        #[arg(long, default_value_t = 5.5)]
        wells_mgd: f64,
        #[arg(long, default_value_t = 22.0)]
        coag_tons: f64,
        #[arg(long, default_value_t = 38.0)]
        i15_mm_hr: f64,
        #[arg(long, default_value_t = 160)]
        trials: usize,
        #[arg(long)]
        timeline_csv_out: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Compile an OpenStreetMap .osm / .osm.pbf file and verify lane/signal topology
    Import {
        #[arg(long)]
        input: PathBuf,
    },
}

fn build_world(
    scenario: BuiltinScenario,
    osm_path: Option<&PathBuf>,
    gtfs_path: Option<&PathBuf>,
    snapshot_path: Option<&PathBuf>,
    agents: Option<usize>,
) -> Result<SimulationWorld, String> {
    if let Some(snap_file) = snapshot_path {
        let (mut world, _plume) =
            SimulationWorld::load_snapshot_json(snap_file).map_err(|e| e.to_string())?;
        if let Some(n) = agents {
            world.seed_50k_megacity_crowd(n);
        }
        return Ok(world);
    }

    let ds = if let Some(p) = osm_path {
        load_osm_path(p).map_err(|e| e.to_string())?
    } else {
        let xml = match scenario {
            BuiltinScenario::RiverfrontBridges => builtin_riverfront_bridges_osm(),
            BuiltinScenario::HarborPeninsula => builtin_harbor_peninsula_osm(),
            BuiltinScenario::RadialRingPlazas => builtin_radial_ring_plazas_osm(),
            BuiltinScenario::FoothillCanyonWui => builtin_foothill_canyon_wui_osm(),
            BuiltinScenario::Metropolis50k => builtin_metropolis_50k_osm(),
            BuiltinScenario::Roundabout => builtin_roundabout_osm(),
            BuiltinScenario::Braess => builtin_braess_diamond_osm(),
        };
        parse_osm_xml_str(&xml).map_err(|e| e.to_string())?
    };
    let mut world = SimulationWorld::new(RoadNetwork::compile(&ds));
    if let Some(gdir) = gtfs_path {
        world.load_gtfs_directory(gdir)?;
    }
    if let Some(n) = agents {
        world.seed_50k_megacity_crowd(n);
    } else if scenario == BuiltinScenario::Metropolis50k {
        world.seed_50k_megacity_crowd(52_000);
    }
    Ok(world)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Gui {
        scenario: BuiltinScenario::RiverfrontBridges,
        osm: None,
        gtfs: None,
        load_snapshot: None,
        agents: None,
        max_pressure: false,
        evp_units: 0,
        vsync: false,
    }) {
        Commands::Gui {
            scenario,
            osm,
            gtfs,
            load_snapshot,
            agents,
            max_pressure,
            evp_units,
            vsync,
        } => {
            let mut world = build_world(
                scenario,
                osm.as_ref(),
                gtfs.as_ref(),
                load_snapshot.as_ref(),
                agents,
            )?;
            if max_pressure {
                world.network.enable_network_max_pressure();
            }
            for _ in 0..evp_units {
                let _ = world.dispatch_emergency_responder();
            }
            #[cfg(feature = "gui")]
            {
                if let Some(ref out_path) = cli.screenshot {
                    gui::render_headless_screenshot(world, out_path, cli.screenshot_view)
                        .map_err(|e| format!("Headless screenshot error: {e}"))?;
                } else {
                    gui::run_gui(world, vsync).map_err(|e| format!("GUI error: {e}"))?;
                }
            }
            #[cfg(not(feature = "gui"))]
            {
                let _ = (world, vsync);
                eprintln!(
                    "Compiled without `gui` feature. Use `simulate` or recompile with `--features gui`."
                );
            }
        }
        Commands::Tui {
            scenario,
            osm,
            gtfs,
            load_snapshot,
            max_pressure,
        } => {
            let mut world = build_world(
                scenario,
                osm.as_ref(),
                gtfs.as_ref(),
                load_snapshot.as_ref(),
                None,
            )?;
            if max_pressure {
                world.network.enable_network_max_pressure();
            }
            #[cfg(feature = "tui")]
            {
                tui::run_tui(world)?;
            }
            #[cfg(not(feature = "tui"))]
            {
                let _ = world;
                eprintln!("Compiled without `tui` feature.");
            }
        }
        Commands::Simulate {
            scenario,
            osm,
            gtfs,
            load_snapshot,
            seconds,
            agents,
            tod_upzone,
            max_pressure,
            json_out,
            save_snapshot,
            csv_out,
        } => {
            let mut world = build_world(
                scenario,
                osm.as_ref(),
                gtfs.as_ref(),
                load_snapshot.as_ref(),
                agents,
            )?;
            if max_pressure {
                world.network.enable_network_max_pressure();
            }
            if tod_upzone {
                apply_tod_upzoning_policy(&mut world.parcels, &mut world.network);
                world.transit_policy.headway_min = 5.0;
                world.rebuild_cyclists();
                world.refresh_urban_metrics();
            }
            let steps = ((seconds / 0.1).round() as usize).max(10);
            for _ in 0..steps {
                world.step(0.1);
            }
            if let Some(snap_out) = save_snapshot {
                let include_ag = world.total_active_agents() <= 20_000;
                world.save_snapshot_json(
                    &snap_out,
                    PlumePhysicsParams::default(),
                    "CLI Simulation Snapshot",
                    include_ag,
                )?;
            }
            if let Some(csv_path) = csv_out {
                world.export_telemetry_csv(&csv_path)?;
            }
            let rep = world.summary_report(if tod_upzone {
                "15-Min TOD Suite"
            } else {
                "Baseline"
            });
            let pretty = serde_json::to_string_pretty(&rep)?;
            println!("{pretty}");
            if let Some(out) = json_out {
                fs::write(out, pretty)?;
            }
        }
        Commands::Compare {
            scenario,
            osm,
            gtfs,
            seconds,
            csv_baseline,
            csv_tod,
        } => {
            let mut base = build_world(scenario, osm.as_ref(), gtfs.as_ref(), None, None)?;
            let mut tod = build_world(scenario, osm.as_ref(), gtfs.as_ref(), None, None)?;
            base.transit_gtfs.fare_mode = ug_core::BoardingFareMode::FrontDoorCash;
            tod.transit_gtfs.fare_mode = ug_core::BoardingFareMode::AllDoorFareFreeBrt;
            tod.transit_gtfs.coach_spec = ug_core::BusCoachSpec::Articulated60FtBrt;
            tod.transit_gtfs.hold_early_buses = true;
            tod.network.enable_network_max_pressure();
            apply_tod_upzoning_policy(&mut tod.parcels, &mut tod.network);
            tod.transit_policy.headway_min = 5.0;
            tod.rebuild_cyclists();
            tod.refresh_urban_metrics();

            let steps = ((seconds / 0.1).round() as usize).max(10);
            for _ in 0..steps {
                base.step(0.1);
                tod.step(0.1);
            }
            if let Some(cb) = csv_baseline {
                base.export_telemetry_csv(&cb)?;
            }
            if let Some(ct) = csv_tod {
                tod.export_telemetry_csv(&ct)?;
            }
            let rb = base.summary_report("Status Quo Baseline");
            let rt = tod.summary_report("TOD + BRT + MaxPressure");
            println!("{:<28} | {:>18} | {:>18}", "Metric", rb.label, rt.label);
            println!("{}", "-".repeat(70));
            println!(
                "{:<28} | {:>17.1}% | {:>17.1}%",
                "Auto Mode Share",
                rb.mode_shares.auto * 100.0,
                rt.mode_shares.auto * 100.0
            );
            println!(
                "{:<28} | {:>17.1}% | {:>17.1}%",
                "Transit + Bike + Walk",
                (1.0 - rb.mode_shares.auto) * 100.0,
                (1.0 - rt.mode_shares.auto) * 100.0
            );
            println!(
                "{:<28} | {:>14.1} mph | {:>14.1} mph",
                "Mean Traffic Speed", rb.mean_speed_mph, rt.mean_speed_mph
            );
            println!(
                "{:<28} | {:>15.1} s | {:>15.1} s",
                "Mean GTFS Stop Dwell",
                rb.transit_telemetry.mean_dwell_s,
                rt.transit_telemetry.mean_dwell_s
            );
            println!(
                "{:<28} | {:>11} (LOS {}) | {:>11} (LOS {})",
                "GTFS Boardings & Load LOS",
                rb.transit_telemetry.cumulative_boardings,
                rb.transit_telemetry.tcqsm_crowding_los,
                rt.transit_telemetry.cumulative_boardings,
                rt.transit_telemetry.tcqsm_crowding_los
            );
            println!(
                "{:<28} | {:>13.1} kg/h | {:>13.1} kg/h",
                "VSP Tailpipe CO2", rb.environment.co2_kg_per_hr, rt.environment.co2_kg_per_hr
            );
            println!(
                "{:<28} | {:>+12.1}k/ac/y | {:>+12.1}k/ac/y",
                "Net Municipal Fiscal",
                rb.ledger.net_fiscal_per_acre_k_yr,
                rt.ledger.net_fiscal_per_acre_k_yr
            );
        }
        Commands::Batch {
            scenario,
            osm,
            gtfs,
            load_snapshot,
            seconds,
            warmup,
            demand_scales,
            replicates,
            roundabout_radius,
            roundabout_lanes,
            max_roundabouts,
            tod_upzone,
            threads,
            csv_out,
            json_out,
        } => {
            let scales = batch::parse_demand_scales_arg(&demand_scales)?;
            let base_world = build_world(
                scenario,
                osm.as_ref(),
                gtfs.as_ref(),
                load_snapshot.as_ref(),
                None,
            )?;
            let cfg = BatchExperimentConfig {
                duration_s: seconds,
                warmup_s: warmup,
                dt_s: 0.1,
                demand_multipliers: scales,
                replicates,
                roundabout_radius_m: roundabout_radius,
                roundabout_lanes,
                max_roundabouts,
                include_tod_upzone: tod_upzone,
                base_seed: 0x9E37_79B9_7F4A_7C15,
            };
            batch::execute_batch_cli(
                &base_world,
                &cfg,
                threads,
                &csv_out,
                json_out.as_deref(),
            )?;
        }
        Commands::Mclp {
            scenario,
            osm,
            load_snapshot,
            target_pct,
            threshold_s,
            max_stations,
            mexclp,
            busy_prob,
            endogenous_queue,
            hypercube,
            max_engines,
            backup_target_pct,
            keep_existing,
            csv_out,
            json_out,
        } => {
            let world = build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(800))?;
            let obj_model = if !mexclp {
                SitingObjectiveModel::DeterministicMclp1974
            } else if backup_target_pct > 0.0 {
                SitingObjectiveModel::MexclpPlusHoganBackup1986
            } else {
                SitingObjectiveModel::DaskinMexclp1983
            };
            let cfg = MclpSitingConfig {
                target_coverage_pct: target_pct,
                threshold_travel_s: threshold_s,
                max_stations_budget: max_stations,
                keep_existing_fire_stations: keep_existing,
                objective_model: obj_model,
                busyness_model: if hypercube {
                    BusynessModel::LarsonHypercube1974
                } else if endogenous_queue {
                    BusynessModel::EndogenousWorkloadQueue
                } else {
                    BusynessModel::FixedBusyFraction
                },
                apparatus_busy_prob: busy_prob,
                max_engines_per_station: max_engines,
                target_second_due_cov_pct: backup_target_pct.max(70.0),
                ..MclpSitingConfig::default()
            };
            let rep = world.solve_mclp_station_siting(&cfg);
            println!(
                "=== urban-grid v2.6 MEXCLP & Second-Due Backup Siting ({:.1} ms | {} candidates | {}) ===",
                rep.solve_time_ms,
                rep.candidate_pool_size,
                cfg.objective_model.label()
            );
            for pt in &rep.pareto_curve {
                println!(
                    "  M={:>2} units ({:>2} stns, q={:.2}) -> k>=1 {:>5.1}% | E[Cov] {:>5.1}% | 2nd-Due {:>5.1}% | E[T] {:>5.1}s | 2ndP90 {:>5.1}s",
                    pt.total_apparatus,
                    pt.total_fire_stations,
                    pt.effective_busy_prob_q,
                    pt.parcels_within_threshold_pct,
                    pt.expected_cov_pct,
                    pt.second_due_backup_cov_pct,
                    pt.expected_travel_s,
                    pt.second_due_p90_travel_s
                );
            }
            println!(
                "Optimal Fleet: {} units across {} stations (target met: {}) -> E[Cov]={:.1}% | 2nd-Due={:.1}% | k>=1={:.1}%",
                rep.optimal_total_apparatus,
                rep.optimal_total_stations,
                rep.target_met,
                rep.achieved_expected_cov_pct,
                rep.achieved_second_due_cov_pct,
                rep.achieved_parcel_cov_pct
            );
            for (i, site) in rep.selected_sites.iter().enumerate() {
                println!(
                    "  [Station #{}] {} engine(s) on Corridor #{} `{}` at ({:.1}, {:.1}) — standalone {:.1}% pop cov",
                    i + 1,
                    site.allocated_engines,
                    site.segment_id,
                    site.corridor_name,
                    site.world_pos.x,
                    site.world_pos.y,
                    site.standalone_resident_cov_pct
                );
            }
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} MEXCLP Pareto rows -> {:?}", cout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::StressTest {
            scenario,
            osm,
            load_snapshot,
            target_pct,
            threshold_s,
            auto_site_mclp,
            harden_mclp,
            n_minus_2,
            csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(800))?;
            let mclp_cfg = MclpSitingConfig {
                target_coverage_pct: target_pct,
                threshold_travel_s: threshold_s,
                ..MclpSitingConfig::default()
            };
            let stress_cfg = ResilienceStressConfig {
                interdiction_depth: if n_minus_2 {
                    DisruptionInterdictionDepth::DualLinkNMinus2
                } else {
                    DisruptionInterdictionDepth::SingleLinkNMinus1
                },
                target_coverage_pct: target_pct,
                threshold_travel_s: threshold_s,
                ..ResilienceStressConfig::default()
            };

            if auto_site_mclp {
                let sited = if harden_mclp {
                    world.solve_disruption_hardened_mclp(&mclp_cfg, &stress_cfg)
                } else {
                    world.solve_mclp_station_siting(&mclp_cfg)
                };
                world.apply_mclp_siting_solution(&sited);
            }

            let rep = world.run_resilience_stress_test(&stress_cfg);
            println!(
                "=== urban-grid v2.3 Arterial Bridge/Corridor Closure Stress-Test ({:.1} ms | {} scenarios | {} stations) ===",
                rep.solve_time_ms, rep.evaluated_scenarios, rep.fire_station_count
            );
            println!(
                "  Intact Baseline : {:.1}% parcels | {:.1}% residents <= {:.0}s | P90 {:.1}s",
                rep.baseline_parcel_cov_pct,
                rep.baseline_resident_cov_pct,
                threshold_s,
                rep.baseline_p90_travel_s
            );
            println!(
                "  Worst Disruption: {:.1}% parcels (-{:.1} pts) | P90 +{:.1}s | Pass Rate >= {:.0}%: {:.1}% ({}/{})",
                rep.worst_case_parcel_cov_pct,
                rep.max_coverage_drop_pct_pts,
                rep.max_p90_surge_s,
                target_pct,
                rep.resilience_pass_rate_pct,
                rep.scenarios_retaining_target,
                rep.evaluated_scenarios
            );
            for v in rep.top_vulnerabilities.iter().take(8) {
                println!(
                    "  [Rank #{:>2} | N-{}] {} -> {:.1}% cov (-{:.1} pts) | P90 {:>5.1}s | NRI {:>6.0} | Isolated {}",
                    v.rank,
                    v.interdiction_order,
                    v.label,
                    v.post_parcel_cov_pct,
                    v.coverage_drop_pct_pts,
                    v.post_p90_travel_s,
                    v.nri_system_delay_parcel_s,
                    v.isolated_parcels
                );
            }
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} ranked vulnerability rows -> {:?}", cout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::SeismicMc {
            scenario,
            osm,
            load_snapshot,
            mw,
            distance_km,
            vs30,
            trials,
            retrofit_top_k,
            crew_speedup,
            pipe_replace_pct,
            pipe_crews,
            hazus_pgv_law,
            backup_power,
            auto_site_mclp,
            csv_out,
            recovery_csv_out,
            pipeline_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(800))?;
            if auto_site_mclp {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig::default());
                world.apply_mclp_siting_solution(&sited);
            }
            let scfg = SeismicScenarioConfig {
                moment_magnitude_mw: mw,
                rupture_standoff_km: distance_km,
                vs30_m_per_s: vs30,
                monte_carlo_trials: trials,
                retrofit_top_k_bridges: retrofit_top_k,
                repair_crew_speedup: crew_speedup,
                pipe_fragility_law: if hazus_pgv_law {
                    PipelineFragilityLaw::HazusORourkeAyala1993
                } else {
                    PipelineFragilityLaw::Ala2001LinearPgv
                },
                seismic_pipe_replacement_pct: pipe_replace_pct,
                water_sewer_repair_crews: pipe_crews,
                critical_lifeline_backup_power: backup_power,
                ..SeismicScenarioConfig::default()
            };
            let rep = world.run_seismic_monte_carlo(&scfg);
            println!(
                "=== urban-grid v2.4 Seismic Bridge Fragility & Monte Carlo ({:.1} ms | {} trials | Mw {:.1} @ {:.1} km) ===",
                rep.solve_time_ms, rep.config.monte_carlo_trials, mw, distance_km
            );
            println!(
                "  Intact Baseline : {:.1}% cov (P90 {:.1}s) | Mean Bridge Sa(1.0s): {:.2}g ({} bridges, {} retrofitted)",
                rep.baseline_parcel_cov_pct,
                rep.baseline_p90_travel_s,
                rep.mean_median_sa10_g,
                rep.bridge_count,
                retrofit_top_k.min(rep.bridge_count)
            );
            println!(
                "  Post-Quake Cov  : Mean {:.1}% | P50 {:.1}% | P10 {:.1}% | P05 {:.1}% | P(>=90%) = {:.1}%",
                rep.mean_parcel_cov_pct,
                rep.p50_parcel_cov_pct,
                rep.p10_parcel_cov_pct,
                rep.p05_parcel_cov_pct,
                rep.prob_meeting_nfpa_target_pct
            );
            println!(
                "  Post-Quake P90  : Mean {:.1}s | P90-tail {:.1}s | P95-tail {:.1}s | E[Closed Bridges] = {:.2}",
                rep.mean_p90_travel_s,
                rep.p90_p90_travel_s,
                rep.p95_p90_travel_s,
                rep.expected_closed_bridges
            );
            for b in rep.bridge_inventory.iter().take(8) {
                println!(
                    "  [Bridge #{}] {} ({}) -> Sa(1s)={:.2}g | MC Closure={:.1}% (Ext {:.1}% + Comp {:.1}%)",
                    b.bridge_id + 1,
                    b.label,
                    b.effective_archetype.short_label(),
                    b.median_site_sa10_g,
                    b.mc_closure_rate_pct,
                    b.mc_state_pcts[3],
                    b.mc_state_pcts[4]
                );
            }
            println!(
                "  Hazus 90-Day Recovery (Crew {:.1}x) | Bruneau Resilience Index R90 = {:.1}% | >=90% Cov in {:?} d:",
                crew_speedup,
                rep.bruneau_resilience_index_90d_pct,
                rep.days_to_90pct_mean_recovery
            );
            println!(
                "  Buried Pipelines ({:.1} km | Median PGV {:.1} cm/s): Day-0 Water Breaks={:.2}, Leaks={:.2} | Sewer Breaks={:.2} | Sub Outages={:.2}",
                rep.total_water_pipe_km,
                rep.median_corridor_pgv_cm_s,
                rep.expected_day0_water_breaks,
                rep.expected_day0_water_leaks,
                rep.expected_day0_sewer_breaks,
                rep.expected_day0_substation_outages
            );
            println!(
                "  90-Day Lifeline Resilience R90: Emergency {:.1}% | Electric {:.1}% | Water>=20psi {:.1}% | Sewer {:.1}% | Multi-Lifeline {:.1}%",
                rep.bruneau_resilience_index_90d_pct,
                rep.electric_resilience_index_90d_pct,
                rep.water_resilience_index_90d_pct,
                rep.wastewater_resilience_index_90d_pct,
                rep.multi_lifeline_resilience_90d_pct
            );
            for ep in &rep.recovery_timeline {
                if ep.day == 0.0 || ep.is_hazus_primary_benchmark {
                    println!(
                        "    {:>14}: Cov {:>5.1}% | Elec {:>5.1}% | Watr>=20psi {:>5.1}% | Sewr {:>5.1}% | RemRep W={:.1}/S={:.1}",
                        ep.label,
                        ep.mean_parcel_cov_pct,
                        ep.mean_electric_service_pct,
                        ep.mean_water_20psi_service_pct,
                        ep.mean_wastewater_service_pct,
                        ep.expected_water_repairs_remaining,
                        ep.expected_sewer_repairs_remaining
                    );
                }
            }
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} bridge fragility rows -> {:?}", cout);
            }
            if let Some(rcout) = recovery_csv_out {
                let n = rep.export_recovery_csv_path(&rcout)?;
                println!("Exported {n} Hazus recovery epochs -> {:?}", rcout);
            }
            if let Some(pcout) = pipeline_csv_out {
                let n = rep.export_pipeline_csv_path(&pcout)?;
                println!("Exported {n} buried pipeline fragility rows -> {:?}", pcout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Hypercube {
            scenario,
            osm,
            load_snapshot,
            calls_per_hr,
            service_mins,
            loss_mode,
            calibrate_mu,
            stations_budget,
            csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(800))?;
            if world.emergency_coverage.fire_station_count == 0 {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    busyness_model: BusynessModel::LarsonHypercube1974,
                    system_calls_per_hr: calls_per_hr,
                    mean_service_duration_mins: service_mins,
                    max_stations_budget: stations_budget,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let hcfg = HypercubeConfig {
                system_calls_per_hr: calls_per_hr,
                base_service_duration_mins: service_mins,
                calibrate_travel_service_rate: calibrate_mu,
                queue_discipline: if loss_mode {
                    HypercubeQueueDiscipline::ZeroCapacityMutualAidLoss
                } else {
                    HypercubeQueueDiscipline::InfiniteCapacityFcfsQueue
                },
                ..HypercubeConfig::default()
            };
            let rep = world.solve_active_fleet_hypercube(&hcfg);
            println!(
                "=== urban-grid v2.7 Larson (1974) Exact M/M/N Spatial Hypercube ({:.2} ms | N={} | {} states | {} sweeps) ===",
                rep.solve_time_ms,
                rep.num_units,
                rep.state_space_size,
                rep.gauss_seidel_iterations
            );
            println!(
                "  Conservation Err: {:.2e} | P0 Idle: {:.1}% | P(All Busy/Q): {:.2}% | W_q: {:.1}s | E[Resp]: {:.1}s",
                rep.rate_conservation_error,
                rep.prob_all_idle_p0 * 100.0,
                rep.prob_all_busy_or_queued * 100.0,
                rep.mean_queue_wait_s,
                rep.mean_total_response_s
            );
            println!(
                "  Busy Fraction   : Mean {:.1}% (Min {:.1}% .. Max {:.1}% | Imbalance {:.1} pts) | Cross-District: {:.1}%",
                rep.mean_busy_fraction_rho * 100.0,
                rep.min_busy_fraction_rho * 100.0,
                rep.max_busy_fraction_rho * 100.0,
                rep.workload_imbalance_spread * 100.0,
                rep.cross_district_dispatch_pct
            );
            for u in &rep.unit_metrics {
                println!(
                    "    [Unit #{}] {:<28} -> rho={:>5.1}% | mu={:.2}/hr | 1st={:>4.1}% | Bkp={:>4.1}% | E[T]={:>5.1}s",
                    u.unit_idx + 1,
                    u.label,
                    u.busy_fraction_rho * 100.0,
                    u.service_rate_mu_per_hr,
                    u.primary_dispatch_share_pct,
                    u.backup_dispatch_share_pct,
                    u.mean_unit_travel_s
                );
            }
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} hypercube unit rows -> {:?}", cout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Redistrict {
            scenario,
            osm,
            load_snapshot,
            equity_weight,
            max_travel_penalty_s,
            calls_per_hr,
            service_mins,
            stations_budget,
            csv_out,
            pareto_csv_out,
            json_out,
        } => {
            let mut world = build_world(
                scenario,
                osm.as_ref(),
                None,
                load_snapshot.as_ref(),
                Some(0),
            )?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    busyness_model: BusynessModel::LarsonHypercube1974,
                    system_calls_per_hr: calls_per_hr,
                    mean_service_duration_mins: service_mins,
                    max_stations_budget: stations_budget.max(2),
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let obj_mode = if equity_weight <= 1e-6 {
                RedistrictingObjectiveMode::TravelTimeOptimalCarter1972
            } else if equity_weight >= 0.90 {
                RedistrictingObjectiveMode::StrictWorkloadEqualization
            } else {
                RedistrictingObjectiveMode::BalancedParetoKnee
            };
            let rcfg = RedistrictingConfig {
                hypercube: HypercubeConfig {
                    system_calls_per_hr: calls_per_hr,
                    base_service_duration_mins: service_mins,
                    ..HypercubeConfig::default()
                },
                objective_mode: obj_mode,
                workload_equity_weight: equity_weight,
                max_travel_penalty_s,
                ..RedistrictingConfig::default()
            };
            let rep = world.optimize_response_districts(&rcfg);
            let _ = world.apply_redistricting_plan_tracked(&rep);
            println!(
                "=== urban-grid v2.8 Carter, Chaiken & Ignall (1972) / Larson Hypercube Redistricting ({:.2} ms | {} evals) ===",
                rep.solve_time_ms, rep.hypercube_evaluations
            );
            println!(
                "  Workload Spread : {:.2}% -> {:.2}% ({:.1}% reduction) | RMSE: {:.2}% -> {:.2}%",
                rep.voronoi_workload_spread * 100.0,
                rep.optimized_workload_spread * 100.0,
                rep.workload_spread_reduction_pct,
                rep.voronoi_workload_rmse * 100.0,
                rep.optimized_workload_rmse * 100.0
            );
            println!(
                "  Mean Travel E[T]: {:.2}s -> {:.2}s ({:+.2}s) | Carter-1972 Pure Travel Gain: -{:.2}s | Shifted: {}/{} parcels ({:.1}% demand)",
                rep.voronoi_hypercube.mean_dispatch_travel_s,
                rep.optimized_hypercube.mean_dispatch_travel_s,
                rep.travel_time_delta_s,
                rep.carter_min_travel_gain_s,
                rep.reassigned_parcel_count,
                rep.num_parcels,
                rep.reassigned_demand_pct
            );
            for u in &rep.unit_summaries {
                println!(
                    "    [Unit #{}] {:<28} -> s={:+6.1}s | Beats {:>2}->{:>2} | rho {:.1}% -> {:.1}% | E[T]={:.1}s",
                    u.unit_idx + 1,
                    u.label,
                    u.boundary_offset_s,
                    u.voronoi_parcel_count,
                    u.optimized_parcel_count,
                    u.voronoi_busy_rho * 100.0,
                    u.optimized_busy_rho * 100.0,
                    u.mean_unit_travel_s
                );
            }
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} redistricted unit rows -> {:?}", cout);
            }
            if let Some(pout) = pareto_csv_out {
                let n = rep.export_pareto_csv_path(&pout)?;
                println!("Exported {n} Pareto frontier rows -> {:?}", pout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Diurnal {
            scenario,
            osm,
            load_snapshot,
            daily_calls_per_hr,
            rush_vc_scale,
            equity_weight,
            daytime_12h_window,
            stations_budget,
            hourly_csv_out,
            table_csv_out,
            json_out,
        } => {
            let mut world = build_world(
                scenario,
                osm.as_ref(),
                None,
                load_snapshot.as_ref(),
                Some(0),
            )?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    busyness_model: BusynessModel::LarsonHypercube1974,
                    system_calls_per_hr: daily_calls_per_hr,
                    max_stations_budget: stations_budget.max(2),
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let dcfg = DiurnalProfileConfig {
                window_preset: if daytime_12h_window {
                    DiurnalWindowPreset::DaytimeCommercial12Hr
                } else {
                    DiurnalWindowPreset::BimodalCommutePeak8Hr
                },
                daily_mean_calls_per_hr: daily_calls_per_hr,
                rush_vc_scale,
                redistrict: RedistrictingConfig {
                    workload_equity_weight: equity_weight,
                    ..RedistrictingConfig::default()
                },
                ..DiurnalProfileConfig::default()
            };
            let rep = world.solve_diurnal_dispatch_tables(&dcfg);
            let _ = world.apply_diurnal_dispatch_tables_tracked(&rep, DiurnalDispatchRegime::PeakRush);
            println!(
                "=== urban-grid v2.9 Diurnal 24-Hr Profiles & Peak/Off-Peak Dispatch Tables ({:.2} ms | {} units | {} parcels) ===",
                rep.solve_time_ms, rep.num_units, rep.num_parcels
            );
            println!(
                "  Peak Table   ({:>2} hrs): {:.2} calls/hr | Arterial V/C={:.2} (+{:.1}% BPR) | E[T]={:.1}s | Spread={:.1} pts",
                rep.peak_table.active_hours.len(),
                rep.peak_table.mean_calls_per_hr,
                rep.peak_table.mean_arterial_vc,
                rep.peak_table.mean_bpr_slowdown_pct,
                rep.peak_table.redistricting_report.optimized_hypercube.mean_dispatch_travel_s,
                rep.peak_table.redistricting_report.optimized_workload_spread * 100.0
            );
            println!(
                "  Off-Peak Tbl ({:>2} hrs): {:.2} calls/hr | Arterial V/C={:.2} (+{:.1}% BPR) | E[T]={:.1}s | Spread={:.1} pts",
                rep.offpeak_table.active_hours.len(),
                rep.offpeak_table.mean_calls_per_hr,
                rep.offpeak_table.mean_arterial_vc,
                rep.offpeak_table.mean_bpr_slowdown_pct,
                rep.offpeak_table.redistricting_report.optimized_hypercube.mean_dispatch_travel_s,
                rep.offpeak_table.redistricting_report.optimized_workload_spread * 100.0
            );
            println!(
                "  24h Green-Kolesar PSA : Voronoi Spread {:.2}% -> Dual-Table {:.2}% | E[T] {:.1}s -> {:.1}s | Beat Switches: {} primary ({:.1}% demand), {} 1st/2nd-due",
                rep.daily_voronoi_workload_spread * 100.0,
                rep.daily_dual_table_workload_spread * 100.0,
                rep.daily_voronoi_mean_travel_s,
                rep.daily_dual_table_mean_travel_s,
                rep.primary_beat_switches_count,
                rep.switched_demand_share_pct,
                rep.any_rank12_switches_count
            );
            if let Some(hout) = hourly_csv_out {
                let n = rep.export_hourly_csv_path(&hout)?;
                println!("Exported {n} diurnal hourly PSA rows -> {:?}", hout);
            }
            if let Some(tout) = table_csv_out {
                let n = rep.export_dispatch_table_csv_path(&tout)?;
                println!("Exported {n} Peak/Off-Peak parcel dispatch table rows -> {:?}", tout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Utilities {
            scenario,
            osm,
            load_snapshot,
            clock_hour,
            ev_pct,
            solar_pct,
            fire_flow_gpm,
            storm_mm_hr,
            trip_all_substations,
            backup_generators,
            csv_out,
            json_out,
        } => {
            let mut world = build_world(
                scenario,
                osm.as_ref(),
                None,
                load_snapshot.as_ref(),
                Some(0),
            )?;
            world.diurnal_clock_hour = clock_hour;
            let ucfg = UtilityGridsConfig {
                ev_charging_penetration_pct: ev_pct,
                solar_pv_penetration_pct: solar_pct,
                active_fire_flow_demand_gpm: fire_flow_gpm,
                storm_rainfall_mm_hr: storm_mm_hr,
                tripped_substation_indices: if trip_all_substations {
                    vec![0, 1]
                } else {
                    Vec::new()
                },
                critical_backup_generators_enabled: backup_generators,
                ..UtilityGridsConfig::default()
            };
            let rep = world.evaluate_utility_grids(&ucfg);
            println!(
                "=== urban-grid v3.0 Coupled Electrical, Water & Wastewater Grids ({:.2} ms | Composite Health {:.1}%) ===",
                rep.solve_time_ms, rep.composite_lifeline_score_pct
            );
            println!(
                "  [ELECTRICAL 12.47kV] Energized: {:.1}% | V = {:.3}..{:.3} p.u. (ANSI C84.1: {:.1}%) | Load {:.2} MW - PV {:.2} MW | Losses {:.1} kW ({:.2}%) | Dark Signals: {}",
                rep.electrical.energized_nodes_pct,
                rep.electrical.min_voltage_pu,
                rep.electrical.max_voltage_pu,
                rep.electrical.ansi_c84_1_compliance_pct,
                rep.electrical.total_load_mw,
                rep.electrical.solar_pv_generation_mw,
                rep.electrical.line_resistive_losses_kw,
                rep.electrical.loss_pct,
                rep.electrical.darkened_signal_junctions_count
            );
            println!(
                "  [POTABLE WATER]      Boosters: {} | Pressure = {:.1}..{:.1} psi (Mean {:.1} psi | >=20psi Fire: {:.1}%) | Supply {:.1} L/s (Fire {:.1} L/s) | Pump {:.1} kW",
                if rep.water.booster_pumps_energized { "ONLINE" } else { "OFF (GRAVITY)" },
                rep.water.min_residual_pressure_psi,
                rep.water.max_residual_pressure_psi,
                rep.water.mean_residual_pressure_psi,
                rep.water.nodes_meeting_20psi_fire_pct,
                rep.water.total_supply_lps,
                rep.water.fire_hydrant_demand_lps,
                rep.water.pumping_power_kw
            );
            println!(
                "  [WASTEWATER SEWER]   WWTP: {} ({:.1}% cap) | DWF {:.1} L/s + Storm RDII {:.1} L/s = {:.1} L/s | Max d/D = {:.2} | SSO Nodes: {} ({:.1} L/s)",
                if rep.wastewater.wwtp_energized { "ONLINE" } else { "TRIPPED" },
                rep.wastewater.wwtp_utilization_pct,
                rep.wastewater.dry_weather_sanitary_lps,
                rep.wastewater.wet_weather_rdii_lps,
                rep.wastewater.total_wwtp_influent_lps,
                rep.wastewater.max_pipe_d_over_d,
                rep.wastewater.surcharged_sso_nodes_count,
                rep.wastewater.estimated_sso_overflow_lps
            );
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} utility junction rows -> {:?}", cout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Ffe {
            scenario,
            osm,
            load_snapshot,
            mw,
            distance_km,
            vs30,
            trials,
            wind_mps,
            wind_deg,
            horizon_mins,
            pipe_replace_pct,
            cistern_pct,
            gas_shutoff_pct,
            backup_power,
            csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let fcfg = FfeConfig {
                seismic: SeismicScenarioConfig {
                    moment_magnitude_mw: mw,
                    rupture_standoff_km: distance_km,
                    vs30_m_per_s: vs30,
                    monte_carlo_trials: trials,
                    seismic_pipe_replacement_pct: pipe_replace_pct,
                    critical_lifeline_backup_power: backup_power,
                    ..SeismicScenarioConfig::default()
                },
                ignition_model: FfeIgnitionModel::Scawthorn2005QuadraticPga,
                wind_speed_mps: wind_mps,
                wind_direction_deg: wind_deg,
                conflagration_horizon_mins: horizon_mins,
                initial_attack_flashover_s: 360.0,
                seismic_gas_shutoff_pct: gas_shutoff_pct,
                auxiliary_cistern_coverage_pct: cistern_pct,
            };
            let rep = world.run_ffe_monte_carlo(&fcfg);
            println!(
                "=== urban-grid v3.2 Scawthorn / Hazus FFE & Wind Conflagration ({:.1} ms | {} trials | Mw {:.1} | Wind {:.1} m/s) ===",
                rep.solve_time_ms, trials, mw, wind_mps
            );
            println!(
                "  Exposure & Fleet : {} parcels ({:.2}M sq ft, ${:.0}M value) | {} engines | Mean Hydrant {:.1} psi ({:.1}% <20psi dry)",
                rep.total_parcels,
                rep.total_floor_area_mmsqft,
                rep.total_exposed_value_musd,
                rep.total_fire_engines,
                rep.mean_post_quake_hydrant_psi,
                rep.dry_hydrant_parcels_pct
            );
            println!(
                "  Ignitions & Supp : E[Ignitions]={:.2} (P90={}) | Knockdowns={:.2} | Dry-Hydrant Fails={:.2} | Escaped={:.2}",
                rep.expected_ignitions,
                rep.p90_ignitions,
                rep.expected_knockdowns_suppressed,
                rep.expected_dry_hydrant_failures,
                rep.expected_escaped_conflagrations
            );
            println!(
                "  Conflagration    : E[Burned]={:.2} parcels (P90={}) | {:.3}M sq ft | Displaced={:.0} res | E[Loss]=${:.1}M (P90=${:.1}M)",
                rep.expected_burned_parcels,
                rep.p90_burned_parcels,
                rep.expected_burned_floor_area_mmsqft,
                rep.expected_displaced_residents,
                rep.expected_fire_loss_musd,
                rep.p90_fire_loss_musd
            );
            if let Some(cout) = csv_out {
                let n = rep.export_csv_path(&cout)?;
                println!("Exported {n} FFE parcel risk rows -> {:?}", cout);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Surge {
            scenario,
            osm,
            load_snapshot,
            policy,
            mw,
            distance_km,
            vs30,
            trials,
            wind_mps,
            surge_hours,
            hosp_retrofit_pct,
            backup_lifelines,
            aba_burn_surge,
            field_hospitals,
            hosp_csv_out,
            parcel_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let div_pol = match policy {
                DiversionCliPolicy::Closest => AmbulanceDiversionPolicy::ClosestHospitalNoDiversion,
                DiversionCliPolicy::Threshold => {
                    AmbulanceDiversionPolicy::ThresholdSaturationDiversion
                }
                DiversionCliPolicy::Dynamic => {
                    AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage
                }
            };
            let scfg = HospitalSurgeConfig {
                ffe: FfeConfig {
                    seismic: SeismicScenarioConfig {
                        moment_magnitude_mw: mw,
                        rupture_standoff_km: distance_km,
                        vs30_m_per_s: vs30,
                        monte_carlo_trials: trials,
                        ..SeismicScenarioConfig::default()
                    },
                    wind_speed_mps: wind_mps,
                    ..FfeConfig::default()
                },
                diversion_policy: div_pol,
                surge_window_hours: surge_hours,
                hospital_seismic_retrofit_pct: hosp_retrofit_pct,
                hospital_onsite_backup_lifelines: backup_lifelines,
                aba_burn_surge_expansion: aba_burn_surge,
                deploy_mobile_field_hospitals: field_hospitals,
                ..HospitalSurgeConfig::default()
            };
            let rep = world.run_hospital_surge_monte_carlo(&scfg);
            println!(
                "=== urban-grid v3.3 Hospital ED Surge & Dynamic Ambulance Diversion ({:.1} ms | {} trials | Mw {:.1} | {}) ===",
                rep.solve_time_ms,
                trials,
                mw,
                div_pol.label()
            );
            println!(
                "  Casualties & Fleet : Total={:.1} (Red={:.2}, Burn={:.2}, Yel={:.2}, Grn={:.1}) | {} ambulances | ED Func={:.1}%",
                rep.expected_total_casualties,
                rep.expected_red_trauma,
                rep.expected_burn_critical,
                rep.expected_yellow_delayed,
                rep.expected_green_walkin,
                rep.total_ambulances,
                rep.mean_system_ed_functionality_pct
            );
            println!(
                "  Queue & Diversion  : AmbTrav={:.1}m | Diverted={:.2} ({:.1}%) | Red/Burn Wait={:.1}m (P90={:.1}m) | Treat={:.1}m",
                rep.mean_ambulance_transport_min,
                rep.expected_diverted_ambulances,
                rep.ambulance_diversion_rate_pct,
                rep.mean_red_burn_ed_wait_min,
                rep.p90_red_burn_ed_wait_min,
                rep.mean_red_burn_time_to_treatment_min
            );
            println!(
                "  Clinical Outcomes  : Golden-Hour (<=60m)={:.1}% | Burn-Center Match={:.1}% | Prev. Surge Deaths={:.2}",
                rep.golden_hour_treated_pct,
                rep.burn_matched_to_burn_center_pct,
                rep.expected_preventable_surge_deaths
            );
            if let Some(hpath) = hosp_csv_out {
                let n = rep.export_hospital_csv_path(&hpath)?;
                println!("Exported {n} hospital ED surge telemetry rows -> {:?}", hpath);
            }
            if let Some(ppath) = parcel_csv_out {
                let n = rep.export_parcel_csv_path(&ppath)?;
                println!("Exported {n} parcel casualty routing rows -> {:?}", ppath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Medevac {
            scenario,
            osm,
            load_snapshot,
            avionics,
            fleet,
            mw,
            distance_km,
            vs30,
            trials,
            wind_mps,
            ceiling_ft,
            visibility_sm,
            harden_pads,
            wrbdc_mutual_aid,
            helipad_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let av_reg = match avionics {
                MedevacAvionicsCli::DayVfr => MedevacAvionicsRegime::DayVfrPart135,
                MedevacAvionicsCli::NightVfr => MedevacAvionicsRegime::NightUnassistedVfr,
                MedevacAvionicsCli::Nvis => MedevacAvionicsRegime::NightNvisHtawsVfr,
                MedevacAvionicsCli::PinsIfr => MedevacAvionicsRegime::WaasPinSIfr,
            };
            let mcfg = AeromedicalMedevacConfig {
                surge: HospitalSurgeConfig {
                    ffe: FfeConfig {
                        seismic: SeismicScenarioConfig {
                            moment_magnitude_mw: mw,
                            rupture_standoff_km: distance_km,
                            vs30_m_per_s: vs30,
                            monte_carlo_trials: trials,
                            ..SeismicScenarioConfig::default()
                        },
                        wind_speed_mps: wind_mps,
                        ..FfeConfig::default()
                    },
                    ..HospitalSurgeConfig::default()
                },
                avionics_regime: av_reg,
                airframe_class: RotorcraftAirframeClass::MediumTwinIcuAw139,
                rotorcraft_fleet_count: fleet,
                ambient_ceiling_ft: ceiling_ft,
                ambient_visibility_sm: visibility_sm,
                helipad_seismic_and_ground_lz_hardening: harden_pads,
                wrbdc_regional_burn_mutual_aid: wrbdc_mutual_aid,
            };
            let rep = world.run_aeromedical_medevac_mc(&mcfg);
            println!(
                "=== urban-grid v3.4 Rotary-Wing Medevac & Burn-Center Transfers ({:.1} ms | {} trials | Mw {:.1} | {}) ===",
                rep.solve_time_ms,
                trials,
                mw,
                av_reg.label()
            );
            println!(
                "  Helipads & Smoke  : {} pads | StructOK={:.1}% | Mean Smoke Vis={:.2} SM | Flight-Envelope Open={:.1}%",
                rep.total_helipads,
                rep.mean_helipad_structural_ok_pct,
                rep.mean_helipad_smoke_vis_sm,
                rep.system_flight_envelope_open_pct
            );
            println!(
                "  Sorties & Scrubs  : BurnCrit={:.2} | SecXfer Req={:.2} | AirXfer={:.2} (WRBDC={:.2}) | CCT Ground={:.2} | Scrubs={:.2}",
                rep.expected_burn_critical_casualties,
                rep.expected_burn_requiring_secondary_transfer,
                rep.expected_air_secondary_burn_transfers,
                rep.expected_wrbdc_regional_air_transfers,
                rep.expected_cct_ground_fallback_transfers,
                rep.expected_weather_smoke_scrubbed_sorties
            );
            println!(
                "  Transfer & Surviv : Air={:.1}m vs CCT={:.1}m (Mean={:.1}m) | <=4hr Burn Ctr={:.1}% | Prev. Xfer Deaths={:.2}",
                rep.mean_air_burn_transfer_min,
                rep.mean_cct_ground_transfer_min,
                rep.mean_secondary_burn_transfer_min,
                rep.burn_definitive_care_within_4hr_pct,
                rep.expected_preventable_burn_transfer_deaths
            );
            if let Some(hpath) = helipad_csv_out {
                let n = rep.export_helipad_csv_path(&hpath)?;
                println!("Exported {n} helipad & flight-envelope rows -> {:?}", hpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Evacuate {
            scenario,
            osm,
            load_snapshot,
            staging,
            contraflow,
            mw,
            distance_km,
            trials,
            wind_mps,
            visibility_sm,
            stage2_delay_min,
            stage3_delay_min,
            parcel_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let stg_pol = match staging {
                EvacStagingCli::Unstaged => EvacuationStagingStrategy::SimultaneousUnstaged,
                EvacStagingCli::Staged3Zone => EvacuationStagingStrategy::FireProximityThreeStage,
                EvacStagingCli::StagedVuln => {
                    EvacuationStagingStrategy::VulnerabilityAndFireWeighted
                }
            };
            let cf_pol = match contraflow {
                ContraflowCli::None => ContraflowReversalPolicy::NoContraflowBaseline,
                ContraflowCli::FullOneWay => ContraflowReversalPolicy::FullUnprotectedOneWayOut,
                ContraflowCli::EmsProtected => {
                    ContraflowReversalPolicy::EmsProtectedArterialContraflow
                }
            };
            let ecfg = EvacuationContraflowConfig {
                medevac: AeromedicalMedevacConfig {
                    surge: HospitalSurgeConfig {
                        ffe: FfeConfig {
                            seismic: SeismicScenarioConfig {
                                moment_magnitude_mw: mw,
                                rupture_standoff_km: distance_km,
                                monte_carlo_trials: trials,
                                ..SeismicScenarioConfig::default()
                            },
                            wind_speed_mps: wind_mps,
                            ..FfeConfig::default()
                        },
                        ..HospitalSurgeConfig::default()
                    },
                    ambient_visibility_sm: visibility_sm,
                    ..AeromedicalMedevacConfig::default()
                },
                staging_strategy: stg_pol,
                contraflow_policy: cf_pol,
                stage2_release_delay_min: stage2_delay_min,
                stage3_release_delay_min: stage3_delay_min,
                ..EvacuationContraflowConfig::default()
            };
            let rep = world.run_evacuation_contraflow_mc(&ecfg);
            println!(
                "=== urban-grid v3.5 Staged Evacuation & Arterial Contraflow ({:.1} ms | {} trials | {} + {}) ===",
                rep.solve_time_ms,
                trials,
                stg_pol.short_label(),
                cf_pol.short_label()
            );
            println!(
                "  Demand & Reversal : {:.0} evacuees ({:.0} vehs) | {} contraflow segs (+{:.2} ln-km) | {} EMS/LZ segs",
                rep.total_evacuees,
                rep.total_evac_vehicles,
                rep.contraflow_reversed_segments_count,
                rep.contraflow_lane_km_added,
                rep.protected_ems_lz_segments_count
            );
            println!(
                "  FFE & Smoke       : Thermally Closed={:.1} segs | Mean Smoke Speed={:.1}% | Peak Bottleneck V/C={:.2}",
                rep.expected_thermally_closed_segments,
                rep.mean_road_smoke_speed_factor * 100.0,
                rep.peak_bottleneck_vc_ratio
            );
            println!(
                "  Clearance & EMS   : T50={:.1}m | T90={:.1}m | T95={:.1}m | Fire Overrun={:.1} res | EMS-LZ Access={:.1}% ({:.1}m in)",
                rep.t50_clearance_min,
                rep.t90_clearance_min,
                rep.t95_clearance_min,
                rep.expected_fire_overrun_residents,
                rep.ems_lz_corridor_access_preserved_pct,
                rep.mean_ems_inbound_response_min
            );
            if let Some(ppath) = parcel_csv_out {
                let n = rep.export_parcel_csv_path(&ppath)?;
                println!("Exported {n} staged parcel evacuation rows -> {:?}", ppath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Shelters {
            scenario,
            osm,
            load_snapshot,
            policy,
            filtration,
            open_shelters,
            sqft_per_person,
            hepa_cadr,
            mw,
            distance_km,
            trials,
            wind_mps,
            backup_lifelines,
            shelter_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let s_pol = match policy {
                ShelterPolicyCli::NearestUncoord => {
                    ShelterSitingAndRoutingPolicy::UncoordinatedNearestShelter
                }
                ShelterPolicyCli::CapDist => ShelterSitingAndRoutingPolicy::CapacitatedDistanceOnly,
                ShelterPolicyCli::CleanAir2Stage => {
                    ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated
                }
            };
            let f_std = match filtration {
                AirFilterCli::Merv8 => AirFiltrationStandard::BaselineMerv8Unsealed,
                AirFilterCli::Merv13 => AirFiltrationStandard::RecirculatingMerv13Hvac,
                AirFilterCli::Merv13Hepa => {
                    AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa
                }
            };
            let scfg = EmergencySheltersConfig {
                evac: EvacuationContraflowConfig {
                    medevac: AeromedicalMedevacConfig {
                        surge: HospitalSurgeConfig {
                            ffe: FfeConfig {
                                seismic: SeismicScenarioConfig {
                                    moment_magnitude_mw: mw,
                                    rupture_standoff_km: distance_km,
                                    monte_carlo_trials: trials,
                                    ..SeismicScenarioConfig::default()
                                },
                                wind_speed_mps: wind_mps,
                                ..FfeConfig::default()
                            },
                            ..HospitalSurgeConfig::default()
                        },
                        ..AeromedicalMedevacConfig::default()
                    },
                    ..EvacuationContraflowConfig::default()
                },
                filtration_standard: f_std,
                siting_routing_policy: s_pol,
                max_open_shelters_budget: open_shelters,
                sqft_per_evacuee: sqft_per_person,
                portable_hepa_cadr_m3h_per_100m2: hepa_cadr,
                shelter_backup_power_water_sanitation: backup_lifelines,
                ..EmergencySheltersConfig::default()
            };
            let rep = world.run_emergency_shelters_mc(&scfg);
            println!(
                "=== urban-grid v3.6 Post-Earthquake Clean-Air Shelters & Evacuee Routing ({:.1} ms | {} trials | {} + {}) ===",
                rep.solve_time_ms,
                trials,
                s_pol.short_label(),
                f_std.short_label()
            );
            println!(
                "  Shelters & Cots   : {}/{} open | Nominal={} cots | Expected Usable={:.0} cots",
                rep.selected_open_shelters_count,
                rep.candidate_shelters_count,
                rep.total_nominal_cot_capacity,
                rep.expected_usable_cot_capacity
            );
            println!(
                "  Air & Habitability: Displaced={:.0} (P90={}) | CleanAir Sheltered={:.0} ({:.1}%) | Mean InPM2.5={:.1} ug/m3",
                rep.total_displaced_evacuees,
                rep.p90_displaced_evacuees,
                rep.expected_sheltered_in_clean_air,
                rep.sheltered_clean_air_pct,
                rep.mean_assigned_indoor_pm25_ug_m3
            );
            println!(
                "  Routing & Deficit : Mean Trav={:.1}m (P90={:.1}m) | Unmet/Overcrowd={:.1} | Smoke Exceed={:.1}",
                rep.mean_shelter_travel_min,
                rep.p90_shelter_travel_min,
                rep.expected_overcrowded_or_unmet_evacuees,
                rep.expected_smoke_exceedance_evacuees
            );
            if let Some(spath) = shelter_csv_out {
                let n = rep.export_shelter_csv_path(&spath)?;
                println!("Exported {n} emergency shelter habitability rows -> {:?}", spath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Resupply {
            scenario,
            osm,
            load_snapshot,
            policy,
            days,
            init_reserve_hrs,
            water_trucks,
            diesel_trucks,
            mre_trucks,
            oxygen_trucks,
            heli_sling,
            mw,
            distance_km,
            trials,
            site_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let r_pol = match policy {
                ResupplyPolicyCli::MilkRun => ResupplyDispatchPolicy::StaticEqualRotationMilkRun,
                ResupplyPolicyCli::MyopicSs => {
                    ResupplyDispatchPolicy::GreedyMyopicReorderThreshold
                }
                ResupplyPolicyCli::ClarkeWrightIrp => {
                    ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp
                }
            };
            let rcfg = CommodityResupplyConfig {
                shelters: EmergencySheltersConfig {
                    evac: EvacuationContraflowConfig {
                        medevac: AeromedicalMedevacConfig {
                            surge: HospitalSurgeConfig {
                                ffe: FfeConfig {
                                    seismic: SeismicScenarioConfig {
                                        moment_magnitude_mw: mw,
                                        rupture_standoff_km: distance_km,
                                        monte_carlo_trials: trials,
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
                dispatch_policy: r_pol,
                horizon_days: days,
                water_tanker_trucks: water_trucks,
                diesel_fuel_trucks: diesel_trucks,
                mre_ration_trucks: mre_trucks,
                medical_oxygen_trucks: oxygen_trucks,
                initial_onsite_reserve_hours: init_reserve_hrs,
                enable_helicopter_sling_resupply: heli_sling,
            };
            let rep = world.run_commodity_resupply_mc(&rcfg);
            println!(
                "=== urban-grid v3.7 Multi-Day Shelter & Hospital Commodity Resupply ({:.1} ms | {}d | {} trials | {}) ===",
                rep.solve_time_ms,
                days,
                trials,
                r_pol.label()
            );
            println!(
                "  Delivered Totals  : Water={:.1} kL | Diesel={:.1} kL | MREs={:.1}k meals | Medical O2={:.1} kL | Heli Sling={:.1}",
                rep.total_water_delivered_kl,
                rep.total_diesel_delivered_kl,
                rep.total_mre_delivered_k,
                rep.total_oxygen_delivered_kl,
                rep.expected_helicopter_sling_sorties
            );
            println!(
                "  Stockout & VRP    : Water={:.0} p-hr | Diesel={:.1} site-hr | O2={:.0} pt-hr | CW Tour Sav={:.1}%",
                rep.expected_water_stockout_person_hrs,
                rep.expected_diesel_blackout_site_hrs,
                rep.expected_oxygen_shortfall_patient_hrs,
                rep.mean_convoy_tour_savings_pct
            );
            println!(
                "  Sustainment Score : System Supply Security={:.1}% | Total Deprivation Social Cost={:.0}",
                rep.system_supply_security_pct,
                rep.total_deprivation_social_cost
            );
            if let Some(spath) = site_csv_out {
                let n = rep.export_site_csv_path(&spath)?;
                println!("Exported {n} sustainment site resupply rows -> {:?}", spath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::TransitEvac {
            scenario,
            osm,
            load_snapshot,
            lane_mode,
            policy,
            buses,
            ada_vans,
            pickup_depots,
            window_min,
            mw,
            distance_km,
            trials,
            depot_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let l_mode = match lane_mode {
                BusLaneModeCli::MixedQueue => BusContraflowLaneMode::MixedAutoQueueNoPriority,
                BusLaneModeCli::CurbBusOnly => BusContraflowLaneMode::CurbsideBusOnlyNoContraflow,
                BusLaneModeCli::BusEmsContraflow => {
                    BusContraflowLaneMode::DedicatedBusPlusEmsContraflow
                }
            };
            let d_pol = match policy {
                TransitEvacPolicyCli::FixedGtfs => TransitDispatchStrategy::FixedGtfsHeadwayLoop,
                TransitEvacPolicyCli::DemandProp => {
                    TransitDispatchStrategy::DemandProportionalShuttle
                }
                TransitEvacPolicyCli::CleanAirTbep => {
                    TransitDispatchStrategy::FireUrgencyCleanAirTbep
                }
            };
            let tcfg = TransitEvacuationConfig {
                shelters: EmergencySheltersConfig {
                    evac: EvacuationContraflowConfig {
                        medevac: AeromedicalMedevacConfig {
                            surge: HospitalSurgeConfig {
                                ffe: FfeConfig {
                                    seismic: SeismicScenarioConfig {
                                        moment_magnitude_mw: mw,
                                        rupture_standoff_km: distance_km,
                                        monte_carlo_trials: trials,
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
                    shelter_backup_power_water_sanitation: true,
                    ..EmergencySheltersConfig::default()
                },
                lane_mode: l_mode,
                dispatch_strategy: d_pol,
                coach_spec: BusCoachSpec::Articulated60FtBrt,
                fare_mode: BoardingFareMode::AllDoorFareFreeBrt,
                transit_bus_fleet: buses,
                ada_paratransit_vans: ada_vans,
                max_pickup_depots: pickup_depots,
                walk_catchment_radius_m: 400.0,
                clearance_window_min: window_min,
                carless_demand_multiplier: 1.0,
            };
            let rep = world.run_transit_bus_evacuation_mc(&tcfg);
            println!(
                "=== urban-grid v3.8 Coupled GTFS Transit-Bus & ADA Clean-Air Evacuation ({:.1} ms | {} trials | {} | {}) ===",
                rep.solve_time_ms,
                trials,
                l_mode.short_label(),
                d_pol.short_label()
            );
            println!(
                "  Vulnerable Demand : Carless={:.1} pax ({:.1}% evacuated) | ADA Mobility-Limited={:.1} pax ({:.1}% evacuated)",
                rep.total_carless_demand_pax,
                rep.evacuated_carless_pct,
                rep.total_ada_mobility_limited_pax,
                rep.evacuated_ada_pct
            );
            println!(
                "  Shuttle & EMS     : Bus Cycle={:.1} min ({:.2} trips/bus) | ADA={:.2} trips/van | EMS Inbound Spine={:.1}%",
                rep.mean_bus_shuttle_cycle_min,
                rep.mean_trips_per_bus,
                rep.mean_trips_per_ada_van,
                rep.ems_inbound_access_preserved_pct
            );
            println!(
                "  Clean-Air Outcome : Combined Clean-Air Sheltered={:.1}% | Stranded={:.1} pax (P90={:.1})",
                rep.combined_clean_air_sheltered_pct,
                rep.stranded_vulnerable_pax,
                rep.p90_stranded_vulnerable_pax
            );
            if let Some(dpath) = depot_csv_out {
                let n = rep.export_depot_csv_path(&dpath)?;
                println!("Exported {n} GTFS pickup depot rows -> {:?}", dpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Warning {
            scenario,
            osm,
            load_snapshot,
            telecom_backup,
            channels,
            staging,
            hours_post_quake,
            night_event,
            sirens,
            patrol_units,
            mw,
            distance_km,
            trials,
            tower_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let tb = match telecom_backup {
                TelecomBackupCli::Legacy3h => {
                    TelecomBackupPowerStandard::LegacyCommercial2to4HrBattery
                }
                TelecomBackupCli::Fcc8h => TelecomBackupPowerStandard::FccEightHourBatteryStandard,
                TelecomBackupCli::Cpuc72hLeo => {
                    TelecomBackupPowerStandard::Cpuc72HrHardenedWithLeoBackhaul
                }
            };
            let ch = match channels {
                AlertChannelCli::WeaOnly => AlertChannelMix::WeaAndReverse911Only,
                AlertChannelCli::WeaSirens => AlertChannelMix::WeaPlusOutdoorAcousticSirens,
                AlertChannelCli::MultiModal => AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor,
            };
            let st = match staging {
                DepartureStagingCli::Unstaged => DepartureStagingPolicy::UnstagedImmediateOnReceipt,
                DepartureStagingCli::StaticWea => DepartureStagingPolicy::StaticGeoZoneWeaStaging,
                DepartureStagingCli::AdaptiveBlackout => {
                    DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging
                }
            };
            let wcfg = WarningMobilizationConfig {
                transit_evac: TransitEvacuationConfig {
                    shelters: EmergencySheltersConfig {
                        evac: EvacuationContraflowConfig {
                            medevac: AeromedicalMedevacConfig {
                                surge: HospitalSurgeConfig {
                                    ffe: FfeConfig {
                                        seismic: SeismicScenarioConfig {
                                            moment_magnitude_mw: mw,
                                            rupture_standoff_km: distance_km,
                                            monte_carlo_trials: trials,
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
                telecom_backup: tb,
                channel_mix: ch,
                staging_policy: st,
                hours_post_mainshock: hours_post_quake,
                nocturnal_sleeping_event: night_event,
                outdoor_siren_sites: sirens,
                canvass_patrol_units: patrol_units,
            };
            let rep = world.run_warning_mobilization_mc(&wcfg);
            println!(
                "=== urban-grid v3.9 Cellular Blackout, WEA/Siren Mobilization & Delayed Staging ({:.1} ms | {} trials | {} | {}) ===",
                rep.solve_time_ms,
                trials,
                tb.short_label(),
                ch.short_label()
            );
            println!(
                "  Cellular Fragility: BTS Blackout={:.1}% | Residents in WEA Shadow={:.1}% (t=+{:.1}h post-quake)",
                rep.mean_cell_towers_blackout_pct,
                rep.population_in_wea_blackout_pct,
                hours_post_quake
            );
            println!(
                "  Mobilization Curve: Mean Receipt={:.1} min | P50 Depart={:.1} min | P90 Depart={:.1} min",
                rep.mean_warning_receipt_min,
                rep.median_mobilization_departure_min,
                rep.p90_mobilization_departure_min
            );
            println!(
                "  Staging & Safety  : Inner Fire Zone Clr={:.1} min | Peak V/C={:.2} | Fire-Trapped={:.1} pax (P90={:.1})",
                rep.inner_fire_zone_clearance_min,
                rep.network_peak_vc_ratio,
                rep.late_departure_fire_trapped_pax,
                rep.p90_fire_trapped_pax
            );
            if let Some(tpath) = tower_csv_out {
                let n = rep.export_tower_csv_path(&tpath)?;
                println!("Exported {n} cellular BTS fragility rows -> {:?}", tpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::FemaComms {
            scenario,
            osm,
            load_snapshot,
            eoc_uplink,
            posture,
            eas_radio_pct,
            nwr_same_pct,
            mers_units,
            hours_post_quake,
            night_event,
            mw,
            distance_km,
            trials,
            asset_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let ul = match eoc_uplink {
                EocUplinkCli::FiberOnly => EocIpaswUplinkMode::CommercialTerrestrialFiberOnly,
                EocUplinkCli::FiberFnarsHf => EocIpaswUplinkMode::FiberPlusFnarsHfSkywaveRelay,
                EocUplinkCli::FullMersFnars => {
                    EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity
                }
            };
            let pos = match posture {
                FemaPostureCli::WeaOnly => FemaCommsPosture::IpaswWea30CellularOnly,
                FemaPostureCli::TriPathIpasw => FemaCommsPosture::IpaswTriPathWeaEasPepNwrSame,
                FemaPostureCli::FullIpaswMers => {
                    FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment
                }
            };
            let fcfg = FemaCommsConfig {
                warning: WarningMobilizationConfig {
                    transit_evac: TransitEvacuationConfig {
                        shelters: EmergencySheltersConfig {
                            evac: EvacuationContraflowConfig {
                                medevac: AeromedicalMedevacConfig {
                                    surge: HospitalSurgeConfig {
                                        ffe: FfeConfig {
                                            seismic: SeismicScenarioConfig {
                                                moment_magnitude_mw: mw,
                                                rupture_standoff_km: distance_km,
                                                monte_carlo_trials: trials,
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
                    hours_post_mainshock: hours_post_quake,
                    nocturnal_sleeping_event: night_event,
                    ..WarningMobilizationConfig::default()
                },
                eoc_uplink: ul,
                posture: pos,
                household_eas_am_fm_radio_pct: eas_radio_pct,
                household_nwr_same_receiver_pct: nwr_same_pct,
                fema_mers_irv_units: mers_units,
                airlift_mers_flyaway_kits: true,
            };
            let rep = world.run_fema_comms_mc(&fcfg);
            println!(
                "=== urban-grid v4.0 FEMA IPAWS (CAP/WEA/PEP/NWR), FNARS & Region 10 MERS ESF-2 ({:.1} ms | {} trials) ===",
                rep.solve_time_ms,
                trials
            );
            println!(
                "  Origination & RF  : COG EOC CAP={:.1}% | NPWS PEP EAS={:.1}% | NOAA NWR SAME={:.1}% | WEA 3.0={:.1}%",
                rep.eoc_cap_origination_uptime_pct,
                rep.npws_pep_broadcast_uptime_pct,
                rep.noaa_nwr_same_uptime_pct,
                rep.commercial_wea30_uptime_pct
            );
            println!(
                "  Public & ESF #2   : Verified IPAWS Public Reach={:.1}% | NIFOG Responder LMR={:.1}% | MERS Units={}",
                rep.public_ipaws_verified_reach_pct,
                rep.esf2_nifog_responder_lmr_coverage_pct,
                rep.deployed_mers_units_count
            );
            println!(
                "  Mobilization Gap  : Mean IPAWS Mobilization={:.1} min | Unwarned Isolated={:.1} pax (P90={:.1})",
                rep.mean_ipaws_mobilization_min,
                rep.unwarned_isolated_residents_pax,
                rep.p90_unwarned_residents_pax
            );
            if let Some(apath) = asset_csv_out {
                let n = rep.export_asset_csv_path(&apath)?;
                println!("Exported {n} FEMA IPAWS/MERS asset rows -> {:?}", apath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::FemaPods {
            scenario,
            osm,
            load_snapshot,
            routing,
            isb_policy,
            pods,
            trailers_53ft,
            needy_pop_pct,
            type1_super_pods,
            mw,
            distance_km,
            trials,
            pod_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let r_pol = match routing {
                PodRoutingCli::NearestRush => IpaswPodRoutingPolicy::UncoordinatedNearestPodRush,
                PodRoutingCli::StaticDistricts => {
                    IpaswPodRoutingPolicy::StaticPrePublishedDistricts
                }
                PodRoutingCli::IpaswBalanced => {
                    IpaswPodRoutingPolicy::DynamicIpaswQueueAndStockBalanced
                }
            };
            let i_pol = match isb_policy {
                IsbPolicyCli::OvernightStatic => IsbReplenishmentPolicy::OvernightStaticEqualPush,
                IsbPolicyCli::DynamicHotshot => {
                    IsbReplenishmentPolicy::DynamicIsbPullWithHotshotRestock
                }
            };
            let pcfg = FemaIsbPodsConfig {
                fema_comms: FemaCommsConfig {
                    warning: WarningMobilizationConfig {
                        transit_evac: TransitEvacuationConfig {
                            shelters: EmergencySheltersConfig {
                                evac: EvacuationContraflowConfig {
                                    medevac: AeromedicalMedevacConfig {
                                        surge: HospitalSurgeConfig {
                                            ffe: FfeConfig {
                                                seismic: SeismicScenarioConfig {
                                                    moment_magnitude_mw: mw,
                                                    rupture_standoff_km: distance_km,
                                                    monte_carlo_trials: trials,
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
                routing_policy: r_pol,
                isb_policy: i_pol,
                active_county_pods: pods,
                isb_daily_53ft_trailers: trailers_53ft,
                base_population_needing_pods_pct: needy_pop_pct,
                upgrade_arterial_pods_to_type1: type1_super_pods,
            };
            let rep = world.run_fema_isb_pods_mc(&pcfg);
            println!(
                "=== urban-grid v4.1 FEMA Incident Staging Base (ISB) & USACE IS-26 PODs ({:.1} ms | {} trials | {}) ===",
                rep.solve_time_ms,
                trials,
                r_pol.short_label()
            );
            println!(
                "  Commodity Demand  : Needy={:.1} pax | Served={:.1} pax ({:.1}% met) | IPAWS Informed={:.1}%",
                rep.total_needy_residents_pax,
                rep.daily_served_residents_pax,
                rep.population_demand_met_pct,
                rep.ipaws_informed_households_pct
            );
            println!(
                "  POD Queueing      : Mean Queue Wait={:.1} min | P90 Access={:.1} min | Peak Street Spillback={:.1} vehs",
                rep.mean_pod_queue_wait_min,
                rep.p90_pod_total_access_min,
                rep.peak_arterial_spillback_vehs
            );
            println!(
                "  ISB Supply Chain  : Wasted Dry-Run Stockout Trips={:.1} | ISB Bridge Detour=+{:.1} min",
                rep.wasted_dry_run_stockout_trips,
                rep.isb_bridge_detour_delay_min
            );
            if let Some(ppath) = pod_csv_out {
                let n = rep.export_pod_csv_path(&ppath)?;
                println!("Exported {n} FEMA C-POD site rows -> {:?}", ppath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::FemaUsar {
            scenario,
            osm,
            load_snapshot,
            policy,
            type1_task_forces,
            dozer_crews,
            horizon_hrs,
            mw,
            distance_km,
            trials,
            grid_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let u_pol = match policy {
                UsarDebrisPolicyCli::HaulOff => DebrisAndUsarPolicy::UncoordinatedPerimeterHaulOff,
                UsarDebrisPolicyCli::PodsFirst => DebrisAndUsarPolicy::LogisticsPodCorridorsFirstPush,
                UsarDebrisPolicyCli::TeethOfComb => {
                    DebrisAndUsarPolicy::CoOptimizedTeethOfCombUsarAndPodPush
                }
            };
            let ucfg = FemaUsarDebrisConfig {
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
                                                        moment_magnitude_mw: mw,
                                                        rupture_standoff_km: distance_km,
                                                        monte_carlo_trials: trials,
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
                policy: u_pol,
                fema_type1_usar_task_forces: type1_task_forces,
                heavy_debris_push_crews: dozer_crews,
                horizon_hours: horizon_hrs,
                live_injection_snapshot_hour: 12.0,
            };
            let rep = world.run_fema_usar_debris_mc(&ucfg);
            println!(
                "=== urban-grid v4.2 FEMA Type 1 US&R Task Force & ESF-3 Debris Clearance ({:.1} ms | {} trials | {}) ===",
                rep.solve_time_ms,
                trials,
                u_pol.short_label()
            );
            println!(
                "  Roadway Rubble    : Street Debris={:.0} tons ({:.1} blocked segs -> {:.0} tons cleared | {:.1}% open)",
                rep.total_street_debris_tons,
                rep.initial_blocked_segments_count,
                rep.cleared_debris_tons,
                rep.final_open_corridors_pct
            );
            println!(
                "  US&R Extrication  : Entrapped={:.1} | Live Rescues={:.1} ({:.1}% saved | P10={:.1})",
                rep.total_initial_entrapped_victims,
                rep.expected_live_rescues,
                rep.entrapped_survival_rate_pct,
                rep.p10_live_rescues
            );
            println!(
                "  ISB->POD Reopening: All County C-POD Supply Corridors Reopened @ H+{:.1} hours",
                rep.time_to_reopen_all_pods_hrs
            );
            if let Some(gpath) = grid_csv_out {
                let n = rep.export_grid_csv_path(&gpath)?;
                println!("Exported {n} FEMA USNG collapse search grid rows -> {:?}", gpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::FemaRecovery {
            scenario,
            osm,
            load_snapshot,
            policy,
            prt_teams,
            gen_packs,
            tdms_sites,
            dump_trucks,
            drcs,
            mrics,
            mw,
            trials,
            facility_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let r_pol = match policy {
                FemaRecoveryPolicyCli::AdHoc => FemaRecoveryExecutionPolicy::SequentialAdHoc,
                FemaRecoveryPolicyCli::PowerFirst => FemaRecoveryExecutionPolicy::LifelinePowerFirst,
                FemaRecoveryPolicyCli::IntegratedSuite => {
                    FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite
                }
            };
            let mut rcfg = FemaGsbDrcTdmsConfig {
                policy: r_pol,
                usace_249th_prt_teams: prt_teams,
                gsb_generator_packs_inventory: gen_packs,
                active_tdms_sites: tdms_sites,
                debris_haul_dump_trucks: dump_trucks,
                fixed_drc_centers: drcs,
                mobile_mric_units: mrics,
                ..FemaGsbDrcTdmsConfig::default()
            };
            rcfg.fema_usar.fema_pods.fema_comms.warning.transit_evac.shelters.evac.medevac.surge.ffe.seismic.moment_magnitude_mw = mw;
            rcfg.fema_usar.fema_pods.fema_comms.warning.transit_evac.shelters.evac.medevac.surge.ffe.seismic.monte_carlo_trials = trials;
            let rep = world.run_fema_gsb_drc_tdms_mc(&rcfg);
            println!(
                "=== urban-grid v5.0 FEMA 249th GSB Prime Power, TDMS Debris & DRC/MRIC IA ({:.1} ms | {} trials | {}) ===",
                rep.solve_time_ms,
                trials,
                r_pol.short_label()
            );
            println!(
                "  USACE 249th GSB   : {:.1}% Lifelines <=72h | Mean Hookup={:.1}h | Deployed Prime Power={:.2} MW",
                rep.energized_lifelines_72h_pct,
                rep.mean_time_to_emergency_power_hrs,
                rep.deployed_prime_power_mw
            );
            println!(
                "  FEMA-325 TDMS     : Total Debris={:.0}t | 14d Hauled={:.1}% | Volume Saved={:.0} yd3 | HHW Safe={:.1}% | {} Freon units",
                rep.total_disaster_debris_tons,
                rep.curbside_haul_completed_14d_pct,
                rep.tdms_landfill_volume_saved_yd3,
                rep.hhw_contamination_prevented_pct,
                rep.white_goods_freon_units
            );
            println!(
                "  DRC + MRIC Intake : IA Registered={:.1}% ({:.0} HH) | Mean Wait={:.1}m | Transit Equity={:.1}% | IA+SBA=${:.2}M",
                rep.ia_registered_14d_pct,
                rep.affected_ia_eligible_households,
                rep.mean_survivor_drc_wait_min,
                rep.transit_walkable_intake_equity_pct,
                rep.projected_ia_and_sba_disbursement_musd
            );
            if let Some(fpath) = facility_csv_out {
                let n = rep.export_facility_csv_path(&fpath)?;
                println!("Exported {n} FEMA GSB/TDMS/DRC/MRIC site rows -> {:?}", fpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Wildfire {
            scenario,
            osm,
            load_snapshot,
            camera_feed,
            export_sample_feed,
            posture,
            fuel,
            wind_mph,
            rh_pct,
            slope_pct,
            night_nir,
            starlink_mdus,
            hotshot_crews,
            airtankers,
            trials,
            camera_csv_out,
            json_out,
        } => {
            if let Some(sample_out) = export_sample_feed {
                if let Some(parent) = sample_out.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(&sample_out, builtin_western_wildfire_camera_feed_json())?;
                println!(
                    "Exported sample ALERTWest wildfire camera feed JSON -> {:?}",
                    sample_out
                );
            }
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let custom_cams = if let Some(cpath) = camera_feed {
                load_wildfire_camera_feed_path(&cpath, world.network.bounds)?
            } else {
                Vec::new()
            };
            let w_pst = match posture {
                WildfirePostureCli::Legacy911 => {
                    WildfireDetectionAndAttackPosture::Legacy911GroundOnly
                }
                WildfirePostureCli::FixedPtzIa => {
                    WildfireDetectionAndAttackPosture::AlertWestFixedPtzInitialAttack
                }
                WildfirePostureCli::FullAerial => {
                    WildfireDetectionAndAttackPosture::FullAlertWestMduAndAerialAttack
                }
            };
            let w_fuel = match fuel {
                WuiFuelCli::GrassFm1 => WuiFuelModel::ShortGrassFm1,
                WuiFuelCli::BrushFm5 => WuiFuelModel::BrushWuiFm5,
                WuiFuelCli::ChaparralFm4 => WuiFuelModel::HeavyChaparralSlashFm4,
                WuiFuelCli::TimberFm10 => WuiFuelModel::ConiferTimberUnderstoryFm10,
            };
            let wcfg = WildfireAlertWestConfig {
                posture: w_pst,
                fuel_model: w_fuel,
                wind_speed_mph: wind_mph,
                relative_humidity_pct: rh_pct,
                terrain_slope_pct: slope_pct,
                nocturnal_ignition: night_nir,
                starlink_mdu_units: starlink_mdus,
                type1_hotshot_crews_20pax: hotshot_crews,
                fixed_wing_airtankers: airtankers,
                monte_carlo_trials: trials,
                custom_camera_sites: custom_cams,
                ..WildfireAlertWestConfig::default()
            };
            let rep = world.run_wildfire_alertwest_mc(&wcfg);
            println!(
                "=== urban-grid v5.0 ALERTWest Cameras + NWCG WUI Wildfire ({:.1} ms | {} trials | {}) ===",
                rep.solve_time_ms,
                trials,
                w_pst.short_label()
            );
            println!(
                "  ALERTWest         : {} Cameras ({} LOS, {} Seismic) | WLS CEP90={:.1} m | Detect->Dispatch={:.1} min",
                rep.active_camera_count,
                rep.cameras_with_los_count,
                rep.pnsn_shakealert_colocated_count,
                rep.triangulation_cep90_m,
                rep.mean_detection_to_dispatch_min
            );
            println!(
                "  Rothermel/NWCG    : ROS={:.1} ch/hr | Intensity={:.0} Btu/ft/s | Flame={:.1} ft ({})",
                rep.fire_behavior.rate_of_spread_chains_per_hr,
                rep.fire_behavior.byram_intensity_btu_ft_s,
                rep.fire_behavior.flame_length_ft,
                rep.fire_behavior.threshold_class.label()
            );
            println!(
                "  WUI Suppression   : Burned={:.1} ac (P90={:.1} ac) | Contained={:.1}% | Phos-Chek={:.0} gal | Level-3 GO={:.0} pax",
                rep.final_burned_area_acres,
                rep.p90_burned_area_acres,
                rep.final_containment_pct,
                rep.phos_chek_dropped_gallons,
                rep.level3_go_evac_residents
            );
            if let Some(cpath) = camera_csv_out {
                let n = rep.export_camera_csv_path(&cpath)?;
                println!("Exported {n} ALERTWest wildfire camera rows -> {:?}", cpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::DebrisFlow {
            scenario,
            osm,
            load_snapshot,
            baer_policy,
            i15_mm_hr,
            storm_mm,
            soil_kf,
            basins,
            trials,
            basin_csv_out,
            culvert_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let b_pol = match baer_policy {
                BaerPolicyCli::Untreated => BaerWatershedMitigationPolicy::UntreatedBurnScar,
                BaerPolicyCli::AerialMulch => {
                    BaerWatershedMitigationPolicy::AerialWoodStrandMulchOnly
                }
                BaerPolicyCli::FullBaer => {
                    BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing
                }
            };
            let mut dcfg = BurnScarDebrisFlowConfig {
                baer_policy: b_pol,
                peak_i15_intensity_mm_hr: i15_mm_hr,
                storm_6hr_depth_mm: storm_mm,
                soil_kf_erodibility: soil_kf,
                delineated_basins_count: basins,
                ..BurnScarDebrisFlowConfig::default()
            };
            dcfg.wildfire.monte_carlo_trials = trials;
            let rep = world.run_burn_scar_debris_flow_mc(&dcfg);
            println!(
                "=== urban-grid v5.1 Post-Wildfire Burn-Scar Runoff, USGS Debris-Flow Basins & Culvert Washouts ({:.1} ms | {} trials | {}) ===",
                rep.solve_time_ms,
                trials,
                b_pol.short_label()
            );
            println!(
                "  Hydrophobic CN    : Mean CN={:.1} | Runoff={:.1} mm ({:.2}x vs {:.1} mm unburned) | Burned Bmh={:.2} km2",
                rep.mean_post_fire_curve_number,
                rep.mean_hydrophobic_runoff_depth_mm,
                rep.runoff_amplification_ratio,
                rep.unburned_baseline_runoff_mm,
                rep.burned_watershed_area_km2
            );
            println!(
                "  USGS M1 + Gartner : Fire-Wide Median I15_P50={:.1} mm/hr ({}/{} basins >P50) | Sediment={:.0} m3 (BAER Saved={:.0} m3)",
                rep.firewide_median_i15_p50_mm_hr,
                rep.basins_exceeding_p50_count,
                rep.basin_rows.len(),
                rep.total_predicted_sediment_volume_m3,
                rep.baer_trapped_or_prevented_sediment_m3
            );
            println!(
                "  FHWA HDS-5 Washout: E[Washed Out]={:.1} (P90={}) / {} Culverts | Post-Washout <=4m Cov={:.1}% | RDII Sewer={:.1} mm/hr",
                rep.expected_washed_out_culverts,
                rep.p90_washed_out_culverts,
                rep.evaluated_culverts_count,
                rep.post_washout_4min_fire_cov_pct,
                rep.coupled_rdii_sewer_inflow_mm_hr
            );
            if let Some(bpath) = basin_csv_out {
                let n = rep.export_basin_csv_path(&bpath)?;
                println!("Exported {n} USGS post-fire debris-flow basin rows -> {:?}", bpath);
            }
            if let Some(cpath) = culvert_csv_out {
                let n = rep.export_culvert_csv_path(&cpath)?;
                println!("Exported {n} FHWA HDS-5 culvert crossing rows -> {:?}", cpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::WtpTurbidity {
            scenario,
            osm,
            load_snapshot,
            policy,
            shutoff_ntu,
            wtp_mgd,
            storage_mg,
            wells_mgd,
            coag_tons,
            i15_mm_hr,
            trials,
            timeline_csv_out,
            json_out,
        } => {
            let mut world =
                build_world(scenario, osm.as_ref(), None, load_snapshot.as_ref(), Some(0))?;
            if load_snapshot.is_none() {
                let sited = world.solve_mclp_station_siting(&MclpSitingConfig {
                    objective_model: SitingObjectiveModel::DaskinMexclp1983,
                    apparatus_busy_prob: 0.35,
                    target_coverage_pct: 95.0,
                    max_stations_budget: 5,
                    ..MclpSitingConfig::default()
                });
                world.apply_mclp_siting_solution(&sited);
            }
            let w_pol = match policy {
                WtpPolicyCli::RunToFailure => WtpCoagulantAndIntakePolicy::RunToFailureAlumBaseline,
                WtpPolicyCli::IntakeShutoff => {
                    WtpCoagulantAndIntakePolicy::AutomatedIntakeShutoffAlumOnly
                }
                WtpPolicyCli::FerricPacWells => {
                    WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie
                }
            };
            let mut wcfg = WtpTurbidityStorageConfig {
                policy: w_pol,
                intake_shutoff_threshold_ntu: shutoff_ntu,
                wtp_design_capacity_mgd: wtp_mgd,
                finished_storage_capacity_mg: storage_mg,
                emergency_well_intertie_mgd: wells_mgd,
                onsite_coagulant_inventory_tons: coag_tons,
                ..WtpTurbidityStorageConfig::default()
            };
            wcfg.debris_flow.peak_i15_intensity_mm_hr = i15_mm_hr;
            wcfg.debris_flow.wildfire.monte_carlo_trials = trials;
            let rep = world.run_wtp_turbidity_storage_mc(&wcfg);
            println!(
                "=== urban-grid v5.2 Post-Wildfire WTP Ash/Turbidity, Coagulant & Storage Drawdown ({:.1} ms | {} trials | {}) ===",
                rep.solve_time_ms,
                trials,
                w_pol.short_label()
            );
            println!(
                "  Source River Peak : Turbidity={:.0} NTU | Pyrogenic DOC={:.1} mg/L | Wood-Ash pH={:.2} ({:.1}h > {:.0} NTU)",
                rep.peak_raw_river_ntu,
                rep.peak_raw_doc_mg_l,
                rep.peak_raw_ash_ph,
                rep.hours_exceeding_shutoff_ntu,
                shutoff_ntu
            );
            println!(
                "  WTP & Coagulation : Intake Shutoff={:.1}h | Peak Dose={:.1} mg/L ({:.1}t used, {:.1}t sludge) | Peak CFE={:.2} NTU ({:.1}% <=0.3 NTU)",
                rep.intake_shutoff_duration_hrs,
                rep.peak_coagulant_dose_mg_l,
                rep.total_coagulant_consumed_tons,
                rep.total_alum_ferric_sludge_tons,
                rep.peak_cfe_filtered_ntu,
                rep.ieswtr_03ntu_compliance_pct
            );
            println!(
                "  Storage & BWA     : Min Storage={:.1}% (P10={:.1}%) | Min High-Zone={:.1} psi | P(Boil-Water Advisory)={:.1}% ({:.0} pax)",
                rep.min_finished_storage_fill_pct,
                rep.p10_min_storage_fill_pct,
                rep.min_high_zone_pressure_psi,
                rep.boil_water_advisory_prob_pct,
                rep.expected_residents_under_bwa
            );
            if let Some(tpath) = timeline_csv_out {
                let n = rep.export_timeline_csv_path(&tpath)?;
                println!("Exported {n} hourly WTP turbidity & storage rows -> {:?}", tpath);
            }
            if let Some(jout) = json_out {
                if let Some(parent) = jout.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(jout, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Commands::Import { input } => {
            let ds = load_osm_path(&input)?;
            let net = RoadNetwork::compile(&ds);
            println!(
                "Imported {:?}: {} raw ways -> {} junctions, {} directed segments, {} metric lanes, {} turn splines.",
                input,
                ds.ways.len(),
                net.junctions.len(),
                net.segments.len(),
                net.lanes.len(),
                net.connectors.len()
            );
        }
    }
    Ok(())
}

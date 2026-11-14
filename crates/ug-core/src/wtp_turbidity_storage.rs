//! Post-Wildfire Burn-Scar Ash & Turbidity Spikes, Municipal Surface-Water Intake Shutoffs,
//! Enhanced Coagulant Dosing (`Alum` / `FeCl3` + `PAC`), and Finished-Water Storage Drawdown
//! (`crates/ug-core/src/wtp_turbidity_storage.rs`):
//!
//! 1. **Riverine Ash, Turbidity (`NTU`), Pyrogenic `DOC` (`mg/L`) & Alkaline `pH` Chemograph**:
//!    Couples `BurnScarDebrisFlowReport` watershed sediment yields (`Gartner et al., 2014`) and
//!    hydrophobic runoff amplification into a `72-hour` storm-pulse advection-dispersion wave at
//!    the municipal surface-water intake (`3.5 NTU` baseflow -> `400–5,500+ NTU` ash/mud spike,
//!    `DOC` `2.1 -> 28 mg/L`, and `CaO/KOH` wood-ash leachate `pH 7.4 -> 9.2`).
//! 2. **SCADA Raw-Water Intake Shutoffs, Jar-Test Coagulant Stoichiometry & Filter Derating**:
//!    - Models primary coagulant demand (`Al2(SO4)3` Alum vs. `FeCl3` Ferric Chloride), Powdered
//!      Activated Carbon (`PAC` for pyrogenic `DOC` & taste/odor), acid `pH` conditioning, liquid
//!      coagulant bulk-tank inventory depletion (`gallons`), alum/ferric sludge solids (`kg/hr`),
//!      and anthracite/sand filter backwash recycle losses (`6% -> 28%` net `WTP` derating).
//!    - Evaluates **Combined Filter Effluent (`CFE`) turbidity (`NTU`)** against the **US EPA
//!      `IESWTR` `0.30 NTU` 95th-percentile** and **`1.0 NTU` acute breakthrough** limits.
//! 3. **Finished-Water Clearwell + Pressure-Zone Storage Drawdown & `< 20 psi` Boil-Water Advisories**:
//!    Integrates hourly clearwell + elevated pressure-zone storage tanks (`MG` & `% full`) against
//!    diurnal municipal demand (`USFA` / `AWWA` curve), active WUI fire-hydrant demand (`GPM`), and
//!    emergency alluvial groundwater well / intertie blending (`MGD`), tracking `< 20 psi`
//!    depressurization and mandatory **Boil-Water Advisory (`BWA`)** triggers!

use crate::burn_scar_debris_flow::{
    run_burn_scar_debris_flow_monte_carlo, BurnScarDebrisFlowConfig,
};
use crate::diurnal::USFA_DIURNAL_CALL_MULTIPLIERS;
use crate::emergency::EmergencyFacility;
use crate::network::{RoadNetwork, Vec2};
use crate::zoning::Parcel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

pub const WTP_TIMELINE_CSV_HEADER: &str = "hour,raw_river_ntu,raw_doc_mg_l,raw_ash_ph,intake_open_pct,coagulant_dose_mg_l,pac_dose_mg_l,cfe_filtered_ntu,wtp_net_output_mgd,well_intertie_mgd,municipal_demand_mgd,storage_fill_pct,min_zone_pressure_psi,bwa_active";
pub const WTP_FACILITY_CSV_HEADER: &str = "node_id,node_code,node_kind,x_m,y_m,nominal_cap_mg_or_mgd,min_level_or_uptime_pct,peak_ntu_or_psi,bwa_triggered";

/// Municipal `WTP` raw-water intake shutoff, coagulation chemistry & backup intertie operating policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WtpCoagulantAndIntakePolicy {
    /// Keeps river intake open up to `1,400 NTU` on conventional Alum without presedimentation or
    /// acid pH trim — risks rapid coagulant tank exhaustion, clarifier upset & `CFE > 0.30 NTU`
    /// filter turbidity breakthrough!
    RunToFailureAlumBaseline,
    /// Automated SCADA turbidimeter **shuts the surface-water intake whenever `T_raw > shutoff_ntu`**
    /// (preventing filter breakthrough), relying on finished-water clearwell & elevated tanks while
    /// the ash plume passes.
    AutomatedIntakeShutoffAlumOnly,
    /// Full post-fire `WTP` resilience package: off-channel presedimentation polymer basin (`62%`
    /// `NTU` pre-drop) + **enhanced `FeCl3` + `PAC` + `CO2` pH-controlled coagulation** + automated
    /// peak-spike intake shutoff + **emergency groundwater wells / intertie + Stage-2 demand curtailment**!
    PresedFerricPacAndWellIntertie,
}

impl WtpCoagulantAndIntakePolicy {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::RunToFailureAlumBaseline => "Open Intake + Conventional Alum (Breakthrough Risk)",
            Self::AutomatedIntakeShutoffAlumOnly => {
                "SCADA Intake Shutoff + Alum (Storage Drawdown)"
            }
            Self::PresedFerricPacAndWellIntertie => {
                "Presed + FeCl3/PAC Coag + Wells/Intertie"
            }
        }
    }
}

/// Spatial water-supply asset category for viewport rendering and CSV telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterSupplyAssetKind {
    SurfaceWaterRiverIntake,
    PresedimentationAndWtpPlant,
    EmergencyGroundwaterWellfield,
    ClearwellAndPressureZoneTank,
}

impl WaterSupplyAssetKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::SurfaceWaterRiverIntake => "RIV-INTAKE",
            Self::PresedimentationAndWtpPlant => "WTP-COAG",
            Self::EmergencyGroundwaterWellfield => "WELL-TIE",
            Self::ClearwellAndPressureZoneTank => "TANK-PZ",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoagulantJarTestResult {
    pub coagulant_dose_mg_l: f64,
    pub pac_dose_mg_l: f64,
    pub acid_ph_trim_mg_l: f64,
    pub backwash_loss_fraction: f64,
    pub sludge_dry_kg_per_mg: f64,
    pub cfe_filtered_ntu: f64,
    pub ieswtr_03ntu_compliant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WtpTurbidityStorageConfig {
    pub debris_flow: BurnScarDebrisFlowConfig,
    pub policy: WtpCoagulantAndIntakePolicy,
    /// SCADA raw-water turbidity shutoff threshold (`NTU`, `100..=1,500 NTU`, default `350 NTU`).
    pub intake_shutoff_threshold_ntu: f64,
    /// Nominal municipal surface-water `WTP` design capacity (`MGD`, Million Gallons/Day).
    pub wtp_design_capacity_mgd: f64,
    /// Total finished-water clearwell + elevated pressure-zone tank storage capacity (`MG`).
    pub finished_storage_capacity_mg: f64,
    /// Initial finished-water storage fill at storm onset (`%`, `50..=100%`, default `92%`).
    pub initial_storage_fill_pct: f64,
    /// Emergency groundwater wellfield + neighboring utility intertie capacity (`MGD`, `0..=18 MGD`).
    pub emergency_well_intertie_mgd: f64,
    /// On-site bulk liquid coagulant (`Alum` / `FeCl3`) chemical storage inventory (`dry-ton equiv`, `4..=60 tons`).
    pub onsite_coagulant_inventory_tons: f64,
    /// Active WUI / post-fire firefighting hydrant draft (`GPM`, `0..=4,500 GPM`).
    pub active_fire_hydrant_draft_gpm: f64,
    /// Simulation horizon (`hours`, `24..=120 hrs`, default `72 hrs`).
    pub simulation_horizon_hours: usize,
}

impl Default for WtpTurbidityStorageConfig {
    fn default() -> Self {
        Self {
            debris_flow: BurnScarDebrisFlowConfig::default(),
            policy: WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie,
            intake_shutoff_threshold_ntu: 350.0,
            wtp_design_capacity_mgd: 16.0,
            finished_storage_capacity_mg: 12.5,
            initial_storage_fill_pct: 92.0,
            emergency_well_intertie_mgd: 5.5,
            onsite_coagulant_inventory_tons: 22.0,
            active_fire_hydrant_draft_gpm: 1_500.0,
            simulation_horizon_hours: 72,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WtpHourlyTelemetryPoint {
    pub hour: usize,
    pub raw_river_ntu: f64,
    pub raw_doc_mg_l: f64,
    pub raw_ash_ph: f64,
    pub intake_open_pct: f64,
    pub coagulant_dose_mg_l: f64,
    pub pac_dose_mg_l: f64,
    pub cfe_filtered_ntu: f64,
    pub wtp_net_output_mgd: f64,
    pub well_intertie_mgd: f64,
    pub municipal_demand_mgd: f64,
    pub storage_fill_pct: f64,
    pub min_zone_pressure_psi: f64,
    pub bwa_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterSupplyFacilityRow {
    pub node_id: usize,
    pub node_code: String,
    pub kind: WaterSupplyAssetKind,
    pub world_pos: Vec2,
    pub nominal_cap_mg_or_mgd: f64,
    pub min_level_or_uptime_pct: f64,
    pub peak_ntu_or_psi: f64,
    pub bwa_triggered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WtpTurbidityStorageReport {
    pub config: WtpTurbidityStorageConfig,
    pub solve_time_ms: f64,
    /// Source-water post-fire ash & sediment chemograph peaks
    pub peak_raw_river_ntu: f64,
    pub peak_raw_doc_mg_l: f64,
    pub peak_raw_ash_ph: f64,
    pub hours_exceeding_shutoff_ntu: f64,
    /// `WTP` Intake, Coagulant Chemistry & EPA `IESWTR` Filtration performance
    pub intake_shutoff_duration_hrs: f64,
    pub peak_coagulant_dose_mg_l: f64,
    pub total_coagulant_consumed_tons: f64,
    pub coagulant_stockout_prob_pct: f64,
    pub total_alum_ferric_sludge_tons: f64,
    pub peak_cfe_filtered_ntu: f64,
    pub ieswtr_03ntu_compliance_pct: f64,
    /// Finished-Water Clearwell / Tank Drawdown & `< 20 psi` Boil-Water Advisory (`BWA`) outcomes
    pub min_finished_storage_fill_pct: f64,
    pub p10_min_storage_fill_pct: f64,
    pub fire_reserve_breach_prob_pct: f64,
    pub min_high_zone_pressure_psi: f64,
    pub boil_water_advisory_prob_pct: f64,
    pub expected_residents_under_bwa: f64,
    /// Spatial vectors, facility rows, hourly curve & per-parcel GPU cache
    pub raw_to_wtp_to_tank_arcs: Vec<(Vec2, Vec2)>,
    pub facility_rows: Vec<WaterSupplyFacilityRow>,
    pub hourly_timeline: Vec<WtpHourlyTelemetryPoint>,
    pub parcel_water_security_pct: Vec<f64>,
    pub parcel_min_pressure_psi: Vec<f64>,
}

impl WtpTurbidityStorageReport {
    pub fn to_timeline_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.hourly_timeline.len() * 140);
        out.push_str(WTP_TIMELINE_CSV_HEADER);
        out.push('\n');
        for pt in &self.hourly_timeline {
            out.push_str(&format!(
                "{},{:.1},{:.2},{:.2},{:.1},{:.1},{:.1},{:.3},{:.2},{:.2},{:.2},{:.2},{:.1},{}\n",
                pt.hour,
                pt.raw_river_ntu,
                pt.raw_doc_mg_l,
                pt.raw_ash_ph,
                pt.intake_open_pct,
                pt.coagulant_dose_mg_l,
                pt.pac_dose_mg_l,
                pt.cfe_filtered_ntu,
                pt.wtp_net_output_mgd,
                pt.well_intertie_mgd,
                pt.municipal_demand_mgd,
                pt.storage_fill_pct,
                pt.min_zone_pressure_psi,
                pt.bwa_active
            ));
        }
        out
    }

    pub fn to_facility_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.facility_rows.len() * 120);
        out.push_str(WTP_FACILITY_CSV_HEADER);
        out.push('\n');
        for r in &self.facility_rows {
            out.push_str(&format!(
                "{},{},{},{:.1},{:.1},{:.2},{:.1},{:.1},{}\n",
                r.node_id,
                r.node_code,
                r.kind.badge(),
                r.world_pos.x,
                r.world_pos.y,
                r.nominal_cap_mg_or_mgd,
                r.min_level_or_uptime_pct,
                r.peak_ntu_or_psi,
                r.bwa_triggered
            ));
        }
        out
    }

    pub fn export_timeline_csv_path(&self, path: &Path) -> Result<usize, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, self.to_timeline_csv_string()).map_err(|e| e.to_string())?;
        Ok(self.hourly_timeline.len())
    }
}

/// Empirical post-wildfire jar-test coagulant stoichiometry (`Hohner et al., 2016/2019`; `Writer et al., 2014` AWWA),
/// filter backwash water loss fraction, clarifier dry-sludge production (`kg / Million Gallons`),
/// and Combined Filter Effluent (`CFE`) turbidity (`NTU`, EPA `IESWTR <= 0.30 NTU` standard).
pub fn compute_postfire_coagulant_jar_test(
    raw_ntu_entering_plant: f64,
    raw_doc_mg_l: f64,
    raw_ash_ph: f64,
    enhanced_ferric_pac_mode: bool,
    coagulant_tank_empty: bool,
) -> CoagulantJarTestResult {
    let ntu = raw_ntu_entering_plant.max(0.5);
    let doc = raw_doc_mg_l.max(1.0);
    let ph = raw_ash_ph.clamp(6.5, 10.2);

    // Primary coagulant demand (`mg/L`):
    // - Colloidal ash/clay surface area scales ~ `ntu^0.58`
    // - Pyrogenic aromatic DOC exerts stoichiometric charge-neutralization demand (`~4.8 mg coag / mg DOC` per Writer et al. 2014)
    // - Alkaline wood-ash pH > 8.0 penalizes unbuffered Alum (`Al(OH)4-` soluble aluminate!) unless acid-trimmed or switched to FeCl3
    let ph_penalty = if enhanced_ferric_pac_mode {
        1.0 + (ph - 8.2).max(0.0) * 0.08
    } else {
        1.0 + (ph - 7.8).max(0.0) * 0.42
    };
    let raw_coag_mg_l =
        (11.0 + 3.65 * ntu.powf(0.56) + 4.75 * (doc - 1.8).max(0.0)) * ph_penalty;
    let coag_dose_mg_l = raw_coag_mg_l.clamp(12.0, 265.0);

    let pac_dose_mg_l = if enhanced_ferric_pac_mode && doc > 4.5 {
        ((doc - 3.5) * 2.4).clamp(5.0, 45.0)
    } else {
        0.0
    };
    let acid_ph_trim_mg_l = if enhanced_ferric_pac_mode && ph > 7.6 {
        ((ph - 7.4) * 18.0).clamp(0.0, 42.0)
    } else {
        0.0
    };

    // Rapid sand/anthracite filter headloss & backwash recycle loss (`5%` clean -> `28%` under heavy floc solids)
    let backwash_loss_fraction = (0.045 + 0.00024 * ntu + 0.00065 * coag_dose_mg_l).clamp(0.045, 0.30);

    // AWWA dry sludge solids (`1 mg/L` across `1 MG` = `3.7854 kg`): suspended solids (`~1.25 * NTU` mg/L) + metal hydroxide precipitate (`0.44 * coag` + `PAC`)
    let sludge_mg_l = 1.25 * ntu + 0.44 * coag_dose_mg_l + pac_dose_mg_l;
    let sludge_dry_kg_per_mg = sludge_mg_l * 3.78541;

    // Combined Filter Effluent (`CFE`) turbidity (`NTU`):
    // - If chemical tank runs dry during ash spike: uncoagulated colloidal ash passes straight through (`CFE > 2.5 NTU`!)
    // - Conventional Alum without pH control upsets above `~420 NTU` or `pH > 8.5`
    // - Enhanced FeCl3 + PAC + pH trim maintains `CFE <= 0.30 NTU` up to `~950 NTU` influent!
    let cfe_ntu = if coagulant_tank_empty {
        (0.45 + 0.018 * ntu).clamp(0.55, 18.0)
    } else if enhanced_ferric_pac_mode {
        let overload = (ntu - 920.0).max(0.0) / 650.0;
        (0.06 + 0.00018 * ntu + 0.32 * overload * overload).clamp(0.05, 4.5)
    } else {
        let alk_upset = (ph - 8.15).max(0.0) * 0.28;
        let overload = (ntu - 360.0).max(0.0) / 380.0;
        (0.09 + 0.00042 * ntu + alk_upset + 0.55 * overload * overload).clamp(0.07, 9.5)
    };

    CoagulantJarTestResult {
        coagulant_dose_mg_l: if coagulant_tank_empty {
            0.0
        } else {
            coag_dose_mg_l
        },
        pac_dose_mg_l,
        acid_ph_trim_mg_l,
        backwash_loss_fraction,
        sludge_dry_kg_per_mg,
        cfe_filtered_ntu: cfe_ntu,
        ieswtr_03ntu_compliant: cfe_ntu <= 0.30,
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn u01_from_u64(v: u64) -> f64 {
    ((v >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Executes a `Rayon`-parallelized `72-hour` Monte Carlo simulation coupling `BurnScarDebrisFlowReport`
/// ash/sediment pulses to municipal surface-water intake shutoffs, coagulant jar-test dosing,
/// finished-water storage tank drawdown, and `< 20 psi` / `IESWTR` Boil-Water Advisories (`BWA`).
pub fn run_wtp_turbidity_storage_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &WtpTurbidityStorageConfig,
) -> WtpTurbidityStorageReport {
    let t0 = Instant::now();
    let df_rep = run_burn_scar_debris_flow_monte_carlo(net, parcels, facilities, &cfg.debris_flow);
    let b = net.bounds;
    let span_x = (b.max_x - b.min_x).max(160.0);
    let span_y = (b.max_y - b.min_y).max(160.0);

    // 1. Place Spatial Water-Supply Chain: River Intake -> Presed/WTP -> Wellfield Intertie + 3 PZ Storage Tanks
    let intake_pos = Vec2 {
        x: b.min_x + 0.16 * span_x,
        y: b.min_y + 0.80 * span_y,
    };
    let wtp_pos = Vec2 {
        x: b.min_x + 0.28 * span_x,
        y: b.min_y + 0.72 * span_y,
    };
    let wellfield_pos = Vec2 {
        x: b.min_x + 0.44 * span_x,
        y: b.min_y + 0.86 * span_y,
    };
    let tank_positions = [
        (
            "Tank-PZ1-Clearwell",
            Vec2 {
                x: b.min_x + 0.36 * span_x,
                y: b.min_y + 0.62 * span_y,
            },
            0.45,
        ),
        (
            "Tank-PZ2-Midtown",
            Vec2 {
                x: b.min_x + 0.58 * span_x,
                y: b.min_y + 0.48 * span_y,
            },
            0.35,
        ),
        (
            "Tank-PZ3-FoothillWUI",
            Vec2 {
                x: b.min_x + 0.78 * span_x,
                y: b.min_y + 0.28 * span_y,
            },
            0.20,
        ),
    ];

    // 2. Compute Baseline Municipal Potable Demand (`MGD`) + Active Fire-Hydrant Suppression Draft (`MGD`)
    let total_residents: f64 = parcels.iter().map(Parcel::residents).sum::<f64>().max(600.0);
    let total_jobs: f64 = parcels.iter().map(Parcel::jobs).sum::<f64>().max(200.0);
    // Scale per-capita demand into realistic municipal district/city WTP loading (`~45–65%` of design `wtp_design_capacity_mgd`)
    let raw_model_mgd = (total_residents * 115.0 + total_jobs * 45.0) / 1_000_000.0;
    let base_municipal_mgd = raw_model_mgd.max(cfg.wtp_design_capacity_mgd * 0.54);
    // 1,500 GPM = 2.16 MGD during peak initial firefighting window (Hours 0..14)
    let fire_draft_mgd = (cfg.active_fire_hydrant_draft_gpm.max(0.0) * 1_440.0) / 1_000_000.0;

    // 3. Couple Burn-Scar Sediment Volume (`m^3`) & Runoff Amplification into Peak River Turbidity (`NTU`), `DOC`, and `pH`!
    let sed_k_m3 = (df_rep.total_predicted_sediment_volume_m3 / 1_000.0).max(0.2);
    let mean_peak_ntu =
        (85.0 + 92.0 * sed_k_m3.powf(0.78) * df_rep.runoff_amplification_ratio.sqrt())
            .clamp(95.0, 6_500.0);
    let mean_peak_doc =
        (3.2 + 1.85 * sed_k_m3.powf(0.52) + 0.14 * (df_rep.mean_post_fire_curve_number - 60.0).max(0.0))
            .clamp(3.5, 34.0);
    let mean_peak_ph =
        (7.45 + 0.26 * sed_k_m3.ln().max(0.0) + 0.015 * (df_rep.mean_post_fire_curve_number - 65.0).max(0.0))
            .clamp(7.5, 9.45);

    let horizon_hrs = cfg.simulation_horizon_hours.clamp(24, 144);
    let trials = cfg.debris_flow.wildfire.monte_carlo_trials.clamp(16, 2_500);
    let seed0 = cfg.debris_flow.wildfire.random_seed ^ 0x5754_504E_5455_0520;
    let enhanced_mode = cfg.policy == WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie;

    #[derive(Clone)]
    struct WtpTrial {
        peak_ntu: f64,
        peak_doc: f64,
        peak_ph: f64,
        hrs_gt_shutoff: f64,
        shutoff_hrs: f64,
        peak_coag_mg_l: f64,
        coag_used_tons: f64,
        coag_stockout: bool,
        sludge_tons: f64,
        peak_cfe_ntu: f64,
        ieswtr_ok_hrs_pct: f64,
        min_storage_pct: f64,
        min_pz3_psi: f64,
        fire_reserve_breached: bool,
        bwa_triggered: bool,
        hr_ntu: Vec<f64>,
        hr_doc: Vec<f64>,
        hr_ph: Vec<f64>,
        hr_open: Vec<f64>,
        hr_coag: Vec<f64>,
        hr_pac: Vec<f64>,
        hr_cfe: Vec<f64>,
        hr_wtp_mgd: Vec<f64>,
        hr_well_mgd: Vec<f64>,
        hr_dem_mgd: Vec<f64>,
        hr_stor_pct: Vec<f64>,
        hr_psi: Vec<f64>,
        hr_bwa: Vec<bool>,
    }

    let wtp_trials: Vec<WtpTrial> = (0..trials)
        .into_par_iter()
        .map(|tr| {
            let s = splitmix64(seed0.wrapping_add((tr as u64).wrapping_mul(0x9E37_79B9)));
            let plume_jit = 0.76 + 0.52 * u01_from_u64(s);
            let arrival_hr = 4.5 + 3.0 * u01_from_u64(splitmix64(s ^ 0x33));
            // Recession time constant (`hours`): multi-pulse tailing over 14–26 hours
            let recession_tau_hr = 9.5 + 5.5 * u01_from_u64(splitmix64(s ^ 0x77));

            let cap_mg = cfg.finished_storage_capacity_mg.max(1.5);
            let mut stor_mg = cap_mg * (cfg.initial_storage_fill_pct.clamp(35.0, 100.0) / 100.0);
            let mut coag_rem_tons = cfg.onsite_coagulant_inventory_tons.max(0.5);
            let mut coag_used_tons = 0.0;
            let mut sludge_tons = 0.0;
            let mut coag_stockout = false;

            let mut peak_ntu: f64 = 3.5;
            let mut peak_doc: f64 = 2.1;
            let mut peak_ph: f64 = 7.4;
            let mut hrs_gt_shutoff = 0.0;
            let mut shutoff_hrs = 0.0;
            let mut peak_coag_mg_l: f64 = 0.0;
            let mut peak_cfe_ntu: f64 = 0.08;
            let mut ieswtr_ok_count = 0usize;
            let mut min_storage_pct = (stor_mg / cap_mg) * 100.0;
            let mut min_pz3_psi: f64 = 64.0;
            let mut bwa_triggered = false;

            let mut hr_ntu = Vec::with_capacity(horizon_hrs);
            let mut hr_doc = Vec::with_capacity(horizon_hrs);
            let mut hr_ph = Vec::with_capacity(horizon_hrs);
            let mut hr_open = Vec::with_capacity(horizon_hrs);
            let mut hr_coag = Vec::with_capacity(horizon_hrs);
            let mut hr_pac = Vec::with_capacity(horizon_hrs);
            let mut hr_cfe = Vec::with_capacity(horizon_hrs);
            let mut hr_wtp_mgd = Vec::with_capacity(horizon_hrs);
            let mut hr_well_mgd = Vec::with_capacity(horizon_hrs);
            let mut hr_dem_mgd = Vec::with_capacity(horizon_hrs);
            let mut hr_stor_pct = Vec::with_capacity(horizon_hrs);
            let mut hr_psi = Vec::with_capacity(horizon_hrs);
            let mut hr_bwa = Vec::with_capacity(horizon_hrs);

            for h in 0..horizon_hrs {
                let t = h as f64;
                // Gamma-like asymmetric riverine ash/sediment pulse `u * exp(1 - u)` peaking at `arrival_hr + 4.5h`
                let pulse = if t <= arrival_hr {
                    0.0
                } else {
                    let u = (t - arrival_hr) / recession_tau_hr;
                    let main_wave = (u * (1.0 - u).exp()).max(0.0);
                    // Secondary after-storm tributary wave 14h later
                    let u2 = ((t - (arrival_hr + 13.5)).max(0.0)) / (recession_tau_hr * 0.85);
                    let sec_wave = if t > arrival_hr + 13.5 {
                        0.42 * (u2 * (1.0 - u2).exp()).max(0.0)
                    } else {
                        0.0
                    };
                    (main_wave + sec_wave).clamp(0.0, 1.35)
                };
                // First-flush wood ash `pH` spikes slightly ahead of peak mineral sediment!
                let ash_flush = if t <= arrival_hr - 1.0 {
                    0.0
                } else {
                    let ua = (t - (arrival_hr - 1.0)) / (recession_tau_hr * 0.65);
                    (ua * (1.0 - ua).exp()).clamp(0.0, 1.2)
                };

                let raw_ntu = 3.5 + (mean_peak_ntu * plume_jit - 3.5).max(0.0) * pulse;
                let raw_doc = 2.1 + (mean_peak_doc * plume_jit.sqrt() - 2.1).max(0.0) * pulse.powf(0.85);
                let raw_ph = 7.42 + (mean_peak_ph - 7.42).max(0.0) * ash_flush;

                peak_ntu = peak_ntu.max(raw_ntu);
                peak_doc = peak_doc.max(raw_doc);
                peak_ph = peak_ph.max(raw_ph);
                if raw_ntu > cfg.intake_shutoff_threshold_ntu {
                    hrs_gt_shutoff += 1.0;
                }

                // Off-channel presedimentation basin + polymer pre-settling (`62%` NTU removal in Policy 3)
                let plant_influent_ntu = if enhanced_mode {
                    (raw_ntu * 0.38).max(2.5)
                } else {
                    raw_ntu
                };

                let effective_cutoff = match cfg.policy {
                    WtpCoagulantAndIntakePolicy::RunToFailureAlumBaseline => 1_450.0,
                    WtpCoagulantAndIntakePolicy::AutomatedIntakeShutoffAlumOnly => {
                        cfg.intake_shutoff_threshold_ntu
                    }
                    WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie => {
                        // Presedimentation + FeCl3 allows safe treatment until presed effluent exceeds `1.85 * cutoff`
                        cfg.intake_shutoff_threshold_ntu * 1.85
                    }
                };

                let intake_open = plant_influent_ntu <= effective_cutoff;
                let (wtp_net_mgd, coag_mg_l, pac_mg_l, cfe_ntu, ieswtr_ok) = if intake_open {
                    let jar = compute_postfire_coagulant_jar_test(
                        plant_influent_ntu,
                        raw_doc,
                        raw_ph,
                        enhanced_mode,
                        coag_rem_tons <= 0.02,
                    );
                    let gross_mg_hr = cfg.wtp_design_capacity_mgd / 24.0;
                    // Chemical consumption: `1 mg/L * 1 MG = 8.3454 lb = 0.0041727 short tons`
                    let tons_needed_hr =
                        gross_mg_hr * (jar.coagulant_dose_mg_l + 0.5 * jar.pac_dose_mg_l) * 0.004_172_7;
                    if coag_rem_tons < tons_needed_hr {
                        coag_stockout = true;
                        coag_rem_tons = 0.0;
                    } else {
                        coag_rem_tons -= tons_needed_hr;
                        coag_used_tons += tons_needed_hr;
                    }
                    sludge_tons += (gross_mg_hr * jar.sludge_dry_kg_per_mg) / 907.185;
                    let net_mgd =
                        cfg.wtp_design_capacity_mgd * (1.0 - jar.backwash_loss_fraction);
                    (
                        net_mgd,
                        jar.coagulant_dose_mg_l,
                        jar.pac_dose_mg_l,
                        jar.cfe_filtered_ntu,
                        jar.ieswtr_03ntu_compliant,
                    )
                } else {
                    shutoff_hrs += 1.0;
                    (0.0, 0.0, 0.0, 0.08, true)
                };

                peak_coag_mg_l = peak_coag_mg_l.max(coag_mg_l);
                peak_cfe_ntu = peak_cfe_ntu.max(cfe_ntu);
                if ieswtr_ok {
                    ieswtr_ok_count += 1;
                }

                // Emergency groundwater wells / intertie activate during intake shutoff or when storage < 78%
                let cur_fill_pct = (stor_mg / cap_mg) * 100.0;
                let well_mgd = match cfg.policy {
                    WtpCoagulantAndIntakePolicy::RunToFailureAlumBaseline => 0.0,
                    WtpCoagulantAndIntakePolicy::AutomatedIntakeShutoffAlumOnly => {
                        if !intake_open || cur_fill_pct < 65.0 {
                            cfg.emergency_well_intertie_mgd * 0.35
                        } else {
                            0.0
                        }
                    }
                    WtpCoagulantAndIntakePolicy::PresedFerricPacAndWellIntertie => {
                        if !intake_open || cur_fill_pct < 82.0 {
                            cfg.emergency_well_intertie_mgd
                        } else {
                            cfg.emergency_well_intertie_mgd * 0.30
                        }
                    }
                };

                let diurn = USFA_DIURNAL_CALL_MULTIPLIERS[h % 24];
                // Stage-2 IPAWS water curtailment trims outdoor irrigation/commercial demand by 18% in Policy 3
                let curtail_fac = if enhanced_mode && (!intake_open || cur_fill_pct < 75.0) {
                    0.82
                } else {
                    1.0
                };
                let fire_mgd_now = if h < 16 { fire_draft_mgd } else { fire_draft_mgd * 0.15 };
                let demand_mgd =
                    base_municipal_mgd * (0.58 + 0.42 * diurn) * curtail_fac + fire_mgd_now;

                // Hourly tank mass balance (`MG`)
                let net_delta_mg = (wtp_net_mgd + well_mgd - demand_mgd) / 24.0;
                stor_mg = (stor_mg + net_delta_mg).clamp(0.0, cap_mg);
                let next_fill_pct = (stor_mg / cap_mg) * 100.0;
                min_storage_pct = min_storage_pct.min(next_fill_pct);

                // Foothill High-Zone (`PZ3`) residual pressure (`psi`): nominal ~58 psi at >=65% tank level,
                // dropping below 20 psi mandatory BWA threshold when tank storage falls below ~24%!
                let head_scale = if next_fill_pct >= 65.0 {
                    1.0
                } else {
                    (0.14 + 0.86 * (next_fill_pct / 65.0)).clamp(0.10, 1.0)
                };
                let pz3_psi = (58.0 * head_scale - if h < 16 { 6.5 } else { 2.0 }).max(2.0);
                min_pz3_psi = min_pz3_psi.min(pz3_psi);

                // EPA / Primacy Agency Boil-Water Advisory (`BWA`) triggered by either:
                // (a) Acute filter turbidity breakthrough (`CFE > 1.0 NTU`), OR
                // (b) Distribution depressurization `< 20.0 psi`!
                if cfe_ntu > 1.0 || pz3_psi < 20.0 {
                    bwa_triggered = true;
                }

                hr_ntu.push(raw_ntu);
                hr_doc.push(raw_doc);
                hr_ph.push(raw_ph);
                hr_open.push(if intake_open { 100.0 } else { 0.0 });
                hr_coag.push(coag_mg_l);
                hr_pac.push(pac_mg_l);
                hr_cfe.push(cfe_ntu);
                hr_wtp_mgd.push(wtp_net_mgd);
                hr_well_mgd.push(well_mgd);
                hr_dem_mgd.push(demand_mgd);
                hr_stor_pct.push(next_fill_pct);
                hr_psi.push(pz3_psi);
                hr_bwa.push(bwa_triggered);
            }

            WtpTrial {
                peak_ntu,
                peak_doc,
                peak_ph,
                hrs_gt_shutoff,
                shutoff_hrs,
                peak_coag_mg_l,
                coag_used_tons,
                coag_stockout,
                sludge_tons,
                peak_cfe_ntu,
                ieswtr_ok_hrs_pct: (ieswtr_ok_count as f64 / horizon_hrs as f64) * 100.0,
                min_storage_pct,
                min_pz3_psi,
                fire_reserve_breached: min_storage_pct < 35.0,
                bwa_triggered,
                hr_ntu,
                hr_doc,
                hr_ph,
                hr_open,
                hr_coag,
                hr_pac,
                hr_cfe,
                hr_wtp_mgd,
                hr_well_mgd,
                hr_dem_mgd,
                hr_stor_pct,
                hr_psi,
                hr_bwa,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let peak_raw_river_ntu = wtp_trials.iter().map(|r| r.peak_ntu).sum::<f64>() * inv_t;
    let peak_raw_doc_mg_l = wtp_trials.iter().map(|r| r.peak_doc).sum::<f64>() * inv_t;
    let peak_raw_ash_ph = wtp_trials.iter().map(|r| r.peak_ph).sum::<f64>() * inv_t;
    let hours_exceeding_shutoff_ntu =
        wtp_trials.iter().map(|r| r.hrs_gt_shutoff).sum::<f64>() * inv_t;
    let intake_shutoff_duration_hrs =
        wtp_trials.iter().map(|r| r.shutoff_hrs).sum::<f64>() * inv_t;
    let peak_coagulant_dose_mg_l =
        wtp_trials.iter().map(|r| r.peak_coag_mg_l).sum::<f64>() * inv_t;
    let total_coagulant_consumed_tons =
        wtp_trials.iter().map(|r| r.coag_used_tons).sum::<f64>() * inv_t;
    let coagulant_stockout_prob_pct = wtp_trials.iter().filter(|r| r.coag_stockout).count() as f64
        * 100.0
        * inv_t;
    let total_alum_ferric_sludge_tons =
        wtp_trials.iter().map(|r| r.sludge_tons).sum::<f64>() * inv_t;
    let peak_cfe_filtered_ntu = wtp_trials.iter().map(|r| r.peak_cfe_ntu).sum::<f64>() * inv_t;
    let ieswtr_03ntu_compliance_pct =
        wtp_trials.iter().map(|r| r.ieswtr_ok_hrs_pct).sum::<f64>() * inv_t;

    let min_finished_storage_fill_pct =
        wtp_trials.iter().map(|r| r.min_storage_pct).sum::<f64>() * inv_t;
    let mut sorted_stor: Vec<f64> = wtp_trials.iter().map(|r| r.min_storage_pct).collect();
    sorted_stor.sort_by(|a, b_v| a.partial_cmp(b_v).unwrap_or(std::cmp::Ordering::Equal));
    let p10_min_storage_fill_pct = sorted_stor[((sorted_stor.len() as f64 * 0.10).floor() as usize)
        .min(sorted_stor.len() - 1)];

    let fire_reserve_breach_prob_pct = wtp_trials
        .iter()
        .filter(|r| r.fire_reserve_breached)
        .count() as f64
        * 100.0
        * inv_t;
    let min_high_zone_pressure_psi =
        wtp_trials.iter().map(|r| r.min_pz3_psi).sum::<f64>() * inv_t;
    let boil_water_advisory_prob_pct =
        wtp_trials.iter().filter(|r| r.bwa_triggered).count() as f64 * 100.0 * inv_t;

    // Turbidity breakthrough at the WTP places 100% of system residents on BWA; localized `<20 psi` tank drawdown affects upper pressure zones (~42% of residents)
    let systemwide_breakthrough_frac = wtp_trials
        .iter()
        .filter(|r| r.peak_cfe_ntu > 1.0)
        .count() as f64
        * inv_t;
    let pressure_only_bwa_frac = (boil_water_advisory_prob_pct / 100.0 - systemwide_breakthrough_frac)
        .max(0.0);
    let expected_residents_under_bwa =
        total_residents * (systemwide_breakthrough_frac + 0.44 * pressure_only_bwa_frac);

    // 4. Build 72-Hour Ensemble Mean Timeline
    let mut hourly_timeline = Vec::with_capacity(horizon_hrs);
    for h in 0..horizon_hrs {
        let m_psi = wtp_trials.iter().map(|r| r.hr_psi[h]).sum::<f64>() * inv_t;
        let m_cfe = wtp_trials.iter().map(|r| r.hr_cfe[h]).sum::<f64>() * inv_t;
        let bwa_votes = wtp_trials.iter().filter(|r| r.hr_bwa[h]).count();
        hourly_timeline.push(WtpHourlyTelemetryPoint {
            hour: h,
            raw_river_ntu: wtp_trials.iter().map(|r| r.hr_ntu[h]).sum::<f64>() * inv_t,
            raw_doc_mg_l: wtp_trials.iter().map(|r| r.hr_doc[h]).sum::<f64>() * inv_t,
            raw_ash_ph: wtp_trials.iter().map(|r| r.hr_ph[h]).sum::<f64>() * inv_t,
            intake_open_pct: wtp_trials.iter().map(|r| r.hr_open[h]).sum::<f64>() * inv_t,
            coagulant_dose_mg_l: wtp_trials.iter().map(|r| r.hr_coag[h]).sum::<f64>() * inv_t,
            pac_dose_mg_l: wtp_trials.iter().map(|r| r.hr_pac[h]).sum::<f64>() * inv_t,
            cfe_filtered_ntu: m_cfe,
            wtp_net_output_mgd: wtp_trials.iter().map(|r| r.hr_wtp_mgd[h]).sum::<f64>() * inv_t,
            well_intertie_mgd: wtp_trials.iter().map(|r| r.hr_well_mgd[h]).sum::<f64>() * inv_t,
            municipal_demand_mgd: wtp_trials.iter().map(|r| r.hr_dem_mgd[h]).sum::<f64>() * inv_t,
            storage_fill_pct: wtp_trials.iter().map(|r| r.hr_stor_pct[h]).sum::<f64>() * inv_t,
            min_zone_pressure_psi: m_psi,
            bwa_active: bwa_votes * 2 >= trials || m_cfe > 1.0 || m_psi < 20.0,
        });
    }

    // 5. Build Facility Telemetry Rows & Transmission Arcs
    let mut raw_to_wtp_to_tank_arcs = vec![(intake_pos, wtp_pos), (wellfield_pos, wtp_pos)];
    for &(_, t_pos, _) in &tank_positions {
        raw_to_wtp_to_tank_arcs.push((wtp_pos, t_pos));
    }

    let intake_uptime_pct =
        ((horizon_hrs as f64 - intake_shutoff_duration_hrs) / horizon_hrs as f64 * 100.0)
            .clamp(0.0, 100.0);
    let mut facility_rows = vec![
        WaterSupplyFacilityRow {
            node_id: 0,
            node_code: "River-Intake-Raw".to_string(),
            kind: WaterSupplyAssetKind::SurfaceWaterRiverIntake,
            world_pos: intake_pos,
            nominal_cap_mg_or_mgd: cfg.wtp_design_capacity_mgd,
            min_level_or_uptime_pct: intake_uptime_pct,
            peak_ntu_or_psi: peak_raw_river_ntu,
            bwa_triggered: false,
        },
        WaterSupplyFacilityRow {
            node_id: 1,
            node_code: "WTP-Coag-Filtration".to_string(),
            kind: WaterSupplyAssetKind::PresedimentationAndWtpPlant,
            world_pos: wtp_pos,
            nominal_cap_mg_or_mgd: cfg.wtp_design_capacity_mgd,
            min_level_or_uptime_pct: ieswtr_03ntu_compliance_pct,
            peak_ntu_or_psi: peak_cfe_filtered_ntu,
            bwa_triggered: peak_cfe_filtered_ntu > 1.0,
        },
        WaterSupplyFacilityRow {
            node_id: 2,
            node_code: "Wellfield-Intertie".to_string(),
            kind: WaterSupplyAssetKind::EmergencyGroundwaterWellfield,
            world_pos: wellfield_pos,
            nominal_cap_mg_or_mgd: cfg.emergency_well_intertie_mgd,
            min_level_or_uptime_pct: if cfg.policy
                == WtpCoagulantAndIntakePolicy::RunToFailureAlumBaseline
            {
                0.0
            } else {
                98.0
            },
            peak_ntu_or_psi: 0.12,
            bwa_triggered: false,
        },
    ];

    for (idx, &(code, t_pos, cap_share)) in tank_positions.iter().enumerate() {
        let zone_elev_pen = (idx as f64) * 6.5;
        let zone_min_psi = (min_high_zone_pressure_psi + (2 - idx) as f64 * 8.5 - zone_elev_pen * 0.2)
            .clamp(2.0, 75.0);
        facility_rows.push(WaterSupplyFacilityRow {
            node_id: facility_rows.len(),
            node_code: code.to_string(),
            kind: WaterSupplyAssetKind::ClearwellAndPressureZoneTank,
            world_pos: t_pos,
            nominal_cap_mg_or_mgd: cfg.finished_storage_capacity_mg * cap_share,
            min_level_or_uptime_pct: (min_finished_storage_fill_pct - (idx as f64) * 3.5)
                .clamp(0.0, 100.0),
            peak_ntu_or_psi: zone_min_psi,
            bwa_triggered: zone_min_psi < 20.0 || peak_cfe_filtered_ntu > 1.0,
        });
    }

    // 6. Per-Parcel Water Security & Minimum Pressure Mapping across Low/Mid/High Pressure Zones
    let mut parcel_water_security_pct = Vec::with_capacity(parcels.len());
    let mut parcel_min_pressure_psi = Vec::with_capacity(parcels.len());
    for p in parcels {
        // North-East / upper foothill parcels sit at higher hydraulic elevation -> lower residual `psi` during tank drawdown!
        let ux = ((p.center.x - b.min_x) / span_x).clamp(0.0, 1.0);
        let uy = (1.0 - (p.center.y - b.min_y) / span_y).clamp(0.0, 1.0);
        let elev_factor = (0.55 * ux + 0.45 * uy).clamp(0.0, 1.0);
        let p_psi = (min_high_zone_pressure_psi + (1.0 - elev_factor) * 19.5).clamp(2.0, 78.0);
        let cfe_pen = if peak_cfe_filtered_ntu > 0.30 {
            ((peak_cfe_filtered_ntu - 0.30) * 38.0).clamp(0.0, 65.0)
        } else {
            0.0
        };
        let psi_score = ((p_psi / 42.0) * 100.0).clamp(5.0, 100.0);
        let sec_pct = (0.55 * psi_score + 0.45 * min_finished_storage_fill_pct - cfe_pen)
            .clamp(3.0, 100.0);
        parcel_water_security_pct.push(sec_pct);
        parcel_min_pressure_psi.push(p_psi);
    }

    WtpTurbidityStorageReport {
        config: cfg.clone(),
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        peak_raw_river_ntu,
        peak_raw_doc_mg_l,
        peak_raw_ash_ph,
        hours_exceeding_shutoff_ntu,
        intake_shutoff_duration_hrs,
        peak_coagulant_dose_mg_l,
        total_coagulant_consumed_tons,
        coagulant_stockout_prob_pct,
        total_alum_ferric_sludge_tons,
        peak_cfe_filtered_ntu,
        ieswtr_03ntu_compliance_pct,
        min_finished_storage_fill_pct,
        p10_min_storage_fill_pct,
        fire_reserve_breach_prob_pct,
        min_high_zone_pressure_psi,
        boil_water_advisory_prob_pct,
        expected_residents_under_bwa,
        raw_to_wtp_to_tank_arcs,
        facility_rows,
        hourly_timeline,
        parcel_water_security_pct,
        parcel_min_pressure_psi,
    }
}

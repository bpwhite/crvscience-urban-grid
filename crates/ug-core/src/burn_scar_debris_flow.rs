//! Post-Wildfire Burn-Scar Hydrophobic Runoff, USGS Debris-Flow Hazard Basins (`Staley et al.,
//! 2016/2017 M1` + `Gartner et al., 2014`), and `FHWA HDS-5` Stormwater Culvert / Roadway Washout
//! Coupling (`crates/ug-core/src/burn_scar_debris_flow.rs`):
//!
//! 1. **USFS `BAER` Soil Burn Severity (`dNBR`) & Hydrophobic NRCS Curve Number (`CN`) Shift**:
//!    Couples directly to `WildfireAlertWestConfig` burn footprints, mapping Unburned (`CN ~ 60`),
//!    Low (`CN ~ 73`), Moderate (`CN ~ 86`), and High water-repellent hydrophobic mineral soils
//!    (`CN ~ 94`, `dNBR >= 440`) into NRCS `TR-55` overland sheetflow depth (`mm`), clear-water
//!    peak discharge `Q_w` (`m^3/s`), and sediment-bulked peak discharge `Q_bulked = BF * Q_w`.
//! 2. **USGS Post-Fire Debris-Flow Likelihood (`Staley et al., 2016/2017 M1`) & Volume (`Gartner et al., 2014`)**:
//!    - `X = -3.63 + 0.41 * R15 * MH23 + 0.67 * R15 * (dNBR / 1000) + 0.70 * R15 * KF`,
//!      `P_df = exp(X) / (1 + exp(X))`, where `R15 = I15_mm_hr / 4.0` (`mm` in 15 min), plus exact
//!      inversion for the basin `P = 50%` 15-minute rainfall intensity threshold `I15_P50` (`mm/hr`).
//!    - `ln(V_m3) = 4.22 + 0.39 * sqrt(I15_mm_hr) + 0.36 * ln(Bmh_km2) + 0.13 * sqrt(Relief_m)`
//!      (`Gartner et al., 2014` emergency assessment sediment yield) & combined USGS Hazard Rating.
//! 3. **`FHWA HDS-5` Stormwater Culvert Woody-Slash Plugging, `HW/D` Overtopping & Roadway Washout**:
//!    Evaluates submerged inlet-control orifice headwater ratio `HW/D` under post-fire woody debris
//!    plugging (`40–82%` barrel occlusion without grizzly trash racks), embankment crown overtopping
//!    (`HW/D > 1.50`), roadway scour washout closures (`seg.closed = true`), severed co-located
//!    buried water/sewer mains, and muddy stormflow `RDII` sanitary sewer surcharge!

use crate::emergency::EmergencyFacility;
use crate::network::{RoadNetwork, Vec2};
use crate::wildfire_alertwest_ohaz::{run_wildfire_alertwest_monte_carlo, WildfireAlertWestConfig};
use crate::zoning::Parcel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

pub const USGS_BASIN_CSV_HEADER: &str = "basin_id,basin_code,headwater_x_m,headwater_y_m,outlet_x_m,outlet_y_m,area_km2,bmh_km2,relief_m,prop_hm23,mean_dnbr,effective_cn,i15_p50_thresh_mm_hr,m1_likelihood_pct,gartner_volume_m3,peak_bulked_q_m3s,combined_hazard";
pub const CULVERT_WASHOUT_CSV_HEADER: &str = "culvert_id,road_segment_id,road_name,basin_id,x_m,y_m,barrels,diameter_m,debris_plug_pct,peak_hw_over_d,washout_prob_pct,hydraulic_state";

/// USFS `BAER` (*Burned Area Emergency Response*, `RMRS-GTR-243`) post-fire Soil Burn Severity (`SBS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoilBurnSeverityClass {
    UnburnedOrVeryLow,
    LowSeverity,
    ModerateHydrophobic,
    HighHydrophobic,
}

impl SoilBurnSeverityClass {
    pub fn from_dnbr(dnbr: f64) -> Self {
        if dnbr >= 440.0 {
            Self::HighHydrophobic
        } else if dnbr >= 270.0 {
            Self::ModerateHydrophobic
        } else if dnbr >= 100.0 {
            Self::LowSeverity
        } else {
            Self::UnburnedOrVeryLow
        }
    }

    pub fn post_fire_curve_number(self) -> f64 {
        match self {
            Self::UnburnedOrVeryLow => 60.0,
            Self::LowSeverity => 73.0,
            Self::ModerateHydrophobic => 86.0,
            Self::HighHydrophobic => 94.0,
        }
    }
}

/// USGS Operational Combined Post-Fire Debris-Flow Hazard Class (`Staley et al., 2017`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsgsCombinedHazardRating {
    Low,
    Moderate,
    High,
    Extreme,
}

impl UsgsCombinedHazardRating {
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "LOW",
            Self::Moderate => "MODERATE",
            Self::High => "HIGH",
            Self::Extreme => "EXTREME",
        }
    }

    pub fn from_prob_and_volume(prob_01: f64, vol_m3: f64) -> Self {
        let p_score = if prob_01 >= 0.80 {
            4
        } else if prob_01 >= 0.60 {
            3
        } else if prob_01 >= 0.40 {
            2
        } else if prob_01 >= 0.20 {
            1
        } else {
            0
        };
        let v_score = if vol_m3 >= 100_000.0 {
            4
        } else if vol_m3 >= 10_000.0 {
            3
        } else if vol_m3 >= 1_000.0 {
            2
        } else {
            1
        };
        let sum = p_score + v_score;
        if sum >= 6 {
            Self::Extreme
        } else if sum >= 4 {
            Self::High
        } else if sum >= 2 {
            Self::Moderate
        } else {
            Self::Low
        }
    }
}

/// USFS `BAER` hillslope, channel & highway-drainage stabilization treatment posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaerWatershedMitigationPolicy {
    /// Raw hydrophobic burn scar (`CN = 94` on high SBS), unguarded CMP culverts (`up to 78%` woody
    /// debris plugging), and zero upslope sediment retention.
    UntreatedBurnScar,
    /// USFS `BAER` helicopter wood-strand / straw mulching on steep moderate/high-severity slopes
    /// (lowers hydrophobic `CN` & sheet/rill sediment delivery), but culverts remain un-retrofitted.
    AerialWoodStrandMulchOnly,
    /// Full `BAER` + DOT package: aerial wood-strand mulching + **grizzly debris-deflector trash racks**
    /// + storm-patrol excavator cleanout + **`FHWA HDS-5` twin-box culvert upsizing** on arterials +
    /// alluvial-fan debris basins!
    FullBaerMulchTrashRacksAndUpsizing,
}

impl BaerWatershedMitigationPolicy {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::UntreatedBurnScar => "Untreated Hydrophobic Burn Scar",
            Self::AerialWoodStrandMulchOnly => "BAER Aerial Wood-Strand Mulch Only",
            Self::FullBaerMulchTrashRacksAndUpsizing => {
                "Full BAER Mulch + Grizzly Racks + Box Culverts"
            }
        }
    }
}

/// `FHWA HDS-5` stormwater crossing hydraulic status during peak post-fire storm runoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CulvertHydraulicState {
    /// `HW/D <= 1.0`: Unsubmerged free-surface flow; roadway intact.
    PassableClear,
    /// `1.0 < HW/D <= 1.35`: Submerged inlet-control orifice flow within embankment freeboard.
    SurchargedInlet,
    /// `1.35 < HW/D <= 1.60`: Shoulder/pavement overtopping sheetflow (`50%` speed reduction).
    OvertoppingFlooded,
    /// `HW/D > 1.60` or debris-flow avulsion: Roadway embankment scour breach (`seg.closed = true`).
    EmbankmentWashoutClosed,
}

impl CulvertHydraulicState {
    pub fn badge(self) -> &'static str {
        match self {
            Self::PassableClear => "CLV-OK",
            Self::SurchargedInlet => "CLV-SUR",
            Self::OvertoppingFlooded => "CLV-OVT",
            Self::EmbankmentWashoutClosed => "CLV-WASH",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnScarDebrisFlowConfig {
    pub wildfire: WildfireAlertWestConfig,
    pub baer_policy: BaerWatershedMitigationPolicy,
    /// Design storm peak 15-minute rainfall intensity `I15` (`mm/hr`, `8..=96 mm/hr`).
    pub peak_i15_intensity_mm_hr: f64,
    /// Total 6-hour convective/atmospheric-river storm depth `P_storm` (`mm`, `15..=180 mm`).
    pub storm_6hr_depth_mm: f64,
    /// STATSGO/SSURGO fine-earth soil erodibility factor `KF` (`0.10..=0.50`, default `0.28`).
    pub soil_kf_erodibility: f64,
    /// Number of delineated upslope WUI drainage sub-basins (`4..=12`, default `8`).
    pub delineated_basins_count: usize,
    /// Baseline corrugated metal pipe (`CMP`) culvert barrel diameter (`meters`, `0.6..=2.4 m`).
    pub baseline_culvert_diameter_m: f64,
}

impl Default for BurnScarDebrisFlowConfig {
    fn default() -> Self {
        Self {
            wildfire: WildfireAlertWestConfig::default(),
            baer_policy: BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing,
            peak_i15_intensity_mm_hr: 34.0,
            storm_6hr_depth_mm: 68.0,
            soil_kf_erodibility: 0.28,
            delineated_basins_count: 8,
            baseline_culvert_diameter_m: 1.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsgsDebrisFlowBasinRow {
    pub basin_id: usize,
    pub basin_code: String,
    pub headwater_pos: Vec2,
    pub outlet_pos: Vec2,
    pub area_km2: f64,
    pub bmh_km2: f64,
    pub relief_m: f64,
    pub prop_hm23: f64,
    pub mean_dnbr: f64,
    pub effective_cn: f64,
    pub i15_p50_thresh_mm_hr: f64,
    pub m1_likelihood_pct: f64,
    pub gartner_volume_m3: f64,
    pub peak_bulked_q_m3s: f64,
    pub combined_hazard: UsgsCombinedHazardRating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StormwaterCulvertCrossingRow {
    pub culvert_id: usize,
    pub road_segment_id: usize,
    pub road_name: String,
    pub basin_id: usize,
    pub world_pos: Vec2,
    pub barrels: usize,
    pub diameter_m: f64,
    pub debris_plug_pct: f64,
    pub peak_hw_over_d: f64,
    pub washout_prob_pct: f64,
    pub hydraulic_state: CulvertHydraulicState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntensityThresholdCurvePoint {
    pub i15_mm_hr: f64,
    pub mean_basin_likelihood_pct: f64,
    pub max_basin_likelihood_pct: f64,
    pub expected_washed_out_culverts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnScarDebrisFlowReport {
    pub config: BurnScarDebrisFlowConfig,
    pub solve_time_ms: f64,
    /// Watershed & Hydrophobic Runoff aggregates
    pub burned_watershed_area_km2: f64,
    pub mean_post_fire_curve_number: f64,
    pub mean_hydrophobic_runoff_depth_mm: f64,
    pub unburned_baseline_runoff_mm: f64,
    pub runoff_amplification_ratio: f64,
    /// USGS `M1` & `Gartner (2014)` Debris-Flow Basin metrics
    pub firewide_median_i15_p50_mm_hr: f64,
    pub basins_exceeding_p50_count: usize,
    pub total_predicted_sediment_volume_m3: f64,
    pub baer_trapped_or_prevented_sediment_m3: f64,
    /// `FHWA HDS-5` Culvert, Roadway Washout & Coupled Lifeline impacts
    pub evaluated_culverts_count: usize,
    pub surcharged_or_overtopped_culverts: usize,
    pub expected_washed_out_culverts: f64,
    pub p90_washed_out_culverts: usize,
    pub post_washout_4min_fire_cov_pct: f64,
    pub coupled_rdii_sewer_inflow_mm_hr: f64,
    /// Spatial vectors, tables & per-parcel GPU cache
    pub basin_rows: Vec<UsgsDebrisFlowBasinRow>,
    pub culvert_rows: Vec<StormwaterCulvertCrossingRow>,
    pub id_threshold_curve: Vec<IntensityThresholdCurvePoint>,
    pub washed_out_segment_ids: Vec<usize>,
    pub overtopped_slow_segment_ids: Vec<usize>,
    pub parcel_runoff_depth_mm: Vec<f64>,
    pub parcel_debris_hazard_pct: Vec<f64>,
}

impl BurnScarDebrisFlowReport {
    pub fn to_basin_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.basin_rows.len() * 160);
        out.push_str(USGS_BASIN_CSV_HEADER);
        out.push('\n');
        for b in &self.basin_rows {
            out.push_str(&format!(
                "{},{},{:.1},{:.1},{:.1},{:.1},{:.3},{:.3},{:.1},{:.3},{:.1},{:.1},{:.2},{:.2},{:.1},{:.2},{}\n",
                b.basin_id,
                b.basin_code,
                b.headwater_pos.x,
                b.headwater_pos.y,
                b.outlet_pos.x,
                b.outlet_pos.y,
                b.area_km2,
                b.bmh_km2,
                b.relief_m,
                b.prop_hm23,
                b.mean_dnbr,
                b.effective_cn,
                b.i15_p50_thresh_mm_hr,
                b.m1_likelihood_pct,
                b.gartner_volume_m3,
                b.peak_bulked_q_m3s,
                b.combined_hazard.label()
            ));
        }
        out
    }

    pub fn to_culvert_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.culvert_rows.len() * 140);
        out.push_str(CULVERT_WASHOUT_CSV_HEADER);
        out.push('\n');
        for c in &self.culvert_rows {
            let safe_name = c.road_name.replace(',', ";");
            out.push_str(&format!(
                "{},{},{},{},{:.1},{:.1},{},{:.2},{:.1},{:.2},{:.2},{}\n",
                c.culvert_id,
                c.road_segment_id,
                safe_name,
                c.basin_id,
                c.world_pos.x,
                c.world_pos.y,
                c.barrels,
                c.diameter_m,
                c.debris_plug_pct,
                c.peak_hw_over_d,
                c.washout_prob_pct,
                c.hydraulic_state.badge()
            ));
        }
        out
    }

    pub fn export_basin_csv_path(&self, path: &Path) -> Result<usize, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, self.to_basin_csv_string()).map_err(|e| e.to_string())?;
        Ok(self.basin_rows.len())
    }

    pub fn export_culvert_csv_path(&self, path: &Path) -> Result<usize, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, self.to_culvert_csv_string()).map_err(|e| e.to_string())?;
        Ok(self.culvert_rows.len())
    }
}

/// Exact **USGS `M1` Logistic Regression Model** (`Staley et al., 2016/2017`, USGS OFR 2016-1106):
/// `X = -3.63 + 0.41 * R15 * MH23 + 0.67 * R15 * (dNBR / 1000.0) + 0.70 * R15 * KF`
/// `P = exp(X) / (1.0 + exp(X))`
/// where `r15_mm` is 15-minute rainfall accumulation in `mm` (`= I15_mm_hr / 4.0`).
#[inline]
pub fn usgs_staley_m1_debris_flow_likelihood(
    r15_mm: f64,
    prop_hm23: f64,
    dnbr: f64,
    kf: f64,
) -> f64 {
    let r = r15_mm.max(0.0);
    let x1 = prop_hm23.clamp(0.0, 1.0);
    let x2 = (dnbr.clamp(0.0, 1_200.0)) / 1_000.0;
    let x3 = kf.clamp(0.02, 0.85);
    let link_x = -3.63 + r * (0.41 * x1 + 0.67 * x2 + 0.70 * x3);
    if link_x >= 35.0 {
        1.0
    } else if link_x <= -35.0 {
        0.0
    } else {
        let ex = link_x.exp();
        ex / (1.0 + ex)
    }
}

/// Exact inversion of `Staley et al. (2017) M1` for `P = 0.50` (`X = 0`), returning the basin's
/// 15-minute rainfall intensity threshold `I15_P50` in `mm/hr` (`4.0 * R15_P50`).
#[inline]
pub fn usgs_staley_m1_i15_p50_threshold_mm_hr(prop_hm23: f64, dnbr: f64, kf: f64) -> f64 {
    let x1 = prop_hm23.clamp(0.0, 1.0);
    let x2 = (dnbr.clamp(0.0, 1_200.0)) / 1_000.0;
    let x3 = kf.clamp(0.02, 0.85);
    let slope = (0.41 * x1 + 0.67 * x2 + 0.70 * x3).max(1e-4);
    let r15_p50_mm = 3.63 / slope;
    4.0 * r15_p50_mm
}

/// Exact **USGS Emergency Assessment Sediment Volume Model** (`Gartner et al., 2014`):
/// `ln(V_m3) = 4.22 + 0.39 * sqrt(I15_mm_hr) + 0.36 * ln(Bmh_km2) + 0.13 * sqrt(Relief_m)`
#[inline]
pub fn usgs_gartner_2014_debris_volume_m3(
    i15_mm_hr: f64,
    bmh_km2: f64,
    relief_m: f64,
) -> f64 {
    if bmh_km2 <= 1e-4 || i15_mm_hr <= 0.1 {
        return 0.0;
    }
    let ln_v = 4.22
        + 0.39 * i15_mm_hr.max(0.0).sqrt()
        + 0.36 * bmh_km2.max(0.005).ln()
        + 0.13 * relief_m.max(5.0).sqrt();
    ln_v.clamp(-5.0, 14.2).exp()
}

/// Evaluates NRCS / USFS `BAER` post-fire hydrophobic direct storm runoff depth (`mm`) and
/// sediment-bulked peak discharge `Q_bulked` (`m^3/s`) using `Ia = 0.08 * S` for water-repellent soils.
pub fn nrcs_baer_hydrophobic_runoff_and_peak_q(
    storm_depth_mm: f64,
    i15_mm_hr: f64,
    area_km2: f64,
    cn_eff: f64,
    sediment_conc_cv: f64,
) -> (f64, f64) {
    let cn = cn_eff.clamp(35.0, 98.5);
    let s_mm = 25_400.0 / cn - 254.0;
    // Water-repellent burn-scar soils exhibit reduced initial abstraction (`Ia ~ 0.08 S` vs `0.20 S`)
    let ia_mm = if cn >= 82.0 { 0.08 * s_mm } else { 0.20 * s_mm };
    let p = storm_depth_mm.max(0.0);
    let q_depth_mm = if p <= ia_mm {
        0.0
    } else {
        let num = (p - ia_mm) * (p - ia_mm);
        num / (p - ia_mm + s_mm).max(1e-6)
    };

    // Rational / SCS unit-hydrograph peak clear-water discharge driven by short-burst `I15`
    let runoff_coeff = if p > 1.0 {
        (q_depth_mm / p).clamp(0.05, 0.96)
    } else {
        0.10
    };
    // Q_w [m^3/s] = 0.2778 * C * I_eff [mm/hr] * A [km^2]
    let q_clear_m3s = 0.2778 * runoff_coeff * i15_mm_hr.max(0.0) * area_km2.max(0.02);
    let cv = sediment_conc_cv.clamp(0.0, 0.58);
    let bulking_factor = 1.0 / (1.0 - cv);
    (q_depth_mm, q_clear_m3s * bulking_factor)
}

/// Evaluates `FHWA HDS-5` (*Hydraulic Design of Highway Culverts*) inlet-control headwater-to-diameter
/// ratio `HW / D` and hydraulic washout state given bulked discharge `q_bulked_m3s`, `barrels`,
/// barrel `diameter_m`, and woody-debris occlusion fraction `plug_frac` (`0..0.92`).
pub fn fhwa_hds5_culvert_headwater_ratio(
    q_bulked_m3s: f64,
    barrels: usize,
    diameter_m: f64,
    plug_frac: f64,
) -> (f64, CulvertHydraulicState) {
    let d = diameter_m.max(0.35);
    let n_b = barrels.max(1) as f64;
    let open_frac = (1.0 - plug_frac.clamp(0.0, 0.92)).max(0.08);
    let a_single = std::f64::consts::PI * 0.25 * d * d;
    let a_eff_total = n_b * a_single * open_frac;
    // Dimensionless orifice Froude discharge intensity `Q* = Q / (A_eff * sqrt(g * D))`
    let q_star = q_bulked_m3s.max(0.0) / (a_eff_total * (9.80665 * d).sqrt()).max(1e-6);

    // HDS-5 unsubmerged-to-submerged smooth inlet-control transition (`CMP / Concrete headwall`):
    // At Q* = 0.50 -> HW/D ~ 0.92; at Q* = 0.75 -> HW/D ~ 1.38; at Q* >= 0.90 -> HW/D > 1.65 (overtopping!)
    let hw_over_d = if q_star <= 0.48 {
        (1.78 * q_star).clamp(0.05, 0.88)
    } else {
        (0.56 + 1.44 * q_star * q_star).clamp(0.88, 4.80)
    };

    let state = if hw_over_d <= 1.00 {
        CulvertHydraulicState::PassableClear
    } else if hw_over_d <= 1.35 {
        CulvertHydraulicState::SurchargedInlet
    } else if hw_over_d <= 1.60 {
        CulvertHydraulicState::OvertoppingFlooded
    } else {
        CulvertHydraulicState::EmbankmentWashoutClosed
    };
    (hw_over_d, state)
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

/// Executes a `Rayon`-parallelized post-wildfire burn-scar hydrophobic runoff, USGS `M1`/`Gartner`
/// debris-flow basin, and `FHWA HDS-5` stormwater culvert washout Monte Carlo simulation.
pub fn run_burn_scar_debris_flow_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &BurnScarDebrisFlowConfig,
) -> BurnScarDebrisFlowReport {
    let t0 = Instant::now();
    // 1. Run coupled `ALERTWest / OHAZ` wildfire module to obtain burn footprint & parcel burn severity
    let wf_rep = run_wildfire_alertwest_monte_carlo(net, parcels, facilities, &cfg.wildfire);
    let b = net.bounds;
    let span_x = (b.max_x - b.min_x).max(160.0);
    let span_y = (b.max_y - b.min_y).max(160.0);
    let city_center = Vec2 {
        x: 0.5 * (b.min_x + b.max_x),
        y: 0.5 * (b.min_y + b.max_y),
    };

    // 2. Delineate `M` Upslope WUI Drainage Sub-Basins radially around the WUI foothill perimeter,
    //    concentrated around the upwind `true_ignition_pos` burn scar!
    let n_basins = cfg.delineated_basins_count.clamp(4, 12);
    let ign_angle = (wf_rep.true_ignition_pos.y - city_center.y)
        .atan2(wf_rep.true_ignition_pos.x - city_center.x);

    let mulch_active = matches!(
        cfg.baer_policy,
        BaerWatershedMitigationPolicy::AerialWoodStrandMulchOnly
            | BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing
    );
    let full_baer =
        cfg.baer_policy == BaerWatershedMitigationPolicy::FullBaerMulchTrashRacksAndUpsizing;

    // Burn-size scaling factor from wildfire report (`ALERTWest` early containment shrinks `Bmh`!)
    let burn_scale = (wf_rep.final_burned_area_acres / 85.0).clamp(0.28, 2.15);

    #[derive(Clone)]
    struct BasinGeom {
        id: usize,
        code: String,
        headwater: Vec2,
        outlet: Vec2,
        area_km2: f64,
        bmh_km2: f64,
        relief_m: f64,
        prop_hm23: f64,
        raw_dnbr: f64,
        eff_dnbr: f64,
        eff_cn: f64,
    }

    let mut basins = Vec::with_capacity(n_basins);
    for k in 0..n_basins {
        let frac = (k as f64) / (n_basins as f64) - 0.5;
        let ang = ign_angle + frac * std::f64::consts::PI * 1.15;
        let hw = Vec2 {
            x: (city_center.x + ang.cos() * 0.47 * span_x).clamp(b.min_x + 12.0, b.max_x - 12.0),
            y: (city_center.y + ang.sin() * 0.47 * span_y).clamp(b.min_y + 12.0, b.max_y - 12.0),
        };
        let out = Vec2 {
            x: (city_center.x + ang.cos() * 0.23 * span_x).clamp(b.min_x + 35.0, b.max_x - 35.0),
            y: (city_center.y + ang.sin() * 0.23 * span_y).clamp(b.min_y + 35.0, b.max_y - 35.0),
        };
        let dist_to_ign = hw.dist(wf_rep.true_ignition_pos);
        let prox = (-((dist_to_ign / (0.42 * span_x.max(span_y))).powi(2))).exp();

        let area_km2 = 1.15 + 0.38 * ((k * 3 % 5) as f64);
        let raw_dnbr = (145.0 + 565.0 * prox * burn_scale.min(1.35)).clamp(85.0, 820.0);
        // USFS BAER aerial wood-strand mulching protects bare hydrophobic ash against raindrop splash/rilling
        let eff_dnbr = if mulch_active && raw_dnbr >= 270.0 {
            raw_dnbr * 0.74
        } else {
            raw_dnbr
        };
        let sbs = SoilBurnSeverityClass::from_dnbr(raw_dnbr);
        let base_cn = sbs.post_fire_curve_number();
        let eff_cn = if mulch_active && base_cn >= 84.0 {
            (base_cn - 7.5).max(74.0)
        } else {
            base_cn
        };
        let burned_mh_frac = if raw_dnbr >= 270.0 {
            (0.22 + 0.62 * prox * burn_scale.min(1.3)).clamp(0.08, 0.92)
        } else {
            0.04
        };
        let steep_frac = (cfg.wildfire.terrain_slope_pct / 36.0).clamp(0.25, 0.92);
        let prop_hm23 = (burned_mh_frac * steep_frac * if mulch_active { 0.78 } else { 1.0 })
            .clamp(0.02, 0.90);
        let bmh_km2 = (area_km2 * burned_mh_frac).max(0.01);
        let relief_m = 165.0 + cfg.wildfire.terrain_slope_pct * 7.5 + (k % 3) as f64 * 42.0;

        basins.push(BasinGeom {
            id: k,
            code: format!("USGS-Basin-{:02}", k + 1),
            headwater: hw,
            outlet: out,
            area_km2,
            bmh_km2,
            relief_m,
            prop_hm23,
            raw_dnbr,
            eff_dnbr,
            eff_cn,
        });
    }

    // 3. Match Each Basin's Alluvial-Fan Drainage Channel to its Closest Roadway Culvert Crossing!
    let mut culvert_Bindings: Vec<(usize, usize, Vec2)> = Vec::new();
    let mut used_segs: Vec<usize> = Vec::new();
    for bg in &basins {
        let mid_channel = Vec2 {
            x: 0.45 * bg.headwater.x + 0.55 * bg.outlet.x,
            y: 0.45 * bg.headwater.y + 0.55 * bg.outlet.y,
        };
        let mut best_sid = None;
        let mut best_d = f64::INFINITY;
        let mut best_pt = mid_channel;
        for (sid, seg) in net.segments.iter().enumerate() {
            if used_segs.contains(&sid) || seg.base_pts.is_empty() {
                continue;
            }
            let pt = seg.base_pts[seg.base_pts.len() / 2];
            let d = pt.dist(mid_channel);
            if d < best_d {
                best_d = d;
                best_sid = Some(sid);
                best_pt = pt;
            }
        }
        if let Some(sid) = best_sid {
            used_segs.push(sid);
            culvert_Bindings.push((bg.id, sid, best_pt));
        }
    }

    // 4. Run Parallel Monte Carlo across Spatially Variable Convective Storm Cells & Debris Plugging
    let trials = cfg.wildfire.monte_carlo_trials.clamp(16, 2_500);
    let seed0 = cfg.wildfire.random_seed ^ 0xBAEB_DF10_2017;
    let n_culv = culvert_Bindings.len();

    #[derive(Clone)]
    struct StormTrial {
        basin_prob: Vec<f64>,
        basin_vol_m3: Vec<f64>,
        basin_q_m3s: Vec<f64>,
        basin_runoff_mm: Vec<f64>,
        culv_hwd: Vec<f64>,
        culv_plug_pct: Vec<f64>,
        culv_washed: Vec<bool>,
        washed_count: usize,
        trapped_sed_m3: f64,
    }

    let storm_trials: Vec<StormTrial> = (0..trials)
        .into_par_iter()
        .map(|tr| {
            let s = splitmix64(seed0.wrapping_add((tr as u64).wrapping_mul(0x9E37_79B9)));
            let cell_jit = 0.80 + 0.44 * u01_from_u64(s);

            let mut basin_prob = Vec::with_capacity(n_basins);
            let mut basin_vol_m3 = Vec::with_capacity(n_basins);
            let mut basin_q_m3s = Vec::with_capacity(n_basins);
            let mut basin_runoff_mm = Vec::with_capacity(n_basins);
            let mut trapped_sed_m3 = 0.0;

            for bg in &basins {
                let b_jit = 0.88 + 0.26 * u01_from_u64(splitmix64(s ^ (bg.id as u64 + 31)));
                let i15_local = cfg.peak_i15_intensity_mm_hr * cell_jit * b_jit;
                let r15_mm = i15_local / 4.0;
                let p_df = usgs_staley_m1_debris_flow_likelihood(
                    r15_mm,
                    bg.prop_hm23,
                    bg.eff_dnbr,
                    cfg.soil_kf_erodibility,
                );
                let raw_vol =
                    usgs_gartner_2014_debris_volume_m3(i15_local, bg.bmh_km2, bg.relief_m);
                let delivered_vol = if full_baer {
                    trapped_sed_m3 += raw_vol * 0.48;
                    raw_vol * 0.52
                } else if mulch_active {
                    trapped_sed_m3 += raw_vol * 0.24;
                    raw_vol * 0.76
                } else {
                    raw_vol
                };
                // Volumetric sediment concentration `C_v` rises with `p_df` (`0.08` clear flood -> `0.46` debris surge)
                let cv = (0.06 + 0.40 * p_df * if full_baer { 0.68 } else { 1.0 }).clamp(0.05, 0.50);
                let (q_mm, q_bulk) = nrcs_baer_hydrophobic_runoff_and_peak_q(
                    cfg.storm_6hr_depth_mm * cell_jit,
                    i15_local,
                    bg.area_km2,
                    bg.eff_cn,
                    cv,
                );
                basin_prob.push(p_df * 100.0);
                basin_vol_m3.push(delivered_vol);
                basin_q_m3s.push(q_bulk);
                basin_runoff_mm.push(q_mm);
            }

            let mut culv_hwd = Vec::with_capacity(n_culv);
            let mut culv_plug_pct = Vec::with_capacity(n_culv);
            let mut culv_washed = Vec::with_capacity(n_culv);
            let mut washed_count = 0usize;

            for (c_idx, &(b_id, sid, _)) in culvert_Bindings.iter().enumerate() {
                let p_jit = 0.85 + 0.30 * u01_from_u64(splitmix64(s ^ (c_idx as u64 + 97)));
                let p_df01 = basin_prob[b_id] / 100.0;
                let base_lanes = net.segments.get(sid).map(|sg| sg.lane_count).unwrap_or(1);
                // Arterials (`>= 2 lanes/dir`) get twin-barrel or upsized box culverts under Full BAER!
                let barrels = if full_baer && base_lanes >= 2 {
                    2
                } else if base_lanes >= 3 {
                    2
                } else {
                    1
                };
                let diam_m = if full_baer {
                    cfg.baseline_culvert_diameter_m * 1.38
                } else {
                    cfg.baseline_culvert_diameter_m
                };
                // Woody slash + cobble debris plugging at culvert inlet:
                // Untreated: up to `78%` blocked when `p_df` is high; BAER Grizzly Trash Racks cut plugging by ~72%!
                let raw_plug = (0.14 + 0.64 * p_df01 * p_jit).clamp(0.05, 0.86);
                let plug_frac = if full_baer {
                    (raw_plug * 0.28).clamp(0.04, 0.30)
                } else if mulch_active {
                    (raw_plug * 0.80).clamp(0.05, 0.75)
                } else {
                    raw_plug
                };

                let (hwd, state) = fhwa_hds5_culvert_headwater_ratio(
                    basin_q_m3s[b_id],
                    barrels,
                    diam_m,
                    plug_frac,
                );
                let avulsion_trigger = basin_vol_m3[b_id] > 9_500.0 && p_df01 > 0.68 && !full_baer;
                let is_washed =
                    state == CulvertHydraulicState::EmbankmentWashoutClosed || avulsion_trigger;
                if is_washed {
                    washed_count += 1;
                }
                culv_hwd.push(hwd);
                culv_plug_pct.push(plug_frac * 100.0);
                culv_washed.push(is_washed);
            }

            StormTrial {
                basin_prob,
                basin_vol_m3,
                basin_q_m3s,
                basin_runoff_mm,
                culv_hwd,
                culv_plug_pct,
                culv_washed,
                washed_count,
                trapped_sed_m3,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);

    // 5. Aggregate Basin Rows & Fire-Wide Median `I15_P50` Threshold
    let mut basin_rows = Vec::with_capacity(n_basins);
    let mut p50_list = Vec::with_capacity(n_basins);
    let mut basins_exceeding_p50 = 0usize;
    let mut total_sed_m3 = 0.0;
    let mut runoff_sum_mm = 0.0;
    let mut cn_sum = 0.0;
    let mut burned_area_km2 = 0.0;

    for (b_i, bg) in basins.iter().enumerate() {
        let i15_p50 = usgs_staley_m1_i15_p50_threshold_mm_hr(
            bg.prop_hm23,
            bg.eff_dnbr,
            cfg.soil_kf_erodibility,
        );
        p50_list.push(i15_p50);
        if cfg.peak_i15_intensity_mm_hr >= i15_p50 {
            basins_exceeding_p50 += 1;
        }
        let m_prob = storm_trials.iter().map(|r| r.basin_prob[b_i]).sum::<f64>() * inv_t;
        let m_vol = storm_trials
            .iter()
            .map(|r| r.basin_vol_m3[b_i])
            .sum::<f64>()
            * inv_t;
        let m_q = storm_trials.iter().map(|r| r.basin_q_m3s[b_i]).sum::<f64>() * inv_t;
        let m_roff = storm_trials
            .iter()
            .map(|r| r.basin_runoff_mm[b_i])
            .sum::<f64>()
            * inv_t;
        total_sed_m3 += m_vol;
        runoff_sum_mm += m_roff;
        cn_sum += bg.eff_cn;
        burned_area_km2 += bg.bmh_km2;

        basin_rows.push(UsgsDebrisFlowBasinRow {
            basin_id: bg.id,
            basin_code: bg.code.clone(),
            headwater_pos: bg.headwater,
            outlet_pos: bg.outlet,
            area_km2: bg.area_km2,
            bmh_km2: bg.bmh_km2,
            relief_m: bg.relief_m,
            prop_hm23: bg.prop_hm23,
            mean_dnbr: bg.raw_dnbr,
            effective_cn: bg.eff_cn,
            i15_p50_thresh_mm_hr: i15_p50,
            m1_likelihood_pct: m_prob,
            gartner_volume_m3: m_vol,
            peak_bulked_q_m3s: m_q,
            combined_hazard: UsgsCombinedHazardRating::from_prob_and_volume(m_prob / 100.0, m_vol),
        });
    }

    p50_list.sort_by(|a, b_v| a.partial_cmp(b_v).unwrap_or(std::cmp::Ordering::Equal));
    let firewide_median_i15_p50_mm_hr = p50_list[p50_list.len() / 2];

    let (unburned_runoff_mm, _) = nrcs_baer_hydrophobic_runoff_and_peak_q(
        cfg.storm_6hr_depth_mm,
        cfg.peak_i15_intensity_mm_hr,
        1.5,
        60.0,
        0.04,
    );
    let mean_hydrophobic_runoff_depth_mm = runoff_sum_mm / (n_basins as f64);
    let runoff_amplification_ratio =
        (mean_hydrophobic_runoff_depth_mm / unburned_runoff_mm.max(0.8)).max(1.0);
    let baer_trapped_or_prevented_sediment_m3 =
        storm_trials.iter().map(|r| r.trapped_sed_m3).sum::<f64>() * inv_t;

    // 6. Aggregate Culvert Rows, Roadway Washout Closures & Post-Washout 4-Min Emergency Coverage
    let mut culvert_rows = Vec::with_capacity(n_culv);
    let mut washed_out_segment_ids = Vec::new();
    let mut overtopped_slow_segment_ids = Vec::new();
    let mut surcharged_or_overtopped_culverts = 0usize;

    for (c_idx, &(b_id, sid, c_pos)) in culvert_Bindings.iter().enumerate() {
        let m_hwd = storm_trials.iter().map(|r| r.culv_hwd[c_idx]).sum::<f64>() * inv_t;
        let m_plug = storm_trials
            .iter()
            .map(|r| r.culv_plug_pct[c_idx])
            .sum::<f64>()
            * inv_t;
        let wash_pct = storm_trials
            .iter()
            .filter(|r| r.culv_washed[c_idx])
            .count() as f64
            * 100.0
            * inv_t;
        let base_lanes = net.segments.get(sid).map(|sg| sg.lane_count).unwrap_or(1);
        let barrels = if full_baer && base_lanes >= 2 {
            2
        } else if base_lanes >= 3 {
            2
        } else {
            1
        };
        let diam_m = if full_baer {
            cfg.baseline_culvert_diameter_m * 1.38
        } else {
            cfg.baseline_culvert_diameter_m
        };
        let state = if wash_pct >= 45.0 || m_hwd > 1.60 {
            washed_out_segment_ids.push(sid);
            surcharged_or_overtopped_culverts += 1;
            CulvertHydraulicState::EmbankmentWashoutClosed
        } else if m_hwd > 1.35 {
            overtopped_slow_segment_ids.push(sid);
            surcharged_or_overtopped_culverts += 1;
            CulvertHydraulicState::OvertoppingFlooded
        } else if m_hwd > 1.00 {
            surcharged_or_overtopped_culverts += 1;
            CulvertHydraulicState::SurchargedInlet
        } else {
            CulvertHydraulicState::PassableClear
        };
        let rname = net
            .segments
            .get(sid)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("Corridor #{sid}"));

        culvert_rows.push(StormwaterCulvertCrossingRow {
            culvert_id: c_idx,
            road_segment_id: sid,
            road_name: rname,
            basin_id: b_id,
            world_pos: c_pos,
            barrels,
            diameter_m: diam_m,
            debris_plug_pct: m_plug,
            peak_hw_over_d: m_hwd,
            washout_prob_pct: wash_pct,
            hydraulic_state: state,
        });
    }

    let expected_washed_out_culverts =
        storm_trials.iter().map(|r| r.washed_count as f64).sum::<f64>() * inv_t;
    let mut sorted_wash: Vec<usize> = storm_trials.iter().map(|r| r.washed_count).collect();
    sorted_wash.sort_unstable();
    let p90_washed_out_culverts =
        sorted_wash[((sorted_wash.len() as f64 * 0.90).floor() as usize).min(sorted_wash.len() - 1)];

    // Evaluate post-washout NFPA 1710 4-min fire coverage on a cloned network with washed-out culverts closed!
    let mut washed_net = net.clone();
    for &sid in &washed_out_segment_ids {
        if let Some(seg) = washed_net.segments.get_mut(sid) {
            seg.closed = true;
        }
    }
    let mut temp_parcels = parcels.to_vec();
    let (cov_summary, _, _) = crate::emergency::evaluate_emergency_coverage(
        &washed_net,
        &mut temp_parcels,
        facilities,
        &crate::emergency::EvpConfig::default(),
    );

    // 7. Sample USGS Intensity-Duration Curve across `I15 = 8..=80 mm/hr` for interactive plotting
    let mut id_threshold_curve = Vec::with_capacity(13);
    for step in 0..=12 {
        let i15_s = 8.0 + (step as f64) * 6.0;
        let r15_s = i15_s / 4.0;
        let mut sum_p = 0.0;
        let mut max_p: f64 = 0.0;
        let mut exp_w = 0.0;
        for bg in &basins {
            let p = usgs_staley_m1_debris_flow_likelihood(
                r15_s,
                bg.prop_hm23,
                bg.eff_dnbr,
                cfg.soil_kf_erodibility,
            ) * 100.0;
            sum_p += p;
            max_p = max_p.max(p);
            if p >= 55.0 && !full_baer {
                exp_w += p / 100.0;
            } else if p >= 78.0 && full_baer {
                exp_w += 0.35 * (p / 100.0);
            }
        }
        id_threshold_curve.push(IntensityThresholdCurvePoint {
            i15_mm_hr: i15_s,
            mean_basin_likelihood_pct: sum_p / (n_basins as f64),
            max_basin_likelihood_pct: max_p,
            expected_washed_out_culverts: exp_w.min(n_culv as f64),
        });
    }

    // 8. Per-Parcel Runoff & Alluvial-Fan Debris-Flow Runout Hazard Mapping
    let mut parcel_runoff_depth_mm = Vec::with_capacity(parcels.len());
    let mut parcel_debris_hazard_pct = Vec::with_capacity(parcels.len());
    for (p_idx, p) in parcels.iter().enumerate() {
        let nearest_b = basin_rows
            .iter()
            .min_by(|a, b_r| {
                p.center
                    .dist(a.outlet_pos)
                    .partial_cmp(&p.center.dist(b_r.outlet_pos))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let wf_risk = wf_rep
            .parcel_burn_risk_pct
            .get(p_idx)
            .copied()
            .unwrap_or(0.0);
        let p_cn = (62.0 + 0.32 * wf_risk - if mulch_active { 5.5 } else { 0.0 }).clamp(58.0, 95.0);
        let (p_roff, _) = nrcs_baer_hydrophobic_runoff_and_peak_q(
            cfg.storm_6hr_depth_mm,
            cfg.peak_i15_intensity_mm_hr,
            (p.area_ha / 100.0).max(0.02),
            p_cn,
            0.08,
        );
        let fan_haz = if let Some(br) = nearest_b {
            let d_out = p.center.dist(br.outlet_pos);
            let runout_decay = (-((d_out / 340.0).powi(2))).exp();
            (br.m1_likelihood_pct * (0.35 + 0.65 * runout_decay)).clamp(0.0, 100.0)
        } else {
            wf_risk * 0.5
        };
        parcel_runoff_depth_mm.push(p_roff);
        parcel_debris_hazard_pct.push(fan_haz);
    }

    let coupled_rdii_sewer_inflow_mm_hr =
        (cfg.peak_i15_intensity_mm_hr * (runoff_amplification_ratio * 0.52).clamp(0.65, 1.85))
            .clamp(0.0, 80.0);

    BurnScarDebrisFlowReport {
        config: cfg.clone(),
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        burned_watershed_area_km2: burned_area_km2,
        mean_post_fire_curve_number: cn_sum / (n_basins as f64),
        mean_hydrophobic_runoff_depth_mm,
        unburned_baseline_runoff_mm: unburned_runoff_mm,
        runoff_amplification_ratio,
        firewide_median_i15_p50_mm_hr,
        basins_exceeding_p50_count: basins_exceeding_p50,
        total_predicted_sediment_volume_m3: total_sed_m3,
        baer_trapped_or_prevented_sediment_m3,
        evaluated_culverts_count: n_culv,
        surcharged_or_overtopped_culverts,
        expected_washed_out_culverts,
        p90_washed_out_culverts,
        post_washout_4min_fire_cov_pct: cov_summary.parcels_within_4min_pct,
        coupled_rdii_sewer_inflow_mm_hr,
        basin_rows,
        culvert_rows,
        id_threshold_curve,
        washed_out_segment_ids,
        overtopped_slow_segment_ids,
        parcel_runoff_depth_mm,
        parcel_debris_hazard_pct,
    }
}

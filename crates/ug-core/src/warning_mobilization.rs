//! Post-Earthquake Cellular Base-Station (`BTS`) Blackout Fragility, FEMA IPAWS Wireless Emergency
//! Alerts (`WEA`) vs. Outdoor Acoustic Warning Sirens (`NIST TN 1950`) & Hi-Lo / Door-to-Door
//! Canvassing Warning Diffusion, Lindell-Perry (`PADM` / `EMBLEM2`) Mobilization S-Curves, and
//! Blackout-Compensated Delayed Departure Staging
//!
//! Grounded in:
//! - Tang, A. et al. (2014 / 2020), *Telecommunication Systems' Performance: Christchurch
//!   Earthquakes*, EERI *Earthquake Spectra* 30(1):231–252
//!   (`https://pubs.geoscienceworld.org/eeri/earthquake-spectra/article/30/1/231/587075/Telecommunication-Systems-Performance-Christchurch`)
//!   & Kaikoura Telecom Cabinet Battery Depletion (`https://bulletin.nzsee.org.nz/article/view/84/70`).
//! - FCC Report & Order 15-98 (`8-hour` commercial backup battery rule,
//!   `https://docs.fcc.gov/public/attachments/FCC-15-98A1.pdf`) & California PUC Decision 20-07-011 /
//!   21-02-029 (`72-hour` wireless backup generator & resiliency mandate,
//!   `https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M344/K021/344021480.pdf`).
//! - NIST Technical Note 1950, *Outdoor Siren Systems: A Review of Technology, Usage, and Public
//!   Response During Emergencies* (`https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1950.pdf`)
//!   & US NRC / FEMA Acoustic Attenuation Criteria (`https://www.nrc.gov/docs/ml1025/ML102510343.pdf`).
//! - Lindell, M. K. & Perry, R. W. (2012), *The Protective Action Decision Model (`PADM`): Theoretical
//!   Modifications and Additional Evidence*, *Risk Analysis* 32(4):616–632
//!   (`https://www.researchgate.net/publication/271767724_The_protective_action_decision_model_Theoretical_modifications_and_additional_evidence`),
//!   Lindell (2008) `EMBLEM2` (`https://www.researchgate.net/publication/222422955_EMBLEM2_An_empirically_based_large_scale_evacuation_time_estimate_model`),
//!   and *Multiscale Wildfire-Evacuation Modeling: Differential Access to Safe Egress*
//!   (`https://pmc.ncbi.nlm.nih.gov/articles/PMC13507379/`).

use crate::emergency::EmergencyFacility;
use crate::evacuation_contraflow::smoke_visibility_speed_and_capacity_factors;
use crate::ffe_conflagration::{
    hamada_hazus_spread_rates, scawthorn_hazus_ignition_rate_per_mmsqft,
};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi,
};
use crate::transit_evacuation::TransitEvacuationConfig;
use crate::utilities::{MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const WARNING_TOWER_CSV_HEADER: &str = "tower_id,name,mount_archetype,coverage_radius_m,structural_intact_pct,fiber_backhaul_ok_pct,battery_energized_pct,wea_cell_online_pct,served_parcels";
pub const WARNING_CURVE_CSV_HEADER: &str = "minute,wea_only_mobilized_pct,wea_plus_siren_mobilized_pct,multimodal_mobilized_pct,staged_network_departed_pct,safe_corridor_cleared_pct";

const SQFT_PER_M2: f64 = 10.763_910_4;

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64_open01(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        u.clamp(1e-12, 1.0 - 1e-12)
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[inline]
fn sample_poisson_icdf(lambda: f64, u01: f64) -> usize {
    if lambda <= 1e-12 {
        return 0;
    }
    let lam = lambda.min(45.0);
    let mut p = (-lam).exp();
    let mut cdf = p;
    if u01 <= cdf {
        return 0;
    }
    for k in 1..=96 {
        p *= lam / (k as f64);
        cdf += p;
        if u01 <= cdf {
            return k;
        }
    }
    96
}

fn cholesky_lower(corr: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = corr[i * n + j];
            if i == j {
                s += 1e-9;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = s.max(1e-12).sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j].max(1e-12);
            }
        }
    }
    l
}

/// Cellular base-station (`BTS`) backup power & seismic/backhaul hardening standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelecomBackupPowerStandard {
    /// Aged commercial VRLA battery cabinets (~2.5 hr mean endurance under 10x post-quake RAN surge, single-path fiber).
    LegacyCommercial2to4HrBattery,
    /// FCC Report & Order 15-98 nominal `8-hour` backup battery cabinet (~6.5 hr effective under surge).
    FccEightHourBatteryStandard,
    /// CPUC Decision 20-07-011 `72-hour` diesel/fuel-cell backup + TIA-222-H seismic bracing + LEO satellite backhaul!
    Cpuc72HrHardenedWithLeoBackhaul,
}

impl Default for TelecomBackupPowerStandard {
    fn default() -> Self {
        Self::LegacyCommercial2to4HrBattery
    }
}

impl TelecomBackupPowerStandard {
    pub fn label(self) -> &'static str {
        match self {
            Self::LegacyCommercial2to4HrBattery => {
                "Legacy Commercial VRLA (~2.5–4h Battery, Single Fiber)"
            }
            Self::FccEightHourBatteryStandard => {
                "FCC 15-98 Standard (8-Hour Battery Cabinet)"
            }
            Self::Cpuc72HrHardenedWithLeoBackhaul => {
                "CPUC 72-Hr Diesel/Fuel-Cell + TIA-222-H + LEO Satellite"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::LegacyCommercial2to4HrBattery => "Legacy-3h-VRLA",
            Self::FccEightHourBatteryStandard => "FCC-8h-Batt",
            Self::Cpuc72HrHardenedWithLeoBackhaul => "CPUC-72h+LEO",
        }
    }

    /// Returns `(median_battery_hours_under_surge, structural_pga_median_g_mult, fiber_redundancy_survival_bonus)`.
    pub fn resilience_params(self) -> (f64, f64, f64) {
        match self {
            Self::LegacyCommercial2to4HrBattery => (2.8, 1.0, 0.0),
            Self::FccEightHourBatteryStandard => (6.8, 1.15, 0.25),
            Self::Cpuc72HrHardenedWithLeoBackhaul => (72.0, 1.55, 0.88),
        }
    }
}

/// Emergency public warning & alert dissemination channel portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertChannelMix {
    /// Relies strictly on cellular IPAWS `WEA` + opt-in Reverse-911 SMS (fails in `BTS` blackout shadow zones!).
    WeaAndReverse911Only,
    /// Adds fixed solar-backed Outdoor Acoustic Warning Sirens (`124 dBC @ 30.5 m`, NIST TN 1950).
    WeaPlusOutdoorAcousticSirens,
    /// Full FEMA layered alert portfolio: IPAWS `WEA` + Outdoor Voice/Tone Sirens + Police/Fire Hi-Lo Cruisers & Door-to-Door Canvassing!
    MultiModalWeaSirenHiLoDoorToDoor,
}

impl Default for AlertChannelMix {
    fn default() -> Self {
        Self::MultiModalWeaSirenHiLoDoorToDoor
    }
}

impl AlertChannelMix {
    pub fn label(self) -> &'static str {
        match self {
            Self::WeaAndReverse911Only => "Cellular IPAWS WEA & Reverse-911 Only",
            Self::WeaPlusOutdoorAcousticSirens => "IPAWS WEA + Fixed Outdoor Acoustic Sirens (124 dBC)",
            Self::MultiModalWeaSirenHiLoDoorToDoor => {
                "Multi-Modal: WEA + Outdoor Sirens + Hi-Lo Cruisers & Door-to-Door"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::WeaAndReverse911Only => "WEA-Only",
            Self::WeaPlusOutdoorAcousticSirens => "WEA+Sirens",
            Self::MultiModalWeaSirenHiLoDoorToDoor => "WEA+Siren+HiLo+D2D",
        }
    }
}

/// Household departure-staging coordination under heterogeneous warning delays & cellular blackouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DepartureStagingPolicy {
    /// Unstaged immediate departure upon alert receipt: intact outer `WEA` zones clog arterials ahead of delayed inner blackout zones!
    UnstagedImmediateOnReceipt,
    /// Fixed geo-zone `WEA` offsets (`0 / 20 / 40 min`) assuming 100% cellular reach (fails when inner fire zones suffer cell blackout).
    StaticGeoZoneWeaStaging,
    /// Blackout-compensated adaptive staging: dispatches Hi-Lo/Door-to-Door units to `BTS`-dark inner blocks while holding outer `WEA` zones!
    BlackoutCompensatedAdaptiveStaging,
}

impl Default for DepartureStagingPolicy {
    fn default() -> Self {
        Self::BlackoutCompensatedAdaptiveStaging
    }
}

impl DepartureStagingPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::UnstagedImmediateOnReceipt => {
                "Unstaged Immediate Departure (Outer WEA Zones Clog Bottlenecks)"
            }
            Self::StaticGeoZoneWeaStaging => {
                "Static 0/20/40m Geo-Zone WEA Staging (Blackout-Blind)"
            }
            Self::BlackoutCompensatedAdaptiveStaging => {
                "Blackout-Compensated Adaptive Hold + Targeted Hi-Lo/D2D Priority"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::UnstagedImmediateOnReceipt => "Unstaged-Rush",
            Self::StaticGeoZoneWeaStaging => "Static-WEA-Zones",
            Self::BlackoutCompensatedAdaptiveStaging => "Adaptive-Blackout-Hold",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellSiteMountArchetype {
    /// Unanchored or lightly bolted urban rooftop macrocell (`median PGA = 0.56 g`).
    UrbanRooftopMacrocell,
    /// Tubular steel monopole tower (`median PGA = 0.68 g`, vulnerable to soft-soil foundation tilt).
    TubularSteelMonopole,
    /// Braced 3/4-legged steel lattice telecom tower (`median PGA = 0.88 g`).
    BracedLatticeTower,
}

impl CellSiteMountArchetype {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::UrbanRooftopMacrocell => "Rooftop-Macro",
            Self::TubularSteelMonopole => "Steel-Monopole",
            Self::BracedLatticeTower => "Lattice-Tower",
        }
    }

    pub fn base_median_pga_g(self) -> f64 {
        match self {
            Self::UrbanRooftopMacrocell => 0.56,
            Self::TubularSteelMonopole => 0.68,
            Self::BracedLatticeTower => 0.88,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningMobilizationConfig {
    /// Chained GTFS/ADA transit evacuation, clean-air shelters, contraflow, FFE & seismic MC config.
    pub transit_evac: TransitEvacuationConfig,
    pub telecom_backup: TelecomBackupPowerStandard,
    pub channel_mix: AlertChannelMix,
    pub staging_policy: DepartureStagingPolicy,
    /// Elapsed hours after the mainshock when wind-driven `FFE` evacuation order is issued (`0.25..=36.0 hr`, default `4.5 hr`).
    pub hours_post_mainshock: f64,
    /// True for a `02:00 AM` nocturnal earthquake/conflagration (sleeping households, phone DND, `62 dBC` indoor wake threshold).
    pub nocturnal_sleeping_event: bool,
    /// Number of solar-battery municipal Outdoor Acoustic Warning Sirens (`124 dBC @ 30.5 m`, `0..=12`, default `6`).
    pub outdoor_siren_sites: usize,
    /// Active law-enforcement / fire Hi-Lo siren + Door-to-Door canvass teams (`1..=24`, default `8`).
    pub canvass_patrol_units: usize,
}

impl Default for WarningMobilizationConfig {
    fn default() -> Self {
        Self {
            transit_evac: TransitEvacuationConfig::default(),
            telecom_backup: TelecomBackupPowerStandard::LegacyCommercial2to4HrBattery,
            channel_mix: AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor,
            staging_policy: DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging,
            hours_post_mainshock: 4.5,
            nocturnal_sleeping_event: true,
            outdoor_siren_sites: 6,
            canvass_patrol_units: 8,
        }
    }
}

/// Computes outdoor or indoor sound pressure level (`dBC`) from a municipal rotating/omnidirectional
/// warning siren rated `source_dbc_at_30m` (typically `124 dBC` at `30.48 m` / `100 ft`) per
/// NIST TN 1950 & NRC/FEMA `ML102510343` spherical spreading + urban clutter/air absorption
/// (`-0.0085 dB/m`) and closed-window building envelope insertion loss (`-22 dB`).
#[inline]
pub fn compute_siren_spl_dbc(
    dist_m: f64,
    source_dbc_at_30m: f64,
    indoor_closed_window: bool,
) -> f64 {
    let d = dist_m.max(10.0);
    let geometric_loss_db = 20.0 * (d / 30.48).log10();
    let clutter_air_loss_db = 0.0085 * (d - 30.48).max(0.0);
    let envelope_loss_db = if indoor_closed_window { 22.0 } else { 0.0 };
    (source_dbc_at_30m - geometric_loss_db - clutter_air_loss_db - envelope_loss_db).max(25.0)
}

/// Evaluates the Lindell & Perry (2012) `PADM` / `EMBLEM2` cumulative household mobilization fraction
/// `M(t) in [0, reach_asymptote]` at `minute >= 0` via discrete Riemann convolution `(W * p)(t)` of:
/// - Weibull warning-diffusion receipt CDF `W(u) = reach * (1 - exp(-(u / warn_scale)^warn_shape))`
/// - Weibull household preparation PDF `p(tau)` with scale `prep_scale_min` and shape `k_p = 2.1`.
pub fn evaluate_padm_mobilization_cdf(
    minute: f64,
    weibull_warn_scale_min: f64,
    weibull_warn_shape: f64,
    prep_scale_min: f64,
    reach_asymptote: f64,
) -> f64 {
    if minute <= 0.0 || reach_asymptote <= 0.0 {
        return 0.0;
    }
    let asym = reach_asymptote.clamp(0.0, 1.0);
    let a_w = weibull_warn_scale_min.max(0.6);
    let k_w = weibull_warn_shape.clamp(0.7, 5.0);
    let a_p = prep_scale_min.max(1.0);
    let k_p = 2.1f64;

    let steps = 48usize;
    let dt = minute / (steps as f64);
    let mut conv = 0.0f64;
    for i in 0..steps {
        let u = (i as f64 + 0.5) * dt;
        let rem = (minute - u).max(0.0);
        // Warning arrival density `w(u) = dW/du`
        let u_norm = u / a_w;
        let w_pdf = asym * (k_w / a_w) * u_norm.powf(k_w - 1.0) * (-u_norm.powf(k_w)).exp();
        // Preparation completion CDF `P(t - u) = 1 - exp(-((t - u)/a_p)^k_p)`
        let p_cdf = 1.0 - (-(rem / a_p).powf(k_p)).exp();
        conv += w_pdf * p_cdf * dt;
    }
    conv.clamp(0.0, asym)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellTowerTelemetryRow {
    pub tower_id: usize,
    pub name: String,
    pub mount_archetype: CellSiteMountArchetype,
    pub world_pos: Vec2,
    pub coverage_radius_m: f64,
    pub structural_intact_pct: f64,
    pub fiber_backhaul_ok_pct: f64,
    pub battery_energized_pct: f64,
    pub wea_cell_online_pct: f64,
    pub served_parcels: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirenSiteRow {
    pub siren_id: usize,
    pub name: String,
    pub world_pos: Vec2,
    pub radius_70dbc_outdoor_m: f64,
    pub radius_62dbc_indoor_wake_m: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MobilizationCurvePoint {
    pub minute: f64,
    pub wea_only_mobilized_pct: f64,
    pub wea_plus_siren_mobilized_pct: f64,
    pub multimodal_mobilized_pct: f64,
    pub staged_network_departed_pct: f64,
    pub safe_corridor_cleared_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningMobilizationReport {
    pub config: WarningMobilizationConfig,
    pub cell_towers_count: usize,
    pub outdoor_sirens_count: usize,
    pub mean_cell_towers_blackout_pct: f64,
    pub population_in_wea_blackout_pct: f64,
    pub mean_warning_receipt_min: f64,
    pub median_mobilization_departure_min: f64,
    pub p90_mobilization_departure_min: f64,
    pub inner_fire_zone_clearance_min: f64,
    pub network_peak_vc_ratio: f64,
    pub late_departure_fire_trapped_pax: f64,
    pub p90_fire_trapped_pax: f64,
    pub safe_evacuated_within_window_pct: f64,
    pub tower_rows: Vec<CellTowerTelemetryRow>,
    pub siren_rows: Vec<SirenSiteRow>,
    pub mobilization_curves: Vec<MobilizationCurvePoint>,
    pub canvass_sweep_paths: Vec<(Vec2, Vec2)>,
    pub parcel_mobilized_60m_pct: Vec<f64>,
    pub parcel_departure_delay_min: Vec<f64>,
    pub parcel_recommended_stage: Vec<u8>,
    pub solve_time_ms: f64,
}

impl WarningMobilizationReport {
    pub fn to_tower_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.tower_rows.len() * 128);
        out.push_str(WARNING_TOWER_CSV_HEADER);
        out.push('\n');
        for t in &self.tower_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{:.1},{:.2},{:.2},{:.2},{:.2},{}\n",
                t.tower_id,
                t.name.replace('"', "'"),
                t.mount_archetype.short_label(),
                t.coverage_radius_m,
                t.structural_intact_pct,
                t.fiber_backhaul_ok_pct,
                t.battery_energized_pct,
                t.wea_cell_online_pct,
                t.served_parcels,
            ));
        }
        out
    }

    pub fn to_curve_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.mobilization_curves.len() * 96);
        out.push_str(WARNING_CURVE_CSV_HEADER);
        out.push('\n');
        for c in &self.mobilization_curves {
            out.push_str(&format!(
                "{:.1},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                c.minute,
                c.wea_only_mobilized_pct,
                c.wea_plus_siren_mobilized_pct,
                c.multimodal_mobilized_pct,
                c.staged_network_departed_pct,
                c.safe_corridor_cleared_pct,
            ));
        }
        out
    }

    pub fn export_tower_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_tower_csv_string())?;
        Ok(self.tower_rows.len())
    }
}

#[derive(Clone)]
struct SynthCellTower {
    tower_id: usize,
    name: String,
    archetype: CellSiteMountArchetype,
    world_pos: Vec2,
    coverage_radius_m: f64,
    bridge_hung_backhaul: bool,
}

fn synthesize_cellular_bts_grid(net: &RoadNetwork) -> Vec<SynthCellTower> {
    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(150.0);
    let sy = (b.max_y - b.min_y).max(150.0);
    let cell_r = (0.34 * sx.hypot(sy)).clamp(280.0, 1_150.0);

    let specs = [
        ("Downtown CBD Rooftop eNodeB", CellSiteMountArchetype::UrbanRooftopMacrocell, 0.48, 0.50, true),
        ("Riverfront Monopole BTS-2", CellSiteMountArchetype::TubularSteelMonopole, 0.62, 0.54, true),
        ("University Campus Rooftop gNodeB", CellSiteMountArchetype::UrbanRooftopMacrocell, 0.24, 0.48, false),
        ("North Kings Arterial Monopole", CellSiteMountArchetype::TubularSteelMonopole, 0.44, 0.18, false),
        ("South Commercial Monopole BTS-5", CellSiteMountArchetype::TubularSteelMonopole, 0.46, 0.84, true),
        ("West Hills Braced Lattice BTS-6", CellSiteMountArchetype::BracedLatticeTower, 0.16, 0.22, false),
        ("Eastgate Industrial Lattice BTS-7", CellSiteMountArchetype::BracedLatticeTower, 0.84, 0.30, true),
        ("Southeast Collector Monopole BTS-8", CellSiteMountArchetype::TubularSteelMonopole, 0.80, 0.78, false),
    ];

    specs
        .into_iter()
        .enumerate()
        .map(|(i, (nm, arch, fx, fy, br_bh))| SynthCellTower {
            tower_id: i,
            name: nm.to_string(),
            archetype: arch,
            world_pos: Vec2 {
                x: b.min_x + fx * sx,
                y: b.min_y + fy * sy,
            },
            coverage_radius_m: cell_r,
            bridge_hung_backhaul: br_bh,
        })
        .collect()
}

fn synthesize_outdoor_sirens(net: &RoadNetwork, count: usize) -> Vec<SirenSiteRow> {
    if count == 0 {
        return Vec::new();
    }
    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(150.0);
    let sy = (b.max_y - b.min_y).max(150.0);
    let anchors = [
        ("Civic Plaza Omni Siren #1", 0.50, 0.50),
        ("Northside Park Siren #2", 0.42, 0.20),
        ("West Campus Siren #3", 0.22, 0.48),
        ("South Riverfront Siren #4", 0.52, 0.80),
        ("Eastgate Fire-District Siren #5", 0.80, 0.34),
        ("Southeast Greenway Siren #6", 0.78, 0.74),
        ("Northwest Ridge Siren #7", 0.20, 0.20),
        ("Southwest Residential Siren #8", 0.24, 0.80),
        ("Midtown Couplet Siren #9", 0.58, 0.36),
        ("Industrial Yard Siren #10", 0.86, 0.56),
        ("Central High Siren #11", 0.36, 0.62),
        ("North River Crossing Siren #12", 0.64, 0.16),
    ];

    // Precompute exact 70 dBC outdoor & 62 dBC indoor closed-window radii by bisection on `compute_siren_spl_dbc`
    let solve_r = |target_db: f64, indoor: bool| -> f64 {
        let mut lo = 15.0f64;
        let mut hi = 3_500.0f64;
        for _ in 0..32 {
            let mid = 0.5 * (lo + hi);
            if compute_siren_spl_dbc(mid, 124.0, indoor) >= target_db {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let r_70_out = solve_r(70.0, false);
    let r_62_in = solve_r(62.0, true);

    anchors
        .into_iter()
        .take(count.min(12))
        .enumerate()
        .map(|(i, (nm, fx, fy))| SirenSiteRow {
            siren_id: i,
            name: nm.to_string(),
            world_pos: Vec2 {
                x: b.min_x + fx * sx,
                y: b.min_y + fy * sy,
            },
            radius_70dbc_outdoor_m: r_70_out,
            radius_62dbc_indoor_wake_m: r_62_in,
        })
        .collect()
}

#[derive(Clone, Copy)]
struct HeapState {
    cost_s: f64,
    seg: usize,
}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.cost_s.to_bits() == other.cost_s.to_bits() && self.seg == other.seg
    }
}
impl Eq for HeapState {}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost_s
            .partial_cmp(&self.cost_s)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.seg.cmp(&other.seg))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn build_seg_adjacency(net: &RoadNetwork) -> Vec<Vec<(usize, f64)>> {
    let n = net.segments.len();
    let mut adj = vec![Vec::new(); n];
    for c in &net.connectors {
        if c.from_lane_id >= net.lanes.len() || c.to_lane_id >= net.lanes.len() {
            continue;
        }
        let u = net.lanes[c.from_lane_id].segment_id;
        let v = net.lanes[c.to_lane_id].segment_id;
        if u < n && v < n && u != v {
            let turn_s = match net.junctions[c.junction_id].control_mode {
                ControlMode::Roundabout => 2.5,
                ControlMode::MaxPressureAdaptive => 4.2,
                ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::PermissiveYield => match c.turn_type {
                    TurnType::Through => 4.5,
                    TurnType::Right => 5.5,
                    TurnType::Left => 8.0,
                },
            };
            if adj[u].iter().all(|&(nb, _)| nb != v) {
                adj[u].push((v, turn_s));
            }
        }
    }
    adj
}

fn dijkstra_multi_source(
    origins: &[usize],
    seg_trav_s: &[f64],
    seg_blocked: &[bool],
    adj: &[Vec<(usize, f64)>],
) -> Vec<f64> {
    let n = seg_trav_s.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut pq = BinaryHeap::new();
    for &s in origins {
        if s < n && !seg_blocked[s] && 0.5 * seg_trav_s[s] < dist[s] {
            dist[s] = 0.5 * seg_trav_s[s];
            pq.push(HeapState {
                cost_s: dist[s],
                seg: s,
            });
        }
    }
    while let Some(HeapState { cost_s, seg }) = pq.pop() {
        if cost_s > dist[seg] + 1e-9 {
            continue;
        }
        for &(nxt, turn_s) in &adj[seg] {
            if seg_blocked[nxt] {
                continue;
            }
            let cand = cost_s + turn_s + seg_trav_s[nxt];
            if cand + 1e-9 < dist[nxt] {
                dist[nxt] = cand;
                pq.push(HeapState {
                    cost_s: cand,
                    seg: nxt,
                });
            }
        }
    }
    dist
}

/// Runs `Rayon`-parallelized Monte Carlo simulation of Post-Earthquake Cellular Tower (`BTS`) Blackout
/// Fragility, `WEA` vs. Outdoor Siren & Hi-Lo / Door-to-Door Warning Mobilization Curves, and
/// Blackout-Compensated Delayed Departure Staging under Common Random Numbers (`CRN`).
pub fn run_warning_mobilization_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &WarningMobilizationConfig,
) -> WarningMobilizationReport {
    let t0 = Instant::now();
    let n_seg = net.segments.len();
    let n_parcels = parcels.len();
    let seis_cfg = &cfg.transit_evac.shelters.evac.medevac.surge.ffe.seismic;
    let ffe_cfg = &cfg.transit_evac.shelters.evac.medevac.surge.ffe;
    let trials = seis_cfg.monte_carlo_trials.clamp(16, 5_000);
    let win_min = cfg.transit_evac.clearance_window_min.clamp(45.0, 240.0);

    let towers = synthesize_cellular_bts_grid(net);
    let n_towers = towers.len();
    let sirens = synthesize_outdoor_sirens(net, cfg.outdoor_siren_sites);

    // Map each parcel to primary & secondary backup `BTS` towers + nearest open road segment + max siren SPL
    let mut parcel_primary_bts = vec![0usize; n_parcels];
    let mut parcel_secondary_bts: Vec<Option<usize>> = vec![None; n_parcels];
    let mut parcel_served_cnt = vec![0usize; n_towers];
    let mut parcel_siren_indoor_db = vec![25.0f64; n_parcels];
    let mut parcel_siren_outdoor_db = vec![25.0f64; n_parcels];
    let mut parcel_seg = vec![0usize; n_parcels];
    let mut parcel_pop = vec![20.0f64; n_parcels];

    for (pi, p) in parcels.iter().enumerate() {
        let area_m2 = p.area_ha * 10_000.0;
        let est_pop = p.residents() * 0.42 + p.jobs() * 0.55;
        let pop = (12.0 + est_pop * 0.35 + (area_m2 / 550.0).min(24.0)).clamp(8.0, 68.0);
        parcel_pop[pi] = pop;

        let mut d_list: Vec<(usize, f64)> = towers
            .iter()
            .map(|t| (t.tower_id, p.center.dist(t.world_pos)))
            .collect();
        d_list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        if let Some(&(t0_id, _)) = d_list.first() {
            parcel_primary_bts[pi] = t0_id;
            parcel_served_cnt[t0_id] += 1;
        }
        if let Some(&(t1_id, d1)) = d_list.get(1) {
            if d1 <= towers[t1_id].coverage_radius_m * 1.18 {
                parcel_secondary_bts[pi] = Some(t1_id);
            }
        }

        for sr in &sirens {
            let d = p.center.dist(sr.world_pos);
            parcel_siren_outdoor_db[pi] =
                parcel_siren_outdoor_db[pi].max(compute_siren_spl_dbc(d, 124.0, false));
            parcel_siren_indoor_db[pi] =
                parcel_siren_indoor_db[pi].max(compute_siren_spl_dbc(d, 124.0, true));
        }

        let sid = net
            .segments
            .iter()
            .filter(|s| !s.closed && !s.base_pts.is_empty())
            .min_by(|a, b| {
                let da = a.base_pts[a.base_pts.len() / 2].dist(p.center);
                let db = b.base_pts[b.base_pts.len() / 2].dist(p.center);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .map(|s| s.id)
            .unwrap_or(0);
        parcel_seg[pi] = sid;
    }

    let adj = build_seg_adjacency(net);
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, seis_cfg);

    // Police / Fire origin segments for Hi-Lo & Door-to-Door patrol dispatch
    let mut patrol_origins: Vec<usize> = facilities
        .iter()
        .map(|f| f.segment_id)
        .filter(|&s| s < n_seg)
        .collect();
    if patrol_origins.is_empty() && n_seg > 0 {
        patrol_origins.push(0);
        patrol_origins.push(n_seg / 2);
    }

    // Spatial ground-motion correlation across `n_towers`
    let mut corr = vec![0.0; n_towers * n_towers];
    for i in 0..n_towers {
        for j in 0..n_towers {
            let d_km = towers[i].world_pos.dist(towers[j].world_pos) / 1000.0;
            corr[i * n_towers + j] = (-d_km / 4.2).exp();
        }
    }
    let chol = cholesky_lower(&corr, n_towers);

    let base_sa10_g = crate::seismic_mc::median_sa10_gmpe_g(
        seis_cfg.moment_magnitude_mw,
        seis_cfg.rupture_standoff_km,
        seis_cfg.vs30_m_per_s,
    );

    let (med_batt_hr, pga_hard_mult, fiber_bonus) = cfg.telecom_backup.resilience_params();
    let hrs_after = cfg.hours_post_mainshock.clamp(0.1, 72.0);
    let n_patrol = cfg.canvass_patrol_units.clamp(1, 32) as f64;
    let curve_pts = 13usize; // every 10 min across `0..=120m` (scaled to `win_min`)
    let wind_rad = ffe_cfg.wind_direction_deg.to_radians();
    let wind_vec = Vec2 {
        x: wind_rad.cos(),
        y: wind_rad.sin(),
    };

    #[derive(Clone)]
    struct TrialOut {
        towers_dark_pct: f64,
        pop_wea_dark_pct: f64,
        mean_warn_min: f64,
        p50_dep_min: f64,
        p90_dep_min: f64,
        inner_clr_min: f64,
        peak_vc: f64,
        trapped_pax: f64,
        safe_evac_pct: f64,
        tw_struct_ok: Vec<f64>,
        tw_fiber_ok: Vec<f64>,
        tw_batt_ok: Vec<f64>,
        tw_online: Vec<f64>,
        c_wea: Vec<f64>,
        c_siren: Vec<f64>,
        c_multi: Vec<f64>,
        c_dep: Vec<f64>,
        c_clr: Vec<f64>,
        p_mob60: Vec<f64>,
        p_delay: Vec<f64>,
        p_stage: Vec<f64>,
    }

    let base_seed = seis_cfg.random_seed ^ 0x5745_4153_3339_0011;
    let trial_outs: Vec<TrialOut> = (0..trials)
        .into_par_iter()
        .map(|tr_idx| {
            let mut rng = SplitMix64::new(
                base_seed.wrapping_add((tr_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let z_inter = rng.next_standard_normal();
            let mut z_ind = vec![0.0; n_towers];
            for z in &mut z_ind {
                *z = rng.next_standard_normal();
            }
            let mut z_corr = vec![0.0; n_towers];
            for i in 0..n_towers {
                let mut s = 0.0;
                for j in 0..=i {
                    s += chol[i * n_towers + j] * z_ind[j];
                }
                z_corr[i] = s;
            }

            let sa10_city =
                (base_sa10_g * (0.34 * z_inter + 0.22 * z_corr[0]).exp()).clamp(0.03, 2.6);
            let pga_city =
                crate::seismic_mc::median_pga_g_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);
            let pgv_cm_s = median_pgv_cm_s_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);

            // Commercial AC feeder blackout probability per substation zone
            let p_grid_out = normal_cdf_phi((pga_city / 0.30).ln() / 0.50);

            // Bridge seismic closures (also severs bridge-hung fiber conduits unless LEO/ring redundant!)
            let mut seg_closed = vec![false; n_seg];
            let mut any_bridge_closed = false;
            for br in &bridges {
                if rng.next_f64_open01() < br.effective_archetype.exceedance_probs(sa10_city)[2] {
                    any_bridge_closed = true;
                    for &sid in &br.segment_ids {
                        if sid < n_seg {
                            seg_closed[sid] = true;
                        }
                    }
                }
            }

            // Evaluate each Cellular BTS site: Structural && Backhaul && (GridPower || BatteryEndurance > hrs_after)
            let pipe_rr = ala_hazus_pipe_repair_rate_per_km(
                pgv_cm_s,
                seis_cfg.pipe_fragility_law,
                seis_cfg.baseline_water_pipe_material,
                250.0,
                false,
            );
            let mut tw_struct = vec![0.0; n_towers];
            let mut tw_fiber = vec![0.0; n_towers];
            let mut tw_batt = vec![0.0; n_towers];
            let mut tw_alive = vec![false; n_towers];

            for (ti, tw) in towers.iter().enumerate() {
                let pga_tw = (pga_city * (0.20 * z_corr[ti]).exp()).clamp(0.03, 2.8);
                let med_struct_g = tw.archetype.base_median_pga_g() * pga_hard_mult;
                let s_ok = rng.next_f64_open01()
                    >= normal_cdf_phi((pga_tw / med_struct_g.max(0.18)).ln() / 0.52);

                let fiber_rr = pipe_rr * 0.36 * (1.0 - fiber_bonus);
                let bridge_cut =
                    tw.bridge_hung_backhaul && any_bridge_closed && (rng.next_f64_open01() > fiber_bonus);
                let f_ok = !bridge_cut && (rng.next_f64_open01() >= (1.0 - (-fiber_rr * 1.6).exp()));

                let ac_on = rng.next_f64_open01() >= p_grid_out;
                // Sample lognormal battery endurance (`dispersion = 0.38` for VRLA cell aging & 10x call-surge load)
                let batt_endurance_hr =
                    med_batt_hr * (0.38 * rng.next_standard_normal()).exp().clamp(0.35, 2.4);
                let pwr_ok = ac_on || (batt_endurance_hr >= hrs_after);

                tw_struct[ti] = if s_ok { 1.0 } else { 0.0 };
                tw_fiber[ti] = if f_ok { 1.0 } else { 0.0 };
                tw_batt[ti] = if pwr_ok { 1.0 } else { 0.0 };
                tw_alive[ti] = s_ok && f_ok && pwr_ok;
            }

            // Sample Scawthorn/Hazus FFE conflagration fronts & parcel fire arrival deadlines `t_fire_arr_min[pi]`
            let total_sqft: f64 = parcels
                .iter()
                .map(|p| p.area_ha * 10_000.0 * SQFT_PER_M2 * 0.55)
                .sum();
            let ign_rate = scawthorn_hazus_ignition_rate_per_mmsqft(
                pga_city,
                ZoneCode::MU4,
                ffe_cfg.ignition_model,
                ffe_cfg.seismic_gas_shutoff_pct,
            );
            let n_ign = sample_poisson_icdf(
                (ign_rate * (total_sqft / 1_000_000.0)).max(0.22),
                rng.next_f64_open01(),
            )
            .max(1);
            let water_ok = rng.next_f64_open01() > (1.0 - (-pipe_rr * 1.8).exp());
            let hydrant_psi = if water_ok {
                54.0
            } else {
                (14.0 * PSI_PER_METER_H2O * 0.35).min(MIN_FIRE_FLOW_RESIDUAL_PSI - 4.0)
            };
            let rates = hamada_hazus_spread_rates(ZoneCode::R3, ffe_cfg.wind_speed_mps);
            let spread_m_min =
                (rates.v_down_m_min * if hydrant_psi >= 20.0 { 0.62 } else { 1.35 }).clamp(2.2, 22.0);

            let mut fire_centers = Vec::with_capacity(n_ign);
            for _ in 0..n_ign {
                if !parcels.is_empty() {
                    fire_centers.push(parcels[(rng.next_u64() as usize) % parcels.len()].center);
                }
            }

            // Compute Patrol Cruiser travel time to every segment for Hi-Lo & Door-to-Door canvassing
            let mut seg_trav_s = vec![10.0; n_seg];
            for seg in &net.segments {
                if seg.closed {
                    seg_closed[seg.id] = true;
                }
                let mid = if seg.base_pts.is_empty() {
                    Vec2 { x: 0.0, y: 0.0 }
                } else {
                    seg.base_pts[seg.base_pts.len() / 2]
                };
                let mut pm25 = 12.0;
                for &fc in &fire_centers {
                    let dx = mid.x - fc.x;
                    let dy = mid.y - fc.y;
                    let along = (dx * wind_vec.x + dy * wind_vec.y).max(-30.0);
                    let cross = (-dx * wind_vec.y + dy * wind_vec.x).abs();
                    if along >= -20.0 {
                        pm25 += 310.0
                            * (-(cross * cross) / (2.0 * (95.0 + 0.22 * along.max(0.0)).powi(2)))
                                .exp()
                            * (-along.max(0.0) / 1450.0).exp();
                    }
                }
                let vis_m = (3912.0 / (0.0042 * pm25 + 0.12)).clamp(40.0, 16_000.0);
                let (v_fac, _) = smoke_visibility_speed_and_capacity_factors(vis_m / 1_609.344);
                seg_trav_s[seg.id] = (seg.length / (seg.speed_mps * v_fac).max(3.5)).max(1.2);
            }
            let patrol_dist_s =
                dijkstra_multi_source(&patrol_origins, &seg_trav_s, &seg_closed, &adj);

            // Per-parcel warning-diffusion parameters `(alpha_w, k_w, prep_scale, reach)` under WEA-only, WEA+Siren, and Multi-Modal!
            let night = cfg.nocturnal_sleeping_event;
            let mut p_wea_dark = vec![false; n_parcels];
            let mut p_fire_dist_m = vec![900.0f64; n_parcels];
            let mut p_fire_deadline_min = vec![win_min; n_parcels];

            for pi in 0..n_parcels {
                let t0_ok = tw_alive[parcel_primary_bts[pi]]
                    && parcels[pi].center.dist(towers[parcel_primary_bts[pi]].world_pos)
                        <= towers[parcel_primary_bts[pi]].coverage_radius_m * 1.15;
                let t1_ok = parcel_secondary_bts[pi].map_or(false, |t1| tw_alive[t1]);
                p_wea_dark[pi] = !(t0_ok || t1_ok);

                let d_fire = fire_centers
                    .iter()
                    .map(|fc| parcels[pi].center.dist(*fc))
                    .fold(1_400.0f64, f64::min);
                p_fire_dist_m[pi] = d_fire;
                // Inner fire-front parcels face flame/lethal smoke arrival within `28..95 min`!
                p_fire_deadline_min[pi] =
                    (24.0 + d_fire / spread_m_min.max(2.0)).clamp(26.0, win_min * 1.35);
            }

            // Rank parcels for targeted Hi-Lo / Door-to-Door patrol dispatch:
            // Under BlackoutCompensatedAdaptiveStaging, patrols rush FIRST to `WEA-dark + near-fire` blocks!
            let mut canvass_order: Vec<usize> = (0..n_parcels).collect();
            canvass_order.sort_by(|&a, &b| {
                let prio = |pi: usize| -> f64 {
                    let dark_bonus = if p_wea_dark[pi]
                        && cfg.staging_policy
                            == DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging
                    {
                        450.0
                    } else if p_wea_dark[pi] {
                        120.0
                    } else {
                        0.0
                    };
                    let ada_bonus = match parcels[pi].zone {
                        ZoneCode::R3 | ZoneCode::CBD => 65.0,
                        _ => 0.0,
                    };
                    dark_bonus + ada_bonus - p_fire_dist_m[pi]
                };
                prio(b).partial_cmp(&prio(a)).unwrap_or(Ordering::Equal)
            });

            let mut parcel_canvass_arr_min = vec![95.0f64; n_parcels];
            for (rank, &pi) in canvass_order.iter().enumerate() {
                let drive_s = patrol_dist_s
                    .get(parcel_seg[pi])
                    .copied()
                    .unwrap_or(480.0);
                let drive_min = if drive_s.is_finite() {
                    (drive_s / 60.0).clamp(1.5, 22.0)
                } else {
                    18.0
                };
                // Each patrol unit sweeps ~1 block every 2.1 min with Hi-Lo + Door-to-Door knocking
                let queue_sweep_min = ((rank as f64) / n_patrol) * 2.15;
                parcel_canvass_arr_min[pi] = drive_min + queue_sweep_min;
            }

            // Helper returning `(warn_scale_min, warn_shape, prep_scale_min, asymptote_reach)` for any `AlertChannelMix`
            let channel_params = |pi: usize, mix: AlertChannelMix| -> (f64, f64, f64, f64) {
                let wea_ok = !p_wea_dark[pi];
                let siren_audible = if night {
                    parcel_siren_indoor_db[pi] >= 62.0
                } else {
                    parcel_siren_outdoor_db[pi] >= 68.0
                };
                let siren_partial = parcel_siren_indoor_db[pi] >= 54.0;

                // 1. WEA + Reverse-911 alone (falls back to slow neighbor/smoke environmental cues when BTS is dark!)
                let (w_wea, k_wea, p_wea, r_wea) = if wea_ok {
                    if night {
                        (6.5, 1.45, 16.5, 0.78) // DND / sleeping smartphone penalty
                    } else {
                        (2.8, 1.85, 11.5, 0.92)
                    }
                } else {
                    // Functional communication isolation! Only visual smoke / neighbor knocking (`38-58 min` scale!)
                    if night {
                        (52.0, 1.25, 21.0, 0.54)
                    } else {
                        (34.0, 1.35, 16.0, 0.72)
                    }
                };

                if mix == AlertChannelMix::WeaAndReverse911Only {
                    return (w_wea, k_wea, p_wea, r_wea);
                }

                // 2. Add Outdoor Acoustic Warning Sirens (survive grid blackout via solar DC cabinet!)
                let (w_sir, k_sir, p_sir, r_sir) = if siren_audible {
                    let w_s = if night { 5.2 } else { 3.4 };
                    // Tone siren requires brief PADM Milling/Confirmation (+3m prep) if WEA text is dark
                    let p_s = if wea_ok {
                        p_wea * 0.90
                    } else if night {
                        17.5
                    } else {
                        13.5
                    };
                    (w_wea.min(w_s), k_wea.max(1.75), p_s, r_wea.max(0.90))
                } else if siren_partial {
                    (w_wea.min(16.0), k_wea.max(1.45), p_wea * 0.96, r_wea.max(0.80))
                } else {
                    (w_wea, k_wea, p_wea, r_wea)
                };

                if mix == AlertChannelMix::WeaPlusOutdoorAcousticSirens {
                    return (w_sir, k_sir, p_sir, r_sir);
                }

                // 3. Full Multi-Modal: WEA + Outdoor Sirens + Hi-Lo Patrol Cruisers & Door-to-Door Knocking!
                let t_canv = parcel_canvass_arr_min[pi];
                // Authoritative uniformed Hi-Lo + Door-to-Door cuts PADM milling/hesitation sharply (`prep = 8.5..11.5m`, `98.5%` compliance!)
                let w_eff = w_sir.min(t_canv + 1.8);
                let p_eff = p_sir.min(if night { 11.2 } else { 8.4 });
                let r_eff = r_sir.max(0.985);
                (w_eff, k_sir.max(1.95), p_eff, r_eff)
            };

            // Staging hold offset `stage_hold_min[pi]` & stage tier `1..=3`
            let mut stage_tier = vec![1u8; n_parcels];
            let mut stage_hold_min = vec![0.0f64; n_parcels];
            for pi in 0..n_parcels {
                let d_f = p_fire_dist_m[pi];
                let tier = if d_f <= 240.0 {
                    1u8
                } else if d_f <= 520.0 {
                    2u8
                } else {
                    3u8
                };
                stage_tier[pi] = tier;

                stage_hold_min[pi] = match cfg.staging_policy {
                    DepartureStagingPolicy::UnstagedImmediateOnReceipt => 0.0,
                    DepartureStagingPolicy::StaticGeoZoneWeaStaging => {
                        // Static WEA staging can only hold households who actually receive the WEA geo-zone message!
                        if !p_wea_dark[pi] {
                            match tier {
                                1 => 0.0,
                                2 => 18.0,
                                _ => 34.0,
                            }
                        } else {
                            0.0
                        }
                    }
                    DepartureStagingPolicy::BlackoutCompensatedAdaptiveStaging => {
                        // Holds intact outer WEA zones (`Tier 2/3`) longer when inner Tier 1 blocks suffer cellular blackout,
                        // preventing outer-zone cars from gridlocking arterial bottlenecks before delayed Tier 1 evacuees arrive!
                        let inner_blackout_frac = {
                            let mut d_cnt = 0usize;
                            let mut tot_in = 0usize;
                            for pj in 0..n_parcels {
                                if p_fire_dist_m[pj] <= 260.0 {
                                    tot_in += 1;
                                    if p_wea_dark[pj] {
                                        d_cnt += 1;
                                    }
                                }
                            }
                            if tot_in > 0 {
                                (d_cnt as f64) / (tot_in as f64)
                            } else {
                                0.0
                            }
                        };
                        match tier {
                            1 => 0.0,
                            2 => 14.0 + 14.0 * inner_blackout_frac,
                            _ => 28.0 + 22.0 * inner_blackout_frac,
                        }
                    }
                };
            }

            // Evaluate time-stepped network departure loading, BPR bottleneck queue V/C, and FFE fire-overrun trapping!
            let tot_pop: f64 = parcel_pop.iter().sum::<f64>().max(1.0);
            let mut c_wea = vec![0.0; curve_pts];
            let mut c_siren = vec![0.0; curve_pts];
            let mut c_multi = vec![0.0; curve_pts];
            let mut c_dep = vec![0.0; curve_pts];
            let mut c_clr = vec![0.0; curve_pts];

            // Precompute active channel parameters per parcel
            let active_params: Vec<(f64, f64, f64, f64)> = (0..n_parcels)
                .map(|pi| channel_params(pi, cfg.channel_mix))
                .collect();

            // Simulate 5-minute bins across `0..=win_min` to capture dynamic bottleneck V/C surge!
            let dt_bin = 5.0f64;
            let n_bins = ((win_min / dt_bin).ceil() as usize).max(8);
            let road_cap_pax_per_5min = ((n_seg.max(12) as f64) * 9.5).max(85.0);
            let mut queue_pax = 0.0f64;
            let mut peak_vc = 0.45f64;
            let mut prev_dep_parcel = vec![0.0f64; n_parcels];
            let mut parcel_median_clear_min = vec![win_min * 1.25; n_parcels];
            let mut parcel_cleared_by_deadline = vec![0.0f64; n_parcels];

            for b_i in 1..=n_bins {
                let t_m = (b_i as f64) * dt_bin;
                let mut new_depart_pax = 0.0f64;
                for pi in 0..n_parcels {
                    let (aw, kw, ap, r_max) = active_params[pi];
                    let eff_t = (t_m - stage_hold_min[pi]).max(0.0);
                    let frac = evaluate_padm_mobilization_cdf(eff_t, aw, kw, ap, r_max);
                    let d_pax = (frac - prev_dep_parcel[pi]).max(0.0) * parcel_pop[pi];
                    prev_dep_parcel[pi] = frac;
                    new_depart_pax += d_pax;
                }
                let demand_in_bin = new_depart_pax + 0.45 * queue_pax;
                let vc = (demand_in_bin / road_cap_pax_per_5min).clamp(0.25, 2.85);
                peak_vc = peak_vc.max(vc);
                let bpr_delay_min = 6.5 * (1.0 + 0.34 * vc.powi(3));
                let discharged = (queue_pax + new_depart_pax).min(road_cap_pax_per_5min * 1.18);
                queue_pax = (queue_pax + new_depart_pax - discharged).max(0.0);

                for pi in 0..n_parcels {
                    let (aw, kw, ap, r_max) = active_params[pi];
                    let eff_t = (t_m - stage_hold_min[pi]).max(0.0);
                    let dep_frac = evaluate_padm_mobilization_cdf(eff_t, aw, kw, ap, r_max);
                    if dep_frac >= 0.50 && parcel_median_clear_min[pi] > win_min * 1.20 {
                        parcel_median_clear_min[pi] = t_m + bpr_delay_min;
                    }
                    // Fraction departed early enough (`t_m + bpr_delay_min <= p_fire_deadline_min[pi]`) to beat FFE front!
                    if t_m + bpr_delay_min <= p_fire_deadline_min[pi] {
                        parcel_cleared_by_deadline[pi] =
                            parcel_cleared_by_deadline[pi].max(dep_frac);
                    }
                }
            }

            // Populate the 13 comparison curve points (`0 .. win_min`)
            for ci in 0..curve_pts {
                let tm = (ci as f64 / (curve_pts - 1) as f64) * win_min;
                let mut s_w = 0.0;
                let mut s_ws = 0.0;
                let mut s_mm = 0.0;
                let mut s_dp = 0.0;
                let mut s_cl = 0.0;
                for pi in 0..n_parcels {
                    let pop = parcel_pop[pi];
                    let (aw0, kw0, ap0, r0) =
                        channel_params(pi, AlertChannelMix::WeaAndReverse911Only);
                    let (aw1, kw1, ap1, r1) =
                        channel_params(pi, AlertChannelMix::WeaPlusOutdoorAcousticSirens);
                    let (aw2, kw2, ap2, r2) =
                        channel_params(pi, AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor);
                    s_w += pop * evaluate_padm_mobilization_cdf(tm, aw0, kw0, ap0, r0);
                    s_ws += pop * evaluate_padm_mobilization_cdf(tm, aw1, kw1, ap1, r1);
                    s_mm += pop * evaluate_padm_mobilization_cdf(tm, aw2, kw2, ap2, r2);

                    let (awa, kwa, apa, ra) = active_params[pi];
                    let dep_f = evaluate_padm_mobilization_cdf(
                        (tm - stage_hold_min[pi]).max(0.0),
                        awa,
                        kwa,
                        apa,
                        ra,
                    );
                    let clr_lag = 7.0 * (1.0 + 0.25 * peak_vc.powi(2));
                    let clr_f = evaluate_padm_mobilization_cdf(
                        (tm - stage_hold_min[pi] - clr_lag).max(0.0),
                        awa,
                        kwa,
                        apa,
                        ra,
                    );
                    s_dp += pop * dep_f;
                    s_cl += pop * clr_f;
                }
                c_wea[ci] = 100.0 * (s_w / tot_pop).clamp(0.0, 1.0);
                c_siren[ci] = 100.0 * (s_ws / tot_pop).clamp(0.0, 1.0);
                c_multi[ci] = 100.0 * (s_mm / tot_pop).clamp(0.0, 1.0);
                c_dep[ci] = 100.0 * (s_dp / tot_pop).clamp(0.0, 1.0);
                c_clr[ci] = 100.0 * (s_cl / tot_pop).clamp(0.0, 1.0);
            }

            let dark_towers = tw_alive.iter().filter(|&&ok| !ok).count();
            let towers_dark_pct = 100.0 * (dark_towers as f64) / (n_towers.max(1) as f64);
            let dark_pop: f64 = (0..n_parcels)
                .filter(|&pi| p_wea_dark[pi])
                .map(|pi| parcel_pop[pi])
                .sum();
            let pop_wea_dark_pct = 100.0 * (dark_pop / tot_pop).clamp(0.0, 1.0);

            let mean_warn_min: f64 = (0..n_parcels)
                .map(|pi| active_params[pi].0 * 0.89 * parcel_pop[pi])
                .sum::<f64>()
                / tot_pop;

            // Interpolate 50% and 90% network departure times from `c_dep`
            let interp_pct = |target: f64| -> f64 {
                for ci in 1..curve_pts {
                    if c_dep[ci] >= target {
                        let c0 = c_dep[ci - 1];
                        let c1 = c_dep[ci];
                        let t0_m = ((ci - 1) as f64 / (curve_pts - 1) as f64) * win_min;
                        let t1_m = (ci as f64 / (curve_pts - 1) as f64) * win_min;
                        let u = ((target - c0) / (c1 - c0).max(1e-6)).clamp(0.0, 1.0);
                        return t0_m + u * (t1_m - t0_m);
                    }
                }
                win_min * 1.15
            };
            let p50_dep_min = interp_pct(50.0);
            let p90_dep_min = interp_pct(85.0);

            let mut inner_sum = 0.0;
            let mut inner_w = 0.0;
            let mut trapped_pax = 0.0;
            let mut p_mob60 = vec![0.0; n_parcels];
            let mut p_delay = vec![0.0; n_parcels];
            let mut p_stage_f = vec![1.0; n_parcels];

            for pi in 0..n_parcels {
                let (awa, kwa, apa, ra) = active_params[pi];
                p_mob60[pi] = 100.0
                    * evaluate_padm_mobilization_cdf(
                        (60.0 - stage_hold_min[pi]).max(0.0),
                        awa,
                        kwa,
                        apa,
                        ra,
                    );
                p_delay[pi] = awa * 0.89 + apa * 0.88 + stage_hold_min[pi];
                p_stage_f[pi] = stage_tier[pi] as f64;

                if stage_tier[pi] == 1 {
                    inner_sum += parcel_median_clear_min[pi] * parcel_pop[pi];
                    inner_w += parcel_pop[pi];
                }
                // Residents in fire-threatened blocks (`deadline <= win_min`) who fail to mobilize & clear before `p_fire_deadline_min`
                if p_fire_deadline_min[pi] <= win_min * 1.05 {
                    let safe_f = parcel_cleared_by_deadline[pi].clamp(0.0, 1.0);
                    trapped_pax += parcel_pop[pi] * (1.0 - safe_f);
                }
            }
            let inner_clr_min = if inner_w > 0.0 {
                inner_sum / inner_w
            } else {
                p50_dep_min + 10.0
            };
            let safe_evac_pct = c_clr[curve_pts - 1].clamp(0.0, 100.0);

            TrialOut {
                towers_dark_pct,
                pop_wea_dark_pct,
                mean_warn_min,
                p50_dep_min,
                p90_dep_min,
                inner_clr_min,
                peak_vc,
                trapped_pax,
                safe_evac_pct,
                tw_struct_ok: tw_struct,
                tw_fiber_ok: tw_fiber,
                tw_batt_ok: tw_batt,
                tw_online: tw_alive.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
                c_wea,
                c_siren,
                c_multi,
                c_dep,
                c_clr,
                p_mob60,
                p_delay,
                p_stage: p_stage_f,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let mut trap_sorted: Vec<f64> = trial_outs.iter().map(|o| o.trapped_pax).collect();
    trap_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p90_trap = trap_sorted[(((trials as f64) * 0.90).floor() as usize).min(trials - 1)];

    let mut tower_rows = Vec::with_capacity(n_towers);
    for (ti, tw) in towers.iter().enumerate() {
        tower_rows.push(CellTowerTelemetryRow {
            tower_id: ti,
            name: tw.name.clone(),
            mount_archetype: tw.archetype,
            world_pos: tw.world_pos,
            coverage_radius_m: tw.coverage_radius_m,
            structural_intact_pct: 100.0
                * trial_outs.iter().map(|o| o.tw_struct_ok[ti]).sum::<f64>()
                * inv_t,
            fiber_backhaul_ok_pct: 100.0
                * trial_outs.iter().map(|o| o.tw_fiber_ok[ti]).sum::<f64>()
                * inv_t,
            battery_energized_pct: 100.0
                * trial_outs.iter().map(|o| o.tw_batt_ok[ti]).sum::<f64>()
                * inv_t,
            wea_cell_online_pct: 100.0
                * trial_outs.iter().map(|o| o.tw_online[ti]).sum::<f64>()
                * inv_t,
            served_parcels: parcel_served_cnt[ti],
        });
    }

    let mut mobilization_curves = Vec::with_capacity(curve_pts);
    for ci in 0..curve_pts {
        mobilization_curves.push(MobilizationCurvePoint {
            minute: (ci as f64 / (curve_pts - 1) as f64) * win_min,
            wea_only_mobilized_pct: trial_outs.iter().map(|o| o.c_wea[ci]).sum::<f64>() * inv_t,
            wea_plus_siren_mobilized_pct: trial_outs.iter().map(|o| o.c_siren[ci]).sum::<f64>()
                * inv_t,
            multimodal_mobilized_pct: trial_outs.iter().map(|o| o.c_multi[ci]).sum::<f64>()
                * inv_t,
            staged_network_departed_pct: trial_outs.iter().map(|o| o.c_dep[ci]).sum::<f64>()
                * inv_t,
            safe_corridor_cleared_pct: trial_outs.iter().map(|o| o.c_clr[ci]).sum::<f64>() * inv_t,
        });
    }

    let mut parcel_mobilized_60m_pct = vec![100.0; n_parcels];
    let mut parcel_departure_delay_min = vec![10.0; n_parcels];
    let mut parcel_recommended_stage = vec![1u8; n_parcels];
    for pi in 0..n_parcels {
        parcel_mobilized_60m_pct[pi] =
            trial_outs.iter().map(|o| o.p_mob60[pi]).sum::<f64>() * inv_t;
        parcel_departure_delay_min[pi] =
            trial_outs.iter().map(|o| o.p_delay[pi]).sum::<f64>() * inv_t;
        let st_avg = trial_outs.iter().map(|o| o.p_stage[pi]).sum::<f64>() * inv_t;
        parcel_recommended_stage[pi] = (st_avg.round() as u8).clamp(1, 3);
    }

    // Representative Hi-Lo / Door-to-Door patrol canvass vectors toward darkest cellular pockets
    let mut canvass_sweep_paths = Vec::new();
    if cfg.channel_mix == AlertChannelMix::MultiModalWeaSirenHiLoDoorToDoor && !parcels.is_empty() {
        let mut dark_parcels: Vec<usize> = (0..n_parcels).collect();
        dark_parcels.sort_by(|&a, &b| {
            let ta = tower_rows[parcel_primary_bts[a]].wea_cell_online_pct;
            let tb = tower_rows[parcel_primary_bts[b]].wea_cell_online_pct;
            ta.partial_cmp(&tb).unwrap_or(Ordering::Equal)
        });
        for (i, &pi) in dark_parcels
            .iter()
            .take(cfg.canvass_patrol_units.min(6))
            .enumerate()
        {
            let orig_seg = patrol_origins[i % patrol_origins.len()];
            let p0 = if orig_seg < n_seg && !net.segments[orig_seg].base_pts.is_empty() {
                net.segments[orig_seg].base_pts[0]
            } else {
                parcels[0].center
            };
            canvass_sweep_paths.push((p0, parcels[pi].center));
        }
    }

    WarningMobilizationReport {
        config: cfg.clone(),
        cell_towers_count: n_towers,
        outdoor_sirens_count: sirens.len(),
        mean_cell_towers_blackout_pct: trial_outs.iter().map(|o| o.towers_dark_pct).sum::<f64>()
            * inv_t,
        population_in_wea_blackout_pct: trial_outs.iter().map(|o| o.pop_wea_dark_pct).sum::<f64>()
            * inv_t,
        mean_warning_receipt_min: trial_outs.iter().map(|o| o.mean_warn_min).sum::<f64>() * inv_t,
        median_mobilization_departure_min: trial_outs.iter().map(|o| o.p50_dep_min).sum::<f64>()
            * inv_t,
        p90_mobilization_departure_min: trial_outs.iter().map(|o| o.p90_dep_min).sum::<f64>()
            * inv_t,
        inner_fire_zone_clearance_min: trial_outs.iter().map(|o| o.inner_clr_min).sum::<f64>()
            * inv_t,
        network_peak_vc_ratio: trial_outs.iter().map(|o| o.peak_vc).sum::<f64>() * inv_t,
        late_departure_fire_trapped_pax: trial_outs.iter().map(|o| o.trapped_pax).sum::<f64>()
            * inv_t,
        p90_fire_trapped_pax: p90_trap,
        safe_evacuated_within_window_pct: trial_outs.iter().map(|o| o.safe_evac_pct).sum::<f64>()
            * inv_t,
        tower_rows,
        siren_rows: sirens,
        mobilization_curves,
        canvass_sweep_paths,
        parcel_mobilized_60m_pct,
        parcel_departure_delay_min,
        parcel_recommended_stage,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

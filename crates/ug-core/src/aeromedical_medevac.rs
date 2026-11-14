//! Post-Earthquake Rotary-Wing Aeromedical (`HEMS` / `Medevac`) Helipads, FAA 14 CFR § 135.609 +
//! Koschmieder Smoke-Plume & Wind Flight-Envelope Constraints, and Secondary Inter-Hospital
//! Regional Burn-Center (`ABA` / `WRBDC`) Transfer Engine
//!
//! Grounded in:
//! - FAA 14 CFR § 135.609 (Subpart L — Helicopter Air Ambulance VFR Ceiling & Visibility Minimums,
//!   `https://www.ecfr.gov/current/title-14/chapter-I/subchapter-G/part-135/subpart-L/section-135.609`)
//!   and FAA Notice N 8900.289 OpSpec A021 NVIS / HTAWS & PinS IFR operations
//!   (`https://www.faa.gov/documentLibrary/media/Notice/N_8900.289.pdf`).
//! - FAA Advisory Circular AC 150/5390-2 Hospital Rooftop vs. Surface Heliport Design
//!   (`https://rosap.ntl.bts.gov/view/dot/70775/dot_70775_DS1.pdf`).
//! - Koschmieder Meteorological Visibility Law (`Vis = 3.912 / b_ext`) under Particulate Smoke Plumes
//!   (Cai et al. 2024, `https://www.researchgate.net/publication/371647186_Measurement_of_Fugitive_Particulate_Matter_Emission_Current_State_and_Trends`).
//! - Post-Earthquake Helicopter Emergency Medical Services (`HEMS`, `https://pmc.ncbi.nlm.nih.gov/articles/PMC6637809/`),
//!   Western Region Burn Disaster Consortium (`WRBDC`,
//!   `https://files.asprtracie.hhs.gov/documents/wrbdc-2024-burn-mass-casualty-operations-plan.pdf`),
//!   and Oregon Health Authority *State Mass Burn Casualty Plan*
//!   (`https://www.oregon.gov/oha/PH/PROVIDERPARTNERRESOURCES/EMS/EMSFORCHILDREN/Documents/Oregon-Mass-Burn-Casualty-Plan.pdf`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::ffe_conflagration::{
    hamada_hazus_spread_rates, scawthorn_hazus_ignition_rate_per_mmsqft,
};
use crate::hospital_surge::{
    hazus_parcel_casualty_rates, hospital_fault_tree_functionality,
    synthesize_hospital_surge_roster, HospitalSurgeConfig, HospitalTraumaDesignation,
};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi, BuriedPipeMaterialClass, SeismicDamageState,
};
use crate::utilities::{MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const MEDEVAC_HELIPAD_CSV_HEADER: &str = "pad_id,name,pad_type,host_hospital_id,structural_ok_pct,smoke_vis_sm,flight_envelope_open_pct,primary_scene_airlifts,outbound_burn_transfers,inbound_burn_transfers,weather_smoke_scrubs";
pub const MEDEVAC_TRANSFER_ARC_CSV_HEADER: &str = "from_pad_id,from_name,to_pad_id,to_name,is_external_wrbdc_hub,distance_km,expected_air_transfers,expected_cct_ground_fallbacks,mean_air_transfer_min,mean_cct_ground_min";

const SQFT_PER_M2: f64 = 10.763_910_4;
const KM_PER_STATUTE_MILE: f64 = 1.609_344;

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

/// FAA 14 CFR § 135.609 & OpSpec A021 Helicopter Air Ambulance (`HAA`) ceiling & visibility operating regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MedevacAvionicsRegime {
    /// 14 CFR § 135.609 Day Local VFR (`Ceiling >= 800 ft`, `Visibility >= 2.0 SM`).
    DayVfrPart135,
    /// 14 CFR § 135.609 Night Unassisted VFR (`Ceiling >= 1,000 ft`, `Visibility >= 3.0 SM`).
    NightUnassistedVfr,
    /// FAA OpSpec A021 Night NVIS + HTAWS VFR (`Ceiling >= 800 ft`, `Visibility >= 2.5 SM`).
    NightNvisHtawsVfr,
    /// Hospital WAAS GPS Point-in-Space (`PinS`) Special Helicopter IFR (`Ceiling >= 350 ft`, `Visibility >= 0.75 SM`).
    WaasPinSIfr,
}

impl Default for MedevacAvionicsRegime {
    fn default() -> Self {
        Self::DayVfrPart135
    }
}

impl MedevacAvionicsRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::DayVfrPart135 => "14 CFR 135.609 Day VFR (800ft / 2.0 SM)",
            Self::NightUnassistedVfr => "135.609 Night Unassisted (1000ft / 3.0 SM)",
            Self::NightNvisHtawsVfr => "OpSpec A021 NVIS+HTAWS (800ft / 2.5 SM)",
            Self::WaasPinSIfr => "WAAS GPS PinS Helicopter IFR (350ft / 0.75 SM)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::DayVfrPart135 => "Day-VFR",
            Self::NightUnassistedVfr => "Night-VFR",
            Self::NightNvisHtawsVfr => "NVIS-VFR",
            Self::WaasPinSIfr => "PinS-IFR",
        }
    }

    /// Returns `(min_ceiling_ft, min_visibility_statute_miles)`.
    #[inline]
    pub fn weather_minima(self) -> (f64, f64) {
        match self {
            Self::DayVfrPart135 => (800.0, 2.0),
            Self::NightUnassistedVfr => (1_000.0, 3.0),
            Self::NightNvisHtawsVfr => (800.0, 2.5),
            Self::WaasPinSIfr => (350.0, 0.75),
        }
    }
}

/// Aeromedical rotorcraft airframe performance, litter capacity & crosswind envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotorcraftAirframeClass {
    /// Light twin HEMS (`Airbus H135 / Bell 429`: `250 km/h` cruise, `1` ICU litter, `15.5 m/s` max wind).
    LightTwinHemsEc135,
    /// Medium twin ICU transport (`Leonardo AW139 / S-76`: `285 km/h` cruise, `2` ICU litters, `18.5 m/s` max wind).
    MediumTwinIcuAw139,
    /// Heavy National Guard `UH-60M MEDEVAC` (`278 km/h` cruise, `4` litters, `21.5 m/s` max wind; surface/L1 pads).
    HeavyNationalGuardUh60,
}

impl Default for RotorcraftAirframeClass {
    fn default() -> Self {
        Self::MediumTwinIcuAw139
    }
}

impl RotorcraftAirframeClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::LightTwinHemsEc135 => "Light Twin HEMS (H135/B429, 1 litter, 135 kt)",
            Self::MediumTwinIcuAw139 => "Medium Twin ICU (AW139, 2 litters, 154 kt)",
            Self::HeavyNationalGuardUh60 => "National Guard MEDEVAC (UH-60M, 4 litters, 150 kt)",
        }
    }

    /// Returns `(cruise_kmh, litters_per_sortie, max_sustained_wind_mps, pad_hot_turnaround_min)`.
    #[inline]
    pub fn specs(self) -> (f64, usize, f64, f64) {
        match self {
            Self::LightTwinHemsEc135 => (250.0, 1, 15.5, 9.0),
            Self::MediumTwinIcuAw139 => (285.0, 2, 18.5, 11.5),
            Self::HeavyNationalGuardUh60 => (278.0, 4, 21.5, 14.0),
        }
    }
}

/// FAA AC 150/5390-2D helipad structural & operational configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HelipadStructuralType {
    /// Elevated hospital rooftop TLOF/FATO (vulnerable to building drift, elevator loss & rooftop wake turbulence).
    HospitalRooftopElevated,
    /// Seismically hardened ground-level hospital surface pad (`AC 150/5390-2D` direct ambulance/stretcher roll-off).
    HospitalSurfaceGradePad,
    /// Pre-designated `PARK` / municipal greenbelt emergency casualty airlift Landing Zone (`LZ`).
    DesignatedParkEmergencyLz,
    /// Off-map Western Region Burn Disaster Consortium (`WRBDC`) tertiary burn center hub (e.g. Portland / Seattle).
    ExternalRegionalBurnHub,
}

impl HelipadStructuralType {
    pub fn short_badge(self) -> &'static str {
        match self {
            Self::HospitalRooftopElevated => "H-ROOF",
            Self::HospitalSurfaceGradePad => "H-SURF",
            Self::DesignatedParkEmergencyLz => "LZ-PARK",
            Self::ExternalRegionalBurnHub => "WRBDC-HUB",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AeromedicalMedevacConfig {
    /// Coupled hospital ED surge, seismic fragility & Scawthorn/Hamada FFE configuration.
    pub surge: HospitalSurgeConfig,
    pub avionics_regime: MedevacAvionicsRegime,
    pub airframe_class: RotorcraftAirframeClass,
    /// Dedicated HEMS / National Guard rotorcraft staged in the regional pool (default `3`).
    pub rotorcraft_fleet_count: usize,
    /// Synoptic ambient cloud ceiling above ground level (`feet AGL`, default `1,600 ft`).
    pub ambient_ceiling_ft: f64,
    /// Synoptic pre-fire meteorological visibility (`statute miles`, default `6.0 SM`).
    pub ambient_visibility_sm: f64,
    /// Retrofits hospital rooftop pads to seismic `SPC-5` + surface grade backup pads & pre-designates Park LZs.
    pub helipad_seismic_and_ground_lz_hardening: bool,
    /// Activates interstate/intrastate `WRBDC` regional burn-center transfer mutual aid (Portland 128 km / Seattle 365 km).
    pub wrbdc_regional_burn_mutual_aid: bool,
}

impl Default for AeromedicalMedevacConfig {
    fn default() -> Self {
        Self {
            surge: HospitalSurgeConfig::default(),
            avionics_regime: MedevacAvionicsRegime::DayVfrPart135,
            airframe_class: RotorcraftAirframeClass::MediumTwinIcuAw139,
            rotorcraft_fleet_count: 3,
            ambient_ceiling_ft: 1_600.0,
            ambient_visibility_sm: 6.0,
            helipad_seismic_and_ground_lz_hardening: false,
            wrbdc_regional_burn_mutual_aid: true,
        }
    }
}

/// Computes Koschmieder meteorological visibility (`statute miles`) and smoke-lowered effective ceiling (`ft`)
/// from ambient `(ambient_vis_sm, ambient_ceiling_ft)` and local FFE plume `PM2.5` concentration (`ug/m^3`).
///
/// Uses `Vis_km = 3.912 / (b_ambient + b_smoke)` per Cai et al. (2024), where `b_smoke = 0.0042 * PM2.5_ug_m3`.
#[inline]
pub fn koschmieder_smoke_visibility_and_ceiling(
    ambient_vis_sm: f64,
    ambient_ceiling_ft: f64,
    pm25_ug_m3: f64,
) -> (f64, f64) {
    let amb_km = (ambient_vis_sm.clamp(0.15, 25.0)) * KM_PER_STATUTE_MILE;
    let b_amb_km_inv = 3.912 / amb_km;
    let b_smoke_km_inv = 0.0042 * pm25_ug_m3.max(0.0);
    let vis_km = 3.912 / (b_amb_km_inv + b_smoke_km_inv).max(1e-6);
    let vis_sm = (vis_km / KM_PER_STATUTE_MILE).clamp(0.05, 25.0);

    // Dense smoke plumes also create an obscuration ceiling aloft
    let smoke_ceil_cap_ft = 2_800.0 / (1.0 + 0.0032 * pm25_ug_m3.max(0.0));
    let eff_ceil_ft = ambient_ceiling_ft.clamp(100.0, 12_000.0).min(smoke_ceil_cap_ft);
    (vis_sm, eff_ceil_ft)
}

/// Evaluates whether a rotorcraft sortie at a given helipad/LZ is within its legal & aerodynamic flight envelope.
#[inline]
pub fn evaluate_helipad_flight_envelope(
    regime: MedevacAvionicsRegime,
    airframe: RotorcraftAirframeClass,
    pad_type: HelipadStructuralType,
    pad_structurally_open: bool,
    local_vis_sm: f64,
    local_ceiling_ft: f64,
    local_wind_mps: f64,
) -> bool {
    if !pad_structurally_open {
        return false;
    }
    let (req_ceil_ft, req_vis_sm) = regime.weather_minima();
    // Designated Park Emergency LZs lack published WAAS PinS instrument procedures (`Day/Night VFR` minima apply)
    let (eff_req_ceil, eff_req_vis) =
        if pad_type == HelipadStructuralType::DesignatedParkEmergencyLz
            && regime == MedevacAvionicsRegime::WaasPinSIfr
        {
            (600.0, 1.50) // Special Helicopter SVFR emergency scene minimum
        } else {
            (req_ceil_ft, req_vis_sm)
        };

    if local_ceiling_ft < eff_req_ceil || local_vis_sm < eff_req_vis {
        return false;
    }

    let (_, _, max_wind_mps, _) = airframe.specs();
    // Elevated rooftop pads experience building-edge vortex shear (~15% tighter crosswind envelope)
    let pad_wind_cap = if pad_type == HelipadStructuralType::HospitalRooftopElevated {
        max_wind_mps * 0.85
    } else {
        max_wind_mps
    };
    local_wind_mps <= pad_wind_cap
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelipadTelemetryRow {
    pub pad_id: usize,
    pub name: String,
    pub pad_type: HelipadStructuralType,
    pub host_hospital_id: Option<usize>,
    pub world_pos: Vec2,
    pub external_distance_km: f64,
    pub structural_ok_pct: f64,
    pub mean_smoke_vis_sm: f64,
    pub flight_envelope_open_pct: f64,
    pub expected_primary_scene_airlifts: f64,
    pub expected_outbound_burn_transfers: f64,
    pub expected_inbound_burn_transfers: f64,
    pub expected_weather_smoke_scrubs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterHospitalTransferArc {
    pub from_pad_id: usize,
    pub from_name: String,
    pub from_pos: Vec2,
    pub to_pad_id: usize,
    pub to_name: String,
    pub to_pos: Vec2,
    pub is_external_wrbdc_hub: bool,
    pub distance_km: f64,
    pub expected_air_transfers: f64,
    pub expected_cct_ground_fallbacks: f64,
    pub mean_air_transfer_min: f64,
    pub mean_cct_ground_min: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MedevacEpochPoint {
    pub hour: f64,
    pub mean_pad_smoke_vis_sm: f64,
    pub envelope_open_pads_pct: f64,
    pub cumulative_air_burn_transfers: f64,
    pub cumulative_cct_ground_fallbacks: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AeromedicalMedevacReport {
    pub config: AeromedicalMedevacConfig,
    pub total_helipads: usize,
    pub mean_helipad_structural_ok_pct: f64,
    pub mean_helipad_smoke_vis_sm: f64,
    pub system_flight_envelope_open_pct: f64,
    pub expected_burn_critical_casualties: f64,
    pub expected_burn_requiring_secondary_transfer: f64,
    pub expected_air_secondary_burn_transfers: f64,
    pub expected_wrbdc_regional_air_transfers: f64,
    pub expected_cct_ground_fallback_transfers: f64,
    pub expected_primary_scene_airlifts: f64,
    pub expected_weather_smoke_scrubbed_sorties: f64,
    pub mean_secondary_burn_transfer_min: f64,
    pub mean_air_burn_transfer_min: f64,
    pub mean_cct_ground_transfer_min: f64,
    pub burn_definitive_care_within_4hr_pct: f64,
    pub expected_preventable_burn_transfer_deaths: f64,
    pub timeline: Vec<MedevacEpochPoint>,
    pub helipad_rows: Vec<HelipadTelemetryRow>,
    pub transfer_arcs: Vec<InterHospitalTransferArc>,
    pub parcel_smoke_vis_sm: Vec<f64>,
    pub parcel_airlift_save_min: Vec<f64>,
    pub solve_time_ms: f64,
}

impl AeromedicalMedevacReport {
    pub fn to_helipad_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.helipad_rows.len() * 140 + 256);
        out.push_str(MEDEVAC_HELIPAD_CSV_HEADER);
        out.push('\n');
        for p in &self.helipad_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{},{:.1},{:.2},{:.1},{:.2},{:.2},{:.2},{:.2}\n",
                p.pad_id,
                p.name.replace('"', "'"),
                p.pad_type.short_badge(),
                p.host_hospital_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "NONE".to_string()),
                p.structural_ok_pct,
                p.mean_smoke_vis_sm,
                p.flight_envelope_open_pct,
                p.expected_primary_scene_airlifts,
                p.expected_outbound_burn_transfers,
                p.expected_inbound_burn_transfers,
                p.expected_weather_smoke_scrubs
            ));
        }
        out
    }

    pub fn to_transfer_arc_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.transfer_arcs.len() * 128 + 256);
        out.push_str(MEDEVAC_TRANSFER_ARC_CSV_HEADER);
        out.push('\n');
        for a in &self.transfer_arcs {
            out.push_str(&format!(
                "{},\"{}\",{},\"{}\",{},{:.1},{:.3},{:.3},{:.1},{:.1}\n",
                a.from_pad_id,
                a.from_name.replace('"', "'"),
                a.to_pad_id,
                a.to_name.replace('"', "'"),
                a.is_external_wrbdc_hub,
                a.distance_km,
                a.expected_air_transfers,
                a.expected_cct_ground_fallbacks,
                a.mean_air_transfer_min,
                a.mean_cct_ground_min
            ));
        }
        out
    }

    pub fn export_helipad_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_helipad_csv_string())?;
        Ok(self.helipad_rows.len())
    }
}

fn median_sa10_gmpe_g(mw: f64, r_rup_km: f64, vs30_mps: f64) -> f64 {
    let m = mw.clamp(5.0, 9.3);
    let r = r_rup_km.max(1.5);
    let h_eff = (2.5 + (0.48 * (m - 6.0)).exp().min(18.0)).max(3.5);
    let r_eff = (r * r + h_eff * h_eff).sqrt();
    let dm = m - 6.5;
    let mag_term = if dm <= 0.0 {
        0.92 * dm
    } else {
        0.74 * dm - 0.085 * dm * dm
    };
    let dist_term = -(1.08 + 0.055 * (m - 6.5)) * (r_eff / 10.0).ln() - 0.0032 * r_eff;
    let site_term = 0.38 * (760.0 / vs30_mps.clamp(140.0, 1100.0)).ln();
    ((0.24_f64).ln() + mag_term + dist_term + site_term)
        .exp()
        .clamp(0.005, 3.50)
}

#[derive(Copy, Clone, PartialEq)]
struct DijkNode {
    cost: f64,
    id: usize,
}
impl Eq for DijkNode {}
impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Executes `Rayon`-parallelized post-earthquake rotary-wing `HEMS` / `Medevac` simulations coupling
/// hospital rooftop/surface helipad fragility, Koschmieder FFE smoke-plume visibility & wind flight-envelope
/// gates, primary bridge-islanded scene airlifts, and secondary inter-hospital `ABA`/`WRBDC` burn-center transfers.
pub fn run_aeromedical_medevac_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &AeromedicalMedevacConfig,
) -> AeromedicalMedevacReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let n_segs = net.segments.len();
    let n_junc = net.junctions.len();
    let trials = cfg.surge.ffe.seismic.monte_carlo_trials.clamp(20, 5_000);

    let hospitals = synthesize_hospital_surge_roster(net, facilities, &cfg.surge);
    let n_hosp = hospitals.len();

    // Build helipad roster:
    // - 1 helipad per hospital (`HospitalSurfaceGradePad` if `helipad_seismic_and_ground_lz_hardening` or `L1`/`MFH`, else `HospitalRooftopElevated`)
    // - Up to 2 `DesignatedParkEmergencyLz` pads on `ZoneCode::PARK` parcels (west & east river banks)
    // - Up to 2 `ExternalRegionalBurnHub` nodes (`Oregon Burn Center — Portland 128 km`, `UW Harborview — Seattle 365 km`)
    #[derive(Clone)]
    struct PadSpec {
        pad_id: usize,
        name: String,
        pad_type: HelipadStructuralType,
        host_hosp: Option<usize>,
        pos: Vec2,
        ext_dist_km: f64,
        ext_burn_beds: f64,
    }

    let mut pads: Vec<PadSpec> = Vec::new();
    for (hi, h) in hospitals.iter().enumerate() {
        let p_type = if h.designation == HospitalTraumaDesignation::MobileFieldHospital
            || cfg.helipad_seismic_and_ground_lz_hardening
        {
            HelipadStructuralType::HospitalSurfaceGradePad
        } else {
            HelipadStructuralType::HospitalRooftopElevated
        };
        pads.push(PadSpec {
            pad_id: pads.len(),
            name: format!("{} Pad", h.name),
            pad_type: p_type,
            host_hosp: Some(hi),
            pos: h.world_pos,
            ext_dist_km: 0.0,
            ext_burn_beds: 0.0,
        });
    }

    // Add Park Emergency Landing Zones (`LZ`) for primary airlift of bridge-severed neighborhoods
    let b = net.bounds;
    let mid_x = 0.5 * (b.min_x + b.max_x);
    for (side_lbl, west_side) in [("Westbank Park Emergency LZ", true), ("Eastbank Park Emergency LZ", false)] {
        if let Some(pk_p) = parcels
            .iter()
            .filter(|p| {
                (p.zone == ZoneCode::PARK || cfg.helipad_seismic_and_ground_lz_hardening)
                    && ((west_side && p.center.x <= mid_x) || (!west_side && p.center.x > mid_x))
            })
            .min_by(|a, bp| {
                a.center
                    .y
                    .partial_cmp(&bp.center.y)
                    .unwrap_or(Ordering::Equal)
            })
        {
            pads.push(PadSpec {
                pad_id: pads.len(),
                name: side_lbl.to_string(),
                pad_type: HelipadStructuralType::DesignatedParkEmergencyLz,
                host_hosp: None,
                pos: pk_p.center,
                ext_dist_km: 0.0,
                ext_burn_beds: 0.0,
            });
        }
    }

    let portland_pad_idx = pads.len();
    pads.push(PadSpec {
        pad_id: portland_pad_idx,
        name: "Oregon Burn Center (Legacy Emanuel, Portland 128 km)".to_string(),
        pad_type: HelipadStructuralType::ExternalRegionalBurnHub,
        host_hosp: None,
        pos: Vec2 {
            x: b.min_x + 0.82 * (b.max_x - b.min_x),
            y: b.min_y + 18.0, // North-northeast edge gateway marker for viewport arc rendering
        },
        ext_dist_km: 128.0,
        ext_burn_beds: if cfg.wrbdc_regional_burn_mutual_aid {
            24.0 // Oregon Mass Burn Casualty Plan 24-patient surge capacity
        } else {
            0.0
        },
    });

    let seattle_pad_idx = pads.len();
    pads.push(PadSpec {
        pad_id: seattle_pad_idx,
        name: "UW Harborview Regional Burn Ctr (WRBDC Seattle 365 km)".to_string(),
        pad_type: HelipadStructuralType::ExternalRegionalBurnHub,
        host_hosp: None,
        pos: Vec2 {
            x: b.min_x + 0.92 * (b.max_x - b.min_x),
            y: b.min_y + 18.0,
        },
        ext_dist_km: 365.0,
        ext_burn_beds: if cfg.wrbdc_regional_burn_mutual_aid {
            32.0
        } else {
            0.0
        },
    });

    let n_pads = pads.len();
    let park_lz_indices: Vec<usize> = pads
        .iter()
        .filter(|p| p.pad_type == HelipadStructuralType::DesignatedParkEmergencyLz)
        .map(|p| p.pad_id)
        .collect();

    let fire_station_segs: Vec<usize> = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation && f.segment_id < n_segs)
        .map(|f| f.segment_id)
        .collect();
    let n_engines = (fire_station_segs.len().max(1) * 2).max(2);

    // Precompute parcel & hospital seismic medians + FFE Hamada spread adjacency
    let west_x = b.min_x;
    let mut parcel_pop = Vec::with_capacity(n_parcels);
    let mut parcel_tfa = Vec::with_capacity(n_parcels);
    let mut parcel_sa10 = Vec::with_capacity(n_parcels);
    let mut parcel_pga = Vec::with_capacity(n_parcels);
    let mut parcel_near_junc = Vec::with_capacity(n_parcels);
    let mut parcel_near_seg = Vec::with_capacity(n_parcels);
    let mut parcel_acc_s = Vec::with_capacity(n_parcels);
    let mut parcel_hamada = Vec::with_capacity(n_parcels);

    for p in parcels {
        let sp = p.zone.spec();
        let pop = (p.residents() * 0.85 + p.jobs() * 0.55).max(if p.zone == ZoneCode::PARK {
            0.5
        } else {
            14.0
        });
        let raw_tfa = (p.area_ha * 10_000.0 * sp.far * SQFT_PER_M2) / 1_000_000.0;
        let tfa = if p.zone == ZoneCode::PARK {
            0.001
        } else {
            raw_tfa.max(0.045)
        };
        let loc_km = ((p.center.x - west_x).max(0.0)) / 1000.0;
        let r_km = (cfg.surge.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(
            cfg.surge.ffe.seismic.moment_magnitude_mw,
            r_km,
            cfg.surge.ffe.seismic.vs30_m_per_s,
        );
        let pga = (0.86
            * sa10
            * (cfg.surge.ffe.seismic.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
        .clamp(0.01, 3.0);

        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                p.center
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&p.center.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);

        let mut b_sid = 0usize;
        let mut b_acc = 20.0;
        let mut b_sd = f64::INFINITY;
        for (sid, seg) in net.segments.iter().enumerate() {
            if !seg.closed && !seg.base_pts.is_empty() {
                let d = p.center.dist(seg.base_pts[seg.base_pts.len() / 2]);
                if d < b_sd {
                    b_sd = d;
                    b_sid = sid;
                    b_acc = d / 7.5;
                }
            }
        }

        parcel_pop.push(pop);
        parcel_tfa.push(tfa);
        parcel_sa10.push(sa10);
        parcel_pga.push(pga);
        parcel_near_junc.push(nj);
        parcel_near_seg.push(b_sid);
        parcel_acc_s.push(b_acc);
        parcel_hamada.push(hamada_hazus_spread_rates(p.zone, cfg.surge.ffe.wind_speed_mps));
    }

    let mut hosp_sa10 = Vec::with_capacity(n_hosp);
    let mut hosp_pga = Vec::with_capacity(n_hosp);
    let mut hosp_near_junc = Vec::with_capacity(n_hosp);
    for h in &hospitals {
        let loc_km = ((h.world_pos.x - west_x).max(0.0)) / 1000.0;
        let r_km = (cfg.surge.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(
            cfg.surge.ffe.seismic.moment_magnitude_mw,
            r_km,
            cfg.surge.ffe.seismic.vs30_m_per_s,
        );
        let pga = (0.86
            * sa10
            * (cfg.surge.ffe.seismic.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
        .clamp(0.01, 3.0);
        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                h.world_pos
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&h.world_pos.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        hosp_sa10.push(sa10);
        hosp_pga.push(pga);
        hosp_near_junc.push(nj);
    }

    // Hamada fire-spread graph
    let wind_rad = cfg.surge.ffe.wind_direction_deg.to_radians();
    let wx_hat = wind_rad.cos();
    let wy_hat = wind_rad.sin();
    let horizon_m = cfg.surge.ffe.conflagration_horizon_mins.clamp(15.0, 720.0);
    let mut fire_spread_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_parcels];
    for i in 0..n_parcels {
        if parcels[i].zone == ZoneCode::PARK {
            continue;
        }
        let pi = &parcels[i];
        let rates = parcel_hamada[i];
        for k in 0..n_parcels {
            if i == k || parcels[k].zone == ZoneCode::PARK {
                continue;
            }
            let pk = &parcels[k];
            let dx = pk.center.x - pi.center.x;
            let dy = pk.center.y - pi.center.y;
            let center_dist = dx.hypot(dy);
            if !(1.0..=280.0).contains(&center_dist) {
                continue;
            }
            let gap_x = (pk.min.x - pi.max.x).max(pi.min.x - pk.max.x).max(0.0);
            let gap_y = (pk.min.y - pi.max.y).max(pi.min.y - pk.max.y).max(0.0);
            let edge_gap_m = gap_x.hypot(gap_y);
            let cos_th = (dx * wx_hat + dy * wy_hat) / center_dist;
            let sin_th = (-dx * wy_hat + dy * wx_hat).abs() / center_dist;
            let dir_jump_cap = if cos_th >= 0.25 {
                rates.max_jump_gap_downwind_m * cos_th + 14.0 * (1.0 - cos_th)
            } else if cos_th >= -0.25 {
                14.5 + 0.85 * cfg.surge.ffe.wind_speed_mps
            } else {
                10.5 + 0.25 * cfg.surge.ffe.wind_speed_mps
            };
            if edge_gap_m > dir_jump_cap {
                continue;
            }
            let a_semi = if cos_th >= 0.0 {
                rates.v_down_m_min
            } else {
                rates.v_up_m_min
            };
            let b_semi = rates.v_cross_m_min;
            let inv_r = ((cos_th / a_semi.max(0.05)).powi(2) + (sin_th / b_semi.max(0.05)).powi(2))
                .sqrt();
            let v_theta = (1.0 / inv_r.max(1e-6)).max(0.05);
            let step_min = center_dist / v_theta + 2.5 + 0.18 * edge_gap_m;
            if step_min <= horizon_m {
                fire_spread_adj[i].push((k, step_min));
            }
        }
    }

    let bridges = enumerate_seismic_bridge_inventory(net, &cfg.surge.ffe.seismic);
    let n_br = bridges.len();

    let mut u_corridors: Vec<(usize, f64, Vec2, bool)> = Vec::new();
    let mut junc_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_junc];
    for s in &net.segments {
        if s.from_junc < s.to_junc
            || !net
                .segments
                .iter()
                .any(|o| o.from_junc == s.to_junc && o.to_junc == s.from_junc)
        {
            let cid = u_corridors.len();
            let mid = s.base_pts[s.base_pts.len() / 2];
            u_corridors.push((cid, s.length.max(12.0), mid, s.priority >= 2));
            junc_adj[s.from_junc].push((s.to_junc, cid));
            junc_adj[s.to_junc].push((s.from_junc, cid));
        }
    }
    let n_corr = u_corridors.len();
    let n_rep_corr = ((n_corr as f64)
        * (cfg.surge.ffe.seismic.seismic_pipe_replacement_pct.clamp(0.0, 100.0) / 100.0))
        .round() as usize;
    let wtp_junc = (0..n_junc)
        .min_by(|&a, &bj| {
            let tgt = Vec2 {
                x: b.min_x,
                y: 0.5 * (b.min_y + b.max_y),
            };
            net.junctions[a]
                .pos
                .dist(tgt)
                .partial_cmp(&net.junctions[bj].pos.dist(tgt))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);

    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    let base_seg_trav: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                f64::INFINITY
            } else {
                s.length / (s.ema_speed_mps.max(s.speed_mps * 0.48) * speed_boost).clamp(3.2, 29.0)
            }
        })
        .collect();
    let mut base_fwd: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    let mut base_rev: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let sig_pen = match net.junctions[seg.to_junc].control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => {
                if evp_cfg.enabled {
                    2.2
                } else {
                    14.5
                }
            }
            ControlMode::Roundabout => 3.4,
            ControlMode::PermissiveYield => 1.8,
        };
        for &lid in &seg.lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let conn = &net.connectors[cid];
                let out_sid = conn.to_seg_id;
                if net.segments[out_sid].closed {
                    continue;
                }
                let t_pen = match conn.turn_type {
                    TurnType::Through => 0.5,
                    TurnType::Right => 2.2,
                    TurnType::Left => 3.0,
                };
                if !base_fwd[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    base_fwd[seg.id].push((out_sid, sig_pen + t_pen + base_seg_trav[out_sid]));
                    base_rev[out_sid].push((seg.id, sig_pen + t_pen + base_seg_trav[seg.id]));
                }
            }
        }
    }

    let n_sites = n_parcels + n_br + n_hosp;
    let mut site_pts = Vec::with_capacity(n_sites);
    for p in parcels {
        site_pts.push(p.center);
    }
    for br in &bridges {
        let sid = br.segment_ids[0];
        site_pts.push(net.segments[sid].base_pts[net.segments[sid].base_pts.len() / 2]);
    }
    for h in &hospitals {
        site_pts.push(h.world_pos);
    }
    let range_km = if cfg.surge.ffe.seismic.vs30_m_per_s < 270.0 {
        29.5
    } else {
        25.0
    };
    let mut corr_mat = vec![0.0; n_sites * n_sites];
    for i in 0..n_sites {
        for j in 0..n_sites {
            let d_km = site_pts[i].dist(site_pts[j]) / 1000.0;
            corr_mat[i * n_sites + j] = (-3.0 * d_km / range_km).exp();
        }
    }
    let chol = cholesky_lower(&corr_mat, n_sites);

    let horizon_h = cfg.surge.surge_window_hours.clamp(2.0, 24.0);
    let n_waves = ((horizon_h * 4.0).round() as usize).clamp(8, 96);
    let mut wave_weights = Vec::with_capacity(n_waves);
    for w in 0..n_waves {
        let t_hr = (w as f64 + 0.5) * 0.25;
        wave_weights.push((t_hr / 1.15).powf(1.35) * (-t_hr / 1.45).exp() + 0.04);
    }
    let w_sum: f64 = wave_weights.iter().sum::<f64>().max(1e-9);
    for rw in &mut wave_weights {
        *rw /= w_sum;
    }

    // Enumerate candidate secondary burn-transfer arcs:
    // Every non-L1 hospital pad `h_src` -> `[L1 Burn Center pad 0, Portland Burn Hub, Seattle Burn Hub]`
    // plus `L1 Burn Center pad 0` -> `[Portland Burn Hub, Seattle Burn Hub]` when local L1 overflows!
    let mut arc_defs: Vec<(usize, usize, bool, f64)> = Vec::new();
    for hi in 0..n_hosp {
        if hospitals[hi].designation == HospitalTraumaDesignation::MobileFieldHospital {
            continue;
        }
        if hi != 0 {
            let d_loc_km = (pads[hi].pos.dist(pads[0].pos) / 1000.0).max(2.5);
            arc_defs.push((hi, 0, false, d_loc_km));
        }
        arc_defs.push((hi, portland_pad_idx, true, 128.0));
        arc_defs.push((hi, seattle_pad_idx, true, 365.0));
    }
    let n_arcs = arc_defs.len();

    let (cruise_kmh, litters_per_ac, _, pad_turn_min) = cfg.airframe_class.specs();
    let fleet_n = cfg.rotorcraft_fleet_count.clamp(1, 16);

    #[derive(Clone)]
    struct MedTrialOut {
        burn_crit_tot: f64,
        burn_sec_req: f64,
        air_sec_xfers: f64,
        wrbdc_air_xfers: f64,
        cct_ground_xfers: f64,
        scene_airlifts: f64,
        wx_smoke_scrubs: f64,
        sum_sec_xfer_min: f64,
        sum_air_xfer_min: f64,
        sum_cct_xfer_min: f64,
        burn_4hr_ok: f64,
        prev_burn_deaths: f64,
        pad_struct_ok: Vec<f64>,
        pad_smoke_vis: Vec<f64>,
        pad_env_open: Vec<f64>,
        pad_scene_lifts: Vec<f64>,
        pad_out_burn: Vec<f64>,
        pad_in_burn: Vec<f64>,
        pad_scrubs: Vec<f64>,
        arc_air: Vec<f64>,
        arc_cct: Vec<f64>,
        arc_air_min_sum: Vec<f64>,
        arc_cct_min_sum: Vec<f64>,
        parcel_vis_sm: Vec<f64>,
        parcel_save_min: Vec<f64>,
        wave_vis: Vec<f64>,
        wave_env_pct: Vec<f64>,
        wave_air_cum: Vec<f64>,
        wave_cct_cum: Vec<f64>,
    }

    let seed_base = cfg.surge.ffe.seismic.random_seed ^ 0xA3E0_3ED4_3400_C719;

    let trial_outs: Vec<MedTrialOut> = (0..trials)
        .into_par_iter()
        .map(|trial_idx| {
            let mut rng = SplitMix64::new(
                seed_base.wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let eta_inter = rng.next_standard_normal();
            let mut z_ind = vec![0.0; n_sites];
            for z in &mut z_ind {
                *z = rng.next_standard_normal();
            }
            let mut eps_intra = vec![0.0; n_sites];
            for i in 0..n_sites {
                let mut s = 0.0;
                for k in 0..=i {
                    s += chol[i * n_sites + k] * z_ind[k];
                }
                eps_intra[i] = s;
            }
            let tau_b = cfg.surge.ffe.seismic.inter_event_tau;
            let phi_w = cfg.surge.ffe.seismic.intra_event_phi;

            // Trial gust perturbation around configured FFE wind speed (`+-12%`)
            let trial_wind_mps =
                (cfg.surge.ffe.wind_speed_mps * (1.0 + 0.12 * rng.next_standard_normal())).max(0.0);

            // 1. Bridge closures & Dijkstra travel times
            let mut seg_closed = vec![false; n_segs];
            let mut closed_br_cnt = 0usize;
            for (bi, br) in bridges.iter().enumerate() {
                let sa_r = (br.median_site_sa10_g.ln()
                    + tau_b * eta_inter
                    + phi_w * eps_intra[n_parcels + bi])
                    .exp();
                let probs = br.effective_archetype.exceedance_probs(sa_r);
                if rng.next_f64_open01() < probs[2] {
                    closed_br_cnt += 1;
                    for &sid in &br.segment_ids {
                        if sid < n_segs {
                            seg_closed[sid] = true;
                        }
                    }
                }
            }

            let mut seg_engine_s = vec![f64::INFINITY; n_segs];
            let mut heap = BinaryHeap::new();
            for &fs_sid in &fire_station_segs {
                if !seg_closed[fs_sid] && base_seg_trav[fs_sid].is_finite() {
                    let c0 = 60.0 + 0.5 * base_seg_trav[fs_sid];
                    if c0 < seg_engine_s[fs_sid] {
                        seg_engine_s[fs_sid] = c0;
                        heap.push(DijkNode {
                            cost: c0,
                            id: fs_sid,
                        });
                    }
                }
            }
            while let Some(DijkNode { cost, id }) = heap.pop() {
                if cost > seg_engine_s[id] {
                    continue;
                }
                for &(nxt, w) in &base_fwd[id] {
                    if !seg_closed[nxt] && cost + w < seg_engine_s[nxt] {
                        seg_engine_s[nxt] = cost + w;
                        heap.push(DijkNode {
                            cost: cost + w,
                            id: nxt,
                        });
                    }
                }
            }

            let mut hosp_parcel_min = vec![vec![45.0; n_parcels]; n_hosp];
            for (hi, h) in hospitals.iter().enumerate() {
                let mut seg_to_h = vec![f64::INFINITY; n_segs];
                heap.clear();
                if h.segment_id < n_segs
                    && !seg_closed[h.segment_id]
                    && base_seg_trav[h.segment_id].is_finite()
                {
                    let init_s = 0.5 * base_seg_trav[h.segment_id];
                    seg_to_h[h.segment_id] = init_s;
                    heap.push(DijkNode {
                        cost: init_s,
                        id: h.segment_id,
                    });
                }
                while let Some(DijkNode { cost, id }) = heap.pop() {
                    if cost > seg_to_h[id] {
                        continue;
                    }
                    for &(prev_sid, w) in &base_rev[id] {
                        if !seg_closed[prev_sid] && cost + w < seg_to_h[prev_sid] {
                            seg_to_h[prev_sid] = cost + w;
                            heap.push(DijkNode {
                                cost: cost + w,
                                id: prev_sid,
                            });
                        }
                    }
                }
                for pi in 0..n_parcels {
                    let sid = parcel_near_seg[pi];
                    let s_to_h = if seg_to_h[sid].is_finite() {
                        seg_to_h[sid] + parcel_acc_s[pi]
                    } else {
                        (parcels[pi].center.dist(h.world_pos) / 3.4) + 420.0
                    };
                    let resp_s = if seg_engine_s[sid].is_finite() {
                        (seg_engine_s[sid] + parcel_acc_s[pi]).min(720.0)
                    } else {
                        480.0
                    };
                    hosp_parcel_min[hi][pi] = (resp_s + s_to_h) / 60.0 + 4.5;
                }
            }

            // 2. Water mains, substation power & FFE conflagration
            let mut pipe_open = vec![true; n_corr];
            let mut leak_lps = 0.0;
            for &(cid, len_m, mid, is_art) in &u_corridors {
                let loc_km = ((mid.x - west_x).max(0.0)) / 1000.0;
                let r_km = (cfg.surge.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
                let sa_c = (median_sa10_gmpe_g(
                    cfg.surge.ffe.seismic.moment_magnitude_mw,
                    r_km,
                    cfg.surge.ffe.seismic.vs30_m_per_s,
                )
                .ln()
                    + tau_b * eta_inter
                    + phi_w * rng.next_standard_normal())
                .exp();
                let pgv = median_pgv_cm_s_from_sa10(sa_c, cfg.surge.ffe.seismic.vs30_m_per_s)
                    .clamp(1.0, 350.0);
                let mat = if cid < n_rep_corr {
                    BuriedPipeMaterialClass::SeismicErdipHdpe
                } else {
                    cfg.surge.ffe.seismic.baseline_water_pipe_material
                };
                let rr = ala_hazus_pipe_repair_rate_per_km(
                    pgv,
                    mat,
                    if is_art { 300.0 } else { 200.0 },
                    cfg.surge.ffe.seismic.pipe_fragility_law,
                );
                let n_rep = sample_poisson_icdf(rr * (len_m / 1000.0), rng.next_f64_open01());
                if n_rep > 0 {
                    let mut brk = 0usize;
                    for _ in 0..n_rep {
                        if rng.next_f64_open01() < 0.25 {
                            brk += 1;
                        }
                    }
                    if brk > 0 {
                        pipe_open[cid] = false;
                    }
                    leak_lps += ((n_rep - brk) as f64) * 6.5 + (brk as f64) * 24.0;
                }
            }

            let sub_sa = (parcel_sa10.first().copied().unwrap_or(0.35).ln() + tau_b * eta_inter).exp();
            let sub_grid_ok =
                rng.next_f64_open01() >= normal_cdf_phi((sub_sa.ln() - (0.58_f64).ln()) / 0.55);
            let pump_head_m =
                if sub_grid_ok || cfg.surge.ffe.seismic.critical_lifeline_backup_power {
                    55.0
                } else {
                    23.0
                };
            let mut w_dist = vec![usize::MAX; n_junc];
            let mut q = VecDeque::new();
            if wtp_junc < n_junc {
                w_dist[wtp_junc] = 0;
                q.push_back(wtp_junc);
            }
            while let Some(u) = q.pop_front() {
                for &(v, cid) in &junc_adj[u] {
                    if pipe_open[cid] && w_dist[v] == usize::MAX {
                        w_dist[v] = w_dist[u] + 1;
                        q.push_back(v);
                    }
                }
            }

            let mut p_sa_real = vec![0.0; n_parcels];
            let mut ignitions: Vec<(usize, f64)> = Vec::new();
            for pi in 0..n_parcels {
                let sa_r = (parcel_sa10[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.8);
                let pga_r = (parcel_pga[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.5);
                p_sa_real[pi] = sa_r;
                let ig_rate = scawthorn_hazus_ignition_rate_per_mmsqft(
                    pga_r,
                    parcels[pi].zone,
                    cfg.surge.ffe.ignition_model,
                    cfg.surge.ffe.seismic_gas_shutoff_pct,
                );
                if sample_poisson_icdf(ig_rate * parcel_tfa[pi], rng.next_f64_open01()) > 0 {
                    let sid = parcel_near_seg[pi];
                    let eng_s = if seg_engine_s[sid].is_finite() {
                        seg_engine_s[sid] + parcel_acc_s[pi]
                    } else {
                        9999.0
                    };
                    ignitions.push((pi, eng_s));
                }
            }
            ignitions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let drawdown_m =
                ((leak_lps + (ignitions.len() as f64) * 58.0) / 145.0).powf(1.852) * 5.8;
            let junc_psi: Vec<f64> = (0..n_junc)
                .map(|ji| {
                    if w_dist[ji] == usize::MAX {
                        0.0
                    } else {
                        ((pump_head_m - ((w_dist[ji] as f64) * 1.45 + drawdown_m)).max(0.0))
                            * PSI_PER_METER_H2O
                    }
                })
                .collect();

            let cistern_prob =
                (cfg.surge.ffe.auxiliary_cistern_coverage_pct.clamp(0.0, 100.0)) / 100.0;
            let mut rem_eng = n_engines;
            let mut escaped_seeds = Vec::new();
            let mut knockdown_parcels = vec![false; n_parcels];
            for &(pi, eng_s) in &ignitions {
                let has_w = junc_psi[parcel_near_junc[pi]] >= MIN_FIRE_FLOW_RESIDUAL_PSI
                    || rng.next_f64_open01() < cistern_prob;
                if rem_eng > 0 && eng_s <= cfg.surge.ffe.initial_attack_flashover_s && has_w {
                    rem_eng -= 1;
                    knockdown_parcels[pi] = true;
                } else {
                    if rem_eng > 0 && eng_s <= cfg.surge.ffe.initial_attack_flashover_s {
                        rem_eng = rem_eng.saturating_sub(1);
                    }
                    escaped_seeds.push(pi);
                }
            }

            let mut burn_time_min = vec![f64::INFINITY; n_parcels];
            let mut fheap = BinaryHeap::new();
            for &sp in &escaped_seeds {
                burn_time_min[sp] = 0.0;
                fheap.push(DijkNode { cost: 0.0, id: sp });
            }
            while let Some(DijkNode { cost, id }) = fheap.pop() {
                if cost > burn_time_min[id] {
                    continue;
                }
                for &(nxt, step_m) in &fire_spread_adj[id] {
                    let nt = cost + step_m;
                    if nt <= horizon_m && nt < burn_time_min[nxt] {
                        burn_time_min[nxt] = nt;
                        fheap.push(DijkNode { cost: nt, id: nxt });
                    }
                }
            }

            // 3. Helipad seismic structural + elevator/standpipe availability & hospital burn beds
            let mut h_burn_beds = vec![0.0; n_hosp];
            let mut pad_struct_ok = vec![true; n_pads];
            for (hi, h) in hospitals.iter().enumerate() {
                let h_idx = n_parcels + n_br + hi;
                let sa_h = (hosp_sa10[hi].ln() + tau_b * eta_inter + phi_w * eps_intra[h_idx]).exp();
                let pga_h = (hosp_pga[hi].ln() + tau_b * eta_inter + phi_w * eps_intra[h_idx]).exp();
                let psi_h = junc_psi[hosp_near_junc[hi]];
                let backup_ok = cfg.surge.hospital_onsite_backup_lifelines
                    || cfg.surge.ffe.seismic.critical_lifeline_backup_power
                    || cfg.helipad_seismic_and_ground_lz_hardening;
                let q_func = hospital_fault_tree_functionality(
                    h.designation,
                    h.seismically_retrofitted || cfg.helipad_seismic_and_ground_lz_hardening,
                    sa_h,
                    pga_h,
                    rng.next_standard_normal(),
                    sub_grid_ok,
                    psi_h,
                    backup_ok,
                );
                h_burn_beds[hi] = h.baseline_burn_beds * q_func;

                // Rooftop helipad fails if building q_func < 0.48 (Extensive/Complete drift or elevator power loss)
                // AND UH-60 heavy airframes cannot land on unhardened community rooftop pads!
                let heavy_restricted = cfg.airframe_class
                    == RotorcraftAirframeClass::HeavyNationalGuardUh60
                    && pads[hi].pad_type == HelipadStructuralType::HospitalRooftopElevated
                    && h.designation != HospitalTraumaDesignation::Level1TraumaAndBurnCenter;

                pad_struct_ok[hi] = if pads[hi].pad_type
                    == HelipadStructuralType::HospitalRooftopElevated
                {
                    q_func >= 0.48 && (sub_grid_ok || backup_ok) && !heavy_restricted
                } else {
                    q_func >= 0.18
                };
            }

            // 4. Sample parcel Red & BurnCritical casualties
            let mut p_red = vec![0usize; n_parcels];
            let mut p_burn = vec![0usize; n_parcels];
            for pi in 0..n_parcels {
                let (rf, _, _) = hazus_parcel_casualty_rates(p_sa_real[pi], parcels[pi].zone);
                let pop = parcel_pop[pi];
                p_red[pi] = sample_poisson_icdf(pop * rf, rng.next_f64_open01());
                if burn_time_min[pi].is_finite() {
                    let trap = (1.0 - 0.55 * (burn_time_min[pi] / horizon_m).clamp(0.0, 1.0))
                        .clamp(0.35, 1.0);
                    p_burn[pi] = sample_poisson_icdf(pop * 0.0095 * trap, rng.next_f64_open01());
                } else if knockdown_parcels[pi] {
                    p_burn[pi] = sample_poisson_icdf(pop * 0.0014, rng.next_f64_open01());
                }
            }

            // Helper computing Gaussian downwind FFE smoke PM2.5 (`ug/m^3`) at world position `pt` at `time_min`
            let smoke_pm25_at = |pt: Vec2, time_min: f64| -> f64 {
                let mut pm = 0.0;
                let u_adv = trial_wind_mps.max(1.4);
                for qi in 0..n_parcels {
                    if !burn_time_min[qi].is_finite() || burn_time_min[qi] > time_min {
                        continue;
                    }
                    // Active flaming/smoldering intensity decays gradually after 120 min
                    let age_m = (time_min - burn_time_min[qi]).max(0.0);
                    let burn_intensity = (-age_m / 150.0).exp().max(0.22);
                    let dx = pt.x - parcels[qi].center.x;
                    let dy = pt.y - parcels[qi].center.y;
                    let x_down = dx * wx_hat + dy * wy_hat;
                    let y_cross = (-dx * wy_hat + dy * wx_hat).abs();
                    if x_down >= -45.0 {
                        let eff_x = x_down.max(25.0);
                        let sig_y = 55.0 + 0.24 * eff_x;
                        let gauss = (-0.5 * (y_cross / sig_y).powi(2)).exp();
                        let q_src = parcel_tfa[qi] * 46_000.0 * burn_intensity;
                        pm += (q_src / (u_adv * sig_y)) * gauss;
                    }
                }
                pm
            };

            // Evaluate mid-surge (`t = 90 min`) parcel smoke visibility & potential airlift time savings for GPU heatmap
            let mut parcel_vis_sm = vec![cfg.ambient_visibility_sm; n_parcels];
            let mut parcel_save_min = vec![0.0; n_parcels];
            for pi in 0..n_parcels {
                let pm = smoke_pm25_at(parcels[pi].center, 90.0);
                let (v_sm, c_ft) = koschmieder_smoke_visibility_and_ceiling(
                    cfg.ambient_visibility_sm,
                    cfg.ambient_ceiling_ft,
                    pm,
                );
                parcel_vis_sm[pi] = v_sm;
                let best_gnd_m = (0..n_hosp)
                    .map(|hi| hosp_parcel_min[hi][pi])
                    .fold(f64::INFINITY, f64::min);
                if let Some(&lz_id) = park_lz_indices.iter().min_by(|&&a, &&b_lz| {
                    parcels[pi]
                        .center
                        .dist(pads[a].pos)
                        .partial_cmp(&parcels[pi].center.dist(pads[b_lz].pos))
                        .unwrap_or(Ordering::Equal)
                }) {
                    let env_ok = evaluate_helipad_flight_envelope(
                        cfg.avionics_regime,
                        cfg.airframe_class,
                        HelipadStructuralType::DesignatedParkEmergencyLz,
                        true,
                        v_sm,
                        c_ft,
                        trial_wind_mps,
                    ) && evaluate_helipad_flight_envelope(
                        cfg.avionics_regime,
                        cfg.airframe_class,
                        pads[0].pad_type,
                        pad_struct_ok[0],
                        v_sm,
                        c_ft,
                        trial_wind_mps,
                    );
                    if env_ok {
                        let d_lz_m = parcels[pi].center.dist(pads[lz_id].pos);
                        let air_km = (pads[lz_id].pos.dist(pads[0].pos) / 1000.0).max(1.5);
                        let tot_air_m = (d_lz_m / 240.0) + 6.5 + (air_km / cruise_kmh) * 60.0;
                        parcel_save_min[pi] = (best_gnd_m - tot_air_m).max(0.0);
                    }
                }
            }

            // 5. Simulate 15-minute surge waves for primary scene airlifts & secondary inter-hospital burn transfers
            let mut local_burn_occ = vec![0.0; n_hosp];
            let mut ext_portland_rem = pads[portland_pad_idx].ext_burn_beds;
            let mut ext_seattle_rem = pads[seattle_pad_idx].ext_burn_beds;

            let mut burn_crit_tot = 0.0;
            let mut burn_sec_req = 0.0;
            let mut air_sec_xfers = 0.0;
            let mut wrbdc_air_xfers = 0.0;
            let mut cct_ground_xfers = 0.0;
            let mut scene_airlifts = 0.0;
            let mut wx_smoke_scrubs = 0.0;
            let mut sum_sec_xfer_min = 0.0;
            let mut sum_air_xfer_min = 0.0;
            let mut sum_cct_xfer_min = 0.0;
            let mut burn_4hr_ok = 0.0;
            let mut prev_burn_deaths = 0.0;

            let mut pad_smoke_sum = vec![0.0; n_pads];
            let mut pad_env_sum = vec![0.0; n_pads];
            let mut pad_scene_lifts = vec![0.0; n_pads];
            let mut pad_out_burn = vec![0.0; n_pads];
            let mut pad_in_burn = vec![0.0; n_pads];
            let mut pad_scrubs = vec![0.0; n_pads];

            let mut arc_air = vec![0.0; n_arcs];
            let mut arc_cct = vec![0.0; n_arcs];
            let mut arc_air_min_sum = vec![0.0; n_arcs];
            let mut arc_cct_min_sum = vec![0.0; n_arcs];

            let mut wave_vis = vec![0.0; n_waves];
            let mut wave_env_pct = vec![0.0; n_waves];
            let mut wave_air_cum = vec![0.0; n_waves];
            let mut wave_cct_cum = vec![0.0; n_waves];

            for (wi, &ww) in wave_weights.iter().enumerate() {
                let t_min = (wi as f64 + 0.5) * 15.0;

                // Evaluate current smoke visibility & flight envelope at every helipad
                let mut pad_env_now = vec![false; n_pads];
                let mut vis_hosp_acc = 0.0;
                let mut env_hosp_acc = 0.0;
                for pid in 0..n_pads {
                    let (v_sm, c_ft) = if pads[pid].pad_type
                        == HelipadStructuralType::ExternalRegionalBurnHub
                    {
                        // External WRBDC hubs (Portland/Seattle) are outside local FFE smoke plume
                        (cfg.ambient_visibility_sm.max(3.5), cfg.ambient_ceiling_ft.max(1200.0))
                    } else {
                        let pm = smoke_pm25_at(pads[pid].pos, t_min);
                        koschmieder_smoke_visibility_and_ceiling(
                            cfg.ambient_visibility_sm,
                            cfg.ambient_ceiling_ft,
                            pm,
                        )
                    };
                    let open = evaluate_helipad_flight_envelope(
                        cfg.avionics_regime,
                        cfg.airframe_class,
                        pads[pid].pad_type,
                        pad_struct_ok[pid],
                        v_sm,
                        c_ft,
                        trial_wind_mps,
                    );
                    pad_env_now[pid] = open;
                    pad_smoke_sum[pid] += v_sm;
                    if open {
                        pad_env_sum[pid] += 1.0;
                    }
                    if pid < n_hosp {
                        vis_hosp_acc += v_sm;
                        if open {
                            env_hosp_acc += 1.0;
                        }
                    }
                }
                wave_vis[wi] = vis_hosp_acc / (n_hosp.max(1) as f64);
                wave_env_pct[wi] = (env_hosp_acc / (n_hosp.max(1) as f64)) * 100.0;

                // Available rotorcraft patient-litter capacity in this 15-min wave
                // (`~1.6` local/regional mission cycles per hour = `0.40 * fleet * litters` per 15m)
                let mut rem_air_litters = (fleet_n as f64) * (litters_per_ac as f64) * 0.45;

                // A. Primary scene airlift for bridge-detoured critical casualties (`gnd_min >= 22 min`)
                let mut pending_hosp_burn = vec![0.0; n_hosp];
                for pi in 0..n_parcels {
                    let b_vol = (p_burn[pi] as f64) * ww;
                    let r_vol = (p_red[pi] as f64) * ww;
                    if b_vol + r_vol <= 1e-9 {
                        continue;
                    }
                    burn_crit_tot += b_vol;

                    let closest_h = (0..n_hosp)
                        .filter(|&hi| {
                            hospitals[hi].designation != HospitalTraumaDesignation::MobileFieldHospital
                        })
                        .min_by(|&a, &bh| {
                            hosp_parcel_min[a][pi]
                                .partial_cmp(&hosp_parcel_min[bh][pi])
                                .unwrap_or(Ordering::Equal)
                        })
                        .unwrap_or(0);
                    let gnd_m = hosp_parcel_min[closest_h][pi];

                    // Check if primary scene airlift via Park LZ to Level-I Trauma/Burn Center saves >= 8 min
                    let mut airlifted_b = 0.0;
                    if gnd_m >= 22.0 && rem_air_litters > 1e-6 && !park_lz_indices.is_empty() {
                        let best_lz = *park_lz_indices
                            .iter()
                            .min_by(|&&a, &&b_lz| {
                                parcels[pi]
                                    .center
                                    .dist(pads[a].pos)
                                    .partial_cmp(&parcels[pi].center.dist(pads[b_lz].pos))
                                    .unwrap_or(Ordering::Equal)
                            })
                            .unwrap();
                        let air_m = (parcels[pi].center.dist(pads[best_lz].pos) / 240.0)
                            + pad_turn_min * 0.7
                            + ((pads[best_lz].pos.dist(pads[0].pos) / 1000.0).max(1.5)
                                / cruise_kmh)
                                * 60.0;
                        if air_m + 8.0 < gnd_m {
                            if pad_env_now[best_lz] && pad_env_now[0] {
                                let req = b_vol + r_vol;
                                let lifted = req.min(rem_air_litters);
                                rem_air_litters -= lifted;
                                scene_airlifts += lifted;
                                pad_scene_lifts[best_lz] += lifted;
                                pad_in_burn[0] += lifted * (b_vol / req);
                                airlifted_b = lifted * (b_vol / req);
                                pending_hosp_burn[0] += airlifted_b;
                            } else {
                                wx_smoke_scrubs += b_vol + r_vol;
                                pad_scrubs[best_lz] += b_vol + r_vol;
                            }
                        }
                    }

                    let rem_b = (b_vol - airlifted_b).max(0.0);
                    if rem_b > 1e-9 {
                        pending_hosp_burn[closest_h] += rem_b;
                    }
                }

                // B. Admit `BurnCritical` patients to local Burn ICU beds where available; queue overflow
                //    and non-burn hospital (`L2`/`L3`) burn arrivals for Secondary Inter-Hospital Burn Transfer!
                let bridge_detour_pen_min = (closed_br_cnt as f64) * 18.0;
                for hi in 0..n_hosp {
                    let arr_b = pending_hosp_burn[hi];
                    if arr_b <= 1e-9 {
                        continue;
                    }
                    let free_local = (h_burn_beds[hi] - local_burn_occ[hi]).max(0.0);
                    let admitted = arr_b.min(free_local);
                    local_burn_occ[hi] += admitted;
                    // Locally admitted at verified Burn Center -> immediate definitive burn care (`<= 4 hr` OK!)
                    burn_4hr_ok += admitted * 0.96;

                    let mut need_xfer = (arr_b - admitted).max(0.0);
                    if need_xfer <= 1e-9 {
                        continue;
                    }
                    burn_sec_req += need_xfer;

                    // Candidate receiving Burn Centers in priority order:
                    // 1) Local Level-I Burn Center (`pad 0`, if `hi != 0` and `pad 0` still has free burn beds)
                    // 2) Oregon Burn Center — Portland (`128 km`, `WRBDC`)
                    // 3) UW Harborview Burn Center — Seattle (`365 km`, `WRBDC`)
                    let mut dest_candidates: Vec<(usize, bool, f64)> = Vec::new();
                    if hi != 0 && (h_burn_beds[0] - local_burn_occ[0]) > 0.05 {
                        let d_km = (pads[hi].pos.dist(pads[0].pos) / 1000.0).max(2.5);
                        dest_candidates.push((0, false, d_km));
                    }
                    if cfg.wrbdc_regional_burn_mutual_aid {
                        dest_candidates.push((portland_pad_idx, true, 128.0));
                        dest_candidates.push((seattle_pad_idx, true, 365.0));
                    } else if hi != 0 {
                        let d_km = (pads[hi].pos.dist(pads[0].pos) / 1000.0).max(2.5);
                        dest_candidates.push((0, false, d_km));
                    } else {
                        // Without WRBDC mutual aid, local L1 overflow must wait ~5h for regional ground convoy
                        dest_candidates.push((portland_pad_idx, true, 128.0));
                    }

                    for (dst_pad, is_ext, dist_km) in dest_candidates {
                        if need_xfer <= 1e-9 {
                            break;
                        }
                        let dst_avail = if dst_pad == 0 {
                            (h_burn_beds[0] - local_burn_occ[0]).max(0.0)
                        } else if dst_pad == portland_pad_idx {
                            if cfg.wrbdc_regional_burn_mutual_aid {
                                ext_portland_rem.max(0.0)
                            } else {
                                0.0 // Forces ground convoy delay if mutual-aid air bridge disabled
                            }
                        } else if cfg.wrbdc_regional_burn_mutual_aid {
                            ext_seattle_rem.max(0.0)
                        } else {
                            0.0
                        };

                        let chunk = if dst_avail > 0.05 {
                            need_xfer.min(dst_avail)
                        } else if dst_pad == seattle_pad_idx || !cfg.wrbdc_regional_burn_mutual_aid {
                            need_xfer
                        } else {
                            continue;
                        };

                        let arc_idx = arc_defs
                            .iter()
                            .position(|&(f, t, _, _)| f == hi && t == dst_pad);

                        // ED airway/fluid stabilization (`28 min`) + air vs. ground CCT transfer time
                        let air_min = 28.0 + pad_turn_min + (dist_km / cruise_kmh) * 60.0;
                        let cct_ground_min = 38.0
                            + (dist_km / 68.0) * 60.0
                            + bridge_detour_pen_min
                            + if !cfg.wrbdc_regional_burn_mutual_aid && is_ext {
                                95.0
                            } else {
                                0.0
                            };

                        let air_allowed = pad_env_now[hi]
                            && pad_env_now[dst_pad]
                            && (cfg.wrbdc_regional_burn_mutual_aid || !is_ext);

                        let air_sent = if air_allowed && rem_air_litters > 1e-6 {
                            chunk.min(rem_air_litters)
                        } else {
                            if !pad_env_now[hi] || !pad_env_now[dst_pad] {
                                wx_smoke_scrubs += chunk;
                                pad_scrubs[hi] += chunk;
                            }
                            0.0
                        };
                        rem_air_litters = (rem_air_litters - air_sent).max(0.0);
                        let cct_sent = (chunk - air_sent).max(0.0);

                        if dst_pad == 0 {
                            local_burn_occ[0] += chunk;
                        } else if dst_pad == portland_pad_idx {
                            ext_portland_rem = (ext_portland_rem - chunk).max(0.0);
                        } else {
                            ext_seattle_rem = (ext_seattle_rem - chunk).max(0.0);
                        }

                        if air_sent > 1e-9 {
                            air_sec_xfers += air_sent;
                            if is_ext {
                                wrbdc_air_xfers += air_sent;
                            }
                            pad_out_burn[hi] += air_sent;
                            pad_in_burn[dst_pad] += air_sent;
                            sum_sec_xfer_min += air_sent * air_min;
                            sum_air_xfer_min += air_sent * air_min;
                            let p_4h = 1.0 / (1.0 + ((air_min - 240.0) / 32.0).exp());
                            burn_4hr_ok += air_sent * p_4h;
                            let ex_m = (air_min - 150.0).max(0.0);
                            prev_burn_deaths +=
                                air_sent * (1.0 - (-0.0026 * ex_m).exp()).clamp(0.0, 0.55);
                            if let Some(ai) = arc_idx {
                                arc_air[ai] += air_sent;
                                arc_air_min_sum[ai] += air_sent * air_min;
                            }
                        }

                        if cct_sent > 1e-9 {
                            cct_ground_xfers += cct_sent;
                            sum_sec_xfer_min += cct_sent * cct_ground_min;
                            sum_cct_xfer_min += cct_sent * cct_ground_min;
                            let p_4h = 1.0 / (1.0 + ((cct_ground_min - 240.0) / 32.0).exp());
                            burn_4hr_ok += cct_sent * p_4h;
                            let ex_m = (cct_ground_min - 150.0).max(0.0);
                            prev_burn_deaths +=
                                cct_sent * (1.0 - (-0.0026 * ex_m).exp()).clamp(0.0, 0.65);
                            if let Some(ai) = arc_idx {
                                arc_cct[ai] += cct_sent;
                                arc_cct_min_sum[ai] += cct_sent * cct_ground_min;
                            }
                        }

                        need_xfer = (need_xfer - chunk).max(0.0);
                    }
                }

                // Gradual burn bed turnover (`~1.5%` per 15 min)
                for occ in &mut local_burn_occ {
                    *occ *= 0.985;
                }
                wave_air_cum[wi] = air_sec_xfers;
                wave_cct_cum[wi] = cct_ground_xfers;
            }

            let nw_f = n_waves as f64;
            MedTrialOut {
                burn_crit_tot,
                burn_sec_req,
                air_sec_xfers,
                wrbdc_air_xfers,
                cct_ground_xfers,
                scene_airlifts,
                wx_smoke_scrubs,
                sum_sec_xfer_min,
                sum_air_xfer_min,
                sum_cct_xfer_min,
                burn_4hr_ok,
                prev_burn_deaths,
                pad_struct_ok: pad_struct_ok
                    .into_iter()
                    .map(|ok| if ok { 100.0 } else { 0.0 })
                    .collect(),
                pad_smoke_vis: pad_smoke_sum.into_iter().map(|s| s / nw_f).collect(),
                pad_env_open: pad_env_sum.into_iter().map(|s| (s / nw_f) * 100.0).collect(),
                pad_scene_lifts,
                pad_out_burn,
                pad_in_burn,
                pad_scrubs,
                arc_air,
                arc_cct,
                arc_air_min_sum,
                arc_cct_min_sum,
                parcel_vis_sm,
                parcel_save_min,
                wave_vis,
                wave_env_pct,
                wave_air_cum,
                wave_cct_cum,
            }
        })
        .collect();

    let tf = trials as f64;
    let exp_burn_tot = trial_outs.iter().map(|t| t.burn_crit_tot).sum::<f64>() / tf;
    let exp_sec_req = trial_outs.iter().map(|t| t.burn_sec_req).sum::<f64>() / tf;
    let exp_air_sec = trial_outs.iter().map(|t| t.air_sec_xfers).sum::<f64>() / tf;
    let exp_wrbdc = trial_outs.iter().map(|t| t.wrbdc_air_xfers).sum::<f64>() / tf;
    let exp_cct = trial_outs.iter().map(|t| t.cct_ground_xfers).sum::<f64>() / tf;
    let exp_scene = trial_outs.iter().map(|t| t.scene_airlifts).sum::<f64>() / tf;
    let exp_scrubs = trial_outs.iter().map(|t| t.wx_smoke_scrubs).sum::<f64>() / tf;
    let exp_prev_d = trial_outs.iter().map(|t| t.prev_burn_deaths).sum::<f64>() / tf;

    let tot_sec_all: f64 = trial_outs.iter().map(|t| t.burn_sec_req).sum();
    let tot_air_all: f64 = trial_outs.iter().map(|t| t.air_sec_xfers).sum();
    let tot_cct_all: f64 = trial_outs.iter().map(|t| t.cct_ground_xfers).sum();
    let tot_burn_all: f64 = trial_outs.iter().map(|t| t.burn_crit_tot).sum();

    let mean_sec_min = if tot_sec_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_sec_xfer_min).sum::<f64>() / tot_sec_all
    } else {
        48.0
    };
    let mean_air_min = if tot_air_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_air_xfer_min).sum::<f64>() / tot_air_all
    } else {
        52.0
    };
    let mean_cct_min = if tot_cct_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_cct_xfer_min).sum::<f64>() / tot_cct_all
    } else {
        175.0
    };
    let burn_4hr_pct = if tot_burn_all > 1e-9 {
        ((trial_outs.iter().map(|t| t.burn_4hr_ok).sum::<f64>() / tot_burn_all) * 100.0)
            .clamp(0.0, 100.0)
    } else {
        100.0
    };

    let mut helipad_rows = Vec::with_capacity(n_pads);
    for (pid, p) in pads.iter().enumerate() {
        helipad_rows.push(HelipadTelemetryRow {
            pad_id: pid,
            name: p.name.clone(),
            pad_type: p.pad_type,
            host_hospital_id: p.host_hosp,
            world_pos: p.pos,
            external_distance_km: p.ext_dist_km,
            structural_ok_pct: trial_outs.iter().map(|t| t.pad_struct_ok[pid]).sum::<f64>() / tf,
            mean_smoke_vis_sm: trial_outs.iter().map(|t| t.pad_smoke_vis[pid]).sum::<f64>() / tf,
            flight_envelope_open_pct: trial_outs.iter().map(|t| t.pad_env_open[pid]).sum::<f64>()
                / tf,
            expected_primary_scene_airlifts: trial_outs
                .iter()
                .map(|t| t.pad_scene_lifts[pid])
                .sum::<f64>()
                / tf,
            expected_outbound_burn_transfers: trial_outs
                .iter()
                .map(|t| t.pad_out_burn[pid])
                .sum::<f64>()
                / tf,
            expected_inbound_burn_transfers: trial_outs
                .iter()
                .map(|t| t.pad_in_burn[pid])
                .sum::<f64>()
                / tf,
            expected_weather_smoke_scrubs: trial_outs
                .iter()
                .map(|t| t.pad_scrubs[pid])
                .sum::<f64>()
                / tf,
        });
    }

    let local_pad_cnt = n_hosp.max(1) as f64;
    let mean_struct_pct = helipad_rows
        .iter()
        .take(n_hosp)
        .map(|r| r.structural_ok_pct)
        .sum::<f64>()
        / local_pad_cnt;
    let mean_vis_sm = helipad_rows
        .iter()
        .take(n_hosp)
        .map(|r| r.mean_smoke_vis_sm)
        .sum::<f64>()
        / local_pad_cnt;
    let sys_env_pct = helipad_rows
        .iter()
        .take(n_hosp)
        .map(|r| r.flight_envelope_open_pct)
        .sum::<f64>()
        / local_pad_cnt;

    let mut transfer_arcs = Vec::with_capacity(n_arcs);
    for (ai, &(f_id, t_id, is_ext, dist_km)) in arc_defs.iter().enumerate() {
        let a_sum: f64 = trial_outs.iter().map(|t| t.arc_air[ai]).sum();
        let c_sum: f64 = trial_outs.iter().map(|t| t.arc_cct[ai]).sum();
        let am_sum: f64 = trial_outs.iter().map(|t| t.arc_air_min_sum[ai]).sum();
        let cm_sum: f64 = trial_outs.iter().map(|t| t.arc_cct_min_sum[ai]).sum();
        transfer_arcs.push(InterHospitalTransferArc {
            from_pad_id: f_id,
            from_name: pads[f_id].name.clone(),
            from_pos: pads[f_id].pos,
            to_pad_id: t_id,
            to_name: pads[t_id].name.clone(),
            to_pos: pads[t_id].pos,
            is_external_wrbdc_hub: is_ext,
            distance_km: dist_km,
            expected_air_transfers: a_sum / tf,
            expected_cct_ground_fallbacks: c_sum / tf,
            mean_air_transfer_min: if a_sum > 1e-9 {
                am_sum / a_sum
            } else {
                28.0 + pad_turn_min + (dist_km / cruise_kmh) * 60.0
            },
            mean_cct_ground_min: if c_sum > 1e-9 {
                cm_sum / c_sum
            } else {
                38.0 + (dist_km / 68.0) * 60.0
            },
        });
    }

    let mut timeline = Vec::with_capacity(n_waves);
    for wi in 0..n_waves {
        timeline.push(MedevacEpochPoint {
            hour: (wi as f64 + 1.0) * 0.25,
            mean_pad_smoke_vis_sm: trial_outs.iter().map(|t| t.wave_vis[wi]).sum::<f64>() / tf,
            envelope_open_pads_pct: trial_outs.iter().map(|t| t.wave_env_pct[wi]).sum::<f64>() / tf,
            cumulative_air_burn_transfers: trial_outs
                .iter()
                .map(|t| t.wave_air_cum[wi])
                .sum::<f64>()
                / tf,
            cumulative_cct_ground_fallbacks: trial_outs
                .iter()
                .map(|t| t.wave_cct_cum[wi])
                .sum::<f64>()
                / tf,
        });
    }

    let parcel_smoke_vis_sm: Vec<f64> = (0..n_parcels)
        .map(|pi| trial_outs.iter().map(|t| t.parcel_vis_sm[pi]).sum::<f64>() / tf)
        .collect();
    let parcel_airlift_save_min: Vec<f64> = (0..n_parcels)
        .map(|pi| trial_outs.iter().map(|t| t.parcel_save_min[pi]).sum::<f64>() / tf)
        .collect();

    AeromedicalMedevacReport {
        config: cfg.clone(),
        total_helipads: n_pads,
        mean_helipad_structural_ok_pct: mean_struct_pct,
        mean_helipad_smoke_vis_sm: mean_vis_sm,
        system_flight_envelope_open_pct: sys_env_pct,
        expected_burn_critical_casualties: exp_burn_tot,
        expected_burn_requiring_secondary_transfer: exp_sec_req,
        expected_air_secondary_burn_transfers: exp_air_sec,
        expected_wrbdc_regional_air_transfers: exp_wrbdc,
        expected_cct_ground_fallback_transfers: exp_cct,
        expected_primary_scene_airlifts: exp_scene,
        expected_weather_smoke_scrubbed_sorties: exp_scrubs,
        mean_secondary_burn_transfer_min: mean_sec_min,
        mean_air_burn_transfer_min: mean_air_min,
        mean_cct_ground_transfer_min: mean_cct_min,
        burn_definitive_care_within_4hr_pct: burn_4hr_pct,
        expected_preventable_burn_transfer_deaths: exp_prev_d,
        timeline,
        helipad_rows,
        transfer_arcs,
        parcel_smoke_vis_sm,
        parcel_airlift_save_min,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

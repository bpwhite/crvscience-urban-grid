//! Post-Earthquake Emergency Evacuation Shelter Siting, ATC-20 Seismic + Water/Sewer/Power
//! Lifeline Habitability Fault Trees, EPA / ASHRAE `MERV-13` + Portable `HEPA` (`CADR`) Smoke
//! Filtration Mass-Balance, and Capacitated Multi-Shelter Evacuee Assignment Routing
//!
//! Grounded in:
//! - FEMA P-361 *Design and Construction Guidance for Community Safe Rooms* & ARC 4496 Congregate
//!   Shelter Standards (`40-60 sq ft / evacuee`, ATC-20 Immediate-Occupancy green-tag & lifeline
//!   habitability, `https://mema.maryland.gov/Documents/fema361_Safe%20Room%20Construction.pdf`).
//! - US EPA *Research on Air Cleaners to Reduce Wildfire Smoke Indoors*
//!   (`https://www.epa.gov/air-research/research-diy-air-cleaners-reduce-wildfire-smoke-indoors`),
//!   Health Canada *Guidance for Cleaner Air Spaces during Wildfire Smoke Events*
//!   (`https://www.canada.ca/content/dam/hc-sc/documents/services/publications/healthy-living/guidance-cleaner-air-spaces-during-wildfire-smoke-events/guidance-cleaner-air-spaces-during-wildfire-smoke-events.pdf`),
//!   and NCCEH *Wildfire Smoke Indoor-Outdoor Evidence Review*
//!   (`https://ccnse.ca/sites/default/files/2026-02/NCCEH%20%E2%80%93%20Wildfire%20Smoke%20Indoor%20Outdoor%20Evidence%20Review%2020260202.pdf`).
//! - Two-Stage Post-Earthquake Shelter Location-Allocation & Capacitated Evacuee Assignment
//!   (*MDPI Sustainability* 17(23):10747, `https://www.mdpi.com/2071-1050/17/23/10747`;
//!   *Geomatics, Natural Hazards and Risk*, `https://www.tandfonline.com/doi/full/10.1080/19475705.2019.1570977`).

use crate::emergency::EmergencyFacility;
use crate::evacuation_contraflow::{
    plan_contraflow_and_protected_ems_corridors, smoke_visibility_speed_and_capacity_factors,
    EvacuationContraflowConfig,
};
use crate::ffe_conflagration::{
    hamada_hazus_spread_rates, scawthorn_hazus_ignition_rate_per_mmsqft,
};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi, BuriedPipeMaterialClass,
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

pub const SHELTER_FACILITY_CSV_HEADER: &str = "shelter_id,name,archetype,selected_open,gross_floor_sqft,nominal_cot_capacity,atc20_green_tag_pct,lifeline_habitable_pct,outdoor_pm25_ug_m3,indoor_pm25_ug_m3,clean_air_ok_pct,mean_effective_capacity,expected_assigned_evacuees,peak_occupancy_pct";
pub const SHELTER_PARCEL_ASSIGN_CSV_HEADER: &str = "parcel_id,zone_code,displaced_evacuees,primary_shelter_id,mean_travel_min,indoor_pm25_at_assigned_ug_m3,sheltered_clean_air_pct,unmet_overflow_evacuees";

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

/// Structural & enclosure archetype of candidate municipal mass-care shelters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShelterFacilityArchetype {
    /// Large university / municipal sports fieldhouse (`42,000 sq ft`, steel long-span frame, high HVAC volume).
    UniversityFieldhouseArena,
    /// Public high/middle school gymnasium (`18,000 sq ft`, CMU/steel frame).
    PublicSchoolGymnasium,
    /// Downtown civic community / convention center (`26,000 sq ft`, RC moment frame).
    CivicCommunityCenter,
    /// Open-air seismic park tent pavilion (`24,000 sq ft` tented area: immune to building collapse, but high outdoor smoke infiltration unless sealed positive-pressure HEPA tents deployed!).
    OpenParkTentPavilion,
}

impl ShelterFacilityArchetype {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::UniversityFieldhouseArena => "Fieldhouse Arena",
            Self::PublicSchoolGymnasium => "School Gymnasium",
            Self::CivicCommunityCenter => "Civic Center",
            Self::OpenParkTentPavilion => "Park Tent Camp",
        }
    }

    pub fn badge_prefix(self) -> &'static str {
        match self {
            Self::UniversityFieldhouseArena => "SH-ARENA",
            Self::PublicSchoolGymnasium => "SH-GYM",
            Self::CivicCommunityCenter => "SH-CIVIC",
            Self::OpenParkTentPavilion => "SH-PARK",
        }
    }

    /// Returns `(usable_congregate_sqft, ceiling_height_m, baseline_sa10_green_tag_median_g, base_air_changes_per_hr)`.
    pub fn physical_specs(self) -> (f64, f64, f64, f64) {
        match self {
            Self::UniversityFieldhouseArena => (42_000.0, 9.5, 0.72, 0.42),
            Self::PublicSchoolGymnasium => (18_500.0, 7.2, 0.48, 0.58),
            Self::CivicCommunityCenter => (26_000.0, 5.5, 0.56, 0.48),
            Self::OpenParkTentPavilion => (28_000.0, 3.6, 9.99, 2.60),
        }
    }
}

/// Indoor smoke filtration & HVAC recirculation tier per US EPA / Health Canada / ASHRAE Guideline 44.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AirFiltrationStandard {
    /// Legacy `MERV-8` commercial HVAC without portable HEPA units (`I/O ~ 0.65-0.75` during smoke events).
    BaselineMerv8Unsealed,
    /// Upgraded `MERV-13` central HVAC in 85% recirculation mode (`~68-74%` outdoor `PM2.5` reduction when powered).
    RecirculatingMerv13Hvac,
    /// Cleaner Air Resilience Center: `MERV-13` HVAC + Vestibule Airlocks + Portable `HEPA` (`CADR`) bank (`~88-93%` `PM2.5` reduction!).
    CleanerAirMerv13PlusPortableHepa,
}

impl Default for AirFiltrationStandard {
    fn default() -> Self {
        Self::CleanerAirMerv13PlusPortableHepa
    }
}

impl AirFiltrationStandard {
    pub fn label(self) -> &'static str {
        match self {
            Self::BaselineMerv8Unsealed => "Baseline MERV-8 HVAC (I/O ~ 0.68)",
            Self::RecirculatingMerv13Hvac => "MERV-13 Recirculating HVAC (~70% PM2.5 Cut)",
            Self::CleanerAirMerv13PlusPortableHepa => {
                "ASHRAE44 MERV-13 + Portable HEPA Bank (~90% Cut)"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::BaselineMerv8Unsealed => "MERV-8",
            Self::RecirculatingMerv13Hvac => "MERV-13",
            Self::CleanerAirMerv13PlusPortableHepa => "MERV13+HEPA",
        }
    }
}

/// Shelter siting & post-earthquake evacuee routing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShelterSitingAndRoutingPolicy {
    /// Uncoordinated evacuees self-route strictly to the geographically nearest open shelter (causes severe overflow & smoke exposure!).
    UncoordinatedNearestShelter,
    /// Capacitated Shortest-Time Shelter Routing (`Sherali / Bayram`) without upwind smoke-plume siting optimization.
    CapacitatedDistanceOnly,
    /// Two-Stage Hazard-Aware Siting + Capacitated Clean-Air & Lifeline Assignment (`MDPI 2025` / `EPA Cleaner Air`).
    HazardAwareCleanAirCapacitated,
}

impl Default for ShelterSitingAndRoutingPolicy {
    fn default() -> Self {
        Self::HazardAwareCleanAirCapacitated
    }
}

impl ShelterSitingAndRoutingPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::UncoordinatedNearestShelter => "Uncoordinated Nearest-Shelter Self-Routing",
            Self::CapacitatedDistanceOnly => "Capacitated Travel-Time Assignment (Smoke-Blind)",
            Self::HazardAwareCleanAirCapacitated => {
                "2-Stage Upwind/Seismic Siting + Clean-Air Capacitated Routing"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::UncoordinatedNearestShelter => "Nearest-Uncoord",
            Self::CapacitatedDistanceOnly => "Cap-DistOnly",
            Self::HazardAwareCleanAirCapacitated => "2Stage-CleanAir",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencySheltersConfig {
    /// Coupled evacuation contraflow, medevac, hospital surge, seismic fragility & FFE parameters.
    pub evac: EvacuationContraflowConfig,
    pub filtration_standard: AirFiltrationStandard,
    pub siting_routing_policy: ShelterSitingAndRoutingPolicy,
    /// Number of municipal mass-care shelters opened from the candidate pool (`2..=8`, default `4`).
    pub max_open_shelters_budget: usize,
    /// ARC 4496 / FEMA congregate floor-area standard per evacuee (`sq ft / person`, `40 = acute`, `60 = multi-day/ADA`).
    pub sqft_per_evacuee: f64,
    /// Fraction (`0..=100%`) of shelter buildings retrofitted to Immediate-Occupancy (`IO`) seismic structural bracing.
    pub shelter_seismic_retrofit_pct: f64,
    /// Equips shelters with emergency diesel generators, potable water bladders & seismic chemical sanitation vaults!
    pub shelter_backup_power_water_sanitation: bool,
    /// Portable `HEPA` Clean Air Delivery Rate (`CADR` in `m^3/h` deployed per `100 m^2` of shelter floor area, default `420.0`).
    pub portable_hepa_cadr_m3h_per_100m2: f64,
    /// Maximum acceptable 24-hr indoor `PM2.5` threshold (`ug/m^3`, US EPA NAAQS = `35.0 ug/m^3`).
    pub indoor_pm25_health_limit_ug_m3: f64,
}

impl Default for EmergencySheltersConfig {
    fn default() -> Self {
        Self {
            evac: EvacuationContraflowConfig::default(),
            filtration_standard: AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
            siting_routing_policy: ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated,
            max_open_shelters_budget: 4,
            sqft_per_evacuee: 50.0,
            shelter_seismic_retrofit_pct: 50.0,
            shelter_backup_power_water_sanitation: false,
            portable_hepa_cadr_m3h_per_100m2: 420.0,
            indoor_pm25_health_limit_ug_m3: 35.0,
        }
    }
}

/// Steady-state well-mixed indoor particulate mass-balance equation (`ug/m^3`) per US EPA / ASHRAE Guideline 44:
///
/// `C_in = C_out * (P * lambda_inf) / (lambda_inf + k_dep + k_hvac + k_hepa)`
///
/// where `k_hvac` and `k_hepa = CADR_total / V_room` require electrical power (`has_power == true`)!
#[inline]
pub fn compute_indoor_shelter_pm25_ug_m3(
    outdoor_pm25_ug_m3: f64,
    archetype: ShelterFacilityArchetype,
    filtration: AirFiltrationStandard,
    portable_hepa_cadr_per_100m2: f64,
    has_electric_power: bool,
) -> f64 {
    let c_out = outdoor_pm25_ug_m3.max(2.0);
    let (sqft, ceil_m, _, base_ach) = archetype.physical_specs();
    let area_m2 = sqft / SQFT_PER_M2;
    let vol_m3 = (area_m2 * ceil_m).max(200.0);

    // Vestibule airlock & weatherstripping tightening under CleanerAir protocol
    let (penetration_p, lambda_inf_hr) = match (archetype, filtration) {
        (
            ShelterFacilityArchetype::OpenParkTentPavilion,
            AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
        ) => {
            if has_electric_power {
                // Positive-pressure sealed disaster medical/shelter tent with HEPA air-handler
                (0.78, 0.95)
            } else {
                (0.96, base_ach)
            }
        }
        (ShelterFacilityArchetype::OpenParkTentPavilion, _) => (0.96, base_ach),
        (_, AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa) => (0.72, base_ach * 0.68),
        (_, AirFiltrationStandard::RecirculatingMerv13Hvac) => (0.80, base_ach * 0.85),
        (_, AirFiltrationStandard::BaselineMerv8Unsealed) => (0.88, base_ach),
    };

    let k_dep_hr = 0.14; // Gravitational & surface deposition loss rate (`hr^-1`)
    let (k_hvac_hr, k_hepa_hr) = if !has_electric_power {
        // Power blackout shuts off central HVAC blowers and portable plug-in HEPA units!
        (0.0, 0.0)
    } else {
        match filtration {
            AirFiltrationStandard::BaselineMerv8Unsealed => {
                // MERV-8 captures ~20% of sub-micron smoke PM2.5 at 2.2 recirculating ACH
                (2.2 * 0.20, 0.0)
            }
            AirFiltrationStandard::RecirculatingMerv13Hvac => {
                // MERV-13 captures ~78% of PM2.5 per pass at 2.8 recirculating ACH
                (2.8 * 0.78, 0.0)
            }
            AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa => {
                let hvac = if archetype == ShelterFacilityArchetype::OpenParkTentPavilion {
                    1.2 * 0.85
                } else {
                    3.0 * 0.82
                };
                let tot_cadr_m3h =
                    (area_m2 / 100.0) * portable_hepa_cadr_per_100m2.clamp(0.0, 2_000.0);
                let hepa = tot_cadr_m3h / vol_m3;
                (hvac, hepa)
            }
        }
    };

    let io_ratio =
        (penetration_p * lambda_inf_hr) / (lambda_inf_hr + k_dep_hr + k_hvac_hr + k_hepa_hr).max(0.05);
    (c_out * io_ratio.clamp(0.03, 1.0)).max(1.5)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateShelterNode {
    pub shelter_id: usize,
    pub name: String,
    pub archetype: ShelterFacilityArchetype,
    pub segment_id: usize,
    pub world_pos: Vec2,
    pub gross_floor_sqft: f64,
    pub nominal_cot_capacity: usize,
    pub seismically_retrofitted: bool,
}

/// Synthesizes 6 geographically distributed candidate municipal emergency shelters across the road network.
pub fn enumerate_candidate_shelters(
    net: &RoadNetwork,
    parcels: &[Parcel],
    cfg: &EmergencySheltersConfig,
) -> Vec<CandidateShelterNode> {
    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(120.0);
    let sy = (b.max_y - b.min_y).max(120.0);

    let specs = [
        (
            "Westbank University Fieldhouse",
            ShelterFacilityArchetype::UniversityFieldhouseArena,
            Vec2 {
                x: b.min_x + 0.18 * sx,
                y: b.min_y + 0.46 * sy,
            },
        ),
        (
            "Northside High School Gymnasium",
            ShelterFacilityArchetype::PublicSchoolGymnasium,
            Vec2 {
                x: b.min_x + 0.42 * sx,
                y: b.min_y + 0.16 * sy,
            },
        ),
        (
            "Central Civic Convention Hall",
            ShelterFacilityArchetype::CivicCommunityCenter,
            Vec2 {
                x: b.min_x + 0.52 * sx,
                y: b.min_y + 0.50 * sy,
            },
        ),
        (
            "South Meadows School Gymnasium",
            ShelterFacilityArchetype::PublicSchoolGymnasium,
            Vec2 {
                x: b.min_x + 0.34 * sx,
                y: b.min_y + 0.84 * sy,
            },
        ),
        (
            "Eastgate Regional Arena Shelter",
            ShelterFacilityArchetype::UniversityFieldhouseArena,
            Vec2 {
                x: b.min_x + 0.84 * sx,
                y: b.min_y + 0.28 * sy,
            },
        ),
        (
            "Greenbelt Seismic Park Tent Pavilion",
            ShelterFacilityArchetype::OpenParkTentPavilion,
            parcels
                .iter()
                .find(|p| p.zone == ZoneCode::PARK)
                .map(|p| p.center)
                .unwrap_or(Vec2 {
                    x: b.min_x + 0.78 * sx,
                    y: b.min_y + 0.76 * sy,
                }),
        ),
    ];

    let open_segs: Vec<usize> = net
        .segments
        .iter()
        .filter(|s| !s.closed && !s.base_pts.is_empty())
        .map(|s| s.id)
        .collect();
    let n_cand = specs.len();
    let retro_n = ((n_cand as f64) * (cfg.shelter_seismic_retrofit_pct.clamp(0.0, 100.0) / 100.0))
        .round() as usize;
    let sqft_pp = cfg.sqft_per_evacuee.clamp(30.0, 110.0);

    let mut out = Vec::with_capacity(n_cand);
    for (idx, (name, arch, target)) in specs.into_iter().enumerate() {
        let best_sid = open_segs
            .iter()
            .copied()
            .filter(|&sid| out.iter().all(|c: &CandidateShelterNode| c.segment_id != sid))
            .min_by(|&a, &bs| {
                let pa = net.segments[a].base_pts[net.segments[a].base_pts.len() / 2];
                let pb = net.segments[bs].base_pts[net.segments[bs].base_pts.len() / 2];
                pa.dist(target)
                    .partial_cmp(&pb.dist(target))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        let wpos = if best_sid < net.segments.len() && !net.segments[best_sid].base_pts.is_empty() {
            net.segments[best_sid].base_pts[net.segments[best_sid].base_pts.len() / 2]
        } else {
            target
        };
        let (sqft, _, _, _) = arch.physical_specs();
        let cap = ((sqft / sqft_pp).floor() as usize).max(50);
        out.push(CandidateShelterNode {
            shelter_id: idx,
            name: name.to_string(),
            archetype: arch,
            segment_id: best_sid,
            world_pos: wpos,
            gross_floor_sqft: sqft,
            nominal_cot_capacity: cap,
            seismically_retrofitted: idx < retro_n
                || arch == ShelterFacilityArchetype::OpenParkTentPavilion,
        });
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShelterHabitabilityRow {
    pub shelter_id: usize,
    pub name: String,
    pub archetype: ShelterFacilityArchetype,
    pub world_pos: Vec2,
    pub selected_open: bool,
    pub seismically_retrofitted: bool,
    pub gross_floor_sqft: f64,
    pub nominal_cot_capacity: usize,
    pub atc20_green_tag_pct: f64,
    pub lifeline_habitable_pct: f64,
    pub mean_outdoor_pm25_ug_m3: f64,
    pub mean_indoor_pm25_ug_m3: f64,
    pub clean_air_ok_pct: f64,
    pub mean_effective_capacity: f64,
    pub expected_assigned_evacuees: f64,
    pub peak_occupancy_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelShelterAssignmentRow {
    pub parcel_id: usize,
    pub zone_code: ZoneCode,
    pub center: Vec2,
    pub displaced_evacuees: f64,
    pub primary_shelter_id: usize,
    pub mean_travel_min: f64,
    pub indoor_pm25_at_assigned_ug_m3: f64,
    pub sheltered_clean_air_pct: f64,
    pub unmet_overflow_evacuees: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShelterFiltrationCurvePoint {
    pub outdoor_pm25_ug_m3: f64,
    pub merv8_indoor_pm25: f64,
    pub merv13_indoor_pm25: f64,
    pub merv13_hepa_indoor_pm25: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencySheltersReport {
    pub config: EmergencySheltersConfig,
    pub candidate_shelters_count: usize,
    pub selected_open_shelters_count: usize,
    pub total_displaced_evacuees: f64,
    pub p90_displaced_evacuees: usize,
    pub total_nominal_cot_capacity: usize,
    pub expected_usable_cot_capacity: f64,
    pub expected_sheltered_in_clean_air: f64,
    pub sheltered_clean_air_pct: f64,
    pub expected_overcrowded_or_unmet_evacuees: f64,
    pub expected_smoke_exceedance_evacuees: f64,
    pub mean_shelter_travel_min: f64,
    pub p90_shelter_travel_min: f64,
    pub mean_assigned_indoor_pm25_ug_m3: f64,
    pub filtration_response_curve: Vec<ShelterFiltrationCurvePoint>,
    pub shelter_rows: Vec<ShelterHabitabilityRow>,
    pub parcel_rows: Vec<ParcelShelterAssignmentRow>,
    pub solve_time_ms: f64,
}

impl EmergencySheltersReport {
    pub fn to_shelter_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.shelter_rows.len() * 140 + 256);
        out.push_str(SHELTER_FACILITY_CSV_HEADER);
        out.push('\n');
        for s in &self.shelter_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{},{:.0},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}\n",
                s.shelter_id,
                s.name.replace('"', "'"),
                s.archetype.short_label(),
                s.selected_open,
                s.gross_floor_sqft,
                s.nominal_cot_capacity,
                s.atc20_green_tag_pct,
                s.lifeline_habitable_pct,
                s.mean_outdoor_pm25_ug_m3,
                s.mean_indoor_pm25_ug_m3,
                s.clean_air_ok_pct,
                s.mean_effective_capacity,
                s.expected_assigned_evacuees,
                s.peak_occupancy_pct
            ));
        }
        out
    }

    pub fn to_parcel_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.parcel_rows.len() * 96 + 256);
        out.push_str(SHELTER_PARCEL_ASSIGN_CSV_HEADER);
        out.push('\n');
        for p in &self.parcel_rows {
            out.push_str(&format!(
                "{},{:?},{:.1},{},{:.2},{:.1},{:.1},{:.2}\n",
                p.parcel_id,
                p.zone_code,
                p.displaced_evacuees,
                p.primary_shelter_id,
                p.mean_travel_min,
                p.indoor_pm25_at_assigned_ug_m3,
                p.sheltered_clean_air_pct,
                p.unmet_overflow_evacuees
            ));
        }
        out
    }

    pub fn export_shelter_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_shelter_csv_string())?;
        Ok(self.shelter_rows.len())
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

/// Executes Two-Stage Post-Earthquake Shelter Siting & `Rayon`-parallelized Monte Carlo evaluation
/// coupling ATC-20 structural green-tag fragility, potable water + sanitary sewer + backup power
/// habitability, EPA/ASHRAE `MERV-13`/`HEPA` indoor smoke filtration, and capacitated evacuee routing.
pub fn run_emergency_shelters_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &EmergencySheltersConfig,
) -> EmergencySheltersReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let n_segs = net.segments.len();
    let n_junc = net.junctions.len();
    let seis = &cfg.evac.medevac.surge.ffe.seismic;
    let ffe = &cfg.evac.medevac.surge.ffe;
    let trials = seis.monte_carlo_trials.clamp(20, 5_000);

    let candidates = enumerate_candidate_shelters(net, parcels, cfg);
    let n_cand = candidates.len();
    let budget_k = cfg.max_open_shelters_budget.clamp(1, n_cand);

    let (_, _, cf_added, rem_inbound, _) =
        plan_contraflow_and_protected_ems_corridors(net, parcels, facilities, &cfg.evac);

    // Stage 1: Select `budget_k` open shelters
    // - Under `HazardAwareCleanAirCapacitated`, score each candidate by capacity * ATC-20 reliability * upwind FFE smoke clearance * spatial coverage!
    // - Otherwise, select largest/central nominal capacity shelters without upwind/seismic screening.
    let wind_rad = ffe.wind_direction_deg.to_radians();
    let wx_hat = wind_rad.cos();
    let wy_hat = wind_rad.sin();
    let b = net.bounds;
    let map_center = Vec2 {
        x: 0.5 * (b.min_x + b.max_x),
        y: 0.5 * (b.min_y + b.max_y),
    };
    let west_x = b.min_x;

    let mut selected_open = vec![false; n_cand];
    let mut ranked_cands: Vec<(usize, f64)> = candidates
        .iter()
        .map(|c| {
            let loc_km = ((c.world_pos.x - west_x).max(0.0)) / 1000.0;
            let sa_med = median_sa10_gmpe_g(
                seis.moment_magnitude_mw,
                (seis.rupture_standoff_km + loc_km).max(1.5),
                seis.vs30_m_per_s,
            );
            let (_, _, base_med_g, _) = c.archetype.physical_specs();
            let eff_med_g = if c.seismically_retrofitted {
                base_med_g * 1.75
            } else {
                base_med_g
            };
            let p_green = 1.0 - normal_cdf_phi((sa_med.ln() - eff_med_g.ln()) / 0.58);

            // Downwind projection from dense downtown core (`> 0` means downwind in smoke plume!)
            let downwind_proj = (c.world_pos.x - map_center.x) * wx_hat
                + (c.world_pos.y - map_center.y) * wy_hat;
            let smoke_safety = if downwind_proj > 60.0 { 0.68 } else { 1.18 };
            let tent_smoke_pen = if c.archetype == ShelterFacilityArchetype::OpenParkTentPavilion
                && cfg.filtration_standard != AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa
            {
                0.55
            } else {
                1.0
            };

            let score = match cfg.siting_routing_policy {
                ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated => {
                    (c.nominal_cot_capacity as f64) * p_green.max(0.15) * smoke_safety * tent_smoke_pen
                }
                _ => {
                    // Conventional peacetime selection favors central / largest halls regardless of fault/wind
                    (c.nominal_cot_capacity as f64)
                        / (1.0 + c.world_pos.dist(map_center) / 1500.0)
                }
            };
            (c.shelter_id, score)
        })
        .collect();
    ranked_cands.sort_by(|a, b_sc| b_sc.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    for &(cid, _) in ranked_cands.iter().take(budget_k) {
        selected_open[cid] = true;
    }

    // Precompute parcel exposure, FFE Hamada spread graph, bridges & water/sewer corridors
    let mut parcel_res = Vec::with_capacity(n_parcels);
    let mut parcel_tfa = Vec::with_capacity(n_parcels);
    let mut parcel_sa10 = Vec::with_capacity(n_parcels);
    let mut parcel_pga = Vec::with_capacity(n_parcels);
    let mut parcel_near_junc = Vec::with_capacity(n_parcels);
    let mut parcel_near_seg = Vec::with_capacity(n_parcels);
    let mut parcel_hamada = Vec::with_capacity(n_parcels);

    let seg_mids: Vec<Vec2> = net
        .segments
        .iter()
        .map(|s| {
            if s.base_pts.is_empty() {
                Vec2 { x: 0.0, y: 0.0 }
            } else {
                s.base_pts[s.base_pts.len() / 2]
            }
        })
        .collect();

    for p in parcels {
        let sp = p.zone.spec();
        let res = if p.zone == ZoneCode::PARK {
            0.0
        } else {
            (p.residents() + 0.18 * p.jobs()).max(18.0)
        };
        let raw_tfa = (p.area_ha * 10_000.0 * sp.far * SQFT_PER_M2) / 1_000_000.0;
        let tfa = if p.zone == ZoneCode::PARK {
            0.001
        } else {
            raw_tfa.max(0.045)
        };
        let loc_km = ((p.center.x - west_x).max(0.0)) / 1000.0;
        let r_km = (seis.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(seis.moment_magnitude_mw, r_km, seis.vs30_m_per_s);
        let pga = (0.86 * sa10 * (seis.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
            .clamp(0.01, 3.0);

        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                p.center
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&p.center.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        let b_sid = (0..n_segs)
            .filter(|&sid| !net.segments[sid].closed && !net.segments[sid].base_pts.is_empty())
            .min_by(|&a, &bs| {
                p.center
                    .dist(seg_mids[a])
                    .partial_cmp(&p.center.dist(seg_mids[bs]))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);

        parcel_res.push(res);
        parcel_tfa.push(tfa);
        parcel_sa10.push(sa10);
        parcel_pga.push(pga);
        parcel_near_junc.push(nj);
        parcel_near_seg.push(b_sid);
        parcel_hamada.push(hamada_hazus_spread_rates(p.zone, ffe.wind_speed_mps));
    }

    let mut shelt_sa10 = Vec::with_capacity(n_cand);
    let mut shelt_near_junc = Vec::with_capacity(n_cand);
    for c in &candidates {
        let loc_km = ((c.world_pos.x - west_x).max(0.0)) / 1000.0;
        let sa10 = median_sa10_gmpe_g(
            seis.moment_magnitude_mw,
            (seis.rupture_standoff_km + loc_km).max(1.5),
            seis.vs30_m_per_s,
        );
        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                c.world_pos
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&c.world_pos.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        shelt_sa10.push(sa10);
        shelt_near_junc.push(nj);
    }

    let horizon_m = ffe.conflagration_horizon_mins.clamp(30.0, 360.0);
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
                14.5 + 0.85 * ffe.wind_speed_mps
            } else {
                10.5 + 0.25 * ffe.wind_speed_mps
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

    let bridges = enumerate_seismic_bridge_inventory(net, facilities, seis);
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
            u_corridors.push((cid, s.length.max(12.0), seg_mids[s.id], s.priority >= 2));
            junc_adj[s.from_junc].push((s.to_junc, cid));
            junc_adj[s.to_junc].push((s.from_junc, cid));
        }
    }
    let n_corr = u_corridors.len();
    let n_rep_corr =
        ((n_corr as f64) * (seis.seismic_pipe_replacement_pct.clamp(0.0, 100.0) / 100.0)).round()
            as usize;
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

    let mut base_rev: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let sig_pen = match net.junctions[seg.to_junc].control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => 5.5,
            ControlMode::Roundabout => 3.0,
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
                    TurnType::Through => 0.4,
                    TurnType::Right => 1.8,
                    TurnType::Left => 2.6,
                };
                if !base_rev[out_sid].iter().any(|&(t, _)| t == seg.id) {
                    base_rev[out_sid].push((seg.id, sig_pen + t_pen));
                }
            }
        }
    }

    let fire_buf_m = cfg.evac.fire_perimeter_road_buffer_m.clamp(20.0, 160.0);
    let mut seg_adj_parcels: Vec<Vec<usize>> = vec![Vec::new(); n_segs];
    for sid in 0..n_segs {
        for (pi, p) in parcels.iter().enumerate() {
            let dx = (seg_mids[sid].x - p.max.x)
                .max(p.min.x - seg_mids[sid].x)
                .max(0.0);
            let dy = (seg_mids[sid].y - p.max.y)
                .max(p.min.y - seg_mids[sid].y)
                .max(0.0);
            if dx.hypot(dy) <= fire_buf_m {
                seg_adj_parcels[sid].push(pi);
            }
        }
    }

    let n_sites = n_parcels + n_br + n_cand;
    let mut site_pts = Vec::with_capacity(n_sites);
    for p in parcels {
        site_pts.push(p.center);
    }
    for br in &bridges {
        site_pts.push(seg_mids[br.segment_ids[0]]);
    }
    for c in &candidates {
        site_pts.push(c.world_pos);
    }
    let range_km = if seis.vs30_m_per_s < 270.0 { 29.5 } else { 25.0 };
    let mut corr_mat = vec![0.0; n_sites * n_sites];
    for i in 0..n_sites {
        for j in 0..n_sites {
            let d_km = site_pts[i].dist(site_pts[j]) / 1000.0;
            corr_mat[i * n_sites + j] = (-3.0 * d_km / range_km).exp();
        }
    }
    let chol = cholesky_lower(&corr_mat, n_sites);

    #[derive(Clone)]
    struct SheltTrialOut {
        displaced_tot: f64,
        usable_cap_tot: f64,
        clean_air_sheltered: f64,
        unmet_or_overcrowded: f64,
        smoke_exceed_evacuees: f64,
        sum_trav_min: f64,
        p90_trav_min: f64,
        sum_in_pm25: f64,
        sh_green: Vec<f64>,
        sh_habitable: Vec<f64>,
        sh_out_pm25: Vec<f64>,
        sh_in_pm25: Vec<f64>,
        sh_clean_ok: Vec<f64>,
        sh_eff_cap: Vec<f64>,
        sh_assigned: Vec<f64>,
        sh_occ_pct: Vec<f64>,
        p_disp: Vec<f64>,
        p_shelt_vote: Vec<usize>,
        p_trav_m: Vec<f64>,
        p_in_pm25: Vec<f64>,
        p_clean_ok: Vec<f64>,
        p_unmet: Vec<f64>,
    }

    let seed_base = seis.random_seed ^ 0x5EE1_7E95_3600_F18A;
    let n_engines = (facilities.len().max(2)) * 2;

    let trial_outs: Vec<SheltTrialOut> = (0..trials)
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
            let tau_b = seis.inter_event_sigma_ln;
            let phi_w = seis.intra_event_sigma_ln;

            // 1. Bridge closures
            let mut br_closed = vec![false; n_segs];
            for (bi, br) in bridges.iter().enumerate() {
                let sa_r = (br.median_site_sa10_g.ln()
                    + tau_b * eta_inter
                    + phi_w * eps_intra[n_parcels + bi])
                    .exp();
                if rng.next_f64_open01() < br.effective_archetype.exceedance_probs(sa_r)[2] {
                    for &sid in &br.segment_ids {
                        if sid < n_segs {
                            br_closed[sid] = true;
                        }
                    }
                }
            }

            // 2. Buried water & sanitary sewer fragility + electric substation state
            let mut water_open = vec![true; n_corr];
            let mut sewer_intact = vec![true; n_corr];
            let mut leak_lps = 0.0;
            for &(cid, len_m, mid, is_art) in &u_corridors {
                let loc_km = ((mid.x - west_x).max(0.0)) / 1000.0;
                let sa_c = (median_sa10_gmpe_g(
                    seis.moment_magnitude_mw,
                    (seis.rupture_standoff_km + loc_km).max(1.5),
                    seis.vs30_m_per_s,
                )
                .ln()
                    + tau_b * eta_inter
                    + phi_w * rng.next_standard_normal())
                .exp();
                let pgv = median_pgv_cm_s_from_sa10(sa_c, seis.vs30_m_per_s).clamp(1.0, 350.0);
                let w_mat = if cid < n_rep_corr {
                    BuriedPipeMaterialClass::SeismicErdipHdpe
                } else {
                    seis.baseline_water_pipe_material
                };
                let s_mat = if cid < n_rep_corr {
                    BuriedPipeMaterialClass::SeismicErdipHdpe
                } else {
                    seis.baseline_sewer_pipe_material
                };
                let rr_w = ala_hazus_pipe_repair_rate_per_km(
                    pgv,
                    seis.pipe_fragility_law,
                    w_mat,
                    if is_art { 300.0 } else { 200.0 },
                    false,
                );
                let rr_s = ala_hazus_pipe_repair_rate_per_km(
                    pgv,
                    seis.pipe_fragility_law,
                    s_mat,
                    250.0,
                    true,
                );
                let nw = sample_poisson_icdf(rr_w * (len_m / 1000.0), rng.next_f64_open01());
                if nw > 0 {
                    if rng.next_f64_open01() < 0.35 {
                        water_open[cid] = false;
                    }
                    leak_lps += (nw as f64) * 11.0;
                }
                let ns = sample_poisson_icdf(rr_s * (len_m / 1000.0), rng.next_f64_open01());
                if ns > 0 && rng.next_f64_open01() < 0.45 {
                    sewer_intact[cid] = false;
                }
            }

            let sub_sa =
                (parcel_sa10.first().copied().unwrap_or(0.35).ln() + tau_b * eta_inter).exp();
            let grid_power_ok =
                rng.next_f64_open01() >= normal_cdf_phi((sub_sa.ln() - (0.58_f64).ln()) / 0.55);
            let pump_head_m = if grid_power_ok || seis.critical_lifeline_backup_power {
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
                    if water_open[cid] && w_dist[v] == usize::MAX {
                        w_dist[v] = w_dist[u] + 1;
                        q.push_back(v);
                    }
                }
            }

            // 3. Coupled FFE ignitions + Hamada conflagration + displaced residential population
            let mut ignitions = Vec::new();
            let mut p_disp = vec![0.0; n_parcels];
            for pi in 0..n_parcels {
                if parcels[pi].zone == ZoneCode::PARK {
                    continue;
                }
                let sa_r = (parcel_sa10[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.8);
                let pga_r = (parcel_pga[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.5);
                // Hazus displaced household fraction (structural extensive/complete + red/yellow tag)
                let p_uninhab = normal_cdf_phi((sa_r.ln() - (0.52_f64).ln()) / 0.62);
                p_disp[pi] = parcel_res[pi] * (0.08 + 0.68 * p_uninhab);

                let ig_rate = scawthorn_hazus_ignition_rate_per_mmsqft(
                    pga_r,
                    parcels[pi].zone,
                    ffe.ignition_model,
                    ffe.seismic_gas_shutoff_pct,
                );
                if sample_poisson_icdf(ig_rate * parcel_tfa[pi], rng.next_f64_open01()) > 0 {
                    ignitions.push(pi);
                }
            }

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

            let cistern_p = (ffe.auxiliary_cistern_coverage_pct.clamp(0.0, 100.0)) / 100.0;
            let mut rem_eng = n_engines;
            let mut escaped_seeds = Vec::new();
            for &pi in &ignitions {
                let has_w = junc_psi[parcel_near_junc[pi]] >= MIN_FIRE_FLOW_RESIDUAL_PSI
                    || rng.next_f64_open01() < cistern_p;
                if rem_eng > 0 && has_w {
                    rem_eng -= 1;
                } else {
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

            // Burned parcels displace 100% of remaining residents!
            for pi in 0..n_parcels {
                if burn_time_min[pi].is_finite() {
                    p_disp[pi] = parcel_res[pi];
                }
            }
            let displaced_tot: f64 = p_disp.iter().sum();

            // Gaussian downwind smoke PM2.5 (`ug/m^3`) at t = 120 min
            let u_adv = ffe.wind_speed_mps.max(1.4);
            let smoke_pm25_at = |pt: Vec2| -> f64 {
                let mut pm = 12.0; // ambient background particulate baseline (`ug/m^3`)
                for qi in 0..n_parcels {
                    if !burn_time_min[qi].is_finite() || burn_time_min[qi] > 120.0 {
                        continue;
                    }
                    let age = (120.0 - burn_time_min[qi]).max(0.0);
                    let inten = (-age / 160.0).exp().max(0.30);
                    let dx = pt.x - parcels[qi].center.x;
                    let dy = pt.y - parcels[qi].center.y;
                    let x_down = dx * wx_hat + dy * wy_hat;
                    let y_cross = (-dx * wy_hat + dy * wx_hat).abs();
                    if x_down >= -45.0 {
                        let sig_y = 55.0 + 0.24 * x_down.max(25.0);
                        pm += (parcel_tfa[qi] * 48_000.0 * inten / (u_adv * sig_y))
                            * (-0.5 * (y_cross / sig_y).powi(2)).exp();
                    }
                }
                pm
            };

            // 4. Evaluate each candidate shelter's ATC-20 green-tag, power/water/sewer habitability & HEPA indoor PM2.5!
            let mut sh_green = vec![0.0; n_cand];
            let mut sh_habitable = vec![0.0; n_cand];
            let mut sh_out_pm25 = vec![12.0; n_cand];
            let mut sh_in_pm25 = vec![8.0; n_cand];
            let mut sh_clean_ok = vec![0.0; n_cand];
            let mut sh_eff_cap = vec![0.0; n_cand];

            for (ci, c) in candidates.iter().enumerate() {
                let sa_s = (shelt_sa10[ci].ln()
                    + tau_b * eta_inter
                    + phi_w * eps_intra[n_parcels + n_br + ci])
                    .exp();
                let (_, _, base_med_g, _) = c.archetype.physical_specs();
                let eff_med_g = if c.seismically_retrofitted {
                    base_med_g * 1.75
                } else {
                    base_med_g
                };
                let p_exceed_green = normal_cdf_phi((sa_s.ln() - eff_med_g.ln()) / 0.56);
                let green_tagged = rng.next_f64_open01() >= p_exceed_green;
                if green_tagged {
                    sh_green[ci] = 100.0;
                }

                let sj = shelt_near_junc[ci];
                let water_ok = junc_psi[sj] >= 15.0 || cfg.shelter_backup_power_water_sanitation;
                let local_sewer_ok = junc_adj[sj]
                    .iter()
                    .all(|&(_, corr_id)| sewer_intact[corr_id])
                    || cfg.shelter_backup_power_water_sanitation;
                let power_ok = grid_power_ok
                    || cfg.shelter_backup_power_water_sanitation
                    || seis.critical_lifeline_backup_power;

                let out_pm = smoke_pm25_at(c.world_pos);
                let in_pm = compute_indoor_shelter_pm25_ug_m3(
                    out_pm,
                    c.archetype,
                    cfg.filtration_standard,
                    cfg.portable_hepa_cadr_m3h_per_100m2,
                    power_ok,
                );
                sh_out_pm25[ci] = out_pm;
                sh_in_pm25[ci] = in_pm;
                if in_pm <= cfg.indoor_pm25_health_limit_ug_m3 {
                    sh_clean_ok[ci] = 100.0;
                }

                // Lifeline habitability multiplier:
                // - Red/Yellow structural tag -> 0 capacity (closed!)
                // - Broken water or backed-up sanitary sewer without backup bladders/latrines cuts usable capacity to 38%!
                if selected_open[ci] && green_tagged {
                    let life_mult = match (water_ok, local_sewer_ok, power_ok) {
                        (true, true, true) => 1.00,
                        (true, true, false) => 0.78,
                        (true, false, _) | (false, true, _) => 0.45,
                        (false, false, _) => 0.25,
                    };
                    if water_ok && local_sewer_ok {
                        sh_habitable[ci] = 100.0;
                    }
                    sh_eff_cap[ci] = (c.nominal_cot_capacity as f64) * life_mult;
                }
            }

            // 5. Compute post-quake + FFE thermally closed + smoke-slowed + contraflow road travel times into each shelter
            let mut seg_trav_min = vec![f64::INFINITY; n_segs];
            for sid in 0..n_segs {
                let eff_lanes = rem_inbound[sid] + cf_added[sid];
                if net.segments[sid].closed || br_closed[sid] || eff_lanes == 0 {
                    continue;
                }
                if seg_adj_parcels[sid]
                    .iter()
                    .any(|&pi| burn_time_min[pi] <= 90.0)
                {
                    continue;
                }
                let out_pm = smoke_pm25_at(seg_mids[sid]);
                let vis_sm = (3.912 / (3.912 / (4.5 * 1.609344) + 0.0042 * out_pm)) / 1.609344;
                let (v_fac, _) = smoke_visibility_speed_and_capacity_factors(vis_sm);
                let cf_speedup = if cf_added[sid] > 0 { 1.15 } else { 1.0 };
                let eff_v = (net.segments[sid].speed_mps * v_fac * cf_speedup).clamp(2.5, 30.0);
                seg_trav_min[sid] = (net.segments[sid].length / eff_v) / 60.0;
            }

            let mut shelt_parcel_min = vec![vec![42.0; n_parcels]; n_cand];
            for (ci, c) in candidates.iter().enumerate() {
                if !selected_open[ci] {
                    continue;
                }
                let mut d_min = vec![f64::INFINITY; n_segs];
                let mut heap = BinaryHeap::new();
                if c.segment_id < n_segs && seg_trav_min[c.segment_id].is_finite() {
                    d_min[c.segment_id] = seg_trav_min[c.segment_id];
                    heap.push(DijkNode {
                        cost: d_min[c.segment_id],
                        id: c.segment_id,
                    });
                }
                while let Some(DijkNode { cost, id }) = heap.pop() {
                    if cost > d_min[id] {
                        continue;
                    }
                    for &(prev_sid, turn_s) in &base_rev[id] {
                        if !seg_trav_min[prev_sid].is_finite() {
                            continue;
                        }
                        let nc = cost + (turn_s / 60.0) + seg_trav_min[prev_sid];
                        if nc < d_min[prev_sid] {
                            d_min[prev_sid] = nc;
                            heap.push(DijkNode {
                                cost: nc,
                                id: prev_sid,
                            });
                        }
                    }
                }
                for pi in 0..n_parcels {
                    let sid = parcel_near_seg[pi];
                    shelt_parcel_min[ci][pi] = if d_min[sid].is_finite() {
                        d_min[sid] + (parcels[pi].center.dist(seg_mids[sid]) / 280.0)
                    } else {
                        (parcels[pi].center.dist(c.world_pos) / 140.0) + 12.0
                    };
                }
            }

            // 6. Stage 2: Evacuee Shelter-Assignment Routing across parcels (urgent fire-front blocks assigned first)
            let mut parcel_order: Vec<usize> = (0..n_parcels)
                .filter(|&pi| p_disp[pi] > 1e-6)
                .collect();
            parcel_order.sort_by(|&a, &bp| {
                burn_time_min[a]
                    .partial_cmp(&burn_time_min[bp])
                    .unwrap_or(Ordering::Equal)
            });

            let mut rem_cap = sh_eff_cap.clone();
            let mut sh_assigned = vec![0.0; n_cand];
            let mut p_shelt_vote = vec![0usize; n_parcels];
            let mut p_trav_m = vec![0.0; n_parcels];
            let mut p_in_pm25 = vec![10.0; n_parcels];
            let mut p_clean_ok = vec![100.0; n_parcels];
            let mut p_unmet = vec![0.0; n_parcels];

            let mut clean_air_sheltered = 0.0;
            let mut unmet_or_overcrowded = 0.0;
            let mut smoke_exceed_evacuees = 0.0;
            let mut sum_trav_min = 0.0;
            let mut sum_in_pm25 = 0.0;
            let mut parcel_trav_samples: Vec<(f64, f64)> = Vec::with_capacity(parcel_order.len());

            let default_open_id = (0..n_cand).find(|&ci| selected_open[ci]).unwrap_or(0);

            for &pi in &parcel_order {
                let demand = p_disp[pi];
                match cfg.siting_routing_policy {
                    ShelterSitingAndRoutingPolicy::UncoordinatedNearestShelter => {
                        // Self-evacuees drive blindly to the closest selected shelter even if full or smoky!
                        let chosen = (0..n_cand)
                            .filter(|&ci| selected_open[ci] && sh_eff_cap[ci] > 1.0)
                            .min_by(|&a, &bc| {
                                shelt_parcel_min[a][pi]
                                    .partial_cmp(&shelt_parcel_min[bc][pi])
                                    .unwrap_or(Ordering::Equal)
                            })
                            .unwrap_or(default_open_id);

                        let fit = demand.min(rem_cap[chosen].max(0.0));
                        let ovf = (demand - fit).max(0.0);
                        rem_cap[chosen] = (rem_cap[chosen] - fit).max(0.0);
                        sh_assigned[chosen] += demand;
                        unmet_or_overcrowded += ovf;
                        p_unmet[pi] = ovf;

                        let tm = shelt_parcel_min[chosen][pi] + if ovf > 0.1 { 14.0 } else { 0.0 };
                        let ipm = sh_in_pm25[chosen];
                        p_shelt_vote[pi] = chosen;
                        p_trav_m[pi] = tm;
                        p_in_pm25[pi] = ipm;
                        sum_trav_min += demand * tm;
                        sum_in_pm25 += demand * ipm;
                        parcel_trav_samples.push((tm, demand));

                        if fit > 0.0 && ipm <= cfg.indoor_pm25_health_limit_ug_m3 {
                            clean_air_sheltered += fit;
                            p_clean_ok[pi] = (fit / demand) * 100.0;
                        } else {
                            p_clean_ok[pi] = 0.0;
                        }
                        if ipm > cfg.indoor_pm25_health_limit_ug_m3 {
                            smoke_exceed_evacuees += demand;
                        }
                    }
                    ShelterSitingAndRoutingPolicy::CapacitatedDistanceOnly
                    | ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated => {
                        // Split/route parcel demand across open shelters with remaining capacity
                        let smoke_aware = cfg.siting_routing_policy
                            == ShelterSitingAndRoutingPolicy::HazardAwareCleanAirCapacitated;
                        let mut rem_dem = demand;
                        let mut best_sh = default_open_id;
                        let mut best_chunk = 0.0;
                        let mut acc_tm = 0.0;
                        let mut acc_pm = 0.0;
                        let mut acc_clean = 0.0;

                        while rem_dem > 1e-6 {
                            let cand_opt = (0..n_cand)
                                .filter(|&ci| selected_open[ci] && rem_cap[ci] > 0.5)
                                .min_by(|&a, &bc| {
                                    let cost_fn = |ci: usize| -> f64 {
                                        let t_c = shelt_parcel_min[ci][pi];
                                        let smoke_pen = if smoke_aware {
                                            let ex = (sh_in_pm25[ci]
                                                - cfg.indoor_pm25_health_limit_ug_m3)
                                                .max(0.0);
                                            ex * 0.85 + sh_in_pm25[ci] * 0.12
                                        } else {
                                            0.0
                                        };
                                        t_c + smoke_pen
                                    };
                                    cost_fn(a)
                                        .partial_cmp(&cost_fn(bc))
                                        .unwrap_or(Ordering::Equal)
                                });

                            if let Some(ci) = cand_opt {
                                let take = rem_dem.min(rem_cap[ci]);
                                rem_cap[ci] = (rem_cap[ci] - take).max(0.0);
                                sh_assigned[ci] += take;
                                if take > best_chunk {
                                    best_chunk = take;
                                    best_sh = ci;
                                }
                                let tm = shelt_parcel_min[ci][pi];
                                let ipm = sh_in_pm25[ci];
                                acc_tm += take * tm;
                                acc_pm += take * ipm;
                                if ipm <= cfg.indoor_pm25_health_limit_ug_m3 {
                                    clean_air_sheltered += take;
                                    acc_clean += take;
                                } else {
                                    smoke_exceed_evacuees += take;
                                }
                                rem_dem -= take;
                            } else {
                                // All open municipal shelters at 100% cot capacity -> overflow!
                                unmet_or_overcrowded += rem_dem;
                                p_unmet[pi] += rem_dem;
                                sh_assigned[best_sh] += rem_dem;
                                let tm = shelt_parcel_min[best_sh][pi] + 18.0;
                                let ipm = sh_in_pm25[best_sh];
                                acc_tm += rem_dem * tm;
                                acc_pm += rem_dem * ipm;
                                if ipm > cfg.indoor_pm25_health_limit_ug_m3 {
                                    smoke_exceed_evacuees += rem_dem;
                                }
                                break;
                            }
                        }

                        p_shelt_vote[pi] = best_sh;
                        p_trav_m[pi] = acc_tm / demand;
                        p_in_pm25[pi] = acc_pm / demand;
                        p_clean_ok[pi] = (acc_clean / demand) * 100.0;
                        sum_trav_min += acc_tm;
                        sum_in_pm25 += acc_pm;
                        parcel_trav_samples.push((p_trav_m[pi], demand));
                    }
                }
            }

            parcel_trav_samples.sort_by(|a, bp| a.0.partial_cmp(&bp.0).unwrap_or(Ordering::Equal));
            let mut p90_trav_min = 0.0;
            let mut cum_w = 0.0;
            for &(tm, w) in &parcel_trav_samples {
                cum_w += w;
                p90_trav_min = tm;
                if cum_w >= 0.90 * displaced_tot {
                    break;
                }
            }

            let sh_occ_pct: Vec<f64> = (0..n_cand)
                .map(|ci| {
                    if sh_eff_cap[ci] > 1.0 {
                        (sh_assigned[ci] / sh_eff_cap[ci]) * 100.0
                    } else if sh_assigned[ci] > 0.1 {
                        200.0
                    } else {
                        0.0
                    }
                })
                .collect();

            SheltTrialOut {
                displaced_tot,
                usable_cap_tot: sh_eff_cap.iter().sum(),
                clean_air_sheltered,
                unmet_or_overcrowded,
                smoke_exceed_evacuees,
                sum_trav_min,
                p90_trav_min,
                sum_in_pm25,
                sh_green,
                sh_habitable,
                sh_out_pm25,
                sh_in_pm25,
                sh_clean_ok,
                sh_eff_cap,
                sh_assigned,
                sh_occ_pct,
                p_disp,
                p_shelt_vote,
                p_trav_m,
                p_in_pm25,
                p_clean_ok,
                p_unmet,
            }
        })
        .collect();

    let tf = trials as f64;
    let mut disp_sorted: Vec<usize> = trial_outs
        .iter()
        .map(|t| t.displaced_tot.round() as usize)
        .collect();
    disp_sorted.sort_unstable();
    let p90_idx = (((trials - 1) as f64) * 0.90).round() as usize;

    let tot_disp_all: f64 = trial_outs.iter().map(|t| t.displaced_tot).sum();
    let exp_disp = tot_disp_all / tf;
    let exp_usable = trial_outs.iter().map(|t| t.usable_cap_tot).sum::<f64>() / tf;
    let exp_clean = trial_outs.iter().map(|t| t.clean_air_sheltered).sum::<f64>() / tf;
    let exp_unmet = trial_outs.iter().map(|t| t.unmet_or_overcrowded).sum::<f64>() / tf;
    let exp_smk_ex = trial_outs.iter().map(|t| t.smoke_exceed_evacuees).sum::<f64>() / tf;

    let mean_trav = if tot_disp_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_trav_min).sum::<f64>() / tot_disp_all
    } else {
        8.0
    };
    let mean_in_pm = if tot_disp_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_in_pm25).sum::<f64>() / tot_disp_all
    } else {
        8.0
    };
    let p90_trav = trial_outs.iter().map(|t| t.p90_trav_min).sum::<f64>() / tf;

    let mut shelter_rows = Vec::with_capacity(n_cand);
    let mut nom_cap_open = 0usize;
    for (ci, c) in candidates.iter().enumerate() {
        if selected_open[ci] {
            nom_cap_open += c.nominal_cot_capacity;
        }
        shelter_rows.push(ShelterHabitabilityRow {
            shelter_id: ci,
            name: c.name.clone(),
            archetype: c.archetype,
            world_pos: c.world_pos,
            selected_open: selected_open[ci],
            seismically_retrofitted: c.seismically_retrofitted,
            gross_floor_sqft: c.gross_floor_sqft,
            nominal_cot_capacity: c.nominal_cot_capacity,
            atc20_green_tag_pct: trial_outs.iter().map(|t| t.sh_green[ci]).sum::<f64>() / tf,
            lifeline_habitable_pct: trial_outs.iter().map(|t| t.sh_habitable[ci]).sum::<f64>() / tf,
            mean_outdoor_pm25_ug_m3: trial_outs.iter().map(|t| t.sh_out_pm25[ci]).sum::<f64>() / tf,
            mean_indoor_pm25_ug_m3: trial_outs.iter().map(|t| t.sh_in_pm25[ci]).sum::<f64>() / tf,
            clean_air_ok_pct: trial_outs.iter().map(|t| t.sh_clean_ok[ci]).sum::<f64>() / tf,
            mean_effective_capacity: trial_outs.iter().map(|t| t.sh_eff_cap[ci]).sum::<f64>() / tf,
            expected_assigned_evacuees: trial_outs.iter().map(|t| t.sh_assigned[ci]).sum::<f64>()
                / tf,
            peak_occupancy_pct: trial_outs.iter().map(|t| t.sh_occ_pct[ci]).sum::<f64>() / tf,
        });
    }

    let mut parcel_rows = Vec::with_capacity(n_parcels);
    for (pi, p) in parcels.iter().enumerate() {
        let mut votes = vec![0usize; n_cand];
        for t in &trial_outs {
            votes[t.p_shelt_vote[pi]] += 1;
        }
        let modal_sh = votes
            .iter()
            .enumerate()
            .max_by_key(|&(_, v)| v)
            .map(|(ci, _)| ci)
            .unwrap_or(0);

        parcel_rows.push(ParcelShelterAssignmentRow {
            parcel_id: pi,
            zone_code: p.zone,
            center: p.center,
            displaced_evacuees: trial_outs.iter().map(|t| t.p_disp[pi]).sum::<f64>() / tf,
            primary_shelter_id: modal_sh,
            mean_travel_min: trial_outs.iter().map(|t| t.p_trav_m[pi]).sum::<f64>() / tf,
            indoor_pm25_at_assigned_ug_m3: trial_outs.iter().map(|t| t.p_in_pm25[pi]).sum::<f64>()
                / tf,
            sheltered_clean_air_pct: trial_outs.iter().map(|t| t.p_clean_ok[pi]).sum::<f64>() / tf,
            unmet_overflow_evacuees: trial_outs.iter().map(|t| t.p_unmet[pi]).sum::<f64>() / tf,
        });
    }

    // Sample indoor vs. outdoor PM2.5 response curve (`0 .. 500 ug/m^3`) for `egui` visualization
    let mut filtration_response_curve = Vec::with_capacity(26);
    for k in 0..=25 {
        let cout = (k as f64) * 20.0;
        filtration_response_curve.push(ShelterFiltrationCurvePoint {
            outdoor_pm25_ug_m3: cout,
            merv8_indoor_pm25: compute_indoor_shelter_pm25_ug_m3(
                cout,
                ShelterFacilityArchetype::PublicSchoolGymnasium,
                AirFiltrationStandard::BaselineMerv8Unsealed,
                0.0,
                true,
            ),
            merv13_indoor_pm25: compute_indoor_shelter_pm25_ug_m3(
                cout,
                ShelterFacilityArchetype::PublicSchoolGymnasium,
                AirFiltrationStandard::RecirculatingMerv13Hvac,
                0.0,
                true,
            ),
            merv13_hepa_indoor_pm25: compute_indoor_shelter_pm25_ug_m3(
                cout,
                ShelterFacilityArchetype::PublicSchoolGymnasium,
                AirFiltrationStandard::CleanerAirMerv13PlusPortableHepa,
                cfg.portable_hepa_cadr_m3h_per_100m2,
                true,
            ),
        });
    }

    EmergencySheltersReport {
        config: cfg.clone(),
        candidate_shelters_count: n_cand,
        selected_open_shelters_count: budget_k,
        total_displaced_evacuees: exp_disp,
        p90_displaced_evacuees: disp_sorted[p90_idx],
        total_nominal_cot_capacity: nom_cap_open,
        expected_usable_cot_capacity: exp_usable,
        expected_sheltered_in_clean_air: exp_clean,
        sheltered_clean_air_pct: if exp_disp > 1e-9 {
            ((exp_clean / exp_disp) * 100.0).clamp(0.0, 100.0)
        } else {
            100.0
        },
        expected_overcrowded_or_unmet_evacuees: exp_unmet,
        expected_smoke_exceedance_evacuees: exp_smk_ex,
        mean_shelter_travel_min: mean_trav,
        p90_shelter_travel_min: p90_trav,
        mean_assigned_indoor_pm25_ug_m3: mean_in_pm,
        filtration_response_curve,
        shelter_rows,
        parcel_rows,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

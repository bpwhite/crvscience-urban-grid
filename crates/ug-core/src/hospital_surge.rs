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

//! Post-Earthquake Hospital Emergency-Department (`ED`) Surge Capacity, Coupled Seismic Trauma +
//! Fire-Following-Earthquake (`FFE`) Burn/Smoke Priority Triage Queueing, and Dynamic Ambulance
//! Hospital-Diversion Routing Engine
//!
//! Grounded in:
//! - FEMA P-58 & Hazus Earthquake Fault-Tree Hospital Functionality & Discrete-Event ED Surge
//!   Bottlenecks (*Earthquake Spectra* 39(1):402-433,
//!   `https://www.researchgate.net/publication/365475413_Evaluating_post-earthquake_functionality_and_surge_capacity_of_hospital_emergency_departments_using_discrete_event_simulation`).
//! - Ceferino et al. / Merino-Pena et al. (2024), Multi-Severity Post-Earthquake Hospital Resource
//!   Demand (*Earthquake Spectra* 40(4):2402-2429,
//!   `https://escholarship.org/content/qt75w3g2g2/qt75w3g2g2.pdf`).
//! - Kearns et al. (2020), *Actionable, Revised (v.3), and Amplified American Burn Association
//!   Triage Tables for Mass Casualties* (`https://pmc.ncbi.nlm.nih.gov/articles/PMC7530540/`) and
//!   Western Region Burn Disaster Consortium (*WRBDC State/Regional Mass Burn Casualty Plan*).
//! - Spatial-Network Load Balancing & Dynamic Ambulance Diversion / Interhospital Transfer
//!   (*Disaster Medicine and Public Health Preparedness*,
//!   `https://www.cambridge.org/core/journals/disaster-medicine-and-public-health-preparedness/article/spatialnetwork-framework-for-load-balancing-and-patient-transfer-management-during-emergencies/E0CC2E9FFEF04BE4803CA82D1DE89D0D`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::ffe_conflagration::{
    hamada_hazus_spread_rates, scawthorn_hazus_ignition_rate_per_mmsqft, FfeConfig,
};
use crate::network::{sample_polyline_fast, ControlMode, RoadNetwork, TurnType, Vec2, LANE_WIDTH_M};
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

pub const HOSPITAL_SURGE_FACILITY_CSV_HEADER: &str = "hospital_id,name,designation,retrofitted,baseline_ed_bays,mean_functional_ed_bays,mean_functional_burn_beds,mean_functionality_pct,water_20psi_ok_pct,ambulance_arrivals,walkin_arrivals,diverted_out_ambulances,peak_ed_utilization_pct,mean_red_burn_wait_min,mean_yellow_wait_min,time_on_diversion_pct";
pub const HOSPITAL_SURGE_PARCEL_CSV_HEADER: &str = "parcel_id,zone_code,expected_red_trauma,expected_burn_critical,expected_yellow_delayed,expected_green_walkin,primary_hospital_id,diversion_prob_pct,mean_red_burn_treat_min,golden_hour_pct";

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

/// Clinical trauma & ABA burn capability tier of an emergency receiving facility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HospitalTraumaDesignation {
    /// Regional Level-I Trauma & Verified ABA Burn Center (`36` ED bays, `8` Trauma ORs, `16` Burn ICU beds).
    Level1TraumaAndBurnCenter,
    /// Level-II Community Trauma Hospital (`22` ED bays, `4` Trauma ORs, `2` burn stabilization bays).
    Level2CommunityTrauma,
    /// Level-III Satellite Acute Receiving Hospital (`14` ED bays, `2` procedure rooms, `0` Burn ICU beds).
    Level3AcuteReceiving,
    /// Rapidly deployed tented Mobile Field Hospital / Alternate Care Site (`20` delayed/minor cots, base-isolated).
    MobileFieldHospital,
}

impl HospitalTraumaDesignation {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Level1TraumaAndBurnCenter => "Lvl-I Trauma + Burn Ctr",
            Self::Level2CommunityTrauma => "Lvl-II Community Trauma",
            Self::Level3AcuteReceiving => "Lvl-III Satellite ED",
            Self::MobileFieldHospital => "Mobile Field Hospital",
        }
    }

    pub fn badge_prefix(self) -> &'static str {
        match self {
            Self::Level1TraumaAndBurnCenter => "L1-BURN",
            Self::Level2CommunityTrauma => "L2-TRM",
            Self::Level3AcuteReceiving => "L3-ED",
            Self::MobileFieldHospital => "MFH",
        }
    }

    pub fn baseline_capacities(self) -> (f64, f64, f64) {
        // Returns `(staffed_ed_bays, trauma_or_rooms, aba_burn_icu_beds)`
        match self {
            Self::Level1TraumaAndBurnCenter => (36.0, 8.0, 16.0),
            Self::Level2CommunityTrauma => (22.0, 4.0, 2.0),
            Self::Level3AcuteReceiving => (14.0, 2.0, 0.0),
            Self::MobileFieldHospital => (20.0, 1.0, 0.0),
        }
    }
}

/// START Mass-Casualty & American Burn Association (`ABA`) disaster triage acuity tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CasualtyTriageAcuity {
    /// Hazus Severity 3 / START Red Immediate life-threatening hemorrhage, crush, or TBI (`Golden Hour <= 60 min`).
    RedImmediateTrauma,
    /// Coupled FFE severe thermal burn (`>20% TBSA`) or inhalation airway injury requiring ABA Burn ICU/surge care.
    BurnCriticalSpecialty,
    /// Hazus Severity 2 / START Yellow Delayed major fracture, soft-tissue, or moderate (`10-20% TBSA`) burn.
    YellowDelayedTrauma,
    /// Hazus Severity 1 / START Green Minor "walking wounded" self-evacuating to nearby EDs or Field Hospitals.
    GreenWalkingWounded,
}

/// EMS & Medical Operations Coordination Center (`MOCC`) ambulance hospital-selection and diversion rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbulanceDiversionPolicy {
    /// Myopic baseline: always transport to the geographically closest hospital regardless of ED queue or Burn ICU availability.
    ClosestHospitalNoDiversion,
    /// Threshold Red-Alert Diversion: bypass hospitals exceeding `diversion_occupancy_threshold_pct` ED saturation to next closest.
    ThresholdSaturationDiversion,
    /// System-Optimal MOCC Routing (Dean 2008 / Ceferino): minimize acuity-weighted `Travel + Priority ED Queue Wait + Burn Specialty Penalty`.
    DynamicWaitPlusTravelTriage,
}

impl Default for AmbulanceDiversionPolicy {
    fn default() -> Self {
        Self::DynamicWaitPlusTravelTriage
    }
}

impl AmbulanceDiversionPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::ClosestHospitalNoDiversion => "Closest Hospital (No Diversion)",
            Self::ThresholdSaturationDiversion => "Threshold Red-Alert Diversion (>=90%)",
            Self::DynamicWaitPlusTravelTriage => "Dynamic MOCC Wait+Travel + Burn Triage",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ClosestHospitalNoDiversion => "Closest-Only",
            Self::ThresholdSaturationDiversion => "90%-Divert",
            Self::DynamicWaitPlusTravelTriage => "Dynamic-MOCC",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HospitalSurgeConfig {
    /// Coupled seismic ground-motion, ALA/Hazus pipeline fragility, and Scawthorn/Hamada FFE parameters.
    pub ffe: FfeConfig,
    pub diversion_policy: AmbulanceDiversionPolicy,
    /// Acute post-earthquake emergency-department surge horizon (`hours`, default `6.0 h`).
    pub surge_window_hours: f64,
    /// Occupancy / saturation trigger (`50..=150%`) for `ThresholdSaturationDiversion` & diversion status telemetry.
    pub diversion_occupancy_threshold_pct: f64,
    /// ALS/BLS transport ambulances staged per municipal Fire/EMS station (default `3`).
    pub ambulances_per_station: usize,
    /// Minimum regional receiving hospital count synthesized if fewer `Hospital` pins exist on map (default `3`).
    pub min_receiving_hospitals: usize,
    /// Fraction (`0..=100%`) of hospitals upgraded to OSHPD `SPC-5` / base-isolated seismic structural + nonstructural bracing.
    pub hospital_seismic_retrofit_pct: f64,
    /// Enables 72-hour on-site hospital emergency diesel generators & pressurized seismic water storage tanks.
    pub hospital_onsite_backup_lifelines: bool,
    /// Activates American Burn Association (`ABA`) `+50%` Crisis/Contingency Burn Surge bed expansion & tele-burn triage.
    pub aba_burn_surge_expansion: bool,
    /// Deploys Disaster Medical Assistance Team (`DMAT`) Mobile Field Hospitals to intercept Green/low-Yellow convergence.
    pub deploy_mobile_field_hospitals: bool,
}

impl Default for HospitalSurgeConfig {
    fn default() -> Self {
        Self {
            ffe: FfeConfig::default(),
            diversion_policy: AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage,
            surge_window_hours: 6.0,
            diversion_occupancy_threshold_pct: 90.0,
            ambulances_per_station: 3,
            min_receiving_hospitals: 3,
            hospital_seismic_retrofit_pct: 35.0,
            hospital_onsite_backup_lifelines: false,
            aba_burn_surge_expansion: false,
            deploy_mobile_field_hospitals: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HospitalSurgeNode {
    pub hospital_id: usize,
    pub name: String,
    pub designation: HospitalTraumaDesignation,
    pub segment_id: usize,
    pub s_m: f64,
    pub world_pos: Vec2,
    pub seismically_retrofitted: bool,
    pub baseline_ed_bays: f64,
    pub baseline_or_rooms: f64,
    pub baseline_burn_beds: f64,
}

/// Evaluates FEMA P-58 / Hazus fault-tree post-earthquake hospital operational capacity fraction (`0.05..=1.0`)
/// combining structural/nonstructural `Sa(1.0s)` + `PGA` damage with electric power and `>= 20 psi` water supply.
pub fn hospital_fault_tree_functionality(
    designation: HospitalTraumaDesignation,
    seismically_retrofitted: bool,
    site_sa10_g: f64,
    site_pga_g: f64,
    std_norm_draw: f64,
    grid_power_ok: bool,
    municipal_water_psi: f64,
    onsite_backup_lifelines: bool,
) -> f64 {
    if designation == HospitalTraumaDesignation::MobileFieldHospital {
        // Rapidly deployed tented field hospital with independent generator & bladder water
        return 0.96;
    }

    // Effective demand IM blending spectral drift (`Sa 1.0s`) and nonstructural acceleration (`PGA`)
    let eff_im_g = (0.55 * site_sa10_g + 0.45 * site_pga_g).max(0.005);
    let (medians, beta): ([f64; 4], f64) = if seismically_retrofitted {
        // OSHPD SPC-5 / NPC-5 base-isolated & braced medical gas/sprinkler hospital
        ([0.82, 1.24, 1.72, 2.38], 0.55)
    } else {
        // Conventional / legacy unbraced nonstructural ceiling, chiller & medical-gas piping
        ([0.28, 0.46, 0.72, 1.08], 0.58)
    };

    let u01 = normal_cdf_phi(std_norm_draw);
    let ln_im = eff_im_g.ln();
    let mut ex = [0.0; 4];
    for (k, &med) in medians.iter().enumerate() {
        ex[k] = normal_cdf_phi((ln_im - med.ln()) / beta).clamp(0.0, 1.0);
    }
    for k in 1..4 {
        if ex[k] > ex[k - 1] {
            ex[k] = ex[k - 1];
        }
    }
    let struct_func = if u01 < ex[3] {
        0.08 // Complete: structural red-tag, parking-lot triage only
    } else if u01 < ex[2] {
        0.26 // Extensive: major nonstructural pipe/ceiling rupture
    } else if u01 < ex[1] {
        0.62 // Moderate: partial ward & imaging outage
    } else if u01 < ex[0] {
        0.88 // Slight: minor equipment recalibration
    } else {
        1.00
    };

    // Lifeline fault-tree product (power for CT/OR/ventilators; >=20 psi water for sterilization/chillers/burn care)
    let power_factor: f64 = if grid_power_ok {
        1.0
    } else if onsite_backup_lifelines {
        0.96
    } else {
        0.55
    };
    let water_ok = municipal_water_psi >= MIN_FIRE_FLOW_RESIDUAL_PSI;
    let water_factor: f64 = if water_ok {
        1.0
    } else if onsite_backup_lifelines {
        0.95
    } else {
        0.60
    };

    (struct_func * power_factor * water_factor).clamp(0.06, 1.0)
}

/// Computes expected FEMA Hazus Ch. 13 indoor seismic trauma casualty rates `(red_frac, yellow_frac, green_frac)`
/// per exposed occupant as a function of site `Sa(1.0s)` and `ZoneCode` structural stock.
#[inline]
pub fn hazus_parcel_casualty_rates(sa10_g: f64, zone: ZoneCode) -> (f64, f64, f64) {
    if zone == ZoneCode::PARK || sa10_g < 0.06 {
        return (0.0, 0.0, 0.0);
    }
    // Structural vulnerability factor by occupancy building stock (URM/older concrete in CBD/IND vs wood-frame R1)
    let stock_vuln = match zone {
        ZoneCode::CBD => 1.32,
        ZoneCode::IND => 1.24,
        ZoneCode::MU4 => 1.12,
        ZoneCode::R3 => 0.92,
        ZoneCode::R1 => 0.62,
        ZoneCode::PARK => 0.0,
    };
    let p_mod = normal_cdf_phi((sa10_g.ln() - (0.42_f64).ln()) / 0.60);
    let p_ext = normal_cdf_phi((sa10_g.ln() - (0.76_f64).ln()) / 0.60);
    let p_cmp = normal_cdf_phi((sa10_g.ln() - (1.18_f64).ln()) / 0.60);

    // Hazus Ch. 13 indoor casualty severity fractions across Moderate, Extensive, and Complete damage states
    let green_frac = stock_vuln * (0.0025 * p_mod + 0.012 * p_ext + 0.048 * p_cmp);
    let yellow_frac = stock_vuln * (0.0005 * p_mod + 0.0038 * p_ext + 0.019 * p_cmp);
    let red_frac = stock_vuln * (0.00008 * p_mod + 0.00095 * p_ext + 0.0078 * p_cmp);
    (
        red_frac.clamp(0.0, 0.08),
        yellow_frac.clamp(0.0, 0.14),
        green_frac.clamp(0.0, 0.25),
    )
}

/// Synthesizes the regional hospital receiving network from placed `EmergencyFacilityKind::Hospital`
/// pins, augmenting up to `cfg.min_receiving_hospitals` on distinct arterial corridors if needed so
/// multi-hospital diversion & regional Burn Center routing always operate on a realistic medical network.
pub fn synthesize_hospital_surge_roster(
    net: &RoadNetwork,
    facilities: &[EmergencyFacility],
    cfg: &HospitalSurgeConfig,
) -> Vec<HospitalSurgeNode> {
    let open_segs: Vec<usize> = net
        .segments
        .iter()
        .filter(|s| !s.closed && !s.lane_ids.is_empty() && s.length >= 20.0)
        .map(|s| s.id)
        .collect();
    if open_segs.is_empty() {
        return Vec::new();
    }

    let mut raw_hosps: Vec<(String, usize, f64, Vec2)> = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::Hospital && f.segment_id < net.segments.len())
        .map(|f| (f.name.clone(), f.segment_id, f.s_m, f.world_pos))
        .collect();

    let b = net.bounds;
    let span_x = (b.max_x - b.min_x).max(100.0);
    let span_y = (b.max_y - b.min_y).max(100.0);
    let candidate_anchors = [
        (
            "Regional Level-I Trauma & Burn Center",
            Vec2 {
                x: b.min_x + 0.74 * span_x,
                y: b.min_y + 0.32 * span_y,
            },
        ),
        (
            "Westside Level-II Community Hospital",
            Vec2 {
                x: b.min_x + 0.24 * span_x,
                y: b.min_y + 0.66 * span_y,
            },
        ),
        (
            "South Valley Satellite Acute ED",
            Vec2 {
                x: b.min_x + 0.54 * span_x,
                y: b.min_y + 0.82 * span_y,
            },
        ),
        (
            "North Corridor Medical Center",
            Vec2 {
                x: b.min_x + 0.42 * span_x,
                y: b.min_y + 0.18 * span_y,
            },
        ),
    ];

    let target_count = cfg.min_receiving_hospitals.clamp(2, 6);
    for (lbl, anchor) in &candidate_anchors {
        if raw_hosps.len() >= target_count {
            break;
        }
        let best_sid = open_segs
            .iter()
            .copied()
            .filter(|&sid| {
                let mid = net.segments[sid].base_pts[net.segments[sid].base_pts.len() / 2];
                raw_hosps
                    .iter()
                    .all(|(_, esid, _, epos)| *esid != sid && epos.dist(mid) > 120.0)
            })
            .min_by(|&a, &b_sid| {
                let pa = net.segments[a].base_pts[net.segments[a].base_pts.len() / 2];
                let pb = net.segments[b_sid].base_pts[net.segments[b_sid].base_pts.len() / 2];
                let ca = pa.dist(*anchor) - (net.segments[a].priority as f64) * 16.0;
                let cb = pb.dist(*anchor) - (net.segments[b_sid].priority as f64) * 16.0;
                ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
            })
            .unwrap_or(open_segs[raw_hosps.len() % open_segs.len()]);

        let seg = &net.segments[best_sid];
        let s_m = seg.length * 0.5;
        let (pt, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, s_m);
        let off = (seg.lane_count as f64 + 1.2) * LANE_WIDTH_M;
        let wpos = Vec2 {
            x: pt.x - ang.sin() * off,
            y: pt.y + ang.cos() * off,
        };
        raw_hosps.push(((*lbl).to_string(), best_sid, s_m, wpos));
    }

    let n_perm = raw_hosps.len();
    let retro_n = ((n_perm as f64) * (cfg.hospital_seismic_retrofit_pct.clamp(0.0, 100.0) / 100.0))
        .round() as usize;

    let mut nodes = Vec::with_capacity(n_perm + 1);
    for (idx, (name, sid, s_m, wpos)) in raw_hosps.into_iter().enumerate() {
        let designation = match idx {
            0 => HospitalTraumaDesignation::Level1TraumaAndBurnCenter,
            1 => HospitalTraumaDesignation::Level2CommunityTrauma,
            _ => HospitalTraumaDesignation::Level3AcuteReceiving,
        };
        let (ed_bays, or_rooms, mut burn_beds) = designation.baseline_capacities();
        if cfg.aba_burn_surge_expansion && burn_beds > 0.0 {
            // ABA +50% Contingency/Crisis Burn Surge bed expansion
            burn_beds = (burn_beds * 1.50).ceil();
        }
        let disp_name = if idx == 0 && !name.contains("Burn") {
            format!("{name} (Lvl-I + Burn)")
        } else {
            name
        };
        nodes.push(HospitalSurgeNode {
            hospital_id: idx,
            name: disp_name,
            designation,
            segment_id: sid,
            s_m,
            world_pos: wpos,
            seismically_retrofitted: idx < retro_n,
            baseline_ed_bays: ed_bays,
            baseline_or_rooms: or_rooms,
            baseline_burn_beds: burn_beds,
        });
    }

    if cfg.deploy_mobile_field_hospitals {
        // Site DMAT Mobile Field Hospital near downtown centroid to intercept walk-in & delayed surge
        let center_anchor = Vec2 {
            x: b.min_x + 0.48 * span_x,
            y: b.min_y + 0.48 * span_y,
        };
        let mfh_sid = open_segs
            .iter()
            .copied()
            .min_by(|&a, &b_sid| {
                let pa = net.segments[a].base_pts[net.segments[a].base_pts.len() / 2];
                let pb = net.segments[b_sid].base_pts[net.segments[b_sid].base_pts.len() / 2];
                pa.dist(center_anchor)
                    .partial_cmp(&pb.dist(center_anchor))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(open_segs[0]);
        let seg = &net.segments[mfh_sid];
        let s_m = seg.length * 0.5;
        let (pt, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, s_m);
        let off = (seg.lane_count as f64 + 1.4) * LANE_WIDTH_M;
        let (ed_bays, or_rooms, burn_beds) =
            HospitalTraumaDesignation::MobileFieldHospital.baseline_capacities();
        nodes.push(HospitalSurgeNode {
            hospital_id: nodes.len(),
            name: "DMAT Mobile Field Hospital".to_string(),
            designation: HospitalTraumaDesignation::MobileFieldHospital,
            segment_id: mfh_sid,
            s_m,
            world_pos: Vec2 {
                x: pt.x + ang.sin() * off,
                y: pt.y - ang.cos() * off,
            },
            seismically_retrofitted: true,
            baseline_ed_bays: ed_bays,
            baseline_or_rooms: or_rooms,
            baseline_burn_beds: burn_beds,
        });
    }

    nodes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HospitalEdTelemetryRow {
    pub hospital_id: usize,
    pub name: String,
    pub designation: HospitalTraumaDesignation,
    pub segment_id: usize,
    pub world_pos: Vec2,
    pub seismically_retrofitted: bool,
    pub baseline_ed_bays: f64,
    pub mean_functional_ed_bays: f64,
    pub mean_functional_burn_beds: f64,
    pub mean_functionality_pct: f64,
    pub water_20psi_ok_pct: f64,
    pub expected_ambulance_arrivals: f64,
    pub expected_burn_arrivals: f64,
    pub expected_walkin_arrivals: f64,
    pub expected_diverted_out_ambulances: f64,
    pub peak_ed_utilization_pct: f64,
    pub mean_red_burn_wait_min: f64,
    pub mean_yellow_wait_min: f64,
    pub time_on_diversion_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelCasualtyRoutingRow {
    pub parcel_id: usize,
    pub zone_code: ZoneCode,
    pub center: Vec2,
    pub expected_red_trauma: f64,
    pub expected_burn_critical: f64,
    pub expected_yellow_delayed: f64,
    pub expected_green_walkin: f64,
    pub closest_hospital_id: usize,
    pub primary_hospital_id: usize,
    pub diversion_prob_pct: f64,
    pub mean_red_burn_treat_min: f64,
    pub golden_hour_pct: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SurgeEpochPoint {
    pub hour: f64,
    pub arrivals_red_burn: f64,
    pub arrivals_yellow: f64,
    pub active_ed_queue_patients: f64,
    pub diverted_ambulances_cum: f64,
    pub mean_ed_wait_min: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HospitalSurgeMonteCarloReport {
    pub config: HospitalSurgeConfig,
    pub total_ambulances: usize,
    pub expected_total_casualties: f64,
    pub expected_red_trauma: f64,
    pub p90_red_trauma: usize,
    pub expected_burn_critical: f64,
    pub p90_burn_critical: usize,
    pub expected_yellow_delayed: f64,
    pub expected_green_walkin: f64,
    pub mean_system_ed_functionality_pct: f64,
    pub expected_diverted_ambulances: f64,
    pub ambulance_diversion_rate_pct: f64,
    pub expected_secondary_burn_transfers: f64,
    pub mean_ambulance_transport_min: f64,
    pub mean_red_burn_ed_wait_min: f64,
    pub p90_red_burn_ed_wait_min: f64,
    pub mean_red_burn_time_to_treatment_min: f64,
    pub golden_hour_treated_pct: f64,
    pub burn_matched_to_burn_center_pct: f64,
    pub expected_preventable_surge_deaths: f64,
    pub surge_timeline: Vec<SurgeEpochPoint>,
    pub hospital_rows: Vec<HospitalEdTelemetryRow>,
    pub parcel_rows: Vec<ParcelCasualtyRoutingRow>,
    pub solve_time_ms: f64,
}

impl HospitalSurgeMonteCarloReport {
    pub fn to_hospital_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.hospital_rows.len() * 140 + 256);
        out.push_str(HOSPITAL_SURGE_FACILITY_CSV_HEADER);
        out.push('\n');
        for h in &self.hospital_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{},{:.1},{:.2},{:.2},{:.1},{:.1},{:.2},{:.2},{:.2},{:.1},{:.2},{:.2},{:.1}\n",
                h.hospital_id,
                h.name.replace('"', "'"),
                h.designation.short_label(),
                h.seismically_retrofitted,
                h.baseline_ed_bays,
                h.mean_functional_ed_bays,
                h.mean_functional_burn_beds,
                h.mean_functionality_pct,
                h.water_20psi_ok_pct,
                h.expected_ambulance_arrivals,
                h.expected_walkin_arrivals,
                h.expected_diverted_out_ambulances,
                h.peak_ed_utilization_pct,
                h.mean_red_burn_wait_min,
                h.mean_yellow_wait_min,
                h.time_on_diversion_pct
            ));
        }
        out
    }

    pub fn to_parcel_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.parcel_rows.len() * 96 + 256);
        out.push_str(HOSPITAL_SURGE_PARCEL_CSV_HEADER);
        out.push('\n');
        for r in &self.parcel_rows {
            out.push_str(&format!(
                "{},{:?},{:.3},{:.3},{:.3},{:.3},{},{:.1},{:.2},{:.1}\n",
                r.parcel_id,
                r.zone_code,
                r.expected_red_trauma,
                r.expected_burn_critical,
                r.expected_yellow_delayed,
                r.expected_green_walkin,
                r.primary_hospital_id,
                r.diversion_prob_pct,
                r.mean_red_burn_treat_min,
                r.golden_hour_pct
            ));
        }
        out
    }

    pub fn export_hospital_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_hospital_csv_string())?;
        Ok(self.hospital_rows.len())
    }

    pub fn export_parcel_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_parcel_csv_string())?;
        Ok(self.parcel_rows.len())
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

/// Runs `Rayon`-parallelized post-earthquake hospital ED surge, coupled FFE thermal-burn & seismic
/// trauma priority triage queueing, and dynamic ambulance hospital-diversion Monte Carlo simulations.
pub fn run_hospital_surge_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &HospitalSurgeConfig,
) -> HospitalSurgeMonteCarloReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let n_segs = net.segments.len();
    let n_junc = net.junctions.len();
    let trials = cfg.ffe.seismic.monte_carlo_trials.clamp(20, 5_000);

    let hospitals = synthesize_hospital_surge_roster(net, facilities, cfg);
    let n_hosp = hospitals.len();

    let fire_station_segs: Vec<usize> = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation && f.segment_id < n_segs)
        .map(|f| f.segment_id)
        .collect();
    let n_stations = fire_station_segs.len().max(1);
    let n_engines = (n_stations * 2).max(2);
    let total_ambulances = (n_stations * cfg.ambulances_per_station.clamp(1, 12)).max(2);

    // 1. Precompute parcel occupant exposure, floor area, nearest junction, and median Sa(1s) / PGA
    let west_x = net.bounds.min_x;
    let mut parcel_pop = Vec::with_capacity(n_parcels);
    let mut parcel_tfa_mmsqft = Vec::with_capacity(n_parcels);
    let mut parcel_med_sa10 = Vec::with_capacity(n_parcels);
    let mut parcel_med_pga = Vec::with_capacity(n_parcels);
    let mut parcel_near_junc = Vec::with_capacity(n_parcels);
    let mut parcel_near_seg = Vec::with_capacity(n_parcels);
    let mut parcel_access_s = Vec::with_capacity(n_parcels);
    let mut parcel_hamada = Vec::with_capacity(n_parcels);

    for p in parcels {
        let sp = p.zone.spec();
        // Daytime/evening blended indoor population (`residents + 0.55 * jobs`)
        let occ_pop = (p.residents() * 0.85 + p.jobs() * 0.55).max(if p.zone == ZoneCode::PARK {
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
        let r_km = (cfg.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(
            cfg.ffe.seismic.moment_magnitude_mw,
            r_km,
            cfg.ffe.seismic.vs30_m_per_s,
        );
        let pga = (0.86
            * sa10
            * (cfg.ffe.seismic.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
        .clamp(0.01, 3.0);

        let mut nj = 0usize;
        let mut bd2 = f64::INFINITY;
        for (ji, j) in net.junctions.iter().enumerate() {
            let d2 = (p.center.x - j.pos.x).powi(2) + (p.center.y - j.pos.y).powi(2);
            if d2 < bd2 {
                bd2 = d2;
                nj = ji;
            }
        }

        let mut b_sid = 0usize;
        let mut b_acc = 20.0;
        let mut b_sd = f64::INFINITY;
        for (sid, seg) in net.segments.iter().enumerate() {
            if !seg.closed && !seg.base_pts.is_empty() {
                let mid = seg.base_pts[seg.base_pts.len() / 2];
                let d = p.center.dist(mid);
                if d < b_sd {
                    b_sd = d;
                    b_sid = sid;
                    b_acc = d / 7.5;
                }
            }
        }

        parcel_pop.push(occ_pop);
        parcel_tfa_mmsqft.push(tfa);
        parcel_med_sa10.push(sa10);
        parcel_med_pga.push(pga);
        parcel_near_junc.push(nj);
        parcel_near_seg.push(b_sid);
        parcel_access_s.push(b_acc);
        parcel_hamada.push(hamada_hazus_spread_rates(p.zone, cfg.ffe.wind_speed_mps));
    }

    // 2. Precompute hospital site GMPE & nearest junction
    let mut hosp_med_sa10 = Vec::with_capacity(n_hosp);
    let mut hosp_med_pga = Vec::with_capacity(n_hosp);
    let mut hosp_near_junc = Vec::with_capacity(n_hosp);
    for h in &hospitals {
        let loc_km = ((h.world_pos.x - west_x).max(0.0)) / 1000.0;
        let r_km = (cfg.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(
            cfg.ffe.seismic.moment_magnitude_mw,
            r_km,
            cfg.ffe.seismic.vs30_m_per_s,
        );
        let pga = (0.86
            * sa10
            * (cfg.ffe.seismic.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
        .clamp(0.01, 3.0);
        let mut nj = 0usize;
        let mut bd2 = f64::INFINITY;
        for (ji, j) in net.junctions.iter().enumerate() {
            let d2 = (h.world_pos.x - j.pos.x).powi(2) + (h.world_pos.y - j.pos.y).powi(2);
            if d2 < bd2 {
                bd2 = d2;
                nj = ji;
            }
        }
        hosp_med_sa10.push(sa10);
        hosp_med_pga.push(pga);
        hosp_near_junc.push(nj);
    }

    // 3. Precompute Hamada parcel-to-parcel fire-spread graph for coupled FFE burn casualties
    let wind_rad = cfg.ffe.wind_direction_deg.to_radians();
    let wx_hat = wind_rad.cos();
    let wy_hat = wind_rad.sin();
    let horizon_m = cfg.ffe.conflagration_horizon_mins.clamp(15.0, 720.0);
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
                14.5 + 0.85 * cfg.ffe.wind_speed_mps
            } else {
                10.5 + 0.25 * cfg.ffe.wind_speed_mps
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
            let jump_delay_min = 2.5 + 0.18 * edge_gap_m;
            let step_min = center_dist / v_theta + jump_delay_min;
            if step_min <= horizon_m {
                fire_spread_adj[i].push((k, step_min));
            }
        }
    }

    // 4. Bridges, buried water corridors, and Jayaram-Baker (2009) spatial correlation Cholesky
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, &cfg.ffe.seismic);
    let n_br = bridges.len();

    let mut u_corridors: Vec<(usize, f64, Vec2, bool)> = Vec::new();
    let mut junc_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_junc];
    for s in &net.segments {
        if s.from_junc < s.to_junc || !net.segments.iter().any(|o| o.from_junc == s.to_junc && o.to_junc == s.from_junc) {
            let cid = u_corridors.len();
            let mid = s.base_pts[s.base_pts.len() / 2];
            u_corridors.push((cid, s.length.max(12.0), mid, s.priority >= 2));
            junc_adj[s.from_junc].push((s.to_junc, cid));
            junc_adj[s.to_junc].push((s.from_junc, cid));
        }
    }
    let n_corr = u_corridors.len();
    let n_rep_corr = ((n_corr as f64)
        * (cfg.ffe.seismic.seismic_pipe_replacement_pct.clamp(0.0, 100.0) / 100.0))
        .round() as usize;

    let b = net.bounds;
    let wtp_junc = (0..n_junc)
        .min_by(|&a, &bj| {
            let da = net.junctions[a].pos.dist(Vec2 {
                x: b.min_x,
                y: 0.5 * (b.min_y + b.max_y),
            });
            let db = net.junctions[bj].pos.dist(Vec2 {
                x: b.min_x,
                y: 0.5 * (b.min_y + b.max_y),
            });
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);

    // Precompute road base travel times & adjacency for rapid per-trial reverse Dijkstra from each hospital
    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    let base_seg_trav: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                f64::INFINITY
            } else {
                let eff_v = (s.ema_speed_mps.max(s.speed_mps * 0.48) * speed_boost).clamp(3.2, 29.0);
                s.length / eff_v
            }
        })
        .collect();
    let mut base_fwd: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    let mut base_rev: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let junc = &net.junctions[seg.to_junc];
        let sig_pen = match junc.control_mode {
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

    // Jayaram-Baker spatial correlation across `[parcels | bridges | hospitals]`
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
    let range_km = if cfg.ffe.seismic.vs30_m_per_s < 270.0 {
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

    // 15-minute surge waves across `surge_window_hours`
    let horizon_h = cfg.surge_window_hours.clamp(2.0, 24.0);
    let n_waves = ((horizon_h * 4.0).round() as usize).clamp(8, 96);
    // Log-normal / gamma-like temporal arrival profile peaking around Hour 1.25 - 2.0 post-quake
    let mut wave_weights = Vec::with_capacity(n_waves);
    for w in 0..n_waves {
        let t_hr = (w as f64 + 0.5) * 0.25;
        let rate = (t_hr / 1.15).powf(1.35) * (-t_hr / 1.45).exp() + 0.04;
        wave_weights.push(rate);
    }
    let w_sum: f64 = wave_weights.iter().sum::<f64>().max(1e-9);
    for rw in &mut wave_weights {
        *rw /= w_sum;
    }

    // Closest hospital baseline index per parcel (under intact network)
    let mut intact_closest_hosp = vec![0usize; n_parcels];
    for (pi, p) in parcels.iter().enumerate() {
        intact_closest_hosp[pi] = (0..n_hosp)
            .filter(|&hi| hospitals[hi].designation != HospitalTraumaDesignation::MobileFieldHospital)
            .min_by(|&a, &bh| {
                p.center
                    .dist(hospitals[a].world_pos)
                    .partial_cmp(&p.center.dist(hospitals[bh].world_pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
    }

    #[derive(Clone)]
    struct TrialOut {
        red_cnt: usize,
        burn_cnt: usize,
        yellow_cnt: usize,
        green_cnt: usize,
        diverted_cnt: f64,
        amb_total: f64,
        sec_burn_transfers: f64,
        sum_amb_travel_min: f64,
        sum_red_wait_min: f64,
        sum_red_treat_min: f64,
        golden_hr_ok: f64,
        burn_matched_ok: f64,
        preventable_deaths: f64,
        mean_ed_func_pct: f64,
        hosp_func_bays: Vec<f64>,
        hosp_func_burn: Vec<f64>,
        hosp_func_pct: Vec<f64>,
        hosp_water_ok: Vec<f64>,
        hosp_amb_in: Vec<f64>,
        hosp_burn_in: Vec<f64>,
        hosp_walk_in: Vec<f64>,
        hosp_div_out: Vec<f64>,
        hosp_peak_util: Vec<f64>,
        hosp_red_wait_sum: Vec<f64>,
        hosp_red_w_den: Vec<f64>,
        hosp_yel_wait_sum: Vec<f64>,
        hosp_yel_w_den: Vec<f64>,
        hosp_div_waves: Vec<f64>,
        parcel_red: Vec<f64>,
        parcel_burn: Vec<f64>,
        parcel_yel: Vec<f64>,
        parcel_grn: Vec<f64>,
        parcel_hosp_votes: Vec<usize>,
        parcel_div_frac: Vec<f64>,
        parcel_treat_min: Vec<f64>,
        parcel_gh_frac: Vec<f64>,
        wave_rb: Vec<f64>,
        wave_yel: Vec<f64>,
        wave_queue: Vec<f64>,
        wave_div_cum: Vec<f64>,
        wave_wait: Vec<f64>,
    }

    let seed_base = cfg.ffe.seismic.random_seed ^ 0xD17E_5EA7_3300_A91B;

    let trial_outs: Vec<TrialOut> = (0..trials)
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

            let tau_b = cfg.ffe.seismic.inter_event_sigma_ln;
            let phi_w = cfg.ffe.seismic.intra_event_sigma_ln;

            // A. Realize bridge closures
            let mut seg_closed = vec![false; n_segs];
            for (bi, br) in bridges.iter().enumerate() {
                let ln_im = br.median_site_sa10_g.ln()
                    + tau_b * eta_inter
                    + phi_w * eps_intra[n_parcels + bi];
                let sa_real = ln_im.exp();
                let probs = br.effective_archetype.exceedance_probs(sa_real);
                let u_ds = rng.next_f64_open01();
                let ds = if u_ds < probs[3] {
                    SeismicDamageState::Complete
                } else if u_ds < probs[2] {
                    SeismicDamageState::Extensive
                } else if u_ds < probs[1] {
                    SeismicDamageState::Moderate
                } else if u_ds < probs[0] {
                    SeismicDamageState::Slight
                } else {
                    SeismicDamageState::None
                };
                if matches!(
                    ds,
                    SeismicDamageState::Extensive | SeismicDamageState::Complete
                ) {
                    for &sid in &br.segment_ids {
                        if sid < n_segs {
                            seg_closed[sid] = true;
                        }
                    }
                }
            }

            // B. Fire-engine forward Dijkstra + ambulance reverse Dijkstra into each hospital
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
                    let u = net.segments[h.segment_id].from_junc;
                    let v = net.segments[h.segment_id].to_junc;
                    for s in &net.segments {
                        if s.from_junc == v && s.to_junc == u && !seg_closed[s.id] && base_seg_trav[s.id].is_finite() {
                            let twin_s = 0.65 * base_seg_trav[s.id];
                            if twin_s < seg_to_h[s.id] {
                                seg_to_h[s.id] = twin_s;
                                heap.push(DijkNode {
                                    cost: twin_s,
                                    id: s.id,
                                });
                            }
                        }
                    }
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
                    let scene_to_hosp_s = if seg_to_h[sid].is_finite() {
                        seg_to_h[sid] + parcel_access_s[pi]
                    } else {
                        // Detour fallback via slow local surface grid if bridge severs primary arterial
                        (parcels[pi].center.dist(h.world_pos) / 4.2) + 210.0
                    };
                    let amb_resp_s = if seg_engine_s[sid].is_finite() {
                        (seg_engine_s[sid] + parcel_access_s[pi]).min(720.0)
                    } else {
                        360.0
                    };
                    // Pre-hospital response + on-scene rapid START stabilization (4.5 min) + transport to hospital `hi`
                    hosp_parcel_min[hi][pi] = (amb_resp_s + scene_to_hosp_s) / 60.0 + 4.5;
                }
            }

            // C. Realize buried water pipe breaks + substation power & nodal pressures
            let mut pipe_open = vec![true; n_corr];
            let mut leak_lps = 0.0;
            for &(cid, len_m, mid, is_art) in &u_corridors {
                let loc_km = ((mid.x - west_x).max(0.0)) / 1000.0;
                let r_km = (cfg.ffe.seismic.rupture_standoff_km + loc_km).max(1.5);
                let sa_c = (median_sa10_gmpe_g(
                    cfg.ffe.seismic.moment_magnitude_mw,
                    r_km,
                    cfg.ffe.seismic.vs30_m_per_s,
                )
                .ln()
                    + tau_b * eta_inter
                    + phi_w * rng.next_standard_normal())
                .exp();
                let pgv =
                    median_pgv_cm_s_from_sa10(sa_c, cfg.ffe.seismic.vs30_m_per_s).clamp(1.0, 350.0);
                let mat = if cid < n_rep_corr {
                    BuriedPipeMaterialClass::SeismicErdipHdpe
                } else {
                    cfg.ffe.seismic.baseline_water_pipe_material
                };
                let rr = ala_hazus_pipe_repair_rate_per_km(
                    pgv,
                    cfg.ffe.seismic.pipe_fragility_law,
                    mat,
                    if is_art { 300.0 } else { 200.0 },
                    false,
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

            let sub_sa = (parcel_med_sa10.first().copied().unwrap_or(0.35).ln() + tau_b * eta_inter)
                .exp();
            let sub_grid_ok = rng.next_f64_open01() >= normal_cdf_phi((sub_sa.ln() - (0.58_f64).ln()) / 0.55);
            let pump_head_m = if sub_grid_ok || cfg.ffe.seismic.critical_lifeline_backup_power {
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

            // D. Coupled Scawthorn/Hazus FFE ignitions + Hamada spread + Hazus trauma casualties
            let mut p_sa_real = vec![0.0; n_parcels];
            let mut p_pga_real = vec![0.0; n_parcels];
            let mut ignitions: Vec<(usize, f64)> = Vec::new();
            for pi in 0..n_parcels {
                let sa_r = (parcel_med_sa10[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.8);
                let pga_r = (parcel_med_pga[pi].ln() + tau_b * eta_inter + phi_w * eps_intra[pi])
                    .exp()
                    .clamp(0.005, 3.5);
                p_sa_real[pi] = sa_r;
                p_pga_real[pi] = pga_r;

                let ig_rate = scawthorn_hazus_ignition_rate_per_mmsqft(
                    pga_r,
                    parcels[pi].zone,
                    cfg.ffe.ignition_model,
                    cfg.ffe.seismic_gas_shutoff_pct,
                );
                let n_ig = sample_poisson_icdf(ig_rate * parcel_tfa_mmsqft[pi], rng.next_f64_open01());
                if n_ig > 0 {
                    let sid = parcel_near_seg[pi];
                    let eng_s = if seg_engine_s[sid].is_finite() {
                        seg_engine_s[sid] + parcel_access_s[pi]
                    } else {
                        9999.0
                    };
                    ignitions.push((pi, eng_s));
                }
            }
            ignitions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let fire_draw_lps = (ignitions.len() as f64) * 58.0;
            let drawdown_m = ((leak_lps + fire_draw_lps) / 145.0).powf(1.852) * 5.8;
            let junc_psi: Vec<f64> = (0..n_junc)
                .map(|ji| {
                    if w_dist[ji] == usize::MAX {
                        0.0
                    } else {
                        let loss_m = (w_dist[ji] as f64) * 1.45 + drawdown_m;
                        ((pump_head_m - loss_m).max(0.0)) * PSI_PER_METER_H2O
                    }
                })
                .collect();

            let cistern_prob = (cfg.ffe.auxiliary_cistern_coverage_pct.clamp(0.0, 100.0)) / 100.0;
            let mut rem_eng = n_engines;
            let mut escaped_seeds = Vec::new();
            let mut knockdown_parcels = vec![false; n_parcels];
            for &(pi, eng_s) in &ignitions {
                let psi = junc_psi[parcel_near_junc[pi]];
                let has_w = psi >= MIN_FIRE_FLOW_RESIDUAL_PSI || rng.next_f64_open01() < cistern_prob;
                if rem_eng > 0 && eng_s <= cfg.ffe.initial_attack_flashover_s && has_w {
                    rem_eng -= 1;
                    knockdown_parcels[pi] = true;
                } else {
                    if rem_eng > 0 && eng_s <= cfg.ffe.initial_attack_flashover_s {
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

            // E. Realize post-earthquake hospital capacities via fault-tree functionality
            let mut h_ed_bays = vec![1.0; n_hosp];
            let mut h_burn_beds = vec![0.0; n_hosp];
            let mut h_func_pct = vec![100.0; n_hosp];
            let mut h_water_ok = vec![0.0; n_hosp];
            for (hi, h) in hospitals.iter().enumerate() {
                let h_idx = n_parcels + n_br + hi;
                let sa_h = (hosp_med_sa10[hi].ln() + tau_b * eta_inter + phi_w * eps_intra[h_idx])
                    .exp();
                let pga_h = (hosp_med_pga[hi].ln() + tau_b * eta_inter + phi_w * eps_intra[h_idx])
                    .exp();
                let psi_h = junc_psi[hosp_near_junc[hi]];
                let q_func = hospital_fault_tree_functionality(
                    h.designation,
                    h.seismically_retrofitted,
                    sa_h,
                    pga_h,
                    rng.next_standard_normal(),
                    sub_grid_ok,
                    psi_h,
                    cfg.hospital_onsite_backup_lifelines
                        || cfg.ffe.seismic.critical_lifeline_backup_power,
                );
                h_func_pct[hi] = q_func * 100.0;
                h_ed_bays[hi] = (h.baseline_ed_bays * q_func).max(1.2);
                h_burn_beds[hi] = h.baseline_burn_beds * q_func;
                h_water_ok[hi] = if psi_h >= MIN_FIRE_FLOW_RESIDUAL_PSI
                    || cfg.hospital_onsite_backup_lifelines
                    || h.designation == HospitalTraumaDesignation::MobileFieldHospital
                {
                    100.0
                } else {
                    0.0
                };
            }

            // F. Sample parcel casualties `(Red, Burn, Yellow, Green)`
            let mut p_red = vec![0usize; n_parcels];
            let mut p_burn = vec![0usize; n_parcels];
            let mut p_yel = vec![0usize; n_parcels];
            let mut p_grn = vec![0usize; n_parcels];

            for pi in 0..n_parcels {
                let (rf, yf, gf) = hazus_parcel_casualty_rates(p_sa_real[pi], parcels[pi].zone);
                let pop = parcel_pop[pi];
                let r_cnt = sample_poisson_icdf(pop * rf, rng.next_f64_open01());
                let mut y_cnt = sample_poisson_icdf(pop * yf, rng.next_f64_open01());
                let g_cnt = sample_poisson_icdf(pop * gf, rng.next_f64_open01());

                // Coupled thermal burn & smoke inhalation casualties from FFE conflagration
                let mut b_cnt = 0usize;
                if burn_time_min[pi].is_finite() {
                    // Rapidly arriving wind-driven conflagration traps more occupants before evacuation
                    let trap_factor = (1.0 - 0.55 * (burn_time_min[pi] / horizon_m).clamp(0.0, 1.0))
                        .clamp(0.35, 1.0);
                    let lam_burn_crit = pop * 0.0085 * trap_factor;
                    let lam_burn_mod = pop * 0.0140 * trap_factor;
                    b_cnt = sample_poisson_icdf(lam_burn_crit, rng.next_f64_open01());
                    y_cnt += sample_poisson_icdf(lam_burn_mod, rng.next_f64_open01());
                } else if knockdown_parcels[pi] {
                    b_cnt = sample_poisson_icdf(pop * 0.0012, rng.next_f64_open01());
                }

                p_red[pi] = r_cnt;
                p_burn[pi] = b_cnt;
                p_yel[pi] = y_cnt;
                p_grn[pi] = g_cnt;
            }

            let tot_red: usize = p_red.iter().sum();
            let tot_burn: usize = p_burn.iter().sum();
            let tot_yel: usize = p_yel.iter().sum();
            let tot_grn: usize = p_grn.iter().sum();

            // G. Multi-wave priority ED triage queueing & dynamic ambulance hospital-diversion simulation
            // Service rates per functional ED bay (`patients / 15-min wave`):
            // - Red/Burn acute resuscitation stabilization before OR/ICU handoff: ~75 min (`0.20 / bay / 15m`)
            // - Yellow delayed acute care: ~60 min (`0.25 / bay / 15m`)
            // - Green walk-in fast-track: ~25 min (`0.60 / bay / 15m`, consumes up to 25% of bays if not diverted to MFH)
            let mut q_p1 = vec![0.0; n_hosp]; // Active + queued Red & Burn patients
            let mut q_p2 = vec![0.0; n_hosp]; // Active + queued Yellow patients
            let mut q_p3 = vec![0.0; n_hosp]; // Walk-in Green patients
            let mut burn_occ = vec![0.0; n_hosp]; // Occupied ABA Burn ICU / Surge beds

            let mut hosp_amb_in = vec![0.0; n_hosp];
            let mut hosp_burn_in = vec![0.0; n_hosp];
            let mut hosp_walk_in = vec![0.0; n_hosp];
            let mut hosp_div_out = vec![0.0; n_hosp];
            let mut hosp_peak_util = vec![0.0f64; n_hosp];
            let mut hosp_red_wait_sum = vec![0.0; n_hosp];
            let mut hosp_red_w_den = vec![0.0; n_hosp];
            let mut hosp_yel_wait_sum = vec![0.0; n_hosp];
            let mut hosp_yel_w_den = vec![0.0; n_hosp];
            let mut hosp_div_waves = vec![0.0; n_hosp];

            let mut diverted_cnt = 0.0;
            let mut amb_total = 0.0;
            let mut sec_burn_transfers = 0.0;
            let mut sum_amb_travel_min = 0.0;
            let mut sum_red_wait_min = 0.0;
            let mut sum_red_treat_min = 0.0;
            let mut golden_hr_ok = 0.0;
            let mut burn_matched_ok = 0.0;
            let mut preventable_deaths = 0.0;

            let mut parcel_hosp_load = vec![vec![0.0; n_hosp]; n_parcels];
            let mut parcel_div_num = vec![0.0; n_parcels];
            let mut parcel_amb_den = vec![0.0; n_parcels];
            let mut parcel_treat_sum = vec![0.0; n_parcels];
            let mut parcel_gh_sum = vec![0.0; n_parcels];
            let mut parcel_rb_den = vec![0.0; n_parcels];

            let mut wave_rb = vec![0.0; n_waves];
            let mut wave_yel = vec![0.0; n_waves];
            let mut wave_queue = vec![0.0; n_waves];
            let mut wave_div_cum = vec![0.0; n_waves];
            let mut wave_wait = vec![0.0; n_waves];

            let div_thresh_frac = (cfg.diversion_occupancy_threshold_pct.clamp(50.0, 160.0)) / 100.0;
            let mfh_idx = hospitals
                .iter()
                .position(|h| h.designation == HospitalTraumaDesignation::MobileFieldHospital);

            for (wi, &ww) in wave_weights.iter().enumerate() {
                // Ambulance fleet dispatch queueing delay when 15-min wave demand exceeds fleet cycle capacity
                let wave_amb_demand = ((tot_red + tot_burn + tot_yel) as f64) * ww;
                let amb_cap_15m = (total_ambulances as f64) * 0.52; // ~29 min round-trip turnaround
                let fleet_queue_min = if wave_amb_demand > amb_cap_15m {
                    ((wave_amb_demand - amb_cap_15m) / amb_cap_15m.max(1.0)) * 11.5
                } else {
                    0.0
                };

                let mut w_wait_acc = 0.0;
                let mut w_wait_den = 0.0;

                // 1. Walk-in Green convergence first (uncontrolled self-presentation unless MFH deployed)
                for pi in 0..n_parcels {
                    let g_arr = (p_grn[pi] as f64) * ww;
                    if g_arr <= 1e-9 {
                        continue;
                    }
                    if let Some(mi) = mfh_idx {
                        // Triage officers & public messaging route 65% of walking wounded to Mobile Field Hospital
                        q_p3[mi] += 0.65 * g_arr;
                        hosp_walk_in[mi] += 0.65 * g_arr;
                        let ch = intact_closest_hosp[pi];
                        q_p3[ch] += 0.35 * g_arr;
                        hosp_walk_in[ch] += 0.35 * g_arr;
                    } else {
                        let ch = intact_closest_hosp[pi];
                        q_p3[ch] += g_arr;
                        hosp_walk_in[ch] += g_arr;
                    }
                }

                // Helper closure computing expected priority ED resuscitation wait (minutes) at hospital `hi`
                let calc_p1_wait_min =
                    |hi: usize, qp1: &[f64], qp3: &[f64]| -> f64 {
                        let bays = h_ed_bays[hi];
                        // Unscreened walk-in crowding ties up up to 22% of acute bays until cleared
                        let walkin_drag = (qp3[hi] * 0.14).min(bays * 0.22);
                        let eff_bays = (bays - walkin_drag).max(0.8);
                        let mu_p1_per_min = eff_bays / 72.0; // 72-min mean trauma resuscitation bay occupancy
                        let rho = (qp1[hi] / eff_bays).max(0.0);
                        let ovf_wait = ((qp1[hi] - eff_bays * 0.85).max(0.0)) / mu_p1_per_min.max(0.01);
                        let congestion_tail = 4.5 * rho.powi(2);
                        ovf_wait + congestion_tail
                    };

                let calc_p2_wait_min =
                    |hi: usize, qp1: &[f64], qp2: &[f64], qp3: &[f64]| -> f64 {
                        let w1 = calc_p1_wait_min(hi, qp1, qp3);
                        let bays = (h_ed_bays[hi] - (qp3[hi] * 0.14).min(h_ed_bays[hi] * 0.22)).max(0.8);
                        let mu_tot_per_min = bays / 60.0;
                        w1 + ((qp2[hi] + 0.5 * qp1[hi] - bays * 0.70).max(0.0))
                            / mu_tot_per_min.max(0.01)
                            + 6.0 * ((qp1[hi] + qp2[hi]) / bays).powi(2)
                    };

                // 2. Dispatch Priority 1 (`BurnCritical` & `RedImmediate`) + Priority 2 (`YellowDelayed`) ambulances
                for pi in 0..n_parcels {
                    let b_arr = (p_burn[pi] as f64) * ww;
                    let r_arr = (p_red[pi] as f64) * ww;
                    let y_arr = (p_yel[pi] as f64) * ww;
                    if b_arr + r_arr + y_arr <= 1e-9 {
                        continue;
                    }

                    wave_rb[wi] += b_arr + r_arr;
                    wave_yel[wi] += y_arr;

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

                    // Select receiving hospital for `BurnCritical`, `RedImmediate`, and `YellowDelayed`
                    for &(acuity, vol) in &[
                        (CasualtyTriageAcuity::BurnCriticalSpecialty, b_arr),
                        (CasualtyTriageAcuity::RedImmediateTrauma, r_arr),
                        (CasualtyTriageAcuity::YellowDelayedTrauma, y_arr),
                    ] {
                        if vol <= 1e-9 {
                            continue;
                        }
                        let chosen_h = match cfg.diversion_policy {
                            AmbulanceDiversionPolicy::ClosestHospitalNoDiversion => closest_h,
                            AmbulanceDiversionPolicy::ThresholdSaturationDiversion => {
                                let util_c = (q_p1[closest_h] + q_p2[closest_h] + 0.35 * q_p3[closest_h])
                                    / h_ed_bays[closest_h];
                                if util_c < div_thresh_frac {
                                    closest_h
                                } else {
                                    // Divert to closest non-saturated permanent hospital (or least-loaded if all saturated)
                                    (0..n_hosp)
                                        .filter(|&hi| {
                                            hospitals[hi].designation
                                                != HospitalTraumaDesignation::MobileFieldHospital
                                                && ((q_p1[hi] + q_p2[hi] + 0.35 * q_p3[hi])
                                                    / h_ed_bays[hi])
                                                    < div_thresh_frac
                                        })
                                        .min_by(|&a, &bh| {
                                            hosp_parcel_min[a][pi]
                                                .partial_cmp(&hosp_parcel_min[bh][pi])
                                                .unwrap_or(Ordering::Equal)
                                        })
                                        .unwrap_or_else(|| {
                                            (0..n_hosp)
                                                .filter(|&hi| {
                                                    hospitals[hi].designation
                                                        != HospitalTraumaDesignation::MobileFieldHospital
                                                })
                                                .min_by(|&a, &bh| {
                                                    let ua = (q_p1[a] + q_p2[a]) / h_ed_bays[a];
                                                    let ub = (q_p1[bh] + q_p2[bh]) / h_ed_bays[bh];
                                                    ua.partial_cmp(&ub).unwrap_or(Ordering::Equal)
                                                })
                                                .unwrap_or(closest_h)
                                        })
                                }
                            }
                            AmbulanceDiversionPolicy::DynamicWaitPlusTravelTriage => {
                                // System-Optimal MOCC routing minimizing Travel + Marginal Priority Wait + Specialty Penalty
                                (0..n_hosp)
                                    .filter(|&hi| {
                                        // Mobile Field Hospitals accept YellowDelayed, while Red/Burn go to surgical/trauma hospitals
                                        acuity == CasualtyTriageAcuity::YellowDelayedTrauma
                                            || hospitals[hi].designation
                                                != HospitalTraumaDesignation::MobileFieldHospital
                                    })
                                    .min_by(|&a, &bh| {
                                        let score_h = |hi: usize| -> f64 {
                                            let t_min = hosp_parcel_min[hi][pi];
                                            let w_min = if acuity
                                                == CasualtyTriageAcuity::YellowDelayedTrauma
                                            {
                                                calc_p2_wait_min(hi, &q_p1, &q_p2, &q_p3)
                                            } else {
                                                // Include marginal congestion externality `1.35 * w1` per Dean (2008) / Deo & Gurvich
                                                1.35 * calc_p1_wait_min(hi, &q_p1, &q_p3)
                                            };
                                            let spec_pen = match acuity {
                                                CasualtyTriageAcuity::BurnCriticalSpecialty => {
                                                    let free_burn = h_burn_beds[hi] - burn_occ[hi];
                                                    if free_burn >= 0.5 {
                                                        0.0
                                                    } else if h_burn_beds[hi] > 0.5 {
                                                        18.0
                                                    } else {
                                                        42.0 // Secondary inter-hospital burn transfer penalty
                                                    }
                                                }
                                                CasualtyTriageAcuity::RedImmediateTrauma => {
                                                    match hospitals[hi].designation {
                                                        HospitalTraumaDesignation::Level1TraumaAndBurnCenter => 0.0,
                                                        HospitalTraumaDesignation::Level2CommunityTrauma => 4.5,
                                                        HospitalTraumaDesignation::Level3AcuteReceiving => 16.0,
                                                        HospitalTraumaDesignation::MobileFieldHospital => 90.0,
                                                    }
                                                }
                                                _ => 0.0,
                                            };
                                            t_min + w_min + spec_pen
                                        };
                                        score_h(a)
                                            .partial_cmp(&score_h(bh))
                                            .unwrap_or(Ordering::Equal)
                                    })
                                    .unwrap_or(closest_h)
                            }
                        };

                        amb_total += vol;
                        parcel_amb_den[pi] += vol;
                        parcel_hosp_load[pi][chosen_h] += vol;
                        hosp_amb_in[chosen_h] += vol;
                        if chosen_h != closest_h {
                            diverted_cnt += vol;
                            hosp_div_out[closest_h] += vol;
                            parcel_div_num[pi] += vol;
                        }

                        let trav_m = hosp_parcel_min[chosen_h][pi] + fleet_queue_min;
                        sum_amb_travel_min += vol * trav_m;

                        match acuity {
                            CasualtyTriageAcuity::BurnCriticalSpecialty
                            | CasualtyTriageAcuity::RedImmediateTrauma => {
                                let w1 = calc_p1_wait_min(chosen_h, &q_p1, &q_p3);
                                let spec_delay = if acuity
                                    == CasualtyTriageAcuity::BurnCriticalSpecialty
                                {
                                    hosp_burn_in[chosen_h] += vol;
                                    let avail_b = (h_burn_beds[chosen_h] - burn_occ[chosen_h]).max(0.0);
                                    let matched = vol.min(avail_b);
                                    burn_occ[chosen_h] += matched;
                                    burn_matched_ok += matched;
                                    let unmatched = (vol - matched).max(0.0);
                                    sec_burn_transfers += unmatched;
                                    if vol > 1e-9 {
                                        (unmatched / vol) * 34.0
                                    } else {
                                        0.0
                                    }
                                } else if hospitals[chosen_h].designation
                                    == HospitalTraumaDesignation::Level3AcuteReceiving
                                {
                                    11.0
                                } else {
                                    0.0
                                };

                                let tot_treat_m = trav_m + w1 + spec_delay;
                                q_p1[chosen_h] += vol;

                                sum_red_wait_min += vol * (w1 + spec_delay);
                                sum_red_treat_min += vol * tot_treat_m;
                                hosp_red_wait_sum[chosen_h] += vol * w1;
                                hosp_red_w_den[chosen_h] += vol;
                                w_wait_acc += vol * (w1 + spec_delay);
                                w_wait_den += vol;

                                // Smooth Golden-Hour (`<= 60 min`) probability & preventable surge mortality hazard
                                let gh_p = 1.0 / (1.0 + ((tot_treat_m - 60.0) / 8.5).exp());
                                golden_hr_ok += vol * gh_p;
                                parcel_treat_sum[pi] += vol * tot_treat_m;
                                parcel_gh_sum[pi] += vol * gh_p;
                                parcel_rb_den[pi] += vol;

                                // Preventable post-arrival / pre-resuscitation mortality escalates past 45 min delay
                                let excess_m = (tot_treat_m - 42.0).max(0.0);
                                let p_mort = (1.0 - (-0.0082 * excess_m).exp()).clamp(0.0, 0.72);
                                preventable_deaths += vol * p_mort;
                            }
                            CasualtyTriageAcuity::YellowDelayedTrauma => {
                                let w2 = calc_p2_wait_min(chosen_h, &q_p1, &q_p2, &q_p3);
                                q_p2[chosen_h] += vol;
                                hosp_yel_wait_sum[chosen_h] += vol * w2;
                                hosp_yel_w_den[chosen_h] += vol;
                                let excess_y = (trav_m + w2 - 150.0).max(0.0);
                                preventable_deaths +=
                                    vol * (1.0 - (-0.0014 * excess_y).exp()).clamp(0.0, 0.22);
                            }
                            CasualtyTriageAcuity::GreenWalkingWounded => {}
                        }
                    }
                }

                // 3. Advance 15-minute ED service completions across all hospitals (Strict Priority: P1 -> P2 -> P3)
                let mut tot_active_q = 0.0;
                for hi in 0..n_hosp {
                    let bays = h_ed_bays[hi];
                    let util_now = ((q_p1[hi] + q_p2[hi] + 0.35 * q_p3[hi]) / bays) * 100.0;
                    if util_now > hosp_peak_util[hi] {
                        hosp_peak_util[hi] = util_now;
                    }
                    if util_now >= cfg.diversion_occupancy_threshold_pct {
                        hosp_div_waves[hi] += 1.0;
                    }

                    // Priority 1 (`Red` & `Burn`) seizes acute bays first (`15 / 72 = 0.2083` completions/bay/15m)
                    let cap_bays_p1 = (bays - (q_p3[hi] * 0.12).min(bays * 0.20)).max(0.8);
                    let srv_p1 = q_p1[hi].min(cap_bays_p1 * (15.0 / 72.0));
                    let used_bays_p1 = (srv_p1 / (15.0 / 72.0)).min(bays);
                    q_p1[hi] = (q_p1[hi] - srv_p1).max(0.0);

                    // Remaining bays serve Priority 2 (`Yellow`, `15 / 55 = 0.2727` completions/bay/15m)
                    let rem_bays_p2 = (bays - used_bays_p1).max(0.0);
                    let srv_p2 = q_p2[hi].min(rem_bays_p2 * (15.0 / 55.0));
                    let used_bays_p2 = (srv_p2 / (15.0 / 55.0)).min(rem_bays_p2);
                    q_p2[hi] = (q_p2[hi] - srv_p2).max(0.0);

                    // Fast-track + residual bays clear `Green` walk-ins (`15 / 24 = 0.625` completions/bay/15m)
                    let rem_bays_p3 = (bays - used_bays_p1 - used_bays_p2 + 1.5).max(0.5);
                    let srv_p3 = q_p3[hi].min(rem_bays_p3 * (15.0 / 24.0));
                    q_p3[hi] = (q_p3[hi] - srv_p3).max(0.0);

                    // Gradual step-down turnover of initial burn stabilization beds (`~8%` per hour = `2%` per 15m)
                    burn_occ[hi] = (burn_occ[hi] * 0.98).max(0.0);

                    tot_active_q += q_p1[hi] + q_p2[hi];
                }

                wave_queue[wi] = tot_active_q;
                wave_div_cum[wi] = diverted_cnt;
                wave_wait[wi] = if w_wait_den > 1e-9 {
                    w_wait_acc / w_wait_den
                } else {
                    0.0
                };
            }

            let parcel_hosp_votes: Vec<usize> = (0..n_parcels)
                .map(|pi| {
                    parcel_hosp_load[pi]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                        .map(|(hi, &v)| if v > 1e-9 { hi } else { intact_closest_hosp[pi] })
                        .unwrap_or(intact_closest_hosp[pi])
                })
                .collect();
            let parcel_div_frac: Vec<f64> = (0..n_parcels)
                .map(|pi| {
                    if parcel_amb_den[pi] > 1e-9 {
                        (parcel_div_num[pi] / parcel_amb_den[pi]) * 100.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let parcel_treat_min: Vec<f64> = (0..n_parcels)
                .map(|pi| {
                    if parcel_rb_den[pi] > 1e-9 {
                        parcel_treat_sum[pi] / parcel_rb_den[pi]
                    } else {
                        hosp_parcel_min[intact_closest_hosp[pi]][pi] + 8.0
                    }
                })
                .collect();
            let parcel_gh_frac: Vec<f64> = (0..n_parcels)
                .map(|pi| {
                    if parcel_rb_den[pi] > 1e-9 {
                        (parcel_gh_sum[pi] / parcel_rb_den[pi]) * 100.0
                    } else {
                        95.0
                    }
                })
                .collect();

            let tot_base_bays: f64 = hospitals.iter().map(|h| h.baseline_ed_bays).sum::<f64>().max(1.0);
            let tot_func_bays: f64 = h_ed_bays.iter().sum::<f64>();

            TrialOut {
                red_cnt: tot_red,
                burn_cnt: tot_burn,
                yellow_cnt: tot_yel,
                green_cnt: tot_grn,
                diverted_cnt,
                amb_total,
                sec_burn_transfers,
                sum_amb_travel_min,
                sum_red_wait_min,
                sum_red_treat_min,
                golden_hr_ok,
                burn_matched_ok,
                preventable_deaths,
                mean_ed_func_pct: ((tot_func_bays / tot_base_bays) * 100.0).min(125.0),
                hosp_func_bays: h_ed_bays,
                hosp_func_burn: h_burn_beds,
                hosp_func_pct: h_func_pct,
                hosp_water_ok: h_water_ok,
                hosp_amb_in,
                hosp_burn_in,
                hosp_walk_in,
                hosp_div_out,
                hosp_peak_util,
                hosp_red_wait_sum,
                hosp_red_w_den,
                hosp_yel_wait_sum,
                hosp_yel_w_den,
                hosp_div_waves,
                parcel_red: p_red.into_iter().map(|x| x as f64).collect(),
                parcel_burn: p_burn.into_iter().map(|x| x as f64).collect(),
                parcel_yel: p_yel.into_iter().map(|x| x as f64).collect(),
                parcel_grn: p_grn.into_iter().map(|x| x as f64).collect(),
                parcel_hosp_votes,
                parcel_div_frac,
                parcel_treat_min,
                parcel_gh_frac,
                wave_rb,
                wave_yel,
                wave_queue,
                wave_div_cum,
                wave_wait,
            }
        })
        .collect();

    // Aggregate Monte Carlo ensemble statistics
    let tf = trials as f64;
    let mut red_vec: Vec<usize> = trial_outs.iter().map(|t| t.red_cnt).collect();
    let mut burn_vec: Vec<usize> = trial_outs.iter().map(|t| t.burn_cnt).collect();
    let mut red_wait_trials: Vec<f64> = trial_outs
        .iter()
        .map(|t| {
            let rb = (t.red_cnt + t.burn_cnt) as f64;
            if rb > 0.0 {
                t.sum_red_wait_min / rb
            } else {
                0.0
            }
        })
        .collect();
    red_vec.sort_unstable();
    burn_vec.sort_unstable();
    red_wait_trials.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p90_idx = (((trials - 1) as f64) * 0.90).round() as usize;

    let exp_red = trial_outs.iter().map(|t| t.red_cnt as f64).sum::<f64>() / tf;
    let exp_burn = trial_outs.iter().map(|t| t.burn_cnt as f64).sum::<f64>() / tf;
    let exp_yel = trial_outs.iter().map(|t| t.yellow_cnt as f64).sum::<f64>() / tf;
    let exp_grn = trial_outs.iter().map(|t| t.green_cnt as f64).sum::<f64>() / tf;
    let exp_div = trial_outs.iter().map(|t| t.diverted_cnt).sum::<f64>() / tf;
    let exp_amb = trial_outs.iter().map(|t| t.amb_total).sum::<f64>() / tf;
    let exp_sec_tr = trial_outs.iter().map(|t| t.sec_burn_transfers).sum::<f64>() / tf;
    let exp_prev_d = trial_outs.iter().map(|t| t.preventable_deaths).sum::<f64>() / tf;
    let mean_func = trial_outs.iter().map(|t| t.mean_ed_func_pct).sum::<f64>() / tf;

    let tot_rb_all: f64 = trial_outs
        .iter()
        .map(|t| (t.red_cnt + t.burn_cnt) as f64)
        .sum();
    let tot_amb_all: f64 = trial_outs.iter().map(|t| t.amb_total).sum();
    let tot_burn_all: f64 = trial_outs.iter().map(|t| t.burn_cnt as f64).sum();

    let mean_amb_trav = if tot_amb_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_amb_travel_min).sum::<f64>() / tot_amb_all
    } else {
        8.0
    };
    let mean_rb_wait = if tot_rb_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_red_wait_min).sum::<f64>() / tot_rb_all
    } else {
        0.0
    };
    let mean_rb_treat = if tot_rb_all > 1e-9 {
        trial_outs.iter().map(|t| t.sum_red_treat_min).sum::<f64>() / tot_rb_all
    } else {
        mean_amb_trav
    };
    let gh_pct = if tot_rb_all > 1e-9 {
        (trial_outs.iter().map(|t| t.golden_hr_ok).sum::<f64>() / tot_rb_all) * 100.0
    } else {
        100.0
    };
    let burn_match_pct = if tot_burn_all > 1e-9 {
        (trial_outs.iter().map(|t| t.burn_matched_ok).sum::<f64>() / tot_burn_all) * 100.0
    } else {
        100.0
    };

    let mut hospital_rows = Vec::with_capacity(n_hosp);
    for (hi, h) in hospitals.iter().enumerate() {
        let rw_sum: f64 = trial_outs.iter().map(|t| t.hosp_red_wait_sum[hi]).sum();
        let rw_den: f64 = trial_outs.iter().map(|t| t.hosp_red_w_den[hi]).sum();
        let yw_sum: f64 = trial_outs.iter().map(|t| t.hosp_yel_wait_sum[hi]).sum();
        let yw_den: f64 = trial_outs.iter().map(|t| t.hosp_yel_w_den[hi]).sum();
        hospital_rows.push(HospitalEdTelemetryRow {
            hospital_id: hi,
            name: h.name.clone(),
            designation: h.designation,
            segment_id: h.segment_id,
            world_pos: h.world_pos,
            seismically_retrofitted: h.seismically_retrofitted,
            baseline_ed_bays: h.baseline_ed_bays,
            mean_functional_ed_bays: trial_outs.iter().map(|t| t.hosp_func_bays[hi]).sum::<f64>()
                / tf,
            mean_functional_burn_beds: trial_outs.iter().map(|t| t.hosp_func_burn[hi]).sum::<f64>()
                / tf,
            mean_functionality_pct: trial_outs.iter().map(|t| t.hosp_func_pct[hi]).sum::<f64>()
                / tf,
            water_20psi_ok_pct: trial_outs.iter().map(|t| t.hosp_water_ok[hi]).sum::<f64>() / tf,
            expected_ambulance_arrivals: trial_outs.iter().map(|t| t.hosp_amb_in[hi]).sum::<f64>()
                / tf,
            expected_burn_arrivals: trial_outs.iter().map(|t| t.hosp_burn_in[hi]).sum::<f64>() / tf,
            expected_walkin_arrivals: trial_outs.iter().map(|t| t.hosp_walk_in[hi]).sum::<f64>()
                / tf,
            expected_diverted_out_ambulances: trial_outs
                .iter()
                .map(|t| t.hosp_div_out[hi])
                .sum::<f64>()
                / tf,
            peak_ed_utilization_pct: trial_outs.iter().map(|t| t.hosp_peak_util[hi]).sum::<f64>()
                / tf,
            mean_red_burn_wait_min: if rw_den > 1e-9 { rw_sum / rw_den } else { 0.0 },
            mean_yellow_wait_min: if yw_den > 1e-9 { yw_sum / yw_den } else { 0.0 },
            time_on_diversion_pct: (trial_outs.iter().map(|t| t.hosp_div_waves[hi]).sum::<f64>()
                / (tf * n_waves as f64))
                * 100.0,
        });
    }

    let mut parcel_rows = Vec::with_capacity(n_parcels);
    for (pi, p) in parcels.iter().enumerate() {
        let mut vote_cnt = vec![0usize; n_hosp];
        for t in &trial_outs {
            vote_cnt[t.parcel_hosp_votes[pi]] += 1;
        }
        let primary_h = vote_cnt
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| c)
            .map(|(hi, _)| hi)
            .unwrap_or(intact_closest_hosp[pi]);

        parcel_rows.push(ParcelCasualtyRoutingRow {
            parcel_id: pi,
            zone_code: p.zone,
            center: p.center,
            expected_red_trauma: trial_outs.iter().map(|t| t.parcel_red[pi]).sum::<f64>() / tf,
            expected_burn_critical: trial_outs.iter().map(|t| t.parcel_burn[pi]).sum::<f64>() / tf,
            expected_yellow_delayed: trial_outs.iter().map(|t| t.parcel_yel[pi]).sum::<f64>() / tf,
            expected_green_walkin: trial_outs.iter().map(|t| t.parcel_grn[pi]).sum::<f64>() / tf,
            closest_hospital_id: intact_closest_hosp[pi],
            primary_hospital_id: primary_h,
            diversion_prob_pct: trial_outs.iter().map(|t| t.parcel_div_frac[pi]).sum::<f64>() / tf,
            mean_red_burn_treat_min: trial_outs
                .iter()
                .map(|t| t.parcel_treat_min[pi])
                .sum::<f64>()
                / tf,
            golden_hour_pct: trial_outs.iter().map(|t| t.parcel_gh_frac[pi]).sum::<f64>() / tf,
        });
    }

    let mut surge_timeline = Vec::with_capacity(n_waves);
    for wi in 0..n_waves {
        surge_timeline.push(SurgeEpochPoint {
            hour: (wi as f64 + 1.0) * 0.25,
            arrivals_red_burn: trial_outs.iter().map(|t| t.wave_rb[wi]).sum::<f64>() / tf,
            arrivals_yellow: trial_outs.iter().map(|t| t.wave_yel[wi]).sum::<f64>() / tf,
            active_ed_queue_patients: trial_outs.iter().map(|t| t.wave_queue[wi]).sum::<f64>() / tf,
            diverted_ambulances_cum: trial_outs.iter().map(|t| t.wave_div_cum[wi]).sum::<f64>() / tf,
            mean_ed_wait_min: trial_outs.iter().map(|t| t.wave_wait[wi]).sum::<f64>() / tf,
        });
    }

    HospitalSurgeMonteCarloReport {
        config: cfg.clone(),
        total_ambulances,
        expected_total_casualties: exp_red + exp_burn + exp_yel + exp_grn,
        expected_red_trauma: exp_red,
        p90_red_trauma: red_vec[p90_idx],
        expected_burn_critical: exp_burn,
        p90_burn_critical: burn_vec[p90_idx],
        expected_yellow_delayed: exp_yel,
        expected_green_walkin: exp_grn,
        mean_system_ed_functionality_pct: mean_func,
        expected_diverted_ambulances: exp_div,
        ambulance_diversion_rate_pct: if exp_amb > 1e-9 {
            (exp_div / exp_amb) * 100.0
        } else {
            0.0
        },
        expected_secondary_burn_transfers: exp_sec_tr,
        mean_ambulance_transport_min: mean_amb_trav,
        mean_red_burn_ed_wait_min: mean_rb_wait,
        p90_red_burn_ed_wait_min: red_wait_trials[p90_idx],
        mean_red_burn_time_to_treatment_min: mean_rb_treat,
        golden_hour_treated_pct: gh_pct,
        burn_matched_to_burn_center_pct: burn_match_pct,
        expected_preventable_surge_deaths: exp_prev_d,
        surge_timeline,
        hospital_rows,
        parcel_rows,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

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

//! Post-Earthquake Multi-Day Shelter & Hospital Commodity Resupply Inventory-Routing (`IRP` /
//! Clarke-Wright + 2-Opt `SDVRP`) for Potable Water Tankers, Backup Generator Diesel Fuel,
//! MRE Rations, and Medical Oxygen (`LOX` / Cylinders) Across Time-Restoring Damaged Road Networks
//!
//! Grounded in:
//! - FEMA / ASCE Emergency Relief Truck Payload & Daily Consumption Benchmarks (`18,000 L` water
//!   tankers, `21,744 MRE` dry-vans, `3-15 L/person/day` water & `2-3 MREs/person/day`,
//!   `https://ascelibrary.org/doi/10.1061/NHREFO.NHENG-2061`).
//! - CMS 42 CFR § 482.15(e)(3) / Joint Commission 96-Hour Generator Fuel Rule & NFPA 110 Class X
//!   Diesel Resupply (`https://asprtracie.hhs.gov/documents/utility-failure-tip-sheet-fuel.pdf`;
//!   `https://backuppower.ai/healthcare/96-hour-fuel-rule/`).
//! - ASPR TRACIE & WHO Emergency Medical Team (`EMT`) Medical Oxygen (`LOX` & `H`-Cylinder) Surge
//!   Resilience (`https://asprtracie.hhs.gov/documents/utility-failure-tip-sheet-oxygen.pdf`;
//!   `https://extranet.who.int/emt/sites/default/files/9789240092884-eng.pdf`).
//! - Holguín-Veras et al. (2013 / 2015), *Inventory-Allocation Distribution Models for Postdisaster
//!   Humanitarian Logistics with Explicit Consideration of Deprivation Costs*, *Transportation Science*
//!   (`https://pubsonline.informs.org/doi/10.1287/trsc.2014.0565`;
//!   `https://www.tandfonline.com/doi/full/10.1080/00207543.2026.2651396`).

use crate::emergency::EmergencyFacility;
use crate::emergency_shelters::{enumerate_candidate_shelters, EmergencySheltersConfig};
use crate::hospital_surge::{synthesize_hospital_surge_roster, HospitalTraumaDesignation};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi, BuriedPipeMaterialClass, SeismicDamageState,
};
use crate::utilities::{MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O};
use crate::zoning::Parcel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const RESUPPLY_SITE_CSV_HEADER: &str = "site_id,name,site_kind,occupants_or_beds,daily_water_l,daily_diesel_l,daily_mre_meals,daily_oxygen_kl,water_stockout_hrs,diesel_blackout_hrs,mre_stockout_hrs,oxygen_stockout_hrs,service_security_pct,deprivation_cost";
pub const RESUPPLY_DAILY_CSV_HEADER: &str = "day,open_bridges_pct,grid_power_restored_pct,water_mains_restored_pct,water_delivered_kl,diesel_delivered_kl,mre_delivered_k,oxygen_delivered_kl,water_stockout_person_hrs,diesel_blackout_site_hrs,oxygen_shortfall_patient_hrs,cumul_deprivation_cost";

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

/// Four critical post-earthquake relief & lifeline sustainment commodities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReliefCommodityKind {
    /// Bulk potable water tanker (`18,000 L = 4,750 gal` per tanker truck).
    PotableWaterLiters,
    /// Emergency diesel generator fuel bobtail/tanker (`11,400 L = 3,000 gal` per fuel truck).
    GeneratorDieselLiters,
    /// Shelf-stable MRE rations dry-van (`21,744 meals` per logistics van).
    MreRationsMeals,
    /// Medical liquid oxygen (`LOX`) / compressed `H`-cylinder truck (`120.0 kL` gaseous O2 eq per truck).
    MedicalOxygenKiloLiters,
}

impl ReliefCommodityKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::PotableWaterLiters => "Potable Water Tanker (18,000 L)",
            Self::GeneratorDieselLiters => "Diesel Generator Fuel (11,400 L)",
            Self::MreRationsMeals => "MRE Rations Dry-Van (21,744 meals)",
            Self::MedicalOxygenKiloLiters => "Medical Oxygen LOX/Cyl (120 kL)",
        }
    }

    pub fn short_tag(self) -> &'static str {
        match self {
            Self::PotableWaterLiters => "H2O",
            Self::GeneratorDieselLiters => "DSL",
            Self::MreRationsMeals => "MRE",
            Self::MedicalOxygenKiloLiters => "O2",
        }
    }

    /// Standard truck payload capacity (in native units: `L`, `L`, `meals`, `kL`) & pump/dock unload time (`minutes`).
    #[inline]
    pub fn truck_specs(self) -> (f64, f64) {
        match self {
            Self::PotableWaterLiters => (18_000.0, 24.0),
            Self::GeneratorDieselLiters => (11_400.0, 22.0),
            Self::MreRationsMeals => (21_744.0, 28.0),
            Self::MedicalOxygenKiloLiters => (120.0, 26.0),
        }
    }
}

/// Multi-day humanitarian Inventory-Routing (`IRP`) & vehicle dispatch policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResupplyDispatchPolicy {
    /// Fixed equal-share cyclic milk-run rotation (ignores dynamic lifeline outages, burn surge & stockout urgency).
    StaticEqualRotationMilkRun,
    /// Myopic `(s, S)` reorder threshold (`<= 25%` tank triggers single-site out-and-back dispatch without tour savings).
    GreedyMyopicReorderThreshold,
    /// Holguín-Veras Deprivation-Cost `IRP` + Clarke-Wright Savings + 2-Opt Multi-Stop Split-Delivery Routing!
    DeprivationAwareClarkeWrightIrp,
}

impl Default for ResupplyDispatchPolicy {
    fn default() -> Self {
        Self::DeprivationAwareClarkeWrightIrp
    }
}

impl ResupplyDispatchPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::StaticEqualRotationMilkRun => "Static Equal-Share Milk-Run Rotation",
            Self::GreedyMyopicReorderThreshold => "Myopic Single-Stop Reorder Threshold (s,S)",
            Self::DeprivationAwareClarkeWrightIrp => {
                "Deprivation-Aware IRP + Clarke-Wright 2-Opt VRP"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::StaticEqualRotationMilkRun => "Static-MilkRun",
            Self::GreedyMyopicReorderThreshold => "Myopic-sS",
            Self::DeprivationAwareClarkeWrightIrp => "Deprivation-CW-IRP",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemandNodeKind {
    MassCareShelter,
    TraumaBurnHospital,
}

impl DemandNodeKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::MassCareShelter => "SHELT",
            Self::TraumaBurnHospital => "HOSP",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityResupplyConfig {
    /// Coupled clean-air shelters, contraflow evacuation, medevac, hospital surge, seismic & FFE config.
    pub shelters: EmergencySheltersConfig,
    pub dispatch_policy: ResupplyDispatchPolicy,
    /// Post-earthquake multi-day sustainment simulation horizon (`days`, default `7`, `3..=14`).
    pub horizon_days: usize,
    /// Regional Staging Area (`RSA`) fleet counts by specialized vehicle class.
    pub water_tanker_trucks: usize,
    pub diesel_fuel_trucks: usize,
    pub mre_ration_trucks: usize,
    pub medical_oxygen_trucks: usize,
    /// Pre-positioned on-site reserve duration at `t = 0` (`hours`, e.g. `36 h` baseline vs. `96 h` CMS/TJC standard).
    pub initial_onsite_reserve_hours: f64,
    /// Enables `UH-60` / `AW139` rotary-wing sling-load / internal emergency resupply to bridge-severed sites!
    pub enable_helicopter_sling_resupply: bool,
}

impl Default for CommodityResupplyConfig {
    fn default() -> Self {
        Self {
            shelters: EmergencySheltersConfig::default(),
            dispatch_policy: ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp,
            horizon_days: 7,
            water_tanker_trucks: 2,
            diesel_fuel_trucks: 2,
            mre_ration_trucks: 2,
            medical_oxygen_trucks: 2,
            initial_onsite_reserve_hours: 36.0,
            enable_helicopter_sling_resupply: true,
        }
    }
}

/// Holguín-Veras et al. (2013) strictly convex superlinear human deprivation cost `gamma(tau_hours)`
/// per affected person/patient experiencing `deprivation_hours` of stockout (`gamma(0) = 0`, `gamma'' > 0`).
#[inline]
pub fn holguin_veras_deprivation_cost(
    commodity: ReliefCommodityKind,
    deprivation_hours: f64,
    affected_population: f64,
) -> f64 {
    if deprivation_hours <= 1e-9 || affected_population <= 1e-9 {
        return 0.0;
    }
    let tau = deprivation_hours.min(96.0);
    // Exponential convexity parameters `(a, b)` ordered by clinical survival acuity:
    // Medical Oxygen > Potable Water > Backup Generator Diesel (ICU/HEPA power) > MRE Rations
    let (scale, rate_b) = match commodity {
        ReliefCommodityKind::MedicalOxygenKiloLiters => (42.0, 0.145),
        ReliefCommodityKind::PotableWaterLiters => (6.5, 0.088),
        ReliefCommodityKind::GeneratorDieselLiters => (4.8, 0.076),
        ReliefCommodityKind::MreRationsMeals => (1.8, 0.046),
    };
    affected_population * scale * ((rate_b * tau).exp() - 1.0)
}

/// Solves a multi-stop tour `[depot -> stops... -> depot]` over pairwise travel-time matrix `dist_mat`
/// (`0 = depot`, `1..=m` = active stops in `stop_indices`) using Nearest-Insertion / Clarke-Wright
/// savings initialization followed by exact **2-Opt edge-swap local search**.
///
/// Returns `(ordered_stop_indices, total_round_trip_travel_min)`.
pub fn solve_clarke_wright_2opt_tour(
    depot_idx: usize,
    stop_indices: &[usize],
    dist_mat: &[Vec<f64>],
) -> (Vec<usize>, f64) {
    if stop_indices.is_empty() {
        return (Vec::new(), 0.0);
    }
    if stop_indices.len() == 1 {
        let s = stop_indices[0];
        let rt = dist_mat[depot_idx][s] + dist_mat[s][depot_idx];
        return (vec![s], rt);
    }

    // Clarke & Wright (1964) parallel route-merging savings `S(i, j) = d(0, i) + d(j, 0) - d(i, j)`
    let m = stop_indices.len();
    let mut routes: Vec<Vec<usize>> = stop_indices.iter().map(|&s| vec![s]).collect();
    let mut savings: Vec<(f64, usize, usize)> = Vec::with_capacity(m * (m - 1));
    for i in 0..m {
        for j in 0..m {
            if i == j {
                continue;
            }
            let u = stop_indices[i];
            let v = stop_indices[j];
            let sav = dist_mat[depot_idx][v] + dist_mat[u][depot_idx] - dist_mat[u][v];
            if sav > 0.0 {
                savings.push((sav, u, v));
            }
        }
    }
    savings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    for &(_, u, v) in &savings {
        let r_u = routes.iter().position(|r| r.last() == Some(&u));
        let r_v = routes.iter().position(|r| r.first() == Some(&v));
        if let (Some(iu), Some(iv)) = (r_u, r_v) {
            if iu != iv {
                let tail = routes.remove(iv);
                let adj_iu = if iv < iu { iu - 1 } else { iu };
                routes[adj_iu].extend(tail);
            }
        }
    }

    let mut tour: Vec<usize> = routes.into_iter().flatten().collect();

    // Exact 2-Opt local search refinement
    let tour_cost = |seq: &[usize]| -> f64 {
        let mut c = dist_mat[depot_idx][seq[0]];
        for w in seq.windows(2) {
            c += dist_mat[w[0]][w[1]];
        }
        c + dist_mat[*seq.last().unwrap()][depot_idx]
    };

    let mut best_c = tour_cost(&tour);
    let n = tour.len();
    if n >= 3 {
        let mut improved = true;
        let mut iter_guard = 0usize;
        while improved && iter_guard < 32 {
            improved = false;
            iter_guard += 1;
            for i in 0..n - 1 {
                for k in i + 1..n {
                    tour[i..=k].reverse();
                    let cand_c = tour_cost(&tour);
                    if cand_c + 1e-7 < best_c {
                        best_c = cand_c;
                        improved = true;
                    } else {
                        tour[i..=k].reverse();
                    }
                }
            }
        }
    }

    (tour, best_c)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteResupplyTelemetryRow {
    pub site_id: usize,
    pub name: String,
    pub site_kind: DemandNodeKind,
    pub world_pos: Vec2,
    pub occupants_or_beds: f64,
    pub peak_daily_water_l: f64,
    pub peak_daily_diesel_l: f64,
    pub peak_daily_mre_meals: f64,
    pub peak_daily_oxygen_kl: f64,
    pub mean_water_stockout_hrs: f64,
    pub mean_diesel_blackout_hrs: f64,
    pub mean_mre_stockout_hrs: f64,
    pub mean_oxygen_stockout_hrs: f64,
    pub service_security_pct: f64,
    pub expected_deprivation_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledConvoyTour {
    pub commodity: ReliefCommodityKind,
    pub stop_positions: Vec<Vec2>,
    pub stop_names: Vec<String>,
    pub round_trip_min: f64,
    pub payload_delivered: f64,
    pub via_helicopter_airdrop: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DailyResupplyPoint {
    pub day: usize,
    pub open_bridges_pct: f64,
    pub grid_power_restored_pct: f64,
    pub water_mains_restored_pct: f64,
    pub water_delivered_kl: f64,
    pub diesel_delivered_kl: f64,
    pub mre_delivered_k: f64,
    pub oxygen_delivered_kl: f64,
    pub water_stockout_person_hrs: f64,
    pub diesel_blackout_site_hrs: f64,
    pub oxygen_shortfall_patient_hrs: f64,
    pub cumulative_deprivation_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityResupplyReport {
    pub config: CommodityResupplyConfig,
    pub depot_world_pos: Vec2,
    pub total_demand_sites: usize,
    pub total_water_delivered_kl: f64,
    pub total_diesel_delivered_kl: f64,
    pub total_mre_delivered_k: f64,
    pub total_oxygen_delivered_kl: f64,
    pub expected_helicopter_sling_sorties: f64,
    pub expected_water_stockout_person_hrs: f64,
    pub expected_diesel_blackout_site_hrs: f64,
    pub expected_mre_shortfall_person_hrs: f64,
    pub expected_oxygen_shortfall_patient_hrs: f64,
    pub system_supply_security_pct: f64,
    pub total_deprivation_social_cost: f64,
    pub mean_convoy_tour_savings_pct: f64,
    pub daily_timeline: Vec<DailyResupplyPoint>,
    pub site_rows: Vec<SiteResupplyTelemetryRow>,
    pub representative_day1_tours: Vec<ScheduledConvoyTour>,
    pub parcel_security_pct: Vec<f64>,
    pub parcel_deprivation_idx: Vec<f64>,
    pub solve_time_ms: f64,
}

impl CommodityResupplyReport {
    pub fn to_site_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.site_rows.len() * 150 + 256);
        out.push_str(RESUPPLY_SITE_CSV_HEADER);
        out.push('\n');
        for r in &self.site_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{:.0},{:.0},{:.0},{:.0},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1}\n",
                r.site_id,
                r.name.replace('"', "'"),
                r.site_kind.badge(),
                r.occupants_or_beds,
                r.peak_daily_water_l,
                r.peak_daily_diesel_l,
                r.peak_daily_mre_meals,
                r.peak_daily_oxygen_kl,
                r.mean_water_stockout_hrs,
                r.mean_diesel_blackout_hrs,
                r.mean_mre_stockout_hrs,
                r.mean_oxygen_stockout_hrs,
                r.service_security_pct,
                r.expected_deprivation_cost
            ));
        }
        out
    }

    pub fn to_daily_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.daily_timeline.len() * 128 + 256);
        out.push_str(RESUPPLY_DAILY_CSV_HEADER);
        out.push('\n');
        for d in &self.daily_timeline {
            out.push_str(&format!(
                "{},{:.1},{:.1},{:.1},{:.2},{:.2},{:.2},{:.2},{:.1},{:.2},{:.2},{:.1}\n",
                d.day,
                d.open_bridges_pct,
                d.grid_power_restored_pct,
                d.water_mains_restored_pct,
                d.water_delivered_kl,
                d.diesel_delivered_kl,
                d.mre_delivered_k,
                d.oxygen_delivered_kl,
                d.water_stockout_person_hrs,
                d.diesel_blackout_site_hrs,
                d.oxygen_shortfall_patient_hrs,
                d.cumulative_deprivation_cost
            ));
        }
        out
    }

    pub fn export_site_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_site_csv_string())?;
        Ok(self.site_rows.len())
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

#[inline]
fn hazus_bridge_reopen_prob_at_day(ds: SeismicDamageState, day: f64, crew_speedup: f64) -> f64 {
    let eff_d = day * crew_speedup.clamp(0.4, 4.0);
    match ds {
        SeismicDamageState::None | SeismicDamageState::Slight => 1.0,
        SeismicDamageState::Moderate => normal_cdf_phi((eff_d - 2.5) / 2.7),
        SeismicDamageState::Extensive => normal_cdf_phi((eff_d - 14.0) / 9.5), // Emergency shoring lane for relief trucks
        SeismicDamageState::Complete => normal_cdf_phi((eff_d - 65.0) / 30.0),
    }
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

/// Runs `Rayon`-parallelized post-earthquake Multi-Day (`Days 1..=horizon_days`) Commodity Resupply
/// Inventory-Routing (`IRP` / Clarke-Wright 2-Opt `SDVRP`) across shelters and trauma/burn hospitals.
pub fn run_commodity_resupply_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &CommodityResupplyConfig,
) -> CommodityResupplyReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let n_segs = net.segments.len();
    let n_junc = net.junctions.len();
    let seis = &cfg.shelters.evac.medevac.surge.ffe.seismic;
    let trials = seis.monte_carlo_trials.clamp(20, 5_000);
    let n_days = cfg.horizon_days.clamp(3, 14);

    // 1. Assemble Regional Staging Area (`RSA` Depot = Node 0) + Open Shelters + Receiving Hospitals
    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(120.0);
    let sy = (b.max_y - b.min_y).max(120.0);
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

    // Regional Staging Area (`RSA` Highway / Railhead Logistics Depot on northwest arterial perimeter)
    let depot_target = Vec2 {
        x: b.min_x + 0.08 * sx,
        y: b.min_y + 0.14 * sy,
    };
    let depot_seg = (0..n_segs)
        .filter(|&sid| !net.segments[sid].closed && !net.segments[sid].base_pts.is_empty())
        .min_by(|&a, &bs| {
            seg_mids[a]
                .dist(depot_target)
                .partial_cmp(&seg_mids[bs].dist(depot_target))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);
    let depot_pos = seg_mids.get(depot_seg).copied().unwrap_or(depot_target);

    #[derive(Clone)]
    struct SustainmentSite {
        site_id: usize,
        name: String,
        kind: DemandNodeKind,
        seg_id: usize,
        near_junc: usize,
        pos: Vec2,
        base_pop_or_beds: f64,
        diesel_burn_l_per_day_outage: f64,
        oxygen_kl_per_day: f64,
    }

    let cand_shelters = enumerate_candidate_shelters(net, parcels, &cfg.shelters);
    let budget_k = cfg
        .shelters
        .max_open_shelters_budget
        .clamp(1, cand_shelters.len());
    let hospitals =
        synthesize_hospital_surge_roster(net, facilities, &cfg.shelters.evac.medevac.surge);

    let mut sites: Vec<SustainmentSite> = Vec::new();
    for sh in cand_shelters.iter().take(budget_k) {
        let occ = (sh.nominal_cot_capacity as f64) * 0.78;
        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                sh.world_pos
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&sh.world_pos.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        // Shelter backup generator powers lighting, refrigeration, HVAC blowers & portable HEPA banks (~38 L/hr)
        let dsl_day = 920.0 + occ * 0.65;
        // ~3.5% of shelter evacuees require chronic DME oxygen / smoke-inhalation O2 therapy (`2.88 kL/day` each)
        let o2_kl_day = (occ * 0.035) * 2.88;
        sites.push(SustainmentSite {
            site_id: sites.len(),
            name: sh.name.clone(),
            kind: DemandNodeKind::MassCareShelter,
            seg_id: sh.segment_id,
            near_junc: nj,
            pos: sh.world_pos,
            base_pop_or_beds: occ,
            diesel_burn_l_per_day_outage: dsl_day,
            oxygen_kl_per_day: o2_kl_day,
        });
    }

    for h in &hospitals {
        let nj = (0..n_junc)
            .min_by(|&a, &bj| {
                h.world_pos
                    .dist(net.junctions[a].pos)
                    .partial_cmp(&h.world_pos.dist(net.junctions[bj].pos))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        let (beds_eq, dsl_day, o2_kl_day) = match h.designation {
            HospitalTraumaDesignation::Level1TraumaAndBurnCenter => (185.0, 4_200.0, 96.0),
            HospitalTraumaDesignation::Level2CommunityTrauma => (105.0, 2_400.0, 52.0),
            HospitalTraumaDesignation::Level3AcuteReceiving => (58.0, 1_350.0, 26.0),
            HospitalTraumaDesignation::MobileFieldHospital => (45.0, 780.0, 18.0),
        };
        sites.push(SustainmentSite {
            site_id: sites.len(),
            name: h.name.clone(),
            kind: DemandNodeKind::TraumaBurnHospital,
            seg_id: h.segment_id,
            near_junc: nj,
            pos: h.world_pos,
            base_pop_or_beds: beds_eq,
            diesel_burn_l_per_day_outage: dsl_day,
            oxygen_kl_per_day: o2_kl_day,
        });
    }
    let n_sites = sites.len();
    let n_vrp_nodes = 1 + n_sites; // `0 = RSA Depot`, `1..=n_sites = SustainmentSites`

    // Map each parcel to its closest shelter & hospital for GPU security/deprivation projection
    let mut parcel_shelt_idx = vec![0usize; n_parcels];
    let mut parcel_hosp_idx = vec![0usize; n_parcels];
    for (pi, p) in parcels.iter().enumerate() {
        parcel_shelt_idx[pi] = sites
            .iter()
            .filter(|s| s.kind == DemandNodeKind::MassCareShelter)
            .min_by(|a, bs| {
                p.center
                    .dist(a.pos)
                    .partial_cmp(&p.center.dist(bs.pos))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|s| s.site_id)
            .unwrap_or(0);
        parcel_hosp_idx[pi] = sites
            .iter()
            .filter(|s| s.kind == DemandNodeKind::TraumaBurnHospital)
            .min_by(|a, bs| {
                p.center
                    .dist(a.pos)
                    .partial_cmp(&p.center.dist(bs.pos))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|s| s.site_id)
            .unwrap_or(0);
    }

    // Road forward adjacency for heavy relief trucks (`0.82x` heavy-tanker speed factor)
    let mut base_fwd: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    let truck_seg_min: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                f64::INFINITY
            } else {
                (s.length / (s.speed_mps * 0.82).clamp(3.0, 24.0)) / 60.0
            }
        })
        .collect();
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let sig_m = match net.junctions[seg.to_junc].control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => 0.14,
            ControlMode::Roundabout => 0.08,
            ControlMode::PermissiveYield => 0.04,
        };
        for &lid in &seg.lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let conn = &net.connectors[cid];
                let out_sid = conn.to_seg_id;
                if net.segments[out_sid].closed {
                    continue;
                }
                let t_m = match conn.turn_type {
                    TurnType::Through => 0.02,
                    TurnType::Right => 0.06,
                    TurnType::Left => 0.09,
                };
                if !base_fwd[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    base_fwd[seg.id].push((out_sid, sig_m + t_m + truck_seg_min[out_sid]));
                }
            }
        }
    }

    // Bridges & buried water pipe corridors
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, seis);
    let n_br = bridges.len();
    let west_x = b.min_x;

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

    // Spatial correlation across bridges
    let mut br_corr = vec![1.0; n_br.max(1) * n_br.max(1)];
    if n_br > 0 {
        for i in 0..n_br {
            for j in 0..n_br {
                let d_km =
                    seg_mids[bridges[i].segment_ids[0]].dist(seg_mids[bridges[j].segment_ids[0]])
                        / 1000.0;
                br_corr[i * n_br + j] = (-3.0 * d_km / 25.0).exp();
            }
        }
    }
    let br_chol = if n_br > 0 {
        cholesky_lower(&br_corr, n_br)
    } else {
        vec![1.0]
    };

    let commodities = [
        (
            ReliefCommodityKind::PotableWaterLiters,
            cfg.water_tanker_trucks.clamp(1, 16),
        ),
        (
            ReliefCommodityKind::GeneratorDieselLiters,
            cfg.diesel_fuel_trucks.clamp(1, 16),
        ),
        (
            ReliefCommodityKind::MreRationsMeals,
            cfg.mre_ration_trucks.clamp(1, 16),
        ),
        (
            ReliefCommodityKind::MedicalOxygenKiloLiters,
            cfg.medical_oxygen_trucks.clamp(1, 16),
        ),
    ];

    #[derive(Clone)]
    struct ResupTrialOut {
        tot_water_kl: f64,
        tot_diesel_kl: f64,
        tot_mre_k: f64,
        tot_o2_kl: f64,
        heli_sorties: f64,
        water_so_phrs: f64,
        diesel_bo_hrs: f64,
        mre_so_phrs: f64,
        o2_so_phrs: f64,
        tot_dep_cost: f64,
        cw_savings_pct: f64,
        site_so_hrs: Vec<[f64; 4]>,
        site_sec_pct: Vec<f64>,
        site_dep_cost: Vec<f64>,
        day_br_pct: Vec<f64>,
        day_pwr_pct: Vec<f64>,
        day_wtr_pct: Vec<f64>,
        day_deliv: Vec<[f64; 4]>,
        day_w_so: Vec<f64>,
        day_d_bo: Vec<f64>,
        day_o2_so: Vec<f64>,
        day_cum_dep: Vec<f64>,
        day1_tours: Vec<ScheduledConvoyTour>,
    }

    let seed_base = seis.random_seed ^ 0x7350_CC1A_3700_D49E;
    let init_res_days = (cfg.initial_onsite_reserve_hours.clamp(6.0, 168.0)) / 24.0;
    let pipe_crews = seis.water_sewer_repair_crews.max(1) as f64;

    let trial_outs: Vec<ResupTrialOut> = (0..trials)
        .into_par_iter()
        .map(|trial_idx| {
            let mut rng = SplitMix64::new(
                seed_base.wrapping_add((trial_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let eta_inter = rng.next_standard_normal();
            let tau_b = seis.inter_event_sigma_ln;
            let phi_w = seis.intra_event_sigma_ln;

            // 1. Sample initial mainshock bridge damage states
            let mut br_ds = vec![SeismicDamageState::None; n_br];
            if n_br > 0 {
                let mut z_b = vec![0.0; n_br];
                for z in &mut z_b {
                    *z = rng.next_standard_normal();
                }
                for bi in 0..n_br {
                    let mut eps = 0.0;
                    for k in 0..=bi {
                        eps += br_chol[bi * n_br + k] * z_b[k];
                    }
                    let sa_r = (bridges[bi].median_site_sa10_g.ln()
                        + tau_b * eta_inter
                        + phi_w * eps)
                        .exp();
                    let probs = bridges[bi].effective_archetype.exceedance_probs(sa_r);
                    let u = rng.next_f64_open01();
                    br_ds[bi] = if u < probs[3] {
                        SeismicDamageState::Complete
                    } else if u < probs[2] {
                        SeismicDamageState::Extensive
                    } else if u < probs[1] {
                        SeismicDamageState::Moderate
                    } else if u < probs[0] {
                        SeismicDamageState::Slight
                    } else {
                        SeismicDamageState::None
                    };
                }
            }

            // 2. Sample buried water pipe repair counts & substation outage duration (`sub_restore_day`)
            let mut corr_repairs = vec![0usize; n_corr];
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
                let mat = if cid < n_rep_corr {
                    BuriedPipeMaterialClass::SeismicErdipHdpe
                } else {
                    seis.baseline_water_pipe_material
                };
                let rr = ala_hazus_pipe_repair_rate_per_km(
                    pgv,
                    seis.pipe_fragility_law,
                    mat,
                    if is_art { 300.0 } else { 200.0 },
                    false,
                );
                corr_repairs[cid] =
                    sample_poisson_icdf(rr * (len_m / 1000.0), rng.next_f64_open01());
            }
            let total_pipe_breaks: usize = corr_repairs.iter().sum();

            let ref_sa = (median_sa10_gmpe_g(
                seis.moment_magnitude_mw,
                seis.rupture_standoff_km.max(2.0),
                seis.vs30_m_per_s,
            )
            .ln()
                + tau_b * eta_inter)
                .exp();
            let p_sub_out = normal_cdf_phi((ref_sa.ln() - (0.48_f64).ln()) / 0.55);
            let sub_restore_day = if rng.next_f64_open01() < p_sub_out {
                // Hazus substation transformer/switchgear restoration: 2.5 to 6.5 days
                (2.2 + 3.8 * rng.next_f64_open01()) / seis.repair_crew_speedup.clamp(0.5, 3.0)
            } else {
                0.0
            };

            // 3. Initialize on-site storage & starting inventories `(Water, Diesel, MRE, Oxygen)`
            let mut inv = vec![[0.0f64; 4]; n_sites];
            let mut tank_cap = vec![[0.0f64; 4]; n_sites];
            for (si, s) in sites.iter().enumerate() {
                let peak_w = s.base_pop_or_beds * 15.0;
                let peak_d = s.diesel_burn_l_per_day_outage;
                let peak_m = if s.kind == DemandNodeKind::MassCareShelter {
                    s.base_pop_or_beds * 2.5
                } else {
                    s.base_pop_or_beds * 3.0
                };
                let peak_o = s.oxygen_kl_per_day;
                let peaks = [peak_w, peak_d, peak_m, peak_o];
                for c_idx in 0..4 {
                    // Storage capacity holds at least `max(init_res_days, 4.0)` days of surge demand
                    tank_cap[si][c_idx] = peaks[c_idx] * init_res_days.max(4.0);
                    inv[si][c_idx] = peaks[c_idx] * init_res_days;
                }
            }

            let mut site_so_hrs = vec![[0.0f64; 4]; n_sites];
            let mut site_dep_cost = vec![0.0f64; n_sites];
            let mut tot_deliv = [0.0f64; 4];
            let mut heli_sorties = 0.0;
            let mut water_so_phrs = 0.0;
            let mut diesel_bo_hrs = 0.0;
            let mut mre_so_phrs = 0.0;
            let mut o2_so_phrs = 0.0;
            let mut tot_dep_cost = 0.0;
            let mut cw_sav_sum = 0.0;
            let mut cw_sav_cnt = 0usize;

            let mut day_br_pct = Vec::with_capacity(n_days);
            let mut day_pwr_pct = Vec::with_capacity(n_days);
            let mut day_wtr_pct = Vec::with_capacity(n_days);
            let mut day_deliv = Vec::with_capacity(n_days);
            let mut day_w_so = Vec::with_capacity(n_days);
            let mut day_d_bo = Vec::with_capacity(n_days);
            let mut day_o2_so = Vec::with_capacity(n_days);
            let mut day_cum_dep = Vec::with_capacity(n_days);
            let mut day1_tours: Vec<ScheduledConvoyTour> = Vec::new();

            let node_segs: Vec<usize> = std::iter::once(depot_seg)
                .chain(sites.iter().map(|s| s.seg_id))
                .collect();
            let node_pos: Vec<Vec2> = std::iter::once(depot_pos)
                .chain(sites.iter().map(|s| s.pos))
                .collect();

            // 4. Step day-by-day across `day_idx = 0..n_days` (`Day 1 .. Day N`)
            for day_idx in 0..n_days {
                let day_f = (day_idx + 1) as f64;

                // A. Realize Day-`d` reopened bridges & pairwise truck travel-time matrix `dist_mat[u][v]` (minutes)
                let mut seg_closed = vec![false; n_segs];
                let mut open_br = 0usize;
                for (bi, br) in bridges.iter().enumerate() {
                    let p_open =
                        hazus_bridge_reopen_prob_at_day(br_ds[bi], day_f - 0.5, seis.repair_crew_speedup);
                    let u_det = ((bi as f64 * 0.618_033_988_75 + trial_idx as f64 * 0.314_159)
                        .fract())
                    .abs();
                    if u_det <= p_open {
                        open_br += 1;
                    } else {
                        for &sid in &br.segment_ids {
                            if sid < n_segs {
                                seg_closed[sid] = true;
                            }
                        }
                    }
                }
                let open_br_pct = if n_br > 0 {
                    (open_br as f64 / n_br as f64) * 100.0
                } else {
                    100.0
                };

                let mut dist_mat = vec![vec![999.0; n_vrp_nodes]; n_vrp_nodes];
                let mut islanded_site = vec![false; n_sites];
                for u_n in 0..n_vrp_nodes {
                    dist_mat[u_n][u_n] = 0.0;
                    let mut d_min = vec![f64::INFINITY; n_segs];
                    let mut heap = BinaryHeap::new();
                    let src_sid = node_segs[u_n];
                    if src_sid < n_segs && !seg_closed[src_sid] && truck_seg_min[src_sid].is_finite()
                    {
                        d_min[src_sid] = 0.5 * truck_seg_min[src_sid];
                        heap.push(DijkNode {
                            cost: d_min[src_sid],
                            id: src_sid,
                        });
                    }
                    while let Some(DijkNode { cost, id }) = heap.pop() {
                        if cost > d_min[id] {
                            continue;
                        }
                        for &(nxt, w) in &base_fwd[id] {
                            if !seg_closed[nxt] && cost + w < d_min[nxt] {
                                d_min[nxt] = cost + w;
                                heap.push(DijkNode {
                                    cost: cost + w,
                                    id: nxt,
                                });
                            }
                        }
                    }
                    for v_n in 0..n_vrp_nodes {
                        if u_n == v_n {
                            continue;
                        }
                        let dst_sid = node_segs[v_n];
                        if dst_sid < n_segs && d_min[dst_sid].is_finite() {
                            dist_mat[u_n][v_n] = d_min[dst_sid] + 3.5;
                        } else {
                            if u_n == 0 && v_n >= 1 {
                                islanded_site[v_n - 1] = true;
                            }
                            // Unbridged emergency pontoon/detour penalty (`78 min` one-way)
                            dist_mat[u_n][v_n] =
                                (node_pos[u_n].dist(node_pos[v_n]) / 120.0) + 72.0;
                        }
                    }
                }

                // B. Realize Day-`d` electrical grid & buried water main restoration
                let grid_ok_today = (day_f - 0.5) >= sub_restore_day;
                let repaired_breaks = ((day_f - 0.5) * pipe_crews * 2.2).floor() as usize;
                let mut rem_breaks_budget = total_pipe_breaks.saturating_sub(repaired_breaks);
                let mut pipe_open_today = vec![true; n_corr];
                for cid in (0..n_corr).rev() {
                    if corr_repairs[cid] > 0 && rem_breaks_budget > 0 {
                        pipe_open_today[cid] = false;
                        rem_breaks_budget = rem_breaks_budget.saturating_sub(corr_repairs[cid]);
                    }
                }
                let mut w_dist = vec![usize::MAX; n_junc];
                let mut q = VecDeque::new();
                if wtp_junc < n_junc {
                    w_dist[wtp_junc] = 0;
                    q.push_back(wtp_junc);
                }
                while let Some(u) = q.pop_front() {
                    for &(v, cid) in &junc_adj[u] {
                        if pipe_open_today[cid] && w_dist[v] == usize::MAX {
                            w_dist[v] = w_dist[u] + 1;
                            q.push_back(v);
                        }
                    }
                }
                let pump_h = if grid_ok_today || seis.critical_lifeline_backup_power {
                    55.0
                } else {
                    24.0
                };
                let wtr_ok_juncs = (0..n_junc)
                    .filter(|&ji| {
                        w_dist[ji] != usize::MAX
                            && (pump_h - (w_dist[ji] as f64) * 1.45) * PSI_PER_METER_H2O
                                >= MIN_FIRE_FLOW_RESIDUAL_PSI
                    })
                    .count();
                let wtr_rest_pct = (wtr_ok_juncs as f64 / (n_junc.max(1) as f64)) * 100.0;

                // Compute each site's Day-`d` consumption vector `[Water_L, Diesel_L, MRE_meals, Oxygen_kL]`
                // (Shelter occupancy tapers gently `-6%/day` after Day 3 as green-tagged homes reoccupy)
                let occ_taper = if day_idx <= 2 {
                    1.0
                } else {
                    (1.0 - 0.06 * ((day_idx - 2) as f64)).max(0.55)
                };
                let mut daily_burn = vec![[0.0f64; 4]; n_sites];
                for (si, s) in sites.iter().enumerate() {
                    let pop = s.base_pop_or_beds * occ_taper;
                    let site_water_ok = w_dist[s.near_junc] != usize::MAX
                        && (pump_h - (w_dist[s.near_junc] as f64) * 1.45) * PSI_PER_METER_H2O
                            >= 18.0;
                    // Potable water tanker demand: 15 L/person/day when main broken, 3.5 L/person/day bottled drinking buffer when main intact
                    daily_burn[si][0] = pop * if site_water_ok { 3.5 } else { 15.0 };
                    // Generator diesel demand: full load during grid outage, 8% daily exercise/standby top-off when grid restored
                    daily_burn[si][1] = s.diesel_burn_l_per_day_outage
                        * if grid_ok_today { 0.08 } else { 1.0 };
                    // MRE meals
                    daily_burn[si][2] = pop
                        * if s.kind == DemandNodeKind::MassCareShelter {
                            2.5
                        } else {
                            2.2
                        };
                    // Medical oxygen (`kL`)
                    daily_burn[si][3] = s.oxygen_kl_per_day * occ_taper;
                }

                // C. Dispatch each commodity's truck fleet (`10-hour = 600 min` daily shift per truck)
                let mut d_deliv_today = [0.0f64; 4];
                let mut d_w_so = 0.0;
                let mut d_d_bo = 0.0;
                let mut d_o2_so = 0.0;

                for (c_idx, &(comm_kind, n_trucks)) in commodities.iter().enumerate() {
                    let (truck_cap, dock_min) = comm_kind.truck_specs();
                    let fleet_shift_budget_min = (n_trucks as f64) * 600.0;
                    let mut rem_shift_min = fleet_shift_budget_min;

                    match cfg.dispatch_policy {
                        ResupplyDispatchPolicy::StaticEqualRotationMilkRun => {
                            // Rotates a fixed subset of 3 sites per day in index order with equal shares
                            let start_s = (day_idx * 3) % n_sites;
                            let mut rot_stops = Vec::new();
                            for k in 0..3.min(n_sites) {
                                rot_stops.push(1 + ((start_s + k) % n_sites));
                            }
                            let mut seq_min = dist_mat[0][rot_stops[0]];
                            for w in rot_stops.windows(2) {
                                seq_min += dist_mat[w[0]][w[1]];
                            }
                            seq_min += dist_mat[*rot_stops.last().unwrap()][0]
                                + (rot_stops.len() as f64) * dock_min;

                            let max_trips = (rem_shift_min / seq_min.max(25.0)).floor().max(1.0);
                            let trips = max_trips.min(n_trucks as f64);
                            let share_per_stop =
                                (trips * truck_cap) / (rot_stops.len().max(1) as f64);
                            for &vn in &rot_stops {
                                let si = vn - 1;
                                let room = (tank_cap[si][c_idx] - inv[si][c_idx]
                                    + daily_burn[si][c_idx])
                                    .max(0.0);
                                let added = share_per_stop.min(room);
                                inv[si][c_idx] += added;
                                d_deliv_today[c_idx] += added;
                            }
                        }
                        ResupplyDispatchPolicy::GreedyMyopicReorderThreshold => {
                            // Single-stop out-and-back dispatches only when inventory < 30% of tank_cap
                            for si in 0..n_sites {
                                if inv[si][c_idx] > 0.30 * tank_cap[si][c_idx] {
                                    continue;
                                }
                                let rt_min =
                                    dist_mat[0][si + 1] + dist_mat[si + 1][0] + dock_min + 12.0;
                                if rem_shift_min >= rt_min * 0.5 {
                                    let need = (tank_cap[si][c_idx] - inv[si][c_idx]
                                        + daily_burn[si][c_idx] * 0.5)
                                        .max(0.0);
                                    let loads = (need / truck_cap)
                                        .ceil()
                                        .min((rem_shift_min / rt_min).max(0.5));
                                    let deliv = (loads * truck_cap).min(need);
                                    inv[si][c_idx] += deliv;
                                    d_deliv_today[c_idx] += deliv;
                                    rem_shift_min = (rem_shift_min - loads * rt_min).max(0.0);
                                }
                            }
                        }
                        ResupplyDispatchPolicy::DeprivationAwareClarkeWrightIrp => {
                            // 1. Score every site by projected 36-hour Holguín-Veras deprivation hazard!
                            let mut urgent_sites: Vec<(usize, f64, f64)> = Vec::new();
                            for si in 0..n_sites {
                                let burn = daily_burn[si][c_idx];
                                if burn <= 1e-6 {
                                    continue;
                                }
                                let proj_36h_deficit = (1.5 * burn - inv[si][c_idx]).max(0.0);
                                if proj_36h_deficit > 0.02 * burn {
                                    let proj_so_hrs = (proj_36h_deficit / burn) * 24.0;
                                    let haz = holguin_veras_deprivation_cost(
                                        comm_kind,
                                        proj_so_hrs + 4.0,
                                        sites[si].base_pop_or_beds,
                                    );
                                    let target_fill = (tank_cap[si][c_idx] - inv[si][c_idx]
                                        + 0.85 * burn)
                                        .max(proj_36h_deficit);
                                    urgent_sites.push((si, haz, target_fill));
                                }
                            }
                            urgent_sites
                                .sort_by(|a, bs| bs.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                            // Helicopter sling-load bypass for bridge-islanded high-acuity sites!
                            let mut ground_stops: Vec<usize> = Vec::new();
                            let mut ground_needs: Vec<(usize, f64)> = Vec::new();
                            for &(si, _, target_fill) in &urgent_sites {
                                if islanded_site[si] && cfg.enable_helicopter_sling_resupply {
                                    // UH-60 / AW139 sling-load delivers emergency 24h bridge-bypass ration
                                    let air_cap = truck_cap * 0.38;
                                    let air_deliv = target_fill.min(air_cap);
                                    inv[si][c_idx] += air_deliv;
                                    d_deliv_today[c_idx] += air_deliv;
                                    heli_sorties += 1.0;
                                    if trial_idx == 0 && day_idx == 0 {
                                        day1_tours.push(ScheduledConvoyTour {
                                            commodity: comm_kind,
                                            stop_positions: vec![depot_pos, sites[si].pos],
                                            stop_names: vec![
                                                "RSA Depot".to_string(),
                                                sites[si].name.clone(),
                                            ],
                                            round_trip_min: 18.0,
                                            payload_delivered: air_deliv,
                                            via_helicopter_airdrop: true,
                                        });
                                    }
                                } else {
                                    ground_stops.push(si + 1);
                                    ground_needs.push((si, target_fill));
                                }
                            }

                            if !ground_stops.is_empty() {
                                let (ordered_vrp, tour_drive_min) =
                                    solve_clarke_wright_2opt_tour(0, &ground_stops, &dist_mat);
                                let radial_drive_min: f64 = ground_stops
                                    .iter()
                                    .map(|&vn| dist_mat[0][vn] + dist_mat[vn][0])
                                    .sum();
                                if radial_drive_min > 1.0 {
                                    cw_sav_sum += ((radial_drive_min - tour_drive_min).max(0.0)
                                        / radial_drive_min)
                                        * 100.0;
                                    cw_sav_cnt += 1;
                                }

                                let tot_need: f64 = ground_needs.iter().map(|x| x.1).sum();
                                let req_loads = (tot_need / truck_cap).ceil().max(1.0);
                                let min_per_tour =
                                    (tour_drive_min / req_loads.min(2.5)) + 1.6 * dock_min;
                                let max_loads = (rem_shift_min / min_per_tour.max(20.0)).max(1.0);
                                let mut avail_payload = (req_loads.min(max_loads)) * truck_cap;

                                if trial_idx == 0 && day_idx == 0 {
                                    let mut pts = Vec::with_capacity(ordered_vrp.len() + 2);
                                    let mut nms = Vec::with_capacity(ordered_vrp.len() + 2);
                                    pts.push(depot_pos);
                                    nms.push("RSA Depot".to_string());
                                    for &vn in &ordered_vrp {
                                        pts.push(sites[vn - 1].pos);
                                        nms.push(sites[vn - 1].name.clone());
                                    }
                                    pts.push(depot_pos);
                                    nms.push("RSA Depot".to_string());
                                    day1_tours.push(ScheduledConvoyTour {
                                        commodity: comm_kind,
                                        stop_positions: pts,
                                        stop_names: nms,
                                        round_trip_min: tour_drive_min
                                            + (ordered_vrp.len() as f64) * dock_min,
                                        payload_delivered: avail_payload.min(tot_need),
                                        via_helicopter_airdrop: false,
                                    });
                                }

                                // Allocate `avail_payload` in descending deprivation priority (`ground_needs` is already urgency-sorted!)
                                for &(si, need) in &ground_needs {
                                    if avail_payload <= 1e-6 {
                                        break;
                                    }
                                    let give = need.min(avail_payload);
                                    inv[si][c_idx] =
                                        (inv[si][c_idx] + give).min(tank_cap[si][c_idx] * 1.15);
                                    d_deliv_today[c_idx] += give;
                                    avail_payload -= give;
                                }
                            }
                        }
                    }

                    // D. Consume 24h demand & tally stockout hours + Holguín-Veras deprivation cost!
                    for (si, s) in sites.iter().enumerate() {
                        let burn = daily_burn[si][c_idx];
                        if burn <= 1e-9 {
                            continue;
                        }
                        if inv[si][c_idx] >= burn {
                            inv[si][c_idx] -= burn;
                        } else {
                            let deficit = burn - inv[si][c_idx];
                            inv[si][c_idx] = 0.0;
                            let so_hrs = ((deficit / burn) * 24.0).clamp(0.0, 24.0);
                            site_so_hrs[si][c_idx] += so_hrs;

                            let aff_pop = match comm_kind {
                                ReliefCommodityKind::MedicalOxygenKiloLiters => {
                                    if s.kind == DemandNodeKind::TraumaBurnHospital {
                                        (s.base_pop_or_beds * 0.22).max(4.0)
                                    } else {
                                        (s.base_pop_or_beds * 0.04).max(2.0)
                                    }
                                }
                                _ => s.base_pop_or_beds,
                            };

                            match comm_kind {
                                ReliefCommodityKind::PotableWaterLiters => {
                                    let ph = so_hrs * aff_pop;
                                    water_so_phrs += ph;
                                    d_w_so += ph;
                                }
                                ReliefCommodityKind::GeneratorDieselLiters => {
                                    diesel_bo_hrs += so_hrs;
                                    d_d_bo += so_hrs;
                                }
                                ReliefCommodityKind::MreRationsMeals => {
                                    mre_so_phrs += so_hrs * aff_pop;
                                }
                                ReliefCommodityKind::MedicalOxygenKiloLiters => {
                                    let oph = so_hrs * aff_pop;
                                    o2_so_phrs += oph;
                                    d_o2_so += oph;
                                }
                            }

                            let dep_c = holguin_veras_deprivation_cost(comm_kind, so_hrs, aff_pop);
                            site_dep_cost[si] += dep_c;
                            tot_dep_cost += dep_c;
                        }
                    }
                }

                for c_idx in 0..4 {
                    tot_deliv[c_idx] += d_deliv_today[c_idx];
                }
                day_br_pct.push(open_br_pct);
                day_pwr_pct.push(if grid_ok_today { 100.0 } else { 0.0 });
                day_wtr_pct.push(wtr_rest_pct);
                day_deliv.push(d_deliv_today);
                day_w_so.push(d_w_so);
                day_d_bo.push(d_d_bo);
                day_o2_so.push(d_o2_so);
                day_cum_dep.push(tot_dep_cost);
            }

            let tot_site_comm_hrs = (n_days as f64) * 24.0 * 4.0;
            let site_sec_pct: Vec<f64> = (0..n_sites)
                .map(|si| {
                    let so_sum: f64 = site_so_hrs[si].iter().sum();
                    ((1.0 - so_sum / tot_site_comm_hrs) * 100.0).clamp(0.0, 100.0)
                })
                .collect();

            ResupTrialOut {
                tot_water_kl: tot_deliv[0] / 1000.0,
                tot_diesel_kl: tot_deliv[1] / 1000.0,
                tot_mre_k: tot_deliv[2] / 1000.0,
                tot_o2_kl: tot_deliv[3],
                heli_sorties,
                water_so_phrs,
                diesel_bo_hrs,
                mre_so_phrs,
                o2_so_phrs,
                tot_dep_cost,
                cw_savings_pct: if cw_sav_cnt > 0 {
                    cw_sav_sum / (cw_sav_cnt as f64)
                } else {
                    0.0
                },
                site_so_hrs,
                site_sec_pct,
                site_dep_cost,
                day_br_pct,
                day_pwr_pct,
                day_wtr_pct,
                day_deliv,
                day_w_so,
                day_d_bo,
                day_o2_so,
                day_cum_dep,
                day1_tours,
            }
        })
        .collect();

    let tf = trials as f64;
    let representative_day1_tours = trial_outs
        .first()
        .map(|t| t.day1_tours.clone())
        .unwrap_or_default();

    let mut site_rows = Vec::with_capacity(n_sites);
    for (si, s) in sites.iter().enumerate() {
        site_rows.push(SiteResupplyTelemetryRow {
            site_id: si,
            name: s.name.clone(),
            site_kind: s.kind,
            world_pos: s.pos,
            occupants_or_beds: s.base_pop_or_beds,
            peak_daily_water_l: s.base_pop_or_beds * 15.0,
            peak_daily_diesel_l: s.diesel_burn_l_per_day_outage,
            peak_daily_mre_meals: s.base_pop_or_beds
                * if s.kind == DemandNodeKind::MassCareShelter {
                    2.5
                } else {
                    2.2
                },
            peak_daily_oxygen_kl: s.oxygen_kl_per_day,
            mean_water_stockout_hrs: trial_outs.iter().map(|t| t.site_so_hrs[si][0]).sum::<f64>()
                / tf,
            mean_diesel_blackout_hrs: trial_outs.iter().map(|t| t.site_so_hrs[si][1]).sum::<f64>()
                / tf,
            mean_mre_stockout_hrs: trial_outs.iter().map(|t| t.site_so_hrs[si][2]).sum::<f64>()
                / tf,
            mean_oxygen_stockout_hrs: trial_outs.iter().map(|t| t.site_so_hrs[si][3]).sum::<f64>()
                / tf,
            service_security_pct: trial_outs.iter().map(|t| t.site_sec_pct[si]).sum::<f64>() / tf,
            expected_deprivation_cost: trial_outs.iter().map(|t| t.site_dep_cost[si]).sum::<f64>()
                / tf,
        });
    }

    let sys_sec_pct = site_rows.iter().map(|r| r.service_security_pct).sum::<f64>()
        / (n_sites.max(1) as f64);

    let mut daily_timeline = Vec::with_capacity(n_days);
    for di in 0..n_days {
        daily_timeline.push(DailyResupplyPoint {
            day: di + 1,
            open_bridges_pct: trial_outs.iter().map(|t| t.day_br_pct[di]).sum::<f64>() / tf,
            grid_power_restored_pct: trial_outs.iter().map(|t| t.day_pwr_pct[di]).sum::<f64>() / tf,
            water_mains_restored_pct: trial_outs.iter().map(|t| t.day_wtr_pct[di]).sum::<f64>()
                / tf,
            water_delivered_kl: trial_outs
                .iter()
                .map(|t| t.day_deliv[di][0] / 1000.0)
                .sum::<f64>()
                / tf,
            diesel_delivered_kl: trial_outs
                .iter()
                .map(|t| t.day_deliv[di][1] / 1000.0)
                .sum::<f64>()
                / tf,
            mre_delivered_k: trial_outs
                .iter()
                .map(|t| t.day_deliv[di][2] / 1000.0)
                .sum::<f64>()
                / tf,
            oxygen_delivered_kl: trial_outs.iter().map(|t| t.day_deliv[di][3]).sum::<f64>() / tf,
            water_stockout_person_hrs: trial_outs.iter().map(|t| t.day_w_so[di]).sum::<f64>() / tf,
            diesel_blackout_site_hrs: trial_outs.iter().map(|t| t.day_d_bo[di]).sum::<f64>() / tf,
            oxygen_shortfall_patient_hrs: trial_outs.iter().map(|t| t.day_o2_so[di]).sum::<f64>()
                / tf,
            cumulative_deprivation_cost: trial_outs.iter().map(|t| t.day_cum_dep[di]).sum::<f64>()
                / tf,
        });
    }

    let mut parcel_security_pct = Vec::with_capacity(n_parcels);
    let mut parcel_deprivation_idx = Vec::with_capacity(n_parcels);
    for pi in 0..n_parcels {
        let sh_s = site_rows
            .get(parcel_shelt_idx[pi])
            .map(|r| r.service_security_pct)
            .unwrap_or(100.0);
        let hp_s = site_rows
            .get(parcel_hosp_idx[pi])
            .map(|r| r.service_security_pct)
            .unwrap_or(100.0);
        let sh_d = site_rows
            .get(parcel_shelt_idx[pi])
            .map(|r| r.expected_deprivation_cost)
            .unwrap_or(0.0);
        let hp_d = site_rows
            .get(parcel_hosp_idx[pi])
            .map(|r| r.expected_deprivation_cost)
            .unwrap_or(0.0);
        parcel_security_pct.push(0.60 * sh_s + 0.40 * hp_s);
        parcel_deprivation_idx.push(((0.60 * sh_d + 0.40 * hp_d) / 12_000.0).clamp(0.0, 1.0));
    }

    CommodityResupplyReport {
        config: cfg.clone(),
        depot_world_pos: depot_pos,
        total_demand_sites: n_sites,
        total_water_delivered_kl: trial_outs.iter().map(|t| t.tot_water_kl).sum::<f64>() / tf,
        total_diesel_delivered_kl: trial_outs.iter().map(|t| t.tot_diesel_kl).sum::<f64>() / tf,
        total_mre_delivered_k: trial_outs.iter().map(|t| t.tot_mre_k).sum::<f64>() / tf,
        total_oxygen_delivered_kl: trial_outs.iter().map(|t| t.tot_o2_kl).sum::<f64>() / tf,
        expected_helicopter_sling_sorties: trial_outs.iter().map(|t| t.heli_sorties).sum::<f64>()
            / tf,
        expected_water_stockout_person_hrs: trial_outs.iter().map(|t| t.water_so_phrs).sum::<f64>()
            / tf,
        expected_diesel_blackout_site_hrs: trial_outs.iter().map(|t| t.diesel_bo_hrs).sum::<f64>()
            / tf,
        expected_mre_shortfall_person_hrs: trial_outs.iter().map(|t| t.mre_so_phrs).sum::<f64>()
            / tf,
        expected_oxygen_shortfall_patient_hrs: trial_outs.iter().map(|t| t.o2_so_phrs).sum::<f64>()
            / tf,
        system_supply_security_pct: sys_sec_pct,
        total_deprivation_social_cost: trial_outs.iter().map(|t| t.tot_dep_cost).sum::<f64>() / tf,
        mean_convoy_tour_savings_pct: trial_outs.iter().map(|t| t.cw_savings_pct).sum::<f64>() / tf,
        daily_timeline,
        site_rows,
        representative_day1_tours,
        parcel_security_pct,
        parcel_deprivation_idx,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

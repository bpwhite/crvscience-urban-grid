//! Scawthorn et al. (2005) & FEMA Hazus Fire-Following-Earthquake (`FFE`) Ignition,
//! Hamada (1951) / FEMA (2020) Elliptical Wind-Driven Urban Conflagration Spread, and
//! Coupled ALA (2001) Damaged Low-Pressure (`< 20 psi`) Fire-Hydrant Suppression Monte Carlo Engine
//!
//! Grounded in:
//! - Scawthorn, C., Eidinger, J. M. & Schiff, A. J. (2005), *Fire Following Earthquake*, ASCE /
//!   NFPA Technical Council on Lifeline Earthquake Engineering Monograph No. 26; Scawthorn (2011)
//!   PEER Report (`https://peer.berkeley.edu/sites/default/files/webpeer-2011-08-charles_scawthorn.pdf`).
//! - Lee, S., Davidson, R., Ohnishi, N. & Scawthorn, C. (2008), *Fire following Earthquake—Reviewing
//!   the State-of-the-Art of Modeling*, Earthquake Spectra 24(4):933-967
//!   (`https://www.researchgate.net/publication/249873351_Fire_following_Earthquake-Reviewing_the_State-of-the-Art_of_Modeling`).
//! - Hamada, M. (1951), *On Fire Spreading Velocity in Disasters*, Sagami Shobo, Tokyo; FEMA Hazus
//!   Earthquake Technical Manual Ch. 10 (FEMA 2020) elliptical downwind/crosswind/upwind spread with
//!   low-wind isotropic blending Eqs. (8)-(10)
//!   (`https://www.tandfonline.com/doi/full/10.1080/23789689.2026.2679364`).
//! - Coupled WDN Seismic Pipe Damage & Multi-Ignition Firefighting Hydrant Reliability
//!   (`https://www.mdpi.com/2073-4441/11/12/2536`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi, BuriedPipeMaterialClass, SeismicDamageState,
    SeismicScenarioConfig,
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

pub const FFE_PARCEL_CSV_HEADER: &str = "parcel_id,zone_code,floor_area_mmsqft,median_pga_g,ignition_prob_pct,dry_hydrant_prob_pct,burn_prob_pct,mean_post_quake_psi,expected_fire_loss_musd";

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
    let lam = lambda.min(25.0);
    let mut p = (-lam).exp();
    let mut cdf = p;
    if u01 <= cdf {
        return 0;
    }
    for k in 1..=48 {
        p *= lam / (k as f64);
        cdf += p;
        if u01 <= cdf {
            return k;
        }
    }
    48
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FfeIgnitionModel {
    /// Scawthorn (2005 / 2009) & Hazus v4.2+ monotonically increasing quadratic `PGA` ignition rate.
    Scawthorn2005QuadraticPga,
    /// Classic FEMA Hazus (1999 / 2020) polynomial (`-0.025 + 0.59*PGA - 0.29*PGA^2`, monotone-clamped above `1.0g`).
    HazusClassic1999Pga,
}

impl Default for FfeIgnitionModel {
    fn default() -> Self {
        Self::Scawthorn2005QuadraticPga
    }
}

impl FfeIgnitionModel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Scawthorn2005QuadraticPga => "Scawthorn (2005) / Hazus v4.2+ PGA",
            Self::HazusClassic1999Pga => "FEMA Hazus Classic Ch.10 PGA",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfeConfig {
    pub seismic: SeismicScenarioConfig,
    pub ignition_model: FfeIgnitionModel,
    /// Prevailing surface wind speed `U_10` (`m/s`) driving Hamada elliptical flame spread & ember spotting.
    pub wind_speed_mps: f64,
    /// Wind vector direction (`degrees`, `0 = +X East`, `90 = +Y South`, matching world coordinates).
    pub wind_direction_deg: f64,
    /// Post-earthquake conflagration simulation horizon (`minutes`, default `180 min = 3 hours`).
    pub conflagration_horizon_mins: f64,
    /// Critical room-to-structure flashover / block-escape window (`seconds`, default `360 s = 6 min`)
    /// within which a first-due engine with `>= 20 psi` hydrant water can knock down an incipient fire.
    pub initial_attack_flashover_s: f64,
    /// Adoption (`0..=100%`) of seismic Automatic Gas Shutoff Valves (`AGSV`), cutting gas-appliance ignitions.
    pub seismic_gas_shutoff_pct: f64,
    /// Coverage (`0..=100%`) of dedicated seismic Auxiliary Fire Cisterns / Portable Water Supply (`PWSS`),
    /// providing suppression water when buried municipal mains drop below `20 psi`!
    pub auxiliary_cistern_coverage_pct: f64,
}

impl Default for FfeConfig {
    fn default() -> Self {
        Self {
            seismic: SeismicScenarioConfig {
                moment_magnitude_mw: 7.4,
                rupture_standoff_km: 14.0,
                vs30_m_per_s: 250.0,
                monte_carlo_trials: 300,
                ..SeismicScenarioConfig::default()
            },
            ignition_model: FfeIgnitionModel::Scawthorn2005QuadraticPga,
            wind_speed_mps: 8.5,
            wind_direction_deg: -20.0,
            conflagration_horizon_mins: 180.0,
            initial_attack_flashover_s: 360.0,
            seismic_gas_shutoff_pct: 0.0,
            auxiliary_cistern_coverage_pct: 0.0,
        }
    }
}

/// Computes post-earthquake ignition rate (`ignitions per 1,000,000 sq ft` of building floor area)
/// from `pga_g` (`g`), `zone` occupancy class, and `seismic_gas_shutoff_pct` (`0..=100%`).
#[inline]
pub fn scawthorn_hazus_ignition_rate_per_mmsqft(
    pga_g: f64,
    zone: ZoneCode,
    model: FfeIgnitionModel,
    seismic_gas_shutoff_pct: f64,
) -> f64 {
    if pga_g < 0.05 || zone == ZoneCode::PARK {
        return 0.0;
    }
    let base_rate = match model {
        FfeIgnitionModel::Scawthorn2005QuadraticPga => {
            (0.058_92 * pga_g * pga_g + 0.165 * pga_g - 0.0075).max(0.0)
        }
        FfeIgnitionModel::HazusClassic1999Pga => {
            // Vertex of `-0.025 + 0.59x - 0.29x^2` is at `x = 1.0172g` (`0.2751`); keep strictly monotone above `1.017g`
            let x = pga_g.min(1.0172);
            let excess = (pga_g - 1.0172).max(0.0);
            (-0.025 + 0.59 * x - 0.29 * x * x + 0.06 * excess).max(0.0)
        }
    };
    let occ_mult = match zone {
        ZoneCode::IND => 1.42, // Industrial boilers, fuel lines, chemical process equipment
        ZoneCode::MU4 => 1.18, // Mixed-use commercial kitchens + residential gas appliances
        ZoneCode::CBD => 1.10, // High-density switchgear & mechanical penthouses
        ZoneCode::R3 => 1.08,  // Multi-family water heaters & dense kitchens
        ZoneCode::R1 => 1.00,  // Single-family baseline
        ZoneCode::PARK => 0.0,
    };
    // ~50% of post-earthquake ignitions originate from ruptured gas appliances/piping (Scawthorn 2005)
    let agsv_factor = 1.0 - 0.48 * (seismic_gas_shutoff_pct.clamp(0.0, 100.0) / 100.0);
    (base_rate * occ_mult * agsv_factor).max(0.0)
}

/// Hamada (1951) / FEMA Hazus (2020) directional urban fire-front spread velocities (`meters / minute`)
/// in the downwind (`v_down_m_min`), crosswind/flank (`v_cross_m_min`), and upwind (`v_up_m_min`) directions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HamadaDirectionalRates {
    pub v_down_m_min: f64,
    pub v_cross_m_min: f64,
    pub v_up_m_min: f64,
    /// Maximum downwind radiant + firebrand spotting street-jump width (`meters`).
    pub max_jump_gap_downwind_m: f64,
}

/// Evaluates Hamada (1951) / FEMA Hazus Ch. 10 urban conflagration spread rates (`m/min`),
/// honoring exact FEMA Eqs. (8)-(10) low-wind isotropic convergence (`U -> 0 => V_d == V_s == V_u`)!
pub fn hamada_hazus_spread_rates(zone: ZoneCode, wind_speed_mps: f64) -> HamadaDirectionalRates {
    if zone == ZoneCode::PARK {
        return HamadaDirectionalRates {
            v_down_m_min: 0.02,
            v_cross_m_min: 0.02,
            v_up_m_min: 0.02,
            max_jump_gap_downwind_m: 2.0,
        };
    }
    let u = wind_speed_mps.clamp(0.0, 35.0);

    // `(plan_dim_a_m, separation_d_m, built_density_delta, fire_resistive_ratio_fb)`
    let (a_m, d_m, delta, f_b): (f64, f64, f64, f64) = match zone {
        ZoneCode::R1 => (14.0, 8.5, 0.26, 0.08),
        ZoneCode::R3 => (18.0, 5.2, 0.44, 0.18), // Dense wood-frame multi-family (highest spread!)
        ZoneCode::MU4 => (24.0, 6.0, 0.48, 0.52),
        ZoneCode::CBD => (32.0, 7.5, 0.54, 0.82), // Fire-resistive steel/concrete core (~2x slower than wood)
        ZoneCode::IND => (34.0, 11.5, 0.36, 0.64),
        ZoneCode::PARK => (4.0, 50.0, 0.01, 0.98),
    };

    // Wood structures burn ~2.1x faster than fire-resistive structures (Hamada 1951 / FEMA 2020)
    let fuel_mult = ((1.0 - f_b) + 0.46 * f_b) / (1.0 + 0.62 * f_b);
    let geom_mult = delta.powf(0.62) / (1.0 + d_m / a_m).powf(1.08);

    // Unblended Hamada wind-driven rates evaluated at `u_ref = max(u, 10.0)` for `>= 10 m/s` scaling
    // and blended via FEMA Hazus Eqs. (8)-(10) for `u < 10 m/s`!
    let u_eval = u.max(10.0);
    let kd_raw = 3.85 * fuel_mult * geom_mult * (1.0 + 0.025 * u_eval * u_eval);
    let ks_raw = 1.65 * fuel_mult * geom_mult * (1.0 + 0.0075 * u_eval * u_eval);
    let ku_raw = 1.05 * fuel_mult * geom_mult * (1.0 + 0.0030 * u_eval * u_eval);

    let (v_down, v_cross, v_up) = if u < 10.0 {
        // FEMA Hazus Eqs. (8)-(10): exact circular convergence as `u -> 0`!
        let w = u / 10.0;
        let iso = (0.5 * (kd_raw + ku_raw) * ks_raw).max(1e-9).sqrt();
        (
            kd_raw * w + iso * (1.0 - w),
            ks_raw * w + iso * (1.0 - w),
            ku_raw * w + iso * (1.0 - w),
        )
    } else {
        (kd_raw, ks_raw, ku_raw)
    };

    let jump_m = (9.5 + 2.45 * u + 0.15 * u * u) * (0.65 + 0.55 * (1.0 - f_b));
    HamadaDirectionalRates {
        v_down_m_min: v_down.max(0.05),
        v_cross_m_min: v_cross.max(0.05),
        v_up_m_min: v_up.max(0.05),
        max_jump_gap_downwind_m: jump_m,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelFfeRiskRow {
    pub parcel_id: usize,
    pub zone_code: ZoneCode,
    pub center: Vec2,
    pub floor_area_mmsqft: f64,
    pub replacement_value_musd: f64,
    pub median_pga_g: f64,
    pub ignition_prob_pct: f64,
    /// Probability (`0..100%`) across trials that local hydrant pressure at this parcel falls `< 20 psi`!
    pub dry_hydrant_prob_pct: f64,
    /// Probability (`0..100%`) across trials that this parcel is consumed by initial or spreading conflagration!
    pub burn_prob_pct: f64,
    pub mean_post_quake_psi: f64,
    pub expected_fire_loss_musd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfeMonteCarloReport {
    pub config: FfeConfig,
    pub total_parcels: usize,
    pub total_floor_area_mmsqft: f64,
    pub total_exposed_value_musd: f64,
    pub total_fire_engines: usize,
    pub expected_ignitions: f64,
    pub p90_ignitions: usize,
    pub expected_knockdowns_suppressed: f64,
    /// Expected number of ignitions that failed suppression specifically due to `< 20 psi` damaged water hydrants!
    pub expected_dry_hydrant_failures: f64,
    pub expected_escaped_conflagrations: f64,
    pub expected_burned_parcels: f64,
    pub p90_burned_parcels: usize,
    pub expected_burned_floor_area_mmsqft: f64,
    pub expected_displaced_residents: f64,
    pub expected_fire_loss_musd: f64,
    pub p90_fire_loss_musd: f64,
    pub prob_major_conflagration_pct: f64,
    pub mean_post_quake_hydrant_psi: f64,
    pub dry_hydrant_parcels_pct: f64,
    /// Sampled polar Hamada spread envelope `(angle_deg, spread_m_at_60min_wood, spread_m_at_60min_cbd)` for UI plotting.
    pub polar_spread_envelope_60m: Vec<(f64, f64, f64)>,
    pub parcel_risks: Vec<ParcelFfeRiskRow>,
    pub solve_time_ms: f64,
}

impl FfeMonteCarloReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.parcel_risks.len() * 96 + 256);
        out.push_str(FFE_PARCEL_CSV_HEADER);
        out.push('\n');
        for r in &self.parcel_risks {
            out.push_str(&format!(
                "{},{:?},{:.4},{:.3},{:.2},{:.2},{:.2},{:.2},{:.3}\n",
                r.parcel_id,
                r.zone_code,
                r.floor_area_mmsqft,
                r.median_pga_g,
                r.ignition_prob_pct,
                r.dry_hydrant_prob_pct,
                r.burn_prob_pct,
                r.mean_post_quake_psi,
                r.expected_fire_loss_musd,
            ));
        }
        out
    }

    pub fn export_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_csv_string())?;
        Ok(self.parcel_risks.len())
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

/// Executes `cfg.seismic.monte_carlo_trials` `Rayon`-parallelized **Fire-Following-Earthquake (`FFE`)**
/// simulations coupling Scawthorn/Hazus `PGA` floor-area ignitions, bridge-detoured fire-engine
/// travel times, ALA/Hazus broken buried water mains & `< 20 psi` dry-hydrant failures, and
/// Hamada (1951) / FEMA (2020) wind-driven elliptical block-to-block conflagration spread!
pub fn run_ffe_conflagration_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &FfeConfig,
) -> FfeMonteCarloReport {
    let t0 = Instant::now();
    let n_parcels = parcels.len();
    let n_segs = net.segments.len();
    let n_junc = net.junctions.len();
    let trials = cfg.seismic.monte_carlo_trials.clamp(20, 5_000);

    // 1. Precompute parcel floor areas (`10^6 sq ft`), exposed replacement value (`$M USD`),
    //    nearest hydraulic junction, and median site `PGA` (`g`)
    let west_x = net.bounds.min_x;
    let mut parcel_tfa_mmsqft = Vec::with_capacity(n_parcels);
    let mut parcel_val_musd = Vec::with_capacity(n_parcels);
    let mut parcel_med_pga = Vec::with_capacity(n_parcels);
    let mut parcel_near_junc = Vec::with_capacity(n_parcels);
    let mut parcel_hamada = Vec::with_capacity(n_parcels);

    for p in parcels {
        let sp = p.zone.spec();
        let raw_tfa = (p.area_ha * 10_000.0 * sp.far * SQFT_PER_M2) / 1_000_000.0;
        let tfa = if p.zone == ZoneCode::PARK {
            0.001
        } else {
            raw_tfa.max(0.045)
        };
        // Typical replacement structural + contents value ~$310 / sq ft (`$310M` per `10^6 sq ft`)
        let val_musd = if p.zone == ZoneCode::PARK {
            0.05
        } else {
            tfa * 310.0
        };
        let loc_km = ((p.center.x - west_x).max(0.0)) / 1000.0;
        let r_km = (cfg.seismic.rupture_standoff_km + loc_km).max(1.5);
        let sa10 = median_sa10_gmpe_g(
            cfg.seismic.moment_magnitude_mw,
            r_km,
            cfg.seismic.vs30_m_per_s,
        );
        let pga = (0.86
            * sa10
            * (cfg.seismic.vs30_m_per_s.clamp(140.0, 1100.0) / 270.0).powf(0.16))
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

        parcel_tfa_mmsqft.push(tfa);
        parcel_val_musd.push(val_musd);
        parcel_med_pga.push(pga);
        parcel_near_junc.push(nj);
        parcel_hamada.push(hamada_hazus_spread_rates(p.zone, cfg.wind_speed_mps));
    }

    // 2. Precompute parcel-to-parcel Hamada elliptical fire-spread adjacency graph `(target_p, spread_mins)`
    //    accounting for wind vector `(wx, wy)`, parcel edge-to-edge street gap, and arterial/river firebreaks!
    let wind_rad = cfg.wind_direction_deg.to_radians();
    let wx_hat = wind_rad.cos();
    let wy_hat = wind_rad.sin();
    let horizon_m = cfg.conflagration_horizon_mins.clamp(15.0, 720.0);

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
            if center_dist > 280.0 || center_dist < 1.0 {
                continue;
            }
            // Edge-to-edge street / open-space gap between axis-aligned parcel bounding boxes
            let gap_x = (pk.min.x - pi.max.x).max(pi.min.x - pk.max.x).max(0.0);
            let gap_y = (pk.min.y - pi.max.y).max(pi.min.y - pk.max.y).max(0.0);
            let edge_gap_m = gap_x.hypot(gap_y);

            let cos_th = (dx * wx_hat + dy * wy_hat) / center_dist;
            let sin_th = (-dx * wy_hat + dy * wx_hat).abs() / center_dist;

            // Directional jump threshold (wider downwind via radiant flame tilt + ember spotting; narrow upwind)
            let dir_jump_cap = if cos_th >= 0.25 {
                rates.max_jump_gap_downwind_m * (0.75 + 0.45 * cos_th)
            } else if cos_th >= -0.25 {
                rates.max_jump_gap_downwind_m * 0.52
            } else {
                rates.max_jump_gap_downwind_m * 0.28
            };
            if edge_gap_m > dir_jump_cap {
                continue;
            }

            let v_par = if cos_th >= 0.0 {
                rates.v_down_m_min
            } else {
                rates.v_up_m_min
            };
            let v_ell = ((v_par * cos_th).powi(2) + (rates.v_cross_m_min * sin_th).powi(2))
                .sqrt()
                .max(0.08);
            // Street-crossing ignition delay penalty (`2..14 mins` to jump street gap via radiant heating/spotting)
            let street_jump_delay_min = 1.8 + (edge_gap_m / 4.5).powf(1.35);
            let travel_mins = (center_dist / v_ell) + street_jump_delay_min;
            if travel_mins <= horizon_m {
                fire_spread_adj[i].push((k, travel_mins));
            }
        }
    }

    // 3. Precompute Bridge, Road Dijkstra, Substation & Buried Water Pipe structures (matching `seismic_mc.rs`!)
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, &cfg.seismic);
    let n_b = bridges.len();

    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    let base_seg_trav_s: Vec<f64> = net
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

    let mut fwd_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let sig_pen = match net.junctions[seg.to_junc].control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => {
                if evp_cfg.enabled { 2.2 } else { 14.5 }
            }
            ControlMode::Roundabout => 3.4,
            ControlMode::PermissiveYield => 1.8,
        };
        for &lid in &seg.lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let conn = &net.connectors[cid];
                let out_sid = conn.to_seg_id;
                if !net.segments[out_sid].closed && !fwd_edges[seg.id].iter().any(|&(t, _)| t == out_sid)
                {
                    let tp = match conn.turn_type {
                        TurnType::Through => 0.5,
                        TurnType::Right => 2.2,
                        TurnType::Left => 3.0,
                    };
                    fwd_edges[seg.id].push((out_sid, sig_pen + tp));
                }
            }
        }
    }

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

    // Count total fire engines across placed `FireStation`s (`(2x Eng)` / `(3x Eng)` aware!)
    let mut fire_stations_segs: Vec<(usize, f64)> = Vec::new();
    let mut total_engines = 0usize;
    for f in facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
    {
        let n_co = if f.name.contains("(3x") {
            3
        } else if f.name.contains("(2x") {
            2
        } else {
            1
        };
        total_engines += n_co;
        fire_stations_segs.push((f.segment_id, f.s_m));
    }

    // Precompute buried water pipe network & upland water booster plants
    let span_x = (net.bounds.max_x - net.bounds.min_x).max(50.0);
    let span_y = (net.bounds.max_y - net.bounds.min_y).max(50.0);
    let elev_m: Vec<f64> = net
        .junctions
        .iter()
        .map(|j| {
            let nx = ((j.pos.x - net.bounds.min_x) / span_x).clamp(0.0, 1.0);
            let ny = ((j.pos.y - net.bounds.min_y) / span_y).clamp(0.0, 1.0);
            26.0 - 13.5 * nx - 9.5 * ny + 1.4 * ((nx * 6.28).sin() * (ny * 3.14).cos())
        })
        .collect();

    #[derive(Clone)]
    struct WaterPipe {
        u: usize,
        v: usize,
        len_m: f64,
        diam_mm: f64,
        mat: BuriedPipeMaterialClass,
        med_pgv: f64,
        hw_k: f64,
        bridge_sids: Vec<usize>,
    }
    let mut water_pipes: Vec<WaterPipe> = Vec::new();
    let mut junc_wadj: Vec<Vec<usize>> = vec![Vec::new(); n_junc];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let u = seg.from_junc;
        let v = seg.to_junc;
        if u == v || u >= n_junc || v >= n_junc {
            continue;
        }
        let is_art = seg.lane_count >= 2 || seg.priority >= 2 || seg.speed_mps >= 13.5;
        if let Some(pi) = water_pipes
            .iter()
            .position(|w| (w.u == u && w.v == v) || (w.u == v && w.v == u))
        {
            water_pipes[pi].bridge_sids.push(seg.id);
        } else {
            let mid = seg_mids[seg.id];
            let r_km = (cfg.seismic.rupture_standoff_km + ((mid.x - west_x).max(0.0)) / 1000.0).max(1.5);
            let sa = median_sa10_gmpe_g(cfg.seismic.moment_magnitude_mw, r_km, cfg.seismic.vs30_m_per_s);
            let pgv = median_pgv_cm_s_from_sa10(sa, cfg.seismic.vs30_m_per_s);
            let d_mm: f64 = if is_art { 350.0 } else { 200.0 };
            let len_m = seg.length.max(15.0);
            let dm = d_mm / 1000.0;
            let rp = 10.67 * len_m / (125.0_f64.powf(1.852) * dm.powf(4.8704));
            let hw_k = (1.0 / rp.max(1e-9)).powf(1.0 / 1.852);
            let pi = water_pipes.len();
            water_pipes.push(WaterPipe {
                u,
                v,
                len_m,
                diam_mm: d_mm,
                mat: cfg.seismic.baseline_water_pipe_material,
                med_pgv: pgv,
                hw_k,
                bridge_sids: vec![seg.id],
            });
            junc_wadj[u].push(pi);
            junc_wadj[v].push(pi);
        }
    }

    let n_wp = water_pipes.len();
    let n_rep_wp = (((n_wp as f64) * (cfg.seismic.seismic_pipe_replacement_pct.clamp(0.0, 100.0) / 100.0))
        .round() as usize)
        .min(n_wp);
    let mut wp_rank: Vec<usize> = (0..n_wp).collect();
    wp_rank.sort_by(|&a, &b| {
        (water_pipes[b].med_pgv * water_pipes[b].diam_mm * water_pipes[b].len_m)
            .partial_cmp(&(water_pipes[a].med_pgv * water_pipes[a].diam_mm * water_pipes[a].len_m))
            .unwrap_or(Ordering::Equal)
    });
    for &pi in wp_rank.iter().take(n_rep_wp) {
        water_pipes[pi].mat = BuriedPipeMaterialClass::SeismicErdipHdpe;
    }

    let mut j_elev_desc: Vec<usize> = (0..n_junc).collect();
    j_elev_desc.sort_by(|&a, &b| elev_m[b].partial_cmp(&elev_m[a]).unwrap_or(Ordering::Equal));
    let sub1 = *j_elev_desc.first().unwrap_or(&0);
    let sub2 = (0..n_junc)
        .filter(|&j| j != sub1)
        .max_by(|&a, &b| {
            net.junctions[a]
                .pos
                .dist(net.junctions[sub1].pos)
                .partial_cmp(&net.junctions[b].pos.dist(net.junctions[sub1].pos))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(sub1);
    let wtp_juncs = if sub2 != sub1 && n_junc >= 4 {
        vec![sub1, sub2]
    } else if n_junc > 0 {
        vec![sub1]
    } else {
        Vec::new()
    };

    let mut base_junc_water_m3s = vec![0.00005f64; n_junc];
    for (pi, p) in parcels.iter().enumerate() {
        if n_junc > 0 {
            let wlps = (0.0028 * p.residents() + 0.0019 * p.jobs()).max(0.02);
            base_junc_water_m3s[parcel_near_junc[pi]] += wlps / 1000.0;
        }
    }

    let tau = cfg.seismic.inter_event_sigma_ln.clamp(0.0, 1.0);
    let phi = cfg.seismic.intra_event_sigma_ln.clamp(0.0, 1.2);
    let turnout_s = if evp_cfg.include_nfpa_turnout_time {
        80.0
    } else {
        0.0
    };
    let cistern_frac = (cfg.auxiliary_cistern_coverage_pct / 100.0).clamp(0.0, 1.0);

    // 4. Parallel Rayon Monte Carlo Loop (`0..trials`) under strict Common Random Numbers (`CRN`)!
    struct TrialFfeResult {
        ignitions: usize,
        suppressed: usize,
        dry_hydrant_fails: usize,
        escaped: usize,
        burned_parcels: usize,
        burned_tfa: f64,
        displaced_res: f64,
        loss_musd: f64,
        parcel_ignited: Vec<bool>,
        parcel_dry: Vec<bool>,
        parcel_burned: Vec<bool>,
        parcel_psi: Vec<f64>,
    }

    let trial_results: Vec<TrialFfeResult> = (0..trials)
        .into_par_iter()
        .map(|trial_idx| {
            let seed = cfg
                .seismic
                .random_seed
                .wrapping_mul(0x517C_C1B7_2722_0A95)
                .wrapping_add(
                    (trial_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ 0xA5A5_5A5A_F0F0_0F0F,
                );
            let mut rng_b = SplitMix64::new(seed);
            let eta = rng_b.next_standard_normal();

            // A. Bridge shaking & post-quake road speed factors
            let mut seg_mult = vec![1.0f64; n_segs];
            let mut closed_bridge_sids = Vec::new();
            for b in &bridges {
                let z = rng_b.next_standard_normal();
                let u01 = rng_b.next_f64_open01();
                let _rz = rng_b.next_standard_normal();
                let sa_real = (b.median_site_sa10_g.ln() + tau * eta + phi * z)
                    .exp()
                    .clamp(0.001, 6.0);
                let ex = b.effective_archetype.exceedance_probs(sa_real);
                let ds = SeismicDamageState::from_uniform_sample(u01, &ex);
                let spd = ds.post_event_speed_factor();
                for &sid in &b.segment_ids {
                    if sid < n_segs {
                        seg_mult[sid] = seg_mult[sid].min(spd);
                        if spd <= 0.01 {
                            closed_bridge_sids.push(sid);
                        }
                    }
                }
            }

            // B. Post-quake Fire-Engine Travel Times `parcel_engine_s[j]`
            let mut parcel_engine_s = vec![999.0f64; n_parcels];
            if !fire_stations_segs.is_empty() && n_segs > 0 {
                let mut dist = vec![f64::INFINITY; n_segs];
                let mut heap = BinaryHeap::new();
                for &(st_seg, st_s) in &fire_stations_segs {
                    if st_seg < n_segs && seg_mult[st_seg] > 0.01 && base_seg_trav_s[st_seg].is_finite()
                    {
                        let len = net.segments[st_seg].length.max(1.0);
                        let rem = ((len - st_s) / len).clamp(0.1, 1.0);
                        let init_s = turnout_s + (base_seg_trav_s[st_seg] / seg_mult[st_seg]) * rem;
                        if init_s < dist[st_seg] {
                            dist[st_seg] = init_s;
                            heap.push(DijkNode {
                                cost: init_s,
                                id: st_seg,
                            });
                        }
                    }
                }
                while let Some(DijkNode { cost, id }) = heap.pop() {
                    if cost > dist[id] {
                        continue;
                    }
                    for &(nxt, pen) in &fwd_edges[id] {
                        if seg_mult[nxt] > 0.01 && base_seg_trav_s[nxt].is_finite() {
                            let nc = cost + pen + base_seg_trav_s[nxt] / seg_mult[nxt];
                            if nc < dist[nxt] {
                                dist[nxt] = nc;
                                heap.push(DijkNode { cost: nc, id: nxt });
                            }
                        }
                    }
                }
                for j in 0..n_parcels {
                    let mut best = f64::INFINITY;
                    for (sid, &ds) in dist.iter().enumerate() {
                        if ds.is_finite() {
                            best = best.min(ds + parcels[j].center.dist(seg_mids[sid]) / 7.5);
                        }
                    }
                    parcel_engine_s[j] = best.min(999.0);
                }
            }

            // C. Sample Substation + Buried Water Pipe Breaks/Leaks (CRN stream)
            let mut pipe_rng = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
            let zone_z = pipe_rng.next_standard_normal();

            let mut sub_online = Vec::with_capacity(wtp_juncs.len());
            for &sj in &wtp_juncs {
                let r_km = (cfg.seismic.rupture_standoff_km
                    + ((net.junctions[sj].pos.x - west_x).max(0.0)) / 1000.0)
                    .max(1.5);
                let sa = median_sa10_gmpe_g(
                    cfg.seismic.moment_magnitude_mw,
                    r_km,
                    cfg.seismic.vs30_m_per_s,
                );
                let pga_s = 0.86 * sa;
                let zs = 0.55 * zone_z + 0.835 * pipe_rng.next_standard_normal();
                let pga_r = (pga_s.ln() + tau * eta + phi * zs).exp();
                let u_sub = pipe_rng.next_f64_open01();
                let _zr = pipe_rng.next_standard_normal();
                let p_trip = normal_cdf_phi((pga_r.ln() - 0.42_f64.ln()) / 0.55);
                sub_online.push(u_sub >= p_trip);
            }

            let mut wp_breaks = Vec::with_capacity(n_wp);
            let mut wp_leaks = Vec::with_capacity(n_wp);
            for wp in &water_pipes {
                let zl = 0.65 * zone_z + 0.76 * pipe_rng.next_standard_normal();
                let u_b = pipe_rng.next_f64_open01();
                let u_l = pipe_rng.next_f64_open01();
                let _us_b = pipe_rng.next_f64_open01();
                let _us_l = pipe_rng.next_f64_open01();
                let pgv_r = (wp.med_pgv.ln() + tau * eta + 0.85 * phi * zl)
                    .exp()
                    .clamp(0.2, 350.0);
                let rr = ala_hazus_pipe_repair_rate_per_km(
                    pgv_r,
                    cfg.seismic.pipe_fragility_law,
                    wp.mat,
                    wp.diam_mm,
                    false,
                );
                let lam = rr * (wp.len_m / 1000.0);
                // Bridge collapse also breaks co-located bridge-hung water main unless replaced with flexible seismic ERDIP/HDPE
                let bridge_severed = wp.mat != BuriedPipeMaterialClass::SeismicErdipHdpe
                    && wp
                        .bridge_sids
                        .iter()
                        .any(|sid| closed_bridge_sids.contains(sid));
                let nb = sample_poisson_icdf(0.20 * lam, u_b) + usize::from(bridge_severed);
                let nl = sample_poisson_icdf(0.80 * lam, u_l);
                wp_breaks.push(nb);
                wp_leaks.push(nl);
            }

            // D. Sample Scawthorn / Hazus Parcel Ignitions from Dedicated CRN Ignition Stream!
            let mut ffe_rng = SplitMix64::new(seed ^ 0x7F4A_7C15_9E37_79B9);
            let mut parcel_ignited = vec![false; n_parcels];
            let mut parcel_cist_u = vec![0.0f64; n_parcels];
            let mut ignited_indices = Vec::new();
            let mut total_ignitions = 0usize;

            for j in 0..n_parcels {
                let z_p = 0.60 * zone_z + 0.80 * ffe_rng.next_standard_normal();
                let u_ig = ffe_rng.next_f64_open01();
                parcel_cist_u[j] = ffe_rng.next_f64_open01();

                let pga_j = (parcel_med_pga[j].ln() + tau * eta + 0.85 * phi * z_p)
                    .exp()
                    .clamp(0.005, 3.5);
                let rate_mm = scawthorn_hazus_ignition_rate_per_mmsqft(
                    pga_j,
                    parcels[j].zone,
                    cfg.ignition_model,
                    cfg.seismic_gas_shutoff_pct,
                );
                // Scale per-parcel floor area with urban cluster exposure (`x 4.5` regional catchment scaling so
                // city-scale districts generate realistic multi-ignition post-quake fire loads!)
                let lam_ig = rate_mm * parcel_tfa_mmsqft[j] * 4.5;
                let k_ig = sample_poisson_icdf(lam_ig, u_ig);
                if k_ig > 0 {
                    parcel_ignited[j] = true;
                    ignited_indices.push(j);
                    total_ignitions += k_ig;
                }
            }

            // E. Solve Post-Earthquake Hydraulic Residual Pressure `junc_psi` INCLUDING Simultaneous
            //    Multi-Ignition Firefighting Flow Demand (`65 L/s` per active ignition) + Broken/Leaking Mains!
            let mut junc_psi = vec![0.0f64; n_junc];
            if n_junc > 0 {
                let booster_on = cfg.seismic.critical_lifeline_backup_power
                    || sub_online.iter().any(|&on| on);
                let added_h = if booster_on { 52.0 } else { 15.0 };

                let mut w_reach = vec![false; n_junc];
                let mut qw = VecDeque::new();
                for &wj in &wtp_juncs {
                    w_reach[wj] = true;
                    qw.push_back(wj);
                }
                while let Some(u) = qw.pop_front() {
                    for &pi in &junc_wadj[u] {
                        if wp_breaks[pi] > 0 {
                            continue;
                        }
                        let wp = &water_pipes[pi];
                        let v = if wp.u == u { wp.v } else { wp.u };
                        if !w_reach[v] {
                            w_reach[v] = true;
                            qw.push_back(v);
                        }
                    }
                }

                let mut dem_m3s = base_junc_water_m3s.clone();
                for pi in 0..n_wp {
                    if wp_breaks[pi] > 0 || wp_leaks[pi] > 0 {
                        let bleed = ((wp_breaks[pi] as f64) * 0.0165
                            + (wp_leaks[pi] as f64) * 0.0095)
                            * if booster_on { 1.0 } else { 0.38 };
                        let wp = &water_pipes[pi];
                        dem_m3s[wp.u] += 0.5 * bleed;
                        dem_m3s[wp.v] += 0.5 * bleed;
                    }
                }
                // Concurrent multi-ignition firefighting demand (`65 L/s` ~ `1,030 GPM` per active fire!)
                for &ig_j in &ignited_indices {
                    // Only draw from municipal main if not served by an independent auxiliary cistern!
                    if parcel_cist_u[ig_j] >= cistern_frac {
                        dem_m3s[parcel_near_junc[ig_j]] += 0.065;
                    }
                }

                let mut head = vec![0.0f64; n_junc];
                let init_h = wtp_juncs
                    .iter()
                    .map(|&wj| elev_m[wj] + added_h)
                    .sum::<f64>()
                    / (wtp_juncs.len().max(1) as f64);
                for i in 0..n_junc {
                    head[i] = if w_reach[i] { init_h - 3.0 } else { elev_m[i] };
                }
                for &wj in &wtp_juncs {
                    head[wj] = elev_m[wj] + added_h;
                }

                for _it in 0..36 {
                    let mut max_d = 0.0f64;
                    for i in 0..n_junc {
                        if !w_reach[i] || wtp_juncs.contains(&i) {
                            continue;
                        }
                        let mut g_sum = 0.0;
                        let mut gh_sum = 0.0;
                        for &pi in &junc_wadj[i] {
                            if wp_breaks[pi] > 0 {
                                continue;
                            }
                            let wp = &water_pipes[pi];
                            let nb = if wp.u == i { wp.v } else { wp.u };
                            if !w_reach[nb] {
                                continue;
                            }
                            let dh = (head[i] - head[nb]).abs().max(0.01);
                            let g = wp.hw_k * dh.powf(-0.460043);
                            g_sum += g;
                            gh_sum += g * head[nb];
                        }
                        if g_sum > 1e-12 {
                            let h_t = (gh_sum - dem_m3s[i]) / g_sum;
                            let h_n = 0.45 * head[i] + 0.55 * h_t;
                            max_d = max_d.max((h_n - head[i]).abs());
                            head[i] = h_n;
                        }
                    }
                    if max_d < 5e-3 {
                        break;
                    }
                }
                for i in 0..n_junc {
                    junc_psi[i] = if w_reach[i] {
                        ((head[i] - elev_m[i]) * PSI_PER_METER_H2O).max(0.0)
                    } else {
                        0.0
                    };
                }
            }

            let mut parcel_psi = vec![0.0f64; n_parcels];
            let mut parcel_dry = vec![false; n_parcels];
            for j in 0..n_parcels {
                let psi = if n_junc > 0 {
                    junc_psi[parcel_near_junc[j]]
                } else {
                    0.0
                };
                parcel_psi[j] = psi;
                parcel_dry[j] = psi < MIN_FIRE_FLOW_RESIDUAL_PSI && parcel_cist_u[j] >= cistern_frac;
            }

            // F. Initial Fire-Engine Dispatch & Hydrant Suppression vs. Escaped Conflagrations!
            // Sort simultaneous ignitions by earliest arriving engine (`parcel_engine_s[j]`)
            ignited_indices.sort_by(|&a, &b| {
                parcel_engine_s[a]
                    .partial_cmp(&parcel_engine_s[b])
                    .unwrap_or(Ordering::Equal)
            });

            let mut engines_left = total_engines;
            let mut suppressed = 0usize;
            let mut dry_hydrant_fails = 0usize;
            let mut escaped_sources = Vec::new();
            let flashover_s = cfg.initial_attack_flashover_s.clamp(120.0, 900.0);

            for &ig_j in &ignited_indices {
                let arrives_in_time = engines_left > 0 && parcel_engine_s[ig_j] <= flashover_s;
                if arrives_in_time {
                    // Engine commits to scene
                    engines_left -= 1;
                    let has_water = !parcel_dry[ig_j];
                    if has_water {
                        suppressed += 1;
                    } else {
                        // Engine arrived in time, but hydrant pressure < 20 psi ("Dry Hydrant Failure")!
                        dry_hydrant_fails += 1;
                        escaped_sources.push(ig_j);
                    }
                } else {
                    if parcel_dry[ig_j] {
                        dry_hydrant_fails += 1;
                    }
                    escaped_sources.push(ig_j);
                }
            }

            // G. Hamada (1951) / FEMA Hazus Elliptical Wind-Driven Block-to-Block Conflagration Spread!
            let mut parcel_burned = vec![false; n_parcels];
            if !escaped_sources.is_empty() {
                let mut ignite_time_min = vec![f64::INFINITY; n_parcels];
                let mut p_heap = BinaryHeap::new();
                for &src_j in &escaped_sources {
                    ignite_time_min[src_j] = 0.0;
                    p_heap.push(DijkNode {
                        cost: 0.0,
                        id: src_j,
                    });
                }
                while let Some(DijkNode { cost, id }) = p_heap.pop() {
                    if cost > ignite_time_min[id] || cost > horizon_m {
                        continue;
                    }
                    parcel_burned[id] = true;
                    for &(nb_p, dt_min) in &fire_spread_adj[id] {
                        // Defensive perimeter stand if neighbor parcel has BOTH intact >= 35 psi hydrant water
                        // AND high fire-resistive construction (`CBD`), slowing spread by `1.8x`!
                        let def_slow = if !parcel_dry[nb_p] && parcel_psi[nb_p] >= 35.0 {
                            1.45
                        } else {
                            1.0
                        };
                        let next_t = cost + dt_min * def_slow;
                        if next_t <= horizon_m && next_t < ignite_time_min[nb_p] {
                            ignite_time_min[nb_p] = next_t;
                            p_heap.push(DijkNode {
                                cost: next_t,
                                id: nb_p,
                            });
                        }
                    }
                }
            }

            // Suppressed incipient fires suffer minor single-room loss (`4%` of parcel value);
            // Escaped / conflagration-burned parcels suffer `85%` structural & contents loss!
            let mut burned_parcels = 0usize;
            let mut burned_tfa = 0.0;
            let mut displaced_res = 0.0;
            let mut loss_musd = 0.0;
            for j in 0..n_parcels {
                if parcel_burned[j] {
                    burned_parcels += 1;
                    burned_tfa += parcel_tfa_mmsqft[j];
                    displaced_res += parcels[j].residents();
                    loss_musd += 0.85 * parcel_val_musd[j];
                } else if parcel_ignited[j] {
                    loss_musd += 0.04 * parcel_val_musd[j];
                }
            }

            TrialFfeResult {
                ignitions: total_ignitions,
                suppressed,
                dry_hydrant_fails,
                escaped: escaped_sources.len(),
                burned_parcels,
                burned_tfa,
                displaced_res,
                loss_musd,
                parcel_ignited,
                parcel_dry,
                parcel_burned,
                parcel_psi,
            }
        })
        .collect();

    // 5. Aggregate Monte Carlo Expectations, P90 Tail Losses, and Per-Parcel Conflagration Risk Rows
    let inv_n = 1.0 / (trials as f64);
    let inv_n_pct = 100.0 / (trials as f64);
    let q90_idx = (((trials - 1) as f64) * 0.90).round() as usize;

    let mut ign_vec: Vec<usize> = trial_results.iter().map(|r| r.ignitions).collect();
    ign_vec.sort_unstable();
    let mut brn_vec: Vec<usize> = trial_results.iter().map(|r| r.burned_parcels).collect();
    brn_vec.sort_unstable();
    let mut loss_vec: Vec<f64> = trial_results.iter().map(|r| r.loss_musd).collect();
    loss_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mut p_ig_cnt = vec![0usize; n_parcels];
    let mut p_dry_cnt = vec![0usize; n_parcels];
    let mut p_brn_cnt = vec![0usize; n_parcels];
    let mut p_psi_sum = vec![0.0f64; n_parcels];

    for tr in &trial_results {
        for j in 0..n_parcels {
            if tr.parcel_ignited[j] {
                p_ig_cnt[j] += 1;
            }
            if tr.parcel_dry[j] {
                p_dry_cnt[j] += 1;
            }
            if tr.parcel_burned[j] {
                p_brn_cnt[j] += 1;
            }
            p_psi_sum[j] += tr.parcel_psi[j];
        }
    }

    let mut parcel_risks = Vec::with_capacity(n_parcels);
    let mut psi_city_sum = 0.0;
    let mut dry_city_sum = 0.0;
    for j in 0..n_parcels {
        let ig_pct = (p_ig_cnt[j] as f64) * inv_n_pct;
        let dry_pct = (p_dry_cnt[j] as f64) * inv_n_pct;
        let brn_pct = (p_brn_cnt[j] as f64) * inv_n_pct;
        let m_psi = p_psi_sum[j] * inv_n;
        psi_city_sum += m_psi;
        dry_city_sum += dry_pct;
        let exp_loss = (brn_pct / 100.0) * 0.85 * parcel_val_musd[j]
            + ((ig_pct - brn_pct).max(0.0) / 100.0) * 0.04 * parcel_val_musd[j];
        parcel_risks.push(ParcelFfeRiskRow {
            parcel_id: parcels[j].id,
            zone_code: parcels[j].zone,
            center: parcels[j].center,
            floor_area_mmsqft: parcel_tfa_mmsqft[j],
            replacement_value_musd: parcel_val_musd[j],
            median_pga_g: parcel_med_pga[j],
            ignition_prob_pct: ig_pct,
            dry_hydrant_prob_pct: dry_pct,
            burn_prob_pct: brn_pct,
            mean_post_quake_psi: m_psi,
            expected_fire_loss_musd: exp_loss,
        });
    }

    // Polar 60-minute Hamada elliptical spread envelope (`0..=360 deg`) for `R3` Wood-Frame vs `CBD` Fire-Resistive
    let r_wood = hamada_hazus_spread_rates(ZoneCode::R3, cfg.wind_speed_mps);
    let r_cbd = hamada_hazus_spread_rates(ZoneCode::CBD, cfg.wind_speed_mps);
    let mut polar_spread_envelope_60m = Vec::with_capacity(37);
    for step in 0..=36 {
        let ang_deg = (step as f64) * 10.0;
        let rel_rad = (ang_deg - cfg.wind_direction_deg).to_radians();
        let ct = rel_rad.cos();
        let st = rel_rad.sin();
        let vw = ((if ct >= 0.0 {
            r_wood.v_down_m_min
        } else {
            r_wood.v_up_m_min
        } * ct)
            .powi(2)
            + (r_wood.v_cross_m_min * st).powi(2))
        .sqrt();
        let vc = ((if ct >= 0.0 {
            r_cbd.v_down_m_min
        } else {
            r_cbd.v_up_m_min
        } * ct)
            .powi(2)
            + (r_cbd.v_cross_m_min * st).powi(2))
        .sqrt();
        polar_spread_envelope_60m.push((ang_deg, vw * 60.0, vc * 60.0));
    }

    let major_conf_cnt = trial_results
        .iter()
        .filter(|r| r.burned_parcels >= 3)
        .count();

    FfeMonteCarloReport {
        config: cfg.clone(),
        total_parcels: n_parcels,
        total_floor_area_mmsqft: parcel_tfa_mmsqft.iter().sum(),
        total_exposed_value_musd: parcel_val_musd.iter().sum(),
        total_fire_engines: total_engines,
        expected_ignitions: trial_results.iter().map(|r| r.ignitions as f64).sum::<f64>() * inv_n,
        p90_ignitions: ign_vec[q90_idx],
        expected_knockdowns_suppressed: trial_results
            .iter()
            .map(|r| r.suppressed as f64)
            .sum::<f64>()
            * inv_n,
        expected_dry_hydrant_failures: trial_results
            .iter()
            .map(|r| r.dry_hydrant_fails as f64)
            .sum::<f64>()
            * inv_n,
        expected_escaped_conflagrations: trial_results
            .iter()
            .map(|r| r.escaped as f64)
            .sum::<f64>()
            * inv_n,
        expected_burned_parcels: trial_results
            .iter()
            .map(|r| r.burned_parcels as f64)
            .sum::<f64>()
            * inv_n,
        p90_burned_parcels: brn_vec[q90_idx],
        expected_burned_floor_area_mmsqft: trial_results.iter().map(|r| r.burned_tfa).sum::<f64>()
            * inv_n,
        expected_displaced_residents: trial_results.iter().map(|r| r.displaced_res).sum::<f64>()
            * inv_n,
        expected_fire_loss_musd: trial_results.iter().map(|r| r.loss_musd).sum::<f64>() * inv_n,
        p90_fire_loss_musd: loss_vec[q90_idx],
        prob_major_conflagration_pct: (major_conf_cnt as f64) * inv_n_pct,
        mean_post_quake_hydrant_psi: psi_city_sum / (n_parcels.max(1) as f64),
        dry_hydrant_parcels_pct: dry_city_sum / (n_parcels.max(1) as f64),
        polar_spread_envelope_60m,
        parcel_risks,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

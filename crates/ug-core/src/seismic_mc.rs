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

//! Probabilistic Seismic Bridge Fragility Curves, FEMA Hazus Post-Earthquake Restoration Functions
//! (Days 1, 7, 30, and 90), and Multi-Week Monte Carlo Emergency Coverage Recovery Curves
//!
//! Grounded in:
//! - FEMA Hazus Earthquake Model Highway Bridge Fragility Functions (`HWB` classes, 4 damage
//!   states: Slight, Moderate, Extensive, Complete; lognormal dispersion `beta = 0.60` on
//!   `Sa(1.0 s)` in `g`) — `https://ascelibrary.com/doi/10.1061/PPSCFX.SCENG-1464` and
//!   Regional DOT Highway Seismic Risk Models.
//! - FEMA Hazus Highway Bridge Continuous Normal Restoration Functions (`Slight: 0.6 +- 0.6 d`,
//!   `Moderate: 2.5 +- 2.7 d`, `Extensive: 75.0 +- 42.0 d`, `Complete: 230.0 +- 110.0 d`) —
//!   `https://rosap.ntl.bts.gov/view/dot/27476/dot_27476_DS1.pdf` and ASME Journal of Risk and
//!   Uncertainty (`https://asmedigitalcollection.asme.org/risk/article/10/2/021104/1195450/Fragility-Analysis-and-Resilience-Assessment-of`).
//! - Bruneau et al. (2003), *A Framework to Quantitatively Assess and Enhance the Seismic
//!   Resilience of Communities*, Earthquake Spectra 19(4):733-752
//!   (`https://pubs.geoscienceworld.org/eeri/earthquake-spectra/article/19/4/733/584942/A-Framework-to-Quantitatively-Assess-and-Enhance`).
//! - Shinozuka, Feng, Lee & Naganuma (2000), *Statistical Analysis of Fragility Curves*, ASCE
//!   Journal of Engineering Mechanics 126(12):1224-1231
//!   (`https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9399%282000%29126%3A12%281224%29`).
//! - Jayaram & Baker (2009), *Correlation model for spatially distributed ground-motion
//!   intensities*, Earthquake Engineering & Structural Dynamics 38(15):1687-1708
//!   (`https://www.jackwbaker.com/Publications/Jayaram_Baker_(2009)_spatial_correlation,_EESD.pdf`).
//! - FEMA / USFA NFPA 1710 240-second (4-minute) response travel-time benchmark
//!   (`https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf`).

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind, EvpConfig};
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::utilities::{MIN_FIRE_FLOW_RESIDUAL_PSI, PSI_PER_METER_H2O};
use crate::zoning::Parcel;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::f64::consts::{PI, SQRT_2};
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const SEISMIC_BRIDGE_CSV_HEADER: &str = "bridge_id,bridge_label,segment_ids,base_archetype,effective_archetype,retrofitted,median_site_sa10_g,analytical_p_slight,analytical_p_moderate,analytical_p_extensive,analytical_p_complete,mc_prob_none_pct,mc_prob_slight_pct,mc_prob_moderate_pct,mc_prob_extensive_pct,mc_prob_complete_pct,mc_closure_rate_day0_pct,func_day1_pct,func_day7_pct,func_day30_pct,func_day90_pct,expected_repair_days";

pub const RECOVERY_TIMELINE_CSV_HEADER: &str = "horizon_day,epoch_label,mean_parcel_cov_pct,p10_parcel_cov_pct,p50_parcel_cov_pct,p90_parcel_cov_pct,mean_resident_cov_pct,p10_resident_cov_pct,mean_p90_travel_s,p90_tail_travel_s,prob_meeting_nfpa_target_pct,expected_closed_bridges,mean_bridge_functionality_pct,mean_electric_service_pct,mean_water_20psi_service_pct,mean_wastewater_service_pct,mean_composite_lifeline_pct,expected_water_repairs_rem,expected_sewer_repairs_rem";
pub const SEISMIC_PIPELINE_CSV_HEADER: &str = "corridor_id,corridor_label,length_m,water_diam_mm,sewer_diam_mm,water_material,sewer_material,seismic_replaced,median_pgv_cm_s,water_rr_per_km,sewer_rr_per_km,mc_water_break_prob_pct,mc_sewer_break_prob_pct,mc_expected_water_repairs,mc_expected_sewer_repairs";

/// Post-earthquake recovery evaluation horizons (days), anchored on Hazus benchmarks
/// (**Day 0 Impact, Day 1, Day 7, Day 30, and Day 90**, plus intermediate Days 3, 14, and 60).
pub const RECOVERY_TIMELINE_DAYS: [f64; 8] = [0.0, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 90.0];

/// Primary Hazus milestone indices inside `RECOVERY_TIMELINE_DAYS` (`Day 1`, `Day 7`, `Day 30`, `Day 90`).
pub const HAZUS_BENCHMARK_DAYS: [f64; 4] = [1.0, 7.0, 30.0, 90.0];

/// Standard normal cumulative distribution function `Phi(z)` via Abramowitz & Stegun (1964) 7.1.26
/// rational approximation (`|error| < 1.5e-7` across `(-inf, +inf)`).
#[inline]
pub fn normal_cdf_phi(z: f64) -> f64 {
    if !z.is_finite() {
        return if z.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let x = (z / SQRT_2).abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc_x = poly * (-x * x).exp();
    if z >= 0.0 {
        1.0 - 0.5 * erfc_x
    } else {
        0.5 * erfc_x
    }
}

/// Deterministic SplitMix64 PRNG for thread-independent Common Random Numbers (`CRN`) across
/// `Rayon` Monte Carlo earthquake & bridge restoration realizations.
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

fn cholesky_lower_triangular(corr: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = corr[i * n + j];
            if i == j {
                sum += 1e-9;
            }
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = sum.max(1e-12).sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j].max(1e-12);
            }
        }
    }
    l
}

/// FEMA Hazus `HWB` / ODOT highway bridge structural archetypes parameterized by 1.0-second
/// spectral acceleration `Sa(1.0 s)` (in `g`) lognormal medians and dispersion `beta_ds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeStructuralArchetype {
    /// Pre-1975 multi-span simply supported concrete/steel girder (`HWB5`/`HWB12` unretrofitted).
    Pre1975MultiSpanSimplySupported,
    /// Conventional seismic-code continuous concrete box/T-girder (`HWB15`/`HWB17`).
    ConventionalContinuousConcrete,
    /// Column-jacketed & cable-restrainer seismically retrofitted bridge (`HWB16`/`HWB23`).
    SeismicallyRetrofittedCollar,
    /// Essential base-isolated / modern lifeline crossing (`HWB28`).
    EssentialBaseIsolatedLifeline,
}

impl BridgeStructuralArchetype {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Pre1975MultiSpanSimplySupported => "HWB5 Pre-1975 Unretrofitted",
            Self::ConventionalContinuousConcrete => "HWB15 Conventional Continuous",
            Self::SeismicallyRetrofittedCollar => "HWB16 Jacketed Retrofit",
            Self::EssentialBaseIsolatedLifeline => "HWB28 Base-Isolated Lifeline",
        }
    }

    pub fn fragility_params_sa10_g(self) -> ([f64; 4], f64) {
        match self {
            Self::Pre1975MultiSpanSimplySupported => ([0.26, 0.36, 0.46, 0.72], 0.60),
            Self::ConventionalContinuousConcrete => ([0.46, 0.66, 0.90, 1.32], 0.60),
            Self::SeismicallyRetrofittedCollar => ([0.68, 0.96, 1.32, 1.88], 0.60),
            Self::EssentialBaseIsolatedLifeline => ([0.92, 1.32, 1.78, 2.48], 0.55),
        }
    }

    pub fn exceedance_probs(self, im_sa10_g: f64) -> [f64; 4] {
        if im_sa10_g <= 1e-6 {
            return [0.0; 4];
        }
        let (medians, beta) = self.fragility_params_sa10_g();
        let ln_im = im_sa10_g.ln();
        let mut p = [0.0; 4];
        for (k, &med) in medians.iter().enumerate() {
            let z = (ln_im - med.ln()) / beta;
            p[k] = normal_cdf_phi(z).clamp(0.0, 1.0);
        }
        for k in 1..4 {
            if p[k] > p[k - 1] {
                p[k] = p[k - 1];
            }
        }
        p
    }

    pub fn discrete_state_probs(self, im_sa10_g: f64) -> [f64; 5] {
        let ex = self.exceedance_probs(im_sa10_g);
        [
            (1.0 - ex[0]).max(0.0),
            (ex[0] - ex[1]).max(0.0),
            (ex[1] - ex[2]).max(0.0),
            (ex[2] - ex[3]).max(0.0),
            ex[3].clamp(0.0, 1.0),
        ]
    }

    pub fn sample_curve_points(self, max_im_g: f64, num_pts: usize) -> Vec<(f64, [f64; 4])> {
        let n = num_pts.max(8);
        (0..n)
            .map(|i| {
                let im = (i as f64 / (n - 1) as f64) * max_im_g.max(0.5);
                (im, self.exceedance_probs(im))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SeismicDamageState {
    None = 0,
    Slight = 1,
    Moderate = 2,
    Extensive = 3,
    Complete = 4,
}

impl SeismicDamageState {
    /// Immediate Day-0 post-earthquake emergency apparatus speed multiplier (`0.0` = bridge closed).
    #[inline]
    pub fn post_event_speed_factor(self) -> f64 {
        match self {
            Self::None => 1.00,
            Self::Slight => 0.85,
            Self::Moderate => 0.50,
            Self::Extensive | Self::Complete => 0.0,
        }
    }

    #[inline]
    pub fn is_closed(self) -> bool {
        matches!(self, Self::Extensive | Self::Complete)
    }

    /// FEMA Hazus Highway Bridge Continuous Restoration Normal Parameters `(mean_days, sd_days)`:
    /// - Slight: `0.6 +- 0.6 days`
    /// - Moderate: `2.5 +- 2.7 days`
    /// - Extensive: `75.0 +- 42.0 days`
    /// - Complete: `230.0 +- 110.0 days`
    #[inline]
    pub fn hazus_restoration_mean_sd_days(self) -> (f64, f64) {
        match self {
            Self::None => (0.0, 0.01),
            Self::Slight => (0.6, 0.6),
            Self::Moderate => (2.5, 2.7),
            Self::Extensive => (75.0, 42.0),
            Self::Complete => (230.0, 110.0),
        }
    }

    /// Continuous FEMA Hazus Normal CDF restoration functionality `F_ds(t) in [0.0, 1.0]` at
    /// `day >= 0.0` scaled by `crew_speedup` (`1.0` = standard Hazus restoration rate).
    /// Anchored at `t = 0` to `self.post_event_speed_factor()` and monotonically approaching `1.0`.
    pub fn hazus_functionality_at_day(self, day: f64, crew_speedup: f64) -> f64 {
        if self == Self::None {
            return 1.0;
        }
        let t = day.max(0.0) * crew_speedup.clamp(0.25, 5.0);
        if t <= 1e-9 {
            return self.post_event_speed_factor();
        }
        let (mean_d, sd_d) = self.hazus_restoration_mean_sd_days();
        let phi_0 = normal_cdf_phi(-mean_d / sd_d);
        let phi_t = normal_cdf_phi((t - mean_d) / sd_d);
        let norm_prog = ((phi_t - phi_0) / (1.0 - phi_0).max(1e-9)).clamp(0.0, 1.0);
        let q0 = self.post_event_speed_factor();
        (q0 + (1.0 - q0) * norm_prog).clamp(0.0, 1.0)
    }

    /// Evaluates a discrete bridge's speed multiplier (`0.0` = closed, `0.45..=1.0` = open) at
    /// `day` inside a Monte Carlo realization given its sampled full-restoration completion time
    /// `full_repair_days` (including emergency shoring / temporary lane reopening for Extensive/Complete).
    #[inline]
    pub fn realized_speed_factor_at_day(self, day: f64, full_repair_days: f64) -> f64 {
        if self == Self::None || day >= full_repair_days {
            return 1.00;
        }
        match self {
            Self::None => 1.00,
            Self::Slight => 0.85,
            Self::Moderate => 0.50,
            Self::Extensive => {
                // Emergency column shoring opens restricted 1-lane emergency access at 55% of full repair time
                if day >= 0.55 * full_repair_days {
                    0.52
                } else {
                    0.0
                }
            }
            Self::Complete => {
                // Temporary modular/Bailey emergency span opens restricted passage at 65% of full replacement time
                if day >= 0.65 * full_repair_days {
                    0.45
                } else {
                    0.0
                }
            }
        }
    }

    #[inline]
    pub fn from_uniform_sample(u01: f64, ex_probs: &[f64; 4]) -> Self {
        if u01 < ex_probs[3] {
            Self::Complete
        } else if u01 < ex_probs[2] {
            Self::Extensive
        } else if u01 < ex_probs[1] {
            Self::Moderate
        } else if u01 < ex_probs[0] {
            Self::Slight
        } else {
            Self::None
        }
    }
}

/// Samples the 4 Hazus continuous damage-state restoration curves `[Slight, Moderate, Extensive, Complete]`
/// across `day = 0.0 ..= max_day` (`90.0` days) for UI plotting.
pub fn sample_hazus_restoration_curves(
    max_day: f64,
    crew_speedup: f64,
    num_pts: usize,
) -> Vec<(f64, [f64; 4])> {
    let n = num_pts.max(10);
    let span = max_day.max(7.0);
    (0..n)
        .map(|i| {
            let d = (i as f64 / (n - 1) as f64) * span;
            (
                d,
                [
                    SeismicDamageState::Slight.hazus_functionality_at_day(d, crew_speedup),
                    SeismicDamageState::Moderate.hazus_functionality_at_day(d, crew_speedup),
                    SeismicDamageState::Extensive.hazus_functionality_at_day(d, crew_speedup),
                    SeismicDamageState::Complete.hazus_functionality_at_day(d, crew_speedup),
                ],
            )
        })
        .collect()
}

/// Empirical Peak Ground Velocity (`PGV`) wave-propagation repair-rate (`RR`) formulation for
/// buried potable water and sanitary sewer pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineFragilityLaw {
    /// American Lifelines Alliance (2001) linear `PGV` relation (`RR = K_1 * K_2 * 0.00187 * PGV_in_s`
    /// per `1,000 ft` = `K_1 * K_2 * 0.0024154 * PGV_cm_s` repairs/km).
    Ala2001LinearPgv,
    /// FEMA Hazus / O'Rourke & Ayala (1993) power-law `PGV` relation
    /// (`RR = K_1 * 0.0001 * PGV_cm_s^2.25` repairs/km).
    HazusORourkeAyala1993,
}

impl Default for PipelineFragilityLaw {
    fn default() -> Self {
        Self::Ala2001LinearPgv
    }
}

impl PipelineFragilityLaw {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ala2001LinearPgv => "ALA (2001) Linear PGV (0.00242*PGV)",
            Self::HazusORourkeAyala1993 => "FEMA Hazus O'Rourke (0.0001*PGV^2.25)",
        }
    }
}

/// Buried water & sanitary sewer pipe material classes with ALA (2001) Table 4-1 & FEMA Hazus
/// seismic vulnerability modification factors `K_1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuriedPipeMaterialClass {
    /// Legacy brittle Cast Iron (`CI`), Asbestos Cement (`AC`), or Vitrified Clay (`VCP`) sewer (`K_1 = 1.00`).
    BrittleCastIronVitrifiedClay,
    /// Segmented rubber-gasketed Ductile Iron (`DI`) or `PVC` (`K_1 = 0.50` ALA / `0.30` Hazus).
    SegmentedDuctileIronPvc,
    /// Arc-welded Steel or restrained-joint Ductile Iron (`K_1 = 0.25` ALA / `0.15` Hazus).
    WeldedSteelRestrainedJoint,
    /// Earthquake-Resistant Ductile Iron Pipe (`ISO 16134 ERDIP`) or butt-fused `HDPE` (`K_1 = 0.08`).
    SeismicErdipHdpe,
}

impl Default for BuriedPipeMaterialClass {
    fn default() -> Self {
        Self::BrittleCastIronVitrifiedClay
    }
}

impl BuriedPipeMaterialClass {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::BrittleCastIronVitrifiedClay => "Brittle CI / VCP (K1=1.0)",
            Self::SegmentedDuctileIronPvc => "Ductile Iron / PVC (K1=0.5/0.3)",
            Self::WeldedSteelRestrainedJoint => "Welded Steel (K1=0.25)",
            Self::SeismicErdipHdpe => "Seismic ERDIP / HDPE (K1=0.08)",
        }
    }

    #[inline]
    pub fn k1_factor(self, law: PipelineFragilityLaw) -> f64 {
        match (self, law) {
            (Self::BrittleCastIronVitrifiedClay, _) => 1.00,
            (Self::SegmentedDuctileIronPvc, PipelineFragilityLaw::Ala2001LinearPgv) => 0.50,
            (Self::SegmentedDuctileIronPvc, PipelineFragilityLaw::HazusORourkeAyala1993) => 0.30,
            (Self::WeldedSteelRestrainedJoint, PipelineFragilityLaw::Ala2001LinearPgv) => 0.25,
            (Self::WeldedSteelRestrainedJoint, PipelineFragilityLaw::HazusORourkeAyala1993) => 0.15,
            (Self::SeismicErdipHdpe, _) => 0.08,
        }
    }
}

/// Computes ALA (2001) or FEMA Hazus buried pipeline seismic repair rate `RR` (`repairs / km`)
/// as a function of `pgv_cm_s` (`cm/s`), pipe `material`, `diam_mm`, and `is_gravity_sewer`.
#[inline]
pub fn ala_hazus_pipe_repair_rate_per_km(
    pgv_cm_s: f64,
    law: PipelineFragilityLaw,
    material: BuriedPipeMaterialClass,
    diam_mm: f64,
    is_gravity_sewer: bool,
) -> f64 {
    let pgv = pgv_cm_s.max(0.0);
    if pgv <= 1e-6 {
        return 0.0;
    }
    let k1 = material.k1_factor(law);
    // ALA (2001) diameter & rigid-manhole-joint modifiers
    let k2_diam = if diam_mm >= 350.0 { 0.82 } else { 1.12 };
    let sewer_mult = if is_gravity_sewer { 1.15 } else { 1.0 };
    let base_rr = match law {
        PipelineFragilityLaw::Ala2001LinearPgv => {
            // Exact conversion: `0.00187 repairs / 1,000 ft / (in/s)` -> `0.00241538 repairs / km / (cm/s)`
            0.002_415_38 * pgv * k2_diam
        }
        PipelineFragilityLaw::HazusORourkeAyala1993 => 0.0001 * pgv.powf(2.25) * k2_diam,
    };
    (k1 * sewer_mult * base_rr).max(0.0)
}

/// Samples ALA (2001) / Hazus `PGV` repair-rate curves `(pgv_cm_s, [RR_CI, RR_DI, RR_Steel, RR_ERDIP])`
/// across `0 ..= max_pgv_cm_s` (`120 cm/s`) for interactive UI plotting.
pub fn sample_ala_hazus_pgv_curves(
    law: PipelineFragilityLaw,
    max_pgv_cm_s: f64,
    num_pts: usize,
) -> Vec<(f64, [f64; 4])> {
    let n = num_pts.max(10);
    let span = max_pgv_cm_s.max(20.0);
    let mats = [
        BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
        BuriedPipeMaterialClass::SegmentedDuctileIronPvc,
        BuriedPipeMaterialClass::WeldedSteelRestrainedJoint,
        BuriedPipeMaterialClass::SeismicErdipHdpe,
    ];
    (0..n)
        .map(|i| {
            let pgv = (i as f64 / (n - 1) as f64) * span;
            (
                pgv,
                [
                    ala_hazus_pipe_repair_rate_per_km(pgv, law, mats[0], 250.0, false),
                    ala_hazus_pipe_repair_rate_per_km(pgv, law, mats[1], 250.0, false),
                    ala_hazus_pipe_repair_rate_per_km(pgv, law, mats[2], 250.0, false),
                    ala_hazus_pipe_repair_rate_per_km(pgv, law, mats[3], 250.0, false),
                ],
            )
        })
        .collect()
}

/// Exact inverse-CDF Poisson sampler `N ~ Poisson(lambda)` driven by a single uniform `u in (0, 1)`:
/// guarantees strict Common Random Numbers (`CRN`) monotonicity (`lambda_1 <= lambda_2 => N_1 <= N_2`)!
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicScenarioConfig {
    pub moment_magnitude_mw: f64,
    pub rupture_standoff_km: f64,
    pub vs30_m_per_s: f64,
    pub inter_event_sigma_ln: f64,
    pub intra_event_sigma_ln: f64,
    pub spatial_corr_range_km: f64,
    pub monte_carlo_trials: usize,
    pub random_seed: u64,
    pub retrofit_top_k_bridges: usize,
    pub retrofit_target_archetype: BridgeStructuralArchetype,
    /// Post-earthquake DOT repair crew acceleration multiplier (`1.0` = baseline Hazus restoration;
    /// `> 1.0` = accelerated 24/7 emergency shoring & modular bridging).
    #[serde(default = "default_crew_speedup")]
    pub repair_crew_speedup: f64,
    pub max_bridge_inventory: usize,
    pub target_coverage_pct: f64,
    pub threshold_travel_s: f64,
    /// Buried water & sanitary sewer pipeline `PGV` repair-rate fragility formulation.
    #[serde(default)]
    pub pipe_fragility_law: PipelineFragilityLaw,
    /// Baseline water distribution pipe material prior to targeted seismic ERDIP/HDPE replacement.
    #[serde(default)]
    pub baseline_water_pipe_material: BuriedPipeMaterialClass,
    /// Baseline gravity sanitary sewer pipe material prior to seismic replacement / CIPP lining.
    #[serde(default)]
    pub baseline_sewer_pipe_material: BuriedPipeMaterialClass,
    /// Percentage (`0..=100%`) of highest-risk buried water & sewer corridors upgraded to
    /// `BuriedPipeMaterialClass::SeismicErdipHdpe` (`ISO 16134` chain-joint ERDIP / HDPE).
    #[serde(default)]
    pub seismic_pipe_replacement_pct: f64,
    /// Dedicated municipal water & wastewater pipeline repair crews (`Porter 2016` / `Hazus`).
    #[serde(default = "default_pipe_crews")]
    pub water_sewer_repair_crews: usize,
    /// Emergency diesel / battery microgrid backup power on water booster plants & sewage lift/WWTP.
    #[serde(default)]
    pub critical_lifeline_backup_power: bool,
}

fn default_crew_speedup() -> f64 {
    1.0
}

fn default_pipe_crews() -> usize {
    6
}

impl Default for SeismicScenarioConfig {
    fn default() -> Self {
        Self {
            moment_magnitude_mw: 7.2,
            rupture_standoff_km: 16.0,
            vs30_m_per_s: 270.0,
            inter_event_sigma_ln: 0.32,
            intra_event_sigma_ln: 0.55,
            spatial_corr_range_km: 25.0,
            monte_carlo_trials: 500,
            random_seed: 2026_1113,
            retrofit_top_k_bridges: 0,
            retrofit_target_archetype: BridgeStructuralArchetype::SeismicallyRetrofittedCollar,
            repair_crew_speedup: 1.0,
            max_bridge_inventory: 32,
            target_coverage_pct: 90.0,
            threshold_travel_s: 240.0,
            pipe_fragility_law: PipelineFragilityLaw::Ala2001LinearPgv,
            baseline_water_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
            baseline_sewer_pipe_material: BuriedPipeMaterialClass::BrittleCastIronVitrifiedClay,
            seismic_pipe_replacement_pct: 0.0,
            water_sewer_repair_crews: 6,
            critical_lifeline_backup_power: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicBridgeAsset {
    pub bridge_id: usize,
    pub label: String,
    pub segment_ids: Vec<usize>,
    pub midpoint: Vec2,
    pub p0: Vec2,
    pub p1: Vec2,
    pub length_m: f64,
    pub base_archetype: BridgeStructuralArchetype,
    pub effective_archetype: BridgeStructuralArchetype,
    pub is_retrofitted: bool,
    pub site_distance_km: f64,
    pub median_site_sa10_g: f64,
    pub analytical_exceedance_at_median: [f64; 4],
    pub mc_state_pcts: [f64; 5],
    pub mc_closure_rate_pct: f64,
    /// Expected post-earthquake bridge functionality (`0..100%`) at Hazus benchmark **Days 1, 7, 30, and 90**.
    pub expected_func_day1_pct: f64,
    pub expected_func_day7_pct: f64,
    pub expected_func_day30_pct: f64,
    pub expected_func_day90_pct: f64,
    /// Expected repair duration (days) unconditionally averaged across all Monte Carlo realizations.
    pub expected_repair_days: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloTrialSummary {
    pub trial_idx: usize,
    pub closed_bridges_count: usize,
    pub degraded_bridges_count: usize,
    pub closed_segment_ids: Vec<usize>,
    pub post_parcel_cov_pct: f64,
    pub post_resident_cov_pct: f64,
    pub post_p90_travel_s: f64,
    pub post_mean_travel_s: f64,
    pub isolated_parcels: usize,
    pub retains_nfpa_target: bool,
}

/// Statistical summary of network emergency coverage & bridge functionality at a specific
/// post-earthquake horizon day (`Day 0, 1, 3, 7, 14, 30, 60, 90`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryEpochSummary {
    pub epoch_idx: usize,
    pub day: f64,
    pub label: String,
    pub is_hazus_primary_benchmark: bool,
    pub mean_parcel_cov_pct: f64,
    pub p10_parcel_cov_pct: f64,
    pub p50_parcel_cov_pct: f64,
    pub p90_parcel_cov_pct: f64,
    pub mean_resident_cov_pct: f64,
    pub p10_resident_cov_pct: f64,
    pub mean_p90_travel_s: f64,
    pub p90_tail_travel_s: f64,
    pub prob_meeting_nfpa_target_pct: f64,
    pub expected_closed_bridges: f64,
    pub mean_bridge_functionality_pct: f64,
    /// Multi-lifeline post-earthquake service percentages (`0..100%`) at `day`:
    /// - Electrical (`%` energized buses within ANSI voltage bounds)
    /// - Potable Water (`%` nodes maintaining `>= 20 psi` residual fire-hydrant pressure)
    /// - Wastewater (`%` nodes free of `SSO` surcharge with `WWTP` operational)
    pub mean_electric_service_pct: f64,
    pub mean_water_20psi_service_pct: f64,
    pub mean_wastewater_service_pct: f64,
    pub mean_composite_lifeline_pct: f64,
    pub expected_water_repairs_remaining: f64,
    pub expected_sewer_repairs_remaining: f64,
    /// Closed segment IDs from the 10th-percentile coverage realization on `day` for live viewport scrubbing!
    pub p10_closed_segment_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicPipelineAssetSummary {
    pub corridor_id: usize,
    pub label: String,
    pub u_junc: usize,
    pub v_junc: usize,
    pub midpoint: Vec2,
    pub length_m: f64,
    pub water_diam_mm: f64,
    pub sewer_diam_mm: f64,
    pub water_material: BuriedPipeMaterialClass,
    pub sewer_material: BuriedPipeMaterialClass,
    pub is_seismically_replaced: bool,
    pub median_pgv_cm_s: f64,
    pub water_rr_per_km: f64,
    pub sewer_rr_per_km: f64,
    pub mc_water_break_prob_pct: f64,
    pub mc_sewer_break_prob_pct: f64,
    pub mc_expected_water_repairs: f64,
    pub mc_expected_sewer_repairs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicMonteCarloReport {
    pub config: SeismicScenarioConfig,
    pub fire_station_count: usize,
    pub bridge_count: usize,
    pub baseline_parcel_cov_pct: f64,
    pub baseline_resident_cov_pct: f64,
    pub baseline_p90_travel_s: f64,
    pub mean_parcel_cov_pct: f64,
    pub p50_parcel_cov_pct: f64,
    pub p10_parcel_cov_pct: f64,
    pub p05_parcel_cov_pct: f64,
    pub mean_resident_cov_pct: f64,
    pub p10_resident_cov_pct: f64,
    pub mean_p90_travel_s: f64,
    pub p50_p90_travel_s: f64,
    pub p90_p90_travel_s: f64,
    pub p95_p90_travel_s: f64,
    pub prob_meeting_nfpa_target_pct: f64,
    pub expected_closed_bridges: f64,
    pub p90_closed_bridges: usize,
    pub expected_isolated_parcels: f64,
    pub coverage_histogram_bins: [usize; 10],
    pub p90_travel_cdf: Vec<[f64; 2]>,
    pub reference_fragility_curve: Vec<(f64, [f64; 4])>,
    pub reference_archetype: BridgeStructuralArchetype,
    /// Pre-sampled Hazus restoration curves `(day, [Slight, Moderate, Extensive, Complete])` across `0..=90d`.
    pub hazus_restoration_curves: Vec<(f64, [f64; 4])>,
    /// Multi-week emergency coverage & bridge restoration trajectory across `RECOVERY_TIMELINE_DAYS`.
    pub recovery_timeline: Vec<RecoveryEpochSummary>,
    /// Bruneau et al. (2003) 90-day normalized community emergency coverage resilience index (`0..=100%`).
    pub bruneau_resilience_index_90d_pct: f64,
    /// Interpolated post-event day (`0..=90`) when expected parcel coverage recovers to `target_coverage_pct` (`None` if `> 90d`).
    pub days_to_90pct_mean_recovery: Option<f64>,
    pub mean_median_sa10_g: f64,
    pub median_corridor_pgv_cm_s: f64,
    pub total_water_pipe_km: f64,
    pub total_sewer_pipe_km: f64,
    pub expected_day0_water_leaks: f64,
    pub expected_day0_water_breaks: f64,
    pub expected_day0_sewer_leaks: f64,
    pub expected_day0_sewer_breaks: f64,
    pub expected_day0_substation_outages: f64,
    /// 90-day Bruneau resilience integrals (`0..=100%`) across Electrical, Water, Wastewater & Composite lifelines!
    pub electric_resilience_index_90d_pct: f64,
    pub water_resilience_index_90d_pct: f64,
    pub wastewater_resilience_index_90d_pct: f64,
    pub multi_lifeline_resilience_90d_pct: f64,
    pub ala_pgv_repair_curves: Vec<(f64, [f64; 4])>,
    pub pipeline_inventory: Vec<SeismicPipelineAssetSummary>,
    pub bridge_inventory: Vec<SeismicBridgeAsset>,
    pub p10_realization: MonteCarloTrialSummary,
    pub worst_realization: MonteCarloTrialSummary,
    pub solve_time_ms: f64,
}

impl SeismicMonteCarloReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str(SEISMIC_BRIDGE_CSV_HEADER);
        out.push('\n');
        for b in &self.bridge_inventory {
            let esc = if b.label.contains(',') || b.label.contains('"') {
                format!("\"{}\"", b.label.replace('"', "\"\""))
            } else {
                b.label.clone()
            };
            let sids = b
                .segment_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(";");
            out.push_str(&format!(
                "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                b.bridge_id,
                esc,
                sids,
                b.base_archetype.short_label(),
                b.effective_archetype.short_label(),
                b.is_retrofitted,
                b.median_site_sa10_g,
                b.analytical_exceedance_at_median[0],
                b.analytical_exceedance_at_median[1],
                b.analytical_exceedance_at_median[2],
                b.analytical_exceedance_at_median[3],
                b.mc_state_pcts[0],
                b.mc_state_pcts[1],
                b.mc_state_pcts[2],
                b.mc_state_pcts[3],
                b.mc_state_pcts[4],
                b.mc_closure_rate_pct,
                b.expected_func_day1_pct,
                b.expected_func_day7_pct,
                b.expected_func_day30_pct,
                b.expected_func_day90_pct,
                b.expected_repair_days,
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
        Ok(self.bridge_inventory.len())
    }

    /// Exports the multi-week post-earthquake recovery curve (`Days 0, 1, 3, 7, 14, 30, 60, 90`) as RFC 4180 CSV.
    pub fn to_recovery_csv_string(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(RECOVERY_TIMELINE_CSV_HEADER);
        out.push('\n');
        for ep in &self.recovery_timeline {
            out.push_str(&format!(
                "{:.1},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3}\n",
                ep.day,
                ep.label,
                ep.mean_parcel_cov_pct,
                ep.p10_parcel_cov_pct,
                ep.p50_parcel_cov_pct,
                ep.p90_parcel_cov_pct,
                ep.mean_resident_cov_pct,
                ep.p10_resident_cov_pct,
                ep.mean_p90_travel_s,
                ep.p90_tail_travel_s,
                ep.prob_meeting_nfpa_target_pct,
                ep.expected_closed_bridges,
                ep.mean_bridge_functionality_pct,
                ep.mean_electric_service_pct,
                ep.mean_water_20psi_service_pct,
                ep.mean_wastewater_service_pct,
                ep.mean_composite_lifeline_pct,
                ep.expected_water_repairs_remaining,
                ep.expected_sewer_repairs_remaining,
            ));
        }
        out
    }

    pub fn export_recovery_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_recovery_csv_string())?;
        Ok(self.recovery_timeline.len())
    }

    pub fn to_pipeline_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.pipeline_inventory.len() * 120 + 256);
        out.push_str(SEISMIC_PIPELINE_CSV_HEADER);
        out.push('\n');
        for p in &self.pipeline_inventory {
            let esc = if p.label.contains(',') || p.label.contains('"') {
                format!("\"{}\"", p.label.replace('"', "\"\""))
            } else {
                p.label.clone()
            };
            out.push_str(&format!(
                "{},{},{:.1},{:.0},{:.0},{},{},{},{:.2},{:.4},{:.4},{:.2},{:.2},{:.3},{:.3}\n",
                p.corridor_id,
                esc,
                p.length_m,
                p.water_diam_mm,
                p.sewer_diam_mm,
                p.water_material.short_label(),
                p.sewer_material.short_label(),
                p.is_seismically_replaced,
                p.median_pgv_cm_s,
                p.water_rr_per_km,
                p.sewer_rr_per_km,
                p.mc_water_break_prob_pct,
                p.mc_sewer_break_prob_pct,
                p.mc_expected_water_repairs,
                p.mc_expected_sewer_repairs,
            ));
        }
        out
    }

    pub fn export_pipeline_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_pipeline_csv_string())?;
        Ok(self.pipeline_inventory.len())
    }
}

pub fn median_sa10_gmpe_g(mw: f64, r_rup_km: f64, vs30_mps: f64) -> f64 {
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
    let ln_sa = (0.24_f64).ln() + mag_term + dist_term + site_term;
    ln_sa.exp().clamp(0.005, 3.50)
}

#[derive(Copy, Clone, PartialEq)]
struct DijkNode {
    cost_s: f64,
    seg_id: usize,
}
impl Eq for DijkNode {}
impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost_s.partial_cmp(&self.cost_s).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn enumerate_seismic_bridge_inventory(
    net: &RoadNetwork,
    facilities: &[EmergencyFacility],
    cfg: &SeismicScenarioConfig,
) -> Vec<SeismicBridgeAsset> {
    let mut station_host_segs = Vec::new();
    for f in facilities {
        if f.kind == EmergencyFacilityKind::FireStation && f.segment_id < net.segments.len() {
            station_host_segs.push(f.segment_id);
            let u = net.segments[f.segment_id].from_junc;
            let v = net.segments[f.segment_id].to_junc;
            for s in &net.segments {
                if s.from_junc == v && s.to_junc == u {
                    station_host_segs.push(s.id);
                }
            }
        }
    }

    let mut visited = vec![false; net.segments.len()];
    let mut raw: Vec<(Vec<usize>, String, u8, usize, f64, Vec2, Vec2, Vec2, bool)> = Vec::new();

    for seg in &net.segments {
        if seg.closed || seg.lane_ids.is_empty() || seg.is_roundabout || visited[seg.id] {
            continue;
        }
        if station_host_segs.contains(&seg.id) {
            continue;
        }
        let n_lc = seg.name.to_lowercase();
        let explicit_bridge = n_lc.contains("bridge")
            || n_lc.contains("viaduct")
            || n_lc.contains("overpass")
            || n_lc.contains("crossing")
            || n_lc.contains("span");

        let major_span = seg.priority >= 3 || seg.lane_count >= 2 || seg.length >= 95.0;
        if !explicit_bridge && !major_span && net.segments.len() > 28 {
            continue;
        }

        visited[seg.id] = true;
        let mut sids = vec![seg.id];
        let u = seg.from_junc;
        let v = seg.to_junc;
        if let Some(tw) = net.segments.iter().find(|s| {
            s.id != seg.id && !s.closed && !visited[s.id] && s.from_junc == v && s.to_junc == u
        }) {
            visited[tw.id] = true;
            sids.push(tw.id);
        }

        let p0 = seg.base_pts.first().copied().unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        let p1 = seg.base_pts.last().copied().unwrap_or(p0);
        let mid = if seg.base_pts.is_empty() {
            p0
        } else {
            seg.base_pts[seg.base_pts.len() / 2]
        };
        let tag = if explicit_bridge {
            seg.name.clone()
        } else {
            format!("{} Overpass (#{})", seg.name, seg.id)
        };
        raw.push((
            sids,
            tag,
            seg.priority,
            seg.lane_count,
            seg.length,
            mid,
            p0,
            p1,
            explicit_bridge,
        ));
    }

    let cx = (net.bounds.min_x + net.bounds.max_x) * 0.5;
    let cy = (net.bounds.min_y + net.bounds.max_y) * 0.5;
    let center = Vec2 { x: cx, y: cy };
    raw.sort_by(|a, b| {
        b.8.cmp(&a.8)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| {
                a.5.dist(center)
                    .partial_cmp(&b.5.dist(center))
                    .unwrap_or(Ordering::Equal)
            })
    });

    let cap = cfg.max_bridge_inventory.clamp(4, 64);
    if raw.len() > cap {
        raw.truncate(cap);
    }

    let west_x = net.bounds.min_x;
    let mut assets: Vec<SeismicBridgeAsset> = raw
        .into_iter()
        .enumerate()
        .map(
            |(idx, (sids, label, prio, lanes, len_m, mid, p0, p1, explicit_bridge))| {
                let base_arch = if explicit_bridge {
                    if idx % 3 == 2 && lanes >= 4 {
                        BridgeStructuralArchetype::ConventionalContinuousConcrete
                    } else {
                        BridgeStructuralArchetype::Pre1975MultiSpanSimplySupported
                    }
                } else if idx % 2 == 0 || prio <= 3 {
                    BridgeStructuralArchetype::Pre1975MultiSpanSimplySupported
                } else {
                    BridgeStructuralArchetype::ConventionalContinuousConcrete
                };

                let local_offset_km = ((mid.x - west_x).max(0.0)) / 1000.0;
                let r_km = (cfg.rupture_standoff_km + local_offset_km).max(1.5);
                let med_sa = median_sa10_gmpe_g(cfg.moment_magnitude_mw, r_km, cfg.vs30_m_per_s);
                let ex_base = base_arch.exceedance_probs(med_sa);

                SeismicBridgeAsset {
                    bridge_id: idx,
                    label,
                    segment_ids: sids,
                    midpoint: mid,
                    p0,
                    p1,
                    length_m: len_m,
                    base_archetype: base_arch,
                    effective_archetype: base_arch,
                    is_retrofitted: false,
                    site_distance_km: r_km,
                    median_site_sa10_g: med_sa,
                    analytical_exceedance_at_median: ex_base,
                    mc_state_pcts: [0.0; 5],
                    mc_closure_rate_pct: ex_base[2] * 100.0,
                    expected_func_day1_pct: 100.0,
                    expected_func_day7_pct: 100.0,
                    expected_func_day30_pct: 100.0,
                    expected_func_day90_pct: 100.0,
                    expected_repair_days: 0.0,
                }
            },
        )
        .collect();

    assets.sort_by(|a, b| {
        b.analytical_exceedance_at_median[2]
            .partial_cmp(&a.analytical_exceedance_at_median[2])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.bridge_id.cmp(&b.bridge_id))
    });
    for (rank_i, asset) in assets.iter_mut().enumerate() {
        asset.bridge_id = rank_i;
        if rank_i < cfg.retrofit_top_k_bridges
            && (cfg.retrofit_target_archetype as usize) > (asset.base_archetype as usize)
        {
            asset.effective_archetype = cfg.retrofit_target_archetype;
            asset.is_retrofitted = true;
            asset.analytical_exceedance_at_median =
                asset.effective_archetype.exceedance_probs(asset.median_site_sa10_g);
        }
    }

    assets
}

/// Newmark-Hall / FEMA Hazus conversion from `Sa(1.0 s)` (`g`) and `Vs30` (`m/s`) to Peak Ground
/// Velocity `PGV` (`cm/s`) and Peak Ground Acceleration `PGA` (`g`).
#[inline]
pub fn median_pgv_cm_s_from_sa10(sa10_g: f64, vs30_mps: f64) -> f64 {
    let site_amp = (760.0 / vs30_mps.clamp(140.0, 1100.0)).powf(0.14);
    (96.5 * sa10_g.max(0.001) * site_amp).clamp(0.5, 280.0)
}

#[inline]
pub fn median_pga_g_from_sa10(sa10_g: f64, vs30_mps: f64) -> f64 {
    let stiff = (vs30_mps.clamp(140.0, 1100.0) / 270.0).powf(0.16);
    (0.86 * sa10_g.max(0.001) * stiff).clamp(0.005, 3.0)
}

#[derive(Clone)]
struct EpochTrialPoint {
    parcel_cov_pct: f64,
    resident_cov_pct: f64,
    p90_travel_s: f64,
    closed_bridges: usize,
    mean_bridge_func: f64,
    electric_service_pct: f64,
    water_20psi_service_pct: f64,
    wastewater_service_pct: f64,
    composite_lifeline_pct: f64,
    water_repairs_rem: usize,
    sewer_repairs_rem: usize,
    retains_target: bool,
    closed_segment_ids: Vec<usize>,
}

/// Executes `cfg.monte_carlo_trials` spatially correlated earthquake realizations AND multi-week
/// Hazus bridge restoration trajectories (`Days 0, 1, 3, 7, 14, 30, 60, 90`) in parallel via `Rayon`.
pub fn run_seismic_monte_carlo_simulation(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
    cfg: &SeismicScenarioConfig,
) -> SeismicMonteCarloReport {
    let t0 = Instant::now();
    let n_segs = net.segments.len();
    let n_parcels = parcels.len();
    let trials = cfg.monte_carlo_trials.clamp(20, 5_000);
    let threshold_s = cfg.threshold_travel_s.clamp(30.0, 900.0);
    let target_pct = cfg.target_coverage_pct.clamp(1.0, 100.0);
    let crew_speedup = cfg.repair_crew_speedup.clamp(0.25, 5.0);

    let mut bridges = enumerate_seismic_bridge_inventory(net, facilities, cfg);
    let n_b = bridges.len();

    let b_range_km = cfg.spatial_corr_range_km.max(1.0);
    let mut corr_mat = vec![0.0; n_b * n_b];
    for i in 0..n_b {
        for j in 0..n_b {
            if i == j {
                corr_mat[i * n_b + j] = 1.0;
            } else {
                let h_km = bridges[i].midpoint.dist(bridges[j].midpoint) / 1000.0;
                corr_mat[i * n_b + j] = (-3.0 * h_km / b_range_km).exp();
            }
        }
    }
    let chol_l = if n_b > 0 {
        cholesky_lower_triangular(&corr_mat, n_b)
    } else {
        Vec::new()
    };

    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    let base_seg_trav_s: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                f64::INFINITY
            } else {
                let eff_v = (s.ema_speed_mps.max(s.speed_mps * 0.48) * speed_boost).clamp(3.2, 29.0);
                let stall_pen = if s.stalled_lane.is_some() && s.lane_count == 1 {
                    28.0
                } else {
                    0.0
                };
                s.length / eff_v + stall_pen
            }
        })
        .collect();

    let mut fwd_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
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
                let turn_pen = match conn.turn_type {
                    TurnType::Through => 0.5,
                    TurnType::Right => 2.2,
                    TurnType::Left => 3.0,
                };
                if !fwd_edges[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    fwd_edges[seg.id].push((out_sid, sig_pen + turn_pen));
                }
            }
        }
    }

    let mut twin_rev: Vec<Option<usize>> = vec![None; n_segs];
    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let u = seg.from_junc;
        let v = seg.to_junc;
        twin_rev[seg.id] = net
            .segments
            .iter()
            .find(|s| s.id != seg.id && !s.closed && s.from_junc == v && s.to_junc == u)
            .map(|s| s.id);
    }

    let mut access_matrix_s = vec![f64::INFINITY; n_parcels * n_segs];
    for (p_idx, p) in parcels.iter().enumerate() {
        let row_off = p_idx * n_segs;
        for (sid, s) in net.segments.iter().enumerate() {
            if !s.base_pts.is_empty() {
                let mid = s.base_pts[s.base_pts.len() / 2];
                access_matrix_s[row_off + sid] = p.center.dist(mid) / 7.5;
            }
        }
    }

    let turnout_s = if evp_cfg.include_nfpa_turnout_time {
        80.0
    } else {
        0.0
    };
    let fire_stations: Vec<(usize, f64)> = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
        .map(|f| (f.segment_id, f.s_m))
        .collect();
    let parcel_residents: Vec<f64> = parcels.iter().map(|p| p.residents()).collect();
    let total_residents: f64 = parcel_residents.iter().sum();

    let eval_dijkstra_with_speed_factors = |seg_speed_mult: &[f64]| -> (f64, f64, f64, f64, usize) {
        if n_parcels == 0 {
            return (100.0, 100.0, 0.0, 0.0, 0);
        }
        if fire_stations.is_empty() || n_segs == 0 {
            return (0.0, 0.0, 999.0, 999.0, n_parcels);
        }
        let mut dist = vec![f64::INFINITY; n_segs];
        let mut heap = BinaryHeap::new();

        for &(st_seg, st_s_m) in &fire_stations {
            if st_seg >= n_segs {
                continue;
            }
            let mult = seg_speed_mult[st_seg];
            let len = net.segments[st_seg].length.max(1.0);
            let rem_frac = ((len - st_s_m) / len).clamp(0.1, 1.0);
            if mult > 0.01 && base_seg_trav_s[st_seg].is_finite() {
                let init_s = turnout_s + (base_seg_trav_s[st_seg] / mult) * rem_frac;
                if init_s < dist[st_seg] {
                    dist[st_seg] = init_s;
                    heap.push(DijkNode {
                        cost_s: init_s,
                        seg_id: st_seg,
                    });
                }
            }
            if let Some(tw) = twin_rev[st_seg] {
                let tw_mult = seg_speed_mult[tw];
                if tw_mult > 0.01 && base_seg_trav_s[tw].is_finite() {
                    let tw_init =
                        turnout_s + (base_seg_trav_s[tw] / tw_mult) * (1.0 - rem_frac + 0.15);
                    if tw_init < dist[tw] {
                        dist[tw] = tw_init;
                        heap.push(DijkNode {
                            cost_s: tw_init,
                            seg_id: tw,
                        });
                    }
                }
            }
        }

        while let Some(DijkNode { cost_s, seg_id }) = heap.pop() {
            if cost_s > dist[seg_id] {
                continue;
            }
            for &(nxt, fixed_pen) in &fwd_edges[seg_id] {
                let mult = seg_speed_mult[nxt];
                if mult <= 0.01 || !base_seg_trav_s[nxt].is_finite() {
                    continue;
                }
                let nc = cost_s + fixed_pen + (base_seg_trav_s[nxt] / mult);
                if nc < dist[nxt] {
                    dist[nxt] = nc;
                    heap.push(DijkNode {
                        cost_s: nc,
                        seg_id: nxt,
                    });
                }
            }
        }

        let mut cov_p = 0usize;
        let mut cov_r = 0.0;
        let mut isolated = 0usize;
        let mut p_times = Vec::with_capacity(n_parcels);

        for p_idx in 0..n_parcels {
            let row_off = p_idx * n_segs;
            let mut best = f64::INFINITY;
            for (sid, &d_seg) in dist.iter().enumerate() {
                if d_seg.is_finite() {
                    let tot = d_seg + access_matrix_s[row_off + sid];
                    if tot < best {
                        best = tot;
                    }
                }
            }
            let clamped = if best.is_finite() { best } else { 999.0 };
            if clamped >= 998.5 {
                isolated += 1;
            }
            if clamped <= threshold_s {
                cov_p += 1;
                cov_r += parcel_residents[p_idx];
            }
            p_times.push(clamped);
        }

        p_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mean_s = p_times.iter().sum::<f64>() / n_parcels as f64;
        let p90_idx = (((n_parcels - 1) as f64) * 0.90).round() as usize;
        let p90_s = p_times[p90_idx.min(n_parcels - 1)];
        let p_cov = (cov_p as f64 / n_parcels as f64) * 100.0;
        let r_cov = if total_residents > 0.0 {
            (cov_r / total_residents) * 100.0
        } else {
            p_cov
        };
        (p_cov, r_cov, p90_s, mean_s, isolated)
    };

    let intact_mult = vec![1.0; n_segs];
    let (base_p_cov, base_r_cov, base_p90_s, base_mean_s, _) =
        eval_dijkstra_with_speed_factors(&intact_mult);

    let tau = cfg.inter_event_sigma_ln.clamp(0.0, 1.0);
    let phi = cfg.intra_event_sigma_ln.clamp(0.0, 1.2);
    let n_epochs = RECOVERY_TIMELINE_DAYS.len();

    // =========================================================================
    // PRECOMPUTE BURIED WATER / SEWER PIPELINE INVENTORY & LIFELINE TOPOLOGY
    // =========================================================================
    let n_junc = net.junctions.len();
    let b_box = net.bounds;
    let span_x = (b_box.max_x - b_box.min_x).max(50.0);
    let span_y = (b_box.max_y - b_box.min_y).max(50.0);
    let elev_m: Vec<f64> = net
        .junctions
        .iter()
        .map(|j| {
            let nx = ((j.pos.x - b_box.min_x) / span_x).clamp(0.0, 1.0);
            let ny = ((j.pos.y - b_box.min_y) / span_y).clamp(0.0, 1.0);
            26.0 - 13.5 * nx - 9.5 * ny + 1.4 * ((nx * 6.28).sin() * (ny * 3.14).cos())
        })
        .collect();

    #[derive(Clone)]
    struct PipeCorridor {
        u: usize,
        v: usize,
        label: String,
        mid: Vec2,
        len_m: f64,
        is_arterial: bool,
        water_diam_mm: f64,
        sewer_diam_mm: f64,
        water_mat: BuriedPipeMaterialClass,
        sewer_mat: BuriedPipeMaterialClass,
        replaced: bool,
        med_pgv_cm_s: f64,
        water_rr_km: f64,
        sewer_rr_km: f64,
        hw_k: f64,
        bridge_seg_ids: Vec<usize>,
    }

    let mut pipe_corridors: Vec<PipeCorridor> = Vec::new();
    let mut junc_pipe_adj: Vec<Vec<usize>> = vec![Vec::new(); n_junc];
    let west_x = b_box.min_x;

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
        if let Some(pi) = pipe_corridors
            .iter()
            .position(|c| (c.u == u && c.v == v) || (c.u == v && c.v == u))
        {
            if is_art {
                pipe_corridors[pi].is_arterial = true;
            }
            pipe_corridors[pi].bridge_seg_ids.push(seg.id);
        } else {
            let mid = if seg.base_pts.is_empty() {
                net.junctions[u].pos
            } else {
                seg.base_pts[seg.base_pts.len() / 2]
            };
            let local_km = ((mid.x - west_x).max(0.0)) / 1000.0;
            let r_km = (cfg.rupture_standoff_km + local_km).max(1.5);
            let med_sa = median_sa10_gmpe_g(cfg.moment_magnitude_mw, r_km, cfg.vs30_m_per_s);
            let med_pgv = median_pgv_cm_s_from_sa10(med_sa, cfg.vs30_m_per_s);
            let w_d_mm: f64 = if is_art { 350.0 } else { 200.0 };
            let s_d_mm: f64 = if is_art { 450.0 } else { 280.0 };
            let len_m = seg.length.max(15.0);
            let d_m = w_d_mm / 1000.0;
            let r_pipe = 10.67 * len_m / (125.0_f64.powf(1.852) * d_m.powf(4.8704));
            let hw_k = (1.0 / r_pipe.max(1e-9)).powf(1.0 / 1.852);

            let pi = pipe_corridors.len();
            pipe_corridors.push(PipeCorridor {
                u,
                v,
                label: format!("{} Main (#{})", seg.name, seg.id),
                mid,
                len_m,
                is_arterial: is_art,
                water_diam_mm: w_d_mm,
                sewer_diam_mm: s_d_mm,
                water_mat: cfg.baseline_water_pipe_material,
                sewer_mat: cfg.baseline_sewer_pipe_material,
                replaced: false,
                med_pgv_cm_s: med_pgv,
                water_rr_km: 0.0,
                sewer_rr_km: 0.0,
                hw_k,
                bridge_seg_ids: vec![seg.id],
            });
            junc_pipe_adj[u].push(pi);
            junc_pipe_adj[v].push(pi);
        }
    }

    // Prioritize top `seismic_pipe_replacement_pct`% corridors (by `med_pgv * water_diam_mm * len_m`) for ERDIP/HDPE replacement!
    let n_pipes = pipe_corridors.len();
    let n_replace = (((n_pipes as f64) * (cfg.seismic_pipe_replacement_pct.clamp(0.0, 100.0) / 100.0))
        .round() as usize)
        .min(n_pipes);
    let mut rank_order: Vec<usize> = (0..n_pipes).collect();
    rank_order.sort_by(|&a, &b| {
        let sa = pipe_corridors[a].med_pgv_cm_s * pipe_corridors[a].water_diam_mm * pipe_corridors[a].len_m;
        let sb = pipe_corridors[b].med_pgv_cm_s * pipe_corridors[b].water_diam_mm * pipe_corridors[b].len_m;
        sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
    });
    for &pi in rank_order.iter().take(n_replace) {
        pipe_corridors[pi].water_mat = BuriedPipeMaterialClass::SeismicErdipHdpe;
        pipe_corridors[pi].sewer_mat = BuriedPipeMaterialClass::SeismicErdipHdpe;
        pipe_corridors[pi].replaced = true;
    }
    for pc in &mut pipe_corridors {
        pc.water_rr_km = ala_hazus_pipe_repair_rate_per_km(
            pc.med_pgv_cm_s,
            cfg.pipe_fragility_law,
            pc.water_mat,
            pc.water_diam_mm,
            false,
        );
        pc.sewer_rr_km = ala_hazus_pipe_repair_rate_per_km(
            pc.med_pgv_cm_s,
            cfg.pipe_fragility_law,
            pc.sewer_mat,
            pc.sewer_diam_mm,
            true,
        );
    }

    // Anchor Substations, Water Booster Plants, Lift Station & WWTP on `net.junctions`
    let mut junc_elev_desc: Vec<usize> = (0..n_junc).collect();
    junc_elev_desc.sort_by(|&a, &b| elev_m[b].partial_cmp(&elev_m[a]).unwrap_or(Ordering::Equal));
    let sub1_j = *junc_elev_desc.first().unwrap_or(&0);
    let sub2_j = (0..n_junc)
        .filter(|&j| j != sub1_j)
        .max_by(|&a, &b| {
            net.junctions[a]
                .pos
                .dist(net.junctions[sub1_j].pos)
                .partial_cmp(&net.junctions[b].pos.dist(net.junctions[sub1_j].pos))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(sub1_j);
    let sub_juncs = if sub2_j != sub1_j && n_junc >= 4 {
        vec![sub1_j, sub2_j]
    } else if n_junc > 0 {
        vec![sub1_j]
    } else {
        Vec::new()
    };
    let wtp_juncs = sub_juncs.clone();
    let wwtp_j = *junc_elev_desc.last().unwrap_or(&0);

    //Shortest hops to WWTP for gravity sewer routing
    let mut hops_wwtp = vec![usize::MAX; n_junc];
    if n_junc > 0 {
        let mut qw = VecDeque::new();
        hops_wwtp[wwtp_j] = 0;
        qw.push_back(wwtp_j);
        while let Some(u) = qw.pop_front() {
            for &pi in &junc_pipe_adj[u] {
                let pc = &pipe_corridors[pi];
                let v = if pc.u == u { pc.v } else { pc.u };
                if hops_wwtp[v] == usize::MAX {
                    hops_wwtp[v] = hops_wwtp[u] + 1;
                    qw.push_back(v);
                }
            }
        }
    }
    let mut sewer_drain_order: Vec<usize> = (0..n_junc).collect();
    sewer_drain_order.sort_by(|&a, &b| {
        hops_wwtp[b]
            .cmp(&hops_wwtp[a])
            .then_with(|| elev_m[b].partial_cmp(&elev_m[a]).unwrap_or(Ordering::Equal))
    });

    // Baseline per-junction municipal water demand (`L/s`) + fire-hydrant demand & dry-weather sewer (`L/s`)
    let mut base_junc_water_lps = vec![0.05f64; n_junc];
    let mut base_junc_sewer_lps = vec![0.04f64; n_junc];
    for p in parcels {
        if n_junc == 0 {
            break;
        }
        let mut nj = 0usize;
        let mut bd2 = f64::INFINITY;
        for (ji, j) in net.junctions.iter().enumerate() {
            let d2 = (p.center.x - j.pos.x).powi(2) + (p.center.y - j.pos.y).powi(2);
            if d2 < bd2 {
                bd2 = d2;
                nj = ji;
            }
        }
        let wlps = (0.0028 * p.residents() + 0.0019 * p.jobs()).max(0.02);
        base_junc_water_lps[nj] += wlps;
        base_junc_sewer_lps[nj] += 0.85 * wlps;
    }
    if n_junc > 0 {
        base_junc_water_lps[n_junc / 2] += 63.0; // `1,000 GPM` post-earthquake firefighting demand!
    }

    // Fast coupled multi-lifeline evaluator returning `(elec_pct, water_20psi_pct, sewer_nosso_pct)`
    let eval_lifelines_state = |sub_online: &[bool],
                                closed_bridge_segs: &[usize],
                                water_breaks: &[usize],
                                water_leaks: &[usize],
                                sewer_breaks: &[usize]|
     -> (f64, f64, f64) {
        if n_junc == 0 {
            return (100.0, 100.0, 100.0);
        }
        // 1. Electrical BFS energization (bridge closures sever bridge-hung feeders!)
        let mut energized = vec![false; n_junc];
        let mut q_e = VecDeque::new();
        for (si, &sj) in sub_juncs.iter().enumerate() {
            if *sub_online.get(si).unwrap_or(&false) {
                energized[sj] = true;
                q_e.push_back(sj);
            }
        }
        while let Some(u) = q_e.pop_front() {
            for &pi in &junc_pipe_adj[u] {
                let pc = &pipe_corridors[pi];
                let bridge_cut = pc
                    .bridge_seg_ids
                    .iter()
                    .any(|sid| closed_bridge_segs.contains(sid));
                if bridge_cut {
                    continue;
                }
                let v = if pc.u == u { pc.v } else { pc.u };
                if !energized[v] {
                    energized[v] = true;
                    q_e.push_back(v);
                }
            }
        }
        let elec_pct =
            (energized.iter().filter(|&&e| e).count() as f64 / n_junc as f64) * 100.0;

        // 2. Potable Water Hazen-Williams head with severed main breaks + pressurized leak emitters (`12 L/s` per leak)
        let booster_on = cfg.critical_lifeline_backup_power
            || wtp_juncs.iter().any(|&wj| energized[wj]);
        let added_h = if booster_on { 52.0 } else { 15.0 };

        // Reachability from water plants across unbroken water mains
        let mut w_reach = vec![false; n_junc];
        let mut q_w = VecDeque::new();
        for &wj in &wtp_juncs {
            w_reach[wj] = true;
            q_w.push_back(wj);
        }
        while let Some(u) = q_w.pop_front() {
            for &pi in &junc_pipe_adj[u] {
                if water_breaks[pi] > 0 {
                    continue;
                }
                let pc = &pipe_corridors[pi];
                let v = if pc.u == u { pc.v } else { pc.u };
                if !w_reach[v] {
                    w_reach[v] = true;
                    q_w.push_back(v);
                }
            }
        }

        let mut junc_dem_m3s: Vec<f64> =
            base_junc_water_lps.iter().map(|&l| l / 1000.0).collect();
        for pi in 0..n_pipes {
            let nbrk = water_breaks[pi];
            let nlk = water_leaks[pi];
            if nbrk > 0 || nlk > 0 {
                // Unrepaired breaks bleed ~16.0 L/s at pressurized valve collars while severing through-flow;
                // unrepaired annular leaks bleed ~9.5 L/s. Repairing either strictly reduces nodal bleed!
                let bleed_m3s = ((nbrk as f64) * 0.0160 + (nlk as f64) * 0.0095)
                    * if booster_on { 1.0 } else { 0.35 };
                let pc = &pipe_corridors[pi];
                junc_dem_m3s[pc.u] += 0.5 * bleed_m3s;
                junc_dem_m3s[pc.v] += 0.5 * bleed_m3s;
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

        for _it in 0..32 {
            let mut max_d = 0.0f64;
            for i in 0..n_junc {
                if !w_reach[i] || wtp_juncs.contains(&i) {
                    continue;
                }
                let mut g_sum = 0.0;
                let mut gh_sum = 0.0;
                for &pi in &junc_pipe_adj[i] {
                    if water_breaks[pi] > 0 {
                        continue;
                    }
                    let pc = &pipe_corridors[pi];
                    let nb = if pc.u == i { pc.v } else { pc.u };
                    if !w_reach[nb] {
                        continue;
                    }
                    let dh = (head[i] - head[nb]).abs().max(0.01);
                    let g = pc.hw_k * dh.powf(-0.460043);
                    g_sum += g;
                    gh_sum += g * head[nb];
                }
                if g_sum > 1e-12 {
                    let h_t = (gh_sum - junc_dem_m3s[i]) / g_sum;
                    let h_n = 0.45 * head[i] + 0.55 * h_t;
                    max_d = max_d.max((h_n - head[i]).abs());
                    head[i] = h_n;
                }
            }
            if max_d < 5e-3 {
                break;
            }
        }
        let water_ok_cnt = (0..n_junc)
            .filter(|&i| {
                w_reach[i]
                    && ((head[i] - elev_m[i]) * PSI_PER_METER_H2O) >= MIN_FIRE_FLOW_RESIDUAL_PSI
            })
            .count();
        let water_pct = (water_ok_cnt as f64 / n_junc as f64) * 100.0;

        // 3. Wastewater gravity conveyance + collapsed sewer breaks + WWTP power status
        let wwtp_on = cfg.critical_lifeline_backup_power || energized[wwtp_j];
        let mut sso = vec![false; n_junc];
        let mut s_flow = base_junc_sewer_lps.clone();
        for &u in &sewer_drain_order {
            if u == wwtp_j {
                continue;
            }
            let mut open_down: Option<usize> = None;
            for &pi in &junc_pipe_adj[u] {
                if sewer_breaks[pi] > 0 {
                    continue;
                }
                let pc = &pipe_corridors[pi];
                let v = if pc.u == u { pc.v } else { pc.u };
                if hops_wwtp[v] < hops_wwtp[u] {
                    open_down = Some(v);
                    break;
                }
            }
            if let Some(vd) = open_down {
                s_flow[vd] += s_flow[u];
            } else {
                // Collapsed gravity sewer main backs up upstream manhole (`SSO`)!
                sso[u] = true;
            }
        }
        if !wwtp_on {
            sso[wwtp_j] = true;
        }
        let sewer_ok_cnt = sso.iter().filter(|&&bad| !bad).count();
        let sewer_pct = if !wwtp_on {
            ((sewer_ok_cnt as f64 / n_junc as f64) * 65.0).clamp(0.0, 100.0)
        } else {
            (sewer_ok_cnt as f64 / n_junc as f64) * 100.0
        };

        (elec_pct, water_pct, sewer_pct)
    };

    let base_crews = (cfg.water_sewer_repair_crews.max(1) as f64) * crew_speedup;

    // Parallel Rayon Monte Carlo loop evaluating Day 0 impact + multi-week Hazus & ALA restoration across `RECOVERY_TIMELINE_DAYS`!
    struct PerTrialFullOutput {
        day0_summary: MonteCarloTrialSummary,
        states: Vec<SeismicDamageState>,
        repair_days: Vec<f64>,
        pipe_w_breaks: Vec<usize>,
        pipe_w_leaks: Vec<usize>,
        pipe_s_breaks: Vec<usize>,
        pipe_s_leaks: Vec<usize>,
        sub_out_day0: usize,
        epoch_points: Vec<EpochTrialPoint>,
    }

    let trial_outputs: Vec<PerTrialFullOutput> = (0..trials)
        .into_par_iter()
        .map(|trial_idx| {
            let seed = cfg
                .random_seed
                .wrapping_mul(0x517C_C1B7_2722_0A95)
                .wrapping_add(
                    (trial_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ 0xA5A5_5A5A_F0F0_0F0F,
                );
            let mut rng = SplitMix64::new(seed);

            let eta = rng.next_standard_normal();
            let mut indep_z = Vec::with_capacity(n_b);
            let mut unif_u = Vec::with_capacity(n_b);
            let mut repair_z = Vec::with_capacity(n_b);
            for _ in 0..n_b {
                indep_z.push(rng.next_standard_normal());
                unif_u.push(rng.next_f64_open01());
                repair_z.push(rng.next_standard_normal());
            }

            let mut states = Vec::with_capacity(n_b);
            let mut repair_days = Vec::with_capacity(n_b);

            for i in 0..n_b {
                let mut corr_z = 0.0;
                let row_off = i * n_b;
                for k in 0..=i {
                    corr_z += chol_l[row_off + k] * indep_z[k];
                }
                let ln_im = bridges[i].median_site_sa10_g.ln() + tau * eta + phi * corr_z;
                let sa10_realized_g = ln_im.exp().clamp(0.001, 6.0);
                let ex_probs = bridges[i]
                    .effective_archetype
                    .exceedance_probs(sa10_realized_g);
                let ds = SeismicDamageState::from_uniform_sample(unif_u[i], &ex_probs);
                states.push(ds);

                // Sample physical Hazus restoration completion day `T_rep ~ max(0.05, Normal(mean_ds, sd_ds)) / crew_speedup`
                let t_rep = if ds == SeismicDamageState::None {
                    0.0
                } else {
                    let (m_d, s_d) = ds.hazus_restoration_mean_sd_days();
                    ((m_d + s_d * repair_z[i]).max(m_d * 0.15)) / crew_speedup
                };
                repair_days.push(t_rep);
            }

            // Sample spatially correlated buried pipeline leaks/breaks & Hazus substation outages
            // from an isolated CRN child stream sharing the exact earthquake event residual `eta`!
            let mut pipe_rng = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
            let zone_z = pipe_rng.next_standard_normal();

            // Hazus `ESS1` Medium-Voltage Substation PGA fragility (`Extensive/Trip median = 0.42g, beta = 0.55`)
            let mut sub_repair_days = Vec::with_capacity(sub_juncs.len());
            let mut sub_out_day0 = 0usize;
            for &sj in &sub_juncs {
                let loc_km = ((net.junctions[sj].pos.x - west_x).max(0.0)) / 1000.0;
                let r_km = (cfg.rupture_standoff_km + loc_km).max(1.5);
                let med_sa = median_sa10_gmpe_g(cfg.moment_magnitude_mw, r_km, cfg.vs30_m_per_s);
                let med_pga = median_pga_g_from_sa10(med_sa, cfg.vs30_m_per_s);
                let z_sub = 0.55 * zone_z + 0.835 * pipe_rng.next_standard_normal();
                let pga_real = (med_pga.ln() + tau * eta + phi * z_sub).exp().clamp(0.005, 4.0);
                let u_sub = pipe_rng.next_f64_open01();
                let z_rep = pipe_rng.next_standard_normal();
                let p_comp = normal_cdf_phi((pga_real.ln() - 0.78_f64.ln()) / 0.55);
                let p_ext = normal_cdf_phi((pga_real.ln() - 0.42_f64.ln()) / 0.55);
                let t_sub = if u_sub < p_comp {
                    sub_out_day0 += 1;
                    ((14.0 + 6.5 * z_rep).max(2.0)) / crew_speedup
                } else if u_sub < p_ext {
                    sub_out_day0 += 1;
                    ((3.0 + 2.0 * z_rep).max(0.4)) / crew_speedup
                } else {
                    0.0
                };
                sub_repair_days.push(t_sub);
            }

            let mut pipe_w_breaks = Vec::with_capacity(n_pipes);
            let mut pipe_w_leaks = Vec::with_capacity(n_pipes);
            let mut pipe_s_breaks = Vec::with_capacity(n_pipes);
            let mut pipe_s_leaks = Vec::with_capacity(n_pipes);

            for pc in &pipe_corridors {
                let z_loc = 0.65 * zone_z + 0.76 * pipe_rng.next_standard_normal();
                let uw_b = pipe_rng.next_f64_open01();
                let uw_l = pipe_rng.next_f64_open01();
                let us_b = pipe_rng.next_f64_open01();
                let us_l = pipe_rng.next_f64_open01();

                let pgv_real = (pc.med_pgv_cm_s.ln() + tau * eta + 0.85 * phi * z_loc)
                    .exp()
                    .clamp(0.2, 350.0);
                let len_km = pc.len_m / 1000.0;
                let rr_w = ala_hazus_pipe_repair_rate_per_km(
                    pgv_real,
                    cfg.pipe_fragility_law,
                    pc.water_mat,
                    pc.water_diam_mm,
                    false,
                );
                let rr_s = ala_hazus_pipe_repair_rate_per_km(
                    pgv_real,
                    cfg.pipe_fragility_law,
                    pc.sewer_mat,
                    pc.sewer_diam_mm,
                    true,
                );
                let lam_w = rr_w * len_km;
                let lam_s = rr_s * len_km;

                // ALA (2001) / Hazus wave-propagation split: 20% breaks, 80% leaks
                pipe_w_breaks.push(sample_poisson_icdf(0.20 * lam_w, uw_b));
                pipe_w_leaks.push(sample_poisson_icdf(0.80 * lam_w, uw_l));
                pipe_s_breaks.push(sample_poisson_icdf(0.20 * lam_s, us_b));
                pipe_s_leaks.push(sample_poisson_icdf(0.80 * lam_s, us_l));
            }

            let tot_w_brk0: usize = pipe_w_breaks.iter().sum();
            let tot_w_lk0: usize = pipe_w_leaks.iter().sum();
            let tot_s_brk0: usize = pipe_s_breaks.iter().sum();
            let tot_s_lk0: usize = pipe_s_leaks.iter().sum();

            // Evaluate each horizon in `RECOVERY_TIMELINE_DAYS` (`[0, 1, 3, 7, 14, 30, 60, 90]`),
            // caching identical `seg_mult` & lifeline state vectors when unchanged!
            let mut epoch_points: Vec<EpochTrialPoint> = Vec::with_capacity(n_epochs);
            let mut prev_life_sig: Option<(Vec<bool>, usize, usize, usize, usize)> = None;
            let mut prev_life_res: (f64, f64, f64) = (100.0, 100.0, 100.0);
            let mut prev_mult: Option<Vec<f64>> = None;
            let mut prev_eval: (f64, f64, f64, f64, usize) =
                (base_p_cov, base_r_cov, base_p90_s, base_mean_s, 0);
            let mut day0_degraded = 0usize;
            let mut day0_mean_s = base_mean_s;
            let mut day0_isolated = 0usize;

            for (ep_i, &day) in RECOVERY_TIMELINE_DAYS.iter().enumerate() {
                let mut seg_mult = vec![1.0_f64; n_segs];
                let mut closed_seg_ids = Vec::new();
                let mut closed_b = 0usize;
                let mut deg_b = 0usize;
                let mut func_sum = 0.0;

                for i in 0..n_b {
                    let spd = states[i].realized_speed_factor_at_day(day, repair_days[i]);
                    // Smooth expected Hazus curve contribution for continuous functionality metric
                    func_sum += states[i].hazus_functionality_at_day(day, crew_speedup);
                    if spd <= 0.01 {
                        closed_b += 1;
                        for &sid in &bridges[i].segment_ids {
                            if sid < n_segs {
                                seg_mult[sid] = 0.0;
                                closed_seg_ids.push(sid);
                            }
                        }
                    } else if spd < 0.99 {
                        deg_b += 1;
                        for &sid in &bridges[i].segment_ids {
                            if sid < n_segs {
                                seg_mult[sid] = seg_mult[sid].min(spd);
                            }
                        }
                    }
                }

                let same_as_prev = prev_mult
                    .as_ref()
                    .map(|pm| {
                        pm.iter()
                            .zip(seg_mult.iter())
                            .all(|(&a, &b)| (a - b).abs() < 1e-9)
                    })
                    .unwrap_or(false);

                let eval_res = if same_as_prev {
                    prev_eval
                } else if closed_b == 0 && deg_b == 0 {
                    (base_p_cov, base_r_cov, base_p90_s, base_mean_s, 0)
                } else {
                    eval_dijkstra_with_speed_factors(&seg_mult)
                };
                prev_mult = Some(seg_mult);
                prev_eval = eval_res;

                if ep_i == 0 {
                    day0_degraded = deg_b;
                    day0_mean_s = eval_res.3;
                    day0_isolated = eval_res.4;
                }

                let mean_func = if n_b > 0 {
                    (func_sum / (n_b as f64)) * 100.0
                } else {
                    100.0
                };
                let retains =
                    eval_res.0 + 1e-6 >= target_pct && eval_res.1 + 1e-6 >= target_pct;

                // Crew-constrained break-first, leak-second water & sewer repair queue (Porter 2016 / Hazus):
                // Half of utility crews work potable water (`1.2` breaks/crew-day, `2.4` leaks/crew-day),
                // half work deeper gravity sewer trenches (`0.9` breaks/crew-day, `1.8` leaks/crew-day).
                let w_crew_days = 0.55 * base_crews * day;
                let s_crew_days = 0.45 * base_crews * day;

                let w_brk_fixed = ((w_crew_days * 1.2).floor() as usize).min(tot_w_brk0);
                let w_rem_cd = (w_crew_days - (w_brk_fixed as f64) / 1.2).max(0.0);
                let w_lk_fixed = ((w_rem_cd * 2.4).floor() as usize).min(tot_w_lk0);

                let s_brk_fixed = ((s_crew_days * 0.9).floor() as usize).min(tot_s_brk0);
                let s_rem_cd = (s_crew_days - (s_brk_fixed as f64) / 0.9).max(0.0);
                let s_lk_fixed = ((s_rem_cd * 1.8).floor() as usize).min(tot_s_lk0);

                let w_rep_rem = (tot_w_brk0 - w_brk_fixed) + (tot_w_lk0 - w_lk_fixed);
                let s_rep_rem = (tot_s_brk0 - s_brk_fixed) + (tot_s_lk0 - s_lk_fixed);

                let sub_on_now: Vec<bool> =
                    sub_repair_days.iter().map(|&td| day >= td).collect();
                let cur_sig = (
                    sub_on_now.clone(),
                    tot_w_brk0 - w_brk_fixed,
                    tot_w_lk0 - w_lk_fixed,
                    tot_s_brk0 - s_brk_fixed,
                    closed_seg_ids.len(),
                );

                let (elec_pct, water_pct, sewer_pct) =
                    if prev_life_sig.as_ref() == Some(&cur_sig) {
                        prev_life_res
                    } else {
                        // Clear breaks/leaks sequentially in priority order (arterials & largest mains first!)
                        let mut cur_wb = pipe_w_breaks.clone();
                        let mut rem_fix = w_brk_fixed;
                        for &pi in &rank_order {
                            if rem_fix == 0 {
                                break;
                            }
                            let take = rem_fix.min(cur_wb[pi]);
                            cur_wb[pi] -= take;
                            rem_fix -= take;
                        }
                        let mut cur_wl = pipe_w_leaks.clone();
                        let mut rem_lfix = w_lk_fixed;
                        for &pi in &rank_order {
                            if rem_lfix == 0 {
                                break;
                            }
                            let take = rem_lfix.min(cur_wl[pi]);
                            cur_wl[pi] -= take;
                            rem_lfix -= take;
                        }
                        let mut cur_sb = pipe_s_breaks.clone();
                        let mut rem_sfix = s_brk_fixed;
                        for &pi in &rank_order {
                            if rem_sfix == 0 {
                                break;
                            }
                            let take = rem_sfix.min(cur_sb[pi]);
                            cur_sb[pi] -= take;
                            rem_sfix -= take;
                        }
                        let res_l = eval_lifelines_state(
                            &sub_on_now,
                            &closed_seg_ids,
                            &cur_wb,
                            &cur_wl,
                            &cur_sb,
                        );
                        prev_life_sig = Some(cur_sig);
                        prev_life_res = res_l;
                        res_l
                    };

                let comp_life =
                    0.28 * eval_res.0 + 0.24 * elec_pct + 0.26 * water_pct + 0.22 * sewer_pct;

                epoch_points.push(EpochTrialPoint {
                    parcel_cov_pct: eval_res.0,
                    resident_cov_pct: eval_res.1,
                    p90_travel_s: eval_res.2,
                    closed_bridges: closed_b,
                    mean_bridge_func: mean_func,
                    electric_service_pct: elec_pct,
                    water_20psi_service_pct: water_pct,
                    wastewater_service_pct: sewer_pct,
                    composite_lifeline_pct: comp_life,
                    water_repairs_rem: w_rep_rem,
                    sewer_repairs_rem: s_rep_rem,
                    retains_target: retains,
                    closed_segment_ids: closed_seg_ids,
                });
            }

            let ep0 = &epoch_points[0];
            let day0_summary = MonteCarloTrialSummary {
                trial_idx,
                closed_bridges_count: ep0.closed_bridges,
                degraded_bridges_count: day0_degraded,
                closed_segment_ids: ep0.closed_segment_ids.clone(),
                post_parcel_cov_pct: ep0.parcel_cov_pct,
                post_resident_cov_pct: ep0.resident_cov_pct,
                post_p90_travel_s: ep0.p90_travel_s,
                post_mean_travel_s: day0_mean_s,
                isolated_parcels: day0_isolated,
                retains_nfpa_target: ep0.retains_target,
            };

            PerTrialFullOutput {
                day0_summary,
                states,
                repair_days,
                pipe_w_breaks,
                pipe_w_leaks,
                pipe_s_breaks,
                pipe_s_leaks,
                sub_out_day0,
                epoch_points,
            }
        })
        .collect();

    // Aggregate per-bridge empirical damage-state percentages + Day 1, 7, 30, 90 expected functionality!
    let inv_n_pct = 100.0 / (trials as f64);
    for b_idx in 0..n_b {
        let mut counts = [0usize; 5];
        let mut rep_sum = 0.0;
        let mut f1_sum = 0.0;
        let mut f7_sum = 0.0;
        let mut f30_sum = 0.0;
        let mut f90_sum = 0.0;

        for tout in &trial_outputs {
            let ds = tout.states[b_idx];
            counts[ds as usize] += 1;
            rep_sum += tout.repair_days[b_idx];
            f1_sum += ds.hazus_functionality_at_day(1.0, crew_speedup);
            f7_sum += ds.hazus_functionality_at_day(7.0, crew_speedup);
            f30_sum += ds.hazus_functionality_at_day(30.0, crew_speedup);
            f90_sum += ds.hazus_functionality_at_day(90.0, crew_speedup);
        }
        for k in 0..5 {
            bridges[b_idx].mc_state_pcts[k] = (counts[k] as f64) * inv_n_pct;
        }
        bridges[b_idx].mc_closure_rate_pct =
            bridges[b_idx].mc_state_pcts[3] + bridges[b_idx].mc_state_pcts[4];
        bridges[b_idx].expected_func_day1_pct = f1_sum * inv_n_pct;
        bridges[b_idx].expected_func_day7_pct = f7_sum * inv_n_pct;
        bridges[b_idx].expected_func_day30_pct = f30_sum * inv_n_pct;
        bridges[b_idx].expected_func_day90_pct = f90_sum * inv_n_pct;
        bridges[b_idx].expected_repair_days = rep_sum / (trials as f64);
    }
    bridges.sort_by(|a, b| {
        b.mc_closure_rate_pct
            .partial_cmp(&a.mc_closure_rate_pct)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.median_site_sa10_g
                    .partial_cmp(&a.median_site_sa10_g)
                    .unwrap_or(Ordering::Equal)
            })
    });

    let q_idx = |q: f64| -> usize {
        (((trials - 1) as f64) * q.clamp(0.0, 1.0)).round() as usize
    };

    // Synthesize `RecoveryEpochSummary` across `RECOVERY_TIMELINE_DAYS` (`[0, 1, 3, 7, 14, 30, 60, 90]`)
    let mut recovery_timeline: Vec<RecoveryEpochSummary> = Vec::with_capacity(n_epochs);
    for (ep_i, &day) in RECOVERY_TIMELINE_DAYS.iter().enumerate() {
        let mut ep_slice: Vec<&EpochTrialPoint> =
            trial_outputs.iter().map(|t| &t.epoch_points[ep_i]).collect();
        ep_slice.sort_by(|a, b| {
            a.parcel_cov_pct
                .partial_cmp(&b.parcel_cov_pct)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    b.p90_travel_s
                        .partial_cmp(&a.p90_travel_s)
                        .unwrap_or(Ordering::Equal)
                })
        });

        let p10_pt = ep_slice[q_idx(0.10)];
        let p50_cov = ep_slice[q_idx(0.50)].parcel_cov_pct;
        let p90_cov = ep_slice[q_idx(0.90)].parcel_cov_pct;
        let mean_cov =
            ep_slice.iter().map(|e| e.parcel_cov_pct).sum::<f64>() / (trials as f64);

        let mut r_covs: Vec<f64> = ep_slice.iter().map(|e| e.resident_cov_pct).collect();
        r_covs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let p10_res = r_covs[q_idx(0.10)];
        let mean_res = r_covs.iter().sum::<f64>() / (trials as f64);

        let mut p90_ts: Vec<f64> = ep_slice.iter().map(|e| e.p90_travel_s).collect();
        p90_ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mean_p90_t = p90_ts.iter().sum::<f64>() / (trials as f64);
        let tail_p90_t = p90_ts[q_idx(0.90)];

        let pass_n = ep_slice.iter().filter(|e| e.retains_target).count();
        let prob_target = (pass_n as f64 / trials as f64) * 100.0;
        let exp_closed =
            ep_slice.iter().map(|e| e.closed_bridges as f64).sum::<f64>() / (trials as f64);
        let mean_func =
            ep_slice.iter().map(|e| e.mean_bridge_func).sum::<f64>() / (trials as f64);
        let mean_elec =
            ep_slice.iter().map(|e| e.electric_service_pct).sum::<f64>() / (trials as f64);
        let mean_water = ep_slice
            .iter()
            .map(|e| e.water_20psi_service_pct)
            .sum::<f64>()
            / (trials as f64);
        let mean_sewer = ep_slice
            .iter()
            .map(|e| e.wastewater_service_pct)
            .sum::<f64>()
            / (trials as f64);
        let mean_comp_life = ep_slice
            .iter()
            .map(|e| e.composite_lifeline_pct)
            .sum::<f64>()
            / (trials as f64);
        let exp_w_rem = ep_slice
            .iter()
            .map(|e| e.water_repairs_rem as f64)
            .sum::<f64>()
            / (trials as f64);
        let exp_s_rem = ep_slice
            .iter()
            .map(|e| e.sewer_repairs_rem as f64)
            .sum::<f64>()
            / (trials as f64);

        let is_primary = HAZUS_BENCHMARK_DAYS
            .iter()
            .any(|&bd| (bd - day).abs() < 1e-6);
        let label = if day == 0.0 {
            "Day 0 (Impact)".to_string()
        } else {
            format!("Day {:.0}", day)
        };

        recovery_timeline.push(RecoveryEpochSummary {
            epoch_idx: ep_i,
            day,
            label,
            is_hazus_primary_benchmark: is_primary,
            mean_parcel_cov_pct: mean_cov,
            p10_parcel_cov_pct: p10_pt.parcel_cov_pct,
            p50_parcel_cov_pct: p50_cov,
            p90_parcel_cov_pct: p90_cov,
            mean_resident_cov_pct: mean_res,
            p10_resident_cov_pct: p10_res,
            mean_p90_travel_s: mean_p90_t,
            p90_tail_travel_s: tail_p90_t,
            prob_meeting_nfpa_target_pct: prob_target,
            expected_closed_bridges: exp_closed,
            mean_bridge_functionality_pct: mean_func,
            mean_electric_service_pct: mean_elec,
            mean_water_20psi_service_pct: mean_water,
            mean_wastewater_service_pct: mean_sewer,
            mean_composite_lifeline_pct: mean_comp_life,
            expected_water_repairs_remaining: exp_w_rem,
            expected_sewer_repairs_remaining: exp_s_rem,
            p10_closed_segment_ids: p10_pt.closed_segment_ids.clone(),
        });
    }

    // Compute Bruneau et al. (2003) 90-day normalized resilience integral `R_90 = (1/90) * int_0^90 (Cov(t)/Cov_base) dt`
    let denom_cov = base_p_cov.max(1.0);
    let mut area_cov_days = 0.0;
    let total_span_days = RECOVERY_TIMELINE_DAYS.last().copied().unwrap_or(90.0).max(1.0);
    for w in recovery_timeline.windows(2) {
        let dt_d = (w[1].day - w[0].day).max(0.0);
        let q0 = (w[0].mean_parcel_cov_pct / denom_cov).clamp(0.0, 1.0);
        let q1 = (w[1].mean_parcel_cov_pct / denom_cov).clamp(0.0, 1.0);
        area_cov_days += 0.5 * (q0 + q1) * dt_d;
    }
    let bruneau_resilience_index_90d_pct =
        ((area_cov_days / total_span_days) * 100.0).clamp(0.0, 100.0);

    // Interpolate post-event day when `mean_parcel_cov_pct` crosses `target_pct`
    let mut days_to_90pct_mean_recovery: Option<f64> = None;
    if recovery_timeline[0].mean_parcel_cov_pct + 1e-6 >= target_pct {
        days_to_90pct_mean_recovery = Some(0.0);
    } else {
        for w in recovery_timeline.windows(2) {
            if w[0].mean_parcel_cov_pct < target_pct
                && w[1].mean_parcel_cov_pct + 1e-6 >= target_pct
            {
                let dc = (w[1].mean_parcel_cov_pct - w[0].mean_parcel_cov_pct).max(1e-6);
                let frac = ((target_pct - w[0].mean_parcel_cov_pct) / dc).clamp(0.0, 1.0);
                days_to_90pct_mean_recovery = Some(w[0].day + frac * (w[1].day - w[0].day));
                break;
            }
        }
    }

    // Aggregate Day-0 buried pipeline break/leak statistics & 90-day multi-lifeline Bruneau indices
    let mut tot_wb = 0usize;
    let mut tot_wl = 0usize;
    let mut tot_sb = 0usize;
    let mut tot_sl = 0usize;
    let mut tot_sub_out = 0usize;
    let mut pipe_wb_trials = vec![0usize; n_pipes];
    let mut pipe_sb_trials = vec![0usize; n_pipes];
    let mut pipe_w_rep_sum = vec![0usize; n_pipes];
    let mut pipe_s_rep_sum = vec![0usize; n_pipes];

    for tout in &trial_outputs {
        tot_sub_out += tout.sub_out_day0;
        for pi in 0..n_pipes {
            let wb = tout.pipe_w_breaks[pi];
            let wl = tout.pipe_w_leaks[pi];
            let sb = tout.pipe_s_breaks[pi];
            let sl = tout.pipe_s_leaks[pi];
            tot_wb += wb;
            tot_wl += wl;
            tot_sb += sb;
            tot_sl += sl;
            if wb > 0 {
                pipe_wb_trials[pi] += 1;
            }
            if sb > 0 {
                pipe_sb_trials[pi] += 1;
            }
            pipe_w_rep_sum[pi] += wb + wl;
            pipe_s_rep_sum[pi] += sb + sl;
        }
    }

    let mut pipeline_inventory: Vec<SeismicPipelineAssetSummary> = pipe_corridors
        .iter()
        .enumerate()
        .map(|(pi, pc)| SeismicPipelineAssetSummary {
            corridor_id: pi,
            label: pc.label.clone(),
            u_junc: pc.u,
            v_junc: pc.v,
            midpoint: pc.mid,
            length_m: pc.len_m,
            water_diam_mm: pc.water_diam_mm,
            sewer_diam_mm: pc.sewer_diam_mm,
            water_material: pc.water_mat,
            sewer_material: pc.sewer_mat,
            is_seismically_replaced: pc.replaced,
            median_pgv_cm_s: pc.med_pgv_cm_s,
            water_rr_per_km: pc.water_rr_km,
            sewer_rr_per_km: pc.sewer_rr_km,
            mc_water_break_prob_pct: (pipe_wb_trials[pi] as f64) * inv_n_pct,
            mc_sewer_break_prob_pct: (pipe_sb_trials[pi] as f64) * inv_n_pct,
            mc_expected_water_repairs: (pipe_w_rep_sum[pi] as f64) / (trials as f64),
            mc_expected_sewer_repairs: (pipe_s_rep_sum[pi] as f64) / (trials as f64),
        })
        .collect();
    pipeline_inventory.sort_by(|a, b| {
        b.mc_expected_water_repairs
            .partial_cmp(&a.mc_expected_water_repairs)
            .unwrap_or(Ordering::Equal)
    });

    let integrate_90d = |f: fn(&RecoveryEpochSummary) -> f64| -> f64 {
        let mut area = 0.0;
        for w in recovery_timeline.windows(2) {
            let dt_d = (w[1].day - w[0].day).max(0.0);
            area += 0.5 * (f(&w[0]).clamp(0.0, 100.0) + f(&w[1]).clamp(0.0, 100.0)) * dt_d;
        }
        (area / total_span_days).clamp(0.0, 100.0)
    };
    let electric_resilience_index_90d_pct = integrate_90d(|e| e.mean_electric_service_pct);
    let water_resilience_index_90d_pct = integrate_90d(|e| e.mean_water_20psi_service_pct);
    let wastewater_resilience_index_90d_pct = integrate_90d(|e| e.mean_wastewater_service_pct);
    let multi_lifeline_resilience_90d_pct = integrate_90d(|e| e.mean_composite_lifeline_pct);

    let total_pipe_km: f64 = pipe_corridors.iter().map(|c| c.len_m / 1000.0).sum();
    let median_corridor_pgv_cm_s = if n_pipes > 0 {
        pipe_corridors.iter().map(|c| c.med_pgv_cm_s).sum::<f64>() / (n_pipes as f64)
    } else {
        median_pgv_cm_s_from_sa10(
            median_sa10_gmpe_g(
                cfg.moment_magnitude_mw,
                cfg.rupture_standoff_km,
                cfg.vs30_m_per_s,
            ),
            cfg.vs30_m_per_s,
        )
    };
    let ala_pgv_repair_curves = sample_ala_hazus_pgv_curves(cfg.pipe_fragility_law, 120.0, 41);

    let mut summaries: Vec<MonteCarloTrialSummary> =
        trial_outputs.into_iter().map(|t| t.day0_summary).collect();
    summaries.sort_by(|a, b| {
        a.post_parcel_cov_pct
            .partial_cmp(&b.post_parcel_cov_pct)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.post_p90_travel_s
                    .partial_cmp(&a.post_p90_travel_s)
                    .unwrap_or(Ordering::Equal)
            })
    });

    let worst_realization = summaries[0].clone();
    let p05_parcel_cov_pct = summaries[q_idx(0.05)].post_parcel_cov_pct;
    let p10_realization = summaries[q_idx(0.10)].clone();
    let p10_parcel_cov_pct = p10_realization.post_parcel_cov_pct;
    let p50_parcel_cov_pct = summaries[q_idx(0.50)].post_parcel_cov_pct;
    let mean_parcel_cov_pct =
        summaries.iter().map(|s| s.post_parcel_cov_pct).sum::<f64>() / (trials as f64);

    let mut res_covs: Vec<f64> = summaries.iter().map(|s| s.post_resident_cov_pct).collect();
    res_covs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p10_resident_cov_pct = res_covs[q_idx(0.10)];
    let mean_resident_cov_pct = res_covs.iter().sum::<f64>() / (trials as f64);

    let mut p90_times: Vec<f64> = summaries.iter().map(|s| s.post_p90_travel_s).collect();
    p90_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean_p90_travel_s = p90_times.iter().sum::<f64>() / (trials as f64);
    let p50_p90_travel_s = p90_times[q_idx(0.50)];
    let p90_p90_travel_s = p90_times[q_idx(0.90)];
    let p95_p90_travel_s = p90_times[q_idx(0.95)];

    let mut closed_counts: Vec<usize> = summaries.iter().map(|s| s.closed_bridges_count).collect();
    closed_counts.sort_unstable();
    let expected_closed_bridges =
        closed_counts.iter().sum::<usize>() as f64 / (trials as f64);
    let p90_closed_bridges = closed_counts[q_idx(0.90)];
    let expected_isolated_parcels = summaries
        .iter()
        .map(|s| s.isolated_parcels as f64)
        .sum::<f64>()
        / (trials as f64);

    let pass_cnt = summaries.iter().filter(|s| s.retains_nfpa_target).count();
    let prob_meeting_nfpa_target_pct = (pass_cnt as f64 / trials as f64) * 100.0;

    let mut hist = [0usize; 10];
    for s in &summaries {
        let bin = ((s.post_parcel_cov_pct / 10.0).floor() as isize).clamp(0, 9) as usize;
        hist[bin] += 1;
    }

    let cdf_pts = 25usize;
    let p90_travel_cdf: Vec<[f64; 2]> = (0..cdf_pts)
        .map(|i| {
            let q = i as f64 / (cdf_pts - 1) as f64;
            [q, p90_times[q_idx(q)]]
        })
        .collect();

    let ref_arch = bridges
        .first()
        .map(|b| b.effective_archetype)
        .unwrap_or(BridgeStructuralArchetype::Pre1975MultiSpanSimplySupported);
    let ref_curve = ref_arch.sample_curve_points(2.2, 45);
    let hazus_restoration_curves = sample_hazus_restoration_curves(90.0, crew_speedup, 46);
    let mean_median_sa10_g = if n_b > 0 {
        bridges.iter().map(|b| b.median_site_sa10_g).sum::<f64>() / (n_b as f64)
    } else {
        median_sa10_gmpe_g(
            cfg.moment_magnitude_mw,
            cfg.rupture_standoff_km,
            cfg.vs30_m_per_s,
        )
    };

    SeismicMonteCarloReport {
        config: cfg.clone(),
        fire_station_count: fire_stations.len(),
        bridge_count: n_b,
        baseline_parcel_cov_pct: base_p_cov,
        baseline_resident_cov_pct: base_r_cov,
        baseline_p90_travel_s: base_p90_s,
        mean_parcel_cov_pct,
        p50_parcel_cov_pct,
        p10_parcel_cov_pct,
        p05_parcel_cov_pct,
        mean_resident_cov_pct,
        p10_resident_cov_pct,
        mean_p90_travel_s,
        p50_p90_travel_s,
        p90_p90_travel_s,
        p95_p90_travel_s,
        prob_meeting_nfpa_target_pct,
        expected_closed_bridges,
        p90_closed_bridges,
        expected_isolated_parcels,
        coverage_histogram_bins: hist,
        p90_travel_cdf,
        reference_fragility_curve: ref_curve,
        reference_archetype: ref_arch,
        hazus_restoration_curves,
        recovery_timeline,
        bruneau_resilience_index_90d_pct,
        days_to_90pct_mean_recovery,
        mean_median_sa10_g,
        median_corridor_pgv_cm_s,
        total_water_pipe_km: total_pipe_km,
        total_sewer_pipe_km: total_pipe_km,
        expected_day0_water_leaks: (tot_wl as f64) / (trials as f64),
        expected_day0_water_breaks: (tot_wb as f64) / (trials as f64),
        expected_day0_sewer_leaks: (tot_sl as f64) / (trials as f64),
        expected_day0_sewer_breaks: (tot_sb as f64) / (trials as f64),
        expected_day0_substation_outages: (tot_sub_out as f64) / (trials as f64),
        electric_resilience_index_90d_pct,
        water_resilience_index_90d_pct,
        wastewater_resilience_index_90d_pct,
        multi_lifeline_resilience_90d_pct,
        ala_pgv_repair_curves,
        pipeline_inventory,
        bridge_inventory: bridges,
        p10_realization,
        worst_realization,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

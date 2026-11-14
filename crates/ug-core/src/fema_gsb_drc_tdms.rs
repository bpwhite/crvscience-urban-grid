//! Completing the FEMA Response & Recovery Architecture (`ESF #3` + `ESF #6` / Individual Assistance):
//! 1. **FEMA Generator Staging Base (`GSB`) & USACE `249th Engineer Battalion (Prime Power)` + `PRT`s**:
//!    Pre-installation switchgear/load assessments (`1.5–3.5 hrs/site`), prime-power mobile diesel
//!    generator sizing (`50 kW` C-POD / MRIC / BTS pack, `250 kW` Water Booster / Lift Station /
//!    Clean-Air Shelter pack, `800–1,500 kW` Trauma Hospital / WTP / WWTP pack), and post-debris
//!    convoy dispatch energizing dark municipal lifelines.
//! 2. **EPA / FEMA-325 Temporary Debris Management Sites (`TDMS` / `TDSRS`) & Phase II Curbside Haul-Off**:
//!    4-stream disaster debris segregation (`Vegetative` air-curtain/grinding `75%` volume reduction,
//!    `Construction & Demolition C&D` aggregate recycling, mandatory **EPA Phase 1 `HHW` hazardous
//!    waste sweep**, and `White Goods` EPA Sec. 608 Freon recovery) with certified load-ticket
//!    monitoring towers and `30 yd^3` (`9.5 ton`) tandem dump-truck cycle times.
//! 3. **FEMA Individual Assistance (`IA`) Disaster Recovery Centers (`DRC`s) & Roving `MRIC`s**:
//!    Fixed multi-agency `DRC`s (`FEMA IA` + `U.S. SBA` disaster loans + document scan + appeals,
//!    `Mon–Sat 07:00–19:00`, `26 min` mean service) paired with transit-accessible **Mobile
//!    Registration Intake Centers (`MRIC`s)** (`08:00–17:00`, rapid `11 min` intake) evaluated via
//!    `M/G/c` queueing and GTFS walkability equity across a 14-day recovery horizon.

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind};
use crate::fema_usar_debris::{run_fema_usar_debris_monte_carlo, FemaUsarDebrisConfig};
use crate::network::{RoadNetwork, Vec2};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

pub const FEMA_RECOVERY_FACILITY_CSV_HEADER: &str = "site_id,site_code,facility_kind,x_m,y_m,generator_kw,energized_hour,daily_capacity_or_tons,mean_wait_min,utilization_pct,transit_walk_equity_pct";
pub const FEMA_RECOVERY_DAILY_CSV_HEADER: &str = "day,critical_lifelines_energized_pct,curbside_debris_hauled_pct,tdms_volume_saved_yd3,ia_households_registered_pct,mean_drc_mric_wait_min,disbursed_ia_sba_musd";

/// Integrated execution policy governing USACE `249th Engineer` `GSB` prime-power dispatch,
/// EPA/FEMA-325 `TDMS` debris segregation, and `DRC` + `MRIC` survivor registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemaRecoveryExecutionPolicy {
    /// Ad-hoc FIFO generator dispatch, single direct-to-landfill haul (no `TDMS` `75%` grinding
    /// reduction or EPA Phase 1 `HHW` sweep), and fixed downtown `DRC`s only (no roving `MRIC`s).
    SequentialAdHoc,
    /// Prioritizes USACE `249th Engineer` `GSB` prime-power packs to Hospitals, Water/Sewer &
    /// Clean-Air Shelters first, paired with standard `TDMS` sorting and fixed `DRC`s.
    LifelinePowerFirst,
    /// Full FEMA `ESF #3` + `IA` co-optimization: pre-assessed `249th Engineer` `GSB` prime power +
    /// decentralized `TDMS` (`75%` vegetative reduction + EPA Phase 1 `HHW` + Freon recovery) +
    /// fixed `DRC`s (`FEMA IA + SBA`) augmented by GTFS-aligned roving `MRIC` intake vans.
    IntegratedFemaEsf3AndIaSuite,
}

impl FemaRecoveryExecutionPolicy {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::SequentialAdHoc => "Ad-Hoc Direct-Haul / Fixed DRC Only",
            Self::LifelinePowerFirst => "249th Prime-Power Lifelines First",
            Self::IntegratedFemaEsf3AndIaSuite => "Integrated GSB + TDMS (75% Red.) + DRC/MRIC",
        }
    }
}

/// Classification of FEMA recovery & emergency power infrastructure nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemaRecoverySiteKind {
    /// Regional FEMA Generator Staging Base (`GSB`) & USACE `249th Engineer` Prime Power yard.
    GeneratorStagingBase,
    /// Critical lifeline receiving a `GSB` prime-power mobile diesel pack (`50–1,250 kW`).
    CriticalLifelineGenTarget,
    /// FEMA-325 / EPA Temporary Debris Management Site (`TDMS` / `TDSRS`) with inspection tower.
    TemporaryDebrisSiteTdms,
    /// Fixed multi-agency FEMA + State + U.S. SBA Disaster Recovery Center (`DRC`).
    FixedDisasterRecoveryCenterDrc,
    /// Roving transit-accessible Mobile Registration Intake Center (`MRIC`).
    MobileRegistrationIntakeMric,
}

impl FemaRecoverySiteKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::GeneratorStagingBase => "FEMA-GSB",
            Self::CriticalLifelineGenTarget => "249th-GEN",
            Self::TemporaryDebrisSiteTdms => "FEMA-TDMS",
            Self::FixedDisasterRecoveryCenterDrc => "FEMA-DRC",
            Self::MobileRegistrationIntakeMric => "FEMA-MRIC",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaGsbDrcTdmsConfig {
    pub fema_usar: FemaUsarDebrisConfig,
    pub policy: FemaRecoveryExecutionPolicy,
    /// Number of USACE `249th Engineer Battalion` / Emergency Power `PRT` assessment & rig teams.
    pub usace_249th_prt_teams: usize,
    /// Total FEMA `GSB` mobile generator inventory (`50 kW` to `1,250 kW` trailer-mounted units).
    pub gsb_generator_packs_inventory: usize,
    /// Number of decentralized FEMA-325 Temporary Debris Management Sites (`TDMS`) opened (`1..=4`).
    pub active_tdms_sites: usize,
    /// Number of `30 yd^3` (`~9.5 ton`) certified ADMS haul-off dump trucks operating daily.
    pub debris_haul_dump_trucks: usize,
    /// Number of fixed multi-agency FEMA + SBA Disaster Recovery Centers (`DRC`s, `1..=5`).
    pub fixed_drc_centers: usize,
    /// Number of roving transit-accessible Mobile Registration Intake Centers (`MRIC`s, `0..=6`).
    pub mobile_mric_units: usize,
    /// Staffing specialists (`FEMA IA + SBA`) per fixed `DRC` (`4..=24`).
    pub specialists_per_drc: usize,
    /// Recovery simulation horizon (`days`, typically `14` days post-event).
    pub recovery_horizon_days: usize,
}

impl Default for FemaGsbDrcTdmsConfig {
    fn default() -> Self {
        Self {
            fema_usar: FemaUsarDebrisConfig::default(),
            policy: FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite,
            usace_249th_prt_teams: 4,
            gsb_generator_packs_inventory: 14,
            active_tdms_sites: 2,
            debris_haul_dump_trucks: 28,
            fixed_drc_centers: 2,
            mobile_mric_units: 3,
            specialists_per_drc: 12,
            recovery_horizon_days: 14,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaRecoveryFacilityRow {
    pub site_id: usize,
    pub site_code: String,
    pub kind: FemaRecoverySiteKind,
    pub world_pos: Vec2,
    pub generator_kw: f64,
    pub energized_hour: f64,
    pub daily_capacity_or_tons: f64,
    pub mean_wait_min: f64,
    pub utilization_pct: f64,
    pub transit_walk_equity_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaRecoveryDailyPoint {
    pub day: usize,
    pub critical_lifelines_energized_pct: f64,
    pub curbside_debris_hauled_pct: f64,
    pub tdms_volume_saved_yd3: f64,
    pub ia_households_registered_pct: f64,
    pub mean_drc_mric_wait_min: f64,
    pub disbursed_ia_sba_musd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaGsbDrcTdmsReport {
    pub config: FemaGsbDrcTdmsConfig,
    pub solve_time_ms: f64,
    pub gsb_world_pos: Vec2,
    /// USACE `249th Engineer` / `GSB` prime-power metrics
    pub total_critical_lifeline_targets: usize,
    pub energized_lifelines_72h_pct: f64,
    pub mean_time_to_emergency_power_hrs: f64,
    pub deployed_prime_power_mw: f64,
    /// EPA / FEMA-325 `TDMS` Phase II debris sorting & haul-off metrics
    pub total_disaster_debris_tons: f64,
    pub vegetative_debris_tons: f64,
    pub cd_structural_debris_tons: f64,
    pub hhw_hazardous_swept_tons: f64,
    pub white_goods_freon_units: usize,
    pub curbside_haul_completed_14d_pct: f64,
    pub tdms_landfill_volume_saved_yd3: f64,
    pub hhw_contamination_prevented_pct: f64,
    /// FEMA `IA` + `SBA` `DRC` & `MRIC` survivor intake metrics
    pub affected_ia_eligible_households: f64,
    pub ia_registered_14d_pct: f64,
    pub mean_survivor_drc_wait_min: f64,
    pub p90_survivor_access_plus_wait_min: f64,
    pub transit_walkable_intake_equity_pct: f64,
    pub projected_ia_and_sba_disbursement_musd: f64,
    /// Spatial vectors & per-parcel GPU heatmap cache
    pub gsb_dispatch_arcs: Vec<(Vec2, Vec2)>,
    pub tdms_haul_arcs: Vec<(Vec2, Vec2)>,
    pub facility_rows: Vec<FemaRecoveryFacilityRow>,
    pub daily_timeline: Vec<FemaRecoveryDailyPoint>,
    pub parcel_recovery_score_pct: Vec<f64>,
    pub parcel_drc_wait_min: Vec<f64>,
}

impl FemaGsbDrcTdmsReport {
    pub fn to_facility_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.facility_rows.len() * 140);
        out.push_str(FEMA_RECOVERY_FACILITY_CSV_HEADER);
        out.push('\n');
        for r in &self.facility_rows {
            out.push_str(&format!(
                "{},{},{},{:.2},{:.2},{:.0},{:.2},{:.1},{:.2},{:.1},{:.1}\n",
                r.site_id,
                r.site_code,
                r.kind.badge(),
                r.world_pos.x,
                r.world_pos.y,
                r.generator_kw,
                r.energized_hour,
                r.daily_capacity_or_tons,
                r.mean_wait_min,
                r.utilization_pct,
                r.transit_walk_equity_pct
            ));
        }
        out
    }

    pub fn to_daily_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.daily_timeline.len() * 96);
        out.push_str(FEMA_RECOVERY_DAILY_CSV_HEADER);
        out.push('\n');
        for d in &self.daily_timeline {
            out.push_str(&format!(
                "{},{:.2},{:.2},{:.1},{:.2},{:.2},{:.3}\n",
                d.day,
                d.critical_lifelines_energized_pct,
                d.curbside_debris_hauled_pct,
                d.tdms_volume_saved_yd3,
                d.ia_households_registered_pct,
                d.mean_drc_mric_wait_min,
                d.disbursed_ia_sba_musd
            ));
        }
        out
    }

    pub fn export_facility_csv_path(&self, path: &Path) -> Result<usize, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, self.to_facility_csv_string()).map_err(|e| e.to_string())?;
        Ok(self.facility_rows.len())
    }
}

/// Sizes a FEMA `GSB` prime-power diesel generator pack (`kW`) and USACE `249th Engineer`
/// electrical assessment + hookup duration (`hours`) per USACE Emergency Power `PRT` doctrine.
pub fn size_usace_249th_prime_power_pack(
    is_hospital: bool,
    is_water_or_wwtp: bool,
    is_shelter: bool,
    pre_assessed_registry: bool,
) -> (f64, f64) {
    let kw = if is_hospital {
        1150.0
    } else if is_water_or_wwtp {
        650.0
    } else if is_shelter {
        250.0
    } else {
        60.0
    };
    let base_rig_hrs = if is_hospital {
        3.6
    } else if is_water_or_wwtp {
        2.9
    } else if is_shelter {
        2.1
    } else {
        1.3
    };
    // USACE 249th pre-scripted National Generator Assessment database cuts on-site switchgear sizing by 42%
    let install_hrs = if pre_assessed_registry {
        base_rig_hrs * 0.58
    } else {
        base_rig_hrs
    };
    (kw, install_hrs)
}

/// Allen-Cunneen `M/G/c` survivor intake queue wait (`minutes`) at a FEMA `DRC` or `MRIC`
/// given hourly household arrival rate `lambda_hr`, `c` intake specialists, mean service time
/// `service_min`, and service time coefficient of variation `cv_s`.
pub fn evaluate_drc_mric_mgc_wait_min(
    lambda_hr: f64,
    specialists_c: usize,
    service_min: f64,
    cv_s: f64,
) -> (f64, f64) {
    let c = specialists_c.max(1) as f64;
    let mu_hr = 60.0 / service_min.max(2.0);
    let cap_hr = c * mu_hr;
    let rho = (lambda_hr / cap_hr).clamp(0.01, 1.35);
    if rho >= 0.985 {
        let overload_excess = (lambda_hr - 0.96 * cap_hr).max(0.0);
        // Over-capacity shift backlog spill (capped per 10h daily shift)
        let wait_min = 42.0 + (overload_excess / cap_hr.max(1.0)) * 165.0;
        return (wait_min.clamp(42.0, 240.0), (rho * 100.0).min(135.0));
    }
    // Erlang-C heavy-traffic approximation scaled by Allen-Cunneen `(1 + CV_s^2)/2`
    let erlang_p = rho.powf((2.0 * (c + 1.0)).sqrt() - 1.0);
    let lq = erlang_p * (rho / (1.0 - rho)) * 0.5 * (1.0 + cv_s * cv_s);
    let wq_min = (lq / lambda_hr.max(0.15)) * 60.0;
    (wq_min.clamp(0.5, 240.0), rho * 100.0)
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

/// Executes a `Rayon`-parallelized Monte Carlo evaluation coupling FEMA `GSB` (`249th Engineer`)
/// prime-power deployment, EPA/FEMA-325 `TDMS` Phase II debris sorting & haul-off, and `DRC`/`MRIC`
/// Individual Assistance survivor intake across `cfg.recovery_horizon_days`.
pub fn run_fema_gsb_drc_tdms_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &FemaGsbDrcTdmsConfig,
) -> FemaGsbDrcTdmsReport {
    let t0 = Instant::now();
    let usar_rep = run_fema_usar_debris_monte_carlo(net, parcels, facilities, &cfg.fema_usar);
    let b = net.bounds;
    let span_x = (b.max_x - b.min_x).max(120.0);
    let span_y = (b.max_y - b.min_y).max(120.0);

    // 1. Place Regional FEMA Generator Staging Base (`GSB`) near West highway interchange
    let gsb_pos = Vec2 {
        x: b.min_x + 0.08 * span_x,
        y: b.min_y + 0.54 * span_y,
    };

    // 2. Enumerate Critical Municipal Lifeline Targets needing Prime Power (`Hospitals`, `WTP/WWTP`, `Shelters`, `PODs`, `DRCs`)
    let pre_assessed = cfg.policy == FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite;
    let mut lifeline_targets: Vec<(String, Vec2, f64, f64, u8)> = Vec::new();
    for fac in facilities {
        let is_hosp = fac.kind == EmergencyFacilityKind::Hospital;
        let (kw, rig_h) = size_usace_249th_prime_power_pack(is_hosp, false, false, pre_assessed);
        let prio = if is_hosp { 0u8 } else { 2u8 };
        lifeline_targets.push((
            format!("{} #{}", fac.kind.short_prefix(), fac.id + 1),
            fac.world_pos,
            kw,
            rig_h,
            prio,
        ));
    }
    // Add Municipal Water Booster, Wastewater Treatment Plant, and 2 Clean-Air Congregate Shelters + 2 C-PODs
    let synthetic_lifelines = [
        (
            "WTP-Booster-1",
            Vec2 {
                x: b.min_x + 0.28 * span_x,
                y: b.min_y + 0.72 * span_y,
            },
            false,
            true,
            false,
            1u8,
        ),
        (
            "WWTP-Main",
            Vec2 {
                x: b.min_x + 0.84 * span_x,
                y: b.min_y + 0.24 * span_y,
            },
            false,
            true,
            false,
            1u8,
        ),
        (
            "LiftStation-N",
            Vec2 {
                x: b.min_x + 0.48 * span_x,
                y: b.min_y + 0.82 * span_y,
            },
            false,
            true,
            false,
            2u8,
        ),
        (
            "Shelter-HEPA-E",
            Vec2 {
                x: b.min_x + 0.68 * span_x,
                y: b.min_y + 0.62 * span_y,
            },
            false,
            false,
            true,
            1u8,
        ),
        (
            "Shelter-HEPA-W",
            Vec2 {
                x: b.min_x + 0.32 * span_x,
                y: b.min_y + 0.36 * span_y,
            },
            false,
            false,
            true,
            1u8,
        ),
        (
            "CPOD-Prime-1",
            Vec2 {
                x: b.min_x + 0.54 * span_x,
                y: b.min_y + 0.48 * span_y,
            },
            false,
            false,
            false,
            3u8,
        ),
    ];
    for (code, pos, ih, iw, is_sh, pr) in synthetic_lifelines {
        let (kw, rig_h) = size_usace_249th_prime_power_pack(ih, iw, is_sh, pre_assessed);
        lifeline_targets.push((code.to_string(), pos, kw, rig_h, pr));
    }

    if cfg.policy != FemaRecoveryExecutionPolicy::SequentialAdHoc {
        // Prioritize Tier-0 Hospitals -> Tier-1 Water/WWTP/Shelters -> Tier-2/3 Fire/PODs
        lifeline_targets.sort_by(|a, b_t| {
            a.4.cmp(&b_t.4).then_with(|| {
                a.1.dist(gsb_pos)
                    .partial_cmp(&b_t.1.dist(gsb_pos))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
    }

    // 3. Site FEMA-325 Temporary Debris Management Sites (`TDMS`) around industrial/perimeter aprons
    let n_tdms = if cfg.policy == FemaRecoveryExecutionPolicy::SequentialAdHoc {
        1usize
    } else {
        cfg.active_tdms_sites.clamp(1, 4)
    };
    let tdms_anchors = [
        Vec2 {
            x: b.min_x + 0.90 * span_x,
            y: b.min_y + 0.16 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.14 * span_x,
            y: b.min_y + 0.88 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.88 * span_x,
            y: b.min_y + 0.86 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.16 * span_x,
            y: b.min_y + 0.16 * span_y,
        },
    ];
    let tdms_positions: Vec<Vec2> = tdms_anchors.iter().copied().take(n_tdms).collect();

    // 4. Site Fixed Multi-Agency `DRC`s (`FEMA IA + SBA`) and Roving `MRIC` Vans
    let n_drc = cfg.fixed_drc_centers.clamp(1, 5);
    let n_mric = if cfg.policy == FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite {
        cfg.mobile_mric_units.min(6)
    } else if cfg.policy == FemaRecoveryExecutionPolicy::LifelinePowerFirst {
        (cfg.mobile_mric_units / 2).min(3)
    } else {
        0
    };
    let drc_anchors = [
        Vec2 {
            x: b.min_x + 0.50 * span_x,
            y: b.min_y + 0.50 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.74 * span_x,
            y: b.min_y + 0.68 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.26 * span_x,
            y: b.min_y + 0.30 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.28 * span_x,
            y: b.min_y + 0.74 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.76 * span_x,
            y: b.min_y + 0.28 * span_y,
        },
    ];
    let mric_anchors = [
        Vec2 {
            x: b.min_x + 0.20 * span_x,
            y: b.min_y + 0.52 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.62 * span_x,
            y: b.min_y + 0.24 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.82 * span_x,
            y: b.min_y + 0.50 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.42 * span_x,
            y: b.min_y + 0.82 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.36 * span_x,
            y: b.min_y + 0.18 * span_y,
        },
        Vec2 {
            x: b.min_x + 0.64 * span_x,
            y: b.min_y + 0.84 * span_y,
        },
    ];
    let drc_positions: Vec<Vec2> = drc_anchors.iter().copied().take(n_drc).collect();
    let mric_positions: Vec<Vec2> = mric_anchors.iter().copied().take(n_mric).collect();

    // 5. Compute Total Debris Streams (Structural Rubble + Vegetative Blowdown) & Affected IA Households
    let structural_rubble_tons = usar_rep.total_street_debris_tons.max(420.0) * 1.65;
    let canopy_veg_tons: f64 = parcels
        .iter()
        .map(|p| p.area_ha * p.zone.spec().canopy_fraction * 46.0)
        .sum::<f64>()
        .max(180.0);
    let total_debris_tons = structural_rubble_tons + canopy_veg_tons;
    let veg_tons = total_debris_tons * 0.32;
    let cd_tons = total_debris_tons * 0.54;
    let hhw_tons = total_debris_tons * 0.05;
    let white_goods_tons = total_debris_tons * 0.09;
    let white_goods_units = ((white_goods_tons * 2000.0) / 185.0).round() as usize;

    let total_residents: f64 = parcels.iter().map(Parcel::residents).sum::<f64>().max(500.0);
    let total_households = (total_residents / 2.45).max(200.0);
    let severity_frac = ((cfg.fema_usar.fema_pods.fema_comms.warning.transit_evac.shelters.evac.medevac.surge.ffe.seismic.moment_magnitude_mw - 6.2) / 2.6)
        .clamp(0.18, 0.78);
    let ia_eligible_households = (total_households * severity_frac).max(90.0);

    // 6. Parallel Monte Carlo Trials across Stochastic Debris Delays, PRT Rigging & Survivor Intake Waves
    let trials = cfg
        .fema_usar
        .fema_pods
        .fema_comms
        .warning
        .transit_evac
        .shelters
        .evac
        .medevac
        .surge
        .ffe
        .seismic
        .monte_carlo_trials
        .clamp(16, 2_500);
    let base_seed = cfg
        .fema_usar
        .fema_pods
        .fema_comms
        .warning
        .transit_evac
        .shelters
        .evac
        .medevac
        .surge
        .ffe
        .seismic
        .random_seed
        ^ 0xFE4A_2490_D8C5_7D45;
    let horizon_days = cfg.recovery_horizon_days.clamp(3, 30);
    let prt_teams = cfg.usace_249th_prt_teams.max(1);
    let gen_inv = cfg.gsb_generator_packs_inventory;
    let n_targets = lifeline_targets.len().max(1);
    let reopen_penalty_hr = (usar_rep.time_to_reopen_all_pods_hrs * 0.35).clamp(0.4, 18.0);

    #[derive(Clone)]
    struct TrialRes {
        target_energized_hr: Vec<f64>,
        energized_72h_pct: f64,
        mean_power_hr: f64,
        hauled_14d_pct: f64,
        vol_saved_yd3: f64,
        ia_reg_14d_pct: f64,
        mean_wait_min: f64,
        p90_access_wait_min: f64,
        daily_power_pct: Vec<f64>,
        daily_haul_pct: Vec<f64>,
        daily_vol_saved: Vec<f64>,
        daily_ia_pct: Vec<f64>,
        daily_wait_min: Vec<f64>,
    }

    let trial_results: Vec<TrialRes> = (0..trials)
        .into_par_iter()
        .map(|tr| {
            let s0 = splitmix64(base_seed.wrapping_add((tr as u64).wrapping_mul(0x9E37_79B9)));
            let jitter_road = 0.84 + 0.36 * u01_from_u64(s0);
            let jitter_demand = 0.88 + 0.28 * u01_from_u64(splitmix64(s0 ^ 0x11));

            // A. USACE 249th PRT + GSB Prime Power Schedule across `prt_teams` parallel rig crews
            let mut team_free_hr = vec![1.2 * jitter_road; prt_teams];
            let mut target_energized_hr = vec![999.0; n_targets];
            for (idx, (_, t_pos, _, rig_h, _)) in lifeline_targets.iter().enumerate() {
                if idx >= gen_inv {
                    // Must wait for mutual-aid FEMA National Cross-Dock generator resupply (+54h)
                    target_energized_hr[idx] = 56.0 + (idx - gen_inv) as f64 * 8.5;
                    continue;
                }
                let best_team = team_free_hr
                    .iter()
                    .enumerate()
                    .min_by(|a, b_c| a.1.partial_cmp(b_c.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let dist_km = (gsb_pos.dist(*t_pos) * 1.32 / 1000.0).max(0.8);
                let convoy_hr = (dist_km / 26.0) * jitter_road
                    + if cfg.policy == FemaRecoveryExecutionPolicy::SequentialAdHoc {
                        reopen_penalty_hr * 0.55
                    } else {
                        reopen_penalty_hr * 0.22
                    };
                let finish_hr = team_free_hr[best_team] + convoy_hr + rig_h * jitter_road;
                team_free_hr[best_team] = finish_hr + 0.45;
                target_energized_hr[idx] = finish_hr;
            }
            let energized_72h = target_energized_hr
                .iter()
                .filter(|&&h| h <= 72.0)
                .count() as f64
                * 100.0
                / (n_targets as f64);
            let mean_power_hr =
                target_energized_hr.iter().map(|h| h.min(168.0)).sum::<f64>() / (n_targets as f64);

            // B. FEMA-325 TDMS Haul-Off Cycle Times (`30 yd^3` = `9.5 ton` payload per dump truck)
            let mean_haul_km = if cfg.policy == FemaRecoveryExecutionPolicy::SequentialAdHoc {
                // Single distant regional subtitle-D landfill without local TDMS staging!
                28.5
            } else {
                (4.6 / (n_tdms as f64).sqrt()).clamp(2.1, 6.5)
            };
            let cycle_min = 14.0
                + (2.0 * mean_haul_km / 34.0) * 60.0 * jitter_road
                + if pre_assessed { 3.5 } else { 7.5 }; // ADMS automated e-ticket vs paper tower check
            let trips_per_truck_day = (10.5 * 60.0 / cycle_min).max(2.5);
            let daily_haul_cap_tons = (cfg.debris_haul_dump_trucks.max(1) as f64)
                * trips_per_truck_day
                * 9.5;

            // Vegetative 75% volume reduction (`4:1` air-curtain/mulching) + 38% C&D aggregate recycling
            //(`1 ton veg ~ 4.0 yd^3` loose; `1 ton C&D ~ 2.0 yd^3`)
            let veg_vol_reduction_frac = if cfg.policy == FemaRecoveryExecutionPolicy::SequentialAdHoc
            {
                0.0
            } else {
                0.75
            };
            let cd_recycle_frac = match cfg.policy {
                FemaRecoveryExecutionPolicy::SequentialAdHoc => 0.05,
                FemaRecoveryExecutionPolicy::LifelinePowerFirst => 0.28,
                FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite => 0.42,
            };

            // C. FEMA `DRC` (`12h/day`, `26 min` service) + `MRIC` (`9h/day`, `11 min` rapid intake)
            let total_drc_c = n_drc * cfg.specialists_per_drc.max(2);
            let total_mric_c = n_mric * 4;
            let ipaws_awareness_boost = match cfg.policy {
                FemaRecoveryExecutionPolicy::SequentialAdHoc => 0.72,
                FemaRecoveryExecutionPolicy::LifelinePowerFirst => 0.86,
                FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite => 0.97,
            };

            let mut cumul_hauled_tons = 0.0;
            let mut cumul_registered_hh = 0.0;
            let mut daily_power_pct = Vec::with_capacity(horizon_days);
            let mut daily_haul_pct = Vec::with_capacity(horizon_days);
            let mut daily_vol_saved = Vec::with_capacity(horizon_days);
            let mut daily_ia_pct = Vec::with_capacity(horizon_days);
            let mut daily_wait_min = Vec::with_capacity(horizon_days);
            let mut wait_sum = 0.0;

            for day in 1..=horizon_days {
                let end_hr = (day * 24) as f64;
                let p_pct = target_energized_hr
                    .iter()
                    .filter(|&&h| h <= end_hr)
                    .count() as f64
                    * 100.0
                    / (n_targets as f64);
                daily_power_pct.push(p_pct);

                // Phase II haul ramps up once Phase I US&R push opens corridors over Days 1..3
                let corridor_open_fac = ((day as f64) / 2.6).clamp(0.38, 1.0);
                cumul_hauled_tons =
                    (cumul_hauled_tons + daily_haul_cap_tons * corridor_open_fac).min(total_debris_tons);
                let haul_pct = (cumul_hauled_tons / total_debris_tons * 100.0).clamp(0.0, 100.0);
                daily_haul_pct.push(haul_pct);

                let hauled_frac = cumul_hauled_tons / total_debris_tons;
                let saved_yd3 = hauled_frac
                    * (veg_tons * 4.0 * veg_vol_reduction_frac + cd_tons * 2.0 * cd_recycle_frac);
                daily_vol_saved.push(saved_yd3);

                // Daily survivor registration wave peaks Days 2..6 then tapers as remaining pool shrinks
                let rem_hh = (ia_eligible_households * ipaws_awareness_boost - cumul_registered_hh)
                    .max(0.0);
                let wave_rate = if day == 1 {
                    0.11
                } else if day <= 5 {
                    0.23
                } else {
                    0.16
                };
                let arriving_today = (rem_hh * wave_rate * jitter_demand).max(1.5);
                let mric_share = if total_mric_c > 0 {
                    ((total_mric_c as f64 * 1.9) / (total_drc_c as f64 + total_mric_c as f64 * 1.9))
                        .clamp(0.15, 0.52)
                } else {
                    0.0
                };
                let drc_lambda_hr = (arriving_today * (1.0 - mric_share)) / 12.0;
                let (w_drc, _) = evaluate_drc_mric_mgc_wait_min(
                    drc_lambda_hr / (n_drc as f64),
                    cfg.specialists_per_drc.max(2),
                    26.0,
                    0.55,
                );
                let w_blend = if n_mric > 0 {
                    let mric_lambda_hr = (arriving_today * mric_share) / 9.0;
                    let (w_mric, _) = evaluate_drc_mric_mgc_wait_min(
                        mric_lambda_hr / (n_mric as f64),
                        4,
                        11.0,
                        0.38,
                    );
                    (1.0 - mric_share) * w_drc + mric_share * w_mric
                } else {
                    w_drc
                };
                // Excessive >95 min queues cause balking/deferred return next day
                let balk_keep = (1.0 - ((w_blend - 65.0).max(0.0) / 260.0)).clamp(0.48, 1.0);
                let daily_cap_hh = (total_drc_c as f64) * (12.0 * 60.0 / 26.0)
                    + (total_mric_c as f64) * (9.0 * 60.0 / 11.0);
                let served_today = (arriving_today * balk_keep).min(daily_cap_hh);
                cumul_registered_hh =
                    (cumul_registered_hh + served_today).min(ia_eligible_households);
                daily_ia_pct.push((cumul_registered_hh / ia_eligible_households) * 100.0);
                daily_wait_min.push(w_blend);
                wait_sum += w_blend;
            }

            let mean_w = wait_sum / (horizon_days as f64);
            let travel_p90 = if n_mric > 0 { 14.5 } else { 28.0 };
            TrialRes {
                target_energized_hr,
                energized_72h_pct: energized_72h,
                mean_power_hr,
                hauled_14d_pct: *daily_haul_pct.last().unwrap_or(&0.0),
                vol_saved_yd3: *daily_vol_saved.last().unwrap_or(&0.0),
                ia_reg_14d_pct: *daily_ia_pct.last().unwrap_or(&0.0),
                mean_wait_min: mean_w,
                p90_access_wait_min: mean_w * 1.32 + travel_p90,
                daily_power_pct,
                daily_haul_pct,
                daily_vol_saved,
                daily_ia_pct,
                daily_wait_min,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let energized_lifelines_72h_pct = trial_results
        .iter()
        .map(|r| r.energized_72h_pct)
        .sum::<f64>()
        * inv_t;
    let mean_time_to_emergency_power_hrs =
        trial_results.iter().map(|r| r.mean_power_hr).sum::<f64>() * inv_t;
    let curbside_haul_completed_14d_pct =
        trial_results.iter().map(|r| r.hauled_14d_pct).sum::<f64>() * inv_t;
    let tdms_landfill_volume_saved_yd3 =
        trial_results.iter().map(|r| r.vol_saved_yd3).sum::<f64>() * inv_t;
    let ia_registered_14d_pct =
        trial_results.iter().map(|r| r.ia_reg_14d_pct).sum::<f64>() * inv_t;
    let mean_survivor_drc_wait_min =
        trial_results.iter().map(|r| r.mean_wait_min).sum::<f64>() * inv_t;
    let p90_survivor_access_plus_wait_min = trial_results
        .iter()
        .map(|r| r.p90_access_wait_min)
        .sum::<f64>()
        * inv_t;

    let hhw_contamination_prevented_pct = match cfg.policy {
        FemaRecoveryExecutionPolicy::SequentialAdHoc => 18.0,
        FemaRecoveryExecutionPolicy::LifelinePowerFirst => 78.0,
        FemaRecoveryExecutionPolicy::IntegratedFemaEsf3AndIaSuite => 96.5,
    };

    // Average combined FEMA IA Housing/ONA grant (~$8,400) + prorated SBA low-interest physical loan (~$14,200) per registered household
    let projected_ia_and_sba_disbursement_musd =
        (ia_eligible_households * (ia_registered_14d_pct / 100.0) * 22_600.0) / 1_000_000.0;

    // 7. Build Daily Recovery Trajectory
    let mut daily_timeline = Vec::with_capacity(horizon_days);
    for d_i in 0..horizon_days {
        let p_pct = trial_results
            .iter()
            .map(|r| r.daily_power_pct[d_i])
            .sum::<f64>()
            * inv_t;
        let h_pct = trial_results
            .iter()
            .map(|r| r.daily_haul_pct[d_i])
            .sum::<f64>()
            * inv_t;
        let v_sav = trial_results
            .iter()
            .map(|r| r.daily_vol_saved[d_i])
            .sum::<f64>()
            * inv_t;
        let ia_pct = trial_results
            .iter()
            .map(|r| r.daily_ia_pct[d_i])
            .sum::<f64>()
            * inv_t;
        let w_min = trial_results
            .iter()
            .map(|r| r.daily_wait_min[d_i])
            .sum::<f64>()
            * inv_t;
        let disb = (ia_eligible_households * (ia_pct / 100.0) * 22_600.0) / 1_000_000.0;
        daily_timeline.push(FemaRecoveryDailyPoint {
            day: d_i + 1,
            critical_lifelines_energized_pct: p_pct,
            curbside_debris_hauled_pct: h_pct,
            tdms_volume_saved_yd3: v_sav,
            ia_households_registered_pct: ia_pct,
            mean_drc_mric_wait_min: w_min,
            disbursed_ia_sba_musd: disb,
        });
    }

    // 8. Build Facility Telemetry Table & Spatial Vectors (`GSB -> Lifelines`, `Parcels -> TDMS`, `Survivors -> DRC/MRIC`)
    let mut facility_rows = Vec::new();
    let mut gsb_dispatch_arcs = Vec::new();
    let mut deployed_kw_sum = 0.0;

    facility_rows.push(FemaRecoveryFacilityRow {
        site_id: 0,
        site_code: "FEMA-GSB-Region10".to_string(),
        kind: FemaRecoverySiteKind::GeneratorStagingBase,
        world_pos: gsb_pos,
        generator_kw: (gen_inv as f64) * 350.0,
        energized_hour: 0.5,
        daily_capacity_or_tons: gen_inv as f64,
        mean_wait_min: 0.0,
        utilization_pct: ((n_targets.min(gen_inv) as f64 / gen_inv.max(1) as f64) * 100.0)
            .min(100.0),
        transit_walk_equity_pct: 100.0,
    });

    for (idx, (code, pos, kw, _, _)) in lifeline_targets.iter().enumerate() {
        let mean_en_hr = trial_results
            .iter()
            .map(|r| r.target_energized_hr[idx])
            .sum::<f64>()
            * inv_t;
        if idx < gen_inv {
            deployed_kw_sum += *kw;
            gsb_dispatch_arcs.push((gsb_pos, *pos));
        }
        facility_rows.push(FemaRecoveryFacilityRow {
            site_id: facility_rows.len(),
            site_code: code.clone(),
            kind: FemaRecoverySiteKind::CriticalLifelineGenTarget,
            world_pos: *pos,
            generator_kw: *kw,
            energized_hour: mean_en_hr,
            daily_capacity_or_tons: *kw * 24.0,
            mean_wait_min: 0.0,
            utilization_pct: if mean_en_hr <= 72.0 { 94.0 } else { 55.0 },
            transit_walk_equity_pct: 100.0,
        });
    }

    for (idx, t_pos) in tdms_positions.iter().enumerate() {
        facility_rows.push(FemaRecoveryFacilityRow {
            site_id: facility_rows.len(),
            site_code: format!("TDMS-SortYard-{}", idx + 1),
            kind: FemaRecoverySiteKind::TemporaryDebrisSiteTdms,
            world_pos: *t_pos,
            generator_kw: 60.0,
            energized_hour: 3.5,
            daily_capacity_or_tons: (total_debris_tons * (curbside_haul_completed_14d_pct / 100.0))
                / (n_tdms as f64),
            mean_wait_min: if pre_assessed { 3.5 } else { 7.5 },
            utilization_pct: 86.0,
            transit_walk_equity_pct: hhw_contamination_prevented_pct,
        });
    }

    let mut intake_sites: Vec<(Vec2, bool)> = Vec::new();
    for (idx, d_pos) in drc_positions.iter().enumerate() {
        intake_sites.push((*d_pos, false));
        facility_rows.push(FemaRecoveryFacilityRow {
            site_id: facility_rows.len(),
            site_code: format!("FEMA-SBA-DRC-{}", idx + 1),
            kind: FemaRecoverySiteKind::FixedDisasterRecoveryCenterDrc,
            world_pos: *d_pos,
            generator_kw: 125.0,
            energized_hour: 4.2,
            daily_capacity_or_tons: (cfg.specialists_per_drc.max(2) as f64) * (12.0 * 60.0 / 26.0),
            mean_wait_min: mean_survivor_drc_wait_min * 1.08,
            utilization_pct: 82.0,
            transit_walk_equity_pct: 88.0,
        });
    }
    for (idx, m_pos) in mric_positions.iter().enumerate() {
        intake_sites.push((*m_pos, true));
        facility_rows.push(FemaRecoveryFacilityRow {
            site_id: facility_rows.len(),
            site_code: format!("FEMA-MRIC-Van-{}", idx + 1),
            kind: FemaRecoverySiteKind::MobileRegistrationIntakeMric,
            world_pos: *m_pos,
            generator_kw: 35.0,
            energized_hour: 2.0,
            daily_capacity_or_tons: 4.0 * (9.0 * 60.0 / 11.0),
            mean_wait_min: (mean_survivor_drc_wait_min * 0.62).max(3.5),
            utilization_pct: 79.0,
            transit_walk_equity_pct: 96.0,
        });
    }

    // 9. Per-Parcel Recovery Composite Score, Walk/GTFS Intake Equity & TDMS Haul Vectors
    let mut parcel_recovery_score_pct = Vec::with_capacity(parcels.len());
    let mut parcel_drc_wait_min = Vec::with_capacity(parcels.len());
    let mut tdms_haul_arcs = Vec::new();
    let mut walkable_res_sum = 0.0;
    let mut total_res_check = 0.0;

    for (p_idx, p) in parcels.iter().enumerate() {
        let nearest_intake_m = intake_sites
            .iter()
            .map(|(pos, _)| p.center.dist(*pos))
            .fold(f64::INFINITY, f64::min);
        // Walkable <= 800m OR 15-min GTFS bus feeder access (`<= 1,450m` corridor)
        let transit_access_min = (nearest_intake_m / 82.0).min(nearest_intake_m / 240.0 + 5.5);
        let res = p.residents();
        total_res_check += res;
        if transit_access_min <= 20.0 {
            walkable_res_sum += res;
        }

        let nearest_tdms = tdms_positions
            .iter()
            .copied()
            .min_by(|a, b_t| {
                p.center
                    .dist(*a)
                    .partial_cmp(&p.center.dist(*b_t))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(gsb_pos);
        if p_idx % 5 == 0 && (p.zone == ZoneCode::CBD || p.zone == ZoneCode::MU4 || p.debris_blockage_tons > 12.0) {
            tdms_haul_arcs.push((p.center, nearest_tdms));
        }

        let p_wait = mean_survivor_drc_wait_min + transit_access_min * 0.65;
        let score = (0.36 * energized_lifelines_72h_pct
            + 0.32 * curbside_haul_completed_14d_pct
            + 0.32 * ia_registered_14d_pct
            - (p_wait - 18.0).max(0.0) * 0.35)
            .clamp(5.0, 100.0);
        parcel_recovery_score_pct.push(score);
        parcel_drc_wait_min.push(p_wait);
    }

    let transit_walkable_intake_equity_pct = if total_res_check > 1e-6 {
        (walkable_res_sum / total_res_check * 100.0).clamp(0.0, 100.0)
    } else {
        100.0
    };

    FemaGsbDrcTdmsReport {
        config: cfg.clone(),
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        gsb_world_pos: gsb_pos,
        total_critical_lifeline_targets: n_targets,
        energized_lifelines_72h_pct,
        mean_time_to_emergency_power_hrs,
        deployed_prime_power_mw: deployed_kw_sum / 1000.0,
        total_disaster_debris_tons: total_debris_tons,
        vegetative_debris_tons: veg_tons,
        cd_structural_debris_tons: cd_tons,
        hhw_hazardous_swept_tons: hhw_tons,
        white_goods_freon_units,
        curbside_haul_completed_14d_pct,
        tdms_landfill_volume_saved_yd3,
        hhw_contamination_prevented_pct,
        affected_ia_eligible_households: ia_eligible_households,
        ia_registered_14d_pct,
        mean_survivor_drc_wait_min,
        p90_survivor_access_plus_wait_min,
        transit_walkable_intake_equity_pct,
        projected_ia_and_sba_disbursement_musd,
        gsb_dispatch_arcs,
        tdms_haul_arcs,
        facility_rows,
        daily_timeline,
        parcel_recovery_score_pct,
        parcel_drc_wait_min,
    }
}

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

use crate::network::{BoundingBox, RoadNetwork, Vec2};
use serde::{Deserialize, Serialize};

/// Municipal land-use zoning categories with Floor-Area Ratio (FAR), demographic/job density,
/// surface imperviousness, tree canopy, and annual tax revenue yield.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZoneCode {
    R1,
    R3,
    MU4,
    CBD,
    IND,
    PARK,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZoneSpec {
    pub code: ZoneCode,
    pub label: &'static str,
    pub far: f64,
    pub residents_per_ha: f64,
    pub jobs_per_ha: f64,
    pub impervious_fraction: f64,
    pub canopy_fraction: f64,
    pub tax_yield_k_per_acre: f64,
}

impl ZoneCode {
    pub fn spec(self) -> ZoneSpec {
        match self {
            Self::R1 => ZoneSpec {
                code: self,
                label: "R1 Single-Family",
                far: 0.5,
                residents_per_ha: 28.0,
                jobs_per_ha: 4.0,
                impervious_fraction: 0.42,
                canopy_fraction: 0.35,
                tax_yield_k_per_acre: 6.5,
            },
            Self::R3 => ZoneSpec {
                code: self,
                label: "R3 Missing-Middle",
                far: 1.6,
                residents_per_ha: 95.0,
                jobs_per_ha: 18.0,
                impervious_fraction: 0.58,
                canopy_fraction: 0.26,
                tax_yield_k_per_acre: 16.0,
            },
            Self::MU4 => ZoneSpec {
                code: self,
                label: "MU-4 Mixed-Use TOD",
                far: 3.5,
                residents_per_ha: 185.0,
                jobs_per_ha: 145.0,
                impervious_fraction: 0.74,
                canopy_fraction: 0.20,
                tax_yield_k_per_acre: 34.0,
            },
            Self::CBD => ZoneSpec {
                code: self,
                label: "C-CBD Civic/Core",
                far: 6.0,
                residents_per_ha: 140.0,
                jobs_per_ha: 360.0,
                impervious_fraction: 0.86,
                canopy_fraction: 0.12,
                tax_yield_k_per_acre: 54.0,
            },
            Self::IND => ZoneSpec {
                code: self,
                label: "I-Tech Research",
                far: 2.2,
                residents_per_ha: 10.0,
                jobs_per_ha: 190.0,
                impervious_fraction: 0.78,
                canopy_fraction: 0.14,
                tax_yield_k_per_acre: 26.0,
            },
            Self::PARK => ZoneSpec {
                code: self,
                label: "G-Park Bioswale Canopy",
                far: 0.0,
                residents_per_ha: 0.0,
                jobs_per_ha: 3.0,
                impervious_fraction: 0.08,
                canopy_fraction: 0.82,
                tax_yield_k_per_acre: 1.0,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parcel {
    pub id: usize,
    pub min: Vec2,
    pub max: Vec2,
    pub center: Vec2,
    pub area_ha: f64,
    pub zone: ZoneCode,
    pub isochrone_minutes: f64,
    #[serde(default)]
    pub fire_travel_s: f64,
    #[serde(default)]
    pub hospital_travel_s: f64,
    /// Assigned primary first-due hypercube response unit (`0..N`) from Carter-Chaiken-Ignall /
    /// Larson redistricting (`None` = dynamic closest-unit Voronoi territory).
    #[serde(default)]
    pub response_district_unit: Option<u16>,
    /// Home road `segment_id` of the assigned primary response unit's firehouse.
    #[serde(default)]
    pub response_district_station_seg: Option<u32>,
    /// Separate **Peak Rush** CAD dispatch table primary unit (`0..N`) & station segment.
    #[serde(default)]
    pub peak_district_unit: Option<u16>,
    #[serde(default)]
    pub peak_district_station_seg: Option<u32>,
    /// Separate **Off-Peak Free-Flow** CAD dispatch table primary unit (`0..N`) & station segment.
    #[serde(default)]
    pub offpeak_district_unit: Option<u16>,
    #[serde(default)]
    pub offpeak_district_station_seg: Option<u32>,
    /// Cached post-earthquake Fire-Following-Earthquake (`FFE`) burn probability (`0..100%`) &
    /// dry-hydrant `< 20 psi` probability (`0..100%`) for GPU `FireFollowingQuake` visualization.
    #[serde(default)]
    pub ffe_burn_prob_pct: f32,
    #[serde(default)]
    pub ffe_dry_hydrant_prob_pct: f32,
    /// Cached post-earthquake Red/Burn casualty time-to-definitive-treatment (`minutes`) &
    /// dynamic ambulance diversion probability (`0..100%`) for GPU `HospitalTraumaSurge` rendering.
    #[serde(default)]
    pub surge_red_burn_treat_min: f32,
    #[serde(default)]
    pub surge_diversion_prob_pct: f32,
    /// Cached Koschmieder FFE smoke visibility (`statute miles`) & primary rotary-wing airlift
    /// time savings (`minutes`) for GPU `AeromedicalMedevac` visualization.
    #[serde(default)]
    pub medevac_smoke_vis_sm: f32,
    #[serde(default)]
    pub medevac_airlift_save_min: f32,
    /// Cached staged neighborhood evacuation tier (`1..=3`) & 90% parcel clearance time (`minutes`)
    /// for GPU `EvacuationContraflow` visualization.
    #[serde(default)]
    pub evac_stage_tier: u8,
    #[serde(default)]
    pub evac_clearance_min: f32,
    /// Cached post-earthquake clean-air shelter coverage (`0..=100%`) & assigned indoor `PM2.5`
    /// (`ug/m^3`) for GPU `EmergencyShelters` visualization.
    #[serde(default)]
    pub shelter_clean_air_pct: f32,
    #[serde(default)]
    pub shelter_indoor_pm25: f32,
    /// Cached multi-day commodity resupply security (`0..=100%`) & normalized deprivation index
    /// (`0..=1`) for GPU `CommodityResupply` visualization.
    #[serde(default)]
    pub resupply_security_pct: f32,
    #[serde(default)]
    pub resupply_deprivation_idx: f32,
    /// Cached carless + ADA transit evacuation clean-air clearance (`0..=100%`) & total walk+wait
    /// time (`minutes`) for GPU `TransitBusEvacuation` heatmap rendering.
    #[serde(default)]
    pub transit_evac_cleared_pct: f32,
    #[serde(default)]
    pub transit_evac_wait_min: f32,
    /// Cached 60-min household warning+mobilization completion (`0..=100%`) & total departure
    /// delay (`minutes`) for GPU `WarningMobilization` heatmap rendering.
    #[serde(default)]
    pub warning_mobilized_60m_pct: f32,
    #[serde(default)]
    pub warning_departure_delay_min: f32,
    /// Cached FEMA `IPAWS` (`WEA 3.0 + EAS PEP + NWR SAME + MERS`) verified household reach (`0..=100%`)
    /// and `ESF #2` `NIFOG` tactical `LMR` responder interoperability (`0..=100%`) for GPU visualization.
    #[serde(default)]
    pub fema_ipaws_reach_pct: f32,
    #[serde(default)]
    pub fema_esf2_lmr_pct: f32,
    /// Cached FEMA `C-POD` drive+queue access time (`minutes`) and household demand served (`0..=100%`)
    /// for GPU `FemaIsbPods` heatmap visualization.
    #[serde(default)]
    pub pod_total_access_min: f32,
    #[serde(default)]
    pub pod_served_pct: f32,
    /// Cached FEMA `Type 1 US&R` entrapped live-rescue survival (`0..=100%`) & parcel collapse
    /// rubble (`tons`) for GPU `FemaUsarDebris` heatmap visualization.
    #[serde(default)]
    pub usar_saved_pct: f32,
    #[serde(default)]
    pub debris_blockage_tons: f32,
    /// Cached FEMA `GSB + TDMS + DRC/MRIC` composite recovery readiness (`0..=100%`) & survivor
    /// `DRC/MRIC` access+wait time (`minutes`) for GPU `FemaGsbDrcTdms` visualization.
    #[serde(default)]
    pub fema_recovery_score_pct: f32,
    #[serde(default)]
    pub fema_drc_wait_min: f32,
    /// Cached `ALERTWest` / `UOregon OHAZ` WUI wildfire burn risk (`0..=100%`) & camera viewshed
    /// coverage (`0..=100%`) for GPU `WildfireAlertWestResponse` visualization.
    #[serde(default)]
    pub wildfire_burn_risk_pct: f32,
    #[serde(default)]
    pub wildfire_camera_vis_pct: f32,
    /// Cached post-wildfire hydrophobic storm runoff depth (`mm`) & USGS alluvial-fan debris-flow
    /// hazard (`0..=100%`) for GPU `BurnScarDebrisFlow` visualization.
    #[serde(default)]
    pub burn_scar_runoff_mm: f32,
    #[serde(default)]
    pub debris_flow_hazard_pct: f32,
    /// Cached post-wildfire `WTP` finished-water security (`0..=100%`) & minimum residual pressure
    /// (`psi`) during turbidity intake shutoffs for GPU `WtpTurbidityStorage` visualization.
    #[serde(default)]
    pub wtp_water_security_pct: f32,
    #[serde(default)]
    pub wtp_min_pressure_psi: f32,
}

impl Parcel {
    pub fn residents(&self) -> f64 {
        self.zone.spec().residents_per_ha * self.area_ha
    }

    pub fn jobs(&self) -> f64 {
        self.zone.spec().jobs_per_ha * self.area_ha
    }
}

pub fn generate_parcels_for_bounds(bounds: BoundingBox) -> Vec<Parcel> {
    let span_x = (bounds.max_x - bounds.min_x - 50.0).max(120.0);
    let span_y = (bounds.max_y - bounds.min_y - 50.0).max(120.0);
    let cols = ((span_x / 115.0).round() as usize).clamp(3, 12);
    let rows = ((span_y / 115.0).round() as usize).clamp(3, 12);
    let cw = span_x / cols as f64;
    let ch = span_y / rows as f64;
    let start_x = bounds.min_x + 25.0;
    let start_y = bounds.min_y + 25.0;
    let mid_x = 0.5 * (bounds.min_x + bounds.max_x);
    let mid_y = 0.5 * (bounds.min_y + bounds.max_y);
    let min_span = span_x.min(span_y);

    let mut parcels = Vec::with_capacity(rows * cols);
    let mut pid = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            let cx = start_x + (c as f64 + 0.5) * cw;
            let cy = start_y + (r as f64 + 0.5) * ch;
            let pad = 13.5;
            let pw = (cw - pad * 2.0).max(24.0);
            let ph = (ch - pad * 2.0).max(24.0);
            let dist = (cx - mid_x).hypot(cy - mid_y);

            let zone = if dist < min_span * 0.22 {
                ZoneCode::CBD
            } else if dist < min_span * 0.36 {
                ZoneCode::MU4
            } else if (r + c) % 5 == 0 {
                ZoneCode::PARK
            } else if c + 1 == cols && r == 0 {
                ZoneCode::IND
            } else {
                ZoneCode::R1
            };

            parcels.push(Parcel {
                id: pid,
                min: Vec2 {
                    x: cx - pw * 0.5,
                    y: cy - ph * 0.5,
                },
                max: Vec2 {
                    x: cx + pw * 0.5,
                    y: cy + ph * 0.5,
                },
                center: Vec2 { x: cx, y: cy },
                area_ha: (pw * ph) / 10_000.0,
                zone,
                isochrone_minutes: 6.0,
                fire_travel_s: 120.0,
                hospital_travel_s: 180.0,
                response_district_unit: None,
                response_district_station_seg: None,
                peak_district_unit: None,
                peak_district_station_seg: None,
                offpeak_district_unit: None,
                offpeak_district_station_seg: None,
                ffe_burn_prob_pct: 0.0,
                ffe_dry_hydrant_prob_pct: 0.0,
                surge_red_burn_treat_min: 0.0,
                surge_diversion_prob_pct: 0.0,
                medevac_smoke_vis_sm: 6.0,
                medevac_airlift_save_min: 0.0,
                evac_stage_tier: 2,
                evac_clearance_min: 35.0,
                shelter_clean_air_pct: 100.0,
                shelter_indoor_pm25: 6.0,
                resupply_security_pct: 100.0,
                resupply_deprivation_idx: 0.0,
                transit_evac_cleared_pct: 100.0,
                transit_evac_wait_min: 8.0,
                warning_mobilized_60m_pct: 95.0,
                warning_departure_delay_min: 12.0,
                fema_ipaws_reach_pct: 98.0,
                fema_esf2_lmr_pct: 98.0,
                pod_total_access_min: 10.0,
                pod_served_pct: 100.0,
                usar_saved_pct: 95.0,
                debris_blockage_tons: 0.0,
                fema_recovery_score_pct: 96.0,
                fema_drc_wait_min: 12.0,
                wildfire_burn_risk_pct: 0.0,
                wildfire_camera_vis_pct: 100.0,
                burn_scar_runoff_mm: 4.0,
                debris_flow_hazard_pct: 0.0,
                wtp_water_security_pct: 100.0,
                wtp_min_pressure_psi: 58.0,
            });
            pid += 1;
        }
    }
    parcels
}

pub fn apply_tod_upzoning_policy(parcels: &mut [Parcel], net: &mut RoadNetwork) {
    for (i, p) in parcels.iter_mut().enumerate() {
        p.zone = if i % 5 == 0 {
            ZoneCode::PARK
        } else if i % 3 == 0 {
            ZoneCode::CBD
        } else {
            ZoneCode::MU4
        };
    }
    for seg in &mut net.segments {
        if seg.lane_count >= 2 {
            seg.has_bus_lane = true;
        }
        seg.has_bike_lane = true;
    }
}

/// Solves Wilson (1967) doubly-constrained entropy-maximizing Gravity OD matrix
/// `T_ij = A_i * B_j * P_i * A_j * exp(-beta * c_ij)` via Iterative Proportional Fitting (Furness).
pub fn compute_gravity_od_matrix(parcels: &[Parcel], beta_per_min: f64) -> Vec<Vec<f64>> {
    let n = parcels.len();
    if n == 0 {
        return Vec::new();
    }
    // Hourly trip productions P_i (0.42 peak trips/resident) & attractions A_j (0.55 peak trips/job)
    let mut prod: Vec<f64> = parcels.iter().map(|p| p.residents() * 0.42 + 0.5).collect();
    let raw_attr: Vec<f64> = parcels.iter().map(|p| p.jobs() * 0.55 + 0.5).collect();

    let sum_p: f64 = prod.iter().sum();
    let sum_a: f64 = raw_attr.iter().sum();
    let scale = if sum_a > 1e-6 { sum_p / sum_a } else { 1.0 };
    let attr: Vec<f64> = raw_attr.into_iter().map(|a| a * scale).collect();
    if sum_p < 1e-6 {
        prod.fill(1.0);
    }

    // Generalized travel time c_ij in minutes (circuity 1.28 at 24 km/h urban average)
    let mut f_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let d_m = if i == j {
                65.0
            } else {
                parcels[i].center.dist(parcels[j].center) * 1.28
            };
            let c_min = d_m / 400.0 + 1.5;
            f_mat[i][j] = (-beta_per_min * c_min).exp();
        }
    }

    let mut a_bal = vec![1.0; n];
    let mut b_bal = vec![1.0; n];
    for _iter in 0..25 {
        let mut max_err: f64 = 0.0;
        for i in 0..n {
            let denom: f64 = (0..n).map(|j| b_bal[j] * attr[j] * f_mat[i][j]).sum();
            a_bal[i] = if denom > 1e-12 { 1.0 / denom } else { 0.0 };
        }
        for j in 0..n {
            let denom: f64 = (0..n).map(|i| a_bal[i] * prod[i] * f_mat[i][j]).sum();
            let next_b = if denom > 1e-12 { 1.0 / denom } else { 0.0 };
            max_err = max_err.max((next_b - b_bal[j]).abs());
            b_bal[j] = next_b;
        }
        if max_err < 1e-4 {
            break;
        }
    }

    let mut od = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            od[i][j] = a_bal[i] * b_bal[j] * prod[i] * attr[j] * f_mat[i][j];
        }
    }
    od
}

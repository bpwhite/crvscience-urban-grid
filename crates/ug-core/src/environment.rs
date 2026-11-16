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

use crate::multimodal::TransitPolicy;
use crate::network::{RoadNetwork, Vec2};
use crate::traffic::{Vehicle, VehicleClass};
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnvironmentalSnapshot {
    pub co2_kg_per_hr: f64,
    pub nox_g_per_hr: f64,
    pub pm25_g_per_hr: f64,
    pub corridor_noise_dba: f64,
    pub stormwater_runoff_lps: f64,
    pub heat_island_delta_c: f64,
    pub access_15min_pct: f64,
}

/// Computes Jimenez-Palacios (1999) Vehicle Specific Power (`VSP`, kW/tonne):
/// `VSP = v * (1.1 * a + 0.132) + 0.000302 * v^3`
pub fn vehicle_specific_power_kw_t(speed_mps: f64, accel_mps2: f64) -> f64 {
    speed_mps * (1.1 * accel_mps2.max(-2.0) + 0.132) + 0.000_302 * speed_mps.powi(3)
}

pub fn evaluate_environment_and_isochrones(
    net: &mut RoadNetwork,
    parcels: &mut [Parcel],
    vehicles: &[Vehicle],
    policy: &TransitPolicy,
    anchor_pos: Vec2,
) -> EnvironmentalSnapshot {
    for seg in &mut net.segments {
        seg.vsp_intensity *= 0.88;
    }

    let mut co2_gps = 0.0;
    let mut nox_mgps = 0.0;
    let mut pm25_mgps = 0.0;
    let mut sum_speed = 0.0;

    for v in vehicles {
        let vsp = vehicle_specific_power_kw_t(v.v, v.acc);
        let mass_mult = match v.v_class {
            VehicleClass::TransitBus => 2.6,
            VehicleClass::FreightTruck => 2.2,
            VehicleClass::EmergencyResponder => 1.85,
            VehicleClass::Sedan => 1.0,
        };
        let pos_vsp = vsp.max(0.0);
        co2_gps += (0.85 + pos_vsp * 0.24) * mass_mult;
        nox_mgps += (1.10 + pos_vsp * 0.65) * mass_mult;
        let brake_wear = if v.acc < -1.2 { 0.06 } else { 0.0 };
        pm25_mgps += (0.04 + pos_vsp * 0.022 + brake_wear) * mass_mult;
        sum_speed += v.v;

        if let Some(seg_id) = v.current_segment_id(&net.lanes) {
            net.segments[seg_id].vsp_intensity =
                (net.segments[seg_id].vsp_intensity + 0.045 * mass_mult).min(1.0);
        }
    }

    let mean_v = if vehicles.is_empty() {
        8.5
    } else {
        sum_speed / vehicles.len() as f64
    };

    let (mut area_sum, mut impv_sum, mut canopy_sum) = (0.0, 0.0, 0.0);
    for p in parcels.iter() {
        let sp = p.zone.spec();
        area_sum += p.area_ha;
        impv_sum += sp.impervious_fraction * p.area_ha;
        canopy_sum += sp.canopy_fraction * p.area_ha;
    }
    let avg_impv = if area_sum > 0.0 { impv_sum / area_sum } else { 0.6 };
    let avg_canopy = if area_sum > 0.0 { canopy_sum / area_sum } else { 0.25 };

    // Rational Method Q = C * i * A for 25 mm/h storm (69.44 L/s per impervious ha)
    let runoff_lps = avg_impv * 69.44 * area_sum;
    let heat_delta_c = (2.8 * avg_impv - 2.6 * avg_canopy).max(-0.4);
    let noise_dba = 48.0
        + 10.0 * (1.0 + vehicles.len() as f64 * 0.85 + mean_v * 0.9).log10()
        - 2.5 * avg_canopy;

    // Moreno et al. (2021) 15-Minute City multi-modal accessibility contours
    let total_km: f64 = net.segments.iter().map(|s| s.length / 1000.0).sum();
    let bike_km: f64 = net
        .segments
        .iter()
        .filter(|s| s.has_bike_lane)
        .map(|s| s.length / 1000.0)
        .sum();
    let bike_cov = (bike_km / (total_km * 0.65).max(0.25)).min(1.0);
    let eff_m_per_min = 80.0
        * (1.0 + 0.85 * bike_cov + 0.95 * (15.0 / policy.headway_min.max(4.0)) * 0.45);

    let mut within_15 = 0usize;
    for p in parcels.iter_mut() {
        let manhattan_m = ((p.center.x - anchor_pos.x).abs() + (p.center.y - anchor_pos.y).abs()) * 0.82;
        p.isochrone_minutes = manhattan_m / eff_m_per_min.max(65.0) + 1.2;
        if p.isochrone_minutes <= 15.0 {
            within_15 += 1;
        }
    }
    let access_pct = if parcels.is_empty() {
        0.0
    } else {
        (within_15 as f64 / parcels.len() as f64) * 100.0
    };

    EnvironmentalSnapshot {
        co2_kg_per_hr: co2_gps * 3.6,
        nox_g_per_hr: nox_mgps * 3.6,
        pm25_g_per_hr: pm25_mgps * 3.6,
        corridor_noise_dba: noise_dba,
        stormwater_runoff_lps: runoff_lps,
        heat_island_delta_c: heat_delta_c,
        access_15min_pct: access_pct,
    }
}

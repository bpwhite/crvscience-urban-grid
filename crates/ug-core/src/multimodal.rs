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

use crate::network::RoadNetwork;
use crate::zoning::{Parcel, ZoneCode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransitPolicy {
    pub headway_min: f64,
    pub fare_usd: f64,
    pub cordon_pricing_usd: f64,
    pub couple_gravity_demand: bool,
}

impl Default for TransitPolicy {
    fn default() -> Self {
        Self {
            headway_min: 12.0,
            fare_usd: 0.0, // Fare-free municipal bus baseline (e.g. Metro Transit System)
            cordon_pricing_usd: 2.0,
            couple_gravity_demand: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModeShares {
    pub auto: f64,
    pub transit: f64,
    pub bike: f64,
    pub walk: f64,
    /// Log-sum accessibility (McFadden inclusive value)
    pub logsum_utility: f64,
}

/// Evaluates McFadden (1974) Multinomial Logit (MNL) mode split across Auto, Bus/BRT, Bike, and Walk.
pub fn evaluate_mode_split(
    net: &RoadNetwork,
    parcels: &[Parcel],
    policy: &TransitPolicy,
    mean_auto_speed_mps: f64,
) -> ModeShares {
    let mut total_km = 0.0;
    let mut bus_km = 0.0;
    let mut bike_km = 0.0;
    for s in &net.segments {
        let km = s.length / 1000.0;
        total_km += km;
        if s.has_bus_lane {
            bus_km += km;
        }
        if s.has_bike_lane {
            bike_km += km;
        }
    }
    let bike_cov = (bike_km / (total_km * 0.65).max(0.25)).min(1.0);
    let bus_cov = (bus_km / (total_km * 0.35).max(0.25)).min(1.0);

    let mixed_blocks = parcels
        .iter()
        .filter(|p| matches!(p.zone, ZoneCode::MU4 | ZoneCode::CBD | ZoneCode::PARK))
        .count();
    let mixed_entropy = if parcels.is_empty() {
        0.3
    } else {
        mixed_blocks as f64 / parcels.len() as f64
    };

    let avg_trip_km = 3.2;
    let v_auto = mean_auto_speed_mps.max(2.0);
    let t_auto_min = (avg_trip_km * 1000.0 / v_auto) / 60.0 + 3.0;
    let v_bus = 6.2 + bus_cov * 3.8;
    let t_bus_min = (avg_trip_km * 1000.0 / v_bus) / 60.0 + policy.headway_min * 0.5 + 4.0;
    let t_bike_min = (avg_trip_km / 18.5) * 60.0;
    let t_walk_min = ((avg_trip_km * (1.0 - 0.45 * mixed_entropy)).max(0.9) / 4.8) * 60.0;

    let u_auto = 1.45 - 0.09 * t_auto_min - 0.18 * policy.cordon_pricing_usd;
    let u_transit = -0.25 - 0.085 * t_bus_min - 0.28 * policy.fare_usd + 1.05 * bus_cov;
    let u_bike = -0.95 - 0.09 * t_bike_min + 1.65 * bike_cov;
    let u_walk = -0.65 - 0.065 * t_walk_min + 1.45 * mixed_entropy;

    let ea = u_auto.exp();
    let et = u_transit.exp();
    let eb = u_bike.exp();
    let ew = u_walk.exp();
    let sum = ea + et + eb + ew;

    ModeShares {
        auto: ea / sum,
        transit: et / sum,
        bike: eb / sum,
        walk: ew / sum,
        logsum_utility: sum.ln(),
    }
}

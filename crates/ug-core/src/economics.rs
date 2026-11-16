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

use crate::environment::EnvironmentalSnapshot;
use crate::emergency::{EmergencyCoverageSummary, EvpTelemetry};
use crate::max_pressure::MaxPressureTelemetry;
use crate::transit_gtfs::TransitTelemetrySnapshot;
use crate::multimodal::{ModeShares, TransitPolicy};
use crate::network::RoadNetwork;
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MunicipalLedger {
    pub total_residents: usize,
    pub total_jobs: usize,
    pub jobs_housing_ratio: f64,
    pub gross_tax_revenue_k_yr: f64,
    pub infra_and_transit_cost_k_yr: f64,
    pub net_fiscal_per_acre_k_yr: f64,
    pub hcm_los: char,
}

pub fn evaluate_municipal_ledger(
    net: &RoadNetwork,
    parcels: &[Parcel],
    policy: &TransitPolicy,
    mean_speed_mph: f64,
) -> MunicipalLedger {
    let mut res = 0.0;
    let mut jobs = 0.0;
    let mut area_ha = 0.0;
    let mut gross_tax_k = 0.0;

    for p in parcels {
        let sp = p.zone.spec();
        res += p.residents();
        jobs += p.jobs();
        area_ha += p.area_ha;
        gross_tax_k += sp.tax_yield_k_per_acre * (p.area_ha * 2.471_05);
    }

    let mut lane_km = 0.0;
    let mut bike_km = 0.0;
    for s in &net.segments {
        let km = s.length / 1000.0;
        lane_km += km * s.lane_count as f64;
        if s.has_bike_lane {
            bike_km += km;
        }
    }

    let transit_net_cost_k = (60.0 / policy.headway_min.max(3.0))
        * 14.0
        * (1.0 - policy.fare_usd * 0.22).max(0.2);
    let infra_cost_k = lane_km * 28.0 + bike_km * 6.0 + transit_net_cost_k;
    let total_acres = (area_ha * 2.471_05).max(1.0);

    let los = if mean_speed_mph >= 21.0 {
        'A'
    } else if mean_speed_mph >= 17.0 {
        'B'
    } else if mean_speed_mph >= 13.0 {
        'C'
    } else if mean_speed_mph >= 9.5 {
        'D'
    } else if mean_speed_mph >= 6.5 {
        'E'
    } else {
        'F'
    };

    let households = (res / 2.3).max(1.0);
    MunicipalLedger {
        total_residents: res.round() as usize,
        total_jobs: jobs.round() as usize,
        jobs_housing_ratio: jobs / households,
        gross_tax_revenue_k_yr: gross_tax_k,
        infra_and_transit_cost_k_yr: infra_cost_k,
        net_fiscal_per_acre_k_yr: (gross_tax_k - infra_cost_k) / total_acres,
        hcm_los: los,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSummaryReport {
    pub label: String,
    pub mean_speed_mph: f64,
    pub avg_trip_time_s: f64,
    pub throughput_veh_per_hr: f64,
    pub mode_shares: ModeShares,
    pub environment: EnvironmentalSnapshot,
    pub ledger: MunicipalLedger,
    pub transit_telemetry: TransitTelemetrySnapshot,
    pub max_pressure_telemetry: MaxPressureTelemetry,
    pub evp_telemetry: EvpTelemetry,
    pub emergency_coverage: EmergencyCoverageSummary,
}

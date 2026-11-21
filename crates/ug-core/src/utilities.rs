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

//! Coupled Municipal Lifeline Grid Models:
//! 1. **Electrical Distribution Grid (`12.47 kV` Feeders)**: Linearized Baran & Wu (1989) DistFlow /
//!    nodal admittance solver, ANSI C84.1 Range A (`0.95..1.05 p.u.`) voltage regulation, rooftop
//!    solar PV & EV fast-charging loads, feeder ampacity `% MVA`, and substation outage cascades.
//! 2. **Potable Water Distribution Grid**: Todini & Pilati (1988) / Hazen-Williams pressurized
//!    looped pipe-network hydraulics (`h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.8704)`), electrically
//!    driven booster pumps, and USFA/AWWA `>= 20 psi` (`14.06 m H2O`) fire-hydrant residual pressure.
//! 3. **Wastewater & Sanitary Sewer Grid**: Manning's open-channel gravity hydraulics
//!    (`Q_full = (1/n) * A * R_h^(2/3) * S_0^(1/2)`), zoning-coupled Rainfall-Derived Infiltration &
//!    Inflow (`RDII`, US EPA SSOAP), electrically powered lift stations, WWTP hydraulic/BOD load,
//!    and Sanitary Sewer Overflow (`SSO`) surcharge detection.
//! 4. **Rinaldi, Peerenboom & Kelly (2001) Cross-Lifeline Interdependency Cascades**:
//!    Electric feeder/substation blackouts propagate to un-backed-up water booster pumps, sewage lift
//!    stations, and signalized traffic intersections unless islanded microgrid/backup power is active.

use crate::diurnal::USFA_DIURNAL_CALL_MULTIPLIERS;
use crate::emergency::{EmergencyFacility, EmergencyFacilityKind};
use crate::network::{ControlMode, RoadNetwork, Vec2};
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const UTILITY_NODE_CSV_HEADER: &str = "junction_id,pos_x,pos_y,elevation_m,energized,voltage_pu,ansi_c84_1_ok,electric_load_kw,water_head_m,water_pressure_psi,fire_flow_20psi_ok,sewer_inflow_lps,sewer_d_over_d,sewer_surcharged_sso,signal_darkened";

pub const PSI_PER_METER_H2O: f64 = 1.422_334;
pub const MIN_FIRE_FLOW_RESIDUAL_PSI: f64 = 20.0; // USFA / NFPA / AWWA minimum residual hydrant pressure

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UtilityHeatmapSublayer {
    /// Composite worst-of-three lifeline health score across Electric, Water & Wastewater (`0..1`).
    CompositeLifelineHealth,
    /// Electrical primary distribution voltage profile (`p.u.`, ANSI C84.1 `0.95..1.05` green).
    ElectricVoltagePu,
    /// Potable water residual pressure (`psi`, crimson `< 20 psi` NFPA fire-flow deficit).
    WaterResidualPressurePsi,
    /// Sanitary sewer pipe filling ratio `d/D` & wet-weather `SSO` surcharge hotspots.
    WastewaterSewerLoading,
}

impl Default for UtilityHeatmapSublayer {
    fn default() -> Self {
        Self::CompositeLifelineHealth
    }
}

impl UtilityHeatmapSublayer {
    pub fn label(self) -> &'static str {
        match self {
            Self::CompositeLifelineHealth => "Composite 3-Grid Lifeline Health",
            Self::ElectricVoltagePu => "Electrical Bus Voltage (ANSI C84.1 p.u.)",
            Self::WaterResidualPressurePsi => "Potable Water Pressure (psi / 20-psi Fire)",
            Self::WastewaterSewerLoading => "Wastewater Sewer d/D & SSO Surcharge",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityGridsConfig {
    pub heatmap_sublayer: UtilityHeatmapSublayer,
    /// Substation On-Load Tap Changer (`OLTC`) bus voltage setpoint (`p.u.`, typical `1.025`).
    pub substation_voltage_pu: f64,
    /// EV charging load penetration (`%` of residential/commercial vehicle fleet charging).
    pub ev_charging_penetration_pct: f64,
    /// Rooftop Solar PV distributed generation penetration (`%`, peaks near 12:00-13:00).
    pub solar_pv_penetration_pct: f64,
    /// Active fire-hydrant suppression demand (`GPM`, US gallons/min; `1,500 GPM = 94.635 L/s`).
    pub active_fire_flow_demand_gpm: f64,
    /// Hazen-Williams pipe roughness coefficient `C` (`130` new ductile iron, `100` aged cast iron).
    pub hazen_williams_c: f64,
    /// Wet-weather storm rainfall intensity (`mm/hr`) driving EPA `RDII` sewer inflow.
    pub storm_rainfall_mm_hr: f64,
    /// Manning's roughness `n` for gravity sanitary sewer mains (`0.013` concrete / PVC).
    pub manning_sewer_n: f64,
    /// Indices (`0..num_substations`) of tripped / faulted electrical substations (`N-1` / `N-2` outage).
    pub tripped_substation_indices: Vec<usize>,
    /// When true, closed street/bridge segments (`seg.closed == true`) also sever co-located
    /// bridge-hung feeders, water mains, and sewer interceptors!
    pub couple_closed_roads_to_utility_mains: bool,
    /// When true, battery microgrids & emergency diesel generators keep water booster pumps,
    /// sewage lift stations, and traffic signals online during primary electrical grid outages!
    pub critical_backup_generators_enabled: bool,
    /// Finished-water clearwell & elevated storage tank fill (`%`, `0..=100%`, default `100.0%`).
    /// Below `65%`, drawdown lowers effective tank/booster hydraulic grade line (`HGL`), causing
    /// upper pressure zones to sag below `20 psi` when tanks approach depletion during WTP shutoffs!
    #[serde(default = "default_finished_storage_fill_pct")]
    pub finished_water_storage_fill_pct: f64,
}

fn default_finished_storage_fill_pct() -> f64 {
    100.0
}

impl Default for UtilityGridsConfig {
    fn default() -> Self {
        Self {
            heatmap_sublayer: UtilityHeatmapSublayer::CompositeLifelineHealth,
            substation_voltage_pu: 1.025,
            ev_charging_penetration_pct: 18.0,
            solar_pv_penetration_pct: 22.0,
            active_fire_flow_demand_gpm: 1_500.0,
            hazen_williams_c: 125.0,
            storm_rainfall_mm_hr: 4.0,
            manning_sewer_n: 0.013,
            tripped_substation_indices: Vec::new(),
            couple_closed_roads_to_utility_mains: true,
            critical_backup_generators_enabled: false,
            finished_water_storage_fill_pct: 100.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UtilityAssetKind {
    ElectricSubstation,
    WaterTreatmentBooster,
    SewageLiftStation,
    WastewaterTreatmentPlant,
}

impl UtilityAssetKind {
    pub fn short_badge(self) -> &'static str {
        match self {
            Self::ElectricSubstation => "SUB",
            Self::WaterTreatmentBooster => "WTP",
            Self::SewageLiftStation => "LIFT",
            Self::WastewaterTreatmentPlant => "WWTP",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityFacilityNode {
    pub id: usize,
    pub kind: UtilityAssetKind,
    pub label: String,
    pub junction_id: usize,
    pub world_pos: Vec2,
    pub online: bool,
    pub on_backup_power: bool,
    pub load_or_output_kw: f64,
    pub flow_lps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityJunctionTelemetry {
    pub junction_id: usize,
    pub pos: Vec2,
    pub elevation_m: f64,
    // Electrical
    pub energized: bool,
    pub voltage_pu: f64,
    pub ansi_c84_1_compliant: bool,
    pub real_load_kw: f64,
    pub solar_gen_kw: f64,
    pub signal_darkened: bool,
    // Potable Water
    pub hydraulic_head_m: f64,
    pub residual_pressure_psi: f64,
    pub water_demand_lps: f64,
    pub meets_20psi_fire_flow: bool,
    // Wastewater
    pub sanitary_dwf_lps: f64,
    pub wet_weather_rdii_lps: f64,
    pub cumulative_sewer_flow_lps: f64,
    pub sewer_d_over_d: f64,
    pub sewer_surcharged_sso: bool,
    /// Combined `[0.0 .. 1.0]` health index (`1.0` = nominal across all 3 grids, `0.0` = outage/surcharge).
    pub composite_health: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricalGridSummary {
    pub num_substations_total: usize,
    pub num_substations_online: usize,
    pub energized_nodes_pct: f64,
    pub total_load_mw: f64,
    pub solar_pv_generation_mw: f64,
    pub pump_and_wwtp_load_mw: f64,
    pub net_substation_import_mw: f64,
    pub line_resistive_losses_kw: f64,
    pub loss_pct: f64,
    pub min_voltage_pu: f64,
    pub max_voltage_pu: f64,
    pub ansi_c84_1_compliance_pct: f64,
    pub max_feeder_loading_pct: f64,
    pub overloaded_feeders_count: usize,
    pub darkened_signal_junctions_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterGridSummary {
    pub num_water_plants_online: usize,
    pub booster_pumps_energized: bool,
    pub total_municipal_demand_lps: f64,
    pub fire_hydrant_demand_lps: f64,
    pub total_supply_lps: f64,
    pub pumping_power_kw: f64,
    pub min_residual_pressure_psi: f64,
    pub mean_residual_pressure_psi: f64,
    pub max_residual_pressure_psi: f64,
    pub nodes_meeting_20psi_fire_pct: f64,
    pub low_pressure_deficit_nodes: usize,
    pub max_pipe_velocity_mps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WastewaterGridSummary {
    pub num_lift_stations_total: usize,
    pub num_lift_stations_operational: usize,
    pub wwtp_energized: bool,
    pub dry_weather_sanitary_lps: f64,
    pub wet_weather_rdii_lps: f64,
    pub total_wwtp_influent_lps: f64,
    pub wwtp_design_capacity_lps: f64,
    pub wwtp_utilization_pct: f64,
    pub influent_bod5_kg_per_hr: f64,
    pub lift_and_wwtp_power_kw: f64,
    pub max_pipe_d_over_d: f64,
    pub mean_pipe_d_over_d: f64,
    pub surcharged_sso_nodes_count: usize,
    pub estimated_sso_overflow_lps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityGridsReport {
    pub clock_hour: f64,
    pub electrical: ElectricalGridSummary,
    pub water: WaterGridSummary,
    pub wastewater: WastewaterGridSummary,
    pub assets: Vec<UtilityFacilityNode>,
    pub node_telemetry: Vec<UtilityJunctionTelemetry>,
    /// System-wide composite 3-grid lifeline resilience score (`0..100%`).
    pub composite_lifeline_score_pct: f64,
    pub solve_time_ms: f64,
}

impl Default for UtilityGridsReport {
    fn default() -> Self {
        Self {
            clock_hour: 8.0,
            electrical: ElectricalGridSummary {
                num_substations_total: 0,
                num_substations_online: 0,
                energized_nodes_pct: 100.0,
                total_load_mw: 0.0,
                solar_pv_generation_mw: 0.0,
                pump_and_wwtp_load_mw: 0.0,
                net_substation_import_mw: 0.0,
                line_resistive_losses_kw: 0.0,
                loss_pct: 0.0,
                min_voltage_pu: 1.0,
                max_voltage_pu: 1.0,
                ansi_c84_1_compliance_pct: 100.0,
                max_feeder_loading_pct: 0.0,
                overloaded_feeders_count: 0,
                darkened_signal_junctions_count: 0,
            },
            water: WaterGridSummary {
                num_water_plants_online: 0,
                booster_pumps_energized: true,
                total_municipal_demand_lps: 0.0,
                fire_hydrant_demand_lps: 0.0,
                total_supply_lps: 0.0,
                pumping_power_kw: 0.0,
                min_residual_pressure_psi: 55.0,
                mean_residual_pressure_psi: 60.0,
                max_residual_pressure_psi: 70.0,
                nodes_meeting_20psi_fire_pct: 100.0,
                low_pressure_deficit_nodes: 0,
                max_pipe_velocity_mps: 0.0,
            },
            wastewater: WastewaterGridSummary {
                num_lift_stations_total: 0,
                num_lift_stations_operational: 0,
                wwtp_energized: true,
                dry_weather_sanitary_lps: 0.0,
                wet_weather_rdii_lps: 0.0,
                total_wwtp_influent_lps: 0.0,
                wwtp_design_capacity_lps: 100.0,
                wwtp_utilization_pct: 0.0,
                influent_bod5_kg_per_hr: 0.0,
                lift_and_wwtp_power_kw: 0.0,
                max_pipe_d_over_d: 0.0,
                mean_pipe_d_over_d: 0.0,
                surcharged_sso_nodes_count: 0,
                estimated_sso_overflow_lps: 0.0,
            },
            assets: Vec::new(),
            node_telemetry: Vec::new(),
            composite_lifeline_score_pct: 100.0,
            solve_time_ms: 0.0,
        }
    }
}

impl UtilityGridsReport {
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity(self.node_telemetry.len() * 110 + 256);
        out.push_str(UTILITY_NODE_CSV_HEADER);
        out.push('\n');
        for n in &self.node_telemetry {
            out.push_str(&format!(
                "{},{:.2},{:.2},{:.2},{},{:.4},{},{:.2},{:.2},{:.2},{},{:.3},{:.3},{},{}\n",
                n.junction_id,
                n.pos.x,
                n.pos.y,
                n.elevation_m,
                n.energized,
                n.voltage_pu,
                n.ansi_c84_1_compliant,
                n.real_load_kw,
                n.hydraulic_head_m,
                n.residual_pressure_psi,
                n.meets_20psi_fire_flow,
                n.cumulative_sewer_flow_lps,
                n.sewer_d_over_d,
                n.sewer_surcharged_sso,
                n.signal_darkened,
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
        Ok(self.node_telemetry.len())
    }
}

/// Converts circular pipe flow ratio `q_ratio = Q / Q_full` into normalized depth `d / D`
/// using an accurate monotonic Camp (1946) / Manning partial-pipe inverse approximation
/// (`q/Q_full == 0 -> d/D = 0`, `q/Q_full == 0.50 -> d/D = 0.50`, `q/Q_full >= 1.0 -> surcharged >= 1.0`).
#[inline]
pub fn manning_circular_pipe_d_over_d(q_over_qfull: f64) -> f64 {
    if q_over_qfull <= 0.0 {
        return 0.0;
    }
    if q_over_qfull >= 1.0 {
        // Surcharged pressurized regime above pipe crown (`d/D > 1.0` indicates hydraulic grade surcharge)
        return (1.0 + 0.35 * (q_over_qfull - 1.0)).min(2.5);
    }
    // Exact symmetry at half-full (`q/Q_full = 0.5 -> d/D = 0.5`) with circular wetted-perimeter curvature
    let u = q_over_qfull.clamp(0.0, 1.0);
    (0.5 + 0.5 * (2.0 * u - 1.0).signum() * (2.0 * u - 1.0).abs().powf(0.76)).clamp(0.02, 0.99)
}

/// Hazen-Williams SI pipe head loss `h_f` (meters of water) for length `len_m`, flow `q_m3_s`,
/// roughness `c_hw`, and internal diameter `diam_m`:
/// `h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.8704)`.
#[inline]
pub fn hazen_williams_head_loss_m(len_m: f64, q_m3_s: f64, c_hw: f64, diam_m: f64) -> f64 {
    let q = q_m3_s.abs();
    if q <= 1e-12 || len_m <= 0.0 {
        return 0.0;
    }
    let denom = c_hw.max(40.0).powf(1.852) * diam_m.max(0.05).powf(4.8704);
    10.67 * len_m * q.powf(1.852) / denom.max(1e-12)
}

/// Manning's SI full-pipe gravity discharge capacity `Q_full` (`m^3/s`) for circular sewer diameter
/// `diam_m`, roughness `n_manning`, and invert slope `slope_m_per_m`:
/// `Q_full = (1 / n) * (pi * D^2 / 4) * (D / 4)^(2/3) * S^(1/2)`.
#[inline]
pub fn manning_full_pipe_capacity_m3_s(diam_m: f64, n_manning: f64, slope_m_per_m: f64) -> f64 {
    let d = diam_m.max(0.10);
    let area = std::f64::consts::PI * 0.25 * d * d;
    let r_h = 0.25 * d;
    let s = slope_m_per_m.max(0.0012);
    (1.0 / n_manning.max(0.008)) * area * r_h.powf(2.0 / 3.0) * s.sqrt()
}

/// Evaluates the coupled **Electrical (`12.47 kV` DistFlow)**, **Potable Water (`Hazen-Williams`)**,
/// and **Wastewater (`Manning + EPA RDII`)** grids across `net` and `parcels` at `clock_hour`.
pub fn evaluate_coupled_utility_grids(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    clock_hour: f64,
    cfg: &UtilityGridsConfig,
) -> UtilityGridsReport {
    let t0 = Instant::now();
    let n_junc = net.junctions.len();
    if n_junc == 0 {
        return UtilityGridsReport::default();
    }

    let hr_idx = (clock_hour.floor() as isize).rem_euclid(24) as usize;
    let diurnal_mult = USFA_DIURNAL_CALL_MULTIPLIERS[hr_idx];
    // Solar PV bell curve between 06:30 and 18:30 peaking at 12:30 solar noon
    let solar_cf = {
        let dh = ((clock_hour.rem_euclid(24.0)) - 12.5).abs();
        if dh < 6.0 {
            ((std::f64::consts::PI * 0.5 * (dh / 6.0)).cos())
                .powi(2)
                .clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    // Evening EV home/commercial charging surge (`17:00..22:00` + morning workplace `08:00..11:00`)
    let ev_mult = if (17..22).contains(&hr_idx) {
        1.55
    } else if (8..12).contains(&hr_idx) {
        1.10
    } else {
        0.45
    };

    // 1. Synthesize gentle urban topographic elevation `z_m` sloping from NW high ground (`+26 m`)
    //    down toward SE riverfront (`+2 m`) where the `WWTP` sits, plus localized low pockets!
    let b = net.bounds;
    let span_x = (b.max_x - b.min_x).max(50.0);
    let span_y = (b.max_y - b.min_y).max(50.0);
    let elev_m: Vec<f64> = net
        .junctions
        .iter()
        .map(|j| {
            let nx = ((j.pos.x - b.min_x) / span_x).clamp(0.0, 1.0);
            let ny = ((j.pos.y - b.min_y) / span_y).clamp(0.0, 1.0);
            // NW (`nx=0, ny=0`) is upland (+26m); SE (`nx=1, ny=1`) is river outfall (+3m)
            26.0 - 13.5 * nx - 9.5 * ny + 1.4 * ((nx * 6.28).sin() * (ny * 3.14).cos())
        })
        .collect();

    // Build undirected utility corridor edges `(u, v, len_m, is_arterial, intact)` from `net.segments`
    #[derive(Clone, Copy)]
    struct CorridorEdge {
        u: usize,
        v: usize,
        len_m: f64,
        is_arterial: bool,
    }
    let mut edges: Vec<CorridorEdge> = Vec::new();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_junc];
    for seg in &net.segments {
        if cfg.couple_closed_roads_to_utility_mains && seg.closed {
            continue;
        }
        let u = seg.from_junc;
        let v = seg.to_junc;
        if u == v || u >= n_junc || v >= n_junc {
            continue;
        }
        let is_art = seg.lane_count >= 2 || seg.priority >= 2 || seg.speed_mps >= 13.5;
        if let Some(ei) = edges
            .iter()
            .position(|e| (e.u == u && e.v == v) || (e.u == v && e.v == u))
        {
            if is_art {
                edges[ei].is_arterial = true;
            }
        } else {
            let ei = edges.len();
            edges.push(CorridorEdge {
                u,
                v,
                len_m: seg.length.max(12.0),
                is_arterial: is_art,
            });
            adj[u].push(ei);
            adj[v].push(ei);
        }
    }

    // 2. Select anchor junctions for Substations, Water Treatment/Booster Plants, Lift Stations & WWTP
    //    - Substations: NW upland hub (`min nx+ny`) and NE/SW secondary hub (`max |nx - ny|`)
    let mut junc_order_elev_desc: Vec<usize> = (0..n_junc).collect();
    junc_order_elev_desc.sort_by(|&a, &b| elev_m[b].partial_cmp(&elev_m[a]).unwrap());

    let sub1_j = junc_order_elev_desc[0];
    let sub2_j = (0..n_junc)
        .filter(|&j| j != sub1_j)
        .max_by(|&a, &b| {
            let da = net.junctions[a].pos.dist(net.junctions[sub1_j].pos);
            let db = net.junctions[b].pos.dist(net.junctions[sub1_j].pos);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap_or(sub1_j);
    let substation_juncs = if sub2_j != sub1_j && n_junc >= 4 {
        vec![sub1_j, sub2_j]
    } else {
        vec![sub1_j]
    };

    // Water Treatment / High-Service Booster Plants at the two highest-elevation upland nodes
    let wtp1_j = junc_order_elev_desc[0];
    let wtp2_j = junc_order_elev_desc
        .iter()
        .copied()
        .find(|&j| j != wtp1_j && net.junctions[j].pos.dist(net.junctions[wtp1_j].pos) > span_x * 0.30)
        .unwrap_or(wtp1_j);
    let wtp_juncs = if wtp2_j != wtp1_j && n_junc >= 6 {
        vec![wtp1_j, wtp2_j]
    } else {
        vec![wtp1_j]
    };

    // WWTP at lowest-elevation riverfront outfall node; Lift Station at secondary low pocket
    let wwtp_j = *junc_order_elev_desc.last().unwrap_or(&0);
    let lift_j = junc_order_elev_desc
        .iter()
        .rev()
        .copied()
        .find(|&j| j != wwtp_j && net.junctions[j].pos.dist(net.junctions[wwtp_j].pos) > span_x * 0.28)
        .unwrap_or(wwtp_j);
    let lift_juncs = if lift_j != wwtp_j && n_junc >= 5 {
        vec![lift_j]
    } else {
        Vec::new()
    };

    // 3. Map each Parcel to its nearest Junction & aggregate parcel Electric, Water, and Wastewater loads
    let mut node_elec_load_kw = vec![0.0f64; n_junc];
    let mut node_solar_kw = vec![0.0f64; n_junc];
    let mut node_water_lps = vec![0.0f64; n_junc];
    let mut node_dwf_lps = vec![0.0f64; n_junc];
    let mut node_rdii_lps = vec![0.0f64; n_junc];
    let mut total_bod_kg_hr = 0.0f64;

    let ev_frac = (cfg.ev_charging_penetration_pct / 100.0).clamp(0.0, 1.0);
    let pv_frac = (cfg.solar_pv_penetration_pct / 100.0).clamp(0.0, 1.0);

    for p in parcels {
        let mut near_j = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (ji, j) in net.junctions.iter().enumerate() {
            let dx = p.center.x - j.pos.x;
            let dy = p.center.y - j.pos.y;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                near_j = ji;
            }
        }

        let res = p.residents();
        let jobs = p.jobs();
        let sp = p.zone.spec();

        // Electrical load: residential (~1.35 kW/cap peak) + commercial/industrial (~2.4 kW/job) + EV charging
        let base_kw = (0.72 * res + 1.45 * jobs + 4.0 * p.area_ha) * (0.55 + 0.45 * diurnal_mult);
        let ev_kw = (0.38 * res + 0.22 * jobs) * ev_frac * ev_mult;
        let pv_cap_kw = (0.85 * res + 0.95 * jobs + 12.0 * p.area_ha) * pv_frac;
        let pv_kw = pv_cap_kw * solar_cf;

        node_elec_load_kw[near_j] += base_kw + ev_kw;
        node_solar_kw[near_j] += pv_kw;

        // Potable water demand: ~210 L/cap/day (`0.00243 L/s/resident`) + commercial (`0.0016 L/s/job`), scaled diurnally
        let w_lps = ((0.0028 * res + 0.0019 * jobs) * (0.50 + 0.55 * diurnal_mult)).max(0.02);
        node_water_lps[near_j] += w_lps;

        // Sanitary Dry-Weather Flow (`85%` indoor water return) + EPA Wet-Weather `RDII` from storm rainfall & imperviousness!
        let dwf = 0.85 * w_lps;
        // Rational RDII: fraction of surface stormwater infiltrating/inflowing sanitary manholes & defective laterals
        // High imperviousness (`CBD=0.86`, `IND=0.78`) produces markedly higher inflow than green `PARK=0.08` bioswales!
        let rdii_coeff = 0.042 * sp.impervious_fraction + 0.008 * (1.0 - sp.canopy_fraction);
        let rdii = cfg.storm_rainfall_mm_hr.max(0.0) * 2.778 * p.area_ha * rdii_coeff;
        node_dwf_lps[near_j] += dwf;
        node_rdii_lps[near_j] += rdii;
        total_bod_kg_hr += (res * 0.0027 + jobs * 0.0014) * (0.65 + 0.35 * diurnal_mult);
    }

    // Add Traffic Signal Controller electrical loads (`1.4 kW` per signalized junction)
    for (ji, j) in net.junctions.iter().enumerate() {
        if matches!(
            j.control_mode,
            ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::MaxPressureAdaptive
        ) {
            node_elec_load_kw[ji] += 1.4;
        }
    }

    // Inject active Fire-Hydrant Suppression Flow (`GPM -> L/s`, `1 GPM = 0.0630902 L/s`) at centroid
    // fire-demand junction (nearest placed FireStation or central CBD junction)
    let fire_flow_lps = (cfg.active_fire_flow_demand_gpm.max(0.0)) * 0.063_090_2;
    let hydrant_j = facilities
        .iter()
        .find(|f| f.kind == EmergencyFacilityKind::FireStation)
        .and_then(|f| net.segments.get(f.segment_id).map(|s| s.to_junc))
        .unwrap_or(n_junc / 2);
    let mut node_water_with_fire_lps = node_water_lps.clone();
    if hydrant_j < n_junc {
        node_water_with_fire_lps[hydrant_j] += fire_flow_lps;
    }

    // Preliminary estimate of water & wastewater pumping kW (refined once energizing status is solved!)
    let tot_mun_water_lps: f64 = node_water_lps.iter().sum();
    let tot_dwf_lps: f64 = node_dwf_lps.iter().sum();
    let tot_rdii_lps: f64 = node_rdii_lps.iter().sum();
    let nom_booster_kw = (9.81 * (tot_mun_water_lps + fire_flow_lps) * 46.0) / (0.78 * 1000.0);
    let nom_wwtp_kw = 38.0 + 0.42 * (tot_dwf_lps + tot_rdii_lps);
    let nom_lift_kw = 14.0 + 0.18 * (tot_dwf_lps + tot_rdii_lps);

    for &wj in &wtp_juncs {
        node_elec_load_kw[wj] += nom_booster_kw / (wtp_juncs.len() as f64);
    }
    for &lj in &lift_juncs {
        node_elec_load_kw[lj] += nom_lift_kw;
    }
    node_elec_load_kw[wwtp_j] += nom_wwtp_kw;

    // =========================================================================
    // SUBSYSTEM 1: 12.47 kV ELECTRICAL DISTRIBUTION POWER FLOW (DISTFLOW / NODAL)
    // =========================================================================
    let active_subs: Vec<usize> = substation_juncs
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, _)| !cfg.tripped_substation_indices.contains(idx))
        .map(|(_, jid)| jid)
        .collect();

    // BFS reachability from online substations across intact distribution feeders
    let mut energized = vec![false; n_junc];
    let mut q_bfs = VecDeque::new();
    for &sj in &active_subs {
        energized[sj] = true;
        q_bfs.push_back(sj);
    }
    while let Some(u) = q_bfs.pop_front() {
        for &ei in &adj[u] {
            let e = edges[ei];
            let v = if e.u == u { e.v } else { e.u };
            if !energized[v] {
                energized[v] = true;
                q_bfs.push_back(v);
            }
        }
    }

    // Linearized 12.47 kV DistFlow nodal voltage solve:
    // Base voltage `V_base = 12.47 kV`, `S_base = 10 MVA` (`10,000 kVA`) -> `Z_base = 15.55 ohm`.
    // Typical overhead ACSR 336.4 kcmil arterial (`r = 0.19 ohm/km`, `x = 0.38 ohm/km`) vs lateral (`r = 0.48 ohm/km`).
    // Under linearized DistFlow (`v_i - v_j approx (r_ij * P_ij + x_ij * Q_ij) / V_0`, `pf = 0.92` -> `Q = 0.426 * P`),
    // effective voltage-drop impedance per kW-km is `(r + 0.426 * x) / (12,470 * 1,000)`.
    let v_set = cfg.substation_voltage_pu.clamp(0.95, 1.05);
    let mut voltage_pu = vec![0.0f64; n_junc];
    for j in 0..n_junc {
        if energized[j] {
            voltage_pu[j] = v_set;
        }
    }

    let edge_conductance: Vec<f64> = edges
        .iter()
        .map(|e| {
            let len_km = (e.len_m / 1000.0).max(0.015);
            let r_eff_ohm = if e.is_arterial {
                (0.19 + 0.426 * 0.36) * len_km
            } else {
                (0.46 + 0.426 * 0.41) * len_km
            };
            // Convert `Z_base / r_eff` into per-unit conductance per kW (`S_base = 10,000 kW -> Z_base = 15.55 ohm`)
            15.55 / r_eff_ohm.max(1e-4)
        })
        .collect();

    if !active_subs.is_empty() {
        for _iter in 0..160 {
            let mut max_dv = 0.0f64;
            for i in 0..n_junc {
                if !energized[i] || active_subs.contains(&i) {
                    continue;
                }
                let net_p_pu = (node_elec_load_kw[i] - node_solar_kw[i]) / 10_000.0;
                let mut g_sum = 0.0;
                let mut gv_sum = 0.0;
                for &ei in &adj[i] {
                    let e = edges[ei];
                    let nb = if e.u == i { e.v } else { e.u };
                    if energized[nb] {
                        let g = edge_conductance[ei];
                        g_sum += g;
                        gv_sum += g * voltage_pu[nb];
                    }
                }
                if g_sum > 1e-9 {
                    let v_new = ((gv_sum - net_p_pu) / g_sum).clamp(0.72, 1.09);
                    let next_v = 0.35 * voltage_pu[i] + 0.65 * v_new;
                    max_dv = max_dv.max((next_v - voltage_pu[i]).abs());
                    voltage_pu[i] = next_v;
                }
            }
            if max_dv < 1e-6 {
                break;
            }
        }
    }

    // Compute feeder loading (`%` of thermal MVA rating) and `I^2 R` resistive losses (`kW`)
    let mut line_losses_kw = 0.0f64;
    let mut max_feeder_loading_pct = 0.0f64;
    let mut overloaded_feeders_count = 0usize;
    for (ei, e) in edges.iter().enumerate() {
        if !energized[e.u] || !energized[e.v] {
            continue;
        }
        let dv = (voltage_pu[e.u] - voltage_pu[e.v]).abs();
        let flow_pu = edge_conductance[ei] * dv;
        let flow_mva = (flow_pu * 10.0) / 0.92; // `10 MVA` base, `pf = 0.92`
        let thermal_limit_mva = if e.is_arterial { 8.5 } else { 3.2 };
        let loading_pct = (flow_mva / thermal_limit_mva) * 100.0;
        if loading_pct > max_feeder_loading_pct {
            max_feeder_loading_pct = loading_pct;
        }
        if loading_pct > 100.0 {
            overloaded_feeders_count += 1;
        }
        // `P_loss = G_real * dV^2 * S_base_kW`
        line_losses_kw += 0.68 * edge_conductance[ei] * dv * dv * 10_000.0;
    }

    // =========================================================================
    // SUBSYSTEM 2: POTABLE WATER GRID (HAZEN-WILLIAMS + BOOSTER CASCADE + 20 PSI FIRE)
    // =========================================================================
    let booster_grid_powered = wtp_juncs
        .iter()
        .any(|&wj| energized[wj] && voltage_pu[wj] >= 0.88);
    let booster_operational = booster_grid_powered || cfg.critical_backup_generators_enabled;

    // Source hydraulic grade line (`HGL`, meters):
    // - With energized/backed-up high-service booster pumps: `elev + 52.0 m` (~74 psi static head)
    // - Without booster power (gravity standpipe reserve only!): drops to `elev + 15.5 m` (~22 psi static -> sags < 20 psi under demand/fire!)
    let storage_head_scale = if cfg.finished_water_storage_fill_pct >= 65.0 {
        1.0
    } else {
        (0.22 + 0.78 * (cfg.finished_water_storage_fill_pct.clamp(0.0, 65.0) / 65.0))
            .clamp(0.22, 1.0)
    };
    let booster_added_head_m = (if booster_operational { 52.0 } else { 15.5 }) * storage_head_scale;
    let mut head_m = vec![0.0f64; n_junc];
    let avg_src_h = wtp_juncs
        .iter()
        .map(|&wj| elev_m[wj] + booster_added_head_m)
        .sum::<f64>()
        / (wtp_juncs.len() as f64);
    head_m.fill(avg_src_h - 4.0);
    for &wj in &wtp_juncs {
        head_m[wj] = elev_m[wj] + booster_added_head_m;
    }

    // Hazen-Williams conductance `Q_ij = sign(dH) * K_hw * |dH|^0.54` (`m^3/s`)
    // From `h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.8704)`:
    // `Q = (C^1.852 * D^4.8704 / (10.67 * L))^(1 / 1.852) * |dH|^0.539957`
    let edge_hw_k: Vec<(f64, f64)> = edges
        .iter()
        .map(|e| {
            let diam_m: f64 = if e.is_arterial { 0.35 } else { 0.20 };
            let r_pipe = 10.67 * e.len_m
                / (cfg.hazen_williams_c.clamp(60.0, 150.0).powf(1.852) * diam_m.powf(4.8704));
            let k_hw = (1.0 / r_pipe.max(1e-9)).powf(1.0 / 1.852);
            let area = std::f64::consts::PI * 0.25 * diam_m * diam_m;
            (k_hw, area)
        })
        .collect();

    // Iterative Todini-Pilati / Secant Conductance Hydraulic Network Relaxation (with fire-flow load)
    for _iter in 0..180 {
        let mut max_dh = 0.0f64;
        for i in 0..n_junc {
            if wtp_juncs.contains(&i) {
                head_m[i] = elev_m[i] + booster_added_head_m;
                continue;
            }
            let demand_m3_s = node_water_with_fire_lps[i] / 1000.0;
            let mut g_sec_sum = 0.0;
            let mut gh_sum = 0.0;
            for &ei in &adj[i] {
                let e = edges[ei];
                let nb = if e.u == i { e.v } else { e.u };
                let dh_abs = (head_m[i] - head_m[nb]).abs().max(0.002);
                // Secant hydraulic conductance `G_ij = K_hw * |dH|^(0.54 - 1.0)`
                let g_sec = edge_hw_k[ei].0 * dh_abs.powf(-0.460043);
                g_sec_sum += g_sec;
                gh_sum += g_sec * head_m[nb];
            }
            if g_sec_sum > 1e-12 {
                let h_target = (gh_sum - demand_m3_s) / g_sec_sum;
                let h_next = 0.45 * head_m[i] + 0.55 * h_target;
                max_dh = max_dh.max((h_next - head_m[i]).abs());
                head_m[i] = h_next;
            }
        }
        if max_dh < 1e-4 {
            break;
        }
    }

    let mut max_pipe_vel_mps = 0.0f64;
    for (ei, e) in edges.iter().enumerate() {
        let dh = (head_m[e.u] - head_m[e.v]).abs();
        let q_m3_s = edge_hw_k[ei].0 * dh.powf(0.539957);
        let vel = q_m3_s / edge_hw_k[ei].1.max(1e-6);
        if vel > max_pipe_vel_mps {
            max_pipe_vel_mps = vel;
        }
    }

    let residual_psi: Vec<f64> = (0..n_junc)
        .map(|i| ((head_m[i] - elev_m[i]) * PSI_PER_METER_H2O).max(0.0))
        .collect();

    // =========================================================================
    // SUBSYSTEM 3: WASTEWATER GRAVITY SEWER + RDII + LIFT STATIONS + WWTP
    // =========================================================================
    let wwtp_operational =
        (energized[wwtp_j] && voltage_pu[wwtp_j] >= 0.88) || cfg.critical_backup_generators_enabled;
    let mut lift_online_count = 0usize;
    let mut failed_lift_juncs: Vec<usize> = Vec::new();
    for &lj in &lift_juncs {
        let ok =
            (energized[lj] && voltage_pu[lj] >= 0.88) || cfg.critical_backup_generators_enabled;
        if ok {
            lift_online_count += 1;
        } else {
            failed_lift_juncs.push(lj);
        }
    }

    // Route sanitary `DWF + RDII` downslope from highest-elevation nodes -> `wwtp_j`
    let mut cum_sewer_lps: Vec<f64> = (0..n_junc)
        .map(|i| node_dwf_lps[i] + node_rdii_lps[i])
        .collect();
    let mut node_d_over_d = vec![0.18f64; n_junc];
    let mut node_sso = vec![false; n_junc];
    let mut sso_overflow_lps = 0.0f64;

    // Compute shortest topological distance to `wwtp_j` as clean tie-breaker for flat pockets / lift-station force mains
    let mut hops_to_wwtp = vec![usize::MAX; n_junc];
    let mut q_w = VecDeque::new();
    hops_to_wwtp[wwtp_j] = 0;
    q_w.push_back(wwtp_j);
    while let Some(u) = q_w.pop_front() {
        for &ei in &adj[u] {
            let e = edges[ei];
            let v = if e.u == u { e.v } else { e.u };
            if hops_to_wwtp[v] == usize::MAX {
                hops_to_wwtp[v] = hops_to_wwtp[u] + 1;
                q_w.push_back(v);
            }
        }
    }

    // Sort nodes strictly descending by `(elev_m, hops_to_wwtp)` so every node drains into its steepest downslope / WWTP-ward neighbor
    let mut drain_order: Vec<usize> = (0..n_junc).collect();
    drain_order.sort_by(|&a, &b| {
        hops_to_wwtp[b]
            .cmp(&hops_to_wwtp[a])
            .then_with(|| elev_m[b].partial_cmp(&elev_m[a]).unwrap())
    });

    for &u in &drain_order {
        if u == wwtp_j {
            continue;
        }
        // If `u` is a failed unpowered sewage lift station, wet-well stalls and overflows locally (`SSO`)!
        if failed_lift_juncs.contains(&u) {
            node_d_over_d[u] = 1.65;
            node_sso[u] = true;
            sso_overflow_lps += cum_sewer_lps[u] * 0.85;
            continue;
        }

        // Pick best downstream neighbor closer to `wwtp_j`
        let mut best_out: Option<(usize, usize)> = None;
        let mut best_score = f64::NEG_INFINITY;
        for &ei in &adj[u] {
            let e = edges[ei];
            let v = if e.u == u { e.v } else { e.u };
            if hops_to_wwtp[v] < hops_to_wwtp[u] {
                let dz = elev_m[u] - elev_m[v];
                let sc = dz * 10.0 + if e.is_arterial { 4.0 } else { 0.0 };
                if sc > best_score {
                    best_score = sc;
                    best_out = Some((v, ei));
                }
            }
        }

        if let Some((v_down, ei)) = best_out {
            let e = edges[ei];
            let slope = ((elev_m[u] - elev_m[v_down]).max(0.18) / e.len_m).clamp(0.0018, 0.045);
            // Trunk interceptors near WWTP (`hops <= 2` or arterial) use larger concrete mains (`0.55..0.75 m`)
            let diam_m = if hops_to_wwtp[u] <= 2 {
                0.70
            } else if e.is_arterial {
                0.45
            } else {
                0.28
            };
            let q_full_lps =
                manning_full_pipe_capacity_m3_s(diam_m, cfg.manning_sewer_n, slope) * 1000.0;
            let q_ratio = cum_sewer_lps[u] / q_full_lps.max(1.0);
            let dd = manning_circular_pipe_d_over_d(q_ratio);
            node_d_over_d[u] = dd;

            if q_ratio > 1.0 {
                node_sso[u] = true;
                let spill = (cum_sewer_lps[u] - q_full_lps).max(0.0);
                sso_overflow_lps += spill;
                cum_sewer_lps[v_down] += q_full_lps;
            } else {
                cum_sewer_lps[v_down] += cum_sewer_lps[u];
            }
        } else {
            // Disconnected sewer pocket (e.g., severed interceptor bridge!)
            node_d_over_d[u] = 1.40;
            node_sso[u] = true;
            sso_overflow_lps += cum_sewer_lps[u];
        }
    }

    let wwtp_influent_lps = cum_sewer_lps[wwtp_j];
    // Design WWTP peak capacity ~ `3.2x` dry-weather baseline
    let wwtp_design_cap_lps = (tot_dwf_lps * 3.4).max(35.0);
    let wwtp_util_pct = (wwtp_influent_lps / wwtp_design_cap_lps) * 100.0;
    if !wwtp_operational {
        node_sso[wwtp_j] = true;
        node_d_over_d[wwtp_j] = 1.80;
        sso_overflow_lps += wwtp_influent_lps * 0.75;
    } else if wwtp_influent_lps > wwtp_design_cap_lps {
        node_sso[wwtp_j] = true;
        node_d_over_d[wwtp_j] = manning_circular_pipe_d_over_d(wwtp_influent_lps / wwtp_design_cap_lps);
        sso_overflow_lps += wwtp_influent_lps - wwtp_design_cap_lps;
    } else {
        node_d_over_d[wwtp_j] =
            manning_circular_pipe_d_over_d((wwtp_influent_lps / wwtp_design_cap_lps) * 0.82);
    }

    // =========================================================================
    // ASSEMBLE CROSS-LIFELINE TELEMETRY, ASSETS & HEALTH SCORES
    // =========================================================================
    let mut energized_cnt = 0usize;
    let mut ansi_ok_cnt = 0usize;
    let mut darkened_signals = 0usize;
    let mut min_v = 2.0f64;
    let mut max_v = 0.0f64;

    let mut fire_20psi_ok_cnt = 0usize;
    let mut min_psi = f64::INFINITY;
    let mut max_psi = 0.0f64;
    let mut sum_psi = 0.0f64;

    let mut sso_cnt = 0usize;
    let mut max_dd = 0.0f64;
    let mut sum_dd = 0.0f64;

    let mut node_telemetry = Vec::with_capacity(n_junc);
    let mut health_sum = 0.0f64;

    for i in 0..n_junc {
        let en = energized[i];
        let v_pu = voltage_pu[i];
        if en {
            energized_cnt += 1;
            min_v = min_v.min(v_pu);
            max_v = max_v.max(v_pu);
        } else {
            min_v = 0.0;
        }
        let ansi_ok = en && (0.95..=1.05).contains(&v_pu);
        if ansi_ok {
            ansi_ok_cnt += 1;
        }

        let is_sig = matches!(
            net.junctions[i].control_mode,
            ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::MaxPressureAdaptive
        );
        let sig_dark = is_sig && (!en || v_pu < 0.85) && !cfg.critical_backup_generators_enabled;
        if sig_dark {
            darkened_signals += 1;
        }

        let psi = residual_psi[i];
        min_psi = min_psi.min(psi);
        max_psi = max_psi.max(psi);
        sum_psi += psi;
        let fire_ok = psi >= MIN_FIRE_FLOW_RESIDUAL_PSI;
        if fire_ok {
            fire_20psi_ok_cnt += 1;
        }

        let dd = node_d_over_d[i];
        let sso = node_sso[i];
        if sso {
            sso_cnt += 1;
        }
        max_dd = max_dd.max(dd);
        sum_dd += dd;

        let e_health = if !en {
            0.0
        } else if ansi_ok {
            1.0
        } else {
            (1.0 - ((0.95 - v_pu).max(v_pu - 1.05) * 8.0)).clamp(0.15, 0.85)
        };
        let w_health = if psi >= 35.0 {
            1.0
        } else if psi >= MIN_FIRE_FLOW_RESIDUAL_PSI {
            0.65 + 0.35 * ((psi - 20.0) / 15.0)
        } else {
            (psi / 20.0).clamp(0.0, 0.55)
        };
        let s_health = if sso {
            0.10
        } else if dd <= 0.75 {
            1.0
        } else {
            (1.0 - (dd - 0.75) * 2.4).clamp(0.25, 0.95)
        };
        let comp = 0.38 * e_health + 0.34 * w_health + 0.28 * s_health;
        health_sum += comp;

        node_telemetry.push(UtilityJunctionTelemetry {
            junction_id: i,
            pos: net.junctions[i].pos,
            elevation_m: elev_m[i],
            energized: en,
            voltage_pu: v_pu,
            ansi_c84_1_compliant: ansi_ok,
            real_load_kw: node_elec_load_kw[i],
            solar_gen_kw: node_solar_kw[i],
            signal_darkened: sig_dark,
            hydraulic_head_m: head_m[i],
            residual_pressure_psi: psi,
            water_demand_lps: node_water_with_fire_lps[i],
            meets_20psi_fire_flow: fire_ok,
            sanitary_dwf_lps: node_dwf_lps[i],
            wet_weather_rdii_lps: node_rdii_lps[i],
            cumulative_sewer_flow_lps: cum_sewer_lps[i],
            sewer_d_over_d: dd,
            sewer_surcharged_sso: sso,
            composite_health: comp,
        });
    }

    if min_v > 1.5 {
        min_v = 0.0;
    }

    let tot_load_kw: f64 = node_elec_load_kw.iter().sum();
    let tot_pv_kw: f64 = node_solar_kw
        .iter()
        .enumerate()
        .filter(|&(i, _)| energized[i])
        .map(|(_, &v)| v)
        .sum();
    let energized_load_kw: f64 = node_elec_load_kw
        .iter()
        .enumerate()
        .filter(|&(i, _)| energized[i])
        .map(|(_, &v)| v)
        .sum();
    let realized_pump_kw = if booster_operational {
        nom_booster_kw
    } else {
        0.0
    };
    let realized_ww_kw = (if wwtp_operational { nom_wwtp_kw } else { 0.0 })
        + (lift_online_count as f64) * nom_lift_kw;
    let net_import_kw = (energized_load_kw - tot_pv_kw + line_losses_kw).max(0.0);

    let mut assets = Vec::new();
    let mut aid = 0usize;
    for (idx, &sj) in substation_juncs.iter().enumerate() {
        let on = !cfg.tripped_substation_indices.contains(&idx);
        assets.push(UtilityFacilityNode {
            id: aid,
            kind: UtilityAssetKind::ElectricSubstation,
            label: format!("12.47kV Substation #{}", idx + 1),
            junction_id: sj,
            world_pos: net.junctions[sj].pos,
            online: on,
            on_backup_power: false,
            load_or_output_kw: if on && !active_subs.is_empty() {
                net_import_kw / (active_subs.len() as f64)
            } else {
                0.0
            },
            flow_lps: 0.0,
        });
        aid += 1;
    }
    for (idx, &wj) in wtp_juncs.iter().enumerate() {
        let grid_ok = energized[wj] && voltage_pu[wj] >= 0.88;
        assets.push(UtilityFacilityNode {
            id: aid,
            kind: UtilityAssetKind::WaterTreatmentBooster,
            label: format!("Water Booster Plant #{}", idx + 1),
            junction_id: wj,
            world_pos: net.junctions[wj].pos,
            online: booster_operational,
            on_backup_power: !grid_ok && cfg.critical_backup_generators_enabled,
            load_or_output_kw: realized_pump_kw / (wtp_juncs.len() as f64),
            flow_lps: (tot_mun_water_lps + fire_flow_lps) / (wtp_juncs.len() as f64),
        });
        aid += 1;
    }
    for (idx, &lj) in lift_juncs.iter().enumerate() {
        let grid_ok = energized[lj] && voltage_pu[lj] >= 0.88;
        let on = grid_ok || cfg.critical_backup_generators_enabled;
        assets.push(UtilityFacilityNode {
            id: aid,
            kind: UtilityAssetKind::SewageLiftStation,
            label: format!("Sewage Lift Station #{}", idx + 1),
            junction_id: lj,
            world_pos: net.junctions[lj].pos,
            online: on,
            on_backup_power: !grid_ok && cfg.critical_backup_generators_enabled,
            load_or_output_kw: if on { nom_lift_kw } else { 0.0 },
            flow_lps: cum_sewer_lps[lj],
        });
        aid += 1;
    }
    {
        let grid_ok = energized[wwtp_j] && voltage_pu[wwtp_j] >= 0.88;
        assets.push(UtilityFacilityNode {
            id: aid,
            kind: UtilityAssetKind::WastewaterTreatmentPlant,
            label: "Municipal WWTP Outfall".to_string(),
            junction_id: wwtp_j,
            world_pos: net.junctions[wwtp_j].pos,
            online: wwtp_operational,
            on_backup_power: !grid_ok && cfg.critical_backup_generators_enabled,
            load_or_output_kw: if wwtp_operational { nom_wwtp_kw } else { 0.0 },
            flow_lps: wwtp_influent_lps,
        });
    }

    let nf = n_junc as f64;
    UtilityGridsReport {
        clock_hour,
        electrical: ElectricalGridSummary {
            num_substations_total: substation_juncs.len(),
            num_substations_online: active_subs.len(),
            energized_nodes_pct: (energized_cnt as f64 / nf) * 100.0,
            total_load_mw: tot_load_kw / 1000.0,
            solar_pv_generation_mw: tot_pv_kw / 1000.0,
            pump_and_wwtp_load_mw: (realized_pump_kw + realized_ww_kw) / 1000.0,
            net_substation_import_mw: net_import_kw / 1000.0,
            line_resistive_losses_kw: line_losses_kw,
            loss_pct: if energized_load_kw > 1e-6 {
                (line_losses_kw / energized_load_kw) * 100.0
            } else {
                0.0
            },
            min_voltage_pu: min_v,
            max_voltage_pu: max_v,
            ansi_c84_1_compliance_pct: (ansi_ok_cnt as f64 / nf) * 100.0,
            max_feeder_loading_pct,
            overloaded_feeders_count,
            darkened_signal_junctions_count: darkened_signals,
        },
        water: WaterGridSummary {
            num_water_plants_online: if booster_operational {
                wtp_juncs.len()
            } else {
                0
            },
            booster_pumps_energized: booster_operational,
            total_municipal_demand_lps: tot_mun_water_lps,
            fire_hydrant_demand_lps: fire_flow_lps,
            total_supply_lps: tot_mun_water_lps + fire_flow_lps,
            pumping_power_kw: realized_pump_kw,
            min_residual_pressure_psi: if min_psi.is_finite() { min_psi } else { 0.0 },
            mean_residual_pressure_psi: sum_psi / nf,
            max_residual_pressure_psi: max_psi,
            nodes_meeting_20psi_fire_pct: (fire_20psi_ok_cnt as f64 / nf) * 100.0,
            low_pressure_deficit_nodes: n_junc.saturating_sub(fire_20psi_ok_cnt),
            max_pipe_velocity_mps: max_pipe_vel_mps,
        },
        wastewater: WastewaterGridSummary {
            num_lift_stations_total: lift_juncs.len(),
            num_lift_stations_operational: lift_online_count,
            wwtp_energized: wwtp_operational,
            dry_weather_sanitary_lps: tot_dwf_lps,
            wet_weather_rdii_lps: tot_rdii_lps,
            total_wwtp_influent_lps: wwtp_influent_lps,
            wwtp_design_capacity_lps: wwtp_design_cap_lps,
            wwtp_utilization_pct: wwtp_util_pct,
            influent_bod5_kg_per_hr: total_bod_kg_hr,
            lift_and_wwtp_power_kw: realized_ww_kw,
            max_pipe_d_over_d: max_dd,
            mean_pipe_d_over_d: sum_dd / nf,
            surcharged_sso_nodes_count: sso_cnt,
            estimated_sso_overflow_lps: sso_overflow_lps,
        },
        assets,
        node_telemetry,
        composite_lifeline_score_pct: ((health_sum / nf) * 100.0).clamp(0.0, 100.0),
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

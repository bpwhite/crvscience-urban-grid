//! Scenario snapshot `.json` serialization (`ScenarioSnapshot`) and RFC 4180 29-column time-series
//! `.csv` telemetry recorder & exporter (`TelemetryTimeSeriesRecorder`).

use crate::gpu_pack::PlumePhysicsParams;
use crate::emergency::{EmergencyFacility, EvpConfig};
use crate::max_pressure::MaxPressureConfig;
use crate::multimodal::TransitPolicy;
use crate::network::RoadNetwork;
use crate::traffic::{
    CyclistAgent, IdmParams, MobilParams, PedestrianAgent, SimulationWorld, Vehicle,
};
use crate::transit_gtfs::TransitGtfsSystem;
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const SNAPSHOT_SCHEMA_VERSION: &str = "urban-grid-snapshot/v1.5";

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Complete portable `.json` scenario snapshot capturing network geometry, signal timings,
/// Complete Streets allocations, parcel zoning, GTFS routes & stop queues, physics parameters,
/// atmospheric plume wind settings, and (optionally) all active microscopic agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSnapshot {
    pub schema_version: String,
    pub title: String,
    pub sim_time_s: f64,
    pub includes_microscopic_agents: bool,
    pub network: RoadNetwork,
    pub parcels: Vec<Parcel>,
    pub transit_policy: TransitPolicy,
    pub transit_gtfs: TransitGtfsSystem,
    #[serde(default)]
    pub mp_config: MaxPressureConfig,
    #[serde(default)]
    pub evp_config: EvpConfig,
    #[serde(default)]
    pub emergency_facilities: Vec<EmergencyFacility>,
    pub idm: IdmParams,
    pub mobil: MobilParams,
    pub plume_physics: PlumePhysicsParams,
    pub base_demand_per_lane_hr: f64,
    pub heavy_share_pct: f64,
    pub max_vehicles: usize,
    pub watched_segment_id: usize,
    pub vehicles: Vec<Vehicle>,
    pub cyclists: Vec<CyclistAgent>,
    pub pedestrians: Vec<PedestrianAgent>,
}

impl ScenarioSnapshot {
    pub fn capture(
        world: &SimulationWorld,
        plume_physics: PlumePhysicsParams,
        title: impl Into<String>,
        include_agents: bool,
    ) -> Self {
        Self {
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            title: title.into(),
            sim_time_s: world.sim_time_s,
            includes_microscopic_agents: include_agents,
            network: world.network.clone(),
            parcels: world.parcels.clone(),
            transit_policy: world.transit_policy,
            transit_gtfs: world.transit_gtfs.clone(),
            mp_config: world.mp_config,
            evp_config: world.evp_config,
            emergency_facilities: world.emergency_facilities.clone(),
            idm: world.idm,
            mobil: world.mobil,
            plume_physics,
            base_demand_per_lane_hr: world.base_demand_per_lane_hr,
            heavy_share_pct: world.heavy_share_pct,
            max_vehicles: world.max_vehicles,
            watched_segment_id: world.watched_segment_id,
            vehicles: if include_agents {
                world.vehicles.clone()
            } else {
                Vec::new()
            },
            cyclists: if include_agents {
                world.cyclists.clone()
            } else {
                Vec::new()
            },
            pedestrians: if include_agents {
                world.pedestrians.clone()
            } else {
                Vec::new()
            },
        }
    }

    pub fn to_json_pretty(&self) -> Result<String, PersistenceError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json_str(json_str: &str) -> Result<Self, PersistenceError> {
        Ok(serde_json::from_str(json_str)?)
    }

    pub fn save_json_path(&self, path: &Path) -> Result<usize, PersistenceError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let body = self.to_json_pretty()?;
        let byte_len = body.len();
        fs::write(path, body)?;
        Ok(byte_len)
    }

    pub fn load_json_path(path: &Path) -> Result<Self, PersistenceError> {
        let text = fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    pub fn into_world_and_plume(self) -> (SimulationWorld, PlumePhysicsParams) {
        let plume = self.plume_physics;
        let world = SimulationWorld::from_snapshot(self);
        (world, plume)
    }
}

/// Single time-series telemetry observation sampled periodically (default 1.0 Hz) across all
/// traffic, multi-modal, GTFS transit, environmental, and municipal fiscal subsystems.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TelemetrySampleRow {
    pub sim_time_s: f64,
    pub active_agents_total: usize,
    pub vehicles_active: usize,
    pub vehicles_queued: usize,
    pub cyclists_active: usize,
    pub pedestrians_active: usize,
    pub mean_speed_mph: f64,
    pub throughput_veh_per_hr: f64,
    pub avg_trip_time_s: f64,
    pub mode_share_auto_pct: f64,
    pub mode_share_transit_pct: f64,
    pub mode_share_bike_pct: f64,
    pub mode_share_walk_pct: f64,
    pub gtfs_active_buses: usize,
    pub gtfs_onboard_pax: u32,
    pub gtfs_waiting_curb_pax: u32,
    pub gtfs_cumulative_boardings: u64,
    pub gtfs_mean_dwell_s: f64,
    pub gtfs_mean_load_factor_pct: f64,
    pub gtfs_bunching_pct: f64,
    pub gtfs_crowding_los: char,
    pub vsp_co2_kg_per_hr: f64,
    pub vsp_nox_g_per_hr: f64,
    pub vsp_pm25_g_per_hr: f64,
    pub corridor_noise_dba: f64,
    pub stormwater_runoff_lps: f64,
    pub access_15min_pct: f64,
    pub net_fiscal_per_acre_k_yr: f64,
    pub hcm_los: char,
}

pub const CSV_HEADER_COLUMNS: &str = "sim_time_s,active_agents_total,vehicles_active,vehicles_queued,cyclists_active,pedestrians_active,mean_speed_mph,throughput_veh_per_hr,avg_trip_time_s,mode_share_auto_pct,mode_share_transit_pct,mode_share_bike_pct,mode_share_walk_pct,gtfs_active_buses,gtfs_onboard_pax,gtfs_waiting_curb_pax,gtfs_cumulative_boardings,gtfs_mean_dwell_s,gtfs_mean_load_factor_pct,gtfs_bunching_pct,gtfs_crowding_los,vsp_co2_kg_per_hr,vsp_nox_g_per_hr,vsp_pm25_g_per_hr,corridor_noise_dba,stormwater_runoff_lps,access_15min_pct,net_fiscal_per_acre_k_yr,hcm_los";

/// Bounded in-memory time-series recorder with RFC 4180 `.csv` export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryTimeSeriesRecorder {
    pub sample_interval_s: f64,
    pub max_rows: usize,
    pub rows: VecDeque<TelemetrySampleRow>,
    accum_s: f64,
}

impl Default for TelemetryTimeSeriesRecorder {
    fn default() -> Self {
        Self::new(1.0, 14_400)
    }
}

impl TelemetryTimeSeriesRecorder {
    pub fn new(sample_interval_s: f64, max_rows: usize) -> Self {
        Self {
            sample_interval_s: sample_interval_s.max(0.1),
            max_rows: max_rows.max(60),
            rows: VecDeque::with_capacity(1024),
            accum_s: 0.0,
        }
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.accum_s = 0.0;
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn maybe_record(&mut self, dt: f64, world: &SimulationWorld) {
        self.accum_s += dt;
        if self.accum_s < self.sample_interval_s {
            return;
        }
        self.accum_s %= self.sample_interval_s;
        self.push_snapshot_now(world);
    }

    pub fn push_snapshot_now(&mut self, world: &SimulationWorld) {
        let ms = world.mode_shares;
        let gtfs = world.transit_gtfs.telemetry;
        let env = world.env_snapshot;
        let row = TelemetrySampleRow {
            sim_time_s: world.sim_time_s,
            active_agents_total: world.total_active_agents(),
            vehicles_active: world.vehicles.len(),
            vehicles_queued: world.queued_vehicles(),
            cyclists_active: world.cyclists.len(),
            pedestrians_active: world.pedestrians.len(),
            mean_speed_mph: world.mean_speed_mph(),
            throughput_veh_per_hr: world.throughput_veh_per_hr(),
            avg_trip_time_s: world.avg_trip_time_s(),
            mode_share_auto_pct: ms.auto * 100.0,
            mode_share_transit_pct: ms.transit * 100.0,
            mode_share_bike_pct: ms.bike * 100.0,
            mode_share_walk_pct: ms.walk * 100.0,
            gtfs_active_buses: gtfs.active_scheduled_buses,
            gtfs_onboard_pax: gtfs.onboard_passengers,
            gtfs_waiting_curb_pax: gtfs.waiting_passengers,
            gtfs_cumulative_boardings: gtfs.cumulative_boardings,
            gtfs_mean_dwell_s: gtfs.mean_dwell_s,
            gtfs_mean_load_factor_pct: gtfs.mean_load_factor_pct,
            gtfs_bunching_pct: gtfs.bunching_pct,
            gtfs_crowding_los: gtfs.tcqsm_crowding_los,
            vsp_co2_kg_per_hr: env.co2_kg_per_hr,
            vsp_nox_g_per_hr: env.nox_g_per_hr,
            vsp_pm25_g_per_hr: env.pm25_g_per_hr,
            corridor_noise_dba: env.corridor_noise_dba,
            stormwater_runoff_lps: env.stormwater_runoff_lps,
            access_15min_pct: env.access_15min_pct,
            net_fiscal_per_acre_k_yr: world.ledger.net_fiscal_per_acre_k_yr,
            hcm_los: world.ledger.hcm_los,
        };
        if self.rows.len() >= self.max_rows {
            self.rows.pop_front();
        }
        self.rows.push_back(row);
    }

    /// Formats all recorded observations into an RFC 4180 compliant UTF-8 CSV string (`\r\n` rows).
    pub fn to_csv_string(&self) -> String {
        let mut out = String::with_capacity((self.rows.len() + 1) * 192);
        out.push_str(CSV_HEADER_COLUMNS);
        out.push_str("\r\n");
        for r in &self.rows {
            let _ = write!(
                out,
                "{:.2},{},{},{},{},{},{:.2},{:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{},{},{:.2},{:.2},{:.2},{},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1},{:.2},{}\r\n",
                r.sim_time_s,
                r.active_agents_total,
                r.vehicles_active,
                r.vehicles_queued,
                r.cyclists_active,
                r.pedestrians_active,
                r.mean_speed_mph,
                r.throughput_veh_per_hr,
                r.avg_trip_time_s,
                r.mode_share_auto_pct,
                r.mode_share_transit_pct,
                r.mode_share_bike_pct,
                r.mode_share_walk_pct,
                r.gtfs_active_buses,
                r.gtfs_onboard_pax,
                r.gtfs_waiting_curb_pax,
                r.gtfs_cumulative_boardings,
                r.gtfs_mean_dwell_s,
                r.gtfs_mean_load_factor_pct,
                r.gtfs_bunching_pct,
                r.gtfs_crowding_los,
                r.vsp_co2_kg_per_hr,
                r.vsp_nox_g_per_hr,
                r.vsp_pm25_g_per_hr,
                r.corridor_noise_dba,
                r.stormwater_runoff_lps,
                r.access_15min_pct,
                r.net_fiscal_per_acre_k_yr,
                r.hcm_los,
            );
        }
        out
    }

    pub fn export_csv_path(&self, path: &Path) -> Result<usize, PersistenceError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let csv_body = self.to_csv_string();
        fs::write(path, csv_body)?;
        Ok(self.rows.len())
    }
}

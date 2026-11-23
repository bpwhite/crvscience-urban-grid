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

use crate::economics::{evaluate_municipal_ledger, MunicipalLedger, ScenarioSummaryReport};
use crate::emergency::{
    evaluate_emergency_coverage, step_emergency_preemption, synthesize_default_facilities,
    EmergencyCoverageSummary, EmergencyFacility, EmergencyFacilityKind, EvpConfig, EvpTelemetry,
};
use crate::environment::{evaluate_environment_and_isochrones, EnvironmentalSnapshot};
use crate::max_pressure::{step_max_pressure_controllers, MaxPressureConfig, MaxPressureTelemetry};
use crate::multimodal::{evaluate_mode_split, ModeShares, TransitPolicy};
use crate::network::{ControlMode, Lane, LightColor, RoadNetwork, TurnType, Vec2, LANE_WIDTH_M};
use crate::gpu_pack::PlumePhysicsParams;
use crate::history::EditHistoryStack;
use crate::persistence::{PersistenceError, ScenarioSnapshot, TelemetryTimeSeriesRecorder};
use crate::transit_gtfs::{PendingBusDispatch, TransitGtfsSystem};
use crate::utilities::{
    evaluate_coupled_utility_grids, UtilityGridsConfig, UtilityGridsReport,
};
use crate::zoning::{generate_parcels_for_network, Parcel};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VehicleClass {
    Sedan,
    TransitBus,
    FreightTruck,
    EmergencyResponder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CarrierRef {
    Lane(usize),
    Connector(usize),
}

/// Intelligent Driver Model (IDM) parameters — Treiber, Hennecke & Helbing (2000).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IdmParams {
    pub time_headway_s: f64,
    pub min_jam_gap_m: f64,
    pub max_accel_mps2: f64,
    pub comfort_decel_mps2: f64,
    pub hard_decel_mps2: f64,
}

impl Default for IdmParams {
    fn default() -> Self {
        Self {
            time_headway_s: 1.35,
            min_jam_gap_m: 2.0,
            max_accel_mps2: 1.6,
            comfort_decel_mps2: 2.2,
            hard_decel_mps2: 8.5,
        }
    }
}

/// General Lane-Changing Model MOBIL — Kesting, Treiber & Helbing (2007).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MobilParams {
    pub politeness: f64,
    pub threshold_mps2: f64,
    pub safe_decel_mps2: f64,
}

impl Default for MobilParams {
    fn default() -> Self {
        Self {
            politeness: 0.30,
            threshold_mps2: 0.20,
            safe_decel_mps2: 4.0,
        }
    }
}

#[inline]
pub fn idm_acceleration(
    v: f64,
    v0: f64,
    gap: f64,
    dv: f64,
    a_max: f64,
    idm: &IdmParams,
) -> f64 {
    let eff_v0 = v0.max(1.0);
    let free_term = 1.0 - (v / eff_v0).powi(4);
    if !gap.is_finite() || gap > 220.0 {
        return (a_max * free_term).clamp(-idm.hard_decel_mps2, a_max);
    }
    let eff_gap = gap.max(0.15);
    let s_star = idm.min_jam_gap_m
        + (v * idm.time_headway_s
            + (v * dv) / (2.0 * (a_max * idm.comfort_decel_mps2).sqrt()))
        .max(0.0);
    let acc = a_max * (free_term - (s_star / eff_gap).powi(2));
    acc.clamp(-idm.hard_decel_mps2, a_max)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vehicle {
    pub id: u64,
    pub v_class: VehicleClass,
    pub length_m: f64,
    pub width_m: f64,
    pub accel_mult: f64,
    pub v_max_mps: f64,
    pub carrier: CarrierRef,
    pub s_m: f64,
    pub v: f64,
    pub acc: f64,
    pub lat_visual_offset_m: f64,
    pub lc_cooldown_s: f64,
    pub dest_seg_id: Option<usize>,
    pub planned_conn_id: Option<usize>,
    pub turn_intent: TurnType,
    pub spawn_time_s: f64,
    /// Optional fixed-route GTFS assignment (`TransitGtfsSystem::routes[idx]`).
    pub gtfs_route_idx: Option<u16>,
    pub gtfs_route_seq: u16,
    pub gtfs_next_stop_order: u16,
    pub dwell_remaining_s: f32,
    pub onboard_pax: u16,
    pub crush_capacity: u16,
}

impl Vehicle {
    #[inline]
    pub fn current_segment_id(&self, lanes: &[Lane]) -> Option<usize> {
        match self.carrier {
            CarrierRef::Lane(lid) => Some(lanes[lid].segment_id),
            CarrierRef::Connector(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CyclistAgent {
    pub segment_id: usize,
    pub s_m: f64,
    pub v_mps: f64,
    pub lateral_jitter_m: f32,
}

/// Microscopic sidewalk & signalized crosswalk pedestrian agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PedestrianAgent {
    pub segment_id: usize,
    pub side_sign: i8,
    pub walk_dir: i8,
    pub lateral_jitter_m: f32,
    pub s_m: f64,
    pub v_mps: f64,
    pub cur_speed_mps: f64,
    pub phase_seed: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceTimeColumn {
    pub sim_time_s: f64,
    pub segment_id: usize,
    pub samples: Vec<(f32, f32)>,
}

#[derive(Clone)]
pub struct SimulationWorld {
    pub network: RoadNetwork,
    pub parcels: Vec<Parcel>,
    pub transit_policy: TransitPolicy,
    pub transit_gtfs: TransitGtfsSystem,
    pub mp_config: MaxPressureConfig,
    pub mp_telemetry: MaxPressureTelemetry,
    pub evp_config: EvpConfig,
    pub evp_telemetry: EvpTelemetry,
    pub emergency_facilities: Vec<EmergencyFacility>,
    pub emergency_coverage: EmergencyCoverageSummary,
    pub segment_fire_travel_s: Vec<f64>,
    pub segment_hospital_travel_s: Vec<f64>,
    pub evp_corridor_segments: Vec<bool>,
    pub siren_seg_s: Vec<Vec<f64>>,
    pub idm: IdmParams,
    pub mobil: MobilParams,
    pub base_demand_per_lane_hr: f64,
    pub effective_demand_per_lane_hr: f64,
    pub heavy_share_pct: f64,
    pub max_vehicles: usize,
    pub sim_time_s: f64,
    pub vehicles: Vec<Vehicle>,
    pub cyclists: Vec<CyclistAgent>,
    pub pedestrians: Vec<PedestrianAgent>,
    pub completed_trips: VecDeque<(f64, f64)>,
    pub xt_history: VecDeque<SpaceTimeColumn>,
    pub watched_segment_id: usize,
    pub mode_shares: ModeShares,
    pub env_snapshot: EnvironmentalSnapshot,
    pub ledger: MunicipalLedger,
    pub telemetry_log: TelemetryTimeSeriesRecorder,
    pub edit_history: EditHistoryStack,
    /// Simulated 24-hour clock (`0.0 .. 24.0`, default `8.0` AM) for diurnal demand & Peak/Off-Peak dispatch tables.
    pub diurnal_clock_hour: f64,
    /// When enabled, `step(dt)` advances `diurnal_clock_hour`, modulates vehicle & emergency call rates
    /// along the 24-hour USFA/BPR profile, and auto-selects Peak vs. Off-Peak parcel dispatch tables!
    pub diurnal_live_clock_enabled: bool,
    pub utility_config: UtilityGridsConfig,
    pub utility_report: UtilityGridsReport,
    next_veh_id: u64,
    spawn_accum: f64,
    evp_spawn_accum_s: f64,
    completed_ev_trips: u64,
    yielding_vehicles_count: usize,
    route_timer_s: f64,
    mobil_timer_s: f64,
    xt_timer_s: f64,
    rng_state: u64,
}

impl SimulationWorld {
    pub fn new(network: RoadNetwork) -> Self {
        let parcels = generate_parcels_for_network(&network);
        let watched = network
            .segments
            .iter()
            .max_by_key(|s| (s.lane_count * 1000) + s.length as usize)
            .map(|s| s.id)
            .unwrap_or(0);

        let transit_policy = TransitPolicy::default();
        let mode_shares = evaluate_mode_split(&network, &parcels, &transit_policy, 9.0);
        let transit_gtfs =
            TransitGtfsSystem::synthesize_for_network(&network, &parcels, transit_policy.headway_min);

        let mut net_clone = network;
        let mut parcels_clone = parcels;
        let anchor = net_clone
            .junctions
            .first()
            .map(|j| j.pos)
            .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        let env_snapshot = evaluate_environment_and_isochrones(
            &mut net_clone,
            &mut parcels_clone,
            &[],
            &transit_policy,
            anchor,
        );
        let ledger = evaluate_municipal_ledger(&net_clone, &parcels_clone, &transit_policy, 20.0);

        let init_facilities = synthesize_default_facilities(&net_clone);
        let n_segs = net_clone.segments.len();
        let default_max_veh = (net_clone.lanes.len() * 18).clamp(600, 45_000);
        let default_peds = (n_segs * 14).clamp(180, 18_000);

        let mut world = Self {
            network: net_clone,
            parcels: parcels_clone,
            transit_policy,
            transit_gtfs,
            mp_config: MaxPressureConfig::default(),
            mp_telemetry: MaxPressureTelemetry::default(),
            evp_config: EvpConfig::default(),
            evp_telemetry: EvpTelemetry::default(),
            emergency_facilities: init_facilities,
            emergency_coverage: EmergencyCoverageSummary::default(),
            segment_fire_travel_s: vec![120.0; n_segs],
            segment_hospital_travel_s: vec![180.0; n_segs],
            evp_corridor_segments: vec![false; n_segs],
            siren_seg_s: vec![Vec::new(); n_segs],
            idm: IdmParams::default(),
            mobil: MobilParams::default(),
            base_demand_per_lane_hr: 340.0,
            effective_demand_per_lane_hr: 340.0,
            heavy_share_pct: 16.0,
            max_vehicles: default_max_veh,
            sim_time_s: 0.0,
            vehicles: Vec::new(),
            cyclists: Vec::new(),
            pedestrians: Vec::new(),
            completed_trips: VecDeque::new(),
            xt_history: VecDeque::with_capacity(240),
            watched_segment_id: watched,
            mode_shares,
            env_snapshot,
            ledger,
            telemetry_log: TelemetryTimeSeriesRecorder::default(),
            edit_history: EditHistoryStack::default(),
            diurnal_clock_hour: 8.0,
            diurnal_live_clock_enabled: false,
            utility_config: UtilityGridsConfig::default(),
            utility_report: UtilityGridsReport::default(),
            next_veh_id: 1,
            spawn_accum: 0.0,
            evp_spawn_accum_s: 0.0,
            completed_ev_trips: 0,
            yielding_vehicles_count: 0,
            route_timer_s: 0.0,
            mobil_timer_s: 0.0,
            xt_timer_s: 0.0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        };
        world.rebuild_cyclists();
        world.rebuild_pedestrians(default_peds);
        world.refresh_urban_metrics();
        let mut telemetry_log = std::mem::take(&mut world.telemetry_log);
        telemetry_log.push_snapshot_now(&world);
        world.telemetry_log = telemetry_log;
        world
    }

    pub fn from_snapshot(mut snap: ScenarioSnapshot) -> Self {
        snap.network.recompute_dynamic_routes();
        let next_id = snap
            .vehicles
            .iter()
            .map(|v| v.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let default_peds = (snap.network.segments.len() * 14).clamp(180, 18_000);

        let mode_shares = evaluate_mode_split(
            &snap.network,
            &snap.parcels,
            &snap.transit_policy,
            8.5,
        );
        let anchor = snap
            .network
            .junctions
            .first()
            .map(|j| j.pos)
            .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        let env_snapshot = evaluate_environment_and_isochrones(
            &mut snap.network,
            &mut snap.parcels,
            &snap.vehicles,
            &snap.transit_policy,
            anchor,
        );
        let ledger = evaluate_municipal_ledger(
            &snap.network,
            &snap.parcels,
            &snap.transit_policy,
            19.0,
        );

        let n_segs = snap.network.segments.len();
        let emergency_facilities = if snap.emergency_facilities.is_empty() {
            synthesize_default_facilities(&snap.network)
        } else {
            snap.emergency_facilities
        };
        let mut world = Self {
            network: snap.network,
            parcels: snap.parcels,
            transit_policy: snap.transit_policy,
            transit_gtfs: snap.transit_gtfs,
            evp_corridor_segments: vec![false; n_segs],
            siren_seg_s: vec![Vec::new(); n_segs],
            mp_config: snap.mp_config,
            mp_telemetry: MaxPressureTelemetry::default(),
            segment_fire_travel_s: vec![120.0; n_segs],
            segment_hospital_travel_s: vec![180.0; n_segs],
            emergency_facilities,
            emergency_coverage: EmergencyCoverageSummary::default(),
            evp_config: snap.evp_config,
            evp_telemetry: EvpTelemetry::default(),
            idm: snap.idm,
            mobil: snap.mobil,
            base_demand_per_lane_hr: snap.base_demand_per_lane_hr,
            effective_demand_per_lane_hr: snap.base_demand_per_lane_hr,
            heavy_share_pct: snap.heavy_share_pct,
            max_vehicles: snap.max_vehicles,
            sim_time_s: snap.sim_time_s,
            vehicles: snap.vehicles,
            cyclists: snap.cyclists,
            pedestrians: snap.pedestrians,
            completed_trips: VecDeque::new(),
            xt_history: VecDeque::with_capacity(240),
            watched_segment_id: snap.watched_segment_id,
            mode_shares,
            env_snapshot,
            ledger,
            telemetry_log: TelemetryTimeSeriesRecorder::default(),
            edit_history: EditHistoryStack::default(),
            diurnal_clock_hour: 8.0,
            diurnal_live_clock_enabled: false,
            utility_config: UtilityGridsConfig::default(),
            utility_report: UtilityGridsReport::default(),
            next_veh_id: next_id,
            spawn_accum: 0.0,
            evp_spawn_accum_s: 0.0,
            completed_ev_trips: 0,
            yielding_vehicles_count: 0,
            route_timer_s: 0.0,
            mobil_timer_s: 0.0,
            xt_timer_s: 0.0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        };
        if !snap.includes_microscopic_agents {
            world.rebuild_cyclists();
            world.rebuild_pedestrians(default_peds);
            if world.vehicles.is_empty() {
                world.seed_initial_traffic(280);
            }
        }
        world.refresh_urban_metrics();
        let mut telemetry_log = std::mem::take(&mut world.telemetry_log);
        telemetry_log.push_snapshot_now(&world);
        world.telemetry_log = telemetry_log;
        world
    }

    pub fn save_snapshot_json(
        &self,
        path: &Path,
        plume_physics: PlumePhysicsParams,
        title: impl Into<String>,
        include_agents: bool,
    ) -> Result<usize, PersistenceError> {
        let snap = ScenarioSnapshot::capture(self, plume_physics, title, include_agents);
        snap.save_json_path(path)
    }

    pub fn load_snapshot_json(
        path: &Path,
    ) -> Result<(Self, PlumePhysicsParams), PersistenceError> {
        let snap = ScenarioSnapshot::load_json_path(path)?;
        Ok(snap.into_world_and_plume())
    }

    pub fn export_telemetry_csv(&self, path: &Path) -> Result<usize, PersistenceError> {
        self.telemetry_log.export_csv_path(path)
    }

    pub fn load_gtfs_directory(&mut self, dir: &Path) -> Result<(), String> {
        let sys = TransitGtfsSystem::load_gtfs_dir(
            dir,
            &self.network,
            &self.parcels,
            self.transit_policy.headway_min,
        )
        .map_err(|e| e.to_string())?;
        self.transit_gtfs = sys;
        Ok(())
    }

    #[inline]
    pub fn total_active_agents(&self) -> usize {
        self.vehicles.len() + self.cyclists.len() + self.pedestrians.len()
    }

    fn next_rand_f64(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64)
    }

    pub fn rebuild_cyclists(&mut self) {
        self.cyclists.clear();
        for seg in &self.network.segments {
            if seg.has_bike_lane && !seg.closed {
                let n = ((seg.length / 42.0).round() as usize).max(1);
                for i in 0..n {
                    let jitter = (((seg.id * 31 + i * 17) % 100) as f32 / 100.0 - 0.5) * 0.55;
                    self.cyclists.push(CyclistAgent {
                        segment_id: seg.id,
                        s_m: (seg.length * (i as f64 + 0.3)) / n as f64,
                        v_mps: 4.8 + (i % 4) as f64 * 0.62,
                        lateral_jitter_m: jitter,
                    });
                }
            }
        }
    }

    pub fn populate_cyclists_target(&mut self, target_count: usize) {
        self.cyclists.clear();
        let open_segs: Vec<usize> = self
            .network
            .segments
            .iter()
            .filter(|s| !s.closed)
            .map(|s| s.id)
            .collect();
        if open_segs.is_empty() || target_count == 0 {
            return;
        }
        self.cyclists.reserve(target_count);
        for idx in 0..target_count {
            let sid = open_segs[idx % open_segs.len()];
            let len = self.network.segments[sid].length.max(8.0);
            let frac = self.next_rand_f64();
            let jitter = (self.next_rand_f64() as f32 - 0.5) * 0.85;
            let v_mps = 4.2 + self.next_rand_f64() * 2.6;
            self.cyclists.push(CyclistAgent {
                segment_id: sid,
                s_m: frac * len,
                v_mps,
                lateral_jitter_m: jitter,
            });
        }
    }

    pub fn rebuild_pedestrians(&mut self, target_count: usize) {
        self.pedestrians.clear();
        let open_segs: Vec<usize> = self
            .network
            .segments
            .iter()
            .filter(|s| !s.closed)
            .map(|s| s.id)
            .collect();
        if open_segs.is_empty() || target_count == 0 {
            return;
        }
        self.pedestrians.reserve(target_count);
        for idx in 0..target_count {
            let sid = open_segs[idx % open_segs.len()];
            let len = self.network.segments[sid].length.max(8.0);
            let side_sign = if idx % 2 == 0 { 1i8 } else { -1i8 };
            let walk_dir = if (idx / 2) % 2 == 0 { 1i8 } else { -1i8 };
            let v_mps = 1.15 + self.next_rand_f64() * 0.58;
            let lateral_jitter_m = (self.next_rand_f64() as f32 - 0.5) * 1.35;
            let s_m = self.next_rand_f64() * len;
            self.pedestrians.push(PedestrianAgent {
                segment_id: sid,
                side_sign,
                walk_dir,
                lateral_jitter_m,
                s_m,
                v_mps,
                cur_speed_mps: v_mps,
                phase_seed: (idx & 0xFFFF) as u16,
            });
        }
    }

    /// Seeds an initial stationary snapshot of vehicles, scheduled GTFS buses, cyclists, and
    /// pedestrians at `sim_time_s = 0.0` without advancing the simulation clock.
    pub fn seed_initial_traffic(&mut self, target_total: usize) {
        self.seed_crowd_internal(target_total.max(120), false);
    }

    pub fn seed_50k_megacity_crowd(&mut self, target_total: usize) {
        self.seed_crowd_internal(target_total.max(1_000), true);
    }

    fn seed_crowd_internal(&mut self, total: usize, stress_50k: bool) {
        let target_veh = ((total as f64) * 0.46).round() as usize;
        let target_cyc = ((total as f64) * 0.20).round() as usize;
        let target_ped = total.saturating_sub(target_veh + target_cyc);

        if stress_50k {
            for (i, seg) in self.network.segments.iter_mut().enumerate() {
                if i % 2 == 0 {
                    seg.has_bike_lane = true;
                }
            }
            self.max_vehicles = (target_veh + 6_000).max(50_000);
            self.base_demand_per_lane_hr = 780.0;
        }

        self.vehicles.clear();

        let mut open_lanes: Vec<usize> = self
            .network
            .lanes
            .iter()
            .filter(|l| !self.network.segments[l.segment_id].closed && l.length >= 14.0)
            .map(|l| l.id)
            .collect();
        if !stress_50k && open_lanes.len() > target_veh && target_veh > 0 {
            let stride = (open_lanes.len() / target_veh).max(1);
            open_lanes = open_lanes.into_iter().step_by(stride).collect();
        }

        if !open_lanes.is_empty() {
            let mut sorted_sinks: Vec<usize> = self.network.sinks.iter().copied().collect();
            sorted_sinks.sort_unstable();

            let per_lane = (target_veh.div_ceil(open_lanes.len())).max(1);
            self.vehicles.reserve(target_veh);

            'outer: for &lid in &open_lanes {
                let lane_len = self.network.lanes[lid].length;
                let seg_id = self.network.lanes[lid].segment_id;
                let seg_speed = self.network.segments[seg_id].speed_mps;
                let max_fit = ((lane_len - 4.0) / 7.6).floor().max(1.0) as usize;
                let count_here = per_lane.min(max_fit);
                let spacing = (lane_len - 4.0) / (count_here as f64);

                for slot in 0..count_here {
                    if self.vehicles.len() >= target_veh {
                        break 'outer;
                    }
                    let s_m = 2.5 + (slot as f64 + 0.35) * spacing;
                    let roll = self.next_rand_f64() * 100.0;
                    let (v_class, length_m, width_m, accel_mult, v_max_mps, pax, crush) =
                        if roll < 5.0 && spacing >= 14.0 {
                            (VehicleClass::TransitBus, 12.2, 2.55, 0.68, 22.0, 26u16, 60u16)
                        } else if roll < 14.0 && spacing >= 9.6 {
                            (VehicleClass::FreightTruck, 7.8, 2.35, 0.74, 26.0, 0u16, 0u16)
                        } else {
                            (VehicleClass::Sedan, 4.4, 1.9, 1.0, 36.0, 0u16, 0u16)
                        };
                    let dest_seg_id = if sorted_sinks.is_empty() {
                        None
                    } else {
                        let pick = sorted_sinks[(self.next_veh_id as usize) % sorted_sinks.len()];
                        if pick != seg_id {
                            Some(pick)
                        } else {
                            None
                        }
                    };
                    let v_init =
                        (seg_speed * (0.55 + 0.35 * self.next_rand_f64())).min(spacing * 0.75);
                    let id = self.next_veh_id;
                    self.next_veh_id += 1;
                    let lc_cooldown_s = self.next_rand_f64() * 2.0;
                    self.vehicles.push(Vehicle {
                        id,
                        v_class,
                        length_m,
                        width_m,
                        accel_mult,
                        v_max_mps,
                        carrier: CarrierRef::Lane(lid),
                        s_m,
                        v: v_init,
                        acc: 0.0,
                        lat_visual_offset_m: 0.0,
                        lc_cooldown_s,
                        dest_seg_id,
                        planned_conn_id: None,
                        turn_intent: TurnType::Through,
                        spawn_time_s: self.sim_time_s,
                        gtfs_route_idx: None,
                        gtfs_route_seq: 0,
                        gtfs_next_stop_order: 0,
                        dwell_remaining_s: 0.0,
                        onboard_pax: pax,
                        crush_capacity: crush,
                    });
                }
            }

            let shortfall = target_veh.saturating_sub(self.vehicles.len());
            self.populate_cyclists_target(target_cyc + shortfall / 2);
            self.rebuild_pedestrians(target_ped + (shortfall - shortfall / 2));
        } else {
            self.populate_cyclists_target(target_cyc);
            self.rebuild_pedestrians(target_ped);
        }

        // Also seed active scheduled GTFS buses along each fixed route immediately
        for r_idx in 0..self.transit_gtfs.routes.len() {
            let chain_len = self.transit_gtfs.routes[r_idx].segment_chain.len();
            if chain_len == 0 {
                continue;
            }
            let (len_m, wid_m, seated, crush) = self.transit_gtfs.coach_spec.dimensions_and_capacity();
            for k in 0..2 {
                let seq = (k * (chain_len / 2)).min(chain_len - 1);
                let sid = self.transit_gtfs.routes[r_idx].segment_chain[seq];
                if self.network.segments[sid].closed {
                    continue;
                }
                if let Some(&curb_lid) = self.network.segments[sid].lane_ids.last() {
                    let bus_s = len_m + 4.0;
                    let lane_len = self.network.lanes[curb_lid].length;
                    let has_room = lane_len > bus_s + 2.0
                        && self.vehicles.iter().all(|v| {
                            v.carrier != CarrierRef::Lane(curb_lid)
                                || (bus_s + 1.5 <= v.s_m - v.length_m)
                                || (v.s_m + 1.5 <= bus_s - len_m)
                        });
                    if !has_room {
                        continue;
                    }
                    let next_stop_order = self.transit_gtfs.routes[r_idx]
                        .route_stops
                        .iter()
                        .position(|&(_, c_idx)| c_idx >= seq)
                        .unwrap_or(self.transit_gtfs.routes[r_idx].route_stops.len())
                        as u16;
                    let id = self.next_veh_id;
                    self.next_veh_id += 1;
                    self.vehicles.push(Vehicle {
                        id,
                        v_class: VehicleClass::TransitBus,
                        length_m: len_m,
                        width_m: wid_m,
                        accel_mult: 0.68,
                        v_max_mps: 21.0,
                        carrier: CarrierRef::Lane(curb_lid),
                        s_m: bus_s,
                        v: 7.5,
                        acc: 0.0,
                        lat_visual_offset_m: 0.0,
                        lc_cooldown_s: 3.0,
                        dest_seg_id: self.transit_gtfs.routes[r_idx].segment_chain.last().copied(),
                        planned_conn_id: None,
                        turn_intent: TurnType::Through,
                        spawn_time_s: self.sim_time_s,
                        gtfs_route_idx: Some(r_idx as u16),
                        gtfs_route_seq: seq as u16,
                        gtfs_next_stop_order: next_stop_order,
                        dwell_remaining_s: 0.0,
                        onboard_pax: (seated / 2) + (k as u16 * 7),
                        crush_capacity: crush,
                    });
                }
            }
        }

        self.refresh_urban_metrics();
    }

    pub fn refresh_urban_metrics(&mut self) {
        let mean_v_mps = if self.vehicles.is_empty() {
            8.5
        } else {
            self.vehicles.iter().map(|v| v.v).sum::<f64>() / self.vehicles.len() as f64
        };
        self.mode_shares = evaluate_mode_split(
            &self.network,
            &self.parcels,
            &self.transit_policy,
            mean_v_mps,
        );
        self.transit_gtfs.refresh_stop_catchments(&self.parcels);

        if self.transit_policy.couple_gravity_demand {
            let res: f64 = self.parcels.iter().map(|p| p.residents()).sum();
            let jobs: f64 = self.parcels.iter().map(|p| p.jobs()).sum();
            let grav_scale = (res.max(400.0) * jobs.max(400.0)).sqrt() / 2400.0;
            self.effective_demand_per_lane_hr = (self.base_demand_per_lane_hr
                * grav_scale
                * (self.mode_shares.auto / 0.64))
                .clamp(90.0, 1150.0);
        } else {
            self.effective_demand_per_lane_hr = self.base_demand_per_lane_hr;
        }

        let anchor = self
            .network
            .junctions
            .first()
            .map(|j| j.pos)
            .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        self.env_snapshot = evaluate_environment_and_isochrones(
            &mut self.network,
            &mut self.parcels,
            &self.vehicles,
            &self.transit_policy,
            anchor,
        );
        self.ledger = evaluate_municipal_ledger(
            &self.network,
            &self.parcels,
            &self.transit_policy,
            mean_v_mps * 2.236_94,
        );
        let (cov_sum, seg_f, seg_h) = evaluate_emergency_coverage(
            &self.network,
            &mut self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
        );
        self.emergency_coverage = cov_sum;
        self.segment_fire_travel_s = seg_f;
        self.segment_hospital_travel_s = seg_h;
        self.utility_report = evaluate_coupled_utility_grids(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            self.diurnal_clock_hour,
            &self.utility_config,
        );
    }

    pub fn set_rng_seed(&mut self, seed: u64) {
        self.rng_state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    }

    pub fn mean_speed_mph(&self) -> f64 {
        if self.vehicles.is_empty() {
            0.0
        } else {
            (self.vehicles.iter().map(|v| v.v).sum::<f64>() / self.vehicles.len() as f64) * 2.236_94
        }
    }

    pub fn queued_vehicles(&self) -> usize {
        self.vehicles.iter().filter(|v| v.v * 2.236_94 < 2.2).count()
    }

    pub fn throughput_veh_per_hr(&self) -> f64 {
        let win = self.sim_time_s.clamp(10.0, 90.0);
        (self.completed_trips.len() as f64 / win) * 3600.0
    }

    pub fn avg_trip_time_s(&self) -> f64 {
        if self.completed_trips.is_empty() {
            0.0
        } else {
            self.completed_trips.iter().map(|(d, _)| *d).sum::<f64>()
                / self.completed_trips.len() as f64
        }
    }

    pub fn summary_report(&self, label: impl Into<String>) -> ScenarioSummaryReport {
        ScenarioSummaryReport {
            label: label.into(),
            mean_speed_mph: self.mean_speed_mph(),
            avg_trip_time_s: self.avg_trip_time_s(),
            throughput_veh_per_hr: self.throughput_veh_per_hr(),
            mode_shares: self.mode_shares,
            environment: self.env_snapshot,
            ledger: self.ledger.clone(),
            transit_telemetry: self.transit_gtfs.telemetry,
            max_pressure_telemetry: self.mp_telemetry,
            evp_telemetry: self.evp_telemetry,
            emergency_coverage: self.emergency_coverage,
        }
    }

    /// Dispatches an active siren `EmergencyResponder` on `preferred_seg` (or highest-priority
    /// open boundary source) routed toward a cross-town boundary sink.
    pub fn dispatch_emergency_on_segment(&mut self, preferred_seg: Option<usize>) -> Option<u64> {
        let start_sid = preferred_seg
            .filter(|&sid| sid < self.network.segments.len() && !self.network.segments[sid].closed && !self.network.segments[sid].lane_ids.is_empty())
            .or_else(|| {
                self.network
                    .sources
                    .iter()
                    .copied()
                    .filter(|&sid| !self.network.segments[sid].closed && !self.network.segments[sid].lane_ids.is_empty())
                    .max_by_key(|&sid| self.network.segments[sid].priority)
            })
            .or_else(|| {
                self.network
                    .segments
                    .iter()
                    .find(|s| !s.closed && !s.lane_ids.is_empty())
                    .map(|s| s.id)
            })?;
        let lid = self.network.segments[start_sid].lane_ids[0];
        let start_pos = self.network.segments[start_sid].base_pts[0];
        let hosp_pos_opt = self
            .emergency_facilities
            .iter()
            .filter(|f| f.kind == EmergencyFacilityKind::Hospital)
            .map(|f| f.world_pos)
            .find(|hp| hp.dist(start_pos) >= 85.0);
        let mut best_sink: Option<(usize, f64)> = None;
        for &sk in &self.network.sinks {
            if sk != start_sid && self.network.next_hop.contains_key(&(start_sid, sk)) {
                let sp = self.network.segments[sk].base_pts[0];
                let score = if let Some(hp) = hosp_pos_opt {
                    // Prefer reachable sink corridor passing closest to the placed Trauma Hospital!
                    100_000.0 - sp.dist(hp)
                } else {
                    start_pos.dist(sp)
                };
                if best_sink.map(|(_, bs)| score > bs).unwrap_or(true) {
                    best_sink = Some((sk, score));
                }
            }
        }
        let id = self.next_veh_id;
        self.next_veh_id += 1;
        let seg_v = self.network.segments[start_sid].speed_mps;
        self.vehicles.push(Vehicle {
            id,
            v_class: VehicleClass::EmergencyResponder,
            length_m: 6.4,
            width_m: 2.25,
            accel_mult: 1.45,
            v_max_mps: (seg_v * 1.42).clamp(18.0, 31.0),
            carrier: CarrierRef::Lane(lid),
            s_m: 2.0,
            v: (seg_v * 1.15).clamp(8.0, 22.0),
            acc: 0.0,
            lat_visual_offset_m: 0.0,
            lc_cooldown_s: 0.0,
            dest_seg_id: best_sink.map(|(sk, _)| sk),
            planned_conn_id: None,
            turn_intent: TurnType::Through,
            spawn_time_s: self.sim_time_s,
            gtfs_route_idx: None,
            gtfs_route_seq: 0,
            gtfs_next_stop_order: 0,
            dwell_remaining_s: 0.0,
            onboard_pax: 0,
            crush_capacity: 0,
        });
        Some(id)
    }

    pub fn dispatch_emergency_responder(&mut self) -> Option<u64> {
        let station_sids: Vec<(usize, f64)> = self
            .emergency_facilities
            .iter()
            .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
            .chain(
                self.emergency_facilities
                    .iter()
                    .filter(|f| f.kind == EmergencyFacilityKind::Hospital),
            )
            .filter(|f| {
                f.segment_id < self.network.segments.len()
                    && !self.network.segments[f.segment_id].closed
                    && !self.network.segments[f.segment_id].lane_ids.is_empty()
            })
            .map(|f| (f.segment_id, f.s_m))
            .collect();

        if !station_sids.is_empty() {
            // Sample an incident parcel weighted by emergency call demand and prefer its assigned
            // Carter-Chaiken-Ignall / Larson primary district firehouse if currently idle!
            let mut chosen_pair: Option<(usize, f64)> = None;
            if !self.parcels.is_empty() {
                let tot_w: f64 = self
                    .parcels
                    .iter()
                    .map(|p| (p.residents() + 0.35 * p.jobs()).max(1.0))
                    .sum();
                let mut draw = self.next_rand_f64() * tot_w.max(1.0);
                let mut call_parcel_idx = 0usize;
                for (pi, p) in self.parcels.iter().enumerate() {
                    draw -= (p.residents() + 0.35 * p.jobs()).max(1.0);
                    if draw <= 0.0 {
                        call_parcel_idx = pi;
                        break;
                    }
                }
                let cur_hr = (self.diurnal_clock_hour.floor() as isize).rem_euclid(24) as u8;
                let is_peak_now = (7..10).contains(&cur_hr) || (15..20).contains(&cur_hr);
                let p_ref = &self.parcels[call_parcel_idx];
                let regime_seg = if self.diurnal_live_clock_enabled {
                    if is_peak_now {
                        p_ref.peak_district_station_seg.or(p_ref.response_district_station_seg)
                    } else {
                        p_ref.offpeak_district_station_seg.or(p_ref.response_district_station_seg)
                    }
                } else {
                    p_ref.response_district_station_seg
                };
                if let Some(pref_seg) = regime_seg.map(|s| s as usize)
                {
                    // Check that `pref_seg` is open and does not already have an outgoing responder right on its apron
                    let busy_on_apron = self.vehicles.iter().any(|v| {
                        v.v_class == VehicleClass::EmergencyResponder
                            && matches!(v.carrier, CarrierRef::Lane(lid) if self.network.lanes[lid].segment_id == pref_seg)
                    });
                    if !busy_on_apron {
                        chosen_pair = station_sids.iter().copied().find(|&(sid, _)| sid == pref_seg);
                    }
                }
            }
            let (sid, station_s) = chosen_pair.unwrap_or_else(|| {
                let pick = (self.next_rand_f64() * station_sids.len() as f64) as usize
                    % station_sids.len();
                station_sids[pick]
            });
            if let Some(ev_id) = self.dispatch_emergency_on_segment(Some(sid)) {
                if let Some(last_v) = self.vehicles.last_mut() {
                    if last_v.id == ev_id {
                        let llen = self.network.segments[sid].length;
                        last_v.s_m = station_s.clamp(1.5, (llen - 2.0).max(1.5));
                    }
                }
                return Some(ev_id);
            }
        }

        let n_src = self.network.sources.len();
        let chosen = if n_src > 0 {
            let idx = (self.next_rand_f64() * n_src as f64) as usize % n_src;
            Some(self.network.sources[idx])
        } else {
            None
        };
        self.dispatch_emergency_on_segment(chosen)
    }

    fn spawn_scheduled_gtfs_buses(&mut self, pending: Vec<PendingBusDispatch>) {
        if pending.is_empty() {
            return;
        }
        for disp in pending {
            if self.network.segments[disp.start_segment_id].closed {
                continue;
            }
            let Some(&curb_lid) = self.network.segments[disp.start_segment_id].lane_ids.last() else {
                continue;
            };
            let clear = !self.vehicles.iter().any(|v| {
                v.carrier == CarrierRef::Lane(curb_lid)
                    && (v.s_m - v.length_m) < (self.idm.min_jam_gap_m + 5.5)
            });
            if !clear {
                continue;
            }
            let dest_seg = self.transit_gtfs.routes[disp.route_idx]
                .segment_chain
                .last()
                .copied();
            let v_init = (self.network.segments[disp.start_segment_id].speed_mps * 0.78).max(3.0);
            let id = self.next_veh_id;
            self.next_veh_id += 1;
            self.vehicles.push(Vehicle {
                id,
                v_class: VehicleClass::TransitBus,
                length_m: disp.length_m,
                width_m: disp.width_m,
                accel_mult: 0.68,
                v_max_mps: 21.5,
                carrier: CarrierRef::Lane(curb_lid),
                s_m: 0.6,
                v: v_init,
                acc: 0.0,
                lat_visual_offset_m: 0.0,
                lc_cooldown_s: 3.0,
                dest_seg_id: dest_seg,
                planned_conn_id: None,
                turn_intent: TurnType::Through,
                spawn_time_s: self.sim_time_s,
                gtfs_route_idx: Some(disp.route_idx as u16),
                gtfs_route_seq: 0,
                gtfs_next_stop_order: 0,
                dwell_remaining_s: 0.0,
                onboard_pax: disp.initial_pax,
                crush_capacity: disp.crush_capacity,
            });
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.sim_time_s += dt;
        self.route_timer_s += dt;
        let route_interval = if self.vehicles.len() > 10_000 { 4.0 } else { 2.0 };
        if self.route_timer_s >= route_interval {
            self.route_timer_s = 0.0;
            self.network.recompute_dynamic_routes();
            self.refresh_urban_metrics();
        }

        // 1. Advance GTFS curbside passenger queues and dispatch scheduled transit runs
        let pending_buses = self.transit_gtfs.step_arrivals_and_dispatch(
            dt,
            self.sim_time_s,
            self.mode_shares.transit,
            self.transit_policy.headway_min,
        );
        self.spawn_scheduled_gtfs_buses(pending_buses);

        // 2. Parallel Rayon update of cyclists and sidewalk/crosswalk pedestrians
        let net_segs = &self.network.segments;
        let net_juncs = &self.network.junctions;
        let sim_t = self.sim_time_s;

        self.cyclists.par_iter_mut().for_each(|cyc| {
            let seg = &net_segs[cyc.segment_id];
            if !seg.closed {
                let len = seg.length.max(8.0);
                cyc.s_m += cyc.v_mps * dt;
                if cyc.s_m >= len {
                    let out_segs = &net_juncs[seg.to_junc].out_segs;
                    if !out_segs.is_empty() {
                        let next_sid = out_segs[(cyc.segment_id + 1) % out_segs.len()];
                        if !net_segs[next_sid].closed {
                            cyc.segment_id = next_sid;
                        }
                    }
                    cyc.s_m = 0.5;
                }
            }
        });

        self.pedestrians.par_iter_mut().for_each(|ped| {
            let seg = &net_segs[ped.segment_id];
            let len = seg.length.max(8.0);
            let dir = ped.walk_dir as f64;
            let near_end =
                (ped.walk_dir > 0 && ped.s_m > len - 3.2) || (ped.walk_dir < 0 && ped.s_m < 3.2);

            let jid = if ped.walk_dir > 0 {
                seg.to_junc
            } else {
                seg.from_junc
            };
            let junc = &net_juncs[jid];
            let walk_green = match junc.control_mode {
                ControlMode::Signal2Phase | ControlMode::Signal4PhaseProtectedLeft => {
                    let cyc_s = (junc.ns_green_s + junc.ew_green_s + 10.0).max(20.0);
                    let p = (sim_t + junc.offset_s).rem_euclid(cyc_s);
                    if seg.approach_axis == 0 {
                        p < junc.ns_green_s
                    } else {
                        p >= junc.ns_green_s + 5.0 && p < junc.ns_green_s + 5.0 + junc.ew_green_s
                    }
                }
                _ => true,
            };

            if near_end && !walk_green {
                ped.cur_speed_mps = 0.0;
            } else {
                ped.cur_speed_mps = ped.v_mps;
                ped.s_m += dir * ped.v_mps * dt;
                if ped.s_m >= len {
                    let outs = &junc.out_segs;
                    if !outs.is_empty() {
                        ped.segment_id = outs[(ped.phase_seed as usize) % outs.len()];
                        ped.s_m = 0.8;
                    } else {
                        ped.s_m = len - 0.8;
                        ped.walk_dir = -1;
                    }
                } else if ped.s_m <= 0.0 {
                    let ins = &junc.in_segs;
                    if !ins.is_empty() {
                        let prev_sid = ins[(ped.phase_seed as usize) % ins.len()];
                        ped.segment_id = prev_sid;
                        ped.s_m = (net_segs[prev_sid].length - 0.8).max(0.8);
                    } else {
                        ped.s_m = 0.8;
                        ped.walk_dir = 1;
                    }
                }
            }
        });

        if self.diurnal_live_clock_enabled {
            // Advance 24-hour diurnal clock (15 sim-seconds = 15 clock-minutes -> 96s full 24h cycle)
            let prev_hr = (self.diurnal_clock_hour.floor() as isize).rem_euclid(24) as u8;
            self.diurnal_clock_hour = (self.diurnal_clock_hour + dt * 0.25).rem_euclid(24.0);
            let new_hr = (self.diurnal_clock_hour.floor() as isize).rem_euclid(24) as u8;
            let rush_vc = crate::diurnal::diurnal_arterial_vc_ratio(new_hr, 1.0);
            self.effective_demand_per_lane_hr =
                (self.base_demand_per_lane_hr * (0.35 + 0.85 * rush_vc)).clamp(40.0, 2200.0);
            if new_hr != prev_hr {
                let is_pk = (7..10).contains(&new_hr) || (15..20).contains(&new_hr);
                for p in &mut self.parcels {
                    if is_pk {
                        if let Some(u) = p.peak_district_unit {
                            p.response_district_unit = Some(u);
                            p.response_district_station_seg = p.peak_district_station_seg;
                        }
                    } else if let Some(u) = p.offpeak_district_unit {
                        p.response_district_unit = Some(u);
                        p.response_district_station_seg = p.offpeak_district_station_seg;
                    }
                }
            }
        }
        self.spawn_boundary_vehicles(dt);
        let diurnal_ev_mult = if self.diurnal_live_clock_enabled {
            let hr = (self.diurnal_clock_hour.floor() as isize).rem_euclid(24) as u8;
            crate::diurnal::diurnal_call_multiplier(hr, 1.0)
        } else {
            1.0
        };
        if self.evp_config.enabled && self.evp_config.auto_dispatch_per_min > 0.01 {
            self.evp_spawn_accum_s +=
                dt * (self.evp_config.auto_dispatch_per_min * diurnal_ev_mult / 60.0);
            if self.evp_spawn_accum_s >= 1.0 {
                self.evp_spawn_accum_s -= 1.0;
                if self.evp_telemetry.active_responders < 4 {
                    let _ = self.dispatch_emergency_responder();
                }
            }
        }

        // 3. Build per-lane and per-connector sorted vehicle index lists
        let mut lane_vehs: Vec<Vec<usize>> = vec![Vec::new(); self.network.lanes.len()];
        let mut conn_vehs: Vec<Vec<usize>> = vec![Vec::new(); self.network.connectors.len()];
        for (vi, v) in self.vehicles.iter().enumerate() {
            match v.carrier {
                CarrierRef::Lane(lid) => lane_vehs[lid].push(vi),
                CarrierRef::Connector(cid) => conn_vehs[cid].push(vi),
            }
        }
        let veh_sort_ref = &self.vehicles;
        lane_vehs.par_iter_mut().for_each(|list| {
            list.sort_unstable_by(|&a, &b| {
                veh_sort_ref[a]
                    .s_m
                    .partial_cmp(&veh_sort_ref[b].s_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        });
        conn_vehs.par_iter_mut().for_each(|list| {
            list.sort_unstable_by(|&a, &b| {
                veh_sort_ref[a]
                    .s_m
                    .partial_cmp(&veh_sort_ref[b].s_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        });

        // 4. Assign planned connectors (honoring fixed GTFS segment chains for scheduled buses!)
        for lid in 0..self.network.lanes.len() {
            for &vi in &lane_vehs[lid] {
                let needs_pick = self.vehicles[vi]
                    .planned_conn_id
                    .map(|cid| self.network.connectors[cid].from_lane_id != lid)
                    .unwrap_or(true);
                if needs_pick {
                    let picked = self.pick_connector_for_vehicle(&self.vehicles[vi], lid);
                    self.vehicles[vi].planned_conn_id = picked;
                    self.vehicles[vi].turn_intent = picked
                        .map(|cid| self.network.connectors[cid].turn_type)
                        .unwrap_or(TurnType::Through);
                }
            }
        }

        // 4b. Evaluate corridor green-wave Emergency Vehicle Preemption (EVP) & per-segment siren fields
        {
            let RoadNetwork {
                junctions,
                segments,
                lanes,
                connectors,
                next_hop,
                ..
            } = &mut self.network;
            let (evp_tel, siren_map) = step_emergency_preemption(
                dt,
                &self.evp_config,
                junctions,
                segments,
                lanes,
                connectors,
                next_hop,
                &self.vehicles,
                &lane_vehs,
                &mut self.evp_corridor_segments,
                self.completed_ev_trips,
                self.yielding_vehicles_count,
            );
            self.evp_telemetry = evp_tel;
            self.siren_seg_s = siren_map;
        }

        // 5. Evaluate MOBIL multi-lane switching
        self.mobil_timer_s += dt;
        let mobil_period = if self.vehicles.len() > 4_000 { 0.20 } else { 0.0 };
        if self.mobil_timer_s >= mobil_period {
            let m_dt = self.mobil_timer_s.max(dt);
            self.mobil_timer_s = 0.0;
            self.step_mobil_lane_changes(m_dt, &mut lane_vehs);
        }

        // Precompute O(1) rank of each vehicle inside its sorted lane/connector list
        let mut veh_rank = vec![0u32; self.vehicles.len()];
        for list in &lane_vehs {
            for (rank, &vi) in list.iter().enumerate() {
                veh_rank[vi] = rank as u32;
            }
        }
        for list in &conn_vehs {
            for (rank, &vi) in list.iter().enumerate() {
                veh_rank[vi] = rank as u32;
            }
        }

        // 6. Transit Signal Priority (TSP) early green nudge for approaching buses
        {
            let RoadNetwork {
                segments,
                lanes,
                junctions,
                ..
            } = &mut self.network;
            for seg in segments.iter() {
                if !seg.has_bus_lane || seg.closed {
                    continue;
                }
                if let Some(&bus_lid) = seg.lane_ids.last() {
                    let lane_len = lanes[bus_lid].length;
                    let has_approaching_bus = lane_vehs[bus_lid].iter().rev().take(4).any(|&vi| {
                        self.vehicles[vi].v_class == VehicleClass::TransitBus
                            && self.vehicles[vi].dwell_remaining_s <= 0.0
                            && (lane_len - self.vehicles[vi].s_m) < 34.0
                    });
                    if has_approaching_bus {
                        let jid = seg.to_junc;
                        if matches!(
                            junctions[jid].control_mode,
                            ControlMode::Signal2Phase | ControlMode::Signal4PhaseProtectedLeft
                        ) && !junctions[jid].hold_phase
                        {
                            junctions[jid].manual_step_offset_s += dt * 0.85;
                        }
                    }
                }
            }
        }

        // 6b. Step decentralized Varaiya (2013) Max-Pressure adaptive signal controllers!
        self.network.mp_allow_4phase = self.mp_config.allow_protected_left_phases;
        {
            let RoadNetwork {
                junctions,
                segments,
                lanes,
                connectors,
                ..
            } = &mut self.network;
            self.mp_telemetry = step_max_pressure_controllers(
                dt,
                &self.mp_config,
                junctions,
                segments,
                lanes,
                connectors,
                &self.vehicles,
                &lane_vehs,
            );
        }

        // 7. Parallel Rayon calculation of IDM accelerations (including smooth GTFS curbside stop braking!)
        let idm = self.idm;
        let sim_time = self.sim_time_s;
        let net = &self.network;
        let gtfs_ref = &self.transit_gtfs;
        let veh_ref = &self.vehicles;

        let new_accels: Vec<f64> = (0..veh_ref.len())
            .into_par_iter()
            .map(|vi| {
                let veh = &veh_ref[vi];
                if veh.dwell_remaining_s > 0.0 {
                    return 0.0;
                }
                let rank = veh_rank[vi] as usize;
                match veh.carrier {
                    CarrierRef::Lane(lid) => {
                        let lane = &net.lanes[lid];
                        let seg = &net.segments[lane.segment_id];
                        let is_ev = veh.v_class == VehicleClass::EmergencyResponder;
                        let speed_cap = if is_ev { seg.speed_mps * 1.38 } else { seg.speed_mps };
                        let v0 = veh.v_max_mps.min(speed_cap);
                        let a_max = idm.max_accel_mps2 * veh.accel_mult;
                        let list = &lane_vehs[lid];

                        // Find nearest non-pulled-over leader (emergency units pass alongside cars yielded >= 0.80 m right!)
                        let mut eff_lead_opt: Option<&Vehicle> = None;
                        if is_ev {
                            for &cand_vi in &list[rank + 1..] {
                                let cand = &veh_ref[cand_vi];
                                if cand.v_class == VehicleClass::EmergencyResponder
                                    || cand.lat_visual_offset_m < 0.80
                                {
                                    eff_lead_opt = Some(cand);
                                    break;
                                }
                            }
                        } else if rank + 1 < list.len() {
                            eff_lead_opt = Some(&veh_ref[list[rank + 1]]);
                        }

                        let (mut gap, mut dv) = (999.0, 0.0);
                        if let Some(lead) = eff_lead_opt {
                            gap = lead.s_m - lead.length_m - veh.s_m;
                            dv = veh.v - lead.v;
                        } else if let Some(cid) = veh.planned_conn_id {
                            let conn = &net.connectors[cid];
                            let permitted = Self::can_enter_connector_static(
                                veh, conn, net, veh_ref, &lane_vehs, &conn_vehs, &idm, sim_time,
                            );
                            if !permitted {
                                gap = (lane.length - 0.6 - veh.s_m).max(0.15);
                                dv = veh.v;
                            } else if let Some(&c_first) = conn_vehs[cid].first() {
                                let clead = &veh_ref[c_first];
                                gap = (lane.length - veh.s_m) + clead.s_m - clead.length_m;
                                dv = veh.v - clead.v;
                            } else if let Some(&d_first) = lane_vehs[conn.to_lane_id].first() {
                                let dlead = &veh_ref[d_first];
                                gap = (lane.length - veh.s_m) + conn.length + dlead.s_m
                                    - dlead.length_m;
                                dv = veh.v - dlead.v;
                            }
                        }

                        // Scheduled GTFS curbside stop target braking
                        if let Some(r_idx) = veh.gtfs_route_idx {
                            if let Some(route) = gtfs_ref.routes.get(r_idx as usize) {
                                if let Some(&(stop_id, chain_seq)) =
                                    route.route_stops.get(veh.gtfs_next_stop_order as usize)
                                {
                                    if chain_seq == veh.gtfs_route_seq as usize {
                                        let stop = &gtfs_ref.stops[stop_id];
                                        let need_stop =
                                            stop.waiting_pax >= 1.0 || veh.onboard_pax > 0;
                                        if need_stop && stop.s_m >= veh.s_m - 0.8 {
                                            let g_stop = (stop.s_m + idm.min_jam_gap_m - 0.5 - veh.s_m).max(0.12);
                                            if g_stop < gap {
                                                gap = g_stop;
                                                dv = veh.v;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if seg.stalled_lane == Some(lane.index) {
                            let stall_pos = seg.length * 0.52;
                            if stall_pos > veh.s_m + 0.5 {
                                let g_stall = (stall_pos - 2.5 - veh.s_m).max(0.15);
                                if g_stall < gap {
                                    gap = g_stall;
                                    dv = veh.v;
                                }
                            }
                        }
                        idm_acceleration(veh.v, v0, gap, dv, a_max, &idm)
                    }
                    CarrierRef::Connector(cid) => {
                        let conn = &net.connectors[cid];
                        let v0 = veh.v_max_mps.min(conn.speed_mps);
                        let a_max = idm.max_accel_mps2 * veh.accel_mult;
                        let list = &conn_vehs[cid];
                        let (mut gap, mut dv) = (999.0, 0.0);
                        if rank + 1 < list.len() {
                            let lead = &veh_ref[list[rank + 1]];
                            gap = lead.s_m - lead.length_m - veh.s_m;
                            dv = veh.v - lead.v;
                        } else if let Some(&d_first) = lane_vehs[conn.to_lane_id].first() {
                            let dlead = &veh_ref[d_first];
                            gap = (conn.length - veh.s_m) + dlead.s_m - dlead.length_m;
                            dv = veh.v - dlead.v;
                        }
                        idm_acceleration(veh.v, v0, gap, dv, a_max, &idm)
                    }
                }
            })
            .collect();

        for (vi, &acc) in new_accels.iter().enumerate() {
            self.vehicles[vi].acc = acc;
        }

        // 8. Semi-implicit integration, GTFS curbside stop boarding/alighting dwell, & carrier transitions
        let mut retained = Vec::with_capacity(self.vehicles.len());
        let old_vehicles = std::mem::take(&mut self.vehicles);
        let mut lane_entry_rear = vec![f64::INFINITY; self.network.lanes.len()];
        for v in &old_vehicles {
            if let CarrierRef::Lane(lid) = v.carrier {
                lane_entry_rear[lid] = lane_entry_rear[lid].min(v.s_m - v.length_m);
            }
        }
        let mut active_bus_loads: Vec<(usize, usize, f64, u16, u16)> = Vec::new();

        for mut veh in old_vehicles {
            if veh.dwell_remaining_s > 0.0 {
                veh.dwell_remaining_s = (veh.dwell_remaining_s - dt as f32).max(0.0);
                veh.v = 0.0;
                veh.acc = 0.0;
                if let Some(r_idx) = veh.gtfs_route_idx {
                    active_bus_loads.push((
                        r_idx as usize,
                        veh.gtfs_route_seq as usize,
                        veh.s_m,
                        veh.onboard_pax,
                        veh.crush_capacity,
                    ));
                }
                retained.push(veh);
                continue;
            }

            // Acoustic siren shoulder pull-over yielding for civilian vehicles ahead of any EmergencyResponder
            let mut pulling_over = false;
            if self.evp_config.enabled && veh.v_class != VehicleClass::EmergencyResponder {
                if let CarrierRef::Lane(lid) = veh.carrier {
                    let sid = self.network.lanes[lid].segment_id;
                    if let Some(sirens) = self.siren_seg_s.get(sid) {
                        if sirens.iter().any(|&ev_s| {
                            let ahead_d = veh.s_m - ev_s;
                            (-10.0..=self.evp_config.siren_yield_radius_m).contains(&ahead_d)
                        }) {
                            pulling_over = true;
                        }
                    }
                }
            }

            let prev_v = veh.v;
            let v_raw = (prev_v + veh.acc * dt).max(0.0);
            let v_cap_pull = if pulling_over && v_raw > 4.2 {
                // Cap yielding speed drop cleanly by physical IDM hard deceleration per step
                let min_step_v = (prev_v - self.idm.hard_decel_mps2 * dt).max(4.2);
                v_raw.min(min_step_v).max(0.0)
            } else {
                v_raw
            };
            if pulling_over {
                veh.acc = ((v_cap_pull - prev_v) / dt.max(1e-6)).max(-self.idm.hard_decel_mps2);
            }
            let ds = 0.5 * (prev_v + v_cap_pull) * dt;
            veh.v = v_cap_pull;
            veh.s_m += ds;
            if pulling_over {
                // Smoothly pull +1.28 m toward right-hand curb shoulder (`+nx, +ny`)
                let alpha = (dt * 4.8).clamp(0.0, 1.0);
                veh.lat_visual_offset_m += (1.28 - veh.lat_visual_offset_m) * alpha;
            } else {
                veh.lat_visual_offset_m *= (-dt * 6.5).exp();
            }

            // Check if a scheduled GTFS bus has arrived at its next curbside stop
            if let (CarrierRef::Lane(lid), Some(r_idx)) = (veh.carrier, veh.gtfs_route_idx) {
                let r_usize = r_idx as usize;
                let stop_ord = veh.gtfs_next_stop_order as usize;
                let stop_match = self
                    .transit_gtfs
                    .routes
                    .get(r_usize)
                    .and_then(|r| r.route_stops.get(stop_ord).copied());

                if let Some((stop_id, chain_seq)) = stop_match {
                    let seg_id = self.network.lanes[lid].segment_id;
                    let stop_s = self.transit_gtfs.stops[stop_id].s_m;
                    if chain_seq == veh.gtfs_route_seq as usize
                        && self.transit_gtfs.stops[stop_id].segment_id == seg_id
                        && veh.s_m >= stop_s - 1.6
                    {
                        let outcome = self.transit_gtfs.process_bus_stop_service(
                            r_usize,
                            stop_ord,
                            veh.onboard_pax,
                            self.sim_time_s,
                            self.transit_policy.headway_min,
                        );
                        veh.onboard_pax = outcome.new_onboard;
                        veh.gtfs_next_stop_order += 1;
                        if outcome.dwell_s > 0.05 {
                            veh.s_m = stop_s;
                            veh.v = 0.0;
                            veh.acc = 0.0;
                            veh.dwell_remaining_s = outcome.dwell_s;
                        }
                    }
                }
            }

            match veh.carrier {
                CarrierRef::Lane(lid) => {
                    let lane_len = self.network.lanes[lid].length;
                    let seg_id = self.network.lanes[lid].segment_id;
                    if veh.s_m >= lane_len {
                        let Some(cid) = veh.planned_conn_id else {
                            if self.network.sinks.contains(&seg_id) || veh.gtfs_route_idx.is_some()
                            {
                                if veh.v_class == VehicleClass::EmergencyResponder {
                                    self.completed_ev_trips += 1;
                                }
                                self.completed_trips
                                    .push_back((self.sim_time_s - veh.spawn_time_s, self.sim_time_s));
                            }
                            continue;
                        };
                        let conn = &self.network.connectors[cid];
                        let light = self.network.connector_light(
                            &self.network.junctions[conn.junction_id],
                            conn,
                            self.sim_time_s,
                        );
                        if light != LightColor::Red && !self.network.segments[conn.to_seg_id].closed
                        {
                            let over = veh.s_m - lane_len;
                            veh.carrier = CarrierRef::Connector(cid);
                            veh.s_m = over.min(conn.length - 0.1);
                        } else {
                            veh.s_m = lane_len - 0.35;
                            veh.v = 0.0;
                            veh.acc = 0.0;
                        }
                    }
                }
                CarrierRef::Connector(cid) => {
                    let conn_len = self.network.connectors[cid].length;
                    let dst_lid = self.network.connectors[cid].to_lane_id;
                    if veh.s_m >= conn_len {
                        let max_safe_s = lane_entry_rear[dst_lid] - self.idm.min_jam_gap_m;
                        if max_safe_s < 0.05 {
                            veh.s_m = (conn_len - 0.05).max(0.0);
                            veh.v = 0.0;
                            veh.acc = 0.0;
                            retained.push(veh);
                            continue;
                        }
                        let over = veh.s_m - conn_len;
                        let dst_len = self.network.lanes[dst_lid].length;
                        let dst_seg = self.network.lanes[dst_lid].segment_id;
                        veh.carrier = CarrierRef::Lane(dst_lid);
                        veh.s_m = over.min(dst_len - 0.5).min(max_safe_s);
                        veh.planned_conn_id = None;
                        lane_entry_rear[dst_lid] =
                            lane_entry_rear[dst_lid].min(veh.s_m - veh.length_m);

                        if let Some(r_idx) = veh.gtfs_route_idx {
                            if let Some(route) = self.transit_gtfs.routes.get(r_idx as usize) {
                                let next_seq = veh.gtfs_route_seq as usize + 1;
                                if route.segment_chain.get(next_seq) == Some(&dst_seg) {
                                    veh.gtfs_route_seq = next_seq as u16;
                                } else if let Some(found) =
                                    route.segment_chain.iter().position(|&s| s == dst_seg)
                                {
                                    veh.gtfs_route_seq = found as u16;
                                }
                                // Retire scheduled bus cleanly when it reaches final segment past all stops
                                if veh.gtfs_route_seq as usize + 1 >= route.segment_chain.len()
                                    && veh.gtfs_next_stop_order as usize >= route.route_stops.len()
                                    && veh.s_m > dst_len * 0.78
                                {
                                    self.completed_trips.push_back((
                                        self.sim_time_s - veh.spawn_time_s,
                                        self.sim_time_s,
                                    ));
                                    continue;
                                }
                            }
                        } else if self.network.sinks.contains(&dst_seg)
                            && self.network.lanes[dst_lid].out_connectors.is_empty()
                            && veh.s_m > dst_len * 0.85
                        {
                            if veh.v_class == VehicleClass::EmergencyResponder {
                                self.completed_ev_trips += 1;
                            }
                            self.completed_trips
                                .push_back((self.sim_time_s - veh.spawn_time_s, self.sim_time_s));
                            continue;
                        }
                    }
                }
            }

            if let Some(r_idx) = veh.gtfs_route_idx {
                active_bus_loads.push((
                    r_idx as usize,
                    veh.gtfs_route_seq as usize,
                    veh.s_m,
                    veh.onboard_pax,
                    veh.crush_capacity,
                ));
            }
            retained.push(veh);
        }
        self.yielding_vehicles_count = retained
            .iter()
            .filter(|v| v.v_class != VehicleClass::EmergencyResponder && v.lat_visual_offset_m >= 0.65)
            .count();
        self.evp_telemetry.yielding_vehicles = self.yielding_vehicles_count;
        self.evp_telemetry.completed_ev_trips = self.completed_ev_trips;
        self.vehicles = retained;

        self.transit_gtfs
            .update_telemetry_from_fleet(&active_bus_loads, self.transit_policy.headway_min);

        // Update segment EMA speeds
        let mut seg_sums = vec![(0.0_f64, 0usize); self.network.segments.len()];
        for v in &self.vehicles {
            if let CarrierRef::Lane(lid) = v.carrier {
                let sid = self.network.lanes[lid].segment_id;
                seg_sums[sid].0 += v.v;
                seg_sums[sid].1 += 1;
            }
        }
        for (sid, seg) in self.network.segments.iter_mut().enumerate() {
            let target = if seg_sums[sid].1 > 0 {
                seg_sums[sid].0 / seg_sums[sid].1 as f64
            } else {
                seg.speed_mps
            };
            seg.ema_speed_mps = seg.ema_speed_mps * 0.94 + target * 0.06;
        }

        let cutoff = self.sim_time_s - 90.0;
        while self
            .completed_trips
            .front()
            .map(|&(_, t)| t < cutoff)
            .unwrap_or(false)
        {
            self.completed_trips.pop_front();
        }

        self.xt_timer_s += dt;
        if self.xt_timer_s >= 0.25 {
            self.xt_timer_s = 0.0;
            let w_sid = self
                .watched_segment_id
                .min(self.network.segments.len().saturating_sub(1));
            let mut samples = Vec::new();
            for v in &self.vehicles {
                if let CarrierRef::Lane(lid) = v.carrier {
                    if self.network.lanes[lid].segment_id == w_sid {
                        let norm =
                            (v.s_m / self.network.lanes[lid].length.max(1.0)).clamp(0.0, 1.0);
                        samples.push((norm as f32, (v.v * 2.236_94) as f32));
                    }
                }
            }
            self.xt_history.push_back(SpaceTimeColumn {
                sim_time_s: self.sim_time_s,
                segment_id: w_sid,
                samples,
            });
            if self.xt_history.len() > 240 {
                self.xt_history.pop_front();
            }
        }

        // Sample 1 Hz time-series telemetry for CSV export
        let mut recorder = std::mem::take(&mut self.telemetry_log);
        recorder.maybe_record(dt, self);
        self.telemetry_log = recorder;
    }

    #[allow(clippy::too_many_arguments)]
    fn can_enter_connector_static(
        veh: &Vehicle,
        conn: &crate::network::TurnConnector,
        net: &RoadNetwork,
        vehicles: &[Vehicle],
        lane_vehs: &[Vec<usize>],
        conn_vehs: &[Vec<usize>],
        idm: &IdmParams,
        sim_time_s: f64,
    ) -> bool {
        if net.segments[conn.to_seg_id].closed {
            return false;
        }
        if let Some(&first_dst) = lane_vehs[conn.to_lane_id].first() {
            let rv = &vehicles[first_dst];
            if rv.s_m - rv.length_m < idm.min_jam_gap_m + 2.2 && rv.v < 1.5 {
                return false;
            }
        }
        if let Some(&first_conn) = conn_vehs[conn.id].first() {
            let cv = &vehicles[first_conn];
            if cv.s_m - cv.length_m < idm.min_jam_gap_m + 1.0 {
                return false;
            }
        }

        let junc = &net.junctions[conn.junction_id];
        let light = net.connector_light(junc, conn, sim_time_s);
        if light == LightColor::Red {
            return false;
        }
        if light == LightColor::Yellow {
            if let CarrierRef::Lane(lid) = veh.carrier {
                let dist_stop = (net.lanes[lid].length - veh.s_m).max(0.0);
                let stop_d = (veh.v * veh.v) / (2.0 * idm.comfort_decel_mps2);
                if dist_stop > 2.5 && stop_d <= dist_stop + 2.0 {
                    return false;
                }
            }
        }

        if junc.control_mode == ControlMode::Roundabout
            && !net.segments[conn.from_seg_id].is_roundabout
        {
            for &in_sid in &junc.in_segs {
                if !net.segments[in_sid].is_roundabout {
                    continue;
                }
                for &rlid in &net.segments[in_sid].lane_ids {
                    if let Some(&lead_i) = lane_vehs[rlid].last() {
                        let lv = &vehicles[lead_i];
                        let rem = net.lanes[rlid].length - lv.s_m;
                        if rem < 26.0 && (rem / lv.v.max(2.5)) < 2.9 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn pick_connector_for_vehicle(&self, veh: &Vehicle, lid: usize) -> Option<usize> {
        let lane = &self.network.lanes[lid];
        if lane.out_connectors.is_empty() {
            return None;
        }

        // Fixed-route GTFS bus: follow exact `route.segment_chain[gtfs_route_seq + 1]` and prefer rightmost curb/BRT lane!
        if let Some(r_idx) = veh.gtfs_route_idx {
            if let Some(route) = self.transit_gtfs.routes.get(r_idx as usize) {
                let next_seq = veh.gtfs_route_seq as usize + 1;
                if let Some(&target_seg) = route.segment_chain.get(next_seq) {
                    if !self.network.segments[target_seg].closed {
                        let best_curb = lane
                            .out_connectors
                            .iter()
                            .copied()
                            .filter(|&cid| self.network.connectors[cid].to_seg_id == target_seg)
                            .max_by_key(|&cid| {
                                self.network.lanes[self.network.connectors[cid].to_lane_id].index
                            });
                        if best_curb.is_some() {
                            return best_curb;
                        }
                    }
                } else {
                    return None;
                }
            }
        }

        if let Some(dest) = veh.dest_seg_id {
            if let Some(&next_seg) = self.network.next_hop.get(&(lane.segment_id, dest)) {
                let mut count = 0usize;
                let mut picked = None;
                let target_idx = (veh.id as usize) % lane.out_connectors.len().max(1);
                for &cid in &lane.out_connectors {
                    let to_s = self.network.connectors[cid].to_seg_id;
                    if to_s == next_seg && !self.network.segments[to_s].closed {
                        if picked.is_none() || count == target_idx {
                            picked = Some(cid);
                        }
                        count += 1;
                    }
                }
                if picked.is_some() {
                    return picked;
                }
            }
        }
        let open_len = lane
            .out_connectors
            .iter()
            .filter(|&&cid| !self.network.segments[self.network.connectors[cid].to_seg_id].closed)
            .count();
        if open_len == 0 {
            return None;
        }
        let pick_n = (veh.id as usize) % open_len;
        lane.out_connectors
            .iter()
            .copied()
            .filter(|&cid| !self.network.segments[self.network.connectors[cid].to_seg_id].closed)
            .nth(pick_n)
    }

    fn spawn_boundary_vehicles(&mut self, dt: f64) {
        let open_sources: Vec<usize> = self
            .network
            .sources
            .iter()
            .copied()
            .filter(|&sid| !self.network.segments[sid].closed)
            .collect();
        if open_sources.is_empty() {
            return;
        }
        let entry_lanes: usize = open_sources
            .iter()
            .map(|&sid| self.network.segments[sid].lane_count)
            .sum();
        let rate_per_s = (self.effective_demand_per_lane_hr * entry_lanes.max(2) as f64) / 3600.0;
        self.spawn_accum += rate_per_s * dt;
        if self.spawn_accum < 1.0 {
            return;
        }

        let mut lane_min_rear = vec![f64::INFINITY; self.network.lanes.len()];
        for v in &self.vehicles {
            if let CarrierRef::Lane(lid) = v.carrier {
                let rear = v.s_m - v.length_m;
                if rear < lane_min_rear[lid] {
                    lane_min_rear[lid] = rear;
                }
            }
        }

        let mut sorted_sinks: Vec<usize> = self.network.sinks.iter().copied().collect();
        sorted_sinks.sort_unstable();

        let total_w: usize = open_sources
            .iter()
            .map(|&sid| {
                self.network.segments[sid].lane_count
                    * self.network.segments[sid].priority as usize
            })
            .sum();

        while self.spawn_accum >= 1.0 {
            self.spawn_accum -= 1.0;
            if self.vehicles.len() >= self.max_vehicles {
                break;
            }
            let mut pick = (self.next_rand_f64() * total_w.max(1) as f64) as isize;
            let mut chosen_sid = open_sources[0];
            for &sid in &open_sources {
                pick -= (self.network.segments[sid].lane_count
                    * self.network.segments[sid].priority as usize) as isize;
                if pick <= 0 {
                    chosen_sid = sid;
                    break;
                }
            }

            let n_lanes = self.network.segments[chosen_sid].lane_ids.len();
            let offset = (self.next_rand_f64() * n_lanes as f64) as usize;
            let mut spawn_lid = None;
            for k in 0..n_lanes {
                let lid = self.network.segments[chosen_sid].lane_ids[(k + offset) % n_lanes];
                if lane_min_rear[lid] >= self.idm.min_jam_gap_m + 6.5 {
                    spawn_lid = Some(lid);
                    break;
                }
            }
            let Some(lid) = spawn_lid else { continue };

            let dest_seg_id = if sorted_sinks.is_empty() {
                None
            } else {
                let mut found = None;
                let base_idx = (self.next_rand_f64() * sorted_sinks.len() as f64) as usize;
                for probe in 0..sorted_sinks.len().min(8) {
                    let sk = sorted_sinks[(base_idx + probe) % sorted_sinks.len()];
                    if sk != chosen_sid && self.network.next_hop.contains_key(&(chosen_sid, sk)) {
                        found = Some(sk);
                        break;
                    }
                }
                found
            };

            let roll = self.next_rand_f64() * 100.0;
            // Scheduled buses come from TransitGtfsSystem; boundary spawns generate freight trucks & passenger sedans
            let (v_class, length_m, width_m, accel_mult, v_max_mps) = if roll < self.heavy_share_pct {
                (VehicleClass::FreightTruck, 8.4, 2.35, 0.72, 26.0)
            } else {
                (VehicleClass::Sedan, 4.5, 1.9, 1.0, 36.0)
            };

            let v_init = (self.network.segments[chosen_sid].speed_mps * 0.82).max(2.5);
            let jitter = 0.92 + self.next_rand_f64() * 0.16;
            let id = self.next_veh_id;
            self.next_veh_id += 1;
            lane_min_rear[lid] = 0.5 - length_m;
            self.vehicles.push(Vehicle {
                id,
                v_class,
                length_m,
                width_m,
                accel_mult,
                v_max_mps: v_max_mps * jitter,
                carrier: CarrierRef::Lane(lid),
                s_m: 0.5,
                v: v_init,
                acc: 0.0,
                lat_visual_offset_m: 0.0,
                lc_cooldown_s: 1.0,
                dest_seg_id,
                planned_conn_id: None,
                turn_intent: TurnType::Through,
                spawn_time_s: self.sim_time_s,
                gtfs_route_idx: None,
                gtfs_route_seq: 0,
                gtfs_next_stop_order: 0,
                dwell_remaining_s: 0.0,
                onboard_pax: 0,
                crush_capacity: 0,
            });
        }
    }

    fn step_mobil_lane_changes(&mut self, dt: f64, lane_vehs: &mut [Vec<usize>]) {
        for v in &mut self.vehicles {
            v.lc_cooldown_s = (v.lc_cooldown_s - dt).max(0.0);
        }

        for sid in 0..self.network.segments.len() {
            let seg_lc = self.network.segments[sid].lane_count;
            if self.network.segments[sid].closed || seg_lc < 2 {
                continue;
            }
            let lane_ids = self.network.segments[sid].lane_ids.clone();
            let seg_speed = self.network.segments[sid].speed_mps;
            let stalled_lane = self.network.segments[sid].stalled_lane;

            for (li, &lid) in lane_ids.iter().enumerate() {
                let lane_len = self.network.lanes[lid].length;
                let snapshot = lane_vehs[lid].clone();

                for &vi in &snapshot {
                    // Scheduled GTFS buses hold the right-hand curb/BRT lane so they never miss curbside stops
                    if self.vehicles[vi].gtfs_route_idx.is_some()
                        || self.vehicles[vi].dwell_remaining_s > 0.0
                    {
                        continue;
                    }
                    if self.vehicles[vi].lc_cooldown_s > 0.0
                        || self.vehicles[vi].s_m < 5.0
                        || (lane_len - self.vehicles[vi].s_m) < 8.0
                    {
                        continue;
                    }

                    let mut candidates = [usize::MAX; 2];
                    let mut n_cand = 0usize;
                    if li > 0 {
                        candidates[n_cand] = li - 1;
                        n_cand += 1;
                    }
                    if li + 1 < seg_lc {
                        // Do not allow sedans to merge into a dedicated Bus/BRT curb lane!
                        let is_dedicated_bus_lane =
                            self.network.segments[sid].has_bus_lane && (li + 2 == seg_lc);
                        let can_use_bus_lane = matches!(
                            self.vehicles[vi].v_class,
                            VehicleClass::TransitBus | VehicleClass::EmergencyResponder
                        );
                        if !is_dedicated_bus_lane || can_use_bus_lane {
                            candidates[n_cand] = li + 1;
                            n_cand += 1;
                        }
                    }

                    for &target_li in &candidates[..n_cand] {
                        if stalled_lane == Some(target_li) {
                            continue;
                        }
                        let target_lid = lane_ids[target_li];
                        let veh_s = self.vehicles[vi].s_m;
                        let veh_len = self.vehicles[vi].length_m;
                        let veh_v = self.vehicles[vi].v;
                        let v0 = self.vehicles[vi].v_max_mps.min(seg_speed);
                        let a_max = self.idm.max_accel_mps2 * self.vehicles[vi].accel_mult;

                        let tlist = &lane_vehs[target_lid];
                        let ins_pos = tlist.partition_point(|&tvi| self.vehicles[tvi].s_m < veh_s);
                        let new_fol = if ins_pos > 0 {
                            Some(tlist[ins_pos - 1])
                        } else {
                            None
                        };
                        let new_lead = if ins_pos < tlist.len() {
                            Some(tlist[ins_pos])
                        } else {
                            None
                        };

                        let gap_ahead_new = new_lead
                            .map(|l| self.vehicles[l].s_m - self.vehicles[l].length_m - veh_s)
                            .unwrap_or(999.0);
                        let gap_behind_new = new_fol
                            .map(|f| veh_s - veh_len - self.vehicles[f].s_m)
                            .unwrap_or(999.0);
                        if gap_ahead_new < self.idm.min_jam_gap_m + 1.2
                            || gap_behind_new < self.idm.min_jam_gap_m + 1.2
                        {
                            continue;
                        }

                        let (cur_pos, old_lead) = {
                            let cur_list = &lane_vehs[lid];
                            let Ok(cp) = cur_list.binary_search_by(|&x| {
                                self.vehicles[x]
                                    .s_m
                                    .partial_cmp(&veh_s)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            }) else {
                                break;
                            };
                            if cur_list[cp] != vi {
                                continue;
                            }
                            let ol = if cp + 1 < cur_list.len() {
                                Some(cur_list[cp + 1])
                            } else {
                                None
                            };
                            (cp, ol)
                        };

                        let gap_cur = old_lead
                            .map(|l| self.vehicles[l].s_m - self.vehicles[l].length_m - veh_s)
                            .unwrap_or(999.0);
                        let dv_cur = old_lead.map(|l| veh_v - self.vehicles[l].v).unwrap_or(0.0);
                        let a_cur = idm_acceleration(veh_v, v0, gap_cur, dv_cur, a_max, &self.idm);
                        let dv_new = new_lead.map(|l| veh_v - self.vehicles[l].v).unwrap_or(0.0);
                        let a_tilde =
                            idm_acceleration(veh_v, v0, gap_ahead_new, dv_new, a_max, &self.idm);

                        let (mut a_nf_before, mut a_nf_after) = (0.0, 0.0);
                        if let Some(nf) = new_fol {
                            let nfv = &self.vehicles[nf];
                            let nf_v0 = nfv.v_max_mps.min(seg_speed);
                            let nf_a = self.idm.max_accel_mps2 * nfv.accel_mult;
                            let g_before = new_lead
                                .map(|l| self.vehicles[l].s_m - self.vehicles[l].length_m - nfv.s_m)
                                .unwrap_or(999.0);
                            let dv_before =
                                new_lead.map(|l| nfv.v - self.vehicles[l].v).unwrap_or(0.0);
                            a_nf_before =
                                idm_acceleration(nfv.v, nf_v0, g_before, dv_before, nf_a, &self.idm);
                            a_nf_after = idm_acceleration(
                                nfv.v,
                                nf_v0,
                                gap_behind_new,
                                nfv.v - veh_v,
                                nf_a,
                                &self.idm,
                            );
                            if a_nf_after < -self.mobil.safe_decel_mps2 {
                                continue;
                            }
                        }

                        let is_ev = self.vehicles[vi].v_class == VehicleClass::EmergencyResponder;
                        let siren_nearby = !is_ev
                            && self.evp_config.enabled
                            && self.siren_seg_s.get(sid).map_or(false, |sirens| {
                                sirens.iter().any(|&ev_s| {
                                    let d = veh_s - ev_s;
                                    (-12.0..=self.evp_config.siren_yield_radius_m).contains(&d)
                                })
                            });

                        // Civilian vehicles under siren alert strongly bias right (`target_li > li`) and never cut left (`target_li < li`)!
                        if siren_nearby && target_li < li {
                            continue;
                        }
                        let siren_pullover_bias = if siren_nearby && target_li > li {
                            6.8
                        } else if is_ev {
                            2.4
                        } else {
                            0.0
                        };

                        let stall_bias = if stalled_lane == Some(li) { 3.5 } else { 0.0 };
                        let eff_politeness = if is_ev || siren_nearby {
                            0.05
                        } else {
                            self.mobil.politeness
                        };
                        let incentive = (a_tilde - a_cur + stall_bias + siren_pullover_bias)
                            + eff_politeness * (a_nf_after - a_nf_before);
                        if incentive > self.mobil.threshold_mps2 {
                            lane_vehs[lid].remove(cur_pos);
                            self.vehicles[vi].carrier = CarrierRef::Lane(target_lid);
                            self.vehicles[vi].planned_conn_id = None;
                            self.vehicles[vi].lc_cooldown_s = if is_ev { 0.65 } else { 2.0 };
                            self.vehicles[vi].lat_visual_offset_m =
                                (li as f64 - target_li as f64) * LANE_WIDTH_M;
                            lane_vehs[target_lid].insert(ins_pos, vi);
                            break;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::RoadNetwork;
    use crate::osm::{builtin_riverfront_bridges_osm, parse_osm_xml_str};

    #[test]
    fn multi_siren_per_segment_preserved_and_yielding_obeys_hard_decel() {
        let osm = parse_osm_xml_str(&builtin_riverfront_bridges_osm()).expect("builtin osm");
        let net = RoadNetwork::compile(&osm);
        let mut world = SimulationWorld::new(net);

        // Choose a long segment with multiple lanes if available
        let seg_id = world
            .network
            .segments
            .iter()
            .position(|s| !s.closed && s.length > 120.0 && !s.lane_ids.is_empty())
            .expect("long open segment");
        let lane_id = *world.network.segments[seg_id].lane_ids.last().unwrap();
        let seg_len = world.network.segments[seg_id].length;

        world.vehicles.clear();
        world.evp_config.enabled = true;
        world.evp_config.siren_yield_radius_m = 60.0;

        let make_veh = |id: u64, class: VehicleClass, s_m: f64, v: f64| Vehicle {
            id,
            v_class: class,
            carrier: CarrierRef::Lane(lane_id),
            s_m,
            length_m: if class == VehicleClass::EmergencyResponder { 6.5 } else { 4.5 },
            width_m: 2.0,
            v_max_mps: 25.0,
            accel_mult: 1.0,
            v,
            acc: 0.0,
            lat_visual_offset_m: 0.0,
            lc_cooldown_s: 0.0,
            dest_seg_id: None,
            planned_conn_id: None,
            turn_intent: TurnType::Through,
            spawn_time_s: 0.0,
            gtfs_route_idx: None,
            gtfs_route_seq: 0,
            gtfs_next_stop_order: 0,
            dwell_remaining_s: 0.0,
            onboard_pax: 0,
            crush_capacity: 0,
        };

        // Two sirens far apart on the same segment, plus a fast civilian car ahead of the second siren
        let s1 = 10.0;
        let s2 = (seg_len * 0.72).max(95.0).min(seg_len - 25.0);
        let car_s = (s2 + 12.0).min(seg_len - 5.0);
        world.vehicles.push(make_veh(1, VehicleClass::EmergencyResponder, s1, 14.0));
        world.vehicles.push(make_veh(2, VehicleClass::EmergencyResponder, s2, 14.0));
        world.vehicles.push(make_veh(3, VehicleClass::Sedan, car_s, 20.0));

        let dt = 0.05;
        let prev_v = 20.0;
        world.step(dt);

        // Both sirens must remain recorded on seg_id without overwriting one another
        assert!(
            world.siren_seg_s[seg_id].len() >= 2,
            "Expected >= 2 siren entries on segment {seg_id}, got {:?}",
            world.siren_seg_s[seg_id]
        );
        let car = world.vehicles.iter().find(|v| v.id == 3).expect("civilian car");
        let observed_decel = (prev_v - car.v) / dt;
        assert!(
            observed_decel <= world.idm.hard_decel_mps2 + 1e-6,
            "Yielding vehicle decel {observed_decel:.3} m/s^2 exceeded hard_decel_mps2 ({:.3})",
            world.idm.hard_decel_mps2
        );
        assert!(
            car.lat_visual_offset_m > 0.01,
            "Car ahead of second siren must begin right-shoulder pull-over yield"
        );
    }
}

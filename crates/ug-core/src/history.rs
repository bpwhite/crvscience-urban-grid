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

//! Bounded Command/Memento undo-redo history stack (`EditHistoryStack`) for live CAD street
//! editing (`DrawRoad`, `EditLanes`, `ConvertRoundabout`), parcel land-use zoning (`PaintZoning`,
//! `TodUpzoning`), and corridor/signal policies while preserving active microscopic agents.

use crate::diurnal::{
    solve_diurnal_peak_offpeak_dispatch_tables, DiurnalDispatchRegime, DiurnalProfileConfig,
    DiurnalRedistrictingReport,
};
use crate::network::ControlMode;
use crate::utilities::{evaluate_coupled_utility_grids, UtilityGridsConfig, UtilityGridsReport};
use crate::emergency::{EmergencyFacility, EmergencyFacilityKind};
use crate::emergency_shelters::{
    run_emergency_shelters_monte_carlo as exec_emergency_shelters_mc, EmergencySheltersConfig,
    EmergencySheltersReport,
};
use crate::ffe_conflagration::{
    run_ffe_conflagration_monte_carlo, FfeConfig, FfeMonteCarloReport,
};
use crate::aeromedical_medevac::{
    run_aeromedical_medevac_monte_carlo as exec_aeromedical_medevac_mc, AeromedicalMedevacConfig,
    AeromedicalMedevacReport,
};
use crate::commodity_resupply::{
    run_commodity_resupply_monte_carlo as exec_commodity_resupply_mc, CommodityResupplyConfig,
    CommodityResupplyReport,
};
use crate::transit_evacuation::{
    run_transit_bus_evacuation_monte_carlo as exec_transit_bus_evacuation_mc,
    TransitEvacuationConfig, TransitEvacuationReport,
};
use crate::fema_comms::{
    run_fema_comms_monte_carlo as exec_fema_comms_mc, FemaCommsConfig, FemaCommsReport,
};
use crate::fema_isb_pods::{
    run_fema_isb_pods_monte_carlo as exec_fema_isb_pods_mc, FemaIsbPodsConfig, FemaIsbPodsReport,
};
use crate::fema_usar_debris::{
    run_fema_usar_debris_monte_carlo as exec_fema_usar_debris_mc, FemaUsarDebrisConfig,
    FemaUsarDebrisReport,
};
use crate::fema_gsb_drc_tdms::{
    run_fema_gsb_drc_tdms_monte_carlo as exec_fema_gsb_drc_tdms_mc, FemaGsbDrcTdmsConfig,
    FemaGsbDrcTdmsReport,
};
use crate::wildfire_alertwest_ohaz::{
    run_wildfire_alertwest_monte_carlo as exec_wildfire_alertwest_mc, WildfireAlertWestConfig,
    WildfireAlertWestReport,
};
use crate::burn_scar_debris_flow::{
    run_burn_scar_debris_flow_monte_carlo as exec_burn_scar_debris_flow_mc,
    BurnScarDebrisFlowConfig, BurnScarDebrisFlowReport,
};
use crate::wtp_turbidity_storage::{
    run_wtp_turbidity_storage_monte_carlo as exec_wtp_turbidity_storage_mc,
    WtpTurbidityStorageConfig, WtpTurbidityStorageReport,
};
use crate::warning_mobilization::{
    run_warning_mobilization_monte_carlo as exec_warning_mobilization_mc,
    WarningMobilizationConfig, WarningMobilizationReport,
};
use crate::evacuation_contraflow::{
    run_evacuation_contraflow_monte_carlo as exec_evacuation_contraflow_mc,
    EvacuationContraflowConfig, EvacuationContraflowReport,
};
use crate::hospital_surge::{
    run_hospital_surge_monte_carlo as exec_hospital_surge_mc, HospitalSurgeConfig,
    HospitalSurgeMonteCarloReport,
};
use crate::hypercube::{
    optimize_network_response_districts, solve_network_hypercube, HypercubeConfig, HypercubeReport,
    RedistrictingConfig, RedistrictingReport,
};
use crate::mclp::{
    solve_mclp_fire_station_siting, MclpOptimizationReport, MclpSitingConfig,
};
use crate::resilience::{
    evaluate_resilience_stress_test, solve_robust_mclp_under_disruptions, ResilienceStressConfig,
    ResilienceStressReport,
};
use crate::seismic_mc::{
    run_seismic_monte_carlo_simulation, MonteCarloTrialSummary, RecoveryEpochSummary,
    SeismicMonteCarloReport, SeismicScenarioConfig,
};
use crate::multimodal::TransitPolicy;
use crate::network::{sample_polyline_fast, Junction, RoadNetwork, RoadSegment, Vec2};
use crate::network_edit::hit_test_segment;
use crate::traffic::{CarrierRef, SimulationWorld};
use crate::transit_gtfs::shortest_segment_chain;
use crate::zoning::{apply_tod_upzoning_policy, Parcel, ZoneCode};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const DEFAULT_MAX_HISTORY_ENTRIES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditCategory {
    DrawRoad,
    EditLanes,
    ConvertRoundabout,
    PaintZoning,
    TodUpzoning,
    CorridorPolicy,
    SignalControl,
    EmergencyFacility,
}

impl EditCategory {
    pub fn short_tag(self) -> &'static str {
        match self {
            Self::DrawRoad => "DrawRoad",
            Self::EditLanes => "EditLanes",
            Self::ConvertRoundabout => "Roundabout",
            Self::PaintZoning => "Zoning",
            Self::TodUpzoning => "TOD",
            Self::CorridorPolicy => "Corridor",
            Self::SignalControl => "Signals",
            Self::EmergencyFacility => "Facility",
        }
    }
}

/// Dual-tier state memento:
/// - `ZoningAndPolicy` restores parcel land-use & transit policy in `O(P)` without rebuilding lanes.
/// - `TopologyAndZoning` restores full street segments, junctions, parcels, and policies while
///   remapping live vehicles, cyclists, pedestrians, and GTFS stops/routes onto the target network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditStateSnapshot {
    ZoningAndPolicy {
        parcels: Vec<Parcel>,
        transit_policy: TransitPolicy,
        facilities: Vec<EmergencyFacility>,
    },
    TopologyAndZoning {
        junctions: Vec<Junction>,
        segments: Vec<RoadSegment>,
        parcels: Vec<Parcel>,
        transit_policy: TransitPolicy,
        facilities: Vec<EmergencyFacility>,
        watched_segment_id: usize,
    },
}

impl EditStateSnapshot {
    pub fn capture_zoning(world: &SimulationWorld) -> Self {
        Self::ZoningAndPolicy {
            parcels: world.parcels.clone(),
            transit_policy: world.transit_policy,
            facilities: world.emergency_facilities.clone(),
        }
    }

    pub fn capture_topology(world: &SimulationWorld) -> Self {
        Self::TopologyAndZoning {
            junctions: world.network.junctions.clone(),
            segments: world.network.segments.clone(),
            parcels: world.parcels.clone(),
            transit_policy: world.transit_policy,
            facilities: world.emergency_facilities.clone(),
            watched_segment_id: world.watched_segment_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditCommandRecord {
    pub id: u64,
    pub category: EditCategory,
    pub label: String,
    pub sim_time_s: f64,
    pub before: EditStateSnapshot,
    pub after: EditStateSnapshot,
}

#[derive(Debug, Clone)]
pub struct UndoRedoOutcome {
    pub id: u64,
    pub category: EditCategory,
    pub label: String,
    pub undo_remaining: usize,
    pub redo_remaining: usize,
}

#[derive(Debug, Clone)]
pub struct EditHistoryStack {
    undo_stack: VecDeque<EditCommandRecord>,
    redo_stack: Vec<EditCommandRecord>,
    max_entries: usize,
    next_cmd_id: u64,
}

impl Default for EditHistoryStack {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_HISTORY_ENTRIES)
    }
}

impl EditHistoryStack {
    pub fn new(max_entries: usize) -> Self {
        let cap = max_entries.max(4);
        Self {
            undo_stack: VecDeque::with_capacity(cap),
            redo_stack: Vec::new(),
            max_entries: cap,
            next_cmd_id: 1,
        }
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    #[inline]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    #[inline]
    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    #[inline]
    pub fn redo_len(&self) -> usize {
        self.redo_stack.len()
    }

    pub fn peek_undo(&self) -> Option<&EditCommandRecord> {
        self.undo_stack.back()
    }

    pub fn peek_redo(&self) -> Option<&EditCommandRecord> {
        self.redo_stack.last()
    }

    pub fn undo_entries(&self) -> &VecDeque<EditCommandRecord> {
        &self.undo_stack
    }

    pub fn redo_entries(&self) -> &[EditCommandRecord] {
        &self.redo_stack
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Commits an atomic `(before -> after)` transition onto the undo stack and clears any
    /// invalidated branching redo entries.
    pub fn push_transition(
        &mut self,
        category: EditCategory,
        label: impl Into<String>,
        sim_time_s: f64,
        before: EditStateSnapshot,
        after: EditStateSnapshot,
    ) -> u64 {
        let id = self.next_cmd_id;
        self.next_cmd_id = self.next_cmd_id.saturating_add(1);
        self.redo_stack.clear();
        if self.undo_stack.len() >= self.max_entries {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(EditCommandRecord {
            id,
            category,
            label: label.into(),
            sim_time_s,
            before,
            after,
        });
        id
    }
}

#[derive(Clone, Copy)]
struct LiveVehicleAnchor {
    seg_id: usize,
    lane_idx: usize,
    s_m: f64,
    pos: Vec2,
    heading_rad: f64,
}

#[derive(Clone, Copy)]
struct LiveCorridorAnchor {
    seg_id: usize,
    s_m: f64,
    pos: Vec2,
}

fn min_dist_to_polyline(pts: &[Vec2], p: Vec2) -> f64 {
    if pts.is_empty() {
        return f64::INFINITY;
    }
    if pts.len() == 1 {
        return pts[0].dist(p);
    }
    let mut best = f64::INFINITY;
    for w in pts.windows(2) {
        let dx = w[1].x - w[0].x;
        let dy = w[1].y - w[0].y;
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 1e-9 {
            (((p.x - w[0].x) * dx + (p.y - w[0].y) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let proj = Vec2 {
            x: w[0].x + dx * t,
            y: w[0].y + dy * t,
        };
        best = best.min(proj.dist(p));
    }
    best
}

fn find_heading_aligned_segment(
    net: &RoadNetwork,
    pos: Vec2,
    heading_rad: f64,
    max_dist_m: f64,
) -> Option<(usize, f64)> {
    let hx = heading_rad.cos();
    let hy = heading_rad.sin();
    let mut best: Option<(f64, usize, f64)> = None;
    for seg in &net.segments {
        if seg.closed || seg.lane_ids.is_empty() || seg.base_pts.len() < 2 {
            continue;
        }
        let mut cum = 0.0;
        for w in seg.base_pts.windows(2) {
            let dx = w[1].x - w[0].x;
            let dy = w[1].y - w[0].y;
            let l2 = dx * dx + dy * dy;
            let l = l2.sqrt().max(1e-6);
            let t = if l2 > 1e-9 {
                (((pos.x - w[0].x) * dx + (pos.y - w[0].y) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let px = w[0].x + dx * t;
            let py = w[0].y + dy * t;
            let d = (pos.x - px).hypot(pos.y - py);
            if d <= max_dist_m {
                let ux = dx / l;
                let uy = dy / l;
                let align_penalty = (1.0 - (ux * hx + uy * hy)).max(0.0) * 9.0;
                let score = d + align_penalty;
                if best.map(|(bs, _, _)| score < bs).unwrap_or(true) {
                    best = Some((score, seg.id, cum + t * l));
                }
            }
            cum += l;
        }
    }
    best.map(|(_, sid, s_m)| (sid, s_m))
}

/// Applies an `EditStateSnapshot` onto `world`, preserving live microscopic agents across
/// topological street-network and roundabout undo/redo transitions.
pub fn apply_edit_snapshot_to_world(world: &mut SimulationWorld, snap: &EditStateSnapshot) {
    match snap {
        EditStateSnapshot::ZoningAndPolicy {
            parcels,
            transit_policy,
            facilities,
        } => {
            world.parcels = parcels.clone();
            world.transit_policy = *transit_policy;
            world.emergency_facilities = facilities.clone();
            world.refresh_urban_metrics();
        }
        EditStateSnapshot::TopologyAndZoning {
            junctions,
            segments,
            parcels,
            transit_policy,
            facilities,
            watched_segment_id,
        } => {
            let veh_anchors: Vec<LiveVehicleAnchor> = world
                .vehicles
                .iter()
                .map(|v| match v.carrier {
                    CarrierRef::Lane(lid) => {
                        let lane = &world.network.lanes[lid];
                        let (pos, heading_rad) =
                            sample_polyline_fast(&lane.pts, &lane.cum_s, v.s_m);
                        LiveVehicleAnchor {
                            seg_id: lane.segment_id,
                            lane_idx: lane.index,
                            s_m: v.s_m,
                            pos,
                            heading_rad,
                        }
                    }
                    CarrierRef::Connector(cid) => {
                        let conn = &world.network.connectors[cid];
                        let (pos, heading_rad) =
                            sample_polyline_fast(&conn.pts, &conn.cum_s, v.s_m);
                        LiveVehicleAnchor {
                            seg_id: conn.to_seg_id,
                            lane_idx: 0,
                            s_m: 1.0,
                            pos,
                            heading_rad,
                        }
                    }
                })
                .collect();

            let cyc_anchors: Vec<LiveCorridorAnchor> = world
                .cyclists
                .iter()
                .map(|c| {
                    let seg = &world.network.segments[c.segment_id];
                    let (pos, _) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, c.s_m);
                    LiveCorridorAnchor {
                        seg_id: c.segment_id,
                        s_m: c.s_m,
                        pos,
                    }
                })
                .collect();

            let ped_anchors: Vec<LiveCorridorAnchor> = world
                .pedestrians
                .iter()
                .map(|p| {
                    let seg = &world.network.segments[p.segment_id];
                    let (pos, _) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, p.s_m);
                    LiveCorridorAnchor {
                        seg_id: p.segment_id,
                        s_m: p.s_m,
                        pos,
                    }
                })
                .collect();

            world.network.junctions = junctions.clone();
            world.network.segments = segments.clone();
            world.parcels = parcels.clone();
            world.transit_policy = *transit_policy;
            world.emergency_facilities = facilities.clone();
            world.network.rebuild_topology_from_segments(true);

            let n_segs = world.network.segments.len();
            let old_vehs = std::mem::take(&mut world.vehicles);
            let mut kept_vehs = Vec::with_capacity(old_vehs.len());

            for (mut veh, anc) in old_vehs.into_iter().zip(veh_anchors.into_iter()) {
                let resolved: Option<(usize, usize, f64)> = if anc.seg_id < n_segs
                    && !world.network.segments[anc.seg_id].closed
                    && !world.network.segments[anc.seg_id].lane_ids.is_empty()
                    && min_dist_to_polyline(&world.network.segments[anc.seg_id].base_pts, anc.pos)
                        <= 22.0
                {
                    Some((anc.seg_id, anc.lane_idx, anc.s_m))
                } else {
                    find_heading_aligned_segment(&world.network, anc.pos, anc.heading_rad, 38.0)
                        .map(|(sid, proj_s)| (sid, anc.lane_idx, proj_s))
                };

                if let Some((sid, lane_i, target_s)) = resolved {
                    let seg = &world.network.segments[sid];
                    let li = lane_i.min(seg.lane_ids.len() - 1);
                    let lid = seg.lane_ids[li];
                    let llen = world.network.lanes[lid].length;
                    veh.carrier = CarrierRef::Lane(lid);
                    veh.s_m = target_s.clamp(0.5, (llen - 0.5).max(0.5));
                    veh.planned_conn_id = None;
                    if let Some(dst) = veh.dest_seg_id {
                        if dst >= n_segs || !world.network.sinks.contains(&dst) {
                            veh.dest_seg_id = None;
                        }
                    }
                    kept_vehs.push(veh);
                }
            }
            world.vehicles = kept_vehs;

            // Remap cyclists & pedestrians onto restored topology
            let old_cycs = std::mem::take(&mut world.cyclists);
            for (mut cyc, anc) in old_cycs.into_iter().zip(cyc_anchors.into_iter()) {
                if anc.seg_id < n_segs
                    && !world.network.segments[anc.seg_id].closed
                    && world.network.segments[anc.seg_id].has_bike_lane
                    && min_dist_to_polyline(&world.network.segments[anc.seg_id].base_pts, anc.pos)
                        <= 24.0
                {
                    let len = world.network.segments[anc.seg_id].length.max(4.0);
                    cyc.segment_id = anc.seg_id;
                    cyc.s_m = anc.s_m.clamp(0.5, len - 0.5);
                    world.cyclists.push(cyc);
                } else if let Some((sid, s_m, _)) =
                    hit_test_segment(&world.network, anc.pos.x, anc.pos.y, 32.0)
                {
                    if !world.network.segments[sid].closed
                        && world.network.segments[sid].has_bike_lane
                    {
                        let len = world.network.segments[sid].length.max(4.0);
                        cyc.segment_id = sid;
                        cyc.s_m = s_m.clamp(0.5, len - 0.5);
                        world.cyclists.push(cyc);
                    }
                }
            }
            if world.cyclists.is_empty()
                && world.network.segments.iter().any(|s| s.has_bike_lane && !s.closed)
            {
                world.rebuild_cyclists();
            }

            let old_peds = std::mem::take(&mut world.pedestrians);
            for (mut ped, anc) in old_peds.into_iter().zip(ped_anchors.into_iter()) {
                if anc.seg_id < n_segs
                    && !world.network.segments[anc.seg_id].closed
                    && min_dist_to_polyline(&world.network.segments[anc.seg_id].base_pts, anc.pos)
                        <= 24.0
                {
                    let len = world.network.segments[anc.seg_id].length.max(4.0);
                    ped.segment_id = anc.seg_id;
                    ped.s_m = anc.s_m.clamp(0.5, len - 0.5);
                    world.pedestrians.push(ped);
                } else if let Some((sid, s_m, _)) =
                    hit_test_segment(&world.network, anc.pos.x, anc.pos.y, 32.0)
                {
                    if !world.network.segments[sid].closed {
                        let len = world.network.segments[sid].length.max(4.0);
                        ped.segment_id = sid;
                        ped.s_m = s_m.clamp(0.5, len - 0.5);
                        world.pedestrians.push(ped);
                    }
                }
            }

            for fac in &mut world.emergency_facilities {
                if fac.segment_id < n_segs && !world.network.segments[fac.segment_id].closed {
                    let len = world.network.segments[fac.segment_id].length.max(6.0);
                    fac.s_m = fac.s_m.clamp(3.0, (len - 3.0).max(3.0));
                } else if let Some((best_sid, best_s, _)) =
                    hit_test_segment(&world.network, fac.world_pos.x, fac.world_pos.y, 120.0)
                {
                    fac.segment_id = best_sid;
                    let len = world.network.segments[best_sid].length.max(6.0);
                    fac.s_m = best_s.clamp(3.0, (len - 3.0).max(3.0));
                }
            }

            // Repair GTFS stops, route segment chains, and active scheduled bus indices
            for stop in &mut world.transit_gtfs.stops {
                if stop.segment_id < n_segs && !world.network.segments[stop.segment_id].closed {
                    let len = world.network.segments[stop.segment_id].length.max(6.0);
                    stop.s_m = stop.s_m.clamp(3.0, (len - 3.0).max(3.0));
                } else if let Some((best_sid, best_s, _)) =
                    hit_test_segment(&world.network, stop.world_pos.x, stop.world_pos.y, 85.0)
                {
                    stop.segment_id = best_sid;
                    let len = world.network.segments[best_sid].length.max(6.0);
                    stop.s_m = best_s.clamp(3.0, (len - 3.0).max(3.0));
                }
            }
            for route in &mut world.transit_gtfs.routes {
                if route.route_stops.is_empty() {
                    continue;
                }
                let mut new_chain: Vec<usize> = Vec::new();
                let mut updated_stops: Vec<(usize, usize)> = Vec::new();
                for (idx, &(stop_id, _)) in route.route_stops.iter().enumerate() {
                    let t_seg = world.transit_gtfs.stops[stop_id].segment_id;
                    if idx == 0 {
                        new_chain.push(t_seg);
                        updated_stops.push((stop_id, 0));
                    } else {
                        let prev = *new_chain.last().unwrap();
                        let sub = shortest_segment_chain(&world.network, prev, t_seg);
                        if !sub.is_empty() {
                            for &s in sub.iter().skip(1) {
                                new_chain.push(s);
                            }
                            updated_stops.push((stop_id, new_chain.len() - 1));
                        }
                    }
                }
                if !new_chain.is_empty() {
                    route.segment_chain = new_chain;
                    route.route_stops = updated_stops;
                }
            }
            for veh in &mut world.vehicles {
                if let Some(r_idx) = veh.gtfs_route_idx {
                    if let Some(route) = world.transit_gtfs.routes.get(r_idx as usize) {
                        let cur_seg = match veh.carrier {
                            CarrierRef::Lane(lid) => world.network.lanes[lid].segment_id,
                            CarrierRef::Connector(cid) => world.network.connectors[cid].to_seg_id,
                        };
                        if let Some(pos_idx) = route.segment_chain.iter().position(|&s| s == cur_seg)
                        {
                            veh.gtfs_route_seq = pos_idx as u16;
                            veh.gtfs_next_stop_order = route
                                .route_stops
                                .iter()
                                .position(|&(_, chain_pos)| chain_pos >= pos_idx)
                                .unwrap_or(route.route_stops.len())
                                as u16;
                        } else {
                            veh.gtfs_route_idx = None;
                            veh.dwell_remaining_s = 0.0;
                        }
                    } else {
                        veh.gtfs_route_idx = None;
                    }
                }
            }

            world.watched_segment_id =
                (*watched_segment_id).min(world.network.segments.len().saturating_sub(1));
            world.refresh_urban_metrics();
        }
    }
}

impl SimulationWorld {
    /// Pops and applies the most recent command's `before` memento (`Ctrl+Z`).
    pub fn undo_edit(&mut self) -> Option<UndoRedoOutcome> {
        let record = self.edit_history.undo_stack.pop_back()?;
        apply_edit_snapshot_to_world(self, &record.before);
        let id = record.id;
        let category = record.category;
        let label = record.label.clone();
        self.edit_history.redo_stack.push(record);
        Some(UndoRedoOutcome {
            id,
            category,
            label,
            undo_remaining: self.edit_history.undo_len(),
            redo_remaining: self.edit_history.redo_len(),
        })
    }

    /// Pops and reapplies the most recently undone command's `after` memento (`Ctrl+Y` / `Ctrl+Shift+Z`).
    pub fn redo_edit(&mut self) -> Option<UndoRedoOutcome> {
        let record = self.edit_history.redo_stack.pop()?;
        apply_edit_snapshot_to_world(self, &record.after);
        let id = record.id;
        let category = record.category;
        let label = record.label.clone();
        if self.edit_history.undo_stack.len() >= self.edit_history.max_entries {
            self.edit_history.undo_stack.pop_front();
        }
        self.edit_history.undo_stack.push_back(record);
        Some(UndoRedoOutcome {
            id,
            category,
            label,
            undo_remaining: self.edit_history.undo_len(),
            redo_remaining: self.edit_history.redo_len(),
        })
    }

    /// Updates `parcels[parcel_id].zone` and records a lightweight `ZoningAndPolicy` undo entry
    /// whenever the parcel's zone actually changes.
    pub fn set_parcel_zone(&mut self, parcel_id: usize, new_zone: ZoneCode) -> bool {
        let Some(old_zone) = self.parcels.get(parcel_id).map(|p| p.zone) else {
            return false;
        };
        if old_zone == new_zone {
            return false;
        }
        let before = EditStateSnapshot::capture_zoning(self);
        self.parcels[parcel_id].zone = new_zone;
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::PaintZoning,
            format!(
                "Paint parcel #{parcel_id}: {:?} -> {:?}",
                old_zone, new_zone
            ),
            self.sim_time_s,
            before,
            after,
        );
        true
    }

    /// Hit-tests `(wx, wy)` against city parcels, updates the matching parcel to `new_zone`,
    /// and pushes an undoable `PaintZoning` command if the zone changed.
    pub fn paint_parcel_zone_at(&mut self, wx: f64, wy: f64, new_zone: ZoneCode) -> Option<usize> {
        let pid = self
            .parcels
            .iter()
            .find(|p| wx >= p.min.x && wx <= p.max.x && wy >= p.min.y && wy <= p.max.y)
            .map(|p| p.id)?;
        if self.set_parcel_zone(pid, new_zone) {
            Some(pid)
        } else {
            None
        }
    }

    /// Applies 15-minute Transit-Oriented Development (`TOD`) upzoning + BRT + protected bike grid
    /// as a single atomic undoable command.
    pub fn apply_tod_upzoning_tracked(&mut self) {
        let before = EditStateSnapshot::capture_topology(self);
        apply_tod_upzoning_policy(&mut self.parcels, &mut self.network);
        self.transit_policy.headway_min = 5.0;
        self.rebuild_cyclists();
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::TodUpzoning,
            "Apply 15-Min TOD Upzoning + BRT + Bike Grid",
            self.sim_time_s,
            before,
            after,
        );
    }

    /// Wraps a corridor or signal-control mutation in an atomic `TopologyAndZoning` history entry.
    pub fn execute_topology_edit<F>(
        &mut self,
        category: EditCategory,
        label: impl Into<String>,
        mutate_fn: F,
    ) where
        F: FnOnce(&mut SimulationWorld),
    {
        let before = EditStateSnapshot::capture_topology(self);
        mutate_fn(self);
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history
            .push_transition(category, label, self.sim_time_s, before, after);
    }

    /// Interactively places a new `FireStation` or `Hospital` near `(wx, wy)` snapped to the
    /// closest open roadway, records an undoable `EmergencyFacility` command, and recomputes
    /// NFPA 1710 4-minute coverage contours.
    pub fn place_emergency_facility_at(
        &mut self,
        wx: f64,
        wy: f64,
        kind: EmergencyFacilityKind,
    ) -> Option<usize> {
        let (seg_id, s_m, _) = hit_test_segment(&self.network, wx, wy, 180.0).or_else(|| {
            self.network
                .segments
                .iter()
                .filter(|s| !s.closed && !s.lane_ids.is_empty())
                .map(|s| {
                    let mid = s.base_pts[s.base_pts.len() / 2];
                    (s.id, s.length * 0.5, mid)
                })
                .min_by(|a, b| {
                    let da = a.2.dist(Vec2 { x: wx, y: wy });
                    let db = b.2.dist(Vec2 { x: wx, y: wy });
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
        })?;

        let before = EditStateSnapshot::capture_zoning(self);
        let seg = &self.network.segments[seg_id];
        let clamped_s = s_m.clamp(3.0, (seg.length - 3.0).max(3.0));
        let (pt, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, clamped_s);
        let curb_off = (seg.lane_count as f64 + 1.25) * crate::network::LANE_WIDTH_M;
        let world_pos = Vec2 {
            x: pt.x - ang.sin() * curb_off,
            y: pt.y + ang.cos() * curb_off,
        };
        let next_id = self
            .emergency_facilities
            .iter()
            .map(|f| f.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let count_kind = self
            .emergency_facilities
            .iter()
            .filter(|f| f.kind == kind)
            .count()
            + 1;
        let name = format!("{} #{count_kind}", kind.label());

        self.emergency_facilities.push(EmergencyFacility {
            id: next_id,
            kind,
            name: name.clone(),
            segment_id: seg_id,
            s_m: clamped_s,
            world_pos,
            heading_rad: ang,
        });
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!("Place {name} on corridor #{seg_id}"),
            self.sim_time_s,
            before,
            after,
        );
        Some(next_id)
    }

    /// Removes the closest `EmergencyFacility` within `max_dist_m` of `(wx, wy)` as an undoable command.
    pub fn remove_emergency_facility_near(
        &mut self,
        wx: f64,
        wy: f64,
        max_dist_m: f64,
    ) -> Option<EmergencyFacility> {
        let cursor = Vec2 { x: wx, y: wy };
        let (idx, _) = self
            .emergency_facilities
            .iter()
            .enumerate()
            .map(|(i, f)| (i, f.world_pos.dist(cursor)))
            .filter(|&(_, d)| d <= max_dist_m)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        let before = EditStateSnapshot::capture_zoning(self);
        let removed = self.emergency_facilities.remove(idx);
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!("Remove {}", removed.name),
            self.sim_time_s,
            before,
            after,
        );
        Some(removed)
    }

    /// Runs the `Rayon`-parallelized Church & ReVelle (1974) MCLP + 1-Swap `GAS` station-siting
    /// optimizer against the current street network, live traffic speeds, and parcel demographics.
    pub fn solve_mclp_station_siting(&self, cfg: &MclpSitingConfig) -> MclpOptimizationReport {
        solve_mclp_fire_station_siting(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        )
    }

    /// Commits an `MclpOptimizationReport`'s selected Fire Station placements onto `self` as a
    /// single atomic undoable command (`Ctrl+Z` / `Ctrl+Y`), preserving all placed Hospitals.
    pub fn apply_mclp_siting_solution(&mut self, report: &MclpOptimizationReport) -> usize {
        let before = EditStateSnapshot::capture_zoning(self);
        if !report.config.keep_existing_fire_stations {
            self.emergency_facilities
                .retain(|f| f.kind != EmergencyFacilityKind::FireStation);
        }
        let mut next_id = self
            .emergency_facilities
            .iter()
            .map(|f| f.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let base_fs_num = self
            .emergency_facilities
            .iter()
            .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
            .count();

        for (i, site) in report.selected_sites.iter().enumerate() {
            self.emergency_facilities.push(EmergencyFacility {
                id: next_id,
                kind: EmergencyFacilityKind::FireStation,
                name: if site.allocated_engines >= 2 {
                    format!(
                        "MEXCLP Station #{} ({}x Eng)",
                        base_fs_num + i + 1,
                        site.allocated_engines
                    )
                } else {
                    format!("Fire Station #{}", base_fs_num + i + 1)
                },
                segment_id: site.segment_id,
                s_m: site.s_m,
                world_pos: site.world_pos,
                heading_rad: site.heading_rad,
            });
            next_id += 1;
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!(
                "Apply MCLP Optimal Siting ({} Fire Stations -> {:.0}% <= {:.0}s)",
                report.optimal_total_stations,
                report.achieved_parcel_cov_pct,
                report.config.threshold_travel_s
            ),
            self.sim_time_s,
            before,
            after,
        );
        self.emergency_coverage.fire_station_count
    }

    /// Counts currently closed directed road segments on `self.network`.
    pub fn closed_segment_count(&self) -> usize {
        self.network.segments.iter().filter(|s| s.closed).count()
    }

    /// Sets `closed` on `segment_ids`, evacuates/reroutes active vehicles, cyclists, pedestrians,
    /// and GTFS bus chains onto open streets, recomputes dynamic routing & NFPA 1710 coverage, and
    /// pushes an atomic `CorridorPolicy` undo/redo snapshot (`Ctrl+Z` / `Ctrl+Y`).
    pub fn set_segments_closed_tracked(
        &mut self,
        segment_ids: &[usize],
        closed: bool,
        label: impl Into<String>,
    ) -> usize {
        let mut target_ids: Vec<usize> = segment_ids
            .iter()
            .copied()
            .filter(|&sid| sid < self.network.segments.len() && self.network.segments[sid].closed != closed)
            .collect();
        target_ids.sort_unstable();
        target_ids.dedup();
        if target_ids.is_empty() {
            return 0;
        }

        let before = EditStateSnapshot::capture_topology(self);
        for &sid in &target_ids {
            self.network.segments[sid].closed = closed;
        }

        if closed {
            // Relocate any active vehicles currently traversing a newly closed carriageway
            let old_vehs = std::mem::take(&mut self.vehicles);
            let mut kept_vehs = Vec::with_capacity(old_vehs.len());
            for mut veh in old_vehs {
                let on_closed = match veh.carrier {
                    CarrierRef::Lane(lid) => {
                        let sid = self.network.lanes[lid].segment_id;
                        self.network.segments[sid].closed
                    }
                    CarrierRef::Connector(cid) => {
                        let conn = &self.network.connectors[cid];
                        self.network.segments[conn.from_seg_id].closed
                            || self.network.segments[conn.to_seg_id].closed
                    }
                };
                if !on_closed {
                    kept_vehs.push(veh);
                } else {
                    let (pos, heading) = match veh.carrier {
                        CarrierRef::Lane(lid) => {
                            let l = &self.network.lanes[lid];
                            sample_polyline_fast(&l.pts, &l.cum_s, veh.s_m)
                        }
                        CarrierRef::Connector(cid) => {
                            let c = &self.network.connectors[cid];
                            sample_polyline_fast(&c.pts, &c.cum_s, veh.s_m)
                        }
                    };
                    if let Some((new_sid, proj_s)) =
                        find_heading_aligned_segment(&self.network, pos, heading, 90.0)
                    {
                        let seg = &self.network.segments[new_sid];
                        if !seg.lane_ids.is_empty() {
                            let lid = seg.lane_ids[0];
                            let llen = self.network.lanes[lid].length;
                            veh.carrier = CarrierRef::Lane(lid);
                            veh.s_m = proj_s.clamp(0.5, (llen - 0.5).max(0.5));
                            veh.planned_conn_id = None;
                            kept_vehs.push(veh);
                        }
                    }
                }
            }
            self.vehicles = kept_vehs;
            self.cyclists
                .retain(|c| !self.network.segments[c.segment_id].closed);
            self.pedestrians
                .retain(|p| !self.network.segments[p.segment_id].closed);
        }

        self.network.recompute_dynamic_routes();

        // Repair GTFS bus chains around any newly closed or reopened arterial corridor
        for route in &mut self.transit_gtfs.routes {
            if route.route_stops.is_empty() {
                continue;
            }
            let mut new_chain: Vec<usize> = Vec::new();
            let mut updated_stops: Vec<(usize, usize)> = Vec::new();
            for (idx, &(stop_id, _)) in route.route_stops.iter().enumerate() {
                let t_seg = self.transit_gtfs.stops[stop_id].segment_id;
                if self.network.segments[t_seg].closed {
                    continue;
                }
                if new_chain.is_empty() || idx == 0 {
                    new_chain.push(t_seg);
                    updated_stops.push((stop_id, new_chain.len() - 1));
                } else {
                    let prev = *new_chain.last().unwrap();
                    let sub = shortest_segment_chain(&self.network, prev, t_seg);
                    if !sub.is_empty() {
                        for &s in sub.iter().skip(1) {
                            new_chain.push(s);
                        }
                        updated_stops.push((stop_id, new_chain.len() - 1));
                    }
                }
            }
            if !new_chain.is_empty() {
                route.segment_chain = new_chain;
                route.route_stops = updated_stops;
            }
        }

        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            label,
            self.sim_time_s,
            before,
            after,
        );
        target_ids.len()
    }

    /// Toggles road/bridge closure at `(wx, wy)` (closing both directions when `bidirectional` is
    /// true), returning `(primary_segment_id, new_closed_state, corridor_name)`.
    pub fn toggle_corridor_closure_at(
        &mut self,
        wx: f64,
        wy: f64,
        bidirectional: bool,
        force_state: Option<bool>,
    ) -> Option<(usize, bool, String)> {
        let (sid, _, _) = hit_test_segment(&self.network, wx, wy, 20.0)?;
        let cur_closed = self.network.segments[sid].closed;
        let next_closed = force_state.unwrap_or(!cur_closed);
        let name = self.network.segments[sid].name.clone();
        let mut ids = vec![sid];
        if bidirectional {
            let u = self.network.segments[sid].from_junc;
            let v = self.network.segments[sid].to_junc;
            for s in &self.network.segments {
                if s.id != sid && s.from_junc == v && s.to_junc == u {
                    ids.push(s.id);
                }
            }
        }
        let action_word = if next_closed { "Close" } else { "Reopen" };
        let changed = self.set_segments_closed_tracked(
            &ids,
            next_closed,
            format!("{action_word} corridor `{name}` (#{sid})"),
        );
        if changed > 0 {
            Some((sid, next_closed, name))
        } else {
            None
        }
    }

    /// Reopens all currently closed corridors across the network as a single undoable action.
    pub fn clear_all_road_closures_tracked(&mut self) -> usize {
        let closed_ids: Vec<usize> = self
            .network
            .segments
            .iter()
            .filter(|s| s.closed)
            .map(|s| s.id)
            .collect();
        self.set_segments_closed_tracked(
            &closed_ids,
            false,
            format!("Reopen all {} closed road segments", closed_ids.len()),
        )
    }

    /// Runs the `Rayon`-parallelized arterial/bridge closure stress test (`N-1` & `N-2` cut-sets)
    /// against the active Fire Station layout on `self`.
    pub fn run_resilience_stress_test(
        &self,
        cfg: &ResilienceStressConfig,
    ) -> ResilienceStressReport {
        evaluate_resilience_stress_test(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        )
    }

    /// Solves a Disruption-Hardened (Minimax-Fortified) MCLP Fire Station siting plan that accounts
    /// for worst-case arterial/bridge closures across the current street network.
    pub fn solve_disruption_hardened_mclp(
        &self,
        mclp_cfg: &MclpSitingConfig,
        stress_cfg: &ResilienceStressConfig,
    ) -> MclpOptimizationReport {
        solve_robust_mclp_under_disruptions(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            mclp_cfg,
            stress_cfg,
        )
    }

    /// Runs `Rayon`-parallelized spatially correlated seismic ground-motion & Hazus bridge
    /// fragility Monte Carlo simulations against `self`.
    pub fn run_seismic_monte_carlo(&self, cfg: &SeismicScenarioConfig) -> SeismicMonteCarloReport {
        run_seismic_monte_carlo_simulation(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        )
    }

    /// Injects a specific seismic Monte Carlo realization's bridge closures onto the live network
    /// as an atomic undoable command (`Ctrl+Z` / `Ctrl+Y`).
    pub fn inject_seismic_realization_tracked(
        &mut self,
        trial: &MonteCarloTrialSummary,
        label: impl Into<String>,
    ) -> usize {
        let currently_closed: Vec<usize> = self
            .network
            .segments
            .iter()
            .filter(|s| s.closed)
            .map(|s| s.id)
            .collect();
        if !currently_closed.is_empty() {
            for sid in currently_closed {
                self.network.segments[sid].closed = false;
            }
        }
        self.set_segments_closed_tracked(&trial.closed_segment_ids, true, label)
    }

    /// Injects the 10th-percentile post-earthquake bridge closure state at a specific Hazus
    /// recovery horizon (`Day 0, 1, 7, 30, 90`) onto the live street network as a single atomic
    /// undoable command (`Ctrl+Z` / `Ctrl+Y`). If all bridges have reopened by `epoch.day`,
    /// reopens all closed segments cleanly!
    pub fn inject_recovery_epoch_tracked(&mut self, epoch: &RecoveryEpochSummary) -> usize {
        if epoch.p10_closed_segment_ids.is_empty() {
            return self.clear_all_road_closures_tracked();
        }
        let before = EditStateSnapshot::capture_topology(self);
        for s in &mut self.network.segments {
            s.closed = false;
        }
        for &sid in &epoch.p10_closed_segment_ids {
            if sid < self.network.segments.len() {
                self.network.segments[sid].closed = true;
            }
        }
        self.vehicles.retain(|v| match v.carrier {
            CarrierRef::Lane(lid) => !self.network.segments[self.network.lanes[lid].segment_id].closed,
            CarrierRef::Connector(cid) => {
                let c = &self.network.connectors[cid];
                !self.network.segments[c.from_seg_id].closed
                    && !self.network.segments[c.to_seg_id].closed
            }
        });
        self.cyclists
            .retain(|c| !self.network.segments[c.segment_id].closed);
        self.pedestrians
            .retain(|p| !self.network.segments[p.segment_id].closed);
        self.network.recompute_dynamic_routes();
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "Hazus Recovery {} P10 State ({} segs closed)",
                epoch.label,
                epoch.p10_closed_segment_ids.len()
            ),
            self.sim_time_s,
            before,
            after,
        );
        epoch.p10_closed_segment_ids.len()
    }

    /// Solves the exact Larson (1974) `2^N` `M/M/N` spatial hypercube queueing state-transition
    /// equations across all currently placed `FireStation`s on `self`, returning heterogeneous
    /// per-unit busy fractions `rho_n`, cross-district backup shares, and queue delay metrics.
    pub fn solve_active_fleet_hypercube(&self, cfg: &HypercubeConfig) -> HypercubeReport {
        solve_network_hypercube(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        )
    }

    /// Optimizes Carter, Chaiken & Ignall (1972) / Larson (1974) hypercube response-district
    /// boundaries across all placed `FireStation`s on `self`.
    pub fn optimize_response_districts(&self, cfg: &RedistrictingConfig) -> RedistrictingReport {
        optimize_network_response_districts(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        )
    }

    /// Commits a `RedistrictingReport`'s per-parcel primary response districts onto `self.parcels`
    /// as a single atomic undoable command (`Ctrl+Z` / `Ctrl+Y`), returning the count of parcels
    /// shifted relative to closest-unit Voronoi beats.
    pub fn apply_redistricting_plan_tracked(&mut self, rep: &RedistrictingReport) -> usize {
        if rep.optimized_parcel_units.len() != self.parcels.len() || rep.num_units == 0 {
            return 0;
        }
        let before = EditStateSnapshot::capture_zoning(self);
        for (j, p) in self.parcels.iter_mut().enumerate() {
            let u_idx = rep.optimized_parcel_units[j];
            p.response_district_unit = Some(u_idx as u16);
            p.response_district_station_seg = rep
                .unit_summaries
                .get(u_idx)
                .map(|s| s.station_segment_id as u32);
        }
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!(
                "Carter-Larson Redistrict ({} units, {} beats shifted)",
                rep.num_units, rep.reassigned_parcel_count
            ),
            self.sim_time_s,
            before,
            after,
        );
        rep.reassigned_parcel_count
    }

    /// Clears custom parcel response-district assignments back to closest-unit Voronoi (`None`)
    /// as an atomic undoable command (`Ctrl+Z` / `Ctrl+Y`).
    pub fn clear_response_districts_tracked(&mut self) -> usize {
        let assigned = self
            .parcels
            .iter()
            .filter(|p| p.response_district_unit.is_some())
            .count();
        if assigned == 0 {
            return 0;
        }
        let before = EditStateSnapshot::capture_zoning(self);
        for p in &mut self.parcels {
            p.response_district_unit = None;
            p.response_district_station_seg = None;
            p.peak_district_unit = None;
            p.peak_district_station_seg = None;
            p.offpeak_district_unit = None;
            p.offpeak_district_station_seg = None;
        }
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!("Reset {assigned} Parcels to Voronoi Response Beats"),
            self.sim_time_s,
            before,
            after,
        );
        assigned
    }

    /// Computes 24-hour diurnal emergency call & BPR rush-hour travel-time profiles and optimizes
    /// separate **Peak Rush** and **Off-Peak Free-Flow** response-district dispatch tables.
    pub fn solve_diurnal_dispatch_tables(
        &self,
        cfg: &DiurnalProfileConfig,
    ) -> DiurnalRedistrictingReport {
        solve_diurnal_peak_offpeak_dispatch_tables(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        )
    }

    /// Commits both the **Peak Rush** and **Off-Peak Free-Flow** response-district dispatch tables
    /// onto `self.parcels` and activates `initial_regime` on the live map as an atomic undoable
    /// command (`Ctrl+Z` / `Ctrl+Y`), returning the count of parcels that switch beats between
    /// Off-Peak and Peak!
    pub fn apply_diurnal_dispatch_tables_tracked(
        &mut self,
        rep: &DiurnalRedistrictingReport,
        initial_regime: DiurnalDispatchRegime,
    ) -> usize {
        if rep.num_units == 0 || rep.num_parcels != self.parcels.len() {
            return 0;
        }
        let before = EditStateSnapshot::capture_zoning(self);
        for (j, p) in self.parcels.iter_mut().enumerate() {
            let pk_u = rep.peak_table.parcel_primary_units[j];
            let op_u = rep.offpeak_table.parcel_primary_units[j];
            let pk_seg = rep
                .peak_table
                .redistricting_report
                .unit_summaries
                .get(pk_u)
                .map(|s| s.station_segment_id as u32);
            let op_seg = rep
                .offpeak_table
                .redistricting_report
                .unit_summaries
                .get(op_u)
                .map(|s| s.station_segment_id as u32);

            p.peak_district_unit = Some(pk_u as u16);
            p.peak_district_station_seg = pk_seg;
            p.offpeak_district_unit = Some(op_u as u16);
            p.offpeak_district_station_seg = op_seg;

            match initial_regime {
                DiurnalDispatchRegime::PeakRush => {
                    p.response_district_unit = Some(pk_u as u16);
                    p.response_district_station_seg = pk_seg;
                }
                DiurnalDispatchRegime::OffPeakFreeFlow => {
                    p.response_district_unit = Some(op_u as u16);
                    p.response_district_station_seg = op_seg;
                }
            }
        }
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!(
                "Diurnal Peak/Off-Peak Dispatch Tables ({} beat switches, {} active)",
                rep.primary_beat_switches_count,
                initial_regime.short_tag()
            ),
            self.sim_time_s,
            before,
            after,
        );
        rep.primary_beat_switches_count
    }

    /// Toggles the active parcel `response_district_unit` map between the stored **Peak Rush** and
    /// **Off-Peak Free-Flow** dispatch tables as an undoable transition (`Ctrl+Z` / `Ctrl+Y`).
    pub fn activate_diurnal_regime_tracked(&mut self, regime: DiurnalDispatchRegime) -> usize {
        let has_tables = self
            .parcels
            .iter()
            .any(|p| p.peak_district_unit.is_some() && p.offpeak_district_unit.is_some());
        if !has_tables {
            return 0;
        }
        let before = EditStateSnapshot::capture_zoning(self);
        let mut changed = 0usize;
        for p in &mut self.parcels {
            let (target_u, target_seg) = match regime {
                DiurnalDispatchRegime::PeakRush => (p.peak_district_unit, p.peak_district_station_seg),
                DiurnalDispatchRegime::OffPeakFreeFlow => {
                    (p.offpeak_district_unit, p.offpeak_district_station_seg)
                }
            };
            if let Some(u) = target_u {
                if p.response_district_unit != Some(u) {
                    changed += 1;
                }
                p.response_district_unit = Some(u);
                p.response_district_station_seg = target_seg;
            }
        }
        let after = EditStateSnapshot::capture_zoning(self);
        self.edit_history.push_transition(
            EditCategory::EmergencyFacility,
            format!("Switch Active Dispatch Table -> {} ({changed} beats updated)", regime.short_tag()),
            self.sim_time_s,
            before,
            after,
        );
        changed
    }

    /// Evaluates the coupled **Electrical (`12.47 kV` DistFlow)**, **Potable Water (`Hazen-Williams`)**,
    /// and **Wastewater (`Manning + RDII`)** grids with `cfg` and updates `self.utility_report`.
    pub fn evaluate_utility_grids(&mut self, cfg: &UtilityGridsConfig) -> UtilityGridsReport {
        self.utility_config = cfg.clone();
        let rep = evaluate_coupled_utility_grids(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            self.diurnal_clock_hour,
            &self.utility_config,
        );
        self.utility_report = rep.clone();
        rep
    }

    /// Propagates electrical blackout signal darkening onto live intersection controllers by
    /// degrading unpowered signalized intersections (`signal_darkened == true`) to 4-way stop
    /// `ControlMode::PermissiveYield` as an atomic undoable command (`Ctrl+Z` / `Ctrl+Y`)!
    pub fn apply_darkened_signal_fallback_tracked(&mut self) -> usize {
        let rep = evaluate_coupled_utility_grids(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            self.diurnal_clock_hour,
            &self.utility_config,
        );
        self.utility_report = rep.clone();
        let darkened_ids: Vec<usize> = rep
            .node_telemetry
            .iter()
            .filter(|n| n.signal_darkened)
            .map(|n| n.junction_id)
            .collect();
        if darkened_ids.is_empty() {
            return 0;
        }
        let before = EditStateSnapshot::capture_topology(self);
        let mut degraded = 0usize;
        for jid in darkened_ids {
            if let Some(j) = self.network.junctions.get_mut(jid) {
                if j.control_mode != ControlMode::PermissiveYield
                    && j.control_mode != ControlMode::Roundabout
                {
                    j.control_mode = ControlMode::PermissiveYield;
                    degraded += 1;
                }
            }
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::SignalControl,
            format!("Grid Outage Cascade: {degraded} Darkened Signals -> 4-Way Yield"),
            self.sim_time_s,
            before,
            after,
        );
        degraded
    }

    /// Runs `Rayon`-parallelized Scawthorn / Hazus **Fire-Following-Earthquake (`FFE`)** ignition &
    /// Hamada wind-driven conflagration Monte Carlo simulations and caches per-parcel burn &
    /// dry-hydrant probabilities onto `self.parcels` for GPU heatmap rendering.
    pub fn run_ffe_monte_carlo(&mut self, cfg: &FfeConfig) -> FfeMonteCarloReport {
        let rep = run_ffe_conflagration_monte_carlo(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        );
        for (p, r) in self.parcels.iter_mut().zip(rep.parcel_risks.iter()) {
            p.ffe_burn_prob_pct = r.burn_prob_pct as f32;
            p.ffe_dry_hydrant_prob_pct = r.dry_hydrant_prob_pct as f32;
        }
        rep
    }

    /// Runs `Rayon`-parallelized post-earthquake Hospital ED Surge, coupled FFE Burn/Trauma Priority
    /// Triage Queueing, and Dynamic Ambulance Diversion Monte Carlo simulations, caching per-parcel
    /// time-to-treatment & diversion probabilities onto `self.parcels` for GPU heatmap rendering.
    pub fn run_hospital_surge_monte_carlo(
        &mut self,
        cfg: &HospitalSurgeConfig,
    ) -> HospitalSurgeMonteCarloReport {
        let rep = exec_hospital_surge_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        );
        for (p, r) in self.parcels.iter_mut().zip(rep.parcel_rows.iter()) {
            p.surge_red_burn_treat_min = r.mean_red_burn_treat_min as f32;
            p.surge_diversion_prob_pct = r.diversion_prob_pct as f32;
        }
        rep
    }

    /// Runs `Rayon`-parallelized post-earthquake Rotary-Wing Medevac, Koschmieder FFE Smoke/Weather
    /// Flight-Envelope, and Secondary Inter-Hospital Burn-Center Transfer Monte Carlo simulations,
    /// caching per-parcel smoke visibility & airlift time savings onto `self.parcels`.
    pub fn run_aeromedical_medevac_mc(
        &mut self,
        cfg: &AeromedicalMedevacConfig,
    ) -> AeromedicalMedevacReport {
        let rep = exec_aeromedical_medevac_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_smoke_vis_sm.len() {
                p.medevac_smoke_vis_sm = rep.parcel_smoke_vis_sm[i] as f32;
                p.medevac_airlift_save_min = rep.parcel_airlift_save_min[i] as f32;
            }
        }
        rep
    }

    /// Runs `Rayon`-parallelized post-earthquake + active `FFE` conflagration perimeter road-closure,
    /// smoke speed-reduction, staged evacuation, and arterial contraflow Monte Carlo simulations.
    pub fn run_evacuation_contraflow_mc(
        &mut self,
        cfg: &EvacuationContraflowConfig,
    ) -> EvacuationContraflowReport {
        let rep = exec_evacuation_contraflow_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.evp_config,
            cfg,
        );
        for (p, r) in self.parcels.iter_mut().zip(rep.parcel_rows.iter()) {
            p.evac_stage_tier = r.evac_stage;
            p.evac_clearance_min = r.mean_clearance_min as f32;
        }
        rep
    }

    /// Applies an `EvacuationContraflowReport` directly onto the live microscopic `RoadNetwork`
    /// (reallocating contraflow outbound + donor inbound lanes while preserving EMS-protected spines,
    /// closing thermally overrun segments, and scaling smoke speeds) as a `Ctrl+Z` undoable command!
    pub fn apply_evacuation_contraflow_to_network(
        &mut self,
        rep: &EvacuationContraflowReport,
    ) -> (usize, usize) {
        let before = EditStateSnapshot::capture_topology(self);
        let mut reversed_segs = 0usize;
        let mut thermal_closed_segs = 0usize;

        for c in &rep.corridor_rows {
            if c.segment_id >= self.network.segments.len() {
                continue;
            }
            if c.thermal_closure_prob_pct >= 50.0 {
                self.network.segments[c.segment_id].closed = true;
                thermal_closed_segs += 1;
                continue;
            }
            if c.contraflow_added_lanes > 0 {
                let new_out = (c.base_lanes + c.contraflow_added_lanes).clamp(1, 8);
                self.network.segments[c.segment_id].lane_count = new_out;
                reversed_segs += 1;
            } else if c.remaining_inbound_lanes < c.base_lanes {
                if c.remaining_inbound_lanes == 0 {
                    self.network.segments[c.segment_id].closed = true;
                } else {
                    self.network.segments[c.segment_id].lane_count = c.remaining_inbound_lanes;
                }
            }
            let v_fac = c.mean_smoke_speed_factor.clamp(0.35, 1.0);
            self.network.segments[c.segment_id].speed_mps =
                (self.network.segments[c.segment_id].speed_mps * v_fac).clamp(3.5, 33.0);
            self.network.segments[c.segment_id].ema_speed_mps =
                self.network.segments[c.segment_id].speed_mps;
        }

        crate::network_edit::rebuild_world_after_network_edit(self, None);
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "Apply {} Contraflow ({reversed_segs} reversed, {thermal_closed_segs} FFE closed)",
                rep.config.contraflow_policy.short_label()
            ),
            self.sim_time_s,
            before,
            after,
        );
        (reversed_segs, thermal_closed_segs)
    }

    /// Runs Two-Stage Post-Earthquake Emergency Shelter Siting, ATC-20/Lifeline Habitability,
    /// `MERV-13`/`HEPA` Smoke Filtration, and Capacitated Evacuee Assignment Monte Carlo simulations,
    /// caching per-parcel clean-air shelter coverage & indoor `PM2.5` onto `self.parcels`.
    pub fn run_emergency_shelters_mc(
        &mut self,
        cfg: &EmergencySheltersConfig,
    ) -> EmergencySheltersReport {
        let rep = exec_emergency_shelters_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (p, r) in self.parcels.iter_mut().zip(rep.parcel_rows.iter()) {
            p.shelter_clean_air_pct = r.sheltered_clean_air_pct as f32;
            p.shelter_indoor_pm25 = r.indoor_pm25_at_assigned_ug_m3 as f32;
        }
        rep
    }

    /// Runs `Rayon`-parallelized Multi-Day Shelter & Hospital Commodity Resupply Inventory-Routing
    /// (`IRP` / Clarke-Wright 2-Opt `SDVRP`) Monte Carlo simulations and caches supply security &
    /// deprivation indices onto `self.parcels` for GPU heatmap rendering.
    pub fn run_commodity_resupply_mc(
        &mut self,
        cfg: &CommodityResupplyConfig,
    ) -> CommodityResupplyReport {
        let rep = exec_commodity_resupply_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_security_pct.len() {
                p.resupply_security_pct = rep.parcel_security_pct[i] as f32;
                p.resupply_deprivation_idx = rep.parcel_deprivation_idx[i] as f32;
            }
        }
        rep
    }

    /// Runs `Rayon`-parallelized GTFS Transit-Bus & ADA Paratransit Evacuation Monte Carlo (`TBEP`)
    /// coupled to `self.transit_gtfs` neighborhood pickup depots, Bus+EMS contraflow lanes, and
    /// `MERV-13 + HEPA` clean-air shelters, caching per-parcel clearance & wait metrics onto `self.parcels`.
    pub fn run_transit_bus_evacuation_mc(
        &mut self,
        cfg: &TransitEvacuationConfig,
    ) -> TransitEvacuationReport {
        let rep = exec_transit_bus_evacuation_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            &self.transit_gtfs,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_cleared_pct.len() {
                p.transit_evac_cleared_pct = rep.parcel_cleared_pct[i] as f32;
                p.transit_evac_wait_min = rep.parcel_wait_min[i] as f32;
            }
        }
        rep
    }

    /// Commits a `TransitEvacuationReport` Bus-Plus-EMS Contraflow configuration directly onto the
    /// live microscopic `RoadNetwork` (activating `has_bus_lane = true` on priority shuttle/EMS spines,
    /// reallocating contraflow lanes, switching `transit_gtfs` to `AllDoorFareFreeBrt` emergency boarding,
    /// and closing thermally overrun `FFE` segments) as an atomic `Ctrl+Z`/`Ctrl+Y` undoable command!
    pub fn apply_bus_ems_contraflow_to_network(
        &mut self,
        rep: &TransitEvacuationReport,
    ) -> (usize, usize) {
        let before = EditStateSnapshot::capture_topology(self);
        let mut bus_ems_segs = 0usize;
        let mut contraflow_segs = 0usize;

        for cs in &rep.contraflow_segments {
            if cs.segment_id >= self.network.segments.len() {
                continue;
            }
            if cs.ffe_thermal_closure_prob_pct >= 55.0 && !cs.ems_hospital_spine {
                self.network.segments[cs.segment_id].closed = true;
                continue;
            }
            if cs.bus_ems_priority_lane {
                self.network.segments[cs.segment_id].has_bus_lane = true;
                bus_ems_segs += 1;
            }
            if cs.contraflow_added_lanes > 0 {
                let new_lanes = (cs.base_lanes + cs.contraflow_added_lanes).clamp(1, 8);
                self.network.segments[cs.segment_id].lane_count = new_lanes;
                contraflow_segs += 1;
            }
        }
        self.transit_gtfs.fare_mode = rep.config.fare_mode;
        self.transit_gtfs.coach_spec = rep.config.coach_spec;

        crate::network_edit::rebuild_world_after_network_edit(self, None);
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "Apply {} ({bus_ems_segs} Bus+EMS segs, {contraflow_segs} reversed)",
                rep.config.lane_mode.short_label()
            ),
            self.sim_time_s,
            before,
            after,
        );
        (bus_ems_segs, contraflow_segs)
    }

    /// Runs `Rayon`-parallelized Post-Earthquake Cellular Tower Blackout Fragility, `WEA` vs.
    /// Outdoor Siren & Hi-Lo/Door-to-Door Warning Mobilization Curves, and Delayed Departure
    /// Staging Monte Carlo simulations, caching parcel mobilization metrics onto `self.parcels`.
    pub fn run_warning_mobilization_mc(
        &mut self,
        cfg: &WarningMobilizationConfig,
    ) -> WarningMobilizationReport {
        let rep = exec_warning_mobilization_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_mobilized_60m_pct.len() {
                p.warning_mobilized_60m_pct = rep.parcel_mobilized_60m_pct[i] as f32;
                p.warning_departure_delay_min = rep.parcel_departure_delay_min[i] as f32;
            }
        }
        rep
    }

    /// Commits a `WarningMobilizationReport`'s Blackout-Compensated Departure Staging Tiers (`1..=3`)
    /// and delayed departure offsets directly onto `self.parcels` as an atomic `Ctrl+Z` undoable edit!
    pub fn apply_warning_staging_to_parcels(
        &mut self,
        rep: &WarningMobilizationReport,
    ) -> usize {
        let before = EditStateSnapshot::capture_topology(self);
        let mut updated = 0usize;
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_recommended_stage.len() {
                p.evac_stage_tier = rep.parcel_recommended_stage[i];
                p.evac_clearance_min = rep.parcel_departure_delay_min[i] as f32;
                p.warning_mobilized_60m_pct = rep.parcel_mobilized_60m_pct[i] as f32;
                p.warning_departure_delay_min = rep.parcel_departure_delay_min[i] as f32;
                updated += 1;
            }
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::PaintZoning,
            format!(
                "Apply {} Warning Staging ({updated} parcels)",
                rep.config.staging_policy.short_label()
            ),
            self.sim_time_s,
            before,
            after,
        );
        updated
    }

    /// Runs `Rayon`-parallelized FEMA Communications Architecture (`IPAWS-OPEN CAP v1.2`, `WEA 3.0`,
    /// `NPWS`/`PEP` `EAS`, `NOAA NWR SAME`, `FNARS` HF, and Region 10 `MERS` `ESF #2` `NIFOG` LMR)
    /// Monte Carlo simulations and caches parcel IPAWS & ESF-2 LMR coverage onto `self.parcels`.
    pub fn run_fema_comms_mc(&mut self, cfg: &FemaCommsConfig) -> FemaCommsReport {
        let rep = exec_fema_comms_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_ipaws_reach_pct.len() {
                p.fema_ipaws_reach_pct = rep.parcel_ipaws_reach_pct[i] as f32;
                p.fema_esf2_lmr_pct = rep.parcel_esf2_lmr_ok_pct[i] as f32;
            }
        }
        rep
    }

    /// Commits a `FemaCommsReport` `IPAWS` + `MERS` `ESF #2` field posture onto `self.parcels`
    /// (updating parcel mobilization & `IPAWS`/`LMR` readiness) as a `Ctrl+Z` undoable command!
    pub fn apply_fema_mers_posture_tracked(&mut self, rep: &FemaCommsReport) -> usize {
        let before = EditStateSnapshot::capture_topology(self);
        let mut updated = 0usize;
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_ipaws_reach_pct.len() {
                p.fema_ipaws_reach_pct = rep.parcel_ipaws_reach_pct[i] as f32;
                p.fema_esf2_lmr_pct = rep.parcel_esf2_lmr_ok_pct[i] as f32;
                p.warning_mobilized_60m_pct = rep.parcel_ipaws_reach_pct[i] as f32;
                updated += 1;
            }
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::PaintZoning,
            format!(
                "Deploy FEMA {} ({} MERS units, {updated} parcels)",
                rep.config.posture.short_label(),
                rep.deployed_mers_units_count
            ),
            self.sim_time_s,
            before,
            after,
        );
        updated
    }

    /// Runs `Rayon`-parallelized FEMA Incident Staging Base (`ISB`), USACE `IS-26` `Type I/II/III`
    /// & Pedestrian `C-POD` `M/G/c` Queueing + Spillback, and `IPAWS` Public `POD` Routing Monte Carlo,
    /// caching per-parcel POD access time & service completion onto `self.parcels`.
    pub fn run_fema_isb_pods_mc(&mut self, cfg: &FemaIsbPodsConfig) -> FemaIsbPodsReport {
        let rep = exec_fema_isb_pods_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_total_access_min.len() {
                p.pod_total_access_min = rep.parcel_total_access_min[i] as f32;
                p.pod_served_pct = rep.parcel_served_pct[i] as f32;
            }
        }
        rep
    }

    /// Injects `C-POD` vehicular queue street-spillback speed degradation from a `FemaIsbPodsReport`
    /// directly onto adjacent live `RoadSegment`s as a `Ctrl+Z`/`Ctrl+Y` undoable topology edit!
    pub fn apply_pod_spillback_to_network(&mut self, rep: &FemaIsbPodsReport) -> usize {
        let before = EditStateSnapshot::capture_topology(self);
        let mut affected = 0usize;
        for (sid, &fac) in rep.segment_spillback_speed_factor.iter().enumerate() {
            if sid < self.network.segments.len() && fac < 0.98 {
                let new_v = (self.network.segments[sid].speed_mps * fac).clamp(2.5, 33.0);
                self.network.segments[sid].speed_mps = new_v;
                self.network.segments[sid].ema_speed_mps = new_v;
                affected += 1;
            }
        }
        crate::network_edit::rebuild_world_after_network_edit(self, None);
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "Inject {} POD Queue Spillback ({affected} segs slowed)",
                rep.config.routing_policy.short_label()
            ),
            self.sim_time_s,
            before,
            after,
        );
        affected
    }

    /// Runs `Rayon`-parallelized FEMA `Type 1 US&R` (`USNG` structural collapse search grids +
    /// Coburn-Spence `72-hr` survival), `ESF #3` Heavy-Equipment Phase I Debris Push, and `ISB -> C-POD`
    /// Route Reopening Monte Carlo, caching parcel survival & rubble metrics onto `self.parcels`.
    pub fn run_fema_usar_debris_mc(&mut self, cfg: &FemaUsarDebrisConfig) -> FemaUsarDebrisReport {
        let rep = exec_fema_usar_debris_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_saved_pct.len() {
                p.usar_saved_pct = rep.parcel_saved_pct[i] as f32;
                p.debris_blockage_tons = rep.parcel_debris_tons[i] as f32;
            }
        }
        rep
    }

    /// Applies the `FemaUsarDebrisReport` roadway rubble blockage & `ESF #3` bulldozer reopening state
    /// at `snapshot_hour` directly onto `self.network.segments` (closing segments whose `reopen_hour > snapshot_hour`
    /// and reopening cleared Phase I emergency lanes) as a `Ctrl+Z` undoable command!
    pub fn apply_debris_and_reopening_at_hour(
        &mut self,
        rep: &FemaUsarDebrisReport,
        snapshot_hour: f64,
    ) -> (usize, usize) {
        let before = EditStateSnapshot::capture_topology(self);
        let mut still_blocked = 0usize;
        let mut reopened_push = 0usize;
        for (sid, &reopen_h) in rep.segment_reopen_hour.iter().enumerate() {
            if sid >= self.network.segments.len() {
                continue;
            }
            if reopen_h > snapshot_hour + 1e-6 {
                self.network.segments[sid].closed = true;
                still_blocked += 1;
            } else if reopen_h > 0.05 {
                // Cleared via Phase I Emergency Push (1 emergency lane punched open at 75% speed)!
                self.network.segments[sid].closed = false;
                self.network.segments[sid].speed_mps =
                    (self.network.segments[sid].speed_mps * 0.78).clamp(4.5, 28.0);
                self.network.segments[sid].ema_speed_mps = self.network.segments[sid].speed_mps;
                reopened_push += 1;
            }
        }
        crate::network_edit::rebuild_world_after_network_edit(self, None);
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "US&R/ESF-3 Debris State @ H+{snapshot_hour:.0}h ({reopened_push} pushed open, {still_blocked} blocked)"
            ),
            self.sim_time_s,
            before,
            after,
        );
        (reopened_push, still_blocked)
    }

    /// Runs `Rayon`-parallelized FEMA `GSB` (`249th Engineer` Prime Power), `FEMA-325` `TDMS`
    /// Phase II Debris Sorting/Haul-Off, and `DRC`/`MRIC` Individual Assistance Monte Carlo,
    /// caching parcel recovery readiness & `DRC` wait onto `self.parcels`.
    pub fn run_fema_gsb_drc_tdms_mc(&mut self, cfg: &FemaGsbDrcTdmsConfig) -> FemaGsbDrcTdmsReport {
        let rep = exec_fema_gsb_drc_tdms_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_recovery_score_pct.len() {
                p.fema_recovery_score_pct = rep.parcel_recovery_score_pct[i] as f32;
                p.fema_drc_wait_min = rep.parcel_drc_wait_min[i] as f32;
            }
        }
        rep
    }

    /// Energizes municipal utility prime-power backup from a `FemaGsbDrcTdmsReport` (`249th Engineer`
    /// `GSB` mobile diesel packs) and commits parcel recovery state as a `Ctrl+Z` undoable command!
    pub fn apply_fema_gsb_backup_power_tracked(&mut self, rep: &FemaGsbDrcTdmsReport) -> usize {
        let before = EditStateSnapshot::capture_topology(self);
        self.utility_config.critical_backup_generators_enabled = true;
        self.utility_config.tripped_substation_indices.clear();
        let mut updated = 0usize;
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_recovery_score_pct.len() {
                p.fema_recovery_score_pct = rep.parcel_recovery_score_pct[i] as f32;
                p.fema_drc_wait_min = rep.parcel_drc_wait_min[i] as f32;
                updated += 1;
            }
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::PaintZoning,
            format!(
                "Deploy USACE 249th GSB Prime Power ({:.2} MW, {} lifelines)",
                rep.deployed_prime_power_mw, rep.total_critical_lifeline_targets
            ),
            self.sim_time_s,
            before,
            after,
        );
        updated
    }

    /// Runs `Rayon`-parallelized `ALERTWest` + `UOregon OHAZ` AI PTZ Wildfire Camera Triangulation
    /// & `NWCG` WUI Wildfire Suppression Monte Carlo, caching parcel burn risk & camera viewshed.
    pub fn run_wildfire_alertwest_mc(
        &mut self,
        cfg: &WildfireAlertWestConfig,
    ) -> WildfireAlertWestReport {
        let rep = exec_wildfire_alertwest_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_burn_risk_pct.len() {
                p.wildfire_burn_risk_pct = rep.parcel_burn_risk_pct[i] as f32;
                p.wildfire_camera_vis_pct = rep.parcel_camera_viewshed_pct[i] as f32;
            }
        }
        rep
    }

    /// Cuts an emergency WUI shaded fuelbreak (`ZoneCode::PARK`) along high-risk interface parcels,
    /// closes active flame-impinged road segments, and applies Oregon Level 1/2/3 evacuation tiers
    /// onto `self.parcels` as an atomic `Ctrl+Z` undoable command!
    pub fn apply_wui_fuelbreak_and_fire_closures_tracked(
        &mut self,
        rep: &WildfireAlertWestReport,
    ) -> (usize, usize) {
        let before = EditStateSnapshot::capture_topology(self);
        let mut fuelbreak_parcels = 0usize;
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_burn_risk_pct.len() {
                let risk = rep.parcel_burn_risk_pct[i];
                // Convert outer perimeter WUI collar (`32%..=62%` risk) on R1/IND parcels into shaded fuelbreak
                if (32.0..=62.0).contains(&risk)
                    && matches!(p.zone, ZoneCode::R1 | ZoneCode::IND)
                    && fuelbreak_parcels < 10
                {
                    p.zone = ZoneCode::PARK;
                    p.wildfire_burn_risk_pct = (risk * 0.35) as f32;
                    fuelbreak_parcels += 1;
                } else {
                    p.wildfire_burn_risk_pct = risk as f32;
                }
                p.wildfire_camera_vis_pct = rep.parcel_camera_viewshed_pct[i] as f32;
                // Map Oregon Level 3 (GO!) -> Stage 1, Level 2 (SET) -> Stage 2, Level 1 (READY) -> Stage 3
                p.evac_stage_tier = match rep.parcel_evac_level[i] {
                    3 => 1,
                    2 => 2,
                    _ => 3,
                };
            }
        }
        let mut closed_segs = 0usize;
        for seg in &mut self.network.segments {
            if let Some(mid) = seg.base_pts.get(seg.base_pts.len() / 2) {
                if mid.dist(rep.true_ignition_pos) <= 185.0 && !seg.closed && closed_segs < 8 {
                    seg.closed = true;
                    closed_segs += 1;
                }
            }
        }
        if closed_segs > 0 {
            crate::network_edit::rebuild_world_after_network_edit(self, None);
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "WUI Fuelbreak & Level 3 Evac ({fuelbreak_parcels} fuelbreak pcl, {closed_segs} fire roadblocks)"
            ),
            self.sim_time_s,
            before,
            after,
        );
        (fuelbreak_parcels, closed_segs)
    }

    /// Runs `Rayon`-parallelized Post-Wildfire Burn-Scar Hydrophobic Runoff, USGS `Staley M1` +
    /// `Gartner (2014)` Debris-Flow Basins, and `FHWA HDS-5` Stormwater Culvert Washout Monte Carlo,
    /// caching parcel runoff depth & debris-flow hazard onto `self.parcels`.
    pub fn run_burn_scar_debris_flow_mc(
        &mut self,
        cfg: &BurnScarDebrisFlowConfig,
    ) -> BurnScarDebrisFlowReport {
        let rep = exec_burn_scar_debris_flow_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_runoff_depth_mm.len() {
                p.burn_scar_runoff_mm = rep.parcel_runoff_depth_mm[i] as f32;
                p.debris_flow_hazard_pct = rep.parcel_debris_hazard_pct[i] as f32;
            }
        }
        rep
    }

    /// Injects `FHWA HDS-5` culvert embankment washout closures (`seg.closed = true`), overtopped
    /// pavement ponding speed reductions (`0.52x`), co-located utility pipe breaks, and muddy
    /// stormflow `RDII` sanitary sewer surcharge onto the live city as a `Ctrl+Z` undoable command!
    pub fn apply_culvert_washouts_and_rdii_tracked(
        &mut self,
        rep: &BurnScarDebrisFlowReport,
    ) -> (usize, usize) {
        let before = EditStateSnapshot::capture_topology(self);
        let mut washed = 0usize;
        let mut flooded = 0usize;
        for &sid in &rep.washed_out_segment_ids {
            if let Some(seg) = self.network.segments.get_mut(sid) {
                if !seg.closed {
                    seg.closed = true;
                    washed += 1;
                }
            }
        }
        for &sid in &rep.overtopped_slow_segment_ids {
            if let Some(seg) = self.network.segments.get_mut(sid) {
                if !seg.closed {
                    let v_new = (seg.speed_mps * 0.52).clamp(3.0, 22.0);
                    seg.speed_mps = v_new;
                    seg.ema_speed_mps = v_new;
                    flooded += 1;
                }
            }
        }
        self.utility_config.couple_closed_roads_to_utility_mains = true;
        self.utility_config.storm_rainfall_mm_hr = rep.coupled_rdii_sewer_inflow_mm_hr;
        crate::network_edit::rebuild_world_after_network_edit(self, None);
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::CorridorPolicy,
            format!(
                "Burn-Scar Culvert Washouts & RDII ({washed} washed out, {flooded} overtopped, {:.0} mm/hr RDII)",
                rep.coupled_rdii_sewer_inflow_mm_hr
            ),
            self.sim_time_s,
            before,
            after,
        );
        (washed, flooded)
    }

    /// Runs `Rayon`-parallelized `72-hour` Post-Wildfire Source-Water Ash/Turbidity, `WTP` Intake
    /// Shutoff, Coagulant Dosing & Finished-Water Storage Drawdown Monte Carlo, caching parcel
    /// water security & minimum pressure onto `self.parcels`.
    pub fn run_wtp_turbidity_storage_mc(
        &mut self,
        cfg: &WtpTurbidityStorageConfig,
    ) -> WtpTurbidityStorageReport {
        let rep = exec_wtp_turbidity_storage_mc(
            &self.network,
            &self.parcels,
            &self.emergency_facilities,
            cfg,
        );
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_water_security_pct.len() {
                p.wtp_water_security_pct = rep.parcel_water_security_pct[i] as f32;
                p.wtp_min_pressure_psi = rep.parcel_min_pressure_psi[i] as f32;
            }
        }
        rep
    }

    /// Injects the post-wildfire `WTP` minimum finished-water clearwell/tank storage level
    /// (`finished_water_storage_fill_pct`) and active fire-hydrant draft onto `self.utility_config`
    /// as a `Ctrl+Z` undoable command so live Hazen-Williams nodal pressures & `< 20 psi` `BWA` zones update!
    pub fn apply_wtp_storage_drawdown_tracked(
        &mut self,
        rep: &WtpTurbidityStorageReport,
    ) -> f64 {
        let before = EditStateSnapshot::capture_topology(self);
        self.utility_config.finished_water_storage_fill_pct = rep.min_finished_storage_fill_pct;
        self.utility_config.active_fire_flow_demand_gpm = rep.config.active_fire_hydrant_draft_gpm;
        self.utility_config.heatmap_sublayer =
            crate::utilities::UtilityHeatmapSublayer::WaterResidualPressurePsi;
        for (i, p) in self.parcels.iter_mut().enumerate() {
            if i < rep.parcel_water_security_pct.len() {
                p.wtp_water_security_pct = rep.parcel_water_security_pct[i] as f32;
                p.wtp_min_pressure_psi = rep.parcel_min_pressure_psi[i] as f32;
            }
        }
        self.refresh_urban_metrics();
        let after = EditStateSnapshot::capture_topology(self);
        self.edit_history.push_transition(
            EditCategory::PaintZoning,
            format!(
                "WTP Turbidity Drawdown (Min Storage {:.1}%, Min High-Zone {:.1} psi)",
                rep.min_finished_storage_fill_pct, rep.min_high_zone_pressure_psi
            ),
            self.sim_time_s,
            before,
            after,
        );
        self.utility_report.water.nodes_meeting_20psi_fire_pct
    }
}


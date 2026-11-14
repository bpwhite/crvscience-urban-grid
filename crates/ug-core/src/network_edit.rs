//! Interactive road-segment drawing (with automatic junction snapping & mid-block T-junction
//! splitting), NACTO lane-count / road-diet editing, and FHWA multi-lane modern roundabout
//! conversion, all preserving live microscopic agents across topology rebuilds.

use crate::history::{apply_edit_snapshot_to_world, EditCategory, EditStateSnapshot};
use crate::emergency::JunctionEvpState;
use crate::max_pressure::JunctionMpState;
use crate::network::{
    build_cumulative_lengths, polyline_length, sample_polyline, ControlMode, Junction, RoadNetwork,
    RoadSegment, Vec2,
};
use crate::traffic::{CarrierRef, CyclistAgent, PedestrianAgent, SimulationWorld};
use crate::transit_gtfs::shortest_segment_chain;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NetworkEditError {
    #[error("Road segment must be at least 14.0 m long (got {0:.1} m).")]
    RoadTooShort(f64),
    #[error("Start and end points snapped to the same junction #{0}.")]
    SameEndpoint(usize),
    #[error("Segment #{0} out of bounds.")]
    InvalidSegment(usize),
    #[error("Junction #{0} out of bounds.")]
    InvalidJunction(usize),
    #[error("Junction #{0} is already part of a roundabout.")]
    AlreadyRoundabout(usize),
    #[error("Roundabout conversion requires at least 3 connected arms (junction #{0} has {1}).")]
    InsufficientArms(usize, usize),
}

/// Parameters for interactively drawn road corridors in the viewport.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NewRoadSpec {
    pub lanes_per_dir: usize,
    pub bidirectional: bool,
    pub speed_mph: f64,
    pub has_bus_lane: bool,
    pub has_bike_lane: bool,
}

impl Default for NewRoadSpec {
    fn default() -> Self {
        Self {
            lanes_per_dir: 2,
            bidirectional: true,
            speed_mph: 25.0,
            has_bus_lane: false,
            has_bike_lane: true,
        }
    }
}

/// Snap candidate returned while hovering the viewport cursor in `DrawRoad` mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapTarget {
    ExistingJunction { junc_id: usize, pos: Vec2 },
    SplitSegment { seg_id: usize, s_m: f64, pos: Vec2 },
    FreePoint { pos: Vec2 },
}

impl SnapTarget {
    pub fn pos(self) -> Vec2 {
        match self {
            Self::ExistingJunction { pos, .. }
            | Self::SplitSegment { pos, .. }
            | Self::FreePoint { pos } => pos,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RoundaboutBuildSummary {
    pub arms_connected: usize,
    pub inscribed_radius_m: f64,
    pub ring_lanes: usize,
    pub new_ring_segments: usize,
}

#[derive(Clone, Copy)]
struct SavedVehLoc {
    seg_id: usize,
    lane_idx: usize,
    s_m: f64,
}

fn capture_vehicle_locs(world: &SimulationWorld) -> Vec<SavedVehLoc> {
    world
        .vehicles
        .iter()
        .map(|v| match v.carrier {
            CarrierRef::Lane(lid) => {
                let lane = &world.network.lanes[lid];
                SavedVehLoc {
                    seg_id: lane.segment_id,
                    lane_idx: lane.index,
                    s_m: v.s_m,
                }
            }
            CarrierRef::Connector(cid) => {
                let conn = &world.network.connectors[cid];
                SavedVehLoc {
                    seg_id: conn.to_seg_id,
                    lane_idx: 0,
                    s_m: 1.0,
                }
            }
        })
        .collect()
}

/// Rebuilds lanes, connectors, signals, routes, and GTFS stop/route chains while remapping all
/// live vehicles, cyclists, and pedestrians onto the updated topology without dangling indices.
pub fn rebuild_world_after_network_edit(
    world: &mut SimulationWorld,
    veh_remap: Option<&[(usize, usize, f64)]>, // optional `(old_seg, new_seg_after_split, split_s_m)`
) {
    let mut locs = capture_vehicle_locs(world);
    if let Some(splits) = veh_remap {
        for loc in &mut locs {
            for &(old_sid, new_sid, split_s) in splits {
                if loc.seg_id == old_sid && loc.s_m > split_s {
                    loc.seg_id = new_sid;
                    loc.s_m -= split_s;
                }
            }
        }
    }

    world.network.rebuild_topology_from_segments(true);

    let n_segs = world.network.segments.len();
    let old_vehs = std::mem::take(&mut world.vehicles);
    let mut kept_vehs = Vec::with_capacity(old_vehs.len());

    for (mut veh, loc) in old_vehs.into_iter().zip(locs.into_iter()) {
        if loc.seg_id >= n_segs {
            continue;
        }
        let seg = &world.network.segments[loc.seg_id];
        if seg.closed || seg.lane_ids.is_empty() {
            continue;
        }
        let li = loc.lane_idx.min(seg.lane_ids.len() - 1);
        let new_lid = seg.lane_ids[li];
        let new_len = world.network.lanes[new_lid].length;
        veh.carrier = CarrierRef::Lane(new_lid);
        veh.s_m = loc.s_m.clamp(0.5, (new_len - 0.5).max(0.5));
        veh.planned_conn_id = None;
        if let Some(dst) = veh.dest_seg_id {
            if dst >= n_segs || !world.network.sinks.contains(&dst) {
                veh.dest_seg_id = None;
            }
        }
        kept_vehs.push(veh);
    }
    world.vehicles = kept_vehs;

    // Clamp cyclists & pedestrians onto valid open segments
    world
        .cyclists
        .retain(|c| c.segment_id < n_segs && !world.network.segments[c.segment_id].closed);
    for c in &mut world.cyclists {
        let len = world.network.segments[c.segment_id].length.max(4.0);
        c.s_m = c.s_m.clamp(0.5, len - 0.5);
    }
    world
        .pedestrians
        .retain(|p| p.segment_id < n_segs && !world.network.segments[p.segment_id].closed);
    for p in &mut world.pedestrians {
        let len = world.network.segments[p.segment_id].length.max(4.0);
        p.s_m = p.s_m.clamp(0.5, len - 0.5);
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

    // Repair GTFS stop coordinates & route segment chains across modified corridors/roundabouts
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
    // Synchronize active GTFS scheduled buses onto updated `route.segment_chain` indices
    for veh in &mut world.vehicles {
        if let Some(r_idx) = veh.gtfs_route_idx {
            if let Some(route) = world.transit_gtfs.routes.get(r_idx as usize) {
                let cur_seg = match veh.carrier {
                    CarrierRef::Lane(lid) => world.network.lanes[lid].segment_id,
                    CarrierRef::Connector(cid) => world.network.connectors[cid].to_seg_id,
                };
                if let Some(pos_idx) = route.segment_chain.iter().position(|&s| s == cur_seg) {
                    veh.gtfs_route_seq = pos_idx as u16;
                    veh.gtfs_next_stop_order = route
                        .route_stops
                        .iter()
                        .position(|&(_, chain_pos)| chain_pos >= pos_idx)
                        .unwrap_or(route.route_stops.len()) as u16;
                } else {
                    veh.gtfs_route_idx = None;
                    veh.dwell_remaining_s = 0.0;
                }
            } else {
                veh.gtfs_route_idx = None;
            }
        }
    }

    world.watched_segment_id = world
        .watched_segment_id
        .min(world.network.segments.len().saturating_sub(1));
    world.refresh_urban_metrics();
}

/// Finds the closest active `Junction` within `max_dist_m` of `(wx, wy)`.
pub fn hit_test_junction(net: &RoadNetwork, wx: f64, wy: f64, max_dist_m: f64) -> Option<usize> {
    let p = Vec2 { x: wx, y: wy };
    net.junctions
        .iter()
        .filter(|j| !j.in_segs.is_empty() || !j.out_segs.is_empty())
        .map(|j| (j.id, j.pos.dist(p)))
        .filter(|&(_, d)| d <= max_dist_m)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
}

/// Finds the closest `RoadSegment` carriageway to `(wx, wy)` within `max_dist_m`, returning
/// `(segment_id, longitudinal_s_m, centerline_projection_pos)`.
pub fn hit_test_segment(
    net: &RoadNetwork,
    wx: f64,
    wy: f64,
    max_dist_m: f64,
) -> Option<(usize, f64, Vec2)> {
    let mut best: Option<(f64, usize, f64, Vec2)> = None;
    for seg in &net.segments {
        if seg.lane_ids.is_empty() || seg.base_pts.len() < 2 {
            continue;
        }
        // Test against actual offset lanes so clicking right-hand vs left-hand carriageway picks the exact direction!
        for &lid in &seg.lane_ids {
            let lane = &net.lanes[lid];
            let mut cum = 0.0;
            for w in lane.pts.windows(2) {
                let dx = w[1].x - w[0].x;
                let dy = w[1].y - w[0].y;
                let l2 = dx * dx + dy * dy;
                let l = l2.sqrt().max(1e-6);
                let t = if l2 > 1e-9 {
                    (((wx - w[0].x) * dx + (wy - w[0].y) * dy) / l2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let px = w[0].x + dx * t;
                let py = w[0].y + dy * t;
                let d = (wx - px).hypot(wy - py);
                if d <= max_dist_m && best.map(|(bd, _, _, _)| d < bd).unwrap_or(true) {
                    let frac = ((cum + t * l) / lane.length.max(1.0)).clamp(0.0, 1.0);
                    let raw_len = polyline_length(&seg.base_pts);
                    let (base_p, _) = sample_polyline(&seg.base_pts, frac * raw_len);
                    best = Some((d, seg.id, frac * raw_len, base_p));
                }
                cum += l;
            }
        }
    }
    best.map(|(_, sid, s_m, p)| (sid, s_m, p))
}

/// Evaluates snap priority for interactive CAD road drawing:
/// 1. Existing active `Junction` within `16 m`
/// 2. Interior of an existing `RoadSegment` within `14 m` (splitting it into a T-junction!)
/// 3. Free world coordinate `(wx, wy)`
pub fn preview_draw_snap(net: &RoadNetwork, wx: f64, wy: f64) -> SnapTarget {
    if let Some(jid) = hit_test_junction(net, wx, wy, 16.0) {
        return SnapTarget::ExistingJunction {
            junc_id: jid,
            pos: net.junctions[jid].pos,
        };
    }
    if let Some((sid, s_m, proj_pos)) = hit_test_segment(net, wx, wy, 14.0) {
        let raw_len = polyline_length(&net.segments[sid].base_pts);
        if s_m >= 12.0 && (raw_len - s_m) >= 12.0 {
            return SnapTarget::SplitSegment {
                seg_id: sid,
                s_m,
                pos: proj_pos,
            };
        }
    }
    SnapTarget::FreePoint {
        pos: Vec2 { x: wx, y: wy },
    }
}

fn slice_polyline(pts: &[Vec2], s_start: f64, s_end: f64) -> Vec<Vec2> {
    let total = polyline_length(pts);
    let a = s_start.clamp(0.0, total);
    let b = s_end.clamp(a + 1.0, total.max(a + 1.0));
    let n_steps = (((b - a) / 8.0).ceil() as usize).max(2);
    let mut out = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let s = a + (b - a) * (i as f64 / n_steps as f64);
        let (p, _) = sample_polyline(pts, s);
        out.push(p);
    }
    out
}

fn realize_snap_endpoint(
    world: &mut SimulationWorld,
    snap: SnapTarget,
    splits_out: &mut Vec<(usize, usize, f64)>,
) -> usize {
    match snap {
        SnapTarget::ExistingJunction { junc_id, .. } => junc_id,
        SnapTarget::FreePoint { pos } => {
            let jid = world.network.junctions.len();
            world.network.junctions.push(Junction {
                id: jid,
                osm_node_id: format!("edit_node_{jid}"),
                pos,
                in_segs: Vec::new(),
                out_segs: Vec::new(),
                connectors: Vec::new(),
                is_roundabout_node: false,
                arm_count: 0,
                control_mode: ControlMode::PermissiveYield,
                ns_green_s: 22.0,
                ew_green_s: 20.0,
                yellow_s: 3.5,
                all_red_s: 1.5,
                offset_s: 0.0,
                hold_phase: false,
                held_at_s: 0.0,
                manual_step_offset_s: 0.0,
                axis_angle: 0.0,
                mp_state: JunctionMpState::default(),
                evp_state: JunctionEvpState::default(),
            });
            jid
        }
        SnapTarget::SplitSegment { seg_id, s_m, pos } => {
            let new_jid = world.network.junctions.len();
            world.network.junctions.push(Junction {
                id: new_jid,
                osm_node_id: format!("split_node_{new_jid}"),
                pos,
                in_segs: Vec::new(),
                out_segs: Vec::new(),
                connectors: Vec::new(),
                is_roundabout_node: false,
                arm_count: 0,
                control_mode: ControlMode::Signal2Phase,
                ns_green_s: 22.0,
                ew_green_s: 20.0,
                yellow_s: 3.5,
                all_red_s: 1.5,
                offset_s: 0.0,
                hold_phase: false,
                held_at_s: 0.0,
                manual_step_offset_s: 0.0,
                axis_angle: 0.0,
                mp_state: JunctionMpState::default(),
                evp_state: JunctionEvpState::default(),
            });

            let u = world.network.segments[seg_id].from_junc;
            let v = world.network.segments[seg_id].to_junc;

            // Find optional twin reverse segment (`v -> u`) so splitting a two-way street splits both directions!
            let twin_opt = world
                .network
                .segments
                .iter()
                .find(|s| s.id != seg_id && s.from_junc == v && s.to_junc == u && !s.is_roundabout)
                .map(|s| s.id);

            let split_one = |segments: &mut Vec<RoadSegment>,
                             sid: usize,
                             cut_s: f64,
                             splits_vec: &mut Vec<(usize, usize, f64)>| {
                let orig = segments[sid].clone();
                let raw_total = polyline_length(&orig.base_pts);
                let safe_cut = cut_s.clamp(6.0, (raw_total - 6.0).max(6.0));
                let mut pts_a = slice_polyline(&orig.base_pts, 0.0, safe_cut);
                if let Some(last) = pts_a.last_mut() {
                    *last = pos;
                }
                let mut pts_b = slice_polyline(&orig.base_pts, safe_cut, raw_total);
                if let Some(first) = pts_b.first_mut() {
                    *first = pos;
                }

                segments[sid].to_junc = new_jid;
                segments[sid].length = polyline_length(&pts_a);
                segments[sid].base_cum_s = build_cumulative_lengths(&pts_a);
                segments[sid].base_pts = pts_a;

                let b_id = segments.len();
                let b_len = polyline_length(&pts_b);
                let b_cum = build_cumulative_lengths(&pts_b);
                segments.push(RoadSegment {
                    id: b_id,
                    from_junc: new_jid,
                    to_junc: orig.to_junc,
                    base_pts: pts_b,
                    base_cum_s: b_cum,
                    length: b_len,
                    lane_ids: Vec::new(),
                    ..orig
                });
                splits_vec.push((sid, b_id, safe_cut));
            };

            split_one(&mut world.network.segments, seg_id, s_m, splits_out);
            if let Some(twin_sid) = twin_opt {
                let twin_total = polyline_length(&world.network.segments[twin_sid].base_pts);
                split_one(
                    &mut world.network.segments,
                    twin_sid,
                    (twin_total - s_m).max(6.0),
                    splits_out,
                );
            }
            new_jid
        }
    }
}

/// Draws a new one-way or two-way road corridor between `start_world` and `end_world`, snapping to
/// existing junctions or automatically splitting mid-block segments into signalized T-intersections.
pub fn draw_road_segment(
    world: &mut SimulationWorld,
    start_world: Vec2,
    end_world: Vec2,
    spec: NewRoadSpec,
) -> Result<Vec<usize>, NetworkEditError> {
    let snap_a = preview_draw_snap(&world.network, start_world.x, start_world.y);
    let snap_b = preview_draw_snap(&world.network, end_world.x, end_world.y);
    if let (
        SnapTarget::ExistingJunction { junc_id: ja, .. },
        SnapTarget::ExistingJunction { junc_id: jb, .. },
    ) = (snap_a, snap_b)
    {
        if ja == jb {
            return Err(NetworkEditError::SameEndpoint(ja));
        }
    }
    let dist_m = snap_a.pos().dist(snap_b.pos());
    if dist_m < 14.0 {
        return Err(NetworkEditError::RoadTooShort(dist_m));
    }

    let before = EditStateSnapshot::capture_topology(world);
    let mut splits = Vec::new();
    let u = realize_snap_endpoint(world, snap_a, &mut splits);
    // Re-evaluate snap_b if snap_a split the exact same segment!
    let snap_b_updated = match (snap_a, snap_b) {
        (SnapTarget::SplitSegment { seg_id: sa, .. }, SnapTarget::SplitSegment { seg_id: sb, .. })
            if sa == sb =>
        {
            preview_draw_snap(&world.network, end_world.x, end_world.y)
        }
        _ => snap_b,
    };
    let v = realize_snap_endpoint(world, snap_b_updated, &mut splits);
    if u == v {
        apply_edit_snapshot_to_world(world, &before);
        return Err(NetworkEditError::SameEndpoint(u));
    }

    let p_u = world.network.junctions[u].pos;
    let p_v = world.network.junctions[v].pos;
    let pts_fwd = vec![p_u, p_v];
    let len_m = p_u.dist(p_v);
    let lanes = spec.lanes_per_dir.clamp(1, 5);
    let mph = spec.speed_mph.clamp(15.0, 65.0);
    let mps = mph * 0.447_04;
    let prio = if lanes >= 3 {
        4
    } else if lanes == 2 {
        3
    } else {
        2
    };
    let hwy = if lanes >= 3 {
        "primary"
    } else if lanes == 2 {
        "secondary"
    } else {
        "tertiary"
    };
    let way_id = format!("custom_way_{}", world.network.segments.len() + 1);
    let road_name = format!("Custom Corridor #{}", world.network.segments.len() + 1);

    let mut created_ids = Vec::new();
    let fwd_id = world.network.segments.len();
    world.network.segments.push(RoadSegment {
        id: fwd_id,
        way_id: way_id.clone(),
        name: road_name.clone(),
        highway: hwy.to_string(),
        priority: prio,
        is_roundabout: false,
        is_oneway: !spec.bidirectional,
        from_junc: u,
        to_junc: v,
        base_cum_s: build_cumulative_lengths(&pts_fwd),
        base_pts: pts_fwd,
        length: len_m,
        lane_count: lanes,
        speed_mph: mph,
        speed_mps: mps,
        approach_axis: 0,
        has_bus_lane: spec.has_bus_lane,
        has_bike_lane: spec.has_bike_lane,
        closed: false,
        stalled_lane: None,
        ema_speed_mps: mps,
        vsp_intensity: 0.05,
        lane_ids: Vec::new(),
    });
    created_ids.push(fwd_id);

    if spec.bidirectional {
        let pts_rev = vec![p_v, p_u];
        let rev_id = world.network.segments.len();
        world.network.segments.push(RoadSegment {
            id: rev_id,
            way_id,
            name: road_name,
            highway: hwy.to_string(),
            priority: prio,
            is_roundabout: false,
            is_oneway: false,
            from_junc: v,
            to_junc: u,
            base_cum_s: build_cumulative_lengths(&pts_rev),
            base_pts: pts_rev,
            length: len_m,
            lane_count: lanes,
            speed_mph: mph,
            speed_mps: mps,
            approach_axis: 0,
            has_bus_lane: spec.has_bus_lane,
            has_bike_lane: spec.has_bike_lane,
            closed: false,
            stalled_lane: None,
            ema_speed_mps: mps,
            vsp_intensity: 0.05,
            lane_ids: Vec::new(),
        });
        created_ids.push(rev_id);
    }

    // Seed cyclists & sidewalk pedestrians on the newly drawn street
    for &sid in &created_ids {
        if spec.has_bike_lane {
            world.cyclists.push(CyclistAgent {
                segment_id: sid,
                s_m: len_m * 0.35,
                v_mps: 5.1,
                lateral_jitter_m: 0.0,
            });
        }
        world.pedestrians.push(PedestrianAgent {
            segment_id: sid,
            side_sign: 1,
            walk_dir: 1,
            lateral_jitter_m: 0.1,
            s_m: len_m * 0.5,
            v_mps: 1.35,
            cur_speed_mps: 1.35,
            phase_seed: sid as u16,
        });
    }

    rebuild_world_after_network_edit(world, Some(&splits));
    let after = EditStateSnapshot::capture_topology(world);
    world.edit_history.push_transition(
        EditCategory::DrawRoad,
        format!(
            "Draw {}-ln {} road ({:.0} m)",
            lanes,
            if spec.bidirectional { "two-way" } else { "one-way" },
            len_m
        ),
        world.sim_time_s,
        before,
        after,
    );
    Ok(created_ids)
}

/// Sets the directional lane count (`1..=5`) on `seg_id` (and optionally its twin reverse
/// carriageway when `symmetric_corridor == true`), rebuilding metric lanes & turn splines live.
pub fn set_segment_lane_count(
    world: &mut SimulationWorld,
    seg_id: usize,
    new_lanes: usize,
    symmetric_corridor: bool,
) -> Result<usize, NetworkEditError> {
    if seg_id >= world.network.segments.len() {
        return Err(NetworkEditError::InvalidSegment(seg_id));
    }
    let clamped = new_lanes.clamp(1, 5);
    let u = world.network.segments[seg_id].from_junc;
    let v = world.network.segments[seg_id].to_junc;
    let prev_lanes = world.network.segments[seg_id].lane_count;
    let twin_diff = symmetric_corridor
        && !world.network.segments[seg_id].is_roundabout
        && world
            .network
            .segments
            .iter()
            .any(|s| s.from_junc == v && s.to_junc == u && !s.is_roundabout && s.lane_count != clamped);
    if prev_lanes == clamped && !twin_diff {
        return Ok(clamped);
    }

    let before = EditStateSnapshot::capture_topology(world);
    world.network.segments[seg_id].lane_count = clamped;
    if clamped < 2 {
        world.network.segments[seg_id].has_bus_lane = false;
    }

    if symmetric_corridor && !world.network.segments[seg_id].is_roundabout {
        for seg in &mut world.network.segments {
            if seg.from_junc == v && seg.to_junc == u && !seg.is_roundabout {
                seg.lane_count = clamped;
                if clamped < 2 {
                    seg.has_bus_lane = false;
                }
            }
        }
    }

    rebuild_world_after_network_edit(world, None);
    let after = EditStateSnapshot::capture_topology(world);
    world.edit_history.push_transition(
        EditCategory::EditLanes,
        format!("Edit corridor #{seg_id} lanes: {prev_lanes} -> {clamped} ln/dir"),
        world.sim_time_s,
        before,
        after,
    );
    Ok(clamped)
}

pub fn adjust_segment_lane_count(
    world: &mut SimulationWorld,
    seg_id: usize,
    delta: isize,
    symmetric_corridor: bool,
) -> Result<usize, NetworkEditError> {
    if seg_id >= world.network.segments.len() {
        return Err(NetworkEditError::InvalidSegment(seg_id));
    }
    let cur = world.network.segments[seg_id].lane_count as isize;
    let target = (cur + delta).clamp(1, 5) as usize;
    set_segment_lane_count(world, seg_id, target, symmetric_corridor)
}

/// Converts a 3+ arm intersection (`junc_id`) into a multi-segment curved FHWA Modern Roundabout
/// with inscribed radius `radius_m` (`12..=32 m`), `ring_lanes` (`1..=3`) counter-clockwise
/// circulating lanes, trimmed approach flares, and yield-at-entry gap acceptance.
pub fn convert_junction_to_roundabout(
    world: &mut SimulationWorld,
    junc_id: usize,
    radius_m: f64,
    ring_lanes: usize,
) -> Result<RoundaboutBuildSummary, NetworkEditError> {
    if junc_id >= world.network.junctions.len() {
        return Err(NetworkEditError::InvalidJunction(junc_id));
    }
    if world.network.junctions[junc_id].is_roundabout_node {
        return Err(NetworkEditError::AlreadyRoundabout(junc_id));
    }

    let c_pos = world.network.junctions[junc_id].pos;
    let in_segs = world.network.junctions[junc_id].in_segs.clone();
    let out_segs = world.network.junctions[junc_id].out_segs.clone();

    let mut min_raw_len = f64::INFINITY;
    for &sid in in_segs.iter().chain(out_segs.iter()) {
        let l = polyline_length(&world.network.segments[sid].base_pts);
        min_raw_len = min_raw_len.min(l);
    }
    if !min_raw_len.is_finite() {
        return Err(NetworkEditError::InsufficientArms(junc_id, 0));
    }
    let r_m = radius_m.clamp(12.0, (min_raw_len * 0.36).clamp(12.0, 32.0));

    // Cluster incident segments by outward geographic counter-clockwise angle `phi` in `[0, 2*PI)`:
    // `phi = 0` East (+X), `phi = PI/2` North (-Y), `phi = PI` West (-X), `phi = 3*PI/2` South (+Y).
    struct ArmCluster {
        phi_sum: f64,
        count: usize,
        in_sids: Vec<usize>,
        out_sids: Vec<usize>,
    }
    impl ArmCluster {
        fn mean_phi(&self) -> f64 {
            (self.phi_sum / self.count.max(1) as f64).rem_euclid(std::f64::consts::TAU)
        }
    }

    let angle_diff = |a: f64, b: f64| -> f64 {
        let mut d = (a - b) % std::f64::consts::TAU;
        if d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        } else if d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        d.abs()
    };

    let mut arms: Vec<ArmCluster> = Vec::new();
    let mut insert_arm_seg = |phi: f64, sid: usize, is_incoming: bool| {
        let norm_phi = phi.rem_euclid(std::f64::consts::TAU);
        if let Some(arm) = arms
            .iter_mut()
            .find(|a| angle_diff(a.mean_phi(), norm_phi) < 0.32)
        {
            // Unwrap norm_phi close to current mean before averaging
            let mean = arm.mean_phi();
            let mut adj = norm_phi;
            if adj - mean > std::f64::consts::PI {
                adj -= std::f64::consts::TAU;
            } else if mean - adj > std::f64::consts::PI {
                adj += std::f64::consts::TAU;
            }
            arm.phi_sum += adj;
            arm.count += 1;
            if is_incoming {
                arm.in_sids.push(sid);
            } else {
                arm.out_sids.push(sid);
            }
        } else {
            arms.push(ArmCluster {
                phi_sum: norm_phi,
                count: 1,
                in_sids: if is_incoming { vec![sid] } else { Vec::new() },
                out_sids: if is_incoming { Vec::new() } else { vec![sid] },
            });
        }
    };

    for &sid in &in_segs {
        let pts = &world.network.segments[sid].base_pts;
        let raw_l = polyline_length(pts);
        let (p_arm, _) = sample_polyline(pts, (raw_l - r_m.min(raw_l * 0.5)).max(0.0));
        let phi = (-(p_arm.y - c_pos.y)).atan2(p_arm.x - c_pos.x);
        insert_arm_seg(phi, sid, true);
    }
    for &sid in &out_segs {
        let pts = &world.network.segments[sid].base_pts;
        let raw_l = polyline_length(pts);
        let (p_arm, _) = sample_polyline(pts, r_m.min(raw_l * 0.5));
        let phi = (-(p_arm.y - c_pos.y)).atan2(p_arm.x - c_pos.x);
        insert_arm_seg(phi, sid, false);
    }

    if arms.len() < 3 {
        return Err(NetworkEditError::InsufficientArms(junc_id, arms.len()));
    }

    let before = EditStateSnapshot::capture_topology(world);
    arms.sort_by(|a, b| {
        a.mean_phi()
            .partial_cmp(&b.mean_phi())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let ring_pt = |phi: f64| -> Vec2 {
        Vec2 {
            x: c_pos.x + r_m * phi.cos(),
            y: c_pos.y - r_m * phi.sin(),
        }
    };

    let n_arms = arms.len();
    let mut ring_jids = Vec::with_capacity(n_arms);
    for (k, arm) in arms.iter().enumerate() {
        let phi = arm.mean_phi();
        let r_jid = world.network.junctions.len();
        world.network.junctions.push(Junction {
            id: r_jid,
            osm_node_id: format!("rb_{junc_id}_arm_{k}"),
            pos: ring_pt(phi),
            in_segs: Vec::new(),
            out_segs: Vec::new(),
            connectors: Vec::new(),
            is_roundabout_node: true,
            arm_count: 3,
            control_mode: ControlMode::Roundabout,
            ns_green_s: 20.0,
            ew_green_s: 20.0,
            yellow_s: 3.0,
            all_red_s: 1.0,
            offset_s: 0.0,
            hold_phase: false,
            held_at_s: 0.0,
            manual_step_offset_s: 0.0,
            axis_angle: 0.0,
            mp_state: JunctionMpState::default(),
                evp_state: JunctionEvpState::default(),
        });
        ring_jids.push((r_jid, phi));
    }

    // Trim & re-anchor each arm's incoming and outgoing segments onto its perimeter flare node
    for (arm, &(r_jid, phi)) in arms.iter().zip(ring_jids.iter()) {
        let anchor = ring_pt(phi);
        for &in_sid in &arm.in_sids {
            let orig_pts = world.network.segments[in_sid].base_pts.clone();
            let raw_l = polyline_length(&orig_pts);
            let mut trimmed = slice_polyline(&orig_pts, 0.0, (raw_l - r_m).max(6.0));
            if let Some(last) = trimmed.last_mut() {
                *last = anchor;
            }
            world.network.segments[in_sid].to_junc = r_jid;
            world.network.segments[in_sid].length = polyline_length(&trimmed);
            world.network.segments[in_sid].base_cum_s = build_cumulative_lengths(&trimmed);
            world.network.segments[in_sid].base_pts = trimmed;
        }
        for &out_sid in &arm.out_sids {
            let orig_pts = world.network.segments[out_sid].base_pts.clone();
            let raw_l = polyline_length(&orig_pts);
            let mut trimmed = slice_polyline(&orig_pts, r_m.min((raw_l - 6.0).max(1.0)), raw_l);
            if let Some(first) = trimmed.first_mut() {
                *first = anchor;
            }
            world.network.segments[out_sid].from_junc = r_jid;
            world.network.segments[out_sid].length = polyline_length(&trimmed);
            world.network.segments[out_sid].base_cum_s = build_cumulative_lengths(&trimmed);
            world.network.segments[out_sid].base_pts = trimmed;
        }
    }

    // Build curved counter-clockwise circulatory ring segments connecting perimeter flare nodes
    let eff_ring_lanes = ring_lanes.clamp(1, 3);
    let rb_way_id = format!("rb_ring_{junc_id}");
    for k in 0..n_arms {
        let (u_jid, phi0) = ring_jids[k];
        let (v_jid, mut phi1) = ring_jids[(k + 1) % n_arms];
        if phi1 <= phi0 {
            phi1 += std::f64::consts::TAU;
        }
        let arc_steps = 6usize;
        let mut arc_pts = Vec::with_capacity(arc_steps + 1);
        for s in 0..=arc_steps {
            let t = s as f64 / arc_steps as f64;
            arc_pts.push(ring_pt(phi0 + (phi1 - phi0) * t));
        }
        let arc_len = polyline_length(&arc_pts).max(6.0);
        let sid = world.network.segments.len();
        world.network.segments.push(RoadSegment {
            id: sid,
            way_id: rb_way_id.clone(),
            name: format!("Roundabout #{junc_id} Ring"),
            highway: "primary".to_string(),
            priority: 6,
            is_roundabout: true,
            is_oneway: true,
            from_junc: u_jid,
            to_junc: v_jid,
            base_cum_s: build_cumulative_lengths(&arc_pts),
            base_pts: arc_pts,
            length: arc_len,
            lane_count: eff_ring_lanes,
            speed_mph: 20.0,
            speed_mps: 20.0 * 0.447_04,
            approach_axis: 0,
            has_bus_lane: false,
            has_bike_lane: true,
            closed: false,
            stalled_lane: None,
            ema_speed_mps: 20.0 * 0.447_04,
            vsp_intensity: 0.05,
            lane_ids: Vec::new(),
        });
    }

    // Retire old point-intersection node cleanly
    world.network.junctions[junc_id].control_mode = ControlMode::PermissiveYield;
    world.network.junctions[junc_id].in_segs.clear();
    world.network.junctions[junc_id].out_segs.clear();
    world.network.junctions[junc_id].connectors.clear();

    rebuild_world_after_network_edit(world, None);
    let after = EditStateSnapshot::capture_topology(world);
    world.edit_history.push_transition(
        EditCategory::ConvertRoundabout,
        format!(
            "Convert junction #{junc_id} -> {eff_ring_lanes}-ln roundabout (R={r_m:.0}m)"
        ),
        world.sim_time_s,
        before,
        after,
    );

    Ok(RoundaboutBuildSummary {
        arms_connected: n_arms,
        inscribed_radius_m: r_m,
        ring_lanes: eff_ring_lanes,
        new_ring_segments: n_arms,
    })
}

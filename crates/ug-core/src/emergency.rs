//! Emergency Vehicle Preemption (`EVP`) green-wave corridor lookahead controller and acoustic siren
//! zone tracker grounded in the FHWA Traffic Signal Timing Manual (Chapter 9: Traffic Signal
//! Preemption & Chapter 5: Queue Clearance `Gq = 3 + 2n`).

use crate::max_pressure::{MpPhaseId, MpTransitionStage};
use crate::network::{ControlMode, Junction, LightColor, Lane, RoadSegment, TurnConnector, TurnType};
use crate::traffic::{CarrierRef, Vehicle, VehicleClass};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EvpConfig {
    /// Master enable for corridor green-wave signal preemption & siren pull-over yielding.
    pub enabled: bool,
    /// Base upstream V2I / optical geofence lookahead distance ahead of each responder (m).
    pub lookahead_distance_m: f64,
    /// Maximum downstream signalized intersections preempted simultaneously per responder.
    pub max_corridor_hops: usize,
    /// Acoustic siren radius (m) ahead of an approaching responder that triggers civilian
    /// `MOBIL` right-lane pull-over changes and single-lane curb shoulder yielding.
    pub siren_yield_radius_m: f64,
    /// Minimum safe yellow change interval (s) displayed to interrupted cross-street green
    /// movements before preemption locks green (per MUTCD / FHWA Chapter 9).
    pub min_clearance_yellow_s: f64,
    /// Brief green hold linger (s) after the responder clears the stop-bar before releasing.
    pub post_passage_hold_s: f64,
    /// Background stochastic dispatch rate of emergency response units (units / simulated minute).
    pub auto_dispatch_per_min: f64,
    /// When true, adds NFPA 1710 turnout time (80 s Fire / 60 s EMS) onto travel time.
    #[serde(default)]
    pub include_nfpa_turnout_time: bool,
}

impl Default for EvpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lookahead_distance_m: 260.0,
            max_corridor_hops: 3,
            siren_yield_radius_m: 72.0,
            min_clearance_yellow_s: 2.0,
            post_passage_hold_s: 1.5,
            auto_dispatch_per_min: 0.35,
            include_nfpa_turnout_time: false,
        }
    }
}

/// Per-junction Emergency Vehicle Preemption (`EVP`) runtime state stored on `Junction`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct JunctionEvpState {
    pub active: bool,
    pub target_in_seg: Option<usize>,
    pub target_axis: u8,
    pub transition_rem_s: f64,
    pub hold_linger_s: f64,
    pub preemption_activations: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EvpTelemetry {
    pub active_responders: usize,
    pub preempted_junctions: usize,
    pub preempted_segments: usize,
    pub yielding_vehicles: usize,
    pub completed_ev_trips: u64,
    pub mean_ev_speed_mph: f64,
    pub total_preemptions_triggered: u64,
}

/// Overrides the normal Fixed-Time or Max-Pressure `LightColor` at `junc` when `evp_state.active`
/// is engaged by an approaching `VehicleClass::EmergencyResponder`.
#[inline]
pub fn evp_override_connector_light(
    state: &JunctionEvpState,
    conn: &TurnConnector,
    base_light: LightColor,
) -> Option<LightColor> {
    if !state.active {
        return None;
    }
    let on_evp_axis = conn.approach_axis == state.target_axis;
    let on_exact_seg = state.target_in_seg == Some(conn.from_seg_id);

    // During MUTCD safe transition clearance, conflicting green movements step cleanly through Yellow
    if state.transition_rem_s > 0.0 {
        if (on_exact_seg || on_evp_axis) && base_light == LightColor::Green {
            return Some(LightColor::Green);
        }
        if !on_evp_axis && base_light == LightColor::Green {
            return Some(LightColor::Yellow);
        }
        return Some(LightColor::Red);
    }

    // Active corridor green-wave lock: responder's approach (all turns) + non-conflicting same-axis through/right get Green!
    if on_exact_seg || (on_evp_axis && conn.turn_type != TurnType::Left) {
        Some(LightColor::Green)
    } else {
        Some(LightColor::Red)
    }
}

/// Advances multi-hop corridor green-wave signal preemption across `junctions`, populates
/// `evp_corridor_segments` (for GPU green-wave rendering), and returns per-segment active siren
/// rear-most `s_m` coordinates for `O(1)` `MOBIL` pull-over & shoulder yielding checks.
#[allow(clippy::too_many_arguments)]
pub fn step_emergency_preemption(
    dt: f64,
    cfg: &EvpConfig,
    junctions: &mut [Junction],
    segments: &[RoadSegment],
    lanes: &[Lane],
    connectors: &[TurnConnector],
    next_hop: &HashMap<(usize, usize), usize>,
    vehicles: &[Vehicle],
    lane_vehs: &[Vec<usize>],
    evp_corridor_segments: &mut Vec<bool>,
    completed_ev_trips: u64,
    yielding_vehicles: usize,
) -> (EvpTelemetry, Vec<Option<f64>>) {
    let n_segs = segments.len();
    evp_corridor_segments.clear();
    evp_corridor_segments.resize(n_segs, false);
    let mut siren_seg_s: Vec<Option<f64>> = vec![None; n_segs];

    if !cfg.enabled {
        for j in junctions.iter_mut() {
            j.evp_state.active = false;
            j.evp_state.transition_rem_s = 0.0;
            j.evp_state.hold_linger_s = 0.0;
        }
        return (
            EvpTelemetry {
                completed_ev_trips,
                ..EvpTelemetry::default()
            },
            siren_seg_s,
        );
    }

    // Collect desired `(junction_id -> (in_seg_id, approach_axis, dist_m))` keeping closest responder per junction
    let mut desired_preempt: Vec<Option<(usize, u8, f64)>> = vec![None; junctions.len()];
    let mut ev_count = 0usize;
    let mut ev_speed_sum_mph = 0.0;

    for veh in vehicles {
        if veh.v_class != VehicleClass::EmergencyResponder {
            continue;
        }
        ev_count += 1;
        ev_speed_sum_mph += veh.v * 2.236_94;

        let (start_sid, rem_on_cur_m) = match veh.carrier {
            CarrierRef::Lane(lid) => {
                let l = &lanes[lid];
                let sid = l.segment_id;
                siren_seg_s[sid] = Some(
                    siren_seg_s[sid]
                        .map(|prev| prev.min(veh.s_m))
                        .unwrap_or(veh.s_m),
                );
                (sid, (l.length - veh.s_m).max(0.0))
            }
            CarrierRef::Connector(cid) => {
                let conn = &connectors[cid];
                let sid = conn.to_seg_id;
                siren_seg_s[sid] = Some(0.0);
                (sid, segments[sid].length)
            }
        };

        evp_corridor_segments[start_sid] = true;

        // Walk up to `max_corridor_hops` downstream intersections along the responder's shortest-path route
        let mut cur_sid = start_sid;
        let mut dist_acc_m = rem_on_cur_m;

        for hop in 0..cfg.max_corridor_hops.max(1) {
            if segments[cur_sid].closed {
                break;
            }
            let jid = segments[cur_sid].to_junc;
            let axis = segments[cur_sid].approach_axis;

            // FHWA Chapter 5 queue clearance bonus `Gq = 3 + 2n` so queued vehicles flush ahead of EV arrival
            let queued_on_seg: usize = segments[cur_sid]
                .lane_ids
                .iter()
                .map(|&lid| {
                    lane_vehs[lid]
                        .iter()
                        .filter(|&&vi| vehicles[vi].v < 2.5)
                        .count()
                })
                .max()
                .unwrap_or(0);
            let queue_flush_bonus_m =
                (3.0 + 2.0 * queued_on_seg as f64) * veh.v.max(11.0) * 0.45;
            let eff_range_m = cfg.lookahead_distance_m + queue_flush_bonus_m.min(140.0);

            if hop == 0 || dist_acc_m <= eff_range_m {
                evp_corridor_segments[cur_sid] = true;
                let replace = desired_preempt[jid]
                    .map(|(_, _, best_d)| dist_acc_m < best_d)
                    .unwrap_or(true);
                if replace {
                    desired_preempt[jid] = Some((cur_sid, axis, dist_acc_m));
                }
            } else {
                break;
            }

            // Propagate acoustic siren warning across the junction into the immediate downstream hop
            let next_opt = if let Some(dst) = veh.dest_seg_id {
                next_hop.get(&(cur_sid, dst)).copied()
            } else {
                junctions[jid]
                    .out_segs
                    .iter()
                    .copied()
                    .find(|&osid| !segments[osid].closed)
            };

            let Some(next_sid) = next_opt else { break };
            if hop == 0 && rem_on_cur_m < cfg.siren_yield_radius_m * 0.65 {
                // Alert vehicles near the entry of `next_sid` so the receiving lane clears early
                siren_seg_s[next_sid] = Some(
                    siren_seg_s[next_sid]
                        .map(|p| p.min(-rem_on_cur_m))
                        .unwrap_or(-rem_on_cur_m),
                );
            }
            dist_acc_m += segments[next_sid].length;
            cur_sid = next_sid;
        }
    }

    let mut preempted_junctions = 0usize;
    let mut total_activations = 0u64;

    for (jid, junc) in junctions.iter_mut().enumerate() {
        let is_signalized = matches!(
            junc.control_mode,
            ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::MaxPressureAdaptive
        );
        if !is_signalized {
            junc.evp_state.active = false;
            continue;
        }

        if let Some((in_sid, req_axis, _dist)) = desired_preempt[jid] {
            if !junc.evp_state.active {
                // Check whether a conflicting axis was actively displaying Green
                let had_conflict_green = match junc.control_mode {
                    ControlMode::MaxPressureAdaptive => {
                        let a_idx = junc.mp_state.active_phase.index();
                        let mp_axis = if a_idx < 2 { 0 } else { 1 };
                        junc.mp_state.stage == MpTransitionStage::Green && mp_axis != req_axis
                    }
                    ControlMode::Signal2Phase | ControlMode::Signal4PhaseProtectedLeft => true,
                    _ => false,
                };
                junc.evp_state.transition_rem_s = if had_conflict_green {
                    cfg.min_clearance_yellow_s
                } else {
                    0.0
                };
                junc.evp_state.preemption_activations += 1;
            } else if junc.evp_state.target_axis != req_axis {
                junc.evp_state.transition_rem_s = cfg.min_clearance_yellow_s;
            } else {
                junc.evp_state.transition_rem_s = (junc.evp_state.transition_rem_s - dt).max(0.0);
            }

            junc.evp_state.active = true;
            junc.evp_state.target_in_seg = Some(in_sid);
            junc.evp_state.target_axis = req_axis;
            junc.evp_state.hold_linger_s = cfg.post_passage_hold_s;

            // Synchronize Max-Pressure internal phase so post-passage recovery is seamless
            if junc.control_mode == ControlMode::MaxPressureAdaptive
                && junc.evp_state.transition_rem_s <= 0.0
            {
                junc.mp_state.active_phase = if req_axis == 0 {
                    MpPhaseId::Axis0ThroughRight
                } else {
                    MpPhaseId::Axis1ThroughRight
                };
                junc.mp_state.stage = MpTransitionStage::Green;
                junc.mp_state.stage_elapsed_s = 0.0;
            }
        } else if junc.evp_state.active {
            junc.evp_state.transition_rem_s = (junc.evp_state.transition_rem_s - dt).max(0.0);
            junc.evp_state.hold_linger_s -= dt;
            if junc.evp_state.hold_linger_s <= 0.0 {
                junc.evp_state.active = false;
                junc.evp_state.target_in_seg = None;
                junc.evp_state.hold_linger_s = 0.0;
            }
        }

        if junc.evp_state.active {
            preempted_junctions += 1;
        }
        total_activations += junc.evp_state.preemption_activations;
    }

    let preempted_segments = evp_corridor_segments.iter().filter(|&&b| b).count();
    let mean_ev_speed_mph = if ev_count > 0 {
        ev_speed_sum_mph / ev_count as f64
    } else {
        0.0
    };

    (
        EvpTelemetry {
            active_responders: ev_count,
            preempted_junctions,
            preempted_segments,
            yielding_vehicles,
            completed_ev_trips,
            mean_ev_speed_mph,
            total_preemptions_triggered: total_activations,
        },
        siren_seg_s,
    )
}


// ============================================================================
// Emergency Facility Placement (Fire Stations & Trauma Hospitals) & Live
// Multi-Source Dijkstra NFPA 1710 4-Minute (240 s) Coverage Solver
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergencyFacilityKind {
    FireStation,
    Hospital,
}

impl EmergencyFacilityKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::FireStation => "Fire Station",
            Self::Hospital => "Trauma Hospital",
        }
    }

    pub fn short_prefix(self) -> &'static str {
        match self {
            Self::FireStation => "FS",
            Self::Hospital => "HOSP",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyFacility {
    pub id: usize,
    pub kind: EmergencyFacilityKind,
    pub name: String,
    pub segment_id: usize,
    pub s_m: f64,
    pub world_pos: crate::network::Vec2,
    pub heading_rad: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EmergencyCoverageSummary {
    pub fire_station_count: usize,
    pub hospital_count: usize,
    /// Percentage of city parcels reachable within <= 240 s (4.0 min) per NFPA 1710 § 4.1.2.1(3).
    pub parcels_within_4min_pct: f64,
    /// Percentage of city residents reachable within <= 240 s (4.0 min).
    pub residents_within_4min_pct: f64,
    /// Percentage of city parcels reachable within <= 480 s (8.0 min ALS / full-alarm standard).
    pub parcels_within_8min_als_pct: f64,
    pub mean_fire_travel_s: f64,
    /// 90th-percentile fractal fire travel time (NFPA 1710 targets P90 <= 240 s).
    pub p90_fire_travel_s: f64,
    /// Mean emergency ambulance transport time from parcels to the nearest Trauma Hospital (s).
    pub mean_hospital_transport_s: f64,
}

/// Seeds 1 initial municipal Fire Station and 1 Regional Trauma Hospital on separated open
/// arterial corridors so every newly compiled map starts with realistic emergency infrastructure.
pub fn synthesize_default_facilities(net: &crate::network::RoadNetwork) -> Vec<EmergencyFacility> {
    let open_arterials: Vec<&RoadSegment> = net
        .segments
        .iter()
        .filter(|s| !s.closed && !s.lane_ids.is_empty() && s.length >= 24.0)
        .collect();
    if open_arterials.is_empty() {
        return Vec::new();
    }

    let b = net.bounds;
    let west_target = crate::network::Vec2 {
        x: b.min_x + 0.28 * (b.max_x - b.min_x),
        y: 0.50 * (b.min_y + b.max_y),
    };
    let east_target = crate::network::Vec2 {
        x: b.min_x + 0.74 * (b.max_x - b.min_x),
        y: b.min_y + 0.34 * (b.max_y - b.min_y),
    };

    let pick_closest = |target: crate::network::Vec2, avoid_sid: Option<usize>| -> &RoadSegment {
        open_arterials
            .iter()
            .copied()
            .filter(|s| Some(s.id) != avoid_sid)
            .min_by(|a, b_seg| {
                let da = a.base_pts[a.base_pts.len() / 2].dist(target) - (a.priority as f64) * 18.0;
                let db = b_seg.base_pts[b_seg.base_pts.len() / 2].dist(target)
                    - (b_seg.priority as f64) * 18.0;
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(open_arterials[0])
    };

    let fs_seg = pick_closest(west_target, None);
    let fs_s = fs_seg.length * 0.5;
    let (fs_pt, fs_ang) =
        crate::network::sample_polyline_fast(&fs_seg.base_pts, &fs_seg.base_cum_s, fs_s);
    let fs_off = (fs_seg.lane_count as f64 + 1.1) * crate::network::LANE_WIDTH_M;
    let fs_pos = crate::network::Vec2 {
        x: fs_pt.x - fs_ang.sin() * fs_off,
        y: fs_pt.y + fs_ang.cos() * fs_off,
    };

    let hosp_seg = pick_closest(east_target, Some(fs_seg.id));
    let hosp_s = hosp_seg.length * 0.5;
    let (hp_pt, hp_ang) =
        crate::network::sample_polyline_fast(&hosp_seg.base_pts, &hosp_seg.base_cum_s, hosp_s);
    let hp_off = (hosp_seg.lane_count as f64 + 1.2) * crate::network::LANE_WIDTH_M;
    let hp_pos = crate::network::Vec2 {
        x: hp_pt.x - hp_ang.sin() * hp_off,
        y: hp_pt.y + hp_ang.cos() * hp_off,
    };

    vec![
        EmergencyFacility {
            id: 0,
            kind: EmergencyFacilityKind::FireStation,
            name: "Fire Station #1".to_string(),
            segment_id: fs_seg.id,
            s_m: fs_s,
            world_pos: fs_pos,
            heading_rad: fs_ang,
        },
        EmergencyFacility {
            id: 1,
            kind: EmergencyFacilityKind::Hospital,
            name: "Regional Trauma Hospital".to_string(),
            segment_id: hosp_seg.id,
            s_m: hosp_s,
            world_pos: hp_pos,
            heading_rad: hp_ang,
        },
    ]
}

/// Evaluates multi-source live network travel times (seconds) from all placed `FireStation`s (forward
/// response) and into all placed `Hospital`s (reverse patient transport), accounting for live
/// corridor speeds, road closures, and whether `EvpConfig::enabled` grants green-wave signal preemption.
pub fn evaluate_emergency_coverage(
    net: &crate::network::RoadNetwork,
    parcels: &mut [crate::zoning::Parcel],
    facilities: &[EmergencyFacility],
    evp_cfg: &EvpConfig,
) -> (EmergencyCoverageSummary, Vec<f64>, Vec<f64>) {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Copy, Clone, PartialEq)]
    struct State {
        cost_s: f64,
        seg_id: usize,
    }
    impl Eq for State {}
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.cost_s.partial_cmp(&self.cost_s).unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = net.segments.len();
    let mut fwd_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut rev_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    let speed_boost = if evp_cfg.enabled { 1.32 } else { 1.02 };
    let seg_trav_s: Vec<f64> = net
        .segments
        .iter()
        .map(|s| {
            if s.closed {
                f64::INFINITY
            } else {
                let eff_v = (s.ema_speed_mps.max(s.speed_mps * 0.48) * speed_boost).clamp(3.2, 29.0);
                let stall_pen = if s.stalled_lane.is_some() && s.lane_count == 1 {
                    28.0
                } else {
                    0.0
                };
                s.length / eff_v + stall_pen
            }
        })
        .collect();

    for seg in &net.segments {
        if seg.closed {
            continue;
        }
        let junc = &net.junctions[seg.to_junc];
        let sig_pen = match junc.control_mode {
            ControlMode::Signal2Phase
            | ControlMode::Signal4PhaseProtectedLeft
            | ControlMode::MaxPressureAdaptive => {
                if evp_cfg.enabled {
                    2.2 // Preempted green-wave intersection traversal
                } else {
                    14.5 // Expected un-preempted red-phase stop delay
                }
            }
            ControlMode::Roundabout => 3.4,
            ControlMode::PermissiveYield => 1.8,
        };

        for &lid in &seg.lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let conn = &net.connectors[cid];
                let out_sid = conn.to_seg_id;
                if net.segments[out_sid].closed {
                    continue;
                }
                let turn_pen = match conn.turn_type {
                    TurnType::Through => 0.5,
                    TurnType::Right => 2.2,
                    TurnType::Left => 3.0,
                };
                let step_cost = sig_pen + turn_pen + seg_trav_s[out_sid];
                if !fwd_adj[seg.id].iter().any(|&(t, _)| t == out_sid) {
                    fwd_adj[seg.id].push((out_sid, step_cost));
                    rev_adj[out_sid].push((seg.id, sig_pen + turn_pen + seg_trav_s[seg.id]));
                }
            }
        }
    }

    let fire_turnout_s = if evp_cfg.include_nfpa_turnout_time {
        80.0
    } else {
        0.0
    };
    let ems_turnout_s = if evp_cfg.include_nfpa_turnout_time {
        60.0
    } else {
        0.0
    };

    // 1. Multi-source forward Dijkstra from all FireStations
    let mut seg_fire_s = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    let mut fire_station_count = 0usize;
    let mut hospital_count = 0usize;

    for fac in facilities {
        match fac.kind {
            EmergencyFacilityKind::FireStation => {
                fire_station_count += 1;
                if fac.segment_id < n && !net.segments[fac.segment_id].closed {
                    let rem_frac = ((net.segments[fac.segment_id].length - fac.s_m)
                        / net.segments[fac.segment_id].length.max(1.0))
                    .clamp(0.1, 1.0);
                    let init_s = fire_turnout_s + seg_trav_s[fac.segment_id] * rem_frac;
                    if init_s < seg_fire_s[fac.segment_id] {
                        seg_fire_s[fac.segment_id] = init_s;
                        heap.push(State {
                            cost_s: init_s,
                            seg_id: fac.segment_id,
                        });
                    }
                    // Also seed twin reverse segment if station sits on a two-way street!
                    let u = net.segments[fac.segment_id].from_junc;
                    let v = net.segments[fac.segment_id].to_junc;
                    for s in &net.segments {
                        if s.from_junc == v && s.to_junc == u && !s.closed {
                            let twin_init = fire_turnout_s + seg_trav_s[s.id] * (1.0 - rem_frac + 0.15);
                            if twin_init < seg_fire_s[s.id] {
                                seg_fire_s[s.id] = twin_init;
                                heap.push(State {
                                    cost_s: twin_init,
                                    seg_id: s.id,
                                });
                            }
                        }
                    }
                }
            }
            EmergencyFacilityKind::Hospital => {
                hospital_count += 1;
            }
        }
    }

    while let Some(State { cost_s, seg_id }) = heap.pop() {
        if cost_s > seg_fire_s[seg_id] {
            continue;
        }
        for &(nxt, w) in &fwd_adj[seg_id] {
            let nc = cost_s + w;
            if nc < seg_fire_s[nxt] {
                seg_fire_s[nxt] = nc;
                heap.push(State {
                    cost_s: nc,
                    seg_id: nxt,
                });
            }
        }
    }

    // 2. Multi-source reverse Dijkstra into all Trauma Hospitals
    let mut seg_hosp_s = vec![f64::INFINITY; n];
    heap.clear();
    for fac in facilities {
        if fac.kind == EmergencyFacilityKind::Hospital
            && fac.segment_id < n
            && !net.segments[fac.segment_id].closed
        {
            let frac = (fac.s_m / net.segments[fac.segment_id].length.max(1.0)).clamp(0.1, 1.0);
            let init_s = ems_turnout_s + seg_trav_s[fac.segment_id] * frac;
            if init_s < seg_hosp_s[fac.segment_id] {
                seg_hosp_s[fac.segment_id] = init_s;
                heap.push(State {
                    cost_s: init_s,
                    seg_id: fac.segment_id,
                });
            }
            let u = net.segments[fac.segment_id].from_junc;
            let v = net.segments[fac.segment_id].to_junc;
            for s in &net.segments {
                if s.from_junc == v && s.to_junc == u && !s.closed {
                    let twin_init = ems_turnout_s + seg_trav_s[s.id] * (1.0 - frac + 0.15);
                    if twin_init < seg_hosp_s[s.id] {
                        seg_hosp_s[s.id] = twin_init;
                        heap.push(State {
                            cost_s: twin_init,
                            seg_id: s.id,
                        });
                    }
                }
            }
        }
    }

    while let Some(State { cost_s, seg_id }) = heap.pop() {
        if cost_s > seg_hosp_s[seg_id] {
            continue;
        }
        for &(prev_sid, w) in &rev_adj[seg_id] {
            let nc = cost_s + w;
            if nc < seg_hosp_s[prev_sid] {
                seg_hosp_s[prev_sid] = nc;
                heap.push(State {
                    cost_s: nc,
                    seg_id: prev_sid,
                });
            }
        }
    }

    // 3. Evaluate parcel-level NFPA 1710 4-minute (<= 240 s) & 8-minute (<= 480 s) coverage
    let mut within_4m = 0usize;
    let mut within_8m = 0usize;
    let mut tot_res = 0.0;
    let mut res_within_4m = 0.0;
    let mut fire_times = Vec::with_capacity(parcels.len());
    let mut hosp_sum = 0.0;
    let mut hosp_cnt = 0usize;

    for p in parcels.iter_mut() {
        let mut best_f = f64::INFINITY;
        let mut best_h = f64::INFINITY;
        for (sid, seg) in net.segments.iter().enumerate() {
            if seg.closed || seg.base_pts.is_empty() {
                continue;
            }
            let mid = seg.base_pts[seg.base_pts.len() / 2];
            let access_s = p.center.dist(mid) / 7.5; // local block-drive + hose-lay access speed
            if seg_fire_s[sid].is_finite() {
                best_f = best_f.min(seg_fire_s[sid] + access_s);
            }
            if seg_hosp_s[sid].is_finite() {
                best_h = best_h.min(seg_hosp_s[sid] + access_s);
            }
        }
        p.fire_travel_s = if best_f.is_finite() { best_f } else { 999.0 };
        p.hospital_travel_s = if best_h.is_finite() { best_h } else { 999.0 };

        let r = p.residents();
        tot_res += r;
        if p.fire_travel_s <= 240.0 {
            within_4m += 1;
            res_within_4m += r;
        }
        if p.fire_travel_s <= 480.0 {
            within_8m += 1;
        }
        fire_times.push(p.fire_travel_s);
        if best_h.is_finite() {
            hosp_sum += best_h;
            hosp_cnt += 1;
        }
    }

    fire_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p_len = parcels.len().max(1) as f64;
    let mean_fire = if fire_times.is_empty() {
        0.0
    } else {
        fire_times.iter().sum::<f64>() / fire_times.len() as f64
    };
    let p90_fire = if fire_times.is_empty() {
        0.0
    } else {
        let idx = (((fire_times.len() - 1) as f64) * 0.90).round() as usize;
        fire_times[idx.min(fire_times.len() - 1)]
    };

    let summary = EmergencyCoverageSummary {
        fire_station_count,
        hospital_count,
        parcels_within_4min_pct: (within_4m as f64 / p_len) * 100.0,
        residents_within_4min_pct: if tot_res > 0.0 {
            (res_within_4m / tot_res) * 100.0
        } else {
            (within_4m as f64 / p_len) * 100.0
        },
        parcels_within_8min_als_pct: (within_8m as f64 / p_len) * 100.0,
        mean_fire_travel_s: mean_fire,
        p90_fire_travel_s: p90_fire,
        mean_hospital_transport_s: if hosp_cnt > 0 {
            hosp_sum / hosp_cnt as f64
        } else {
            999.0
        },
    };

    (summary, seg_fire_s, seg_hosp_s)
}

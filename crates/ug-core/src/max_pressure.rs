//! Decentralized Max-Pressure adaptive traffic signal controller (Varaiya 2013) with
//! capacity-aware downstream spillback gating (Gregoire et al. 2015), multimodal GTFS onboard
//! passenger weighting, switching-loss hysteresis, and Yellow -> All-Red clearance transitions.

use crate::network::{
    ControlMode, Junction, Lane, LightColor, RoadSegment, TurnConnector, TurnType,
};
use crate::traffic::{Vehicle, VehicleClass};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Conflict-free NEMA-style intersection signal stages dynamically selectable by Max-Pressure.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MpPhaseId {
    #[default]
    Axis0ThroughRight = 0,
    Axis0ProtectedLeft = 1,
    Axis1ThroughRight = 2,
    Axis1ProtectedLeft = 3,
}

impl MpPhaseId {
    pub const ALL: [Self; 4] = [
        Self::Axis0ThroughRight,
        Self::Axis0ProtectedLeft,
        Self::Axis1ThroughRight,
        Self::Axis1ProtectedLeft,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Axis0ThroughRight => "Axis A Thru/Rt",
            Self::Axis0ProtectedLeft => "Axis A Prot. Left",
            Self::Axis1ThroughRight => "Axis B Thru/Rt",
            Self::Axis1ProtectedLeft => "Axis B Prot. Left",
        }
    }
}

/// Physical clearance sub-stage ensuring safe Yellow and All-Red intervals on every phase switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MpTransitionStage {
    #[default]
    Green,
    Yellow {
        next_phase: MpPhaseId,
    },
    AllRed {
        next_phase: MpPhaseId,
    },
}

/// Per-junction decentralized Max-Pressure controller state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JunctionMpState {
    pub active_phase: MpPhaseId,
    pub stage: MpTransitionStage,
    pub stage_elapsed_s: f64,
    /// Accumulated red wait time (seconds) per phase when non-empty, preventing side-street starvation.
    pub phase_wait_s: [f64; 4],
    /// Latest instantaneous phase pressures `P(p)` (PCE * veh/s) for UI inspection & telemetry.
    pub phase_pressures: [f64; 4],
    pub spillback_gated_count: u16,
    pub phase_switches: u64,
}

impl Default for JunctionMpState {
    fn default() -> Self {
        Self {
            active_phase: MpPhaseId::Axis0ThroughRight,
            stage: MpTransitionStage::Green,
            stage_elapsed_s: 0.0,
            phase_wait_s: [0.0; 4],
            phase_pressures: [0.0; 4],
            spillback_gated_count: 0,
            phase_switches: 0,
        }
    }
}

/// Tunable parameters for citywide Max-Pressure signal control.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MaxPressureConfig {
    /// Minimum green time (s) before a phase switch is permitted (pedestrian & driver safety floor).
    pub min_green_s: f64,
    /// Maximum green time (s) when competing phases have non-zero waiting queues.
    pub max_green_s: f64,
    /// Fractional pressure margin (`0.0..0.60`) required to overcome yellow + all-red switching loss.
    pub hysteresis_ratio: f64,
    /// Gregoire et al. (2015) normalized downstream storage penalty & spillback gate (`occ >= 0.88`).
    pub capacity_aware_spillback: bool,
    /// Weights queued TransitBus vehicles by onboard passenger count (`1.8 + 0.32 * onboard_pax`).
    pub passenger_weighted_transit: bool,
    /// Enables dedicated Protected-Left phases (`Axis0ProtectedLeft`, `Axis1ProtectedLeft`) on 3+ lane approaches.
    pub allow_protected_left_phases: bool,
    /// Starvation fairness bonus added per second of red wait on occupied approaches.
    pub starvation_aging_per_s: f64,
}

impl Default for MaxPressureConfig {
    fn default() -> Self {
        Self {
            min_green_s: 7.0,
            max_green_s: 48.0,
            hysteresis_ratio: 0.18,
            capacity_aware_spillback: true,
            passenger_weighted_transit: true,
            allow_protected_left_phases: true,
            starvation_aging_per_s: 0.05,
        }
    }
}

/// Network-wide telemetry across all `ControlMode::MaxPressureAdaptive` junctions.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MaxPressureTelemetry {
    pub controlled_junctions: usize,
    pub mean_junction_pressure: f64,
    pub max_junction_pressure: f64,
    pub total_phase_switches: u64,
    pub spillback_gated_movements: usize,
}

/// Determines which `MpPhaseId` serves a `TurnConnector`.
/// Dedicated protected-left phases are used on `>= 3`-lane approaches (where left lanes are
/// separated from primary through lanes), while 1- and 2-lane approaches group all same-axis
/// movements together to prevent shared-lane head-of-line blocking.
#[inline]
pub fn phase_for_connector(
    conn: &TurnConnector,
    in_seg_lane_count: usize,
    allow_protected_left: bool,
) -> MpPhaseId {
    let split_left = allow_protected_left && in_seg_lane_count >= 3 && conn.turn_type == TurnType::Left;
    if conn.approach_axis == 0 {
        if split_left {
            MpPhaseId::Axis0ProtectedLeft
        } else {
            MpPhaseId::Axis0ThroughRight
        }
    } else if split_left {
        MpPhaseId::Axis1ProtectedLeft
    } else {
        MpPhaseId::Axis1ThroughRight
    }
}

/// Returns the instantaneous `LightColor` for `conn` at a `MaxPressureAdaptive` junction.
#[inline]
pub fn mp_connector_light(
    state: &JunctionMpState,
    conn: &TurnConnector,
    in_seg_lane_count: usize,
    allow_protected_left: bool,
) -> LightColor {
    let conn_phase = phase_for_connector(conn, in_seg_lane_count, allow_protected_left);
    match state.stage {
        MpTransitionStage::Green => {
            if conn_phase == state.active_phase {
                LightColor::Green
            } else {
                LightColor::Red
            }
        }
        MpTransitionStage::Yellow { .. } => {
            if conn_phase == state.active_phase {
                LightColor::Yellow
            } else {
                LightColor::Red
            }
        }
        MpTransitionStage::AllRed { .. } => LightColor::Red,
    }
}

#[inline]
fn movement_saturation_veh_per_s(turn: TurnType) -> f64 {
    match turn {
        TurnType::Through => 0.52, // ~1,872 veh/h/ln
        TurnType::Right => 0.45,   // ~1,620 veh/h/ln
        TurnType::Left => 0.42,    // ~1,512 veh/h/ln
    }
}

/// Updates all `ControlMode::MaxPressureAdaptive` junctions in parallel (`Rayon`) from live
/// per-lane upstream & downstream vehicle queues.
pub fn step_max_pressure_controllers(
    dt: f64,
    config: &MaxPressureConfig,
    junctions: &mut [Junction],
    segments: &[RoadSegment],
    lanes: &[Lane],
    connectors: &[TurnConnector],
    vehicles: &[Vehicle],
    lane_vehs: &[Vec<usize>],
) -> MaxPressureTelemetry {
    // 1. Precompute downstream Passenger-Car-Equivalent (PCE) occupancy & physical storage capacity per lane
    let n_lanes = lanes.len();
    let mut lane_down_pce = vec![0.0f64; n_lanes];
    let mut lane_cap_pce = vec![2.0f64; n_lanes];

    for lid in 0..n_lanes {
        lane_cap_pce[lid] = (lanes[lid].length / 7.2).max(2.0);
        let mut pce = 0.0;
        for &vi in &lane_vehs[lid] {
            let v = &vehicles[vi];
            pce += match v.v_class {
                VehicleClass::Sedan => 1.0,
                VehicleClass::FreightTruck => 1.65,
                VehicleClass::TransitBus => 2.1,
                VehicleClass::EmergencyResponder => 1.8,
            };
        }
        lane_down_pce[lid] = pce;
    }

    // 2. Evaluate each MaxPressureAdaptive junction independently in parallel via Rayon
    junctions.par_iter_mut().for_each(|junc| {
        if junc.control_mode != ControlMode::MaxPressureAdaptive
            || junc.hold_phase
            || junc.evp_state.active
        {
            return;
        }

        let mut phase_press = [0.0f64; 4];
        let mut phase_raw_up = [0.0f64; 4];
        let mut gated_cnt = 0u16;

        // Compute movement weights w(l -> m) across all turn connectors at this junction
        for &cid in &junc.connectors {
            let conn = &connectors[cid];
            if segments[conn.from_seg_id].closed || segments[conn.to_seg_id].closed {
                continue;
            }
            let in_lc = segments[conn.from_seg_id].lane_count;
            let p_id = phase_for_connector(conn, in_lc, config.allow_protected_left_phases);
            let p_idx = p_id.index();

            let src_lid = conn.from_lane_id;
            let dst_lid = conn.to_lane_id;
            let src_len = lanes[src_lid].length.max(4.0);
            let src_list = &lane_vehs[src_lid];

            // Upstream queue weight specifically demanding `cid` (or uncommitted proportional fallback)
            let n_out = lanes[src_lid].out_connectors.len().max(1) as f64;
            let lead_conn = src_list.last().and_then(|&lvi| vehicles[lvi].planned_conn_id);

            let mut up_w = 0.0f64;
            for &vi in src_list {
                let veh = &vehicles[vi];
                let contrib = match veh.planned_conn_id {
                    Some(pcid) if pcid == cid => 1.0,
                    None => 1.0 / n_out,
                    _ => 0.0,
                };
                if contrib <= 0.0 {
                    continue;
                }
                let person_pce = match veh.v_class {
                    VehicleClass::Sedan => 1.0,
                    VehicleClass::FreightTruck => 1.6,
                    VehicleClass::EmergencyResponder => 3.5,
                    VehicleClass::TransitBus => {
                        if config.passenger_weighted_transit {
                            1.8 + 0.32 * (veh.onboard_pax as f64)
                        } else {
                            2.0
                        }
                    }
                };
                // Position-weighted urgency: vehicles queued near the stop bar exert strongest pressure
                let pos_norm = (veh.s_m / src_len).clamp(0.0, 1.0);
                let queue_boost = if veh.v < 2.2 { 1.35 } else { 0.85 };
                up_w += contrib * person_pce * (0.35 + 0.65 * pos_norm) * queue_boost;
            }

            // Discount movement if stop-bar lead vehicle is committed to a different phase on a shared lane
            if let Some(l_cid) = lead_conn {
                let lead_phase = phase_for_connector(
                    &connectors[l_cid],
                    in_lc,
                    config.allow_protected_left_phases,
                );
                if lead_phase != p_id {
                    up_w *= 0.25;
                }
            }

            phase_raw_up[p_idx] += up_w;

            // Downstream backpressure & Gregoire et al. (2015) capacity-aware spillback gate
            let dst_pce = lane_down_pce[dst_lid];
            let dst_cap = lane_cap_pce[dst_lid];
            let dst_occ = (dst_pce / dst_cap).clamp(0.0, 1.35);

            let net_w = if config.capacity_aware_spillback {
                if dst_occ >= 0.88 && !segments[conn.to_seg_id].is_roundabout {
                    gated_cnt = gated_cnt.saturating_add(1);
                    0.0
                } else {
                    let back_p = dst_pce * (0.65 + 1.45 * dst_occ * dst_occ);
                    (up_w - back_p).max(0.0)
                }
            } else {
                (up_w - dst_pce * 0.75).max(0.0)
            };

            let sat = movement_saturation_veh_per_s(conn.turn_type);
            phase_press[p_idx] += sat * net_w;
        }

        junc.mp_state.phase_pressures = phase_press;
        junc.mp_state.spillback_gated_count = gated_cnt;
        junc.mp_state.stage_elapsed_s += dt;

        let cur_idx = junc.mp_state.active_phase.index();
        for p in 0..4 {
            if p == cur_idx && junc.mp_state.stage == MpTransitionStage::Green {
                junc.mp_state.phase_wait_s[p] = 0.0;
            } else if phase_raw_up[p] > 0.08 {
                junc.mp_state.phase_wait_s[p] += dt;
            } else {
                junc.mp_state.phase_wait_s[p] = 0.0;
            }
        }

        match junc.mp_state.stage {
            MpTransitionStage::Green => {
                if junc.mp_state.stage_elapsed_s >= config.min_green_s.max(3.0) {
                    let mut eff_scores = [0.0f64; 4];
                    for p in 0..4 {
                        let aging = if phase_raw_up[p] > 0.08 {
                            config.starvation_aging_per_s * junc.mp_state.phase_wait_s[p]
                        } else {
                            0.0
                        };
                        eff_scores[p] = phase_press[p] + aging;
                    }

                    let mut best_p = cur_idx;
                    let mut best_score = eff_scores[cur_idx];
                    for (p, &sc) in eff_scores.iter().enumerate() {
                        if sc > best_score {
                            best_score = sc;
                            best_p = p;
                        }
                    }

                    let cur_score = eff_scores[cur_idx];
                    let max_green_reached =
                        junc.mp_state.stage_elapsed_s >= config.max_green_s.max(12.0);
                    let beats_hysteresis = best_score
                        > cur_score * (1.0 + config.hysteresis_ratio.max(0.0)) + 0.25;

                    if best_p != cur_idx
                        && best_score > 0.08
                        && (beats_hysteresis || (max_green_reached && phase_raw_up[best_p] > 0.08))
                    {
                        let next_phase = MpPhaseId::ALL[best_p];
                        junc.mp_state.stage = MpTransitionStage::Yellow { next_phase };
                        junc.mp_state.stage_elapsed_s = 0.0;
                    }
                }
            }
            MpTransitionStage::Yellow { next_phase } => {
                if junc.mp_state.stage_elapsed_s >= junc.yellow_s.max(2.0) {
                    junc.mp_state.stage = MpTransitionStage::AllRed { next_phase };
                    junc.mp_state.stage_elapsed_s = 0.0;
                }
            }
            MpTransitionStage::AllRed { next_phase } => {
                if junc.mp_state.stage_elapsed_s >= junc.all_red_s.max(0.5) {
                    junc.mp_state.active_phase = next_phase;
                    junc.mp_state.stage = MpTransitionStage::Green;
                    junc.mp_state.stage_elapsed_s = 0.0;
                    junc.mp_state.phase_wait_s[next_phase.index()] = 0.0;
                    junc.mp_state.phase_switches += 1;
                }
            }
        }
    });

    // 3. Aggregate telemetry across all Max-Pressure junctions
    let mut controlled = 0usize;
    let mut sum_press = 0.0f64;
    let mut max_press = 0.0f64;
    let mut total_switches = 0u64;
    let mut gated_sum = 0usize;

    for j in junctions.iter() {
        if j.control_mode == ControlMode::MaxPressureAdaptive {
            controlled += 1;
            let jp: f64 = j.mp_state.phase_pressures.iter().sum();
            sum_press += jp;
            max_press = max_press.max(jp);
            total_switches += j.mp_state.phase_switches;
            gated_sum += j.mp_state.spillback_gated_count as usize;
        }
    }

    MaxPressureTelemetry {
        controlled_junctions: controlled,
        mean_junction_pressure: if controlled > 0 {
            sum_press / controlled as f64
        } else {
            0.0
        },
        max_junction_pressure: max_press,
        total_phase_switches: total_switches,
        spillback_gated_movements: gated_sum,
    }
}

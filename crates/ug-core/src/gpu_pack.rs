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

//! Zero-allocation `#[repr(C)]` `bytemuck::Pod` GPU instance packers and 256x256 RGBA8 heatmap
//! rasterizer for the custom `wgpu` WGSL instanced render pass in `ug-app`.

use crate::emergency::EmergencyFacilityKind;
use crate::environment::vehicle_specific_power_kw_t;
use crate::network::{sample_polyline_fast, LightColor, TurnType, LANE_WIDTH_M};
use crate::traffic::{CarrierRef, SimulationWorld, VehicleClass};
use crate::utilities::UtilityHeatmapSublayer;
use crate::zoning::ZoneCode;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const HEATMAP_DIM: usize = 256;
pub const HEATMAP_BYTES_PER_ROW: u32 = (HEATMAP_DIM * 4) as u32; // 1024 == 4 * 256 (exact wgpu COPY_BYTES_PER_ROW_ALIGNMENT)
pub const HEATMAP_BYTE_LEN: usize = HEATMAP_DIM * HEATMAP_DIM * 4;

// Bitflags packed into `GpuAgentInstance::flags`
pub const AGENT_KIND_SEDAN: u32 = 0;
pub const AGENT_KIND_BUS: u32 = 1;
pub const AGENT_KIND_TRUCK: u32 = 2;
pub const AGENT_KIND_CYCLIST: u32 = 3;
pub const AGENT_KIND_PEDESTRIAN: u32 = 4;
pub const AGENT_KIND_EMERGENCY: u32 = 5;
pub const AGENT_FLAG_BRAKING: u32 = 1 << 4;
pub const AGENT_FLAG_BLINK_LEFT: u32 = 1 << 5;
pub const AGENT_FLAG_BLINK_RIGHT: u32 = 1 << 6;
pub const AGENT_FLAG_DWELLING: u32 = 1 << 7;

// Road quad styles packed into `GpuRoadQuadInstance::style_kind`
pub const ROAD_STYLE_ASPHALT: u32 = 0;
pub const ROAD_STYLE_BUS_LANE: u32 = 1;
pub const ROAD_STYLE_BIKE_LANE: u32 = 2;
pub const ROAD_STYLE_SIDEWALK: u32 = 3;
pub const ROAD_STYLE_SIGNAL_BAR: u32 = 4;
pub const ROAD_STYLE_CLOSED: u32 = 5;
pub const ROAD_STYLE_BUS_STOP: u32 = 6;
pub const ROAD_STYLE_EVP_PREEMPT: u32 = 7;
pub const ROAD_STYLE_FIRE_STATION: u32 = 8;
pub const ROAD_STYLE_HOSPITAL: u32 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeatmapLayer {
    Off = 0,
    Zoning = 1,
    Isochrone15Min = 2,
    VspEmissions = 3,
    AgentDensity = 4,
    EmergencyResponse4Min = 5,
    ResponseDistricts = 6,
    UtilityLifelines = 7,
    FireFollowingQuake = 8,
    HospitalTraumaSurge = 9,
    AeromedicalMedevac = 10,
    EvacuationContraflow = 11,
    EmergencyShelters = 12,
    CommodityResupply = 13,
    TransitBusEvacuation = 14,
    WarningMobilization = 15,
    FemaCommsArchitecture = 16,
    FemaIsbPods = 17,
    FemaUsarDebris = 18,
    FemaGsbDrcTdms = 19,
    WildfireAlertWestResponse = 20,
    BurnScarDebrisFlow = 21,
    WtpTurbidityStorage = 22,
}

/// Interactive atmospheric wind & turbulent diffusion parameters driving the GPU `@compute` plume pass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlumePhysicsParams {
    pub wind_x_mps: f32,
    pub wind_y_mps: f32,
    pub diffusivity: f32,
    pub decay_rate: f32,
}

impl Default for PlumePhysicsParams {
    fn default() -> Self {
        Self {
            wind_x_mps: 2.8,  // Prevailing westerly breeze (+X east)
            wind_y_mps: -1.1, // Slight southerly-to-northerly drift (-Y north)
            diffusivity: 0.42,
            decay_rate: 0.85,
        }
    }
}

/// 48-byte `#[repr(C)]` uniform buffer for `shaders/plume_compute.wgsl` (16-byte WebGPU uniform aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuPlumeComputeUniform {
    pub world_min: [f32; 2],  // offset 0..8
    pub world_span: [f32; 2], // offset 8..16
    pub wind_mps: [f32; 2],   // offset 16..24
    pub dt_s: f32,            // offset 24..28
    pub diffusivity: f32,     // offset 28..32
    pub decay_rate: f32,      // offset 32..36
    pub source_gain: f32,     // offset 36..40
    pub agent_count: u32,     // offset 40..44
    pub mode_and_reset: u32,  // offset 44..48 (bits 0..7: HeatmapLayer; bit 8: reset history)
}

/// 48-byte `#[repr(C)]` uniform buffer aligned to 16-byte WebGPU `var<uniform>` rules.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuCameraUniform {
    pub cam_center: [f32; 2],    // offset 0..8
    pub viewport_px: [f32; 2],   // offset 8..16
    pub world_min: [f32; 2],     // offset 16..24
    pub world_span: [f32; 2],    // offset 24..32
    pub zoom_px_per_m: f32,      // offset 32..36
    pub sim_time_s: f32,         // offset 36..40
    pub heatmap_opacity: f32,    // offset 40..44
    pub heatmap_mode: u32,       // offset 44..48
}

/// 32-byte `#[repr(C)]` per-instance oriented quad for road lanes, bike/bus lanes, sidewalks, and signal bars.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuRoadQuadInstance {
    pub p0: [f32; 2],      // @location(0) Float32x2 (offset 0..8)
    pub p1: [f32; 2],      // @location(1) Float32x2 (offset 8..16)
    pub half_width_m: f32, // @location(2) Float32   (offset 16..20)
    pub style_kind: u32,   // @location(3) Uint32    (offset 20..24)
    pub rgba: [u8; 4],     // @location(4) Unorm8x4  (offset 24..28)
    pub aux_param: f32,    // @location(5) Float32   (offset 28..32)
}

/// 32-byte `#[repr(C)]` per-instance SDF agent record for 50,000+ vehicles, cyclists, and pedestrians.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuAgentInstance {
    pub pos: [f32; 2],     // @location(0) Float32x2 (offset 0..8)
    pub heading_rad: f32,  // @location(1) Float32   (offset 8..12)
    pub speed_mps: f32,    // @location(2) Float32   (offset 12..16)
    pub half_len_m: f32,   // @location(3) Float32   (offset 16..20)
    pub half_wid_m: f32,   // @location(4) Float32   (offset 20..24)
    pub rgba: [u8; 4],     // @location(5) Unorm8x4  (offset 24..28)
    pub flags: u32,        // @location(6) Uint32    (offset 28..32)
}

/// Reusable zero-allocation staging arena across GUI frames.
pub struct GpuStagingArena {
    pub road_quads: Vec<GpuRoadQuadInstance>,
    pub agents: Vec<GpuAgentInstance>,
    pub heatmap_field_a: Vec<f32>,
    pub heatmap_field_b: Vec<f32>,
    pub heatmap_field_c: Vec<f32>,
    pub heatmap_rgba: Vec<u8>,
}

impl Default for GpuStagingArena {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuStagingArena {
    pub fn new() -> Self {
        Self {
            road_quads: Vec::with_capacity(16_384),
            agents: Vec::with_capacity(65_536),
            heatmap_field_a: vec![0.0; HEATMAP_DIM * HEATMAP_DIM],
            heatmap_field_b: vec![0.0; HEATMAP_DIM * HEATMAP_DIM],
            heatmap_field_c: vec![0.0; HEATMAP_DIM * HEATMAP_DIM],
            heatmap_rgba: vec![0u8; HEATMAP_BYTE_LEN],
        }
    }
}

#[inline]
fn speed_color_rgba(v_class: VehicleClass, mph: f64) -> [u8; 4] {
    match v_class {
        VehicleClass::EmergencyResponder => [248, 250, 252, 255],
        VehicleClass::TransitBus => [248, 113, 113, 255],
        VehicleClass::FreightTruck => [192, 132, 252, 255],
        VehicleClass::Sedan => {
            if mph < 4.5 {
                [244, 63, 94, 255]
            } else if mph < 14.0 {
                [251, 146, 60, 255]
            } else if mph < 24.0 {
                [250, 204, 21, 255]
            } else {
                [52, 211, 153, 255]
            }
        }
    }
}

/// Packs all active motor vehicles, cyclists, and sidewalk pedestrians into `out` in parallel via `Rayon`
/// with sub-frame kinematic extrapolation (`interp_dt`) for butter-smooth 120+ FPS motion.
pub fn pack_agents_into(
    world: &SimulationWorld,
    interp_dt: f64,
    out: &mut Vec<GpuAgentInstance>,
) {
    let n_veh = world.vehicles.len();
    let n_cyc = world.cyclists.len();
    let n_ped = world.pedestrians.len();
    let total = n_veh + n_cyc + n_ped;

    out.resize(total, GpuAgentInstance::default());
    let (veh_slice, rest) = out.split_at_mut(n_veh);
    let (cyc_slice, ped_slice) = rest.split_at_mut(n_cyc);

    let net = &world.network;

    // 1. Vehicles (Sedan, TransitBus, FreightTruck)
    veh_slice
        .par_iter_mut()
        .zip(world.vehicles.par_iter())
        .for_each(|(dst, veh)| {
            let s_ext = (veh.s_m + veh.v * interp_dt + 0.5 * veh.acc * interp_dt * interp_dt).max(0.0);
            let center_s = (s_ext - veh.length_m * 0.5).max(0.0);
            let (base_pos, heading, near_turn) = match veh.carrier {
                CarrierRef::Lane(lid) => {
                    let lane = &net.lanes[lid];
                    let (p, ang) = sample_polyline_fast(&lane.pts, &lane.cum_s, center_s);
                    let rem = lane.length - s_ext;
                    (p, ang, rem < 52.0)
                }
                CarrierRef::Connector(cid) => {
                    let conn = &net.connectors[cid];
                    let (p, ang) = sample_polyline_fast(&conn.pts, &conn.cum_s, center_s);
                    (p, ang, true)
                }
            };

            let nx = -heading.sin();
            let ny = heading.cos();
            let px = (base_pos.x + nx * veh.lat_visual_offset_m) as f32;
            let py = (base_pos.y + ny * veh.lat_visual_offset_m) as f32;

            let kind = match veh.v_class {
                VehicleClass::Sedan => AGENT_KIND_SEDAN,
                VehicleClass::TransitBus => AGENT_KIND_BUS,
                VehicleClass::FreightTruck => AGENT_KIND_TRUCK,
                VehicleClass::EmergencyResponder => AGENT_KIND_EMERGENCY,
            };
            let load_u8 = if veh.crush_capacity > 0 {
                (((veh.onboard_pax as u32) * 255) / (veh.crush_capacity as u32)).min(255)
            } else {
                0
            };
            let vsp_kw_t = vehicle_specific_power_kw_t(veh.v, veh.acc).max(0.0);
            let mass_mult = match veh.v_class {
                VehicleClass::TransitBus => 2.4,
                VehicleClass::FreightTruck => 2.1,
                VehicleClass::EmergencyResponder => 1.85,
                VehicleClass::Sedan => 1.0,
            };
            let brake_spike = if veh.acc < -1.2 { 0.45 } else { 0.0 };
            let vsp_norm = (((0.28 + vsp_kw_t * 0.14 + brake_spike) * mass_mult) / 6.5).clamp(0.02, 1.0);
            let vsp_u8 = (vsp_norm * 255.0).round() as u32;
            let mut flags =
                kind | (((veh.id as u32) & 0xFF) << 8) | (load_u8 << 16) | (vsp_u8 << 24);
            if veh.dwell_remaining_s > 0.02 {
                flags |= AGENT_FLAG_DWELLING | AGENT_FLAG_BRAKING;
            } else if veh.acc < -0.85 || veh.v < 0.35 {
                flags |= AGENT_FLAG_BRAKING;
            }
            if (near_turn && veh.turn_intent == TurnType::Left) || veh.lat_visual_offset_m < -0.35 {
                flags |= AGENT_FLAG_BLINK_LEFT;
            } else if (near_turn && veh.turn_intent == TurnType::Right)
                || veh.lat_visual_offset_m > 0.35
            {
                flags |= AGENT_FLAG_BLINK_RIGHT;
            }

            *dst = GpuAgentInstance {
                pos: [px, py],
                heading_rad: heading as f32,
                speed_mps: veh.v as f32,
                half_len_m: (veh.length_m * 0.5) as f32,
                half_wid_m: (veh.width_m * 0.5) as f32,
                rgba: speed_color_rgba(veh.v_class, veh.v * 2.236_94),
                flags,
            };
        });

    // 2. Protected & Shared Bike-Lane Cyclists
    cyc_slice
        .par_iter_mut()
        .zip(world.cyclists.par_iter())
        .for_each(|(dst, cyc)| {
            let seg = &net.segments[cyc.segment_id];
            let s_ext = (cyc.s_m + cyc.v_mps * interp_dt) % seg.length.max(8.0);
            let (p, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, s_ext);
            let road_outer = if seg.is_oneway {
                (seg.lane_count as f64 * 0.5) * LANE_WIDTH_M + 1.35
            } else {
                (seg.lane_count as f64) * LANE_WIDTH_M + 1.35
            };
            let lat = road_outer + cyc.lateral_jitter_m as f64;
            let nx = -ang.sin();
            let ny = ang.cos();
            *dst = GpuAgentInstance {
                pos: [(p.x + nx * lat) as f32, (p.y + ny * lat) as f32],
                heading_rad: ang as f32,
                speed_mps: cyc.v_mps as f32,
                half_len_m: 1.05,
                half_wid_m: 0.48,
                rgba: [16, 185, 129, 255],
                flags: AGENT_KIND_CYCLIST | (((cyc.segment_id as u32) & 0xFF) << 8),
            };
        });

    // 3. Sidewalk & Crosswalk Pedestrians
    ped_slice
        .par_iter_mut()
        .zip(world.pedestrians.par_iter())
        .for_each(|(dst, ped)| {
            let seg = &net.segments[ped.segment_id];
            let s_ext = (ped.s_m + (ped.walk_dir as f64) * ped.cur_speed_mps * interp_dt)
                .clamp(0.2, seg.length.max(2.0));
            let (p, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, s_ext);
            let sidewalk_base = if seg.is_oneway {
                (seg.lane_count as f64 * 0.5) * LANE_WIDTH_M + 3.1
            } else {
                (seg.lane_count as f64) * LANE_WIDTH_M + 3.1
            };
            let lat = (ped.side_sign as f64) * sidewalk_base + ped.lateral_jitter_m as f64;
            let nx = -ang.sin();
            let ny = ang.cos();
            let rgba = if ped.cur_speed_mps < 0.1 {
                [251, 191, 36, 240] // curb-waiting pedestrian at crosswalk
            } else {
                [56, 189, 248, 245] // walking pedestrian
            };
            let ped_heading = if ped.walk_dir >= 0 {
                ang
            } else {
                ang + std::f64::consts::PI
            };
            *dst = GpuAgentInstance {
                pos: [(p.x + nx * lat) as f32, (p.y + ny * lat) as f32],
                heading_rad: ped_heading as f32,
                speed_mps: ped.cur_speed_mps as f32,
                half_len_m: 0.52,
                half_wid_m: 0.52,
                rgba,
                flags: AGENT_KIND_PEDESTRIAN | ((ped.phase_seed as u32 & 0xFF) << 8),
            };
        });
}

/// Packs road lane segments, dedicated red BRT lanes, emerald bike lanes, sidewalks, and live
/// red/yellow/green signal stop-bars into `out` for single-pass instanced rendering.
pub fn pack_road_quads_into(world: &SimulationWorld, out: &mut Vec<GpuRoadQuadInstance>) {
    out.clear();
    let net = &world.network;

    for seg in &net.segments {
        // Sidewalk ribbons on outer curb
        let curb_off = if seg.is_oneway {
            (seg.lane_count as f64 * 0.5) * LANE_WIDTH_M + 2.9
        } else {
            (seg.lane_count as f64) * LANE_WIDTH_M + 2.9
        };
        for w in seg.base_pts.windows(2) {
            let dx = w[1].x - w[0].x;
            let dy = w[1].y - w[0].y;
            let l = dx.hypot(dy);
            if l < 0.5 {
                continue;
            }
            let nx = -dy / l;
            let ny = dx / l;
            out.push(GpuRoadQuadInstance {
                p0: [(w[0].x + nx * curb_off) as f32, (w[0].y + ny * curb_off) as f32],
                p1: [(w[1].x + nx * curb_off) as f32, (w[1].y + ny * curb_off) as f32],
                half_width_m: 1.15,
                style_kind: ROAD_STYLE_SIDEWALK,
                rgba: [30, 41, 59, 155],
                aux_param: 0.0,
            });
            if seg.has_bike_lane && !seg.closed {
                let bike_off = curb_off - 1.55;
                out.push(GpuRoadQuadInstance {
                    p0: [(w[0].x + nx * bike_off) as f32, (w[0].y + ny * bike_off) as f32],
                    p1: [(w[1].x + nx * bike_off) as f32, (w[1].y + ny * bike_off) as f32],
                    half_width_m: 0.82,
                    style_kind: ROAD_STYLE_BIKE_LANE,
                    rgba: [16, 185, 129, 175],
                    aux_param: 1.0,
                });
            }
        }

        // Metric motor / Bus-BRT lanes
        for &lid in &seg.lane_ids {
            let lane = &net.lanes[lid];
            let (style_kind, rgba) = if seg.closed {
                (ROAD_STYLE_CLOSED, [88, 28, 45, 225])
            } else if seg.has_bus_lane && lane.index + 1 == seg.lane_count {
                (ROAD_STYLE_BUS_LANE, [98, 30, 42, 235])
            } else {
                (ROAD_STYLE_ASPHALT, [28, 37, 54, 240])
            };
            for w in lane.pts.windows(2) {
                out.push(GpuRoadQuadInstance {
                    p0: [w[0].x as f32, w[0].y as f32],
                    p1: [w[1].x as f32, w[1].y as f32],
                    half_width_m: (LANE_WIDTH_M * 0.52) as f32,
                    style_kind,
                    rgba,
                    aux_param: lane.index as f32,
                });
            }
        }

        // Active Emergency Vehicle Preemption (EVP) green-wave corridor guide ribbon
        if world
            .evp_corridor_segments
            .get(seg.id)
            .copied()
            .unwrap_or(false)
            && !seg.closed
        {
            if let Some(&lead_lid) = seg.lane_ids.first() {
                for w in net.lanes[lead_lid].pts.windows(2) {
                    out.push(GpuRoadQuadInstance {
                        p0: [w[0].x as f32, w[0].y as f32],
                        p1: [w[1].x as f32, w[1].y as f32],
                        half_width_m: 0.65,
                        style_kind: ROAD_STYLE_EVP_PREEMPT,
                        rgba: [34, 211, 238, 185],
                        aux_param: 1.0,
                    });
                }
            }
        }
    }

    // Active signalized stop-bars at lane approaches
    for j in &net.junctions {
        for &in_sid in &j.in_segs {
            for &lid in &net.segments[in_sid].lane_ids {
                let lane = &net.lanes[lid];
                let Some(&cid) = lane.out_connectors.first() else {
                    continue;
                };
                if lane.pts.len() < 2 {
                    continue;
                }
                let light = net.connector_light(j, &net.connectors[cid], world.sim_time_s);
                let is_evp_green = j.evp_state.active
                    && j.evp_state.transition_rem_s <= 0.0
                    && light == LightColor::Green;
                let (style_kind, rgba, hw_bar) = if is_evp_green {
                    (ROAD_STYLE_EVP_PREEMPT, [34, 211, 238, 255], 0.95)
                } else {
                    let c = match light {
                        LightColor::Green => [16, 185, 129, 245],
                        LightColor::Yellow => [250, 204, 21, 250],
                        LightColor::Red => [244, 63, 94, 250],
                    };
                    (ROAD_STYLE_SIGNAL_BAR, c, 0.72)
                };
                let end = lane.pts[lane.pts.len() - 1];
                let ang = lane.end_angle;
                let nx = -ang.sin() * (LANE_WIDTH_M * 0.46);
                let ny = ang.cos() * (LANE_WIDTH_M * 0.46);
                out.push(GpuRoadQuadInstance {
                    p0: [(end.x - nx) as f32, (end.y - ny) as f32],
                    p1: [(end.x + nx) as f32, (end.y + ny) as f32],
                    half_width_m: hw_bar,
                    style_kind,
                    rgba,
                    aux_param: 1.0,
                });
            }
        }
    }

    // Curbside GTFS TransitStop shelter pads colored by waiting passenger queue crowding
    for stop in &world.transit_gtfs.stops {
        let c = stop.heading_rad.cos() * 2.1;
        let s = stop.heading_rad.sin() * 2.1;
        let q = stop.waiting_pax;
        let rgba = if q >= 12.0 {
            [244, 63, 94, 250] // Heavy curb queue (>12 waiting riders)
        } else if q >= 4.0 {
            [251, 191, 36, 245] // Moderate queue (4..12 riders)
        } else {
            [56, 189, 248, 235] // Clear / light stop queue
        };
        out.push(GpuRoadQuadInstance {
            p0: [(stop.world_pos.x - c) as f32, (stop.world_pos.y - s) as f32],
            p1: [(stop.world_pos.x + c) as f32, (stop.world_pos.y + s) as f32],
            half_width_m: 0.95,
            style_kind: ROAD_STYLE_BUS_STOP,
            rgba,
            aux_param: q as f32,
        });
    }

    // Placed Emergency Facilities (Fire Stations & Trauma Hospitals)
    for fac in &world.emergency_facilities {
        let c = fac.heading_rad.cos() * 4.8;
        let s = fac.heading_rad.sin() * 4.8;
        let (style_kind, rgba) = match fac.kind {
            EmergencyFacilityKind::FireStation => (ROAD_STYLE_FIRE_STATION, [225, 29, 72, 250]),
            EmergencyFacilityKind::Hospital => (ROAD_STYLE_HOSPITAL, [14, 165, 233, 250]),
        };
        out.push(GpuRoadQuadInstance {
            p0: [(fac.world_pos.x - c) as f32, (fac.world_pos.y - s) as f32],
            p1: [(fac.world_pos.x + c) as f32, (fac.world_pos.y + s) as f32],
            half_width_m: 4.8,
            style_kind,
            rgba,
            aux_param: fac.id as f32,
        });
    }
}

fn separable_blur_5tap(field_in: &[f32], temp: &mut [f32], field_out: &mut [f32]) {
    const K: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    // Horizontal pass
    for y in 0..HEATMAP_DIM {
        let row = y * HEATMAP_DIM;
        for x in 0..HEATMAP_DIM {
            let mut acc = 0.0f32;
            for (ki, &kw) in K.iter().enumerate() {
                let sx = (x as isize + ki as isize - 2).clamp(0, (HEATMAP_DIM - 1) as isize) as usize;
                acc += field_in[row + sx] * kw;
            }
            temp[row + x] = acc;
        }
    }
    // Vertical pass
    for y in 0..HEATMAP_DIM {
        for x in 0..HEATMAP_DIM {
            let mut acc = 0.0f32;
            for (ki, &kw) in K.iter().enumerate() {
                let sy = (y as isize + ki as isize - 2).clamp(0, (HEATMAP_DIM - 1) as isize) as usize;
                acc += temp[sy * HEATMAP_DIM + x] * kw;
            }
            field_out[y * HEATMAP_DIM + x] = acc;
        }
    }
}

#[inline]
fn thermal_colormap(t: f32) -> [u8; 4] {
    let x = t.clamp(0.0, 1.0);
    if x < 0.02 {
        return [0, 0, 0, 0];
    }
    let (r, g, b) = if x < 0.33 {
        let u = x / 0.33;
        (30.0 + 125.0 * u, 18.0 + 28.0 * u, 75.0 + 115.0 * u)
    } else if x < 0.68 {
        let u = (x - 0.33) / 0.35;
        (155.0 + 89.0 * u, 46.0 + 72.0 * u, 190.0 - 110.0 * u)
    } else {
        let u = (x - 0.68) / 0.32;
        (244.0 + 11.0 * u, 118.0 + 122.0 * u, 80.0 + 95.0 * u)
    };
    let alpha = ((x.sqrt() * 215.0).round() as i32).clamp(0, 225) as u8;
    [r as u8, g as u8, b as u8, alpha]
}

#[inline]
fn density_plasma_colormap(t: f32) -> [u8; 4] {
    let x = t.clamp(0.0, 1.0);
    if x < 0.015 {
        return [0, 0, 0, 0];
    }
    let (r, g, b) = if x < 0.35 {
        let u = x / 0.35;
        (14.0 + 28.0 * u, 116.0 + 72.0 * u, 144.0 + 98.0 * u)
    } else if x < 0.70 {
        let u = (x - 0.35) / 0.35;
        (42.0 + 208.0 * u, 188.0 + 16.0 * u, 242.0 - 205.0 * u)
    } else {
        let u = (x - 0.70) / 0.30;
        (250.0, 204.0 - 140.0 * u, 37.0 + 57.0 * u)
    };
    let alpha = ((x.sqrt() * 220.0).round() as i32).clamp(0, 230) as u8;
    [r as u8, g as u8, b as u8, alpha]
}

#[inline]
fn response_time_4min_colormap(sec: f32) -> [u8; 4] {
    if !sec.is_finite() {
        return [136, 19, 55, 175];
    }
    // Crisp white-cyan highlight contour right along the 240 s (4.0-minute) NFPA 1710 threshold!
    if (234.0..=246.0).contains(&sec) {
        return [236, 254, 255, 230];
    }
    let (r, g, b, a) = if sec <= 120.0 {
        let u = (sec / 120.0).clamp(0.0, 1.0);
        (16.0 + 20.0 * u, 195.0 + 28.0 * u, 175.0 - 45.0 * u, 178.0)
    } else if sec <= 240.0 {
        let u = ((sec - 120.0) / 120.0).clamp(0.0, 1.0);
        (36.0 + 214.0 * u, 223.0 - 19.0 * u, 130.0 - 109.0 * u, 180.0)
    } else if sec <= 360.0 {
        let u = ((sec - 240.0) / 120.0).clamp(0.0, 1.0);
        (250.0, 204.0 - 78.0 * u, 21.0 + 25.0 * u, 176.0)
    } else if sec <= 480.0 {
        let u = ((sec - 360.0) / 120.0).clamp(0.0, 1.0);
        (248.0 - 6.0 * u, 126.0 - 68.0 * u, 46.0 + 48.0 * u, 178.0)
    } else {
        let u = ((sec - 480.0) / 180.0).clamp(0.0, 1.0);
        (242.0 - 95.0 * u, 58.0 - 38.0 * u, 94.0 - 30.0 * u, 185.0)
    };
    [r as u8, g as u8, b as u8, a as u8]
}

/// Builds a `256x256` RGBA8 GPU texture (`1024` bytes/row) for the active `HeatmapLayer`.
pub fn build_heatmap_rgba(
    world: &SimulationWorld,
    layer: HeatmapLayer,
    agents: &[GpuAgentInstance],
    arena: &mut GpuStagingArena,
) {
    let b = world.network.bounds;
    let span_x = (b.max_x - b.min_x).max(10.0);
    let span_y = (b.max_y - b.min_y).max(10.0);

    arena.heatmap_rgba.fill(0);
    if layer == HeatmapLayer::Off {
        return;
    }

    match layer {
        HeatmapLayer::Off => {}
        HeatmapLayer::Zoning | HeatmapLayer::Isochrone15Min => {
            for p in &world.parcels {
                let x0 = (((p.min.x - b.min_x) / span_x) * HEATMAP_DIM as f64)
                    .floor()
                    .clamp(0.0, (HEATMAP_DIM - 1) as f64) as usize;
                let x1 = (((p.max.x - b.min_x) / span_x) * HEATMAP_DIM as f64)
                    .ceil()
                    .clamp(0.0, HEATMAP_DIM as f64) as usize;
                let y0 = (((p.min.y - b.min_y) / span_y) * HEATMAP_DIM as f64)
                    .floor()
                    .clamp(0.0, (HEATMAP_DIM - 1) as f64) as usize;
                let y1 = (((p.max.y - b.min_y) / span_y) * HEATMAP_DIM as f64)
                    .ceil()
                    .clamp(0.0, HEATMAP_DIM as f64) as usize;

                let col = if layer == HeatmapLayer::Zoning {
                    match p.zone {
                        ZoneCode::R1 => [251, 191, 36, 130],
                        ZoneCode::R3 => [251, 146, 60, 145],
                        ZoneCode::MU4 => [192, 132, 252, 155],
                        ZoneCode::CBD => [56, 189, 248, 165],
                        ZoneCode::IND => [129, 140, 248, 150],
                        ZoneCode::PARK => [16, 185, 129, 165],
                    }
                } else if p.isochrone_minutes <= 5.0 {
                    [16, 185, 129, 165]
                } else if p.isochrone_minutes <= 10.0 {
                    [56, 189, 248, 155]
                } else if p.isochrone_minutes <= 15.0 {
                    [250, 204, 21, 145]
                } else {
                    [244, 63, 94, 145]
                };

                for y in y0..y1 {
                    let row = y * HEATMAP_DIM;
                    for x in x0..x1 {
                        let idx = (row + x) * 4;
                        arena.heatmap_rgba[idx..idx + 4].copy_from_slice(&col);
                    }
                }
            }
        }
        HeatmapLayer::VspEmissions => {
            arena.heatmap_field_a.fill(0.0);
            let n_veh = world.vehicles.len().min(agents.len());
            for i in 0..n_veh {
                let ag = &agents[i];
                let veh = &world.vehicles[i];
                let gx = (((ag.pos[0] as f64 - b.min_x) / span_x) * HEATMAP_DIM as f64) as isize;
                let gy = (((ag.pos[1] as f64 - b.min_y) / span_y) * HEATMAP_DIM as f64) as isize;
                if gx >= 0 && gx < HEATMAP_DIM as isize && gy >= 0 && gy < HEATMAP_DIM as isize {
                    let vsp = vehicle_specific_power_kw_t(veh.v, veh.acc).max(0.0);
                    let mass = match veh.v_class {
                        VehicleClass::TransitBus => 2.4,
                        VehicleClass::FreightTruck => 2.1,
                        VehicleClass::EmergencyResponder => 1.85,
                        VehicleClass::Sedan => 1.0,
                    };
                    let weight = ((0.35 + vsp * 0.14) * mass) as f32;
                    arena.heatmap_field_a[gy as usize * HEATMAP_DIM + gx as usize] += weight;
                }
            }
            // Two-pass separable diffusion for smooth atmospheric dispersion plumes
            separable_blur_5tap(&arena.heatmap_field_a, &mut arena.heatmap_field_b, &mut arena.heatmap_field_c);
            separable_blur_5tap(&arena.heatmap_field_c, &mut arena.heatmap_field_b, &mut arena.heatmap_field_a);

            let norm = if n_veh > 5_000 { 4.2 } else { 1.35 };
            for (i, &val) in arena.heatmap_field_a.iter().enumerate() {
                let rgba = thermal_colormap((val / norm).clamp(0.0, 1.0));
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::AgentDensity => {
            arena.heatmap_field_a.fill(0.0);
            for ag in agents {
                let gx = (((ag.pos[0] as f64 - b.min_x) / span_x) * HEATMAP_DIM as f64) as isize;
                let gy = (((ag.pos[1] as f64 - b.min_y) / span_y) * HEATMAP_DIM as f64) as isize;
                if gx >= 0 && gx < HEATMAP_DIM as isize && gy >= 0 && gy < HEATMAP_DIM as isize {
                    let w = if (ag.flags & 0xF) <= AGENT_KIND_TRUCK {
                        1.0
                    } else {
                        0.65
                    };
                    arena.heatmap_field_a[gy as usize * HEATMAP_DIM + gx as usize] += w;
                }
            }
            separable_blur_5tap(&arena.heatmap_field_a, &mut arena.heatmap_field_b, &mut arena.heatmap_field_c);
            let norm = if agents.len() > 10_000 { 5.5 } else { 1.4 };
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let rgba = density_plasma_colormap((val / norm).clamp(0.0, 1.0));
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::EmergencyResponse4Min => {
            // Bin open segment sample points into a 16x16 coarse spatial index for sub-millisecond 256x256 evaluation
            const BINS: usize = 16;
            let mut grid: Vec<Vec<(f32, f32, f32)>> = vec![Vec::new(); BINS * BINS];
            for (sid, seg) in world.network.segments.iter().enumerate() {
                if seg.closed || seg.base_pts.is_empty() {
                    continue;
                }
                let base_s = world
                    .segment_fire_travel_s
                    .get(sid)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                if !base_s.is_finite() {
                    continue;
                }
                let n_pts = seg.base_pts.len();
                let step = (n_pts / 3).max(1);
                for idx in (0..n_pts).step_by(step) {
                    let pt = seg.base_pts[idx];
                    let bx = (((pt.x - b.min_x) / span_x) * BINS as f64)
                        .floor()
                        .clamp(0.0, (BINS - 1) as f64) as usize;
                    let by = (((pt.y - b.min_y) / span_y) * BINS as f64)
                        .floor()
                        .clamp(0.0, (BINS - 1) as f64) as usize;
                    grid[by * BINS + bx].push((pt.x as f32, pt.y as f32, base_s as f32));
                }
            }

            let has_any = grid.iter().any(|cell| !cell.is_empty());
            if !has_any {
                for px in arena.heatmap_rgba.chunks_exact_mut(4) {
                    px.copy_from_slice(&[136, 19, 55, 175]);
                }
                return;
            }

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                let by = (gy * BINS / HEATMAP_DIM).min(BINS - 1);
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let bx = (gx * BINS / HEATMAP_DIM).min(BINS - 1);
                    let mut best_sec = f32::INFINITY;

                    let y_lo = by.saturating_sub(2);
                    let y_hi = (by + 2).min(BINS - 1);
                    let x_lo = bx.saturating_sub(2);
                    let x_hi = (bx + 2).min(BINS - 1);
                    for cy in y_lo..=y_hi {
                        for cx in x_lo..=x_hi {
                            for &(sx, sy, st) in &grid[cy * BINS + cx] {
                                let d = (wx - sx).hypot(wy - sy);
                                let cand = st + d / 7.2;
                                if cand < best_sec {
                                    best_sec = cand;
                                }
                            }
                        }
                    }
                    // Fallback global sweep if local 5x5 neighborhood was empty on sparse periphery
                    if !best_sec.is_finite() {
                        for cell in &grid {
                            for &(sx, sy, st) in cell {
                                let cand = st + (wx - sx).hypot(wy - sy) / 7.2;
                                if cand < best_sec {
                                    best_sec = cand;
                                }
                            }
                        }
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] = best_sec;
                }
            }

            // Light 5-tap smoothing for clean continuous iso-contours
            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &sec) in arena.heatmap_field_c.iter().enumerate() {
                let rgba = response_time_4min_colormap(sec);
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::ResponseDistricts => {
            // Categorical Carter-Chaiken-Ignall / Larson Response-District (Beat) Territory Map
            // with luminous gold-white inter-district boundary seams!
            const PALETTE: [[u8; 4]; 8] = [
                [244, 63, 94, 152],   // Unit 1: Crimson Rose
                [56, 189, 248, 152],  // Unit 2: Sky Cyan
                [16, 185, 129, 152],  // Unit 3: Emerald
                [250, 204, 21, 152],  // Unit 4: Amber Gold
                [168, 85, 247, 152],  // Unit 5: Royal Violet
                [251, 146, 60, 152],  // Unit 6: Tangerine
                [45, 212, 191, 152],  // Unit 7: Aqua Teal
                [236, 72, 153, 152],  // Unit 8: Magenta
            ];

            let fire_stations: Vec<(f32, f32)> = world
                .emergency_facilities
                .iter()
                .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
                .map(|f| (f.world_pos.x as f32, f.world_pos.y as f32))
                .collect();

            // Build parcel beat centers `(cx, cy, beat_id)` using explicit `response_district_unit`
            // if optimized, or fallback nearest placed `FireStation` index!
            let parcel_beats: Vec<(f32, f32, u16)> = world
                .parcels
                .iter()
                .enumerate()
                .map(|(pi, p)| {
                    let uid = if let Some(u) = p.response_district_unit {
                        u
                    } else if !fire_stations.is_empty() {
                        let px = p.center.x as f32;
                        let py = p.center.y as f32;
                        let mut best_i = 0usize;
                        let mut best_d2 = f32::INFINITY;
                        for (fi, &(fx, fy)) in fire_stations.iter().enumerate() {
                            let d2 = (px - fx) * (px - fx) + (py - fy) * (py - fy);
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_i = fi;
                            }
                        }
                        best_i as u16
                    } else {
                        (pi % 4) as u16
                    };
                    (p.center.x as f32, p.center.y as f32, uid)
                })
                .collect();

            if parcel_beats.is_empty() {
                return;
            }

            let mut beat_grid = vec![0u16; HEATMAP_DIM * HEATMAP_DIM];
            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut best_uid = 0u16;
                    let mut best_d2 = f32::INFINITY;
                    for &(px, py, uid) in &parcel_beats {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py);
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best_uid = uid;
                        }
                    }
                    beat_grid[gy * HEATMAP_DIM + gx] = best_uid;
                }
            }

            for gy in 0..HEATMAP_DIM {
                for gx in 0..HEATMAP_DIM {
                    let idx = gy * HEATMAP_DIM + gx;
                    let uid = beat_grid[idx];
                    let is_seam = (gx + 1 < HEATMAP_DIM && beat_grid[idx + 1] != uid)
                        || (gy + 1 < HEATMAP_DIM && beat_grid[idx + HEATMAP_DIM] != uid);
                    let col = if is_seam {
                        [254, 243, 199, 242]
                    } else {
                        PALETTE[(uid as usize) % PALETTE.len()]
                    };
                    arena.heatmap_rgba[idx * 4..idx * 4 + 4].copy_from_slice(&col);
                }
            }
        }
        HeatmapLayer::UtilityLifelines => {
            // Inverse-distance spatial interpolation of junction-level Electrical Voltage (`p.u.`),
            // Potable Water Residual Pressure (`psi`), Wastewater Sewer `d/D`, or Composite Health!
            let nodes = &world.utility_report.node_telemetry;
            if nodes.is_empty() {
                return;
            }
            let sub = world.utility_config.heatmap_sublayer;
            let node_vals: Vec<(f32, f32, f32)> = nodes
                .iter()
                .map(|n| {
                    // Normalize into a health metric `h in [0.0 (critical red) .. 1.0 (healthy green)]`
                    let h = match sub {
                        UtilityHeatmapSublayer::CompositeLifelineHealth => n.composite_health as f32,
                        UtilityHeatmapSublayer::ElectricVoltagePu => {
                            if !n.energized {
                                0.0
                            } else if (0.95..=1.05).contains(&n.voltage_pu) {
                                (0.72 + 0.28 * (1.0 - ((n.voltage_pu - 1.0).abs() / 0.05))) as f32
                            } else {
                                ((n.voltage_pu - 0.85) / 0.10).clamp(0.05, 0.55) as f32
                            }
                        }
                        UtilityHeatmapSublayer::WaterResidualPressurePsi => {
                            if n.residual_pressure_psi >= 45.0 {
                                1.0
                            } else if n.residual_pressure_psi >= 20.0 {
                                (0.55 + 0.45 * ((n.residual_pressure_psi - 20.0) / 25.0)) as f32
                            } else {
                                (n.residual_pressure_psi / 20.0 * 0.45).clamp(0.0, 0.45) as f32
                            }
                        }
                        UtilityHeatmapSublayer::WastewaterSewerLoading => {
                            if n.sewer_surcharged_sso {
                                0.05
                            } else {
                                (1.0 - (n.sewer_d_over_d / 1.05)).clamp(0.12, 1.0) as f32
                            }
                        }
                    };
                    (n.pos.x as f32, n.pos.y as f32, h.clamp(0.0, 1.0))
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut vh_sum = 0.0f32;
                    for &(nx, ny, nh) in &node_vals {
                        let d2 = (wx - nx) * (wx - nx) + (wy - ny) * (wy - ny) + 225.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        vh_sum += w * nh;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { vh_sum / w_sum } else { 1.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &h_raw) in arena.heatmap_field_c.iter().enumerate() {
                let h = h_raw.clamp(0.0, 1.0);
                // Crisp semantic ramp: Crimson Outage/SSO (`0.0`) -> Amber Sag/Warning (`0.5`) -> Cyan/Emerald Nominal (`1.0`)
                let rgba = if h < 0.48 {
                    let t = h / 0.48;
                    [
                        (239.0 + 11.0 * t) as u8,
                        (44.0 + 150.0 * t) as u8,
                        (68.0 - 36.0 * t) as u8,
                        178,
                    ]
                } else {
                    let t = (h - 0.48) / 0.52;
                    [
                        (250.0 - 218.0 * t) as u8,
                        (194.0 + 20.0 * t) as u8,
                        (32.0 + 132.0 * t) as u8,
                        166,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::FireFollowingQuake => {
            // Smooth spatial ember/conflagration field combining `ffe_burn_prob_pct` and
            // fuel/dry-hydrant vulnerability across `world.parcels`!
            if world.parcels.is_empty() {
                return;
            }
            let has_cached_mc = world.parcels.iter().any(|p| p.ffe_burn_prob_pct > 0.01);
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let risk01 = if has_cached_mc {
                        // Scale so 25%+ burn probability glows incandescent amber-white!
                        (p.ffe_burn_prob_pct / 28.0 + 0.18 * (p.ffe_dry_hydrant_prob_pct / 100.0))
                            .clamp(0.0, 1.0)
                    } else {
                        // Static fuel preview prior to running FFE Monte Carlo
                        match p.zone {
                            ZoneCode::PARK => 0.04,
                            ZoneCode::CBD => 0.35,
                            ZoneCode::IND => 0.52,
                            ZoneCode::R1 => 0.48,
                            ZoneCode::MU4 => 0.62,
                            ZoneCode::R3 => 0.82,
                        }
                    };
                    (p.center.x as f32, p.center.y as f32, risk01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, r01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 260.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * r01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Ember-to-Conflagration palette: Deep Slate (`0.0`) -> Smoldering Crimson (`0.45`) -> Flame Orange (`0.75`) -> Incandescent Gold (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (24.0 + 185.0 * u) as u8,
                        (24.0 + 14.0 * u) as u8,
                        (42.0 + 22.0 * u) as u8,
                        (85.0 + 95.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        (209.0 + 42.0 * u) as u8,
                        (38.0 + 108.0 * u) as u8,
                        (64.0 - 36.0 * u) as u8,
                        190,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        253,
                        (146.0 + 94.0 * u) as u8,
                        (28.0 + 110.0 * u) as u8,
                        212,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::HospitalTraumaSurge => {
            if world.parcels.is_empty() {
                return;
            }
            let has_cached_surge = world
                .parcels
                .iter()
                .any(|p| p.surge_red_burn_treat_min > 0.1);
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let strain01 = if has_cached_surge {
                        // Normalize so <=20 min is cool medical emerald/cyan (0.10), 60 min Golden-Hour
                        // threshold is warm amber-crimson (0.68), and >=95 min ED queue overload is 1.0!
                        let t_norm = ((p.surge_red_burn_treat_min - 12.0) / 78.0).clamp(0.0, 1.0);
                        (0.82 * t_norm + 0.18 * (p.surge_diversion_prob_pct / 100.0))
                            .clamp(0.0, 1.0)
                    } else {
                        ((p.hospital_travel_s as f32) / 480.0).clamp(0.05, 0.95)
                    };
                    (p.center.x as f32, p.center.y as f32, strain01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f32 as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 280.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Golden-Hour Triage Palette: Medical Teal (`0.0`) -> Triage Gold (`0.50`) -> Surge Crimson (`0.78`) -> Critical Magenta (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (16.0 + 218.0 * u) as u8,
                        (185.0 - 10.0 * u) as u8,
                        (142.0 - 110.0 * u) as u8,
                        (95.0 + 75.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        (234.0 + 10.0 * u) as u8,
                        (175.0 - 118.0 * u) as u8,
                        (32.0 + 58.0 * u) as u8,
                        192,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (57.0 + 95.0 * u) as u8,
                        (90.0 + 148.0 * u) as u8,
                        214,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::AeromedicalMedevac => {
            if world.parcels.is_empty() {
                return;
            }
            // Visualizes flight-envelope hazard (`0.0` = clear VFR airlift corridor, `0.55` = PinS IFR
            // smoke margin `1.5..3 SM`, `1.0` = dense FFE smoke `< 0.75 SM` grounding medevac!)
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let vis = p.medevac_smoke_vis_sm.max(0.05);
                    let smoke_obsc01 = (1.0 - ((vis - 0.5) / 4.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                    let airlift_bonus = (p.medevac_airlift_save_min / 25.0).clamp(0.0, 0.25);
                    let val01 = (smoke_obsc01 * 0.88 + (1.0 - smoke_obsc01) * airlift_bonus)
                        .clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 270.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Aviation Flight-Envelope Palette: Clear VFR Sky Cyan (`0.0`) -> PinS IFR Amber (`0.52`) -> Grounded Smoke Crimson-Ash (`1.0`)
                let rgba = if t < 0.48 {
                    let u = t / 0.48;
                    [
                        (14.0 + 230.0 * u) as u8,
                        (165.0 + 26.0 * u) as u8,
                        (233.0 - 195.0 * u) as u8,
                        (92.0 + 78.0 * u) as u8,
                    ]
                } else if t < 0.80 {
                    let u = (t - 0.48) / 0.32;
                    [
                        (244.0 - 20.0 * u) as u8,
                        (191.0 - 132.0 * u) as u8,
                        (38.0 + 36.0 * u) as u8,
                        192,
                    ]
                } else {
                    let u = (t - 0.80) / 0.20;
                    [
                        (224.0 - 45.0 * u) as u8,
                        (59.0 - 22.0 * u) as u8,
                        (74.0 + 42.0 * u) as u8,
                        215,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::EvacuationContraflow => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let clr_norm = ((p.evac_clearance_min - 18.0) / 115.0).clamp(0.0, 1.0);
                    let stg_urg = match p.evac_stage_tier {
                        1 => 0.92,
                        2 => 0.52,
                        _ => 0.16,
                    };
                    let val01 = (0.65 * clr_norm + 0.35 * stg_urg).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 265.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Evacuation Clearance Palette: Stage-3 Safe Refuge Teal (`0.0`) -> Stage-2 Buffer Gold (`0.50`) -> Stage-1 Urgent Fire-Front Crimson (`1.0`)
                let rgba = if t < 0.46 {
                    let u = t / 0.46;
                    [
                        (20.0 + 225.0 * u) as u8,
                        (184.0 + 14.0 * u) as u8,
                        (166.0 - 130.0 * u) as u8,
                        (94.0 + 76.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.46) / 0.32;
                    [
                        (245.0 - 4.0 * u) as u8,
                        (198.0 - 134.0 * u) as u8,
                        (36.0 + 44.0 * u) as u8,
                        192,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (64.0 + 52.0 * u) as u8,
                        (80.0 + 132.0 * u) as u8,
                        214,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::EmergencyShelters => {
            if world.parcels.is_empty() {
                return;
            }
            // Strain `0.0` = 100% sheltered in clean HEPA air (`<= 12 ug/m^3`);
            // `1.0` = overflowed or smoke-contaminated (`> 55 ug/m^3`) shelter assignment!
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let deficit01 = (1.0 - (p.shelter_clean_air_pct / 100.0)).clamp(0.0, 1.0);
                    let pm_norm = ((p.shelter_indoor_pm25 - 8.0) / 55.0).clamp(0.0, 1.0);
                    let val01 = (0.62 * deficit01 + 0.38 * pm_norm).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 270.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Clean-Air Shelter Palette: HEPA Emerald-Sky (`0.0`) -> Moderate MERV-13 Gold (`0.50`) -> Smoke/Overflow Crimson (`1.0`)
                let rgba = if t < 0.46 {
                    let u = t / 0.46;
                    [
                        (16.0 + 228.0 * u) as u8,
                        (195.0 - 5.0 * u) as u8,
                        (155.0 - 118.0 * u) as u8,
                        (92.0 + 78.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.46) / 0.32;
                    [
                        244,
                        (190.0 - 128.0 * u) as u8,
                        (37.0 + 50.0 * u) as u8,
                        192,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (62.0 + 48.0 * u) as u8,
                        (87.0 + 135.0 * u) as u8,
                        214,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::CommodityResupply => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let insecurity01 = ((100.0 - p.resupply_security_pct) / 35.0).clamp(0.0, 1.0);
                    let dep01 = p.resupply_deprivation_idx.clamp(0.0, 1.0);
                    let val01 = (0.55 * insecurity01 + 0.45 * dep01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 275.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Logistics Sustainment Palette: Stocked Supply Cyan-Emerald (`0.0`) -> Reorder Amber (`0.50`) -> Stockout Deprivation Crimson (`1.0`)
                let rgba = if t < 0.46 {
                    let u = t / 0.46;
                    [
                        (18.0 + 226.0 * u) as u8,
                        (188.0 + 6.0 * u) as u8,
                        (212.0 - 172.0 * u) as u8,
                        (94.0 + 76.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.46) / 0.32;
                    [
                        244,
                        (194.0 - 132.0 * u) as u8,
                        (40.0 + 48.0 * u) as u8,
                        192,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (62.0 + 50.0 * u) as u8,
                        (88.0 + 130.0 * u) as u8,
                        214,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::TransitBusEvacuation => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let unserved01 = ((100.0 - p.transit_evac_cleared_pct) / 55.0).clamp(0.0, 1.0);
                    let wait01 = ((p.transit_evac_wait_min - 8.0) / 75.0).clamp(0.0, 1.0);
                    let val01 = (0.62 * unserved01 + 0.38 * wait01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 250.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Transit Clean-Air Evacuation Palette: Fast Bus+EMS Clearance Mint-Teal (`0.0`) -> Queue Wait Gold (`0.50`) -> Stranded Carless/ADA Rose-Magenta (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (16.0 + 234.0 * u) as u8,
                        (196.0 + 8.0 * u) as u8,
                        (160.0 - 128.0 * u) as u8,
                        (96.0 + 74.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        250,
                        (204.0 - 132.0 * u) as u8,
                        (32.0 + 68.0 * u) as u8,
                        194,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (72.0 - 24.0 * u) as u8,
                        (100.0 + 96.0 * u) as u8,
                        216,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::WarningMobilization => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let unwarned01 =
                        ((100.0 - p.warning_mobilized_60m_pct) / 65.0).clamp(0.0, 1.0);
                    let delay01 = ((p.warning_departure_delay_min - 10.0) / 55.0).clamp(0.0, 1.0);
                    let val01 = (0.58 * unwarned01 + 0.42 * delay01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 255.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Warning & Telecom Blackout Palette: Prompt WEA/Siren Alert Sky-Cyan (`0.0`) -> Delayed Staging Amber (`0.50`) -> Cellular Blackout Isolation Violet-Crimson (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (34.0 + 216.0 * u) as u8,
                        (211.0 - 20.0 * u) as u8,
                        (238.0 - 202.0 * u) as u8,
                        (96.0 + 74.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        250,
                        (191.0 - 135.0 * u) as u8,
                        (36.0 + 110.0 * u) as u8,
                        194,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        (250.0 - 26.0 * u) as u8,
                        (56.0 - 18.0 * u) as u8,
                        (146.0 + 68.0 * u) as u8,
                        218,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::FemaCommsArchitecture => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let ipaws_gap01 =
                        ((100.0 - p.fema_ipaws_reach_pct) / 60.0).clamp(0.0, 1.0);
                    let lmr_gap01 = ((100.0 - p.fema_esf2_lmr_pct) / 65.0).clamp(0.0, 1.0);
                    let val01 = (0.60 * ipaws_gap01 + 0.40 * lmr_gap01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 250.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // FEMA IPAWS & ESF-2 Palette: Resilient IPAWS+MERS Cobalt-Teal (`0.0`) -> Single-Pathway Amber (`0.50`) -> CAP/LMR Shadow Crimson (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (24.0 + 226.0 * u) as u8,
                        (192.0 + 8.0 * u) as u8,
                        (228.0 - 188.0 * u) as u8,
                        (98.0 + 72.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        250,
                        (200.0 - 138.0 * u) as u8,
                        (40.0 + 52.0 * u) as u8,
                        194,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (62.0 - 20.0 * u) as u8,
                        (92.0 + 78.0 * u) as u8,
                        218,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::FemaIsbPods => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let wait01 = ((p.pod_total_access_min - 8.0) / 75.0).clamp(0.0, 1.0);
                    let unserved01 = ((100.0 - p.pod_served_pct) / 45.0).clamp(0.0, 1.0);
                    let val01 = (0.55 * wait01 + 0.45 * unserved01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 250.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // FEMA ISB & C-POD Palette: Fast IPAWS-Balanced POD Access Emerald-Sky (`0.0`) -> Queue Wait Amber (`0.50`) -> Spillback & Dry-Run Stockout Crimson (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (20.0 + 228.0 * u) as u8,
                        (204.0 - 8.0 * u) as u8,
                        (176.0 - 140.0 * u) as u8,
                        (98.0 + 72.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        248,
                        (196.0 - 134.0 * u) as u8,
                        (36.0 + 54.0 * u) as u8,
                        194,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (62.0 - 20.0 * u) as u8,
                        (90.0 + 82.0 * u) as u8,
                        218,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::FemaUsarDebris => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let debris01 = (p.debris_blockage_tons / 220.0).clamp(0.0, 1.0);
                    let loss01 = ((100.0 - p.usar_saved_pct) / 65.0).clamp(0.0, 1.0);
                    let val01 = (0.52 * debris01 + 0.48 * loss01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 245.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // FEMA US&R & Debris Palette: Cleared USNG Sector Teal-Emerald (`0.0`) -> Active Dozer Push Orange-Gold (`0.50`) -> Heavy URM/RC Rubble Entrapment Crimson (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (24.0 + 226.0 * u) as u8,
                        (198.0 - 12.0 * u) as u8,
                        (168.0 - 136.0 * u) as u8,
                        (98.0 + 72.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        250,
                        (186.0 - 128.0 * u) as u8,
                        (32.0 + 48.0 * u) as u8,
                        196,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (58.0 - 22.0 * u) as u8,
                        (80.0 + 76.0 * u) as u8,
                        220,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::FemaGsbDrcTdms => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let deficit01 = ((100.0 - p.fema_recovery_score_pct) / 70.0).clamp(0.0, 1.0);
                    let wait01 = ((p.fema_drc_wait_min - 8.0) / 55.0).clamp(0.0, 1.0);
                    let val01 = (0.60 * deficit01 + 0.40 * wait01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 245.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // FEMA GSB/TDMS/DRC Palette: Energized + Fast MRIC/DRC Intake Cyan-Emerald (`0.0`) -> Queue/Haul Backlog Gold (`0.50`) -> Unpowered / Long DRC Wait Magenta-Rose (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (20.0 + 224.0 * u) as u8,
                        (206.0 - 10.0 * u) as u8,
                        (182.0 - 138.0 * u) as u8,
                        (98.0 + 70.0 * u) as u8,
                    ]
                } else if t < 0.78 {
                    let u = (t - 0.45) / 0.33;
                    [
                        246,
                        (196.0 - 130.0 * u) as u8,
                        (44.0 + 68.0 * u) as u8,
                        196,
                    ]
                } else {
                    let u = (t - 0.78) / 0.22;
                    [
                        244,
                        (66.0 - 24.0 * u) as u8,
                        (112.0 + 66.0 * u) as u8,
                        220,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::WildfireAlertWestResponse => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let burn01 = (p.wildfire_burn_risk_pct / 85.0).clamp(0.0, 1.0);
                    let blind01 = ((100.0 - p.wildfire_camera_vis_pct) / 85.0).clamp(0.0, 1.0);
                    let val01 = (0.82 * burn01 + 0.18 * blind01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 235.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // ALERTWest / OHAZ WUI Wildfire Palette: Clear Viewshed Forest-Teal (`0.0`) -> Ember / Level-2 SET Amber (`0.48`) -> Active Crown Flame / Level-3 GO Scarlet (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (18.0 + 232.0 * u) as u8,
                        (188.0 - 12.0 * u) as u8,
                        (138.0 - 110.0 * u) as u8,
                        (94.0 + 78.0 * u) as u8,
                    ]
                } else if t < 0.76 {
                    let u = (t - 0.45) / 0.31;
                    [
                        251,
                        (176.0 - 112.0 * u) as u8,
                        (28.0 + 18.0 * u) as u8,
                        202,
                    ]
                } else {
                    let u = (t - 0.76) / 0.24;
                    [
                        248,
                        (64.0 - 38.0 * u) as u8,
                        (46.0 + 48.0 * u) as u8,
                        226,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::BurnScarDebrisFlow => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let haz01 = (p.debris_flow_hazard_pct / 85.0).clamp(0.0, 1.0);
                    let roff01 = (p.burn_scar_runoff_mm / 58.0).clamp(0.0, 1.0);
                    let val01 = (0.64 * haz01 + 0.36 * roff01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 240.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // Post-Fire Hydrophobic Runoff & USGS Debris-Flow Palette: Infiltrating Watershed Cyan-Teal (`0.0`) -> Hydrophobic Sheetflow Ochre-Amber (`0.48`) -> Extreme Debris-Flow & Culvert Washout Crimson-Plum (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (22.0 + 216.0 * u) as u8,
                        (182.0 - 20.0 * u) as u8,
                        (196.0 - 156.0 * u) as u8,
                        (96.0 + 76.0 * u) as u8,
                    ]
                } else if t < 0.76 {
                    let u = (t - 0.45) / 0.31;
                    [
                        238,
                        (162.0 - 104.0 * u) as u8,
                        (40.0 + 52.0 * u) as u8,
                        200,
                    ]
                } else {
                    let u = (t - 0.76) / 0.24;
                    [
                        (238.0 - 28.0 * u) as u8,
                        (58.0 - 24.0 * u) as u8,
                        (92.0 + 96.0 * u) as u8,
                        226,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
        HeatmapLayer::WtpTurbidityStorage => {
            if world.parcels.is_empty() {
                return;
            }
            let p_pts: Vec<(f32, f32, f32)> = world
                .parcels
                .iter()
                .map(|p| {
                    let def01 = ((100.0 - p.wtp_water_security_pct) / 80.0).clamp(0.0, 1.0);
                    let bwa01 = ((38.0 - p.wtp_min_pressure_psi).max(0.0) / 26.0).clamp(0.0, 1.0);
                    let val01 = (0.58 * def01 + 0.42 * bwa01).clamp(0.04, 1.0);
                    (p.center.x as f32, p.center.y as f32, val01)
                })
                .collect();

            for gy in 0..HEATMAP_DIM {
                let wy = (b.min_y + (gy as f64 + 0.5) / HEATMAP_DIM as f64 * span_y) as f32;
                for gx in 0..HEATMAP_DIM {
                    let wx = (b.min_x + (gx as f64 + 0.5) / HEATMAP_DIM as f64 * span_x) as f32;
                    let mut w_sum = 0.0f32;
                    let mut r_sum = 0.0f32;
                    for &(px, py, s01) in &p_pts {
                        let d2 = (wx - px) * (wx - px) + (wy - py) * (wy - py) + 240.0;
                        let w = 1.0 / d2;
                        w_sum += w;
                        r_sum += w * s01;
                    }
                    arena.heatmap_field_a[gy * HEATMAP_DIM + gx] =
                        if w_sum > 0.0 { r_sum / w_sum } else { 0.0 };
                }
            }

            separable_blur_5tap(
                &arena.heatmap_field_a,
                &mut arena.heatmap_field_b,
                &mut arena.heatmap_field_c,
            );
            for (i, &val) in arena.heatmap_field_c.iter().enumerate() {
                let t = val.clamp(0.0, 1.0);
                // WTP Finished-Water & Boil-Water Palette: Pressurized Clean Potable Azure-Cyan (`0.0`) -> Tank Drawdown Amber (`0.48`) -> <20 psi / Turbidity Breakthrough BWA Crimson (`1.0`)
                let rgba = if t < 0.45 {
                    let u = t / 0.45;
                    [
                        (16.0 + 228.0 * u) as u8,
                        (192.0 - 14.0 * u) as u8,
                        (236.0 - 188.0 * u) as u8,
                        (96.0 + 76.0 * u) as u8,
                    ]
                } else if t < 0.76 {
                    let u = (t - 0.45) / 0.31;
                    [
                        244,
                        (178.0 - 118.0 * u) as u8,
                        (48.0 + 36.0 * u) as u8,
                        202,
                    ]
                } else {
                    let u = (t - 0.76) / 0.24;
                    [
                        244,
                        (60.0 - 26.0 * u) as u8,
                        (84.0 + 68.0 * u) as u8,
                        226,
                    ]
                };
                arena.heatmap_rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba);
            }
        }
    }
}


/// Deterministic CPU reference implementation of the semi-Lagrangian upwind advection + 5-point
/// isotropic diffusion scheme executed on GPU by `shaders/plume_compute.wgsl` (`cs_advect_diffuse`).
pub fn step_plume_advection_diffusion_cpu(
    src_field: &[f32],
    source_inject: &[f32],
    dst_field: &mut [f32],
    world_span: [f32; 2],
    params: PlumePhysicsParams,
    dt_s: f32,
) {
    let dim = HEATMAP_DIM;
    let dx_m = (world_span[0] / dim as f32).max(0.5);
    let dy_m = (world_span[1] / dim as f32).max(0.5);
    let shift_x = (params.wind_x_mps * dt_s) / dx_m;
    let shift_y = (params.wind_y_mps * dt_s) / dy_m;
    let diff = params.diffusivity.clamp(0.0, 0.95);
    let decay = (-params.decay_rate.max(0.0) * dt_s).exp();

    let sample_bilinear = |fx: f32, fy: f32| -> f32 {
        let cx = fx.clamp(0.0, (dim - 1) as f32);
        let cy = fy.clamp(0.0, (dim - 1) as f32);
        let x0 = cx.floor() as usize;
        let y0 = cy.floor() as usize;
        let x1 = (x0 + 1).min(dim - 1);
        let y1 = (y0 + 1).min(dim - 1);
        let tx = cx - x0 as f32;
        let ty = cy - y0 as f32;
        let c00 = src_field[y0 * dim + x0];
        let c10 = src_field[y0 * dim + x1];
        let c01 = src_field[y1 * dim + x0];
        let c11 = src_field[y1 * dim + x1];
        let top = c00 * (1.0 - tx) + c10 * tx;
        let bot = c01 * (1.0 - tx) + c11 * tx;
        top * (1.0 - ty) + bot * ty
    };

    for y in 0..dim {
        for x in 0..dim {
            let bx = x as f32 - shift_x;
            let by = y as f32 - shift_y;
            let center = sample_bilinear(bx, by);
            let n_l = sample_bilinear(bx - 1.0, by);
            let n_r = sample_bilinear(bx + 1.0, by);
            let n_u = sample_bilinear(bx, by - 1.0);
            let n_d = sample_bilinear(bx, by + 1.0);
            let avg4 = 0.25 * (n_l + n_r + n_u + n_d);
            let diffused = center * (1.0 - diff * 0.55) + avg4 * (diff * 0.55);
            let idx = y * dim + x;
            dst_field[idx] = diffused * decay + source_inject[idx];
        }
    }
}

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

// Custom wgpu WGSL instanced render pipeline for `urban-grid` v1.1.0
// Pass 1 (`vs_heatmap` / `fs_heatmap`): Hardware-bilinear 256x256 GPU heatmap with topographic isolines
// Pass 2 (`vs_road` / `fs_road`):       Instanced metric lanes, BRT lanes, bikeways, sidewalks & signal stop-bars
// Pass 3 (`vs_agent` / `fs_agent`):     50,000+ instanced SDF vehicles (brake lights + blinkers), cyclists & pedestrians

struct CameraUniform {
    cam_center: vec2<f32>,
    viewport_px: vec2<f32>,
    world_min: vec2<f32>,
    world_span: vec2<f32>,
    zoom_px_per_m: f32,
    sim_time_s: f32,
    heatmap_opacity: f32,
    heatmap_mode: u32,
};

@group(0) @binding(0) var<uniform> cam: CameraUniform;
@group(0) @binding(1) var heatmap_tex: texture_2d<f32>;
@group(0) @binding(2) var heatmap_sampler: sampler;

fn world_to_clip(world_xy: vec2<f32>) -> vec4<f32> {
    let half_vp = max(cam.viewport_px * 0.5, vec2<f32>(1.0, 1.0));
    let rel_px = (world_xy - cam.cam_center) * cam.zoom_px_per_m;
    // Note: world +Y points South (downward in egui), whereas WebGPU NDC +Y points UP.
    return vec4<f32>(rel_px.x / half_vp.x, -rel_px.y / half_vp.y, 0.0, 1.0);
}

fn quad_corner(vid: u32) -> vec2<f32> {
    switch vid {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>( 1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>( 1.0,  1.0); }
        default: { return vec2<f32>(-1.0,  1.0); }
    }
}

// ============================================================================
// PASS 1: Bilinear GPU Heatmap Texture Overlay (Zoning, Isochrones, VSP, Density)
// ============================================================================

struct HeatmapOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_heatmap(@builtin(vertex_index) vid: u32) -> HeatmapOut {
    let c = quad_corner(vid);
    let uv = c * 0.5 + vec2<f32>(0.5, 0.5);
    let world_pos = cam.world_min + uv * cam.world_span;
    var out: HeatmapOut;
    out.clip_pos = world_to_clip(world_pos);
    out.uv = uv;
    return out;
}

@fragment
fn fs_heatmap(in: HeatmapOut) -> @location(0) vec4<f32> {
    // Always sample in uniform control flow for strict Vulkan/DX12/Metal Naga compliance
    let raw = textureSample(heatmap_tex, heatmap_sampler, in.uv);
    if (cam.heatmap_mode == 0u || raw.a <= 0.004) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var rgb = raw.rgb;
    var alpha = raw.a * cam.heatmap_opacity;

    // Subtle topographic iso-contour bands on continuous fields (`3u..=4u` & `7u+`), keeping `5u` (baked 60s EMS contours) & `6u` ResponseDistricts crisp!
    if (cam.heatmap_mode == 3u || cam.heatmap_mode == 4u || cam.heatmap_mode >= 7u) {
        let lum = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
        let band = abs(fract(lum * 7.0) - 0.5);
        let contour = 1.0 - smoothstep(0.0, 0.055, band);
        rgb = mix(rgb, rgb * 1.28 + vec3<f32>(0.08, 0.10, 0.12), contour * 0.35);
    }

    return vec4<f32>(rgb, clamp(alpha, 0.0, 0.92));
}

// ============================================================================
// PASS 2: Instanced Oriented Road, Bus/BRT, Protected Bike, Sidewalk & Signal Quads
// ============================================================================

struct RoadInstanceIn {
    @location(0) p0: vec2<f32>,
    @location(1) p1: vec2<f32>,
    @location(2) half_width_m: f32,
    @location(3) style_kind: u32,
    @location(4) color: vec4<f32>,
    @location(5) aux_param: f32,
};

struct RoadVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) style_kind: u32,
    @location(3) seg_len_m: f32,
    @location(4) @interpolate(flat) aux_param: f32,
};

@vertex
fn vs_road(@builtin(vertex_index) vid: u32, inst: RoadInstanceIn) -> RoadVarying {
    let corner = quad_corner(vid);
    let delta = inst.p1 - inst.p0;
    let seg_len = max(length(delta), 0.05);
    let fwd = delta / seg_len;
    let right = vec2<f32>(-fwd.y, fwd.x);

    let min_hw = 0.75 / max(cam.zoom_px_per_m, 0.15);
    let hw = max(inst.half_width_m, min_hw);
    var end_cap = 0.18;
    if (inst.style_kind == 4u || inst.style_kind == 7u) {
        end_cap = 0.38;
    } else if (inst.style_kind == 10u) {
        if (inst.aux_param >= 0.5 && inst.aux_param < 1.5) {
            end_cap = 0.0;
        } else if (inst.aux_param < 0.5) {
            end_cap = 1.4;
        }
    }

    let along_t = corner.x * 0.5 + 0.5;
    let world_pos = mix(inst.p0 - fwd * end_cap, inst.p1 + fwd * end_cap, along_t) + right * (corner.y * hw);

    var out: RoadVarying;
    out.clip_pos = world_to_clip(world_pos);
    out.local_uv = vec2<f32>(along_t * seg_len, corner.y);
    out.color = inst.color;
    out.style_kind = inst.style_kind;
    out.seg_len_m = seg_len;
    out.aux_param = inst.aux_param;
    return out;
}

@fragment
fn fs_road(in: RoadVarying) -> @location(0) vec4<f32> {
    let edge_aa = 1.0 - smoothstep(0.84, 1.0, abs(in.local_uv.y));
    var rgb = in.color.rgb;
    var alpha = in.color.a * edge_aa;

    if (in.style_kind == 0u) {
        // Subtle dashed lane-edge guide when zoomed in
        let edge_line = smoothstep(0.76, 0.90, abs(in.local_uv.y));
        let dash = step(0.45, fract(in.local_uv.x * 0.14));
        let zoom_fade = clamp((cam.zoom_px_per_m - 1.2) * 0.45, 0.0, 1.0);
        rgb = mix(rgb, vec3<f32>(0.36, 0.44, 0.56), edge_line * dash * 0.32 * zoom_fade);
    } else if (in.style_kind == 1u) {
        // Dedicated BRT/Bus lane subtle crimson border
        let border = smoothstep(0.70, 0.92, abs(in.local_uv.y));
        rgb = mix(rgb, vec3<f32>(0.96, 0.36, 0.44), border * 0.38);
    } else if (in.style_kind == 2u) {
        // Protected bikeway crisp white buffer stripe
        let buffer_stripe = smoothstep(0.72, 0.92, abs(in.local_uv.y));
        rgb = mix(rgb, vec3<f32>(0.68, 0.98, 0.84), buffer_stripe * 0.45);
    } else if (in.style_kind == 4u) {
        // Glowing intersection signal stop-bar
        let core = 1.0 - smoothstep(0.0, 0.72, abs(in.local_uv.y));
        rgb = rgb * (1.0 + 0.35 * core);
    } else if (in.style_kind == 5u) {
        // Closed road diagonal hazard chevrons
        let stripe = step(0.5, fract((in.local_uv.x + in.local_uv.y * 1.8) * 0.28));
        rgb = mix(rgb, vec3<f32>(0.95, 0.35, 0.42), stripe * 0.45);
    } else if (in.style_kind == 6u) {
        // Curbside GTFS TransitStop shelter pad with crisp white bay outline
        let inner = (1.0 - smoothstep(0.48, 0.78, abs(in.local_uv.y)))
                  * (1.0 - smoothstep(in.seg_len_m * 0.68, in.seg_len_m * 0.96, abs(in.local_uv.x - in.seg_len_m * 0.5) * 2.0));
        rgb = mix(vec3<f32>(0.96, 0.98, 1.0), rgb, 0.72 + 0.28 * inner);
    } else if (in.style_kind == 7u) {
        // Emergency Vehicle Preemption (EVP) animated green-wave corridor & signal bar
        let wave = 0.55 + 0.45 * sin(in.local_uv.x * 0.38 - cam.sim_time_s * 9.5);
        let core = 1.0 - smoothstep(0.0, 0.85, abs(in.local_uv.y));
        rgb = mix(vec3<f32>(0.06, 0.88, 0.65), vec3<f32>(0.32, 0.96, 1.0), wave) * (1.0 + 0.32 * core);
        alpha = clamp(alpha * (0.72 + 0.28 * wave), 0.0, 0.96);
    } else if (in.style_kind == 8u || in.style_kind == 9u) {
        // Emergency Facility Pad: `8u` Fire Station (crimson bay shield) / `9u` Trauma Hospital (Swiss medical cross)
        let ux = (in.local_uv.x / max(in.seg_len_m * 0.5, 0.5)) - 1.0;
        let uy = in.local_uv.y;
        let box_d = max(abs(ux), abs(uy));
        if (box_d > 0.96) {
            discard;
        }
        let border = smoothstep(0.76, 0.92, box_d);
        if (in.style_kind == 9u) {
            // White medical `+` cross inside Hospital pad
            let arm_h = step(abs(ux), 0.52) * step(abs(uy), 0.19);
            let arm_v = step(abs(ux), 0.19) * step(abs(uy), 0.52);
            let cross_m = clamp(arm_h + arm_v, 0.0, 1.0);
            rgb = mix(rgb, vec3<f32>(0.98, 0.99, 1.0), max(cross_m * 0.95, border * 0.85));
        } else {
            // Fire Station apparatus bay chevrons + gold-white perimeter trim
            let bay_bars = step(abs(ux), 0.56) * step(0.45, fract((uy + 1.0) * 2.2));
            rgb = mix(rgb, vec3<f32>(1.0, 0.92, 0.72), max(bay_bars * 0.82, border * 0.88));
        }
        alpha = 0.96;
    } else if (in.style_kind == 10u) {
        // River / Canal / Harbor Bay water rendering
        let flow = 0.5 + 0.5 * sin(in.local_uv.x * 0.08 - cam.sim_time_s * 1.15);
        if (in.aux_param >= 0.5 && in.aux_param < 1.5) {
            // Bay polygon interior slab: seamless deep marine water
            rgb = mix(rgb, vec3<f32>(0.08, 0.34, 0.58), flow * 0.14);
            alpha = in.color.a;
        } else if (in.aux_param >= 1.5) {
            // Coastal shallow shoreline contour
            let shore = 1.0 - smoothstep(0.35, 0.98, abs(in.local_uv.y));
            rgb = mix(vec3<f32>(0.12, 0.44, 0.68), vec3<f32>(0.24, 0.62, 0.84), shore * 0.55);
            alpha = in.color.a * shore;
        } else {
            // River / Creek / Canal channel with deep center and sunlit shallow bank rim
            let bank = smoothstep(0.62, 0.96, abs(in.local_uv.y));
            let deep = vec3<f32>(0.04, 0.24, 0.48);
            let shallow = vec3<f32>(0.14, 0.48, 0.74);
            rgb = mix(deep, shallow, bank * 0.72 + flow * 0.10);
            alpha = in.color.a * (1.0 - smoothstep(0.88, 1.0, abs(in.local_uv.y)));
        }
    } else if (in.style_kind == 11u) {
        // Bridge deck with concrete/steel side parapets and end abutments
        let parapet = smoothstep(0.76, 0.91, abs(in.local_uv.y));
        let outer_shadow = smoothstep(0.92, 1.0, abs(in.local_uv.y));
        let abutment = 1.0 - smoothstep(0.0, 4.0, min(in.local_uv.x, in.seg_len_m - in.local_uv.x));
        rgb = mix(rgb, vec3<f32>(0.90, 0.94, 0.99), parapet * 0.88);
        rgb = mix(rgb, vec3<f32>(0.68, 0.76, 0.86), abutment * 0.45);
        rgb = mix(rgb, vec3<f32>(0.08, 0.12, 0.20), outer_shadow * 0.75);
        alpha = in.color.a * (1.0 - smoothstep(0.94, 1.0, abs(in.local_uv.y)));
    }

    return vec4<f32>(rgb, alpha);
}

// ============================================================================
// PASS 3: 50,000+ Instanced Multi-Modal SDF Agents (Vehicles, Cyclists, Pedestrians)
// ============================================================================

struct AgentInstanceIn {
    @location(0) pos: vec2<f32>,
    @location(1) heading_rad: f32,
    @location(2) speed_mps: f32,
    @location(3) half_len_m: f32,
    @location(4) half_wid_m: f32,
    @location(5) color: vec4<f32>,
    @location(6) flags: u32,
};

struct AgentVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) flags: u32,
    @location(3) px_radius: f32,
};

fn sd_round_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@vertex
fn vs_agent(@builtin(vertex_index) vid: u32, inst: AgentInstanceIn) -> AgentVarying {
    let corner = quad_corner(vid);
    let kind = inst.flags & 15u;

    // Guarantee crisp visibility even when zoomed out; EmergencyResponder (`kind == 5u`) gets a wider halo quad for siren rings
    let is_motor = (kind <= 2u) || (kind == 5u);
    let min_px = select(select(1.45, 1.95, is_motor), 3.20, kind == 5u);
    let min_world_half = min_px / max(cam.zoom_px_per_m, 0.08);

    let halo_scale = select(1.22, 2.75, kind == 5u);
    let hl = max(inst.half_len_m, min_world_half) * halo_scale;
    let hw = max(inst.half_wid_m, min_world_half * 0.85) * halo_scale;

    let c = cos(inst.heading_rad);
    let s = sin(inst.heading_rad);
    let fwd = vec2<f32>(c, s);
    let right = vec2<f32>(-s, c);

    let world_pos = inst.pos + fwd * (corner.x * hl) + right * (corner.y * hw);

    var out: AgentVarying;
    out.clip_pos = world_to_clip(world_pos);
    out.uv = corner * halo_scale;
    out.color = inst.color;
    out.flags = inst.flags;
    out.px_radius = hw * cam.zoom_px_per_m;
    return out;
}

@fragment
fn fs_agent(in: AgentVarying) -> @location(0) vec4<f32> {
    let kind = in.flags & 15u;
    let detail_blend = clamp((in.px_radius - 2.2) / 3.0, 0.0, 1.0);

    // Pedestrian (`kind == 4u`): anti-aliased circular crowd agent with shoulder highlight
    if (kind == 4u) {
        let d = length(in.uv) - 0.92;
        let aa = max(fwidth(d), 0.08);
        let mask = 1.0 - smoothstep(-aa, aa, d);
        if (mask <= 0.01) {
            discard;
        }
        let core = 1.0 - smoothstep(0.0, 0.65, length(in.uv));
        let rgb = mix(in.color.rgb, vec3<f32>(0.94, 0.98, 1.0), core * 0.35 * detail_blend);
        return vec4<f32>(rgb, mask * in.color.a);
    }

    // Cyclist (`kind == 3u`): oriented diamond/capsule frame + rider helmet dot
    if (kind == 3u) {
        let d_diam = (abs(in.uv.x) * 0.82 + abs(in.uv.y) * 1.12) - 0.88;
        let aa = max(fwidth(d_diam), 0.08);
        let mask = 1.0 - smoothstep(-aa, aa, d_diam);
        if (mask <= 0.01) {
            discard;
        }
        let helmet = 1.0 - smoothstep(0.18, 0.36, length(in.uv - vec2<f32>(0.12, 0.0)));
        let rgb = mix(in.color.rgb, vec3<f32>(0.92, 1.0, 0.96), helmet * 0.65 * detail_blend);
        return vec4<f32>(rgb, mask);
    }

    // Motor Vehicles (`kind == 0u` Sedan, `1u` TransitBus, `2u` FreightTruck)
    let corner_r = select(0.28, 0.16, kind == 1u || kind == 2u);
    let d_body = sd_round_box(in.uv, vec2<f32>(0.98, 0.92), corner_r);
    let aa = max(fwidth(d_body), 0.06);
    let body_mask = 1.0 - smoothstep(-aa, aa, d_body);

    var rgb = in.color.rgb;
    var out_alpha = body_mask * in.color.a;

    // Street-level chassis details (windshield, bus roof stripe, headlights)
    if (detail_blend > 0.02 && body_mask > 0.05) {
        let ws = (1.0 - smoothstep(0.0, 0.12, abs(in.uv.x - 0.36) - 0.18))
               * (1.0 - smoothstep(0.0, 0.10, abs(in.uv.y) - 0.68));
        rgb = mix(rgb, vec3<f32>(0.07, 0.11, 0.18), ws * 0.72 * detail_blend);

        if (kind == 1u) {
            // Live GTFS passenger crowding meter along the TransitBus roof (`bits 16..23`)
            let load_frac = f32((in.flags >> 16u) & 255u) / 255.0;
            let bar_box = (1.0 - smoothstep(0.0, 0.09, abs(in.uv.y) - 0.24))
                        * (1.0 - smoothstep(0.0, 0.09, abs(in.uv.x + 0.08) - 0.64));
            let fill_x = clamp((in.uv.x + 0.72) / 1.28, 0.0, 1.0);
            let filled = step(fill_x, max(load_frac, 0.08));
            var load_col = vec3<f32>(0.12, 0.90, 0.58);
            if (load_frac > 0.82) {
                load_col = vec3<f32>(1.0, 0.20, 0.32);
            } else if (load_frac > 0.58) {
                load_col = vec3<f32>(0.99, 0.76, 0.12);
            }
            let roof_rgb = mix(vec3<f32>(0.12, 0.16, 0.24), load_col, filled);
            rgb = mix(rgb, roof_rgb, bar_box * 0.88 * detail_blend);
        }
    }

    // Dual glowing red brake lamps at rear bumper (`in.uv.x ~ -0.96`) when decelerating or stopped
    if ((in.flags & 16u) != 0u) {
        let d_bl = min(
            length(in.uv - vec2<f32>(-0.96, -0.56)),
            length(in.uv - vec2<f32>(-0.96,  0.56))
        );
        let brake_glow = 1.0 - smoothstep(0.06, 0.34, d_bl);
        rgb = mix(rgb, vec3<f32>(1.0, 0.12, 0.18), brake_glow * 0.92);
        out_alpha = max(out_alpha, brake_glow * 0.88);
    }

    // Time-animated 2 Hz amber turn-signal blinkers at front corners (`in.uv.x ~ +0.94`)
    let phase_seed = f32((in.flags >> 8u) & 255u) / 255.0;
    let blink_on = step(0.42, fract(cam.sim_time_s * 2.1 + phase_seed));
    if (blink_on > 0.5) {
        if ((in.flags & 32u) != 0u) {
            // Left blinker (`y < 0` in right-handed `(fwd, right)` vehicle frame)
            let d_sig = length(in.uv - vec2<f32>(0.92, -0.66));
            let glow = 1.0 - smoothstep(0.05, 0.34, d_sig);
            rgb = mix(rgb, vec3<f32>(1.0, 0.76, 0.08), glow * 0.95);
            out_alpha = max(out_alpha, glow * 0.90);
        }
        if ((in.flags & 64u) != 0u) {
            // Right blinker (`y > 0`)
            let d_sig = length(in.uv - vec2<f32>(0.92, 0.66));
            let glow = 1.0 - smoothstep(0.05, 0.34, d_sig);
            rgb = mix(rgb, vec3<f32>(1.0, 0.76, 0.08), glow * 0.95);
            out_alpha = max(out_alpha, glow * 0.90);
        }
    }

    // Curbside GTFS boarding/alighting 4 Hz amber door flashers (`AGENT_FLAG_DWELLING == 128u`)
    if ((in.flags & 128u) != 0u) {
        let dwell_pulse = 0.35 + 0.65 * step(0.45, fract(cam.sim_time_s * 3.8 + phase_seed));
        let d_door = min(
            length(in.uv - vec2<f32>( 0.42, 0.88)),
            length(in.uv - vec2<f32>(-0.28, 0.88))
        );
        let door_glow = (1.0 - smoothstep(0.05, 0.36, d_door)) * dwell_pulse;
        rgb = mix(rgb, vec3<f32>(0.22, 0.92, 1.0), door_glow * 0.95);
        out_alpha = max(out_alpha, door_glow * 0.92);
    }

    // EmergencyResponder (`kind == 5u`): crimson-and-white livery, 7.5 Hz alternating Red/Blue
    // dual roof strobe lightbars, and expanding concentric acoustic siren wavefront rings!
    if (kind == 5u) {
        if (body_mask > 0.05) {
            let hood_stripe = 1.0 - smoothstep(0.0, 0.12, abs(in.uv.x + 0.15) - 0.42);
            rgb = mix(vec3<f32>(0.97, 0.98, 1.0), vec3<f32>(0.92, 0.14, 0.22), hood_stripe * 0.65);
        }

        let strobe_phase = fract(cam.sim_time_s * 7.5 + phase_seed);
        let red_active = step(0.5, strobe_phase);
        let blue_active = 1.0 - red_active;

        let d_left_bar = length((in.uv - vec2<f32>(0.18, -0.48)) * vec2<f32>(1.1, 0.85));
        let d_right_bar = length((in.uv - vec2<f32>(0.18, 0.48)) * vec2<f32>(1.1, 0.85));

        let red_glow = (1.0 - smoothstep(0.04, 0.95, d_left_bar)) * (0.25 + 0.75 * red_active);
        let blue_glow = (1.0 - smoothstep(0.04, 0.95, d_right_bar)) * (0.25 + 0.75 * blue_active);

        rgb = mix(rgb, vec3<f32>(1.0, 0.10, 0.20), clamp(red_glow * 1.15, 0.0, 1.0));
        rgb = mix(rgb, vec3<f32>(0.12, 0.56, 1.0), clamp(blue_glow * 1.15, 0.0, 1.0));
        out_alpha = max(out_alpha, max(red_glow, blue_glow) * 0.92);

        // Concentric expanding acoustic siren wavefront rings in outer UV [0.95 .. 2.70]
        let r_ellip = length(in.uv * vec2<f32>(0.78, 0.92));
        if (r_ellip > 0.85 && r_ellip < 2.68) {
            let wave_t = fract(r_ellip * 1.15 - cam.sim_time_s * 2.35);
            let ring = (1.0 - smoothstep(0.0, 0.13, abs(wave_t - 0.5)))
                     * (1.0 - smoothstep(1.05, 2.65, r_ellip));
            let ring_col = mix(vec3<f32>(0.22, 0.88, 1.0), vec3<f32>(1.0, 0.26, 0.38), red_active);
            rgb = mix(rgb, ring_col, ring * 0.78);
            out_alpha = max(out_alpha, ring * 0.58);
        }
    }

    if (out_alpha <= 0.01) {
        discard;
    }
    return vec4<f32>(rgb, clamp(out_alpha, 0.0, 1.0));
}

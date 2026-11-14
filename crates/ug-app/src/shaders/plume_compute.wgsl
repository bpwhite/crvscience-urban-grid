// Custom wgpu @compute WGSL pipeline for 256x256 Semi-Lagrangian Advection-Diffusion Plumes
// Kernel 1 (`cs_clear_accum`):    Clears 65,536 fixed-point atomic accumulation cells
// Kernel 2 (`cs_splat_agents`):   Atomically splats 50,000+ agents' VSP emissions or crowd weights
// Kernel 3 (`cs_advect_diffuse`): Semi-Lagrangian upwind wind back-trace + 9-tap eddy diffusion +
//                                 exponential decay + thermal/plasma `textureStore` into Rgba8Unorm

const GRID_DIM: u32 = 256u;
const GRID_CELLS: u32 = 65536u;

struct PlumeUniform {
    world_min: vec2<f32>,
    world_span: vec2<f32>,
    wind_mps: vec2<f32>,
    dt_s: f32,
    diffusivity: f32,
    decay_rate: f32,
    source_gain: f32,
    agent_count: u32,
    mode_and_reset: u32,
};

// Matches `ug_core::GpuAgentInstance` (32 bytes, zero padding)
struct StoredAgent {
    pos: vec2<f32>,
    heading_rad: f32,
    speed_mps: f32,
    half_len_m: f32,
    half_wid_m: f32,
    rgba_packed: u32,
    flags: u32,
};

@group(0) @binding(0) var<uniform> params: PlumeUniform;
@group(0) @binding(1) var<storage, read> agents: array<StoredAgent>;
@group(0) @binding(2) var<storage, read_write> accum_atomic: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> plume_src: array<f32>;
@group(0) @binding(4) var<storage, read_write> plume_dst: array<f32>;
@group(0) @binding(5) var out_heatmap_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(256, 1, 1)
fn cs_clear_accum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < GRID_CELLS) {
        atomicStore(&accum_atomic[idx], 0u);
    }
}

@compute @workgroup_size(64, 1, 1)
fn cs_splat_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ai = gid.x;
    if (ai >= params.agent_count) {
        return;
    }
    let ag = agents[ai];
    let mode = params.mode_and_reset & 255u;
    let kind = ag.flags & 15u;

    // Mode 3 (`VspEmissions`) splats motor vehicles only (`kind <= 2u`);
    // Mode 4 (`AgentDensity`) splats all vehicles, cyclists, and sidewalk pedestrians.
    if (mode == 3u && (kind == 3u || kind == 4u)) {
        return;
    }

    let uv = (ag.pos - params.world_min) / max(params.world_span, vec2<f32>(10.0, 10.0));
    if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
        return;
    }

    let gx = min(u32(uv.x * f32(GRID_DIM)), GRID_DIM - 1u);
    let gy = min(u32(uv.y * f32(GRID_DIM)), GRID_DIM - 1u);
    let cell_idx = gy * GRID_DIM + gx;

    var weight_fp: u32 = 0u;
    if (mode == 3u) {
        let vsp_u8 = (ag.flags >> 24u) & 255u;
        weight_fp = max(vsp_u8, 4u) * 16u;
    } else {
        weight_fp = select(650u, 1000u, kind <= 2u || kind == 5u);
    }

    atomicAdd(&accum_atomic[cell_idx], weight_fp);
}

fn sample_src_clamped(ix: i32, iy: i32) -> f32 {
    let cx = u32(clamp(ix, 0, i32(GRID_DIM) - 1));
    let cy = u32(clamp(iy, 0, i32(GRID_DIM) - 1));
    return plume_src[cy * GRID_DIM + cx];
}

fn sample_src_bilinear(fx: f32, fy: f32) -> f32 {
    let cx = clamp(fx, 0.0, f32(GRID_DIM - 1u));
    let cy = clamp(fy, 0.0, f32(GRID_DIM - 1u));
    let x0 = i32(floor(cx));
    let y0 = i32(floor(cy));
    let tx = cx - f32(x0);
    let ty = cy - f32(y0);

    let c00 = sample_src_clamped(x0, y0);
    let c10 = sample_src_clamped(x0 + 1, y0);
    let c01 = sample_src_clamped(x0, y0 + 1);
    let c11 = sample_src_clamped(x0 + 1, y0 + 1);

    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

fn thermal_vsp_ramp(t_in: f32) -> vec4<f32> {
    let x = clamp(t_in, 0.0, 1.0);
    if (x < 0.015) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    var rgb: vec3<f32>;
    if (x < 0.33) {
        let u = x / 0.33;
        rgb = vec3<f32>(0.118 + 0.490 * u, 0.071 + 0.110 * u, 0.294 + 0.451 * u);
    } else if (x < 0.68) {
        let u = (x - 0.33) / 0.35;
        rgb = vec3<f32>(0.608 + 0.349 * u, 0.181 + 0.282 * u, 0.745 - 0.431 * u);
    } else {
        let u = (x - 0.68) / 0.32;
        rgb = vec3<f32>(0.957 + 0.043 * u, 0.463 + 0.478 * u, 0.314 + 0.372 * u);
    }
    let alpha = clamp(sqrt(x) * 0.86, 0.0, 0.90);
    return vec4<f32>(rgb, alpha);
}

fn plasma_density_ramp(t_in: f32) -> vec4<f32> {
    let x = clamp(t_in, 0.0, 1.0);
    if (x < 0.012) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    var rgb: vec3<f32>;
    if (x < 0.35) {
        let u = x / 0.35;
        rgb = vec3<f32>(0.055 + 0.110 * u, 0.455 + 0.282 * u, 0.565 + 0.384 * u);
    } else if (x < 0.70) {
        let u = (x - 0.35) / 0.35;
        rgb = vec3<f32>(0.165 + 0.815 * u, 0.737 + 0.063 * u, 0.949 - 0.804 * u);
    } else {
        let u = (x - 0.70) / 0.30;
        rgb = vec3<f32>(0.980, 0.800 - 0.549 * u, 0.145 + 0.224 * u);
    }
    let alpha = clamp(sqrt(x) * 0.88, 0.0, 0.92);
    return vec4<f32>(rgb, alpha);
}

@compute @workgroup_size(8, 8, 1)
fn cs_advect_diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= GRID_DIM || gid.y >= GRID_DIM) {
        return;
    }
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let cell_idx = gid.y * GRID_DIM + gid.x;

    let mode = params.mode_and_reset & 255u;
    let reset_history = (params.mode_and_reset & 256u) != 0u;

    // 3x3 spatial kernel over atomic source accumulation so sub-cell streets form smooth plumes immediately
    var inject_sum: f32 = 0.0;
    for (var ky: i32 = -1; ky <= 1; ky = ky + 1) {
        for (var kx: i32 = -1; kx <= 1; kx = kx + 1) {
            let sx = u32(clamp(ix + kx, 0, i32(GRID_DIM) - 1));
            let sy = u32(clamp(iy + ky, 0, i32(GRID_DIM) - 1));
            let raw_u = atomicLoad(&accum_atomic[sy * GRID_DIM + sx]);
            let kw = select(select(0.0625, 0.125, kx == 0 || ky == 0), 0.25, kx == 0 && ky == 0);
            inject_sum = inject_sum + (f32(raw_u) * 0.001) * kw;
        }
    }

    let dt = clamp(params.dt_s, 0.001, 0.08);
    var next_val: f32 = 0.0;

    if (reset_history) {
        next_val = inject_sum * params.source_gain * 0.45;
    } else {
        let cell_m = max(params.world_span / f32(GRID_DIM), vec2<f32>(0.5, 0.5));
        // Semi-Lagrangian upwind departure point back-trace (`x_dep = x - u * dt`)
        let wind_scale = select(0.35, 1.0, mode == 3u);
        let dep_x = f32(ix) - (params.wind_mps.x * wind_scale * dt) / cell_m.x;
        let dep_y = f32(iy) - (params.wind_mps.y * wind_scale * dt) / cell_m.y;

        let c_mid = sample_src_bilinear(dep_x, dep_y);
        let c_l = sample_src_bilinear(dep_x - 1.0, dep_y);
        let c_r = sample_src_bilinear(dep_x + 1.0, dep_y);
        let c_u = sample_src_bilinear(dep_x, dep_y - 1.0);
        let c_d = sample_src_bilinear(dep_x, dep_y + 1.0);
        let lap_avg = 0.25 * (c_l + c_r + c_u + c_d);

        let diff_w = clamp(params.diffusivity, 0.02, 0.95) * 0.58;
        let advected = mix(c_mid, lap_avg, diff_w);
        let decay = exp(-max(params.decay_rate, 0.05) * dt);

        next_val = clamp(advected * decay + inject_sum * params.source_gain * dt, 0.0, 12.0);
    }

    plume_dst[cell_idx] = next_val;

    let norm_val = clamp(next_val, 0.0, 1.0);
    let rgba = select(plasma_density_ramp(norm_val), thermal_vsp_ramp(norm_val), mode == 3u);
    textureStore(out_heatmap_tex, vec2<i32>(ix, iy), rgba);
}

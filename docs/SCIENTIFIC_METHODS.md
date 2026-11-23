# Urban-Grid (`v5.2.0`) Scientific Methods & Verification

## 1. Microscopic Traffic, Emergency Yielding & Multi-Siren Preemption (`traffic.rs`, `emergency.rs`)

- **IDM + MOBIL Kinematics**: Vehicles evolve via Treiber et al. (2000) Intelligent Driver Model (`IDM`) longitudinal acceleration (`max_accel = 1.6 m/s^2`, `hard_decel = 8.5 m/s^2`) and Kesting et al. (2007) `MOBIL` discrete lane selection with curbside GTFS bus dwell holds.
- **Multi-Siren Segment Tracking (`Vec<Vec<f64>>`)**: `step_emergency_preemption` preserves every active responder `s_m` coordinate per road segment (`siren_seg_s[sid].push(...)`) rather than keeping a single overwritten scalar, ensuring multiple responders on long arterial segments independently alert civilian vehicles ahead (`-10..=siren_yield_radius_m`).
- **Physically Bounded Yielding Deceleration**: Yielding civilian vehicles pulling onto the curb shoulder (`+1.28 m`) smoothly ramp speed downward toward `4.2 m/s` capped per step by physical IDM braking (`prev_v - idm.hard_decel_mps2 * dt`), preventing unphysical instantaneous velocity clamps (`<= hard_decel_mps2`).

## 2. Active-Graph 15-Minute Multimodal Accessibility & Acoustic Noise (`environment.rs`)

- **Network-Routed 15-Minute City Isochrones**: `evaluate_environment_and_isochrones` constructs an open-street multimodal segment graph (`!seg.closed`) across city junctions weighted by bike/bus corridor speeds (`eff_m_per_min`), solving single-source Dijkstra from the civic anchor `anchor_pos` to every parcel's closest local street. When all river bridges close (`seg.closed = true`), across-river parcels become graph-unreachable (`999.0 min`) rather than falsely reachable via straight-line Euclidean shortcuts.
- **Density- & Speed-Scaled Traffic Noise (`dBA`)**: Evaluates network lane density (`veh / total_open_lane_km`), fleet heavy-vehicle acoustic multipliers (`4.5x` buses/trucks, `3.5x` responders), and tire-pavement rolling power `(v_kph / 35.0)^2.4` combined with tree canopy attenuation (`-2.5 * avg_canopy`), clamped to `[38.0, 92.0]` dBA.

## 3. GPU Scalar Advection-Diffusion & Rendering Fidelity (`plume_compute.wgsl`, `urban_viewport.wgsl`)

- **Physical 2D Stencil Diffusion & Wind Separation**:
  - `cs_advect_diffuse` derives the 5-point explicit stencil mixing weight physically from grid spacing `cell_m` and time step `dt`: `diff_w = clamp(4.0 * K_eddy * dt / dx^2, 0.0, 0.94)`.
  - Mode 3 (`VspEmissions`) advects downwind with `params.wind_mps`; Mode 4 (`AgentDensity`) zeroes wind drift (`wind_scale = 0.0`) so pedestrian/vehicle crowds diffuse isotropically without blowing away.
  - When `self.paused == true`, GPU plume compute dispatch is skipped so frozen plumes do not diffuse or decay at pause.
- **Raw Scalar Iso-Contours & Decoupled Chassis LOD**:
  - `urban_viewport.wgsl` samples the true float32 scalar field (`@binding(3) scalar_field: array<f32>`) bilinearly (`plume_state_a/b` for GPU plumes; `cpu_scalar_buffer` for CPU parcel/hazard fields) so iso-contours trace concentration/severity level sets independently of colormap RGB luminance.
  - `vs_agent` computes `out.px_radius = body_hw * cam.zoom_px_per_m` using the physical vehicle body half-width decoupled from `halo_scale`, preventing oversized siren quads from triggering premature LOD chassis details.

## 4. Headless Benchmark Telemetry (`validation/urban_grid_validation_summary.json`)

- **T1 (`IDM`/`MOBIL` Fundamental Diagram)**: Across `240 -> 2,100 veh/h/ln` on `32.56` open lane-km (`Riverfront Bridges Metro`), throughput climbs from `1,890` to `25,830 veh/hr` with `0` collisions (`global min bumper headway = 1.929 m`).
- **T2 (`NFPA 1710` `MCLP` & `MEXCLP`/Hypercube Siting)**: `MCLP` achieves `100.0%` 4-minute parcel & resident coverage (`p90 = 58.16 s`), while 4-station `MEXCLP` + Larson (1974) hypercube lowers `p90` to `33.9 s` (`81.95%` queueing-adjusted expected coverage).
- **T3 (`Cascadia M9.0` Lifelines & Evacuation)**: Post-quake bridge closures & `PGV` breaks evaluate restoration across `Day 1 -> 90` paired with EMS-protected contraflow clearance (`cargo test --workspace`).

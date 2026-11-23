# Urban-Grid (`v5.2.0`) User Guide

Welcome to **Urban-Grid (`v5.2.0`)**, a standalone, pure-Rust urban planning, fixed-route GTFS transit, 50k+ agent microscopic traffic, and multi-lifeline hazard/resilience workbench. This guide complements [`urban-grid-v5.2-user-guide.pdf`](urban-grid-v5.2-user-guide.pdf) with interactive controls, hotkeys, GPU heatmap legends, and CLI usage.

## 1. Execution Modes

| Mode | Command | Purpose |
| :--- | :--- | :--- |
| **Desktop Workbench (`egui` + `wgpu`)** | `cargo run --release` (or `-- gui`) | Opens onto the 5-domain Main Entrance Hub (`paused` at `t = 0.0 s`) with 3-pass WGSL rendering & GPU `@compute` advection-diffusion plumes |
| **Terminal Studio (`Ratatui`)** | `cargo run --release -- tui` | Interactive pure-terminal urban simulation monitor and dispatcher |
| **Headless CLI** | `cargo run --release -- <subcommand> [FLAGS]` | Parallel `batch` sweeps, single `simulate` runs, `compare` A/B runs, `mclp`, `stress-test`, `seismic-mc`, `hypercube`, `redistrict`, `diurnal`, `utilities`, `ffe`, `surge`, `medevac`, `evacuate`, `shelters`, `resupply`, `transit-evac`, `warning`, `fema-*`, `wildfire`, `debris-flow`, `wtp-turbidity`, `import` |
| **Offscreen Screenshot** | `cargo run --release -- gui --screenshot out.png` | Renders the live `wgpu` workbench viewport directly to PNG |

## 2. Keyboard Shortcuts & Viewport Controls

| Control / Shortcut | Scope | Action |
| :--- | :--- | :--- |
| **`Space`** | Workbench | Play / Pause microscopic physics & GPU plume compute advection |
| **`[`** / **`]`** | Workbench | Halve / Double simulation speed scale (`0.25x .. 16.0x`) |
| **`R`** or **`0`** | Viewport | Fit camera center & zoom cleanly to active road network bounds |
| **`1` .. `7`** | CAD Tools | Select `1: Inspect`, `2: Draw Road`, `3: Edit Lanes (+/-)`, `4: Convert Roundabout`, `5: Paint Zoning`, `6: Place Emergency Facility`, `7: Bridge/Road Closure` |
| **`E`** | Dispatch | Spawn Emergency Responder with multi-hop corridor green-wave preemption & acoustic siren yielding |
| **`Ctrl + Z`** / **`Ctrl + Y`** | Undo/Redo | 64-step history across road drawing, lanes, roundabouts, zoning, closures & redistricting |
| **`Ctrl + K`** | Navigation | Fuzzy Quick-Jump Command Palette across all modules, scenarios, and layers |
| **`F1`** / **`Home`** / **`Escape`** | Navigation | Toggle Main Entrance Hub / Dismiss active About modal or Command Palette |
| **Left-Drag / Wheel** | Viewport | Pan map (or paint zoning parcels when Tool `5` active) / Zoom (`0.18 .. 10.5 px/m`) |

## 3. GPU Heatmap Overlays, Legends & HUD Indicators

Every active `HeatmapLayer` (`layer != HeatmapLayer::Off`) renders a **contextual bottom-left legend & unit colorbar**:
- **Continuous Layers**: Scaled perceptual ramps with physical units (`kW/t` for `VspEmissions`, `ag/cell` for `AgentDensity`, `0..360 s` with `240 s` NFPA 1710 marker for `EmergencyResponse4Min`, `p.u.`/`psi`/`d/D` for `UtilityLifelines`, `%` burn/survival/clearance/reach, `min` surge/POD access, `HW/D` debris flow, `NTU` WTP turbidity).
- **Categorical Parcel & District Layers**: Discrete swatch keys for `Zoning` (`R1`, `R3`, `MU4`, `CBD`, `IND`, `PARK`), `Isochrone15Min` (`<=5m`, `<=10m`, `<=15m`, `>15m`), and `ResponseDistricts` (`U1..U5` + seam boundary).
- **Dynamic Scale Bar & Wind Rose**: Bottom-right HUD renders an adaptive metric scale bar (`10 m .. 1000 m`) coupled to `cam_zoom` alongside a compass wind-rose vector arrow showing prevailing wind direction and speed (`m/s`).
- **Zero-Clone Buffer Pool**: Road quads and agent instance buffers swap lock-free through a pre-allocated double-buffered pool (`gpu_recycle_pool`), avoiding per-frame deep `Vec` clones.

## 4. Essential CLI Flags

- `--scenario <NAME>`: Built-in morphology (`riverfront-bridges`, `harbor-peninsula`, `radial-ring`, `foothill-canyon-wui`, `metropolis50k`)
- `--gtfs <DIR>` / `--osm <FILE>`: Load GTFS feed directory or custom `.osm` XML/PBF map
- `--max-pressure` / `--tod-upzone`: Activate Varaiya adaptive signal control or TOD upzoning
- `--seconds <F64>` / `--demand-scales <CSV>` / `--replicates <N>` / `--threads <N>`: Headless simulation horizon & parallel CRN sweep settings
- `--csv-out <PATH>` / `--save-snapshot <PATH>`: Export RFC 4180 telemetry CSVs and lossless `.json` scenario snapshots

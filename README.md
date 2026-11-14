# `urban-grid` v4.3.0 — Standalone Microscopic Traffic, GTFS Transit, 50k+ `wgpu` Crowd & Urban Planning Suite

`urban-grid` is a self-contained, offline-first Rust urban planning, fixed-route public transit, and microscopic multi-modal traffic simulator in an independent Cargo workspace. It compiles real-world OpenStreetMap (`.osm` XML, `.osm.pbf`, and live Overpass API bounding boxes) and [GTFS Schedule](https://gtfs.org/documentation/schedule/reference/) feeds into 3.5 m metric lane geometry, coupling parcel land-use zoning, doubly-constrained gravity trip distribution, multinomial logit mode choice, scheduled bus routes with curbside dwell & ridership crowding, microscopic `IDM` + `MOBIL` + sidewalk/crosswalk crowd dynamics, environmental emissions/hydrology, municipal fiscal solvency, and a **custom 3-pass [`wgpu`](https://github.com/gfx-rs/wgpu) WGSL instanced viewport renderer** sustaining **50,000+ simultaneous agents plus GPU heatmap textures at 120+ FPS**.

## Workspace Architecture

- **`crates/ug-core`** — Pure-Rust deterministic simulation, transit, urban systems, and GPU buffer-packing library:
  - `osm.rs` — Streaming [OpenStreetMap](https://wiki.openstreetmap.org/wiki/Key:highway) `.osm` v0.6 XML parser ([`quick-xml`](https://github.com/tafia/quick-xml)), optional `.osm.pbf` binary reader ([`osmpbf`](https://github.com/b-r-u/osmpbf)), optional [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) client, equirectangular WGS84-to-meter projection, and four built-in `.osm` networks (including Downtown [Corvallis, Oregon](https://en.wikivoyage.org/wiki/Corvallis_(Oregon)) couplets anchored on the [Benton County Courthouse](https://en.wikipedia.org/wiki/National_Register_of_Historic_Places_listings_in_Benton_County,_Oregon) and a 144-intersection `Megacity50k` supergrid).
  - `network.rs` — Node-degree topological splitter, 3.5 m lane polyline offsetter with `O(log K)` binary-searched cumulative arc-length tables, Complete Streets (`has_bus_lane` + `has_bike_lane`), cubic Bezier intersection turn splines, [OSM `type=restriction`](https://wiki.openstreetmap.org/wiki/Relation:restriction) enforcement, 2-phase/4-phase/roundabout signal controllers, and reverse-Dijkstra dynamic routing.
  - `zoning.rs` — Parcel land-use zoning (`R1`, `R3`, `MU4`, `CBD`, `IND`, `PARK`) and [Wilson (1967)](https://www.semanticscholar.org/paper/A-statistical-theory-of-spatial-distribution-models-Wilson/2f8308996a9bf467ad7548fcdd357504938c2ad6) doubly-constrained Iterative Proportional Fitting (`IPF`) Gravity Origin-Destination solver.
  - `network_edit.rs` — Live agent-preserving CAD street editor supporting interactive two-click road drawing (with junction snapping and automatic mid-block T-intersection splitting), [NACTO](https://nacto.org/publication/urban-street-design-guide/street-design-elements/lane-width/) lane widening and road-diet reallocation (`1..=5` lanes/dir), and [FHWA](https://highways.dot.gov/safety/data-analysis-tools/rsdp/rsdp-tools/roundabouts-informational-guide-second-edition) multi-lane modern roundabout conversion.
  - `history.rs` — Bounded 64-step Command/Memento undo-redo stack (`EditHistoryStack`, `EditStateSnapshot`) supporting `Ctrl+Z` / `Ctrl+Y` (`Ctrl+Shift+Z`) via [`egui::KeyboardShortcut`](https://docs.rs/egui/latest/egui/struct.KeyboardShortcut.html) and [`egui::InputState::consume_shortcut`](https://docs.rs/egui/latest/egui/struct.InputState.html) across road drawing, lane reallocation, roundabout conversions, parcel zoning, and signal policies without interrupting live microscopic traffic.
  - `emergency.rs` — Multi-intersection corridor green-wave Emergency Vehicle Preemption (`EVP`) grounded in the [FHWA Traffic Signal Timing Manual Chapter 9](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter9.htm) and [Chapter 5 Queue Clearance (`Gq = 3 + 2n`)](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter5.htm), paired with acoustic-siren `MOBIL` right-lane pull-over biases, 1-lane shoulder yielding, and 7.5 Hz red/blue strobe + concentric siren-ring WGSL shaders, plus interactive **Fire Station & Trauma Hospital placement** (`EmergencyFacility`) with live multi-source Dijkstra [NFPA 1710 4-minute (240 s) response-time coverage heatmaps](https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf).
  - `mclp.rs` — `Rayon`-parallelized [Church & ReVelle (1974) Maximal Covering Location Problem (`MCLP`)](https://people.geog.ucsb.edu/~forest/G294download/MAX_COVER_RLC_CSR.pdf) + 1-swap Greedy-Adding-with-Substitution (`GAS`) fire-station siting optimizer finding the minimum station placements required to achieve `>= 90%` [NFPA 1710 4-minute (`240 s`) coverage](https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf).
  - `max_pressure.rs` — Decentralized [Varaiya (2013)](https://www.semanticscholar.org/paper/The-Max-Pressure-Controller-for-Arbitrary-Networks-Varaiya/abafcf4d5f1231c05d385dd36ed7fca420da293f) Max-Pressure adaptive signal controller (`ControlMode::MaxPressureAdaptive`) with [Gregoire et al. (2015)](https://www.researchgate.net/publication/257028475_Capacity-Aware_Backpressure_Traffic_Signal_Control) capacity-aware downstream spillback gating, GTFS passenger-weighted transit pressure, switching-loss hysteresis ([arXiv:2610.32280](https://arxiv.org/html/2610.32280v1)), and safe Yellow/All-Red clearance intervals.
  - `multimodal.rs` — [McFadden (1974)](https://escholarship.org/uc/item/61s3q2xr) Multinomial Logit (`MNL`) mode split across Auto, Bus/BRT, Protected Bike, and Walk, plus Transit Signal Priority (`TSP`) early-green recall.
  - `transit_gtfs.rs` — Pure-Rust [GTFS Schedule](https://gtfs.org/documentation/schedule/reference/) CSV feed compiler (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`) + automatic arterial route/stop synthesizer, WGS84 stop-to-curb map-matching, parcel-coupled curbside passenger queues, [TRB `TCQSM`](https://onlinepubs.trb.org/onlinepubs/tcrp/tcrp_rpt_165ch-03.pdf) boarding/alighting dwell-time physics (`AllDoorFareFreeBrt`, `SmartcardTap`, `FrontDoorCash`) with standee crowding friction, pass-up tracking, anti-bunching headway holding, and live passenger load `LOS A–F` telemetry.
  - `traffic.rs` — [`Rayon`](https://github.com/rayon-rs/rayon)-parallelized [Treiber et al. (2000) `IDM`](https://link.aps.org/doi/10.1103/PhysRevE.62.1805) car-following (including smooth `IDM` curbside bus-stop braking and door-open dwell holds), [Kesting et al. (2007) `MOBIL`](https://akesting.de/download/MOBIL_TRR_2007.pdf) lane-changing around stopped buses, `CyclistAgent`s, sidewalk/crosswalk `PedestrianAgent`s, and the 50,000+ crowd seeder.
  - `environment.rs` — [Moreno et al. (2021)](https://www.mdpi.com/2624-6511/4/1/6) 15-minute city accessibility contours, [Jiménez-Palacios (1999)](https://dspace.mit.edu/entities/publication/154ebfa2-e765-452e-aa0a-c6f94fd2a880) Vehicle Specific Power (`VSP`) `CO2`/`NOx`/`PM2.5` tailpipe emissions, traffic noise (`dBA`), Urban Heat Island (`delta_T`), and Rational Method (`Q = C i A`) stormwater runoff.
  - `economics.rs` — Value-per-acre municipal fiscal ledger, Jobs-Housing balance, Highway Capacity Manual (`HCM`) `LOS A-F`, and deterministic A/B policy comparison.
  - `batch_eval.rs` — [`Rayon`](https://docs.rs/rayon/latest/rayon/iter/trait.IntoParallelIterator.html) work-stealing multi-scenario batch experiment engine comparing **Fixed-Time**, **Max-Pressure Adaptive**, and **Modern Roundabout** regimes across demand sweeps and Common Random Numbers (`CRN`) replicates, exporting 28-column [RFC 4180](https://www.rfc-editor.org/info/rfc4180/) summary `.csv` scorecards with baseline deltas and `REGIME_MEAN_SUMMARY` roll-ups.
  - `persistence.rs` — Lossless `.json` scenario snapshot save/load (`ScenarioSnapshot` via [`serde_json`](https://docs.rs/serde_json/latest/serde_json/fn.to_string_pretty.html)) and automatic 1 Hz 29-column [RFC 4180](https://www.rfc-editor.org/info/rfc4180/) `.csv` time-series telemetry recorder (`TelemetryTimeSeriesRecorder`).
  - `gpu_pack.rs` — Zero-allocation `#[repr(C)]` `bytemuck::Pod` instance packers (`GpuCameraUniform` 48 B, `GpuRoadQuadInstance` 32 B, `GpuAgentInstance` 32 B) encoding queue-colored curbside `TransitStop` shelter pads, live onboard bus passenger load bars, and 4 Hz boarding door flashers alongside `256x256` RGBA8 GPU heatmaps.
- **`crates/ug-app`** — Unified binary supporting three execution modes:
  - **Custom `wgpu` + `egui` Desktop Workbench (`wgpu_pass.rs` + `shaders/urban_viewport.wgsl`)**: Integrates directly into [`egui_wgpu::CallbackTrait`](https://docs.rs/egui-wgpu/latest/egui_wgpu/trait.CallbackTrait.html) ([`egui`](https://github.com/emilk/egui) `0.30` / `wgpu 23.0`, avoiding Windows 11 backend split panics in [egui #7943](https://github.com/emilk/egui/issues/7943)), executing a **3-kernel [`wgpu` `@compute`](https://docs.rs/wgpu/latest/wgpu/struct.ComputePipelineDescriptor.html) semi-Lagrangian wind advection-diffusion pass** (`shaders/plume_compute.wgsl`: `cs_clear_accum`, `cs_splat_agents`, `cs_advect_diffuse`) followed by **3 [hardware-instanced](https://sotrh.github.io/learn-wgpu/beginner/tutorial7-instancing/) GPU draw calls per frame** (`shaders/urban_viewport.wgsl`) at 120+ FPS (`PresentMode::AutoNoVsync`), plus an asynchronous **`A/B Sweep` dock tab** running isolated [`rayon::ThreadPoolBuilder`](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html) Fixed-Time vs. Max-Pressure vs. Roundabout experiments via non-blocking [`std::sync::mpsc`](https://doc.rust-lang.org/std/sync/mpsc/) polling.
  - **Terminal Studio ([`Ratatui`](https://docs.rs/ratatui/latest/ratatui/))**: `cargo run --release -- tui`
  - **Headless Parallel Experiment & Batch CLI (`batch.rs`)**: `batch`, `simulate`, `compare`, and `import` subcommands with `--gtfs <DIR>`, `--demand-scales`, `--replicates`, and `--threads`.

## Quick Start (Windows 11 MSVC & Linux)

```bash
# 1. Run unit, GTFS dwell/crowding, WGSL Pod byte-alignment, and 50,000+ crowd tests
cargo test --workspace --release

# 2. Launch 120+ FPS wgpu WGSL Workbench in 52,000-Agent Megacity Stress Mode (default)
cargo run --release

# 3. Launch Downtown Corvallis Couplets with the bundled Corvallis GTFS feed fixture
cargo run --release -- gui --scenario corvallis --gtfs assets/corvallis_gtfs --max-pressure

# 4. Compare Front-Door Cash Baseline vs. 15-Min TOD + All-Door BRT and export 1 Hz CSV trajectories
cargo run --release -- compare --scenario corvallis --gtfs assets/corvallis_gtfs --seconds 300 --csv-baseline exports/base.csv --csv-tod exports/tod.csv

# 5. Parallel Rayon multi-regime experiment (Fixed-Time vs. Max-Pressure vs. Modern Roundabout) -> CSV scorecard
cargo run --release -- batch --scenario corvallis --gtfs assets/corvallis_gtfs --seconds 180 --demand-scales 0.75,1.0,1.35 --replicates 2 --csv-out exports/corvallis_scorecard.csv

# 6. Headless single run saving both a reusable .json ScenarioSnapshot and a .csv telemetry log
cargo run --release -- simulate --scenario corvallis --tod-upzone --seconds 180 --save-snapshot exports/corvallis_tod.json --csv-out exports/corvallis_tod.csv
```

## v4.3.0 — Arterial Bridge/Corridor Closure Stress-Test & Disruption-Hardened MCLP

- **`ug_core::resilience` (`N-1` & `N-2` Interdiction Sweep)**: Parallel `Rayon` evaluation of single-corridor (`N-1`) and simultaneous two-corridor (`N-2`) bridge/arterial closures grounded in [Church, Scaparra & Middleton (2004)](https://www.tandfonline.com/doi/abs/10.1111/j.1467-8306.2004.00410.x) and the [Scott et al. (2006) Network Robustness Index (`NRI`)](https://www.researchgate.net/publication/222652140_Network_Robustness_Index_A_new_method_for_identifying_critical_links_and_evaluating_the_performance_of_transportation_networks).
- **Disruption-Hardened MCLP (`solve_robust_mclp_under_disruptions`)**: Re-solves [NFPA 1710 4-minute (`240 s`)](https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf) station siting across the ensemble of intact + worst-case disrupted graphs ([Church & Scaparra fortification](https://kar.kent.ac.uk/25492/1/79_Web_version.pdf)), undoable via `Ctrl+Z` / `Ctrl+Y`.
- **Interactive Viewport & CLI**: Tool `7: Bridge/Road Closure`, Inspector Dock `"Stress-Test"` tab with ranked vulnerability bar chart and 1-click live closure injection, plus `cargo run --release -- stress-test --scenario megacity50k --harden-mclp --csv-out exports/stress.csv`.

## v4.3.0 — Probabilistic Seismic Bridge Fragility Curves & Monte Carlo Post-Event Travel-Time Simulations

- **`ug_core::seismic_mc` (Hazus Lognormal Fragility & Spatially Correlated Ground Motions)**: Models [FEMA Hazus highway bridge classes (`HWB5`–`HWB28`, `beta = 0.60`)](https://ascelibrary.com/doi/10.1061/PPSCFX.SCENG-1464) and [Shinozuka et al. (2000) two-parameter lognormal fragility curves](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9399%282000%29126%3A12%281224%29) across `None / Slight / Moderate / Extensive / Complete` damage states, driven by Cholesky-correlated [Jayaram & Baker (2009) intra-event `Sa(1.0s)` fields](https://www.jackwbaker.com/Publications/Jayaram_Baker_(2009)_spatial_correlation,_EESD.pdf).
- **Parallel `Rayon` Monte Carlo & CRN Bridge Retrofit What-Ifs**: Evaluates post-earthquake [NFPA 1710 4-minute (`240 s`)](https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf) coverage distributions (`Mean`, `P50`, `P10`, `P05`), P90/P95 travel-time tails, Common-Random-Numbers (`CRN`) column-jacket / base-isolation bridge retrofits, and 1-click live `P10` / `Worst` earthquake realization injection (`Ctrl+Z` undoable) inside the `"Stress-Test"` dock and `urban-grid seismic-mc` CLI.

## v4.3.0 — Hazus Post-Earthquake Bridge Restoration Functions (Days 1, 7, 30, and 90) & Multi-Week Recovery Curves

- **FEMA Hazus Continuous Normal Restoration Curves**: Implements [Hazus highway bridge restoration functions (`Slight: 0.6±0.6 d`, `Moderate: 2.5±2.7 d`, `Extensive: 75±42 d`, `Complete: 230±110 d`)](https://rosap.ntl.bts.gov/view/dot/27476/dot_27476_DS1.pdf) with staged emergency shoring and temporary span reopening across `Days 0, 1, 3, 7, 14, 30, 60, and 90`.
- **Multi-Week Coverage Recovery & Bruneau (2003) Resilience Index**: Computes `P10 / P50 / Mean / P90` [NFPA 1710 4-minute coverage](https://apps.usfa.fema.gov/pdf/efop/efo49426.pdf) recovery trajectories, the [Bruneau et al. (2003) 90-day Resilience Index (`R_90`)](https://pubs.geoscienceworld.org/eeri/earthquake-spectra/article/19/4/733/584942/A-Framework-to-Quantitatively-Assess-and-Enhance), CRN repair-crew speedup what-ifs, and undoable (`Ctrl+Z`) live `Day 1 / 7 / 30 / 90` network state injection in `egui` and CLI.

## v4.3.0 — Daskin (1983) Maximum Expected Covering Location Problem (MEXCLP) & Second-Due Backup Coverage

- **Daskin (1983) `MEXCLP` (`1 - q^k` Expected Coverage & Multi-Engine Co-Location)**: Upgrades `ug_core::mclp` from deterministic `MCLP` to [Daskin (1983) `MEXCLP`](https://pubsonline.informs.org/doi/10.1287/trsc.17.1.48) with both fixed busy fraction `q` and endogenous M/G/c call-workload queueing `q(M) = lambda * tau / M` ([Batta, Dolan & Krishnamurthy 1989](https://pubsonline.informs.org/doi/10.1287/trsc.23.4.277)), multi-company double/triple-engine houses, and single-unit substitution (`GAS`).
- **Second-Due Backup Coverage (`k >= 2`, Hogan & ReVelle 1986)**: Evaluates and optimizes overlapping second-due backup coverage (`%`), queueing-weighted expected travel time `sum_r (1-q) q^{r-1} t_r`, and second-due P90 travel time ([Hogan & ReVelle 1986](https://ideas.repec.org/a/pal/jorsoc/v57y2006i1d10.1057_palgrave.jors.2601991.html)) with a 3-series `egui` Pareto chart and concentric multi-company viewport rings.

## v4.3.0 — Larson (1974) Exact M/M/N Hypercube Spatial Queueing Solver & Heterogeneous Per-Unit Workloads

- **`ug_core::hypercube` (Exact `2^N` CTMC Balance Equations + SOR Iteration)**: Implements [Larson (1974) exact `2^N` binary server-state hypercube transition equations](https://www.researchgate.net/publication/222270336_A_hypercube_queuing_model_for_facility_location_and_redistricting_in_urban_emergency_services) under strict parcel preference lists, supporting both `M/M/N/inf` FCFS dispatch queueing (`P_Q` geometric saturation tail) and `M/M/N/N` mutual-aid loss, travel-time-calibrated heterogeneous service rates `mu_n`, and [Larson (1975) `Q`-factor iteration](https://pubsonline.informs.org/doi/10.1287/opre.23.5.845) for `N > 12`.
- **Embedded Hypercube-MEXCLP & Per-Unit `rho_n` Diagnostics**: Wires `BusynessModel::LarsonHypercube1974` directly into `solve_mclp_fire_station_siting`, adds an interactive per-unit heterogeneous busy-fraction (`rho_n`) bar chart in `egui`, and exposes `cargo run --release -- hypercube`.

## What's New in v4.3.0 — Carter, Chaiken & Ignall (1972) / Larson Hypercube Response-District Boundary Optimization

- **Carter, Chaiken & Ignall (1972) Optimal Boundary Offset Theorem (`crates/ug-core/src/hypercube.rs`)**: Parameterizes undominated first-due response districts (`beats`) via zero-sum additive dispatch potentials `s_n` (`sum_n s_n = 0` seconds), assigning each parcel `j` to `u*(j) = argmin_n (t_{n,j} + s_n)`. Positive offsets `s_n > 0` contract overworked central districts inward so core units remain available for nearby high-density calls while idle peripheral units absorb border parcels.
- **Iterative Exact `2^N` Hypercube Workload-Equalization & Pareto Tradeoff Solver**: Combines Larson (1974) workload-feedback updates (`s_n <- s_n + gain * (rho_n - mean_rho)`) with coordinate breakpoint search over the exact `2^N` `M/M/N` hypercube CTMC to trace the undominated **Mean Travel Time `E[T]` vs. Workload Imbalance (`max_rho - min_rho`) Frontier** across `equity_weight in [0.0, 1.0]`, including pure Carter-Chaiken-Ignall travel-time minimization (`equity_weight = 0.0`), balanced Pareto knee (`0.55`), and strict workload equalization.
- **GPU `ResponseDistricts` Heatmap, Seam Overlay, `Ctrl+Z` Undo & `redistrict` CLI**: Adds `HeatmapLayer::ResponseDistricts` with luminous gold-white inter-district boundary seams on the 256x256 `wgpu` overlay, undoable `SimulationWorld::apply_redistricting_plan_tracked`, live beat-aware emergency vehicle dispatch, an interactive Pareto frontier & station offset inspector in `egui`, and `cargo run --release -- redistrict`.

## What's New in v4.3.0 — Diurnal 24-Hour Call & Rush-Hour Travel Profiles + Peak/Off-Peak Dispatch Tables

- **24-Hour USFA/EMS Call Arrival & Commuter Land-Use Shift (`crates/ug-core/src/diurnal.rs`)**: Models exact unit-mean 24-hour call multipliers (`03:00` overnight trough `0.32x` vs. `17:00` surge `1.54x`) alongside spatial commuter population shifts into `CBD`/`IND`/`MU-4` workplaces during daytime hours and back to `R1`/`R3` housing overnight.
- **Bureau of Public Roads (`BPR`, 1964) Tidal Rush-Hour Link Travel-Time Profiles**: Applies directional AM inbound (`07:00-09:59`) and PM outbound (`15:00-18:59`) arterial `V/C` congestion (`t = t_0 * [1 + alpha * (V/C)^beta]`) and signal queue clearance scaling across all 24 hours.
- **Separate Peak vs. Off-Peak Carter-Chaiken-Ignall Dispatch Tables & Green-Kolesar (1991) 24-Hr `PSA`**: Solves dedicated **Peak Rush** and **Off-Peak Free-Flow** response-district run cards (`1st-due` & `2nd-due`), evaluates the 24-hour Pointwise Stationary Approximation (`PSA`) hypercube trajectory, highlights regime-switching boundary parcels on the `wgpu` viewport, supports undoable Peak/Off-Peak table toggling & live 24-hour clock modulation, and adds `cargo run --release -- diurnal`.

## What's New in v4.3.0 — Diurnal 24-Hour Call & Rush-Hour Travel Profiles + Peak/Off-Peak Dispatch Tables

- **24-Hour USFA/EMS Call Arrival & Commuter Land-Use Shift (`crates/ug-core/src/diurnal.rs`)**: Models exact unit-mean 24-hour call multipliers (`03:00` overnight trough `0.32x` vs. `17:00` surge `1.54x`) alongside spatial commuter population shifts into `CBD`/`IND`/`MU-4` workplaces during daytime hours and back to `R1`/`R3` housing overnight.
- **Bureau of Public Roads (`BPR`, 1964) Tidal Rush-Hour Link Travel-Time Profiles**: Applies directional AM inbound (`07:00-09:59`) and PM outbound (`15:00-18:59`) arterial `V/C` congestion (`t = t_0 * [1 + alpha * (V/C)^beta]`) and signal queue clearance scaling across all 24 hours.
- **Separate Peak vs. Off-Peak Carter-Chaiken-Ignall Dispatch Tables & Green-Kolesar (1991) 24-Hr `PSA`**: Solves dedicated **Peak Rush** and **Off-Peak Free-Flow** response-district run cards (`1st-due` & `2nd-due`), evaluates the 24-hour Pointwise Stationary Approximation (`PSA`) hypercube trajectory, highlights regime-switching boundary parcels on the `wgpu` viewport, supports undoable Peak/Off-Peak table toggling & live 24-hour clock modulation, and adds `cargo run --release -- diurnal`.

## What's New in v4.3.0 — Coupled Electrical, Potable Water & Wastewater Lifeline Grids

- **12.47 kV Electrical Distribution Grid (`crates/ug-core/src/utilities.rs`)**: Linearized Baran & Wu (1989) DistFlow / nodal admittance power-flow solver with substation OLTC voltage regulation, ANSI C84.1 Range A (`0.95..1.05 p.u.`) compliance, feeder `% MVA` ampacity & `I^2 R` resistive losses, diurnal rooftop solar PV & EV fast-charging profiles, and `N-1`/`N-2` substation blackout simulation.
- **Potable Water Distribution Grid (`Todini-Pilati / Hazen-Williams`)**: Looped pressurized pipe-network hydraulic solver (`h_f = 10.67 * L * Q^1.852 / (C^1.852 * D^4.8704)`), electrically driven high-service water booster stations, and USFA/AWWA `>= 20 psi` (`14.06 m H2O`) fire-hydrant residual pressure checks under active `1,500+ GPM` suppression demand.
- **Wastewater & Sanitary Sewer Grid (`Manning + US EPA RDII`)**: Topographic gravity sewer routing (`Q_full = (1/n) * A * R_h^(2/3) * S_0^(1/2)`), zoning imperviousness-coupled Rainfall-Derived Infiltration & Inflow (`RDII`), sewage lift stations, `WWTP` hydraulic & `BOD5` capacity utilization, and Sanitary Sewer Overflow (`SSO`) surcharge alerts.
- **Rinaldi, Peerenboom & Kelly (2001) Cross-Lifeline Cascades, `HeatmapLayer::UtilityLifelines` & `utilities` CLI**: Substation outages propagate to un-backed-up water booster pumps (`< 20 psi` pressure collapse), sewage lift stations (`SSO` wet-well backup), and darkened traffic signals (`Ctrl+Z` cascade injection) unless critical microgrid/backup power is enabled; adds `DockTab::UtilityGrids`, `HeatmapLayer::UtilityLifelines`, and `cargo run --release -- utilities`.

## What's New in v4.3.0 — ALA (2001) / Hazus PGV Buried Pipeline Fragility & Multi-Lifeline Seismic Restoration

- **ALA (2001) & FEMA Hazus `PGV` Buried Water & Sewer Pipeline Repair-Rate Fragility (`crates/ug-core/src/seismic_mc.rs`)**: Implements both ALA (2001) linear wave-propagation (`RR = K_1 * K_2 * 0.00187 * PGV_in_s` per `1,000 ft` = `0.0024154 * PGV_cm_s` repairs/km) and Hazus O'Rourke & Ayala (1993) (`RR = K_1 * 0.0001 * PGV_cm_s^2.25` repairs/km) across four pipe classes (`BrittleCastIronVitrifiedClay`, `SegmentedDuctileIronPvc`, `WeldedSteelRestrainedJoint`, and `SeismicErdipHdpe`), sampling spatially correlated CRN Poisson `80%` annular leaks and `20%` severed breaks.
- **Coupled Multi-Lifeline Post-Earthquake Hydraulics & Crew-Prioritized 90-Day Restoration**: Integrates Hazus `ESS` electrical substation `PGA` fragility, pressurized water main breaks + leak orifice demand emitters (`>= 20 psi` fire-flow checks), gravity sewer interceptor collapses (`SSO` surcharges), cross-lifeline booster/lift power cascades, and Porter (2016) / Hazus break-first crew repair queues across Days `0, 1, 3, 7, 14, 30, 60, 90`, plus interactive ERDIP/HDPE pipe-replacement & multi-lifeline recovery curves in `egui` and `ug-app seismic-mc`.

## What's New in v4.3.0 — Scawthorn / Hazus Fire-Following-Earthquake (FFE) & Wind-Driven Conflagration

- **Scawthorn et al. (2005) & FEMA Hazus Ch. 10 Post-Earthquake Ignitions (`crates/ug-core/src/ffe_conflagration.rs`)**: Computes Poisson floor-area ignitions (`Ignitions / 10^6 sq ft TFA`) from spatially correlated `PGA`, `ZoneCode` occupancy hazard (`IND`/`MU-4`/`CBD`/`R3`/`R1`), and seismic Automatic Gas Shutoff Valve (`AGSV`) adoption.
- **Coupled `< 20 psi` Damaged Hydrant Suppression & Multi-Ignition Drawdown**: Couples bridge-detoured fire-engine arrival times (`<= flashover_s`) with post-earthquake Hazen-Williams pipe breaks, annular leaks, substation booster blackouts, simultaneous multi-fire water drawdowns (`65 L/s` per fire), and Auxiliary Fire Cisterns (`PWSS` / `AWSS`) to distinguish suppressed knockdowns from **Dry-Hydrant Escaped Conflagrations**.
- **Hamada (1951) / FEMA (2020) Elliptical Wind-Driven Urban Fire Spread & GPU Burn-Scar Heatmap**: Simulates downwind, crosswind, and upwind flame-front advance (including FEMA Eqs. 8-10 low-wind isotropic convergence, wood-vs-fire-resistive building ratios, street-width radiant/ember jump caps, and `PARK` greenbelt firebreaks), rendered live via `HeatmapLayer::FireFollowingQuake`, `StressSubMode::FireFollowingQuake` in `egui`, and `cargo run --release -- ffe`.

## What's New in v4.3.0 — Post-Earthquake Hospital ED Surge, Burn/Trauma Triage Queueing & Dynamic Ambulance Diversion

- **FEMA P-58 / Hazus Fault-Tree Hospital ED Surge Capacity (`crates/ug-core/src/hospital_surge.rs`)**: Models regional Level-I Trauma & Verified ABA Burn Centers, Level-II/III Acute Receiving Hospitals, and deployable DMAT Mobile Field Hospitals (`MFH`) under spatially correlated `Sa(1.0s)`/`PGA` structural + nonstructural damage coupled to electric substation and `>= 20 psi` municipal water fault trees.
- **Coupled Seismic Trauma + FFE Burn/Smoke Priority Triage Queueing**: Generates FEMA Hazus Ch. 13 indoor trauma casualties (`START Red Immediate`, `Yellow Delayed`, `Green Walk-In`) alongside active Scawthorn/Hamada Fire-Following-Earthquake (`FFE`) thermal burn & inhalation casualties (`BurnCritical >20% TBSA`), advancing strict priority multi-class ED resuscitation bay & ABA `+50%` Burn Surge ICU queues over 15-minute surge waves.
- **Dynamic MOCC Ambulance Hospital-Diversion Routing & GPU Golden-Hour Heatmap**: Compares `ClosestHospitalNoDiversion`, `ThresholdSaturationDiversion` (`>= 90%` Red-Alert), and `DynamicWaitPlusTravelTriage` (minimizing `Travel + Marginal Priority ED Queue Wait + Burn Specialty Transfer Penalty`), rendered via `HeatmapLayer::HospitalTraumaSurge`, `StressSubMode::HospitalTraumaSurge` in `egui`, and `cargo run --release -- surge`.

## What's New in v4.3.0 — Post-Earthquake Rotary-Wing Medevac Helipads, Smoke/Weather Flight Envelopes & Burn-Center Transfers

- **FAA AC 150/5390-2 Hospital Rooftop/Surface Helipads & Park Emergency LZs (`crates/ug-core/src/aeromedical_medevac.rs`)**: Models elevated hospital rooftop `TLOF`/`FATO` pads (coupled to building drift, elevator backup power & UH-60 weight limits), seismically hardened surface pads, pre-designated `PARK` emergency landing zones (`LZ`s), and regional Western Region Burn Disaster Consortium (`WRBDC`) burn hubs (Portland `128 km`, Seattle `365 km`).
- **FAA 14 CFR § 135.609 + Koschmieder FFE Smoke-Plume & Crosswind Flight-Envelope Gates**: Evaluates Gaussian downwind `PM2.5` smoke dispersion from active Fire-Following-Earthquake (`FFE`) conflagrations via Koschmieder optical extinction (`Vis_km = 3.912 / b_ext`), comparing `Day VFR`, `Night Unassisted VFR`, `OpSpec A021 NVIS+HTAWS`, and `WAAS GPS PinS Helicopter IFR` (`350 ft / 0.75 SM`) alongside airframe crosswind limits (`H135`, `AW139`, `UH-60M`).
- **Primary Bridge-Islanded Airlift & Secondary Inter-Hospital Burn-Center Transfers**: Simulates rotary-wing vs. ground `CCT` fallback transfers for overflowed & non-burn (`L2`/`L3`) `BurnCritical` patients, rendered via `HeatmapLayer::AeromedicalMedevac`, `StressSubMode::AeromedicalMedevac` in `egui`, and `cargo run --release -- medevac`.

## What's New in v4.3.0 — FFE Conflagration Road Closures, Smoke Speed Reduction, Staged Evacuation & Arterial Contraflow

- **Active FFE Perimeter Road Severance & Wetterberg/Intini Smoke Speed Reduction (`crates/ug-core/src/evacuation_contraflow.rs`)**: Dynamically severs road segments encroached within `fire_perimeter_road_buffer_m` by advancing Scawthorn/Hamada conflagrations and degrades driving speeds & lane saturation flows under Gaussian downwind `PM2.5` Koschmieder smoke visibility (`Wetterberg et al. 2021` / `WUI-NITY`).
- **Protected Emergency Ambulance / Hospital / Helipad-LZ Corridors & Arterial Contraflow Reversal**: Compares `NoContraflowBaseline`, `FullUnprotectedOneWayOut`, and FHWA/Wolshon `EmsProtectedArterialContraflow` (reversing surplus inbound arterial lanes toward perimeter refuges while strictly preserving `>= 1` dedicated inbound responder lane along Fire, Trauma-Hospital, and Park-`LZ` spines), with live undoable (`Ctrl+Z`/`Ctrl+Y`) application to microscopic `wgpu` traffic!
- **Staged 3-Zone Neighborhood Evacuation & Clearance S-Curves**: Evaluates `SimultaneousUnstaged`, `FireProximityThreeStage`, and `VulnerabilityAndFireWeighted` mobilization (`T50`/`T90`/`T95` clearance times & fire-overrun exposure), rendered via `HeatmapLayer::EvacuationContraflow`, `StressSubMode::EvacuationContraflow` in `egui`, and `cargo run --release -- evacuate`.

## What's New in v4.3.0 — FFE Conflagration Road Closures, Smoke Speed Reduction, Staged Evacuation & Arterial Contraflow

- **Active FFE Perimeter Road Severance & Wetterberg/Intini Smoke Speed Reduction (`crates/ug-core/src/evacuation_contraflow.rs`)**: Dynamically severs road segments encroached within `fire_perimeter_road_buffer_m` by advancing Scawthorn/Hamada conflagrations and degrades driving speeds & lane saturation flows under Gaussian downwind `PM2.5` Koschmieder smoke visibility (`Wetterberg et al. 2021` / `WUI-NITY`).
- **Protected Emergency Ambulance / Hospital / Helipad-LZ Corridors & Arterial Contraflow Reversal**: Compares `NoContraflowBaseline`, `FullUnprotectedOneWayOut`, and FHWA/Wolshon `EmsProtectedArterialContraflow` (reversing surplus inbound arterial lanes toward perimeter refuges while strictly preserving `>= 1` dedicated inbound responder lane along Fire, Trauma-Hospital, and Park-`LZ` spines), with live undoable (`Ctrl+Z`/`Ctrl+Y`) application to microscopic `wgpu` traffic!
- **Staged 3-Zone Neighborhood Evacuation & Clearance S-Curves**: Evaluates `SimultaneousUnstaged`, `FireProximityThreeStage`, and `VulnerabilityAndFireWeighted` mobilization (`T50`/`T90`/`T95` clearance times & fire-overrun exposure), rendered via `HeatmapLayer::EvacuationContraflow`, `StressSubMode::EvacuationContraflow` in `egui`, and `cargo run --release -- evacuate`.

## What's New in v4.3.0 — Post-Earthquake Emergency Shelter Siting, Lifeline Habitability, HEPA Smoke Filtration & Evacuee Routing

- **FEMA P-361 / ARC 4496 Congregate Shelter Siting & ATC-20 Lifeline Habitability (`crates/ug-core/src/emergency_shelters.rs`)**: Evaluates municipal Fieldhouse Arenas, School Gymnasiums, Civic Convention Halls, and Seismic Park Tent Pavilions (`40-60 sq ft/person`) under spatially correlated ATC-20 structural green-tag fragility coupled to potable water, sanitary sewer, and emergency generator fault trees.
- **US EPA / ASHRAE Guideline 44 `MERV-13` + Portable `HEPA` (`CADR`) Smoke-Filtration Mass-Balance**: Solves well-mixed indoor particulate infiltration-filtration (`C_in = C_out * P * lambda_inf / (lambda_inf + k_dep + k_hvac + CADR / V)`) coupled to electrical grid / backup-generator power against the `35 ug/m^3` US EPA `PM2.5` threshold.
- **Two-Stage Hazard-Aware Siting & Capacitated Evacuee Assignment Routing**: Compares `UncoordinatedNearestShelter`, `CapacitatedDistanceOnly`, and `HazardAwareCleanAirCapacitated` across post-quake bridge-detoured, `FFE`-closed, contraflow-boosted networks via `HeatmapLayer::EmergencyShelters`, `StressSubMode::EmergencyShelters` in `egui`, and `cargo run --release -- shelters`.

## What's New in v4.3.0 — Post-Earthquake Multi-Day Shelter & Hospital Commodity Resupply VRP

- **Multi-Commodity Lifeline Logistics (`crates/ug-core/src/commodity_resupply.rs`)**: Models 4 specialized disaster sustainment fleets from a Regional Staging Area (`RSA` depot) across `Days 1..=14` — **Potable Water Tankers (`18,000 L`)**, **Backup Generator Diesel Tankers (`11,400 L`, CMS/NFPA 110 96-hr rule)**, **MRE Ration Dry-Vans (`21,744 meals`)**, and **Medical Oxygen `LOX`/Cylinder Trucks (`120 kL`)** — dynamically coupled to daily bridge reopening and municipal electric/water restoration!
- **Holguín-Veras Deprivation-Cost `IRP` + Clarke-Wright 2-Opt Multi-Stop `SDVRP`**: Compares `StaticEqualRotationMilkRun`, `GreedyMyopicReorderThreshold`, and `DeprivationAwareClarkeWrightIrp` (minimizing convex human deprivation costs via Clarke-Wright savings + exact 2-opt tour consolidation + `UH-60` helicopter sling-load bridge bypass), rendered via `HeatmapLayer::CommodityResupply`, `StressSubMode::CommodityResupply` in `egui`, and `cargo run --release -- resupply`.

## What's New in v4.3.0 — Coupled GTFS Transit-Bus & ADA Paratransit Evacuation into Clean-Air Shelters

- **GTFS Neighborhood Pickup Depots & Two-Tier Vulnerable Evacuation (`crates/ug-core/src/transit_evacuation.rs`)**: Directly couples `TransitGtfsSystem` stops into `400 m` walkable **Neighborhood Pickup Depots (`[GTFS-PP]`)** served by `40-ft` (`50 pax`) and `60-ft` articulated (`80 pax`, TCQSM `AllDoorFareFreeBrt` emergency dwell) shuttle coaches for carless households, alongside curb-to-door **ADA Wheelchair-Lift Paratransit Vans** for mobility-limited residents (`TRB Special Report 294` & `FHWA Special-Needs Evacuation`).
- **Dedicated Bus-Plus-EMS Priority Contraflow Lanes & Clean-Air `TBEP` Routing**: Compares `MixedAutoQueueNoPriority`, `CurbsideBusOnlyNoContraflow`, and `DedicatedBusPlusEmsContraflow` (Zhao et al. 2015 contraflow lane shared exclusively with bidirectional ambulances/fire apparatus, `Ctrl+Z` undoable on the live road network!) paired with Bish (2011) `FireUrgencyCleanAirTbep` dispatch into ATC-20 green-tagged `MERV-13 + HEPA` clean-air shelters (`HeatmapLayer::TransitBusEvacuation`, `StressSubMode::TransitBusEvacuation`, and `cargo run --release -- transit-evac`).

## What's New in v4.3.0 — Cellular Tower Blackout Fragility, WEA/Siren/Door-to-Door Mobilization & Delayed Staging

- **Post-Earthquake Cellular Base-Station (`BTS`) Blackout Fragility (`crates/ug-core/src/warning_mobilization.rs`)**: Models urban rooftop macrocell, steel monopole, and braced lattice `BTS` failure across structural `PGA` shaking (`Tang et al., EERI Earthquake Spectra`), buried/bridge-hung fiber backhaul `PGV` breaks, and electric-grid blackout + VRLA battery depletion (`Legacy 2.8h` vs. `FCC 15-98 8h` vs. `CPUC D.20-07-011 72h + LEO Satellite`).
- **Multi-Channel Warning Diffusion (`WEA` vs. `124 dBC` Outdoor Sirens vs. Hi-Lo/Door-to-Door) & Delayed Departure Staging**: Convolves Lindell & Perry (`PADM` / `EMBLEM2`) Weibull warning-receipt + preparation S-curves with NIST TN 1950 outdoor/indoor siren acoustics (`-22 dB` closed-window insertion loss) and `BlackoutCompensatedAdaptiveStaging` (`Ctrl+Z` undoable parcel staging!), rendered via `HeatmapLayer::WarningMobilization`, `StressSubMode::WarningMobilization`, and `cargo run --release -- warning`.

## What's New in v4.3.0 — FEMA Communications Systems (IPAWS-OPEN CAP v1.2, NPWS/PEP EAS, NOAA SAME, FNARS & MERS ESF #2)

- **FEMA `IPAWS-OPEN` (`CAP v1.2`) Multi-Pathway Public Alerting (`crates/ug-core/src/fema_comms.rs`)**: Models County `COG` EOC `CAP v1.2` XML origination resilience (`Commercial Fiber` vs. **`FNARS` `3–30 MHz` HF ALE/NVIS Skywave** + **`MERS` Ku/Ka/LEO VSAT**) feeding **`WEA 3.0`** (`360-char` `0.1-mi` polygon cell-broadcast), **`EAS` via FEMA `NPWS` Primary Entry Point (`PEP`)** (`50 kW` EMP/seismic-hardened `60-day` diesel AM/FM groundwave broadcast, `47 CFR Part 11`), and **`NOAA Weather Radio All Hazards` (`NWEM` `162.550 MHz` VHF + `1050 Hz` `SAME` bedroom auto-wake alarm)**.
- **FEMA Region 10 `MERS` (`Bothell MOC` `IRV`/`MEOV` + Fly-Away Kits) & `ESF #2` `NIFOG` Tactical `LMR` Interoperability**: Deploys mobile `MERS` satellite/repeater/FirstNet `COLT` vehicles across damaged bridges (with `UH-60` Fly-Away airlift fallback and `Ctrl+Z` undoable parcel/EOC state updates), rendered via `HeatmapLayer::FemaCommsArchitecture`, `StressSubMode::FemaCommsArchitecture`, and `cargo run --release -- fema-comms`.

## What's New in v4.3.0 — FEMA Communications Systems (IPAWS-OPEN CAP v1.2, NPWS/PEP EAS, NOAA SAME, FNARS & MERS ESF #2)

- **FEMA `IPAWS-OPEN` (`CAP v1.2`) Multi-Pathway Public Alerting (`crates/ug-core/src/fema_comms.rs`)**: Models County `COG` EOC `CAP v1.2` XML origination resilience (`Commercial Fiber` vs. **`FNARS` `3–30 MHz` HF ALE/NVIS Skywave** + **`MERS` Ku/Ka/LEO VSAT**) feeding **`WEA 3.0`** (`360-char` `0.1-mi` polygon cell-broadcast), **`EAS` via FEMA `NPWS` Primary Entry Point (`PEP`)** (`50 kW` EMP/seismic-hardened `60-day` diesel AM/FM groundwave broadcast, `47 CFR Part 11`), and **`NOAA Weather Radio All Hazards` (`NWEM` `162.550 MHz` VHF + `1050 Hz` `SAME` bedroom auto-wake alarm)**.
- **FEMA Region 10 `MERS` (`Bothell MOC` `IRV`/`MEOV` + Fly-Away Kits) & `ESF #2` `NIFOG` Tactical `LMR` Interoperability**: Deploys mobile `MERS` satellite/repeater/FirstNet `COLT` vehicles across damaged bridges (with `UH-60` Fly-Away airlift fallback and `Ctrl+Z` undoable parcel/EOC state updates), rendered via `HeatmapLayer::FemaCommsArchitecture`, `StressSubMode::FemaCommsArchitecture`, and `cargo run --release -- fema-comms`.

## What's New in v4.3.0 — FEMA Incident Staging Bases (ISB), USACE IS-26 POD Queueing & IPAWS Public POD Routing

- **FEMA `ISB -> CSA -> C-POD` Echelon Logistics & USACE `IS-26` Typed POD Queueing (`crates/ug-core/src/fema_isb_pods.rs`)**: Models federal **Incident Staging Base (`ISB`)** `53-ft` commodity tractor-trailer allocation + mid-shift `MERS` hotshot/`CH-47` restock into **Type I (`4 lanes, 560 veh/hr, 20,000 pax/day`)**, **Type II (`2 lanes, 280 veh/hr, 10,000 pax/day`)**, **Type III (`1 lane, 140 veh/hr, 5,000 pax/day`)**, and **Pedestrian/Transit Walk-Up `C-POD`s** using Sakasegawa/Kingman `M/G/c` + fluid-overload queueing and on-street arterial queue spillback!
- **FEMA `IPAWS` (`WEA 3.0` + `NPWS`/`PEP` `EAS` + `NOAA SAME`) Public `POD` Routing**: Compares `UncoordinatedNearestPodRush`, `StaticPrePublishedDistricts`, and `DynamicIpaswQueueAndStockBalanced` (logit wait/stockout diversion + odd/even arrival staggering + `Ctrl+Z` undoable live roadway spillback injection), rendered via `HeatmapLayer::FemaIsbPods`, `StressSubMode::FemaIsbPods`, and `cargo run --release -- fema-pods`.

## What's New in v4.3.0 — FEMA US&R Type 1 Task Force Search Grids, Roadway Debris Push & ISB/POD Reopening

- **FEMA `ESF #9` `Type 1 US&R Task Force` (`80`-Person 24-Hr Dual Shifts) & `USNG` Collapse Grids (`crates/ug-core/src/fema_usar_debris.rs`)**: Partitions the urban footprint into **US National Grid (`USNG`)** collapse sectors (`URM` brick, non-ductile `RC` slab, and wood/steel archetypes), modeling Hazus-MH street-spill rubble tonnage, FEMA/INSARAG `[!] -> [/] -> [X]` search progression, canine/Delsar void triage, heavy concrete breaching squads, and **Coburn-Spence / Macintyre `72-hour` entrapped rubble survival decay `S(t)`**!
- **Co-Optimized `ESF #3` `"Teeth-of-a-Comb"` Bulldozer Phase I Push & `ISB -> C-POD` Route Reopening**: Compares `UncoordinatedPerimeterHaulOff` (`32 t/h`), `LogisticsPodCorridorsFirstPush` (`115 t/h`), and `CoOptimizedTeethOfCombUsarAndPodPush` (`128 t/h` Caunhye 2020 bi-objective emergency push + `Ctrl+Z` undoable Hour-`H` roadway debris/reopening injection!), rendered via `HeatmapLayer::FemaUsarDebris`, `StressSubMode::FemaUsarDebris`, and `cargo run --release -- fema-usar`.

## What's New in v4.3.0 — Clean Main Entrance Menu Hub, 5-Domain Launcher & `Ctrl+K` Command Palette

- **Dedicated Main Entrance Menu Hub (`GuiScreen::MainEntranceHub` in `crates/ug-app/src/gui.rs`)**: Opens cleanly onto a structured architectural Entrance Hub featuring **4 Built-In Benchmark Scenario Cards + Snapshot `.json` Loader**, a **Live Simulation World Telemetry Banner**, and a searchable **5-Domain, 20-Module Directory** (1. *Traffic, Street CAD & Multimodal Transit*; 2. *Emergency Districting & Utility Grids*; 3. *Seismic Fragility, FFE & Medical Surge*; 4. *Evacuation, Clean-Air Shelters & VRP Sustainment*; 5. *FEMA ESF-2 Communications, ISB/POD Logistics & Type-1 US&R*) that configures the exact `DockTab`, `StressSubMode`, `ViewportTool`, and `HeatmapLayer` in one click!
- **Decluttered Workbench Header, Collapsible CAD Bar, 4-Category Resilience Filter & `Ctrl+K` Palette**: Toggle effortlessly between `[Home Menu (F1)]` and the `120+ FPS` `wgpu` map workbench (`F1` / `Home`), filter any of the 20+ modules or scenarios from anywhere via **`Ctrl+K`**, collapse the CAD bar when inspecting resilience runs, and browse `DockTab::ResilienceStress` cleanly via 4 domain category pills.

## v5.0.0 — Complete FEMA Recovery Suite & ALERTWest / UOregon OHAZ WUI Wildfire Response

- **Complete FEMA `ESF #3` & Individual Assistance (`IA`) Suite (`crates/ug-core/src/fema_gsb_drc_tdms.rs`)**:
  - **FEMA Generator Staging Base (`GSB`) & USACE `249th Engineer Battalion (Prime Power)`**: Pre-installation switchgear/load assessments, prime-power diesel pack sizing (`60–1,150 kW`), and convoy dispatch to hospitals, water/wastewater plants, clean-air shelters, and `C-POD`s (`Ctrl+Z` undoable live backup power activation).
  - **EPA / FEMA-325 Temporary Debris Management Sites (`TDMS`)**: 4-stream disaster debris segregation (`Vegetative` `75%` air-curtain/grinding volume reduction, `C&D` aggregate recycling, mandatory EPA Phase 1 `HHW` sweep, and `White Goods` Sec. 608 Freon recovery) with ADMS load-ticket towers and `30 yd^3` dump-truck haul-off fleets.
  - **FEMA `DRC` & Roving `MRIC` Individual Assistance Queueing**: Fixed multi-agency `DRC`s (`FEMA IA + U.S. SBA` disaster loans + appeals) paired with GTFS-walkable Mobile Registration Intake Centers (`MRIC`s) modeled via Allen-Cunneen `M/G/c` survivor intake queues.
- **`ALERTWest` & University of Oregon `OHAZ` Wildfire Camera Network + `NWCG` WUI Suppression (`crates/ug-core/src/wildfire_alertwest_ohaz.rs`)**:
  - **Native `ALERTWest` (`https://www.alertwest.org/`) & `UOregon OHAZ` (`https://ohaz.uoregon.edu/wildfire-cameras/`) Feed Linker**: Parses `.json`, GeoJSON `FeatureCollection`, and `.csv` camera/detection feeds (including bundled `data/alertwest_ohaz_oregon_cameras.json`) with `360°` pan, `40x` optical zoom, `30 mi` day / `40 mi` near-infrared (`NIR`) night vision, `2-min` AI scan cadence, co-located `PNSN`/`ShakeAlert` seismic telemetry, and `< 50 lb` Starlink **Mobile Deployment Units (`MDU`s)**.
  - **Weighted Least-Squares (`WLS`) Multi-Camera Cross-Azimuth Triangulation**: Intersects `>= 2` camera smoke bearings to localize WUI ignitions (`CEP_90` meters) in `2–4 min` vs. `911` visual calls.
  - **Rothermel (1972) / `NWCG` (`RMRS-GTR-253`) WUI Fire Behavior & Interagency Suppression**: Evaluates flame length (`<= 4 ft`, `4–8 ft`, `8–11 ft`, `> 11 ft` crowning thresholds), `20`-person **Type 1 Hotshot Crews (`IHC`)**, **Wildland Bulldozers**, **Type 3/6 WUI Engines**, **Helitack Buckets**, and **`VLAT`/`LAT` `Phos-Chek` Retardant Lines**, coupled to Oregon Level 1/2/3 `IPAWS` evacuation & `Ctrl+Z` fuelbreaks.

```bash
# Run FEMA ESF-3 Prime Power GSB + TDMS Debris + DRC/MRIC Recovery Monte Carlo
cargo run --release -- fema-recovery --scenario corvallis --policy integrated-suite --prt-teams 4 --drcs 2 --mrics 3

# Run ALERTWest + UOregon OHAZ PTZ Camera Triangulation & NWCG WUI Wildfire Suppression with linked camera JSON
cargo run --release -- wildfire --scenario corvallis --camera-feed data/alertwest_ohaz_oregon_cameras.json --posture full-aerial --wind-mph 26 --rh-pct 12
```

## v5.1.0 — Post-Wildfire Burn-Scar Hydrophobic Runoff, USGS Debris-Flow Basins & Culvert Washouts

- **Post-Wildfire Watershed & Culvert Coupling (`crates/ug-core/src/burn_scar_debris_flow.rs`)**:
  - **USFS `BAER` Soil Burn Severity (`dNBR`) & Hydrophobic Curve Numbers (`CN`)**: Couples wildfire burn perimeters from `wildfire_alertwest_ohaz.rs` to water-repellent soil `CN` shifts (`60 -> 86/94`), reduced initial abstraction (`Ia = 0.08 S`), overland sheetflow amplification, and sediment-bulked peak discharge `Q_bulked`.
  - **USGS `Staley et al. (2016/2017) M1` Likelihood & `Gartner et al. (2014)` Sediment Volume**: Evaluates `X = -3.63 + 0.41*R15*MH23 + 0.67*R15*(dNBR/1000) + 0.70*R15*KF`, exact basin `I15_P50` (`mm/hr`) intensity thresholds, and `ln(V_m3)` debris-flow yields across delineated upslope basins.
  - **`FHWA HDS-5` Stormwater Culvert Plugging, `HW/D` Overtopping & Roadway/Sewer Coupling**: Simulates woody-slash barrel occlusion, submerged orifice headwater `HW/D`, roadway embankment scour washouts (`Ctrl+Z` undoable road closures + severed utility mains + sanitary sewer `RDII` storm surcharge), and USFS `BAER` wood-strand mulch + grizzly trash-rack + twin-box culvert retrofits.

```bash
# Run Post-Wildfire Burn-Scar Hydrophobic Runoff, USGS M1/Gartner Basins & FHWA HDS-5 Culvert Washouts
cargo run --release -- debris-flow --scenario corvallis --baer-policy full-baer --i15-mm-hr 38 --storm-mm 72
```

## v5.2.0 — Post-Wildfire Ash/Turbidity Spikes, WTP Intake Shutoffs, Coagulant Dosing & Storage Drawdown

- **Post-Wildfire Source-Water & `WTP` Process Coupling (`crates/ug-core/src/wtp_turbidity_storage.rs`)**:
  - **Riverine Ash, Turbidity (`NTU`), Pyrogenic `DOC` & Alkaline `pH` Chemographs**: Couples `BurnScarDebrisFlowReport` sediment yields (`Gartner et al., 2014`) into `72-hour` storm pulses at the surface-water intake (`3.5 -> 5,000+ NTU`, `DOC` `2.1 -> 30 mg/L`, wood-ash `pH -> 9.2`).
  - **SCADA Intake Shutoffs & Enhanced `FeCl3` + `PAC` Coagulation**: Evaluates jar-test coagulant stoichiometry (`Hohner et al., 2016/2019`; `Writer et al., 2014`), bulk chemical tank inventory depletion, clarifier sludge solids, filter backwash recycle derating (`5% -> 28%`), and US EPA `IESWTR` (`<= 0.30 NTU`) Combined Filter Effluent compliance.
  - **Finished-Water Clearwell / Pressure-Zone Tank Drawdown & `< 20 psi` Boil-Water Advisories**: Integrates clearwell + elevated tank storage (`MG`) against diurnal municipal demand, active WUI fire-hydrant draft (`GPM`), and emergency groundwater well/intertie blending (`MGD`), with `Ctrl+Z` undoable coupling into `utilities.rs` Hazen-Williams residual hydrant pressures.

```bash
# Run 72-Hour Post-Wildfire WTP Turbidity Shutoff, Coagulant Dosing & Finished-Water Storage Drawdown
cargo run --release -- wtp-turbidity --scenario corvallis --policy ferric-pac-wells --shutoff-ntu 350 --wells-mgd 5.5
```

## Citation & CRV Science Apps Hub

- **CRV Science Apps Hub**: [https://crvscience.com/apps](https://crvscience.com/apps)
- **Companion Preprint (`APP-PREPRINT-11`)**: *Urban-Grid: Microscopic Traffic Simulation, Maximal Covering Emergency Siting, and Multi-Hazard Lifeline Resilience* — [https://crvscience.com/apps/preprints/urban-grid-microscopic-traffic-resilience](https://crvscience.com/apps/preprints/urban-grid-microscopic-traffic-resilience)
- **Concept DOI**: [10.5281/zenodo.24222394](https://doi.org/10.5281/zenodo.24222394)
- **Repository**: [https://github.com/bpwhite/crvscience-urban-grid](https://github.com/bpwhite/crvscience-urban-grid)
- **Validation Telemetry**: See [`validation/README.md`](validation/README.md) (`cargo test --workspace`).

```bibtex
@article{white2026urbangrid,
  author  = {White, Bryan P.},
  title   = {Urban-Grid: Microscopic Traffic Simulation, Maximal Covering Emergency Siting, and Multi-Hazard Lifeline Resilience},
  journal = {CRV Science Apps Preprints},
  year    = {2026},
  doi     = {10.5281/zenodo.24222394},
  url     = {https://crvscience.com/apps/preprints/urban-grid-microscopic-traffic-resilience},
  note    = {Preprint ID: APP-PREPRINT-11; Software version 5.2.0}
}
```


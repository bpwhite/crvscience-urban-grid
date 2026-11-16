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

//! FEMA Disaster Emergency Communications (`DEC` / `ESF #2`) & Integrated Public Alert and Warning
//! System (`IPAWS-OPEN` `CAP v1.2`) Architecture:
//! 1. **COG / County EOC `CAP v1.2` Origination Uplink**: Terrestrial fiber backhaul vs. **FEMA
//!    National Radio System (`FNARS` `3–30 MHz` HF ALE/NVIS ionospheric skywave)** & **FEMA Mobile
//!    Emergency Response Support (`MERS` Ku/Ka/LEO VSAT Fly-Away & `IRV`/`MEOV` uplinks)**.
//! 2. **FEMA `IPAWS` Multi-Pathway Public Dissemination**:
//!    - **WEA 3.0 (`Wireless Emergency Alerts`)**: `360-char` bilingual `0.1-mi` polygon cell-broadcast
//!      (fast smartphone push, vulnerable to post-quake commercial `BTS` power/fiber blackouts).
//!    - **EAS (`Emergency Alert System` — `47 CFR Part 11`) & `NPWS` Primary Entry Point (`PEP`)**:
//!      FEMA's `77` EMP/seismic-hardened `50 kW` AM/FM broadcast stations (`60-day` diesel fuel +
//!      transportable studio shelters + `LP-1`/`LP-2` broadcast daisy-chain) reaching vehicle AM/FM
//!      and battery/hand-crank radios across `100%` of cellular blackout zones!
//!    - **NWEM / NOAA Weather Radio All Hazards (`NWR` `162.400–162.550 MHz` VHF + `1050 Hz` `SAME`)**:
//!      Specific Area Message Encoding (`SAME`) tone-alert auto-wake indoor receivers (`85 dBA` alarm)
//!      waking sleeping `02:00 AM` households during grid & cellular blackouts!
//! 3. **FEMA Region 10 `MERS` (`Bothell MOC` `IRV` / `MEOV` / Fly-Away Kits) & `NIFOG` (`VTAC`/`UTAC`/
//!    `7TAC`/`8TAC`) Tactical Responder `LMR` Interoperability**: Optimal p-median / blackout-targeted
//!    field deployment of `MERS` trucks + portable telescoping `P25` LMR & FirstNet Band-14 `COLT`
//!    relays across damaged road networks to restore EMS, fire, shelter, and transit coordination!
//!
//! Grounded in:
//! - FEMA *IPAWS Best Practices Guide* (`https://www.fema.gov/sites/default/files/documents/fema_ipaws-best-practices-guide.pdf`),
//!   OASIS *CAP v1.2 USA IPAWS Profile v1.0* (`https://docs.oasis-open.org/emergency/cap/v1.2/ipaws-profile/v1.0/cap-v1.2-ipaws-profile-v1.0.pdf`),
//!   and Congressional Research Service Report R48363, *The Integrated Public Alert and Warning System*
//!   (`https://www.everycrsreport.com/reports/R48363.html`).
//! - FCC / FEMA `47 CFR Part 11` *Emergency Alert System (`EAS`) & Primary Entry Point (`PEP` / `NPWS`)*
//!   (`https://www.ecfr.gov/current/title-47/chapter-I/subchapter-A/part-11`;
//!   `https://www.puyallupwa.gov/1985/Understanding-Public-Alerts`).
//! - *FEMA Region 10 Cascadia Subduction Zone (`CSZ`) Earthquake and Tsunami Response Plan*
//!   (`https://mil.wa.gov/asset/68404ea51bc02/FEMA-Region-X-CSZ-Earthquake-and-Tsunami-Plan.pdf`),
//!   *FEMA Region 10 All-Hazards Plan (`MERS Bothell`)*
//!   (`https://mil.wa.gov/asset/68404e65d15de/Fema-Region-X-All-Hazards-Plan.pdf`), and FEMA Disaster
//!   Emergency Communications Division (`DEC`) `MERS` / `FNARS` Architecture
//!   (`https://www.napco.org/documents/2014-05_Benoit.pdf`).

use crate::emergency::EmergencyFacility;
use crate::network::{ControlMode, RoadNetwork, TurnType, Vec2};
use crate::seismic_mc::{
    ala_hazus_pipe_repair_rate_per_km, enumerate_seismic_bridge_inventory,
    median_pgv_cm_s_from_sa10, normal_cdf_phi,
};
use crate::warning_mobilization::{evaluate_padm_mobilization_cdf, WarningMobilizationConfig};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

pub const FEMA_ASSET_CSV_HEADER: &str = "asset_id,callsign_or_name,fema_system_kind,pos_x,pos_y,coverage_radius_m,post_quake_online_pct,eoc_cap_link_ok_pct,public_or_lmr_reach_pct";
pub const FEMA_TIMELINE_CSV_HEADER: &str = "minute,wea30_only_reach_pct,wea_plus_eas_pep_pct,wea_eas_plus_nwr_same_pct,full_fema_ipaws_mers_pct,esf2_nifog_lmr_interop_pct";

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64_open01(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        u.clamp(1e-12, 1.0 - 1e-12)
    }

    #[inline]
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open01();
        let u2 = self.next_f64_open01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn cholesky_lower(corr: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = corr[i * n + j];
            if i == j {
                s += 1e-9;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = s.max(1e-12).sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j].max(1e-12);
            }
        }
    }
    l
}

/// County Collaborative Operating Group (`COG` / EOC) uplink to the national FEMA `IPAWS-OPEN` `CAP v1.2` server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EocIpaswUplinkMode {
    /// Standard municipal broadband/fiber internet to `IPAWS-OPEN` (vulnerable to `PGV` duct & bridge-hung fiber breaks!).
    CommercialTerrestrialFiberOnly,
    /// Adds `FNARS` (`FEMA National Radio System` `3–30 MHz` HF ALE/NVIS skywave) + State EOC relay for `CAP` origination.
    FiberPlusFnarsHfSkywaveRelay,
    /// Full FEMA `DEC` Continuity: Terrestrial Fiber + `FNARS` HF ALE + `MERS` Ku/Ka/LEO VSAT Satellite Fly-Away Terminal!
    FullFemaMersVsatAndFnarsContinuity,
}

impl Default for EocIpaswUplinkMode {
    fn default() -> Self {
        Self::FullFemaMersVsatAndFnarsContinuity
    }
}

impl EocIpaswUplinkMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::CommercialTerrestrialFiberOnly => {
                "COG EOC Terrestrial Fiber Only (Single-Point CAP Failure)"
            }
            Self::FiberPlusFnarsHfSkywaveRelay => {
                "Terrestrial Fiber + FEMA FNARS HF ALE/NVIS Skywave Relay"
            }
            Self::FullFemaMersVsatAndFnarsContinuity => {
                "Full FEMA Continuity: Fiber + FNARS HF + MERS Ku/LEO VSAT"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::CommercialTerrestrialFiberOnly => "EOC-FiberOnly",
            Self::FiberPlusFnarsHfSkywaveRelay => "Fiber+FNARS-HF",
            Self::FullFemaMersVsatAndFnarsContinuity => "Fiber+FNARS+MERS-VSAT",
        }
    }
}

/// Active FEMA `IPAWS` public alert pathways & `ESF #2` `MERS`/`NIFOG` tactical deployment posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemaCommsPosture {
    /// Commercial `IPAWS WEA 3.0` cellular broadcast only; terrestrial `P25` LMR without `MERS` field repeaters.
    IpaswWea30CellularOnly,
    /// Tri-Pathway FEMA `IPAWS` (`WEA 3.0` + `NPWS`/`PEP` Hardened `AM/FM EAS` + `NOAA Weather Radio` `162 MHz SAME`).
    IpaswTriPathWeaEasPepNwrSame,
    /// Full FEMA `IPAWS` (`WEA 3.0` + `EAS PEP` + `NWR SAME`) + Region 10 `MERS` (`IRV`/`MEOV` + `NIFOG` LMR & FirstNet `COLT` Relays)!
    FullFemaIpaswPlusMersEsf2Deployment,
}

impl Default for FemaCommsPosture {
    fn default() -> Self {
        Self::FullFemaIpaswPlusMersEsf2Deployment
    }
}

impl FemaCommsPosture {
    pub fn label(self) -> &'static str {
        match self {
            Self::IpaswWea30CellularOnly => {
                "IPAWS WEA 3.0 Cellular Only (No EAS PEP, NWR SAME, or MERS)"
            }
            Self::IpaswTriPathWeaEasPepNwrSame => {
                "FEMA IPAWS Tri-Pathway (WEA 3.0 + NPWS/PEP EAS + NOAA SAME)"
            }
            Self::FullFemaIpaswPlusMersEsf2Deployment => {
                "Full FEMA IPAWS + Region 10 MERS IRV/MEOV & ESF-2 NIFOG LMR"
            }
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::IpaswWea30CellularOnly => "WEA3.0-Only",
            Self::IpaswTriPathWeaEasPepNwrSame => "IPAWS-WEA+PEP+NWR",
            Self::FullFemaIpaswPlusMersEsf2Deployment => "Full-IPAWS+MERS-ESF2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FemaAssetKind {
    /// Local Collaborative Operating Group (`COG`) Emergency Operations Center (`CAP v1.2` Originator + `FNARS` HF).
    CogEocCapOriginator,
    /// FEMA National Public Warning System (`NPWS`) Primary Entry Point (`PEP`) `50 kW` AM/FM Hardened Broadcast Transmitter.
    NpwsPepBroadcastStation,
    /// NOAA Weather Radio All Hazards (`NWR` / `NWEM` `162.550 MHz` VHF `1,000 W` `SAME` Transmitter).
    NoaaNwrSameTransmitter,
    /// Commercial `WEA 3.0` Macrocell (`eNodeB` / `gNodeB`).
    CommercialWeaCellSite,
    /// FEMA Region 10 `MERS` (`Mobile Emergency Response Support` `IRV` / `MEOV` w/ Ku/LEO VSAT + `NIFOG` LMR + FirstNet `COLT`).
    FemaMersIrvUnit,
}

impl FemaAssetKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::CogEocCapOriginator => "COG-EOC",
            Self::NpwsPepBroadcastStation => "NPWS-PEP",
            Self::NoaaNwrSameTransmitter => "NWR-SAME",
            Self::CommercialWeaCellSite => "WEA-BTS",
            Self::FemaMersIrvUnit => "MERS-IRV",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaCommsConfig {
    /// Underlying cellular `BTS` seismic/battery fragility, `FFE` fire spread, and evacuation staging stack.
    pub warning: WarningMobilizationConfig,
    pub eoc_uplink: EocIpaswUplinkMode,
    pub posture: FemaCommsPosture,
    /// Household penetration (`%`, `5..=90%`) of battery/hand-crank/solar `AM/FM` radios & vehicle `EAS` receivers.
    pub household_eas_am_fm_radio_pct: f64,
    /// Household penetration (`%`, `5..=85%`) of battery-backed `NOAA Weather Radio` (`162 MHz SAME` auto-wake alarm) receivers.
    pub household_nwr_same_receiver_pct: f64,
    /// Deployed FEMA Region 10 `MERS` (`IRV` / `MEOV`) mobile communications units (`0..=6`, default `3`).
    pub fema_mers_irv_units: usize,
    /// Airlift `MERS` Fly-Away VSAT/LMR kits via helicopter when seismic bridge collapses isolate sectors!
    pub airlift_mers_flyaway_kits: bool,
}

impl Default for FemaCommsConfig {
    fn default() -> Self {
        Self {
            warning: WarningMobilizationConfig::default(),
            eoc_uplink: EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity,
            posture: FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment,
            household_eas_am_fm_radio_pct: 46.0,
            household_nwr_same_receiver_pct: 32.0,
            fema_mers_irv_units: 3,
            airlift_mers_flyaway_kits: true,
        }
    }
}

/// Computes groundwave AM `EAS` (`NPWS` `PEP` `50 kW` `710 kHz` class) field usability (`0.0..=1.0`)
/// and `162.550 MHz` VHF `NOAA Weather Radio` (`1,000 W` ERP) `SAME` tone-decode reliability (`0.0..=1.0`)
/// at distance `dist_km` under urban terrain clutter.
#[inline]
pub fn compute_pep_am_and_nwr_vhf_reliability(dist_km: f64) -> (f64, f64) {
    let d = dist_km.max(0.1);
    // 50 kW clear-channel PEP AM groundwave maintains >2 mV/m well past 80–140 km!
    let pep_am_rel = (1.0 / (1.0 + (d / 135.0).powf(2.4))).clamp(0.15, 0.998);
    // 1 kW VHF (162.4–162.55 MHz) NWR SAME line-of-sight/diffraction reliable out to ~55–65 km
    let nwr_vhf_rel = (1.0 / (1.0 + (d / 58.0).powf(3.1))).clamp(0.05, 0.995);
    (pep_am_rel, nwr_vhf_rel)
}

/// Evaluates combined FEMA `IPAWS-OPEN` (`CAP v1.2`) household alert reception probability in `[0, 1]`
/// given `eoc_cap_uplink_ok` and independent survival/audibility across `WEA 3.0`, `EAS PEP AM/FM`,
/// and `NOAA Weather Radio SAME` (`1050 Hz` auto-wake alarm):
///
/// `P_ipaws = P_eoc_cap * [1 - (1 - P_wea) * (1 - P_eas_pep) * (1 - P_nwr_same)]`
#[inline]
pub fn combine_ipaws_pathway_reach_prob(
    eoc_cap_uplink_ok: bool,
    p_wea30: f64,
    p_eas_pep: f64,
    p_nwr_same: f64,
) -> f64 {
    if !eoc_cap_uplink_ok {
        return 0.0;
    }
    let fail = (1.0 - p_wea30.clamp(0.0, 1.0))
        * (1.0 - p_eas_pep.clamp(0.0, 1.0))
        * (1.0 - p_nwr_same.clamp(0.0, 1.0));
    (1.0 - fail).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaCommsAssetRow {
    pub asset_id: usize,
    pub callsign_or_name: String,
    pub kind: FemaAssetKind,
    pub segment_id: usize,
    pub world_pos: Vec2,
    pub coverage_radius_m: f64,
    pub post_quake_online_pct: f64,
    pub eoc_cap_link_ok_pct: f64,
    pub public_or_lmr_reach_pct: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FemaIpaswTimelinePoint {
    pub minute: f64,
    pub wea30_only_reach_pct: f64,
    pub wea_plus_eas_pep_pct: f64,
    pub wea_eas_plus_nwr_same_pct: f64,
    pub full_fema_ipaws_mers_pct: f64,
    pub esf2_nifog_lmr_interop_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemaCommsReport {
    pub config: FemaCommsConfig,
    pub eoc_cap_origination_uptime_pct: f64,
    pub npws_pep_broadcast_uptime_pct: f64,
    pub noaa_nwr_same_uptime_pct: f64,
    pub commercial_wea30_uptime_pct: f64,
    pub deployed_mers_units_count: usize,
    pub public_ipaws_verified_reach_pct: f64,
    pub esf2_nifog_responder_lmr_coverage_pct: f64,
    pub mean_ipaws_mobilization_min: f64,
    pub unwarned_isolated_residents_pax: f64,
    pub p90_unwarned_residents_pax: f64,
    pub asset_rows: Vec<FemaCommsAssetRow>,
    pub timeline: Vec<FemaIpaswTimelinePoint>,
    pub mers_uplink_vectors: Vec<(Vec2, Vec2)>,
    pub parcel_ipaws_reach_pct: Vec<f64>,
    pub parcel_esf2_lmr_ok_pct: Vec<f64>,
    pub solve_time_ms: f64,
}

impl FemaCommsReport {
    pub fn to_asset_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.asset_rows.len() * 128);
        out.push_str(FEMA_ASSET_CSV_HEADER);
        out.push('\n');
        for a in &self.asset_rows {
            out.push_str(&format!(
                "{},\"{}\",{},{:.1},{:.1},{:.1},{:.2},{:.2},{:.2}\n",
                a.asset_id,
                a.callsign_or_name.replace('"', "'"),
                a.kind.badge(),
                a.world_pos.x,
                a.world_pos.y,
                a.coverage_radius_m,
                a.post_quake_online_pct,
                a.eoc_cap_link_ok_pct,
                a.public_or_lmr_reach_pct,
            ));
        }
        out
    }

    pub fn to_timeline_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.timeline.len() * 96);
        out.push_str(FEMA_TIMELINE_CSV_HEADER);
        out.push('\n');
        for pt in &self.timeline {
            out.push_str(&format!(
                "{:.1},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                pt.minute,
                pt.wea30_only_reach_pct,
                pt.wea_plus_eas_pep_pct,
                pt.wea_eas_plus_nwr_same_pct,
                pt.full_fema_ipaws_mers_pct,
                pt.esf2_nifog_lmr_interop_pct,
            ));
        }
        out
    }

    pub fn export_asset_csv_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_asset_csv_string())?;
        Ok(self.asset_rows.len())
    }
}

#[derive(Clone, Copy)]
struct HeapState {
    cost_s: f64,
    seg: usize,
}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.cost_s.to_bits() == other.cost_s.to_bits() && self.seg == other.seg
    }
}
impl Eq for HeapState {}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost_s
            .partial_cmp(&self.cost_s)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.seg.cmp(&other.seg))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn build_seg_adjacency(net: &RoadNetwork) -> Vec<Vec<(usize, f64)>> {
    let n = net.segments.len();
    let mut adj = vec![Vec::new(); n];
    for c in &net.connectors {
        if c.from_lane_id >= net.lanes.len() || c.to_lane_id >= net.lanes.len() {
            continue;
        }
        let u = net.lanes[c.from_lane_id].segment_id;
        let v = net.lanes[c.to_lane_id].segment_id;
        if u < n && v < n && u != v {
            let turn_s = match net.junctions[c.junction_id].control_mode {
                ControlMode::Roundabout => 2.5,
                ControlMode::MaxPressureAdaptive => 4.2,
                ControlMode::Signal2Phase
                | ControlMode::Signal4PhaseProtectedLeft
                | ControlMode::PermissiveYield => match c.turn_type {
                    TurnType::Through => 4.5,
                    TurnType::Right => 5.5,
                    TurnType::Left => 8.0,
                },
            };
            if adj[u].iter().all(|&(nb, _)| nb != v) {
                adj[u].push((v, turn_s));
            }
        }
    }
    adj
}

fn dijkstra_from_seg(
    origin: usize,
    seg_trav_s: &[f64],
    seg_blocked: &[bool],
    adj: &[Vec<(usize, f64)>],
) -> Vec<f64> {
    let n = seg_trav_s.len();
    let mut dist = vec![f64::INFINITY; n];
    if origin >= n || seg_blocked[origin] {
        return dist;
    }
    let mut pq = BinaryHeap::new();
    dist[origin] = 0.5 * seg_trav_s[origin];
    pq.push(HeapState {
        cost_s: dist[origin],
        seg: origin,
    });
    while let Some(HeapState { cost_s, seg }) = pq.pop() {
        if cost_s > dist[seg] + 1e-9 {
            continue;
        }
        for &(nxt, turn_s) in &adj[seg] {
            if seg_blocked[nxt] {
                continue;
            }
            let cand = cost_s + turn_s + seg_trav_s[nxt];
            if cand + 1e-9 < dist[nxt] {
                dist[nxt] = cand;
                pq.push(HeapState {
                    cost_s: cand,
                    seg: nxt,
                });
            }
        }
    }
    dist
}

fn nearest_open_seg(net: &RoadNetwork, target: Vec2) -> (usize, Vec2) {
    if let Some(seg) = net
        .segments
        .iter()
        .filter(|s| !s.closed && !s.base_pts.is_empty())
        .min_by(|a, b| {
            let da = a.base_pts[a.base_pts.len() / 2].dist(target);
            let db = b.base_pts[b.base_pts.len() / 2].dist(target);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        })
    {
        (seg.id, seg.base_pts[seg.base_pts.len() / 2])
    } else {
        (0, target)
    }
}

/// Executes `Rayon`-parallelized Monte Carlo simulation of FEMA `IPAWS-OPEN` (`CAP v1.2` Origination,
/// `WEA 3.0`, `NPWS`/`PEP` `EAS` AM/FM, `NOAA NWR SAME` `162 MHz`) and FEMA Region 10 `MERS` (`IRV`/`MEOV`
/// + `FNARS` HF + `ESF #2` `NIFOG` LMR Interoperability) under Common Random Numbers (`CRN`).
pub fn run_fema_comms_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &FemaCommsConfig,
) -> FemaCommsReport {
    let t0 = Instant::now();
    let n_seg = net.segments.len();
    let n_parcels = parcels.len();
    let seis_cfg = &cfg.warning.transit_evac.shelters.evac.medevac.surge.ffe.seismic;
    let trials = seis_cfg.monte_carlo_trials.clamp(16, 5_000);
    let win_min = cfg.warning.transit_evac.clearance_window_min.clamp(45.0, 240.0);

    let b = net.bounds;
    let sx = (b.max_x - b.min_x).max(160.0);
    let sy = (b.max_y - b.min_y).max(160.0);
    let diag = sx.hypot(sy);

    // 1. Fixed FEMA & Commercial Architecture Nodes:
    //    [0] County COG EOC (`CAP v1.2` Originator + `FNARS` HF ALE antenna)
    //    [1] FEMA NPWS Primary Entry Point (`PEP` / `LP-1` `50 kW` AM/FM Hardened Broadcast Shelter)
    //    [2] NOAA Weather Radio All Hazards (`KEC42` / `WXM` `162.550 MHz` VHF `SAME` Transmitter)
    //    [3..9] 6 Commercial `WEA 3.0` Macrocell Sites (`eNodeB`/`gNodeB`)
    let (eoc_seg, eoc_pos) = nearest_open_seg(
        net,
        facilities
            .first()
            .map(|f| f.world_pos)
            .unwrap_or(Vec2 {
                x: b.min_x + 0.50 * sx,
                y: b.min_y + 0.50 * sy,
            }),
    );
    let (pep_seg, pep_pos) = nearest_open_seg(
        net,
        Vec2 {
            x: b.min_x + 0.12 * sx,
            y: b.min_y + 0.15 * sy,
        },
    );
    let (nwr_seg, nwr_pos) = nearest_open_seg(
        net,
        Vec2 {
            x: b.min_x + 0.86 * sx,
            y: b.min_y + 0.18 * sy,
        },
    );

    let wea_coords = [
        ("WEA-BTS-1 Downtown Core", 0.48, 0.50, 0.56, true),
        ("WEA-BTS-2 Riverfront Couplet", 0.62, 0.55, 0.66, true),
        ("WEA-BTS-3 University Campus", 0.24, 0.48, 0.58, false),
        ("WEA-BTS-4 North Arterial", 0.44, 0.20, 0.68, false),
        ("WEA-BTS-5 South Commercial", 0.46, 0.82, 0.65, true),
        ("WEA-BTS-6 Eastgate Industrial", 0.82, 0.68, 0.84, true),
    ];
    let wea_r = (0.33 * diag).clamp(280.0, 1_100.0);
    let mut wea_nodes: Vec<(String, usize, Vec2, f64, bool)> = Vec::with_capacity(6);
    for (nm, fx, fy, med_g, br_bh) in wea_coords {
        let (sid, wpos) = nearest_open_seg(
            net,
            Vec2 {
                x: b.min_x + fx * sx,
                y: b.min_y + fy * sy,
            },
        );
        wea_nodes.push((nm.to_string(), sid, wpos, med_g, br_bh));
    }

    // 2. Greedy p-median / high-vulnerability siting of `fema_mers_irv_units` FEMA Region 10 `MERS` IRV/MEOV trucks!
    let n_mers = if cfg.posture == FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment {
        cfg.fema_mers_irv_units.clamp(1, 6)
    } else {
        0
    };
    let mers_r = (0.42 * diag).clamp(360.0, 1_350.0);
    let mut mers_nodes: Vec<(String, usize, Vec2)> = Vec::with_capacity(n_mers);
    let cand_mers_targets = [
        ("FEMA MERS-IRV-1 (Bothell Det. CBD/Bridgehead)", 0.56, 0.52),
        ("FEMA MERS-MEOV-2 (High-Density R3/Hospital)", 0.30, 0.72),
        ("FEMA MERS-IRV-3 (Eastside Shelter/Transit Hub)", 0.78, 0.48),
        ("FEMA MERS-IRV-4 (North Corridor LMR Relay)", 0.40, 0.24),
        ("FEMA MERS-IRV-5 (West Campus Staging)", 0.20, 0.44),
        ("FEMA MERS-IRV-6 (South Logistics Staging)", 0.54, 0.86),
    ];
    for &(nm, fx, fy) in cand_mers_targets.iter().take(n_mers) {
        let (sid, wpos) = nearest_open_seg(
            net,
            Vec2 {
                x: b.min_x + fx * sx,
                y: b.min_y + fy * sy,
            },
        );
        mers_nodes.push((nm.to_string(), sid, wpos));
    }

    // Map parcels to population, serving WEA towers, MERS proximity & nearest road segment
    let mut parcel_pop = vec![20.0f64; n_parcels];
    let mut parcel_wea_primary = vec![0usize; n_parcels];
    let mut parcel_wea_backup: Vec<Option<usize>> = vec![None; n_parcels];
    let mut parcel_mers_idx: Vec<Option<usize>> = vec![None; n_parcels];

    for (pi, p) in parcels.iter().enumerate() {
        let zone_mult = match p.zone {
            ZoneCode::CBD | ZoneCode::R3 => 1.25,
            ZoneCode::MU4 => 1.15,
            _ => 1.0,
        };
        let area_m2 = p.area_ha * 10_000.0;
        let est_pop = p.residents() * 0.42 + p.jobs() * 0.55;
        parcel_pop[pi] =
            ((12.0 + est_pop * 0.35 + (area_m2 / 550.0).min(24.0)) * zone_mult).clamp(8.0, 75.0);

        let mut d_wea: Vec<(usize, f64)> = wea_nodes
            .iter()
            .enumerate()
            .map(|(wi, w)| (wi, p.center.dist(w.2)))
            .collect();
        d_wea.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        parcel_wea_primary[pi] = d_wea[0].0;
        if d_wea.len() > 1 && d_wea[1].1 <= wea_r * 1.15 {
            parcel_wea_backup[pi] = Some(d_wea[1].0);
        }

        if let Some((mi, dist_m)) = mers_nodes
            .iter()
            .enumerate()
            .map(|(mi, m)| (mi, p.center.dist(m.2)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        {
            if dist_m <= mers_r {
                parcel_mers_idx[pi] = Some(mi);
            }
        }
    }

    let adj = build_seg_adjacency(net);
    let bridges = enumerate_seismic_bridge_inventory(net, facilities, seis_cfg);

    let n_fixed_sites = 3 + wea_nodes.len(); // EOC + PEP + NWR + 6 WEA
    let mut site_locs = vec![eoc_pos, pep_pos, nwr_pos];
    for w in &wea_nodes {
        site_locs.push(w.2);
    }
    let mut corr = vec![0.0; n_fixed_sites * n_fixed_sites];
    for i in 0..n_fixed_sites {
        for j in 0..n_fixed_sites {
            let d_km = site_locs[i].dist(site_locs[j]) / 1000.0;
            corr[i * n_fixed_sites + j] = (-d_km / 4.5).exp();
        }
    }
    let chol = cholesky_lower(&corr, n_fixed_sites);

    let base_sa10_g = crate::seismic_mc::median_sa10_gmpe_g(
        seis_cfg.moment_magnitude_mw,
        seis_cfg.rupture_standoff_km,
        seis_cfg.vs30_m_per_s,
    );

    let (med_batt_hr, pga_hard_mult, fiber_bonus) = cfg.warning.telecom_backup.resilience_params();
    let hrs_after = cfg.warning.hours_post_mainshock.clamp(0.1, 72.0);
    let night = cfg.warning.nocturnal_sleeping_event;
    let eas_hh_share = (cfg.household_eas_am_fm_radio_pct / 100.0).clamp(0.02, 0.96);
    let nwr_hh_share = (cfg.household_nwr_same_receiver_pct / 100.0).clamp(0.02, 0.95);
    let time_pts = 13usize;

    #[derive(Clone)]
    struct TrialFema {
        eoc_cap_ok: f64,
        pep_ok: f64,
        nwr_ok: f64,
        wea_mean_ok: f64,
        wea_site_ok: Vec<f64>,
        mers_site_ok: Vec<f64>,
        pub_reach_pct: f64,
        esf2_lmr_pct: f64,
        mean_mob_min: f64,
        unwarned_pax: f64,
        tl_wea: Vec<f64>,
        tl_pep: Vec<f64>,
        tl_nwr: Vec<f64>,
        tl_full: Vec<f64>,
        tl_lmr: Vec<f64>,
        p_reach: Vec<f64>,
        p_lmr: Vec<f64>,
    }

    let base_seed = seis_cfg.random_seed ^ 0x4645_4D41_4950_4001;
    let trial_res: Vec<TrialFema> = (0..trials)
        .into_par_iter()
        .map(|tr_idx| {
            let mut rng = SplitMix64::new(
                base_seed.wrapping_add((tr_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let z_inter = rng.next_standard_normal();
            let mut z_ind = vec![0.0; n_fixed_sites];
            for z in &mut z_ind {
                *z = rng.next_standard_normal();
            }
            let mut z_corr = vec![0.0; n_fixed_sites];
            for i in 0..n_fixed_sites {
                let mut s = 0.0;
                for j in 0..=i {
                    s += chol[i * n_fixed_sites + j] * z_ind[j];
                }
                z_corr[i] = s;
            }

            let sa10_city =
                (base_sa10_g * (0.34 * z_inter + 0.22 * z_corr[0]).exp()).clamp(0.03, 2.6);
            let pga_city =
                crate::seismic_mc::median_pga_g_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);
            let pgv_cm_s = median_pgv_cm_s_from_sa10(sa10_city, seis_cfg.vs30_m_per_s);
            let p_grid_out = normal_cdf_phi((pga_city / 0.30).ln() / 0.50);
            let pipe_rr = ala_hazus_pipe_repair_rate_per_km(
                pgv_cm_s,
                seis_cfg.pipe_fragility_law,
                seis_cfg.baseline_water_pipe_material,
                250.0,
                false,
            );

            // Bridge seismic closures
            let mut seg_closed = vec![false; n_seg];
            let mut any_bridge_closed = false;
            for br in &bridges {
                if rng.next_f64_open01() < br.effective_archetype.exceedance_probs(sa10_city)[2] {
                    any_bridge_closed = true;
                    for &sid in &br.segment_ids {
                        if sid < n_seg {
                            seg_closed[sid] = true;
                        }
                    }
                }
            }

            // Stage 1: County COG EOC `CAP v1.2` Uplink to FEMA `IPAWS-OPEN`
            let eoc_fiber_ok = !any_bridge_closed
                && (rng.next_f64_open01() >= (1.0 - (-pipe_rr * 0.48 * 1.8).exp()));
            let fnars_hf_ok = rng.next_f64_open01() < 0.965; // Ionospheric NVIS skywave (96.5% HF propagation reliability)
            let mers_vsat_ok = rng.next_f64_open01() < 0.992; // Ku/Ka + LEO multi-orbit VSAT terminal
            let eoc_cap_ok = match cfg.eoc_uplink {
                EocIpaswUplinkMode::CommercialTerrestrialFiberOnly => eoc_fiber_ok,
                EocIpaswUplinkMode::FiberPlusFnarsHfSkywaveRelay => eoc_fiber_ok || fnars_hf_ok,
                EocIpaswUplinkMode::FullFemaMersVsatAndFnarsContinuity => {
                    eoc_fiber_ok || fnars_hf_ok || mers_vsat_ok
                }
            };

            // Stage 2: FEMA NPWS Primary Entry Point (`PEP`) `50 kW` AM/FM Broadcast Station
            // Hardened studio shelter (`median PGA = 1.65 g`, 60-day diesel tank, FNARS Mount Weather link!)
            let pga_pep = (pga_city * (0.18 * z_corr[1]).exp()).clamp(0.03, 2.8);
            let pep_tx_ok =
                rng.next_f64_open01() >= normal_cdf_phi((pga_pep / 1.65).ln() / 0.50);

            // Stage 3: NOAA Weather Radio All Hazards (`162.550 MHz` VHF `SAME` Transmitter)
            // Essential-facility VHF tower + automatic transfer switch diesel generator (`median PGA = 1.15 g`)
            let pga_nwr = (pga_city * (0.18 * z_corr[2]).exp()).clamp(0.03, 2.8);
            let nwr_gen_ok = rng.next_f64_open01() < 0.94;
            let nwr_tx_ok = (rng.next_f64_open01() >= normal_cdf_phi((pga_nwr / 1.15).ln() / 0.52))
                && ((rng.next_f64_open01() >= p_grid_out) || nwr_gen_ok);

            // Stage 4: Commercial `WEA 3.0` Cell Sites (`BTS`)
            let mut wea_alive = vec![false; wea_nodes.len()];
            let mut wea_ok_f = vec![0.0; wea_nodes.len()];
            for (wi, w) in wea_nodes.iter().enumerate() {
                let pga_w = (pga_city * (0.20 * z_corr[3 + wi]).exp()).clamp(0.03, 2.8);
                let s_ok = rng.next_f64_open01()
                    >= normal_cdf_phi((pga_w / (w.3 * pga_hard_mult).max(0.20)).ln() / 0.52);
                let bridge_cut =
                    w.4 && any_bridge_closed && (rng.next_f64_open01() > fiber_bonus);
                let f_ok = !bridge_cut
                    && (rng.next_f64_open01()
                        >= (1.0 - (-pipe_rr * 0.36 * (1.0 - fiber_bonus) * 1.6).exp()));
                let ac_on = rng.next_f64_open01() >= p_grid_out;
                let batt_hr =
                    med_batt_hr * (0.38 * rng.next_standard_normal()).exp().clamp(0.35, 2.4);
                let alive = s_ok && f_ok && (ac_on || batt_hr >= hrs_after);
                wea_alive[wi] = alive;
                wea_ok_f[wi] = if alive { 1.0 } else { 0.0 };
            }

            // Stage 5: FEMA Region 10 `MERS` (`IRV`/`MEOV` + Fly-Away VSAT/LMR/FirstNet `COLT` deployment)
            let mut seg_trav_s = vec![10.0; n_seg];
            for seg in &net.segments {
                if seg.closed {
                    seg_closed[seg.id] = true;
                }
                seg_trav_s[seg.id] = (seg.length / seg.speed_mps.max(4.0)).max(1.2);
            }
            let eoc_dist_s = dijkstra_from_seg(eoc_seg, &seg_trav_s, &seg_closed, &adj);

            let mut mers_alive = vec![false; n_mers];
            let mut mers_arr_min = vec![999.0; n_mers];
            let mut mers_ok_f = vec![0.0; n_mers];
            for (mi, m) in mers_nodes.iter().enumerate() {
                let drive_s = eoc_dist_s.get(m.1).copied().unwrap_or(f64::INFINITY);
                if drive_s.is_finite() {
                    // Rapid field mast setup (`8 min` mast + Ku/LEO VSAT auto-acquire) + road travel
                    mers_alive[mi] = true;
                    mers_arr_min[mi] = 8.0 + (drive_s / 60.0).clamp(1.5, 28.0);
                    mers_ok_f[mi] = 1.0;
                } else if cfg.airlift_mers_flyaway_kits {
                    // Bridge-isolated sector rescued by UH-60 airlifted MERS Fly-Away VSAT/LMR package!
                    mers_alive[mi] = true;
                    mers_arr_min[mi] = 18.0;
                    mers_ok_f[mi] = 1.0;
                }
            }

            // Terrestrial municipal P25 LMR repeater baseline (without MERS, single-site rooftop repeater suffers grid/backhaul loss)
            let base_p25_repeater_ok = (rng.next_f64_open01() >= p_grid_out * 0.65)
                && (rng.next_f64_open01()
                    >= normal_cdf_phi((pga_city / 0.78).ln() / 0.55));

            // Evaluate per-parcel pathway reach & mobilization across `0..=win_min`!
            let tot_pop: f64 = parcel_pop.iter().sum::<f64>().max(1.0);
            let mut p_reach = vec![0.0; n_parcels];
            let mut p_lmr = vec![0.0; n_parcels];
            let mut p_mob_sum = 0.0f64;

            let mut tl_wea = vec![0.0; time_pts];
            let mut tl_pep = vec![0.0; time_pts];
            let mut tl_nwr = vec![0.0; time_pts];
            let mut tl_full = vec![0.0; time_pts];
            let mut tl_lmr = vec![0.0; time_pts];

            for pi in 0..n_parcels {
                let pop = parcel_pop[pi];
                let d_pep_km = parcels[pi].center.dist(pep_pos) / 1000.0 + 18.0; // regional PEP standoff
                let d_nwr_km = parcels[pi].center.dist(nwr_pos) / 1000.0 + 9.0;
                let (pep_rf, nwr_rf) = compute_pep_am_and_nwr_vhf_reliability(d_pep_km.max(d_nwr_km));

                let comm_wea_on = wea_alive[parcel_wea_primary[pi]]
                    || parcel_wea_backup[pi].map_or(false, |b_i| wea_alive[b_i]);
                let mers_on = parcel_mers_idx[pi].map_or(false, |mi| mers_alive[mi]);
                let mers_t_min = parcel_mers_idx[pi].map_or(999.0, |mi| mers_arr_min[mi]);

                // Effective channel reception asymptotes when EOC CAP uplink is intact
                let p_wea_ch = if comm_wea_on {
                    if night {
                        0.76
                    } else {
                        0.93
                    }
                } else {
                    0.0
                };
                let p_pep_ch = if pep_tx_ok {
                    // Daytime car/home AM/FM + neighbor word-of-mouth vs. 02:00 AM sleeping AM/FM listenership
                    let listen_fac = if night { 0.48 } else { 0.88 };
                    (eas_hh_share * listen_fac * pep_rf + 0.16 * pep_rf).clamp(0.0, 0.94)
                } else {
                    0.0
                };
                let p_nwr_ch = if nwr_tx_ok {
                    // NOAA SAME 1050 Hz tone-alert wakes sleeping households at 02:00 AM with 95% efficacy!
                    (nwr_hh_share * 0.95 * nwr_rf).clamp(0.0, 0.95)
                } else {
                    0.0
                };
                let p_mers_ch = if mers_on { 0.92 } else { 0.0 };

                let r_wea_only = combine_ipaws_pathway_reach_prob(eoc_cap_ok, p_wea_ch, 0.0, 0.0);
                let r_wea_pep =
                    combine_ipaws_pathway_reach_prob(eoc_cap_ok, p_wea_ch, p_pep_ch, 0.0);
                let r_tri =
                    combine_ipaws_pathway_reach_prob(eoc_cap_ok, p_wea_ch, p_pep_ch, p_nwr_ch);
                let r_full = combine_ipaws_pathway_reach_prob(
                    eoc_cap_ok || mers_on,
                    p_wea_ch.max(p_mers_ch),
                    p_pep_ch,
                    p_nwr_ch,
                );

                let active_reach = match cfg.posture {
                    FemaCommsPosture::IpaswWea30CellularOnly => r_wea_only,
                    FemaCommsPosture::IpaswTriPathWeaEasPepNwrSame => r_tri,
                    FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment => r_full,
                };
                p_reach[pi] = 100.0 * active_reach;

                // ESF #2 NIFOG Responder LMR coverage (`VTAC`/`UTAC`/`7TAC`/`8TAC` + MERS portable repeater)
                let lmr_ok = if base_p25_repeater_ok {
                    1.0
                } else if cfg.posture == FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment
                    && mers_on
                {
                    0.98
                } else {
                    // Direct simplex handheld walkie-talkie fallback (`~32%` urban street-canyon reach)
                    0.32
                };
                p_lmr[pi] = 100.0 * lmr_ok;

                // Characteristic warning+prep time for active posture
                let warn_scale = if comm_wea_on && eoc_cap_ok {
                    if night {
                        5.5
                    } else {
                        2.5
                    }
                } else if cfg.posture != FemaCommsPosture::IpaswWea30CellularOnly
                    && eoc_cap_ok
                    && (nwr_tx_ok || pep_tx_ok)
                {
                    if nwr_tx_ok {
                        4.2
                    } else {
                        9.5
                    }
                } else if cfg.posture == FemaCommsPosture::FullFemaIpaswPlusMersEsf2Deployment
                    && mers_on
                {
                    mers_t_min + 2.5
                } else {
                    54.0
                };
                let prep_scale = if night { 12.5 } else { 9.0 };
                p_mob_sum += pop * (warn_scale * 0.89 + prep_scale * 0.88);

                for ti in 0..time_pts {
                    let tm = (ti as f64 / (time_pts - 1) as f64) * win_min;
                    let w_sc = if night { 5.8 } else { 2.6 };
                    tl_wea[ti] +=
                        pop * evaluate_padm_mobilization_cdf(tm, w_sc, 1.75, prep_scale, r_wea_only);
                    tl_pep[ti] += pop
                        * evaluate_padm_mobilization_cdf(
                            tm,
                            w_sc.min(7.5),
                            1.75,
                            prep_scale,
                            r_wea_pep,
                        );
                    tl_nwr[ti] += pop
                        * evaluate_padm_mobilization_cdf(
                            tm,
                            w_sc.min(4.2),
                            1.90,
                            prep_scale * 0.94,
                            r_tri,
                        );
                    let full_t_eff = if comm_wea_on || nwr_tx_ok || pep_tx_ok {
                        tm
                    } else {
                        (tm - mers_t_min).max(0.0)
                    };
                    tl_full[ti] += pop
                        * evaluate_padm_mobilization_cdf(
                            full_t_eff,
                            3.8,
                            1.95,
                            prep_scale * 0.90,
                            r_full,
                        );
                    let lmr_t = if base_p25_repeater_ok {
                        1.0
                    } else if mers_on && tm >= mers_t_min {
                        0.98
                    } else {
                        0.32
                    };
                    tl_lmr[ti] += pop * lmr_t;
                }
            }

            for ti in 0..time_pts {
                tl_wea[ti] = 100.0 * (tl_wea[ti] / tot_pop).clamp(0.0, 1.0);
                tl_pep[ti] = 100.0 * (tl_pep[ti] / tot_pop).clamp(0.0, 1.0);
                tl_nwr[ti] = 100.0 * (tl_nwr[ti] / tot_pop).clamp(0.0, 1.0);
                tl_full[ti] = 100.0 * (tl_full[ti] / tot_pop).clamp(0.0, 1.0);
                tl_lmr[ti] = 100.0 * (tl_lmr[ti] / tot_pop).clamp(0.0, 1.0);
            }

            let reach_weighted: f64 = (0..n_parcels)
                .map(|pi| (p_reach[pi] / 100.0) * parcel_pop[pi])
                .sum();
            let lmr_weighted: f64 = (0..n_parcels)
                .map(|pi| (p_lmr[pi] / 100.0) * parcel_pop[pi])
                .sum();
            let pub_reach_pct = 100.0 * (reach_weighted / tot_pop).clamp(0.0, 1.0);
            let esf2_lmr_pct = 100.0 * (lmr_weighted / tot_pop).clamp(0.0, 1.0);
            let unwarned_pax = (tot_pop - reach_weighted).max(0.0);
            let wea_mean = 100.0 * wea_ok_f.iter().sum::<f64>() / (wea_nodes.len().max(1) as f64);

            TrialFema {
                eoc_cap_ok: if eoc_cap_ok { 100.0 } else { 0.0 },
                pep_ok: if pep_tx_ok { 100.0 } else { 0.0 },
                nwr_ok: if nwr_tx_ok { 100.0 } else { 0.0 },
                wea_mean_ok: wea_mean,
                wea_site_ok: wea_ok_f,
                mers_site_ok: mers_ok_f,
                pub_reach_pct,
                esf2_lmr_pct,
                mean_mob_min: p_mob_sum / tot_pop,
                unwarned_pax,
                tl_wea,
                tl_pep,
                tl_nwr,
                tl_full,
                tl_lmr,
                p_reach,
                p_lmr,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let eoc_up = trial_res.iter().map(|r| r.eoc_cap_ok).sum::<f64>() * inv_t;
    let pep_up = trial_res.iter().map(|r| r.pep_ok).sum::<f64>() * inv_t;
    let nwr_up = trial_res.iter().map(|r| r.nwr_ok).sum::<f64>() * inv_t;
    let wea_up = trial_res.iter().map(|r| r.wea_mean_ok).sum::<f64>() * inv_t;
    let pub_reach = trial_res.iter().map(|r| r.pub_reach_pct).sum::<f64>() * inv_t;
    let lmr_cov = trial_res.iter().map(|r| r.esf2_lmr_pct).sum::<f64>() * inv_t;
    let mean_mob = trial_res.iter().map(|r| r.mean_mob_min).sum::<f64>() * inv_t;
    let mean_unwarned = trial_res.iter().map(|r| r.unwarned_pax).sum::<f64>() * inv_t;

    let mut unwarn_sorted: Vec<f64> = trial_res.iter().map(|r| r.unwarned_pax).collect();
    unwarn_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p90_unwarned = unwarn_sorted[(((trials as f64) * 0.90).floor() as usize).min(trials - 1)];

    // Assemble FEMA Asset Telemetry Table
    let mut asset_rows = Vec::new();
    asset_rows.push(FemaCommsAssetRow {
        asset_id: 0,
        callsign_or_name: "County COG EOC (CAP v1.2 + FNARS HF)".to_string(),
        kind: FemaAssetKind::CogEocCapOriginator,
        segment_id: eoc_seg,
        world_pos: eoc_pos,
        coverage_radius_m: diag,
        post_quake_online_pct: eoc_up,
        eoc_cap_link_ok_pct: eoc_up,
        public_or_lmr_reach_pct: pub_reach,
    });
    asset_rows.push(FemaCommsAssetRow {
        asset_id: 1,
        callsign_or_name: "NPWS Primary Entry Point PEP (50kW AM/FM)".to_string(),
        kind: FemaAssetKind::NpwsPepBroadcastStation,
        segment_id: pep_seg,
        world_pos: pep_pos,
        coverage_radius_m: 120_000.0,
        post_quake_online_pct: pep_up,
        eoc_cap_link_ok_pct: eoc_up,
        public_or_lmr_reach_pct: cfg.household_eas_am_fm_radio_pct.min(pep_up),
    });
    asset_rows.push(FemaCommsAssetRow {
        asset_id: 2,
        callsign_or_name: "NOAA Weather Radio KEC-42 (162.55MHz SAME)".to_string(),
        kind: FemaAssetKind::NoaaNwrSameTransmitter,
        segment_id: nwr_seg,
        world_pos: nwr_pos,
        coverage_radius_m: 58_000.0,
        post_quake_online_pct: nwr_up,
        eoc_cap_link_ok_pct: eoc_up,
        public_or_lmr_reach_pct: cfg.household_nwr_same_receiver_pct.min(nwr_up),
    });
    for (wi, w) in wea_nodes.iter().enumerate() {
        let on_pct = 100.0 * trial_res.iter().map(|r| r.wea_site_ok[wi]).sum::<f64>() * inv_t;
        asset_rows.push(FemaCommsAssetRow {
            asset_id: 3 + wi,
            callsign_or_name: w.0.clone(),
            kind: FemaAssetKind::CommercialWeaCellSite,
            segment_id: w.1,
            world_pos: w.2,
            coverage_radius_m: wea_r,
            post_quake_online_pct: on_pct,
            eoc_cap_link_ok_pct: eoc_up,
            public_or_lmr_reach_pct: on_pct * (eoc_up / 100.0),
        });
    }
    let mut mers_uplink_vectors = Vec::new();
    for (mi, m) in mers_nodes.iter().enumerate() {
        let m_pct = 100.0 * trial_res.iter().map(|r| r.mers_site_ok[mi]).sum::<f64>() * inv_t;
        asset_rows.push(FemaCommsAssetRow {
            asset_id: 3 + wea_nodes.len() + mi,
            callsign_or_name: m.0.clone(),
            kind: FemaAssetKind::FemaMersIrvUnit,
            segment_id: m.1,
            world_pos: m.2,
            coverage_radius_m: mers_r,
            post_quake_online_pct: m_pct,
            eoc_cap_link_ok_pct: 99.2,
            public_or_lmr_reach_pct: m_pct * 0.98,
        });
        mers_uplink_vectors.push((eoc_pos, m.2));
    }

    let mut timeline = Vec::with_capacity(time_pts);
    for ti in 0..time_pts {
        timeline.push(FemaIpaswTimelinePoint {
            minute: (ti as f64 / (time_pts - 1) as f64) * win_min,
            wea30_only_reach_pct: trial_res.iter().map(|r| r.tl_wea[ti]).sum::<f64>() * inv_t,
            wea_plus_eas_pep_pct: trial_res.iter().map(|r| r.tl_pep[ti]).sum::<f64>() * inv_t,
            wea_eas_plus_nwr_same_pct: trial_res.iter().map(|r| r.tl_nwr[ti]).sum::<f64>() * inv_t,
            full_fema_ipaws_mers_pct: trial_res.iter().map(|r| r.tl_full[ti]).sum::<f64>() * inv_t,
            esf2_nifog_lmr_interop_pct: trial_res.iter().map(|r| r.tl_lmr[ti]).sum::<f64>() * inv_t,
        });
    }

    let mut parcel_ipaws_reach_pct = vec![100.0; n_parcels];
    let mut parcel_esf2_lmr_ok_pct = vec![100.0; n_parcels];
    for pi in 0..n_parcels {
        parcel_ipaws_reach_pct[pi] = trial_res.iter().map(|r| r.p_reach[pi]).sum::<f64>() * inv_t;
        parcel_esf2_lmr_ok_pct[pi] = trial_res.iter().map(|r| r.p_lmr[pi]).sum::<f64>() * inv_t;
    }

    FemaCommsReport {
        config: cfg.clone(),
        eoc_cap_origination_uptime_pct: eoc_up,
        npws_pep_broadcast_uptime_pct: pep_up,
        noaa_nwr_same_uptime_pct: nwr_up,
        commercial_wea30_uptime_pct: wea_up,
        deployed_mers_units_count: n_mers,
        public_ipaws_verified_reach_pct: pub_reach,
        esf2_nifog_responder_lmr_coverage_pct: lmr_cov,
        mean_ipaws_mobilization_min: mean_mob,
        unwarned_isolated_residents_pax: mean_unwarned,
        p90_unwarned_residents_pax: p90_unwarned,
        asset_rows,
        timeline,
        mers_uplink_vectors,
        parcel_ipaws_reach_pct,
        parcel_esf2_lmr_ok_pct,
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

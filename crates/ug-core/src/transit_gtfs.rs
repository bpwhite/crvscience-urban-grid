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

//! Fixed-route GTFS Schedule ingestion (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`),
//! procedural arterial route/stop synthesis, curbside stop passenger queues, TRB TCQSM boarding &
//! alighting dwell-time physics, anti-bunching headway holding, and live ridership crowding telemetry.

use crate::network::{sample_polyline_fast, RoadNetwork, Vec2, LANE_WIDTH_M};
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GtfsError {
    #[error("I/O error reading GTFS directory: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid or empty GTFS feed: {0}")]
    Feed(String),
}

/// Fare-collection & door-channel configuration governing passenger dwell times per
/// TRB Transit Capacity and Quality of Service Manual (TCQSM, TCRP Report 165 Ch. 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoardingFareMode {
    /// Fare-free all-door boarding (e.g. Metro Transit System / full BRT) — fastest dwell.
    AllDoorFareFreeBrt,
    /// Dual-channel smartcard / contactless tap boarding.
    SmartcardTap,
    /// Single-stream front-door farebox / cash boarding — highest dwell & bunching sensitivity.
    FrontDoorCash,
}

impl BoardingFareMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::AllDoorFareFreeBrt => "All-Door Fare-Free / BRT",
            Self::SmartcardTap => "Contactless Smartcard (2-Door)",
            Self::FrontDoorCash => "Front-Door Farebox / Cash",
        }
    }

    /// Returns `(door_clearance_s, sec_per_boarding_pax, sec_per_alighting_pax, simultaneous_doors)`.
    pub fn tcqsm_timings(self) -> (f64, f64, f64, bool) {
        match self {
            Self::AllDoorFareFreeBrt => (2.4, 1.75, 1.20, true),
            Self::SmartcardTap => (3.1, 2.65, 1.50, true),
            Self::FrontDoorCash => (3.8, 4.20, 1.85, false),
        }
    }
}

/// Transit coach physical dimensions and seated vs. standee crush passenger capacities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusCoachSpec {
    /// 40-ft (12.2 m) low-floor urban transit bus: 38 seated, 60 crush capacity.
    Standard40Ft,
    /// 60-ft (18.3 m) articulated BRT coach: 58 seated, 95 crush capacity.
    Articulated60FtBrt,
}

impl BusCoachSpec {
    pub fn label(self) -> &'static str {
        match self {
            Self::Standard40Ft => "40-ft Low-Floor (38 seat / 60 crush)",
            Self::Articulated60FtBrt => "60-ft Articulated BRT (58 seat / 95 crush)",
        }
    }

    pub fn dimensions_and_capacity(self) -> (f64, f64, u16, u16) {
        match self {
            Self::Standard40Ft => (12.2, 2.55, 38, 60),
            Self::Articulated60FtBrt => (18.3, 2.60, 58, 95),
        }
    }
}

/// A curbside or BRT platform transit stop snapped to a specific `RoadSegment` and longitudinal `s_m`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitStop {
    pub id: usize,
    pub gtfs_stop_id: String,
    pub name: String,
    pub segment_id: usize,
    pub s_m: f64,
    pub world_pos: Vec2,
    pub heading_rad: f64,
    /// Base hourly passenger production weight derived from nearby parcels' residents.
    pub base_origin_pax_hr: f64,
    /// Attraction weight derived from nearby parcels' commercial/civic jobs.
    pub attraction_weight: f64,
    /// Continuous + discrete passenger queue waiting on the curb platform.
    pub waiting_pax: f64,
    pub total_boardings: u64,
    pub total_alightings: u64,
    pub denied_passups: u64,
    pub last_bus_departure_s: Option<f64>,
}

/// A fixed-route transit pattern compiled into a connected chain of `RoadSegment`s and ordered stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitRoute {
    pub id: usize,
    pub gtfs_route_id: String,
    pub short_name: String,
    pub long_name: String,
    pub color_rgb: [u8; 3],
    pub segment_chain: Vec<usize>,
    /// `(stop_id, index_in_segment_chain)` ordered along the trip from origin to terminal.
    pub route_stops: Vec<(usize, usize)>,
    /// Explicit GTFS trip departure offsets (seconds from `t = 0`), supplemented by periodic headway dispatch.
    pub explicit_departures_s: Vec<f64>,
    pub next_explicit_idx: usize,
    pub next_periodic_dispatch_s: f64,
    pub active_runs: usize,
}

/// Aggregated real-time GTFS transit service, dwell, crowding, and reliability snapshot.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TransitTelemetrySnapshot {
    pub route_count: usize,
    pub stop_count: usize,
    pub active_scheduled_buses: usize,
    pub waiting_passengers: u32,
    pub onboard_passengers: u32,
    pub cumulative_boardings: u64,
    pub cumulative_alightings: u64,
    pub denied_passups: u64,
    pub mean_dwell_s: f64,
    pub mean_load_factor_pct: f64,
    pub peak_load_factor_pct: f64,
    pub bunching_pct: f64,
    pub tcqsm_crowding_los: char,
}

/// Complete runtime GTFS schedule, curbside stop state, and dwell/crowding manager inside `SimulationWorld`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitGtfsSystem {
    pub stops: Vec<TransitStop>,
    pub routes: Vec<TransitRoute>,
    pub fare_mode: BoardingFareMode,
    pub coach_spec: BusCoachSpec,
    pub hold_early_buses: bool,
    pub telemetry: TransitTelemetrySnapshot,
    dwell_sum_s: f64,
    dwell_events: u64,
}

#[derive(Debug, Clone)]
pub struct PendingBusDispatch {
    pub route_idx: usize,
    pub start_segment_id: usize,
    pub length_m: f64,
    pub width_m: f64,
    pub crush_capacity: u16,
    pub initial_pax: u16,
}

#[derive(Debug, Clone)]
pub struct DwellOutcome {
    pub dwell_s: f32,
    pub boarded: u16,
    pub alighted: u16,
    pub new_onboard: u16,
}

fn split_csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            fields.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    fields.push(cur.trim().to_string());
    fields
}

pub fn parse_gtfs_csv_table(csv_text: &str) -> Vec<HashMap<String, String>> {
    let mut lines = csv_text
        .lines()
        .map(|l| l.trim_end_matches('\r').trim_start_matches('\u{feff}'))
        .filter(|l| !l.trim().is_empty());
    let Some(header_line) = lines.next() else {
        return Vec::new();
    };
    let headers = split_csv_row(header_line);
    let mut rows = Vec::new();
    for line in lines {
        let vals = split_csv_row(line);
        let mut map = HashMap::with_capacity(headers.len());
        for (i, h) in headers.iter().enumerate() {
            if let Some(v) = vals.get(i) {
                map.insert(h.clone(), v.clone());
            }
        }
        rows.push(map);
    }
    rows
}

fn parse_gtfs_hms_seconds(hms: &str) -> Option<f64> {
    let parts: Vec<&str> = hms.trim().split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let s: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

fn parse_hex_rgb(hex: &str, fallback: [u8; 3]) -> [u8; 3] {
    let clean = hex.trim().trim_start_matches('#');
    if clean.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&clean[0..2], 16),
            u8::from_str_radix(&clean[2..4], 16),
            u8::from_str_radix(&clean[4..6], 16),
        ) {
            return [r, g, b];
        }
    }
    fallback
}

/// Projects a metric point `(wx, wy)` onto the closest open `RoadSegment`, returning `(seg_id, clamped_s_m, curb_pos, heading_rad)`.
fn snap_metric_to_curb(net: &RoadNetwork, wx: f64, wy: f64) -> Option<(usize, f64, Vec2, f64)> {
    let mut best: Option<(f64, usize, f64)> = None;
    for seg in &net.segments {
        if seg.closed || seg.length < 14.0 || seg.base_pts.len() < 2 {
            continue;
        }
        let mut cum = 0.0;
        for w in seg.base_pts.windows(2) {
            let dx = w[1].x - w[0].x;
            let dy = w[1].y - w[0].y;
            let l2 = dx * dx + dy * dy;
            let seg_l = l2.sqrt().max(1e-6);
            let t = if l2 > 1e-9 {
                (((wx - w[0].x) * dx + (wy - w[0].y) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let px = w[0].x + dx * t;
            let py = w[0].y + dy * t;
            let d = (wx - px).hypot(wy - py);
            let s_here = cum + t * seg_l;
            if best.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                best = Some((d, seg.id, s_here));
            }
            cum += seg_l;
        }
    }
    let (_, sid, raw_s) = best?;
    let seg = &net.segments[sid];
    let margin = (seg.length * 0.18).clamp(4.0, 9.5);
    let clamped_s = raw_s.clamp(margin, (seg.length - margin).max(margin));
    let (p_center, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, clamped_s);
    let curb_off = if seg.is_oneway {
        (seg.lane_count as f64 * 0.5) * LANE_WIDTH_M + 1.85
    } else {
        (seg.lane_count as f64) * LANE_WIDTH_M + 1.85
    };
    let nx = -ang.sin();
    let ny = ang.cos();
    let curb_pos = Vec2 {
        x: p_center.x + nx * curb_off,
        y: p_center.y + ny * curb_off,
    };
    Some((sid, clamped_s, curb_pos, ang))
}

fn compute_stop_parcel_weights(world_pos: Vec2, parcels: &[Parcel]) -> (f64, f64) {
    if parcels.is_empty() {
        return (18.0, 1.0);
    }
    let mut res_acc = 0.0;
    let mut job_acc = 0.0;
    for p in parcels {
        let d = world_pos.dist(p.center);
        // Exponential walking catchment kernel (~350 m transit stop shed)
        let w = (-d / 240.0).exp();
        res_acc += p.residents() * w;
        job_acc += p.jobs() * w;
    }
    let base_origin_hr = (res_acc * 0.085 + job_acc * 0.035).clamp(8.0, 240.0);
    let attr_w = (0.35 + job_acc * 0.015 + res_acc * 0.004).clamp(0.25, 8.0);
    (base_origin_hr, attr_w)
}

/// Shortest connected `RoadSegment` path between two segments using turn connectors.
pub fn shortest_segment_chain(net: &RoadNetwork, start_seg: usize, end_seg: usize) -> Vec<usize> {
    if start_seg == end_seg {
        return vec![start_seg];
    }
    #[derive(Copy, Clone, PartialEq)]
    struct Node {
        cost: f64,
        seg: usize,
    }
    impl Eq for Node {}
    impl Ord for Node {
        fn cmp(&self, other: &Self) -> Ordering {
            other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = net.segments.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();

    dist[start_seg] = 0.0;
    heap.push(Node {
        cost: 0.0,
        seg: start_seg,
    });

    while let Some(Node { cost, seg: u }) = heap.pop() {
        if u == end_seg {
            break;
        }
        if cost > dist[u] {
            continue;
        }
        let mut seen = HashSet::new();
        for &lid in &net.segments[u].lane_ids {
            for &cid in &net.lanes[lid].out_connectors {
                let v = net.connectors[cid].to_seg_id;
                if net.segments[v].closed || !seen.insert(v) {
                    continue;
                }
                let step_cost = net.segments[v].length / net.segments[v].speed_mps.max(3.0) + 1.5;
                let nd = cost + step_cost;
                if nd < dist[v] {
                    dist[v] = nd;
                    prev[v] = u;
                    heap.push(Node { cost: nd, seg: v });
                }
            }
        }
    }

    if !dist[end_seg].is_finite() {
        return Vec::new();
    }
    let mut path = Vec::new();
    let mut cur = end_seg;
    while cur != usize::MAX {
        path.push(cur);
        if cur == start_seg {
            break;
        }
        cur = prev[cur];
    }
    path.reverse();
    path
}

impl TransitGtfsSystem {
    /// Loads a standard unzipped GTFS Schedule directory (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`)
    /// and compiles its stops and fixed-route trips directly onto `net`.
    pub fn load_gtfs_dir(
        dir: &Path,
        net: &RoadNetwork,
        parcels: &[Parcel],
        headway_min: f64,
    ) -> Result<Self, GtfsError> {
        let stops_txt = fs::read_to_string(dir.join("stops.txt"))?;
        let routes_txt = fs::read_to_string(dir.join("routes.txt"))?;
        let trips_txt = fs::read_to_string(dir.join("trips.txt"))?;
        let stop_times_txt = fs::read_to_string(dir.join("stop_times.txt"))?;
        Self::from_gtfs_csv_strs(
            &stops_txt,
            &routes_txt,
            &trips_txt,
            &stop_times_txt,
            net,
            parcels,
            headway_min,
        )
    }

    pub fn from_gtfs_csv_strs(
        stops_csv: &str,
        routes_csv: &str,
        trips_csv: &str,
        stop_times_csv: &str,
        net: &RoadNetwork,
        parcels: &[Parcel],
        headway_min: f64,
    ) -> Result<Self, GtfsError> {
        let raw_stops = parse_gtfs_csv_table(stops_csv);
        let raw_routes = parse_gtfs_csv_table(routes_csv);
        let raw_trips = parse_gtfs_csv_table(trips_csv);
        let raw_st = parse_gtfs_csv_table(stop_times_csv);

        let mut stops = Vec::new();
        let mut stop_by_gtfs: HashMap<String, usize> = HashMap::new();
        for row in &raw_stops {
            let Some(sid_str) = row.get("stop_id") else {
                continue;
            };
            let lat: f64 = row
                .get("stop_lat")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let lon: f64 = row
                .get("stop_lon")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let (wx, wy) = net.projector.project(lat, lon);
            if let Some((seg_id, s_m, world_pos, heading_rad)) = snap_metric_to_curb(net, wx, wy) {
                let id = stops.len();
                let (base_origin_pax_hr, attraction_weight) =
                    compute_stop_parcel_weights(world_pos, parcels);
                let name = row
                    .get("stop_name")
                    .cloned()
                    .unwrap_or_else(|| sid_str.clone());
                stops.push(TransitStop {
                    id,
                    gtfs_stop_id: sid_str.clone(),
                    name,
                    segment_id: seg_id,
                    s_m,
                    world_pos,
                    heading_rad,
                    base_origin_pax_hr,
                    attraction_weight,
                    waiting_pax: (base_origin_pax_hr * 0.05).clamp(1.0, 9.0),
                    total_boardings: 0,
                    total_alightings: 0,
                    denied_passups: 0,
                    last_bus_departure_s: None,
                });
                stop_by_gtfs.insert(sid_str.clone(), id);
            }
        }

        if stops.is_empty() {
            return Err(GtfsError::Feed("No GTFS stops could be snapped to network.".into()));
        }

        // Group stop_times by trip_id sorted by stop_sequence
        let mut st_by_trip: HashMap<String, Vec<(u32, usize, Option<f64>)>> = HashMap::new();
        for row in &raw_st {
            let (Some(tid), Some(sid_str)) = (row.get("trip_id"), row.get("stop_id")) else {
                continue;
            };
            let Some(&stop_idx) = stop_by_gtfs.get(sid_str) else {
                continue;
            };
            let seq: u32 = row
                .get("stop_sequence")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let dep_s = row
                .get("departure_time")
                .or_else(|| row.get("arrival_time"))
                .and_then(|t| parse_gtfs_hms_seconds(t));
            st_by_trip.entry(tid.clone()).or_default().push((seq, stop_idx, dep_s));
        }
        for list in st_by_trip.values_mut() {
            list.sort_by_key(|(seq, _, _)| *seq);
        }

        let mut route_meta: HashMap<String, (String, String, [u8; 3])> = HashMap::new();
        let palette = [
            [239, 68, 68],
            [56, 189, 248],
            [16, 185, 129],
            [251, 191, 36],
            [192, 132, 252],
        ];
        for (ri, r) in raw_routes.iter().enumerate() {
            if let Some(rid) = r.get("route_id") {
                let short = r
                    .get("route_short_name")
                    .cloned()
                    .unwrap_or_else(|| format!("R{}", ri + 1));
                let long = r
                    .get("route_long_name")
                    .cloned()
                    .unwrap_or_else(|| short.clone());
                let col = r
                    .get("route_color")
                    .map(|c| parse_hex_rgb(c, palette[ri % palette.len()]))
                    .unwrap_or(palette[ri % palette.len()]);
                route_meta.insert(rid.clone(), (short, long, col));
            }
        }

        // Group trips by (route_id, direction_id) so each directional pattern becomes a scheduled TransitRoute
        let mut grouped_trips: HashMap<(String, String), Vec<(String, Vec<(u32, usize, Option<f64>)>)>> =
            HashMap::new();
        for tr in &raw_trips {
            let (Some(rid), Some(tid)) = (tr.get("route_id"), tr.get("trip_id")) else {
                continue;
            };
            let dir_id = tr.get("direction_id").cloned().unwrap_or_else(|| "0".into());
            if let Some(st_list) = st_by_trip.get(tid) {
                if !st_list.is_empty() {
                    grouped_trips
                        .entry((rid.clone(), dir_id))
                        .or_default()
                        .push((tid.clone(), st_list.clone()));
                }
            }
        }

        let mut min_t0 = f64::INFINITY;
        for trips in grouped_trips.values() {
            for (_, st_list) in trips {
                if let Some(t0) = st_list.first().and_then(|x| x.2) {
                    min_t0 = min_t0.min(t0);
                }
            }
        }
        if !min_t0.is_finite() {
            min_t0 = 0.0;
        }

        let mut routes = Vec::new();
        let mut keys: Vec<(String, String)> = grouped_trips.keys().cloned().collect();
        keys.sort();

        for (idx, key) in keys.into_iter().enumerate() {
            let trips = &grouped_trips[&key];
            let (_, template_stops) = &trips[0];

            // Build connected segment_chain across all stops in the pattern
            let mut seg_chain: Vec<usize> = Vec::new();
            let mut route_stops: Vec<(usize, usize)> = Vec::new();

            for (si, &(_, stop_idx, _)) in template_stops.iter().enumerate() {
                let target_seg = stops[stop_idx].segment_id;
                if si == 0 {
                    seg_chain.push(target_seg);
                    route_stops.push((stop_idx, 0));
                } else {
                    let prev_seg = *seg_chain.last().unwrap();
                    let sub = shortest_segment_chain(net, prev_seg, target_seg);
                    if sub.is_empty() {
                        continue;
                    }
                    for &s in sub.iter().skip(1) {
                        seg_chain.push(s);
                    }
                    route_stops.push((stop_idx, seg_chain.len() - 1));
                }
            }

            // Extend segment_chain to a downstream boundary sink if available so buses exit cleanly after terminal stop
            if let Some(&last_seg) = seg_chain.last() {
                if !net.sinks.contains(&last_seg) {
                    let mut best_tail: Vec<usize> = Vec::new();
                    for &sk in net.sinks.iter().take(16) {
                        let tail = shortest_segment_chain(net, last_seg, sk);
                        if !tail.is_empty()
                            && (best_tail.is_empty() || tail.len() < best_tail.len())
                        {
                            best_tail = tail;
                        }
                    }
                    for &s in best_tail.iter().skip(1) {
                        seg_chain.push(s);
                    }
                }
            }

            if seg_chain.is_empty() || route_stops.is_empty() {
                continue;
            }

            let mut explicit_deps: Vec<f64> = trips
                .iter()
                .filter_map(|(_, st)| st.first().and_then(|x| x.2).map(|t| (t - min_t0).max(0.0)))
                .collect();
            explicit_deps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let (short_base, long_base, col) = route_meta
                .get(&key.0)
                .cloned()
                .unwrap_or_else(|| (key.0.clone(), key.0.clone(), [56, 189, 248]));
            let dir_suffix = if key.1 == "1" { "B" } else { "A" };
            let rid_idx = routes.len();
            let hw_s = (headway_min * 60.0).max(45.0);
            let stagger = (idx as f64 * 4.5) % hw_s;

            routes.push(TransitRoute {
                id: rid_idx,
                gtfs_route_id: format!("{}_{}", key.0, key.1),
                short_name: format!("{}{}", short_base, dir_suffix),
                long_name: long_base,
                color_rgb: col,
                segment_chain: seg_chain,
                route_stops,
                explicit_departures_s: explicit_deps,
                next_explicit_idx: 0,
                next_periodic_dispatch_s: stagger,
                active_runs: 0,
            });
        }

        if routes.is_empty() {
            return Ok(Self::synthesize_for_network(net, parcels, headway_min));
        }

        let mut sys = Self {
            stops,
            routes,
            fare_mode: BoardingFareMode::AllDoorFareFreeBrt,
            coach_spec: BusCoachSpec::Standard40Ft,
            hold_early_buses: false,
            telemetry: TransitTelemetrySnapshot::default(),
            dwell_sum_s: 0.0,
            dwell_events: 0,
        };
        sys.telemetry.route_count = sys.routes.len();
        sys.telemetry.stop_count = sys.stops.len();
        Ok(sys)
    }

    /// Synthesizes realistic arterial GTFS routes and curbside stops automatically for any
    /// road network (`RiverfrontBridges`, `HarborPeninsula`, `RadialRingPlazas`, `FoothillCanyonWui`, `Metropolis50k`, or imported `.osm` map).
    pub fn synthesize_for_network(
        net: &RoadNetwork,
        parcels: &[Parcel],
        headway_min: f64,
    ) -> Self {
        let mut stops: Vec<TransitStop> = Vec::new();
        let mut routes: Vec<TransitRoute> = Vec::new();

        // Select up to 8 major boundary-to-boundary arterial corridors
        let mut candidate_sources: Vec<usize> = net
            .sources
            .iter()
            .copied()
            .filter(|&sid| !net.segments[sid].closed)
            .collect();
        candidate_sources.sort_by_key(|&sid| {
            std::cmp::Reverse((
                net.segments[sid].has_bus_lane as u8,
                net.segments[sid].priority,
                net.segments[sid].lane_count,
                sid,
            ))
        });

        let max_routes = if net.junctions.len() >= 60 { 8 } else { 4 };
        let mut sorted_sinks: Vec<usize> = net.sinks.iter().copied().collect();
        sorted_sinks.sort_unstable();

        let palette = [
            [239, 68, 68],
            [56, 189, 248],
            [16, 185, 129],
            [251, 191, 36],
            [192, 132, 252],
            [244, 114, 182],
            [45, 212, 191],
            [251, 146, 60],
        ];

        let mut stop_on_seg: HashMap<usize, usize> = HashMap::new();

        for &src_sid in &candidate_sources {
            if routes.len() >= max_routes {
                break;
            }
            let src_pos = net.junctions[net.segments[src_sid].from_junc].pos;
            // Pick the farthest reachable sink to form a true cross-town trunk route
            let mut best_chain: Vec<usize> = Vec::new();
            let mut best_dist = 0.0;
            for &sk_sid in &sorted_sinks {
                if sk_sid == src_sid {
                    continue;
                }
                let sk_pos = net.junctions[net.segments[sk_sid].to_junc].pos;
                let d = src_pos.dist(sk_pos);
                if d <= best_dist {
                    continue;
                }
                let chain = shortest_segment_chain(net, src_sid, sk_sid);
                if chain.len() >= 2 {
                    best_dist = d;
                    best_chain = chain;
                }
            }
            if best_chain.is_empty() {
                continue;
            }

            // Place curbside stops every ~170 m along the route chain
            let mut route_stops: Vec<(usize, usize)> = Vec::new();
            let mut dist_since_stop = 140.0;
            for (ci, &seg_id) in best_chain.iter().enumerate() {
                let seg = &net.segments[seg_id];
                dist_since_stop += seg.length;
                if seg.length >= 22.0 && !seg.is_roundabout && dist_since_stop >= 165.0 {
                    dist_since_stop = 0.0;
                    let stop_idx = if let Some(&existing) = stop_on_seg.get(&seg_id) {
                        existing
                    } else {
                        let s_m = (seg.length * 0.58).clamp(5.0, (seg.length - 5.0).max(5.0));
                        let (p_c, ang) = sample_polyline_fast(&seg.base_pts, &seg.base_cum_s, s_m);
                        let curb_off = if seg.is_oneway {
                            (seg.lane_count as f64 * 0.5) * LANE_WIDTH_M + 1.85
                        } else {
                            (seg.lane_count as f64) * LANE_WIDTH_M + 1.85
                        };
                        let world_pos = Vec2 {
                            x: p_c.x - ang.sin() * curb_off,
                            y: p_c.y + ang.cos() * curb_off,
                        };
                        let (base_origin_pax_hr, attraction_weight) =
                            compute_stop_parcel_weights(world_pos, parcels);
                        let id = stops.len();
                        stops.push(TransitStop {
                            id,
                            gtfs_stop_id: format!("STOP_{:03}", id + 1),
                            name: format!("{} Stop #{}", seg.name, id + 1),
                            segment_id: seg_id,
                            s_m,
                            world_pos,
                            heading_rad: ang,
                            base_origin_pax_hr,
                            attraction_weight,
                            waiting_pax: (base_origin_pax_hr * 0.06).clamp(1.5, 11.0),
                            total_boardings: 0,
                            total_alightings: 0,
                            denied_passups: 0,
                            last_bus_departure_s: None,
                        });
                        stop_on_seg.insert(seg_id, id);
                        id
                    };
                    route_stops.push((stop_idx, ci));
                }
            }

            if route_stops.is_empty() {
                continue;
            }
            let r_id = routes.len();
            let first_name = &net.segments[best_chain[0]].name;
            let last_name = &net.segments[*best_chain.last().unwrap()].name;
            routes.push(TransitRoute {
                id: r_id,
                gtfs_route_id: format!("CTS_{}", r_id + 1),
                short_name: format!("{}", r_id + 1),
                long_name: format!("{} -> {}", first_name, last_name),
                color_rgb: palette[r_id % palette.len()],
                segment_chain: best_chain,
                route_stops,
                explicit_departures_s: Vec::new(),
                next_explicit_idx: 0,
                next_periodic_dispatch_s: r_id as f64 * 3.5,
                active_runs: 0,
            });
            let _ = headway_min;
        }

        let mut sys = Self {
            stops,
            routes,
            fare_mode: BoardingFareMode::AllDoorFareFreeBrt,
            coach_spec: BusCoachSpec::Standard40Ft,
            hold_early_buses: false,
            telemetry: TransitTelemetrySnapshot::default(),
            dwell_sum_s: 0.0,
            dwell_events: 0,
        };
        sys.telemetry.route_count = sys.routes.len();
        sys.telemetry.stop_count = sys.stops.len();
        sys
    }

    pub fn refresh_stop_catchments(&mut self, parcels: &[Parcel]) {
        for stop in &mut self.stops {
            let (orig, attr) = compute_stop_parcel_weights(stop.world_pos, parcels);
            stop.base_origin_pax_hr = orig;
            stop.attraction_weight = attr;
        }
    }

    /// Advances curbside passenger arrivals and emits scheduled bus dispatches due at `sim_time_s`.
    pub fn step_arrivals_and_dispatch(
        &mut self,
        dt: f64,
        sim_time_s: f64,
        transit_mode_share: f64,
        headway_min: f64,
    ) -> Vec<PendingBusDispatch> {
        // Scale curbside arrivals by live McFadden transit mode share (baseline ~0.14)
        let share_mult = (transit_mode_share / 0.14).clamp(0.25, 4.5);
        for stop in &mut self.stops {
            let arrival_per_s = (stop.base_origin_pax_hr * share_mult) / 3600.0;
            stop.waiting_pax = (stop.waiting_pax + arrival_per_s * dt).min(85.0);
        }

        // Accelerated simulation headway compression so users immediately observe multiple scheduled
        // transit runs per corridor within seconds while preserving exact relative policy scaling:
        // e.g. 5 min BRT -> every ~18 s sim dispatch; 15 min -> every ~42 s; 30 min -> every ~75 s.
        let dispatch_interval_s = (8.0 + headway_min * 2.25).clamp(12.0, 90.0);
        let (length_m, width_m, seated_cap, crush_cap) = self.coach_spec.dimensions_and_capacity();

        let mut dispatches = Vec::new();
        for route in &mut self.routes {
            if route.segment_chain.is_empty() {
                continue;
            }
            let mut should_fire = false;
            while route.next_explicit_idx < route.explicit_departures_s.len()
                && sim_time_s >= route.explicit_departures_s[route.next_explicit_idx]
            {
                route.next_explicit_idx += 1;
                should_fire = true;
            }
            if sim_time_s >= route.next_periodic_dispatch_s {
                route.next_periodic_dispatch_s = sim_time_s + dispatch_interval_s;
                should_fire = true;
            }
            if should_fire {
                let init_pax = ((seated_cap as f64 * 0.28 * share_mult.min(2.0)).round() as u16)
                    .min(crush_cap);
                dispatches.push(PendingBusDispatch {
                    route_idx: route.id,
                    start_segment_id: route.segment_chain[0],
                    length_m,
                    width_m,
                    crush_capacity: crush_cap,
                    initial_pax: init_pax,
                });
            }
        }
        dispatches
    }

    /// Evaluates passenger alighting, queue boarding, crush pass-ups, and TRB TCQSM dwell time
    /// when a scheduled bus reaches `stop_order` on `route_idx`.
    pub fn process_bus_stop_service(
        &mut self,
        route_idx: usize,
        stop_order: usize,
        current_onboard: u16,
        sim_time_s: f64,
        headway_min: f64,
    ) -> DwellOutcome {
        let Some(route) = self.routes.get(route_idx) else {
            return DwellOutcome {
                dwell_s: 0.0,
                boarded: 0,
                alighted: 0,
                new_onboard: current_onboard,
            };
        };
        let total_route_stops = route.route_stops.len().max(1);
        let Some(&(stop_id, _)) = route.route_stops.get(stop_order) else {
            return DwellOutcome {
                dwell_s: 0.0,
                boarded: 0,
                alighted: 0,
                new_onboard: current_onboard,
            };
        };

        let (_, _, seated_cap, crush_cap) = self.coach_spec.dimensions_and_capacity();
        let (clearance_s, sec_per_board, sec_per_alight, all_door_parallel) =
            self.fare_mode.tcqsm_timings();

        let stop = &mut self.stops[stop_id];
        let is_terminal = stop_order + 1 >= total_route_stops;
        let progress = (stop_order + 1) as f64 / total_route_stops as f64;
        let alight_prob = if is_terminal {
            1.0
        } else {
            (0.14 + 0.34 * progress + 0.04 * stop.attraction_weight).clamp(0.12, 0.72)
        };
        let alighted = ((current_onboard as f64 * alight_prob).round() as u16).min(current_onboard);
        let after_alight = current_onboard.saturating_sub(alighted);

        let waiting_whole = if is_terminal {
            0u16
        } else {
            stop.waiting_pax.floor().max(0.0) as u16
        };
        let avail_space = crush_cap.saturating_sub(after_alight);
        let boarded = waiting_whole.min(avail_space);
        let denied = waiting_whole.saturating_sub(boarded);

        stop.waiting_pax = (stop.waiting_pax - boarded as f64).max(0.0);
        stop.total_boardings += boarded as u64;
        stop.total_alightings += alighted as u64;
        stop.denied_passups += denied as u64;

        let new_onboard = after_alight + boarded;

        if boarded == 0 && alighted == 0 {
            stop.last_bus_departure_s = Some(sim_time_s);
            return DwellOutcome {
                dwell_s: 0.0,
                boarded: 0,
                alighted: 0,
                new_onboard,
            };
        }

        let t_board = boarded as f64 * sec_per_board;
        let t_alight = alighted as f64 * sec_per_alight;
        let pax_service_s = if all_door_parallel {
            t_board.max(t_alight)
        } else {
            t_board + 0.65 * t_alight
        };

        // TCQSM standee crowding friction multiplier above seated capacity
        let standees = new_onboard.saturating_sub(seated_cap) as f64;
        let standee_span = crush_cap.saturating_sub(seated_cap).max(1) as f64;
        let crowding_mult = 1.0 + 0.48 * (standees / standee_span).clamp(0.0, 1.0);

        let mut dwell_s = clearance_s + pax_service_s * crowding_mult;

        // Optional anti-bunching headway holding control (Daganzo 2009)
        if self.hold_early_buses {
            let target_hw_s = (8.0 + headway_min * 2.25).clamp(12.0, 90.0);
            if let Some(prev_dep) = stop.last_bus_departure_s {
                let actual_hw = sim_time_s - prev_dep;
                if actual_hw < target_hw_s * 0.72 {
                    let hold_extra = ((target_hw_s * 0.72 - actual_hw) * 0.5).clamp(0.0, 10.0);
                    dwell_s += hold_extra;
                }
            }
        }

        let final_dwell = dwell_s.clamp(2.5, 42.0);
        stop.last_bus_departure_s = Some(sim_time_s + final_dwell);

        self.dwell_sum_s += final_dwell;
        self.dwell_events += 1;

        DwellOutcome {
            dwell_s: final_dwell as f32,
            boarded,
            alighted,
            new_onboard,
        }
    }

    /// Refreshes live ridership, load factor %, headway bunching %, and TCQSM crowding LOS A-F.
    pub fn update_telemetry_from_fleet(
        &mut self,
        bus_loads: &[(usize, usize, f64, u16, u16)], // (route_idx, route_seq, s_m, onboard_pax, crush_cap)
        headway_min: f64,
    ) {
        for r in &mut self.routes {
            r.active_runs = 0;
        }
        let mut total_onboard = 0u32;
        let mut sum_lf = 0.0f64;
        let mut peak_lf = 0.0f64;

        let (_, _, seated_cap, default_crush) = self.coach_spec.dimensions_and_capacity();
        let mut sum_seated_ratio = 0.0f64;

        // Group longitudinal route progress to detect Newell-Potts bus bunching
        let mut route_positions: HashMap<usize, Vec<f64>> = HashMap::new();

        for &(r_idx, r_seq, s_m, pax, crush) in bus_loads {
            if let Some(r) = self.routes.get_mut(r_idx) {
                r.active_runs += 1;
            }
            total_onboard += pax as u32;
            let eff_crush = crush.max(default_crush).max(1) as f64;
            let lf = (pax as f64 / eff_crush) * 100.0;
            sum_lf += lf;
            peak_lf = peak_lf.max(lf);
            sum_seated_ratio += pax as f64 / (seated_cap.max(1) as f64);

            let approx_route_m = (r_seq as f64) * 145.0 + s_m;
            route_positions.entry(r_idx).or_default().push(approx_route_m);
        }

        let mut pairs = 0usize;
        let mut bunched = 0usize;
        let expected_spacing_m = ((8.0 + headway_min * 2.25) * 7.5).max(95.0);
        for positions in route_positions.values_mut() {
            if positions.len() >= 2 {
                positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                for w in positions.windows(2) {
                    pairs += 1;
                    if (w[1] - w[0]) < expected_spacing_m * 0.45 {
                        bunched += 1;
                    }
                }
            }
        }

        let n_buses = bus_loads.len();
        let mean_lf = if n_buses > 0 {
            sum_lf / n_buses as f64
        } else {
            0.0
        };
        let mean_seated_ratio = if n_buses > 0 {
            sum_seated_ratio / n_buses as f64
        } else {
            0.0
        };

        // TCQSM Passenger Load Level of Service (TCRP Report 165 Exhibit 3-27)
        let los = if mean_seated_ratio <= 0.50 {
            'A'
        } else if mean_seated_ratio <= 0.80 {
            'B'
        } else if mean_seated_ratio <= 1.00 {
            'C'
        } else if mean_seated_ratio <= 1.25 {
            'D'
        } else if mean_seated_ratio <= 1.50 {
            'E'
        } else {
            'F'
        };

        let waiting_sum: f64 = self.stops.iter().map(|s| s.waiting_pax).sum();
        let board_sum: u64 = self.stops.iter().map(|s| s.total_boardings).sum();
        let alight_sum: u64 = self.stops.iter().map(|s| s.total_alightings).sum();
        let passup_sum: u64 = self.stops.iter().map(|s| s.denied_passups).sum();

        self.telemetry = TransitTelemetrySnapshot {
            route_count: self.routes.len(),
            stop_count: self.stops.len(),
            active_scheduled_buses: n_buses,
            waiting_passengers: waiting_sum.round() as u32,
            onboard_passengers: total_onboard,
            cumulative_boardings: board_sum,
            cumulative_alightings: alight_sum,
            denied_passups: passup_sum,
            mean_dwell_s: if self.dwell_events > 0 {
                self.dwell_sum_s / self.dwell_events as f64
            } else {
                0.0
            },
            mean_load_factor_pct: mean_lf,
            peak_load_factor_pct: peak_lf,
            bunching_pct: if pairs > 0 {
                (bunched as f64 / pairs as f64) * 100.0
            } else {
                0.0
            },
            tcqsm_crowding_los: los,
        };
    }
}

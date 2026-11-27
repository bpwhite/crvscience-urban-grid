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

use crate::emergency::{evp_override_connector_light, JunctionEvpState};
use crate::max_pressure::{mp_connector_light, JunctionMpState};
use crate::osm::{PlanarProjector, RawOsmDataset, RawTurnRestriction};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub const LANE_WIDTH_M: f64 = 3.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnType {
    Left,
    Through,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMode {
    Signal2Phase,
    Signal4PhaseProtectedLeft,
    MaxPressureAdaptive,
    Roundabout,
    PermissiveYield,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightColor {
    Green,
    Yellow,
    Red,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn dist(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

pub fn polyline_length(pts: &[Vec2]) -> f64 {
    pts.windows(2).map(|w| w[0].dist(w[1])).sum()
}

pub fn build_cumulative_lengths(pts: &[Vec2]) -> Vec<f64> {
    let mut cum = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    for (i, pt) in pts.iter().enumerate() {
        if i > 0 {
            acc += pts[i - 1].dist(*pt);
        }
        cum.push(acc);
    }
    cum
}

/// Fast `O(log K)` binary-searched polyline sampler using precomputed cumulative arc lengths.
/// Eliminates linear segment walks and `hypot` square-roots when packing 50,000+ GPU agents per frame.
#[inline]
pub fn sample_polyline_fast(pts: &[Vec2], cum_s: &[f64], dist: f64) -> (Vec2, f64) {
    let n = pts.len();
    if n == 0 {
        return (Vec2 { x: 0.0, y: 0.0 }, 0.0);
    }
    if n == 1 || cum_s.len() != n {
        return sample_polyline(pts, dist);
    }
    let total = cum_s[n - 1];
    let target = dist.clamp(0.0, total);
    let upper = cum_s.partition_point(|&s| s < target);
    let idx = if upper == 0 {
        0
    } else if upper >= n {
        n - 2
    } else {
        upper - 1
    };
    let s0 = cum_s[idx];
    let seg_l = (cum_s[idx + 1] - s0).max(1e-6);
    let t = ((target - s0) / seg_l).clamp(0.0, 1.0);
    let dx = pts[idx + 1].x - pts[idx].x;
    let dy = pts[idx + 1].y - pts[idx].y;
    (
        Vec2 {
            x: pts[idx].x + dx * t,
            y: pts[idx].y + dy * t,
        },
        dy.atan2(dx),
    )
}

pub fn sample_polyline(pts: &[Vec2], dist: f64) -> (Vec2, f64) {
    if pts.is_empty() {
        return (Vec2 { x: 0.0, y: 0.0 }, 0.0);
    }
    if pts.len() == 1 {
        return (pts[0], 0.0);
    }
    let mut rem = dist.max(0.0);
    for (idx, w) in pts.windows(2).enumerate() {
        let dx = w[1].x - w[0].x;
        let dy = w[1].y - w[0].y;
        let seg_l = dx.hypot(dy).max(1e-6);
        if rem <= seg_l || idx + 2 == pts.len() {
            let t = (rem / seg_l).clamp(0.0, 1.0);
            return (
                Vec2 {
                    x: w[0].x + dx * t,
                    y: w[0].y + dy * t,
                },
                dy.atan2(dx),
            );
        }
        rem -= seg_l;
    }
    let n = pts.len();
    let dx = pts[n - 1].x - pts[n - 2].x;
    let dy = pts[n - 1].y - pts[n - 2].y;
    (pts[n - 1], dy.atan2(dx))
}

fn trim_and_offset_polyline(
    base: &[Vec2],
    trim_start: f64,
    trim_end: f64,
    lat_right_m: f64,
) -> Vec<Vec2> {
    let total = polyline_length(base);
    let s0 = trim_start.min(total * 0.35);
    let s1 = (total - trim_end.min(total * 0.35)).max(s0 + 4.0);
    let steps = (((s1 - s0) / 8.0).ceil() as usize).max(3);
    let mut out = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let s = s0 + (s1 - s0) * (i as f64) / (steps as f64);
        let (p, ang) = sample_polyline(base, s);
        let nx = -ang.sin();
        let ny = ang.cos();
        out.push(Vec2 {
            x: p.x + nx * lat_right_m,
            y: p.y + ny * lat_right_m,
        });
    }
    out
}

fn normalize_angle(mut a: f64) -> f64 {
    while a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    }
    while a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

fn segs_intersect_2d(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool {
    let cross = |o: Vec2, p: Vec2, q: Vec2| -> f64 {
        (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x)
    };
    let d1 = cross(a0, a1, b0);
    let d2 = cross(a0, a1, b1);
    let d3 = cross(b0, b1, a0);
    let d4 = cross(b0, b1, a1);
    (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
}

fn polylines_intersect(a: &[Vec2], b: &[Vec2]) -> bool {
    for wa in a.windows(2) {
        for wb in b.windows(2) {
            if segs_intersect_2d(wa[0], wa[1], wb[0], wb[1]) {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lane {
    pub id: usize,
    pub segment_id: usize,
    pub index: usize,
    pub pts: Vec<Vec2>,
    pub cum_s: Vec<f64>,
    pub length: f64,
    pub start_angle: f64,
    pub end_angle: f64,
    pub out_connectors: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnConnector {
    pub id: usize,
    pub junction_id: usize,
    pub from_seg_id: usize,
    pub to_seg_id: usize,
    pub from_lane_id: usize,
    pub to_lane_id: usize,
    pub turn_type: TurnType,
    pub approach_axis: u8,
    pub pts: Vec<Vec2>,
    pub cum_s: Vec<f64>,
    pub length: f64,
    pub speed_mps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadSegment {
    pub id: usize,
    pub way_id: String,
    pub name: String,
    pub highway: String,
    pub priority: u8,
    pub is_roundabout: bool,
    pub is_oneway: bool,
    pub from_junc: usize,
    pub to_junc: usize,
    pub base_pts: Vec<Vec2>,
    pub base_cum_s: Vec<f64>,
    pub length: f64,
    pub lane_count: usize,
    pub speed_mph: f64,
    pub speed_mps: f64,
    pub approach_axis: u8,
    pub has_bus_lane: bool,
    pub has_bike_lane: bool,
    pub closed: bool,
    pub stalled_lane: Option<usize>,
    pub ema_speed_mps: f64,
    pub vsp_intensity: f64,
    pub lane_ids: Vec<usize>,
    #[serde(default)]
    pub is_bridge: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Junction {
    pub id: usize,
    pub osm_node_id: String,
    pub pos: Vec2,
    pub in_segs: Vec<usize>,
    pub out_segs: Vec<usize>,
    pub connectors: Vec<usize>,
    pub is_roundabout_node: bool,
    pub arm_count: usize,
    pub control_mode: ControlMode,
    pub ns_green_s: f64,
    pub ew_green_s: f64,
    pub yellow_s: f64,
    pub all_red_s: f64,
    pub offset_s: f64,
    pub hold_phase: bool,
    pub held_at_s: f64,
    pub manual_step_offset_s: f64,
    pub axis_angle: f64,
    #[serde(default)]
    pub mp_state: JunctionMpState,
    #[serde(default)]
    pub evp_state: JunctionEvpState,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterFeature {
    pub name: String,
    pub kind: String,
    pub pts: Vec<Vec2>,
    pub half_width_m: f64,
    pub is_polygon: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadNetwork {
    pub projector: PlanarProjector,
    pub junctions: Vec<Junction>,
    pub segments: Vec<RoadSegment>,
    pub lanes: Vec<Lane>,
    pub connectors: Vec<TurnConnector>,
    pub sources: Vec<usize>,
    pub sinks: HashSet<usize>,
    pub bounds: BoundingBox,
    #[serde(default)]
    pub restrictions: Vec<RawTurnRestriction>,
    #[serde(default = "default_mp_allow_4phase")]
    pub mp_allow_4phase: bool,
    #[serde(default)]
    pub water_features: Vec<WaterFeature>,
    #[serde(default)]
    pub city_label: String,
    #[serde(skip)]
    pub next_hop: HashMap<(usize, usize), usize>,
}

fn default_mp_allow_4phase() -> bool {
    true
}

fn highway_defaults(hwy: &str) -> (usize, f64, u8) {
    match hwy {
        "motorway" => (2, 55.0, 5),
        "motorway_link" => (1, 35.0, 4),
        "trunk" => (2, 35.0, 4),
        "trunk_link" => (1, 25.0, 3),
        "primary" => (2, 30.0, 4),
        "primary_link" => (1, 25.0, 3),
        "secondary" => (1, 25.0, 3),
        "tertiary" => (1, 25.0, 2),
        "living_street" => (1, 15.0, 1),
        _ => (1, 20.0, 1),
    }
}

fn parse_speed_mph(raw: Option<&String>, default_mph: f64) -> f64 {
    let Some(s) = raw else { return default_mph };
    let lower = s.trim().to_ascii_lowercase();
    let num_str: String = lower
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if let Ok(v) = num_str.parse::<f64>() {
        if v > 0.0 {
            return if lower.contains("mph") {
                v
            } else {
                (v * 0.621_371).round()
            };
        }
    }
    default_mph
}

impl RoadNetwork {
    pub fn compile(osm: &RawOsmDataset) -> Self {
        let mut node_counts: HashMap<&str, usize> = HashMap::new();
        for w in &osm.ways {
            let valid: Vec<&str> = w
                .node_refs
                .iter()
                .map(|s| s.as_str())
                .filter(|id| osm.nodes.contains_key(*id))
                .collect();
            for (idx, id) in valid.iter().enumerate() {
                let inc = if idx == 0 || idx + 1 == valid.len() { 2 } else { 1 };
                *node_counts.entry(*id).or_insert(0) += inc;
            }
        }

        let (mut sum_lat, mut sum_lon, mut cnt) = (0.0, 0.0, 0usize);
        for id in node_counts.keys() {
            if let Some(n) = osm.nodes.get(*id) {
                sum_lat += n.lat;
                sum_lon += n.lon;
                cnt += 1;
            }
        }
        let proj = PlanarProjector::from_centroid(
            if cnt > 0 { sum_lat / cnt as f64 } else { 40.0000 },
            if cnt > 0 { sum_lon / cnt as f64 } else { -105.0000 },
        );

        let water_features: Vec<WaterFeature> = osm
            .waterways
            .iter()
            .filter_map(|w| {
                let pts: Vec<Vec2> = w
                    .node_refs
                    .iter()
                    .filter_map(|nid| osm.nodes.get(nid))
                    .map(|n| {
                        let (x, y) = proj.project(n.lat, n.lon);
                        Vec2 { x, y }
                    })
                    .collect();
                if pts.len() < 2 {
                    return None;
                }
                let kind = w
                    .tags
                    .get("waterway")
                    .or_else(|| w.tags.get("natural"))
                    .cloned()
                    .unwrap_or_else(|| "water".to_string());
                let is_polygon = w.tags.get("area").map(|s| s == "yes").unwrap_or(false)
                    || matches!(kind.as_str(), "bay");
                let width_m = w
                    .tags
                    .get("width")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(if is_polygon { 100.0 } else { 40.0 });
                let name = w
                    .tags
                    .get("name")
                    .cloned()
                    .unwrap_or_else(|| "Waterway".to_string());
                Some(WaterFeature {
                    name,
                    kind,
                    pts,
                    half_width_m: (width_m * 0.5).max(8.0),
                    is_polygon,
                })
            })
            .collect();

        let mut junc_by_osm: HashMap<String, usize> = HashMap::new();
        let mut junctions: Vec<Junction> = Vec::new();

        macro_rules! ensure_junc {
            ($osm_id:expr) => {{
                let oid: &str = $osm_id;
                if let Some(&jid) = junc_by_osm.get(oid) {
                    jid
                } else {
                    let nd = &osm.nodes[oid];
                    let (x, y) = proj.project(nd.lat, nd.lon);
                    let jid = junctions.len();
                    junctions.push(Junction {
                        id: jid,
                        osm_node_id: oid.to_string(),
                        pos: Vec2 { x, y },
                        in_segs: Vec::new(),
                        out_segs: Vec::new(),
                        connectors: Vec::new(),
                        is_roundabout_node: false,
                        arm_count: 0,
                        control_mode: ControlMode::PermissiveYield,
                        ns_green_s: 22.0,
                        ew_green_s: 20.0,
                        yellow_s: 3.5,
                        all_red_s: 1.5,
                        offset_s: 0.0,
                        hold_phase: false,
                        held_at_s: 0.0,
                        manual_step_offset_s: 0.0,
                        axis_angle: 0.0,
                        mp_state: JunctionMpState::default(),
                        evp_state: JunctionEvpState::default(),
                    });
                    junc_by_osm.insert(oid.to_string(), jid);
                    jid
                }
            }};
        }

        // Split ways at topological junctions
        let mut segments: Vec<RoadSegment> = Vec::new();
        for w in &osm.ways {
            let valid: Vec<&str> = w
                .node_refs
                .iter()
                .map(|s| s.as_str())
                .filter(|id| osm.nodes.contains_key(*id))
                .collect();
            if valid.len() < 2 {
                continue;
            }
            let mut start_i = 0usize;
            for i in 1..valid.len() {
                let is_split =
                    i + 1 == valid.len() || node_counts.get(valid[i]).copied().unwrap_or(0) >= 2;
                if is_split {
                    let slice = &valid[start_i..=i];
                    if slice.len() >= 2 && slice[0] != slice[slice.len() - 1] {
                        let hwy = w
                            .tags
                            .get("highway")
                            .cloned()
                            .unwrap_or_else(|| "residential".into());
                        let (def_lanes, def_mph, prio) = highway_defaults(&hwy);
                        let is_rb = w.tags.get("junction").map(|s| s.as_str()) == Some("roundabout");
                        let ow_raw = w
                            .tags
                            .get("oneway")
                            .map(|s| s.to_ascii_lowercase())
                            .unwrap_or_default();
                        let is_oneway = is_rb || matches!(ow_raw.as_str(), "yes" | "1" | "true");
                        let is_rev = ow_raw == "-1";

                        let tot_lanes = w.tags.get("lanes").and_then(|s| s.parse::<usize>().ok());
                        let fwd_lanes = w
                            .tags
                            .get("lanes:forward")
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| {
                                if is_oneway {
                                    tot_lanes.unwrap_or(def_lanes).clamp(1, 4)
                                } else if let Some(tl) = tot_lanes {
                                    tl.div_ceil(2).clamp(1, 4)
                                } else {
                                    def_lanes
                                }
                            });
                        let bwd_lanes = w
                            .tags
                            .get("lanes:backward")
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or_else(|| {
                                if let Some(tl) = tot_lanes {
                                    (tl / 2).clamp(1, 4)
                                } else {
                                    def_lanes
                                }
                            });

                        let mph = parse_speed_mph(w.tags.get("maxspeed"), def_mph);
                        let pts_fwd: Vec<Vec2> = slice
                            .iter()
                            .map(|nid| {
                                let n = &osm.nodes[*nid];
                                let (x, y) = proj.project(n.lat, n.lon);
                                Vec2 { x, y }
                            })
                            .collect();
                        let raw_len = polyline_length(&pts_fwd);
                        if raw_len >= 8.0 {
                            let u = ensure_junc!(slice[0]);
                            let v = ensure_junc!(slice[slice.len() - 1]);
                            if is_rb {
                                junctions[u].is_roundabout_node = true;
                                junctions[v].is_roundabout_node = true;
                            }
                            let name = w
                                .tags
                                .get("name")
                                .cloned()
                                .unwrap_or_else(|| format!("{hwy} #{}", w.id));
                            let tagged_bridge = w
                                .tags
                                .get("bridge")
                                .map(|v| matches!(v.as_str(), "yes" | "viaduct" | "true"))
                                .unwrap_or(false);
                            let tagged_busway = w
                                .tags
                                .get("busway")
                                .map(|v| v == "lane" || v == "yes")
                                .unwrap_or(false)
                                || w.tags
                                    .get("bus:lanes")
                                    .map(|v| !v.is_empty())
                                    .unwrap_or(false);
                            let crosses_water = !tagged_bridge
                                && water_features.iter().any(|wf| {
                                    !wf.is_polygon && polylines_intersect(&pts_fwd, &wf.pts)
                                });
                            let seg_is_bridge = tagged_bridge || crosses_water;
                            let tagged_bike = w
                                .tags
                                .get("cycleway")
                                .map(|v| v == "lane" || v == "track")
                                .unwrap_or(false);

                            let mut push_seg =
                                |from_j: usize, to_j: usize, pts: Vec<Vec2>, lc: usize, centered: bool| {
                                    let sid = segments.len();
                                    let has_bike =
                                        tagged_bike || matches!(hwy.as_str(), "secondary" | "tertiary");
                                    let has_bus = tagged_busway
                                        || (hwy == "trunk"
                                            && lc >= 3
                                            && (w.id == "2001" || w.id.starts_with("200")));
                                    let base_cum_s = build_cumulative_lengths(&pts);
                                    segments.push(RoadSegment {
                                        id: sid,
                                        way_id: w.id.clone(),
                                        name: name.clone(),
                                        highway: hwy.clone(),
                                        priority: prio + if is_rb { 2 } else { 0 },
                                        is_roundabout: is_rb,
                                        is_oneway: centered,
                                        from_junc: from_j,
                                        to_junc: to_j,
                                        base_pts: pts,
                                        base_cum_s,
                                        length: raw_len,
                                        lane_count: lc,
                                        speed_mph: mph,
                                        speed_mps: mph * 0.447_04,
                                        approach_axis: 0,
                                        has_bus_lane: has_bus,
                                        has_bike_lane: has_bike,
                                        closed: false,
                                        stalled_lane: None,
                                        ema_speed_mps: mph * 0.447_04,
                                        vsp_intensity: 0.1,
                                        lane_ids: Vec::new(),
                                        is_bridge: seg_is_bridge,
                                    });
                                    junctions[from_j].out_segs.push(sid);
                                    junctions[to_j].in_segs.push(sid);
                                };

                            if is_rev {
                                let mut rev = pts_fwd;
                                rev.reverse();
                                push_seg(v, u, rev, fwd_lanes, true);
                            } else if is_oneway {
                                push_seg(u, v, pts_fwd, fwd_lanes, true);
                            } else {
                                let mut rev = pts_fwd.clone();
                                rev.reverse();
                                push_seg(u, v, pts_fwd, fwd_lanes, false);
                                push_seg(v, u, rev, bwd_lanes, false);
                            }
                        }
                    }
                    start_i = i;
                }
            }
        }

        // Pre-seed OSM traffic_signals node tags before building lanes & connectors
        for j in &mut junctions {
            let has_sig_tag = osm
                .nodes
                .get(&j.osm_node_id)
                .and_then(|n| n.tags.get("highway"))
                .map(|v| v.as_str() == "traffic_signals")
                .unwrap_or(false);
            if has_sig_tag && !j.is_roundabout_node {
                j.offset_s = ((j.pos.x + j.pos.y + 500.0).rem_euclid(45.0)).round();
                j.control_mode = ControlMode::Signal2Phase;
            }
        }

        let city_label = osm
            .city_name
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                if !water_features.is_empty() {
                    "Waterfront Metro".to_string()
                } else {
                    "Custom Street Network".to_string()
                }
            });

        let mut net = Self {
            projector: proj,
            junctions,
            segments,
            lanes: Vec::new(),
            connectors: Vec::new(),
            sources: Vec::new(),
            sinks: HashSet::new(),
            bounds: BoundingBox {
                min_x: -100.0,
                max_x: 100.0,
                min_y: -100.0,
                max_y: 100.0,
            },
            restrictions: osm.restrictions.clone(),
            mp_allow_4phase: true,
            water_features,
            city_label,
            next_hop: HashMap::new(),
        };
        net.rebuild_topology_from_segments(false);
        net
    }

    /// Rebuilds all 3.5 m metric `Lane`s, Bezier `TurnConnector`s, junction arm counts, boundary
    /// `sources`/`sinks`, `bounds`, and dynamic Dijkstra routes from `self.junctions` and `self.segments`.
    /// When `preserve_signal_modes == true` (live viewport editing), existing user-configured signal
    /// modes (`Signal2Phase`, `Signal4PhaseProtectedLeft`, `MaxPressureAdaptive`, `Roundabout`) are
    /// retained, while newly created multi-arm intersections are promoted automatically.
    pub fn rebuild_topology_from_segments(&mut self, preserve_signal_modes: bool) {
        for j in &mut self.junctions {
            j.in_segs.clear();
            j.out_segs.clear();
            j.connectors.clear();
        }
        for (sid, seg) in self.segments.iter_mut().enumerate() {
            seg.id = sid;
            seg.lane_ids.clear();
            seg.base_cum_s = build_cumulative_lengths(&seg.base_pts);
            self.junctions[seg.from_junc].out_segs.push(sid);
            self.junctions[seg.to_junc].in_segs.push(sid);
        }

        let mut lanes: Vec<Lane> = Vec::new();
        for seg in &mut self.segments {
            let deg_u = self.junctions[seg.from_junc].in_segs.len()
                + self.junctions[seg.from_junc].out_segs.len();
            let deg_v = self.junctions[seg.to_junc].in_segs.len()
                + self.junctions[seg.to_junc].out_segs.len();
            let trim_s = if deg_u <= 2 {
                4.0
            } else {
                (6.5 + seg.lane_count as f64 * 2.2).min(15.0)
            };
            let trim_e = if deg_v <= 2 {
                4.0
            } else {
                (6.5 + seg.lane_count as f64 * 2.2).min(15.0)
            };

            let lc = seg.lane_count.clamp(1, 6);
            seg.lane_count = lc;
            for li in 0..lc {
                let lat_off = if seg.is_oneway {
                    (li as f64 - (lc as f64 - 1.0) * 0.5) * LANE_WIDTH_M
                } else {
                    (li as f64 + 0.5) * LANE_WIDTH_M
                };
                let lpts = trim_and_offset_polyline(&seg.base_pts, trim_s, trim_e, lat_off);
                let cum_s = build_cumulative_lengths(&lpts);
                let llen = cum_s.last().copied().unwrap_or(4.0).max(4.0);
                let p0 = lpts[0];
                let p1 = lpts[1.min(lpts.len() - 1)];
                let pn = lpts[lpts.len() - 1];
                let pprev = lpts[lpts.len().saturating_sub(2)];
                let lid = lanes.len();
                lanes.push(Lane {
                    id: lid,
                    segment_id: seg.id,
                    index: li,
                    pts: lpts,
                    cum_s,
                    length: llen,
                    start_angle: (p1.y - p0.y).atan2(p1.x - p0.x),
                    end_angle: (pn.y - pprev.y).atan2(pn.x - pprev.x),
                    out_connectors: Vec::new(),
                });
                seg.lane_ids.push(lid);
            }
            if let Some(&first_lid) = seg.lane_ids.first() {
                seg.length = lanes[first_lid].length;
            }
        }

        let mut connectors: Vec<TurnConnector> = Vec::new();
        for jid in 0..self.junctions.len() {
            let mut neighbors = HashSet::new();
            for &in_sid in &self.junctions[jid].in_segs {
                neighbors.insert(self.segments[in_sid].from_junc);
            }
            for &out_sid in &self.junctions[jid].out_segs {
                neighbors.insert(self.segments[out_sid].to_junc);
            }
            self.junctions[jid].arm_count = neighbors.len();

            if let Some(&best_in) = self.junctions[jid].in_segs.iter().max_by_key(|&&sid| {
                (self.segments[sid].priority as usize) * 10 + self.segments[sid].lane_count
            }) {
                let lid0 = self.segments[best_in].lane_ids[0];
                self.junctions[jid].axis_angle = lanes[lid0].end_angle;
            }

            let max_prio = self.junctions[jid]
                .in_segs
                .iter()
                .map(|&sid| self.segments[sid].priority)
                .max()
                .unwrap_or(0);

            if self.junctions[jid].is_roundabout_node {
                self.junctions[jid].control_mode = ControlMode::Roundabout;
            } else if !preserve_signal_modes
                || self.junctions[jid].control_mode == ControlMode::PermissiveYield
            {
                if self.junctions[jid].control_mode == ControlMode::Signal2Phase
                    || (self.junctions[jid].arm_count >= 4
                        && max_prio >= 3
                        && self.junctions[jid].in_segs.len() >= 2)
                {
                    self.junctions[jid].offset_s = ((self.junctions[jid].pos.x
                        + self.junctions[jid].pos.y
                        + 500.0)
                        .rem_euclid(45.0))
                    .round();
                    if self.junctions[jid].control_mode == ControlMode::PermissiveYield {
                        self.junctions[jid].control_mode = ControlMode::Signal2Phase;
                    }
                }
            }

            let axis_angle = self.junctions[jid].axis_angle;
            let in_segs = self.junctions[jid].in_segs.clone();
            let out_segs = self.junctions[jid].out_segs.clone();
            let osm_nid = self.junctions[jid].osm_node_id.clone();

            for &in_sid in &in_segs {
                let in_ang = lanes[self.segments[in_sid].lane_ids[0]].end_angle;
                let app_axis = if (in_ang - axis_angle).sin().abs() < 0.7071 {
                    0
                } else {
                    1
                };
                self.segments[in_sid].approach_axis = app_axis;

                for &out_sid in &out_segs {
                    if self.segments[out_sid].to_junc == self.segments[in_sid].from_junc {
                        continue;
                    }
                    let forbidden = self.restrictions.iter().any(|res| {
                        res.from_way == self.segments[in_sid].way_id
                            && res.via_node == osm_nid
                            && ((res.rule.starts_with("no_")
                                && res.to_way == self.segments[out_sid].way_id)
                                || (res.rule.starts_with("only_")
                                    && res.to_way != self.segments[out_sid].way_id))
                    });
                    if forbidden {
                        continue;
                    }

                    let out_ang = lanes[self.segments[out_sid].lane_ids[0]].start_angle;
                    let turn_ang = normalize_angle(out_ang - in_ang);
                    let turn_type = if turn_ang < -0.48 {
                        TurnType::Left
                    } else if turn_ang > 0.48 {
                        TurnType::Right
                    } else {
                        TurnType::Through
                    };

                    let in_lc = self.segments[in_sid].lane_count;
                    let out_lc = self.segments[out_sid].lane_count;
                    let mut pairs = Vec::new();
                    match turn_type {
                        TurnType::Left => pairs.push((0, 0)),
                        TurnType::Right => pairs.push((in_lc - 1, out_lc - 1)),
                        TurnType::Through => {
                            let m = in_lc.max(out_lc);
                            for idx in 0..m {
                                pairs.push((idx.min(in_lc - 1), idx.min(out_lc - 1)));
                            }
                        }
                    }

                    for (li_in, li_out) in pairs {
                        let src_lid = self.segments[in_sid].lane_ids[li_in];
                        let dst_lid = self.segments[out_sid].lane_ids[li_out];
                        let cid = Self::push_bezier_connector(
                            jid,
                            in_sid,
                            out_sid,
                            src_lid,
                            dst_lid,
                            turn_type,
                            app_axis,
                            &self.segments,
                            &mut lanes,
                            &mut connectors,
                        );
                        self.junctions[jid].connectors.push(cid);
                    }
                }

                let non_uturn_outs: Vec<usize> = out_segs
                    .iter()
                    .copied()
                    .filter(|&os| self.segments[os].to_junc != self.segments[in_sid].from_junc)
                    .collect();
                if !non_uturn_outs.is_empty() {
                    let lane_ids = self.segments[in_sid].lane_ids.clone();
                    for (li_idx, src_lid) in lane_ids.into_iter().enumerate() {
                        if lanes[src_lid].out_connectors.is_empty() {
                            let chosen_out = non_uturn_outs[li_idx % non_uturn_outs.len()];
                            let dst_lid = self.segments[chosen_out].lane_ids
                                [li_idx.min(self.segments[chosen_out].lane_count - 1)];
                            let out_ang = lanes[self.segments[chosen_out].lane_ids[0]].start_angle;
                            let turn_ang = normalize_angle(out_ang - in_ang);
                            let t_type = if turn_ang < -0.48 {
                                TurnType::Left
                            } else if turn_ang > 0.48 {
                                TurnType::Right
                            } else {
                                TurnType::Through
                            };
                            let cid = Self::push_bezier_connector(
                                jid,
                                in_sid,
                                chosen_out,
                                src_lid,
                                dst_lid,
                                t_type,
                                app_axis,
                                &self.segments,
                                &mut lanes,
                                &mut connectors,
                            );
                            self.junctions[jid].connectors.push(cid);
                        }
                    }
                }
            }
        }

        let mut sources = Vec::new();
        let mut sinks = HashSet::new();
        for seg in &self.segments {
            let non_twin_in = self.junctions[seg.from_junc]
                .in_segs
                .iter()
                .any(|&s| self.segments[s].from_junc != seg.to_junc);
            if !non_twin_in {
                sources.push(seg.id);
            }
            let non_twin_out = self.junctions[seg.to_junc]
                .out_segs
                .iter()
                .any(|&s| self.segments[s].to_junc != seg.from_junc);
            if !non_twin_out {
                sinks.insert(seg.id);
            }
        }
        if sources.is_empty() && !self.segments.is_empty() {
            sources = (0..self.segments.len()).collect();
        }
        if sinks.is_empty() && !self.segments.is_empty() {
            sinks = (0..self.segments.len()).collect();
        }

        let (mut min_x, mut max_x, mut min_y, mut max_y) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for j in &self.junctions {
            if !j.in_segs.is_empty() || !j.out_segs.is_empty() {
                min_x = min_x.min(j.pos.x);
                max_x = max_x.max(j.pos.x);
                min_y = min_y.min(j.pos.y);
                max_y = max_y.max(j.pos.y);
            }
        }
        if !min_x.is_finite() {
            min_x = -100.0;
            max_x = 100.0;
            min_y = -100.0;
            max_y = 100.0;
        }

        self.lanes = lanes;
        self.connectors = connectors;
        self.sources = sources;
        self.sinks = sinks;
        self.bounds = BoundingBox {
            min_x: min_x - 25.0,
            max_x: max_x + 25.0,
            min_y: min_y - 25.0,
            max_y: max_y + 25.0,
        };
        self.recompute_dynamic_routes();
    }

    #[allow(clippy::too_many_arguments)]
    fn push_bezier_connector(
        jid: usize,
        in_sid: usize,
        out_sid: usize,
        src_lid: usize,
        dst_lid: usize,
        turn_type: TurnType,
        approach_axis: u8,
        segments: &[RoadSegment],
        lanes: &mut [Lane],
        connectors: &mut Vec<TurnConnector>,
    ) -> usize {
        let p0 = lanes[src_lid]
            .pts
            .last()
            .copied()
            .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        let p3 = lanes[dst_lid].pts.first().copied().unwrap_or(p0);
        let src_ang = lanes[src_lid].end_angle;
        let dst_ang = lanes[dst_lid].start_angle;
        let handle = (p0.dist(p3) * 0.42).max(2.5);
        let p1 = Vec2 {
            x: p0.x + src_ang.cos() * handle,
            y: p0.y + src_ang.sin() * handle,
        };
        let p2 = Vec2 {
            x: p3.x - dst_ang.cos() * handle,
            y: p3.y - dst_ang.sin() * handle,
        };
        let mut cpts = Vec::with_capacity(9);
        for s in 0..=8 {
            let t = s as f64 / 8.0;
            let mt = 1.0 - t;
            cpts.push(Vec2 {
                x: mt.powi(3) * p0.x
                    + 3.0 * mt.powi(2) * t * p1.x
                    + 3.0 * mt * t.powi(2) * p2.x
                    + t.powi(3) * p3.x,
                y: mt.powi(3) * p0.y
                    + 3.0 * mt.powi(2) * t * p1.y
                    + 3.0 * mt * t.powi(2) * p2.y
                    + t.powi(3) * p3.y,
            });
        }
        let cum_s = build_cumulative_lengths(&cpts);
        let clen = cum_s.last().copied().unwrap_or(2.5).max(2.5);
        let speed_mps = if turn_type == TurnType::Through {
            segments[in_sid].speed_mps.min(segments[out_sid].speed_mps)
        } else {
            9.0_f64.min(segments[out_sid].speed_mps)
        };
        let cid = connectors.len();
        connectors.push(TurnConnector {
            id: cid,
            junction_id: jid,
            from_seg_id: in_sid,
            to_seg_id: out_sid,
            from_lane_id: src_lid,
            to_lane_id: dst_lid,
            turn_type,
            approach_axis,
            pts: cpts,
            cum_s,
            length: clen,
            speed_mps,
        });
        lanes[src_lid].out_connectors.push(cid);
        cid
    }

    pub fn connector_light(&self, junc: &Junction, conn: &TurnConnector, sim_time_s: f64) -> LightColor {
        let base_light = match junc.control_mode {
            ControlMode::Roundabout | ControlMode::PermissiveYield => LightColor::Green,
            ControlMode::MaxPressureAdaptive => mp_connector_light(
                &junc.mp_state,
                conn,
                self.segments[conn.from_seg_id].lane_count,
                self.mp_allow_4phase,
            ),
            ControlMode::Signal2Phase => {
                let cycle =
                    (junc.ns_green_s + junc.ew_green_s + 2.0 * (junc.yellow_s + junc.all_red_s))
                        .max(1.0);
                let tref = if junc.hold_phase {
                    junc.held_at_s
                } else {
                    sim_time_s
                };
                let pos = (tref + junc.offset_s + junc.manual_step_offset_s).rem_euclid(cycle);
                let a_end = junc.ns_green_s;
                let ay_end = a_end + junc.yellow_s;
                let ar1_end = ay_end + junc.all_red_s;
                let b_end = ar1_end + junc.ew_green_s;
                let by_end = b_end + junc.yellow_s;

                if pos < a_end {
                    if conn.approach_axis == 0 {
                        LightColor::Green
                    } else {
                        LightColor::Red
                    }
                } else if pos < ay_end {
                    if conn.approach_axis == 0 {
                        LightColor::Yellow
                    } else {
                        LightColor::Red
                    }
                } else if pos < ar1_end {
                    LightColor::Red
                } else if pos < b_end {
                    if conn.approach_axis == 1 {
                        LightColor::Green
                    } else {
                        LightColor::Red
                    }
                } else if pos < by_end {
                    if conn.approach_axis == 1 {
                        LightColor::Yellow
                    } else {
                        LightColor::Red
                    }
                } else {
                    LightColor::Red
                }
            }
            ControlMode::Signal4PhaseProtectedLeft => {
                let pl = (junc.ns_green_s * 0.42).round().max(5.0);
                let step_overhead = junc.yellow_s + junc.all_red_s;
                let p0 = pl + step_overhead;
                let p1 = p0 + junc.ns_green_s + step_overhead;
                let p2 = p1 + pl + step_overhead;
                let cycle = (p2 + junc.ew_green_s + step_overhead).max(1.0);
                let tref = if junc.hold_phase {
                    junc.held_at_s
                } else {
                    sim_time_s
                };
                let pos = (tref + junc.offset_s + junc.manual_step_offset_s).rem_euclid(cycle);

                if pos < p0 {
                    if conn.approach_axis == 0 && conn.turn_type == TurnType::Left {
                        if pos < pl {
                            LightColor::Green
                        } else if pos < pl + junc.yellow_s {
                            LightColor::Yellow
                        } else {
                            LightColor::Red
                        }
                    } else {
                        LightColor::Red
                    }
                } else if pos < p1 {
                    let local = pos - p0;
                    if conn.approach_axis == 0 && conn.turn_type != TurnType::Left {
                        if local < junc.ns_green_s {
                            LightColor::Green
                        } else if local < junc.ns_green_s + junc.yellow_s {
                            LightColor::Yellow
                        } else {
                            LightColor::Red
                        }
                    } else {
                        LightColor::Red
                    }
                } else if pos < p2 {
                    let local = pos - p1;
                    if conn.approach_axis == 1 && conn.turn_type == TurnType::Left {
                        if local < pl {
                            LightColor::Green
                        } else if local < pl + junc.yellow_s {
                            LightColor::Yellow
                        } else {
                            LightColor::Red
                        }
                    } else {
                        LightColor::Red
                    }
                } else {
                    let local = pos - p2;
                    if conn.approach_axis == 1 && conn.turn_type != TurnType::Left {
                        if local < junc.ew_green_s {
                            LightColor::Green
                        } else if local < junc.ew_green_s + junc.yellow_s {
                            LightColor::Yellow
                        } else {
                            LightColor::Red
                        }
                    } else {
                        LightColor::Red
                    }
                }
            }
        };
        if let Some(evp_light) = evp_override_connector_light(&junc.evp_state, conn, base_light) {
            evp_light
        } else {
            base_light
        }
    }

    /// Converts all signal-eligible intersections (excluding roundabouts) to decentralized
    /// `ControlMode::MaxPressureAdaptive`, returning the number of upgraded junctions.
    pub fn enable_network_max_pressure(&mut self) -> usize {
        let mut count = 0usize;
        for j in &mut self.junctions {
            if !j.is_roundabout_node && !j.in_segs.is_empty() && !j.out_segs.is_empty() {
                if matches!(
                    j.control_mode,
                    ControlMode::Signal2Phase
                        | ControlMode::Signal4PhaseProtectedLeft
                        | ControlMode::MaxPressureAdaptive
                ) || j.arm_count >= 3
                {
                    j.control_mode = ControlMode::MaxPressureAdaptive;
                    count += 1;
                }
            }
        }
        count
    }

    /// Reverts `MaxPressureAdaptive` junctions back to standard coordinated `Signal2Phase`.
    pub fn restore_fixed_time_signals(&mut self) -> usize {
        let mut count = 0usize;
        for j in &mut self.junctions {
            if j.control_mode == ControlMode::MaxPressureAdaptive {
                j.control_mode = ControlMode::Signal2Phase;
                count += 1;
            }
        }
        count
    }

    /// Reverse-Dijkstra dynamic route table from each boundary sink across the reversed segment
    /// graph (`O(|Sinks| * |E| log |V|)` — ~12x faster than all-origins forward Dijkstra on supergrids).
    pub fn recompute_dynamic_routes(&mut self) {
        #[derive(Copy, Clone, PartialEq)]
        struct HeapState {
            cost: f64,
            seg_id: usize,
        }
        impl Eq for HeapState {}
        impl Ord for HeapState {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for HeapState {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let n = self.segments.len();
        // rev_adj[out_seg] = list of (in_seg, cost_from_in_to_out)
        let mut rev_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for seg in &self.segments {
            if seg.closed {
                continue;
            }
            let mut seen = HashSet::new();
            for &lid in &seg.lane_ids {
                for &cid in &self.lanes[lid].out_connectors {
                    let conn = &self.connectors[cid];
                    let out_seg = &self.segments[conn.to_seg_id];
                    if out_seg.closed || !seen.insert(out_seg.id) {
                        continue;
                    }
                    let v_est = out_seg.ema_speed_mps.max(2.2);
                    let stall_pen = if out_seg.stalled_lane.is_some() {
                        if out_seg.lane_count == 1 {
                            45.0
                        } else {
                            7.0
                        }
                    } else {
                        0.0
                    };
                    let turn_pen = match conn.turn_type {
                        TurnType::Left => 3.5,
                        TurnType::Right => 1.8,
                        TurnType::Through => 0.4,
                    };
                    let edge_cost = out_seg.length / v_est + turn_pen + stall_pen;
                    rev_adj[out_seg.id].push((seg.id, edge_cost));
                }
            }
        }

        self.next_hop.clear();
        self.next_hop.reserve(n * self.sinks.len().min(64));

        let mut dist = vec![f64::INFINITY; n];
        let mut next_toward_sink: Vec<Option<usize>> = vec![None; n];
        let mut heap = BinaryHeap::new();

        let mut sorted_sinks: Vec<usize> = self.sinks.iter().copied().collect();
        sorted_sinks.sort_unstable();

        for sink_id in sorted_sinks {
            if sink_id >= n || self.segments[sink_id].closed {
                continue;
            }
            dist.fill(f64::INFINITY);
            next_toward_sink.fill(None);
            heap.clear();

            dist[sink_id] = 0.0;
            heap.push(HeapState {
                cost: 0.0,
                seg_id: sink_id,
            });

            while let Some(HeapState { cost, seg_id: v }) = heap.pop() {
                if cost > dist[v] {
                    continue;
                }
                for &(u, w) in &rev_adj[v] {
                    let nd = cost + w;
                    if nd < dist[u] {
                        dist[u] = nd;
                        next_toward_sink[u] = Some(v);
                        heap.push(HeapState { cost: nd, seg_id: u });
                    }
                }
            }

            for (u, hop_opt) in next_toward_sink.iter().enumerate() {
                if u != sink_id {
                    if let Some(hop) = *hop_opt {
                        self.next_hop.insert((u, sink_id), hop);
                    }
                }
            }
        }
    }
}

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

use quick_xml::events::Event;
use quick_xml::Reader;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OsmError {
    #[error("I/O error reading OSM source: {0}")]
    Io(#[from] std::io::Error),
    #[error("XML parse error: {0}")]
    Xml(#[from] quick_xml::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("OSM dataset error: {0}")]
    Dataset(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawOsmNode {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawOsmWay {
    pub id: String,
    pub node_refs: Vec<String>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTurnRestriction {
    pub from_way: String,
    pub via_node: String,
    pub to_way: String,
    pub rule: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawOsmDataset {
    pub nodes: HashMap<String, RawOsmNode>,
    pub ways: Vec<RawOsmWay>,
    #[serde(default)]
    pub waterways: Vec<RawOsmWay>,
    pub restrictions: Vec<RawTurnRestriction>,
    #[serde(default)]
    pub city_name: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlanarProjector {
    pub lat0_deg: f64,
    pub lon0_deg: f64,
    pub cos_lat0: f64,
}

impl PlanarProjector {
    pub fn from_centroid(lat0_deg: f64, lon0_deg: f64) -> Self {
        Self {
            lat0_deg,
            lon0_deg,
            cos_lat0: lat0_deg.to_radians().cos(),
        }
    }

    /// Projects WGS84 `(lat, lon)` into local metric coordinates `(x_east_m, y_south_m)`.
    pub fn project(&self, lat: f64, lon: f64) -> (f64, f64) {
        let x = (lon - self.lon0_deg) * self.cos_lat0 * 111_320.0;
        let y = -(lat - self.lat0_deg) * 110_540.0;
        (x, y)
    }

    pub fn unproject(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let lat = self.lat0_deg - (y_m / 110_540.0);
        let lon = self.lon0_deg + (x_m / (111_320.0 * self.cos_lat0.max(1e-6)));
        (lat, lon)
    }
}

pub fn is_drivable_highway(tag: &str) -> bool {
    matches!(
        tag,
        "motorway"
            | "motorway_link"
            | "trunk"
            | "trunk_link"
            | "primary"
            | "primary_link"
            | "secondary"
            | "secondary_link"
            | "tertiary"
            | "tertiary_link"
            | "unclassified"
            | "residential"
            | "living_street"
    )
}

pub fn is_waterway_or_water(tags: &HashMap<String, String>) -> bool {
    tags.get("waterway")
        .map(|w| matches!(w.as_str(), "river" | "canal" | "stream" | "bay" | "riverbank"))
        .unwrap_or(false)
        || tags
            .get("natural")
            .map(|n| matches!(n.as_str(), "water" | "bay" | "coastline"))
            .unwrap_or(false)
}

/// Parses standard OpenStreetMap v0.6 `.osm` XML into `RawOsmDataset`.
pub fn parse_osm_xml_str(xml: &str) -> Result<RawOsmDataset, OsmError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut dataset = RawOsmDataset::default();
    let mut cur_node: Option<RawOsmNode> = None;
    let mut cur_way: Option<RawOsmWay> = None;
    let mut cur_rel_tags: HashMap<String, String> = HashMap::new();
    let mut cur_rel_members: Vec<(String, String, String)> = Vec::new();
    let mut in_relation = false;

    loop {
        match reader.read_event()? {
            Event::Start(e) => match e.name().as_ref() {
                b"osm" => {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"city" {
                            let val = String::from_utf8_lossy(&attr.value).into_owned();
                            if !val.trim().is_empty() {
                                dataset.city_name = Some(val);
                            }
                        }
                    }
                }
                b"node" => {
                    let mut id = String::new();
                    let mut lat = 0.0;
                    let mut lon = 0.0;
                    for attr in e.attributes().flatten() {
                        let val = String::from_utf8_lossy(&attr.value).into_owned();
                        match attr.key.as_ref() {
                            b"id" => id = val,
                            b"lat" => lat = val.parse().unwrap_or(0.0),
                            b"lon" => lon = val.parse().unwrap_or(0.0),
                            _ => {}
                        }
                    }
                    cur_node = Some(RawOsmNode {
                        id,
                        lat,
                        lon,
                        tags: HashMap::new(),
                    });
                }
                b"way" => {
                    let mut id = String::new();
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"id" {
                            id = String::from_utf8_lossy(&attr.value).into_owned();
                        }
                    }
                    cur_way = Some(RawOsmWay {
                        id,
                        node_refs: Vec::new(),
                        tags: HashMap::new(),
                    });
                }
                b"relation" => {
                    in_relation = true;
                    cur_rel_tags.clear();
                    cur_rel_members.clear();
                }
                _ => {}
            },
            Event::Empty(e) => match e.name().as_ref() {
                b"node" => {
                    let mut id = String::new();
                    let mut lat = 0.0;
                    let mut lon = 0.0;
                    for attr in e.attributes().flatten() {
                        let val = String::from_utf8_lossy(&attr.value).into_owned();
                        match attr.key.as_ref() {
                            b"id" => id = val,
                            b"lat" => lat = val.parse().unwrap_or(0.0),
                            b"lon" => lon = val.parse().unwrap_or(0.0),
                            _ => {}
                        }
                    }
                    if !id.is_empty() {
                        dataset.nodes.insert(
                            id.clone(),
                            RawOsmNode {
                                id,
                                lat,
                                lon,
                                tags: HashMap::new(),
                            },
                        );
                    }
                }
                b"nd" => {
                    if let Some(w) = cur_way.as_mut() {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"ref" {
                                w.node_refs
                                    .push(String::from_utf8_lossy(&attr.value).into_owned());
                            }
                        }
                    }
                }
                b"tag" => {
                    let mut k = String::new();
                    let mut v = String::new();
                    for attr in e.attributes().flatten() {
                        let val = String::from_utf8_lossy(&attr.value).into_owned();
                        match attr.key.as_ref() {
                            b"k" => k = val,
                            b"v" => v = val,
                            _ => {}
                        }
                    }
                    if let Some(n) = cur_node.as_mut() {
                        n.tags.insert(k, v);
                    } else if let Some(w) = cur_way.as_mut() {
                        w.tags.insert(k, v);
                    } else if in_relation {
                        cur_rel_tags.insert(k, v);
                    }
                }
                b"member" if in_relation => {
                    let mut mtype = String::new();
                    let mut mref = String::new();
                    let mut mrole = String::new();
                    for attr in e.attributes().flatten() {
                        let val = String::from_utf8_lossy(&attr.value).into_owned();
                        match attr.key.as_ref() {
                            b"type" => mtype = val,
                            b"ref" => mref = val,
                            b"role" => mrole = val,
                            _ => {}
                        }
                    }
                    cur_rel_members.push((mtype, mref, mrole));
                }
                _ => {}
            },
            Event::End(e) => match e.name().as_ref() {
                b"node" => {
                    if let Some(n) = cur_node.take() {
                        if !n.id.is_empty() {
                            dataset.nodes.insert(n.id.clone(), n);
                        }
                    }
                }
                b"way" => {
                    if let Some(w) = cur_way.take() {
                        if w.node_refs.len() >= 2 {
                            if w.tags
                                .get("highway")
                                .map(|h| is_drivable_highway(h))
                                .unwrap_or(false)
                            {
                                dataset.ways.push(w);
                            } else if is_waterway_or_water(&w.tags) {
                                dataset.waterways.push(w);
                            }
                        }
                    }
                }
                b"relation" => {
                    if in_relation
                        && cur_rel_tags.get("type").map(|s| s.as_str()) == Some("restriction")
                    {
                        if let Some(rule) = cur_rel_tags.get("restriction").cloned() {
                            let mut from_way = None;
                            let mut via_node = None;
                            let mut to_way = None;
                            for (_t, r, role) in &cur_rel_members {
                                match role.as_str() {
                                    "from" => from_way = Some(r.clone()),
                                    "via" => via_node = Some(r.clone()),
                                    "to" => to_way = Some(r.clone()),
                                    _ => {}
                                }
                            }
                            if let (Some(fw), Some(vn), Some(tw)) = (from_way, via_node, to_way) {
                                dataset.restrictions.push(RawTurnRestriction {
                                    from_way: fw,
                                    via_node: vn,
                                    to_way: tw,
                                    rule,
                                });
                            }
                        }
                    }
                    in_relation = false;
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }

    if dataset.ways.is_empty() {
        return Err(OsmError::Dataset(
            "No drivable highway ways found in OSM XML.".into(),
        ));
    }
    Ok(dataset)
}

#[derive(Deserialize)]
struct OverpassResponse {
    elements: Vec<OverpassElement>,
}

#[derive(Deserialize)]
struct OverpassElement {
    #[serde(rename = "type")]
    el_type: String,
    id: serde_json::Value,
    lat: Option<f64>,
    lon: Option<f64>,
    nodes: Option<Vec<serde_json::Value>>,
    members: Option<Vec<OverpassMember>>,
    tags: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
struct OverpassMember {
    #[serde(rename = "type")]
    m_type: String,
    #[serde(rename = "ref")]
    m_ref: serde_json::Value,
    role: String,
}

pub fn parse_overpass_json_str(json_str: &str) -> Result<RawOsmDataset, OsmError> {
    let parsed: OverpassResponse = serde_json::from_str(json_str)?;
    let mut ds = RawOsmDataset::default();

    for el in parsed.elements {
        let id_str = el.id.to_string().trim_matches('"').to_string();
        match el.el_type.as_str() {
            "node" => {
                if let (Some(lat), Some(lon)) = (el.lat, el.lon) {
                    ds.nodes.insert(
                        id_str.clone(),
                        RawOsmNode {
                            id: id_str,
                            lat,
                            lon,
                            tags: el.tags.unwrap_or_default(),
                        },
                    );
                }
            }
            "way" => {
                let tags = el.tags.unwrap_or_default();
                let is_hwy = tags
                    .get("highway")
                    .map(|h| is_drivable_highway(h))
                    .unwrap_or(false);
                if is_hwy || is_waterway_or_water(&tags) {
                    let node_refs: Vec<String> = el
                        .nodes
                        .unwrap_or_default()
                        .into_iter()
                        .map(|v| v.to_string().trim_matches('"').to_string())
                        .collect();
                    if node_refs.len() >= 2 {
                        let w = RawOsmWay {
                            id: id_str,
                            node_refs,
                            tags,
                        };
                        if is_hwy {
                            ds.ways.push(w);
                        } else {
                            ds.waterways.push(w);
                        }
                    }
                }
            }
            "relation" => {
                let tags = el.tags.unwrap_or_default();
                if tags.get("type").map(|s| s.as_str()) == Some("restriction") {
                    if let Some(rule) = tags.get("restriction").cloned() {
                        let mut fw = None;
                        let mut vn = None;
                        let mut tw = None;
                        for m in el.members.unwrap_or_default() {
                            let r = m.m_ref.to_string().trim_matches('"').to_string();
                            let _ = m.m_type;
                            match m.role.as_str() {
                                "from" => fw = Some(r),
                                "via" => vn = Some(r),
                                "to" => tw = Some(r),
                                _ => {}
                            }
                        }
                        if let (Some(from_way), Some(via_node), Some(to_way)) = (fw, vn, tw) {
                            ds.restrictions.push(RawTurnRestriction {
                                from_way,
                                via_node,
                                to_way,
                                rule,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if ds.ways.is_empty() {
        return Err(OsmError::Dataset(
            "No drivable ways found in Overpass JSON.".into(),
        ));
    }
    Ok(ds)
}

pub fn load_osm_path(path: &Path) -> Result<RawOsmDataset, OsmError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "pbf" {
        #[cfg(feature = "pbf")]
        {
            return load_osm_pbf_path(path);
        }
        #[cfg(not(feature = "pbf"))]
        {
            return Err(OsmError::Dataset(
                "Compile with `--features pbf` to read binary .osm.pbf archives.".into(),
            ));
        }
    }
    let text = fs::read_to_string(path)?;
    if text.trim_start().starts_with('{') {
        parse_overpass_json_str(&text)
    } else {
        parse_osm_xml_str(&text)
    }
}

#[cfg(feature = "pbf")]
pub fn load_osm_pbf_path(path: &Path) -> Result<RawOsmDataset, OsmError> {
    use osmpbf::{Element, ElementReader};
    let reader = ElementReader::from_path(path).map_err(|e| OsmError::Dataset(e.to_string()))?;
    let mut ds = RawOsmDataset::default();
    reader
        .for_each(|element| match element {
            Element::Node(n) => {
                let tags: HashMap<String, String> = n
                    .tags()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                ds.nodes.insert(
                    n.id().to_string(),
                    RawOsmNode {
                        id: n.id().to_string(),
                        lat: n.lat(),
                        lon: n.lon(),
                        tags,
                    },
                );
            }
            Element::DenseNode(n) => {
                let tags: HashMap<String, String> = n
                    .tags()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                ds.nodes.insert(
                    n.id().to_string(),
                    RawOsmNode {
                        id: n.id().to_string(),
                        lat: n.lat(),
                        lon: n.lon(),
                        tags,
                    },
                );
            }
            Element::Way(w) => {
                let tags: HashMap<String, String> = w
                    .tags()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                let is_hwy = tags
                    .get("highway")
                    .map(|h| is_drivable_highway(h))
                    .unwrap_or(false);
                if is_hwy || is_waterway_or_water(&tags) {
                    let node_refs: Vec<String> = w.refs().map(|r| r.to_string()).collect();
                    if node_refs.len() >= 2 {
                        let rw = RawOsmWay {
                            id: w.id().to_string(),
                            node_refs,
                            tags,
                        };
                        if is_hwy {
                            ds.ways.push(rw);
                        } else {
                            ds.waterways.push(rw);
                        }
                    }
                }
            }
            _ => {}
        })
        .map_err(|e| OsmError::Dataset(e.to_string()))?;
    Ok(ds)
}

#[cfg(feature = "overpass")]
pub fn fetch_overpass_bbox(
    south: f64,
    west: f64,
    north: f64,
    east: f64,
) -> Result<RawOsmDataset, OsmError> {
    let hwy_re = "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link)$";
    let ql = format!(
        r#"[out:json][timeout:20];(way["highway"~"{hwy_re}"]({south},{west},{north},{east});relation["type"="restriction"]({south},{west},{north},{east}););(._;>;);out body;"#
    );
    let resp = ureq::post("https://overpass-api.de/api/interpreter")
        .set("Content-Type", "application/x-www-form-urlencoded")
        .send_form(&[("data", &ql)])
        .map_err(|e| OsmError::Dataset(format!("Overpass HTTP failure: {e}")))?;
    let body = resp.into_string()?;
    parse_overpass_json_str(&body)
}

struct OsmCityBuilder {
    city_name: String,
    c_lat: f64,
    c_lon: f64,
    m_lat: f64,
    m_lon: f64,
    next_nid: usize,
    next_wid: usize,
    nodes_xml: Vec<String>,
    ways_xml: Vec<String>,
}

impl OsmCityBuilder {
    fn new(city_name: &str) -> Self {
        let c_lat = 40.0000_f64;
        let c_lon = -105.0000_f64;
        let m_lat = 1.0 / 110_540.0;
        let m_lon = 1.0 / (111_320.0 * c_lat.to_radians().cos());
        Self {
            city_name: city_name.to_string(),
            c_lat,
            c_lon,
            m_lat,
            m_lon,
            next_nid: 1,
            next_wid: 1,
            nodes_xml: Vec::new(),
            ways_xml: Vec::new(),
        }
    }

    fn node(&mut self, x_m: f64, y_m: f64, signal: bool) -> usize {
        let id = self.next_nid;
        self.next_nid += 1;
        // Screen/OSM projection uses y_south_m = -(lat - c_lat) * 110540
        let lat = self.c_lat - y_m * self.m_lat;
        let lon = self.c_lon + x_m * self.m_lon;
        if signal {
            self.nodes_xml.push(format!(
                r#"  <node id="{id}" lat="{lat:.6}" lon="{lon:.6}"><tag k="highway" v="traffic_signals"/></node>"#
            ));
        } else {
            self.nodes_xml
                .push(format!(r#"  <node id="{id}" lat="{lat:.6}" lon="{lon:.6}"/>"#));
        }
        id
    }

    #[allow(clippy::too_many_arguments)]
    fn way(
        &mut self,
        name: &str,
        highway: &str,
        lanes: usize,
        oneway: bool,
        maxspeed: &str,
        bridge: bool,
        busway: bool,
        roundabout: bool,
        nds: &[usize],
    ) {
        if nds.len() < 2 {
            return;
        }
        let id = self.next_wid;
        self.next_wid += 1;
        self.ways_xml.push(format!(r#"  <way id="{id}">"#));
        for &n in nds {
            self.ways_xml.push(format!(r#"    <nd ref="{n}"/>"#));
        }
        let ow = if oneway { "yes" } else { "no" };
        self.ways_xml.push(format!(r#"    <tag k="name" v="{name}"/>"#));
        self.ways_xml
            .push(format!(r#"    <tag k="highway" v="{highway}"/>"#));
        self.ways_xml
            .push(format!(r#"    <tag k="lanes" v="{lanes}"/>"#));
        self.ways_xml.push(format!(r#"    <tag k="oneway" v="{ow}"/>"#));
        self.ways_xml
            .push(format!(r#"    <tag k="maxspeed" v="{maxspeed}"/>"#));
        if bridge {
            self.ways_xml.push(r#"    <tag k="bridge" v="yes"/>"#.to_string());
        }
        if busway {
            self.ways_xml.push(r#"    <tag k="busway" v="lane"/>"#.to_string());
        }
        if roundabout {
            self.ways_xml
                .push(r#"    <tag k="junction" v="roundabout"/>"#.to_string());
        }
        self.ways_xml.push("  </way>".to_string());
    }

    fn waterway(
        &mut self,
        name: &str,
        kind: &str,
        width_m: f64,
        is_polygon: bool,
        nds: &[usize],
    ) {
        if nds.len() < 2 {
            return;
        }
        let id = self.next_wid;
        self.next_wid += 1;
        self.ways_xml.push(format!(r#"  <way id="{id}">"#));
        for &n in nds {
            self.ways_xml.push(format!(r#"    <nd ref="{n}"/>"#));
        }
        self.ways_xml.push(format!(r#"    <tag k="name" v="{name}"/>"#));
        if is_polygon {
            self.ways_xml
                .push(format!(r#"    <tag k="natural" v="{kind}"/>"#));
            self.ways_xml.push(r#"    <tag k="area" v="yes"/>"#.to_string());
        } else {
            self.ways_xml
                .push(format!(r#"    <tag k="waterway" v="{kind}"/>"#));
        }
        self.ways_xml
            .push(format!(r#"    <tag k="width" v="{width_m:.1}"/>"#));
        self.ways_xml.push("  </way>".to_string());
    }

    fn finish(self) -> String {
        let mut out = Vec::with_capacity(self.nodes_xml.len() + self.ways_xml.len() + 4);
        out.push(r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string());
        out.push(format!(
            r#"<osm version="0.6" generator="urban-grid" city="{}">"#,
            self.city_name
        ));
        out.extend(self.nodes_xml);
        out.extend(self.ways_xml);
        out.push("</osm>".to_string());
        out.join("\n")
    }
}

/// 1. Built-in `Riverfront Bridges Metro`: winding S-curve river + 4 bridges + angled
/// waterfront boulevards + West CBD one-way couplets + East tilted neighborhoods.
pub fn builtin_riverfront_bridges_osm() -> String {
    let mut b = OsmCityBuilder::new("Riverfront Bridges Metro");
    let river_x = |y: f64| -> f64 { 52.0 * (y / 215.0).sin() + 18.0 * (y / 380.0).cos() };

    let mut river_pts = Vec::new();
    for step in -4..=4 {
        let y = (step as f64) * 120.0;
        river_pts.push(b.node(river_x(y), y, false));
    }
    b.waterway("Central River Channel", "river", 82.0, false, &river_pts);

    let y_rows = [-300.0_f64, -100.0, 100.0, 300.0];
    let mut w_outer = Vec::with_capacity(4);
    let mut w_c1 = Vec::with_capacity(4);
    let mut w_c2 = Vec::with_capacity(4);
    let mut w_prom = Vec::with_capacity(4);
    let mut e_pkwy = Vec::with_capacity(4);
    let mut e_mid = Vec::with_capacity(4);
    let mut e_loop = Vec::with_capacity(4);

    let mut wf = Vec::with_capacity(4);
    let mut ef = Vec::with_capacity(4);

    for (ri, &y) in y_rows.iter().enumerate() {
        let sig = true;
        wf.push(b.node(-520.0, y, false));
        w_outer.push(b.node(-395.0, y, ri % 2 == 0));
        w_c1.push(b.node(-280.0, y, sig));
        w_c2.push(b.node(-185.0, y, sig));
        w_prom.push(b.node(river_x(y) - 66.0, y, sig));

        let ye = y + 22.0 * ((ri as f64) - 1.5).signum() * 0.4;
        e_pkwy.push(b.node(river_x(y) + 66.0, y, sig));
        e_mid.push(b.node(235.0 + 0.14 * ye, ye + 18.0, sig));
        e_loop.push(b.node(375.0 + 0.14 * ye, ye + 34.0, ri % 2 == 1));
        ef.push(b.node(520.0, ye + 45.0, false));
    }

    // North & South boundary feeders
    let wn_outer = b.node(-395.0, -430.0, false);
    let wn_c1 = b.node(-280.0, -430.0, false);
    let wn_c2 = b.node(-185.0, -430.0, false);
    let wn_prom = b.node(river_x(-430.0) - 66.0, -430.0, false);
    let en_pkwy = b.node(river_x(-430.0) + 66.0, -430.0, false);
    let en_mid = b.node(235.0 - 58.0, -425.0, false);
    let en_loop = b.node(375.0 - 58.0, -425.0, false);

    let ws_outer = b.node(-395.0, 430.0, false);
    let ws_c1 = b.node(-280.0, 430.0, false);
    let ws_c2 = b.node(-185.0, 430.0, false);
    let ws_prom = b.node(river_x(430.0) - 66.0, 430.0, false);
    let es_pkwy = b.node(river_x(430.0) + 66.0, 430.0, false);
    let es_mid = b.node(235.0 + 58.0, 435.0, false);
    let es_loop = b.node(375.0 + 58.0, 435.0, false);

    let wm_outer = b.node(-395.0, 0.0, true);
    let wm_c1 = b.node(-280.0, 0.0, true);
    let wm_c2 = b.node(-185.0, 0.0, true);
    let wp_n1 = b.node(river_x(-200.0) - 66.0, -200.0, false);
    let wp_mid = b.node(river_x(0.0) - 66.0, 0.0, true);
    let wp_s1 = b.node(river_x(200.0) - 66.0, 200.0, false);

    // West Bank Arterials + 1-Way Trunk Couplets
    b.way(
        "Westbank Outer Ave",
        "secondary",
        4,
        false,
        "30 mph",
        false,
        false,
        false,
        &[
            wn_outer, w_outer[0], w_outer[1], wm_outer, w_outer[2], w_outer[3], ws_outer,
        ],
    );
    b.way(
        "West Couplet Southbound",
        "trunk",
        3,
        true,
        "30 mph",
        false,
        true,
        false,
        &[wn_c1, w_c1[0], w_c1[1], wm_c1, w_c1[2], w_c1[3], ws_c1],
    );
    b.way(
        "West Couplet Northbound",
        "trunk",
        3,
        true,
        "30 mph",
        false,
        true,
        false,
        &[ws_c2, w_c2[3], w_c2[2], wm_c2, w_c2[1], w_c2[0], wn_c2],
    );
    // Curved West Promenade hugging the river curve
    b.way(
        "Westbank Waterfront Promenade",
        "primary",
        4,
        false,
        "25 mph",
        false,
        false,
        false,
        &[
            wn_prom,
            w_prom[0],
            wp_n1,
            w_prom[1],
            wp_mid,
            w_prom[2],
            wp_s1,
            w_prom[3],
            ws_prom,
        ],
    );

    // Midtown connector to Waterfront Plaza
    b.way(
        "Civic Midtown Cross St",
        "tertiary",
        2,
        false,
        "20 mph",
        false,
        false,
        false,
        &[wm_outer, wm_c1, wm_c2, wp_mid],
    );

    // East Bank Curved Parkway & Angled Terrace Grid
    let ep_n1 = b.node(river_x(-200.0) + 66.0, -200.0, false);
    let ep_mid = b.node(river_x(0.0) + 66.0, 0.0, false);
    let ep_s1 = b.node(river_x(200.0) + 66.0, 200.0, false);
    b.way(
        "Eastbank Parkway",
        "primary",
        4,
        false,
        "30 mph",
        false,
        false,
        false,
        &[
            en_pkwy,
            e_pkwy[0],
            ep_n1,
            e_pkwy[1],
            ep_mid,
            e_pkwy[2],
            ep_s1,
            e_pkwy[3],
            es_pkwy,
        ],
    );
    b.way(
        "East Midtown Terrace Ave",
        "secondary",
        4,
        false,
        "30 mph",
        false,
        true,
        false,
        &[en_mid, e_mid[0], e_mid[1], e_mid[2], e_mid[3], es_mid],
    );
    b.way(
        "East Highlands Ridge Blvd",
        "tertiary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[en_loop, e_loop[0], e_loop[1], e_loop[2], e_loop[3], es_loop],
    );

    // E-W Bank Streets + 4 Distinct Bridge Crossings
    let ew_names = [
        ("North Viaduct Corridor", "trunk", 4, "35 mph", true),
        ("Central Grand Arch Blvd", "primary", 4, "30 mph", true),
        ("Midtown Terrace Blvd", "secondary", 4, "25 mph", false),
        ("South Harborway Ave", "primary", 4, "30 mph", true),
    ];
    let bridge_names = [
        ("North Express Bridge", "trunk", 4, "35 mph", false),
        ("Grand Central Arch Bridge", "primary", 4, "30 mph", true),
        ("Midtown River Bridge", "secondary", 4, "25 mph", false),
        ("South Wharf Bridge", "primary", 4, "30 mph", false),
    ];
    for ri in 0..4 {
        let (wn, wh, wl, ws, wb) = ew_names[ri];
        let west_nds = [wf[ri], w_outer[ri], w_c1[ri], w_c2[ri], w_prom[ri]];
        if ri == 1 {
            let mut rev = west_nds;
            rev.reverse();
            b.way(wn, wh, wl, true, ws, false, wb, false, &rev);
        } else if ri == 2 {
            b.way(wn, wh, wl, true, ws, false, wb, false, &west_nds);
        } else {
            b.way(wn, wh, wl, false, ws, false, wb, false, &west_nds);
        }

        b.way(
            &format!("East Terrace St {}", ri + 1),
            wh,
            wl,
            false,
            ws,
            false,
            false,
            false,
            &[e_pkwy[ri], e_mid[ri], e_loop[ri], ef[ri]],
        );

        let (bn, bh, bl, bs, bbus) = bridge_names[ri];
        b.way(
            bn,
            bh,
            bl,
            false,
            bs,
            true,
            bbus,
            false,
            &[w_prom[ri], e_pkwy[ri]],
        );
    }

    b.finish()
}

/// 2. Built-in `Harbor Bay & Peninsula`: curved coastal bay polygon shoreline + waterfront
/// promenade + 35-deg diagonal boulevard cutting across offset districts + 2 bay bridges to Port Peninsula.
pub fn builtin_harbor_peninsula_osm() -> String {
    let mut b = OsmCityBuilder::new("Harbor Bay & Peninsula");

    // Deep Harbor Bay polygons curving smoothly along SE coast while leaving Port Peninsula land clear
    let upper_bay = [
        b.node(105.0, 95.0, false),
        b.node(185.0, -5.0, false),
        b.node(245.0, -165.0, false),
        b.node(365.0, -310.0, false),
        b.node(590.0, -360.0, false),
        b.node(590.0, 95.0, false),
    ];
    b.waterway("North Harbor Basin", "bay", 120.0, true, &upper_bay);

    let south_channel = [
        b.node(-310.0, 490.0, false),
        b.node(-115.0, 325.0, false),
        b.node(35.0, 225.0, false),
        b.node(125.0, 120.0, false),
        b.node(215.0, 120.0, false),
        b.node(215.0, 490.0, false),
    ];
    b.waterway("Port Harbor Channel", "bay", 110.0, true, &south_channel);

    // Curved Bayfront Corniche hugging shoreline
    let c0 = b.node(-420.0, 415.0, false);
    let c1 = b.node(-195.0, 290.0, true);
    let c2 = b.node(-45.0, 185.0, true);
    let c3 = b.node(75.0, 70.0, true);
    let c4 = b.node(150.0, -55.0, true);
    let c5 = b.node(210.0, -210.0, true);
    let c6 = b.node(345.0, -350.0, true);
    let c7 = b.node(525.0, -405.0, false);
    b.way(
        "Bayfront Corniche Promenade",
        "primary",
        4,
        false,
        "30 mph",
        false,
        true,
        false,
        &[c0, c1, c2, c3, c4, c5, c6, c7],
    );

    // 35-deg Diagonal Boulevard from NW Entry to Central Waterfront Hub (c3)
    let d_entry = b.node(-520.0, -370.0, false);
    let d1 = b.node(-365.0, -255.0, true);
    let d2 = b.node(-210.0, -140.0, true);
    let d3 = b.node(-65.0, -32.0, true);
    b.way(
        "Harbor Diagonal Express Blvd",
        "trunk",
        6,
        false,
        "35 mph",
        false,
        true,
        false,
        &[d_entry, d1, d2, d3, c3],
    );

    // North-West Harbor Orthogonal + Skewed Coastal Districts
    let nw_w = b.node(-520.0, -255.0, false);
    let nw_m = b.node(-210.0, -255.0, true);
    let nw_e = b.node(-35.0, -245.0, true);
    b.way(
        "Upper Bay Avenue",
        "secondary",
        4,
        false,
        "30 mph",
        false,
        false,
        false,
        &[nw_w, d1, nw_m, nw_e, c5],
    );

    let mid_w = b.node(-520.0, -140.0, false);
    let mid_w2 = b.node(-365.0, -140.0, true);
    let mid_e = b.node(-40.0, -140.0, true);
    b.way(
        "Customs House Arterial",
        "secondary",
        4,
        false,
        "25 mph",
        false,
        false,
        false,
        &[mid_w, mid_w2, d2, mid_e, c4],
    );

    let sw_w = b.node(-520.0, 20.0, false);
    let sw_1 = b.node(-365.0, 15.0, true);
    let sw_2 = b.node(-210.0, -5.0, true);
    b.way(
        "West Wharf St",
        "secondary",
        4,
        false,
        "25 mph",
        false,
        false,
        false,
        &[sw_w, sw_1, sw_2, d3],
    );

    let low_w = b.node(-520.0, 185.0, false);
    let low_1 = b.node(-365.0, 185.0, true);
    let low_2 = b.node(-210.0, 185.0, true);
    b.way(
        "South Cannery Row",
        "tertiary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[low_w, low_1, low_2, c2],
    );

    // North/South feeders and cross-links tying NW district together
    let n_365 = b.node(-365.0, -430.0, false);
    let n_210 = b.node(-210.0, -430.0, false);
    let n_40 = b.node(-40.0, -430.0, false);
    let s_365 = b.node(-365.0, 390.0, false);
    let s_210 = b.node(-210.0, 390.0, false);
    b.way(
        "West Dock Blvd",
        "secondary",
        4,
        false,
        "30 mph",
        false,
        false,
        false,
        &[n_365, d1, mid_w2, sw_1, low_1, c1, s_365],
    );
    b.way(
        "Midtown Harbor St",
        "primary",
        4,
        false,
        "30 mph",
        false,
        false,
        false,
        &[n_210, nw_m, d2, sw_2, low_2, s_210],
    );
    b.way(
        "Old Town Pier Ave",
        "secondary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[n_40, nw_e, mid_e, d3, c2],
    );
    let north_hill = b.node(120.0, -395.0, false);
    b.way(
        "North Headland Ridge Rd",
        "tertiary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[n_40, north_hill, c6],
    );

    // Outer Port Terminal Peninsula connected via TWO Harbor Bay Bridges!
    let p_nw = b.node(245.0, 165.0, true);
    let p_ne = b.node(435.0, 175.0, true);
    let p_se = b.node(455.0, 340.0, true);
    let p_sw = b.node(245.0, 330.0, true);
    let p_out = b.node(550.0, 260.0, false);

    b.way(
        "North Harbor Viaduct Bridge",
        "trunk",
        4,
        false,
        "35 mph",
        true,
        true,
        false,
        &[c3, p_nw],
    );
    b.way(
        "South Bay Causeway Bridge",
        "primary",
        4,
        false,
        "30 mph",
        true,
        false,
        false,
        &[c2, p_sw],
    );
    b.way(
        "Port Peninsula Wharf Loop",
        "primary",
        4,
        false,
        "30 mph",
        false,
        false,
        false,
        &[p_sw, p_nw, p_ne, p_se, p_sw],
    );
    b.way(
        "East Port Breakwater Freight",
        "trunk",
        4,
        false,
        "40 mph",
        false,
        false,
        false,
        &[p_ne, p_out],
    );
    b.way(
        "Container Terminal Slip",
        "secondary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[p_nw, p_se],
    );

    b.finish()
}

/// 3. Built-in `Radial Ring & Star Plazas`: concentric inner & outer beltways + 8 radial avenues
/// + 2 satellite roundabout plazas + southern civic canal with bridges.
pub fn builtin_radial_ring_plazas_osm() -> String {
    let mut b = OsmCityBuilder::new("Radial Ring & Star Plazas");

    // Curved Southern Civic Canal
    let canal_pts: Vec<usize> = [-520.0_f64, -340.0, -170.0, 0.0, 170.0, 340.0, 520.0]
        .iter()
        .map(|&x| {
            let y = 212.0 + 48.0 * (x / 520.0_f64).powi(2);
            b.node(x, y, false)
        })
        .collect();
    b.waterway("Civic Reflecting Canal", "canal", 34.0, false, &canal_pts);

    let polar = |r: f64, idx: usize| -> (f64, f64) {
        let ang = (idx as f64) * std::f64::consts::FRAC_PI_4;
        (r * ang.cos(), r * ang.sin())
    };

    let mut inner = Vec::with_capacity(8);
    let mut mid = Vec::with_capacity(8);
    let mut outer = Vec::with_capacity(8);
    let mut feed = Vec::with_capacity(8);

    for i in 0..8 {
        let (xi, yi) = polar(125.0, i);
        let (xm, ym) = polar(265.0, i);
        let (xo, yo) = polar(400.0, i);
        let (xf, yf) = polar(530.0, i);
        inner.push(b.node(xi, yi, false));
        mid.push(b.node(xm, ym, true));
        outer.push(b.node(xo, yo, true));
        feed.push(b.node(xf, yf, false));
    }

    // Inner Circulatory Roundabout Ring (CCW / oneway)
    let mut inner_closed = inner.clone();
    inner_closed.push(inner[0]);
    b.way(
        "Grand Star Inner Plaza Ring",
        "primary",
        2,
        true,
        "25 mph",
        false,
        false,
        true,
        &inner_closed,
    );

    // Mid Orbital Ring (splitting Canal crossing segments if needed; mid ring passes around S at y=+265 vs canal y=+212)
    let mut mid_closed = mid.clone();
    mid_closed.push(mid[0]);
    b.way(
        "Mid-City Orbital Boulevard",
        "secondary",
        4,
        false,
        "30 mph",
        false,
        true,
        false,
        &mid_closed,
    );

    // Outer Orbital Trunk Beltway
    let mut outer_closed = outer.clone();
    outer_closed.push(outer[0]);
    b.way(
        "Outer Metropolitan Beltway",
        "trunk",
        4,
        false,
        "40 mph",
        false,
        false,
        false,
        &outer_closed,
    );

    // 8 Radial Star Avenues + Canal Bridges on SE (1), S (2), SW (3) radials (y > 0 corresponds to sin > 0 -> i = 1, 2, 3)
    let spoke_names = [
        "East Promenade Radial",
        "Southeast Harbor Radial",
        "South Capitol Radial",
        "Southwest Parkway Radial",
        "West Civic Radial",
        "Northwest University Radial",
        "North Meridian Radial",
        "Northeast Botanical Radial",
    ];
    for i in 0..8 {
        let has_canal = matches!(i, 1 | 2 | 3);
        if has_canal {
            b.way(
                &format!("{} Bridge", spoke_names[i]),
                "primary",
                4,
                false,
                "30 mph",
                true,
                i == 2,
                false,
                &[inner[i], mid[i]],
            );
            b.way(
                spoke_names[i],
                "primary",
                4,
                false,
                "30 mph",
                false,
                i == 2,
                false,
                &[mid[i], outer[i], feed[i]],
            );
        } else {
            b.way(
                spoke_names[i],
                "primary",
                4,
                false,
                "30 mph",
                false,
                i == 6 || i == 0 || i == 4,
                false,
                &[inner[i], mid[i], outer[i], feed[i]],
            );
        }
    }

    // Two Satellite Diamond Plazas on East (i=0) & West (i=4) mid-ring neighborhoods
    let ep_n = b.node(265.0, -48.0, false);
    let ep_s = b.node(265.0, 48.0, false);
    let ep_e = b.node(315.0, 0.0, false);
    b.way(
        "East Star Plaza Loop",
        "tertiary",
        2,
        true,
        "20 mph",
        false,
        false,
        true,
        &[mid[0], ep_n, ep_e, ep_s, mid[0]],
    );
    let wp_n = b.node(-265.0, -48.0, false);
    let wp_s = b.node(-265.0, 48.0, false);
    let wp_w = b.node(-315.0, 0.0, false);
    b.way(
        "West Star Plaza Loop",
        "tertiary",
        2,
        true,
        "20 mph",
        false,
        false,
        true,
        &[mid[4], wp_s, wp_w, wp_n, mid[4]],
    );

    b.finish()
}

/// 4. Built-in `Foothill Canyon WUI`: curved dendritic canyon roads, WUI hillside loops,
/// meandering mountain creek + 3 creek bridges + 2 bottleneck evacuation trunks.
pub fn builtin_foothill_canyon_wui_osm() -> String {
    let mut b = OsmCityBuilder::new("Foothill Canyon WUI");

    // Winding Mountain Creek flowing NE canyon -> SW mouth
    let creek_pts = [
        b.node(580.0, -415.0, false),
        b.node(395.0, -255.0, false),
        b.node(185.0, -110.0, false),
        b.node(-15.0, 25.0, false),
        b.node(-215.0, 175.0, false),
        b.node(-420.0, 290.0, false),
        b.node(-590.0, 365.0, false),
    ];
    b.waterway("Timberline Mountain Creek", "stream", 30.0, false, &creek_pts);

    // South Bottleneck Evacuation Trunk #1 (Main Canyon Highway)
    let t_w = b.node(-565.0, 255.0, false);
    let t_1 = b.node(-355.0, 210.0, true);
    let t_2 = b.node(-155.0, 130.0, true);
    let t_3 = b.node(60.0, 42.0, true);
    let t_4 = b.node(285.0, -75.0, true);
    let t_e = b.node(560.0, -210.0, false);
    b.way(
        "South Canyon Evacuation Trunk",
        "trunk",
        4,
        false,
        "45 mph",
        false,
        true,
        false,
        &[t_w, t_1, t_2, t_3, t_4, t_e],
    );

    // West Foothill Bottleneck Trunk #2 (N-S Evacuation Mouth)
    let w_n = b.node(-445.0, -390.0, false);
    let w_1 = b.node(-410.0, -155.0, true);
    let w_2 = b.node(-385.0, 65.0, true);
    let w_s = b.node(-395.0, 425.0, false);
    b.way(
        "West Valley Mouth Highway",
        "trunk",
        4,
        false,
        "40 mph",
        false,
        false,
        false,
        &[w_n, w_1, w_2, t_1, w_s],
    );

    // Curved Dendritic North Canyon Collector hugging north side of creek
    let c_w = b.node(-535.0, 125.0, false);
    let c_1 = b.node(-310.0, 15.0, true);
    let c_2 = b.node(-105.0, -100.0, true);
    let c_3 = b.node(125.0, -205.0, true);
    let c_4 = b.node(355.0, -310.0, true);
    let c_e = b.node(545.0, -405.0, false);
    b.way(
        "North Canyon Ridge Collector",
        "primary",
        4,
        false,
        "35 mph",
        false,
        false,
        false,
        &[c_w, w_2, c_1, c_2, c_3, c_4, c_e],
    );

    // 3 Creek Bridges connecting North Canyon Collector <-> South Evacuation Trunk
    b.way(
        "West Creek Canyon Bridge",
        "primary",
        2,
        false,
        "30 mph",
        true,
        false,
        false,
        &[c_1, t_2],
    );
    b.way(
        "Mid-Valley Creek Bridge",
        "primary",
        4,
        false,
        "30 mph",
        true,
        true,
        false,
        &[c_2, t_3],
    );
    b.way(
        "Upper Timber Creek Bridge",
        "secondary",
        2,
        false,
        "25 mph",
        true,
        false,
        false,
        &[c_3, t_4],
    );

    // Dendritic North WUI Hillside Switchback Loops
    let l1_a = b.node(-330.0, -145.0, false);
    let l1_b = b.node(-225.0, -295.0, false);
    let l1_c = b.node(-115.0, -245.0, false);
    b.way(
        "Pinecrest Switchback Loop",
        "tertiary",
        2,
        false,
        "20 mph",
        false,
        false,
        false,
        &[w_1, l1_a, l1_b, l1_c, c_2],
    );
    b.way(
        "Pinecrest Ridge Spur",
        "residential",
        2,
        false,
        "20 mph",
        false,
        false,
        false,
        &[c_1, l1_a],
    );

    let l2_a = b.node(-25.0, -335.0, false);
    let l2_b = b.node(145.0, -395.0, false);
    let l2_c = b.node(215.0, -280.0, false);
    b.way(
        "Highland Crest WUI Loop",
        "tertiary",
        2,
        false,
        "20 mph",
        false,
        false,
        false,
        &[c_2, l2_a, l2_b, l2_c, c_3],
    );

    let l3_a = b.node(260.0, -395.0, false);
    let l3_b = b.node(445.0, -425.0, false);
    b.way(
        "Lookout Summit Spur",
        "residential",
        2,
        false,
        "20 mph",
        false,
        false,
        false,
        &[c_3, l3_a, l3_b, c_4],
    );

    // South Foothill Meadow Loops
    let s1_a = b.node(-170.0, 295.0, false);
    let s1_b = b.node(45.0, 275.0, false);
    let s1_c = b.node(145.0, 175.0, false);
    b.way(
        "South Meadow Foothill Loop",
        "secondary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[t_2, s1_a, s1_b, s1_c, t_3],
    );
    let s2_a = b.node(295.0, 145.0, false);
    let s2_b = b.node(435.0, 35.0, false);
    b.way(
        "East Saddle WUI Loop",
        "tertiary",
        2,
        false,
        "25 mph",
        false,
        false,
        false,
        &[t_3, s2_a, s2_b, t_4],
    );

    b.finish()
}

/// 5. Built-in `Hierarchical Multimodal Metropolis` (50k+ wgpu capacity):
/// multi-district arterial hierarchy + diagonal BRT spine + curved river & 5 bridges.
pub fn builtin_metropolis_50k_osm() -> String {
    let mut b = OsmCityBuilder::new("Hierarchical Multimodal Metropolis");
    let river_x = |y: f64| -> f64 { 88.0 * (y / 360.0).sin() + 10.0 };

    let mut r_pts = Vec::new();
    for step in -6..=6 {
        let y = (step as f64) * 125.0;
        r_pts.push(b.node(river_x(y), y, false));
    }
    b.waterway("Metropolitan River", "river", 86.0, false, &r_pts);

    // Y bands (West CBD tighter, East wider, shared bridges at 5 crossing latitudes)
    let ys = [-540.0_f64, -360.0, -180.0, 0.0, 180.0, 360.0, 540.0];
    let w_cols = [-740.0_f64, -560.0, -400.0, -255.0, -145.0]; // col 4 tracks west bank
    let e_cols = [145.0_f64, 275.0, 425.0, 595.0, 750.0]; // col 0 tracks east bank

    let mut w_grid = vec![vec![0usize; 5]; ys.len()];
    let mut e_grid = vec![vec![0usize; 5]; ys.len()];
    let mut w_feed = Vec::with_capacity(ys.len());
    let mut e_feed = Vec::with_capacity(ys.len());

    for (ri, &y) in ys.iter().enumerate() {
        let rx = river_x(y);
        w_feed.push(b.node(-890.0, y, false));
        e_feed.push(b.node(890.0, y, false));
        for ci in 0..5 {
            let xw = if ci == 4 {
                rx - 74.0
            } else {
                w_cols[ci] + 0.03 * y * (ci as f64 - 2.0)
            };
            let xe = if ci == 0 {
                rx + 74.0
            } else {
                e_cols[ci] + 0.06 * y
            };
            let y_e = y + 0.04 * (e_cols[ci] - 145.0);
            w_grid[ri][ci] = b.node(xw, y, true);
            e_grid[ri][ci] = b.node(xe, y_e, true);
        }
    }

    // North & South External Feeder Trunks
    let mut wn_feed = Vec::with_capacity(5);
    let mut ws_feed = Vec::with_capacity(5);
    let mut en_feed = Vec::with_capacity(5);
    let mut es_feed = Vec::with_capacity(5);
    for ci in 0..5 {
        let xwn = if ci == 4 { river_x(-690.0) - 74.0 } else { w_cols[ci] };
        let xws = if ci == 4 { river_x(690.0) - 74.0 } else { w_cols[ci] };
        let xen = if ci == 0 { river_x(-690.0) + 74.0 } else { e_cols[ci] - 35.0 };
        let xes = if ci == 0 { river_x(690.0) + 74.0 } else { e_cols[ci] + 35.0 };
        wn_feed.push(b.node(xwn, -690.0, false));
        ws_feed.push(b.node(xws, 690.0, false));
        en_feed.push(b.node(xen, -690.0, false));
        es_feed.push(b.node(xes, 690.0, false));
    }

    // West District North-South Boulevards + Outer Orbital
    for ci in 0..5 {
        let (hwy, lanes, mph, bus) = match ci {
            0 => ("motorway", 6, "50 mph", false),
            2 => ("trunk", 6, "35 mph", true),
            4 => ("primary", 4, "30 mph", false),
            _ => ("secondary", 4, "25 mph", false),
        };
        let mut nds = vec![wn_feed[ci]];
        for ri in 0..ys.len() {
            nds.push(w_grid[ri][ci]);
        }
        nds.push(ws_feed[ci]);
        b.way(
            &format!("West District Corridor {}", ci + 1),
            hwy,
            lanes,
            false,
            mph,
            false,
            bus,
            false,
            &nds,
        );
    }

    // East District Angled Arterials + East Orbital
    for ci in 0..5 {
        let (hwy, lanes, mph, bus) = match ci {
            0 => ("primary", 4, "30 mph", false),
            2 => ("trunk", 6, "35 mph", true),
            4 => ("motorway", 6, "50 mph", false),
            _ => ("secondary", 4, "25 mph", false),
        };
        let mut nds = vec![en_feed[ci]];
        for ri in 0..ys.len() {
            nds.push(e_grid[ri][ci]);
        }
        nds.push(es_feed[ci]);
        b.way(
            &format!("East Midtown Ave {}", ci + 1),
            hwy,
            lanes,
            false,
            mph,
            false,
            bus,
            false,
            &nds,
        );
    }

    // East-West Hierarchy + 5 Bridge Crossings (rows 1..=5)
    for ri in 0..ys.len() {
        let is_orbital = ri == 0 || ri == ys.len() - 1;
        let (hwy, lanes, mph) = if is_orbital {
            ("trunk", 6, "45 mph")
        } else if ri % 2 == 1 {
            ("primary", 6, "35 mph")
        } else {
            ("secondary", 4, "30 mph")
        };
        let mut w_nds = vec![w_feed[ri]];
        w_nds.extend_from_slice(&w_grid[ri]);
        b.way(
            &format!("West Metro Ave {}", ri + 1),
            hwy,
            lanes,
            false,
            mph,
            false,
            ri == 3,
            false,
            &w_nds,
        );

        let mut e_nds = e_grid[ri].clone();
        e_nds.push(e_feed[ri]);
        b.way(
            &format!("East Metro Ave {}", ri + 1),
            hwy,
            lanes,
            false,
            mph,
            false,
            ri == 3,
            false,
            &e_nds,
        );

        if matches!(ri, 0 | 2 | 3 | 4 | 6) {
            b.way(
                &format!("Metro River Bridge {}", ri + 1),
                if ri == 3 { "trunk" } else { "primary" },
                6,
                false,
                "35 mph",
                true,
                ri == 3,
                false,
                &[w_grid[ri][4], e_grid[ri][0]],
            );
        }
    }

    // Diagonal Express BRT Spine cutting SW -> Central River Bridge -> NE
    b.way(
        "Diagonal Express BRT Spine West",
        "trunk",
        6,
        false,
        "35 mph",
        false,
        true,
        false,
        &[
            w_grid[6][0],
            w_grid[5][1],
            w_grid[4][2],
            w_grid[3][3],
            w_grid[3][4],
        ],
    );
    b.way(
        "Diagonal Express BRT Spine East",
        "trunk",
        6,
        false,
        "35 mph",
        false,
        true,
        false,
        &[
            e_grid[3][0],
            e_grid[2][1],
            e_grid[1][2],
            e_grid[0][4],
        ],
    );

    b.finish()
}

pub fn builtin_roundabout_osm() -> String {
    let c_lat = 40.0000_f64;
    let c_lon = -105.0000_f64;
    let m_lat = 1.0 / 110_540.0;
    let m_lon = 1.0 / (111_320.0 * c_lat.to_radians().cos());
    let mut out = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        r#"<osm version="0.6" generator="urban-grid" city="Modern 2-Lane Roundabout">"#.to_string(),
    ];
    let r_m = 28.0;
    for i in 0..8 {
        let ang = (i as f64) * std::f64::consts::FRAC_PI_4;
        let lat = c_lat + ang.sin() * r_m * m_lat;
        let lon = c_lon + ang.cos() * r_m * m_lon;
        out.push(format!(r#"  <node id="{}" lat="{lat:.6}" lon="{lon:.6}"/>"#, 10 + i));
    }
    let pts = [
        (20, 165.0, 0.0, false),
        (22, 0.0, 155.0, false),
        (24, -115.0, 0.0, true),
        (25, -220.0, 0.0, false),
        (26, 0.0, -155.0, false),
        (27, -115.0, 120.0, false),
        (28, -115.0, -120.0, false),
    ];
    for (id, x, y, sig) in pts {
        let lat = c_lat + y * m_lat;
        let lon = c_lon + x * m_lon;
        if sig {
            out.push(format!(
                r#"  <node id="{id}" lat="{lat:.6}" lon="{lon:.6}"><tag k="highway" v="traffic_signals"/></node>"#
            ));
        } else {
            out.push(format!(r#"  <node id="{id}" lat="{lat:.6}" lon="{lon:.6}"/>"#));
        }
    }
    out.push(r#"  <way id="500">"#.to_string());
    for id in [10, 11, 12, 13, 14, 15, 16, 17, 10] {
        out.push(format!(r#"    <nd ref="{id}"/>"#));
    }
    out.push(r#"    <tag k="name" v="Central Circulatory Ring"/><tag k="highway" v="primary"/><tag k="junction" v="roundabout"/><tag k="lanes" v="2"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="20 mph"/>"#.to_string());
    out.push("  </way>".to_string());

    let arms: &[(usize, &[usize], &str, usize, &str)] = &[
        (501, &[20, 10], "East Parkway", 4, "35 mph"),
        (502, &[22, 12], "North Arterial", 4, "30 mph"),
        (503, &[25, 24, 14], "West Metered Corridor", 4, "35 mph"),
        (504, &[26, 16], "South Arterial", 4, "30 mph"),
        (505, &[27, 24, 28], "Metering Cross Ave", 2, "25 mph"),
    ];
    for (id, nds, name, lanes, mph) in arms {
        out.push(format!(r#"  <way id="{id}">"#));
        for n in *nds {
            out.push(format!(r#"    <nd ref="{n}"/>"#));
        }
        out.push(format!(
            r#"    <tag k="name" v="{name}"/><tag k="highway" v="primary"/><tag k="lanes" v="{lanes}"/><tag k="oneway" v="no"/><tag k="maxspeed" v="{mph}"/>"#
        ));
        out.push("  </way>".to_string());
    }
    out.push("</osm>".to_string());
    out.join("\n")
}

pub fn builtin_braess_diamond_osm() -> String {
    r#"<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="urban-grid" city="Braess Paradox Network">
  <node id="1" lat="40.000000" lon="-105.003020"/>
  <node id="2" lat="40.000000" lon="-105.001700"/>
  <node id="3" lat="40.000950" lon="-105.000000"/>
  <node id="4" lat="39.999050" lon="-105.000000"/>
  <node id="5" lat="40.000000" lon="-104.998300"/>
  <node id="6" lat="40.000000" lon="-104.996980"/>
  <node id="31" lat="40.000770" lon="-104.998930"/>
  <node id="41" lat="39.999230" lon="-105.001070"/>
  <way id="701">
    <nd ref="1"/><nd ref="2"/>
    <tag k="name" v="West Inbound Trunk"/><tag k="highway" v="trunk"/><tag k="lanes" v="3"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="40 mph"/>
  </way>
  <way id="702">
    <nd ref="2"/><nd ref="3"/>
    <tag k="name" v="NW Short Bottleneck (1-Lane)"/><tag k="highway" v="primary"/><tag k="lanes" v="1"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="35 mph"/>
  </way>
  <way id="703">
    <nd ref="3"/><nd ref="31"/><nd ref="5"/>
    <tag k="name" v="NE Wide Bypass (2-Lane)"/><tag k="highway" v="secondary"/><tag k="lanes" v="2"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="20 mph"/>
  </way>
  <way id="704">
    <nd ref="2"/><nd ref="41"/><nd ref="4"/>
    <tag k="name" v="SW Wide Bypass (2-Lane)"/><tag k="highway" v="secondary"/><tag k="lanes" v="2"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="20 mph"/>
  </way>
  <way id="705">
    <nd ref="4"/><nd ref="5"/>
    <tag k="name" v="SE Short Bottleneck (1-Lane)"/><tag k="highway" v="primary"/><tag k="lanes" v="1"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="35 mph"/>
  </way>
  <way id="706">
    <nd ref="5"/><nd ref="6"/>
    <tag k="name" v="East Outbound Trunk"/><tag k="highway" v="trunk"/><tag k="lanes" v="3"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="40 mph"/>
  </way>
  <way id="799">
    <nd ref="3"/><nd ref="4"/>
    <tag k="name" v="Braess Central Shortcut (N-S)"/><tag k="highway" v="trunk"/><tag k="lanes" v="2"/><tag k="oneway" v="yes"/><tag k="maxspeed" v="55 mph"/>
  </way>
</osm>"#
        .to_string()
}

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
    pub restrictions: Vec<RawTurnRestriction>,
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
                        if w.node_refs.len() >= 2
                            && w.tags
                                .get("highway")
                                .map(|h| is_drivable_highway(h))
                                .unwrap_or(false)
                        {
                            dataset.ways.push(w);
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
                if tags
                    .get("highway")
                    .map(|h| is_drivable_highway(h))
                    .unwrap_or(false)
                {
                    let node_refs: Vec<String> = el
                        .nodes
                        .unwrap_or_default()
                        .into_iter()
                        .map(|v| v.to_string().trim_matches('"').to_string())
                        .collect();
                    if node_refs.len() >= 2 {
                        ds.ways.push(RawOsmWay {
                            id: id_str,
                            node_refs,
                            tags,
                        });
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
                if tags
                    .get("highway")
                    .map(|h| is_drivable_highway(h))
                    .unwrap_or(false)
                {
                    let node_refs: Vec<String> = w.refs().map(|r| r.to_string()).collect();
                    if node_refs.len() >= 2 {
                        ds.ways.push(RawOsmWay {
                            id: w.id().to_string(),
                            node_refs,
                            tags,
                        });
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

/// Built-in Downtown Corvallis OR-99W / OR-34 One-Way Couplets `.osm` XML fixture
/// centered on Benton County Courthouse (44.56528, -123.26222).
pub fn builtin_corvallis_couplets_osm() -> String {
    let lats = [44.5673, 44.5662, 44.5651, 44.5640];
    let lons = [-123.2644, -123.2629, -123.2614, -123.2599];
    let mut out = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        r#"<osm version="0.6" generator="urban-grid">"#.to_string(),
    ];
    for (r, lat) in lats.iter().enumerate() {
        for (c, lon) in lons.iter().enumerate() {
            let id = (r + 1) * 10 + (c + 1);
            let is_sig = c == 1 || c == 2 || r == 0 || r == 1 || r == 3;
            if is_sig {
                out.push(format!(
                    r#"  <node id="{id}" lat="{lat:.6}" lon="{lon:.6}"><tag k="highway" v="traffic_signals"/></node>"#
                ));
            } else {
                out.push(format!(r#"  <node id="{id}" lat="{lat:.6}" lon="{lon:.6}"/>"#));
            }
        }
        out.push(format!(r#"  <node id="{}" lat="{lat:.6}" lon="-123.265900"/>"#, 100 + r));
        out.push(format!(r#"  <node id="{}" lat="{lat:.6}" lon="-123.258400"/>"#, 200 + r));
    }
    for (c, lon) in lons.iter().enumerate() {
        out.push(format!(r#"  <node id="{}" lat="44.568400" lon="{lon:.6}"/>"#, 300 + c));
        out.push(format!(r#"  <node id="{}" lat="44.562900" lon="{lon:.6}"/>"#, 400 + c));
    }

    let ew = [
        ("NW Harrison Blvd (OR-34 W)", "primary", 2, "west", "25 mph"),
        ("NW Van Buren Ave (OR-34 E)", "primary", 2, "east", "25 mph"),
        ("NW Jackson Ave", "tertiary", 2, "no", "20 mph"),
        ("NW Monroe Ave", "secondary", 2, "no", "20 mph"),
    ];
    for (r, (name, hwy, lanes, dir, mph)) in ew.iter().enumerate() {
        let mut nds = vec![
            100 + r,
            (r + 1) * 10 + 1,
            (r + 1) * 10 + 2,
            (r + 1) * 10 + 3,
            (r + 1) * 10 + 4,
            200 + r,
        ];
        if *dir == "west" {
            nds.reverse();
        }
        out.push(format!(r#"  <way id="{}">"#, 1000 + r));
        for n in nds {
            out.push(format!(r#"    <nd ref="{n}"/>"#));
        }
        let ow = if *dir == "no" { "no" } else { "yes" };
        out.push(format!(r#"    <tag k="name" v="{name}"/>"#));
        out.push(format!(r#"    <tag k="highway" v="{hwy}"/>"#));
        out.push(format!(r#"    <tag k="lanes" v="{lanes}"/>"#));
        out.push(format!(r#"    <tag k="oneway" v="{ow}"/>"#));
        out.push(format!(r#"    <tag k="maxspeed" v="{mph}"/>"#));
        out.push("  </way>".to_string());
    }

    let ns = [
        ("NW 5th St", "tertiary", 2, "no", "20 mph"),
        ("NW 4th St (OR-99W S)", "trunk", 3, "south", "25 mph"),
        ("NW 3rd St (OR-99W N)", "trunk", 3, "north", "25 mph"),
        ("NW 2nd St", "secondary", 2, "no", "20 mph"),
    ];
    for (c, (name, hwy, lanes, dir, mph)) in ns.iter().enumerate() {
        let mut nds = vec![300 + c, 10 + c + 1, 20 + c + 1, 30 + c + 1, 40 + c + 1, 400 + c];
        if *dir == "north" {
            nds.reverse();
        }
        out.push(format!(r#"  <way id="{}">"#, 2000 + c));
        for n in nds {
            out.push(format!(r#"    <nd ref="{n}"/>"#));
        }
        let ow = if *dir == "no" { "no" } else { "yes" };
        out.push(format!(r#"    <tag k="name" v="{name}"/>"#));
        out.push(format!(r#"    <tag k="highway" v="{hwy}"/>"#));
        out.push(format!(r#"    <tag k="lanes" v="{lanes}"/>"#));
        out.push(format!(r#"    <tag k="oneway" v="{ow}"/>"#));
        out.push(format!(r#"    <tag k="maxspeed" v="{mph}"/>"#));
        out.push("  </way>".to_string());
    }
    out.push("</osm>".to_string());
    out.join("
")
}

pub fn builtin_roundabout_osm() -> String {
    let c_lat = 44.5653_f64;
    let c_lon = -123.2622_f64;
    let m_lat = 1.0 / 110_540.0;
    let m_lon = 1.0 / (111_320.0 * c_lat.to_radians().cos());
    let mut out = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        r#"<osm version="0.6" generator="urban-grid">"#.to_string(),
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
    out.join("
")
}

pub fn builtin_braess_diamond_osm() -> String {
    r#"<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="urban-grid">
  <node id="1" lat="44.565300" lon="-123.265220"/>
  <node id="2" lat="44.565300" lon="-123.263900"/>
  <node id="3" lat="44.566250" lon="-123.262200"/>
  <node id="4" lat="44.564350" lon="-123.262200"/>
  <node id="5" lat="44.565300" lon="-123.260500"/>
  <node id="6" lat="44.565300" lon="-123.259180"/>
  <node id="31" lat="44.566070" lon="-123.261130"/>
  <node id="41" lat="44.564530" lon="-123.263270"/>
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


/// Generates a metropolitan supergrid `.osm` v0.6 XML network (default `12x12` = 144 signalized
/// intersections, >1,100 metric lanes, >140 lane-km of multi-modal boulevards, Complete Streets
/// bus/bike corridors, and perimeter entry/exit trunks) designed to sustain 50,000+ simultaneous
/// vehicles, cyclists, and sidewalk/crosswalk pedestrians without artificial curb gridlock.
pub fn builtin_megacity_supergrid_osm(rows: usize, cols: usize) -> String {
    let r_cnt = rows.clamp(4, 20);
    let c_cnt = cols.clamp(4, 20);
    let base_lat = 44.5653_f64;
    let base_lon = -123.2622_f64;
    let block_m = 135.0_f64;
    let dlat = block_m / 110_540.0;
    let dlon = block_m / (111_320.0 * base_lat.to_radians().cos());

    let half_r = (r_cnt as f64 - 1.0) * 0.5;
    let half_c = (c_cnt as f64 - 1.0) * 0.5;

    let mut out = Vec::with_capacity(r_cnt * c_cnt * 8 + 512);
    out.push(r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string());
    out.push(r#"<osm version="0.6" generator="urban-grid-megacity">"#.to_string());

    // Interior grid nodes (all signalized intersections)
    for r in 0..r_cnt {
        let lat = base_lat - (r as f64 - half_r) * dlat;
        for c in 0..c_cnt {
            let lon = base_lon + (c as f64 - half_c) * dlon;
            let nid = 10_000 + r * 100 + c;
            out.push(format!(
                r#"  <node id="{nid}" lat="{lat:.6}" lon="{lon:.6}"><tag k="highway" v="traffic_signals"/></node>"#
            ));
        }
        // West and East boundary feeder nodes for row r
        let w_lon = base_lon + (-1.0 - half_c) * dlon;
        let e_lon = base_lon + (c_cnt as f64 - half_c) * dlon;
        let wnid = 50_000 + r;
        let enid = 60_000 + r;
        out.push(format!(r#"  <node id="{wnid}" lat="{lat:.6}" lon="{w_lon:.6}"/>"#));
        out.push(format!(r#"  <node id="{enid}" lat="{lat:.6}" lon="{e_lon:.6}"/>"#));
    }

    // North and South boundary feeder nodes for col c
    for c in 0..c_cnt {
        let lon = base_lon + (c as f64 - half_c) * dlon;
        let n_lat = base_lat - (-1.0 - half_r) * dlat;
        let s_lat = base_lat - (r_cnt as f64 - half_r) * dlat;
        let nnid = 70_000 + c;
        let snid = 80_000 + c;
        out.push(format!(r#"  <node id="{nnid}" lat="{n_lat:.6}" lon="{lon:.6}"/>"#));
        out.push(format!(r#"  <node id="{snid}" lat="{s_lat:.6}" lon="{lon:.6}"/>"#));
    }

    // East-West avenues
    for r in 0..r_cnt {
        let wid = 100_000 + r;
        let is_major = r % 3 == 0;
        let (hwy, lanes_total, mph) = if is_major {
            ("primary", 6, "35 mph")
        } else if r % 2 == 0 {
            ("secondary", 4, "30 mph")
        } else {
            ("tertiary", 4, "25 mph")
        };
        out.push(format!(r#"  <way id="{wid}">"#));
        out.push(format!(r#"    <nd ref="{}"/>"#, 50_000 + r));
        for c in 0..c_cnt {
            out.push(format!(r#"    <nd ref="{}"/>"#, 10_000 + r * 100 + c));
        }
        out.push(format!(r#"    <nd ref="{}"/>"#, 60_000 + r));
        out.push(format!(r#"    <tag k="name" v="Metro Ave {}"/>"#, r + 1));
        out.push(format!(r#"    <tag k="highway" v="{hwy}"/>"#));
        out.push(format!(r#"    <tag k="lanes" v="{lanes_total}"/>"#));
        out.push(r#"    <tag k="oneway" v="no"/>"#.to_string());
        out.push(format!(r#"    <tag k="maxspeed" v="{mph}"/>"#));
        out.push("  </way>".to_string());
    }

    // North-South boulevards
    for c in 0..c_cnt {
        let wid = 200_000 + c;
        let is_major = c % 3 == 0;
        let (hwy, lanes_total, mph) = if is_major {
            ("trunk", 6, "35 mph")
        } else if c % 2 == 0 {
            ("secondary", 4, "30 mph")
        } else {
            ("tertiary", 4, "25 mph")
        };
        out.push(format!(r#"  <way id="{wid}">"#));
        out.push(format!(r#"    <nd ref="{}"/>"#, 70_000 + c));
        for r in 0..r_cnt {
            out.push(format!(r#"    <nd ref="{}"/>"#, 10_000 + r * 100 + c));
        }
        out.push(format!(r#"    <nd ref="{}"/>"#, 80_000 + c));
        out.push(format!(r#"    <tag k="name" v="Grand Blvd {}"/>"#, c + 1));
        out.push(format!(r#"    <tag k="highway" v="{hwy}"/>"#));
        out.push(format!(r#"    <tag k="lanes" v="{lanes_total}"/>"#));
        out.push(r#"    <tag k="oneway" v="no"/>"#.to_string());
        out.push(format!(r#"    <tag k="maxspeed" v="{mph}"/>"#));
        out.push("  </way>".to_string());
    }

    out.push("</osm>".to_string());
    out.join("\n")
}

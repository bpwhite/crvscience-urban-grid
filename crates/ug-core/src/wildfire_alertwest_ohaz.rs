//! `ALERTWest` (`https://www.alertwest.org/` / `https://alertwest.live/`) & University of Oregon
//! `Oregon Hazards Lab (OHAZ)` (`https://ohaz.uoregon.edu/wildfire-cameras/`) AI PTZ Wildfire
//! Camera Detection, Cross-Azimuth Triangulation, and `NWCG` / Rothermel WUI Wildfire Suppression:
//! - **Native Multi-Format Camera & Ignition Feed Linker (`.json`, GeoJSON `FeatureCollection`, `.csv`)**:
//!   Ingests `ALERTWest` and `UOregon OHAZ` (`Alert OR`) camera metadata (`360°` pan, `220°` tilt,
//!   up to `40x` optical zoom, `30 mi` / `48.3 km` daytime & `40 mi` / `64.4 km` near-infrared `NIR`
//!   nighttime detection range, `2.0-min` panoramic AI scan cadence, co-located `PNSN`/`ShakeAlert`
//!   seismic telemetry, and `< 50 lb` Starlink + Li-ion solar **Mobile Deployment Units (`MDU`s)**).
//! - **Weighted Least-Squares (`WLS`) Multi-Camera Cross-Azimuth Bearing Triangulation**:
//!   Intersects `K >= 2` optical/`NIR` camera smoke bearings `(x_k, y_k, theta_k, zoom_k)` to
//!   pinpoint WUI ignitions `(x_hat, y_hat)` and `CEP_90` error radii (`meters`) within `2–4 min`
//!   vs. delayed `911` visual reporting (`18–52 min`).
//! - **Rothermel (1972) / `NWCG` (`RMRS-GTR-253`) WUI Surface-to-Crown Fire Behavior & Suppression**:
//!   Evaluates Rate of Spread (`chains/hr`), Byram Fireline Intensity (`Btu/ft/s`), Thomas Flame
//!   Length (`ft`), `NWCG` sustained line construction for **`20`-person Type 1 Hotshot Crews (`IHC`)**
//!   and **Wildland Bulldozers**, **Type 3/6 WUI Engine** structure triage, **Helitack Bucket Drops**,
//!   and **`DC-10 VLAT` / `LAT` `Phos-Chek` Long-Term Retardant Lines (`1.47–2.21 gpc`)**, coupled
//!   directly to `IPAWS` Level 1/2/3 staged neighborhood evacuation!

use crate::emergency::{EmergencyFacility, EmergencyFacilityKind};
use crate::network::{BoundingBox, RoadNetwork, Vec2};
use crate::zoning::{Parcel, ZoneCode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

pub const WILDFIRE_CAMERA_CSV_HEADER: &str = "camera_id,network_provider,camera_kind,lat,lon,x_m,y_m,elevation_m,azimuth_deg,optical_zoom_x,day_range_mi,nir_night_range_mi,pnsn_shakealert_colocated,detected_bearing_deg,ai_smoke_confidence,live_image_url";
pub const WILDFIRE_TIMELINE_CSV_HEADER: &str = "minute,active_fire_area_acres,uncontained_perimeter_ft,built_containment_line_ft,containment_pct,flame_length_ft,evacuated_wui_residents";

/// Camera platform class across the `ALERTWest` & `UOregon OHAZ` (`Alert OR`) ecosystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertWestOhazCameraKind {
    /// `UOregon OHAZ` mountaintop `360°` PTZ (`40x` optical zoom, `30 mi` day / `40 mi` `NIR` night,
    /// `2-min` AI panoramic scan, often co-located with `PNSN`/`ShakeAlert` seismic sensors).
    OhazMountaintopFixedPtz,
    /// `ALERTWest` municipal / utility WUI lookout tower camera (`360°` PTZ, `2-min` AI scan).
    AlertWestWuiLookoutTower,
    /// `UOregon OHAZ` portable **Mobile Deployment Unit (`MDU`)** (`< 50 lb` per component, `2–4`
    /// firefighter crew setup, Li-ion + solar power, and **Starlink** LEO satellite backhaul).
    OhazStarlinkMobileDeploymentUnit,
}

impl AlertWestOhazCameraKind {
    pub fn badge(self) -> &'static str {
        match self {
            Self::OhazMountaintopFixedPtz => "OHAZ-PTZ",
            Self::AlertWestWuiLookoutTower => "AWEST-WUI",
            Self::OhazStarlinkMobileDeploymentUnit => "OHAZ-MDU",
        }
    }
}

/// Canonical `ALERTWest` / `UOregon OHAZ` camera telemetry & smoke-bearing record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertWestOhazCameraSite {
    pub camera_id: String,
    pub network_provider: String,
    pub kind: AlertWestOhazCameraKind,
    pub lat: f64,
    pub lon: f64,
    #[serde(default)]
    pub world_pos: Vec2,
    pub elevation_m: f64,
    pub pan_azimuth_deg: f64,
    pub fov_deg: f64,
    pub tilt_deg: f64,
    pub optical_zoom_x: f64,
    pub day_range_mi: f64,
    pub nir_night_range_mi: f64,
    pub scan_interval_min: f64,
    pub co_located_pnsn_shakealert: bool,
    pub live_image_endpoint: String,
    pub portal_url: String,
    #[serde(default)]
    pub latest_ai_smoke_confidence: Option<f64>,
    #[serde(default)]
    pub latest_detected_bearing_deg: Option<f64>,
}

/// Detection & interagency suppression posture for WUI wildfire response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WildfireDetectionAndAttackPosture {
    /// Legacy public `911` visual smoke calls (`18–48 min` delay) + ground engines/handcrews only.
    Legacy911GroundOnly,
    /// Fixed `ALERTWest` + `UOregon OHAZ` mountaintop AI PTZ `2-min` scan triangulation + ground initial attack.
    AlertWestOhazFixedPtzInitialAttack,
    /// Full `ALERTWest` + `OHAZ` fixed PTZ + portable Starlink `MDU` blind-spot fill + `VLAT`/`LAT`
    /// `Phos-Chek` retardant + Helitack + Type-1 Hotshots/Dozers + `IPAWS` WUI evacuation!
    FullAlertWestOhazMduAndAerialAttack,
}

impl WildfireDetectionAndAttackPosture {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Legacy911GroundOnly => "Legacy 911 Call + Ground Only",
            Self::AlertWestOhazFixedPtzInitialAttack => "ALERTWest/OHAZ 2-Min AI PTZ + Ground IA",
            Self::FullAlertWestOhazMduAndAerialAttack => {
                "ALERTWest/OHAZ + Starlink MDU + VLAT Retardant + IHC"
            }
        }
    }
}

/// Standard Anderson (1982) / Rothermel (1972) WUI Fire Behavior Fuel Models (`FBFM`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WuiFuelModel {
    /// `FBFM 1`: Cured short grass / oak savanna (`0.74 tons/ac` fine herbaceous load, rapid ROS).
    ShortGrassFm1,
    /// `FBFM 5`: Green/cured WUI brush, blackberry & regenerating shrub (`2.0 tons/ac`).
    BrushWuiFm5,
    /// `FBFM 4`: Heavy volatile chaparral / mature shrub / slash (`5.0 tons/ac`, extreme flame length).
    HeavyChaparralSlashFm4,
    /// `FBFM 10`: Pacific Northwest closed conifer timber with heavy dead-and-down understory & ladder fuels (`12.0 tons/ac`).
    ConiferTimberUnderstoryFm10,
}

impl WuiFuelModel {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::ShortGrassFm1 => "FM1 Cured Grass / Oak Savanna",
            Self::BrushWuiFm5 => "FM5 WUI Brush / Shrub",
            Self::HeavyChaparralSlashFm4 => "FM4 Heavy Chaparral / Slash",
            Self::ConiferTimberUnderstoryFm10 => "FM10 Conifer Timber + Ladder Fuels",
        }
    }
}

/// USDA Forest Service `RMRS-GTR-253` Fire Suppression Interpretation Class from Flame Length (`ft`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NwcgSuppressionThresholdClass {
    /// `L_f <= 4.0 ft` (`< 100 Btu/ft/s`): Direct hand-tool attack at head/flanks by Hotshot crews.
    DirectHandToolsUnder4Ft,
    /// `4.0 < L_f <= 8.0 ft` (`100–500 Btu/ft/s`): Too intense for hand tools; Dozers, Engines & Retardant hold flanks.
    EquipmentAndRetardant4To8Ft,
    /// `8.0 < L_f <= 11.0 ft` (`500–1,000 Btu/ft/s`): Torching & spotting; indirect parallel control lines + VLAT drops.
    IndirectParallel8To11Ft,
    /// `L_f > 11.0 ft` (`> 1,000 Btu/ft/s`): Active crowning & long-range spotting; WUI life-safety triage & evacuation.
    CrowningExtremeOver11Ft,
}

impl NwcgSuppressionThresholdClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::DirectHandToolsUnder4Ft => "<=4 ft: Direct Handcrew Line",
            Self::EquipmentAndRetardant4To8Ft => "4-8 ft: Dozer/Engine/Aerial Flank",
            Self::IndirectParallel8To11Ft => "8-11 ft: Torching — Indirect + VLAT",
            Self::CrowningExtremeOver11Ft => ">11 ft: Crowning — WUI Evac Priority",
        }
    }
}

/// WUI Structure Protection Triage category per `NWCG` Wildland Urban Interface Operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WuiStructureTriageCategory {
    SafeOutsideZone,
    DefensibleStandalone,
    DefensibleWithEnginePrep,
    NonDefensibleImmediateEvac,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RothermelFireBehaviorMetrics {
    pub fine_dead_fuel_moisture_pct: f64,
    pub rate_of_spread_chains_per_hr: f64,
    pub rate_of_spread_m_per_min: f64,
    pub byram_intensity_btu_ft_s: f64,
    pub flame_length_ft: f64,
    pub spotting_distance_m: f64,
    pub threshold_class: NwcgSuppressionThresholdClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WildfireAlertWestConfig {
    pub posture: WildfireDetectionAndAttackPosture,
    pub fuel_model: WuiFuelModel,
    /// Midflame/20-ft wind speed (`mph`, `0..=65 mph`).
    pub wind_speed_mph: f64,
    /// Prevailing wind direction (`degrees` clockwise from North, direction fire pushes TOWARD).
    pub wind_push_bearing_deg: f64,
    /// Ambient air temperature (`Fahrenheit`, `45..=115 F`).
    pub ambient_temp_f: f64,
    /// Ambient relative humidity (`RH %`, `5..=85%`).
    pub relative_humidity_pct: f64,
    /// Mean WUI topographic slope (`%`, `0..=60%`).
    pub terrain_slope_pct: f64,
    /// Nocturnal ignition flag (switches `ALERTWest`/`OHAZ` cameras to `40-mi NIR` night mode; grounds day-only VFR tankers).
    pub nocturnal_ignition: bool,
    /// Number of portable `UOregon OHAZ` `< 50 lb` Starlink **Mobile Deployment Units (`MDU`s)** (`0..=4`).
    pub ohaz_starlink_mdu_units: usize,
    /// Number of `20`-person **Type 1 Interagency Hotshot Crews (`IHC`)** (`1..=8`).
    pub type1_hotshot_crews_20pax: usize,
    /// Number of **Type 3 / Type 6 WUI Wildland Engines** (`2..=18`).
    pub wui_wildland_engines: usize,
    /// Number of **Type 2 Wildland Bulldozers** (`0..=6`).
    pub wildland_bulldozers: usize,
    /// Number of **Type 1 Helitack Rotorcraft** (`2,000-gal` bucket, `0..=4`).
    pub helitack_rotorcraft: usize,
    /// Number of **Fixed-Wing Airtankers (`DC-10 VLAT` / `LAT` `Phos-Chek` retardant)** (`0..=4`).
    pub fixed_wing_airtankers: usize,
    /// Simulation horizon (`minutes` from ignition, `60..=360 min`).
    pub simulation_horizon_min: usize,
    pub monte_carlo_trials: usize,
    pub random_seed: u64,
    /// Optional external/custom `ALERTWest` / `OHAZ` camera sites loaded from `.json`/GeoJSON/`.csv`.
    #[serde(default)]
    pub custom_camera_sites: Vec<AlertWestOhazCameraSite>,
}

impl Default for WildfireAlertWestConfig {
    fn default() -> Self {
        Self {
            posture: WildfireDetectionAndAttackPosture::FullAlertWestOhazMduAndAerialAttack,
            fuel_model: WuiFuelModel::ConiferTimberUnderstoryFm10,
            wind_speed_mph: 24.0,
            wind_push_bearing_deg: 235.0, // East-northeast dry downslope wind pushing SW into WUI
            ambient_temp_f: 94.0,
            relative_humidity_pct: 14.0,
            terrain_slope_pct: 22.0,
            nocturnal_ignition: false,
            ohaz_starlink_mdu_units: 2,
            type1_hotshot_crews_20pax: 3,
            wui_wildland_engines: 8,
            wildland_bulldozers: 2,
            helitack_rotorcraft: 2,
            fixed_wing_airtankers: 2,
            simulation_horizon_min: 180,
            monte_carlo_trials: 160,
            random_seed: 0x0AA2_A1E7_5000,
            custom_camera_sites: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WildfireTimelinePoint {
    pub minute: usize,
    pub active_fire_area_acres: f64,
    pub uncontained_perimeter_ft: f64,
    pub built_containment_line_ft: f64,
    pub containment_pct: f64,
    pub flame_length_ft: f64,
    pub evacuated_wui_residents: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WildfireAlertWestReport {
    pub config: WildfireAlertWestConfig,
    pub solve_time_ms: f64,
    pub fire_behavior: RothermelFireBehaviorMetrics,
    /// `ALERTWest` & `UOregon OHAZ` Camera Detection & Cross-Azimuth Triangulation metrics
    pub active_camera_count: usize,
    pub cameras_with_los_count: usize,
    pub pnsn_shakealert_colocated_count: usize,
    pub true_ignition_pos: Vec2,
    pub triangulated_ignition_pos: Vec2,
    pub triangulation_cep90_m: f64,
    pub mean_detection_to_dispatch_min: f64,
    pub size_at_initial_attack_acres: f64,
    /// Interagency `NWCG` Suppression & WUI Triage outcomes
    pub final_burned_area_acres: f64,
    pub p90_burned_area_acres: f64,
    pub final_containment_pct: f64,
    pub time_to_full_containment_min: Option<f64>,
    pub total_handcrew_dozer_line_ft: f64,
    pub phos_chek_retardant_line_ft: f64,
    pub phos_chek_dropped_gallons: f64,
    pub wui_structures_saved_count: f64,
    pub expected_wui_structures_lost: f64,
    pub level3_go_evac_residents: f64,
    /// Spatial vectors & per-parcel GPU heatmap cache
    pub camera_sites: Vec<AlertWestOhazCameraSite>,
    pub camera_bearing_rays: Vec<(Vec2, Vec2, f64)>,
    pub retardant_drop_segments: Vec<(Vec2, Vec2)>,
    pub handcrew_dozer_segments: Vec<(Vec2, Vec2)>,
    pub timeline: Vec<WildfireTimelinePoint>,
    pub parcel_burn_risk_pct: Vec<f64>,
    pub parcel_camera_viewshed_pct: Vec<f64>,
    pub parcel_evac_level: Vec<u8>,
    pub parcel_triage: Vec<WuiStructureTriageCategory>,
}

impl WildfireAlertWestReport {
    pub fn to_camera_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.camera_sites.len() * 180);
        out.push_str(WILDFIRE_CAMERA_CSV_HEADER);
        out.push('\n');
        for c in &self.camera_sites {
            out.push_str(&format!(
                "{},{},{},{:.5},{:.5},{:.1},{:.1},{:.0},{:.1},{:.1},{:.1},{:.1},{},{:.1},{:.2},{}\n",
                c.camera_id,
                c.network_provider,
                c.kind.badge(),
                c.lat,
                c.lon,
                c.world_pos.x,
                c.world_pos.y,
                c.elevation_m,
                c.pan_azimuth_deg,
                c.optical_zoom_x,
                c.day_range_mi,
                c.nir_night_range_mi,
                c.co_located_pnsn_shakealert,
                c.latest_detected_bearing_deg.unwrap_or(-1.0),
                c.latest_ai_smoke_confidence.unwrap_or(0.0),
                c.live_image_endpoint
            ));
        }
        out
    }

    pub fn to_timeline_csv_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.timeline.len() * 96);
        out.push_str(WILDFIRE_TIMELINE_CSV_HEADER);
        out.push('\n');
        for pt in &self.timeline {
            out.push_str(&format!(
                "{},{:.2},{:.0},{:.0},{:.1},{:.2},{:.1}\n",
                pt.minute,
                pt.active_fire_area_acres,
                pt.uncontained_perimeter_ft,
                pt.built_containment_line_ft,
                pt.containment_pct,
                pt.flame_length_ft,
                pt.evacuated_wui_residents
            ));
        }
        out
    }

    pub fn export_camera_csv_path(&self, path: &Path) -> Result<usize, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(path, self.to_camera_csv_string()).map_err(|e| e.to_string())?;
        Ok(self.camera_sites.len())
    }
}

/// Built-in realistic Oregon `OHAZ` (`https://ohaz.uoregon.edu/wildfire-cameras/`) &
/// `ALERTWest` (`https://www.alertwest.org/`) camera network JSON feed for immediate offline testing
/// and live `.json` export/import.
pub fn builtin_oregon_ohaz_alertwest_feed_json() -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "feed_source": "ALERTWest (https://www.alertwest.org/) & UOregon Oregon Hazards Lab OHAZ (https://ohaz.uoregon.edu/wildfire-cameras/)",
        "scan_cadence_minutes": 2.0,
        "cameras": [
            {
                "camera_id": "Axis-MarysPeak",
                "network_provider": "UOregon OHAZ / Alert OR",
                "kind": "OhazMountaintopFixedPtz",
                "lat": 44.5044,
                "lon": -123.5512,
                "elevation_m": 1249.0,
                "pan_azimuth_deg": 76.0,
                "fov_deg": 62.0,
                "tilt_deg": -4.5,
                "optical_zoom_x": 40.0,
                "day_range_mi": 30.0,
                "nir_night_range_mi": 40.0,
                "scan_interval_min": 2.0,
                "co_located_pnsn_shakealert": true,
                "live_image_endpoint": "https://alertwest.live/api/firecams/v0/currentimage?name=Axis-MarysPeak",
                "portal_url": "https://ohaz.uoregon.edu/wildfire-cameras/"
            },
            {
                "camera_id": "Axis-McCullochPeak",
                "network_provider": "UOregon OHAZ / Alert OR",
                "kind": "OhazMountaintopFixedPtz",
                "lat": 44.6148,
                "lon": -123.3375,
                "elevation_m": 661.0,
                "pan_azimuth_deg": 118.0,
                "fov_deg": 62.0,
                "tilt_deg": -3.2,
                "optical_zoom_x": 40.0,
                "day_range_mi": 30.0,
                "nir_night_range_mi": 40.0,
                "scan_interval_min": 2.0,
                "co_located_pnsn_shakealert": true,
                "live_image_endpoint": "https://alertwest.live/api/firecams/v0/currentimage?name=Axis-McCullochPeak",
                "portal_url": "https://ohaz.uoregon.edu/wildfire-cameras/"
            },
            {
                "camera_id": "Axis-VineyardMtn",
                "network_provider": "ALERTWest / ODF WUI",
                "kind": "AlertWestWuiLookoutTower",
                "lat": 44.6295,
                "lon": -123.2680,
                "elevation_m": 472.0,
                "pan_azimuth_deg": 148.0,
                "fov_deg": 58.0,
                "tilt_deg": -2.8,
                "optical_zoom_x": 36.0,
                "day_range_mi": 30.0,
                "nir_night_range_mi": 40.0,
                "scan_interval_min": 2.0,
                "co_located_pnsn_shakealert": false,
                "live_image_endpoint": "https://alertwest.live/api/firecams/v0/currentimage?name=Axis-VineyardMtn",
                "portal_url": "https://www.alertwest.org/"
            },
            {
                "camera_id": "Axis-GrizzlyMtn",
                "network_provider": "UOregon OHAZ / Alert OR",
                "kind": "OhazMountaintopFixedPtz",
                "lat": 44.4382,
                "lon": -120.9558,
                "elevation_m": 1718.0,
                "pan_azimuth_deg": 248.0,
                "fov_deg": 62.0,
                "tilt_deg": -4.0,
                "optical_zoom_x": 40.0,
                "day_range_mi": 30.0,
                "nir_night_range_mi": 40.0,
                "scan_interval_min": 2.0,
                "co_located_pnsn_shakealert": true,
                "live_image_endpoint": "https://alertwest.live/api/firecams/v0/currentimage?name=Axis-GrizzlyMtn",
                "portal_url": "https://ohaz.uoregon.edu/wildfire-cameras/"
            },
            {
                "camera_id": "Axis-FoothillsSouthWUI",
                "network_provider": "ALERTWest Utility Partner",
                "kind": "AlertWestWuiLookoutTower",
                "lat": 44.5180,
                "lon": -123.2240,
                "elevation_m": 285.0,
                "pan_azimuth_deg": 32.0,
                "fov_deg": 60.0,
                "tilt_deg": -2.0,
                "optical_zoom_x": 32.0,
                "day_range_mi": 28.0,
                "nir_night_range_mi": 36.0,
                "scan_interval_min": 2.0,
                "co_located_pnsn_shakealert": false,
                "live_image_endpoint": "https://alertwest.live/api/firecams/v0/currentimage?name=Axis-FoothillsSouthWUI",
                "portal_url": "https://www.alertwest.org/"
            }
        ]
    }))
    .unwrap_or_default()
}

/// Projects geographic `(lat, lon)` camera coordinates onto the active simulation `BoundingBox`
/// (preserving perimeter lookout geometry around the urban WUI grid when regional mountaintop
/// coordinates extend beyond downtown bounds).
pub fn project_cameras_to_bounds(cameras: &mut [AlertWestOhazCameraSite], bounds: BoundingBox) {
    if cameras.is_empty() {
        return;
    }
    let span_x = (bounds.max_x - bounds.min_x).max(150.0);
    let span_y = (bounds.max_y - bounds.min_y).max(150.0);
    let min_lat = cameras.iter().map(|c| c.lat).fold(f64::INFINITY, f64::min);
    let max_lat = cameras
        .iter()
        .map(|c| c.lat)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lon = cameras.iter().map(|c| c.lon).fold(f64::INFINITY, f64::min);
    let max_lon = cameras
        .iter()
        .map(|c| c.lon)
        .fold(f64::NEG_INFINITY, f64::max);
    let d_lat = (max_lat - min_lat).max(1e-4);
    let d_lon = (max_lon - min_lon).max(1e-4);

    for (idx, cam) in cameras.iter_mut().enumerate() {
        if cam.world_pos.x.abs() > 1e-3 || cam.world_pos.y.abs() > 1e-3 {
            continue;
        }
        let u = if cameras.len() == 1 {
            0.85
        } else {
            ((cam.lon - min_lon) / d_lon).clamp(0.0, 1.0)
        };
        let v = if cameras.len() == 1 {
            0.15
        } else {
            // Screen/world +Y is South in metric SVG/OSM projections (`min_y` = North ridge)
            (1.0 - (cam.lat - min_lat) / d_lat).clamp(0.0, 1.0)
        };
        let pad = 0.06 + (idx % 2) as f64 * 0.03;
        cam.world_pos = Vec2 {
            x: bounds.min_x + (pad + u * (1.0 - 2.0 * pad)) * span_x,
            y: bounds.min_y + (pad + v * (1.0 - 2.0 * pad)) * span_y,
        };
    }
}

/// Parses an `ALERTWest` (`alertwest.org`) or `UOregon OHAZ` (`ohaz.uoregon.edu/wildfire-cameras/`)
/// camera & smoke-detection feed in **JSON** (`[...]`, `{"cameras": [...]}`), **GeoJSON**
/// (`FeatureCollection`), or **CSV** format!
pub fn parse_alertwest_ohaz_feed_str(
    raw: &str,
    bounds: BoundingBox,
) -> Result<Vec<AlertWestOhazCameraSite>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Empty ALERTWest/OHAZ feed content".to_string());
    }

    let mut sites: Vec<AlertWestOhazCameraSite> = Vec::new();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let val: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|e| format!("JSON parse error: {e}"))?;
        if let Some(arr) = val.as_array() {
            for item in arr {
                sites.push(parse_camera_json_obj(item)?);
            }
        } else if let Some(arr) = val.get("cameras").and_then(|v| v.as_array()) {
            for item in arr {
                sites.push(parse_camera_json_obj(item)?);
            }
        } else if let Some(feats) = val.get("features").and_then(|v| v.as_array()) {
            // GeoJSON FeatureCollection support (`features[].geometry.coordinates = [lon, lat]`)
            for feat in feats {
                let props = feat.get("properties").unwrap_or(feat);
                let mut merged = props.clone();
                if let Some(coords) = feat
                    .get("geometry")
                    .and_then(|g| g.get("coordinates"))
                    .and_then(|c| c.as_array())
                {
                    if coords.len() >= 2 {
                        if let Some(obj) = merged.as_object_mut() {
                            obj.entry("lon".to_string()).or_insert(coords[0].clone());
                            obj.entry("lat".to_string()).or_insert(coords[1].clone());
                        }
                    }
                }
                sites.push(parse_camera_json_obj(&merged)?);
            }
        } else {
            return Err(
                "Expected JSON camera array, `{\"cameras\": [...]}`, or GeoJSON `FeatureCollection`"
                    .to_string(),
            );
        }
    } else {
        // CSV parser (`camera_id,network_provider,kind,lat,lon,elevation_m,azimuth_deg,zoom_x,shakealert,...`)
        for (line_no, line) in trimmed.lines().enumerate() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = l.split(',').map(|s| s.trim()).collect();
            if line_no == 0 && cols[0].to_ascii_lowercase().contains("id") {
                continue;
            }
            if cols.len() < 5 {
                return Err(format!("CSV line {} has fewer than 5 columns", line_no + 1));
            }
            let cam_id = cols[0].to_string();
            let provider = cols[1].to_string();
            let kind = if cols[2].to_ascii_lowercase().contains("mdu") {
                AlertWestOhazCameraKind::OhazStarlinkMobileDeploymentUnit
            } else if cols[2].to_ascii_lowercase().contains("wui") {
                AlertWestOhazCameraKind::AlertWestWuiLookoutTower
            } else {
                AlertWestOhazCameraKind::OhazMountaintopFixedPtz
            };
            let lat: f64 = cols[3]
                .parse()
                .map_err(|_| format!("Invalid lat on CSV line {}", line_no + 1))?;
            let lon: f64 = cols[4]
                .parse()
                .map_err(|_| format!("Invalid lon on CSV line {}", line_no + 1))?;
            let elev_m: f64 = cols.get(5).and_then(|s| s.parse().ok()).unwrap_or(650.0);
            let az_deg: f64 = cols.get(6).and_then(|s| s.parse().ok()).unwrap_or(90.0);
            let zoom_x: f64 = cols.get(7).and_then(|s| s.parse().ok()).unwrap_or(40.0);
            let shake: bool = cols
                .get(8)
                .map(|s| matches!(s.to_ascii_lowercase().as_str(), "true" | "1" | "yes"))
                .unwrap_or(true);
            let bearing: Option<f64> = cols
                .get(9)
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|&v| v >= 0.0);
            let conf: Option<f64> = cols
                .get(10)
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|&v| v > 0.0);
            sites.push(AlertWestOhazCameraSite {
                live_image_endpoint: format!(
                    "https://alertwest.live/api/firecams/v0/currentimage?name={cam_id}"
                ),
                portal_url: if provider.to_ascii_lowercase().contains("ohaz") {
                    "https://ohaz.uoregon.edu/wildfire-cameras/".to_string()
                } else {
                    "https://www.alertwest.org/".to_string()
                },
                camera_id: cam_id,
                network_provider: provider,
                kind,
                lat,
                lon,
                world_pos: Vec2::default(),
                elevation_m: elev_m,
                pan_azimuth_deg: az_deg,
                fov_deg: 62.0,
                tilt_deg: -3.5,
                optical_zoom_x: zoom_x.clamp(1.0, 40.0),
                day_range_mi: 30.0,
                nir_night_range_mi: 40.0,
                scan_interval_min: 2.0,
                co_located_pnsn_shakealert: shake,
                latest_ai_smoke_confidence: conf,
                latest_detected_bearing_deg: bearing,
            });
        }
    }

    if sites.is_empty() {
        return Err("No camera records found in ALERTWest/OHAZ feed".to_string());
    }
    project_cameras_to_bounds(&mut sites, bounds);
    Ok(sites)
}

fn parse_camera_json_obj(v: &serde_json::Value) -> Result<AlertWestOhazCameraSite, String> {
    if let Ok(direct) = serde_json::from_value::<AlertWestOhazCameraSite>(v.clone()) {
        return Ok(direct);
    }
    let id = v
        .get("camera_id")
        .or_else(|| v.get("id"))
        .or_else(|| v.get("name"))
        .and_then(|x| x.as_str())
        .unwrap_or("Axis-OHAZ-Cam")
        .to_string();
    let provider = v
        .get("network_provider")
        .or_else(|| v.get("provider"))
        .or_else(|| v.get("network"))
        .and_then(|x| x.as_str())
        .unwrap_or("ALERTWest / UOregon OHAZ")
        .to_string();
    let kind_str = v
        .get("kind")
        .and_then(|x| x.as_str())
        .unwrap_or("OhazMountaintopFixedPtz")
        .to_ascii_lowercase();
    let kind = if kind_str.contains("mdu") || kind_str.contains("mobile") {
        AlertWestOhazCameraKind::OhazStarlinkMobileDeploymentUnit
    } else if kind_str.contains("wui") || kind_str.contains("lookout") {
        AlertWestOhazCameraKind::AlertWestWuiLookoutTower
    } else {
        AlertWestOhazCameraKind::OhazMountaintopFixedPtz
    };
    let lat = v
        .get("lat")
        .or_else(|| v.get("latitude"))
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("Camera `{id}` missing numeric `lat`"))?;
    let lon = v
        .get("lon")
        .or_else(|| v.get("lng"))
        .or_else(|| v.get("longitude"))
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("Camera `{id}` missing numeric `lon`"))?;
    let elev = v
        .get("elevation_m")
        .or_else(|| v.get("elev_m"))
        .and_then(|x| x.as_f64())
        .unwrap_or(820.0);
    let az = v
        .get("pan_azimuth_deg")
        .or_else(|| v.get("azimuth_deg"))
        .or_else(|| v.get("azimuth"))
        .and_then(|x| x.as_f64())
        .unwrap_or(90.0);
    let fov = v.get("fov_deg").and_then(|x| x.as_f64()).unwrap_or(62.0);
    let tilt = v.get("tilt_deg").and_then(|x| x.as_f64()).unwrap_or(-3.5);
    let zoom = v
        .get("optical_zoom_x")
        .or_else(|| v.get("zoom_x"))
        .and_then(|x| x.as_f64())
        .unwrap_or(40.0)
        .clamp(1.0, 40.0);
    let day_mi = v
        .get("day_range_mi")
        .and_then(|x| x.as_f64())
        .unwrap_or(30.0);
    let nir_mi = v
        .get("nir_night_range_mi")
        .and_then(|x| x.as_f64())
        .unwrap_or(40.0);
    let scan_min = v
        .get("scan_interval_min")
        .and_then(|x| x.as_f64())
        .unwrap_or(2.0);
    let shake = v
        .get("co_located_pnsn_shakealert")
        .or_else(|| v.get("shakealert"))
        .and_then(|x| x.as_bool())
        .unwrap_or(true);
    let endpoint = v
        .get("live_image_endpoint")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("https://alertwest.live/api/firecams/v0/currentimage?name={id}"));
    let portal = v
        .get("portal_url")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "https://ohaz.uoregon.edu/wildfire-cameras/".to_string());
    let conf = v
        .get("latest_ai_smoke_confidence")
        .or_else(|| v.get("smoke_confidence"))
        .and_then(|x| x.as_f64());
    let brg = v
        .get("latest_detected_bearing_deg")
        .or_else(|| v.get("bearing_deg"))
        .and_then(|x| x.as_f64());

    Ok(AlertWestOhazCameraSite {
        camera_id: id,
        network_provider: provider,
        kind,
        lat,
        lon,
        world_pos: Vec2::default(),
        elevation_m: elev,
        pan_azimuth_deg: az,
        fov_deg: fov,
        tilt_deg: tilt,
        optical_zoom_x: zoom,
        day_range_mi: day_mi,
        nir_night_range_mi: nir_mi,
        scan_interval_min: scan_min,
        co_located_pnsn_shakealert: shake,
        live_image_endpoint: endpoint,
        portal_url: portal,
        latest_ai_smoke_confidence: conf,
        latest_detected_bearing_deg: brg,
    })
}

pub fn load_alertwest_ohaz_feed_path(
    path: &Path,
    bounds: BoundingBox,
) -> Result<Vec<AlertWestOhazCameraSite>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("Failed reading {:?}: {e}", path))?;
    parse_alertwest_ohaz_feed_str(&content, bounds)
}

/// Solves Weighted Least-Squares (`WLS`) multi-camera cross-azimuth bearing triangulation
/// given `observations = &[(camera_world_pos, compass_bearing_deg, angular_sigma_deg)]`,
/// returning `(estimated_pos, cep90_error_m)`!
/// Compass convention: `0° = North (-Y)`, `90° = East (+X)`, `180° = South (+Y)`, `270° = West (-X)`.
pub fn triangulate_camera_bearings_wls(
    observations: &[(Vec2, f64, f64)],
    fallback_guess: Vec2,
) -> (Vec2, f64) {
    if observations.is_empty() {
        return (fallback_guess, 1_450.0);
    }
    if observations.len() == 1 {
        let (c0, brg_deg, sig_deg) = observations[0];
        let rad = brg_deg.to_radians();
        let dir = Vec2 {
            x: rad.sin(),
            y: -rad.cos(),
        };
        let d_est = c0.dist(fallback_guess).max(250.0);
        let est = Vec2 {
            x: c0.x + dir.x * d_est,
            y: c0.y + dir.y * d_est,
        };
        let cep90 = (d_est * sig_deg.to_radians() * 2.15 + 160.0).clamp(95.0, 1_200.0);
        return (est, cep90);
    }

    // Normal equations minimizing sum_k w_k * (n_k . (p - c_k))^2 where n_k is perpendicular to ray u_k
    let mut a11 = 0.0;
    let mut a12 = 0.0;
    let mut a22 = 0.0;
    let mut b1 = 0.0;
    let mut b2 = 0.0;

    for &(ck, brg_deg, sig_deg) in observations {
        let rad = brg_deg.to_radians();
        // Ray unit vector u = (sin(theta), -cos(theta)) -> unit normal n = (cos(theta), sin(theta))
        let nx = rad.cos();
        let ny = rad.sin();
        let sig_rad = sig_deg.max(0.08).to_radians();
        let w = 1.0 / (sig_rad * sig_rad);
        let rhs = nx * ck.x + ny * ck.y;
        a11 += w * nx * nx;
        a12 += w * nx * ny;
        a22 += w * ny * ny;
        b1 += w * nx * rhs;
        b2 += w * ny * rhs;
    }

    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-9 {
        return (fallback_guess, 420.0);
    }
    let x_hat = (a22 * b1 - a12 * b2) / det;
    let y_hat = (a11 * b2 - a12 * b1) / det;
    let p_hat = Vec2 { x: x_hat, y: y_hat };

    // Convert angular covariance into spatial error covariance using range `R_k = |p_hat - c_k|`
    let mut m11 = 0.0;
    let mut m12 = 0.0;
    let mut m22 = 0.0;
    for &(ck, brg_deg, sig_deg) in observations {
        let rad = brg_deg.to_radians();
        let nx = rad.cos();
        let ny = rad.sin();
        let r_m = ck.dist(p_hat).max(120.0);
        let sigma_perp_m = r_m * sig_deg.max(0.10).to_radians();
        let w_m = 1.0 / (sigma_perp_m * sigma_perp_m);
        m11 += w_m * nx * nx;
        m12 += w_m * nx * ny;
        m22 += w_m * ny * ny;
    }
    let m_det = (m11 * m22 - m12 * m12).max(1e-9);
    let cov_xx = m22 / m_det;
    let cov_yy = m11 / m_det;
    // Rayleigh 90% Circular Error Probable (`CEP_90 ~ 2.146 * sqrt(0.5 * (sigma_x^2 + sigma_y^2))`)
    let cep90_m = (2.146 * (0.5 * (cov_xx + cov_yy)).max(0.25).sqrt()).clamp(4.0, 950.0);
    (p_hat, cep90_m)
}

/// Evaluates Rothermel (1972) / Byram (1959) / Thomas (1963) surface-to-crown WUI fire behavior
/// and USDA Forest Service `RMRS-GTR-253` suppression threshold classification.
pub fn compute_rothermel_wui_fire_behavior(
    fuel: WuiFuelModel,
    wind_mph: f64,
    temp_f: f64,
    rh_pct: f64,
    slope_pct: f64,
) -> RothermelFireBehaviorMetrics {
    // 1-hr fine dead fuel moisture (`%`) per NWCG Fireline Handbook psychrometric table approximation
    let temp_adj = ((temp_f - 70.0) * 0.055).clamp(-1.5, 2.8);
    let fdfm_pct = (2.2 + 0.165 * rh_pct.clamp(4.0, 95.0) - temp_adj).clamp(2.5, 24.0);
    let moisture_damping = ((26.0 - fdfm_pct) / 20.0).clamp(0.18, 1.22);

    // Fuel-model base spread (`chains/hr` @ `0 mph`, `8%` moisture) & heat release (`Btu/ft^2`)
    let (base_ros_ch_hr, heat_per_area_btu_ft2, wind_exp) = match fuel {
        WuiFuelModel::ShortGrassFm1 => (14.5, 290.0, 1.38),
        WuiFuelModel::BrushWuiFm5 => (6.8, 720.0, 1.28),
        WuiFuelModel::HeavyChaparralSlashFm4 => (11.2, 2_150.0, 1.32),
        WuiFuelModel::ConiferTimberUnderstoryFm10 => (3.9, 1_480.0, 1.18),
    };

    let u_eff = (wind_mph.clamp(0.0, 75.0) / 8.0).powf(wind_exp);
    let phi_w = 1.0 + 2.35 * u_eff;
    let tan_slope = (slope_pct.clamp(0.0, 80.0) / 100.0).min(0.85);
    let phi_s = 5.275 * 0.08 * tan_slope * tan_slope * 18.0;

    let ros_chains_hr = (base_ros_ch_hr * moisture_damping * (phi_w + phi_s)).clamp(0.5, 280.0);
    // 1 chain = 66 ft = 20.1168 m -> chains/hr * 0.018333 = ft/s; chains/hr * 0.33528 = m/min
    let ros_ft_s = ros_chains_hr * (66.0 / 3600.0);
    let ros_m_min = ros_chains_hr * (20.1168 / 60.0);

    // Byram (1959) Fireline Intensity `I_B = H_A * R` (`Btu/ft/s`)
    let ib_btu_ft_s = (heat_per_area_btu_ft2 * moisture_damping * ros_ft_s).max(5.0);
    // Thomas (1963) / RMRS-GTR-253 Flame Length `L_f = 0.45 * I_B^0.46` (`ft`)
    let flame_len_ft = (0.45 * ib_btu_ft_s.powf(0.46)).clamp(0.8, 65.0);

    // Albini (1979) maximum wind-driven ember spotting distance (`meters`)
    let spot_m = if flame_len_ft >= 7.5 {
        (38.0 * flame_len_ft.sqrt() * (1.0 + wind_mph.max(0.0) / 14.0)).clamp(50.0, 2_400.0)
    } else {
        (8.0 * flame_len_ft * (wind_mph.max(0.0) / 20.0)).clamp(0.0, 180.0)
    };

    let threshold_class = if flame_len_ft <= 4.0 {
        NwcgSuppressionThresholdClass::DirectHandToolsUnder4Ft
    } else if flame_len_ft <= 8.0 {
        NwcgSuppressionThresholdClass::EquipmentAndRetardant4To8Ft
    } else if flame_len_ft <= 11.0 {
        NwcgSuppressionThresholdClass::IndirectParallel8To11Ft
    } else {
        NwcgSuppressionThresholdClass::CrowningExtremeOver11Ft
    };

    RothermelFireBehaviorMetrics {
        fine_dead_fuel_moisture_pct: fdfm_pct,
        rate_of_spread_chains_per_hr: ros_chains_hr,
        rate_of_spread_m_per_min: ros_m_min,
        byram_intensity_btu_ft_s: ib_btu_ft_s,
        flame_length_ft: flame_len_ft,
        spotting_distance_m: spot_m,
        threshold_class,
    }
}

/// Sustained fireline construction rate (`ft/hr`) per `20`-person **Type 1 Interagency Hotshot
/// Crew (`IHC`)** and per **Type 2 Wildland Bulldozer** from USDA FS / `NWCG` production tables.
pub fn nwcg_crew_and_dozer_line_rates_ft_hr(
    fuel: WuiFuelModel,
    slope_pct: f64,
    direct_attack: bool,
) -> (f64, f64) {
    let ihc_base = match (fuel, direct_attack) {
        (WuiFuelModel::ShortGrassFm1, _) => 1_780.0,
        (WuiFuelModel::BrushWuiFm5, true) => 1_089.0,
        (WuiFuelModel::BrushWuiFm5, false) => 720.0,
        (WuiFuelModel::HeavyChaparralSlashFm4, true) => 436.0,
        (WuiFuelModel::HeavyChaparralSlashFm4, false) => 365.0,
        (WuiFuelModel::ConiferTimberUnderstoryFm10, true) => 693.0,
        (WuiFuelModel::ConiferTimberUnderstoryFm10, false) => 455.0,
    };
    let slope_factor = (1.0 - (slope_pct.clamp(0.0, 65.0) - 25.0).max(0.0) * 0.011).clamp(0.48, 1.0);
    let dozer_base = match fuel {
        WuiFuelModel::ShortGrassFm1 => 3_650.0,
        WuiFuelModel::BrushWuiFm5 => 2_650.0,
        WuiFuelModel::HeavyChaparralSlashFm4 => 1_820.0,
        WuiFuelModel::ConiferTimberUnderstoryFm10 => 2_150.0,
    };
    (ihc_base * slope_factor, dozer_base * slope_factor)
}

#[inline]
fn compass_bearing_deg(from: Vec2, to: Vec2) -> f64 {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    // +X = East (90°), -Y = North (0°)
    let mut deg = dx.atan2(-dy).to_degrees();
    if deg < 0.0 {
        deg += 360.0;
    }
    deg
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn u01_from_u64(v: u64) -> f64 {
    ((v >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Executes a `Rayon`-parallelized `ALERTWest` + `UOregon OHAZ` AI PTZ Camera Detection,
/// Multi-Camera Cross-Azimuth Triangulation, and `NWCG` Interagency WUI Wildfire Suppression
/// Monte Carlo simulation.
pub fn run_wildfire_alertwest_monte_carlo(
    net: &RoadNetwork,
    parcels: &[Parcel],
    facilities: &[EmergencyFacility],
    cfg: &WildfireAlertWestConfig,
) -> WildfireAlertWestReport {
    let t0 = Instant::now();
    let b = net.bounds;
    let span_x = (b.max_x - b.min_x).max(160.0);
    let span_y = (b.max_y - b.min_y).max(160.0);

    // 1. Assemble Active Camera Roster (Custom loaded feed OR built-in Oregon OHAZ / ALERTWest catalog + Starlink MDUs)
    let mut cameras = if !cfg.custom_camera_sites.is_empty() {
        let mut cs = cfg.custom_camera_sites.clone();
        project_cameras_to_bounds(&mut cs, b);
        cs
    } else {
        parse_alertwest_ohaz_feed_str(&builtin_oregon_ohaz_alertwest_feed_json(), b)
            .unwrap_or_default()
    };

    if cfg.posture == WildfireDetectionAndAttackPosture::FullAlertWestOhazMduAndAerialAttack {
        let mdu_n = cfg.ohaz_starlink_mdu_units.min(4);
        let mdu_spots = [
            (
                "OHAZ-MDU-Starlink-1",
                Vec2 {
                    x: b.min_x + 0.84 * span_x,
                    y: b.min_y + 0.14 * span_y,
                },
                44.6020,
                -123.2180,
            ),
            (
                "OHAZ-MDU-Starlink-2",
                Vec2 {
                    x: b.min_x + 0.91 * span_x,
                    y: b.min_y + 0.44 * span_y,
                },
                44.5710,
                -123.2040,
            ),
            (
                "OHAZ-MDU-Starlink-3",
                Vec2 {
                    x: b.min_x + 0.64 * span_x,
                    y: b.min_y + 0.08 * span_y,
                },
                44.6350,
                -123.2550,
            ),
            (
                "OHAZ-MDU-Starlink-4",
                Vec2 {
                    x: b.min_x + 0.14 * span_x,
                    y: b.min_y + 0.22 * span_y,
                },
                44.5950,
                -123.3450,
            ),
        ];
        for &(id, wpos, lat, lon) in mdu_spots.iter().take(mdu_n) {
            cameras.push(AlertWestOhazCameraSite {
                camera_id: id.to_string(),
                network_provider: "UOregon OHAZ Starlink MDU (<50 lb)".to_string(),
                kind: AlertWestOhazCameraKind::OhazStarlinkMobileDeploymentUnit,
                lat,
                lon,
                world_pos: wpos,
                elevation_m: 390.0,
                pan_azimuth_deg: 215.0,
                fov_deg: 65.0,
                tilt_deg: -2.5,
                optical_zoom_x: 36.0,
                day_range_mi: 25.0,
                nir_night_range_mi: 34.0,
                scan_interval_min: 2.0,
                co_located_pnsn_shakealert: false,
                live_image_endpoint: format!(
                    "https://alertwest.live/api/firecams/v0/currentimage?name={id}"
                ),
                portal_url: "https://ohaz.uoregon.edu/wildfire-cameras/".to_string(),
                latest_ai_smoke_confidence: None,
                latest_detected_bearing_deg: None,
            });
        }
    }

    // 2. Determine Upwind WUI Ignition Origin based on `wind_push_bearing_deg`
    let push_rad = cfg.wind_push_bearing_deg.to_radians();
    let push_dir = Vec2 {
        x: push_rad.sin(),
        y: -push_rad.cos(),
    };
    let city_center = Vec2 {
        x: 0.5 * (b.min_x + b.max_x),
        y: 0.5 * (b.min_y + b.max_y),
    };
    let true_ignition = Vec2 {
        x: (city_center.x - push_dir.x * 0.34 * span_x).clamp(b.min_x + 35.0, b.max_x - 35.0),
        y: (city_center.y - push_dir.y * 0.34 * span_y).clamp(b.min_y + 35.0, b.max_y - 35.0),
    };

    // 3. Evaluate `ALERTWest` / `UOregon OHAZ` Camera Line-of-Sight, `NIR` Night Vision & WLS Triangulation
    let mut bearing_obs: Vec<(Vec2, f64, f64)> = Vec::new();
    let mut camera_bearing_rays: Vec<(Vec2, Vec2, f64)> = Vec::new();
    let pnsn_colocated = cameras
        .iter()
        .filter(|c| c.co_located_pnsn_shakealert)
        .count();

    for (idx, cam) in cameras.iter_mut().enumerate() {
        let dist_m = cam.world_pos.dist(true_ignition);
        let max_range_m = if cfg.nocturnal_ignition {
            cam.nir_night_range_mi * 1_609.34 // 40 mi (64.4 km) NIR night vision!
        } else {
            cam.day_range_mi * 1_609.34 // 30 mi (48.3 km) optical day range!
        };
        // Simulate ridge canyon occlusion on 1 distant fixed camera unless Starlink MDUs fill blind spots
        let terrain_clear = cam.kind == AlertWestOhazCameraKind::OhazStarlinkMobileDeploymentUnit
            || idx % 4 != 3
            || dist_m < 0.55 * span_x.max(span_y);
        if cfg.posture != WildfireDetectionAndAttackPosture::Legacy911GroundOnly
            && dist_m <= max_range_m
            && terrain_clear
        {
            let true_brg = compass_bearing_deg(cam.world_pos, true_ignition);
            let measured_brg = cam.latest_detected_bearing_deg.unwrap_or_else(|| {
                // Deterministic micro-jitter proportional to `1 / optical_zoom_x` (`~0.18°` at `40x`)
                let sigma = 7.2 / cam.optical_zoom_x.max(4.0);
                let u = u01_from_u64(splitmix64(cfg.random_seed ^ (idx as u64 + 17))) - 0.5;
                (true_brg + u * sigma).rem_euclid(360.0)
            });
            let angular_sigma_deg = (7.2 / cam.optical_zoom_x.max(4.0)).clamp(0.16, 1.8);
            let conf = cam.latest_ai_smoke_confidence.unwrap_or_else(|| {
                let nir_bonus: f64 = if cfg.nocturnal_ignition { 0.04 } else { 0.0 };
                (0.91 + nir_bonus - (dist_m / max_range_m) * 0.14).clamp(0.68, 0.99)
            });
            cam.latest_detected_bearing_deg = Some(measured_brg);
            cam.latest_ai_smoke_confidence = Some(conf);
            cam.pan_azimuth_deg = measured_brg;
            bearing_obs.push((cam.world_pos, measured_brg, angular_sigma_deg));
        }
    }

    let (triangulated_pos, cep90_m) = if cfg.posture
        == WildfireDetectionAndAttackPosture::Legacy911GroundOnly
    {
        (
            Vec2 {
                x: true_ignition.x + 145.0,
                y: true_ignition.y - 110.0,
            },
            680.0,
        )
    } else {
        triangulate_camera_bearings_wls(&bearing_obs, true_ignition)
    };

    for cam in &cameras {
        if let (Some(brg), Some(conf)) = (
            cam.latest_detected_bearing_deg,
            cam.latest_ai_smoke_confidence,
        ) {
            let rad = brg.to_radians();
            let ray_len = cam.world_pos.dist(triangulated_pos) * 1.06;
            let end_pt = Vec2 {
                x: cam.world_pos.x + rad.sin() * ray_len,
                y: cam.world_pos.y - rad.cos() * ray_len,
            };
            camera_bearing_rays.push((cam.world_pos, end_pt, conf));
        }
    }

    // 4. Evaluate Rothermel Fire Behavior & NWCG Production Rates
    let fb = compute_rothermel_wui_fire_behavior(
        cfg.fuel_model,
        cfg.wind_speed_mph,
        cfg.ambient_temp_f,
        cfg.relative_humidity_pct,
        cfg.terrain_slope_pct,
    );
    let direct_attack_ok = fb.flame_length_ft <= 4.5;
    let (ihc_ft_hr, dozer_ft_hr) = nwcg_crew_and_dozer_line_rates_ft_hr(
        cfg.fuel_model,
        cfg.terrain_slope_pct,
        direct_attack_ok,
    );

    // Closest Fire Station travel time to WUI ignition (`minutes`)
    let nearest_fs_min = facilities
        .iter()
        .filter(|f| f.kind == EmergencyFacilityKind::FireStation)
        .map(|f| (f.world_pos.dist(true_ignition) * 1.28 / 13.5) / 60.0)
        .fold(9.5, f64::min);

    let horizon_min = cfg.simulation_horizon_min.clamp(30, 480);
    let step_min = 5usize;
    let n_steps = horizon_min / step_min + 1;
    let trials = cfg.monte_carlo_trials.clamp(16, 2_500);
    let cameras_los = bearing_obs.len();

    #[derive(Clone)]
    struct FireTrial {
        detect_dispatch_min: f64,
        ia_size_acres: f64,
        final_acres: f64,
        final_contain_pct: f64,
        contain_time_min: Option<f64>,
        ground_line_ft: f64,
        retardant_line_ft: f64,
        retardant_gal: f64,
        area_curve: Vec<f64>,
        perim_curve: Vec<f64>,
        line_curve: Vec<f64>,
        contain_curve: Vec<f64>,
    }

    let fire_trials: Vec<FireTrial> = (0..trials)
        .into_par_iter()
        .map(|tr| {
            let s0 = splitmix64(cfg.random_seed.wrapping_add((tr as u64).wrapping_mul(0x9E37_79B9)));
            let wind_jit = 0.85 + 0.34 * u01_from_u64(s0);
            let detect_jit = 0.80 + 0.45 * u01_from_u64(splitmix64(s0 ^ 0x55));

            // Detection + CAD verification delay (`minutes`):
            // - Legacy 911: `19.5 min` day / `44.0 min` night (unwitnessed nocturnal WUI start!)
            // - ALERTWest/OHAZ 2-min panoramic AI scan + human ops-center verification: `2.4–3.6 min` day or `NIR` night!
            let detect_dispatch_min = match cfg.posture {
                WildfireDetectionAndAttackPosture::Legacy911GroundOnly => {
                    let base = if cfg.nocturnal_ignition { 43.0 } else { 19.0 };
                    base * detect_jit
                }
                WildfireDetectionAndAttackPosture::AlertWestOhazFixedPtzInitialAttack => {
                    if cameras_los >= 2 {
                        2.8 * detect_jit
                    } else if cameras_los == 1 {
                        5.2 * detect_jit
                    } else {
                        16.5 * detect_jit
                    }
                }
                WildfireDetectionAndAttackPosture::FullAlertWestOhazMduAndAerialAttack => {
                    if cameras_los >= 2 {
                        2.1 * detect_jit
                    } else {
                        3.8 * detect_jit
                    }
                }
            };

            let engine_arrival_min = detect_dispatch_min + nearest_fs_min;
            let airtanker_arrival_min = detect_dispatch_min + 16.0;
            let aerial_enabled = cfg.posture
                == WildfireDetectionAndAttackPosture::FullAlertWestOhazMduAndAerialAttack;
            // Fixed-wing VLAT/LAT require daylight VFR; Type 1 NVIS Helitack can fly night bucket drops if wind <= 38 mph
            let tanker_flyable =
                aerial_enabled && !cfg.nocturnal_ignition && cfg.wind_speed_mph * wind_jit <= 42.0;
            let helo_flyable = aerial_enabled && cfg.wind_speed_mph * wind_jit <= 38.0;

            // Elliptical length-to-breadth ratio `LB = 0.936 * exp(0.2566 * U_mps) + 0.461 * exp(-0.1548 * U_mps) - 0.397` (Anderson 1983)
            let u_mps = (cfg.wind_speed_mph * wind_jit * 0.44704).clamp(0.0, 28.0);
            let lb_ratio = (0.936 * (0.2566 * u_mps * 0.35).exp()
                + 0.461 * (-0.1548 * u_mps * 0.35).exp()
                - 0.397)
                .clamp(1.1, 6.5);

            let head_ros_m_min = fb.rate_of_spread_m_per_min * wind_jit;
            let mut major_axis_m = 14.0;
            let mut built_ground_ft = 0.0;
            let mut built_retardant_ft = 0.0;
            let mut dropped_gal = 0.0;
            let mut ia_size_acres = 0.25;
            let mut contain_time_min = None;

            let mut area_curve = Vec::with_capacity(n_steps);
            let mut perim_curve = Vec::with_capacity(n_steps);
            let mut line_curve = Vec::with_capacity(n_steps);
            let mut contain_curve = Vec::with_capacity(n_steps);

            for st in 0..n_steps {
                let m = (st * step_min) as f64;
                let semi_a_m = 0.5 * major_axis_m;
                let semi_b_m = (semi_a_m / lb_ratio).max(5.0);
                let area_m2 = std::f64::consts::PI * semi_a_m * semi_b_m;
                let area_acres = area_m2 / 4_046.856;
                // Ramanujan ellipse perimeter in feet (`1 m = 3.28084 ft`)
                let h_ell = ((semi_a_m - semi_b_m) / (semi_a_m + semi_b_m)).powi(2);
                let perim_ft = std::f64::consts::PI
                    * (semi_a_m + semi_b_m)
                    * (1.0 + (3.0 * h_ell) / (10.0 + (4.0 - 3.0 * h_ell).sqrt()))
                    * 3.28084;

                if (m - engine_arrival_min).abs() < (step_min as f64) * 0.6 {
                    ia_size_acres = area_acres;
                }

                let total_line_ft = built_ground_ft + built_retardant_ft * 0.85;
                let contain_pct = ((total_line_ft / perim_ft.max(60.0)) * 100.0).clamp(0.0, 100.0);
                if contain_pct >= 99.5 && contain_time_min.is_none() {
                    contain_time_min = Some(m);
                }

                area_curve.push(area_acres);
                perim_curve.push((perim_ft - total_line_ft).max(0.0));
                line_curve.push(total_line_ft);
                contain_curve.push(contain_pct);

                if st + 1 < n_steps && contain_pct < 99.8 {
                    let dt_hr = (step_min as f64) / 60.0;
                    if m >= engine_arrival_min {
                        // Ground crews + Wildland Engines + Dozers construct mineral-soil & wet line
                        let engine_wetline_ft_hr = (cfg.wui_wildland_engines as f64)
                            * if fb.flame_length_ft <= 8.0 {
                                420.0
                            } else {
                                210.0
                            };
                        let ground_rate_ft_hr = (cfg.type1_hotshot_crews_20pax as f64) * ihc_ft_hr
                            + (cfg.wildland_bulldozers as f64) * dozer_ft_hr
                            + engine_wetline_ft_hr;
                        built_ground_ft += ground_rate_ft_hr * dt_hr;
                    }
                    if helo_flyable && m >= detect_dispatch_min + 11.0 {
                        // Type 1 Helitack 2,000-gal bucket drops (~4 drops/hr/helo -> ~280 ft wet knockdown/drop)
                        let drops = (cfg.helitack_rotorcraft as f64) * 4.0 * dt_hr;
                        built_ground_ft += drops * 260.0;
                        dropped_gal += drops * 1_800.0;
                    }
                    if tanker_flyable && m >= airtanker_arrival_min {
                        // Fixed-wing LAT/VLAT Phos-Chek drops (`~1.6 sorties/hr/tanker`, `4,500 gal` avg,
                        // `1,350 ft` line @ `NWCG` Coverage Level 4 = `1.8 gpc`)
                        let sorties = (cfg.fixed_wing_airtankers as f64) * 1.55 * dt_hr;
                        built_retardant_ft += sorties * 1_350.0;
                        dropped_gal += sorties * 4_500.0;
                    }

                    // Retardant + flank containment dampens uncontained head/backing progression
                    let retardant_knockdown = if built_retardant_ft > 200.0 {
                        0.36 // Gimenez et al. (2004) Phos-Chek Superiority Factor > 0.6 knockdown!
                    } else if helo_flyable && m >= detect_dispatch_min + 11.0 {
                        0.68
                    } else {
                        1.0
                    };
                    let uncontained_frac = (1.0 - contain_pct / 100.0).clamp(0.0, 1.0);
                    major_axis_m +=
                        head_ros_m_min * 1.18 * (step_min as f64) * retardant_knockdown * uncontained_frac;
                }
            }

            FireTrial {
                detect_dispatch_min,
                ia_size_acres,
                final_acres: *area_curve.last().unwrap_or(&0.5),
                final_contain_pct: *contain_curve.last().unwrap_or(&0.0),
                contain_time_min,
                ground_line_ft: built_ground_ft,
                retardant_line_ft: built_retardant_ft,
                retardant_gal: dropped_gal,
                area_curve,
                perim_curve,
                line_curve,
                contain_curve,
            }
        })
        .collect();

    let inv_t = 1.0 / (trials as f64);
    let mean_detection_to_dispatch_min = fire_trials
        .iter()
        .map(|r| r.detect_dispatch_min)
        .sum::<f64>()
        * inv_t;
    let size_at_initial_attack_acres =
        fire_trials.iter().map(|r| r.ia_size_acres).sum::<f64>() * inv_t;
    let final_burned_area_acres = fire_trials.iter().map(|r| r.final_acres).sum::<f64>() * inv_t;
    let mut sorted_acres: Vec<f64> = fire_trials.iter().map(|r| r.final_acres).collect();
    sorted_acres.sort_by(|a, b_v| a.partial_cmp(b_v).unwrap_or(std::cmp::Ordering::Equal));
    let p90_burned_area_acres = sorted_acres
        [((sorted_acres.len() as f64 * 0.90).floor() as usize).min(sorted_acres.len() - 1)];
    let final_containment_pct = fire_trials
        .iter()
        .map(|r| r.final_contain_pct)
        .sum::<f64>()
        * inv_t;
    let total_handcrew_dozer_line_ft =
        fire_trials.iter().map(|r| r.ground_line_ft).sum::<f64>() * inv_t;
    let phos_chek_retardant_line_ft = fire_trials
        .iter()
        .map(|r| r.retardant_line_ft)
        .sum::<f64>()
        * inv_t;
    let phos_chek_dropped_gallons =
        fire_trials.iter().map(|r| r.retardant_gal).sum::<f64>() * inv_t;

    let contained_samples: Vec<f64> = fire_trials
        .iter()
        .filter_map(|r| r.contain_time_min)
        .collect();
    let time_to_full_containment_min = if contained_samples.len() * 2 >= trials {
        Some(contained_samples.iter().sum::<f64>() / (contained_samples.len() as f64))
    } else {
        None
    };

    // 5. Spatial Elliptical Fire Footprint, WUI Structure Triage & Level 1/2/3 Evacuation Tiers
    let final_area_m2 = (final_burned_area_acres * 4_046.856).max(400.0);
    let u_mps = (cfg.wind_speed_mph * 0.44704).clamp(0.0, 28.0);
    let lb_ratio = (0.936 * (0.2566 * u_mps * 0.35).exp()
        + 0.461 * (-0.1548 * u_mps * 0.35).exp()
        - 0.397)
        .clamp(1.1, 6.5);
    let semi_a = ((final_area_m2 * lb_ratio) / std::f64::consts::PI)
        .sqrt()
        .max(45.0);
    let semi_b = (semi_a / lb_ratio).max(28.0);
    let ellipse_center = Vec2 {
        x: true_ignition.x + push_dir.x * semi_a * 0.68,
        y: true_ignition.y + push_dir.y * semi_a * 0.68,
    };
    let perp_dir = Vec2 {
        x: -push_dir.y,
        y: push_dir.x,
    };

    // Anchor VLAT Phos-Chek retardant line ahead of the fire head and Hotshot/Dozer flanking lines
    let mut retardant_drop_segments = Vec::new();
    if phos_chek_retardant_line_ft > 120.0 {
        let head_anchor = Vec2 {
            x: ellipse_center.x + push_dir.x * (semi_a * 1.05),
            y: ellipse_center.y + push_dir.y * (semi_a * 1.05),
        };
        let half_w = (phos_chek_retardant_line_ft * 0.3048 * 0.38).clamp(65.0, 420.0);
        retardant_drop_segments.push((
            Vec2 {
                x: head_anchor.x - perp_dir.x * half_w,
                y: head_anchor.y - perp_dir.y * half_w,
            },
            Vec2 {
                x: head_anchor.x + perp_dir.x * half_w,
                y: head_anchor.y + perp_dir.y * half_w,
            },
        ));
    }
    let mut handcrew_dozer_segments = Vec::new();
    for sign in [-1.0, 1.0] {
        let p_rear = Vec2 {
            x: true_ignition.x + perp_dir.x * sign * (semi_b * 0.55),
            y: true_ignition.y + perp_dir.y * sign * (semi_b * 0.55),
        };
        let p_flank = Vec2 {
            x: ellipse_center.x + push_dir.x * (semi_a * 0.72) + perp_dir.x * sign * (semi_b * 1.08),
            y: ellipse_center.y + push_dir.y * (semi_a * 0.72) + perp_dir.y * sign * (semi_b * 1.08),
        };
        handcrew_dozer_segments.push((p_rear, p_flank));
    }

    let mut parcel_burn_risk_pct = Vec::with_capacity(parcels.len());
    let mut parcel_camera_viewshed_pct = Vec::with_capacity(parcels.len());
    let mut parcel_evac_level = Vec::with_capacity(parcels.len());
    let mut parcel_triage = Vec::with_capacity(parcels.len());
    let mut structures_saved = 0.0;
    let mut structures_lost = 0.0;
    let mut level3_go_residents = 0.0;

    let engine_prep_shield = ((cfg.wui_wildland_engines as f64) / 10.0).clamp(0.20, 0.86);

    for p in parcels {
        let rel_x = p.center.x - ellipse_center.x;
        let rel_y = p.center.y - ellipse_center.y;
        let along = rel_x * push_dir.x + rel_y * push_dir.y;
        let across = rel_x * perp_dir.x + rel_y * perp_dir.y;
        let norm_ell = ((along / (semi_a + fb.spotting_distance_m * 0.35)).powi(2)
            + (across / (semi_b + fb.spotting_distance_m * 0.22)).powi(2))
        .sqrt();

        let raw_risk = ((-1.35 * norm_ell * norm_ell).exp() * 100.0).clamp(0.0, 99.0);
        let risk_pct = if final_containment_pct >= 92.0 {
            raw_risk * 0.58
        } else {
            raw_risk
        };

        // Count how many ALERTWest / OHAZ cameras have unobstructed optical/NIR viewshed over parcel `p`
        let vis_cams = cameras
            .iter()
            .filter(|c| p.center.dist(c.world_pos) <= c.day_range_mi * 1_609.34)
            .count();
        let viewshed_pct = ((vis_cams as f64 / 3.0) * 100.0).clamp(15.0, 100.0);

        // Oregon 3-Level Evacuation Standard: Level 3 (GO! Immediate) = tier 1, Level 2 (SET) = tier 2, Level 1 (READY) = tier 3
        let evac_lvl = if norm_ell <= 1.25 {
            level3_go_residents += p.residents();
            3u8
        } else if norm_ell <= 2.05 {
            2u8
        } else {
            1u8
        };

        let est_structures = (p.residents() / 2.6 + p.jobs() / 12.0).max(1.0);
        let triage = if norm_ell > 1.85 {
            WuiStructureTriageCategory::SafeOutsideZone
        } else if p.zone == ZoneCode::PARK || p.zone == ZoneCode::CBD {
            WuiStructureTriageCategory::DefensibleStandalone
        } else if fb.flame_length_ft > 11.0 && norm_ell < 0.85 {
            WuiStructureTriageCategory::NonDefensibleImmediateEvac
        } else {
            WuiStructureTriageCategory::DefensibleWithEnginePrep
        };

        if norm_ell <= 1.25 && p.zone != ZoneCode::PARK {
            let loss_prob = (risk_pct / 100.0) * (1.0 - engine_prep_shield * 0.72);
            let lost = est_structures * loss_prob * 0.42;
            structures_lost += lost;
            structures_saved += (est_structures - lost).max(0.0) * (risk_pct / 100.0);
        }

        parcel_burn_risk_pct.push(risk_pct);
        parcel_camera_viewshed_pct.push(viewshed_pct);
        parcel_evac_level.push(evac_lvl);
        parcel_triage.push(triage);
    }

    // 6. Build Timeline Points
    let mut timeline = Vec::with_capacity(n_steps);
    for st in 0..n_steps {
        let min_v = st * step_min;
        let m_area = fire_trials.iter().map(|r| r.area_curve[st]).sum::<f64>() * inv_t;
        let m_perim = fire_trials.iter().map(|r| r.perim_curve[st]).sum::<f64>() * inv_t;
        let m_line = fire_trials.iter().map(|r| r.line_curve[st]).sum::<f64>() * inv_t;
        let m_cont = fire_trials.iter().map(|r| r.contain_curve[st]).sum::<f64>() * inv_t;
        // IPAWS + ALERTWest early detection accelerates Level 3 WUI evacuation mobilization S-curve
        let evac_lag = mean_detection_to_dispatch_min + 8.0;
        let evac_prog = if (min_v as f64) <= mean_detection_to_dispatch_min {
            0.0
        } else {
            (1.0 - (-(((min_v as f64) - evac_lag).max(0.0) / 28.0)).exp()).clamp(0.0, 1.0)
        };
        timeline.push(WildfireTimelinePoint {
            minute: min_v,
            active_fire_area_acres: m_area,
            uncontained_perimeter_ft: m_perim,
            built_containment_line_ft: m_line,
            containment_pct: m_cont,
            flame_length_ft: fb.flame_length_ft * (1.0 - 0.35 * (m_cont / 100.0)),
            evacuated_wui_residents: level3_go_residents * evac_prog,
        });
    }

    WildfireAlertWestReport {
        config: cfg.clone(),
        solve_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        fire_behavior: fb,
        active_camera_count: cameras.len(),
        cameras_with_los_count: cameras_los,
        pnsn_shakealert_colocated_count: pnsn_colocated,
        true_ignition_pos: true_ignition,
        triangulated_ignition_pos: triangulated_pos,
        triangulation_cep90_m: cep90_m,
        mean_detection_to_dispatch_min,
        size_at_initial_attack_acres,
        final_burned_area_acres,
        p90_burned_area_acres,
        final_containment_pct,
        time_to_full_containment_min,
        total_handcrew_dozer_line_ft,
        phos_chek_retardant_line_ft,
        phos_chek_dropped_gallons,
        wui_structures_saved_count: structures_saved,
        expected_wui_structures_lost: structures_lost,
        level3_go_evac_residents: level3_go_residents,
        camera_sites: cameras,
        camera_bearing_rays,
        retardant_drop_segments,
        handcrew_dozer_segments,
        timeline,
        parcel_burn_risk_pct,
        parcel_camera_viewshed_pct,
        parcel_evac_level,
        parcel_triage,
    }
}

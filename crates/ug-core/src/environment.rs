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

use crate::multimodal::TransitPolicy;
use crate::network::{RoadNetwork, Vec2};
use crate::traffic::{Vehicle, VehicleClass};
use crate::zoning::Parcel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnvironmentalSnapshot {
    pub co2_kg_per_hr: f64,
    pub nox_g_per_hr: f64,
    pub pm25_g_per_hr: f64,
    pub corridor_noise_dba: f64,
    pub stormwater_runoff_lps: f64,
    pub heat_island_delta_c: f64,
    pub access_15min_pct: f64,
}

/// Computes Jimenez-Palacios (1999) Vehicle Specific Power (`VSP`, kW/tonne):
/// `VSP = v * (1.1 * a + 0.132) + 0.000302 * v^3`
pub fn vehicle_specific_power_kw_t(speed_mps: f64, accel_mps2: f64) -> f64 {
    speed_mps * (1.1 * accel_mps2.max(-2.0) + 0.132) + 0.000_302 * speed_mps.powi(3)
}

pub fn evaluate_environment_and_isochrones(
    net: &mut RoadNetwork,
    parcels: &mut [Parcel],
    vehicles: &[Vehicle],
    policy: &TransitPolicy,
    anchor_pos: Vec2,
) -> EnvironmentalSnapshot {
    for seg in &mut net.segments {
        seg.vsp_intensity *= 0.88;
    }

    let mut co2_gps = 0.0;
    let mut nox_mgps = 0.0;
    let mut pm25_mgps = 0.0;
    let mut sum_speed = 0.0;

    for v in vehicles {
        let vsp = vehicle_specific_power_kw_t(v.v, v.acc);
        let mass_mult = match v.v_class {
            VehicleClass::TransitBus => 2.6,
            VehicleClass::FreightTruck => 2.2,
            VehicleClass::EmergencyResponder => 1.85,
            VehicleClass::Sedan => 1.0,
        };
        let pos_vsp = vsp.max(0.0);
        co2_gps += (0.85 + pos_vsp * 0.24) * mass_mult;
        nox_mgps += (1.10 + pos_vsp * 0.65) * mass_mult;
        let brake_wear = if v.acc < -1.2 { 0.06 } else { 0.0 };
        pm25_mgps += (0.04 + pos_vsp * 0.022 + brake_wear) * mass_mult;
        sum_speed += v.v;

        if let Some(seg_id) = v.current_segment_id(&net.lanes) {
            net.segments[seg_id].vsp_intensity =
                (net.segments[seg_id].vsp_intensity + 0.045 * mass_mult).min(1.0);
        }
    }

    let mean_v = if vehicles.is_empty() {
        8.5
    } else {
        sum_speed / vehicles.len() as f64
    };

    let (mut area_sum, mut impv_sum, mut canopy_sum) = (0.0, 0.0, 0.0);
    for p in parcels.iter() {
        let sp = p.zone.spec();
        area_sum += p.area_ha;
        impv_sum += sp.impervious_fraction * p.area_ha;
        canopy_sum += sp.canopy_fraction * p.area_ha;
    }
    let avg_impv = if area_sum > 0.0 { impv_sum / area_sum } else { 0.6 };
    let avg_canopy = if area_sum > 0.0 { canopy_sum / area_sum } else { 0.25 };

    // Rational Method Q = C * i * A for 25 mm/h storm (69.44 L/s per impervious ha)
    let runoff_lps = avg_impv * 69.44 * area_sum;
    let heat_delta_c = (2.8 * avg_impv - 2.6 * avg_canopy).max(-0.4);
    let total_lane_km: f64 = net
        .segments
        .iter()
        .filter(|s| !s.closed)
        .map(|s| (s.lane_count.max(1) as f64) * s.length / 1000.0)
        .sum::<f64>()
        .max(0.25);
    let veh_per_lane_km = vehicles.len() as f64 / total_lane_km;
    let heavy_equiv = if vehicles.is_empty() {
        1.0
    } else {
        vehicles
            .iter()
            .map(|v| match v.v_class {
                VehicleClass::TransitBus | VehicleClass::FreightTruck => 4.5,
                VehicleClass::EmergencyResponder => 3.5,
                VehicleClass::Sedan => 1.0,
            })
            .sum::<f64>()
            / (vehicles.len() as f64)
    };
    let speed_kph = (mean_v * 3.6).max(5.0);
    let rolling_factor = (speed_kph / 35.0).powf(2.4);
    let acoustic_energy =
        1.0 + veh_per_lane_km * 0.25 * heavy_equiv * (0.65 + rolling_factor);
    let noise_dba =
        (46.0 + 10.0 * acoustic_energy.log10() - 2.5 * avg_canopy).clamp(38.0, 92.0);

    // Moreno et al. (2021) 15-Minute City multi-modal accessibility routed over active road/bridge graph
    let total_km: f64 = net.segments.iter().filter(|s| !s.closed).map(|s| s.length / 1000.0).sum();
    let bike_km: f64 = net
        .segments
        .iter()
        .filter(|s| !s.closed && s.has_bike_lane)
        .map(|s| s.length / 1000.0)
        .sum();
    let bike_cov = (bike_km / (total_km * 0.65).max(0.25)).min(1.0);
    let eff_m_per_min = (80.0
        * (1.0 + 0.85 * bike_cov + 0.95 * (15.0 / policy.headway_min.max(4.0)) * 0.45))
        .max(65.0);

    let n_segs = net.segments.len();
    let seg_mid = |pts: &[Vec2]| -> Vec2 {
        if pts.is_empty() {
            Vec2 { x: 0.0, y: 0.0 }
        } else {
            pts[pts.len() / 2]
        }
    };
    let mut seg_mids = Vec::with_capacity(n_segs);
    for s in &net.segments {
        seg_mids.push(seg_mid(&s.base_pts));
    }

    // Build junction-based multimodal adjacency across open road & bridge segments (`!seg.closed`)
    let mut seg_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_segs];
    let seg_transit_min = |sid: usize| -> f64 {
        let s = &net.segments[sid];
        let mode_mult = (if s.has_bike_lane { 1.15 } else { 1.0 })
            * (if s.has_bus_lane { 1.10 } else { 1.0 });
        s.length / (eff_m_per_min * mode_mult)
    };
    for j in &net.junctions {
        let mut incident: Vec<usize> = j
            .in_segs
            .iter()
            .chain(j.out_segs.iter())
            .copied()
            .filter(|&sid| sid < n_segs && !net.segments[sid].closed)
            .collect();
        incident.sort_unstable();
        incident.dedup();
        for &u in &incident {
            for &v in &incident {
                if u != v {
                    seg_adj[u].push((v, seg_transit_min(v)));
                }
            }
        }
    }

    // Multi-modal Dijkstra from open road segment nearest `anchor_pos`
    #[derive(Copy, Clone, PartialEq)]
    struct NodeCost {
        min_cost: f64,
        sid: usize,
    }
    impl Eq for NodeCost {}
    impl Ord for NodeCost {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .min_cost
                .partial_cmp(&self.min_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.sid.cmp(&other.sid))
        }
    }
    impl PartialOrd for NodeCost {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut seg_min_time = vec![f64::INFINITY; n_segs];
    let mut heap = std::collections::BinaryHeap::new();
    let anchor_seed = (0..n_segs)
        .filter(|&sid| !net.segments[sid].closed && !net.segments[sid].base_pts.is_empty())
        .min_by(|&a, &b| {
            let da = seg_mids[a].dist(anchor_pos);
            let db = seg_mids[b].dist(anchor_pos);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(seed_sid) = anchor_seed {
        let d_seed_min = seg_mids[seed_sid].dist(anchor_pos) / eff_m_per_min;
        let t0 = d_seed_min + 0.5 * seg_transit_min(seed_sid);
        seg_min_time[seed_sid] = t0;
        heap.push(NodeCost {
            min_cost: t0,
            sid: seed_sid,
        });
    }
    while let Some(NodeCost { min_cost, sid }) = heap.pop() {
        if min_cost > seg_min_time[sid] {
            continue;
        }
        for &(nxt, edge_m) in &seg_adj[sid] {
            let nc = min_cost + edge_m;
            if nc < seg_min_time[nxt] {
                seg_min_time[nxt] = nc;
                heap.push(NodeCost {
                    min_cost: nc,
                    sid: nxt,
                });
            }
        }
    }

    let mut within_15 = 0usize;
    for p in parcels.iter_mut() {
        // Attach parcel to its nearest local road segment; disconnected islands across closed bridges remain unreachable
        let nearest_sid = (0..n_segs)
            .filter(|&sid| !net.segments[sid].base_pts.is_empty())
            .min_by(|&a, &b| {
                let da = seg_mids[a].dist(p.center);
                let db = seg_mids[b].dist(p.center);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        p.isochrone_minutes = if let Some(sid) = nearest_sid {
            if !net.segments[sid].closed && seg_min_time[sid].is_finite() {
                seg_min_time[sid] + (seg_mids[sid].dist(p.center) / eff_m_per_min) + 1.2
            } else {
                999.0
            }
        } else {
            999.0
        };
        if p.isochrone_minutes <= 15.0 {
            within_15 += 1;
        }
    }
    let access_pct = if parcels.is_empty() {
        0.0
    } else {
        (within_15 as f64 / parcels.len() as f64) * 100.0
    };

    EnvironmentalSnapshot {
        co2_kg_per_hr: co2_gps * 3.6,
        nox_g_per_hr: nox_mgps * 3.6,
        pm25_g_per_hr: pm25_mgps * 3.6,
        corridor_noise_dba: noise_dba,
        stormwater_runoff_lps: runoff_lps,
        heat_island_delta_c: heat_delta_c,
        access_15min_pct: access_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::RoadNetwork;
    use crate::osm::{builtin_riverfront_bridges_osm, parse_osm_xml_str};
    use crate::traffic::{CarrierRef, SimulationWorld, Vehicle, VehicleClass};

    #[test]
    fn closing_all_river_bridges_lowers_access_15min_pct_and_noise_scales_with_density_speed() {
        let osm = parse_osm_xml_str(&builtin_riverfront_bridges_osm()).expect("builtin osm");
        let net_open = RoadNetwork::compile(&osm);
        let mut world = SimulationWorld::new(net_open);
        let open_access = world.env_snapshot.access_15min_pct;

        let bridge_count = world.network.segments.iter().filter(|s| s.is_bridge).count();
        assert!(bridge_count >= 1, "Riverfront metro must include river bridge spans");
        for seg in &mut world.network.segments {
            if seg.is_bridge {
                seg.closed = true;
            }
        }
        let anchor = world
            .network
            .junctions
            .first()
            .map(|j| j.pos)
            .unwrap_or(Vec2 { x: 0.0, y: 0.0 });
        let closed_snap = evaluate_environment_and_isochrones(
            &mut world.network,
            &mut world.parcels,
            &world.vehicles,
            &world.transit_policy,
            anchor,
        );
        assert!(
            closed_snap.access_15min_pct < open_access,
            "Closing all {bridge_count} river bridges must lower 15-min graph access ({} -> {})",
            open_access,
            closed_snap.access_15min_pct
        );

        // Noise level scales physically with lane density & rolling speed
        let mut low_speed_vehs = Vec::new();
        let mut high_speed_vehs = Vec::new();
        for i in 0..40 {
            let v_low = Vehicle {
                id: i + 1,
                v_class: VehicleClass::Sedan,
                carrier: CarrierRef::Lane(0),
                s_m: 10.0,
                length_m: 4.5,
                width_m: 1.9,
                v_max_mps: 25.0,
                accel_mult: 1.0,
                v: 3.0,
                acc: 0.0,
                lat_visual_offset_m: 0.0,
                lc_cooldown_s: 0.0,
                dest_seg_id: None,
                planned_conn_id: None,
                turn_intent: crate::network::TurnType::Through,
                spawn_time_s: 0.0,
                gtfs_route_idx: None,
                gtfs_route_seq: 0,
                gtfs_next_stop_order: 0,
                dwell_remaining_s: 0.0,
                onboard_pax: 0,
                crush_capacity: 0,
            };
            let mut v_fast = v_low.clone();
            v_fast.v = 20.0;
            low_speed_vehs.push(v_low);
            high_speed_vehs.push(v_fast);
        }
        let s_low = evaluate_environment_and_isochrones(
            &mut world.network,
            &mut world.parcels,
            &low_speed_vehs,
            &world.transit_policy,
            anchor,
        );
        let s_high = evaluate_environment_and_isochrones(
            &mut world.network,
            &mut world.parcels,
            &high_speed_vehs,
            &world.transit_policy,
            anchor,
        );
        assert!(
            s_high.corridor_noise_dba > s_low.corridor_noise_dba + 2.0,
            "Faster rolling speeds must elevate corridor dBA ({:.2} vs {:.2})",
            s_high.corridor_noise_dba,
            s_low.corridor_noise_dba
        );
    }
}

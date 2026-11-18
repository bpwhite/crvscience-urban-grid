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

#![cfg(feature = "tui")]

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};
use ratatui::Terminal;
use std::io::stdout;
use std::path::Path;
use std::time::{Duration, Instant};
use ug_core::{EditCategory, PlumePhysicsParams, SimulationWorld};

pub fn run_tui(mut world: SimulationWorld) -> std::io::Result<()> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut paused = false;
    let mut last_tick = Instant::now();
    let mut status_msg: Option<(String, bool)> = None;

    loop {
        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char(' ') => paused = !paused,
                    KeyCode::Char('t') => {
                        world.apply_tod_upzoning_tracked();
                    }
                    KeyCode::Char('b') => {
                        world.execute_topology_edit(
                            EditCategory::CorridorPolicy,
                            "Toggle Way #799 Closure",
                            |w| {
                                for seg in &mut w.network.segments {
                                    if seg.way_id == "799" {
                                        seg.closed = !seg.closed;
                                    }
                                }
                                w.network.recompute_dynamic_routes();
                            },
                        );
                    }
                    KeyCode::Char('m') => {
                        world.execute_topology_edit(
                            EditCategory::SignalControl,
                            "Toggle Citywide Max-Pressure",
                            |w| {
                                if w.mp_telemetry.controlled_junctions > 0 {
                                    w.network.restore_fixed_time_signals();
                                } else {
                                    w.network.enable_network_max_pressure();
                                }
                            },
                        );
                    }
                    KeyCode::Char('u') => {
                        let _ = world.undo_edit();
                    }
                    KeyCode::Char('r') => {
                        let _ = world.redo_edit();
                    }
                    KeyCode::Char('v') => {
                        let _ = world.dispatch_emergency_responder();
                    }
                    KeyCode::Char('s') => {
                        match world.save_snapshot_json(
                            Path::new("exports/tui_snapshot.json"),
                            PlumePhysicsParams::default(),
                            "TUI Snapshot",
                            world.total_active_agents() <= 15_000,
                        ) {
                            Ok(bytes) => {
                                status_msg = Some((
                                    format!("Saved exports/tui_snapshot.json ({bytes} B)"),
                                    false,
                                ));
                            }
                            Err(e) => {
                                status_msg = Some((format!("Snapshot save failed: {e}"), true));
                            }
                        }
                    }
                    KeyCode::Char('e') => {
                        match world.export_telemetry_csv(Path::new("exports/tui_telemetry.csv")) {
                            Ok(rows) => {
                                status_msg = Some((
                                    format!("Exported {rows} rows -> exports/tui_telemetry.csv"),
                                    false,
                                ));
                            }
                            Err(e) => {
                                status_msg = Some((format!("CSV export failed: {e}"), true));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let elapsed = last_tick.elapsed().as_secs_f64().min(0.1);
        last_tick = Instant::now();
        if !paused {
            let sub = ((elapsed / 0.05).ceil() as usize).max(1);
            for _ in 0..sub {
                world.step(elapsed / sub as f64);
            }
        }

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(6),
                    Constraint::Length(6),
                    Constraint::Min(5),
                ])
                .split(f.area());

            let status_sfx = match &status_msg {
                Some((msg, _)) => format!(" | {msg}"),
                None => String::new(),
            };
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    " urban-grid v5.2 TUI ",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "| t={:.1}s | Veh={} Queued={} | Speed={:.1} mph | HCM LOS {} | [Space] Pause [V] Dispatch EV [M] MP [T] TOD [U/R] Undo [S] JSON [E] CSV [Q] Quit{}",
                    world.sim_time_s,
                    world.vehicles.len(),
                    world.queued_vehicles(),
                    world.mean_speed_mph(),
                    world.ledger.hcm_los,
                    status_sfx
                )),
            ]))
            .block(Block::default().borders(Borders::ALL).title("Simulation Telemetry"));
            f.render_widget(header, chunks[0]);

            let ms = world.mode_shares;
            let mode_gauge = Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("McFadden (1974) Mode Split — Active + Transit Share vs. Private Auto"),
                )
                .gauge_style(Style::default().fg(Color::Green))
                .percent(((1.0 - ms.auto) * 100.0).round() as u16)
                .label(format!(
                    "Sustainable Mode Share {:.0}% (Bus {:.0}% | Bike {:.0}% | Walk {:.0}%) — Auto {:.0}%",
                    (1.0 - ms.auto) * 100.0,
                    ms.transit * 100.0,
                    ms.bike * 100.0,
                    ms.walk * 100.0,
                    ms.auto * 100.0
                ));
            f.render_widget(mode_gauge, chunks[1]);

            let env = world.env_snapshot;
            let led = &world.ledger;
            let ledger_text = Paragraph::new(vec![
                Line::from(format!(
                    "Residents: {}  |  Jobs: {} (J/H Ratio {:.2})  |  15-Min Access: {:.0}% of parcels",
                    led.total_residents, led.total_jobs, led.jobs_housing_ratio, env.access_15min_pct
                )),
                Line::from(format!(
                    "VSP Emissions: {:.1} kg CO2/h, {:.1} g PM2.5/h  |  Noise: {:.1} dBA  |  Stormwater Runoff: {:.0} L/s",
                    env.co2_kg_per_hr, env.pm25_g_per_hr, env.corridor_noise_dba, env.stormwater_runoff_lps
                )),
                Line::from(format!(
                    "Municipal Fiscal: {:+.1}k/ac/yr | GTFS: {} buses, {} onboard ({:.0}% load, LOS {}), {:.1}s dwell, {} curb wait",
                    led.net_fiscal_per_acre_k_yr,
                    world.transit_gtfs.telemetry.active_scheduled_buses,
                    world.transit_gtfs.telemetry.onboard_passengers,
                    world.transit_gtfs.telemetry.mean_load_factor_pct,
                    world.transit_gtfs.telemetry.tcqsm_crowding_los,
                    world.transit_gtfs.telemetry.mean_dwell_s,
                    world.transit_gtfs.telemetry.waiting_passengers
                )),
            ])
            .block(Block::default().borders(Borders::ALL).title("Urban Planning & Environmental Ledger"));
            f.render_widget(ledger_text, chunks[2]);

            let mut lines = Vec::new();
            for seg in world.network.segments.iter().take(8) {
                let count = world
                    .vehicles
                    .iter()
                    .filter(|v| v.current_segment_id(&world.network.lanes) == Some(seg.id))
                    .count();
                let bar = "#".repeat(count.min(36));
                lines.push(Line::from(format!(
                    "{:<28} | {:>2} ln | {:>4.1} mph | {:<36} ({})",
                    seg.name,
                    seg.lane_count,
                    seg.ema_speed_mps * 2.236_94,
                    bar,
                    count
                )));
            }
            let corridors = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("Corridor Queue & Speed Monitor"));
            f.render_widget(corridors, chunks[3]);
        })?;
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

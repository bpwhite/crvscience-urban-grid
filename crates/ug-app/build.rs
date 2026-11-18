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

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn blend_pixel(dst: &mut [u8], r: u8, g: u8, b: u8, alpha: f32) {
    let a = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - a;
    dst[0] = (dst[0] as f32 * inv + r as f32 * a).round().clamp(0.0, 255.0) as u8;
    dst[1] = (dst[1] as f32 * inv + g as f32 * a).round().clamp(0.0, 255.0) as u8;
    dst[2] = (dst[2] as f32 * inv + b as f32 * a).round().clamp(0.0, 255.0) as u8;
    dst[3] = dst[3].max((a * 255.0).round().clamp(0.0, 255.0) as u8);
}

fn render_urban_grid_rgba(size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; size * size * 4];
    let s = size as f32;
    let half = 0.5 * s;
    let margin = (0.06 * s).max(1.0);
    let box_half = half - margin;
    let corner_r = (0.20 * s).max(2.0);

    for y in 0..size {
        for x in 0..size {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let dx = (px - half).abs() - (box_half - corner_r);
            let dy = (py - half).abs() - (box_half - corner_r);
            let sd_badge =
                dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0) - corner_r;

            let idx = (y * size + x) * 4;
            if sd_badge > 0.75 {
                continue;
            }
            let edge_cov = (0.75 - sd_badge).clamp(0.0, 1.0);

            // Deep slate rounded badge background (#0F172A) with subtle vertical gradient
            let v_grad = py / s;
            let mut r = 15.0 + 6.0 * (1.0 - v_grad);
            let mut g = 23.0 + 10.0 * (1.0 - v_grad);
            let mut b = 42.0 + 18.0 * (1.0 - v_grad);

            // Subtle slate rim (#38BDF8 tint)
            if sd_badge > -1.4_f32.max(-0.045 * s) {
                let rim = ((sd_badge + (0.045 * s).max(1.2)) / ((0.045 * s).max(1.2))).clamp(0.0, 1.0);
                r = r * (1.0 - 0.55 * rim) + 56.0 * (0.55 * rim);
                g = g * (1.0 - 0.55 * rim) + 189.0 * (0.55 * rim);
                b = b * (1.0 - 0.55 * rim) + 248.0 * (0.55 * rim);
            }

            let pixel = &mut buf[idx..idx + 4];
            pixel[0] = r.round() as u8;
            pixel[1] = g.round() as u8;
            pixel[2] = b.round() as u8;
            pixel[3] = (edge_cov * 255.0).round() as u8;

            // Normalized coords in [-1, 1] inside badge
            let nx = (px - half) / box_half;
            let ny = (py - half) / box_half;

            // Secondary emerald grid avenues (#10B981) at +/-0.48
            let g_minor_w = (0.055 * s).max(0.9) / box_half;
            let d_minor_x = ((nx.abs() - 0.48).abs()).min((ny.abs() - 0.48).abs());
            if d_minor_x < g_minor_w && nx.abs() < 0.86 && ny.abs() < 0.86 {
                let cov = (1.0 - d_minor_x / g_minor_w).clamp(0.0, 1.0) * 0.72 * edge_cov;
                blend_pixel(pixel, 16, 185, 129, cov);
            }

            // Primary cyan orthogonal arterial cross (#38BDF8)
            let art_w = (0.085 * s).max(1.2) / box_half;
            let d_art = nx.abs().min(ny.abs());
            if d_art < art_w && nx.abs() < 0.90 && ny.abs() < 0.90 {
                let cov = (1.0 - d_art / art_w).clamp(0.0, 1.0) * 0.92 * edge_cov;
                blend_pixel(pixel, 56, 189, 248, cov);
            }

            // Diagonal transit/bypass corridor
            let d_diag = ((nx + ny).abs() * std::f32::consts::FRAC_1_SQRT_2) * 0.92;
            if d_diag < g_minor_w * 0.85 && nx.hypot(ny) < 0.82 {
                let cov = (1.0 - d_diag / (g_minor_w * 0.85)).clamp(0.0, 1.0) * 0.45 * edge_cov;
                blend_pixel(pixel, 125, 211, 252, cov);
            }

            // 4-min EMS pulse ring in amber (#FACC15) around central interchange
            let rad = nx.hypot(ny);
            let ring_d = (rad - 0.44).abs();
            let ring_w = (0.070 * s).max(1.0) / box_half;
            if ring_d < ring_w {
                let cov = (1.0 - ring_d / ring_w).clamp(0.0, 1.0) * 0.95 * edge_cov;
                blend_pixel(pixel, 250, 204, 21, cov);
            }

            // Central interchange hub (dark slate core + crisp amber/crimson beacon)
            if rad < 0.24 {
                let cov = ((0.24 - rad) / 0.05).clamp(0.0, 1.0) * edge_cov;
                blend_pixel(pixel, 15, 23, 42, cov);
            }
            if rad < 0.145 {
                let cov = ((0.145 - rad) / 0.04).clamp(0.0, 1.0) * edge_cov;
                blend_pixel(pixel, 244, 63, 94, cov);
            }
            if rad < 0.065 {
                let cov = ((0.065 - rad) / 0.03).clamp(0.0, 1.0) * edge_cov;
                blend_pixel(pixel, 254, 240, 138, cov);
            }
        }
    }
    buf
}

fn write_ico(path: &Path, sizes: &[usize]) -> std::io::Result<()> {
    let mut images: Vec<(usize, Vec<u8>)> = Vec::with_capacity(sizes.len());
    for &sz in sizes {
        let rgba = render_urban_grid_rgba(sz);
        let and_stride = ((sz + 31) / 32) * 4;
        let xor_len = sz * sz * 4;
        let and_len = and_stride * sz;
        let mut dib = Vec::with_capacity(40 + xor_len + and_len);

        // BITMAPINFOHEADER (40 bytes)
        dib.extend_from_slice(&40u32.to_le_bytes());
        dib.extend_from_slice(&(sz as i32).to_le_bytes());
        dib.extend_from_slice(&((sz * 2) as i32).to_le_bytes()); // height * 2 (XOR + AND)
        dib.extend_from_slice(&1u16.to_le_bytes()); // planes
        dib.extend_from_slice(&32u16.to_le_bytes()); // bpp
        dib.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
        dib.extend_from_slice(&((xor_len + and_len) as u32).to_le_bytes());
        dib.extend_from_slice(&0i32.to_le_bytes());
        dib.extend_from_slice(&0i32.to_le_bytes());
        dib.extend_from_slice(&0u32.to_le_bytes());
        dib.extend_from_slice(&0u32.to_le_bytes());

        // Bottom-up BGRA scanlines
        for row in (0..sz).rev() {
            for col in 0..sz {
                let i = (row * sz + col) * 4;
                dib.push(rgba[i + 2]);
                dib.push(rgba[i + 1]);
                dib.push(rgba[i]);
                dib.push(rgba[i + 3]);
            }
        }
        // 1-bit AND mask (0 = use alpha channel)
        dib.resize(dib.len() + and_len, 0u8);
        images.push((sz, dib));
    }

    let count = images.len() as u16;
    let mut out = Vec::new();
    // ICONDIR
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // type = 1 (ICO)
    out.extend_from_slice(&count.to_le_bytes());

    let mut offset = 6usize + 16 * images.len();
    for (sz, dib) in &images {
        let dim_byte = if *sz >= 256 { 0u8 } else { *sz as u8 };
        out.push(dim_byte);
        out.push(dim_byte);
        out.push(0u8); // palette
        out.push(0u8); // reserved
        out.extend_from_slice(&1u16.to_le_bytes()); // planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bpp
        out.extend_from_slice(&(dib.len() as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += dib.len();
    }
    for (_, dib) in images {
        out.extend_from_slice(&dib);
    }
    fs::write(path, out)
}

fn find_rc_exe() -> Option<PathBuf> {
    if let Ok(path_var) = env::var("PATH") {
        for dir in env::split_paths(&path_var) {
            let cand = dir.join("rc.exe");
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    for root_var in ["ProgramFiles(x86)", "ProgramFiles"] {
        if let Ok(root) = env::var(root_var) {
            let kits_bin = Path::new(&root).join("Windows Kits").join("10").join("bin");
            if let Ok(entries) = fs::read_dir(&kits_bin) {
                let mut versions: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|x| x.path()))
                    .filter(|p| p.is_dir())
                    .collect();
                versions.sort();
                versions.reverse();
                for ver in versions {
                    let cand = ver.join("x64").join("rc.exe");
                    if cand.is_file() {
                        return Some(cand);
                    }
                }
            }
        }
    }
    None
}

fn find_windres() -> Option<String> {
    for cand in ["windres", "x86_64-w64-mingw32-windres"] {
        if Command::new(cand).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(cand.to_string());
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let rgba64 = render_urban_grid_rgba(64);
    fs::write(out_dir.join("urban_grid_icon_64.rgba"), &rgba64)
        .expect("Failed to write urban_grid_icon_64.rgba");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let ico_path = out_dir.join("urban_grid.ico");
    write_ico(&ico_path, &[16, 32, 48, 64]).expect("Failed to write urban_grid.ico");

    let ico_rc_str = ico_path.to_string_lossy().replace('\\', "/");
    let rc_content = format!(
        r#"1 ICON "{ico_rc_str}"

1 VERSIONINFO
FILEVERSION 5,2,0,0
PRODUCTVERSION 5,2,0,0
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
            VALUE "CompanyName", "CRV Science Network\0"
            VALUE "FileDescription", "Urban-Grid Microscopic Traffic, GTFS Transit & Multi-Hazard Lifeline Workbench\0"
            VALUE "FileVersion", "5.2.0.0\0"
            VALUE "InternalName", "urban-grid\0"
            VALUE "LegalCopyright", "Copyright CRV Science Network (GPL-3.0-or-later)\0"
            VALUE "OriginalFilename", "12_Urban-Grid.exe\0"
            VALUE "ProductName", "Urban-Grid\0"
            VALUE "ProductVersion", "5.2.0\0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0409, 1200
    END
END
"#
    );
    let rc_path = out_dir.join("app_res.rc");
    fs::write(&rc_path, rc_content).expect("Failed to write app_res.rc");

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "gnu" {
        if let Some(windres) = find_windres() {
            let obj_path = out_dir.join("app_res.o");
            let status = Command::new(&windres)
                .args([
                    rc_path.to_string_lossy().as_ref(),
                    "-O",
                    "coff",
                    "-o",
                    obj_path.to_string_lossy().as_ref(),
                ])
                .status();
            if status.map(|s| s.success()).unwrap_or(false) {
                println!("cargo:rustc-link-arg={}", obj_path.display());
            } else {
                println!("cargo:warning=windres failed to compile Windows resource file; skipping embedded icon/VERSIONINFO");
            }
        } else {
            println!("cargo:warning=windres not found on PATH; skipping embedded Windows icon/VERSIONINFO");
        }
    } else if target_env == "msvc" {
        if let Some(rc_exe) = find_rc_exe() {
            let res_path = out_dir.join("app_res.res");
            let status = Command::new(&rc_exe)
                .args([
                    "/nologo",
                    "/fo",
                    res_path.to_string_lossy().as_ref(),
                    rc_path.to_string_lossy().as_ref(),
                ])
                .status();
            if status.map(|s| s.success()).unwrap_or(false) {
                println!("cargo:rustc-link-arg={}", res_path.display());
            } else {
                println!("cargo:warning=rc.exe failed to compile Windows resource file; skipping embedded icon/VERSIONINFO");
            }
        } else {
            println!("cargo:warning=rc.exe not found; skipping embedded Windows icon/VERSIONINFO");
        }
    }
}

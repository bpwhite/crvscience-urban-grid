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

#![cfg(feature = "gui")]

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use ug_core::{
    GpuAgentInstance, GpuCameraUniform, GpuPlumeComputeUniform, GpuRoadQuadInstance,
    HEATMAP_BYTES_PER_ROW, HEATMAP_DIM,
};

const INITIAL_ROAD_QUAD_CAPACITY: usize = 16_384;
const INITIAL_AGENT_CAPACITY: usize = 65_536;
const GRID_CELL_COUNT: usize = HEATMAP_DIM * HEATMAP_DIM;

/// Payload transferred lock-free once per frame from `UrbanGridGuiApp::update` into `CallbackTrait::prepare`.
pub struct ViewportFramePayload {
    pub camera: GpuCameraUniform,
    pub plume_compute: Option<GpuPlumeComputeUniform>,
    pub road_quads: Vec<GpuRoadQuadInstance>,
    pub agents: Vec<GpuAgentInstance>,
    /// Populated only for discrete parcel/static layers when dirty;
    /// dynamic `VspEmissions` and `AgentDensity` plumes execute 100% on GPU via `plume_compute.wgsl`.
    pub heatmap_rgba: Option<Vec<u8>>,
    /// Normalized `[0..1]` scalar field matching `heatmap_rgba` for contour extraction without RGB luma distortion.
    pub heatmap_scalar: Option<Vec<f32>>,
    /// Recycled capacity-warm vectors swapped back after `queue.write_buffer` to avoid per-frame deep clones.
    pub recycle_pool: Arc<Mutex<(Vec<GpuRoadQuadInstance>, Vec<GpuAgentInstance>)>>,
}

/// Persistent GPU buffers, ping-pong `@compute` storage fields, bilinear heatmap texture, and
/// compiled WGSL compute + render pipelines stored in `egui_wgpu::Renderer::callback_resources`.
pub struct UrbanViewportGpuResources {
    camera_buffer: wgpu::Buffer,
    plume_uniform_buffer: wgpu::Buffer,
    accum_atomic_buffer: wgpu::Buffer,
    plume_state_a: wgpu::Buffer,
    plume_state_b: wgpu::Buffer,
    cpu_scalar_buffer: wgpu::Buffer,
    heatmap_texture: wgpu::Texture,
    heatmap_view: wgpu::TextureView,
    compute_bgl: wgpu::BindGroupLayout,
    compute_bg_a_to_b: wgpu::BindGroup,
    compute_bg_b_to_a: wgpu::BindGroup,
    ping_pong_flip: bool,
    use_plume_scalar_buf: bool,
    cs_clear_pipeline: wgpu::ComputePipeline,
    cs_splat_pipeline: wgpu::ComputePipeline,
    cs_advect_pipeline: wgpu::ComputePipeline,
    render_bg_plume_a: wgpu::BindGroup,
    render_bg_plume_b: wgpu::BindGroup,
    render_bg_cpu: wgpu::BindGroup,
    heatmap_pipeline: wgpu::RenderPipeline,
    road_pipeline: wgpu::RenderPipeline,
    agent_pipeline: wgpu::RenderPipeline,
    road_buffer: wgpu::Buffer,
    road_capacity: usize,
    pub road_count: u32,
    agent_buffer: wgpu::Buffer,
    agent_capacity: usize,
    pub agent_count: u32,
    pub heatmap_enabled: bool,
}

impl UrbanViewportGpuResources {
    fn build_compute_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        plume_uniform_buffer: &wgpu::Buffer,
        agent_buffer: &wgpu::Buffer,
        accum_atomic_buffer: &wgpu::Buffer,
        plume_state_a: &wgpu::Buffer,
        plume_state_b: &wgpu::Buffer,
        heatmap_view: &wgpu::TextureView,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let make_bg = |label: &str, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: plume_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: agent_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: accum_atomic_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(heatmap_view),
                    },
                ],
            })
        };
        (
            make_bg("urban_plume_bg_a_to_b", plume_state_a, plume_state_b),
            make_bg("urban_plume_bg_b_to_a", plume_state_b, plume_state_a),
        )
    }

    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("urban_viewport_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/urban_viewport.wgsl").into()),
        });
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("urban_plume_compute_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/plume_compute.wgsl").into()),
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_camera_uniform_buf"),
            size: size_of::<GpuCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let plume_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_plume_compute_uniform_buf"),
            size: size_of::<GpuPlumeComputeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accum_atomic_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_plume_accum_atomic_buf"),
            size: (GRID_CELL_COUNT * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let plume_state_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_plume_state_a"),
            size: (GRID_CELL_COUNT * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let plume_state_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_plume_state_b"),
            size: (GRID_CELL_COUNT * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cpu_scalar_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_cpu_scalar_field"),
            size: (GRID_CELL_COUNT * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let heatmap_size = wgpu::Extent3d {
            width: HEATMAP_DIM as u32,
            height: HEATMAP_DIM as u32,
            depth_or_array_layers: 1,
        };
        let heatmap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("urban_heatmap_rgba8_tex"),
            size: heatmap_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let heatmap_view = heatmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let heatmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("urban_heatmap_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let road_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_road_instance_buf"),
            size: (INITIAL_ROAD_QUAD_CAPACITY * size_of::<GpuRoadQuadInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared across BOTH `@compute` (`STORAGE`) and `vs_agent` (`VERTEX`) — zero duplicate copy!
        let agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("urban_agent_instance_buf"),
            size: (INITIAL_AGENT_CAPACITY * size_of::<GpuAgentInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Compute Pipeline Setup ---
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("urban_plume_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuPlumeComputeUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let (compute_bg_a_to_b, compute_bg_b_to_a) = Self::build_compute_bind_groups(
            device,
            &compute_bgl,
            &plume_uniform_buffer,
            &agent_buffer,
            &accum_atomic_buffer,
            &plume_state_a,
            &plume_state_b,
            &heatmap_view,
        );

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("urban_plume_compute_layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });

        let make_cp = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let cs_clear_pipeline = make_cp("urban_cs_clear_accum", "cs_clear_accum");
        let cs_splat_pipeline = make_cp("urban_cs_splat_agents", "cs_splat_agents");
        let cs_advect_pipeline = make_cp("urban_cs_advect_diffuse", "cs_advect_diffuse");

        // --- Render Pipeline Setup ---
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("urban_viewport_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<GpuCameraUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let make_render_bg = |label: &str, scalar_buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &render_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&heatmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&heatmap_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scalar_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let render_bg_plume_a = make_render_bg("urban_viewport_bg_plume_a", &plume_state_a);
        let render_bg_plume_b = make_render_bg("urban_viewport_bg_plume_b", &plume_state_b);
        let render_bg_cpu = make_render_bg("urban_viewport_bg_cpu", &cpu_scalar_buffer);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("urban_viewport_pipeline_layout"),
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            });

        let alpha_target = [Some(wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("urban_heatmap_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_heatmap"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_heatmap"),
                targets: &alpha_target,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let road_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuRoadQuadInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 20,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 24,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 28,
                    shader_location: 5,
                },
            ],
        };

        let road_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("urban_road_instanced_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_road"),
                buffers: &[road_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_road"),
                targets: &alpha_target,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let agent_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuAgentInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 16,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 20,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Unorm8x4,
                    offset: 24,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 28,
                    shader_location: 6,
                },
            ],
        };

        let agent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("urban_agent_instanced_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_agent"),
                buffers: &[agent_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_agent"),
                targets: &alpha_target,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            camera_buffer,
            plume_uniform_buffer,
            accum_atomic_buffer,
            plume_state_a,
            plume_state_b,
            cpu_scalar_buffer,
            heatmap_texture,
            heatmap_view,
            compute_bgl,
            compute_bg_a_to_b,
            compute_bg_b_to_a,
            ping_pong_flip: false,
            use_plume_scalar_buf: false,
            cs_clear_pipeline,
            cs_splat_pipeline,
            cs_advect_pipeline,
            render_bg_plume_a,
            render_bg_plume_b,
            render_bg_cpu,
            heatmap_pipeline,
            road_pipeline,
            agent_pipeline,
            road_buffer,
            road_capacity: INITIAL_ROAD_QUAD_CAPACITY,
            road_count: 0,
            agent_buffer,
            agent_capacity: INITIAL_AGENT_CAPACITY,
            agent_count: 0,
            heatmap_enabled: true,
        }
    }

    pub fn upload_and_dispatch_compute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &mut ViewportFramePayload,
    ) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&frame.camera));
        self.heatmap_enabled = frame.camera.heatmap_mode != 0;
        self.use_plume_scalar_buf = matches!(frame.camera.heatmap_mode, 3 | 4);

        // Upload discrete CPU textures ONLY for static parcel / resilience overlays
        if let Some(rgba) = &frame.heatmap_rgba {
            let expected = (HEATMAP_BYTES_PER_ROW as usize) * HEATMAP_DIM;
            if rgba.len() == expected {
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.heatmap_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    rgba,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(HEATMAP_BYTES_PER_ROW),
                        rows_per_image: Some(HEATMAP_DIM as u32),
                    },
                    wgpu::Extent3d {
                        width: HEATMAP_DIM as u32,
                        height: HEATMAP_DIM as u32,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        if let Some(scalar) = &frame.heatmap_scalar {
            if scalar.len() == GRID_CELL_COUNT {
                queue.write_buffer(&self.cpu_scalar_buffer, 0, bytemuck::cast_slice(scalar));
            }
        }

        if frame.road_quads.len() > self.road_capacity {
            self.road_capacity = frame.road_quads.len().next_power_of_two().max(1024);
            self.road_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("urban_road_instance_buf_grown"),
                size: (self.road_capacity * size_of::<GpuRoadQuadInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.road_count = frame.road_quads.len() as u32;
        if !frame.road_quads.is_empty() {
            queue.write_buffer(&self.road_buffer, 0, bytemuck::cast_slice(&frame.road_quads));
        }

        if frame.agents.len() > self.agent_capacity {
            self.agent_capacity = frame.agents.len().next_power_of_two().max(4096);
            self.agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("urban_agent_instance_buf_grown"),
                size: (self.agent_capacity * size_of::<GpuAgentInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // Rebind compute bind groups to the newly grown agent storage buffer!
            let (bg_ab, bg_ba) = Self::build_compute_bind_groups(
                device,
                &self.compute_bgl,
                &self.plume_uniform_buffer,
                &self.agent_buffer,
                &self.accum_atomic_buffer,
                &self.plume_state_a,
                &self.plume_state_b,
                &self.heatmap_view,
            );
            self.compute_bg_a_to_b = bg_ab;
            self.compute_bg_b_to_a = bg_ba;
        }
        self.agent_count = frame.agents.len() as u32;
        if !frame.agents.is_empty() {
            queue.write_buffer(&self.agent_buffer, 0, bytemuck::cast_slice(&frame.agents));
        }

        // Dispatch 3-kernel wgpu @compute semi-Lagrangian advection-diffusion pass directly on GPU!
        if let Some(plume_uni) = &frame.plume_compute {
            queue.write_buffer(
                &self.plume_uniform_buffer,
                0,
                bytemuck::bytes_of(plume_uni),
            );

            let active_bg = if self.ping_pong_flip {
                &self.compute_bg_b_to_a
            } else {
                &self.compute_bg_a_to_b
            };

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("urban_plume_advect_diffuse_cpass"),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, active_bg, &[]);

                // 1. Clear 65,536 atomic accumulation cells (`256 * 256 / 256 = 256` workgroups)
                cpass.set_pipeline(&self.cs_clear_pipeline);
                cpass.dispatch_workgroups((GRID_CELL_COUNT / 256) as u32, 1, 1);

                // 2. Splat 50,000+ agents atomically (`ceil(agent_count / 64)` workgroups)
                if self.agent_count > 0 {
                    cpass.set_pipeline(&self.cs_splat_pipeline);
                    cpass.dispatch_workgroups(self.agent_count.div_ceil(64), 1, 1);
                }

                // 3. Semi-Lagrangian wind advection + eddy diffusion + `textureStore` (`32 x 32` of 8x8)
                cpass.set_pipeline(&self.cs_advect_pipeline);
                cpass.dispatch_workgroups(
                    (HEATMAP_DIM / 8) as u32,
                    (HEATMAP_DIM / 8) as u32,
                    1,
                );
            }

            self.ping_pong_flip = !self.ping_pong_flip;
        }

        // Return capacity-warm instance vectors to GUI recycle pool after `queue.write_buffer` staging
        if let Ok(mut pool) = frame.recycle_pool.lock() {
            frame.road_quads.clear();
            frame.agents.clear();
            pool.0 = std::mem::take(&mut frame.road_quads);
            pool.1 = std::mem::take(&mut frame.agents);
        }
    }
}

pub struct UrbanViewportPaintCallback {
    pub payload: Arc<Mutex<Option<ViewportFramePayload>>>,
}

impl egui_wgpu::CallbackTrait for UrbanViewportPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = callback_resources.get_mut::<UrbanViewportGpuResources>() else {
            return Vec::new();
        };
        if let Ok(mut guard) = self.payload.lock() {
            if let Some(mut frame) = guard.take() {
                gpu.upload_and_dispatch_compute(device, queue, egui_encoder, &mut frame);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = callback_resources.get::<UrbanViewportGpuResources>() else {
            return;
        };

        let active_bg = if gpu.use_plume_scalar_buf {
            if gpu.ping_pong_flip {
                &gpu.render_bg_plume_b
            } else {
                &gpu.render_bg_plume_a
            }
        } else {
            &gpu.render_bg_cpu
        };
        render_pass.set_bind_group(0, active_bg, &[]);

        if gpu.heatmap_enabled {
            render_pass.set_pipeline(&gpu.heatmap_pipeline);
            render_pass.draw(0..6, 0..1);
        }

        if gpu.road_count > 0 {
            render_pass.set_pipeline(&gpu.road_pipeline);
            render_pass.set_vertex_buffer(0, gpu.road_buffer.slice(..));
            render_pass.draw(0..6, 0..gpu.road_count);
        }

        if gpu.agent_count > 0 {
            render_pass.set_pipeline(&gpu.agent_pipeline);
            render_pass.set_vertex_buffer(0, gpu.agent_buffer.slice(..));
            render_pass.draw(0..6, 0..gpu.agent_count);
        }
    }
}

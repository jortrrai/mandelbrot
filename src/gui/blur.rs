//! Auto blur during flights: while a flight moves the view, the image is drawn with a light Gaussian blur on the GPU, which brings a moving frame much closer to
//! the finished, supersampled image (measured in issue #118: a full-resolution frame with one sample has 40–46% less error with σ 0.8 pixel).
//!
//! The blur is a fragment shader (wgpu) that draws the texture of the image in an egui paint callback, so it costs the CPU nothing. Its strength follows what the frame
//! shows, see `blur_sigma`.

use std::sync::Arc;

use eframe::egui;

/// The most pixels a tap of the blur is from its center (the kernel is (2 × this + 1)² taps): enough for 2.5 σ up to σ 2.4
const MAX_RADIUS: i32 = 6;

/// The blur of a frame that shows previews of `block` x `block` pixels (1: full resolution) with `samples` samples per pixel: σ = 0.8 × (100 / p)^0.6 pixels, where
/// p is the resolution in percent that the frame really has (100 / block, or 100 × √samples). Measured in #118: within about 3% of the best σ at every scale.
pub(super) fn blur_sigma(block: usize, samples: u8) -> f32 {
    blur_sigma_of_percent(if block > 1 { 100.0 / block as f32 } else { 100.0 * f32::from(samples.max(1)).sqrt() })
}

/// The blur of a frame that really has `percent` of the resolution, see `blur_sigma`
pub(super) fn blur_sigma_of_percent(percent: f32) -> f32 {
    0.8 * (100.0 / percent).powf(0.6)
}

/// The part of `rect` inside `visible`, and the part of the texture coordinates `uv` (which cover `rect`) under it, or `None` if no part of `rect` is visible
pub(super) fn visible_part(rect: egui::Rect, visible: egui::Rect, uv: egui::Rect) -> Option<(egui::Rect, egui::Rect)> {
    let shown = rect.intersect(visible);
    if !shown.is_positive() {
        return None;
    }
    //The same fractions of `uv` as `shown` is of `rect`
    let at = |position: egui::Pos2| uv.min + (position - rect.min) / rect.size() * uv.size();
    Some((shown, egui::Rect::from_min_max(at(shown.min), at(shown.max))))
}

/// The shader of the blur, in WGSL: a quad over the viewport (the rectangle of the image) whose texture coordinates go from the top left of `uv` to its bottom
/// right, and a Gaussian of `sigma` image pixels (one image pixel is `step` in the texture) in gamma-space colors, like the supersampling averages them. egui's
/// textures are sRGB, so the samples are made gamma again first, and the result made linear for a framebuffer that is sRGB
const SHADER: &str = r"
struct Uniforms { uv: vec4<f32>, step: vec2<f32>, sigma: f32, radius: i32, linear_output: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u));
    var out: VertexOut;
    out.uv = vec2<f32>(mix(u.uv.x, u.uv.z, corner.x), mix(u.uv.w, u.uv.y, corner.y));
    out.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
fn gamma_from_linear(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, 12.92 * c, c < vec3<f32>(0.0031308));
}
fn linear_from_gamma(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c < vec3<f32>(0.04045));
}
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = -u.radius; y <= u.radius; y++) {
        for (var x = -u.radius; x <= u.radius; x++) {
            let weight = exp(-f32(x * x + y * y) / (2.0 * u.sigma * u.sigma));
            sum += weight * gamma_from_linear(textureSampleLevel(image, image_sampler, in.uv + vec2<f32>(f32(x), f32(y)) * u.step, 0.0).rgb);
            total += weight;
        }
    }
    let color = sum / total;
    if (u.linear_output != 0u) {
        return vec4<f32>(linear_from_gamma(color), 1.0);
    }
    return vec4<f32>(color, 1.0);
}
";

/// The uniforms of `SHADER` (48 bytes, as WGSL lays them out)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    uv: [f32; 4],
    step: [f32; 2],
    sigma: f32,
    radius: i32,
    linear_output: u32,
    padding: [u32; 3],
}

/// The pipeline of the blur and what its bind groups are made of, kept with the renderer of the window (see `Blur::new`)
struct BlurResources {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    linear_output: bool,
}

/// The bind group of the blur of the frame that is drawn (made in `prepare`, used in `paint`)
struct BlurBindGroup(wgpu::BindGroup);

impl BlurResources {
    /// The pipeline of the blur on `device`, drawing into textures of `target_format`
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> BlurResources {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blur"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("blur"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blur"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: wgpu::PipelineCompilationOptions::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: wgpu::PipelineCompilationOptions::default(), targets: &[Some(target_format.into())] }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("blur"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        BlurResources { pipeline, layout, sampler, linear_output: target_format.is_srgb() }
    }
}

/// The blur of the window: its pipeline is made once, on the wgpu device of the window (issue #140: the window draws with wgpu)
pub(super) struct Blur {
    render_state: eframe::egui_wgpu::RenderState,
}

impl Blur {
    /// The blur for the wgpu render state of the window
    pub(super) fn new(render_state: &eframe::egui_wgpu::RenderState) -> Arc<Blur> {
        let resources = BlurResources::new(&render_state.device, render_state.target_format);
        render_state.renderer.write().callback_resources.insert(resources);
        Arc::new(Blur { render_state: render_state.clone() })
    }

    /// A paint callback that draws `texture` (its part `uv`, showing an image of `width` x `height` pixels) blurred by `sigma` image pixels over the rectangle `rect`,
    /// only where it is inside `visible`, or `None` if nothing of it is.
    ///
    /// Only the visible part is drawn, with the part of the texture it shows: egui clamps the viewport of a paint callback to the screen, so a rectangle that reaches
    /// beyond it (a best-effort flight moves and scales the image past the edges of the window) would squeeze the whole image into what is left of it.
    pub(super) fn paint(self: &Arc<Blur>, rect: egui::Rect, visible: egui::Rect, texture: egui::TextureId, uv: egui::Rect, (width, height): (usize, usize), sigma: f32) -> Option<egui::PaintCallback> {
        let view = self.render_state.renderer.read().texture(&texture)?.texture.as_ref()?.create_view(&wgpu::TextureViewDescriptor::default());
        let step = [uv.width() / width as f32, uv.height() / height as f32];
        let (rect, uv) = visible_part(rect, visible, uv)?;
        let radius = ((2.5 * sigma).ceil() as i32).clamp(1, MAX_RADIUS);
        let uniforms = Uniforms { uv: [uv.min.x, uv.min.y, uv.max.x, uv.max.y], step, sigma, radius, linear_output: 0, padding: [0; 3] };
        Some(eframe::egui_wgpu::Callback::new_paint_callback(rect, BlurCallback { view, uniforms }))
    }
}

/// A blurred image to draw, see `Blur::paint`
struct BlurCallback {
    view: wgpu::TextureView,
    uniforms: Uniforms,
}

impl eframe::egui_wgpu::CallbackTrait for BlurCallback {
    fn prepare(&self, device: &wgpu::Device, _queue: &wgpu::Queue, _screen: &eframe::egui_wgpu::ScreenDescriptor, _encoder: &mut wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources) -> Vec<wgpu::CommandBuffer> {
        use wgpu::util::DeviceExt;
        let Some(blur) = resources.get::<BlurResources>() else { return Vec::new() };
        let uniforms = Uniforms { linear_output: u32::from(blur.linear_output), ..self.uniforms };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("blur"), contents: bytemuck::bytes_of(&uniforms), usage: wgpu::BufferUsages::UNIFORM });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur"),
            layout: &blur.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&blur.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: buffer.as_entire_binding() },
            ],
        });
        resources.insert(BlurBindGroup(bind_group));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, render_pass: &mut wgpu::RenderPass<'static>, resources: &eframe::egui_wgpu::CallbackResources) {
        let (Some(blur), Some(bind_group)) = (resources.get::<BlurResources>(), resources.get::<BlurBindGroup>()) else { return };
        render_pass.set_pipeline(&blur.pipeline);
        render_pass.set_bind_group(0, &bind_group.0, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

/// The colors of a `width` x `height` image blurred by a Gaussian of `sigma` pixels, like the shader of `Blur` (gamma-space colors, radius 2.5 sigma at most
/// `MAX_RADIUS`, the edge pixels repeated), on all render threads: for saving an image of the blurred quality preset
pub(super) fn blurred(colors: &[crate::coloring::TrueColor], width: usize, height: usize, sigma: f32) -> Vec<crate::coloring::TrueColor> {
    let radius: isize = ((2.5 * sigma).ceil() as isize).clamp(1, isize::try_from(MAX_RADIUS).unwrap_or(6));
    let weights: Vec<f32> = (-radius..=radius).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = weights.iter().sum();
    let weights: Vec<f32> = weights.iter().map(|weight| weight / total).collect();
    //One pass along a direction: `horizontal`, or down the columns
    let pass = |source: &[[f32; 3]], horizontal: bool| -> Vec<[f32; 3]> {
        let mut out = vec![[0.0; 3]; width * height];
        let rows_per_chunk = height.div_ceil(crate::rendering::render_thread_count()).max(1);
        std::thread::scope(|scope| {
            for (chunk, block) in out.chunks_mut(rows_per_chunk * width).enumerate() {
                let weights = &weights;
                scope.spawn(move || {
                    for (index, pixel) in block.iter_mut().enumerate() {
                        let (x, y) = (index % width, chunk * rows_per_chunk + index / width);
                        let mut sum = [0.0; 3];
                        for (offset, weight) in (-radius..=radius).zip(weights) {
                            //(The edge pixels are repeated)
                            let (sx, sy) = if horizontal { (x.saturating_add_signed(offset).min(width - 1), y) } else { (x, y.saturating_add_signed(offset).min(height - 1)) };
                            for (channel, value) in sum.iter_mut().zip(source[sy * width + sx]) {
                                *channel += value * weight;
                            }
                        }
                        *pixel = sum;
                    }
                });
            }
        });
        out
    };
    let image: Vec<[f32; 3]> = colors.iter().map(|color| [f32::from(color.red), f32::from(color.green), f32::from(color.blue)]).collect();
    pass(&pass(&image, true), false).iter().map(|[red, green, blue]| crate::coloring::TrueColor::new(red.round() as u8, green.round() as u8, blue.round() as u8)).collect()
}

/// A quality preset: samples per pixel, and whether the image is blurred (the blur preset: one sample and the blur of a full-resolution frame, σ 0.8, see #118)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Quality {
    pub(super) samples: u8,
    pub(super) blurred: bool,
}

impl Quality {
    /// The presets of the quality lists, fastest first
    /// The blurred ones blur by `blur_sigma(1, samples)`: measured in issue #186 as the best of three strengths, 4x + blur came close to 8x in 57% of its time,
    /// and 8x + blur halfway between 8x and 16x in about half the time of 16x
    pub(super) const PRESETS: [Quality; 11] = [
        Quality { samples: 1, blurred: false }, Quality { samples: 1, blurred: true }, Quality { samples: 2, blurred: false }, Quality { samples: 4, blurred: false },
        Quality { samples: 4, blurred: true }, Quality { samples: 8, blurred: false }, Quality { samples: 8, blurred: true }, Quality { samples: 16, blurred: false },
        Quality { samples: 16, blurred: true }, Quality { samples: 32, blurred: false }, Quality { samples: 64, blurred: false },
    ];

    /// The name in the list
    pub(super) fn name(self) -> String {
        match (self.samples, self.blurred) {
            (1, false) => String::from("1x (fastest)"),
            (1, true) => String::from("1x + blur (smooth, fast)"),
            (4, true) => String::from("4x + blur (close to 8x)"),
            (8, true) => String::from("8x + blur (between 8x and 16x)"),
            (_, true) => format!("{}x + blur", self.samples),
            (64, false) => String::from("64x (best)"),
            (samples, false) => format!("{samples}x"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blur_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("the blur shader parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).expect("the blur shader is valid");
    }

    #[test]
    fn the_uniforms_are_laid_out_as_wgsl_lays_them_out() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 48);
    }

    #[test]
    fn the_blur_shader_draws_what_the_blur_of_saved_images_computes() {
        //On a GPU (skipped without one): a 32 x 16 image in an sRGB texture, as egui keeps images, blurred into a texture by the pipeline of the window,
        //against `blurred`, which saved images and videos use
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("no GPU: the test of the blur shader is skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let (width, height, sigma) = (32_usize, 16_usize, 1.2_f32);
        let colors: Vec<crate::coloring::TrueColor> = (0..width * height).map(|index| {
            let (x, y) = (index % width, index / width);
            crate::coloring::TrueColor::new(if (x / 3 + y / 2) % 2 == 0 { 230 } else { 20 }, (x * 8) as u8, (y * 16) as u8)
        }).collect();
        let size = wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 };
        let image = device.create_texture(&wgpu::TextureDescriptor { label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba8UnormSrgb, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[] });
        let rgba: Vec<u8> = colors.iter().flat_map(|color| [color.red, color.green, color.blue, 255]).collect();
        queue.write_texture(image.as_image_copy(), &rgba, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width as u32 * 4), rows_per_image: None }, size);
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let target = device.create_texture(&wgpu::TextureDescriptor { label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[] });
        let callback = BlurCallback { view: image.create_view(&wgpu::TextureViewDescriptor::default()), uniforms: Uniforms { uv: [0.0, 0.0, 1.0, 1.0], step: [1.0 / width as f32, 1.0 / height as f32], sigma, radius: ((2.5 * sigma).ceil() as i32).clamp(1, MAX_RADIUS), linear_output: 0, padding: [0; 3] } };
        let mut resources = eframe::egui_wgpu::CallbackResources::default();
        resources.insert(BlurResources::new(&device, format));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let screen = eframe::egui_wgpu::ScreenDescriptor { size_in_pixels: [width as u32, height as u32], pixels_per_point: 1.0 };
        eframe::egui_wgpu::CallbackTrait::prepare(&callback, &device, &queue, &screen, &mut encoder, &mut resources);
        {
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let attachment = wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[Some(attachment)], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None }).forget_lifetime();
            let whole = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width as f32, height as f32));
            let info = egui::PaintCallbackInfo { viewport: whole, clip_rect: whole, pixels_per_point: 1.0, screen_size_px: [width as u32, height as u32] };
            eframe::egui_wgpu::CallbackTrait::paint(&callback, info, &mut pass, &resources);
        }
        let row = (width as u32 * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(row) * height as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        encoder.copy_texture_to_buffer(target.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } }, size);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let expected = blurred(&colors, width, height, sigma);
        let mut largest = 0;
        for y in 3..height - 3 {
            for x in 3..width - 3 {
                let (pixel, want) = (&mapped[y * row as usize + x * 4..][..3], &expected[y * width + x]);
                largest = largest.max(pixel[0].abs_diff(want.red)).max(pixel[1].abs_diff(want.green)).max(pixel[2].abs_diff(want.blue));
            }
        }
        assert!(largest <= 3, "the GPU's blur is the CPU's within rounding (away from the clamped edges): at most {largest} apart");
    }
}

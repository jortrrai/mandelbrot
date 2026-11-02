//! Auto blur during flights: while a flight moves the view, the image is drawn with a light Gaussian blur on the GPU, which brings a moving frame much closer to
//! the finished, supersampled image (measured in issue #118: a full-resolution frame with one sample has 40–46% less error with σ 0.8 pixel).
//!
//! The blur is a fragment shader that draws the texture of the image in an egui paint callback, so it costs the CPU nothing. Its strength follows what the frame
//! shows, see `blur_sigma`.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_glow;
use eframe::glow::{self, HasContext};

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

/// The shader program of the blur, made once from the OpenGL context of the window
pub(super) struct Blur {
    program: glow::Program,
    vertex_array: glow::VertexArray,
}

impl Blur {
    /// The blur for the OpenGL context `gl`, or `None` where its shaders are not available (OpenGL before 3.1, or OpenGL ES 2): the image is then drawn as it is
    pub(super) fn new(gl: &glow::Context) -> Option<Arc<Blur>> {
        let version = egui_glow::ShaderVersion::get(gl);
        if !version.is_new_shader_interface() {
            return None;
        }
        let header = if matches!(version, egui_glow::ShaderVersion::Es300) { format!("{}precision highp float;\n", version.version_declaration()) } else { version.version_declaration().to_owned() };
        //A quad over the whole viewport (the rectangle of the image), from the vertex number; the texture coordinates go from the top left of `u_uv` to its bottom right
        let vertex = format!("{header}uniform vec4 u_uv;\nout vec2 v_uv;\nvoid main() {{\n    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));\n    v_uv = vec2(mix(u_uv.x, u_uv.z, corner.x), mix(u_uv.w, u_uv.y, corner.y));\n    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);\n}}\n");
        //A Gaussian of `u_sigma` image pixels (one image pixel is `u_step` in the texture), in the gamma-space colors of the texture, like the supersampling averages them
        let fragment = format!("{header}uniform sampler2D u_sampler;\nuniform vec2 u_step;\nuniform float u_sigma;\nuniform int u_radius;\nin vec2 v_uv;\nout vec4 f_color;\nvoid main() {{\n    vec3 sum = vec3(0.0);\n    float total = 0.0;\n    for (int y = -u_radius; y <= u_radius; y++) {{\n        for (int x = -u_radius; x <= u_radius; x++) {{\n            float weight = exp(-float(x * x + y * y) / (2.0 * u_sigma * u_sigma));\n            sum += weight * texture(u_sampler, v_uv + vec2(float(x), float(y)) * u_step).rgb;\n            total += weight;\n        }}\n    }}\n    f_color = vec4(sum / total, 1.0);\n}}\n");
        //SAFETY: plain OpenGL calls on the context of the window, on the thread that owns it
        unsafe {
            let program = gl.create_program().ok()?;
            let mut shaders = Vec::new();
            for (kind, source) in [(glow::VERTEX_SHADER, &vertex), (glow::FRAGMENT_SHADER, &fragment)] {
                let shader = gl.create_shader(kind).ok()?;
                gl.shader_source(shader, source);
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    eprintln!("The blur shader did not compile: {}", gl.get_shader_info_log(shader));
                    return None;
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }
            gl.link_program(program);
            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            if !gl.get_program_link_status(program) {
                eprintln!("The blur shader did not link: {}", gl.get_program_info_log(program));
                return None;
            }
            let vertex_array = gl.create_vertex_array().ok()?;
            Some(Arc::new(Blur { program, vertex_array }))
        }
    }

    /// A paint callback that draws `texture` (its part `uv`, showing an image of `width` x `height` pixels) blurred by `sigma` image pixels over the rectangle `rect`,
    /// only where it is inside `visible`, or `None` if nothing of it is.
    ///
    /// Only the visible part is drawn, with the part of the texture it shows: egui clamps the viewport of a paint callback to the screen, so a rectangle that reaches
    /// beyond it (a best-effort flight moves and scales the image past the edges of the window) would squeeze the whole image into what is left of it.
    pub(super) fn paint(self: &Arc<Blur>, rect: egui::Rect, visible: egui::Rect, texture: egui::TextureId, uv: egui::Rect, (width, height): (usize, usize), sigma: f32) -> Option<egui::PaintCallback> {
        let blur = Arc::clone(self);
        let step = [uv.width() / width as f32, uv.height() / height as f32];
        let (rect, uv) = visible_part(rect, visible, uv)?;
        let radius = ((2.5 * sigma).ceil() as i32).clamp(1, MAX_RADIUS);
        let callback = egui_glow::CallbackFn::new(move |_info, painter| {
            let Some(texture) = painter.texture(texture) else { return };
            let gl = painter.gl();
            //SAFETY: plain OpenGL calls on the context of the window, within egui's paint callback (egui restores its own state afterwards)
            unsafe {
                gl.use_program(Some(blur.program));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                gl.uniform_1_i32(gl.get_uniform_location(blur.program, "u_sampler").as_ref(), 0);
                gl.uniform_4_f32(gl.get_uniform_location(blur.program, "u_uv").as_ref(), uv.min.x, uv.min.y, uv.max.x, uv.max.y);
                gl.uniform_2_f32(gl.get_uniform_location(blur.program, "u_step").as_ref(), step[0], step[1]);
                gl.uniform_1_f32(gl.get_uniform_location(blur.program, "u_sigma").as_ref(), sigma);
                gl.uniform_1_i32(gl.get_uniform_location(blur.program, "u_radius").as_ref(), radius);
                gl.bind_vertex_array(Some(blur.vertex_array));
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
        });
        Some(egui::PaintCallback { rect, callback: Arc::new(callback) })
    }
}

/// How many pixels a tap of the blur of `sigma` pixels reaches from its center (2.5 sigma, from 1 to `MAX_RADIUS`): the rows of a strip that `blurred` needs
/// from the strips next to it (see `export`)
pub(super) fn blur_radius(sigma: f32) -> usize {
    ((2.5 * sigma).ceil() as usize).clamp(1, MAX_RADIUS.unsigned_abs() as usize)
}

/// The colors of a `width` x `height` image blurred by a Gaussian of `sigma` pixels, like the shader of `Blur` (gamma-space colors, radius 2.5 sigma at most
/// `MAX_RADIUS`, the edge pixels repeated), on all render threads: for saving an image of the blurred quality preset
pub(super) fn blurred(colors: &[crate::coloring::TrueColor], width: usize, height: usize, sigma: f32) -> Vec<crate::coloring::TrueColor> {
    let radius = blur_radius(sigma).cast_signed();
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

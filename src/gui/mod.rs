//! The graphical user interface: the fractal on the left, the controls in the top right and a scrollable status log in the bottom right.
//! The terminal is only used for developer output.
//!
//! * `render_job`: the background threads that render the view layer by layer.
//! * `actions`: what the buttons and clicks do.
//! * `controls`: the controls on the right.
//! * `fit`: the render size that follows the size of the window.
//! * `history`: the views that were visited, for Back and Forward.
//! * `pointer`: the mouse on the image: clicks, the wheel and dragging.
//! * `points`: the points of interest with their thumbnails.
//! * `save_size`: the size of the saved image: the presets and the custom size.
//! * `status`: the current view, the status log and the progress bar.

mod actions;
mod blur;
mod controls;
mod fit;
mod flight;
mod flight_job;
mod history;
mod pointer;
mod points;
mod render_job;
mod save_size;
mod status;
mod transform;
mod video;
#[cfg(test)]
mod tests;

pub use status::{format_zoom, CurrentView};

use self::fit::Fit;
use self::history::ViewHistory;
use self::points::{ThumbnailJob, ThumbnailMessage};
use self::render_job::{JobMessage, Progress, RenderBars};
use self::save_size::SaveSize;

use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};

use crate::auto_iterations::{AutoIterationSettings, AutoPreset};
use crate::coloring::{Channel, ColorChannelMapping, TrueColor};
use crate::complex_plane::ComplexPlane;
use crate::mandelbrot_set::MandelbrotSet;
use crate::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
use crate::points_of_interest::{self, PointOfInterest};
use crate::rendering::{self};
use crate::resolution::{self, Resolution};
use crate::thumbnails::{self, Lru};
use crate::{ColoringFunction, Config, InteractionVariables, COLORING_FUNCTION, COLOR_CHANNEL_MAPPING};

/// Width in points of the right side panel, used to size the initial window
pub const SIDE_PANEL_WIDTH: f32 = 380.0;
/// Minimum height in points of the window, so that the controls and the status log have room
pub const MIN_WINDOW_HEIGHT: f32 = 700.0;

const COLORING_FUNCTIONS: [(&str, ColoringFunction); 2] = [("HSV", TrueColor::new_from_hsv_colors), ("Bernstein polynomials", TrueColor::new_from_bernstein_polynomials)];

const COLOR_CHANNEL_MAPPINGS: [&str; 27] = [
    "RGB", "RBG", "GRB", "GBR", "BRG", "BGR", "RRR", "GGG", "BBB", "RRG", "RRB", "RGG", "RGR", "RBB", "RBR", "GGB", "GGR", "GBB", "GBG", "GRG", "GRR", "BBG", "BBR", "BGB", "BGG", "BRB", "BRR",
];

/// Converts colors to RGB bytes for a texture, applying the color channel mapping while converting; the same colors as `PixelBuffer::colors_to_pixels`
fn colors_to_rgb_bytes(colors: &[TrueColor], mapping: ColorChannelMapping) -> Vec<u8> {
    let [red, green, blue] = mapping.sources().map(Channel::index);
    let mut rgb: Vec<u8> = Vec::with_capacity(colors.len() * 3);
    for color in colors {
        let channels = [color.red, color.green, color.blue];
        rgb.extend_from_slice(&[channels[red], channels[green], channels[blue]]);
    }
    rgb
}

/// The part of a texture that covers the whole image
/// The warning in the status log when zooming in stops at the end of the fast GPU math, see `MandelbrotApp::zoom_refused`
const ZOOM_LIMIT_TEXT: &str = "Zoom limit: deeper than a scale of about 10⁻⁹ the fast GPU math switches back to exact (slower) math. Turn off \"Limit the zoom to fast GPU math\" to zoom deeper.";

const FULL_TEXTURE: egui::Rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));

/// The texture of a coarse preview with blocks of `block` x `block` pixels: one texel per block, with the color of its top-left pixel (the pixel whose sample the block shows),
/// and the part of the texture to draw over the whole image. That part puts the center of every texel on the center of its pixel, so a smoothly scaled preview is not shifted.
fn coarse_texture(colors: &[TrueColor], mapping: ColorChannelMapping, width: usize, height: usize, block: usize) -> (ColorImage, egui::Rect) {
    let (columns, rows) = (width.div_ceil(block), height.div_ceil(block));
    let samples: Vec<TrueColor> = (0..rows).flat_map(|row| (0..columns).map(move |column| colors[row * block * width + column * block])).collect();
    let image = ColorImage::from_rgb([columns, rows], &colors_to_rgb_bytes(&samples, mapping));
    //Screen pixel x shows texture coordinate (x / block + offset) / columns; the offset maps the center of pixel `i * block` to the center of texel `i`
    let offset = 0.5 - 0.5 / block as f32;
    let uv = |pixels: usize, texels: usize| (offset / texels as f32, (pixels as f32 / block as f32 + offset) / texels as f32);
    let ((left, right), (top, bottom)) = (uv(width, columns), uv(height, rows));
    (image, egui::Rect::from_min_max(egui::pos2(left, top), egui::pos2(right, bottom)))
}

/// The state of the window: the view that is shown, the settings of the render, the render job that is running, the points of interest with their thumbnails and the status log
#[allow(clippy::struct_excessive_bools)] // The flags are independent switches of the window
pub struct MandelbrotApp {
    c: ComplexPlane,
    p: PixelBuffer,
    m: MandelbrotSet,
    vars: InteractionVariables,
    config: Config,
    coloring_function: ColoringFunction,
    coloring_index: usize,
    supersampling_amount: u8,
    image_supersampling_amount: u8,
    /// Whether the wheel is zooming or the mouse is dragging the view, see `pointer`: the status log then does not get a line for every render
    gesture_active: bool,
    /// The fractions of pixels (columns, rows) of a drag that were not used yet
    drag_remainder: (f32, f32),
    /// The views that were visited, see `history`
    history: ViewHistory,
    /// Whether the render size follows the size of the window, see `fit`
    fit: Fit,
    /// The size of the saved image, see `save_size`
    save_size: SaveSize,
    /// Edited values of the max iterations field, applied when editing ends
    max_iterations_input: u32,
    /// Whether the maximum iterations are chosen automatically for the zoom of every render, see `auto_iterations`
    auto_iterations: bool,
    auto_settings: AutoIterationSettings,
    auto_preset: AutoPreset,
    /// Whether the displayed auto limit is only the starting limit of the running render, because the probes that refine it have not finished
    limit_pending: bool,
    /// Whether supersampling puts the extra samples on boundary and detail pixels only, see `rendering::Render::adaptive`
    adaptive_supersampling: bool,
    /// Edited value of the supersampling field, applied when editing ends
    supersampling_input: u8,
    /// The points of interest of the folder, see `points_of_interest`, and the folder they were loaded from
    points: Vec<PointOfInterest>,
    points_folder: std::path::PathBuf,
    /// The thumbnails of the points of interest that were shown lately, at most `thumbnails::MAX_CACHED`, and the one that is being rendered, see `thumbnails`
    thumbs: Lru<TextureHandle>,
    thumb_job: Option<ThumbnailJob>,
    thumb_results: Receiver<ThumbnailMessage>,
    thumb_sender: Sender<ThumbnailMessage>,
    thumb_job_counter: u64,
    /// How many thumbnails were started in total, and the keys of the thumbnails that were visible in the last frame
    thumbnails_started: u64,
    thumbs_visible: Vec<u64>,
    /// Whether going to a point of interest also restores the appearance saved in its file (the coloring and the color channels); off by default
    restore_appearance: bool,
    /// Whether the saved points of interest are marked on the view at their locations, with their names (off by default)
    show_points: bool,
    /// The name and the description being typed to save the current view as a point of interest, while that form is open
    new_point: Option<(String, String)>,
    /// How fast a flight to a point of interest goes (fast by default), and the flight that is under way, see `flight`
    pace: flight::Pace,
    /// The last flight, for a video of it, the video under way, and the frame rate of videos, see `video`
    last_flight: Option<video::VideoFlight>,
    video_job: Option<video::VideoJob>,
    video_fps: u32,
    /// The speed of a video in percent of the flight's, its file format, and whether its frames are also written as PNG files, see `video`
    video_speed: u16,
    video_format: video::VideoFormat,
    video_pngs: bool,
    /// Where videos are written (`saved`)
    video_folder: std::path::PathBuf,
    flight: Option<points::PointFlight>,
    /// Whether the running job renders the frames of the flight (see `flight_job`), and the time between two frames of the screen, which the job keeps to
    flight_job: bool,
    frame_interval: Duration,
    /// How complete the frames of a flight have to be (best effort by default), see `flight::FlightRendering`
    flight_rendering: flight::FlightRendering,
    /// Immersion: the screen never shows black where nothing is rendered yet; the view only moves once the new view has at least a complete low-resolution frame
    immersion: bool,
    /// Whether the render of the current view has shown a frame that is complete enough (see `required_readiness`), and whether it has shown any frame
    frame_ready: bool,
    frame_accepted: bool,
    /// Edited text of the "go to view" fields
    view_inputs: [String; 3],
    /// Messages shown to the user in the status box
    log: Vec<String>,
    texture: Option<TextureHandle>,
    texture_outdated: bool,
    /// The blur of the image during flights (`None` without a suitable OpenGL context, as in tests), see `blur`
    blur: Option<std::sync::Arc<blur::Blur>>,
    /// Whether the image is blurred while a flight moves the view, see `blur`
    auto_blur: bool,
    /// The samples per pixel of the frame on the screen (0 for a preview), for the strength of the blur
    shown_samples: u8,
    /// The part of the texture that covers the image: all of it, except for a coarse preview, see `coarse_texture`
    texture_uv: egui::Rect,
    /// The block size of the preview that the colors of `p` show, 1 for a full layer: a coarse preview is uploaded small and scaled up smoothly
    preview_block: usize,
    /// The block sizes of the previews of the running render (see `render_job::preview_levels`), and how many it has shown, to know the block size of the next one
    preview_levels: Vec<usize>,
    previews_received: usize,
    /// The view that the colors of `p` show. It differs from `c` while the render of a new view has not shown its first layer yet: the image is then drawn moved and scaled, see `transform`
    shown_plane: ComplexPlane,
    /// The receiving end of the running background rendering job, if any. While a job runs, the controls are disabled.
    job: Option<Receiver<JobMessage>>,
    /// The colors of frames that the screen no longer shows, which the jobs fill with later frames: a big image is then not allocated (and its memory cleared by
    /// the system) for every frame, see `render_job::SparePool`
    spare_colors: render_job::SparePool,
    /// Set while a progressive render runs
    progress: Option<Progress>,
    /// The progress bar of the last progressive render
    bars: Option<RenderBars>,
    /// The time per pixel of the first layer of the last render, to predict whether the next render needs previews, see `render_job`
    first_layer_per_pixel: Option<Duration>,
    /// Whether the last render showed a full layer (not only a preview) of the view `shown_plane`
    layer_shown: bool,
    /// The message and the time of the last render during a gesture, until the full-quality render after the gesture starts, see `render_job`
    settle: Option<(String, std::time::Instant, bool)>,
    /// Where the samples of the last render arrive when its job stops, to continue them, see `render_job`
    samples: Option<std::sync::mpsc::Receiver<crate::rendering::SampleState>>,
    /// Whether every pixel of the buffer has been rendered with the full supersampling amount, so a move can reuse it
    buffer_is_complete: bool,
    /// Messages from background tasks that do not change the view, e.g. saving an image
    notices: Receiver<String>,
    notice_sender: Sender<String>,
    /// The GPU that helps with rendering once it is opened and checked (see `rendering::gpu`), whether to use it, and where the result of opening it arrives
    gpu: Option<std::sync::Arc<rendering::gpu::GpuIterator>>,
    use_gpu: bool,
    /// Whether the GPU uses its fast double-float loop (on by default): faster, not bit for bit the CPU's, see `rendering::Render::fast_gpu`. Saved images are always exact
    fast_gpu: bool,
    /// Whether nothing deeper than the fast GPU math is rendered (off by default), see `zoom_refused` and `keep_within_zoom_limit`
    limit_zoom: bool,
    gpu_opening: Option<Receiver<Result<rendering::gpu::GpuIterator, String>>>,
}

impl MandelbrotApp {
    #[allow(clippy::too_many_lines)] // The initial value of every field
    pub(super) fn new(config: &Config) -> MandelbrotApp {
        let c = ComplexPlane::new(config.window_width, config.window_height);
        let shown_plane = c.clone();
        let mut p = PixelBuffer::new(PixelPlane::new(config.window_width, config.window_height));
        p.color_channel_mapping = COLOR_CHANNEL_MAPPING;
        let m = MandelbrotSet::new(config.max_iterations, config.orbit_radius);
        //The tests never read or write the presets file of the user
        let presets_path = if cfg!(test) { None } else { resolution::custom_presets_path() };
        let (save_size, preset_warnings) = SaveSize::new(Resolution::new(config.image_width, config.image_height), presets_path);
        let (notice_sender, notices) = mpsc::channel();
        let (thumb_sender, thumb_results) = mpsc::channel();
        let mut app = MandelbrotApp {
            c,
            p,
            max_iterations_input: m.max_iterations,
            auto_iterations: true,
            auto_settings: AutoPreset::Balanced.settings(),
            auto_preset: AutoPreset::Balanced,
            limit_pending: false,
            adaptive_supersampling: true,
            m,
            vars: InteractionVariables::default(),
            config: config.clone(),
            coloring_function: COLORING_FUNCTION,
            coloring_index: 1, //Bernstein polynomials, see COLORING_FUNCTIONS
            supersampling_amount: config.supersampling_amount,
            supersampling_input: config.supersampling_amount,
            image_supersampling_amount: config.supersampling_amount,
            gesture_active: false,
            drag_remainder: (0.0, 0.0),
            history: ViewHistory::default(),
            fit: Fit::new(config.fit_window),
            save_size,
            points: Vec::new(),
            thumbs: Lru::new(thumbnails::MAX_CACHED),
            thumb_job: None,
            thumb_results,
            thumb_sender,
            thumb_job_counter: 0,
            thumbnails_started: 0,
            thumbs_visible: Vec::new(),
            points_folder: std::path::PathBuf::new(),
            restore_appearance: false,
            show_points: false,
            new_point: None,
            pace: flight::Pace::Fast,
            last_flight: None,
            video_job: None,
            video_fps: 30,
            video_speed: 100,
            video_format: video::VideoFormat::HighQuality,
            video_pngs: true,
            video_folder: std::path::PathBuf::from("saved"),
            flight: None,
            flight_job: false,
            frame_interval: Duration::from_secs_f64(1.0 / 60.0),
            flight_rendering: flight::FlightRendering::BestEffort,
            immersion: false,
            frame_ready: false,
            frame_accepted: false,
            view_inputs: [String::new(), String::new(), String::new()],
            log: Vec::new(),
            texture: None,
            texture_outdated: true,
            blur: None,
            auto_blur: true,
            shown_samples: 0,
            texture_uv: FULL_TEXTURE,
            preview_block: 1,
            preview_levels: Vec::new(),
            previews_received: 0,
            shown_plane,
            job: None,
            spare_colors: render_job::SparePool::default(),
            progress: None,
            bars: None,
            first_layer_per_pixel: None,
            layer_shown: false,
            settle: None,
            samples: None,
            buffer_is_complete: false,
            notices,
            gpu: None,
            use_gpu: true,
            fast_gpu: true,
            limit_zoom: false,
            gpu_opening: config.gpu.then(|| {
                //Opening the GPU and checking it against the CPU takes a moment: in the background, the first renders use the CPU
                let (sender, receiver) = mpsc::channel();
                std::thread::spawn(move || {
                    let _ = sender.send(rendering::gpu::GpuIterator::new());
                });
                receiver
            }),
            notice_sender,
        };
        app.log(format!("Window: {}x{} pixels, {} CPU threads used for rendering", config.window_width, config.window_height, rendering::render_thread_count()));
        app.log(format!("Max. iterations: {}, orbit radius: {}, supersampling: {}x", config.max_iterations, config.orbit_radius, config.supersampling_amount));
        for warning in preset_warnings {
            app.log(warning);
        }
        app.load_points_of_interest(points_of_interest::find_folder());
        app.render("Rendered the default view");
        app
    }

    /// Adds a message to the status box, and prints it to the terminal for developers
    /// Whether zooming in to `zoomed` is refused because the zoom is limited to the views of the fast GPU math (`limit_zoom`, with the fast GPU math on, and a GPU
    /// that is open or still opening); then the warning is logged, once for a row of refused zooms
    pub(super) fn zoom_refused(&mut self, zoomed: &ComplexPlane) -> bool {
        let limited = self.limit_zoom && self.fast_gpu && self.use_gpu && (self.gpu.is_some() || self.gpu_opening.is_some());
        if !limited || zoomed.precision_steps() >= rendering::FAST_GPU_STEPS {
            return false;
        }
        if self.log.last().map(String::as_str) != Some(ZOOM_LIMIT_TEXT) {
            self.log(ZOOM_LIMIT_TEXT);
        }
        true
    }

    /// Keeps the view within the zoom limit (see `zoom_refused`) whatever led past it (a larger window, the limit or the fast GPU math switched on at a deep view):
    /// zooms out around the center to the limit, with the warning. Every render starts with this, so nothing deeper is ever rendered while the limit is on
    pub(super) fn keep_within_zoom_limit(&mut self) {
        for _ in 0..64 {
            if !self.zoom_refused(&self.c.clone()) {
                return;
            }
            let factor = rendering::FAST_GPU_STEPS / self.c.precision_steps().max(f64::MIN_POSITIVE);
            self.c.scale(factor.clamp(1.0, 1e6) * 1.001);
        }
    }

    pub(super) fn log(&mut self, message: impl Into<String>) {
        let message = message.into();
        println!("{message}");
        self.log.push(message);
    }

    pub(super) fn view_description(&self) -> String {
        let center = self.c.center();
        format!("center = ({}, {}), scale = {}", center.x, center.y, self.c.get_scale())
    }

    pub(super) fn is_busy(&self) -> bool {
        self.job.is_some()
    }

    /// Whether the controls can be used: not while a move runs, but always while a progressive render runs, since using a control interrupts it
    pub(super) fn controls_enabled(&self) -> bool {
        self.job.is_none() || self.progress.is_some()
    }

    /// Prepares what needs the OpenGL context of the window: the blur of flights (see `blur`)
    pub fn use_gl(&mut self, gl: Option<&eframe::glow::Context>) {
        self.blur = gl.and_then(blur::Blur::new);
    }

    pub(super) fn update_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_outdated && self.texture.is_some() {
            return;
        }
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let (image, uv, options) = if self.preview_block > 1 {
            let (image, uv) = coarse_texture(&self.p.colors, self.p.color_channel_mapping, width, height, self.preview_block);
            (image, uv, TextureOptions::LINEAR)
        } else {
            (ColorImage::from_rgb([width, height], &colors_to_rgb_bytes(&self.p.colors, self.p.color_channel_mapping)), FULL_TEXTURE, TextureOptions::NEAREST)
        };
        self.texture_uv = uv;
        match &mut self.texture {
            Some(texture) => texture.set(image, options),
            None => self.texture = Some(ctx.load_texture("mandelbrot", image, options)),
        }
        self.texture_outdated = false;
    }
}

impl MandelbrotApp {
    /// Draws the image as large as fits in the area of `ui` (keeping its aspect ratio) and handles the mouse on it. Returns the size of the area in points.
    pub(super) fn image_ui(&mut self, ui: &mut egui::Ui) -> egui::Vec2 {
        let available = ui.available_size();
        let Some(texture) = self.texture.clone() else { return available };
        let (width, height) = (self.p.pixel_plane.width as f32, self.p.pixel_plane.height as f32);
        let scale = (available.x / width).min(available.y / height);
        let size = egui::vec2(width * scale, height * scale);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        //The image of the last render, at its place in the current view (the whole rectangle when they are the same view, or when the view is held, see `drawn_fraction`)
        let target = transform::rect_of(self.drawn_fraction(), rect);
        match self.blur.as_ref().filter(|_| self.auto_blur && self.flight.is_some()) {
            Some(blur) => {
                let sigma = blur::blur_sigma(self.preview_block, self.shown_samples);
                if let Some(callback) = blur.paint(target, rect, texture.id(), self.texture_uv, (self.p.pixel_plane.width, self.p.pixel_plane.height), sigma) {
                    ui.painter_at(rect).add(callback);
                }
            }
            None => {
                ui.painter_at(rect).image(texture.id(), target, self.texture_uv, Color32::WHITE);
            }
        }
        if self.show_points {
            points::draw_markers(&ui.painter_at(rect), &points::marker_positions(&self.points, self.displayed_plane(), rect));
        }
        self.handle_image_pointer(ui, &response, size);
        available
    }
}

/// How complete a frame of a render is, or has to be before the screen shows it, from the least to the most complete
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Readiness {
    /// Anything (no frame needs to wait)
    Any,
    /// A complete frame of blocks of at most this many pixels across: a preview (of the whole view, or of the pixels a move exposed next to the ones it kept), or a full layer
    Blocks(usize),
    /// At least this many supersampling layers (1: a full-resolution frame)
    Layers(u8),
    /// The finished frame: every supersampling layer
    SuperSampled,
}

impl Readiness {
    /// Any complete frame, however coarse its blocks
    pub(super) const COMPLETE: Readiness = Readiness::Blocks(usize::MAX);
    /// A full layer
    pub(super) const FULL_RESOLUTION: Readiness = Readiness::Layers(1);

    /// The order of completeness: finer blocks, then more layers
    const fn rank(self) -> (u8, usize) {
        match self {
            Readiness::Any => (0, 0),
            Readiness::Blocks(block) => (1, usize::MAX - block),
            Readiness::Layers(layers) => (2, layers as usize),
            Readiness::SuperSampled => (3, 0),
        }
    }

    /// What a flight with `rendering` waits for, see `flight::FlightRendering`
    fn of_flight(rendering: flight::FlightRendering) -> Readiness {
        match rendering {
            flight::FlightRendering::BestEffort => Readiness::Any,
            flight::FlightRendering::LowRes => Readiness::COMPLETE,
            flight::FlightRendering::Resolution(percent) if percent < 100 => match (100.0 / f64::from(percent.max(1))).round() as usize {
                0 | 1 => Readiness::FULL_RESOLUTION,
                block => Readiness::Blocks(block),
            },
            flight::FlightRendering::Resolution(percent) => Readiness::Layers((f64::from(percent) / 100.0).powi(2).round().clamp(1.0, 64.0) as u8),
            flight::FlightRendering::SuperSampled => Readiness::SuperSampled,
        }
    }
}

impl Ord for Readiness {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl PartialOrd for Readiness {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl MandelbrotApp {
    /// How complete a frame has to be before the screen shows it: a complete frame with immersion, and what the flight rendering asks for during a flight
    pub(super) fn required_readiness(&self) -> Readiness {
        let flight = if self.flight.is_some() { self.flight_readiness() } else { Readiness::Any };
        flight.max(if self.immersion { Readiness::COMPLETE } else { Readiness::Any })
    }

    /// What the frames of a flight have to be, see `flight::FlightRendering`; immersion asks for at least a complete frame
    pub(super) fn flight_readiness(&self) -> Readiness {
        let mode = Readiness::of_flight(self.flight_rendering);
        mode.max(if self.immersion { Readiness::COMPLETE } else { Readiness::Any })
    }

    /// Where the image is drawn, as a fraction of the image area: moved and scaled to the current view (best effort), or held where it is, so the screen never shows
    /// black where nothing is rendered: the view on the screen then is the view of the last frame that was complete enough, and it moves when the next one is
    pub(super) fn drawn_fraction(&self) -> egui::Rect {
        if self.required_readiness() == Readiness::Any { transform::shown_fraction(&self.shown_plane, &self.c) } else { FULL_TEXTURE }
    }

    /// The view that the screen shows: the current view, or with a held view (see `drawn_fraction`) the view of the frame on the screen
    pub(super) fn displayed_plane(&self) -> &ComplexPlane {
        if self.required_readiness() == Readiness::Any { &self.c } else { &self.shown_plane }
    }
}

impl eframe::App for MandelbrotApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job();
        self.poll_thumbnails(&ctx);
        self.poll_video(&ctx);
        if self.is_busy() {
            ctx.request_repaint_after(Duration::from_millis(100)); //Keep polling for the finished job
        }
        self.update_texture(&ctx);

        egui::Panel::right("side_panel").default_size(SIDE_PANEL_WIDTH).min_size(280.0).show(ui, |ui| self.side_panel_ui(ui));

        let mut fit_area = egui::Vec2::ZERO;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::BLACK)).show(ui, |ui| fit_area = self.image_ui(ui));
        self.fly(&ctx);
        self.settle_gesture(&ctx);
        self.fit_to_window(&ctx, fit_area);
    }
}

//! The graphical user interface: the fractal on the left, the controls in the top right and a scrollable status log in the bottom right.
//! The terminal is only used for developer output.
//!
//! * `render_job`: the background threads that render the view layer by layer.
//! * `actions`: what the buttons and clicks do.
//! * `controls`: the controls on the right.
//! * `fit`: the render size that follows the size of the window.
//! * `points`: the points of interest with their thumbnails.
//! * `save_size`: the size of the saved image: the presets and the custom size.
//! * `status`: the current view, the status log and the progress bar.

mod actions;
mod controls;
mod fit;
mod points;
mod render_job;
mod save_size;
mod status;
#[cfg(test)]
mod tests;

pub use status::{format_zoom, CurrentView};

use self::fit::Fit;
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
    /// Edited text of the "go to view" fields
    view_inputs: [String; 3],
    /// Messages shown to the user in the status box
    log: Vec<String>,
    texture: Option<TextureHandle>,
    texture_outdated: bool,
    /// The receiving end of the running background rendering job, if any. While a job runs, the controls are disabled.
    job: Option<Receiver<JobMessage>>,
    /// Set while a progressive render runs
    progress: Option<Progress>,
    /// The progress bar of the last progressive render
    bars: Option<RenderBars>,
    /// Whether every pixel of the buffer has been rendered with the full supersampling amount, so a move can reuse it
    buffer_is_complete: bool,
    /// Messages from background tasks that do not change the view, e.g. saving an image
    notices: Receiver<String>,
    notice_sender: Sender<String>,
}

impl MandelbrotApp {
    pub(super) fn new(config: &Config) -> MandelbrotApp {
        let c = ComplexPlane::new(config.window_width, config.window_height);
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
            auto_iterations: false,
            auto_settings: AutoPreset::Balanced.settings(),
            auto_preset: AutoPreset::Balanced,
            limit_pending: false,
            adaptive_supersampling: false,
            m,
            vars: InteractionVariables::default(),
            config: config.clone(),
            coloring_function: COLORING_FUNCTION,
            coloring_index: 1, //Bernstein polynomials, see COLORING_FUNCTIONS
            supersampling_amount: config.supersampling_amount,
            supersampling_input: config.supersampling_amount,
            image_supersampling_amount: config.supersampling_amount,
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
            view_inputs: [String::new(), String::new(), String::new()],
            log: Vec::new(),
            texture: None,
            texture_outdated: true,
            job: None,
            progress: None,
            bars: None,
            buffer_is_complete: false,
            notices,
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

    pub(super) fn update_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_outdated && self.texture.is_some() {
            return;
        }
        let rgb = colors_to_rgb_bytes(&self.p.colors, self.p.color_channel_mapping);
        let image = ColorImage::from_rgb([self.p.pixel_plane.width, self.p.pixel_plane.height], &rgb);
        match &mut self.texture {
            Some(texture) => texture.set(image, TextureOptions::NEAREST),
            None => self.texture = Some(ctx.load_texture("mandelbrot", image, TextureOptions::NEAREST)),
        }
        self.texture_outdated = false;
    }
}

impl eframe::App for MandelbrotApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job();
        self.poll_thumbnails(&ctx);
        if self.is_busy() {
            ctx.request_repaint_after(Duration::from_millis(100)); //Keep polling for the finished job
        }
        self.update_texture(&ctx);

        egui::Panel::right("side_panel").default_size(SIDE_PANEL_WIDTH).min_size(280.0).show(ui, |ui| self.side_panel_ui(ui));

        let mut fit_area = egui::Vec2::ZERO;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::BLACK)).show(ui, |ui| {
            let available = ui.available_size();
            fit_area = available;
            let Some(texture) = self.texture.clone() else { return };
            let (width, height) = (self.p.pixel_plane.width as f32, self.p.pixel_plane.height as f32);
            let scale = (available.x / width).min(available.y / height);
            let size = egui::vec2(width * scale, height * scale);
            let response = ui.add(egui::Image::new((texture.id(), size)).sense(egui::Sense::click()));
            if let Some(position) = response.interact_pointer_pos() {
                let relative = (position - response.rect.min) / size;
                let (x, y) = (relative.x * width, relative.y * height);
                if response.clicked() {
                    self.handle_image_click(x, y, false);
                } else if response.secondary_clicked() {
                    self.handle_image_click(x, y, true);
                }
            }
        });
        self.fit_to_window(&ctx, fit_area);
    }
}

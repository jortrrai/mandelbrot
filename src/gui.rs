//! The graphical user interface: the fractal on the left, the controls in the top right and a scrollable status log in the bottom right.
//! The terminal is only used for developer output.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};

use crate::auto_iterations::{self, AutoIterationSettings, AutoPreset};
use crate::coloring::{ColorChannelMapping, TrueColor};
use crate::complex_plane::{ComplexPlane, View};
use crate::mandelbrot_set::MandelbrotSet;
use crate::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
use crate::rendering::ExposedRegion;
use crate::{rendering, ColoringFunction, Config, InteractionVariables, COLORING_FUNCTION, COLOR_CHANNEL_MAPPING};
use crate::{VIEW_0, VIEW_1, VIEW_2, VIEW_3, VIEW_4, VIEW_5, VIEW_6, VIEW_7, VIEW_8, VIEW_9};

/// Width in points of the right side panel, used to size the initial window
pub const SIDE_PANEL_WIDTH: f32 = 380.0;
/// Minimum height in points of the window, so that the controls and the status log have room
pub const MIN_WINDOW_HEIGHT: f32 = 700.0;

const COLORING_FUNCTIONS: [(&str, ColoringFunction); 2] = [("HSV", TrueColor::new_from_hsv_colors), ("Bernstein polynomials", TrueColor::new_from_bernstein_polynomials)];

const COLOR_CHANNEL_MAPPINGS: [&str; 27] = [
    "RGB", "RBG", "GRB", "GBR", "BRG", "BGR", "RRR", "GGG", "BBB", "RRG", "RRB", "RGG", "RGR", "RBB", "RBR", "GGB", "GGR", "GBB", "GBG", "GRG", "GRR", "BBG", "BBR", "BGB", "BGG", "BRB", "BRR",
];

const VIEWS: [(&str, &View); 10] = [
    ("View 1", &VIEW_1), ("View 2", &VIEW_2), ("View 3", &VIEW_3), ("View 4", &VIEW_4), ("View 5", &VIEW_5),
    ("View 6", &VIEW_6), ("View 7", &VIEW_7), ("View 8", &VIEW_8), ("View 9", &VIEW_9), ("View 10", &VIEW_0),
];

/// A message from a rendering job that runs on a background thread
enum JobMessage {
    /// An intermediate image: the running average of the first `pass` of `total` supersampling layers
    Pass { colors: Vec<TrueColor>, pass: u8, total: u8, layer_time: Duration },
    /// The iteration limit that auto iterations chose for the render, sent as soon as it is known, before the first layer is rendered
    Limit { max_iterations: u32, palette_scale: u32 },
    /// The job has finished (or was cancelled), this is its final result
    Done { c: ComplexPlane, p: PixelBuffer, m: MandelbrotSet, message: String, elapsed: Duration, completed: bool },
}

/// What the render progress bar shows: the time of every finished layer and how far the current layer is.
/// It stays visible after the render has finished, showing the benchmark of that render.
struct RenderBars {
    total_layers: u8,
    /// How long every finished layer took
    layer_times: Vec<Duration>,
    /// When the render started, and the current layer started, for the running timers
    started: Instant,
    layer_started: Instant,
    /// The pixels of the current layer that are done, and the amount of pixels in a layer
    pixels_done: Arc<AtomicUsize>,
    pixels_total: usize,
    /// How long the whole render took, and whether it was completed, once it has finished
    finished: Option<(Duration, bool)>,
}

/// A progressive render that is currently running: how far it is and how to cancel it
struct Progress {
    pass: u8,
    total: u8,
    cancel: Arc<AtomicBool>,
}

/// Converts colors to RGB bytes for a texture, applying the color channel mapping while converting; the same colors as `PixelBuffer::colors_to_pixels`
fn colors_to_rgb_bytes(colors: &[TrueColor], mapping: &ColorChannelMapping) -> Vec<u8> {
    let (red, green, blue) = mapping.get_r_g_b_mapping();
    let channel_index = |channel: char| match channel { 'R' => 0, 'G' => 1, _ => 2 };
    let (red, green, blue) = (channel_index(red), channel_index(green), channel_index(blue));
    let mut rgb: Vec<u8> = Vec::with_capacity(colors.len() * 3);
    for color in colors {
        let channels = [color.red, color.green, color.blue];
        rgb.extend_from_slice(&[channels[red], channels[green], channels[blue]]);
    }
    rgb
}

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
    /// Edited values of the max iterations field, applied when editing ends
    max_iterations_input: u32,
    /// Whether the maximum iterations are chosen automatically for the zoom of every render, see `auto_iterations`
    auto_iterations: bool,
    auto_settings: AutoIterationSettings,
    auto_preset: AutoPreset,
    /// Whether the displayed auto limit is only the starting limit of the running render, because the probes that refine it have not finished
    limit_pending: bool,
    /// Whether supersampling puts the extra samples on boundary and detail pixels only, see `rendering::render_region_adaptively`
    adaptive_supersampling: bool,
    /// Edited value of the supersampling field, applied when editing ends
    supersampling_input: u8,
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
    pub fn new(config: &Config) -> MandelbrotApp {
        let c = ComplexPlane::new(config.window_width, config.window_height);
        let mut p = PixelBuffer::new(PixelPlane::new(config.window_width, config.window_height));
        p.color_channel_mapping = COLOR_CHANNEL_MAPPING;
        let m = MandelbrotSet::new(config.max_iterations, config.orbit_radius);
        let (notice_sender, notices) = mpsc::channel();
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
        app.render("Rendered the default view");
        app
    }

    /// Adds a message to the status box, and prints it to the terminal for developers
    fn log(&mut self, message: impl Into<String>) {
        let message = message.into();
        println!("{message}");
        self.log.push(message);
    }

    fn view_description(&self) -> String {
        let center = self.c.center();
        format!("center = ({}, {}), scale = {}", center.x, center.y, self.c.get_scale())
    }

    fn is_busy(&self) -> bool {
        self.job.is_some()
    }

    /// Whether the controls can be used: not while a move runs, but always while a progressive render runs, since using a control interrupts it
    fn controls_enabled(&self) -> bool {
        self.job.is_none() || self.progress.is_some()
    }

    /// Cancels the running progressive render, if any, and ignores everything it still reports.
    /// The image of the last completed layer stays on screen until the next render shows its first layer.
    fn interrupt_progressive_render(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.cancel.store(true, Ordering::Relaxed);
            self.job = None;
        }
    }

    /// Takes the messages of the background job: intermediate images and the final result
    fn poll_job(&mut self) {
        while let Ok(notice) = self.notices.try_recv() {
            self.log(notice);
        }
        loop {
            let Some(receiver) = &self.job else { return };
            match receiver.try_recv() {
                Ok(JobMessage::Pass { colors, pass, total, layer_time }) => {
                    self.p.colors = colors;
                    self.texture_outdated = true;
                    if let Some(progress) = &mut self.progress {
                        progress.pass = pass;
                        progress.total = total;
                    }
                    if let Some(bars) = &mut self.bars {
                        bars.layer_times.push(layer_time);
                        bars.layer_started = Instant::now();
                    }
                }
                Ok(JobMessage::Limit { max_iterations, palette_scale }) => {
                    //Only the running job can send this: the receiver of an interrupted job is dropped, so a stale limit never reaches the display
                    self.m.max_iterations = max_iterations;
                    self.m.palette_scale = Some(palette_scale);
                    self.limit_pending = false;
                }
                Ok(JobMessage::Done { c, p, m, message, elapsed, completed }) => {
                    self.job = None;
                    self.limit_pending = false;
                    self.m = m; //In auto mode this holds the chosen limit; the manual value in `max_iterations_input` is kept
                    self.buffer_is_complete = completed;
                    if self.progress.take().is_some() {
                        if let Some(bars) = &mut self.bars {
                            bars.finished = Some((elapsed, completed));
                        }
                    }
                    self.c = c;
                    self.p = p;
                    self.texture_outdated = true;
                    let description = self.view_description();
                    self.log(format!("{message} in {:.2} s: {description}", elapsed.as_secs_f64()));
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.job = None;
                    self.progress = None;
                    self.log("Rendering failed, see the terminal for details");
                }
            }
        }
    }

    /// Renders the complete view in the background, then logs `message` and the current view.
    /// Every supersampling layer is shown as soon as it is computed, see `rendering::render_region_progressively`
    fn render(&mut self, message: &str) {
        self.start_render(message, None);
    }

    /// Renders the view after it was moved, then logs `message` and the current view. `shift` is the `(rows_up, columns_right)` to shift the pixel buffer with
    /// `PixelBuffer::translate_buffer`: the old image is shifted and only the pixels that were exposed by the move are rendered, progressively.
    /// If the buffer is not complete, because a render was interrupted or cancelled, or if the shift is too large, the complete view is rendered.
    fn render_moved(&mut self, message: &str, shift: (i128, i128)) {
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let reusable = self.progress.is_none() && self.buffer_is_complete;
        let strip = if reusable { ExposedRegion::from_shift(width, height, shift.0, shift.1).map(|region| (shift, region)) } else { None };
        self.start_render(message, strip);
    }

    fn start_render(&mut self, message: &str, strip: Option<((i128, i128), ExposedRegion)>) {
        self.interrupt_progressive_render();
        self.buffer_is_complete = false;
        let (sender, receiver) = mpsc::channel();
        let (c, mut p) = (self.c.clone(), self.p.clone());
        let adaptive = self.adaptive_supersampling.then(rendering::AdaptiveSampling::default);
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        //The pixels of a buffer that was rendered with the other mode have another limit and palette, so they cannot be reused
        let strip = strip.filter(|_| self.auto_iterations == self.m.palette_scale.is_some());
        //Publish the limit of the new render right away, before anything is rendered: the starting limit of the auto settings (cheap and deterministic), refined by the job as soon as its probes finish
        self.limit_pending = false;
        if self.auto_iterations {
            if strip.is_none() {
                let initial = self.auto_settings.initial_limit(self.c.get_scale());
                self.m.max_iterations = initial;
                self.m.palette_scale = Some(initial);
                self.limit_pending = true;
            }
        } else {
            self.m.max_iterations = self.max_iterations_input;
            self.m.palette_scale = None;
        }
        let mut m = self.m.clone();
        let (supersampling_amount, coloring_function) = (self.supersampling_amount, self.coloring_function);
        let (width, height) = (p.pixel_plane.width, p.pixel_plane.height);
        let pixels_total = strip.as_ref().map_or(width * height, |(_, region)| region.pixel_count(width, height));
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let pixels_done = Arc::new(AtomicUsize::new(0));
        let thread_pixels_done = Arc::clone(&pixels_done);
        let message = message.to_string();
        std::thread::spawn(move || {
            let started = Instant::now();
            let mut message = message;
            if auto_settings.is_some() && strip.is_some() {
                //A move keeps the limit of the image that is being moved, so only the exposed strip is rendered; the limit is chosen again on zooms, jumps and setting changes
            } else if let Some(settings) = &auto_settings {
                //Choose the iteration limit for this zoom first; a cancelled choice cancels the render
                match auto_iterations::choose_limit(&c, &m, width, height, settings, rendering::render_thread_count(), &thread_cancel) {
                    Some(result) => {
                        let _ = sender.send(JobMessage::Limit { max_iterations: result.limit, palette_scale: result.palette_scale });
                        m.max_iterations = result.limit;
                        m.palette_scale = Some(result.palette_scale);
                        message = format!("{message} (auto iterations: limit {}, {} rounds, {:?}, {:.2} s)", result.limit, result.rounds.len(), result.stop, started.elapsed().as_secs_f64());
                        println!("{result:?}");
                    }
                    None => {
                        let _ = sender.send(JobMessage::Done { c, p, m, message: String::from("Cancelled the rendering"), elapsed: started.elapsed(), completed: false });
                        return;
                    }
                }
            } else if auto_settings.is_none() {
                m.palette_scale = None;
            }
            let region = match strip {
                Some((shift, region)) => {
                    p.translate_buffer(shift.0, shift.1);
                    region
                }
                None => ExposedRegion::full(width, height),
            };
            let base = p.colors.clone();
            let mut layer_started = started;
            let completed = rendering::render_region_adaptively(&c, &m, width, height, &region, &base, supersampling_amount, adaptive.as_ref(), coloring_function, &thread_cancel, &thread_pixels_done, |colors, pass| {
                //Only the colors are handed over, the window converts them when it draws; converting them here would delay the next layer
                p.colors.copy_from_slice(colors);
                let layer_time = layer_started.elapsed();
                println!("Rendered supersampling layer {pass} of {supersampling_amount} in {:.3} s", layer_time.as_secs_f64());
                let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), pass, total: supersampling_amount, layer_time });
                layer_started = Instant::now();
            });
            p.update_pixels(); //The pixels of the last completed layer
            let message = if completed { message } else { String::from("Cancelled the rendering, showing the layers that were completed") };
            let _ = sender.send(JobMessage::Done { c, p, m, message, elapsed: started.elapsed(), completed }); //The receiver is gone if the window was closed
        });
        let now = Instant::now();
        self.bars = Some(RenderBars { total_layers: supersampling_amount.clamp(1, 64), layer_times: Vec::new(), started: now, layer_started: now, pixels_done, pixels_total, finished: None });
        self.job = Some(receiver);
        self.progress = Some(Progress { pass: 0, total: supersampling_amount, cancel });
        self.log("Rendering...");
    }

    /// Selects an auto iterations preset. With auto iterations on, the limit of the preset is published immediately (see `start_render`) and the render with it starts in the background.
    fn select_preset(&mut self, preset: AutoPreset) {
        self.auto_preset = preset;
        self.auto_settings = preset.settings();
        if self.auto_iterations {
            self.render(&format!("Set the auto iterations preset to {}", preset.name()));
        } else {
            self.log(format!("Set the auto iterations preset to {}", preset.name()));
        }
    }

    fn translate(&mut self, name: &str, rows_up: i16, columns_right: i16) {
        rendering::translate_complex_plane_by_pixels(&mut self.c, rows_up, columns_right);
        //Moving the view to the right moves the content of the buffer to the left
        self.render_moved(&format!("Moved {name} {} pixels", self.vars.translation_amount), (i128::from(rows_up), -i128::from(columns_right)));
    }

    /// Saves the current view as an image on its own background thread. This does not interrupt a running render;
    /// while one runs, the buffer only holds a partially refined image, so the image is rendered again for saving.
    fn save_image(&mut self) {
        let time_stamp = chrono::Utc::now().to_string();
        let file_name = time_stamp.replace(':', "-").replace(' ', "_");
        let (c, p, m) = (self.c.clone(), self.p.clone(), self.m.clone());
        let (config, image_supersampling_amount, coloring_function) = (self.config.clone(), self.image_supersampling_amount, self.coloring_function);
        let buffer_is_complete = self.progress.is_none() && !self.adaptive_supersampling; //An adaptively sampled buffer is not a full-quality image of the saved quality
        let sender = self.notice_sender.clone();
        std::thread::spawn(move || {
            let view = c.get_view();
            if config.window_scale == 1.0 && buffer_is_complete {
                p.save_as_png(&time_stamp, &view, &m, image_supersampling_amount);
            } else {
                let mut image_p = PixelBuffer::new(PixelPlane::new(config.image_width, config.image_height));
                let mut image_c = ComplexPlane::new(config.image_width, config.image_height);
                image_p.color_channel_mapping = p.color_channel_mapping;
                image_c.set_view(&view);
                rendering::render_complex_plane_into_buffer(&mut image_p, &image_c, &m, image_supersampling_amount, coloring_function);
                image_p.save_as_png(&time_stamp, &view, &m, image_supersampling_amount);
            }
            let _ = sender.send(format!("Saved the image to saved/{file_name}.png"));
        });
        self.log("Saving the image...");
    }

    fn go_to_view(&mut self) {
        let parsed: Vec<Result<f64, _>> = self.view_inputs.iter().map(|text| text.trim().parse::<f64>()).collect();
        match parsed.as_slice() {
            [Ok(x), Ok(y), Ok(scale)] if *scale > 0.0 => {
                self.c.set_view(&View::new(*x, *y, *scale));
                self.render("Moved to the entered view");
            }
            _ => self.log("Could not go to the view: x, y and scale should be numbers, and scale should be positive"),
        }
    }

    /// Handles a click on the fractal at `(x, y)` in pixel buffer coordinates
    fn handle_image_click(&mut self, x: f32, y: f32, centering: bool) {
        let complex = self.c.complex_from_pixel_plane(x.into(), y.into());
        if centering {
            let shift = rendering::center_complex_plane_on(&mut self.c, &complex);
            self.render_moved(&format!("Centered the view on ({:.10}, {:.10})", complex.x, complex.y), shift);
        } else {
            self.log(format!("Clicked pixel ({}, {}) = complex point ({:.10}, {:.10})", x as usize, y as usize, complex.x, complex.y));
        }
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_outdated && self.texture.is_some() {
            return;
        }
        let rgb = colors_to_rgb_bytes(&self.p.colors, &self.p.color_channel_mapping);
        let image = ColorImage::from_rgb([self.p.pixel_plane.width, self.p.pixel_plane.height], &rgb);
        match &mut self.texture {
            Some(texture) => texture.set(image, TextureOptions::NEAREST),
            None => self.texture = Some(ctx.load_texture("mandelbrot", image, TextureOptions::NEAREST)),
        }
        self.texture_outdated = false;
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");

        ui.label("Move");
        egui::Grid::new("move_grid").show(ui, |ui| {
            ui.label("");
            if ui.button("Up").clicked() {
                self.translate("up", self.vars.translation_amount.into(), 0);
            }
            ui.end_row();
            if ui.button("Left").clicked() {
                self.translate("left", 0, -i16::from(self.vars.translation_amount));
            }
            if ui.button("Down").clicked() {
                self.translate("down", -i16::from(self.vars.translation_amount), 0);
            }
            if ui.button("Right").clicked() {
                self.translate("right", 0, self.vars.translation_amount.into());
            }
            ui.end_row();
        });
        ui.horizontal(|ui| {
            ui.label("Move step (pixels):");
            ui.add(egui::DragValue::new(&mut self.vars.translation_amount).range(1..=255));
        });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Zoom in").clicked() {
                self.c.scale(self.vars.scaling_factor());
                self.render("Zoomed in");
            }
            if ui.button("Zoom out").clicked() {
                self.c.scale(self.vars.inverse_scaling_factor());
                self.render("Zoomed out");
            }
            if ui.button("Reset view").clicked() {
                self.c.reset();
                self.render("Reset the view");
            }
        });
        ui.horizontal(|ui| {
            ui.label("Zoom speed (1 = fastest, 9 = slowest):");
            ui.add(egui::DragValue::new(&mut self.vars.scale_numerator).range(1.0..=9.0).speed(0.1).max_decimals(0));
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Go to view:");
            egui::ComboBox::from_id_salt("views").selected_text("Pick a view").show_ui(ui, |ui| {
                for (name, view) in VIEWS {
                    if ui.selectable_label(false, name).clicked() {
                        self.c.set_view(view);
                        self.render(&format!("Moved to {name}"));
                    }
                }
            });
        });
        ui.horizontal(|ui| {
            for (input, hint) in self.view_inputs.iter_mut().zip(["x", "y", "scale"]) {
                ui.add(egui::TextEdit::singleline(input).hint_text(hint).desired_width(80.0));
            }
            if ui.button("Go").clicked() {
                self.go_to_view();
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Max. iterations:");
            let response = ui.add_enabled(!self.auto_iterations, egui::DragValue::new(&mut self.max_iterations_input).range(1..=1_000_000).speed(10.0));
            if (response.drag_stopped() || response.lost_focus()) && self.max_iterations_input != self.m.max_iterations {
                self.m.max_iterations = self.max_iterations_input;
                self.render(&format!("Set max. iterations to {}", self.m.max_iterations));
            }
        });
        let mut preset_selected: Option<AutoPreset> = None;
        let mut auto_changed = false;
        ui.horizontal(|ui| {
            auto_changed = ui.checkbox(&mut self.auto_iterations, "Auto iterations").on_hover_text("Chooses the max. iterations for the zoom of every render: a starting limit that grows with the magnification, raised while the extra iterations make more pixels escape. A heuristic, not a proof: very narrow detail can be missed.").changed();
            egui::ComboBox::from_id_salt("auto_preset").selected_text(self.auto_preset.name()).show_ui(ui, |ui| {
                for preset in AutoPreset::SELECTABLE {
                    if ui.selectable_label(self.auto_preset == preset, preset.name()).on_hover_text(match preset {
                        AutoPreset::Performance => "Lower limits, coarser probes: faster, may leave deep boundary detail unresolved",
                        AutoPreset::Balanced => "The default",
                        _ => "Higher limits, finer probes: resolves more structure, can be much slower",
                    }).clicked() && self.auto_preset != preset {
                        preset_selected = Some(preset);
                    }
                }
            });
        });
        if let Some(preset) = preset_selected {
            self.select_preset(preset);
        }
        if auto_changed {
            if self.auto_iterations {
                self.render("Turned auto iterations on");
            } else {
                self.render(&format!("Turned auto iterations off, max. iterations {}", self.max_iterations_input));
            }
        }
        if ui.checkbox(&mut self.adaptive_supersampling, "Adaptive supersampling").on_hover_text("Takes the first 2 samples of every pixel, and the remaining samples only for boundary and detail pixels (and their neighbors). Faster on smooth views; a heuristic that can differ from full supersampling. Saved images always use the full saved image quality.").changed() {
            let state = if self.adaptive_supersampling { "on" } else { "off" };
            self.render(&format!("Turned adaptive supersampling {state}"));
        }
        if self.auto_iterations {
            ui.label(format!("Current limit: {} iterations ({})", self.m.max_iterations, if self.limit_pending { "starting value, being refined..." } else { "chosen automatically" }));
            ui.horizontal(|ui| {
                ui.label("Start at zoom 1:");
                let a = ui.add(egui::DragValue::new(&mut self.auto_settings.start).range(10..=100_000).speed(10.0));
                ui.label("+ per zoom doubling:");
                let b = ui.add(egui::DragValue::new(&mut self.auto_settings.per_doubling).range(0..=10_000).speed(5.0));
                ui.label("up to:");
                let c = ui.add(egui::DragValue::new(&mut self.auto_settings.ceiling).range(100..=200_000).speed(100.0));
                if [a, b, c].iter().any(|r| r.drag_stopped() || r.lost_focus()) {
                    self.auto_preset = if self.auto_settings.start == self.auto_preset.settings().start && self.auto_settings.per_doubling == self.auto_preset.settings().per_doubling && self.auto_settings.ceiling == self.auto_preset.settings().ceiling { self.auto_preset } else { AutoPreset::Custom };
                    self.render("Changed the auto iterations settings");
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Quality (supersampling, 1x-64x):");
            let response = ui.add(egui::DragValue::new(&mut self.supersampling_input).range(1..=64).suffix("x"));
            if (response.drag_stopped() || response.lost_focus()) && self.supersampling_input != self.supersampling_amount {
                self.supersampling_amount = self.supersampling_input;
                self.image_supersampling_amount = self.supersampling_input;
                self.render(&format!("Set the supersampling to {}x", self.supersampling_amount));
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Coloring:");
            egui::ComboBox::from_id_salt("coloring").selected_text(COLORING_FUNCTIONS[self.coloring_index].0).show_ui(ui, |ui| {
                for (index, (name, function)) in COLORING_FUNCTIONS.into_iter().enumerate() {
                    if ui.selectable_label(index == self.coloring_index, name).clicked() && index != self.coloring_index {
                        self.coloring_index = index;
                        self.coloring_function = function;
                        self.render(&format!("Set the coloring to {name}"));
                    }
                }
            });
        });
        ui.horizontal(|ui| {
            ui.label("Color channels (R, G, B):");
            let current = format!("{:?}", self.p.color_channel_mapping);
            egui::ComboBox::from_id_salt("channels").selected_text(&current).show_ui(ui, |ui| {
                for name in COLOR_CHANNEL_MAPPINGS {
                    if ui.selectable_label(name == current, name).clicked() && name != current {
                        if let Ok(mapping) = ColorChannelMapping::new(name) {
                            self.p.color_channel_mapping = mapping;
                            if self.progress.is_some() {
                                self.render(&format!("Set the color channel mapping to {name}")); //The running render still uses the old mapping
                            } else {
                                self.p.update_pixels();
                                self.texture_outdated = true;
                                self.log(format!("Set the color channel mapping to {name}"));
                            }
                        }
                    }
                }
            });
        });

        ui.separator();
        ui.heading("Save Image");
        ui.horizontal(|ui| {
            ui.label("Saved image quality:");
            ui.add(egui::DragValue::new(&mut self.image_supersampling_amount).range(1..=64).suffix("x"));
        });
        if ui.button("Save image").clicked() {
            self.save_image();
        }
        ui.separator();
        ui.label("Click the fractal to see a point's coordinates, right-click to center the view on it.");
    }

    /// Draws the render progress bar: a main bar with a smaller bar for every supersampling layer, each showing how long that layer took.
    /// The caption above it shows the total time of the render.
    fn render_bars_ui(ui: &mut egui::Ui, bars: &RenderBars) {
        let caption = match bars.finished {
            Some((total, true)) => format!("Rendered in {:.2} s", total.as_secs_f64()),
            Some((total, false)) => format!("Cancelled after {} of {} layers, {:.2} s", bars.layer_times.len(), bars.total_layers, total.as_secs_f64()),
            None => format!("Rendering: {:.1} s", bars.started.elapsed().as_secs_f64()),
        };
        ui.label(caption);

        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 30.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(40));
        let inner = rect.shrink(3.0);
        let gap = 2.0;
        let layers = f32::from(bars.total_layers);
        let layer_width = (inner.width() - gap * (layers - 1.0)) / layers;
        let current_fraction = (bars.pixels_done.load(Ordering::Relaxed) as f32 / bars.pixels_total.max(1) as f32).min(1.0);
        for layer in 0..usize::from(bars.total_layers) {
            let left = inner.left() + layer as f32 * (layer_width + gap);
            let layer_rect = egui::Rect::from_min_size(egui::pos2(left, inner.top()), egui::vec2(layer_width, inner.height()));
            painter.rect_filled(layer_rect, 2.0, Color32::from_gray(70));
            let finished_time = bars.layer_times.get(layer);
            let is_current = finished_time.is_none() && layer == bars.layer_times.len() && bars.finished.is_none();
            let fraction = if finished_time.is_some() { 1.0 } else if is_current { current_fraction.clamp(0.0, 1.0) } else { 0.0 };
            let fill = egui::Rect::from_min_size(layer_rect.min, egui::vec2(layer_width * fraction, layer_rect.height()));
            painter.rect_filled(fill, 2.0, Color32::from_rgb(60, 140, 220));
            let label = match finished_time {
                Some(time) => Some(format!("{:.2}s", time.as_secs_f64())),
                None if is_current => Some(format!("{:.1}s", bars.layer_started.elapsed().as_secs_f64())),
                None => None,
            };
            if let Some(label) = label {
                if layer_width >= 40.0 {
                    painter.text(layer_rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(11.0), Color32::WHITE);
                }
            }
        }
    }

    fn log_ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Status");
            if self.is_busy() {
                ui.spinner();
                match &self.progress {
                    Some(progress) => {
                        ui.label(format!("Rendering layer {} of {}...", (progress.pass + 1).min(progress.total), progress.total));
                        if ui.button("Cancel").clicked() {
                            progress.cancel.store(true, Ordering::Relaxed);
                        }
                    }
                    None => {
                        ui.label("Rendering...");
                    }
                }
            }
        });
        if let Some(bars) = &self.bars {
            Self::render_bars_ui(ui, bars);
            ui.add_space(4.0);
        }
        egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
            for message in &self.log {
                ui.add(egui::Label::new(message).wrap());
            }
        });
    }
}

impl eframe::App for MandelbrotApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job();
        if self.is_busy() {
            ctx.request_repaint_after(Duration::from_millis(100)); //Keep polling for the finished job
        }
        self.update_texture(&ctx);

        egui::Panel::right("side_panel").default_size(SIDE_PANEL_WIDTH).min_size(280.0).show(ui, |ui| {
            egui::Panel::bottom("status_panel").resizable(true).default_size(220.0).min_size(80.0).show(ui, |ui| self.log_ui(ui));
            egui::CentralPanel::default().show(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let enabled = self.controls_enabled();
                    ui.add_enabled_ui(enabled, |ui| self.controls_ui(ui));
                });
            });
        });

        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::BLACK)).show(ui, |ui| {
            let Some(texture) = self.texture.clone() else { return };
            let (width, height) = (self.p.pixel_plane.width as f32, self.p.pixel_plane.height as f32);
            let available = ui.available_size();
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_for_job(app: &mut MandelbrotApp) {
        let start = std::time::Instant::now();
        while app.is_busy() {
            assert!(start.elapsed() < Duration::from_secs(60), "the background job did not finish");
            std::thread::sleep(Duration::from_millis(10));
            app.poll_job();
        }
    }

    #[test]
    fn rendering_runs_in_the_background_and_updates_the_buffer() {
        let config = Config::build(["mandelbrot", "64", "48", "50"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        assert!(app.is_busy(), "the initial render should run in the background");
        wait_for_job(&mut app);
        assert!(app.p.pixels.iter().any(|pixel| *pixel != 0), "the default view should not be all black");
        assert!(app.log.last().unwrap().starts_with("Rendered the default view"));

        let before = app.p.pixels.clone();
        app.c.scale(0.5);
        app.render("Zoomed in");
        assert!(app.is_busy());
        wait_for_job(&mut app);
        assert_ne!(before, app.p.pixels, "zooming should change the image");
    }

    #[test]
    fn a_new_action_interrupts_a_progressive_render_and_starts_over() {
        let config = Config::build(["mandelbrot", "64", "48", "50", "8"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        assert!(app.progress.is_some());
        let first_cancel = Arc::clone(&app.progress.as_ref().unwrap().cancel);

        //Moving while the first render is still running interrupts it, and renders the moved view from scratch
        app.translate("up", 10, 0);
        assert!(first_cancel.load(Ordering::Relaxed), "the first render should have been cancelled");
        assert!(app.progress.is_some(), "a new progressive render should be running");
        let moved_center = app.c.center();
        wait_for_job(&mut app);

        assert!(app.progress.is_none());
        assert_eq!(app.p.pixels.len(), 64 * 48);
        assert!(app.p.pixels.iter().any(|pixel| *pixel != 0));
        assert_eq!(app.c.center().y, moved_center.y, "the moved view should be kept");
        assert!(!app.log.iter().any(|message| message.starts_with("Cancelled")), "interrupting is not reported as a cancelled render");
        assert!(app.log.last().unwrap().starts_with("Moved up 10 pixels"));
    }

    #[test]
    fn the_progress_bar_records_the_time_of_every_layer() {
        let config = Config::build(["mandelbrot", "64", "48", "50", "3"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);

        let bars = app.bars.as_ref().expect("a full render should have a progress bar");
        assert_eq!(bars.total_layers, 3);
        assert_eq!(bars.layer_times.len(), 3, "every layer should have a time");
        let (total, completed) = bars.finished.expect("the render should be finished");
        assert!(completed);
        assert!(total >= bars.layer_times.iter().sum(), "the total includes all layers");
        assert!(app.log.last().unwrap().contains(" s: "), "the log should mention the render time: {:?}", app.log.last());

        //Moving renders the exposed strip progressively too, with its own bar
        app.translate("up", 10, 0);
        assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 0, "the move starts a new bar");
        wait_for_job(&mut app);
        assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 3, "every layer of the move is rendered and timed");
        assert!(app.log.last().unwrap().starts_with("Moved up 10 pixels in"));
    }

    #[test]
    fn moving_reuses_the_old_image_and_only_renders_the_exposed_strip() {
        let config = Config::build(["mandelbrot", "64", "48", "50", "3"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        let old_pixels = app.p.pixels.clone();

        app.translate("up", 10, 0); //The content of the buffer moves down by 10 rows
        assert_eq!(app.bars.as_ref().unwrap().pixels_total, 10 * 64, "only the exposed top strip should be rendered");
        wait_for_job(&mut app);

        assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 3, "every layer of the strip is rendered progressively");
        for y in 10..48 {
            assert_eq!(app.p.pixels[y * 64..(y + 1) * 64], old_pixels[(y - 10) * 64..(y - 9) * 64], "row {y} should be the shifted old image");
        }
        assert!(app.buffer_is_complete);
    }

    #[test]
    fn moving_after_a_cancelled_render_renders_everything() {
        let config = Config::build(["mandelbrot", "64", "48", "50", "6"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        app.progress.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
        wait_for_job(&mut app);
        assert!(!app.buffer_is_complete, "a cancelled render leaves a partially refined buffer");

        app.translate("left", 0, -10);
        assert_eq!(app.bars.as_ref().unwrap().pixels_total, 64 * 48, "a partially refined buffer cannot be reused");
        wait_for_job(&mut app);
        assert!(app.buffer_is_complete);
    }

    #[test]
    fn texture_bytes_equal_the_converted_pixels_for_all_mappings() {
        let colors: Vec<TrueColor> = (0..=255u8).map(|i| TrueColor::new(i, 255 - i, i / 3)).collect();
        for name in COLOR_CHANNEL_MAPPINGS {
            let mapping = ColorChannelMapping::new(name).unwrap();
            let pixels = PixelBuffer::colors_to_pixels(&colors, &mapping);
            let expected: Vec<u8> = pixels.iter().flat_map(|pixel| [(pixel >> 16) as u8, (pixel >> 8) as u8, *pixel as u8]).collect();
            assert_eq!(colors_to_rgb_bytes(&colors, &mapping), expected, "{name}");
        }
    }

    #[test]
    fn auto_iterations_switch_with_manual_and_keep_the_palette_stable_on_moves() {
        let config = Config::build(["mandelbrot", "64", "48", "50"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        assert_eq!(app.m.max_iterations, 50);
        assert_eq!(app.m.palette_scale, None, "manual mode colors exactly as before");
        let manual_pixels = app.p.pixels.clone();

        //Turn auto on: the limit is chosen for the zoom, and shown in the max. iterations field
        app.auto_iterations = true;
        app.render("Turned auto iterations on");
        wait_for_job(&mut app);
        let limit = app.m.max_iterations;
        assert_eq!(app.m.palette_scale, Some(app.auto_settings.initial_limit(app.c.get_scale())));
        assert!(app.log.last().unwrap().contains("auto iterations: limit"), "{:?}", app.log.last());
        assert_eq!(app.max_iterations_input, 50, "the manual value is kept while auto is on");

        //A move keeps the limit and only renders the strip
        app.translate("up", 10, 0);
        assert_eq!(app.bars.as_ref().unwrap().pixels_total, 10 * 64);
        wait_for_job(&mut app);
        assert_eq!(app.m.max_iterations, limit);

        //A zoom chooses a new limit, a deeper zoom never a lower start
        app.c.scale(1e-4);
        app.render("Zoomed in");
        wait_for_job(&mut app);
        assert!(app.m.max_iterations >= app.auto_settings.initial_limit(app.c.get_scale()));
        assert!(app.m.max_iterations > limit.min(300) || limit >= 300);

        //Turn auto off: the manual value is used again and the image is the manual image
        app.auto_iterations = false;
        app.c.reset();
        app.render("Turned auto iterations off");
        wait_for_job(&mut app);
        assert_eq!(app.m.max_iterations, 50);
        assert_eq!(app.m.palette_scale, None);
        assert_eq!(app.p.pixels, manual_pixels, "back in manual mode the same view renders the same image");
    }

    #[test]
    fn moving_after_turning_auto_off_does_not_reuse_an_auto_colored_buffer() {
        let config = Config::build(["mandelbrot", "64", "48", "50"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        app.auto_iterations = true;
        app.render("auto");
        wait_for_job(&mut app);
        app.auto_iterations = false;
        app.translate("up", 10, 0);
        wait_for_job(&mut app);
        assert_eq!(app.m.palette_scale, None);
        assert_eq!(app.bars.as_ref().unwrap().pixels_total, 64 * 48, "the buffer was colored with the auto palette and is rendered again");
    }

    #[test]
    fn presets_and_adaptive_supersampling_toggle_independently_of_auto_iterations() {
        let config = Config::build(["mandelbrot", "64", "48", "50", "4"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        assert_eq!(app.auto_preset, AutoPreset::Balanced, "Balanced is the default");
        assert!(!app.auto_iterations && !app.adaptive_supersampling);
        let uniform = app.p.pixels.clone();

        //Adaptive supersampling works in manual mode, and switching it off gives the uniform image again
        app.adaptive_supersampling = true;
        app.render("adaptive on");
        wait_for_job(&mut app);
        assert_eq!(app.m.palette_scale, None);
        assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 4, "every layer is still reported");
        app.adaptive_supersampling = false;
        app.render("adaptive off");
        wait_for_job(&mut app);
        assert_eq!(app.p.pixels, uniform);

        //A preset changes the settings used by auto iterations
        app.auto_iterations = true;
        app.adaptive_supersampling = true;
        app.auto_preset = AutoPreset::HighDetail;
        app.auto_settings = AutoPreset::HighDetail.settings();
        app.render("high detail");
        wait_for_job(&mut app);
        let high = app.m.max_iterations;
        app.auto_preset = AutoPreset::Performance;
        app.auto_settings = AutoPreset::Performance.settings();
        app.render("performance");
        wait_for_job(&mut app);
        assert!(app.m.max_iterations <= high);
        assert_eq!(app.m.palette_scale, Some(AutoPreset::Performance.settings().initial_limit(app.c.get_scale())));
    }

    #[test]
    fn selecting_a_preset_shows_its_limit_immediately_and_stale_renders_cannot_overwrite_it() {
        let config = Config::build(["mandelbrot", "64", "48", "50", "2"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        app.c.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.000064));
        app.auto_iterations = true;
        app.render("auto");
        wait_for_job(&mut app);
        let balanced = app.m.max_iterations;

        //Selecting: the starting limit of the preset is shown before any poll, while the render is only starting
        app.select_preset(AutoPreset::HighDetail);
        let initial = AutoPreset::HighDetail.settings().initial_limit(app.c.get_scale());
        assert_eq!((app.m.max_iterations, app.m.palette_scale), (initial, Some(initial)), "published synchronously on selection");
        assert!(app.limit_pending && app.is_busy());
        assert_ne!(app.m.max_iterations, balanced);

        //The refined limit is published before the render has finished
        let start = std::time::Instant::now();
        while app.limit_pending {
            assert!(start.elapsed() < Duration::from_secs(60));
            std::thread::sleep(Duration::from_millis(5));
            app.poll_job();
        }
        assert!(app.m.max_iterations >= initial);
        let high = app.m.max_iterations;
        wait_for_job(&mut app);
        assert_eq!(app.m.max_iterations, high);

        //Selecting twice quickly: the first render is interrupted and cannot change what is shown
        app.select_preset(AutoPreset::Performance);
        let first_cancel = Arc::clone(&app.progress.as_ref().unwrap().cancel);
        app.select_preset(AutoPreset::Balanced);
        assert!(first_cancel.load(Ordering::Relaxed));
        let balanced_initial = AutoPreset::Balanced.settings().initial_limit(app.c.get_scale());
        assert_eq!(app.m.max_iterations, balanced_initial);
        wait_for_job(&mut app);
        assert_eq!(app.m.palette_scale, Some(balanced_initial));
        assert_eq!(app.m.max_iterations, balanced, "the same view and preset give the same limit as before, not the Performance one");
        assert!(!app.limit_pending);
    }

    /// The texts that `controls_ui` draws, in drawing order
    fn controls_texts(app: &mut MandelbrotApp) -> Vec<String> {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| app.controls_ui(ui));
        let texts = output.shapes.iter().filter_map(|shape| match &shape.shape {
            egui::epaint::Shape::Text(text) => Some(text.galley.text().to_string()),
            _ => None,
        }).collect();
        output.drop_without_applying_deltas();
        texts
    }

    #[test]
    fn saving_has_its_own_section_and_the_rendering_quality_stays_with_the_rendering_controls() {
        let config = Config::build(["mandelbrot", "64", "48", "50"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        let texts = controls_texts(&mut app);
        let position = |needle: &str| texts.iter().position(|t| t.starts_with(needle)).unwrap_or_else(|| panic!("{needle:?} is not drawn: {texts:?}"));
        let (quality, coloring, channels) = (position("Quality (supersampling"), position("Coloring:"), position("Color channels"));
        let (heading, saved_quality, save) = (position("Save Image"), position("Saved image quality:"), position("Save image"));
        assert!(quality < coloring && coloring < channels, "the rendering quality is with the rendering controls");
        assert!(channels < heading && heading < saved_quality && saved_quality < save, "the saved image quality and the Save button are together in the Save Image section, after the rendering controls");
        assert_eq!(texts.iter().filter(|t| t.starts_with("Saved image quality")).count(), 1);
    }
}

//! The graphical user interface: the fractal on the left, the controls in the top right and a scrollable status log in the bottom right.
//! The terminal is only used for developer output.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};

use crate::coloring::{ColorChannelMapping, TrueColor};
use crate::complex_plane::{ComplexPlane, View};
use crate::mandelbrot_set::MandelbrotSet;
use crate::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
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
    Pass { colors: Vec<TrueColor>, pixels: Vec<u32>, pass: u8, total: u8, layer_time: Duration },
    /// The job has finished (or was cancelled), this is its final result
    Done { c: ComplexPlane, p: PixelBuffer, message: String, elapsed: Duration, completed: bool },
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
    /// The rows of the current layer that are done, and the amount of rows in a layer
    rows_done: Arc<AtomicUsize>,
    rows: usize,
    /// How long the whole render took, and whether it was completed, once it has finished
    finished: Option<(Duration, bool)>,
}

/// A progressive render that is currently running: how far it is and how to cancel it
struct Progress {
    pass: u8,
    total: u8,
    cancel: Arc<AtomicBool>,
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

    /// Runs `job` on a clone of the complex plane and pixel buffer on a background thread, so the window stays responsive.
    /// When the job finishes, the results replace the current plane and buffer, and `message` and the current view are logged.
    fn start_job(&mut self, message: String, job: impl FnOnce(&mut ComplexPlane, &mut PixelBuffer, &MandelbrotSet, u8, ColoringFunction) + Send + 'static) {
        let (sender, receiver) = mpsc::channel();
        let (mut c, mut p, m) = (self.c.clone(), self.p.clone(), self.m.clone());
        let (supersampling_amount, coloring_function) = (self.supersampling_amount, self.coloring_function);
        std::thread::spawn(move || {
            let started = Instant::now();
            job(&mut c, &mut p, &m, supersampling_amount, coloring_function);
            let _ = sender.send(JobMessage::Done { c, p, message, elapsed: started.elapsed(), completed: true }); //The receiver is gone if the window was closed
        });
        self.job = Some(receiver);
        self.bars = None; //The bars would show an older render
        self.log("Rendering...");
    }

    /// Takes the messages of the background job: intermediate images and the final result
    fn poll_job(&mut self) {
        while let Ok(notice) = self.notices.try_recv() {
            self.log(notice);
        }
        loop {
            let Some(receiver) = &self.job else { return };
            match receiver.try_recv() {
                Ok(JobMessage::Pass { colors, pixels, pass, total, layer_time }) => {
                    self.p.colors = colors;
                    self.p.pixels = pixels;
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
                Ok(JobMessage::Done { c, p, message, elapsed, completed }) => {
                    self.job = None;
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

    /// Renders the complete view in the background, then logs `message` and the current view
    /// Every supersampling layer is shown as soon as it is computed, see `rendering::render_progressively`
    fn render(&mut self, message: &str) {
        self.interrupt_progressive_render();
        let (sender, receiver) = mpsc::channel();
        let (c, mut p, m) = (self.c.clone(), self.p.clone(), self.m.clone());
        let (supersampling_amount, coloring_function) = (self.supersampling_amount, self.coloring_function);
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let rows_done = Arc::new(AtomicUsize::new(0));
        let thread_rows_done = Arc::clone(&rows_done);
        let message = message.to_string();
        std::thread::spawn(move || {
            let (width, height) = (p.pixel_plane.width, p.pixel_plane.height);
            let started = Instant::now();
            let mut layer_started = started;
            let completed = rendering::render_progressively(&c, &m, width, height, supersampling_amount, coloring_function, &thread_cancel, &thread_rows_done, |colors, pass| {
                p.colors.copy_from_slice(colors);
                p.update_pixels();
                let layer_time = layer_started.elapsed();
                println!("Rendered supersampling layer {pass} of {supersampling_amount} in {:.3} s", layer_time.as_secs_f64());
                let _ = sender.send(JobMessage::Pass { colors: p.colors.clone(), pixels: p.pixels.clone(), pass, total: supersampling_amount, layer_time });
                layer_started = Instant::now();
            });
            let message = if completed { message } else { String::from("Cancelled the rendering, showing the layers that were completed") };
            let _ = sender.send(JobMessage::Done { c, p, message, elapsed: started.elapsed(), completed }); //The receiver is gone if the window was closed
        });
        let now = Instant::now();
        self.bars = Some(RenderBars { total_layers: supersampling_amount.clamp(1, 64), layer_times: Vec::new(), started: now, layer_started: now, rows_done, rows: self.p.pixel_plane.height, finished: None });
        self.job = Some(receiver);
        self.progress = Some(Progress { pass: 0, total: supersampling_amount, cancel });
        self.log("Rendering...");
    }

    fn translate(&mut self, name: &str, rows_up: i16, columns_right: i16) {
        let message = format!("Moved {name} {} pixels", self.vars.translation_amount);
        if self.progress.is_some() {
            //The buffer only holds a partially refined image, so it cannot be reused: move the view and start over
            rendering::translate_complex_plane_by_pixels(&mut self.c, rows_up, columns_right);
            self.render(&message);
            return;
        }
        self.start_job(message, move |c, p, m, supersampling_amount, coloring_function| {
            rendering::translate_and_render_efficiently(c, p, m, rows_up, columns_right, supersampling_amount, coloring_function);
        });
    }

    /// Saves the current view as an image on its own background thread. This does not interrupt a running render;
    /// while one runs, the buffer only holds a partially refined image, so the image is rendered again for saving.
    fn save_image(&mut self) {
        let time_stamp = chrono::Utc::now().to_string();
        let file_name = time_stamp.replace(':', "-").replace(' ', "_");
        let (c, p, m) = (self.c.clone(), self.p.clone(), self.m.clone());
        let (config, image_supersampling_amount, coloring_function) = (self.config.clone(), self.image_supersampling_amount, self.coloring_function);
        let buffer_is_complete = self.progress.is_none();
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
        if self.job.is_some() && self.progress.is_none() {
            return; //The view is being moved, clicks would map to a stale image
        }
        let complex = self.c.complex_from_pixel_plane(x.into(), y.into());
        if centering && self.progress.is_some() {
            self.c.set_center(&complex);
            self.render(&format!("Centered the view on ({:.10}, {:.10})", complex.x, complex.y));
        } else if centering {
            let message = format!("Centered the view on ({:.10}, {:.10})", complex.x, complex.y);
            self.start_job(message, move |c, p, m, supersampling_amount, coloring_function| {
                rendering::translate_to_center_and_render_efficiently(c, p, m, &complex, supersampling_amount, coloring_function);
            });
        } else {
            self.log(format!("Clicked pixel ({}, {}) = complex point ({:.10}, {:.10})", x as usize, y as usize, complex.x, complex.y));
        }
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_outdated && self.texture.is_some() {
            return;
        }
        let mut rgb: Vec<u8> = Vec::with_capacity(self.p.pixels.len() * 3);
        for pixel in &self.p.pixels {
            rgb.extend_from_slice(&[(pixel >> 16) as u8, (pixel >> 8) as u8, *pixel as u8]);
        }
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
            let response = ui.add(egui::DragValue::new(&mut self.max_iterations_input).range(1..=1_000_000).speed(10.0));
            if (response.drag_stopped() || response.lost_focus()) && self.max_iterations_input != self.m.max_iterations {
                self.m.max_iterations = self.max_iterations_input;
                self.render(&format!("Set max. iterations to {}", self.m.max_iterations));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Quality (supersampling, 1x-64x):");
            let response = ui.add(egui::DragValue::new(&mut self.supersampling_input).range(1..=64).suffix("x"));
            if (response.drag_stopped() || response.lost_focus()) && self.supersampling_input != self.supersampling_amount {
                self.supersampling_amount = self.supersampling_input;
                self.image_supersampling_amount = self.supersampling_input;
                self.render(&format!("Set the supersampling to {}x", self.supersampling_amount));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Saved image quality:");
            ui.add(egui::DragValue::new(&mut self.image_supersampling_amount).range(1..=64).suffix("x"));
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
        if ui.button("Save image").clicked() {
            self.save_image();
        }
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
        let current_fraction = bars.rows_done.load(Ordering::Relaxed) as f32 / bars.rows.max(1) as f32;
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

        //A move only redraws a strip, so the bars of the full render are removed
        app.translate("up", 10, 0);
        assert!(app.bars.is_none(), "the bars of the previous render should be removed when only a strip is rendered");
        wait_for_job(&mut app);
    }
}

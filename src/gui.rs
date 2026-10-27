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
use crate::rendering::ExposedRegion;
use crate::{rendering, ColoringFunction, Config, InteractionVariables, COLORING_FUNCTION, COLOR_CHANNEL_MAPPING};
use crate::points_of_interest::{self, Coloring, PointOfInterest};
use crate::thumbnails::{self, Lru, ThumbnailRequest};

/// Width in points of the right side panel, used to size the initial window
pub const SIDE_PANEL_WIDTH: f32 = 380.0;
/// Minimum height in points of the window, so that the controls and the status log have room
pub const MIN_WINDOW_HEIGHT: f32 = 700.0;

const COLORING_FUNCTIONS: [(&str, ColoringFunction); 2] = [("HSV", TrueColor::new_from_hsv_colors), ("Bernstein polynomials", TrueColor::new_from_bernstein_polynomials)];

const COLOR_CHANNEL_MAPPINGS: [&str; 27] = [
    "RGB", "RBG", "GRB", "GBR", "BRG", "BGR", "RRR", "GGG", "BBB", "RRG", "RRB", "RGG", "RGR", "RBB", "RBR", "GGB", "GGR", "GBB", "GBG", "GRG", "GRR", "BBG", "BBR", "BGB", "BGG", "BRB", "BRR",
];


/// A message from a rendering job that runs on a background thread
enum JobMessage {
    /// An intermediate image: the running average of the first `pass` of `total` supersampling layers
    Pass { colors: Vec<TrueColor>, pixels: Vec<u32>, pass: u8, total: u8, layer_time: Duration },
    /// The job has finished (or was cancelled), this is its final result
    Done { c: ComplexPlane, p: PixelBuffer, message: String, elapsed: Duration, completed: bool },
}

/// The thumbnail that is being rendered on a background thread
struct ThumbnailJob {
    id: u64,
    key: u64,
    cancel: Arc<AtomicBool>,
}

/// A finished thumbnail job: its id, its key, and the image (width, height, RGB bytes) if it was not cancelled
struct ThumbnailMessage {
    job: u64,
    key: u64,
    image: Option<(usize, usize, Vec<u8>)>,
}

/// What the list of points of interest showed in a frame
#[derive(Default)]
struct PointsFrame {
    /// The point that was clicked
    chosen: Option<usize>,
    /// The thumbnails of the rows that are visible: their keys, and for those that are not cached yet, what to render
    visible: Vec<u64>,
    missing: Vec<(u64, ThumbnailRequest)>,
}

/// The height of a row of the list of points of interest
const POINT_ROW_HEIGHT: f32 = 104.0;
/// How many rows the list shows without scrolling
const POINT_ROWS_SHOWN: f32 = 3.0;

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
    pub fn new(config: &Config) -> MandelbrotApp {
        let c = ComplexPlane::new(config.window_width, config.window_height);
        let mut p = PixelBuffer::new(PixelPlane::new(config.window_width, config.window_height));
        p.color_channel_mapping = COLOR_CHANNEL_MAPPING;
        let m = MandelbrotSet::new(config.max_iterations, config.orbit_radius);
        let (notice_sender, notices) = mpsc::channel();
        let (thumb_sender, thumb_results) = mpsc::channel();
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
        app.load_points_of_interest(points_of_interest::find_folder());
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
        self.cancel_thumbnail_job(); //The main view has priority over thumbnails
        self.interrupt_progressive_render();
        self.buffer_is_complete = false;
        let (sender, receiver) = mpsc::channel();
        let (c, mut p, m) = (self.c.clone(), self.p.clone(), self.m.clone());
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
            let region = match strip {
                Some((shift, region)) => {
                    p.translate_buffer(shift.0, shift.1);
                    region
                }
                None => ExposedRegion::full(width, height),
            };
            let base = p.colors.clone();
            let mut layer_started = started;
            let completed = rendering::render_region_progressively(&c, &m, width, height, &region, &base, supersampling_amount, coloring_function, &thread_cancel, &thread_pixels_done, |colors, pass| {
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
        self.bars = Some(RenderBars { total_layers: supersampling_amount.clamp(1, 64), layer_times: Vec::new(), started: now, layer_started: now, pixels_done, pixels_total, finished: None });
        self.job = Some(receiver);
        self.progress = Some(Progress { pass: 0, total: supersampling_amount, cancel });
        self.log("Rendering...");
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

    fn cancel_thumbnail_job(&mut self) {
        if let Some(job) = self.thumb_job.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The thumbnail request of a point: its location with the appearance that going to the point would use (the saved one when appearances are restored, otherwise the current one)
    fn thumbnail_request_for(point: &PointOfInterest, coloring_index: usize, restore_appearance: bool, mapping: ColorChannelMapping, window: (usize, usize)) -> ThumbnailRequest {
        let current = if coloring_index == 0 { Coloring::Hsv } else { Coloring::Bernstein }; //See COLORING_FUNCTIONS
        let saved = point.appearance.as_ref().filter(|_| restore_appearance);
        ThumbnailRequest {
            location: point.location.clone(),
            coloring: saved.and_then(|appearance| appearance.coloring).unwrap_or(current),
            color_channels: saved.and_then(|appearance| appearance.color_channels).unwrap_or(mapping),
            width: thumbnails::THUMBNAIL_WIDTH,
            height: thumbnails::thumbnail_height(window.0, window.1),
        }
    }

    /// Takes the thumbnails that were finished by the background thread
    fn poll_thumbnails(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.thumb_results.try_recv() {
            if self.thumb_job.as_ref().is_some_and(|job| job.id == message.job) {
                self.thumb_job = None;
            }
            if let Some((width, height, rgb)) = message.image {
                let texture = ctx.load_texture(format!("point-of-interest-{}", message.key), ColorImage::from_rgb([width, height], &rgb), TextureOptions::LINEAR);
                self.thumbs.insert(message.key, texture); //Drops the least recently shown thumbnail when there are too many
            }
        }
    }

    /// Decides which thumbnail to render next: only a visible one that is not cached, only one at a time, and never while the main view renders.
    /// A running thumbnail that is no longer visible, or that is in the way of the main view, is cancelled.
    fn schedule_thumbnails(&mut self, ctx: &egui::Context, frame: &PointsFrame) {
        if self.thumb_job.as_ref().is_some_and(|job| !frame.visible.contains(&job.key) || self.is_busy()) {
            self.cancel_thumbnail_job();
        }
        if !self.is_busy() && self.thumb_job.is_none() {
            if let Some((key, request)) = frame.missing.first().cloned() {
                self.thumb_job_counter += 1;
                self.thumbnails_started += 1;
                let (id, cancel) = (self.thumb_job_counter, Arc::new(AtomicBool::new(false)));
                self.thumb_job = Some(ThumbnailJob { id, key, cancel: Arc::clone(&cancel) });
                let (sender, orbit_radius) = (self.thumb_sender.clone(), self.config.orbit_radius);
                std::thread::spawn(move || {
                    let image = thumbnails::render_thumbnail(&request, orbit_radius, &cancel).map(|colors| {
                        let pixels = PixelBuffer::colors_to_pixels(&colors, &request.color_channels);
                        let rgb: Vec<u8> = pixels.iter().flat_map(|pixel| [(pixel >> 16) as u8, (pixel >> 8) as u8, *pixel as u8]).collect();
                        (request.width, request.height, rgb)
                    });
                    let _ = sender.send(ThumbnailMessage { job: id, key, image }); //The receiver is gone if the window was closed
                });
            }
        }
        if self.thumb_job.is_some() || !frame.missing.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(30)); //Keep polling for the thumbnail, and start the next one when the main view is idle
        }
    }

    /// Draws the list of points of interest with their thumbnails. Only the rows that are visible are drawn, and only their thumbnails are looked up or requested,
    /// so the work does not depend on how many points there are.
    fn points_ui(&mut self, ui: &mut egui::Ui) -> PointsFrame {
        let mut frame = PointsFrame::default();
        let (width, height) = (thumbnails::THUMBNAIL_WIDTH as f32, thumbnails::thumbnail_height(self.config.window_width, self.config.window_height) as f32);
        egui::ScrollArea::vertical().id_salt("points_of_interest_list").max_height(POINT_ROW_HEIGHT * POINT_ROWS_SHOWN).auto_shrink([false, true]).show_rows(ui, POINT_ROW_HEIGHT, self.points.len(), |ui, rows| {
            for index in rows {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), POINT_ROW_HEIGHT - 4.0), egui::Sense::click());
                let point = &self.points[index];
                let request = Self::thumbnail_request_for(point, self.coloring_index, self.restore_appearance, self.p.color_channel_mapping, (self.config.window_width, self.config.window_height));
                //Visible: at least a quarter of the row is inside the list and inside the scrolling controls around it (not clipped away)
                let shown = rect.intersect(ui.clip_rect());
                let visible = shown.is_positive() && shown.height() >= rect.height() * 0.25;
                let thumbnail_rect = egui::Rect::from_min_size(rect.min + egui::vec2(2.0, 2.0), egui::vec2(width, height));
                let painter = ui.painter_at(rect);
                if response.hovered() {
                    painter.rect_filled(rect, 4.0, ui.visuals().widgets.hovered.weak_bg_fill);
                }
                let key = request.key();
                if visible {
                    frame.visible.push(key);
                    match self.thumbs.get(key) {
                        Some(texture) => {
                            painter.image(texture.id(), thumbnail_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
                        }
                        None => {
                            painter.rect_filled(thumbnail_rect, 2.0, Color32::from_gray(30));
                            painter.text(thumbnail_rect.center(), egui::Align2::CENTER_CENTER, "...", egui::FontId::proportional(14.0), Color32::GRAY);
                            frame.missing.push((key, request));
                        }
                    }
                }
                painter.text(egui::pos2(thumbnail_rect.right() + 8.0, thumbnail_rect.top() + 2.0), egui::Align2::LEFT_TOP, &point.name, egui::FontId::proportional(14.0), ui.visuals().strong_text_color());
                let response = if point.description.is_empty() { response } else { response.on_hover_text(&point.description) };
                if response.clicked() {
                    frame.chosen = Some(index);
                }
            }
        });
        frame
    }

    /// Loads the points of interest of `folder`, and logs how many were found and every problem (a skipped file, an ignored appearance)
    fn load_points_of_interest(&mut self, folder: std::path::PathBuf) {
        let report = points_of_interest::load_folder(&folder);
        self.log(format!("Points of interest: {} loaded from {}", report.points.len(), folder.display()));
        for problem in report.problems {
            self.log(format!("Points of interest: {problem}"));
        }
        self.points = report.points;
        self.points_folder = folder;
    }

    /// Goes to the point of interest at `index`: its center, scale and maximum iterations are applied, and with `restore_appearance` also the coloring and color channels saved in its file.
    /// Otherwise the current appearance, and the supersampling, stay as they are.
    fn go_to_point_of_interest(&mut self, index: usize) {
        let Some(point) = self.points.get(index).cloned() else { return };
        let location = &point.location;
        self.c.set_view(&View::new(location.real, location.imaginary, location.scale));
        self.m.max_iterations = location.max_iterations;
        self.max_iterations_input = location.max_iterations;
        let mut restored = String::new();
        if let (true, Some(appearance)) = (self.restore_appearance, &point.appearance) {
            if let Some(coloring) = appearance.coloring {
                self.coloring_index = match coloring {
                    Coloring::Hsv => 0,
                    Coloring::Bernstein => 1,
                }; //See COLORING_FUNCTIONS
                self.coloring_function = COLORING_FUNCTIONS[self.coloring_index].1;
                restored += &format!(", coloring {}", COLORING_FUNCTIONS[self.coloring_index].0);
            }
            if let Some(mapping) = appearance.color_channels {
                self.p.color_channel_mapping = mapping;
                restored += &format!(", color channels {mapping:?}");
            }
        }
        let message = if restored.is_empty() { format!("Moved to {}", point.name) } else { format!("Moved to {} and restored its appearance{restored}", point.name) };
        self.render(&message);
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
            ui.label("Points of interest:");
            if ui.button("Reload").on_hover_text(format!("Loads the .json files of {} again", self.points_folder.display())).clicked() {
                self.load_points_of_interest(self.points_folder.clone());
            }
        });
        let frame = self.points_ui(ui);
        self.thumbs_visible = frame.visible.clone();
        let ctx = ui.ctx().clone();
        self.schedule_thumbnails(&ctx, &frame);
        if let Some(index) = frame.chosen {
            self.go_to_point_of_interest(index);
        }
        ui.checkbox(&mut self.restore_appearance, "Restore the saved appearance (coloring, color channels)").on_hover_text("Off: going to a place keeps your current appearance. The supersampling is never part of a place.");
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
        let current_fraction = bars.pixels_done.load(Ordering::Relaxed) as f32 / bars.pixels_total.max(1) as f32;
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
        self.poll_thumbnails(&ctx);
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

    fn app_with_points(files: &[(&str, &str)]) -> (MandelbrotApp, std::path::PathBuf) {
        let config = Config::build(["mandelbrot", "64", "48", "50", "3"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        static NEXT_FOLDER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let folder = std::env::temp_dir().join(format!("mandelbrot-gui-poi-{}-{}", std::process::id(), NEXT_FOLDER.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        for (name, text) in files {
            std::fs::write(folder.join(name), text).unwrap();
        }
        app.load_points_of_interest(folder.clone());
        (app, folder)
    }

    const HSV_BGR_PLACE: &str = r#"{ "format_version": 1, "name": "Test Place", "location": { "center": { "real": -0.7498100000000001, "imaginary": -0.020300000000000054 }, "scale": 0.00006400000000000002, "max_iterations": 321 },
        "appearance": { "coloring": "hsv", "color_channels": "BGR" } }"#;

    #[test]
    fn the_application_lists_the_points_of_interest_of_the_folder_at_startup() {
        let config = Config::build(["mandelbrot", "64", "48", "50"].map(String::from).into_iter()).unwrap();
        let mut app = MandelbrotApp::new(&config);
        wait_for_job(&mut app);
        assert_eq!(app.points.len(), 10, "the ten former views are found without a list of file names in the code");
        assert!(app.points.iter().any(|point| point.name == "Sunlit Reef"), "the former View 10 (VIEW_0)");
        assert!(app.points.windows(2).all(|pair| pair[0].name.to_lowercase() <= pair[1].name.to_lowercase()), "listed by name");
        assert!(app.log.iter().any(|message| message.starts_with("Points of interest: 10 loaded from")));
        assert!(!app.restore_appearance);
    }

    #[test]
    fn going_to_a_point_applies_the_location_and_keeps_the_current_appearance_and_supersampling() {
        let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
        let (coloring_index, mapping, supersampling) = (app.coloring_index, app.p.color_channel_mapping, app.supersampling_amount);
        app.go_to_point_of_interest(0);
        wait_for_job(&mut app);
        let center = app.c.center();
        assert!((center.x - -0.7498100000000001).abs() < 1e-12 && (center.y - -0.020300000000000054).abs() < 1e-12, "{center:?}");
        assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18);
        assert_eq!((app.m.max_iterations, app.max_iterations_input), (321, 321));
        assert_eq!((app.coloring_index, app.p.color_channel_mapping, app.supersampling_amount), (coloring_index, mapping, supersampling), "the appearance is only restored on request");
        assert!(app.log.last().unwrap().starts_with("Moved to Test Place in"), "{:?}", app.log.last());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn restoring_the_appearance_applies_the_saved_coloring_and_channels_but_never_the_supersampling() {
        let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
        let supersampling = app.supersampling_amount;
        app.restore_appearance = true;
        app.go_to_point_of_interest(0);
        wait_for_job(&mut app);
        assert_eq!(app.coloring_index, 0, "HSV");
        assert_eq!(format!("{:?}", app.p.color_channel_mapping), "BGR");
        assert_eq!(app.supersampling_amount, supersampling);
        assert!(app.log.last().unwrap().contains("restored its appearance"), "{:?}", app.log.last());
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn a_malformed_file_is_reported_in_the_status_box_and_the_other_places_still_load() {
        let (app, folder) = app_with_points(&[("good.json", HSV_BGR_PLACE), ("bad.json", "{ this is not json"), ("zero.json", &HSV_BGR_PLACE.replace("0.00006400000000000002", "0"))]);
        assert_eq!(app.points.len(), 1);
        assert!(app.log.iter().any(|message| message.starts_with("Points of interest: bad.json")), "{:?}", app.log);
        assert!(app.log.iter().any(|message| message.starts_with("Points of interest: zero.json: the scale should be")));
        assert!(app.log.iter().any(|message| message.starts_with("Points of interest: 1 loaded from")));
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn reloading_finds_files_that_were_added_and_a_missing_folder_is_reported_not_fatal() {
        let (mut app, folder) = app_with_points(&[("good.json", HSV_BGR_PLACE)]);
        std::fs::write(folder.join("mine.json"), HSV_BGR_PLACE.replace("Test Place", "My Place")).unwrap();
        app.load_points_of_interest(folder.clone());
        assert_eq!(app.points.len(), 2, "a user's own file appears after reloading");
        let _ = std::fs::remove_dir_all(&folder);
        app.load_points_of_interest(folder);
        assert!(app.points.is_empty() && app.log.iter().any(|message| message.contains("Cannot read the points of interest folder")));
    }

    /// One frame of the controls (with the lists of points of interest) in a window of 600 x `height` points; finished thumbnails are taken first, like `eframe::App::ui` does
    fn controls_frame(app: &mut MandelbrotApp, ctx: &egui::Context, height: f32) {
        app.poll_thumbnails(ctx);
        let mut output = ctx.run_ui(egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, height))), ..Default::default() }, |ui| app.controls_ui(ui));
        output.drop_without_applying_deltas();
    }

    /// Frames until every visible thumbnail is cached (and no job runs), with a time limit
    fn frames_until_thumbnails_are_done(app: &mut MandelbrotApp, ctx: &egui::Context, height: f32) {
        let start = std::time::Instant::now();
        loop {
            controls_frame(app, ctx, height);
            if app.thumb_job.is_none() && !app.thumbs_visible.is_empty() && app.thumbs_visible.iter().all(|key| app.thumbs.contains(*key)) {
                return;
            }
            assert!(start.elapsed() < Duration::from_secs(60), "the thumbnails were not finished");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// `count` places; with `slow` they are boundary views with an enormous iteration limit (capped by the thumbnails), which take a noticeable time to render
    fn many_points(count: usize, slow: bool) -> Vec<(String, String)> {
        (0..count).map(|i| {
            let text = HSV_BGR_PLACE.replace("Test Place", &format!("Place {i:03}"));
            let text = if slow { text.replace("\"max_iterations\": 321", "\"max_iterations\": 1000000") } else { text.replace("0.00006400000000000002", &format!("0.{:04}", 2 + i)).replace("\"max_iterations\": 321", "\"max_iterations\": 120") };
            (format!("place-{i:03}.json"), text)
        }).collect()
    }

    fn app_with_many_points(count: usize) -> (MandelbrotApp, std::path::PathBuf) {
        app_with_places(many_points(count, false))
    }

    fn app_with_places(files: Vec<(String, String)>) -> (MandelbrotApp, std::path::PathBuf) {
        let borrowed: Vec<(&str, &str)> = files.iter().map(|(name, text)| (name.as_str(), text.as_str())).collect();
        app_with_points(&borrowed)
    }

    #[test]
    fn only_the_visible_thumbnails_are_rendered_however_many_points_there_are() {
        let (mut app, folder) = app_with_many_points(25);
        assert_eq!(app.points.len(), 25);
        let ctx = egui::Context::default();
        frames_until_thumbnails_are_done(&mut app, &ctx, 1000.0);
        let visible = app.thumbs_visible.len();
        assert!((1..=5).contains(&visible), "a list of three rows shows a few of the 25 points: {visible}");
        assert_eq!(app.thumbnails_started, visible as u64, "exactly the visible thumbnails were rendered, each once");
        assert_eq!(app.thumbs.len(), visible);
        //More frames do not render anything more
        for _ in 0..5 {
            controls_frame(&mut app, &ctx, 1000.0);
        }
        assert_eq!(app.thumbnails_started, visible as u64);
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn nothing_is_rendered_while_the_list_is_scrolled_or_clipped_out_of_view() {
        let (mut app, folder) = app_with_many_points(25);
        let ctx = egui::Context::default();
        for _ in 0..10 {
            controls_frame(&mut app, &ctx, 100.0); //The list is below the bottom of this window
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(app.thumbs_visible.is_empty());
        assert_eq!((app.thumbnails_started, app.thumbs.len()), (0, 0));
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn a_running_thumbnail_is_cancelled_when_it_leaves_the_view_and_when_the_main_view_renders() {
        let (mut app, folder) = app_with_places(many_points(25, true));
        let ctx = egui::Context::default();
        controls_frame(&mut app, &ctx, 1000.0);
        let job = app.thumb_job.as_ref().expect("a visible thumbnail is rendered when the window is idle");
        let first_cancel = Arc::clone(&job.cancel);
        //The list goes out of view: the thumbnail is cancelled
        controls_frame(&mut app, &ctx, 100.0);
        assert!(first_cancel.load(Ordering::Relaxed) && app.thumb_job.is_none());

        //A thumbnail runs, then the main view renders: the thumbnail is cancelled and no other one starts while the main view renders
        std::thread::sleep(Duration::from_millis(100));
        controls_frame(&mut app, &ctx, 1000.0);
        let started = app.thumbnails_started;
        let second_cancel = Arc::clone(&app.thumb_job.as_ref().map(|job| &job.cancel).expect("a thumbnail job runs again"));
        app.render("Rendering the main view");
        assert!(second_cancel.load(Ordering::Relaxed) && app.thumb_job.is_none(), "the main view has priority");
        for _ in 0..10 {
            controls_frame(&mut app, &ctx, 1000.0);
            if !app.is_busy() {
                break;
            }
            assert_eq!(app.thumbnails_started, started, "no thumbnail starts while the main view renders");
            assert!(app.thumb_job.is_none());
        }
        //When the main view is done, the thumbnails continue
        wait_for_job(&mut app);
        frames_until_thumbnails_are_done(&mut app, &ctx, 1000.0);
        assert!(app.thumbnails_started > started);
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn the_resident_thumbnails_are_bounded_and_follow_the_appearance_that_going_to_a_place_would_use() {
        let (mut app, folder) = app_with_many_points(3);
        let ctx = egui::Context::default();
        for key in 0..300u64 {
            app.thumbs.insert(key, ctx.load_texture(format!("t{key}"), ColorImage::from_rgb([2, 2], &[0u8; 12]), TextureOptions::LINEAR));
            assert!(app.thumbs.len() <= thumbnails::MAX_CACHED);
        }
        //The thumbnail looks like the place does when going to it: with the current appearance, or the saved one when appearances are restored
        let point = app.points[0].clone();
        let current = MandelbrotApp::thumbnail_request_for(&point, app.coloring_index, false, app.p.color_channel_mapping, (64, 48));
        let restored = MandelbrotApp::thumbnail_request_for(&point, app.coloring_index, true, app.p.color_channel_mapping, (64, 48));
        assert_eq!(current.coloring, Coloring::Bernstein, "the current appearance of the window");
        assert_eq!(restored.coloring, Coloring::Hsv, "the saved appearance of the file");
        assert_ne!(current.key(), restored.key(), "another appearance is another thumbnail");
        assert_eq!(current.location, restored.location);
        assert_eq!(current.height, thumbnails::thumbnail_height(64, 48));
        let _ = std::fs::remove_dir_all(folder);
    }
}

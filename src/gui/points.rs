//! The list of points of interest of the window, with their thumbnails (see `crate::thumbnails`).

use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, TextureOptions};

use crate::coloring::ColorChannelMapping;
use crate::complex_plane::{ComplexPlane, View};
use crate::mandelbrot_set::ColorCycle;
use crate::pixel_buffer::PixelBuffer;
use crate::points_of_interest::{self, Coloring, PointOfInterest};
use crate::thumbnails::{self, ThumbnailRequest};
use super::flight::{Camera, Flight, Pace, Route};
use super::{MandelbrotApp, Readiness, COLORING_FUNCTIONS};

/// The most a flight that waits for its frames moves on after a frame, in flight time: two frames at 60 Hz (see `flight_job`)
pub(super) const MAX_WAITING_STEP: Duration = Duration::from_millis(33);

/// A flight to a point of interest that is under way: the path, how far along it the flight is, and the point to apply on arrival
pub(super) struct PointFlight {
    pub(super) flight: Flight,
    /// The time of the flight so far, and the frame it was last moved in: the flight clock follows the frames of the window
    pub(super) elapsed: Duration,
    pub(super) last_frame: Instant,
    pub(super) point: PointOfInterest,
    /// With auto iterations: the limits of the frames on the way
    pub(super) limits: Option<FlightLimits>,
    /// When the flight started, and when each of its frames was shown (the first image of a view on the way that is complete enough for the flight rendering), for the frame rate, see `fps`
    pub(super) started: Instant,
    pub(super) frames: Vec<Instant>,
    /// The frame rate in the progress bar, and when it was last updated: once a second, so it can be read (see `update_fps`)
    pub(super) fps_shown: usize,
    pub(super) fps_updated: Instant,
    /// The color cycle where the flight starts and at the place: with "Restore the saved appearance", the place's divide and offset; the frames on the way blend
    /// them by the share of the path, so the colors change gradually and only on the way (see `palette_at`)
    pub(super) palettes: (ColorCycle, ColorCycle),
}

impl PointFlight {
    /// The color cycle of the frame at `elapsed`, see `palettes`
    pub(super) fn palette_at(&self, elapsed: Duration) -> ColorCycle {
        self.palettes.0.blend(self.palettes.1, self.flight.share_of_path(elapsed))
    }

    /// The frames shown in the second before `now`
    pub(super) fn fps(&self, now: Instant) -> usize {
        self.frames.iter().rev().take_while(|shown| now.saturating_duration_since(**shown) <= Duration::from_secs(1)).count()
    }

    /// Updates the frame rate in the progress bar to `fps(now)` when it is a second old
    pub(super) fn update_fps(&mut self, now: Instant) {
        if now.saturating_duration_since(self.fps_updated) >= Duration::from_secs(1) {
            (self.fps_shown, self.fps_updated) = (self.fps(now), now);
        }
    }

    /// The frames shown so far, the time since the start, and the average frame rate, for the status log
    fn summary(&self, now: Instant) -> String {
        let seconds = now.saturating_duration_since(self.started).as_secs_f64();
        format!("{} frames in {seconds:.1} s, {:.1} FPS on average", self.frames.len(), self.frames.len() as f64 / seconds.max(1e-9))
    }
}

/// The iteration limits of a flight with auto iterations: for every frame the zoom rule of the preset (`AutoIterationSettings::initial_limit`) times a factor.
///
/// The zoom rule is only the starting value of auto iterations: at a deep place the probes often raise the limit by a factor 10 or more. So the factor is what auto
/// iterations chose at the start of the flight (its limit over the zoom rule there), and what the probes choose for the destination (run in the background when
/// the flight starts), blended by the zoom between the two. A frame at the zoom of the start or the destination gets exactly that limit, so the arrival does not jump.
pub(super) struct FlightLimits {
    /// The scale and the factor at the start
    start: (f64, f64),
    /// The scale of the destination, and its factor once its probes are done
    destination_scale: f64,
    destination_factor: Option<f64>,
    /// The limit that the probes choose for the destination, when they are done
    pending: Option<std::sync::mpsc::Receiver<u32>>,
}

impl FlightLimits {
    /// The limit for a frame at `scale`, with the zoom rule of `settings`
    pub(super) fn limit_at(&mut self, scale: f64, settings: &crate::auto_iterations::AutoIterationSettings) -> u32 {
        if let Some(limit) = self.pending.as_ref().and_then(|receiver| receiver.try_recv().ok()) {
            self.destination_factor = Some(f64::from(limit) / f64::from(settings.initial_limit(self.destination_scale)));
            self.pending = None;
        }
        let (start_scale, start_factor) = self.start;
        let factor = match self.destination_factor {
            Some(destination_factor) => {
                //How far the zoom is from the start to the destination (0 to 1, on a logarithmic scale): geometric between the two factors
                let span = (self.destination_scale / start_scale).ln();
                let t = if span.abs() < 1e-12 { 1.0 } else { ((scale / start_scale).ln() / span).clamp(0.0, 1.0) };
                start_factor * (destination_factor / start_factor).powf(t)
            }
            None => start_factor,
        };
        (f64::from(settings.initial_limit(scale)) * factor).round().clamp(1.0, f64::from(u32::MAX)) as u32
    }

    /// Waits until the probes of the destination are done (in tests)
    #[cfg(test)]
    pub(super) fn wait_for_destination(&mut self) {
        if let Some(limit) = self.pending.as_ref().and_then(|receiver| receiver.recv().ok()) {
            //Put it back for the next frame to take
            let (sender, receiver) = std::sync::mpsc::channel();
            let _ = sender.send(limit);
            self.pending = Some(receiver);
        }
    }
}


/// The colors of depth, from the shallowest place of the list to the deepest: cool to hot, so a glance tells shallow (fast) from deep (slow)
const DEPTH_COLORS: [(u8, u8, u8); 4] = [(80, 150, 255), (60, 205, 170), (255, 190, 60), (255, 85, 75)];

/// How deep the places of the list are compared with each other: the zoom of every place on a log scale between the shallowest and the deepest place
pub(super) struct DepthScale {
    shallowest: f64,
    deepest: f64,
}

impl DepthScale {
    /// The scale of the places `points`
    pub(super) fn of(points: &[PointOfInterest]) -> DepthScale {
        let depths = points.iter().map(|point| -point.location.scale.ln());
        DepthScale { shallowest: depths.clone().fold(f64::INFINITY, f64::min), deepest: depths.fold(f64::NEG_INFINITY, f64::max) }
    }

    /// The color of the depth of `point`: the first of `DEPTH_COLORS` for the shallowest place (and when all places are equally deep), the last for the deepest,
    /// and a blend of the two nearest colors in between
    pub(super) fn color(&self, point: &PointOfInterest) -> Color32 {
        let range = self.deepest - self.shallowest;
        let t = if range > 0.0 { ((-point.location.scale.ln() - self.shallowest) / range).clamp(0.0, 1.0) } else { 0.0 };
        let at = t * (DEPTH_COLORS.len() - 1) as f64;
        let (index, weight) = ((at.floor() as usize).min(DEPTH_COLORS.len() - 2), at - at.floor().min((DEPTH_COLORS.len() - 2) as f64));
        let (from, to) = (DEPTH_COLORS[index], DEPTH_COLORS[index + 1]);
        let mix = |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * weight).round() as u8;
        Color32::from_rgb(mix(from.0, to.0), mix(from.1, to.1), mix(from.2, to.2))
    }
}

/// Where the points of interest are on the screen when the view `plane` fills `image_rect`: the name, the position and the depth color (see `DepthScale`) of every
/// point inside the image
pub(super) fn marker_positions(points: &[PointOfInterest], plane: &ComplexPlane, image_rect: egui::Rect) -> Vec<(String, egui::Pos2, Color32)> {
    let (width, height) = ((plane.length_x / plane.increment_x).round(), (plane.length_y / plane.increment_y).round());
    let depth = DepthScale::of(points);
    points.iter().filter_map(|point| {
        //The pixel of the point (see `ComplexPlane::complex_from_pixel_plane`: the screen y is the negative of the imaginary part)
        let x = (point.location.real - plane.min_x) / plane.increment_x;
        let y = (-point.location.imaginary - plane.min_y) / plane.increment_y;
        let position = image_rect.lerp_inside(egui::vec2((x / width) as f32, (y / height) as f32));
        image_rect.contains(position).then(|| (point.name.clone(), position, depth.color(point)))
    }).collect()
}

/// Draws the markers of `marker_positions`: small, semi-transparent circles with a soft glow in the color of their depth and the names next to them, subtle so they
/// do not hide the fractal
pub(super) fn draw_markers(painter: &egui::Painter, markers: &[(String, egui::Pos2, Color32)]) {
    for (name, position, color) in markers {
        let glow = |alpha: u8| Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
        //The glow: wide, faint circles around a small core
        for (radius, alpha) in [(10.0, 12), (7.0, 22), (5.0, 36)] {
            painter.circle_filled(*position, radius, glow(alpha));
        }
        painter.circle_filled(*position, 3.0, glow(210));
        painter.circle_stroke(*position, 3.0, egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(0, 0, 0, 90)));
        //The name, with a faint shadow so it can be read on light and dark parts of the fractal
        let (anchor, font) = (*position + egui::vec2(9.0, -1.0), egui::FontId::proportional(12.0));
        painter.text(anchor + egui::vec2(1.0, 1.0), egui::Align2::LEFT_CENTER, name, font.clone(), Color32::from_rgba_unmultiplied(0, 0, 0, 140));
        painter.text(anchor, egui::Align2::LEFT_CENTER, name, font, Color32::from_rgba_unmultiplied(255, 255, 255, 190));
    }
}

/// The zoom of a point of interest as the list shows it, like the zoom of the current view (see `status::format_zoom`)
pub(super) fn point_zoom_text(point: &PointOfInterest) -> String {
    format!("Zoom {}", super::status::format_zoom(1.0 / point.location.scale))
}

/// The thumbnail that is being rendered on a background thread
pub(super) struct ThumbnailJob {
    pub(super) id: u64,
    pub(super) key: u64,
    pub(super) cancel: Arc<AtomicBool>,
}

/// A finished thumbnail job: its id, its key, and the image (width, height, RGB bytes) if it was not cancelled
pub(super) struct ThumbnailMessage {
    pub(super) job: u64,
    pub(super) key: u64,
    pub(super) image: Option<(usize, usize, Vec<u8>)>,
}

/// What the list of points of interest showed in a frame
#[derive(Default)]
pub(super) struct PointsFrame {
    /// The point that was clicked
    pub(super) chosen: Option<usize>,
    /// The thumbnails of the rows that are visible: their keys, and for those that are not cached yet, what to render
    pub(super) visible: Vec<u64>,
    pub(super) missing: Vec<(u64, ThumbnailRequest)>,
}

/// The height of a row of the list of points of interest
pub(super) const POINT_ROW_HEIGHT: f32 = 104.0;
/// How many rows the list shows without scrolling
pub(super) const POINT_ROWS_SHOWN: f32 = 3.0;

impl MandelbrotApp {
    /// Cancels the thumbnail that is being rendered, if any (a render of the main view has priority, and a thumbnail that left the view is not needed)
    pub(super) fn cancel_thumbnail_job(&mut self) {
        if let Some(job) = self.thumb_job.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The thumbnail request of a point: its location with the appearance that going to the point would use (the saved one when appearances are restored, otherwise the current one)
    pub(super) fn thumbnail_request_for(point: &PointOfInterest, coloring_index: usize, restore_appearance: bool, mapping: ColorChannelMapping, palette: ColorCycle, window: (usize, usize)) -> ThumbnailRequest {
        let current = if coloring_index == 0 { Coloring::Hsv } else { Coloring::Bernstein }; //See COLORING_FUNCTIONS
        let saved = point.appearance.as_ref().filter(|_| restore_appearance);
        ThumbnailRequest {
            location: point.location.clone(),
            coloring: saved.and_then(|appearance| appearance.coloring).unwrap_or(current),
            color_channels: saved.and_then(|appearance| appearance.color_channels).unwrap_or(mapping),
            palette: saved.map_or(palette, |appearance| appearance.palette_over(palette)),
            width: thumbnails::THUMBNAIL_WIDTH,
            height: thumbnails::thumbnail_height(window.0, window.1),
        }
    }

    /// Takes the thumbnails that were finished by the background thread
    pub(super) fn poll_thumbnails(&mut self, ctx: &egui::Context) {
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
    pub(super) fn schedule_thumbnails(&mut self, ctx: &egui::Context, frame: &PointsFrame) {
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
    pub(super) fn points_ui(&mut self, ui: &mut egui::Ui) -> PointsFrame {
        let mut frame = PointsFrame::default();
        let depth = DepthScale::of(&self.points);
        let (width, height) = (thumbnails::THUMBNAIL_WIDTH as f32, thumbnails::thumbnail_height(self.p.pixel_plane.width, self.p.pixel_plane.height) as f32);
        egui::ScrollArea::vertical().id_salt("points_of_interest_list").max_height(POINT_ROW_HEIGHT * POINT_ROWS_SHOWN).auto_shrink([false, true]).show_rows(ui, POINT_ROW_HEIGHT, self.points.len(), |ui, rows| {
            for index in rows {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), POINT_ROW_HEIGHT - 4.0), egui::Sense::click());
                let point = &self.points[index];
                let request = Self::thumbnail_request_for(point, self.coloring_index, self.restore_appearance, self.p.color_channel_mapping, self.m.palette, (self.p.pixel_plane.width, self.p.pixel_plane.height));
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
                //The zoom of the place, before it is clicked: a deep place can take long to render. Its color is its depth among the places, see `DepthScale`
                painter.text(egui::pos2(thumbnail_rect.right() + 8.0, thumbnail_rect.top() + 22.0), egui::Align2::LEFT_TOP, point_zoom_text(point), egui::FontId::proportional(12.0), depth.color(point));
                let response = if point.description.is_empty() { response } else { response.on_hover_text(&point.description) };
                if response.clicked() {
                    frame.chosen = Some(index);
                }
            }
        });
        frame
    }

    /// Loads the points of interest of `folder`, and logs how many were found and every problem (a skipped file, an ignored appearance)
    pub(super) fn load_points_of_interest(&mut self, folder: std::path::PathBuf) {
        let report = points_of_interest::load_folder(&folder);
        self.log(format!("Points of interest: {} loaded from {}", report.points.len(), folder.display()));
        for problem in report.problems {
            self.log(format!("Points of interest: {problem}"));
        }
        self.points = report.points;
        self.points_folder = folder;
    }

    /// Saves the current view as a new point of interest with the name and description of the form (see `controls::new_point_form`), in the folder of the list,
    /// and loads the list again so it shows up (last). The location is the view, the iterations of its render, and the appearance as it is now.
    pub(super) fn save_new_point(&mut self) {
        let Some((name, description)) = self.new_point.take() else { return };
        let name = name.trim().to_string();
        if name.is_empty() {
            self.new_point = Some((name, description));
            return;
        }
        let center = self.c.center();
        let coloring = if self.coloring_index == 0 { Coloring::Hsv } else { Coloring::Bernstein }; //See COLORING_FUNCTIONS
        let point = PointOfInterest {
            name: name.clone(),
            description: description.trim().to_string(),
            location: points_of_interest::Location { real: center.x, imaginary: center.y, scale: self.c.get_scale(), max_iterations: self.m.max_iterations },
            appearance: Some(points_of_interest::Appearance {
                coloring: Some(coloring),
                color_channels: Some(self.p.color_channel_mapping),
                iteration_divide: Some(self.m.palette.divide),
                color_offset: Some(self.m.palette.offset),
                stretch_palette: Some(self.m.palette.stretch),
            }),
            file_name: String::new(),
        };
        match points_of_interest::save_new(&self.points_folder, &point) {
            Ok(file_name) => {
                self.log(format!("Saved the view as the point of interest {name} ({file_name})"));
                self.load_points_of_interest(self.points_folder.clone());
            }
            Err(err) => {
                self.log(format!("Could not save the point of interest {name} in {}: {err}", self.points_folder.display()));
                self.new_point = Some((name, description)); //Keep what was typed
            }
        }
    }

    /// Flies to the point of interest at `index` by the shortest path at the chosen pace (see `flight`); `fly` moves the view on every frame and arrives with `arrive_at`.
    /// With no flight (`Pace::NoFlight`) the place is shown at once.
    pub(super) fn go_to_point_of_interest(&mut self, index: usize) {
        let Some(point) = self.points.get(index).cloned() else { return };
        let center = self.c.center();
        let from = Camera { x: center.x, y: center.y, scale: self.c.get_scale() };
        let to = Camera { x: point.location.real, y: point.location.imaginary, scale: point.location.scale };
        let mut destination = self.c.clone();
        destination.set_view(&View::new(to.x, to.y, to.scale));
        if self.zoom_refused(&destination) {
            return; //Deeper than the zoom limit: no flight
        }
        if self.pace == Pace::NoFlight {
            self.arrive_at(&point);
            return;
        }
        let flight = Flight::with(from, to, self.pace, self.route, self.constant_speed);
        let route = if self.route == Route::Shortest { String::new() } else { format!(", {} route", self.route.name().to_lowercase()) };
        self.log(format!("Flying to {} ({}{route}, {:.1} s)", point.name, self.pace.name(), flight.duration().as_secs_f64()));
        let limits = self.auto_iterations.then(|| self.flight_limits(to));
        let now = Instant::now();
        let destination_palette = point.appearance.as_ref().filter(|_| self.restore_appearance).map_or(self.m.palette, |appearance| appearance.palette_over(self.m.palette));
        let palettes = (self.m.palette, destination_palette);
        self.last_flight = Some(super::video::VideoFlight { from, to, pace: self.pace, route: self.route, constant_speed: self.constant_speed, name: point.name.clone(), limits: (self.m.max_iterations, point.location.max_iterations), palettes });
        self.flight = Some(PointFlight { flight, elapsed: Duration::ZERO, last_frame: now, point, limits, started: now, frames: Vec::new(), fps_shown: 0, fps_updated: now, palettes });
        self.frame_ready = true; //The view on the screen is where the flight starts
    }

    /// The limits of a flight to `destination` with auto iterations, see `FlightLimits`: the factor of the current view, and the probes of the destination started in the background
    /// (on half the render threads, so the first frames of the flight still render), exactly as the render on arrival runs them
    fn flight_limits(&self, destination: Camera) -> FlightLimits {
        let start_scale = self.c.get_scale();
        let start = (start_scale, f64::from(self.m.max_iterations) / f64::from(self.auto_settings.initial_limit(start_scale)));
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let mut plane = ComplexPlane::new(width, height);
        plane.set_view(&destination.view());
        let (settings, mut m) = (self.auto_settings.clone(), self.m.clone());
        let destination_scale = plane.get_scale();
        let initial = settings.initial_limit(destination_scale);
        m.max_iterations = initial;
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let threads = (crate::rendering::render_thread_count() / 2).max(1);
            if let Some(result) = crate::auto_iterations::choose_limit(&plane, &m, width, height, &settings, threads, &AtomicBool::new(false)) {
                let _ = sender.send(result.limit); //The flight may be over already
            }
        });
        FlightLimits { start, destination_scale, destination_factor: None, pending: Some(receiver) }
    }

    /// Stops the flight where it is (the Cancel button): the view stays, gets a full-quality render, and the controls are free again
    pub(super) fn cancel_flight(&mut self) {
        if let Some(flight) = self.flight.take() {
            self.log(format!("The flight to {} stopped: {}", flight.point.name, flight.summary(Instant::now())));
            self.render(&format!("Stopped the flight to {}", flight.point.name));
        }
    }

    /// Ends the flight at its destination at once (the Skip button)
    pub(super) fn skip_flight(&mut self) {
        if let Some(flight) = self.flight.take() {
            self.log(format!("The flight to {} was skipped: {}", flight.point.name, flight.summary(Instant::now())));
            self.arrive_at(&flight.point);
        }
    }

    /// Moves the view along the flight to a point of interest, if one is under way, and asks for the next frame
    pub(super) fn fly(&mut self, ctx: &egui::Context) {
        self.frame_interval = Duration::from_secs_f32(ctx.input(|input| input.predicted_dt).clamp(1.0 / 500.0, 0.1));
        self.fly_at(Instant::now());
        if self.flight.is_some() {
            ctx.request_repaint();
        }
    }

    /// The frame of the flight at `now`: a render of the view on the way (with the steady limit of a gesture, towards the full quality until the next frame interrupts it), or the point itself on arrival
    ///
    /// With a flight rendering that waits for frames (or immersion), a job renders the frames one after the other (see `flight_job`): the camera only moves on once
    /// the frame of its current view is complete enough, and then by at most `MAX_WAITING_STEP`, so a slow view makes a slow, smooth crawl instead of jumps.
    pub(super) fn fly_at(&mut self, now: Instant) {
        //A refresh of the screen (this runs once per frame of the window): the frame on it has stayed one more refresh (frames ahead, see `render_job::next_job_message`)
        self.frame_hold = self.frame_hold.saturating_sub(1);
        let Some(mut flight) = self.flight.take() else { return };
        flight.update_fps(now);
        //A flight that waits for its frames is rendered by its own job, which moves the flight clock on (see `flight_job`); the window arrives when its last frame is there
        if self.flight_readiness() != Readiness::Any {
            if flight.flight.arrived(flight.elapsed) {
                self.log(format!("The flight to {}: {}", flight.point.name, flight.summary(now)));
                self.arrive_at(&flight.point);
                return;
            }
            let running = self.flight_job && self.is_busy();
            self.flight = Some(flight);
            if !running {
                self.start_flight_job(self.frame_interval);
            }
            return;
        }
        flight.elapsed += now.saturating_duration_since(flight.last_frame);
        flight.last_frame = now;
        let elapsed = flight.elapsed;
        if flight.flight.arrived(elapsed) {
            self.log(format!("The flight to {}: {}", flight.point.name, flight.summary(now)));
            self.arrive_at(&flight.point);
            return;
        }
        self.c.set_view(&flight.flight.camera_at(elapsed).view());
        if self.auto_iterations {
            //Auto iterations follow the zoom all the way: the limit of this frame (see `FlightLimits`)
            let scale = self.c.get_scale();
            let limit = match &mut flight.limits {
                Some(limits) => limits.limit_at(scale, &self.auto_settings),
                None => self.auto_settings.initial_limit(scale),
            };
            self.m.max_iterations = limit;
        }
        //The color cycle on the way, from the start to the place's (see `PointFlight::palettes`)
        self.m.palette = flight.palette_at(elapsed);
        //Best effort: a frame whose render has not shown anything yet is not interrupted, or on an expensive view almost no render would ever be shown. The camera
        //moves on anyway (the image on the screen follows it, see `transform`), and the next render starts at the latest camera as soon as this one has shown a frame.
        if self.is_busy() && !self.frame_accepted {
            self.flight = Some(flight);
            return;
        }
        //Every frame renders towards the full quality: the next frame interrupts it when the camera moves on, so a frame gets as many layers as the time allows
        self.start_render_as(&format!("Flying to {}", flight.point.name), super::render_job::RenderKind::FlightFrame);
        self.flight = Some(flight);
    }

    /// Goes to `point`: its center, scale and maximum iterations are applied, and with `restore_appearance` also the coloring, color channels and color cycle saved in its file.
    /// Otherwise the current appearance, and the supersampling, stay as they are.
    fn arrive_at(&mut self, point: &PointOfInterest) {
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
                let _ = write!(restored, ", coloring {}", COLORING_FUNCTIONS[self.coloring_index].0);
            }
            if let Some(mapping) = appearance.color_channels {
                self.p.color_channel_mapping = mapping;
                let _ = write!(restored, ", color channels {mapping:?}");
            }
            let palette = appearance.palette_over(self.m.palette);
            if palette != self.m.palette {
                self.m.palette = palette;
                let _ = write!(restored, ", colors {}", palette_description(palette));
            }
        }
        let message = if restored.is_empty() { format!("Moved to {}", point.name) } else { format!("Moved to {} and restored its appearance{restored}", point.name) };
        self.render(&message);
    }
}

/// How a color cycle is described in the log: "divide 4, offset 120", or "stretched over the limit"
pub(super) fn palette_description(palette: ColorCycle) -> String {
    if palette.stretch { String::from("stretched over the limit") } else { format!("divide {}, offset {:.0}", palette.divide, palette.offset) }
}

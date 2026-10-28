//! The list of points of interest of the window, with their thumbnails (see `crate::thumbnails`).

use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, TextureOptions};

use crate::coloring::ColorChannelMapping;
use crate::complex_plane::{ComplexPlane, View};
use crate::pixel_buffer::PixelBuffer;
use crate::points_of_interest::{self, Coloring, PointOfInterest};
use crate::thumbnails::{self, ThumbnailRequest};
use super::flight::{Camera, Flight};
use super::{MandelbrotApp, Readiness, COLORING_FUNCTIONS};

/// The most a flight that waits for its frames moves on after a frame, in flight time: two frames at 60 Hz
const MAX_WAITING_STEP: Duration = Duration::from_millis(33);

/// A flight to a point of interest that is under way: the path, how far along it the flight is, and the point to apply on arrival
pub(super) struct PointFlight {
    pub(super) flight: Flight,
    /// The time of the flight so far, and the frame it was last moved in: the flight clock follows the frames of the window
    pub(super) elapsed: Duration,
    pub(super) last_frame: Instant,
    pub(super) point: PointOfInterest,
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
    pub(super) fn thumbnail_request_for(point: &PointOfInterest, coloring_index: usize, restore_appearance: bool, mapping: ColorChannelMapping, window: (usize, usize)) -> ThumbnailRequest {
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
        let (width, height) = (thumbnails::THUMBNAIL_WIDTH as f32, thumbnails::thumbnail_height(self.p.pixel_plane.width, self.p.pixel_plane.height) as f32);
        egui::ScrollArea::vertical().id_salt("points_of_interest_list").max_height(POINT_ROW_HEIGHT * POINT_ROWS_SHOWN).auto_shrink([false, true]).show_rows(ui, POINT_ROW_HEIGHT, self.points.len(), |ui, rows| {
            for index in rows {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), POINT_ROW_HEIGHT - 4.0), egui::Sense::click());
                let point = &self.points[index];
                let request = Self::thumbnail_request_for(point, self.coloring_index, self.restore_appearance, self.p.color_channel_mapping, (self.p.pixel_plane.width, self.p.pixel_plane.height));
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
    pub(super) fn load_points_of_interest(&mut self, folder: std::path::PathBuf) {
        let report = points_of_interest::load_folder(&folder);
        self.log(format!("Points of interest: {} loaded from {}", report.points.len(), folder.display()));
        for problem in report.problems {
            self.log(format!("Points of interest: {problem}"));
        }
        self.points = report.points;
        self.points_folder = folder;
    }

    /// Flies to the point of interest at `index`, by the chosen route at the chosen pace (see `flight`); `fly` moves the view on every frame and arrives with `arrive_at`
    pub(super) fn go_to_point_of_interest(&mut self, index: usize) {
        let Some(point) = self.points.get(index).cloned() else { return };
        let center = self.c.center();
        let from = Camera { x: center.x, y: center.y, scale: self.c.get_scale() };
        let to = Camera { x: point.location.real, y: point.location.imaginary, scale: point.location.scale };
        let reset_plane = ComplexPlane::new(self.p.pixel_plane.width, self.p.pixel_plane.height);
        let reset = Camera { x: reset_plane.center().x, y: reset_plane.center().y, scale: reset_plane.get_scale() };
        let flight = Flight::new(from, to, self.route, self.pace, reset);
        self.log(format!("Flying to {} ({}, {}, {:.1} s)", point.name, self.route.name(), self.pace.name(), flight.duration().as_secs_f64()));
        self.flight = Some(PointFlight { flight, elapsed: Duration::ZERO, last_frame: Instant::now(), point });
        self.frame_ready = true; //The view on the screen is where the flight starts
    }

    /// Stops the flight where it is (the Cancel button): the view stays, gets a full-quality render, and the controls are free again
    pub(super) fn cancel_flight(&mut self) {
        if let Some(flight) = self.flight.take() {
            self.render(&format!("Stopped the flight to {}", flight.point.name));
        }
    }

    /// Ends the flight at its destination at once (the Skip button)
    pub(super) fn skip_flight(&mut self) {
        if let Some(flight) = self.flight.take() {
            self.arrive_at(&flight.point);
        }
    }

    /// Moves the view along the flight to a point of interest, if one is under way, and asks for the next frame
    pub(super) fn fly(&mut self, ctx: &egui::Context) {
        self.fly_at(Instant::now());
        if self.flight.is_some() {
            ctx.request_repaint();
        }
    }

    /// The frame of the flight at `now`: a quick render of the view on the way (like a frame of a gesture), or the point itself on arrival
    ///
    /// With a flight rendering that waits for frames (or immersion), the camera only moves on once the frame of its current view is complete enough, and then by at
    /// most `MAX_WAITING_STEP`: the flight clock stands still while it waits, so a slow view makes a slow, smooth crawl instead of jumps.
    pub(super) fn fly_at(&mut self, now: Instant) {
        let Some(mut flight) = self.flight.take() else { return };
        let since_last_frame = now.saturating_duration_since(flight.last_frame);
        flight.last_frame = now;
        if self.flight_readiness() == Readiness::Any {
            flight.elapsed += since_last_frame;
        } else if self.frame_ready {
            flight.elapsed += since_last_frame.min(MAX_WAITING_STEP);
        } else {
            self.flight = Some(flight); //Waits for the frame
            return;
        }
        let elapsed = flight.elapsed;
        if flight.flight.arrived(elapsed) {
            self.arrive_at(&flight.point);
            return;
        }
        self.c.set_view(&flight.flight.camera_at(elapsed).view());
        if self.auto_iterations {
            //Auto iterations follow the zoom all the way: the limit (and the palette) for the zoom of this frame, by the zoom rule of the preset; the probes, which
            //would cost a frame time of their own, refine it on arrival
            let limit = self.auto_settings.initial_limit(self.c.get_scale());
            (self.m.max_iterations, self.m.palette_scale) = (limit, Some(limit));
        }
        let all_layers = self.flight_rendering == super::flight::FlightRendering::SuperSampled;
        self.start_render_as(&format!("Flying to {}", flight.point.name), false, true, all_layers);
        self.flight = Some(flight);
    }

    /// Goes to `point`: its center, scale and maximum iterations are applied, and with `restore_appearance` also the coloring and color channels saved in its file.
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
        }
        let message = if restored.is_empty() { format!("Moved to {}", point.name) } else { format!("Moved to {} and restored its appearance{restored}", point.name) };
        self.render(&message);
    }
}

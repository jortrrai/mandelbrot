//! Rendering jobs: the background threads that render the view layer by layer, and the app state that follows them.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};


use crate::auto_iterations::{self};
use crate::coloring::TrueColor;
use crate::complex_plane::ComplexPlane;
use crate::mandelbrot_set::MandelbrotSet;
use crate::pixel_buffer::PixelBuffer;
use crate::rendering::{self, ExposedRegion};
use crate::rendering::Render;
use eframe::egui;

use super::MandelbrotApp;


/// A preview of a new view is only shown if the first layer is expected to take longer than this, about a frame: a faster first layer is shown directly, see `Render::preview`
const PREVIEW_WORTHWHILE: Duration = Duration::from_millis(16);

/// The full-quality render after a drag or a wheel zoom starts when the gesture stops, or when it has not changed the view for this long (a mouse button held still)
const GESTURE_PAUSE: Duration = Duration::from_millis(150);

/// A message from a rendering job that runs on a background thread
pub(super) enum JobMessage {
    /// An intermediate image: the running average of the first `pass` of `total` supersampling layers, or a low-resolution preview of the first layer when `pass` is 0
    Pass { colors: Vec<TrueColor>, plane: ComplexPlane, pass: u8, total: u8, layer_time: Duration },
    /// The iteration limit that auto iterations chose for the render, sent as soon as it is known, before the first layer is rendered
    Limit { max_iterations: u32, palette_scale: u32 },
    /// The job has finished (or was cancelled), this is its final result
    Done { c: ComplexPlane, p: PixelBuffer, m: MandelbrotSet, message: String, elapsed: Duration, completed: bool },
}

/// What the render progress bar shows: the time of every finished layer and how far the current layer is.
/// It stays visible after the render has finished, showing the benchmark of that render.
pub(super) struct RenderBars {
    pub(super) total_layers: u8,
    /// How long every finished layer took
    pub(super) layer_times: Vec<Duration>,
    /// When the render started, and the current layer started, for the running timers
    pub(super) started: Instant,
    pub(super) layer_started: Instant,
    /// The pixels of the current layer that are done, and the amount of pixels in a layer
    pub(super) pixels_done: Arc<AtomicUsize>,
    pub(super) pixels_total: usize,
    /// How long the whole render took, and whether it was completed, once it has finished
    pub(super) finished: Option<(Duration, bool)>,
}

/// A progressive render that is currently running: how far it is and how to cancel it
pub(super) struct Progress {
    pub(super) pass: u8,
    pub(super) total: u8,
    pub(super) cancel: Arc<AtomicBool>,
}

impl MandelbrotApp {
    /// Cancels the running progressive render, if any, and ignores everything it still reports.
    /// The image of the last completed layer stays on screen until the next render shows its first layer.
    pub(super) fn interrupt_progressive_render(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.cancel.store(true, Ordering::Relaxed);
            self.job = None;
        }
    }

    /// Takes the messages of the background job: intermediate images and the final result
    pub(super) fn poll_job(&mut self) {
        while let Ok(notice) = self.notices.try_recv() {
            self.log(notice);
        }
        loop {
            let Some(receiver) = &self.job else { return };
            match receiver.try_recv() {
                Ok(JobMessage::Pass { colors, plane, pass, total, layer_time }) => {
                    self.p.colors = colors;
                    self.shown_plane = plane;
                    self.texture_outdated = true;
                    self.layer_shown = pass > 0;
                    //A preview of the whole view is uploaded small and scaled up smoothly; a preview of a strip sits next to sharp pixels, so it stays sharp
                    let whole_view = self.bars.as_ref().is_some_and(|bars| bars.pixels_total == self.p.pixel_plane.width * self.p.pixel_plane.height);
                    self.preview_block = if pass == 0 && whole_view { rendering::PREVIEW_BLOCKS.get(self.previews_received).copied().unwrap_or(1) } else { 1 };
                    self.previews_received += usize::from(pass == 0);
                    if pass == 0 {
                        continue; //A preview is not a layer: the progress and the layer times are not changed
                    }
                    if let Some(progress) = &mut self.progress {
                        progress.pass = pass;
                        progress.total = total;
                    }
                    if let Some(bars) = &mut self.bars {
                        if pass == 1 {
                            self.first_layer_per_pixel = Some(layer_time / u32::try_from(bars.pixels_total.max(1)).unwrap_or(u32::MAX));
                        }
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
                    self.buffer_is_complete = completed && self.settle.is_none(); //A render during a gesture has only the first layer
                    self.layer_shown |= completed;
                    if self.progress.take().is_some() {
                        if let Some(bars) = &mut self.bars {
                            bars.finished = Some((elapsed, completed));
                        }
                    }
                    self.shown_plane = c.clone();
                    self.c = c;
                    self.p = p;
                    self.texture_outdated = true;
                    if !self.gesture_active {
                        let description = self.view_description();
                        self.log(format!("{message} in {:.2} s: {description}", elapsed.as_secs_f64()));
                    }
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
    /// Every supersampling layer is shown as soon as it is computed, see `rendering::Render`
    pub(super) fn render(&mut self, message: &str) {
        self.start_render(message, None);
    }

    /// Renders the view after it was moved, then logs `message` and the current view. `shift` is the `(rows_up, columns_right)` to shift the pixel buffer with
    /// `PixelBuffer::translate_buffer`: the old image is shifted and only the pixels that were exposed by the move are rendered, progressively.
    /// If the buffer is not complete, because a render was interrupted or cancelled, or if the shift is too large, the complete view is rendered.
    /// During a drag the image on the screen is always shifted, whatever its quality, by the distance between its view and the new one: the full-quality render follows when the drag stops.
    pub(super) fn render_moved(&mut self, message: &str, shift: (i128, i128)) {
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let shift = if self.gesture_active { rendering::pixel_shift(&self.shown_plane, &self.c) } else { (self.progress.is_none() && self.buffer_is_complete).then_some(shift) };
        let strip = shift.and_then(|shift| ExposedRegion::from_shift(width, height, shift.0, shift.1).map(|region| (shift, region)));
        self.start_render(message, strip);
    }

    /// After a drag or a wheel zoom, starts the full-quality render of the view as soon as the gesture stops or pauses for `GESTURE_PAUSE`
    pub(super) fn settle_gesture(&mut self, ctx: &egui::Context) {
        let Some((_, changed)) = &self.settle else { return };
        if let (true, Some(remaining)) = (self.gesture_active, GESTURE_PAUSE.checked_sub(changed.elapsed())) {
            ctx.request_repaint_after(remaining);
            return;
        }
        if let Some((message, _)) = self.settle.take() {
            self.start_render_as(&message, None, false);
        }
    }

    /// Starts a render of the view on a background thread and logs `message` when it is done. With `strip` the old image is shifted and only the exposed region is rendered; otherwise everything is.
    /// A render that is running is interrupted first. During a drag or a wheel zoom the render is quick, see `start_render_as`.
    pub(super) fn start_render(&mut self, message: &str, strip: Option<((i128, i128), ExposedRegion)>) {
        self.start_render_as(message, strip, self.gesture_active);
    }

    /// Starts a render like `start_render`. With `during_gesture` it is a quick render for a view that is about to change again: previews and the first layer only, with
    /// the iteration limit of the image that is shown (no auto iterations probes); the full-quality render follows when the gesture stops, see `settle_gesture`.
    fn start_render_as(&mut self, message: &str, strip: Option<((i128, i128), ExposedRegion)>, during_gesture: bool) {
        self.cancel_thumbnail_job(); //The main view has priority over thumbnails
        self.interrupt_progressive_render();
        self.history.record(&self.c);
        self.buffer_is_complete = false;
        self.settle = during_gesture.then(|| (message.to_string(), Instant::now()));
        //Whether the screen already shows a full layer of this view (e.g. the last render of a gesture): then previews would only make it coarser again
        let layer_on_screen = self.layer_shown && rendering::pixel_shift(&self.shown_plane, &self.c) == Some((0, 0));
        self.layer_shown = false;
        self.previews_received = 0;
        let (sender, receiver) = mpsc::channel();
        let (c, mut p) = (self.c.clone(), self.p.clone());
        let adaptive = self.adaptive_supersampling.then(rendering::AdaptiveSampling::default);
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        //The pixels of a buffer that was rendered with the other mode have another limit and palette, so they cannot be reused
        let strip = strip.filter(|_| self.auto_iterations == self.m.palette_scale.is_some());
        //Publish the limit of the new render right away, before anything is rendered: the starting limit of the auto settings (cheap and deterministic), refined by the job as soon as its probes finish
        self.limit_pending = false;
        if self.auto_iterations {
            if strip.is_none() && !during_gesture {
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
        let supersampling_amount = if during_gesture { 1 } else { self.supersampling_amount };
        let coloring_function = self.coloring_function;
        let (width, height) = (p.pixel_plane.width, p.pixel_plane.height);
        let pixels_total = strip.as_ref().map_or(width * height, |(_, region)| region.pixel_count(width, height));
        //Previews only when the first layer is expected to take longer than a frame (always for the first render); a fast view then skips even the first preview level
        let expected_first_layer = self.first_layer_per_pixel.map(|per_pixel| per_pixel * u32::try_from(pixels_total).unwrap_or(u32::MAX));
        let preview = (!layer_on_screen && expected_first_layer.is_none_or(|expected| expected > PREVIEW_WORTHWHILE)).then_some(PREVIEW_WORTHWHILE);
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let pixels_done = Arc::new(AtomicUsize::new(0));
        let thread_pixels_done = Arc::clone(&pixels_done);
        let message = message.to_string();
        std::thread::spawn(move || {
            let started = Instant::now();
            let mut message = message;
            if auto_settings.is_some() && (strip.is_some() || during_gesture) {
                //A move keeps the limit of the image that is being moved, so only the exposed strip is rendered; the limit is chosen again on zooms, jumps and setting changes.
                //A gesture keeps it too, so the colors stay steady while the view changes; the full-quality render after the gesture chooses it
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
            let completed = Render::new(&c, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).only(&region, &base).preview(preview).cancel_with(&thread_cancel).count_pixels_in(&thread_pixels_done).run( |colors, pass| {
                //Only the colors are handed over, the window converts them when it draws; converting them here would delay the next layer
                p.colors.copy_from_slice(colors);
                if pass == 0 {
                    let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), plane: c.clone(), pass, total: supersampling_amount, layer_time: Duration::ZERO });
                    return;
                }
                let layer_time = layer_started.elapsed();
                println!("Rendered supersampling layer {pass} of {supersampling_amount} in {:.3} s", layer_time.as_secs_f64());
                let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), plane: c.clone(), pass, total: supersampling_amount, layer_time });
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
        if !self.gesture_active {
            self.log("Rendering...");
        }
    }
}

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
use super::MandelbrotApp;


/// A message from a rendering job that runs on a background thread
pub(super) enum JobMessage {
    /// An intermediate image: the running average of the first `pass` of `total` supersampling layers
    Pass { colors: Vec<TrueColor>, pass: u8, total: u8, layer_time: Duration },
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
    /// Every supersampling layer is shown as soon as it is computed, see `rendering::Render`
    pub(super) fn render(&mut self, message: &str) {
        self.start_render(message, None);
    }

    /// Renders the view after it was moved, then logs `message` and the current view. `shift` is the `(rows_up, columns_right)` to shift the pixel buffer with
    /// `PixelBuffer::translate_buffer`: the old image is shifted and only the pixels that were exposed by the move are rendered, progressively.
    /// If the buffer is not complete, because a render was interrupted or cancelled, or if the shift is too large, the complete view is rendered.
    pub(super) fn render_moved(&mut self, message: &str, shift: (i128, i128)) {
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let reusable = self.progress.is_none() && self.buffer_is_complete;
        let strip = if reusable { ExposedRegion::from_shift(width, height, shift.0, shift.1).map(|region| (shift, region)) } else { None };
        self.start_render(message, strip);
    }

    /// Starts a render of the view on a background thread and logs `message` when it is done. With `strip` the old image is shifted and only the exposed region is rendered; otherwise everything is.
    /// A render that is running is interrupted first.
    pub(super) fn start_render(&mut self, message: &str, strip: Option<((i128, i128), ExposedRegion)>) {
        self.cancel_thumbnail_job(); //The main view has priority over thumbnails
        self.interrupt_progressive_render();
        self.history.record(&self.c);
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
            let completed = Render::new(&c, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).only(&region, &base).cancel_with(&thread_cancel).count_pixels_in(&thread_pixels_done).run( |colors, pass| {
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
}

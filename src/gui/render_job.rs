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
use crate::rendering::{self, Render};
use eframe::egui;

use super::{MandelbrotApp, Readiness};


/// A preview of a new view is only shown if the first layer is expected to take longer than this, about a frame: a faster first layer is shown directly, see `Render::preview`
const PREVIEW_WORTHWHILE: Duration = Duration::from_millis(16);

/// The GPU joins the first pass of a render once it has run for this long (measured: 3 ms takes 3 ms off a deep 1x render and leaves the 6 ms default view as it
/// is; 0 ms slows that view down), so passes that are fast on the CPU alone never wait for it. Later passes that are known to be slow get it at once, see `Render::gpu`
const GPU_JOIN_AFTER: Duration = Duration::from_millis(3);

/// The full-quality render after a drag or a wheel zoom starts when the gesture stops, or when it has not changed the view for this long (a mouse button held still)
const GESTURE_PAUSE: Duration = Duration::from_millis(150);

/// What a render is for, which decides its iteration limit, its layers, and whether the history records it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RenderKind {
    /// A view the window settles on: every layer, recorded in the history. With `moved` the view is the last view moved by whole pixels (an arrow, a drag,
    /// centering on a click), so it keeps the iteration limit and continues the samples of the last render, see `render_moved`
    Settled { moved: bool },
    /// A frame of a drag or a wheel zoom, for a view that is about to change again: previews and the first layer only, with the iteration limit of the image that
    /// is shown (no auto iterations probes); the full-quality render follows when the gesture stops, see `settle_gesture`
    GestureFrame { moved: bool },
    /// A frame of a flight to a point of interest: the iteration limit that the flight sets (see `points::fly_at`), every layer until the next frame interrupts it
    FlightFrame,
}

impl RenderKind {
    /// Whether the view is the last view moved by whole pixels
    const fn moved(self) -> bool {
        match self {
            RenderKind::Settled { moved } | RenderKind::GestureFrame { moved } => moved,
            RenderKind::FlightFrame => false,
        }
    }

    /// Whether the view is about to change again (a frame of a gesture or a flight): not recorded in the history, with a steady iteration limit
    const fn passing(self) -> bool {
        !matches!(self, RenderKind::Settled { .. })
    }

    /// Whether every supersampling layer is rendered (all but the frames of a gesture, which render the first layer only)
    const fn all_layers(self) -> bool {
        !matches!(self, RenderKind::GestureFrame { .. })
    }
}

/// A message from a rendering job that runs on a background thread
pub(super) enum JobMessage {
    /// An intermediate image: the running average of the first `pass` of `total` supersampling layers, or a low-resolution preview of the first layer when `pass` is 0
    /// `whole` is true when the render started from nothing, so the pass covers the whole view (not only pixels that a move exposed)
    Pass { colors: Vec<TrueColor>, plane: ComplexPlane, pass: u8, total: u8, layer_time: Duration, whole: bool },
    /// The iteration limit that auto iterations chose for the render, sent as soon as it is known, before the first layer is rendered
    Limit { max_iterations: u32, palette_scale: u32 },
    /// The job has finished (or was cancelled), this is its final result; `reported` is whether it showed anything of `c` (otherwise `p` still shows the previous view)
    Done { c: ComplexPlane, p: PixelBuffer, m: MandelbrotSet, message: String, elapsed: Duration, completed: bool, reported: bool },
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
        if let Some(opened) = self.gpu_opening.as_ref().and_then(|receiver| receiver.try_recv().ok()) {
            self.gpu_opening = None;
            match opened {
                Ok(gpu) => {
                    self.log(format!("GPU: {}, checked against the CPU: it computes the same samples, bit for bit", gpu.name()));
                    self.gpu = Some(Arc::new(gpu));
                }
                Err(reason) => self.log(format!("GPU not used: {reason}")),
            }
        }
        loop {
            let Some(receiver) = &self.job else { return };
            match receiver.try_recv() {
                Ok(JobMessage::Pass { colors, plane, pass, total, layer_time, whole }) => {
                    //A preview of the whole view is uploaded small and scaled up smoothly; a preview of a strip sits next to sharp pixels, so it stays sharp
                    let preview_block = if pass == 0 && whole { rendering::PREVIEW_BLOCKS.get(self.previews_received).copied().unwrap_or(1) } else { 1 };
                    self.previews_received += usize::from(pass == 0);
                    //A frame that is not complete enough is not shown: the screen keeps the last one that was (immersion, flight rendering)
                    //Every preview is complete: a block with pixels without samples always shows one of their samples, also in the strip a move exposed (see `Render::preview`)
                    let readiness = if pass > 0 && pass >= total { Readiness::SuperSampled } else if pass > 0 { Readiness::FullResolution } else { Readiness::Complete };
                    //(a layer that is held back still counts for the progress bar below)
                    if readiness >= self.required_readiness() {
                        self.frame_ready = true;
                        self.frame_accepted = true;
                        self.p.colors = colors;
                        self.shown_plane = plane;
                        self.texture_outdated = true;
                        self.layer_shown = pass > 0;
                        self.preview_block = preview_block;
                    }
                    if pass == 0 {
                        continue; //A preview is not a layer: the progress and the layer times are not changed
                    }
                    if let Some(progress) = &mut self.progress {
                        progress.pass = pass;
                        progress.total = total;
                    }
                    if let Some(bars) = &mut self.bars {
                        if pass == 1 && whole {
                            //Only a render of the whole view tells the time per pixel: a continued render only computes the pixels that miss samples
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
                Ok(JobMessage::Done { c, p, m, message, elapsed, completed, reported }) => {
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
                    //A cancelled render that showed no frame (or none complete enough) leaves the screen as it was
                    if completed || self.frame_accepted {
                        if reported {
                            self.shown_plane = c.clone();
                        }
                        self.p = p;
                    }
                    if self.flight.is_none() {
                        self.c = c; //During a flight the camera moves on while a render runs (best effort), so it is not set back to the view of the render
                    }
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
        self.start_render(message, false);
    }

    /// Renders the view after it was moved by whole pixels (an arrow, a drag, centering on a click), then logs `message` and the current view.
    /// The move keeps the iteration limit (also with auto iterations, once it is settled), so the render continues the samples of the last render, see `start_render_as`:
    /// the pixels that stay visible keep every sample they have, also from a render that was interrupted, and only the missing samples are computed.
    pub(super) fn render_moved(&mut self, message: &str) {
        self.start_render(message, true);
    }

    /// After a drag or a wheel zoom, starts the full-quality render of the view as soon as the gesture stops or pauses for `GESTURE_PAUSE`
    pub(super) fn settle_gesture(&mut self, ctx: &egui::Context) {
        if self.flight.is_some() {
            return; //A flight renders its destination with full quality when it arrives, see `points::fly_at`
        }
        let Some((_, changed, _)) = &self.settle else { return };
        if let (true, Some(remaining)) = (self.gesture_active, GESTURE_PAUSE.checked_sub(changed.elapsed())) {
            ctx.request_repaint_after(remaining);
            return;
        }
        if let Some((message, _, moved)) = self.settle.take() {
            self.start_render_as(&message, RenderKind::Settled { moved });
        }
    }

    /// Starts a render of the view on a background thread and logs `message` when it is done; `moved` is whether the view was only moved, see `render_moved`.
    /// A render that is running is interrupted first. During a drag or a wheel zoom the render is quick, see `start_render_as`.
    pub(super) fn start_render(&mut self, message: &str, moved: bool) {
        self.start_render_as(message, if self.gesture_active { RenderKind::GestureFrame { moved } } else { RenderKind::Settled { moved } });
    }

    /// Starts a render like `start_render`, of the `kind` that says what it is for (see `RenderKind`).
    ///
    /// Every render continues the samples of the render before it (`SampleState`): the job waits for the state of the job it interrupts, which keeps everything that
    /// job computed. The renderer reuses it when the view is that view moved by whole pixels with the same settings, and starts from nothing otherwise.
    pub(super) fn start_render_as(&mut self, message: &str, kind: RenderKind) {
        let (moved, during_gesture, all_layers) = (kind.moved(), kind.passing(), kind.all_layers());
        self.cancel_thumbnail_job(); //The main view has priority over thumbnails
        self.interrupt_progressive_render();
        (self.frame_ready, self.frame_accepted) = (false, false);
        self.flight = None; //Any other render stops a flight to a point of interest where it is; a frame of the flight puts it back, see `points::fly_at`
        if !during_gesture {
            //Only the views that the window settles on: not every frame of a drag, a wheel zoom or a flight
            self.history.record(&self.c);
        }
        self.buffer_is_complete = false;
        self.settle = during_gesture.then(|| (message.to_string(), Instant::now(), moved));
        //The samples of the last render, handed from job to job; this job hands its own on when it stops, finished or not
        let previous_samples = self.samples.take();
        let (samples_sender, samples_receiver) = mpsc::channel();
        self.samples = Some(samples_receiver);
        //Whether the screen already shows a full layer of this view (e.g. the last render of a gesture): then previews would only make it coarser again
        let layer_on_screen = self.layer_shown && rendering::pixel_shift(&self.shown_plane, &self.c) == Some((0, 0));
        self.layer_shown = false;
        self.previews_received = 0;
        let (sender, receiver) = mpsc::channel();
        let (c, mut p) = (self.c.clone(), self.p.clone());
        //A move keeps the iteration limit once it is settled; while the probes of the last render had not chosen it yet, the move chooses it like any render
        let keeps_limit = (moved || during_gesture) && !self.limit_pending;
        let adaptive = self.adaptive_supersampling.then(rendering::AdaptiveSampling::default);
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        //Publish the limit of the new render right away, before anything is rendered: the starting limit of the auto settings (cheap and deterministic), refined by the job as soon as its probes finish
        self.limit_pending = false;
        if self.auto_iterations {
            if !keeps_limit {
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
        let supersampling_amount = if all_layers { self.supersampling_amount } else { 1 };
        let coloring_function = self.coloring_function;
        let (width, height) = (p.pixel_plane.width, p.pixel_plane.height);
        let pixels_total = width * height;
        //Previews only when the first layer is expected to take longer than a frame (always for the first render); a fast view then skips even the first preview level
        let expected_first_layer = self.first_layer_per_pixel.map(|per_pixel| per_pixel * u32::try_from(pixels_total).unwrap_or(u32::MAX));
        let preview = (!layer_on_screen && expected_first_layer.is_none_or(|expected| expected > PREVIEW_WORTHWHILE)).then_some(PREVIEW_WORTHWHILE);
        let gpu = self.gpu.clone().filter(|_| self.use_gpu);
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let pixels_done = Arc::new(AtomicUsize::new(0));
        let thread_pixels_done = Arc::clone(&pixels_done);
        let message = message.to_string();
        std::thread::spawn(move || {
            let started = Instant::now();
            //Waits for the interrupted job to stop (it finishes the row it is on) and hand over its samples
            let mut samples = previous_samples.and_then(|receiver| receiver.recv().ok()).unwrap_or_default();
            let mut message = message;
            if auto_settings.is_some() && keeps_limit {
                //A move keeps the limit of the image that is being moved, so the samples of the pixels that stay visible are reused; the limit is chosen again on zooms, jumps and setting changes.
                //A gesture keeps it too, so the colors stay steady while the view changes; the full-quality render after a wheel zoom chooses it
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
                        let _ = samples_sender.send(samples);
                        let _ = sender.send(JobMessage::Done { c, p, m, message: String::from("Cancelled the rendering"), elapsed: started.elapsed(), completed: false, reported: false });
                        return;
                    }
                }
            } else if auto_settings.is_none() {
                m.palette_scale = None;
            }
            let kept = samples.prepare_for(&c, width, height, &m, coloring_function);
            let whole = kept == 0;
            if kept > 0 {
                message = format!("{message} (continued the samples of {kept} pixels)");
            }
            let mut layer_started = started;
            let mut reported = false;
            let completed = Render::new(&c, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).resume(&mut samples).preview(preview).gpu(gpu.as_deref().map(|gpu| (gpu, GPU_JOIN_AFTER))).cancel_with(&thread_cancel).count_pixels_in(&thread_pixels_done).run( |colors, pass| {
                //Only the colors are handed over, the window converts them when it draws; converting them here would delay the next layer
                p.colors.copy_from_slice(colors);
                reported = true;
                if pass == 0 {
                    let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), plane: c.clone(), pass, total: supersampling_amount, layer_time: Duration::ZERO, whole });
                    return;
                }
                let layer_time = layer_started.elapsed();
                println!("Rendered supersampling layer {pass} of {supersampling_amount} in {:.3} s", layer_time.as_secs_f64());
                let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), plane: c.clone(), pass, total: supersampling_amount, layer_time, whole });
                layer_started = Instant::now();
            });
            let _ = samples_sender.send(samples); //To the next job, or kept for it if there is none yet
            p.update_pixels(); //The pixels of the last completed layer
            let message = if completed { message } else { String::from("Cancelled the rendering, showing the layers that were completed") };
            let _ = sender.send(JobMessage::Done { c, p, m, message, elapsed: started.elapsed(), completed, reported }); //The receiver is gone if the window was closed
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

//! Rendering jobs: the background threads that render the view layer by layer, and the app state that follows them.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};


use crate::auto_iterations::{self};
use crate::coloring::TrueColor;
use crate::complex_plane::ComplexPlane;
use crate::mandelbrot_set::{ColorCycle, MandelbrotSet};
use crate::rendering::{self, Render};
use eframe::egui;

use super::{MandelbrotApp, Readiness};


/// A preview of a new view is only shown if the first layer is expected to take longer than this, about a frame: a faster first layer is shown directly, see `Render::preview`
pub(super) const PREVIEW_WORTHWHILE: Duration = Duration::from_millis(16);

/// The GPU joins the first pass of a render once it has run for this long (measured: 3 ms takes 3 ms off a deep 1x render and leaves the 6 ms default view as it
/// is; 0 ms slows that view down), so passes that are fast on the CPU alone never wait for it. Later passes that are known to be slow get it at once, see `Render::gpu`
pub(super) const GPU_JOIN_AFTER: Duration = Duration::from_millis(3);

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

/// The colors of frames that the window no longer shows, shared by the window and every job: the window puts the colors a new frame replaces here, and the jobs
/// copy their frames into them (`copy`). Shared, so colors are not lost with a job that was interrupted
#[derive(Clone, Default)]
pub(super) struct SparePool(Arc<std::sync::Mutex<Vec<Vec<TrueColor>>>>);

impl SparePool {
    /// Enough for the frames in flight between a job and the window; more would only hold memory
    pub(super) const KEPT: usize = 3;

    /// Keeps `colors` for a later frame, unless the pool is full
    pub(super) fn put(&self, colors: Vec<TrueColor>) {
        let mut pool = self.0.lock().unwrap();
        if pool.len() < Self::KEPT {
            pool.push(colors);
        }
    }

    /// How many spare colors are kept
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    /// A copy of `colors`, in spare colors of the same size when there are any, else newly allocated
    pub(super) fn copy(&self, colors: &[TrueColor]) -> Vec<TrueColor> {
        let spare = {
            let mut pool = self.0.lock().unwrap();
            pool.retain(|spare| spare.capacity() >= colors.len()); //Colors of another window size are no use
            pool.pop()
        };
        let mut copy = spare.unwrap_or_default();
        copy.clear();
        copy.extend_from_slice(colors);
        copy
    }

    /// Colors of `length` pixels in rows of `stride` that hold the `small.0` x `small.1` image `colors` in their top-left corner (the rest is not set): a scaled
    /// frame of Auto, which keeps the size of the image the window holds, so only the rows of the small image are copied
    pub(super) fn packed(&self, colors: &[TrueColor], small: (usize, usize), length: usize, stride: usize) -> Vec<TrueColor> {
        let spare = {
            let mut pool = self.0.lock().unwrap();
            pool.retain(|spare| spare.capacity() >= length);
            pool.pop()
        };
        let mut packed = spare.unwrap_or_default();
        packed.resize(length, TrueColor::new(0, 0, 0)); //(Only when the spare colors were not of this size: nothing is written to colors that are)
        for (row, line) in colors.chunks_exact(small.0).take(small.1).enumerate() {
            packed[row * stride..row * stride + small.0].copy_from_slice(line);
        }
        packed
    }
}

/// A message from a rendering job that runs on a background thread
pub(super) enum JobMessage {
    /// An intermediate image: the running average of the first `pass` of `total` supersampling layers, or a low-resolution preview of the first layer when `pass` is 0
    /// `whole` is true when the render started from nothing, so the pass covers the whole view (not only pixels that a move exposed)
    Pass { colors: Vec<TrueColor>, plane: ComplexPlane, pass: u8, total: u8, layer_time: Duration, whole: bool },
    /// The iteration limit that auto iterations chose for the render, sent as soon as it is known, before the first layer is rendered
    Limit { max_iterations: u32 },
    /// A frame of a flight (see `flight_job`) is complete enough: the flight clock at the frame, its view, its iteration limit and palette, and which preview of
    /// the render it is (the `Pass` that follows carries its colors)
    /// `scaled` is the size of an Auto frame below full resolution, whose colors are the rows of that smaller image (see `SparePool::packed`)
    FlightFrame { elapsed: Duration, plane: ComplexPlane, max_iterations: u32, palette: ColorCycle, preview: usize, levels: Vec<usize>, scaled: Option<(usize, usize)> },
    /// The job has finished (or was cancelled), this is its final result; `reported` is whether it showed anything of `c` (otherwise `p` still shows the previous view)
    Done { c: ComplexPlane, m: MandelbrotSet, message: String, elapsed: Duration, completed: bool, reported: bool },
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
    /// For a job that renders several frames (a flight, see `flight_job`): stops the job, where `cancel` only stops its frame
    pub(super) stop: Option<Arc<AtomicBool>>,
}

impl MandelbrotApp {
    /// Cancels the running progressive render, if any, and ignores everything it still reports.
    /// The image of the last completed layer stays on screen until the next render shows its first layer.
    pub(super) fn interrupt_progressive_render(&mut self) {
        if let Some(progress) = self.progress.take() {
            if let Some(stop) = &progress.stop {
                stop.store(true, Ordering::SeqCst);
            }
            progress.cancel.store(true, Ordering::SeqCst);
            self.job = None;
        }
        //Frames rendered ahead by a stopped flight job are never shown
        self.flight_queue.clear();
        self.frame_hold = 0;
    }

    /// The next message of the job, `None` without a job. The frames of a flight job that renders ahead (see `flight_job`) wait in `flight_queue` until the frame on
    /// the screen has stayed there for one refresh (`frame_hold`, counted down by `points::fly`); the other messages follow in order.
    fn next_job_message(&mut self) -> Option<Result<JobMessage, TryRecvError>> {
        let receiver = self.job.as_ref()?;
        if !(self.flight_job && self.frames_ahead > 0) && self.flight_queue.is_empty() {
            return Some(receiver.try_recv());
        }
        let mut disconnected = false;
        loop {
            match receiver.try_recv() {
                Ok(message) => self.flight_queue.push_back(message),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        match self.flight_queue.front() {
            Some(JobMessage::FlightFrame { .. }) if self.frame_hold > 0 => Some(Err(TryRecvError::Empty)),
            Some(JobMessage::FlightFrame { .. }) => {
                self.frames_queued.fetch_sub(1, Ordering::SeqCst);
                self.flight_queue.pop_front().map(Ok)
            }
            Some(_) => self.flight_queue.pop_front().map(Ok),
            None if disconnected => Some(Err(TryRecvError::Disconnected)),
            None => Some(Err(TryRecvError::Empty)),
        }
    }

    /// Takes the messages of the background job: intermediate images and the final result
    #[allow(clippy::too_many_lines)] // One match over every message of a job
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
            let Some(next) = self.next_job_message() else { return };
            match next {
                Ok(JobMessage::Pass { colors, plane, pass, total, layer_time, whole }) => {
                    //A preview of the whole view is uploaded small and scaled up smoothly; a preview of a strip sits next to sharp pixels, so it stays sharp
                    let level = self.preview_levels.get(self.previews_received).copied().unwrap_or(1);
                    let preview_block = if pass == 0 && whole { level } else { 1 };
                    self.previews_received += usize::from(pass == 0);
                    //A frame that is not complete enough is not shown: the screen keeps the last one that was (immersion, flight rendering)
                    //Every preview is complete: a block with pixels without samples always shows one of their samples, also in the strip a move exposed (see `Render::preview`)
                    let readiness = if pass > 0 && pass >= total { Readiness::SuperSampled } else if pass > 0 { Readiness::Layers(pass) } else { Readiness::Blocks(level) };
                    //(a layer that is held back still counts for the progress bar below)
                    if readiness >= self.required_readiness() {
                        if let (false, false, Some(flight)) = (self.flight_job, self.frame_accepted, &mut self.flight) {
                            flight.frames.push(Instant::now()); //A new frame of the flight (not a finer image of the same view), for its frame rate
                        }
                        self.frame_ready = true;
                        self.frame_accepted = true;
                        let replaced = std::mem::replace(&mut self.p.colors, colors);
                        self.spare_colors.put(replaced);
                        self.shown_plane = plane;
                        self.texture_outdated = true;
                        self.layer_shown = pass > 0;
                        self.preview_block = preview_block;
                        self.shown_samples = pass;
                        self.shown_scaled = self.frame_scaled.take(); //(Only a frame of a flight job can be scaled)
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
                Ok(JobMessage::Limit { max_iterations }) => {
                    //Only the running job can send this: the receiver of an interrupted job is dropped, so a stale limit never reaches the display
                    self.m.max_iterations = max_iterations;
                    self.limit_pending = false;
                }
                Ok(JobMessage::FlightFrame { elapsed, plane, max_iterations, palette, preview, levels, scaled }) => {
                    //A frame rendered ahead stays on the screen for one refresh before the next one shows (see `next_job_message`)
                    self.frame_hold = u32::from(self.frames_ahead > 0);
                    if let Some(flight) = &mut self.flight {
                        flight.elapsed = elapsed;
                        flight.frames.push(Instant::now()); //A frame of the flight, for its frame rate
                    }
                    self.c = plane; //The camera is where the frame is
                    (self.m.max_iterations, self.m.palette) = (max_iterations, palette);
                    self.previews_received = preview; //So the pass that follows is known as that preview (or layer)
                    self.preview_levels = levels; //(The levels of this frame: with Auto every frame can have its own)
                    self.frame_scaled = scaled;
                }
                Ok(JobMessage::Done { c, m, message, elapsed, completed, reported }) => {
                    self.landing_blur = false; //The render of the place is finished: sharp
                    self.job = None;
                    self.flight_job = false;
                    self.flight_frame_job = false;
                    self.limit_pending = false;
                    //In auto mode this holds the chosen limit; the manual value in `max_iterations_input` is kept, and so is the Smooth colors setting of the window
                    self.m = MandelbrotSet { smooth_coloring: self.m.smooth_coloring, ..m };
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
                        self.p.update_pixels(); //The colors are those of the last frame shown, which arrived with its pass
                    }
                    if self.flight.is_none() {
                        self.c = c; //During a flight the camera moves on while a render runs (best effort), so it is not set back to the view of the render
                    }
                    self.texture_outdated = true;
                    if !self.gesture_active && !self.color_cycling && !message.is_empty() { //(Not every frame of a gesture or step of the color cycling)
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
        if self.color_cycling && !self.gesture_active {
            return; //Every step of the cycling is a one-sample frame like those of a gesture; the full-quality render follows when it stops, see `cycle::toggle_color_cycle`
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

    /// The block sizes of the previews of a render of `kind`, coarsest first: for a flight frame only those the screen would show (none when it waits for full layers,
    /// only the size it waits for when it waits for blocks of a size), else `rendering::PREVIEW_BLOCKS`
    pub(super) fn preview_levels(&self, kind: RenderKind) -> Vec<usize> {
        match (kind, self.flight_readiness()) {
            (RenderKind::FlightFrame, Readiness::Blocks(block)) if block != usize::MAX => vec![block.max(2)],
            (RenderKind::FlightFrame, readiness) if readiness >= Readiness::FULL_RESOLUTION => Vec::new(),
            _ => rendering::PREVIEW_BLOCKS.to_vec(),
        }
    }

    /// Starts a render like `start_render`, of the `kind` that says what it is for (see `RenderKind`).
    ///
    /// Every render continues the samples of the render before it (`SampleState`): the job waits for the state of the job it interrupts, which keeps everything that
    /// job computed. The renderer reuses it when the view is that view moved by whole pixels with the same settings, and starts from nothing otherwise.
    #[allow(clippy::too_many_lines)] // One place that sets up every render: its settings, its state and the thread that runs it
    pub(super) fn start_render_as(&mut self, message: &str, kind: RenderKind) {
        self.landing_blur = false; //(The render at a place a flight arrived at sets it again, see `points::arrive_at`)
        self.cancel_neighborhood_job();
        self.keep_within_zoom_limit();
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
        //A view deeper than 64-bit floats is rendered by the deep renderer (see `deep_view`)
        if !matches!(kind, RenderKind::FlightFrame) && self.deep_view().is_some() {
            (self.layer_shown, self.flight_job, self.flight_frame_job) = (false, false, false);
            self.flight_queue.clear();
            self.bars = None;
            self.start_deep_render(message);
            return;
        }
        //The samples of the last render, handed from job to job; this job hands its own on when it stops, finished or not
        let previous_samples = self.samples.take();
        let (samples_sender, samples_receiver) = mpsc::channel();
        self.samples = Some(samples_receiver);
        //Whether the screen already shows a full layer of this view (e.g. the last render of a gesture): then previews would only make it coarser again
        let layer_on_screen = self.layer_shown && rendering::pixel_shift(&self.shown_plane, &self.c) == Some((0, 0));
        self.layer_shown = false;
        self.previews_received = 0;
        let levels = self.preview_levels(kind);
        self.preview_levels.clone_from(&levels);
        let (sender, receiver) = mpsc::channel();
        let spare_colors = self.spare_colors.clone();
        //The job only needs the view: the window keeps the image (every frame it shows arrives as a `Pass`), so a big image is not copied for every render
        let c = self.c.clone();
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
                self.limit_pending = true;
            }
        } else {
            self.m.max_iterations = self.max_iterations_input;
        }
        let mut m = self.m.clone();
        let supersampling_amount = if all_layers { self.supersampling_amount } else { 1 };
        let coloring_function = self.coloring_function;
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let pixels_total = width * height;
        //Previews only when the first layer is expected to take longer than a frame (always for the first render); a fast view then skips even the first preview level
        let expected_first_layer = self.first_layer_per_pixel.map(|per_pixel| per_pixel * u32::try_from(pixels_total).unwrap_or(u32::MAX));
        let preview = (!layer_on_screen && !levels.is_empty() && expected_first_layer.is_none_or(|expected| expected > PREVIEW_WORTHWHILE)).then_some(PREVIEW_WORTHWHILE);
        //Frames that move (a drag, a flight) may use the f32 loop; the image the view settles on never does
        let (gpu, fast_gpu, f32_frames) = (self.gpu.clone().filter(|_| self.use_gpu), self.fast_gpu, self.f32_frames && during_gesture);
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
                        let _ = sender.send(JobMessage::Limit { max_iterations: result.limit });
                        m.max_iterations = result.limit;
                        message = format!("{message} (auto iterations: limit {}, {} rounds, {:?}, {:.2} s)", result.limit, result.rounds.len(), result.stop, started.elapsed().as_secs_f64());
                        println!("{result:?}");
                    }
                    None => {
                        let _ = samples_sender.send(samples);
                        let _ = sender.send(JobMessage::Done { c, m, message: String::from("Cancelled the rendering"), elapsed: started.elapsed(), completed: false, reported: false });
                        return;
                    }
                }
            }
            let kept = samples.prepare_for(&c, width, height, &m, coloring_function, fast_gpu && gpu.is_some(), f32_frames && fast_gpu && gpu.is_some());
            let whole = kept == 0;
            if kept > 0 {
                message = format!("{message} (continued the samples of {kept} pixels)");
            }
            let mut layer_started = started;
            let mut reported = false;
            let completed = Render::new(&c, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).resume(&mut samples).preview(preview).preview_blocks(&levels).gpu(gpu.as_deref().map(|gpu| (gpu, GPU_JOIN_AFTER))).fast_gpu(fast_gpu).f32_gpu(f32_frames).cancel_with(&thread_cancel).count_pixels_in(&thread_pixels_done).run( |colors, pass| {
                //Only the colors are handed over, the window converts them when it draws; converting them here would delay the next layer
                reported = true;
                if pass == 0 {
                    let _ = sender.send(JobMessage::Pass { colors: spare_colors.copy(colors), plane: c.clone(), pass, total: supersampling_amount, layer_time: Duration::ZERO, whole });
                    return;
                }
                let layer_time = layer_started.elapsed();
                println!("Rendered supersampling layer {pass} of {supersampling_amount} in {:.3} s", layer_time.as_secs_f64());
                let _ = sender.send(JobMessage::Pass { colors: spare_colors.copy(colors), plane: c.clone(), pass, total: supersampling_amount, layer_time, whole });
                layer_started = Instant::now();
            });
            let _ = samples_sender.send(samples); //To the next job, or kept for it if there is none yet
            let message = if completed { message } else { String::from("Cancelled the rendering, showing the layers that were completed") };
            let _ = sender.send(JobMessage::Done { c, m, message, elapsed: started.elapsed(), completed, reported }); //The receiver is gone if the window was closed
        });
        let now = Instant::now();
        self.bars = Some(RenderBars { total_layers: supersampling_amount.clamp(1, 64), layer_times: Vec::new(), started: now, layer_started: now, pixels_done, pixels_total, finished: None });
        self.flight_queue.clear();
        self.job = Some(receiver);
        self.flight_job = false;
        self.flight_frame_job = matches!(kind, RenderKind::FlightFrame);
        self.progress = Some(Progress { pass: 0, total: supersampling_amount, cancel, stop: None });
        if !self.gesture_active && !self.color_cycling {
            self.log("Rendering...");
        }
    }
}

//! Pipelined flights: with a flight rendering that waits for its frames (see `flight::FlightRendering`), one job renders the frames of the flight one after the
//! other. As soon as a frame is complete enough it goes to the window and the job starts the view of the next frame, without waiting for the next frame of the
//! window: the CPU threads and the GPU go from frame to frame without a pause, and the finer layers of a frame that the flight leaves behind are not computed.
//!
//! A frame that waits for one full layer keeps only its colors (`Render::colors_only`), as the next frame replaces it; the frame at the destination keeps its
//! samples, so the render after the arrival continues from them. Once a frame has taken longer than the join delay of the GPU, the next frames give the GPU work
//! at once: a flight frame is never one of the fast renders the delay is for.
//!
//! The flight clock moves on by the time a frame took, at most one frame of the screen (`frame_interval`, and at most `MAX_WAITING_STEP`): a slow view makes a
//! slow, smooth crawl instead of jumps. The job renders no more frames per second than the screen shows. A best-effort flight follows the clock instead, see
//! `points::fly_at`.
//!
//! With frames ahead (`MandelbrotApp::frames_ahead`), the job does not wait for the screen: it renders up to that many frames ahead of the one on the screen, each
//! one refresh further along the clock, and the window shows one of them every refresh (see `render_job::next_job_message`), so the CPU threads and the GPU
//! never wait for the screen.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use super::points::MAX_WAITING_STEP;
use super::render_job::{JobMessage, Progress, RenderKind, GPU_JOIN_AFTER, PREVIEW_WORTHWHILE};
use super::flight::{auto_scale, FlightRendering};
use super::{MandelbrotApp, Readiness};
use crate::complex_plane::ComplexPlane;
use crate::rendering::{self, Render};

impl MandelbrotApp {
    /// Starts the job that renders the frames of the flight under way, from where its clock is, see the module. Its frames arrive as `JobMessage::FlightFrame`
    /// followed by the `Pass` of the frame; it stops when it has rendered the frame at the destination (the window then arrives, see `points::fly_at`)
    #[allow(clippy::too_many_lines)] // The settings of the job and its loop over the frames, which share their state
    pub(super) fn start_flight_job(&mut self, frame_interval: Duration) {
        let Some(flight) = &mut self.flight else { return };
        let (path, mut elapsed, mut limits) = (flight.flight.clone(), flight.elapsed, flight.limits.take());
        self.cancel_thumbnail_job();
        self.interrupt_progressive_render();
        (self.frame_ready, self.frame_accepted, self.buffer_is_complete, self.layer_shown) = (false, false, false, false);
        let previous_samples = self.samples.take();
        let (samples_sender, samples_receiver) = mpsc::channel();
        self.samples = Some(samples_receiver);
        let (sender, receiver) = mpsc::channel();
        let spare_colors = self.spare_colors.clone();
        let (mut required, mut levels) = (self.flight_readiness(), self.preview_levels(RenderKind::FlightFrame));
        self.preview_levels.clone_from(&levels);
        //With Auto: the frame time to keep, and the scale of the next frame (the share of the pixels across), steered by the time of the frames (see `flight::auto_scale`)
        let auto = match self.flight_rendering {
            FlightRendering::Auto(fps) => Some(Duration::from_secs_f64(1.0 / f64::from(fps.max(1)))),
            _ => None,
        };
        let mut scale = 0.7;
        let adaptive = self.adaptive_supersampling.then(rendering::AdaptiveSampling::default);
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        let mut m = self.m.clone();
        if !self.auto_iterations {
            (m.max_iterations, m.palette_scale) = (self.max_iterations_input, None);
        }
        let (supersampling_amount, coloring_function) = (self.supersampling_amount, self.coloring_function);
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let pixels = u32::try_from(width * height).unwrap_or(u32::MAX);
        let mut first_layer_per_pixel = self.first_layer_per_pixel;
        let (gpu, fast_gpu, f32_frames) = (self.gpu.clone().filter(|_| self.use_gpu), self.fast_gpu, self.f32_frames);
        //`cancel` stops the frame that is rendered (the job sets it when the frame is complete enough), `stop` the whole job (the window sets it, then `cancel`)
        let (cancel, stop) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (thread_cancel, thread_stop) = (Arc::clone(&cancel), Arc::clone(&stop));
        //Frames ahead: the frames the job has sent that the window has not shown yet (see `render_job::next_job_message`)
        self.frames_queued = Arc::new(AtomicUsize::new(0));
        let clock = Clock { frame_interval, duration: path.duration(), ahead: self.frames_ahead, queued: Arc::clone(&self.frames_queued), stop: Arc::clone(&stop) };
        let queued = Arc::clone(&self.frames_queued);
        self.flight_queue.clear();
        self.frame_hold = 0;
        std::thread::spawn(move || {
            let started = Instant::now();
            let mut samples = previous_samples.and_then(|receiver| receiver.recv().ok()).unwrap_or_default();
            let mut plane = ComplexPlane::new(width, height);
            let mut frame_started = Instant::now();
            //The join delay of the GPU for the next frame: none once a frame has taken longer than a few delays
            let mut join_after = GPU_JOIN_AFTER;
            while !thread_stop.load(Ordering::SeqCst) {
                thread_cancel.store(false, Ordering::SeqCst);
                if thread_stop.load(Ordering::SeqCst) {
                    break; //The window stopped the job between the check and the reset
                }
                plane.set_view(&path.camera_at(elapsed).view());
                if let Some(settings) = &auto_settings {
                    //The limit of the zoom of this frame (see `points::FlightLimits`), and the palette of its zoom, which is what auto iterations uses
                    let scale = plane.get_scale();
                    let palette = settings.initial_limit(scale);
                    m.max_iterations = limits.as_mut().map_or(palette, |limits| limits.limit_at(scale, settings));
                    m.palette_scale = Some(palette);
                }
                let last = path.arrived(elapsed);
                if auto.is_some() {
                    (required, levels) = (Readiness::FULL_RESOLUTION, Vec::new());
                }
                if let (true, false, Some(small)) = (auto.is_some() && scale < 1.0, last, scaled_size(width, height, scale)) {
                    //An Auto frame below full resolution: the same view in fewer pixels, which the window scales up (see `MandelbrotApp::shown_scaled`)
                    let mut small_plane = ComplexPlane::new(small.0, small.1);
                    small_plane.set_view(&plane.get_view());
                    let mut shown = false;
                    Render::new(&small_plane, &m, small.0, small.1, coloring_function).gpu(gpu.as_deref().map(|gpu| (gpu, join_after))).fast_gpu(fast_gpu).f32_gpu(f32_frames).cancel_with(&thread_cancel)
                        .colors_only().run(|colors, pass| {
                            if pass == 0 || shown {
                                return;
                            }
                            shown = true;
                            let _ = sender.send(JobMessage::FlightFrame { elapsed, plane: plane.clone(), max_iterations: m.max_iterations, palette_scale: m.palette_scale, preview: 0, levels: Vec::new(), scaled: Some(small) });
                            queued.fetch_add(1, Ordering::SeqCst);
                            let _ = sender.send(JobMessage::Pass { colors: spare_colors.packed(colors, small, width * height, width), plane: plane.clone(), pass: 1, total: supersampling_amount, layer_time: Duration::ZERO, whole: true });
                            thread_cancel.store(true, Ordering::SeqCst);
                        });
                    if !next_frame(shown, &mut scale, auto, &mut join_after, &mut elapsed, &mut frame_started, &clock) {
                        break;
                    }
                    continue;
                }
                let preview = (!levels.is_empty() && first_layer_per_pixel.is_none_or(|per_pixel| per_pixel * pixels > PREVIEW_WORTHWHILE)).then_some(PREVIEW_WORTHWHILE);
                let (mut previews, mut shown, mut layer_started) = (0, false, Instant::now());
                let render = Render::new(&plane, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).resume(&mut samples).preview(preview).preview_blocks(&levels)
                    .gpu(gpu.as_deref().map(|gpu| (gpu, join_after))).fast_gpu(fast_gpu).f32_gpu(f32_frames && !last).cancel_with(&thread_cancel);
                let render = if required == Readiness::FULL_RESOLUTION && !last { render.colors_only() } else { render };
                render.run(|colors, pass| {
                        if pass == 1 {
                            first_layer_per_pixel = Some(layer_started.elapsed() / pixels.max(1));
                        }
                        layer_started = Instant::now();
                        let readiness = if pass == 0 { Readiness::Blocks(levels.get(previews).copied().unwrap_or(1)) } else if pass >= supersampling_amount { Readiness::SuperSampled } else { Readiness::Layers(pass) };
                        if shown || readiness < required {
                            previews += usize::from(pass == 0);
                            return;
                        }
                        shown = true;
                        let _ = sender.send(JobMessage::FlightFrame { elapsed, plane: plane.clone(), max_iterations: m.max_iterations, palette_scale: m.palette_scale, preview: previews, levels: levels.clone(), scaled: None });
                        queued.fetch_add(1, Ordering::SeqCst);
                        let _ = sender.send(JobMessage::Pass { colors: spare_colors.copy(colors), plane: plane.clone(), pass, total: supersampling_amount, layer_time: Duration::ZERO, whole: true });
                        thread_cancel.store(true, Ordering::SeqCst); //Complete enough: on to the next frame
                    });
                if last || !next_frame(shown, &mut scale, auto, &mut join_after, &mut elapsed, &mut frame_started, &clock) {
                    break; //The frame at the destination is there, or the window stopped the job
                }
            }
            let _ = samples_sender.send(samples);
            let _ = sender.send(JobMessage::Done { c: plane, m, message: String::new(), elapsed: started.elapsed(), completed: false, reported: true });
        });
        self.flight_job = true;
        self.bars = None;
        self.job = Some(receiver);
        self.progress = Some(Progress { pass: 0, total: supersampling_amount, cancel, stop: Some(stop) });
    }
}

/// The size of a frame of `scale` (the share of the pixels across) of a `width` x `height` image, or `None` at full resolution
fn scaled_size(width: usize, height: usize, scale: f64) -> Option<(usize, usize)> {
    let size = (((width as f64 * scale).round() as usize).max(1), ((height as f64 * scale).round() as usize).max(1));
    (size.0 < width || size.1 < height).then_some(size)
}

/// What moves the clock of the flight job on after a frame (see `next_frame`): the time between two refreshes of the screen, the duration of the flight, and the
/// frames it may render ahead with the count of those waiting in the window, and the stop flag of the job
struct Clock {
    frame_interval: Duration,
    duration: Duration,
    ahead: usize,
    queued: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}

/// After a frame of the flight job: the scale of the next Auto frame from the time this one took, the join delay of the GPU, the wait for the screen (or for the
/// window, with frames ahead) and the flight clock, see the module. Returns whether there is a next frame: not when the window stopped this one (`shown` is false)
fn next_frame(shown: bool, scale: &mut f64, auto: Option<Duration>, join_after: &mut Duration, elapsed: &mut Duration, frame_started: &mut Instant, clock: &Clock) -> bool {
    if !shown {
        return false;
    }
    let took = frame_started.elapsed();
    if let Some(target) = auto {
        *scale = auto_scale(*scale, took, target);
    }
    if took > 4 * GPU_JOIN_AFTER {
        *join_after = Duration::ZERO; //(Measured before the wait for the screen below)
    }
    let step = clock.frame_interval.min(MAX_WAITING_STEP);
    if clock.ahead == 0 {
        //No more frames than the screen shows; the clock moves on by the time the frame took, at most one step
        if let Some(wait) = clock.frame_interval.checked_sub(frame_started.elapsed()) {
            std::thread::sleep(wait);
        }
        *elapsed = (*elapsed + frame_started.elapsed().min(step)).min(clock.duration);
    } else {
        //Frames ahead: the next frame shows one refresh later, and the job goes on with it at once, unless the window already has `ahead` frames waiting
        *elapsed = (*elapsed + step).min(clock.duration);
        while clock.queued.load(Ordering::SeqCst) > clock.ahead && !clock.stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_micros(500));
        }
    }
    *frame_started = Instant::now();
    true
}

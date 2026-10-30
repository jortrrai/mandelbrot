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

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use super::points::MAX_WAITING_STEP;
use super::render_job::{JobMessage, Progress, RenderKind, GPU_JOIN_AFTER, PREVIEW_WORTHWHILE};
use super::flight::{auto_block, FlightRendering};
use super::{temporal, MandelbrotApp, Readiness};
use crate::complex_plane::ComplexPlane;
use crate::rendering::{self, Render};

impl MandelbrotApp {
    /// Starts the job that renders the frames of the flight under way, from where its clock is, see the module. Its frames arrive as `JobMessage::FlightFrame`
    /// followed by the `Pass` of the frame; it stops when it has rendered the frame at the destination (the window then arrives, see `points::fly_at`)
    #[allow(clippy::too_many_lines)] // The settings of the job and its loop over the frames, which share their state
    pub(super) fn start_flight_job(&mut self, frame_interval: Duration) {
        let Some(flight) = &mut self.flight else { return };
        let (path, mut elapsed, mut limits, palettes) = (flight.flight.clone(), flight.elapsed, flight.limits.take(), flight.palettes);
        self.cancel_thumbnail_job();
        self.cancel_neighborhood_job();
        self.interrupt_progressive_render();
        (self.frame_ready, self.frame_accepted, self.buffer_is_complete, self.layer_shown) = (false, false, false, false);
        let previous_samples = self.samples.take();
        let (samples_sender, samples_receiver) = mpsc::channel();
        self.samples = Some(samples_receiver);
        let (sender, receiver) = mpsc::channel();
        let spare_colors = self.spare_colors.clone();
        let (mut required, mut levels) = (self.flight_readiness(), self.preview_levels(RenderKind::FlightFrame));
        self.preview_levels.clone_from(&levels);
        //With Auto: the frame time to keep, and the time a full-resolution frame takes, estimated from the frames so far (see `flight::auto_block`)
        //(With even pacing the frames have to fit their refreshes with the margin of `pacing`, or every frame would take one refresh more)
        let margin = if self.frames_ahead > 0 && self.even_pacing { PACING_MARGIN } else { 1.0 };
        let auto = match self.flight_rendering {
            FlightRendering::Auto(fps) => Some(Duration::from_secs_f64(1.0 / f64::from(fps.max(1)) / margin)),
            _ => None,
        };
        let (mut block, mut full_frame) = (2, None::<Duration>);
        let adaptive = self.adaptive_supersampling.then(rendering::AdaptiveSampling::default);
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        let mut m = self.m.clone();
        if !self.auto_iterations {
            m.max_iterations = self.max_iterations_input;
        }
        let (supersampling_amount, coloring_function) = (self.supersampling_amount, self.coloring_function);
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let pixels = u32::try_from(width * height).unwrap_or(u32::MAX);
        let mut first_layer_per_pixel = self.first_layer_per_pixel;
        let (gpu, fast_gpu, f32_frames, blend_frames) = (self.gpu.clone().filter(|_| self.use_gpu), self.fast_gpu, self.f32_frames, self.blend_frames);
        //`cancel` stops the frame that is rendered (the job sets it when the frame is complete enough), `stop` the whole job (the window sets it, then `cancel`)
        let (cancel, stop) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (thread_cancel, thread_stop) = (Arc::clone(&cancel), Arc::clone(&stop));
        //Frames ahead: the frames the job has sent that the window has not shown yet (see `render_job::next_job_message`)
        let (ahead, even) = (self.frames_ahead, self.even_pacing);
        self.frames_queued = Arc::new(AtomicUsize::new(0));
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
            //With Blend frames: the image of the frames so far, and the sample (jitter) the next frame takes, see `temporal`
            let (mut history, mut next_sample) = (None, 0_u8);
            //With frames ahead and even pacing: the refreshes every frame stays on the screen, and the render time of the frames so far (see `pacing`)
            let (mut refreshes, mut render_time) = (1_u32, None::<Duration>);
            while !thread_stop.load(Ordering::SeqCst) {
                thread_cancel.store(false, Ordering::SeqCst);
                if thread_stop.load(Ordering::SeqCst) {
                    break; //The window stopped the job between the check and the reset
                }
                plane.set_view(&path.camera_at(elapsed).view());
                if let Some(settings) = &auto_settings {
                    //The limit of the zoom of this frame (see `points::FlightLimits`)
                    let scale = plane.get_scale();
                    m.max_iterations = limits.as_mut().map_or_else(|| settings.initial_limit(scale), |limits| limits.limit_at(scale, settings));
                }
                //The color cycle of this frame, on the way from the start to the place's (see `points::PointFlight::palettes`)
                m.palette = palettes.0.blend(palettes.1, path.share_of_path(elapsed));
                if let Some(target) = auto {
                    //The block of this frame: from the estimate, or half resolution for the first frame
                    block = full_frame.map_or(block, |full| auto_block(full, target, block));
                    (required, levels) = if block > 1 { (Readiness::Blocks(block), vec![block]) } else { (Readiness::FULL_RESOLUTION, Vec::new()) };
                }
                let preview = (!levels.is_empty() && first_layer_per_pixel.is_none_or(|per_pixel| per_pixel * pixels > PREVIEW_WORTHWHILE)).then_some(PREVIEW_WORTHWHILE);
                let (mut previews, mut shown, mut layer_started) = (0, false, Instant::now());
                let render = Render::new(&plane, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).resume(&mut samples).preview(preview).preview_blocks(&levels)
                    .gpu(gpu.as_deref().map(|gpu| (gpu, join_after))).fast_gpu(fast_gpu).f32_gpu(f32_frames && !path.arrived(elapsed)).cancel_with(&thread_cancel);
                let last = path.arrived(elapsed);
                let blending = blend_frames && required == Readiness::FULL_RESOLUTION && !last;
                let render = match (required == Readiness::FULL_RESOLUTION && !last, blending) {
                    (true, true) => render.colors_only_of_sample(next_sample),
                    (true, false) => render.colors_only(),
                    (false, _) => render,
                };
                next_sample = next_sample.wrapping_add(1) % 64;
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
                        let _ = sender.send(JobMessage::FlightFrame { elapsed, plane: plane.clone(), max_iterations: m.max_iterations, palette: m.palette, preview: previews, levels: levels.clone(), refreshes });
                        queued.fetch_add(1, Ordering::SeqCst);
                        let colors = if blending {
                            let mut shown = spare_colors.copy(&[]);
                            temporal::blend(&mut history, &plane, colors, width, height, &mut shown);
                            shown
                        } else {
                            spare_colors.copy(colors)
                        };
                        let _ = sender.send(JobMessage::Pass { colors, plane: plane.clone(), pass, total: supersampling_amount, layer_time: Duration::ZERO, whole: true });
                        thread_cancel.store(true, Ordering::SeqCst); //Complete enough: on to the next frame
                    });
                if auto.is_some() && shown {
                    //What a full-resolution frame would take, from this frame and its pixels, smoothed over a few frames
                    let full = frame_started.elapsed().mul_f64((block * block) as f64);
                    full_frame = Some(full_frame.map_or(full, |before| before.mul_f64(0.5) + full.mul_f64(0.5)));
                }
                if frame_started.elapsed() > 4 * GPU_JOIN_AFTER {
                    join_after = Duration::ZERO; //(Measured before the wait for the screen below)
                }
                if !shown || last {
                    break; //Stopped by the window, or the frame at the destination is there
                }
                if ahead == 0 {
                    //No more frames than the screen shows; the clock moves on by the time the frame took, at most one step
                    if let Some(wait) = frame_interval.checked_sub(frame_started.elapsed()) {
                        std::thread::sleep(wait);
                    }
                    elapsed = (elapsed + frame_started.elapsed().min(frame_interval.min(MAX_WAITING_STEP))).min(path.duration());
                } else {
                    //Frames ahead: the clock moves on by the time this frame will be on the screen, its refreshes (at most one step), and the job goes on with the next
                    //frame at once, unless the window already has `ahead` frames waiting
                    let shown_for = frame_interval * if even { refreshes } else { 1 };
                    elapsed = (elapsed + shown_for.min(MAX_WAITING_STEP)).min(path.duration());
                    let took = frame_started.elapsed();
                    render_time = Some(render_time.map_or(took, |before| before.mul_f64(0.7) + took.mul_f64(0.3)));
                    if even {
                        refreshes = pacing(render_time.unwrap_or(took), frame_interval, refreshes);
                    }
                    while queued.load(Ordering::SeqCst) > ahead && !thread_stop.load(Ordering::SeqCst) {
                        std::thread::sleep(Duration::from_micros(500));
                    }
                }
                frame_started = Instant::now();
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

/// The time to spare that even pacing wants for a frame: 15% of its refreshes
const PACING_MARGIN: f64 = 1.15;

/// The refreshes every frame stays on the screen with even pacing: the fewest whose time fits `render_time` (the render time of the frames so far) with 15% to spare,
/// so the frames keep up; fewer than `current` only with 30% to spare, so a render time near a boundary does not switch back and forth
pub(super) fn pacing(render_time: Duration, refresh: Duration, current: u32) -> u32 {
    let needed = |spare: f64| (render_time.as_secs_f64() * spare / refresh.as_secs_f64().max(1e-6)).ceil().max(1.0) as u32;
    let (at_least, relaxed) = (needed(PACING_MARGIN), needed(1.3));
    if at_least > current { at_least } else { relaxed.min(current) }
}

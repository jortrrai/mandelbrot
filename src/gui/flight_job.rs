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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use super::points::MAX_WAITING_STEP;
use super::render_job::{JobMessage, Progress, RenderKind, GPU_JOIN_AFTER, PREVIEW_WORTHWHILE};
use super::{MandelbrotApp, Readiness};
use crate::complex_plane::ComplexPlane;
use crate::rendering::{self, Render};

impl MandelbrotApp {
    /// Starts the job that renders the frames of the flight under way, from where its clock is, see the module. Its frames arrive as `JobMessage::FlightFrame`
    /// followed by the `Pass` of the frame; it stops when it has rendered the frame at the destination (the window then arrives, see `points::fly_at`)
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
        let required = self.flight_readiness();
        let levels = self.preview_levels(RenderKind::FlightFrame);
        self.preview_levels.clone_from(&levels);
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
                let preview = (!levels.is_empty() && first_layer_per_pixel.is_none_or(|per_pixel| per_pixel * pixels > PREVIEW_WORTHWHILE)).then_some(PREVIEW_WORTHWHILE);
                let (mut previews, mut shown, mut layer_started) = (0, false, Instant::now());
                let render = Render::new(&plane, &m, width, height, coloring_function).supersampling(supersampling_amount).adaptive(adaptive).resume(&mut samples).preview(preview).preview_blocks(&levels)
                    .gpu(gpu.as_deref().map(|gpu| (gpu, join_after))).fast_gpu(fast_gpu).f32_gpu(f32_frames && !path.arrived(elapsed)).cancel_with(&thread_cancel);
                let last = path.arrived(elapsed);
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
                        let _ = sender.send(JobMessage::FlightFrame { elapsed, plane: plane.clone(), max_iterations: m.max_iterations, palette_scale: m.palette_scale, preview: previews });
                        let _ = sender.send(JobMessage::Pass { colors: spare_colors.copy(colors), plane: plane.clone(), pass, total: supersampling_amount, layer_time: Duration::ZERO, whole: true });
                        thread_cancel.store(true, Ordering::SeqCst); //Complete enough: on to the next frame
                    });
                if frame_started.elapsed() > 4 * GPU_JOIN_AFTER {
                    join_after = Duration::ZERO; //(Measured before the wait for the screen below)
                }
                if !shown || last {
                    break; //Stopped by the window, or the frame at the destination is there
                }
                //No more frames than the screen shows; the clock moves on by the time the frame took, at most one step
                if let Some(wait) = frame_interval.checked_sub(frame_started.elapsed()) {
                    std::thread::sleep(wait);
                }
                elapsed = (elapsed + frame_started.elapsed().min(frame_interval.min(MAX_WAITING_STEP))).min(path.duration());
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

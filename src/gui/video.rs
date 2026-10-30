//! Videos of flights (issue #67): the last flight rendered offline, frame by frame, at a fixed frame rate and the full quality of saved images, however long a
//! frame takes.
//!
//! A background job puts the camera at `n / fps` along the flight path for frame `n`, renders it at the saved image size and quality (the GPU helps as in the
//! window), and writes it at once: as `frame-00001.png`, ... in a folder of `saved/`, and, when `ffmpeg` is installed, also to `saved/<name>.mp4` (H.264,
//! piped frame by frame). Only one frame is in memory at a time, so long or 4K videos never fill the memory. The iteration limit follows the zoom like a flight
//! with auto iterations: from the limit at the start to the limit of the place. Frames are sharp (no motion blur). An interrupted video keeps the frames written.

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use eframe::egui;

use super::flight::{Camera, Flight, Pace, Route};
use super::{colors_to_rgb_bytes, MandelbrotApp};
use crate::complex_plane::ComplexPlane;
use crate::rendering::{self, Render};

/// The frame rates of the list
pub(super) const FRAME_RATES: [u32; 3] = [24, 30, 60];

/// The flight a video is made of: from where to where, at which pace, and the iteration limits at both ends
#[derive(Clone, Debug)]
pub(super) struct VideoFlight {
    pub(super) from: Camera,
    pub(super) to: Camera,
    pub(super) pace: Pace,
    pub(super) route: Route,
    pub(super) name: String,
    /// The color cycle at the start and at the place, blended on the way like a flight's (see `points::PointFlight::palettes`)
    pub(super) palettes: (crate::mandelbrot_set::ColorCycle, crate::mandelbrot_set::ColorCycle),
    pub(super) limits: (u32, u32),
}

/// A video job: its progress (frames written of all) and cancel flag, and the receiver of its final message
pub(super) struct VideoJob {
    done: Arc<AtomicUsize>,
    total: usize,
    cancel: Arc<AtomicBool>,
    started: Instant,
    result: mpsc::Receiver<String>,
}

/// The iteration limit of a frame at `scale`: the limits at the ends scaled like the zoom rule of auto iterations (`initial_limit`), blended by the zoom, as a
/// flight with auto iterations does (see `points::FlightLimits`)
fn limit_at(scale: f64, flight: &VideoFlight, settings: &crate::auto_iterations::AutoIterationSettings) -> u32 {
    let factor = |limit: u32, scale: f64| f64::from(limit) / f64::from(settings.initial_limit(scale).max(1));
    let (from, to) = (factor(flight.limits.0, flight.from.scale), factor(flight.limits.1, flight.to.scale));
    let t = if (flight.to.scale / flight.from.scale).ln().abs() < 1e-12 { 1.0 } else { ((scale / flight.from.scale).ln() / (flight.to.scale / flight.from.scale).ln()).clamp(0.0, 1.0) };
    let limit = f64::from(settings.initial_limit(scale)) * from.powf(1.0 - t) * to.powf(t);
    (limit.round() as u32).clamp(1, 1_000_000)
}

/// `ffmpeg` encoding raw RGB frames of `width` x `height` at `fps` from its standard input into `path` (H.264), if `ffmpeg` can be started
fn start_ffmpeg(width: usize, height: usize, fps: u32, path: &str) -> Option<Child> {
    Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path])
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()
}

impl MandelbrotApp {
    /// Starts the video of the last flight, see the module
    pub(super) fn start_video(&mut self) {
        let Some(flight) = self.last_flight.clone() else { return };
        let resolution = match self.save_size.resolve(self.default_save_resolution(), self.view_resolution()) {
            Ok(resolution) => resolution,
            Err(reason) => {
                self.log(format!("Could not render the video: {reason}"));
                return;
            }
        };
        let (width, height, fps) = (resolution.width, resolution.height, self.video_fps);
        let (path, _) = Flight::with_route(flight.from, flight.to, flight.pace, flight.route);
        let total = (path.duration().as_secs_f64() * f64::from(fps)).ceil() as usize + 1;
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let slug: String = flight.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
        let name = format!("video-{stamp}-{slug}");
        let (mut m, coloring_function, mapping) = (self.m.clone(), self.coloring_function, self.p.color_channel_mapping);
        let (samples, adaptive) = (self.image_supersampling_amount, self.adaptive_supersampling.then(rendering::AdaptiveSampling::default));
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        let (gpu, fast_gpu) = (self.gpu.clone().filter(|_| self.use_gpu), self.fast_gpu);
        let (done, cancel) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)));
        let (thread_done, thread_cancel) = (Arc::clone(&done), Arc::clone(&cancel));
        let (sender, result) = mpsc::channel();
        let destination = flight.name.clone();
        let saved = self.video_folder.clone();
        std::thread::spawn(move || {
            let folder = saved.join(&name);
            if let Err(err) = std::fs::create_dir_all(&folder) {
                let _ = sender.send(format!("Could not render the video: {err}"));
                return;
            }
            let mp4 = saved.join(format!("{name}.mp4")).display().to_string();
            let mut ffmpeg = start_ffmpeg(width, height, fps, &mp4);
            let mut plane = ComplexPlane::new(width, height);
            for frame in 0..total {
                if thread_cancel.load(Ordering::Relaxed) {
                    break;
                }
                let camera = path.camera_at(Duration::from_secs_f64(frame as f64 / f64::from(fps)));
                plane.set_view(&camera.view());
                if let Some(settings) = &auto_settings {
                    m.max_iterations = limit_at(camera.scale, &flight, settings);
                }
                m.palette = flight.palettes.0.blend(flight.palettes.1, path.share_of_path(Duration::from_secs_f64(frame as f64 / f64::from(fps))));
                let mut colors = Vec::new();
                let completed = Render::new(&plane, &m, width, height, coloring_function).supersampling(samples).adaptive(adaptive)
                    .gpu(gpu.as_deref().map(|gpu| (gpu, Duration::ZERO))).fast_gpu(fast_gpu).cancel_with(&thread_cancel).run(|layer, _| colors = layer.to_vec());
                if !completed {
                    break;
                }
                let rgb = colors_to_rgb_bytes(&colors, mapping);
                let written = std::fs::File::create(folder.join(format!("frame-{:05}.png", frame + 1))).and_then(|file| {
                    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
                    encoder.set_color(png::ColorType::Rgb);
                    encoder.set_depth(png::BitDepth::Eight);
                    encoder.write_header().and_then(|mut writer| writer.write_image_data(&rgb)).map_err(std::io::Error::other)
                });
                if let Err(err) = written {
                    let _ = sender.send(format!("Could not write frame {} of the video: {err}", frame + 1));
                    return;
                }
                if let Some(child) = &mut ffmpeg {
                    if child.stdin.as_mut().is_none_or(|stdin| stdin.write_all(&rgb).is_err()) {
                        ffmpeg = None; //ffmpeg stopped: the PNG frames go on
                    }
                }
                thread_done.store(frame + 1, Ordering::Relaxed);
            }
            let frames = thread_done.load(Ordering::Relaxed);
            let encoded = ffmpeg.and_then(|mut child| {
                drop(child.stdin.take()); //The end of the frames
                child.wait().ok().filter(std::process::ExitStatus::success)
            }).is_some();
            let complete = if frames == total { "" } else { " (stopped: incomplete)" };
            let video = if encoded { format!(" and {mp4}") } else { String::from(" (ffmpeg was not found, so there is no .mp4; the frames can be encoded with it later)") };
            let _ = sender.send(format!("Video of the flight: {frames} of {total} frames at {fps} FPS{complete} in {}{video}", folder.display()));
        });
        self.log(format!("Rendering a video of the flight to {destination}: {total} frames of {width} x {height} at {fps} FPS"));
        self.video_job = Some(VideoJob { done, total, cancel, started: Instant::now(), result });
    }

    /// Takes the final message of the video job
    pub(super) fn poll_video(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.video_job else { return };
        ctx.request_repaint_after(Duration::from_millis(200));
        if let Ok(message) = job.result.try_recv() {
            self.video_job = None;
            self.log(message);
        }
    }

    /// The video controls: the frame rate, the button, and the progress of a video under way
    pub(super) fn video_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(job) = &self.video_job {
                let done = job.done.load(Ordering::Relaxed);
                let elapsed = job.started.elapsed().as_secs_f64();
                let left = if done > 0 { elapsed / done as f64 * (job.total - done) as f64 } else { 0.0 };
                if ui.button("Cancel video").clicked() {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                ui.add(egui::ProgressBar::new(done as f32 / job.total.max(1) as f32).text(format!("Video: {done} of {} frames, {left:.0} s left", job.total)));
                return;
            }
            let name = self.last_flight.as_ref().map_or_else(|| String::from("no flight yet"), |flight| flight.name.clone());
            if ui.add_enabled(self.last_flight.is_some(), egui::Button::new("Render video of the last flight")).on_hover_text(format!("The last flight ({name}) as a video: every frame rendered offline at the saved image size and quality, at a fixed frame rate, written as PNG frames to saved/ (and as an .mp4 when ffmpeg is installed)")).clicked() {
                self.start_video();
            }
            egui::ComboBox::from_id_salt("video_fps").selected_text(format!("{} FPS", self.video_fps)).width(70.0).show_ui(ui, |ui| {
                for fps in FRAME_RATES {
                    ui.selectable_value(&mut self.video_fps, fps, format!("{fps} FPS"));
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_of_a_video_frame_goes_from_the_limit_at_the_start_to_that_of_the_place() {
        let settings = crate::auto_iterations::AutoPreset::Balanced.settings();
        let flight = VideoFlight { from: Camera { x: -0.75, y: 0.0, scale: 1.0 }, to: Camera { x: -0.7498, y: 0.0203, scale: 6.4e-5 }, pace: Pace::Fast, route: Route::Shortest, name: String::from("Seahorse"), palettes: (crate::mandelbrot_set::ColorCycle::CYCLE, crate::mandelbrot_set::ColorCycle::CYCLE), limits: (300, 5000) };
        assert_eq!(limit_at(1.0, &flight, &settings), 300);
        assert_eq!(limit_at(6.4e-5, &flight, &settings), 5000);
        let middle = limit_at(8e-3, &flight, &settings);
        assert!(middle > 300 && middle < 5000, "{middle}");
    }
}

//! Videos of flights (issue #67): the last flight rendered offline, frame by frame, at a fixed frame rate and the full quality of saved images, however long a
//! frame takes.
//!
//! A background job puts the camera at `n / fps × speed` along the flight path for frame `n` (*Video speed*: 100% is the speed of the flight, 10% a ten times
//! slower video), renders it at the saved image size and quality (the GPU helps as in the window), and hands it on: one thread writes it as `frame-00001.png`, ...
//! to a folder of `saved/` (with *PNG frames*, off by default), another pipes it to `ffmpeg`, when it is installed, which encodes the video in the chosen
//! `VideoQuality` for the chosen `Playback`. Several frames render at once (`FRAMES_AT_ONCE`), and the renderers go on while the writers write the last
//! ones (at most `FRAMES_WAITING` frames wait), so neither the CPU threads nor the GPU wait for the encoder or for the end of a layer. The iteration limit follows the zoom like a flight with auto iterations: from the limit at the start to the limit of
//! the place. Frames are sharp (no motion blur). An interrupted video keeps the frames written.

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};

use eframe::egui;

use super::flight::{Camera, Flight, Pace, Route};
use super::{colors_to_rgb_bytes, MandelbrotApp};
use crate::complex_plane::ComplexPlane;
use crate::rendering::{self, Render};

/// The frame rates of the list
pub(super) const FRAME_RATES: [u32; 7] = [24, 30, 60, 90, 120, 144, 240];

/// The range of the video speed, in percent of the speed of the flight
pub(super) const SPEEDS: std::ops::RangeInclusive<u16> = 1..=400;

/// How many rendered frames may wait for the writers of the video (the PNG frames and ffmpeg) before the renderers wait for them
const FRAMES_WAITING: usize = 2;

/// The range of the frames a video renders at once, see `MandelbrotApp::video_frames_at_once`
pub(super) const FRAMES_AT_ONCE: std::ops::RangeInclusive<usize> = 1..=4;

/// The quality of the video file: how much of the rendered detail the encoder keeps. The tiers keep a fixed quality (the encoder's quantizer), so a busy view
/// gets the bits it needs and a calm one stays small. Measured on 60 frames of a detailed 1080p zoom at 60 FPS (the mean error of a color channel, in levels of
/// 255, on the GPU's encoder with full color): High 3.7, Very High 2.7, Ultra High 1.9, Near Lossless 0.9
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VideoQuality {
    High,
    VeryHigh,
    UltraHigh,
    NearLossless,
    /// FFV1 in an .mkv: exactly the rendered pixels, bit for bit; very large files, too large for most players to play smoothly
    Lossless,
}

impl VideoQuality {
    /// Every tier, in the order of the list
    pub(super) const ALL: [VideoQuality; 5] = [VideoQuality::High, VideoQuality::VeryHigh, VideoQuality::UltraHigh, VideoQuality::NearLossless, VideoQuality::Lossless];

    /// The name in the list
    pub(super) const fn name(self) -> &'static str {
        match self {
            VideoQuality::High => "High",
            VideoQuality::VeryHigh => "Very High",
            VideoQuality::UltraHigh => "Ultra High",
            VideoQuality::NearLossless => "Near Lossless",
            VideoQuality::Lossless => "Lossless",
        }
    }

    /// The quantizer of the GPU's encoder (fixed, `-rc constqp`), and the `-crf` of the CPU's encoders, for the lossy tiers
    const fn quantizers(self) -> (&'static str, &'static str) {
        match self {
            VideoQuality::High => ("24", "18"),
            VideoQuality::VeryHigh => ("20", "14"),
            VideoQuality::UltraHigh => ("16", "10"),
            VideoQuality::NearLossless | VideoQuality::Lossless => ("8", "6"),
        }
    }

    /// The bits a pixel of a frame takes on a busy view (measured on the detailed zoom above; calm views take far less), for the estimate of the size
    const fn bits_per_pixel(self, playback: Playback) -> f64 {
        match (self, playback) {
            (VideoQuality::Lossless, _) => 8.2,
            (VideoQuality::High, Playback::Computers) => 0.65,
            (VideoQuality::VeryHigh, Playback::Computers) => 1.1,
            (VideoQuality::UltraHigh, Playback::Computers) => 1.75,
            (VideoQuality::NearLossless, Playback::Computers) => 3.4,
            (VideoQuality::High, _) => 0.57,
            (VideoQuality::VeryHigh, _) => 0.9,
            (VideoQuality::UltraHigh, _) => 1.3,
            (VideoQuality::NearLossless, _) => 2.3,
        }
    }
}

/// Where the video has to play, which decides its color resolution and codec (lossless videos are always FFV1)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Playback {
    /// HEVC (on an NVIDIA GPU's encoder; H.264 on the CPU otherwise) in 10 bits with the full color resolution (4:4:4): the best colors; computers play it
    /// (VLC, most players), phones and TVs do not
    Computers,
    /// HEVC Main 10 (10 bits, half the color resolution, 4:2:0), which phones, TVs and browsers decode in hardware. Detail of single pixels in strong colors
    /// gets softer
    Phones,
    /// H.264 in 8 bits with half the color resolution: plays on anything, with the softest colors
    Everywhere,
}

impl Playback {
    /// Every choice, in the order of the list (the default first)
    pub(super) const ALL: [Playback; 3] = [Playback::Computers, Playback::Phones, Playback::Everywhere];

    /// The name in the list
    pub(super) const fn name(self) -> &'static str {
        match self {
            Playback::Computers => "Computers",
            Playback::Phones => "Phones and TVs",
            Playback::Everywhere => "Everywhere",
        }
    }
}

/// The extension of the file of a video of `quality`
const fn extension(quality: VideoQuality) -> &'static str {
    match quality {
        VideoQuality::Lossless => "mkv",
        _ => "mp4",
    }
}

/// The arguments of ffmpeg that encode RGB frames at `fps` in `quality` for `playback` (with `nvenc`: on the GPU's HEVC encoder). The colors are converted
/// with the BT.709 matrix and tagged so, so players show them as they were rendered; half-resolution color is made with a sharp filter. A keyframe every
/// half second, B-frames and the index at the start of the file, as video sites ask for
fn arguments(quality: VideoQuality, playback: Playback, nvenc: bool, fps: u32) -> Vec<String> {
    const FULL: &str = "scale=out_color_matrix=bt709:out_range=tv,format=";
    const HALF: &str = "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,format=";
    if quality == VideoQuality::Lossless {
        return ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"].map(String::from).to_vec();
    }
    let (qp, crf) = quality.quantizers();
    let nvenc_hevc = |format: &str| vec![format.to_string(), "-c:v".into(), "hevc_nvenc".into(), "-preset".into(), "p7".into(), "-tune".into(), "hq".into(), "-rc".into(), "constqp".into(), "-qp".into(), qp.into(), "-tag:v".into(), "hvc1".into()];
    let mut arguments = vec![String::from("-vf")];
    arguments.extend(match playback {
        Playback::Computers if nvenc => nvenc_hevc(&format!("{FULL}yuv444p16le")),
        Playback::Computers => vec![format!("{FULL}yuv444p10le"), "-c:v".into(), "libx264".into(), "-crf".into(), crf.into(), "-preset".into(), "medium".into()],
        Playback::Phones if nvenc => [nvenc_hevc(&format!("{HALF}p010le")), vec!["-profile:v".into(), "main10".into()]].concat(),
        Playback::Phones => vec![format!("{HALF}yuv420p10le"), "-c:v".into(), "libx265".into(), "-crf".into(), crf.into(), "-preset".into(), "medium".into(), "-tag:v".into(), "hvc1".into(), "-x265-params".into(), "log-level=error".into()],
        Playback::Everywhere => vec![format!("{HALF}yuv420p"), "-c:v".into(), "libx264".into(), "-crf".into(), crf.into(), "-preset".into(), "medium".into(), "-profile:v".into(), "high".into()],
    });
    arguments.extend(["-g".into(), (fps / 2).max(1).to_string(), "-bf".into(), "2".into(), "-movflags".into(), "+faststart".into()]);
    arguments.extend(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"].map(String::from));
    arguments
}

/// The estimate of a video shown next to its controls: its frames, its length, and the size it takes at most on busy views (see `bits_per_pixel`)
pub(super) fn estimate(duration: Duration, fps: u32, speed: u16, (width, height): (usize, usize), quality: VideoQuality, playback: Playback) -> (usize, Duration, f64) {
    let frames = frame_count(duration, fps, speed);
    let bytes = frames as f64 * (width * height) as f64 * quality.bits_per_pixel(playback) / 8.0;
    (frames, Duration::from_secs_f64(frames as f64 / f64::from(fps)), bytes)
}

/// The smallest video the GPU's HEVC encoder takes (NVENC needs at least about 129 x 33 pixels); a smaller one is encoded on the CPU
const NVENC_SMALLEST: (usize, usize) = (160, 64);

/// Whether ffmpeg can encode with the GPU's HEVC encoder in 10-bit 4:4:4 (an NVIDIA GPU with NVENC), found out once by encoding a tiny clip
fn nvenc_works() -> bool {
    static WORKS: OnceLock<bool> = OnceLock::new();
    *WORKS.get_or_init(|| {
        Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.04", "-vf", "format=yuv444p16le", "-c:v", "hevc_nvenc", "-f", "null", "-"])
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|status| status.success())
    })
}

/// The time along the flight of frame `frame` of a video at `fps` and `speed` percent of the flight's speed
fn frame_time(frame: usize, fps: u32, speed: u16) -> Duration {
    Duration::from_secs_f64(frame as f64 / f64::from(fps) * f64::from(speed.max(1)) / 100.0)
}

/// The frames of a video of a flight of `duration` at `fps` and `speed` percent: from the start to the place, both included
fn frame_count(duration: Duration, fps: u32, speed: u16) -> usize {
    (duration.as_secs_f64() * 100.0 / f64::from(speed.max(1)) * f64::from(fps)).ceil() as usize + 1
}

/// The flight a video is made of: from where to where, at which pace, and the iteration limits at both ends
#[derive(Clone, Debug)]
pub(super) struct VideoFlight {
    pub(super) from: Camera,
    pub(super) to: Camera,
    pub(super) pace: Pace,
    /// The route of the flight, and the speed of the constant pace (see `flight::Flight::with`)
    pub(super) route: Route,
    pub(super) constant_speed: f64,
    /// The speed factor of the other paces (see `flight::Flight::with`)
    pub(super) speed: f64,
    pub(super) name: String,
    pub(super) limits: (u32, u32),
    /// The color cycle at the start and at the place, blended on the way like a flight's (see `points::PointFlight::palettes`)
    pub(super) palettes: (crate::mandelbrot_set::ColorCycle, crate::mandelbrot_set::ColorCycle),
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

/// `ffmpeg` encoding raw RGB frames of `width` x `height` at `fps` from its standard input into `path` in `quality` for `playback`, if `ffmpeg` can be started
fn start_ffmpeg((width, height): (usize, usize), fps: u32, path: &str, quality: VideoQuality, playback: Playback) -> Option<Child> {
    let (size, rate) = (format!("{width}x{height}"), fps.to_string());
    let nvenc = width >= NVENC_SMALLEST.0 && height >= NVENC_SMALLEST.1 && nvenc_works();
    let mut command = Command::new("ffmpeg");
    command.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-r", &rate, "-i", "-"]).args(arguments(quality, playback, nvenc, fps)).arg(path);
    command.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()
}

/// Writes `rgb` (`width` x `height`) as a PNG file at `path`, with fast compression (a video has many frames; ffmpeg makes the small file)
fn write_png(path: &std::path::Path, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.write_header().and_then(|mut writer| writer.write_image_data(rgb)).map_err(std::io::Error::other)
}

impl MandelbrotApp {
    /// Starts the video of the last flight, see the module
    #[allow(clippy::too_many_lines)] // The settings of the job, its two writers and its loop over the frames
    pub(super) fn start_video(&mut self) {
        let Some(flight) = self.last_flight.clone() else { return };
        let resolution = match self.save_size.resolve(self.default_save_resolution(), self.view_resolution()) {
            Ok(resolution) => resolution,
            Err(reason) => {
                self.log(format!("Could not render the video: {reason}"));
                return;
            }
        };
        let (width, height, fps, speed, quality, playback, pngs) = (resolution.width, resolution.height, self.video_fps, self.video_speed, self.video_quality, self.video_playback, self.video_pngs);
        let at_once = self.video_frames_at_once.clamp(1, *FRAMES_AT_ONCE.end());
        let path = Flight::with(flight.from, flight.to, flight.pace, flight.route, flight.constant_speed, flight.speed);
        let total = frame_count(path.duration(), fps, speed);
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let slug: String = flight.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
        let name = format!("video-{stamp}-{slug}");
        let (m, coloring_function, mapping) = (self.m.clone(), self.coloring_function, self.p.color_channel_mapping);
        let (samples, adaptive, blur) = (self.image_supersampling_amount, self.adaptive_supersampling.then(rendering::AdaptiveSampling::default), self.image_quality_blur);
        let auto_settings = self.auto_iterations.then(|| self.auto_settings.clone());
        let (gpu, fast_gpu) = (self.gpu.clone().filter(|_| self.use_gpu), self.fast_gpu);
        let (done, cancel) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)));
        let (thread_done, thread_cancel) = (Arc::clone(&done), Arc::clone(&cancel));
        let (sender, result) = mpsc::channel();
        let destination = flight.name.clone();
        let saved = self.video_folder.clone();
        std::thread::spawn(move || {
            let folder = saved.join(&name);
            let created = if pngs { std::fs::create_dir_all(&folder) } else { std::fs::create_dir_all(&saved) };
            if let Err(err) = created {
                let _ = sender.send(format!("Could not render the video: {err}"));
                return;
            }
            let video = saved.join(format!("{name}.{}", extension(quality))).display().to_string();
            let mut ffmpeg = start_ffmpeg((width, height), fps, &video, quality, playback);
            let encoding = ffmpeg.is_some();
            //The two writers, each with its own queue of frames; a frame counts as done when both have written it
            let (png_done, encoded) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
            let failure = Arc::new(std::sync::Mutex::new(None::<String>));
            let (png_sender, png_receiver) = mpsc::sync_channel::<(usize, Arc<Vec<u8>>)>(FRAMES_WAITING);
            let (ffmpeg_sender, ffmpeg_receiver) = mpsc::sync_channel::<Arc<Vec<u8>>>(FRAMES_WAITING);
            let png_writer = {
                let (folder, png_done, failure, cancel) = (folder.clone(), Arc::clone(&png_done), Arc::clone(&failure), Arc::clone(&thread_cancel));
                std::thread::spawn(move || {
                    for (frame, rgb) in png_receiver {
                        if let Err(err) = write_png(&folder.join(format!("frame-{:05}.png", frame + 1)), &rgb, width, height) {
                            *failure.lock().unwrap() = Some(format!("Could not write frame {} of the video: {err}", frame + 1));
                            cancel.store(true, Ordering::Relaxed);
                            return;
                        }
                        png_done.store(frame + 1, Ordering::Relaxed);
                    }
                })
            };
            let ffmpeg_writer = {
                let (encoded, mut stdin) = (Arc::clone(&encoded), ffmpeg.as_mut().and_then(|child| child.stdin.take()));
                std::thread::spawn(move || {
                    for rgb in ffmpeg_receiver {
                        if stdin.as_mut().is_some_and(|stdin| stdin.write_all(&rgb).is_err()) {
                            stdin = None; //ffmpeg stopped: the PNG frames go on
                        }
                        encoded.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(stdin); //The end of the frames
                })
            };
            let progress = |png_done: &AtomicUsize, encoded: &AtomicUsize, rendered: usize| {
                let png = if pngs { png_done.load(Ordering::Relaxed) } else { rendered };
                let video = if encoding { encoded.load(Ordering::Relaxed) } else { rendered };
                png.min(video)
            };
            //`at_once` renderers take the next frame as soon as they are free (each with all the CPU threads and the GPU, so one fills the gaps the other leaves:
            //the end of a layer, the GPU's batches); the frames go to the writers in their order
            let (next, (frames_sender, frames_receiver)) = (AtomicUsize::new(0), mpsc::sync_channel::<(usize, Option<Vec<u8>>)>(at_once));
            let mut rendered = 0;
            std::thread::scope(|scope| {
                for _ in 0..at_once {
                    let (frames_sender, next, thread_cancel, m, path, flight, auto_settings, gpu) = (frames_sender.clone(), &next, &thread_cancel, m.clone(), &path, &flight, &auto_settings, &gpu);
                    scope.spawn(move || {
                        let (mut plane, mut m) = (ComplexPlane::new(width, height), m);
                        loop {
                            let frame = next.fetch_add(1, Ordering::SeqCst);
                            if frame >= total || thread_cancel.load(Ordering::Relaxed) {
                                return;
                            }
                            let time = frame_time(frame, fps, speed);
                            let camera = path.camera_at(time);
                            plane.set_view(&camera.view());
                            if let Some(settings) = auto_settings {
                                m.max_iterations = limit_at(camera.scale, flight, settings);
                            }
                            m.palette = flight.palettes.0.blend(flight.palettes.1, path.share_of_path(time));
                            let mut colors = Vec::new();
                            let completed = Render::new(&plane, &m, width, height, coloring_function).supersampling(samples).adaptive(adaptive)
                                .gpu(gpu.as_deref().map(|gpu| (gpu, Duration::ZERO))).fast_gpu(fast_gpu).cancel_with(thread_cancel).run(|layer, _| colors = layer.to_vec());
                            if !completed {
                                let _ = frames_sender.send((frame, None));
                                return;
                            }
                            if blur {
                                //The blur preset of the saved image quality, as a saved image has it (one sample with the blur of a full-resolution frame, see `blur::blurred`)
                                colors = super::blur::blurred(&colors, width, height, super::blur::blur_sigma(1, samples));
                            }
                            if frames_sender.send((frame, Some(colors_to_rgb_bytes(&colors, mapping)))).is_err() {
                                return;
                            }
                        }
                    });
                }
                drop(frames_sender);
                let mut waiting = std::collections::BTreeMap::new();
                for (frame, rgb) in &frames_receiver {
                    let Some(rgb) = rgb else {
                        thread_cancel.store(true, Ordering::Relaxed); //Cancelled: the frames in order so far are kept
                        continue;
                    };
                    waiting.insert(frame, Arc::new(rgb));
                    while let Some(rgb) = waiting.remove(&rendered) {
                        //(A full queue makes the renderers wait for the writers; a writer that stopped makes the send fail, and the cancel flag says why)
                        if (pngs && png_sender.send((rendered, Arc::clone(&rgb))).is_err()) || (encoding && ffmpeg_sender.send(rgb).is_err()) {
                            thread_cancel.store(true, Ordering::Relaxed);
                            break;
                        }
                        rendered += 1;
                        thread_done.store(progress(&png_done, &encoded, rendered), Ordering::Relaxed);
                    }
                }
            });
            drop((png_sender, ffmpeg_sender));
            let _ = (png_writer.join(), ffmpeg_writer.join());
            let failure = failure.lock().unwrap().take();
            if let Some(failure) = failure {
                let _ = sender.send(failure);
                return;
            }
            let frames = progress(&png_done, &encoded, rendered);
            thread_done.store(frames, Ordering::Relaxed);
            let encoded = ffmpeg.and_then(|mut child| child.wait().ok().filter(std::process::ExitStatus::success)).is_some();
            let complete = if frames == total { "" } else { " (stopped: incomplete)" };
            let (written, file) = (if pngs { format!(" in {}", folder.display()) } else { String::new() }, if encoded { format!(" as {video}") } else if pngs { String::from(" (ffmpeg was not found or failed, so there is no video file; the frames can be encoded with it later)") } else { String::from(" (ffmpeg was not found or failed, and PNG frames are off: nothing was written)") });
            let _ = sender.send(format!("Video of the flight: {frames} of {total} frames at {fps} FPS, {speed}% speed{complete}{written}{file}"));
        });
        self.log(format!("Rendering a video of the flight to {destination}: {total} frames of {width} x {height} at {fps} FPS, {speed}% speed, {}, {}", quality.name(), playback.name()));
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

    /// The progress of a video under way, with its button to cancel it
    pub(super) fn video_progress_ui(&self, ui: &mut egui::Ui) {
        let Some(job) = &self.video_job else { return };
        ui.horizontal(|ui| {
            let done = job.done.load(Ordering::Relaxed);
            let elapsed = job.started.elapsed().as_secs_f64();
            let left = if done > 0 { elapsed / done as f64 * (job.total - done) as f64 } else { 0.0 };
            if ui.button("Cancel video").clicked() {
                job.cancel.store(true, Ordering::Relaxed);
            }
            ui.add(egui::ProgressBar::new(done as f32 / job.total.max(1) as f32).text(format!("Video: {done} of {} frames, {left:.0} s left", job.total)));
        });
    }

    /// The video controls: the button, the frame rate, the speed, the format and the PNG frames (the progress of a video under way is at the top, see
    /// `video_progress_ui`)
    pub(super) fn video_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Video of the last flight");
        ui.add_enabled_ui(self.video_job.is_none(), |ui| self.video_settings_ui(ui));
    }

    fn video_settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.video_speed).range(SPEEDS).suffix("% speed")).on_hover_text("The speed of the video, in percent of the flight's (1% to 400%; type a value): 10% shows every moment of the flight ten times as long (ten times the frames, and the render time); 200% is twice as fast");
            ui.add(egui::DragValue::new(&mut self.video_frames_at_once).range(FRAMES_AT_ONCE).suffix(" at once")).on_hover_text("How many frames are rendered at the same time (1 to 4), each with all the CPU threads and the GPU: one fills the gaps another leaves, so the CPU and the GPU stay busy; more takes more memory");
            ui.checkbox(&mut self.video_pngs, "PNG frames").on_hover_text("Also write every frame as a PNG file to a folder in saved/ (exact pixels, for editing; large). Off by default: only the video file");
        });
        ui.horizontal(|ui| {
            ui.label("Quality:");
            egui::ComboBox::from_id_salt("video_quality").selected_text(self.video_quality.name()).show_ui(ui, |ui| {
                for quality in VideoQuality::ALL {
                    ui.selectable_value(&mut self.video_quality, quality, quality.name()).on_hover_text(match quality {
                        VideoQuality::High => "The smallest files: fine detail gets a little soft",
                        VideoQuality::VeryHigh => "Detail kept well, files of a sensible size: a good choice to upload to a video site",
                        VideoQuality::UltraHigh => "Detail kept very well; large files",
                        VideoQuality::NearLossless => "Hard to tell from the rendered pixels; very large files",
                        VideoQuality::Lossless => "Exactly the rendered pixels (FFV1 in an .mkv): enormous files, too large for most players to play smoothly; for editing",
                    });
                }
            }).response.on_hover_text("How much of the rendered detail the video file keeps: a fixed quality, so busy views get the bits they need and calm ones stay small");
            ui.add_enabled_ui(self.video_quality != VideoQuality::Lossless, |ui| {
                ui.label("for");
                egui::ComboBox::from_id_salt("video_playback").selected_text(self.video_playback.name()).show_ui(ui, |ui| {
                    for playback in Playback::ALL {
                        ui.selectable_value(&mut self.video_playback, playback, playback.name()).on_hover_text(match playback {
                            Playback::Computers => "Full color resolution in 10 bits (HEVC on an NVIDIA GPU, else H.264): the best colors, no banding; plays on computers (VLC and most players), not on phones and TVs",
                            Playback::Phones => "HEVC Main 10 (10 bits, half the color resolution), which phones, TVs and browsers play in hardware; single pixels of strong colors get softer",
                            Playback::Everywhere => "H.264 in 8 bits with half the color resolution: plays on anything, with the softest colors",
                        });
                    }
                }).response.on_hover_text("Where the video has to play, which sets its color resolution and codec");
            });
        });
        //What the video will be: its length and frames are exact, its size an estimate (an upper end: measured on a busy view)
        if let Some(flight) = &self.last_flight {
            let path = Flight::with(flight.from, flight.to, flight.pace, flight.route, flight.constant_speed, flight.speed);
            let size = self.save_size.resolve(self.default_save_resolution(), self.view_resolution()).map_or((0, 0), |resolution| (resolution.width, resolution.height));
            let (frames, length, bytes) = estimate(path.duration(), self.video_fps, self.video_speed, size, self.video_quality, self.video_playback);
            let png = if self.video_pngs { format!(" + PNG frames about {}", megabytes(frames as f64 * (size.0 * size.1) as f64 * 3.0 / 2.0)) } else { String::new() };
            ui.label(format!("{:.1} s, {frames} frames of {} x {}, up to about {}{png}", length.as_secs_f64(), size.0, size.1, megabytes(bytes)))
                .on_hover_text("The length and the frames of the video, and the size of its file at most (measured on a busy, detailed view; calm views take far less)");
        }
    }
}

/// A size in bytes, in MB or GB
fn megabytes(bytes: f64) -> String {
    if bytes >= 1e9 { format!("{:.1} GB", bytes / 1e9) } else { format!("{:.0} MB", (bytes / 1e6).max(1.0)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_of_a_video_frame_goes_from_the_limit_at_the_start_to_that_of_the_place() {
        let settings = crate::auto_iterations::AutoPreset::Balanced.settings();
        let flight = VideoFlight { from: Camera { x: -0.75, y: 0.0, scale: 1.0 }, to: Camera { x: -0.7498, y: 0.0203, scale: 6.4e-5 }, pace: Pace::Fast, route: Route::Shortest, constant_speed: 3.0, speed: 1.0, name: String::from("Seahorse"), limits: (300, 5000), palettes: (crate::mandelbrot_set::ColorCycle::CYCLE, crate::mandelbrot_set::ColorCycle::CYCLE) };
        assert_eq!(limit_at(1.0, &flight, &settings), 300);
        assert_eq!(limit_at(6.4e-5, &flight, &settings), 5000);
        let middle = limit_at(8e-3, &flight, &settings);
        assert!(middle > 300 && middle < 5000, "{middle}");
    }

    #[test]
    fn a_slower_video_has_more_frames_of_the_same_flight() {
        let second = Duration::from_secs(1);
        assert_eq!(frame_count(second * 2, 30, 100), 61, "2 s at 30 FPS: the start, then 60 more frames");
        assert_eq!(frame_count(second * 2, 30, 10), 601, "10%: ten times the frames");
        assert_eq!(frame_count(second * 2, 240, 400), 121);
        assert_eq!(frame_time(30, 30, 100), second);
        assert_eq!(frame_time(300, 30, 10), second, "at 10% frame 300 (10 s of video) is 1 s into the flight");
        assert_eq!(frame_time(60, 120, 50), Duration::from_millis(250));
        assert!(FRAME_RATES.starts_with(&[24, 30, 60]) && FRAME_RATES.ends_with(&[90, 120, 144, 240]));
    }

    #[test]
    fn the_tiers_keep_a_fixed_quality_for_the_player_they_are_for() {
        let has = |arguments: &[String], wanted: &[&str]| wanted.iter().all(|wanted| arguments.iter().any(|argument| argument.contains(wanted)));
        //Computers: full color in 10 bits, on the GPU or the CPU; a fixed quantizer that follows the tier
        assert!(has(&arguments(VideoQuality::VeryHigh, Playback::Computers, true, 60), &["hevc_nvenc", "yuv444p16le", "constqp", "20", "bt709"]));
        assert!(has(&arguments(VideoQuality::UltraHigh, Playback::Computers, true, 60), &["16"]));
        assert!(has(&arguments(VideoQuality::VeryHigh, Playback::Computers, false, 60), &["libx264", "yuv444p10le", "14"]));
        //Phones: HEVC Main 10, half the color resolution made with a sharp filter; everywhere: 8-bit H.264
        assert!(has(&arguments(VideoQuality::High, Playback::Phones, true, 60), &["hevc_nvenc", "p010le", "main10", "lanczos"]));
        assert!(has(&arguments(VideoQuality::High, Playback::Phones, false, 60), &["libx265", "yuv420p10le"]));
        assert!(has(&arguments(VideoQuality::High, Playback::Everywhere, true, 60), &["libx264", "yuv420p", "18"]));
        //A keyframe every half second and the index at the start; lossless is FFV1 whatever the player
        assert!(has(&arguments(VideoQuality::High, Playback::Everywhere, true, 60), &["-g", "30", "+faststart"]));
        assert_eq!(arguments(VideoQuality::Lossless, Playback::Phones, true, 60), ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]);
        assert_eq!((extension(VideoQuality::Lossless), extension(VideoQuality::NearLossless)), ("mkv", "mp4"));
    }

    #[test]
    fn the_estimate_has_the_exact_frames_and_length_and_a_size_that_grows_with_the_tier() {
        let (frames, length, high) = estimate(Duration::from_secs(13), 60, 50, (1920, 1080), VideoQuality::High, Playback::Computers);
        assert_eq!((frames, length.as_secs()), (1561, 26), "13 s at half speed: 26 s, at 60 FPS");
        let sizes: Vec<f64> = VideoQuality::ALL.iter().map(|quality| estimate(Duration::from_secs(13), 60, 50, (1920, 1080), *quality, Playback::Computers).2).collect();
        assert!(sizes.windows(2).all(|pair| pair[0] < pair[1]) && (sizes[0] - high).abs() < 1.0);
        //Lossless: about 3.2 GB, as Jort measured (3 GB for 26 s)
        assert!((2.5e9..4e9).contains(&sizes[4]), "{}", sizes[4]);
        assert_eq!((megabytes(3.2e9), megabytes(65e6)), (String::from("3.2 GB"), String::from("65 MB")));
    }
}

//! Videos of flights (issue #67): the last flight rendered offline, frame by frame, at a fixed frame rate and the full quality of saved images, however long a
//! frame takes.
//!
//! A background job puts the camera at `n / fps × speed` along the flight path for frame `n` (*Video speed*: 100% is the speed of the flight, 10% a ten times
//! slower video), renders it at the saved image size and quality (the GPU helps as in the window), and hands it on: one thread writes it as `frame-00001.png`, ...
//! to a folder of `saved/` (unless *PNG frames* is off), another pipes it to `ffmpeg`, when it is installed, which encodes the video in the chosen
//! `VideoFormat`. The renderer goes on with the next frame while those two write the last ones (at most `FRAMES_WAITING` frames wait), so neither the CPU
//! threads nor the GPU wait for the encoder. The iteration limit follows the zoom like a flight with auto iterations: from the limit at the start to the limit of
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

/// How many rendered frames may wait for the writers of the video (the PNG frames and ffmpeg) before the renderer waits for them
const FRAMES_WAITING: usize = 2;

/// The encoding of the video file (ffmpeg's)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VideoFormat {
    /// 10 bits with the full color resolution (4:4:4): HEVC on the GPU's encoder (NVENC) when it has one, else H.264 on the CPU (libx264): no banding in smooth
    /// gradients; plays in VLC and most players, not in every browser or phone
    HighQuality,
    /// FFV1 in an .mkv: exactly the rendered pixels, bit for bit; large files
    Lossless,
    /// H.264 in 8 bits with half the color resolution (4:2:0): plays everywhere, but smooth gradients can show bands
    Compatible,
}

impl VideoFormat {
    /// Every format, in the order of the list (the default first)
    pub(super) const ALL: [VideoFormat; 3] = [VideoFormat::HighQuality, VideoFormat::Lossless, VideoFormat::Compatible];

    /// The name in the list
    pub(super) const fn name(self) -> &'static str {
        match self {
            VideoFormat::HighQuality => "High quality (10-bit)",
            VideoFormat::Lossless => "Lossless (FFV1)",
            VideoFormat::Compatible => "Compatible (8-bit H.264)",
        }
    }

    /// The extension of its file
    const fn extension(self) -> &'static str {
        match self {
            VideoFormat::Lossless => "mkv",
            VideoFormat::HighQuality | VideoFormat::Compatible => "mp4",
        }
    }

    /// The arguments of ffmpeg that encode RGB frames in this format (with `nvenc`: on the GPU's HEVC encoder). The colors are converted with the BT.709 matrix
    /// and tagged so, so players show them as they were rendered
    fn arguments(self, nvenc: bool) -> Vec<&'static str> {
        const TAGS: [&str; 6] = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"];
        let mut arguments = match self {
            VideoFormat::HighQuality if nvenc => vec!["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv444p16le", "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "14", "-b:v", "0", "-tag:v", "hvc1"],
            VideoFormat::HighQuality => vec!["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv444p10le", "-c:v", "libx264", "-crf", "14", "-preset", "medium"],
            VideoFormat::Lossless => return vec!["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"],
            VideoFormat::Compatible => vec!["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-crf", "16"],
        };
        arguments.extend(TAGS);
        arguments
    }
}

/// The smallest video the GPU's HEVC encoder takes (NVENC needs at least about 129 x 33 pixels); a smaller one is encoded on the CPU (libx264, which takes any
/// size; libx265 is not used, as some builds crash on small frames)
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

/// `ffmpeg` encoding raw RGB frames of `width` x `height` at `fps` from its standard input into `path` in `format`, if `ffmpeg` can be started
fn start_ffmpeg(width: usize, height: usize, fps: u32, path: &str, format: VideoFormat) -> Option<Child> {
    let (size, rate) = (format!("{width}x{height}"), fps.to_string());
    let mut command = Command::new("ffmpeg");
    command.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-r", &rate, "-i", "-"]).args(format.arguments(format == VideoFormat::HighQuality && width >= NVENC_SMALLEST.0 && height >= NVENC_SMALLEST.1 && nvenc_works())).arg(path);
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
        let (width, height, fps, speed, format, pngs) = (resolution.width, resolution.height, self.video_fps, self.video_speed, self.video_format, self.video_pngs);
        let path = Flight::with(flight.from, flight.to, flight.pace, flight.route, flight.constant_speed);
        let total = frame_count(path.duration(), fps, speed);
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let slug: String = flight.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
        let name = format!("video-{stamp}-{slug}");
        let (mut m, coloring_function, mapping) = (self.m.clone(), self.coloring_function, self.p.color_channel_mapping);
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
            if let Err(err) = std::fs::create_dir_all(&folder) {
                let _ = sender.send(format!("Could not render the video: {err}"));
                return;
            }
            let video = saved.join(format!("{name}.{}", format.extension())).display().to_string();
            let mut ffmpeg = start_ffmpeg(width, height, fps, &video, format);
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
            let mut plane = ComplexPlane::new(width, height);
            let mut rendered = 0;
            for frame in 0..total {
                if thread_cancel.load(Ordering::Relaxed) {
                    break;
                }
                let time = frame_time(frame, fps, speed);
                let camera = path.camera_at(time);
                plane.set_view(&camera.view());
                if let Some(settings) = &auto_settings {
                    m.max_iterations = limit_at(camera.scale, &flight, settings);
                }
                m.palette = flight.palettes.0.blend(flight.palettes.1, path.share_of_path(time));
                let mut colors = Vec::new();
                let completed = Render::new(&plane, &m, width, height, coloring_function).supersampling(samples).adaptive(adaptive)
                    .gpu(gpu.as_deref().map(|gpu| (gpu, Duration::ZERO))).fast_gpu(fast_gpu).cancel_with(&thread_cancel).run(|layer, _| colors = layer.to_vec());
                if !completed {
                    break;
                }
                if blur {
                    //The blur preset of the saved image quality, as a saved image has it (one sample with the blur of a full-resolution frame, see `blur::blurred`)
                    colors = super::blur::blurred(&colors, width, height, super::blur::blur_sigma(1, samples));
                }
                let rgb = Arc::new(colors_to_rgb_bytes(&colors, mapping));
                //(A full queue makes the renderer wait for the writers; a writer that stopped makes the send fail, and the cancel flag says why)
                if pngs && png_sender.send((frame, Arc::clone(&rgb))).is_err() {
                    break;
                }
                if encoding && ffmpeg_sender.send(rgb).is_err() {
                    break;
                }
                rendered = frame + 1;
                thread_done.store(progress(&png_done, &encoded, rendered), Ordering::Relaxed);
            }
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
        self.log(format!("Rendering a video of the flight to {destination}: {total} frames of {width} x {height} at {fps} FPS, {speed}% speed, {}", format.name()));
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
            egui::ComboBox::from_id_salt("video_format").selected_text(self.video_format.name()).show_ui(ui, |ui| {
                for format in VideoFormat::ALL {
                    ui.selectable_value(&mut self.video_format, format, format.name()).on_hover_text(match format {
                        VideoFormat::HighQuality => "10 bits with full color resolution: no banding in smooth gradients. HEVC on the GPU's encoder (NVENC) when it has one, which leaves the CPU to the render; else H.264 on the CPU. Plays in VLC and most players",
                        VideoFormat::Lossless => "FFV1 in an .mkv: exactly the rendered pixels; very large files",
                        VideoFormat::Compatible => "8-bit H.264: plays everywhere (phones, browsers), but smooth gradients can show bands",
                    });
                }
            }).response.on_hover_text("The format of the video file that ffmpeg makes");
            ui.checkbox(&mut self.video_pngs, "PNG frames").on_hover_text("Also write every frame as a PNG file to a folder in saved/ (exact pixels, for editing); off, only the video file is written");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_of_a_video_frame_goes_from_the_limit_at_the_start_to_that_of_the_place() {
        let settings = crate::auto_iterations::AutoPreset::Balanced.settings();
        let flight = VideoFlight { from: Camera { x: -0.75, y: 0.0, scale: 1.0 }, to: Camera { x: -0.7498, y: 0.0203, scale: 6.4e-5 }, pace: Pace::Fast, route: Route::Shortest, constant_speed: 3.0, name: String::from("Seahorse"), limits: (300, 5000), palettes: (crate::mandelbrot_set::ColorCycle::CYCLE, crate::mandelbrot_set::ColorCycle::CYCLE) };
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
    fn the_formats_encode_in_10_bits_losslessly_or_compatibly_with_tagged_colors() {
        let high = VideoFormat::HighQuality.arguments(true);
        assert!(high.contains(&"hevc_nvenc") && high.iter().any(|argument| argument.contains("yuv444p16le")) && high.contains(&"bt709"));
        let cpu = VideoFormat::HighQuality.arguments(false);
        assert!(cpu.contains(&"libx264") && cpu.iter().any(|argument| argument.contains("yuv444p10le")), "10-bit H.264 on the CPU without NVENC");
        assert_eq!(VideoFormat::Lossless.arguments(true), ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]);
        assert_eq!((VideoFormat::Lossless.extension(), VideoFormat::HighQuality.extension()), ("mkv", "mp4"));
        assert!(VideoFormat::Compatible.arguments(true).contains(&"libx264"));
        assert_eq!(VideoFormat::ALL[0], VideoFormat::HighQuality, "the default first");
    }
}

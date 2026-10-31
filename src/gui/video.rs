//! Videos of flights (issue #67): the last flight rendered offline, frame by frame, at a fixed frame rate and the full quality of saved images, however long a
//! frame takes.
//!
//! A background job puts the camera at `n / fps × speed` along the flight path for frame `n` (*Video speed*: 100% is the speed of the flight, 10% a ten times
//! slower video), renders it at the saved image size and quality (the GPU helps as in the window), and hands it on: one thread writes it as `frame-00001.png`, ...
//! to a folder of `saved/` (with *PNG frames*, off by default), another pipes it to `ffmpeg`, when it is installed, which encodes the video in the chosen
//! `VideoQuality` for the chosen `Playback`. Several frames render at once, as many as keep the CPU busy (`FrameScheduler`), and the renderers go on while the writers write the last
//! ones (at most `FRAMES_WAITING` frames wait), so neither the CPU threads nor the GPU wait for the encoder or for the end of a layer. The iteration limit follows the zoom like a flight with auto iterations: from the limit at the start to the limit of
//! the place. Frames are sharp (no motion blur). A video is written in chunks that survive any interruption and can be resumed (see `chunks`).

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};

use eframe::egui;

use super::flight::{Camera, Flight, Pace, Route};
use super::{colors_to_rgb_bytes, MandelbrotApp};
use crate::coloring::TrueColor;
use crate::complex_plane::ComplexPlane;
use crate::rendering::{self, Render};

/// The frame rates of the list
pub(super) const FRAME_RATES: [u32; 7] = [24, 30, 60, 90, 120, 144, 240];

/// The range of the video speed, in percent of the speed of the flight
pub(super) const SPEEDS: std::ops::RangeInclusive<u16> = 1..=400;

/// How many rendered frames may wait for the writers of the video (the PNG frames and ffmpeg) before the renderers wait for them
const FRAMES_WAITING: usize = 2;

/// The most frames a video renders at once: each takes all the CPU threads and the GPU, and the job adds frames while the CPU is not yet busy (see `FrameScheduler`)
const MAX_AT_ONCE: usize = 8;

/// How much faster a video must render with one more frame at once to keep it (1.03: 3% more frames per second); measured differences below that are noise
const WORTH_ONE_MORE: f64 = 1.03;

/// How many measurements the job keeps the number of frames at once after one more did not pay, before it tries again (the frames get dearer along a zoom,
/// which can change what pays)
const HOLD_WINDOWS: u32 = 8;

/// The shortest measurement of the frame rate (and at least twice as many frames as are under way, so every renderer finishes some)
const MIN_WINDOW: Duration = Duration::from_secs(2);

/// The memory of a frame under way, per pixel: its samples (`rendering::SampleState`, about 30 bytes) and its colors on the way to the file, with room to spare
const BYTES_PER_PIXEL_UNDER_WAY: u64 = 48;

/// How many frames of `pixels` the job may render at once: at most `MAX_AT_ONCE`, and together at most a third of the memory that is available (on systems
/// where that is not known, 4)
fn frames_memory_allows(pixels: usize) -> usize {
    let per_frame = (pixels as u64 * BYTES_PER_PIXEL_UNDER_WAY).max(1);
    let allowed = available_memory().map_or(4, |bytes| usize::try_from(bytes / 3 / per_frame).unwrap_or(MAX_AT_ONCE));
    allowed.clamp(1, MAX_AT_ONCE)
}

/// The memory that is available for new work, in bytes (`MemAvailable` of `/proc/meminfo`), where the system says it
fn available_memory() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|line| line.starts_with("MemAvailable:"))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kilobytes| kilobytes * 1024)
}

/// How many frames of a video render at the same time, chosen by the job itself (issue #174), and the frames finished so far, which it measures that by. Every
/// frame under way takes all the CPU threads and hands rows to the GPU, so one fills the gaps another leaves (the end of a layer, the small passes of adaptive
/// supersampling, the GPU's batches). More is not always faster: frames that share the threads also slow each other down, so the job keeps what is measured
/// to be fastest (see `Climber`)
pub(super) struct FrameScheduler {
    /// The frames that may be under way, and those that are
    allowed: AtomicUsize,
    under_way: std::sync::Mutex<usize>,
    free: std::sync::Condvar,
    /// The frames finished by the renderers
    finished: AtomicUsize,
    /// The CPU use of the render threads in the last measurement, in percent, for the progress line
    pub(super) cpu_percent: AtomicUsize,
}

impl FrameScheduler {
    fn new(start: usize) -> FrameScheduler {
        FrameScheduler { allowed: AtomicUsize::new(start), under_way: std::sync::Mutex::new(0), free: std::sync::Condvar::new(), finished: AtomicUsize::new(0), cpu_percent: AtomicUsize::new(0) }
    }

    /// The frames that may be under way now
    pub(super) fn allowed(&self) -> usize {
        self.allowed.load(Ordering::Relaxed)
    }

    fn set_allowed(&self, allowed: usize) {
        self.allowed.store(allowed.max(1), Ordering::Relaxed);
        self.free.notify_all();
    }

    /// Waits until another frame may start (or `cancel` is set: then false)
    fn start_frame(&self, cancel: &AtomicBool) -> bool {
        let mut under_way = self.under_way.lock().unwrap();
        while *under_way >= self.allowed() {
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            under_way = self.free.wait_timeout(under_way, Duration::from_millis(50)).unwrap().0;
        }
        *under_way += 1;
        true
    }

    /// The renderer of a frame stops (`rendered`: with the frame finished): another frame may start
    fn finish_frame(&self, rendered: bool) {
        if rendered {
            self.finished.fetch_add(1, Ordering::Relaxed);
        }
        *self.under_way.lock().unwrap() -= 1;
        self.free.notify_all();
    }
}

/// Finds the number of frames at once that renders fastest, by trying one more and keeping it when the frame rate rises by `WORTH_ONE_MORE` (hill climbing).
/// After one more did not pay it waits `HOLD_WINDOWS` measurements, measures again where it is, and tries again
struct Climber {
    current: usize,
    most: usize,
    /// The frame rate at `current`, the number being tried, and the measurements left before the next try
    rate: Option<f64>,
    trying: Option<usize>,
    hold: u32,
}

impl Climber {
    fn new(start: usize, most: usize) -> Climber {
        Climber { current: start.clamp(1, most.max(1)), most: most.max(1), rate: None, trying: None, hold: 0 }
    }

    /// Takes the frame rate measured with the number this returned last (`current` at first), and returns the number of frames at once for the next measurement
    fn measured(&mut self, rate: f64) -> usize {
        match self.trying.take() {
            Some(more) if rate > self.rate.unwrap_or(0.0) * WORTH_ONE_MORE => {
                (self.current, self.rate) = (more, Some(rate)); //It pays: keep it, and try one more at once
                self.try_one_more()
            }
            Some(_) => {
                self.hold = HOLD_WINDOWS; //It does not: back to where it was, for a while
                self.rate = None;
                self.current
            }
            None if self.hold > 0 => {
                self.hold -= 1;
                self.current
            }
            None => {
                self.rate = Some(rate);
                self.try_one_more()
            }
        }
    }

    fn try_one_more(&mut self) -> usize {
        if self.current < self.most {
            self.trying = Some(self.current + 1);
            self.current + 1
        } else {
            self.current
        }
    }
}

/// The quality of the video file: how much of the rendered detail the encoder keeps. The tiers keep a fixed quality (the encoder's quantizer), so a busy view
/// gets the bits it needs and a calm one stays small. Measured on 60 frames of a detailed 1080p zoom at 60 FPS (the mean error of a color channel, in levels of
/// 255, on the GPU's encoder with full color): High 3.7, Very High 2.7, Ultra High 1.9, Near Lossless 0.9
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
            (VideoQuality::High, Playback::Maximum) => 0.65,
            (VideoQuality::VeryHigh, Playback::Maximum) => 1.1,
            (VideoQuality::UltraHigh, Playback::Maximum) => 1.75,
            (VideoQuality::NearLossless, Playback::Maximum) => 3.4,
            (VideoQuality::High, _) => 0.57,
            (VideoQuality::VeryHigh, _) => 0.9,
            (VideoQuality::UltraHigh, _) => 1.3,
            (VideoQuality::NearLossless, _) => 2.3,
        }
    }
}

/// The format of the video: its codec, bit depth and color resolution (lossless videos are always FFV1)
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) enum Playback {
    /// HEVC (on an NVIDIA GPU's encoder; H.264 on the CPU otherwise) in 10 bits with the full color resolution (4:4:4): the best colors; computers play it
    /// (VLC, most players), phones and TVs do not
    Maximum,
    /// HEVC Main 10 (10 bits, half the color resolution, 4:2:0), which phones, TVs and browsers decode in hardware. Detail of single pixels in strong colors
    /// gets softer
    HighQuality,
    /// H.264 in 8 bits with half the color resolution: plays on anything, with the softest colors
    MostCompatible,
}

impl Playback {
    /// Every choice, in the order of the list (the default first)
    pub(super) const ALL: [Playback; 3] = [Playback::Maximum, Playback::HighQuality, Playback::MostCompatible];

    /// The name in the list
    pub(super) const fn name(self) -> &'static str {
        match self {
            Playback::Maximum => "Maximum",
            Playback::HighQuality => "High quality",
            Playback::MostCompatible => "Most compatible",
        }
    }

    /// The choice of `name`, also by the names it had before (issue #173: "Computers", "Phones and TVs", "Everywhere"), so saved settings keep their choice
    pub(super) fn of_name(name: &str) -> Option<Playback> {
        let name = match name {
            "Computers" => "Maximum",
            "Phones and TVs" => "High quality",
            "Everywhere" => "Most compatible",
            name => name,
        };
        Playback::ALL.into_iter().find(|playback| playback.name() == name)
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
        Playback::Maximum if nvenc => nvenc_hevc(&format!("{FULL}yuv444p16le")),
        Playback::Maximum => vec![format!("{FULL}yuv444p10le"), "-c:v".into(), "libx264".into(), "-crf".into(), crf.into(), "-preset".into(), "medium".into()],
        Playback::HighQuality if nvenc => [nvenc_hevc(&format!("{HALF}p010le")), vec!["-profile:v".into(), "main10".into()]].concat(),
        Playback::HighQuality => vec![format!("{HALF}yuv420p10le"), "-c:v".into(), "libx265".into(), "-crf".into(), crf.into(), "-preset".into(), "medium".into(), "-tag:v".into(), "hvc1".into(), "-x265-params".into(), "log-level=error".into()],
        Playback::MostCompatible => vec![format!("{HALF}yuv420p"), "-c:v".into(), "libx264".into(), "-crf".into(), crf.into(), "-preset".into(), "medium".into(), "-profile:v".into(), "high".into()],
    });
    arguments.extend(["-g".into(), (fps / 2).max(1).to_string(), "-bf".into(), "2".into(), "-movflags".into(), "+faststart".into()]);
    arguments.extend(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"].map(String::from));
    arguments
}

/// The estimate of a video shown next to its controls: its frames, its length, and the size it takes at most on busy views (see `bits_per_pixel`)
pub(super) fn estimate(duration: Duration, fps: u32, speed: f64, (width, height): (usize, usize), quality: VideoQuality, playback: Playback) -> (usize, Duration, f64) {
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
fn frame_time(frame: usize, fps: u32, speed: f64) -> Duration {
    Duration::from_secs_f64(frame as f64 / f64::from(fps) * speed.max(1.0) / 100.0)
}

/// The frames of a video of a flight of `duration` at `fps` and `speed` percent: from the start to the place, both included (a video that fits a length exactly
/// is not one frame longer because of the rounding of the speed)
fn frame_count(duration: Duration, fps: u32, speed: f64) -> usize {
    (duration.as_secs_f64() * 100.0 / speed.max(1.0) * f64::from(fps) - 1e-6).ceil().max(0.0) as usize + 1
}

/// The time along the flight of every frame of a video: at `fps` and `speed` percent of the flight's speed, or following the `music` (see `music`), when the
/// flight of `duration` lasts as long as the track
#[derive(Clone, Debug)]
pub(super) struct FrameClock {
    fps: u32,
    speed: f64,
    music: Option<Arc<super::music::MusicClock>>,
    duration: Duration,
}

impl FrameClock {
    /// The time along the flight of frame `frame`
    fn time(&self, frame: usize) -> Duration {
        self.music.as_ref().map_or_else(|| frame_time(frame, self.fps, self.speed), |music| self.duration.mul_f64(music.share_at(frame as f64 / f64::from(self.fps))))
    }

    /// The frames of the video, from the start to the place, both included
    fn frames(&self) -> usize {
        self.music.as_ref().map_or_else(|| frame_count(self.duration, self.fps, self.speed), |music| (music.length().as_secs_f64() * f64::from(self.fps) - 1e-6).ceil().max(0.0) as usize + 1)
    }
}

/// A speed in percent as the video controls and the log show it: whole percents without decimals
pub(super) fn speed_text(speed: f64) -> String {
    if (speed - speed.round()).abs() < 0.05 { format!("{speed:.0}%") } else { format!("{speed:.1}%") }
}

/// A length as hours, minutes and seconds (`1:02:03`, or `2:03` under an hour)
fn length_text(length: Duration) -> String {
    let seconds = length.as_secs_f64().round() as u64;
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 { format!("{hours}:{minutes:02}:{seconds:02}") } else { format!("{minutes}:{seconds:02}") }
}

/// The flight a video is made of: from where to where, at which pace, and the iteration limits at both ends
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct VideoFlight {
    pub(super) from: Camera,
    pub(super) to: Camera,
    pub(super) pace: Pace,
    /// The route of the flight, and the speed of the constant pace (see `flight::Flight::with`)
    pub(super) route: Route,
    pub(super) constant_speed: f64,
    /// The speed factor of the other paces, and the zoom a dive starts at (see `flight::Flight::with`)
    pub(super) speed: f64,
    pub(super) dive_zoom: f64,
    pub(super) name: String,
    pub(super) limits: (u32, u32),
    /// The color cycle at the start and at the place, blended on the way like a flight's (see `points::PointFlight::palettes`)
    pub(super) palettes: (crate::mandelbrot_set::ColorCycle, crate::mandelbrot_set::ColorCycle),
    /// A flight through keyframes instead (see `sequence`): then `from` and `to` are its first and last keyframes
    #[serde(default)]
    pub(super) sequence: Option<super::sequence::Sequence>,
}

impl VideoFlight {
    /// The path of the flight: through its keyframes, or from `from` to `to` at its pace along its route
    pub(super) fn path(&self) -> Flight {
        self.sequence.as_ref().map_or_else(
            || Flight::with(self.from, self.to, self.pace, self.route, self.constant_speed, self.speed, self.dive_zoom),
            |sequence| Flight::of_sequence(sequence.clone(), (self.constant_speed, self.speed)),
        )
    }

    /// The colors of the frame `time` into the flight: from its keyframes, or blended from those at the start to the place's
    pub(super) fn palette_at(&self, path: &Flight, time: Duration) -> crate::mandelbrot_set::ColorCycle {
        path.palette_at(time).unwrap_or_else(|| self.palettes.0.blend(self.palettes.1, path.share_of_path(time)))
    }
}

/// Everything a video is made of: the flight and every setting of the render, taken when the video is started or added to the queue, so later changes of
/// the controls do not change it
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)] // The settings of a render as the controls have them
pub(super) struct VideoSpec {
    pub(super) flight: VideoFlight,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) fps: u32,
    /// In percent of the flight's speed (exact, also for *Fit to length*)
    pub(super) speed: f64,
    pub(super) quality: VideoQuality,
    pub(super) playback: Playback,
    pub(super) pngs: bool,
    pub(super) blend: bool,
    /// The samples per pixel, adaptive supersampling and the blur preset of saved images
    pub(super) samples: u8,
    pub(super) adaptive: bool,
    pub(super) blur: bool,
    /// The coloring (an index of `COLORING_FUNCTIONS`), the color channels, and the set (its limit without auto iterations, its smooth coloring)
    pub(super) coloring_index: usize,
    #[serde(with = "mapping_text")]
    pub(super) mapping: crate::coloring::ColorChannelMapping,
    pub(super) set: crate::mandelbrot_set::MandelbrotSet,
    pub(super) auto: Option<crate::auto_iterations::AutoIterationSettings>,
    pub(super) use_gpu: bool,
    pub(super) fast_gpu: bool,
    /// The name of the video file (without extension), given when it is first started: a resumed video keeps its name (see `chunks`)
    #[serde(default)]
    pub(super) output: Option<String>,
    /// The audio track the video follows and has as its sound (see `music`)
    #[serde(default)]
    pub(super) music: Option<super::music::Music>,
}

/// The name of a new video of `flight` in `folder` (without extension): the time and the name of the place, with a number when a video of that name (or its
/// chunks) is there already
pub(super) fn video_name(flight: &VideoFlight, folder: &std::path::Path) -> String {
    let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let slug: String = flight.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    let base = format!("video-{stamp}-{slug}");
    let taken = |name: &str| ["mp4", "mkv", "parts"].iter().any(|extension| folder.join(format!("{name}.{extension}")).exists()) || folder.join(name).exists();
    //(Every name is a new file name, so one of the first more than there are entries in the folder is free)
    let entries = std::fs::read_dir(folder).map_or(0, Iterator::count);
    (1..=entries + 2).map(|copy| if copy == 1 { base.clone() } else { format!("{base}-{copy}") }).find(|name| !taken(name)).expect("a free name")
}

/// The color channels of a `VideoSpec` in a file as their text ("RGB", "BGR", ...)
mod mapping_text {
    use crate::coloring::ColorChannelMapping;
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(clippy::trivially_copy_pass_by_ref)] // The signature that `serde(with)` calls
    pub(super) fn serialize<S: Serializer>(mapping: &ColorChannelMapping, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(mapping)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ColorChannelMapping, D::Error> {
        String::deserialize(deserializer)?.parse().map_err(serde::de::Error::custom)
    }
}

/// A video job: its progress (frames written of all) and cancel flag, and the receiver of its final message
pub(super) struct VideoJob {
    done: Arc<AtomicUsize>,
    /// How many frames render at once, and the CPU use (see `FrameScheduler`)
    pub(super) scheduler: Arc<FrameScheduler>,
    total: usize,
    pub(super) cancel: Arc<AtomicBool>,
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

/// `ffmpeg` encoding raw RGB frames of `width` x `height` at `fps` from its standard input into `path` in `quality` for `playback` (with `nvenc` on the GPU's
/// encoder), if `ffmpeg` can be started
fn start_ffmpeg((width, height): (usize, usize), fps: u32, path: &str, quality: VideoQuality, playback: Playback, nvenc: bool) -> Option<Child> {
    let (size, rate) = (format!("{width}x{height}"), fps.to_string());
    let mut command = Command::new("ffmpeg");
    command.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-r", &rate, "-i", "-"]).args(arguments(quality, playback, nvenc, fps)).arg(path);
    command.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()
}

/// A rendered frame of a video, on its way to the writers: its bytes, or with *Blend frames* its colors and view, which are blended in the order of the frames
enum RenderedFrame {
    Rgb(Vec<u8>),
    Colors(Vec<TrueColor>, ComplexPlane),
}

/// The colors of a frame as they are written: with the blur preset of the saved image quality, blurred as a saved image is (one sample with the blur of a
/// full-resolution frame, see `blur::blurred`)
fn finish(colors: Vec<TrueColor>, blur: bool, width: usize, height: usize, samples: u8) -> Vec<TrueColor> {
    if blur { super::blur::blurred(&colors, width, height, super::blur::blur_sigma(1, samples)) } else { colors }
}

/// Writes `rgb` (`width` x `height`) as a PNG file at `path`, with fast compression (a video has many frames; ffmpeg makes the small file)
fn write_png(path: &std::path::Path, rgb: &[u8], width: usize, height: usize, place: &str) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.add_itxt_chunk(String::from(crate::image_metadata::KEYWORD), place.to_string()).map_err(std::io::Error::other)?;
    encoder.write_header().and_then(|mut writer| writer.write_image_data(rgb)).map_err(std::io::Error::other)
}

impl MandelbrotApp {
    /// What a video of the last flight with the current settings would be (see `VideoSpec`), or why there is none
    pub(super) fn video_spec(&self) -> Result<VideoSpec, String> {
        let flight = self.last_flight.clone().ok_or_else(|| String::from("there is no flight yet"))?;
        let resolution = self.save_size.resolve(self.default_save_resolution(), self.view_resolution())?;
        let path = flight.path();
        let music = self.followed_music();
        let speed = if music.is_some() { 100.0 } else { self.video_speed_for(path.duration())? }; //(A video that follows music lasts as long as the track)
        Ok(VideoSpec {
            flight,
            width: resolution.width,
            height: resolution.height,
            fps: self.video_fps,
            speed,
            quality: self.video_quality,
            playback: self.video_playback,
            pngs: self.video_pngs,
            blend: self.video_blend,
            samples: self.image_supersampling_amount,
            adaptive: self.adaptive_supersampling,
            blur: self.image_quality_blur,
            coloring_index: self.coloring_index,
            mapping: self.p.color_channel_mapping,
            set: self.m.clone(),
            auto: self.auto_iterations.then(|| self.auto_settings.clone()),
            use_gpu: self.use_gpu,
            fast_gpu: self.fast_gpu,
            output: None,
            music,
        })
    }

    /// Renders the video of the last flight with the current settings in the background (see the module)
    pub(super) fn start_video(&mut self) {
        match self.video_spec() {
            Ok(spec) => self.start_video_of(spec),
            Err(reason) => self.log(format!("Could not render the video: {reason}")),
        }
    }

    /// Renders the video `spec` in the background (see the module): a video of the last flight, or one of the queue
    #[allow(clippy::too_many_lines)] // The settings of the job, its two writers and its loop over the frames
    pub(super) fn start_video_of(&mut self, spec: VideoSpec) {
        let name = spec.output.clone().unwrap_or_else(|| video_name(&spec.flight, &self.video_folder));
        let spec = VideoSpec { output: Some(name.clone()), ..spec };
        let manifest_spec = spec.clone();
        let VideoSpec { flight, width, height, fps, speed, quality, playback, pngs, blend, samples, adaptive, blur, coloring_index, mapping, set: m, auto, use_gpu, fast_gpu, output: _, music } = spec;
        let chunk_frames = self.video_chunk_frames.unwrap_or(fps as usize * super::chunks::CHUNK_SECONDS).max(1);
        let most_at_once = self.video_at_once_fixed.unwrap_or_else(|| frames_memory_allows(width * height));
        let scheduler = Arc::new(FrameScheduler::new(self.video_at_once_fixed.unwrap_or(2).min(most_at_once)));
        let thread_scheduler = Arc::clone(&scheduler);
        let path = flight.path();
        //The clock of the frames: at the video speed, or following the music (see `music`)
        let music_clock = match music.as_ref().map(|music| super::music::clock_of_track(std::path::Path::new(&music.path), music.strength)).transpose() {
            Ok(clock) => clock.map(Arc::new),
            Err(reason) => {
                self.log(format!("Could not render the video: {reason}"));
                return;
            }
        };
        let clock = FrameClock { fps, speed, music: music_clock, duration: path.duration() };
        let total = clock.frames();
        let coloring_function = super::COLORING_FUNCTIONS[coloring_index.min(super::COLORING_FUNCTIONS.len() - 1)].1;
        let adaptive = adaptive.then(rendering::AdaptiveSampling::default);
        let auto_settings = auto;
        let gpu = self.gpu.clone().filter(|_| use_gpu);
        //With a GPU, one more renderer renders whole frames on the GPU alone (see the renderers below)
        let gpu_renderer = gpu.is_some() && self.video_gpu_renderer;
        let (done, cancel) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)));
        let (thread_done, thread_cancel) = (Arc::clone(&done), Arc::clone(&cancel));
        let (sender, result) = mpsc::channel();
        let destination = flight.name.clone();
        let saved = self.video_folder.clone();
        std::thread::spawn(move || {
            use super::chunks;
            let folder = saved.join(&name);
            let parts = chunks::parts_folder(&saved, &name);
            let created = std::fs::create_dir_all(&saved).and_then(|()| if pngs { std::fs::create_dir_all(&folder) } else { Ok(()) });
            if let Err(err) = created {
                let _ = sender.send(format!("Could not render the video: {err}"));
                return;
            }
            let video = saved.join(format!("{name}.{}", extension(quality)));
            //The chunks so far, when the video was started before (see `chunks`), or a new manifest; ffmpeg is needed for the video file
            let encoding = Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|status| status.success());
            let nvenc_now = width >= NVENC_SMALLEST.0 && height >= NVENC_SMALLEST.1 && nvenc_works();
            let mut manifest = chunks::read_manifest(&parts).filter(|manifest| manifest.total == total).unwrap_or_else(|| chunks::Manifest::new(manifest_spec, total, chunk_frames, nvenc_now));
            if encoding {
                if manifest.nvenc && !nvenc_now {
                    let _ = sender.send(format!("Could not resume the video: it was started with the GPU's encoder (NVENC), which does not work now; its chunks are kept in {}", parts.display()));
                    return;
                }
                if let Err(err) = std::fs::create_dir_all(&parts).and_then(|()| chunks::write_manifest(&parts, &manifest)) {
                    let _ = sender.send(format!("Could not render the video: {err}"));
                    return;
                }
            }
            //Without ffmpeg only PNG frames are written, all of them
            let first = if encoding { manifest.first_frame() } else { 0 };
            let resumed = first > 0;
            //With Blend frames the frames before the first one are rendered again, so the blend continues as before (they are not written)
            let warm_up = if blend { first.min(chunks::WARM_UP_FRAMES) } else { 0 };
            //The two writers, each with its own queue of frames; a frame counts as done when both have written it
            let (png_done, encoded) = (Arc::new(AtomicUsize::new(first)), Arc::new(AtomicUsize::new(first)));
            let failure = Arc::new(std::sync::Mutex::new(None::<String>));
            let (png_sender, png_receiver) = mpsc::sync_channel::<(usize, Arc<Vec<u8>>)>(FRAMES_WAITING);
            let (ffmpeg_sender, ffmpeg_receiver) = mpsc::sync_channel::<Arc<Vec<u8>>>(FRAMES_WAITING);
            let png_writer = {
                let (folder, png_done, failure, cancel) = (folder.clone(), Arc::clone(&png_done), Arc::clone(&failure), Arc::clone(&thread_cancel));
                //Every frame has its view in its metadata (see `image_metadata`): where the camera is, its limit and its colors, and which frame of which video it is
                let (path, flight, auto_settings, limit, name, png_clock) = (path.clone(), flight.clone(), auto_settings.clone(), m.max_iterations, name.clone(), clock.clone());
                let coloring = if coloring_index == 0 { crate::points_of_interest::Coloring::Hsv } else { crate::points_of_interest::Coloring::Bernstein };
                std::thread::spawn(move || {
                    for (frame, rgb) in png_receiver {
                        let time = png_clock.time(frame);
                        let camera = path.camera_at(time);
                        let palette = flight.palette_at(&path, time);
                        let place = crate::points_of_interest::to_json(&crate::points_of_interest::PointOfInterest {
                            name: format!("{} frame {}", flight.name, frame + 1),
                            description: format!("Frame {} of {total} of the video {name}", frame + 1),
                            location: crate::points_of_interest::Location { real: camera.x, imaginary: camera.y, scale: camera.scale, max_iterations: path.limit_at(time).unwrap_or_else(|| auto_settings.as_ref().map_or(limit, |settings| limit_at(camera.scale, &flight, settings))) },
                            appearance: Some(crate::points_of_interest::Appearance { coloring: Some(coloring), color_channels: Some(mapping), iteration_divide: Some(palette.divide), color_offset: Some(palette.offset), stretch_palette: Some(palette.stretch) }),
                            file_name: String::new(),
                        });
                        if let Err(err) = write_png(&folder.join(format!("frame-{:05}.png", frame + 1)), &rgb, width, height, &place) {
                            *failure.lock().unwrap() = Some(format!("Could not write frame {} of the video: {err}", frame + 1));
                            cancel.store(true, Ordering::Relaxed);
                            return;
                        }
                        png_done.store(frame + 1, Ordering::Relaxed);
                    }
                })
            };
            //The encoder writes chunk after chunk, each a file of its own: a chunk counts in the manifest once ffmpeg has finished it
            let ffmpeg_writer = {
                let (encoded, failure, cancel, parts, mut manifest) = (Arc::clone(&encoded), Arc::clone(&failure), Arc::clone(&thread_cancel), parts.clone(), manifest.clone());
                std::thread::spawn(move || {
                    let mut chunk: Option<(Child, usize)> = None; //The ffmpeg of the chunk under way, and its frames so far
                    let finish_chunk = |child: Child, number: usize, complete: bool, manifest: &mut chunks::Manifest| -> Result<(), String> {
                        let mut child = child;
                        drop(child.stdin.take()); //The end of the frames of the chunk
                        let encoded = child.wait().is_ok_and(|status| status.success());
                        if !complete {
                            let _ = std::fs::remove_file(chunks::chunk_path(&parts, number, extension(quality)));
                            return Ok(());
                        }
                        if !encoded {
                            return Err(format!("ffmpeg could not encode chunk {} of the video", number + 1));
                        }
                        manifest.chunks_done = number + 1;
                        chunks::write_manifest(&parts, manifest).map_err(|err| format!("Could not save the progress of the video: {err}"))
                    };
                    for rgb in ffmpeg_receiver {
                        let number = manifest.chunks_done;
                        if chunk.is_none() {
                            let path = chunks::chunk_path(&parts, number, extension(quality)).display().to_string();
                            match start_ffmpeg((width, height), fps, &path, quality, playback, manifest.nvenc) {
                                Some(child) => chunk = Some((child, 0)),
                                None => {
                                    *failure.lock().unwrap() = Some(String::from("Could not start ffmpeg for the video"));
                                    cancel.store(true, Ordering::Relaxed);
                                    return;
                                }
                            }
                        }
                        let (child, frames) = chunk.as_mut().expect("a chunk under way");
                        if child.stdin.as_mut().is_none_or(|stdin| stdin.write_all(&rgb).is_err()) {
                            *failure.lock().unwrap() = Some(format!("ffmpeg stopped while encoding chunk {} of the video", number + 1));
                            cancel.store(true, Ordering::Relaxed);
                            return;
                        }
                        *frames += 1;
                        let written = encoded.fetch_add(1, Ordering::Relaxed) + 1;
                        if *frames == manifest.chunk_frames || written == manifest.total {
                            let (child, _) = chunk.take().expect("a chunk under way");
                            if let Err(err) = finish_chunk(child, number, true, &mut manifest) {
                                *failure.lock().unwrap() = Some(err);
                                cancel.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                    //Stopped in the middle of a chunk: it is not kept (the manifest only counts finished chunks)
                    if let Some((child, _)) = chunk.take() {
                        let number = manifest.chunks_done;
                        let _ = finish_chunk(child, number, false, &mut manifest);
                    }
                })
            };
            let progress = |png_done: &AtomicUsize, encoded: &AtomicUsize, rendered: usize| {
                let png = if pngs { png_done.load(Ordering::Relaxed) } else { rendered };
                let video = if encoding { encoded.load(Ordering::Relaxed) } else { rendered };
                png.min(video)
            };
            //Renderers take the next frame as soon as the scheduler lets them (each with all the CPU threads and the GPU, so one fills the gaps the other leaves:
            //the end of a layer, the GPU's batches); the frames go to the writers in their order
            let scheduler = &thread_scheduler;
            let (next, (frames_sender, frames_receiver)) = (AtomicUsize::new(first - warm_up), mpsc::sync_channel::<(usize, Option<RenderedFrame>)>(most_at_once));
            let mut rendered = first - warm_up;
            std::thread::scope(|scope| {
                //Measures the frame rate and keeps the number of frames at once that renders fastest (see `Climber`), and the CPU use for the progress line
                let (next_ref, steering_cancel) = (&next, &thread_cancel);
                scope.spawn(move || {
                    let threads = rendering::render_thread_count().max(1) as f64;
                    let mut climber = Climber::new(scheduler.allowed(), most_at_once);
                    let running = || next_ref.load(Ordering::Relaxed) < total && !steering_cancel.load(Ordering::Relaxed);
                    while running() {
                        //A new number of frames at once first fills up: the measurement starts after as many frames as are under way
                        let at_once = scheduler.allowed();
                        let settled = scheduler.finished.load(Ordering::Relaxed) + at_once;
                        while running() && scheduler.finished.load(Ordering::Relaxed) < settled {
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        let (started, first, cpu_before) = (Instant::now(), scheduler.finished.load(Ordering::Relaxed), cpu_time::ProcessTime::try_now().ok());
                        while running() && (started.elapsed() < MIN_WINDOW || scheduler.finished.load(Ordering::Relaxed) < first + 2 * at_once) {
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        if !running() {
                            return;
                        }
                        let seconds = started.elapsed().as_secs_f64();
                        if let Some(cpu) = cpu_before.and_then(|before| cpu_time::ProcessTime::try_now().ok().map(|now| now.duration_since(before).as_secs_f64())) {
                            scheduler.cpu_percent.store((100.0 * cpu / (seconds * threads)).round() as usize, Ordering::Relaxed);
                        }
                        let rate = (scheduler.finished.load(Ordering::Relaxed) - first) as f64 / seconds;
                        scheduler.set_allowed(climber.measured(rate));
                    }
                });
                //The renderers that the scheduler lets start frames (with all the CPU threads, handing rows to the GPU), and with a GPU one more that renders whole
                //frames on the GPU alone: the CPU threads take most rows of a frame they share with the GPU, so the GPU would otherwise be idle much of the time
                for renderer in 0..most_at_once + usize::from(gpu_renderer) {
                    let on_the_gpu = renderer == most_at_once;
                    let (frames_sender, next, thread_cancel, m, path, flight, auto_settings, gpu, clock) = (frames_sender.clone(), &next, &thread_cancel, m.clone(), &path, &flight, &auto_settings, &gpu, &clock);
                    scope.spawn(move || {
                        let (mut plane, mut m) = (ComplexPlane::new(width, height), m);
                        loop {
                            if !on_the_gpu && !scheduler.start_frame(thread_cancel) {
                                return;
                            }
                            let frame = next.fetch_add(1, Ordering::SeqCst);
                            if frame >= total || thread_cancel.load(Ordering::Relaxed) {
                                if !on_the_gpu {
                                    scheduler.finish_frame(false);
                                }
                                return;
                            }
                            let time = clock.time(frame);
                            let camera = path.camera_at(time);
                            plane.set_view(&camera.view());
                            if let Some(limit) = path.limit_at(time) {
                                m.max_iterations = limit; //The limits of the keyframes
                            } else if let Some(settings) = auto_settings {
                                m.max_iterations = limit_at(camera.scale, flight, settings);
                            }
                            m.palette = flight.palette_at(path, time);
                            let mut colors = Vec::new();
                            let render = Render::new(&plane, &m, width, height, coloring_function).supersampling(samples).adaptive(adaptive)
                                .gpu(gpu.as_deref().map(|gpu| (gpu, Duration::ZERO))).fast_gpu(fast_gpu).cancel_with(thread_cancel);
                            //Blended one-sample frames each take another sample (jitter), so the blend gathers samples over the frames, as in a flight
                            let render = if blend && samples <= 1 { render.colors_only_of_sample((frame % 64) as u8) } else { render };
                            let render = if on_the_gpu { render.on_the_gpu_only() } else { render };
                            let completed = render.run(|layer, _| colors = layer.to_vec());
                            if on_the_gpu {
                                scheduler.finished.fetch_add(usize::from(completed), Ordering::Relaxed);
                            } else {
                                scheduler.finish_frame(completed);
                            }
                            if !completed {
                                let _ = frames_sender.send((frame, None));
                                return;
                            }
                            let rendered = if blend {
                                RenderedFrame::Colors(colors, plane.clone()) //Blended in order, see below
                            } else {
                                RenderedFrame::Rgb(colors_to_rgb_bytes(&finish(colors, blur, width, height, samples), mapping))
                            };
                            if frames_sender.send((frame, Some(rendered))).is_err() {
                                return;
                            }
                        }
                    });
                }
                drop(frames_sender);
                let (mut waiting, mut history) = (std::collections::BTreeMap::new(), None);
                for (frame, rendered_frame) in &frames_receiver {
                    let Some(rendered_frame) = rendered_frame else {
                        thread_cancel.store(true, Ordering::Relaxed); //Cancelled: the frames in order so far are kept
                        continue;
                    };
                    waiting.insert(frame, rendered_frame);
                    while let Some(rendered_frame) = waiting.remove(&rendered) {
                        let rgb = Arc::new(match rendered_frame {
                            RenderedFrame::Rgb(rgb) => rgb,
                            RenderedFrame::Colors(colors, plane) => {
                                //Blend frames: into the frames before, moved to this view (see `temporal`), then the blur preset
                                let mut shown = Vec::new();
                                super::temporal::blend(&mut history, &plane, &colors, width, height, &mut shown);
                                colors_to_rgb_bytes(&finish(shown, blur, width, height, samples), mapping)
                            }
                        });
                        //(A full queue makes the renderers wait for the writers; a writer that stopped makes the send fail, and the cancel flag says why.
                        //The frames that warm up the blend of a resumed video were written before)
                        if rendered >= first && ((pngs && png_sender.send((rendered, Arc::clone(&rgb))).is_err()) || (encoding && ffmpeg_sender.send(rgb).is_err())) {
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
                let kept = if encoding { format!("; the finished chunks are kept in {}, and Resume continues the video", parts.display()) } else { String::new() };
                let _ = sender.send(format!("{failure}{kept}"));
                return;
            }
            let frames = progress(&png_done, &encoded, rendered);
            thread_done.store(frames, Ordering::Relaxed);
            let speed = speed_text(speed);
            let written = if pngs { format!(" in {}", folder.display()) } else { String::new() };
            if !encoding {
                let (complete, file) = (if frames == total { "" } else { " (stopped: incomplete)" }, if pngs { " (ffmpeg was not found, so there is no video file; the frames can be encoded with it later)" } else { " (ffmpeg was not found, and PNG frames are off: nothing was written)" });
                let _ = sender.send(format!("Video of the flight: {frames} of {total} frames at {fps} FPS, {speed} speed{complete}{written}{file}"));
                return;
            }
            manifest = chunks::read_manifest(&parts).unwrap_or(manifest);
            if manifest.chunks_done < manifest.chunks() {
                let _ = sender.send(format!("Video of the flight paused at {} of {total} frames: {} of {} chunks are kept in {}, and Resume continues it", manifest.first_frame(), manifest.chunks_done, manifest.chunks(), parts.display()));
                return;
            }
            if !chunks::join_chunks(&parts, manifest.chunks(), extension(quality), &video) {
                let _ = sender.send(format!("Could not join the chunks of the video into {}; they are kept in {}", video.display(), parts.display()));
                return;
            }
            let _ = std::fs::remove_dir_all(&parts);
            let mut with_music = String::new();
            if let Some(music) = &music {
                match super::music::add_track(&video, std::path::Path::new(&music.path)) {
                    Ok(()) => with_music = format!(", with the music of {}", music.path),
                    Err(reason) => with_music = format!(" ({reason})"),
                }
            }
            let resumed = if resumed { format!(" (resumed at frame {})", first + 1) } else { String::new() };
            let _ = sender.send(format!("Video of the flight: {total} of {total} frames at {fps} FPS, {speed} speed{resumed}{written} as {}{with_music}", video.display()));
        });
        self.log(format!("Rendering a video of the flight to {destination}: {total} frames of {width} x {height} at {fps} FPS, {} speed, {}, {}", speed_text(speed), quality.name(), playback.name()));
        self.video_job = Some(VideoJob { done, scheduler, total, cancel, started: Instant::now(), result });
    }

    /// Takes the final message of the video job
    pub(super) fn poll_video(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.video_job else { return };
        ctx.request_repaint_after(Duration::from_millis(200));
        if let Ok(message) = job.result.try_recv() {
            //A video that was cancelled, stopped or could not start has not finished (for the queue)
            let finished = !job.cancel.load(Ordering::Relaxed) && message.starts_with("Video of the flight:") && !message.contains("(stopped");
            self.video_job = None;
            self.log(message);
            self.unfinished_videos = None; //Read again (see `unfinished_ui`)
            self.queue_video_ended(finished);
        }
    }

    /// The progress of a video under way, with its button to cancel it
    pub(super) fn video_progress_ui(&self, ui: &mut egui::Ui) {
        let Some(job) = &self.video_job else { return };
        ui.horizontal(|ui| {
            let done = job.done.load(Ordering::Relaxed);
            let elapsed = job.started.elapsed().as_secs_f64();
            let left = if done > 0 { elapsed / done as f64 * (job.total - done) as f64 } else { 0.0 };
            if ui.button("Pause video").on_hover_text("Stops the video: its finished chunks (10 seconds of video each) are kept, and Resume continues it later, also after the program was closed").clicked() {
                job.cancel.store(true, Ordering::Relaxed);
            }
            let (at_once, cpu) = (job.scheduler.allowed(), job.scheduler.cpu_percent.load(Ordering::Relaxed));
            ui.add(egui::ProgressBar::new(done as f32 / job.total.max(1) as f32).text(format!("Video: {done} of {} frames, {left:.0} s left ({at_once} at once, CPU {cpu}%)", job.total)))
                .on_hover_text("The job renders as many frames at once as renders fastest: it tries one more while the video runs and keeps it when the frame rate rises (up to what the memory allows)");
        });
    }

    /// The length typed in the hours, minutes and seconds boxes of *Fit to length* (issue #179): empty boxes count as 0, and a length needs more than 0 seconds
    pub(super) fn target_length(&self) -> Option<Duration> {
        let mut seconds = 0.0;
        for (input, unit) in self.video_length.iter().zip([3600.0, 60.0, 1.0]) {
            let input = input.trim();
            let value = if input.is_empty() { 0.0 } else { input.replace(',', ".").parse::<f64>().ok().filter(|value| value.is_finite() && *value >= 0.0)? };
            seconds += value * unit;
        }
        (seconds > 0.0).then(|| Duration::from_secs_f64(seconds))
    }

    /// The speed of the video of a flight of `duration`, in percent of the flight's: the speed control's, or with *Fit to length* the speed that makes the video
    /// exactly as long as the typed length; an error says why there is none (no length, or a speed outside `SPEEDS`)
    pub(super) fn video_speed_for(&self, duration: Duration) -> Result<f64, String> {
        if !self.video_fit_length {
            return Ok(f64::from(self.video_speed));
        }
        let length = self.target_length().ok_or_else(|| String::from("Type the length of the video in hours, minutes and seconds"))?;
        let speed = duration.as_secs_f64() * 100.0 / length.as_secs_f64();
        let (slowest, fastest) = (f64::from(*SPEEDS.start()), f64::from(*SPEEDS.end()));
        if (slowest..=fastest).contains(&speed) {
            Ok(speed)
        } else {
            Err(format!("{} needs {} speed, outside {slowest:.0}% to {fastest:.0}%: the flight takes {:.1} s at 100%", length_text(length), speed_text(speed), duration.as_secs_f64()))
        }
    }

    /// Continues video `name` of the videos folder from its first unfinished chunk, with the settings it was started with (see `chunks`)
    pub(super) fn resume_video(&mut self, name: &str) {
        match super::chunks::read_manifest(&super::chunks::parts_folder(&self.video_folder, name)) {
            Some(manifest) => self.start_video_of(VideoSpec { output: Some(name.to_string()), ..manifest.spec }),
            None => self.log(format!("Could not resume the video {name}: its manifest cannot be read")),
        }
    }

    /// The videos that were stopped before they were finished, each with *Resume* (the list is read again when a video ends)
    fn unfinished_ui(&mut self, ui: &mut egui::Ui) {
        let videos = self.unfinished_videos.get_or_insert_with(|| super::chunks::unfinished(&self.video_folder).into_iter().map(|(name, manifest)| (name, manifest.first_frame(), manifest.total)).collect());
        let mut resume = None;
        for (name, done, total) in videos.iter() {
            ui.horizontal(|ui| {
                if ui.button("Resume").on_hover_text("Continues this video from its first unfinished chunk, with the settings it was started with").clicked() {
                    resume = Some(name.clone());
                }
                ui.label(format!("Unfinished: {name} ({done} of {total} frames)"));
            });
        }
        if let Some(name) = resume {
            self.resume_video(&name);
        }
    }

    /// The video controls: the button, the frame rate, the speed, the format and the PNG frames (the progress of a video under way is at the top, see
    /// `video_progress_ui`)
    pub(super) fn video_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Video of the last flight");
        ui.add_enabled_ui(self.video_job.is_none(), |ui| {
            self.saved_flights_ui(ui);
            self.video_settings_ui(ui);
        });
        if self.video_job.is_none() {
            self.unfinished_ui(ui);
        }
        self.queue_ui(ui);
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
            ui.add_enabled(!self.video_fit_length, egui::DragValue::new(&mut self.video_speed).range(SPEEDS).suffix("% speed")).on_hover_text("The speed of the video, in percent of the flight's (1% to 400%; type a value): 10% shows every moment of the flight ten times as long (ten times the frames, and the render time); 200% is twice as fast");
            ui.checkbox(&mut self.video_blend, "Blend frames").on_hover_text("Every frame is blended into the frames before it, moved to its view, as Blend frames does in flights: with the 1x qualities every frame takes another sample, so the video gathers samples over time and moves smoothly for the cost of one sample a frame (fine with 1x + blur). With more samples it smooths the motion");
            ui.checkbox(&mut self.video_pngs, "PNG frames").on_hover_text("Also write every frame as a PNG file to a folder in saved/ (exact pixels, for editing; large). Off by default: only the video file");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.video_fit_length, "Fit to length:").on_hover_text("Sets the speed so the video of the last flight lasts exactly this long (hours, minutes, seconds), instead of the speed in percent");
            ui.add_enabled_ui(self.video_fit_length, |ui| {
                for (input, unit) in self.video_length.iter_mut().zip(["h", "m", "s"]) {
                    ui.add(egui::TextEdit::singleline(input).desired_width(28.0).hint_text(unit));
                    ui.label(unit);
                }
            });
        });
        self.song_ui(ui);
        self.music_ui(ui);
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
                ui.label("Format:");
                egui::ComboBox::from_id_salt("video_playback").selected_text(self.video_playback.name()).show_ui(ui, |ui| {
                    for playback in Playback::ALL {
                        ui.selectable_value(&mut self.video_playback, playback, playback.name()).on_hover_text(match playback {
                            Playback::Maximum => "Full color resolution in 10 bits (HEVC on an NVIDIA GPU, else H.264): the best colors, no banding; plays on computers (VLC and most players), not on phones and TVs",
                            Playback::HighQuality => "HEVC Main 10 (10 bits, half the color resolution), which phones, TVs and browsers play in hardware; single pixels of strong colors get a little softer",
                            Playback::MostCompatible => "H.264 in 8 bits with half the color resolution: plays on anything, even old devices, with the softest colors",
                        });
                    }
                }).response.on_hover_text("The codec, bit depth and color resolution of the video: Maximum for the best colors, High quality to play on phones and TVs, Most compatible to play anywhere");
            });
        });
        //What the video will be: its length and frames are exact, its size an estimate (an upper end: measured on a busy view)
        if let Some(flight) = &self.last_flight {
            let path = flight.path();
            let size = self.save_size.resolve(self.default_save_resolution(), self.view_resolution()).map_or((0, 0), |resolution| (resolution.width, resolution.height));
            let music_length = self.followed_music_length();
            let speed = match (self.followed_music(), music_length) {
                (Some(_), Some(length)) => Ok(path.duration().as_secs_f64() * 100.0 / length.as_secs_f64()), //(The average speed: the video lasts as long as the track)
                (Some(music), None) => Err(format!("The music {} cannot be read", music.path)),
                (None, _) => self.video_speed_for(path.duration()),
            };
            match speed {
                Ok(speed) => {
                    let (frames, length, bytes) = estimate(path.duration(), self.video_fps, speed, size, self.video_quality, self.video_playback);
                    let png = if self.video_pngs { format!(" + PNG frames about {}", megabytes(frames as f64 * (size.0 * size.1) as f64 * 3.0 / 2.0)) } else { String::new() };
                    let fitted = if music_length.is_some() && self.follow_music { String::from(", following the music") } else if self.video_fit_length { format!(" at {} speed", speed_text(speed)) } else { String::new() };
                    ui.label(format!("{} ({:.1} s){fitted}, {frames} frames of {} x {}, up to about {}{png}", length_text(length), length.as_secs_f64(), size.0, size.1, megabytes(bytes)))
                        .on_hover_text("The length and the frames of the video, and the size of its file at most (measured on a busy, detailed view; calm views take far less)");
                }
                Err(reason) => {
                    ui.colored_label(egui::Color32::from_rgb(255, 165, 0), reason);
                }
            }
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
        let flight = VideoFlight { from: Camera { x: -0.75, y: 0.0, scale: 1.0 }, to: Camera { x: -0.7498, y: 0.0203, scale: 6.4e-5 }, pace: Pace::Fast, route: Route::Shortest, constant_speed: 3.0, speed: 1.0, dive_zoom: 1.0, name: String::from("Seahorse"), limits: (300, 5000), palettes: (crate::mandelbrot_set::ColorCycle::CYCLE, crate::mandelbrot_set::ColorCycle::CYCLE), sequence: None };
        assert_eq!(limit_at(1.0, &flight, &settings), 300);
        assert_eq!(limit_at(6.4e-5, &flight, &settings), 5000);
        let middle = limit_at(8e-3, &flight, &settings);
        assert!(middle > 300 && middle < 5000, "{middle}");
    }

    #[test]
    fn a_slower_video_has_more_frames_of_the_same_flight() {
        let second = Duration::from_secs(1);
        assert_eq!(frame_count(second * 2, 30, 100.0), 61, "2 s at 30 FPS: the start, then 60 more frames");
        assert_eq!(frame_count(second * 2, 30, 10.0), 601, "10%: ten times the frames");
        assert_eq!(frame_count(second * 2, 240, 400.0), 121);
        assert_eq!(frame_time(30, 30, 100.0), second);
        assert_eq!(frame_time(300, 30, 10.0), second, "at 10% frame 300 (10 s of video) is 1 s into the flight");
        assert_eq!(frame_time(60, 120, 50.0), Duration::from_millis(250));
        assert!(FRAME_RATES.starts_with(&[24, 30, 60]) && FRAME_RATES.ends_with(&[90, 120, 144, 240]));
    }

    #[test]
    fn the_tiers_keep_a_fixed_quality_for_the_player_they_are_for() {
        let has = |arguments: &[String], wanted: &[&str]| wanted.iter().all(|wanted| arguments.iter().any(|argument| argument.contains(wanted)));
        //Maximum: full color in 10 bits, on the GPU or the CPU; a fixed quantizer that follows the tier
        assert!(has(&arguments(VideoQuality::VeryHigh, Playback::Maximum, true, 60), &["hevc_nvenc", "yuv444p16le", "constqp", "20", "bt709"]));
        assert!(has(&arguments(VideoQuality::UltraHigh, Playback::Maximum, true, 60), &["16"]));
        assert!(has(&arguments(VideoQuality::VeryHigh, Playback::Maximum, false, 60), &["libx264", "yuv444p10le", "14"]));
        //High quality: HEVC Main 10, half the color resolution made with a sharp filter; Most compatible: 8-bit H.264
        assert!(has(&arguments(VideoQuality::High, Playback::HighQuality, true, 60), &["hevc_nvenc", "p010le", "main10", "lanczos"]));
        assert!(has(&arguments(VideoQuality::High, Playback::HighQuality, false, 60), &["libx265", "yuv420p10le"]));
        assert!(has(&arguments(VideoQuality::High, Playback::MostCompatible, true, 60), &["libx264", "yuv420p", "18"]));
        //A keyframe every half second and the index at the start; lossless is FFV1 whatever the player
        assert!(has(&arguments(VideoQuality::High, Playback::MostCompatible, true, 60), &["-g", "30", "+faststart"]));
        assert_eq!(arguments(VideoQuality::Lossless, Playback::HighQuality, true, 60), ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]);
        assert_eq!((extension(VideoQuality::Lossless), extension(VideoQuality::NearLossless)), ("mkv", "mp4"));
    }

    #[test]
    fn a_video_that_fits_a_length_has_exactly_its_frames() {
        //200 s at 60 FPS: 12,000 frame intervals, and the frames at both ends
        let flight = Duration::from_secs_f64(37.3);
        let speed = flight.as_secs_f64() * 100.0 / 200.0;
        assert_eq!(frame_count(flight, 60, speed), 12_001);
        assert_eq!(frame_count(Duration::from_secs(13), 60, 50.0), 1561, "a whole speed as before");
        assert_eq!((speed_text(18.65), speed_text(50.0), length_text(Duration::from_secs(200)), length_text(Duration::from_secs(3723))), (String::from("18.6%"), String::from("50%"), String::from("3:20"), String::from("1:02:03")));
    }

    #[test]
    fn the_climber_keeps_one_more_frame_at_once_only_when_it_is_faster() {
        let mut climber = Climber::new(2, 4);
        assert_eq!(climber.measured(10.0), 3, "after a measurement at 2: try 3");
        assert_eq!(climber.measured(12.0), 4, "3 is faster: keep it, try 4");
        assert_eq!(climber.measured(12.1), 3, "4 is not faster (by less than the noise): back to 3");
        for _ in 0..HOLD_WINDOWS {
            assert_eq!(climber.measured(12.0), 3, "and stay there for a while");
        }
        assert_eq!(climber.measured(11.0), 4, "then measure 3 again and try 4 again");
        assert_eq!(climber.measured(13.0), 4, "now 4 pays, and it is the most the memory allows");
        assert_eq!(climber.measured(13.0), 4);
        assert_eq!(Climber::new(2, 1).measured(5.0), 1, "never more than the most");
    }

    #[test]
    fn the_scheduler_lets_as_many_frames_start_as_allowed() {
        let scheduler = FrameScheduler::new(2);
        let cancel = AtomicBool::new(false);
        assert!(scheduler.start_frame(&cancel) && scheduler.start_frame(&cancel));
        cancel.store(true, Ordering::Relaxed);
        assert!(!scheduler.start_frame(&cancel), "no free place, cancelled");
        scheduler.finish_frame(true);
        assert!(scheduler.start_frame(&AtomicBool::new(false)), "a finished frame frees a place");
        scheduler.set_allowed(3);
        assert!(scheduler.start_frame(&AtomicBool::new(false)), "one more allowed");
        assert_eq!(scheduler.finished.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_memory_bounds_the_frames_at_once() {
        assert!((1..=MAX_AT_ONCE).contains(&frames_memory_allows(3840 * 2160)));
        assert_eq!(frames_memory_allows(usize::MAX / 64), 1, "a frame too large for the memory still renders, one at a time");
    }

    #[test]
    fn the_formats_are_found_by_their_names_and_by_the_names_they_had_before() {
        for playback in Playback::ALL {
            assert_eq!(Playback::of_name(playback.name()), Some(playback));
        }
        assert_eq!(Playback::of_name("Computers"), Some(Playback::Maximum));
        assert_eq!(Playback::of_name("Phones and TVs"), Some(Playback::HighQuality));
        assert_eq!(Playback::of_name("Everywhere"), Some(Playback::MostCompatible));
        assert_eq!(Playback::of_name("Somewhere"), None);
    }

    #[test]
    fn the_estimate_has_the_exact_frames_and_length_and_a_size_that_grows_with_the_tier() {
        let (frames, length, high) = estimate(Duration::from_secs(13), 60, 50.0, (1920, 1080), VideoQuality::High, Playback::Maximum);
        assert_eq!((frames, length.as_secs()), (1561, 26), "13 s at half speed: 26 s, at 60 FPS");
        let sizes: Vec<f64> = VideoQuality::ALL.iter().map(|quality| estimate(Duration::from_secs(13), 60, 50.0, (1920, 1080), *quality, Playback::Maximum).2).collect();
        assert!(sizes.windows(2).all(|pair| pair[0] < pair[1]) && (sizes[0] - high).abs() < 1.0);
        //Lossless: about 3.2 GB, as Jort measured (3 GB for 26 s)
        assert!((2.5e9..4e9).contains(&sizes[4]), "{}", sizes[4]);
        assert_eq!((megabytes(3.2e9), megabytes(65e6)), (String::from("3.2 GB"), String::from("65 MB")));
    }
}

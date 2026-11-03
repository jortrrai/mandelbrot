//! **Music plugin (branch `plugin/music`, not in the standard app):** this module is kept here as an optional plugin, off by default: the speed that
//! follows the loudness (strength 0 at first), *Fit to the music* and *Pulse on the beat* (off at first). Taken out of develop because the beat detection
//! triggers on hi-hats and noise, misses the beats of some songs and the pacing does not feel right yet (Jort, 2026-11-03). Low priority: not improved here.
//!
//! Music that steers a flight (issue #185): with *Follow the music* a video of a flight lasts exactly as long as its audio track, moves faster through the
//! loud, energetic parts and slower through the calm ones, and has the track in its file.
//!
//! The track is decoded by ffmpeg (mono, `SAMPLE_RATE`); its loudness (RMS) every `STEP` seconds is smoothed over `SMOOTHING` seconds, so the speed follows
//! the swells and sections rather than every beat, and scaled between the quiet and the loud parts of the track (its 5th and 95th percentiles). The speed of
//! the flight at a moment is `1 - strength + strength · (MIN_SPEED + (MAX_SPEED - MIN_SPEED) · loudness)`, and the flight's clock is that speed added up,
//! scaled so the flight ends with the track. Rendering the same track again gives the same frames (also when a video is resumed).
//!
//! *Pulse on the beat* (issue #221) makes the view boom like a speaker cone on every kick drum of the track: the kicks are found in its bass (40 to 120 Hz, see
//! `kicks_of_samples`), and on each the view zooms in quickly and settles back with a slight bounce past rest (see `pulse_shape`), on top of everything else
//! the video does.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The sample rate the track is analyzed at: plenty for its loudness
const SAMPLE_RATE: usize = 8000;

/// The loudness is measured every `STEP` seconds
const STEP: f64 = 0.05;

/// The loudness is smoothed over this many seconds (a centered moving average)
const SMOOTHING: f64 = 2.0;

/// The speed of the flight in the quietest and the loudest parts, at full strength (relative: the flight's clock is scaled to end with the track)
const MIN_SPEED: f64 = 0.25;
const MAX_SPEED: f64 = 1.75;

/// The music of a video: the audio file, how much it steers the speed (0: an even speed, 1: fully), and how far the view zooms in on a kick (0: no pulse,
/// 0.05: 5%; see `BeatPulse`)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Music {
    pub(super) path: String,
    pub(super) strength: f64,
    /// (Absent in the manifests of videos started before #221: no pulse)
    #[serde(default)]
    pub(super) pulse: f64,
}

/// The sample rate the kicks are found at, and the step of the envelope of the bass, in samples (10 ms)
const BEAT_RATE: usize = 8000;
const BEAT_HOP: usize = 80;

/// The band of a kick drum, in Hz
const KICK_BAND: (f64, f64) = (40.0, 120.0);

/// Two kicks are at least this far apart, in seconds (at most 300 a minute)
const KICK_GAP: f64 = 0.2;

/// The pulse of a kick: how long it pushes in, how fast it decays and how long one swing of its bounce takes, in seconds, and when it is over
const PULSE_ATTACK: f64 = 0.03;
const PULSE_DECAY: f64 = 0.07;
const PULSE_SWING: f64 = 0.4;
const PULSE_END: f64 = 0.6;

/// The strength of the pulse at first (5% zoom) and its range
pub(super) const PULSE_STRENGTH: f64 = 0.02; //(Jort, review 8: "5% is a little bit crazy, 2% is really nice")
pub(super) const PULSE_STRENGTHS: std::ops::RangeInclusive<f64> = 0.0..=0.3;

/// The shape of one pulse `seconds` after its kick: 0 before it, up to 1 in `PULSE_ATTACK` (smoothly), then back to rest within about 0.2 s, past it by about
/// 8% at the bottom of the bounce (a speaker cone), and 0 after `PULSE_END`
pub(super) fn pulse_shape(seconds: f64) -> f64 {
    if !(0.0..PULSE_END).contains(&seconds) {
        return 0.0;
    }
    if seconds < PULSE_ATTACK {
        let rise = seconds / PULSE_ATTACK;
        return rise * rise * (3.0 - 2.0 * rise);
    }
    let after = seconds - PULSE_ATTACK;
    (-after / PULSE_DECAY).exp() * (std::f64::consts::TAU * after / PULSE_SWING).cos()
}

/// The kicks of a track: the moments of its kick drums, each with its weight (harder kicks a little more)
#[derive(Clone, Debug, Default)]
pub(super) struct BeatPulse {
    kicks: Vec<(f64, f64)>,
}

impl BeatPulse {
    /// The kicks found in the samples of a track (see `kicks_of_samples`)
    pub(super) fn of_samples(samples: &[f64], rate: usize) -> BeatPulse {
        BeatPulse { kicks: kicks_of_samples(samples, rate) }
    }

    /// The moments of the kicks, in seconds
    pub(super) fn moments(&self) -> Vec<f64> {
        self.kicks.iter().map(|(moment, _)| *moment).collect()
    }

    /// How far the view is zoomed in by the pulses `seconds` into the track, at `strength` (0.05: 5% at the top of a normal kick); negative at the bottom of
    /// a bounce. The scale of the view is divided by `1 + zoom`
    pub(super) fn zoom_at(&self, seconds: f64, strength: f64) -> f64 {
        let first = self.kicks.partition_point(|(moment, _)| *moment < seconds - PULSE_END);
        self.kicks[first..].iter().take_while(|(moment, _)| *moment <= seconds).map(|(moment, weight)| strength * weight * pulse_shape(seconds - moment)).sum()
    }
}

/// A biquad filter (RBJ's cookbook): `low` true for a low-pass, false for a high-pass at `frequency` Hz, run over `samples` at `rate`
fn biquad(samples: &[f64], rate: usize, frequency: f64, low: bool) -> Vec<f64> {
    let omega = std::f64::consts::TAU * frequency / rate as f64;
    let (sin, cos) = omega.sin_cos();
    let alpha = sin / std::f64::consts::SQRT_2; //(Q = 1/√2: Butterworth)
    let (b0, b1) = if low { ((1.0 - cos) / 2.0, 1.0 - cos) } else { (f64::midpoint(1.0, cos), -(1.0 + cos)) };
    let (a0, a1, a2) = (1.0 + alpha, -2.0 * cos, 1.0 - alpha);
    let (b0, b1, b2, a1, a2) = (b0 / a0, b1 / a0, b0 / a0, a1 / a0, a2 / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
    samples.iter().map(|&x| {
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        (x2, x1, y2, y1) = (x1, x, y1, y);
        y
    }).collect()
}

/// The kicks of the mono `samples` at `rate`: the moments (in seconds) where the loudness of the bass (`KICK_BAND`, every 10 ms, in decibels) jumps up much
/// more than it does around it, at least `KICK_GAP` apart, each with its weight: from 0.7 for a soft kick to 1.3 for a hard one (1 for a typical one)
pub(super) fn kicks_of_samples(samples: &[f64], rate: usize) -> Vec<(f64, f64)> {
    let hop = (rate * BEAT_HOP / BEAT_RATE).max(1);
    let bass = biquad(&biquad(samples, rate, KICK_BAND.1, true), rate, KICK_BAND.0, false);
    let level: Vec<f64> = bass.chunks(hop).map(|chunk| 10.0 * (chunk.iter().map(|sample| sample * sample).sum::<f64>() / chunk.len() as f64 + 1e-10).log10()).collect();
    //The rise of the level (onset strength): how far it is above the highest of the two steps before. Before the track it is as quiet as the quietest bass of
    //the track (its 5th percentile), so a kick right at its start counts, and a track that starts with its bass at its usual level does not
    let mut quietest = level.clone();
    quietest.sort_by(f64::total_cmp);
    let before_the_track = quietest.get(quietest.len() / 20).copied().unwrap_or(0.0);
    let rise: Vec<f64> = (0..level.len()).map(|index| (level[index] - level[index.saturating_sub(2)..index].iter().copied().fold(before_the_track, f64::max)).max(0.0)).collect();
    let step = hop as f64 / rate as f64;
    let (near, around) = ((0.05 / step).round() as usize, (0.5 / step).round() as usize);
    let mut sorted = rise.clone();
    sorted.sort_by(f64::total_cmp);
    let typical = sorted.get(((sorted.len().saturating_sub(1)) as f64 * 0.99) as usize).copied().unwrap_or(0.0);
    let mut kicks: Vec<(f64, f64)> = Vec::new();
    for index in 0..rise.len() {
        let window = |reach: usize| &rise[index.saturating_sub(reach)..(index + reach + 1).min(rise.len())];
        let peak = window(near).iter().all(|other| *other <= rise[index]);
        let mean = window(around).iter().sum::<f64>() / window(around).len() as f64;
        //(A clear jump: well above the rises around it, and at least 6 dB and a quarter of the typical kick, so noise and the tails of notes are not kicks)
        if peak && rise[index] > 2.0 * mean && rise[index] >= 6.0_f64.max(0.25 * typical) {
            let moment = index as f64 * step;
            match kicks.last_mut() {
                Some((last, strength)) if moment - *last < KICK_GAP => {
                    if rise[index] > *strength {
                        (*last, *strength) = (moment, rise[index]);
                    }
                }
                _ => kicks.push((moment, rise[index])),
            }
        }
    }
    let mut strengths: Vec<f64> = kicks.iter().map(|(_, strength)| *strength).collect();
    strengths.sort_by(f64::total_cmp);
    let median = strengths.get(strengths.len() / 2).copied().unwrap_or(1.0).max(f64::MIN_POSITIVE);
    kicks.into_iter().map(|(moment, strength)| (moment, (strength / median).clamp(0.7, 1.3))).collect()
}

/// The kicks of the track at `path` (ffmpeg decodes it), see `kicks_of_samples`
pub(super) fn pulse_of_track(path: &Path) -> Result<BeatPulse, String> {
    Ok(BeatPulse::of_samples(&samples_of_track(path, BEAT_RATE)?, BEAT_RATE))
}

/// The clock of a flight that follows a track: the share of the flight (0 to 1) at every `STEP` of the track
#[derive(Clone, Debug)]
pub(super) struct MusicClock {
    shares: Vec<f64>,
}

impl MusicClock {
    /// The length of the track
    pub(super) fn length(&self) -> Duration {
        Duration::from_secs_f64((self.shares.len().saturating_sub(1)) as f64 * STEP)
    }

    /// The share of the flight `seconds` into the track (0 at the start, 1 at its end)
    pub(super) fn share_at(&self, seconds: f64) -> f64 {
        let position = (seconds / STEP).max(0.0);
        let index = position.floor() as usize;
        if index + 1 >= self.shares.len() {
            return 1.0;
        }
        let fraction = position - index as f64;
        self.shares[index] + (self.shares[index + 1] - self.shares[index]) * fraction
    }
}

/// The clock of the loudness `levels` (one every `STEP`) at `strength`, see the module
pub(super) fn clock_of_levels(levels: &[f64], strength: f64) -> MusicClock {
    let reach = ((SMOOTHING / STEP / 2.0).round() as usize).max(1);
    //A centered moving average, with running sums
    let mut sums = vec![0.0; levels.len() + 1];
    for (index, level) in levels.iter().enumerate() {
        sums[index + 1] = sums[index] + level;
    }
    let smooth: Vec<f64> = (0..levels.len()).map(|index| {
        let (low, high) = (index.saturating_sub(reach), (index + reach + 1).min(levels.len()));
        (sums[high] - sums[low]) / (high - low) as f64
    }).collect();
    let mut sorted = smooth.clone();
    sorted.sort_by(f64::total_cmp);
    let percentile = |share: f64| sorted.get(((sorted.len().saturating_sub(1)) as f64 * share).round() as usize).copied().unwrap_or(0.0);
    let (quiet, loud) = (percentile(0.05), percentile(0.95));
    let strength = strength.clamp(0.0, 1.0);
    let speeds: Vec<f64> = smooth.iter().map(|level| {
        let loudness = if loud > quiet { ((level - quiet) / (loud - quiet)).clamp(0.0, 1.0) } else { 0.5 };
        1.0 - strength + strength * (MIN_SPEED + (MAX_SPEED - MIN_SPEED) * loudness)
    }).collect();
    //The clock: the speeds added up, from 0 at the start to 1 at the end of the track
    let mut shares = Vec::with_capacity(speeds.len() + 1);
    shares.push(0.0);
    for speed in &speeds {
        shares.push(shares.last().copied().unwrap_or(0.0) + speed);
    }
    let total = shares.last().copied().unwrap_or(0.0).max(f64::MIN_POSITIVE);
    for share in &mut shares {
        *share /= total;
    }
    MusicClock { shares }
}

/// The clock of the track at `path`, see the module (ffmpeg decodes it)
pub(super) fn clock_of_track(path: &Path, strength: f64) -> Result<MusicClock, String> {
    Ok(clock_of_levels(&levels_of_track(path)?, strength))
}

/// The samples of the track at `path`, mono at `rate` (ffmpeg decodes it)
fn samples_of_track(path: &Path, rate: usize) -> Result<Vec<f64>, String> {
    let output = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-ac", "1", "-ar", &rate.to_string(), "-f", "s16le", "-"])
        .stdin(std::process::Stdio::null()).output().map_err(|err| format!("Could not start ffmpeg to read the music: {err}"))?;
    if !output.status.success() || output.stdout.len() < 4 {
        return Err(format!("ffmpeg could not read the music {}: {}", path.display(), String::from_utf8_lossy(&output.stderr).trim()));
    }
    Ok(output.stdout.as_chunks::<2>().0.iter().map(|pair| f64::from(i16::from_le_bytes(*pair)) / 32768.0).collect())
}

/// The loudness (RMS) of the track at `path` every `STEP` seconds (ffmpeg decodes it)
fn levels_of_track(path: &Path) -> Result<Vec<f64>, String> {
    let samples = samples_of_track(path, SAMPLE_RATE)?;
    let per_step = (SAMPLE_RATE as f64 * STEP) as usize;
    Ok(samples.chunks(per_step).map(|chunk| (chunk.iter().map(|sample| sample * sample).sum::<f64>() / chunk.len() as f64).sqrt()).collect())
}

/// The loudness is smoothed over this many seconds to find the swells of a track (its sections, not its beats), see `swells_of_levels`
const SWELL_SMOOTHING: f64 = 4.0;

/// The `count` most prominent swells of the loudness `levels` (one every `STEP`): the moments (in seconds) where the smoothed loudness peaks highest above the
/// quiet around it, at least an even share of the track apart, in order. The keyframes of a sequence can land on them (issue #185, see `sequence`)
pub(super) fn swells_of_levels(levels: &[f64], count: usize) -> Vec<f64> {
    if count == 0 || levels.len() < 3 {
        return Vec::new();
    }
    let reach = ((SWELL_SMOOTHING / STEP / 2.0).round() as usize).max(1);
    let mut sums = vec![0.0; levels.len() + 1];
    for (index, level) in levels.iter().enumerate() {
        sums[index + 1] = sums[index] + level;
    }
    let smooth: Vec<f64> = (0..levels.len()).map(|index| {
        let (low, high) = (index.saturating_sub(reach), (index + reach + 1).min(levels.len()));
        (sums[high] - sums[low]) / (high - low) as f64
    }).collect();
    //Every local maximum (the middle of a flat top), with its prominence: how far it rises above the lower of the lowest points on either side before something
    //higher
    let flat = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
    let mut peaks: Vec<(usize, f64)> = (1..smooth.len() - 1).filter_map(|start| {
        if smooth[start] <= smooth[start - 1] || flat(smooth[start], smooth[start - 1]) {
            return None;
        }
        let end = (start..smooth.len()).take_while(|index| flat(smooth[*index], smooth[start])).last().unwrap_or(start);
        (end + 1 < smooth.len() && smooth[end + 1] < smooth[end]).then_some((start, end))
    }).map(|(start, end)| {
        let left = smooth[..start].iter().rev().take_while(|level| **level <= smooth[start]).fold(smooth[start], |low, level| low.min(*level));
        let right = smooth[end + 1..].iter().take_while(|level| **level <= smooth[start]).fold(smooth[start], |low, level| low.min(*level));
        (usize::midpoint(start, end), smooth[start] - left.max(right))
    }).collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let spacing = (smooth.len() / (2 * (count + 1))).max(1);
    let mut chosen: Vec<usize> = Vec::new();
    for (index, _) in peaks {
        if chosen.len() == count {
            break;
        }
        if chosen.iter().all(|other| other.abs_diff(index) >= spacing) && index >= spacing && index + spacing <= smooth.len() {
            chosen.push(index);
        }
    }
    chosen.sort_unstable();
    chosen.into_iter().map(|index| index as f64 * STEP).collect()
}

/// The swells of the track at `path` (see `swells_of_levels`), and its length in seconds
pub(super) fn swells_of_track(path: &Path, count: usize) -> Result<(Vec<f64>, f64), String> {
    let levels = levels_of_track(path)?;
    Ok((swells_of_levels(&levels, count), levels.len() as f64 * STEP))
}

/// Puts the track `music` into the video `video` (which then has it as its sound; a track longer than the video is cut at its end)
pub(super) fn add_track(video: &Path, music: &Path) -> Result<(), String> {
    let extension = video.extension().and_then(|extension| extension.to_str()).unwrap_or("mp4");
    let with_track = video.with_extension(format!("with-music.{extension}"));
    let audio: &[&str] = if extension == "mkv" { &["-c:a", "flac"] } else { &["-c:a", "aac", "-b:a", "320k"] };
    let mut arguments: Vec<String> = ["-v", "error", "-y", "-i"].map(String::from).to_vec();
    arguments.push(video.display().to_string());
    arguments.push(String::from("-i"));
    arguments.push(music.display().to_string());
    arguments.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"].map(String::from));
    arguments.extend(audio.iter().map(|argument| (*argument).to_string()));
    arguments.push(String::from("-shortest"));
    if extension == "mp4" {
        arguments.extend(["-movflags", "+faststart"].map(String::from));
    }
    arguments.push(with_track.display().to_string());
    let added = std::process::Command::new("ffmpeg").args(&arguments).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .status().is_ok_and(|status| status.success());
    if !added {
        let _ = std::fs::remove_file(&with_track);
        return Err(format!("ffmpeg could not add the music to {}", video.display()));
    }
    std::fs::rename(&with_track, video).map_err(|err| format!("Could not replace {} with the video with music: {err}", video.display()))
}

/// The file extensions of audio files that can be dropped on the window as the music of videos
const AUDIO_EXTENSIONS: [&str; 8] = ["mp3", "wav", "flac", "ogg", "opus", "m4a", "aac", "wma"];

/// Whether `path` is an audio file, by its extension
pub(super) fn is_audio(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()).is_some_and(|extension| AUDIO_EXTENSIONS.iter().any(|audio| audio.eq_ignore_ascii_case(extension)))
}

/// The length of the track at `path`, as ffprobe reads it
fn track_length(path: &Path) -> Option<Duration> {
    let output = std::process::Command::new("ffprobe").args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]).arg(path).stdin(std::process::Stdio::null()).output().ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse::<f64>().ok().filter(|seconds| seconds.is_finite() && *seconds > 0.0).map(Duration::from_secs_f64)
}

impl super::MandelbrotApp {
    /// The music the next video follows: with *Follow the music* and a path
    pub(super) fn followed_music(&self) -> Option<Music> {
        let pulse = if self.beat_pulse { self.pulse_strength } else { 0.0 };
        (self.follow_music && !self.music_path.trim().is_empty()).then(|| Music { path: self.music_path.trim().to_string(), strength: self.music_strength, pulse })
    }

    /// The length of the followed track (read once per path), if it can be read
    pub(super) fn followed_music_length(&mut self) -> Option<Duration> {
        let path = self.followed_music()?.path;
        if self.music_length.as_ref().is_none_or(|(known, _)| *known != path) {
            self.music_length = Some((path.clone(), track_length(Path::new(&path))));
        }
        self.music_length.as_ref().and_then(|(_, length)| *length)
    }

    /// Takes the audio file `path` (dropped on the window, or opened with *Open song...*) as the music of videos, which then last as long as it
    pub(super) fn use_music_file(&mut self, path: &Path) {
        self.music_path = path.display().to_string();
        self.follow_music = true;
        //The length boxes show the song's length (issue #221): the video lasts as long as it
        if let Some(length) = track_length(path) {
            let seconds = length.as_secs_f64();
            let (hours, minutes) = ((seconds / 3600.0).floor(), ((seconds % 3600.0) / 60.0).floor());
            let rest = seconds - hours * 3600.0 - minutes * 60.0;
            self.video_length = [if hours > 0.0 { format!("{hours}") } else { String::new() }, format!("{minutes}"), format!("{:.2}", rest).trim_end_matches('0').trim_end_matches('.').to_string()];
        }
        self.log(format!("Videos follow the music of {} now (see Follow the music)", path.display()));
    }

    /// The music field, *Follow the music* and its strength
    pub(super) fn music_ui(&mut self, ui: &mut eframe::egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.follow_music, "Follow the music:").on_hover_text("The video lasts as long as the track, moves faster through its loud, energetic parts and slower through its calm ones, and has the track as its sound");
            ui.add(eframe::egui::TextEdit::singleline(&mut self.music_path).desired_width(160.0).hint_text("audio file (or drop it here)"))
                .on_hover_text("The path of an audio file (MP3, WAV, FLAC, ...); an audio file dropped on the window lands here");
            //(The system's file chooser, modal: the window waits for it. Not in the Android app, which has no file chooser of rfd)
            #[cfg(not(target_os = "android"))]
            if ui.button("Open song...").on_hover_text("Chooses an audio file as the music of videos: they get it as their sound and last exactly as long as it").clicked() {
                let mut dialog = rfd::FileDialog::new().set_title("Open a song for videos").add_filter("Audio", &AUDIO_EXTENSIONS);
                if let Some(home) = super::sharing::home_folder() {
                    dialog = dialog.set_directory(home);
                }
                if let Some(path) = dialog.pick_file() {
                    self.use_music_file(&path);
                }
            }
            ui.add_enabled(self.follow_music, eframe::egui::Slider::new(&mut self.music_strength, 0.0..=1.0).text("strength").custom_formatter(|value, _| format!("{:.0}%", value * 100.0)))
                .on_hover_text("How much the music changes the speed: 0% an even speed all the way, 100% from a quarter of the average speed in the calmest parts to 1.75 times it in the loudest");
        });
        ui.horizontal(|ui| {
            ui.add_enabled(self.follow_music, eframe::egui::Checkbox::new(&mut self.beat_pulse, "Pulse on the beat"))
                .on_hover_text("On every kick drum of the music the view booms like a speaker: it zooms in quickly and settles back with a slight bounce, on top of everything else the video does (Fast dives are off then)");
            ui.add_enabled(self.follow_music && self.beat_pulse, eframe::egui::Slider::new(&mut self.pulse_strength, PULSE_STRENGTHS).text("zoom").custom_formatter(|value, _| format!("{:.1}%", value * 100.0)))
                .on_hover_text("How far the view zooms in on a typical kick; harder kicks a little more, softer ones less");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pulse_pushes_in_fast_and_settles_back_with_a_slight_bounce() {
        assert_eq!((pulse_shape(-0.01), pulse_shape(0.0)), (0.0, 0.0));
        assert!((pulse_shape(PULSE_ATTACK) - 1.0).abs() < 1e-12, "the top after the attack");
        let shape: Vec<f64> = (0..600).map(|ms| pulse_shape(f64::from(ms) / 1000.0)).collect();
        let lowest = shape.iter().copied().fold(f64::MAX, f64::min);
        assert!((-0.12..-0.04).contains(&lowest), "a slight bounce past rest: {lowest}");
        assert!(shape[250..].iter().all(|value| value.abs() < 0.05) && shape[400..].iter().all(|value| value.abs() < 0.01), "at rest again after about a quarter of a second");
        assert_eq!(pulse_shape(PULSE_END), 0.0);
    }

    /// Eight seconds at `rate`: a kick drum (60 Hz, decaying) every half second, hi-hats (2.5 kHz) between them, and a quiet chord all along
    fn drum_loop(rate: usize) -> (Vec<f64>, Vec<f64>) {
        let kicks: Vec<f64> = (0..16).map(|beat| 0.25 + 0.5 * f64::from(beat)).collect();
        let samples = (0..8 * rate).map(|index| {
            let t = index as f64 / rate as f64;
            //(The first kick of every bar of four is louder)
            let kick: f64 = kicks.iter().enumerate().filter(|(_, kick)| t >= **kick).map(|(beat, kick)| (t - kick, if beat % 4 == 0 { 0.8 } else { 0.5 })).map(|(age, loud)| loud * (-age / 0.12).exp() * (std::f64::consts::TAU * 60.0 * age).sin()).sum();
            let hat: f64 = kicks.iter().map(|kick| t - kick - 0.25).filter(|age| *age >= 0.0).map(|age| 0.3 * (-age / 0.03).exp() * (std::f64::consts::TAU * 2500.0 * age).sin()).sum();
            let chord = 0.05 * ((std::f64::consts::TAU * 440.0 * t).sin() + (std::f64::consts::TAU * 554.0 * t).sin());
            kick + hat + chord
        }).collect();
        (samples, kicks)
    }

    #[test]
    fn the_kicks_are_found_in_the_bass_and_not_on_the_hi_hats() {
        let (samples, expected) = drum_loop(BEAT_RATE);
        let kicks = kicks_of_samples(&samples, BEAT_RATE);
        let found: Vec<f64> = kicks.iter().map(|(moment, _)| *moment).collect();
        assert_eq!(found.len(), expected.len(), "every kick once, no hi-hat: {found:?}");
        for (found, expected) in found.iter().zip(&expected) {
            assert!((found - expected).abs() <= 0.03, "on the beat: {found} for {expected}");
        }
        //The first kick of every bar is harder: its pulse is bigger
        assert!(kicks[0].1 > kicks[1].1 && kicks[4].1 > kicks[5].1, "{kicks:?}");
        let pulse = BeatPulse::of_samples(&samples, BEAT_RATE);
        let top = pulse.zoom_at(found[1] + PULSE_ATTACK, 0.05);
        assert!((0.03..=0.07).contains(&top), "about 5% at the top of a kick: {top}");
        assert!(pulse.zoom_at(found[1] - 0.05, 0.05).abs() < 1e-3, "still between the kicks");
    }

    #[test]
    #[ignore = "the kicks of a real track (MANDELBROT_TRACK=song.mp3), by hand"]
    fn the_kicks_of_a_real_track() {
        let path = std::env::var("MANDELBROT_TRACK").expect("MANDELBROT_TRACK");
        let moments = pulse_of_track(Path::new(&path)).unwrap().moments();
        let gaps: Vec<f64> = moments.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let mut sorted = gaps;
        sorted.sort_by(f64::total_cmp);
        let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        println!("{} kicks, median gap {median:.3} s ({:.1} a minute); the first: {:?}", moments.len(), 60.0 / median.max(1e-9), &moments[..moments.len().min(24)]);
    }

    #[test]
    fn the_swells_are_the_most_prominent_peaks_of_the_smoothed_loudness_in_order() {
        //60 seconds: quiet, with swells at 15 s (small), 30 s (big) and 45 s (medium)
        let levels: Vec<f64> = (0..1200).map(|step| {
            let seconds = f64::from(step) * STEP;
            let swell = |at: f64, height: f64| height * (-((seconds - at) / 3.0).powi(2)).exp();
            0.05 + swell(15.0, 0.1) + swell(30.0, 0.6) + swell(45.0, 0.3)
        }).collect();
        let near = |found: f64, at: f64| (found - at).abs() < 1.0;
        let two = swells_of_levels(&levels, 2);
        assert!(two.len() == 2 && near(two[0], 30.0) && near(two[1], 45.0), "the two biggest, in order: {two:?}");
        let three = swells_of_levels(&levels, 3);
        assert!(three.len() == 3 && near(three[0], 15.0), "{three:?}");
        assert_eq!(swells_of_levels(&levels, 0), Vec::<f64>::new());
    }

    #[test]
    fn the_flight_goes_faster_through_the_loud_part_and_ends_with_the_track() {
        //Ten seconds quiet, ten seconds loud
        let levels: Vec<f64> = (0..400).map(|step| if step < 200 { 0.01 } else { 0.5 }).collect();
        let clock = clock_of_levels(&levels, 1.0);
        assert!((clock.length().as_secs_f64() - 20.0).abs() < 1e-9);
        let (half, end) = (clock.share_at(10.0), clock.share_at(20.0));
        assert!(half < 0.25, "the quiet half covers little of the flight: {half}");
        assert!((end - 1.0).abs() < 1e-12 && clock.share_at(0.0).abs() < 1e-12);
        assert!((0..2000).map(|step| clock.share_at(f64::from(step) / 100.0)).collect::<Vec<_>>().windows(2).all(|pair| pair[1] >= pair[0]), "the flight never goes back");
        let even = clock_of_levels(&levels, 0.0);
        assert!((even.share_at(10.0) - 0.5).abs() < 1e-9, "without strength: an even speed");
    }
}

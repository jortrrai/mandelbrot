//! Music that steers a flight (issue #185): with *Follow the music* a video of a flight lasts exactly as long as its audio track, moves faster through the
//! loud, energetic parts and slower through the calm ones, and has the track in its file.
//!
//! The track is decoded by ffmpeg (mono, `SAMPLE_RATE`); its loudness (RMS) every `STEP` seconds is smoothed over `SMOOTHING` seconds, so the speed follows
//! the swells and sections rather than every beat, and scaled between the quiet and the loud parts of the track (its 5th and 95th percentiles). The speed of
//! the flight at a moment is `1 - strength + strength · (MIN_SPEED + (MAX_SPEED - MIN_SPEED) · loudness)`, and the flight's clock is that speed added up,
//! scaled so the flight ends with the track. Rendering the same track again gives the same frames (also when a video is resumed).

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

/// The music of a video: the audio file, and how much it steers the speed (0: an even speed, 1: fully)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Music {
    pub(super) path: String,
    pub(super) strength: f64,
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

/// The loudness (RMS) of the track at `path` every `STEP` seconds (ffmpeg decodes it)
fn levels_of_track(path: &Path) -> Result<Vec<f64>, String> {
    let output = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-f", "s16le", "-"])
        .stdin(std::process::Stdio::null()).output().map_err(|err| format!("Could not start ffmpeg to read the music: {err}"))?;
    if !output.status.success() || output.stdout.len() < 4 {
        return Err(format!("ffmpeg could not read the music {}: {}", path.display(), String::from_utf8_lossy(&output.stderr).trim()));
    }
    let samples: Vec<f64> = output.stdout.as_chunks::<2>().0.iter().map(|pair| f64::from(i16::from_le_bytes(*pair)) / 32768.0).collect();
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
        (self.follow_music && !self.music_path.trim().is_empty()).then(|| Music { path: self.music_path.trim().to_string(), strength: self.music_strength })
    }

    /// The length of the followed track (read once per path), if it can be read
    pub(super) fn followed_music_length(&mut self) -> Option<Duration> {
        let path = self.followed_music()?.path;
        if self.music_length.as_ref().is_none_or(|(known, _)| *known != path) {
            self.music_length = Some((path.clone(), track_length(Path::new(&path))));
        }
        self.music_length.as_ref().and_then(|(_, length)| *length)
    }

    /// Takes the audio file `path` (dropped on the window) as the music of videos
    pub(super) fn use_music_file(&mut self, path: &Path) {
        self.music_path = path.display().to_string();
        self.follow_music = true;
        self.log(format!("Videos follow the music of {} now (see Follow the music)", path.display()));
    }

    /// The music field, *Follow the music* and its strength
    pub(super) fn music_ui(&mut self, ui: &mut eframe::egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.follow_music, "Follow the music:").on_hover_text("The video lasts as long as the track, moves faster through its loud, energetic parts and slower through its calm ones, and has the track as its sound");
            ui.add(eframe::egui::TextEdit::singleline(&mut self.music_path).desired_width(160.0).hint_text("audio file (or drop it here)"))
                .on_hover_text("The path of an audio file (MP3, WAV, FLAC, ...); an audio file dropped on the window lands here");
            ui.add_enabled(self.follow_music, eframe::egui::Slider::new(&mut self.music_strength, 0.0..=1.0).text("strength").custom_formatter(|value, _| format!("{:.0}%", value * 100.0)))
                .on_hover_text("How much the music changes the speed: 0% an even speed all the way, 100% from a quarter of the average speed in the calmest parts to 1.75 times it in the loudest");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

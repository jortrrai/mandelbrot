//! The music of a video (issue #185): with *Music* a video of a flight has an audio file as its sound and lasts exactly as long as it, at an even speed; or
//! as long as a length typed in *Fit to length* that is not the track's, with the track cut there or repeated until then (issue #274).
//!
//! The music that steered a flight came out of the standard app (issue #275): the speed that followed the loudness, *Fit to the music* (keyframes on the
//! swells of the track) and *Pulse on the beat*: the beat detection triggered on hi-hats and noise and missed some songs, and the pacing did not feel right.
//! Their code is kept on the branch `plugin/music`, off by default there.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The music of a video: the audio file (the manifests of older videos also have the strength of its speed and of its pulse, which are no longer used)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Music {
    pub(super) path: String,
    /// Whether the video lasts as long as the track (else a length typed in *Fit to length* that is not the track's: the video lasts that long, and the track
    /// is cut there or repeated until then, issue #274; absent in older manifests: it lasts as long as the track)
    #[serde(default = "follows_by_default")]
    pub(super) follows: bool,
}

const fn follows_by_default() -> bool {
    true
}

/// Puts the track `music` into the video `video` (which then has it as its sound; a track longer than the video is cut at its end)
pub(super) fn add_track(video: &Path, music: &Path, repeat: bool) -> Result<(), String> {
    let extension = video.extension().and_then(|extension| extension.to_str()).unwrap_or("mp4");
    let with_track = video.with_extension(format!("with-music.{extension}"));
    let audio: &[&str] = if extension == "mkv" { &["-c:a", "flac"] } else { &["-c:a", "aac", "-b:a", "320k"] };
    let mut arguments: Vec<String> = ["-v", "error", "-y", "-i"].map(String::from).to_vec();
    arguments.push(video.display().to_string());
    //(Repeated: as often as the video needs, `-shortest` ends it with the video)
    if repeat {
        arguments.extend(["-stream_loop", "-1"].map(String::from));
    }
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
pub(super) fn track_length(path: &Path) -> Option<Duration> {
    let output = std::process::Command::new("ffprobe").args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]).arg(path).stdin(std::process::Stdio::null()).output().ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse::<f64>().ok().filter(|seconds| seconds.is_finite() && *seconds > 0.0).map(Duration::from_secs_f64)
}

impl super::MandelbrotApp {
    /// The music of the next video: with *Music* and a path
    pub(super) fn followed_music(&self) -> Option<Music> {
        (self.follow_music && !self.music_path.trim().is_empty()).then(|| Music { path: self.music_path.trim().to_string(), follows: true })
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
        self.log(format!("Videos have the music of {} now (see Music)", path.display()));
    }

    /// The music field: *Music*, the audio file and *Open song...*
    pub(super) fn music_ui(&mut self, ui: &mut eframe::egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.follow_music, "Music:").on_hover_text("The video has the track as its sound and lasts as long as it, at an even speed (or as long as a length typed in Fit to length: the track is then cut there or repeated)");
            ui.add(eframe::egui::TextEdit::singleline(&mut self.music_path).desired_width(160.0).hint_text("audio file (or drop it here)"))
                .on_hover_text("The path of an audio file (MP3, WAV, FLAC, ...); an audio file dropped on the window lands here");
            //(The system's file chooser, modal: the window waits for it)
            if ui.button("Open song...").on_hover_text("Chooses an audio file as the music of videos: they get it as their sound and last exactly as long as it").clicked() {
                let mut dialog = rfd::FileDialog::new().set_title("Open a song for videos").add_filter("Audio", &AUDIO_EXTENSIONS);
                if let Some(home) = super::sharing::home_folder() {
                    dialog = dialog.set_directory(home);
                }
                if let Some(path) = dialog.pick_file() {
                    self.use_music_file(&path);
                }
            }
        });
    }
}

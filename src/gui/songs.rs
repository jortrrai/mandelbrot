//! The length of a video from a song (issue #178): type a song, pick the right recording from the list (versions of a song have different lengths), and its
//! length fills in *Fit to length*, so the flight's video lasts exactly as long as the song.
//!
//! The recordings and their lengths come from MusicBrainz (<https://musicbrainz.org>), which is free and needs no account: its rules ask for at most one
//! request a second and a User-Agent that names the application and a way to reach its makers. A search is one request, made on a background thread.
//! Only the length is used; the sound itself is not (see issue #185 for music that steers a flight).
//!
//! Spotify links are not supported: Spotify's Web API needs the credentials of a registered application, which this program does not have.
#![cfg_attr(target_os = "android", allow(dead_code))] //(The song search is not in the Android app yet, see `search`)

use std::sync::mpsc;
use std::time::Duration;

use eframe::egui;
use serde::Deserialize;

use super::MandelbrotApp;

/// The User-Agent that MusicBrainz asks for: the application, its version, and where to find its makers
const USER_AGENT: &str = concat!("MandelbrotByJort/", env!("CARGO_PKG_VERSION"), " ( https://github.com/jortrrai/mandelbrot )");

/// The most recordings a search lists
const RESULTS: usize = 10;

/// A recording found: its title, its artists, the release it is on, and its length
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Recording {
    pub(super) title: String,
    pub(super) artist: String,
    pub(super) release: String,
    pub(super) length: Duration,
}

#[derive(Deserialize)]
struct SearchAnswer {
    #[serde(default)]
    recordings: Vec<RawRecording>,
}

#[derive(Deserialize)]
struct RawRecording {
    title: String,
    length: Option<u64>,
    #[serde(rename = "artist-credit", default)]
    artist_credit: Vec<RawCredit>,
    #[serde(default)]
    releases: Vec<RawRelease>,
}

#[derive(Deserialize)]
struct RawCredit {
    name: String,
    #[serde(default)]
    joinphrase: String,
}

#[derive(Deserialize)]
struct RawRelease {
    title: String,
}

/// The recordings with a length in a MusicBrainz search answer (JSON)
pub(super) fn parse_answer(text: &str) -> Result<Vec<Recording>, String> {
    let answer: SearchAnswer = serde_json::from_str(text).map_err(|err| format!("MusicBrainz answered something unexpected: {err}"))?;
    Ok(answer.recordings.into_iter().filter_map(|recording| {
        let length = Duration::from_millis(recording.length.filter(|milliseconds| *milliseconds > 0)?);
        let artist = recording.artist_credit.iter().flat_map(|credit| [credit.name.as_str(), credit.joinphrase.as_str()]).collect::<String>();
        let release = recording.releases.first().map(|release| release.title.clone()).unwrap_or_default();
        Some(Recording { title: recording.title, artist, release, length })
    }).take(RESULTS).collect())
}

/// Searches MusicBrainz for recordings of `query` (blocking: on a background thread)
#[cfg(target_os = "android")]
fn search(_query: &str) -> Result<Vec<Recording>, String> {
    Err(String::from("The song search is not in the Android app yet (issue #219)"))
}

/// Searches MusicBrainz for recordings of `query` (blocking: on a background thread)
#[cfg(not(target_os = "android"))]
fn search(query: &str) -> Result<Vec<Recording>, String> {
    let answer = ureq::get("https://musicbrainz.org/ws/2/recording")
        .query("query", query)
        .query("fmt", "json")
        .query("limit", &(RESULTS * 2).to_string())
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(15))
        .call()
        .map_err(|err| format!("Could not search MusicBrainz: {err}"))?;
    parse_answer(&answer.into_string().map_err(|err| format!("Could not read the answer of MusicBrainz: {err}"))?)
}

/// A length as the boxes of *Fit to length* show it: hours, minutes and seconds (with the milliseconds of a recording)
fn length_boxes(length: Duration) -> [String; 3] {
    let seconds = length.as_secs_f64();
    let (hours, minutes) = ((seconds / 3600.0).floor(), (seconds % 3600.0 / 60.0).floor());
    let rest = ((seconds - hours * 3600.0 - minutes * 60.0) * 1000.0).round() / 1000.0;
    [format!("{hours}"), format!("{minutes}"), format!("{rest}")]
}

/// The length of a recording as the list shows it
fn length_text(length: Duration) -> String {
    let seconds = length.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

impl MandelbrotApp {
    /// Starts a search for the song in the song field
    fn search_song(&mut self) {
        let query = self.song_input.trim().to_string();
        if query.is_empty() || self.song_search.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(search(&query));
        });
        self.song_search = Some(receiver);
        self.log(format!("Searching MusicBrainz for {:?}...", self.song_input.trim()));
    }

    /// Takes the answer of a search under way
    pub(super) fn poll_song_search(&mut self, ctx: &egui::Context) {
        let Some(receiver) = &self.song_search else { return };
        ctx.request_repaint_after(Duration::from_millis(200));
        match receiver.try_recv() {
            Ok(Ok(recordings)) => {
                self.song_search = None;
                self.log(format!("MusicBrainz found {} recordings with a length", recordings.len()));
                self.song_results = recordings;
            }
            Ok(Err(reason)) => {
                self.song_search = None;
                self.log(reason);
            }
            Err(mpsc::TryRecvError::Disconnected) => self.song_search = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    /// Fits the video to the length of `recording`
    pub(super) fn use_recording(&mut self, recording: &Recording) {
        self.video_length = length_boxes(recording.length);
        self.video_fit_length = true;
        self.log(format!("The video fits {} by {} ({}): {}", recording.title, recording.artist, recording.release, length_text(recording.length)));
    }

    /// The song field with *Find song*, and the recordings found, each with *Use*
    pub(super) fn song_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let response = ui.add(egui::TextEdit::singleline(&mut self.song_input).desired_width(180.0).hint_text("song, and artist"))
                .on_hover_text("A song to fit the video to: its recordings and their lengths come from MusicBrainz (free, no account)");
            let entered = response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            let label = if self.song_search.is_some() { "Searching..." } else { "Find song" };
            if ui.add_enabled(self.song_search.is_none() && !self.song_input.trim().is_empty(), egui::Button::new(label)).clicked() || entered {
                self.search_song();
            }
            if !self.song_results.is_empty() && ui.small_button("Clear").clicked() {
                self.song_results.clear();
            }
        });
        let mut chosen = None;
        for recording in &self.song_results {
            ui.horizontal(|ui| {
                if ui.small_button("Use").on_hover_text("Fits the video to the length of this recording").clicked() {
                    chosen = Some(recording.clone());
                }
                let release = if recording.release.is_empty() { String::new() } else { format!(", {}", recording.release) };
                ui.label(format!("{} – {} ({}{release})", recording.title, recording.artist, length_text(recording.length)));
            });
        }
        if let Some(recording) = chosen {
            self.use_recording(&recording);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "a request to MusicBrainz, run by hand"]
    fn musicbrainz_finds_a_song_with_its_length() {
        let recordings = search("Conquest of Paradise Vangelis").unwrap();
        println!("{recordings:#?}");
        assert!(recordings.iter().any(|recording| recording.title.contains("Conquest of Paradise") && recording.length > Duration::from_secs(60)));
    }

    #[test]
    fn the_recordings_of_an_answer_are_listed_with_their_lengths() {
        let answer = r#"{"created":"2026-10-31T19:00:00.000Z","count":2,"offset":0,"recordings":[
            {"id":"a","score":100,"title":"Conquest of Paradise","length":287000,"artist-credit":[{"name":"Vangelis","joinphrase":""}],"releases":[{"id":"r","title":"1492: Conquest of Paradise"}]},
            {"id":"b","score":90,"title":"Conquest of Paradise (live)","artist-credit":[{"name":"Vangelis","joinphrase":" & "},{"name":"Orchestra"}]},
            {"id":"c","score":80,"title":"Light and Shadow","length":226782,"artist-credit":[{"name":"Someone","joinphrase":" feat. "},{"name":"Other"}],"releases":[]}]}"#;
        let recordings = parse_answer(answer).unwrap();
        assert_eq!(recordings.len(), 2, "a recording without a length is left out");
        assert_eq!(recordings[0], Recording { title: String::from("Conquest of Paradise"), artist: String::from("Vangelis"), release: String::from("1492: Conquest of Paradise"), length: Duration::from_secs(287) });
        assert_eq!((recordings[1].artist.as_str(), recordings[1].release.as_str()), ("Someone feat. Other", ""));
        assert!(parse_answer("<html>").is_err());
        assert_eq!(length_boxes(Duration::from_millis(226_782)), [String::from("0"), String::from("3"), String::from("46.782")]);
        assert_eq!(length_text(Duration::from_secs(287)), "4:47");
    }
}

//! Saved flights (issue #180): a flight set up in the window, saved as a JSON file in `flights/`, so it can be rendered again, in other qualities, or in a batch.
//!
//! A file holds the flight, not how it is rendered: where it starts and ends (the center, the scale, the iteration limit and the color cycle at both ends), the
//! route, the pace and its speeds, the zoom a dive starts at, and optionally the length of its video (*Fit to length*). The size, the quality, the frame rate and
//! the format of a video are the settings of the render, so one flight can be rendered many ways.
//!
//! ```json
//! {
//!   "format_version": 1,
//!   "name": "Seahorse dive",
//!   "start": { "center": { "real": -0.75, "imaginary": 0.0 }, "scale": 1.0, "max_iterations": 300, "iteration_divide": 1.0, "color_offset": 0.0 },
//!   "destination": { "center": { "real": -0.7498, "imaginary": 0.0203 }, "scale": 6.4e-5, "max_iterations": 5000, "iteration_divide": 4.0, "color_offset": 120.0 },
//!   "route": "Dive", "pace": "Constant", "constant_speed": 3.0, "speed": 1.0, "dive_zoom": 1.0,
//!   "length_seconds": 200.0
//! }
//! ```
//! Numbers are written as the shortest text that reads back to the same `f64`, so a flight is reproduced exactly. Unknown fields are ignored.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::flight::{Camera, Flight, Pace, Route, CONSTANT_SPEEDS, DIVE_ZOOMS, FLIGHT_SPEEDS};
use super::video::VideoFlight;
use crate::mandelbrot_set::ColorCycle;

/// The version of the format that this program writes
const FORMAT_VERSION: u32 = 1;

/// The center of a view
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Center {
    real: f64,
    imaginary: f64,
}

/// One end of a flight: the view, its iteration limit and its color cycle
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct End {
    center: Center,
    scale: f64,
    max_iterations: u32,
    iteration_divide: f64,
    color_offset: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    stretch_palette: bool,
}

impl End {
    fn of(camera: Camera, max_iterations: u32, palette: ColorCycle) -> End {
        End { center: Center { real: camera.x, imaginary: camera.y }, scale: camera.scale, max_iterations, iteration_divide: palette.divide, color_offset: palette.offset, stretch_palette: palette.stretch }
    }

    const fn camera(&self) -> Camera {
        Camera { x: self.center.real, y: self.center.imaginary, scale: self.scale }
    }

    const fn palette(&self) -> ColorCycle {
        ColorCycle { divide: self.iteration_divide, offset: self.color_offset, stretch: self.stretch_palette }
    }
}

/// The contents of a flight file, see the module
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct FlightFile {
    format_version: u32,
    name: String,
    start: End,
    destination: End,
    route: String,
    pace: String,
    constant_speed: f64,
    speed: f64,
    dive_zoom: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    length_seconds: Option<f64>,
    /// A flight through keyframes (see `sequence`); `start` and `destination` are then its first and last keyframes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sequence: Option<super::sequence::Sequence>,
}

/// A flight loaded from a file: the flight, the length its video should have (if the file gives one), and the name of the file
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SavedFlight {
    pub(super) flight: VideoFlight,
    pub(super) length: Option<std::time::Duration>,
    pub(super) file_name: String,
}

/// The text of the file of `flight`, with the length of its video if it has one
pub(super) fn to_json(flight: &VideoFlight, length: Option<std::time::Duration>) -> String {
    let file = FlightFile {
        format_version: FORMAT_VERSION,
        name: flight.name.clone(),
        start: End::of(flight.from, flight.limits.0, flight.palettes.0),
        destination: End::of(flight.to, flight.limits.1, flight.palettes.1),
        route: flight.route.name().to_string(),
        pace: flight.pace.name().to_string(),
        constant_speed: flight.constant_speed,
        speed: flight.speed,
        dive_zoom: flight.dive_zoom,
        length_seconds: length.map(|length| length.as_secs_f64()),
        sequence: flight.sequence.clone(),
    };
    serde_json::to_string_pretty(&file).expect("a flight is plain text and numbers, which always serialize") + "\n"
}

/// The flight of the text of a file named `file_name`, or why it cannot be used
pub(super) fn parse(text: &str, file_name: &str) -> Result<SavedFlight, String> {
    let file: FlightFile = serde_json::from_str(text).map_err(|err| format!("{file_name}: {err}"))?;
    if file.format_version > FORMAT_VERSION {
        return Err(format!("{file_name}: format version {} is newer than this program reads ({FORMAT_VERSION})", file.format_version));
    }
    let route = Route::ALL.into_iter().find(|route| route.name().eq_ignore_ascii_case(&file.route)).ok_or_else(|| format!("{file_name}: unknown route {:?}", file.route))?;
    let pace = Pace::ALL.into_iter().find(|pace| pace.name().eq_ignore_ascii_case(&file.pace) && *pace != Pace::NoFlight).ok_or_else(|| format!("{file_name}: unknown pace {:?}", file.pace))?;
    let finite = [file.start.center.real, file.start.center.imaginary, file.destination.center.real, file.destination.center.imaginary, file.constant_speed, file.speed, file.dive_zoom].iter().all(|value| value.is_finite());
    if !finite || file.start.scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || file.destination.scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(format!("{file_name}: the numbers of the flight are not usable (scales must be positive)"));
    }
    let speeds = f64::from(*FLIGHT_SPEEDS.start()) / 100.0..=f64::from(*FLIGHT_SPEEDS.end()) / 100.0;
    let flight = VideoFlight {
        from: file.start.camera(),
        to: file.destination.camera(),
        pace,
        route,
        constant_speed: file.constant_speed.clamp(*CONSTANT_SPEEDS.start(), *CONSTANT_SPEEDS.end()),
        speed: file.speed.clamp(*speeds.start(), *speeds.end()),
        dive_zoom: file.dive_zoom.clamp(*DIVE_ZOOMS.start(), *DIVE_ZOOMS.end()),
        name: file.name,
        limits: (file.start.max_iterations.max(1), file.destination.max_iterations.max(1)),
        palettes: (file.start.palette(), file.destination.palette()),
        sequence: file.sequence.filter(super::sequence::Sequence::is_complete),
    };
    let length = file.length_seconds.filter(|seconds| seconds.is_finite() && *seconds > 0.0).map(std::time::Duration::from_secs_f64);
    Ok(SavedFlight { flight, length, file_name: file_name.to_string() })
}

/// The flights of the `.json` files of `folder`, sorted by file name, and a message for every file that cannot be used (a folder that does not exist has none)
pub(super) fn load_folder(folder: &Path) -> (Vec<SavedFlight>, Vec<String>) {
    let Ok(entries) = std::fs::read_dir(folder) else { return (Vec::new(), Vec::new()) };
    let mut files: Vec<std::path::PathBuf> = entries.filter_map(|entry| Some(entry.ok()?.path())).filter(|path| path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("json"))).collect();
    files.sort();
    let (mut flights, mut problems) = (Vec::new(), Vec::new());
    for path in files {
        let file_name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        match std::fs::read_to_string(&path).map_err(|err| format!("{file_name}: {err}")).and_then(|text| parse(&text, &file_name)) {
            Ok(flight) => flights.push(flight),
            Err(problem) => problems.push(problem),
        }
    }
    (flights, problems)
}

/// Saves `flight` as a new file in `folder` (created if needed), named like a new place file; returns the name of the file
pub(super) fn save_new(folder: &Path, flight: &VideoFlight, length: Option<std::time::Duration>) -> std::io::Result<String> {
    std::fs::create_dir_all(folder)?;
    let existing: Vec<String> = std::fs::read_dir(folder)?.filter_map(|entry| entry.ok()?.file_name().into_string().ok()).collect();
    let file_name = crate::points_of_interest::new_file_name(&existing, &flight.name);
    //create_new never overwrites a file that appeared in the meantime
    std::io::Write::write_all(&mut std::fs::OpenOptions::new().write(true).create_new(true).open(folder.join(&file_name))?, to_json(flight, length).as_bytes())?;
    Ok(file_name)
}

impl super::MandelbrotApp {
    /// Reads the flights of the flights folder again, and logs the files that cannot be used
    pub(super) fn reload_saved_flights(&mut self) {
        let (flights, problems) = load_folder(&self.flights_folder);
        self.saved_flights = flights;
        self.selected_flight = self.selected_flight.filter(|index| *index < self.saved_flights.len());
        for problem in problems {
            self.log(format!("Skipped a flight file: {problem}"));
        }
    }

    /// Saves the last flight under the name in the name field (with the length of *Fit to length*, when it is on) as a new file in the flights folder
    pub(super) fn save_last_flight(&mut self) {
        let Some(mut flight) = self.last_flight.clone() else { return };
        let name = self.flight_name_input.trim();
        if !name.is_empty() {
            flight.name = name.to_string();
        }
        let length = self.target_length().filter(|_| self.video_fit_length);
        match save_new(&self.flights_folder, &flight, length) {
            Ok(file_name) => {
                self.log(format!("Saved the flight {} as {}", flight.name, self.flights_folder.join(&file_name).display()));
                self.reload_saved_flights();
                self.selected_flight = self.saved_flights.iter().position(|saved| saved.file_name == file_name);
            }
            Err(err) => self.log(format!("Could not save the flight: {err}")),
        }
    }

    /// Makes saved flight `index` the last flight, ready for *Render video*: the flight controls take its route, pace, speeds and dive zoom, and *Fit to length*
    /// its length when it has one
    pub(super) fn use_saved_flight(&mut self, index: usize) {
        let Some(saved) = self.saved_flights.get(index).cloned() else { return };
        let flight = &saved.flight;
        (self.route, self.pace, self.constant_speed, self.dive_zoom) = (flight.route, flight.pace, flight.constant_speed, flight.dive_zoom);
        self.flight_speed = ((flight.speed * 100.0).round() as u16).clamp(*FLIGHT_SPEEDS.start(), *FLIGHT_SPEEDS.end());
        if let Some(length) = saved.length {
            let seconds = length.as_secs_f64();
            let (hours, minutes) = ((seconds / 3600.0).floor(), (seconds % 3600.0 / 60.0).floor());
            let rest = ((seconds - hours * 3600.0 - minutes * 60.0) * 1000.0).round() / 1000.0;
            self.video_length = [format!("{hours}"), format!("{minutes}"), format!("{rest}")];
            self.video_fit_length = true;
        }
        self.flight_name_input.clone_from(&flight.name);
        self.last_flight = Some(saved.flight.clone());
        self.selected_flight = Some(index);
        self.log(format!("The flight {} ({}) is the last flight now: Render video renders it", flight.name, saved.file_name));
    }

    /// Flies saved flight `index` in the window: from its start, with its colors and limits, to its destination (see `use_saved_flight`)
    pub(super) fn fly_saved_flight(&mut self, index: usize) {
        self.use_saved_flight(index);
        if let Some(flight) = self.last_flight.clone() {
            self.fly_video_flight(&flight);
        }
    }

    /// Flies `flight` in the window: from its start, with its colors and limits, along its path (through its keyframes, if it has them) to its destination,
    /// and makes it the last flight
    pub(super) fn fly_video_flight(&mut self, flight: &VideoFlight) {
        self.c.set_view(&crate::complex_plane::View::new(flight.from.x, flight.from.y, flight.from.scale));
        (self.m.palette, self.m.max_iterations, self.max_iterations_input) = (flight.palettes.0, flight.limits.0, flight.limits.0);
        let point = crate::points_of_interest::PointOfInterest {
            name: flight.name.clone(),
            description: String::new(),
            location: crate::points_of_interest::Location { real: flight.to.x, imaginary: flight.to.y, scale: flight.to.scale, max_iterations: flight.limits.1, exact: None },
            appearance: None,
            file_name: String::new(),
        };
        let path = if flight.sequence.is_some() { flight.path() } else { Flight::with(flight.from, flight.to, self.pace, self.route, self.constant_speed, self.speed_factor(), self.dive_zoom) };
        self.fly_path(point, flight.palettes.1, path);
        self.last_flight = Some(flight.clone()); //Exactly the flight (the speed control has whole percents)
    }

    /// The controls of the saved flights: the name and *Save flight* for the last flight, and the list of saved flights with *Fly*
    pub(super) fn saved_flights_ui(&mut self, ui: &mut eframe::egui::Ui) {
        if !self.saved_flights_loaded {
            self.saved_flights_loaded = true;
            self.reload_saved_flights();
        }
        ui.horizontal(|ui| {
            ui.add(eframe::egui::TextEdit::singleline(&mut self.flight_name_input).desired_width(140.0).hint_text("name of the flight"));
            if ui.add_enabled(self.last_flight.is_some(), eframe::egui::Button::new("Save flight")).on_hover_text(format!("Saves the last flight (where it starts and ends, its colors, route, pace and speeds, and the length of Fit to length when it is on) as a file in {}, to render it again later", self.flights_folder.display())).clicked() {
                self.save_last_flight();
            }
        });
        ui.horizontal(|ui| {
            let selected = self.selected_flight.and_then(|index| self.saved_flights.get(index)).map_or_else(|| String::from("Saved flights"), |saved| saved.flight.name.clone());
            let mut chosen = None;
            eframe::egui::ComboBox::from_id_salt("saved_flights").selected_text(selected).width(160.0).show_ui(ui, |ui| {
                for (index, saved) in self.saved_flights.iter().enumerate() {
                    if ui.selectable_label(self.selected_flight == Some(index), &saved.flight.name).on_hover_text(&saved.file_name).clicked() {
                        chosen = Some(index);
                    }
                }
            }).response.on_hover_text("Choose a saved flight: it becomes the last flight, so Render video renders it, and the flight controls take its settings");
            if let Some(index) = chosen {
                self.use_saved_flight(index);
            }
            if ui.add_enabled(self.selected_flight.is_some(), eframe::egui::Button::new("Fly")).on_hover_text("Flies the chosen flight in the window, from its start").clicked() {
                if let Some(index) = self.selected_flight {
                    self.fly_saved_flight(index);
                }
            }
            if ui.button("Reload").on_hover_text(format!("Reads the flight files of {} again", self.flights_folder.display())).clicked() {
                self.reload_saved_flights();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dive() -> VideoFlight {
        VideoFlight {
            from: Camera { x: -0.75, y: 0.0, scale: 1.0 },
            to: Camera { x: -0.749_810_000_000_000_1, y: -0.020_300_000_000_000_054, scale: 6.400_000_000_000_002e-5 },
            pace: Pace::Constant,
            route: Route::Dive,
            constant_speed: 2.5,
            speed: 1.5,
            dive_zoom: 0.3,
            name: String::from("Seahorse dive"),
            limits: (300, 5000),
            palettes: (ColorCycle { divide: 1.0, offset: 0.0, stretch: false }, ColorCycle { divide: 0.113, offset: 278.0, stretch: false }),
            sequence: None,
        }
    }

    #[test]
    fn a_flight_reads_back_exactly_with_its_length() {
        let length = Some(std::time::Duration::from_secs(200));
        let saved = parse(&to_json(&dive(), length), "01-seahorse-dive.json").unwrap();
        assert_eq!(saved.flight, dive(), "every number exactly");
        assert_eq!((saved.length, saved.file_name.as_str()), (length, "01-seahorse-dive.json"));
        assert_eq!(parse(&to_json(&dive(), None), "x.json").unwrap().length, None);
    }

    #[test]
    fn a_file_that_cannot_be_used_says_why() {
        let text = to_json(&dive(), None);
        assert!(parse("{", "broken.json").unwrap_err().starts_with("broken.json"));
        assert!(parse(&text.replace("\"Dive\"", "\"Spiral\""), "a.json").unwrap_err().contains("unknown route"));
        assert!(parse(&text.replace("\"Constant\"", "\"No flight\""), "b.json").unwrap_err().contains("unknown pace"));
        assert!(parse(&text.replace("\"scale\": 1.0", "\"scale\": 0.0"), "c.json").unwrap_err().contains("positive"));
        assert!(parse(&text.replace("\"format_version\": 1", "\"format_version\": 9"), "d.json").unwrap_err().contains("newer"));
    }

    #[test]
    fn saved_flights_are_listed_from_their_folder_and_never_overwritten() {
        let folder = std::env::temp_dir().join(format!("mandelbrot-flights-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        assert_eq!(load_folder(&folder), (Vec::new(), Vec::new()), "no folder yet: no flights");
        let first = save_new(&folder, &dive(), None).unwrap();
        let second = save_new(&folder, &dive(), Some(std::time::Duration::from_secs(90))).unwrap();
        assert_ne!(first, second);
        std::fs::write(folder.join("zz-broken.json"), "{").unwrap();
        let (flights, problems) = load_folder(&folder);
        assert_eq!(flights.iter().map(|saved| saved.file_name.clone()).collect::<Vec<_>>(), [first, second]);
        assert_eq!(flights[1].length, Some(std::time::Duration::from_secs(90)));
        assert_eq!(problems.len(), 1, "{problems:?}");
        let _ = std::fs::remove_dir_all(&folder);
    }
}

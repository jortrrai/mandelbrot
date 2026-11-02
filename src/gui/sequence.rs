//! Keyframe sequences (issue #194): one flight through views A → B → C → …, chosen by hand at the depths where they look best, each with its own colors.
//!
//! Between two keyframes the camera moves along the path of a flight (van Wijk and Nuij: zooming out only as far as the distance needs; a pure zoom when the
//! centers are the same, so a dive whose keyframes share their center never moves sideways), at a constant speed along that path, in the time given for that
//! transition. The colors go from one keyframe's to the next: the iteration divide on a logarithmic scale (it runs from 0.1 to 1000), the offset the short way
//! around the palette, and the iteration limit on a logarithmic scale too. A sequence is a flight (see `flight::Flight::of_sequence`): it flies in the
//! window, renders as a video, queues and saves like any flight.
//!
//! The keyframes are made in the Flights section: *Add this view* takes the current view with its colors (with the depth control of `depth`, the keyframes of
//! a dive keep exactly the same center).

use eframe::egui;
use serde::{Deserialize, Serialize};

use super::flight::Camera;
use super::MandelbrotApp;
use crate::mandelbrot_set::{ColorCycle, PALETTE_LENGTH};

/// The time of a new transition, in seconds
const TRANSITION_SECONDS: f64 = 5.0;

/// One view of a sequence: where, the iteration limit, and the colors
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Keyframe {
    pub(super) real: f64,
    pub(super) imaginary: f64,
    pub(super) scale: f64,
    pub(super) max_iterations: u32,
    pub(super) divide: f64,
    pub(super) offset: f64,
    #[serde(default)]
    pub(super) stretch: bool,
    /// The rotation of the view, in radians (issue #217)
    #[serde(default)]
    pub(super) rotation: f64,
}

impl Keyframe {
    pub(super) const fn camera(&self) -> Camera {
        Camera { x: self.real, y: self.imaginary, scale: self.scale }
    }

    pub(super) const fn palette(&self) -> ColorCycle {
        ColorCycle { divide: self.divide, offset: self.offset, stretch: self.stretch }
    }
}

/// The keyframes of a sequence and the time of every transition between two of them, in seconds (one fewer than keyframes)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct Sequence {
    pub(super) keyframes: Vec<Keyframe>,
    pub(super) seconds: Vec<f64>,
}

impl Sequence {
    /// Whether it can be flown: at least two keyframes, and a positive time for every transition
    pub(super) fn is_complete(&self) -> bool {
        self.keyframes.len() >= 2 && self.seconds.len() == self.keyframes.len() - 1 && self.seconds.iter().all(|seconds| seconds.is_finite() && *seconds > 0.0)
    }
}

/// The colors at `share` (0 to 1) of the way from `from` to `to`, see the module
pub(super) fn blend_palettes(from: ColorCycle, to: ColorCycle, share: f64) -> ColorCycle {
    let share = share.clamp(0.0, 1.0);
    let divide = if from.divide > 0.0 && to.divide > 0.0 { from.divide * (to.divide / from.divide).powf(share) } else { from.divide + (to.divide - from.divide) * share };
    let turn = (to.offset - from.offset + PALETTE_LENGTH / 2.0).rem_euclid(PALETTE_LENGTH) - PALETTE_LENGTH / 2.0; //The short way around
    ColorCycle { divide, offset: (from.offset + turn * share).rem_euclid(PALETTE_LENGTH), stretch: if share < 0.5 { from.stretch } else { to.stretch } }
}

/// The iteration limit at `share` of the way from `from` to `to`, on a logarithmic scale
pub(super) fn blend_limits(from: u32, to: u32, share: f64) -> u32 {
    let (from, to) = (f64::from(from.max(1)), f64::from(to.max(1)));
    (from * (to / from).powf(share.clamp(0.0, 1.0))).round() as u32
}

impl MandelbrotApp {
    /// The current view as a keyframe: its exact center (see `depth`), scale, iteration limit and colors
    fn current_keyframe(&self) -> Keyframe {
        let location = self.current_point(String::new(), String::new()).location;
        Keyframe { real: location.real, imaginary: location.imaginary, scale: location.scale, max_iterations: location.max_iterations, divide: self.m.palette.divide, offset: self.m.palette.offset, stretch: self.m.palette.stretch, rotation: self.c.rotation() }
    }

    /// The sequence of the keyframes and the typed times (a time that is not a positive number makes it incomplete)
    pub(super) fn sequence(&self) -> Sequence {
        let seconds = self.sequence_seconds.iter().map(|text| text.trim().replace(',', ".").parse::<f64>().unwrap_or(f64::NAN)).collect();
        Sequence { keyframes: self.sequence_keyframes.clone(), seconds }
    }

    /// Adds the current view as the last keyframe
    pub(super) fn add_keyframe(&mut self) {
        if !self.sequence_keyframes.is_empty() {
            self.sequence_seconds.push(format!("{TRANSITION_SECONDS}"));
        }
        self.sequence_keyframes.push(self.current_keyframe());
    }

    /// Shows keyframe `index`: its view, limit and colors
    pub(super) fn go_to_keyframe(&mut self, index: usize) {
        let Some(keyframe) = self.sequence_keyframes.get(index).cloned() else { return };
        self.c.set_view(&crate::complex_plane::View::new(keyframe.real, keyframe.imaginary, keyframe.scale));
        self.c.set_rotation(keyframe.rotation);
        self.depth_center = Some(crate::complex::Complex::new(keyframe.real, keyframe.imaginary));
        (self.m.max_iterations, self.max_iterations_input, self.m.palette) = (keyframe.max_iterations, keyframe.max_iterations, keyframe.palette());
        self.render(&format!("Went to keyframe {}", index + 1));
    }

    /// Makes the sequence the last flight (for *Render video*, the queue and *Save flight*) and flies it in the window from its first keyframe
    pub(super) fn fly_sequence(&mut self) {
        let sequence = self.sequence();
        if !sequence.is_complete() {
            self.log("A sequence needs two keyframes or more, and a time of more than 0 seconds for every transition");
            return;
        }
        let (first, last) = (sequence.keyframes[0].clone(), sequence.keyframes[sequence.keyframes.len() - 1].clone());
        let flight = super::video::VideoFlight {
            from: first.camera(),
            to: last.camera(),
            pace: super::flight::Pace::Constant,
            route: super::flight::Route::Shortest,
            constant_speed: self.constant_speed,
            speed: self.speed_factor(),
            dive_zoom: self.dive_zoom,
            name: if self.flight_name_input.trim().is_empty() { String::from("Sequence") } else { self.flight_name_input.trim().to_string() },
            limits: (first.max_iterations, last.max_iterations),
            palettes: (first.palette(), last.palette()),
            rotations: (first.rotation, last.rotation),
            sequence: Some(sequence),
        };
        self.fly_video_flight(&flight);
    }

    /// Times the transitions so the keyframes land on the swells of the music of *Follow the music* (issue #185): the first keyframe at its start, the last at its
    /// end, the ones between on its most prominent swells (see `music::swells_of_levels`; the longest gaps are split evenly when it has fewer). Videos then
    /// follow the track at an even speed per transition (strength 0), so the keyframes stay on their swells; the times can be changed afterwards
    pub(super) fn fit_sequence_to_music(&mut self) {
        let path = self.music_path.trim().to_string();
        if self.sequence_keyframes.len() < 2 || path.is_empty() {
            self.log("Fitting a sequence to music needs two keyframes or more, and an audio file in Follow the music");
            return;
        }
        let between = self.sequence_keyframes.len() - 2;
        match super::music::swells_of_track(std::path::Path::new(&path), between) {
            Ok((swells, length)) => {
                let mut times: Vec<f64> = std::iter::once(0.0).chain(swells).chain(std::iter::once(length)).collect();
                while times.len() < between + 2 {
                    //Fewer swells than keyframes between: the longest gap is split
                    let longest = (0..times.len() - 1).max_by(|a, b| (times[a + 1] - times[*a]).total_cmp(&(times[b + 1] - times[*b]))).unwrap_or(0);
                    times.insert(longest + 1, f64::midpoint(times[longest], times[longest + 1]));
                }
                self.sequence_seconds = times.windows(2).map(|pair| format!("{:.2}", (pair[1] - pair[0]).max(0.01))).collect();
                (self.follow_music, self.music_strength) = (true, 0.0);
                let at: Vec<String> = times.iter().skip(1).take(between).map(|seconds| format!("{}:{:04.1}", (*seconds / 60.0).floor(), seconds % 60.0)).collect();
                self.log(format!("The keyframes land on the swells of the music at {} (the last at its end, {length:.1} s); videos follow the track at an even speed per transition", if at.is_empty() { String::from("its start and end") } else { at.join(", ") }));
            }
            Err(reason) => self.log(reason),
        }
    }

    /// The keyframes with their times, and the buttons to add, show, move and remove them and to fly the sequence
    pub(super) fn sequence_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("Keyframes: {}", self.sequence_keyframes.len()));
            if ui.button("Add this view").on_hover_text("Adds the current view, with its iteration limit and colors, as the last keyframe of the sequence (the depth control keeps the center exactly, for a dive)").clicked() {
                self.add_keyframe();
            }
            if ui.add_enabled(self.sequence_keyframes.len() >= 2, egui::Button::new("Fly the sequence")).on_hover_text("Flies through the keyframes in order, the colors changing from one keyframe's to the next, and makes it the last flight: Render video, Add to queue and Save flight take it").clicked() {
                self.fly_sequence();
            }
            if ui.add_enabled(self.sequence_keyframes.len() >= 2 && !self.music_path.trim().is_empty(), egui::Button::new("Fit to the music")).on_hover_text("Times the transitions so the keyframes land on the biggest swells of the track of Follow the music, the last one at its end (the times can be changed afterwards)").clicked() {
                self.fit_sequence_to_music();
            }
            if ui.add_enabled(!self.sequence_keyframes.is_empty(), egui::Button::new("Clear")).clicked() {
                (self.sequence_keyframes, self.sequence_seconds) = (Vec::new(), Vec::new());
            }
        });
        let mut action = None;
        for index in 0..self.sequence_keyframes.len() {
            if index > 0 {
                ui.horizontal(|ui| {
                    ui.add_space(24.0);
                    ui.label("then");
                    ui.add(egui::TextEdit::singleline(&mut self.sequence_seconds[index - 1]).desired_width(40.0)).on_hover_text("The time of the transition to the next keyframe, in seconds");
                    ui.label("s to");
                });
            }
            let keyframe = &self.sequence_keyframes[index];
            ui.horizontal(|ui| {
                if ui.small_button("Show").on_hover_text("Shows this keyframe: its view, limit and colors").clicked() {
                    action = Some((index, 0));
                }
                if ui.add_enabled(index > 0, egui::Button::new("Up").small()).clicked() {
                    action = Some((index, -1));
                }
                if ui.add_enabled(index + 1 < self.sequence_keyframes.len(), egui::Button::new("Down").small()).clicked() {
                    action = Some((index, 1));
                }
                if ui.small_button("Remove").clicked() {
                    action = Some((index, 2));
                }
                ui.label(format!("{}. zoom {}, divide {:.3}, offset {:.0}, {} iterations", index + 1, super::status::format_zoom(1.0 / keyframe.scale), keyframe.divide, keyframe.offset, keyframe.max_iterations));
            });
        }
        match action {
            Some((index, 0)) => self.go_to_keyframe(index),
            Some((index, 2)) => {
                self.sequence_keyframes.remove(index);
                if !self.sequence_seconds.is_empty() {
                    self.sequence_seconds.remove(index.min(self.sequence_seconds.len() - 1));
                }
            }
            Some((index, step)) => self.sequence_keyframes.swap(index, index.saturating_add_signed(step)),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_change_evenly_from_one_keyframe_to_the_next() {
        let (from, to) = (ColorCycle { divide: 0.1, offset: 490.0, stretch: false }, ColorCycle { divide: 10.0, offset: 10.0, stretch: true });
        let half = blend_palettes(from, to, 0.5);
        assert!((half.divide - 1.0).abs() < 1e-12, "the divide halfway on a logarithmic scale: {}", half.divide);
        assert!((half.offset - 0.0).abs() < 1e-9 || (half.offset - PALETTE_LENGTH).abs() < 1e-9, "the offset the short way around the palette: {}", half.offset);
        assert!(half.stretch);
        assert_eq!((blend_palettes(from, to, 0.0), blend_palettes(from, to, 1.0).divide), (from, 10.0));
        assert_eq!((blend_limits(100, 10_000, 0.5), blend_limits(100, 10_000, 0.0), blend_limits(100, 10_000, 1.0)), (1000, 100, 10_000));
    }
}

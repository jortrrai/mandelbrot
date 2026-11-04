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
    /// The Julia set of the keyframe, by its parameter c (issue #229); none: the Mandelbrot set. All keyframes of a sequence are in the same set
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) julia: Option<(f64, f64)>,
    /// The exact center of a keyframe in a deep view, as decimal text (see `deep_view`; then `real` and `imaginary` are the nearest 64-bit numbers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) exact: Option<(String, String)>,
    /// The scale of a keyframe deeper than 64-bit floats reach (issue #263; then `scale` is the smallest normal 64-bit float)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) depth: Option<crate::rendering::deep::Depth>,
}

impl Keyframe {
    /// The scale, also beyond 64-bit floats
    pub(super) fn depth(&self) -> crate::rendering::deep::Depth {
        self.depth.unwrap_or_else(|| crate::rendering::deep::Depth::from_f64(self.scale))
    }

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
    /// The exact center of a sequence that dives along one (issue #263): its keyframes in deep views all at this center, and the others (above 64-bit
    /// floats' depth) at its nearest 64-bit point, as the depth control keeps them (see `depth`). `None` for any other sequence, also one without deep
    /// keyframes
    pub(super) fn shared_exact(&self) -> Option<(String, String)> {
        let first = self.keyframes.iter().find_map(|keyframe| keyframe.exact.clone())?;
        let deepest = self.keyframes.iter().map(Keyframe::depth).min_by(|a, b| a.log2().total_cmp(&b.log2()))?;
        let bits = crate::rendering::deep::bits_for_depth(deepest, 1) + 64;
        let parse = |exact: &(String, String)| crate::rendering::deep::Fixed::parse(&exact.0, bits).zip(crate::rendering::deep::Fixed::parse(&exact.1, bits));
        let center = parse(&first)?;
        let nearest = (center.0.to_f64(), center.1.to_f64());
        self.keyframes.iter().all(|keyframe| keyframe.exact.as_ref().map_or_else(|| (keyframe.real.to_bits(), keyframe.imaginary.to_bits()) == (nearest.0.to_bits(), nearest.1.to_bits()), |exact| parse(exact).is_some_and(|other| other == center))).then_some(first)
    }

    /// The scale of its deepest keyframe
    pub(super) fn deepest(&self) -> Option<crate::rendering::deep::Depth> {
        self.keyframes.iter().map(Keyframe::depth).min_by(|a, b| a.log2().total_cmp(&b.log2()))
    }

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
        Keyframe { real: location.real, imaginary: location.imaginary, scale: location.scale, max_iterations: location.max_iterations, divide: self.m.palette.divide, offset: self.m.palette.offset, stretch: self.m.palette.stretch, rotation: self.c.rotation(), julia: location.julia, exact: location.exact, depth: location.deep_scale }
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
        if keyframe.julia.is_some() && self.m.julia.is_none() {
            self.mandelbrot_view = Some(self.c.get_view());
        }
        self.m.julia = keyframe.julia;
        self.c.set_view(&crate::complex_plane::View::new(keyframe.real, keyframe.imaginary, keyframe.scale));
        self.c.set_rotation(keyframe.rotation);
        self.depth_center = Some(crate::complex::Complex::new(keyframe.real, keyframe.imaginary));
        //(A keyframe in a deep view: there again, exactly)
        if let Some(exact) = &keyframe.exact {
            self.go_to_exact(exact, keyframe.depth());
        }
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
        if sequence.keyframes.iter().any(|keyframe| keyframe.julia != first.julia) {
            self.log("A sequence stays in one set: its keyframes are partly in a Julia set and partly elsewhere (or in different Julia sets); make them all in the same set");
            return;
        }
        let (sequence_exact, sequence_deepest) = (sequence.shared_exact(), sequence.deepest());
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
            julia: first.julia,
            sequence: Some(sequence),
            //(A dive along one exact center, also deeper than 64-bit floats: its deep frames there, see `Sequence::shared_exact`)
            to_exact: sequence_exact.clone(),
            to_depth: sequence_exact.as_ref().and(sequence_deepest).filter(|depth| !depth.to_f64().is_normal()),
        };
        self.fly_video_flight(&flight);
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

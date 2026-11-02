//! The center-locked depth control (issue #193): a zoom slider that changes only the scale of the view, never its center, so a dive can be explored up and
//! down along its exact vertical path, and its colors tuned at chosen depths (and those views saved as keyframes of a sequence, see `sequence`).
//!
//! *Lock center* makes the mouse keep the center too: the wheel zooms around the center instead of the cursor, and dragging does not move the view.
//! While the slider is dragged, frames render like those of a gesture (one sample per pixel); the full quality follows when it stops.

use eframe::egui;

use super::MandelbrotApp;
use crate::complex_plane::{Precision, View};

/// The range of the depth slider, in zoom (1 / scale): from ten times further out than the reset view to deeper than the 64-bit floats reach (the view stops
/// where their precision ends, see `Precision`)
pub(super) const DEPTH_ZOOMS: std::ops::RangeInclusive<f64> = 0.1..=1e15;

/// The radius of the center marker at first, and the radii its slider offers, in points (issue #223)
pub(super) const CENTER_MARKER_RADIUS: f32 = 24.0;
pub(super) const CENTER_MARKER_RADII: std::ops::RangeInclusive<f32> = 3.0..=400.0;

/// Draws the center marker (issue #223): a circle of `radius` points around `center`, the center of the view, with a small cross on the exact center. White with
/// a dark edge, so it shows on every color. It is only drawn on the screen, never in saved images or videos
pub(super) fn draw_center_marker(painter: &egui::Painter, center: egui::Pos2, radius: f32) {
    let arm = radius.min(6.0) * 0.5;
    //The dark edge first, the white line on it
    for stroke in [egui::Stroke::new(3.0, egui::Color32::from_black_alpha(170)), egui::Stroke::new(1.2, egui::Color32::from_white_alpha(230))] {
        painter.circle_stroke(center, radius, stroke);
        painter.line_segment([center - egui::vec2(arm, 0.0), center + egui::vec2(arm, 0.0)], stroke);
        painter.line_segment([center - egui::vec2(0.0, arm), center + egui::vec2(0.0, arm)], stroke);
    }
}

impl MandelbrotApp {
    /// The center of the view exactly as the depth control keeps it: the plane stores its edges, so its center read back can differ in the last bits, and
    /// many depth changes would add those up. While the view is where the depth control left it, this is the center it was given
    pub(super) fn exact_center(&self) -> crate::complex::Complex {
        let center = self.c.center();
        match &self.depth_center {
            Some(kept) if (kept.x - center.x).abs() <= self.c.get_scale() * 1e-6 && (kept.y - center.y).abs() <= self.c.get_scale() * 1e-6 => crate::complex::Complex::new(kept.x, kept.y),
            _ => center,
        }
    }

    /// Sets the zoom of the view to `zoom`, keeping its center exactly; a zoom deeper than the precision or the zoom limit allows is refused (false)
    pub(super) fn set_depth(&mut self, zoom: f64) -> bool {
        let zoom = zoom.clamp(*DEPTH_ZOOMS.start(), *DEPTH_ZOOMS.end());
        let center = self.exact_center();
        let mut deeper = self.c.clone();
        deeper.set_view(&View::new(center.x, center.y, 1.0 / zoom));
        let zooming_in = deeper.get_scale() < self.c.get_scale();
        if zooming_in && (deeper.precision() == Precision::Exhausted || self.zoom_refused(&deeper)) {
            return false;
        }
        self.c = deeper;
        self.depth_center = Some(crate::complex::Complex::new(center.x, center.y));
        true
    }

    /// The rotation of the view (issue #217): a slider with a number to type, in degrees or radians, and a button back to 0. It turns the image around the center
    /// of the view, which stays exactly where it is; switching the unit only changes how the angle is shown. The slider stops at its ends (-180° and 180°, the
    /// same view) instead of wrapping around, and moves in fine steps rather than whole degrees (issue #230)
    pub(super) fn rotation_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let radians = self.shown_rotation();
            let (mut value, range, suffix) = if self.rotation_in_degrees { (radians.to_degrees(), -180.0..=180.0, "°") } else { (radians, -std::f64::consts::PI..=std::f64::consts::PI, " rad") };
            //(Without smart aim the slider takes the exact angle under the mouse, not the roundest one within a pixel, which was a whole degree. Its own clamping
            //would round the shown angle every frame and turn the view, so a typed angle beyond an end is stopped there here, only when the user changes it)
            let ends = (*range.start(), *range.end());
            let response = ui.add(egui::Slider::new(&mut value, range).clamping(egui::SliderClamping::Never).smart_aim(false).max_decimals(if self.rotation_in_degrees { 2 } else { 4 }).suffix(suffix).text("rotation"))
                .on_hover_text("Turns the image counterclockwise around the center of the view, which stays where it is (0 is unturned; -180° and 180° are the same view). Places, images, Copy view, keyframes and flights keep the rotation");
            if response.changed() {
                let value = value.clamp(ends.0, ends.1);
                self.turn_to(if self.rotation_in_degrees { value.to_radians() } else { value });
            }
            if ui.add_enabled(self.c.rotation() != 0.0, egui::Button::new("⟲ 0")).on_hover_text("Turns the view back to unturned (0°)").clicked() {
                self.turn_to(0.0);
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.center_marker, "Center marker").on_hover_text("Draws a circle around the exact center of the view, to put a feature right in the middle before turning or diving around it. Only on the screen: saved images and videos do not show it");
            if self.center_marker {
                ui.add(egui::Slider::new(&mut self.center_marker_radius, CENTER_MARKER_RADII).logarithmic(true).integer().suffix(" px").text("size")).on_hover_text("The radius of the circle on the screen");
            }
            ui.selectable_value(&mut self.rotation_in_degrees, true, "degrees").on_hover_text("Show and type the rotation in degrees (the angle stays the same)");
            ui.selectable_value(&mut self.rotation_in_degrees, false, "radians").on_hover_text("Show and type the rotation in radians (the angle stays the same)");
        });
    }

    /// The angle the rotation slider shows: the one it was last set to while the view still has that angle (180° stays 180°), otherwise the view's own
    fn shown_rotation(&self) -> f64 {
        if crate::complex_plane::normalized_angle(self.rotation_slider).to_bits() == self.c.rotation().to_bits() { self.rotation_slider } else { self.c.rotation() }
    }

    /// Turns the view to `radians` (from the rotation controls) and renders it
    pub(super) fn turn_to(&mut self, radians: f64) {
        self.rotation_slider = radians;
        self.c.set_rotation(radians);
        self.render(&format!("Turned the view to {:.2}°", self.c.rotation().to_degrees()));
    }

    /// *Show orbit* and *Julia map* (issue #202, both off at first)
    pub(super) fn julia_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_orbit, "Show orbit").on_hover_text("Draws the orbit of the point under the mouse over the image: 0, c, c² + c, …, the path whose escape (or not) colors that point");
            ui.checkbox(&mut self.show_julia, "Julia map").on_hover_text("Shows the Julia set of the point under the mouse in the top right of the image, live as the mouse moves; a middle click pins the point (and unpins it)");
            if self.julia_pinned.is_some() && ui.small_button("Unpin").clicked() {
                self.julia_pinned = None;
            }
        });
        ui.horizontal(|ui| {
            if let Some(c) = self.m.julia {
                ui.label(format!("Julia set of c = {}", super::julia::complex_text(c)));
                if ui.button("Back to the Mandelbrot set").on_hover_text("Shows the Mandelbrot set again, at the view it had before").clicked() {
                    self.leave_julia();
                }
            } else {
                let center = self.c.center();
                let (c, of) = self.julia_pinned.map_or(((center.x, center.y), "the center"), |pinned| (pinned, "the pinned point"));
                if ui.button(format!("Julia set of {of}")).on_hover_text("Shows the Julia set of that point in the whole view, to explore like the Mandelbrot set (on the CPU: the GPU's loops are the Mandelbrot set's). Pin a point with a middle click on the image (with Julia map), or move it to the center").clicked() {
                    self.enter_julia(c);
                }
            }
        });
    }

    /// The depth slider and *Lock center*
    pub(super) fn depth_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let mut zoom = 1.0 / self.c.get_scale();
            //(Only what is dragged or typed is clamped: a view beyond the range of the slider, zoomed out further with the wheel, stays where it is)
            let response = ui.add(egui::Slider::new(&mut zoom, DEPTH_ZOOMS).clamping(egui::SliderClamping::Edits).logarithmic(true).text("depth").custom_formatter(|zoom, _| format!("×{zoom:.3e}")).custom_parser(|text| text.trim().trim_start_matches(['×', 'x']).trim().parse().ok()))
                .on_hover_text("Zooms in and out around the center of the view, without ever moving it: explore a dive up and down its exact path, and tune its colors at chosen depths");
            self.depth_dragging = response.dragged();
            if response.changed() {
                self.gesture_active = self.depth_dragging; //A frame of a gesture while the slider moves; the full quality when it stops (see `settle_gesture`)
                if self.set_depth(zoom) {
                    self.render(&format!("Set the depth to ×{:.3e} around the center", 1.0 / self.c.get_scale()));
                }
            }
            ui.checkbox(&mut self.lock_center, "Lock center").on_hover_text("The mouse keeps the center of the view too: the wheel zooms around the center instead of the cursor, and dragging and right-clicking do not move the view");
        });
    }
}

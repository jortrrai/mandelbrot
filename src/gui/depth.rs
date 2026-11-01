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

    /// The rotation of the view (issue #217): a slider with a number to type, in degrees or radians. It turns the image around the center of the view, which
    /// stays exactly where it is; switching the unit only changes how the angle is shown
    pub(super) fn rotation_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let radians = self.c.rotation();
            let (mut value, range, suffix) = if self.rotation_in_degrees { (radians.to_degrees(), -180.0..=180.0, "°") } else { (radians, -std::f64::consts::PI..=std::f64::consts::PI, " rad") };
            //(Any typed angle is taken, a whole turn more or less is the same view; only a change by the user turns the view, never the slider's own rounding)
            let response = ui.add(egui::Slider::new(&mut value, range).clamping(egui::SliderClamping::Never).max_decimals(if self.rotation_in_degrees { 2 } else { 4 }).suffix(suffix).text("rotation"))
                .on_hover_text("Turns the image counterclockwise around the center of the view, which stays where it is (type any angle; 0 is unturned). Places, images, Copy view, keyframes and flights keep the rotation");
            if response.changed() {
                let turned = if self.rotation_in_degrees { value.to_radians() } else { value };
                self.c.set_rotation(turned);
                self.render(&format!("Turned the view to {:.2}°", self.c.rotation().to_degrees()));
            }
            ui.selectable_value(&mut self.rotation_in_degrees, true, "degrees").on_hover_text("Show and type the rotation in degrees (the angle stays the same)");
            ui.selectable_value(&mut self.rotation_in_degrees, false, "radians").on_hover_text("Show and type the rotation in radians (the angle stays the same)");
        });
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

//! The center-locked depth control (issue #193): a zoom slider that changes only the scale of the view, never its center, so a dive can be explored up and
//! down along its exact vertical path, and its colors tuned at chosen depths (and those views saved as keyframes of a sequence, see `sequence`).
//!
//! *Lock center* makes the mouse keep the center too: the wheel zooms around the center instead of the cursor, and dragging does not move the view.
//! While the slider is dragged, frames render like those of a gesture (one sample per pixel); the full quality follows when it stops.

use eframe::egui;

use super::MandelbrotApp;
use crate::complex_plane::{Precision, View};

/// The range of the depth slider, in zoom (1 / scale): from ten times further out than the reset view to 10^300, past where 64-bit coordinates end and the
/// deep view takes over (see `deep_view`), as far as the 64-bit floats of its differences reach
pub(super) const DEPTH_ZOOMS: std::ops::RangeInclusive<f64> = 0.1..=1e300;

impl MandelbrotApp {
    /// The center of the view exactly as the depth control keeps it: the plane stores its edges, so its center read back can differ in the last bits, and
    /// many depth changes would add those up. While the view is where the depth control left it, this is the center it was given
    pub(super) fn exact_center(&self) -> crate::complex::Complex {
        let center = self.c.center();
        match &self.depth_center {
            //(Within a millionth of the view, or a few steps of a 64-bit number: what reading the center back from the edges can change)
            Some(kept) if (kept.x - center.x).abs() <= (self.c.get_scale() * 1e-6).max(center.x.abs() * 4.0 * f64::EPSILON)
                && (kept.y - center.y).abs() <= (self.c.get_scale() * 1e-6).max(center.y.abs() * 4.0 * f64::EPSILON) => crate::complex::Complex::new(kept.x, kept.y),
            _ => center,
        }
    }

    /// Sets the zoom of the view to `zoom`, keeping its center exactly; a zoom deeper than the precision or the zoom limit allows is refused (false)
    pub(super) fn set_depth(&mut self, zoom: f64) -> bool {
        let zoom = zoom.clamp(*DEPTH_ZOOMS.start(), *DEPTH_ZOOMS.end());
        let center = self.exact_center();
        let mut deeper = self.c.clone();
        deeper.set_view(&View::new(center.x, center.y, 1.0 / zoom));
        let zooming_in = deeper.get_scale() < self.shown_scale();
        if zooming_in && self.zoom_refused(&deeper) {
            return false;
        }
        if deeper.precision() == Precision::Exhausted || self.deep_view().is_some() {
            self.set_deep_zoom(zoom); //Deeper than 64-bit floats (or back out of it): the deep view, exactly (see `deep_view`)
            return true;
        }
        self.c = deeper;
        self.depth_center = Some(crate::complex::Complex::new(center.x, center.y));
        true
    }

    /// The depth slider and *Lock center*
    pub(super) fn depth_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let mut zoom = 1.0 / self.shown_scale();
            let response = ui.add(egui::Slider::new(&mut zoom, DEPTH_ZOOMS).logarithmic(true).text("depth").custom_formatter(|zoom, _| format!("×{zoom:.3e}")).custom_parser(|text| text.trim().trim_start_matches(['×', 'x']).trim().parse().ok()))
                .on_hover_text("Zooms in and out around the center of the view, without ever moving it: explore a dive up and down its exact path, and tune its colors at chosen depths");
            self.depth_dragging = response.dragged();
            if response.changed() {
                self.gesture_active = self.depth_dragging; //A frame of a gesture while the slider moves; the full quality when it stops (see `settle_gesture`)
                if self.set_depth(zoom) {
                    self.render(&format!("Set the depth to ×{:.3e} around the center", 1.0 / self.shown_scale()));
                }
            }
            ui.checkbox(&mut self.lock_center, "Lock center").on_hover_text("The mouse keeps the center of the view too: the wheel zooms around the center instead of the cursor, and dragging does not move the view");
        });
    }
}

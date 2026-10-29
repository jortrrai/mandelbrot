//! The mouse on the image: a click shows a point, a right click centers on it, the wheel zooms in and out around the cursor and dragging moves the view.
//!
//! The view is changed and rendered at once for every step of a wheel or a drag, which interrupts the render that is still running (see `MandelbrotApp::start_render`).
//! A render of a view that takes longer than a frame therefore only finishes after the gesture stops. While a gesture goes on, the status log does not get a line for every render.

use eframe::egui;

use crate::complex::Complex;
use crate::complex_plane::{ComplexPlane, Precision};
use crate::rendering;
use super::MandelbrotApp;

/// The zoom of the wheel for every point that is scrolled: a notch of the wheel is about 50 points, which zooms by 10%
const WHEEL_ZOOM_PER_POINT: f64 = 0.998;

/// The factor to scale the plane by after the wheel scrolled `scroll` points (up is positive): less than 1 zooms in, more than 1 zooms out
pub(super) fn wheel_zoom_factor(scroll: f32) -> f64 {
    WHEEL_ZOOM_PER_POINT.powf(f64::from(scroll))
}

/// Scales `plane` by `factor` so that the point `anchor` stays where it is on the screen: the center moves to `anchor + (center - anchor) * factor`
pub(super) fn zoom_around(plane: &mut ComplexPlane, anchor: &Complex, factor: f64) {
    let center = plane.center();
    let new_center = Complex::new(anchor.x + (center.x - anchor.x) * factor, anchor.y + (center.y - anchor.y) * factor);
    plane.scale(factor);
    plane.set_center(&new_center);
}

/// The whole pixels to move by for a drag of `delta_points`, when a pixel of the image is `points_per_pixel` points wide on the screen.
/// What is left of a fraction of a pixel is kept in `remainder` for the next frame, so a slow drag still moves.
pub(super) fn whole_pixels(delta_points: f32, remainder: &mut f32, points_per_pixel: f32) -> i64 {
    let total = *remainder + delta_points / points_per_pixel.max(f32::EPSILON);
    let whole = total.trunc();
    *remainder = total - whole;
    whole as i64
}

impl MandelbrotApp {
    /// Handles the mouse on the image `response`, which is shown with the size `size` in points
    pub(super) fn handle_image_pointer(&mut self, ui: &egui::Ui, response: &egui::Response, size: egui::Vec2) {
        let (width, height) = (self.p.pixel_plane.width as f32, self.p.pixel_plane.height as f32);
        let to_buffer = |position: egui::Pos2| {
            let relative = (position - response.rect.min) / size;
            (relative.x * width, relative.y * height)
        };
        let dragging = response.dragged_by(egui::PointerButton::Primary);
        let scroll = if response.hovered() { ui.input(|input| input.smooth_scroll_delta.y) } else { 0.0 };
        self.gesture_active = dragging || scroll.abs() >= 1.0;

        if let Some(position) = response.interact_pointer_pos() {
            let (x, y) = to_buffer(position);
            if response.clicked() {
                self.handle_image_click(x, y, false);
            } else if response.secondary_clicked() {
                self.handle_image_click(x, y, true);
            }
        }
        if dragging {
            let drag = response.drag_delta();
            let points_per_pixel = size.x / width;
            let columns = whole_pixels(-drag.x, &mut self.drag_remainder.0, points_per_pixel);
            let rows_up = whole_pixels(drag.y, &mut self.drag_remainder.1, points_per_pixel);
            if columns != 0 || rows_up != 0 {
                rendering::move_complex_plane_by_pixels(&mut self.c, rows_up, columns);
                self.render_moved("Moved by dragging");
            }
        } else {
            self.drag_remainder = (0.0, 0.0);
        }
        if scroll.abs() >= 1.0 {
            let factor = wheel_zoom_factor(scroll);
            let zooming_in = factor < 1.0;
            if !(zooming_in && self.c.precision() == Precision::Exhausted) {
                if let Some(position) = ui.input(|input| input.pointer.hover_pos()) {
                    let (x, y) = to_buffer(position);
                    let anchor = self.displayed_plane().complex_from_pixel_plane(f64::from(x), f64::from(y)); //The point under the mouse on the screen
                    let mut zoomed = self.c.clone();
                    zoom_around(&mut zoomed, &anchor, factor);
                    if !(zooming_in && self.zoom_refused(&zoomed)) {
                        self.c = zoomed;
                        self.render(if zooming_in { "Zoomed in with the wheel" } else { "Zoomed out with the wheel" });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex_plane::View;

    #[test]
    fn a_notch_of_the_wheel_zooms_by_about_ten_percent_and_scrolling_back_undoes_it() {
        assert!((wheel_zoom_factor(50.0) - 0.905).abs() < 0.001, "{}", wheel_zoom_factor(50.0));
        assert!(wheel_zoom_factor(-50.0) > 1.0);
        assert!((wheel_zoom_factor(37.0) * wheel_zoom_factor(-37.0) - 1.0).abs() < 1e-12);
        assert!((wheel_zoom_factor(0.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zooming_around_a_point_keeps_that_point_under_the_cursor() {
        let mut plane = ComplexPlane::new(1200, 800);
        plane.set_view(&View::new(-0.75, 0.1, 0.5));
        let (x, y) = (900.0, 150.0);
        let anchor = plane.complex_from_pixel_plane(x, y);
        zoom_around(&mut plane, &anchor, 0.25);
        let after = plane.complex_from_pixel_plane(x, y);
        assert!((after.x - anchor.x).abs() < 1e-14 && (after.y - anchor.y).abs() < 1e-14, "{after:?} {anchor:?}");
        assert!((plane.get_scale() - 0.125).abs() < 1e-14);
        zoom_around(&mut plane, &anchor, 4.0);
        let back = plane.complex_from_pixel_plane(x, y);
        assert!((back.x - anchor.x).abs() < 1e-14 && (plane.get_scale() - 0.5).abs() < 1e-14);
    }

    #[test]
    fn a_drag_moves_by_whole_pixels_and_keeps_the_fractions_for_the_next_frame() {
        let mut remainder = 0.0;
        //A pixel is 2 points wide: 3 points are 1 pixel and half a pixel
        assert_eq!(whole_pixels(3.0, &mut remainder, 2.0), 1);
        assert!((remainder - 0.5).abs() < 1e-6);
        assert_eq!(whole_pixels(3.0, &mut remainder, 2.0), 2, "0.5 + 1.5 pixels");
        assert!(remainder.abs() < 1e-6);
        assert_eq!(whole_pixels(-0.5, &mut remainder, 2.0), 0, "a slow drag moves nothing yet");
        let mut slow = 0.0;
        let moved: i64 = (0..8).map(|_| whole_pixels(0.5, &mut slow, 1.0)).sum();
        assert_eq!(moved, 4, "eight slow frames add up to four pixels");
        let mut fresh = 0.0;
        assert_eq!(whole_pixels(10_000.0, &mut fresh, 1.0), 10_000, "a huge jump moves all the way (there is no limit per frame)");
    }
}

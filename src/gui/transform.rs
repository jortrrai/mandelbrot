//! The image on the screen follows the view at once: while a render of a new view runs, the last image is drawn moved and scaled to where it belongs in the new view.
//!
//! A drag or a wheel zoom changes the view on every frame, but a render of the new view takes from a few milliseconds to seconds. Until its first layer arrives, the image of the
//! previous view is drawn at its place in the new view (see `shown_fraction`), so the picture moves with the mouse at the frame rate of the window. The parts of the new view
//! that the old image does not cover stay black until the render fills them.

use eframe::egui;

use crate::complex_plane::ComplexPlane;

/// Where the image of the view `shown` belongs in the image of the view `wanted`, in fractions of the wanted image: from (0, 0) to (1, 1) when the views are the same.
/// A moved view gives a shifted rectangle, a zoomed view a larger (zoomed in) or smaller (zoomed out) one. The sizes of the images in pixels do not matter, only the part of the plane they show.
pub(super) fn shown_fraction(shown: &ComplexPlane, wanted: &ComplexPlane) -> egui::Rect {
    //The bounds are stored with the orientation of the screen, so `min_y` is the top row of both images
    let x = |value: f64| ((value - wanted.min_x) / (wanted.max_x - wanted.min_x)) as f32;
    let y = |value: f64| ((value - wanted.min_y) / (wanted.max_y - wanted.min_y)) as f32;
    egui::Rect::from_min_max(egui::pos2(x(shown.min_x), y(shown.min_y)), egui::pos2(x(shown.max_x), y(shown.max_y)))
}

/// The rectangle on the screen of `fraction` (of the image, see `shown_fraction`) when the image fills `image_rect`
pub(super) fn rect_of(fraction: egui::Rect, image_rect: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_max(image_rect.lerp_inside(fraction.min.to_vec2()), image_rect.lerp_inside(fraction.max.to_vec2()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex_plane::View;
    use crate::gui::pointer::zoom_around;
    use crate::rendering;

    fn assert_close(actual: egui::Rect, expected: egui::Rect) {
        let error = (actual.min - expected.min).length() + (actual.max - expected.max).length();
        assert!(error < 1e-5, "{actual:?} is not {expected:?}");
    }

    fn plane() -> ComplexPlane {
        let mut plane = ComplexPlane::new(1200, 800);
        plane.set_view(&View::new(-0.75, 0.1, 0.5));
        plane
    }

    #[test]
    fn the_same_view_fills_the_image() {
        assert_close(shown_fraction(&plane(), &plane()), egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)));
    }

    #[test]
    fn a_moved_view_shifts_the_old_image_the_other_way() {
        let old = plane();
        let mut new = old.clone();
        //Move the view 120 columns to the right and 80 rows up: the content goes left and down
        rendering::translate_complex_plane_by_pixels(&mut new, 80, 120);
        let (dx, dy) = (120.0 / 1200.0, 80.0 / 800.0);
        assert_close(shown_fraction(&old, &new), egui::Rect::from_min_max(egui::pos2(-dx, dy), egui::pos2(1.0 - dx, 1.0 + dy)));
    }

    #[test]
    fn a_zoom_around_a_point_scales_the_old_image_around_that_point() {
        let old = plane();
        let (x, y) = (900.0, 150.0);
        let anchor = old.complex_from_pixel_plane(x, y);
        let mut new = old.clone();
        zoom_around(&mut new, &anchor, 0.5);
        let rect = rect_of(shown_fraction(&old, &new), egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0)));
        assert!((rect.width() - 2400.0).abs() < 1e-2 && (rect.height() - 1600.0).abs() < 1e-2, "zooming in by 2 doubles the old image: {rect:?}");
        //The point under the cursor stays under the cursor
        let at = rect.lerp_inside(egui::vec2((x / 1200.0) as f32, (y / 800.0) as f32));
        assert!((f64::from(at.x) - x).abs() < 1e-2 && (f64::from(at.y) - y).abs() < 1e-2, "{at:?}");
        zoom_around(&mut new, &anchor, 4.0);
        assert!((shown_fraction(&old, &new).width() - 0.5).abs() < 1e-6, "zooming out by 2 halves it");
    }

    #[test]
    fn a_resized_view_keeps_the_old_image_in_place() {
        let old = plane();
        let new = old.resized(1500, 800);
        let fraction = shown_fraction(&old, &new);
        assert!(fraction.min.x.abs() < 1e-6 && (fraction.width() - 1.0).abs() < 1e-6, "the same part of the real axis: {fraction:?}");
        assert!((fraction.center().y - 0.5).abs() < 1e-6 && (fraction.height() - 1500.0 / 1200.0).abs() < 1e-5, "a wider window shows less of the imaginary axis, around the same center: {fraction:?}");
    }

    #[test]
    fn several_moves_add_up_to_one_shift_of_the_buffer() {
        let old = plane();
        let mut new = old.clone();
        rendering::translate_complex_plane_by_pixels(&mut new, 3, -7);
        assert_eq!(rendering::pixel_shift(&old, &new), Some((3, 7)), "the same shift that a drag gives `render_moved`");
        rendering::translate_complex_plane_by_pixels(&mut new, -10, 25);
        assert_eq!(rendering::pixel_shift(&old, &new), Some((-7, -18)));
        assert_eq!(rendering::pixel_shift(&old, &old), Some((0, 0)));
        let mut zoomed = old.clone();
        zoomed.scale(0.9);
        assert_eq!(rendering::pixel_shift(&old, &zoomed), None, "a zoom is not a shift");
        assert_eq!(rendering::pixel_shift(&old, &old.resized(1300, 800)), None, "neither is a resize");
    }
}

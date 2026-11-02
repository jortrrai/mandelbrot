//! The part of an image that has to be rendered again after a move, and the moves of the complex plane by whole pixels.

use crate::complex_plane::ComplexPlane;

/// The pixels of a buffer that have to be rendered again after the buffer was shifted with `PixelBuffer::translate_buffer`:
///
/// the exposed `columns` and the exposed `rows`. A pixel is exposed if its column or its row is exposed, so a shift in one direction exposes a strip,
/// and a shift in both directions an L shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposedRegion {
    /// The exposed columns: the pixels of these columns have to be rendered again
    pub columns: std::ops::Range<usize>,
    /// The exposed rows
    pub rows: std::ops::Range<usize>,
}

impl ExposedRegion {
    /// Every pixel of a `width` x `height` buffer
    pub fn full(width: usize, height: usize) -> ExposedRegion {
        ExposedRegion { columns: 0..width, rows: 0..height }
    }

    /// The pixels exposed by shifting a `width` x `height` buffer with `PixelBuffer::translate_buffer(rows_up, columns_right)`.
    /// Returns `None` if the shift is as large as the buffer, since nothing of the old image is then left.
    pub fn from_shift(width: usize, height: usize, rows_up: i128, columns_right: i128) -> Option<ExposedRegion> {
        if rows_up.unsigned_abs() >= height as u128 || columns_right.unsigned_abs() >= width as u128 {
            return None;
        }
        let exposed = |shift: i128, size: usize| {
            let amount = shift.unsigned_abs() as usize;
            if shift > 0 { 0..amount } else { (size - amount)..size } //Content moves towards the end for positive shifts, so the start is exposed
        };
        Some(ExposedRegion { columns: exposed(columns_right, width), rows: exposed(rows_up, height) })
    }

    /// Whether the pixel at (`x`, `y`) is exposed
    pub fn contains(&self, x: usize, y: usize) -> bool {
        self.columns.contains(&x) || self.rows.contains(&y)
    }

    /// The amount of pixels in the region of a `width` x `height` buffer
    pub fn pixel_count(&self, width: usize, height: usize) -> usize {
        let (columns, rows) = (self.columns.len().min(width), self.rows.len().min(height));
        columns * height + rows * width - columns * rows
    }
}

/// Moves the complex plane `c` up by `rows_up` and right by `columns_right` whole pixels (any distance), without rendering anything.
///
/// After the move the plane is the old plane shifted by whole pixels, see `pixel_shift`.
pub fn move_complex_plane_by_pixels(c: &mut ComplexPlane, rows_up: i64, columns_right: i64) {
    if c.rotation() == 0.0 {
        c.translate(columns_right as f64 * c.increment_x, -(rows_up as f64) * c.increment_y);
        return;
    }
    //A turned view moves along the screen's axes, which are turned in the plane (see `ComplexPlane::screen_offset`)
    let (right, up) = c.screen_offset(columns_right as f64 * c.increment_x, rows_up as f64 * c.increment_y);
    c.translate(right, -up);
}

/// The shift of the buffer that turns the image of `shown` into the image of `wanted`, if `wanted` is `shown` moved by whole pixels.
///
/// The shift is `(rows_up, columns_right)` for [`crate::pixel_buffer::PixelBuffer::translate_buffer`]; `None` if the views differ in size or zoom. Unlike the move of one frame, this also covers several moves whose renders were interrupted.
pub fn pixel_shift(shown: &ComplexPlane, wanted: &ComplexPlane) -> Option<(i128, i128)> {
    //Turned views: the same rotation, compared as their unturned views (see `ComplexPlane::unturned`)
    if shown.rotation().to_bits() != wanted.rotation().to_bits() {
        return None;
    }
    if shown.rotation() != 0.0 {
        return pixel_shift(&shown.unturned(), &wanted.unturned());
    }
    //A move keeps the size of the pixels exactly (`ComplexPlane::translate` only adds to the bounds), a zoom or a resize changes it
    let same_size = |a: f64, b: f64| a.to_bits() == b.to_bits();
    let same_grid = same_size(shown.increment_x, wanted.increment_x) && same_size(shown.increment_y, wanted.increment_y) && same_size(shown.length_x, wanted.length_x) && same_size(shown.length_y, wanted.length_y);
    let columns = (wanted.min_x - shown.min_x) / wanted.increment_x;
    let rows_up = (shown.min_y - wanted.min_y) / wanted.increment_y;
    let whole = |value: f64| (value - value.round()).abs() < 1e-6 && value.abs() < 1e9;
    (same_grid && whole(columns) && whole(rows_up)).then(|| (rows_up.round() as i128, -(columns.round() as i128)))
}

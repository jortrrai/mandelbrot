//! The part of an image that has to be rendered again after a move, and the moves of the complex plane by whole pixels.

use crate::{complex::Complex, complex_plane::ComplexPlane};

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

/// Moves the complex plane `c` so it is centered on `new_center`, without rendering anything.
/// Returns the `(rows_up, columns_right)` to shift the pixel buffer with `PixelBuffer::translate_buffer`.
pub fn center_complex_plane_on(c: &mut ComplexPlane, new_center: &Complex) -> (i128, i128) {
    let mut translation: Complex = new_center.subtract(&c.center());
    //Mirror the y translation because the screen y is mirrored compared to the complex plane y axis
    translation.y = -translation.y;
    c.translate(translation.x, 0.0);
    let columns = -c.real_to_pixels(translation.x);
    c.translate(0.0, translation.y);
    let rows = -c.imaginary_to_pixels(translation.y);
    (i128::from(rows), i128::from(columns))
}

/// Moves the complex plane `c` up by `rows_up` and right by `columns_right` pixels, without rendering anything.
///
/// Each distance is taken modulo 256 pixels (`as u8`): a move of 256 pixels moves by 0 and one of 300 by 44. The window only moves by small steps.
pub fn translate_complex_plane_by_pixels(c: &mut ComplexPlane, rows_up: i16, columns_right: i16) {
    let row_sign: f64 = if rows_up > 0 {-1.0} else {1.0};
    let column_sign: f64 = if columns_right > 0 {1.0} else {-1.0};
    c.translate(column_sign*c.pixels_to_real(columns_right.unsigned_abs() as u8), row_sign*c.pixels_to_imaginary(rows_up.unsigned_abs() as u8)); 
}

/// The shift of the buffer that turns the image of `shown` into the image of `wanted`, if `wanted` is `shown` moved by whole pixels.
///
/// The shift is `(rows_up, columns_right)` for [`crate::pixel_buffer::PixelBuffer::translate_buffer`]; `None` if the views differ in size or zoom. Unlike the move of one frame, this also covers several moves whose renders were interrupted.
pub fn pixel_shift(shown: &ComplexPlane, wanted: &ComplexPlane) -> Option<(i128, i128)> {
    //A move keeps the size of the pixels exactly (`ComplexPlane::translate` only adds to the bounds), a zoom or a resize changes it
    let same_size = |a: f64, b: f64| a.to_bits() == b.to_bits();
    let same_grid = same_size(shown.increment_x, wanted.increment_x) && same_size(shown.increment_y, wanted.increment_y) && same_size(shown.length_x, wanted.length_x) && same_size(shown.length_y, wanted.length_y);
    let columns = (wanted.min_x - shown.min_x) / wanted.increment_x;
    let rows_up = (shown.min_y - wanted.min_y) / wanted.increment_y;
    let whole = |value: f64| (value - value.round()).abs() < 1e-6 && value.abs() < 1e9;
    (same_grid && whole(columns) && whole(rows_up)).then(|| (rows_up.round() as i128, -(columns.round() as i128)))
}

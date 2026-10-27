//! The part of an image that has to be rendered again after a move, and the moves of the complex plane by whole pixels.

use crate::{complex::Complex, complex_plane::ComplexPlane};

/// The pixels of a buffer that have to be rendered again after the buffer was shifted with `PixelBuffer::translate_buffer`:
/// the exposed `columns` and the exposed `rows`. A pixel is exposed if its column or its row is exposed, so a shift in one direction exposes a strip,
/// and a shift in both directions an L shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposedRegion {
    pub columns: std::ops::Range<usize>,
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

///# Panics
/// If `rows_up` != 0 && `columns_right` != 0
/// Moves the complex plane `c` up by `rows_up` and right by `columns_right` pixels, without rendering anything
pub fn translate_complex_plane_by_pixels(c: &mut ComplexPlane, rows_up: i16, columns_right: i16) {
    let row_sign: f64 = if rows_up > 0 {-1.0} else {1.0};
    let column_sign: f64 = if columns_right > 0 {1.0} else {-1.0};
    c.translate(column_sign*c.pixels_to_real(columns_right.unsigned_abs() as u8), row_sign*c.pixels_to_imaginary(rows_up.unsigned_abs() as u8)); 
}

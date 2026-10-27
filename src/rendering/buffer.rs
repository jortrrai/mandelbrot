//! Rendering a whole image into a pixel buffer in one go: used when an image is saved at another size than the window shows.

use super::Render;
use crate::{coloring::TrueColor, complex_plane::ComplexPlane, mandelbrot_set::MandelbrotSet, pixel_buffer::PixelBuffer};

/// Renders the complex plane `c` into the buffer `p` (its colors, and then its pixels) with `supersampling_amount` samples per pixel (from 1 to 64).
///
/// The buffer decides the size of the image. The colors are those of the last pass of [`Render`]: the average of all the samples of a pixel, so this is the same image that the window shows after its last layer.
/// This is computationally intensive; it is not meant for moves (the window shifts its buffer and only renders what a move exposes).
pub fn render_complex_plane_into_buffer(p: &mut PixelBuffer, c: &ComplexPlane, m: &MandelbrotSet, supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor) {
    let (width, height) = (p.pixel_plane.width, p.pixel_plane.height);
    Render::new(c, m, width, height, coloring_function).supersampling(supersampling_amount).run(|colors, _| p.colors.copy_from_slice(colors));
    p.update_pixels();
}

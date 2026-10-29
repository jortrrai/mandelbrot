//! Rendering a whole image into a pixel buffer in one go: used when an image is saved at another size than the window shows.

use std::time::Duration;

use super::gpu::GpuIterator;
use super::Render;
use crate::{coloring::TrueColor, complex_plane::ComplexPlane, mandelbrot_set::MandelbrotSet, pixel_buffer::PixelBuffer};

/// Renders the complex plane `c` into the buffer `p` (its colors, and then its pixels) with `supersampling_amount` samples per pixel (from 1 to 64).
///
/// The buffer decides the size of the image. The colors are those of the last pass of [`Render`]: the average of all the samples of a pixel, so this is the same image that the window shows after its last layer.
/// This is computationally intensive; it is not meant for moves (the window shifts its buffer and only renders what a move exposes).
pub fn render_complex_plane_into_buffer(p: &mut PixelBuffer, c: &ComplexPlane, m: &MandelbrotSet, supersampling_amount: u8, coloring_function: fn(position: f64, length: f64) -> TrueColor) {
    render_complex_plane_into_buffer_with_gpu(p, c, m, supersampling_amount, coloring_function, None);
}

/// `render_complex_plane_into_buffer`, with the GPU helping from the start of every pass when there is one. The GPU computes bit for bit the same samples
/// (see `gpu`), so the image is the same.
pub fn render_complex_plane_into_buffer_with_gpu(p: &mut PixelBuffer, c: &ComplexPlane, m: &MandelbrotSet, supersampling_amount: u8, coloring_function: fn(position: f64, length: f64) -> TrueColor, gpu: Option<&GpuIterator>) {
    let (width, height) = (p.pixel_plane.width, p.pixel_plane.height);
    //An image that is saved is rendered in one go: the GPU can join at once
    Render::new(c, m, width, height, coloring_function).supersampling(supersampling_amount).gpu(gpu.map(|gpu| (gpu, Duration::ZERO))).run(|colors, _| p.colors.copy_from_slice(colors));
    p.update_pixels();
}

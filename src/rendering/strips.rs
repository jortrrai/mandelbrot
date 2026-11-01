//! Rendering a large image in strips of rows, so its memory stays bounded however tall it is.
//!
//! A render holds about 20 bytes per pixel (the sums of the samples and the colors): over 10 GB for a whole image of 32K at once. A strip has at most
//! `STRIP_PIXELS` pixels.
//!
//! A strip places its samples in the plane of the whole image (see `Render::rows_from`), so the strips together are bit for bit the image rendered in one go:
//! there are no seams.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use super::gpu::GpuIterator;
use super::Render;
use crate::{coloring::TrueColor, complex_plane::ComplexPlane, mandelbrot_set::MandelbrotSet};

/// The most pixels of a strip: about 8 megapixels, 160 MB of samples and colors while it renders (273 rows of a 32K image; an image up to 4K is one strip)
pub const STRIP_PIXELS: usize = 1 << 23;

/// The rows of every strip of an image of `width` x `height` pixels with at most `strip_pixels` pixels a strip (at least one row each), top to bottom
pub fn strips(width: usize, height: usize, strip_pixels: usize) -> Vec<Range<usize>> {
    let rows = (strip_pixels / width.max(1)).clamp(1, height.max(1));
    (0..height).step_by(rows).map(|start| start..(start + rows).min(height)).collect()
}

/// How far a render in strips is, for a progress bar: it is updated while it runs, from the threads of the render
#[derive(Debug, Default)]
pub struct StripProgress {
    /// The rows of the strips that are done
    rows_done: AtomicUsize,
    /// The rows of the strip that is rendering
    rows_now: AtomicUsize,
    /// The passes of the strip that is rendering that are done, and the pixels of the pass after them
    passes_done: AtomicUsize,
    pixels_done: AtomicUsize,
}

impl StripProgress {
    /// The part of an image of `width` x `height` pixels with `passes` passes (samples per pixel) that is done, from 0 to 1
    pub fn fraction(&self, width: usize, height: usize, passes: usize) -> f64 {
        let (rows_now, passes) = (self.rows_now.load(Ordering::Relaxed) as f64, passes.max(1) as f64);
        let pass = (self.pixels_done.load(Ordering::Relaxed) as f64 / (rows_now * width as f64).max(1.0)).min(1.0);
        let strip = ((self.passes_done.load(Ordering::Relaxed) as f64 + pass) / passes).min(1.0);
        ((self.rows_done.load(Ordering::Relaxed) as f64 + rows_now * strip) / height.max(1) as f64).min(1.0)
    }
}

/// Renders the `width` x `height` image of the plane `c` strip by strip and gives every strip to `on_strip`: its rows and their colors, row by row.
///
/// Every strip has at most `strip_pixels` pixels (see `strips`) and `supersampling_amount` samples per pixel, with the GPU helping when there is one. The strips
/// together are the image of `render_complex_plane_into_buffer`, bit for bit.
///
/// Returns `Ok(false)` if it was cancelled (by `cancel`, between or within strips; the strip that was rendering is not given), and the error of `on_strip` if it
/// fails: then it stops.
/// # Errors
/// The first error of `on_strip`
#[allow(clippy::too_many_arguments)] // The image, how to render it, and its progress and cancellation
pub fn render_in_strips<E>(c: &ComplexPlane, m: &MandelbrotSet, (width, height): (usize, usize), supersampling_amount: u8, coloring_function: fn(f64, f64) -> TrueColor, gpu: Option<&GpuIterator>,
    strip_pixels: usize, cancel: &AtomicBool, progress: &StripProgress, mut on_strip: impl FnMut(Range<usize>, Vec<TrueColor>) -> Result<(), E>) -> Result<bool, E> {
    for rows in strips(width, height, strip_pixels) {
        if cancel.load(Ordering::Relaxed) {
            return Ok(false);
        }
        progress.rows_now.store(rows.len(), Ordering::Relaxed);
        progress.passes_done.store(0, Ordering::Relaxed);
        let mut colors = Vec::new();
        //(An image that is saved is rendered in one go: the GPU can join at once)
        let completed = Render::new(c, m, width, rows.len(), coloring_function).supersampling(supersampling_amount).rows_from(rows.start).gpu(gpu.map(|gpu| (gpu, Duration::ZERO)))
            .cancel_with(cancel).count_pixels_in(&progress.pixels_done).run(|pass_colors, pass| {
                progress.passes_done.store(usize::from(pass), Ordering::Relaxed);
                if pass == supersampling_amount.clamp(1, 64) {
                    colors = pass_colors.to_vec();
                }
            });
        if !completed {
            return Ok(false);
        }
        let done = rows.end;
        on_strip(rows, colors)?;
        progress.rows_done.store(done, Ordering::Relaxed);
        progress.rows_now.store(0, Ordering::Relaxed);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer}, rendering::render_complex_plane_into_buffer};

    #[test]
    fn the_strips_cover_every_row_once_within_the_pixels_of_a_strip() {
        for (width, height, budget) in [(30_720, 17_280, STRIP_PIXELS), (15_360, 8640, STRIP_PIXELS), (7, 10, 21), (7, 10, 20), (100, 3, 1), (5, 5, 1_000)] {
            let strips = strips(width, height, budget);
            let rows: Vec<usize> = strips.iter().flat_map(Clone::clone).collect();
            assert_eq!(rows, (0..height).collect::<Vec<_>>(), "every row once, in order ({width} x {height})");
            assert!(strips.iter().all(|strip| strip.len() * width <= budget.max(width)), "no strip has more pixels than the budget (or one row)");
        }
        assert_eq!(strips(30_720, 17_280, STRIP_PIXELS).len(), 64, "273 rows a strip");
        assert_eq!(strips(3840, 2160, STRIP_PIXELS).len(), 1, "4K is one strip");
    }

    /// A view with fine detail and a part of the set, where a sample placed differently would show
    fn seahorse(width: usize, height: usize) -> (ComplexPlane, MandelbrotSet) {
        let mut c = ComplexPlane::new(width, height);
        c.set_view(&crate::complex_plane::View::new(-0.7498, 0.0203, 6.4e-5));
        (c, MandelbrotSet::new(1500, 2.0))
    }

    #[test]
    fn an_image_in_strips_is_the_image_rendered_in_one_go_without_seams() {
        let (width, height) = (61, 47);
        let (c, m) = seahorse(width, height);
        let coloring = crate::coloring::TrueColor::new_from_bernstein_polynomials;
        for samples in [1, 4] {
            let mut whole = PixelBuffer::new(PixelPlane::new(width, height));
            render_complex_plane_into_buffer(&mut whole, &c, &m, samples, coloring);
            //Strips of 5 rows (the last one 2) and of a single row
            for budget in [5 * width, width] {
                let mut joined = Vec::new();
                let (cancel, progress) = (AtomicBool::new(false), StripProgress::default());
                let completed = render_in_strips(&c, &m, (width, height), samples, coloring, None, budget, &cancel, &progress, |rows, colors| {
                    assert_eq!((rows.start, colors.len()), (joined.len() / width, rows.len() * width));
                    joined.extend(colors);
                    Ok::<(), ()>(())
                });
                assert_eq!(completed, Ok(true));
                let rgb = |colors: &[TrueColor]| colors.iter().map(|color| (color.red, color.green, color.blue)).collect::<Vec<_>>();
                assert!(rgb(&joined) == rgb(&whole.colors), "{samples}x in strips of {} rows: the same colors", budget / width);
                assert!((progress.fraction(width, height, usize::from(samples)) - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_render_in_strips_can_be_cancelled_and_stops_at_an_error() {
        let (width, height) = (40, 30);
        let (c, m) = seahorse(width, height);
        let coloring = crate::coloring::TrueColor::new_from_bernstein_polynomials;
        let (cancel, progress) = (AtomicBool::new(false), StripProgress::default());
        let mut given = 0;
        let completed = render_in_strips(&c, &m, (width, height), 2, coloring, None, 4 * width, &cancel, &progress, |_, _| {
            given += 1;
            if given == 2 {
                cancel.store(true, Ordering::Relaxed);
            }
            Ok::<(), ()>(())
        });
        assert_eq!((completed, given), (Ok(false), 2), "no strip after the cancel");
        let fraction = progress.fraction(width, height, 2);
        assert!((fraction - 8.0 / 30.0).abs() < 1e-9, "the progress stays at the strips that are done: {fraction}");
        let (cancel, progress) = (AtomicBool::new(false), StripProgress::default());
        let failed = render_in_strips(&c, &m, (width, height), 1, coloring, None, 4 * width, &cancel, &progress, |rows, _| if rows.start == 0 { Ok(()) } else { Err("disk full") });
        assert_eq!(failed, Err("disk full"));
    }
}

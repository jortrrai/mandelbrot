//! The progressive renderer: one supersampling layer at a time, so every layer can be shown as soon as it is computed.

use std::{thread, sync::{Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}}};

use super::adaptive::{adaptive_mask, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
use super::jitter::sample_jitter;
use super::region::ExposedRegion;
use super::render_thread_count;
use crate::{complex::Complex, complex_plane::ComplexPlane, coloring::TrueColor, mandelbrot_set::{MandelbrotSet, LANES}};

/// The largest `max_iterations` for which `render_region_progressively` precomputes the color of every iteration count
const MAX_COLOR_TABLE_ITERATIONS: u32 = 200_000;

/// Renders the whole Complex plane `c` progressively, one supersampling layer (pass) at a time, so intermediate images can be shown early.
/// See `render_region_progressively`.
pub fn render_progressively(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor, cancel: &AtomicBool, pixels_done: &AtomicUsize, on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
    let base = vec![TrueColor::new(0, 0, 0); width * height];
    render_region_progressively(c, m, width, height, &ExposedRegion::full(width, height), &base, supersampling_amount, coloring_function, cancel, pixels_done, on_pass)
}

/// Renders the pixels of the Complex plane `c` that are inside `region` progressively, one supersampling layer (pass) at a time, so intermediate images can be shown early.
///
/// All other pixels keep the colors of `base`, e.g. the shifted image of a previous render, so moving the view only renders the newly exposed pixels.
/// Pass `k` computes sample `k` of every pixel in the region, with the same deterministic jitter as `render_box_render_complex_plane_into_buffer`,
/// so after the last pass the colors are identical to the ones that function produces.
/// `pixels_done` is reset to 0 at the start of every pass and incremented for every finished pixel, so another thread can show how far the current pass is.
/// After every completed pass, `on_pass` is called with the colors (`width * height`, row by row): `base`, with the region replaced by the running average of all samples so far, and the pass number (1-based).
/// Returns `false` if `cancel` was set before all passes completed; the last completed pass has then been reported. A partially computed pass is discarded.
/// Note: This function is multithreaded; `on_pass` is called on the calling thread
/// # Panics
/// If `lock().unwrap()` panics, or if `base` does not have `width * height` colors
pub fn render_region_progressively(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, region: &ExposedRegion, base: &[TrueColor], supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor, cancel: &AtomicBool, pixels_done: &AtomicUsize, on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
    render_region_adaptively(c, m, width, height, region, base, supersampling_amount, None, coloring_function, cancel, pixels_done, on_pass)
}

/// Like `render_region_progressively`, but with `adaptive` set the extra samples are only taken where the picture shows detail.
///
/// The first `ADAPTIVE_UNIFORM_SAMPLES` samples are taken for every pixel, the later samples only for the pixels that `adaptive_mask` marks as boundary or detail pixels.
/// The other pixels keep the average of their first samples.
/// This is a quality/speed heuristic, not equivalent to full supersampling: a feature that neither of the first two samples of a pixel or its neighbors shows can be missed.
/// With `adaptive` unset, or at most `ADAPTIVE_UNIFORM_SAMPLES` samples, every pixel gets every sample, exactly like `render_region_progressively`.
/// The pass numbering is unchanged (`on_pass` is called once per sample); the later passes are just cheaper.
///
/// # Panics
/// If `base` does not have `width * height` colors, or if a thread of the render panics.
pub fn render_region_adaptively(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, region: &ExposedRegion, base: &[TrueColor], supersampling_amount: u8, adaptive: Option<&AdaptiveSampling>, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor, cancel: &AtomicBool, pixels_done: &AtomicUsize, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
    assert_eq!(base.len(), width * height, "base should have a color for every pixel");
    let supersampling_amount = supersampling_amount.clamp(1, 64);
    let amount_of_threads = render_thread_count();
    let mut sums: Vec<[u32; 3]> = vec![[0; 3]; width * height];
    let mut colors: Vec<TrueColor> = base.to_vec();
    //The color of a pixel only depends on its iteration count, which is at most `max_iterations`, so compute every color once
    let color_table: Option<Vec<TrueColor>> = (m.max_iterations <= MAX_COLOR_TABLE_ITERATIONS).then(|| (0..=m.max_iterations).map(|iterations| m.color(coloring_function, iterations)).collect());
    let adaptive = adaptive.filter(|_| supersampling_amount > ADAPTIVE_UNIFORM_SAMPLES);
    let mut first_colors: Vec<TrueColor> = Vec::new();
    let mut mask: Option<Vec<bool>> = None;
    for sample in 0..supersampling_amount {
        pixels_done.store(0, Ordering::Relaxed);
        let divisor = u32::from(sample) + 1;
        if let (Some(_), Some(mask)) = (adaptive, &mask) {
            //The pixels that get no sample in this pass count as done
            let skipped = (0..height).map(|y| (0..width).filter(|x| region.contains(*x, y) && !mask[y * width + x]).count()).sum();
            pixels_done.store(skipped, Ordering::Relaxed);
        }
        let mask_ref = mask.as_deref();
        let rows = Mutex::new(sums.chunks_mut(width).zip(colors.chunks_mut(width)).enumerate());
        thread::scope(|scope| {
            for _ in 0..amount_of_threads {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_row = rows.lock().unwrap().next();
                    let Some((y, (row, color_row))) = next_row else { return };
                    let xs: Vec<usize> = (0..width).filter(|x| region.contains(*x, y) && mask_ref.is_none_or(|mask| mask[y * width + x])).collect();
                    let sample_point = |x: usize| {
                        let (jitter_x, jitter_y) = sample_jitter(x, y, sample);
                        c.complex_from_pixel_plane(f64::from(x as u32) + jitter_x, f64::from(y as u32) + jitter_y)
                    };
                    let mut add_color = |x: usize, position: f64| {
                        let color = m.interpolated_color(position, |iterations| color_table.as_ref().map_or_else(|| m.color(coloring_function, iterations), |table| table[iterations as usize]));
                        let sum = &mut row[x];
                        sum[0] += u32::from(color.red);
                        sum[1] += u32::from(color.green);
                        sum[2] += u32::from(color.blue);
                    };
                    //Iterate LANES pixels at the same time, the remaining pixels one by one
                    let (groups, remainder) = xs.as_chunks::<LANES>();
                    for group in groups {
                        let points: [Complex; LANES] = std::array::from_fn(|lane| sample_point(group[lane]));
                        let positions = m.position_lanes(&points);
                        for (lane, x) in group.iter().enumerate() {
                            add_color(*x, positions[lane]);
                        }
                    }
                    for x in remainder {
                        add_color(*x, m.position(&sample_point(*x)));
                    }
                    //The running average of all samples so far, computed here on the thread of the row instead of after the pass on a single thread
                    for x in &xs {
                        let sum = row[*x];
                        color_row[*x] = TrueColor::new((sum[0] / divisor) as u8, (sum[1] / divisor) as u8, (sum[2] / divisor) as u8);
                    }
                    let done = xs.len();
                    pixels_done.fetch_add(done, Ordering::Relaxed);
                });
            }
        });
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        if let Some(adaptive) = adaptive {
            if sample == 0 {
                first_colors.clone_from(&colors);
            } else if sample + 1 == ADAPTIVE_UNIFORM_SAMPLES {
                mask = Some(adaptive_mask(width, height, region, &first_colors, &colors, adaptive.threshold));
            }
        }
        on_pass(&colors, sample + 1);
    }
    true
}

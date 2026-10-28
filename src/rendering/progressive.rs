//! The progressive renderer: one supersampling layer at a time, so every layer can be shown as soon as it is computed.

use std::{thread, sync::{Mutex, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}}, time::{Duration, Instant}};

use super::adaptive::{adaptive_mask, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
use super::jitter::sample_jitter;
use super::region::ExposedRegion;
use super::render_thread_count;
use crate::{complex::Complex, complex_plane::ComplexPlane, coloring::TrueColor, mandelbrot_set::{MandelbrotSet, LANES}};

/// The largest `max_iterations` for which `Render::run` precomputes the color of every iteration count
const MAX_COLOR_TABLE_ITERATIONS: u32 = 200_000;

/// The block sizes of the preview levels, coarse to fine, see [`Render::preview`]. The levels are always reported in this order, starting with the first; a render can stop reporting after any level
pub const PREVIEW_BLOCKS: [usize; 3] = [8, 4, 2];
/// Only pixels in the rows and columns that are multiples of this get a preview sample
const SMALLEST_BLOCK: usize = PREVIEW_BLOCKS[PREVIEW_BLOCKS.len() - 1];

/// A render of the complex plane: what to render and how, set step by step and started with [`Render::run`].
///
/// The render goes one supersampling layer (pass) at a time, so intermediate images can be shown early. Pass `k` computes sample `k` of every pixel in the region, with the deterministic jitter
/// of `sample_jitter`, so identical settings always give identical colors, whatever the number of threads. The render is multithreaded; `on_pass` is called on the calling thread.
///
/// ```ignore
/// let completed = Render::new(&plane, &set, width, height, coloring_function)
///     .supersampling(4)
///     .adaptive(Some(AdaptiveSampling::default()))
///     .cancel_with(&cancel)
///     .run(|colors, pass| { /* show the colors of the pass */ });
/// ```
pub struct Render<'a> {
    plane: &'a ComplexPlane,
    set: &'a MandelbrotSet,
    width: usize,
    height: usize,
    coloring_function: fn(u32, u32) -> TrueColor,
    supersampling_amount: u8,
    adaptive: Option<AdaptiveSampling>,
    exposed: Option<(&'a ExposedRegion, &'a [TrueColor])>,
    cancel: Option<&'a AtomicBool>,
    pixels_done: Option<&'a AtomicUsize>,
    preview: Option<Duration>,
}

impl<'a> Render<'a> {
    /// A render of the whole `width` x `height` image of `plane`, with one sample per pixel, colored by `coloring_function` (iterations and maximum iterations to a color)
    pub fn new(plane: &'a ComplexPlane, set: &'a MandelbrotSet, width: usize, height: usize, coloring_function: fn(u32, u32) -> TrueColor) -> Self {
        Render { plane, set, width, height, coloring_function, supersampling_amount: 1, adaptive: None, exposed: None, cancel: None, pixels_done: None, preview: None }
    }

    /// Takes `amount` samples per pixel (limited to 1 to 64), each at its own deterministic jitter inside the pixel
    #[must_use]
    pub fn supersampling(mut self, amount: u8) -> Self {
        self.supersampling_amount = amount;
        self
    }

    /// With `Some`, the samples after the first `ADAPTIVE_UNIFORM_SAMPLES` are only taken where the picture shows detail.
    /// This is a quality/speed heuristic, not equivalent to full supersampling: a feature that neither of the first two samples of a pixel or its neighbors shows can be missed.
    /// With `None`, or at most `ADAPTIVE_UNIFORM_SAMPLES` samples, every pixel gets every sample. The pass numbering is the same either way (one pass per sample); the later adaptive passes are just cheaper.
    #[must_use]
    pub fn adaptive(mut self, adaptive: Option<AdaptiveSampling>) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Only renders the pixels inside `region`; all other pixels keep the colors of `base` (`width * height` colors),
    /// for example the shifted image of a previous render, so moving the view only renders the newly exposed pixels
    #[must_use]
    pub fn only(mut self, region: &'a ExposedRegion, base: &'a [TrueColor]) -> Self {
        self.exposed = Some((region, base));
        self
    }

    /// Stops the render when `cancel` is set; the last completed pass has then been reported and a partially computed pass is discarded
    #[must_use]
    pub fn cancel_with(mut self, cancel: &'a AtomicBool) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Counts the finished pixels of the current pass in `pixels_done` (reset to 0 at the start of every pass), so another thread can show how far the pass is
    #[must_use]
    pub fn count_pixels_in(mut self, pixels_done: &'a AtomicUsize) -> Self {
        self.pixels_done = Some(pixels_done);
        self
    }

    /// With `Some(worthwhile)`, quick previews are shown before the first layer: one sample for every block of 8 x 8 pixels, then 4 x 4, then 2 x 2, each block filled with
    /// the color of its top-left pixel. They are reported to `on_pass` with pass number 0.
    ///
    /// A preview sample is the first sample of its pixel (the same point, with the same jitter), so the first layer does not compute it again: the previews cost
    /// filling the blocks and reporting them, not extra iterations, and the final image is identical with or without them.
    /// A level is only reported if the rest of the first layer is expected to take longer than `worthwhile` (estimated from the samples so far), so a fast view
    /// gets no previews and no extra reporting.
    #[must_use]
    pub fn preview(mut self, worthwhile: Option<Duration>) -> Self {
        self.preview = worthwhile;
        self
    }

    /// Runs the render. After every completed pass `on_pass` is called with the colors of the whole image (`width * height`, row by row), that is `base` with the region replaced
    /// by the running average of all samples so far, and the number of the pass (from 1).
    /// Returns `false` if the render was cancelled before all passes completed.
    ///
    /// # Panics
    /// If `base` does not have `width * height` colors, or if a thread of the render panics.
    pub fn run(self, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
        let Render { plane: c, set: m, width, height, coloring_function, supersampling_amount, adaptive, exposed, cancel, pixels_done, preview } = self;
        let full_region = ExposedRegion::full(width, height);
        let black = vec![TrueColor::new(0, 0, 0); width * height];
        let (region, base) = exposed.unwrap_or((&full_region, &black));
        let (never_cancelled, own_counter) = (AtomicBool::new(false), AtomicUsize::new(0));
        let (cancel, pixels_done) = (cancel.unwrap_or(&never_cancelled), pixels_done.unwrap_or(&own_counter));
        let adaptive = adaptive.as_ref();
        assert_eq!(base.len(), width * height, "base should have a color for every pixel");
        let supersampling_amount = supersampling_amount.clamp(1, 64);
        let amount_of_threads = render_thread_count();
        let mut sums: Vec<[u32; 3]> = vec![[0; 3]; width * height];
        let mut colors: Vec<TrueColor> = base.to_vec();
        //The color of a pixel only depends on its iteration count, which is at most `max_iterations`, so compute every color once
        let color_table: Option<Vec<TrueColor>> = (m.max_iterations <= MAX_COLOR_TABLE_ITERATIONS).then(|| (0..=m.max_iterations).map(|iterations| m.color(coloring_function, iterations)).collect());
        let adaptive = adaptive.filter(|_| supersampling_amount > ADAPTIVE_UNIFORM_SAMPLES);
        let color_of = |position: f64| m.interpolated_color(position, |iterations| color_table.as_ref().map_or_else(|| m.color(coloring_function, iterations), |table| table[iterations as usize]));
        //The pixels whose first sample the previews computed already, empty without previews
        let mut has_first: Vec<bool> = Vec::new();
        let mut precomputed = 0;
        if let Some(worthwhile) = preview {
            match render_previews(c, m, width, region, worthwhile, cancel, &color_of, &mut sums, &mut colors, &mut on_pass) {
                Some((flags, count)) => (has_first, precomputed) = (flags, count),
                None => return false,
            }
        }
        let mut first_colors: Vec<TrueColor> = Vec::new();
        let mut mask: Option<Vec<bool>> = None;
        for sample in 0..supersampling_amount {
            let skip_first = sample == 0 && !has_first.is_empty();
            pixels_done.store(if skip_first { precomputed } else { 0 }, Ordering::Relaxed);
            let divisor = u32::from(sample) + 1;
            if let (Some(_), Some(mask)) = (adaptive, &mask) {
                //The pixels that get no sample in this pass count as done
                let skipped = (0..height).map(|y| (0..width).filter(|x| region.contains(*x, y) && !mask[y * width + x]).count()).sum();
                pixels_done.store(skipped, Ordering::Relaxed);
            }
            let mask_ref = mask.as_deref();
            let has_first = &has_first;
            let rows = Mutex::new(sums.chunks_mut(width).zip(colors.chunks_mut(width)).enumerate());
            thread::scope(|scope| {
                for _ in 0..amount_of_threads {
                    scope.spawn(|| loop {
                        if cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        let next_row = rows.lock().unwrap().next();
                        let Some((y, (row, color_row))) = next_row else { return };
                        let xs: Vec<usize> = (0..width).filter(|x| region.contains(*x, y) && mask_ref.is_none_or(|mask| mask[y * width + x]) && !(skip_first && has_first[y * width + x])).collect();
                        positions(c, m, y, &xs, sample, |x, position| {
                            let color = color_of(position);
                            let sum = &mut row[x];
                            sum[0] += u32::from(color.red);
                            sum[1] += u32::from(color.green);
                            sum[2] += u32::from(color.blue);
                        });
                        //The running average of all samples so far, computed here on the thread of the row instead of after the pass on a single thread
                        for x in &xs {
                            let sum = row[*x];
                            color_row[*x] = TrueColor::new((sum[0] / divisor) as u8, (sum[1] / divisor) as u8, (sum[2] / divisor) as u8);
                        }
                        if skip_first && y % SMALLEST_BLOCK == 0 {
                            //The first samples that the previews computed (top-left pixels of blocks): the color of the pixel itself, whether or not a preview showed it
                            for x in (0..width).step_by(SMALLEST_BLOCK).filter(|x| has_first[y * width + x]) {
                                color_row[x] = sum_color(row[x]);
                            }
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
    }

/// The color of the sum of one sample
fn sum_color(sum: [u32; 3]) -> TrueColor {
    TrueColor::new(sum[0] as u8, sum[1] as u8, sum[2] as u8)
}

/// Computes the position (smooth iteration count) of sample `sample` of the pixels `xs` of row `y` and gives each to `sink`: `LANES` pixels at the same time, the remaining pixels one by one
fn positions(c: &ComplexPlane, m: &MandelbrotSet, y: usize, xs: &[usize], sample: u8, mut sink: impl FnMut(usize, f64)) {
    let sample_point = |x: usize| {
        let (jitter_x, jitter_y) = sample_jitter(x, y, sample);
        c.complex_from_pixel_plane(f64::from(x as u32) + jitter_x, f64::from(y as u32) + jitter_y)
    };
    let (groups, remainder) = xs.as_chunks::<LANES>();
    for group in groups {
        let points: [Complex; LANES] = std::array::from_fn(|lane| sample_point(group[lane]));
        let positions = m.position_lanes(&points);
        for (lane, x) in group.iter().enumerate() {
            sink(*x, positions[lane]);
        }
    }
    for x in remainder {
        sink(*x, m.position(&sample_point(*x)));
    }
}

/// The previews of [`Render::preview`]: computes the first sample of the top-left pixel of every block of each level into `sums`, and reports a level (filled into `colors`)
/// when the rest of the first layer is expected to take longer than `worthwhile`. Returns which pixels have their first sample and how many, or `None` if cancelled.
#[allow(clippy::too_many_arguments)] // The state of `Render::run` that the previews share
fn render_previews(c: &ComplexPlane, m: &MandelbrotSet, width: usize, region: &ExposedRegion, worthwhile: Duration, cancel: &AtomicBool, color_of: &(impl Fn(f64) -> TrueColor + Sync),
    sums: &mut [[u32; 3]], colors: &mut [TrueColor], on_pass: &mut impl FnMut(&[TrueColor], u8)) -> Option<(Vec<bool>, usize)> {
    let height = sums.len() / width;
    let region_pixels = region.pixel_count(width, height);
    let mut has_first = vec![false; sums.len()];
    let mut precomputed = 0;
    let threads = render_thread_count();
    //The time the threads spent computing samples, without starting threads and waiting: on a fast view those would dominate the estimate
    let busy_nanos = AtomicU64::new(0);
    for block in PREVIEW_BLOCKS {
        let count = AtomicUsize::new(0);
        let rows = Mutex::new(sums.chunks_mut(width).zip(has_first.chunks_mut(width)).enumerate().step_by(block));
        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_row = rows.lock().unwrap().next();
                    let Some((y, (row, has_row))) = next_row else { return };
                    let row_started = Instant::now();
                    let xs: Vec<usize> = (0..width).step_by(block).filter(|x| region.contains(*x, y) && !has_row[*x]).collect();
                    positions(c, m, y, &xs, 0, |x, position| {
                        let color = color_of(position);
                        row[x] = [u32::from(color.red), u32::from(color.green), u32::from(color.blue)];
                        has_row[x] = true;
                    });
                    count.fetch_add(xs.len(), Ordering::Relaxed);
                    busy_nanos.fetch_add(u64::try_from(row_started.elapsed().as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
                });
            }
        });
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        precomputed += count.into_inner();
        //The rest of the first layer, at the speed of the samples so far, on all threads
        let per_sample = busy_nanos.load(Ordering::Relaxed) as f64 / precomputed.max(1) as f64;
        let remaining = Duration::from_secs_f64(per_sample * region_pixels.saturating_sub(precomputed) as f64 / threads as f64 / 1e9);
        if remaining <= worthwhile {
            break;
        }
        fill_blocks(width, region, block, sums, &has_first, colors);
        on_pass(colors, 0);
    }
    Some((has_first, precomputed))
}

/// Gives every pixel of `region` the color of the top-left pixel of its `block` x `block` block, if that pixel has its first sample
fn fill_blocks(width: usize, region: &ExposedRegion, block: usize, sums: &[[u32; 3]], has_first: &[bool], colors: &mut [TrueColor]) {
    let rows = Mutex::new(colors.chunks_mut(width).enumerate());
    thread::scope(|scope| {
        for _ in 0..render_thread_count() {
            scope.spawn(|| loop {
                let next_row = rows.lock().unwrap().next();
                let Some((y, row)) = next_row else { return };
                //The pixels of the region in this row are one run: the whole row, or the exposed columns
                let (start, end) = if region.rows.contains(&y) { (0, width) } else { (region.columns.start.min(width), region.columns.end.min(width)) };
                let top = (y - y % block) * width;
                let mut x = start;
                while x < end {
                    let block_end = (x - x % block + block).min(end);
                    let origin = top + x - x % block;
                    if has_first[origin] {
                        row[x..block_end].fill(sum_color(sums[origin]));
                    }
                    x = block_end;
                }
            });
        }
    });
}

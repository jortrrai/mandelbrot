//! The progressive renderer: one supersampling layer at a time, so every layer can be shown as soon as it is computed.

use std::{thread, sync::{Mutex, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}}, time::{Duration, Instant}};

use super::adaptive::{mask_where, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
use super::jitter::sample_jitter;
use super::region::ExposedRegion;
use super::render_thread_count;
use super::state::{SampleKey, SampleState, FOREIGN};
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
    state: Option<&'a mut SampleState>,
}

impl<'a> Render<'a> {
    /// A render of the whole `width` x `height` image of `plane`, with one sample per pixel, colored by `coloring_function` (iterations and maximum iterations to a color)
    pub fn new(plane: &'a ComplexPlane, set: &'a MandelbrotSet, width: usize, height: usize, coloring_function: fn(u32, u32) -> TrueColor) -> Self {
        Render { plane, set, width, height, coloring_function, supersampling_amount: 1, adaptive: None, exposed: None, cancel: None, pixels_done: None, preview: None, state: None }
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

    /// Continues the samples in `state` instead of starting from nothing: a pixel that already has samples only gets the samples it is missing.
    ///
    /// If the view is the view of the state moved by whole pixels, with the same size, iterations, palette and coloring, the samples are shifted along with the view
    /// and only the newly visible pixels start empty; otherwise every pixel starts empty. The state is updated while the render runs, also when it is cancelled
    /// (even within a pass), so the next render continues where this one stopped. The samples of a pixel are keyed by its position in the view of the first render
    /// (see `SampleState`), so continued samples are exactly the samples of a render that was never interrupted.
    ///
    /// A pixel that has more samples than `supersampling` keeps them. With adaptive supersampling the extra samples go to the pixels that show detail in the current
    /// view; a pixel near the edge that got extra samples because of a neighbor that has moved out of the view keeps them, so it can have more samples than in a fresh render.
    /// Not combined with `only`.
    #[must_use]
    pub fn resume(mut self, state: &'a mut SampleState) -> Self {
        self.state = Some(state);
        self
    }

    /// Runs the render. After every completed pass `on_pass` is called with the colors of the whole image (`width * height`, row by row), that is `base` with the region replaced
    /// by the running average of all samples so far, and the number of the pass (from 1).
    /// Returns `false` if the render was cancelled before all passes completed.
    ///
    /// # Panics
    /// If `base` does not have `width * height` colors, or if a thread of the render panics.
    pub fn run(self, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
        let Render { plane: c, set: m, width, height, coloring_function, supersampling_amount, adaptive, exposed, cancel, pixels_done, preview, state } = self;
        let (never_cancelled, own_counter) = (AtomicBool::new(false), AtomicUsize::new(0));
        let (cancel, pixels_done) = (cancel.unwrap_or(&never_cancelled), pixels_done.unwrap_or(&own_counter));
        let supersampling_amount = supersampling_amount.clamp(1, 64);
        let adaptive = adaptive.filter(|_| supersampling_amount > ADAPTIVE_UNIFORM_SAMPLES);
        let (region, base) = exposed.map_or((None, None), |(region, base)| (Some(region), Some(base)));
        if let Some(base) = base {
            assert_eq!(base.len(), width * height, "base should have a color for every pixel");
        }
        debug_assert!(state.is_none() || exposed.is_none(), "a resumed render renders every pixel that misses samples");
        let mut state = state;
        //Only a resumed state is kept after the render, so only then does every pixel need its count of samples updated
        let track = state.is_some();
        //How many pixels kept their samples; without them, the colors so far are the base and every pixel of the render is still empty
        let kept = state.as_mut().map_or(0, |state| state.prepare(c, width, height, SampleKey::new(m, coloring_function)));
        let mut own_state = None;
        let state: &mut SampleState = state.unwrap_or_else(|| own_state.insert(SampleState::for_one_render(width, height, region, base, adaptive.is_some())));
        let origin = state.origin;
        let mut colors: Vec<TrueColor> = base.map_or_else(|| vec![TrueColor::new(0, 0, 0); width * height], <[TrueColor]>::to_vec);
        //How many pixels of the render have each amount of samples, to know how many pixels a pass computes
        let mut pending = [0_usize; 66];
        if kept == 0 {
            pending[0] = region.map_or(width * height, |region| region.pixel_count(width, height));
        } else {
            for (index, count) in state.counts.iter().enumerate().filter(|(_, count)| **count != FOREIGN) {
                pending[usize::from(*count).min(65)] += 1;
                if *count > 0 {
                    colors[index] = average(state.sums[index], *count);
                }
            }
        }
        let total: usize = pending.iter().sum();
        //The color of a pixel only depends on its iteration count, which is at most `max_iterations`, so compute every color once
        let color_table: Option<Vec<TrueColor>> = (m.max_iterations <= MAX_COLOR_TABLE_ITERATIONS).then(|| (0..=m.max_iterations).map(|iterations| m.color(coloring_function, iterations)).collect());
        let color_of = |position: f64| m.interpolated_color(position, |iterations| color_table.as_ref().map_or_else(|| m.color(coloring_function, iterations), |table| table[iterations as usize]));
        //The pixels whose first sample the previews computed, empty without previews
        let mut has_first: Vec<bool> = Vec::new();
        if let Some(worthwhile) = preview.filter(|_| pending[0] > 0) {
            match render_previews(c, m, origin, worthwhile, cancel, &color_of, state, pending[0], &mut colors, &mut on_pass) {
                Some((flags, count)) => {
                    has_first = flags;
                    pending[0] -= count;
                    pending[1] += count;
                }
                None => return false,
            }
        }
        let mut mask: Option<Vec<bool>> = None;
        let mut masked = [0_usize; 66];
        for sample in 0..supersampling_amount {
            let index = usize::from(sample);
            if let Some(adaptive) = adaptive.filter(|_| sample == ADAPTIVE_UNIFORM_SAMPLES) {
                //Every pixel has its uniform samples now: mark the pixels that need more, from the colors of their first two samples
                let counts = &state.counts;
                let new_mask = mask_where(width, height, &|x, y| counts[y * width + x] != FOREIGN, &state.first, &state.second, adaptive.threshold);
                if track {
                    for (count, _) in counts.iter().zip(&new_mask).filter(|(_, marked)| **marked) {
                        masked[usize::from(*count).min(65)] += 1;
                    }
                } else {
                    //Without counts every pixel of the render has the uniform samples now
                    masked[index] = new_mask.iter().filter(|marked| **marked).count();
                }
                mask = Some(new_mask);
            }
            let eligible = if mask.is_some() { masked[index] } else { pending[index] };
            //The pixels that get no sample in this pass count as done
            pixels_done.store(total - eligible, Ordering::Relaxed);
            let processed = if eligible > 0 { sample_pass(c, m, sample, track, state, &mut colors, mask.as_deref(), &has_first, &color_of, cancel, pixels_done) } else { 0 };
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            pending[index] -= processed;
            pending[index + 1] += processed;
            if mask.is_some() {
                masked[index] -= processed;
                masked[index + 1] += processed;
            }
            on_pass(&colors, sample + 1);
        }
        true
    }
}

/// Pass `sample` of `Render::run`: takes sample `sample` of every pixel that has `sample` samples (and is marked by `mask`, if any), updates `state` and `colors`
/// row by row on all threads, and returns how many pixels it sampled. A cancelled pass stops after the rows it started; those rows are complete in `state`.
/// Without `track` (a state that is not kept) every pixel of the render has `sample` samples except the ones the previews computed, so the counts are not updated.
#[allow(clippy::too_many_arguments)] // The state of `Render::run` that a pass shares
fn sample_pass(c: &ComplexPlane, m: &MandelbrotSet, sample: u8, track: bool, state: &mut SampleState, colors: &mut [TrueColor], mask: Option<&[bool]>, has_first: &[bool], color_of: &(impl Fn(f64) -> TrueColor + Sync),
    cancel: &AtomicBool, pixels_done: &AtomicUsize) -> usize {
    let (width, height, origin) = (state.width, state.height, state.origin);
    let processed = AtomicUsize::new(0);
        let divisor = u32::from(sample) + 1;
        let mask_ref = mask;
        let skip_first = sample == 0 && !has_first.is_empty();
        //The colors of the first two samples are only kept when they are needed (adaptive supersampling, a kept state); otherwise every row gets an empty slice
        let (first_rows, second_rows) = (rows_or_empty(&mut state.first, width, height), rows_or_empty(&mut state.second, width, height));
        let rows = Mutex::new(state.sums.chunks_mut(width).zip(state.counts.chunks_mut(width)).zip(first_rows).zip(second_rows).zip(colors.chunks_mut(width)).enumerate());
        thread::scope(|scope| {
            for _ in 0..render_thread_count() {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_row = rows.lock().unwrap().next();
                    let Some((y, ((((row, counts), first), second), color_row))) = next_row else { return };
                    let xs: Vec<usize> = if track {
                        (0..width).filter(|x| counts[*x] == sample && mask_ref.is_none_or(|mask| mask[y * width + x])).collect()
                    } else {
                        (0..width).filter(|x| counts[*x] != FOREIGN && mask_ref.is_none_or(|mask| mask[y * width + x]) && !(skip_first && has_first[y * width + x])).collect()
                    };
                    positions(c, m, y, &xs, sample, origin, |x, position| {
                        let color = color_of(position);
                        let sum = &mut row[x];
                        sum[0] += u32::from(color.red);
                        sum[1] += u32::from(color.green);
                        sum[2] += u32::from(color.blue);
                    });
                    //The running average of all samples so far, computed here on the thread of the row instead of after the pass on a single thread
                    for x in &xs {
                        let sum = row[*x];
                        let color = TrueColor::new((sum[0] / divisor) as u8, (sum[1] / divisor) as u8, (sum[2] / divisor) as u8);
                        color_row[*x] = color;
                        if track {
                            counts[*x] = sample + 1;
                        }
                        match sample {
                            0 if !first.is_empty() => first[*x] = color,
                            1 if !second.is_empty() => second[*x] = color,
                            _ => {}
                        }
                    }
                    if skip_first && y % SMALLEST_BLOCK == 0 {
                        //The first samples that the previews computed (top-left pixels of blocks): the color of the pixel itself, whether or not a preview showed it
                        for x in (0..width).step_by(SMALLEST_BLOCK).filter(|x| has_first[y * width + x]) {
                            color_row[x] = sum_color(row[x]);
                        }
                    }
                    processed.fetch_add(xs.len(), Ordering::Relaxed);
                    pixels_done.fetch_add(xs.len(), Ordering::Relaxed);
                });
            }
        });
    processed.into_inner()
}

/// The color of the sum of one sample
fn sum_color(sum: [u32; 3]) -> TrueColor {
    TrueColor::new(sum[0] as u8, sum[1] as u8, sum[2] as u8)
}

/// The rows of a `width` x `height` grid, or `height` empty rows if the grid is not kept (empty)
fn rows_or_empty<T>(grid: &mut [T], width: usize, height: usize) -> Vec<&mut [T]> {
    if grid.is_empty() { (0..height).map(|_| <&mut [T]>::default()).collect() } else { grid.chunks_mut(width).collect() }
}

/// The average color of `count` samples whose colors add up to `sum`
fn average(sum: [u32; 3], count: u8) -> TrueColor {
    let count = u32::from(count);
    TrueColor::new((sum[0] / count) as u8, (sum[1] / count) as u8, (sum[2] / count) as u8)
}

/// Computes the position (smooth iteration count) of sample `sample` of the pixels `xs` of row `y` and gives each to `sink`: `LANES` pixels at the same time, the remaining pixels one by one.
/// The jitter of a pixel is keyed by its position plus `origin`, see `SampleState`.
fn positions(c: &ComplexPlane, m: &MandelbrotSet, y: usize, xs: &[usize], sample: u8, origin: (isize, isize), mut sink: impl FnMut(usize, f64)) {
    let key_y = y.wrapping_add_signed(origin.1);
    let sample_point = |x: usize| {
        let (jitter_x, jitter_y) = sample_jitter(x.wrapping_add_signed(origin.0), key_y, sample);
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

/// The previews of [`Render::preview`]: computes the first sample of the top-left pixel of every block of each level that has no samples yet into `state`, and reports
/// a level (filled into `colors`) when the rest of the first layer (`empty` pixels without samples at the start) is expected to take longer than `worthwhile`.
/// Returns which pixels got their first sample and how many, or `None` if cancelled.
#[allow(clippy::too_many_arguments)] // The state of `Render::run` that the previews share
fn render_previews(c: &ComplexPlane, m: &MandelbrotSet, origin: (isize, isize), worthwhile: Duration, cancel: &AtomicBool, color_of: &(impl Fn(f64) -> TrueColor + Sync),
    state: &mut SampleState, empty: usize, colors: &mut [TrueColor], on_pass: &mut impl FnMut(&[TrueColor], u8)) -> Option<(Vec<bool>, usize)> {
    let width = state.width;
    let mut has_first = vec![false; state.counts.len()];
    let mut precomputed = 0;
    let threads = render_thread_count();
    //The time the threads spent computing samples, without starting threads and waiting: on a fast view those would dominate the estimate
    let busy_nanos = AtomicU64::new(0);
    for block in PREVIEW_BLOCKS {
        let count = AtomicUsize::new(0);
        let rows = Mutex::new(state.sums.chunks_mut(width).zip(state.counts.chunks_mut(width)).zip(rows_or_empty(&mut state.first, width, state.height)).zip(has_first.chunks_mut(width)).enumerate().step_by(block));
        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_row = rows.lock().unwrap().next();
                    let Some((y, (((row, counts), first), has_row))) = next_row else { return };
                    let row_started = Instant::now();
                    let xs: Vec<usize> = (0..width).step_by(block).filter(|x| counts[*x] == 0).collect();
                    positions(c, m, y, &xs, 0, origin, |x, position| {
                        let color = color_of(position);
                        row[x] = [u32::from(color.red), u32::from(color.green), u32::from(color.blue)];
                        counts[x] = 1;
                        if !first.is_empty() {
                            first[x] = color;
                        }
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
        let remaining = Duration::from_secs_f64(per_sample * empty.saturating_sub(precomputed) as f64 / threads as f64 / 1e9);
        if remaining <= worthwhile {
            break;
        }
        fill_blocks(width, block, state, &has_first, colors);
        on_pass(colors, 0);
    }
    Some((has_first, precomputed))
}

/// Gives every pixel without samples the color of the top-left pixel of its `block` x `block` block, if the previews computed that pixel
fn fill_blocks(width: usize, block: usize, state: &SampleState, has_first: &[bool], colors: &mut [TrueColor]) {
    let rows = Mutex::new(colors.chunks_mut(width).enumerate());
    thread::scope(|scope| {
        for _ in 0..render_thread_count() {
            scope.spawn(|| loop {
                let next_row = rows.lock().unwrap().next();
                let Some((y, row)) = next_row else { return };
                let top = (y - y % block) * width;
                for (x, color) in row.iter_mut().enumerate() {
                    let (index, origin) = (y * width + x, top + x - x % block);
                    if has_first[origin] && (state.counts[index] == 0 || has_first[index]) {
                        *color = sum_color(state.sums[origin]);
                    }
                }
            });
        }
    });
}

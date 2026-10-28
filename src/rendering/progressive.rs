//! The progressive renderer: one supersampling layer at a time, so every layer can be shown as soon as it is computed.

use std::{thread, sync::{Mutex, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}}, time::{Duration, Instant}};

use super::adaptive::{mask_where, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
use super::jitter::sample_jitter;
use super::region::ExposedRegion;
use super::render_thread_count;
use super::gpu::GpuIterator;
use super::state::{SampleKey, SampleState, FOREIGN};
use crate::{complex::Complex, complex_plane::ComplexPlane, coloring::TrueColor, mandelbrot_set::{MandelbrotSet, LANES}};

/// The largest `max_iterations` for which `Render::run` precomputes the color of every iteration count
const MAX_COLOR_TABLE_ITERATIONS: u32 = 200_000;

/// The block sizes of the preview levels, coarse to fine, see [`Render::preview`]. The levels are always reported in this order, starting with the first; a render can stop reporting after any level
pub const PREVIEW_BLOCKS: [usize; 3] = [8, 4, 2];

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
    gpu: Option<(&'a GpuIterator, Duration)>,
    gpu_samples: Option<&'a AtomicUsize>,
}

impl<'a> Render<'a> {
    /// A render of the whole `width` x `height` image of `plane`, with one sample per pixel, colored by `coloring_function` (iterations and maximum iterations to a color)
    pub fn new(plane: &'a ComplexPlane, set: &'a MandelbrotSet, width: usize, height: usize, coloring_function: fn(u32, u32) -> TrueColor) -> Self {
        Render { plane, set, width, height, coloring_function, supersampling_amount: 1, adaptive: None, exposed: None, cancel: None, pixels_done: None, preview: None, state: None, gpu: None, gpu_samples: None }
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

    /// With `Some((gpu, join_after))`, the GPU helps with every pass that has run for `join_after` (see `sample_pass`): it computes bit for bit the same samples
    /// (see `gpu`), so the image is the same; a pass that is done before `join_after` never waits for the GPU.
    #[must_use]
    pub fn gpu(mut self, gpu: Option<(&'a GpuIterator, Duration)>) -> Self {
        self.gpu = gpu;
        self
    }

    /// Adds the number of samples that the GPU computed to `gpu_samples`
    #[must_use]
    pub fn count_gpu_samples_in(mut self, gpu_samples: &'a AtomicUsize) -> Self {
        self.gpu_samples = Some(gpu_samples);
        self
    }

    /// Runs the render. After every completed pass `on_pass` is called with the colors of the whole image (`width * height`, row by row), that is `base` with the region replaced
    /// by the running average of all samples so far, and the number of the pass (from 1).
    /// Returns `false` if the render was cancelled before all passes completed.
    ///
    /// # Panics
    /// If `base` does not have `width * height` colors, or if a thread of the render panics.
    pub fn run(self, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
        let Render { plane: c, set: m, width, height, coloring_function, supersampling_amount, adaptive, exposed, cancel, pixels_done, preview, state, gpu, gpu_samples } = self;
        let own_gpu_samples = AtomicUsize::new(0);
        let gpu_samples = gpu_samples.unwrap_or(&own_gpu_samples);
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
        //Whether every pixel is still empty: a new state, or one that started over
        let empty = state.as_mut().is_none_or(|state| state.prepare(c, width, height, SampleKey::new(m, coloring_function)));
        let mut own_state = None;
        let state: &mut SampleState = state.unwrap_or_else(|| own_state.insert(SampleState::for_one_render(width, height, region, base, adaptive.is_some())));
        //Where the samples are placed: the view where the samples of the state started, or this view
        let (anchor, origin) = (state.anchor.clone().unwrap_or_else(|| c.clone()), state.origin);
        let mut colors: Vec<TrueColor> = base.map_or_else(|| vec![TrueColor::new(0, 0, 0); width * height], <[TrueColor]>::to_vec);
        //How many pixels of the render have each amount of samples, to know how many pixels a pass computes
        let mut pending = [0_usize; 66];
        if empty {
            pending[0] = region.map_or(width * height, |region| region.pixel_count(width, height));
        } else {
            pending = colors_of_kept_samples(state, &mut colors);
        }
        let total: usize = pending.iter().sum();
        //The color of a pixel only depends on its iteration count, which is at most `max_iterations`, so compute every color once
        let color_table: Option<Vec<TrueColor>> = (m.max_iterations <= MAX_COLOR_TABLE_ITERATIONS).then(|| color_table(m, coloring_function));
        let color_of = |position: f64| m.interpolated_color(position, |iterations| color_table.as_ref().map_or_else(|| m.color(coloring_function, iterations), |table| table[iterations as usize]));
        //The pixels whose first sample the previews computed, empty without previews
        let mut has_first: Vec<bool> = Vec::new();
        if let Some(worthwhile) = preview.filter(|_| pending[0] > 0) {
            match render_previews(&anchor, m, origin, worthwhile, cancel, &color_of, state, pending[0], &mut colors, &mut on_pass) {
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
        let mut time_per_sample: Option<Duration> = None;
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
            //The GPU only for a pass that is expected to be slow, from the time per sample of the pass before (unknown for the first pass: then the join delay decides)
            let expected = time_per_sample.map(|per_sample| per_sample * u32::try_from(eligible).unwrap_or(u32::MAX));
            let pass_gpu = gpu.filter(|_| expected.is_none_or(|expected| expected > GPU_WORTHWHILE)).map(|(gpu, join_after)| (gpu, join_after, gpu_samples));
            let pass_started = Instant::now();
            let processed = if eligible > 0 { sample_pass(&anchor, m, sample, track, state, &mut colors, mask.as_deref(), &has_first, &color_of, cancel, pixels_done, pass_gpu) } else { 0 };
            if processed > 0 {
                time_per_sample = Some(pass_started.elapsed() / u32::try_from(processed).unwrap_or(u32::MAX));
            }
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

/// Pass `sample` of `Render::run`: takes sample `sample` (placed in `anchor`, see `positions`) of every pixel that has `sample` samples (and is marked by `mask`, if
/// any), updates `state` and `colors` row by row on all threads, and returns how many pixels it sampled. A cancelled pass stops in the middle of a row; the pixels it
/// sampled are complete in `state`. Without `track` (a state that is not kept) every pixel of the render has `sample` samples except the ones the previews computed,
/// so the counts are not updated.
///
/// With `gpu`, the GPU helps once the pass has run for its join delay (a fast pass never waits for it): it takes `GPU_ROWS` rows at a time from the same queue as
/// the CPU threads, so the work divides itself by speed. The GPU computes the same positions bit for bit (see `gpu`), and the same code applies them to the rows.
/// If the GPU fails, its rows are computed on the CPU and the GPU is not used for the rest of the pass.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // The state of `Render::run` that a pass shares; the CPU threads and the GPU worker share its closures
fn sample_pass(c: &ComplexPlane, m: &MandelbrotSet, sample: u8, track: bool, state: &mut SampleState, colors: &mut [TrueColor], mask: Option<&[bool]>, has_first: &[bool], color_of: &(impl Fn(f64) -> TrueColor + Sync),
    cancel: &AtomicBool, pixels_done: &AtomicUsize, gpu: Option<(&GpuIterator, Duration, &AtomicUsize)>) -> usize {
    let (width, height, origin) = (state.width, state.height, state.origin);
    let started = Instant::now();
    let processed = AtomicUsize::new(0);
    let divisor = u32::from(sample) + 1;
    let skip_first = sample == 0 && !has_first.is_empty();
    //The pixels of row `y` that get this sample
    let select = |y: usize, counts: &[u8]| -> Vec<usize> {
        if track {
            (0..width).filter(|x| counts[*x] == sample && mask.is_none_or(|mask| mask[y * width + x])).collect()
        } else {
            (0..width).filter(|x| counts[*x] != FOREIGN && mask.is_none_or(|mask| mask[y * width + x]) && !(skip_first && has_first[y * width + x])).collect()
        }
    };
    //Adds the samples at `positions` of the pixels `xs` of a row, and updates its running averages and counts: the same for the CPU threads and the GPU
    let apply = |y: usize, row: &mut [[u32; 3]], counts: &mut [u8], first: &mut [TrueColor], second: &mut [TrueColor], color_row: &mut [TrueColor], xs: &[usize], positions: &[f64]| {
        for (x, position) in xs.iter().zip(positions) {
            let color = color_of(*position);
            let sum = &mut row[*x];
            if sample == 0 {
                //The first sample replaces the sum, so a state that starts over does not have to clear its sums
                *sum = [u32::from(color.red), u32::from(color.green), u32::from(color.blue)];
            } else {
                sum[0] += u32::from(color.red);
                sum[1] += u32::from(color.green);
                sum[2] += u32::from(color.blue);
            }
        }
        let xs = &xs[..positions.len()];
        //The running average of all samples so far, computed here on the thread of the row instead of after the pass on a single thread
        for x in xs {
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
        if skip_first {
            //The first samples that the previews computed (the representatives of blocks): the color of the pixel itself, whether or not a preview showed it
            for x in (0..width).filter(|x| has_first[y * width + x]) {
                color_row[x] = sum_color(row[x]);
            }
        }
        processed.fetch_add(xs.len(), Ordering::Relaxed);
        pixels_done.fetch_add(xs.len(), Ordering::Relaxed);
    };
    //The positions of the pixels `xs` of row `y` on the CPU; a cancelled render stops in the middle of the row
    let on_cpu = |y: usize, xs: &[usize]| {
        let mut out = Vec::with_capacity(xs.len());
        positions(c, m, y, xs, sample, origin, Some(cancel), |_, position| out.push(position));
        out
    };
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
                let xs = select(y, counts);
                let positions = on_cpu(y, &xs);
                apply(y, row, counts, first, second, color_row, &xs, &positions);
            });
        }
        if let Some((gpu, join_after, gpu_samples)) = gpu.as_ref() {
            scope.spawn(|| {
                //A fast pass is done before the GPU joins: then it costs nothing
                while started.elapsed() < *join_after {
                    if cancel.load(Ordering::Relaxed) || rows.lock().unwrap().len() == 0 {
                        return;
                    }
                    thread::sleep(Duration::from_micros(200));
                }
                let mut gpu_works = true;
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let batch: Vec<_> = rows.lock().unwrap().by_ref().take(GPU_ROWS).collect();
                    if batch.is_empty() {
                        return;
                    }
                    let selected: Vec<Vec<usize>> = batch.iter().map(|(y, ((((_, counts), _), _), _))| select(*y, counts)).collect();
                    let points: Vec<Complex> = batch.iter().zip(&selected).flat_map(|((y, _), xs)| xs.iter().map(move |x| sample_point(c, *x, *y, sample, origin))).collect();
                    let from_gpu = if gpu_works { gpu.positions(m, &points).ok() } else { None };
                    gpu_works = from_gpu.is_some();
                    if gpu_works {
                        gpu_samples.fetch_add(points.len(), Ordering::Relaxed);
                    }
                    let mut offset = 0;
                    for ((y, ((((row, counts), first), second), color_row)), xs) in batch.into_iter().zip(&selected) {
                        let positions = from_gpu.as_ref().map_or_else(|| on_cpu(y, xs), |all| all[offset..offset + xs.len()].to_vec());
                        offset += xs.len();
                        apply(y, row, counts, first, second, color_row, xs, &positions);
                    }
                }
            });
        }
    });
    processed.into_inner()
}

/// A pass is only given to the GPU when the pass before suggests it takes longer than this: starting the GPU worker for a short pass costs more than it saves
/// (measured: the default view at 32x adaptive, with many passes of a few milliseconds, took 90.5 ms with the GPU in every pass instead of 86.5 ms)
const GPU_WORTHWHILE: Duration = Duration::from_millis(12);

/// The rows the GPU takes from the queue of a pass at a time (about 38,000 pixels at 1200 x 800): enough to keep it busy, few enough that the CPU threads do not
/// wait long for its last batch
const GPU_ROWS: usize = 32;

/// Large color tables are built on all render threads. Starting the threads costs about 0.17 ms, so below this many entries one thread is as fast or faster
/// (measured: 20,000 Bernstein entries take 0.08 ms on one thread and 0.18 ms on all; at 65,376 entries both take about 0.2 ms)
const PARALLEL_COLOR_TABLE_ENTRIES: usize = 50_000;

/// The color of every iteration count from 0 to `max_iterations` of `m`. Every entry is independent of the others, so the table is the same however it is split.
/// Measured on 16 threads, one thread and all threads: 65,376 HSV entries 0.75 and 0.23 ms; 200,000 entries 0.64 and 0.24 ms (Bernstein), 2.45 and 0.45 ms (HSV).
pub(super) fn color_table(m: &MandelbrotSet, coloring_function: fn(u32, u32) -> TrueColor) -> Vec<TrueColor> {
    let entries = m.max_iterations as usize + 1;
    if entries < PARALLEL_COLOR_TABLE_ENTRIES {
        return (0..=m.max_iterations).map(|iterations| m.color(coloring_function, iterations)).collect();
    }
    let mut table = vec![TrueColor::new(0, 0, 0); entries];
    let chunk = entries.div_ceil(render_thread_count());
    thread::scope(|scope| {
        for (index, part) in table.chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                for (offset, color) in part.iter_mut().enumerate() {
                    *color = m.color(coloring_function, (index * chunk + offset) as u32);
                }
            });
        }
    });
    table
}

/// The color of the sum of one sample
fn sum_color(sum: [u32; 3]) -> TrueColor {
    TrueColor::new(sum[0] as u8, sum[1] as u8, sum[2] as u8)
}

/// Gives every pixel of `state` that has samples the average of its samples in `colors`, on all threads, and returns how many pixels of the render have each amount of samples
fn colors_of_kept_samples(state: &SampleState, colors: &mut [TrueColor]) -> [usize; 66] {
    let rows_per_chunk = state.height.div_ceil(render_thread_count()).max(1) * state.width;
    let pending = Mutex::new([0_usize; 66]);
    thread::scope(|scope| {
        for (chunk, colors) in colors.chunks_mut(rows_per_chunk).enumerate() {
            let pending = &pending;
            scope.spawn(move || {
                let start = chunk * rows_per_chunk;
                let mut counted = [0_usize; 66];
                for (offset, color) in colors.iter_mut().enumerate() {
                    let count = state.counts[start + offset];
                    if count != FOREIGN {
                        counted[usize::from(count).min(65)] += 1;
                        if count > 0 {
                            *color = average(state.sums[start + offset], count);
                        }
                    }
                }
                let mut pending = pending.lock().unwrap();
                for (total, count) in pending.iter_mut().zip(counted) {
                    *total += count;
                }
            });
        }
    });
    pending.into_inner().unwrap()
}

/// The chunks of `chunk` cells of a grid, or `count` empty chunks if the grid is not kept (empty)
fn chunks_or_empty<T>(grid: &mut [T], chunk: usize, count: usize) -> Vec<&mut [T]> {
    if grid.is_empty() { (0..count).map(|_| <&mut [T]>::default()).collect() } else { grid.chunks_mut(chunk).collect() }
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
/// A pixel is pixel `(x + origin.0, y + origin.1)` of `anchor`, which places its samples and keys their jitter, see `SampleState`.
///
/// With `cancel`, it stops between groups once `cancel` is set, and returns how many of `xs` (from the start) it gave to `sink`; without, it does all of them.
/// The point of sample `sample` of pixel `(x, y)`: pixel `(x + origin.0, y + origin.1)` of `anchor`, at the jitter of that pixel and sample (see `SampleState`)
fn sample_point(anchor: &ComplexPlane, x: usize, y: usize, sample: u8, origin: (isize, isize)) -> Complex {
    let (anchor_x, anchor_y) = (x.cast_signed() + origin.0, y.cast_signed() + origin.1);
    let (jitter_x, jitter_y) = sample_jitter(anchor_x as usize, anchor_y as usize, sample);
    anchor.complex_from_pixel_plane(anchor_x as f64 + jitter_x, anchor_y as f64 + jitter_y)
}

#[allow(clippy::too_many_arguments)] // The row, the sample and where to place it, and the cancel flag: the inner loop of every pass, kept as one function
fn positions(anchor: &ComplexPlane, m: &MandelbrotSet, y: usize, xs: &[usize], sample: u8, origin: (isize, isize), cancel: Option<&AtomicBool>, mut sink: impl FnMut(usize, f64)) -> usize {
    let sample_point = |x: usize| sample_point(anchor, x, y, sample, origin);
    let (groups, remainder) = xs.as_chunks::<LANES>();
    for (index, group) in groups.iter().enumerate() {
        if cancel.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
            return index * LANES;
        }
        let points: [Complex; LANES] = std::array::from_fn(|lane| sample_point(group[lane]));
        let positions = m.position_lanes(&points);
        for (lane, x) in group.iter().enumerate() {
            sink(*x, positions[lane]);
        }
    }
    for x in remainder {
        sink(*x, m.position(&sample_point(*x)));
    }
    xs.len()
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
    let height = state.height;
    for block in PREVIEW_BLOCKS {
        let count = AtomicUsize::new(0);
        //Every block has a representative: the pixel whose first sample it shows. That is its top-left pixel (always, in a render from nothing), unless that pixel
        //has samples from before (it was kept by a move): then a pixel of the block that a coarser level computed, or else its first pixel without samples. So a block
        //that has pixels without samples always shows a sample of one of them, and a preview of a strip has no gaps.
        let (across, bands) = (width.div_ceil(block), height.div_ceil(block));
        let mut representatives: Vec<Option<usize>> = vec![None; across * bands];
        let band = width * block;
        let first_bands = chunks_or_empty(&mut state.first, band, bands);
        let rows = Mutex::new(state.sums.chunks_mut(band).zip(state.counts.chunks_mut(band)).zip(first_bands).zip(has_first.chunks_mut(band)).zip(representatives.chunks_mut(across)).enumerate());
        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_band = rows.lock().unwrap().next();
                    let Some((by, ((((sums, counts), first), has), representatives))) = next_band else { return };
                    let started = Instant::now();
                    let rows_in_band = counts.len() / width;
                    let mut to_sample: Vec<usize> = Vec::new();
                    for (bx, representative) in representatives.iter_mut().enumerate() {
                        let (left, right) = (bx * block, (bx * block + block).min(width));
                        let mut pixels = (0..rows_in_band).flat_map(|row| (left..right).map(move |x| row * width + x));
                        let chosen = if has[left] || counts[left] == 0 { Some(left) } else { pixels.clone().find(|index| has[*index]).or_else(|| pixels.find(|index| counts[*index] == 0)) };
                        if let Some(index) = chosen {
                            *representative = Some(by * band + index);
                            if counts[index] == 0 {
                                to_sample.push(index);
                            }
                        }
                    }
                    let mut sampled = 0;
                    for row in 0..rows_in_band {
                        let xs: Vec<usize> = to_sample.iter().filter(|index| **index / width == row).map(|index| index % width).collect();
                        if xs.is_empty() {
                            continue;
                        }
                        //Stops in the middle of the band when cancelled: the representatives it sampled are complete
                        let done = positions(c, m, by * block + row, &xs, 0, origin, Some(cancel), |x, position| {
                            let (index, color) = (row * width + x, color_of(position));
                            sums[index] = [u32::from(color.red), u32::from(color.green), u32::from(color.blue)];
                            counts[index] = 1;
                            if !first.is_empty() {
                                first[index] = color;
                            }
                            has[index] = true;
                        });
                        sampled += done;
                        if done < xs.len() {
                            break;
                        }
                    }
                    count.fetch_add(sampled, Ordering::Relaxed);
                    busy_nanos.fetch_add(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
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
        fill_blocks(width, block, state, &has_first, &representatives, colors);
        on_pass(colors, 0);
    }
    Some((has_first, precomputed))
}

/// Gives every pixel without samples (and every pixel the previews computed) the color of the representative of its `block` x `block` block, see `render_previews`
fn fill_blocks(width: usize, block: usize, state: &SampleState, has_first: &[bool], representatives: &[Option<usize>], colors: &mut [TrueColor]) {
    let across = width.div_ceil(block);
    let rows = Mutex::new(colors.chunks_mut(width).enumerate());
    thread::scope(|scope| {
        for _ in 0..render_thread_count() {
            scope.spawn(|| loop {
                let next_row = rows.lock().unwrap().next();
                let Some((y, row)) = next_row else { return };
                let band = (y / block) * across;
                for (x, color) in row.iter_mut().enumerate() {
                    let index = y * width + x;
                    if state.counts[index] == 0 || has_first[index] {
                        if let Some(representative) = representatives[band + x / block] {
                            *color = sum_color(state.sums[representative]);
                        }
                    }
                }
            });
        }
    });
}

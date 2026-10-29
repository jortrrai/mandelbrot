//! The progressive renderer: one supersampling layer at a time, so every layer can be shown as soon as it is computed.

use std::{borrow::Cow, collections::VecDeque, thread, sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}}, time::{Duration, Instant}};

use super::adaptive::{mask_where, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
use super::jitter::sample_jitter;
use super::region::ExposedRegion;
use super::render_thread_count;
use super::gpu::{packed_pixel, FastOutcome, GpuIterator, Outcome, BATCH, MAX_PIXEL};
use super::state::{Prepared, SampleKey, SampleState, FOREIGN};
use crate::{complex::Complex, complex_plane::ComplexPlane, coloring::TrueColor, mandelbrot_set::MandelbrotSet};

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
    coloring_function: fn(f64, f64) -> TrueColor,
    supersampling_amount: u8,
    adaptive: Option<AdaptiveSampling>,
    exposed: Option<(&'a ExposedRegion, &'a [TrueColor])>,
    cancel: Option<&'a AtomicBool>,
    pixels_done: Option<&'a AtomicUsize>,
    preview: Option<Duration>,
    /// The block sizes of the previews, coarsest first, see `preview_blocks`
    preview_blocks: &'a [usize],
    state: Option<&'a mut SampleState>,
    gpu: Option<(&'a GpuIterator, Duration)>,
    gpu_samples: Option<&'a AtomicUsize>,
    /// Whether the GPU uses its fast, approximate arithmetic where the view allows it, see `fast_gpu`
    fast_gpu: bool,
    /// How many CPU threads take rows of a pass (all render threads; none in a test where the GPU computes every sample)
    cpu_threads: usize,
    /// Whether the render keeps its samples (see `colors_only`)
    keep_samples: bool,
}

impl<'a> Render<'a> {
    /// A render of the whole `width` x `height` image of `plane`, with one sample per pixel, colored by `coloring_function` (iterations and maximum iterations to a color)
    pub fn new(plane: &'a ComplexPlane, set: &'a MandelbrotSet, width: usize, height: usize, coloring_function: fn(f64, f64) -> TrueColor) -> Self {
        Render { plane, set, width, height, coloring_function, supersampling_amount: 1, adaptive: None, exposed: None, cancel: None, pixels_done: None, preview: None, preview_blocks: &PREVIEW_BLOCKS, state: None, gpu: None, gpu_samples: None, fast_gpu: false, cpu_threads: render_thread_count(), keep_samples: true }
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

    /// Other block sizes for the previews than `PREVIEW_BLOCKS`, coarsest first (each at least 2): for example only the size that a flight waits for
    #[must_use]
    pub const fn preview_blocks(mut self, blocks: &'a [usize]) -> Self {
        self.preview_blocks = blocks;
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

    /// With `Some((gpu, join_after))`, the GPU helps with the passes (see `sample_pass`): it computes bit for bit the same samples (see `gpu`), so the image is the same.
    /// It joins the first pass once that has run for `join_after`, so a pass that is done before never waits for it; a later pass gets it from the start when the
    /// CPU threads alone would take longer than `GPU_WORTHWHILE` for it, and not at all otherwise.
    #[must_use]
    pub fn gpu(mut self, gpu: Option<(&'a GpuIterator, Duration)>) -> Self {
        self.gpu = gpu;
        self
    }

    /// With `fast`, the GPU uses double-float arithmetic (see `gpu`): about three times as fast, but not bit for bit the CPU's, so a small share of the pixels
    /// (in chaotic parts of the set) gets other colors. Only on views where its precision is ample (`FAST_GPU_STEPS`); deeper views keep the exact f64 loop.
    #[must_use]
    pub const fn fast_gpu(mut self, fast: bool) -> Self {
        self.fast_gpu = fast;
        self
    }

    /// Only the colors of one sample per pixel, without keeping the samples: for a frame of a flight that the next frame replaces. Writing the sum, count and first
    /// color of every pixel costs more memory traffic than coloring it (measured: a 4K flight frame of Frost on the Shoreline in 22 ms instead of 28 ms).
    /// With `resume`, the samples of the state are left as they are and only its memory for the colors is used. Supersampling, adaptive supersampling, previews and
    /// `only` do not apply.
    #[must_use]
    pub const fn colors_only(mut self) -> Self {
        self.keep_samples = false;
        self
    }

    /// Without CPU threads in the passes, so the GPU computes every sample (with `gpu`): to test that whole renders by the GPU are the CPU's, which a share that
    /// depends on the timing of the threads cannot show reliably
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn on_the_gpu_only(mut self) -> Self {
        self.cpu_threads = 0;
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
    #[allow(clippy::too_many_lines)] // The setup of a render (its state, previews, recolor) and its loop over the layers
    pub fn run(self, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
        let Render { plane: c, set: m, width, height, coloring_function, supersampling_amount, adaptive, exposed, cancel, pixels_done, preview, preview_blocks, state, gpu, gpu_samples, fast_gpu, cpu_threads, keep_samples } = self;
        //(The fast loop gets the pixels in 16 bits each, see `gpu::packed_pixel`)
        let fast_gpu = fast_gpu && c.precision_steps() >= FAST_GPU_STEPS && width <= MAX_PIXEL && height <= MAX_PIXEL;
        debug_assert!(cpu_threads > 0 || gpu.is_some(), "a render without CPU threads needs the GPU");
        let own_gpu_samples = AtomicUsize::new(0);
        let gpu_samples = gpu_samples.unwrap_or(&own_gpu_samples);
        let (never_cancelled, own_counter) = (AtomicBool::new(false), AtomicUsize::new(0));
        let (cancel, pixels_done) = (cancel.unwrap_or(&never_cancelled), pixels_done.unwrap_or(&own_counter));
        //The color of every iteration count, computed once (`color_of` below), for both kinds of renders
        let color_table: Option<Vec<TrueColor>> = (m.max_iterations <= MAX_COLOR_TABLE_ITERATIONS).then(|| color_table(m, coloring_function));
        let color_of = |position: f64| m.interpolated_color(position, |iterations| color_table.as_ref().map_or_else(|| m.color(coloring_function, iterations), |table| table[iterations as usize]));
        if !keep_samples {
            debug_assert!(exposed.is_none(), "a render without samples renders the whole view");
            let pass_gpu = gpu.map(|(gpu, join_after)| (gpu, join_after, gpu_samples, fast_gpu));
            return colors_only(c, m, width, height, state, &color_of, cancel, pixels_done, pass_gpu, cpu_threads, &mut on_pass);
        }
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
        let prepared = state.as_mut().map_or(Prepared::Empty, |state| state.prepare(c, width, height, SampleKey::new(m, coloring_function)));
        if let Some(state) = state.as_mut().filter(|state| state.recolor_pending) {
            //Only the palette or the coloring changed: the first samples are recolored from their positions, and the image of one sample is there at once
            state.recolor(&color_of);
        }
        let empty = prepared == Prepared::Empty;
        let mut own_state = None;
        let state: &mut SampleState = state.unwrap_or_else(|| own_state.insert(SampleState::for_one_render(width, height, region, base, adaptive.is_some())));
        //Where the samples are placed: the view where the samples of the state started, or this view
        let (anchor, origin) = (state.anchor.clone().unwrap_or_else(|| c.clone()), state.origin);
        let mut colors = std::mem::take(&mut state.colors);
        colors.clear();
        match base {
            Some(base) => colors.extend_from_slice(base),
            None => colors.resize(width * height, TrueColor::new(0, 0, 0)),
        }
        //How many pixels of the render have each amount of samples, to know how many pixels a pass computes
        let mut pending = [0_usize; 66];
        if empty {
            pending[0] = region.map_or(width * height, |region| region.pixel_count(width, height));
        } else {
            pending = colors_of_kept_samples(state, &mut colors);
        }
        let total: usize = pending.iter().sum();
        //The pixels whose first sample the previews computed, empty without previews
        let mut has_first: Vec<bool> = Vec::new();
        if let Some(worthwhile) = preview.filter(|_| pending[0] > 0) {
            let preview_gpu = gpu.map(|(gpu, join_after)| (gpu, join_after, gpu_samples, fast_gpu));
            match render_previews(&anchor, m, origin, worthwhile, preview_blocks, track, cancel, pixels_done, preview_gpu, cpu_threads, &color_of, state, pending[0], &mut colors, &mut on_pass) {
                Some((flags, count)) => {
                    has_first = flags;
                    pending[0] -= count;
                    pending[1] += count;
                }
                None => {
                    state.colors = colors;
                    return false;
                }
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
            //The GPU only for a pass that the CPU threads alone would take long for, from their time per sample in the pass before; then it joins at once.
            //For the first pass that time is unknown: then the join delay decides
            let expected = time_per_sample.map(|per_sample| per_sample * u32::try_from(eligible).unwrap_or(u32::MAX));
            let pass_gpu = gpu.filter(|_| cpu_threads == 0 || expected.is_none_or(|expected| expected > GPU_WORTHWHILE))
                .map(|(gpu, join_after)| (gpu, if expected.is_some() { Duration::ZERO } else { join_after }, gpu_samples, fast_gpu));
            let (pass_started, gpu_before) = (Instant::now(), gpu_samples.load(Ordering::Relaxed));
            let processed = if eligible > 0 { sample_pass(&anchor, m, sample, track, state, &mut colors, mask.as_deref(), &has_first, &color_of, cancel, pixels_done, pass_gpu, cpu_threads) } else { 0 };
            //The time per sample of the CPU threads alone: the samples of the GPU do not count, so a pass that the GPU made short does not keep it out of the next one
            let on_cpu = processed.saturating_sub(gpu_samples.load(Ordering::Relaxed).saturating_sub(gpu_before));
            if on_cpu > 0 {
                time_per_sample = Some(pass_started.elapsed() / u32::try_from(on_cpu).unwrap_or(u32::MAX));
            }
            if cancel.load(Ordering::Relaxed) {
                state.colors = colors;
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
        state.colors = colors;
        true
    }
}

/// A render of `Render::colors_only`: one pass over every pixel with a state that has no samples to keep (its rows are empty, see `sample_pass`), in the color
/// memory of `state` if there is one. Returns whether it completed.
#[allow(clippy::too_many_arguments)] // The parts of `Render::run` that the pass needs
fn colors_only(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, state: Option<&mut SampleState>, color_of: &(impl Fn(f64) -> TrueColor + Sync), cancel: &AtomicBool,
    pixels_done: &AtomicUsize, gpu: Option<(&GpuIterator, Duration, &AtomicUsize, bool)>, cpu_threads: usize, on_pass: &mut impl FnMut(&[TrueColor], u8)) -> bool {
    let mut state = state;
    let mut colors = state.as_mut().map_or_else(Vec::new, |state| std::mem::take(&mut state.colors));
    colors.clear();
    colors.resize(width * height, TrueColor::new(0, 0, 0));
    sample_pass(c, m, 0, false, &mut SampleState::without_samples(width, height), &mut colors, None, &[], color_of, cancel, pixels_done, gpu, cpu_threads);
    let completed = !cancel.load(Ordering::Relaxed);
    if completed {
        on_pass(&colors, 1);
    }
    if let Some(state) = state {
        state.colors = colors;
    }
    completed
}

/// Pass `sample` of `Render::run`: takes sample `sample` (placed in `anchor`, see `positions`) of every pixel that has `sample` samples (and is marked by `mask`, if
/// any), updates `state` and `colors` row by row on all threads, and returns how many pixels it sampled. A cancelled pass stops in the middle of a row; the pixels it
/// sampled are complete in `state`. Without `track` (a state that is not kept) every pixel of the render has `sample` samples except the ones the previews computed,
/// so the counts are not updated.
///
/// With `gpu`, the GPU helps once the pass has run for its join delay (a fast pass never waits for it). Its worker takes batches of rows from the same queue as the
/// CPU threads, so the work divides itself by speed, and keeps two dispatches on the GPU so that it never waits: each about `GPU_BATCH_TIME` long at the speed
/// measured in the pass, and at most a quarter of the rows that are left, so the CPU threads do not wait long for the last one. The GPU computes the same positions
/// bit for bit (see `gpu`); the CPU threads turn its outcomes into positions and apply them with the same code as their own rows, so the worker only feeds the GPU.
/// If the GPU fails, its rows are computed on the CPU and the GPU is not used for the rest of the pass.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // The state of `Render::run` that a pass shares; the CPU threads and the GPU worker share its closures
fn sample_pass(c: &ComplexPlane, m: &MandelbrotSet, sample: u8, track: bool, state: &mut SampleState, colors: &mut [TrueColor], mask: Option<&[bool]>, has_first: &[bool], color_of: &(impl Fn(f64) -> TrueColor + Sync),
    cancel: &AtomicBool, pixels_done: &AtomicUsize, gpu: Option<(&GpuIterator, Duration, &AtomicUsize, bool)>, cpu_threads: usize) -> usize {
    let (width, height, origin) = (state.width, state.height, state.origin);
    let started = Instant::now();
    let processed = AtomicUsize::new(0);
    let divide = divider(u32::from(sample) + 1);
    let skip_first = sample == 0 && !has_first.is_empty();
    //The pixels of row `y` that get this sample
    //Most rows of a pass are whole (a frame of a flight: every row): checking that at once and sharing this list is several times as fast as choosing pixel by pixel
    let every_pixel: Vec<usize> = (0..width).collect();
    let select = |y: usize, counts: &[u8]| -> Cow<[usize]> {
        let whole = mask.is_none() && if track { counts.iter().all(|count| *count == sample) } else { !skip_first && counts.iter().all(|count| *count != FOREIGN) };
        if whole {
            return Cow::Borrowed(&every_pixel);
        }
        Cow::Owned(if track {
            (0..width).filter(|x| counts[*x] == sample && mask.is_none_or(|mask| mask[y * width + x])).collect()
        } else {
            (0..width).filter(|x| counts[*x] != FOREIGN && mask.is_none_or(|mask| mask[y * width + x]) && !(skip_first && has_first[y * width + x])).collect()
        })
    };
    //Adds the samples at `positions` of the pixels `xs` of a row, and updates its running averages and counts: the same for the CPU threads and the GPU
    let apply = |y: usize, row: &mut [[u32; 3]], counts: &mut [u8], first: &mut [TrueColor], second: &mut [TrueColor], first_positions: &mut [f64], color_row: &mut [TrueColor], xs: &[usize], positions: &[f64]| {
        //The position of a first sample, to recolor it later (see `SampleState::recolor`)
        if sample == 0 && !first_positions.is_empty() {
            for (x, position) in xs.iter().zip(positions) {
                first_positions[*x] = *position;
            }
        }
        let xs = &xs[..positions.len()];
        //A render without samples (`Render::colors_only`): only the colors
        let keep = !row.is_empty();
        if !keep {
            for (x, position) in xs.iter().zip(positions) {
                color_row[*x] = color_of(*position);
            }
        }
        for (x, position) in xs.iter().zip(positions).filter(|_| keep) {
            let color = color_of(*position);
            let sum = &mut row[*x];
            //The running average of all samples so far, computed here on the thread of the row instead of after the pass on a single thread
            let color = if sample == 0 {
                //The first sample replaces the sum, so a state that starts over does not have to clear its sums; the average of one sample is the sample
                *sum = [u32::from(color.red), u32::from(color.green), u32::from(color.blue)];
                color
            } else {
                sum[0] += u32::from(color.red);
                sum[1] += u32::from(color.green);
                sum[2] += u32::from(color.blue);
                TrueColor::new(divide(sum[0]) as u8, divide(sum[1]) as u8, divide(sum[2]) as u8)
            };
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
    let position_rows = rows_or_empty(&mut state.positions, width, height);
    //(A state without samples, see `Render::colors_only`, gives empty rows of sums and counts too)
    let (sum_rows, count_rows) = (rows_or_empty(&mut state.sums, width, height), rows_or_empty(&mut state.counts, width, height));
    let rows = Mutex::new(sum_rows.into_iter().zip(count_rows).zip(first_rows).zip(second_rows).zip(position_rows).zip(colors.chunks_mut(width)).enumerate());
    //The rows that the GPU computed, with their pixels and the outcomes of the batch from where they start (or `None` if the GPU failed: then the CPU computes them).
    //The CPU threads turn the outcomes into positions and apply them, so the GPU worker only feeds the GPU and never competes with them for the cores.
    let gpu_done: Mutex<Vec<GpuRow>> = Mutex::new(Vec::new());
    //The rows that the GPU worker took from `rows` and has not put into `gpu_done` yet: the CPU threads wait for those before they stop
    let gpu_pending = AtomicUsize::new(0);
    let finish_gpu_row = |(y, (((((row, counts), first), second), first_positions), color_row)): PassRow, xs: Cow<[usize]>, outcomes: Option<(Arc<GpuOutcomes>, usize)>| {
        let positions = match outcomes.as_ref().map(|(all, offset)| (&**all, *offset)) {
            Some((GpuOutcomes::Exact(all), offset)) => xs.iter().zip(&all[offset..]).map(|(x, outcome)| outcome.position(m, &sample_point(c, *x, y, sample, origin))).collect(),
            Some((GpuOutcomes::Fast(all), offset)) => all[offset..offset + xs.len()].iter().map(|outcome| outcome.position(m)).collect(),
            None => on_cpu(y, &xs),
        };
        apply(y, row, counts, first, second, first_positions, color_row, &xs, &positions);
    };
    thread::scope(|scope| {
        for _ in 0..cpu_threads {
            scope.spawn(|| loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let from_gpu = gpu_done.lock().unwrap().pop();
                if let Some((row, xs, outcomes)) = from_gpu {
                    finish_gpu_row(row, xs, outcomes);
                    continue;
                }
                let next_row = rows.lock().unwrap().next();
                if let Some((y, (((((row, counts), first), second), first_positions), color_row))) = next_row {
                    let xs = select(y, counts);
                    if xs.is_empty() && !skip_first {
                        continue; //Nothing to sample or to color in this row (a preview level, an adaptive pass)
                    }
                    let positions = on_cpu(y, &xs);
                    apply(y, row, counts, first, second, first_positions, color_row, &xs, &positions);
                    continue;
                }
                //No rows left: the rows on the GPU still come back to these threads (`gpu_pending` drops only after they are in `gpu_done`)
                if gpu_pending.load(Ordering::SeqCst) == 0 && gpu_done.lock().unwrap().is_empty() {
                    return;
                }
                thread::sleep(Duration::from_micros(50));
            });
        }
        if let Some((gpu, join_after, gpu_samples, fast)) = gpu.as_ref() {
            scope.spawn(|| {
                //A fast pass is done before the GPU joins: then it costs nothing
                while started.elapsed() < *join_after {
                    if cancel.load(Ordering::Relaxed) || rows.lock().unwrap().len() == 0 {
                        return;
                    }
                    thread::sleep(Duration::from_micros(200));
                }
                //Two dispatches on the GPU at a time, so it never waits for this thread; each about `GPU_BATCH_TIME` long at the speed measured so far,
                //and at most a quarter of the rows that are left, so the CPU threads do not wait for a large last batch
                let mut in_flight = VecDeque::new();
                let (mut gpu_works, mut rate, mut points_per_row) = (true, None::<f64>, width as f64);
                let (mut points_taken, mut rows_taken) = (0_usize, 0_usize);
                let gpu_started = Instant::now();
                let mut points_finished = 0_usize;
                //The pixels of a batch for the fast loop, kept for the next: a new vector of a megabyte would cost its page faults every batch
                let mut pixels: Vec<u32> = Vec::new();
                loop {
                    while gpu_works && in_flight.len() < 2 && !cancel.load(Ordering::Relaxed) {
                        let points = rate.map_or(GPU_ROWS as f64 * points_per_row, |rate| rate * GPU_BATCH_TIME.as_secs_f64()).max(GPU_MIN_POINTS as f64);
                        let wanted = (points / points_per_row.max(1.0)) as usize;
                        let batch: Vec<PassRow> = {
                            let mut rows = rows.lock().unwrap();
                            let take = wanted.min(rows.len().div_ceil(4)).min((BATCH as f64 / points_per_row.max(1.0)) as usize).max(1); //About `BATCH` points, also for rows with few pixels to sample (previews, adaptive passes)
                            let batch: Vec<PassRow> = rows.by_ref().take(take).collect();
                            gpu_pending.fetch_add(batch.len(), Ordering::SeqCst); //While the rows are locked: a CPU thread never sees them neither in `rows` nor pending
                            drop(rows);
                            batch
                        };
                        if batch.is_empty() {
                            break;
                        }
                        let rows_in_batch = batch.len();
                        let selected: Vec<(PassRow, Cow<[usize]>)> = batch.into_iter().map(|row| {
                            let xs = select(row.0, (row.1).0.0.0.0.1);
                            (row, xs)
                        }).collect();
                        //The fast loop gets the pixels and computes their points itself; the exact loop gets the points
                        let count: usize = selected.iter().map(|(_, xs)| xs.len()).sum();
                        let submitted = (count > 0).then(|| if *fast {
                            pixels.clear();
                            for ((y, _), xs) in &selected {
                                pixels.extend(xs.iter().map(|x| packed_pixel(*x, *y)));
                            }
                            gpu.submit_pixels(m, c, sample, origin, &pixels)
                        } else {
                            let points: Vec<Complex> = selected.iter().flat_map(|((y, _), xs)| xs.iter().map(move |x| sample_point(c, *x, *y, sample, origin))).collect();
                            gpu.submit(m, &points)
                        }); //Rows without pixels of this pass need no dispatch
                        //The points per row of the pass so far (rows with few or no pixels to sample in a preview level or an adaptive pass make it small)
                        (points_taken, rows_taken) = (points_taken + count, rows_taken + rows_in_batch);
                        points_per_row = points_taken as f64 / rows_taken as f64;
                        in_flight.push_back((selected, submitted, count));
                    }
                    let Some((selected, submitted, count)) = in_flight.pop_front() else { break };
                    if cancel.load(Ordering::Relaxed) {
                        return; //Not waiting for the dispatches on the GPU: their rows keep the samples they had, like a row the CPU stops in
                    }
                    let outcomes = submitted.map_or_else(|| Ok(GpuOutcomes::Exact(Vec::new())), |submitted| submitted.and_then(|submitted| if *fast {
                        gpu.finish(submitted).map(GpuOutcomes::Fast)
                    } else {
                        gpu.finish(submitted).map(GpuOutcomes::Exact)
                    }));
                    gpu_works = outcomes.is_ok();
                    if gpu_works {
                        gpu_samples.fetch_add(count, Ordering::Relaxed);
                        points_finished += count;
                        rate = Some(points_finished as f64 / gpu_started.elapsed().as_secs_f64().max(1e-6));
                    }
                    let outcomes = outcomes.ok().map(Arc::new);
                    let rows_in_batch = selected.len();
                    let mut offset = 0;
                    let finished: Vec<_> = selected.into_iter().map(|(row, xs)| {
                        let start = offset;
                        offset += xs.len();
                        (row, xs, outcomes.as_ref().map(|all| (Arc::clone(all), start)))
                    }).collect();
                    if cpu_threads == 0 {
                        for (row, xs, outcomes) in finished {
                            finish_gpu_row(row, xs, outcomes);
                        }
                    } else {
                        gpu_done.lock().unwrap().extend(finished);
                    }
                    gpu_pending.fetch_sub(rows_in_batch, Ordering::SeqCst);
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                }
                //Without CPU threads (a test) the rows that are left after the GPU failed are this thread's
                if cpu_threads == 0 {
                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        let next_row = rows.lock().unwrap().next();
                        let Some(row) = next_row else { return };
                        let xs = select(row.0, (row.1).0.0.0.0.1);
                        finish_gpu_row(row, xs, None);
                    }
                }
            });
        }
    });
    processed.into_inner()
}

/// The fast GPU arithmetic (see `Render::fast_gpu`) is only used where neighboring pixels are at least this many steps of a 64-bit number apart.
///
/// See `ComplexPlane::precision_steps`. Its numbers have about 48 bits instead of 53, 16 times coarser, so this is the 1024 steps where 64-bit numbers start to
/// degrade (`PRECISION_WARNING_STEPS`) times 16. That is a scale of about 10⁻⁹ at 1200 pixels wide; deeper views use the exact loop.
pub const FAST_GPU_STEPS: f64 = 16.0 * crate::complex_plane::PRECISION_WARNING_STEPS;

/// A pass is only given to the GPU when the CPU threads alone would take longer than this for it: starting the GPU worker for a short pass costs more than it saves
/// (measured with the refilling CPU loop: the default view at 32x adaptive took 60.6 ms with 12 ms here, 56.8 ms with 20 ms, and 56.2 ms without the GPU; 30 ms
/// made the slow views slower)
const GPU_WORTHWHILE: Duration = Duration::from_millis(20);

/// The rows the GPU takes from the queue of a pass for its first batch, before its speed is known (about 38,000 pixels at 1200 x 800)
const GPU_ROWS: usize = 32;

/// The fewest points of a dispatch: a dispatch takes at least as long as its slowest point, which one GPU thread iterates step after step (measured on an RTX 5070 Ti
/// in Seahorse Valley: 7.5 ms for 2,000 points as for 8,000), so only a dispatch of many points keeps the GPU busy (preview levels and adaptive passes have few
/// points per row)
const GPU_MIN_POINTS: usize = 1 << 16;

/// How long one dispatch of the GPU should take at the speed measured in the pass: long enough that the gaps between dispatches do not matter, short enough to
/// keep the window responsive (the GPU also draws it)
const GPU_BATCH_TIME: Duration = Duration::from_millis(10);

/// A row that the GPU computed: the row, its pixels, and the outcomes of its batch with where the row starts in them (`None`: the GPU failed)
type GpuRow<'r> = (PassRow<'r>, Cow<'r, [usize]>, Option<(Arc<GpuOutcomes>, usize)>);

/// The outcomes of a dispatch: of the exact loop, which the CPU turns into positions, or of the fast loop, which are positions
enum GpuOutcomes {
    Exact(Vec<Outcome>),
    Fast(Vec<FastOutcome>),
}

/// A row of a pass: its number, sums, counts, first and second colors, and colors
type PassRow<'r> = (usize, (((((&'r mut [[u32; 3]], &'r mut [u8]), &'r mut [TrueColor]), &'r mut [TrueColor]), &'r mut [f64]), &'r mut [TrueColor]));

/// Large color tables are built on all render threads. Starting the threads costs about 0.17 ms, so below this many entries one thread is as fast or faster
/// (measured: 20,000 Bernstein entries take 0.08 ms on one thread and 0.18 ms on all; at 65,376 entries both take about 0.2 ms)
const PARALLEL_COLOR_TABLE_ENTRIES: usize = 50_000;

/// The color of every iteration count from 0 to `max_iterations` of `m`. Every entry is independent of the others, so the table is the same however it is split.
/// Measured on 16 threads, one thread and all threads: 65,376 HSV entries 0.75 and 0.23 ms; 200,000 entries 0.64 and 0.24 ms (Bernstein), 2.45 and 0.45 ms (HSV).
pub(super) fn color_table(m: &MandelbrotSet, coloring_function: fn(f64, f64) -> TrueColor) -> Vec<TrueColor> {
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

/// Division by `divisor` (1 to 65, the samples of a pixel) as a multiplication, exactly `sum / divisor` for every sum of up to 65 colors (at most 255 * 65):
/// with `m = ceil(2^32 / divisor)`, `(sum * m) >> 32` errs by less than `sum * divisor / 2^32`, far below the gap to the next multiple of `divisor` (tested for every case)
pub(super) fn divider(divisor: u32) -> impl Fn(u32) -> u32 {
    let multiplier = (1_u64 << 32).div_ceil(u64::from(divisor));
    move |sum| ((u64::from(sum) * multiplier) >> 32) as u32
}

/// The average color of `count` samples whose colors add up to `sum`
fn average(sum: [u32; 3], count: u8) -> TrueColor {
    let count = u32::from(count);
    TrueColor::new((sum[0] / count) as u8, (sum[1] / count) as u8, (sum[2] / count) as u8)
}

/// Computes the position (smooth iteration count) of sample `sample` of the pixels `xs` of row `y` and gives each to `sink`, see `MandelbrotSet::positions`.
/// A pixel is pixel `(x + origin.0, y + origin.1)` of `anchor`, which places its samples and keys their jitter, see `SampleState`.
///
/// With `cancel`, it stops between groups once `cancel` is set, and returns how many of `xs` (from the start) it gave to `sink`; without, it does all of them.
/// The point of sample `sample` of pixel `(x, y)`: pixel `(x + origin.0, y + origin.1)` of `anchor`, at the jitter of that pixel and sample (see `SampleState`)
pub(super) fn sample_point(anchor: &ComplexPlane, x: usize, y: usize, sample: u8, origin: (isize, isize)) -> Complex {
    let (anchor_x, anchor_y) = (x.cast_signed() + origin.0, y.cast_signed() + origin.1);
    let (jitter_x, jitter_y) = sample_jitter(anchor_x as usize, anchor_y as usize, sample);
    anchor.complex_from_pixel_plane(anchor_x as f64 + jitter_x, anchor_y as f64 + jitter_y)
}

#[allow(clippy::too_many_arguments)] // The row, the sample and where to place it, and the cancel flag: the inner loop of every pass, kept as one function
fn positions(anchor: &ComplexPlane, m: &MandelbrotSet, y: usize, xs: &[usize], sample: u8, origin: (isize, isize), cancel: Option<&AtomicBool>, mut sink: impl FnMut(usize, f64)) -> usize {
    let points: Vec<Complex> = xs.iter().map(|x| sample_point(anchor, *x, y, sample, origin)).collect();
    let mut out = vec![0.0; xs.len()];
    let done = m.positions(&points, &mut out, cancel);
    for (x, position) in xs[..done].iter().zip(&out) {
        sink(*x, *position);
    }
    done
}

/// The previews of [`Render::preview`]: computes the first sample of the top-left pixel of every block of each level that has no samples yet into `state`, and reports
/// a level (filled into `colors`) when the rest of the first layer (`empty` pixels without samples at the start) is expected to take longer than `worthwhile`.
/// Returns which pixels got their first sample and how many, or `None` if cancelled. A level with many samples is computed like a pass, with the GPU (see
/// `sample_level_like_a_pass`); the others on the CPU threads here.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // The state of `Render::run` that the previews share with the passes
fn render_previews(c: &ComplexPlane, m: &MandelbrotSet, origin: (isize, isize), worthwhile: Duration, blocks: &[usize], track: bool, cancel: &AtomicBool, pixels_done: &AtomicUsize,
    gpu: Option<(&GpuIterator, Duration, &AtomicUsize, bool)>, cpu_threads: usize, color_of: &(impl Fn(f64) -> TrueColor + Sync),
    state: &mut SampleState, empty: usize, colors: &mut [TrueColor], on_pass: &mut impl FnMut(&[TrueColor], u8)) -> Option<(Vec<bool>, usize)> {
    let width = state.width;
    let mut has_first = vec![false; state.counts.len()];
    let mut precomputed = 0;
    //The samples the CPU threads computed here, for their speed (`busy_nanos`)
    let mut on_cpu = 0;
    let threads = render_thread_count();
    //The time the threads spent computing samples, without starting threads and waiting: on a fast view those would dominate the estimate
    let busy_nanos = AtomicU64::new(0);
    let height = state.height;
    for &block in blocks {
        //The GPU only for a level with many samples: a dispatch takes at least as long as its slowest point (see `GPU_MIN_POINTS`), which a coarse level does not make up for
        if let Some(gpu) = gpu.filter(|_| cpu_threads == 0 || width.div_ceil(block) * height.div_ceil(block) >= 4 * GPU_MIN_POINTS) {
            let level_started = Instant::now();
            let (representatives, sampled) = sample_level_like_a_pass(c, m, block, track, cancel, pixels_done, gpu, cpu_threads, color_of, state, &mut has_first, colors)?;
            precomputed += sampled;
            let remaining = level_started.elapsed().mul_f64(empty.saturating_sub(precomputed) as f64 / sampled.max(1) as f64);
            if remaining <= worthwhile {
                break;
            }
            fill_blocks(width, block, state, &has_first, &representatives, colors, empty == state.counts.len());
            on_pass(colors, 0);
            continue;
        }
        let count = AtomicUsize::new(0);
        //Every block has a representative: the pixel whose first sample it shows. That is its top-left pixel (always, in a render from nothing), unless that pixel
        //has samples from before (it was kept by a move): then a pixel of the block that a coarser level computed, or else its first pixel without samples. So a block
        //that has pixels without samples always shows a sample of one of them, and a preview of a strip has no gaps.
        let (across, bands) = (width.div_ceil(block), height.div_ceil(block));
        let mut representatives: Vec<Option<usize>> = vec![None; across * bands];
        let band = width * block;
        let (first_bands, position_bands) = (chunks_or_empty(&mut state.first, band, bands), chunks_or_empty(&mut state.positions, band, bands));
        let rows = Mutex::new(state.sums.chunks_mut(band).zip(state.counts.chunks_mut(band)).zip(first_bands).zip(position_bands).zip(has_first.chunks_mut(band)).zip(representatives.chunks_mut(across)).enumerate());
        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_band = rows.lock().unwrap().next();
                    let Some((by, (((((sums, counts), first), first_positions), has), representatives))) = next_band else { return };
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
                            if !first_positions.is_empty() {
                                first_positions[index] = position;
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
        let sampled = count.into_inner();
        (precomputed, on_cpu) = (precomputed + sampled, on_cpu + sampled);
        //The rest of the first layer, at the speed of the samples so far, on all threads
        let per_sample = busy_nanos.load(Ordering::Relaxed) as f64 / on_cpu.max(1) as f64;
        let remaining = Duration::from_secs_f64(per_sample * empty.saturating_sub(precomputed) as f64 / threads as f64 / 1e9);
        if remaining <= worthwhile {
            break;
        }
        fill_blocks(width, block, state, &has_first, &representatives, colors, empty == state.counts.len());
        on_pass(colors, 0);
    }
    Some((has_first, precomputed))
}

/// A preview level computed like a pass: chooses the representatives of the blocks (see `render_previews`), samples those without a sample yet with `sample_pass`
/// (the CPU threads and the GPU), and marks them in `has_first`. Returns the representatives and how many were sampled, or `None` if cancelled.
#[allow(clippy::too_many_arguments)] // The state of `render_previews`
fn sample_level_like_a_pass(c: &ComplexPlane, m: &MandelbrotSet, block: usize, track: bool, cancel: &AtomicBool, pixels_done: &AtomicUsize, gpu: (&GpuIterator, Duration, &AtomicUsize, bool),
    cpu_threads: usize, color_of: &(impl Fn(f64) -> TrueColor + Sync), state: &mut SampleState, has_first: &mut [bool], colors: &mut [TrueColor]) -> Option<(Vec<Option<usize>>, usize)> {
    let (width, height) = (state.width, state.height);
    let (across, bands) = (width.div_ceil(block), height.div_ceil(block));
    let mut representatives: Vec<Option<usize>> = vec![None; across * bands];
    let mut to_sample = vec![false; state.counts.len()];
    let band = width * block;
    let counts = &state.counts;
    let rows = Mutex::new(counts.chunks(band).zip(has_first.chunks(band)).zip(to_sample.chunks_mut(band)).zip(representatives.chunks_mut(across)).enumerate());
    thread::scope(|scope| {
        for _ in 0..render_thread_count() {
            scope.spawn(|| loop {
                let next_band = rows.lock().unwrap().next();
                let Some((by, (((counts, has), to_sample), representatives))) = next_band else { return };
                let rows_in_band = counts.len() / width;
                for (bx, representative) in representatives.iter_mut().enumerate() {
                    let (left, right) = (bx * block, (bx * block + block).min(width));
                    let mut pixels = (0..rows_in_band).flat_map(|row| (left..right).map(move |x| row * width + x));
                    let chosen = if has[left] || counts[left] == 0 { Some(left) } else { pixels.clone().find(|index| has[*index]).or_else(|| pixels.find(|index| counts[*index] == 0)) };
                    if let Some(index) = chosen {
                        *representative = Some(by * band + index);
                        to_sample[index] = counts[index] == 0 && !has[index];
                    }
                }
            });
        }
    });
    //(Without the flags of the first samples: the pixels to sample have none yet, and the colors of the others are filled in after the level)
    let sampled = sample_pass(c, m, 0, track, state, colors, Some(&to_sample), &[], color_of, cancel, pixels_done, Some(gpu), cpu_threads);
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    for &index in representatives.iter().flatten() {
        has_first[index] |= to_sample[index];
    }
    Some((representatives, sampled))
}

/// Gives every pixel without samples (and every pixel the previews computed) the color of the representative of its `block` x `block` block, see `render_previews`.
/// In a render from nothing (`from_nothing`) that is every pixel, so a row is filled block by block without looking at its pixels (measured at 3840 x 2160: 2.4 ms
/// for a level pixel by pixel)
fn fill_blocks(width: usize, block: usize, state: &SampleState, has_first: &[bool], representatives: &[Option<usize>], colors: &mut [TrueColor], from_nothing: bool) {
    let across = width.div_ceil(block);
    let rows = Mutex::new(colors.chunks_mut(width).enumerate());
    thread::scope(|scope| {
        for _ in 0..render_thread_count() {
            scope.spawn(|| loop {
                let next_row = rows.lock().unwrap().next();
                let Some((y, row)) = next_row else { return };
                let band = (y / block) * across;
                if from_nothing {
                    for (segment, representative) in row.chunks_mut(block).zip(&representatives[band..band + across]) {
                        if let Some(representative) = representative {
                            segment.fill(sum_color(state.sums[*representative]));
                        }
                    }
                    continue;
                }
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

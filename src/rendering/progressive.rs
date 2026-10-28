//! The progressive renderer: one supersampling layer at a time, so every layer can be shown as soon as it is computed.

use std::{thread, sync::{Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}}};

use super::adaptive::{adaptive_mask, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
use super::jitter::sample_jitter;
use super::region::ExposedRegion;
use super::render_thread_count;
use crate::{complex::Complex, complex_plane::ComplexPlane, coloring::TrueColor, mandelbrot_set::{MandelbrotSet, LANES}};

/// The largest `max_iterations` for which `Render::run` precomputes the color of every iteration count
const MAX_COLOR_TABLE_ITERATIONS: u32 = 200_000;

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
}

impl<'a> Render<'a> {
    /// A render of the whole `width` x `height` image of `plane`, with one sample per pixel, colored by `coloring_function` (iterations and maximum iterations to a color)
    pub fn new(plane: &'a ComplexPlane, set: &'a MandelbrotSet, width: usize, height: usize, coloring_function: fn(u32, u32) -> TrueColor) -> Self {
        Render { plane, set, width, height, coloring_function, supersampling_amount: 1, adaptive: None, exposed: None, cancel: None, pixels_done: None }
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

    /// Runs the render. After every completed pass `on_pass` is called with the colors of the whole image (`width * height`, row by row), that is `base` with the region replaced
    /// by the running average of all samples so far, and the number of the pass (from 1).
    /// Returns `false` if the render was cancelled before all passes completed.
    ///
    /// # Panics
    /// If `base` does not have `width * height` colors, or if a thread of the render panics.
    pub fn run(self, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
        let Render { plane: c, set: m, width, height, coloring_function, supersampling_amount, adaptive, exposed, cancel, pixels_done } = self;
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
    }

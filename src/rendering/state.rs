//! What a render has computed for every pixel, kept between renders: an interrupted render, or a render of a moved view, continues from it instead of starting again.
//!
//! For every pixel the state holds the sum of its samples, how many samples it has, and the colors after its first and second sample (which adaptive supersampling
//! compares). When the view moves by whole pixels the state is shifted with it. The samples of a pixel are placed in the view where the samples started (the
//! `anchor`), at its position there (its position now plus `origin`) and a jitter keyed by that position. So a pixel keeps exactly the same sample points however the
//! view moves: a continued render computes bit for bit the samples that a render that was never interrupted or moved would have computed. (Placing them in the moved
//! view instead would round its bounds differently, which changes the iteration count of an occasional point right at the edge of the set.)

use super::region::{pixel_shift, ExposedRegion};
use crate::{coloring::TrueColor, complex_plane::ComplexPlane, mandelbrot_set::MandelbrotSet};

/// The count of a pixel that is not part of the render (see `Render::only`): it keeps the color of the base and never gets samples
pub(super) const FOREIGN: u8 = u8::MAX;

/// The settings that the samples of a state depend on: a state is only continued by a render with the same settings
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SampleKey {
    max_iterations: u32,
    orbit_radius: u64,
    palette_scale: Option<u32>,
    smooth_coloring: bool,
    coloring_function: usize,
}

impl SampleKey {
    /// The key of samples of `m` colored by `coloring_function`. Adaptive supersampling and the amount of samples are not part of it: they decide how many samples
    /// a pixel gets, not what a sample is
    pub(super) fn new(m: &MandelbrotSet, coloring_function: fn(u32, u32) -> TrueColor) -> SampleKey {
        //Two pointers to the same function can differ (then the state is just not reused); two different functions never share an address
        SampleKey { max_iterations: m.max_iterations, orbit_radius: m.orbit_radius.to_bits(), palette_scale: m.palette_scale, smooth_coloring: m.smooth_coloring, coloring_function: coloring_function as usize }
    }
}

/// The samples of every pixel of a render, to continue it later, see `Render::resume`
#[derive(Clone, Default)]
pub struct SampleState {
    /// The view and the settings of the samples; `None` for an empty state
    plane: Option<ComplexPlane>,
    /// The view where the samples started: pixel `(x, y)` of `plane` is pixel `(x + origin.0, y + origin.1)` of the anchor
    pub(super) anchor: Option<ComplexPlane>,
    key: Option<SampleKey>,
    pub(super) width: usize,
    pub(super) height: usize,
    /// Added to the position of a pixel to get its position in the anchor, which places its samples and keys their jitter (see `jitter::sample_jitter`)
    pub(super) origin: (isize, isize),
    /// The sum of the colors of the samples of every pixel
    pub(super) sums: Vec<[u32; 3]>,
    /// How many samples every pixel has (or `FOREIGN`)
    pub(super) counts: Vec<u8>,
    /// The color of the first sample, and the average of the first two samples, of every pixel that has them
    pub(super) first: Vec<TrueColor>,
    pub(super) second: Vec<TrueColor>,
    /// The memory of the colors of the last render, so the next render does not allocate (and the system clear) a big image again, see `Render::run`
    pub(super) colors: Vec<TrueColor>,
}

impl SampleState {
    /// An empty state: the first render that resumes it starts from nothing
    pub fn new() -> SampleState {
        SampleState::default()
    }

    /// A state of the same view, settings and sample placement without any samples: rendering it gives the image that this state would have if it had never been
    /// interrupted or moved, to compare with in tests
    #[cfg(test)]
    pub(super) fn emptied(&self) -> SampleState {
        let mut empty = self.clone();
        empty.counts.fill(0);
        empty
    }

    /// How many samples the pixel `(x, y)` has
    pub fn samples(&self, x: usize, y: usize) -> u8 {
        self.counts.get(y * self.width + x).copied().unwrap_or(0)
    }

    /// The view whose samples the state holds, if any
    pub fn plane(&self) -> Option<&ComplexPlane> {
        self.plane.as_ref()
    }

    /// Makes the state fit a render of `plane` (`width` x `height` pixels) of `m` colored by `coloring_function`, like `Render::resume` does when it starts,
    /// and returns how many pixels kept samples. A render that resumes the state afterwards finds it prepared.
    pub fn prepare_for(&mut self, plane: &ComplexPlane, width: usize, height: usize, m: &MandelbrotSet, coloring_function: fn(u32, u32) -> TrueColor) -> usize {
        if self.prepare(plane, width, height, SampleKey::new(m, coloring_function)) { 0 } else { self.counts.iter().filter(|count| **count > 0).count() }
    }

    /// The shift that turns this state into the state of a render of `plane` with `key`, if that render keeps samples of it
    fn kept_shift(&self, plane: &ComplexPlane, width: usize, height: usize, key: &SampleKey) -> Option<(isize, isize)> {
        let same_render = self.width == width && self.height == height && self.key.as_ref() == Some(key);
        let shift = self.plane.as_ref().filter(|_| same_render).and_then(|old| pixel_shift(old, plane))?;
        let (rows_up, columns_right) = (isize::try_from(shift.0).ok()?, isize::try_from(shift.1).ok()?);
        (rows_up.unsigned_abs() < height && columns_right.unsigned_abs() < width).then_some((rows_up, columns_right))
    }

    /// Makes the state fit a render of `plane` (`width` x `height` pixels) with the settings `key`. Returns `true` if it started over (every pixel is empty).
    ///
    /// If `plane` is the view of the state moved by whole pixels, with the same size and settings, the samples are shifted along and the pixels that became
    /// visible start empty; otherwise every pixel starts empty.
    pub(super) fn prepare(&mut self, plane: &ComplexPlane, width: usize, height: usize, key: SampleKey) -> bool {
        let empty = match self.kept_shift(plane, width, height, &key) {
            Some((rows_up, columns_right)) => {
                self.shift(rows_up, columns_right);
                false
            }
            None if self.counts.len() == width * height && self.width == width => {
                //Starting over at the same size reuses the memory: only the counts are cleared, the other values of a pixel are written by its first samples
                self.counts.fill(0);
                (self.anchor, self.origin) = (Some(plane.clone()), (0, 0));
                true
            }
            None => {
                *self = SampleState { anchor: Some(plane.clone()), width, height, sums: vec![[0; 3]; width * height], counts: vec![0; width * height], first: vec![TrueColor::new(0, 0, 0); width * height], second: vec![TrueColor::new(0, 0, 0); width * height], ..SampleState::default() };
                true
            }
        };
        self.plane = Some(plane.clone());
        self.key = Some(key);
        empty
    }

    /// The state of a render that is not kept: `width` x `height` pixels without samples, of which only the pixels of `region` are rendered (all if `None`); the others
    /// are `FOREIGN`. The colors of the first and second samples are only kept with `adaptive` (the mask compares them), with `base` as the color of the other pixels.
    pub(super) fn for_one_render(width: usize, height: usize, region: Option<&ExposedRegion>, base: Option<&[TrueColor]>, adaptive: bool) -> SampleState {
        let mut counts = vec![0; width * height];
        if let Some(region) = region {
            for (y, row) in counts.chunks_mut(width).enumerate() {
                for (_, count) in row.iter_mut().enumerate().filter(|(x, _)| !region.contains(*x, y)) {
                    *count = FOREIGN;
                }
            }
        }
        let colors = || base.map_or_else(|| vec![TrueColor::new(0, 0, 0); width * height], <[TrueColor]>::to_vec);
        let (first, second) = if adaptive { (colors(), colors()) } else { (Vec::new(), Vec::new()) };
        SampleState { width, height, sums: vec![[0; 3]; width * height], counts, first, second, ..SampleState::default() }
    }

    /// Moves the content by `columns_right` and `rows_up` pixels like `PixelBuffer::translate_buffer`: the pixel at `(x, y)` goes to `(x + columns_right, y + rows_up)`.
    /// The origin follows, so every moved pixel keeps the key of its jitter.
    fn shift(&mut self, rows_up: isize, columns_right: isize) {
        if (rows_up, columns_right) == (0, 0) {
            return;
        }
        let (width, height) = (self.width, self.height);
        //Only the counts of the exposed pixels have to be cleared: the other values of a pixel without samples are never read, and are written by its first samples
        shift_grid(&mut self.counts, width, height, rows_up, columns_right, Some(0));
        shift_grid(&mut self.sums, width, height, rows_up, columns_right, None);
        shift_grid(&mut self.first, width, height, rows_up, columns_right, None);
        shift_grid(&mut self.second, width, height, rows_up, columns_right, None);
        self.origin = (self.origin.0 - columns_right, self.origin.1 - rows_up);
    }
}

/// Moves the values of a `width` x `height` grid by `columns_right` and `rows_up`, in place; the cells that nothing moved into get `empty`, or keep their old value with `None`
fn shift_grid<T: Copy>(grid: &mut [T], width: usize, height: usize, rows_up: isize, columns_right: isize, empty: Option<T>) {
    let columns = width.saturating_sub(columns_right.unsigned_abs());
    let (from_x, to_x) = if columns_right >= 0 { (0, columns_right.unsigned_abs()) } else { (columns_right.unsigned_abs(), 0) };
    let mut shift_row = |y: usize| {
        let row = y * width;
        if let Some(from_y) = y.checked_add_signed(-rows_up).filter(|from_y| *from_y < height) {
            grid.copy_within(from_y * width + from_x..from_y * width + from_x + columns, row + to_x);
            if let Some(empty) = empty {
                //The exposed columns of the row
                let exposed = if columns_right >= 0 { row..row + to_x } else { row + columns..row + width };
                grid[exposed].fill(empty);
            }
        } else if let Some(empty) = empty {
            grid[row..row + width].fill(empty);
        }
    };
    //A row is read before it is written: moving down, the rows are shifted from the bottom up
    if rows_up > 0 {
        (0..height).rev().for_each(&mut shift_row);
    } else {
        (0..height).for_each(&mut shift_row);
    }
}

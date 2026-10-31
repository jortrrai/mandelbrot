//! The Mandelbrot set: the escape-time iteration of a point.
//!
//! A point is iterated alone, or four at a time in lockstep so that the CPU's vector instructions can be used. Many points at once (`MandelbrotSet::positions`, which
//! the renderer uses) go through a refilling loop: a lane whose point has stopped takes the next point at once, so long and short orbits do not wait for each other
//! (with AVX2 see `avx2`). Every way gives every point the same result, bit for bit. Points that are certainly inside the set are recognized early (the main cardioid
//! and the period-two bulb, and orbits that repeat exactly). The smooth (continuous) iteration count that the coloring uses is computed here too.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::complex::Complex;

#[cfg(target_arch = "x86_64")]
mod avx2;

/// The Mandelbrot set as a computation: the maximum iterations and the orbit radius, with the escape-time iteration of a point and the coloring options of the renderer
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::unsafe_derive_deserialize)] // Plain numbers without invariants (for the render queue); the unsafe code is the vectorized iteration
pub struct MandelbrotSet {
    /// The most iterations of the escape-time iteration of a point; a point that has not escaped by then counts as inside the set
    pub max_iterations: u32,
    /// If `z` stays within the `orbit_radius` for `max_iterations` steps, `c` is assumed not to tend to infinity
    pub orbit_radius: f64,    
    ///How iteration counts map onto the palette, see `ColorCycle` and `MandelbrotSet::color`
    pub palette: ColorCycle,
    ///Whether colors use the smooth (fractional) iteration position of `iterate_smooth` instead of the integer iteration count, see `MandelbrotSet::position`
    pub smooth_coloring: bool,
}

/// How iteration counts map onto the palette, as in Kalles Fraktaler.
///
/// Around and around a palette of `PALETTE_LENGTH` steps, `divide` iterations a step and starting `offset` steps into it, or stretched once over the iteration limit.
/// As a cycle, a point keeps its color whatever the limit and the zoom, so a manual limit and an automatic one give the same image, and a flight does not recolor
/// what it zooms into.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColorCycle {
    /// Iterations per step of the palette: 1 changes color fastest, more gives broader bands (deep views, whose counts are high and far apart, need more)
    pub divide: f64,
    /// Where the palette starts, in steps (0 to `PALETTE_LENGTH`)
    pub offset: f64,
    /// Whether the palette is stretched once over the iteration limit instead (`divide` and `offset` are not used then)
    pub stretch: bool,
}

impl ColorCycle {
    /// The palette stretched once over the limit: the coloring of the renderer before color cycles, which the library keeps by default
    pub const STRETCHED: ColorCycle = ColorCycle { divide: 1.0, offset: 0.0, stretch: true };
    /// A cycle of the palette every `PALETTE_LENGTH` iterations from its start: the default of the window (a view at zoom 1 looks as with the automatic limit before)
    pub const CYCLE: ColorCycle = ColorCycle { divide: 1.0, offset: 0.0, stretch: false };

    /// The cycle a fraction `t` (0 to 1) of the way from `self` to `to`: the divide in equal ratios (it spans orders of magnitude), the offset the short way round
    /// the palette; stretched only if both are
    pub fn blend(self, to: ColorCycle, t: f64) -> ColorCycle {
        let t = t.clamp(0.0, 1.0);
        let divide = self.divide.max(MIN_DIVIDE) * (to.divide.max(MIN_DIVIDE) / self.divide.max(MIN_DIVIDE)).powf(t);
        let mut turn = (to.offset - self.offset).rem_euclid(PALETTE_LENGTH);
        if turn > PALETTE_LENGTH / 2.0 {
            turn -= PALETTE_LENGTH;
        }
        ColorCycle { divide, offset: (self.offset + turn * t).rem_euclid(PALETTE_LENGTH), stretch: self.stretch && to.stretch }
    }
}

/// The steps of the palette of a `ColorCycle`: with a divide of 1, the colors repeat every this many iterations
pub const PALETTE_LENGTH: f64 = 500.0;

/// The smallest iteration divide (ten steps of the palette for every iteration: a cycle every 50 iterations; finer only looks like noise)
pub const MIN_DIVIDE: f64 = 0.1;

/// The largest iteration divide (a cycle every 500,000 iterations, more than the ceiling of auto iterations; broader only looks like one color)
pub const MAX_DIVIDE: f64 = 1000.0;

/// The amount of points `MandelbrotSet::position_lanes` iterates at the same time
pub const LANES: usize = 4;
/// Smooth coloring continues an escaped orbit until |z|^2 reaches this (radius 256), or for at most `SMOOTH_MAX_EXTRA_STEPS` steps
const SMOOTH_BAILOUT_SQUARED: f64 = 65536.0;
const SMOOTH_MAX_EXTRA_STEPS: u32 = 32;

/// The result of `iterate_lanes_generic`: the iteration counts, and (when smooth) for the points that escaped the value of z at the end of the block of `CHECK_INTERVAL` steps
/// in which they escaped, see `MandelbrotSet::block_end`
struct LaneResult {
    iterations: [u32; LANES],
    escaped: [bool; LANES],
    escape_x: [f64; LANES],
    escape_y: [f64; LANES],
}

/// The lockstep and refilling loops check which points have stopped after every block of this many steps
pub(crate) const CHECK_INTERVAL: u32 = 4;

/// The lanes of the generic refilling loop (`MandelbrotSet::positions_generic`): one vector of 4 (more lanes were slower with all threads busy, which hide the
/// latency of a step already)
const STREAM_LANES: usize = 4;

impl MandelbrotSet {
    /// Creates the set with `max_iterations` and `orbit_radius`; smooth coloring is on and the palette is stretched over the limit (`ColorCycle::STRETCHED`)
    pub fn new(max_iterations: u32, orbit_radius: f64) -> MandelbrotSet {
        MandelbrotSet { max_iterations, orbit_radius, palette: ColorCycle::STRETCHED, smooth_coloring: true }
    }

    /// The color of a point with the given iteration count, see `ColorCycle`. Stretched over the limit, this is exactly `coloring_function(iterations, max_iterations)`.
    /// As a cycle, an escaped point is at `iterations / divide + offset` steps of the palette, around and around a palette of `PALETTE_LENGTH` steps, whatever the limit
    /// and the zoom; points inside the set (`iterations == max_iterations`) get the color of the end of the palette, as when stretched.
    pub fn color(&self, coloring_function: fn(f64, f64) -> crate::coloring::TrueColor, iterations: u32) -> crate::coloring::TrueColor {
        let ColorCycle { divide, offset, stretch } = self.palette;
        if stretch {
            return coloring_function(f64::from(iterations), f64::from(self.max_iterations));
        }
        if iterations >= self.max_iterations {
            return coloring_function(PALETTE_LENGTH, PALETTE_LENGTH);
        }
        let position = (f64::from(iterations) / divide.max(MIN_DIVIDE) + offset).rem_euclid(PALETTE_LENGTH);
        coloring_function(position, PALETTE_LENGTH)
    }

    /// Whether `c` is certified to be inside the set by the cardioid and period-two bulb test, see `is_known_interior`
    pub fn is_certified_interior(&self, c: &Complex) -> bool {
        self.is_known_interior(c)
    }

    /// The position in the color table of a point: `iterate` as a number, or with `smooth_coloring` the smooth position of `iterate_smooth`
    pub fn position(&self, c: &Complex) -> f64 {
        if self.smooth_coloring { self.iterate_smooth(c) } else { f64::from(self.iterate(c)) }
    }

    /// Like `position`, for `LANES` points at the same time: the points are iterated in lockstep, with the same arithmetic in the same order as `position`, so the
    /// compiler can use vector instructions; a point that has stopped stops counting, but the others continue until none is left. Identical to `position` for every point.
    pub fn position_lanes(&self, points: &[Complex; LANES]) -> [f64; LANES] {
        let result = self.iterate_lanes_smooth_dispatch(points);
        std::array::from_fn(|lane| self.position_after_loop(result.iterations[lane], result.escaped[lane], (result.escape_x[lane], result.escape_y[lane]), &points[lane]))
    }

    /// The continuous escape time of a point (normalized iteration count): the iteration count of `iterate` plus a fraction in about [-0.85, 0.53) that depends on how far
    /// beyond the orbit radius the orbit landed, so neighboring points in different integer bands get neighboring positions. Points that do not escape get `max_iterations`.
    /// The integer count and the interior classification are exactly those of `iterate`; only the escaped points do a few extra steps and two logarithms.
    /// Formula: `nu = n - log2(ln|z_n|)` for a large `|z_n|`, see <https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Continuous_(smooth)_coloring>
    pub fn iterate_smooth(&self, c: &Complex) -> f64 {
        if self.is_known_interior(c) {
            return f64::from(self.max_iterations);
        }
        let orbit_radius_squared = self.orbit_radius * self.orbit_radius;
        let mut z = Complex::new(0.0, 0.0);
        for iterations in 0..self.max_iterations {
            z = z.squared().add(c);
            if (z.x * z.x + z.y * z.y) > orbit_radius_squared {
                //Continue to the end of the block of CHECK_INTERVAL steps, like the lockstep loop does for its points, so both paths start the smooth position from the same z
                let end = self.block_end(iterations + 1);
                for _ in (iterations + 1)..end {
                    z = z.squared().add(c);
                }
                return self.smooth_position(end, (z.x, z.y), c);
            }
        }
        f64::from(self.max_iterations)
    }

    /// The number of steps at the end of the block of `CHECK_INTERVAL` steps of the lockstep loop that contains step `step` (1-based), or `max_iterations` for the last block
    fn block_end(&self, step: u32) -> u32 {
        (step.div_ceil(CHECK_INTERVAL) * CHECK_INTERVAL).min(self.max_iterations)
    }

    /// The smooth position of a point that escaped: `steps` is the index of `z` in the orbit (the count of steps taken), `z` a value of the orbit outside the orbit radius.
    /// The orbit is continued to radius 256 (at most `SMOOTH_MAX_EXTRA_STEPS` steps), where the formula is accurate and independent of the orbit radius.
    fn smooth_position(&self, steps: u32, z: (f64, f64), c: &Complex) -> f64 {
        let (mut zx, mut zy) = z;
        let mut steps = steps;
        for _ in 0..SMOOTH_MAX_EXTRA_STEPS {
            if zx * zx + zy * zy >= SMOOTH_BAILOUT_SQUARED {
                break;
            }
            (zx, zy) = (zx * zx - zy * zy + c.x, 2.0 * zx * zy + c.y);
            steps += 1;
        }
        let modulus_squared = zx * zx + zy * zy;
        if modulus_squared <= 1.0 {
            return f64::from(steps - 1).min(f64::from(self.max_iterations.saturating_sub(1))); //The formula needs |z| > 1; only possible with an orbit radius below 1.5 that was left again
        }
        let log_modulus = 0.5 * modulus_squared.ln();
        let position = f64::from(steps - 1) - log_modulus.ln() / std::f64::consts::LN_2;
        position.clamp(0.0, f64::from(self.max_iterations.saturating_sub(1)))
    }

    /// The smooth position of a point `c` that escaped after `steps` steps at `z` (see `smooth_position`), for renderers that iterate it another way (deep zoom,
    /// see `rendering::deep`)
    pub(crate) fn smooth_position_of(&self, steps: u32, z: (f64, f64), c: (f64, f64)) -> f64 {
        if self.smooth_coloring { self.smooth_position(steps, z, &Complex::new(c.0, c.1)) } else { f64::from(steps) }
    }

    /// The color at a (fractional) table position: the colors of the two nearest integer iteration counts, mixed by the fraction in 1/256 steps (integer arithmetic).
    /// `lookup` gives the color of an integer iteration count. A whole position gives exactly `lookup(position)`, which is what coloring by integer counts did.
    pub fn interpolated_color(&self, position: f64, lookup: impl Fn(u32) -> crate::coloring::TrueColor) -> crate::coloring::TrueColor {
        let index = position as u32;
        let weight = ((position - f64::from(index)) * 256.0) as u32;
        let first = lookup(index);
        if weight == 0 || index >= self.max_iterations {
            return first;
        }
        let second = lookup(index + 1);
        let mix = |a: u8, b: u8| ((u32::from(a) * (256 - weight) + u32::from(b) * weight + 128) >> 8) as u8;
        crate::coloring::TrueColor::new(mix(first.red, second.red), mix(first.green, second.green), mix(first.blue, second.blue))
    }

    /// Run the Mandelbrot set algorithm for a single Complex number
    /// Returns the amount of iterations needed before Zn escapes to infinity
    pub fn iterate(&self, c: &Complex) -> u32 {
        if self.is_known_interior(c) {
            return self.max_iterations; //Same result as the escape loop below, which never breaks for points inside the set
        }
        self.iterate_escape_loop(c)
    }

    /// The position of `c` from the outcome of the escape loop of the lockstep path (also the GPU's, see `rendering::gpu`): the iteration count, whether it escaped,
    /// and `z` at the end of the block of `CHECK_INTERVAL` steps in which it escaped. With `smooth_coloring` an escaped point gets its smooth position, otherwise the count.
    pub(crate) fn position_after_loop(&self, iterations: u32, escaped: bool, z_at_block_end: (f64, f64), c: &Complex) -> f64 {
        if escaped && self.smooth_coloring {
            self.smooth_position(self.block_end(iterations + 1), z_at_block_end, c)
        } else {
            f64::from(iterations)
        }
    }


    /// Runs the lockstep loop that also records where each point escaped (see `iterate_lanes_smooth_generic`): with AVX2 when the CPU has it, else without
    fn iterate_lanes_smooth_dispatch(&self, points: &[Complex; LANES]) -> LaneResult {
        let mut out = LaneResult { iterations: [0; LANES], escaped: [false; LANES], escape_x: [0.0; LANES], escape_y: [0.0; LANES] };
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                //SAFETY: the CPU supports AVX2, which was just detected
                unsafe { self.iterate_lanes_smooth_avx2(points, &mut out) };
                return out;
            }
        }
        self.iterate_lanes_smooth_generic(points, &mut out);
        out
    }

    /// The loop of `iterate_lanes_smooth_generic` compiled with AVX2 enabled
    /// # Safety
    /// The CPU must support AVX2 (checked by the caller with `is_x86_feature_detected!`)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn iterate_lanes_smooth_avx2(&self, points: &[Complex; LANES], out: &mut LaneResult) {
        self.iterate_lanes_smooth_generic(points, out);
    }

    /// The lockstep loop that also records, for every point that escaped, z at the end of the block of steps in which it escaped (so the smooth position can be computed). Its arithmetic is exactly that of the plain loop
    // Exact comparison of floats is the cycle certificate (an orbit that repeats bit for bit); inline(always) is a measured performance requirement (see the documentation)
    #[allow(clippy::float_cmp, clippy::inline_always)]
    #[inline(always)]
    fn iterate_lanes_smooth_generic(&self, points: &[Complex; LANES], out: &mut LaneResult) {
        let orbit_radius_squared = self.orbit_radius * self.orbit_radius;
        let mut iterations = [0_u32; LANES];
        let mut active = [true; LANES];
        let (mut cx, mut cy) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        for lane in 0..LANES {
            (cx[lane], cy[lane]) = (points[lane].x, points[lane].y);
            if self.is_known_interior(&points[lane]) {
                active[lane] = false;
                iterations[lane] = self.max_iterations;
            }
        }
        let (mut zx, mut zy) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        //The squares of z from the escape check of the step before: the squares the next step needs (the same numbers, 3 multiplications a step instead of 6)
        let (mut x2, mut y2) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        let (mut captured, mut escape_x, mut escape_y) = ([false; LANES], [0.0_f64; LANES], [0.0_f64; LANES]);
        let (mut saved_x, mut saved_y) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        let (mut blocks_since_saved, mut blocks_until_save) = (0_u32, 1_u32);
        let mut step = 0;
        while step < self.max_iterations {
            if !active.iter().any(|lane_active| *lane_active) {
                break;
            }
            //Only check whether any point is left every few steps; points that already stopped do not count anymore
            for _ in 0..CHECK_INTERVAL.min(self.max_iterations - step) {
                for lane in 0..LANES {
                    //Same arithmetic as z = z.squared().add(c), with the squares of the step before and 2 * zx as zx + zx (exact either way)
                    let new_x = x2[lane] - y2[lane] + cx[lane];
                    let new_y = (zx[lane] + zx[lane]) * zy[lane] + cy[lane];
                    (zx[lane], zy[lane]) = (new_x, new_y);
                    (x2[lane], y2[lane]) = (new_x * new_x, new_y * new_y);
                    let escaped = (x2[lane] + y2[lane]) > orbit_radius_squared;
                    active[lane] &= !escaped;
                    iterations[lane] += u32::from(active[lane]);
                }
            }
            let (block_start, block_end) = (step, (step + CHECK_INTERVAL).min(self.max_iterations));
            step += CHECK_INTERVAL;
            //A point that stopped in this block: keep z at the end of the block (the lanes keep iterating, so this is z at the block end) for the smooth position, without branches
            for lane in 0..LANES {
                //Escaped in this block: the point stopped counting after `step_before` and before the end of the block
                let stopped_now = iterations[lane] >= block_start && iterations[lane] < block_end;
                escape_x[lane] = if stopped_now { zx[lane] } else { escape_x[lane] };
                escape_y[lane] = if stopped_now { zy[lane] } else { escape_y[lane] };
                captured[lane] |= stopped_now;
            }
            //Brent's cycle detection, once per block of steps: if a point is exactly where it was at an earlier block, its orbit repeats forever,
            //and every point of it was inside the orbit radius, so the original loop would count every step: max_iterations
            for lane in 0..LANES {
                if active[lane] && zx[lane] == saved_x[lane] && zy[lane] == saved_y[lane] {
                    active[lane] = false;
                    iterations[lane] = self.max_iterations;
                }
            }
            blocks_since_saved += 1;
            if blocks_since_saved == blocks_until_save {
                (saved_x, saved_y) = (zx, zy);
                blocks_since_saved = 0;
                blocks_until_save = blocks_until_save.saturating_mul(2);
            }
        }
        //Points that escaped have iterations < max_iterations (and a captured z); the others stopped inside the set or ran to the limit
        let escaped = std::array::from_fn(|lane| captured[lane] && iterations[lane] < self.max_iterations);
        *out = LaneResult { iterations, escaped, escape_x, escape_y };
    }

    /// The position (see `position`) of every point of `points` into `out`, exactly `position_lanes` and `position` give it, but faster: a lane whose point has
    /// stopped takes the next point at once instead of waiting for the others (the refill). Every lane keeps its own step and cycle detection, so every point gets
    /// exactly the arithmetic of the lockstep loop, from the same start. With AVX2 see `avx2`; otherwise `positions_generic`.
    ///
    /// With `cancel`, it stops soon after `cancel` is set and returns how many points from the start of `points` have their position in `out`; without, all of them.
    pub fn positions(&self, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
        debug_assert_eq!(points.len(), out.len());
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let wide = std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512vl");
                //SAFETY: the CPU supports AVX2, and AVX-512 if `wide`, which was just detected
                return unsafe { avx2::positions(self, points, out, cancel, wide) };
            }
        }
        self.positions_generic(points, out, cancel)
    }

    /// The refilling loop of `positions`. After every block of `CHECK_INTERVAL` steps, a lane is done when its point escaped in the block (`z` at the end of the
    /// block gives the smooth position, like in the lockstep loop), when its orbit repeats (Brent's cycle detection, with its own counters), or when it has no whole
    /// block left before `max_iterations`: then the last steps are done by `finish_alone`, exactly like the last, shorter block of the lockstep loop.
    // Exact comparison of floats is the cycle certificate (an orbit that repeats bit for bit); inline(always) puts the loop into the AVX2 copy
    #[allow(clippy::float_cmp, clippy::inline_always)]
    #[inline(always)]
    fn positions_generic(&self, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
        const IDLE: usize = usize::MAX;
        let max = self.max_iterations;
        //A lane at this step or later has no whole block left
        let last_whole = max - max % CHECK_INTERVAL;
        let orbit_radius_squared = self.orbit_radius * self.orbit_radius;
        let (mut cx, mut cy, mut zx, mut zy) = ([0.0_f64; STREAM_LANES], [0.0_f64; STREAM_LANES], [0.0_f64; STREAM_LANES], [0.0_f64; STREAM_LANES]);
        let (mut saved_x, mut saved_y) = ([0.0_f64; STREAM_LANES], [0.0_f64; STREAM_LANES]);
        //The squares of z from the escape check of the step before (see `iterate_lanes_smooth_generic`)
        let (mut x2, mut y2) = ([0.0_f64; STREAM_LANES], [0.0_f64; STREAM_LANES]);
        let (mut iterations, mut active) = ([0_u32; STREAM_LANES], [false; STREAM_LANES]);
        //The blocks of steps every lane has done
        let mut blocks = [0_u32; STREAM_LANES];
        let last_whole_block = last_whole / CHECK_INTERVAL;
        let mut index = [IDLE; STREAM_LANES];
        let mut next = 0;
        //Puts the next point that needs the loop into `lane` (points in the cardioid or the period-2 bulb get their position at once), or makes the lane idle
        let load = |lane: usize, next: &mut usize, index: &mut [usize; STREAM_LANES], cx: &mut [f64; STREAM_LANES], cy: &mut [f64; STREAM_LANES], out: &mut [f64]| {
            while *next < points.len() {
                let (at, c) = (*next, &points[*next]);
                *next += 1;
                if last_whole == 0 || self.is_known_interior(c) {
                    out[at] = self.position(c); //Fewer steps than a block (the lockstep loop is the same as the single one then), or inside the set
                    continue;
                }
                (index[lane], cx[lane], cy[lane]) = (at, c.x, c.y);
                return true;
            }
            (index[lane], cx[lane], cy[lane]) = (IDLE, 0.0, 0.0); //An idle lane iterates 0, which stays 0
            false
        };
        let mut live = 0;
        for lane in 0..STREAM_LANES {
            if load(lane, &mut next, &mut index, &mut cx, &mut cy, out) {
                (zx[lane], zy[lane], saved_x[lane], saved_y[lane], iterations[lane], active[lane], blocks[lane]) = (0.0, 0.0, 0.0, 0.0, 0, true, 0);
                (x2[lane], y2[lane]) = (0.0, 0.0);
                live += 1;
            }
        }
        while live > 0 {
            if cancel.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
                //Every point before the first one that a lane still has (and before `next`) is done
                return index.iter().copied().filter(|at| *at != IDLE).min().unwrap_or(next).min(next);
            }
            for _ in 0..CHECK_INTERVAL {
                for lane in 0..STREAM_LANES {
                    //Same arithmetic as z = z.squared().add(c), with the squares of the step before and 2 * zx as zx + zx (exact either way)
                    let new_x = x2[lane] - y2[lane] + cx[lane];
                    let new_y = (zx[lane] + zx[lane]) * zy[lane] + cy[lane];
                    (zx[lane], zy[lane]) = (new_x, new_y);
                    (x2[lane], y2[lane]) = (new_x * new_x, new_y * new_y);
                    let escaped = (x2[lane] + y2[lane]) > orbit_radius_squared;
                    active[lane] &= !escaped;
                    iterations[lane] += u32::from(active[lane]);
                }
            }
            //Which lanes are done: stopped in this block, a repeating orbit, or no whole block left. The others keep their cycle detection like the lockstep loop,
            //which saves z after 1, 3, 7, 15, ... blocks (the gap doubles): after `blocks` blocks when `blocks + 1` is a power of two
            let mut any_done = false;
            let mut repeats = [false; STREAM_LANES];
            for lane in 0..STREAM_LANES {
                blocks[lane] += 1;
                repeats[lane] = active[lane] && zx[lane] == saved_x[lane] && zy[lane] == saved_y[lane];
                any_done |= index[lane] != IDLE && (!active[lane] || repeats[lane] || blocks[lane] >= last_whole_block);
                let save = blocks[lane] & (blocks[lane] + 1) == 0;
                saved_x[lane] = if save { zx[lane] } else { saved_x[lane] };
                saved_y[lane] = if save { zy[lane] } else { saved_y[lane] };
            }
            if !any_done {
                continue;
            }
            for lane in 0..STREAM_LANES {
                let at = index[lane];
                if at == IDLE || (active[lane] && !repeats[lane] && blocks[lane] < last_whole_block) {
                    continue;
                }
                let c = Complex::new(cx[lane], cy[lane]);
                out[at] = if !active[lane] {
                    self.position_after_loop(iterations[lane], true, (zx[lane], zy[lane]), &c) //Escaped in this block
                } else if repeats[lane] {
                    f64::from(max)
                } else {
                    self.finish_alone(blocks[lane] * CHECK_INTERVAL, iterations[lane], (zx[lane], zy[lane]), &c)
                };
                if load(lane, &mut next, &mut index, &mut cx, &mut cy, out) {
                    (zx[lane], zy[lane], saved_x[lane], saved_y[lane], iterations[lane], active[lane], blocks[lane]) = (0.0, 0.0, 0.0, 0.0, 0, true, 0);
                (x2[lane], y2[lane]) = (0.0, 0.0);
                } else {
                    (zx[lane], zy[lane], active[lane]) = (0.0, 0.0, false);
                    live -= 1;
                }
            }
        }
        points.len()
    }

    /// The position of a point that has not stopped after `steps` steps (`iterations` of them counted, `z` the value after them) and has fewer than a whole block of
    /// steps left: the last, shorter block of the lockstep loop, for this point alone
    fn finish_alone(&self, steps: u32, iterations: u32, z: (f64, f64), c: &Complex) -> f64 {
        let orbit_radius_squared = self.orbit_radius * self.orbit_radius;
        let ((mut zx, mut zy), mut iterations, mut active) = (z, iterations, true);
        for _ in steps..self.max_iterations {
            let new_x = zx * zx - zy * zy + c.x;
            let new_y = 2.0 * zx * zy + c.y;
            (zx, zy) = (new_x, new_y);
            active &= (new_x * new_x + new_y * new_y) <= orbit_radius_squared;
            iterations += u32::from(active);
        }
        self.position_after_loop(iterations, !active, (zx, zy), c)
    }

    /// Returns true only if `c` is certainly inside the main cardioid or the period-two bulb, so `c` never escapes.
    /// Conservative: the boundaries are shrunk by `MARGIN`, so points near them fall back to the escape loop.
    /// Only valid for `orbit_radius >= 2.0`, since orbits of points inside the set stay within radius 2, but may leave a smaller radius.
    fn is_known_interior(&self, c: &Complex) -> bool {
        const MARGIN: f64 = 1e-9;
        if self.orbit_radius < 2.0 {
            return false;
        }
        let (x, y) = (c.x, c.y);
        let y_squared = y * y;
        let q = (x - 0.25) * (x - 0.25) + y_squared;
        let in_cardioid = q * (q + (x - 0.25)) <= 0.25 * y_squared - MARGIN;
        let in_bulb = (x + 1.0) * (x + 1.0) + y_squared <= 0.0625 - MARGIN;
        in_cardioid || in_bulb
    }

    /// The Mandelbrot set escape loop, without any shortcuts
    fn iterate_escape_loop(&self, c: &Complex) -> u32 {
        let mut z = Complex::new(0.0, 0.0);
        let mut iterations: u32 = 0;
        let orbit_radius_squared = self.orbit_radius*self.orbit_radius;
        for _ in 0..self.max_iterations {
            z = z.squared().add(c);

            if (z.x * z.x + z.y * z.y) > orbit_radius_squared { //Optimization: square both sides of the Mandelbrot set function, saves us taking the square root
                break;
            }
            iterations += 1;
        }
        iterations
    }
}

impl std::fmt::Debug for MandelbrotSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "max_iterations = {}, orbit_radius = {}", self.max_iterations, self.orbit_radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_color_cycle_keeps_the_color_of_escaped_points_when_the_limit_changes() {
        let f = crate::coloring::TrueColor::new_from_bernstein_polynomials;
        let rgb = |c: crate::coloring::TrueColor| (c.red, c.green, c.blue);
        let stretched = MandelbrotSet::new(1000, 2.0);
        for i in [0, 1, 17, 500, 999, 1000] {
            assert_eq!(rgb(stretched.color(f, i)), rgb(f(f64::from(i), 1000.0)), "stretched over the limit: as before");
        }
        let (mut low, mut high) = (MandelbrotSet::new(400, 2.0), MandelbrotSet::new(9000, 2.0));
        (low.palette, high.palette) = (ColorCycle::CYCLE, ColorCycle::CYCLE);
        for i in [0, 1, 17, 200, 399] {
            assert_eq!(rgb(low.color(f, i)), rgb(high.color(f, i)), "iteration {i} keeps its color");
            assert_eq!(rgb(high.color(f, i)), rgb(high.color(f, i + 500)), "the palette repeats every {PALETTE_LENGTH} iterations");
        }
        assert_eq!(rgb(low.color(f, 400)), rgb(high.color(f, 9000)), "inside the set has the same color at any limit");
        //The divide slows the colors down, the offset shifts them
        let mut slow = high;
        slow.palette = ColorCycle { divide: 4.0, offset: 10.0, stretch: false };
        assert_eq!(rgb(slow.color(f, 400)), rgb(f(110.0, PALETTE_LENGTH)));
        assert_eq!(rgb(slow.color(f, 2000)), rgb(f(10.0, PALETTE_LENGTH)), "around again after 4 x 500 iterations");
    }

    #[test]
    fn a_blend_of_color_cycles_goes_the_short_way_round_and_in_equal_ratios() {
        let (a, b) = (ColorCycle { divide: 1.0, offset: 480.0, stretch: false }, ColorCycle { divide: 100.0, offset: 20.0, stretch: false });
        let half = a.blend(b, 0.5);
        assert!((half.divide - 10.0).abs() < 1e-9 && half.offset.abs() < 1e-9, "{half:?}");
        assert_eq!(a.blend(b, 0.0), a);
        assert!((a.blend(b, 1.0).divide - 100.0).abs() < 1e-9 && (a.blend(b, 1.0).offset - 20.0).abs() < 1e-9);
    }

    const ORBIT_RADII: [f64; 3] = [2.0, 4.0, 1.0];

    #[test]
    fn known_interior_points() {
        let m = MandelbrotSet::new(1000, 2.0);
        for (x, y) in [(0.0, 0.0), (-0.3, 0.0), (0.2, 0.3), (-1.0, 0.0), (-1.1, 0.1), (-0.9, -0.1)] {
            let c = Complex::new(x, y);
            assert!(m.is_known_interior(&c), "({x}, {y}) should be known interior");
            assert_eq!(m.iterate(&c), 1000);
            assert_eq!(m.iterate_escape_loop(&c), 1000);
        }
    }

    #[test]
    fn exterior_and_boundary_points_fall_back_to_the_escape_loop() {
        let m = MandelbrotSet::new(1000, 2.0);
        for (x, y) in [(1.0, 1.0), (0.5, 0.5), (-2.0, 0.0), (0.26, 0.0), (-0.75, 0.1), (-1.25, 0.0), (-0.75, 0.0), (0.25, 0.5)] {
            let c = Complex::new(x, y);
            assert_eq!(m.iterate(&c), m.iterate_escape_loop(&c), "({x}, {y})");
        }
        // Just outside the cardioid on the real axis, and just outside the bulb
        assert!(!m.is_known_interior(&Complex::new(0.2500001, 0.0)));
        assert!(!m.is_known_interior(&Complex::new(-1.2500001, 0.0)));
        assert!(!m.is_known_interior(&Complex::new(-1.25, 0.0))); //On the boundary, margin makes it fall back
    }

    #[test]
    fn small_orbit_radius_never_uses_the_shortcut() {
        let m = MandelbrotSet::new(1000, 1.0);
        assert!(!m.is_known_interior(&Complex::new(0.0, 0.0)));
    }

    /// Compares `iterate` with the original escape loop on a dense grid (default view and boundary-focused views),
    /// for several iteration limits and orbit radii. Samples are jittered like supersampling samples.
    #[test]
    fn iterate_matches_escape_loop() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let views = [(-2.5, 1.0, -1.0, 1.0), (-1.30, -1.20, -0.05, 0.05), (-0.80, -0.70, 0.05, 0.15), (0.20, 0.30, -0.05, 0.05)];
        for max_iterations in [1, 10, 100, 1000] {
            for orbit_radius in ORBIT_RADII {
                let m = MandelbrotSet::new(max_iterations, orbit_radius);
                for (min_x, max_x, min_y, max_y) in views {
                    let n = 150;
                    for i in 0..n {
                        for j in 0..n {
                            let jitter: (f64, f64) = rng.gen();
                            let x = min_x + (max_x - min_x) * (f64::from(i) + jitter.0) / f64::from(n);
                            let y = min_y + (max_y - min_y) * (f64::from(j) + jitter.1) / f64::from(n);
                            let c = Complex::new(x, y);
                            assert_eq!(m.iterate(&c), m.iterate_escape_loop(&c), "c = ({x}, {y}), max_iterations = {max_iterations}, orbit_radius = {orbit_radius}");
                        }
                    }
                }
            }
        }
    }

    /// The lockstep and the refilling loop must give exactly the count of `iterate` for every point (integer coloring), for any mix of interior, exterior and boundary points
    #[test]
    fn lockstep_counts_match_iterate() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let views = [(-0.75, 0.0, 1.25), (-0.7498100000000001, -0.0203, 0.00008), (-1.7862712, 0.0000524, 0.00002), (0.9, 0.9, 0.4), (-0.1, 0.65, 0.12)];
        for max_iterations in [0, 1, 7, 100, 1000, 3000] {
            for orbit_radius in [2.0, 4.0, 1.0] {
                let mut m = MandelbrotSet::new(max_iterations, orbit_radius);
                m.smooth_coloring = false;
                for (x, y, half) in views {
                    for _ in 0..500 {
                        let points: [Complex; LANES] = std::array::from_fn(|_| Complex::new(x + (rng.gen::<f64>() - 0.5) * 2.0 * half, y + (rng.gen::<f64>() - 0.5) * 2.0 * half));
                        let expected: Vec<u32> = points.iter().map(|point| m.iterate(point)).collect();
                        assert_eq!(lockstep_counts(&m, &points), expected, "max_iterations = {max_iterations}, orbit_radius = {orbit_radius}, points = {:?}", points.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>());
                    }
                }
            }
        }
    }

    /// Points inside bulbs other than the main cardioid and the period-2 bulb never escape, and their orbits are found to repeat: the result must still be `max_iterations`
    #[test]
    fn lockstep_counts_match_iterate_inside_other_bulbs_and_deep_in_the_set() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        //Centers of the period-3 bulbs and a period-4 bulb, and a minibrot; the points are scattered around them
        let centers = [(-0.12256116687665362, 0.7448617666197442), (-0.12256116687665362, -0.7448617666197442), (-0.5043401, 0.5627655), (-1.7548776662466927, 0.0), (-1.3107, 0.0)];
        for max_iterations in [50, 1000, 4000] {
            let mut m = MandelbrotSet::new(max_iterations, 2.0);
            m.smooth_coloring = false;
            for (x, y) in centers {
                for half in [0.002, 0.02, 0.1] {
                    for _ in 0..300 {
                        let points: [Complex; LANES] = std::array::from_fn(|_| Complex::new(x + (rng.gen::<f64>() - 0.5) * 2.0 * half, y + (rng.gen::<f64>() - 0.5) * 2.0 * half));
                        let expected: Vec<u32> = points.iter().map(|point| m.iterate(point)).collect();
                        assert_eq!(lockstep_counts(&m, &points), expected, "max_iterations = {max_iterations}, points = {:?}", points.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>());
                    }
                }
            }
        }
        //The center of the period-3 bulb is found to repeat, and iterates to the limit
        let mut m = MandelbrotSet::new(3000, 2.0);
        m.smooth_coloring = false;
        let center = Complex::new(-0.12256116687665362, 0.7448617666197442);
        let points: [Complex; LANES] = std::array::from_fn(|_| Complex::new(center.x, center.y));
        assert_eq!(lockstep_counts(&m, &points), [3000; LANES]);
    }

    /// The iteration counts of `position_lanes` with integer coloring (`m.smooth_coloring` off), where the position is the count
    fn lockstep_counts(m: &MandelbrotSet, points: &[Complex; LANES]) -> Vec<u32> {
        m.position_lanes(points).iter().map(|position| *position as u32).collect()
    }

    /// `positions` (the refilling loop) must give exactly the positions of `position_lanes` (the lockstep loop), bit for bit, for any mix of points, any limit
    /// (also limits that are not a multiple of the block of steps, and below it), orbit radius, and smooth or integer coloring, and any number of points
    #[test]
    fn positions_match_the_lockstep_loop_bit_for_bit() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let views = [(-0.75, 0.0, 1.25), (-0.7498100000000001, -0.0203, 0.00008), (-1.7862712, 0.0000524, 0.00002), (0.9, 0.9, 0.4), (-0.12256116687665362, 0.7448617666197442, 0.02), (-1.3107, 0.0, 0.05)];
        for max_iterations in [0, 1, 3, 4, 5, 7, 100, 321, 1000, 3001] {
            for orbit_radius in [2.0, 4.0, 1.0] {
                for smooth in [true, false] {
                    let mut m = MandelbrotSet::new(max_iterations, orbit_radius);
                    m.smooth_coloring = smooth;
                    for (x, y, half) in views {
                        let count = rng.gen_range(0..300);
                        let points: Vec<Complex> = (0..count).map(|_| Complex::new(x + (rng.gen::<f64>() - 0.5) * 2.0 * half, y + (rng.gen::<f64>() - 0.5) * 2.0 * half)).collect();
                        let expected: Vec<u64> = points.chunks(LANES)
                            .flat_map(|group| <&[Complex; LANES]>::try_from(group).map_or_else(|_| group.iter().map(|c| m.position(c)).collect(), |group| m.position_lanes(group).to_vec()))
                            .map(f64::to_bits).collect();
                        //The AVX2 loop with 2 and (with AVX-512) 3 vectors where the CPU has them, and the generic loop
                        for way in ["avx2", "wide", "generic"] {
                            #[cfg(target_arch = "x86_64")]
                            let (avx2, wide) = (std::arch::is_x86_feature_detected!("avx2"), std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512vl"));
                            #[cfg(not(target_arch = "x86_64"))]
                            let (avx2, wide) = (false, false);
                            let mut out = vec![f64::NAN; points.len()];
                            let done = match way {
                                //SAFETY: the CPU has the instructions, which was just detected
                                #[cfg(target_arch = "x86_64")]
                                "avx2" if avx2 => unsafe { avx2::positions(&m, &points, &mut out, None, false) },
                                #[cfg(target_arch = "x86_64")]
                                "wide" if wide => unsafe { avx2::positions(&m, &points, &mut out, None, true) },
                                "generic" => m.positions_generic(&points, &mut out, None),
                                _ => continue,
                            };
                            assert_eq!(done, points.len());
                            let differ = out.iter().zip(&expected).position(|(a, b)| a.to_bits() != *b);
                            assert!(differ.is_none(), "{way}, max_iterations = {max_iterations}, orbit_radius = {orbit_radius}, smooth = {smooth}, point {:?}: {} instead of {}",
                                differ.map(|at| (points[at].x, points[at].y)), differ.map_or(0.0, |at| out[at]), differ.map_or(0.0, |at| f64::from_bits(expected[at])));
                        }
                    }
                }
            }
        }
    }

    /// A cancelled `positions` returns how many points from the start have their position, and those are right
    #[test]
    fn cancelled_positions_are_a_complete_prefix() {
        let m = MandelbrotSet::new(100_000, 2.0);
        let points: Vec<Complex> = (0..400).map(|i| Complex::new(-0.75 + f64::from(i) * 1e-4, 0.1 + f64::from(i % 7) * 1e-3)).collect();
        let cancel = AtomicBool::new(true);
        let mut out = vec![f64::NAN; points.len()];
        let done = m.positions(&points, &mut out, Some(&cancel));
        assert!(done < points.len(), "cancelled at once: {done}");
        assert!(out[..done].iter().zip(&points).all(|(position, c)| position.to_bits() == m.position(c).to_bits()));
    }

    fn test_points() -> Vec<Complex> {
        let mut points = Vec::new();
        for (min_x, max_x, min_y, max_y) in [(-2.5, 1.0, -1.0, 1.0), (-0.80, -0.70, 0.05, 0.15), (-1.30, -1.20, -0.05, 0.05), (0.20, 0.30, -0.05, 0.05)] {
            for i in 0..60 {
                for j in 0..60 {
                    points.push(Complex::new(min_x + (max_x - min_x) * f64::from(i) / 60.0, min_y + (max_y - min_y) * f64::from(j) / 60.0));
                }
            }
        }
        points
    }

    #[test]
    fn smooth_position_keeps_the_integer_count_and_the_interior_classification() {
        for max_iterations in [1, 10, 300] {
            for orbit_radius in ORBIT_RADII {
                let m = MandelbrotSet::new(max_iterations, orbit_radius);
                for c in test_points() {
                    let (count, position) = (m.iterate(&c), m.iterate_smooth(&c));
                    if count == max_iterations {
                        assert_eq!(position, f64::from(max_iterations), "points that do not escape stay at the limit: {c:?}");
                    } else {
                        assert!(position >= 0.0 && position <= f64::from(max_iterations - 1) && position >= f64::from(count) - 2.0, "{c:?}: count {count}, position {position}");
                    }
                }
            }
        }
    }

    #[test]
    fn smooth_positions_of_the_simd_and_scalar_paths_are_identical() {
        let points = test_points();
        for max_iterations in [1, 7, 300] {
            let m = MandelbrotSet::new(max_iterations, 2.0);
            for group in points.as_chunks::<LANES>().0 {
                let group: [Complex; LANES] = std::array::from_fn(|lane| Complex::new(group[lane].x, group[lane].y));
                let lanes = m.position_lanes(&group);
                let generic = m.iterate_lanes_smooth_dispatch(&group);
                for lane in 0..LANES {
                    assert_eq!(lanes[lane].to_bits(), m.iterate_smooth(&group[lane]).to_bits(), "{:?}", group[lane]);
                    assert_eq!(generic.iterations[lane], m.iterate(&group[lane]));
                }
            }
        }
    }

    /// The loops that are compiled with AVX2 are the same code as the generic loops that CPUs without AVX2 (and other architectures) run: this checks that they give the same bits.
    /// It can only compare them on a CPU that has AVX2, where `position_lanes` picks the AVX2 loop; elsewhere both sides are the generic loop and the test passes trivially.
    #[test]
    fn the_avx2_loops_give_bit_identical_results_to_the_generic_loops() {
        let points = test_points();
        let mut compared = 0;
        for max_iterations in [1, 7, 300, 2000] {
            let m = MandelbrotSet::new(max_iterations, 2.0);
            for group in points.as_chunks::<LANES>().0 {
                let group: [Complex; LANES] = std::array::from_fn(|lane| Complex::new(group[lane].x, group[lane].y));
                let dispatched = m.iterate_lanes_smooth_dispatch(&group);
                let mut generic = LaneResult { iterations: [0; LANES], escaped: [false; LANES], escape_x: [0.0; LANES], escape_y: [0.0; LANES] };
                m.iterate_lanes_smooth_generic(&group, &mut generic);
                assert_eq!((dispatched.iterations, dispatched.escaped), (generic.iterations, generic.escaped), "{group:?}");
                for lane in 0..LANES {
                    assert_eq!((dispatched.escape_x[lane].to_bits(), dispatched.escape_y[lane].to_bits()), (generic.escape_x[lane].to_bits(), generic.escape_y[lane].to_bits()), "{group:?}");
                }
                compared += 1;
            }
        }
        assert!(compared > 100, "{compared} groups were compared");
    }

    #[test]
    fn smooth_position_is_continuous_where_the_integer_count_jumps() {
        let m = MandelbrotSet::new(1000, 2.0);
        let positions: Vec<(u32, f64)> = (0..4000).map(|i| { let c = Complex::new(-0.5 + f64::from(i) * 1.25e-4, 0.9); (m.iterate(&c), m.iterate_smooth(&c)) }).collect();
        //Only where the bands are wide enough to be sampled well, i.e. not right at the boundary of the set
        let pairs: Vec<_> = positions.windows(2).filter(|w| w[0].0 <= 8 && w[1].0 <= 8).collect();
        let jumps = pairs.iter().filter(|w| w[0].0 != w[1].0).count();
        let largest_step = pairs.iter().map(|w| (w[1].1 - w[0].1).abs()).fold(0.0, f64::max);
        assert!(jumps >= 5, "the line crosses several integer bands: {jumps}");
        assert!(largest_step < 0.2, "the smooth position changes gradually across the bands: {largest_step}");
    }

    #[test]
    fn interpolated_color_mixes_neighbors_and_keeps_whole_positions() {
        use crate::coloring::TrueColor;
        let m = MandelbrotSet::new(10, 2.0);
        let lookup = |i: u32| TrueColor::new((i * 10) as u8, 200, (255 - i * 10) as u8);
        let whole = m.interpolated_color(3.0, lookup);
        assert_eq!((whole.red, whole.green, whole.blue), (30, 200, 225));
        let half = m.interpolated_color(3.5, lookup);
        assert_eq!((half.red, half.green, half.blue), (35, 200, 220));
        let inside = m.interpolated_color(10.0, lookup);
        assert_eq!((inside.red, inside.green, inside.blue), (100, 200, 155), "the limit itself is never mixed with a color beyond it");
    }
}

//! Perturbation theory with bilinear approximation (BLA), an experimental alternative to iterating every pixel directly.
//!
//! One reference orbit `Z_n` is iterated for the center of the view. A pixel with offset `dc` from the center follows `z_n = Z_n + dz_n` where
//! `dz_{n+1} = (2 Z_n + dz_n) dz_n + dc`. Where `|dz|` is small compared to `|Z_n|`, many steps can be replaced by one linear step
//! `dz -> A dz + B dc` (a bilinear approximation, merged in a binary tree so skips of 2^k steps are available).
//! Rebasing (`z` becomes the new `dz` and the reference restarts) replaces glitch detection, so one reference orbit serves every pixel.
//!
//! Unlike `MandelbrotSet::iterate`, the result is not bit-identical: the rounding errors are different, so pixels whose orbit is chaotic
//! may escape at a different step. See `examples/perturbation.rs` for the measured differences.
use crate::complex::Complex;

#[derive(Clone, Copy)]
struct Bla {
    ax: f64, ay: f64, //A: dz -> A dz
    bx: f64, by: f64, //B: dc -> B dc
    ///Squared radius of |dz| below which the linear step is valid
    r2: f64,
    ///Largest |Z| inside the span, and bounds on how |dz| can grow inside the span: |dz_k| <= growth |dz| + drift |dc|
    z_max: f64,
    growth: f64,
    drift: f64,
}

pub struct Reference {
    orbit: Vec<(f64, f64)>, //Z_0..=Z_len, with Z_len the last value that is inside the orbit radius
    ///levels[k][i] skips 2^k steps starting at reference index i*2^k
    levels: Vec<Vec<Bla>>,
    orbit_radius: f64,
    max_iterations: u32,
    pub bla_epsilon: f64,
}

impl Reference {
    /// `dc_max` is the largest |dc| of any pixel (sample) the reference will be used for, `bla_epsilon` the allowed relative error of one linear step
    pub fn new(center: &Complex, max_iterations: u32, orbit_radius: f64, dc_max: f64, bla_epsilon: f64) -> Reference {
        let r2 = orbit_radius * orbit_radius;
        let mut orbit = Vec::with_capacity(max_iterations as usize + 1);
        let (mut zx, mut zy) = (0.0_f64, 0.0_f64);
        orbit.push((zx, zy));
        for _ in 0..max_iterations {
            let nx = zx * zx - zy * zy + center.x;
            let ny = 2.0 * zx * zy + center.y;
            if nx * nx + ny * ny > r2 { break; }
            (zx, zy) = (nx, ny);
            orbit.push((zx, zy));
        }
        let len = orbit.len() - 1;
        //Level 0: one step from Z_i: A = 2 Z_i, B = 1, valid while the quadratic term |dz|^2 is below epsilon * |2 Z_i dz|, i.e. |dz| < epsilon |2 Z_i|
        let level0: Vec<Bla> = orbit[..len].iter().map(|&(x, y)| {
            let a = (2.0 * x, 2.0 * y);
            let a_abs = a.0.hypot(a.1);
            let r = bla_epsilon * a_abs;
            Bla { ax: a.0, ay: a.1, bx: 1.0, by: 0.0, r2: r * r, z_max: x.hypot(y), growth: a_abs + r, drift: 1.0 }
        }).collect();
        let mut levels = vec![level0];
        loop {
            let prev = levels.last().unwrap();
            if prev.len() < 2 { break; }
            let next: Vec<Bla> = (0..prev.len() / 2).map(|i| merge(&prev[2 * i], &prev[2 * i + 1], dc_max)).collect();
            levels.push(next);
        }
        Reference { orbit, levels, orbit_radius, max_iterations, bla_epsilon }
    }

    /// The iteration count of the pixel with offset `dc` from the reference point (`center` + `dc` is the pixel), with `use_bla` choosing between plain perturbation and BLA skipping.
    /// Returns `(iterations, linear_steps)`; the second value only exists for the statistics of the benchmark: the amount of loop passes.
    pub fn iterate(&self, dc: (f64, f64), use_bla: bool) -> (u32, u32) {
        let r2 = self.orbit_radius * self.orbit_radius;
        let len = self.orbit.len() - 1;
        let (mut dx, mut dy) = (0.0_f64, 0.0_f64);
        let mut n = 0usize; //Reference index, always below `len` at the start of a pass (rebased otherwise)
        let mut iterations = 0u32; //Steps taken, equal to the iteration of the original loop
        let mut passes = 0u32;
        let dc_abs = dc.0.hypot(dc.1);
        while iterations < self.max_iterations {
            passes += 1;
            if use_bla && len > 0 {
                let remaining = (self.max_iterations - iterations) as usize;
                let d_abs2 = dx * dx + dy * dy;
                let d_abs = d_abs2.sqrt();
                //Largest skip first: aligned to the reference index (2^level divides n), inside the reference and the iteration limit
                let top = if n == 0 { self.levels.len() - 1 } else { (n.trailing_zeros() as usize).min(self.levels.len() - 1) };
                let mut skipped = false;
                for level in (0..=top).rev() {
                    let skip = 1usize << level;
                    if n + skip > len || skip > remaining { continue; }
                    let b = &self.levels[level][n >> level];
                    if d_abs2 < b.r2 && b.z_max + b.growth * d_abs + b.drift * dc_abs < self.orbit_radius * 0.999 {
                        let nx = b.ax * dx - b.ay * dy + b.bx * dc.0 - b.by * dc.1;
                        let ny = b.ax * dy + b.ay * dx + b.bx * dc.1 + b.by * dc.0;
                        (dx, dy) = (nx, ny);
                        n += skip;
                        iterations += skip as u32;
                        skipped = true;
                        break;
                    }
                }
                if skipped {
                    let (zx, zy) = (self.orbit[n].0 + dx, self.orbit[n].1 + dy);
                    if n >= len || zx * zx + zy * zy < dx * dx + dy * dy { (dx, dy) = (zx, zy); n = 0; }
                    continue;
                }
            }
            //Single step: dz' = (2 Z_n + dz) dz + dc
            let (zr, zi) = self.orbit[n];
            let (tx, ty) = (2.0 * zr + dx, 2.0 * zi + dy);
            let nx = tx * dx - ty * dy + dc.0;
            let ny = tx * dy + ty * dx + dc.1;
            n += 1;
            let (zx, zy) = (self.orbit[n].0 + nx, self.orbit[n].1 + ny);
            let z_abs2 = zx * zx + zy * zy;
            if z_abs2 > r2 {
                return (iterations, passes);
            }
            iterations += 1;
            if n >= len || z_abs2 < nx * nx + ny * ny {
                (dx, dy) = (zx, zy); //Rebase: the full value becomes the offset from the restarted reference
                n = 0;
            } else {
                (dx, dy) = (nx, ny);
            }
        }
        (self.max_iterations, passes)
    }
}

fn merge(x: &Bla, y: &Bla, dc_max: f64) -> Bla {
    let (ax, ay) = (y.ax * x.ax - y.ay * x.ay, y.ax * x.ay + y.ay * x.ax);
    let (bx, by) = (y.ax * x.bx - y.ay * x.by + y.bx, y.ax * x.by + y.ay * x.bx + y.by);
    let x_a = x.ax.hypot(x.ay);
    let x_b = x.bx.hypot(x.by);
    //Valid if dz is valid for the first half, and the result of the first half is valid for the second half
    let r_y = y.r2.sqrt();
    let r = x.r2.sqrt().min(((r_y - x_b * dc_max) / x_a).max(0.0));
    let growth = x.growth.max(y.growth * x_a);
    let drift = x.drift.max(y.growth * x_b + y.drift);
    Bla { ax, ay, bx, by, r2: r * r, z_max: x.z_max.max(y.z_max), growth, drift }
}

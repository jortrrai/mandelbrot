//! The spiral route of a flight (issue #68): the camera spirals into (or out of) the place the way the set itself spirals there.
//!
//! Spiral arms of the Mandelbrot set wind around Misiurewicz points: parameters `c₀` whose critical orbit (of 0) becomes periodic after a few steps. Near such a
//! point the set is asymptotically self-similar (Tan Lei): zooming in by `|λ|` turns the picture by `arg λ`, where `λ` is the multiplier of the repelling cycle
//! of `c₀`, so its arms are the logarithmic spirals `c₀ + d · λ^(-t)`. The route finds a Misiurewicz point near the destination (Newton's method over small
//! preperiods and periods), and moves the center of the camera along such a spiral around it while zooming by the same `|λ|^t`, bent a little so that it ends
//! exactly at the destination. The view itself does not rotate (the renderer has no rotated views): the picture slides around `c₀` the way its arms wind.
//!
//! Without a spiral center near the destination, or with a start too far from it (the camera would sweep across the set), the flight takes the shortest route.

use super::flight::Camera;

/// A complex number as (re, im), for the little arithmetic here
type C = (f64, f64);

const fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

const fn sub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}

fn mul(a: C, b: C) -> C {
    (a.0.mul_add(b.0, -(a.1 * b.1)), a.0.mul_add(b.1, a.1 * b.0))
}

fn div(a: C, b: C) -> C {
    let d = b.0.mul_add(b.0, b.1 * b.1);
    ((a.0.mul_add(b.0, a.1 * b.1)) / d, (a.1.mul_add(b.0, -(a.0 * b.1))) / d)
}

fn abs(a: C) -> f64 {
    a.0.hypot(a.1)
}

/// The principal logarithm
fn ln(a: C) -> C {
    (abs(a).ln(), a.1.atan2(a.0))
}

fn exp(a: C) -> C {
    let r = a.0.exp();
    (r * a.1.cos(), r * a.1.sin())
}

/// How far a usable spiral turns per e-fold of zoom, in radians (so a zoom by 10 000 turns it 3 to 14 radians, half a turn to two turns)
const TURN_PER_ZOOM: std::ops::RangeInclusive<f64> = 0.3..=1.5;

/// The largest preperiod and period that the search tries
const MAX_PREPERIOD: u32 = 48;
const MAX_PERIOD: u32 = 6;

/// A spiral center is only used within this many view widths of the destination
const NEAR: f64 = 2.0;

/// A spiral center of the set: a Misiurewicz point `c` and the multiplier `lambda` of its repelling cycle
#[derive(Clone, Copy, Debug)]
pub struct SpiralCenter {
    pub c: (f64, f64),
    pub lambda: (f64, f64),
}

/// The Misiurewicz point nearest to `near` within `radius` (preperiod 1 to `MAX_PREPERIOD`, period 1 to `MAX_PERIOD`) whose spiral turns by `TURN_PER_ZOOM`,
/// found by Newton's method on `f^(k+p)(0) - f^k(0) = 0`, with the multiplier of its cycle; `None` if no search converges to one that close
pub fn center_near(near: C, radius: f64) -> Option<SpiralCenter> {
    let mut best: Option<(f64, SpiralCenter)> = None;
    let small = (1..=MAX_PREPERIOD).flat_map(|preperiod| (1..=MAX_PERIOD).map(move |period| (preperiod, period)));
    for (preperiod, period) in small.chain(near_repeats(near)) {
        {
            let Some(center) = newton(near, preperiod, period) else { continue };
            let distance = abs(sub(center.c, near));
            //A spiral the camera can follow: it turns visibly, but not wildly (a multiplier close to 1 would turn dozens of times on the way)
            let turn = center.lambda.1.atan2(center.lambda.0).abs() / abs(center.lambda).ln();
            if distance <= radius && TURN_PER_ZOOM.contains(&turn) && best.as_ref().is_none_or(|(nearest, _)| distance < *nearest) {
                best = Some((distance, center));
            }
        }
    }
    best.map(|(_, center)| center)
}

/// The preperiods and periods where the orbit of 0 under `c` almost repeats (`z_(n+p)` closest to `z_n`), the closest first: near a Misiurewicz point the
/// orbit follows its preperiodic orbit for a long while, so these are the preperiod and period of the spiral centers near a deep `c`, where Newton's method from
/// `c` converges (from a guess of a small preperiod it rarely does)
fn near_repeats(c: C) -> Vec<(u32, u32)> {
    let mut orbit = vec![(0.0, 0.0)];
    let mut z = (0.0, 0.0);
    for _ in 0..NEAR_REPEAT_STEPS {
        z = add(mul(z, z), c);
        if abs(z) > 2.0 {
            break;
        }
        orbit.push(z);
    }
    let mut repeats: Vec<(f64, u32, u32)> = Vec::new();
    for period in 1..=NEAR_REPEAT_PERIODS {
        for preperiod in 1..orbit.len().saturating_sub(period as usize) {
            let gap = abs(sub(orbit[preperiod + period as usize], orbit[preperiod]));
            repeats.push((gap, preperiod as u32, period));
        }
    }
    repeats.sort_by(|a, b| a.0.total_cmp(&b.0));
    repeats.into_iter().take(NEAR_REPEAT_TRIES).map(|(_, preperiod, period)| (preperiod, period)).collect()
}

/// How far the orbit is followed, the periods looked for, and how many of the closest repeats are tried, see `near_repeats`
const NEAR_REPEAT_STEPS: usize = 2000;
const NEAR_REPEAT_PERIODS: u32 = 8;
const NEAR_REPEAT_TRIES: usize = 40;

/// Newton's method for a Misiurewicz point of `preperiod` and `period` from `start`; the point if it converges to a repelling (`|λ| > 1`) one
fn newton(start: C, preperiod: u32, period: u32) -> Option<SpiralCenter> {
    let mut c = start;
    for _ in 0..64 {
        //z_n and dz_n/dc of the critical orbit, at the preperiod and at preperiod + period
        let (mut z, mut dz) = ((0.0, 0.0), (0.0, 0.0));
        let (mut at_k, mut dz_at_k) = ((0.0, 0.0), (0.0, 0.0));
        for n in 1..=preperiod + period {
            dz = add(mul((2.0 * z.0, 2.0 * z.1), dz), (1.0, 0.0));
            z = add(mul(z, z), c);
            if abs(z) > 1e6 {
                return None;
            }
            if n == preperiod {
                (at_k, dz_at_k) = (z, dz);
            }
        }
        let (g, dg) = (sub(z, at_k), sub(dz, dz_at_k));
        if abs(dg) == 0.0 {
            return None;
        }
        let step = div(g, dg);
        c = sub(c, step);
        if abs(step) <= 1e-15 * abs(c).max(1e-300) {
            break;
        }
    }
    //The orbit must really be preperiodic (not a root of a smaller preperiod or a periodic center), with a repelling cycle
    let mut z = (0.0, 0.0);
    let mut orbit = Vec::with_capacity((preperiod + period + 1) as usize);
    for _ in 0..=preperiod + period {
        orbit.push(z);
        z = add(mul(z, z), c);
    }
    let (k, p) = (preperiod as usize, period as usize);
    if abs(sub(orbit[k + p], orbit[k])) > 1e-9 || abs(sub(orbit[k - 1 + p], orbit[k - 1])) < 1e-9 {
        return None;
    }
    let lambda = orbit[k..k + p].iter().fold((1.0, 0.0), |product, z| mul(product, (2.0 * z.0, 2.0 * z.1)));
    (abs(lambda) > 1.0 + 1e-6 && c.0.is_finite() && c.1.is_finite()).then_some(SpiralCenter { c, lambda })
}

/// The spiral path from `from` to `to` around `center`, see the module
#[derive(Clone, Debug)]
pub struct SpiralPath {
    from: Camera,
    to: Camera,
    center: C,
    /// The offset of the start's center from the spiral center
    start_offset: C,
    /// `ln(λ) · turns`: the spiral, and `ln(correction)`: how it is bent to end at the destination (see `camera_at`)
    spiral: C,
    correction: C,
}

impl SpiralPath {
    /// The spiral path from `from` to `to`, or `None` without a spiral center near `to`, or with a start or destination too far from it
    pub fn new(from: Camera, to: Camera) -> Option<SpiralPath> {
        let width = |camera: Camera| 2.5 * camera.scale;
        let found = center_near((to.x, to.y), NEAR * width(to))?;
        let (start_offset, end_offset) = (sub((from.x, from.y), found.c), sub((to.x, to.y), found.c));
        if abs(start_offset) > NEAR * width(from) || abs(start_offset) == 0.0 || abs(end_offset) == 0.0 {
            return None; //The camera would sweep across the set, or there is nothing to spiral around
        }
        //How many times the zoom is |λ|: the spiral turns by arg λ that many times
        let turns = (from.scale / to.scale).ln() / abs(found.lambda).ln();
        let spiral = mul(ln(found.lambda), (turns, 0.0));
        //Where the pure spiral ends, and the rest of the way to the destination, spread over the flight
        let pure_end = mul(start_offset, exp((-spiral.0, -spiral.1)));
        let correction = ln(div(end_offset, pure_end));
        Some(SpiralPath { from, to, center: found.c, start_offset, spiral, correction })
    }

    /// The camera at the share `u` (0 to 1) of the way: exactly the start at 0 and the destination at 1
    pub fn camera_at(&self, u: f64) -> Camera {
        if u <= 0.0 {
            return self.from;
        }
        if u >= 1.0 {
            return self.to;
        }
        let scale = self.from.scale * (self.to.scale / self.from.scale).powf(u);
        //offset(u) = start_offset · λ^(-turns · u) · correction^u
        let exponent = (u.mul_add(-self.spiral.0, u * self.correction.0), u.mul_add(-self.spiral.1, u * self.correction.1));
        let (x, y) = add(self.center, mul(self.start_offset, exp(exponent)));
        Camera { x, y, scale }
    }

    /// The spiral center
    #[cfg(test)]
    pub const fn center(&self) -> (f64, f64) {
        self.center
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_finds_a_known_misiurewicz_point_with_a_repelling_cycle() {
        //c = i: 0 -> i -> -1 + i -> -i -> -1 + i, preperiod 1 (from z_1 on the orbit is periodic... from z_2) and period 2
        let found = center_near((0.01, 0.99), 0.1).expect("c = i is near");
        assert!(abs(sub(found.c, (0.0, 1.0))) < 1e-12, "{found:?}");
        assert!(abs(found.lambda) > 1.0);
        //Nothing within a tiny radius of a point far from any low-preperiod spiral center
        assert!(center_near((0.25, 0.0), 1e-9).is_none(), "the cusp of the cardioid is no Misiurewicz point of these preperiods");
    }

    #[test]
    fn a_spiral_starts_at_the_start_ends_at_the_destination_and_keeps_its_center_in_view() {
        let center = center_near((-0.1011, 0.9563), 0.01).expect("a spiral center near the tip at -0.1011 + 0.9563i");
        let to = Camera { x: center.c.0 + 1e-5, y: center.c.1 - 2e-5, scale: 4e-5 };
        let from = Camera { x: center.c.0 + 0.02, y: center.c.1 + 0.01, scale: 0.05 };
        let path = SpiralPath::new(from, to).expect("near enough for a spiral");
        assert_eq!(path.camera_at(0.0), from);
        assert_eq!(path.camera_at(1.0), to);
        let mut last_scale = from.scale;
        for i in 1..100 {
            let camera = path.camera_at(f64::from(i) / 100.0);
            assert!(camera.scale < last_scale, "zooms in all the way");
            last_scale = camera.scale;
            let distance = (camera.x - path.center().0).hypot(camera.y - path.center().1);
            assert!(distance <= 2.0 * 2.5 * camera.scale, "the spiral center stays within two view widths: {distance} at scale {}", camera.scale);
        }
        //Too far: the shortest route
        assert!(SpiralPath::new(Camera { x: 0.3, y: 0.0, scale: 0.001 }, to).is_none());
    }
}

#[cfg(test)]
mod survey {
    #[test]
    #[ignore = "a survey of the shipped places, run by hand with --nocapture"]
    fn which_places_get_a_spiral() {
        let report = crate::points_of_interest::load_folder(std::path::Path::new("points_of_interest"));
        let reset = super::Camera { x: -0.75, y: 0.0, scale: 1.0 };
        for point in &report.points {
            let to = super::Camera { x: point.location.real, y: point.location.imaginary, scale: point.location.scale };
            let near = super::center_near((to.x, to.y), 2.0 * 2.5 * to.scale);
            let from_reset = super::SpiralPath::new(reset, to).is_some();
            println!("{:<26} scale {:>9.2e}: center {:?}, from the reset view {from_reset}", point.name, to.scale, near.map(|c| (c.c, super::abs(c.lambda), c.lambda.1.atan2(c.lambda.0))));
        }
    }
}


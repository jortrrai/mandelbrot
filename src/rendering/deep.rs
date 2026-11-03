//! Deep zoom (issue #141, prototype): views far deeper than 64-bit floats can show, by perturbation with rebasing.
//!
//! Below a scale of about 10⁻¹³ neighboring pixels no longer have different 64-bit coordinates. Perturbation computes one orbit, of the center of the view,
//! exactly (fixed point with as many bits as the depth needs, `Fixed`), and for every pixel only the small difference from that orbit, in 64-bit floats:
//! with `Z` the orbit of the center and `c` the pixel's offset from it, the pixel's orbit is `Z + z` with `z ← 2·Z·z + z² + c`. The differences are
//! tiny, but a 64-bit float holds numbers down to 10⁻³⁰⁸, so this reaches scales of 10⁻³⁰⁰.
//!
//! **Rebasing** (Zhuoran, 2021) keeps the differences from growing large compared with the orbit, which is what makes the classic method fail ("glitches"):
//! when the pixel's orbit comes closer to 0 than its difference from the center's orbit (`|Z + z| < |z|`), the difference becomes the pixel's orbit itself
//! and the center's orbit starts again from its beginning. One reference orbit then serves the whole view.
//!
//! The colors use the same smooth iteration count as the 64-bit renderer (see `MandelbrotSet::smooth_position`), so a deep view looks like a shallow one.

use num::bigint::BigInt;
use num::{One, Signed, ToPrimitive, Zero};

use crate::mandelbrot_set::MandelbrotSet;

/// A number in fixed point: `value / 2^bits`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fixed {
    value: BigInt,
    bits: u32,
}

impl Fixed {
    /// The number of the decimal text `text` (`-0.7498`, `1.5e-30`), with `bits` bits after the binary point, or `None` if it is not a number
    pub fn parse(text: &str, bits: u32) -> Option<Fixed> {
        let text = text.trim();
        let (mantissa, exponent) = match text.find(['e', 'E']) {
            Some(at) => (&text[..at], text[at + 1..].parse::<i64>().ok()?),
            None => (text, 0),
        };
        let (negative, digits) = mantissa.strip_prefix('-').map_or_else(|| (false, mantissa.strip_prefix('+').unwrap_or(mantissa)), |rest| (true, rest));
        let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
        if whole.is_empty() && fraction.is_empty() || !whole.chars().chain(fraction.chars()).all(|digit| digit.is_ascii_digit()) {
            return None;
        }
        //value = digits · 10^(exponent - fraction length) · 2^bits, rounded
        let all: BigInt = format!("{whole}{fraction}").trim_start_matches('0').parse().unwrap_or_else(|_| BigInt::zero());
        let ten_power = exponent - i64::try_from(fraction.len()).ok()?;
        let scaled = if ten_power >= 0 {
            (all * BigInt::from(10).pow(u32::try_from(ten_power).ok()?)) << bits
        } else {
            let divisor = BigInt::from(10).pow(u32::try_from(-ten_power).ok()?);
            ((all << (bits + 1)) / &divisor + 1) >> 1
        };
        Some(Fixed { value: if negative { -scaled } else { scaled }, bits })
    }

    /// The bits after the binary point
    pub const fn bits(&self) -> u32 {
        self.bits
    }

    /// The nearest 64-bit float
    pub fn to_f64(&self) -> f64 {
        //The top 64 bits as an integer, scaled by the rest: exact enough for a 64-bit result
        let shift = self.value.bits().saturating_sub(64);
        let top = (&self.value >> shift).to_f64().unwrap_or(0.0);
        //(One scaling by 2^(shift - bits): with more than ~1,000 bits both powers overflowed to infinity and their quotient was NaN, #263)
        times_two_to(top, i32::try_from(i64::from(u32::try_from(shift).unwrap_or(u32::MAX)) - i64::from(self.bits)).unwrap_or(i32::MIN))
    }

    /// This number plus `offset · 2^power`, exactly (a step of a view deeper than 64-bit floats reach, see `Depth`)
    pub fn plus_scaled(&self, offset: f64, power: i32) -> Fixed {
        let (shifted, power) = (Fixed { value: self.value.clone(), bits: self.bits }, i64::from(power));
        if offset == 0.0 || !offset.is_finite() {
            return shifted;
        }
        //offset = mantissa · 2^exponent exactly (a normal or subnormal float); in fixed point: mantissa · 2^(exponent + power + bits)
        let raw = offset.to_bits();
        let exponent = i64::from(((raw >> 52) & 0x7ff) as u16);
        let mantissa = if exponent == 0 { (raw & ((1 << 52) - 1)) << 1 } else { (raw & ((1 << 52) - 1)) | (1 << 52) };
        let shift = (exponent - 1075) + power + i64::from(self.bits);
        let magnitude = BigInt::from(mantissa);
        let scaled = if shift >= 0 { magnitude << u32::try_from(shift).unwrap_or(0) } else { magnitude >> u32::try_from(-shift).unwrap_or(u32::MAX) };
        let scaled = if offset < 0.0 { -scaled } else { scaled };
        Fixed { value: &self.value + scaled, bits: self.bits }
    }

    /// This number plus the 64-bit float `offset` (a step of the view: panning, zooming around a point), exactly
    pub fn plus(&self, offset: f64) -> Fixed {
        if offset == 0.0 || !offset.is_finite() {
            return self.clone();
        }
        //offset = mantissa · 2^exponent exactly; in fixed point: mantissa · 2^(exponent + bits)
        let bits = offset.to_bits();
        let exponent = i64::from(((bits >> 52) & 0x7ff) as u16);
        let mantissa = if exponent == 0 { (bits & ((1 << 52) - 1)) << 1 } else { (bits & ((1 << 52) - 1)) | (1 << 52) };
        let shift = (exponent - 1075) + i64::from(self.bits);
        let magnitude = BigInt::from(mantissa);
        let scaled = if shift >= 0 { magnitude << u32::try_from(shift).unwrap_or(0) } else { magnitude >> u32::try_from(-shift).unwrap_or(u32::MAX) };
        let scaled = if offset < 0.0 { -scaled } else { scaled };
        Fixed { value: &self.value + scaled, bits: self.bits }
    }

    /// The same number with `bits` bits after the binary point (more: exact; fewer: rounded down)
    pub fn with_bits(&self, bits: u32) -> Fixed {
        let value = if bits >= self.bits { &self.value << (bits - self.bits) } else { &self.value >> (self.bits - bits) };
        Fixed { value, bits }
    }

    /// The number as decimal text with enough digits to read back the same fixed-point number (`-0.7498...`)
    pub fn to_decimal(&self) -> String {
        let negative = self.value.is_negative();
        let magnitude = self.value.abs();
        let whole = &magnitude >> self.bits;
        let mut fraction = &magnitude - (&whole << self.bits);
        //A binary fraction of `bits` bits has exactly `bits` decimal digits; enough of them to tell neighbors apart: bits · log10(2) + 2
        let digits = (f64::from(self.bits) * std::f64::consts::LOG10_2).ceil() as usize + 2;
        let mut text = format!("{}{whole}.", if negative { "-" } else { "" });
        for _ in 0..digits {
            fraction *= 10;
            let digit = &fraction >> self.bits;
            fraction -= &digit << self.bits;
            text.push(char::from(b'0' + digit.to_u8().unwrap_or(0)));
        }
        let trimmed = text.trim_end_matches('0');
        if trimmed.ends_with('.') { format!("{trimmed}0") } else { trimmed.to_string() }
    }

    fn add(&self, other: &Fixed) -> Fixed {
        Fixed { value: &self.value + &other.value, bits: self.bits }
    }

    fn sub(&self, other: &Fixed) -> Fixed {
        Fixed { value: &self.value - &other.value, bits: self.bits }
    }

    fn mul(&self, other: &Fixed) -> Fixed {
        Fixed { value: (&self.value * &other.value) >> self.bits, bits: self.bits }
    }

    fn double(&self) -> Fixed {
        Fixed { value: &self.value << 1u32, bits: self.bits }
    }

    fn magnitude_above(&self, limit: u32) -> bool {
        self.value.abs() > (BigInt::one() << (self.bits + limit))
    }
}

/// The bits after the binary point that a view of `scale` and `width` pixels needs: those of a pixel, and 64 more for the arithmetic
pub fn bits_for(scale: f64, width: usize) -> u32 {
    let pixel = 2.5 * scale / width.max(1) as f64;
    (-pixel.log2()).ceil().max(0.0) as u32 + 64
}

/// The orbit of the center `(real, imaginary)` in 64-bit floats, computed exactly in fixed point, up to `max_iterations` steps or until it leaves radius
/// `2^8` (the orbit of a point outside the set stops there)
pub fn reference_orbit(real: &Fixed, imaginary: &Fixed, max_iterations: u32) -> Vec<(f64, f64)> {
    let mut z = (Fixed { value: BigInt::zero(), bits: real.bits }, Fixed { value: BigInt::zero(), bits: real.bits });
    let mut orbit = Vec::with_capacity(max_iterations as usize + 1);
    orbit.push((0.0, 0.0));
    continue_orbit((real, imaginary), &mut z, &mut orbit, max_iterations);
    orbit
}

/// Continues the orbit of `c` from `z` (its last point, in fixed point) for up to `steps` steps, adding them to `orbit`; whether it escaped
fn continue_orbit((real, imaginary): (&Fixed, &Fixed), z: &mut (Fixed, Fixed), orbit: &mut Vec<(f64, f64)>, steps: u32) -> bool {
    let (zx, zy) = z;
    for _ in 0..steps {
        let (xx, yy, xy) = (zx.mul(zx), zy.mul(zy), zx.mul(zy));
        (*zx, *zy) = (xx.sub(&yy).add(real), xy.double().add(imaginary));
        orbit.push((zx.to_f64(), zy.to_f64()));
        if zx.magnitude_above(8) || zy.magnitude_above(8) {
            return true;
        }
    }
    false
}

/// One reference orbit for all the views around one exact center (issue #263: the frames of a dive).
///
/// The orbit, the slowest part of a deep frame with many iterations, is then computed once instead of for every frame. It is computed with the bits of the
/// center it is given (those of the deepest view), as far as a view needs, and continued when a view needs more iterations; a view of `max_iterations` gets
/// exactly the orbit that `reference_orbit` computes
pub struct SharedOrbit {
    real: Fixed,
    imaginary: Fixed,
    state: std::sync::Mutex<OrbitSoFar>,
}

/// The orbit of a `SharedOrbit` so far: its last point in fixed point, its points, and whether it escaped
struct OrbitSoFar {
    z: (Fixed, Fixed),
    points: std::sync::Arc<Vec<(f64, f64)>>,
    escaped: bool,
}

impl SharedOrbit {
    /// The orbit of `(real, imaginary)`, none of it computed yet
    pub fn new(real: &Fixed, imaginary: &Fixed) -> SharedOrbit {
        let zero = Fixed { value: BigInt::zero(), bits: real.bits };
        SharedOrbit { real: real.clone(), imaginary: imaginary.clone(), state: std::sync::Mutex::new(OrbitSoFar { z: (zero.clone(), zero), points: std::sync::Arc::new(vec![(0.0, 0.0)]), escaped: false }) }
    }

    /// The center
    pub const fn center(&self) -> (&Fixed, &Fixed) {
        (&self.real, &self.imaginary)
    }

    /// The orbit up to at least `max_iterations` steps, or until it escaped (it may be longer)
    pub fn orbit(&self, max_iterations: u32) -> std::sync::Arc<Vec<(f64, f64)>> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let have = u32::try_from(state.points.len() - 1).unwrap_or(u32::MAX);
        if !state.escaped && have < max_iterations {
            let mut longer = Vec::with_capacity(max_iterations as usize + 1);
            longer.extend_from_slice(&state.points);
            state.escaped = continue_orbit((&self.real, &self.imaginary), &mut state.z, &mut longer, max_iterations - have);
            state.points = std::sync::Arc::new(longer);
        }
        let points = std::sync::Arc::clone(&state.points);
        drop(state);
        points
    }
}

/// How small the dropped term `z²` of a step must stay against the terms that are kept, for a bilinear approximation (see `Reference`) to stand in for
/// the steps: below 10⁻¹³ of them (measured on issue #141: about 1 pixel in 100 changes its color in the most chaotic views, where a move of a thousandth
/// of a pixel changes a third of them)
const BLA_TOLERANCE: f64 = 1e-13;

fn bla_tolerance() -> f64 {
    #[cfg(test)]
    if let Some(tolerance) = std::env::var("MANDELBROT_BLA_TOLERANCE").ok().and_then(|text| text.parse().ok()) {
        return tolerance;
    }
    BLA_TOLERANCE
}

/// A bilinear approximation of some steps of the perturbation from one step of the reference orbit on: `z ← A·z + B·c`, valid while `|z| < radius`
#[derive(Clone, Copy, Debug)]
struct Bla {
    a: (f64, f64),
    b: (f64, f64),
    radius: f64,
    radius_squared: f64,
}

impl Bla {
    /// `self` (first) followed by `then`, where `max_offset` is the largest `|c|` of the view
    fn then(&self, then: &Bla, max_offset: f64) -> Bla {
        let a = times(then.a, self.a);
        let b = times(then.a, self.b);
        let size_a = self.a.0.hypot(self.a.1);
        let size_b = self.b.0.hypot(self.b.1);
        let radius = self.radius.min(((then.radius - size_b * max_offset) / size_a).max(0.0));
        Bla { a, b: (b.0 + then.b.0, b.1 + then.b.1), radius, radius_squared: radius * radius }
    }
}

fn times((x, y): (f64, f64), (u, v): (f64, f64)) -> (f64, f64) {
    (x * u - y * v, x * v + y * u)
}

/// The orbit of the center of a view (see `reference_orbit`) and, for speed, a table of bilinear approximations (BLA) of its steps.
///
/// **Bilinear approximation** (Zhuoran, 2021): while a pixel's difference `z` from the reference orbit is tiny, the step `z ← 2·Z·z + z² + c` is nearly
/// linear, `z ← A·z + B·c`, and many such steps compose into one: `A₂·(A₁·z + B₁·c) + B₂·c`. The table holds the compositions of 2, 4, 8, ... steps, each
/// with the largest `|z|` at which it is still accurate (`BLA_TOLERANCE`); deep in a view, where the orbits of the pixels follow the center's for
/// thousands of steps, a pixel then skips most of them. The positions differ from step by step perturbation only by that tolerance
pub struct Reference {
    /// The orbit; only its first `length` points are this view's (a shared orbit can be longer, see `SharedOrbit`)
    orbit: std::sync::Arc<Vec<(f64, f64)>>,
    length: usize,
    /// `levels[l][k]`: the approximation of the `2^(l + 1)` steps from step `1 + k · 2^(l + 1)` of the orbit
    levels: Vec<Vec<Bla>>,
}

impl Reference {
    /// The reference orbit of the center `(real, imaginary)`, for a view whose pixels are at most `max_offset` from it; with `approximate` also its table
    /// of bilinear approximations (without it, every step is iterated, as in `position` before BLA)
    pub fn new(m: &MandelbrotSet, (real, imaginary): (&Fixed, &Fixed), max_offset: f64, approximate: bool) -> Reference {
        Reference::of(m, std::sync::Arc::new(reference_orbit(real, imaginary, m.max_iterations)), max_offset, approximate)
    }

    /// `new` with the orbit `orbit` (of at least `m.max_iterations` steps, or until it escaped), see `SharedOrbit`
    pub fn of(m: &MandelbrotSet, orbit: std::sync::Arc<Vec<(f64, f64)>>, max_offset: f64, approximate: bool) -> Reference {
        let length = orbit.len().min(m.max_iterations as usize + 1);
        let levels = if approximate { Self::table(&orbit[..length], max_offset, m.orbit_radius) } else { Vec::new() };
        Reference { orbit, length, levels }
    }

    /// The points of the orbit of this view
    fn points(&self) -> &[(f64, f64)] {
        &self.orbit[..self.length]
    }

    fn table(orbit: &[(f64, f64)], max_offset: f64, orbit_radius: f64) -> Vec<Vec<Bla>> {
        //Single steps from step n = 1, 2, ... (step 0, from Z = 0, is not linear): A = 2·Z, B = 1, valid while |z²| stays below the tolerance of |2·Z·z|,
        //and never where the orbit comes within 0.1% of the orbit radius: a pixel, which is that close to the orbit (|z| ≪ |Z|), could escape inside a skip
        let singles: Vec<Bla> = (1..orbit.len().saturating_sub(1))
            .map(|n| {
                let (x, y) = orbit[n];
                let next = orbit[n + 1];
                let size = x.hypot(y);
                let radius = if next.0.hypot(next.1) < 0.999 * orbit_radius { 2.0 * bla_tolerance() * size } else { 0.0 };
                Bla { a: (2.0 * x, 2.0 * y), b: (1.0, 0.0), radius, radius_squared: radius * radius }
            })
            .collect();
        //Each level composes pairs of the level below
        let mut levels: Vec<Vec<Bla>> = Vec::new();
        loop {
            let below = levels.last().unwrap_or(&singles);
            if below.len() < 2 {
                return levels;
            }
            let level = below.as_chunks::<2>().0.iter().map(|[first, then]| first.then(then, max_offset)).collect();
            levels.push(level);
        }
    }

    /// The longest approximation that a difference of `|z|² = size_squared` at step `at` of the orbit may take, of at most `left` steps: it and its steps
    fn skip(&self, at: usize, size_squared: f64, left: u32) -> Option<(&Bla, u32)> {
        if at == 0 || self.levels.is_empty() {
            return None;
        }
        let aligned = ((at - 1).trailing_zeros() as usize).min(self.levels.len());
        for level in (1..=aligned).rev() {
            let steps = 1_usize << level;
            let Some(bla) = self.levels[level - 1].get((at - 1) >> level) else { continue };
            if steps as u64 <= u64::from(left) && size_squared < bla.radius_squared {
                return Some((bla, steps as u32));
            }
        }
        None
    }
}

/// The smooth position (see `MandelbrotSet::position`) of the point at offset `(dcx, dcy)` from the center of `reference`, by perturbation with rebasing.
///
/// `center` is the center in 64-bit floats, for the steps after the point escaped; see the module, and `Reference` for the steps it skips
pub fn position(m: &MandelbrotSet, reference: &Reference, center: (f64, f64), (dcx, dcy): (f64, f64)) -> f64 {
    position_from(m, reference, center, (dcx, dcy), ((0.0, 0.0), 0, 0))
}

/// `position` from the difference `z` at step `at` of the reference orbit, after `step` steps of the pixel (`position` starts at 0, 0, 0; see `position_scaled`)
fn position_from(m: &MandelbrotSet, reference: &Reference, center: (f64, f64), (dcx, dcy): (f64, f64), ((zx, zy), at, step): ((f64, f64), usize, u32)) -> f64 {
    let orbit = reference.points();
    let radius_squared = m.orbit_radius * m.orbit_radius;
    let (mut zx, mut zy) = (zx, zy); //The difference from the reference orbit
    let mut at = at; //The step of the reference orbit that the difference is from
    let mut step = step;
    while step < m.max_iterations {
        if let Some((bla, steps)) = reference.skip(at, zx * zx + zy * zy, m.max_iterations - step) {
            //z ← A·z + B·c, for `steps` steps at once
            (zx, zy) = (bla.a.0 * zx - bla.a.1 * zy + (bla.b.0 * dcx - bla.b.1 * dcy), bla.a.0 * zy + bla.a.1 * zx + (bla.b.0 * dcy + bla.b.1 * dcx));
            at += steps as usize;
            step += steps;
        } else {
            let (ox, oy) = orbit[at];
            //z ← 2·Z·z + z² + c
            (zx, zy) = (2.0 * (ox * zx - oy * zy) + (zx * zx - zy * zy) + dcx, 2.0 * (ox * zy + oy * zx) + 2.0 * zx * zy + dcy);
            at += 1;
            step += 1;
        }
        let (fx, fy) = (orbit[at].0 + zx, orbit[at].1 + zy); //The pixel's orbit
        let full = fx * fx + fy * fy;
        if full > radius_squared {
            return m.smooth_position_of(step, (fx, fy), (center.0 + dcx, center.1 + dcy));
        }
        //Rebasing: the pixel's orbit is closer to 0 than its difference, or the reference orbit ends
        if full < zx * zx + zy * zy || at + 1 >= orbit.len() {
            (zx, zy) = (fx, fy);
            at = 0;
        }
    }
    f64::from(m.max_iterations)
}

/// A positive number beyond the range of 64-bit floats, `mantissa · 2^exponent`: the scale of a view deeper than 10⁻³⁰⁰ (issue #263)
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Depth {
    /// In `[1, 2)`
    pub mantissa: f64,
    /// The power of two
    pub exponent: i32,
}

impl Depth {
    /// The depth of the decimal text `text` (`2.5e-1000`), or `None` if it is not a positive number
    pub fn parse(text: &str) -> Option<Depth> {
        let text = text.trim();
        let (mantissa, power) = match text.find(['e', 'E']) {
            Some(at) => (text[..at].parse::<f64>().ok()?, text[at + 1..].parse::<i64>().ok()?),
            None => (text.parse::<f64>().ok()?, 0),
        };
        if !(mantissa.is_finite() && mantissa > 0.0) {
            return None;
        }
        //10^power = 2^(power · log2 10): the whole part goes to the exponent, the rest into the mantissa
        let log2 = mantissa.log2() + power as f64 * std::f64::consts::LOG2_10;
        let exponent = log2.floor();
        Some(Depth { mantissa: (log2 - exponent).exp2(), exponent: i32::try_from(exponent as i64).ok()? })
    }

    /// The base-2 logarithm of the depth
    pub fn log2(self) -> f64 {
        self.mantissa.log2() + f64::from(self.exponent)
    }

    /// The depth of a 64-bit float (positive and normal), exactly
    pub fn from_f64(scale: f64) -> Depth {
        let exponent = i32::try_from((scale.to_bits() >> 52) & 0x7ff).unwrap_or(1023) - 1023;
        Depth { mantissa: times_two_to(scale, -exponent), exponent }
    }

    /// The depth as a 64-bit float: 0 below their range, infinite above it
    pub fn to_f64(self) -> f64 {
        times_two_to(self.mantissa, self.exponent)
    }

    /// The depth times `factor` (positive), normalized again
    #[must_use]
    pub fn times(self, factor: f64) -> Depth {
        let other = Depth::from_f64(self.mantissa * factor);
        Depth { mantissa: other.mantissa, exponent: other.exponent + self.exponent }
    }

    /// The depth `2^log2`
    pub fn from_log2(log2: f64) -> Depth {
        let exponent = log2.floor();
        Depth { mantissa: (log2 - exponent).exp2().clamp(1.0, 2.0 - f64::EPSILON), exponent: exponent.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32 }
    }

    /// `1 / depth` (the magnification of a scale)
    #[must_use]
    pub fn reciprocal(self) -> Depth {
        if self.mantissa <= 1.0 {
            return Depth { mantissa: 1.0, exponent: -self.exponent };
        }
        Depth { mantissa: 2.0 / self.mantissa, exponent: -self.exponent - 1 }
    }

    /// The number `10^log10` in scientific notation with `decimals` decimals, `1.25e-1000` (rounded so the mantissa stays below 10)
    pub fn scientific(log10: f64, decimals: usize) -> String {
        let mut power = log10.floor();
        let mut mantissa = 10f64.powf(log10 - power);
        if format!("{mantissa:.decimals$}").starts_with("10") {
            (mantissa, power) = (mantissa / 10.0, power + 1.0);
        }
        format!("{mantissa:.decimals$}e{}", power as i64)
    }

    /// The base-10 logarithm of the depth
    pub fn log10(self) -> f64 {
        self.log2() / std::f64::consts::LOG2_10
    }

    /// The depth as decimal text that `parse` reads back as nearly the same depth (to about 15 digits; for files)
    pub fn to_exact_text(self) -> String {
        Depth::scientific(self.log10(), 15)
    }

    /// The depth as decimal text, `1.25e-1000` (for the log and the view's text)
    pub fn to_text(self) -> String {
        Depth::scientific(self.log10(), 6)
    }
}

/// `bits_for` for a depth beyond 64-bit floats
pub fn bits_for_depth(depth: Depth, width: usize) -> u32 {
    if depth.to_f64().is_normal() {
        return bits_for(depth.to_f64(), width); //(The same bits as before at the depths of 64-bit floats: the same images)
    }
    let pixel_log2 = depth.log2() + (2.5 / width.max(1) as f64).log2();
    (-pixel_log2).ceil().max(0.0) as u32 + 64
}

/// `x · 2^power`, exactly as long as the result is a normal number (in steps that stay within the exponents of a 64-bit float)
fn times_two_to(x: f64, power: i32) -> f64 {
    let (mut x, mut power) = (x, power);
    while power != 0 {
        let step = power.clamp(-1000, 1000);
        x *= f64::from_bits(((1023 + step) as u64) << 52);
        power -= step;
    }
    x
}

/// Below `2^-SCALED_BELOW` a pixel's difference from the reference orbit is kept scaled by a power of two, see `position_scaled`
const SCALED_BELOW: i32 = 1000;

/// The position of the pixel at the offset `c · 2^-scaled` from the center (an `f64` offset that would underflow, scaled up by `2^scaled`), for views deeper
/// than 64-bit floats reach (issue #263).
///
/// While the difference `z` from the reference orbit is that small, its square is negligible against `2·Z·z` (below 2⁻⁵⁰⁰ of it), so a step is linear,
/// `z ← 2·Z·z + c`, and so are the bilinear approximations: `z` and `c` keep one scale, and are scaled down by exact powers of two before `z` could overflow.
/// Once `z` is above `2^-SCALED_BELOW` (where its square starts to count long before it underflows) they are unscaled and `position_from` goes on as usual.
/// A pixel whose scale is already within reach goes to `position` directly, bit for bit as before
pub fn position_scaled(m: &MandelbrotSet, reference: &Reference, center: (f64, f64), (cx, cy): (f64, f64), scaled: i32) -> f64 {
    if scaled <= SCALED_BELOW {
        return position(m, reference, center, (times_two_to(cx, -scaled), times_two_to(cy, -scaled)));
    }
    let orbit = reference.points();
    let radius_squared = m.orbit_radius * m.orbit_radius;
    let (mut zx, mut zy, mut cx, mut cy, mut scaled) = (0.0_f64, 0.0_f64, cx, cy, scaled);
    let (mut at, mut step) = (0_usize, 0_u32);
    while step < m.max_iterations {
        //(Any approximation whose radius is not 0 holds for so small a z)
        if let Some((bla, steps)) = reference.skip(at, 0.0, m.max_iterations - step) {
            (zx, zy) = (bla.a.0 * zx - bla.a.1 * zy + (bla.b.0 * cx - bla.b.1 * cy), bla.a.0 * zy + bla.a.1 * zx + (bla.b.0 * cy + bla.b.1 * cx));
            at += steps as usize;
            step += steps;
        } else {
            let (ox, oy) = orbit[at];
            (zx, zy) = (2.0 * (ox * zx - oy * zy) + cx, 2.0 * (ox * zy + oy * zx) + cy);
            at += 1;
            step += 1;
        }
        //The pixel's orbit is the center's to the last bit here: it escapes with it
        let (fx, fy) = orbit[at];
        if fx * fx + fy * fy > radius_squared {
            return m.smooth_position_of(step, (fx, fy), center);
        }
        if zx * zx + zy * zy > 1e300 {
            (zx, zy, cx, cy, scaled) = (times_two_to(zx, -500), times_two_to(zy, -500), times_two_to(cx, -500), times_two_to(cy, -500), scaled - 500);
        }
        if scaled <= SCALED_BELOW {
            let unscale = |value: f64| times_two_to(value, -scaled);
            return position_from(m, reference, center, (unscale(cx), unscale(cy)), ((unscale(zx), unscale(zy)), at, step));
        }
    }
    f64::from(m.max_iterations)
}

/// `render_fixed` for a view of `depth` (`2.5 · depth` wide), which may be far deeper than 64-bit floats reach; the center needs `bits_for_depth` bits
#[allow(clippy::too_many_arguments)] // The view, its size, how and when to stop
pub fn render_depth(m: &MandelbrotSet, center_real: &Fixed, center_imaginary: &Fixed, (depth, rotation): (Depth, f64), width: usize, height: usize, approximate: bool,
    cancel: &std::sync::atomic::AtomicBool) -> Option<Vec<f64>> {
    use std::sync::atomic::Ordering;
    //A pixel is `pixel · 2^depth.exponent`: the offsets are kept scaled by `2^-depth.exponent` (their largest |c| is far below what the table's radii notice)
    let pixel = 2.5 * depth.mantissa / width as f64;
    let scaled = -depth.exponent;
    let reference = Reference::new(m, (center_real, center_imaginary), 0.0, approximate);
    let center = (center_real.to_f64(), center_imaginary.to_f64());
    let mut positions = vec![0.0; width * height];
    let threads = super::render_thread_count().max(1);
    let rows_per_thread = height.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, rows) in positions.chunks_mut(rows_per_thread * width).enumerate() {
            let reference = &reference;
            scope.spawn(move || {
                for (index, out) in rows.iter_mut().enumerate() {
                    if index % width == 0 && cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let (x, y) = (index % width, chunk * rows_per_thread + index / width);
                    let offset = turned(((x as f64 + 0.5 - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - 0.5) * pixel), rotation);
                    *out = position_scaled(m, reference, center, offset, scaled);
                }
            });
        }
    });
    (!cancel.load(Ordering::Relaxed)).then_some(positions)
}

/// The offset from the center of the point at the offset `(right, up)` on the screen, in a view turned counterclockwise by `rotation` radians.
///
/// Turned as the 64-bit renderer turns it, see `ComplexPlane::complex_from_pixel_plane`
pub fn turned((right, up): (f64, f64), rotation: f64) -> (f64, f64) {
    if rotation == 0.0 {
        return (right, up);
    }
    let (sin, cos) = rotation.sin_cos();
    (right.mul_add(cos, up * sin), up.mul_add(cos, -(right * sin)))
}

/// The smooth positions of a `width` x `height` view of the center `(real, imaginary)` and `scale`, row by row from the top.
///
/// The center is decimal text, as precise as needed; the view is `2.5 · scale` wide. One sample in the middle of every pixel, on all the render threads
pub fn render_positions(m: &MandelbrotSet, real: &str, imaginary: &str, scale: f64, width: usize, height: usize) -> Option<Vec<f64>> {
    let bits = bits_for(scale, width);
    let (center_real, center_imaginary) = (Fixed::parse(real, bits)?, Fixed::parse(imaginary, bits)?);
    render_fixed(m, &center_real, &center_imaginary, (scale, 0.0), width, height, false, &std::sync::atomic::AtomicBool::new(false))
}

/// The colors of a view like `render_fixed`, averaged over `samples` samples a pixel, one layer after the other.
///
/// The samples are at the jitter of the 64-bit renderer (see `jitter`); `on_layer` gets the average after every layer and its number (from 1), and
/// `color_of` colors a smooth position. The view is turned counterclockwise by `rotation` radians; `approximate` skips steps by bilinear approximation
/// (see `Reference`). False when `cancel` stopped it
#[allow(clippy::too_many_arguments)] // The view, its size and samples, how to color, when to stop and where the layers go
pub fn render_layers(m: &MandelbrotSet, center_real: &Fixed, center_imaginary: &Fixed, (depth, rotation): (Depth, f64), (width, height): (usize, usize), samples: u8, color_of: &(impl Fn(f64) -> crate::coloring::TrueColor + Sync),
    approximate: bool, cancel: &std::sync::atomic::AtomicBool, on_layer: impl FnMut(&[crate::coloring::TrueColor], u8)) -> bool {
    render_layers_around(m, (&SharedOrbit::new(center_real, center_imaginary), (0.0, 0.0)), (depth, rotation), (width, height), samples, color_of, approximate, cancel, on_layer)
}

/// `render_layers` for a view around the center of a shared orbit (issue #263, see `SharedOrbit`): the frames of a dive share one.
///
/// The view's center is `(right, up)` scales (see `Depth`) from the orbit's, which it perturbs from; the orbit's center needs the bits of the view
/// (`bits_for_depth`)
#[allow(clippy::too_many_arguments)] // The orbit and the view, its size and samples, how to color, when to stop and where the layers go
pub fn render_layers_around(m: &MandelbrotSet, (orbit, (right, up)): (&SharedOrbit, (f64, f64)), (depth, rotation): (Depth, f64), (width, height): (usize, usize), samples: u8,
    color_of: &(impl Fn(f64) -> crate::coloring::TrueColor + Sync), approximate: bool, cancel: &std::sync::atomic::AtomicBool, mut on_layer: impl FnMut(&[crate::coloring::TrueColor], u8)) -> bool {
    use std::sync::atomic::Ordering;
    //A pixel is `pixel · 2^depth.exponent`: the offsets are scaled by `2^-depth.exponent` (see `position_scaled`; within 64-bit floats' range bit for bit as before)
    let pixel = 2.5 * depth.mantissa / width as f64;
    let scaled = -depth.exponent;
    //(The view's center from the orbit's, in the same scaled units)
    let shift = (right * depth.mantissa, up * depth.mantissa);
    let reach = max_offset(pixel, width, height) + shift.0.hypot(shift.1);
    let reference = Reference::of(m, orbit.orbit(m.max_iterations), times_two_to(reach, depth.exponent), approximate);
    let (center_real, center_imaginary) = orbit.center();
    let center = (center_real.to_f64(), center_imaginary.to_f64());
    let threads = super::render_thread_count().max(1);
    let rows_per_thread = height.div_ceil(threads);
    let mut sums = vec![[0_u32; 3]; width * height];
    let mut colors = vec![crate::coloring::TrueColor::new(0, 0, 0); width * height];
    for sample in 0..samples.max(1) {
        std::thread::scope(|scope| {
            for (chunk, (rows, out)) in sums.chunks_mut(rows_per_thread * width).zip(colors.chunks_mut(rows_per_thread * width)).enumerate() {
                let reference = &reference;
                scope.spawn(move || {
                    for (index, (sum, color)) in rows.iter_mut().zip(out.iter_mut()).enumerate() {
                        if index % width == 0 && cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        let (x, y) = (index % width, chunk * rows_per_thread + index / width);
                        //The first sample in the middle of the pixel, the others at the jitter of the 64-bit renderer
                        let (jx, jy) = if sample == 0 { (0.5, 0.5) } else { super::jitter::sample_jitter(x, y, sample) };
                        let offset = turned(((x as f64 + jx - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - jy) * pixel), rotation);
                        let offset = if shift == (0.0, 0.0) { offset } else { (offset.0 + shift.0, offset.1 + shift.1) };
                        let new = color_of(position_scaled(m, reference, center, offset, scaled));
                        for (total, channel) in sum.iter_mut().zip([new.red, new.green, new.blue]) {
                            *total += u32::from(channel);
                        }
                        let count = u32::from(sample) + 1;
                        *color = crate::coloring::TrueColor::new(((sum[0] + count / 2) / count) as u8, ((sum[1] + count / 2) / count) as u8, ((sum[2] + count / 2) / count) as u8);
                    }
                });
            }
        });
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        on_layer(&colors, sample + 1);
    }
    true
}

/// The largest distance of a pixel of a `width` x `height` view with pixels of size `pixel` from its center (with a pixel to spare for the jitter)
fn max_offset(pixel: f64, width: usize, height: usize) -> f64 {
    pixel * (width as f64 / 2.0 + 1.0).hypot(height as f64 / 2.0 + 1.0)
}

/// Like `render_positions`, for a center in fixed point (with at least `bits_for(scale, width)` bits); `None` when `cancel` was set before it finished.
///
/// The view is turned counterclockwise by `rotation` radians, and `approximate` is as in `render_layers`
#[allow(clippy::too_many_arguments)] // The view, its size, how and when to stop
pub fn render_fixed(m: &MandelbrotSet, center_real: &Fixed, center_imaginary: &Fixed, (scale, rotation): (f64, f64), width: usize, height: usize, approximate: bool,
    cancel: &std::sync::atomic::AtomicBool) -> Option<Vec<f64>> {
    use std::sync::atomic::Ordering;
    let pixel = 2.5 * scale / width as f64;
    let reference = Reference::new(m, (center_real, center_imaginary), max_offset(pixel, width, height), approximate);
    let center = (center_real.to_f64(), center_imaginary.to_f64());
    let mut positions = vec![0.0; width * height];
    let threads = super::render_thread_count().max(1);
    let rows_per_thread = height.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, rows) in positions.chunks_mut(rows_per_thread * width).enumerate() {
            let reference = &reference;
            scope.spawn(move || {
                for (index, out) in rows.iter_mut().enumerate() {
                    if index % width == 0 && cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let (x, y) = (index % width, chunk * rows_per_thread + index / width);
                    let offset = turned(((x as f64 + 0.5 - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - 0.5) * pixel), rotation);
                    *out = position(m, reference, center, offset);
                }
            });
        }
    });
    (!cancel.load(Ordering::Relaxed)).then_some(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The escape counts of the pixels of a `width` x `height` view of `depth` around `(real, imaginary)` in exact fixed-point arithmetic (the reference that
    /// perturbation is checked against): every pixel's own orbit, step by step, until it leaves radius 2
    fn exact_counts(real: &str, imaginary: &str, (mantissa, power): (f64, i64), (width, height): (usize, usize), limit: u32) -> Vec<u32> {
        let bits = bits_for_depth(Depth::parse(&format!("{mantissa}e{power}")).unwrap(), width) + 32;
        let (center_x, center_y) = (Fixed::parse(real, bits).unwrap(), Fixed::parse(imaginary, bits).unwrap());
        let four = Fixed::parse("4", bits).unwrap();
        (0..height).flat_map(|y| (0..width).map(move |x| (x, y))).map(|(x, y)| {
            let pixel = 2.5 * mantissa / width as f64;
            let (right, up) = ((x as f64 + 0.5 - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - 0.5) * pixel);
            let cx = center_x.add(&Fixed::parse(&format!("{right}e{power}"), bits).unwrap());
            let cy = center_y.add(&Fixed::parse(&format!("{up}e{power}"), bits).unwrap());
            let (mut zx, mut zy) = (Fixed::parse("0", bits).unwrap(), Fixed::parse("0", bits).unwrap());
            let mut n = 0;
            while n < limit {
                let (xx, yy, xy) = (zx.mul(&zx), zy.mul(&zy), zx.mul(&zy));
                if xx.add(&yy).sub(&four).value.is_positive() {
                    break;
                }
                (zx, zy) = (xx.sub(&yy).add(&cx), xy.double().add(&cy));
                n += 1;
            }
            n
        }).collect()
    }

    #[test]
    fn a_shared_orbit_is_exactly_the_reference_orbit_however_far_it_was_computed() {
        let bits = 400;
        let (real, imaginary) = (Fixed::parse("-0.7436438870371587", bits).unwrap(), Fixed::parse("0.13182590420531198", bits).unwrap());
        let shared = SharedOrbit::new(&real, &imaginary);
        for iterations in [10, 1000, 300, 5000] {
            let orbit = shared.orbit(iterations);
            let expected = reference_orbit(&real, &imaginary, iterations);
            assert_eq!(orbit[..expected.len()], expected[..], "{iterations} iterations");
            assert!(orbit.len() >= expected.len());
        }
        //An escaping orbit stops where it escapes, however many steps are asked for
        let outside = SharedOrbit::new(&Fixed::parse("0.3", bits).unwrap(), &Fixed::parse("0.6", bits).unwrap());
        assert_eq!(*outside.orbit(100_000), reference_orbit(&Fixed::parse("0.3", bits).unwrap(), &Fixed::parse("0.6", bits).unwrap(), 100_000));
    }

    #[test]
    fn the_frames_of_a_dive_perturb_from_one_shared_orbit_with_the_escape_counts_of_exact_arithmetic() {
        //#263 step 3: the orbit of the destination (10⁻⁴⁰⁰, next to c = i), computed once with its bits, serves frames at every depth on the way, also
        //frames whose center is off the destination's (the pan of a flight): every pixel has the count of its own orbit in exact arithmetic, with and without BLA
        let mut m = MandelbrotSet::new(20_000, 2.0);
        m.smooth_coloring = false;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let (width, height) = (12, 8);
        let bits = bits_for_depth(Depth::parse("1e-400").unwrap(), width);
        let orbit = SharedOrbit::new(&Fixed::parse("3e-401", bits).unwrap(), &Fixed::parse(&format!("1.{}2", "0".repeat(400)), bits).unwrap());
        let color_of = |position: f64| {
            let count = position as u32;
            crate::coloring::TrueColor::new((count & 255) as u8, (count >> 8 & 255) as u8, (count >> 16) as u8)
        };
        for (power, (right, up)) in [(-400_i64, (0.0, 0.0)), (-350, (0.0, 0.0)), (-308, (0.0, 0.0)), (-308, (3.7, -1.2)), (-200, (-0.4, 2.5)), (-30, (0.0, 0.0))] {
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let (real, imaginary) = (orbit.center().0.plus_scaled(right * depth.mantissa, depth.exponent), orbit.center().1.plus_scaled(up * depth.mantissa, depth.exponent));
            let exact = exact_counts(&real.to_decimal(), &imaginary.to_decimal(), (1.0, power), (width, height), m.max_iterations);
            for approximate in [false, true] {
                let mut counts = Vec::new();
                assert!(render_layers_around(&m, (&orbit, (right, up)), (depth, 0.0), (width, height), 1, &color_of, approximate, &cancel, |colors, _| {
                    counts = colors.iter().map(|color| u32::from(color.red) | u32::from(color.green) << 8 | u32::from(color.blue) << 16).collect();
                }));
                assert_eq!(counts, exact, "10^{power}, ({right}, {up}) scales off, BLA {approximate}");
            }
            let distinct: std::collections::BTreeSet<u32> = exact.iter().copied().collect();
            assert!(distinct.len() > if power == -400 { 5 } else { 1 }, "10^{power}: a view with structure: {distinct:?}");
        }
    }

    #[test]
    fn views_beyond_64_bit_floats_have_the_escape_counts_of_exact_arithmetic() {
        //#263: around c = i (a point of the boundary with structure at every depth), at 10⁻³⁰⁰, where the 64-bit differences underflowed (every pixel came out
        //"inside"), and at 10⁻¹⁰⁰⁰: perturbation with scaled differences gives every pixel the count of its own orbit in exact arithmetic, with and without BLA
        let mut m = MandelbrotSet::new(20_000, 2.0);
        m.smooth_coloring = false;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        for power in [-300_i64, -308, -1000] {
            //A center a few views from c = i, so the view shows the structure around it
            let (real, imaginary) = (format!("3e{}", power - 1), format!("1.{}2", "0".repeat((-power) as usize)));
            let (real, imaginary) = (real.as_str(), imaginary.as_str());
            let (width, height) = (12, 8);
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let bits = bits_for_depth(depth, width);
            let (cx, cy) = (Fixed::parse(real, bits).unwrap(), Fixed::parse(imaginary, bits).unwrap());
            let exact = exact_counts(real, imaginary, (1.0, power), (width, height), m.max_iterations);
            for approximate in [false, true] {
                let counts = render_depth(&m, &cx, &cy, (depth, 0.0), width, height, approximate, &cancel).unwrap();
                let differ = counts.iter().zip(&exact).filter(|(a, b)| **a as u32 != **b).count();
                assert_eq!(differ, 0, "10^{power}, BLA {approximate}: {:?} against {exact:?}", counts.iter().map(|count| *count as u32).collect::<Vec<_>>());
                let distinct: std::collections::BTreeSet<u32> = exact.iter().copied().collect();
                assert!(distinct.len() > 5, "10^{power}: a view with structure: {distinct:?}");
            }
        }
    }

    #[test]
    fn decimal_text_reads_exactly_into_fixed_point() {
        let half = Fixed::parse("0.5", 80).unwrap();
        assert_eq!(half.value, BigInt::one() << 79u32);
        assert_eq!(Fixed::parse("-1.25e-3", 80).unwrap().to_f64(), -1.25e-3);
        assert_eq!(Fixed::parse("-0.7498100000000001", 120).unwrap().to_f64(), -0.749_810_000_000_000_1);
        assert!(Fixed::parse("abc", 64).is_none() && Fixed::parse("", 64).is_none());
        //Far beyond 64-bit floats: two centers 10⁻⁵⁰ apart are different numbers
        let bits = bits_for(1e-50, 1000);
        let (a, b) = (Fixed::parse("-1.25e-0", bits).unwrap(), Fixed::parse("-1.25000000000000000000000000000000000000000000000000001", bits).unwrap());
        assert_ne!(a, b);
    }

    #[test]
    fn a_center_moves_by_exact_steps_and_reads_back_from_its_text() {
        let bits = bits_for(1e-60, 1000);
        let center = Fixed::parse("-1.7499999999999999999999999999999999999999999999999999999999999987", bits).unwrap();
        let moved = center.plus(3.25e-62).plus(-3.25e-62);
        assert_eq!(moved, center, "a step and the step back");
        assert_eq!(Fixed::parse(&center.to_decimal(), bits).unwrap(), center, "the text reads back to the same number");
        assert_eq!(Fixed::parse("0.75", 64).unwrap().to_decimal(), "0.75");
        assert_eq!(Fixed::parse("-2", 64).unwrap().to_decimal(), "-2.0");
        assert_eq!(Fixed::parse("0.5", 64).unwrap().plus(0.25).to_f64(), 0.75);
        assert_eq!(center.with_bits(bits + 40).with_bits(bits), center);
    }

    #[test]
    fn perturbation_gives_the_positions_of_the_64_bit_loop_where_both_work() {
        //A view of 10⁻⁵, where the 64-bit loop is exact: perturbation agrees on nearly every pixel (both are approximations in chaotic spots)
        let m = MandelbrotSet::new(2000, 2.0);
        let (real, imaginary, scale, width, height) = ("-0.7498100000000001", "-0.020300000000000054", 6.4e-5, 64, 48);
        let deep = render_positions(&m, real, imaginary, scale, width, height).unwrap();
        let pixel = 2.5 * scale / width as f64;
        let mut close = 0;
        for (index, deep) in deep.iter().enumerate() {
            let (x, y) = (index % width, index / width);
            let c = crate::complex::Complex::new(-0.749_810_000_000_000_1 + (x as f64 + 0.5 - width as f64 / 2.0) * pixel, -0.020_300_000_000_000_054 + (height as f64 / 2.0 - y as f64 - 0.5) * pixel);
            let direct = m.iterate_smooth(&c);
            close += usize::from((deep - direct).abs() < 0.01);
        }
        assert!(close * 100 >= deep.len() * 95, "{close} of {} pixels agree", deep.len());
    }

    #[test]
    fn a_turned_deep_view_shows_the_points_of_the_turned_64_bit_view() {
        //Turned like the 64-bit renderer turns a view, so a deep view looks the same turned as the view just above it
        let mut plane = crate::complex_plane::ComplexPlane::new(64, 48);
        plane.set_view(&crate::complex_plane::View::new(-0.75, 0.1, 0.01));
        for rotation in [0.0, 0.7, -2.5, std::f64::consts::PI] {
            plane.set_rotation(rotation);
            for (right, up) in [(1e-3, 0.0), (0.0, -2e-3), (3e-3, 4e-3)] {
                let (x, y) = turned((right, up), rotation);
                let (ex, ey) = plane.screen_offset(right, up);
                assert!((x - ex).abs() < 1e-18 && (y - ey).abs() < 1e-18, "{rotation}: ({x}, {y}) is not ({ex}, {ey})");
            }
        }
        //And a turned render is the unturned render turned: half a turn shows every pixel at the opposite one
        let m = MandelbrotSet::new(500, 2.0);
        let (real, imaginary) = (Fixed::parse("-0.7498100000000001", 80).unwrap(), Fixed::parse("-0.020300000000000054", 80).unwrap());
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let unturned = render_fixed(&m, &real, &imaginary, (6.4e-5, 0.0), 16, 12, false, &cancel).unwrap();
        let half_turned = render_fixed(&m, &real, &imaginary, (6.4e-5, std::f64::consts::PI), 16, 12, false, &cancel).unwrap();
        let reversed: Vec<f64> = unturned.iter().rev().copied().collect();
        let close = half_turned.iter().zip(&reversed).filter(|(a, b)| (*a - *b).abs() < 0.01).count();
        assert!(close * 100 >= unturned.len() * 95, "{close} of {} pixels agree", unturned.len());
    }

    #[test]
    fn supersampled_layers_average_their_samples_and_the_first_is_the_middle_of_the_pixels() {
        let m = MandelbrotSet::new(2000, 2.0);
        let bits = bits_for(1e-30, 32);
        let (real, imaginary) = (Fixed::parse("0.0000000000000000000000000000003", bits).unwrap(), Fixed::parse("1.0000000000000000000000000000002", bits).unwrap());
        let color_of = |position: f64| crate::coloring::TrueColor::new((position * 3.0) as u8, 0, 0);
        let mut layers = Vec::new();
        assert!(render_layers(&m, &real, &imaginary, (Depth::from_f64(1e-30), 0.0), (32, 24), 4, &color_of, false, &std::sync::atomic::AtomicBool::new(false), |colors, layer| layers.push((layer, colors.to_vec()))));
        assert_eq!(layers.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(), [1, 2, 3, 4]);
        let first = render_fixed(&m, &real, &imaginary, (1e-30, 0.0), 32, 24, false, &std::sync::atomic::AtomicBool::new(false)).unwrap();
        assert!(layers[0].1.iter().zip(&first).all(|(color, position)| color.red == (position * 3.0) as u8), "the first layer: one sample in the middle of every pixel");
        assert!(layers[0].1.iter().zip(&layers[3].1).any(|(a, b)| a.red != b.red), "more samples change the image");
    }

    /// The center of a minibrot of period 8007, 6·10⁻³³ from the seahorse point -0.743643887037158704752191506114774 + 0.131825904205311970493132056385139i
    /// (found by Newton's method on issue #141): a deep view where pixels follow the center's orbit for thousands of steps
    const MINIBROT: (&str, &str) = ("-0.7436438870371587047521915061147797782152562079481812898168839018185", "0.13182590420531197049313205638514067897295227932891896620912767746551");

    fn minibrot_views(scale: f64, (width, height): (usize, usize), iterations: u32) -> (Vec<f64>, Vec<f64>) {
        let m = MandelbrotSet::new(iterations, 2.0);
        let bits = bits_for(scale, width);
        let (real, imaginary) = (Fixed::parse(MINIBROT.0, bits).unwrap(), Fixed::parse(MINIBROT.1, bits).unwrap());
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let step_by_step = render_fixed(&m, &real, &imaginary, (scale, 0.0), width, height, false, &cancel).unwrap();
        (step_by_step, render_fixed(&m, &real, &imaginary, (scale, 0.0), width, height, true, &cancel).unwrap())
    }

    #[test]
    fn bilinear_approximation_skips_the_steps_inside_a_deep_minibrot_with_the_same_result() {
        //Every pixel inside the minibrot: all 100 000 steps, of which the approximation skips nearly all (40 times as fast, measured on issue #141)
        let (step_by_step, approximate) = minibrot_views(1e-33, (32, 18), 100_000);
        assert!(step_by_step.iter().all(|position| *position >= 100_000.0), "inside");
        assert_eq!(step_by_step, approximate);
        let m = MandelbrotSet::new(100_000, 2.0);
        let (real, imaginary) = (Fixed::parse(MINIBROT.0, 200).unwrap(), Fixed::parse(MINIBROT.1, 200).unwrap());
        let reference = Reference::new(&m, (&real, &imaginary), 1e-33, true);
        //From the first step a skip of 512 steps: the orbit passes near 0 at step 998 (a minibrot of period 998 is near), where no approximation holds
        let longest = reference.skip(1, 1e-40, 100_000).map(|(_, steps)| steps);
        assert_eq!(longest, Some(512));
    }

    #[test]
    fn bilinear_approximation_changes_few_pixels_at_the_edge_of_a_deep_minibrot() {
        //At its edge, where a move of a thousandth of a pixel changes a third of the pixels by more than a step (chaos), the approximation changes about 1%
        let (step_by_step, approximate) = minibrot_views(1e-31, (48, 27), 200_000);
        let moved = step_by_step.iter().zip(&approximate).filter(|(a, b)| (*a - *b).abs() > 1.0).count();
        let distinct: std::collections::BTreeSet<u64> = step_by_step.iter().map(|position| *position as u64).collect();
        assert!(distinct.len() > 100, "a view with structure: {} different counts", distinct.len());
        assert!(moved * 100 <= step_by_step.len() * 3, "{moved} of {} pixels moved by more than a step", step_by_step.len());
    }

    /// Measures BLA (issue #141): the time of a deep view with and without bilinear approximation, and how far its pixels move. Run by hand:
    /// `MANDELBROT_DEEP="real imaginary scale iterations width height" cargo test --release measure_bla -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn measure_bla() {
        let spec = std::env::var("MANDELBROT_DEEP").unwrap_or_else(|_| String::from("-0.743643887037158704752191506114774 0.131825904205311970493132056385139 1e-25 20000 320 180"));
        let parts: Vec<&str> = spec.split_whitespace().collect();
        let (scale, iterations, width, height): (f64, u32, usize, usize) = (parts[2].parse().unwrap(), parts[3].parse().unwrap(), parts[4].parse().unwrap(), parts[5].parse().unwrap());
        let m = MandelbrotSet::new(iterations, 2.0);
        let bits = bits_for(scale, width);
        let (real, imaginary) = (Fixed::parse(parts[0], bits).unwrap(), Fixed::parse(parts[1], bits).unwrap());
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let started = std::time::Instant::now();
        let orbit = reference_orbit(&real, &imaginary, iterations);
        let orbit_time = started.elapsed();
        let started = std::time::Instant::now();
        let reference = Reference::new(&m, (&real, &imaginary), 2.5 * scale, true);
        println!("reference orbit: {} steps in {orbit_time:.2?}; orbit and BLA table {:.2?}, {} levels", orbit.len(), started.elapsed(), reference.levels.len());
        let started = std::time::Instant::now();
        let exact = render_fixed(&m, &real, &imaginary, (scale, 0.0), width, height, false, &cancel).unwrap();
        let exact_time = started.elapsed();
        let started = std::time::Instant::now();
        let approximate = render_fixed(&m, &real, &imaginary, (scale, 0.0), width, height, true, &cancel).unwrap();
        let approximate_time = started.elapsed();
        let moved_by = |other: &[f64], limit: f64| exact.iter().zip(other).filter(|(a, b)| (*a - *b).abs() > limit).count() as f64 * 100.0 / exact.len() as f64;
        let differences: Vec<f64> = exact.iter().zip(&approximate).map(|(a, b)| (a - b).abs()).collect();
        let moved = |limit: f64| moved_by(&approximate, limit);
        //The baseline: how far the pixels move when the view moves by a thousandth of a pixel (what no one sees)
        let pixel = 2.5 * scale / width as f64;
        let shifted = render_fixed(&m, &real.plus(pixel * 1e-3), &imaginary.plus(pixel * 1e-3), (scale, 0.0), width, height, false, &cancel).unwrap();
        println!("baseline, a shift of 1/1000 pixel: moved by over 0.01: {:.2}%, 1: {:.2}%", moved_by(&shifted, 0.01), moved_by(&shifted, 1.0));
        let inside = exact.iter().filter(|position| **position >= f64::from(iterations)).count() as f64 * 100.0 / exact.len() as f64;
        let mean = exact.iter().filter(|position| **position < f64::from(iterations)).sum::<f64>() / exact.iter().filter(|position| **position < f64::from(iterations)).count().max(1) as f64;
        println!("{spec}: {inside:.1}% inside, mean position {mean:.0}; step by step {exact_time:.2?}, BLA {approximate_time:.2?} ({:.1}x); pixels moved by over 0.001: {:.2}%, 0.01: {:.2}%, 0.1: {:.2}%, 1: {:.2}%; largest {:.3}",
            exact_time.as_secs_f64() / approximate_time.as_secs_f64(), moved(0.001), moved(0.01), moved(0.1), moved(1.0), differences.iter().copied().fold(0.0, f64::max));
    }

    #[test]
    fn a_view_of_ten_to_the_minus_forty_has_structure() {
        //Around c = i, a point of the boundary (a Misiurewicz point: its orbit 0, i, i - 1, -i, i - 1, ... is periodic after a step), which has structure at
        //every depth; 10⁻⁴⁰ wide, where the 64-bit loop would give one value for every pixel
        let m = MandelbrotSet::new(5000, 2.0);
        let positions = render_positions(&m, "0.0000000000000000000000000000000000000000003", "1.0000000000000000000000000000000000000000002", 1e-40, 48, 32).unwrap();
        let distinct: std::collections::BTreeSet<u64> = positions.iter().map(|position| (position * 1000.0) as u64).collect();
        assert!(distinct.len() > 20, "the pixels differ: {} different positions", distinct.len());
    }
}

#[cfg(test)]
mod depth_timing {
    use super::*;

    /// Times views around c = i at depths from 10⁻¹⁰⁰ to 10⁻¹⁰⁰⁰ (issue #263), step by step and with BLA. Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn time_views_at_depths() {
        let m = MandelbrotSet::new(50_000, 2.0);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let (width, height) = (480, 270);
        for power in [-100_i64, -200, -300, -308, -500, -1000] {
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let bits = bits_for_depth(depth, width);
            let (cx, cy) = (Fixed::parse(&format!("3e{}", power - 1), bits).unwrap(), Fixed::parse(&format!("1.{}2", "0".repeat((-power) as usize)), bits).unwrap());
            let started = std::time::Instant::now();
            let step_by_step = render_depth(&m, &cx, &cy, (depth, 0.0), width, height, false, &cancel).unwrap();
            let step_time = started.elapsed();
            let started = std::time::Instant::now();
            let with_bla = render_depth(&m, &cx, &cy, (depth, 0.0), width, height, true, &cancel).unwrap();
            let bla_time = started.elapsed();
            let mean = step_by_step.iter().sum::<f64>() / step_by_step.len() as f64;
            let moved = step_by_step.iter().zip(&with_bla).filter(|(a, b)| (*a - *b).abs() > 0.01).count();
            println!("10^{power}: {bits} bits, mean position {mean:.0}; step by step {:.0} ms, BLA {:.0} ms ({:.1}x), {moved} pixels moved by over 0.01", step_time.as_secs_f64() * 1000.0, bla_time.as_secs_f64() * 1000.0, step_time.as_secs_f64() / bla_time.as_secs_f64());
        }
    }

    /// Times the frames of a dive into c = i (whose orbit never escapes, so it runs to the limit, as at a minibrot) from 10⁻⁵⁰ to 10⁻³⁰⁸ and to 10⁻¹⁰⁰⁰:
    /// each frame with its own orbit against one shared orbit (issue #263, step 3), and how many pixels differ. Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn time_dives_with_one_orbit() {
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let color_of = |position: f64| {
            let code = (position * 1000.0) as u32;
            crate::coloring::TrueColor::new((code & 255) as u8, (code >> 8 & 255) as u8, (code >> 16) as u8)
        };
        for (deepest, iterations, (width, height)) in [(-308_i64, 50_000_u32, (480, 270)), (-308, 50_000, (1920, 1080)), (-1000, 50_000, (480, 270)), (-1000, 200_000, (480, 270))] {
            let m = MandelbrotSet::new(iterations, 2.0);
            let frames: Vec<Depth> = (0..12).map(|k| Depth::from_log2((-50.0 + (deepest as f64 + 50.0) * f64::from(k) / 11.0) * std::f64::consts::LOG2_10)).collect();
            let (real, imaginary) = ("0", "1");
            let started = std::time::Instant::now();
            let mut own = Vec::new();
            for depth in &frames {
                let bits = bits_for_depth(*depth, width);
                let (cx, cy) = (Fixed::parse(real, bits).unwrap(), Fixed::parse(imaginary, bits).unwrap());
                render_layers(&m, &cx, &cy, (*depth, 0.0), (width, height), 1, &color_of, true, &cancel, |colors, _| own.push(colors.to_vec()));
            }
            let own_time = started.elapsed();
            let started = std::time::Instant::now();
            let bits = bits_for_depth(*frames.last().unwrap(), width);
            let shared = SharedOrbit::new(&Fixed::parse(real, bits).unwrap(), &Fixed::parse(imaginary, bits).unwrap());
            let mut one = Vec::new();
            for depth in &frames {
                render_layers_around(&m, (&shared, (0.0, 0.0)), (*depth, 0.0), (width, height), 1, &color_of, true, &cancel, |colors, _| one.push(colors.to_vec()));
            }
            let shared_time = started.elapsed();
            let differ: usize = own.iter().zip(&one).map(|(a, b)| a.iter().zip(b).filter(|(a, b)| (a.red, a.green, a.blue) != (b.red, b.green, b.blue)).count()).sum();
            println!("to 10^{deepest}, {iterations} iterations, {width}x{height}, 12 frames: own orbits {:.0} ms, one orbit {:.0} ms ({:.1}x); {differ} of {} pixels differ (by over 0.001)",
                own_time.as_secs_f64() * 1000.0, shared_time.as_secs_f64() * 1000.0, own_time.as_secs_f64() / shared_time.as_secs_f64(), width * height * 12);
        }
    }

    /// Times the reference orbit and the BLA table of a view around c = i (whose orbit never escapes) at depths and limits (issue #263, step 3: the share of
    /// a frame that one orbit for a whole dive saves). Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn time_reference_orbits() {
        for power in [-100_i64, -308, -1000] {
            for iterations in [50_000_u32, 500_000] {
                let depth = Depth::parse(&format!("1e{power}")).unwrap();
                let bits = bits_for_depth(depth, 1920);
                let (cx, cy) = (Fixed::parse(&format!("3e{}", power - 1), bits).unwrap(), Fixed::parse(&format!("1.{}2", "0".repeat((-power) as usize)), bits).unwrap());
                let started = std::time::Instant::now();
                let orbit = reference_orbit(&cx, &cy, iterations);
                let orbit_time = started.elapsed();
                let started = std::time::Instant::now();
                let table = Reference::table(&orbit, 1e-300, 2.0);
                let table_time = started.elapsed();
                println!("10^{power}, {iterations} iterations, {bits} bits: orbit {:.0} ms ({} steps), BLA table {:.1} ms ({} levels)", orbit_time.as_secs_f64() * 1000.0, orbit.len(), table_time.as_secs_f64() * 1000.0, table.len());
            }
        }
    }
}

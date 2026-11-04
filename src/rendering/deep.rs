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
//! So there are no glitches to detect and fix (issue #263, step 4): next to the boundary points i, -i and -2, at depths from 10⁻⁵⁰ to 10⁻⁵⁰⁰ and with the
//! orbit at the view's center or 37 or 1000 views away from it, every pixel has the escape count of exact arithmetic, with and without BLA (the tests and
//! `survey_glitches`). At the tip c = -2, the only point of the set on the circle of radius 2 where orbits escape, the orbit runs along that circle, where
//! 64-bit floats cannot tell its sides: there the escape test is exact (`outside`, with the rounding residuals of the reference orbit).
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

    /// This number as `mantissa · 2^-power` with the mantissa a 64-bit float in [0.5, 1) (or 0), also where the number is beyond 64-bit floats
    pub fn to_scaled(&self) -> (f64, i32) {
        let length = self.value.bits();
        if length == 0 {
            return (0.0, 0);
        }
        let shift = length.saturating_sub(64);
        let top = (&self.value >> shift).to_f64().unwrap_or(0.0);
        let mantissa = times_two_to(top, -i32::try_from(length - shift).unwrap_or(i32::MAX));
        (mantissa, i32::try_from(i64::from(self.bits) - i64::try_from(length).unwrap_or(i64::MAX)).unwrap_or(i32::MAX))
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

    /// `(self - other) · 2^-power` as the nearest 64-bit float (how far apart two deep centers are, in units of a view's `2^exponent`, see `Depth`)
    pub fn offset_from(&self, other: &Fixed, power: i32) -> f64 {
        let bits = self.bits.max(other.bits);
        let difference = Fixed { value: self.with_bits(bits).value - other.with_bits(bits).value, bits };
        let (mantissa, scaled) = difference.to_scaled();
        times_two_to(mantissa, (-i64::from(scaled) - i64::from(power)).clamp(-4000, 4000) as i32) //(Beyond ±4000 it is 0 or infinite anyway)
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
    reference_orbit_with_residuals(real, imaginary, max_iterations).0
}

/// The rounding residual of a point of a reference orbit: the exact point less its 64-bit float.
///
/// As mantissas and powers of two (`mantissa · 2^-power` each), since it can be far smaller than 64-bit floats reach (at the tip c = -2 the orbit is within
/// 10⁻⁵⁰⁰ of 2, say; see `outside`)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Residual {
    x: (f64, i32),
    y: (f64, i32),
}

impl Residual {
    /// The residual times `2^scaled` (in the units of a scaled difference, see `position_scaled`; 0 for the plain residual)
    fn times(self, scaled: i32) -> (f64, f64) {
        (times_two_to(self.x.0, scaled.saturating_sub(self.x.1)), times_two_to(self.y.0, scaled.saturating_sub(self.y.1)))
    }
}

/// `reference_orbit`, and the rounding residual of every point (for the escape test next to the circle, see `outside`)
pub fn reference_orbit_with_residuals(real: &Fixed, imaginary: &Fixed, max_iterations: u32) -> (Vec<(f64, f64)>, Vec<Residual>) {
    let mut z = (Fixed { value: BigInt::zero(), bits: real.bits }, Fixed { value: BigInt::zero(), bits: real.bits });
    let (mut orbit, mut residuals) = (Vec::with_capacity(max_iterations as usize + 1), Vec::with_capacity(max_iterations as usize + 1));
    orbit.push((0.0, 0.0));
    residuals.push(Residual::default());
    continue_orbit((real, imaginary), &mut z, (&mut orbit, &mut residuals), max_iterations);
    (orbit, residuals)
}

/// Continues the orbit of `c` from `z` (its last point, in fixed point) for up to `steps` steps, adding them to `orbit`; whether it escaped
fn continue_orbit((real, imaginary): (&Fixed, &Fixed), z: &mut (Fixed, Fixed), (orbit, residuals): (&mut Vec<(f64, f64)>, &mut Vec<Residual>), steps: u32) -> bool {
    let (zx, zy) = z;
    for _ in 0..steps {
        let (xx, yy, xy) = (zx.mul(zx), zy.mul(zy), zx.mul(zy));
        (*zx, *zy) = (xx.sub(&yy).add(real), xy.double().add(imaginary));
        let (x, y) = (zx.to_f64(), zy.to_f64());
        orbit.push((x, y));
        //(The 64-bit float of a point of the orbit has at most 53 bits, which the fixed point holds exactly: the difference is exact before it is rounded)
        residuals.push(Residual { x: zx.plus(-x).to_scaled(), y: zy.plus(-y).to_scaled() });
        if zx.magnitude_above(8) || zy.magnitude_above(8) {
            return true;
        }
    }
    false
}

/// The points of a reference orbit and their rounding residuals, shared by the views that use them (see `SharedOrbit::orbit_and_residuals`)
pub type OrbitAndResiduals = (std::sync::Arc<Vec<(f64, f64)>>, std::sync::Arc<Vec<Residual>>);

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
    residuals: std::sync::Arc<Vec<Residual>>,
    escaped: bool,
}

impl SharedOrbit {
    /// The orbit of `(real, imaginary)`, none of it computed yet
    pub fn new(real: &Fixed, imaginary: &Fixed) -> SharedOrbit {
        let zero = Fixed { value: BigInt::zero(), bits: real.bits };
        SharedOrbit { real: real.clone(), imaginary: imaginary.clone(), state: std::sync::Mutex::new(OrbitSoFar { z: (zero.clone(), zero), points: std::sync::Arc::new(vec![(0.0, 0.0)]), residuals: std::sync::Arc::new(vec![Residual::default()]), escaped: false }) }
    }

    /// The center
    pub const fn center(&self) -> (&Fixed, &Fixed) {
        (&self.real, &self.imaginary)
    }

    /// The orbit up to at least `max_iterations` steps, or until it escaped (it may be longer)
    pub fn orbit(&self, max_iterations: u32) -> std::sync::Arc<Vec<(f64, f64)>> {
        self.orbit_and_residuals(max_iterations).0
    }

    /// `orbit`, and the rounding residuals of its points (see `reference_orbit_with_residuals`)
    pub fn orbit_and_residuals(&self, max_iterations: u32) -> OrbitAndResiduals {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let have = u32::try_from(state.points.len() - 1).unwrap_or(u32::MAX);
        if !state.escaped && have < max_iterations {
            let (mut longer, mut residuals) = (Vec::with_capacity(max_iterations as usize + 1), Vec::with_capacity(max_iterations as usize + 1));
            longer.extend_from_slice(&state.points);
            residuals.extend_from_slice(&state.residuals);
            let OrbitSoFar { z, .. } = &mut *state;
            let mut z = z.clone();
            let escaped = continue_orbit((&self.real, &self.imaginary), &mut z, (&mut longer, &mut residuals), max_iterations - have);
            (state.z, state.escaped, state.points, state.residuals) = (z, escaped, std::sync::Arc::new(longer), std::sync::Arc::new(residuals));
        }
        let points = (std::sync::Arc::clone(&state.points), std::sync::Arc::clone(&state.residuals));
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

/// What the GPU's deep loop needs besides the orbit (see `Reference::gpu_table`, `GpuIterator::deep_outcomes`): the approximations, five numbers each (A, B,
/// the square of the radius), level after level, where every level starts and how many it has, and the residuals of the orbit
struct GpuTable {
    table: Vec<f64>,
    starts: Vec<u32>,
    lengths: Vec<u32>,
    residuals: Vec<super::gpu::DeepResidual>,
}

impl GpuTable {
    /// The input of a dispatch for the orbit `orbit`
    fn input<'a>(&'a self, orbit: &'a [(f64, f64)]) -> super::gpu::DeepInput<'a> {
        super::gpu::DeepInput { orbit, residuals: &self.residuals, table: &self.table, starts: &self.starts, lengths: &self.lengths }
    }
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
    /// The rounding residual of every point of the orbit (see `outside`)
    residuals: std::sync::Arc<Vec<Residual>>,
    length: usize,
    /// `levels[l][k]`: the approximation of the `2^(l + 1)` steps from step `1 + k · 2^(l + 1)` of the orbit
    levels: Vec<Vec<Bla>>,
    /// `widest[at]`: the largest square of a radius of the approximations from step `at` (0 without any): a difference at least that large takes none,
    /// which `skip` sees in one comparison (most steps near the escape, where the differences are large)
    widest: Vec<f64>,
}

impl Reference {
    /// The reference orbit of the center `(real, imaginary)`, for a view whose pixels are at most `max_offset` from it; with `approximate` also its table
    /// of bilinear approximations (without it, every step is iterated, as in `position` before BLA)
    pub fn new(m: &MandelbrotSet, (real, imaginary): (&Fixed, &Fixed), max_offset: f64, approximate: bool) -> Reference {
        let (orbit, residuals) = reference_orbit_with_residuals(real, imaginary, m.max_iterations);
        Reference::of(m, (std::sync::Arc::new(orbit), std::sync::Arc::new(residuals)), max_offset, approximate)
    }

    /// `new` with the orbit `orbit` (of at least `m.max_iterations` steps, or until it escaped), see `SharedOrbit`
    pub fn of(m: &MandelbrotSet, (orbit, residuals): OrbitAndResiduals, max_offset: f64, approximate: bool) -> Reference {
        let length = orbit.len().min(m.max_iterations as usize + 1);
        let levels = if approximate { Self::table(&orbit[..length], max_offset, m.orbit_radius) } else { Vec::new() };
        let widest = if levels.is_empty() { Vec::new() } else {
            (0..length).map(|at| if at == 0 { 0.0 } else {
                let aligned = ((at - 1).trailing_zeros() as usize).min(levels.len());
                (1..=aligned).filter_map(|level| levels[level - 1].get((at - 1) >> level)).map(|bla| bla.radius_squared).fold(0.0, f64::max)
            }).collect()
        };
        Reference { orbit, residuals, length, levels, widest }
    }

    /// The points of the orbit of this view
    fn points(&self) -> &[(f64, f64)] {
        &self.orbit[..self.length]
    }

    /// The rounding residuals of the points of the orbit of this view (see `outside`)
    fn residuals(&self) -> &[Residual] {
        &self.residuals[..self.length]
    }

    /// The table of bilinear approximations for the GPU (see `GpuIterator::deep_outcomes`): A, B and the square of the radius of each, level after level,
    /// where every level starts and how many it has, and the residuals of the orbit's points
    fn gpu_table(&self) -> GpuTable {
        let (mut table, mut starts, mut lengths) = (Vec::new(), Vec::new(), Vec::new());
        for level in self.levels.iter().take(32) {
            starts.push(u32::try_from(table.len() / 5).unwrap_or(u32::MAX));
            lengths.push(u32::try_from(level.len()).unwrap_or(u32::MAX));
            table.extend(level.iter().flat_map(|bla| [bla.a.0, bla.a.1, bla.b.0, bla.b.1, bla.radius_squared]));
        }
        let residuals = self.residuals().iter().map(|residual| super::gpu::DeepResidual { mx: residual.x.0, my: residual.y.0, px: residual.x.1, py: residual.y.1 }).collect();
        GpuTable { table, starts, lengths, residuals }
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
        //(Larger than every approximation from here: none of them, as the search below would find)
        if self.widest.get(at).is_none_or(|widest| size_squared >= *widest) {
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

/// `a · b` as a sum of two 64-bit floats, exactly (Dekker's product, with Veltkamp's split: no fused multiply-add, so the GPU gives the same, see
/// `gpu_deep.wgsl`)
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let split = |x: f64| {
        let t = 134_217_729.0 * x;
        let high = t - (t - x);
        (high, x - high)
    };
    let product = a * b;
    let ((a_high, a_low), (b_high, b_low)) = (split(a), split(b));
    (product, ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low)
}

/// `a + b` as a sum of two 64-bit floats, exactly (Knuth's sum)
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    (sum, (a - (sum - b_virtual)) + (b - b_virtual))
}

/// Whether the pixel's orbit, the point `Z + ε + z` of the reference orbit `Z` (its 64-bit float), its rounding residual `ε` and the pixel's difference `z`
/// (scaled by `2^scaled`, see `position_scaled`; 0 for a plain one), is outside the circle of `radius_squared`, where `full` is `|Z + z|²` in 64-bit floats.
///
/// `full` decides, except within 10⁻¹² of the circle: there 64-bit floats cannot tell the side, and at the tip c = -2 of the set, the one point of it on
/// the circle of radius 2, the orbit runs along the circle (2 → 2 → ...), so deep views there crossed it in bands of the wrong count (issue #263). Then the
/// side is that of `|Z + ε + z|² - R² = (|Z|² - R²) + 2·Z·(ε + z) + |ε + z|²`, with `|Z|² - R²` computed exactly (`two_product`, `two_sum`) and all of it
/// in units of `2^-scaled`, where the residual and the difference, however small, are numbers
fn outside(full: f64, radius_squared: f64, (ox, oy): (f64, f64), residual: Residual, (zx, zy): (f64, f64), scaled: i32) -> bool {
    //(One comparison for the many points well inside, as before)
    if full < radius_squared * (1.0 - 1e-12) {
        return false;
    }
    if full > radius_squared * (1.0 + 1e-12) {
        return true;
    }
    let (ex, ey) = residual.times(scaled);
    let (dx, dy) = (ex + zx, ey + zy);
    let ((xx, xx_low), (yy, yy_low)) = (two_product(ox, ox), two_product(oy, oy));
    let (sum, sum_low) = two_sum(xx, yy);
    let (beyond, beyond_low) = two_sum(sum, -radius_squared);
    //(|Z|² - R² in units of the difference: infinite when the point is far from the circle at that scale, where the difference no longer counts)
    let away = times_two_to(beyond, scaled) + times_two_to((sum_low + beyond_low) + (xx_low + yy_low), scaled);
    away + (2.0 * (ox * dx + oy * dy) + times_two_to(dx * dx + dy * dy, -scaled)) > 0.0
}

/// The smooth position (see `MandelbrotSet::position`) of the point at offset `(dcx, dcy)` from the center of `reference`, by perturbation with rebasing.
///
/// `center` is the center in 64-bit floats, for the steps after the point escaped; see the module, and `Reference` for the steps it skips
pub fn position(m: &MandelbrotSet, reference: &Reference, center: (f64, f64), (dcx, dcy): (f64, f64)) -> f64 {
    position_from(m, reference, center, (dcx, dcy), ((0.0, 0.0), 0, 0))
}

/// `position` from the difference `z` at step `at` of the reference orbit, after `step` steps of the pixel (`position` starts at 0, 0, 0; see `position_scaled`)
fn position_from(m: &MandelbrotSet, reference: &Reference, center: (f64, f64), (dcx, dcy): (f64, f64), ((zx, zy), at, step): ((f64, f64), usize, u32)) -> f64 {
    let (orbit, residuals) = (reference.points(), reference.residuals());
    let radius_squared = m.orbit_radius * m.orbit_radius;
    let near_circle = radius_squared * (1.0 - 1e-12);
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
        //(The residual only next to the circle: elsewhere one comparison, as fast as before, see `outside`)
        if full >= near_circle && outside(full, radius_squared, orbit[at], residuals[at], (zx, zy), 0) {
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

/// The pixels that `positions` iterates at the same time on one thread
const LANES: usize = 4;

/// The state of a pixel in `positions`: its number, its `c` from the center, its difference from the reference orbit, the step of that orbit, its steps
struct Lane {
    index: usize,
    c: (f64, f64),
    z: (f64, f64),
    at: usize,
    step: u32,
}

/// `position` of the points `offset_of(index)` from the center for every index of `indices`, `LANES` at a time.
///
/// Each point does exactly the steps of `position`, in the same order, but the steps of `LANES` points are interleaved, so the processor works on one
/// while another waits for its last result (a step waits for the one before it). A point that is done makes room for the next. The same positions, bit
/// for bit
pub fn positions(m: &MandelbrotSet, reference: &Reference, center: (f64, f64), indices: std::ops::Range<usize>, offset_of: impl Fn(usize) -> (f64, f64)) -> Vec<f64> {
    let (orbit, residuals) = (reference.points(), reference.residuals());
    let radius_squared = m.orbit_radius * m.orbit_radius;
    let near_circle = radius_squared * (1.0 - 1e-12);
    let start = indices.start;
    let mut out = vec![0.0; indices.len()];
    let mut pending = indices;
    let mut next_lane = || pending.next().map(|index| Lane { index, c: offset_of(index), z: (0.0, 0.0), at: 0, step: 0 });
    let mut lanes: [Option<Lane>; LANES] = std::array::from_fn(|_| next_lane());
    loop {
        let mut active = false;
        for slot in &mut lanes {
            let Some(lane) = slot else { continue };
            active = true;
            let (dcx, dcy) = lane.c;
            //(As `position_from`, one step)
            let done = if lane.step >= m.max_iterations {
                Some(f64::from(m.max_iterations))
            } else {
                let (mut zx, mut zy) = lane.z;
                if let Some((bla, steps)) = reference.skip(lane.at, zx * zx + zy * zy, m.max_iterations - lane.step) {
                    (zx, zy) = (bla.a.0 * zx - bla.a.1 * zy + (bla.b.0 * dcx - bla.b.1 * dcy), bla.a.0 * zy + bla.a.1 * zx + (bla.b.0 * dcy + bla.b.1 * dcx));
                    lane.at += steps as usize;
                    lane.step += steps;
                } else {
                    let (ox, oy) = orbit[lane.at];
                    (zx, zy) = (2.0 * (ox * zx - oy * zy) + (zx * zx - zy * zy) + dcx, 2.0 * (ox * zy + oy * zx) + 2.0 * zx * zy + dcy);
                    lane.at += 1;
                    lane.step += 1;
                }
                let (fx, fy) = (orbit[lane.at].0 + zx, orbit[lane.at].1 + zy);
                let full = fx * fx + fy * fy;
                if full >= near_circle && outside(full, radius_squared, orbit[lane.at], residuals[lane.at], (zx, zy), 0) {
                    Some(m.smooth_position_of(lane.step, (fx, fy), (center.0 + dcx, center.1 + dcy)))
                } else {
                    if full < zx * zx + zy * zy || lane.at + 1 >= orbit.len() {
                        (zx, zy) = (fx, fy);
                        lane.at = 0;
                    }
                    lane.z = (zx, zy);
                    None
                }
            };
            if let Some(position) = done {
                out[lane.index - start] = position;
                *slot = next_lane();
            }
        }
        if !active {
            return out;
        }
    }
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

/// `SCALED_BELOW` of the GPU's 32-bit deep loop (issue #288), within the range of 32-bit floats
const SCALED_BELOW_32: i32 = 100;

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
        //The pixel's orbit is the center's to the last bit here: it escapes with it (on the circle, by the side `z` is on, see `outside_scaled`)
        let (fx, fy) = orbit[at];
        let full = fx * fx + fy * fy;
        if full >= radius_squared * (1.0 - 1e-12) && outside(full, radius_squared, orbit[at], reference.residuals()[at], (zx, zy), scaled) {
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
    color_of: &(impl Fn(f64) -> crate::coloring::TrueColor + Sync), approximate: bool, cancel: &std::sync::atomic::AtomicBool, on_layer: impl FnMut(&[crate::coloring::TrueColor], u8)) -> bool {
    render_layers_on(m, (orbit, (right, up)), (depth, rotation), (width, height), samples, color_of, approximate, None, cancel, on_layer)
}

/// `render_layers_around`, with the positions of every layer on `gpu` when it can.
///
/// That is when it iterates deep views (`GpuIterator::deep_works`: bit for bit the CPU's) and the offsets fit 64-bit floats (to about 10⁻³⁰⁰, see
/// `position_scaled`); the CPU computes the rest
#[allow(clippy::too_many_arguments)] // The orbit and the view, its size and samples, how to color, the GPU, when to stop and where the layers go
pub fn render_layers_on(m: &MandelbrotSet, (orbit, (right, up)): (&SharedOrbit, (f64, f64)), (depth, rotation): (Depth, f64), (width, height): (usize, usize), samples: u8,
    color_of: &(impl Fn(f64) -> crate::coloring::TrueColor + Sync), approximate: bool, gpu: Option<(&super::gpu::GpuIterator, bool)>, cancel: &std::sync::atomic::AtomicBool,
    on_layer: impl FnMut(&[crate::coloring::TrueColor], u8)) -> bool {
    //A pixel is `pixel · 2^depth.exponent`: the offsets are scaled by `2^-depth.exponent` (see `position_scaled`; within 64-bit floats' range bit for bit as before)
    let pixel = 2.5 * depth.mantissa / width as f64;
    //(The view's center from the orbit's, in the same scaled units)
    let shift = (right * depth.mantissa, up * depth.mantissa);
    let reach = max_offset(pixel, width, height) + shift.0.hypot(shift.1);
    //The offset of a pixel's sample from the orbit's center, scaled by `2^-depth.exponent`: the first sample in the middle of the pixel, the others at the
    //jitter of the 64-bit renderer
    let offset_of = |index: usize, sample: u8| {
        let (x, y) = (index % width, index / width);
        let (jx, jy) = if sample == 0 { (0.5, 0.5) } else { super::jitter::sample_jitter(x, y, sample) };
        let offset = turned(((x as f64 + jx - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - jy) * pixel), rotation);
        if shift == (0.0, 0.0) { offset } else { (offset.0 + shift.0, offset.1 + shift.1) }
    };
    render_points_on(m, orbit, (depth.exponent, reach), (width * height, &offset_of), samples, color_of, approximate, gpu, cancel, on_layer)
}

/// The colors of `pixels` points around the center of `orbit`, averaged over `samples` samples each, one layer after the other.
///
/// `render_layers_on` renders a grid of them, #287 rings; the offset of a point's sample from the center is `offset_of(point, sample) · 2^exponent`, at
/// most `reach · 2^exponent`
#[allow(clippy::too_many_arguments)] // The orbit and the points, their samples, how to color, the GPU, when to stop and where the layers go
pub fn render_points_on(m: &MandelbrotSet, orbit: &SharedOrbit, (exponent, reach): (i32, f64), (pixels, offset_of): (usize, &(impl Fn(usize, u8) -> (f64, f64) + Sync)), samples: u8,
    color_of: &(impl Fn(f64) -> crate::coloring::TrueColor + Sync), approximate: bool, gpu: Option<(&super::gpu::GpuIterator, bool)>, cancel: &std::sync::atomic::AtomicBool,
    mut on_layer: impl FnMut(&[crate::coloring::TrueColor], u8)) -> bool {
    use std::sync::atomic::Ordering;
    let scaled = -exponent;
    let reference = Reference::of(m, orbit.orbit_and_residuals(m.max_iterations), times_two_to(reach, exponent), approximate);
    let (center_real, center_imaginary) = orbit.center();
    let center = (center_real.to_f64(), center_imaginary.to_f64());
    let threads = super::render_thread_count().max(1);
    let per_thread = pixels.div_ceil(threads).max(1);
    let mut sums = vec![[0_u32; 3]; pixels];
    let mut colors = vec![crate::coloring::TrueColor::new(0, 0, 0); pixels];
    //(With `true`: the 32-bit loop, an approximation, see `GpuIterator::deep_outcomes32`)
    let gpu = gpu.filter(|(gpu, single)| if *single { gpu.has_deep32() } else { gpu.deep_works() });
    let table = gpu.map(|_| reference.gpu_table());
    let chunks = pixels.div_ceil(SHARE_CHUNK);
    for sample in 0..samples.max(1) {
        //The positions of the layer: the CPU threads and the GPU (where it can) take the next chunk of pixels as soon as they are free, so each computes as
        //much as its speed on this view allows, without a setting or a measurement (issue #263); both give the same positions, bit for bit
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (sender, receiver) = std::sync::mpsc::channel::<(usize, Vec<f64>)>();
        //(Within 64-bit floats' range, `LANES` pixels at a time, see `positions`; deeper, the scaled phase first, see `position_scaled`)
        let cpu_positions = |range: std::ops::Range<usize>| -> Vec<f64> {
            if scaled <= SCALED_BELOW {
                positions(m, &reference, center, range, |index| {
                    let (cx, cy) = offset_of(index, sample);
                    (times_two_to(cx, -scaled), times_two_to(cy, -scaled))
                })
            } else {
                range.map(|index| position_scaled(m, &reference, center, offset_of(index, sample), scaled)).collect()
            }
        };
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (sender, next, cpu_positions) = (sender.clone(), &next, &cpu_positions);
                scope.spawn(move || loop {
                    let chunk = next.fetch_add(1, Ordering::Relaxed);
                    if chunk >= chunks || cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let range = chunk * SHARE_CHUNK..((chunk + 1) * SHARE_CHUNK).min(pixels);
                    let _ = sender.send((range.start, cpu_positions(range)));
                });
            }
            if let Some(((gpu, single), table)) = gpu.zip(table.as_ref()) {
                let (sender, next, cpu_positions, offset_of, reference) = (sender.clone(), &next, &cpu_positions, &offset_of, &reference);
                scope.spawn(move || loop {
                    let chunk = next.fetch_add(GPU_CHUNKS, Ordering::Relaxed);
                    if chunk >= chunks || cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let range = chunk * SHARE_CHUNK..((chunk + GPU_CHUNKS) * SHARE_CHUNK).min(pixels);
                    //The offsets as `position_scaled` takes them: unscaled at this depth, else scaled (the GPU does the scaled phase too)
                    let plain = scaled <= if single { SCALED_BELOW_32 } else { SCALED_BELOW };
                    let offsets: Vec<(f64, f64)> = range.clone().map(|index| {
                        let (cx, cy) = offset_of(index, sample);
                        if plain { (times_two_to(cx, -scaled), times_two_to(cy, -scaled)) } else { (cx, cy) }
                    }).collect();
                    //The smooth position of an escaped orbit as `position_from` and `position_scaled` end (in the scaled phase: of the center); the GPU failed:
                    //these on the CPU
                    let scale = if plain { 0 } else { scaled };
                    let dispatched = if single { gpu.deep_outcomes32(m, &table.input(reference.points()), &offsets, scale) } else { gpu.deep_outcomes(m, &table.input(reference.points()), &offsets, scale) };
                    let positions = dispatched.map_or_else(|_| cpu_positions(range.clone()), |outcomes| {
                        offsets.iter().zip(&outcomes).map(|(&offset, outcome)| position_of_outcome(m, center, offset, scale, outcome)).collect()
                    });
                    let _ = sender.send((range.start, positions));
                });
            }
        });
        drop(sender);
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let mut positions = vec![0.0; pixels];
        for (start, part) in receiver {
            positions[start..start + part.len()].copy_from_slice(&part);
        }
        //The colors, added to the samples so far
        std::thread::scope(|scope| {
            for ((rows, out), positions) in sums.chunks_mut(per_thread).zip(colors.chunks_mut(per_thread)).zip(positions.chunks(per_thread)) {
                scope.spawn(move || {
                    for ((sum, color), position) in rows.iter_mut().zip(out.iter_mut()).zip(positions) {
                        let new = color_of(*position);
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

/// The smooth position of the GPU's outcome for the offset `offset` (scaled by `2^scaled`, 0: plain) from `center`, as `position_from` and
/// `position_scaled` end: of the pixel's point when it escaped, of the center when it escaped in the scaled phase (where the two are the same to the last
/// bit), the limit when it did not
fn position_of_outcome(m: &MandelbrotSet, center: (f64, f64), (cx, cy): (f64, f64), scaled: i32, outcome: &super::gpu::DeepOutcome) -> f64 {
    match outcome.escaped {
        1 => m.smooth_position_of(outcome.steps, (outcome.zx, outcome.zy), (center.0 + times_two_to(cx, -scaled), center.1 + times_two_to(cy, -scaled))),
        2 => m.smooth_position_of(outcome.steps, (outcome.zx, outcome.zy), center),
        _ => f64::from(m.max_iterations),
    }
}

/// The share of pixels whose iteration count the GPU's 32-bit deep loop (`GpuIterator::deep_outcomes32`, issue #288) gives as the CPU's 64-bit one, on views
/// next to c = i at 10⁻³⁰, 10⁻²⁰⁰ and 10⁻³²⁰ (the scaled phase) and in the seahorse valley at 10⁻²⁰, with and without BLA: an approximation, so not all;
/// the self-test takes the loop when the share is high (a broken shader gives far less)
pub(crate) fn gpu32_agrees(gpu: &super::gpu::GpuIterator) -> f64 {
    let near_i = |power: i64| (power, format!("3e{}", power - 1), format!("1.{}2", "0".repeat((-power) as usize)));
    let cases = [near_i(-30), near_i(-200), near_i(-320), (-20, String::from("-0.74364388703715870475"), String::from("0.13182590420531197049"))];
    let (mut same, mut all) = (0_usize, 0_usize);
    for (power, real, imaginary) in cases {
        let mut m = MandelbrotSet::new(3000, 2.0);
        m.smooth_coloring = false;
        let (width, height) = (40, 30);
        let Some(depth) = Depth::parse(&format!("1e{power}")) else { return 0.0 };
        let bits = bits_for_depth(depth, width);
        let (Some(cx), Some(cy)) = (Fixed::parse(&real, bits), Fixed::parse(&imaginary, bits)) else { return 0.0 };
        let center = (cx.to_f64(), cy.to_f64());
        let pixel = 2.5 * depth.mantissa / width as f64;
        let scaled = -depth.exponent;
        let plain = scaled <= SCALED_BELOW_32;
        let offsets: Vec<(f64, f64)> = (0..width * height).map(|index| {
            let (x, y) = (index % width, index / width);
            let offset = turned(((x as f64 + 0.5 - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - 0.5) * pixel), 0.3);
            if plain { (times_two_to(offset.0, depth.exponent), times_two_to(offset.1, depth.exponent)) } else { offset }
        }).collect();
        for approximate in [false, true] {
            let reference = Reference::new(&m, (&cx, &cy), times_two_to(max_offset(pixel, width, height), depth.exponent), approximate);
            let table = reference.gpu_table();
            let Ok(outcomes) = gpu.deep_outcomes32(&m, &table.input(reference.points()), &offsets, if plain { 0 } else { scaled }) else { return 0.0 };
            for (&offset, outcome) in offsets.iter().zip(&outcomes) {
                let cpu = position_scaled(&m, &reference, center, if plain { (times_two_to(offset.0, scaled), times_two_to(offset.1, scaled)) } else { offset }, scaled);
                same += usize::from(position_of_outcome(&m, center, offset, if plain { 0 } else { scaled }, outcome).floor().to_bits() == cpu.floor().to_bits());
                all += 1;
            }
        }
    }
    if all == 0 { 0.0 } else { same as f64 / all as f64 }
}

/// Whether `gpu`'s deep loop (`gpu_deep.wgsl`) gives every pixel the CPU's position, bit for bit: views of 10⁻³⁰ and 10⁻²⁰⁰ next to c = i, of 10⁻²⁰ in
/// the seahorse valley and of 10⁻⁶⁰ at the tip c = -2 (across the escape circle, see `outside`), with and without bilinear approximation, smooth and not (the GPU's self-test, see `GpuIterator::deep_works`)
pub(crate) fn gpu_agrees(gpu: &super::gpu::GpuIterator) -> bool {
    let tip = |power: i64| (format!("-1.{}9", "9".repeat((-power) as usize)), format!("4e{}", power - 1));
    let cases = [(-30_i64, String::from("3e-31"), String::from("1.000000000000000000000000000002"), 3000, true), (-200, String::from("3e-201"), format!("1.{}2", "0".repeat(200)), 3000, false),
        (-20, String::from("-0.74364388703715870475"), String::from("0.13182590420531197049"), 4000, true), (-60, tip(-60).0, tip(-60).1, 3000, true),
        //(Beyond 10⁻³⁰⁰: the scaled phase of `position_scaled`)
        (-320, String::from("3e-321"), format!("1.{}2", "0".repeat(320)), 3000, true), (-400, tip(-400).0, tip(-400).1, 3000, false)];
    for (power, real, imaginary, iterations, smooth) in cases {
        let (real, imaginary) = (real.as_str(), imaginary.as_str());
        let mut m = MandelbrotSet::new(iterations, 2.0);
        m.smooth_coloring = smooth;
        let (width, height) = (40, 30);
        let depth = Depth::parse(&format!("1e{power}")).expect("a depth");
        let bits = bits_for_depth(depth, width);
        let (Some(cx), Some(cy)) = (Fixed::parse(real, bits), Fixed::parse(imaginary, bits)) else { return false };
        let center = (cx.to_f64(), cy.to_f64());
        let pixel = 2.5 * depth.mantissa / width as f64;
        //(The offsets as `render_layers_on` gives them: plain to about 10⁻³⁰⁰, scaled by 2^scaled beyond)
        let (scaled, plain) = (-depth.exponent, -depth.exponent <= SCALED_BELOW);
        let offsets: Vec<(f64, f64)> = (0..width * height).map(|index| {
            let (x, y) = (index % width, index / width);
            let (jx, jy) = super::jitter::sample_jitter(x, y, 3);
            let offset = turned(((x as f64 + jx - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - jy) * pixel), 0.3);
            if plain { (times_two_to(offset.0, depth.exponent), times_two_to(offset.1, depth.exponent)) } else { offset }
        }).collect();
        for approximate in [false, true] {
            let reference = Reference::new(&m, (&cx, &cy), times_two_to(max_offset(pixel, width, height), depth.exponent), approximate);
            let table = reference.gpu_table();
            let Ok(outcomes) = gpu.deep_outcomes(&m, &table.input(reference.points()), &offsets, if plain { 0 } else { scaled }) else { return false };
            let agrees = offsets.iter().zip(&outcomes).all(|(&offset, outcome)| {
                let cpu_position = if plain { position(&m, &reference, center, offset) } else { position_scaled(&m, &reference, center, offset, scaled) };
                position_of_outcome(&m, center, offset, if plain { 0 } else { scaled }, outcome).to_bits() == cpu_position.to_bits()
            });
            if !agrees || outcomes.len() != offsets.len() {
                return false;
            }
        }
    }
    true
}

/// The pixels a CPU thread takes at a time from a layer of a deep view (see `render_layers_on`), and the chunks of them the GPU takes at a time
const SHARE_CHUNK: usize = 2048;
const GPU_CHUNKS: usize = 64;

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

    #[test]
    fn the_widest_radius_of_a_step_turns_away_exactly_the_differences_that_no_approximation_takes() {
        //#263: `skip` with its one comparison against `widest` gives what the search of the levels alone gives, for every step and size
        let search = |reference: &Reference, at: usize, size_squared: f64, left: u32| -> Option<(f64, u32)> {
            if at == 0 || reference.levels.is_empty() {
                return None;
            }
            let aligned = ((at - 1).trailing_zeros() as usize).min(reference.levels.len());
            (1..=aligned).rev().find_map(|level| reference.levels[level - 1].get((at - 1) >> level).filter(|bla| (1_u64 << level) <= u64::from(left) && size_squared < bla.radius_squared).map(|bla| (bla.radius_squared, 1_u32 << level)))
        };
        let m = MandelbrotSet::new(3000, 2.0);
        let bits = bits_for_depth(Depth::parse("1e-100").unwrap(), 400);
        let (cx, cy) = (Fixed::parse("3e-101", bits).unwrap(), Fixed::parse(&format!("1.{}2", "0".repeat(100)), bits).unwrap());
        let reference = Reference::new(&m, (&cx, &cy), 1e-100, true);
        assert!(!reference.levels.is_empty());
        let mut taken = 0;
        for at in 0..reference.points().len() {
            for size_squared in [0.0, 1e-250, 1e-210, 1e-200, 1e-150, 1e-100, 1e-50, 1e-10, 1.0, f64::NAN] {
                for left in [1, 2, 7, 64, 3000] {
                    let skipped = reference.skip(at, size_squared, left).map(|(bla, steps)| (bla.radius_squared, steps));
                    assert_eq!(skipped.map(|(radius, steps)| (radius.to_bits(), steps)), search(&reference, at, size_squared, left).map(|(radius, steps)| (radius.to_bits(), steps)), "step {at}, |z|² {size_squared}, {left} left");
                    taken += usize::from(skipped.is_some());
                }
            }
        }
        assert!(taken > 1000, "{taken} approximations taken");
    }

    #[test]
    fn pixels_iterated_four_at_a_time_have_exactly_the_positions_of_one_at_a_time() {
        //#263: `positions` interleaves the steps of `LANES` pixels; each does the steps of `position`, so the positions are the same, bit for bit (near i
        //at 10^-100, in the seahorse valley at 10^-30 with points inside the set, at the tip c = -2; with and without BLA; and a count not a multiple of 4)
        let near_i = (format!("3e{}", -101), format!("1.{}2", "0".repeat(100)), -100_i32, 3000_u32);
        let seahorse = (String::from("-0.743643887037158704752191506114774"), String::from("0.131825904205311970493132056385139"), -30, 5000);
        let tip = (String::from("-2"), String::from("0"), -40, 2000);
        for (real, imaginary, power, limit) in [near_i, seahorse, tip] {
            let m = MandelbrotSet::new(limit, 2.0);
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let (width, height) = (61, 37);
            let bits = bits_for_depth(depth, width);
            let (cx, cy) = (Fixed::parse(&real, bits).unwrap(), Fixed::parse(&imaginary, bits).unwrap());
            let pixel = 2.5 * depth.to_f64() / width as f64;
            let offset_of = |index: usize| (((index % width) as f64 + 0.5 - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - (index / width) as f64 - 0.5) * pixel);
            for approximate in [false, true] {
                let reference = Reference::new(&m, (&cx, &cy), max_offset(pixel, width, height), approximate);
                let center = (cx.to_f64(), cy.to_f64());
                let together = positions(&m, &reference, center, 3..width * height, offset_of);
                let alone: Vec<f64> = (3..width * height).map(|index| position(&m, &reference, center, offset_of(index))).collect();
                assert!(together.iter().zip(&alone).all(|(a, b)| a.to_bits() == b.to_bits()), "{real} at 10^{power}, BLA {approximate}");
                //(The seahorse valley at this limit is all inside: the pixels run to the limit there)
                assert!(power == -30 || alone.iter().any(|position| *position < f64::from(limit)), "some escape: {real} at 10^{power}");
            }
        }
    }

    /// The escape counts of the pixels of a `width` x `height` view of `depth` around `(real, imaginary)` in exact fixed-point arithmetic (the reference that
    /// perturbation is checked against): every pixel's own orbit, step by step, until it leaves radius 2
    pub(super) fn exact_counts(real: &str, imaginary: &str, (mantissa, power): (f64, i64), (width, height): (usize, usize), limit: u32) -> Vec<u32> {
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
    fn the_gpu_iterates_deep_views_bit_for_bit_like_the_cpu() {
        //#263: the perturbation loop on the GPU (`gpu_deep.wgsl`), with BLA and rebasing: the positions of the CPU, bit for bit (see `gpu_agrees`)
        let Ok(gpu) = crate::rendering::gpu::GpuIterator::new() else {
            eprintln!("no suitable GPU: skipped");
            return;
        };
        assert!(gpu_agrees(&gpu), "the deep loop of {} differs from the CPU's", gpu.name());
        assert!(gpu.deep_works(), "found so by the self-test");
        //A whole render on the GPU: the same colors as on the CPU
        let mut m = MandelbrotSet::new(3000, 2.0);
        m.smooth_coloring = true;
        let color_of = |position: f64| crate::coloring::TrueColor::new((position * 7.0) as u8, (position * 3.0) as u8, position as u8);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        //(10⁻¹²⁰ with plain differences, 10⁻⁴⁰⁰ with the scaled phase)
        for power in [120, 400] {
            let depth = Depth::parse(&format!("1e-{power}")).unwrap();
            let bits = bits_for_depth(depth, 64);
            let orbit = SharedOrbit::new(&Fixed::parse(&format!("3e-{}", power + 1), bits).unwrap(), &Fixed::parse(&format!("1.{}2", "0".repeat(power)), bits).unwrap());
            for approximate in [false, true] {
                let (mut cpu, mut on_gpu) = (Vec::new(), Vec::new());
                assert!(render_layers_on(&m, (&orbit, (0.4, -1.5)), (depth, 0.7), (64, 48), 3, &color_of, approximate, None, &cancel, |colors, _| cpu = colors.to_vec()));
                assert!(render_layers_on(&m, (&orbit, (0.4, -1.5)), (depth, 0.7), (64, 48), 3, &color_of, approximate, Some((&gpu, false)), &cancel, |colors, _| on_gpu = colors.to_vec()));
                assert!(cpu.iter().zip(&on_gpu).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)), "10^-{power}, BLA {approximate}: the same colors");
            }
        }
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
    fn deep_views_at_the_tip_cross_the_escape_circle_where_exact_arithmetic_does() {
        //#263: at the tip c = -2 the orbit runs along the escape circle (2 → 2 → ...), and 64-bit floats cannot tell its sides; before `outside` the pixels just
        //outside it (escaped at once in exact arithmetic) went on like those inside, in bands across the view. Now every pixel has the exact count, also
        //beyond 64-bit floats (10⁻⁵⁰⁰, where the rounding residuals of the orbit are numbers only with their powers of two) and off-center
        let mut m = MandelbrotSet::new(20_000, 2.0);
        m.smooth_coloring = false;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let color_of = |position: f64| {
            let count = position as u32;
            crate::coloring::TrueColor::new((count & 255) as u8, (count >> 8 & 255) as u8, (count >> 16) as u8)
        };
        let (width, height) = (12, 8);
        for power in [-60_i64, -308, -500] {
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let bits = bits_for_depth(depth, width) + 16;
            let (real, imaginary) = (format!("-1.{}9", "9".repeat((-power) as usize)), format!("4e{}", power - 1));
            let exact = exact_counts(&real, &imaginary, (1.0, power), (width, height), m.max_iterations);
            let distinct: std::collections::BTreeSet<u32> = exact.iter().copied().collect();
            assert!(distinct.contains(&1) && distinct.len() > 5, "10^{power}: the view crosses the circle (count 1) and has structure: {distinct:?}");
            for (right, up) in [(0.0, 0.0), (37.0, -11.0)] {
                let orbit = SharedOrbit::new(&Fixed::parse(&real, bits).unwrap().plus_scaled(right * depth.mantissa, depth.exponent), &Fixed::parse(&imaginary, bits).unwrap().plus_scaled(up * depth.mantissa, depth.exponent));
                for approximate in [false, true] {
                    let mut counts = Vec::new();
                    render_layers_around(&m, (&orbit, (-right, -up)), (depth, 0.0), (width, height), 1, &color_of, approximate, &cancel, |colors, _| {
                        counts = colors.iter().map(|color| u32::from(color.red) | u32::from(color.green) << 8 | u32::from(color.blue) << 16).collect();
                    });
                    assert_eq!(counts, exact, "10^{power}, the orbit ({right}, {up}) views off, BLA {approximate}");
                }
            }
        }
    }

    #[test]
    fn perturbing_from_an_orbit_far_off_the_view_has_no_glitches() {
        //#263 step 4: rebasing (see `position_from`) leaves no glitches to detect: views next to i and -i at 10⁻³⁰⁸, perturbed from an orbit 37 and 1000
        //views away (whose orbit escapes long before the pixels'), have the escape counts of exact arithmetic, with and without BLA (`survey_glitches` looks
        //wider)
        let mut m = MandelbrotSet::new(20_000, 2.0);
        m.smooth_coloring = false;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let color_of = |position: f64| {
            let count = position as u32;
            crate::coloring::TrueColor::new((count & 255) as u8, (count >> 8 & 255) as u8, (count >> 16) as u8)
        };
        let (width, height) = (12, 8);
        let depth = Depth::parse("1e-308").unwrap();
        let bits = bits_for_depth(depth, width) + 16;
        for (real, imaginary) in [(String::from("3e-309"), format!("1.{}2", "0".repeat(308))), (String::from("-2e-309"), format!("-1.{}3", "0".repeat(308)))] {
            let exact = exact_counts(&real, &imaginary, (1.0, -308), (width, height), m.max_iterations);
            for (right, up) in [(37.0, -11.0), (-1000.0, 400.0)] {
                let orbit = SharedOrbit::new(&Fixed::parse(&real, bits).unwrap().plus_scaled(right * depth.mantissa, depth.exponent), &Fixed::parse(&imaginary, bits).unwrap().plus_scaled(up * depth.mantissa, depth.exponent));
                for approximate in [false, true] {
                    let mut counts = Vec::new();
                    render_layers_around(&m, (&orbit, (-right, -up)), (depth, 0.0), (width, height), 1, &color_of, approximate, &cancel, |colors, _| {
                        counts = colors.iter().map(|color| u32::from(color.red) | u32::from(color.green) << 8 | u32::from(color.blue) << 16).collect();
                    });
                    assert_eq!(counts, exact, "next to {imaginary:.4}, the orbit ({right}, {up}) views off, BLA {approximate}");
                }
            }
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

    /// Looks for glitches (issue #263, step 4): views next to the exact boundary points i, -i and -2 at depths to 10⁻⁵⁰⁰, perturbed from an orbit at their
    /// center and from one 37 and 1000 views away (as the frames of a flight are, see `SharedOrbit`), with and without BLA, against exact arithmetic.
    /// Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a survey, run by hand (about a minute)"]
    fn survey_glitches() {
        let mut m = MandelbrotSet::new(20_000, 2.0);
        m.smooth_coloring = false;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let color_of = |position: f64| {
            let count = position as u32;
            crate::coloring::TrueColor::new((count & 255) as u8, (count >> 8 & 255) as u8, (count >> 16) as u8)
        };
        let (width, height) = (16, 12);
        let mut total = (0, 0);
        for power in [-50_i64, -150, -308, -500] {
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let bits = bits_for_depth(depth, width) + 16;
            let digits = (-power) as usize;
            //Next to the point: a few views off it, so the view shows the structure around it
            let points = [("i", format!("3e{}", power - 1), format!("1.{}2", "0".repeat(digits))), ("-i", format!("-2e{}", power - 1), format!("-1.{}3", "0".repeat(digits))),
                ("-2", format!("-1.{}9", "9".repeat(digits)), format!("4e{}", power - 1))];
            for (name, real, imaginary) in &points {
                let exact = super::tests::exact_counts(real, imaginary, (1.0, power), (width, height), m.max_iterations);
                let distinct: std::collections::BTreeSet<u32> = exact.iter().copied().collect();
                for (right, up) in [(0.0, 0.0), (37.0, -11.0), (-1000.0, 400.0)] {
                    //The orbit's center (right, up) scales from the view's: the view is (-right, -up) from it
                    let (center_real, center_imaginary) = (Fixed::parse(real, bits).unwrap().plus_scaled(right * depth.mantissa, depth.exponent), Fixed::parse(imaginary, bits).unwrap().plus_scaled(up * depth.mantissa, depth.exponent));
                    let orbit = SharedOrbit::new(&center_real, &center_imaginary);
                    for approximate in [false, true] {
                        let mut counts = Vec::new();
                        render_layers_around(&m, (&orbit, (-right, -up)), (depth, 0.0), (width, height), 1, &color_of, approximate, &cancel, |colors, _| {
                            counts = colors.iter().map(|color| u32::from(color.red) | u32::from(color.green) << 8 | u32::from(color.blue) << 16).collect();
                        });
                        let differ = counts.iter().zip(&exact).filter(|(a, b)| a != b).count();
                        total = (total.0 + differ, total.1 + exact.len());
                        if differ > 0 {
                            println!("10^{power} near {name}, orbit ({right}, {up}) views off, BLA {approximate}: {differ} of {} pixels differ; {:?} against {:?}", exact.len(),
                                counts.iter().zip(&exact).filter(|(a, b)| a != b).take(4).collect::<Vec<_>>(), distinct.len());
                        }
                    }
                }
                println!("10^{power} near {name}: {} distinct counts, from {} to {}", distinct.len(), distinct.first().unwrap(), distinct.last().unwrap());
            }
        }
        println!("{} of {} pixels differ from exact arithmetic", total.0, total.1);
    }

    /// Times deep frames on the CPU alone against the CPU and the GPU together (`render_layers_on`, issue #263), with and without BLA, at 1920 x 1080 next to
    /// c = i and in the seahorse valley, and checks the images are the same. Run by hand on an otherwise idle GPU with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand on an idle GPU"]
    fn time_deep_frames_on_the_cpu_and_with_the_gpu() {
        let Ok(gpu) = crate::rendering::gpu::GpuIterator::new() else { return };
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let color_of = |position: f64| crate::coloring::TrueColor::new((position * 7.0) as u8, (position * 3.0) as u8, position as u8);
        let (width, height) = (1920, 1080);
        let near_i = |power: i64| (String::from("i"), power, format!("3e{}", power - 1), format!("1.{}2", "0".repeat((-power) as usize)), 20_000_u32);
        for (name, power, real, imaginary, iterations) in [near_i(-100), near_i(-250), near_i(-308), near_i(-500),
            (String::from("seahorse"), -30, String::from("-0.743643887037158704752191506114774"), String::from("0.131825904205311970493132056385139"), 50_000)] {
            let m = MandelbrotSet::new(iterations, 2.0);
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let bits = bits_for_depth(depth, width);
            let orbit = SharedOrbit::new(&Fixed::parse(&real, bits).unwrap(), &Fixed::parse(&imaginary, bits).unwrap());
            orbit.orbit(iterations); //(The orbit once, before the timings)
            for approximate in [false, true] {
                let mut timed = Vec::new();
                for on_the_gpu in [false, true] {
                    let started = std::time::Instant::now();
                    let mut colors = Vec::new();
                    render_layers_on(&m, (&orbit, (0.0, 0.0)), (depth, 0.0), (width, height), 1, &color_of, approximate, on_the_gpu.then_some((&gpu, false)), &cancel, |layer, _| colors = layer.to_vec());
                    timed.push((started.elapsed().as_secs_f64() * 1000.0, colors));
                }
                let same = timed[0].1.iter().zip(&timed[1].1).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue));
                println!("{name} 10^{power}, BLA {approximate}: CPU {:.0} ms, CPU + GPU {:.0} ms ({:.2}x), the same image: {same}", timed[0].0, timed[1].0, timed[0].0 / timed[1].0);
            }
        }
    }

    /// A keyframe of a deep Fast dive (#263): 3840 x 2160 at 1 sample on the CPU, near i and in the seahorse valley, with and without BLA, and the share of
    /// its reference (the orbit's table and the BLA levels, see `Reference::of`) and its colors. Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn time_a_keyframe_and_its_parts() {
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let (width, height) = (3840, 2160);
        let near_i = |power: i64| (String::from("i"), power, format!("3e{}", power - 1), format!("1.{}2", "0".repeat((-power) as usize)), 20_000_u32);
        //(MANDELBROT_KEYFRAME_SEAHORSE: also the seahorse valley, minutes a keyframe)
        let seahorse = std::env::var("MANDELBROT_KEYFRAME_SEAHORSE").is_ok().then(|| (String::from("seahorse"), -30, String::from("-0.743643887037158704752191506114774"), String::from("0.131825904205311970493132056385139"), 50_000));
        for (name, power, real, imaginary, iterations) in [near_i(-20), near_i(-100)].into_iter().chain(seahorse) {
            let mut m = MandelbrotSet::new(iterations, 2.0);
            m.palette = crate::mandelbrot_set::ColorCycle { divide: 4.0, offset: 0.0, stretch: false };
            let color_of = |position: f64| m.interpolated_color(position, |count| m.color(crate::coloring::TrueColor::new_from_bernstein_polynomials, count));
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let bits = bits_for_depth(depth, width);
            let orbit = SharedOrbit::new(&Fixed::parse(&real, bits).unwrap(), &Fixed::parse(&imaginary, bits).unwrap());
            orbit.orbit(iterations);
            for approximate in [false, true] {
                let started = std::time::Instant::now();
                let pixel = 2.5 * depth.mantissa / width as f64;
                let reference = Reference::of(&m, orbit.orbit_and_residuals(iterations), times_two_to(max_offset(pixel, width, height), depth.exponent), approximate);
                let reference_time = started.elapsed().as_secs_f64() * 1000.0;
                std::hint::black_box(&reference);
                let started = std::time::Instant::now();
                render_layers_on(&m, (&orbit, (0.0, 0.0)), (depth, 0.0), (width, height), 1, &color_of, approximate, None, &cancel, |layer, _| { std::hint::black_box(layer); });
                let total = started.elapsed().as_secs_f64() * 1000.0;
                let started = std::time::Instant::now();
                let recolored: Vec<crate::coloring::TrueColor> = (0..width * height).map(|index| color_of(f64::from((index % 977) as u32))).collect();
                std::hint::black_box(&recolored);
                let coloring = started.elapsed().as_secs_f64() * 1000.0;
                println!("{name} 10^{power}, BLA {approximate}: {total:.0} ms a keyframe; its reference {reference_time:.0} ms; coloring 8.3 M positions on one thread {coloring:.0} ms");
            }
        }
    }

    /// Issue #288: the GPU's 32-bit deep loop against its 64-bit one, at 1920 x 1080 next to c = i (10⁻¹⁰⁰, 10⁻³⁰⁸, 10⁻⁵⁰⁰) and in the seahorse valley
    /// (10⁻³⁰), with and without BLA: the times (CPU and GPU sharing the frame), the share of pixels of the same color and the PSNR of the colors; and at
    /// 16 x 12 against exact arithmetic, the share of pixels whose count is exact in each. Run by hand on an idle GPU with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand on an idle GPU"]
    fn measure_deep_views_in_32_bit_floats() {
        let Ok(gpu) = crate::rendering::gpu::GpuIterator::new() else { return };
        println!("the 32-bit loop agrees with the CPU's counts on {:.1}% of the self-test's pixels; usable: {}", 100.0 * gpu32_agrees(&gpu), gpu.has_deep32());
        let cancel = std::sync::atomic::AtomicBool::new(false);
        //(The position in 24 bits, in 1/256ths of an iteration, so the two loops' positions can be compared, not just colors)
        let color_of = |position: f64| {
            let code = (position * 256.0) as u32;
            crate::coloring::TrueColor::new((code & 255) as u8, (code >> 8 & 255) as u8, (code >> 16 & 255) as u8)
        };
        let decode = |color: &crate::coloring::TrueColor| f64::from(u32::from(color.red) | u32::from(color.green) << 8 | u32::from(color.blue) << 16) / 256.0;
        let near_i = |power: i64| (String::from("i"), power, format!("3e{}", power - 1), format!("1.{}2", "0".repeat((-power) as usize)), 20_000_u32);
        for (name, power, real, imaginary, iterations) in [near_i(-100), near_i(-308), near_i(-500),
            (String::from("seahorse"), -30, String::from("-0.743643887037158704752191506114774"), String::from("0.131825904205311970493132056385139"), 50_000)] {
            let m = MandelbrotSet::new(iterations, 2.0);
            let depth = Depth::parse(&format!("1e{power}")).unwrap();
            let (width, height) = (1920, 1080);
            let bits = bits_for_depth(depth, width);
            let orbit = SharedOrbit::new(&Fixed::parse(&real, bits).unwrap(), &Fixed::parse(&imaginary, bits).unwrap());
            orbit.orbit(iterations);
            for approximate in [false, true] {
                let mut runs = Vec::new();
                for single in [false, true] {
                    let started = std::time::Instant::now();
                    let mut colors = Vec::new();
                    render_layers_on(&m, (&orbit, (0.0, 0.0)), (depth, 0.0), (width, height), 1, &color_of, approximate, Some((&gpu, single)), &cancel, |layer, _| colors = layer.to_vec());
                    runs.push((started.elapsed().as_secs_f64() * 1000.0, colors));
                }
                let (f64_colors, f32_colors) = (&runs[0].1, &runs[1].1);
                let differences: Vec<f64> = f64_colors.iter().zip(f32_colors).map(|(a, b)| (decode(a) - decode(b)).abs()).collect();
                let same = differences.iter().filter(|difference| **difference == 0.0).count();
                let (over_one, worst) = (differences.iter().filter(|difference| **difference >= 1.0).count(), differences.iter().copied().fold(0.0, f64::max));
                let mean = differences.iter().sum::<f64>() / differences.len() as f64;
                println!("{name} 10^{power}, BLA {approximate}: f64 {:.0} ms, f32 {:.0} ms ({:.2}x); positions the same (to 1/256) {:.2}%, off by a whole iteration or more {:.3}%, mean difference {mean:.4}, worst {worst:.1}",
                    runs[0].0, runs[1].0, runs[0].0 / runs[1].0, 100.0 * same as f64 / differences.len() as f64, 100.0 * over_one as f64 / differences.len() as f64);
            }
            //Against exact arithmetic, small
            let mut counting = m.clone();
            counting.smooth_coloring = false;
            counting.max_iterations = counting.max_iterations.min(5000);
            let (small_width, small_height) = (16, 12);
            let exact = super::tests::exact_counts(&real, &imaginary, (1.0, power), (small_width, small_height), counting.max_iterations);
            let small_orbit = SharedOrbit::new(&Fixed::parse(&real, bits_for_depth(depth, small_width) + 16).unwrap(), &Fixed::parse(&imaginary, bits_for_depth(depth, small_width) + 16).unwrap());
            let count_color = |position: f64| {
                let count = position as u32;
                crate::coloring::TrueColor::new((count & 255) as u8, (count >> 8 & 255) as u8, (count >> 16) as u8)
            };
            for single in [false, true] {
                let mut counts = Vec::new();
                render_layers_on(&counting, (&small_orbit, (0.0, 0.0)), (depth, 0.0), (small_width, small_height), 1, &count_color, false, Some((&gpu, single)), &cancel, |colors, _| {
                    counts = colors.iter().map(|color| u32::from(color.red) | u32::from(color.green) << 8 | u32::from(color.blue) << 16).collect();
                });
                let right = counts.iter().zip(&exact).filter(|(a, b)| a == b).count();
                println!("{name} 10^{power}, {}: {right} of {} counts exact", if single { "f32" } else { "f64" }, exact.len());
            }
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

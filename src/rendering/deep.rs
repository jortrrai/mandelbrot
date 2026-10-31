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

    /// The nearest 64-bit float
    pub fn to_f64(&self) -> f64 {
        //The top 64 bits as an integer, scaled by the rest: exact enough for a 64-bit result
        let shift = self.value.bits().saturating_sub(64);
        let top = (&self.value >> shift).to_f64().unwrap_or(0.0);
        top * 2f64.powi(i32::try_from(shift).unwrap_or(i32::MAX)) / 2f64.powi(i32::try_from(self.bits).unwrap_or(i32::MAX))
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
    let (mut zx, mut zy) = (Fixed { value: BigInt::zero(), bits: real.bits }, Fixed { value: BigInt::zero(), bits: real.bits });
    let mut orbit = Vec::with_capacity(max_iterations as usize + 1);
    orbit.push((0.0, 0.0));
    for _ in 0..max_iterations {
        let (xx, yy, xy) = (zx.mul(&zx), zy.mul(&zy), zx.mul(&zy));
        (zx, zy) = (xx.sub(&yy).add(real), xy.double().add(imaginary));
        orbit.push((zx.to_f64(), zy.to_f64()));
        if zx.magnitude_above(8) || zy.magnitude_above(8) {
            break;
        }
    }
    orbit
}

/// The smooth position (see `MandelbrotSet::position`) of the point at offset `(dcx, dcy)` from the center of `orbit`, by perturbation with rebasing.
///
/// `center` is the center in 64-bit floats, for the steps after the point escaped; see the module
pub fn position(m: &MandelbrotSet, orbit: &[(f64, f64)], center: (f64, f64), (dcx, dcy): (f64, f64)) -> f64 {
    let radius_squared = m.orbit_radius * m.orbit_radius;
    let (mut zx, mut zy) = (0.0_f64, 0.0_f64); //The difference from the reference orbit
    let mut reference = 0; //The step of the reference orbit that the difference is from
    for step in 0..m.max_iterations {
        let (ox, oy) = orbit[reference];
        //z ← 2·Z·z + z² + c
        (zx, zy) = (2.0 * (ox * zx - oy * zy) + (zx * zx - zy * zy) + dcx, 2.0 * (ox * zy + oy * zx) + 2.0 * zx * zy + dcy);
        reference += 1;
        let (fx, fy) = (orbit[reference].0 + zx, orbit[reference].1 + zy); //The pixel's orbit
        let full = fx * fx + fy * fy;
        if full > radius_squared {
            return m.smooth_position_of(step + 1, (fx, fy), (center.0 + dcx, center.1 + dcy));
        }
        //Rebasing: the pixel's orbit is closer to 0 than its difference, or the reference orbit ends
        if full < zx * zx + zy * zy || reference + 1 >= orbit.len() {
            (zx, zy) = (fx, fy);
            reference = 0;
        }
    }
    f64::from(m.max_iterations)
}

/// The smooth positions of a `width` x `height` view of the center `(real, imaginary)` and `scale`, row by row from the top.
///
/// The center is decimal text, as precise as needed; the view is `2.5 · scale` wide. One sample in the middle of every pixel, on all the render threads
pub fn render_positions(m: &MandelbrotSet, real: &str, imaginary: &str, scale: f64, width: usize, height: usize) -> Option<Vec<f64>> {
    let bits = bits_for(scale, width);
    let (center_real, center_imaginary) = (Fixed::parse(real, bits)?, Fixed::parse(imaginary, bits)?);
    render_fixed(m, &center_real, &center_imaginary, scale, width, height, &std::sync::atomic::AtomicBool::new(false))
}

/// Like `render_positions`, for a center in fixed point (with at least `bits_for(scale, width)` bits); `None` when `cancel` was set before it finished
pub fn render_fixed(m: &MandelbrotSet, center_real: &Fixed, center_imaginary: &Fixed, scale: f64, width: usize, height: usize, cancel: &std::sync::atomic::AtomicBool) -> Option<Vec<f64>> {
    use std::sync::atomic::Ordering;
    let orbit = reference_orbit(center_real, center_imaginary, m.max_iterations);
    let center = (center_real.to_f64(), center_imaginary.to_f64());
    let pixel = 2.5 * scale / width as f64;
    let mut positions = vec![0.0; width * height];
    let threads = super::render_thread_count().max(1);
    let rows_per_thread = height.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, rows) in positions.chunks_mut(rows_per_thread * width).enumerate() {
            let orbit = &orbit;
            scope.spawn(move || {
                for (index, out) in rows.iter_mut().enumerate() {
                    if index % width == 0 && cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let (x, y) = (index % width, chunk * rows_per_thread + index / width);
                    let offset = ((x as f64 + 0.5 - width as f64 / 2.0) * pixel, (height as f64 / 2.0 - y as f64 - 0.5) * pixel);
                    *out = position(m, orbit, center, offset);
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
    fn a_view_of_ten_to_the_minus_forty_has_structure() {
        //Around c = i, a point of the boundary (a Misiurewicz point: its orbit 0, i, i - 1, -i, i - 1, ... is periodic after a step), which has structure at
        //every depth; 10⁻⁴⁰ wide, where the 64-bit loop would give one value for every pixel
        let m = MandelbrotSet::new(5000, 2.0);
        let positions = render_positions(&m, "0.0000000000000000000000000000000000000000003", "1.0000000000000000000000000000000000000000002", 1e-40, 48, 32).unwrap();
        let distinct: std::collections::BTreeSet<u64> = positions.iter().map(|position| (position * 1000.0) as u64).collect();
        assert!(distinct.len() > 20, "the pixels differ: {} different positions", distinct.len());
    }
}

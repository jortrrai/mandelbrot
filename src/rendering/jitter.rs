//! The deterministic jitter of the supersampling samples.
//!
//! Every sample of a pixel is taken at a pseudo-random offset inside the pixel. The offset only depends on the pixel, the number of the sample and the hash below, never on thread scheduling,
//! so identical settings always render identical images.

/// SplitMix64 finalizer, a well-mixed 64-bit hash: <https://prng.di.unimi.it/splitmix64.c>
pub(super) fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic supersampling jitter in [0, 1) x [0, 1) for sample number `sample` of the pixel at (`x`, `y`).
/// Depends only on its arguments, not on thread scheduling, so identical inputs always render identical images.
pub(super) fn sample_jitter(x: usize, y: usize, sample: u8) -> (f64, f64) {
    let h = splitmix64(splitmix64(splitmix64(x as u64) ^ y as u64) ^ u64::from(sample));
    let to_unit = |bits: u64| (bits >> 11) as f64 / (1u64 << 53) as f64; //53 random bits, like rand's f64 sampling
    (to_unit(h), to_unit(splitmix64(h)))
}

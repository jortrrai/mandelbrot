//! Tests of the rendering module: the progressive and adaptive renderers, their regions, the jitter and the moves.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::jitter::{sample_jitter, splitmix64};
use super::*;
use crate::{coloring::TrueColor, complex_plane::ComplexPlane, mandelbrot_set::MandelbrotSet, pixel_buffer::PixelBuffer};
use crate::pixel_buffer::pixel_plane::PixelPlane;

fn render(width: usize, height: usize, supersampling_amount: u8) -> Vec<(u8, u8, u8)> {
    let mut p = PixelBuffer::new(PixelPlane::new(width, height));
    let c = ComplexPlane::new(width, height);
    let m = MandelbrotSet::new(200, 2.0);
    render_complex_plane_into_buffer(&mut p, &c, &m, supersampling_amount, TrueColor::new_from_bernstein_polynomials);
    p.colors.iter().map(|c| (c.red, c.green, c.blue)).collect()
}

#[test]
fn render_thread_count_leaves_one_thread_free_but_is_at_least_one() {
    let threads = render_thread_count();
    assert!(threads >= 1);
    assert_eq!(threads, (num_cpus::get() - 1).max(1));
}

#[test]
fn rendering_is_reproducible() {
    for ssaa in [1, 4] {
        let first = render(64, 48, ssaa);
        for _ in 0..3 {
            assert_eq!(first, render(64, 48, ssaa), "ssaa = {ssaa}");
        }
    }
}

#[test]
fn progressive_rendering_matches_the_normal_render_and_reports_every_pass() {
    let (width, height, supersampling_amount) = (48, 32, 4);
    let expected = render(width, height, supersampling_amount);
    let c = ComplexPlane::new(width, height);
    let m = MandelbrotSet::new(200, 2.0);
    let pixels_done = AtomicUsize::new(0);
    let mut passes = Vec::new();
    let mut last = Vec::new();
    let completed = Render::new(&c, &m, width, height, TrueColor::new_from_bernstein_polynomials).supersampling(supersampling_amount).count_pixels_in(&pixels_done).run( |colors, pass| {
        passes.push(pass);
        last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect();
    });
    assert!(completed);
    assert_eq!(pixels_done.load(Ordering::Relaxed), width * height, "every pixel of the last pass should be counted");
    assert_eq!(passes, [1, 2, 3, 4]);
    assert_eq!(last, expected, "the last pass should equal the normal render");
}

#[test]
fn exposed_region_covers_the_pixels_a_shift_exposes() {
    assert_eq!(ExposedRegion::from_shift(10, 8, 0, 3), Some(ExposedRegion { columns: 0..3, rows: 8..8 }), "content moves right: the left strip is exposed");
    assert_eq!(ExposedRegion::from_shift(10, 8, 0, -3), Some(ExposedRegion { columns: 7..10, rows: 8..8 }), "content moves left: the right strip is exposed");
    assert_eq!(ExposedRegion::from_shift(10, 8, 2, 0), Some(ExposedRegion { columns: 10..10, rows: 0..2 }), "content moves down: the top strip is exposed");
    assert_eq!(ExposedRegion::from_shift(10, 8, -2, 0), Some(ExposedRegion { columns: 10..10, rows: 6..8 }), "content moves up: the bottom strip is exposed");
    assert_eq!(ExposedRegion::from_shift(10, 8, 8, 0), None, "a shift as large as the buffer exposes everything");
    let l_shape = ExposedRegion::from_shift(10, 8, 2, 3).unwrap();
    assert_eq!(l_shape.pixel_count(10, 8), 3 * 8 + 2 * 10 - 3 * 2);
    assert_eq!((0..10).flat_map(|x| (0..8).map(move |y| (x, y))).filter(|(x, y)| l_shape.contains(*x, *y)).count(), l_shape.pixel_count(10, 8));
}

#[test]
fn region_rendering_only_changes_the_region_and_matches_the_normal_render_there() {
    let (width, height, supersampling_amount) = (48, 32, 3);
    let full = render(width, height, supersampling_amount);
    let c = ComplexPlane::new(width, height);
    let m = MandelbrotSet::new(200, 2.0);
    let base = vec![TrueColor::new(1, 2, 3); width * height]; //Stands for an old image
    let region = ExposedRegion::from_shift(width, height, 4, -5).unwrap();
    let pixels_done = AtomicUsize::new(0);
    let mut last = Vec::new();
    let mut passes = 0;
    Render::new(&c, &m, width, height, TrueColor::new_from_bernstein_polynomials).supersampling(supersampling_amount).only(&region, &base).count_pixels_in(&pixels_done).run( |colors, pass| {
        passes = pass;
        last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect();
    });
    assert_eq!(passes, 3);
    assert_eq!(pixels_done.load(Ordering::Relaxed), region.pixel_count(width, height), "only the pixels of the region are rendered");
    for (index, color) in last.iter().enumerate() {
        let expected = if region.contains(index % width, index / width) { full[index] } else { (1, 2, 3) };
        assert_eq!(*color, expected, "pixel {index}");
    }
}

#[test]
fn progressive_rendering_can_be_cancelled() {
    let c = ComplexPlane::new(16, 16);
    let m = MandelbrotSet::new(50, 2.0);
    let cancel = AtomicBool::new(false);
    let mut passes = 0;
    let completed = Render::new(&c, &m, 16, 16, TrueColor::new_from_bernstein_polynomials).supersampling(8).cancel_with(&cancel).run( |_, pass| {
        passes = pass;
        if pass == 2 {
            cancel.store(true, Ordering::Relaxed);
        }
    });
    assert!(!completed);
    assert_eq!(passes, 2, "no pass should be reported after cancelling");
}

#[test]
fn sample_jitter_is_in_unit_square_and_varies() {
    let mut distinct = std::collections::HashSet::new();
    for (x, y, sample) in (0..20).flat_map(|x| (0..20).flat_map(move |y| (0..4).map(move |s| (x, y, s)))) {
        let (jx, jy) = sample_jitter(x, y, sample);
        assert!((0.0..1.0).contains(&jx) && (0.0..1.0).contains(&jy));
        assert_eq!((jx, jy), sample_jitter(x, y, sample));
        distinct.insert(jx.to_bits());
    }
    assert!(distinct.len() > 1500, "jitter should vary per pixel and sample, got {} distinct values", distinct.len());
}

fn adaptive_render(c: &ComplexPlane, m: &MandelbrotSet, w: usize, h: usize, region: &ExposedRegion, base: &[TrueColor], samples: u8, adaptive: Option<&AdaptiveSampling>, cancel: &AtomicBool) -> (Vec<(u8, u8, u8)>, u8, bool) {
    let mut last = Vec::new();
    let mut passes = 0;
    let done = Render::new(c, m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(samples).adaptive(adaptive.copied()).only(region, base).cancel_with(cancel).run( |colors, pass| {
        last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect();
        passes = pass;
    });
    (last, passes, done)
}

fn boundary_plane(w: usize, h: usize) -> (ComplexPlane, MandelbrotSet) {
    let mut c = ComplexPlane::new(w, h);
    c.set_view(&crate::complex_plane::View::new(-0.7498100000000001, -0.020300000000000054, 0.004));
    (c, MandelbrotSet::new(300, 2.0))
}

#[test]
fn adaptive_sampling_is_uniform_where_marked_and_the_first_samples_elsewhere() {
    let (w, h) = (96, 64);
    let (c, m) = boundary_plane(w, h);
    let (region, base, never) = (ExposedRegion::full(w, h), vec![TrueColor::new(0, 0, 0); w * h], AtomicBool::new(false));
    let uniform2 = adaptive_render(&c, &m, w, h, &region, &base, 2, None, &never).0;
    let uniform8 = adaptive_render(&c, &m, w, h, &region, &base, 8, None, &never).0;
    let adaptive = AdaptiveSampling::default();
    let (result, passes, done) = adaptive_render(&c, &m, w, h, &region, &base, 8, Some(&adaptive), &never);
    assert!(done);
    assert_eq!(passes, 8, "every sample is still reported as a pass");
    let (mut refined, mut skipped) = (0, 0);
    for i in 0..w * h {
        if result[i] == uniform8[i] { refined += 1; } else {
            assert_eq!(result[i], uniform2[i], "pixel {i} is neither fully supersampled nor the average of the first samples");
            skipped += 1;
        }
    }
    assert!(refined > 0 && skipped > 0, "a boundary view has both detail and smooth pixels: {refined} refined, {skipped} skipped");
    assert_eq!(result, adaptive_render(&c, &m, w, h, &region, &base, 8, Some(&adaptive), &never).0, "adaptive rendering is deterministic");
}

#[test]
fn adaptive_sampling_with_at_most_two_samples_equals_uniform_and_threshold_zero_refines_everything_that_varies() {
    let (w, h) = (48, 32);
    let (c, m) = boundary_plane(w, h);
    let (region, base, never) = (ExposedRegion::full(w, h), vec![TrueColor::new(0, 0, 0); w * h], AtomicBool::new(false));
    for samples in [1, 2] {
        assert_eq!(adaptive_render(&c, &m, w, h, &region, &base, samples, Some(&AdaptiveSampling::default()), &never).0, adaptive_render(&c, &m, w, h, &region, &base, samples, None, &never).0);
    }
    let everything = AdaptiveSampling { threshold: 0 };
    let a = adaptive_render(&c, &m, w, h, &region, &base, 5, Some(&everything), &never).0;
    let u = adaptive_render(&c, &m, w, h, &region, &base, 5, None, &never).0;
    let differing = a.iter().zip(&u).filter(|(a, b)| a != b).count();
    assert!(differing * 20 < w * h, "with threshold 0 almost every pixel is refined and equals the uniform render: {differing} differ");
}

#[test]
fn adaptive_sampling_marks_thin_features_and_their_neighbors() {
    let (w, h) = (9, 9);
    let region = ExposedRegion::full(w, h);
    let black = TrueColor::new(0, 0, 0);
    let mut colors = vec![black; w * h];
    colors[4 * w + 4] = TrueColor::new(255, 255, 255); //One bright pixel
    let first = colors.clone();
    let mask = adaptive_mask(w, h, &region, &first, &colors, 48);
    for y in 0..h {
        for x in 0..w {
            let near = (3..=5).contains(&x) && (3..=5).contains(&y);
            let near_ring = (2..=6).contains(&x) && (2..=6).contains(&y);
            assert_eq!(mask[y * w + x], near_ring, "({x}, {y}); the pixel, its neighbors (they differ from it) and their neighbors are marked, near = {near}");
        }
    }
    //A pixel whose two samples differ is marked even if all the averages are the same
    let mut first = vec![black; w * h];
    first[0] = TrueColor::new(200, 200, 200);
    let flat = vec![black; w * h];
    assert!(adaptive_mask(w, h, &region, &first, &flat, 48)[0]);
    assert!(!adaptive_mask(w, h, &region, &flat, &flat, 48).iter().any(|m| *m));
}

#[test]
fn adaptive_sampling_respects_the_region_and_can_be_cancelled() {
    let (w, h) = (48, 32);
    let (c, m) = boundary_plane(w, h);
    let strip = ExposedRegion::from_shift(w, h, 8, 0).unwrap();
    let base = vec![TrueColor::new(7, 8, 9); w * h];
    let (result, _, done) = adaptive_render(&c, &m, w, h, &strip, &base, 6, Some(&AdaptiveSampling::default()), &AtomicBool::new(false));
    assert!(done);
    for y in 0..h { for x in 0..w { if !strip.contains(x, y) { assert_eq!(result[y * w + x], (7, 8, 9)); } } }
    let (_, passes, done) = adaptive_render(&c, &m, w, h, &strip, &base, 6, Some(&AdaptiveSampling::default()), &AtomicBool::new(true));
    assert!(!done && passes == 0);
}

/// The colors of the progressive render must be those of the scalar functions for the same sample points, for both coloring modes (the width is no multiple of the lanes, so the scalar remainder is used too)
#[test]
fn rendering_colors_every_pixel_like_the_scalar_functions_with_and_without_smooth_coloring() {
    let (w, h) = (30, 14);
    let (c, mut m) = boundary_plane(w, h);
    for smooth in [false, true] {
        m.smooth_coloring = smooth;
        let (region, base, never) = (ExposedRegion::full(w, h), vec![TrueColor::new(0, 0, 0); w * h], AtomicBool::new(false));
        let (rendered, _, _) = adaptive_render(&c, &m, w, h, &region, &base, 1, None, &never);
        for y in 0..h {
            for x in 0..w {
                let (jx, jy) = sample_jitter(x, y, 0);
                let point = c.complex_from_pixel_plane(x as f64 + jx, y as f64 + jy);
                let expected = if smooth {
                    m.interpolated_color(m.iterate_smooth(&point), |i| m.color(TrueColor::new_from_bernstein_polynomials, i))
                } else {
                    m.color(TrueColor::new_from_bernstein_polynomials, m.iterate(&point))
                };
                assert_eq!(rendered[y * w + x], (expected.red, expected.green, expected.blue), "smooth = {smooth}, pixel ({x}, {y})");
            }
        }
    }
}

#[test]
fn smooth_rendering_is_deterministic_and_progressive_passes_are_all_reported() {
    let (w, h) = (40, 24);
    let (c, m) = boundary_plane(w, h);
    let (region, base, never) = (ExposedRegion::full(w, h), vec![TrueColor::new(0, 0, 0); w * h], AtomicBool::new(false));
    let (a, passes, done) = adaptive_render(&c, &m, w, h, &region, &base, 3, None, &never);
    assert!(done && passes == 3);
    assert_eq!(a, adaptive_render(&c, &m, w, h, &region, &base, 3, None, &never).0);
    let (_, passes, done) = adaptive_render(&c, &m, w, h, &region, &base, 3, None, &AtomicBool::new(true));
    assert!(!done && passes == 0, "cancelling still works");
}

#[test]
fn centering_and_moving_by_pixels_agree_with_the_shift_of_the_buffer() {
    let mut plane = ComplexPlane::new(120, 80);
    let start = plane.center();
    //Moving up by 10 rows and right by 7 columns moves the center up (a larger imaginary part) and right
    translate_complex_plane_by_pixels(&mut plane, 10, 0);
    assert!((plane.center().y - (start.y + 10.0 * plane.increment_y)).abs() < 1e-12 && (plane.center().x - start.x).abs() < 1e-12);
    translate_complex_plane_by_pixels(&mut plane, 0, 7);
    assert!((plane.center().x - (start.x + 7.0 * plane.increment_x)).abs() < 1e-12);
    translate_complex_plane_by_pixels(&mut plane, -10, 0);
    translate_complex_plane_by_pixels(&mut plane, 0, -7);
    assert!((plane.center().x - start.x).abs() < 1e-12 && (plane.center().y - start.y).abs() < 1e-12, "the moves are undone");

    //Centering on the point under pixel (90, 25) returns the shift of the buffer: the content moves left and down. The pixel counts are truncated, not rounded, so a distance of
    //30 pixels that is 29.999... in floating point shifts by 29 (an existing imprecision of one pixel, which is kept because the moves of the window depend on it)
    let mut plane = ComplexPlane::new(120, 80);
    let target = plane.complex_from_pixel_plane(90.0, 25.0);
    let shift = center_complex_plane_on(&mut plane, &target);
    assert_eq!(shift, (14, -29), "(rows_up, columns_right) of the content");
    assert!((plane.center().x - target.x).abs() < 1e-12 && (plane.center().y - target.y).abs() < 1e-12);
}

#[test]
fn the_hash_behind_the_jitter_is_splitmix64() {
    //The reference outputs of SplitMix64 (https://prng.di.unimi.it/splitmix64.c) for the seed 0: the first three values
    assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    assert_eq!(splitmix64(0x9E37_79B9_7F4A_7C15), 0x6E78_9E6A_A1B9_65F4);
    assert_eq!(splitmix64(0x3C6E_F372_FE94_F82A), 0x06C4_5D18_8009_454F);
}

#[test]
fn the_jitter_of_a_sample_never_changes() {
    //Pinned bit for bit: only integer operations and an exact conversion of 53 bits, so the same on every platform. A change would change every supersampled image.
    let expected = [
        ((0, 0, 0), (0x3fc1_c13a_de1c_7e5c, 0x3fc0_983a_4555_6400)),
        ((1, 0, 0), (0x3fe6_3140_5e8d_b1b0, 0x3fec_5032_bbbb_3dc9)),
        ((0, 1, 0), (0x3fd1_396e_6040_319e, 0x3fea_3804_e0d0_f309)),
        ((0, 0, 1), (0x3fc7_9953_c24b_633c, 0x3feb_3dde_fe0e_7a6f)),
        ((1199, 799, 63), (0x3fef_65b3_3ab0_ce5b, 0x3fd0_f16b_7245_b4ae)),
    ];
    for ((x, y, sample), (jitter_x, jitter_y)) in expected {
        let (actual_x, actual_y) = sample_jitter(x, y, sample);
        assert_eq!((actual_x.to_bits(), actual_y.to_bits()), (jitter_x, jitter_y), "pixel ({x}, {y}), sample {sample}: {actual_x:?}, {actual_y:?}");
    }
}

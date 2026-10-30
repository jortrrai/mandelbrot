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

#[allow(clippy::too_many_arguments)] // A test helper: the arguments are the settings of one render, and every test passes them at once
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
fn moving_by_pixels_agrees_with_the_shift_of_the_buffer_at_any_distance() {
    let mut plane = ComplexPlane::new(120, 80);
    let start = plane.center();
    //Moving up by 10 rows and right by 7 columns moves the center up (a larger imaginary part) and right
    move_complex_plane_by_pixels(&mut plane, 10, 0);
    assert!((plane.center().y - (start.y + 10.0 * plane.increment_y)).abs() < 1e-12 && (plane.center().x - start.x).abs() < 1e-12);
    move_complex_plane_by_pixels(&mut plane, 0, 7);
    assert!((plane.center().x - (start.x + 7.0 * plane.increment_x)).abs() < 1e-12);
    move_complex_plane_by_pixels(&mut plane, -10, 0);
    move_complex_plane_by_pixels(&mut plane, 0, -7);
    assert!((plane.center().x - start.x).abs() < 1e-12 && (plane.center().y - start.y).abs() < 1e-12, "the moves are undone");

    //Any distance: the old helper took it modulo 256 pixels, so 300 moved by 44
    for (rows_up, columns_right) in [(300, -1000), (-256, 256), (3, 255)] {
        let start = ComplexPlane::new(1200, 800);
        let mut moved = start.clone();
        move_complex_plane_by_pixels(&mut moved, rows_up, columns_right);
        assert_eq!(pixel_shift(&start, &moved), Some((i128::from(rows_up), -i128::from(columns_right))), "the content moves the other way");
    }
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

/// Every image that a render reported, with its pass number
type Reports = Vec<(u8, Vec<(u8, u8, u8)>)>;

/// Renders with or without previews; returns every reported image with its pass number, and whether the render completed
#[allow(clippy::too_many_arguments)] // A test helper: the arguments are the settings of one render, like `adaptive_render`
fn preview_render(c: &ComplexPlane, m: &MandelbrotSet, w: usize, h: usize, region: &ExposedRegion, base: &[TrueColor], samples: u8, preview: Option<std::time::Duration>, cancel: &AtomicBool) -> (Reports, bool) {
    let mut reports = Vec::new();
    let adaptive = Some(AdaptiveSampling::default());
    let done = Render::new(c, m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(samples).adaptive(adaptive).only(region, base).preview(preview).cancel_with(cancel).run(|colors, pass| reports.push((pass, colors.iter().map(|c| (c.red, c.green, c.blue)).collect())));
    (reports, done)
}

#[test]
fn previews_do_not_change_the_layers_and_come_coarse_to_fine_before_them() {
    let (w, h) = (61, 37); //Not a multiple of the block sizes
    let (c, m) = boundary_plane(w, h);
    let never = AtomicBool::new(false);
    let base = vec![TrueColor::new(9, 9, 9); w * h];
    for region in [ExposedRegion::full(w, h), ExposedRegion::from_shift(w, h, 5, -11).unwrap()] {
        let (plain, done) = preview_render(&c, &m, w, h, &region, &base, 4, None, &never);
        assert!(done);
        //A zero threshold reports every level
        let (with_previews, done) = preview_render(&c, &m, w, h, &region, &base, 4, Some(std::time::Duration::ZERO), &never);
        assert!(done);
        let passes: Vec<u8> = with_previews.iter().map(|(pass, _)| *pass).collect();
        assert_eq!(passes, [0, 0, 0, 1, 2, 3, 4]);
        assert_eq!(with_previews[3..], plain[..], "the layers are identical with and without previews");

        //In a preview, a block shows the first sample of its top-left pixel: exactly the color of that pixel in the first layer. Pixels outside the region keep the base
        let first_layer = &plain[0].1;
        for (level, block) in [8, 4, 2].into_iter().enumerate() {
            let preview = &with_previews[level].1;
            for y in 0..h {
                for x in 0..w {
                    let (ox, oy) = (x - x % block, y - y % block);
                    let expected = if !region.contains(x, y) { (9, 9, 9) } else if region.contains(ox, oy) { first_layer[oy * w + ox] } else { preview[y * w + x] };
                    assert_eq!(preview[y * w + x], expected, "block {block}, pixel ({x}, {y})");
                }
            }
        }
    }
}

#[test]
fn a_fast_render_skips_the_previews_and_a_cancelled_preview_cancels_the_render() {
    let (w, h) = (64, 48);
    let (c, m) = boundary_plane(w, h);
    let base = vec![TrueColor::new(0, 0, 0); w * h];
    let region = ExposedRegion::full(w, h);
    //A tiny image is much faster than an hour: no previews, the same layers
    let (reports, done) = preview_render(&c, &m, w, h, &region, &base, 2, Some(std::time::Duration::from_secs(3600)), &AtomicBool::new(false));
    assert!(done);
    assert_eq!(reports.iter().map(|(pass, _)| *pass).collect::<Vec<u8>>(), [1, 2]);
    assert_eq!(reports, preview_render(&c, &m, w, h, &region, &base, 2, None, &AtomicBool::new(false)).0);
    let (reports, done) = preview_render(&c, &m, w, h, &region, &base, 2, Some(std::time::Duration::ZERO), &AtomicBool::new(true));
    assert!(!done && reports.is_empty());
}

/// Renders `samples` samples of plane `c` into `state`; cancels after `cancel_after_pass` passes, or in the middle of a pass once `cancel_after_pixels` pixels of it are done.
/// Returns the colors of the last reported pass and whether the render completed.
fn resumed(state: &mut SampleState, c: &ComplexPlane, m: &MandelbrotSet, samples: u8, adaptive: Option<AdaptiveSampling>, cancel_after_pass: Option<u8>, cancel_after_pixels: Option<usize>) -> (Vec<(u8, u8, u8)>, bool) {
    let (w, h) = (state_width(c), state_height(c));
    let cancel = AtomicBool::new(false);
    let done = AtomicUsize::new(0);
    let mut last = Vec::new();
    let completed = std::thread::scope(|scope| {
        if let Some(pixels) = cancel_after_pixels {
            //Watches the progress of the pass and cancels in the middle of it
            let (cancel, done) = (&cancel, &done);
            scope.spawn(move || while !cancel.load(Ordering::Relaxed) {
                if done.load(Ordering::Relaxed) >= pixels {
                    cancel.store(true, Ordering::Relaxed);
                }
                std::hint::spin_loop();
            });
        }
        let completed = Render::new(c, m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(samples).adaptive(adaptive).resume(state).cancel_with(&cancel).count_pixels_in(&done).run(|colors, pass| {
            last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect();
            if cancel_after_pass == Some(pass) {
                cancel.store(true, Ordering::Relaxed);
            }
        });
        cancel.store(true, Ordering::Relaxed); //Stops the watcher
        completed
    });
    (last, completed)
}

fn state_width(c: &ComplexPlane) -> usize {
    (c.length_x / c.increment_x).round() as usize
}

fn state_height(c: &ComplexPlane) -> usize {
    (c.length_y / c.increment_y).round() as usize
}

#[test]
fn a_resumed_render_from_nothing_is_the_plain_render() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    for adaptive in [None, Some(AdaptiveSampling::default())] {
        let plain = adaptive_render(&c, &m, w, h, &ExposedRegion::full(w, h), &vec![TrueColor::new(0, 0, 0); w * h], 5, adaptive.as_ref(), &AtomicBool::new(false)).0;
        let (colors, completed) = resumed(&mut SampleState::new(), &c, &m, 5, adaptive, None, None);
        assert!(completed);
        assert_eq!(colors, plain, "adaptive: {adaptive:?}");
    }
}

#[test]
fn an_interrupted_render_continues_where_it_stopped_and_gives_the_same_image() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    for adaptive in [None, Some(AdaptiveSampling::default())] {
        let (expected, _) = resumed(&mut SampleState::new(), &c, &m, 6, adaptive, None, None);
        //Between passes, and in the middle of a pass (the watcher stops it after some pixels; where exactly does not matter)
        for (after_pass, after_pixels) in [(Some(1), None), (Some(3), None), (None, Some(200)), (None, Some(1500))] {
            let mut state = SampleState::new();
            let (_, completed) = resumed(&mut state, &c, &m, 6, adaptive, after_pass, after_pixels);
            assert!(!completed);
            let partial: usize = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| usize::from(state.samples(x, y))).sum();
            assert!(partial > 0, "the interrupted render kept its samples");
            let (colors, completed) = resumed(&mut state, &c, &m, 6, adaptive, None, None);
            assert!(completed);
            assert_eq!(colors, expected, "adaptive {adaptive:?}, interrupted after pass {after_pass:?} / {after_pixels:?} pixels");
        }
    }
}

/// The colors after rendering `c` into a new state, interrupted after `interrupted_after` passes (or complete), then moving the view `rows_up` and `columns_right`
/// and continuing to `samples` samples; and the moved plane with the state
#[allow(clippy::too_many_arguments)] // A test helper: the settings of the two renders
fn moved_render(c: &ComplexPlane, m: &MandelbrotSet, samples: u8, adaptive: Option<AdaptiveSampling>, interrupted_after: Option<u8>, rows_up: i16, columns_right: i16) -> (Vec<(u8, u8, u8)>, ComplexPlane, SampleState) {
    let mut state = SampleState::new();
    let _ = resumed(&mut state, c, m, samples, adaptive, interrupted_after, None);
    let mut moved = c.clone();
    move_complex_plane_by_pixels(&mut moved, i64::from(rows_up), i64::from(columns_right));
    let (colors, completed) = resumed(&mut state, &moved, m, samples, adaptive, None, None);
    assert!(completed);
    (colors, moved, state)
}

#[test]
fn a_moved_view_keeps_the_samples_of_the_pixels_that_stay_visible() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    let (complete, _) = resumed(&mut SampleState::new(), &c, &m, 4, None, None, None);
    for interrupted_after in [None, Some(1), Some(2)] {
        let mut state = SampleState::new();
        let _ = resumed(&mut state, &c, &m, 4, None, interrupted_after, None);
        let before = state.samples(10, 10);
        //Move the view 5 columns to the right and 3 rows up: the content moves 5 columns left and 3 rows down
        let mut moved = c.clone();
        move_complex_plane_by_pixels(&mut moved, 3, 5);
        //A render that is cancelled before it starts only shifts the state: the pixels that stay visible keep their samples, the exposed ones have none
        let cancelled = AtomicBool::new(true);
        let _ = Render::new(&moved, &m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(4).resume(&mut state).cancel_with(&cancelled).run(|_, _| {});
        assert_eq!(state.samples(10 - 5, 10 + 3), before);
        let empty = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).filter(|(x, y)| state.samples(*x, *y) == 0).count();
        assert_eq!(empty, 5 * h + 3 * w - 5 * 3, "only the exposed strips are empty");

        //Continuing gives exactly the original complete image where the view overlaps, and exactly what a move after a complete render gives everywhere
        let (colors, completed) = resumed(&mut state, &moved, &m, 4, None, None, None);
        assert!(completed);
        for y in 3..h {
            for x in 0..w - 5 {
                assert_eq!(colors[y * w + x], complete[(y - 3) * w + x + 5], "({x}, {y}), interrupted after {interrupted_after:?}");
            }
        }
        assert_eq!(colors, moved_render(&c, &m, 4, None, None, 3, 5).0, "interrupted after {interrupted_after:?}");
    }
}

#[test]
fn with_adaptive_supersampling_an_interrupted_and_moved_render_matches_away_from_the_edges() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    let adaptive = Some(AdaptiveSampling::default());
    let (expected, _, expected_state) = moved_render(&c, &m, 8, adaptive, None, -4, -6);
    let (colors, _, state) = moved_render(&c, &m, 8, adaptive, Some(4), -4, -6);
    for y in 0..h {
        for x in 0..w {
            //The mask of a pixel at the edge can depend on a neighbor that moved out of the view (or was not there before the move); elsewhere the image is the same
            if x >= 2 && y >= 2 && x + 2 < w && y + 2 < h {
                assert_eq!(colors[y * w + x], expected[y * w + x], "({x}, {y})");
                assert_eq!(state.samples(x, y), expected_state.samples(x, y), "({x}, {y})");
            }
        }
    }
}

#[test]
fn other_settings_or_a_zoom_start_over() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    let mut state = SampleState::new();
    let _ = resumed(&mut state, &c, &m, 3, None, None, None);
    let other_limit = MandelbrotSet::new(301, 2.0);
    let (colors, _) = resumed(&mut state, &c, &other_limit, 3, None, None, None);
    assert_eq!(colors, resumed(&mut SampleState::new(), &c, &other_limit, 3, None, None, None).0, "another limit renders everything again");
    let mut zoomed = c.clone();
    zoomed.scale(0.5);
    let (colors, _) = resumed(&mut state, &zoomed, &other_limit, 3, None, None, None);
    assert_eq!(colors, resumed(&mut SampleState::new(), &zoomed, &other_limit, 3, None, None, None).0, "so does a zoom");
}

#[test]
fn any_sequence_of_moves_and_interruptions_ends_in_the_complete_image() {
    //Random sequences of moves (by up to 20 pixels each way) and interruptions (after a random pass, in the middle of a pass after a random amount of pixels, or none),
    //ended by one render that completes. The result is compared with a render of the last view from nothing, with the same placement of the samples:
    //every pixel has at least its samples, and a pixel with the same amount of samples has exactly the same color
    let (w, h) = (48, 32);
    let (c, m) = boundary_plane(w, h);
    let mut random = 0x5EED_u64;
    let mut next = |below: u64| {
        random = splitmix64(random);
        random % below
    };
    for adaptive in [None, Some(AdaptiveSampling::default())] {
        for sequence in 0..12 {
            let mut state = SampleState::new();
            let mut plane = c.clone();
            for _ in 0..=next(5) {
                let (after_pass, after_pixels) = match next(3) {
                    0 => (Some(1 + next(5) as u8), None),
                    1 => (None, Some(1 + next(1500) as usize)),
                    _ => (None, None),
                };
                let _ = resumed(&mut state, &plane, &m, 6, adaptive, after_pass, after_pixels);
                move_complex_plane_by_pixels(&mut plane, next(41).cast_signed() - 20, next(41).cast_signed() - 20);
            }
            let (colors, completed) = resumed(&mut state, &plane, &m, 6, adaptive, None, None);
            assert!(completed);
            let mut fresh = state.emptied();
            let (expected, _) = resumed(&mut fresh, &plane, &m, 6, adaptive, None, None);
            let mut same = 0;
            for y in 0..h {
                for x in 0..w {
                    let (has, needs) = (state.samples(x, y), fresh.samples(x, y));
                    assert!(has >= needs, "sequence {sequence}, adaptive {adaptive:?}: ({x}, {y}) has {has} of its {needs} samples");
                    if has == needs {
                        assert_eq!(colors[y * w + x], expected[y * w + x], "sequence {sequence}, adaptive {adaptive:?}, ({x}, {y})");
                        same += 1;
                    }
                }
            }
            if adaptive.is_none() {
                assert_eq!(same, w * h, "without adaptive supersampling every pixel has exactly its samples");
            }
        }
    }
}

#[test]
fn a_preview_of_the_strip_a_move_exposed_has_no_gaps() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    let mut state = SampleState::new();
    let _ = resumed(&mut state, &c, &m, 2, None, None, None);
    //Moves that expose strips at every edge, some not aligned to the 8 x 8 blocks of the coarsest preview
    for (rows_up, columns_right) in [(0, 5), (0, -5), (3, 0), (-3, 0), (2, -7)] {
        let mut moved = state.plane().unwrap().clone();
        move_complex_plane_by_pixels(&mut moved, i64::from(rows_up), i64::from(columns_right));
        let mut previews = Vec::new();
        let completed = Render::new(&moved, &m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(2).resume(&mut state).preview(Some(std::time::Duration::ZERO)).run(|colors, pass| {
            if pass == 0 {
                previews.push(colors.to_vec());
            }
        });
        assert!(completed && !previews.is_empty(), "{rows_up}, {columns_right}");
        //Every exposed pixel of a coarse preview shows the first sample of a pixel of its block, never the black of a gap (the pixels that kept their samples show their own colors)
        let exposed = ExposedRegion::from_shift(w, h, i128::from(rows_up), -i128::from(columns_right)).unwrap();
        for (level, preview) in previews.iter().enumerate() {
            let block = PREVIEW_BLOCKS[level];
            let firsts: Vec<(u8, u8, u8)> = (0..w * h).map(|index| (state.first[index].red, state.first[index].green, state.first[index].blue)).collect();
            for y in 0..h {
                for x in (0..w).filter(|x| exposed.contains(*x, y)) {
                    let color = (preview[y * w + x].red, preview[y * w + x].green, preview[y * w + x].blue);
                    let (left, top) = (x - x % block, y - y % block);
                    let in_block = (top..(top + block).min(h)).any(|by| (left..(left + block).min(w)).any(|bx| firsts[by * w + bx] == color));
                    assert!(in_block, "move ({rows_up}, {columns_right}), block {block}: ({x}, {y}) shows {color:?}, not a first sample of its block");
                }
            }
        }
    }
}

#[test]
fn a_color_table_built_on_all_threads_is_the_table_of_one_thread() {
    for coloring in [TrueColor::new_from_bernstein_polynomials as fn(u32, u32) -> TrueColor, TrueColor::new_from_hsv_colors] {
        for (limit, palette) in [(100, None), (40_000, Some(2043)), (65_376, Some(2043)), (200_000, None)] {
            let mut m = MandelbrotSet::new(limit, 2.0);
            m.palette_scale = palette;
            let table = super::progressive::color_table(&m, coloring);
            assert_eq!(table.len(), limit as usize + 1);
            assert!(table.iter().enumerate().all(|(iterations, color)| {
                let expected = m.color(coloring, iterations as u32);
                (color.red, color.green, color.blue) == (expected.red, expected.green, expected.blue)
            }), "limit {limit}");
        }
    }
}

#[test]
fn a_render_with_the_gpu_is_bit_for_bit_the_render_without_it() {
    //Runs where there is a GPU with f64 shaders. The GPU computes every sample (no CPU threads), so the whole render by the GPU is compared with the CPU's,
    //and once with both together (the share of the GPU then depends on the timing of the threads, so it is not checked)
    let gpu = match super::gpu::GpuIterator::new() {
        Ok(gpu) => gpu,
        Err(reason) => {
            println!("No GPU to test: {reason}");
            return;
        }
    };
    let joined = Some((&gpu, std::time::Duration::ZERO));
    let gpu_samples = AtomicUsize::new(0);
    let (w, h) = (160, 120);
    let (c, m) = boundary_plane(w, h);
    let render = |gpu: Option<(&super::gpu::GpuIterator, std::time::Duration)>, only_gpu: bool, adaptive: Option<AdaptiveSampling>, state: Option<&mut SampleState>, plane: &ComplexPlane| {
        let mut last = Vec::new();
        let render = Render::new(plane, &m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(6).adaptive(adaptive).gpu(gpu).count_gpu_samples_in(&gpu_samples);
        let render = if only_gpu { render.on_the_gpu_only() } else { render };
        let render = match state { Some(state) => render.resume(state), None => render };
        let completed = render.run(|colors, _| last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect::<Vec<_>>());
        assert!(completed);
        last
    };
    for adaptive in [None, Some(AdaptiveSampling::default())] {
        let cpu = render(None, false, adaptive, None, &c);
        let before = gpu_samples.load(Ordering::Relaxed);
        assert_eq!(render(joined, true, adaptive, None, &c), cpu, "a render from nothing by the GPU, adaptive {adaptive:?}");
        assert!(gpu_samples.load(Ordering::Relaxed) - before >= w * h * 2, "the GPU computed every sample (at least the 2 uniform ones of every pixel)");
        assert_eq!(render(joined, false, adaptive, None, &c), cpu, "with the CPU threads, adaptive {adaptive:?}");
        //A render that continues a state after a move
        let (mut with, mut without) = (SampleState::new(), SampleState::new());
        let _ = render(joined, true, adaptive, Some(&mut with), &c);
        let _ = render(None, false, adaptive, Some(&mut without), &c);
        let mut moved = c.clone();
        move_complex_plane_by_pixels(&mut moved, 7, -11);
        assert_eq!(render(joined, true, adaptive, Some(&mut with), &moved), render(None, false, adaptive, Some(&mut without), &moved), "a moved view, adaptive {adaptive:?}");
        assert!((0..h).all(|y| (0..w).all(|x| with.samples(x, y) == without.samples(x, y))), "the same samples");
    }
}

#[test]
fn previews_computed_with_the_gpu_are_bit_for_bit_the_previews_of_the_cpu() {
    let gpu = match super::gpu::GpuIterator::new() {
        Ok(gpu) => gpu,
        Err(reason) => {
            println!("No GPU to test: {reason}");
            return;
        }
    };
    let (w, h) = (160, 120);
    let (c, m) = boundary_plane(w, h);
    let gpu_samples = AtomicUsize::new(0);
    //Every frame the render reports, previews (pass 0) included; a preview threshold of zero reports every level
    let frames = |gpu: Option<(&super::gpu::GpuIterator, std::time::Duration)>, only_gpu: bool| {
        let mut frames = Vec::new();
        let render = Render::new(&c, &m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(2).preview(Some(std::time::Duration::ZERO)).gpu(gpu).count_gpu_samples_in(&gpu_samples);
        let render = if only_gpu { render.on_the_gpu_only() } else { render };
        assert!(render.run(|colors, pass| frames.push((pass, colors.iter().map(|c| (c.red, c.green, c.blue)).collect::<Vec<_>>()))));
        frames
    };
    let cpu = frames(None, false);
    assert!(cpu.iter().filter(|(pass, _)| *pass == 0).count() >= 2, "the previews are reported");
    let before = gpu_samples.load(Ordering::Relaxed);
    //Without CPU threads every level is sampled like a pass, by the GPU (with them only the levels with many samples are)
    assert!(frames(Some((&gpu, std::time::Duration::ZERO)), true) == cpu, "every preview and layer of the GPU is the CPU's");
    assert!(gpu_samples.load(Ordering::Relaxed) > before, "the GPU computed the samples");
}

#[test]
fn a_render_with_the_fast_gpu_keeps_almost_every_pixel_and_deep_views_exact() {
    let gpu = match super::gpu::GpuIterator::new() {
        Ok(gpu) => gpu,
        Err(reason) => {
            println!("No GPU to test: {reason}");
            return;
        }
    };
    let (w, h) = (160, 120);
    let render = |c: &ComplexPlane, m: &MandelbrotSet, fast: Option<bool>| {
        let mut last = Vec::new();
        let render = Render::new(c, m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(2);
        let render = match fast { Some(fast) => render.gpu(Some((&gpu, std::time::Duration::ZERO))).fast_gpu(fast).on_the_gpu_only(), None => render };
        assert!(render.run(|colors, _| last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect::<Vec<_>>()));
        last
    };
    //A boundary view: the fast loop gives almost every pixel its exact color
    let (c, m) = boundary_plane(w, h);
    let (exact, fast) = (render(&c, &m, None), render(&c, &m, Some(true)));
    let differ = exact.iter().zip(&fast).filter(|(a, b)| a != b).count();
    assert!(differ * 100 < w * h * 2, "{differ} of {} pixels differ", w * h);
    //Too deep for its precision (`FAST_GPU_STEPS`): the exact loop, bit for bit
    let mut deep = ComplexPlane::new(w, h);
    deep.set_view(&crate::complex_plane::View::new(-0.743_643_887_037_151, 0.131_825_904_205_33, 1e-11));
    assert!(deep.precision_steps() < super::progressive::FAST_GPU_STEPS);
    let m = MandelbrotSet::new(3000, 2.0);
    assert_eq!(render(&deep, &m, Some(true)), render(&deep, &m, None));
}

#[test]
fn an_image_saved_with_the_gpu_has_the_pixels_of_one_saved_without_it() {
    let gpu = match super::gpu::GpuIterator::new() {
        Ok(gpu) => gpu,
        Err(reason) => {
            println!("No GPU to test: {reason}");
            return;
        }
    };
    let (c, m) = boundary_plane(200, 150);
    let (mut without, mut with) = (PixelBuffer::new(PixelPlane::new(200, 150)), PixelBuffer::new(PixelPlane::new(200, 150)));
    render_complex_plane_into_buffer(&mut without, &c, &m, 3, TrueColor::new_from_bernstein_polynomials);
    render_complex_plane_into_buffer_with_gpu(&mut with, &c, &m, 3, TrueColor::new_from_bernstein_polynomials, Some(&gpu));
    assert_eq!(with.pixels, without.pixels);
}

#[test]
fn a_render_of_only_colors_is_the_first_layer_and_leaves_the_samples_of_its_state_alone() {
    let (w, h) = (61, 37);
    let (c, m) = boundary_plane(w, h);
    let colors_only = |state: Option<&mut SampleState>, gpu: Option<&super::gpu::GpuIterator>| {
        let mut reports = Vec::new();
        let render = Render::new(&c, &m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(4).colors_only();
        let render = match gpu { Some(gpu) => render.gpu(Some((gpu, std::time::Duration::ZERO))).on_the_gpu_only(), None => render };
        let render = match state { Some(state) => render.resume(state), None => render };
        assert!(render.run(|colors, pass| reports.push((pass, colors.iter().map(|c| (c.red, c.green, c.blue)).collect::<Vec<_>>()))));
        reports
    };
    let (first_layer, _) = resumed(&mut SampleState::new(), &c, &m, 1, None, None, None);
    assert_eq!(colors_only(None, None), [(1, first_layer.clone())], "one pass: the first layer, whatever the supersampling");
    //A state with 3 samples of every pixel keeps them, and a render that resumes it has nothing left to do
    let mut state = SampleState::new();
    let (three, _) = resumed(&mut state, &c, &m, 3, None, None, None);
    assert_eq!(colors_only(Some(&mut state), None), [(1, first_layer.clone())]);
    assert!((0..h).all(|y| (0..w).all(|x| state.samples(x, y) == 3)), "the samples are the state's");
    assert_eq!(resumed(&mut state, &c, &m, 3, None, None, None).0, three);
    //By the GPU (the exact loop): the same colors
    if let Ok(gpu) = super::gpu::GpuIterator::new() {
        assert_eq!(colors_only(None, Some(&gpu)), [(1, first_layer)]);
    }
}

/// The division of the running average by a multiplication is exactly the integer division for every sum of up to 65 samples
#[test]
fn the_divider_divides_exactly() {
    for divisor in 1..=65_u32 {
        let divide = super::progressive::divider(divisor);
        for sum in 0..=255 * divisor {
            assert_eq!(divide(sum), sum / divisor, "{sum} / {divisor}");
        }
    }
}

#[test]
fn the_f32_loop_keeps_most_pixels_of_a_shallow_view_and_is_not_used_deeper() {
    let Ok(gpu) = super::gpu::GpuIterator::new() else { return };
    let (w, h) = (480, 320);
    let render = |view: &crate::complex_plane::View, max: u32, single: bool| {
        let mut c = ComplexPlane::new(w, h);
        c.set_view(view);
        let m = MandelbrotSet::new(max, 2.0);
        let mut last = Vec::new();
        assert!(Render::new(&c, &m, w, h, TrueColor::new_from_bernstein_polynomials).gpu(Some((&gpu, std::time::Duration::ZERO))).fast_gpu(true).f32_gpu(single).on_the_gpu_only()
            .run(|colors, _| last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect::<Vec<_>>()));
        last
    };
    for (name, view, max) in [("the whole set", crate::complex_plane::View::new(-0.75, 0.0, 1.0), 500), ("Frost", crate::complex_plane::View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1000)] {
        let (double, single) = (render(&view, max, false), render(&view, max, true));
        let differ = double.iter().zip(&single).filter(|(a, b)| a != b).count();
        println!("{name}: {:.2}% of the pixels differ from the double-float loop", differ as f64 * 100.0 / (w * h) as f64);
        assert!(differ * 10 < w * h, "{name}: {differ} of {} pixels", w * h);
    }
    //Too deep for f32 (Seahorse at 6.4e-5): the double-float loop, the same pixels
    let seahorse = crate::complex_plane::View::new(-0.7498100000000001, -0.020300000000000054, 0.000064);
    assert_eq!(render(&seahorse, 2000, true), render(&seahorse, 2000, false));
}

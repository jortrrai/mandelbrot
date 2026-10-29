//! Regression tests of the renderer through its public API, on small images, so that an accidental change of what the renderer computes is caught.
//!
//! What is compared exactly and what is not:
//! * **Iteration counts and a render with an integer test palette** are pinned exactly. They only use `+`, `-`, `*` and comparisons of `f64` (and integer arithmetic), which IEEE 754 defines
//!   bit for bit, and Rust does not fuse them (no FMA, no fast-math), so the same numbers come out on every CPU, with or without AVX2, and with any math library.
//! * **Smooth positions** use `ln`, which a math library may round differently in the last bit, so they are compared with a tolerance (`1e-9`) against sampled values, not pinned bit for bit.
//! * The real palettes (`Bernstein`, `HSV`) are not pinned at all: they use `powi` and `fmod`-like functions that may differ between compilers; they are covered by the unit tests, which compare them with themselves.
//!
//! If a pinned checksum fails, the renderer's output changed: find out why before updating it (`cargo test --test render_regression -- --nocapture` prints the actual values).
use std::sync::atomic::AtomicBool;

use mandelbrot::auto_iterations::{choose_limit, AutoIterationSettings};
use mandelbrot::coloring::TrueColor;
use mandelbrot::complex::Complex;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::rendering::{AdaptiveSampling, Render};

/// FNV-1a, 64 bits
fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// A palette of integer arithmetic only, so that the colors do not depend on any floating point function
fn integer_palette(iterations: u32, max_iterations: u32) -> TrueColor {
    if iterations >= max_iterations {
        TrueColor::new(255, 255, 255)
    } else {
        TrueColor::new((iterations * 7 % 256) as u8, (iterations * 13 % 256) as u8, (iterations * 3 % 251) as u8)
    }
}

struct View4 {
    name: &'static str,
    view: View,
    max_iterations: u32,
}

fn views() -> Vec<View4> {
    vec![
        View4 { name: "default", view: View::new(-0.75, 0.0, 1.0), max_iterations: 300 },
        View4 { name: "seahorse (boundary)", view: View::new(-0.7498100000000001, -0.020300000000000054, 0.00006400000000000002), max_iterations: 500 },
        View4 { name: "exterior", view: View::new(0.9, 0.9, 0.3), max_iterations: 100 },
        View4 { name: "inside the period-3 bulb", view: View::new(-0.12256116687665362, 0.7448617666197442, 0.03), max_iterations: 300 },
        View4 { name: "real axis tip", view: View::new(-1.9, 0.0, 0.05), max_iterations: 200 },
    ]
}

fn plane(view: &View, width: usize, height: usize) -> ComplexPlane {
    let mut plane = ComplexPlane::new(width, height);
    plane.set_view(view);
    plane
}

fn pixel_centers(plane: &ComplexPlane, width: usize, height: usize) -> Vec<Complex> {
    (0..width * height).map(|i| plane.complex_from_pixel_plane((i % width) as f64 + 0.5, (i / width) as f64 + 0.5)).collect()
}

#[test]
fn iteration_counts_of_small_views_match_pinned_checksums_on_the_scalar_and_the_simd_path() {
    let (width, height) = (48, 32);
    let mut actual = Vec::new();
    for view in views() {
        let points = pixel_centers(&plane(&view.view, width, height), width, height);
        let set = MandelbrotSet::new(view.max_iterations, 2.0);
        let scalar: Vec<u32> = points.iter().map(|point| set.iterate(point)).collect();
        //The loop the renderer uses (SIMD), with integer coloring: the position is the count
        let mut integer = set.clone();
        integer.smooth_coloring = false;
        let mut positions = vec![0.0; points.len()];
        integer.positions(&points, &mut positions, None);
        let refilled: Vec<u32> = positions.iter().map(|position| *position as u32).collect();
        assert_eq!(refilled, scalar, "{}: the refilling (SIMD) path equals the scalar path", view.name);
        actual.push(fnv(scalar.iter().flat_map(|count| count.to_le_bytes())));
    }
    assert_eq!(actual, PINNED_COUNTS, "the iteration counts of the views {:?} changed (actual checksums are listed on the left)", views().iter().map(|view| view.name).collect::<Vec<_>>());
}

fn render(view: &View4, width: usize, height: usize, samples: u8, adaptive: Option<&AdaptiveSampling>) -> Vec<TrueColor> {
    let plane = plane(&view.view, width, height);
    let mut set = MandelbrotSet::new(view.max_iterations, 2.0);
    set.smooth_coloring = false; //Integer positions: the colors then only depend on the counts and the integer palette
    let _base = vec![TrueColor::new(0, 0, 0); width * height];
    let mut last = Vec::new();
    Render::new(&plane, &set, width, height, integer_palette).supersampling(samples).adaptive(adaptive.copied()).run( |colors, _| {
        last.clear();
        last.extend_from_slice(colors);
    });
    last
}

fn color_checksum(colors: &[TrueColor]) -> u64 {
    fnv(colors.iter().flat_map(|color| [color.red, color.green, color.blue]))
}

#[test]
fn small_renders_with_an_integer_palette_match_pinned_checksums_uniform_and_adaptive() {
    let (width, height) = (40, 24);
    let mut actual = Vec::new();
    for view in &views() {
        let uniform = render(view, width, height, 4, None);
        let adaptive = render(view, width, height, 4, Some(&AdaptiveSampling::default()));
        assert_eq!(color_checksum(&render(view, width, height, 4, None)), color_checksum(&uniform), "{}: rendering is deterministic", view.name);
        actual.push((color_checksum(&uniform), color_checksum(&adaptive)));
    }
    assert_eq!(actual, PINNED_RENDERS, "a small supersampled render changed (actual (uniform, adaptive) checksums are listed on the left)");
}

/// Sampled points with their exact iteration count and their smooth position: (real, imaginary, max iterations, count, position)
const GOLDEN_SAMPLES: [(f64, f64, u32, u32, f64); 10] = GOLDEN;

#[test]
fn smooth_positions_match_golden_samples_within_a_tolerance_and_counts_exactly() {
    for (real, imaginary, max_iterations, count, position) in GOLDEN_SAMPLES {
        let set = MandelbrotSet::new(max_iterations, 2.0);
        let point = Complex::new(real, imaginary);
        println!("({real}, {imaginary}): count {}, position {:?}", set.iterate(&point), set.iterate_smooth(&point));
        assert_eq!(set.iterate(&point), count, "({real}, {imaginary}): the iteration count");
        let actual = set.iterate_smooth(&point);
        assert!((actual - position).abs() < 1e-9, "({real}, {imaginary}): the smooth position is {actual:?}, expected {position:?}");
        let lanes = set.position_lanes(&[Complex::new(real, imaginary), Complex::new(real, imaginary), Complex::new(real, imaginary), Complex::new(real, imaginary)]);
        assert_eq!(lanes[0].to_bits(), actual.to_bits(), "the SIMD path gives the same position as the scalar path");
    }
}

#[test]
fn the_automatic_iteration_limit_stays_close_to_what_a_much_higher_fixed_limit_shows() {
    //Loose bounds, not exact numbers: the choice is a heuristic on a sparse grid (see `auto_iterations`)
    let (width, height) = (128, 80);
    let reference_limit = 30_000;
    let settings = AutoIterationSettings::default();
    let cases = [
        ("default view", View::new(-0.75, 0.0, 1.0)),
        ("boundary, a few thousand times magnified", View::new(-0.7498100000000001, -0.020300000000000054, 0.0004)),
        ("inside the cardioid", View::new(-0.1, 0.0, 0.05)),
    ];
    for (name, view) in cases {
        let plane = plane(&view, width, height);
        let set = MandelbrotSet::new(1000, 2.0);
        let result = choose_limit(&plane, &set, width, height, &settings, 2, &AtomicBool::new(false)).expect("not cancelled");
        assert!(result.limit >= settings.initial_limit(plane.get_scale()) && result.limit <= settings.ceiling, "{name}: the limit {} is between the starting limit and the ceiling", result.limit);
        let other_threads = choose_limit(&plane, &set, width, height, &settings, 5, &AtomicBool::new(false)).unwrap();
        assert_eq!((result.limit, result.palette_scale, result.steps, result.stop.clone()), (other_threads.limit, other_threads.palette_scale, other_threads.steps, other_threads.stop), "{name}: the choice does not depend on the thread count");

        let points = pixel_centers(&plane, width, height);
        let chosen = MandelbrotSet::new(result.limit, 2.0);
        let reference = MandelbrotSet::new(reference_limit.max(result.limit), 2.0);
        let missed = points.iter().filter(|point| chosen.iterate(point) == result.limit && reference.iterate(point) < reference.max_iterations).count();
        let share = missed as f64 / points.len() as f64;
        println!("{name}: limit {} ({:?}), {:.3}% of the pixels are at the limit but escape under {}", result.limit, result.stop, 100.0 * share, reference.max_iterations);
        assert!(share < 0.05, "{name}: {:.2}% of the pixels are at the limit {} but escape under {}", 100.0 * share, result.limit, reference.max_iterations);
    }
    //A view that is only inside the set has nothing to refine
    let inside = plane(&View::new(-0.1, 0.0, 0.05), width, height);
    let result = choose_limit(&inside, &MandelbrotSet::new(1000, 2.0), width, height, &settings, 2, &AtomicBool::new(false)).unwrap();
    assert_eq!(result.limit, settings.initial_limit(inside.get_scale()), "certified interior points never raise the limit");
}

const PINNED_COUNTS: [u64; 5] = [234279750419972505, 5351827529005234398, 1732089381624647781, 7266845989527984933, 2998447490787724261];
const PINNED_RENDERS: [(u64, u64); 5] = [(4701750573522797798, 11102828296827810589), (15485460399544318285, 15485460399544318285), (8103064987490314918, 1292624869494761081), (12156545888424545509, 12156545888424545509), (8956487748394157651, 15678852834774847641)];
const GOLDEN: [(f64, f64, u32, u32, f64); 10] = [
    (0.0, 0.0, 300, 300, 300.0),                     //Inside the cardioid: never escapes, the position is the limit
    (1.0, 1.0, 100, 1, 0.8063197415255243),          //Far outside: escapes at once
    (-0.75, 0.1, 500, 32, 31.953334233507412),       //Near the cusp between the cardioid and the bulb
    (0.35, 0.4, 300, 16, 15.546595059267563),
    (-1.0, 0.3, 300, 34, 34.51591433342411),
    (-2.03, -0.13, 200, 0, 2.101220845012304),       //Hovers near radius 2 before it escapes: the smooth position is well above the count
    (0.26, 0.0, 500, 29, 28.520435492507897),        //Just outside the cardioid
    (-0.1011, 0.9563, 1000, 38, 37.45333547896049),
    (-0.745, 0.113, 2000, 126, 126.18957721964559),  //Slow boundary point
    (0.2501, 0.0, 2000, 311, 311.16594412281296),    //Just outside the cardioid cusp: many iterations
];

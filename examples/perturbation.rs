//! Compares direct iteration with perturbation (with and without BLA) on single-threaded work, run with `cargo run --release --example perturbation`.
//! Arguments: optional name filter.
use std::time::Instant;
use mandelbrot::complex::Complex;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::{MandelbrotSet, LANES};
use mandelbrot::perturbation::Reference;

fn views() -> Vec<(&'static str, View, u32)> {
    vec![
        ("VIEW_1 0.1", View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1000),
        ("VIEW_5 2e-5", View::new(-0.4375218333333333, 0.5632133750000003, 0.00002), 1000),
        ("detail 6.4e-5", View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), 5000),
        ("VIEW_7 1.7e-5", View::new(-1.7862712000000047, 0.000052399999999991516, 0.00001677721600000001), 5000),
        ("VIEW_0 4e-6", View::new(-0.437520465811966, 0.5632133750000006, 0.000004), 5000),
        ("seahorse 1e-6", View::new(-0.743643887037151, 0.131825904205330, 1e-6), 5000),
        ("seahorse 1e-8", View::new(-0.743643887037151, 0.131825904205330, 1e-8), 20000),
    ]
}

fn main() {
    let filter = std::env::args().nth(1);
    let (w, h) = (600usize, 400usize); //Half size in each direction: single-threaded runs
    for (name, view, max_it) in views() {
        if filter.as_ref().map_or(false, |f| !name.contains(f.as_str())) { continue; }
        let mut c = ComplexPlane::new(w, h);
        c.set_view(&view);
        let m = MandelbrotSet::new(max_it, 2.0);
        let points: Vec<Complex> = (0..w * h).map(|i| c.complex_from_pixel_plane((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)).collect();
        let center = c.complex_from_pixel_plane(w as f64 / 2.0, h as f64 / 2.0);
        let dcs: Vec<(f64, f64)> = points.iter().map(|p| (p.x - center.x, p.y - center.y)).collect();
        let dc_max = dcs.iter().map(|d| d.0.hypot(d.1)).fold(0.0, f64::max);

        let t = Instant::now();
        let direct: Vec<u32> = points.iter().map(|p| m.iterate(p)).collect();
        let t_scalar = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut lanes = Vec::with_capacity(points.len());
        for g in points.chunks_exact(LANES) { lanes.extend_from_slice(&m.iterate_lanes(&std::array::from_fn(|i| Complex::new(g[i].x, g[i].y)))); }
        let t_lanes = t.elapsed().as_secs_f64();
        assert_eq!(&lanes[..], &direct[..lanes.len()]);
        println!("\n{name} (pixel {:.1e}, {max_it} it): direct scalar {:.3}s, direct 4-lane AVX2 {:.3}s", c.increment_x, t_scalar, t_lanes);

        for (label, eps, bla) in [("perturbation, no BLA", 0.0, false), ("BLA eps 1e-6", 1e-6, true), ("BLA eps 1e-9", 1e-9, true), ("BLA eps 1e-12", 1e-12, true)] {
            let t = Instant::now();
            let r = Reference::new(&center, max_it, 2.0, dc_max, eps);
            let t_ref = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let results: Vec<(u32, u32)> = dcs.iter().map(|d| r.iterate(*d, bla)).collect();
            let t_it = t.elapsed().as_secs_f64();
            let mismatches = results.iter().zip(&direct).filter(|(a, b)| a.0 != **b).count();
            let max_diff = results.iter().zip(&direct).map(|(a, b)| a.0.abs_diff(*b)).max().unwrap_or(0);
            let mean_diff = results.iter().zip(&direct).map(|(a, b)| f64::from(a.0.abs_diff(*b))).sum::<f64>() / results.len() as f64;
            let passes: u64 = results.iter().map(|r| u64::from(r.1)).sum();
            let steps: u64 = results.iter().map(|r| u64::from(r.0)).sum();
            println!("  {label:<22} ref {:.4}s + pixels {:.3}s = {:.3}s  ({:+.0}% vs 4-lane)  loop passes/iterations {:.3}  mismatching pixels {} ({:.3}%), max diff {max_diff}, mean diff {mean_diff:.4}",
                t_ref, t_it, t_ref + t_it, 100.0 * ((t_ref + t_it) / t_lanes - 1.0), passes as f64 / steps.max(1) as f64, mismatches, 100.0 * mismatches as f64 / results.len() as f64);
        }
    }
}

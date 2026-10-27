//! Where the time of smooth coloring goes, single threaded: the lockstep loop with and without the escape capture, and the smooth position itself.
//! Run with `cargo run --release --example smooth_cost`.
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::Instant;
use mandelbrot::coloring::TrueColor;
use mandelbrot::complex::Complex;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::{MandelbrotSet, LANES};

fn main() {
    let (w, h) = (600usize, 400usize);
    for (name, view, max_it) in [("detail", View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), 5000u32), ("default", View::new(-0.75, 0.0, 1.0), 1000), ("exterior", View::new(0.9, 0.9, 0.3), 1000)] {
        let mut c = ComplexPlane::new(w, h);
        c.set_view(&view);
        let m = MandelbrotSet::new(max_it, 2.0);
        let points: Vec<Complex> = (0..w * h).map(|i| c.complex_from_pixel_plane((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)).collect();
        let groups: Vec<[Complex; LANES]> = points.chunks_exact(LANES).map(|g| std::array::from_fn(|i| Complex::new(g[i].x, g[i].y))).collect();
        let time = |label: &str, f: &dyn Fn() -> f64| {
            let mut best = f64::MAX;
            let mut sink = 0.0;
            for _ in 0..7 { let t = Instant::now(); sink += f(); best = best.min(t.elapsed().as_secs_f64() * 1000.0); }
            println!("{name:<9} {label:<34} {best:8.2} ms (best of 7, sink {sink:.0})");
        };
        for smooth in [false, true] {
            let mut m2 = m.clone();
            m2.smooth_coloring = smooth;
            let (w2, h2) = (1200usize, 800usize);
            let mut c2 = ComplexPlane::new(w2, h2);
            c2.set_view(&view);
            time(&format!("full render 15 threads, smooth={smooth}"), &|| { let mut v = 0.0; mandelbrot::rendering::render_progressively(&c2, &m2, w2, h2, 1, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &AtomicUsize::new(0), |colors, _| v += f64::from(colors[colors.len() / 2].red)); v });
        }
        let threads = std::env::var("THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or_else(mandelbrot::rendering::render_thread_count);
        let chunk = groups.len().div_ceil(threads);
        for smooth in [false, true] {
            time(&format!("lanes only, {threads} threads, smooth={smooth}"), &|| {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = groups.chunks(chunk).map(|part| scope.spawn(|| part.iter().map(|g| if smooth { m.iterate_lanes_smooth(g).iter().sum::<f64>() } else { m.iterate_lanes(g).iter().map(|v| f64::from(*v)).sum::<f64>() }).sum::<f64>())).collect();
                    handles.into_iter().map(|h| h.join().unwrap()).sum()
                })
            });
        }
        time("iterate_lanes (integer)", &|| groups.iter().map(|g| m.iterate_lanes(g).iter().map(|v| f64::from(*v)).sum::<f64>()).sum());
        time("iterate_lanes_smooth", &|| groups.iter().map(|g| m.iterate_lanes_smooth(g).iter().sum::<f64>()).sum());
        time("scalar iterate (integer)", &|| points.iter().map(|p| f64::from(m.iterate(p))).sum());
        time("scalar iterate_smooth", &|| points.iter().map(|p| m.iterate_smooth(p)).sum());
    }
}

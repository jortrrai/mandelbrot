//! Work analysis: how the iteration counts of a view are distributed, run with `cargo run --release --example analyze`.
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;

fn views() -> Vec<(&'static str, View, u32)> {
    vec![
        ("default", View::new(-0.75, 0.0, 1.0), 1000),
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
    let (w, h) = (1200usize, 800usize);
    for (name, view, max_it) in views() {
        let mut c = ComplexPlane::new(w, h);
        c.set_view(&view);
        let m = MandelbrotSet::new(max_it, 2.0);
        let mut counts = Vec::with_capacity(w * h);
        let t = std::time::Instant::now();
        for y in 0..h { for x in 0..w { counts.push(m.iterate(&c.complex_from_pixel_plane(x as f64 + 0.5, y as f64 + 0.5))); } }
        let secs = t.elapsed().as_secs_f64();
        let n = counts.len() as f64;
        let interior = counts.iter().filter(|&&i| i == max_it).count();
        let esc: Vec<u32> = counts.iter().copied().filter(|&i| i < max_it).collect();
        let total: u64 = esc.iter().map(|&i| u64::from(i)).sum();
        let hi = esc.iter().filter(|&&i| i >= max_it / 10).count();
        println!("{name:<14} max_it {max_it:>6} pixel {:.1e}: interior {:>5.1}%  escaped<10%max {:>5.1}%  escaped>=10%max {:>5.1}%  mean escaped iters {:>7.1}  scalar single-thread {:.2}s",
            c.increment_x, 100.0 * interior as f64 / n, 100.0 * (esc.len() - hi) as f64 / n, 100.0 * hi as f64 / n, total as f64 / esc.len().max(1) as f64, secs);
    }
}

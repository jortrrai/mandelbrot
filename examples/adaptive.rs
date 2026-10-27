//! Adaptive versus uniform supersampling at matched settings, run with `cargo run --release --example adaptive [samples] [runs]`.
//! Reports the share of pixels that get all samples, the sample count, the render time, the cost of the detection, and the difference of the image to the uniform render.
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::Instant;
use mandelbrot::coloring::TrueColor;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::rendering::{adaptive_mask, render_region_adaptively, AdaptiveSampling, ExposedRegion};

fn views() -> Vec<(&'static str, &'static str, View, u32)> {
    vec![
        ("smooth", "default view", View::new(-0.75, 0.0, 1.0), 1000),
        ("smooth", "exterior", View::new(0.9, 0.9, 0.3), 1000),
        ("boundary", "VIEW_1 (0.1)", View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1000),
        ("boundary", "detail 6.4e-5", View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), 5000),
        ("fine detail", "VIEW_7 1.7e-5", View::new(-1.7862712000000047, 0.000052399999999991516, 0.00001677721600000001), 5000),
        ("fine detail", "seahorse 1e-6", View::new(-0.743643887037151, 0.131825904205330, 1e-6), 5000),
    ]
}

fn render(c: &ComplexPlane, m: &MandelbrotSet, w: usize, h: usize, samples: u8, adaptive: Option<&AdaptiveSampling>) -> Vec<TrueColor> {
    let base = vec![TrueColor::new(0, 0, 0); w * h];
    let mut last = Vec::new();
    render_region_adaptively(c, m, w, h, &ExposedRegion::full(w, h), &base, samples, adaptive, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &AtomicUsize::new(0), |colors, _| { last.clear(); last.extend_from_slice(colors); });
    last
}

fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    f(); f();
    let mut t: Vec<f64> = (0..runs).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1000.0 }).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

fn diff(a: &[TrueColor], b: &[TrueColor]) -> (f64, f64, u8) {
    let (mut sum, mut over, mut max) = (0u64, 0usize, 0u8);
    for (x, y) in a.iter().zip(b) {
        let d = [x.red.abs_diff(y.red), x.green.abs_diff(y.green), x.blue.abs_diff(y.blue)];
        sum += d.iter().map(|v| u64::from(*v)).sum::<u64>();
        let m = *d.iter().max().unwrap();
        max = max.max(m);
        over += usize::from(m > 8);
    }
    (sum as f64 / (a.len() * 3) as f64, 100.0 * over as f64 / a.len() as f64, max)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let samples: u8 = args.next().and_then(|a| a.parse().ok()).unwrap_or(8);
    let runs: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(7);
    let (w, h) = (1200usize, 800usize);
    let adaptive = AdaptiveSampling::default();
    println!("{w}x{h}, {samples} samples per pixel, {} threads, {adaptive:?}, {runs} runs (median)\n", mandelbrot::rendering::render_thread_count());
    println!("{:<12} {:<15} {:>9} {:>9} {:>8} {:>9} {:>8} | {:>26} | {:>26}", "kind", "view", "uniform", "adaptive", "change", "refined", "mask ms", "adaptive vs uniform", "2 samples vs uniform");
    println!("{:<12} {:<15} {:>9} {:>9} {:>8} {:>9} {:>8} | {:>26} | {:>26}", "", "", "ms", "ms", "", "pixels", "", "mean|max|% px >8", "mean|max|% px >8");
    for (kind, name, view, max_it) in views() {
        let mut c = ComplexPlane::new(w, h);
        c.set_view(&view);
        let m = MandelbrotSet::new(max_it, 2.0);
        let uniform = render(&c, &m, w, h, samples, None);
        let adaptive_image = render(&c, &m, w, h, samples, Some(&adaptive));
        let two = render(&c, &m, w, h, 2, None);
        let t_uniform = median_ms(runs, || { std::hint::black_box(render(&c, &m, w, h, samples, None)); });
        let t_adaptive = median_ms(runs, || { std::hint::black_box(render(&c, &m, w, h, samples, Some(&adaptive))); });
        //Detection cost and the share of refined pixels, from the 2-sample image and the first sample
        let first = render(&c, &m, w, h, 1, None);
        let region = ExposedRegion::full(w, h);
        let mut mask = Vec::new();
        let t_mask = median_ms(runs, || { mask = adaptive_mask(w, h, &region, &first, &two, adaptive.threshold); });
        let refined = mask.iter().filter(|m| **m).count() as f64 / (w * h) as f64;
        let (a, b) = (diff(&adaptive_image, &uniform), diff(&two, &uniform));
        println!("{kind:<12} {name:<15} {t_uniform:>9.1} {t_adaptive:>9.1} {:>+7.0}% {:>8.1}% {t_mask:>8.2} | {:>8.3} {:>4} {:>9.3}% | {:>8.3} {:>4} {:>9.3}%",
            100.0 * (t_adaptive / t_uniform - 1.0), 100.0 * refined, a.0, a.2, a.1, b.0, b.2, b.1);
    }
    println!("\nsamples taken: uniform {samples} per pixel; adaptive 2 + {}*refined share.", samples - 2);
}

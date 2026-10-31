//! Evaluates the automatic iteration limit against a fixed high-limit reference, run with `cargo run --release --example auto_iterations`.
//! For every view: the chosen limit, the rounds, the time of the choice, and how many pixels (full resolution, pixel centers) differ from a render at the reference limit:
//! `missed` = pixels that are at the chosen limit but escape under the reference limit.
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use mandelbrot::auto_iterations::{choose_limit, AutoIterationSettings};
use mandelbrot::complex::Complex;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;

fn views() -> Vec<(&'static str, View)> {
    vec![
        ("default", View::new(-0.75, 0.0, 1.0)),
        ("VIEW_1 0.1", View::new(-0.6604166666666667, 0.4437500000000001, 0.1)),
        ("inside cardioid", View::new(-0.1, 0.0, 0.05)),
        ("inside bulb 0.03", View::new(-0.12256116687665362, 0.7448617666197442, 0.03)),
        ("VIEW_5 2e-5", View::new(-0.4375218333333333, 0.5632133750000003, 0.00002)),
        ("detail 6.4e-5", View::new(-0.7498100000000001, -0.020300000000000054, 0.000064)),
        ("VIEW_7 1.7e-5", View::new(-1.7862712000000047, 0.000052399999999991516, 0.00001677721600000001)),
        ("VIEW_0 4e-6", View::new(-0.437520465811966, 0.5632133750000006, 0.000004)),
        ("seahorse 1e-6", View::new(-0.743643887037151, 0.131825904205330, 1e-6)),
        ("seahorse 1e-8", View::new(-0.743643887037151, 0.131825904205330, 1e-8)),
    ]
}

fn render(c: &ComplexPlane, m: &MandelbrotSet, w: usize, h: usize, threads: usize) -> Vec<u32> {
    let points: Vec<Complex> = (0..w * h).map(|i| c.complex_from_pixel_plane((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)).collect();
    //Integer coloring: the position of a point is its iteration count
    let mut integer = m.clone();
    integer.smooth_coloring = false;
    let mut out = vec![0.0; w * h];
    let chunk = points.len() / threads + 1;
    std::thread::scope(|s| {
        for (o, p) in out.chunks_mut(chunk).zip(points.chunks(chunk)) {
            let integer = &integer;
            s.spawn(move || integer.positions(p, o, None));
        }
    });
    out.iter().map(|position| *position as u32).collect()
}

fn main() {
    let reference_limit: u32 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(50_000);
    let (w, h) = (1200usize, 800usize);
    let threads = mandelbrot::rendering::render_thread_count();
    let mut settings = AutoIterationSettings::default();
    if let Some(s) = std::env::args().nth(2).and_then(|a| a.parse().ok()) { settings.probe_stride = s; }
    if let Some(f) = std::env::args().nth(3).and_then(|a| a.parse().ok()) { settings.min_new_escape_fraction = f; }
    println!("settings: {settings:?}\nreference limit {reference_limit}, image {w}x{h}, {threads} threads\n");
    for (name, view) in views() {
        let mut c = ComplexPlane::new(w, h);
        c.set_view(&view);
        let m = MandelbrotSet::new(1000, 2.0);
        let t = Instant::now();
        let r = choose_limit(&c, &m, w, h, &settings, threads, &AtomicBool::new(false)).unwrap();
        let t_choose = t.elapsed().as_secs_f64();
        let auto_m = MandelbrotSet::new(r.limit, 2.0);
        let t = Instant::now();
        let auto = render(&c, &auto_m, w, h, threads);
        let t_auto = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let reference = render(&c, &MandelbrotSet::new(reference_limit, 2.0), w, h, threads);
        let t_ref = t.elapsed().as_secs_f64();
        let n = (w * h) as f64;
        let missed = auto.iter().zip(&reference).filter(|(a, r)| **a == auto_m.max_iterations && **r < reference_limit).count();
        let differ_escaped = auto.iter().zip(&reference).filter(|(a, r)| **a < auto_m.max_iterations && a != r).count();
        let ref_unresolved = reference.iter().filter(|r| **r == reference_limit).count();
        println!("{name:<17} limit {:>6} (initial {:>5}, {} rounds, {:?})  choose {:.3}s ({:.0}% of the {:.3}s render at that limit; fixed-{} render {:.2}s)\n  {:>6.3}% pixels missed (at limit, escape by {})   {:.4}% escaped-pixels differ   {:.2}% still at the reference limit",
            r.limit, r.palette_scale, r.rounds.len(), r.stop, t_choose, 100.0 * t_choose / t_auto, t_auto, reference_limit, t_ref, 100.0 * missed as f64 / n, reference_limit, 100.0 * differ_escaped as f64 / n, 100.0 * ref_unresolved as f64 / n);
        for round in &r.rounds { println!("    round limit {:>6}: escaped {:>6} certified {:>6} unresolved {:>6} newly escaped {:>5}", round.limit, round.escaped, round.certified, round.unresolved, round.newly_escaped); }
    }
}

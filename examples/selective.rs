//! Selective smoothing against full smoothing and integer coloring, run with `cargo run --release --example selective [runs]`.
//! For every workload and setting: the median render time, the change against full smoothing, and how the image differs from the fully smoothed image
//! (mean channel difference, share of pixels with a channel sum difference above 2, largest channel sum difference).
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::Instant;
use mandelbrot::coloring::TrueColor;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::rendering;

struct Workload { name: &'static str, view: View, max_iterations: u32, palette_scale: Option<u32>, samples: u8 }

fn render(w: &Workload, m: &MandelbrotSet, size: (usize, usize)) -> Vec<TrueColor> {
    let mut c = ComplexPlane::new(size.0, size.1);
    c.set_view(&w.view);
    let mut last = Vec::new();
    rendering::render_progressively(&c, m, size.0, size.1, w.samples, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &AtomicUsize::new(0), |colors, _| { last.clear(); last.extend_from_slice(colors); });
    last
}

fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    f(); f();
    let mut t: Vec<f64> = (0..runs).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1000.0 }).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

fn diff(a: &[TrueColor], b: &[TrueColor]) -> (f64, f64, u32) {
    let (mut sum, mut over, mut max) = (0u64, 0usize, 0u32);
    for (x, y) in a.iter().zip(b) {
        let d = [x.red.abs_diff(y.red), x.green.abs_diff(y.green), x.blue.abs_diff(y.blue)].map(u32::from);
        sum += u64::from(d[0] + d[1] + d[2]);
        over += usize::from(d[0] + d[1] + d[2] > 2);
        max = max.max(d[0] + d[1] + d[2]);
    }
    (sum as f64 / (a.len() * 3) as f64, 100.0 * over as f64 / a.len() as f64, max)
}

fn main() {
    let runs: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(9);
    let size = (1200usize, 800usize);
    let workloads = [
        Workload { name: "default 1000 it", view: View::new(-0.75, 0.0, 1.0), max_iterations: 1000, palette_scale: None, samples: 1 },
        Workload { name: "exterior 1000 it", view: View::new(0.9, 0.9, 0.3), max_iterations: 1000, palette_scale: None, samples: 1 },
        Workload { name: "interior-bulb 1000", view: View::new(-0.12256116687665362, 0.7448617666197442, 0.03), max_iterations: 1000, palette_scale: None, samples: 1 },
        Workload { name: "detail 5000 it", view: View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), max_iterations: 5000, palette_scale: None, samples: 1 },
        Workload { name: "default auto (500/250)", view: View::new(-0.75, 0.0, 1.0), max_iterations: 500, palette_scale: Some(250), samples: 1 },
        Workload { name: "view1 auto (1164/582)", view: View::new(-0.6604166666666667, 0.4437500000000001, 0.1), max_iterations: 1164, palette_scale: Some(582), samples: 1 },
        Workload { name: "detail auto (5000/1643)", view: View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), max_iterations: 5000, palette_scale: Some(1643), samples: 1 },
        Workload { name: "default auto 4x", view: View::new(-0.75, 0.0, 1.0), max_iterations: 500, palette_scale: Some(250), samples: 4 },
    ];
    println!("1200x800, {} threads, {runs} runs (median); diff = vs full smoothing: mean channel | % px with sum>2 | max sum\n", rendering::render_thread_count());
    println!("{:<24} {:<22} {:>9} {:>9} | {:>7} {:>8} {:>5}", "workload", "setting", "ms", "vs full", "mean", "% px>2", "max");
    for w in &workloads {
        let mut m = MandelbrotSet::new(w.max_iterations, 2.0);
        m.palette_scale = w.palette_scale;
        let full = render(w, &m, size);
        let t_full = median_ms(runs, || { std::hint::black_box(render(w, &m, size)); });
        println!("{:<24} {:<22} {t_full:>9.2} {:>9} |", w.name, "full smoothing", "");
        let mut int = m.clone();
        int.smooth_coloring = false;
        let t_int = median_ms(runs, || { std::hint::black_box(render(w, &int, size)); });
        let d = diff(&render(w, &int, size), &full);
        println!("{:<24} {:<22} {t_int:>9.2} {:>+8.0}% | {:>7.3} {:>7.2}% {:>5}", "", "integer (no smoothing)", 100.0 * (t_int / t_full - 1.0), d.0, d.1, d.2);
        for threshold in [4u32, 8, 16, 32] {
            for replay in [false, true] {
                let mut s = m.clone();
                s.smooth_threshold = Some(threshold);
                s.selective_replay = replay;
                let t = median_ms(runs, || { std::hint::black_box(render(w, &s, size)); });
                let d = diff(&render(w, &s, size), &full);
                println!("{:<24} {:<22} {t:>9.2} {:>+8.0}% | {:>7.3} {:>7.2}% {:>5}", "", format!("T={threshold} {}", if replay { "replay" } else { "capture" }), 100.0 * (t / t_full - 1.0), d.0, d.1, d.2);
            }
        }
    }
}

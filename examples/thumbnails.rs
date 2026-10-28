//! Cost of the thumbnails of the points of interest, run with `cargo run --release --example thumbnails`.
//! Times one thumbnail of every place of the shipped folder, and measures what a thumbnail does to a main render: continuously in the background (what the policy avoids),
//! and with the policy of the window (a thumbnail that is running is cancelled when the main render starts).
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use mandelbrot::coloring::{ColorChannelMapping, TrueColor};
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::points_of_interest::{self, Coloring, Location};
use mandelbrot::rendering::{self, Render};
use mandelbrot::thumbnails::{self, ThumbnailRequest};

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn time_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    median((0..runs).map(|_| { let start = Instant::now(); f(); start.elapsed().as_secs_f64() * 1000.0 }).collect())
}

fn main_render() {
    let (w, h) = (1200, 800);
    let mut c = ComplexPlane::new(w, h);
    c.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.00006400000000000002));
    let m = MandelbrotSet::new(5000, 2.0);
    std::hint::black_box(Render::new(&c, &m, w, h, TrueColor::new_from_bernstein_polynomials).supersampling(4).run( |_, _| {}));
}

fn main() {
    let report = points_of_interest::load_folder(&points_of_interest::find_folder());
    let height = thumbnails::thumbnail_height(1200, 800);
    let requests: Vec<(String, ThumbnailRequest)> = report.points.iter().map(|p| (p.name.clone(), ThumbnailRequest { location: p.location.clone(), coloring: Coloring::Bernstein, color_channels: ColorChannelMapping::RGB, width: thumbnails::THUMBNAIL_WIDTH, height })).collect();
    println!("{} places, thumbnails {}x{height}, {} samples per pixel, {} render threads\n", requests.len(), thumbnails::THUMBNAIL_WIDTH, thumbnails::THUMBNAIL_SAMPLES, rendering::render_thread_count());
    let mut total = 0.0;
    for (name, request) in &requests {
        let ms = time_ms(7, || { std::hint::black_box(thumbnails::render_thumbnail(request, 2.0, &AtomicBool::new(false))); });
        total += ms;
        println!("  {name:<24} {ms:6.2} ms ({} it)", request.location.max_iterations);
    }
    println!("  all {}: {total:.1} ms, {:.1} ms per place; a texture is {} KB, at most {} are kept = {:.1} MB\n", requests.len(), total / requests.len() as f64, thumbnails::THUMBNAIL_WIDTH * height * 4 / 1024, thumbnails::MAX_CACHED, (thumbnails::THUMBNAIL_WIDTH * height * 4 * thumbnails::MAX_CACHED) as f64 / 1e6);

    let alone = time_ms(7, main_render);
    println!("main render (seahorse-embers, 1200x800, 4x, 5000 it) alone: {alone:.1} ms");

    //Without a policy: thumbnails render continuously while the main render runs
    let stop = Arc::new(AtomicBool::new(false));
    let background = { let (stop, requests) = (stop.clone(), requests.clone()); std::thread::spawn(move || { while !stop.load(Ordering::Relaxed) { for (_, request) in &requests { if stop.load(Ordering::Relaxed) { break; } std::hint::black_box(thumbnails::render_thumbnail(request, 2.0, &AtomicBool::new(false))); } } }) };
    std::thread::sleep(std::time::Duration::from_millis(50));
    let busy = time_ms(7, main_render);
    stop.store(true, Ordering::Relaxed);
    background.join().unwrap();
    println!("  with thumbnails rendering continuously in the background (no policy): {busy:.1} ms ({:+.0}%)", 100.0 * (busy / alone - 1.0));

    //With the policy: a thumbnail (the slowest kind: an enormous limit, capped) is running when the main render starts, so it is cancelled first
    let slow = ThumbnailRequest { location: Location { real: -0.7498100000000001, imaginary: -0.020300000000000054, scale: 0.00006400000000000002, max_iterations: 1_000_000 }, ..requests[0].1.clone() };
    let slow_alone = time_ms(5, || { std::hint::black_box(thumbnails::render_thumbnail(&slow, 2.0, &AtomicBool::new(false))); });
    let mut main_times = Vec::new();
    let mut cancel_latencies = Vec::new();
    for _ in 0..7 {
        let cancel = Arc::new(AtomicBool::new(false));
        let job = { let (cancel, slow) = (cancel.clone(), slow.clone()); std::thread::spawn(move || { let result = thumbnails::render_thumbnail(&slow, 2.0, &cancel); (Instant::now(), result.is_none()) }) };
        std::thread::sleep(std::time::Duration::from_millis(20));
        let start = Instant::now();
        cancel.store(true, Ordering::Relaxed); //What the window does when a render starts
        main_render();
        main_times.push(start.elapsed().as_secs_f64() * 1000.0);
        let (finished, was_cancelled) = job.join().unwrap();
        if was_cancelled { cancel_latencies.push(finished.saturating_duration_since(start).as_secs_f64() * 1000.0); }
    }
    let main_median = median(main_times);
    println!("  slowest thumbnail ({slow_alone:.0} ms alone) running when the main render starts, cancelled by the policy: main render {main_median:.1} ms ({:+.0}%); {} of 7 thumbnails were still running and stopped {:.1} ms after the cancel (median)", 100.0 * (main_median / alone - 1.0), cancel_latencies.len(), if cancel_latencies.is_empty() { f64::NAN } else { median(cancel_latencies) });
}

//! Timing harness for the renderer, run with `cargo run --release --example perf`.
//! Prints the median, minimum and maximum time of every stage, and a checksum of the rendered colors:
//! the checksum is the same for any version that renders the same image, so it proves that an optimization did not change the output.
//! Optional arguments: the amount of runs (default 9) and a name filter for the workloads, e.g. `cargo run --release --example perf -- 15 detail`.
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::{Duration, Instant};

use mandelbrot::coloring::{ColorChannelMapping, TrueColor};
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::pixel_buffer::PixelBuffer;
use mandelbrot::rendering;

const WIDTH: usize = 1200;
const HEIGHT: usize = 800;
const WARMUP_RUNS: usize = 2;

struct Workload {
    name: &'static str,
    view: View,
    max_iterations: u32,
    supersampling_amount: u8,
}

fn workloads() -> Vec<Workload> {
    vec![
        Workload { name: "default-view", view: View::new(-0.75, 0.0, 1.0), max_iterations: 1000, supersampling_amount: 1 },
        Workload { name: "default-view-4x", view: View::new(-0.75, 0.0, 1.0), max_iterations: 1000, supersampling_amount: 4 },
        Workload { name: "detail", view: View::new(-0.7498100000000001, -0.020300000000000054, 0.00006400000000000002), max_iterations: 5000, supersampling_amount: 1 },
        Workload { name: "exterior", view: View::new(0.9, 0.9, 0.3), max_iterations: 1000, supersampling_amount: 1 },
        //Almost no iterating: measures the cost of everything around the iteration loop (jitter, coordinates, colors, sums, threads)
        Workload { name: "overhead", view: View::new(-0.75, 0.0, 1.0), max_iterations: 1, supersampling_amount: 1 },
        //Inside the period-3 bulb: interior points that are not in the main cardioid or period-2 bulb, so they iterate until the limit
        Workload { name: "interior-bulb", view: View::new(-0.12256116687665362, 0.7448617666197442, 0.03), max_iterations: 1000, supersampling_amount: 1 },
    ]
}

fn summarize(name: &str, mut times: Vec<Duration>) {
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    println!("{name:<44} median {:>9.2} ms   min {:>9.2} ms   max {:>9.2} ms   ({} runs)", ms(times[times.len() / 2]), ms(times[0]), ms(times[times.len() - 1]), times.len());
}

fn time_runs(runs: usize, mut run: impl FnMut()) -> Vec<Duration> {
    for _ in 0..WARMUP_RUNS {
        run();
    }
    (0..runs).map(|_| {
        let start = Instant::now();
        run();
        start.elapsed()
    }).collect()
}

fn checksum(colors: &[TrueColor]) -> u64 {
    colors.iter().fold(0xcbf2_9ce4_8422_2325, |hash, color| {
        [color.red, color.green, color.blue].iter().fold(hash, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

fn render(workload: &Workload, coloring_function: fn(u32, u32) -> TrueColor) -> Vec<TrueColor> {
    let mut c = ComplexPlane::new(WIDTH, HEIGHT);
    c.set_view(&workload.view);
    let m = MandelbrotSet::new(workload.max_iterations, 2.0);
    let mut last = Vec::new();
    rendering::render_progressively(&c, &m, WIDTH, HEIGHT, workload.supersampling_amount, coloring_function, &AtomicBool::new(false), &AtomicUsize::new(0), |colors, _| {
        last.clear();
        last.extend_from_slice(colors);
    });
    last
}

fn main() {
    let mut args = std::env::args().skip(1);
    let runs: usize = args.next().and_then(|arg| arg.parse().ok()).unwrap_or(9);
    let filter = args.next();

    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| info.lines().find(|line| line.starts_with("model name")).map(|line| line.split(':').nth(1).unwrap_or("").trim().to_string())).unwrap_or_default();
    println!("CPU: {cpu}, render threads: {}, image: {WIDTH}x{HEIGHT}, runs: {runs} after {WARMUP_RUNS} warmup runs", rendering::render_thread_count());
    println!("The checksums identify the rendered image, they should not change between versions that render the same image.\n");

    for workload in workloads().iter().filter(|w| filter.as_ref().map_or(true, |f| w.name.contains(f.as_str()))) {
        for (coloring_name, coloring_function) in [("bernstein", TrueColor::new_from_bernstein_polynomials as fn(u32, u32) -> TrueColor), ("hsv", TrueColor::new_from_hsv_colors)] {
            let label = format!("render {} ({coloring_name}, {} it, {}x)", workload.name, workload.max_iterations, workload.supersampling_amount);
            let checksum = checksum(&render(workload, coloring_function));
            let times = time_runs(runs, || {
                std::hint::black_box(render(workload, coloring_function));
            });
            summarize(&label, times);
            println!("{:<44} checksum {checksum:016x}", "");
        }
    }

    //Shifting the pixel buffer, which is what a move does before rendering the exposed strip
    if filter.as_ref().map_or(true, |f| "shift".contains(f.as_str())) {
        let mut buffer = PixelBuffer::new(mandelbrot::pixel_buffer::pixel_plane::PixelPlane::new(WIDTH, HEIGHT));
        buffer.colors = render(&workloads()[0], TrueColor::new_from_bernstein_polynomials);
        buffer.update_pixels();
        buffer.translate_buffer(10, -20);
        let checksum = checksum(&buffer.colors) ^ buffer.pixels.iter().fold(0u64, |hash, pixel| hash.rotate_left(5) ^ u64::from(*pixel));
        summarize("translate_buffer (shift by 10 rows, 20 columns)", time_runs(runs * 3, || {
            buffer.translate_buffer(10, -20);
            buffer.translate_buffer(-10, 20);
        }));
        println!("{:<44} checksum {checksum:016x}  (time is for two shifts)", "");
    }

    //Color conversion of a whole buffer, including resolving the channel mapping
    if filter.as_ref().map_or(true, |f| "conversion".contains(f.as_str())) {
        let colors = render(&workloads()[0], TrueColor::new_from_bernstein_polynomials);
        let mapping = ColorChannelMapping::RGB;
        let pixels = PixelBuffer::colors_to_pixels(&colors, &mapping);
        let checksum = pixels.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, pixel| (hash ^ u64::from(*pixel)).wrapping_mul(0x0000_0100_0000_01b3));
        summarize("color conversion (colors_to_pixels, 960k)", time_runs(runs * 3, || {
            std::hint::black_box(PixelBuffer::colors_to_pixels(&colors, &mapping));
        }));
        println!("{:<44} checksum {checksum:016x}", "");
    }
}

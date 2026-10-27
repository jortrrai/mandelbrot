//! Renders a few views to PNG files in `saved/` of the current directory, to look at the coloring: `cargo run --release --example snapshot -- <name prefix>`.
use std::sync::atomic::{AtomicBool, AtomicUsize};
use mandelbrot::coloring::TrueColor;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
use mandelbrot::rendering;

fn main() {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| String::from("snapshot"));
    let (w, h) = (800usize, 500usize);
    let views = [
        ("default", View::new(-0.75, 0.0, 1.0), 1000u32, 1u8),
        ("view1", View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1000, 1),
        ("exterior", View::new(0.9, 0.9, 0.3), 1000, 1),
        ("detail", View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), 5000, 1),
        ("view1-4x", View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1000, 4),
    ];
    for (name, view, max_iterations, samples) in views {
        let mut c = ComplexPlane::new(w, h);
        c.set_view(&view);
        let m = MandelbrotSet::new(max_iterations, 2.0);
        let mut p = PixelBuffer::new(PixelPlane::new(w, h));
        rendering::render_progressively(&c, &m, w, h, samples, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &AtomicUsize::new(0), |colors, _| p.colors.copy_from_slice(colors));
        p.update_pixels();
        p.save_as_png(&format!("{prefix}-{name}"), &view, &m, samples);
    }
}

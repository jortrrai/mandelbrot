//! Renders views with integer and with smooth coloring side by side (palette scaled like auto iterations at zoom 1), run with `cargo run --release --example smooth_compare`.
//! Writes `saved/compare-<view>-integer.png` and `-smooth.png` in the current directory.
use std::sync::atomic::{AtomicBool, AtomicUsize};
use mandelbrot::coloring::TrueColor;
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
use mandelbrot::rendering;

fn main() {
    let (w, h) = (800usize, 500usize);
    let views = [
        ("default-auto250", View::new(-0.75, 0.0, 1.0), 500u32, Some(250u32)),
        ("view1-auto582", View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1164, Some(582)),
        ("view1-manual", View::new(-0.6604166666666667, 0.4437500000000001, 0.1), 1000, None),
        ("mid-manual", View::new(-0.7498100000000001, -0.020300000000000054, 0.5), 400, None),
    ];
    for (name, view, max_iterations, palette_scale) in views {
        for smooth in [false, true] {
            let mut c = ComplexPlane::new(w, h);
            c.set_view(&view);
            let mut m = MandelbrotSet::new(max_iterations, 2.0);
            m.palette_scale = palette_scale;
            m.smooth_coloring = smooth;
            let mut p = PixelBuffer::new(PixelPlane::new(w, h));
            rendering::render_progressively(&c, &m, w, h, 1, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &AtomicUsize::new(0), |colors, _| p.colors.copy_from_slice(colors));
            p.update_pixels();
            p.save_as_png(&format!("compare-{name}-{}", if smooth { "smooth" } else { "integer" }), &view, &m, 1);
        }
    }
}

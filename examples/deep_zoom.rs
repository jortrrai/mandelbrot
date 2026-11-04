//! Renders a view deeper than 64-bit floats can show to a PNG file, with the deep zoom prototype (perturbation with rebasing, see `rendering::deep`).
//!
//! ```text
//! cargo run --release --example deep_zoom -- <real> <imaginary> <scale> <width> <height> <max iterations> <file.png> [iteration divide]
//! cargo run --release --example deep_zoom -- 0.0000000000000000000000000000000000000000003 1.0000000000000000000000000000000000000000002 1e-40 960 640 5000 deep.png
//! ```
//! The center is decimal text with as many digits as the depth needs; the view is `2.5 · scale` wide. The colors are those of the window's defaults
//! (Bernstein polynomials, the default color cycle, or with the iteration divide given).

use std::io::BufWriter;

use mandelbrot::coloring::TrueColor;
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::rendering::deep;

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (Some([real, imaginary, scale, width, height, iterations, file]), divide) = (arguments.get(..7).and_then(|first| <&[String; 7]>::try_from(first).ok()), arguments.get(7)) else {
        eprintln!("usage: deep_zoom <real> <imaginary> <scale> <width> <height> <max iterations> <file.png> [iteration divide]");
        std::process::exit(2);
    };
    let (scale, width, height, iterations): (f64, usize, usize, u32) = (scale.parse().expect("a scale"), width.parse().expect("a width"), height.parse().expect("a height"), iterations.parse().expect("iterations"));
    let mut m = MandelbrotSet::new(iterations, 2.0);
    if let Some(divide) = divide {
        m.palette.divide = divide.parse().expect("an iteration divide"); //Below 1: the colors change faster with the iterations
        m.palette.stretch = false; //(A palette that repeats along the iterations, as in the window)
    }
    let started = std::time::Instant::now();
    let positions = deep::render_positions(&m, real, imaginary, scale, width, height).expect("the center is decimal text");
    let rendered = started.elapsed();
    let (lowest, highest) = positions.iter().fold((f64::MAX, f64::MIN), |(low, high), position| (low.min(*position), high.max(*position)));
    println!("smooth iteration counts from {lowest:.3} to {highest:.3} (the limit is {iterations})");
    let mut rgb = Vec::with_capacity(width * height * 3);
    for position in positions {
        let color = m.interpolated_color(position, |count| m.color(TrueColor::new_from_bernstein_polynomials, count));
        rgb.extend([color.red, color.green, color.blue]);
    }
    let mut encoder = png::Encoder::new(BufWriter::new(std::fs::File::create(file).expect("the file can be created")), u32::try_from(width).unwrap(), u32::try_from(height).unwrap());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().and_then(|mut writer| writer.write_image_data(&rgb)).expect("the image can be written");
    println!("{width} x {height} at scale {scale:e} ({} bits) in {:.2} s: {file}", deep::bits_for(scale, width), rendered.as_secs_f64());
}

//! Output baseline of the renderer: renders many representative configurations and prints one checksum per configuration.
//!
//! Use it to check that a change (a refactoring, for example) does not change what the program produces:
//! ```text
//! cargo run --release --example image_baseline | grep '^BASELINE' > before.txt   # on the old code
//! cargo run --release --example image_baseline | grep '^BASELINE' > after.txt    # on the new code
//! diff before.txt after.txt                                                      # no output: nothing changed
//! ```
//! It covers the places of the `points_of_interest` folder (the reset view and ten more) and the default view, with the real palettes (Bernstein, HSV), smooth and integer coloring, the palette of automatic iterations,
//! uniform and adaptive supersampling, the channel mappings, moves that render only the exposed strip, the old non-progressive render path used for saving, a saved PNG file (read back),
//! thumbnails, and the choices of the automatic iteration limit.
//!
//! The values are reproducible on the same machine and build; they are **not** guaranteed to be the same on another CPU or math library (the real palettes and the smooth positions use `powi` and `ln`),
//! so compare runs of the same machine, not runs of different machines. `tests/render_regression.rs` has the checks that are platform independent.
use std::sync::atomic::AtomicBool;

use mandelbrot::auto_iterations::{choose_limit, AutoIterationSettings};
use mandelbrot::coloring::{ColorChannelMapping, TrueColor};
use mandelbrot::complex_plane::{ComplexPlane, View};
use mandelbrot::mandelbrot_set::MandelbrotSet;
use mandelbrot::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
use mandelbrot::points_of_interest::{self, Coloring};
use mandelbrot::rendering::{self, AdaptiveSampling, ExposedRegion, Render};
use mandelbrot::thumbnails::{self, ThumbnailRequest};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;

/// FNV-1a, 64 bits
fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn colors_checksum(colors: &[TrueColor]) -> u64 {
    fnv(colors.iter().flat_map(|color| [color.red, color.green, color.blue]))
}

fn report(name: &str, checksum: u64) {
    println!("BASELINE {name} {checksum:016x}");
}

fn plane(view: &View, width: usize, height: usize) -> ComplexPlane {
    let mut plane = ComplexPlane::new(width, height);
    plane.set_view(view);
    plane
}

/// Renders with every layer and returns the final colors
fn render(plane: &ComplexPlane, set: &MandelbrotSet, samples: u8, adaptive: Option<&AdaptiveSampling>, coloring: fn(f64, f64) -> TrueColor) -> Vec<TrueColor> {
    let _base = vec![TrueColor::new(0, 0, 0); WIDTH * HEIGHT];
    let mut last = Vec::new();
    Render::new(plane, set, WIDTH, HEIGHT, coloring).supersampling(samples).adaptive(adaptive.copied()).run( |colors, _| {
        last.clear();
        last.extend_from_slice(colors);
    });
    last
}

fn main() {
    let report_of_folder = points_of_interest::load_folder(&points_of_interest::find_folder());
    let mut views: Vec<(String, View, u32)> = vec![(String::from("default"), View::new(-0.75, 0.0, 1.0), 1000)];
    views.extend(report_of_folder.points.iter().map(|point| (point.file_name.trim_end_matches(".json").to_string(), View::new(point.location.real, point.location.imaginary, point.location.scale), point.location.max_iterations)));

    //The palettes, coloring modes and supersampling of every view
    for (name, view, max_iterations) in &views {
        let plane = plane(view, WIDTH, HEIGHT);
        let smooth = MandelbrotSet::new(*max_iterations, 2.0);
        let mut integer = smooth.clone();
        integer.smooth_coloring = false;
        let mut scaled = smooth.clone();
        scaled.palette = mandelbrot::mandelbrot_set::ColorCycle { divide: f64::from((max_iterations / 2000).max(1)), offset: 0.0, stretch: false };
        report(&format!("{name} bernstein smooth 1x"), colors_checksum(&render(&plane, &smooth, 1, None, TrueColor::new_from_bernstein_polynomials)));
        report(&format!("{name} hsv smooth 2x"), colors_checksum(&render(&plane, &smooth, 2, None, TrueColor::new_from_hsv_colors)));
        report(&format!("{name} bernstein integer 1x"), colors_checksum(&render(&plane, &integer, 1, None, TrueColor::new_from_bernstein_polynomials)));
        report(&format!("{name} bernstein smooth color-cycle 1x"), colors_checksum(&render(&plane, &scaled, 1, None, TrueColor::new_from_bernstein_polynomials)));
        report(&format!("{name} bernstein smooth adaptive 4x"), colors_checksum(&render(&plane, &smooth, 4, Some(&AdaptiveSampling::default()), TrueColor::new_from_bernstein_polynomials)));
        report(&format!("{name} bernstein smooth uniform 4x"), colors_checksum(&render(&plane, &smooth, 4, None, TrueColor::new_from_bernstein_polynomials)));
    }

    //The color channel mappings (what the window draws)
    let (_, default_view, _) = &views[0];
    let default_colors = render(&plane(default_view, WIDTH, HEIGHT), &MandelbrotSet::new(1000, 2.0), 1, None, TrueColor::new_from_bernstein_polynomials);
    for mapping in ["RGB", "BGR", "GBR", "RRG", "BBB"] {
        let pixels = PixelBuffer::colors_to_pixels(&default_colors, &ColorChannelMapping::new(mapping).expect("a valid mapping"));
        report(&format!("channel mapping {mapping}"), fnv(pixels.iter().flat_map(|pixel| pixel.to_le_bytes())));
    }

    //Moves: shift the buffer and render only the exposed region (a strip, and an L shape)
    let (_, seahorse, seahorse_iterations) = views.iter().find(|(name, _, _)| name.contains("seahorse")).expect("the seahorse place exists");
    for (name, rows_up, columns_right) in [("strip up", 9, 0), ("strip left", 0, -13), ("L shape", 7, 11)] {
        let mut plane = plane(seahorse, WIDTH, HEIGHT);
        let set = MandelbrotSet::new(*seahorse_iterations, 2.0);
        let full = render(&plane, &set, 2, None, TrueColor::new_from_bernstein_polynomials);
        let mut buffer = PixelBuffer::new(PixelPlane::new(WIDTH, HEIGHT));
        buffer.colors.copy_from_slice(&full);
        buffer.translate_buffer(rows_up, columns_right);
        rendering::move_complex_plane_by_pixels(&mut plane, rows_up as i64, -(columns_right as i64));
        let region = ExposedRegion::from_shift(WIDTH, HEIGHT, rows_up, columns_right).expect("a small shift");
        let mut last = Vec::new();
        Render::new(&plane, &set, WIDTH, HEIGHT, TrueColor::new_from_bernstein_polynomials).supersampling(2).only(&region, &buffer.colors).run( |colors, _| {
            last.clear();
            last.extend_from_slice(colors);
        });
        report(&format!("move {name}"), colors_checksum(&last));
    }


    //The old non-progressive render path (used when saving at another scale), with its progress printed to the terminal
    let mut old_buffer = PixelBuffer::new(PixelPlane::new(64, 48));
    rendering::render_complex_plane_into_buffer(&mut old_buffer, &plane(seahorse, 64, 48), &MandelbrotSet::new(*seahorse_iterations, 2.0), 2, TrueColor::new_from_bernstein_polynomials);
    report("render_complex_plane_into_buffer 2x", colors_checksum(&old_buffer.colors));
    report("render_complex_plane_into_buffer pixels", fnv(old_buffer.pixels.iter().flat_map(|pixel| pixel.to_le_bytes())));

    //A saved PNG file, read back: the raster and the text chunks
    let directory = std::env::temp_dir().join(format!("mandelbrot-baseline-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("a temporary directory");
    std::env::set_current_dir(&directory).expect("the temporary directory");
    let mut png_buffer = PixelBuffer::new(PixelPlane::new(WIDTH, HEIGHT));
    png_buffer.colors.copy_from_slice(&default_colors);
    png_buffer.update_pixels();
    png_buffer.save_as_png("baseline", default_view, &MandelbrotSet::new(1000, 2.0), 1, None).expect("the baseline image can be saved");
    let decoder = png::Decoder::new(std::fs::File::open(directory.join("saved").join("baseline.png")).expect("the saved file"));
    let mut reader = decoder.read_info().expect("a valid PNG");
    let mut raster = vec![0; reader.output_buffer_size()];
    reader.next_frame(&mut raster).expect("the raster");
    let mut texts: Vec<String> = reader.info().uncompressed_latin1_text.iter().map(|chunk| format!("{}={}", chunk.keyword, chunk.text)).collect();
    texts.sort();
    report("png raster", fnv(raster));
    report("png text chunks", fnv(texts.join("\n").bytes()));
    let _ = std::fs::remove_dir_all(&directory);

    //Thumbnails of the places, as the window renders them
    for point in &report_of_folder.points {
        let request = ThumbnailRequest { location: point.location.clone(), coloring: Coloring::Bernstein, color_channels: ColorChannelMapping::RGB, palette: mandelbrot::mandelbrot_set::ColorCycle::CYCLE, width: thumbnails::THUMBNAIL_WIDTH, height: thumbnails::thumbnail_height(1200, 800) };
        let colors = thumbnails::render_thumbnail(&request, 2.0, &AtomicBool::new(false)).expect("not cancelled");
        report(&format!("thumbnail {}", point.file_name.trim_end_matches(".json")), colors_checksum(&colors));
        report(&format!("thumbnail key {}", point.file_name.trim_end_matches(".json")), request.key());
    }

    //The choices of the automatic iteration limit with every preset
    let presets = [("performance", mandelbrot::auto_iterations::AutoPreset::Performance), ("balanced", mandelbrot::auto_iterations::AutoPreset::Balanced), ("high detail", mandelbrot::auto_iterations::AutoPreset::HighDetail)];
    let _ = AutoIterationSettings::default();
    for (name, view, _) in views.iter().take(5) {
        for (preset_name, preset) in presets {
            let settings = preset.settings();
            let plane = plane(view, 160, 100);
            let result = choose_limit(&plane, &MandelbrotSet::new(1000, 2.0), 160, 100, &settings, 3, &AtomicBool::new(false)).expect("not cancelled");
            let text = format!("{} {} {} {:?} {:?}", result.limit, result.palette_scale, result.steps, result.stop, result.rounds.iter().map(|round| (round.limit, round.escaped, round.certified, round.unresolved)).collect::<Vec<_>>());
            report(&format!("auto iterations {name} {preset_name}"), fnv(text.bytes()));
        }
    }
}

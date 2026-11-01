//! Saving an image of any size up to 32K with bounded memory: it is rendered in strips (see `rendering::strips`), blurred strip by strip for a blurred quality,
//! and written to the PNG file row by row on a thread of its own while the next strip renders. At most two strips (and the rows the blur needs from the strips
//! next to them) are in memory, never the whole image.
//!
//! The file is written as `<name>.png.part` and only gets its name when it is complete, so a cancelled or failed save leaves nothing behind.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use crate::coloring::TrueColor;
use crate::complex_plane::ComplexPlane;
use crate::mandelbrot_set::MandelbrotSet;
use crate::pixel_buffer::{PngMetadata, PngRows};
use crate::rendering::gpu::GpuIterator;
use crate::rendering::strips::{render_in_strips, StripProgress};
use crate::resolution::Resolution;

/// A save of an image that is running, for its progress bar and its Cancel button
pub(super) struct ImageSave {
    pub(super) size: Resolution,
    /// The passes of every strip (samples per pixel)
    pub(super) passes: usize,
    pub(super) progress: Arc<StripProgress>,
    pub(super) cancel: Arc<AtomicBool>,
    /// Set by the thread of the save when it has ended (saved, cancelled or failed; its message is in the status log)
    pub(super) ended: Arc<AtomicBool>,
    pub(super) started: Instant,
}

impl ImageSave {
    /// The part that is done, from 0 to 1
    pub(super) fn fraction(&self) -> f64 {
        self.progress.fraction(self.size.width, self.size.height, self.passes)
    }

    /// The text of the progress bar: the size, the percentage and, after a few seconds, the time that is left
    pub(super) fn text(&self) -> String {
        let (fraction, elapsed) = (self.fraction(), self.started.elapsed().as_secs_f64());
        let left = if fraction > 0.01 && elapsed > 3.0 { format!(", about {} left", super::video::length_text(std::time::Duration::from_secs_f64(elapsed * (1.0 - fraction) / fraction))) } else { String::new() };
        format!("Saving {}: {:.0}%{left}", self.size, fraction * 100.0)
    }
}

/// The file that `path` is written to until it is complete
pub(super) fn part_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// How to render an image that is saved
pub(super) struct Picture<'a> {
    /// The plane of the whole image (its size is `size`)
    pub(super) plane: &'a ComplexPlane,
    pub(super) size: Resolution,
    pub(super) m: &'a MandelbrotSet,
    pub(super) samples: u8,
    pub(super) coloring_function: fn(f64, f64) -> TrueColor,
    pub(super) gpu: Option<&'a GpuIterator>,
    /// The blur of a blurred quality, in pixels, if any (see `blur::blurred`)
    pub(super) blur: Option<f32>,
}

/// Renders `picture` in strips of at most `strip_pixels` pixels and writes it to the PNG file `path` with `metadata`, row by row (see the module). Returns
/// `Ok(false)` if it was cancelled by `cancel`: then no file is left.
/// # Errors
/// The file cannot be written; then no file is left either
pub(super) fn save_png(path: &Path, picture: &Picture, metadata: &PngMetadata, strip_pixels: usize, cancel: &AtomicBool, progress: &StripProgress) -> std::io::Result<bool> {
    let part = part_path(path);
    let saved = write_png(&part, picture, metadata, strip_pixels, cancel, progress);
    match saved {
        Ok(true) => std::fs::rename(&part, path).map(|()| true),
        Ok(false) | Err(_) => {
            let _ = std::fs::remove_file(&part);
            saved
        }
    }
}

/// `save_png` without the renaming
fn write_png(path: &Path, picture: &Picture, metadata: &PngMetadata, strip_pixels: usize, cancel: &AtomicBool, progress: &StripProgress) -> std::io::Result<bool> {
    let Resolution { width, height } = picture.size;
    let mut png = PngRows::create(path, width, height, metadata)?;
    let mapping = metadata.color_channel_mapping;
    //One strip waits for the writer while the next renders
    let (rows_sender, rows) = mpsc::sync_channel::<Vec<TrueColor>>(1);
    std::thread::scope(|scope| {
        let writer = scope.spawn(move || -> std::io::Result<PngRows> {
            for colors in rows {
                png.write_rows(&colors, mapping)?;
            }
            Ok(png)
        });
        let radius = picture.blur.map_or(0, super::blur::blur_radius);
        //The rows that are rendered and not yet all written: from `window_start`. A row is written when the rows the blur needs below it are there too
        let (mut window, mut window_start, mut next_row) = (Vec::new(), 0, 0);
        let rendered = render_in_strips(picture.plane, picture.m, (width, height), picture.samples, picture.coloring_function, picture.gpu, strip_pixels, cancel, progress, |rows, colors| {
            window.extend(colors);
            let window_end = rows.end;
            let out_end = if window_end == height { height } else { window_end.saturating_sub(radius) };
            if out_end <= next_row {
                return Ok(());
            }
            let window_rows = window_end - window_start;
            //Within the window the blur of a written row only reaches rows of the window, or the edge of the image, where it repeats the edge like the whole image
            let blurred = picture.blur.map(|sigma| super::blur::blurred(&window, width, window_rows, sigma));
            let out = blurred.as_ref().unwrap_or(&window)[(next_row - window_start) * width..(out_end - window_start) * width].to_vec();
            rows_sender.send(out).map_err(|_| ())?;
            next_row = out_end;
            let keep_from = next_row.saturating_sub(radius).max(window_start);
            window.drain(..(keep_from - window_start) * width);
            window_start = keep_from;
            Ok(())
        });
        drop(rows_sender);
        let png = writer.join().unwrap_or_else(|_| Err(std::io::Error::other("the writer of the image stopped")));
        match (rendered, png) {
            (_, Err(err)) => Err(err),
            (Ok(true), Ok(png)) => png.finish().map(|()| true),
            (Ok(false) | Err(()), Ok(_)) => Ok(false),
        }
    })
    .map(|completed| completed && !cancel.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coloring::ColorChannelMapping;
    use crate::complex_plane::View;
    use crate::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};

    fn seahorse(width: usize, height: usize) -> (ComplexPlane, MandelbrotSet) {
        let mut c = ComplexPlane::new(width, height);
        c.set_view(&View::new(-0.7498, 0.0203, 6.4e-5));
        (c, MandelbrotSet::new(1500, 2.0))
    }

    fn folder(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("mandelbrot-export-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    /// The RGB bytes of the PNG at `path`
    fn decoded(path: &Path) -> Vec<u8> {
        let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap())).read_info().unwrap();
        let mut bytes = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut bytes).unwrap();
        bytes.truncate(info.buffer_size());
        bytes
    }

    #[test]
    fn an_image_saved_in_strips_is_the_whole_image_with_its_blur_and_its_view() {
        let (width, height) = (53, 41);
        let (c, m) = seahorse(width, height);
        let coloring = TrueColor::new_from_bernstein_polynomials;
        let mapping = ColorChannelMapping::new("BRG").unwrap();
        let view = c.get_view();
        let point = crate::points_of_interest::PointOfInterest {
            name: String::from("Seahorse"),
            description: String::new(),
            location: crate::points_of_interest::Location { real: -0.7498, imaginary: 0.0203, scale: 6.4e-5, max_iterations: 1500 },
            appearance: None,
            file_name: String::new(),
        };
        let place = crate::points_of_interest::to_json(&point);
        let metadata = PngMetadata { view: &view, m: &m, supersampling_amount: 4, color_channel_mapping: mapping, place: Some(&place) };
        let folder = folder("seams");
        for blur in [None, Some(super::super::blur::blur_sigma(1, 4)), Some(2.4)] {
            //The whole image, rendered and blurred in one go, as before
            let mut whole = PixelBuffer::new(PixelPlane::new(width, height));
            crate::rendering::render_complex_plane_into_buffer(&mut whole, &c, &m, 4, coloring);
            if let Some(sigma) = blur {
                whole.colors = super::super::blur::blurred(&whole.colors, width, height, sigma);
            }
            let expected = crate::pixel_buffer::rgb_bytes(&whole.colors, mapping);
            //Strips of 3 rows (fewer than the blur reaches) and of 10 rows
            for strip_rows in [3, 10, height] {
                let path = folder.join(format!("strips-{strip_rows}.png"));
                let picture = Picture { plane: &c, size: Resolution::new(width, height), m: &m, samples: 4, coloring_function: coloring, gpu: None, blur };
                let (cancel, progress) = (AtomicBool::new(false), StripProgress::default());
                assert!(save_png(&path, &picture, &metadata, strip_rows * width, &cancel, &progress).unwrap());
                assert!(decoded(&path) == expected, "blur {blur:?} in strips of {strip_rows} rows: the same pixels as the whole image");
                assert!(!part_path(&path).exists());
                let opened = crate::image_metadata::view_of_png(&path).unwrap();
                assert_eq!(opened.location, point.location, "the view is in the file");
            }
        }
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_cancelled_or_failed_save_leaves_no_file() {
        let (width, height) = (40, 30);
        let (c, m) = seahorse(width, height);
        let view = c.get_view();
        let metadata = PngMetadata { view: &view, m: &m, supersampling_amount: 1, color_channel_mapping: ColorChannelMapping::RGB, place: None };
        let picture = Picture { plane: &c, size: Resolution::new(width, height), m: &m, samples: 1, coloring_function: TrueColor::new_from_bernstein_polynomials, gpu: None, blur: None };
        let folder = folder("cancel");
        let path = folder.join("cancelled.png");
        let (cancel, progress) = (AtomicBool::new(true), StripProgress::default());
        assert!(!save_png(&path, &picture, &metadata, 4 * width, &cancel, &progress).unwrap());
        assert!(!path.exists() && !part_path(&path).exists(), "nothing is left of a cancelled save");
        assert!(save_png(&folder.join("no-such-folder").join("x.png"), &picture, &metadata, 4 * width, &AtomicBool::new(false), &progress).is_err());
        std::fs::remove_dir_all(&folder).unwrap();
    }

    /// The peak memory of this process in bytes (`VmHWM`, Linux only)
    fn peak_memory() -> Option<usize> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
        line.split_whitespace().nth(1)?.parse::<usize>().ok().map(|kilobytes| kilobytes * 1024)
    }

    #[test]
    #[ignore = "a measurement: renders a 32K image (MANDELBROT_EXPORT=\"width height samples\" for another), run by hand with --release"]
    fn a_32k_image_is_saved_with_bounded_memory() {
        let setting = std::env::var("MANDELBROT_EXPORT").unwrap_or_else(|_| String::from("30720 17280 1"));
        let numbers: Vec<usize> = setting.split_whitespace().map(|number| number.parse().unwrap()).collect();
        let (size, samples) = (Resolution::new(numbers[0], numbers[1]), u8::try_from(numbers[2]).unwrap());
        let mut c = ComplexPlane::new(size.width, size.height);
        c.set_view(&View::new(-0.7498, 0.0203, 6.4e-5));
        let m = MandelbrotSet::new(1500, 2.0);
        let view = c.get_view();
        let metadata = PngMetadata { view: &view, m: &m, supersampling_amount: samples, color_channel_mapping: ColorChannelMapping::RGB, place: None };
        let picture = Picture { plane: &c, size, m: &m, samples, coloring_function: TrueColor::new_from_bernstein_polynomials, gpu: None, blur: None };
        let folder = folder("measure");
        let path = folder.join("large.png");
        let started = Instant::now();
        let (cancel, progress) = (AtomicBool::new(false), StripProgress::default());
        assert!(save_png(&path, &picture, &metadata, crate::rendering::strips::STRIP_PIXELS, &cancel, &progress).unwrap());
        let (bytes, peak) = (std::fs::metadata(&path).unwrap().len(), peak_memory().unwrap_or(0));
        println!("{size} at {samples}x: {:.1} s, file {:.0} MB, peak memory {:.0} MB (the whole image would be {:.0} MB of colors alone)", started.elapsed().as_secs_f64(), bytes as f64 / 1e6, peak as f64 / 1e6, size.pixels() as f64 * 3.0 / 1e6);
        std::fs::remove_dir_all(&folder).unwrap();
        assert!(peak < 1_000_000_000, "the memory stays bounded");
    }
}

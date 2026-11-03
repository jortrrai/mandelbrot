//! What Fast dives cost in quality (issue #271): a flight rendered frame by frame and with Fast dives (see `dive_keyframes`), the same frames both ways as
//! lossless PNG frames, compared frame by frame.
//!
//! `fast_dives_against_frame_by_frame_in_quality` (run by hand) writes to `MANDELBROT_MEASURE_DIR` (default `target/measure-271`):
//! * `report.md`: the settings, the render times of both, and PSNR and SSIM over all frames (mean, worst, median, the 5th percentile), the worst frames;
//! * `frames.csv`: PSNR and SSIM of every frame;
//! * `worst-*.png`, `median-*.png`: the frame by frame render, the Fast dives render and their difference (×8) side by side, and a 400% crop of each around
//!   where they differ most.
//!
//! PSNR is over the RGB channels against the frame by frame render (8 bits, peak 255); SSIM is the mean of 8 x 8 windows of the luma (BT.709), with the usual
//! constants. With the fast GPU math, frame by frame is not the same twice either (two runs of a 4K flight measured 61.8 dB against each other, see #254):
//! `MANDELBROT_NOISE=1` also renders it twice, for that floor.
//!
//! Frame by frame at the same samples is itself noisy where the image is detailed (a few samples a pixel), and Fast dives averages 1 to 4 keyframe pixels
//! per pixel, so it is smoother: against it, Fast dives can look worse than it is. `MANDELBROT_REFERENCE_SAMPLES=64` renders the flight frame by frame
//! at that many samples as well, and the report compares both, frame by frame and Fast dives, against that better reference.

/// The PSNR of `test` against `reference` (RGB bytes), in dB; infinite when they are the same
pub(super) fn psnr(reference: &[u8], test: &[u8]) -> f64 {
    let squared: f64 = reference.iter().zip(test).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum();
    let mse = squared / reference.len().max(1) as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

/// The luma (BT.709) of RGB bytes
fn luma(rgb: &[u8]) -> Vec<f64> {
    rgb.as_chunks::<3>().0.iter().map(|pixel| 0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])).collect()
}

/// The SSIM of `test` against `reference` (RGB bytes, `width` x `height`): the mean over 8 x 8 windows of the luma (1: the same)
pub(super) fn ssim(reference: &[u8], test: &[u8], width: usize, height: usize) -> f64 {
    let (a, b) = (luma(reference), luma(test));
    let (c1, c2) = ((0.01_f64 * 255.0).powi(2), (0.03_f64 * 255.0).powi(2));
    let (mut total, mut windows) = (0.0, 0);
    for top in (0..height.saturating_sub(7)).step_by(8) {
        for left in (0..width.saturating_sub(7)).step_by(8) {
            let at = |image: &[f64], x: usize, y: usize| image[(top + y) * width + left + x];
            let (mut mean_a, mut mean_b) = (0.0, 0.0);
            for y in 0..8 {
                for x in 0..8 {
                    (mean_a, mean_b) = (mean_a + at(&a, x, y), mean_b + at(&b, x, y));
                }
            }
            (mean_a, mean_b) = (mean_a / 64.0, mean_b / 64.0);
            let (mut var_a, mut var_b, mut covariance) = (0.0, 0.0, 0.0);
            for y in 0..8 {
                for x in 0..8 {
                    let (da, db) = (at(&a, x, y) - mean_a, at(&b, x, y) - mean_b);
                    (var_a, var_b, covariance) = (var_a + da * da, var_b + db * db, covariance + da * db);
                }
            }
            (var_a, var_b, covariance) = (var_a / 63.0, var_b / 63.0, covariance / 63.0);
            total += ((2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)) / ((mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2));
            windows += 1;
        }
    }
    if windows == 0 { 1.0 } else { total / f64::from(windows) }
}

/// The top left corner of the `size` x `size` square (clamped to the image) where `test` differs most from `reference`
fn most_different(reference: &[u8], test: &[u8], (width, height): (usize, usize), size: usize) -> (usize, usize) {
    let block = size.max(1);
    let mut best = ((0, 0), -1.0);
    for top in (0..height).step_by(block / 2 + 1) {
        for left in (0..width).step_by(block / 2 + 1) {
            let (top, left) = (top.min(height.saturating_sub(block)), left.min(width.saturating_sub(block)));
            let mut error = 0.0;
            for y in top..(top + block).min(height) {
                let row = (y * width + left) * 3..(y * width + (left + block).min(width)) * 3;
                error += reference[row.clone()].iter().zip(&test[row]).map(|(a, b)| f64::from(a.abs_diff(*b))).sum::<f64>();
            }
            if error > best.1 {
                best = ((left, top), error);
            }
        }
    }
    best.0
}

/// The frame by frame render, the Fast dives render and their difference (×8, clamped), side by side: an RGB image `3 · width` x `height`
fn side_by_side(reference: &[u8], test: &[u8], (width, height): (usize, usize)) -> Vec<u8> {
    let mut out = Vec::with_capacity(reference.len() * 3);
    for y in 0..height {
        let row = y * width * 3..(y + 1) * width * 3;
        out.extend_from_slice(&reference[row.clone()]);
        out.extend_from_slice(&test[row.clone()]);
        out.extend(reference[row.clone()].iter().zip(&test[row]).map(|(a, b)| a.abs_diff(*b).saturating_mul(8)));
    }
    out
}

/// The `size` x `size` square at `corner` of an RGB image `width` wide, each pixel `zoom` x `zoom` (nearest neighbor, so the pixels show)
fn crop(rgb: &[u8], width: usize, (left, top): (usize, usize), size: usize, zoom: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * zoom * zoom * 3);
    for y in 0..size * zoom {
        for x in 0..size * zoom {
            let at = ((top + y / zoom) * width + left + x / zoom) * 3;
            out.extend_from_slice(&rgb[at..at + 3]);
        }
    }
    out
}

fn write_rgb(path: &std::path::Path, rgb: &[u8], width: usize, height: usize) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut encoder = png::Encoder::new(file, u32::try_from(width).unwrap(), u32::try_from(height).unwrap());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().unwrap().write_image_data(rgb).unwrap();
}

fn read_rgb(path: &std::path::Path) -> (Vec<u8>, usize, usize) {
    let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap())).read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).unwrap();
    pixels.truncate(info.buffer_size());
    (pixels, info.width as usize, info.height as usize)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn the_metrics_see_the_same_image_as_perfect_and_noise_as_worse() {
        let (width, height) = (32, 16);
        let image: Vec<u8> = (0..width * height * 3).map(|index| (index * 37 % 251) as u8).collect();
        assert!(psnr(&image, &image).is_infinite());
        assert!((ssim(&image, &image, width, height) - 1.0).abs() < 1e-12);
        let slightly: Vec<u8> = image.iter().enumerate().map(|(index, value)| if index % 7 == 0 { value.saturating_add(4) } else { *value }).collect();
        let badly: Vec<u8> = image.iter().enumerate().map(|(index, value)| if index % 2 == 0 { value.saturating_add(60) } else { *value }).collect();
        assert!(psnr(&image, &slightly) > psnr(&image, &badly) && psnr(&image, &badly) > 0.0);
        assert!(ssim(&image, &slightly, width, height) > ssim(&image, &badly, width, height));
        //(Every 7th byte off by 4: an MSE of 16/7)
        assert!((psnr(&image, &slightly) - 10.0 * (255.0_f64 * 255.0 / (16.0 / 7.0)).log10()).abs() < 0.2);
        //The square that differs most is found
        let mut spot = image.clone();
        for y in 4..8 {
            for x in 20..24 {
                spot[(y * width + x) * 3] ^= 0xff;
            }
        }
        let (left, top) = most_different(&image, &spot, (width, height), 4);
        //(On a grid of half the square: a square that overlaps the spot)
        assert!(left + 4 > 20 && left < 24 && top + 4 > 4 && top < 8, "({left}, {top})");
    }

    /// The PNG frames of the video of `flight` rendered with Fast dives on or off, and how long it took
    fn render(flight: &super::super::flight_files::SavedFlight, (width, height, samples): (usize, usize, u8), keyframe_dives: bool, settings: Option<&str>) -> (Vec<std::path::PathBuf>, f64, std::path::PathBuf) {
        let folder = std::env::temp_dir().join(format!("mandelbrot-271-{}-{keyframe_dives}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let mut app = super::super::MandelbrotApp::new(&crate::Config::sized(64, 48, 50, 1));
        if let Some(settings) = settings {
            app.use_settings_file(Some(std::path::PathBuf::from(settings)));
        }
        app.gpu = crate::rendering::gpu::GpuIterator::new().ok().map(Arc::new);
        app.video_folder = folder.clone();
        app.save_size.choice = super::super::save_size::SaveSizeChoice::Custom;
        (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount) = (width, height, samples);
        (app.video_keyframe_dives, app.video_pngs) = (keyframe_dives, true);
        app.saved_flights = vec![flight.clone()];
        app.use_saved_flight(0);
        let started = std::time::Instant::now();
        app.start_video();
        let ctx = eframe::egui::Context::default();
        while app.video_job.is_some() {
            app.poll_video(&ctx);
            std::thread::sleep(Duration::from_millis(5));
        }
        let seconds = started.elapsed().as_secs_f64();
        let frames_folder = std::fs::read_dir(&folder).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.is_dir()).expect("the PNG frames");
        let mut frames: Vec<std::path::PathBuf> = std::fs::read_dir(&frames_folder).unwrap().filter_map(Result::ok).map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|extension| extension == "png")).collect();
        frames.sort();
        (frames, seconds, folder)
    }

    /// Issue #271: a saved flight frame by frame and with Fast dives, compared frame by frame, see the module. `MANDELBROT_FLIGHT`: the flight file;
    /// `MANDELBROT_SIZE`="width height samples" (default 1280 720 8); `MANDELBROT_SETTINGS`: a copy of a settings file (its quality, GPU and math are used)
    #[test]
    #[ignore = "a measurement of what Fast dives cost in quality (#271), run by hand with --release --nocapture"]
    fn fast_dives_against_frame_by_frame_in_quality() {
        let Ok(path) = std::env::var("MANDELBROT_FLIGHT") else { return };
        let size = std::env::var("MANDELBROT_SIZE").unwrap_or_else(|_| String::from("1280 720 8"));
        let numbers: Vec<usize> = size.split_whitespace().map(|number| number.parse().unwrap()).collect();
        let (width, height, samples) = (numbers[0], numbers[1], u8::try_from(numbers[2]).unwrap());
        let settings = std::env::var("MANDELBROT_SETTINGS").ok();
        let out = std::path::PathBuf::from(std::env::var("MANDELBROT_MEASURE_DIR").unwrap_or_else(|_| String::from("target/measure-271")));
        std::fs::create_dir_all(&out).unwrap();
        let flight = super::super::flight_files::parse(&std::fs::read_to_string(&path).unwrap(), "flight.json").unwrap();
        let (reference, reference_seconds, reference_folder) = render(&flight, (width, height, samples), false, settings.as_deref());
        let (fast, fast_seconds, fast_folder) = render(&flight, (width, height, samples), true, settings.as_deref());
        assert_eq!(reference.len(), fast.len(), "the same frames both ways");
        //Every frame: PSNR and SSIM against frame by frame
        let mut rows: Vec<(usize, f64, f64)> = Vec::with_capacity(fast.len());
        for (index, (a, b)) in reference.iter().zip(&fast).enumerate() {
            let ((a, w, h), (b, _, _)) = (read_rgb(a), read_rgb(b));
            rows.push((index, psnr(&a, &b), ssim(&a, &b, w, h)));
        }
        let noise = std::env::var("MANDELBROT_NOISE").is_ok().then(|| {
            let (again, _, folder) = render(&flight, (width, height, samples), false, settings.as_deref());
            let values: Vec<f64> = reference.iter().zip(&again).map(|(a, b)| psnr(&read_rgb(a).0, &read_rgb(b).0)).filter(|value| value.is_finite()).collect();
            let _ = std::fs::remove_dir_all(folder);
            (values.iter().sum::<f64>() / values.len().max(1) as f64, again.len() - values.len())
        });
        let csv: String = std::iter::once(String::from("frame,psnr_db,ssim\n")).chain(rows.iter().map(|(index, value, structure)| format!("{},{value:.3},{structure:.5}\n", index + 1))).collect();
        std::fs::write(out.join("frames.csv"), csv).unwrap();
        //(Identical frames, the 64-bit part before the first keyframe say, count as 100 dB in the averages)
        let finite = |value: f64| if value.is_finite() { value } else { 100.0 };
        let mut by_psnr = rows.clone();
        by_psnr.sort_by(|a, b| finite(a.1).total_cmp(&finite(b.1)));
        let mean = |values: &mut dyn Iterator<Item = f64>| {
            let values: Vec<f64> = values.collect();
            values.iter().sum::<f64>() / values.len().max(1) as f64
        };
        let (mean_psnr, mean_ssim) = (mean(&mut rows.iter().map(|row| finite(row.1))), mean(&mut rows.iter().map(|row| row.2)));
        let percentile = |share: f64| by_psnr[((by_psnr.len() - 1) as f64 * share).round() as usize];
        let (worst, median, fifth) = (by_psnr[0], percentile(0.5), percentile(0.05));
        let identical = rows.iter().filter(|row| row.1.is_infinite()).count();
        //The worst and the median frame side by side, and a 400% crop around where they differ most
        for (label, (index, _, _)) in [("worst", worst), ("median", median)] {
            let ((a, w, h), (b, _, _)) = (read_rgb(&reference[index]), read_rgb(&fast[index]));
            write_rgb(&out.join(format!("{label}-frame-{}.png", index + 1)), &side_by_side(&a, &b, (w, h)), w * 3, h);
            let square = (w.min(h) / 8).max(8);
            let corner = most_different(&a, &b, (w, h), square);
            let (crop_a, crop_b) = (crop(&a, w, corner, square, 4), crop(&b, w, corner, square, 4));
            write_rgb(&out.join(format!("{label}-frame-{}-crop-400.png", index + 1)), &side_by_side(&crop_a, &crop_b, (square * 4, square * 4)), square * 12, square * 4);
        }
        let worst_list: Vec<String> = by_psnr.iter().take(5).map(|(index, value, structure)| format!("frame {} ({value:.1} dB, SSIM {structure:.4})", index + 1)).collect();
        //Against a better reference: frame by frame at more samples
        let better = std::env::var("MANDELBROT_REFERENCE_SAMPLES").ok().and_then(|samples| samples.parse::<u8>().ok()).map(|reference_samples| {
            let (truth, truth_seconds, truth_folder) = render(&flight, (width, height, reference_samples), false, settings.as_deref());
            let against = |frames: &[std::path::PathBuf]| {
                let values: Vec<(f64, f64)> = truth.iter().zip(frames).map(|(a, b)| {
                    let ((a, w, h), (b, _, _)) = (read_rgb(a), read_rgb(b));
                    (psnr(&a, &b), ssim(&a, &b, w, h))
                }).collect();
                let worst = values.iter().map(|value| value.0).fold(f64::INFINITY, f64::min);
                (values.iter().map(|value| value.0.min(100.0)).sum::<f64>() / values.len().max(1) as f64, values.iter().map(|value| value.1).sum::<f64>() / values.len().max(1) as f64, worst)
            };
            let (one_by_one, fast) = (against(&reference), against(&fast));
            let _ = std::fs::remove_dir_all(truth_folder);
            format!("Against frame by frame at **{reference_samples} samples** ({truth_seconds:.1} s), a better reference than either:\n\n| {samples} samples | mean PSNR | mean SSIM | worst frame |\n|---|---|---|---|\n\
                | frame by frame | {:.1} dB | {:.4} | {:.1} dB |\n| Fast dives | {:.1} dB | {:.4} | {:.1} dB |\n\n", one_by_one.0, one_by_one.1, one_by_one.2, fast.0, fast.1, fast.2)
        });
        let report = format!(
            "# Fast dives against frame by frame (#271)\n\nFlight `{path}`, {} frames at {width} x {height}, {samples} samples a pixel{}.\n\n\
             | | time | frames per second |\n|---|---|---|\n| frame by frame | {reference_seconds:.1} s | {:.1} |\n| Fast dives | {fast_seconds:.1} s | {:.1} |\n\n\
             Speed-up: **{:.2}x**.\n\nAgainst frame by frame (PSNR over RGB; SSIM of the luma in 8 x 8 windows; identical frames count as 100 dB in the mean):\n\n\
             | | PSNR | SSIM |\n|---|---|---|\n| mean | {mean_psnr:.1} dB | {mean_ssim:.4} |\n| median frame | {:.1} dB | {:.4} |\n| 5th percentile | {:.1} dB | {:.4} |\n| worst frame | {:.1} dB | {:.4} |\n\n\
             {identical} frames are identical. Worst: {}.\n\n{}{}Images: `worst-*.png` and `median-*.png` (frame by frame | Fast dives | difference ×8), with 400% crops where they differ most.\n",
            fast.len(), settings.as_ref().map_or_else(String::new, |settings| format!(", settings `{settings}`")),
            reference.len() as f64 / reference_seconds, fast.len() as f64 / fast_seconds, reference_seconds / fast_seconds,
            finite(median.1), median.2, finite(fifth.1), fifth.2, finite(worst.1), worst.2, worst_list.join(", "),
            noise.map_or_else(String::new, |(value, same)| format!("Run-to-run noise of frame by frame itself (rendered twice): {value:.1} dB mean over the frames that differ ({same} identical).\n\n")),
            better.unwrap_or_default());
        std::fs::write(out.join("report.md"), &report).unwrap();
        println!("{report}");
        let _ = (std::fs::remove_dir_all(reference_folder), std::fs::remove_dir_all(fast_folder));
    }
}

//! Blending the frames of a flight over time (issue #132): every full-resolution frame of a flight takes another sample of its pixels (a new jitter, see
//! `Render::colors_only_of_sample`), and is blended into the image of the frames before, moved and scaled to its view. The flight path is known exactly and the
//! set does not move, so the image converges towards a supersampled one while the camera moves. Measured on flights at 480 x 320 against 16 samples per pixel:
//! 22% (Frost on the Shoreline) to 26% (Seahorse Embers) less error per frame with the first version, which read the image before at the nearest pixel; reading
//! it bilinearly within the range of the new frame (see `blend`) halves the error of a single frame on a zoom, and costs about 10 ms of a 4K frame (all threads).
//!
//! A pixel that the image before does not cover (the edges a zoom out or a pan uncover) takes the new frame alone. The blend starts again with every flight.

use crate::coloring::TrueColor;
use crate::complex_plane::ComplexPlane;
use crate::rendering::render_thread_count;

/// The least weight of a new frame: the image follows the view within about 1 / `MIN_WEIGHT` frames (measured in #118: 0.1 is the best of 0.1 to 0.5)
const MIN_WEIGHT: f32 = 0.1;

/// The image of the frames so far: its view, its colors (the 8-bit colors that are shown), how many frames it has, and the memory of the next blend (kept, so a
/// 4K frame does not allocate and fault in its memory every time). The blend reads and writes the colors as bytes, in integers and whole rows where it can, which
/// the compiler turns into vector instructions: it is limited by memory (floats made it two and a half times as slow); the rounding of a frame is at most half a step
pub(super) struct History {
    plane: ComplexPlane,
    colors: Vec<TrueColor>,
    frames: u32,
    scratch: Vec<TrueColor>,
}

/// The weight of frame `frames + 1` in 256ths: `1 / (frames + 1)` (the plain average while there are few frames), at least `MIN_WEIGHT`
fn weight(frames: u32) -> u32 {
    ((1.0 / (frames as f32 + 1.0)).max(MIN_WEIGHT) * 256.0).round() as u32
}

/// Where the pixels `0..count` of an axis of the view are in the image before (`position = index * scale + offset`): the two nearest pixels there and the
/// weight of the second in 256ths, or `None` outside it
fn axis(count: usize, scale: f64, offset: f64) -> Vec<Option<(usize, usize, u32)>> {
    (0..count).map(|index| {
        let position = index as f64 * scale + offset;
        if position < -1e-9 || position > (count - 1) as f64 + 1e-9 {
            return None;
        }
        let position = position.clamp(0.0, (count - 1) as f64);
        let low = position.floor() as usize;
        Some((low, (low + 1).min(count - 1), ((position - low as f64) * 256.0).round() as u32))
    }).collect()
}

/// The least and the most of every channel of a row of colors (as bytes, 3 a pixel) over every pixel and its left and right neighbors (an edge pixel has one)
fn row_range(row: &[u8], low: &mut [u8], high: &mut [u8]) {
    let n = row.len();
    if n <= 3 {
        low.copy_from_slice(row);
        high.copy_from_slice(row);
        return;
    }
    for ((((low, high), left), middle), right) in low[3..n - 3].iter_mut().zip(&mut high[3..n - 3]).zip(&row[..n - 6]).zip(&row[3..n - 3]).zip(&row[6..]) {
        (*low, *high) = ((*left).min(*middle).min(*right), (*left).max(*middle).max(*right));
    }
    for i in 0..3 {
        let last = n - 3 + i;
        (low[i], high[i]) = (row[i].min(row[i + 3]), row[i].max(row[i + 3]));
        (low[last], high[last]) = (row[last].min(row[last - 3]), row[last].max(row[last - 3]));
    }
}

/// Blends the frame `colors` of view `plane` (`width` x `height` pixels) into `history` and writes the colors to show into `shown`; the first frame starts the history.
///
/// The image before is read where the pixel was, between its four nearest pixels (bilinear: the nearest pixel alone made a zoom duplicate or skip whole rows and
/// columns of it, which swept over the picture like waves), and kept within the colors of the new frame around the pixel (3 x 3, the clamp of temporal
/// anti-aliasing), so it cannot hold on to what the view no longer shows. Measured on flights at 480 x 320 against 16 samples per pixel, against the nearest pixel
/// without the clamp: 28% less error on a zoom into Seahorse Valley, 21% on Frost, 59% on a pan
pub(super) fn blend(history: &mut Option<History>, plane: &ComplexPlane, colors: &[TrueColor], width: usize, height: usize, shown: &mut Vec<TrueColor>) {
    blend_with(history, plane, colors, width, height, shown, ZOOM_WEIGHT);
}

/// How much more a new frame weighs per unit of zoom since the frame before (the natural logarithm of the ratio of their scales): on a zoom the image before is
/// stretched, so it is blurrier the faster the view zooms (issue #188, see `blend_with`)
pub(super) const ZOOM_WEIGHT: f32 = 0.0;

/// `blend`, where the new frame weighs at least `MIN_WEIGHT + zoom_weight · |ln(scale / scale before)|`
pub(super) fn blend_with(history: &mut Option<History>, plane: &ComplexPlane, colors: &[TrueColor], width: usize, height: usize, shown: &mut Vec<TrueColor>, zoom_weight: f32) {
    shown.clear();
    //(A view turned otherwise than the frames before starts the blend again: its pixels are not the turned image before; with the same rotation the views
    //compare as their unturned views, see `ComplexPlane::unturned`)
    let Some(before) = history.as_mut().filter(|before| before.colors.len() == width * height && before.plane.rotation().to_bits() == plane.rotation().to_bits()) else {
        shown.extend_from_slice(colors);
        *history = Some(History { plane: plane.clone(), colors: colors.to_vec(), frames: 1, scratch: Vec::new() });
        return;
    };
    shown.resize(width * height, TrueColor::new(0, 0, 0));
    let zoom = (plane.increment_x / before.plane.increment_x).ln().abs() as f32;
    let new_weight = weight(before.frames).max(((MIN_WEIGHT + zoom_weight * zoom).min(1.0) * 256.0).round() as u32);
    //Pixel (x, y) of the view is at (x * scale_x + offset_x, y * scale_y + offset_y) in the image before (the centers of the pixels, both with the screen's y down):
    //an axis-aligned map, so the neighbors and weights are computed once per column and once per row
    let (scale_x, scale_y) = (plane.increment_x / before.plane.increment_x, plane.increment_y / before.plane.increment_y);
    let (now, then) = (plane.unturned(), before.plane.unturned());
    let columns = axis(width, scale_x, (now.min_x - then.min_x) / then.increment_x + 0.5 * scale_x - 0.5);
    let rows = axis(height, scale_y, (now.min_y - then.min_y) / then.increment_y + 0.5 * scale_y - 0.5);
    before.scratch.resize(width * height, TrueColor::new(0, 0, 0));
    let (stride, rows_per_chunk) = (3 * width, height.div_ceil(render_thread_count()).max(1));
    let (new, old): (&[u8], &[u8]) = (bytemuck::cast_slice(colors), bytemuck::cast_slice(&before.colors));
    let (columns, rows) = (&columns, &rows);
    let blended: &mut [u8] = bytemuck::cast_slice_mut(&mut before.scratch);
    std::thread::scope(|scope| {
        for (chunk, blended) in blended.chunks_mut(rows_per_chunk * stride).enumerate() {
            scope.spawn(move || {
                let row_of = |y: usize| &new[y.min(height - 1) * stride..][..stride];
                //The ranges of the rows above, at and below the row (rolling: every row's range is computed once), and of the 3 x 3 pixels around every pixel
                let mut ranges = [(); 3].map(|()| (vec![0_u8; stride], vec![0_u8; stride]));
                let first = chunk * rows_per_chunk;
                for (k, y) in [first.saturating_sub(1), first].into_iter().enumerate() {
                    let (low, high) = &mut ranges[k + 1];
                    row_range(row_of(y), low, high);
                }
                let (mut least, mut most, mut between) = (vec![0_u8; stride], vec![0_u8; stride], vec![0_u32; stride]);
                for (row_index, blended) in blended.chunks_mut(stride).enumerate() {
                    let y = first + row_index;
                    ranges.rotate_left(1);
                    let (low, high) = &mut ranges[2];
                    row_range(row_of(y + 1), low, high);
                    let new_row = row_of(y);
                    let Some((y0, y1, fy)) = rows[y] else {
                        blended.copy_from_slice(new_row); //Not covered by the image before: the new frame alone
                        continue;
                    };
                    let [(low_above, high_above), (low_here, high_here), (low_below, high_below)] = &ranges;
                    for (((least, a), b), c) in least.iter_mut().zip(low_above).zip(low_here).zip(low_below) {
                        *least = (*a).min(*b).min(*c);
                    }
                    for (((most, a), b), c) in most.iter_mut().zip(high_above).zip(high_here).zip(high_below) {
                        *most = (*a).max(*b).max(*c);
                    }
                    //The image before between its rows y0 and y1, in 256ths
                    for ((between, top), bottom) in between.iter_mut().zip(&old[y0 * stride..][..stride]).zip(&old[y1 * stride..][..stride]) {
                        *between = u32::from(*top) * (256 - fy) + u32::from(*bottom) * fy;
                    }
                    for (x, column) in columns.iter().enumerate() {
                        let Some((x0, x1, fx)) = *column else {
                            blended[3 * x..3 * x + 3].copy_from_slice(&new_row[3 * x..3 * x + 3]);
                            continue;
                        };
                        for channel in 0..3 {
                            let i = 3 * x + channel;
                            //Bilinear, rounded, within the range of the new frame
                            let previous = ((between[3 * x0 + channel] * (256 - fx) + between[3 * x1 + channel] * fx + 32_768) >> 16) as u8;
                            let previous = previous.clamp(least[i], most[i]);
                            //Rounded; a value that rounding would keep although the new frame differs moves one step towards it, or it would stick up to
                            //1 / (2 · weight) steps away (5 with a weight of 10%)
                            let new = new_row[i];
                            let mixed = ((u32::from(previous) * (256 - new_weight) + u32::from(new) * new_weight + 128) >> 8) as u8;
                            blended[i] = if mixed != previous { mixed } else if new > previous { previous + 1 } else if new < previous { previous - 1 } else { previous };
                        }
                    }
                }
            });
        }
    });
    shown.copy_from_slice(&before.scratch);
    std::mem::swap(&mut before.colors, &mut before.scratch);
    (before.plane, before.frames) = (plane.clone(), before.frames + 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex_plane::View;

    fn plane(view: &View) -> ComplexPlane {
        let mut plane = ComplexPlane::new(40, 30);
        plane.set_view(view);
        plane
    }

    #[test]
    fn frames_of_the_same_view_average_and_then_follow_with_a_least_weight() {
        let view = plane(&View::new(-0.5, 0.0, 1.0));
        let mut history = None;
        //The pixel (index 0) has the value; its neighbors are 0 and 250, as samples of a detailed view differ, so the clamp keeps every value
        let frame = |value: u8| (0..40 * 30).map(|i| if i == 0 { value } else if i % 2 == 0 { 0 } else { 250 }).map(|v| TrueColor::new(v, v, v)).collect::<Vec<_>>();
        let mut shown = Vec::new();
        let mut blended = |history: &mut Option<History>, value: u8| {
            blend(history, &view, &frame(value), 40, 30, &mut shown);
            shown[0].red
        };
        assert_eq!(blended(&mut history, 100), 100, "the first frame as it is");
        assert_eq!(blended(&mut history, 200), 150, "the average of two");
        assert_eq!(blended(&mut history, 50), 117, "of three: (100 + 200 + 50) / 3");
        for _ in 0..100 {
            let _ = blended(&mut history, 0);
        }
        assert_eq!(blended(&mut history, 0), 0, "an old image fades: every frame weighs at least 10%");
    }

    #[test]
    fn a_moved_view_takes_the_image_before_from_where_it_was_and_the_new_frame_where_it_was_not() {
        let before = plane(&View::new(-0.5, 0.0, 1.0));
        let mut history = None;
        let mut first = vec![TrueColor::new(40, 40, 40); 40 * 30];
        first[10 * 40 + 20] = TrueColor::new(60, 60, 60);
        let mut shown = Vec::new();
        blend(&mut history, &before, &first, 40, 30, &mut shown);
        //The view moved 5 pixels to the right: the brighter pixel is at x 15 now; the 5 columns at the right edge are new. The new frame is a checkerboard of 0 and
        //80, so the colors before are within its range everywhere
        let mut moved = before.clone();
        crate::rendering::move_complex_plane_by_pixels(&mut moved, 0, 5);
        let checkers: Vec<TrueColor> = (0..40 * 30).map(|i| if (i % 40 + i / 40) % 2 == 0 { TrueColor::new(80, 80, 80) } else { TrueColor::new(0, 0, 0) }).collect();
        blend(&mut history, &moved, &checkers, 40, 30, &mut shown);
        assert_eq!(shown[10 * 40 + 15].red, 30, "(60 + 0) / 2 at its new place");
        assert_eq!(shown[10 * 40 + 20].red, 60, "(40 + 80) / 2 where it was");
        assert_eq!(shown[10 * 40 + 38].red, 80, "an uncovered pixel takes the new frame alone");
    }

    #[test]
    fn the_image_before_stays_within_the_colors_of_the_new_frame_around_a_pixel_so_nothing_ghosts() {
        let view = plane(&View::new(-0.5, 0.0, 1.0));
        let mut history = None;
        let mut bright = vec![TrueColor::new(20, 20, 20); 40 * 30];
        bright[10 * 40 + 20] = TrueColor::new(250, 250, 250);
        let mut shown = Vec::new();
        blend(&mut history, &view, &bright, 40, 30, &mut shown);
        //The bright pixel is gone from the new frame, whose pixels around it are 20 to 30: the old 250 is clamped to 30 first
        let mut next = vec![TrueColor::new(20, 20, 20); 40 * 30];
        next[11 * 40 + 21] = TrueColor::new(30, 30, 30);
        blend(&mut history, &view, &next, 40, 30, &mut shown);
        assert_eq!(shown[10 * 40 + 20].red, 25, "(30 + 20) / 2, not (250 + 20) / 2");
    }

    #[test]
    fn half_a_pixel_away_the_image_before_is_read_between_its_pixels() {
        let view = plane(&View::new(-0.5, 0.0, 1.0));
        let mut history = None;
        let stripes: Vec<TrueColor> = (0..40 * 30).map(|i| if i % 2 == 0 { TrueColor::new(100, 100, 100) } else { TrueColor::new(0, 0, 0) }).collect();
        let mut shown = Vec::new();
        blend(&mut history, &view, &stripes, 40, 30, &mut shown);
        //Half a pixel to the right: every pixel lies between a stripe of 100 and one of 0, so the image before gives 50 there (the nearest pixel gave 0 or 100)
        let mut moved = view.clone();
        moved.min_x += moved.increment_x / 2.0;
        blend(&mut history, &moved, &stripes, 40, 30, &mut shown);
        assert_eq!((shown[5 * 40 + 10].red, shown[5 * 40 + 11].red), (75, 25), "(50 + 100) / 2 and (50 + 0) / 2");
    }
}

#[cfg(test)]
mod timing {
    #[test]
    #[ignore = "timing, run by hand"]
    fn blend_time_at_4k() {
        let (w, h) = (3840, 2160);
        let mut plane = crate::complex_plane::ComplexPlane::new(w, h);
        plane.set_view(&crate::complex_plane::View::new(-0.5, 0.0, 1.0));
        let colors = vec![crate::coloring::TrueColor::new(10, 20, 30); w * h];
        let mut history = None;
        let mut shown = Vec::new();
        super::blend(&mut history, &plane, &colors, w, h, &mut shown);
        for _ in 0..3 {
            plane.scale(0.99);
            let started = std::time::Instant::now();
            super::blend(&mut history, &plane, &colors, w, h, &mut shown);
            println!("blend at 4K: {:.1} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
    }

    /// The quality of the blend over a flight: every frame against a render of its view with 16 samples per pixel (the error), and how much the shown image
    /// changes from frame to frame against how much that reference changes (the flicker; 1 is the motion alone)
    #[test]
    #[ignore = "measurement, run by hand"]
    #[allow(clippy::cast_lossless)] // Counts of a measurement
    fn blend_quality_on_flights() {
        use crate::coloring::TrueColor;
        use crate::complex_plane::{ComplexPlane, View};
        use crate::mandelbrot_set::MandelbrotSet;
        use crate::rendering::Render;
        let (w, h) = (480, 320);
        let render = |plane: &ComplexPlane, m: &MandelbrotSet, sample: Option<u8>| {
            let mut out = Vec::new();
            let r = Render::new(plane, m, w, h, TrueColor::new_from_bernstein_polynomials);
            let r = match sample { Some(sample) => r.colors_only_of_sample(sample), None => r.supersampling(16) };
            r.run(|colors, _| out = colors.to_vec());
            out
        };
        let diff = |a: &[TrueColor], b: &[TrueColor]| a.iter().zip(b).map(|(a, b)| (i32::from(a.red) - i32::from(b.red)).abs() + (i32::from(a.green) - i32::from(b.green)).abs() + (i32::from(a.blue) - i32::from(b.blue)).abs()).sum::<i32>() as f64 / (3 * a.len()) as f64;
        for (name, x, y, from, to, frames) in [("Seahorse zoom", -0.7498, 0.0203, 0.02, 0.0005, 90), ("Frost zoom", -0.66041, 0.44375, 0.1, 0.004, 90), ("Seahorse pan", -0.7498, 0.0203, 0.004, 0.004, 60)] {
            let m = MandelbrotSet::new(1500, 2.0);
            let planes: Vec<ComplexPlane> = (0..frames).map(|frame| {
                let t = frame as f64 / (frames - 1) as f64;
                let mut plane = ComplexPlane::new(w, h);
                let pan = if f64::abs(from - to) < 1e-12 { 0.5 * from * (t - 0.5) } else { 0.0 };
                plane.set_view(&View::new(x + pan, y, from * (to / from).powf(t)));
                plane
            }).collect();
            let references: Vec<Vec<TrueColor>> = planes.iter().map(|plane| render(plane, &m, None)).collect();
            let reference_flicker: f64 = references.windows(2).map(|pair| diff(&pair[0], &pair[1])).sum::<f64>();
            for sampling in [false, true] {
                let (mut history, mut shown, mut previous) = (None, Vec::new(), Vec::new());
                let (mut error, mut flicker) = (0.0, 0.0);
                for (frame, plane) in planes.iter().enumerate() {
                    let colors = render(plane, &m, Some((frame % 64) as u8));
                    if sampling {
                        super::blend(&mut history, plane, &colors, w, h, &mut shown);
                    } else {
                        shown = colors;
                    }
                    if frame >= 10 {
                        error += diff(&shown, &references[frame]);
                    }
                    if frame > 0 {
                        flicker += diff(&shown, &previous);
                    }
                    previous.clone_from(&shown);
                }
                println!("{name}: {:<10} error {:.2}, flicker {:.2}", if sampling { "blended" } else { "one frame" }, error / (frames - 10) as f64, flicker / reference_flicker);
            }
        }
    }
}

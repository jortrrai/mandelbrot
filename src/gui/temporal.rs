//! Blending the frames of a flight over time (issue #132): every full-resolution frame of a flight takes another sample of its pixels (a new jitter, see
//! `Render::colors_only_of_sample`), and is blended into the image of the frames before, moved and scaled to its view. The flight path is known exactly and the
//! set does not move, so the image converges towards a supersampled one while the camera moves. Measured on flights at 480 x 320 against 16 samples per pixel:
//! 22% (Frost on the Shoreline) to 26% (Seahorse Embers) less error per frame; it costs about 7 ms of a 4K frame (all threads).
//!
//! A pixel that the image before does not cover (the edges a zoom out or a pan uncover) takes the new frame alone. The blend starts again with every flight.

use crate::coloring::TrueColor;
use crate::complex_plane::ComplexPlane;
use crate::rendering::render_thread_count;

/// The least weight of a new frame: the image follows the view within about 1 / `MIN_WEIGHT` frames (measured in #118: 0.1 is the best of 0.1 to 0.5)
const MIN_WEIGHT: f32 = 0.1;

/// The image of the frames so far: its view, its colors, how many frames it has, and the memory of the next blend (kept, so a 4K frame does not allocate and
/// fault in its memory every time). The colors are the 8-bit colors that are shown: the blend of a 4K frame is limited by memory, and floats made it two and a
/// half times as slow (measured: 11 ms instead of about 4 ms); the rounding of every frame is at most half a color step
pub(super) struct History {
    plane: ComplexPlane,
    colors: Vec<TrueColor>,
    frames: u32,
    scratch: Vec<TrueColor>,
}

/// The weight of frame `frames + 1`: `1 / (frames + 1)` (the plain average while there are few frames), at least `MIN_WEIGHT`
fn weight(frames: u32) -> f32 {
    (1.0 / (frames as f32 + 1.0)).max(MIN_WEIGHT)
}

/// Where the pixels `0..count` of an axis of the view are in the image before (`position = index * scale + offset`): the two nearest pixels there and the
/// weight of the second, or `None` outside it
fn axis(count: usize, scale: f64, offset: f64) -> Vec<Option<(usize, usize, f32)>> {
    (0..count).map(|index| {
        let position = index as f64 * scale + offset;
        if position < 0.0 || position > (count - 1) as f64 {
            return None;
        }
        let low = position.floor() as usize;
        Some((low, (low + 1).min(count - 1), (position - low as f64) as f32))
    }).collect()
}

/// Blends the frame `colors` of view `plane` (`width` x `height` pixels) into `history` and writes the colors to show into `shown`; the first frame starts the history
pub(super) fn blend(history: &mut Option<History>, plane: &ComplexPlane, colors: &[TrueColor], width: usize, height: usize, shown: &mut Vec<TrueColor>) {
    shown.clear();
    let Some(before) = history.as_mut().filter(|before| before.colors.len() == width * height) else {
        shown.extend_from_slice(colors);
        let colors = colors.to_vec();
        *history = Some(History { plane: plane.clone(), colors, frames: 1, scratch: Vec::new() });
        return;
    };
    shown.resize(width * height, TrueColor::new(0, 0, 0));
    let new_weight = weight(before.frames);
    //Pixel (x, y) of the view is at (x * scale_x + offset_x, y * scale_y + offset_y) in the image before (the centers of the pixels, both with the screen's y down):
    //an axis-aligned map, so the neighbors and weights are computed once per column and once per row
    let (scale_x, scale_y) = (plane.increment_x / before.plane.increment_x, plane.increment_y / before.plane.increment_y);
    let columns = axis(width, scale_x, (plane.min_x - before.plane.min_x) / before.plane.increment_x + 0.5 * scale_x - 0.5);
    let rows = axis(height, scale_y, (plane.min_y - before.plane.min_y) / before.plane.increment_y + 0.5 * scale_y - 0.5);
    before.scratch.resize(width * height, TrueColor::new(0, 0, 0));
    let rows_per_chunk = height.div_ceil(render_thread_count()).max(1);
    let (old, columns, rows) = (&before.colors, &columns, &rows);
    std::thread::scope(|scope| {
        for (chunk, (blended, shown)) in before.scratch.chunks_mut(rows_per_chunk * width).zip(shown.chunks_mut(rows_per_chunk * width)).enumerate() {
            scope.spawn(move || {
                for (row_index, (blended, shown)) in blended.chunks_mut(width).zip(shown.chunks_mut(width)).enumerate() {
                    let y = chunk * rows_per_chunk + row_index;
                    let row = rows[y];
                    for (x, (pixel, out)) in blended.iter_mut().zip(shown.iter_mut()).enumerate() {
                        let color = colors[y * width + x];
                        let new = [f32::from(color.red), f32::from(color.green), f32::from(color.blue)];
                        //Only where the image before covers the pixel (with its 4 nearest pixels): elsewhere the new frame alone
                        let (Some((y0, y1, fy)), Some((x0, x1, fx))) = (row, columns[x]) else {
                            (*pixel, *out) = (color, color);
                            continue;
                        };
                        //The nearest pixel of the image before: one read instead of the four of bilinear weights, which also smear the history (and plain
                        //multiplications: without FMA in the target, `mul_add` is a call to the library)
                        let previous = old[if fy < 0.5 { y0 } else { y1 } * width + if fx < 0.5 { x0 } else { x1 }];
                        //Rounded by adding a half (`round` is a library call); a value that rounding would keep although the new frame differs moves one step towards
                        //it, or it would stick up to 1 / (2 · weight) steps away (5 with a weight of 10%)
                        let mix = |previous: u8, new: f32| {
                            let mixed = (f32::from(previous) * (1.0 - new_weight) + new * new_weight + 0.5) as u8;
                            match (mixed == previous, new) {
                                (true, new) if new > f32::from(previous) => previous + 1,
                                (true, new) if new < f32::from(previous) => previous - 1,
                                _ => mixed,
                            }
                        };
                        *pixel = TrueColor::new(mix(previous.red, new[0]), mix(previous.green, new[1]), mix(previous.blue, new[2]));
                        *out = *pixel;
                    }
                }
            });
        }
    });
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
        let frame = |value: u8| vec![TrueColor::new(value, value, value); 40 * 30];
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
        let mut first = vec![TrueColor::new(0, 0, 0); 40 * 30];
        first[10 * 40 + 20] = TrueColor::new(200, 200, 200);
        let mut shown = Vec::new();
        blend(&mut history, &before, &first, 40, 30, &mut shown);
        //The view moved 5 pixels to the right: the bright pixel is at x 15 now; the 5 columns at the right edge are new
        let mut moved = before.clone();
        crate::rendering::move_complex_plane_by_pixels(&mut moved, 0, 5);
        blend(&mut history, &moved, &vec![TrueColor::new(40, 40, 40); 40 * 30], 40, 30, &mut shown);
        assert_eq!(shown[10 * 40 + 15].red, 120, "(200 + 40) / 2 at its new place");
        assert_eq!(shown[10 * 40 + 20].red, 20, "(0 + 40) / 2 where it was");
        assert_eq!(shown[10 * 40 + 38].red, 40, "an uncovered pixel takes the new frame alone");
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
}

//! Dive videos from keyframe images (issue #209): a dive into a fixed center renders one image per doubling of the zoom, and every frame is composed from
//! the image of its depth, instead of rendering every frame from nothing.
//!
//! A frame at scale `s` shows the view of width `2.5 s`. Keyframe `k` is the view at scale `S_k = S_0 / 2^k` (from the shallowest frame down) at twice the
//! frame's size on each side, so it serves the frames from `S_k / 2` to `S_k`: one of their pixels covers one to two of its pixels on each side. Each part of
//! the picture is rendered about four times instead of once per frame of the doubling, which for a slow dive is hundreds of frames.
//!
//! A frame takes the colors of its keyframe, averaged over the area of each of its pixels (four bilinear taps inside it, in the gamma-space colors, like the
//! supersampling averages them), turned by the frame's rotation. The colors and the iteration limit are those of the keyframe (its limit is the limit of the
//! deepest frame it serves), so a dive whose colors change on the way is rendered frame by frame as before, see `DivePlan::of`.

use std::time::Duration;

use crate::coloring::TrueColor;
use crate::complex_plane::{ComplexPlane, View};
use crate::mandelbrot_set::MandelbrotSet;
use crate::rendering::{self, gpu::GpuIterator, AdaptiveSampling, Render};

/// The largest frame that keyframes are made for: a keyframe has four times its pixels (7680 x 4320 for 4K) and is kept in memory while the frames are composed
pub(super) const LARGEST_FRAME: usize = 3840 * 2160;

/// What a dive needs to be composed from keyframes: its fixed center and the scales of its shallowest and deepest frames
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DivePlan {
    /// The real and imaginary parts of the center
    pub(super) center: (f64, f64),
    pub(super) shallowest: f64,
    pub(super) deepest: f64,
}

impl DivePlan {
    /// The plan of the frames at the moments `times` of a flight whose camera is at `camera_at(t)`, or `None` when they do not share one center exactly (a
    /// flight that also moves sideways) or do not zoom
    pub(super) fn of(times: impl Iterator<Item = Duration>, camera_at: impl Fn(Duration) -> ((f64, f64), f64)) -> Option<DivePlan> {
        let mut plan: Option<DivePlan> = None;
        for time in times {
            let (center, scale) = camera_at(time);
            if !(scale.is_finite() && scale > 0.0) {
                return None;
            }
            match &mut plan {
                None => plan = Some(DivePlan { center, shallowest: scale, deepest: scale }),
                Some(plan) => {
                    if center.0.to_bits() != plan.center.0.to_bits() || center.1.to_bits() != plan.center.1.to_bits() {
                        return None;
                    }
                    (plan.shallowest, plan.deepest) = (plan.shallowest.max(scale), plan.deepest.min(scale));
                }
            }
        }
        plan.filter(|plan| plan.shallowest > plan.deepest)
    }

    /// The scales of the keyframes: from the shallowest frame, halving, until the last one serves the deepest frame
    pub(super) fn scales(&self) -> Vec<f64> {
        let count = (self.shallowest / self.deepest).log2().ceil().max(0.0) as usize + 1;
        (0..count).map(|k| self.shallowest / f64::from(1_u32 << k.min(60))).collect()
    }
}

/// The keyframe images of a dive, see the module
pub(super) struct DiveKeyframes {
    /// The size of the frames; a keyframe is twice as large on each side
    width: usize,
    height: usize,
    scales: Vec<f64>,
    images: Vec<Vec<TrueColor>>,
}

impl DiveKeyframes {
    /// Renders the keyframes of `plan` for frames of `width` x `height` with `samples` samples per pixel of a keyframe (adaptive or not), in the set that
    /// `set_at(scale)` gives for a keyframe's deepest frame, on all CPU threads with the GPU. `on_keyframe(k, count)` is told before keyframe `k`. Returns
    /// `None` when cancelled.
    #[allow(clippy::too_many_arguments)] // The dive, the frames, the quality and how to render
    pub(super) fn render(plan: DivePlan, (width, height): (usize, usize), samples: u8, adaptive: Option<AdaptiveSampling>, coloring_function: fn(f64, f64) -> TrueColor,
        set_at: impl Fn(f64) -> MandelbrotSet, gpu: Option<&GpuIterator>, fast_gpu: bool, cancel: &std::sync::atomic::AtomicBool, mut on_keyframe: impl FnMut(usize, usize)) -> Option<DiveKeyframes> {
        let scales = plan.scales();
        let (big_width, big_height) = (width * 2, height * 2);
        let mut images = Vec::with_capacity(scales.len());
        for (index, scale) in scales.iter().enumerate() {
            on_keyframe(index, scales.len());
            let mut plane = ComplexPlane::new(big_width, big_height);
            plane.set_view(&View::new(plan.center.0, plan.center.1, *scale));
            //(The limit of the deepest frame this keyframe serves)
            let m = set_at(scale / 2.0);
            let mut colors = Vec::new();
            let completed = Render::new(&plane, &m, big_width, big_height, coloring_function).supersampling(samples).adaptive(adaptive)
                .gpu(gpu.map(|gpu| (gpu, Duration::ZERO))).fast_gpu(fast_gpu).cancel_with(cancel)
                .run(|layer, pass| {
                    if pass >= samples.clamp(1, 64) {
                        colors = layer.to_vec();
                    }
                });
            if !completed {
                return None;
            }
            images.push(colors);
        }
        Some(DiveKeyframes { width, height, scales, images })
    }

    /// The frame at `scale`, turned counterclockwise by `rotation` (radians), composed from the keyframe of its depth (see the module)
    #[cfg(test)]
    pub(super) fn compose(&self, scale: f64, rotation: f64) -> Vec<TrueColor> {
        self.compose_moments(&[(scale, rotation)])
    }

    /// The frame that averages the frames at the `(scale, rotation)` of every moment (motion blur), each composed from the keyframe of its depth (see the module)
    /// and rounded to 8 bits before the average, in one pass over the pixels on all CPU threads
    pub(super) fn compose_moments(&self, moments: &[(f64, f64)]) -> Vec<TrueColor> {
        let (width, height) = (self.width, self.height);
        let (big_width, big_height) = (width * 2, height * 2);
        let moments: Vec<Moment> = moments.iter().map(|&(scale, rotation)| Moment::new(self, scale, rotation)).collect();
        let count = moments.len() as f32;
        let mut frame = vec![TrueColor::new(0, 0, 0); width * height];
        let rows_per_chunk = height.div_ceil(rendering::render_thread_count()).max(1);
        std::thread::scope(|scope| {
            for (chunk, rows) in frame.chunks_mut(rows_per_chunk * width).enumerate() {
                let moments = &moments;
                scope.spawn(move || {
                    for (row_index, row) in rows.chunks_mut(width).enumerate() {
                        let y = chunk * rows_per_chunk + row_index;
                        //The rows of the keyframe that the taps of this row read, for every unturned moment
                        let tap_rows: Vec<Option<[Tap; 2]>> = moments.iter().map(|moment| moment.columns.as_ref().map(|_| TAPS_Y.map(|tap_y| Tap::at((y as f64 + tap_y - height as f64 / 2.0) * moment.footprint + big_height as f64 / 2.0, big_height)))).collect();
                        for (x, pixel) in row.iter_mut().enumerate() {
                            let mut average = [0.0_f32; 3];
                            for (moment, tap_rows) in moments.iter().zip(&tap_rows) {
                                //Four taps inside the pixel, its offset from the middle turned like the view (the image turns counterclockwise: R(-θ), with y down on the screen)
                                let mut sum = [0.0_f32; 3];
                                for (tap, (tap_x, tap_y)) in TAPS.iter().enumerate() {
                                    let color = match (&moment.columns, tap_rows) {
                                        (Some(columns), Some(tap_rows)) => bilinear_at(moment.image, big_width, columns[tap % 2][x], tap_rows[tap / 2]),
                                        _ => {
                                            let (dx, dy) = ((x as f64 + tap_x - width as f64 / 2.0) * moment.footprint, (y as f64 + tap_y - height as f64 / 2.0) * moment.footprint);
                                            let (turned_x, turned_y) = (dx * moment.cos - dy * moment.sin, dx * moment.sin + dy * moment.cos);
                                            bilinear(moment.image, big_width, big_height, turned_x + big_width as f64 / 2.0, turned_y + big_height as f64 / 2.0)
                                        }
                                    };
                                    for (channel, value) in sum.iter_mut().zip(color) {
                                        *channel += value;
                                    }
                                }
                                //(Each moment rounded to its color, as a frame of it would be)
                                for (channel, value) in average.iter_mut().zip(sum) {
                                    *channel += (value / 4.0).round();
                                }
                            }
                            *pixel = TrueColor::new((average[0] / count).round() as u8, (average[1] / count).round() as u8, (average[2] / count).round() as u8);
                        }
                    }
                });
            }
        });
        frame
    }

    /// The scales of the keyframes, shallowest first
    #[cfg(test)]
    pub(super) fn scales(&self) -> &[f64] {
        &self.scales
    }
}

/// The four taps inside a pixel, in its own units, in the order their colors are added
const TAPS: [(f64, f64); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];
/// The two columns and the two rows of the taps
const TAPS_X: [f64; 2] = [0.25, 0.75];
const TAPS_Y: [f64; 2] = [0.25, 0.75];

/// One moment of a composed frame: its keyframe, the size of a frame pixel in keyframe pixels and the turn; for an unturned moment the columns of the keyframe
/// that the two taps of every column read, worked out once (the rows are worked out per row)
struct Moment<'a> {
    image: &'a [TrueColor],
    footprint: f64,
    sin: f64,
    cos: f64,
    columns: Option<[Vec<Tap>; 2]>,
}

impl<'a> Moment<'a> {
    fn new(keyframes: &'a DiveKeyframes, scale: f64, rotation: f64) -> Moment<'a> {
        let (width, big_width) = (keyframes.width, keyframes.width * 2);
        //The keyframe that serves this scale: the deepest whose scale is at least the frame's
        let index = keyframes.scales.iter().rposition(|keyframe| *keyframe >= scale * (1.0 - 1e-12)).unwrap_or(0);
        //A pixel of the frame in pixels of the keyframe (1 to 2)
        let footprint = (scale / keyframes.scales[index]) * 2.0;
        let (sin, cos) = rotation.sin_cos();
        //Unturned, a tap's column depends on its column only (dx · 1 - dy · 0 is dx exactly), so the taps read the same pixels as when worked out each time
        let columns = (sin.to_bits() == 0.0_f64.to_bits() && cos.to_bits() == 1.0_f64.to_bits()).then(|| TAPS_X.map(|tap_x| (0..width).map(|x| Tap::at((x as f64 + tap_x - width as f64 / 2.0) * footprint + big_width as f64 / 2.0, big_width)).collect()));
        Moment { image: &keyframes.images[index], footprint, sin, cos, columns }
    }
}

/// Where a tap falls along one side of a keyframe: the two pixels it is between and its share of the second, as `bilinear` works them out
#[derive(Clone, Copy)]
struct Tap {
    first: usize,
    second: usize,
    share: f32,
}

impl Tap {
    fn at(position: f64, length: usize) -> Tap {
        let position = (position - 0.5).clamp(0.0, (length - 1) as f64);
        let first = position.floor() as usize;
        Tap { first, second: (first + 1).min(length - 1), share: (position - first as f64) as f32 }
    }
}

/// The color of `image` (`width` pixels wide) between the pixels of the taps `column` and `row`, exactly as `bilinear` computes it
fn bilinear_at(image: &[TrueColor], width: usize, column: Tap, row: Tap) -> [f32; 3] {
    let at = |x: usize, y: usize| {
        let color = image[y * width + x];
        [f32::from(color.red), f32::from(color.green), f32::from(color.blue)]
    };
    let (a, b, c, d) = (at(column.first, row.first), at(column.second, row.first), at(column.first, row.second), at(column.second, row.second));
    let mut out = [0.0; 3];
    for channel in 0..3 {
        let upper = a[channel] + (b[channel] - a[channel]) * column.share;
        let lower = c[channel] + (d[channel] - c[channel]) * column.share;
        out[channel] = upper + (lower - upper) * row.share;
    }
    out
}

/// The color of `image` (`width` x `height`) at the point (`x`, `y`) in pixels (pixel `i` covers `i` to `i + 1`), between its four nearest pixel centers;
/// outside the image the edge pixels repeat
fn bilinear(image: &[TrueColor], width: usize, height: usize, x: f64, y: f64) -> [f32; 3] {
    let (x, y) = ((x - 0.5).clamp(0.0, (width - 1) as f64), (y - 0.5).clamp(0.0, (height - 1) as f64));
    let (left, top) = (x.floor() as usize, y.floor() as usize);
    let (right, bottom) = ((left + 1).min(width - 1), (top + 1).min(height - 1));
    let (fx, fy) = ((x - left as f64) as f32, (y - top as f64) as f32);
    let at = |column: usize, row: usize| {
        let color = image[row * width + column];
        [f32::from(color.red), f32::from(color.green), f32::from(color.blue)]
    };
    let (a, b, c, d) = (at(left, top), at(right, top), at(left, bottom), at(right, bottom));
    let mut out = [0.0; 3];
    for channel in 0..3 {
        let upper = a[channel] + (b[channel] - a[channel]) * fx;
        let lower = c[channel] + (d[channel] - c[channel]) * fx;
        out[channel] = upper + (lower - upper) * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dive_with_a_fixed_center_has_a_keyframe_per_doubling_and_a_moving_flight_none() {
        let center = (-0.7498, 0.0203);
        let times = (0..100).map(|frame| Duration::from_millis(frame * 10));
        let plan = DivePlan::of(times.clone(), |time| (center, 1.0 / 2f64.powf(time.as_secs_f64() * 10.0))).unwrap();
        assert_eq!((plan.shallowest, plan.center), (1.0, center));
        let scales = plan.scales();
        assert!(scales.len() >= 10 && *scales.last().unwrap() <= plan.deepest && scales.windows(2).all(|pair| (pair[0] / pair[1] - 2.0).abs() < 1e-12), "{scales:?}");
        assert!(DivePlan::of(times.clone(), |time| ((-0.75 + time.as_secs_f64(), 0.0), 0.5)).is_none(), "a flight that moves sideways");
        assert!(DivePlan::of(times, |_| (center, 0.5)).is_none(), "no zoom");
    }

    /// The frame of `moments` as it was composed before #236: every moment a frame of its own (each tap worked out on its own), averaged
    fn composed_frame_by_frame(keyframes: &DiveKeyframes, moments: &[(f64, f64)]) -> Vec<TrueColor> {
        let (width, height, big_width, big_height) = (keyframes.width, keyframes.height, keyframes.width * 2, keyframes.height * 2);
        let frames: Vec<Vec<TrueColor>> = moments.iter().map(|&(scale, rotation)| {
            let index = keyframes.scales.iter().rposition(|keyframe| *keyframe >= scale * (1.0 - 1e-12)).unwrap_or(0);
            let footprint = (scale / keyframes.scales[index]) * 2.0;
            let (sin, cos) = rotation.sin_cos();
            (0..width * height).map(|index_of_pixel| {
                let (x, y) = (index_of_pixel % width, index_of_pixel / width);
                let mut sum = [0.0_f32; 3];
                for (tap_x, tap_y) in TAPS {
                    let (dx, dy) = ((x as f64 + tap_x - width as f64 / 2.0) * footprint, (y as f64 + tap_y - height as f64 / 2.0) * footprint);
                    let (turned_x, turned_y) = (dx * cos - dy * sin, dx * sin + dy * cos);
                    let color = bilinear(&keyframes.images[index], big_width, big_height, turned_x + big_width as f64 / 2.0, turned_y + big_height as f64 / 2.0);
                    for (channel, value) in sum.iter_mut().zip(color) {
                        *channel += value;
                    }
                }
                TrueColor::new((sum[0] / 4.0).round() as u8, (sum[1] / 4.0).round() as u8, (sum[2] / 4.0).round() as u8)
            }).collect()
        }).collect();
        let count = frames.len() as f32;
        (0..width * height).map(|index| {
            let mut sum = [0.0_f32; 3];
            for frame in &frames {
                sum[0] += f32::from(frame[index].red);
                sum[1] += f32::from(frame[index].green);
                sum[2] += f32::from(frame[index].blue);
            }
            TrueColor::new((sum[0] / count).round() as u8, (sum[1] / count).round() as u8, (sum[2] / count).round() as u8)
        }).collect()
    }

    #[test]
    fn frames_composed_in_one_pass_are_bit_for_bit_the_frames_composed_one_moment_at_a_time() {
        //#236: the moments of a frame in one pass, and the taps of unturned moments worked out once per column and row, give exactly the frames of before
        let (width, height) = (60, 36);
        let plan = DivePlan { center: (-0.7498, 0.0203), shallowest: 0.01, deepest: 0.003 };
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let keyframes = DiveKeyframes::render(plan, (width, height), 4, None, TrueColor::new_from_bernstein_polynomials, |_| MandelbrotSet::new(400, 2.0), None, false, &cancel, |_, _| {}).unwrap();
        for moments in [vec![(0.008, 0.0)], vec![(0.0049, 0.0), (0.0052, 0.0), (0.0061, 0.0), (0.0099, 0.0)], vec![(0.007, 0.3)], vec![(0.006, 0.0), (0.0062, 0.01), (0.0064, 0.02)]] {
            let one_pass = keyframes.compose_moments(&moments);
            let reference = composed_frame_by_frame(&keyframes, &moments);
            assert!(one_pass.iter().zip(&reference).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)), "{moments:?}");
        }
    }

    #[test]
    fn a_composed_frame_is_close_to_the_frame_rendered_directly_and_turns_with_the_view() {
        let (width, height) = (96, 64);
        let plan = DivePlan { center: (-0.7498, 0.0203), shallowest: 0.01, deepest: 0.003 };
        let set_at = |_| MandelbrotSet::new(800, 2.0);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let coloring = TrueColor::new_from_bernstein_polynomials;
        let keyframes = DiveKeyframes::render(plan, (width, height), 8, None, coloring, set_at, None, false, &cancel, |_, _| {}).unwrap();
        assert_eq!(keyframes.scales().len(), 3);
        //Between two keyframes, against the same frame rendered at 32x
        for scale in [0.01, 0.007, 0.0045] {
            let composed = keyframes.compose(scale, 0.0);
            let mut plane = ComplexPlane::new(width, height);
            plane.set_view(&View::new(-0.7498, 0.0203, scale));
            let mut direct = Vec::new();
            assert!(Render::new(&plane, &set_at(scale), width, height, coloring).supersampling(32).run(|layer, _| direct = layer.to_vec()));
            let error: f64 = composed.iter().zip(&direct).map(|(a, b)| f64::from(a.red.abs_diff(b.red)) + f64::from(a.green.abs_diff(b.green)) + f64::from(a.blue.abs_diff(b.blue))).sum::<f64>() / (3 * width * height) as f64;
            assert!(error < 12.0, "scale {scale}: mean difference {error}");
        }
        //A half turn shows the frame upside down: its first pixel is the last pixel of the unturned frame
        let (straight, turned) = (keyframes.compose(0.007, 0.0), keyframes.compose(0.007, std::f64::consts::PI));
        let difference = |a: TrueColor, b: TrueColor| a.red.abs_diff(b.red).max(a.green.abs_diff(b.green)).max(a.blue.abs_diff(b.blue));
        assert!(difference(turned[0], straight[width * height - 1]) <= 2 && difference(turned[width * height - 1], straight[0]) <= 2);
    }
}

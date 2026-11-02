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
    pub(super) fn compose(&self, scale: f64, rotation: f64) -> Vec<TrueColor> {
        let (width, height) = (self.width, self.height);
        //The keyframe that serves this scale: the deepest whose scale is at least the frame's
        let index = self.scales.iter().rposition(|keyframe| *keyframe >= scale * (1.0 - 1e-12)).unwrap_or(0);
        let (image, keyframe_scale) = (&self.images[index], self.scales[index]);
        let (big_width, big_height) = (width * 2, height * 2);
        //A pixel of the frame in pixels of the keyframe (1 to 2), and the turn
        let footprint = (scale / keyframe_scale) * 2.0;
        let (sin, cos) = rotation.sin_cos();
        let mut frame = vec![TrueColor::new(0, 0, 0); width * height];
        let rows_per_chunk = height.div_ceil(rendering::render_thread_count()).max(1);
        std::thread::scope(|scope| {
            for (chunk, rows) in frame.chunks_mut(rows_per_chunk * width).enumerate() {
                scope.spawn(move || {
                    for (index, pixel) in rows.iter_mut().enumerate() {
                        let (x, y) = (index % width, chunk * rows_per_chunk + index / width);
                        //Four taps inside the pixel, its offset from the middle turned like the view (the image turns counterclockwise: R(-θ), with y down on the screen)
                        let mut sum = [0.0_f32; 3];
                        for (tap_x, tap_y) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                            let (dx, dy) = ((x as f64 + tap_x - width as f64 / 2.0) * footprint, (y as f64 + tap_y - height as f64 / 2.0) * footprint);
                            let (turned_x, turned_y) = (dx * cos - dy * sin, dx * sin + dy * cos);
                            let color = bilinear(image, big_width, big_height, turned_x + big_width as f64 / 2.0, turned_y + big_height as f64 / 2.0);
                            for (channel, value) in sum.iter_mut().zip(color) {
                                *channel += value;
                            }
                        }
                        *pixel = TrueColor::new((sum[0] / 4.0).round() as u8, (sum[1] / 4.0).round() as u8, (sum[2] / 4.0).round() as u8);
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

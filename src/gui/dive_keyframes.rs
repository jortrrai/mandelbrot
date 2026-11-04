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
    /// The images on the GPU, which then composes the unturned frames (see `compose_moments`)
    on_gpu: Option<rendering::gpu::ComposeImages>,
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
        //(Without room on the GPU the CPU composes every frame)
        let on_gpu = gpu.and_then(|gpu| gpu.upload_images(&images, big_width, big_height).ok());
        Some(DiveKeyframes { width, height, scales, images, on_gpu })
    }

    /// The frame at `scale`, turned counterclockwise by `rotation` (radians), composed from the keyframe of its depth (see the module)
    #[cfg(test)]
    pub(super) fn compose(&self, scale: f64, rotation: f64) -> Vec<TrueColor> {
        self.compose_moments(&[(scale, rotation)], None)
    }

    /// The frame that averages the frames at the `(scale, rotation)` of every moment (motion blur), each composed from the keyframe of its depth (see the module)
    /// and rounded to 8 bits before the average, in one pass over the pixels: on `gpu` when every moment is unturned and the images are there, else on all
    /// CPU threads, bit for bit the same
    pub(super) fn compose_moments(&self, moments: &[(f64, f64)], gpu: Option<&rendering::gpu::GpuIterator>) -> Vec<TrueColor> {
        let (width, height) = (self.width, self.height);
        let moments: Vec<Moment> = moments.iter().map(|&(scale, rotation)| Moment::new(self, scale, rotation)).collect();
        if let (Some(gpu), Some(images)) = (gpu, &self.on_gpu) {
            let unturned: Option<Vec<(usize, Vec<Tap>)>> = moments.iter().map(|moment| moment.gpu_taps(height).map(|taps| (moment.index, taps))).collect();
            if let Some(colors) = unturned.and_then(|unturned| gpu.compose(images, (width, height), &unturned).ok()) {
                return colors;
            }
        }
        compose_on_the_cpu(&moments, (width, height))
    }

    /// The scales of the keyframes, shallowest first
    #[cfg(test)]
    pub(super) fn scales(&self) -> &[f64] {
        &self.scales
    }
}

/// A dive into an exact center, also deeper than 64-bit floats reach (issue #263): the keyframes of `DiveKeyframes`, at the scales `S_0 / 2^k` as exact
/// `Depth`s, rendered when the frames reach their depth and kept only while they are in use, so a dive of a thousand doublings (to 10⁻³⁰⁸) needs a few
/// in memory at a time instead of all of them. The frames are composed from them on the CPU exactly as `DiveKeyframes` composes them
pub(super) struct DeepDive<'a> {
    width: usize,
    height: usize,
    /// The base-2 logarithm of the scale of the first keyframe, and the number of keyframes
    shallowest: crate::rendering::deep::Depth,
    count: usize,
    /// Renders the keyframe at a scale, twice the frame's size on each side (`None`: cancelled)
    render: Box<dyn Fn(crate::rendering::deep::Depth) -> Option<Vec<TrueColor>> + Sync + 'a>,
    kept: std::sync::Mutex<Vec<KeptKeyframe>>,
    /// One keyframe is rendered at a time (each with all the threads)
    rendering: std::sync::Mutex<()>,
    /// The GPU that composes the unturned frames, as for `DiveKeyframes` (bit for bit the CPU's; each kept keyframe is put on it when it is rendered)
    gpu: Option<&'a GpuIterator>,
}

/// A keyframe a `DeepDive` keeps: its number and its image
type KeptKeyframe = (usize, KeyframeImage);

/// A keyframe image of a `DeepDive`, and the image on the GPU (if it could be put there)
type KeyframeImage = (std::sync::Arc<Vec<TrueColor>>, Option<std::sync::Arc<rendering::gpu::ComposeImages>>);

/// The keyframes a `DeepDive` keeps: the one in use and the two before it (for frames that are still composed from them)
const KEPT_KEYFRAMES: usize = 3;

impl<'a> DeepDive<'a> {
    /// The base-2 logarithms of the scales of the shallowest and the deepest frame of a flight whose camera is `deep_at(t)` at the moments `times`, if the
    /// frames all have the center of the destination (a dive) and zoom
    pub(super) fn plan(times: impl Iterator<Item = Duration>, deep_at: impl Fn(Duration) -> super::flight::DeepCamera) -> Option<(f64, f64)> {
        let mut range: Option<(f64, f64)> = None;
        for time in times {
            let camera = deep_at(time);
            if camera.right != 0.0 || camera.up != 0.0 || !camera.ln_scale.is_finite() {
                return None;
            }
            let log2 = camera.ln_scale / std::f64::consts::LN_2;
            range = Some(range.map_or((log2, log2), |(shallowest, deepest)| (shallowest.max(log2), deepest.min(log2))));
        }
        range.filter(|(shallowest, deepest)| shallowest > deepest)
    }

    /// The dive of frames `width` x `height` from the scale `2^shallowest` to `2^deepest` (see `plan`), whose keyframes `render` renders
    pub(super) fn new((width, height): (usize, usize), (shallowest, deepest): (f64, f64), render: impl Fn(crate::rendering::deep::Depth) -> Option<Vec<TrueColor>> + Sync + 'a) -> DeepDive<'a> {
        let count = (shallowest - deepest).ceil().max(0.0) as usize + 1;
        DeepDive { width, height, shallowest: crate::rendering::deep::Depth::from_log2(shallowest), count, render: Box::new(render), kept: std::sync::Mutex::new(Vec::new()), rendering: std::sync::Mutex::new(()), gpu: None }
    }

    /// This dive with `gpu` composing its unturned frames (as `DiveKeyframes::compose_moments` does: the same frames, bit for bit)
    #[must_use]
    pub(super) fn with_gpu(self, gpu: Option<&'a GpuIterator>) -> DeepDive<'a> {
        DeepDive { gpu, ..self }
    }

    /// The number of keyframes
    #[cfg(test)]
    pub(super) const fn count(&self) -> usize {
        self.count
    }

    /// The scale of keyframe `index`: the first one's halved `index` times, exactly
    fn depth(&self, index: usize) -> crate::rendering::deep::Depth {
        crate::rendering::deep::Depth { exponent: self.shallowest.exponent - i32::try_from(index).unwrap_or(i32::MAX), ..self.shallowest }
    }

    /// Keyframe `index`, rendered if it is not kept (`None`: cancelled)
    fn keyframe(&self, index: usize) -> Option<KeyframeImage> {
        let find = || self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().find(|(kept, _)| *kept == index).map(|(_, image)| image.clone());
        if let Some(image) = find() {
            return Some(image);
        }
        let _rendering = self.rendering.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(image) = find() {
            return Some(image); //(Rendered while this one waited)
        }
        let image = std::sync::Arc::new((self.render)(self.depth(index))?);
        //(Without room on the GPU, the CPU composes its frames)
        let on_gpu = self.gpu.and_then(|gpu| gpu.upload_images(std::slice::from_ref(&*image), self.width * 2, self.height * 2).ok()).map(std::sync::Arc::new);
        let mut kept = self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        kept.push((index, (std::sync::Arc::clone(&image), on_gpu.clone())));
        //(The deepest few: the frames go deeper)
        kept.sort_by_key(|(kept, _)| std::cmp::Reverse(*kept));
        kept.truncate(KEPT_KEYFRAMES);
        drop(kept);
        Some((image, on_gpu))
    }

    /// The keyframe that serves the scale `e^ln_scale`: the deepest whose scale is at least the frame's
    fn index_of(&self, ln_scale: f64) -> usize {
        let log2 = ln_scale / std::f64::consts::LN_2;
        ((self.shallowest.log2() - log2 + 1e-9).floor().max(0.0) as usize).min(self.count - 1)
    }

    /// Renders the keyframe after the deepest of `moments` if it is not kept and no keyframe is being rendered (a frame's thread does this after sending its
    /// frame, while the other threads compose the next frames: else they all wait for it when the first of its frames comes). The same keyframe, rendered
    /// earlier: the frames are the same, bit for bit
    pub(super) fn prefetch_after(&self, moments: &[(f64, f64)]) {
        let Some(next) = moments.iter().map(|&(ln_scale, _)| self.index_of(ln_scale) + 1).max().filter(|next| *next < self.count) else {
            return;
        };
        if self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().any(|(kept, _)| *kept == next) {
            return;
        }
        if let Ok(rendering) = self.rendering.try_lock() {
            drop(rendering);
            let _ = self.keyframe(next);
        }
    }

    /// The frame that averages the frames at the moments `(ln_scale, rotation)` (the natural logarithm of the scale), each composed from the keyframe of its
    /// depth, as `DiveKeyframes::compose_moments` composes them on the CPU (`None`: cancelled while a keyframe was rendered)
    pub(super) fn compose_moments(&self, moments: &[(f64, f64)]) -> Option<Vec<TrueColor>> {
        let mut images = Vec::with_capacity(moments.len());
        let mut placed = Vec::with_capacity(moments.len());
        for &(ln_scale, rotation) in moments {
            let log2 = ln_scale / std::f64::consts::LN_2;
            //The keyframe that serves this scale, and a pixel of the frame in its pixels (1 to 2)
            let index = self.index_of(ln_scale);
            let footprint = (log2 - self.depth(index).log2() + 1.0).exp2();
            images.push(self.keyframe(index)?);
            placed.push((index, footprint, rotation));
        }
        let moments: Vec<Moment> = placed.iter().zip(&images).map(|(&(index, footprint, rotation), (image, _))| Moment::with((index, image), footprint, rotation, self.width)).collect();
        //On the GPU when every moment is unturned and of one keyframe that is there (a frame between two keyframes, or turned, goes to the CPU)
        if let (Some(gpu), Some((_, Some(on_gpu)))) = (self.gpu, images.first()) {
            let one_keyframe = moments.iter().all(|moment| moment.index == moments[0].index);
            let unturned: Option<Vec<(usize, Vec<Tap>)>> = moments.iter().map(|moment| moment.gpu_taps(self.height).map(|taps| (0, taps))).collect();
            if let Some(colors) = unturned.filter(|_| one_keyframe).and_then(|unturned| gpu.compose(on_gpu, (self.width, self.height), &unturned).ok()) {
                return Some(colors);
            }
        }
        Some(compose_on_the_cpu(&moments, (self.width, self.height)))
    }
}

/// The frame that averages `moments`, composed on all CPU threads (see `DiveKeyframes::compose_moments`)
fn compose_on_the_cpu(moments: &[Moment], (width, height): (usize, usize)) -> Vec<TrueColor> {
    let (big_width, big_height) = (width * 2, height * 2);
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

/// The four taps inside a pixel, in its own units, in the order their colors are added
const TAPS: [(f64, f64); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];
/// A dive into an exact center from an exponential map (issue #287, an experiment, off by default: see `rendering::exponential_map`): its bands rendered
/// as the frames reach them, and kept only while frames read them. Instead of a keyframe a doubling at twice a frame's size, a band of rings a doubling
/// (about 0.64 times the pixels), finest at the frames' centers; a turned frame costs no more than an unturned one
pub(super) struct ExponentialDive<'a> {
    width: usize,
    height: usize,
    /// The bands' size, and the corner of the shallowest frame (the first row of band 0)
    size: crate::rendering::exponential_map::MapSize,
    top: crate::rendering::deep::Depth,
    /// The band of the deepest frame's corner: the bands below it have fewer columns (see `MapSize::below_the_last`)
    last: usize,
    /// The deepest frame's scale (log2)
    deepest: f64,
    /// The band with its first row at the radius, its size, and the scale (in logarithms) of the deepest frame that reads it, `None` when cancelled
    render: Box<dyn Fn(crate::rendering::deep::Depth, crate::rendering::exponential_map::MapSize, f64) -> Option<Vec<TrueColor>> + Sync + 'a>,
    map: std::sync::RwLock<crate::rendering::exponential_map::ExponentialMap>,
    /// One band is rendered at a time (each with all the threads)
    rendering: std::sync::Mutex<()>,
    /// The first band of every frame being composed: no band from there down is forgotten while it is
    in_flight: std::sync::Mutex<Vec<usize>>,
    taps: crate::rendering::exponential_map::FrameTaps,
}

/// The bands above a frame's corner that an `ExponentialDive` keeps for frames that are still being composed (the frames go deeper)
const KEPT_ABOVE: usize = 2;

impl<'a> ExponentialDive<'a> {
    /// The dive of frames `width` x `height` with scales from `2^shallowest` down to `2^deepest`; `render` renders a band (see `ExponentialDive::render`)
    pub(super) fn new((width, height): (usize, usize), (shallowest, deepest): (f64, f64),
        render: impl Fn(crate::rendering::deep::Depth, crate::rendering::exponential_map::MapSize, f64) -> Option<Vec<TrueColor>> + Sync + 'a) -> ExponentialDive<'a> {
        use crate::rendering::exponential_map::{ExponentialMap, FrameTaps, MapSize};
        let size = MapSize::for_frames((width, height), 1.0);
        let top = crate::rendering::deep::Depth::from_log2(shallowest + Self::corner_log2((width, height)));
        let last = ((shallowest - deepest).max(0.0) + 1e-9).floor() as usize;
        ExponentialDive { width, height, size, top, last, deepest, render: Box::new(render), map: std::sync::RwLock::new(ExponentialMap::new(size, top)), rendering: std::sync::Mutex::new(()), in_flight: std::sync::Mutex::new(Vec::new()), taps: FrameTaps::new((width, height)) }
    }

    /// The log2 of the distance of a frame's corner from its center, in scales (a frame is `2.5 · scale` wide)
    fn corner_log2((width, height): (usize, usize)) -> f64 {
        (2.5 / width as f64 * (width as f64 / 2.0).hypot(height as f64 / 2.0)).log2()
    }

    /// The bands that the frame at the scale `e^ln_scale` reads: from the band of its corner to the deepest that its center reads
    fn bands_of(&self, ln_scale: f64) -> std::ops::RangeInclusive<usize> {
        let corner = ((self.top.log2() - ln_scale / std::f64::consts::LN_2 - Self::corner_log2((self.width, self.height))) + 1e-9).floor().max(0.0) as usize;
        let below = crate::rendering::exponential_map::MapSize::doublings_below_the_corner((self.width, self.height));
        corner.min(self.last)..=corner.min(self.last) + below
    }

    /// Band `band` in the map, rendered if it is not there (false: cancelled)
    fn band(&self, band: usize) -> bool {
        let there = || self.map.read().unwrap_or_else(std::sync::PoisonError::into_inner).has(band);
        if there() {
            return true;
        }
        let _rendering = self.rendering.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if there() {
            return true;
        }
        let size = if band > self.last { self.size.below_the_last(band - self.last) } else { self.size };
        //(The deepest frame that reads it has its corner in it, or it is the last frame)
        let reader_log2 = (self.top.log2() - Self::corner_log2((self.width, self.height)) - (band.min(self.last) + 1) as f64).max(self.deepest);
        //(Its radius by its exponent: `2^-band` underflows 64-bit floats past 10⁻³⁰⁸)
        let top = crate::rendering::deep::Depth { exponent: self.top.exponent - i32::try_from(band).unwrap_or(i32::MAX), ..self.top };
        let Some(colors) = (self.render)(top, size, reader_log2 * std::f64::consts::LN_2) else {
            return false;
        };
        self.map.write().unwrap_or_else(std::sync::PoisonError::into_inner).insert(band, colors);
        true
    }

    /// The frame that averages the frames at the `(ln_scale, rotation)` of every moment (motion blur), each read from the map and rounded to 8 bits before
    /// the average; `None` when a band it needs was cancelled
    pub(super) fn compose_moments(&self, moments: &[(f64, f64)]) -> Option<Vec<TrueColor>> {
        let needed: std::collections::BTreeSet<usize> = moments.iter().flat_map(|&(ln_scale, _)| self.bands_of(ln_scale)).collect();
        let first = needed.first().copied().unwrap_or(0);
        //(Other threads compose other frames at the same time: this frame's bands stay while it is composed, and it reads them all under one lock)
        self.in_flight.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(first);
        let frames = loop {
            {
                let map = self.map.read().unwrap_or_else(std::sync::PoisonError::into_inner);
                if needed.iter().all(|band| map.has(*band)) {
                    break Some(moments.iter().map(|&(ln_scale, rotation)| map.frame(&self.taps, crate::rendering::deep::Depth::from_log2(ln_scale / std::f64::consts::LN_2), rotation)).collect::<Vec<_>>());
                }
            }
            if !needed.iter().all(|band| self.band(*band)) {
                break None;
            }
        };
        //The bands above the first band of every frame being composed, but a few, are read no more (the frames go deeper)
        let lowest = {
            let mut in_flight = self.in_flight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(at) = in_flight.iter().position(|band| *band == first) {
                in_flight.swap_remove(at);
            }
            in_flight.iter().copied().min().unwrap_or(first).min(first)
        };
        if lowest > KEPT_ABOVE {
            let mut map = self.map.write().unwrap_or_else(std::sync::PoisonError::into_inner);
            if map.first_band().is_some_and(|band| band < lowest - KEPT_ABOVE) {
                map.forget_above(lowest - KEPT_ABOVE);
            }
        }
        let frames = frames?;
        let count = frames.len().max(1) as u32;
        Some((0..self.width * self.height).map(|index| {
            let sum = frames.iter().fold([0_u32; 3], |sum, frame| [sum[0] + u32::from(frame[index].red), sum[1] + u32::from(frame[index].green), sum[2] + u32::from(frame[index].blue)]);
            let [red, green, blue] = sum.map(|total| ((total + count / 2) / count) as u8);
            TrueColor::new(red, green, blue)
        }).collect())
    }
}

/// The two columns and the two rows of the taps
const TAPS_X: [f64; 2] = [0.25, 0.75];
const TAPS_Y: [f64; 2] = [0.25, 0.75];

/// One moment of a composed frame: its keyframe, the size of a frame pixel in keyframe pixels and the turn; for an unturned moment the columns of the keyframe
/// that the two taps of every column read, worked out once (the rows are worked out per row)
struct Moment<'a> {
    index: usize,
    image: &'a [TrueColor],
    footprint: f64,
    sin: f64,
    cos: f64,
    columns: Option<[Vec<Tap>; 2]>,
}

impl<'a> Moment<'a> {
    fn new(keyframes: &'a DiveKeyframes, scale: f64, rotation: f64) -> Moment<'a> {
        //The keyframe that serves this scale: the deepest whose scale is at least the frame's
        let index = keyframes.scales.iter().rposition(|keyframe| *keyframe >= scale * (1.0 - 1e-12)).unwrap_or(0);
        //A pixel of the frame in pixels of the keyframe (1 to 2)
        let footprint = (scale / keyframes.scales[index]) * 2.0;
        Moment::with((index, &keyframes.images[index]), footprint, rotation, keyframes.width)
    }

    /// The moment of keyframe `index`, `image`, whose pixel is `footprint` keyframe pixels, turned by `rotation`, of frames `width` wide
    fn with((index, image): (usize, &'a [TrueColor]), footprint: f64, rotation: f64, width: usize) -> Moment<'a> {
        let big_width = width * 2;
        let (sin, cos) = rotation.sin_cos();
        //Unturned, a tap's column depends on its column only (dx · 1 - dy · 0 is dx exactly), so the taps read the same pixels as when worked out each time
        let columns = (sin.to_bits() == 0.0_f64.to_bits() && cos.to_bits() == 1.0_f64.to_bits()).then(|| TAPS_X.map(|tap_x| (0..width).map(|x| Tap::at((x as f64 + tap_x - width as f64 / 2.0) * footprint + big_width as f64 / 2.0, big_width)).collect()));
        Moment { index, image, footprint, sin, cos, columns }
    }

    /// For an unturned moment, its taps as `GpuIterator::compose` takes them: the two taps of every column, then the two of every row, of frames `height` high
    fn gpu_taps(&self, height: usize) -> Option<Vec<Tap>> {
        let big_height = height * 2;
        self.columns.as_ref().map(|columns| {
            let rows = TAPS_Y.map(|tap_y| (0..height).map(move |y| Tap::at((y as f64 + tap_y - height as f64 / 2.0) * self.footprint + big_height as f64 / 2.0, big_height)));
            columns.iter().flatten().copied().chain(rows.into_iter().flatten()).collect()
        })
    }
}

/// Where a tap falls along one side of a keyframe: the two pixels it is between and its share of the second, as `bilinear` works them out (the GPU's type, see
/// `GpuIterator::compose`)
type Tap = rendering::gpu::ComposeTap;

trait TapAt {
    fn at(position: f64, length: usize) -> Tap;
}

impl TapAt for Tap {
    fn at(position: f64, length: usize) -> Tap {
        let position = (position - 0.5).clamp(0.0, (length - 1) as f64);
        let first = position.floor() as usize;
        Tap { first: first as u32, second: (first + 1).min(length - 1) as u32, share: (position - first as f64) as f32 }
    }
}

/// The color of `image` (`width` pixels wide) between the pixels of the taps `column` and `row`, exactly as `bilinear` computes it
fn bilinear_at(image: &[TrueColor], width: usize, column: Tap, row: Tap) -> [f32; 3] {
    let at = |x: usize, y: usize| {
        let color = image[y * width + x];
        [f32::from(color.red), f32::from(color.green), f32::from(color.blue)]
    };
    let (left, right, top, bottom) = (column.first as usize, column.second as usize, row.first as usize, row.second as usize);
    let (a, b, c, d) = (at(left, top), at(right, top), at(left, bottom), at(right, bottom));
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

    /// Times composing frames of a deep dive (`DeepDive::compose_moments`) at 1920 x 1080, one moment a frame (no motion blur), on the CPU and with the GPU: the
    /// share of a frame's time that composing takes in a long deep video (issue #263). Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn time_composing_deep_dive_frames() {
        let (width, height) = (1920, 1080);
        let image: Vec<TrueColor> = (0..width * height * 4).map(|index| TrueColor::new((index % 251) as u8, (index % 241) as u8, (index % 239) as u8)).collect();
        let gpu = GpuIterator::new().ok();
        for gpu in [None, gpu.as_ref()] {
            let dive = DeepDive::new((width, height), (-40.0, -80.0), |_| Some(image.clone())).with_gpu(gpu);
            //(The keyframes first, so only composing is timed)
            for k in 0..4 {
                dive.compose_moments(&[((-40.0 - f64::from(k)) * std::f64::consts::LN_2, 0.0)]).unwrap();
            }
            for rotation in [0.0, 0.3] {
                let started = std::time::Instant::now();
                let frames = 60;
                for frame in 0..frames {
                    let log2 = -40.0 - 3.0 * f64::from(frame) / f64::from(frames);
                    dive.compose_moments(&[(log2 * std::f64::consts::LN_2, rotation)]).unwrap();
                }
                println!("rotation {rotation}, {}: {:.1} ms a frame", if gpu.is_some() { "GPU" } else { "CPU" }, started.elapsed().as_secs_f64() * 1000.0 / f64::from(frames));
            }
        }
    }

    fn psnr(a: &[TrueColor], b: &[TrueColor]) -> f64 {
        let squared = a.iter().zip(b).map(|(a, b)| [(a.red, b.red), (a.green, b.green), (a.blue, b.blue)].iter().map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>()).sum::<f64>() / (3 * a.len()) as f64;
        10.0 * (255.0 * 255.0 / squared.max(1e-12)).log10()
    }

    /// Fast dives against the exponential map on one deep dive (#287): the time per doubling of each, and how close their frames are to the frames rendered
    /// one by one at 4 times the samples (PSNR), on the CPU (Embla trains on the GPU). Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn measure_the_exponential_map_against_the_keyframes_of_fast_dives() {
        let (width, height) = (320, 180);
        let samples = 4;
        let doublings = 6;
        //The place: MANDELBROT_MAP_PLACE="real imaginary log2-of-the-shallowest-scale limit", else near i at 10^-400
        let place = std::env::var("MANDELBROT_MAP_PLACE").ok();
        let (m, orbit, shallowest) = place.map_or_else(|| {
            let (m, orbit) = crate::rendering::exponential_map::tests::deep_place(400);
            (m, orbit, crate::rendering::deep::Depth::from_log2(-400.0 * std::f64::consts::LOG2_10))
        }, |place| {
            let parts: Vec<&str> = place.split_whitespace().collect();
            let log2: f64 = parts[2].parse().unwrap();
            let bits = crate::rendering::deep::bits_for_depth(crate::rendering::deep::Depth::from_log2(log2 - 40.0), 4096);
            let (real, imaginary) = (crate::rendering::deep::Fixed::parse(parts[0], bits).unwrap(), crate::rendering::deep::Fixed::parse(parts[1], bits).unwrap());
            (MandelbrotSet::new(parts[3].parse().unwrap(), 2.0), crate::rendering::deep::SharedOrbit::new(&real, &imaginary), crate::rendering::deep::Depth::from_log2(log2))
        });
        let color_of = |position: f64| m.interpolated_color(position, |count| m.color(TrueColor::new_from_bernstein_polynomials, count));
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let render = |scale: crate::rendering::deep::Depth, (w, h): (usize, usize), samples: u8| {
            let mut colors = Vec::new();
            assert!(crate::rendering::deep::render_layers_on(&m, (&orbit, (0.0, 0.0)), (scale, 0.0), (w, h), samples, &color_of, false, None, &cancel, |layer, _| colors = layer.to_vec()));
            colors
        };
        //The frames: 30 a doubling, and the reference of each
        let scales: Vec<crate::rendering::deep::Depth> = (0..doublings * 30).map(|frame| shallowest.times((-(frame as f64) / 30.0).exp2())).collect();
        let references: Vec<Vec<TrueColor>> = scales.iter().step_by(7).map(|scale| render(*scale, (width, height), samples * 4)).collect();
        //Fast dives: a keyframe a doubling at twice the size, as `DeepDive` renders them (and one more below the last frame)
        let started = std::time::Instant::now();
        let mut keyframe_times = Vec::new();
        let keyframes: Vec<Vec<TrueColor>> = (0..=doublings).map(|k| {
            let keyframe_started = std::time::Instant::now();
            let keyframe = render(shallowest.times((-(k as f64)).exp2()), (width * 2, height * 2), samples);
            keyframe_times.push((keyframe_started.elapsed().as_secs_f64() * 1000.0).round());
            keyframe
        }).collect();
        println!("  keyframe times (ms): {keyframe_times:?}");
        let keyframe_time = started.elapsed().as_secs_f64() / (doublings + 1) as f64;
        let log2 = |scale: crate::rendering::deep::Depth| scale.log2() * std::f64::consts::LN_2;
        let dive = DeepDive::new((width, height), (shallowest.log2(), shallowest.log2() - doublings as f64), |depth| {
            let k = (shallowest.log2() - depth.log2()).round() as usize;
            Some(keyframes[k].clone())
        });
        let fast: Vec<Vec<TrueColor>> = scales.iter().step_by(7).map(|scale| dive.compose_moments(&[(log2(*scale), 0.0)]).unwrap()).collect();
        //The exponential map
        for fineness in [1.0, std::f64::consts::SQRT_2] {
            let size = crate::rendering::exponential_map::MapSize::for_frames((width, height), fineness);
            let top = shallowest.times(2.5 / width as f64 * (width as f64 / 2.0).hypot(height as f64 / 2.0));
            let mut map = crate::rendering::exponential_map::ExponentialMap::new(size, top);
            let started = std::time::Instant::now();
            let bands = doublings + crate::rendering::exponential_map::MapSize::doublings_below_the_corner((width, height)) + 1;
            let mut times = Vec::new();
            for band in 0..bands {
                let band_started = std::time::Instant::now();
                map.insert(band, crate::rendering::exponential_map::render_band(&m, &orbit, top.times((-(band as f64)).exp2()), size, samples, &color_of, false, None, &cancel).unwrap());
                times.push(band_started.elapsed().as_secs_f64());
            }
            let band_time = started.elapsed().as_secs_f64() / bands as f64;
            println!("  band times (ms): {:?}", times.iter().map(|time| (time * 1000.0).round()).collect::<Vec<_>>());
            let started = std::time::Instant::now();
            let taps = crate::rendering::exponential_map::FrameTaps::new((width, height));
            let read: Vec<Vec<TrueColor>> = scales.iter().step_by(7).map(|scale| map.frame(&taps, *scale, 0.0)).collect();
            let compose = started.elapsed().as_secs_f64() / read.len() as f64;
            let mean = |frames: &[Vec<TrueColor>]| frames.iter().zip(&references).map(|(frame, reference)| psnr(frame, reference)).sum::<f64>() / references.len() as f64;
            //(The outer pixels, more than 80% of the way to a corner: where the map is coarsest)
            let outer = |frame: &[TrueColor]| -> Vec<TrueColor> {
                let corner = (width as f64 / 2.0).hypot(height as f64 / 2.0);
                frame.iter().enumerate().filter(|(index, _)| ((index % width) as f64 + 0.5 - width as f64 / 2.0).hypot((index / width) as f64 + 0.5 - height as f64 / 2.0) > 0.8 * corner).map(|(_, color)| *color).collect()
            };
            let mean_outer = |frames: &[Vec<TrueColor>]| frames.iter().zip(&references).map(|(frame, reference)| psnr(&outer(frame), &outer(reference))).sum::<f64>() / references.len() as f64;
            println!("  outer pixels: PSNR {:.2} dB (Fast dives {:.2} dB)", mean_outer(&read), mean_outer(&fast));
            println!("map, fineness {fineness:.2}: {} x {} a doubling ({} pixels, keyframes {}), {:.0} ms a doubling (keyframes {:.0} ms), {:.1} ms to read a frame; PSNR {:.2} dB (Fast dives {:.2} dB)",
                size.columns, size.rows, size.band_pixels(), width * height * 4, band_time * 1000.0, keyframe_time * 1000.0, compose * 1000.0, mean(&read), mean(&fast));
        }
        //The tail: the bands below the last frame's with fewer columns (`MapSize::below_the_last`), against the full bands
        let size = crate::rendering::exponential_map::MapSize::for_frames((width, height), 1.0);
        let top = shallowest.times(2.5 / width as f64 * (width as f64 / 2.0).hypot(height as f64 / 2.0));
        let render_map = |tail: bool| {
            let mut map = crate::rendering::exponential_map::ExponentialMap::new(size, top);
            let mut tail_time = 0.0;
            for band in 0..=doublings + crate::rendering::exponential_map::MapSize::doublings_below_the_corner((width, height)) {
                let band_size = if tail && band > doublings { size.below_the_last(band - doublings) } else { size };
                let started = std::time::Instant::now();
                map.insert(band, crate::rendering::exponential_map::render_band(&m, &orbit, top.times((-(band as f64)).exp2()), band_size, samples, &color_of, false, None, &cancel).unwrap());
                if band > doublings {
                    tail_time += started.elapsed().as_secs_f64();
                }
            }
            (map, tail_time)
        };
        let ((full, full_time), (tailed, tail_time)) = (render_map(false), render_map(true));
        let taps = crate::rendering::exponential_map::FrameTaps::new((width, height));
        let last_frames: Vec<crate::rendering::deep::Depth> = scales.iter().rev().take(30).copied().collect();
        let worst = last_frames.iter().map(|scale| psnr(&full.frame(&taps, *scale, 0.0), &tailed.frame(&taps, *scale, 0.0))).fold(f64::INFINITY, f64::min);
        println!("tail below the last frame: {:.0} ms with full bands, {:.0} ms with fewer columns; the last doubling's frames differ at worst by PSNR {worst:.2} dB", full_time * 1000.0, tail_time * 1000.0);
        //(Which is closer to the frames rendered one by one, in the last doubling)
        let last_doubling: Vec<(usize, &Vec<TrueColor>)> = (0..scales.len()).step_by(7).zip(&references).filter(|(frame, _)| *frame >= (doublings - 1) * 30).collect();
        let against = |map: &crate::rendering::exponential_map::ExponentialMap| last_doubling.iter().map(|(frame, reference)| psnr(&map.frame(&taps, scales[*frame], 0.0), reference)).sum::<f64>() / last_doubling.len() as f64;
        println!("  the last doubling against frame by frame: PSNR {:.2} dB with full bands, {:.2} dB with fewer columns", against(&full), against(&tailed));
    }

    #[test]
    fn a_deep_dive_renders_each_keyframe_once_keeps_a_few_and_composes_like_the_keyframes_of_a_shallow_dive() {
        //#263: keyframes at exact halvings, rendered as the frames reach them; within 64-bit floats the frames are bit for bit those of `DiveKeyframes`
        let (width, height) = (20, 12);
        let image_of = |depth: crate::rendering::deep::Depth| -> Vec<TrueColor> {
            let seed = depth.exponent.unsigned_abs();
            (0..width * height * 4).map(|index| TrueColor::new((index as u32 * 7 + seed * 31) as u8, (index as u32 * 13 + seed) as u8, (seed * 5) as u8)).collect()
        };
        let renders = std::sync::atomic::AtomicUsize::new(0);
        let (shallowest, deepest) = (-3.0, -1100.5);
        let dive = DeepDive::new((width, height), (shallowest, deepest), |depth| {
            renders.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(image_of(depth))
        });
        assert_eq!(dive.count(), 1099);
        //Frames going down, a few per keyframe: every keyframe rendered once, and never more than a few kept
        let mut used = std::collections::BTreeSet::new();
        for step in 0..=((shallowest - deepest) / 0.3) as usize {
            let log2 = shallowest - 0.3 * step as f64;
            dive.compose_moments(&[(log2 * std::f64::consts::LN_2, 0.0)]).unwrap();
            assert!(dive.kept.lock().unwrap().len() <= KEPT_KEYFRAMES);
            used.insert((shallowest - log2 + 1e-9).floor() as usize);
        }
        assert_eq!(renders.load(std::sync::atomic::Ordering::Relaxed), used.len(), "each keyframe that a frame uses, once");
        assert!(used.len() + 1 >= dive.count());
        //Within 64-bit floats: the frames of `DiveKeyframes` with the same keyframes
        let scales: Vec<f64> = (0..20).map(|k| 2f64.powf(shallowest) / f64::from(1_u32 << k)).collect();
        let keyframes = DiveKeyframes { width, height, images: scales.iter().map(|scale| image_of(crate::rendering::deep::Depth::from_f64(*scale))).collect(), scales: scales.clone(), on_gpu: None };
        let fresh = DeepDive::new((width, height), (shallowest, shallowest - 19.0), |depth| Some(image_of(depth)));
        for moments in [vec![(0.1, 0.0)], vec![(0.0049, 0.0), (0.0052, 0.0), (0.0061, 0.3)], vec![(2f64.powf(shallowest) / 4.0, 0.0)], vec![(1e-5, 0.0)]] {
            let expected = keyframes.compose_moments(&moments, None);
            let composed = fresh.compose_moments(&moments.iter().map(|&(scale, rotation)| (scale.ln(), rotation)).collect::<Vec<_>>()).unwrap();
            assert!(composed.iter().zip(&expected).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)), "{moments:?}");
        }
    }

    #[test]
    fn a_deep_dive_renders_the_next_keyframe_ahead_once_and_its_frames_are_the_same() {
        //#263: after a frame, the keyframe after it is rendered ahead (while other threads compose), once; the frames are the ones without it
        let (width, height) = (20, 12);
        let image_of = |depth: crate::rendering::deep::Depth| -> Vec<TrueColor> {
            let seed = depth.exponent.unsigned_abs();
            (0..width * height * 4).map(|index| TrueColor::new((index as u32 * 7 + seed * 31) as u8, (index as u32 * 13 + seed) as u8, (seed * 5) as u8)).collect()
        };
        let renders = std::sync::atomic::AtomicUsize::new(0);
        let (ahead, plain) = (DeepDive::new((width, height), (-3.0, -13.0), |depth| {
            renders.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(image_of(depth))
        }), DeepDive::new((width, height), (-3.0, -13.0), |depth| Some(image_of(depth))));
        let ln = |log2: f64| log2 * std::f64::consts::LN_2;
        let moments = [(ln(-3.2), 0.0)];
        let first = ahead.compose_moments(&moments).unwrap();
        ahead.prefetch_after(&moments);
        assert_eq!(renders.load(std::sync::atomic::Ordering::Relaxed), 2, "the keyframe of the frame and the next one");
        assert!(ahead.kept.lock().unwrap().iter().any(|(index, _)| *index == 1));
        ahead.prefetch_after(&moments);
        let next = ahead.compose_moments(&[(ln(-4.5), 0.0)]).unwrap();
        assert_eq!(renders.load(std::sync::atomic::Ordering::Relaxed), 2, "neither rendered again");
        let same = |a: &[TrueColor], b: &[TrueColor]| a.iter().zip(b).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue));
        assert!(same(&first, &plain.compose_moments(&moments).unwrap()) && same(&next, &plain.compose_moments(&[(ln(-4.5), 0.0)]).unwrap()));
        //(The last keyframe has none after it)
        ahead.prefetch_after(&[(ln(-13.0), 0.0)]);
        assert!(renders.load(std::sync::atomic::Ordering::Relaxed) <= ahead.count());
    }

    #[test]
    fn frames_composed_in_one_pass_are_bit_for_bit_the_frames_composed_one_moment_at_a_time() {
        //#236: the moments of a frame in one pass, and the taps of unturned moments worked out once per column and row, give exactly the frames of before
        let (width, height) = (60, 36);
        let plan = DivePlan { center: (-0.7498, 0.0203), shallowest: 0.01, deepest: 0.003 };
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let keyframes = DiveKeyframes::render(plan, (width, height), 4, None, TrueColor::new_from_bernstein_polynomials, |_| MandelbrotSet::new(400, 2.0), None, false, &cancel, |_, _| {}).unwrap();
        for moments in [vec![(0.008, 0.0)], vec![(0.0049, 0.0), (0.0052, 0.0), (0.0061, 0.0), (0.0099, 0.0)], vec![(0.007, 0.3)], vec![(0.006, 0.0), (0.0062, 0.01), (0.0064, 0.02)]] {
            let one_pass = keyframes.compose_moments(&moments, None);
            let reference = composed_frame_by_frame(&keyframes, &moments);
            assert!(one_pass.iter().zip(&reference).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)), "{moments:?}");
        }
    }

    #[test]
    fn the_gpu_composes_bit_for_bit_the_frames_of_the_cpu() {
        //#236: the GPU composes the unturned frames of a dive; it gives exactly the colors of the CPU (and a turned moment goes to the CPU)
        let Ok(gpu) = GpuIterator::new() else {
            eprintln!("no suitable GPU: skipped");
            return;
        };
        let (width, height) = (90, 50);
        let plan = DivePlan { center: (-0.7498, 0.0203), shallowest: 0.01, deepest: 0.002 };
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let keyframes = DiveKeyframes::render(plan, (width, height), 4, None, TrueColor::new_from_bernstein_polynomials, |_| MandelbrotSet::new(400, 2.0), Some(&gpu), false, &cancel, |_, _| {}).unwrap();
        assert!(keyframes.on_gpu.is_some(), "the keyframes are on the GPU");
        for moments in [vec![(0.008, 0.0)], vec![(0.0021, 0.0), (0.0035, 0.0), (0.0049, 0.0), (0.0052, 0.0), (0.0061, 0.0), (0.0074, 0.0), (0.0088, 0.0), (0.0099, 0.0)], vec![(0.006, 0.0), (0.0062, 0.01)]] {
            let (gpu_frame, cpu_frame) = (keyframes.compose_moments(&moments, Some(&gpu)), keyframes.compose_moments(&moments, None));
            let differ = gpu_frame.iter().zip(&cpu_frame).filter(|(a, b)| (a.red, a.green, a.blue) != (b.red, b.green, b.blue)).count();
            assert_eq!(differ, 0, "{moments:?}");
        }
    }

    #[test]
    fn the_gpu_composes_bit_for_bit_the_frames_of_a_deep_dive_that_the_cpu_composes() {
        //#263: the unturned frames of a deep dive (one keyframe for all moments) on the GPU, the others on the CPU: the same colors either way
        let Ok(gpu) = GpuIterator::new() else {
            eprintln!("no suitable GPU: skipped");
            return;
        };
        let (width, height) = (90, 50);
        let image_of = |depth: crate::rendering::deep::Depth| -> Vec<TrueColor> {
            let seed = depth.exponent.unsigned_abs();
            (0..width * height * 4).map(|index| TrueColor::new((index as u32 * 7 + seed * 31) as u8, (index as u32 * 13 + seed) as u8, (index as u32 / 3 + seed * 5) as u8)).collect()
        };
        let (cpu, on_gpu) = (DeepDive::new((width, height), (-1000.0, -1010.0), |depth| Some(image_of(depth))), DeepDive::new((width, height), (-1000.0, -1010.0), |depth| Some(image_of(depth))).with_gpu(Some(&gpu)));
        let ln = |log2: f64| log2 * std::f64::consts::LN_2;
        for moments in [vec![(ln(-1000.2), 0.0)], vec![(ln(-1003.1), 0.0), (ln(-1003.4), 0.0), (ln(-1003.6), 0.0), (ln(-1003.9), 0.0)], vec![(ln(-1004.5), 0.0), (ln(-1005.5), 0.0)], vec![(ln(-1006.3), 0.2)]] {
            let (gpu_frame, cpu_frame) = (on_gpu.compose_moments(&moments).unwrap(), cpu.compose_moments(&moments).unwrap());
            let differ = gpu_frame.iter().zip(&cpu_frame).filter(|(a, b)| (a.red, a.green, a.blue) != (b.red, b.green, b.blue)).count();
            assert_eq!(differ, 0, "{moments:?}");
        }
        assert!(on_gpu.kept.lock().unwrap().iter().all(|(_, (_, on_gpu))| on_gpu.is_some()), "the keyframes are on the GPU");
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

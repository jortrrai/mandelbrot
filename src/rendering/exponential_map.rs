//! Dive frames from an exponential map (issue #287, the method of Fraktaler 3 / Kalles Fraktaler 2+ with zoomasm, implemented from its description).
//!
//! A dive into a fixed center is rendered once as rings around the center: row `r` of the map is the ring at the radius `top · 2^(-r / rows)`, column `c`
//! the angle `2π · c / columns`. A frame at any depth and turn is then read from the map: a pixel at the distance `d` from the frame's center is the ring at
//! that distance, so going deeper is moving down the rows and turning is moving along the columns. The map is rendered in bands of one doubling
//! (`rows` rows), each a view of its own depth for the deep renderer, so it reaches any depth with a few bands in memory.
//!
//! The map is as fine as a frame at the frame's corners (with `fineness` 1) and finer towards its center, where every frame reads the deeper rows.

use super::deep::{render_points_on, Depth, SharedOrbit};
use crate::coloring::TrueColor;
use crate::mandelbrot_set::MandelbrotSet;

/// The size of an exponential map: its columns (the angles), and its rows per doubling of the radius
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapSize {
    /// The angles around the center
    pub columns: usize,
    /// The rings per doubling of the radius
    pub rows: usize,
}

impl MapSize {
    /// The map of frames `width` x `height`: a column and a row are `1 / fineness` frame pixels at the frame's corners (square map pixels: a row is as tall
    /// as a column is wide at every radius)
    #[must_use]
    pub fn for_frames((width, height): (usize, usize), fineness: f64) -> MapSize {
        let corner = (width as f64 / 2.0).hypot(height as f64 / 2.0);
        //(A multiple of 64, so that the levels of a band halve it evenly, see `ExponentialMap::insert`)
        let columns = ((std::f64::consts::TAU * corner * fineness / 64.0).ceil() as usize) * 64;
        let rows = (columns as f64 * std::f64::consts::LN_2 / std::f64::consts::TAU).ceil() as usize;
        MapSize { columns, rows }
    }

    /// The pixels of a band (one doubling)
    #[must_use]
    pub fn band_pixels(self) -> usize {
        self.columns * self.rows
    }

    /// The size of the band `below` doublings below the band of the last frame's corner: frames read it only nearer their center, at most
    /// `1 / 2^below` of the way to the corner, so it needs that share of the columns (the tail of a dive then costs about one band)
    #[must_use]
    pub fn below_the_last(self, below: usize) -> MapSize {
        MapSize { columns: (self.columns >> below.min(usize::BITS as usize - 1)).max(MIN_COLUMNS), ..self }
    }

    /// The doublings below a frame's corner that its pixels read, down to a tenth of a pixel from its center (the deepest bands a frame needs)
    #[must_use]
    pub fn doublings_below_the_corner((width, height): (usize, usize)) -> usize {
        ((width as f64 / 2.0).hypot(height as f64 / 2.0) / 0.1).log2().ceil() as usize
    }
}

/// The fewest columns of a band (see `MapSize::below_the_last`)
const MIN_COLUMNS: usize = 64;

/// The band of an exponential map whose first row is the ring at the radius `top`: one doubling, `size.rows` rows of `size.columns`.
///
/// Averaged over `samples` samples a pixel (at the jitter of the 64-bit renderer); `None` when `cancel` stopped it
#[allow(clippy::too_many_arguments)] // The orbit, the band and its samples, how to color, the GPU and when to stop
pub fn render_band(m: &MandelbrotSet, orbit: &SharedOrbit, top: Depth, size: MapSize, samples: u8, color_of: &(impl Fn(f64) -> TrueColor + Sync), approximate: bool,
    gpu: Option<(&super::gpu::GpuIterator, bool)>, cancel: &std::sync::atomic::AtomicBool) -> Option<Vec<TrueColor>> {
    let MapSize { columns, rows } = size;
    let offset_of = |index: usize, sample: u8| {
        let (column, row) = (index % columns, index / columns);
        let (jx, jy) = if sample == 0 { (0.5, 0.5) } else { super::jitter::sample_jitter(column, row, sample) };
        let radius = top.mantissa * (-(row as f64 + jy) / rows as f64).exp2();
        let (sin, cos) = (std::f64::consts::TAU * (column as f64 + jx) / columns as f64).sin_cos();
        (radius * cos, radius * sin)
    };
    //Where 64-bit floats still tell the band's pixels apart, the 64-bit loop iterates them (as the 64-bit renderer does: it knows the main cardioid and
    //bulb, which perturbation iterates to the limit); deeper, the deep renderer
    let (center_real, center_imaginary) = orbit.center();
    let center = (center_real.to_f64(), center_imaginary.to_f64());
    let spacing = top.to_f64() / 2.0 * std::f64::consts::TAU / columns as f64;
    let magnitude = (center.0.abs() + top.to_f64()).max(center.1.abs() + top.to_f64());
    if top.to_f64().is_normal() && spacing / (f64::EPSILON * magnitude.max(f64::MIN_POSITIVE)) >= crate::complex_plane::PRECISION_LIMIT_STEPS {
        return render_band_in_64_bits(m, center, (top.exponent, size), samples, color_of, &offset_of, cancel);
    }
    let mut band = None;
    let done = render_points_on(m, orbit, (top.exponent, top.mantissa), (size.band_pixels(), &offset_of), samples, color_of, approximate, gpu, cancel, |layer, number| {
        if number == samples.max(1) {
            band = Some(layer.to_vec());
        }
    });
    band.filter(|_| done)
}

/// `render_band` in 64-bit floats: the points `center + offset_of(point, sample) · 2^exponent`, on all the render threads
fn render_band_in_64_bits(m: &MandelbrotSet, center: (f64, f64), (exponent, size): (i32, MapSize), samples: u8, color_of: &(impl Fn(f64) -> TrueColor + Sync),
    offset_of: &(impl Fn(usize, u8) -> (f64, f64) + Sync), cancel: &std::sync::atomic::AtomicBool) -> Option<Vec<TrueColor>> {
    let pixels = size.band_pixels();
    //(`offset_of` is in units of the band's exponent: back to the plane)
    let unit = f64::from(exponent).exp2();
    let per_thread = pixels.div_ceil(super::render_thread_count().max(1)).max(1);
    let mut sums = vec![[0_u32; 3]; pixels];
    for sample in 0..samples.max(1) {
        std::thread::scope(|scope| {
            for (part, sums) in sums.chunks_mut(per_thread).enumerate() {
                scope.spawn(move || {
                    let start = part * per_thread;
                    let points: Vec<crate::complex::Complex> = (start..start + sums.len()).map(|index| {
                        let (x, y) = offset_of(index, sample);
                        crate::complex::Complex::new(center.0 + x * unit, center.1 + y * unit)
                    }).collect();
                    let mut positions = vec![0.0; points.len()];
                    if m.positions(&points, &mut positions, Some(cancel)) < points.len() {
                        return;
                    }
                    for (sum, position) in sums.iter_mut().zip(positions) {
                        let color = color_of(position);
                        for (total, channel) in sum.iter_mut().zip([color.red, color.green, color.blue]) {
                            *total += u32::from(channel);
                        }
                    }
                });
            }
        });
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
    }
    let count = u32::from(samples.max(1));
    Some(sums.iter().map(|sum| {
        let [red, green, blue] = sum.map(|total| ((total + count / 2) / count) as u8);
        TrueColor::new(red, green, blue)
    }).collect())
}

/// The bands of a dive's exponential map that frames are read from: band `k` starts at the radius `top / 2^k`
pub struct ExponentialMap {
    /// The size of its bands
    pub size: MapSize,
    /// The radius of the first row of band 0
    pub top: Depth,
    /// The bands there are, by number (the frames read only these: a row outside them is the nearest row in them), each with its coarser levels (see
    /// `insert`)
    bands: std::collections::BTreeMap<usize, Vec<Vec<TrueColor>>>,
}

/// How far apart the map's pixels are at most along a ring where a frame reads them, in frame pixels (the taps are half a pixel apart): nearer a frame's
/// center the map is finer than that, and a frame reads a coarser level of it there, its columns averaged two by two as often as it takes (so the
/// center does not alias)
const SPACING: f64 = 0.5;

/// Where a frame's pixel taps the map (as the keyframes of Fast dives are tapped, see `dive_keyframes`): two by two in a pixel
const TAPS: [f64; 2] = [0.25, 0.75];

impl ExponentialMap {
    /// A map of bands of `size`, band 0 starting at the radius `top`, without bands yet
    #[must_use]
    pub fn new(size: MapSize, top: Depth) -> ExponentialMap {
        ExponentialMap { size, top, bands: std::collections::BTreeMap::new() }
    }

    /// Band `band` (of `self.size.rows` rows, of the columns it was rendered with), and its coarser levels: each with half the columns of the one before
    /// (averaged two by two), down to `MIN_COLUMNS / 2`
    pub fn insert(&mut self, band: usize, colors: Vec<TrueColor>) {
        let rows = self.size.rows;
        let mut levels = vec![colors];
        while let Some(finer) = levels.last().filter(|finer| finer.len() / rows >= MIN_COLUMNS && (finer.len() / rows).is_multiple_of(2)) {
            let coarser: Vec<TrueColor> = finer.as_chunks::<2>().0.iter().map(|pair| {
                let mean = |a: u8, b: u8| (u16::from(a) + u16::from(b)).div_ceil(2) as u8;
                TrueColor::new(mean(pair[0].red, pair[1].red), mean(pair[0].green, pair[1].green), mean(pair[0].blue, pair[1].blue))
            }).collect();
            levels.push(coarser);
        }
        self.bands.insert(band, levels);
    }

    /// Whether band `band` is there
    #[must_use]
    pub fn has(&self, band: usize) -> bool {
        self.bands.contains_key(&band)
    }

    /// The first band there is
    #[must_use]
    pub fn first_band(&self) -> Option<usize> {
        self.bands.keys().next().copied()
    }

    /// Takes away the bands above `band` (that no frame reads any more)
    pub fn forget_above(&mut self, band: usize) {
        self.bands = self.bands.split_off(&band);
    }

    /// The frame `width` x `height` of the dive at `scale` (its view `2.5 · scale` wide), turned counterclockwise by `rotation` radians, on all the render
    /// threads, with the taps of `taps` (made for frames of this size)
    #[must_use]
    pub fn frame(&self, taps: &FrameTaps, scale: Depth, rotation: f64) -> Vec<TrueColor> {
        let (width, height) = (taps.width, taps.height);
        let rows = self.size.rows;
        //The row of a tap is its row in a map whose first row is at the frame's pixel, moved down by the doublings from the map's top to the pixel; its
        //angle moves back by the turn (as `deep::turned` turns the offsets of a frame: by -rotation in the plane)
        let log2_pixel = scale.log2() + (2.5 / width as f64).log2();
        let row_shift = (self.top.log2() - log2_pixel) * rows as f64 - 0.5;
        let turn_shift = -rotation / std::f64::consts::TAU;
        //(The rows there are: a row outside them is the nearest row in them. A band has the columns it was rendered with, see `MapSize::below_the_last`)
        let first_band = self.bands.keys().next().copied().unwrap_or(0);
        let last_band = self.bands.keys().next_back().copied().unwrap_or(0);
        //(Every row of the map with its levels, and for each the log2 of the spacing of its finest level along a ring a frame pixel from the center; a band
        //that is not there, between two that are, is the one above it: a frame reads only bands that are there, one after the other)
        let levels_of = (first_band..=last_band).filter_map(|band| self.bands.range(..=band).next_back().map(|(_, levels)| levels));
        let map_rows: Vec<(Vec<&[TrueColor]>, f64)> = levels_of.flat_map(|levels| (0..rows).map(move |row| {
            let columns = levels[0].len() / rows;
            (levels.iter().map(|level| { let columns = level.len() / rows; &level[row * columns..(row + 1) * columns] }).collect(), (std::f64::consts::TAU / columns as f64).log2())
        })).collect();
        let (first, last) = (first_band * rows, first_band * rows + map_rows.len() - 1);
        //The color of row `row` at the angle `turns` (in turns; pixel centers at half columns, wrapping around), between the two columns around it, at the
        //level whose spacing is at most `SPACING` at `2^log2_distance` frame pixels from the center
        let along = |row: usize, turns: f64, log2_distance: f64| -> [f32; 3] {
            let (levels, log2_spacing) = &map_rows[row - first];
            let level = ((SPACING.log2() - log2_spacing - log2_distance).floor().max(0.0) as usize).min(levels.len() - 1);
            let colors = levels[level];
            let columns = colors.len();
            //(`turns` is within one turn, see `at`)
            let u = turns.mul_add(columns as f64, -0.5);
            let u = if u < 0.0 { u + columns as f64 } else { u };
            let column = u.floor();
            let fx = (u - column) as f32;
            let column = (column as usize).min(columns - 1);
            let next = if column + 1 == columns { 0 } else { column + 1 };
            let (a, b) = (colors[column], colors[next]);
            [(a.red, b.red), (a.green, b.green), (a.blue, b.blue)].map(|(a, b)| f32::from(a) * (1.0 - fx) + f32::from(b) * fx)
        };
        //The color at the angle `turns` and the row `v` (continuous, row centers at whole numbers), from the two rows around it as `bilinear` in
        //`dive_keyframes`
        let at = |turns: f64, v: f64, log2_distance: f64| -> [f32; 3] {
            let v = v.clamp(first as f64, last as f64);
            let turns = turns - turns.floor();
            let row = v.floor();
            let fy = (v - row) as f32;
            let row = row as usize;
            let (above, below) = (along(row, turns, log2_distance), along((row + 1).min(last), turns, log2_distance));
            std::array::from_fn(|channel| above[channel] * (1.0 - fy) + below[channel] * fy)
        };
        let mut frame = vec![TrueColor::new(0, 0, 0); width * height];
        let threads = super::render_thread_count().max(1);
        let per_thread = (width * height).div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (out, pixel_taps) in frame.chunks_mut(per_thread).zip(taps.taps.chunks(per_thread * TAPS.len() * TAPS.len())) {
                let at = &at;
                scope.spawn(move || {
                    for (color, pixel_taps) in out.iter_mut().zip(pixel_taps.chunks(TAPS.len() * TAPS.len())) {
                        let mut sum = [0.0_f32; 3];
                        for &(column, row) in pixel_taps {
                            let tap = at(f64::from(column) + turn_shift, f64::from(row) * rows as f64 + row_shift, -f64::from(row));
                            for (total, channel) in sum.iter_mut().zip(tap) {
                                *total += channel;
                            }
                        }
                        let [red, green, blue] = sum.map(|total| (total / 4.0).round() as u8);
                        *color = TrueColor::new(red, green, blue);
                    }
                });
            }
        });
        frame
    }
}

/// Where the taps of the pixels of frames `width` x `height` fall in an exponential map, worked out once for all the frames of a dive.
///
/// For each tap: its angle (in turns) and its distance from the center (in doublings below a pixel)
pub struct FrameTaps {
    width: usize,
    height: usize,
    taps: Vec<(f32, f32)>,
}

impl FrameTaps {
    /// The taps of frames `width` x `height`
    #[must_use]
    pub fn new((width, height): (usize, usize)) -> FrameTaps {
        let mut taps = Vec::with_capacity(width * height * TAPS.len() * TAPS.len());
        for y in 0..height {
            for x in 0..width {
                for tap_y in TAPS {
                    for tap_x in TAPS {
                        let (right, up) = (x as f64 + tap_x - width as f64 / 2.0, height as f64 / 2.0 - y as f64 - tap_y);
                        taps.push(((up.atan2(right) / std::f64::consts::TAU) as f32, (-right.hypot(up).log2()) as f32));
                    }
                }
            }
        }
        FrameTaps { width, height, taps }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::rendering::deep::{bits_for_depth, render_layers_on, Fixed};

    pub fn deep_place(power: usize) -> (MandelbrotSet, SharedOrbit) {
        let bits = bits_for_depth(Depth::from_log2(-(power as f64) * std::f64::consts::LOG2_10 - 40.0), 4096);
        let (real, imaginary) = (Fixed::parse(&format!("3e-{}", power + 1), bits).unwrap(), Fixed::parse(&format!("1.{}2", "0".repeat(power)), bits).unwrap());
        (MandelbrotSet::new(3000, 2.0), SharedOrbit::new(&real, &imaginary))
    }

    fn mean_difference(a: &[TrueColor], b: &[TrueColor]) -> f64 {
        a.iter().zip(b).map(|(a, b)| f64::from(a.red.abs_diff(b.red)) + f64::from(a.green.abs_diff(b.green)) + f64::from(a.blue.abs_diff(b.blue))).sum::<f64>() / (3 * a.len()) as f64
    }

    #[test]
    fn a_map_is_as_fine_as_a_frame_at_its_corners() {
        let size = MapSize::for_frames((1920, 1080), 1.0);
        assert_eq!((size.columns, size.rows), (6976, 770));
        assert_eq!(MapSize::doublings_below_the_corner((1920, 1080)), 14);
    }

    #[test]
    fn frames_read_from_the_map_are_close_to_the_frames_rendered_directly_also_turned_and_past_64_bit_floats() {
        //#287: frames between two doublings at 10^-400, read from the bands below them, against the same frames rendered directly by the deep renderer
        let (width, height) = (96, 64);
        let (mut m, orbit) = deep_place(400);
        //(A cycling palette, so that the structure shows)
        m.palette = crate::mandelbrot_set::ColorCycle { divide: 1.0, offset: 0.0, stretch: false };
        let color_of = |position: f64| m.interpolated_color(position, |count| m.color(TrueColor::new_from_bernstein_polynomials, count));
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let size = MapSize::for_frames((width, height), 1.0);
        //(Near i at 2^-26, with filaments, and at 10^-400)
        for shallowest in [Depth::from_log2(-26.0), Depth::from_log2(-400.0 * std::f64::consts::LOG2_10)] {
            //(The map starts at the corner of the shallowest frame: 2.5 · scale / width a pixel)
            let top = shallowest.times(2.5 / width as f64 * (width as f64 / 2.0).hypot(height as f64 / 2.0));
            let mut map = ExponentialMap::new(size, top);
            for band in 0..MapSize::doublings_below_the_corner((width, height)) + 2 {
                map.insert(band, render_band(&m, &orbit, top.times((-(band as f64)).exp2()), size, 4, &color_of, false, None, &cancel).unwrap());
            }
            for (doublings, rotation) in [(0.0_f64, 0.0), (0.4, 0.0), (1.7, 0.0), (0.9, 1.0)] {
                let scale = shallowest.times((-doublings).exp2());
                let read = map.frame(&FrameTaps::new((width, height)), scale, rotation);
                let mut direct = Vec::new();
                assert!(render_layers_on(&m, (&orbit, (0.0, 0.0)), (scale, rotation), (width, height), 16, &color_of, false, None, &cancel, |layer, _| direct = layer.to_vec()));
                let difference = mean_difference(&read, &direct);
                assert!(difference < 12.0, "{doublings} doublings down, turned {rotation}: mean difference {difference}");
            }
        }
    }

    /// Times reading 1920 x 1080 frames from a map (of made-up colors), unturned and turned: compare `time_composing_deep_dive_frames` of Fast dives.
    /// Run by hand with --release --ignored --nocapture
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn time_reading_frames_from_a_map() {
        let (width, height) = (1920, 1080);
        let size = MapSize::for_frames((width, height), 1.0);
        let band: Vec<TrueColor> = (0..size.band_pixels()).map(|index| TrueColor::new((index % 251) as u8, (index % 241) as u8, (index % 239) as u8)).collect();
        let top = Depth::from_log2(-40.0);
        let mut map = ExponentialMap::new(size, top);
        for k in 0..=MapSize::doublings_below_the_corner((width, height)) + 3 {
            map.insert(k, band.clone());
        }
        let started = std::time::Instant::now();
        let taps = FrameTaps::new((width, height));
        println!("taps worked out in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);
        let shallowest = top.times(width as f64 / 2.5 / (width as f64 / 2.0).hypot(height as f64 / 2.0));
        for rotation in [0.0, 0.3] {
            let started = std::time::Instant::now();
            let frames = 60;
            for frame in 0..frames {
                std::hint::black_box(map.frame(&taps, shallowest.times((-3.0 * f64::from(frame) / f64::from(frames)).exp2()), rotation));
            }
            println!("rotation {rotation}: {:.1} ms a frame", started.elapsed().as_secs_f64() * 1000.0 / f64::from(frames));
        }
    }
}

//! Thumbnails of the points of interest, rendered on demand with the renderer of the window.
//!
//! The cost is kept flat, however many points of interest there are:
//! * only the thumbnails that are **visible** in the list are generated (the window reports the visible ones every frame, see `MandelbrotApp::points_ui`);
//! * one thumbnail is rendered at a time, on a background thread, and never while the main view is rendering: starting a render cancels the thumbnail that is being rendered;
//! * at most `MAX_CACHED` thumbnails are kept (`Lru`): the least recently shown one is dropped, and shown again later, it is rendered again;
//! * a thumbnail never iterates beyond `MAX_THUMBNAIL_ITERATIONS`, so a place with an enormous limit cannot keep the CPUs busy.
//!
//! A thumbnail shows the place as going to it would show it: the location of the file (center, scale, maximum iterations) with the appearance that going to the place would use
//! (the saved one when the window restores appearances, otherwise the current one). It is rendered with `THUMBNAIL_SAMPLES` samples per pixel, whatever the supersampling of the window is.
//! Thumbnails are not stored on disk: ten of them take less than a tenth of a second to render (see `examples/thumbnails.rs`), and a stored image would go stale whenever the renderer changes.
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use crate::coloring::{ColorChannelMapping, TrueColor};
use crate::complex_plane::{ComplexPlane, View};
use crate::mandelbrot_set::{ColorCycle, MandelbrotSet};
use crate::points_of_interest::{Coloring, Location};
use crate::rendering::Render;

/// The most thumbnails that are kept at the same time
pub const MAX_CACHED: usize = 64;
/// The width of a thumbnail in pixels; the height follows the aspect ratio of the window
pub const THUMBNAIL_WIDTH: usize = 144;
/// Samples per pixel of a thumbnail
pub const THUMBNAIL_SAMPLES: u8 = 4;
/// A thumbnail iterates at most this long, even if the file asks for more
pub const MAX_THUMBNAIL_ITERATIONS: u32 = 20_000;
/// Bump this when the renderer or the coloring changes what a place looks like, so that nothing keeps showing the old look
pub const RENDER_REVISION: u32 = 1;

/// The shortest thumbnail in pixels, for a very wide window
pub const MIN_THUMBNAIL_HEIGHT: usize = 32;
/// The tallest thumbnail in pixels, for a square or tall window: the rows of the list are 100 points high
pub const MAX_THUMBNAIL_HEIGHT: usize = 96;

/// The height of a thumbnail for a window of `window_width` x `window_height` pixels: it follows the aspect ratio of the window, from `MIN_THUMBNAIL_HEIGHT` to `MAX_THUMBNAIL_HEIGHT`
pub fn thumbnail_height(window_width: usize, window_height: usize) -> usize {
    let height = ((THUMBNAIL_WIDTH * window_height.max(1)) as f64 / window_width.max(1) as f64).round() as usize;
    height.clamp(MIN_THUMBNAIL_HEIGHT, MAX_THUMBNAIL_HEIGHT)
}

/// What a thumbnail shows
#[derive(Clone, Debug, PartialEq)]
pub struct ThumbnailRequest {
    /// The place: the center, the scale and the maximum iterations (capped by `MAX_THUMBNAIL_ITERATIONS`)
    pub location: Location,
    /// The coloring of the thumbnail
    pub coloring: Coloring,
    /// The color channel mapping of the thumbnail
    pub color_channels: ColorChannelMapping,
    /// The color cycle of the thumbnail, see `ColorCycle`
    pub palette: ColorCycle,
    /// The width in pixels
    pub width: usize,
    /// The height in pixels
    pub height: usize,
}

impl ThumbnailRequest {
    /// A key that is the same exactly when the thumbnail would look the same: it changes with the file's numbers, the appearance, the size and `RENDER_REVISION`
    pub fn key(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        feed(&self.location.real.to_bits().to_le_bytes());
        feed(&self.location.imaginary.to_bits().to_le_bytes());
        feed(&self.location.scale.to_bits().to_le_bytes());
        feed(&self.location.max_iterations.min(MAX_THUMBNAIL_ITERATIONS).to_le_bytes());
        feed(self.coloring.key().as_bytes());
        feed(format!("{:?}", self.color_channels).as_bytes());
        feed(&self.palette.divide.to_bits().to_le_bytes());
        feed(&self.palette.offset.to_bits().to_le_bytes());
        feed(&[u8::from(self.palette.stretch)]);
        feed(&(self.width as u64).to_le_bytes());
        feed(&(self.height as u64).to_le_bytes());
        feed(&RENDER_REVISION.to_le_bytes());
        hash
    }
}

/// Renders the thumbnail: `width * height` colors, row by row, with the channel mapping not yet applied; `None` if `cancel` was set before it was finished.
/// Deterministic, like every render.
pub fn render_thumbnail(request: &ThumbnailRequest, orbit_radius: f64, cancel: &AtomicBool) -> Option<Vec<TrueColor>> {
    let mut plane = ComplexPlane::new(request.width, request.height);
    plane.set_view(&View::new(request.location.real, request.location.imaginary, request.location.scale));
    let mut set = MandelbrotSet::new(request.location.max_iterations.min(MAX_THUMBNAIL_ITERATIONS), orbit_radius);
    set.palette = request.palette;
    let coloring_function = match request.coloring {
        Coloring::Hsv => TrueColor::new_from_hsv_colors,
        Coloring::Bernstein => TrueColor::new_from_bernstein_polynomials,
    };
    let mut last = None;
    let completed = Render::new(&plane, &set, request.width, request.height, coloring_function).supersampling(THUMBNAIL_SAMPLES).cancel_with(cancel).run( |colors, _| {
        last = Some(colors.to_vec());
    });
    if completed { last } else { None }
}

/// A cache that keeps at most `capacity` values and drops the one that was used longest ago
pub struct Lru<V> {
    capacity: usize,
    tick: u64,
    entries: HashMap<u64, (V, u64)>,
}

impl<V> Lru<V> {
    /// Creates a cache that keeps at most `capacity` values (at least one)
    pub fn new(capacity: usize) -> Lru<V> {
        Lru { capacity: capacity.max(1), tick: 0, entries: HashMap::new() }
    }

    /// The amount of values in the cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether there is a value for `key`, without counting it as used
    pub fn contains(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }

    /// Gets a value and counts it as used now
    pub fn get(&mut self, key: u64) -> Option<&V> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(&key).map(|(value, used)| {
            *used = tick;
            &*value
        })
    }

    /// Adds a value (counted as used now), and drops the least recently used ones while there are more than `capacity`. Returns what was dropped.
    pub fn insert(&mut self, key: u64, value: V) -> Vec<V> {
        self.tick += 1;
        self.entries.insert(key, (value, self.tick));
        let mut dropped = Vec::new();
        while self.entries.len() > self.capacity {
            let Some(oldest) = self.entries.iter().min_by_key(|(_, (_, used))| *used).map(|(key, _)| *key) else { break };
            if let Some((value, _)) = self.entries.remove(&oldest) {
                dropped.push(value);
            }
        }
        dropped
    }

    /// Removes all values
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ThumbnailRequest {
        ThumbnailRequest { location: Location { real: -0.7498100000000001, imaginary: -0.020300000000000054, scale: 0.02, max_iterations: 300 }, coloring: Coloring::Bernstein, color_channels: ColorChannelMapping::RGB, palette: ColorCycle::CYCLE, width: 36, height: 24 }
    }

    #[test]
    fn the_height_follows_the_window_aspect_ratio() {
        assert_eq!(thumbnail_height(1200, 800), 96);
        assert_eq!(thumbnail_height(800, 800), MAX_THUMBNAIL_HEIGHT, "a square window would need 144, which does not fit in the row");
        assert_eq!(thumbnail_height(2000, 400), MIN_THUMBNAIL_HEIGHT);
        assert_eq!(thumbnail_height(0, 0), MAX_THUMBNAIL_HEIGHT);
    }

    #[test]
    fn the_key_changes_with_everything_that_changes_the_picture_and_only_with_that() {
        let base = request();
        assert_eq!(base.key(), request().key());
        let mut other = request();
        other.location.real += 1e-15;
        assert_ne!(other.key(), base.key());
        let mut other = request();
        other.location.scale *= 2.0;
        assert_ne!(other.key(), base.key());
        let mut other = request();
        other.location.max_iterations = 301;
        assert_ne!(other.key(), base.key());
        let mut other = request();
        other.coloring = Coloring::Hsv;
        assert_ne!(other.key(), base.key());
        let mut other = request();
        other.color_channels = ColorChannelMapping::BGR;
        assert_ne!(other.key(), base.key());
        let mut other = request();
        other.width = 37;
        assert_ne!(other.key(), base.key());
        let (mut high, mut higher) = (request(), request());
        (high.location.max_iterations, higher.location.max_iterations) = (MAX_THUMBNAIL_ITERATIONS, MAX_THUMBNAIL_ITERATIONS * 50);
        assert_eq!(high.key(), higher.key(), "iterations beyond the cap do not change the thumbnail");
    }

    #[test]
    fn a_thumbnail_is_deterministic_has_the_requested_size_and_shows_the_place() {
        let never = AtomicBool::new(false);
        let first = render_thumbnail(&request(), 2.0, &never).unwrap();
        assert_eq!(first.len(), 36 * 24);
        let second = render_thumbnail(&request(), 2.0, &never).unwrap();
        assert!(first.iter().zip(&second).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)), "rendering is deterministic");
        let distinct: std::collections::HashSet<(u8, u8, u8)> = first.iter().map(|c| (c.red, c.green, c.blue)).collect();
        assert!(distinct.len() > 10, "a boundary view is not a flat color: {}", distinct.len());
    }

    #[test]
    fn a_cancelled_thumbnail_gives_nothing() {
        assert!(render_thumbnail(&request(), 2.0, &AtomicBool::new(true)).is_none());
    }

    #[test]
    fn the_cache_never_holds_more_than_its_capacity_and_drops_the_least_recently_used() {
        let mut cache: Lru<u32> = Lru::new(3);
        for key in 0..3 {
            assert_eq!(cache.insert(key, key as u32), Vec::<u32>::new());
        }
        assert_eq!(cache.get(0), Some(&0), "0 is used again, so 1 is now the oldest");
        assert_eq!(cache.insert(3, 3), vec![1]);
        assert!(cache.contains(0) && !cache.contains(1) && cache.contains(2) && cache.contains(3));
        for key in 10..1000 {
            cache.insert(key, key as u32);
            assert!(cache.len() <= 3, "the cache stays bounded");
        }
        assert!(cache.contains(999) && cache.contains(998) && cache.contains(997));
        cache.clear();
        assert!(cache.is_empty());
    }
}

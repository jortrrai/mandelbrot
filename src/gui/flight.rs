//! A flight of the camera from one view to another: smooth panning and zooming, used to go to a point of interest.
//!
//! A leg follows the path of van Wijk and Nuij, "Smooth and efficient zooming and panning" (IEEE Symposium on Information Visualization, 2003): to travel far it zooms out first and in again,
//! so the picture moves at the same apparent speed at every zoom, and the length `S` of the path measures how much the view changes. A flight always takes this shortest
//! path. Its duration follows from the length at the speed of its pace, and the pace eases the start and the end.

use super::spiral::SpiralPath;
use std::time::Duration;

use crate::complex_plane::View;

/// The curvature of the path: how much the camera zooms out to travel, the value that van Wijk and Nuij found pleasant (about sqrt(2))
const RHO: f64 = 1.42;

/// How fast and how smoothly a flight moves
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    Fast,
    Normal,
    Cinematic,
    /// No flight: going to a place shows it at once (see `MandelbrotApp::go_to_point_of_interest`)
    NoFlight,
}

impl Pace {
    /// Every pace, in the order of the drop-down (the default, see `MandelbrotApp::new`, first)
    pub const ALL: [Pace; 4] = [Pace::Fast, Pace::Normal, Pace::Cinematic, Pace::NoFlight];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            Pace::Fast => "Fast",
            Pace::Normal => "Normal",
            Pace::Cinematic => "Cinematic",
            Pace::NoFlight => "No flight",
        }
    }

    /// The speed in units of path length per second, and the shortest and longest duration of a flight
    const fn speed(self) -> (f64, f64, f64) {
        match self {
            Pace::Fast => (4.0, 0.5, 3.0),
            Pace::Normal => (2.0, 1.0, 6.0),
            Pace::Cinematic => (1.0, 2.0, 12.0),
            Pace::NoFlight => (f64::INFINITY, 0.0, 0.0), //A flight of no time: arrived at once
        }
    }

    /// The share of the path that is done after the share `t` (0 to 1) of the time: smooth at both ends (smootherstep for cinematic, smoothstep otherwise)
    fn ease(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Pace::Cinematic => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
            Pace::Fast | Pace::Normal => t * t * (3.0 - 2.0 * t),
            Pace::NoFlight => 1.0,
        }
    }
}

/// How complete the frames of a flight have to be before the camera moves on
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlightRendering {
    /// The camera follows the clock, and the screen shows whatever the render has so far (the image moved and scaled, with black where nothing is rendered yet)
    BestEffort,
    /// The camera waits for a complete low-resolution frame (a preview of the whole view, or a full layer)
    LowRes,
    /// The camera waits for a frame of this resolution, in percent of the view's pixels across: below 100 a preview of blocks of `100 / percent` pixels (25: blocks of
    /// 4 x 4), 100 a full layer (full resolution), above 100 `(percent / 100)²` supersampling layers (200: 4 samples per pixel)
    Resolution(u16),
    /// Every frame is rendered with all its supersampling layers, and the camera waits until the last one is done: the finished image of every view on the way, the slowest
    SuperSampled,
    /// The camera waits for a frame of the resolution that keeps this many frames per second: the flight job chooses it frame by frame from how long the frames take
    /// (see `auto_block`), from full resolution down to blocks of `AUTO_COARSEST` pixels; never supersampled
    Auto(u16),
}

/// The coarsest frames of `FlightRendering::Auto`: blocks of 8 x 8 pixels (12.5%)
pub const AUTO_COARSEST: usize = 8;

/// The block size of the next frame of `FlightRendering::Auto` (1: full resolution): the finest whose expected time, `full` (the time of a full-resolution frame,
/// estimated from the frames so far) divided by its pixels, fits the time of a frame at the target rate (`target`). A finer block than the `current` one must fit with
/// 15% to spare, so a view whose time is right at a boundary does not switch back and forth every frame.
pub fn auto_block(full: Duration, target: Duration, current: usize) -> usize {
    let fits = |block: usize, spare: f64| full.as_secs_f64() / (block * block) as f64 <= target.as_secs_f64() * spare;
    (1..=AUTO_COARSEST).find(|block| fits(*block, if *block < current { 0.85 } else { 1.0 })).unwrap_or(AUTO_COARSEST)
}

impl FlightRendering {
    /// A full layer of every view on the way (100%): sharp all the way, slow where the view is expensive
    pub const FULL_RESOLUTION: FlightRendering = FlightRendering::Resolution(100);

    /// The modes of the drop-down; any other resolution can be typed in
    pub const ALL: [FlightRendering; 7] = [FlightRendering::BestEffort, FlightRendering::LowRes, FlightRendering::Resolution(25), FlightRendering::Resolution(50), FlightRendering::FULL_RESOLUTION, FlightRendering::SuperSampled, FlightRendering::Auto(60)];

    /// The name of the mode: the drop-down shows it, with the resolution in percent next to it for `Resolution`
    pub const fn name(self) -> &'static str {
        match self {
            FlightRendering::BestEffort => "Best effort",
            FlightRendering::LowRes => "Low res",
            FlightRendering::Resolution(_) => "Resolution",
            FlightRendering::SuperSampled => "Super sampled",
            FlightRendering::Auto(_) => "Auto",
        }
    }

    /// The name of an entry of the drop-down: the presets of `Resolution` by their resolution
    pub const fn entry(self) -> &'static str {
        match self {
            FlightRendering::Resolution(25) => "Quarter resolution",
            FlightRendering::Resolution(50) => "Half resolution",
            FlightRendering::Resolution(100) => "Full resolution",
            mode => mode.name(),
        }
    }
}

/// A view as the path sees it: the center and the width (the scale of the view)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub x: f64,
    pub y: f64,
    pub scale: f64,
}

impl Camera {
    /// The view of this camera
    pub const fn view(self) -> View {
        View::new(self.x, self.y, self.scale)
    }
}

/// One leg of a flight: the zoom-and-pan path from `from` to `to`
#[derive(Clone, Debug)]
struct Leg {
    from: Camera,
    to: Camera,
    /// The length of the path
    length: f64,
    /// The distance between the centers, and the parameters r0 of the path (see `camera_at`); `None` for a pure zoom (the centers are the same)
    pan: Option<(f64, f64)>,
}

impl Leg {
    fn new(from: Camera, to: Camera) -> Leg {
        let (w0, w1) = (from.scale, to.scale);
        let u1 = (to.x - from.x).hypot(to.y - from.y);
        //Below a ten-thousandth of the smaller view, a pan is a pure zoom: the formulas of a pan divide by the distance
        if u1 <= 1e-4 * w0.min(w1) {
            return Leg { from, to, length: (w1 / w0).ln().abs() / RHO, pan: None };
        }
        let rho2 = RHO * RHO;
        let b = |w: f64, sign: f64| (w1 * w1 - w0 * w0 + sign * rho2 * rho2 * u1 * u1) / (2.0 * w * rho2 * u1);
        //r = ln(-b + sqrt(b² + 1)) = -asinh(b), without the cancellation of the logarithm for a large b
        let (r0, r1) = (-b(w0, 1.0).asinh(), -b(w1, -1.0).asinh());
        Leg { from, to, length: (r1 - r0) / RHO, pan: Some((u1, r0)) }
    }

    /// The camera at path length `s` (0 to `length`); exactly the start at 0 and the end at `length`
    fn camera_at(&self, s: f64) -> Camera {
        if s <= 0.0 {
            return self.from;
        }
        if s >= self.length {
            return self.to;
        }
        let (from, to, w0) = (self.from, self.to, self.from.scale);
        match self.pan {
            None => {
                let scale = w0 * ((to.scale / w0).ln() * s / self.length).exp();
                let along = s / self.length; //A tiny pan, spread over the zoom
                Camera { x: from.x + (to.x - from.x) * along, y: from.y + (to.y - from.y) * along, scale }
            }
            Some((u1, r0)) => {
                let rs = RHO * s + r0;
                let u = w0 / (RHO * RHO) * (r0.cosh() * rs.tanh() - r0.sinh());
                let scale = w0 * r0.cosh() / rs.cosh();
                let along = u / u1;
                Camera { x: from.x + (to.x - from.x) * along, y: from.y + (to.y - from.y) * along, scale }
            }
        }
    }
}

/// The route of a flight: the way the camera takes, see `Flight::with_route`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The shortest path: zooming out only as far as the distance needs (van Wijk and Nuij)
    Shortest,
    /// Spiraling into the place the way the set spirals there, see `spiral`
    Spiral,
}

impl Route {
    /// Every route, in the order of the drop-down (the default first)
    pub const ALL: [Route; 2] = [Route::Shortest, Route::Spiral];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            Route::Shortest => "Shortest",
            Route::Spiral => "Spiral",
        }
    }
}

/// The path of a flight
#[derive(Clone, Debug)]
enum Path {
    Shortest(Leg),
    Spiral(SpiralPath),
}

/// A flight from one view to another, see the module
#[derive(Clone, Debug)]
pub struct Flight {
    path: Path,
    /// The length of the shortest path, which sets the duration (the spiral takes as long, so the paces mean the same)
    length: f64,
    pace: Pace,
    duration: Duration,
}

impl Flight {
    /// A flight from `from` to `to` at `pace`, by the shortest path: zooming out only as far as the distance needs
    pub fn new(from: Camera, to: Camera, pace: Pace) -> Flight {
        let leg = Leg::new(from, to);
        let (speed, shortest, longest) = pace.speed();
        let duration = Duration::from_secs_f64((leg.length / speed).clamp(shortest, longest));
        Flight { length: leg.length, path: Path::Shortest(leg), pace, duration }
    }

    /// A flight by `route`; with `Route::Spiral` also whether the spiral was possible (without a spiral center near the destination, or with a start too far from
    /// it, the flight takes the shortest path)
    pub fn with_route(from: Camera, to: Camera, pace: Pace, route: Route) -> (Flight, bool) {
        let mut flight = Flight::new(from, to, pace);
        if route == Route::Spiral {
            match SpiralPath::new(from, to) {
                Some(spiral) => flight.path = Path::Spiral(spiral),
                None => return (flight, false),
            }
        }
        (flight, true)
    }

    /// How long the flight takes
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    /// The camera `elapsed` after the start: the start view at 0, exactly the destination from `duration` on
    pub fn camera_at(&self, elapsed: Duration) -> Camera {
        let share = self.pace.ease(elapsed.as_secs_f64() / self.duration.as_secs_f64());
        match &self.path {
            Path::Shortest(leg) => leg.camera_at(share * self.length),
            Path::Spiral(spiral) => spiral.camera_at(share),
        }
    }

    /// The share of the path (0 to 1) that the camera has covered `elapsed` after the start, as the progress bar shows it
    pub fn share_of_path(&self, elapsed: Duration) -> f64 {
        self.pace.ease(elapsed.as_secs_f64() / self.duration.as_secs_f64())
    }

    /// Whether the flight has arrived `elapsed` after the start
    pub fn arrived(&self, elapsed: Duration) -> bool {
        elapsed >= self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESET: Camera = Camera { x: -0.75, y: 0.0, scale: 1.0 };
    const SEAHORSE: Camera = Camera { x: -0.7498, y: 0.0203, scale: 0.00006 };
    const DEEP: Camera = Camera { x: -0.743_643_887_037_151, y: 0.131_825_904_205_33, scale: 1e-12 };

    fn seconds(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    #[test]
    fn a_flight_starts_at_the_start_and_ends_exactly_at_the_destination() {
        for pace in Pace::ALL.into_iter().filter(|pace| *pace != Pace::NoFlight) {
            for (from, to) in [(RESET, SEAHORSE), (SEAHORSE, DEEP), (DEEP, RESET), (SEAHORSE, SEAHORSE)] {
                let flight = Flight::new(from, to, pace);
                assert_eq!(flight.camera_at(Duration::ZERO), from, "{pace:?}");
                assert_eq!(flight.camera_at(flight.duration()), to, "{pace:?}");
                assert_eq!(flight.camera_at(flight.duration() * 2), to);
                assert!(flight.arrived(flight.duration()) && !flight.arrived(flight.duration() / 2));
            }
        }
    }

    #[test]
    fn a_flight_zooms_out_only_as_far_as_the_distance_needs() {
        let widest = |flight: &Flight| (0..=1000).map(|i| flight.camera_at(flight.duration().mul_f64(f64::from(i) / 1000.0)).scale).fold(0.0, f64::max);
        let widest_shortest = widest(&Flight::new(SEAHORSE, DEEP, Pace::Normal));
        //About as wide as the distance between the two points, to keep both in view, and far less than the whole set
        let distance = (DEEP.x - SEAHORSE.x).hypot(DEEP.y - SEAHORSE.y);
        assert!(widest_shortest > 0.5 * distance && widest_shortest < 2.0 * distance, "zooms out only as far as the distance needs: {widest_shortest} for {distance}");
    }

    #[test]
    fn the_picture_moves_smoothly_without_jumps() {
        //Between two frames (1/60 s) the view moves and zooms by a small part of itself, at every depth (at most 35% zoom, for a zoom by 10^12 at the fast pace)
        for (from, to) in [(RESET, DEEP), (DEEP, SEAHORSE)] {
            for pace in Pace::ALL {
                let flight = Flight::new(from, to, pace);
                let frames = (flight.duration().as_secs_f64() * 60.0).ceil() as u32;
                for frame in 0..frames {
                    let (a, b) = (flight.camera_at(seconds(f64::from(frame) / 60.0)), flight.camera_at(seconds(f64::from(frame + 1) / 60.0)));
                    let zoom = (b.scale / a.scale).ln().abs();
                    let pan = (b.x - a.x).hypot(b.y - a.y) / a.scale.min(b.scale);
                    assert!(zoom < 0.3 && pan < 0.6, "{pace:?} frame {frame}: zoom {zoom}, pan {pan} views");
                }
            }
        }
    }

    #[test]
    fn faster_paces_take_less_time_within_their_limits() {
        for (from, to) in [(RESET, SEAHORSE), (SEAHORSE, DEEP), (RESET, Camera { x: -0.74, y: 0.0, scale: 0.9 })] {
            let durations: Vec<f64> = [Pace::Fast, Pace::Normal, Pace::Cinematic].map(|pace| Flight::new(from, to, pace).duration().as_secs_f64()).to_vec();
            assert!(durations[0] <= durations[1] && durations[1] <= durations[2], "{durations:?}");
            assert!(durations[0] >= 0.5 && durations[2] <= 12.0, "{durations:?}");
        }
        //Even a zoom by a factor 10^12 is capped
        assert_eq!(Flight::new(RESET, DEEP, Pace::Cinematic).duration(), seconds(12.0));
    }

    #[test]
    fn the_share_of_the_path_goes_from_0_to_1_without_going_back() {
        let flight = Flight::new(RESET, SEAHORSE, Pace::Cinematic);
        let shares: Vec<f64> = (0..=100).map(|i| flight.share_of_path(flight.duration().mul_f64(f64::from(i) / 100.0))).collect();
        assert_eq!((shares[0], shares[100]), (0.0, 1.0));
        assert!(shares.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn the_drop_downs_list_the_defaults_first() {
        assert_eq!(Pace::ALL[0], Pace::Fast);
    }
}

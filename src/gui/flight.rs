//! A flight of the camera from one view to another: smooth panning and zooming, used to go to a point of interest.
//!
//! A leg follows the path of van Wijk and Nuij, "Smooth and efficient zooming and panning" (IEEE Symposium on Information Visualization, 2003): to travel far it zooms out first and in again,
//! so the picture moves at the same apparent speed at every zoom, and the length `S` of the path measures how much the view changes. A flight always takes this shortest
//! path. Its duration follows from the length at the speed of its pace, and the pace eases the start and the end.
//!
//! Two options change that: the *Constant* pace moves along the path at one speed from start to end (no easing), at a speed of the user's choice, and the
//! *Straight* route goes in a straight line, zooming steadily from one scale to the other without zooming out on the way (see `Route`).

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
    /// One speed from start to end, without easing: the speed of the slider next to the drop-down (see `MandelbrotApp::constant_speed`)
    Constant,
    /// No flight: going to a place shows it at once (see `MandelbrotApp::go_to_point_of_interest`)
    NoFlight,
}

/// The speed of the `Pace::Constant` slider at first, in doublings of the zoom per second
pub const CONSTANT_SPEED: f64 = 3.0;

/// The range of the `Pace::Constant` slider, in doublings of the zoom per second
pub const CONSTANT_SPEEDS: std::ops::RangeInclusive<f64> = 0.25..=20.0;

/// The way a flight takes from one view to another
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The shortest path of van Wijk and Nuij: zooming out (only as far as the distance needs) to travel, see the module
    Shortest,
    /// A straight line: the center moves straight to the destination while the scale changes steadily from one view to the other, never zooming out on the way.
    /// The center moves at one speed on the screen, so it travels most where the view is widest; far apart at a deep zoom, the picture rushes by
    Straight,
}

impl Route {
    /// Every route, in the order of the drop-down (the default first)
    pub const ALL: [Route; 2] = [Route::Shortest, Route::Straight];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            Route::Shortest => "Shortest",
            Route::Straight => "Straight",
        }
    }
}

impl Pace {
    /// Every pace, in the order of the drop-down (the default, see `MandelbrotApp::new`, first)
    pub const ALL: [Pace; 5] = [Pace::Fast, Pace::Normal, Pace::Cinematic, Pace::Constant, Pace::NoFlight];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            Pace::Fast => "Fast",
            Pace::Normal => "Normal",
            Pace::Cinematic => "Cinematic",
            Pace::Constant => "Constant",
            Pace::NoFlight => "No flight",
        }
    }

    /// The speed in units of path length per second, and the shortest and longest duration of a flight; `constant` is the speed of `Pace::Constant`, in doublings
    /// of the zoom per second (a zoom by 2 is `ln 2 / RHO` of path length)
    fn speed(self, constant: f64) -> (f64, f64, f64) {
        match self {
            Pace::Constant => (constant.clamp(*CONSTANT_SPEEDS.start(), *CONSTANT_SPEEDS.end()) * std::f64::consts::LN_2 / RHO, 0.1, 600.0),
            Pace::Fast => (4.0, 0.5, 3.0),
            Pace::Normal => (2.0, 1.0, 6.0),
            Pace::Cinematic => (1.0, 2.0, 12.0),
            Pace::NoFlight => (f64::INFINITY, 0.0, 0.0), //A flight of no time: arrived at once
        }
    }

    /// The share of the path that is done after the share `t` (0 to 1) of the time: smooth at both ends (smootherstep for cinematic, smoothstep otherwise), or the
    /// same share for the constant pace
    fn ease(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Pace::Cinematic => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
            Pace::Fast | Pace::Normal => t * t * (3.0 - 2.0 * t),
            Pace::Constant => t,
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
    /// The camera waits for a frame of the resolution that keeps this many frames per second: the flight job chooses its scale frame by frame from how long the
    /// frames take (see `auto_scale`), from full resolution down to `AUTO_COARSEST` of the pixels across, and the window scales it up smoothly; never supersampled
    Auto(u16),
}

/// The coarsest frames of `FlightRendering::Auto`: an eighth of the pixels across (12.5%)
pub const AUTO_COARSEST: f64 = 0.125;

/// The share of the frame time that the frames of `FlightRendering::Auto` aim for, so the frames whose time varies still keep the frame rate
const AUTO_AIM: f64 = 0.9;

/// The scale (the share of the pixels across, `AUTO_COARSEST` to 1) of the next frame of `FlightRendering::Auto`, from the scale of the last frame and the time it
/// `took`, for a frame time of `target`.
///
/// It steers by the measured time instead of predicting it: the time of a frame grows with its pixels (the square of the scale), but part of it is the same at
/// every scale (the frame's way to the window, the latency of the GPU), so a prediction from the pixels alone chooses too coarse a frame. Every frame moves the
/// scale half of the way, in proportion, to the one that would fit (at most 15% finer or 20% coarser), until the frames take between 90% and 100% of the aim:
/// a steady view then keeps its scale, in whole percents, instead of changing it every frame.
pub fn auto_scale(scale: f64, took: Duration, target: Duration) -> f64 {
    let (aim, took) = (target.as_secs_f64() * AUTO_AIM, took.as_secs_f64().max(1e-6));
    if (0.9 * aim..=aim).contains(&took) {
        return scale;
    }
    let step = (aim / took).sqrt().sqrt().clamp(0.8, 1.15);
    let next = ((scale * step) * 100.0).round() / 100.0;
    //(At least a percent, so a small step is not rounded away)
    let next = if took > aim { next.min(scale - 0.01) } else { next.max(scale + 0.01) };
    next.clamp(AUTO_COARSEST, 1.0)
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
    /// A straight leg (see `Route::Straight`): the natural logarithm of the zoom, `ln(to.scale / from.scale)`
    straight: Option<f64>,
}

impl Leg {
    fn new(from: Camera, to: Camera) -> Leg {
        let (w0, w1) = (from.scale, to.scale);
        let u1 = (to.x - from.x).hypot(to.y - from.y);
        //Below a ten-thousandth of the smaller view, a pan is a pure zoom: the formulas of a pan divide by the distance
        if u1 <= 1e-4 * w0.min(w1) {
            return Leg { from, to, length: (w1 / w0).ln().abs() / RHO, pan: None, straight: None };
        }
        let rho2 = RHO * RHO;
        let b = |w: f64, sign: f64| (w1 * w1 - w0 * w0 + sign * rho2 * rho2 * u1 * u1) / (2.0 * w * rho2 * u1);
        //r = ln(-b + sqrt(b² + 1)) = -asinh(b), without the cancellation of the logarithm for a large b
        let (r0, r1) = (-b(w0, 1.0).asinh(), -b(w1, -1.0).asinh());
        Leg { from, to, length: (r1 - r0) / RHO, pan: Some((u1, r0)), straight: None }
    }

    /// A straight leg from `from` to `to`, see `Route::Straight`. Along it the scale is `w0 · e^(zoom · τ)` (τ from 0 to 1) and the center has moved the share
    /// `(e^(zoom · τ) - 1) / (e^zoom - 1)` of the way, so the center moves the same number of views every moment. Its length, in the units of van Wijk and Nuij,
    /// is that of the zoom (`|zoom| / RHO`) and of the pan in views (the distance over the logarithmic mean of the scales), together
    fn straight(from: Camera, to: Camera) -> Leg {
        let zoom = (to.scale / from.scale).ln();
        let distance = (to.x - from.x).hypot(to.y - from.y);
        let mean_scale = if zoom.abs() < 1e-12 { from.scale } else { (to.scale - from.scale) / zoom };
        Leg { from, to, length: (zoom / RHO).hypot(distance / mean_scale), pan: None, straight: Some(zoom) }
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
        if let Some(zoom) = self.straight {
            let t = s / self.length;
            let along = if zoom.abs() < 1e-12 { t } else { (zoom * t).exp_m1() / zoom.exp_m1() };
            return Camera { x: from.x + (to.x - from.x) * along, y: from.y + (to.y - from.y) * along, scale: w0 * (zoom * t).exp() };
        }
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

/// A flight from one view to another, see the module
#[derive(Clone, Debug)]
pub struct Flight {
    leg: Leg,
    pace: Pace,
    duration: Duration,
}

impl Flight {
    /// A flight from `from` to `to` at `pace`, by the shortest path: zooming out only as far as the distance needs
    #[cfg(test)]
    pub fn new(from: Camera, to: Camera, pace: Pace) -> Flight {
        Flight::with(from, to, pace, Route::Shortest, CONSTANT_SPEED)
    }

    /// A flight from `from` to `to` at `pace` along `route`; `constant` is the speed of `Pace::Constant`, in doublings of the zoom per second
    pub fn with(from: Camera, to: Camera, pace: Pace, route: Route, constant: f64) -> Flight {
        let leg = match route {
            Route::Shortest => Leg::new(from, to),
            Route::Straight => Leg::straight(from, to),
        };
        let (speed, shortest, longest) = pace.speed(constant);
        let duration = Duration::from_secs_f64((leg.length / speed).clamp(shortest, longest));
        Flight { leg, pace, duration }
    }

    /// How long the flight takes
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    /// The camera `elapsed` after the start: the start view at 0, exactly the destination from `duration` on
    pub fn camera_at(&self, elapsed: Duration) -> Camera {
        self.leg.camera_at(self.pace.ease(elapsed.as_secs_f64() / self.duration.as_secs_f64()) * self.leg.length)
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
    fn the_constant_pace_covers_the_same_path_every_moment_at_the_speed_of_its_slider() {
        //The path length per second is the same from start to end: no slow start or arrival
        let flight = Flight::with(RESET, SEAHORSE, Pace::Constant, Route::Shortest, 3.0);
        let shares: Vec<f64> = (0..=10).map(|i| flight.share_of_path(flight.duration().mul_f64(f64::from(i) / 10.0))).collect();
        assert!(shares.windows(2).all(|pair| (pair[1] - pair[0] - 0.1).abs() < 1e-9), "{shares:?}");
        //A pure zoom by 2^12 at 3 doublings per second takes 4 s, at 6 doublings 2 s
        let deeper = Camera { scale: RESET.scale / 4096.0, ..RESET };
        assert!((Flight::with(RESET, deeper, Pace::Constant, Route::Shortest, 3.0).duration().as_secs_f64() - 4.0).abs() < 1e-9);
        assert!((Flight::with(RESET, deeper, Pace::Constant, Route::Shortest, 6.0).duration().as_secs_f64() - 2.0).abs() < 1e-9);
        //The deepest zoom is not capped at 12 s like the eased paces: 10^12 is about 40 doublings, 13 s at 3 per second
        assert!(Flight::with(RESET, DEEP, Pace::Constant, Route::Shortest, 3.0).duration() > seconds(12.0));
    }

    #[test]
    fn a_straight_flight_goes_in_a_line_without_zooming_out_and_moves_the_same_views_every_moment() {
        for (from, to) in [(RESET, SEAHORSE), (SEAHORSE, DEEP), (DEEP, RESET), (SEAHORSE, Camera { x: -0.7, ..SEAHORSE })] {
            let flight = Flight::with(from, to, Pace::Constant, Route::Straight, 3.0);
            assert_eq!((flight.camera_at(Duration::ZERO), flight.camera_at(flight.duration())), (from, to));
            let cameras: Vec<Camera> = (0..=200).map(|i| flight.camera_at(flight.duration().mul_f64(f64::from(i) / 200.0))).collect();
            for camera in &cameras {
                //Never wider than the wider end, and on the line between the centers
                assert!(camera.scale <= from.scale.max(to.scale) * (1.0 + 1e-9));
                let cross = (camera.x - from.x) * (to.y - from.y) - (camera.y - from.y) * (to.x - from.x);
                assert!(cross.abs() <= 1e-9 * (to.x - from.x).hypot(to.y - from.y).max(1e-300), "{camera:?}");
            }
            //The pan in views (distance over the scale) is the same every step, and so is the zoom
            let steps: Vec<(f64, f64)> = cameras.windows(2).map(|pair| ((pair[1].x - pair[0].x).hypot(pair[1].y - pair[0].y) / (pair[0].scale * pair[1].scale).sqrt(), (pair[1].scale / pair[0].scale).ln())).collect();
            let (pan, zoom) = steps[0];
            assert!(steps.iter().all(|(p, z)| (p - pan).abs() <= 0.02 * pan.max(1e-12) + 1e-12 && (z - zoom).abs() <= 1e-6 * zoom.abs() + 1e-12), "{from:?} -> {to:?}: {:?}", &steps[..3]);
        }
    }

    #[test]
    fn the_drop_downs_list_the_defaults_first() {
        assert_eq!(Pace::ALL[0], Pace::Fast);
    }
}

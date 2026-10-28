//! A flight of the camera from one view to another: smooth panning and zooming, used to go to a point of interest.
//!
//! A leg follows the path of van Wijk and Nuij, "Smooth and efficient zooming and panning" (IEEE Symposium on Information Visualization, 2003): to travel far it zooms out first and in again,
//! so the picture moves at the same apparent speed at every zoom, and the length `S` of the path measures how much the view changes. A flight has one leg (the
//! shortest route) or two (via the reset view). Its duration follows from the length at the speed of its pace, and the pace eases the start and the end.

use std::time::Duration;

use crate::complex_plane::View;

/// The curvature of the path: how much the camera zooms out to travel, the value that van Wijk and Nuij found pleasant (about sqrt(2))
const RHO: f64 = 1.42;

/// The way a flight goes to its destination
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Straight to the destination: zooming out only as far as the distance needs
    Shortest,
    /// First to the reset view (the whole set), then to the destination
    ViaReset,
}

impl Route {
    /// Every route, in the order of the drop-down
    pub const ALL: [Route; 2] = [Route::ViaReset, Route::Shortest];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            Route::Shortest => "Shortest route",
            Route::ViaReset => "Via reset",
        }
    }
}

/// How fast and how smoothly a flight moves
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    Fast,
    Normal,
    Cinematic,
}

impl Pace {
    /// Every pace, in the order of the drop-down
    pub const ALL: [Pace; 3] = [Pace::Cinematic, Pace::Normal, Pace::Fast];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            Pace::Fast => "Fast",
            Pace::Normal => "Normal",
            Pace::Cinematic => "Cinematic",
        }
    }

    /// The speed in units of path length per second, and the shortest and longest duration of a flight
    const fn speed(self) -> (f64, f64, f64) {
        match self {
            Pace::Fast => (4.0, 0.5, 3.0),
            Pace::Normal => (2.0, 1.0, 6.0),
            Pace::Cinematic => (1.0, 2.0, 12.0),
        }
    }

    /// The share of the path that is done after the share `t` (0 to 1) of the time: smooth at both ends (smootherstep for cinematic, smoothstep otherwise)
    fn ease(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Pace::Cinematic => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
            Pace::Fast | Pace::Normal => t * t * (3.0 - 2.0 * t),
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
    /// The camera waits for a full-resolution frame (the first layer): sharp all the way, slow where the view is expensive
    MaxQuality,
}

impl FlightRendering {
    /// Every mode, in the order of the drop-down
    pub const ALL: [FlightRendering; 3] = [FlightRendering::BestEffort, FlightRendering::LowRes, FlightRendering::MaxQuality];

    /// The name in the drop-down
    pub const fn name(self) -> &'static str {
        match self {
            FlightRendering::BestEffort => "Best effort",
            FlightRendering::LowRes => "Low res",
            FlightRendering::MaxQuality => "Max quality",
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

/// A flight from one view to another, see the module
#[derive(Clone, Debug)]
pub struct Flight {
    legs: Vec<Leg>,
    pace: Pace,
    duration: Duration,
}

impl Flight {
    /// A flight from `from` to `to` by `route` at `pace`; `reset` is the reset view that the route via reset passes
    pub fn new(from: Camera, to: Camera, route: Route, pace: Pace, reset: Camera) -> Flight {
        let legs = match route {
            Route::Shortest => vec![Leg::new(from, to)],
            Route::ViaReset => vec![Leg::new(from, reset), Leg::new(reset, to)],
        };
        let length: f64 = legs.iter().map(|leg| leg.length).sum();
        let (speed, shortest, longest) = pace.speed();
        let duration = Duration::from_secs_f64((length / speed).clamp(shortest, longest));
        Flight { legs, pace, duration }
    }

    /// How long the flight takes
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    /// The camera `elapsed` after the start: the start view at 0, exactly the destination from `duration` on
    pub fn camera_at(&self, elapsed: Duration) -> Camera {
        let total: f64 = self.legs.iter().map(|leg| leg.length).sum();
        let mut s = self.pace.ease(elapsed.as_secs_f64() / self.duration.as_secs_f64()) * total;
        for leg in &self.legs {
            if s < leg.length {
                return leg.camera_at(s);
            }
            s -= leg.length;
        }
        self.legs.last().map_or_else(|| unreachable!("a flight has a leg"), |leg| leg.to)
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
        for route in Route::ALL {
            for pace in Pace::ALL {
                for (from, to) in [(RESET, SEAHORSE), (SEAHORSE, DEEP), (DEEP, RESET), (SEAHORSE, SEAHORSE)] {
                    let flight = Flight::new(from, to, route, pace, RESET);
                    assert_eq!(flight.camera_at(Duration::ZERO), from, "{route:?} {pace:?}");
                    assert_eq!(flight.camera_at(flight.duration()), to, "{route:?} {pace:?}");
                    assert_eq!(flight.camera_at(flight.duration() * 2), to);
                    assert!(flight.arrived(flight.duration()) && !flight.arrived(flight.duration() / 2));
                }
            }
        }
    }

    #[test]
    fn the_route_via_reset_passes_the_reset_view_and_the_shortest_route_does_not_zoom_out_that_far() {
        let via = Flight::new(SEAHORSE, DEEP, Route::ViaReset, Pace::Normal, RESET);
        assert_eq!(via.legs[0].camera_at(via.legs[0].length), RESET, "the first leg ends at the reset view");
        let widest = |flight: &Flight| (0..=1000).map(|i| flight.camera_at(flight.duration().mul_f64(f64::from(i) / 1000.0)).scale).fold(0.0, f64::max);
        assert!(widest(&via) > 0.95 && widest(&via) <= 1.0 + 1e-9, "via reset shows the whole set: {}", widest(&via));
        let shortest = Flight::new(SEAHORSE, DEEP, Route::Shortest, Pace::Normal, RESET);
        let widest_shortest = widest(&shortest);
        //About as wide as the distance between the two points, to keep both in view, and far less than the whole set
        let distance = (DEEP.x - SEAHORSE.x).hypot(DEEP.y - SEAHORSE.y);
        assert!(widest_shortest > 0.5 * distance && widest_shortest < 2.0 * distance, "zooms out only as far as the distance needs: {widest_shortest} for {distance}");
    }

    #[test]
    fn the_picture_moves_smoothly_without_jumps() {
        //Between two frames (1/60 s) the view moves and zooms by a small part of itself, at every depth (at most 35% zoom, for a zoom by 10^12 at the fast pace)
        for (from, to) in [(RESET, DEEP), (DEEP, SEAHORSE)] {
            for pace in Pace::ALL {
                let flight = Flight::new(from, to, Route::Shortest, pace, RESET);
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
            let durations: Vec<f64> = [Pace::Fast, Pace::Normal, Pace::Cinematic].map(|pace| Flight::new(from, to, Route::Shortest, pace, RESET).duration().as_secs_f64()).to_vec();
            assert!(durations[0] <= durations[1] && durations[1] <= durations[2], "{durations:?}");
            assert!(durations[0] >= 0.5 && durations[2] <= 12.0, "{durations:?}");
        }
        //Even a zoom by a factor 10^12 is capped
        assert_eq!(Flight::new(RESET, DEEP, Route::ViaReset, Pace::Cinematic, RESET).duration(), seconds(12.0));
    }

    #[test]
    fn the_share_of_the_path_goes_from_0_to_1_without_going_back() {
        let flight = Flight::new(RESET, SEAHORSE, Route::ViaReset, Pace::Cinematic, RESET);
        let shares: Vec<f64> = (0..=100).map(|i| flight.share_of_path(flight.duration().mul_f64(f64::from(i) / 100.0))).collect();
        assert_eq!((shares[0], shares[100]), (0.0, 1.0));
        assert!(shares.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn the_defaults_are_via_reset_and_cinematic() {
        assert_eq!(Route::ALL[0], Route::ViaReset);
        assert_eq!(Pace::ALL[0], Pace::Cinematic);
    }
}

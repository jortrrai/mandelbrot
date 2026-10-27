use crate::complex::Complex;

#[derive(Clone)]
pub struct MandelbrotSet {
    pub max_iterations: u32,
    ///If z remains within the orbit_radius in max_iterations, we assume c does not tend to infinity
    pub orbit_radius: f64,    
}

/// The amount of points `MandelbrotSet::iterate_lanes` iterates at the same time
pub const LANES: usize = 4;
/// `iterate_lanes` checks whether every point has stopped after this many steps
const CHECK_INTERVAL: u32 = 8;

impl MandelbrotSet {
    pub fn new(max_iterations: u32, orbit_radius: f64) -> MandelbrotSet {
        MandelbrotSet { max_iterations, orbit_radius }
    }

    /// Run the Mandelbrot set algorithm for a single Complex number
    /// Returns the amount of iterations needed before Zn escapes to infinity
    pub fn iterate(&self, c: &Complex) -> u32 {
        if self.is_known_interior(c) {
            return self.max_iterations; //Same result as the escape loop below, which never breaks for points inside the set
        }
        self.iterate_escape_loop(c)
    }

    /// Runs the Mandelbrot set algorithm for `LANES` Complex numbers at the same time, with exactly the same result for every point as `iterate`.
    /// The points are iterated in lockstep, with the same arithmetic in the same order, so the compiler can use vector instructions.
    /// A point that has escaped, or is known to be inside the set, stops counting, but the other points continue until none is left.
    pub fn iterate_lanes(&self, points: &[Complex; LANES]) -> [u32; LANES] {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                //SAFETY: the CPU supports AVX2, which was just detected
                return unsafe { self.iterate_lanes_avx2(points) };
            }
        }
        self.iterate_lanes_generic(points)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn iterate_lanes_avx2(&self, points: &[Complex; LANES]) -> [u32; LANES] {
        self.iterate_lanes_generic(points)
    }

    #[inline(always)]
    fn iterate_lanes_generic(&self, points: &[Complex; LANES]) -> [u32; LANES] {
        let orbit_radius_squared = self.orbit_radius * self.orbit_radius;
        let mut iterations = [0_u32; LANES];
        let mut active = [true; LANES];
        let (mut cx, mut cy) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        for lane in 0..LANES {
            (cx[lane], cy[lane]) = (points[lane].x, points[lane].y);
            if self.is_known_interior(&points[lane]) {
                active[lane] = false;
                iterations[lane] = self.max_iterations;
            }
        }
        let (mut zx, mut zy) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        let mut step = 0;
        while step < self.max_iterations {
            if !active.iter().any(|lane_active| *lane_active) {
                break;
            }
            //Only check whether any point is left every few steps; points that already stopped do not count anymore
            for _ in 0..CHECK_INTERVAL.min(self.max_iterations - step) {
                for lane in 0..LANES {
                    //Same arithmetic as z = z.squared().add(c)
                    let new_x = zx[lane] * zx[lane] - zy[lane] * zy[lane] + cx[lane];
                    let new_y = 2.0 * zx[lane] * zy[lane] + cy[lane];
                    (zx[lane], zy[lane]) = (new_x, new_y);
                    let escaped = (new_x * new_x + new_y * new_y) > orbit_radius_squared;
                    active[lane] &= !escaped;
                    iterations[lane] += u32::from(active[lane]);
                }
            }
            step += CHECK_INTERVAL;
        }
        iterations
    }

    /// Returns true only if `c` is certainly inside the main cardioid or the period-two bulb, so `c` never escapes.
    /// Conservative: the boundaries are shrunk by `MARGIN`, so points near them fall back to the escape loop.
    /// Only valid for `orbit_radius >= 2.0`, since orbits of points inside the set stay within radius 2, but may leave a smaller radius.
    fn is_known_interior(&self, c: &Complex) -> bool {
        const MARGIN: f64 = 1e-9;
        if self.orbit_radius < 2.0 {
            return false;
        }
        let (x, y) = (c.x, c.y);
        let y_squared = y * y;
        let q = (x - 0.25) * (x - 0.25) + y_squared;
        let in_cardioid = q * (q + (x - 0.25)) <= 0.25 * y_squared - MARGIN;
        let in_bulb = (x + 1.0) * (x + 1.0) + y_squared <= 0.0625 - MARGIN;
        in_cardioid || in_bulb
    }

    /// The Mandelbrot set escape loop, without any shortcuts
    fn iterate_escape_loop(&self, c: &Complex) -> u32 {
        let mut z = Complex::new(0.0, 0.0);
        let mut iterations: u32 = 0;
        let orbit_radius_squared = self.orbit_radius*self.orbit_radius;
        for _ in 0..self.max_iterations {
            z = z.squared().add(c);

            if (z.x * z.x + z.y * z.y) > orbit_radius_squared { //Optimization: square both sides of the Mandelbrot set function, saves us taking the square root
                break;
            }
            iterations += 1;
        }
        iterations
    }
}

impl std::fmt::Debug for MandelbrotSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "max_iterations = {}, orbit_radius = {}", self.max_iterations, self.orbit_radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORBIT_RADII: [f64; 3] = [2.0, 4.0, 1.0];

    #[test]
    fn known_interior_points() {
        let m = MandelbrotSet::new(1000, 2.0);
        for (x, y) in [(0.0, 0.0), (-0.3, 0.0), (0.2, 0.3), (-1.0, 0.0), (-1.1, 0.1), (-0.9, -0.1)] {
            let c = Complex::new(x, y);
            assert!(m.is_known_interior(&c), "({x}, {y}) should be known interior");
            assert_eq!(m.iterate(&c), 1000);
            assert_eq!(m.iterate_escape_loop(&c), 1000);
        }
    }

    #[test]
    fn exterior_and_boundary_points_fall_back_to_the_escape_loop() {
        let m = MandelbrotSet::new(1000, 2.0);
        for (x, y) in [(1.0, 1.0), (0.5, 0.5), (-2.0, 0.0), (0.26, 0.0), (-0.75, 0.1), (-1.25, 0.0), (-0.75, 0.0), (0.25, 0.5)] {
            let c = Complex::new(x, y);
            assert_eq!(m.iterate(&c), m.iterate_escape_loop(&c), "({x}, {y})");
        }
        // Just outside the cardioid on the real axis, and just outside the bulb
        assert!(!m.is_known_interior(&Complex::new(0.2500001, 0.0)));
        assert!(!m.is_known_interior(&Complex::new(-1.2500001, 0.0)));
        assert!(!m.is_known_interior(&Complex::new(-1.25, 0.0))); //On the boundary, margin makes it fall back
    }

    #[test]
    fn small_orbit_radius_never_uses_the_shortcut() {
        let m = MandelbrotSet::new(1000, 1.0);
        assert!(!m.is_known_interior(&Complex::new(0.0, 0.0)));
    }

    /// Compares `iterate` with the original escape loop on a dense grid (default view and boundary-focused views),
    /// for several iteration limits and orbit radii. Samples are jittered like supersampling samples.
    #[test]
    fn iterate_matches_escape_loop() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let views = [(-2.5, 1.0, -1.0, 1.0), (-1.30, -1.20, -0.05, 0.05), (-0.80, -0.70, 0.05, 0.15), (0.20, 0.30, -0.05, 0.05)];
        for max_iterations in [1, 10, 100, 1000] {
            for orbit_radius in ORBIT_RADII {
                let m = MandelbrotSet::new(max_iterations, orbit_radius);
                for (min_x, max_x, min_y, max_y) in views {
                    let n = 150;
                    for i in 0..n {
                        for j in 0..n {
                            let jitter: (f64, f64) = rng.gen();
                            let x = min_x + (max_x - min_x) * (f64::from(i) + jitter.0) / f64::from(n);
                            let y = min_y + (max_y - min_y) * (f64::from(j) + jitter.1) / f64::from(n);
                            let c = Complex::new(x, y);
                            assert_eq!(m.iterate(&c), m.iterate_escape_loop(&c), "c = ({x}, {y}), max_iterations = {max_iterations}, orbit_radius = {orbit_radius}");
                        }
                    }
                }
            }
        }
    }

    /// `iterate_lanes` must give exactly the result of `iterate` for every point, for any mix of interior, exterior and boundary points
    #[test]
    fn iterate_lanes_matches_iterate() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let views = [(-0.75, 0.0, 1.25), (-0.7498100000000001, -0.0203, 0.00008), (-1.7862712, 0.0000524, 0.00002), (0.9, 0.9, 0.4), (-0.1, 0.65, 0.12)];
        for max_iterations in [0, 1, 7, 100, 1000, 3000] {
            for orbit_radius in [2.0, 4.0, 1.0] {
                let m = MandelbrotSet::new(max_iterations, orbit_radius);
                for (x, y, half) in views {
                    for _ in 0..500 {
                        let points: [Complex; LANES] = std::array::from_fn(|_| Complex::new(x + (rng.gen::<f64>() - 0.5) * 2.0 * half, y + (rng.gen::<f64>() - 0.5) * 2.0 * half));
                        let expected: Vec<u32> = points.iter().map(|point| m.iterate(point)).collect();
                        assert_eq!(m.iterate_lanes(&points).to_vec(), expected, "max_iterations = {max_iterations}, orbit_radius = {orbit_radius}, points = {:?}", points.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}

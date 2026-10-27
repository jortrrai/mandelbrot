use crate::complex::Complex;

#[derive(Clone)]
pub struct MandelbrotSet {
    pub max_iterations: u32,
    ///If z remains within the orbit_radius in max_iterations, we assume c does not tend to infinity
    pub orbit_radius: f64,    
    ///When set (automatic iteration limit), the palette is scaled to this many iterations instead of `max_iterations`, so raising the limit does not recolor the image,
    ///see `MandelbrotSet::color`
    pub palette_scale: Option<u32>,
    ///Whether colors use the smooth (fractional) iteration position of `iterate_smooth` instead of the integer iteration count, see `MandelbrotSet::position`
    pub smooth_coloring: bool,
}

/// The amount of points `MandelbrotSet::iterate_lanes` iterates at the same time
pub const LANES: usize = 4;
/// Smooth coloring continues an escaped orbit until |z|^2 reaches this (radius 256), or for at most `SMOOTH_MAX_EXTRA_STEPS` steps
const SMOOTH_BAILOUT_SQUARED: f64 = 65536.0;
const SMOOTH_MAX_EXTRA_STEPS: u32 = 32;

/// The orbits of the `LANES` points of `iterate_lanes_core`: the iteration counts, and for the points that escaped (when smooth) the value of z at the escape
struct LaneResult {
    iterations: [u32; LANES],
    escaped: [bool; LANES],
    escape_x: [f64; LANES],
    escape_y: [f64; LANES],
}

/// `iterate_lanes` checks whether every point has stopped after this many steps
const CHECK_INTERVAL: u32 = 4;

impl MandelbrotSet {
    pub fn new(max_iterations: u32, orbit_radius: f64) -> MandelbrotSet {
        MandelbrotSet { max_iterations, orbit_radius, palette_scale: None, smooth_coloring: true }
    }

    /// The color of a point with the given iteration count. Without a `palette_scale` this is exactly `coloring_function(iterations, max_iterations)`.
    /// With one, points inside the set (`iterations == max_iterations`) get the color the palette gives to its last iteration, and escaped points repeat the palette every `palette_scale` iterations,
    /// so a point that escaped keeps its color when the limit is raised.
    pub fn color(&self, coloring_function: fn(u32, u32) -> crate::coloring::TrueColor, iterations: u32) -> crate::coloring::TrueColor {
        match self.palette_scale {
            None => coloring_function(iterations, self.max_iterations),
            Some(scale) if iterations >= self.max_iterations => coloring_function(scale, scale),
            Some(scale) => coloring_function(iterations % scale, scale),
        }
    }

    /// Whether `c` is certified to be inside the set by the cardioid and period-two bulb test, see `is_known_interior`
    pub fn is_certified_interior(&self, c: &Complex) -> bool {
        self.is_known_interior(c)
    }

    /// The position in the color table of a point: `iterate` as a number, or with `smooth_coloring` the smooth position of `iterate_smooth`
    pub fn position(&self, c: &Complex) -> f64 {
        if self.smooth_coloring { self.iterate_smooth(c) } else { f64::from(self.iterate(c)) }
    }

    /// Like `position`, for `LANES` points at the same time
    pub fn position_lanes(&self, points: &[Complex; LANES]) -> [f64; LANES] {
        if self.smooth_coloring { self.iterate_lanes_smooth(points) } else { self.iterate_lanes(points).map(f64::from) }
    }

    /// The continuous escape time of a point (normalized iteration count): the iteration count of `iterate` plus a fraction in about [-0.85, 0.53) that depends on how far
    /// beyond the orbit radius the orbit landed, so neighboring points in different integer bands get neighboring positions. Points that do not escape get `max_iterations`.
    /// The integer count and the interior classification are exactly those of `iterate`; only the escaped points do a few extra steps and two logarithms.
    /// Formula: nu = n - log2(ln|z_n|) for a large |z_n|, see <https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#Continuous_(smooth)_coloring>
    pub fn iterate_smooth(&self, c: &Complex) -> f64 {
        if self.is_known_interior(c) {
            return f64::from(self.max_iterations);
        }
        let orbit_radius_squared = self.orbit_radius * self.orbit_radius;
        let mut z = Complex::new(0.0, 0.0);
        let mut iterations: u32 = 0;
        for _ in 0..self.max_iterations {
            z = z.squared().add(c);
            if (z.x * z.x + z.y * z.y) > orbit_radius_squared {
                return self.smooth_position(iterations, (z.x, z.y), c);
            }
            iterations += 1;
        }
        f64::from(self.max_iterations)
    }

    /// The smooth position of a point that escaped: `iterations` is the count of `iterate`, `z` the first value of its orbit outside the orbit radius.
    /// The orbit is continued to radius 256 (at most `SMOOTH_MAX_EXTRA_STEPS` steps), where the formula is accurate and independent of the orbit radius.
    fn smooth_position(&self, iterations: u32, z: (f64, f64), c: &Complex) -> f64 {
        let (mut zx, mut zy) = z;
        let mut steps = iterations + 1;
        for _ in 0..SMOOTH_MAX_EXTRA_STEPS {
            if zx * zx + zy * zy >= SMOOTH_BAILOUT_SQUARED {
                break;
            }
            (zx, zy) = (zx * zx - zy * zy + c.x, 2.0 * zx * zy + c.y);
            steps += 1;
        }
        let modulus_squared = zx * zx + zy * zy;
        if modulus_squared <= 1.0 {
            return f64::from(iterations); //The formula needs |z| > 1; only possible with an orbit radius below 1.5 that was left again
        }
        let log_modulus = 0.5 * modulus_squared.ln();
        let position = f64::from(steps - 1) - log_modulus.ln() / std::f64::consts::LN_2;
        position.clamp(0.0, f64::from(self.max_iterations.saturating_sub(1)))
    }

    /// The color at a (fractional) table position: the colors of the two nearest integer iteration counts, mixed by the fraction in 1/256 steps (integer arithmetic).
    /// `lookup` gives the color of an integer iteration count. A whole position gives exactly `lookup(position)`, which is what coloring by integer counts did.
    pub fn interpolated_color(&self, position: f64, lookup: impl Fn(u32) -> crate::coloring::TrueColor) -> crate::coloring::TrueColor {
        let index = position as u32;
        let weight = ((position - f64::from(index)) * 256.0) as u32;
        let first = lookup(index);
        if weight == 0 || index >= self.max_iterations {
            return first;
        }
        let second = lookup(index + 1);
        let mix = |a: u8, b: u8| ((u32::from(a) * (256 - weight) + u32::from(b) * weight + 128) >> 8) as u8;
        crate::coloring::TrueColor::new(mix(first.red, second.red), mix(first.green, second.green), mix(first.blue, second.blue))
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
        self.iterate_lanes_dispatch::<false>(points).iterations
    }

    /// Like `iterate_lanes`, but returns the smooth color position of every point, see `iterate_smooth`; identical to it for every point
    pub fn iterate_lanes_smooth(&self, points: &[Complex; LANES]) -> [f64; LANES] {
        let result = self.iterate_lanes_dispatch::<true>(points);
        std::array::from_fn(|lane| {
            if result.escaped[lane] {
                self.smooth_position(result.iterations[lane], (result.escape_x[lane], result.escape_y[lane]), &points[lane])
            } else {
                f64::from(result.iterations[lane])
            }
        })
    }

    fn iterate_lanes_dispatch<const SMOOTH: bool>(&self, points: &[Complex; LANES]) -> LaneResult {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                //SAFETY: the CPU supports AVX2, which was just detected
                return unsafe { self.iterate_lanes_avx2::<SMOOTH>(points) };
            }
        }
        self.iterate_lanes_generic::<SMOOTH>(points)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn iterate_lanes_avx2<const SMOOTH: bool>(&self, points: &[Complex; LANES]) -> LaneResult {
        self.iterate_lanes_generic::<SMOOTH>(points)
    }

    #[inline(always)]
    fn iterate_lanes_generic<const SMOOTH: bool>(&self, points: &[Complex; LANES]) -> LaneResult {
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
        let (mut escaped_lane, mut escape_x, mut escape_y) = ([false; LANES], [0.0_f64; LANES], [0.0_f64; LANES]);
        let (mut saved_x, mut saved_y) = ([0.0_f64; LANES], [0.0_f64; LANES]);
        let (mut blocks_since_saved, mut blocks_until_save) = (0_u32, 1_u32);
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
                    if SMOOTH {
                        //Remember z at the step where the point escapes, for the smooth position
                        let first_escape = escaped & active[lane];
                        escape_x[lane] = if first_escape { new_x } else { escape_x[lane] };
                        escape_y[lane] = if first_escape { new_y } else { escape_y[lane] };
                        escaped_lane[lane] |= first_escape;
                    }
                    active[lane] &= !escaped;
                    iterations[lane] += u32::from(active[lane]);
                }
            }
            step += CHECK_INTERVAL;
            //Brent's cycle detection, once per block of steps: if a point is exactly where it was at an earlier block, its orbit repeats forever,
            //and every point of it was inside the orbit radius, so the original loop would count every step: max_iterations
            for lane in 0..LANES {
                if active[lane] && zx[lane] == saved_x[lane] && zy[lane] == saved_y[lane] {
                    active[lane] = false;
                    iterations[lane] = self.max_iterations;
                }
            }
            blocks_since_saved += 1;
            if blocks_since_saved == blocks_until_save {
                (saved_x, saved_y) = (zx, zy);
                blocks_since_saved = 0;
                blocks_until_save = blocks_until_save.saturating_mul(2);
            }
        }
        LaneResult { iterations, escaped: escaped_lane, escape_x, escape_y }
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

    #[test]
    fn the_palette_scale_keeps_the_color_of_escaped_points_when_the_limit_changes() {
        let f = crate::coloring::TrueColor::new_from_bernstein_polynomials;
        let manual = MandelbrotSet::new(1000, 2.0);
        for i in [0, 1, 17, 500, 999, 1000] {
            let (a, b) = (manual.color(f, i), f(i, 1000));
            assert_eq!((a.red, a.green, a.blue), (b.red, b.green, b.blue), "manual mode colors as before");
        }
        let (mut low, mut high) = (MandelbrotSet::new(400, 2.0), MandelbrotSet::new(9000, 2.0));
        (low.palette_scale, high.palette_scale) = (Some(400), Some(400));
        for i in [0, 1, 17, 200, 399] {
            let (a, b) = (low.color(f, i), high.color(f, i));
            assert_eq!((a.red, a.green, a.blue), (b.red, b.green, b.blue), "iteration {i} keeps its color");
        }
        let (a, b) = (low.color(f, 400), high.color(f, 9000));
        assert_eq!((a.red, a.green, a.blue), (b.red, b.green, b.blue), "inside the set has the same color at any limit");
    }

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

    /// Points inside bulbs other than the main cardioid and the period-2 bulb never escape, and their orbits are found to repeat: the result must still be `max_iterations`
    #[test]
    fn iterate_lanes_matches_iterate_inside_other_bulbs_and_deep_in_the_set() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        //Centers of the period-3 bulbs and a period-4 bulb, and a minibrot; the points are scattered around them
        let centers = [(-0.12256116687665362, 0.7448617666197442), (-0.12256116687665362, -0.7448617666197442), (-0.5043401, 0.5627655), (-1.7548776662466927, 0.0), (-1.3107, 0.0)];
        for max_iterations in [50, 1000, 4000] {
            let m = MandelbrotSet::new(max_iterations, 2.0);
            for (x, y) in centers {
                for half in [0.002, 0.02, 0.1] {
                    for _ in 0..300 {
                        let points: [Complex; LANES] = std::array::from_fn(|_| Complex::new(x + (rng.gen::<f64>() - 0.5) * 2.0 * half, y + (rng.gen::<f64>() - 0.5) * 2.0 * half));
                        let expected: Vec<u32> = points.iter().map(|point| m.iterate(point)).collect();
                        assert_eq!(m.iterate_lanes(&points).to_vec(), expected, "max_iterations = {max_iterations}, points = {:?}", points.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>());
                    }
                }
            }
        }
        //The center of the period-3 bulb is found to repeat, and iterates to the limit
        let m = MandelbrotSet::new(3000, 2.0);
        let center = Complex::new(-0.12256116687665362, 0.7448617666197442);
        let points: [Complex; LANES] = std::array::from_fn(|_| Complex::new(center.x, center.y));
        assert_eq!(m.iterate_lanes(&points), [3000; LANES]);
    }

    fn test_points() -> Vec<Complex> {
        let mut points = Vec::new();
        for (min_x, max_x, min_y, max_y) in [(-2.5, 1.0, -1.0, 1.0), (-0.80, -0.70, 0.05, 0.15), (-1.30, -1.20, -0.05, 0.05), (0.20, 0.30, -0.05, 0.05)] {
            for i in 0..60 {
                for j in 0..60 {
                    points.push(Complex::new(min_x + (max_x - min_x) * f64::from(i) / 60.0, min_y + (max_y - min_y) * f64::from(j) / 60.0));
                }
            }
        }
        points
    }

    #[test]
    fn smooth_position_keeps_the_integer_count_and_the_interior_classification() {
        for max_iterations in [1, 10, 300] {
            for orbit_radius in ORBIT_RADII {
                let m = MandelbrotSet::new(max_iterations, orbit_radius);
                for c in test_points() {
                    let (count, position) = (m.iterate(&c), m.iterate_smooth(&c));
                    if count == max_iterations {
                        assert_eq!(position, f64::from(max_iterations), "points that do not escape stay at the limit: {c:?}");
                    } else {
                        assert!(position >= 0.0 && position <= f64::from(max_iterations - 1) && position >= f64::from(count) - 2.0, "{c:?}: count {count}, position {position}");
                    }
                }
            }
        }
    }

    #[test]
    fn smooth_positions_of_the_simd_and_scalar_paths_are_identical() {
        let points = test_points();
        for max_iterations in [1, 7, 300] {
            let m = MandelbrotSet::new(max_iterations, 2.0);
            for group in points.chunks_exact(LANES) {
                let group: [Complex; LANES] = std::array::from_fn(|lane| Complex::new(group[lane].x, group[lane].y));
                let lanes = m.iterate_lanes_smooth(&group);
                let generic = m.iterate_lanes_generic::<true>(&group);
                for lane in 0..LANES {
                    assert_eq!(lanes[lane].to_bits(), m.iterate_smooth(&group[lane]).to_bits(), "{:?}", group[lane]);
                    assert_eq!(generic.iterations[lane], m.iterate(&group[lane]));
                }
            }
        }
    }

    #[test]
    fn smooth_position_is_continuous_where_the_integer_count_jumps() {
        let m = MandelbrotSet::new(1000, 2.0);
        let positions: Vec<(u32, f64)> = (0..4000).map(|i| { let c = Complex::new(-0.5 + f64::from(i) * 1.25e-4, 0.9); (m.iterate(&c), m.iterate_smooth(&c)) }).collect();
        //Only where the bands are wide enough to be sampled well, i.e. not right at the boundary of the set
        let pairs: Vec<_> = positions.windows(2).filter(|w| w[0].0 <= 8 && w[1].0 <= 8).collect();
        let jumps = pairs.iter().filter(|w| w[0].0 != w[1].0).count();
        let largest_step = pairs.iter().map(|w| (w[1].1 - w[0].1).abs()).fold(0.0, f64::max);
        assert!(jumps >= 5, "the line crosses several integer bands: {jumps}");
        assert!(largest_step < 0.2, "the smooth position changes gradually across the bands: {largest_step}");
    }

    #[test]
    fn interpolated_color_mixes_neighbors_and_keeps_whole_positions() {
        use crate::coloring::TrueColor;
        let m = MandelbrotSet::new(10, 2.0);
        let lookup = |i: u32| TrueColor::new((i * 10) as u8, 200, (255 - i * 10) as u8);
        let whole = m.interpolated_color(3.0, lookup);
        assert_eq!((whole.red, whole.green, whole.blue), (30, 200, 225));
        let half = m.interpolated_color(3.5, lookup);
        assert_eq!((half.red, half.green, half.blue), (35, 200, 220));
        let inside = m.interpolated_color(10.0, lookup);
        assert_eq!((inside.red, inside.green, inside.blue), (100, 200, 155), "the limit itself is never mixed with a color beyond it");
    }
}

//! The part of the complex plane that an image shows, and the mapping between its pixels and complex numbers.

use crate::complex::Complex;

/// # Complex plane
/// In mathematics, the complex plane is the plane formed by the complex numbers, with a Cartesian coordinate system such that the x-axis, called the real axis, is formed by the real numbers, and the y-axis, called the imaginary axis, is formed by the imaginary numbers.
/// \
/// The complex plane allows a geometric interpretation of complex numbers. Under addition, they add like vectors. The multiplication of two complex numbers can be expressed more easily in polar coordinates—the magnitude or modulus of the product is the product of the two
///  absolute values, or moduli, and the angle or argument of the product is the sum of the two angles, or arguments. In particular, multiplication by a complex number of modulus 1 acts as a rotation.
/// ## Notational conventions
/// ### Complex numbers
/// In complex analysis, the complex numbers are customarily represented by the symbol z, which can be separated into its real (x) and imaginary (y) parts:
/// \
/// ```text
/// z = x + iy
/// ```
/// for example: z = 4 + 5i, where x and y are real numbers, and i is the imaginary unit. In this customary notation the complex number z corresponds to the point (x, y) in the Cartesian plane.
/// \
/// In the Cartesian plane the point (x, y) can also be represented in polar coordinates as
/// ```text
/// (x, y) = (rcosθ, rsinθ)  (r, θ)= (√(x^2 + y^2), arctan(y/x))
/// ```
/// ### Complex plane notation ℂ
/// Complex plane is denoted as ℂ.
#[derive(Clone)]
pub struct ComplexPlane {
    // Complex plane dimensions
    /// The smallest real part that the plane shows
    pub min_x: f64,
    /// The largest real part that the plane shows
    pub max_x: f64,
    /// The width of the plane on the real axis (`max_x - min_x`)
    pub length_x: f64,
    /// The smallest `y` of the plane. The `y` bounds are stored with the orientation of the screen: the imaginary part of a point is the negative of its `y`
    pub min_y: f64,
    /// The largest `y` of the plane, see `min_y`
    pub max_y: f64,
    /// The height of the plane (`max_y - min_y`)
    pub length_y: f64,
    // Complex plane increments
    /// The distance on the real axis between two neighboring pixel columns
    pub increment_x: f64,
    /// The distance on the imaginary axis between two neighboring pixel rows
    pub increment_y: f64,
    // Pixel plane width and height
    width: usize,
    height: usize,
}

impl ComplexPlane {
    /// Creates the dimensions of a Complex plane for viewing the Mandelbrot set based on the width and height of a screen in pixels.
    /// By default: Complex plane: R ∈ [-2,0.5] and C will be calculated to preserve proportionality.
    pub fn new(width: usize, height: usize) -> ComplexPlane {
        let float_width = f64::from(width as u32);
        let float_height = f64::from(height as u32);
        // Complex plane dimensions
        let min_x: f64 = -2.0;
        let max_x: f64 = 1.0 / 2.0;
        let length_x: f64 = max_x - min_x;
        let aspect_ratio_h_w: f64 = float_height / float_width;
        let min_y: f64 = -(length_x * aspect_ratio_h_w / 2.0);
        let max_y: f64 = -min_y;
        let length_y: f64 = max_y - min_y;
        // Complex plane increments
        let increment_x: f64 = length_x / float_width;
        let increment_y: f64 = length_y / float_height;
        ComplexPlane {
            min_x,
            max_x,
            length_x,
            min_y,
            max_y,
            length_y,
            increment_x,
            increment_y,
            width,
            height,
        }
    }

    /// Translate the Complex plane by adding x to `min_x` and `max_x`, and y to `min_y` and `max_y`
    pub fn translate(&mut self, x: f64, y: f64) {
        self.min_x += x;
        self.max_x += x;
        self.min_y += y;
        self.max_y += y;
    }

    /// Convert the point (x,y) in the pixel plane to the complex number a+bi in the complex plane
    pub fn complex_from_pixel_plane(&self, x: f64, y: f64) -> Complex {
        let x = x.mul_add(self.increment_x, self.min_x);
        let y = -y.mul_add(self.increment_y, self.min_y); //Negate because math plane is bottom-top, and screen plane is top-bottom
        let c = Complex::new(x, y);
        c
    }

    /// Resets the total translation and scaling applied to the Complex plane by `translate` and `scale`
    pub fn reset(&mut self) {
        *self = ComplexPlane::new(self.width, self.height);
    }

    //Returns the total scale applied to the Complex plane
    /// The scale of the plane: its width on the real axis divided by 2.5, so a smaller scale is a deeper zoom and scale 1 is the starting view
    pub fn get_scale(&self) -> f64 {
        let s = self.length_x / 2.5;
        s
    }

    /// How many steps between neighboring 64-bit numbers fit between two neighboring pixels, at the coordinates of this view: the pixel spacing divided by `f64::EPSILON` times the largest
    /// coordinate on that axis, for the finer of the two axes. The coordinates of the pixels can only be told apart as long as this is more than 1; at 1 or less, neighboring pixels share one
    /// coordinate and the image turns into blocks. The iteration amplifies rounding errors, so the image degrades well before that: see `ComplexPlane::precision`.
    pub fn precision_steps(&self) -> f64 {
        let steps = |increment: f64, a: f64, b: f64| increment.abs() / (f64::EPSILON * a.abs().max(b.abs()).max(f64::MIN_POSITIVE));
        steps(self.increment_x, self.min_x, self.max_x).min(steps(self.increment_y, self.min_y, self.max_y))
    }

    /// How well 64-bit numbers still resolve this view, see `Precision`
    pub fn precision(&self) -> Precision {
        let steps = self.precision_steps();
        if steps < PRECISION_LIMIT_STEPS {
            Precision::Exhausted
        } else if steps < PRECISION_WARNING_STEPS {
            Precision::Degrading
        } else {
            Precision::Full
        }
    }

    /// Scale the complex plane, by multiplying the complex plane dimensions and increments by factor.
    /// If factor > 1.0: zoom out
    /// If factor < 1.0: zoom in
    pub fn scale(&mut self, factor: f64) {
        let center = self.center();
        self.min_x *= factor;
        self.max_x *= factor;
        self.min_y *= factor;
        self.max_y *= factor;
        self.length_x *= factor;
        self.length_y *= factor;
        self.increment_x *= factor;
        self.increment_y *= factor;
        self.set_center(&center);
    }

    /// Returns the center of the Complex plane bounded by `min_x`, `min_y`, `max_x`, `max_y`
    pub fn center(&self) -> Complex {
        let x = self.min_x + (self.max_x - self.min_x) / 2.0;
        let y = -(self.min_y + (self.max_y - self.min_y) / 2.0); //Negate because math plane is bottom-top, and screen plane is top-bottom
        Complex::new(x, y)
    }

    /// Translate `min_x`, `max_x`, `min_y`, `max_y` so that center becomes the center of the Complex plane
    /// Returns the translation
    pub fn set_center(&mut self, center: &Complex) -> Complex {
        let old = self.center();
        let mut translation = center.subtract(&old);
        translation.y = -translation.y; //Negate because the Complex plane and pixel plane are flipped
        /*println!("DEBUG set_center():");
        println!("\tcenter: {:?}", center);
        println!("\told: {:?}", old);
        println!("\ttranslation: {:?}", translation);*/
        self.translate(translation.x, translation.y);
        translation
    }

    /// Set the Complex plane at Center (x,y) at the given scale, where scale == 1 => max_x-min_x=2.5
    pub fn set_view_separated(&mut self, x: f64, y: f64, scale: f64) {
        self.reset();
        self.set_center(&Complex::new(x, y));
        self.scale(scale);
    }

    /// Set the Complex plane at Center (x,y) at the given scale, where scale == 1 => max_x-min_x=2.5
    pub fn set_view(&mut self, view: &View)
    {
        self.set_view_separated(view.x, view.y, view.scale);
    }

    /// The distance on the imaginary axis that `amount` pixel rows cover
    pub fn pixels_to_imaginary(&self, amount: u8) -> f64 {
        f64::from(amount) * self.increment_y 
    }

    /// The distance on the real axis that `amount` pixel columns cover
    pub fn pixels_to_real(&self, amount: u8) -> f64 {
        f64::from(amount) * self.increment_x
    }

    /// The amount of pixel rows that an imaginary distance `value` covers, truncated toward zero (and saturated to the range of `i16`)
    pub fn imaginary_to_pixels(&self, value: f64) -> i16 {
        (value / self.increment_y) as i16
    }

    /// The amount of pixel columns that a real distance `value` covers, truncated toward zero (and saturated to the range of `i16`)
    pub fn real_to_pixels(&self, value: f64) -> i16 {
        (value / self.increment_x) as i16
    }

    /// The view that the plane shows now: its center and its scale
    pub fn get_view(&self) -> View {
        let center = self.center();
        let scale = self.get_scale();
        View::new(center.x, center.y, scale)
    }
}

/// Below this many steps of a 64-bit number per pixel (`ComplexPlane::precision_steps`), the view counts as `Precision::Degrading`: for the default 1200 pixel wide window, a zoom of about 2e10
pub const PRECISION_WARNING_STEPS: f64 = 1024.0;
/// Below this many steps of a 64-bit number per pixel, the view counts as `Precision::Exhausted`: for the default window, a zoom of about 1e12
pub const PRECISION_LIMIT_STEPS: f64 = 16.0;

/// How well the 64-bit numbers of the renderer still resolve a view
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Every pixel has its own coordinate, with plenty of room to spare
    Full,
    /// The coordinates of neighboring pixels are only `PRECISION_WARNING_STEPS` or fewer steps of a 64-bit number apart: the image may start to show noise or blocks
    Degrading,
    /// Fewer than `PRECISION_LIMIT_STEPS` steps between neighboring pixels: the image is no longer trustworthy and zooming in further is refused
    Exhausted,
}

/// A view of the complex plane: the center and the scale (see `ComplexPlane::get_scale`)
pub struct View {
    x: f64,
    y: f64,
    scale: f64
}

impl View {
    /// Creates a view with the center `x + yi` and the given scale
    pub const fn new(x: f64, y: f64, scale: f64) -> View {
        View {x,y,scale}
    }
}

impl std::fmt::Debug for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "z = {:?}, scale = {}", Complex::new(self.x, self.y), self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_runs_out_at_the_zoom_where_neighboring_pixels_share_a_coordinate() {
        let at_scale = |scale: f64| {
            let mut plane = ComplexPlane::new(1200, 800);
            plane.set_view(&View::new(-0.75, 0.1, 1.0));
            plane.scale(scale);
            plane
        };
        assert_eq!(ComplexPlane::new(1200, 800).precision(), Precision::Full);
        assert_eq!(at_scale(1e-9).precision(), Precision::Full);
        assert_eq!(at_scale(1e-11).precision(), Precision::Degrading);
        assert_eq!(at_scale(1e-13).precision(), Precision::Exhausted);
        //The meaning of the number: at 1 step or fewer, neighboring pixels really do share a coordinate; with room to spare, they never do
        let distinct_columns = |plane: &ComplexPlane| (0..1199).filter(|x| plane.complex_from_pixel_plane(f64::from(*x), 0.0).x.to_bits() != plane.complex_from_pixel_plane(f64::from(*x + 1), 0.0).x.to_bits()).count();
        assert_eq!(distinct_columns(&at_scale(1e-9)), 1199);
        assert!(at_scale(1e-15).precision_steps() < 1.0 && distinct_columns(&at_scale(1e-15)) < 1199);
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    #[allow(clippy::float_cmp)] // `-2.0`, `0.5` and `min_y == -max_y` are exact by construction
    fn a_new_plane_shows_the_whole_set_with_the_aspect_ratio_of_the_window() {
        let plane = ComplexPlane::new(1200, 800);
        assert_eq!((plane.min_x, plane.max_x, plane.length_x), (-2.0, 0.5, 2.5));
        assert!(close(plane.max_y, 2.5 * (800.0 / 1200.0) / 2.0) && plane.min_y == -plane.max_y);
        assert!(close(plane.increment_x, 2.5 / 1200.0) && close(plane.increment_y, plane.length_y / 800.0));
        assert!(close(plane.increment_x, plane.increment_y), "the pixels are square");
        assert_eq!(plane.get_scale(), 1.0);
    }

    #[test]
    fn pixels_map_to_complex_numbers_with_the_imaginary_axis_pointing_up() {
        let plane = ComplexPlane::new(1200, 800);
        let top_left = plane.complex_from_pixel_plane(0.0, 0.0);
        assert!(close(top_left.x, -2.0) && close(top_left.y, plane.max_y), "the top of the window is the largest imaginary part");
        let bottom_right = plane.complex_from_pixel_plane(1200.0, 800.0);
        assert!(close(bottom_right.x, 0.5) && close(bottom_right.y, plane.min_y));
        let middle = plane.complex_from_pixel_plane(600.0, 400.0);
        assert!(close(middle.x, -0.75) && middle.y.abs() < 1e-15);
    }

    #[test]
    fn a_view_is_set_and_read_back_by_its_center_and_scale() {
        let mut plane = ComplexPlane::new(640, 480);
        plane.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.000064));
        let view = plane.get_view();
        assert!(close(view.x, -0.7498100000000001) && close(view.y, -0.020300000000000054) && close(view.scale, 0.000064), "{view:?}");
        let center = plane.center();
        assert!(close(center.x, view.x) && close(center.y, view.y));
        assert!(close(plane.length_x, 2.5 * 0.000064));
        plane.reset();
        assert_eq!((plane.get_scale(), plane.center().x), (1.0, -0.75));
    }

    #[test]
    fn scaling_keeps_the_center_and_moving_moves_the_bounds() {
        let mut plane = ComplexPlane::new(1200, 800);
        let center = plane.center();
        plane.scale(0.5);
        assert!(close(plane.get_scale(), 0.5) && close(plane.center().x, center.x) && close(plane.center().y, center.y));
        assert!(close(plane.increment_x, 2.5 * 0.5 / 1200.0));
        plane.translate(0.25, -0.125);
        assert!(close(plane.center().x, center.x + 0.25), "moving right by 0.25");
        assert!(close(plane.center().y, center.y + 0.125), "the y bounds are stored with the screen orientation, so the imaginary part moves the other way");
    }

    #[test]
    fn pixel_distances_convert_to_plane_distances_and_back_truncating_towards_zero() {
        let plane = ComplexPlane::new(1200, 800);
        assert!(close(plane.pixels_to_real(10), 10.0 * plane.increment_x) && close(plane.pixels_to_imaginary(10), 10.0 * plane.increment_y));
        assert_eq!(plane.real_to_pixels(10.5 * plane.increment_x), 10);
        assert_eq!(plane.real_to_pixels(-10.5 * plane.increment_x), -10);
        assert_eq!(plane.imaginary_to_pixels(3.0 * plane.increment_y), 3);
    }

    #[test]
    fn the_text_of_a_view_is_what_is_stored_in_the_metadata_of_saved_images() {
        assert_eq!(format!("{:?}", View::new(-0.75, 0.1, 0.25)), "z = -0.75 + 0.1i, scale = 0.25");
    }
}

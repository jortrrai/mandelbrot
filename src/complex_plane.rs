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
    /// How far the image is turned counterclockwise around the center of the view, in radians from -π to π (issue #217); 0 for most views, which are then
    /// computed exactly as before rotation existed
    rotation: f64,
    /// The cosine and the sine of `rotation`
    rotation_cos: f64,
    rotation_sin: f64,
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
            rotation: 0.0,
            rotation_cos: 1.0,
            rotation_sin: 0.0,
        }
    }

    /// The plane for a window of `width` x `height` pixels that shows the same part of the real axis and has the same center.
    /// The real bounds are copied exactly, so resizing a window again and again never drifts; the imaginary bounds follow the new aspect ratio, so the pixels stay square.
    pub fn resized(&self, width: usize, height: usize) -> ComplexPlane {
        let (float_width, float_height) = (f64::from(width as u32), f64::from(height as u32));
        let length_y = self.length_x * (float_height / float_width);
        let middle_y = self.min_y + (self.max_y - self.min_y) / 2.0; //Stored with the orientation of the screen, like the bounds
        let (min_y, max_y) = (middle_y - length_y / 2.0, middle_y + length_y / 2.0);
        ComplexPlane { min_x: self.min_x, max_x: self.max_x, length_x: self.length_x, min_y, max_y, length_y: max_y - min_y, increment_x: self.length_x / float_width, increment_y: (max_y - min_y) / float_height, width, height, ..*self }
    }

    /// How far the image is turned counterclockwise around the center, in radians from -π to π (see `set_rotation`)
    pub const fn rotation(&self) -> f64 {
        self.rotation
    }

    /// Turns the image counterclockwise by `radians` around the center of the view (any angle; it is kept from -π to π, so a whole turn is exactly no turn).
    /// The center stays exactly where it is
    pub fn set_rotation(&mut self, radians: f64) {
        let turned = radians.rem_euclid(std::f64::consts::TAU);
        let turned = if turned >= std::f64::consts::PI { turned - std::f64::consts::TAU } else { turned };
        self.rotation = if turned.is_finite() { turned } else { 0.0 };
        (self.rotation_sin, self.rotation_cos) = if self.rotation == 0.0 { (0.0, 1.0) } else { self.rotation.sin_cos() };
    }

    /// The view as it would be without its rotation, turned back around the origin of the plane: the same size and pixels, with the screen's axes as its own.
    /// Two views with the same rotation compare as their unturned views do (how far apart in pixels, where one is in the other), so the code that shifts
    /// buffers and draws previews works on these
    pub fn unturned(&self) -> ComplexPlane {
        if self.rotation == 0.0 {
            return self.clone();
        }
        let center = self.center();
        //The center turned back by the rotation: R(θ) · center, the inverse of the turn that `complex_from_pixel_plane` applies
        let back = Complex::new(center.x * self.rotation_cos - center.y * self.rotation_sin, center.x * self.rotation_sin + center.y * self.rotation_cos);
        let mut plane = ComplexPlane { rotation: 0.0, rotation_cos: 1.0, rotation_sin: 0.0, ..self.clone() };
        plane.set_center(&back);
        plane
    }

    /// The point `point` turned back like `unturned` turns the view: where it is in the unturned view
    pub fn unturned_point(&self, point: &Complex) -> Complex {
        Complex::new(point.x * self.rotation_cos - point.y * self.rotation_sin, point.x * self.rotation_sin + point.y * self.rotation_cos)
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
        if self.rotation == 0.0 {
            return Complex::new(x, y);
        }
        //A turned view: the image turned counterclockwise by the rotation shows at this pixel the point turned clockwise around the center, R(-θ) · offset
        let center = self.center();
        let (dx, dy) = (x - center.x, y - center.y);
        Complex::new(center.x + dx.mul_add(self.rotation_cos, dy * self.rotation_sin), center.y + dy.mul_add(self.rotation_cos, -(dx * self.rotation_sin)))
    }

    /// The offset in the plane of a move of `right` and `up` (in the plane's units, along the screen's axes): turned like the view
    pub fn screen_offset(&self, right: f64, up: f64) -> (f64, f64) {
        if self.rotation == 0.0 {
            return (right, up);
        }
        (right.mul_add(self.rotation_cos, up * self.rotation_sin), up.mul_add(self.rotation_cos, -(right * self.rotation_sin)))
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

    /// Set the Complex plane at Center (x,y) at the given scale, where scale == 1 => max_x-min_x=2.5. The rotation stays as it is
    pub fn set_view_separated(&mut self, x: f64, y: f64, scale: f64) {
        let rotation = self.rotation;
        self.reset();
        self.set_center(&Complex::new(x, y));
        self.scale(scale);
        self.set_rotation(rotation);
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
    #[allow(clippy::float_cmp)] // The real bounds are copied, so they are exactly equal
    fn resizing_keeps_the_real_bounds_and_the_center_and_makes_the_pixels_square() {
        let mut plane = ComplexPlane::new(1200, 800);
        plane.set_view(&View::new(-0.7498, 0.0203, 0.00064));
        let wide = plane.resized(1600, 600);
        assert_eq!((wide.min_x, wide.max_x, wide.length_x), (plane.min_x, plane.max_x, plane.length_x));
        assert!(close(wide.center().x, plane.center().x) && close(wide.center().y, plane.center().y));
        assert!(close(wide.increment_x, wide.increment_y), "the pixels are square");
        assert!(close(wide.length_y, wide.length_x * 600.0 / 1600.0));
        assert!(close(wide.get_scale(), plane.get_scale()), "the zoom is about the real axis, so it does not change");
        //Resizing to the same size changes nothing, and resizing back and forth does not drift
        let same = plane.resized(1200, 800);
        assert_eq!((same.min_y, same.max_y, same.increment_x), (plane.min_y, plane.max_y, plane.increment_x));
        let mut drifting = plane.clone();
        for step in 0..1000 {
            drifting = drifting.resized(300 + step % 7 * 100, 200 + step % 5 * 90);
        }
        assert_eq!((drifting.min_x, drifting.max_x), (plane.min_x, plane.max_x));
        assert!(drifting.center().y.to_bits().abs_diff(plane.center().y.to_bits()) < 1000, "{} {}", drifting.center().y, plane.center().y);
    }

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

    #[test]
    fn a_turned_view_turns_the_image_around_its_center_which_stays_exactly_where_it_is() {
        //Issue #217
        let mut plane = ComplexPlane::new(200, 100);
        plane.set_view(&View::new(-0.75, 0.1, 0.01));
        let unturned: Vec<Complex> = (0..5).map(|k| plane.complex_from_pixel_plane(f64::from(k) * 37.5, f64::from(k) * 21.0)).collect();
        let center = plane.center();
        plane.set_rotation(std::f64::consts::FRAC_PI_2);
        assert_eq!((plane.center().x, plane.center().y), (center.x, center.y), "the center does not move");
        //Turned a quarter counterclockwise: the pixel 10 to the right of the middle shows what was 10 below it
        let right = plane.complex_from_pixel_plane(110.0, 50.0);
        let increment = plane.increment_x;
        assert!((right.x - center.x).abs() < 1e-12 && (right.y - (center.y - 10.0 * increment)).abs() < 1e-12, "{right:?}");
        let middle = plane.complex_from_pixel_plane(100.0, 50.0);
        assert!((middle.x - center.x).abs() < 1e-15 && (middle.y - center.y).abs() < 1e-15);
        //A whole turn is exactly no turn: the pixels of the unturned view, bit for bit
        plane.set_rotation(std::f64::consts::TAU);
        assert_eq!(plane.rotation(), 0.0);
        let again: Vec<Complex> = (0..5).map(|k| plane.complex_from_pixel_plane(f64::from(k) * 37.5, f64::from(k) * 21.0)).collect();
        assert!(again.iter().zip(&unturned).all(|(a, b)| a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()));
        //Any angle is kept from -π to π; a new view or size keeps the rotation
        plane.set_rotation(3.0 * std::f64::consts::PI / 2.0);
        assert!((plane.rotation() + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        plane.set_view(&View::new(0.3, 0.0, 0.5));
        assert!((plane.rotation() + std::f64::consts::FRAC_PI_2).abs() < 1e-12, "set_view keeps it");
        assert_eq!(plane.resized(300, 300).rotation(), plane.rotation(), "a resize keeps it");
    }

    #[test]
    fn a_turned_view_moves_along_the_screen_and_its_moves_are_whole_pixels() {
        let mut plane = ComplexPlane::new(160, 120);
        plane.set_view(&View::new(-0.75, 0.1, 0.02));
        plane.set_rotation(0.7);
        let mut moved = plane.clone();
        crate::rendering::move_complex_plane_by_pixels(&mut moved, 3, -5);
        //The same shift of the buffer as the same move of the unturned view (the buffer moves opposite to the view, see `pixel_shift`)
        let mut unturned = plane.clone();
        unturned.set_rotation(0.0);
        let mut unturned_moved = unturned.clone();
        crate::rendering::move_complex_plane_by_pixels(&mut unturned_moved, 3, -5);
        assert_eq!(crate::rendering::pixel_shift(&plane, &moved), crate::rendering::pixel_shift(&unturned, &unturned_moved), "the move is found again, so the samples move along");
        assert!(crate::rendering::pixel_shift(&plane, &moved).is_some());
        //The point at a pixel of the moved view is the point 5 columns to the left and 3 rows up of the same pixel in the view before
        let (a, b) = (moved.complex_from_pixel_plane(80.0, 60.0), plane.complex_from_pixel_plane(75.0, 57.0));
        assert!((a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12, "{a:?} {b:?}");
        let mut turned = plane.clone();
        turned.set_rotation(0.8);
        assert_eq!(crate::rendering::pixel_shift(&plane, &turned), None, "another rotation is not a move");
    }
}

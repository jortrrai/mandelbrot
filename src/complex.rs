use std::fmt;

///Complex number (x + bi), i^2 = -1: <https://en.wikipedia.org/wiki/Complex_number>
pub struct Complex {
    /// Real part
    pub x: f64,
    /// Imaginary part
    pub y: f64,
}

impl Complex {
    pub const fn new(x: f64, y: f64) -> Complex {
        Complex { x, y }
    }

    ///Addition
    pub fn add(&self, c: &Complex) -> Complex {
        let new_x = self.x + c.x;
        let new_y = self.y + c.y;
        Complex { x: new_x, y: new_y }
    }

    ///Subtraction, add the negative of c
    pub fn subtract(&self, c: &Complex) -> Complex {
        let negated_c = c.multiply_real(-1.0);
        self.add(&negated_c)
    }

    ///Multiply with a real
    pub fn multiply_real(&self, r: f64) -> Complex {
        let new_x = r * self.x;
        let new_y = r * self.y;
        Complex { x: new_x, y: new_y }
    }

    ///Multiply with a complex
    pub fn multiply(&self, c: &Complex) -> Complex {
        let new_x = self.x * c.x - self.y * c.y;
        let new_y = self.x * c.y + self.y * c.x;
        Complex { x: new_x, y: new_y }
    }

    ///Square the complex
    pub fn squared(&self) -> Complex {
        let new_x = self.x * self.x - self.y * self.y;
        let new_y = 2.0 * self.x * self.y;
        Complex { x: new_x, y: new_y }
    }

    ///Calculate the absolute value of the complex (Pythagorean length of the complex, seen as x vector in the complex plane)
    pub fn abs(&self) -> f64 {
        f64::sqrt(self.x * self.x + self.y * self.y)
    }
}

impl fmt::Debug for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {}i",
            self.x,
            if self.y > 0.0 { '+' } else { '-' },
            self.y.abs()
        )
    }
}
//Complex

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(c: &Complex) -> (f64, f64) {
        (c.x, c.y)
    }

    #[test]
    fn arithmetic_follows_the_rules_of_complex_numbers() {
        let (a, b) = (Complex::new(1.0, 2.0), Complex::new(3.0, -4.0));
        assert_eq!(parts(&a.add(&b)), (4.0, -2.0));
        assert_eq!(parts(&a.subtract(&b)), (-2.0, 6.0));
        assert_eq!(parts(&a.multiply_real(2.5)), (2.5, 5.0));
        assert_eq!(parts(&a.multiply(&b)), (11.0, 2.0), "(1 + 2i)(3 - 4i) = 11 + 2i");
        assert_eq!(parts(&a.squared()), (-3.0, 4.0), "(1 + 2i)^2 = -3 + 4i");
        assert_eq!(Complex::new(3.0, 4.0).abs(), 5.0);
    }

    #[test]
    fn squared_uses_exactly_the_arithmetic_of_the_iteration() {
        //The renderer relies on `z.squared().add(c)` being x*x - y*y + cx and 2*x*y + cy, evaluated in this order
        let (z, c) = (Complex::new(0.1, 0.7), Complex::new(-0.75, 0.1));
        let next = z.squared().add(&c);
        assert_eq!(next.x.to_bits(), (0.1_f64 * 0.1 - 0.7 * 0.7 + -0.75).to_bits());
        assert_eq!(next.y.to_bits(), (2.0_f64 * 0.1 * 0.7 + 0.1).to_bits());
    }

    #[test]
    fn debug_prints_the_signed_imaginary_part_as_it_always_has() {
        assert_eq!(format!("{:?}", Complex::new(1.5, 2.0)), "1.5 + 2i");
        assert_eq!(format!("{:?}", Complex::new(1.5, -2.0)), "1.5 - 2i");
        //A quirk that is kept because the text ends up in the metadata of saved images: an imaginary part of exactly 0 is printed with a minus sign
        assert_eq!(format!("{:?}", Complex::new(-0.75, 0.0)), "-0.75 - 0i");
    }
}

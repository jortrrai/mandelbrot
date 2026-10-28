
#[derive(Clone)]
pub struct PixelPlane {
    pub width: usize,
    pub height: usize,
    aspect_ratio_width: usize,
    aspect_ratio_height: usize
}

impl PixelPlane {
    pub fn new(width: usize, height: usize) -> PixelPlane{
        let gcd = num::integer::gcd(width, height); //Needed to compute the aspect ratio of the pixel plane
        PixelPlane {width, height, aspect_ratio_width: width/gcd, aspect_ratio_height: height/gcd}
    }

    /// Prints: "Pixel plane: size is {}x{} and aspect ratio is {}:{}", `self.width`, `self.height`, `self.aspect_ratio_width`, `self.aspect_ratio_height`)
    pub fn print(&self) {
        println!("Pixel plane: size is {}x{} and aspect ratio is {}:{}",self.width, self.height, self.aspect_ratio_width, self.aspect_ratio_height);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_aspect_ratio_is_the_size_without_its_common_divisor() {
        let plane = PixelPlane::new(1200, 800);
        assert_eq!((plane.width, plane.height, plane.aspect_ratio_width, plane.aspect_ratio_height), (1200, 800, 3, 2));
        let plane = PixelPlane::new(64, 48);
        assert_eq!((plane.aspect_ratio_width, plane.aspect_ratio_height), (4, 3));
        let plane = PixelPlane::new(7, 7);
        assert_eq!((plane.aspect_ratio_width, plane.aspect_ratio_height), (1, 1));
    }
}

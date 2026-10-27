/// The size of the pixels of an image: `width` columns and `height` rows
#[derive(Clone)]
pub struct PixelPlane {
    /// The amount of columns
    pub width: usize,
    /// The amount of rows
    pub height: usize,
}

impl PixelPlane {
    /// A pixel plane of `width` x `height` pixels
    pub fn new(width: usize, height: usize) -> PixelPlane {
        PixelPlane { width, height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pixel_plane_has_the_given_size() {
        let plane = PixelPlane::new(1200, 800);
        assert_eq!((plane.width, plane.height), (1200, 800));
    }
}

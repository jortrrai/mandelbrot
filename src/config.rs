//! The settings the window starts with.
//!
//! The program has no command-line arguments: the window has controls for all of them (the size follows the window, the iterations, the quality and the size
//! of a saved image).

//The defaults
const WIDTH: usize = 1200;
const HEIGHT: usize = 800;
const MAX_ITERATIONS: u32 = 1000;
const ORBIT_RADIUS: f64 = 2.0;
const SUPERSAMPLING_AMOUNT: u8 = 32;

/// The settings the window starts with: the size of the window and of a saved image, the iterations and the supersampling
#[derive(Clone, Debug)]
pub struct Config {
    // Window dimensions in pixels
    /// The width of the window (the render) in pixels at the start
    pub window_width: usize,
    /// The height of the window (the render) in pixels at the start
    pub window_height: usize,
    // Mandelbrot set parameters
    /// The most iterations of the escape-time iteration of a point
    pub max_iterations: u32,
    /// The radius that the orbit of a point has to leave to count as escaped
    pub orbit_radius: f64,      //If z remains within the orbit_radius in max_iterations, we assume c does not tend to infinity
    // Rendering parameters
    /// The amount of samples per pixel, from 1
    pub supersampling_amount: u8,
    /// The default size in pixels of a saved image
    pub image_width: usize,
    /// The default height in pixels of a saved image
    pub image_height: usize,
    /// Whether the window starts with the render size following the size of the window (always, except in tests that need a fixed small size)
    pub fit_window: bool,
}


impl Default for Config {
    /// The settings the window starts with
    fn default() -> Config {
        Config { window_width: WIDTH, window_height: HEIGHT, max_iterations: MAX_ITERATIONS, orbit_radius: ORBIT_RADIUS, supersampling_amount: SUPERSAMPLING_AMOUNT, image_width: WIDTH, image_height: HEIGHT, fit_window: true }
    }
}

impl Config {
    /// The settings of a window with a fixed render size of `width` x `height` pixels (which does not follow the window), `max_iterations` and `supersampling_amount`,
    /// for tests that need a small, fast window
    #[cfg(test)]
    pub(crate) fn sized(width: usize, height: usize, max_iterations: u32, supersampling_amount: u8) -> Config {
        Config { window_width: width, window_height: height, image_width: width, image_height: height, max_iterations, supersampling_amount, fit_window: false, ..Config::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults() {
        let config = Config::default();
        assert_eq!((config.window_width, config.window_height, config.image_width, config.image_height), (1200, 800, 1200, 800));
        assert_eq!((config.max_iterations, config.supersampling_amount, config.orbit_radius), (1000, 32, 2.0));
        assert!(config.fit_window, "the render size follows the window");
    }

    #[test]
    fn a_sized_config_keeps_its_size() {
        let config = Config::sized(64, 48, 50, 3);
        assert_eq!((config.window_width, config.window_height, config.image_width, config.image_height, config.max_iterations, config.supersampling_amount), (64, 48, 64, 48, 50, 3));
        assert!(!config.fit_window);
    }
}

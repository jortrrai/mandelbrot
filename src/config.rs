//! The configuration that the command line gives: `cargo run --release -- <width> <height> <max_iterations> <supersampling_amount> <window_scale>`.

use std::{str::FromStr, fmt::{Display, self}};

//Argument default values
static WIDTH: usize = 1200;
static HEIGHT: usize = 800;
static MAX_ITERATIONS: u32 = 1000;
static ORBIT_RADIUS: f64 = 2.0;
static SUPERSAMPLING_AMOUNT: u8 = 1;
static WINDOW_SCALE: f64 = 1.0;

/// The settings that the command line gives: the size of the window and the image, the iterations, the supersampling and the scale of the window
#[derive(Clone)]
pub struct Config {
    // Window dimensions in pixels
    /// The width of the window in pixels (the image width times the window scale)
    pub window_width: usize,
    /// The height of the window in pixels (the image height times the window scale)
    pub window_height: usize,
    // Mandelbrot set parameters
    /// The most iterations of the escape-time iteration of a point
    pub max_iterations: u32,
    /// The radius that the orbit of a point has to leave to count as escaped
    pub orbit_radius: f64,      //If z remains within the orbit_radius in max_iterations, we assume c does not tend to infinity
    // Rendering parameters
    /// The amount of samples per pixel, from 1
    pub supersampling_amount: u8,
    //Window scaling factor
    /// The scale of the window relative to the image
    pub window_scale: f64,
    //Scaled window dimensions in pixels (used in images)
    /// The width in pixels of a saved image
    pub image_width: usize,
    /// The height in pixels of a saved image
    pub image_height: usize
}


impl Config {
    /// Parse the command line arguments from e.g. `env::args` in the following format
    /// ```ignore
    /// cargo run -- width height max_iterations
    /// ```
    /// # Errors
    /// Returns an Error if any of the given arguments couldn't be parsed into their types
    pub fn build(mut args: impl Iterator<Item = String>) -> Result<Config, String> {
        args.next(); //Skip the first argument as it is the name of the executable

        //First argument
        let image_width = Config::parse_argument("width", args.next(), WIDTH)?; 

        //Second argument
        let image_height = Config::parse_argument("height", args.next(), HEIGHT)?;

        //Third argument
        let max_iterations = Config::parse_argument("max_iterations", args.next(), MAX_ITERATIONS)?;

        //Fourth argument
        let supersampling_amount = Config::parse_argument("supersampling_amount", args.next(), SUPERSAMPLING_AMOUNT)?;

        //Fifth argument
        let window_scale = Config::parse_argument("window_scale", args.next(), WINDOW_SCALE)?;
        //Scale width and height
        let window_width = (f64::from(image_width as u32) * window_scale) as usize;
        let window_height = (f64::from(image_height as u32) * window_scale) as usize;

        Ok(Config {window_width, window_height, max_iterations, orbit_radius: ORBIT_RADIUS, supersampling_amount, window_scale, image_width, image_height})
    }

    ///Parses an argument to a T value if possible, returns an error if not. Returns default if argument is None\
    ///If Some(arg) == "-", return default
    /// # Errors
    /// Return an Error if the given argument cannot be parsed to a T type
    pub fn parse_argument<T: FromStr + Display>(name: &str, argument: Option<String>, default: T) -> Result<T, String> 
    where <T as std::str::FromStr>::Err: Display{
        match argument {
            Some(arg) => {
                if arg == "-" {
                    Config::print_no_argument_given(name, &default);
                    return Ok(default);
                }
                match arg.parse::<T>() {
                    Ok(val) => Ok(val),
                    Err(err) => Err(err.to_string() + &format!(" for {} argument", name)),
                }
            },
            None =>  {
                Config::print_no_argument_given(name, &default);
                Ok(default)
            }
        }
    }

    /// Prints that the argument `name` was not given and that `default` is used
    pub fn print_no_argument_given<T: std::fmt::Display>(name: &str, default: &T) {
        println!("No {} argument given, using default: {}", name, default);
    }
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { //TODO: Improve debug printing format legibility
        f.debug_struct("Config").field("window_width", &self.window_width).field("window_height", &self.window_height).field("max_iterations", &self.max_iterations).field("orbit_radius", &self.orbit_radius).field("supersampling_amount", &self.supersampling_amount).field("window_scale", &self.window_scale).field("image_width", &self.image_width).field("image_height", &self.image_height).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(arguments: &[&str]) -> Result<Config, String> {
        Config::build(std::iter::once("mandelbrot").chain(arguments.iter().copied()).map(String::from))
    }

    #[test]
    fn without_arguments_the_defaults_are_used() {
        let config = build(&[]).unwrap();
        assert_eq!((config.window_width, config.window_height, config.image_width, config.image_height), (1200, 800, 1200, 800));
        assert_eq!((config.max_iterations, config.supersampling_amount), (1000, 1));
        assert_eq!((config.orbit_radius, config.window_scale), (2.0, 1.0));
    }

    #[test]
    fn every_argument_is_read_in_order_and_a_dash_means_the_default() {
        let config = build(&["640", "-", "250", "4", "0.5"]).unwrap();
        assert_eq!((config.image_width, config.image_height), (640, 800));
        assert_eq!((config.window_width, config.window_height), (320, 400), "the window is scaled, the saved image is not");
        assert_eq!((config.max_iterations, config.supersampling_amount, config.window_scale), (250, 4, 0.5));
    }

    #[test]
    fn an_argument_that_cannot_be_parsed_is_reported_with_its_name() {
        assert_eq!(build(&["abc"]).unwrap_err(), "invalid digit found in string for width argument");
        assert!(build(&["10", "10", "x"]).unwrap_err().ends_with("for max_iterations argument"));
        assert!(build(&["10", "10", "10", "300"]).unwrap_err().ends_with("for supersampling_amount argument"), "256 and more do not fit in a u8");
        assert!(build(&["10", "10", "10", "1", "wide"]).unwrap_err().ends_with("for window_scale argument"));
    }
}

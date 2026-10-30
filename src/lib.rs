//! Mandelbrot by Jort: a viewer of the Mandelbrot set with a graphical window.
//!
//! The modules, from the mathematics to the window:
//! * [`complex`], [`complex_plane`]: complex numbers, and the part of the complex plane that an image shows.
//! * [`mandelbrot_set`]: the escape-time iteration, scalar and four points at a time, with the interior shortcuts and the smooth (continuous) iteration count.
//! * [`coloring`], [`pixel_buffer`]: the palettes, the channel mappings, and the buffer of pixels with saving to PNG.
//! * [`rendering`]: the progressive and adaptive renderers that turn a view into colors.
//! * [`auto_iterations`], [`thumbnails`], [`points_of_interest`]: the automatic iteration limit, the thumbnails and the named places.
//! * [`resolution`]: the image resolutions: built-in presets, the custom presets of the user and the file that keeps them.
//! * [`gui`]: the window. [`config`] holds the settings the window starts with, [`interaction`] holds the move and zoom steps.
//!
//! The same view and settings always render the same image: the supersampling jitter is a hash of the pixel and the sample, never random.
#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::perf,
    clippy::correctness,
    clippy::suspicious,
    clippy::complexity
)]
#![allow(
    // Image sizes, pixel counts and iteration counts are far below 2^23, so converting them to floats is exact
    clippy::cast_precision_loss,
    clippy::must_use_candidate,
    clippy::multiple_crate_versions,
    clippy::uninlined_format_args,
    clippy::let_and_return,
    clippy::missing_const_for_fn,
    clippy::use_self,
    clippy::cast_possible_truncation,
    clippy::module_name_repetitions,
    clippy::needless_return,
    clippy::return_self_not_must_use,
    clippy::unreadable_literal,
    clippy::single_match_else,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    clippy::cast_sign_loss
)]

use std::error::Error;

use coloring::ColorChannelMapping;
pub use config::Config;
pub use interaction::InteractionVariables;

use crate::coloring::TrueColor;

mod banner;
pub mod complex_plane;
pub mod complex;
pub mod pixel_buffer;
pub mod mandelbrot_set;
pub mod rendering;
pub mod auto_iterations;
pub mod coloring;
pub mod gui;
pub mod config;
pub mod interaction;
pub mod points_of_interest;
pub mod resolution;
pub mod thumbnails;

/// A function that colors a pixel from its iteration count and the maximum iterations, e.g. `TrueColor::new_from_bernstein_polynomials`
type ColoringFunction = fn(position: f64, length: f64) -> TrueColor;
/// The coloring that the window starts with
const COLORING_FUNCTION: ColoringFunction = TrueColor::new_from_bernstein_polynomials;

/// The color channel mapping that the window starts with
const COLOR_CHANNEL_MAPPING: ColorChannelMapping = ColorChannelMapping::RGB;

/// Runs the application: prints the banner, then opens the window and runs its event loop until it is closed.
/// Everything except reading the configuration and reporting errors, which `main` does.
/// # Errors
/// Returns an Error if the GUI window cannot be created or the GUI event loop fails
pub fn run(config: &Config) -> Result<(), Box<dyn Error>> {
    banner::print_banner();

    let mut app = gui::MandelbrotApp::new(config);
    if app.use_settings_file(gui::settings_path()) {
        app.render_with_settings();
    }
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Mandelbrot set viewer")
            .with_inner_size([config.window_width as f32 + gui::SIDE_PANEL_WIDTH, (config.window_height as f32).max(gui::MIN_WINDOW_HEIGHT)]),
        ..Default::default()
    };
    eframe::run_native("Mandelbrot set viewer", options, Box::new(|creation_context| {
        app.use_gl(creation_context.gl.as_deref());
        Ok(Box::new(app))
    }))?;
    Ok(())
}

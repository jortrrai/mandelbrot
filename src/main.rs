//! The application: runs the window of the `mandelbrot` library with the default settings.

use std::process;

use mandelbrot::{Config, self};

fn main() {
    if std::env::args().len() > 1 {
        eprintln!("The command-line arguments are no longer used: the window has controls for the size, the iterations and the quality. Starting with the defaults.");
    }
    if let Err(err) = mandelbrot::run(&Config::default()) {
        eprintln!("Application error: {}", err);
        process::exit(1);
    }
}

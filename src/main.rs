//! The application: runs the window of the `mandelbrot` library with the default settings.

use std::process;

use mandelbrot::{Config, self};

fn main() {
    //With arguments, the command line (see `gui::cli`: mandelbrot video …); without, the window
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if !arguments.is_empty() {
        process::exit(mandelbrot::gui::command_line(&arguments));
    }
    if let Err(err) = mandelbrot::run(&Config::default()) {
        eprintln!("Application error: {}", err);
        process::exit(1);
    }
}

//! The application: runs the window of the `mandelbrot` library with the default settings.

use std::process;

use mandelbrot::{Config, self};

fn main() {
    //With arguments, the command line (see `gui::cli`: mandelbrot video …); without, the window
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if !arguments.is_empty() {
        let code = mandelbrot::gui::command_line(&arguments);
        //Renders still under way in the background must not use the GPU while the program ends (issue #200)
        mandelbrot::rendering::gpu::close_gpus();
        process::exit(code);
    }
    if let Err(err) = mandelbrot::run(&Config::default()) {
        eprintln!("Application error: {}", err);
        process::exit(1);
    }
}

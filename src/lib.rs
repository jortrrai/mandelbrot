#![warn(
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

use crate::coloring::TrueColor;
use crate::complex_plane::View;

pub mod complex_plane;
pub mod complex;
pub mod pixel_buffer;
pub mod mandelbrot_set;
pub mod rendering;
pub mod auto_iterations;
pub mod coloring;
pub mod gui;
pub mod config;

//Coloring function
type ColoringFunction = fn(iterations: u32, max_iterations: u32) -> TrueColor;
static COLORING_FUNCTION : ColoringFunction = TrueColor::new_from_bernstein_polynomials;

//Color channel mapping
static COLOR_CHANNEL_MAPPING: ColorChannelMapping = ColorChannelMapping::RGB;

//Views
static VIEW_1: View = View::new(-0.6604166666666667, 0.4437500000000001, 0.1);
static VIEW_2: View = View::new(-1.0591666666666668, 0.2629166666666668, 0.01);
static VIEW_3: View = View::new(-0.4624999999999999, 0.55, 0.1);
static VIEW_4: View = View::new(-0.46395833333333325, 0.5531250000000001, 0.03);
static VIEW_5: View = View::new(-0.4375218333333333, 0.5632133750000003, 0.00002000000000000002);
static VIEW_6: View = View::new(-0.7498100000000001, -0.020300000000000054, 0.00006400000000000002);
static VIEW_7: View = View::new(-1.7862712000000047, 0.000052399999999991516, 0.00001677721600000001); 
static VIEW_8: View = View::new(-1.7862581627050718, 0.00005198056959995248, 0.000006039797760000003); 
static VIEW_9: View = View::new( -0.4687339999999999, 0.5425518958333333, 0.000010000000000000003);
static VIEW_0: View = View::new( -0.437520465811966, 0.5632133750000006, 0.000004000000000000004);

//Banner values
static VERSION: &str = "1.4";

pub struct InteractionVariables{
    ///Variable determining the amount of rows and columns are translated by pressing the 4 arrow keys
    pub translation_amount: u8,
    ///Variable denoting the user scaling speed; the lower this value, the more aggressive the zooming will become
    pub scale_denominator: f64,
    pub scale_numerator: f64,
}

impl InteractionVariables{
    pub fn new(translation_amount: u8, scale_numerator: f64, scale_denominator: f64) -> InteractionVariables {
        InteractionVariables { translation_amount, scale_denominator, scale_numerator }
    }

    pub fn scaling_factor(&self) -> f64 {
        self.scale_numerator / self.scale_denominator
    }

    pub fn inverse_scaling_factor(&self) -> f64 {
        self.scale_denominator / self.scale_numerator
    }

    pub fn increment_translation_amount(&mut self) {
        self.translation_amount = self.translation_amount.saturating_add(1);
    }

    pub fn decrement_translation_amount(&mut self) {
        if self.translation_amount > 1 {
            self.translation_amount -=1;
        }
    }

    pub fn increment_scale_numerator(&mut self) {
        if self.scale_numerator < 9.0 {
            self.scale_numerator += 1.0;
        }
    }

    pub fn decrement_scale_numerator(&mut self) {
        if self.scale_numerator > 1.0 {
            self.scale_numerator -= 1.0;
        }
    }
}

impl Default for InteractionVariables{
    fn default() -> Self {
        InteractionVariables { translation_amount:10, scale_numerator: 9.0, scale_denominator: 10.0 }
    }
}

///Prints Mandelbrot ASCII art :) </br> 
///Prints the `application_banner`, `author_banner`, and `version`
fn print_banner()
{
//Made using: https://patorjk.com/software/taag/#p=display&f=Big&t=Mandelbrot
let application_banner = r"
__  __                 _      _ _               _   
|  \/  |               | |    | | |             | |  
| \  / | __ _ _ __   __| | ___| | |__  _ __ ___ | |_ 
| |\/| |/ _` | '_ \ / _` |/ _ \ | '_ \| '__/ _ \| __|
| |  | | (_| | | | | (_| |  __/ | |_) | | | (_) | |_ 
|_|  |_|\__,_|_| |_|\__,_|\___|_|_.__/|_|  \___/ \__|";
//Made using: https://patorjk.com/software/taag/#p=display&f=Small%20Slant&t=by%20Jort
let author_banner = r"
   __             __         __ 
  / /  __ __  __ / /__  ____/ /_
 / _ \/ // / / // / _ \/ __/ __/
/_.__/\_, /  \___/\___/_/  \__/ 
     /___/                      ";
let version = VERSION;
println!("{}{}v{}\n\n", application_banner, author_banner, version);
}

///Prints a command info tip for the users benefit
fn print_command_info() {
    let tip = "Run Mandelbrot using:";
    let command = "cargo run --release -- <width> <height> <max_iterations> <supersampling_amount> <window_scale>";
    let command_info = "where <arg> means substitute with the value of arg\nuse '-' to use the default value of arg";
    println!("{}\n\t{}\n{}\n",tip, command, command_info);
}

///Holds all the logic currently in the main function that isn't involved with setting up configuration or handling errors, to make `main` concise and
///easy to verify by inspection
/// # Errors
/// Returns an Error if the GUI window cannot be created or the GUI event loop fails
pub fn run(config: &Config) -> Result<(), Box<dyn Error>> {
    //Print the banner
    print_banner();
    //Print command info
    print_command_info();

    let app = gui::MandelbrotApp::new(config);
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Mandelbrot set viewer")
            .with_inner_size([config.window_width as f32 + gui::SIDE_PANEL_WIDTH, (config.window_height as f32).max(gui::MIN_WINDOW_HEIGHT)]),
        ..Default::default()
    };
    eframe::run_native("Mandelbrot set viewer", options, Box::new(|_creation_context| Ok(Box::new(app))))?;
    Ok(())
}

//! Colors: the true color of a pixel, the palettes that turn an iteration count into a color, and the mappings of the color channels.

use std::{fmt, str::FromStr};

use angular_units::Deg;
use num::traits::Pow;
use prisma::{Hsv, Rgb, FromColor};

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
///A mapping from ColorChannelMapping -> RGB, the first character denotes the new red channel, the second character the new green channel,
/// the third character the new blue channel.\
/// E.g: Ok(ColorChannelMapping::BGR) means that red will get the value of blue, green the value of green, and blue the value of red:
/// B => R, G => G, R => B.
pub enum ColorChannelMapping {
    /// Red takes the blue channel, green the blue channel and blue the blue channel
    BBB,
    /// Red takes the blue channel, green the blue channel and blue the green channel
    BBG,
    /// Red takes the blue channel, green the blue channel and blue the red channel
    BBR,
    /// Red takes the blue channel, green the green channel and blue the blue channel
    BGB,
    /// Red takes the blue channel, green the green channel and blue the green channel
    BGG,
    /// Red takes the blue channel, green the green channel and blue the red channel
    BGR,
    /// Red takes the blue channel, green the red channel and blue the blue channel
    BRB,
    /// Red takes the blue channel, green the red channel and blue the green channel
    BRG,
    /// Red takes the blue channel, green the red channel and blue the red channel
    BRR,
    /// Red takes the green channel, green the blue channel and blue the blue channel
    GBB,
    /// Red takes the green channel, green the blue channel and blue the green channel
    GBG,
    /// Red takes the green channel, green the blue channel and blue the red channel
    GBR,
    /// Red takes the green channel, green the green channel and blue the blue channel
    GGB,
    /// Red takes the green channel, green the green channel and blue the green channel
    GGG,
    /// Red takes the green channel, green the green channel and blue the red channel
    GGR,
    /// Red takes the green channel, green the red channel and blue the blue channel
    GRB,
    /// Red takes the green channel, green the red channel and blue the green channel
    GRG,
    /// Red takes the green channel, green the red channel and blue the red channel
    GRR,
    /// Red takes the red channel, green the blue channel and blue the blue channel
    RBB,
    /// Red takes the red channel, green the blue channel and blue the green channel
    RBG,
    /// Red takes the red channel, green the blue channel and blue the red channel
    RBR,
    /// Red takes the red channel, green the green channel and blue the green channel
    RGG,
    /// Red takes the red channel, green the green channel and blue the red channel
    RGR,
    /// Red takes the red channel, green the red channel and blue the blue channel
    RRB,
    /// Red takes the red channel, green the red channel and blue the green channel
    RRG,
    /// Red takes the red channel, green the red channel and blue the red channel
    RRR,
    /// Every channel keeps its own value (the identity)
    RGB,
}

impl ColorChannelMapping {
    /// Parses a mapping such as "RGB" or "bgr" (case-insensitive)
    /// # Errors
    /// Returns an error message if `r_g_b` is not three characters from `R`, `G` and `B`
    pub fn new(r_g_b: &str) -> Result<ColorChannelMapping,String> {
        match &r_g_b.to_uppercase()[..] {
            "BBB" => Ok(ColorChannelMapping::BBB),
            "BBG" => Ok(ColorChannelMapping::BBG),
            "BBR" => Ok(ColorChannelMapping::BBR),
            "BGB" => Ok(ColorChannelMapping::BGB),
            "BGG" => Ok(ColorChannelMapping::BGG),
            "BGR" => Ok(ColorChannelMapping::BGR),
            "BRB" => Ok(ColorChannelMapping::BRB),
            "BRG" => Ok(ColorChannelMapping::BRG),
            "BRR" => Ok(ColorChannelMapping::BRR),
            "GBB" => Ok(ColorChannelMapping::GBB),
            "GBG" => Ok(ColorChannelMapping::GBG),
            "GBR" => Ok(ColorChannelMapping::GBR),
            "GGB" => Ok(ColorChannelMapping::GGB),
            "GGG" => Ok(ColorChannelMapping::GGG),
            "GGR" => Ok(ColorChannelMapping::GGR),
            "GRB" => Ok(ColorChannelMapping::GRB),
            "GRG" => Ok(ColorChannelMapping::GRG),
            "GRR" => Ok(ColorChannelMapping::GRR),
            "RBB" => Ok(ColorChannelMapping::RBB),
            "RBG" => Ok(ColorChannelMapping::RBG),
            "RBR" => Ok(ColorChannelMapping::RBR),
            "RGG" => Ok(ColorChannelMapping::RGG),
            "RGR" => Ok(ColorChannelMapping::RGR),
            "RRB" => Ok(ColorChannelMapping::RRB),
            "RRG" => Ok(ColorChannelMapping::RRG),
            "RRR" => Ok(ColorChannelMapping::RRR),
            "RGB" => Ok(ColorChannelMapping::RGB),
            _ => Err(String::from("Invalid r_g_b string, should be a string of length three with characters 'R', 'G' or 'B'"))
        }
    }
    ///Returns (x,y,z) where x,y,z ∈ {'R', 'G', 'B'}
    pub fn get_r_g_b_mapping(&self) -> (char,char,char) {
        let r_g_b = self.to_string().chars().collect::<Vec<char>>();
        (r_g_b[0],r_g_b[1],r_g_b[2])
    }
}

impl fmt::Display for ColorChannelMapping {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", *self)
    }
}

impl FromStr for ColorChannelMapping {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ColorChannelMapping::new(s)
    }
}

/// A 24-bit color: 8 bits for each of the red, green and blue channels
#[derive(Clone, Copy)]
pub struct TrueColor {
    /// The red channel, 0 to 255
    pub red: u8,
    /// The green channel, 0 to 255
    pub green: u8,
    /// The blue channel, 0 to 255
    pub blue: u8
}

impl TrueColor {
    /// Creates a color from its three channels
    pub fn new(red: u8, green: u8, blue: u8) -> TrueColor { 
        TrueColor { red, green, blue } 
    }

    /// Creates a 32-bit color. The encoding for each pixel is `0RGB`:
    /// The upper 8-bits are ignored, the next 8-bits are for the red channel, the next 8-bits
    /// afterwards for the green channel, and the lower 8-bits for the blue channel.
    pub fn to_32_bit(&self, mapping: &ColorChannelMapping) -> u32 {
        self.to_32_bit_resolved(mapping.get_r_g_b_mapping())
    }

    /// Same as `to_32_bit`, but takes an already resolved `ColorChannelMapping::get_r_g_b_mapping()`,
    /// so a conversion of many colors only has to resolve the mapping once.
    pub(crate) fn to_32_bit_resolved(&self, (r_map, g_map, b_map): (char, char, char)) -> u32 {
        let (r, g, b) = (u32::from(self.get_color(r_map)), u32::from(self.get_color(g_map)), u32::from(self.get_color(b_map)));
        (r << 16) | (g << 8) | b
    }

    /// The value of the channel `color`, which is `'R'`, `'G'` or `'B'`
    /// # Panics
    /// If `color` is another character
    pub fn get_color(&self, color: char) -> u8 {
        assert!(color == 'R' || color == 'G' || color == 'B', "Error: color should be equal to R, G, or B, color = {}",color);
        match color {
            'R' => self.red,
            'G' => self.green,
            'B' => self.blue,
            _ => 0,
        }
    }

    /// Creates a 24-bit truecolor struct from a normalized t ∈ [0, 1)\
    /// Maps a normalized value t to a continous 3D color space
    /// ![img](https://solarianprogrammer.com/images/2013/02/28/rgb_smooth.png)\
    /// Source: [Bernstein polynomial coloring](https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/)
    /// The polynomials are r = 9(1-t)t^3, g = 15(1-t)^2 t^2 and b = 8.5(1-t)^3 t, as in the source; none of them exceeds 1, so no channel is ever clipped at 255
    /// (earlier versions had the wrong powers in g and b, which saturated them over wide ranges of t and showed as flat bands with sharp edges).
    fn new_from_bernstein_polynomials_normalized(t: f64) -> TrueColor {
        let t = t.abs().min(0.999);
        let one_minus_t = 1.0-t;
        let red: f64 = 9.0 * one_minus_t * t.pow(3) * 255.0;
        let green: f64 = 15.0 * one_minus_t.pow(2) * t.pow(2) * 255.0;
        let blue: f64 = 8.5 * one_minus_t.pow(3) * t * 255.0;
        TrueColor { red: red as u8, green: green as u8, blue: blue as u8 }
    }

    ///A `coloring_function`
    /// Creates a 24-bit truecolor struct from a normalized t = `iterations`/`max_iterations` ∈ [0, 1)\
    /// Maps a normalized value t to a continous 3D color space
    /// ![img](https://solarianprogrammer.com/images/2013/02/28/rgb_smooth.png)\
    /// Source: [Bernstein polynomial coloring](https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/)
    pub fn new_from_bernstein_polynomials(iterations: u32, max_iterations: u32) -> TrueColor {
        let t: f64 = f64::from(iterations) / f64::from(max_iterations);
        TrueColor::new_from_bernstein_polynomials_normalized(t)
    }

    ///A `coloring_function`
    pub fn new_from_hsv_colors(iterations: u32, max_iterations: u32) -> TrueColor {
        let hue = 0.3 * f64::from(iterations);
        let saturation = 1.0;//0.8;
        let value: f64 = if iterations < max_iterations {1.0} else {0.0};
        let hue_degree = Deg(hue % 359.999);
        let hsv = Hsv::new(hue_degree,saturation,value);
        let rgb = Rgb::from_color(&hsv);
        let red = normalized_to_byte(rgb.red());
        let green = normalized_to_byte(rgb.green());
        let blue = normalized_to_byte(rgb.blue());
        TrueColor { red, green, blue }
    }

    ///Computes the average color of the given colors
    ///Can handle at most 2^24 colors
    pub fn average(colors: &Vec<TrueColor>) -> TrueColor {
        let mut red: u32 = 0;
        let mut green: u32 = 0;
        let mut blue: u32 = 0;
        for color in colors {
            red += u32::from(color.red);
            green += u32::from(color.green);
            blue += u32::from(color.blue);
        }
        let divisor = colors.len().max(1) as u32;
        red /= divisor;
        green /= divisor;
        blue /= divisor;
        let red = red as u8;
        let blue = blue as u8;
        let green = green as u8;
        TrueColor { red, green, blue }
    }

}

///Maps a number t ∈ [0.0, 1.0] to a byte b ∈ [0, 255]
fn normalized_to_byte(t: f64) -> u8 {
    let t = t.abs().min(1.0);
    let byte = (t * 255.0) as i16;
    byte.unsigned_abs() as u8 
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_MAPPINGS: [&str; 27] = [
        "BBB", "BBG", "BBR", "BGB", "BGG", "BGR", "BRB", "BRG", "BRR", "GBB", "GBG", "GBR", "GGB", "GGG",
        "GGR", "GRB", "GRG", "GRR", "RBB", "RBG", "RBR", "RGG", "RGR", "RRB", "RRG", "RRR", "RGB",
    ];

    #[test]
    fn bernstein_polynomials_match_the_source_and_never_clip() {
        let reference = |t: f64| (9.0 * (1.0 - t) * t.powi(3) * 255.0, 15.0 * (1.0 - t).powi(2) * t.powi(2) * 255.0, 8.5 * (1.0 - t).powi(3) * t * 255.0);
        let mut plateau = (0, 0, 0);
        for i in 0..=1000u32 {
            let t = f64::from(i) / 1000.0;
            let color = TrueColor::new_from_bernstein_polynomials(i, 1000);
            let (r, g, b) = reference(t.min(0.999));
            assert!(r < 255.0 && g < 255.0 && b < 255.0, "t = {t}: a channel would be clipped");
            assert_eq!((color.red, color.green, color.blue), (r as u8, g as u8, b as u8), "t = {t}");
            plateau.1 += usize::from(color.green == 255);
            plateau.2 += usize::from(color.blue == 255);
        }
        assert_eq!((plateau.1, plateau.2), (0, 0), "no channel is saturated");
        //The old powers saturated green over t in [0.31, 0.92]
        let mid = TrueColor::new_from_bernstein_polynomials(600, 1000);
        assert!(mid.green < 255 && mid.blue < 255);
    }

    #[test]
    fn to_32_bit_matches_reference_for_all_mappings() {
        let colors = [(0, 0, 0), (255, 255, 255), (1, 2, 3), (255, 0, 128), (17, 200, 99)];
        for name in ALL_MAPPINGS {
            let mapping = ColorChannelMapping::new(name).unwrap();
            for (r, g, b) in colors {
                let color = TrueColor::new(r, g, b);
                let channel = |c: char| u32::from(match c { 'R' => r, 'G' => g, _ => b });
                let n: Vec<char> = name.chars().collect();
                let expected = (channel(n[0]) << 16) | (channel(n[1]) << 8) | channel(n[2]);
                assert_eq!(color.to_32_bit(&mapping), expected, "{name} {r},{g},{b}");
                assert_eq!(color.to_32_bit_resolved(mapping.get_r_g_b_mapping()), expected);
            }
        }
    }

    #[test]
    fn colors_to_pixels_matches_per_pixel_conversion() {
        let colors: Vec<TrueColor> = (0..=255u8).map(|i| TrueColor::new(i, 255 - i, i / 2)).collect();
        for name in ALL_MAPPINGS {
            let mapping = ColorChannelMapping::new(name).unwrap();
            let expected: Vec<u32> = colors.iter().map(|c| c.to_32_bit(&mapping)).collect();
            assert_eq!(crate::pixel_buffer::PixelBuffer::colors_to_pixels(&colors, &mapping), expected, "{name}");
        }
    }

    #[test]
    fn color_channel_mappings_are_parsed_case_insensitively_and_invalid_ones_are_refused() {
        for name in ALL_MAPPINGS {
            assert_eq!(format!("{:?}", ColorChannelMapping::new(&name.to_lowercase()).unwrap()), name);
            assert_eq!(ColorChannelMapping::new(name).unwrap().to_string(), name);
            assert_eq!(name.parse::<ColorChannelMapping>().unwrap(), ColorChannelMapping::new(name).unwrap());
            assert_eq!(ColorChannelMapping::new(name).unwrap().get_r_g_b_mapping(), (name.chars().next().unwrap(), name.chars().nth(1).unwrap(), name.chars().nth(2).unwrap()));
        }
        for invalid in ["", "RG", "RGBA", "XYZ", "R G"] {
            assert!(ColorChannelMapping::new(invalid).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn the_average_of_colors_truncates_each_channel_and_an_empty_average_is_black() {
        let average = TrueColor::average(&vec![TrueColor::new(255, 0, 1), TrueColor::new(0, 1, 2), TrueColor::new(0, 1, 2)]);
        assert_eq!((average.red, average.green, average.blue), (85, 0, 1), "255 / 3 = 85, 2 / 3 = 0, 5 / 3 = 1");
        let single = TrueColor::average(&vec![TrueColor::new(9, 8, 7)]);
        assert_eq!((single.red, single.green, single.blue), (9, 8, 7));
        let empty = TrueColor::average(&Vec::new());
        assert_eq!((empty.red, empty.green, empty.blue), (0, 0, 0));
    }

    #[test]
    fn the_channels_of_a_color_are_found_by_letter() {
        let color = TrueColor::new(1, 2, 3);
        assert_eq!((color.get_color('R'), color.get_color('G'), color.get_color('B')), (1, 2, 3));
        assert_eq!(color.to_32_bit(&ColorChannelMapping::RGB), 0x0001_0203);
        assert_eq!(color.to_32_bit(&ColorChannelMapping::BGR), 0x0003_0201);
    }

    /// The colors of both palettes at some positions, to within one level per channel (the palettes use floating point functions that may round differently on another platform)
    #[test]
    fn the_palettes_give_the_colors_they_always_gave() {
        let near = |actual: TrueColor, expected: (u8, u8, u8)| {
            let (a, e) = ([actual.red, actual.green, actual.blue], [expected.0, expected.1, expected.2]);
            assert!(a.iter().zip(e).all(|(a, e)| a.abs_diff(e) <= 1), "{a:?} is not near {e:?}");
        };
        //Bernstein polynomials of t = iterations / max_iterations: black at t = 0 and almost black inside the set (t = 1 is clamped to 0.999, which leaves a little red)
        near(TrueColor::new_from_bernstein_polynomials(0, 1000), (0, 0, 0));
        near(TrueColor::new_from_bernstein_polynomials(100, 1000), (2, 30, 158));
        near(TrueColor::new_from_bernstein_polynomials(200, 1000), (14, 97, 221));
        near(TrueColor::new_from_bernstein_polynomials(400, 1000), (88, 220, 187));
        near(TrueColor::new_from_bernstein_polynomials(500, 1000), (143, 239, 135));
        near(TrueColor::new_from_bernstein_polynomials(750, 1000), (242, 134, 25));
        near(TrueColor::new_from_bernstein_polynomials(1000, 1000), (2, 0, 0));
        //HSV: the hue turns 0.3 degrees per iteration (a full turn after about 1200), full value outside the set and black inside it
        near(TrueColor::new_from_hsv_colors(0, 1000), (255, 0, 0));
        near(TrueColor::new_from_hsv_colors(100, 1000), (255, 127, 0));
        near(TrueColor::new_from_hsv_colors(200, 1000), (255, 255, 0));
        near(TrueColor::new_from_hsv_colors(400, 1000), (0, 255, 0));
        near(TrueColor::new_from_hsv_colors(500, 1000), (0, 255, 127));
        near(TrueColor::new_from_hsv_colors(750, 1000), (0, 63, 255));
        near(TrueColor::new_from_hsv_colors(999, 1000), (253, 0, 255));
        near(TrueColor::new_from_hsv_colors(1000, 1000), (0, 0, 0));
    }
}

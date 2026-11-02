//! The buffer of pixels of an image: the colors, the 32-bit pixels that the window draws, shifting for moves, and saving as a PNG file.

use std::{path::Path, fs::File, io::BufWriter};

use crate::{coloring::{TrueColor, ColorChannelMapping}, complex_plane::View, mandelbrot_set::MandelbrotSet};

use self::pixel_plane::PixelPlane;

/// The size of a pixel buffer
pub mod pixel_plane;

/// The pixels of an image, as colors and as the 32-bit pixels that the window draws, with the channel mapping that gets the pixels from the colors
#[derive(Clone)]
pub struct PixelBuffer {
    /// The size of the image
    pub pixel_plane: PixelPlane,
    /// The color of every pixel, row by row
    pub colors: Vec<TrueColor>,
    /// The pixels as `0RGB` 32-bit values: the colors with the channel mapping applied (`update_pixels`)
    pub pixels: Vec<u32>,
    /// How the channels of the colors are placed in the pixels
    pub color_channel_mapping: ColorChannelMapping,
}

impl PixelBuffer {
    /// Creates a black buffer of the given size
    pub fn new(pixel_plane: PixelPlane) -> PixelBuffer {
        // Create a buffer to store pixel data in the form of TrueColor's
        let black = TrueColor::new(0, 0, 0);
        let colors: Vec<TrueColor> = vec![black; pixel_plane.width * pixel_plane.height];
        let color_channel_mapping = ColorChannelMapping::RGB;
        let pixels: Vec<u32> = PixelBuffer::colors_to_pixels(&colors, &color_channel_mapping);
        PixelBuffer { pixel_plane, colors, pixels, color_channel_mapping}
    }

    /// Converts a buffer index to a screen coordinate
    pub fn index_to_point(&self, index: usize) -> (usize, usize) {
        (index % self.pixel_plane.width, index / self.pixel_plane.width)
    }

    /// Converts a screen coordinate to a buffer index
    pub fn point_to_index(&self, x: usize, y: usize) -> usize {
        y * self.pixel_plane.width + x
    }

    ///Converts a `TrueColor` vector to minifb compatible u32 pixel values
    pub fn colors_to_pixels(colors: &[TrueColor], color_channel_mapping: &ColorChannelMapping) -> Vec<u32> {
        let mapping = color_channel_mapping.sources(); //Get the channels once for the whole conversion, not once per pixel
        colors.iter().map(|x| x.to_32_bit_resolved(mapping)).collect()
    }

    ///Updates pixels from colors
    pub fn update_pixels(&mut self) {
        self.pixels = PixelBuffer::colors_to_pixels(&self.colors, &self.color_channel_mapping);
    }

    /// Translate the complex plane in the `buffer` `rows` up and `columns` to the right.
    /// This operation is significantly less expensive than rendering the image again, as it does not rerender anything in the complex plane, it simply
    /// get rids of `rows.abs()` rows and `columns.abs()` columns, and moves the image rows to the right and columns up.
    /// Note: The vacated rows and columns have to be rendered again, see `rendering::Render::only` and `rendering::ExposedRegion`.
    pub fn translate_buffer(&mut self, rows_up: i128, columns_right: i128) {
        //Iterate over the correct y's in the correct order
        let y_range : Vec<usize> = if rows_up > 0 {((rows_up as usize)..self.pixel_plane.height).rev().collect()} else {(0..((self.pixel_plane.height as i128 + rows_up) as usize)).collect()};
        //Iterate over the correct x's in the correct order
        let x_range : Vec<usize> = if columns_right > 0 {((columns_right as usize)..self.pixel_plane.width).rev().collect()} else {(0..((self.pixel_plane.width as i128 + columns_right) as usize)).collect()};

        for y in y_range {
            let other_y = (y as i128-rows_up) as usize;
            //println!("y: {y} and other_y: {other_y}");
            for x in &x_range {
                let other_x = (*x as i128 - columns_right) as usize;
                //println!("x: {} and other_x: {other_x}",*x);
                let index = self.point_to_index(*x, y);
                let other_index = self.point_to_index(other_x, other_y);
                self.colors[index] = self.colors[other_index];
                self.pixels[index] = self.pixels[other_index];
            }
        }
    }

    /// Saves the `PixelBuffer` as an RGB png image to `saved/{file_name_without_extension}.png`\
    /// Stores the current `ComplexPlane` View in the png's metadata under the view keyword\
    /// Stores the `supersampling_amount` in the metadata\
    /// Also stores author and application metadata
    /// # Errors
    /// Returns the error if the `saved` folder or the file cannot be created, or the image cannot be encoded or written
    pub fn save_as_png(&self, file_name_without_extension: &str, view: &View, m: &MandelbrotSet, supersampling_amount: u8, place: Option<&str>) -> std::io::Result<()> {
        let file_name_without_extension = file_name_without_extension.replace(':', "-").replace(' ', "_"); //Replace ':' with '-' for Windows file system. Replace ' ' with '_' because spaces are annoying in filenames.
        let file_name = format!("saved{}{}.png", std::path::MAIN_SEPARATOR_STR, file_name_without_extension);
        std::fs::create_dir_all("saved")?; //Create the saved folder if it does not exist
        self.write_png(Path::new(&file_name), view, m, supersampling_amount, place)
    }

    /// Writes the `PixelBuffer` as an RGB png image to `path`, with the metadata of `save_as_png`, and `place` (the view in the format of a point of interest
    /// file, see `image_metadata`) in an `iTXt` chunk
    /// # Errors
    /// Returns the error if the file cannot be created, or the image cannot be encoded or written
    fn write_png(&self, path: &Path, view: &View, m: &MandelbrotSet, supersampling_amount: u8, place: Option<&str>) -> std::io::Result<()> {
        let metadata = PngMetadata { view, m, supersampling_amount, color_channel_mapping: self.color_channel_mapping, place };
        let mut png = PngRows::create(path, self.pixel_plane.width, self.pixel_plane.height, &metadata)?;
        png.write_rows(&self.colors, self.color_channel_mapping)?;
        png.finish()
    }
}

/// What a saved PNG image says about itself: its view and settings, as texts (see `PngRows::create`)
pub struct PngMetadata<'a> {
    /// The view of the image
    pub view: &'a View,
    /// The set: its limit and colors
    pub m: &'a MandelbrotSet,
    /// The samples per pixel
    pub supersampling_amount: u8,
    /// How the channels of the colors are placed in the pixels
    pub color_channel_mapping: ColorChannelMapping,
    /// The view in the format of a point of interest file (see `image_metadata`), in an `iTXt` chunk
    pub place: Option<&'a str>,
}

/// A PNG image that is written row by row, so a large image never has to be in memory as a whole: the rows are filtered and compressed as they come
pub struct PngRows {
    writer: png::StreamWriter<'static, BufWriter<File>>,
}

impl PngRows {
    /// Creates the file at `path` for an RGB image of `width` x `height` pixels, with the texts of `metadata` before the pixels
    /// # Errors
    /// The file cannot be created or the header cannot be written
    pub fn create(path: &Path, width: usize, height: usize, metadata: &PngMetadata) -> std::io::Result<PngRows> {
        let side = |side: usize| u32::try_from(side).map_err(std::io::Error::other);
        let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), side(width)?, side(height)?);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let texts = [
            ("view", format!("{:?}", metadata.view)),
            ("mandelbrot_set", format!("{:?}", metadata.m)),
            ("supersampling_amount", format!("{}x", metadata.supersampling_amount)),
            ("application", String::from("Mandelbrot by Jort (https://github.com/jortrr/mandelbrot)")),
            ("author", String::from("jortrr (https://github.com/jortrr/)")),
            ("color_channel_mapping", format!("{:?}", metadata.color_channel_mapping)),
        ];
        for (keyword, text) in texts {
            encoder.add_text_chunk(String::from(keyword), text).map_err(std::io::Error::other)?;
        }
        if let Some(place) = metadata.place {
            encoder.add_itxt_chunk(String::from(crate::image_metadata::KEYWORD), place.to_string()).map_err(std::io::Error::other)?;
        }
        let writer = encoder.write_header().and_then(png::Writer::into_stream_writer).map_err(std::io::Error::other)?;
        Ok(PngRows { writer })
    }

    /// Writes the next whole rows of the image: `colors`, row by row, with their channels placed by `mapping`
    /// # Errors
    /// The rows cannot be written, or they are more than the image has
    pub fn write_rows(&mut self, colors: &[TrueColor], mapping: ColorChannelMapping) -> std::io::Result<()> {
        use std::io::Write;
        self.writer.write_all(&rgb_bytes(colors, mapping))
    }

    /// Finishes the image after its last row
    /// # Errors
    /// Fewer rows were written than the image has, or the end cannot be written
    pub fn finish(self) -> std::io::Result<()> {
        self.writer.finish().map_err(std::io::Error::other)
    }
}

/// The bytes of `colors` in a PNG of RGB pixels: red, green and blue of every pixel, with the channels placed by `mapping`
pub fn rgb_bytes(colors: &[TrueColor], mapping: ColorChannelMapping) -> Vec<u8> {
    let [r_source, g_source, b_source] = mapping.sources();
    let mut data: Vec<u8> = Vec::with_capacity(colors.len() * 3);
    for color in colors {
        data.extend([color.channel(r_source), color.channel(g_source), color.channel(b_source)]);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 5 x 4 buffer in which every pixel has its own color, `(x, y)` encoded in the channels
    fn numbered_buffer() -> PixelBuffer {
        let mut buffer = PixelBuffer::new(PixelPlane::new(5, 4));
        for y in 0..4 {
            for x in 0..5 {
                let index = buffer.point_to_index(x, y);
                buffer.colors[index] = TrueColor::new(x as u8, y as u8, 99);
            }
        }
        buffer.update_pixels();
        buffer
    }

    #[test]
    fn a_saved_image_opens_at_its_view_again() {
        //Issue #183: the view chunk of `write_png` is read back exactly by `image_metadata`
        let buffer = numbered_buffer();
        let (view, m) = (View::new(-0.749_810_000_000_000_1, -0.020_300_000_000_000_054, 6.4e-5), MandelbrotSet::new(321, 2.0));
        let point = crate::points_of_interest::PointOfInterest {
            name: String::from("Saved"),
            description: String::new(),
            location: crate::points_of_interest::Location { real: -0.749_810_000_000_000_1, imaginary: -0.020_300_000_000_000_054, scale: 6.4e-5, max_iterations: 321, rotation: 0.0, julia: None },
            appearance: Some(crate::points_of_interest::Appearance { iteration_divide: Some(0.113), color_offset: Some(278.0), ..Default::default() }),
            file_name: String::new(),
        };
        let path = std::env::temp_dir().join(format!("mandelbrot-view-png-{}.png", std::process::id()));
        buffer.write_png(&path, &view, &m, 4, Some(&crate::points_of_interest::to_json(&point))).unwrap();
        let read = crate::image_metadata::view_of_png(&path).unwrap();
        assert_eq!((read.location, read.appearance), (point.location, point.appearance));
        //Without the chunk (an image of an older version) the readable texts give the view and the limit
        buffer.write_png(&path, &view, &m, 4, None).unwrap();
        let old = crate::image_metadata::view_of_png(&path).unwrap();
        assert_eq!((old.location.real, old.location.imaginary, old.location.scale, old.location.max_iterations), (-0.749_810_000_000_000_1, -0.020_300_000_000_000_054, 6.4e-5, 321));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writing_a_png_reports_errors_instead_of_panicking() {
        let buffer = numbered_buffer();
        let (view, m) = (View::new(-0.75, 0.0, 1.0), MandelbrotSet::new(10, 2.0));
        let folder = std::env::temp_dir().join(format!("mandelbrot-png-test-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("ok.png");
        buffer.write_png(&path, &view, &m, 1, Some("{}")).expect("a png can be written to an existing folder");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        assert!(bytes.windows(4).any(|window| window == b"view"), "the view is stored in the metadata");
        let missing = folder.join("no-such-folder").join("x.png");
        assert!(buffer.write_png(&missing, &view, &m, 1, None).is_err(), "a folder that does not exist is an error, not a panic");
        std::fs::remove_dir_all(&folder).unwrap();
    }

    fn color_at(buffer: &PixelBuffer, x: usize, y: usize) -> (u8, u8) {
        let color = buffer.colors[buffer.point_to_index(x, y)];
        (color.red, color.green)
    }

    #[test]
    fn a_new_buffer_is_black_and_indexes_convert_both_ways() {
        let buffer = PixelBuffer::new(PixelPlane::new(5, 4));
        assert_eq!((buffer.colors.len(), buffer.pixels.len()), (20, 20));
        assert!(buffer.colors.iter().all(|color| (color.red, color.green, color.blue) == (0, 0, 0)) && buffer.pixels.iter().all(|pixel| *pixel == 0));
        for index in 0..20 {
            let (x, y) = buffer.index_to_point(index);
            assert_eq!(buffer.point_to_index(x, y), index);
            assert!(x < 5 && y < 4);
        }
        assert_eq!(buffer.index_to_point(7), (2, 1));
    }

    #[test]
    fn the_pixels_follow_the_colors_and_the_channel_mapping() {
        let mut buffer = numbered_buffer();
        assert_eq!(buffer.pixels[buffer.point_to_index(3, 2)], 0x0003_0263);
        buffer.color_channel_mapping = ColorChannelMapping::BGR;
        buffer.update_pixels();
        assert_eq!(buffer.pixels[buffer.point_to_index(3, 2)], 0x0063_0203);
    }

    #[test]
    fn shifting_the_buffer_moves_the_content_and_leaves_the_vacated_rows_and_columns_alone() {
        //Content moves down when the view moves up, right when the view moves right
        let mut buffer = numbered_buffer();
        buffer.translate_buffer(1, 0);
        assert_eq!(color_at(&buffer, 2, 1), (2, 0), "row 0 moved to row 1");
        assert_eq!(color_at(&buffer, 4, 3), (4, 2));
        assert_eq!(color_at(&buffer, 2, 0), (2, 0), "the vacated row 0 keeps its old colors, it has to be rendered again");

        let mut buffer = numbered_buffer();
        buffer.translate_buffer(0, -2);
        assert_eq!(color_at(&buffer, 0, 3), (2, 3), "column 2 moved to column 0");
        assert_eq!(color_at(&buffer, 2, 1), (4, 1));
        assert_eq!(color_at(&buffer, 4, 2), (4, 2), "the vacated last columns are not touched");

        let mut buffer = numbered_buffer();
        buffer.translate_buffer(-2, 3);
        assert_eq!(color_at(&buffer, 4, 0), (1, 2), "row 2 / column 1 moved up by 2 and right by 3");
        assert_eq!(color_at(&buffer, 3, 1), (0, 3));
        let pixels_follow = (0..20).all(|index| buffer.pixels[index] == (u32::from(buffer.colors[index].red) << 16) | (u32::from(buffer.colors[index].green) << 8) | u32::from(buffer.colors[index].blue));
        assert!(pixels_follow, "colors and pixels are shifted together");
    }
}

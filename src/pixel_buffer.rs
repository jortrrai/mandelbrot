use std::{path::Path, fs::File, io::BufWriter};

use crate::{coloring::{TrueColor, ColorChannelMapping}, complex_plane::View, mandelbrot_set::MandelbrotSet};

use self::pixel_plane::PixelPlane;

pub mod pixel_plane;

#[derive(Clone)]
pub struct PixelBuffer {
    pub pixel_plane: PixelPlane,
    pub colors: Vec<TrueColor>,
    pub pixels: Vec<u32>,
    pub color_channel_mapping: ColorChannelMapping,
}

impl PixelBuffer {
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
        let mapping = color_channel_mapping.get_r_g_b_mapping(); //Resolve the mapping once for the whole conversion, not once per pixel
        colors.iter().map(|x| x.to_32_bit_resolved(mapping)).collect()
    }

    ///Updates pixels from colors
    pub fn update_pixels(&mut self) {
        self.pixels = PixelBuffer::colors_to_pixels(&self.colors, &self.color_channel_mapping);
    }

    /// Translate the complex plane in the `buffer` `rows` up and `columns` to the right.
    /// This operation is significantly less expensive than the `render_box_render_complex_plane_into_buffer` function, as it does not rerender anything in the complex plane, it simply
    /// get rids of `rows.abs()` rows and `columns.abs()` columns, and moves the image rows to the right and columns up.
    /// Note: The removed rows and columns should be rerendered by the `render_box_render_complex_plane_into_buffer` function.
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

    ///Saves the `PixelBuffer` as an RGB png image to `saved/{file_name_without_extension}.png` </br>
    ///Stores the current `ComplexPlane` View in the png's metadata under the view keyword </br>
    ///Stores the `supersampling_amount` in the metadata </br>
    ///Also stores author and application metadata
    /// # Panics
    /// If the file `saved/{file_name_without_extension}.png` cannot be created
    pub fn save_as_png(&self, file_name_without_extension: &str, view: &View, m: &MandelbrotSet, supersampling_amount: u8) {
        let file_name_without_extension = file_name_without_extension.replace(':', "-").replace(' ', "_"); //Replace ':' with '-' for Windows file system. Replace ' ' with '_' because spaces are annoying in filenames.
        let file_name = format!("saved{}{}.png", std::path::MAIN_SEPARATOR_STR, file_name_without_extension);
        match std::fs::create_dir_all("saved") { //Create the saved folder if it does not exist 
            Ok(()) => (), //Currently not doing anything with the Result of trying to create the saved folder
            Err(err) => eprintln!("{}", err),
        }
        let path = Path::new(&file_name);
        let file = File::create(path).unwrap();
        let w = &mut BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, self.pixel_plane.width as u32, self.pixel_plane.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let view_text = format!("{:?}", view);
        encoder.add_text_chunk(String::from("view"), view_text).unwrap();
        let mandelbrot_set_text = format!("{:?}", m);
        encoder.add_text_chunk(String::from("mandelbrot_set"), mandelbrot_set_text).unwrap();
        let supersampling_amount_text = format!("{}x", supersampling_amount);
        encoder.add_text_chunk(String::from("supersampling_amount"), supersampling_amount_text).unwrap();
        encoder.add_text_chunk(String::from("application"), String::from("Mandelbrot by Jort (https://github.com/jortrr/mandelbrot)")).unwrap();
        encoder.add_text_chunk(String::from("author"), String::from("jortrr (https://github.com/jortrr/)")).unwrap();
        let color_channel_mapping_text = format!("{:?}", self.color_channel_mapping);
        encoder.add_text_chunk(String::from("color_channel_mapping"), color_channel_mapping_text).unwrap();
        let mut data: Vec<u8> = Vec::new();
        let (r_map, g_map, b_map) = self.color_channel_mapping.get_r_g_b_mapping();
        for color in &self.colors {
            data.push(color.get_color(r_map));
            data.push(color.get_color(g_map));
            data.push(color.get_color(b_map));
        }
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data).unwrap();
    }
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

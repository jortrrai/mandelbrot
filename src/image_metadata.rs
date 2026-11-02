//! The view of a saved image, read back from its metadata (issue #183), so an image can be opened in the window again.
//!
//! Saved images and the PNG frames of videos carry an `iTXt` chunk `mandelbrot` with the view in the format of a point of interest file (center, scale,
//! iteration limit and appearance, see `points_of_interest`), which reads back exactly. Images saved before that only have the readable `tEXt` chunks:
//! `view` (`z = -0.75 - 0i, scale = 1`), `mandelbrot_set` (`max_iterations = 1000, orbit_radius = 2`) and `color_channel_mapping`; their view and limit
//! are read from those.

use std::path::Path;

use crate::points_of_interest::{self, Appearance, Location, PointOfInterest};

/// The keyword of the chunk with the view of an image
pub const KEYWORD: &str = "mandelbrot";

/// The view of the PNG image at `path`, named after the file, or why there is none
///
/// # Errors
/// The file cannot be read or decoded, or has no view in its metadata
pub fn view_of_png(path: &Path) -> Result<PointOfInterest, String> {
    let file_name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let file = std::fs::File::open(path).map_err(|err| format!("{file_name}: {err}"))?;
    let reader = png::Decoder::new(std::io::BufReader::new(file)).read_info().map_err(|err| format!("{file_name}: {err}"))?;
    let info = reader.info();
    let texts: Vec<(String, String)> = info.utf8_text.iter().filter_map(|chunk| Some((chunk.keyword.clone(), chunk.get_text().ok()?)))
        .chain(info.uncompressed_latin1_text.iter().map(|chunk| (chunk.keyword.clone(), chunk.text.clone())))
        .chain(info.compressed_latin1_text.iter().filter_map(|chunk| Some((chunk.keyword.clone(), chunk.get_text().ok()?))))
        .collect();
    view_of_texts(&texts, &file_name)
}

/// The view in the text chunks `texts` (keyword, text) of an image named `file_name`, see the module
///
/// # Errors
/// There is no view in the texts, or it cannot be read
pub fn view_of_texts(texts: &[(String, String)], file_name: &str) -> Result<PointOfInterest, String> {
    let text = |keyword: &str| texts.iter().find(|(key, _)| key == keyword).map(|(_, text)| text.as_str());
    let name = Path::new(file_name).file_stem().map_or_else(|| file_name.to_string(), |stem| stem.to_string_lossy().into_owned());
    if let Some(json) = text(KEYWORD) {
        let (point, _) = points_of_interest::parse(json, file_name)?;
        return Ok(point);
    }
    let view = text("view").ok_or_else(|| format!("{file_name} has no view in its metadata (it was not saved by this program)"))?;
    let (real, imaginary, scale) = parse_view(view).ok_or_else(|| format!("{file_name}: cannot read its view {view:?}"))?;
    let max_iterations = text("mandelbrot_set").and_then(|set| set.split(',').find_map(|part| part.trim().strip_prefix("max_iterations = ")?.trim().parse().ok())).unwrap_or(1000);
    let appearance = text("color_channel_mapping").and_then(|mapping| mapping.trim().parse().ok()).map(|mapping| Appearance { color_channels: Some(mapping), ..Appearance::default() });
    Ok(PointOfInterest { name, description: String::new(), location: Location { real, imaginary, scale, max_iterations, rotation: 0.0, julia: None }, appearance, file_name: file_name.to_string() })
}

/// The center and scale of the text of a view: `z = -0.75 - 0i, scale = 1` (see `complex_plane::View`'s `Debug`)
fn parse_view(view: &str) -> Option<(f64, f64, f64)> {
    let (z, scale) = view.trim().strip_prefix("z = ")?.split_once(", scale = ")?;
    let (real, sign, imaginary) = if let Some((real, imaginary)) = z.split_once(" + ") { (real, 1.0, imaginary) } else { let (real, imaginary) = z.split_once(" - ")?; (real, -1.0, imaginary) };
    let (real, imaginary, scale) = (real.trim().parse::<f64>().ok()?, sign * imaginary.trim().strip_suffix('i')?.parse::<f64>().ok()?, scale.trim().parse::<f64>().ok()?);
    (real.is_finite() && imaginary.is_finite() && scale.is_finite() && scale > 0.0).then_some((real, imaginary, scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_of_an_old_image_is_read_from_its_readable_texts() {
        let texts = [
            (String::from("view"), String::from("z = -0.7498100000000001 - 0.020300000000000054i, scale = 0.000064")),
            (String::from("mandelbrot_set"), String::from("max_iterations = 321, orbit_radius = 2")),
            (String::from("color_channel_mapping"), String::from("BGR")),
        ];
        let point = view_of_texts(&texts, "2023-06-26_12-48-59.png").unwrap();
        assert_eq!((point.location.real, point.location.imaginary, point.location.scale, point.location.max_iterations), (-0.749_810_000_000_000_1, -0.020_300_000_000_000_054, 0.000_064, 321));
        assert_eq!(point.name, "2023-06-26_12-48-59");
        assert_eq!(point.appearance.and_then(|appearance| appearance.color_channels).map(|mapping| mapping.to_string()), Some(String::from("BGR")));
        assert_eq!(parse_view("z = -0.75 - 0i, scale = 1"), Some((-0.75, -0.0, 1.0)), "the minus sign of an imaginary part of 0");
        assert_eq!(parse_view("z = 1e-5 + 2.5e-7i, scale = 3e-12"), Some((1e-5, 2.5e-7, 3e-12)));
        assert!(view_of_texts(&[], "photo.png").unwrap_err().contains("no view"));
    }

    #[test]
    fn the_view_chunk_of_a_new_image_reads_back_exactly() {
        let point = PointOfInterest {
            name: String::from("x"),
            description: String::new(),
            location: Location { real: -0.749_810_000_000_000_1, imaginary: 0.1, scale: 6.400_000_000_000_002e-5, max_iterations: 5000, rotation: 0.0, julia: None },
            appearance: Some(Appearance { iteration_divide: Some(0.113), color_offset: Some(278.0), ..Appearance::default() }),
            file_name: String::new(),
        };
        let texts = [(String::from(KEYWORD), points_of_interest::to_json(&point))];
        let read = view_of_texts(&texts, "image.png").unwrap();
        assert_eq!((read.location, read.appearance), (point.location, point.appearance));
    }
}

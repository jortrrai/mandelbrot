//! Points of interest: named locations in the Mandelbrot set, loaded from the JSON files of the `points_of_interest` folder.
//!
//! A file has a **location** (where to look: the center, the scale, the maximum iterations) and an optional **appearance** (how it looks: the coloring and the color channel mapping).
//! Going to a point of interest always applies the location; the appearance is only applied when the user asks to restore it.
//! The supersampling amount is deliberately not stored: it depends on the machine and on how patient the user is, not on the place.
//!
//! ```json
//! {
//!   "format_version": 1,
//!   "name": "Frost on the Shoreline",
//!   "description": "Optional text shown as a tooltip.",
//!   "location": {
//!     "center": { "real": -0.6604166666666667, "imaginary": 0.4437500000000001 },
//!     "scale": 0.1,
//!     "max_iterations": 1000
//!   },
//!   "appearance": { "coloring": "bernstein", "color_channels": "RGB" }
//! }
//! ```
//! * `scale` is the scale of the renderer (`ComplexPlane::get_scale`): the view is `2.5 * scale` wide on the real axis, so a smaller scale is a deeper zoom, and the magnification is `1 / scale`.
//!   Numbers are `f64` and are read exactly: the shortest decimal text of an `f64` reads back to the same number (`serde_json` needs its `float_roundtrip` feature for that; without it
//!   `-0.46395833333333325` was read one bit off, which a test of the shipped files caught).
//! * `max_iterations` is a whole number from 1 to 1,000,000, the range of the control in the window.
//! * `coloring` is `"hsv"` or `"bernstein"`; `color_channels` is a mapping such as `"RGB"` or `"BGR"`. Both are optional.
//! * The points are listed in alphabetical order of their **file names** (ignoring case), not of their display names: start a file name with a digit to choose its place in the list (with more than ten files, use leading zeros so that `10-` does not sort before `2-`).
//!   The bundled files are named `0-frost-on-the-shoreline.json` to `9-sunlit-reef.json`: the number of the view they replaced, counting from 0 (`VIEW_1` is 0, `VIEW_9` is 8 and `VIEW_0` is 9), a `-`, and the slugified name.
//!   Fields that are not known are ignored, so newer files still load.
//!
//! A file that cannot be used is skipped with a message that names the file; the others still load. See `load_folder`.
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::coloring::ColorChannelMapping;

/// The name of the folder with the point of interest files
pub const FOLDER_NAME: &str = "points_of_interest";
/// The only `format_version` that is understood
pub const FORMAT_VERSION: u32 = 1;
/// The allowed maximum iterations, the range of the control in the window
pub const MAX_ITERATIONS_RANGE: RangeInclusive<u32> = 1..=1_000_000;

/// The coloring of an appearance
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coloring {
    /// The HSV palette
    Hsv,
    /// The Bernstein polynomial palette
    Bernstein,
}

impl Coloring {
    /// The coloring with the key `key` (case-insensitive), if there is one
    pub fn from_key(key: &str) -> Option<Coloring> {
        match key.trim().to_lowercase().as_str() {
            "hsv" => Some(Coloring::Hsv),
            "bernstein" => Some(Coloring::Bernstein),
            _ => None,
        }
    }

    /// The key of the coloring as it is written in a file
    pub fn key(self) -> &'static str {
        match self {
            Coloring::Hsv => "hsv",
            Coloring::Bernstein => "bernstein",
        }
    }
}

/// Where to look
#[derive(Clone, Debug, PartialEq)]
pub struct Location {
    /// The real part of the center of the view
    pub real: f64,
    /// The imaginary part of the center of the view
    pub imaginary: f64,
    /// The scale of the renderer, see the module documentation
    pub scale: f64,
    /// The most iterations of the escape-time iteration
    pub max_iterations: u32,
}

/// How it looks: the parts that are not given are not changed when the appearance is restored
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Appearance {
    /// The coloring, if the file gives one
    pub coloring: Option<Coloring>,
    /// The color channel mapping, if the file gives one
    pub color_channels: Option<ColorChannelMapping>,
}

/// A named place: where to look, and optionally how it should look
#[derive(Clone, Debug, PartialEq)]
pub struct PointOfInterest {
    /// The display name
    pub name: String,
    /// A longer text for a tooltip, possibly empty
    pub description: String,
    /// Where to look
    pub location: Location,
    /// How it looks; `None` if the file has no appearance
    pub appearance: Option<Appearance>,
    /// The name of the file it was loaded from
    pub file_name: String,
}

#[derive(Deserialize)]
struct RawCenter {
    real: f64,
    imaginary: f64,
}

#[derive(Deserialize)]
struct RawLocation {
    center: RawCenter,
    scale: f64,
    max_iterations: u32,
}

#[derive(Deserialize)]
struct RawAppearance {
    #[serde(default)]
    coloring: Option<String>,
    #[serde(default)]
    color_channels: Option<String>,
}

#[derive(Deserialize)]
struct RawPointOfInterest {
    format_version: u32,
    name: String,
    #[serde(default)]
    description: String,
    location: RawLocation,
    #[serde(default)]
    appearance: Option<RawAppearance>,
}

/// Parses and checks the text of a point of interest file. `file_name` is only used in the messages.
///
/// Returns the point and warnings (an appearance that could not be used is left out with a warning), or the reason that the file cannot be used.
///
/// # Errors
/// The text is not valid JSON or lacks a field, the format version is not supported, the name is empty, the center is not finite, the scale is not positive or the maximum iterations are outside the allowed range.
pub fn parse(text: &str, file_name: &str) -> Result<(PointOfInterest, Vec<String>), String> {
    let raw: RawPointOfInterest = serde_json::from_str(text).map_err(|error| format!("{file_name}: {error}"))?;
    if raw.format_version != FORMAT_VERSION {
        return Err(format!("{file_name}: format_version {} is not supported (expected {FORMAT_VERSION})", raw.format_version));
    }
    let name = raw.name.trim().to_string();
    if name.is_empty() {
        return Err(format!("{file_name}: the name is empty"));
    }
    let location = raw.location;
    if !location.center.real.is_finite() || !location.center.imaginary.is_finite() {
        return Err(format!("{file_name}: the center should be finite numbers"));
    }
    if !(location.scale.is_finite() && location.scale > 0.0) {
        return Err(format!("{file_name}: the scale should be a positive number"));
    }
    if !MAX_ITERATIONS_RANGE.contains(&location.max_iterations) {
        return Err(format!("{file_name}: max_iterations should be from {} to {}, not {}", MAX_ITERATIONS_RANGE.start(), MAX_ITERATIONS_RANGE.end(), location.max_iterations));
    }
    let mut warnings = Vec::new();
    let appearance = raw.appearance.map(|raw| {
        let mut appearance = Appearance::default();
        if let Some(key) = raw.coloring {
            match Coloring::from_key(&key) {
                Some(coloring) => appearance.coloring = Some(coloring),
                None => warnings.push(format!("{file_name}: the coloring {key:?} is unknown (use \"hsv\" or \"bernstein\"), it is ignored")),
            }
        }
        if let Some(channels) = raw.color_channels {
            match ColorChannelMapping::new(channels.trim()) {
                Ok(mapping) => appearance.color_channels = Some(mapping),
                Err(_) => warnings.push(format!("{file_name}: the color channels {channels:?} are not a mapping like \"RGB\", they are ignored")),
            }
        }
        appearance
    });
    let location = Location { real: location.center.real, imaginary: location.center.imaginary, scale: location.scale, max_iterations: location.max_iterations };
    Ok((PointOfInterest { name, description: raw.description, location, appearance, file_name: file_name.to_string() }, warnings))
}

/// What loading a folder gave
#[derive(Debug, Default)]
pub struct LoadReport {
    /// The usable points, sorted by file name (ignoring case)
    pub points: Vec<PointOfInterest>,
    /// One message for every problem: files that were skipped, appearances that were ignored, a folder that cannot be read
    pub problems: Vec<String>,
}

/// Loads every `.json` file of `folder`. Other files are ignored; a file that cannot be used is skipped and reported in `problems`, the rest loads.
///
/// A folder that does not exist or cannot be read gives no points and a problem.
pub fn load_folder(folder: &Path) -> LoadReport {
    let mut report = LoadReport::default();
    let entries = match std::fs::read_dir(folder) {
        Ok(entries) => entries,
        Err(error) => {
            report.problems.push(format!("Cannot read the points of interest folder {}: {error}", folder.display()));
            return report;
        }
    };
    let mut files: Vec<PathBuf> = entries.filter_map(Result::ok).map(|entry| entry.path()).filter(|path| path.is_file() && path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("json"))).collect();
    files.sort();
    for path in files {
        let file_name = path.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        match std::fs::read_to_string(&path) {
            Ok(text) => match parse(&text, &file_name) {
                Ok((point, warnings)) => {
                    report.points.push(point);
                    report.problems.extend(warnings);
                }
                Err(problem) => report.problems.push(problem),
            },
            Err(error) => report.problems.push(format!("{file_name}: cannot be read: {error}")),
        }
    }
    report.points.sort_by(|a, b| a.file_name.to_lowercase().cmp(&b.file_name.to_lowercase()).then_with(|| a.file_name.cmp(&b.file_name)));
    report
}

/// The folder to load: `points_of_interest` in the current directory, or else the one next to the sources this program was built from (so `cargo run` works from any directory).
///
/// If neither exists, the one in the current directory, which `load_folder` then reports.
pub fn find_folder() -> PathBuf {
    let in_current_directory = PathBuf::from(FOLDER_NAME);
    let next_to_sources = Path::new(env!("CARGO_MANIFEST_DIR")).join(FOLDER_NAME);
    if in_current_directory.is_dir() {
        in_current_directory
    } else if next_to_sources.is_dir() {
        next_to_sources
    } else {
        in_current_directory
    }
}

/// The file name (without extension and without its number) for a name: lowercase letters and digits, other characters become a single `-`, apostrophes disappear.
/// "Comet's Crown" gives "comets-crown".
pub fn slugify(name: &str) -> String {
    let mut slug = String::new();
    let mut pending_dash = false;
    for character in name.chars() {
        if character.is_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.extend(character.to_lowercase());
        } else if character != '\'' && character != '’' {
            pending_dash = true;
        }
    }
    slug
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{
        "format_version": 1, "name": " Test Place ", "description": "d",
        "location": { "center": { "real": -0.75, "imaginary": 0.1 }, "scale": 0.5, "max_iterations": 200 },
        "appearance": { "coloring": "HSV", "color_channels": "bgr" },
        "something_new": [1, 2, 3]
    }"#;

    fn temp_folder(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!("mandelbrot-poi-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        folder
    }

    #[test]
    fn a_valid_file_is_parsed_and_unknown_fields_are_ignored() {
        let (point, warnings) = parse(VALID, "test.json").unwrap();
        assert!(warnings.is_empty());
        assert_eq!(point.name, "Test Place");
        assert_eq!(point.location, Location { real: -0.75, imaginary: 0.1, scale: 0.5, max_iterations: 200 });
        let appearance = point.appearance.unwrap();
        assert_eq!(appearance.coloring, Some(Coloring::Hsv));
        assert_eq!(format!("{:?}", appearance.color_channels.unwrap()), "BGR");
        assert_eq!(point.description, "d");
    }

    #[test]
    fn the_appearance_is_optional_and_parts_of_it_are_too() {
        let text = VALID.replace(r#""appearance": { "coloring": "HSV", "color_channels": "bgr" },"#, "");
        assert!(parse(&text, "a.json").unwrap().0.appearance.is_none());
        let text = VALID.replace(r#""color_channels": "bgr""#, r#""x": 1"#);
        let appearance = parse(&text, "a.json").unwrap().0.appearance.unwrap();
        assert_eq!((appearance.coloring, appearance.color_channels.is_none()), (Some(Coloring::Hsv), true));
    }

    #[test]
    fn numbers_keep_every_digit() {
        let text = VALID.replace("-0.75", "-0.7498100000000001").replace("0.5,", "0.00006400000000000002,");
        let (point, _) = parse(&text, "a.json").unwrap();
        assert_eq!(point.location.real.to_bits(), (-0.7498100000000001_f64).to_bits());
        assert_eq!(point.location.scale.to_bits(), 0.00006400000000000002_f64.to_bits());
    }

    #[test]
    fn malformed_files_are_rejected_with_the_file_name() {
        let cases = [
            ("not json at all", "broken.json"),
            ("{}", "empty.json"),
            (&VALID.replace("\"scale\": 0.5", "\"scale\": 0"), "zero-scale.json"),
            (&VALID.replace("\"scale\": 0.5", "\"scale\": -1"), "negative-scale.json"),
            (&VALID.replace("\"scale\": 0.5", "\"scale\": 1e999"), "infinite-scale.json"),
            (&VALID.replace("\"max_iterations\": 200", "\"max_iterations\": 0"), "zero-iterations.json"),
            (&VALID.replace("\"max_iterations\": 200", "\"max_iterations\": 5000000"), "too-many-iterations.json"),
            (&VALID.replace("\"max_iterations\": 200", "\"max_iterations\": 2.5"), "fractional-iterations.json"),
            (&VALID.replace("\"real\": -0.75,", ""), "missing-real.json"),
            (&VALID.replace("\"format_version\": 1", "\"format_version\": 2"), "future.json"),
            (&VALID.replace(" Test Place ", "  "), "no-name.json"),
        ];
        for (text, file) in cases {
            let problem = parse(text, file).expect_err(file);
            assert!(problem.starts_with(file), "{problem}");
        }
    }

    #[test]
    fn an_unusable_appearance_is_ignored_with_a_warning_and_the_location_stays() {
        let text = VALID.replace("\"HSV\"", "\"rainbow\"").replace("\"bgr\"", "\"xyz\"");
        let (point, warnings) = parse(&text, "a.json").unwrap();
        assert_eq!(warnings.len(), 2);
        assert_eq!(point.appearance, Some(Appearance::default()));
        assert_eq!(point.location.max_iterations, 200);
    }

    #[test]
    fn a_folder_loads_the_good_files_sorts_them_and_reports_the_bad_ones() {
        let folder = temp_folder("mixed");
        std::fs::write(folder.join("2-ant.json"), VALID.replace("Test Place", "Ant")).unwrap();
        std::fs::write(folder.join("1-zebra.json"), VALID.replace("Test Place", "Zebra")).unwrap();
        std::fs::write(folder.join("0-Yak.JSON"), VALID.replace("Test Place", "Yak")).unwrap();
        std::fs::write(folder.join("b-cat.json"), VALID.replace("Test Place", "Cat")).unwrap();
        std::fs::write(folder.join("A-bee.json"), VALID.replace("Test Place", "Bee")).unwrap();
        std::fs::write(folder.join("broken.json"), "{ nope").unwrap();
        std::fs::write(folder.join("notes.txt"), "not a point").unwrap();
        std::fs::create_dir(folder.join("sub.json")).unwrap();
        let report = load_folder(&folder);
        let names: Vec<&str> = report.points.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["Yak", "Zebra", "Ant", "Bee", "Cat"], "alphabetical order of the file names (0-, 1-, 2-, A-bee, b-cat, ignoring case), not of the display names");
        assert_eq!(report.problems.len(), 1);
        assert!(report.problems[0].starts_with("broken.json"));
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn a_missing_folder_gives_a_problem_and_no_points() {
        let report = load_folder(&std::env::temp_dir().join("mandelbrot-poi-test-does-not-exist"));
        assert!(report.points.is_empty() && report.problems.len() == 1, "{report:?}");
    }

    #[test]
    fn slugify_makes_file_names() {
        assert_eq!(slugify("Frost on the Shoreline"), "frost-on-the-shoreline");
        assert_eq!(slugify("Comet's Crown"), "comets-crown");
        assert_eq!(slugify("  The  Narrow--Pass! "), "the-narrow-pass");
    }

    /// The numbers of the ten views that were hard-coded in `lib.rs` (`VIEW_1` to `VIEW_9`, then `VIEW_0`), by the name of the file that holds them now, in their original order; the file names count from 0
    const FORMER_VIEWS: [(&str, &str, f64, f64, f64); 10] = [
        ("0-frost-on-the-shoreline", "Frost on the Shoreline", -0.6604166666666667, 0.4437500000000001, 0.1),
        ("1-frozen-constellation", "Frozen Constellation", -1.0591666666666668, 0.2629166666666668, 0.01),
        ("2-harbor-of-moons", "Harbor of Moons", -0.4624999999999999, 0.55, 0.1),
        ("3-the-narrow-pass", "The Narrow Pass", -0.46395833333333325, 0.5531250000000001, 0.03),
        ("4-coral-atoll", "Coral Atoll", -0.4375218333333333, 0.5632133750000003, 0.00002000000000000002),
        ("5-seahorse-embers", "Seahorse Embers", -0.7498100000000001, -0.020300000000000054, 0.00006400000000000002),
        ("6-lone-comet", "Lone Comet", -1.7862712000000047, 0.000052399999999991516, 0.00001677721600000001),
        ("7-comets-crown", "Comet's Crown", -1.7862581627050718, 0.00005198056959995248, 0.000006039797760000003),
        ("8-the-whirlpool", "The Whirlpool", -0.4687339999999999, 0.5425518958333333, 0.000010000000000000003),
        ("9-sunlit-reef", "Sunlit Reef", -0.437520465811966, 0.5632133750000006, 0.000004000000000000004),
    ];

    #[test]
    fn the_shipped_folder_has_the_ten_former_views_in_their_original_order_with_exactly_their_numbers() {
        let report = load_folder(&find_folder());
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert_eq!(report.points.len(), 10);
        //The file names sort as View 1 to View 10 (VIEW_0 last), numbered 0 to 9, and that is the order of the list
        for (index, (point, (file, name, real, imaginary, scale))) in report.points.iter().zip(FORMER_VIEWS).enumerate() {
            assert_eq!(point.file_name, format!("{file}.json"), "position {}", index + 1);
            assert!(file.starts_with(&format!("{index}-")), "the number of the file is its place, counting from 0");
            assert_eq!(point.name, name);
            assert_eq!(format!("{index}-{}", slugify(&point.name)), file, "the file is named after its number and the display name");
            assert!(!point.description.is_empty());
            assert_eq!((point.location.real.to_bits(), point.location.imaginary.to_bits(), point.location.scale.to_bits()), (real.to_bits(), imaginary.to_bits(), scale.to_bits()), "{file}");
            assert!(MAX_ITERATIONS_RANGE.contains(&point.location.max_iterations));
            let appearance = point.appearance.as_ref().expect("the default appearance is written out");
            assert_eq!((appearance.coloring, format!("{:?}", appearance.color_channels.unwrap())), (Some(Coloring::Bernstein), String::from("RGB")), "the default appearance of the application");
        }
        assert_eq!(report.points[9].file_name, "9-sunlit-reef.json", "VIEW_0, the tenth view, is number 9");
        let mut sorted: Vec<String> = report.points.iter().map(|p| p.file_name.to_lowercase()).collect();
        sorted.sort();
        assert_eq!(sorted, report.points.iter().map(|p| p.file_name.to_lowercase()).collect::<Vec<_>>(), "listed in alphabetical order of the file names");
    }
}

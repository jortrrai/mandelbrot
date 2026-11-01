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
//!   "appearance": { "coloring": "bernstein", "color_channels": "RGB", "iteration_divide": 4, "color_offset": 120 }
//! }
//! ```
//! * `scale` is the scale of the renderer (`ComplexPlane::get_scale`): the view is `2.5 * scale` wide on the real axis, so a smaller scale is a deeper zoom, and the magnification is `1 / scale`.
//!   Numbers are `f64` and are read exactly: the shortest decimal text of an `f64` reads back to the same number (`serde_json` needs its `float_roundtrip` feature for that; without it
//!   `-0.46395833333333325` was read one bit off, which a test of the shipped files caught).
//! * `max_iterations` is a whole number from 1 to 1,000,000, the range of the control in the window.
//! * `coloring` is `"hsv"` or `"bernstein"`; `color_channels` is a mapping such as `"RGB"` or `"BGR"`. `iteration_divide` (0.1 to 1000; a positive number outside that range is moved to the nearest end) and `color_offset` (0 to 500)
//!   are the color cycle of the place (`mandelbrot_set::ColorCycle`), and `stretch_palette` (`true` or `false`) whether the palette is stretched over the limit instead.
//!   All are optional.
//! * The points are listed by **zoom**, the shallowest first (the largest `scale` at the top), so the deeper places are further down. Places with the same zoom are listed in
//!   alphabetical order of their file names (ignoring case), not of their display names.
//!   The bundled files are named `00-reset-view.json` (the reset view, the shallowest, so first in the list), then `01-frost-on-the-shoreline.json` to `10-sunlit-reef.json`: the number of
//!   the view they replaced, counting from 1 with two digits (`VIEW_1` is 01, `VIEW_9` is 09 and `VIEW_0` is 10), a `-`, and the slugified name.
//!   Fields that are not known are ignored, so newer files still load.
//!
//! A file that cannot be used is skipped with a message that names the file; the others still load. See `load_folder`.
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::coloring::ColorChannelMapping;
use crate::mandelbrot_set::{ColorCycle, MAX_DIVIDE, MIN_DIVIDE, PALETTE_LENGTH};

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
    /// How far the view is turned counterclockwise around its center, in degrees (issue #217; a file without it is not turned)
    pub rotation: f64,
}

/// How it looks: the parts that are not given are not changed when the appearance is restored
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Appearance {
    /// The coloring, if the file gives one
    pub coloring: Option<Coloring>,
    /// The color channel mapping, if the file gives one
    pub color_channels: Option<ColorChannelMapping>,
    /// The iteration divide of the color cycle, if the file gives one, see `ColorCycle`
    pub iteration_divide: Option<f64>,
    /// The color offset of the color cycle, if the file gives one
    pub color_offset: Option<f64>,
    /// Whether the palette is stretched over the limit, if the file says so
    pub stretch_palette: Option<bool>,
}

impl Appearance {
    /// `palette` with the parts of the color cycle that this appearance gives
    pub fn palette_over(&self, palette: ColorCycle) -> ColorCycle {
        ColorCycle { divide: self.iteration_divide.unwrap_or(palette.divide), offset: self.color_offset.unwrap_or(palette.offset), stretch: self.stretch_palette.unwrap_or(palette.stretch) }
    }
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
    #[serde(default)]
    rotation: f64,
}

#[derive(Deserialize)]
struct RawAppearance {
    #[serde(default)]
    coloring: Option<String>,
    #[serde(default)]
    color_channels: Option<String>,
    #[serde(default)]
    iteration_divide: Option<f64>,
    #[serde(default)]
    color_offset: Option<f64>,
    #[serde(default)]
    stretch_palette: Option<bool>,
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
        match raw.iteration_divide {
            Some(divide) if (MIN_DIVIDE..=MAX_DIVIDE).contains(&divide) => appearance.iteration_divide = Some(divide),
            //(Outside the range of the slider: the nearest end, as the slider would show it)
            Some(divide) if divide > 0.0 => appearance.iteration_divide = Some(divide.clamp(MIN_DIVIDE, MAX_DIVIDE)),
            Some(divide) => warnings.push(format!("{file_name}: the iteration divide {divide} should be more than 0, it is ignored")),
            None => {}
        }
        match raw.color_offset {
            Some(offset) if offset.is_finite() => appearance.color_offset = Some(offset.rem_euclid(PALETTE_LENGTH)),
            Some(offset) => warnings.push(format!("{file_name}: the color offset {offset} should be a number, it is ignored")),
            None => {}
        }
        appearance.stretch_palette = raw.stretch_palette;
        appearance
    });
    let location = Location { real: location.center.real, imaginary: location.center.imaginary, scale: location.scale, max_iterations: location.max_iterations, rotation: if location.rotation.is_finite() { location.rotation } else { 0.0 } };
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

/// The places that come with the program, built into it: used when there is no folder of places to read (the Android app, or the program started elsewhere,
/// issue #219)
pub const SHIPPED: [(&str, &str); 11] = [
    ("00-reset-view.json", include_str!("../points_of_interest/00-reset-view.json")),
    ("01-frost-on-the-shoreline.json", include_str!("../points_of_interest/01-frost-on-the-shoreline.json")),
    ("02-frozen-constellation.json", include_str!("../points_of_interest/02-frozen-constellation.json")),
    ("03-harbor-of-moons.json", include_str!("../points_of_interest/03-harbor-of-moons.json")),
    ("04-the-narrow-pass.json", include_str!("../points_of_interest/04-the-narrow-pass.json")),
    ("05-coral-atoll.json", include_str!("../points_of_interest/05-coral-atoll.json")),
    ("06-seahorse-embers.json", include_str!("../points_of_interest/06-seahorse-embers.json")),
    ("07-lone-comet.json", include_str!("../points_of_interest/07-lone-comet.json")),
    ("08-comets-crown.json", include_str!("../points_of_interest/08-comets-crown.json")),
    ("09-the-whirlpool.json", include_str!("../points_of_interest/09-the-whirlpool.json")),
    ("10-sunlit-reef.json", include_str!("../points_of_interest/10-sunlit-reef.json")),
];

/// The places of `SHIPPED`, shallowest first
pub fn shipped() -> Vec<PointOfInterest> {
    let mut points: Vec<PointOfInterest> = SHIPPED.iter().filter_map(|(name, text)| parse(text, name).ok().map(|(point, _)| point)).collect();
    points.sort_by(|a, b| b.location.scale.total_cmp(&a.location.scale));
    points
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
    report.points.sort_by(|a, b| {
        //Zoom ascending is scale descending: the shallowest place first
        b.location.scale.total_cmp(&a.location.scale).then_with(|| a.file_name.to_lowercase().cmp(&b.file_name.to_lowercase())).then_with(|| a.file_name.cmp(&b.file_name))
    });
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

/// The text of a point of interest file for `point`, in the format that `parse` reads (the description and the appearance are left out when there are none)
///
/// # Panics
/// Never in practice: a point of interest is plain text and numbers, which always serialize
pub fn to_json(point: &PointOfInterest) -> String {
    let location = &point.location;
    let mut file = serde_json::json!({
        "format_version": 1,
        "name": point.name,
        "location": {
            "center": { "real": location.real, "imaginary": location.imaginary },
            "scale": location.scale,
            "max_iterations": location.max_iterations,
        },
    });
    //(Only a turned view has a rotation: the files of the others are as before)
    if location.rotation != 0.0 {
        file["location"]["rotation"] = serde_json::Value::from(location.rotation);
    }
    if !point.description.is_empty() {
        file["description"] = serde_json::Value::from(point.description.clone());
    }
    if let Some(appearance) = &point.appearance {
        let mut fields = serde_json::Map::new();
        if let Some(coloring) = appearance.coloring {
            fields.insert(String::from("coloring"), serde_json::Value::from(coloring.key()));
        }
        if let Some(mapping) = appearance.color_channels {
            fields.insert(String::from("color_channels"), serde_json::Value::from(format!("{mapping:?}")));
        }
        for (key, value) in [("iteration_divide", appearance.iteration_divide), ("color_offset", appearance.color_offset)] {
            if let Some(value) = value {
                fields.insert(String::from(key), serde_json::Value::from(value));
            }
        }
        if let Some(stretch) = appearance.stretch_palette {
            fields.insert(String::from("stretch_palette"), serde_json::Value::from(stretch));
        }
        file["appearance"] = serde_json::Value::Object(fields);
    }
    serde_json::to_string_pretty(&file).expect("a point of interest is plain text and numbers, which always serialize") + "\n"
}

/// The file name for a new point of interest called `name` in a folder with the files `existing`.
///
/// It is the next number (two digits, after the highest number in the folder, so it is listed last), a `-` and the slugified name, made unique if needed.
///
/// # Panics
/// Never: of the candidates, one more than there are existing files, one is always free
pub fn new_file_name(existing: &[String], name: &str) -> String {
    let number = existing.iter().filter_map(|file| file.split('-').next()?.parse::<u32>().ok()).max().map_or(0, |highest| highest + 1);
    let slug = Some(slugify(name)).filter(|slug| !slug.is_empty()).unwrap_or_else(|| String::from("place"));
    let base = format!("{number:02}-{slug}");
    //One more candidate than there are files, so one of them is free
    (1..=existing.len() + 1).map(|copy| if copy == 1 { format!("{base}.json") } else { format!("{base}-{copy}.json") }).find(|file| !existing.iter().any(|other| other.eq_ignore_ascii_case(file))).expect("a free name")
}

/// Saves `point` as a new file in `folder` (created if needed), see `new_file_name`; returns the name of the file
///
/// # Errors
/// The folder cannot be read or created, or the file cannot be written
pub fn save_new(folder: &Path, point: &PointOfInterest) -> std::io::Result<String> {
    std::fs::create_dir_all(folder)?;
    let existing: Vec<String> = std::fs::read_dir(folder)?.filter_map(|entry| entry.ok()?.file_name().into_string().ok()).collect();
    let file_name = new_file_name(&existing, &point.name);
    //create_new never overwrites a file that appeared in the meantime
    std::io::Write::write_all(&mut std::fs::OpenOptions::new().write(true).create_new(true).open(folder.join(&file_name))?, to_json(point).as_bytes())?;
    Ok(file_name)
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
        assert_eq!(warnings, Vec::<String>::new());
        assert_eq!(point.name, "Test Place");
        assert_eq!(point.location, Location { real: -0.75, imaginary: 0.1, scale: 0.5, max_iterations: 200, rotation: 0.0 });
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
        let deep = |name: &str| VALID.replace("Test Place", name).replace("\"scale\": 0.5", "\"scale\": 0.001");
        let shallow = |name: &str| VALID.replace("Test Place", name).replace("\"scale\": 0.5", "\"scale\": 2.0");
        std::fs::write(folder.join("2-ant.json"), VALID.replace("Test Place", "Ant")).unwrap();
        std::fs::write(folder.join("1-zebra.json"), VALID.replace("Test Place", "Zebra")).unwrap();
        std::fs::write(folder.join("0-Yak.JSON"), VALID.replace("Test Place", "Yak")).unwrap();
        std::fs::write(folder.join("b-cat.json"), VALID.replace("Test Place", "Cat")).unwrap();
        std::fs::write(folder.join("A-bee.json"), VALID.replace("Test Place", "Bee")).unwrap();
        std::fs::write(folder.join("0-deep.json"), deep("Deep")).unwrap();
        std::fs::write(folder.join("z-shallow.json"), shallow("Shallow")).unwrap();
        std::fs::write(folder.join("broken.json"), "{ nope").unwrap();
        std::fs::write(folder.join("notes.txt"), "not a point").unwrap();
        std::fs::create_dir(folder.join("sub.json")).unwrap();
        let report = load_folder(&folder);
        let names: Vec<&str> = report.points.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["Shallow", "Yak", "Zebra", "Ant", "Bee", "Cat", "Deep"], "by zoom, the shallowest first whatever the file name; the same zoom in alphabetical order of the file names (0-, 1-, 2-, A-bee, b-cat, ignoring case), not of the display names");
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

    /// The numbers of the ten views that were hard-coded in `lib.rs` (`VIEW_1` to `VIEW_9`, then `VIEW_0`), by the name of the file that holds them now, in their original order; the file names count from 01
    const FORMER_VIEWS: [(&str, &str, f64, f64, f64); 10] = [
        ("01-frost-on-the-shoreline", "Frost on the Shoreline", -0.6604166666666667, 0.4437500000000001, 0.1),
        ("02-frozen-constellation", "Frozen Constellation", -1.0591666666666668, 0.2629166666666668, 0.01),
        ("03-harbor-of-moons", "Harbor of Moons", -0.4624999999999999, 0.55, 0.1),
        ("04-the-narrow-pass", "The Narrow Pass", -0.46395833333333325, 0.5531250000000001, 0.03),
        ("05-coral-atoll", "Coral Atoll", -0.4375218333333333, 0.5632133750000003, 0.00002000000000000002),
        ("06-seahorse-embers", "Seahorse Embers", -0.7498100000000001, -0.020300000000000054, 0.00006400000000000002),
        ("07-lone-comet", "Lone Comet", -1.7862712000000047, 0.000052399999999991516, 0.00001677721600000001),
        ("08-comets-crown", "Comet's Crown", -1.7862581627050718, 0.00005198056959995248, 0.000006039797760000003),
        ("09-the-whirlpool", "The Whirlpool", -0.4687339999999999, 0.5425518958333333, 0.000010000000000000003),
        ("10-sunlit-reef", "Sunlit Reef", -0.437520465811966, 0.5632133750000006, 0.000004000000000000004),
    ];

    #[test]
    fn the_shipped_folder_has_the_reset_view_first_then_the_ten_former_views_by_zoom_with_exactly_their_numbers() {
        let report = load_folder(&find_folder());
        assert!(report.problems.is_empty(), "{:?}", report.problems);
        assert_eq!(report.points.len(), 11);
        //The reset view first, the shallowest: exactly the view of `ComplexPlane::new`
        let reset = &report.points[0];
        let plane = crate::complex_plane::ComplexPlane::new(1200, 800);
        assert_eq!((reset.file_name.as_str(), reset.name.as_str()), ("00-reset-view.json", "Reset View"));
        assert_eq!((reset.location.real, reset.location.imaginary, reset.location.scale), (plane.center().x, plane.center().y, plane.get_scale()));
        //The former View 1 to View 10 (VIEW_0 last) are numbered 01 to 10
        for (index, (file, name, real, imaginary, scale)) in FORMER_VIEWS.into_iter().enumerate() {
            let number = format!("{:02}", index + 1);
            let point = report.points.iter().find(|p| p.file_name == format!("{file}.json")).unwrap_or_else(|| panic!("{file} is shipped"));
            assert!(file.starts_with(&format!("{number}-")), "the number of the file is the number of the view it replaced");
            assert_eq!(point.name, name);
            assert_eq!(format!("{number}-{}", slugify(&point.name)), file, "the file is named after its number and the display name");
            assert_ne!(point.description, "");
            assert_eq!((point.location.real.to_bits(), point.location.imaginary.to_bits(), point.location.scale.to_bits()), (real.to_bits(), imaginary.to_bits(), scale.to_bits()), "{file}");
            assert!(MAX_ITERATIONS_RANGE.contains(&point.location.max_iterations));
            let appearance = point.appearance.as_ref().expect("the default appearance is written out");
            assert_eq!((appearance.coloring, format!("{:?}", appearance.color_channels.unwrap())), (Some(Coloring::Bernstein), String::from("RGB")), "the default appearance of the application");
        }
        //Listed by zoom, the shallowest first; Frost on the Shoreline and Harbor of Moons have the same zoom, so they keep the order of their file names
        let files: Vec<&str> = report.points.iter().map(|p| p.file_name.trim_end_matches(".json")).collect();
        assert_eq!(
            files,
            [
                "00-reset-view",
                "01-frost-on-the-shoreline",
                "03-harbor-of-moons",
                "04-the-narrow-pass",
                "02-frozen-constellation",
                "06-seahorse-embers",
                "05-coral-atoll",
                "07-lone-comet",
                "09-the-whirlpool",
                "08-comets-crown",
                "10-sunlit-reef",
            ]
        );
        assert!(report.points.windows(2).all(|pair| pair[0].location.scale >= pair[1].location.scale), "zoom ascending");
    }

    #[test]
    fn a_saved_point_reads_back_exactly() {
        for (description, appearance) in [("", None), ("A spiral with a long arm.", Some(Appearance { coloring: Some(Coloring::Hsv), color_channels: Some(ColorChannelMapping::new("BGR").unwrap()), iteration_divide: Some(4.5), color_offset: Some(120.0), stretch_palette: Some(false) }))] {
            let point = PointOfInterest {
                name: String::from("Comet's Crown"),
                description: String::from(description),
                location: Location { real: -1.7862581627050718, imaginary: 0.00005198056959995248, scale: 0.000006039797760000003, max_iterations: 65_376, rotation: 0.0 },
                appearance,
                file_name: String::new(),
            };
            let (read, warnings) = parse(&to_json(&point), "test.json").unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!((read.location.real.to_bits(), read.location.imaginary.to_bits(), read.location.scale.to_bits()), (point.location.real.to_bits(), point.location.imaginary.to_bits(), point.location.scale.to_bits()), "the view, bit for bit");
            assert_eq!((read.name, read.description, read.location.max_iterations, read.appearance), (point.name.clone(), point.description.clone(), 65_376, point.appearance.clone()));
        }
    }

    #[test]
    fn a_new_file_gets_the_next_number_and_never_the_name_of_an_existing_file() {
        let existing = [String::from("00-reset-view.json"), String::from("10-sunlit-reef.json"), String::from("notes.txt")];
        assert_eq!(new_file_name(&existing, "My Place"), "11-my-place.json");
        assert_eq!(new_file_name(&[], "My Place"), "00-my-place.json");
        assert_eq!(new_file_name(&[String::from("00-my-place.json")], "My Place"), "01-my-place.json");
        assert_eq!(new_file_name(&[String::from("x-my-place.json"), String::from("00-my-place.json")], "My Place"), "01-my-place.json");
        assert_eq!(new_file_name(&existing, "!!!"), "11-place.json", "a name without letters or digits");
        let taken = [String::from("Notes"), String::from("00-a.json")];
        assert_eq!(new_file_name(&taken, "A"), "01-a.json");
    }

    #[test]
    fn an_iteration_divide_outside_the_slider_is_moved_to_its_nearest_end_and_a_negative_one_is_ignored() {
        let with = |divide: &str| parse(&VALID.replace("\"color_channels\": \"bgr\"", &format!("\"color_channels\": \"bgr\", \"iteration_divide\": {divide}")), "test.json").unwrap();
        assert_eq!(with("0.001").0.appearance.unwrap().iteration_divide, Some(0.1));
        assert_eq!(with("100000").0.appearance.unwrap().iteration_divide, Some(1000.0));
        assert_eq!(with("4.5").0.appearance.unwrap().iteration_divide, Some(4.5));
        let (point, warnings) = with("-2");
        assert_eq!(point.appearance.unwrap().iteration_divide, None);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
    }
}

//! Image resolutions: the built-in presets, the custom presets that the user defines in the window, and the file that keeps the custom presets between runs.
//!
//! A custom preset file is JSON and lives in the configuration folder of the user (see `custom_presets_path`):
//!
//! ```json
//! {
//!   "format_version": 1,
//!   "presets": [ { "name": "Print A3", "width": 4961, "height": 3508 } ]
//! }
//! ```
//!
//! A preset that cannot be used (an empty or duplicate name, a size that is not allowed) is skipped with a message; the others still load, see `parse_custom_presets`.
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The smallest width or height of an image in pixels
pub const MIN_SIDE: usize = 16;
/// The largest width or height of an image in pixels
pub const MAX_SIDE: usize = 16_384;
/// The most pixels an image can have (100 megapixels): a render holds a few bytes per pixel in memory, so the limit keeps a typo from asking for tens of gigabytes
pub const MAX_PIXELS: usize = 100_000_000;
/// The only `format_version` of the custom presets file that is understood
pub const FORMAT_VERSION: u32 = 1;
/// The name of the custom presets file
pub const FILE_NAME: &str = "resolution_presets.json";

/// A size in pixels
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolution {
    /// The width in pixels
    pub width: usize,
    /// The height in pixels
    pub height: usize,
}

impl Resolution {
    /// The resolution of `width` x `height` pixels, which may not be allowed, see `validate`
    pub const fn new(width: usize, height: usize) -> Resolution {
        Resolution { width, height }
    }

    /// The number of pixels
    pub const fn pixels(self) -> usize {
        self.width.saturating_mul(self.height)
    }

    /// Checks that the sides are from `MIN_SIDE` to `MAX_SIDE` and that there are at most `MAX_PIXELS` pixels
    /// # Errors
    /// The reason, as a sentence for the status log, if the resolution is not allowed
    pub fn validate(self) -> Result<Resolution, String> {
        let sides = MIN_SIDE..=MAX_SIDE;
        if !sides.contains(&self.width) || !sides.contains(&self.height) {
            return Err(format!("{self} is not allowed: the width and the height are from {MIN_SIDE} to {MAX_SIDE} pixels"));
        }
        if self.pixels() > MAX_PIXELS {
            return Err(format!("{self} is not allowed: {} megapixels is more than the {} megapixels that fit in memory", self.pixels() / 1_000_000, MAX_PIXELS / 1_000_000));
        }
        Ok(self)
    }
}

impl fmt::Display for Resolution {
    /// "1920 x 1080"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} x {}", self.width, self.height)
    }
}

/// A named resolution
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionPreset {
    /// The name shown in the list
    pub name: String,
    /// The size of the image
    pub resolution: Resolution,
}

impl ResolutionPreset {
    /// The text in the list: the name and the size, "Full HD (1920 x 1080)"
    pub fn label(&self) -> String {
        format!("{} ({})", self.name, self.resolution)
    }
}

/// The presets that are always there, from small to large
const BUILT_IN: [(&str, Resolution); 10] = [
    ("HD 720p", Resolution::new(1280, 720)),
    ("Full HD 1080p", Resolution::new(1920, 1080)),
    ("QHD 1440p", Resolution::new(2560, 1440)),
    ("Ultrawide QHD", Resolution::new(3440, 1440)),
    ("4K UHD", Resolution::new(3840, 2160)),
    ("5K", Resolution::new(5120, 2880)),
    ("8K UHD", Resolution::new(7680, 4320)),
    ("Square 2048", Resolution::new(2048, 2048)),
    ("Square 4096", Resolution::new(4096, 4096)),
    ("Portrait 1080 x 1920", Resolution::new(1080, 1920)),
];

/// The built-in presets, see `BUILT_IN`
pub fn built_in_presets() -> Vec<ResolutionPreset> {
    BUILT_IN.iter().map(|(name, resolution)| ResolutionPreset { name: (*name).to_string(), resolution: *resolution }).collect()
}

/// The shape of a custom preset in the file
#[derive(Serialize, Deserialize)]
struct StoredPreset {
    name: String,
    width: usize,
    height: usize,
}

/// The shape of the custom presets file
#[derive(Serialize, Deserialize)]
struct StoredPresets {
    format_version: u32,
    presets: Vec<StoredPreset>,
}

/// Reads the text of a custom presets file. Returns the presets that can be used, and a message for each one that cannot (or for the whole file).
///
/// A name must not be empty, must not be used by a built-in preset (ignoring case), and must not appear twice (the first one counts).
pub fn parse_custom_presets(text: &str) -> (Vec<ResolutionPreset>, Vec<String>) {
    let stored: StoredPresets = match serde_json::from_str(text) {
        Ok(stored) => stored,
        Err(err) => return (Vec::new(), vec![format!("The custom resolutions could not be read: {err}")]),
    };
    if stored.format_version != FORMAT_VERSION {
        return (Vec::new(), vec![format!("The custom resolutions have format_version {}, only {FORMAT_VERSION} is understood", stored.format_version)]);
    }
    let mut presets: Vec<ResolutionPreset> = Vec::new();
    let mut warnings = Vec::new();
    for entry in stored.presets {
        match check_name(&entry.name, &presets).and_then(|()| Resolution::new(entry.width, entry.height).validate()) {
            Ok(resolution) => presets.push(ResolutionPreset { name: entry.name.trim().to_string(), resolution }),
            Err(reason) => warnings.push(format!("Custom resolution \"{}\" is skipped: {reason}", entry.name)),
        }
    }
    (presets, warnings)
}

/// The text of a custom presets file for `presets`, see `parse_custom_presets`
/// # Panics
/// Never in practice: the presets are plain text and numbers, which always serialize
pub fn custom_presets_text(presets: &[ResolutionPreset]) -> String {
    let stored = StoredPresets {
        format_version: FORMAT_VERSION,
        presets: presets.iter().map(|preset| StoredPreset { name: preset.name.clone(), width: preset.resolution.width, height: preset.resolution.height }).collect(),
    };
    serde_json::to_string_pretty(&stored).expect("presets are plain text and numbers, which always serialize") + "\n"
}

/// Checks that `name` can be the name of a custom preset next to `others`.
///
/// It is not empty, not the name of a built-in preset and not used twice (ignoring case and surrounding spaces)
fn check_name(name: &str, others: &[ResolutionPreset]) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(String::from("the name is empty"));
    }
    if BUILT_IN.iter().any(|(built_in, _)| built_in.eq_ignore_ascii_case(name)) {
        return Err(String::from("a built-in preset has that name"));
    }
    if others.iter().any(|other| other.name.eq_ignore_ascii_case(name)) {
        return Err(String::from("the name is used twice"));
    }
    Ok(())
}

/// Adds the custom preset `name` with `resolution` to `presets`, or replaces the size of the one that has that name (ignoring case)
/// # Errors
/// The reason, if the name is empty or is the name of a built-in preset, or the resolution is not allowed (see `Resolution::validate`)
pub fn add_custom_preset(presets: &mut Vec<ResolutionPreset>, name: &str, resolution: Resolution) -> Result<(), String> {
    let name = name.trim();
    let others: Vec<ResolutionPreset> = presets.iter().filter(|preset| !preset.name.eq_ignore_ascii_case(name)).cloned().collect();
    check_name(name, &others)?;
    let resolution = resolution.validate()?;
    match presets.iter_mut().find(|preset| preset.name.eq_ignore_ascii_case(name)) {
        Some(existing) => existing.resolution = resolution,
        None => presets.push(ResolutionPreset { name: name.to_string(), resolution }),
    }
    Ok(())
}

/// Where the custom presets are kept: `FILE_NAME` in the `mandelbrot` folder of the configuration folder of the user.
///
/// That folder is `$XDG_CONFIG_HOME` or `~/.config` on Linux and macOS, and `%APPDATA%` on Windows. Returns `None` if the home of the user is not known.
pub fn custom_presets_path() -> Option<PathBuf> {
    let environment = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    let base = environment("XDG_CONFIG_HOME").or_else(|| environment("APPDATA")).or_else(|| environment("HOME").map(|home| home.join(".config")))?;
    Some(base.join("mandelbrot").join(FILE_NAME))
}

/// Loads the custom presets of the file at `path`. A file that does not exist is no presets and no message.
pub fn load_custom_presets(path: &Path) -> (Vec<ResolutionPreset>, Vec<String>) {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_custom_presets(&text),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => (Vec::new(), Vec::new()),
        Err(err) => (Vec::new(), vec![format!("The custom resolutions of {} could not be read: {err}", path.display())]),
    }
}

/// Saves `presets` to the file at `path`, creating its folder.
///
/// The text is written to a temporary file next to it first, so that a failure never leaves half a file.
/// # Errors
/// The error if the folder or the file cannot be written
pub fn save_custom_presets(path: &Path, presets: &[ResolutionPreset]) -> std::io::Result<()> {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, custom_presets_text(presets))?;
    std::fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(name: &str, width: usize, height: usize) -> ResolutionPreset {
        ResolutionPreset { name: name.to_string(), resolution: Resolution::new(width, height) }
    }

    #[test]
    fn the_sides_and_the_pixel_count_are_limited() {
        assert!(Resolution::new(1920, 1080).validate().is_ok());
        assert!(Resolution::new(MIN_SIDE, MAX_SIDE / 2).validate().is_ok());
        assert!(Resolution::new(MIN_SIDE - 1, 100).validate().unwrap_err().contains("from 16 to 16384"));
        assert!(Resolution::new(100, MAX_SIDE + 1).validate().is_err());
        assert!(Resolution::new(0, 0).validate().is_err());
        assert!(Resolution::new(MAX_SIDE, MAX_SIDE).validate().unwrap_err().contains("268 megapixels is more than the 100"));
        assert_eq!(Resolution::new(usize::MAX, 2).pixels(), usize::MAX, "the pixel count saturates instead of overflowing");
        assert_eq!(Resolution::new(1920, 1080).to_string(), "1920 x 1080");
    }

    #[test]
    fn the_built_in_presets_are_all_allowed_and_have_distinct_names() {
        let presets = built_in_presets();
        assert_eq!(presets.len(), 10);
        for preset in &presets {
            assert!(preset.resolution.validate().is_ok(), "{}", preset.name);
        }
        let mut names: Vec<String> = presets.iter().map(|preset| preset.name.to_lowercase()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 10);
        assert_eq!(presets[1].label(), "Full HD 1080p (1920 x 1080)");
    }

    #[test]
    fn custom_presets_survive_a_round_trip_through_their_text() {
        let presets = vec![preset("Print A3", 4961, 3508), preset("Tiny", 64, 64)];
        let (read, warnings) = parse_custom_presets(&custom_presets_text(&presets));
        assert_eq!((read, warnings), (presets, Vec::new()));
    }

    #[test]
    fn presets_that_cannot_be_used_are_skipped_with_a_message_and_the_others_load() {
        let text = r#"{ "format_version": 1, "presets": [
            { "name": "Good", "width": 800, "height": 600 },
            { "name": "  ", "width": 800, "height": 600 },
            { "name": "full hd 1080P", "width": 800, "height": 600 },
            { "name": "GOOD", "width": 900, "height": 600 },
            { "name": "Huge", "width": 16384, "height": 16384 },
            { "name": "Thin", "width": 4, "height": 600 },
            { "name": "Padded ", "width": 640, "height": 480 }
        ] }"#;
        let (presets, warnings) = parse_custom_presets(text);
        assert_eq!(presets, vec![preset("Good", 800, 600), preset("Padded", 640, 480)]);
        assert_eq!(warnings.len(), 5);
        assert!(warnings[0].contains("the name is empty") && warnings[1].contains("built-in") && warnings[2].contains("used twice"));
        assert!(warnings[3].contains("Huge") && warnings[4].contains("Thin"));
    }

    #[test]
    fn a_file_that_cannot_be_read_gives_one_message_and_no_presets() {
        let (presets, warnings) = parse_custom_presets("{ nope");
        assert!(presets.is_empty() && warnings.len() == 1 && warnings[0].starts_with("The custom resolutions could not be read"));
        let (presets, warnings) = parse_custom_presets(r#"{ "format_version": 2, "presets": [] }"#);
        assert!(presets.is_empty() && warnings[0].contains("format_version 2"));
    }

    #[test]
    fn adding_a_preset_replaces_the_size_of_the_one_with_the_same_name() {
        let mut presets = Vec::new();
        add_custom_preset(&mut presets, "  Poster ", Resolution::new(6000, 4000)).unwrap();
        add_custom_preset(&mut presets, "Banner", Resolution::new(3000, 500)).unwrap();
        add_custom_preset(&mut presets, "poster", Resolution::new(7000, 5000)).unwrap();
        assert_eq!(presets, vec![preset("Poster", 7000, 5000), preset("Banner", 3000, 500)]);
        assert!(add_custom_preset(&mut presets, "4k uhd", Resolution::new(10, 10)).unwrap_err().contains("built-in"));
        assert!(add_custom_preset(&mut presets, "", Resolution::new(800, 600)).is_err());
        assert!(add_custom_preset(&mut presets, "Too big", Resolution::new(20000, 600)).is_err());
        assert_eq!(presets.len(), 2, "a refused preset changes nothing");
    }

    #[test]
    fn custom_presets_are_saved_to_a_file_and_loaded_again() {
        let folder = std::env::temp_dir().join(format!("mandelbrot-resolution-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        let path = folder.join("nested").join(FILE_NAME);
        assert_eq!(load_custom_presets(&path), (Vec::new(), Vec::new()), "a file that does not exist is not a problem");
        let presets = vec![preset("Poster", 6000, 4000)];
        save_custom_presets(&path, &presets).unwrap();
        assert_eq!(load_custom_presets(&path), (presets, Vec::new()));
        assert!(!path.with_extension("json.tmp").exists(), "the temporary file is renamed away");
        std::fs::write(&path, "garbage").unwrap();
        assert_eq!(load_custom_presets(&path).1.len(), 1);
        std::fs::remove_dir_all(&folder).unwrap();
    }
}

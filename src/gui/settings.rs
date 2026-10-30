//! The settings of the window, kept between runs (issue #127): set the values and options you like once, and they are there the next time.
//!
//! They are saved as JSON to the settings file (`settings_path`: `mandelbrot/settings.json` in the configuration folder of the user) whenever one changes, and
//! loaded when the window opens. Every value is optional in the file: a value it does not have keeps its default, and a value the window does not know is ignored,
//! so a file of an older or newer version still loads. What is not a setting stays out: the view, the history, the places. **Reset settings** puts every
//! setting back to its default and saves that.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::flight::{FlightRendering, Pace};
use super::{MandelbrotApp, COLORING_FUNCTIONS};
use crate::auto_iterations::AutoPreset;
use crate::coloring::ColorChannelMapping;

/// The settings, as the file has them (every one optional, see the module)
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct Settings {
    move_step: Option<u8>,
    zoom_speed: Option<f64>,
    fit_window: Option<bool>,
    max_iterations: Option<u32>,
    auto_iterations: Option<bool>,
    auto_preset: Option<String>,
    auto_start: Option<u32>,
    auto_per_doubling: Option<u32>,
    auto_ceiling: Option<u32>,
    adaptive_supersampling: Option<bool>,
    supersampling: Option<u8>,
    saved_image_supersampling: Option<u8>,
    coloring: Option<String>,
    color_channels: Option<String>,
    smooth_colors: Option<bool>,
    use_gpu: Option<bool>,
    fast_gpu: Option<bool>,
    limit_zoom: Option<bool>,
    restore_appearance: Option<bool>,
    show_points: Option<bool>,
    pace: Option<String>,
    flight_rendering: Option<String>,
    flight_resolution: Option<u16>,
    immersion: Option<bool>,
    auto_blur: Option<bool>,
}

/// The auto iterations presets, to find one by its name
const PRESETS: [AutoPreset; 4] = [AutoPreset::Performance, AutoPreset::Balanced, AutoPreset::HighDetail, AutoPreset::Custom];

/// The flight renderings by the name in the file, and the name of a rendering
fn rendering_key(rendering: FlightRendering) -> (&'static str, Option<u16>) {
    match rendering {
        FlightRendering::BestEffort => ("best-effort", None),
        FlightRendering::LowRes => ("low-res", None),
        FlightRendering::Resolution(percent) => ("resolution", Some(percent)),
        FlightRendering::SuperSampled => ("super-sampled", None),
    }
}

fn rendering_of(key: &str, percent: Option<u16>) -> Option<FlightRendering> {
    match key {
        "best-effort" => Some(FlightRendering::BestEffort),
        "low-res" => Some(FlightRendering::LowRes),
        "resolution" => Some(FlightRendering::Resolution(percent.unwrap_or(100).clamp(1, 800))),
        "super-sampled" => Some(FlightRendering::SuperSampled),
        _ => None,
    }
}

/// The settings file of the user: `mandelbrot/settings.json` in `$XDG_CONFIG_HOME`, `~/.config` or `%APPDATA%`; `None` if there is no such folder
pub fn settings_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))?;
    Some(base.join("mandelbrot").join("settings.json"))
}

/// How long the settings wait after a change before they are written (a drag of a value changes it every frame)
const SAVE_AFTER: Duration = Duration::from_millis(500);

impl MandelbrotApp {
    /// The settings of the window now
    pub(super) fn settings(&self) -> Settings {
        let (rendering, percent) = rendering_key(self.flight_rendering);
        Settings {
            move_step: Some(self.vars.translation_amount),
            zoom_speed: Some(self.vars.scale_numerator),
            fit_window: Some(self.fit.enabled),
            max_iterations: Some(self.max_iterations_input),
            auto_iterations: Some(self.auto_iterations),
            auto_preset: Some(self.auto_preset.name().to_string()),
            auto_start: Some(self.auto_settings.start),
            auto_per_doubling: Some(self.auto_settings.per_doubling),
            auto_ceiling: Some(self.auto_settings.ceiling),
            adaptive_supersampling: Some(self.adaptive_supersampling),
            supersampling: Some(self.supersampling_amount),
            saved_image_supersampling: Some(self.image_supersampling_amount),
            coloring: Some(COLORING_FUNCTIONS[self.coloring_index].0.to_string()),
            color_channels: Some(format!("{:?}", self.p.color_channel_mapping)),
            smooth_colors: Some(self.m.smooth_coloring),
            use_gpu: Some(self.use_gpu),
            fast_gpu: Some(self.fast_gpu),
            limit_zoom: Some(self.limit_zoom),
            restore_appearance: Some(self.restore_appearance),
            show_points: Some(self.show_points),
            pace: Some(self.pace.name().to_string()),
            flight_rendering: Some(rendering.to_string()),
            flight_resolution: percent,
            immersion: Some(self.immersion),
            auto_blur: Some(self.auto_blur),
        }
    }

    /// Applies the settings that `settings` has (the others stay as they are)
    pub(super) fn apply_settings(&mut self, settings: &Settings) {
        let set = |value: Option<bool>, target: &mut bool| {
            if let Some(value) = value {
                *target = value;
            }
        };
        if let Some(step) = settings.move_step {
            self.vars.translation_amount = step.max(1);
        }
        if let Some(speed) = settings.zoom_speed.filter(|speed| (1.0..=9.0).contains(speed)) {
            self.vars.scale_numerator = speed;
        }
        set(settings.fit_window, &mut self.fit.enabled);
        if let Some(limit) = settings.max_iterations.filter(|limit| *limit >= 1) {
            self.max_iterations_input = limit;
        }
        set(settings.auto_iterations, &mut self.auto_iterations);
        if let Some(preset) = settings.auto_preset.as_deref().and_then(|name| PRESETS.into_iter().find(|preset| preset.name() == name)) {
            self.auto_preset = preset;
            if preset != AutoPreset::Custom {
                self.auto_settings = preset.settings();
            }
        }
        if self.auto_preset == AutoPreset::Custom {
            for (value, target) in [(settings.auto_start, &mut self.auto_settings.start), (settings.auto_per_doubling, &mut self.auto_settings.per_doubling), (settings.auto_ceiling, &mut self.auto_settings.ceiling)] {
                if let Some(value) = value {
                    *target = value;
                }
            }
        }
        set(settings.adaptive_supersampling, &mut self.adaptive_supersampling);
        if let Some(samples) = settings.supersampling.filter(|samples| (1..=64).contains(samples)) {
            (self.supersampling_amount, self.supersampling_input) = (samples, samples);
        }
        if let Some(samples) = settings.saved_image_supersampling.filter(|samples| (1..=64).contains(samples)) {
            self.image_supersampling_amount = samples;
        }
        if let Some(index) = settings.coloring.as_deref().and_then(|name| COLORING_FUNCTIONS.iter().position(|(known, _)| *known == name)) {
            (self.coloring_index, self.coloring_function) = (index, COLORING_FUNCTIONS[index].1);
        }
        if let Some(mapping) = settings.color_channels.as_deref().and_then(|channels| ColorChannelMapping::new(channels).ok()) {
            self.p.color_channel_mapping = mapping;
            self.texture_outdated = true;
        }
        set(settings.smooth_colors, &mut self.m.smooth_coloring);
        set(settings.use_gpu, &mut self.use_gpu);
        set(settings.fast_gpu, &mut self.fast_gpu);
        set(settings.limit_zoom, &mut self.limit_zoom);
        set(settings.restore_appearance, &mut self.restore_appearance);
        set(settings.show_points, &mut self.show_points);
        if let Some(pace) = settings.pace.as_deref().and_then(|name| Pace::ALL.into_iter().find(|pace| pace.name() == name)) {
            self.pace = pace;
        }
        if let Some(rendering) = settings.flight_rendering.as_deref().and_then(|key| rendering_of(key, settings.flight_resolution)) {
            self.flight_rendering = rendering;
        }
        set(settings.immersion, &mut self.immersion);
        set(settings.auto_blur, &mut self.auto_blur);
    }

    /// Keeps the settings in the file `path` from now on: the settings as they are now are the defaults (for Reset settings), and those of the file are applied.
    /// Returns whether the file changed any.
    pub fn use_settings_file(&mut self, path: Option<PathBuf>) -> bool {
        self.settings_defaults = self.settings();
        let loaded = path.as_deref().and_then(|path| match read_settings(path) {
            Ok(settings) => settings,
            Err(err) => {
                self.log(format!("Settings: could not read {}: {err}; the defaults are used", path.display()));
                None
            }
        });
        if let Some(settings) = &loaded {
            self.apply_settings(settings);
        }
        self.settings_saved = Some(self.settings());
        self.settings_path = path;
        loaded.is_some_and(|settings| settings != Settings::default()) && self.settings() != self.settings_defaults
    }

    /// Writes the settings when they changed (every frame of the window, see `SAVE_AFTER`)
    pub(super) fn keep_settings(&mut self, now: Instant) {
        let Some(path) = self.settings_path.clone() else { return };
        if self.benchmark.is_some() {
            return; //A benchmark tour changes the pace and the rendering for a while, see `benchmark`: those are not the user's settings
        }
        let settings = self.settings();
        if self.settings_saved.as_ref() == Some(&settings) {
            self.settings_changed = None;
            return;
        }
        let changed = *self.settings_changed.get_or_insert(now);
        if now.saturating_duration_since(changed) < SAVE_AFTER {
            return;
        }
        if let Err(err) = write_settings(&path, &settings) {
            self.log(format!("Settings: could not save {}: {err}", path.display()));
        }
        (self.settings_saved, self.settings_changed) = (Some(settings), None);
    }

    /// Puts every setting back to its default, saves that, and renders the view with them (the Reset settings button)
    pub(super) fn reset_settings(&mut self) {
        let defaults = self.settings_defaults.clone();
        self.apply_settings(&defaults);
        if let Some(path) = self.settings_path.clone() {
            if let Err(err) = write_settings(&path, &defaults) {
                self.log(format!("Settings: could not save {}: {err}", path.display()));
            }
        }
        self.settings_saved = Some(self.settings());
        self.render("Reset the settings to their defaults");
    }
}

/// The settings in the file at `path`; `None` if there is no file yet
fn read_settings(path: &Path) -> Result<Option<Settings>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map(Some).map_err(|err| err.to_string()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err.to_string()),
    }
}

fn write_settings(path: &Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(settings).map_err(std::io::Error::other)? + "\n")
}

#[cfg(test)]
impl Settings {
    pub(super) fn supersampling_for_test(self) -> u8 {
        self.supersampling.unwrap_or_default()
    }
}

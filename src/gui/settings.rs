//! The settings of the window, kept between runs (issue #127): set the values and options you like once, and they are there the next time.
//!
//! They are saved as JSON to the settings file (`settings_path`: `mandelbrot/settings.json` in the configuration folder of the user) whenever one changes, and
//! loaded when the window opens. Every value is optional in the file: a value it does not have keeps its default, and a value the window does not know is ignored,
//! so a file of an older or newer version still loads. What is not a setting stays out: the view, the history, the places. **Reset settings** puts every
//! setting back to its default and saves that.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::flight::{FlightRendering, Pace, Route, CONSTANT_SPEEDS, DIVE_ZOOMS, FLIGHT_SPEEDS};
use super::video::{Playback, VideoQuality, FRAMES_AT_ONCE, FRAME_RATES, SPEEDS};
use crate::mandelbrot_set::{MAX_DIVIDE, MIN_DIVIDE, PALETTE_LENGTH};
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
    route: Option<String>,
    constant_speed: Option<f64>,
    flight_speed: Option<u16>,
    dive_zoom: Option<f64>,
    flight_rendering: Option<String>,
    flight_resolution: Option<u16>,
    immersion: Option<bool>,
    auto_blur: Option<bool>,
    cycle_rate: Option<f64>,
    frames_ahead: Option<usize>,
    blend_frames: Option<bool>,
    f32_frames: Option<bool>,
    quality_blur: Option<bool>,
    saved_image_blur: Option<bool>,
    render_neighborhood: Option<bool>,
    neighborhood_size: Option<u16>,
    color_divide: Option<f64>,
    color_offset: Option<f64>,
    stretch_palette: Option<bool>,
    video_fps: Option<u32>,
    video_speed: Option<u16>,
    video_quality: Option<String>,
    video_playback: Option<String>,
    video_pngs: Option<bool>,
    video_blend: Option<bool>,
    video_frames_at_once: Option<usize>,
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
        FlightRendering::Auto(fps) => ("auto", Some(fps)),
    }
}

fn rendering_of(key: &str, percent: Option<u16>) -> Option<FlightRendering> {
    match key {
        "best-effort" => Some(FlightRendering::BestEffort),
        "low-res" => Some(FlightRendering::LowRes),
        "resolution" => Some(FlightRendering::Resolution(percent.unwrap_or(100).clamp(1, 800))),
        "super-sampled" => Some(FlightRendering::SuperSampled),
        "auto" => Some(FlightRendering::Auto(percent.unwrap_or(60).clamp(1, 240))),
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
            route: Some(self.route.name().to_string()),
            constant_speed: Some(self.constant_speed),
            flight_speed: Some(self.flight_speed),
            dive_zoom: Some(self.dive_zoom),
            flight_rendering: Some(rendering.to_string()),
            flight_resolution: percent,
            immersion: Some(self.immersion),
            auto_blur: Some(self.auto_blur),
            cycle_rate: Some(self.cycle_rate),
            frames_ahead: Some(self.frames_ahead),
            blend_frames: Some(self.blend_frames),
            f32_frames: Some(self.f32_frames),
            quality_blur: Some(self.quality_blur),
            saved_image_blur: Some(self.image_quality_blur),
            render_neighborhood: Some(self.render_neighborhood),
            neighborhood_size: Some(self.neighborhood_margin),
            color_divide: Some(self.m.palette.divide),
            color_offset: Some(self.m.palette.offset),
            stretch_palette: Some(self.m.palette.stretch),
            video_fps: Some(self.video_fps),
            video_speed: Some(self.video_speed),
            video_quality: Some(self.video_quality.name().to_string()),
            video_playback: Some(self.video_playback.name().to_string()),
            video_pngs: Some(self.video_pngs),
            video_blend: Some(self.video_blend),
            video_frames_at_once: Some(self.video_frames_at_once),
        }
    }

    /// Applies the settings that `settings` has (the others stay as they are)
    #[allow(clippy::too_many_lines)] // One check per setting
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
        if let Some(route) = settings.route.as_deref().and_then(|name| Route::ALL.into_iter().find(|route| route.name() == name)) {
            self.route = route;
        }
        if let Some(speed) = settings.constant_speed.filter(|speed| CONSTANT_SPEEDS.contains(speed)) {
            self.constant_speed = speed;
        }
        if let Some(speed) = settings.flight_speed.filter(|speed| FLIGHT_SPEEDS.contains(speed)) {
            self.flight_speed = speed;
        }
        if let Some(zoom) = settings.dive_zoom.filter(|zoom| DIVE_ZOOMS.contains(zoom)) {
            self.dive_zoom = zoom;
        }
        if let Some(rendering) = settings.flight_rendering.as_deref().and_then(|key| rendering_of(key, settings.flight_resolution)) {
            self.flight_rendering = rendering;
        }
        set(settings.immersion, &mut self.immersion);
        set(settings.auto_blur, &mut self.auto_blur);
        if let Some(rate) = settings.cycle_rate.filter(|rate| super::cycle::CYCLE_RATES.contains(rate)) {
            self.cycle_rate = rate;
        }
        if let Some(ahead) = settings.frames_ahead.filter(|ahead| *ahead <= 3) {
            self.frames_ahead = ahead;
        }
        set(settings.blend_frames, &mut self.blend_frames);
        set(settings.f32_frames, &mut self.f32_frames);
        set(settings.quality_blur, &mut self.quality_blur);
        set(settings.saved_image_blur, &mut self.image_quality_blur);
        set(settings.render_neighborhood, &mut self.render_neighborhood);
        if let Some(size) = settings.neighborhood_size.filter(|size| (10..=5000).contains(size)) {
            self.neighborhood_margin = size;
        }
        //(A divide outside the range of its slider: the nearest end, as for a place)
        if let Some(divide) = settings.color_divide.filter(|divide| *divide > 0.0) {
            self.m.palette.divide = divide.clamp(MIN_DIVIDE, MAX_DIVIDE);
        }
        if let Some(offset) = settings.color_offset.filter(|offset| offset.is_finite()) {
            self.m.palette.offset = offset.rem_euclid(PALETTE_LENGTH);
        }
        set(settings.stretch_palette, &mut self.m.palette.stretch);
        if let Some(fps) = settings.video_fps.filter(|fps| FRAME_RATES.contains(fps)) {
            self.video_fps = fps;
        }
        if let Some(speed) = settings.video_speed.filter(|speed| SPEEDS.contains(speed)) {
            self.video_speed = speed;
        }
        if let Some(quality) = settings.video_quality.as_deref().and_then(|name| VideoQuality::ALL.into_iter().find(|quality| quality.name() == name)) {
            self.video_quality = quality;
        }
        if let Some(playback) = settings.video_playback.as_deref().and_then(|name| Playback::ALL.into_iter().find(|playback| playback.name() == name)) {
            self.video_playback = playback;
        }
        if let Some(at_once) = settings.video_frames_at_once.filter(|at_once| FRAMES_AT_ONCE.contains(at_once)) {
            self.video_frames_at_once = at_once;
        }
        set(settings.video_pngs, &mut self.video_pngs);
        set(settings.video_blend, &mut self.video_blend);
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

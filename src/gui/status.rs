//! The status side of the window: the current view, the status log and the render progress bar, and the layout of the side panel.

use std::sync::atomic::Ordering;

use eframe::egui::{self, Color32};

use crate::complex::Complex;
use crate::complex_plane::{ComplexPlane, Precision};
use super::render_job::RenderBars;
use super::MandelbrotApp;


/// The level of a message of the status log (issue #205): the log shows the messages of the chosen level and above
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Renders, layers and moves: what the program does all the time
    Debug,
    /// What happened: saved, flown, set
    Info,
    /// What was refused or skipped
    Warning,
    /// What failed
    Error,
}

impl LogLevel {
    pub const ALL: [LogLevel; 4] = [LogLevel::Debug, LogLevel::Info, LogLevel::Warning, LogLevel::Error];

    /// The name in the list and the settings
    pub const fn name(self) -> &'static str {
        match self {
            LogLevel::Debug => "Debug",
            LogLevel::Info => "Info",
            LogLevel::Warning => "Warnings",
            LogLevel::Error => "Errors",
        }
    }

    /// The level of `message`, by what it says (the messages are sentences, see `MandelbrotApp::log`)
    pub fn of(message: &str) -> LogLevel {
        let lower = message.to_lowercase();
        if message.starts_with("Could not") || message.starts_with("GPU error") || lower.contains(" failed") {
            LogLevel::Error
        } else if lower.contains("not allowed") || lower.contains("is skipped") || lower.contains("refused") || lower.contains("cannot be read") || message.starts_with("Warning") {
            LogLevel::Warning
        } else if message.starts_with("Rendering") || message.starts_with("Rendered") || message.starts_with("Cancelled the rendering") || message.starts_with("Moved by dragging")
            || message.starts_with("Zoomed in with the wheel") || message.starts_with("Zoomed in with a double click") || message.starts_with("Zoomed out with the wheel") || message.starts_with("Set the depth to") || message.starts_with("Clicked pixel") {
            LogLevel::Debug
        } else {
            LogLevel::Info
        }
    }
}

/// The colors the mini-map was rendered with: the coloring, the divide and offset (as bits), stretched, and the channels
pub type MinimapColors = (usize, u64, u64, bool, String);

/// The size of the mini-map in pixels, and the iteration limit of its render (the whole set needs few)
const MINIMAP_SIZE: (usize, usize) = (240, 160);
const MINIMAP_ITERATIONS: u32 = 500;

/// The texts of the current view: the center and the zoom, formatted for reading, without losing the precision of the numbers
#[derive(Debug, PartialEq, Eq)]
pub struct CurrentView {
    /// The real part of the center, all digits (the shortest text that reads back as the same number)
    pub real: String,
    /// The scale of the view, all digits (issue #207)
    pub scale: String,
    /// The imaginary part of the center, all digits
    pub imaginary: String,
    /// The zoom: the magnification relative to the starting view, `1 / scale`, see `format_zoom`. At scale 1, the view is 2.5 units wide on the real axis.
    pub zoom: String,
}

impl CurrentView {
    /// The texts for the view with the center `center` and the scale `scale`
    pub fn new(center: &Complex, scale: f64) -> CurrentView {
        CurrentView { real: format!("{}", center.x), imaginary: format!("{}", center.y), scale: format!("{scale}"), zoom: format_zoom(1.0 / scale) }
    }
}

/// The zoom as text, "x" and the magnification: ordinary decimal numbers from 0.001 up to a billion (at most two decimals, none for a whole number, thousands separated by commas:
///
/// "x1", "x2.5", "x156.25", "x15,625", "x250,000"), and scientific notation only beyond that ("x1.00e9", "x1.00e-4").
pub fn format_zoom(magnification: f64) -> String {
    if !magnification.is_finite() || magnification <= 0.0 {
        return String::from("\u{d7}?");
    }
    let trimmed = |text: String| if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text };
    if magnification < 0.001 || magnification >= 1e9 {
        return format!("\u{d7}{magnification:.2e}");
    }
    if magnification >= 999.995 {
        let digits = format!("{:.0}", magnification);
        let mut grouped = String::new();
        for (index, digit) in digits.chars().enumerate() {
            if index > 0 && (digits.len() - index) % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(digit);
        }
        return format!("\u{d7}{grouped}");
    }
    if magnification >= 1.0 {
        format!("\u{d7}{}", trimmed(format!("{magnification:.2}")))
    } else {
        format!("\u{d7}{}", trimmed(format!("{magnification:.4}")))
    }
}

/// The warning of the current view panel when `Precision::Degrading`
const PRECISION_DEGRADING_TEXT: &str = "Close to the limit of 64-bit numbers: the image may start to show noise or blocks.";
/// The warning of the current view panel when `Precision::Exhausted`
const PRECISION_EXHAUSTED_TEXT: &str = "Past the limit of 64-bit numbers: this image is not reliable, and Zoom in is disabled.";

impl MandelbrotApp {
    /// Draws the render progress bar: a main bar with a smaller bar for every supersampling layer, each showing how long that layer took.
    /// The caption above it shows the total time of the render.
    pub(super) fn render_bars_ui(ui: &mut egui::Ui, bars: &RenderBars) {
        let caption = match bars.finished {
            Some((total, true)) => format!("Rendered in {:.2} s", total.as_secs_f64()),
            Some((total, false)) => format!("Cancelled after {} of {} layers, {:.2} s", bars.layer_times.len(), bars.total_layers, total.as_secs_f64()),
            None => format!("Rendering: {:.1} s", bars.started.elapsed().as_secs_f64()),
        };
        ui.label(caption);

        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 30.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(40));
        let inner = rect.shrink(3.0);
        let gap = 2.0;
        let layers = f32::from(bars.total_layers);
        let layer_width = (inner.width() - gap * (layers - 1.0)) / layers;
        let current_fraction = (bars.pixels_done.load(Ordering::Relaxed) as f32 / bars.pixels_total.max(1) as f32).min(1.0);
        for layer in 0..usize::from(bars.total_layers) {
            let left = inner.left() + layer as f32 * (layer_width + gap);
            let layer_rect = egui::Rect::from_min_size(egui::pos2(left, inner.top()), egui::vec2(layer_width, inner.height()));
            painter.rect_filled(layer_rect, 2.0, Color32::from_gray(70));
            let finished_time = bars.layer_times.get(layer);
            let is_current = finished_time.is_none() && layer == bars.layer_times.len() && bars.finished.is_none();
            let fraction = if finished_time.is_some() { 1.0 } else if is_current { current_fraction.clamp(0.0, 1.0) } else { 0.0 };
            let fill = egui::Rect::from_min_size(layer_rect.min, egui::vec2(layer_width * fraction, layer_rect.height()));
            painter.rect_filled(fill, 2.0, Color32::from_rgb(60, 140, 220));
            let label = match finished_time {
                Some(time) => Some(format!("{:.2}s", time.as_secs_f64())),
                None if is_current => Some(format!("{:.1}s", bars.layer_started.elapsed().as_secs_f64())),
                None => None,
            };
            if let Some(label) = label {
                if layer_width >= 40.0 {
                    painter.text(layer_rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(11.0), Color32::WHITE);
                }
            }
        }
    }

    /// The right side of the window: the controls on top, below them the current view, and at the bottom the status
    pub(super) fn side_panel_ui(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status_panel").resizable(true).default_size(220.0).min_size(80.0).show(ui, |ui| self.log_ui(ui));
        //Added after the status panel, so it sits above it
        egui::Panel::bottom("current_view_panel").resizable(false).show(ui, |ui| {
            self.current_view_ui(ui);
            self.minimap_ui(ui);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let enabled = self.controls_enabled();
                ui.add_enabled_ui(enabled, |ui| self.controls_ui(ui));
            });
        });
    }

    /// The mini-map (issue #206, off by default): the whole set, small, with where the view is on it, so you keep your bearings deep down. A view smaller
    /// than a few pixels of the map gets a cross with a small ring around it instead of its outline
    pub(super) fn minimap_ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.show_minimap, "Mini-map").on_hover_text("A small map of the whole set with the view on it, to keep your bearings deep down");
        if !self.show_minimap {
            return;
        }
        let (width, height) = MINIMAP_SIZE;
        //The whole set in the colors of the view, rendered again when they change (a few milliseconds)
        let key = (self.coloring_index, self.m.palette.divide.to_bits(), self.m.palette.offset.to_bits(), self.m.palette.stretch, format!("{:?}", self.p.color_channel_mapping));
        if self.minimap.as_ref().is_none_or(|(old, _, _)| *old != key) {
            let mut plane = ComplexPlane::new(width, height);
            plane.set_view(&crate::complex_plane::View::new(-0.75, 0.0, 1.2));
            let mut set = self.m.clone();
            set.max_iterations = MINIMAP_ITERATIONS;
            let mut buffer = crate::pixel_buffer::PixelBuffer::new(crate::pixel_buffer::pixel_plane::PixelPlane::new(width, height));
            buffer.color_channel_mapping = self.p.color_channel_mapping;
            crate::rendering::render_complex_plane_into_buffer(&mut buffer, &plane, &set, 4, self.coloring_function);
            let rgb = crate::pixel_buffer::rgb_bytes(&buffer.colors, buffer.color_channel_mapping);
            let texture = ui.ctx().load_texture("minimap", egui::ColorImage::from_rgb([width, height], &rgb), egui::TextureOptions::LINEAR);
            self.minimap = Some((key, texture, plane));
        }
        let Some((_, texture, plane)) = &self.minimap else { return };
        let size = egui::vec2(width as f32, height as f32);
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.image(texture.id(), rect, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)), Color32::WHITE);
        //Where a point of the plane is on the map (the map is unturned: a turned view shows as a turned outline)
        let on_map = |point: Complex| rect.min + egui::vec2(((point.x - plane.min_x) / plane.increment_x) as f32, ((-point.y - plane.min_y) / plane.increment_y) as f32);
        let (view_width, view_height) = (self.p.pixel_plane.width as f64, self.p.pixel_plane.height as f64);
        let corners: Vec<egui::Pos2> = [(0.0, 0.0), (view_width, 0.0), (view_width, view_height), (0.0, view_height)].iter().map(|(x, y)| on_map(self.c.complex_from_pixel_plane(*x, *y))).collect();
        let center = on_map(self.c.center());
        let stroke = egui::Stroke::new(1.5, Color32::WHITE);
        let span = corners.iter().map(|corner| corner.distance(center)).fold(0.0_f32, f32::max);
        if span >= 3.0 {
            painter.add(egui::Shape::closed_line(corners, stroke));
        } else {
            painter.circle_stroke(center, 5.0, stroke);
            painter.line_segment([center - egui::vec2(9.0, 0.0), center + egui::vec2(9.0, 0.0)], stroke);
            painter.line_segment([center - egui::vec2(0.0, 9.0), center + egui::vec2(0.0, 9.0)], stroke);
        }
    }

    /// The center and zoom of the view that is shown, read from the complex plane every frame, so it follows every move, zoom and jump at once
    pub(super) fn current_view_ui(&self, ui: &mut egui::Ui) {
        let view = CurrentView::new(&self.c.center(), self.c.get_scale());
        ui.horizontal(|ui| {
            ui.heading("Current view");
            if ui.button("Copy").on_hover_text("Copies the center and the scale with all their digits").clicked() {
                ui.ctx().copy_text(self.view_description());
            }
        });
        egui::Grid::new("current_view_grid").num_columns(2).show(ui, |ui| {
            let value = |ui: &mut egui::Ui, text: &str| {
                ui.add(egui::Label::new(egui::RichText::new(text).monospace()).selectable(true));
            };
            ui.label("Real:");
            value(ui, &view.real);
            ui.end_row();
            ui.label("Imaginary:");
            value(ui, &view.imaginary);
            ui.end_row();
            ui.label("Scale:").on_hover_text("The width of the view on the real axis divided by 2.5 (scale 1 is the starting view): what Copy and Go to view use");
            value(ui, &view.scale);
            ui.end_row();
            ui.label("Zoom:").on_hover_text("The magnification relative to the starting view: 1 / scale. At scale 1 the view is 2.5 units wide on the real axis.");
            value(ui, &view.zoom);
            ui.end_row();
        });
        match self.c.precision() {
            Precision::Full => {}
            Precision::Degrading => {
                ui.colored_label(Color32::from_rgb(230, 160, 40), PRECISION_DEGRADING_TEXT);
            }
            Precision::Exhausted => {
                ui.colored_label(Color32::from_rgb(230, 70, 60), PRECISION_EXHAUSTED_TEXT);
            }
        }
    }

    /// Draws the status: the render that is running with its Cancel button, the progress bar of the last render, and the scrollable log
    pub(super) fn log_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Status");
            egui::ComboBox::from_id_salt("log_level").selected_text(self.log_level.name()).width(80.0).show_ui(ui, |ui| {
                for level in LogLevel::ALL {
                    ui.selectable_value(&mut self.log_level, level, level.name());
                }
            }).response.on_hover_text("The messages the status log shows (and prints): every one (Debug: renders and moves too), Info (what happened), only warnings and errors, or only errors. The hidden ones are kept: a lower level shows them again");
            if self.is_busy() {
                ui.spinner();
                match &self.progress {
                    Some(progress) => {
                        ui.label(format!("Rendering layer {} of {}...", (progress.pass + 1).min(progress.total), progress.total));
                        if ui.button("Cancel").clicked() {
                            progress.cancel.store(true, Ordering::Relaxed);
                        }
                    }
                    None => {
                        ui.label("Rendering...");
                    }
                }
            }
        });
        if let Some(bars) = &self.bars {
            Self::render_bars_ui(ui, bars);
            ui.add_space(4.0);
        }
        egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
            for (index, message) in self.log.iter().enumerate().filter(|(_, message)| LogLevel::of(message) >= self.log_level) {
                let time = self.log_times.get(index).map_or_else(String::new, |time| time.format("%H:%M:%S ").to_string());
                ui.add(egui::Label::new(format!("{time}{message}")).wrap());
            }
        });
    }
}

//! The status side of the window: the current view, the status log and the render progress bar, and the layout of the side panel.

use std::sync::atomic::Ordering;

use eframe::egui::{self, Color32};

use crate::complex::Complex;
use super::render_job::RenderBars;
use super::MandelbrotApp;


/// The texts of the current view: the center and the zoom, formatted for reading, without losing the precision of the numbers
#[derive(Debug, PartialEq, Eq)]
pub struct CurrentView {
    /// The real part of the center, all digits (the shortest text that reads back as the same number)
    pub real: String,
    /// The imaginary part of the center, all digits
    pub imaginary: String,
    /// The zoom: the magnification relative to the starting view, `1 / scale`, see `format_zoom`. At scale 1, the view is 2.5 units wide on the real axis.
    pub zoom: String,
}

impl CurrentView {
    /// The texts for the view with the center `center` and the scale `scale`
    pub fn new(center: &Complex, scale: f64) -> CurrentView {
        CurrentView { real: format!("{}", center.x), imaginary: format!("{}", center.y), zoom: format_zoom(1.0 / scale) }
    }
}

/// The zoom as text, "x" and the magnification: ordinary decimal numbers from 0.001 up to a billion (at most two decimals, none for a whole number, thousands separated by commas:
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
        egui::Panel::bottom("current_view_panel").resizable(false).show(ui, |ui| self.current_view_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let enabled = self.controls_enabled();
                ui.add_enabled_ui(enabled, |ui| self.controls_ui(ui));
            });
        });
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
            ui.label("Zoom:").on_hover_text("The magnification relative to the starting view: 1 / scale. At scale 1 the view is 2.5 units wide on the real axis.");
            value(ui, &view.zoom);
            ui.end_row();
        });
    }

    /// Draws the status: the render that is running with its Cancel button, the progress bar of the last render, and the scrollable log
    pub(super) fn log_ui(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Status");
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
            for message in &self.log {
                ui.add(egui::Label::new(message).wrap());
            }
        });
    }
}

//! The controls of the window: moving, zooming, the places, the iterations, the quality, the colors and saving.


use eframe::egui::{self};

use super::flight::{FlightRendering, Pace, Route};
use crate::auto_iterations::AutoPreset;
use crate::coloring::ColorChannelMapping;
use crate::resolution;
use crate::complex_plane::Precision;
use super::save_size::SaveSizeChoice;
use super::{MandelbrotApp, COLORING_FUNCTIONS, COLOR_CHANNEL_MAPPINGS};


impl MandelbrotApp {
    /// Draws the controls: moving, zooming, the places, the iterations, the quality, the colors and saving. A control that changes the view or the settings starts a new render
    pub(super) fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        self.move_controls(ui);

        ui.separator();
        self.zoom_controls(ui);

        ui.separator();
        self.places_controls(ui);

        ui.separator();
        self.render_settings_controls(ui);

        ui.separator();
        self.color_controls(ui);

        ui.separator();
        self.save_controls(ui);

        ui.separator();
        ui.label("Click the fractal to see a point's coordinates, right-click to center the view on it.");
    }

    /// The buttons that move the view, and the size of a step
    fn move_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Move");
        egui::Grid::new("move_grid").show(ui, |ui| {
            ui.label("");
            if ui.button("Up").clicked() {
                self.translate("up", self.vars.translation_amount.into(), 0);
            }
            ui.end_row();
            if ui.button("Left").clicked() {
                self.translate("left", 0, -i16::from(self.vars.translation_amount));
            }
            if ui.button("Down").clicked() {
                self.translate("down", -i16::from(self.vars.translation_amount), 0);
            }
            if ui.button("Right").clicked() {
                self.translate("right", 0, self.vars.translation_amount.into());
            }
            ui.end_row();
        });
        ui.horizontal(|ui| {
            ui.label("Move step (pixels):");
            ui.add(egui::DragValue::new(&mut self.vars.translation_amount).range(1..=255));
        });
    }

    /// The buttons that zoom, reset the view and set the zoom speed
    fn zoom_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let exhausted = self.c.precision() == Precision::Exhausted;
            let zoom_in = ui.add_enabled(!exhausted, egui::Button::new("Zoom in"));
            if exhausted {
                zoom_in.clone().on_disabled_hover_text("The limit of 64-bit numbers is reached: deeper views would only show rounding noise. Zoom out, or go to another place.");
            }
            if zoom_in.clicked() {
                self.c.scale(self.vars.scaling_factor());
                self.render("Zoomed in");
            }
            if ui.button("Zoom out").clicked() {
                self.c.scale(self.vars.inverse_scaling_factor());
                self.render("Zoomed out");
            }
            if ui.button("Reset view").clicked() {
                self.c.reset();
                self.render("Reset the view");
            }
        });
        ui.horizontal(|ui| {
            ui.label("Views:");
            if ui.add_enabled(self.history.can_go_back(), egui::Button::new("< Back")).on_hover_text("Goes back to the previous view").clicked() {
                self.go_back();
            }
            if ui.add_enabled(self.history.can_go_forward(), egui::Button::new("Forward >")).on_hover_text("Goes forward again after going back").clicked() {
                self.go_forward();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Zoom speed (1 = fastest, 9 = slowest):");
            ui.add(egui::DragValue::new(&mut self.vars.scale_numerator).range(1.0..=9.0).speed(0.1).max_decimals(0));
        });
    }

    /// The form to save the current view as a point of interest, while it is open: a name (required) and a description (optional)
    fn new_point_form(&mut self, ui: &mut egui::Ui) {
        let Some((name, description)) = &mut self.new_point else { return };
        let (mut save, mut cancel) = (false, false);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.add(egui::TextEdit::singleline(name).hint_text("Name (required)").desired_width(f32::INFINITY));
            ui.add(egui::TextEdit::multiline(description).hint_text("Description (optional)").desired_rows(2).desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                save = ui.add_enabled(!name.trim().is_empty(), egui::Button::new("Save")).on_disabled_hover_text("A point of interest needs a name").clicked();
                cancel = ui.button("Cancel").clicked();
            });
        });
        if save {
            self.save_new_point();
        } else if cancel {
            self.new_point = None;
        }
    }

    /// The points of interest and the fields to go to a view by its numbers
    fn places_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Points of interest:");
            if ui.button("Reload").on_hover_text(format!("Loads the .json files of {} again", self.points_folder.display())).clicked() {
                self.load_points_of_interest(self.points_folder.clone());
            }
            if ui.button("Save this view...").on_hover_text("Saves the current view as a new point of interest: its center, zoom, iterations, coloring and color channels").clicked() {
                self.new_point = Some((String::new(), String::new()));
            }
        });
        self.new_point_form(ui);
        let frame = self.points_ui(ui);
        self.thumbs_visible.clone_from(&frame.visible);
        let ctx = ui.ctx().clone();
        self.schedule_thumbnails(&ctx, &frame);
        if let Some(index) = frame.chosen {
            self.go_to_point_of_interest(index);
        }
        ui.checkbox(&mut self.show_points, "Show points of interest").on_hover_text("Marks the saved points of interest on the view where they are, with their names");
        ui.checkbox(&mut self.restore_appearance, "Restore the saved appearance (coloring, color channels)").on_hover_text("Off: going to a place keeps your current appearance. The supersampling is never part of a place.");
        ui.horizontal(|ui| {
            ui.label("Flight:");
            egui::ComboBox::from_id_salt("flight_route").selected_text(self.route.name()).show_ui(ui, |ui| {
                for route in Route::ALL {
                    ui.selectable_value(&mut self.route, route, route.name()).on_hover_text(match route {
                        Route::ViaReset => "First out to the whole set, then in to the place",
                        Route::Shortest => "Straight to the place, zooming out only as far as the distance needs",
                    });
                }
            }).response.on_hover_text("The way to a point of interest");
            egui::ComboBox::from_id_salt("flight_pace").selected_text(self.pace.name()).show_ui(ui, |ui| {
                for pace in Pace::ALL {
                    ui.selectable_value(&mut self.pace, pace, pace.name());
                }
            }).response.on_hover_text("How fast and how smoothly the view moves to a point of interest; using any control or the mouse stops a flight");
            egui::ComboBox::from_id_salt("flight_rendering").selected_text(self.flight_rendering.name()).show_ui(ui, |ui| {
                for mode in FlightRendering::ALL {
                    ui.selectable_value(&mut self.flight_rendering, mode, mode.name()).on_hover_text(match mode {
                        FlightRendering::BestEffort => "The camera follows the clock; the picture fills in as the render goes",
                        FlightRendering::LowRes => "The camera waits for a complete low-resolution frame of every view on the way",
                        FlightRendering::MaxQuality => "The camera waits for a full-resolution frame of every view on the way: sharp, but it can crawl where the view is slow",
                        FlightRendering::SuperSampled => "Every view on the way is rendered with all its supersampling layers, and the camera waits for the finished frame: the best picture, the slowest crawl",
                    });
                }
            }).response.on_hover_text("How complete every frame of a flight is before the camera moves on");
        });
        if let Some(flight) = &self.flight {
            //The progress along the path, and the buttons to stop the flight or to be there at once
            let share = flight.flight.share_of_path(flight.elapsed) as f32;
            let text = format!("To {}: {:.0}% of the way", flight.point.name, share * 100.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").on_hover_text("Stops the flight here and gives you the controls back").clicked() {
                    self.cancel_flight();
                } else if ui.button("Skip").on_hover_text("Goes to the place at once").clicked() {
                    self.skip_flight();
                }
            });
            ui.add(egui::ProgressBar::new(share).text(text));
        }
        ui.horizontal(|ui| {
            for (input, hint) in self.view_inputs.iter_mut().zip(["x", "y", "scale"]) {
                ui.add(egui::TextEdit::singleline(input).hint_text(hint).desired_width(80.0));
            }
            if ui.button("Go").clicked() {
                self.go_to_view();
            }
        });
    }

    /// The maximum iterations (manual or automatic, with its presets), adaptive supersampling and the quality of the window
    fn render_settings_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let mut follow = self.fit.enabled;
            if ui.checkbox(&mut follow, "Match the window size").on_hover_text("The image is rendered with one pixel for every pixel of the area it is shown in, and again when the window is resized. The view keeps its center and zoom. Off: the image keeps its size and is scaled to the window.").changed() {
                self.fit.set_enabled(follow);
            }
            ui.label(format!("({})", self.view_resolution()));
        });
        ui.horizontal(|ui| {
            ui.label("Max. iterations:");
            let response = ui.add_enabled(!self.auto_iterations, egui::DragValue::new(&mut self.max_iterations_input).range(1..=1_000_000).speed(10.0));
            if (response.drag_stopped() || response.lost_focus()) && self.max_iterations_input != self.m.max_iterations {
                self.m.max_iterations = self.max_iterations_input;
                self.render(&format!("Set max. iterations to {}", self.m.max_iterations));
            }
        });
        let mut preset_selected: Option<AutoPreset> = None;
        let mut auto_changed = false;
        ui.horizontal(|ui| {
            auto_changed = ui.checkbox(&mut self.auto_iterations, "Auto iterations").on_hover_text("Chooses the max. iterations for the zoom of every render: a starting limit that grows with the magnification, raised while the extra iterations make more pixels escape. A heuristic, not a proof: very narrow detail can be missed.").changed();
            egui::ComboBox::from_id_salt("auto_preset").selected_text(self.auto_preset.name()).show_ui(ui, |ui| {
                for preset in AutoPreset::SELECTABLE {
                    if ui.selectable_label(self.auto_preset == preset, preset.name()).on_hover_text(match preset {
                        AutoPreset::Performance => "Lower limits, coarser probes: faster, may leave deep boundary detail unresolved",
                        AutoPreset::Balanced => "The default",
                        _ => "Higher limits, finer probes: resolves more structure, can be much slower",
                    }).clicked() && self.auto_preset != preset {
                        preset_selected = Some(preset);
                    }
                }
            });
        });
        if let Some(preset) = preset_selected {
            self.select_preset(preset);
        }
        if auto_changed {
            if self.auto_iterations {
                self.render("Turned auto iterations on");
            } else {
                self.render(&format!("Turned auto iterations off, max. iterations {}", self.max_iterations_input));
            }
        }
        match &self.gpu {
            Some(gpu) => {
                if ui.checkbox(&mut self.use_gpu, "Use the GPU").on_hover_text(format!("{} helps with the slow parts of a render. It computes bit for bit the same samples as the CPU (checked when the window opened), so the image is the same.", gpu.name())).changed() {
                    self.log(format!("Turned the GPU {}", if self.use_gpu { "on" } else { "off" }));
                }
            }
            None => {
                let text = if self.gpu_opening.is_some() { "Opening the GPU..." } else { "No GPU (see the status log)" };
                ui.add_enabled(false, egui::Checkbox::new(&mut false, text));
            }
        }
        ui.checkbox(&mut self.immersion, "Immersion").on_hover_text("Never shows black where nothing is rendered yet: while you move the view, and during flights, the view only moves once the new view has at least a complete low-resolution frame. Off: the view follows at once and fills in as the render goes (best effort).");
        if ui.checkbox(&mut self.adaptive_supersampling, "Adaptive supersampling").on_hover_text("Takes the first 2 samples of every pixel, and the remaining samples only for boundary and detail pixels (and their neighbors). Faster on smooth views; a heuristic that can differ from full supersampling. Saved images always use the full saved image quality.").changed() {
            let state = if self.adaptive_supersampling { "on" } else { "off" };
            self.render(&format!("Turned adaptive supersampling {state}"));
        }
        if self.auto_iterations {
            ui.label(format!("Current limit: {} iterations ({})", self.m.max_iterations, if self.limit_pending { "starting value, being refined..." } else { "chosen automatically" }));
            ui.horizontal(|ui| {
                ui.label("Start at zoom 1:");
                let a = ui.add(egui::DragValue::new(&mut self.auto_settings.start).range(10..=100_000).speed(10.0));
                ui.label("+ per zoom doubling:");
                let b = ui.add(egui::DragValue::new(&mut self.auto_settings.per_doubling).range(0..=10_000).speed(5.0));
                ui.label("up to:");
                let c = ui.add(egui::DragValue::new(&mut self.auto_settings.ceiling).range(100..=200_000).speed(100.0));
                if [a, b, c].iter().any(|r| r.drag_stopped() || r.lost_focus()) {
                    self.auto_preset = if self.auto_settings.start == self.auto_preset.settings().start && self.auto_settings.per_doubling == self.auto_preset.settings().per_doubling && self.auto_settings.ceiling == self.auto_preset.settings().ceiling { self.auto_preset } else { AutoPreset::Custom };
                    self.render("Changed the auto iterations settings");
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Quality (supersampling, 1x-64x):");
            let response = ui.add(egui::DragValue::new(&mut self.supersampling_input).range(1..=64).suffix("x"));
            if (response.drag_stopped() || response.lost_focus()) && self.supersampling_input != self.supersampling_amount {
                self.supersampling_amount = self.supersampling_input;
                self.image_supersampling_amount = self.supersampling_input;
                self.render(&format!("Set the supersampling to {}x", self.supersampling_amount));
            }
        });
    }

    /// The coloring and the color channel mapping
    fn color_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Coloring:");
            egui::ComboBox::from_id_salt("coloring").selected_text(COLORING_FUNCTIONS[self.coloring_index].0).show_ui(ui, |ui| {
                for (index, (name, function)) in COLORING_FUNCTIONS.into_iter().enumerate() {
                    if ui.selectable_label(index == self.coloring_index, name).clicked() && index != self.coloring_index {
                        self.coloring_index = index;
                        self.coloring_function = function;
                        self.render(&format!("Set the coloring to {name}"));
                    }
                }
            });
        });
        ui.horizontal(|ui| {
            ui.label("Color channels (R, G, B):");
            let current = format!("{:?}", self.p.color_channel_mapping);
            egui::ComboBox::from_id_salt("channels").selected_text(&current).show_ui(ui, |ui| {
                for name in COLOR_CHANNEL_MAPPINGS {
                    if ui.selectable_label(name == current, name).clicked() && name != current {
                        if let Ok(mapping) = ColorChannelMapping::new(name) {
                            self.p.color_channel_mapping = mapping;
                            if self.progress.is_some() {
                                self.render(&format!("Set the color channel mapping to {name}")); //The running render still uses the old mapping
                            } else {
                                self.p.update_pixels();
                                self.texture_outdated = true;
                                self.log(format!("Set the color channel mapping to {name}"));
                            }
                        }
                    }
                }
            });
        });
    }

    /// Saving the image, with the quality of the saved image
    fn save_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Save Image");
        ui.horizontal(|ui| {
            ui.label("Saved image quality:");
            ui.add(egui::DragValue::new(&mut self.image_supersampling_amount).range(1..=64).suffix("x"));
        });
        self.save_size_controls(ui);
        if ui.button("Save image").clicked() {
            self.save_image();
        }
    }

    /// The list with the size of the saved image, the fields of a custom size and the buttons that keep it as a preset
    fn save_size_controls(&mut self, ui: &mut egui::Ui) {
        let (default_size, view) = (self.default_save_resolution(), self.view_resolution());
        ui.horizontal(|ui| {
            ui.label("Saved image size:");
            egui::ComboBox::from_id_salt("save_size").selected_text(self.save_size.selected_text(default_size, view)).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.save_size.choice, SaveSizeChoice::Standard, format!("Default ({default_size})"));
                ui.selectable_value(&mut self.save_size.choice, SaveSizeChoice::View, format!("Same as the view ({view})"));
                ui.separator();
                for preset in self.save_size.all_presets() {
                    let custom = self.save_size.is_custom_preset(&preset.name);
                    let label = if custom { format!("{} (custom)", preset.label()) } else { preset.label() };
                    ui.selectable_value(&mut self.save_size.choice, SaveSizeChoice::Preset(preset.name), label);
                }
                ui.separator();
                ui.selectable_value(&mut self.save_size.choice, SaveSizeChoice::Custom, "Custom size...");
            });
        });
        match self.save_size.resolve(default_size, view) {
            Ok(resolution) => ui.label(format!("{:.1} megapixels", resolution.pixels() as f64 / 1e6)),
            Err(reason) => ui.colored_label(egui::Color32::from_rgb(230, 70, 60), reason),
        };
        if self.save_size.choice == SaveSizeChoice::Custom {
            ui.horizontal(|ui| {
                ui.label("Width:");
                ui.add(egui::DragValue::new(&mut self.save_size.custom_width).range(resolution::MIN_SIDE..=resolution::MAX_SIDE));
                ui.label("Height:");
                ui.add(egui::DragValue::new(&mut self.save_size.custom_height).range(resolution::MIN_SIDE..=resolution::MAX_SIDE));
            });
            ui.horizontal(|ui| {
                ui.label("Preset name:");
                ui.add(egui::TextEdit::singleline(&mut self.save_size.preset_name).desired_width(120.0));
                if ui.button("Save as preset").on_hover_text("Keeps this size in the list, also the next time the program starts").clicked() {
                    match self.save_size.add_preset() {
                        Ok(message) => self.log(message),
                        Err(reason) => self.log(format!("Could not save the preset: {reason}")),
                    }
                }
            });
        }
        if let SaveSizeChoice::Preset(name) = self.save_size.choice.clone() {
            if self.save_size.is_custom_preset(&name) && ui.button("Delete this preset").clicked() {
                match self.save_size.delete_preset(&name) {
                    Ok(message) => self.log(message),
                    Err(reason) => self.log(format!("Could not delete the preset: {reason}")),
                }
            }
        }
    }
}

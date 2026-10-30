//! The controls of the window: moving, zooming, the places, the iterations, the quality, the colors and saving.


use super::blur::Quality;
use eframe::egui::{self};

use super::flight::{FlightRendering, Pace, Route, CONSTANT_SPEEDS, FLIGHT_SPEEDS};
use crate::mandelbrot_set::{MAX_DIVIDE, MIN_DIVIDE, PALETTE_LENGTH};
use crate::auto_iterations::AutoPreset;
use crate::coloring::ColorChannelMapping;
use crate::resolution;
use crate::complex_plane::Precision;
use super::save_size::SaveSizeChoice;
use super::{MandelbrotApp, COLORING_FUNCTIONS};


impl MandelbrotApp {
    /// Draws the controls: moving, zooming, the places, the iterations, the quality, the colors and saving. A control that changes the view or the settings starts a new render
    pub(super) fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Controls");
            if ui.small_button("Reset settings").on_hover_text("Puts every setting back to its default (the settings are remembered between runs otherwise, see the README)").clicked() {
                self.reset_settings();
            }
        });
        //What runs (a flight, a video) is always in sight, whichever sections are open
        self.progress_controls(ui);
        self.video_progress_ui(ui);
        //The sections: the ones used most are open at first; a click on a title opens or closes it
        self.section(ui, "Navigation", true, |app, ui| {
            app.move_controls(ui);
            ui.separator();
            app.zoom_controls(ui);
            app.go_to_view_controls(ui);
        });
        self.section(ui, "Places", true, Self::places_controls);
        self.section(ui, "Image quality", true, Self::quality_controls);
        self.section(ui, "Colors", false, Self::color_controls);
        self.section(ui, "Flights", false, Self::flight_controls);
        self.section(ui, "Performance", false, Self::performance_controls);
        self.section(ui, "Save image and video", false, |app, ui| {
            app.save_controls(ui);
            ui.separator();
            app.video_ui(ui);
        });
        self.section(ui, "Benchmark", false, Self::benchmark_ui);

        ui.separator();
        ui.label("Click the fractal to see a point's coordinates, right-click to center the view on it.");
    }

    /// A section of the controls with a title that opens and closes it (`open`: open at first; every section is open in the tests, which read the controls)
    fn section(&mut self, ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut Self, &mut egui::Ui)) {
        egui::CollapsingHeader::new(egui::RichText::new(title).strong().size(16.0)).id_salt(title).default_open(open || cfg!(test)).show(ui, |ui| add(self, ui));
    }

    /// The progress of a flight, with its buttons to stop it or to be there at once
    fn progress_controls(&mut self, ui: &mut egui::Ui) {
        if let Some(flight) = &self.flight {
            //The progress along the path, and the buttons to stop the flight or to be there at once
            let share = flight.flight.share_of_path(flight.elapsed) as f32;
            let resolution = if matches!(self.flight_rendering, FlightRendering::Auto(_)) { format!(", {:.0}% resolution", self.shown_scaled.map_or(100.0 / self.preview_block as f64, |(columns, _)| 100.0 * columns as f64 / self.p.pixel_plane.width as f64)) } else { String::new() };
            let text = format!("To {}: {:.0}% of the way, {} FPS{resolution}", flight.point.name, share * 100.0, flight.fps_shown);
            ui.horizontal(|ui| {
                if ui.button("Cancel").on_hover_text("Stops the flight here and gives you the controls back").clicked() {
                    self.cancel_flight();
                } else if ui.button("Skip").on_hover_text("Goes to the place at once").clicked() {
                    self.skip_flight();
                }
            });
            ui.add(egui::ProgressBar::new(share).text(text));
        }
    }

    /// Going to a view by its center and scale
    fn go_to_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (input, hint) in self.view_inputs.iter_mut().zip(["x", "y", "scale"]) {
                ui.add(egui::TextEdit::singleline(input).hint_text(hint).desired_width(80.0));
            }
            if ui.button("Go").clicked() {
                self.go_to_view();
            }
        });
    }

    /// How flights fly: the pace, the route and the rendering of their frames, frames ahead, and blending and blurring their frames
    fn flight_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Flight:");
            egui::ComboBox::from_id_salt("flight_pace").selected_text(self.pace.name()).show_ui(ui, |ui| {
                for pace in Pace::ALL {
                    let response = ui.selectable_value(&mut self.pace, pace, pace.name());
                    if pace == Pace::NoFlight {
                        response.on_hover_text("Clicking a place shows it at once, without flying there");
                    }
                }
            }).response.on_hover_text("How fast and how smoothly the view moves to a point of interest: Fast, Normal and Cinematic start and end softly; Constant keeps one speed all the way, set by the slider; using any control or the mouse stops a flight");
            if !matches!(self.pace, Pace::Constant | Pace::NoFlight) {
                ui.add(egui::Slider::new(&mut self.flight_speed, FLIGHT_SPEEDS).logarithmic(true).suffix("%")).on_hover_text("The speed of the pace, in percent (10% to 400%): slower or faster, with the same soft start and arrival; it takes effect at once, also during a flight");
            }
            if self.pace == Pace::Constant {
                ui.add(egui::Slider::new(&mut self.constant_speed, CONSTANT_SPEEDS).logarithmic(true).max_decimals(2).suffix(" ×2/s")).on_hover_text("The speed of the constant pace, in doublings of the zoom per second (the pan moves as many views): the same speed from the start to the place; it takes effect at once, also during a flight");
            }
            egui::ComboBox::from_id_salt("flight_route").selected_text(self.route.name()).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.route, Route::Shortest, Route::Shortest.name()).on_hover_text("Zooms out to travel, only as far as the distance needs, and in again: the shortest way for the eye");
                ui.selectable_value(&mut self.route, Route::Straight, Route::Straight.name()).on_hover_text("A straight line: the zoom changes steadily from one view to the other, never zooming out on the way. Between two far-apart places at a deep zoom the picture rushes by");
                ui.selectable_value(&mut self.route, Route::Dive, Route::Dive.name()).on_hover_text("Starts right above the place at zoom 1, wherever the view was, and falls straight down into it, zooming only, without moving sideways (also for videos)");
            }).response.on_hover_text("The route of a flight");
            egui::ComboBox::from_id_salt("flight_rendering").selected_text(self.flight_rendering.name()).show_ui(ui, |ui| {
                for mode in FlightRendering::ALL {
                    ui.selectable_value(&mut self.flight_rendering, mode, mode.entry()).on_hover_text(match mode {
                        FlightRendering::BestEffort => "The camera follows the clock; the picture fills in as the render goes",
                        FlightRendering::LowRes => "The camera waits for a complete low-resolution frame of every view on the way",
                        FlightRendering::Resolution(100) => "The camera waits for a full-resolution frame of every view on the way: sharp, but it can crawl where the view is slow",
                        FlightRendering::Resolution(_) => "The camera waits for a frame of this resolution (blocks of 4 x 4 or 2 x 2 pixels): cleaner than Low res, faster than full resolution",
                        FlightRendering::SuperSampled => "Every view on the way is rendered with all its supersampling layers, and the camera waits for the finished frame: the best picture, the slowest crawl",
                        FlightRendering::Auto(_) => "The resolution of every frame is chosen to keep the frame rate next to it: full resolution where the view is fast, coarser where it is slow",
                    });
                }
            }).response.on_hover_text("How complete every frame of a flight is before the camera moves on");
            if let FlightRendering::Auto(fps) = &mut self.flight_rendering {
                ui.add(egui::DragValue::new(fps).range(1..=240).suffix(" FPS")).on_hover_text("The frame rate that Auto keeps (type a value): every frame of a flight gets the finest resolution that the frames so far say fits, from full resolution down to blocks of 8 x 8 pixels; the screen shows no more frames than it refreshes");
            }
            if let FlightRendering::Resolution(percent) = &mut self.flight_rendering {
                ui.add(egui::DragValue::new(percent).range(1..=800).suffix("%")).on_hover_text("The resolution of the frames of a flight, in percent of the pixels across (type a value): below 100 the camera waits for blocks of 100 / percent pixels (33%: 3 x 3), 100 is full resolution, above 100 it waits for supersampling layers ((percent / 100)²: 200% is 4 samples per pixel, up to the quality setting)");
            }
        });
        ui.horizontal(|ui| {
            ui.label("Frames ahead:");
            ui.add(egui::DragValue::new(&mut self.frames_ahead).range(0..=3)).on_hover_text("How many frames of a flight are rendered ahead of the one on the screen, so the CPU and the GPU never wait for the screen (0: one frame at a time, as before). For flights that wait for their frames (all renderings but Best effort)");
        });
        ui.checkbox(&mut self.blend_frames, "Blend frames during flights").on_hover_text("Flights at full resolution: every frame takes another sample of its pixels and is blended into the frames before it, moved to its view, so a moving image comes close to 4 samples per pixel for the cost of one. Edges that the camera uncovers start from the new frame; the image you arrive at is rendered normally. Off by default (a fast zoom out can show a slight trail).");
        ui.add_enabled(self.blur.is_some(), egui::Checkbox::new(&mut self.auto_blur, "Auto blur during flights")).on_hover_text("While a flight moves the view, the image gets a light blur on the GPU, which brings a moving frame closer to the finished image: stronger for coarse frames, lighter for supersampled ones (σ = 0.8 × (100 / resolution in %)^0.6 pixels). The finished image at the place is never blurred.");
    }

    /// What makes renders faster: the GPU and its options, the neighborhood, and immersion
    fn performance_controls(&mut self, ui: &mut egui::Ui) {
        match &self.gpu {
            Some(gpu) => {
                if ui.checkbox(&mut self.use_gpu, "Use the GPU").on_hover_text(format!("{} helps with the slow parts of a render. It computes bit for bit the same samples as the CPU (checked when the window opened), so the image is the same.", gpu.name())).changed() {
                    self.log(format!("Turned the GPU {}", if self.use_gpu { "on" } else { "off" }));
                }
                let response = ui.add_enabled(self.use_gpu, egui::Checkbox::new(&mut self.fast_gpu, "Fast GPU math"));
                if response.on_hover_text("On by default. The GPU computes with pairs of 32-bit numbers instead of 64-bit numbers: about three times as fast on slow views, but not bit for bit the CPU's. Most of the image is the same; in chaotic parts of the set some pixels get other colors (as they would with any other rounding). Only used down to a zoom of about 10⁻⁹; saved images are always exact.").changed() {
                    self.samples = None; //The samples so far were computed the other way: start again
                    self.render(&format!("Turned fast GPU math {}", if self.fast_gpu { "on" } else { "off" }));
                }
                ui.indent("zoom_limit", |ui| {
                    let response = ui.add_enabled(self.use_gpu && self.fast_gpu, egui::Checkbox::new(&mut self.limit_zoom, "Limit the zoom to fast GPU math")).on_hover_text("Nothing deeper than a scale of about 10⁻⁹ is rendered, where the fast GPU math would switch back to exact math (renders get slower from there): zooming in, Go to view, Back and Forward and flights to deeper places are refused, and anything else (a larger window) zooms out to the limit. Off: as deep as 64-bit numbers allow.");
                    if response.changed() && self.limit_zoom && self.zoom_refused(&self.c.clone()) {
                        self.render("Zoomed out to the zoom limit");
                    }
                });
                ui.indent("f32_frames", |ui| {
                    ui.add_enabled(self.use_gpu && self.fast_gpu, egui::Checkbox::new(&mut self.f32_frames, "Faster moving frames (f32)")).on_hover_text("Frames of flights and drags at shallow zoom (wider than a scale of about 0.01 at 4K) are computed by the GPU in plain 32-bit numbers: several times as fast as the fast GPU math, but in chaotic parts of the set more pixels get other colors while the view moves. The image you stop at is always rendered with the fast or exact math. On by default.");
                });
            }
            None => {
                let text = if self.gpu_opening.is_some() { "Opening the GPU..." } else { "No GPU (see the status log)" };
                ui.add_enabled(false, egui::Checkbox::new(&mut false, text));
            }
        }
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.render_neighborhood, "Render the neighborhood").on_hover_text("Once the view is finished and nothing else runs, the area around it (by default one view to every side, at half the resolution) is rendered too and drawn behind the image, so dragging, panning and zooming out show it at once instead of black").changed() && !self.render_neighborhood {
                self.cancel_neighborhood_job();
                self.neighborhood = None;
            }
            ui.add_enabled(self.render_neighborhood, egui::DragValue::new(&mut self.neighborhood_margin).range(10..=5000).speed(5).suffix("% per side"))
                .on_hover_text("How far the neighborhood reaches past every side, in percent of the view (type a value; 100% is one view to every side). A large neighborhood takes longer and gets a lower resolution, so it stays within 8192 pixels on a side");
        });
        ui.checkbox(&mut self.immersion, "Immersion").on_hover_text("Never shows black where nothing is rendered yet: while you move the view, and during flights, the view only moves once the new view has at least a complete low-resolution frame. Off: the view follows at once and fills in as the render goes (best effort).");
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
                let mut zoomed = self.c.clone();
                zoomed.scale(self.vars.scaling_factor());
                if !self.zoom_refused(&zoomed) {
                    self.c = zoomed;
                    self.render("Zoomed in");
                }
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
    }

    /// The quality of the window, adaptive supersampling, the size of the image, and the maximum iterations (manual or automatic, with its presets)
    fn quality_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Quality:");
            let current = Quality { samples: self.supersampling_amount, blurred: self.quality_blur };
            let mut chosen = current;
            egui::ComboBox::from_id_salt("quality").selected_text(current.name()).show_ui(ui, |ui| {
                for preset in Quality::PRESETS.into_iter().filter(|preset| !preset.blurred || self.blur.is_some()) {
                    ui.selectable_value(&mut chosen, preset, preset.name());
                }
            }).response.on_hover_text("Samples per pixel (supersampling): more is smoother and sharper at the edges of the set, and takes longer. The blur preset takes one sample per pixel and blurs the image a little (σ 0.8 pixel), which comes close to 4x for the time of 1x.");
            if chosen != current {
                (self.supersampling_amount, self.supersampling_input, self.quality_blur) = (chosen.samples, chosen.samples, chosen.blurred);
                (self.image_supersampling_amount, self.image_quality_blur) = (chosen.samples, chosen.blurred);
                self.texture_outdated = true;
                self.render(&format!("Set the quality to {}", chosen.name()));
            }
        });
        if ui.checkbox(&mut self.adaptive_supersampling, "Adaptive supersampling").on_hover_text("Takes the first 2 samples of every pixel, and the remaining samples only for boundary and detail pixels (and their neighbors). Faster on smooth views; a heuristic that can differ from full supersampling. Saved images always use the full saved image quality.").changed() {
            let state = if self.adaptive_supersampling { "on" } else { "off" };
            self.render(&format!("Turned adaptive supersampling {state}"));
        }
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
                        self.render_moved(&format!("Set the coloring to {name}")); //Only the colors change: the first samples are recolored at once (see `SampleState::recolor`)
                    }
                }
            });
        });
        //The color cycle (see `ColorCycle`): how the iteration counts map onto the palette, the same with a manual and an automatic limit
        let before = self.m.palette;
        ui.add_enabled_ui(!self.m.palette.stretch, |ui| {
            ui.horizontal(|ui| {
                ui.label("Iteration divide:");
                //Logarithmic: 0.1, 1, 10, 100 and 1000 are equally far apart; the number next to it can be typed
                ui.add(egui::Slider::new(&mut self.m.palette.divide, MIN_DIVIDE..=MAX_DIVIDE).logarithmic(true).max_decimals(3))
                    .on_hover_text("How many iterations one step of the palette takes (0.1 to 1000; type a value in the box): below 1 the colors change several times per iteration, more gives broader, calmer bands. Deep views, whose iteration counts are high and far apart, often look best with more. A point keeps its color however far you zoom and whatever the iteration limit.");
            });
            ui.horizontal(|ui| {
                ui.label("Offset:");
                //(To 499: 500 steps are the whole palette, the same colors as 0)
                ui.add(egui::Slider::new(&mut self.m.palette.offset, 0.0..=PALETTE_LENGTH - 1.0).step_by(1.0).max_decimals(0))
                    .on_hover_text("Where the palette starts, in steps (0 to 499 of 500): shifts every color along the palette; the colors change at once while you drag");
            });
        });
        ui.checkbox(&mut self.m.palette.stretch, "Stretch the palette over the limit")
            .on_hover_text("On: the palette spans the iteration counts once, from 0 to the limit, without repeating (the colors change with the limit). Off: the palette repeats every 500 steps, see Iteration divide.");
        if self.m.palette != before {
            self.m.palette.offset = self.m.palette.offset.rem_euclid(PALETTE_LENGTH);
            self.render_moved(&format!("Set the colors to {}", super::points::palette_description(self.m.palette))); //Recolored at once, see `SampleState::recolor`
        }
        if ui.checkbox(&mut self.m.smooth_coloring, "Smooth colors").on_hover_text("Colors flow smoothly from one iteration count to the next (the fractional iteration count). Off: every pixel gets the color of its whole iteration count, which shows bands of color and saves two logarithms per pixel outside the set. Saved images use this setting too.").changed() {
            self.render(&format!("Turned smooth colors {}", if self.m.smooth_coloring { "on" } else { "off" }));
        }
        //One drop-down per channel of the screen: the channel of the coloring it shows, so every combination can be made
        let mut chosen = None;
        ui.horizontal(|ui| {
            ui.label("Color channels:").on_hover_text("Which channel of the coloring every channel of the screen shows: red, green and blue as they are (R ← Red, G ← Green, B ← Blue), swapped, or the same channel twice");
            let current = format!("{:?}", self.p.color_channel_mapping);
            for (index, (screen, letter)) in [("R", 'R'), ("G", 'G'), ("B", 'B')].into_iter().enumerate() {
                let source = current.chars().nth(index).unwrap_or(letter);
                ui.label(format!("{screen} ←"));
                egui::ComboBox::from_id_salt(("channel", index)).width(60.0).selected_text(channel_name(source)).show_ui(ui, |ui| {
                    for option in ['R', 'G', 'B'] {
                        if ui.selectable_label(option == source, channel_name(option)).clicked() && option != source {
                            let mut name: Vec<char> = current.chars().collect();
                            name[index] = option;
                            chosen = Some(name.into_iter().collect::<String>());
                        }
                    }
                });
            }
        });
        if let Some(name) = chosen {
            if let Ok(mapping) = ColorChannelMapping::new(&name) {
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

    /// Saving the image, with the quality of the saved image
    fn save_controls(&mut self, ui: &mut egui::Ui) {
        ui.strong("Image");
        ui.horizontal(|ui| {
            ui.label("Saved image quality:");
            let current = Quality { samples: self.image_supersampling_amount, blurred: self.image_quality_blur };
            let mut chosen = current;
            egui::ComboBox::from_id_salt("saved_quality").selected_text(current.name()).show_ui(ui, |ui| {
                for preset in Quality::PRESETS {
                    ui.selectable_value(&mut chosen, preset, preset.name());
                }
            });
            (self.image_supersampling_amount, self.image_quality_blur) = (chosen.samples, chosen.blurred);
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

/// The name of a channel in the drop-downs of the color channels, from its letter
fn channel_name(letter: char) -> &'static str {
    match letter {
        'R' => "Red",
        'G' => "Green",
        _ => "Blue",
    }
}

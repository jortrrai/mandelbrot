//! What the buttons and clicks do: moving, jumping to a view, choosing a preset and saving the image.



use crate::auto_iterations::AutoPreset;
use crate::complex_plane::{ComplexPlane, View};
use crate::pixel_buffer::{pixel_plane::PixelPlane, PixelBuffer};
use crate::rendering::{self};
use crate::resolution::Resolution;
use super::MandelbrotApp;


impl MandelbrotApp {
    /// Selects an auto iterations preset. With auto iterations on, the limit of the preset is published immediately (see `start_render`) and the render with it starts in the background.
    pub(super) fn select_preset(&mut self, preset: AutoPreset) {
        self.auto_preset = preset;
        self.auto_settings = preset.settings();
        if self.auto_iterations {
            self.render(&format!("Set the auto iterations preset to {}", preset.name()));
        } else {
            self.log(format!("Set the auto iterations preset to {}", preset.name()));
        }
    }

    /// Moves the view by `rows_up` rows and `columns_right` columns of pixels (`name` is the direction, for the status log), and renders the exposed strip
    pub(super) fn translate(&mut self, name: &str, rows_up: i16, columns_right: i16) {
        rendering::move_complex_plane_by_pixels(&mut self.c, i64::from(rows_up), i64::from(columns_right));
        //Moving the view to the right moves the content of the buffer to the left
        self.render_moved(&format!("Moved {name} {} pixels", self.vars.translation_amount));
    }

    /// Saves the current view as an image on its own background thread. This does not interrupt a running render;
    /// while one runs, the buffer only holds a partially refined image, so the image is rendered again for saving.
    pub(super) fn save_image(&mut self) {
        if self.save_deep_image() {
            return; //A view deeper than 64-bit floats is saved by the deep renderer (see `deep_view`)
        }
        let resolution = match self.save_size.resolve(self.default_save_resolution(), self.view_resolution()) {
            Ok(resolution) => resolution,
            Err(reason) => {
                self.log(format!("Could not save the image: {reason}"));
                return;
            }
        };
        let time_stamp = chrono::Utc::now().to_string();
        let file_name = time_stamp.replace(':', "-").replace(' ', "_");
        let (c, p, m) = (self.c.clone(), self.p.clone(), self.m.clone());
        let (image_supersampling_amount, coloring_function, blur) = (self.image_supersampling_amount, self.coloring_function, self.image_quality_blur);
        let buffer_is_complete = self.buffer_is_complete && self.progress.is_none() && !self.adaptive_supersampling; //An interrupted, cancelled or adaptively sampled buffer is not a full-quality image of the saved quality
        let sender = self.notice_sender.clone();
        let gpu = self.gpu.clone().filter(|_| self.use_gpu);
        //The view in the image's metadata, so it can be opened again (see `image_metadata`)
        let place = crate::points_of_interest::to_json(&self.current_point(file_name.clone(), String::new()));
        std::thread::spawn(move || {
            let view = c.get_view();
            let (view_width, view_height) = (p.pixel_plane.width, p.pixel_plane.height);
            let same_size_as_view = resolution == Resolution::new(view_width, view_height); //Only then is the image in the buffer the image that is saved
            let mut image_p = if same_size_as_view && buffer_is_complete {
                p
            } else {
                let mut image_p = PixelBuffer::new(PixelPlane::new(resolution.width, resolution.height));
                let mut image_c = ComplexPlane::new(resolution.width, resolution.height);
                image_p.color_channel_mapping = p.color_channel_mapping;
                image_c.set_view(&view);
                rendering::render_complex_plane_into_buffer_with_gpu(&mut image_p, &image_c, &m, image_supersampling_amount, coloring_function, gpu.as_deref());
                image_p
            };
            if blur {
                //The blur preset: the blur of a frame of this many samples, as on the screen (see `blur::blur_sigma`)
                let (width, height) = (image_p.pixel_plane.width, image_p.pixel_plane.height);
                image_p.colors = super::blur::blurred(&image_p.colors, width, height, super::blur::blur_sigma(1, image_supersampling_amount));
                image_p.update_pixels();
            }
            let saved = image_p.save_as_png(&time_stamp, &view, &m, image_supersampling_amount, Some(&place));
            let message = match saved {
                Ok(()) => format!("Saved the image to saved/{file_name}.png"),
                Err(err) => format!("Could not save the image to saved/{file_name}.png: {err}"),
            };
            let _ = sender.send(message);
        });
        self.log("Saving the image...");
    }

    /// Goes back to the view that was shown before this one, if there is one
    pub(super) fn go_back(&mut self) {
        if let Some(plane) = self.history.go_back() {
            if !self.show_remembered_view(&plane, "Went back to the previous view") {
                self.history.go_forward(); //Beyond the zoom limit: the history stays where it was
            }
        }
    }

    /// Goes forward again to the view that was left with Back, if there is one
    pub(super) fn go_forward(&mut self) {
        if let Some(plane) = self.history.go_forward() {
            if !self.show_remembered_view(&plane, "Went forward to the next view") {
                self.history.go_back();
            }
        }
    }

    /// Shows a view from the history at the size of the window now: the same center and width on the real axis. Returns false when the view is beyond the zoom limit
    /// (see `zoom_refused`): then it is not shown
    fn show_remembered_view(&mut self, plane: &ComplexPlane, message: &str) -> bool {
        let resized = plane.resized(self.p.pixel_plane.width, self.p.pixel_plane.height);
        if self.zoom_refused(&resized) {
            return false;
        }
        self.c = resized;
        self.render(message);
        true
    }

    /// The default size of a saved image, see `Config::image_width`
    pub(super) fn default_save_resolution(&self) -> Resolution {
        Resolution::new(self.config.image_width, self.config.image_height)
    }

    /// The size of the view that is shown in the window
    pub(super) fn view_resolution(&self) -> Resolution {
        Resolution::new(self.p.pixel_plane.width, self.p.pixel_plane.height)
    }

    /// Goes to the view that was typed in the three "go to view" fields (x, y and scale; an empty scale keeps the current scale); a message in the status log if they
    /// are not numbers or the scale is not positive
    pub(super) fn go_to_view(&mut self) {
        let current_scale = self.c.get_scale().to_string();
        let parsed: Vec<Result<f64, _>> = self.view_inputs.iter().enumerate().map(|(index, text)| if index == 2 && text.trim().is_empty() { current_scale.as_str() } else { text.trim() }.parse::<f64>()).collect();
        match parsed.as_slice() {
            [Ok(x), Ok(y), Ok(scale)] if *scale > 0.0 => {
                let mut entered = self.c.clone();
                entered.set_view(&View::new(*x, *y, *scale));
                if !self.zoom_refused(&entered) {
                    self.c = entered;
                    self.render("Moved to the entered view");
                }
            }
            _ => self.log("Could not go to the view: x, y and scale should be numbers (an empty scale keeps the current one), and scale should be positive"),
        }
    }

    /// Handles a click on the fractal at `(x, y)` in pixel buffer coordinates
    pub(super) fn handle_image_click(&mut self, x: f32, y: f32, centering: bool) {
        //The point under the mouse in the view on the screen (with immersion, that can be the last complete frame while the view moves on)
        let complex = self.displayed_plane().complex_from_pixel_plane(x.into(), y.into());
        if centering {
            //Moves by whole pixels, so the samples of the pixels that stay visible are kept: the clicked point ends up within half a pixel of the center
            let center = self.c.center();
            let columns_right = ((complex.x - center.x) / self.c.increment_x).round() as i64;
            let rows_up = ((complex.y - center.y) / self.c.increment_y).round() as i64;
            rendering::move_complex_plane_by_pixels(&mut self.c, rows_up, columns_right);
            self.render_moved(&format!("Centered the view on ({:.10}, {:.10})", complex.x, complex.y));
        } else {
            self.log(format!("Clicked pixel ({}, {}) = complex point ({:.10}, {:.10})", x as usize, y as usize, complex.x, complex.y));
        }
    }
}

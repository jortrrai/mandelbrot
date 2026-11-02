//! Sharing a view (issues #183 and #184): *Copy view* puts the current view on the clipboard as one line of text (a point of interest file, see
//! `points_of_interest`: the center, the scale, the iterations and the appearance), for a video description or a message; pasting such a text, or the
//! text of a whole place file, goes there with its appearance. An image saved by this program (or a frame of a video with PNG frames) opened with *Open image*
//! goes to its view too, read from its metadata (see `image_metadata`); older images have their view and iterations in theirs. Dropping an image on the
//! window does the same where the system passes dropped files on (not on Wayland, where winit 0.30 never delivers them).

use eframe::egui;

use super::MandelbrotApp;
use crate::points_of_interest::{self, Coloring, PointOfInterest};

/// The folder that *Open image* starts in: the user's home folder, if it is known. (Not the relative folder `saved`: the desktop portal of Linux only takes an
/// absolute folder that exists, and Jort's file chooser refused to open with it)
#[cfg_attr(target_os = "android", allow(dead_code))]
pub(super) fn home_folder() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(std::path::PathBuf::from).filter(|folder| folder.is_absolute() && folder.is_dir())
}

/// The text of `point` on one line
pub(super) fn one_line(point: &PointOfInterest) -> String {
    let file: serde_json::Value = serde_json::from_str(&points_of_interest::to_json(point)).expect("a place file is JSON");
    file.to_string()
}

impl MandelbrotApp {
    /// Puts the current view on the clipboard as one line of text (see the module)
    pub(super) fn copy_view(&mut self, ctx: &egui::Context) {
        let text = one_line(&self.current_point(String::from("Shared view"), String::new()));
        ctx.copy_text(text.clone());
        self.log(format!("Copied the view to the clipboard: {text}"));
    }

    /// Goes to the view of `text`: a view copied with *Copy view* or the text of a place file, or the path of an image saved by this program
    pub(super) fn go_to_shared(&mut self, text: &str) {
        let text = text.trim();
        let point = if text.starts_with('{') {
            points_of_interest::parse(text, "the pasted view").map(|(point, _)| point)
        } else {
            crate::image_metadata::view_of_png(std::path::Path::new(text.trim_matches('"')))
        };
        match point {
            Ok(point) => self.go_to_shared_point(point),
            Err(reason) => self.log(format!("Could not go to the view: {reason}")),
        }
    }

    /// Goes to the view of the image at `path` (dropped on the window)
    pub(super) fn open_image(&mut self, path: &std::path::Path) {
        match crate::image_metadata::view_of_png(path) {
            Ok(point) => self.go_to_shared_point(point),
            Err(reason) => self.log(format!("Could not open the view of the image: {reason}")),
        }
    }

    /// Goes to a shared view with all of its appearance (whatever *Restore the saved appearance* says: a shared view should look as it was shared), flying
    /// there at the chosen pace
    fn go_to_shared_point(&mut self, point: PointOfInterest) {
        let mut palette = self.m.palette;
        if let Some(appearance) = &point.appearance {
            if let Some(coloring) = appearance.coloring {
                self.coloring_index = match coloring {
                    Coloring::Hsv => 0,
                    Coloring::Bernstein => 1,
                }; //See COLORING_FUNCTIONS
                self.coloring_function = super::COLORING_FUNCTIONS[self.coloring_index].1;
            }
            if let Some(mapping) = appearance.color_channels {
                self.p.color_channel_mapping = mapping;
            }
            palette = appearance.palette_over(palette);
        }
        self.log(format!("Going to the view {}", point.name));
        self.max_iterations_input = point.location.max_iterations;
        self.fly_to(point, palette);
    }

    /// Opens the images dropped on the window (see `open_image`); an audio file becomes the music of videos (see `music`)
    pub(super) fn open_dropped_images(&mut self, ctx: &egui::Context) {
        let dropped: Vec<std::path::PathBuf> = ctx.input(|input| input.raw.dropped_files.iter().map(|file| file.path().to_path_buf()).filter(|path| !path.as_os_str().is_empty()).collect());
        if let Some(path) = dropped.last() {
            if super::music::is_audio(path) {
                self.use_music_file(path);
            } else {
                self.open_image(path);
            }
        }
    }

    /// *Copy view*, and the field to paste a view (or the path of an image) into
    pub(super) fn sharing_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            //(Not in the Android app, which has no file chooser of rfd: views go there as text, with Copy view and Go to it)
            #[cfg(not(target_os = "android"))]
            if ui.button("Open image...").on_hover_text("Goes to the view of an image saved by this program (or a PNG frame of one of its videos), with its colors: its view is in the image").clicked() {
                //(The system's file chooser, modal: the window waits for it; renders go on in the background)
                let mut dialog = rfd::FileDialog::new().set_title("Open an image saved by Mandelbrot").add_filter("PNG image", &["png"]);
                if let Some(home) = home_folder() {
                    dialog = dialog.set_directory(home);
                }
                if let Some(path) = dialog.pick_file() {
                    self.open_image(&path);
                }
            }
            if ui.button("Copy view").on_hover_text("Copies the current view (center, zoom, iterations, colors) to the clipboard as one line of text, to share it, for example in a video description; pasting it below goes there").clicked() {
                self.copy_view(ui.ctx());
            }
            let response = ui.add(egui::TextEdit::singleline(&mut self.shared_view_input).desired_width(180.0).hint_text("paste a view or an image path"))
                .on_hover_text("A view copied with Copy view, the text of a place file, or the path of an image saved by this program");
            let entered = response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (ui.add_enabled(!self.shared_view_input.trim().is_empty(), egui::Button::new("Go to it")).clicked() || entered) && !self.shared_view_input.trim().is_empty() {
                let text = std::mem::take(&mut self.shared_view_input);
                self.go_to_shared(&text);
            }
        });
    }
}

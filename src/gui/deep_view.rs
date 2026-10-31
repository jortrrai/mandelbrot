//! Deep zoom in the window (issue #141, prototype): views deeper than 64-bit floats can show.
//!
//! Where the coordinates of neighboring pixels run out of 64-bit precision (`Precision::Exhausted`), zooming in further switches to the deep view: its center
//! is kept exactly in fixed point (`rendering::deep::Fixed`), and its images are rendered by perturbation with rebasing (`rendering::deep`). The wheel zooms
//! around the cursor and a drag pans as always, by exact steps. Zooming back out to where 64-bit floats suffice, or going anywhere else (a place, the
//! history, a flight, typed coordinates), leaves it.
//!
//! The complex plane of the window stays at the deepest view 64-bit floats can show, at the 64-bit center nearest the exact one, so everything that reads it
//! keeps working; the deep view knows the real scale. A deep view renders the samples of the quality layer by layer, keeps its exact center in saved places,
//! *Copy view* and saved images, and going to such a place enters it again. Not yet in a deep view: adaptive supersampling, the GPU, flights and videos.

use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};
use std::time::Instant;

use super::render_job::{JobMessage, Progress};
use super::MandelbrotApp;
use crate::complex_plane::{Precision, View};
use crate::rendering::deep::{self, Fixed};

/// The exact view deeper than 64-bit floats: its center, its scale, and the view of the complex plane it left (to tell when something else moved it)
pub(super) struct DeepView {
    pub(super) real: Fixed,
    pub(super) imaginary: Fixed,
    pub(super) scale: f64,
    plane: String,
}

impl MandelbrotApp {
    /// Whether a view of `scale` around the current center is too deep for 64-bit floats
    fn too_deep(&self, scale: f64) -> bool {
        let center = self.c.center();
        let mut plane = self.c.clone();
        plane.set_view(&View::new(center.x, center.y, scale));
        plane.precision() == Precision::Exhausted
    }

    /// The deep view, if the window shows one: it is left as soon as anything else moved the complex plane
    pub(super) fn deep_view(&mut self) -> Option<&DeepView> {
        if self.deep.as_ref().is_some_and(|deep| deep.plane != format!("{:?}", self.c.get_view())) {
            self.deep = None;
        }
        self.deep.as_ref()
    }

    /// Sets the exact view (entering or leaving the deep view): deeper than 64-bit floats a deep view, else the complex plane alone
    fn set_deep(&mut self, real: &Fixed, imaginary: &Fixed, scale: f64) {
        let center = (real.to_f64(), imaginary.to_f64());
        if !self.too_deep(scale) {
            self.deep = None;
            self.c.set_view(&View::new(center.0, center.1, scale));
            return;
        }
        //The complex plane at the deepest scale that 64-bit floats can show (the scale doubles until they can)
        let mut plane_scale = scale;
        while self.too_deep(plane_scale) && plane_scale < 1.0 {
            plane_scale *= 2.0;
        }
        self.c.set_view(&View::new(center.0, center.1, plane_scale));
        //The bits the scale needs, and never fewer than the center has: zooming out and in again comes back to exactly the same point
        let bits = deep::bits_for(scale, self.p.pixel_plane.width).max(real.bits()).max(imaginary.bits());
        self.deep = Some(DeepView { real: real.with_bits(bits), imaginary: imaginary.with_bits(bits), scale, plane: format!("{:?}", self.c.get_view()) });
    }

    /// The exact center and scale: of the deep view, or of the complex plane
    fn exact_view(&mut self) -> (Fixed, Fixed, f64) {
        if let Some(deep) = self.deep_view() {
            return (deep.real.clone(), deep.imaginary.clone(), deep.scale);
        }
        let center = self.exact_center();
        let (scale, width) = (self.c.get_scale(), self.p.pixel_plane.width);
        let bits = deep::bits_for(scale * 1e-6, width); //(Room to zoom in)
        let zero = Fixed::parse("0", bits).expect("0 is a number");
        (zero.plus(center.x), zero.plus(center.y), scale)
    }

    /// Whether a zoom by `factor` (below 1: in) is handled by the deep view: zooming in where 64-bit floats run out, or any zoom in a deep view
    pub(super) fn zooms_deep(&mut self, factor: f64) -> bool {
        self.deep_view().is_some() || (factor < 1.0 && self.too_deep(self.c.get_scale() * factor))
    }

    /// Zooms by `factor` around the pixel `(x, y)` of the view (exactly: the point under the cursor stays there), in the deep view
    pub(super) fn zoom_deep(&mut self, factor: f64, (x, y): (f64, f64)) {
        let (real, imaginary, scale) = self.exact_view();
        let (width, height) = (self.p.pixel_plane.width as f64, self.p.pixel_plane.height as f64);
        let pixel = 2.5 * scale / width;
        let (anchor_x, anchor_y) = ((x - width / 2.0) * pixel, (height / 2.0 - y) * pixel);
        self.set_deep(&real.plus(anchor_x * (1.0 - factor)), &imaginary.plus(anchor_y * (1.0 - factor)), scale * factor);
    }

    /// Moves the deep view by whole pixels (as a drag does), exactly
    pub(super) fn pan_deep(&mut self, columns: i64, rows_up: i64) {
        let (real, imaginary, scale) = self.exact_view();
        let pixel = 2.5 * scale / self.p.pixel_plane.width as f64;
        self.set_deep(&real.plus(columns as f64 * pixel), &imaginary.plus(rows_up as f64 * pixel), scale);
    }

    /// Sets the zoom to `zoom` around the exact center, in the deep view when it is too deep for 64-bit floats (for the depth control)
    pub(super) fn set_deep_zoom(&mut self, zoom: f64) {
        let (real, imaginary, _) = self.exact_view();
        self.set_deep(&real, &imaginary, 1.0 / zoom);
    }

    /// The deep view if the window shows it (without leaving it when it does not, see `deep_view`)
    pub(super) fn deep_shown(&self) -> Option<&DeepView> {
        self.deep.as_ref().filter(|deep| deep.plane == format!("{:?}", self.c.get_view()))
    }

    /// The scale of the view that is shown: the deep view's, or the complex plane's
    pub(super) fn shown_scale(&self) -> f64 {
        self.deep_shown().map_or_else(|| self.c.get_scale(), |deep| deep.scale)
    }

    /// Renders the deep view on a background thread (one sample a pixel), as a render of the window: its image arrives as the one layer of the render
    pub(super) fn start_deep_render(&mut self, message: &str) {
        let Some(deep) = self.deep_view() else { return };
        let (real, imaginary, scale) = (deep.real.clone(), deep.imaginary.clone(), deep.scale);
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let mut m = self.m.clone();
        if self.auto_iterations {
            m.max_iterations = self.auto_settings.initial_limit(scale); //(The starting limit of auto iterations, which grows with the zoom)
        }
        let (c, coloring_function) = (self.c.clone(), self.coloring_function);
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let (sender, receiver) = mpsc::channel();
        let samples = if self.quality_blur { self.supersampling_amount.max(1) } else { self.supersampling_amount.clamp(1, 64) };
        let message = format!("{message} (deep zoom, ×{:.3e}, {} iterations, {samples} samples)", 1.0 / scale, m.max_iterations);
        std::thread::spawn(move || {
            let started = Instant::now();
            let color_of = |position: f64| m.interpolated_color(position, |count| m.color(coloring_function, count));
            let mut layer_started = Instant::now();
            let completed = deep::render_layers(&m, &real, &imaginary, scale, (width, height), samples, &color_of, &thread_cancel, |colors, layer| {
                let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), plane: c.clone(), pass: layer, total: samples, layer_time: layer_started.elapsed(), whole: true });
                layer_started = Instant::now();
            });
            let message = if completed { message } else { String::from("Cancelled the deep rendering, showing the layers that were completed") };
            let _ = sender.send(JobMessage::Done { c, m, message, elapsed: started.elapsed(), completed, reported: true });
        });
        self.job = Some(receiver);
        self.progress = Some(Progress { pass: 0, total: samples, cancel, stop: None });
    }

    /// Goes to the exact center `exact` (decimal text) at `scale`: the deep view when it is too deep for 64-bit floats (a place saved in a deep view)
    pub(super) fn go_to_exact(&mut self, exact: &(String, String), scale: f64) {
        let bits = deep::bits_for(scale, self.p.pixel_plane.width);
        if let (Some(real), Some(imaginary)) = (Fixed::parse(&exact.0, bits), Fixed::parse(&exact.1, bits)) {
            self.set_deep(&real, &imaginary, scale);
        }
    }

    /// Saves the deep view as an image (rendered again at the saved size and quality, deep), with its exact view in the metadata; false when there is no deep
    /// view (then the image is saved as always)
    pub(super) fn save_deep_image(&mut self) -> bool {
        let Some(deep) = self.deep_view() else { return false };
        let (real, imaginary, scale) = (deep.real.clone(), deep.imaginary.clone(), deep.scale);
        let resolution = match self.save_size.resolve(self.default_save_resolution(), self.view_resolution()) {
            Ok(resolution) => resolution,
            Err(reason) => {
                self.log(format!("Could not save the image: {reason}"));
                return true;
            }
        };
        let (width, height) = (resolution.width, resolution.height);
        //The center needs the bits of a pixel of the saved size, which can be smaller than a pixel of the window
        let bits = deep::bits_for(scale, width).max(real.bits());
        let (real, imaginary) = (real.with_bits(bits), imaginary.with_bits(bits));
        let time_stamp = chrono::Utc::now().to_string();
        let file_name = time_stamp.replace(':', "-").replace(' ', "_");
        let place = crate::points_of_interest::to_json(&self.current_point(file_name.clone(), String::new()));
        let (m, coloring_function, mapping, samples) = (self.m.clone(), self.coloring_function, self.p.color_channel_mapping, self.image_supersampling_amount);
        let sender = self.notice_sender.clone();
        std::thread::spawn(move || {
            let color_of = |position: f64| m.interpolated_color(position, |count| m.color(coloring_function, count));
            let mut image = crate::pixel_buffer::PixelBuffer::new(crate::pixel_buffer::pixel_plane::PixelPlane::new(width, height));
            image.color_channel_mapping = mapping;
            deep::render_layers(&m, &real, &imaginary, scale, (width, height), samples, &color_of, &AtomicBool::new(false), |colors, layer| {
                if layer == samples.max(1) {
                    image.colors.copy_from_slice(colors);
                }
            });
            image.update_pixels();
            let view = View::new(real.to_f64(), imaginary.to_f64(), scale);
            let message = match image.save_as_png(&time_stamp, &view, &m, samples, Some(&place)) {
                Ok(()) => format!("Saved the deep image to saved/{file_name}.png"),
                Err(err) => format!("Could not save the image to saved/{file_name}.png: {err}"),
            };
            let _ = sender.send(message);
        });
        self.log("Saving the deep image...");
        true
    }
}

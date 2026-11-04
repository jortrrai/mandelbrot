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
use eframe::egui;

use super::{transform, MandelbrotApp};
use crate::complex_plane::{ComplexPlane, Precision, View};
use crate::rendering::deep::{self, Depth, Fixed};

/// The exact view deeper than 64-bit floats: its center, its scale, and the view of the complex plane it left (to tell when something else moved it)
pub(super) struct DeepView {
    pub(super) real: Fixed,
    pub(super) imaginary: Fixed,
    pub(super) scale: Depth,
    plane: String,
}

/// An exact view: its center and its scale (of a deep view, or of a 64-bit plane, see `ExactView::of_plane`)
#[derive(Clone, Debug)]
pub(super) struct ExactView {
    real: Fixed,
    imaginary: Fixed,
    scale: Depth,
}

impl ExactView {
    /// The exact view of the 64-bit plane `plane`, with `bits` bits after the binary point
    fn of_plane(plane: &ComplexPlane, bits: u32) -> ExactView {
        let (center, zero) = (plane.center(), Fixed::parse("0", bits).expect("0 is a number"));
        ExactView { real: zero.plus(center.x), imaginary: zero.plus(center.y), scale: Depth::from_f64(plane.get_scale()) }
    }
}

impl DeepView {
    /// Its exact view
    pub(super) fn exact(&self) -> ExactView {
        ExactView { real: self.real.clone(), imaginary: self.imaginary.clone(), scale: self.scale }
    }
}

/// Where the image of the exact view `shown` belongs in the image of the exact view `wanted` (both turned by `rotation`), in fractions of the wanted image, as
/// `transform::shown_fraction` gives it for 64-bit planes: also where the views are deeper than 64-bit floats can tell apart. `None` when the image would
/// be far too large or small to draw
fn exact_fraction(shown: &ExactView, wanted: &ExactView, rotation: f64, (width, height): (f64, f64)) -> Option<egui::Rect> {
    //In units of `2^exponent` of the wanted view: the view is `2.5 · mantissa` wide (see `Depth`)
    let power = wanted.scale.exponent;
    let (right, up) = (shown.real.offset_from(&wanted.real, power), shown.imaginary.offset_from(&wanted.imaginary, power));
    let (right, up) = deep::turned((right, up), -rotation); //(Back to the screen's directions)
    let across = 2.5 * wanted.scale.mantissa;
    let size = Depth { mantissa: shown.scale.mantissa, exponent: shown.scale.exponent.saturating_sub(power) }.to_f64() / wanted.scale.mantissa;
    let (x, y) = (0.5 + right / across, 0.5 - up / (across * height / width));
    let fraction = egui::Rect::from_center_size(egui::pos2(x as f32, y as f32), egui::vec2(size as f32, size as f32));
    (size.is_finite() && (1e-4..1e4).contains(&size) && x.abs() < 1e4 && y.abs() < 1e4).then_some(fraction)
}

impl MandelbrotApp {
    /// Whether a view of `scale` around the current center is too deep for 64-bit floats (a Julia set never is: the deep renderer draws the Mandelbrot set
    /// only, so a Julia set stops at the depth of 64-bit floats as before)
    fn too_deep(&self, scale: f64) -> bool {
        if self.m.julia.is_some() {
            return false;
        }
        //(Beyond 64-bit floats' range always, #263)
        if !(scale.is_normal()) {
            return true;
        }
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
    fn set_deep(&mut self, real: &Fixed, imaginary: &Fixed, depth: Depth) {
        let (center, scale) = ((real.to_f64(), imaginary.to_f64()), depth.to_f64());
        if !self.too_deep(scale) {
            self.deep = None;
            self.c.set_view(&View::new(center.0, center.1, scale));
            return;
        }
        //The complex plane at the deepest scale that 64-bit floats can show (the scale doubles until they can; from the deepest normal float for deeper views)
        let mut plane_scale = if scale.is_normal() { scale } else { f64::MIN_POSITIVE };
        while self.too_deep(plane_scale) && plane_scale < 1.0 {
            plane_scale *= 2.0;
        }
        self.c.set_view(&View::new(center.0, center.1, plane_scale));
        //The bits the scale needs, and never fewer than the center has: zooming out and in again comes back to exactly the same point
        let bits = deep::bits_for_depth(depth, self.p.pixel_plane.width).max(real.bits()).max(imaginary.bits());
        self.deep = Some(DeepView { real: real.with_bits(bits), imaginary: imaginary.with_bits(bits), scale: depth, plane: format!("{:?}", self.c.get_view()) });
    }

    /// The exact center and scale: of the deep view, or of the complex plane
    fn exact_view(&mut self) -> (Fixed, Fixed, Depth) {
        if let Some(deep) = self.deep_view() {
            return (deep.real.clone(), deep.imaginary.clone(), deep.scale);
        }
        let center = self.exact_center();
        let (scale, width) = (self.c.get_scale(), self.p.pixel_plane.width);
        let bits = deep::bits_for(scale * 1e-6, width); //(Room to zoom in)
        let zero = Fixed::parse("0", bits).expect("0 is a number");
        (zero.plus(center.x), zero.plus(center.y), Depth::from_f64(scale))
    }

    /// Whether a zoom by `factor` (below 1: in) is handled by the deep view: zooming in where 64-bit floats run out, or any zoom in a deep view
    pub(super) fn zooms_deep(&mut self, factor: f64) -> bool {
        self.deep_view().is_some() || (factor < 1.0 && self.too_deep(self.c.get_scale() * factor))
    }

    /// Zooms by `factor` around the pixel `(x, y)` of the view (exactly: the point under the cursor stays there), in the deep view
    pub(super) fn zoom_deep(&mut self, factor: f64, (x, y): (f64, f64)) {
        let (real, imaginary, depth) = self.exact_view();
        let (width, height) = (self.p.pixel_plane.width as f64, self.p.pixel_plane.height as f64);
        //(A pixel is `pixel · 2^depth.exponent`, see `Depth`)
        let pixel = 2.5 * depth.mantissa / width;
        let (anchor_x, anchor_y) = deep::turned(((x - width / 2.0) * pixel, (height / 2.0 - y) * pixel), self.c.rotation());
        self.set_deep(&real.plus_scaled(anchor_x * (1.0 - factor), depth.exponent), &imaginary.plus_scaled(anchor_y * (1.0 - factor), depth.exponent), depth.times(factor));
    }

    /// Moves the deep view by whole pixels (as a drag does), exactly
    pub(super) fn pan_deep(&mut self, columns: i64, rows_up: i64) {
        let (real, imaginary, depth) = self.exact_view();
        let pixel = 2.5 * depth.mantissa / self.p.pixel_plane.width as f64;
        let (right, up) = deep::turned((columns as f64 * pixel, rows_up as f64 * pixel), self.c.rotation());
        self.set_deep(&real.plus_scaled(right, depth.exponent), &imaginary.plus_scaled(up, depth.exponent), depth);
    }

    /// Sets the zoom to `zoom` around the exact center, in the deep view when it is too deep for 64-bit floats (for the depth control)
    pub(super) fn set_deep_zoom(&mut self, zoom: f64) {
        let (real, imaginary, _) = self.exact_view();
        self.set_deep(&real, &imaginary, Depth::from_f64(1.0 / zoom));
    }

    /// Sets the scale to `depth` around the exact center, in the deep view (for the depth control, beyond 64-bit floats)
    pub(super) fn set_deep_depth(&mut self, depth: Depth) {
        let (real, imaginary, _) = self.exact_view();
        self.set_deep(&real, &imaginary, depth);
    }

    /// The deep view if the window shows it (without leaving it when it does not, see `deep_view`)
    pub(super) fn deep_shown(&self) -> Option<&DeepView> {
        self.deep.as_ref().filter(|deep| deep.plane == format!("{:?}", self.c.get_view()))
    }

    /// Where the image of `plane` (exactly `exact`, when it is of a deep view) belongs in the image on the screen, see `transform::shown_fraction`: placed exactly
    /// when either is a deep view, whose 64-bit planes cannot tell where it is
    pub(super) fn fraction_of(&self, plane: &ComplexPlane, exact: Option<&ExactView>) -> egui::Rect {
        let held = self.required_readiness() != super::Readiness::Any; //(Then the screen shows the view of its frame, see `drawn_fraction`)
        let wanted = if held { self.shown_deep.clone() } else { self.deep_shown().map(DeepView::exact) };
        let (wanted_plane, rotation) = (if held { &self.shown_plane } else { &self.c }, self.c.rotation());
        if (exact.is_none() && wanted.is_none()) || plane.rotation().to_bits() != rotation.to_bits() {
            return transform::shown_fraction(plane, wanted_plane);
        }
        let bits = exact.or(wanted.as_ref()).map_or(64, |view| view.real.bits());
        let shown = exact.cloned().unwrap_or_else(|| ExactView::of_plane(plane, bits));
        let wanted = wanted.unwrap_or_else(|| ExactView::of_plane(wanted_plane, bits));
        let size = (self.p.pixel_plane.width as f64, self.p.pixel_plane.height as f64);
        exact_fraction(&shown, &wanted, rotation, size).unwrap_or(egui::Rect::NOTHING)
    }

    /// The scale of the view that is shown: the deep view's, or the complex plane's
    pub(super) fn shown_scale(&self) -> f64 {
        self.deep_shown().map_or_else(|| self.c.get_scale(), |deep| deep.scale.to_f64())
    }

    /// The depth of the view that is shown, also beyond 64-bit floats (see `shown_scale`)
    pub(super) fn shown_depth(&self) -> Depth {
        self.deep_shown().map_or_else(|| Depth::from_f64(self.c.get_scale()), |deep| deep.scale)
    }

    /// Renders the deep view on a background thread (one sample a pixel), as a render of the window: its image arrives as the one layer of the render
    pub(super) fn start_deep_render(&mut self, message: &str) {
        let Some(deep) = self.deep_view() else { return };
        let (real, imaginary, depth) = (deep.real.clone(), deep.imaginary.clone(), deep.scale);
        let (width, height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let mut m = self.m.clone();
        if self.auto_iterations {
            m.max_iterations = if depth.to_f64().is_normal() { self.auto_settings.initial_limit(depth.to_f64()) } else { self.auto_settings.initial_limit_at(-depth.log2()) }; //(The starting limit of auto iterations, which grows with the zoom)
        }
        let (c, coloring_function, fast_deep, deep_f32) = (self.c.clone(), self.coloring_function, self.fast_deep, self.deep_f32);
        let gpu = self.gpu.clone().filter(|_| self.use_gpu); //(Iterates the deep view where it can, see `deep::render_layers_on`)
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let (sender, receiver) = mpsc::channel();
        let samples = if self.quality_blur { self.supersampling_amount.max(1) } else { self.supersampling_amount.clamp(1, 64) };
        let message = format!("{message} (deep zoom, scale {}, {} iterations, {samples} samples)", depth.to_text(), m.max_iterations);
        std::thread::spawn(move || {
            let started = Instant::now();
            let color_of = |position: f64| m.interpolated_color(position, |count| m.color(coloring_function, count));
            let mut layer_started = Instant::now();
            let orbit = deep::SharedOrbit::new(&real, &imaginary);
            let completed = deep::render_layers_on(&m, (&orbit, (0.0, 0.0)), (depth, c.rotation()), (width, height), samples, &color_of, fast_deep, gpu.as_deref().map(|gpu| (gpu, deep_f32)), &thread_cancel, |colors, layer| {
                let _ = sender.send(JobMessage::Pass { colors: colors.to_vec(), plane: c.clone(), pass: layer, total: samples, layer_time: layer_started.elapsed(), whole: true });
                layer_started = Instant::now();
            });
            let message = if completed { message } else { String::from("Cancelled the deep rendering, showing the layers that were completed") };
            let _ = sender.send(JobMessage::Done { c, m, message, elapsed: started.elapsed(), completed, reported: true });
        });
        self.job = Some(receiver);
        self.job_deep = self.deep_shown().map(DeepView::exact);
        self.progress = Some(Progress { pass: 0, total: samples, cancel, stop: None });
    }

    /// Goes to the exact center `exact` (decimal text) at `scale`: the deep view when it is too deep for 64-bit floats (a place saved in a deep view)
    pub(super) fn go_to_exact(&mut self, exact: &(String, String), depth: Depth) {
        let bits = deep::bits_for_depth(depth, self.p.pixel_plane.width);
        if let (Some(real), Some(imaginary)) = (Fixed::parse(&exact.0, bits), Fixed::parse(&exact.1, bits)) {
            self.set_deep(&real, &imaginary, depth);
        }
    }

    /// Saves the deep view as an image (rendered again at the saved size and quality, deep), with its exact view in the metadata; false when there is no deep
    /// view (then the image is saved as always)
    pub(super) fn save_deep_image(&mut self) -> bool {
        let Some(deep) = self.deep_view() else { return false };
        let (real, imaginary, depth) = (deep.real.clone(), deep.imaginary.clone(), deep.scale);
        let resolution = match self.save_size.resolve(self.default_save_resolution(), self.view_resolution()) {
            Ok(resolution) => resolution,
            Err(reason) => {
                self.log(format!("Could not save the image: {reason}"));
                return true;
            }
        };
        let (width, height) = (resolution.width, resolution.height);
        //The center needs the bits of a pixel of the saved size, which can be smaller than a pixel of the window
        let bits = deep::bits_for_depth(depth, width).max(real.bits());
        let (real, imaginary) = (real.with_bits(bits), imaginary.with_bits(bits));
        let time_stamp = chrono::Utc::now().to_string();
        let file_name = time_stamp.replace(':', "-").replace(' ', "_");
        let place = crate::points_of_interest::to_json(&self.current_point(file_name.clone(), String::new()));
        let (m, coloring_function, mapping, samples) = (self.m.clone(), self.coloring_function, self.p.color_channel_mapping, self.image_supersampling_amount);
        let (rotation, fast_deep, deep_f32) = (self.c.rotation(), self.fast_deep, self.deep_f32);
        let gpu = self.gpu.clone().filter(|_| self.use_gpu);
        let sender = self.notice_sender.clone();
        std::thread::spawn(move || {
            let color_of = |position: f64| m.interpolated_color(position, |count| m.color(coloring_function, count));
            let mut image = crate::pixel_buffer::PixelBuffer::new(crate::pixel_buffer::pixel_plane::PixelPlane::new(width, height));
            image.color_channel_mapping = mapping;
            let orbit = deep::SharedOrbit::new(&real, &imaginary);
            deep::render_layers_on(&m, (&orbit, (0.0, 0.0)), (depth, rotation), (width, height), samples, &color_of, fast_deep, gpu.as_deref().map(|gpu| (gpu, deep_f32)), &AtomicBool::new(false), |colors, layer| {
                if layer == samples.max(1) {
                    image.colors.copy_from_slice(colors);
                }
            });
            image.update_pixels();
            let view = View::new(real.to_f64(), imaginary.to_f64(), depth.to_f64());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn view(real: &str, imaginary: &str, scale: &str) -> ExactView {
        let scale = Depth::parse(scale).expect("a depth");
        let bits = deep::bits_for_depth(scale, 1000);
        ExactView { real: Fixed::parse(real, bits).expect("a number"), imaginary: Fixed::parse(imaginary, bits).expect("a number"), scale }
    }

    fn assert_close(actual: egui::Rect, expected: egui::Rect) {
        let error = (actual.min - expected.min).length() + (actual.max - expected.max).length();
        assert!(error < 1e-4, "{actual:?} is not {expected:?}");
    }

    #[test]
    fn the_old_image_of_a_deep_view_moves_with_a_pan_and_grows_with_a_zoom_far_beyond_64_bit_floats() {
        let size = (1000.0, 500.0);
        let old = view("-1.74", "0.01", "1e-400");
        //A pixel is 2.5e-403: 100 pixels to the right and 50 up moves the old image 100 pixels left and 50 down
        let pixel = Depth::parse("2.5e-403").expect("a depth");
        let moved = ExactView { real: old.real.plus_scaled(100.0 * pixel.mantissa, pixel.exponent), imaginary: old.imaginary.plus_scaled(50.0 * pixel.mantissa, pixel.exponent), scale: old.scale };
        let fraction = exact_fraction(&old, &moved, 0.0, size).expect("on the screen");
        assert_close(fraction, egui::Rect::from_min_max(egui::pos2(-0.1, 0.1), egui::pos2(0.9, 1.1)));
        //Zoomed in twice around the center: the old image is twice as large
        let zoomed = ExactView { scale: old.scale.times(0.5), ..old.clone() };
        assert_close(exact_fraction(&old, &zoomed, 0.0, size).expect("on the screen"), egui::Rect::from_min_max(egui::pos2(-0.5, -0.5), egui::pos2(1.5, 1.5)));
        //Turned a quarter: the pan to the right of the screen is up in the plane, and the old image still goes left
        let turned = std::f64::consts::FRAC_PI_2;
        let (right, up) = deep::turned((100.0 * pixel.mantissa, 0.0), turned);
        let moved = ExactView { real: old.real.plus_scaled(right, pixel.exponent), imaginary: old.imaginary.plus_scaled(up, pixel.exponent), scale: old.scale };
        assert_close(exact_fraction(&old, &moved, turned, size).expect("on the screen"), egui::Rect::from_min_max(egui::pos2(-0.1, 0.0), egui::pos2(0.9, 1.0)));
        //An image a million times larger is not drawn
        assert!(exact_fraction(&view("-1.74", "0.01", "1e-394"), &old, 0.0, size).is_none());
    }
}

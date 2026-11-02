//! The neighborhood of the view (issue #144): once the view is finished and nothing else runs, the area around it is rendered too, by default one view to every
//! side (3 x 3 views; *Neighborhood size* sets the share of a view to every side), at half the resolution of the view, so by default it costs about 2.25 views of
//! pixels. A large neighborhood is rendered at a lower resolution, so its texture stays within `MAX_SIDE` pixels. The window draws it behind the image: a drag, a pan or a zoom out then
//! shows the neighborhood at once where the old image does not reach, instead of black, until the render of the new view fills it in.
//!
//! It is only drawn while it has the colors of the view (the same limit, palette, coloring and channels), and it is rendered again when the view has moved
//! out of its middle or zoomed. Any render, flight or gesture stops its job at once.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};

use super::{colors_to_rgb_bytes, transform, MandelbrotApp, FULL_TEXTURE};
use crate::coloring::{ColorChannelMapping, TrueColor};
use crate::complex_plane::{ComplexPlane, View};
use crate::mandelbrot_set::MandelbrotSet;
use crate::rendering::Render;

/// The neighborhood has this many pixels for every pixel of the view it covers (half the resolution)
const RESOLUTION: f64 = 0.5;

/// The longest side of the neighborhood in pixels: a larger neighborhood gets a lower resolution, so it fits a texture of the GPU and the memory
pub(super) const MAX_SIDE: usize = 8192;

/// The size of the neighborhood at first: one view to every side, in percent of the view
pub(super) const DEFAULT_MARGIN: u16 = 100;

/// How many views the neighborhood is across and down: the view in the middle and `margin` percent of a view to every side
fn views_across(margin: u16) -> f64 {
    1.0 + 2.0 * f64::from(margin) / 100.0
}

/// What the colors of a neighborhood depend on: it is only drawn under an image with the same
#[derive(Clone, Debug, PartialEq)]
pub(super) struct NeighborhoodKey {
    max_iterations: u32,
    palette: (u64, u64, bool),
    smooth_coloring: bool,
    orbit_radius: u64,
    coloring_function: usize,
    mapping: ColorChannelMapping,
    julia: Option<(u64, u64)>,
}

impl NeighborhoodKey {
    fn new(m: &MandelbrotSet, coloring_function: crate::ColoringFunction, mapping: ColorChannelMapping) -> NeighborhoodKey {
        NeighborhoodKey {
            max_iterations: m.max_iterations,
            palette: (m.palette.divide.to_bits(), m.palette.offset.to_bits(), m.palette.stretch),
            smooth_coloring: m.smooth_coloring,
            orbit_radius: m.orbit_radius.to_bits(),
            coloring_function: coloring_function as usize,
            mapping,
            julia: m.julia.map(|(x, y)| (x.to_bits(), y.to_bits())),
        }
    }
}

/// A rendered neighborhood: the plane it shows (`around`, the view it was made for, in its middle), its colors as a texture, and what they depend on
pub(super) struct Neighborhood {
    plane: ComplexPlane,
    around: ComplexPlane,
    texture: TextureHandle,
    key: NeighborhoodKey,
    /// Its size, in percent of a view to every side
    margin: u16,
}

/// The job that renders a neighborhood: its cancel flag, and the receiver of its plane and colors (with their size) when it is done
pub(super) struct NeighborhoodJob {
    cancel: Arc<AtomicBool>,
    result: Receiver<(ComplexPlane, Vec<TrueColor>, usize, usize)>,
    around: ComplexPlane,
    key: NeighborhoodKey,
    margin: u16,
}

#[cfg(test)]
impl Neighborhood {
    pub(super) const fn plane(&self) -> &ComplexPlane {
        &self.plane
    }

    pub(super) const fn around(&self) -> &ComplexPlane {
        &self.around
    }

    pub(super) fn texture_size(&self) -> [usize; 2] {
        self.texture.size()
    }
}

impl MandelbrotApp {
    /// Stops the neighborhood job, if one runs (a render, a flight or a gesture has priority)
    pub(super) fn cancel_neighborhood_job(&mut self) {
        if let Some(job) = self.neighborhood_job.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Takes a finished neighborhood, and starts the job for the view when the view is finished, nothing else runs, and the neighborhood there is not the
    /// neighborhood of this view
    pub(super) fn poll_neighborhood(&mut self, ctx: &egui::Context) {
        if let Some((plane, colors, width, height)) = self.neighborhood_job.as_ref().and_then(|job| job.result.try_recv().ok()) {
            let job = self.neighborhood_job.take().expect("the job sent its result");
            let image = ColorImage::from_rgb([width, height], &colors_to_rgb_bytes(&colors, job.key.mapping));
            let texture = ctx.load_texture("neighborhood", image, TextureOptions::LINEAR);
            self.neighborhood = Some(Neighborhood { plane, around: job.around, texture, key: job.key, margin: job.margin });
        }
        let idle = self.render_neighborhood && !self.is_busy() && self.flight.is_none() && !self.gesture_active && self.settle.is_none() && self.buffer_is_complete;
        if !idle || self.neighborhood_job.is_some() {
            return;
        }
        let key = NeighborhoodKey::new(&self.m, self.coloring_function, self.p.color_channel_mapping);
        if self.neighborhood.as_ref().is_some_and(|neighborhood| neighborhood.key == key && neighborhood.margin == self.neighborhood_margin && still_around(&neighborhood.around, &self.c, neighborhood.margin)) {
            return;
        }
        self.start_neighborhood_job(key);
        ctx.request_repaint_after(std::time::Duration::from_millis(50)); //Keep polling for the result
    }

    fn start_neighborhood_job(&mut self, key: NeighborhoodKey) {
        let margin = self.neighborhood_margin;
        let (width, height) = neighborhood_size(self.p.pixel_plane.width, self.p.pixel_plane.height, margin);
        let center = self.c.center();
        let mut plane = ComplexPlane::new(width, height);
        plane.set_view(&View::new(center.x, center.y, self.c.get_scale() * views_across(margin)));
        plane.set_rotation(self.c.rotation()); //(Turned like the view, around the same center)
        let (m, coloring_function) = (self.m.clone(), self.coloring_function);
        let (gpu, fast_gpu) = (self.gpu.clone().filter(|_| self.use_gpu), self.fast_gpu);
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        let (sender, result) = mpsc::channel();
        std::thread::spawn(move || {
            let mut colors = Vec::new();
            let completed = Render::new(&plane, &m, width, height, coloring_function).gpu(gpu.as_deref().map(|gpu| (gpu, std::time::Duration::ZERO))).fast_gpu(fast_gpu)
                .colors_only().cancel_with(&thread_cancel).run(|layer, _| colors = layer.to_vec());
            if completed {
                let _ = sender.send((plane, colors, width, height));
            }
        });
        self.neighborhood_job = Some(NeighborhoodJob { cancel, result, around: self.c.clone(), key, margin });
    }

    /// Draws the neighborhood behind the image in `rect`, at its place in the view on the screen, when it has the colors of the image
    pub(super) fn draw_neighborhood(&self, painter: &egui::Painter, rect: egui::Rect) {
        let key = NeighborhoodKey::new(&self.m, self.coloring_function, self.p.color_channel_mapping);
        if let Some(neighborhood) = self.neighborhood.as_ref().filter(|neighborhood| self.render_neighborhood && neighborhood.key == key) {
            let target = transform::rect_of(transform::shown_fraction(&neighborhood.plane, self.displayed_plane()), rect);
            painter.image(neighborhood.texture.id(), target, FULL_TEXTURE, Color32::WHITE);
        }
    }
}

/// Whether the neighborhood of `margin` percent made for the view `around` still serves the view `view`: the same zoom, and the view moved by at most half the
/// margin, so the neighborhood still reaches at least half its margin past every side (half a view by default)
pub(super) fn still_around(around: &ComplexPlane, view: &ComplexPlane, margin: u16) -> bool {
    //(Turned otherwise, the neighborhood shows other surroundings: it is rendered again)
    let same_zoom = (around.get_scale() / view.get_scale() - 1.0).abs() < 1e-9 && around.rotation().to_bits() == view.rotation().to_bits();
    let ((a, b), share) = ((around.unturned().center(), view.unturned().center()), f64::from(margin) / 200.0);
    same_zoom && (a.x - b.x).abs() <= around.length_x * share && (a.y - b.y).abs() <= around.length_y * share
}

/// The pixels of the neighborhood of `margin` percent of a `width` x `height` view: half the resolution of the view, lower where that would be more than
/// `MAX_SIDE` pixels on a side
pub(super) fn neighborhood_size(width: usize, height: usize, margin: u16) -> (usize, usize) {
    let across = views_across(margin) * RESOLUTION;
    let factor = across.min(MAX_SIDE as f64 / width.max(height).max(1) as f64);
    (((width as f64 * factor).round() as usize).max(1), ((height as f64 * factor).round() as usize).max(1))
}

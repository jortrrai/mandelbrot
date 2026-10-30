//! The neighborhood of the view (issue #144): once the view is finished and nothing else runs, the area around it is rendered too, up to one view to every side
//! (3 x 3 views), at half the resolution of the view, so it costs about 2.25 views of pixels. The window draws it behind the image: a drag, a pan or a zoom out then
//! shows the neighborhood at once where the old image does not reach, instead of black, until the render of the new view fills it in.
//!
//! It is only drawn while it has the colors of the view (the same limit, color cycle, coloring and channels), and it is rendered again when the view has moved
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

/// How many views the neighborhood is across and down (the view in the middle and one to every side)
const VIEWS_ACROSS: f64 = 3.0;

/// The neighborhood has this many pixels across for every pixel of the view (half the resolution: 3 views of half the pixels)
const PIXELS_PER_VIEW_PIXEL: f64 = 1.5;

/// What the colors of a neighborhood depend on: it is only drawn under an image with the same
#[derive(Clone, Debug, PartialEq)]
pub(super) struct NeighborhoodKey {
    max_iterations: u32,
    /// The color cycle: divide, offset, stretched (as bits, so the key compares exactly)
    palette: (u64, u64, bool),
    smooth_coloring: bool,
    orbit_radius: u64,
    coloring_function: usize,
    mapping: ColorChannelMapping,
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
        }
    }
}

/// A rendered neighborhood: the plane it shows (`around`, the view it was made for, in its middle), its colors as a texture, and what they depend on
pub(super) struct Neighborhood {
    plane: ComplexPlane,
    around: ComplexPlane,
    texture: TextureHandle,
    key: NeighborhoodKey,
}

/// The job that renders a neighborhood: its cancel flag, and the receiver of its plane and colors (with their size) when it is done
pub(super) struct NeighborhoodJob {
    cancel: Arc<AtomicBool>,
    result: Receiver<(ComplexPlane, Vec<TrueColor>, usize, usize)>,
    around: ComplexPlane,
    key: NeighborhoodKey,
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
            self.neighborhood = Some(Neighborhood { plane, around: job.around, texture, key: job.key });
        }
        let idle = self.render_neighborhood && !self.is_busy() && self.flight.is_none() && !self.gesture_active && self.settle.is_none() && self.buffer_is_complete;
        if !idle || self.neighborhood_job.is_some() {
            return;
        }
        let key = NeighborhoodKey::new(&self.m, self.coloring_function, self.p.color_channel_mapping);
        if self.neighborhood.as_ref().is_some_and(|neighborhood| neighborhood.key == key && still_around(&neighborhood.around, &self.c)) {
            return;
        }
        self.start_neighborhood_job(key);
        ctx.request_repaint_after(std::time::Duration::from_millis(50)); //Keep polling for the result
    }

    fn start_neighborhood_job(&mut self, key: NeighborhoodKey) {
        let (view_width, view_height) = (self.p.pixel_plane.width, self.p.pixel_plane.height);
        let (width, height) = (((view_width as f64) * PIXELS_PER_VIEW_PIXEL).round() as usize, ((view_height as f64) * PIXELS_PER_VIEW_PIXEL).round() as usize);
        let center = self.c.center();
        let mut plane = ComplexPlane::new(width.max(1), height.max(1));
        plane.set_view(&View::new(center.x, center.y, self.c.get_scale() * VIEWS_ACROSS));
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
        self.neighborhood_job = Some(NeighborhoodJob { cancel, result, around: self.c.clone(), key });
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

/// Whether the neighborhood made for the view `around` still serves the view `view`: the same zoom, and the middle of the view within the middle view of the
/// neighborhood, so it reaches at least half a view past every side
pub(super) fn still_around(around: &ComplexPlane, view: &ComplexPlane) -> bool {
    let same_zoom = (around.get_scale() / view.get_scale() - 1.0).abs() < 1e-9;
    let (a, b) = (around.center(), view.center());
    same_zoom && (a.x - b.x).abs() <= around.length_x / 2.0 && (a.y - b.y).abs() <= around.length_y / 2.0
}

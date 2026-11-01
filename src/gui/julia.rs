//! Julia sets and orbits (issue #202): the Julia set of the point under the mouse, live in a small map in the top right of the image (or pinned), and the orbit
//! of that point drawn over the image.
//!
//! The Julia set of `c` iterates `z = z² + c` from every pixel `z` of its view, where the Mandelbrot set iterates from 0 with `c` the pixel. So the Mandelbrot
//! set is a map of the Julia sets: inside it their sets are connected, near its edge they turn into dust. They are colored like the view (its coloring, palette
//! and smooth colors), with a plain loop on all threads: the map is small, so the renderer of the Mandelbrot set (its interior checks only hold for it) is not
//! needed.

use eframe::egui::{self, Color32};

use crate::coloring::TrueColor;
use crate::complex::Complex;
use crate::complex_plane::ComplexPlane;
use crate::mandelbrot_set::MandelbrotSet;

use super::MandelbrotApp;

/// The size of the Julia map in pixels, the part of the plane it shows (its width on the real axis around 0), and its iteration limit
pub(super) const MAP_SIZE: (usize, usize) = (240, 160);
const MAP_WIDTH: f64 = 3.6;
pub(super) const MAP_ITERATIONS: u32 = 400;

/// What the Julia map was rendered for: its point (as bits), the coloring, the divide and offset (as bits), stretched and smooth colors
pub(super) type MapKey = (u64, u64, usize, u64, u64, bool, bool);

/// How many steps of an orbit are drawn at most
pub(super) const ORBIT_STEPS: usize = 500;

/// The smooth escape time of `z = z² + c` from `z`, like `MandelbrotSet::position` (the iteration count plus a fraction from how far beyond the orbit radius it
/// landed, with smooth colors), or `max_iterations` if it stays
pub(super) fn julia_position(m: &MandelbrotSet, mut z: Complex, c: &Complex) -> f64 {
    let radius_squared = m.orbit_radius * m.orbit_radius;
    for iterations in 0..m.max_iterations {
        z = z.squared().add(c);
        let modulus_squared = z.x * z.x + z.y * z.y;
        if modulus_squared > radius_squared {
            if !m.smooth_coloring {
                return f64::from(iterations + 1);
            }
            //A few more steps, so the logarithm of a large |z| gives a smooth fraction
            for _ in 0..4 {
                z = z.squared().add(c);
            }
            let log_modulus = 0.5 * (z.x * z.x + z.y * z.y).ln();
            return (f64::from(iterations + 5) - log_modulus.ln() / std::f64::consts::LN_2).clamp(0.0, f64::from(m.max_iterations));
        }
    }
    f64::from(m.max_iterations)
}

/// The colors of the Julia set of `c`, `width` x `height` pixels of the plane `plane`, colored like `m` with `coloring_function`, on all render threads
pub(super) fn julia_image(plane: &ComplexPlane, m: &MandelbrotSet, c: &Complex, (width, height): (usize, usize), coloring_function: fn(f64, f64) -> TrueColor) -> Vec<TrueColor> {
    let mut colors = vec![TrueColor::new(0, 0, 0); width * height];
    let rows_per_chunk = height.div_ceil(crate::rendering::render_thread_count()).max(1);
    std::thread::scope(|scope| {
        for (chunk, rows) in colors.chunks_mut(rows_per_chunk * width).enumerate() {
            scope.spawn(move || {
                for (index, pixel) in rows.iter_mut().enumerate() {
                    let (x, y) = (index % width, chunk * rows_per_chunk + index / width);
                    let z = plane.complex_from_pixel_plane(x as f64 + 0.5, y as f64 + 0.5);
                    let position = julia_position(m, z, c);
                    *pixel = m.interpolated_color(position, |iterations| m.color(coloring_function, iterations));
                }
            });
        }
    });
    colors
}

/// The orbit of `c` in the Mandelbrot set: 0, c, c² + c, … for at most `steps` steps, until it leaves the orbit radius (that point included)
pub(super) fn orbit(c: &Complex, steps: usize, orbit_radius: f64) -> Vec<Complex> {
    let mut points = vec![Complex::new(0.0, 0.0)];
    let mut z = Complex::new(0.0, 0.0);
    for _ in 0..steps {
        z = z.squared().add(c);
        points.push(Complex::new(z.x, z.y));
        if z.x * z.x + z.y * z.y > orbit_radius * orbit_radius {
            break;
        }
    }
    points
}

/// The plane the Julia map shows: `MAP_WIDTH` wide around 0
fn map_plane() -> ComplexPlane {
    let mut plane = ComplexPlane::new(MAP_SIZE.0, MAP_SIZE.1);
    plane.set_view(&crate::complex_plane::View::new(0.0, 0.0, MAP_WIDTH / 2.5));
    plane
}

impl MandelbrotApp {
    /// The point of the plane under the mouse, if it is over the image `rect`
    fn point_under_mouse(&self, ui: &egui::Ui, rect: egui::Rect) -> Option<Complex> {
        let position = ui.input(|input| input.pointer.hover_pos()).filter(|position| rect.contains(*position))?;
        let (width, height) = (self.p.pixel_plane.width as f32, self.p.pixel_plane.height as f32);
        let x = (position.x - rect.min.x) / rect.width() * width;
        let y = (position.y - rect.min.y) / rect.height() * height;
        Some(self.displayed_plane().complex_from_pixel_plane(f64::from(x), f64::from(y)))
    }

    /// Draws the orbit of the point under the mouse over the image (with *Show orbit*), and the Julia map in the top right of the image (with *Julia map*):
    /// the Julia set of the pinned point, or of the point under the mouse. A middle click pins the point under the mouse (or unpins)
    pub(super) fn julia_overlays(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        if !self.show_orbit && !self.show_julia {
            return;
        }
        let hovered = self.point_under_mouse(ui, rect);
        if self.show_julia && ui.input(|input| input.pointer.button_clicked(egui::PointerButton::Middle)) {
            self.julia_pinned = match (self.julia_pinned.take(), &hovered) {
                (Some(_), _) | (None, None) => None,
                (None, Some(point)) => Some((point.x, point.y)),
            };
        }
        let painter = ui.painter_at(rect);
        if let (true, Some(point)) = (self.show_orbit, &hovered) {
            //The orbit over the image, where its points are on the screen (also in a turned view)
            let plane = self.displayed_plane().clone();
            let unturned = plane.unturned();
            let on_screen = |z: &Complex| {
                let turned = plane.unturned_point(z);
                rect.min + egui::vec2(((turned.x - unturned.min_x) / unturned.length_x) as f32 * rect.width(), ((-turned.y - unturned.min_y) / unturned.length_y) as f32 * rect.height())
            };
            let points: Vec<egui::Pos2> = orbit(point, ORBIT_STEPS, self.m.orbit_radius).iter().map(on_screen).collect();
            painter.add(egui::Shape::line(points.clone(), egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 170))));
            for point in points.iter().take(64) {
                painter.circle_filled(*point, 1.8, Color32::from_rgba_unmultiplied(255, 255, 255, 200));
            }
        }
        if !self.show_julia {
            return;
        }
        let c = self.julia_pinned.map(|(x, y)| Complex::new(x, y)).or(hovered);
        let Some(c) = c else { return };
        //The map again only for another point or other colors
        let key = (c.x.to_bits(), c.y.to_bits(), self.coloring_index, self.m.palette.divide.to_bits(), self.m.palette.offset.to_bits(), self.m.palette.stretch, self.m.smooth_coloring);
        if self.julia_map.as_ref().is_none_or(|(old, _)| *old != key) {
            let mut m = self.m.clone();
            m.max_iterations = MAP_ITERATIONS;
            let colors = julia_image(&map_plane(), &m, &c, MAP_SIZE, self.coloring_function);
            let rgb = crate::pixel_buffer::rgb_bytes(&colors, self.p.color_channel_mapping);
            let texture = ui.ctx().load_texture("julia", egui::ColorImage::from_rgb(MAP_SIZE.into(), &rgb), egui::TextureOptions::LINEAR);
            self.julia_map = Some((key, texture));
        }
        let Some((_, texture)) = &self.julia_map else { return };
        let size = egui::vec2(MAP_SIZE.0 as f32, MAP_SIZE.1 as f32).min(rect.size() * 0.4);
        let map = egui::Rect::from_min_size(egui::pos2(rect.max.x - size.x - 8.0, rect.min.y + 8.0), size);
        painter.rect_filled(map.expand(2.0), 3.0, Color32::from_black_alpha(160));
        painter.image(texture.id(), map, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)), Color32::WHITE);
        let label = format!("Julia c = {:.6} {} {:.6}i{}", c.x, if c.y < 0.0 { "-" } else { "+" }, c.y.abs(), if self.julia_pinned.is_some() { " (pinned)" } else { "" });
        painter.text(map.left_bottom() + egui::vec2(4.0, -4.0), egui::Align2::LEFT_BOTTOM, label, egui::FontId::proportional(11.0), Color32::WHITE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julia_sets_follow_the_mandelbrot_set_and_orbits_escape_or_stay() {
        let m = MandelbrotSet::new(200, 2.0);
        //c = 0: the Julia set is the unit disk: inside stays, outside escapes
        assert_eq!(julia_position(&m, Complex::new(0.5, 0.0), &Complex::new(0.0, 0.0)), 200.0);
        assert!(julia_position(&m, Complex::new(1.5, 0.0), &Complex::new(0.0, 0.0)) < 10.0);
        //The orbit of a point inside the set stays bounded for all its steps; one outside leaves
        assert_eq!(orbit(&Complex::new(-0.5, 0.0), 300, 2.0).len(), 301);
        let outside = orbit(&Complex::new(1.0, 1.0), 300, 2.0);
        assert!(outside.len() < 5 && outside.last().map_or(0.0, |z| z.x * z.x + z.y * z.y) > 4.0);
        //A map of c = -1 (the basilica): its middle is in the set, its corner is not
        let plane = map_plane();
        let colors = julia_image(&plane, &m, &Complex::new(-1.0, 0.0), MAP_SIZE, TrueColor::new_from_bernstein_polynomials);
        let (middle, corner) = (colors[MAP_SIZE.1 / 2 * MAP_SIZE.0 + MAP_SIZE.0 / 2], colors[0]);
        assert_ne!((middle.red, middle.green, middle.blue), (corner.red, corner.green, corner.blue));
    }
}

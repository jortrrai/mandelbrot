//! The render size that follows the size of the window: how large the area for the image is in pixels, and when the buffer is resized to it.
//!
//! Dragging the edge of a window changes its size many times a second, and every resize would cancel the render that is running and start another. So the buffer is only resized
//! after the wanted size has stayed the same for `SETTLE`; the first time, it is resized right away.

use std::time::{Duration, Instant};

use eframe::egui;

use super::MandelbrotApp;
use crate::resolution::Resolution;

/// How long the wanted size must stay the same before the buffer is resized to it
pub(super) const SETTLE: Duration = Duration::from_millis(300);

/// What to do about the size of the buffer in this frame
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FitStep {
    /// The buffer has the wanted size, or the size is not followed
    Idle,
    /// The wanted size changed lately: look again after this time
    Wait(Duration),
    /// Resize the buffer to this size now
    Resize(Resolution),
}

/// Whether the render size follows the window, and the size that is waiting to be applied
#[derive(Debug)]
pub(super) struct Fit {
    /// Whether the render size follows the window
    pub enabled: bool,
    /// The wanted size that is waiting for `SETTLE`, and since when it has been wanted
    pending: Option<(Resolution, Instant)>,
    /// Whether the buffer has been fitted since the option was turned on: the first fit is not delayed
    fitted: bool,
}

impl Fit {
    /// Following the window or not at the start
    pub const fn new(enabled: bool) -> Fit {
        Fit { enabled, pending: None, fitted: false }
    }

    /// Turns following the window on or off. Turned on, the next fit is immediate.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled != self.enabled {
            self.enabled = enabled;
            self.pending = None;
            self.fitted = false;
        }
    }

    /// What to do about the buffer of size `current`, when the window wants `wanted` (`None` if no size can be followed, e.g. the window is minimized or too large) at time `now`
    pub fn step(&mut self, current: Resolution, wanted: Option<Resolution>, now: Instant) -> FitStep {
        let Some(wanted) = wanted.filter(|wanted| self.enabled && *wanted != current) else {
            self.pending = None;
            return FitStep::Idle;
        };
        if !self.fitted {
            self.fitted = true;
            self.pending = None;
            return FitStep::Resize(wanted);
        }
        match self.pending {
            Some((size, since)) if size == wanted => match SETTLE.checked_sub(now.saturating_duration_since(since)) {
                Some(remaining) if !remaining.is_zero() => FitStep::Wait(remaining),
                _ => {
                    self.pending = None;
                    FitStep::Resize(wanted)
                }
            },
            _ => {
                self.pending = Some((wanted, now));
                FitStep::Wait(SETTLE)
            }
        }
    }
}

/// The size in pixels of an area of `width` x `height` points on a screen with `pixels_per_point` pixels per point (the area is rounded down, so the image never exceeds it).
/// `None` if the area is too small or too large to render at, see `Resolution::validate`.
pub(super) fn wanted_resolution(width: f32, height: f32, pixels_per_point: f32) -> Option<Resolution> {
    let pixels = |points: f32| (points * pixels_per_point).floor();
    let (width, height) = (pixels(width), pixels(height));
    if !(width.is_finite() && height.is_finite() && width >= 1.0 && height >= 1.0) {
        return None;
    }
    Resolution::new(width as usize, height as usize).validate().ok()
}

impl MandelbrotApp {
    /// Resizes the buffer to the area for the image (`available`, in points) when that has been the size of the area for a while, see `Fit`, and renders the view again at the new size.
    /// The view keeps its center and its zoom; a wider area shows the same part of the real axis, a taller area shows more of the imaginary axis.
    pub(super) fn fit_to_window(&mut self, ctx: &egui::Context, available: egui::Vec2) {
        let wanted = wanted_resolution(available.x, available.y, ctx.pixels_per_point());
        match self.fit.step(self.view_resolution(), wanted, Instant::now()) {
            FitStep::Idle => {}
            FitStep::Wait(remaining) => ctx.request_repaint_after(remaining),
            FitStep::Resize(size) => {
                ctx.request_repaint();
                self.resize_view(size);
            }
        }
    }

    /// Gives the view a new buffer of `size` pixels, with the same center and zoom, and renders it
    pub(super) fn resize_view(&mut self, size: Resolution) {
        self.c = self.c.resized(size.width, size.height);
        let mapping = self.p.color_channel_mapping;
        self.p = crate::pixel_buffer::PixelBuffer::new(crate::pixel_buffer::pixel_plane::PixelPlane::new(size.width, size.height));
        self.p.color_channel_mapping = mapping;
        self.shown_plane = self.c.clone();
        self.texture_outdated = true;
        self.render(&format!("Resized the view to {size} pixels"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Resolution = Resolution::new(1000, 700);
    const B: Resolution = Resolution::new(1200, 800);

    #[test]
    fn the_first_fit_is_immediate_and_later_ones_wait_until_the_size_has_settled() {
        let (mut fit, start) = (Fit::new(true), Instant::now());
        assert_eq!(fit.step(A, Some(B), start), FitStep::Resize(B), "the first fit does not wait");
        //The user drags the edge: the wanted size keeps changing, so nothing is resized
        let sizes = [Resolution::new(1210, 800), Resolution::new(1230, 800), Resolution::new(1260, 810)];
        for (index, size) in sizes.into_iter().enumerate() {
            let now = start + Duration::from_millis(50 * (index as u64 + 1));
            assert_eq!(fit.step(B, Some(size), now), FitStep::Wait(SETTLE));
        }
        //The last size stays: wait for the rest of the time, then resize
        let last = sizes[2];
        let since = start + Duration::from_millis(150);
        assert_eq!(fit.step(B, Some(last), since + Duration::from_millis(100)), FitStep::Wait(Duration::from_millis(200)));
        assert_eq!(fit.step(B, Some(last), since + SETTLE), FitStep::Resize(last));
        //After the resize the buffer has the wanted size, and nothing is left to do
        assert_eq!(fit.step(last, Some(last), since + SETTLE), FitStep::Idle);
    }

    #[test]
    fn a_size_that_comes_back_to_the_current_size_cancels_the_wait() {
        let (mut fit, start) = (Fit::new(true), Instant::now());
        fit.step(A, Some(A), start);
        assert_eq!(fit.step(A, Some(B), start), FitStep::Resize(B), "nothing was fitted yet");
        assert_eq!(fit.step(B, Some(A), start), FitStep::Wait(SETTLE));
        assert_eq!(fit.step(B, Some(B), start + Duration::from_millis(10)), FitStep::Idle);
        assert_eq!(fit.step(B, Some(A), start + SETTLE * 2), FitStep::Wait(SETTLE), "the wait starts again from this moment");
    }

    #[test]
    fn nothing_is_resized_when_following_is_off_or_there_is_no_size_to_follow() {
        let (mut fit, start) = (Fit::new(false), Instant::now());
        assert_eq!(fit.step(A, Some(B), start), FitStep::Idle);
        fit.set_enabled(true);
        assert_eq!(fit.step(A, None, start), FitStep::Idle, "a minimized window has no size");
        assert_eq!(fit.step(A, Some(B), start), FitStep::Resize(B), "turned on, the first fit is immediate");
        fit.set_enabled(false);
        assert_eq!(fit.step(B, Some(A), start + SETTLE * 2), FitStep::Idle);
        fit.set_enabled(true);
        assert_eq!(fit.step(B, Some(A), start + SETTLE * 3), FitStep::Resize(A), "turned on again, the first fit is immediate again");
    }

    #[test]
    fn the_wanted_size_is_the_area_in_pixels_rounded_down_and_limited() {
        assert_eq!(wanted_resolution(1200.0, 800.0, 1.0), Some(Resolution::new(1200, 800)));
        assert_eq!(wanted_resolution(1199.9, 799.5, 1.0), Some(Resolution::new(1199, 799)));
        assert_eq!(wanted_resolution(600.0, 400.0, 2.0), Some(Resolution::new(1200, 800)), "a screen with 2 pixels per point has more pixels");
        assert_eq!(wanted_resolution(1.0, 800.0, 1.0), None, "too narrow");
        assert_eq!(wanted_resolution(0.0, 0.0, 1.0), None);
        assert_eq!(wanted_resolution(f32::NAN, 800.0, 1.0), None);
        assert_eq!(wanted_resolution(20000.0, 800.0, 1.0), None, "too wide");
        assert_eq!(wanted_resolution(15000.0, 15000.0, 1.0), None, "too many pixels");
    }
}

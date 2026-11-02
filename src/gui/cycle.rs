//! Color cycling (issue #172): the color offset moves on by itself, `cycle_rate` steps of the palette per second, so the colors flow through the set while the
//! view stays where it is.
//!
//! Every step is a recolor (see `SampleState::recolor`): the first sample of every pixel gets its new color from its position, which is quick, and only that
//! layer is rendered (a frame like those of a drag). The next step follows once the last one is on the screen, so the steps come as fast as the recolors show
//! them, and the offset follows the clock. With *Auto blur during flights* the one-sample frames are blurred like the frames of a flight. When the cycling
//! stops, the render of the view continues to its full quality. A flight, a gesture or a stretched palette (which has no offset) pauses it.

use std::time::Instant;

use eframe::egui;

use super::render_job::RenderKind;
use super::MandelbrotApp;
use crate::mandelbrot_set::PALETTE_LENGTH;

/// The range of the cycling speed, in steps of the palette per second (negative: backwards; the palette has `PALETTE_LENGTH` steps)
pub(super) const CYCLE_RATES: std::ops::RangeInclusive<f64> = -250.0..=250.0;

/// The cycling speed at first: a cycle of the palette in 50 seconds
pub(super) const CYCLE_RATE: f64 = 10.0;

/// How far the cycling divide goes from where it started, in doublings either way (2: from a quarter to four times it)
pub(super) const DIVIDE_SWING: f64 = 2.0;

/// The range of the time of a whole swing of the divide, in seconds, and the time at first
pub(super) const DIVIDE_PERIODS: std::ops::RangeInclusive<f64> = 1.0..=600.0;
pub(super) const DIVIDE_PERIOD: f64 = 20.0;

/// The longest time one step catches up with (a pause of the window does not make the colors jump)
const MAX_STEP_SECONDS: f64 = 0.25;

impl MandelbrotApp {
    /// Moves the color offset on with the clock while the colors cycle, see the module
    pub(super) fn cycle_colors(&mut self, ctx: &egui::Context, now: Instant) {
        if !self.color_cycling && self.divide_cycle.is_none() {
            return;
        }
        let last = *self.cycle_last.get_or_insert(now);
        if self.flight.is_some() || self.gesture_active || self.m.palette.stretch {
            self.cycle_last = Some(now); //Paused: the colors go on from here afterwards
            return;
        }
        ctx.request_repaint();
        if self.is_busy() && !self.layer_shown {
            return; //The last step is not on the screen yet
        }
        let seconds = now.saturating_duration_since(last).as_secs_f64().min(MAX_STEP_SECONDS);
        if seconds <= 0.0 {
            return;
        }
        if self.color_cycling {
            self.m.palette.offset = (self.m.palette.offset + self.cycle_rate * seconds).rem_euclid(PALETTE_LENGTH);
        }
        //The divide breathes around the one it started at: up and down DIVIDE_SWING doublings, a whole swing in `divide_period` seconds (issue #203)
        if let Some((start, phase)) = self.divide_cycle {
            let phase = phase + seconds / self.divide_period.max(0.1);
            let doublings = DIVIDE_SWING * (std::f64::consts::TAU * phase).sin();
            self.m.palette.divide = (start * doublings.exp2()).clamp(crate::mandelbrot_set::MIN_DIVIDE, crate::mandelbrot_set::MAX_DIVIDE);
            self.divide_cycle = Some((start, phase.fract()));
        }
        self.cycle_last = Some(now);
        self.start_render_as("Cycling the colors", RenderKind::GestureFrame { moved: true });
    }

    /// Whether the colors or the divide cycle (then every step is a one-sample frame, see `cycle_colors`)
    pub(super) const fn cycling(&self) -> bool {
        self.color_cycling || self.divide_cycle.is_some()
    }

    /// Starts or stops the cycling of the iteration divide (issue #203); when it stops, the view is rendered with its full quality from the divide it has
    /// reached
    pub(super) fn toggle_divide_cycle(&mut self) {
        self.cycle_last = None;
        self.divide_cycle = match self.divide_cycle {
            Some(_) => None,
            None => Some((self.m.palette.divide, 0.0)),
        };
        if self.divide_cycle.is_none() {
            self.render_moved(&format!("Cycled the divide to {}", super::points::palette_description(self.m.palette)));
        }
    }

    /// Starts or stops the color cycling; when it stops, the view is rendered with its full quality from the colors it has reached
    pub(super) fn toggle_color_cycle(&mut self) {
        self.color_cycling = !self.color_cycling;
        self.cycle_last = None;
        if !self.color_cycling {
            self.render_moved(&format!("Cycled the colors to {}", super::points::palette_description(self.m.palette)));
        }
    }
}

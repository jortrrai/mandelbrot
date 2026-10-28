//! Rendering: the progressive renderer used by the window, adaptive supersampling, the regions that a move exposes, and the whole-buffer renderer used for saving images.
//!
//! * `progressive`: `Render`, the renderer that works one supersampling layer at a time.
//! * `adaptive`: the mask that decides which pixels get more samples.
//! * `region`: the pixels exposed by a move (`ExposedRegion`) and moving the complex plane by whole pixels.
//! * `buffer`: rendering a whole buffer at once, for saving images.
//! * `jitter`: the deterministic offsets of the samples.

mod adaptive;
mod buffer;
mod jitter;
mod progressive;
mod region;
mod state;

pub use adaptive::{adaptive_mask, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
pub use buffer::render_complex_plane_into_buffer;
pub use progressive::{Render, PREVIEW_BLOCKS};
pub use region::{move_complex_plane_by_pixels, pixel_shift, ExposedRegion};
pub use state::SampleState;

/// The amount of threads that rendering functions use: all logical CPU threads except one, which is left free for the rest of the system and the GUI, but at least one
pub fn render_thread_count() -> usize {
    num_cpus::get().saturating_sub(1).max(1)
}

#[cfg(test)]
mod tests;

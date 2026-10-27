//! Rendering: the progressive renderer used by the window, adaptive supersampling, the regions that a move exposes, and the whole-buffer renderer used for saving images.
//!
//! * `progressive`: `Render`, the renderer that works one supersampling layer at a time.
//! * `adaptive`: the mask that decides which pixels get more samples.
//! * `region`: the pixels exposed by a move (`ExposedRegion`) and moving the complex plane by whole pixels.
//! * `buffer`: the renderer of a whole buffer at once.
//! * `jitter`: the deterministic offsets of the samples.

mod adaptive;
mod buffer;
mod jitter;
mod progressive;
mod region;

pub use adaptive::{adaptive_mask, AdaptiveSampling, ADAPTIVE_UNIFORM_SAMPLES};
pub use buffer::{render_box_render_complex_plane_into_buffer, render_complex_plane_into_buffer, RenderBox};
pub use progressive::Render;
pub use region::{center_complex_plane_on, translate_complex_plane_by_pixels, ExposedRegion};

/// The amount of threads that rendering functions use: all logical CPU threads except one, which is left free for the rest of the system and the GUI, but at least one
pub fn render_thread_count() -> usize {
    num_cpus::get().saturating_sub(1).max(1)
}

#[cfg(test)]
mod tests;

//! Adaptive supersampling: the extra samples of a pixel are only taken where the picture shows detail.

use std::thread;

use super::render_thread_count;
use super::region::ExposedRegion;
use crate::coloring::TrueColor;

/// Settings of adaptive supersampling, see `render_region_adaptively`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveSampling {
    /// Two colors differ when the sum of the differences of their red, green and blue channels (0..=765) is above this
    pub threshold: u32,
}

impl Default for AdaptiveSampling {
    fn default() -> Self {
        AdaptiveSampling { threshold: 48 }
    }
}

/// The amount of samples every pixel gets before adaptive supersampling decides where more samples are needed
pub const ADAPTIVE_UNIFORM_SAMPLES: u8 = 2;

/// The distance between two colors: the sum of the differences of the red, green and blue channels (0 to 765)
fn color_distance(a: TrueColor, b: TrueColor) -> u32 {
    u32::from(a.red.abs_diff(b.red)) + u32::from(a.green.abs_diff(b.green)) + u32::from(a.blue.abs_diff(b.blue))
}

/// The pixels (of `region`) that need more samples: a pixel is marked when its two samples differ (`first` is the color of sample 0, `colors` the average of the uniform samples),
/// or when it differs from any of its 8 neighbors, and then every marked pixel also marks its 8 neighbors, so a thin feature that only one of the first samples hit is covered on both sides.
pub fn adaptive_mask(width: usize, height: usize, region: &ExposedRegion, first: &[TrueColor], colors: &[TrueColor], threshold: u32) -> Vec<bool> {
    let rows_per_chunk = height.div_ceil(render_thread_count()).max(1);
    let neighbors = |x: usize, y: usize| (y.saturating_sub(1)..=(y + 1).min(height - 1)).flat_map(move |ny| (x.saturating_sub(1)..=(x + 1).min(width - 1)).map(move |nx| ny * width + nx));
    //Rows are independent in both steps, so they are processed in parallel; the result does not depend on the split
    let mut marked = vec![false; width * height];
    thread::scope(|scope| {
        for (chunk, rows) in marked.chunks_mut(rows_per_chunk * width).enumerate() {
            scope.spawn(move || {
                for (row_in_chunk, row) in rows.chunks_mut(width).enumerate() {
                    let y = chunk * rows_per_chunk + row_in_chunk;
                    for (x, flag) in row.iter_mut().enumerate() {
                        if !region.contains(x, y) {
                            continue;
                        }
                        let index = y * width + x;
                        *flag = color_distance(first[index], colors[index]) > threshold || neighbors(x, y).any(|other| color_distance(colors[index], colors[other]) > threshold);
                    }
                }
            });
        }
    });
    let mut mask = vec![false; width * height];
    let marked = &marked;
    thread::scope(|scope| {
        for (chunk, rows) in mask.chunks_mut(rows_per_chunk * width).enumerate() {
            scope.spawn(move || {
                for (row_in_chunk, row) in rows.chunks_mut(width).enumerate() {
                    let y = chunk * rows_per_chunk + row_in_chunk;
                    for (x, flag) in row.iter_mut().enumerate() {
                        *flag = region.contains(x, y) && neighbors(x, y).any(|other| marked[other]);
                    }
                }
            });
        }
    });
    mask
}

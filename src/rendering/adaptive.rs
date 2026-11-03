//! Adaptive supersampling: the extra samples of a pixel are only taken where the picture shows detail.

use std::thread;

use super::render_thread_count;
use super::region::ExposedRegion;
use crate::coloring::TrueColor;

/// Settings of adaptive supersampling, see `Render::adaptive`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// The pixels (of `region`) that need more samples.
///
/// A pixel is marked when its two samples differ (`first` is the color of sample 0, `colors` the average of the uniform samples), or when it differs from any of its 8 neighbors.
/// Then every marked pixel also marks its 8 neighbors, so a thin feature that only one of the first samples hit is covered on both sides.
pub fn adaptive_mask(width: usize, height: usize, region: &ExposedRegion, first: &[TrueColor], colors: &[TrueColor], threshold: u32) -> Vec<bool> {
    mask_where(width, height, &|x, y| region.contains(x, y), first, colors, threshold)
}

/// `adaptive_mask` for the pixels where `inside` is true, with `colors` the average of the uniform samples of every pixel (and the color of the pixels outside)
pub(super) fn mask_where(width: usize, height: usize, inside: &(impl Fn(usize, usize) -> bool + Sync), first: &[TrueColor], colors: &[TrueColor], threshold: u32) -> Vec<bool> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let rows_per_chunk = height.div_ceil(render_thread_count()).max(1);
    //The rows around row `y` (itself, and those above and below that exist)
    let rows_around = |y: usize| y.saturating_sub(1)..=(y + 1).min(height - 1);
    //Rows are independent in both steps, so they are processed in parallel; the result does not depend on the split. (Plain loops over the neighbors: the
    //iterator chains over them took 50 ms for a 4K frame on 16 threads, while the GPU waited for the mask)
    let mut marked = vec![false; width * height];
    thread::scope(|scope| {
        for (chunk, rows) in marked.chunks_mut(rows_per_chunk * width).enumerate() {
            scope.spawn(move || {
                for (row_in_chunk, row) in rows.chunks_mut(width).enumerate() {
                    let y = chunk * rows_per_chunk + row_in_chunk;
                    for (x, flag) in row.iter_mut().enumerate() {
                        if !inside(x, y) {
                            continue;
                        }
                        let index = y * width + x;
                        let color = colors[index];
                        let (left, right) = (x.saturating_sub(1), (x + 1).min(width - 1));
                        *flag = color_distance(first[index], color) > threshold
                            || rows_around(y).any(|ny| colors[ny * width + left..=ny * width + right].iter().any(|other| color_distance(color, *other) > threshold));
                    }
                }
            });
        }
    });
    //Every marked pixel marks its 8 neighbors: a pixel is in the mask when any pixel of its 3 x 3 block is marked, from the marks of each row widened by one
    let mut widened = vec![false; width * height];
    let marked = &marked;
    thread::scope(|scope| {
        for (chunk, rows) in widened.chunks_mut(rows_per_chunk * width).enumerate() {
            scope.spawn(move || {
                for (row_in_chunk, row) in rows.chunks_mut(width).enumerate() {
                    let y = chunk * rows_per_chunk + row_in_chunk;
                    let source = &marked[y * width..(y + 1) * width];
                    for (x, flag) in row.iter_mut().enumerate() {
                        *flag = source[x.saturating_sub(1)..=(x + 1).min(width - 1)].contains(&true);
                    }
                }
            });
        }
    });
    let mut mask = vec![false; width * height];
    let widened = &widened;
    thread::scope(|scope| {
        for (chunk, rows) in mask.chunks_mut(rows_per_chunk * width).enumerate() {
            scope.spawn(move || {
                for (row_in_chunk, row) in rows.chunks_mut(width).enumerate() {
                    let y = chunk * rows_per_chunk + row_in_chunk;
                    for (x, flag) in row.iter_mut().enumerate() {
                        *flag = rows_around(y).any(|ny| widened[ny * width + x]) && inside(x, y);
                    }
                }
            });
        }
    });
    mask
}

#[cfg(test)]
mod faster_mask {
    use super::*;

    /// `mask_where` as it was before (#246), to compare with
    fn mask_where_before(width: usize, height: usize, inside: &(impl Fn(usize, usize) -> bool + Sync), first: &[TrueColor], colors: &[TrueColor], threshold: u32) -> Vec<bool> {
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
                            if !inside(x, y) {
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
                            *flag = inside(x, y) && neighbors(x, y).any(|other| marked[other]);
                        }
                    }
                });
            }
        });
        mask
    }

    #[test]
    fn the_mask_is_the_mask_of_before_bit_for_bit() {
        //#246: random colors with features of all sizes, every threshold, edges included, and a region
        let (width, height) = (97, 61);
        let mut seed = 12345_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u8
        };
        let colors: Vec<TrueColor> = (0..width * height).map(|index| if index % 7 < 3 { TrueColor::new(next(), next(), next()) } else { TrueColor::new(10, 20, 30) }).collect();
        let first: Vec<TrueColor> = colors.iter().map(|color| if next() < 40 { TrueColor::new(next(), color.green, color.blue) } else { *color }).collect();
        for threshold in [0, 5, 60, 300, 800] {
            for inside in [&(|_: usize, _: usize| true) as &(dyn Fn(usize, usize) -> bool + Sync), &|x: usize, y: usize| !(x + y).is_multiple_of(5) && x > 3] {
                assert_eq!(mask_where(width, height, &inside, &first, &colors, threshold), mask_where_before(width, height, &inside, &first, &colors, threshold), "threshold {threshold}");
            }
        }
    }
}


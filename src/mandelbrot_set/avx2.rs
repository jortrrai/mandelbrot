//! The refilling loop of `MandelbrotSet::positions` with AVX2 instructions: four lanes in one vector, with the bookkeeping after every block of steps (which lanes
//! are done, the cycle detection) as a few vector operations instead of a branch per lane, which is what the compiler makes of the generic loop.
//!
//! A row starts with the lockstep loop (`position_lanes`), which is the cheapest for points that stop within a few blocks, such as most points outside the set:
//! at the first group of four points that runs longer than `LOCKSTEP_POSITIONS`, the rest of the row goes to the refilling loop, where a lane whose point has
//! stopped takes the next one at once, so long and short orbits do not wait for each other.
//!
//! The arithmetic is the same operations in the same order as in `MandelbrotSet::iterate_lanes_smooth_generic` (multiplications, subtractions and additions that
//! round separately; no fused multiply-add), so every position is the same, bit for bit.

use std::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_andnot_pd, _mm256_blendv_pd, _mm256_castpd_si256, _mm256_castsi256_pd,
    _mm256_cmp_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_pd, _mm256_loadu_si256, _mm256_movemask_pd, _mm256_mul_pd, _mm256_or_pd, _mm256_set1_epi64x,
    _mm256_set1_pd, _mm256_setzero_si256, _mm256_storeu_pd, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_sub_pd, _CMP_EQ_OQ, _CMP_GT_OQ,
};
use std::sync::atomic::{AtomicBool, Ordering};

use super::{MandelbrotSet, CHECK_INTERVAL, LANES};
use crate::complex::Complex;

/// A row stays in the lockstep loop while every group of four points has positions below this (about the iterations of the slowest point of the group)
const LOCKSTEP_POSITIONS: f64 = 64.0;

/// The mask of every 4-bit `movemask`: all ones in the lanes whose bit is set
const LANE_MASKS: [[i64; 4]; 16] = {
    let mut masks = [[0; 4]; 16];
    let mut bits = 0;
    while bits < 16 {
        let mut lane = 0;
        while lane < 4 {
            masks[bits][lane] = if bits & (1 << lane) != 0 { -1 } else { 0 };
            lane += 1;
        }
        bits += 1;
    }
    masks
};

/// `MandelbrotSet::positions` with AVX2, see the module
/// # Safety
/// The CPU must support AVX2
#[target_feature(enable = "avx2")]
pub(super) unsafe fn positions(m: &MandelbrotSet, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
    let cancelled = || cancel.is_some_and(|cancel| cancel.load(Ordering::Relaxed));
    let (groups, _) = points.as_chunks::<LANES>();
    let mut start = 0;
    for group in groups {
        if cancelled() {
            return start;
        }
        let positions = m.position_lanes(group);
        out[start..start + LANES].copy_from_slice(&positions);
        start += LANES;
        if positions.iter().any(|position| *position >= LOCKSTEP_POSITIONS) {
            break;
        }
    }
    //SAFETY: the caller's CPU supports AVX2
    start + unsafe { refill(m, &points[start..], &mut out[start..], cancel) }
}

/// The refilling loop over all of `points`; returns how many points from the start have their position, like `MandelbrotSet::positions`
/// # Safety
/// The CPU must support AVX2
#[target_feature(enable = "avx2")]
#[allow(clippy::float_cmp, clippy::too_many_lines, clippy::cast_ptr_alignment)] // The cycle certificate compares floats exactly; one loop with its lane bookkeeping; unaligned loads and stores
unsafe fn refill(m: &MandelbrotSet, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
    const IDLE: usize = usize::MAX;
    let max = m.max_iterations;
    let last_whole = max - max % CHECK_INTERVAL;
    if last_whole == 0 {
        //Fewer steps than a block: the lockstep loop is the single one then
        for (c, position) in points.iter().zip(out.iter_mut()) {
            *position = m.position(c);
        }
        return points.len();
    }
    let (radius_squared, two, one) = (_mm256_set1_pd(m.orbit_radius * m.orbit_radius), _mm256_set1_pd(2.0), _mm256_set1_epi64x(1));
    //A lane with more blocks than this has no whole block left before `max_iterations`
    let last_block = _mm256_set1_epi64x(i64::from(last_whole / CHECK_INTERVAL) - 1);
    let all = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
    //The lanes as arrays, to put points in and take them out one lane at a time
    let (mut cx, mut cy, mut zx, mut zy) = ([0.0_f64; 4], [0.0_f64; 4], [0.0_f64; 4], [0.0_f64; 4]);
    let (mut iterations, mut blocks, mut live) = ([0_i64; 4], [0_i64; 4], [0_i64; 4]);
    let mut index = [IDLE; 4];
    let mut next = 0;
    //The next point that needs the loop (points in the cardioid or the period-2 bulb get their position at once)
    let take = |next: &mut usize, out: &mut [f64]| -> Option<(usize, f64, f64)> {
        while *next < points.len() {
            let (at, c) = (*next, &points[*next]);
            *next += 1;
            if m.is_known_interior(c) {
                out[at] = f64::from(max);
                continue;
            }
            return Some((at, c.x, c.y));
        }
        None
    };
    let mut lanes_live = 0;
    for lane in 0..4 {
        if let Some((at, x, y)) = take(&mut next, out) {
            (index[lane], cx[lane], cy[lane], live[lane]) = (at, x, y, -1);
            lanes_live += 1;
        }
    }
    //SAFETY (the loads and stores below): unaligned loads and stores of arrays of four 8-byte values
    let load = |values: &[f64; 4]| unsafe { _mm256_loadu_pd(values.as_ptr()) };
    let load_integers = |values: &[i64; 4]| unsafe { _mm256_loadu_si256(values.as_ptr().cast::<__m256i>()) };
    let (mut vcx, mut vcy, mut vzx, mut vzy) = (load(&cx), load(&cy), load(&zx), load(&zy));
    let (mut vsx, mut vsy) = (vzx, vzy);
    let (mut vit, mut vblocks) = (load_integers(&iterations), load_integers(&blocks));
    let mut vlive = _mm256_castsi256_pd(load_integers(&live));
    let mut vactive = vlive;
    while lanes_live > 0 {
        if cancel.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
            //Every point before the first one that a lane still has (and before `next`) is done
            return index.iter().copied().filter(|at| *at != IDLE).min().unwrap_or(next).min(next);
        }
        for _ in 0..CHECK_INTERVAL {
            //Same arithmetic as z = z.squared().add(c): (zx * zx - zy * zy) + cx and (2 * zx) * zy + cy
            let new_x = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(vzx, vzx), _mm256_mul_pd(vzy, vzy)), vcx);
            let new_y = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(two, vzx), vzy), vcy);
            (vzx, vzy) = (new_x, new_y);
            let escaped = _mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_add_pd(_mm256_mul_pd(new_x, new_x), _mm256_mul_pd(new_y, new_y)), radius_squared);
            vactive = _mm256_andnot_pd(escaped, vactive);
            vit = _mm256_sub_epi64(vit, _mm256_castpd_si256(vactive)); //An active lane is all ones: -1
        }
        vblocks = _mm256_add_epi64(vblocks, one);
        //Brent's cycle detection: an orbit that is exactly where it was when z was saved repeats forever
        let repeats = _mm256_and_pd(vactive, _mm256_and_pd(_mm256_cmp_pd::<_CMP_EQ_OQ>(vzx, vsx), _mm256_cmp_pd::<_CMP_EQ_OQ>(vzy, vsy)));
        let no_whole_block = _mm256_castsi256_pd(_mm256_cmpgt_epi64(vblocks, last_block));
        //Done: stopped in this block (escaped), a repeating orbit, or no whole block left
        let done = _mm256_movemask_pd(_mm256_and_pd(vlive, _mm256_or_pd(_mm256_andnot_pd(vactive, all), _mm256_or_pd(repeats, no_whole_block))));
        //The lockstep loop saves z after 1, 3, 7, 15, ... blocks (the gap doubles): after `blocks` blocks when `blocks + 1` is a power of two
        let save = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(vblocks, _mm256_add_epi64(vblocks, one)), _mm256_setzero_si256()));
        vsx = _mm256_blendv_pd(vsx, vzx, save);
        vsy = _mm256_blendv_pd(vsy, vzy, save);
        if done == 0 {
            continue;
        }
        //The lanes that are done: read what they need, give them their next point, and start them from 0 (z, the saved z, the counts) with masks
        let (still_active, repeating) = (_mm256_movemask_pd(vactive), _mm256_movemask_pd(repeats));
        //SAFETY: stores of vectors into arrays of four 8-byte values
        unsafe {
            _mm256_storeu_pd(zx.as_mut_ptr(), vzx);
            _mm256_storeu_pd(zy.as_mut_ptr(), vzy);
            _mm256_storeu_si256(iterations.as_mut_ptr().cast::<__m256i>(), vit);
            _mm256_storeu_si256(blocks.as_mut_ptr().cast::<__m256i>(), vblocks);
        }
        let mut remaining = done;
        while remaining != 0 {
            let lane = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let c = Complex::new(cx[lane], cy[lane]);
            let counted = u32::try_from(iterations[lane]).unwrap_or(u32::MAX);
            out[index[lane]] = if still_active & (1 << lane) == 0 {
                m.position_after_loop(counted, true, (zx[lane], zy[lane]), &c) //Escaped in this block: z is at the end of the block
            } else if repeating & (1 << lane) != 0 {
                f64::from(max)
            } else {
                m.finish_alone(u32::try_from(blocks[lane]).unwrap_or(u32::MAX) * CHECK_INTERVAL, counted, (zx[lane], zy[lane]), &c)
            };
            if let Some((at, x, y)) = take(&mut next, out) {
                (index[lane], cx[lane], cy[lane]) = (at, x, y);
            } else {
                (index[lane], cx[lane], cy[lane], live[lane]) = (IDLE, 0.0, 0.0, 0);
                lanes_live -= 1;
            }
        }
        let finished = _mm256_castsi256_pd(load_integers(&LANE_MASKS[done as usize]));
        (vcx, vcy) = (load(&cx), load(&cy));
        (vzx, vzy, vsx, vsy) = (_mm256_andnot_pd(finished, vzx), _mm256_andnot_pd(finished, vzy), _mm256_andnot_pd(finished, vsx), _mm256_andnot_pd(finished, vsy));
        (vit, vblocks) = (_mm256_castpd_si256(_mm256_andnot_pd(finished, _mm256_castsi256_pd(vit))), _mm256_castpd_si256(_mm256_andnot_pd(finished, _mm256_castsi256_pd(vblocks))));
        //A lane that got a point is live and active; one that did not is neither
        vlive = _mm256_castsi256_pd(load_integers(&live));
        vactive = _mm256_or_pd(_mm256_andnot_pd(finished, vactive), _mm256_and_pd(finished, vlive));
    }
    points.len()
}

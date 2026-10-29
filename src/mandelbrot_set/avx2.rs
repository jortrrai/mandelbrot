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

/// `MandelbrotSet::positions` with AVX2, see the module; with `wide` the refilling loop steps 3 vectors instead of 2 (see `refill_wide`)
/// # Safety
/// The CPU must support AVX2, and with `wide` AVX-512F and AVX-512VL
#[target_feature(enable = "avx2")]
pub(super) unsafe fn positions(m: &MandelbrotSet, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>, wide: bool) -> usize {
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
    //SAFETY: the caller's CPU supports AVX2, and AVX-512 with `wide`
    start + unsafe { if wide { refill_wide(m, &points[start..], &mut out[start..], cancel) } else { refill_avx2(m, &points[start..], &mut out[start..], cancel) } }
}

/// The most vectors of four lanes that `refill` steps together, see `refill_avx2` and `refill_wide`
const MAX_VECTORS: usize = 3;

/// The refilling loop with 2 vectors of four lanes, stepped together: independent of each other, so the CPU works on one while it waits for the result of the
/// other (measured on Seahorse Valley at 1x: 103 ms with one vector, 94 ms with two; three need more than the 16 AVX2 registers, see `refill_wide`)
/// # Safety
/// The CPU must support AVX2
#[target_feature(enable = "avx2")]
unsafe fn refill_avx2(m: &MandelbrotSet, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
    //SAFETY: the caller's CPU supports AVX2
    unsafe { refill::<2>(m, points, out, cancel) }
}

/// The refilling loop with 3 vectors, for a CPU with AVX-512: the same AVX2 instructions, but with its 32 vector registers the third vector fits without spilling
/// (measured on Seahorse Valley at 1x on a Ryzen 7 7800X3D: 83 ms instead of 92 ms with 2 vectors; 4 vectors took 86 ms). The same positions, bit for bit
/// # Safety
/// The CPU must support AVX2, AVX-512F and AVX-512VL
#[target_feature(enable = "avx2,avx512f,avx512vl")]
unsafe fn refill_wide(m: &MandelbrotSet, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
    //SAFETY: the caller's CPU supports AVX2 and AVX-512
    unsafe { refill::<3>(m, points, out, cancel) }
}

/// The refilling loop over all of `points` with `VECTORS` vectors of four lanes (at most `MAX_VECTORS`); returns how many points from the start have their
/// position, like `MandelbrotSet::positions`. Always inlined, into `refill_avx2` and `refill_wide`, which compile it with their instructions
/// # Safety
/// The CPU must support AVX2
#[inline(always)]
#[allow(clippy::float_cmp, clippy::too_many_lines, clippy::cast_ptr_alignment, clippy::inline_always)] // The cycle certificate compares floats exactly; one loop with its lane bookkeeping; unaligned loads and stores; inlined into the copies with their instructions
unsafe fn refill<const VECTORS: usize>(m: &MandelbrotSet, points: &[Complex], out: &mut [f64], cancel: Option<&AtomicBool>) -> usize {
    const IDLE: usize = usize::MAX;
    //The lanes of the arrays below that are used (the arrays have room for `MAX_VECTORS` vectors)
    const LANES_IN_ALL: usize = 4 * MAX_VECTORS;
    let lanes_used = 4 * VECTORS;
    let max = m.max_iterations;
    let last_whole = max - max % CHECK_INTERVAL;
    if last_whole == 0 {
        //Fewer steps than a block: the lockstep loop is the single one then
        for (c, position) in points.iter().zip(out.iter_mut()) {
            *position = m.position(c);
        }
        return points.len();
    }
    let (radius_squared, one) = (_mm256_set1_pd(m.orbit_radius * m.orbit_radius), _mm256_set1_epi64x(1));
    //A lane with more blocks than this has no whole block left before `max_iterations`
    let last_block = _mm256_set1_epi64x(i64::from(last_whole / CHECK_INTERVAL) - 1);
    let all = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
    //The lanes as arrays, to put points in and take them out one lane at a time (lane `4 * v + l` is lane `l` of vector `v`)
    let (mut cx, mut cy, mut zx, mut zy) = ([0.0_f64; LANES_IN_ALL], [0.0_f64; LANES_IN_ALL], [0.0_f64; LANES_IN_ALL], [0.0_f64; LANES_IN_ALL]);
    let (mut iterations, mut blocks, mut live) = ([0_i64; LANES_IN_ALL], [0_i64; LANES_IN_ALL], [0_i64; LANES_IN_ALL]);
    let mut index = [IDLE; LANES_IN_ALL];
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
    for lane in 0..lanes_used {
        if let Some((at, x, y)) = take(&mut next, out) {
            (index[lane], cx[lane], cy[lane], live[lane]) = (at, x, y, -1);
            lanes_live += 1;
        }
    }
    //SAFETY (the loads and stores below): unaligned loads and stores of four 8-byte values at offset `4 * v` of arrays of `LANES_IN_ALL`
    let load = |values: &[f64; LANES_IN_ALL], v: usize| unsafe { _mm256_loadu_pd(values.as_ptr().add(4 * v)) };
    let load_integers = |values: &[i64; LANES_IN_ALL], v: usize| unsafe { _mm256_loadu_si256(values.as_ptr().add(4 * v).cast::<__m256i>()) };
    let (mut vcx, mut vcy) = (std::array::from_fn::<_, VECTORS, _>(|v| load(&cx, v)), std::array::from_fn::<_, VECTORS, _>(|v| load(&cy, v)));
    let (mut vzx, mut vzy) = (std::array::from_fn::<_, VECTORS, _>(|v| load(&zx, v)), std::array::from_fn::<_, VECTORS, _>(|v| load(&zy, v)));
    let (mut vsx, mut vsy) = (vzx, vzy);
    //The squares of z, kept from the escape check of the step before: they are the squares the next step needs, so every step multiplies 3 times instead of 6
    let (mut vx2, mut vy2) = (std::array::from_fn::<_, VECTORS, _>(|v| _mm256_mul_pd(vzx[v], vzx[v])), std::array::from_fn::<_, VECTORS, _>(|v| _mm256_mul_pd(vzy[v], vzy[v])));
    let (mut vit, mut vblocks) = (std::array::from_fn::<_, VECTORS, _>(|v| load_integers(&iterations, v)), std::array::from_fn::<_, VECTORS, _>(|v| load_integers(&blocks, v)));
    let mut vlive = std::array::from_fn::<_, VECTORS, _>(|v| _mm256_castsi256_pd(load_integers(&live, v)));
    let mut vactive = vlive;
    while lanes_live > 0 {
        if cancel.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
            //Every point before the first one that a lane still has (and before `next`) is done
            return index.iter().copied().filter(|at| *at != IDLE).min().unwrap_or(next).min(next);
        }
        for _ in 0..CHECK_INTERVAL {
            for v in 0..VECTORS {
                //Same arithmetic as z = z.squared().add(c): (zx * zx - zy * zy) + cx and (2 * zx) * zy + cy, with the squares of the step before (the same numbers),
                //and 2 * zx as zx + zx (exact either way)
                let new_x = _mm256_add_pd(_mm256_sub_pd(vx2[v], vy2[v]), vcx[v]);
                let new_y = _mm256_add_pd(_mm256_mul_pd(_mm256_add_pd(vzx[v], vzx[v]), vzy[v]), vcy[v]);
                (vzx[v], vzy[v]) = (new_x, new_y);
                (vx2[v], vy2[v]) = (_mm256_mul_pd(new_x, new_x), _mm256_mul_pd(new_y, new_y));
                let escaped = _mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_add_pd(vx2[v], vy2[v]), radius_squared);
                vactive[v] = _mm256_andnot_pd(escaped, vactive[v]);
                vit[v] = _mm256_sub_epi64(vit[v], _mm256_castpd_si256(vactive[v])); //An active lane is all ones: -1
            }
        }
        let (mut done, mut still_active, mut repeating) = (0_u32, 0_u32, 0_u32);
        for v in 0..VECTORS {
            vblocks[v] = _mm256_add_epi64(vblocks[v], one);
            //Brent's cycle detection: an orbit that is exactly where it was when z was saved repeats forever
            let repeats = _mm256_and_pd(vactive[v], _mm256_and_pd(_mm256_cmp_pd::<_CMP_EQ_OQ>(vzx[v], vsx[v]), _mm256_cmp_pd::<_CMP_EQ_OQ>(vzy[v], vsy[v])));
            let no_whole_block = _mm256_castsi256_pd(_mm256_cmpgt_epi64(vblocks[v], last_block));
            //Done: stopped in this block (escaped), a repeating orbit, or no whole block left
            let done_here = _mm256_movemask_pd(_mm256_and_pd(vlive[v], _mm256_or_pd(_mm256_andnot_pd(vactive[v], all), _mm256_or_pd(repeats, no_whole_block))));
            done |= (done_here as u32) << (4 * v);
            still_active |= (_mm256_movemask_pd(vactive[v]) as u32) << (4 * v);
            repeating |= (_mm256_movemask_pd(repeats) as u32) << (4 * v);
            //The lockstep loop saves z after 1, 3, 7, 15, ... blocks (the gap doubles): after `blocks` blocks when `blocks + 1` is a power of two
            let save = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(vblocks[v], _mm256_add_epi64(vblocks[v], one)), _mm256_setzero_si256()));
            vsx[v] = _mm256_blendv_pd(vsx[v], vzx[v], save);
            vsy[v] = _mm256_blendv_pd(vsy[v], vzy[v], save);
        }
        if done == 0 {
            continue;
        }
        //The lanes that are done: read what they need, give them their next point, and start them from 0 (z, the saved z, the counts) with masks
        for v in (0..VECTORS).filter(|v| (done >> (4 * v)) & 0xF != 0) {
            //SAFETY: stores of vectors into four 8-byte values at offset `4 * v` of arrays of `LANES_IN_ALL`
            unsafe {
                _mm256_storeu_pd(zx.as_mut_ptr().add(4 * v), vzx[v]);
                _mm256_storeu_pd(zy.as_mut_ptr().add(4 * v), vzy[v]);
                _mm256_storeu_si256(iterations.as_mut_ptr().add(4 * v).cast::<__m256i>(), vit[v]);
                _mm256_storeu_si256(blocks.as_mut_ptr().add(4 * v).cast::<__m256i>(), vblocks[v]);
            }
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
        for v in (0..VECTORS).filter(|v| (done >> (4 * v)) & 0xF != 0) {
            let finished = _mm256_castsi256_pd(integer_vector(&LANE_MASKS[((done >> (4 * v)) & 0xF) as usize]));
            (vcx[v], vcy[v]) = (load(&cx, v), load(&cy, v));
            (vzx[v], vzy[v], vsx[v], vsy[v]) = (_mm256_andnot_pd(finished, vzx[v]), _mm256_andnot_pd(finished, vzy[v]), _mm256_andnot_pd(finished, vsx[v]), _mm256_andnot_pd(finished, vsy[v]));
            (vx2[v], vy2[v]) = (_mm256_andnot_pd(finished, vx2[v]), _mm256_andnot_pd(finished, vy2[v])); //z = 0, so its squares are 0
            vit[v] = _mm256_castpd_si256(_mm256_andnot_pd(finished, _mm256_castsi256_pd(vit[v])));
            vblocks[v] = _mm256_castpd_si256(_mm256_andnot_pd(finished, _mm256_castsi256_pd(vblocks[v])));
            //A lane that got a point is live and active; one that did not is neither
            vlive[v] = _mm256_castsi256_pd(load_integers(&live, v));
            vactive[v] = _mm256_or_pd(_mm256_andnot_pd(finished, vactive[v]), _mm256_and_pd(finished, vlive[v]));
        }
    }
    points.len()
}

/// A vector of the four integers of `values`
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)] // An unaligned load (`loadu`)
fn integer_vector(values: &[i64; 4]) -> __m256i {
    //SAFETY: an unaligned load of four 8-byte values from an array of four
    unsafe { _mm256_loadu_si256(values.as_ptr().cast::<__m256i>()) }
}

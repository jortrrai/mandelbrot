// The escape loop of `MandelbrotSet::iterate_lanes_smooth_generic` in f64, with refilling invocations: an invocation takes a point from `next_point`, and when that
// point is done it takes the next one at once, so the invocations of a warp do not wait for the slowest point of the warp (the warp runs as long as any of its
// invocations has a point). Every invocation keeps the step count and the cycle detection of its own point, so every point gets the arithmetic of the lockstep loop.
// It is compiled to SPIR-V with NoContraction on every floating-point operation (see `gpu.rs`), so the driver cannot fuse a multiplication and an addition:
// every operation rounds exactly like the CPU's, and the results are bit for bit the same.

struct Params {
    radius_squared: f64,
    max_iterations: u32,
    count: u32,
    check_interior: u32,
    pad: u32,
}

struct Point {
    x: f64,
    y: f64,
}

struct Outcome {
    zx: f64,
    zy: f64,
    iterations: u32,
    escaped: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> outcomes: array<Outcome>;
@group(0) @binding(3) var<storage, read_write> next_point: atomic<u32>;

// `MandelbrotSet::is_known_interior`: the main cardioid and the period-two bulb, with the same margin
fn known_interior(x: f64, y: f64) -> bool {
    let y_squared = y * y;
    let q = (x - 0.25lf) * (x - 0.25lf) + y_squared;
    let in_cardioid = q * (q + (x - 0.25lf)) <= 0.25lf * y_squared - 1e-9lf;
    let in_bulb = (x + 1.0lf) * (x + 1.0lf) + y_squared <= 0.0625lf - 1e-9lf;
    return in_cardioid || in_bulb;
}

// The next point that needs the loop (points in the cardioid or the period-two bulb get their outcome at once), or `params.count` when there is none
fn take_point() -> u32 {
    loop {
        let index = atomicAdd(&next_point, 1u);
        if (index >= params.count) {
            return params.count;
        }
        if (params.check_interior != 0u && known_interior(points[index].x, points[index].y)) {
            outcomes[index] = Outcome(0.0lf, 0.0lf, params.max_iterations, 0u);
            continue;
        }
        return index;
    }
    return params.count; // Not reached: naga wants a return after the loop
}

@compute @workgroup_size(64)
fn main() {
    let max_iterations = params.max_iterations;
    var index = take_point();
    var cx = 0.0lf;
    var cy = 0.0lf;
    if (index < params.count) {
        cx = points[index].x;
        cy = points[index].y;
    }
    var zx = 0.0lf;
    var zy = 0.0lf;
    var iterations = 0u;
    var running = true;
    var saved_x = 0.0lf;
    var saved_y = 0.0lf;
    var step = 0u;
    var blocks = 0u;
    loop {
        if (index >= params.count) {
            break;
        }
        // One block of steps, shorter at the end like the last block of the lockstep loop
        let steps = min(4u, max_iterations - step);
        for (var k = 0u; k < steps; k++) {
            // The same arithmetic as z = z.squared().add(c): (zx * zx - zy * zy) + cx and (2 * zx) * zy + cy
            let new_x = zx * zx - zy * zy + cx;
            let new_y = 2.0lf * zx * zy + cy;
            zx = new_x;
            zy = new_y;
            if (running && new_x * new_x + new_y * new_y > params.radius_squared) {
                running = false;
            }
            if (running) {
                iterations += 1u;
            }
        }
        step += 4u;
        blocks += 1u;
        // Escaped in this block: z at the end of the block gives the smooth position
        var done = !running;
        var escaped = !running;
        // Brent's cycle detection once per block: an orbit that is exactly where it was at an earlier block repeats forever, so it never escapes
        if (running && zx == saved_x && zy == saved_y) {
            done = true;
            iterations = max_iterations;
        }
        // The lockstep loop saves z after 1, 3, 7, 15, ... blocks (the gap doubles): after `blocks` blocks when `blocks + 1` is a power of two
        if ((blocks & (blocks + 1u)) == 0u) {
            saved_x = zx;
            saved_y = zy;
        }
        if (step >= max_iterations) {
            done = true;
        }
        if (done) {
            outcomes[index] = Outcome(select(0.0lf, zx, escaped), select(0.0lf, zy, escaped), iterations, select(0u, 1u, escaped && iterations < max_iterations));
            index = take_point();
            if (index < params.count) {
                cx = points[index].x;
                cy = points[index].y;
            }
            zx = 0.0lf;
            zy = 0.0lf;
            iterations = 0u;
            running = true;
            saved_x = 0.0lf;
            saved_y = 0.0lf;
            step = 0u;
            blocks = 0u;
        }
    }
}

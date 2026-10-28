// The escape loop of `MandelbrotSet::iterate_lanes_smooth_generic`, one point per invocation, in f64.
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

// `MandelbrotSet::is_known_interior`: the main cardioid and the period-two bulb, with the same margin
fn known_interior(x: f64, y: f64) -> bool {
    let y_squared = y * y;
    let q = (x - 0.25lf) * (x - 0.25lf) + y_squared;
    let in_cardioid = q * (q + (x - 0.25lf)) <= 0.25lf * y_squared - 1e-9lf;
    let in_bulb = (x + 1.0lf) * (x + 1.0lf) + y_squared <= 0.0625lf - 1e-9lf;
    return in_cardioid || in_bulb;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.count) {
        return;
    }
    let cx = points[index].x;
    let cy = points[index].y;
    let max_iterations = params.max_iterations;
    if (params.check_interior != 0u && known_interior(cx, cy)) {
        outcomes[index] = Outcome(0.0lf, 0.0lf, max_iterations, 0u);
        return;
    }
    var zx = 0.0lf;
    var zy = 0.0lf;
    var iterations = 0u;
    var running = true;
    var escaped = false;
    var escape_x = 0.0lf;
    var escape_y = 0.0lf;
    var saved_x = 0.0lf;
    var saved_y = 0.0lf;
    var blocks_since_saved = 0u;
    var blocks_until_save = 1u;
    var step = 0u;
    while (step < max_iterations && running) {
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
        let block_start = step;
        let block_end = min(step + 4u, max_iterations);
        step += 4u;
        // Escaped in this block: keep z at the end of the block for the smooth position
        if (!running && !escaped && iterations >= block_start && iterations < block_end) {
            escaped = true;
            escape_x = zx;
            escape_y = zy;
        }
        // Brent's cycle detection once per block: an orbit that is exactly where it was at an earlier block repeats forever, so it never escapes
        if (running && zx == saved_x && zy == saved_y) {
            running = false;
            iterations = max_iterations;
        }
        blocks_since_saved += 1u;
        if (blocks_since_saved == blocks_until_save) {
            saved_x = zx;
            saved_y = zy;
            blocks_since_saved = 0u;
            blocks_until_save = select(blocks_until_save * 2u, 4294967295u, blocks_until_save >= 2147483648u);
        }
    }
    outcomes[index] = Outcome(escape_x, escape_y, iterations, select(0u, 1u, escaped && iterations < max_iterations));
}

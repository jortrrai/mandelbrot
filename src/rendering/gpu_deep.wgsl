// The perturbation loop of `rendering::deep::position_from` (issue #263): for every pixel, its difference `z` from the reference orbit `Z`, with the skips of
// the bilinear approximations (BLA) of `Reference::skip` and the rebasing of the CPU, in the same order of operations. Compiled to SPIR-V with NoContraction
// on every floating-point operation (see `gpu.rs`), so every step rounds as the CPU's does: the outcome of every pixel is the CPU's, bit for bit (the
// self-test checks that). The CPU turns an outcome into the smooth position (`MandelbrotSet::smooth_position_of`).

struct Params {
    radius_squared: f64,
    max_iterations: u32,
    count: u32,
    // The levels of the table and the number of points of the orbit
    levels: u32,
    orbit_length: u32,
    // Where every level of the table starts, and how many approximations it has
    level_start: array<u32, 32>,
    level_length: array<u32, 32>,
}

struct Point {
    x: f64,
    y: f64,
}

// `z ← A·z + B·c`, valid while `|z|² < radius_squared`
struct Bla {
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    radius_squared: f64,
}

struct Outcome {
    zx: f64,
    zy: f64,
    steps: u32,
    escaped: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> orbit: array<Point>;
@group(0) @binding(2) var<storage, read> table: array<Bla>;
@group(0) @binding(3) var<storage, read> offsets: array<Point>;
@group(0) @binding(4) var<storage, read_write> outcomes: array<Outcome>;
// The rounding residual of every point of the orbit (`deep::Residual`, as plain numbers: the GPU's views are not deep enough for it to underflow)
@group(0) @binding(5) var<storage, read> residuals: array<Point>;

// `deep::two_product`: a·b as a sum of two numbers, exactly (Dekker, Veltkamp's split; NoContraction keeps every step as written)
fn two_product(a: f64, b: f64) -> vec2<f64> {
    let ta = 134217729.0lf * a;
    let a_high = ta - (ta - a);
    let a_low = a - a_high;
    let tb = 134217729.0lf * b;
    let b_high = tb - (tb - b);
    let b_low = b - b_high;
    let product = a * b;
    return vec2<f64>(product, ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low);
}

// `deep::two_sum`: a + b as a sum of two numbers, exactly (Knuth)
fn two_sum(a: f64, b: f64) -> vec2<f64> {
    let sum = a + b;
    let b_virtual = sum - a;
    return vec2<f64>(sum, (a - (sum - b_virtual)) + (b - b_virtual));
}

// `deep::outside` for a plain difference: whether Z + ε + z is outside the circle, exactly within 10⁻¹² of it (the tip c = -2)
fn outside(full: f64, at: u32, zx: f64, zy: f64) -> bool {
    if (full < params.radius_squared * (1.0lf - 1e-12lf)) {
        return false;
    }
    if (full > params.radius_squared * (1.0lf + 1e-12lf)) {
        return true;
    }
    let ox = orbit[at].x;
    let oy = orbit[at].y;
    let dx = residuals[at].x + zx;
    let dy = residuals[at].y + zy;
    let xx = two_product(ox, ox);
    let yy = two_product(oy, oy);
    let sum = two_sum(xx.x, yy.x);
    let beyond = two_sum(sum.x, -params.radius_squared);
    let away = beyond.x + ((sum.y + beyond.y) + (xx.y + yy.y));
    return away + (2.0lf * (ox * dx + oy * dy) + (dx * dx + dy * dy)) > 0.0lf;
}

// `Reference::skip`: the index in `table` of the longest approximation from step `at` for a difference of `|z|² = size_squared`, of at most `left` steps, and
// its steps (0: none)
fn skip(at: u32, size_squared: f64, left: u32) -> vec2<u32> {
    if (at == 0u || params.levels == 0u) {
        return vec2<u32>(0u, 0u);
    }
    // (The CPU's trailing zeros of 0 are 64: more than the levels, as WGSL's 32 is)
    let aligned = min(countTrailingZeros(at - 1u), params.levels);
    for (var level = aligned; level >= 1u; level--) {
        let steps = 1u << level;
        let k = (at - 1u) >> level;
        if (k >= params.level_length[level - 1u]) {
            continue;
        }
        let index = params.level_start[level - 1u] + k;
        if (steps <= left && size_squared < table[index].radius_squared) {
            return vec2<u32>(index, steps);
        }
    }
    return vec2<u32>(0u, 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 64u;
    for (var index = id.x; index < params.count; index += stride) {
        let dcx = offsets[index].x;
        let dcy = offsets[index].y;
        var zx = 0.0lf;
        var zy = 0.0lf;
        var at = 0u;
        var step = 0u;
        var outcome = Outcome(0.0lf, 0.0lf, params.max_iterations, 0u);
        loop {
            if (step >= params.max_iterations) {
                break;
            }
            let skipped = skip(at, zx * zx + zy * zy, params.max_iterations - step);
            if (skipped.y != 0u) {
                let bla = table[skipped.x];
                // z ← A·z + B·c, for `steps` steps at once (the CPU's order of operations)
                let new_x = bla.ax * zx - bla.ay * zy + (bla.bx * dcx - bla.by * dcy);
                let new_y = bla.ax * zy + bla.ay * zx + (bla.bx * dcy + bla.by * dcx);
                zx = new_x;
                zy = new_y;
                at += skipped.y;
                step += skipped.y;
            } else {
                let ox = orbit[at].x;
                let oy = orbit[at].y;
                // z ← 2·Z·z + z² + c
                let new_x = 2.0lf * (ox * zx - oy * zy) + (zx * zx - zy * zy) + dcx;
                let new_y = 2.0lf * (ox * zy + oy * zx) + 2.0lf * zx * zy + dcy;
                zx = new_x;
                zy = new_y;
                at += 1u;
                step += 1u;
            }
            // The pixel's orbit
            let fx = orbit[at].x + zx;
            let fy = orbit[at].y + zy;
            let full = fx * fx + fy * fy;
            if (outside(full, at, zx, zy)) {
                outcome = Outcome(fx, fy, step, 1u);
                break;
            }
            // Rebasing: the pixel's orbit is closer to 0 than its difference, or the reference orbit ends
            if (full < zx * zx + zy * zy || at + 1u >= params.orbit_length) {
                zx = fx;
                zy = fy;
                at = 0u;
            }
        }
        outcomes[index] = outcome;
    }
}

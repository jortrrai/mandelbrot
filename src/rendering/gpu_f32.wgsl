// The loop of gpu_fast.wgsl in plain f32, for frames that move at shallow zoom (see `Render::f32_gpu`): not the CPU's numbers, and in chaotic parts of the set
// more pixels get another iteration count than with the double-floats; the points are computed in f64 as there, then rounded to f32.
struct Params {
    min_x: f64,
    min_y: f64,
    increment_x: f64,
    increment_y: f64,
    radius_squared: f64,
    max_iterations: u32,
    count: u32,
    check_interior: u32,
    smooth_coloring: u32,
    // Where pixel (0, 0) of the batch is in the anchor plane, see `SampleState`
    origin_x: i32,
    origin_y: i32,
    sample: u32,
    pad: u32,
}
// The position of a point, see `FastOutcome`
struct Outcome { whole: u32, fraction: f32, }

@group(0) @binding(0) var<storage, read> params: Params;
// A pixel of the batch: x in the low 16 bits, y in the high 16 bits
@group(0) @binding(1) var<storage, read> pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> outcomes: array<Outcome>;
@group(0) @binding(3) var<storage, read_write> next_point: atomic<u32>;

// `jitter::splitmix64`
fn splitmix64(value: u64) -> u64 {
    var z = value + 0x9E3779B97F4A7C15lu;
    z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9lu;
    z = (z ^ (z >> 27u)) * 0x94D049BB133111EBlu;
    return z ^ (z >> 31u);
}

// 53 random bits as a number in [0, 1), like `jitter::sample_jitter`
fn to_unit(bits: u64) -> f64 {
    return f64(bits >> 11u) / 9007199254740992.0lf;
}

// `progressive::sample_point`: the point of sample `params.sample` of pixel `index` of the batch
fn sample_point(index: u32) -> vec2<f64> {
    let pixel = pixels[index];
    let anchor_x = i64(pixel & 0xFFFFu) + i64(params.origin_x);
    let anchor_y = i64(pixel >> 16u) + i64(params.origin_y);
    let h = splitmix64(splitmix64(splitmix64(bitcast<u64>(anchor_x)) ^ bitcast<u64>(anchor_y)) ^ u64(params.sample));
    let x = fma(f64(anchor_x) + to_unit(h), params.increment_x, params.min_x);
    let y = -fma(f64(anchor_y) + to_unit(splitmix64(h)), params.increment_y, params.min_y);
    return vec2<f64>(x, y);
}

// `MandelbrotSet::is_known_interior`, in f64 like gpu.wgsl
fn known_interior(x: f64, y: f64) -> bool {
    let y_squared = y * y;
    let q = (x - 0.25lf) * (x - 0.25lf) + y_squared;
    let in_cardioid = q * (q + (x - 0.25lf)) <= 0.25lf * y_squared - 1e-9lf;
    let in_bulb = (x + 1.0lf) * (x + 1.0lf) + y_squared <= 0.0625lf - 1e-9lf;
    return in_cardioid || in_bulb;
}

// The next point that needs the loop, or `params.count` when there is none, see gpu.wgsl
fn take_point() -> u32 {
    loop {
        let index = atomicAdd(&next_point, 1u);
        if (index >= params.count) {
            return params.count;
        }
        if (params.check_interior != 0u) {
            let c = sample_point(index);
            if (known_interior(c.x, c.y)) {
                outcomes[index] = Outcome(params.max_iterations, 0.0);
                continue;
            }
        }
        return index;
    }
    return params.count; // Not reached: naga wants a return after the loop
}

// The f32 closest to `x` (as a double-float with a low part of 0)
fn split(x: f64) -> vec2<f32> {
    return vec2<f32>(f32(x), 0.0);
}

// The arithmetic of the loop in plain f32: the same functions as the double-floats of gpu_fast.wgsl, with only a high part (the low part is always 0). f32 has
// 24 bits instead of about 48: about 16 times as fast on an RTX 5070 Ti, enough for views whose neighboring pixels are far apart (see `F32_GPU_STEPS`), and
// only used for frames that move (see `Render::f32_gpu`)
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x + b.x, 0.0);
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x, 0.0);
}

fn df_sqr(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * a.x, 0.0);
}

// The position of a point that stopped after `iterations` steps, with z at the end of its block: `MandelbrotSet::position_after_loop` and `smooth_position`, the
// orbit continued in double-floats and the two logarithms in f32 (their error, about 10⁻⁷ of an iteration, is far below the 1/256 steps of the coloring)
fn finished(iterations: u32, escaped: bool, z_x: vec2<f32>, z_y: vec2<f32>, cx: vec2<f32>, cy: vec2<f32>) -> Outcome {
    if (!escaped || params.smooth_coloring == 0u) {
        return Outcome(iterations, 0.0);
    }
    var zx = z_x;
    var zy = z_y;
    // `block_end(iterations + 1)`: the steps of z
    var steps = min((iterations + 1u + 3u) / 4u * 4u, params.max_iterations);
    for (var k = 0u; k < 32u; k++) {
        if (zx.x * zx.x + zy.x * zy.x >= 65536.0) {
            break;
        }
        let xx = df_sqr(zx);
        let yy = df_sqr(zy);
        let xy = df_mul(zx, zy);
        let new_x = df_add(df_add(xx, -yy), cx);
        let new_y = df_add(2.0 * xy, cy);
        zx = new_x;
        zy = new_y;
        steps += 1u;
    }
    let modulus_squared = zx.x * zx.x + zy.x * zy.x;
    if (modulus_squared <= 1.0) {
        return Outcome((steps - 1u) | 0x80000000u, 0.0);
    }
    // `steps - 1 - log2(ln |z|)`. The top bit marks a smooth position: the CPU clamps it to [0, max_iterations - 1], see `FastOutcome::position`
    return Outcome((steps - 1u) | 0x80000000u, -log2(0.5 * log(modulus_squared)));
}

@compute @workgroup_size(64)
fn main() {
    let max_iterations = params.max_iterations;
    // The escape check compares the high parts only: it decides when to stop, not the value of z
    let radius_squared = f32(params.radius_squared);
    var index = take_point();
    var cx = vec2<f32>(0.0);
    var cy = vec2<f32>(0.0);
    if (index < params.count) {
        let c = sample_point(index);
        cx = split(c.x);
        cy = split(c.y);
    }
    var zx = vec2<f32>(0.0);
    var zy = vec2<f32>(0.0);
    var iterations = 0u;
    var running = true;
    var saved_x = vec2<f32>(0.0);
    var saved_y = vec2<f32>(0.0);
    var step = 0u;
    var blocks = 0u;
    loop {
        if (index >= params.count) {
            break;
        }
        // One block of steps with the cycle check after it, exactly as in gpu.wgsl
        let steps = min(4u, max_iterations - step);
        for (var k = 0u; k < steps; k++) {
            // z = z² + c: (zx² - zy²) + cx and 2 zx zy + cy
            let xx = df_sqr(zx);
            let yy = df_sqr(zy);
            let xy = df_mul(zx, zy);
            let new_x = df_add(df_add(xx, -yy), cx);
            let new_y = df_add(2.0 * xy, cy);
            zx = new_x;
            zy = new_y;
            if (running && new_x.x * new_x.x + new_y.x * new_y.x > radius_squared) {
                running = false;
            }
            if (running) {
                iterations += 1u;
            }
        }
        step += 4u;
        blocks += 1u;
        var done = !running;
        var escaped = !running;
        if (running && all(zx == saved_x) && all(zy == saved_y)) {
            done = true;
            iterations = max_iterations;
        }
        if ((blocks & (blocks + 1u)) == 0u) {
            saved_x = zx;
            saved_y = zy;
        }
        if (step >= max_iterations) {
            done = true;
        }
        if (done) {
            outcomes[index] = finished(iterations, escaped && iterations < max_iterations, zx, zy, cx, cy);
            index = take_point();
            if (index < params.count) {
                let c = sample_point(index);
                cx = split(c.x);
                cy = split(c.y);
            }
            zx = vec2<f32>(0.0);
            zy = vec2<f32>(0.0);
            iterations = 0u;
            running = true;
            saved_x = vec2<f32>(0.0);
            saved_y = vec2<f32>(0.0);
            step = 0u;
            blocks = 0u;
        }
    }
}

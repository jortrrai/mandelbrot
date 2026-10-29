// The loop of gpu.wgsl in double-float arithmetic: a number is the unevaluated sum hi + lo of two f32, with about 48 bits of mantissa instead of the 53 of an f64.
// GPUs like the RTX 5070 Ti run f32 64 times as fast as f64, so the dozen or so f32 operations of a double-float operation are still about three times as fast
// (see `gpu.rs`). It is not bit for bit the f64 loop: see `Render::fast_gpu`. The algorithms are those of Dekker (1971) and the double-double library of Hida, Li
// and Bailey (QD, 2001): `two_sum` and `quick_two_sum` give the exact rounding error of an addition, the fused multiply-add that of a multiplication. They need
// every operation to round as written, so this shader also gets NoContraction on every operation (see `gpu.rs`).
struct Params {
    radius_squared: f64,
    max_iterations: u32,
    count: u32,
    check_interior: u32,
    pad: u32,
}
struct Point { x: f64, y: f64, }
struct Outcome { zx: f64, zy: f64, iterations: u32, escaped: u32, }

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> outcomes: array<Outcome>;
@group(0) @binding(3) var<storage, read_write> next_point: atomic<u32>;

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
        if (params.check_interior != 0u && known_interior(points[index].x, points[index].y)) {
            outcomes[index] = Outcome(0.0lf, 0.0lf, params.max_iterations, 0u);
            continue;
        }
        return index;
    }
    return params.count; // Not reached: naga wants a return after the loop
}

// The double-float closest to `x`
fn split(x: f64) -> vec2<f32> {
    let hi = f32(x);
    return vec2<f32>(hi, f32(x - f64(hi)));
}

// The f64 of a double-float, for the smooth position that the CPU computes
fn join(a: vec2<f32>) -> f64 {
    return f64(a.x) + f64(a.y);
}

// a + b and its rounding error, when |a| >= |b|
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    return vec2<f32>(s, b - (s - a));
}

// a + b and its rounding error
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    return vec2<f32>(s, (a - (s - v)) + (b - v));
}

// The sum of two double-floats (the faster of the two QD additions: the low parts are added without their own rounding error)
fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    return quick_two_sum(s.x, s.y + (a.y + b.y));
}

// The product of two double-floats
fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = a.x * b.x;
    let e = fma(a.x, b.x, -p) + (a.x * b.y + a.y * b.x);
    return quick_two_sum(p, e);
}

// The square of a double-float
fn df_sqr(a: vec2<f32>) -> vec2<f32> {
    let p = a.x * a.x;
    let e = fma(a.x, a.x, -p) + 2.0 * a.x * a.y;
    return quick_two_sum(p, e);
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
        cx = split(points[index].x);
        cy = split(points[index].y);
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
            outcomes[index] = Outcome(select(0.0lf, join(zx), escaped), select(0.0lf, join(zy), escaped), iterations, select(0u, 1u, escaped && iterations < max_iterations));
            index = take_point();
            if (index < params.count) {
                cx = split(points[index].x);
                cy = split(points[index].y);
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

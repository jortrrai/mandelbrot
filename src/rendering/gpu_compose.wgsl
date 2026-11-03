// A frame of a dive composed from keyframe images (see `gui::dive_keyframes`), bit for bit as the CPU composes it: for every pixel and every moment of its
// motion blur, four bilinear taps of the moment's keyframe in f32 (every operation rounds as on the CPU: NoContraction), their average rounded to a color,
// and the average of those colors over the moments, rounded (the number of moments is a power of two, so multiplying by its inverse divides exactly, as the CPU
// does).

struct Params {
    width: u32,
    height: u32,
    // The width of a keyframe
    big_width: u32,
    moments: u32,
    // 1 / moments
    inverse: f32,
}

// A moment: where its keyframe starts in `images`, and where its taps start in `taps`: two rows of `width` column taps (tap x 0.25, then 0.75), then two of
// `height` row taps (tap y 0.25, then 0.75)
struct Moment {
    image: u32,
    taps: u32,
    unused_a: u32,
    unused_b: u32,
}

struct Tap {
    first: u32,
    second: u32,
    share: f32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> moments: array<Moment>;
@group(0) @binding(2) var<storage, read> taps: array<Tap>;
@group(0) @binding(3) var<storage, read> images: array<u32>;
// The color of every pixel: red | green << 8 | blue << 16
@group(0) @binding(4) var<storage, read_write> colors: array<u32>;

fn texel(index: u32) -> vec3<f32> {
    let packed = images[index];
    return vec3<f32>(f32(packed & 255u), f32((packed >> 8u) & 255u), f32((packed >> 16u) & 255u));
}

// `bilinear_at` of the CPU
fn bilinear(image: u32, column: Tap, row: Tap) -> vec3<f32> {
    let width = params.big_width;
    let a = texel(image + row.first * width + column.first);
    let b = texel(image + row.first * width + column.second);
    let c = texel(image + row.second * width + column.first);
    let d = texel(image + row.second * width + column.second);
    let upper = a + (b - a) * column.share;
    let lower = c + (d - c) * column.share;
    return upper + (lower - upper) * row.share;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let pixels = params.width * params.height;
    let stride = groups.x * 64u;
    for (var pixel = id.x; pixel < pixels; pixel += stride) {
        let x = pixel % params.width;
        let y = pixel / params.width;
        var total = vec3<f32>(0.0, 0.0, 0.0);
        for (var index = 0u; index < params.moments; index++) {
            let moment = moments[index];
            let columns = moment.taps;
            let rows = moment.taps + 2u * params.width;
            var sum = vec3<f32>(0.0, 0.0, 0.0);
            // The taps in the order of the CPU: (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
            for (var tap = 0u; tap < 4u; tap++) {
                let column = taps[columns + (tap % 2u) * params.width + x];
                let row = taps[rows + (tap / 2u) * params.height + y];
                sum = sum + bilinear(moment.image, column, row);
            }
            // (value / 4).round() of the CPU: a quarter is exact, and floor(x + 0.5) rounds half away from zero for these sums
            total = total + floor(sum * 0.25 + vec3<f32>(0.5, 0.5, 0.5));
        }
        // ((sum / moments).round() of the CPU)
        let color = vec3<u32>(floor(total * params.inverse + vec3<f32>(0.5, 0.5, 0.5)));
        colors[pixel] = color.x | (color.y << 8u) | (color.z << 16u);
    }
}

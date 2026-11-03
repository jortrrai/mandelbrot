// The adaptive mask of `adaptive::mask_where` on the GPU (issue #257), for a render without a region: integer arithmetic only, so the mask is the CPU's bit for
// bit. Step 0 marks a pixel whose first two samples differ, or which differs from any of its 8 neighbors (by more than the threshold, the sum of the differences
// of the channels); step 1 marks every pixel that has a marked pixel in its 3 x 3 block. The colors are the CPU's 3-byte colors, read from words.

struct Params {
    width: u32,
    height: u32,
    threshold: u32,
    step: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> first: array<u32>;
@group(0) @binding(2) var<storage, read> second: array<u32>;
@group(0) @binding(3) var<storage, read_write> marked: array<u32>;
@group(0) @binding(4) var<storage, read_write> mask: array<u32>;

// Byte `at` of a byte array kept in words
fn byte_of_first(at: u32) -> u32 {
    return (first[at / 4u] >> ((at % 4u) * 8u)) & 255u;
}

fn byte_of_second(at: u32) -> u32 {
    return (second[at / 4u] >> ((at % 4u) * 8u)) & 255u;
}

fn first_color(index: u32) -> vec3<u32> {
    return vec3<u32>(byte_of_first(3u * index), byte_of_first(3u * index + 1u), byte_of_first(3u * index + 2u));
}

fn second_color(index: u32) -> vec3<u32> {
    return vec3<u32>(byte_of_second(3u * index), byte_of_second(3u * index + 1u), byte_of_second(3u * index + 2u));
}

// `adaptive::color_distance`
fn distance(a: vec3<u32>, b: vec3<u32>) -> u32 {
    let d = max(a, b) - min(a, b);
    return d.x + d.y + d.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let pixels = params.width * params.height;
    let stride = groups.x * 64u;
    for (var index = id.x; index < pixels; index += stride) {
        let x = index % params.width;
        let y = index / params.width;
        let left = select(x - 1u, 0u, x == 0u);
        let right = min(x + 1u, params.width - 1u);
        let top = select(y - 1u, 0u, y == 0u);
        let bottom = min(y + 1u, params.height - 1u);
        if (params.step == 0u) {
            let color = second_color(index);
            var flag = distance(first_color(index), color) > params.threshold;
            for (var ny = top; ny <= bottom; ny++) {
                for (var nx = left; nx <= right; nx++) {
                    flag = flag || distance(color, second_color(ny * params.width + nx)) > params.threshold;
                }
            }
            marked[index] = select(0u, 1u, flag);
        } else {
            var any = false;
            for (var ny = top; ny <= bottom; ny++) {
                for (var nx = left; nx <= right; nx++) {
                    any = any || marked[ny * params.width + nx] != 0u;
                }
            }
            mask[index] = select(0u, 1u, any);
        }
    }
}

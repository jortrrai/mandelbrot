//! The escape-time iteration on the GPU, bit for bit the same as on the CPU.
//!
//! The loop of `MandelbrotSet::position_lanes` runs on the GPU in f64 (`gpu.wgsl`): a fixed number of invocations (`WORKGROUPS`) take points from a counter, and an
//! invocation whose point is done takes the next one at once, so the invocations of a warp do not wait for its slowest point (measured: Seahorse Valley at
//! 1200 x 800 on an RTX 5070 Ti in 88 ms instead of 148 ms with one point per invocation; 256 workgroups were the best of 128 to 1024). A GPU driver may fuse a multiplication and an addition
//! into one fused multiply-add, which rounds differently: in chaotic parts of the set that changes the iteration count of many points (measured on an RTX 5070 Ti:
//! 0.6% of the pixels of Seahorse Valley, 64% at scale 10⁻¹³). So the shader is compiled to SPIR-V here and every floating-point operation gets the `NoContraction`
//! decoration, which forbids that; the SPIR-V is passed to the driver as is. Then every operation rounds like the CPU's and the outcome is the same, bit for bit.
//!
//! The GPU only runs the expensive loop and returns the iteration count, whether the point escaped, and `z` at the end of its block of steps; the CPU turns that into
//! the position with the same code as its own loop (`MandelbrotSet::position_after_loop`), so the smooth position (two logarithms, which SPIR-V has no f64 version
//! of) is computed by the CPU exactly as before.
//!
//! There is also a fast loop (`gpu_fast.wgsl`) in double-float arithmetic: every number is the sum of two f32, about 48 bits instead of 53, and GPUs run f32 much faster
//! than f64 (64 times on an RTX 5070 Ti). Measured at 1200 x 800 on an RTX 5070 Ti: Seahorse Valley in 29 ms instead of 88 ms, the shipped places 1.3 to 3.1 times as
//! fast. It is not bit for bit the CPU's: 0.02% to 1.5% of the pixels of most shipped places get another iteration count, up to 25% in the most chaotic ones (where
//! f64 with fused multiply-adds already changes 15 to 19%, so f64 cannot pin those pixels down either). The renderer only uses it when asked, see `Render::fast_gpu`.
//!
//! `GpuIterator::new` checks a set of points on the GPU against the CPU before it is used, and refuses when anything differs, when there is no suitable GPU,
//! or when it lacks f64 shaders or SPIR-V passthrough: the renderer then stays on the CPU.

use std::borrow::Cow;


use crate::{complex::Complex, complex_plane::{ComplexPlane, View}, mandelbrot_set::MandelbrotSet};

/// About the most points of one dispatch: short dispatches keep the GPU responsive for the window (the renderer aims at it from the points per row it measured, so a
/// dispatch can be somewhat larger)
pub(crate) const BATCH: usize = 1 << 18;

/// The workgroups of 64 invocations of a dispatch: each invocation takes points until there are none left (see `gpu.wgsl`). Enough to keep the GPU busy, few enough
/// that an invocation gets several points of a batch to refill with
const WORKGROUPS: u32 = 256;

/// The outcome of the loop for one point, as the shader writes it
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Outcome {
    zx: f64,
    zy: f64,
    iterations: u32,
    escaped: u32,
}

impl Outcome {
    /// The position of the point `c` with this outcome: exactly `m.position(c)` (see the module)
    pub(crate) fn position(&self, m: &MandelbrotSet, c: &Complex) -> f64 {
        m.position_after_loop(self.iterations, self.escaped != 0, (self.zx, self.zy), c)
    }
}

/// A dispatch that runs on the GPU, see `GpuIterator::submit`
pub(crate) struct Submitted {
    read: wgpu::Buffer,
    index: wgpu::SubmissionIndex,
}

/// A GPU that iterates points exactly like the CPU, see the module
pub struct GpuIterator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The exact loop (`gpu.wgsl`) and the fast double-float loop (`gpu_fast.wgsl`)
    exact: wgpu::ComputePipeline,
    fast: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    name: String,
}

impl GpuIterator {
    /// Opens the most powerful GPU with f64 shaders and SPIR-V passthrough, and checks it against the CPU (see the module).
    ///
    /// # Errors
    /// The reason the GPU cannot be used: no adapter, a missing feature, a failed request, or an outcome that differs from the CPU's
    pub fn new() -> Result<GpuIterator, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|err| format!("no GPU: {err}"))?;
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let needed = wgpu::Features::SHADER_F64 | wgpu::Features::PASSTHROUGH_SHADERS;
        if !adapter.features().contains(needed) {
            return Err(format!("{name} lacks f64 shaders or SPIR-V passthrough"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: needed, required_limits: adapter.limits(), ..Default::default() })).map_err(|err| format!("{name}: {err}"))?;
        //An error outside a dispatch is logged instead of ending the program (the default); the errors of a dispatch are returned by `submit`
        device.on_uncaptured_error(std::sync::Arc::new(|error| eprintln!("GPU error: {error}")));
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mandelbrot"), entries: &[entry(0, true), entry(1, true), entry(2, false), entry(3, false)] });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mandelbrot"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |source: &str| -> Result<wgpu::ComputePipeline, String> {
            let words = spirv_without_contraction(source)?;
            //SAFETY: the SPIR-V is naga's validated output of the shader, with only NoContraction decorations added
            let shader = unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("mandelbrot"),
                    entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint { name: "main".into(), workgroup_size: (64, 1, 1) }]),
                    spirv: Some(Cow::Owned(words)),
                    ..Default::default()
                })
            };
            Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("mandelbrot"), layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None }))
        };
        let (exact, fast) = (pipeline(include_str!("gpu.wgsl"))?, pipeline(include_str!("gpu_fast.wgsl"))?);
        let gpu = GpuIterator { device, queue, exact, fast, layout, name };
        gpu.self_test()?;
        Ok(gpu)
    }

    /// The name and the backend of the GPU
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The position of every point: exactly `m.position(point)`, or with `fast` that of the fast loop (see the module)
    ///
    /// # Errors
    /// The GPU failed (for example the device was lost)
    pub fn positions(&self, m: &MandelbrotSet, points: &[Complex], fast: bool) -> Result<Vec<f64>, String> {
        let mut positions = vec![0.0; points.len()];
        for (batch, out) in points.chunks(BATCH).zip(positions.chunks_mut(BATCH)) {
            let outcomes = self.finish(self.submit(m, batch, fast)?)?;
            //The positions (two logarithms for every escaped point with smooth coloring) on all render threads
            let chunk = batch.len().div_ceil(super::render_thread_count()).max(1);
            std::thread::scope(|scope| {
                for ((points, outcomes), out) in batch.chunks(chunk).zip(outcomes.chunks(chunk)).zip(out.chunks_mut(chunk)) {
                    scope.spawn(move || {
                        for ((c, outcome), position) in points.iter().zip(outcomes).zip(out) {
                            *position = outcome.position(m, c);
                        }
                    });
                }
            });
        }
        Ok(positions)
    }

    /// Starts one dispatch of the exact loop (or with `fast` the fast loop) over `points` (about `BATCH` at most) and returns at once: the GPU runs it while the caller
    /// prepares the next one, see `finish`
    pub(crate) fn submit(&self, m: &MandelbrotSet, points: &[Complex], fast: bool) -> Result<Submitted, String> {
        let count = u32::try_from(points.len()).map_err(|_| String::from("too many points"))?;
        let mut params = Vec::with_capacity(24);
        params.extend((m.orbit_radius * m.orbit_radius).to_le_bytes());
        for value in [m.max_iterations, count, u32::from(m.orbit_radius >= 2.0), 0] {
            params.extend(value.to_le_bytes());
        }
        let coordinates: Vec<f64> = points.iter().flat_map(|c| [c.x, c.y]).collect();
        //Every GPU error of the dispatch (a buffer the driver could not create, a lost device) is caught here and returned, so the CPU computes the batch instead
        //(the window once crashed with a buffer that could not be mapped, see issue #119); the buffers are written by the queue, not mapped when they are created
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let storage = |contents: &[u8]| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: contents.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.queue.write_buffer(&buffer, 0, contents);
            buffer
        };
        //The index of the next point that an invocation takes, see `gpu.wgsl`
        let (params, coordinates, next_point) = (storage(&params), storage(bytemuck::cast_slice(&coordinates)), storage(&0_u32.to_le_bytes()));
        let size = (points.len() * std::mem::size_of::<Outcome>()) as u64;
        let outcomes = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coordinates.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: outcomes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: next_point.as_entire_binding() },
        ] });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(if fast { &self.fast } else { &self.exact });
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64).min(WORKGROUPS), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&outcomes, 0, &read, 0, size);
        let index = self.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        //The scopes are popped in the reverse order of pushing them
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(Submitted { read, index }), |error| Err(format!("{}: {error}", self.name)))
    }

    /// Waits for a dispatch of `submit` and returns the outcome of every point, in the order of its points
    pub(crate) fn finish(&self, submitted: Submitted) -> Result<Vec<Outcome>, String> {
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(submitted.index), timeout: None }).map_err(|err| format!("{}: {err}", self.name))?;
        let view = submitted.read.slice(..).get_mapped_range().map_err(|err| format!("{}: {err:?}", self.name))?;
        Ok(bytemuck::cast_slice(&view).to_vec())
    }

    /// Checks the GPU against the CPU on points of a shallow, a medium and a deep view, with smooth and integer coloring and a small orbit radius; and that the fast
    /// loop gives the iteration count of the CPU on almost every point of the shallow view (all but 1%: measured 0.02% on a whole view)
    fn self_test(&self) -> Result<(), String> {
        for (index, (view, max_iterations, orbit_radius, smooth)) in self_test_cases().into_iter().enumerate() {
            let mut plane = ComplexPlane::new(64, 48);
            plane.set_view(&view);
            let mut m = MandelbrotSet::new(max_iterations, orbit_radius);
            m.smooth_coloring = smooth;
            let points: Vec<Complex> = (0..48).flat_map(|y| (0..64).map(move |x| (x, y))).map(|(x, y)| plane.complex_from_pixel_plane(f64::from(x) + 0.37, f64::from(y) + 0.61)).collect();
            let gpu = self.positions(&m, &points, false)?;
            if let Some((point, (gpu, cpu))) = points.iter().zip(gpu.iter().zip(points.iter().map(|c| m.position(c)))).find(|(_, (gpu, cpu))| gpu.to_bits() != cpu.to_bits()) {
                return Err(format!("{} differs from the CPU at ({}, {}): {gpu} instead of {cpu}", self.name, point.x, point.y));
            }
            if index == 0 {
                let fast = self.positions(&m, &points, true)?;
                let differ = points.iter().zip(&fast).filter(|(c, fast)| fast.floor().to_bits() != m.position(c).floor().to_bits()).count();
                if differ * 100 > points.len() {
                    return Err(format!("the fast loop of {} gives other iteration counts than the CPU for {differ} of {} points", self.name, points.len()));
                }
            }
        }
        Ok(())
    }
}

/// The views of the self-test: (view, iterations, orbit radius, smooth coloring)
fn self_test_cases() -> [(View, u32, f64, bool); 4] {
    [
        (View::new(-0.75, 0.0, 1.0), 500, 2.0, true),
        (View::new(-0.7498100000000001, -0.020300000000000054, 0.000064), 3000, 2.0, true),
        (View::new(-0.743_643_887_037_151, 0.131_825_904_205_33, 1e-10), 8000, 2.0, false),
        (View::new(-0.75, 0.1, 0.5), 300, 1.5, true),
    ]
}

/// The SPIR-V of the WGSL `source`, with an `OpDecorate NoContraction` for the result of every floating-point addition, subtraction, multiplication, division and
/// negation, in the annotations section (before the first type declaration)
fn spirv_without_contraction(source: &str) -> Result<Vec<u32>, String> {
    const FLOAT_OPERATIONS: [u32; 5] = [127, 129, 131, 133, 136]; //OpFNegate, OpFAdd, OpFSub, OpFMul, OpFDiv
    const OP_DECORATE: u32 = 71;
    const NO_CONTRACTION: u32 = 42;
    let module = naga::front::wgsl::parse_str(source).map_err(|err| format!("the shader: {err}"))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::FLOAT64).validate(&module).map_err(|err| format!("the shader: {err:?}"))?;
    let options = naga::back::spv::Options { lang_version: (1, 3), ..Default::default() };
    let pipeline_options = naga::back::spv::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: "main".into() };
    let words = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options)).map_err(|err| format!("the shader: {err}"))?;
    let (mut results, mut first_type, mut at) = (Vec::new(), None, 5);
    while at < words.len() {
        let (count, opcode) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        if (19..=39).contains(&opcode) && first_type.is_none() {
            first_type = Some(at);
        }
        if FLOAT_OPERATIONS.contains(&opcode) {
            results.push(words[at + 2]);
        }
        at += count.max(1);
    }
    let first_type = first_type.ok_or("the shader has no types")?;
    let mut patched = words[..first_type].to_vec();
    for id in results {
        patched.extend([(3 << 16) | OP_DECORATE, id, NO_CONTRACTION]);
    }
    patched.extend(&words[first_type..]);
    Ok(patched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_floating_point_operation_of_the_shaders_gets_no_contraction() {
        for source in [include_str!("gpu.wgsl"), include_str!("gpu_fast.wgsl")] {
            no_contraction_everywhere(source);
        }
    }

    fn no_contraction_everywhere(source: &str) {
        let words = spirv_without_contraction(source).unwrap();
        let mut decorated = std::collections::HashSet::new();
        let (mut operations, mut at) = (Vec::new(), 5);
        while at < words.len() {
            let (count, opcode) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
            if opcode == 71 && words[at + 2] == 42 {
                decorated.insert(words[at + 1]);
            }
            if [127, 129, 131, 133, 136].contains(&opcode) {
                operations.push(words[at + 2]);
            }
            at += count;
        }
        assert!(operations.len() > 10, "the loop has floating-point operations: {}", operations.len());
        assert!(operations.iter().all(|id| decorated.contains(id)), "every one is decorated");
    }

    #[test]
    fn the_gpu_gives_bit_for_bit_the_positions_of_the_cpu() {
        //Runs where there is a GPU with f64 shaders (not on the CI runner); `new` already runs the self-test, this checks a whole 1200 x 800 view too
        let gpu = match GpuIterator::new() {
            Ok(gpu) => gpu,
            Err(reason) => {
                println!("No GPU to test: {reason}");
                return;
            }
        };
        let mut plane = ComplexPlane::new(240, 160);
        plane.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.000064));
        let m = MandelbrotSet::new(5000, 2.0);
        let points: Vec<Complex> = (0..160).flat_map(|y| (0..240).map(move |x| (x, y))).map(|(x, y)| plane.complex_from_pixel_plane(f64::from(x) + 0.5, f64::from(y) + 0.5)).collect();
        let positions = gpu.positions(&m, &points, false).unwrap();
        let differ = points.iter().zip(&positions).filter(|(c, gpu)| gpu.to_bits() != m.position(c).to_bits()).count();
        assert_eq!(differ, 0, "{} of {} positions differ on {}", differ, points.len(), gpu.name());
        //The fast loop: the iteration counts of almost every point (measured: 1.5% differ at 1200 x 800), the colors of the rest are close
        let fast = gpu.positions(&m, &points, true).unwrap();
        let differ = points.iter().zip(&fast).filter(|(c, fast)| fast.floor().to_bits() != m.position(c).floor().to_bits()).count();
        assert!(differ * 100 < points.len() * 3, "the fast loop changes {differ} of {} iteration counts", points.len());
    }
}

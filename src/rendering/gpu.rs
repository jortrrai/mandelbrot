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
//! The fast loop gets pixels instead of points (`submit_pixels`): it computes the point of a sample itself, with the CPU's jitter and arithmetic, and returns the
//! smooth position (`FastOutcome`), so a pixel costs 12 bytes of transfers instead of 40 and the CPU only colors it. At shallow zoom the loop is so fast that
//! those costs decided how much the GPU could do (measured: a 4K flight frame of Frost on the Shoreline with the GPU doing 54% instead of 15%, 34 FPS instead of 21).
//!
//! `GpuIterator::new` checks a set of points on the GPU against the CPU before it is used, and refuses when anything differs, when there is no suitable GPU,
//! or when it lacks f64 shaders or SPIR-V passthrough: the renderer then stays on the CPU.

use std::borrow::Cow;


use crate::{complex::Complex, complex_plane::{ComplexPlane, View}, mandelbrot_set::MandelbrotSet};

/// About the most points of one dispatch. The renderer sizes dispatches by time (`GPU_BATCH_TIME`), which keeps the GPU responsive for the window; this cap only
/// limits the cheap points of shallow views, where every dispatch's fixed cost (writing, submitting, waiting: about half a millisecond) outweighed the work of
/// 262,144 points (#212)
pub(crate) const BATCH: usize = 1 << 20;

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

/// The outcome of the fast loop for one pixel, as `gpu_fast.wgsl` writes it: the position itself, see `position`
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FastOutcome {
    /// The whole part of the position; with the top bit set, a smooth position that is still to be clamped
    whole: u32,
    fraction: f32,
}

impl FastOutcome {
    /// The position: the iteration count, or the smooth position clamped to `[0, max_iterations - 1]` like `MandelbrotSet::smooth_position` does
    pub(crate) fn position(self, m: &MandelbrotSet) -> f64 {
        if self.whole & SMOOTH == 0 {
            return f64::from(self.whole);
        }
        (f64::from(self.whole & !SMOOTH) + f64::from(self.fraction)).clamp(0.0, f64::from(m.max_iterations.saturating_sub(1)))
    }
}

/// The top bit of `FastOutcome::whole`: a smooth position
const SMOOTH: u32 = 1 << 31;

/// A pixel for `GpuIterator::submit_pixels`: `x` in the low 16 bits, `y` in the high 16 bits (both below `MAX_PIXEL`)
pub const fn packed_pixel(x: usize, y: usize) -> u32 {
    (x as u32) | ((y as u32) << 16)
}

/// The pixels of a batch of `submit_pixels` are below this in both directions, see `packed_pixel`
pub(crate) const MAX_PIXEL: usize = 1 << 16;

/// A dispatch that runs on the GPU, see `GpuIterator::submit`
pub(crate) struct Submitted {
    buffers: Buffers,
    /// The bytes of the outcomes
    size: u64,
    index: wgpu::SubmissionIndex,
}

/// The buffers of one dispatch and their bind group, kept for the next dispatch (see `GpuIterator::buffers`): creating them for every dispatch (and wgpu
/// clearing the new outcome buffers) took a feeder of the GPU half a millisecond per dispatch, 100,000 times in a 4K video (#212)
struct Buffers {
    params: wgpu::Buffer,
    input: wgpu::Buffer,
    next_point: wgpu::Buffer,
    outcomes: wgpu::Buffer,
    read: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// The bytes the input and the outcomes can hold
    input_capacity: u64,
    outcome_capacity: u64,
}

/// The bytes of the parameters of a dispatch (the most, those of `submit_pixels`)
const PARAMS_CAPACITY: u64 = 96;

/// A GPU that iterates points exactly like the CPU, see the module
pub struct GpuIterator {
    /// The device and its queue are never dropped: dropping the last of them destroys the Vulkan instance, and NVIDIA's driver crashed doing that on a
    /// thread while other threads or other instances still worked with it (issue #200: a video job that let go of the GPU after the window had, and tests
    /// that each open a GPU, in parallel). A GPU is opened once per program (and once per test), so the memory is negligible
    device: std::mem::ManuallyDrop<wgpu::Device>,
    queue: std::mem::ManuallyDrop<wgpu::Queue>,
    /// The exact loop (`gpu.wgsl`) and the fast double-float loop (`gpu_fast.wgsl`)
    exact: wgpu::ComputePipeline,
    fast: wgpu::ComputePipeline,
    /// The plain f32 loop (`gpu_f32.wgsl`), for moving frames at shallow zoom, see `Render::f32_gpu`
    single: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    name: String,
    /// The buffers of finished dispatches, for the next ones (several renders feed the GPU at once)
    spare: std::sync::Mutex<Vec<Buffers>>,
}

impl GpuIterator {
    /// Opens the most powerful GPU with f64 shaders and SPIR-V passthrough, and checks it against the CPU (see the module).
    ///
    /// # Errors
    /// The reason the GPU cannot be used: no adapter, a missing feature, a failed request, or an outcome that differs from the CPU's
    pub fn new() -> Result<GpuIterator, String> {
        //Vulkan, Metal and DX12 only: the loops need f64 shaders and SPIR-V, which OpenGL does not have, and looking at OpenGL's adapters through EGL could
        //panic in the EGL library while the window's OpenGL context is current on another thread (seen on X11 with NVIDIA), leaving the window without a GPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..wgpu::InstanceDescriptor::new_without_display_handle() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|err| format!("no GPU: {err}"))?;
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let needed = wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64 | wgpu::Features::PASSTHROUGH_SHADERS;
        if !adapter.features().contains(needed) {
            return Err(format!("{name} lacks f64 or 64-bit integer shaders, or SPIR-V passthrough"));
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
        let (exact, fast, single) = (pipeline(include_str!("gpu.wgsl"))?, pipeline(include_str!("gpu_fast.wgsl"))?, pipeline(include_str!("gpu_f32.wgsl"))?);
        let (device, queue) = (std::mem::ManuallyDrop::new(device), std::mem::ManuallyDrop::new(queue));
        let gpu = GpuIterator { device, queue, exact, fast, single, layout, name, spare: std::sync::Mutex::new(Vec::new()) };
        gpu.self_test()?;
        Ok(gpu)
    }

    /// The name and the backend of the GPU
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The position of every point: exactly `m.position(point)` (see the module)
    ///
    /// # Errors
    /// The GPU failed (for example the device was lost)
    pub fn positions(&self, m: &MandelbrotSet, points: &[Complex]) -> Result<Vec<f64>, String> {
        let mut positions = vec![0.0; points.len()];
        for (batch, out) in points.chunks(BATCH).zip(positions.chunks_mut(BATCH)) {
            let outcomes: Vec<Outcome> = self.finish(self.submit(m, batch)?)?;
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

    /// The position of sample `sample` of every pixel of `pixels` (see `packed_pixel`) of `anchor`, where pixel (0, 0) is pixel `origin` of `anchor`, with the fast
    /// loop: the positions the renderer gets with `Render::fast_gpu`
    ///
    /// # Errors
    /// The GPU failed (for example the device was lost)
    pub fn fast_positions(&self, m: &MandelbrotSet, anchor: &ComplexPlane, sample: u8, origin: (isize, isize), pixels: &[u32]) -> Result<Vec<f64>, String> {
        let mut positions = Vec::with_capacity(pixels.len());
        for batch in pixels.chunks(BATCH) {
            let outcomes: Vec<FastOutcome> = self.finish(self.submit_pixels(m, anchor, sample, origin, batch, false)?)?;
            positions.extend(outcomes.into_iter().map(|outcome| outcome.position(m)));
        }
        Ok(positions)
    }

    /// Starts one dispatch of the exact loop over `points` (about `BATCH` at most) and returns at once: the GPU runs it while the caller prepares the next one, see
    /// `finish`, which gives an `Outcome` for every point
    pub(crate) fn submit(&self, m: &MandelbrotSet, points: &[Complex]) -> Result<Submitted, String> {
        let mut params = Vec::with_capacity(24);
        params.extend((m.orbit_radius * m.orbit_radius).to_le_bytes());
        for value in [m.max_iterations, u32::try_from(points.len()).map_err(|_| String::from("too many points"))?, u32::from(m.orbit_radius >= 2.0), 0] {
            params.extend(value.to_le_bytes());
        }
        let coordinates: Vec<f64> = points.iter().flat_map(|c| [c.x, c.y]).collect();
        self.dispatch(&self.exact, &params, bytemuck::cast_slice(&coordinates), points.len(), std::mem::size_of::<Outcome>())
    }

    /// Starts one dispatch of the fast loop over sample `sample` of `pixels` (about `BATCH` at most, see `fast_positions`) and returns at once, see `finish`, which
    /// gives a `FastOutcome` for every pixel. The GPU computes the points itself (see `gpu_fast.wgsl`)
    pub(crate) fn submit_pixels(&self, m: &MandelbrotSet, anchor: &ComplexPlane, sample: u8, origin: (isize, isize), pixels: &[u32], single: bool) -> Result<Submitted, String> {
        let to_i32 = |value: isize| i32::try_from(value).map_err(|_| String::from("the origin is too far away"));
        let mut params = Vec::with_capacity(72);
        for value in [anchor.min_x, anchor.min_y, anchor.increment_x, anchor.increment_y, m.orbit_radius * m.orbit_radius] {
            params.extend(value.to_le_bytes());
        }
        for value in [m.max_iterations, u32::try_from(pixels.len()).map_err(|_| String::from("too many points"))?, u32::from(m.orbit_radius >= 2.0), u32::from(m.smooth_coloring)] {
            params.extend(value.to_le_bytes());
        }
        for value in [to_i32(origin.0)?, to_i32(origin.1)?] {
            params.extend(value.to_le_bytes());
        }
        for value in [u32::from(sample), 0] {
            params.extend(value.to_le_bytes());
        }
        self.dispatch(if single { &self.single } else { &self.fast }, &params, bytemuck::cast_slice(pixels), pixels.len(), std::mem::size_of::<FastOutcome>())
    }

    /// Buffers for a dispatch whose input has `input` bytes and whose outcomes have `outcomes` bytes: spare ones that are large enough, or new ones (the sizes
    /// rounded up to a power of two, so they also serve somewhat larger dispatches later)
    fn buffers(&self, input: u64, outcomes: u64) -> Buffers {
        let mut spare = self.spare.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(index) = spare.iter().position(|buffers| buffers.input_capacity >= input && buffers.outcome_capacity >= outcomes) {
            return spare.swap_remove(index);
        }
        drop(spare);
        let (input_capacity, outcome_capacity) = (input.max(4).next_power_of_two(), outcomes.max(4).next_power_of_two());
        let buffer = |size: u64, usage: wgpu::BufferUsages| self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
        let (params, input, next_point) = (buffer(PARAMS_CAPACITY, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST), buffer(input_capacity, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST), buffer(4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST));
        let (outcomes, read) = (buffer(outcome_capacity, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC), buffer(outcome_capacity, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: outcomes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: next_point.as_entire_binding() },
        ] });
        Buffers { params, input, next_point, outcomes, read, bind, input_capacity, outcome_capacity }
    }

    /// One dispatch of `pipeline` over `count` points with these parameters and input, whose outcomes have `outcome_size` bytes each. The shaders read only
    /// `count` points (see `params.count`), so buffers larger than the dispatch serve it as they are
    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, params: &[u8], input: &[u8], count: usize, outcome_size: usize) -> Result<Submitted, String> {
        let count = u32::try_from(count).map_err(|_| String::from("too many points"))?;
        //Every GPU error of the dispatch (a buffer the driver could not create, a lost device) is caught here and returned, so the CPU computes the batch instead
        //(the window once crashed with a buffer that could not be mapped, see issue #119); the buffers are written by the queue, not mapped when they are created
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let size = (count as usize * outcome_size) as u64;
        let buffers = self.buffers(input.len() as u64, size);
        self.queue.write_buffer(&buffers.params, 0, params);
        if !input.is_empty() {
            self.queue.write_buffer(&buffers.input, 0, input);
        }
        //The index of the next point that an invocation takes, see `gpu.wgsl`
        self.queue.write_buffer(&buffers.next_point, 0, &0_u32.to_le_bytes());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &buffers.bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64).min(WORKGROUPS), 1, 1);
        }
        if size > 0 {
            encoder.copy_buffer_to_buffer(&buffers.outcomes, 0, &buffers.read, 0, size);
        }
        let index = self.queue.submit([encoder.finish()]);
        if size > 0 {
            buffers.read.slice(..size).map_async(wgpu::MapMode::Read, |_| {});
        }
        //The scopes are popped in the reverse order of pushing them
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(Submitted { buffers, size, index }), |error| Err(format!("{}: {error}", self.name)))
    }

    /// Waits for a dispatch of `submit` (`Outcome`) or `submit_pixels` (`FastOutcome`) and returns the outcome of every point, in the order of its points; its
    /// buffers are kept for the next dispatch
    pub(crate) fn finish<T: bytemuck::Pod>(&self, submitted: Submitted) -> Result<Vec<T>, String> {
        let Submitted { buffers, size, index } = submitted;
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).map_err(|err| format!("{}: {err}", self.name))?;
        if size == 0 {
            return Ok(Vec::new());
        }
        let outcomes = {
            let view = buffers.read.slice(..size).get_mapped_range().map_err(|err| format!("{}: {err:?}", self.name))?;
            bytemuck::cast_slice(&view).to_vec()
        };
        buffers.read.unmap();
        self.spare.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(buffers);
        Ok(outcomes)
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
            let gpu = self.positions(&m, &points)?;
            if let Some((point, (gpu, cpu))) = points.iter().zip(gpu.iter().zip(points.iter().map(|c| m.position(c)))).find(|(_, (gpu, cpu))| gpu.to_bits() != cpu.to_bits()) {
                return Err(format!("{} differs from the CPU at ({}, {}): {gpu} instead of {cpu}", self.name, point.x, point.y));
            }
            if index == 0 {
                //The fast loop computes the points of the pixels itself: sample 3 of the pixels, placed 5 pixels right and 2 up in the plane
                let (pixels, cpu) = fast_test_case(&plane, &m, 64, 48);
                let fast = self.fast_positions(&m, &plane, 3, (5, -2), &pixels)?;
                let differ = cpu.iter().zip(&fast).filter(|(cpu, fast)| fast.floor().to_bits() != cpu.floor().to_bits()).count();
                if differ * 100 > pixels.len() {
                    return Err(format!("the fast loop of {} gives other iteration counts than the CPU for {differ} of {} points", self.name, pixels.len()));
                }
            }
        }
        Ok(())
    }
}

/// The `width` x `height` pixels of `plane` packed for `fast_positions`, and the positions of the CPU for sample 3 of them, placed at origin (5, -2) (see `progressive::sample_point`)
fn fast_test_case(plane: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize) -> (Vec<u32>, Vec<f64>) {
    let pixels: Vec<(usize, usize)> = (0..height).flat_map(|y| (0..width).map(move |x| (x, y))).collect();
    let cpu = pixels.iter().map(|(x, y)| m.position(&super::progressive::sample_point(plane, *x, *y, 3, (5, -2)))).collect();
    (pixels.iter().map(|(x, y)| packed_pixel(*x, *y)).collect(), cpu)
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
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::FLOAT64 | naga::valid::Capabilities::SHADER_INT64).validate(&module).map_err(|err| format!("the shader: {err:?}"))?;
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
        for source in [include_str!("gpu.wgsl"), include_str!("gpu_fast.wgsl"), include_str!("gpu_f32.wgsl")] {
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
        let positions = gpu.positions(&m, &points).unwrap();
        let differ = points.iter().zip(&positions).filter(|(c, gpu)| gpu.to_bits() != m.position(c).to_bits()).count();
        assert_eq!(differ, 0, "{} of {} positions differ on {}", differ, points.len(), gpu.name());
        //The fast loop: the iteration counts of almost every point (measured: 1.5% differ at 1200 x 800), the colors of the rest are close
        let (pixels, cpu) = fast_test_case(&plane, &m, 240, 160);
        let fast = gpu.fast_positions(&m, &plane, 3, (5, -2), &pixels).unwrap();
        let differ = cpu.iter().zip(&fast).filter(|(cpu, fast)| fast.floor().to_bits() != cpu.floor().to_bits()).count();
        assert!(differ * 100 < pixels.len() * 3, "the fast loop changes {differ} of {} iteration counts", pixels.len());
        //On a calm view the orbits of the fast loop are the CPU's, so the smooth positions are too, up to the f32 logarithms of the GPU (far below the 1/256 steps of
        //the coloring). In the chaotic parts of a view like this one, an orbit that ends with the same count can end elsewhere, which moves the fraction
        let mut calm = ComplexPlane::new(240, 160);
        calm.set_view(&View::new(-0.75, 0.0, 1.0));
        let m = MandelbrotSet::new(500, 2.0);
        let (pixels, cpu) = fast_test_case(&calm, &m, 240, 160);
        let fast = gpu.fast_positions(&m, &calm, 3, (5, -2), &pixels).unwrap();
        let close = cpu.iter().zip(&fast).filter(|(cpu, fast)| (*cpu - *fast).abs() < 1e-4).count();
        assert!(close * 1000 >= cpu.len() * 999, "only {close} of {} fast positions are the CPU's", cpu.len());
    }
}

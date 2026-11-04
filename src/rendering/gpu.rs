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


use crate::{coloring::TrueColor, complex::Complex, complex_plane::{ComplexPlane, View}, mandelbrot_set::MandelbrotSet};

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
    /// The position of the point of an outcome of the exact loop for pixels (`submit_exact_pixels`), which continues an escaped orbit for smooth coloring
    /// itself and keeps the steps it added in the bits above the lowest of `escaped`: exactly `m.position(c)`, without `c`
    pub(crate) fn position_continued(&self, m: &MandelbrotSet) -> f64 {
        if self.escaped & 1 != 0 && m.smooth_coloring {
            m.smooth_position_at(m.block_end(self.iterations + 1) + (self.escaped >> 1), (self.zx, self.zy))
        } else {
            f64::from(self.iterations)
        }
    }

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

/// The workgroups of 64 invocations of a composed frame: each invocation composes pixels a whole dispatch apart (see `gpu_compose.wgsl`)
const COMPOSE_WORKGROUPS: u32 = 4096;

/// Where a tap of a composed frame falls along one side of a keyframe: the two pixels it is between and its share of the second (see `gui::dive_keyframes`)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComposeTap {
    /// The pixel before the tap
    pub first: u32,
    /// The pixel after the tap (the same at the edge)
    pub second: u32,
    /// The share of the second pixel
    pub share: f32,
}

/// The keyframe images of a dive on the GPU, see `GpuIterator::upload_images`
pub struct ComposeImages {
    buffer: wgpu::Buffer,
    /// The size of a keyframe
    width: u32,
    pixels: u32,
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

/// Every use of the GPU holds this gate open (shared), and `close_gpus` shuts it for good when the program ends: then it waits for the dispatches under way,
/// and render threads that want the GPU after that wait for ever, while the process ends. Without it a render thread could call into Vulkan while the
/// program's exit tears down the Vulkan loader, which crashed (issue #200: a window closed during a GPU render, a test ending while its app still rendered).
/// A thread takes the gate once, never twice (a second shared lock could wait for a waiting `close_gpus`)
static GATE: std::sync::RwLock<()> = std::sync::RwLock::new(());

fn gate() -> std::sync::RwLockReadGuard<'static, ()> {
    GATE.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Shuts the gate of the GPUs (see `GATE`) when the program ends, once for good.
///
/// Called when the window closes and the command line ends, and at exit (on Linux, also for the tests)
pub fn close_gpus() {
    static CLOSED: std::sync::Once = std::sync::Once::new();
    CLOSED.call_once(|| std::mem::forget(GATE.write().unwrap_or_else(std::sync::PoisonError::into_inner)));
}

/// Closes the gate (see `close_gpus`) at the program's exit too, before the Vulkan loader is torn down: the handler is registered after the loader is
/// loaded, so it runs before the loader's own handlers
#[cfg(target_os = "linux")]
fn close_gpus_at_exit() {
    extern "C" fn at_exit() {
        close_gpus();
    }
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    //SAFETY: `at_exit` is a plain function that never unwinds
    REGISTERED.call_once(|| unsafe {
        libc::atexit(at_exit);
    });
}

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
    /// The exact loop fed pixels instead of points (`exact_pixels_source`), used only when the self-test found its outcomes the exact loop's bit for bit
    exact_pixels: Option<wgpu::ComputePipeline>,
    exact_pixels_works: std::sync::atomic::AtomicBool,
    /// The fast loop that colors its samples and adds them to sums on the GPU (`colored_source`), with its own bindings; used only when the self-test found
    /// its sums the CPU's bit for bit
    colored: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    colored_works: std::sync::atomic::AtomicBool,
    /// The adaptive mask on the GPU (`gpu_mask.wgsl`), used only when the self-test found it the CPU's bit for bit
    mask: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    mask_works: std::sync::atomic::AtomicBool,
    /// The perturbation loop of deep views (`gpu_deep.wgsl`, issue #263), used only when the self-test found its outcomes the CPU's bit for bit
    deep: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// The same loop in 32-bit floats (`deep32_source`, issue #288: an approximation, off by default), with the bindings of `deep`
    deep32: Option<wgpu::ComputePipeline>,
    deep32_works: std::sync::atomic::AtomicBool,
    deep_works: std::sync::atomic::AtomicBool,
    layout: wgpu::BindGroupLayout,
    /// The frames of dives composed from keyframes (`gpu_compose.wgsl`, see `compose`), with their own buffers
    compose: wgpu::ComputePipeline,
    compose_layout: wgpu::BindGroupLayout,
    name: String,
    /// The buffers of finished dispatches, for the next ones (several renders feed the GPU at once)
    spare: std::sync::Mutex<Vec<Buffers>>,
    /// The sums and the read-back buffer of the last frame composed, for the next frames of its size (see `compose`)
    compose_buffers: std::sync::Mutex<Option<(u64, wgpu::Buffer, wgpu::Buffer)>>,
}

impl GpuIterator {
    /// Opens the most powerful GPU with f64 shaders and SPIR-V passthrough, and checks it against the CPU (see the module).
    ///
    /// # Errors
    /// The reason the GPU cannot be used: no adapter, a missing feature, a failed request, or an outcome that differs from the CPU's
    pub fn new() -> Result<GpuIterator, String> {
        let gpu = {
            let _gate = gate();
            Self::open()?
        };
        #[cfg(target_os = "linux")]
        close_gpus_at_exit();
        gpu.self_test()?;
        Ok(gpu)
    }

    /// Opens the GPU and its pipelines (see `new`)
    fn open() -> Result<GpuIterator, String> {
        //Vulkan, Metal and DX12 only: the loops need f64 shaders and SPIR-V, which OpenGL does not have, and looking at OpenGL's adapters through EGL could
        //panic in the EGL library while the window's OpenGL context is current on another thread (seen on X11 with NVIDIA), leaving the window without a GPU
        //No debug labels or validation (wgpu turns them on in debug builds): naming a Vulkan object crashed inside the Vulkan loader while another thread
        //opened or closed another instance (issue #200: the tests, each with a GPU, in a debug build), and nothing reads the names
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, flags: wgpu::InstanceFlags::empty(), ..wgpu::InstanceDescriptor::new_without_display_handle() });
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
        let pipeline = |source: &str, pipeline_layout: &wgpu::PipelineLayout| -> Result<wgpu::ComputePipeline, String> {
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
            Ok(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("mandelbrot"), layout: Some(pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None }))
        };
        let (exact, fast, single) = (pipeline(include_str!("gpu.wgsl"), &pipeline_layout)?, pipeline(include_str!("gpu_fast.wgsl"), &pipeline_layout)?, pipeline(include_str!("gpu_f32.wgsl"), &pipeline_layout)?);
        let compose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("compose"), entries: &[entry(0, true), entry(1, true), entry(2, true), entry(3, true), entry(4, false)] });
        let compose_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("compose"), bind_group_layouts: &[Some(&compose_layout)], immediate_size: 0 });
        let compose = pipeline(include_str!("gpu_compose.wgsl"), &compose_pipeline_layout)?;
        //(Without it the CPU computes the points of the exact loop, as before)
        let exact_pixels = pipeline(&exact_pixels_source(), &pipeline_layout).ok();
        //(Without it the CPU renders every deep view, as before)
        let deep = {
            let entries = [entry(0, true), entry(1, true), entry(2, true), entry(3, true), entry(4, false), entry(5, true)];
            let deep_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("deep"), entries: &entries });
            let deep_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("deep"), bind_group_layouts: &[Some(&deep_layout)], immediate_size: 0 });
            let single = pipeline(&deep32_source(), &deep_pipeline_layout).ok();
            pipeline(include_str!("gpu_deep.wgsl"), &deep_pipeline_layout).ok().map(|pipeline| ((pipeline, deep_layout), single))
        };
        let (deep, deep32) = deep.map_or((None, None), |(deep, single)| (Some(deep), single));
        //(Without it the CPU computes the adaptive mask, as before)
        let mask = {
            let entries = [entry(0, true), entry(1, true), entry(2, true), entry(3, false), entry(4, false)];
            let mask_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mask"), entries: &entries });
            let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mask"), bind_group_layouts: &[Some(&mask_layout)], immediate_size: 0 });
            spirv_without_contraction(include_str!("gpu_mask.wgsl")).ok().map(|words| {
                //SAFETY: the SPIR-V is naga's validated output of the shader, with only NoContraction decorations added
                let shader = unsafe {
                    device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some("mask"),
                        entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint { name: "main".into(), workgroup_size: (64, 1, 1) }]),
                        spirv: Some(Cow::Owned(words)),
                        ..Default::default()
                    })
                };
                (device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("mask"), layout: Some(&mask_pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None }), mask_layout)
            })
        };
        //(Without it the CPU colors the samples of the fast loop, as before)
        let colored = {
            let entries = [entry(0, true), entry(1, true), entry(2, false), entry(3, false), entry(4, true), entry(5, false)];
            let colored_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("colored"), entries: &entries });
            let colored_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("colored"), bind_group_layouts: &[Some(&colored_layout)], immediate_size: 0 });
            let words = spirv_without_contraction(&colored_source());
            words.ok().map(|words| {
                //SAFETY: the SPIR-V is naga's validated output of the shader, with only NoContraction decorations added
                let shader = unsafe {
                    device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some("colored"),
                        entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint { name: "main".into(), workgroup_size: (64, 1, 1) }]),
                        spirv: Some(Cow::Owned(words)),
                        ..Default::default()
                    })
                };
                (device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("colored"), layout: Some(&colored_pipeline_layout), module: &shader, entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None }), colored_layout)
            })
        };
        let (device, queue) = (std::mem::ManuallyDrop::new(device), std::mem::ManuallyDrop::new(queue));
        Ok(GpuIterator { device, queue, exact, fast, single, exact_pixels, exact_pixels_works: std::sync::atomic::AtomicBool::new(false), colored, colored_works: std::sync::atomic::AtomicBool::new(false), mask, mask_works: std::sync::atomic::AtomicBool::new(false), deep, deep32, deep32_works: std::sync::atomic::AtomicBool::new(false), deep_works: std::sync::atomic::AtomicBool::new(false), layout, compose, compose_layout, name, spare: std::sync::Mutex::new(Vec::new()), compose_buffers: std::sync::Mutex::new(None) })
    }

    /// Puts the keyframe images of a dive (each `width` x `height` pixels) on the GPU, for `compose`
    ///
    /// # Errors
    /// The reason they could not be put there, such as more bytes than one buffer of the GPU can hold
    pub fn upload_images(&self, images: &[Vec<TrueColor>], width: usize, height: usize) -> Result<ComposeImages, String> {
        let pixels = width * height;
        let bytes = (images.len() * pixels * 4) as u64;
        let limits = self.device.limits();
        if bytes > limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) || images.iter().any(|image| image.len() != pixels) {
            return Err(format!("{} cannot hold {} keyframes of {width} x {height} in one buffer", self.name, images.len()));
        }
        let packed: Vec<u32> = images.iter().flat_map(|image| image.iter().map(|color| u32::from(color.red) | (u32::from(color.green) << 8) | (u32::from(color.blue) << 16))).collect();
        let _gate = gate();
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("keyframes"), size: bytes.max(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&packed));
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(()), |error| Err(format!("{}: {error}", self.name)))?;
        Ok(ComposeImages { buffer, width: u32::try_from(width).map_err(|_| String::from("too wide"))?, pixels: u32::try_from(pixels).map_err(|_| String::from("too large"))? })
    }

    /// The `width` x `height` frame that averages the colors `moments` give its pixels (see `gpu_compose.wgsl`). A moment is the
    /// index of its keyframe in `images` and its taps: `2 * width` column taps (tap x 0.25, then 0.75), then `2 * height` row taps. The number of moments is
    /// a power of two (then the average is exact)
    ///
    /// # Errors
    /// A failure of the GPU, or a number of moments that is not a power of two: the CPU composes the frame then
    pub fn compose(&self, images: &ComposeImages, (width, height): (usize, usize), moments: &[(usize, Vec<ComposeTap>)]) -> Result<Vec<TrueColor>, String> {
        if !moments.len().is_power_of_two() {
            return Err(String::from("the moments are not a power of two"));
        }
        let too_large = |_| String::from("too large");
        let (frame_width, frame_height, pixels) = (u32::try_from(width).map_err(too_large)?, u32::try_from(height).map_err(too_large)?, u32::try_from(width * height).map_err(too_large)?);
        let mut moment_words: Vec<u32> = Vec::with_capacity(moments.len() * 4);
        let mut taps: Vec<ComposeTap> = Vec::new();
        for (image, moment_taps) in moments {
            if moment_taps.len() != 2 * (width + height) {
                return Err(String::from("the taps of a moment do not fit the frame"));
            }
            moment_words.extend([u32::try_from(*image).map_err(too_large)? * images.pixels, u32::try_from(taps.len()).map_err(too_large)?, 0, 0]);
            taps.extend_from_slice(moment_taps);
        }
        let count = u32::try_from(moments.len()).map_err(too_large)?;
        let params = [frame_width, frame_height, images.width, count, (1.0 / count as f32).to_bits(), 0, 0, 0];
        let size = u64::from(pixels) * 4;
        let _gate = gate();
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let with_data = |label: &str, bytes: &[u8]| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.len() as u64).max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.queue.write_buffer(&buffer, 0, bytes);
            buffer
        };
        let (params, moment_buffer, tap_buffer) = (with_data("compose params", bytemuck::cast_slice(&params)), with_data("compose moments", bytemuck::cast_slice(&moment_words)), with_data("compose taps", bytemuck::cast_slice(&taps)));
        //(The two frame-sized buffers of the last frame again, if they fit: creating them for every frame, and wgpu clearing them, cost more than composing,
        //as for the dispatches in #212. The shader writes every pixel's sum, so what they held does not matter)
        let mut kept = self.compose_buffers.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if kept.as_ref().is_some_and(|(kept_size, _, _)| *kept_size != size) {
            *kept = None;
        }
        let (_, sums, read) = &*kept.get_or_insert_with(|| (size, self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("compose sums"), size: size.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("compose read"), size: size.max(16), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })));
        let composed = (|| -> Result<Vec<TrueColor>, String> {
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.compose_layout, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: moment_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tap_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: images.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sums.as_entire_binding() },
            ] });
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.compose);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(pixels.div_ceil(64).clamp(1, COMPOSE_WORKGROUPS), 1, 1);
            }
            encoder.copy_buffer_to_buffer(sums, 0, read, 0, size);
            let index = self.queue.submit([encoder.finish()]);
            read.slice(..size).map_async(wgpu::MapMode::Read, |_| {});
            let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
            errors.into_iter().flatten().next().map_or(Ok(()), |error| Err(format!("{}: {error}", self.name)))?;
            self.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).map_err(|err| format!("{}: {err}", self.name))?;
            let out = {
                let view = read.slice(..size).get_mapped_range().map_err(|err| format!("{}: {err:?}", self.name))?;
                //(8 bits each of red, green and blue in a word, straight to colors: one pass over the frame)
                bytemuck::cast_slice::<u8, u32>(&view).iter().map(|color| TrueColor::new((color & 0xff) as u8, ((color >> 8) & 0xff) as u8, ((color >> 16) & 0xff) as u8)).collect()
            };
            read.unmap();
            Ok(out)
        })();
        //(A failure may leave the read-back buffer mapped: new ones for the next frame)
        if composed.is_err() {
            *kept = None;
        }
        drop(kept);
        composed
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
        let coordinates: Vec<f64> = points.iter().flat_map(|c| [c.x, c.y]).collect();
        self.submit_coordinates(m, &coordinates)
    }

    /// `submit` for points given as their coordinates, real and imaginary part after each other (so a caller that makes many points writes them only once)
    pub(crate) fn submit_coordinates(&self, m: &MandelbrotSet, coordinates: &[f64]) -> Result<Submitted, String> {
        let count = coordinates.len() / 2;
        let mut params = Vec::with_capacity(24);
        params.extend((m.orbit_radius * m.orbit_radius).to_le_bytes());
        for value in [m.max_iterations, u32::try_from(count).map_err(|_| String::from("too many points"))?, u32::from(m.orbit_radius >= 2.0), 0] {
            params.extend(value.to_le_bytes());
        }
        self.dispatch(&self.exact, &params, bytemuck::cast_slice(coordinates), count, std::mem::size_of::<Outcome>())
    }

    /// Starts one dispatch of the fast loop over sample `sample` of `pixels` (about `BATCH` at most, see `fast_positions`) and returns at once, see `finish`, which
    /// gives a `FastOutcome` for every pixel. The GPU computes the points itself (see `gpu_fast.wgsl`)
    pub(crate) fn submit_pixels(&self, m: &MandelbrotSet, anchor: &ComplexPlane, sample: u8, origin: (isize, isize), pixels: &[u32], single: bool) -> Result<Submitted, String> {
        let params = pixel_params(m, anchor, sample, origin, pixels.len())?;
        self.dispatch(if single { &self.single } else { &self.fast }, &params, bytemuck::cast_slice(pixels), pixels.len(), std::mem::size_of::<FastOutcome>())
    }

    /// Whether the exact loop computes the points of pixels itself (`submit_exact_pixels`): when the self-test found its outcomes those of the exact loop
    /// with the CPU's points, bit for bit
    pub(crate) fn exact_pixels_work(&self) -> bool {
        self.exact_pixels_works.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `submit` for sample `sample` of `pixels` of the unturned `anchor` (as `submit_pixels`): the GPU computes their points with the arithmetic of
    /// `progressive::sample_point`, so 4 bytes go to it per point instead of 16 and the CPU computes no points for it. Its outcomes are `Outcome`s (see `finish`)
    /// whose position is `Outcome::position_continued`
    pub(crate) fn submit_exact_pixels(&self, m: &MandelbrotSet, anchor: &ComplexPlane, sample: u8, origin: (isize, isize), pixels: &[u32]) -> Result<Submitted, String> {
        let params = pixel_params(m, anchor, sample, origin, pixels.len())?;
        let pipeline = self.exact_pixels.as_ref().ok_or_else(|| String::from("the exact loop for pixels did not compile"))?;
        self.dispatch(pipeline, &params, bytemuck::cast_slice(pixels), pixels.len(), std::mem::size_of::<Outcome>())
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
        let _gate = gate();
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
        let _gate = gate();
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

    /// Turns the coloring fast loop off or on again (if it passed its self-test), to compare with the CPU's coloring in tests
    #[cfg(test)]
    pub(crate) fn use_colored(&self, on: bool) {
        self.colored_works.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the GPU iterates deep views (`deep_outcomes`): when the self-test found its outcomes the CPU's, bit for bit
    pub(crate) fn deep_works(&self) -> bool {
        self.deep_works.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The outcome of the perturbation loop (`gpu_deep.wgsl`, `rendering::deep::position_from`) for every offset `offsets` from the center of the reference
    /// orbit of `input`, scaled by `2^scaled` (0: plain offsets; see `rendering::deep::position_scaled`), in at most `BATCH` offsets a dispatch
    ///
    /// # Errors
    /// The GPU failed, or there is no such shader
    pub(crate) fn deep_outcomes(&self, m: &MandelbrotSet, input: &DeepInput, offsets: &[(f64, f64)], scaled: i32) -> Result<Vec<DeepOutcome>, String> {
        self.deep_dispatches(m, input, offsets, scaled, false)
    }

    /// `deep_outcomes` in 32-bit floats (`deep32_source`, issue #288): an approximation of the f64 loop, much faster on GPUs whose f64 rate is low, for
    /// offsets scaled by `2^scaled` with `scaled` beyond 100 (f32's range) and plain below
    ///
    /// # Errors
    /// The GPU failed, or there is no such shader
    pub(crate) fn deep_outcomes32(&self, m: &MandelbrotSet, input: &DeepInput, offsets: &[(f64, f64)], scaled: i32) -> Result<Vec<DeepOutcome>, String> {
        self.deep_dispatches(m, input, offsets, scaled, true)
    }

    /// Whether the GPU has the 32-bit deep loop (see `deep_outcomes32`)
    pub(crate) fn has_deep32(&self) -> bool {
        self.deep_works() && self.deep32_works.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn deep_dispatches(&self, m: &MandelbrotSet, input: &DeepInput, offsets: &[(f64, f64)], scaled: i32, single: bool) -> Result<Vec<DeepOutcome>, String> {
        let DeepInput { orbit, residuals, table, starts, lengths } = *input;
        let (deep, layout) = self.deep.as_ref().ok_or_else(|| String::from("the deep shader did not compile"))?;
        let pipeline = if single { self.deep32.as_ref().ok_or_else(|| String::from("the 32-bit deep shader did not compile"))? } else { deep };
        //(In 32-bit floats every number of the input is rounded to f32, the residuals' mantissas too)
        let floats = |values: &[f64]| -> Vec<u8> { if single { values.iter().flat_map(|value| (*value as f32).to_le_bytes()).collect() } else { values.iter().flat_map(|value| value.to_le_bytes()).collect() } };
        if starts.len() > 32 || starts.len() != lengths.len() {
            return Err(String::from("too many levels"));
        }
        let too_large = |_| String::from("too large");
        let storage = wgpu::BufferUsages::STORAGE;
        let _gate = gate();
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let buffer_with = |label: &str, bytes: &[u8]| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.len() as u64).max(16), usage: storage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.queue.write_buffer(&buffer, 0, bytes);
            buffer
        };
        let flat_orbit: Vec<f64> = orbit.iter().flat_map(|(x, y)| [*x, *y]).collect();
        let residual_bytes: Vec<u8> = if single {
            residuals.iter().flat_map(|residual| [(residual.mx as f32).to_le_bytes(), (residual.my as f32).to_le_bytes(), residual.px.to_le_bytes(), residual.py.to_le_bytes()].concat()).collect()
        } else {
            bytemuck::cast_slice(residuals).to_vec()
        };

        let (orbit_buffer, table_buffer) = (buffer_with("deep orbit", &floats(&flat_orbit)), buffer_with("deep table", &floats(table)));
        let residual_buffer = buffer_with("deep residuals", &residual_bytes);
        let mut outcomes = Vec::with_capacity(offsets.len());
        for batch in offsets.chunks(BATCH) {
            let mut params: Vec<u8> = Vec::with_capacity(8 + 16 + 256);
            params.extend(floats(&[m.orbit_radius * m.orbit_radius]));
            for value in [m.max_iterations, u32::try_from(batch.len()).map_err(too_large)?, u32::try_from(starts.len()).map_err(too_large)?, u32::try_from(orbit.len()).map_err(too_large)?] {
                params.extend(value.to_le_bytes());
            }
            for list in [starts, lengths] {
                for level in 0..32 {
                    params.extend(list.get(level).copied().unwrap_or(0).to_le_bytes());
                }
            }
            params.extend(scaled.to_le_bytes());
            params.extend(0_u32.to_le_bytes());
            //(The struct is padded to a multiple of 8 bytes, of 4 in 32-bit floats)
            let align = if single { 4 } else { 8 };
            params.resize(params.len().div_ceil(align) * align, 0);
            let flat: Vec<f64> = batch.iter().flat_map(|(x, y)| [*x, *y]).collect();
            let (params_buffer, offsets_buffer) = (buffer_with("deep params", &params), buffer_with("deep offsets", &floats(&flat)));
            let size = (batch.len() * if single { std::mem::size_of::<DeepOutcome32>() } else { std::mem::size_of::<DeepOutcome>() }) as u64;
            let out = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("deep outcomes"), size: size.max(16), usage: storage | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("deep read"), size: size.max(16), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: orbit_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: table_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: residual_buffer.as_entire_binding() },
            ] });
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(u32::try_from(batch.len().div_ceil(64)).unwrap_or(u32::MAX).clamp(1, DEEP_WORKGROUPS), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &read, 0, size.max(16));
            let index = self.queue.submit([encoder.finish()]);
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            //(One batch at a time: the buffers of a 4K keyframe's 33 M pixels at once would be gigabytes)
            self.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).map_err(|err| format!("{}: {err}", self.name))?;
            {
                let view = read.slice(..).get_mapped_range().map_err(|err| format!("{}: {err:?}", self.name))?;
                if single {
                    outcomes.extend(bytemuck::cast_slice::<u8, DeepOutcome32>(&view)[..batch.len()].iter().map(|outcome| DeepOutcome { zx: f64::from(outcome.zx), zy: f64::from(outcome.zy), steps: outcome.steps, escaped: outcome.escaped }));
                } else {
                    outcomes.extend_from_slice(&bytemuck::cast_slice::<u8, DeepOutcome>(&view)[..batch.len()]);
                }
            }
            read.unmap();
        }
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(()), |error| Err(format!("{}: {error}", self.name)))?;
        Ok(outcomes)
    }

    /// Whether the GPU computes the adaptive mask (`adaptive_mask`): when the self-test found it the CPU's, bit for bit
    pub(crate) fn mask_works(&self) -> bool {
        self.mask_works.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The adaptive mask of a `width` x `height` render without a region (`adaptive::mask_where` with every pixel inside): with `first` the colors of the
    /// first samples and `second` those after the uniform samples
    ///
    /// # Errors
    /// The GPU failed, or there is no such shader
    pub(crate) fn adaptive_mask(&self, first: &[TrueColor], second: &[TrueColor], (width, height): (usize, usize), threshold: u32) -> Result<Vec<bool>, String> {
        let (pipeline, layout) = self.mask.as_ref().ok_or_else(|| String::from("the mask shader did not compile"))?;
        let pixels = width * height;
        if first.len() != pixels || second.len() != pixels {
            return Err(String::from("the colors do not fit the size"));
        }
        let too_large = |_| String::from("too large");
        let words = |colors: &[TrueColor]| {
            let mut bytes = bytemuck::cast_slice::<TrueColor, u8>(colors).to_vec();
            bytes.resize(bytes.len().div_ceil(4) * 4 + 4, 0);
            bytes
        };
        let _gate = gate();
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let buffer_with = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.len() as u64).max(16), usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.queue.write_buffer(&buffer, 0, bytes);
            buffer
        };
        let (w, h, t) = (u32::try_from(width).map_err(too_large)?, u32::try_from(height).map_err(too_large)?, threshold);
        let storage = wgpu::BufferUsages::STORAGE;
        let params = [buffer_with("mask params 0", bytemuck::cast_slice(&[w, h, t, 0_u32]), storage), buffer_with("mask params 1", bytemuck::cast_slice(&[w, h, t, 1_u32]), storage)];
        let (first, second) = (buffer_with("mask first", &words(first), storage), buffer_with("mask second", &words(second), storage));
        let size = (pixels as u64 * 4).max(16);
        let marked = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("marked"), size, usage: storage, mapped_at_creation: false });
        let mask = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mask"), size, usage: storage | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mask read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for params in &params {
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: first.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: second.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: marked.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mask.as_entire_binding() },
            ] });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(u32::try_from(pixels.div_ceil(64)).unwrap_or(u32::MAX).clamp(1, MASK_WORKGROUPS), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&mask, 0, &read, 0, size);
        let index = self.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(()), |error| Err(format!("{}: {error}", self.name)))?;
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).map_err(|err| format!("{}: {err}", self.name))?;
        let flags = {
            let view = read.slice(..).get_mapped_range().map_err(|err| format!("{}: {err:?}", self.name))?;
            bytemuck::cast_slice::<u8, u32>(&view)[..pixels].iter().map(|flag| *flag != 0).collect()
        };
        read.unmap();
        Ok(flags)
    }

    /// Whether the GPU's adaptive mask is the CPU's for a 61 x 37 image of random colors with features of all sizes, at several thresholds
    fn check_mask(&self) -> Result<bool, String> {
        let (width, height) = (61, 37);
        let mut seed = 777_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u8
        };
        let second: Vec<TrueColor> = (0..width * height).map(|index| if index % 5 < 2 { TrueColor::new(next(), next(), next()) } else { TrueColor::new(9, 99, 199) }).collect();
        let first: Vec<TrueColor> = second.iter().map(|color| if next() < 30 { TrueColor::new(color.red, next(), color.blue) } else { *color }).collect();
        for threshold in [0, 12, 90, 400] {
            if self.adaptive_mask(&first, &second, (width, height), threshold)? != super::adaptive::mask_where(width, height, &|_, _| true, &first, &second, threshold) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether the fast loop colors and sums on the GPU (`submit_colored`): when the self-test found its sums those of the CPU's coloring, bit for bit
    pub(crate) fn colored_work(&self) -> bool {
        self.colored_works.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The sums of a frame of `pixels` pixels on the GPU, all 0, and the color of every position (`table`, as `Render::run` makes it), for `submit_colored`
    ///
    /// # Errors
    /// The GPU could not make the buffers
    pub(crate) fn color_sums(&self, table: &[TrueColor], pixels: usize) -> Result<ColorSums, String> {
        let packed: Vec<u32> = table.iter().map(|color| u32::from(color.red) | (u32::from(color.green) << 8) | (u32::from(color.blue) << 16)).collect();
        let _gate = gate();
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let table = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("colors"), size: (packed.len() as u64 * 4).max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&table, 0, bytemuck::cast_slice(&packed));
        let size = (pixels as u64 * 12).max(16);
        let sums = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("sums"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(ColorSums { table, sums, size }), |error| Err(format!("{}: {error}", self.name)))
    }

    /// `submit_pixels` (the fast loop) that colors each sample with the table of `target` and adds it to its sums on the GPU instead of returning its position:
    /// pixel `(x, y)` of the batch adds to sum `y * width + x`. `finish` gives no outcomes for it; `read_sums` the sums
    ///
    /// # Errors
    /// The GPU failed, or there is no such loop
    #[allow(clippy::too_many_arguments)] // `submit_pixels` with the frame's width and sums
    pub(crate) fn submit_colored(&self, m: &MandelbrotSet, anchor: &ComplexPlane, sample: u8, origin: (isize, isize), pixels: &[u32], width: usize, target: &ColorSums) -> Result<Submitted, String> {
        let (pipeline, layout) = self.colored.as_ref().ok_or_else(|| String::from("the coloring fast loop did not compile"))?;
        let mut params = pixel_params(m, anchor, sample, origin, pixels.len())?;
        let at = params.len() - 4;
        params[at..].copy_from_slice(&u32::try_from(width).map_err(|_| String::from("too wide"))?.to_le_bytes());
        let count = u32::try_from(pixels.len()).map_err(|_| String::from("too many points"))?;
        let _gate = gate();
        let (out_of_memory, validation) = (self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), self.device.push_error_scope(wgpu::ErrorFilter::Validation));
        let buffers = self.buffers(std::mem::size_of_val(pixels) as u64, 0);
        self.queue.write_buffer(&buffers.params, 0, &params);
        if !pixels.is_empty() {
            self.queue.write_buffer(&buffers.input, 0, bytemuck::cast_slice(pixels));
        }
        self.queue.write_buffer(&buffers.next_point, 0, &0_u32.to_le_bytes());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffers.params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffers.input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffers.outcomes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buffers.next_point.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: target.table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: target.sums.as_entire_binding() },
        ] });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(64).min(WORKGROUPS), 1, 1);
        }
        let index = self.queue.submit([encoder.finish()]);
        let errors = [pollster::block_on(validation.pop()), pollster::block_on(out_of_memory.pop())];
        errors.into_iter().flatten().next().map_or(Ok(Submitted { buffers, size: 0, index }), |error| Err(format!("{}: {error}", self.name)))
    }

    /// The sums of `target` (red, green, blue of every pixel), once its dispatches are finished
    ///
    /// # Errors
    /// The GPU failed
    pub(crate) fn read_sums(&self, target: &ColorSums) -> Result<Vec<u32>, String> {
        let _gate = gate();
        let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("sums read"), size: target.size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&target.sums, 0, &read, 0, target.size);
        let index = self.queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None }).map_err(|err| format!("{}: {err}", self.name))?;
        let sums = {
            let view = read.slice(..).get_mapped_range().map_err(|err| format!("{}: {err:?}", self.name))?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        read.unmap();
        Ok(sums)
    }

    /// Whether the coloring fast loop gives the sums of the CPU's coloring of the fast loop's positions (`interpolated_color` with a table of distinct colors)
    /// for sample 3 of the 64 x 48 pixels of `plane`, bit for bit
    fn check_colored(&self, plane: &ComplexPlane, m: &MandelbrotSet) -> Result<bool, String> {
        let (pixels, _) = fast_test_case(plane, m, 64, 48);
        let table: Vec<TrueColor> = (0..=m.max_iterations).map(|index| TrueColor::new((index * 37 % 251) as u8, (index * 11 % 253) as u8, (index * 97 % 241) as u8)).collect();
        let positions = self.fast_positions(m, plane, 3, (5, -2), &pixels)?;
        let target = self.color_sums(&table, 64 * 48)?;
        self.finish::<FastOutcome>(self.submit_colored(m, plane, 3, (5, -2), &pixels, 64, &target)?)?;
        let sums = self.read_sums(&target)?;
        Ok(positions.iter().enumerate().all(|(index, position)| {
            let color = m.interpolated_color(*position, |iterations| table[iterations as usize]);
            sums[3 * index..3 * index + 3] == [u32::from(color.red), u32::from(color.green), u32::from(color.blue)]
        }))
    }

    /// Whether the exact loop fed pixels (`submit_exact_pixels`) gives the CPU's positions (`MandelbrotSet::position` of `sample_point`) for sample 3 of the
    /// 64 x 48 pixels of `plane`, placed at origin (5, -2), bit for bit
    fn check_exact_pixels(&self, plane: &ComplexPlane, m: &MandelbrotSet) -> Result<bool, String> {
        let (pixels, _) = fast_test_case(plane, m, 64, 48);
        let points: Vec<Complex> = (0..48).flat_map(|y| (0..64).map(move |x| (x, y))).map(|(x, y)| super::progressive::sample_point(plane, x, y, 3, (5, -2))).collect();
        let outcomes: Vec<Outcome> = self.finish(self.submit_exact_pixels(m, plane, 3, (5, -2), &pixels)?)?;
        Ok(outcomes.len() == points.len() && points.iter().zip(&outcomes).all(|(c, outcome)| outcome.position_continued(m).to_bits() == m.position(c).to_bits()))
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
            //The exact loop that computes its points: its outcomes must be those of the exact loop with the CPU's points, bit for bit, in every case
            if index == 0 || self.exact_pixels_work() {
                self.exact_pixels_works.store(self.check_exact_pixels(&plane, &m).unwrap_or(false), std::sync::atomic::Ordering::Relaxed);
            }
            if index == 0 {
                self.mask_works.store(self.check_mask().unwrap_or(false), std::sync::atomic::Ordering::Relaxed);
                self.deep_works.store(super::deep::gpu_agrees(self), std::sync::atomic::Ordering::Relaxed);
                self.deep32_works.store(self.deep32.is_some() && super::deep::gpu32_agrees(self) >= 0.9, std::sync::atomic::Ordering::Relaxed);
            }
            if index == 0 || self.colored_work() {
                self.colored_works.store(self.check_colored(&plane, &m).unwrap_or(false), std::sync::atomic::Ordering::Relaxed);
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

/// The line of `gpu.wgsl` that writes an outcome, and what the exact loop for pixels writes instead (see `exact_pixels_source`)
const EXACT_OUTCOME: &str = "            outcomes[index] = Outcome(select(0.0lf, zx, escaped), select(0.0lf, zy, escaped), iterations, select(0u, 1u, escaped && iterations < max_iterations));";
const EXACT_OUTCOME_CONTINUED: &str = "            var extra = 0u;
            if (escaped && iterations < max_iterations && params.smooth_coloring != 0u) {
                for (var more = 0u; more < 32u; more++) {
                    if (zx * zx + zy * zy >= 65536.0lf) {
                        break;
                    }
                    let continued_x = zx * zx - zy * zy + cx;
                    let continued_y = 2.0lf * zx * zy + cy;
                    zx = continued_x;
                    zy = continued_y;
                    extra += 1u;
                }
            }
            outcomes[index] = Outcome(select(0.0lf, zx, escaped), select(0.0lf, zy, escaped), iterations, select(0u, 1u | (extra << 1u), escaped && iterations < max_iterations));";

/// The workgroups of 64 invocations of a pass of the mask: each invocation takes pixels a whole dispatch apart (see `gpu_mask.wgsl`)
const MASK_WORKGROUPS: u32 = 4096;

/// The workgroups of 64 invocations of a deep dispatch: each invocation iterates pixels a whole dispatch apart (see `gpu_deep.wgsl`)
const DEEP_WORKGROUPS: u32 = 8192;

/// The input of the deep loop besides the offsets (see `GpuIterator::deep_outcomes`): the reference orbit and the rounding residuals of its points, and the
/// bilinear approximations (five numbers each: A, B and the square of the radius), whose level `l` starts at `starts[l]` and has `lengths[l]` of them
#[derive(Clone, Copy)]
pub(crate) struct DeepInput<'a> {
    pub(crate) orbit: &'a [(f64, f64)],
    pub(crate) residuals: &'a [DeepResidual],
    pub(crate) table: &'a [f64],
    pub(crate) starts: &'a [u32],
    pub(crate) lengths: &'a [u32],
}

/// `DeepOutcome` of the 32-bit deep loop (`deep32_source`)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepOutcome32 {
    zx: f32,
    zy: f32,
    steps: u32,
    escaped: u32,
}

/// The deep loop in 32-bit floats (issue #288), from `gpu_deep.wgsl`: the same steps in f32, the scaled phase below 2^-100 instead of 2^-1000 (f32's range
/// ends at about 10⁻³⁸), powers of two from f32's exponent bits, and the split and the band of the escape test for f32's 24 bits. An approximation of the
/// f64 loop (off by default); the residuals of the escape test next to the circle are those of the f64 orbit, so the tip c = -2 is only exact in f64
fn deep32_source() -> String {
    let mut source = include_str!("gpu_deep.wgsl").to_string();
    for (from, to) in [
        ("bitcast<f64>(u64(u32(1023 + step)) << 52u)", "bitcast<f32>(u32(127 + step) << 23u)"),
        ("clamp(left, -1000, 1000)", "clamp(left, -100, 100)"),
        ("134217729.0lf", "4097.0lf"),
        ("1e-12lf", "1e-5lf"),
        ("1e300lf", "1e30lf"),
        ("-500)", "-50)"),
        ("scaled - 500", "scaled - 50"),
        ("scaled <= 1000", "scaled <= 100"),
        ("params.scaled > 1000", "params.scaled > 100"),
        ("f64", "f32"),
        ("lf", "f"),
    ] {
        source = source.replace(from, to);
    }
    source
}

/// The rounding residual of a point of the reference orbit (`rendering::deep::Residual`): `mx · 2^-px` and `my · 2^-py`
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DeepResidual {
    pub(crate) mx: f64,
    pub(crate) my: f64,
    pub(crate) px: i32,
    pub(crate) py: i32,
}

/// The outcome of the deep loop for one pixel (`gpu_deep.wgsl`): the pixel's orbit when it escaped (1; 2: in the scaled phase, where it is the center's),
/// after `steps` steps
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DeepOutcome {
    pub(crate) zx: f64,
    pub(crate) zy: f64,
    pub(crate) steps: u32,
    pub(crate) escaped: u32,
}

/// The sums of a frame on the GPU and the colors of the positions, see `GpuIterator::color_sums`
pub(crate) struct ColorSums {
    table: wgpu::Buffer,
    sums: wgpu::Buffer,
    size: u64,
}

/// The fast loop (`gpu_fast.wgsl`) that colors its samples and adds them to the sums of the frame: instead of writing the position it computes
/// `FastOutcome::position` and `MandelbrotSet::interpolated_color` with the table of colors, in f64 and integers like the CPU, and adds the color to the
/// pixel's sums with atomic additions (integers, so the order of the samples does not matter). The frame's width is in the last word of the parameters
fn colored_source() -> String {
    const COLORING: &str = "@group(0) @binding(4) var<storage, read> table: array<u32>;
@group(0) @binding(5) var<storage, read_write> sums: array<atomic<u32>>;

// `FastOutcome::position`, then `MandelbrotSet::interpolated_color` with `table`, added to the sums of the pixel of batch index `index`
fn add_color(index: u32, outcome: Outcome) {
    let last = select(params.max_iterations - 1u, 0u, params.max_iterations == 0u);
    var position = f64(outcome.whole);
    if ((outcome.whole & 0x80000000u) != 0u) {
        position = clamp(f64(outcome.whole & 0x7FFFFFFFu) + f64(outcome.fraction), 0.0lf, f64(last));
    }
    let at = u32(position);
    let weight = u32((position - f64(at)) * 256.0lf);
    var color = table[at];
    if (weight != 0u && at < params.max_iterations) {
        let next = table[at + 1u];
        var mixed = 0u;
        for (var shift = 0u; shift < 24u; shift += 8u) {
            let a = (color >> shift) & 255u;
            let b = (next >> shift) & 255u;
            mixed |= ((a * (256u - weight) + b * weight + 128u) >> 8u) << shift;
        }
        color = mixed;
    }
    let pixel = pixels[index];
    let sum = 3u * ((pixel >> 16u) * params.pad + (pixel & 0xFFFFu));
    atomicAdd(&sums[sum], color & 255u);
    atomicAdd(&sums[sum + 1u], (color >> 8u) & 255u);
    atomicAdd(&sums[sum + 2u], (color >> 16u) & 255u);
}
";
    let fast = include_str!("gpu_fast.wgsl");
    let bindings_end = fast.find("@group(0) @binding(3)").expect("the shader has binding 3");
    let line_end = bindings_end + fast[bindings_end..].find('\n').expect("the line ends") + 1;
    let source = format!("{}\n{COLORING}{}", &fast[..line_end], &fast[line_end..]);
    source.replacen("                outcomes[index] = Outcome(params.max_iterations, 0.0);", "                add_color(index, Outcome(params.max_iterations, 0.0));", 1)
        .replacen("            outcomes[index] = finished(iterations, escaped && iterations < max_iterations, zx, zy, cx, cy);", "            add_color(index, finished(iterations, escaped && iterations < max_iterations, zx, zy, cx, cy));", 1)
}

/// The parameters of a dispatch over pixels (`gpu_fast.wgsl` and `exact_pixels_source`): where the pixels are in `anchor` and the loop's settings
fn pixel_params(m: &MandelbrotSet, anchor: &ComplexPlane, sample: u8, origin: (isize, isize), count: usize) -> Result<Vec<u8>, String> {
    let to_i32 = |value: isize| i32::try_from(value).map_err(|_| String::from("the origin is too far away"));
    let mut params = Vec::with_capacity(72);
    for value in [anchor.min_x, anchor.min_y, anchor.increment_x, anchor.increment_y, m.orbit_radius * m.orbit_radius] {
        params.extend(value.to_le_bytes());
    }
    for value in [m.max_iterations, u32::try_from(count).map_err(|_| String::from("too many points"))?, u32::from(m.orbit_radius >= 2.0), u32::from(m.smooth_coloring)] {
        params.extend(value.to_le_bytes());
    }
    for value in [to_i32(origin.0)?, to_i32(origin.1)?] {
        params.extend(value.to_le_bytes());
    }
    for value in [u32::from(sample), 0] {
        params.extend(value.to_le_bytes());
    }
    Ok(params)
}

/// The exact loop (`gpu.wgsl`) fed pixels instead of points: it takes the parameters and the pixels of the fast loop, and computes the point of each with
/// `sample_point` of `gpu_fast.wgsl` (the CPU's jitter and arithmetic, in u64 and f64), so its outcomes are the exact loop's for the CPU's points as long as the
/// GPU's f64 fused multiply-add rounds once, as the CPU's does (the self-test checks that, see `check_exact_pixels`)
fn exact_pixels_source() -> String {
    let (exact, fast) = (include_str!("gpu.wgsl"), include_str!("gpu_fast.wgsl"));
    let between = |text: &'static str, from: &str, to: &str| -> &'static str {
        let start = text.find(from).expect("the shader has the part");
        &text[start..start + text[start..].find(to).expect("the shader has the end of the part")]
    };
    let fast_params = between(fast, "struct Params {", "}\n");
    let point_of_pixel = between(fast, "// `jitter::splitmix64`", "// `MandelbrotSet::is_known_interior`");
    let (exact_params, exact_point) = (between(exact, "struct Params {", "}\n"), between(exact, "struct Point {", "}\n"));
    exact.replacen(exact_params, fast_params, 1)
        .replacen(&format!("{exact_point}}}\n"), "", 1)
        .replacen("@group(0) @binding(1) var<storage, read> points: array<Point>;", &format!("@group(0) @binding(1) var<storage, read> pixels: array<u32>;\n\n{point_of_pixel}"), 1)
        .replace("points[index].x", "sample_point(index).x")
        .replace("points[index].y", "sample_point(index).y")
        //An escaped orbit is continued here for smooth coloring, as `MandelbrotSet::smooth_position` does (to |z|² ≥ 65536, at most 32 steps), and the steps it
        //added go into `escaped` above its lowest bit, so the CPU only takes the logarithms (see `Outcome::position_continued`)
        .replacen(EXACT_OUTCOME, EXACT_OUTCOME_CONTINUED, 1)
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
        no_contraction_everywhere(&exact_pixels_source());
        no_contraction_everywhere(&colored_source());
        super::spirv_without_contraction(include_str!("gpu_mask.wgsl")).expect("the mask shader compiles");
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

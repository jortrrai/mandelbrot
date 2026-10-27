//Temporary file to group together all rendering functionality
use std::{time::Instant, thread, sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering}}, io::{self, Write}};

use crate::{pixel_buffer::PixelBuffer, complex_plane::ComplexPlane, mandelbrot_set::{MandelbrotSet, LANES}, complex::Complex, coloring::TrueColor};

/// The largest `max_iterations` for which `render_region_progressively` precomputes the color of every iteration count
const MAX_COLOR_TABLE_ITERATIONS: u32 = 200_000;

/// SplitMix64 finalizer, a well-mixed 64-bit hash: <https://prng.di.unimi.it/splitmix64.c>
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic supersampling jitter in [0, 1) x [0, 1) for sample number `sample` of the pixel at (`x`, `y`).
/// Depends only on its arguments, not on thread scheduling, so identical inputs always render identical images.
fn sample_jitter(x: usize, y: usize, sample: u8) -> (f64, f64) {
    let h = splitmix64(splitmix64(splitmix64(x as u64) ^ y as u64) ^ u64::from(sample));
    let to_unit = |bits: u64| (bits >> 11) as f64 / (1u64 << 53) as f64; //53 random bits, like rand's f64 sampling
    (to_unit(h), to_unit(splitmix64(h)))
}

/// The amount of threads that rendering functions use: all logical CPU threads except one, which is left free for the rest of the system and the GUI, but at least one
pub fn render_thread_count() -> usize {
    num_cpus::get().saturating_sub(1).max(1)
}

///A box representing the area to render by rendering functions
#[derive(Clone,Copy)]
pub struct RenderBox {
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize
}

impl RenderBox {
    pub fn new(min_x: usize, max_x: usize, min_y: usize, max_y: usize) -> RenderBox { 
        RenderBox { min_x, max_x, min_y, max_y } 
    }

    pub fn print(&self) {
        println!("RenderBox: ({},{}) -> ({},{}) {{{} pixels}}",self.min_x,self.min_y,self.max_x,self.max_y, self.compute_pixel_count());
    }

    pub fn compute_pixel_count(&self) -> usize {
        (self.max_x-self.min_x)*(self.max_y-self.min_y)
    }

    ///Returns whether the point (x,y) is inside the `RenderBox`
    pub fn contains(&self, point: (usize, usize)) -> bool {
        !(point.0 < self.min_x || point.0 > self.max_x || point.1 < self.min_y || point.1 > self.max_y)
    }
}


/// Render the Complex plane c into the 32-bit pixel buffer by applying the Mandelbrot formula iteratively to every Complex point mapped to a pixel in the buffer. 
/// The buffer should have a size of width*height.
/// `orbit_radius` determines when Zn is considered to have gone to infinity.
/// `max_iterations` concerns the maximum amount of times the Mandelbrot formula will be applied to each Complex number.
/// Note: This function is computationally intensive, and should not be used for translations
pub fn render_complex_plane_into_buffer(p: &mut PixelBuffer, c: &ComplexPlane, m: &MandelbrotSet, supersampling_amount: u8,coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor) {
    let render_box = RenderBox::new(0, p.pixel_plane.width, 0, p.pixel_plane.height);
    render_box_render_complex_plane_into_buffer(p, c, m, render_box, supersampling_amount,coloring_function);
}

/// Render the Complex plane c into the 32-bit pixel buffer by applying the Mandelbrot formula iteratively to every Complex point mapped to a pixel in the buffer. 
/// The buffer should have a size of width*height.
/// Only renders Pixels inside the render box denoted by `render_min_x`, `render_max_x`, `render_min_y`, `render_max_y`
/// `orbit_radius` determines when Zn is considered to have gone to infinity.
/// `max_iterations` concerns the maximum amount of times the Mandelbrot formula will be applied to each Complex number.
/// Note: This function is computationally intensive, and should not be used for translations
/// Note: This function is multithreaded
/// * `coloring_function` - e.g. `TrueColor::new_from_hsv`
/// # Panics 
/// If `lock().unwrap()` panics
pub fn render_box_render_complex_plane_into_buffer(p: &mut PixelBuffer, c: &ComplexPlane, m: &MandelbrotSet, render_box: RenderBox, supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor) {
    let time = benchmark_start();
    let supersampling_amount = supersampling_amount.clamp(1, 64); //Supersampling_amount should be at least 1 and atmost 64
    render_box.print();
    println!("SSAA: {}x", supersampling_amount);
    let chunk_size = p.pixel_plane.width;
    let chunks: Vec<Vec<TrueColor>> = p.colors.chunks(chunk_size).map(ToOwned::to_owned).collect();
    let chunks_len = chunks.len();
    //println!("chunks.len(): {}", chunks.len());
    let mut handles = Vec::new();
    let amount_of_threads = render_thread_count(); //Amount of CPU threads to use
    let global_mutex = Arc::new(Mutex::new(0));
    let max_progress: u8 = 30;
    let chunks_len_over_max_progress = chunks_len / max_progress as usize;
    let current_progress_atomic: Arc<Mutex<AtomicU8>>= Arc::new(Mutex::new(AtomicU8::new(0)));

    for _thread_id in 0..amount_of_threads {
        let plane = (*c).clone();
        let buf = chunks.clone();
        let thread_mutex = Arc::clone(&global_mutex);
        let pixel_buffer = (*p).clone();
        let ms = (*m).clone();
        let atm = Arc::clone(&current_progress_atomic);

        let handle = thread::spawn(move || {
            let mut thread_chunks = Vec::new();

            loop {
                let mut data = thread_mutex.lock().unwrap();
                let current_chunk = *data;
                *data+=1;
                drop(data);
                if current_chunk >= chunks_len {
                    return thread_chunks;
                }
                //println!("Thread[{}] takes chunk[{}]", thread_id, current_chunk);
                if current_chunk % chunks_len_over_max_progress == 0 {
                    let current_progress = atm.lock().unwrap().load(Ordering::Relaxed);
                    print_progress_bar(current_progress, max_progress);
                    if current_progress < u8::MAX
                    {
                        atm.lock().unwrap().store(current_progress+1, Ordering::Relaxed);
                    }
                }

                let chunk_start = chunk_size * current_chunk;
                let mut chunk = buf[current_chunk].clone();
                
                for (i, pixel) in chunk.iter_mut().enumerate() {
                    let point = pixel_buffer.index_to_point(i + chunk_start);
                    if !render_box.contains(point)
                    {
                        continue; //Do not render Pixel points outside of the render box
                    }
                    let original_x: f64 = f64::from(point.0 as u32);
                    let original_y: f64 = f64::from(point.1 as u32);
                    //Supersampling, see: https://darkeclipz.github.io/fractals/paper/Fractals%20&%20Rendering%20Techniques.html
                    let mut colors: Vec<TrueColor> = Vec::new();
                    for sample in 0..supersampling_amount {
                        let (jitter_x, jitter_y) = sample_jitter(point.0, point.1, sample);
                        let (x, y) : (f64, f64) = (original_x+jitter_x, original_y+jitter_y);
                        let complex = plane.complex_from_pixel_plane(x, y);
                        let iterations = ms.iterate(&complex);
                        let color = coloring_function(iterations, ms.max_iterations);
                        colors.push(color);    
                    }
                    let supersampled_color = TrueColor::average(&colors);
                    *pixel = supersampled_color;
                }
                thread_chunks.push((current_chunk, chunk.clone()));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        let thread_chunks = handle.join().unwrap();
        for (i, chunk) in thread_chunks{
            let mut index = i*p.pixel_plane.width;
            for color in chunk {
                p.colors[index] = color;
                index+=1;
            }
        }
    }
    p.update_pixels();
    println!();
    benchmark("render_box_render_complex_plane_into_buffer()", time);
}

/// The pixels of a buffer that have to be rendered again after the buffer was shifted with `PixelBuffer::translate_buffer`:
/// the exposed `columns` and the exposed `rows`. A pixel is exposed if its column or its row is exposed, so a shift in one direction exposes a strip,
/// and a shift in both directions an L shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposedRegion {
    pub columns: std::ops::Range<usize>,
    pub rows: std::ops::Range<usize>,
}

impl ExposedRegion {
    /// Every pixel of a `width` x `height` buffer
    pub fn full(width: usize, height: usize) -> ExposedRegion {
        ExposedRegion { columns: 0..width, rows: 0..height }
    }

    /// The pixels exposed by shifting a `width` x `height` buffer with `PixelBuffer::translate_buffer(rows_up, columns_right)`.
    /// Returns `None` if the shift is as large as the buffer, since nothing of the old image is then left.
    pub fn from_shift(width: usize, height: usize, rows_up: i128, columns_right: i128) -> Option<ExposedRegion> {
        if rows_up.unsigned_abs() >= height as u128 || columns_right.unsigned_abs() >= width as u128 {
            return None;
        }
        let exposed = |shift: i128, size: usize| {
            let amount = shift.unsigned_abs() as usize;
            if shift > 0 { 0..amount } else { (size - amount)..size } //Content moves towards the end for positive shifts, so the start is exposed
        };
        Some(ExposedRegion { columns: exposed(columns_right, width), rows: exposed(rows_up, height) })
    }

    pub fn contains(&self, x: usize, y: usize) -> bool {
        self.columns.contains(&x) || self.rows.contains(&y)
    }

    /// The amount of pixels in the region of a `width` x `height` buffer
    pub fn pixel_count(&self, width: usize, height: usize) -> usize {
        let (columns, rows) = (self.columns.len().min(width), self.rows.len().min(height));
        columns * height + rows * width - columns * rows
    }
}

/// Moves the complex plane `c` so it is centered on `new_center`, without rendering anything.
/// Returns the `(rows_up, columns_right)` to shift the pixel buffer with `PixelBuffer::translate_buffer`.
pub fn center_complex_plane_on(c: &mut ComplexPlane, new_center: &Complex) -> (i128, i128) {
    let mut translation: Complex = new_center.subtract(&c.center());
    //Mirror the y translation because the screen y is mirrored compared to the complex plane y axis
    translation.y = -translation.y;
    c.translate(translation.x, 0.0);
    let columns = -c.real_to_pixels(translation.x);
    c.translate(0.0, translation.y);
    let rows = -c.imaginary_to_pixels(translation.y);
    (i128::from(rows), i128::from(columns))
}

/// Renders the whole Complex plane `c` progressively, one supersampling layer (pass) at a time, so intermediate images can be shown early.
/// See `render_region_progressively`.
pub fn render_progressively(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor, cancel: &AtomicBool, pixels_done: &AtomicUsize, on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
    let base = vec![TrueColor::new(0, 0, 0); width * height];
    render_region_progressively(c, m, width, height, &ExposedRegion::full(width, height), &base, supersampling_amount, coloring_function, cancel, pixels_done, on_pass)
}

/// Renders the pixels of the Complex plane `c` that are inside `region` progressively, one supersampling layer (pass) at a time, so intermediate images can be shown early.
/// All other pixels keep the colors of `base`, e.g. the shifted image of a previous render, so moving the view only renders the newly exposed pixels.
/// Pass `k` computes sample `k` of every pixel in the region, with the same deterministic jitter as `render_box_render_complex_plane_into_buffer`,
/// so after the last pass the colors are identical to the ones that function produces.
/// `pixels_done` is reset to 0 at the start of every pass and incremented for every finished pixel, so another thread can show how far the current pass is.
/// After every completed pass, `on_pass` is called with the colors (`width * height`, row by row): `base`, with the region replaced by the running average of all samples so far, and the pass number (1-based).
/// Returns `false` if `cancel` was set before all passes completed; the last completed pass has then been reported. A partially computed pass is discarded.
/// Note: This function is multithreaded; `on_pass` is called on the calling thread
/// # Panics
/// If `lock().unwrap()` panics, or if `base` does not have `width * height` colors
pub fn render_region_progressively(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, region: &ExposedRegion, base: &[TrueColor], supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor, cancel: &AtomicBool, pixels_done: &AtomicUsize, mut on_pass: impl FnMut(&[TrueColor], u8)) -> bool {
    assert_eq!(base.len(), width * height, "base should have a color for every pixel");
    let supersampling_amount = supersampling_amount.clamp(1, 64);
    let amount_of_threads = render_thread_count();
    let mut sums: Vec<[u32; 3]> = vec![[0; 3]; width * height];
    let mut colors: Vec<TrueColor> = base.to_vec();
    //The color of a pixel only depends on its iteration count, which is at most `max_iterations`, so compute every color once
    let color_table: Option<Vec<TrueColor>> = (m.max_iterations <= MAX_COLOR_TABLE_ITERATIONS).then(|| (0..=m.max_iterations).map(|iterations| coloring_function(iterations, m.max_iterations)).collect());
    for sample in 0..supersampling_amount {
        pixels_done.store(0, Ordering::Relaxed);
        let rows = Mutex::new(sums.chunks_mut(width).enumerate());
        thread::scope(|scope| {
            for _ in 0..amount_of_threads {
                scope.spawn(|| loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let next_row = rows.lock().unwrap().next();
                    let Some((y, row)) = next_row else { return };
                    let xs: Vec<usize> = (0..width).filter(|x| region.contains(*x, y)).collect();
                    let sample_point = |x: usize| {
                        let (jitter_x, jitter_y) = sample_jitter(x, y, sample);
                        c.complex_from_pixel_plane(f64::from(x as u32) + jitter_x, f64::from(y as u32) + jitter_y)
                    };
                    let mut add_color = |x: usize, iterations: u32| {
                        let color = color_table.as_ref().map_or_else(|| coloring_function(iterations, m.max_iterations), |table| table[iterations as usize]);
                        let sum = &mut row[x];
                        sum[0] += u32::from(color.red);
                        sum[1] += u32::from(color.green);
                        sum[2] += u32::from(color.blue);
                    };
                    //Iterate LANES pixels at the same time, the remaining pixels one by one
                    let mut groups = xs.chunks_exact(LANES);
                    for group in &mut groups {
                        let points: [Complex; LANES] = std::array::from_fn(|lane| sample_point(group[lane]));
                        let iterations = m.iterate_lanes(&points);
                        for (lane, x) in group.iter().enumerate() {
                            add_color(*x, iterations[lane]);
                        }
                    }
                    for x in groups.remainder() {
                        add_color(*x, m.iterate(&sample_point(*x)));
                    }
                    let done = xs.len();
                    pixels_done.fetch_add(done, Ordering::Relaxed);
                });
            }
        });
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let divisor = u32::from(sample) + 1;
        for (index, (sum, color)) in sums.iter().zip(colors.iter_mut()).enumerate() {
            if region.contains(index % width, index / width) {
                *color = TrueColor::new((sum[0] / divisor) as u8, (sum[1] / divisor) as u8, (sum[2] / divisor) as u8);
            }
        }
        on_pass(&colors, sample + 1);
    }
    true
}

pub fn translate_and_render_complex_plane_buffer(p: &mut PixelBuffer, c: &ComplexPlane, m: &MandelbrotSet, rows: i128, columns: i128, supersampling_amount: u8,coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor) {
    println!("rows: {}, columns: {}",rows, columns);
    let max_x: usize = if columns > 0 {columns as usize} else {p.pixel_plane.width-1};
    let max_y: usize = if rows > 0 {rows as usize} else {p.pixel_plane.height-1};
    p.translate_buffer(rows, columns);
    if rows == 0 {
        let render_box = RenderBox::new((max_x as i128-columns.abs()) as usize, max_x, 0, p.pixel_plane.height);
        render_box_render_complex_plane_into_buffer(p, c, m, render_box, supersampling_amount, coloring_function);
    }
    else if columns == 0 {
        let render_box = RenderBox::new(0, p.pixel_plane.width, (max_y as i128 -rows.abs()) as usize, max_y);
        render_box_render_complex_plane_into_buffer(p, c, m, render_box, supersampling_amount, coloring_function);
    } else {
        println!("ERROR: translate_and_render_complex_plane_buffer() requires that rows == 0 || columns == 0");
    }
}

///# Panics
/// If `rows_up` != 0 && `columns_right` != 0
/// Moves the complex plane `c` up by `rows_up` and right by `columns_right` pixels, without rendering anything
pub fn translate_complex_plane_by_pixels(c: &mut ComplexPlane, rows_up: i16, columns_right: i16) {
    let row_sign: f64 = if rows_up > 0 {-1.0} else {1.0};
    let column_sign: f64 = if columns_right > 0 {1.0} else {-1.0};
    c.translate(column_sign*c.pixels_to_real(columns_right.unsigned_abs() as u8), row_sign*c.pixels_to_imaginary(rows_up.unsigned_abs() as u8)); 
}

pub fn translate_and_render_efficiently(c: &mut ComplexPlane, p: &mut PixelBuffer, m: &MandelbrotSet, rows_up: i16, columns_right: i16, supersampling_amount: u8,coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor) {
    assert!(rows_up == 0 || columns_right == 0, "translate_and_render_efficiently: rows_up should be 0 or columns_right should be 0!");

    translate_complex_plane_by_pixels(c, rows_up, columns_right);
    translate_and_render_complex_plane_buffer(p, c, m, rows_up.into(), (-columns_right).into(), supersampling_amount, coloring_function);
}

pub fn translate_to_center_and_render_efficiently(c: &mut ComplexPlane, p: &mut PixelBuffer, m: &MandelbrotSet, new_center: &Complex, supersampling_amount: u8, coloring_function: fn(iterations: u32, max_iterations: u32) -> TrueColor) {
    let mut translation: Complex = new_center.subtract(&c.center());
    //Mirror the y translation because the screen y is mirrored compared to the complex plane y axis
    translation.y = -translation.y;

    //Translate x, to the right
    c.translate(translation.x, 0.0);
    let columns_right = -c.real_to_pixels(translation.x);
    dbg!(columns_right);
    translate_and_render_complex_plane_buffer(p, c, m, 0, columns_right.into(), supersampling_amount, coloring_function);

    //Translate y, up
    c.translate(0.0, translation.y);
    let rows_up = -c.imaginary_to_pixels(translation.y);
    dbg!(rows_up);
    translate_and_render_complex_plane_buffer(p, c, m, rows_up.into(), 0, supersampling_amount, coloring_function);
}

fn benchmark_start() -> Instant {
    Instant::now()
}

fn benchmark(function: &str,time: Instant) {
    println!("[Benchmark] {}: {:.2?}",function, time.elapsed());
}

///Prints a progress bar on the current line, will print over the contents of the cursor's current line, so make sure the function is given a newline to print over
fn print_progress_bar(current_progress: u8, max_progress: u8) {
    print!("\rProgress: ["); //Print a \r carriage return to return the cursor to the beginning of the line: https://stackoverflow.com/questions/59890270/how-do-i-overwrite-console-output
    for i in 0..max_progress {
        let symbol = if i <= current_progress {'+'} else {'.'};
        print!("{}", symbol);
    }
    print!("]");
    io::stdout().flush().unwrap();
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel_buffer::pixel_plane::PixelPlane;

    fn render(width: usize, height: usize, supersampling_amount: u8) -> Vec<(u8, u8, u8)> {
        let mut p = PixelBuffer::new(PixelPlane::new(width, height));
        let c = ComplexPlane::new(width, height);
        let m = MandelbrotSet::new(200, 2.0);
        render_complex_plane_into_buffer(&mut p, &c, &m, supersampling_amount, TrueColor::new_from_bernstein_polynomials);
        p.colors.iter().map(|c| (c.red, c.green, c.blue)).collect()
    }

    #[test]
    fn render_thread_count_leaves_one_thread_free_but_is_at_least_one() {
        let threads = render_thread_count();
        assert!(threads >= 1);
        assert_eq!(threads, (num_cpus::get() - 1).max(1));
    }

    #[test]
    fn rendering_is_reproducible() {
        for ssaa in [1, 4] {
            let first = render(64, 48, ssaa);
            for _ in 0..3 {
                assert_eq!(first, render(64, 48, ssaa), "ssaa = {ssaa}");
            }
        }
    }

    #[test]
    fn progressive_rendering_matches_the_normal_render_and_reports_every_pass() {
        let (width, height, supersampling_amount) = (48, 32, 4);
        let expected = render(width, height, supersampling_amount);
        let c = ComplexPlane::new(width, height);
        let m = MandelbrotSet::new(200, 2.0);
        let pixels_done = AtomicUsize::new(0);
        let mut passes = Vec::new();
        let mut last = Vec::new();
        let completed = render_progressively(&c, &m, width, height, supersampling_amount, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &pixels_done, |colors, pass| {
            passes.push(pass);
            last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect();
        });
        assert!(completed);
        assert_eq!(pixels_done.load(Ordering::Relaxed), width * height, "every pixel of the last pass should be counted");
        assert_eq!(passes, [1, 2, 3, 4]);
        assert_eq!(last, expected, "the last pass should equal the normal render");
    }

    #[test]
    fn exposed_region_covers_the_pixels_a_shift_exposes() {
        assert_eq!(ExposedRegion::from_shift(10, 8, 0, 3), Some(ExposedRegion { columns: 0..3, rows: 8..8 }), "content moves right: the left strip is exposed");
        assert_eq!(ExposedRegion::from_shift(10, 8, 0, -3), Some(ExposedRegion { columns: 7..10, rows: 8..8 }), "content moves left: the right strip is exposed");
        assert_eq!(ExposedRegion::from_shift(10, 8, 2, 0), Some(ExposedRegion { columns: 10..10, rows: 0..2 }), "content moves down: the top strip is exposed");
        assert_eq!(ExposedRegion::from_shift(10, 8, -2, 0), Some(ExposedRegion { columns: 10..10, rows: 6..8 }), "content moves up: the bottom strip is exposed");
        assert_eq!(ExposedRegion::from_shift(10, 8, 8, 0), None, "a shift as large as the buffer exposes everything");
        let l_shape = ExposedRegion::from_shift(10, 8, 2, 3).unwrap();
        assert_eq!(l_shape.pixel_count(10, 8), 3 * 8 + 2 * 10 - 3 * 2);
        assert_eq!((0..10).flat_map(|x| (0..8).map(move |y| (x, y))).filter(|(x, y)| l_shape.contains(*x, *y)).count(), l_shape.pixel_count(10, 8));
    }

    #[test]
    fn region_rendering_only_changes_the_region_and_matches_the_normal_render_there() {
        let (width, height, supersampling_amount) = (48, 32, 3);
        let full = render(width, height, supersampling_amount);
        let c = ComplexPlane::new(width, height);
        let m = MandelbrotSet::new(200, 2.0);
        let base = vec![TrueColor::new(1, 2, 3); width * height]; //Stands for an old image
        let region = ExposedRegion::from_shift(width, height, 4, -5).unwrap();
        let pixels_done = AtomicUsize::new(0);
        let mut last = Vec::new();
        let mut passes = 0;
        render_region_progressively(&c, &m, width, height, &region, &base, supersampling_amount, TrueColor::new_from_bernstein_polynomials, &AtomicBool::new(false), &pixels_done, |colors, pass| {
            passes = pass;
            last = colors.iter().map(|c| (c.red, c.green, c.blue)).collect();
        });
        assert_eq!(passes, 3);
        assert_eq!(pixels_done.load(Ordering::Relaxed), region.pixel_count(width, height), "only the pixels of the region are rendered");
        for (index, color) in last.iter().enumerate() {
            let expected = if region.contains(index % width, index / width) { full[index] } else { (1, 2, 3) };
            assert_eq!(*color, expected, "pixel {index}");
        }
    }

    #[test]
    fn progressive_rendering_can_be_cancelled() {
        let c = ComplexPlane::new(16, 16);
        let m = MandelbrotSet::new(50, 2.0);
        let cancel = AtomicBool::new(false);
        let mut passes = 0;
        let completed = render_progressively(&c, &m, 16, 16, 8, TrueColor::new_from_bernstein_polynomials, &cancel, &AtomicUsize::new(0), |_, pass| {
            passes = pass;
            if pass == 2 {
                cancel.store(true, Ordering::Relaxed);
            }
        });
        assert!(!completed);
        assert_eq!(passes, 2, "no pass should be reported after cancelling");
    }

    #[test]
    fn sample_jitter_is_in_unit_square_and_varies() {
        let mut distinct = std::collections::HashSet::new();
        for (x, y, sample) in (0..20).flat_map(|x| (0..20).flat_map(move |y| (0..4).map(move |s| (x, y, s)))) {
            let (jx, jy) = sample_jitter(x, y, sample);
            assert!((0.0..1.0).contains(&jx) && (0.0..1.0).contains(&jy));
            assert_eq!((jx, jy), sample_jitter(x, y, sample));
            distinct.insert(jx.to_bits());
        }
        assert!(distinct.len() > 1500, "jitter should vary per pixel and sample, got {} distinct values", distinct.len());
    }
}

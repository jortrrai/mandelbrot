# Mandelbrot by Jort
```
 __  __                 _      _ _               _   
|  \/  |               | |    | | |             | |  
| \  / | __ _ _ __   __| | ___| | |__  _ __ ___ | |_ 
| |\/| |/ _` | '_ \ / _` |/ _ \ | '_ \| '__/ _ \| __|
| |  | | (_| | | | | (_| |  __/ | |_) | | | (_) | |_ 
|_|  |_|\__,_|_| |_|\__,_|\___|_|_.__/|_|  \___/ \__|
   __             __         __      .---.
  / /  __ __  __ / /__  ____/ /_    [ o o ]   and his robotic friends:
 / _ \/ // / / // / _ \/ __/ __/     '-v-'    Claude Code, beep boop
/_.__/\_, /  \___/\___/_/  \__/       /|_|\
     /___/
                                 v0.3.0
```

A Mandelbrot set viewer written in Rust, with a graphical window: explore the set, zoom, jump to named places, and save images. The same view and settings always render exactly the same image.

- **Fast**: all CPU threads but one, four points iterated at a time with the CPU's vector instructions, shortcuts for points that are certainly inside the set, and a progressive render that shows every supersampling layer as soon as it is done (and can be cancelled).
- **Smooth**: continuous (smooth) coloring, supersampling up to 64 samples per pixel, and optional adaptive supersampling that only spends the extra samples where the picture shows detail.
- **Automatic iterations**: a maximum iterations that follows your zoom (with Performance, Balanced and High Detail presets), or set it yourself.
- **Named places** with thumbnails, loaded from JSON files you can add to yourself.
- **Saves PNG images** with the view and the settings stored in the metadata.

---
- [Running](#running)
- [The window](#the-window)
- [Points of interest](#points-of-interest)
- [How an image is made](#how-an-image-is-made)
- [Development](#development)
- [Wallpapers](#wallpapers)
- [Credits](#credits)
---

## Running
Install Rust ([rustup.rs](https://www.rust-lang.org/tools/install)), for example on Linux:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Then, in the root of the repository:
```
cargo run --release
```
The program has no command-line arguments: the window starts with the defaults (a 1200 x 800 render that follows the size of the window, 1,000 iterations, 32x quality) and has controls for all of them.

The terminal is only for developers: everything you need is in the window.

## The window
The fractal is on the left. On the right, from top to bottom, are the controls, the current view and the status.

**Controls**
- **Move**: the arrow buttons move the view by the *Move step* in pixels; only the newly exposed strip is rendered.
- **Zoom in / Zoom out / Reset view**, and the *Zoom speed*. **Zoom in** is disabled when the 64-bit numbers of the renderer run out of precision (see the warning under *Current view*). **< Back / Forward >** walk through the views you visited (the last 200), like the history of a browser.
- **Points of interest**: a list of named places with thumbnails (see below); *Reload* reads the folder again; *Restore the saved appearance* also applies the coloring and the color channels of a place.
- **Settings are remembered**: the values and options of the controls (move step, zoom speed, iteration limit and auto iterations, quality, coloring and color channels, smooth colors, the GPU options, flight pace and rendering, and the other checkboxes) are saved to `mandelbrot/settings.json` in your configuration folder (`~/.config` on Linux, `%APPDATA%` on Windows) whenever one changes, and loaded when the window opens. The view, the history and the places are not settings. **Reset settings** (next to the Controls heading) puts every setting back to its default.
- **Flight**: clicking a place flies the view there with smooth panning and zooming (the path of van Wijk and Nuij, which zooms out to travel). By default it takes the shortest path, zooming out only as far as the distance needs; the pace is *Fast* (the default), *Normal* or *Cinematic*, which start and end softly, *Constant*, which keeps one speed from the start to the place (set with the slider next to it, in doublings of the zoom per second, 0.25 to 20), or *No flight*, which shows the place at once. The route is *Shortest* (the default, zooming out to travel) or *Straight*: the center goes in a straight line while the zoom changes steadily from one view to the other, never zooming out on the way, and the picture moves the same number of views every moment (between two far-apart places at a deep zoom it rushes by). With auto iterations, the limit follows the zoom of every frame of the flight: between what auto iterations chose at the start and what it chooses for the place (worked out in the background when the flight starts), so the detail is right on the way and the arrival does not jump. Using any control or the mouse stops a flight where it is; while it flies, a progress bar shows how much of the way is covered, *Cancel* stops it there and *Skip* goes to the place at once. The rendering of a flight is *Best effort* (the default: the camera follows the clock and the picture fills in), *Low res* (the camera waits for a complete low-resolution frame of every view on the way), *Resolution* (it waits for a frame of the resolution in percent next to the drop-down, which has the presets *Quarter resolution*, *Half resolution* and *Full resolution*: below 100% a frame of blocks of 100 / percent pixels, 25% being blocks of 4 x 4; 100% a full-resolution frame, sharp, but it can crawl where the view is slow; above 100% supersampling layers, 200% being 4 samples per pixel) or *Super sampled* (every view on the way gets all its supersampling layers, and the camera waits for the finished frame: the best picture, the slowest crawl) or *Auto* (a frame rate is typed next to the drop-down, 60 FPS by default, and every frame gets the finest resolution that keeps it: any scale from full resolution down to 12.5% of the pixels across, steered by the time of the frames and scaled up smoothly on the screen, so a view that just misses the frame rate at full resolution gets 80 to 90% instead of half; the progress bar shows the resolution). A flight that waits for its frames is rendered frame after frame by one job: as soon as a frame is complete enough it is shown and the next view is started, without waiting for the next frame of the window, and never faster than the screen shows them. With *Auto blur during flights* (on by default), the image gets a light blur on the GPU while a flight moves it, which brings a moving frame closer to the finished image; its strength follows the resolution of the frame, and the finished image at the place is never blurred. With *Frames ahead* (2 by default) the flight job renders up to that many frames ahead of the one on the screen, so the CPU and the GPU never wait for the screen and every frame is ready at its refresh. While it flies, the progress bar shows the frame rate (the new views shown in the last second, updated once a second), and the status log gives the frames and the average frame rate of every flight.
- **Run benchmark**: a repeatable tour, like the benchmark mode of a game. It renders the reset view (a warm-up), then flies at the cinematic pace with *Low res* frames to the five deepest places that the zoom limit allows, shallow to deep, waiting for the full-quality render at each. Per place it reports the frames, the frame times (minimum, average, maximum, 95th percentile), the average and the 1% low frame rate, the flight time against the planned one, and the time to the finished render after arriving; the slowest place is highlighted. Every run is saved to `benchmarks/` (JSON with every frame time and the settings, and a CSV summary) and compared with the run before in the log. Your settings are restored afterwards; *Stop benchmark* or using the controls during a flight ends the run.
- **Smooth colors** (on by default): colors flow smoothly between iteration counts. Off, every pixel gets the color of its whole iteration count (bands of color), which saves two logarithms per pixel outside the set; saved images follow this setting.
- **Use the GPU** (on when the computer has a suitable GPU): the GPU helps with the slow parts of a render, next to the CPU threads. It computes bit for bit the same samples as the CPU (checked when the window opens, see the status log), so the image is the same; a GPU without 64-bit shaders, or one that does not pass the check, is not used.
- **Fast GPU math** (on by default): the GPU computes with pairs of 32-bit numbers (double-float arithmetic) instead of 64-bit numbers, which GPUs run much faster. Seahorse Valley at 1200 x 800 renders in 31 ms instead of 55 ms, and at 32x adaptive in 0.58 s instead of 1.08 s (RTX 5070 Ti with a Ryzen 7 7800X3D). It is not bit for bit the CPU's: at 1x about 99% of the pixels are the same, and the rest are noisy pixels in chaotic parts of the set, which any other rounding changes too. It is only used down to a scale of about 10⁻⁹ (deeper views use the exact loop), and saved images are always exact.
- **Limit the zoom to fast GPU math** (on by default, with fast GPU math; untick it to zoom deeper, where the GPU cannot help and renders get much slower): nothing deeper than a scale of about 10⁻⁹, where the fast GPU math ends, is rendered. Zooming in (button or wheel), *Go to view*, *Back*/*Forward* and flights to deeper places are refused, and the status log says why; anything else that would go deeper (a larger window, the limit switched on at a deep view) zooms out to the limit first. Switch it off to zoom as deep as 64-bit numbers allow.
- **Faster moving frames (f32)** (on by default, with fast GPU math): frames of flights and drags at shallow zoom (wider than a scale of about 0.01 at 4K) are computed by the GPU in plain 32-bit numbers, several times as fast as the fast GPU math, but 0.5% (the whole set) to 2.4% (Frost on the Shoreline) of the pixels get other colors while the view moves; the image the view stops at is always rendered with the fast or exact math.
- **Immersion**: the window never shows black where nothing is rendered yet. While you move the view, and during flights, the view only moves once the new view has at least a complete low-resolution frame. Off by default.
- **Go to view**: type the `x`, `y` and `scale` of a view; leave `scale` empty to keep the current zoom.
- **Max. iterations**: the most iterations of a point. **Auto iterations** chooses it for the zoom of every render (on by default); its drop-down offers the presets *Performance*, *Balanced* (the default) and *High Detail*, and the fields below it are the settings of the preset. It is a heuristic on a sparse grid, not a proof: very narrow detail can be missed.
- **Adaptive supersampling** (on by default): the first two samples of every pixel, the rest only where the picture shows detail (a heuristic; the same render can differ slightly from full supersampling).
- **Match the window size**: the image is rendered with one pixel for every pixel of the area it is shown in, and rendered again (after the window has stayed the same size for a moment) when the window is resized. The view keeps its center and its zoom: a wider window shows the same part of the real axis, a taller one shows more of the imaginary axis. It is on by default.
- **Render the neighborhood** (on by default): once the view is finished and nothing else runs, the area around it (by default one view to every side; the size next to the checkbox sets how far, in percent of the view, 10% to 5000%; at half the resolution, lower for a neighborhood wider than 8192 pixels) is rendered in the background and drawn behind the image, so dragging, panning and zooming out show it at once instead of black, until the render of the new view refines it. Any render, flight or gesture stops it at once.
- **Quality (supersampling)**: samples per pixel, from 1x to 64x (32x by default).
- **Coloring** (Bernstein polynomials or HSV) and **Color channels** (any of the 27 mappings such as `RGB` or `BGR`).
- **Iteration divide** and **Offset** (the color cycle, as in Kalles Fraktaler): the palette repeats every 500 steps, one step every *divide* iterations, starting *offset* steps in. Both are sliders: the divide is logarithmic, from 0.1 to 1000 (a saved value outside that range is moved to the nearest end), and the offset goes from 0 to 499; the colors change at once while you drag. A point keeps its color whatever the iteration limit and the zoom, so a manual limit and an automatic one give the same image, and zooming in does not recolor what you zoom into. Deep views, whose iteration counts are high and far apart, often look calmer with a larger divide. **Stretch the palette over the limit** instead spans the palette once from 0 to the limit (the colors then change with the limit). A saved place keeps its divide and offset (restored with *Restore the saved appearance*), and a flight to it blends them on the way. Changing the colors (divide, offset, stretch, the coloring) recolors the image at once: the renderer keeps the position of every pixel's first sample, so the first layer is recolored without computing it again, and the further layers follow.
- **Save Image** with its own *Saved image quality* and *Saved image size*: the default size (1200 x 800, selected at the start), the size of the view, one of ten presets (HD 720p to 8K UHD, squares, portrait), a preset of your own, or *Custom size...* (16 to 16,384 pixels per side, at most 100 megapixels). Give a custom size a name and *Save as preset* to keep it in `resolution_presets.json` in your configuration folder (`~/.config/mandelbrot` on Linux). The PNG goes to the `saved` folder, with the view, the settings and the application in its metadata.
- Click the fractal to see the coordinates of a point in the status; right-click to center the view on it. **Scroll** over it to zoom in and out around the cursor (a notch is about 10%; zooming in stops at the precision limit), and **drag** it with the left button to move the view.

**Current view**: the center (real and imaginary, with all digits) and the zoom (the magnification relative to the starting view). *Copy* puts the center and the scale on the clipboard.

**Status**: the messages, a *Cancel* button and a progress bar while a render runs, with the time of every supersampling layer. Any control you use interrupts the render that is running.

## Points of interest
The places in the *Points of interest* list of the window are the `.json` files of the [`points_of_interest`](points_of_interest) folder (the one in the current directory, or else the one next to the sources). Add your own file to the folder and press *Reload*; nothing in the code lists the files. The list is sorted by zoom, the shallowest place at the top and the deepest at the bottom (places with the same zoom in alphabetical order of their file names). The bundled files are `00-reset-view.json` (the reset view, the shallowest, so first), then `01-frost-on-the-shoreline.json` to `10-sunlit-reef.json`, numbered in the order of the views they replaced (`VIEW_0`, the tenth, is `10-sunlit-reef.json`). Every place in the list shows its zoom, so a deep (and slower) place is visible before you click it; the zoom is colored by how deep the place is compared with the others, from blue (the shallowest) through teal and amber to red (the deepest), and the markers of *Show points of interest* glow in the same colors. *Show points of interest* (off by default) marks every place on the view where it is, with a small glowing circle and its name. *Save this view...* saves the current view as a new place: give it a name (and, if you like, a description), and it is written as the next numbered file of the folder (with the iterations, the coloring and the color channels of the view) and listed at its zoom.
A file has a **location** (center, `scale`, `max_iterations`) and an optional **appearance** (`coloring`, `color_channels`). The `scale` is the scale of the renderer: the view is `2.5 * scale` wide, so a smaller scale is a deeper zoom. Going to a place always applies the location; its saved appearance is only applied when *Restore the saved appearance* is ticked, and the supersampling is never part of a place. The list shows a thumbnail of every place. Thumbnails are rendered on demand, only for the rows that are visible, one at a time and never while the main view renders (see `src/thumbnails.rs`); at most 64 are kept in memory. A file that cannot be used is skipped with a message in the status box. See the documentation of `src/points_of_interest.rs` for the format.

## How an image is made
- Every pixel is the average of its supersampling samples. A sample is taken at a pseudo-random offset inside the pixel, which is a hash of the pixel and the sample number, never random: the same settings always give the same image, whatever the number of threads.
- A point `c` is iterated with `z = z^2 + c` until `|z|` leaves the orbit radius 2 or the maximum iterations is reached. Points of the main cardioid and the period-two bulb, and orbits that repeat exactly, are recognized early as inside the set.
- The color comes from a smooth iteration count (the normalized iteration count, so neighboring iteration counts blend instead of forming bands) and a palette.
- The render is progressive, one supersampling layer at a time, on all CPU threads but one; the window shows each layer when it is ready.

See the documentation of the code (`cargo doc --open`) for the details, and [SOURCES.md](SOURCES.md) for the sources.

## Development
```
cargo test                                    # unit tests and the regression tests of tests/
cargo clippy --all-targets                    # the lints, as configured in src/lib.rs
cargo doc --open                              # the documentation of every module
cargo run --release --example perf            # timings of the renderer, with a checksum of every image
cargo run --release --example image_baseline  # a checksum of 115 representative renders (see below)
```
- `examples/image_baseline.rs` is for refactoring: run it before and after a change and compare the lines starting with `BASELINE`; the output must not change. The checksums are reproducible on one machine and build, not across different CPUs or math libraries.
- `tests/render_regression.rs` pins what can be pinned on every platform (iteration counts and renders with an integer palette) and compares the rest with a tolerance.
- `examples/perf.rs`, `adaptive.rs`, `auto_iterations.rs` and `thumbnails.rs` measure the renderer, adaptive supersampling, the automatic iteration limit and the thumbnails.

Where things are: `complex`, `complex_plane` (the part of the plane an image shows), `mandelbrot_set` (the iteration), `coloring`, `pixel_buffer`, `rendering` (the progressive renderer), `auto_iterations`, `thumbnails`, `points_of_interest`, `gui`, `config`.

## Wallpapers
### 3440x1440
![2023-06-26_12-48-59 189737_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/d487285e-d233-4374-bb5c-c46d84f0d83f)
![2023-06-26_12-35-03 632016400_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/ccad4f4a-1bc3-455b-98d9-53c5c0f85db3)
![2023-06-26_15-36-32 408135800_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/9baa2097-7cd7-4e53-b799-bfeab95c14dd)
![2023-06-26_15-25-34 763976800_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/49c02944-1d34-493c-a047-6478b4046052)
![2023-06-26_12-56-26 498671100_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/91813670-bcdd-480d-bb28-0fd68f8dad10)
![2023-06-26_08-54-45 849997300_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/0f28dfac-ed62-4096-8c9a-73370a2d69fb)

### 2560x1600
![2023-06-24 18:50:23 842186828 UTC](https://github.com/jortrr/mandelbrot/assets/38651009/03ce23c3-770e-444a-b163-aa247cfeda7c)
![2023-06-29_11-45-50 778868321_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/845cf9cf-3ac5-4df7-8150-abdd227af18a)
![2023-06-29_11-46-06 821888383_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/4f369655-c36d-42e3-8864-0ae3cf998854)
![2023-06-29_11-48-21 628194507_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/57c09c54-5616-487d-a7e8-66e69d8009db)
![2023-07-01_15-51-57 214248342_UTC](https://github.com/jortrr/mandelbrot/assets/38651009/5f66ad28-0bcd-4648-93c6-4f5915055538)

### 1170x2532 (Iphone 13)
![2023-06-24 19:29:53 436765840 UTC](https://github.com/jortrr/mandelbrot/assets/38651009/9b67dcdb-9dc3-4646-bbb4-96d7dc3ddb8f)

## Credits
- By Jort (jortrr).
- With the help of Claude Code (Anthropic): a large part of the code, the tests and the documentation since version 0.2.0 was written with it, as the commit messages say.
- Licensed under the GNU General Public License, version 3 or later: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

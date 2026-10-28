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
The window can be given a size and some settings; use `-` for the default of an argument:
```
cargo run --release -- <width> <height> <max_iterations> <supersampling_amount> <window_scale>
```
| Argument | Default | Meaning |
|---|---|---|
| `width`, `height` | 1200, 800 | the size of the image in pixels, and the size of a saved image. Without them (or `window_scale`) the image follows the size of the window, see *Match the window size* below |
| `max_iterations` | 1000 | the most iterations of a point |
| `supersampling_amount` | 32 | samples per pixel (1 to 64) |
| `window_scale` | 1.0 | the scale of the window relative to the image, e.g. `0.5` shows a 1920x1080 image in a 960x540 window |

For example `cargo run --release -- 1920 1080 2000 4 0.5`. The terminal is only for developers: everything you need is in the window.

## The window
The fractal is on the left. On the right, from top to bottom, are the controls, the current view and the status.

**Controls**
- **Move**: the arrow buttons move the view by the *Move step* in pixels; only the newly exposed strip is rendered.
- **Zoom in / Zoom out / Reset view**, and the *Zoom speed*. **Zoom in** is disabled when the 64-bit numbers of the renderer run out of precision (see the warning under *Current view*). **< Back / Forward >** walk through the views you visited (the last 200), like the history of a browser.
- **Points of interest**: a list of named places with thumbnails (see below); *Reload* reads the folder again; *Restore the saved appearance* also applies the coloring and the color channels of a place.
- **Flight**: clicking a place flies the view there with smooth panning and zooming (the path of van Wijk and Nuij, which zooms out to travel). The route is *Via reset* (out to the whole set first, the default) or *Shortest route*; the pace is *Cinematic* (the default), *Normal* or *Fast*. Using any control or the mouse stops a flight where it is; while it flies, a progress bar shows how much of the way is covered, *Cancel* stops it there and *Skip* goes to the place at once. The rendering of a flight is *Best effort* (the default: the camera follows the clock and the picture fills in), *Low res* (the camera waits for a complete low-resolution frame of every view on the way) or *Max quality* (it waits for a full-resolution frame: sharp, but it can crawl where the view is slow).
- **Immersion**: the window never shows black where nothing is rendered yet. While you move the view, and during flights, the view only moves once the new view has at least a complete low-resolution frame. Off by default.
- **Go to view**: type the `x`, `y` and `scale` of a view.
- **Max. iterations**: the most iterations of a point. **Auto iterations** chooses it for the zoom of every render (on by default); its drop-down offers the presets *Performance*, *Balanced* (the default) and *High Detail*, and the fields below it are the settings of the preset. It is a heuristic on a sparse grid, not a proof: very narrow detail can be missed.
- **Adaptive supersampling** (on by default): the first two samples of every pixel, the rest only where the picture shows detail (a heuristic; the same render can differ slightly from full supersampling).
- **Match the window size**: the image is rendered with one pixel for every pixel of the area it is shown in, and rendered again (after the window has stayed the same size for a moment) when the window is resized. The view keeps its center and its zoom: a wider window shows the same part of the real axis, a taller one shows more of the imaginary axis. It is on by default, and off when the command line gives a `width`, `height` or `window_scale`.
- **Quality (supersampling)**: samples per pixel, from 1x to 64x (32x by default).
- **Coloring** (Bernstein polynomials or HSV) and **Color channels** (any of the 27 mappings such as `RGB` or `BGR`).
- **Save Image** with its own *Saved image quality* and *Saved image size*: the command line size (the default), the size of the view, one of ten presets (HD 720p to 8K UHD, squares, portrait), a preset of your own, or *Custom size...* (16 to 16,384 pixels per side, at most 100 megapixels). Give a custom size a name and *Save as preset* to keep it in `resolution_presets.json` in your configuration folder (`~/.config/mandelbrot` on Linux). The PNG goes to the `saved` folder, with the view, the settings and the application in its metadata.
- Click the fractal to see the coordinates of a point in the status; right-click to center the view on it. **Scroll** over it to zoom in and out around the cursor (a notch is about 10%; zooming in stops at the precision limit), and **drag** it with the left button to move the view.

**Current view**: the center (real and imaginary, with all digits) and the zoom (the magnification relative to the starting view). *Copy* puts the center and the scale on the clipboard.

**Status**: the messages, a *Cancel* button and a progress bar while a render runs, with the time of every supersampling layer. Any control you use interrupts the render that is running.

## Points of interest
The places in the *Points of interest* list of the window are the `.json` files of the [`points_of_interest`](points_of_interest) folder (the one in the current directory, or else the one next to the sources). Add your own file to the folder and press *Reload*; nothing in the code lists the files. The list is in alphabetical order of the file names, so start a file name with a digit to choose its place: the bundled files are `0-frost-on-the-shoreline.json` to `9-sunlit-reef.json`, numbered from 0 in the order of the views they replaced (`VIEW_0`, the tenth, is `9-sunlit-reef.json`).
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

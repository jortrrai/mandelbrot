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

A Mandelbrot set viewer written in Rust, with a graphical window: explore the set, fly to named places, and save images and videos. The same view and settings always render exactly the same image.

- **Fast**: every CPU thread but one with the CPU's vector instructions, and the GPU next to them; a progressive render that shows every layer as soon as it is done.
- **Smooth**: smooth coloring, supersampling up to 64 samples per pixel (adaptive, where the picture has detail), and a color cycle as in Kalles Fraktaler.
- **Flights** to named places, with frame rates up to the screen's, and **videos** of them, rendered offline.
- **Automatic iterations** that follow the zoom, or your own limit.

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
There are no command-line arguments: everything is in the window. Videos need [`ffmpeg`](https://ffmpeg.org) on the path (without it you can still save their frames as PNG files).

## The window
The fractal is on the left; on the right are the controls, the current view and the status log. The controls are in sections that open and close with a click on their title. Hover over a control to see what it does.

- **Mouse**: drag the fractal to move, scroll to zoom around the cursor, click to see the coordinates of a point, right-click to center on it.
- **Settings** are remembered in `mandelbrot/settings.json` in your configuration folder (`~/.config` on Linux, `%APPDATA%` on Windows); *Reset settings* puts them back. The view, the history and the places are not settings.
- **Files**: saved images (PNG, with the view and the settings in their metadata) and videos go to `saved/`, benchmark runs to `benchmarks/`, your own image sizes to `resolution_presets.json` in the configuration folder.

## Points of interest
The places in the list are the `.json` files of the [`points_of_interest`](points_of_interest) folder (in the current directory, or else next to the sources): add your own and press *Reload*, or use *Save this view...*. The list is sorted by zoom. A file has a **location** (center, `scale`, `max_iterations`) and an optional **appearance** (coloring, color channels, iteration divide and offset); the view is `2.5 * scale` wide. A file that cannot be used is skipped with a message in the status log. See the documentation of `src/points_of_interest.rs` for the format.

## How an image is made
- Every pixel is the average of its samples, each at a pseudo-random offset inside the pixel that is a hash of the pixel and the sample number, never random: the same settings always give the same image, whatever the number of threads.
- A point `c` is iterated with `z = z^2 + c` until `|z|` leaves the orbit radius or the maximum iterations is reached; points certainly inside the set are recognized early.
- The color comes from a smooth iteration count and a palette that repeats along the iterations.

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

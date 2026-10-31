# Development

How the renderer works, and how to test and measure it. Back to the [README](../README.md).

## How an image is made
- Every pixel is the average of its samples. Each sample is at a pseudo-random offset inside the pixel, a hash of the pixel and the sample number, never random: the same settings always give the same image, whatever the number of threads.
- A point `c` is iterated with `z = z^2 + c` until `|z|` leaves the orbit radius or the maximum iterations is reached; points certainly inside the set are recognized early.
- The color comes from a smooth iteration count and a palette that repeats along the iterations.

See the documentation of the code (`cargo doc --open`) for the details, and [SOURCES.md](../SOURCES.md) for the sources.

## Commands
```
cargo test                                    # unit tests and the regression tests of tests/
cargo clippy --all-targets                    # the lints, as configured in src/lib.rs
cargo doc --open                              # the documentation of every module
cargo run --release --example perf            # timings of the renderer, with a checksum of every image
cargo run --release --example image_baseline  # a checksum of 115 representative renders (see below)
```

## Release builds
`.github/workflows/release.yml` builds the program for Linux, Windows and macOS. A pushed tag starting with `v` attaches the packages (the program, the places, the README and the license) to the GitHub release of that tag.

## Checksums and regression tests
- `examples/image_baseline.rs` is for refactoring: run it before and after a change and compare the lines starting with `BASELINE`; the output must not change. The checksums are reproducible on one machine and build, not across different CPUs or math libraries.
- `tests/render_regression.rs` pins what can be pinned on every platform (iteration counts and renders with an integer palette) and compares the rest with a tolerance.
- `examples/perf.rs`, `adaptive.rs`, `auto_iterations.rs` and `thumbnails.rs` measure the renderer, adaptive supersampling, the automatic iteration limit and the thumbnails.

## Where things are
The modules of `src/`: `complex`, `complex_plane` (the part of the plane an image shows), `mandelbrot_set` (the iteration), `coloring`, `pixel_buffer`, `rendering` (the progressive renderer), `auto_iterations`, `thumbnails`, `points_of_interest`, `image_metadata` (the view of a saved image, read back), `gui`, `config`.

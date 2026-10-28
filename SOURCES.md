# Sources

- [Complex number](https://en.wikipedia.org/wiki/Complex_number)
- [Complex plane](https://en.wikipedia.org/wiki/Complex_plane)
- [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Coloring the Mandelbrot set](https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/adding-some-colors)
- [True color(24-bit)](https://en.wikipedia.org/wiki/Color_depth#True_color_(24-bit))
- [Efficiently generating the Mandelbrot and Julia sets](https://theses.liacs.nl/pdf/2018-2019-JonckheereLSde.pdf)
- [Bernstein polynomial coloring](https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/)
- [Fractals & Rendering Techniques](https://darkeclipz.github.io/fractals/paper/Fractals%20&%20Rendering%20Techniques.html)
- [Supersampling](https://en.wikipedia.org/wiki/Supersampling)
- [How To View Image Metadata On Linux](https://ostechnix.com/how-to-view-image-metadata-on-linux/)

## Methods used in the code

- [Plotting algorithms for the Mandelbrot set](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set): the smooth (continuous) iteration count, the cardioid and period-two bulb test, and periodicity checking
- [Cycle detection: Brent's algorithm](https://en.wikipedia.org/wiki/Cycle_detection#Brent's_algorithm): finding orbits that repeat exactly
- [SplitMix64](https://prng.di.unimi.it/splitmix64.c): the hash behind the deterministic supersampling jitter
- [Advanced Vector Extensions (AVX2)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions): iterating several points at once
- [Rust `is_x86_feature_detected!`](https://doc.rust-lang.org/std/arch/macro.is_x86_feature_detected.html) and [`#[target_feature]`](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute): choosing the AVX2 loops at run time
- [egui](https://docs.rs/egui) and [eframe](https://docs.rs/eframe): the window and its controls
- [Text to ASCII Art Generator (TAAG)](https://patorjk.com/software/taag/): the banner
- [Smooth and efficient zooming and panning (van Wijk and Nuij, 2003)](https://www.win.tue.nl/~vanwijk/zoompan.pdf): the path of a flight to a point of interest

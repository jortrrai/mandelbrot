# Place files

The places in the window's list are the `.json` files of the [`points_of_interest`](../points_of_interest) folder: in the current directory, or else next to the sources. Back to the [README](../README.md).

```json
{
  "format_version": 1,
  "name": "Frost on the Shoreline",
  "description": "Optional text shown as a tooltip.",
  "location": {
    "center": { "real": -0.6604166666666667, "imaginary": 0.4437500000000001 },
    "scale": 0.1,
    "max_iterations": 1000
  },
  "appearance": { "coloring": "bernstein", "color_channels": "RGB", "iteration_divide": 4, "color_offset": 120 }
}
```

- **location** (required): the center, the `scale` and the maximum iterations. The view is `2.5 * scale` wide, so a smaller scale is a deeper zoom. Numbers are read exactly.
- **appearance** (optional): the coloring (`"hsv"` or `"bernstein"`), the color channels (such as `"RGB"`), the iteration divide (0.1 to 1000), the color offset (0 to 500) and `stretch_palette`. Going to a place applies it only with *Restore the saved appearance* on.
- The list is sorted by zoom, the shallowest first. A file that can't be used is skipped with a message in the status log.

The complete format is in the documentation of `src/points_of_interest.rs` (`cargo doc --open`).

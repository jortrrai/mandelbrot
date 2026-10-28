//! Tests of the window: rendering jobs, moves, the controls, the points of interest and their thumbnails, and the current view.
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use eframe::egui::{self, ColorImage, TextureOptions};

use super::*;
use crate::auto_iterations::AutoPreset;
use crate::coloring::{ColorChannelMapping, TrueColor};
use crate::complex::Complex;
use crate::complex_plane::View;
use crate::points_of_interest::Coloring;
use crate::thumbnails;

fn wait_for_job(app: &mut MandelbrotApp) {
    let start = std::time::Instant::now();
    while app.is_busy() {
        assert!(start.elapsed() < Duration::from_secs(60), "the background job did not finish");
        std::thread::sleep(Duration::from_millis(10));
        app.poll_job();
    }
}

/// Lets the flight to a point of interest that is under way arrive (as if its time has passed, or with Skip for a flight that waits for its frames) and waits for the render of the point
fn finish_flight(app: &mut MandelbrotApp) {
    if let Some(flight) = &app.flight {
        let arrival = flight.last_frame + flight.flight.duration();
        app.fly_at(arrival);
    }
    app.skip_flight();
    assert!(app.flight.is_none());
    wait_for_job(app);
}

/// An app that has rendered the default view with manual iterations and uniform supersampling, the two switched off (they are on by default)
fn manual_app(config: &Config) -> MandelbrotApp {
    let mut app = MandelbrotApp::new(config);
    wait_for_job(&mut app);
    app.auto_iterations = false;
    app.adaptive_supersampling = false;
    app.render("Rendered the default view");
    wait_for_job(&mut app);
    app
}

#[test]
fn rendering_runs_in_the_background_and_updates_the_buffer() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    assert!(app.is_busy(), "the initial render should run in the background");
    wait_for_job(&mut app);
    assert!(app.p.pixels.iter().any(|pixel| *pixel != 0), "the default view should not be all black");
    assert!(app.log.last().unwrap().starts_with("Rendered the default view"));

    let before = app.p.pixels.clone();
    app.c.scale(0.5);
    app.render("Zoomed in");
    assert!(app.is_busy());
    wait_for_job(&mut app);
    assert_ne!(before, app.p.pixels, "zooming should change the image");
}

#[test]
fn a_new_action_interrupts_a_progressive_render_and_starts_over() {
    let config = Config::sized(64, 48, 50, 8);
    let mut app = MandelbrotApp::new(&config);
    assert!(app.progress.is_some());
    let first_cancel = Arc::clone(&app.progress.as_ref().unwrap().cancel);

    //Moving while the first render is still running interrupts it, and renders the moved view from scratch
    app.translate("up", 10, 0);
    assert!(first_cancel.load(Ordering::Relaxed), "the first render should have been cancelled");
    assert!(app.progress.is_some(), "a new progressive render should be running");
    let moved_center = app.c.center();
    wait_for_job(&mut app);

    assert!(app.progress.is_none());
    assert_eq!(app.p.pixels.len(), 64 * 48);
    assert!(app.p.pixels.iter().any(|pixel| *pixel != 0));
    assert_eq!(app.c.center().y, moved_center.y, "the moved view should be kept");
    assert!(!app.log.iter().any(|message| message.starts_with("Cancelled")), "interrupting is not reported as a cancelled render");
    assert!(app.log.last().unwrap().starts_with("Moved up 10 pixels"));
}

#[test]
fn the_progress_bar_records_the_time_of_every_layer() {
    let config = Config::sized(64, 48, 50, 3);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);

    let bars = app.bars.as_ref().expect("a full render should have a progress bar");
    assert_eq!(bars.total_layers, 3);
    assert_eq!(bars.layer_times.len(), 3, "every layer should have a time");
    let (total, completed) = bars.finished.expect("the render should be finished");
    assert!(completed);
    assert!(total >= bars.layer_times.iter().sum(), "the total includes all layers");
    assert!(app.log.last().unwrap().contains(" s: "), "the log should mention the render time: {:?}", app.log.last());

    //Moving renders the exposed strip progressively too, with its own bar
    app.translate("up", 10, 0);
    assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 0, "the move starts a new bar");
    wait_for_job(&mut app);
    assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 3, "every layer of the move is rendered and timed");
    assert!(app.log.last().unwrap().starts_with("Moved up 10 pixels (continued the samples of "), "{:?}", app.log.last());
}

#[test]
fn moving_reuses_the_old_image_and_only_renders_the_exposed_strip() {
    let config = Config::sized(64, 48, 50, 3);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    let old_pixels = app.p.pixels.clone();

    app.translate("up", 10, 0); //The content of the buffer moves down by 10 rows
    wait_for_job(&mut app);
    assert!(app.log.last().unwrap().starts_with("Moved up 10 pixels (continued the samples of 2432 pixels) in "), "only the exposed top strip is rendered: {:?}", app.log.last());

    assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 3, "every layer of the strip is rendered progressively");
    for y in 10..48 {
        assert_eq!(app.p.pixels[y * 64..(y + 1) * 64], old_pixels[(y - 10) * 64..(y - 9) * 64], "row {y} should be the shifted old image");
    }
    assert!(app.buffer_is_complete);
}

#[test]
fn moving_after_a_cancelled_render_continues_its_samples() {
    let config = Config::sized(64, 48, 50, 6);
    let mut app = manual_app(&config);
    app.render("Again");
    app.progress.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
    wait_for_job(&mut app);
    assert!(!app.buffer_is_complete, "a cancelled render leaves a partially refined buffer");

    app.translate("left", 0, -10);
    wait_for_job(&mut app);
    assert!(app.buffer_is_complete);
    assert!(app.log.last().unwrap().starts_with("Moved left 10 pixels (continued the samples of 2592 pixels) in "), "54 columns kept their samples: {:?}", app.log.last());
}

/// The image of `view` (the start view if `None`) after a complete render, then a move `rows_up` and `columns_right`
fn moved_after_a_complete_render(config: &Config, view: Option<&View>, rows_up: i16, columns_right: i16) -> Vec<u32> {
    let mut app = manual_app(config);
    if let Some(view) = view {
        app.c.set_view(view);
        app.render("The view");
        wait_for_job(&mut app);
    }
    app.translate("somewhere", rows_up, columns_right);
    wait_for_job(&mut app);
    app.p.pixels.clone()
}

#[test]
fn an_interrupted_render_keeps_its_completed_layers_when_the_view_moves() {
    //A finished render of 2 layers leaves exactly the samples of a render of 8 layers that was interrupted after 2 (the renderer tests also interrupt in the middle
    //of a layer); this way the test does not depend on catching a running render, which is slow on some machines and too fast on others
    let config = Config::sized(96, 64, 300, 8);
    let view = View::new(-0.7498100000000001, -0.020300000000000054, 0.004);
    let mut app = manual_app(&config);
    app.c.set_view(&view);
    app.supersampling_amount = 2;
    app.render("Two layers");
    wait_for_job(&mut app);

    //Back to 8 layers and a move: the pixels that stay visible continue from their 2 samples
    app.supersampling_amount = 8;
    app.translate("up", 10, 0);
    wait_for_job(&mut app);
    let message = app.log.last().unwrap().clone();
    assert!(message.starts_with(&format!("Moved up 10 pixels (continued the samples of {} pixels) in ", 96 * (64 - 10))), "{message}");
    //And the result is exactly the image of a render that was not interrupted, moved the same way
    assert_eq!(app.p.pixels, moved_after_a_complete_render(&config, Some(&view), 10, 0));
}

#[test]
fn centering_on_a_click_moves_by_whole_pixels_and_keeps_the_samples() {
    let config = Config::sized(64, 48, 50, 3);
    let mut app = manual_app(&config);
    let target = app.c.complex_from_pixel_plane(50.3, 10.6);
    app.handle_image_click(50.3, 10.6, true);
    wait_for_job(&mut app);
    //18 columns to the right, 13 rows up: the clicked point is within half a pixel of the center
    let center = app.c.center();
    assert!((center.x - target.x).abs() <= app.c.increment_x / 2.0 && (center.y - target.y).abs() <= app.c.increment_y / 2.0, "{center:?} {target:?}");
    assert!(app.log.last().unwrap().contains(&format!("(continued the samples of {} pixels)", (64 - 18) * (48 - 13))), "{:?}", app.log.last());
}

#[test]
fn texture_bytes_equal_the_converted_pixels_for_all_mappings() {
    let colors: Vec<TrueColor> = (0..=255u8).map(|i| TrueColor::new(i, 255 - i, i / 3)).collect();
    for name in COLOR_CHANNEL_MAPPINGS {
        let mapping = ColorChannelMapping::new(name).unwrap();
        let pixels = PixelBuffer::colors_to_pixels(&colors, &mapping);
        let expected: Vec<u8> = pixels.iter().flat_map(|pixel| [(pixel >> 16) as u8, (pixel >> 8) as u8, *pixel as u8]).collect();
        assert_eq!(colors_to_rgb_bytes(&colors, mapping), expected, "{name}");
    }
}

#[test]
fn auto_iterations_switch_with_manual_and_keep_the_palette_stable_on_moves() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = manual_app(&config);
    assert_eq!(app.m.max_iterations, 50);
    assert_eq!(app.m.palette_scale, None, "manual mode colors exactly as before");
    let manual_pixels = app.p.pixels.clone();

    //Turn auto on: the limit is chosen for the zoom, and shown in the max. iterations field
    app.auto_iterations = true;
    app.render("Turned auto iterations on");
    wait_for_job(&mut app);
    let limit = app.m.max_iterations;
    assert_eq!(app.m.palette_scale, Some(app.auto_settings.initial_limit(app.c.get_scale())));
    assert!(app.log.last().unwrap().contains("auto iterations: limit"), "{:?}", app.log.last());
    assert_eq!(app.max_iterations_input, 50, "the manual value is kept while auto is on");

    //A move keeps the limit and only renders the strip
    app.translate("up", 10, 0);
    wait_for_job(&mut app);
    assert_eq!(app.m.max_iterations, limit);
    assert!(app.log.last().unwrap().contains("(continued the samples of 2432 pixels)"), "{:?}", app.log.last());

    //A zoom chooses a new limit, a deeper zoom never a lower start
    app.c.scale(1e-4);
    app.render("Zoomed in");
    wait_for_job(&mut app);
    assert!(app.m.max_iterations >= app.auto_settings.initial_limit(app.c.get_scale()));
    assert!(app.m.max_iterations > limit.min(300) || limit >= 300);

    //Turn auto off: the manual value is used again and the image is the manual image
    app.auto_iterations = false;
    app.c.reset();
    app.render("Turned auto iterations off");
    wait_for_job(&mut app);
    assert_eq!(app.m.max_iterations, 50);
    assert_eq!(app.m.palette_scale, None);
    assert_eq!(app.p.pixels, manual_pixels, "back in manual mode the same view renders the same image");
}

#[test]
fn moving_after_turning_auto_off_does_not_reuse_an_auto_colored_buffer() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    app.auto_iterations = true;
    app.render("auto");
    wait_for_job(&mut app);
    app.auto_iterations = false;
    app.translate("up", 10, 0);
    wait_for_job(&mut app);
    assert_eq!(app.m.palette_scale, None);
    assert_eq!(app.bars.as_ref().unwrap().pixels_total, 64 * 48, "the buffer was colored with the auto palette and is rendered again");
}

#[test]
fn presets_and_adaptive_supersampling_toggle_independently_of_auto_iterations() {
    let config = Config::sized(64, 48, 50, 4);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    assert_eq!(app.auto_preset, AutoPreset::Balanced, "Balanced is the default");
    assert!(app.auto_iterations && app.adaptive_supersampling, "both are on by default");
    app.auto_iterations = false;
    app.adaptive_supersampling = false;
    app.render("manual and uniform");
    wait_for_job(&mut app);
    let uniform = app.p.pixels.clone();

    //Adaptive supersampling works in manual mode, and switching it off gives the uniform image again
    app.adaptive_supersampling = true;
    app.render("adaptive on");
    wait_for_job(&mut app);
    assert_eq!(app.m.palette_scale, None);
    assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), 4, "every layer is still reported");
    app.adaptive_supersampling = false;
    app.render("adaptive off");
    wait_for_job(&mut app);
    assert_eq!(app.p.pixels, uniform);

    //A preset changes the settings used by auto iterations
    app.auto_iterations = true;
    app.adaptive_supersampling = true;
    app.auto_preset = AutoPreset::HighDetail;
    app.auto_settings = AutoPreset::HighDetail.settings();
    app.render("high detail");
    wait_for_job(&mut app);
    let high = app.m.max_iterations;
    app.auto_preset = AutoPreset::Performance;
    app.auto_settings = AutoPreset::Performance.settings();
    app.render("performance");
    wait_for_job(&mut app);
    assert!(app.m.max_iterations <= high);
    assert_eq!(app.m.palette_scale, Some(AutoPreset::Performance.settings().initial_limit(app.c.get_scale())));
}

#[test]
fn selecting_a_preset_shows_its_limit_immediately_and_stale_renders_cannot_overwrite_it() {
    let config = Config::sized(64, 48, 50, 2);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    app.c.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.000064));
    app.auto_iterations = true;
    app.render("auto");
    wait_for_job(&mut app);
    let balanced = app.m.max_iterations;

    //Selecting: the starting limit of the preset is shown before any poll, while the render is only starting
    app.select_preset(AutoPreset::HighDetail);
    let initial = AutoPreset::HighDetail.settings().initial_limit(app.c.get_scale());
    assert_eq!((app.m.max_iterations, app.m.palette_scale), (initial, Some(initial)), "published synchronously on selection");
    assert!(app.limit_pending && app.is_busy());
    assert_ne!(app.m.max_iterations, balanced);

    //The refined limit is published before the render has finished
    let start = std::time::Instant::now();
    while app.limit_pending {
        assert!(start.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(5));
        app.poll_job();
    }
    assert!(app.m.max_iterations >= initial);
    let high = app.m.max_iterations;
    wait_for_job(&mut app);
    assert_eq!(app.m.max_iterations, high);

    //Selecting twice quickly: the first render is interrupted and cannot change what is shown
    app.select_preset(AutoPreset::Performance);
    let first_cancel = Arc::clone(&app.progress.as_ref().unwrap().cancel);
    app.select_preset(AutoPreset::Balanced);
    assert!(first_cancel.load(Ordering::Relaxed));
    let balanced_initial = AutoPreset::Balanced.settings().initial_limit(app.c.get_scale());
    assert_eq!(app.m.max_iterations, balanced_initial);
    wait_for_job(&mut app);
    assert_eq!(app.m.palette_scale, Some(balanced_initial));
    assert_eq!(app.m.max_iterations, balanced, "the same view and preset give the same limit as before, not the Performance one");
    assert!(!app.limit_pending);
}

/// The texts that `controls_ui` draws, in drawing order
fn controls_texts(app: &mut MandelbrotApp) -> Vec<String> {
    let ctx = egui::Context::default();
    let output = ctx.run_ui(egui::RawInput::default(), |ui| app.controls_ui(ui));
    let texts = output.shapes.iter().filter_map(|shape| match &shape.shape {
        egui::epaint::Shape::Text(text) => Some(text.galley.text().to_string()),
        _ => None,
    }).collect();
    output.drop_without_applying_deltas();
    texts
}

#[test]
fn saving_has_its_own_section_and_the_rendering_quality_stays_with_the_rendering_controls() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    let texts = controls_texts(&mut app);
    let position = |needle: &str| texts.iter().position(|t| t.starts_with(needle)).unwrap_or_else(|| panic!("{needle:?} is not drawn: {texts:?}"));
    let (quality, coloring, channels) = (position("Quality (supersampling"), position("Coloring:"), position("Color channels"));
    let (heading, saved_quality, save) = (position("Save Image"), position("Saved image quality:"), position("Save image"));
    assert!(quality < coloring && coloring < channels, "the rendering quality is with the rendering controls");
    assert!(channels < heading && heading < saved_quality && saved_quality < save, "the saved image quality and the Save button are together in the Save Image section, after the rendering controls");
    assert_eq!(texts.iter().filter(|t| t.starts_with("Saved image quality")).count(), 1);
}

fn app_with_points(files: &[(&str, &str)]) -> (MandelbrotApp, std::path::PathBuf) {
    static NEXT_FOLDER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let config = Config::sized(64, 48, 50, 3);
    let mut app = manual_app(&config);
    let folder = std::env::temp_dir().join(format!("mandelbrot-gui-poi-{}-{}", std::process::id(), NEXT_FOLDER.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&folder);
    std::fs::create_dir_all(&folder).unwrap();
    for (name, text) in files {
        std::fs::write(folder.join(name), text).unwrap();
    }
    app.load_points_of_interest(folder.clone());
    (app, folder)
}

const HSV_BGR_PLACE: &str = r#"{ "format_version": 1, "name": "Test Place", "location": { "center": { "real": -0.7498100000000001, "imaginary": -0.020300000000000054 }, "scale": 0.00006400000000000002, "max_iterations": 321 },
    "appearance": { "coloring": "hsv", "color_channels": "BGR" } }"#;

#[test]
fn the_application_lists_the_points_of_interest_of_the_folder_at_startup() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    assert_eq!(app.points.len(), 11, "the reset view and the ten former views are found without a list of file names in the code");
    assert_eq!((app.points[0].name.as_str(), app.points[1].name.as_str(), app.points[10].name.as_str()), ("Reset View", "Frost on the Shoreline", "Sunlit Reef"), "the reset view first, the deepest place (Sunlit Reef) last");
    assert!(app.points.windows(2).all(|pair| pair[0].location.scale >= pair[1].location.scale), "listed by zoom, the shallowest first");
    assert!(app.log.iter().any(|message| message.starts_with("Points of interest: 11 loaded from")));
    assert!(!app.restore_appearance);
}

#[test]
fn going_to_a_point_applies_the_location_and_keeps_the_current_appearance_and_supersampling() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    let (coloring_index, mapping, supersampling) = (app.coloring_index, app.p.color_channel_mapping, app.supersampling_amount);
    app.go_to_point_of_interest(0);
    finish_flight(&mut app);
    let center = app.c.center();
    assert!((center.x - -0.7498100000000001).abs() < 1e-12 && (center.y - -0.020300000000000054).abs() < 1e-12, "{center:?}");
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18);
    assert_eq!((app.m.max_iterations, app.max_iterations_input), (321, 321));
    assert_eq!((app.coloring_index, app.p.color_channel_mapping, app.supersampling_amount), (coloring_index, mapping, supersampling), "the appearance is only restored on request");
    assert!(app.log.last().unwrap().starts_with("Moved to Test Place in"), "{:?}", app.log.last());
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn restoring_the_appearance_applies_the_saved_coloring_and_channels_but_never_the_supersampling() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    let supersampling = app.supersampling_amount;
    app.restore_appearance = true;
    app.go_to_point_of_interest(0);
    finish_flight(&mut app);
    assert_eq!(app.coloring_index, 0, "HSV");
    assert_eq!(format!("{:?}", app.p.color_channel_mapping), "BGR");
    assert_eq!(app.supersampling_amount, supersampling);
    assert!(app.log.last().unwrap().contains("restored its appearance"), "{:?}", app.log.last());
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_malformed_file_is_reported_in_the_status_box_and_the_other_places_still_load() {
    let (app, folder) = app_with_points(&[("good.json", HSV_BGR_PLACE), ("bad.json", "{ this is not json"), ("zero.json", &HSV_BGR_PLACE.replace("0.00006400000000000002", "0"))]);
    assert_eq!(app.points.len(), 1);
    assert!(app.log.iter().any(|message| message.starts_with("Points of interest: bad.json")), "{:?}", app.log);
    assert!(app.log.iter().any(|message| message.starts_with("Points of interest: zero.json: the scale should be")));
    assert!(app.log.iter().any(|message| message.starts_with("Points of interest: 1 loaded from")));
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn reloading_finds_files_that_were_added_and_a_missing_folder_is_reported_not_fatal() {
    let (mut app, folder) = app_with_points(&[("good.json", HSV_BGR_PLACE)]);
    std::fs::write(folder.join("mine.json"), HSV_BGR_PLACE.replace("Test Place", "My Place")).unwrap();
    app.load_points_of_interest(folder.clone());
    assert_eq!(app.points.len(), 2, "a user's own file appears after reloading");
    let _ = std::fs::remove_dir_all(&folder);
    app.load_points_of_interest(folder);
    assert!(app.points.is_empty() && app.log.iter().any(|message| message.contains("Cannot read the points of interest folder")));
}

/// The texts that the side panel draws, with their vertical position, in drawing order
fn side_panel_positions(app: &mut MandelbrotApp) -> Vec<(String, f32)> {
    let ctx = egui::Context::default();
    let output = ctx.run_ui(egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 1000.0))), ..Default::default() }, |ui| app.side_panel_ui(ui));
    let texts = output.shapes.iter().filter_map(|shape| match &shape.shape {
        egui::epaint::Shape::Text(text) => Some((text.galley.text().to_string(), text.pos.y)),
        _ => None,
    }).collect();
    output.drop_without_applying_deltas();
    texts
}

fn side_panel_texts(app: &mut MandelbrotApp) -> Vec<String> {
    side_panel_positions(app).into_iter().map(|(text, _)| text).collect()
}

#[test]
fn the_current_view_texts_keep_all_digits_and_define_the_zoom() {
    let view = CurrentView::new(&Complex::new(-0.7498100000000001, -0.020300000000000054), 0.00006400000000000002);
    assert_eq!(view.real, "-0.7498100000000001");
    assert_eq!(view.imaginary, "-0.020300000000000054");
    assert_eq!(view.real.parse::<f64>().unwrap().to_bits(), (-0.7498100000000001_f64).to_bits(), "the text reads back as the same number");
    assert_eq!(view.imaginary.parse::<f64>().unwrap().to_bits(), (-0.020300000000000054_f64).to_bits());
    assert_eq!(view.zoom, "\u{d7}15,625");
    assert_eq!(CurrentView::new(&Complex::new(0.0, 0.0), 1.0).zoom, "\u{d7}1", "the starting view is x1");
    assert_eq!(CurrentView::new(&Complex::new(0.0, 0.0), 0.1).zoom, "\u{d7}10");
    assert_eq!(CurrentView::new(&Complex::new(0.0, 0.0), 2.0).zoom, "\u{d7}0.5", "zoomed out");
    assert_eq!(CurrentView::new(&Complex::new(0.0, 0.0), 4e-6).zoom, "\u{d7}250,000");
}

#[test]
fn the_zoom_is_an_ordinary_number_in_the_readable_range_and_scientific_only_beyond_it() {
    let cases = [
        (1.0, "\u{d7}1"), (2.0, "\u{d7}2"), (2.5, "\u{d7}2.5"), (3.3333333, "\u{d7}3.33"), (10.0, "\u{d7}10"), (156.25, "\u{d7}156.25"), (999.0, "\u{d7}999"),
        (999.99, "\u{d7}999.99"), (999.996, "\u{d7}1,000"), (1000.0, "\u{d7}1,000"), (15625.000000000002, "\u{d7}15,625"), (250000.0, "\u{d7}250,000"), (123456789.4, "\u{d7}123,456,789"),
        (999999999.0, "\u{d7}999,999,999"), (1e9, "\u{d7}1.00e9"), (1.5e12, "\u{d7}1.50e12"),
        (0.5, "\u{d7}0.5"), (0.25, "\u{d7}0.25"), (0.0625, "\u{d7}0.0625"), (0.001, "\u{d7}0.001"), (0.0009, "\u{d7}9.00e-4"), (1e-4, "\u{d7}1.00e-4"),
    ];
    for (magnification, expected) in cases {
        assert_eq!(format_zoom(magnification), expected, "{magnification}");
        assert!(!format_zoom(magnification).contains("2^"), "no power of two");
    }
    assert_eq!(format_zoom(f64::INFINITY), "\u{d7}?");
    assert_eq!(format_zoom(0.0), "\u{d7}?");
}

#[test]
fn the_current_view_section_is_above_the_status_and_follows_the_navigation() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    let positions = side_panel_positions(&mut app);
    let height = |needle: &str| positions.iter().find(|(text, _)| text.starts_with(needle)).unwrap_or_else(|| panic!("{needle:?} is not drawn: {positions:?}")).1;
    let (controls, current, status) = (height("Controls"), height("Current view"), height("Status"));
    assert!(controls < current && current < status, "the current view is between the controls and the status: {controls} {current} {status}");
    let texts = side_panel_texts(&mut app);
    let before = app.c.center();
    assert!(texts.iter().any(|t| *t == format!("{}", before.x)) && texts.iter().any(|t| *t == format!("{}", before.y)), "{texts:?}");
    assert!(texts.iter().any(|t| t == "\u{d7}1"), "{texts:?}");

    //A move, a zoom and a jump change the texts at once, before any render has finished
    app.translate("up", 10, 0);
    let moved = app.c.center();
    assert_ne!(moved.y, before.y);
    assert!(side_panel_texts(&mut app).iter().any(|t| *t == format!("{}", moved.y)));
    wait_for_job(&mut app);
    app.c.scale(0.1);
    app.render("Zoomed in");
    assert!(side_panel_texts(&mut app).iter().any(|t| t == "\u{d7}10"));
    wait_for_job(&mut app);
    app.c.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.00006400000000000002));
    app.render("Moved to a view");
    let texts = side_panel_texts(&mut app);
    assert!(texts.iter().any(|t| t == "\u{d7}15,625") && texts.iter().any(|t| t.starts_with("-0.74981000000000")), "{texts:?}");
    wait_for_job(&mut app);
}

/// One frame of the controls (with the lists of points of interest) in a window of 600 x `height` points; finished thumbnails are taken first, like `eframe::App::ui` does
fn controls_frame(app: &mut MandelbrotApp, ctx: &egui::Context, height: f32) {
    app.poll_thumbnails(ctx);
    let output = ctx.run_ui(egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, height))), ..Default::default() }, |ui| app.controls_ui(ui));
    output.drop_without_applying_deltas();
}

/// Frames until every visible thumbnail is cached (and no job runs), with a time limit
fn frames_until_thumbnails_are_done(app: &mut MandelbrotApp, ctx: &egui::Context, height: f32) {
    let start = std::time::Instant::now();
    loop {
        controls_frame(app, ctx, height);
        if app.thumb_job.is_none() && !app.thumbs_visible.is_empty() && app.thumbs_visible.iter().all(|key| app.thumbs.contains(*key)) {
            return;
        }
        assert!(start.elapsed() < Duration::from_secs(60), "the thumbnails were not finished");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `count` places; with `slow` they are boundary views with an enormous iteration limit (capped by the thumbnails), which take a noticeable time to render
fn many_points(count: usize, slow: bool) -> Vec<(String, String)> {
    (0..count).map(|i| {
        let text = HSV_BGR_PLACE.replace("Test Place", &format!("Place {i:03}"));
        let text = if slow { text.replace("\"max_iterations\": 321", "\"max_iterations\": 1000000") } else { text.replace("0.00006400000000000002", &format!("0.{:04}", 2 + i)).replace("\"max_iterations\": 321", "\"max_iterations\": 120") };
        (format!("place-{i:03}.json"), text)
    }).collect()
}

fn app_with_many_points(count: usize) -> (MandelbrotApp, std::path::PathBuf) {
    app_with_places(&many_points(count, false))
}

fn app_with_places(files: &[(String, String)]) -> (MandelbrotApp, std::path::PathBuf) {
    let borrowed: Vec<(&str, &str)> = files.iter().map(|(name, text)| (name.as_str(), text.as_str())).collect();
    app_with_points(&borrowed)
}

#[test]
fn only_the_visible_thumbnails_are_rendered_however_many_points_there_are() {
    let (mut app, folder) = app_with_many_points(25);
    assert_eq!(app.points.len(), 25);
    let ctx = egui::Context::default();
    frames_until_thumbnails_are_done(&mut app, &ctx, 1000.0);
    let visible = app.thumbs_visible.len();
    assert!((1..=5).contains(&visible), "a list of three rows shows a few of the 25 points: {visible}");
    assert_eq!(app.thumbnails_started, visible as u64, "exactly the visible thumbnails were rendered, each once");
    assert_eq!(app.thumbs.len(), visible);
    //More frames do not render anything more
    for _ in 0..5 {
        controls_frame(&mut app, &ctx, 1000.0);
    }
    assert_eq!(app.thumbnails_started, visible as u64);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn nothing_is_rendered_while_the_list_is_scrolled_or_clipped_out_of_view() {
    let (mut app, folder) = app_with_many_points(25);
    let ctx = egui::Context::default();
    for _ in 0..10 {
        controls_frame(&mut app, &ctx, 100.0); //The list is below the bottom of this window
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(app.thumbs_visible, Vec::<u64>::new());
    assert_eq!((app.thumbnails_started, app.thumbs.len()), (0, 0));
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_running_thumbnail_is_cancelled_when_it_leaves_the_view_and_when_the_main_view_renders() {
    let (mut app, folder) = app_with_places(&many_points(25, true));
    let ctx = egui::Context::default();
    controls_frame(&mut app, &ctx, 1000.0);
    let job = app.thumb_job.as_ref().expect("a visible thumbnail is rendered when the window is idle");
    let first_cancel = Arc::clone(&job.cancel);
    //The list goes out of view: the thumbnail is cancelled
    controls_frame(&mut app, &ctx, 100.0);
    assert!(first_cancel.load(Ordering::Relaxed) && app.thumb_job.is_none());

    //A thumbnail runs, then the main view renders: the thumbnail is cancelled and no other one starts while the main view renders
    std::thread::sleep(Duration::from_millis(100));
    controls_frame(&mut app, &ctx, 1000.0);
    let started = app.thumbnails_started;
    let second_cancel = Arc::clone(app.thumb_job.as_ref().map(|job| &job.cancel).expect("a thumbnail job runs again"));
    app.render("Rendering the main view");
    assert!(second_cancel.load(Ordering::Relaxed) && app.thumb_job.is_none(), "the main view has priority");
    for _ in 0..10 {
        controls_frame(&mut app, &ctx, 1000.0);
        if !app.is_busy() {
            break;
        }
        assert_eq!(app.thumbnails_started, started, "no thumbnail starts while the main view renders");
        assert!(app.thumb_job.is_none());
    }
    //When the main view is done, the thumbnails continue
    wait_for_job(&mut app);
    frames_until_thumbnails_are_done(&mut app, &ctx, 1000.0);
    assert!(app.thumbnails_started > started);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn the_resident_thumbnails_are_bounded_and_follow_the_appearance_that_going_to_a_place_would_use() {
    let (mut app, folder) = app_with_many_points(3);
    let ctx = egui::Context::default();
    for key in 0..300u64 {
        app.thumbs.insert(key, ctx.load_texture(format!("t{key}"), ColorImage::from_rgb([2, 2], &[0u8; 12]), TextureOptions::LINEAR));
        assert!(app.thumbs.len() <= thumbnails::MAX_CACHED);
    }
    //The thumbnail looks like the place does when going to it: with the current appearance, or the saved one when appearances are restored
    let point = app.points[0].clone();
    let current = MandelbrotApp::thumbnail_request_for(&point, app.coloring_index, false, app.p.color_channel_mapping, (64, 48));
    let restored = MandelbrotApp::thumbnail_request_for(&point, app.coloring_index, true, app.p.color_channel_mapping, (64, 48));
    assert_eq!(current.coloring, Coloring::Bernstein, "the current appearance of the window");
    assert_eq!(restored.coloring, Coloring::Hsv, "the saved appearance of the file");
    assert_ne!(current.key(), restored.key(), "another appearance is another thumbnail");
    assert_eq!(current.location, restored.location);
    assert_eq!(current.height, thumbnails::thumbnail_height(64, 48));
    let _ = std::fs::remove_dir_all(folder);
}

/// Clicks at `position` in the side panel (move, press and release in three frames, like a mouse) and returns the clipboard texts that the window asked for
fn click_side_panel(app: &mut MandelbrotApp, ctx: &egui::Context, position: egui::Pos2) -> Vec<String> {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 1000.0));
    let mut copied = Vec::new();
    for events in [
        vec![egui::Event::PointerMoved(position)],
        vec![egui::Event::PointerButton { pos: position, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }],
        vec![egui::Event::PointerButton { pos: position, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE }],
    ] {
        let output = ctx.run_ui(egui::RawInput { screen_rect: Some(screen), events, ..Default::default() }, |ui| app.side_panel_ui(ui));
        copied.extend(output.platform_output.commands.iter().filter_map(|command| match command {
            egui::OutputCommand::CopyText(text) => Some(text.clone()),
            _ => None,
        }));
        output.drop_without_applying_deltas();
    }
    copied
}

#[test]
fn the_copy_button_asks_for_the_center_and_scale_with_all_digits_to_be_copied() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    app.c.set_view(&View::new(-0.7498100000000001, -0.020300000000000054, 0.00006400000000000002));
    let ctx = egui::Context::default();
    let copy = side_panel_positions(&mut app).into_iter().find(|(text, _)| text == "Copy").expect("the Copy button is drawn");
    //Clicking elsewhere copies nothing; clicking the button copies the description of the view
    assert_eq!(click_side_panel(&mut app, &ctx, egui::pos2(5.0, 5.0)), Vec::<String>::new());
    let x = copy_x(&mut app);
    let copied = click_side_panel(&mut app, &ctx, egui::pos2(x, copy.1 + 4.0));
    assert_eq!(copied.len(), 1, "one click copies once: {copied:?}");
    let center = app.c.center();
    assert_eq!(copied[0], format!("center = ({}, {}), scale = {}", center.x, center.y, app.c.get_scale()));
    assert!(copied[0].contains("scale = 0.00006400000000000002"), "{}", copied[0]);
}

/// The horizontal position of the middle of the Copy button: where the text of the button is drawn
fn copy_x(app: &mut MandelbrotApp) -> f32 {
    let ctx = egui::Context::default();
    let output = ctx.run_ui(egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 1000.0))), ..Default::default() }, |ui| app.side_panel_ui(ui));
    let x = output.shapes.iter().find_map(|shape| match &shape.shape {
        egui::epaint::Shape::Text(text) if text.galley.text() == "Copy" => Some(text.pos.x + text.galley.size().x / 2.0),
        _ => None,
    });
    output.drop_without_applying_deltas();
    x.expect("the Copy button is drawn")
}

/// Saves the image and returns the width and height in the header of the png that was written (the file is removed again)
fn save_and_read_size(app: &mut MandelbrotApp) -> (u32, u32) {
    let before = app.log.len();
    app.save_image();
    let start = std::time::Instant::now();
    let saved_to = loop {
        assert!(start.elapsed() < Duration::from_secs(60), "the image was not saved: {:?}", &app.log[before..]);
        app.poll_job();
        if let Some(message) = app.log[before..].iter().find(|message| message.starts_with("Saved the image to ") || message.starts_with("Could not save")) {
            break message.clone();
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let path = saved_to.strip_prefix("Saved the image to ").unwrap_or_else(|| panic!("{saved_to}")).to_string();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    (u32::from_be_bytes(bytes[16..20].try_into().unwrap()), u32::from_be_bytes(bytes[20..24].try_into().unwrap()))
}

#[test]
fn the_saved_image_has_the_size_that_is_chosen_for_it() {
    use super::save_size::SaveSizeChoice;
    let config = Config::sized(64, 48, 50, 1);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    assert_eq!(save_and_read_size(&mut app), (64, 48), "the default size is the default");
    app.save_size.choice = SaveSizeChoice::Preset(String::from("HD 720p"));
    //A preset needs a render of its own, since the buffer shows another size
    assert_eq!(save_and_read_size(&mut app), (1280, 720));
    app.save_size.choice = SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height) = (80, 20);
    assert_eq!(save_and_read_size(&mut app), (80, 20));
    app.save_size.choice = SaveSizeChoice::View;
    assert_eq!(save_and_read_size(&mut app), (64, 48));
    //A size that is not allowed is refused in the status log and nothing is saved
    app.save_size.choice = SaveSizeChoice::Custom;
    app.save_size.custom_width = 8;
    let before = app.log.len();
    app.save_image();
    assert!(app.log[before..].iter().any(|message| message.starts_with("Could not save the image: ") && message.contains("is not allowed")), "{:?}", &app.log[before..]);
}

#[test]
fn the_render_size_follows_the_window_and_keeps_the_view() {
    let ctx = egui::Context::default();
    let config = Config::default();
    assert!(config.fit_window);
    let mut app = manual_app(&config);
    assert_eq!(app.view_resolution(), Resolution::new(1200, 800), "the first render has the default size");
    app.c.set_view(&View::new(-0.7498, 0.0203, 0.064));
    app.render("Went to a view");
    wait_for_job(&mut app);
    let (before_center, before_scale) = (app.c.center(), app.c.get_scale());
    let original = app.c.clone();

    //The first fit is immediate
    app.fit_to_window(&ctx, egui::vec2(700.0, 500.0));
    wait_for_job(&mut app);
    assert_eq!(app.view_resolution(), Resolution::new(700, 500));
    let (after_center, after_scale) = (app.c.center(), app.c.get_scale());
    assert_eq!((after_center.x.to_bits(), after_scale.to_bits()), (before_center.x.to_bits(), before_scale.to_bits()), "the same part of the real axis is shown");
    assert!((after_center.y - before_center.y).abs() < 1e-15);
    assert!(app.log.last().unwrap().starts_with("Resized the view to 700 x 500 pixels in "), "{:?}", app.log.last());

    //The image is exactly the image that a direct render of the resized plane gives
    let mut direct = PixelBuffer::new(PixelPlane::new(700, 500));
    rendering::render_complex_plane_into_buffer(&mut direct, &original.resized(700, 500), &app.m, app.supersampling_amount, app.coloring_function);
    let colors = |buffer: &PixelBuffer| buffer.colors.iter().map(|c| (c.red, c.green, c.blue)).collect::<Vec<_>>();
    assert_eq!(colors(&app.p), colors(&direct));

    //A change of the window waits until it has settled; the buffer keeps its size meanwhile
    app.fit_to_window(&ctx, egui::vec2(650.0, 450.0));
    assert_eq!(app.view_resolution(), Resolution::new(700, 500));
    assert!(!app.is_busy());

    //Off: the window can be any size
    app.fit.set_enabled(false);
    app.fit_to_window(&ctx, egui::vec2(300.0, 300.0));
    assert!(!app.is_busy() && app.view_resolution() == Resolution::new(700, 500));
}

#[test]
fn a_fixed_size_is_kept_until_the_window_is_followed_on_request() {
    let ctx = egui::Context::default();
    let config = Config::sized(64, 48, 1000, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    app.fit_to_window(&ctx, egui::vec2(500.0, 400.0));
    assert!(!app.is_busy() && app.view_resolution() == Resolution::new(64, 48), "an explicit size is not changed by the window");
    app.fit.set_enabled(true);
    app.fit_to_window(&ctx, egui::vec2(500.0, 400.0));
    wait_for_job(&mut app);
    assert_eq!(app.view_resolution(), Resolution::new(500, 400));
    //Clicking in the resized image maps to the right point: the center of the image is the center of the view
    let center = app.c.center();
    let clicked = app.c.complex_from_pixel_plane(250.0, 200.0);
    assert!((clicked.x - center.x).abs() < app.c.increment_x && (clicked.y - center.y).abs() < app.c.increment_y);
}

#[test]
fn back_and_forward_return_to_the_views_that_were_visited() {
    let config = Config::sized(120, 80, 1000, 32);
    let mut app = manual_app(&config);
    assert!(!app.history.can_go_back(), "the start is the only view");
    let (start_center, start_scale) = (app.c.center(), app.c.get_scale());
    app.c.set_view(&View::new(-0.75, 0.1, 0.5));
    app.render("Went to a view");
    wait_for_job(&mut app);
    app.c.scale(0.5);
    app.render("Zoomed in");
    wait_for_job(&mut app);
    app.max_iterations_input = 300;
    app.m.max_iterations = 300;
    app.render("Set max. iterations to 300"); //The same view again: no new entry
    wait_for_job(&mut app);
    assert_eq!(app.history.len(), 3);

    app.go_back();
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.5).abs() < 1e-12 && (app.c.center().x + 0.75).abs() < 1e-12);
    assert!(app.log.iter().any(|message| message.starts_with("Went back to the previous view in ")));
    app.go_back();
    wait_for_job(&mut app);
    assert_eq!((app.c.center().x, app.c.get_scale()), (start_center.x, start_scale));
    assert!(!app.history.can_go_back() && app.history.can_go_forward());
    app.go_forward();
    wait_for_job(&mut app);
    app.go_forward();
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.25).abs() < 1e-12 && !app.history.can_go_forward());
    assert_eq!(app.history.len(), 3, "walking through the history adds nothing to it");

    //After a resize of the window, an old view is shown at the new size
    app.resize_view(Resolution::new(90, 90));
    wait_for_job(&mut app);
    app.go_back();
    wait_for_job(&mut app);
    assert_eq!(app.view_resolution(), Resolution::new(90, 90));
    assert!((app.c.get_scale() - 0.5).abs() < 1e-12);
}

/// One frame of the image alone in a window of `width` x `height` points, with `events` from the mouse
fn image_frame(app: &mut MandelbrotApp, ctx: &egui::Context, width: f32, height: f32, events: Vec<egui::Event>) {
    app.poll_job();
    app.update_texture(ctx);
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
    let output = ctx.run_ui(egui::RawInput { screen_rect: Some(screen), events, ..Default::default() }, |ui| {
        egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| {
            app.image_ui(ui);
        });
    });
    output.drop_without_applying_deltas();
}

#[test]
fn dragging_moves_the_view_and_the_wheel_zooms_around_the_cursor() {
    let ctx = egui::Context::default();
    let config = Config::sized(120, 80, 1000, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    let pointer = |position: egui::Pos2| egui::Event::PointerMoved(position);
    let button = |position: egui::Pos2, pressed: bool| egui::Event::PointerButton { pos: position, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    let (start, end) = (egui::pos2(60.0, 40.0), egui::pos2(90.0, 60.0));

    //Drag the image 30 points to the right and 20 down: the view moves 30 pixels left and 20 up
    let (center, increment) = (app.c.center(), (app.c.increment_x, app.c.increment_y));
    image_frame(&mut app, &ctx, 120.0, 80.0, vec![pointer(start)]);
    image_frame(&mut app, &ctx, 120.0, 80.0, vec![button(start, true)]);
    for step in 1..=6 {
        let position = start + (end - start) * (step as f32 / 6.0);
        image_frame(&mut app, &ctx, 120.0, 80.0, vec![pointer(position)]);
        wait_for_job(&mut app);
    }
    image_frame(&mut app, &ctx, 120.0, 80.0, vec![button(end, false)]);
    wait_for_job(&mut app);
    let moved = app.c.center();
    let (columns, rows) = ((center.x - moved.x) / increment.0, (moved.y - center.y) / increment.1);
    assert!(columns > 10.0 && columns <= 30.5 && rows > 6.0 && rows <= 20.5, "the view moved {columns} columns left and {rows} rows up");
    assert!(app.log.iter().filter(|message| message.starts_with("Moved by dragging")).count() <= 2, "a drag does not log a line for every render: {:?}", app.log);
    assert!(!app.gesture_active || app.is_busy());

    //Scroll up over a point: it zooms in and the point stays under the cursor
    let cursor = egui::pos2(100.0, 20.0);
    let (scale, under_cursor) = (app.c.get_scale(), app.c.complex_from_pixel_plane(100.0, 20.0));
    image_frame(&mut app, &ctx, 120.0, 80.0, vec![pointer(cursor)]);
    for _ in 0..8 {
        image_frame(&mut app, &ctx, 120.0, 80.0, vec![egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, 25.0), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers: egui::Modifiers::NONE }]);
        wait_for_job(&mut app);
    }
    assert!(app.c.get_scale() < scale * 0.9, "zoomed in from {scale} to {}", app.c.get_scale());
    let after = app.c.complex_from_pixel_plane(100.0, 20.0);
    assert!((after.x - under_cursor.x).abs() < 1e-9 && (after.y - under_cursor.y).abs() < 1e-9, "{after:?} {under_cursor:?}");
}

#[test]
fn the_last_image_follows_a_new_view_until_the_render_of_that_view_arrives() {
    let config = Config::sized(120, 80, 100, 1);
    let mut app = manual_app(&config);
    let full = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    assert_eq!(super::transform::shown_fraction(&app.shown_plane, &app.c), full);

    //Move 12 columns to the right: until the new render shows a layer, the old image is drawn 12 columns (a tenth of the image) to the left
    crate::rendering::move_complex_plane_by_pixels(&mut app.c, 0, 12);
    app.render_moved("Moved");
    let moved = super::transform::shown_fraction(&app.shown_plane, &app.c);
    assert!((moved.min.x + 0.1).abs() < 1e-5 && moved.min.y.abs() < 1e-5, "{moved:?}");
    wait_for_job(&mut app);
    let after = super::transform::shown_fraction(&app.shown_plane, &app.c);
    assert!((after.min - full.min).length() < 1e-5 && (after.max - full.max).length() < 1e-5, "the rendered image fills the view again: {after:?}");
}

#[test]
fn a_wheel_zoom_renders_quickly_with_a_steady_limit_and_the_full_quality_follows_when_it_stops() {
    let ctx = egui::Context::default();
    let config = Config::sized(120, 80, 100, 6);
    let mut app = MandelbrotApp::new(&config); //Auto iterations and adaptive supersampling are on by default
    wait_for_job(&mut app);
    let limit = app.m.max_iterations;

    //Zooming with the wheel: the first layer only, with the limit of the image that is shown
    app.gesture_active = true;
    app.c.scale(0.5);
    app.render("Zoomed in with the wheel");
    assert_eq!(app.bars.as_ref().unwrap().total_layers, 1);
    assert!(!app.limit_pending);
    wait_for_job(&mut app);
    assert_eq!(app.m.max_iterations, limit, "no auto iterations probes during the gesture");
    assert!(!app.buffer_is_complete, "the image of a gesture does not have the full quality");
    app.settle_gesture(&ctx);
    assert!(!app.is_busy(), "the full-quality render waits while the gesture goes on");

    //The wheel stops: the full-quality render starts, chooses the limit again, and shows no previews over the layer that is on the screen
    app.gesture_active = false;
    app.settle_gesture(&ctx);
    assert!(app.is_busy() && app.settle.is_none());
    assert_eq!(app.bars.as_ref().unwrap().total_layers, 6);
    wait_for_job(&mut app);
    assert!(app.buffer_is_complete);
    assert!(app.log.last().unwrap().starts_with("Zoomed in with the wheel (auto iterations: limit "), "{:?}", app.log.last());
}

#[test]
fn a_drag_shifts_the_shown_image_even_when_its_render_was_interrupted() {
    let config = Config::sized(120, 80, 100, 4);
    let mut app = manual_app(&config);
    app.gesture_active = true;
    crate::rendering::move_complex_plane_by_pixels(&mut app.c, 0, 5);
    app.render_moved("Moved by dragging");
    //The next frame of the drag interrupts that render: its samples go on to the next render, which shifts them by the second move
    crate::rendering::move_complex_plane_by_pixels(&mut app.c, 2, 4);
    app.render_moved("Moved by dragging");
    wait_for_job(&mut app);

    //The drag stops: the full-quality render continues every sample, and gives exactly the image of a complete render moved the same way (9 columns right, 2 rows up)
    app.gesture_active = false;
    app.settle_gesture(&egui::Context::default());
    wait_for_job(&mut app);
    assert!(app.buffer_is_complete);
    assert!(app.log.last().unwrap().starts_with(&format!("Moved by dragging (continued the samples of {} pixels) in ", 120 * 80)), "every pixel has its first sample from the drag: {:?}", app.log.last());
    assert_eq!(app.p.pixels, moved_after_a_complete_render(&config, None, 2, 9));
}

#[test]
fn a_coarse_preview_is_uploaded_with_one_texel_per_block_centered_on_its_pixel() {
    let (width, height, block) = (10, 6, 4);
    let colors: Vec<TrueColor> = (0..width * height).map(|i| TrueColor::new(i as u8, 0, 0)).collect();
    let (image, uv) = coarse_texture(&colors, ColorChannelMapping::RGB, width, height, block);
    assert_eq!(image.size, [3, 2], "the last blocks are partial");
    assert_eq!(image.pixels[1].r(), 4, "texel (1, 0) is the color of pixel (4, 0)");
    assert_eq!(image.pixels[3 + 2].r(), (4 * width + 8) as u8, "texel (2, 1) is pixel (8, 4)");
    //Drawn over the whole image, the center of pixel 4 (x = 4.5) shows the center of texel 1, and the center of row 4 the center of texel row 1
    let u = |x: f32| uv.min.x + (uv.max.x - uv.min.x) * x / width as f32;
    let v = |y: f32| uv.min.y + (uv.max.y - uv.min.y) * y / height as f32;
    assert!((u(4.5) - 1.5 / 3.0).abs() < 1e-6 && (u(0.5) - 0.5 / 3.0).abs() < 1e-6, "{uv:?}");
    assert!((v(4.5) - 1.5 / 2.0).abs() < 1e-6, "{uv:?}");
}

#[test]
fn previews_of_the_whole_view_are_uploaded_small_until_the_first_layer_arrives() {
    let config = Config::sized(64, 40, 100, 1);
    let mut app = manual_app(&config);
    let ctx = egui::Context::default();
    //Stand in for a render job of the whole view that reports three previews and a layer
    let (sender, receiver) = std::sync::mpsc::channel();
    app.job = Some(receiver);
    app.previews_received = 0;
    let colors = app.p.colors.clone();
    let mut sizes = Vec::new();
    for pass in [0, 0, 0, 1] {
        sender.send(JobMessage::Pass { colors: colors.clone(), plane: app.c.clone(), pass, total: 1, layer_time: Duration::ZERO, whole: true }).unwrap();
        app.poll_job();
        app.update_texture(&ctx);
        sizes.push((app.preview_block, app.texture.as_ref().unwrap().size()));
    }
    assert_eq!(sizes, [(8, [8, 5]), (4, [16, 10]), (2, [32, 20]), (1, [64, 40])]);
    assert_eq!(app.texture_uv, FULL_TEXTURE);
}

#[test]
fn going_to_a_point_flies_there_at_a_cinematic_pace() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    assert_eq!((app.pace, app.flight_rendering), (super::flight::Pace::Fast, super::flight::FlightRendering::LowRes), "the defaults, for speed");
    (app.pace, app.flight_rendering) = (super::flight::Pace::Cinematic, super::flight::FlightRendering::BestEffort);
    let ctx = egui::Context::default();
    let history = app.history.len();
    app.c.set_view(&View::new(-1.25, 0.1, 0.05));
    app.render("Somewhere else");
    wait_for_job(&mut app);
    app.go_to_point_of_interest(0);
    assert!(app.log.last().unwrap().starts_with("Flying to Test Place (Cinematic, "), "{:?}", app.log.last());
    let (started, duration) = app.flight.as_ref().map(|flight| (flight.last_frame, flight.flight.duration())).unwrap();
    assert!(duration >= Duration::from_secs(2), "{duration:?}");

    //A frame on the way renders towards the full quality (as many layers as it gets before the next frame), and no other render starts while the flight goes on
    app.fly_at(started + duration / 3);
    assert!(app.flight.is_some());
    assert_eq!(app.bars.as_ref().unwrap().total_layers, app.supersampling_amount);
    wait_for_job(&mut app);
    app.settle_gesture(&ctx);
    assert!(!app.is_busy(), "the arrival renders the place itself");
    let scale = app.c.get_scale();
    assert!(scale > 0.05 && scale < 1.0, "a third of the way, the view is zoomed out to travel, but not to the whole set: {scale}");

    //The arrival: the point, its limit, a full-quality render
    finish_flight(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18);
    assert_eq!(app.m.max_iterations, 321);
    assert_eq!(app.bars.as_ref().unwrap().total_layers, app.supersampling_amount.clamp(1, 64));
    assert!(app.log.last().unwrap().starts_with("Moved to Test Place in"), "{:?}", app.log.last());
    assert_eq!(app.history.len(), history + 2, "Back goes to the view before the flight, not to its frames");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn any_other_action_stops_a_flight_where_it_is() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::BestEffort;
    app.go_to_point_of_interest(0);
    assert!(app.log.last().unwrap().contains("(Fast, "), "{:?}", app.log.last());
    let started = app.flight.as_ref().unwrap().last_frame;
    app.fly_at(started + Duration::from_millis(200));
    let on_the_way = app.c.get_view();
    app.translate("up", 10, 0);
    assert!(app.flight.is_none(), "a move stops the flight");
    wait_for_job(&mut app);
    app.fly_at(started + Duration::from_secs(60));
    let mut expected = app.c.clone();
    expected.set_view(&on_the_way);
    crate::rendering::move_complex_plane_by_pixels(&mut expected, 10, 0);
    assert!((app.c.center().y - expected.center().y).abs() < 1e-12, "the view stays where the flight stopped, moved up");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn cancel_stops_a_flight_where_it_is_and_skip_goes_to_the_place_at_once() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.go_to_point_of_interest(0);
    let started = app.flight.as_ref().unwrap().last_frame;
    app.fly_at(started + Duration::from_millis(700));
    let share = app.flight.as_ref().map(|flight| flight.flight.share_of_path(flight.elapsed)).unwrap();
    assert!(share > 0.0 && share < 1.0, "the progress bar is part of the way: {share}");
    let on_the_way = app.c.get_view();
    app.cancel_flight();
    assert!(app.flight.is_none());
    assert_eq!(format!("{:?}", app.c.get_view()), format!("{on_the_way:?}"), "the view stays where the flight was");
    wait_for_job(&mut app);
    assert!(app.buffer_is_complete, "with a full-quality render");
    assert!(app.log.last().unwrap().starts_with("Stopped the flight to Test Place in"), "{:?}", app.log.last());

    app.go_to_point_of_interest(0);
    app.skip_flight();
    assert!(app.flight.is_none());
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18, "at the place");
    assert!(app.log.last().unwrap().starts_with("Moved to Test Place in"), "{:?}", app.log.last());
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn immersion_holds_the_view_until_the_new_view_has_a_complete_frame() {
    let config = Config::sized(64, 48, 50, 2);
    let mut app = manual_app(&config);
    let full = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    //Best effort: the old image moves along at once, with black where the exposed strip is not rendered yet
    app.translate("up", 10, 0);
    assert_ne!(app.drawn_fraction(), full);
    wait_for_job(&mut app);

    //Immersion: the screen keeps the last complete frame, unmoved, until the new view has one
    app.immersion = true;
    let before = app.shown_plane.get_view();
    app.translate("up", 10, 0);
    assert_eq!(app.drawn_fraction(), full, "no black void");
    assert_eq!(format!("{:?}", app.displayed_plane().get_view()), format!("{before:?}"), "the view on the screen has not moved yet");
    wait_for_job(&mut app);
    assert_eq!(format!("{:?}", app.shown_plane.get_view()), format!("{:?}", app.c.get_view()), "it moves when the frame is there");

    //Right-clicking the center of the screen while the view is held centers on what the screen shows, not on the view that is still being rendered
    let shown_center = app.shown_plane.center();
    app.translate("up", 10, 0);
    app.handle_image_click(32.0, 24.0, true);
    let center = app.c.center();
    assert!((center.x - shown_center.x).abs() <= app.c.increment_x && (center.y - shown_center.y).abs() <= app.c.increment_y, "{center:?} {shown_center:?}");
    wait_for_job(&mut app);
}

#[test]
fn a_frame_is_only_shown_when_it_is_complete_enough() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    let ctx = egui::Context::default();
    let colors = |value: u8| vec![TrueColor::new(value, value, value); 64 * 48];
    let show = |app: &mut MandelbrotApp, pass: u8, whole: bool, value: u8| {
        //Stands in for the running render job
        let (sender, receiver) = std::sync::mpsc::channel();
        app.job = Some(receiver);
        app.frame_ready = false;
        sender.send(JobMessage::Pass { colors: colors(value), plane: app.c.clone(), pass, total: 1, layer_time: Duration::ZERO, whole }).unwrap();
        app.poll_job();
        app.update_texture(&ctx);
        (app.p.colors[0].red == value, app.frame_ready)
    };
    //Best effort shows everything; immersion only complete frames (a preview of the whole view, or a layer)
    assert_eq!(show(&mut app, 0, false, 1), (true, true));
    app.immersion = true;
    assert_eq!(show(&mut app, 0, false, 2), (true, true), "a low-resolution preview of only the exposed strip is complete too");
    assert_eq!(show(&mut app, 0, true, 3), (true, true), "and so is a preview of the whole view");
    //A flight in max quality waits for a full-resolution frame
    app.immersion = false;
    app.flight_rendering = super::flight::FlightRendering::MaxQuality;
    app.go_to_point_of_interest(0);
    assert_eq!(show(&mut app, 0, true, 4), (false, false));
    assert_eq!(show(&mut app, 1, true, 5), (true, true));
    app.job = None;
    //Super sampled waits for the last layer: a full layer before it is not shown either
    app.flight_rendering = super::flight::FlightRendering::SuperSampled;
    app.go_to_point_of_interest(0);
    let show_layer = |app: &mut MandelbrotApp, pass: u8, value: u8| {
        let (sender, receiver) = std::sync::mpsc::channel();
        app.job = Some(receiver);
        app.frame_ready = false;
        sender.send(JobMessage::Pass { colors: vec![TrueColor::new(value, value, value); 64 * 48], plane: app.c.clone(), pass, total: 3, layer_time: Duration::ZERO, whole: true }).unwrap();
        app.poll_job();
        (app.p.colors[0].red == value, app.frame_ready)
    };
    assert_eq!(show_layer(&mut app, 2, 6), (false, false), "layer 2 of 3");
    assert_eq!(show_layer(&mut app, 3, 7), (true, true), "the finished frame");
    app.job = None;
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_flight_that_waits_for_its_frames_stands_still_until_the_frame_is_there() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::MaxQuality;
    app.go_to_point_of_interest(0);
    let start = app.flight.as_ref().unwrap().last_frame;
    //The first frame: the view on the screen is complete, so the camera moves by a frame and renders the new view
    app.fly_at(start + Duration::from_millis(16));
    let first = app.c.get_view();
    assert!(!app.frame_ready);
    //A second later the frame is not there yet: the camera and the flight clock wait
    app.fly_at(start + Duration::from_secs(1));
    assert_eq!(format!("{:?}", app.c.get_view()), format!("{first:?}"));
    assert_eq!(app.flight.as_ref().unwrap().elapsed, Duration::from_millis(16));
    //Once the frame is there, the camera moves on by at most one step, however long the wait was
    wait_for_job(&mut app);
    assert!(app.frame_ready);
    app.fly_at(start + Duration::from_secs(3));
    assert_eq!(app.flight.as_ref().unwrap().elapsed, Duration::from_millis(16 + 33));
    assert_ne!(format!("{:?}", app.c.get_view()), format!("{first:?}"));
    app.cancel_flight();
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_super_sampled_flight_renders_every_layer_of_every_frame() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::SuperSampled;
    app.go_to_point_of_interest(0);
    let start = app.flight.as_ref().unwrap().last_frame;
    app.fly_at(start + Duration::from_millis(16));
    assert_eq!(app.bars.as_ref().unwrap().total_layers, app.supersampling_amount, "a frame on the way gets all its layers");
    let first = app.c.get_view();
    app.fly_at(start + Duration::from_secs(1));
    if app.is_busy() {
        assert_eq!(format!("{:?}", app.c.get_view()), format!("{first:?}"), "the camera waits while the frame is not finished");
    }
    wait_for_job(&mut app);
    assert!(app.frame_ready, "the finished frame is there");
    assert_eq!(app.bars.as_ref().unwrap().layer_times.len(), usize::from(app.supersampling_amount), "with every layer");
    app.fly_at(start + Duration::from_secs(2));
    assert_ne!(format!("{:?}", app.c.get_view()), format!("{first:?}"), "and moves on when it is");
    app.cancel_flight();
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn with_auto_iterations_the_limit_follows_the_zoom_of_a_flight_and_reaches_the_limit_of_the_destination() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.auto_iterations = true;
    app.flight_rendering = super::flight::FlightRendering::BestEffort;
    app.render("Auto iterations");
    wait_for_job(&mut app);
    let start_limit = app.m.max_iterations;
    app.go_to_point_of_interest(0);
    app.flight.as_mut().unwrap().limits.as_mut().unwrap().wait_for_destination();
    //What auto iterations chooses at the destination (the render on arrival runs exactly this)
    let location = &app.points[0].location;
    let mut destination = ComplexPlane::new(64, 48);
    destination.set_view(&View::new(location.real, location.imaginary, location.scale));
    let mut m = app.m.clone();
    let initial = app.auto_settings.initial_limit(destination.get_scale());
    (m.max_iterations, m.palette_scale) = (initial, Some(initial));
    let destination_limit = crate::auto_iterations::choose_limit(&destination, &m, 64, 48, &app.auto_settings, 2, &std::sync::atomic::AtomicBool::new(false)).unwrap().limit;
    assert!(destination_limit > initial, "the probes raise the limit above the zoom rule here: {destination_limit} > {initial}");

    let start = app.flight.as_ref().unwrap().last_frame;
    let duration = app.flight.as_ref().unwrap().flight.duration();
    let mut limits = Vec::new();
    for share in [0.2, 0.4, 0.6, 0.8, 0.9999] {
        app.fly_at(start + duration.mul_f64(share));
        let scale = app.c.get_scale();
        assert_eq!(app.m.palette_scale, Some(app.auto_settings.initial_limit(scale)), "the palette of the zoom, as auto iterations uses it");
        limits.push((scale, app.m.max_iterations));
        wait_for_job(&mut app);
        assert_eq!(app.m.max_iterations, limits.last().unwrap().1, "the frame is rendered with it");
    }
    assert!(limits.windows(2).all(|pair| pair[0].1 <= pair[1].1), "zooming in raises the limit: {limits:?}");
    let (last_scale, last_limit) = *limits.last().unwrap();
    assert!((last_scale / destination.get_scale() - 1.0).abs() < 0.01 && (f64::from(last_limit) / f64::from(destination_limit) - 1.0).abs() < 0.02, "near the destination the limit is the destination's: {last_limit} for {destination_limit}");
    assert!(f64::from(limits[0].1) >= f64::from(start_limit) * 0.99, "and it starts from the limit of the start: {limits:?}, {start_limit}");

    //The arrival: the same limit, no jump
    finish_flight(&mut app);
    assert_eq!(app.m.max_iterations, destination_limit);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn every_point_of_interest_shows_its_zoom_in_the_list() {
    let (app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    assert_eq!(super::points::point_zoom_text(&app.points[0]), "Zoom \u{d7}15,625", "scale 0.000064");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn the_markers_of_the_points_of_interest_are_at_their_locations_and_only_inside_the_view() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    assert!(!app.show_points, "off by default");
    let rect = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(640.0, 480.0));
    //The default view: the place is inside it, at the pixel of its location
    let markers = super::points::marker_positions(&app.points, &app.c, rect);
    assert_eq!(markers.len(), 1);
    assert_eq!(markers[0].0, "Test Place");
    let location = &app.points[0].location;
    let pixel = |plane: &ComplexPlane, position: egui::Pos2| {
        let (x, y) = (f64::from((position.x - rect.min.x) / rect.width()) * 64.0, f64::from((position.y - rect.min.y) / rect.height()) * 48.0);
        plane.complex_from_pixel_plane(x, y)
    };
    let at = pixel(&app.c, markers[0].1);
    assert!((at.x - location.real).abs() < 1e-5 && (at.y - location.imaginary).abs() < 1e-5, "{at:?}");
    //A view far away from it: no marker
    app.c.set_view(&View::new(0.3, 0.6, 0.01));
    assert_eq!(super::points::marker_positions(&app.points, &app.c, rect).len(), 0);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn the_current_view_can_be_saved_as_a_point_of_interest() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.c.set_view(&View::new(-0.1, 0.65, 0.002));
    app.render("Somewhere");
    wait_for_job(&mut app);
    //Without a name nothing is saved and the form stays open
    app.new_point = Some((String::from("   "), String::from("ignored")));
    app.save_new_point();
    assert_eq!(app.points.len(), 1);
    assert!(app.new_point.is_some());
    //With a name: a new file, listed at its zoom, at exactly this view
    app.new_point = Some((String::from(" My Place "), String::new()));
    app.save_new_point();
    assert!(app.new_point.is_none(), "the form closes");
    assert!(app.log.iter().any(|message| message == "Saved the view as the point of interest My Place (00-my-place.json)"), "{:?}", app.log);
    assert_eq!(app.points.len(), 2);
    let saved = app.points.iter().find(|point| point.name == "My Place").unwrap();
    let center = app.c.center();
    assert_eq!((saved.location.real.to_bits(), saved.location.imaginary.to_bits(), saved.location.scale.to_bits()), (center.x.to_bits(), center.y.to_bits(), app.c.get_scale().to_bits()));
    assert_eq!((saved.location.max_iterations, saved.description.as_str()), (app.m.max_iterations, ""));
    assert!(folder.join("00-my-place.json").exists());
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_best_effort_flight_does_not_interrupt_a_render_that_has_not_shown_anything() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::BestEffort;
    app.go_to_point_of_interest(0);
    let start = app.flight.as_ref().unwrap().last_frame;
    app.fly_at(start + Duration::from_millis(100));
    let first_job = app.progress.as_ref().map(|progress| Arc::as_ptr(&progress.cancel)).unwrap();
    let first_view = app.c.get_view();
    //The next frame comes before the render has shown anything: the camera moves on, the render goes on
    app.fly_at(start + Duration::from_millis(200));
    assert_ne!(format!("{:?}", app.c.get_view()), format!("{first_view:?}"), "the camera follows the clock");
    assert_eq!(app.progress.as_ref().map(|progress| Arc::as_ptr(&progress.cancel)), Some(first_job), "the render that has not shown anything is not interrupted");
    let camera = app.c.get_view();
    wait_for_job(&mut app);
    assert!(app.frame_accepted);
    assert_eq!(format!("{:?}", app.c.get_view()), format!("{camera:?}"), "the finished render does not set the camera back to its own view");
    //Once it has shown a frame, the next frame starts a render at the latest camera
    app.fly_at(start + Duration::from_millis(300));
    assert!(app.is_busy() && app.progress.as_ref().map(|progress| Arc::as_ptr(&progress.cancel)) != Some(first_job));
    app.cancel_flight();
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn every_kind_of_render_decides_its_layers_its_limit_and_its_history() {
    use super::render_job::RenderKind;
    let config = Config::sized(64, 48, 50, 4);
    let mut app = manual_app(&config);
    let history = app.history.len();
    for (kind, layers, recorded) in [(RenderKind::Settled { moved: false }, 4, true), (RenderKind::GestureFrame { moved: true }, 1, false), (RenderKind::FlightFrame, 4, false)] {
        app.c.scale(0.9);
        let before = app.history.len();
        app.start_render_as("A render", kind);
        assert_eq!(app.bars.as_ref().unwrap().total_layers, layers, "{kind:?}");
        assert_eq!(app.history.len() > before, recorded, "{kind:?}");
        assert_eq!(app.settle.is_some(), kind != RenderKind::Settled { moved: false }, "{kind:?}: a passing frame leaves a render to settle on");
        wait_for_job(&mut app);
    }
    assert_eq!(app.history.len(), history + 1);
}

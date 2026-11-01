//! Tests of the window: rendering jobs, moves, the controls, the points of interest and their thumbnails, and the current view.
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use eframe::egui::{self, ColorImage, TextureOptions};

use super::*;
use crate::mandelbrot_set::ColorCycle;
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
    assert_eq!(app.m.palette, ColorCycle::CYCLE, "the colors do not depend on the limit");
    let manual_pixels = app.p.pixels.clone();

    //Turn auto on: the limit is chosen for the zoom, and shown in the max. iterations field
    app.auto_iterations = true;
    app.render("Turned auto iterations on");
    wait_for_job(&mut app);
    let limit = app.m.max_iterations;
    assert_eq!(app.m.palette, ColorCycle::CYCLE, "auto iterations do not change the colors");
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
    assert_eq!(app.m.palette, ColorCycle::CYCLE);
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
    assert_eq!(app.m.palette, ColorCycle::CYCLE);
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
    assert_eq!(app.m.palette, ColorCycle::CYCLE);
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
    assert_eq!(app.m.palette, ColorCycle::CYCLE);
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
    assert_eq!(app.m.max_iterations, initial, "published synchronously on selection");
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
    assert_eq!(app.m.palette, ColorCycle::CYCLE);
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
fn the_controls_are_in_sections_with_the_image_quality_in_its_own_and_saving_with_the_video() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    let texts = controls_texts(&mut app);
    let position = |needle: &str| texts.iter().position(|t| t.starts_with(needle)).unwrap_or_else(|| panic!("{needle:?} is not drawn: {texts:?}"));
    let sections = ["Navigation", "Places", "Image quality", "Colors", "Flights", "Performance", "Save image and video", "Benchmark"].map(&position);
    assert!(sections.windows(2).all(|pair| pair[0] < pair[1]), "the sections in order: {sections:?}");
    let [_, _, quality_section, colors, flights, performance, saving, benchmark] = sections;
    //Every control in its section
    for (needle, section, next) in [("Quality:", quality_section, colors), ("Max. iterations:", quality_section, colors), ("Coloring:", colors, flights), ("Color channels", colors, flights),
        ("Flight:", flights, performance), ("Frames ahead:", flights, performance), ("Saved image quality:", saving, benchmark), ("Video of the last flight", saving, benchmark)] {
        let at = position(needle);
        assert!(section < at && at < next, "{needle} is in its section");
    }
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
    let current = MandelbrotApp::thumbnail_request_for(&point, app.coloring_index, false, app.p.color_channel_mapping, app.m.palette, (64, 48));
    let restored = MandelbrotApp::thumbnail_request_for(&point, app.coloring_index, true, app.p.color_channel_mapping, app.m.palette, (64, 48));
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
    app.fast_gpu = false; //Compared with the CPU bit for bit below: the fast GPU math may differ in a few pixels
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
    app.settle.as_mut().unwrap().1 = std::time::Instant::now(); //The wheel just turned (the wait above can take longer than the pause on a busy machine)
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
    assert_eq!((app.pace, app.flight_rendering), (super::flight::Pace::Fast, super::flight::FlightRendering::BestEffort), "the defaults, for speed");
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
    app.flight_rendering = super::flight::FlightRendering::FULL_RESOLUTION;
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

/// Lets a flight that waits for its frames run in real time until it arrives, as the window does (see `flight_job`), and returns the flight clock at every frame
fn fly_until_arrived(app: &mut MandelbrotApp) -> Vec<(Duration, usize)> {
    let started = std::time::Instant::now();
    let mut clocks = Vec::new();
    while let Some(flight) = &app.flight {
        assert!(started.elapsed() < Duration::from_secs(60), "the flight did not arrive");
        let frames = flight.frames.len();
        app.poll_job();
        //(Fast frames can arrive more than one at a time: then only the clock of the last is seen, with the frames it took)
        if let Some(flight) = app.flight.as_ref().filter(|flight| flight.frames.len() > frames) {
            clocks.push((flight.elapsed, flight.frames.len() - frames));
        }
        app.fly_at(std::time::Instant::now());
        std::thread::sleep(Duration::from_millis(1));
    }
    clocks
}

#[test]
fn a_flight_that_waits_for_its_frames_is_rendered_frame_after_frame_by_one_job() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::FULL_RESOLUTION;
    app.go_to_point_of_interest(0);
    let (before, duration) = (app.c.get_view(), app.flight.as_ref().unwrap().flight.duration());
    app.fly_at(std::time::Instant::now());
    assert!(app.flight_job && app.is_busy(), "one job renders the frames");
    assert_eq!(format!("{:?}", app.c.get_view()), format!("{before:?}"), "the camera waits for the first frame");
    let clocks = fly_until_arrived(&mut app);
    assert!(clocks.len() >= 3, "frames on the way: {clocks:?}");
    assert!(clocks.windows(2).all(|pair| pair[1].0 > pair[0].0 && pair[1].0.saturating_sub(pair[0].0) <= super::points::MAX_WAITING_STEP * u32::try_from(pair[1].1).unwrap()), "the clock moves on by at most a step per frame: {clocks:?}");
    assert_eq!(clocks.last().unwrap().0, duration, "the last frame is at the destination");
    //The arrival: the place itself, with a full-quality render
    assert!(!app.flight_job);
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18 && app.buffer_is_complete);
    assert!(app.log.iter().any(|line| line.starts_with("The flight to Test Place: ")), "{:?}", app.log);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_super_sampled_flight_shows_every_frame_with_all_its_layers() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::SuperSampled;
    app.go_to_point_of_interest(0);
    app.fly_at(std::time::Instant::now());
    //The first frame on the screen is the finished image of its view: every layer
    let started = std::time::Instant::now();
    while app.flight.as_ref().is_some_and(|flight| flight.frames.is_empty()) {
        assert!(started.elapsed() < Duration::from_secs(30), "no frame");
        app.poll_job();
        std::thread::sleep(Duration::from_millis(1));
    }
    let (width, height) = (app.p.pixel_plane.width, app.p.pixel_plane.height);
    let mut finished = Vec::new();
    rendering::Render::new(&app.shown_plane, &app.m, width, height, app.coloring_function).supersampling(app.supersampling_amount).run(|colors, _| finished = colors.to_vec());
    assert!(app.p.colors.iter().zip(&finished).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)), "the frame is the finished render of its view");
    app.cancel_flight();
    assert!(!app.flight_job, "Cancel stops the job");
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
    app.flight.as_ref().unwrap().limits.as_ref().unwrap().lock().unwrap().wait_for_destination();
    //What auto iterations chooses at the destination (the render on arrival runs exactly this)
    let location = &app.points[0].location;
    let mut destination = ComplexPlane::new(64, 48);
    destination.set_view(&View::new(location.real, location.imaginary, location.scale));
    let mut m = app.m.clone();
    let initial = app.auto_settings.initial_limit(destination.get_scale());
    m.max_iterations = initial;
    let destination_limit = crate::auto_iterations::choose_limit(&destination, &m, 64, 48, &app.auto_settings, 2, &std::sync::atomic::AtomicBool::new(false)).unwrap().limit;
    assert!(destination_limit > initial, "the probes raise the limit above the zoom rule here: {destination_limit} > {initial}");

    let start = app.flight.as_ref().unwrap().last_frame;
    let duration = app.flight.as_ref().unwrap().flight.duration();
    let mut limits = Vec::new();
    for share in [0.2, 0.4, 0.6, 0.8, 0.9999] {
        app.fly_at(start + duration.mul_f64(share));
        let scale = app.c.get_scale();
        assert_eq!(app.m.palette, ColorCycle::CYCLE, "the colors stay the same on the way");
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
fn the_depth_color_runs_from_cool_for_the_shallowest_place_to_hot_for_the_deepest() {
    let config = Config::sized(64, 48, 50, 32);
    let app = MandelbrotApp::new(&config);
    let depth = super::points::DepthScale::of(&app.points);
    let colors: Vec<egui::Color32> = app.points.iter().map(|point| depth.color(point)).collect();
    //The list is sorted by zoom: the reset view first (blue), the deepest place last (red)
    assert_eq!(colors[0], egui::Color32::from_rgb(80, 150, 255));
    assert_eq!(*colors.last().unwrap(), egui::Color32::from_rgb(255, 85, 75));
    //Places of a different zoom get different colors, places of the same zoom the same color
    for (a, b) in app.points.iter().zip(&colors).zip(app.points.iter().zip(&colors).skip(1)) {
        assert_eq!(a.0.location.scale.to_bits() == b.0.location.scale.to_bits(), a.1 == b.1, "{} and {}", a.0.name, b.0.name);
    }
    //A single place, or places that are all equally deep, are the shallowest color
    let single = super::points::DepthScale::of(&app.points[..1]);
    assert_eq!(single.color(&app.points[0]), egui::Color32::from_rgb(80, 150, 255));
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
    //(The first job's cancel flag is kept, so a later job cannot get its address: identity by address only holds while it lives)
    let first_cancel = app.progress.as_ref().map(|progress| Arc::clone(&progress.cancel)).unwrap();
    let first_job = Arc::as_ptr(&first_cancel);
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

#[test]
fn the_smooth_colors_setting_applies_to_every_render_and_the_window_keeps_it() {
    use super::render_job::RenderKind;
    let config = Config::sized(64, 48, 50, 1);
    let mut app = manual_app(&config);
    assert!(app.m.smooth_coloring, "on by default");
    let smooth = app.p.pixels.clone();
    app.m.smooth_coloring = false;
    app.render("Turned smooth colors off");
    wait_for_job(&mut app);
    let integer = app.p.pixels.clone();
    assert_ne!(smooth, integer, "whole iteration counts give other colors");
    assert!(!app.m.smooth_coloring, "the window keeps its setting after the render");
    app.start_render_as("A flight frame", RenderKind::FlightFrame);
    wait_for_job(&mut app);
    assert_eq!(app.p.pixels, integer, "a flight frame follows the setting like any render");
    app.m.smooth_coloring = true;
    app.render("Turned smooth colors on");
    wait_for_job(&mut app);
    assert_eq!(app.p.pixels, smooth);
}

#[test]
fn the_zoom_stops_where_the_fast_gpu_math_ends_with_a_warning_in_the_log() {
    let config = Config::sized(64, 48, 50, 1);
    let mut app = manual_app(&config);
    assert!(app.fast_gpu && app.limit_zoom, "fast GPU math on, and the zoom limited to it by default");
    app.limit_zoom = true;
    let mut deep = app.c.clone();
    deep.set_view(&View::new(-0.743_643_887_037_151, 0.131_825_904_205_33, 1e-11));
    let mut shallow = app.c.clone();
    shallow.set_view(&View::new(-0.743_643_887_037_151, 0.131_825_904_205_33, 1e-8));
    assert!(!app.zoom_refused(&deep), "without a GPU there is no fast GPU math to stay with");
    let (_sender, receiver) = std::sync::mpsc::channel();
    app.gpu_opening = Some(receiver); //A GPU that is still opening counts
    let logged = app.log.len();
    assert!(!app.zoom_refused(&shallow));
    assert!(app.zoom_refused(&deep));
    assert!(app.zoom_refused(&deep));
    assert_eq!(app.log.len(), logged + 1, "one warning for a row of refused zooms");
    assert!(app.log.last().unwrap().starts_with("Zoom limit: "), "{:?}", app.log.last());
    app.limit_zoom = false;
    assert!(!app.zoom_refused(&deep), "the limit switched off");
    (app.limit_zoom, app.fast_gpu) = (true, false);
    assert!(!app.zoom_refused(&deep), "without the fast GPU math there is no limit");
}

#[test]
fn a_flight_frame_computes_only_the_previews_its_flight_rendering_shows() {
    use super::flight::FlightRendering;
    use super::render_job::RenderKind;
    let mut app = manual_app(&Config::sized(64, 48, 50, 1));
    let all = rendering::PREVIEW_BLOCKS.to_vec();
    for (mode, levels) in [(FlightRendering::BestEffort, all.clone()), (FlightRendering::LowRes, all.clone()), (FlightRendering::Resolution(25), vec![4]), (FlightRendering::Resolution(33), vec![3]),
        (FlightRendering::Resolution(50), vec![2]), (FlightRendering::Resolution(10), vec![10]), (FlightRendering::FULL_RESOLUTION, vec![]), (FlightRendering::Resolution(200), vec![]), (FlightRendering::SuperSampled, vec![])] {
        app.flight_rendering = mode;
        assert_eq!(app.preview_levels(RenderKind::FlightFrame), levels, "{mode:?}");
        assert_eq!(app.preview_levels(RenderKind::Settled { moved: false }), all, "other renders keep every preview level");
        assert_eq!(app.preview_levels(RenderKind::GestureFrame { moved: true }), all);
    }
    app.flight_rendering = FlightRendering::BestEffort;
    app.immersion = true;
    assert_eq!(app.preview_levels(RenderKind::FlightFrame), all, "immersion shows complete previews");
}

#[test]
fn the_flight_resolution_decides_what_a_frame_waits_for() {
    use super::flight::FlightRendering;
    let of = |rendering| {
        let mut app = manual_app(&Config::sized(64, 48, 50, 1));
        app.flight_rendering = rendering;
        app.flight_readiness()
    };
    assert_eq!(of(FlightRendering::Resolution(25)), Readiness::Blocks(4), "quarter resolution: blocks of 4 x 4");
    assert_eq!(of(FlightRendering::Resolution(50)), Readiness::Blocks(2));
    assert_eq!(of(FlightRendering::Resolution(33)), Readiness::Blocks(3));
    assert_eq!(of(FlightRendering::Resolution(80)), Readiness::FULL_RESOLUTION, "close to 100%: a full layer");
    assert_eq!(of(FlightRendering::FULL_RESOLUTION), Readiness::Layers(1));
    assert_eq!(of(FlightRendering::Resolution(200)), Readiness::Layers(4), "200%: 4 samples per pixel");
    assert_eq!(of(FlightRendering::Resolution(800)), Readiness::Layers(64), "at most every layer");
    //From the least to the most complete
    let order = [Readiness::Any, Readiness::COMPLETE, Readiness::Blocks(8), Readiness::Blocks(4), Readiness::Blocks(2), Readiness::Layers(1), Readiness::Layers(4), Readiness::SuperSampled];
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
    assert_eq!(FlightRendering::Resolution(37).name(), "Resolution", "the percent is next to it");
    assert_eq!(FlightRendering::Resolution(25).entry(), "Quarter resolution");
}

#[test]
fn a_flight_counts_its_frames_shows_the_frame_rate_and_logs_the_average() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::BestEffort;
    app.go_to_point_of_interest(0);
    let started = app.flight.as_ref().unwrap().started;
    for step in 1..=3 {
        app.fly_at(started + Duration::from_millis(100 * step));
        wait_for_job(&mut app);
    }
    let flight = app.flight.as_ref().unwrap();
    assert!(flight.frames.len() >= 3, "every frame that was shown counts: {}", flight.frames.len());
    let last = *flight.frames.last().unwrap();
    assert_eq!(flight.fps(last), flight.frames.iter().filter(|shown| last.duration_since(**shown) <= Duration::from_secs(1)).count());
    assert_eq!(flight.fps(last + Duration::from_secs(5)), 0, "the frames of the second before");
    assert!(flight.frames.len() <= 4, "one frame per view on the way (the finer images of a view do not count): {}", flight.frames.len());
    //The progress bar shows the rate of the second before, updated once a second so it can be read
    let flight = app.flight.as_mut().unwrap();
    let updated = flight.fps_updated;
    flight.update_fps(updated + Duration::from_millis(500));
    assert_eq!(flight.fps_shown, 0, "not yet a second");
    flight.update_fps(last.max(updated + Duration::from_secs(1)));
    assert_eq!(flight.fps_shown, flight.fps(last.max(updated + Duration::from_secs(1))));
    app.skip_flight();
    assert!(app.log.iter().any(|line| line.starts_with("The flight to Test Place was skipped: ") && line.contains(" frames in ") && line.ends_with(" FPS on average")), "{:?}", app.log);
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn nothing_beyond_the_zoom_limit_is_rendered_whatever_leads_there() {
    const DEEP_PLACE: &str = r#"{ "format_version": 1, "name": "Deep Place", "location": { "center": { "real": -0.743643887037151, "imaginary": 0.13182590420533 }, "scale": 1e-11, "max_iterations": 500 } }"#;
    let (mut app, folder) = app_with_points(&[("deep-place.json", DEEP_PLACE)]);
    let deep = View::new(-0.743_643_887_037_151, 0.131_825_904_205_33, 1e-11);
    //A deep view in the history, from before the limit applied (no GPU yet)
    app.c.set_view(&deep);
    app.render("Deep");
    wait_for_job(&mut app);
    app.c.set_view(&View::new(-0.75, 0.0, 1.0));
    app.render("Shallow");
    wait_for_job(&mut app);
    let (_sender, receiver) = std::sync::mpsc::channel();
    app.gpu_opening = Some(receiver);
    app.limit_zoom = true; //The limit applies from now on
    let shallow = format!("{:?}", app.c.get_view());
    let unchanged = |app: &MandelbrotApp, what: &str| assert_eq!(format!("{:?}", app.c.get_view()), shallow, "{what}: the view stays");
    let warned = |app: &MandelbrotApp| app.log.iter().rev().take(2).any(|line| line.starts_with("Zoom limit: "));

    app.go_back();
    unchanged(&app, "Back");
    assert!(warned(&app) && app.history.can_go_back(), "the history stays where it was");
    app.view_inputs = [String::from("-0.743643887037151"), String::from("0.13182590420533"), String::from("1e-11")];
    app.go_to_view();
    unchanged(&app, "Go to view");
    app.go_to_point_of_interest(0);
    assert!(app.flight.is_none(), "no flight to a place beyond the limit");
    unchanged(&app, "a flight");
    assert!(!app.is_busy(), "and nothing is rendered");

    //Anything else that leads past the limit zooms out to it before rendering
    app.c.set_view(&deep);
    app.render("Somehow deep");
    assert!(app.c.precision_steps() >= rendering::FAST_GPU_STEPS && app.c.get_scale() > 1e-11, "zoomed out to the limit: scale {}", app.c.get_scale());
    assert!(app.c.precision_steps() < rendering::FAST_GPU_STEPS * 1.1, "but not further than needed");
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn go_to_view_keeps_the_current_scale_when_the_scale_is_empty() {
    let mut app = manual_app(&Config::sized(64, 48, 50, 1));
    app.c.set_view(&View::new(-0.75, 0.0, 0.25));
    app.view_inputs = [String::from("-0.5"), String::from("0.1"), String::from(" ")];
    app.go_to_view();
    assert!((app.c.center().x + 0.5).abs() < 1e-12 && (app.c.center().y - 0.1).abs() < 1e-12, "the entered center");
    assert!((app.c.get_scale() - 0.25).abs() < 1e-15, "the current scale: {}", app.c.get_scale());
    wait_for_job(&mut app);
    app.view_inputs = [String::from("-0.5"), String::from("0.1"), String::from("abc")];
    app.go_to_view();
    assert!(app.log.last().unwrap().starts_with("Could not go to the view"), "a scale that is not a number is still refused");
}

#[test]
fn the_spare_colors_are_reused_for_later_frames_and_the_pool_stays_small() {
    use super::render_job::SparePool;
    let pool = SparePool::default();
    let frame = vec![TrueColor::new(1, 2, 3); 100];
    let spare = vec![TrueColor::new(0, 0, 0); 100];
    let address = spare.as_ptr();
    pool.put(spare);
    let copy = pool.copy(&frame);
    assert!(copy.iter().zip(&frame).all(|(a, b)| (a.red, a.green, a.blue) == (b.red, b.green, b.blue)) && copy.len() == frame.len(), "a copy of the frame");
    assert_eq!(copy.as_ptr(), address, "in the memory of the spare colors");
    let second = pool.copy(&frame);
    assert_ne!(second.as_ptr(), address, "a spare is used once");
    for _ in 0..10 {
        pool.put(vec![TrueColor::new(0, 0, 0); 100]);
    }
    assert_eq!(pool.len(), SparePool::KEPT, "no more than a few are kept");
    pool.put(vec![TrueColor::new(0, 0, 0); 10]);
    assert_eq!(pool.copy(&frame).len(), 100);
    assert!(pool.len() < SparePool::KEPT, "colors of another size are dropped");
}

#[test]
fn with_no_flight_going_to_a_place_shows_it_at_once() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    assert!(super::flight::Pace::ALL.contains(&super::flight::Pace::NoFlight), "a choice of the drop-down");
    app.pace = super::flight::Pace::NoFlight;
    app.go_to_point_of_interest(0);
    assert!(app.flight.is_none(), "no flight");
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18 && app.m.max_iterations == 321, "the place itself, with its limit");
    assert!(app.is_busy(), "rendered at once");
    wait_for_job(&mut app);
    assert!(app.log.last().unwrap().starts_with("Moved to Test Place"), "{:?}", app.log.last());
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn the_blur_of_a_flight_frame_follows_the_resolution_it_really_has() {
    use super::blur::blur_sigma;
    //Full resolution with one sample: σ 0.8 (#118); coarser frames get more, supersampled ones less
    assert!((blur_sigma(1, 1) - 0.8).abs() < 1e-6);
    assert!((blur_sigma(4, 0) - 1.84).abs() < 0.01 && (blur_sigma(2, 0) - 1.21).abs() < 0.01);
    assert!((blur_sigma(1, 4) - 0.53).abs() < 0.01 && (blur_sigma(1, 9) - 0.41).abs() < 0.01);
    assert_eq!(blur_sigma(1, 0), blur_sigma(1, 1), "a preview of single pixels is a frame of one sample");
    //Without an OpenGL context (as here) there is no blur, and the image is drawn as it is
    let app = MandelbrotApp::new(&Config::sized(64, 48, 50, 32));
    assert!(app.blur.is_none() && app.auto_blur);
}

#[test]
fn auto_steers_to_the_finest_scale_that_keeps_the_frame_rate_and_holds_it_when_it_fits() {
    use super::flight::{auto_scale, AUTO_COARSEST};
    let ms = |ms: f64| Duration::from_secs_f64(ms / 1000.0);
    let target = ms(1000.0 / 60.0); //60 FPS: the frames aim for 90% of 16.7 ms, 15 ms
    //A frame that takes a fixed 4 ms plus 16 ms at full resolution (20 ms, 50 FPS): the scale settles where it fits, not at the coarse scale that the pixels
    //alone predict (blocks of 2 or 3)
    let took = |scale: f64| ms(4.0 + 16.0 * scale * scale);
    let mut scale = 0.7;
    for _ in 0..40 {
        scale = auto_scale(scale, took(scale), target);
    }
    let fits = took(scale).as_secs_f64() * 1000.0;
    assert!((0.77..=0.83).contains(&scale) && (13.5..=15.0).contains(&fits), "settled at {scale} ({fits:.1} ms)");
    //Steady once it fits: the same scale every frame
    assert_eq!(auto_scale(scale, took(scale), target), scale);
    //Fast views go to full resolution, slow ones down to the coarsest scale and no further
    let mut scale = 0.5;
    for _ in 0..20 {
        scale = auto_scale(scale, ms(5.0), target);
    }
    assert_eq!(scale, 1.0);
    for _ in 0..40 {
        scale = auto_scale(scale, ms(1000.0), target);
    }
    assert_eq!(scale, AUTO_COARSEST);
    //At most 20% coarser or 15% finer in one frame, and at least a percent
    assert_eq!(auto_scale(1.0, ms(1000.0), target), 0.8);
    assert_eq!(auto_scale(0.5, ms(1.0), target), 0.57);
    assert_eq!(auto_scale(0.5, ms(15.5), target), 0.49);
}

#[test]
fn an_auto_flight_shows_frames_of_the_chosen_resolution_and_arrives() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::Auto(60);
    assert_eq!(app.flight_readiness(), Readiness::COMPLETE);
    app.go_to_point_of_interest(0);
    app.fly_at(std::time::Instant::now());
    assert!(app.flight_job, "the flight job renders the frames");
    //The first frame is scaled (70% of the pixels across), and is drawn from the smaller image
    let (started, ctx, width, height) = (std::time::Instant::now(), egui::Context::default(), app.p.pixel_plane.width, app.p.pixel_plane.height);
    while app.shown_scaled.is_none() && app.flight.is_some() {
        assert!(started.elapsed() < Duration::from_secs(30), "no scaled frame");
        app.poll_job();
        std::thread::sleep(Duration::from_millis(1));
    }
    let (columns, rows) = app.shown_scaled.expect("a scaled frame");
    assert_eq!((columns, rows), ((width as f64 * 0.7).round() as usize, (height as f64 * 0.7).round() as usize));
    app.update_texture(&ctx);
    assert_eq!(app.texture.as_ref().unwrap().size(), [columns, rows]);
    let clocks = fly_until_arrived(&mut app);
    assert!(clocks.len() >= 2, "frames on the way: {clocks:?}");
    assert_eq!(app.shown_scaled, None, "the frame at the place is not scaled");
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18 && app.buffer_is_complete, "the place itself, fully rendered");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_blurred_image_that_reaches_past_the_window_is_drawn_only_where_it_is_visible() {
    use super::blur::visible_part;
    let rect = |x0: f32, y0: f32, x1: f32, y1: f32| egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));
    let full = rect(0.0, 0.0, 1.0, 1.0);
    //Zoomed in twice around the center of a 100 x 100 area: the middle half of the texture fills the area
    let (shown, uv) = visible_part(rect(-50.0, -50.0, 150.0, 150.0), rect(0.0, 0.0, 100.0, 100.0), full).unwrap();
    assert_eq!((shown, uv), (rect(0.0, 0.0, 100.0, 100.0), rect(0.25, 0.25, 0.75, 0.75)));
    //Moved half out to the left, with the texture coordinates of a coarse preview
    let (shown, uv) = visible_part(rect(-50.0, 0.0, 50.0, 100.0), rect(0.0, 0.0, 100.0, 100.0), rect(0.1, 0.0, 0.9, 0.8)).unwrap();
    assert_eq!(shown, rect(0.0, 0.0, 50.0, 100.0));
    assert!((uv.min.x - 0.5).abs() < 1e-6 && (uv.max.x - 0.9).abs() < 1e-6 && uv.min.y == 0.0 && (uv.max.y - 0.8).abs() < 1e-6);
    //Inside the area: unchanged; outside: nothing to draw
    assert_eq!(visible_part(rect(10.0, 10.0, 90.0, 90.0), rect(0.0, 0.0, 100.0, 100.0), full), Some((rect(10.0, 10.0, 90.0, 90.0), full)));
    assert_eq!(visible_part(rect(200.0, 0.0, 300.0, 100.0), rect(0.0, 0.0, 100.0, 100.0), full), None);
}

#[test]
fn the_settings_are_kept_in_their_file_and_reset_to_the_defaults() {
    let folder = std::env::temp_dir().join(format!("mandelbrot-settings-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    let path = folder.join("settings.json");
    //A first run: no file yet, the defaults
    let mut app = manual_app(&Config::sized(64, 48, 50, 4));
    assert!(!app.use_settings_file(Some(path.clone())));
    let defaults = app.settings();
    (app.pace, app.fast_gpu, app.supersampling_amount, app.m.smooth_coloring) = (super::flight::Pace::Cinematic, false, 16, false);
    app.flight_rendering = super::flight::FlightRendering::Resolution(37);
    (app.route, app.constant_speed, app.dive_zoom) = (super::flight::Route::Straight, 7.5, 250.0);
    (app.frames_ahead, app.blend_frames, app.f32_frames, app.quality_blur, app.neighborhood_margin) = (1, true, false, true, 250);
    (app.video_fps, app.video_speed, app.video_quality, app.video_playback, app.video_pngs) = (144, 10, super::video::VideoQuality::Lossless, super::video::Playback::HighQuality, true);
    app.video_blend = true;
    (app.video_fit_length, app.video_length) = (true, [String::from("1"), String::new(), String::from("7.5")]);
    (app.music_path, app.follow_music, app.music_strength) = (String::from("/music/track.mp3"), true, 0.8);
    app.m.palette = crate::mandelbrot_set::ColorCycle { divide: 2.5, offset: 40.0, stretch: false };
    //The saved image size (asked by Jort in the review of review-4)
    (app.save_size.choice, app.save_size.portrait, app.save_size.custom_width, app.save_size.custom_height) = (super::save_size::SaveSizeChoice::Preset(String::from("32K")), true, 5000, 3000);
    let now = std::time::Instant::now();
    app.keep_settings(now);
    assert!(!path.exists(), "not on every frame of a drag");
    app.keep_settings(now + Duration::from_secs(1));
    assert!(path.exists(), "written once the change has settled");
    //The next run gets them back
    let mut next = manual_app(&Config::sized(64, 48, 50, 4));
    assert!(next.use_settings_file(Some(path.clone())));
    assert_eq!((next.pace, next.fast_gpu, next.supersampling_amount, next.m.smooth_coloring, next.flight_rendering),
        (super::flight::Pace::Cinematic, false, 16, false, super::flight::FlightRendering::Resolution(37)));
    assert_eq!((next.route, next.constant_speed, next.dive_zoom), (super::flight::Route::Straight, 7.5, 250.0));
    assert_eq!((next.frames_ahead, next.blend_frames, next.f32_frames, next.quality_blur, next.neighborhood_margin), (1, true, false, true, 250));
    assert_eq!((next.video_fps, next.video_speed, next.video_quality, next.video_playback, next.video_pngs), (144, 10, super::video::VideoQuality::Lossless, super::video::Playback::HighQuality, true));
    assert!(next.video_blend);
    assert_eq!((next.video_fit_length, next.target_length()), (true, Some(Duration::from_secs_f64(3607.5))));
    assert_eq!((next.music_path.as_str(), next.follow_music, next.music_strength), ("/music/track.mp3", true, 0.8));
    assert_eq!(next.m.palette, crate::mandelbrot_set::ColorCycle { divide: 2.5, offset: 40.0, stretch: false });
    assert_eq!((next.save_size.choice.clone(), next.save_size.portrait, next.save_size.custom_width, next.save_size.custom_height), (super::save_size::SaveSizeChoice::Preset(String::from("32K")), true, 5000, 3000));
    assert_eq!(next.save_size.resolve(next.default_save_resolution(), next.view_resolution()), Ok(crate::resolution::Resolution::new(17_280, 30_720)), "32K upright, as it was");
    //A file with unknown and missing values loads what it knows
    std::fs::write(&path, r#"{ "pace": "Normal", "from_the_future": 3, "save_size": "preset:A deleted preset" }"#).unwrap();
    let mut partial = manual_app(&Config::sized(64, 48, 50, 4));
    partial.use_settings_file(Some(path));
    assert_eq!((partial.pace, partial.supersampling_amount), (super::flight::Pace::Normal, defaults.clone().supersampling_for_test()));
    assert_eq!(partial.save_size.choice, super::save_size::SaveSizeChoice::Standard, "a preset that is gone: the default size");
    //Reset
    next.reset_settings();
    assert_eq!(next.settings(), defaults);
    wait_for_job(&mut next);
}

#[test]
fn the_frame_statistics_of_the_benchmark() {
    let ms = Duration::from_millis;
    let mut times: Vec<Duration> = vec![ms(10); 99];
    times.push(ms(100));
    let stats = super::benchmark::frame_stats(&times, Duration::from_secs(1));
    assert_eq!((stats.frames, stats.min, stats.max), (100, 10.0, 100.0));
    assert!((stats.average - 10.9).abs() < 1e-9 && (stats.average_fps - 100.0).abs() < 1e-9);
    assert!((stats.low_fps - 10.0).abs() < 1e-9, "the slowest 1% is the 100 ms frame: {}", stats.low_fps);
    assert_eq!(stats.p95, 10.0);
}

#[test]
fn a_benchmark_tour_visits_the_deepest_places_reports_them_and_restores_the_settings() {
    const FROST: &str = r#"{ "format_version": 1, "name": "Frost", "location": { "center": { "real": -0.6604166666666667, "imaginary": 0.4437500000000001 }, "scale": 0.1, "max_iterations": 300 } }"#;
    let (mut app, folder) = app_with_points(&[("01-frost.json", FROST), ("02-test-place.json", HSV_BGR_PLACE)]);
    app.benchmark_folder = folder.join("benchmarks");
    let ctx = egui::Context::default();
    let (pace, rendering) = (app.pace, app.flight_rendering);
    app.start_benchmark();
    assert!(app.benchmark.is_some());
    let started = std::time::Instant::now();
    while app.benchmark.is_some() {
        assert!(started.elapsed() < Duration::from_secs(120), "the tour did not finish");
        app.poll_job();
        app.fly_at(std::time::Instant::now());
        app.step_benchmark(&ctx);
        std::thread::sleep(Duration::from_millis(2));
    }
    let names: Vec<&str> = app.benchmark_results.iter().map(|scene| scene.name.as_str()).collect();
    assert_eq!(names, ["Frost", "Test Place"], "shallow to deep");
    assert!(app.benchmark_results.iter().all(|scene| !scene.frame_times.is_empty() && scene.finished >= scene.first_layer));
    assert_eq!((app.pace, app.flight_rendering), (pace, rendering), "the settings are restored");
    let saved: Vec<_> = std::fs::read_dir(folder.join("benchmarks")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).collect();
    assert_eq!(saved.len(), 2, "a JSON and a CSV: {saved:?}");
    assert!(app.log.iter().any(|line| line.starts_with("Benchmark finished: ")), "{:?}", app.log);
    wait_for_job(&mut app);
}

#[test]
fn a_flight_with_frames_ahead_shows_its_frames_in_order_and_arrives() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flight_rendering = super::flight::FlightRendering::FULL_RESOLUTION;
    app.frames_ahead = 2;
    app.go_to_point_of_interest(0);
    let duration = app.flight.as_ref().unwrap().flight.duration();
    let clocks = fly_until_arrived(&mut app);
    assert!(clocks.len() >= 3, "frames on the way: {clocks:?}");
    assert!(clocks.windows(2).all(|pair| pair[1].0 > pair[0].0), "in order: {clocks:?}");
    assert_eq!(clocks.last().unwrap().0, duration, "the last frame is at the destination");
    assert!(app.flight_queue.is_empty());
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18 && app.buffer_is_complete);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_video_of_the_last_flight_has_a_frame_for_every_step_of_its_frame_rate() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (48, 32, 1, 24);
    (app.pace, app.video_pngs) = (super::flight::Pace::Fast, true); //(Several frames at once: still every frame, in order)
    app.go_to_point_of_interest(0);
    app.skip_flight();
    let duration = super::flight::Flight::new(app.last_flight.as_ref().unwrap().from, app.last_flight.as_ref().unwrap().to, super::flight::Pace::Fast).duration();
    app.start_video();
    let ctx = egui::Context::default();
    let started = std::time::Instant::now();
    while app.video_job.is_some() {
        assert!(started.elapsed() < Duration::from_secs(60), "the video did not finish");
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(5));
    }
    let frames = (duration.as_secs_f64() * 24.0).ceil() as usize + 1;
    let message = app.log.iter().rev().find(|line| line.starts_with("Video of the flight")).unwrap();
    assert!(message.starts_with(&format!("Video of the flight: {frames} of {frames} frames at 24 FPS")), "{message}");
    if std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|output| output.status.success()) {
        assert!(message.contains(".mp4"), "the default format makes an .mp4: {message}");
    }
    let written = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).find(|entry| entry.path().is_dir()).unwrap().path();
    assert_eq!(std::fs::read_dir(&written).unwrap().count(), frames, "one PNG per frame");
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_straight_flight_at_the_constant_pace_arrives_and_says_so_in_the_log() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    (app.pace, app.route, app.constant_speed) = (super::flight::Pace::Constant, super::flight::Route::Straight, 20.0);
    app.go_to_point_of_interest(0);
    assert!(app.log.last().unwrap().contains("(Constant, straight route,"), "{:?}", app.log.last());
    let start = app.c.get_scale();
    let mut widest = start;
    while app.flight.is_some() {
        app.fly_at(std::time::Instant::now());
        app.poll_job();
        widest = widest.max(app.c.get_scale());
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(widest <= start * (1.0 + 1e-9), "a straight flight never zooms out: {widest} from {start}");
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_finished_view_gets_its_neighborhood_which_a_small_move_keeps_and_a_zoom_renders_again() {
    let ctx = egui::Context::default();
    let mut app = manual_app(&Config::sized(64, 48, 50, 4));
    let wait_for_neighborhood = |app: &mut MandelbrotApp| {
        let started = std::time::Instant::now();
        app.poll_neighborhood(&ctx);
        while app.neighborhood_job.is_some() {
            assert!(started.elapsed() < Duration::from_secs(30), "the neighborhood was not rendered");
            std::thread::sleep(Duration::from_millis(2));
            app.poll_neighborhood(&ctx);
        }
    };
    wait_for_neighborhood(&mut app);
    let neighborhood = app.neighborhood.as_ref().expect("rendered once the view was finished");
    //Three views across and down, around the view, at half its resolution
    assert!((neighborhood.plane().get_scale() / app.c.get_scale() - 3.0).abs() < 1e-9);
    assert_eq!(neighborhood.texture_size(), [96, 72]);
    assert!(super::neighborhood::still_around(neighborhood.around(), &app.c, 100));
    //A small move keeps it: no new job
    let mut moved = app.c.clone();
    rendering::move_complex_plane_by_pixels(&mut moved, 5, 10);
    assert!(super::neighborhood::still_around(neighborhood.around(), &moved, 100));
    //A zoom does not; a render stops the job at once
    let mut zoomed = app.c.clone();
    zoomed.scale(0.5);
    assert!(!super::neighborhood::still_around(neighborhood.around(), &zoomed, 100));
    app.c = zoomed;
    app.render("Zoomed");
    assert!(app.neighborhood_job.is_none(), "no neighborhood while a render runs");
    wait_for_job(&mut app);
    wait_for_neighborhood(&mut app);
    assert!(super::neighborhood::still_around(app.neighborhood.as_ref().unwrap().around(), &app.c, 100), "rendered again for the zoomed view");
    //A larger size: rendered again, 2.5 views to every side (6 views across) at half the resolution
    app.neighborhood_margin = 250;
    wait_for_neighborhood(&mut app);
    let neighborhood = app.neighborhood.as_ref().unwrap();
    assert!((neighborhood.plane().get_scale() / app.c.get_scale() - 6.0).abs() < 1e-9);
    assert_eq!(neighborhood.texture_size(), [192, 144]);
}

#[test]
fn a_large_neighborhood_keeps_its_texture_within_the_largest_side_and_a_small_one_is_kept_for_a_shorter_move() {
    use super::neighborhood::{neighborhood_size, still_around, MAX_SIDE};
    assert_eq!(neighborhood_size(3840, 2160, 100), (5760, 3240), "4K: three views of half the pixels");
    let (width, height) = neighborhood_size(3840, 2160, 1000);
    assert_eq!(width, MAX_SIDE, "21 views across would be 40320 pixels");
    assert!((width as f64 / height as f64 - 3840.0 / 2160.0).abs() < 0.01);
    //A margin of 20%: a move of 15% of the view is too far (half the margin is 10%)
    let mut view = ComplexPlane::new(100, 100);
    view.set_view(&crate::complex_plane::View::new(-0.75, 0.0, 1.0));
    let mut moved = view.clone();
    rendering::move_complex_plane_by_pixels(&mut moved, 15, 0);
    assert!(still_around(&view, &moved, 100) && !still_around(&view, &moved, 20));
}

#[test]
fn another_color_cycle_keeps_the_limit_and_shows_the_recolored_image_at_once() {
    let config = Config::sized(64, 48, 50, 8);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    assert!(app.auto_iterations);
    app.fast_gpu = false; //Compared with the CPU bit for bit below: the samples of the fast math are not continued (see `SampleKey`)
    app.render("Exact");
    wait_for_job(&mut app);
    let limit = app.m.max_iterations;
    app.m.palette = ColorCycle { divide: 2.0, offset: 50.0, stretch: false };
    app.render_moved("Set the colors");
    assert_eq!(app.m.max_iterations, limit, "the limit is kept: the samples are only recolored");
    wait_for_job(&mut app);
    //The image is the image of a new render of these colors
    let (width, height) = (app.p.pixel_plane.width, app.p.pixel_plane.height);
    let mut fresh = Vec::new();
    rendering::Render::new(&app.c, &app.m, width, height, app.coloring_function).supersampling(app.supersampling_amount).adaptive(app.adaptive_supersampling.then(rendering::AdaptiveSampling::default)).run(|colors, _| fresh = colors.to_vec());
    let differ = app.p.colors.iter().zip(&fresh).filter(|(a, b)| (a.red, a.green, a.blue) != (b.red, b.green, b.blue)).count();
    assert_eq!(differ, 0, "of {} pixels; log {:?}", fresh.len(), &app.log[app.log.len().saturating_sub(3)..]);
}

#[test]
fn the_quality_presets_go_from_fastest_to_best_with_a_blurred_one_after_1x_4x_8x_and_16x() {
    use super::blur::Quality;
    let samples: Vec<u8> = Quality::PRESETS.iter().map(|preset| preset.samples).collect();
    assert_eq!(samples, [1, 1, 2, 4, 4, 8, 8, 16, 16, 32, 64]);
    let blurred: Vec<u8> = Quality::PRESETS.iter().filter(|preset| preset.blurred).map(|preset| preset.samples).collect();
    assert_eq!(blurred, [1, 4, 8, 16], "issue #186");
    assert_eq!((Quality::PRESETS[1].name(), Quality::PRESETS[4].name()), (String::from("1x + blur (smooth, fast)"), String::from("4x + blur (close to 8x)")));
    assert_eq!(Quality { samples: 12, blurred: false }.name(), "12x", "a quality that is not a preset (from the configuration) is named too");
}

#[test]
fn a_saved_image_of_the_blur_preset_is_blurred_like_the_screen() {
    use crate::coloring::TrueColor;
    let (width, height) = (9, 7);
    //A flat image stays flat; one bright pixel spreads evenly to its neighbors and keeps its brightness on the whole
    let flat = vec![TrueColor::new(40, 80, 120); width * height];
    assert!(super::blur::blurred(&flat, width, height, 0.8).iter().all(|c| (c.red, c.green, c.blue) == (40, 80, 120)));
    let mut dot = vec![TrueColor::new(0, 0, 0); width * height];
    dot[3 * width + 4] = TrueColor::new(255, 255, 255);
    let blurred = super::blur::blurred(&dot, width, height, 0.8);
    let at = |x: usize, y: usize| blurred[y * width + x].red;
    assert!(at(4, 3) < 255 && at(4, 3) > at(3, 3) && at(3, 3) > 0, "the center keeps the most");
    assert_eq!((at(3, 3), at(4, 2)), (at(5, 3), at(4, 4)), "evenly to every side");
    let total: u32 = blurred.iter().map(|c| u32::from(c.red)).sum();
    assert!((240..=270).contains(&total), "the brightness is kept: {total}");
}

#[test]
fn the_color_channels_have_a_drop_down_per_screen_channel_that_shows_its_source() {
    let mut app = manual_app(&Config::sized(64, 48, 50, 4));
    app.p.color_channel_mapping = ColorChannelMapping::new("BGG").unwrap();
    let ctx = egui::Context::default();
    let output = ctx.run_ui(egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 2000.0))), ..Default::default() }, |ui| app.side_panel_ui(ui));
    let texts: Vec<String> = output.shapes.iter().filter_map(|shape| match &shape.shape {
        egui::epaint::Shape::Text(text) => Some(text.galley.text().to_string()),
        _ => None,
    }).collect();
    output.drop_without_applying_deltas();
    let at = texts.iter().position(|text| text == "R:").expect("the drop-down of the red channel");
    assert_eq!(texts[at..at + 6], ["R:", "Blue", "G:", "Green", "B:", "Green"], "{:?}", &texts[at..]);
}
#[test]
fn a_flight_is_saved_loaded_rendered_and_flown_again_exactly() {
    //Issue #180: the last flight saved with a name and the length of Fit to length, chosen again from the list after other settings changed
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flights_folder = folder.join("flights");
    (app.pace, app.route, app.dive_zoom, app.flight_speed) = (super::flight::Pace::Fast, super::flight::Route::Dive, 0.3, 150);
    app.m.palette = crate::mandelbrot_set::ColorCycle { divide: 2.0, offset: 10.0, stretch: false };
    app.go_to_point_of_interest(0);
    app.skip_flight();
    wait_for_job(&mut app);
    let flown = app.last_flight.clone().unwrap();
    assert_eq!(app.flight_name_input, "Test Place", "the name of the place, to start with");
    app.flight_name_input = String::from("My dive");
    (app.video_fit_length, app.video_length) = (true, [String::new(), String::from("1"), String::from("5.5")]);
    app.save_last_flight();
    assert_eq!(app.saved_flights.len(), 1);
    assert!(app.log.last().unwrap().contains("Saved the flight My dive"), "{:?}", app.log.last());
    //Other settings, then the saved flight again
    (app.pace, app.route, app.dive_zoom, app.flight_speed, app.video_fit_length) = (super::flight::Pace::Cinematic, super::flight::Route::Shortest, 1.0, 100, false);
    app.video_length = [String::new(), String::new(), String::new()];
    app.last_flight = None;
    app.saved_flights_loaded = false;
    app.reload_saved_flights();
    app.use_saved_flight(0);
    let loaded = app.last_flight.clone().unwrap();
    assert_eq!(loaded, super::video::VideoFlight { name: String::from("My dive"), ..flown }, "the same flight, exactly");
    assert_eq!((app.pace, app.route, app.dive_zoom, app.flight_speed), (super::flight::Pace::Fast, super::flight::Route::Dive, 0.3, 150), "the controls take its settings");
    assert_eq!((app.video_fit_length, app.target_length()), (true, Some(Duration::from_secs_f64(65.5))), "and its length");
    //Flown in the window: from its start, to its place, ending with its colors
    app.m.palette = crate::mandelbrot_set::ColorCycle { divide: 9.0, offset: 99.0, stretch: false };
    app.fly_saved_flight(0);
    assert_eq!((app.c.center().x, app.c.get_scale()), (flown.from.x, flown.from.scale), "starts where the flight starts");
    assert_eq!(app.m.palette, flown.palettes.0);
    app.skip_flight();
    assert_eq!((app.c.center().x, app.c.get_scale(), app.m.palette), (flown.to.x, flown.to.scale, flown.palettes.1), "ends at its place with its colors");
    assert_eq!(app.last_flight.as_ref(), Some(&loaded));
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn the_queue_renders_its_videos_with_their_own_settings_and_survives_a_restart() {
    //Issue #181
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    let settings = folder.join("config").join("settings.json");
    app.use_settings_file(Some(settings.clone()));
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_quality) = (48, 32, 1, super::video::VideoQuality::Lossless);
    app.pace = super::flight::Pace::Fast;
    app.go_to_point_of_interest(0);
    app.skip_flight();
    app.video_fps = 24;
    app.add_to_queue();
    app.video_fps = 12;
    app.add_to_queue();
    app.video_fps = 60; //Later changes do not change the queued videos
    let queue_file = folder.join("config").join("video_queue.json");
    assert!(queue_file.exists(), "the queue is saved");
    let mut next = manual_app(&Config::sized(64, 48, 50, 4));
    next.use_settings_file(Some(settings));
    assert_eq!(next.video_queue.iter().map(|spec| spec.fps).collect::<Vec<_>>(), [24, 12], "and read by the next run");
    assert!(!next.queue_running, "which waits for Start queue");
    //Run it
    app.queue_running = true;
    let ctx = egui::Context::default();
    let started = std::time::Instant::now();
    while app.queue_running || app.video_job.is_some() {
        assert!(started.elapsed() < Duration::from_secs(120), "the queue did not finish");
        app.poll_video(&ctx);
        app.run_queue();
        std::thread::sleep(Duration::from_millis(5));
    }
    let videos: Vec<&String> = app.log.iter().filter(|line| line.starts_with("Video of the flight")).collect();
    assert_eq!(videos.len(), 2, "{:?}", app.log);
    assert!(videos[0].contains("at 24 FPS") && videos[1].contains("at 12 FPS"), "in order, each with its own settings: {videos:?}");
    assert_eq!(app.log.last().map(String::as_str), Some("The queue is done"));
    assert!(!queue_file.exists(), "an empty queue leaves no file");
    //A cancelled video goes back to the front, and the queue stops
    app.add_to_queue();
    app.queue_running = true;
    app.run_queue();
    assert!(app.video_job.is_some() && app.video_queue.is_empty());
    app.video_job.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
    while app.video_job.is_some() {
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!((app.video_queue.len(), app.queue_running), (1, false));
    assert!(queue_file.exists());
    let _ = std::fs::remove_dir_all(folder);
}

/// Whether ffmpeg can be run: the tests that check video files need it, and say so when it is not there
fn ffmpeg_available() -> bool {
    let available = std::process::Command::new("ffmpeg").arg("-version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|status| status.success());
    if !available {
        eprintln!("ffmpeg is not installed: this test of video files is skipped");
    }
    available
}

/// The frames of the video file `path`, decoded to RGB bytes by ffmpeg
fn decoded_frames(path: &std::path::Path) -> Vec<u8> {
    let output = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).output().unwrap();
    assert!(output.status.success(), "ffmpeg decodes {}", path.display());
    output.stdout
}

/// A lossless 48 x 32 video app of the test place with chunks of 5 frames, the flight done, its videos in `folder`/saved
fn chunked_video_app(blend: bool) -> (MandelbrotApp, std::path::PathBuf) {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (48, 32, if blend { 1 } else { 16 }, 24);
    (app.video_quality, app.video_blend, app.video_chunk_frames, app.pace) = (super::video::VideoQuality::Lossless, blend, Some(5), super::flight::Pace::Fast);
    app.go_to_point_of_interest(0);
    app.skip_flight();
    wait_for_job(&mut app);
    (app, folder)
}

/// Waits for the video under way to end, and returns its last message
fn wait_for_video(app: &mut MandelbrotApp) -> String {
    let (ctx, started) = (egui::Context::default(), std::time::Instant::now());
    while app.video_job.is_some() {
        assert!(started.elapsed() < Duration::from_secs(120), "the video did not end");
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(1));
    }
    app.log.iter().rev().find(|line| line.starts_with("Video of the flight") || line.starts_with("Could not")).cloned().unwrap()
}

#[test]
fn a_paused_video_resumes_from_its_chunks_to_exactly_the_same_video() {
    if !ffmpeg_available() {
        return;
    }
    //Issue #182: lossless frames without blending, so the resumed video must be the uninterrupted one, byte for byte
    let (mut whole, whole_folder) = chunked_video_app(false);
    whole.start_video();
    let message = wait_for_video(&mut whole);
    let video = std::fs::read_dir(whole_folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mkv")).expect(&message);
    let expected = decoded_frames(&video);
    //The same video, paused once two chunks are done
    let (mut app, folder) = chunked_video_app(false);
    app.start_video();
    let saved = folder.join("saved");
    let parts = || std::fs::read_dir(&saved).ok().and_then(|entries| entries.filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "parts")));
    let started = std::time::Instant::now();
    while parts().and_then(|parts| super::chunks::read_manifest(&parts)).is_none_or(|manifest| manifest.chunks_done < 2) {
        assert!(started.elapsed() < Duration::from_secs(60) && app.video_job.is_some(), "two chunks done before the video ends");
        std::thread::sleep(Duration::from_millis(1));
    }
    app.video_job.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
    let paused = wait_for_video(&mut app);
    assert!(paused.contains("paused") && paused.contains("Resume continues it"), "{paused}");
    let parts = parts().expect("the chunks are kept");
    let manifest = super::chunks::read_manifest(&parts).unwrap();
    assert!(manifest.chunks_done >= 2 && manifest.chunks_done < manifest.chunks(), "{manifest:?}");
    //A new run of the program would find it in the list; Resume finishes it
    app.unfinished_videos = None;
    let name = super::chunks::unfinished(&saved).first().map(|(name, _)| name.clone()).expect("listed as unfinished");
    app.resume_video(&name);
    let resumed = wait_for_video(&mut app);
    assert!(resumed.contains("resumed at frame"), "{resumed}");
    assert!(!parts.exists(), "the chunks are joined and removed");
    assert_eq!(decoded_frames(&saved.join(format!("{name}.mkv"))), expected, "exactly the frames of the video that was not paused");
    let _ = (std::fs::remove_dir_all(folder), std::fs::remove_dir_all(whole_folder));
}

#[test]
fn a_chunked_mp4_joins_into_one_video_with_every_frame() {
    if !ffmpeg_available() {
        return;
    }
    //The video file most videos are: HEVC or H.264 in an .mp4 (the GPU's encoder needs at least 160 x 64), from chunks of 5 frames
    let (mut app, folder) = chunked_video_app(false);
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_quality) = (192, 108, 1, super::video::VideoQuality::VeryHigh);
    //(MANDELBROT_TEST_FORMAT="Most compatible" tests the H.264 of computers without NVIDIA's encoder, as CI has)
    if let Ok(format) = std::env::var("MANDELBROT_TEST_FORMAT") {
        app.video_playback = super::video::Playback::of_name(&format).unwrap();
    }
    app.start_video();
    let message = wait_for_video(&mut app);
    let total: usize = message.split(" of ").nth(2).and_then(|rest| rest.split(' ').next()).and_then(|total| total.parse().ok()).expect(&message);
    assert!(total > 10, "more than two chunks: {message}");
    let video = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mp4")).expect(&message);
    assert_eq!(decoded_frames(&video).len(), total * 192 * 108 * 3, "every frame, decodable: {message}");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_video_in_a_relative_folder_like_saved_joins_its_chunks() {
    if !ffmpeg_available() {
        return;
    }
    //The window's videos go to the relative folder saved/, and ffmpeg reads the paths of its list of chunks from the folder of the list: the chunks of every
    //video in the app were not found (found by Jort in the review of review-4: "Could not join the chunks of the video")
    let (mut app, folder) = chunked_video_app(false);
    let relative = std::path::PathBuf::from("target").join(format!("test-relative-videos-{}", std::process::id())).join("saved");
    let _ = std::fs::remove_dir_all(&relative);
    app.video_folder.clone_from(&relative);
    app.start_video();
    let message = wait_for_video(&mut app);
    assert!(message.starts_with("Video of the flight:") && !message.contains("(stopped"), "{message}");
    let total: usize = message.split(" of ").nth(2).and_then(|rest| rest.split(' ').next()).and_then(|total| total.parse().ok()).expect(&message);
    assert!(total > 10, "more than two chunks: {message}");
    let video = std::fs::read_dir(&relative).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mkv")).expect(&message);
    assert_eq!(decoded_frames(&video).len(), total * 48 * 32 * 3, "every frame: {message}");
    let _ = (std::fs::remove_dir_all(relative.parent().unwrap()), std::fs::remove_dir_all(folder));
}

#[test]
fn the_unfinished_videos_are_listed_newest_first_and_a_rendered_one_offers_to_join() {
    //Found by Jort in the review of review-4: he paused a video and pressed Resume, which did nothing. The list was sorted by name, oldest first, so the top row
    //was an older video of the same place, all of whose frames were rendered and whose join had failed (the join bug): Resume tried the join again and failed
    let (mut app, folder) = chunked_video_app(false);
    let saved = folder.join("saved");
    let spec = app.video_spec().unwrap();
    for (name, done) in [("video-2026-11-01_20-24-13-place", 2), ("video-2026-11-01_20-43-18-place", 0)] {
        let parts = super::chunks::parts_folder(&saved, name);
        std::fs::create_dir_all(&parts).unwrap();
        let mut manifest = super::chunks::Manifest::new(spec.clone(), 10, 5, false);
        manifest.chunks_done = done;
        super::chunks::write_manifest(&parts, &manifest).unwrap();
    }
    let names: Vec<String> = super::chunks::unfinished(&saved).into_iter().map(|(name, _)| name).collect();
    assert_eq!(names, ["video-2026-11-01_20-43-18-place", "video-2026-11-01_20-24-13-place"], "the newest first");
    app.unfinished_videos = None;
    let texts = controls_texts(&mut app).join("\n");
    assert!(texts.contains("Unfinished: video-2026-11-01_20-43-18-place (0 of 10 frames)") && texts.contains("Rendered, not joined: video-2026-11-01_20-24-13-place (all 10 frames)") && texts.contains("Join"), "{texts}");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_video_paused_before_its_first_chunk_says_that_nothing_is_kept_yet() {
    if !ffmpeg_available() {
        return;
    }
    let (mut app, folder) = chunked_video_app(false);
    app.video_chunk_frames = Some(100_000);
    app.start_video();
    app.video_job.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
    let message = wait_for_video(&mut app);
    assert!(message.contains("paused before its first chunk") && message.contains("no frames are kept yet"), "{message}");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_resumed_blended_video_warms_up_its_blend_and_has_every_frame() {
    if !ffmpeg_available() {
        return;
    }
    let (mut whole, whole_folder) = chunked_video_app(true);
    whole.start_video();
    let message = wait_for_video(&mut whole);
    let frames = |message: &str| message.split(" of ").nth(2).and_then(|rest| rest.split(' ').next()).and_then(|total| total.parse::<usize>().ok()).unwrap();
    let total = frames(&message);
    //Paused after the first chunk (by writing its manifest as if the program had stopped there), then resumed
    let (mut app, folder) = chunked_video_app(true);
    let spec = super::video::VideoSpec { output: Some(String::from("video-resumed")), ..app.video_spec().unwrap() };
    let parts = super::chunks::parts_folder(&folder.join("saved"), "video-resumed");
    std::fs::create_dir_all(&parts).unwrap();
    let mut manifest = super::chunks::Manifest::new(spec, total, 5, false);
    std::fs::copy(std::fs::read_dir(whole_folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mkv")).unwrap(), parts.join("whole.mkv")).unwrap();
    //(Its first chunk: the first 5 frames of the whole video)
    let cut = std::process::Command::new("ffmpeg").args(["-v", "error", "-y", "-i"]).arg(parts.join("whole.mkv")).args(["-frames:v", "5", "-c", "copy"]).arg(super::chunks::chunk_path(&parts, 0, "mkv")).status().unwrap();
    assert!(cut.success());
    std::fs::remove_file(parts.join("whole.mkv")).unwrap();
    manifest.chunks_done = 1;
    super::chunks::write_manifest(&parts, &manifest).unwrap();
    app.resume_video("video-resumed");
    let resumed = wait_for_video(&mut app);
    assert!(resumed.contains(&format!("{total} of {total} frames")) && resumed.contains("resumed at frame 6"), "{resumed}");
    let (expected, got) = (decoded_frames(&std::fs::read_dir(whole_folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mkv")).unwrap()), decoded_frames(&folder.join("saved").join("video-resumed.mkv")));
    assert_eq!(got.len(), expected.len(), "every frame");
    //The warm-up blends the frames before the chunk again: the frames after it are those of the whole video, nearly (they are blended into 8-bit colors)
    let mean_error = got.iter().zip(&expected).map(|(a, b)| f64::from(a.abs_diff(*b))).sum::<f64>() / got.len() as f64;
    assert!(mean_error < 1.0, "the blend continued as if the video had not stopped: mean error {mean_error}");
    let _ = (std::fs::remove_dir_all(folder), std::fs::remove_dir_all(whole_folder));
}

#[test]
fn a_copied_view_is_pasted_to_the_same_view_and_colors() {
    //Issue #184
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.pace = super::flight::Pace::NoFlight;
    app.go_to_point_of_interest(0);
    (app.coloring_index, app.coloring_function) = (0, super::COLORING_FUNCTIONS[0].1);
    app.m.palette = crate::mandelbrot_set::ColorCycle { divide: 0.113, offset: 278.0, stretch: false };
    let ctx = egui::Context::default();
    app.copy_view(&ctx);
    let text = app.log.last().unwrap().strip_prefix("Copied the view to the clipboard: ").unwrap().to_string();
    assert!(!text.contains('\n'), "one line: {text}");
    let (center, scale, limit) = (app.c.center(), app.c.get_scale(), app.m.max_iterations);
    //Somewhere else, with other colors
    app.c.set_view(&crate::complex_plane::View::new(-0.5, 0.0, 1.0));
    (app.coloring_index, app.m.palette) = (1, crate::mandelbrot_set::ColorCycle { divide: 5.0, offset: 0.0, stretch: false });
    app.go_to_shared(&text);
    assert_eq!((app.c.center().x, app.c.center().y, app.c.get_scale(), app.m.max_iterations), (center.x, center.y, scale, limit), "the same view");
    assert_eq!((app.coloring_index, app.m.palette), (0, crate::mandelbrot_set::ColorCycle { divide: 0.113, offset: 278.0, stretch: false }), "and colors");
    app.go_to_shared("{ not a view");
    assert!(app.log.last().unwrap().starts_with("Could not go to the view"), "{:?}", app.log.last());
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn every_png_frame_of_a_video_has_its_view() {
    //Issue #183: the first frame is the start of the flight, the last its place
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps, app.video_pngs) = (48, 32, 1, 24, true);
    app.pace = super::flight::Pace::Fast;
    app.go_to_point_of_interest(0);
    app.skip_flight();
    let flight = app.last_flight.clone().unwrap();
    app.start_video();
    let message = wait_for_video(&mut app);
    let frames_folder = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.is_dir()).expect(&message);
    let mut frames: Vec<std::path::PathBuf> = std::fs::read_dir(&frames_folder).unwrap().filter_map(Result::ok).map(|entry| entry.path()).collect();
    frames.sort();
    let (first, last) = (crate::image_metadata::view_of_png(&frames[0]).unwrap(), crate::image_metadata::view_of_png(frames.last().unwrap()).unwrap());
    assert_eq!((first.location.real, first.location.scale), (flight.from.x, flight.from.scale), "{}", first.description);
    assert!((last.location.scale / flight.to.scale - 1.0).abs() < 1e-9 && first.description.starts_with("Frame 1 of "), "{} / {}", last.location.scale, first.description);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn the_depth_control_changes_only_the_zoom_and_renders_like_a_gesture_while_dragged() {
    //Issue #193
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.pace = super::flight::Pace::NoFlight;
    app.go_to_point_of_interest(0);
    wait_for_job(&mut app);
    let center = app.c.center();
    for zoom in [0.5, 1e3, 4.2e7, 1.0, 3.3, 77.0, 1e5] {
        assert!(app.set_depth(zoom));
        let kept = app.current_point(String::new(), String::new()).location;
        assert_eq!((kept.real.to_bits(), kept.imaginary.to_bits()), (center.x.to_bits(), center.y.to_bits()), "the center never moves (zoom {zoom})");
        assert!((app.c.center().x - center.x).abs() < 1e-15 && (app.c.center().y - center.y).abs() < 1e-15, "and the plane is there");
        assert!((app.c.get_scale() * zoom - 1.0).abs() < 1e-12);
    }
    assert!(app.set_depth(1.0));
    assert!(!app.set_depth(1e15), "deeper than the 64-bit floats can show: refused");
    assert!((app.c.get_scale() - 1.0).abs() < 1e-12, "and the view stays");
    //Dragged: frames of a gesture (one sample per pixel); released: the full quality
    (app.depth_dragging, app.gesture_active) = (true, true);
    assert!(app.set_depth(1e4));
    app.render("Depth");
    assert_eq!(app.progress.as_ref().map(|progress| progress.total), Some(1), "a frame of a gesture");
    (app.depth_dragging, app.gesture_active) = (false, false);
    app.settle_gesture(&egui::Context::default());
    assert_eq!(app.progress.as_ref().map(|progress| progress.total), Some(app.supersampling_amount), "the full quality when it stops");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_dive_made_of_keyframes_flies_renders_and_saves_with_the_colors_of_each_keyframe() {
    //Issue #194 with #193: keyframes at three depths of one center, each with its own divide, then the sequence flown, rendered and saved
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.flights_folder = folder.join("flights");
    app.pace = super::flight::Pace::NoFlight;
    app.go_to_point_of_interest(0);
    let center = app.current_point(String::new(), String::new()).location;
    for (zoom, divide) in [(1.0, 1.0), (300.0, 0.25), (1.0 / 6.4e-5, 4.0)] {
        assert!(app.set_depth(zoom));
        app.m.palette.divide = divide;
        app.add_keyframe();
    }
    app.sequence_seconds = vec![String::from("0.5"), String::from("1")];
    assert!(app.sequence().is_complete());
    assert!(app.sequence_keyframes.iter().all(|keyframe| keyframe.real.to_bits() == center.real.to_bits() && keyframe.imaginary.to_bits() == center.imaginary.to_bits()), "one center, exactly");
    //Flown in the window: from the first keyframe, to the last, with its colors
    app.flight_name_input = String::from("Keyframe dive");
    app.fly_sequence();
    assert!(app.flight.is_some());
    assert_eq!(app.m.palette.divide, 1.0, "starts with the first keyframe's colors");
    app.skip_flight();
    assert_eq!((app.c.get_scale(), app.m.palette.divide), (6.4e-5, 4.0), "ends at the last keyframe with its colors");
    //Rendered: 1.5 s at 24 FPS, every frame on the same center, the colors of the keyframes on the way
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps, app.video_pngs) = (48, 32, 1, 24, true);
    app.start_video();
    let message = wait_for_video(&mut app);
    assert!(message.starts_with("Video of the flight: 37 of 37 frames"), "{message}");
    let frames_folder = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.is_dir()).unwrap();
    let mut frames: Vec<std::path::PathBuf> = std::fs::read_dir(&frames_folder).unwrap().filter_map(Result::ok).map(|entry| entry.path()).collect();
    frames.sort();
    let views: Vec<crate::points_of_interest::PointOfInterest> = frames.iter().map(|frame| crate::image_metadata::view_of_png(frame).unwrap()).collect();
    assert!(views.iter().all(|view| view.location.real.to_bits() == center.real.to_bits()), "never sideways");
    let divide = |view: &crate::points_of_interest::PointOfInterest| view.appearance.as_ref().and_then(|appearance| appearance.iteration_divide).unwrap();
    assert_eq!((divide(&views[0]), divide(&views[12]), divide(&views[36])), (1.0, 0.25, 4.0), "the divide of each keyframe at its time (frame 12 is 0.5 s)");
    //Saved and read back with its keyframes
    app.save_last_flight();
    app.reload_saved_flights();
    assert_eq!(app.saved_flights[0].flight.sequence, Some(app.sequence()));
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_video_that_follows_music_lasts_as_long_as_the_track_slows_down_in_its_quiet_part_and_has_its_sound() {
    if !ffmpeg_available() {
        return;
    }
    //Issue #185: a track of 2 s of a quiet tone and 2 s of loud noise
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    std::fs::create_dir_all(&folder).unwrap();
    let track = folder.join("track.wav");
    let made = std::process::Command::new("ffmpeg").args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-f", "lavfi", "-i", "anoisesrc=d=2:a=0.8", "-filter_complex", "[0]volume=0.02[quiet];[quiet][1]concat=n=2:v=0:a=1"]).arg(&track).status().unwrap();
    assert!(made.success());
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps, app.video_pngs) = (48, 32, 1, 24, true);
    app.pace = super::flight::Pace::Fast;
    app.go_to_point_of_interest(0);
    app.skip_flight();
    app.use_music_file(&track);
    app.music_strength = 1.0;
    assert_eq!(app.followed_music_length().map(|length| length.as_secs_f64().round()), Some(4.0));
    app.start_video();
    let message = wait_for_video(&mut app);
    assert!(message.starts_with("Video of the flight: 97 of 97 frames") && message.contains("with the music"), "4 s at 24 FPS: {message}");
    let saved = folder.join("saved");
    let video = std::fs::read_dir(&saved).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mp4")).unwrap();
    let streams = std::process::Command::new("ffprobe").args(["-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0"]).arg(&video).output().unwrap();
    assert!(String::from_utf8_lossy(&streams.stdout).contains("audio"), "the track is the sound of the video");
    //Halfway through the track (frame 49, at 2 s) the flight is not yet halfway: it went slowly through the quiet part
    let frames_folder = std::fs::read_dir(&saved).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.is_dir()).unwrap();
    let middle = crate::image_metadata::view_of_png(&frames_folder.join("frame-00049.png")).unwrap();
    let path = app.last_flight.as_ref().unwrap().path();
    assert!(middle.location.scale > path.camera_at(path.duration() / 2).scale * 1.5, "still further out than the middle of the flight: {}", middle.location.scale);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_sequence_fitted_to_music_lands_its_keyframes_on_the_swells_and_its_video_lasts_as_long_as_the_track() {
    //Issue #185: a 12 s track, quiet with a loud burst from 5 to 7 s, and a sequence of 3 keyframes
    if !ffmpeg_available() {
        return;
    }
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    std::fs::create_dir_all(&folder).unwrap();
    let track = folder.join("swell.wav");
    let made = std::process::Command::new("ffmpeg").args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=12", "-af", "volume='if(between(t,5,7),1,0.03)':eval=frame"]).arg(&track).status().unwrap();
    assert!(made.success());
    app.pace = super::flight::Pace::NoFlight;
    app.go_to_point_of_interest(0);
    for zoom in [1.0, 100.0, 1.0 / 6.4e-5] {
        assert!(app.set_depth(zoom));
        app.add_keyframe();
    }
    app.music_path = track.display().to_string();
    app.fit_sequence_to_music();
    let seconds: Vec<f64> = app.sequence().seconds;
    assert!((seconds[0] - 6.0).abs() < 1.0, "the middle keyframe lands on the swell around 6 s: {seconds:?}");
    assert!((seconds.iter().sum::<f64>() - 12.0).abs() < 0.2, "the last at the end of the track: {seconds:?}");
    assert!(app.follow_music && app.music_strength == 0.0);
    //Its video: as long as the track, with its sound
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (48, 32, 1, 10);
    app.fly_sequence();
    app.skip_flight();
    app.start_video();
    let message = wait_for_video(&mut app);
    assert!(message.starts_with("Video of the flight: 121 of 121 frames") && message.contains("with the music"), "12 s at 10 FPS: {message}");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_queued_video_that_cannot_start_stays_at_the_front_and_a_finished_one_is_not_rendered_again() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_quality) = (48, 32, 1, super::video::VideoQuality::Lossless);
    app.pace = super::flight::Pace::Fast;
    app.go_to_point_of_interest(0);
    app.skip_flight();
    //Its music was moved away: it cannot start, so it waits at the front and the queue stops
    (app.follow_music, app.music_path) = (true, folder.join("gone.wav").display().to_string());
    app.add_to_queue();
    app.follow_music = false;
    app.add_to_queue();
    app.queue_running = true;
    app.run_queue();
    assert_eq!((app.video_queue.len(), app.queue_running, app.queue_current.is_none(), app.video_job.is_none()), (2, false, true, true), "{:?}", app.log.last());
    assert!(app.video_queue[0].music.is_some(), "the one that could not start is first");
    //A video whose file is there already (finished while paused, or resumed by hand) is not rendered again
    app.video_queue.remove(0);
    std::fs::create_dir_all(folder.join("saved")).unwrap();
    app.video_queue[0].output = Some(String::from("video-done"));
    std::fs::write(folder.join("saved").join("video-done.mkv"), b"finished").unwrap();
    app.queue_running = true;
    app.run_queue();
    assert!(app.video_job.is_none() && app.video_queue.is_empty(), "skipped: {:?}", app.log.last());
    assert!(app.log.last().unwrap().contains("finished already"));
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_sequence_flown_in_the_window_has_the_limits_of_its_keyframes_also_without_auto_iterations() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    (app.pace, app.auto_iterations, app.flight_rendering) = (super::flight::Pace::NoFlight, false, super::flight::FlightRendering::BestEffort);
    app.go_to_point_of_interest(0);
    for (zoom, limit) in [(1.0, 100), (1.0 / 6.4e-5, 10_000)] {
        assert!(app.set_depth(zoom));
        app.m.max_iterations = limit;
        app.add_keyframe();
    }
    app.sequence_seconds = vec![String::from("2")];
    app.fly_sequence();
    let start = app.flight.as_ref().unwrap().last_frame;
    app.fly_at(start + Duration::from_secs(1));
    assert_eq!(app.m.max_iterations, 1000, "halfway between 100 and 10,000 on a logarithmic scale, as in its video");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_video_that_fits_a_length_lasts_exactly_that_long() {
    //Issue #179: 3 seconds at 24 FPS are 72 frame intervals, with the frames at both ends 73 frames, whatever the flight takes
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (48, 32, 1, 24);
    (app.video_quality, app.video_pngs, app.pace) = (super::video::VideoQuality::Lossless, false, super::flight::Pace::Fast);
    app.go_to_point_of_interest(0);
    app.skip_flight();
    let duration = super::flight::Flight::new(app.last_flight.as_ref().unwrap().from, app.last_flight.as_ref().unwrap().to, super::flight::Pace::Fast).duration();
    app.video_fit_length = true;
    app.video_length = [String::new(), String::new(), String::new()];
    assert!(app.video_speed_for(duration).is_err(), "no length typed");
    app.video_length = [String::from("2"), String::new(), String::new()];
    assert!(app.video_speed_for(duration).unwrap_err().contains("outside"), "two hours would need less than 1% speed");
    app.video_length = [String::from("0"), String::from("0"), String::from("3")];
    let speed = app.video_speed_for(duration).unwrap();
    assert!((speed - duration.as_secs_f64() * 100.0 / 3.0).abs() < 1e-9);
    app.start_video();
    let ctx = egui::Context::default();
    let started = std::time::Instant::now();
    while app.video_job.is_some() {
        assert!(started.elapsed() < Duration::from_secs(60), "the video did not finish");
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(5));
    }
    let message = app.log.iter().rev().find(|line| line.starts_with("Video of the flight")).unwrap();
    assert!(message.starts_with(&format!("Video of the flight: 73 of 73 frames at 24 FPS, {} speed", super::video::speed_text(speed))), "{message}");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_half_speed_lossless_video_without_png_frames_has_twice_the_frames_in_one_file() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.video_folder = folder.join("saved");
    app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
    (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (48, 32, 1, 24);
    (app.video_speed, app.video_quality, app.video_pngs) = (50, super::video::VideoQuality::Lossless, false);
    app.pace = super::flight::Pace::Fast;
    app.go_to_point_of_interest(0);
    app.skip_flight();
    let duration = super::flight::Flight::new(app.last_flight.as_ref().unwrap().from, app.last_flight.as_ref().unwrap().to, super::flight::Pace::Fast).duration();
    app.start_video();
    let ctx = egui::Context::default();
    let started = std::time::Instant::now();
    while app.video_job.is_some() {
        assert!(started.elapsed() < Duration::from_secs(60), "the video did not finish");
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(5));
    }
    let frames = (duration.as_secs_f64() * 2.0 * 24.0).ceil() as usize + 1;
    let message = app.log.iter().rev().find(|line| line.starts_with("Video of the flight")).unwrap();
    assert!(message.starts_with(&format!("Video of the flight: {frames} of {frames} frames at 24 FPS, 50% speed")), "{message}");
    let entries: Vec<std::path::PathBuf> = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).collect();
    assert!(entries.iter().filter(|path| path.is_dir()).all(|path| std::fs::read_dir(path).unwrap().count() == 0), "no PNG frames");
    if std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|output| output.status.success()) {
        let video = entries.iter().find(|path| path.extension().is_some_and(|extension| extension == "mkv")).expect("the .mkv of the video");
        assert!(std::fs::metadata(video).unwrap().len() > 0);
    }
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn cycling_colors_moves_the_offset_with_the_clock_and_ends_with_a_full_render() {
    let ctx = egui::Context::default();
    let mut app = manual_app(&Config::sized(64, 48, 50, 4));
    app.m.palette = crate::mandelbrot_set::ColorCycle { divide: 1.0, offset: 0.0, stretch: false };
    app.render("Colors");
    wait_for_job(&mut app);
    let log_length = app.log.len();
    (app.cycle_rate, app.color_cycling) = (100.0, false);
    app.toggle_color_cycle();
    let started = std::time::Instant::now();
    let mut steps = 0;
    //(Until enough steps, at least 300 ms: a sleep of 5 ms takes about 15 on Windows)
    while (started.elapsed() < Duration::from_millis(300) || steps < 8) && started.elapsed() < Duration::from_secs(3) {
        let offset = app.m.palette.offset;
        app.cycle_colors(&ctx, std::time::Instant::now());
        steps += usize::from((app.m.palette.offset - offset).abs() > 1e-9);
        app.poll_job();
        std::thread::sleep(Duration::from_millis(5));
    }
    let offset = app.m.palette.offset;
    assert!(steps >= 5, "the colors moved in steps: {steps}");
    assert!((offset - 100.0 * started.elapsed().as_secs_f64()).abs() < 10.0 + 15.0 * started.elapsed().as_secs_f64(), "about 100 steps per second: {offset} after {:?}", started.elapsed());
    app.toggle_color_cycle();
    wait_for_job(&mut app);
    assert_eq!(app.m.palette.offset, offset, "stopping keeps the colors it reached");
    assert!(app.buffer_is_complete, "and renders the view to its full quality");
    assert!(app.log.len() <= log_length + 3, "the steps do not fill the log: {:?}", &app.log[log_length..]);
}

#[test]
fn cycling_colors_in_the_window_keeps_its_one_sample_steps_and_the_history() {
    //As the window runs every frame: the cycling step, then the settling of a gesture (which must not turn each step into a full render)
    let ctx = egui::Context::default();
    let mut app = manual_app(&Config::sized(64, 48, 50, 4));
    app.m.palette = crate::mandelbrot_set::ColorCycle { divide: 1.0, offset: 0.0, stretch: false };
    app.render("Colors");
    wait_for_job(&mut app);
    let history = app.history.len();
    (app.cycle_rate, app.color_cycling) = (100.0, false);
    app.toggle_color_cycle();
    let started = std::time::Instant::now();
    let mut steps = 0;
    //(Until enough steps, at least 300 ms: a sleep of 5 ms takes about 15 on Windows)
    while (started.elapsed() < Duration::from_millis(300) || steps < 8) && started.elapsed() < Duration::from_secs(3) {
        let offset = app.m.palette.offset;
        app.poll_job();
        app.cycle_colors(&ctx, std::time::Instant::now());
        app.settle_gesture(&ctx);
        steps += usize::from((app.m.palette.offset - offset).abs() > 1e-9);
        assert!(app.progress.as_ref().is_none_or(|progress| progress.total == 1), "no full-quality render between the steps");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(steps >= 5, "the colors moved in steps: {steps}");
    assert_eq!(app.history.len(), history, "the steps are not views of the history");
    app.toggle_color_cycle();
    wait_for_job(&mut app);
    assert!(app.buffer_is_complete, "stopping renders the view to its full quality");
}

/// Drives the flight under way as the window does, until `done` says so; false after `seconds` without that
fn drive_flight(app: &mut MandelbrotApp, seconds: u64, done: impl Fn(&MandelbrotApp) -> bool) -> bool {
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(seconds) {
        app.poll_job();
        app.fly_at(std::time::Instant::now());
        if done(app) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

#[test]
fn a_best_effort_flight_does_not_wait_for_a_render_from_before_it() {
    //Issue #192: the render of the view before the flight (e.g. a deep place, still choosing its limit) has shown nothing yet; the flight starts its own frames at once
    let (mut app, _folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    (app.flight_rendering, app.route) = (super::flight::FlightRendering::BestEffort, super::flight::Route::Dive);
    app.render("A view before the flight");
    assert!(app.is_busy() && !app.frame_accepted && !app.flight_frame_job);
    app.go_to_point_of_interest(0);
    app.fly_at(std::time::Instant::now());
    assert!(app.flight_frame_job, "the first frame of the flight renders, instead of waiting for the render from before");
}

#[test]
fn a_flight_that_shows_no_frame_for_long_says_what_it_waits_for_once() {
    let (mut app, _folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    app.go_to_point_of_interest(0);
    let flight = app.flight.as_mut().unwrap();
    flight.started -= Duration::from_secs(3);
    let waiting = |app: &MandelbrotApp| app.log.iter().filter(|line| line.contains("has shown no frame")).count();
    app.fly_at(std::time::Instant::now());
    app.fly_at(std::time::Instant::now());
    assert_eq!(waiting(&app), 1, "one line in the status log: {:?}", app.log.last());
}

#[test]
fn after_a_cancelled_dive_the_same_place_can_be_dived_to_again() {
    //Issue #192: a dive that is cancelled halfway, then the same place again, in every kind of flight rendering, with and without auto iterations
    for rendering in [super::flight::FlightRendering::BestEffort, super::flight::FlightRendering::FULL_RESOLUTION, super::flight::FlightRendering::Auto(60)] {
        for auto_iterations in [false, true] {
            let (mut app, _folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
            (app.flight_rendering, app.pace, app.route, app.auto_iterations) = (rendering, super::flight::Pace::Fast, super::flight::Route::Dive, auto_iterations);
            app.go_to_point_of_interest(0);
            assert!(app.flight.is_some(), "{rendering:?}, auto {auto_iterations}: the first dive starts");
            let halfway = |app: &MandelbrotApp| app.flight.as_ref().is_some_and(|flight| flight.elapsed.as_secs_f64() > 0.4 * flight.flight.duration().as_secs_f64());
            assert!(drive_flight(&mut app, 30, halfway), "{rendering:?}, auto {auto_iterations}: the first dive gets halfway");
            app.cancel_flight();
            assert!(app.flight.is_none());
            for _ in 0..3 {
                app.poll_job();
                std::thread::sleep(Duration::from_millis(5));
            }
            app.go_to_point_of_interest(0);
            assert!(app.flight.is_some(), "{rendering:?}, auto {auto_iterations}: the second dive starts");
            let moving = |app: &MandelbrotApp| app.flight.as_ref().is_none_or(|flight| !flight.elapsed.is_zero());
            assert!(drive_flight(&mut app, 30, moving), "{rendering:?}, auto {auto_iterations}: the second dive moves");
            assert!(drive_flight(&mut app, 60, |app| app.flight.is_none()), "{rendering:?}, auto {auto_iterations}: the second dive arrives");
            let place = &app.points[0].location;
            assert_eq!((app.c.center().x, app.c.get_scale()), (place.real, place.scale), "{rendering:?}, auto {auto_iterations}: at the place");
        }
    }
}

#[test]
fn a_flight_whose_colors_change_arrives_with_the_color_controls_on_the_screen() {
    //Found by Jort in the review of review-4 (#192): with Restore the saved appearance on, or a keyframe sequence whose keyframes have other colors, a flight
    //changes the colors a little on every frame. The sliders of Iteration divide and Offset rounded those in-between values to their steps on every frame
    //of the window (egui clamps and rounds an existing value by default), the controls took that for a change of the colors and rendered the view again,
    //which stopped the frame the flight was waiting for: the flight never got going while the Colors section was open
    const PLACE: &str = r#"{ "format_version": 1, "name": "Colored Place", "location": { "center": { "real": -0.7498100000000001, "imaginary": -0.020300000000000054 }, "scale": 0.00006400000000000002, "max_iterations": 321 },
        "appearance": { "iteration_divide": 7.5, "color_offset": 311.0 } }"#;
    for rendering in [super::flight::FlightRendering::FULL_RESOLUTION, super::flight::FlightRendering::BestEffort] {
        let (mut app, folder) = app_with_points(&[("colored-place.json", PLACE)]);
        (app.flight_rendering, app.pace, app.restore_appearance) = (rendering, super::flight::Pace::Fast, true);
        let before = app.m.palette;
        app.go_to_point_of_interest(0);
        assert!(app.flight.is_some(), "{rendering:?}: the flight starts");
        let ctx = egui::Context::default();
        let started = std::time::Instant::now();
        let mut in_between = false;
        while app.flight.is_some() {
            assert!(started.elapsed() < Duration::from_secs(60), "{rendering:?}: the flight arrives: {:?}", app.log.iter().rev().take(4).collect::<Vec<_>>());
            //The controls on the screen, as in the window, every frame (the sections are open in tests)
            ctx.run_ui(egui::RawInput::default(), |ui| app.controls_ui(ui)).drop_without_applying_deltas();
            app.poll_job();
            app.fly_at(std::time::Instant::now());
            in_between |= app.m.palette != before && app.m.palette.offset.fract() != 0.0;
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(in_between, "{rendering:?}: the colors went through in-between values");
        assert!(!app.log.iter().any(|line| line.starts_with("Set the colors to")), "{rendering:?}: the controls never took the flight's colors for a change: {:?}", app.log);
        assert_eq!((app.m.palette.divide, app.m.palette.offset), (7.5, 311.0), "{rendering:?}: the colors of the place");
        let _ = std::fs::remove_dir_all(folder);
    }
}

#[test]
fn open_image_is_a_button_next_to_copy_view() {
    //Asked by Jort in the review of review-4: dropping an image on the window did nothing (his window runs on Wayland, where winit 0.30 delivers no dropped
    //files), so Open image is the way to go to an image's view; it opens the image with `open_image`, which the tests of the image metadata cover
    let config = Config::sized(64, 48, 50, 32);
    let mut app = manual_app(&config);
    let texts = controls_texts(&mut app);
    let open = texts.iter().position(|text| text == "Open image...").expect("the button is there");
    assert_eq!(texts.get(open + 1).map(String::as_str), Some("Copy view"), "{texts:?}");
    //It starts in the home folder: an absolute folder that exists (the relative folder saved/ made the file chooser fail)
    if let Some(home) = super::sharing::home_folder() {
        assert!(home.is_absolute() && home.is_dir(), "{}", home.display());
    }
}

#[test]
fn a_turned_view_is_kept_by_places_and_flights_and_its_unit_only_changes_how_it_is_shown() {
    //Issue #217: the rotation control, places, flights
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    wait_for_job(&mut app);
    let center = app.c.center();
    app.c.set_rotation(std::f64::consts::FRAC_PI_2);
    assert_eq!((app.c.center().x, app.c.center().y), (center.x, center.y), "turned around the center");
    //The control shows the same angle in degrees and in radians
    let texts = controls_texts(&mut app).join("|");
    assert!(texts.contains("|90|°|rotation"), "{texts}");
    app.rotation_in_degrees = false;
    let texts = controls_texts(&mut app).join("|");
    assert!(texts.contains("|1.5708| rad|rotation"), "{texts}");
    assert!((app.c.rotation() - std::f64::consts::FRAC_PI_2).abs() < 1e-15, "switching the unit keeps the angle");
    //A place keeps the rotation, and its file reads back the same
    let point = app.current_point(String::from("Turned"), String::new());
    assert!((point.location.rotation - 90.0).abs() < 1e-9);
    let (read, _) = crate::points_of_interest::parse(&crate::points_of_interest::to_json(&point), "turned.json").unwrap();
    assert_eq!(read.location, point.location);
    assert!(!crate::points_of_interest::to_json(&crate::points_of_interest::PointOfInterest { location: crate::points_of_interest::Location { rotation: 0.0, ..point.location.clone() }, ..point.clone() }).contains("rotation"), "an unturned place is written as before");
    //A flight to an unturned place turns back the short way on the way, and arrives unturned
    app.pace = super::flight::Pace::Fast;
    app.go_to_point_of_interest(0);
    let flight = app.flight.as_ref().unwrap();
    let halfway = flight.flight.duration() / 2;
    let turning = flight.rotation_at(halfway);
    assert!(turning > 0.0 && turning < std::f64::consts::FRAC_PI_2, "on the way: {turning}");
    finish_flight(&mut app);
    assert_eq!(app.c.rotation(), 0.0, "the place is unturned");
    //And back to the turned place, which arrives turned (and its video turns with it)
    app.go_to_shared(&crate::points_of_interest::to_json(&point));
    finish_flight(&mut app);
    assert!((app.c.rotation() - std::f64::consts::FRAC_PI_2).abs() < 1e-12, "{}", app.c.rotation());
    let flight = app.last_flight.clone().unwrap();
    let path = flight.path();
    assert!((flight.rotation_at(&path, path.duration()) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert!(super::flight::blend_angles(3.0, -3.0, 0.5).abs() > 3.0, "the short way around, through π");
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn a_right_click_centers_a_turned_view_on_the_clicked_point() {
    let config = Config::sized(64, 48, 50, 32);
    let mut app = manual_app(&config);
    app.c.set_rotation(1.0);
    let clicked = app.displayed_plane().complex_from_pixel_plane(10.0, 7.0);
    app.handle_image_click(10.0, 7.0, true);
    let center = app.c.center();
    let pixel = app.c.increment_x;
    assert!((center.x - clicked.x).abs() <= pixel && (center.y - clicked.y).abs() <= pixel, "within a pixel of the clicked point: {center:?} {clicked:?}");
}

#[test]
fn a_right_click_does_not_move_a_locked_center() {
    //Found by Jort in the review of review-4 (#193): Lock center kept the center against dragging and the wheel, not against a right click
    let config = Config::sized(64, 48, 50, 32);
    let mut app = manual_app(&config);
    app.lock_center = true;
    let center = app.c.center();
    app.handle_image_click(5.0, 7.0, true);
    assert_eq!((app.c.center().x, app.c.center().y), (center.x, center.y), "the center stays");
    assert!(app.log.last().unwrap().starts_with("Lock center is on"), "{:?}", app.log.last());
    app.lock_center = false;
    app.handle_image_click(5.0, 7.0, true);
    assert_ne!(app.c.center().x, center.x, "without the lock, a right click centers the view on the point");
}

#[test]
fn the_controls_on_the_screen_never_change_a_view_or_colors_that_nobody_touched() {
    //A view zoomed out beyond the depth slider (×0.1) and colors between the steps of the sliders: drawing the controls leaves them as they are
    let config = Config::sized(64, 48, 50, 32);
    let mut app = manual_app(&config);
    app.c.scale(20.0);
    (app.m.palette.divide, app.m.palette.offset) = (1.23456, 77.7);
    let (view, palette, lines) = (app.c.get_view(), app.m.palette, app.log.len());
    let ctx = egui::Context::default();
    for _ in 0..3 {
        ctx.run_ui(egui::RawInput::default(), |ui| app.controls_ui(ui)).drop_without_applying_deltas();
    }
    assert_eq!((format!("{:?}", app.c.get_view()), app.m.palette), (format!("{view:?}"), palette));
    assert_eq!(app.log.len(), lines, "nothing was rendered: {:?}", &app.log[lines..]);
}

#[test]
fn flight_settings_take_effect_during_a_flight_and_a_new_speed_keeps_the_camera_where_it_is() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    (app.flight_rendering, app.pace) = (super::flight::FlightRendering::FULL_RESOLUTION, super::flight::Pace::Cinematic);
    app.go_to_point_of_interest(0);
    app.fly_at(std::time::Instant::now());
    let started = std::time::Instant::now();
    while app.flight.as_ref().is_some_and(|flight| flight.elapsed.is_zero()) {
        assert!(started.elapsed() < Duration::from_secs(30), "no frame");
        app.poll_job();
        app.fly_at(std::time::Instant::now());
        std::thread::sleep(Duration::from_millis(1));
    }
    //Blend frames on: the job starts again with it
    assert_eq!(app.flight_job_options, Some(app.flight_job_options()));
    app.blend_frames = true;
    app.fly_at(std::time::Instant::now());
    assert_eq!(app.flight_job_options, Some(app.flight_job_options()), "restarted with the new setting");
    //Twice as fast: half the time, at the same share of it
    let flight = app.flight.as_ref().unwrap();
    let (share, duration) = (flight.elapsed.as_secs_f64() / flight.flight.duration().as_secs_f64(), flight.flight.duration());
    app.flight_speed = 200;
    app.fly_at(std::time::Instant::now());
    let flight = app.flight.as_ref().unwrap();
    assert!((flight.flight.duration().as_secs_f64() * 2.0 - duration.as_secs_f64()).abs() < 1e-6, "half the time");
    assert!((flight.elapsed.as_secs_f64() / flight.flight.duration().as_secs_f64() - share).abs() < 1e-6, "the same share of it");
    let _ = fly_until_arrived(&mut app);
    wait_for_job(&mut app);
    assert!((app.c.get_scale() - 0.00006400000000000002).abs() < 1e-18 && app.buffer_is_complete);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn with_auto_blur_the_place_stays_blurred_until_its_render_is_finished() {
    let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
    (app.auto_blur, app.pace) = (true, super::flight::Pace::Fast);
    app.go_to_point_of_interest(0);
    app.skip_flight();
    assert!(app.landing_blur, "blurred after the arrival");
    wait_for_job(&mut app);
    assert!(!app.landing_blur, "sharp once the render of the place is finished");
    //Without Auto blur there is no blur at the landing either
    app.auto_blur = false;
    app.go_to_point_of_interest(0);
    app.skip_flight();
    assert!(!app.landing_blur);
    wait_for_job(&mut app);
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
#[ignore = "timing, run by hand"]
fn video_render_time_with_frames_at_once() {
    //(Measurement, run by hand: the time of a video with a fixed number of frames at once and with the number the job chooses (`None`), and what it chose.
    //MANDELBROT_VIDEO_TEST="width height samples speed place" sets the video, e.g. "1920 1080 16 25 09-the-whirlpool.json"; by default a short 1080p one)
    let setup = std::env::var("MANDELBROT_VIDEO_TEST").unwrap_or_else(|_| String::from("1920 1080 4 25 -"));
    let fields: Vec<&str> = setup.split_whitespace().collect();
    let (width, height, samples, speed) = (fields[0].parse().unwrap(), fields[1].parse().unwrap(), fields[2].parse().unwrap(), fields[3].parse().unwrap());
    let place = if fields[4] == "-" { String::from(HSV_BGR_PLACE) } else { std::fs::read_to_string(std::path::Path::new("points_of_interest").join(fields[4])).unwrap() };
    for (fixed, gpu_renderer) in [(Some(2), false), (None, false), (Some(2), true), (None, true)] {
        let (mut app, folder) = app_with_points(&[("test-place.json", &place)]);
        if app.gpu.is_none() {
            app.gpu = rendering::gpu::GpuIterator::new().ok().map(std::sync::Arc::new);
        }
        app.video_folder = folder.join("saved");
        app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
        (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (width, height, samples, 30);
        (app.pace, app.use_gpu, app.video_speed, app.video_at_once_fixed, app.video_gpu_renderer) = (super::flight::Pace::Fast, true, speed, fixed, gpu_renderer);
        app.go_to_point_of_interest(0);
        app.skip_flight();
        wait_for_job(&mut app);
        let started = std::time::Instant::now();
        app.start_video();
        let ctx = egui::Context::default();
        let mut seen = Vec::new();
        while let Some(job) = &app.video_job {
            let state = (job.scheduler.allowed(), job.scheduler.cpu_percent.load(Ordering::Relaxed));
            if seen.last() != Some(&state) {
                seen.push(state);
            }
            app.poll_video(&ctx);
            std::thread::sleep(Duration::from_millis(5));
        }
        println!("{fixed:?} at once, GPU renderer {gpu_renderer}: {:.2} s; (at once, CPU %) on the way: {seen:?}; {}", started.elapsed().as_secs_f64(), app.log.last().unwrap());
        let _ = std::fs::remove_dir_all(folder);
    }
}

#[test]
#[ignore = "a measurement of the CPU and the GPU on a real video (#212), run by hand with --release"]
fn video_throughput_of_a_manifest() {
    //MANDELBROT_MANIFEST: the manifest of a video (its exact settings and flight); MANDELBROT_SPEED: another speed in percent (fewer frames)
    let Ok(path) = std::env::var("MANDELBROT_MANIFEST") else { return };
    let manifest: super::chunks::Manifest = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut spec = manifest.spec;
    if let Some(speed) = std::env::var("MANDELBROT_SPEED").ok().and_then(|speed| speed.parse().ok()) {
        spec.speed = speed;
    }
    //MANDELBROT_SIZE="width height samples": a smaller video; MANDELBROT_KEEP: a lossless video kept at that path (to compare the frames of two builds)
    if let Ok(size) = std::env::var("MANDELBROT_SIZE") {
        let numbers: Vec<usize> = size.split_whitespace().map(|number| number.parse().unwrap()).collect();
        (spec.width, spec.height, spec.samples) = (numbers[0], numbers[1], u8::try_from(numbers[2]).unwrap());
    }
    //MANDELBROT_SHUTTER: motion blur, in degrees (#210)
    if let Some(shutter) = std::env::var("MANDELBROT_SHUTTER").ok().and_then(|shutter| shutter.parse().ok()) {
        spec.shutter = shutter;
    }
    let keep = std::env::var("MANDELBROT_KEEP").ok();
    if keep.is_some() {
        spec.quality = super::video::VideoQuality::Lossless;
    }
    let config = Config::sized(64, 48, 50, 4);
    let mut app = MandelbrotApp::new(&config);
    wait_for_job(&mut app);
    if app.gpu.is_none() {
        app.gpu = rendering::gpu::GpuIterator::new().ok().map(std::sync::Arc::new);
    }
    //MANDELBROT_AT_ONCE: a fixed number of CPU frames at once instead of the job's own choice
    app.video_at_once_fixed = std::env::var("MANDELBROT_AT_ONCE").ok().and_then(|at_once| at_once.parse().ok());
    let folder = std::env::temp_dir().join(format!("mandelbrot-throughput-{}", std::process::id()));
    app.video_folder = folder.join("saved");
    let feed = |name: &str| -> u64 {
        let counter = match name { "submitting" => &rendering::GPU_FEED.submitting, "waiting" => &rendering::GPU_FEED.waiting, "coloring" => &rendering::GPU_FEED.coloring, "dispatches" => &rendering::GPU_FEED.dispatches, _ => &rendering::GPU_FEED.points };
        counter.load(Ordering::Relaxed)
    };
    let before: Vec<u64> = ["submitting", "waiting", "coloring", "dispatches", "points"].iter().map(|name| feed(name)).collect();
    let started = std::time::Instant::now();
    assert!(app.start_video_of(super::video::VideoSpec { output: Some(String::from("throughput")), ..spec }));
    let ctx = egui::Context::default();
    let mut last_print = started;
    while let Some(job) = &app.video_job {
        if last_print.elapsed() >= Duration::from_secs(5) {
            last_print = std::time::Instant::now();
            println!("{:6.1} s: {} frames, {} at once, CPU {}%", started.elapsed().as_secs_f64(), job.scheduler.finished.load(Ordering::Relaxed), job.scheduler.allowed(), job.scheduler.cpu_percent.load(Ordering::Relaxed));
        }
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(5));
    }
    let after: Vec<u64> = ["submitting", "waiting", "coloring", "dispatches", "points"].iter().map(|name| feed(name)).collect();
    let delta: Vec<u64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
    println!("total {:.1} s; {}", started.elapsed().as_secs_f64(), app.log.last().unwrap());
    println!("GPU feeders: submitting {:.1} s, waiting {:.1} s, coloring {:.1} s; {} dispatches, {:.1} M points", delta[0] as f64 / 1e9, delta[1] as f64 / 1e9, delta[2] as f64 / 1e9, delta[3], delta[4] as f64 / 1e6);
    if let Some(keep) = keep {
        std::fs::copy(folder.join("saved").join("throughput.mkv"), keep).unwrap();
    }
    let _ = std::fs::remove_dir_all(folder);
}

#[test]
fn motion_blur_spreads_the_samples_over_the_shutter_and_a_video_with_it_has_every_frame() {
    //Issue #210: the moments of the samples of a frame
    let (time, interval, duration) = (Duration::from_secs(1), Duration::from_millis(100), Duration::from_secs(10));
    let moments = super::video::sample_times(64, time, interval, 180.0, duration);
    assert_eq!(moments.len(), 64);
    assert!(moments[0].abs_diff(time) < Duration::from_millis(1), "sample 0 at the frame: {:?}", moments[0]);
    assert!(moments.iter().all(|moment| moment.abs_diff(time) <= Duration::from_millis(25)), "within half the shutter, 180° of 100 ms: 25 ms either way");
    let mut sorted = moments.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 64, "every sample at its own moment");
    let mean = moments.iter().map(Duration::as_secs_f64).sum::<f64>() / 64.0;
    assert!((mean - 1.0).abs() < 1e-3, "centered on the frame: {mean}");
    assert!(super::video::sample_times(8, time, interval, 0.0, duration).iter().all(|moment| *moment == time), "without motion blur, all at the frame");
    assert!(super::video::sample_times(4, Duration::ZERO, interval, 360.0, duration).iter().all(|moment| *moment <= duration), "within the flight");
    //A video with motion blur: every frame, and the frames differ from those without it
    if !ffmpeg_available() {
        return;
    }
    let render = |shutter: f64| {
        let (mut app, folder) = chunked_video_app(false);
        app.video_shutter = shutter;
        app.image_supersampling_amount = 8;
        app.start_video();
        let message = wait_for_video(&mut app);
        assert!(message.starts_with("Video of the flight:"), "{message}");
        let video = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).map(|entry| entry.path()).find(|path| path.extension().is_some_and(|extension| extension == "mkv")).unwrap();
        let frames = decoded_frames(&video);
        let _ = std::fs::remove_dir_all(folder);
        frames
    };
    let (sharp, blurred) = (render(0.0), render(180.0));
    assert_eq!(sharp.len(), blurred.len(), "every frame");
    assert_ne!(sharp, blurred, "motion blur changes the frames");
}

#[test]
fn a_blended_video_has_every_frame_and_blends_the_frames_after_the_first() {
    let render = |blend: bool| {
        let (mut app, folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
        app.video_folder = folder.join("saved");
        app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
        (app.save_size.custom_width, app.save_size.custom_height, app.image_supersampling_amount, app.video_fps) = (48, 32, 1, 24);
        (app.pace, app.video_pngs, app.video_blend) = (super::flight::Pace::Fast, true, blend);
        app.go_to_point_of_interest(0);
        app.skip_flight();
        app.start_video();
        let ctx = egui::Context::default();
        let started = std::time::Instant::now();
        while app.video_job.is_some() {
            assert!(started.elapsed() < Duration::from_secs(60), "the video did not finish");
            app.poll_video(&ctx);
            std::thread::sleep(Duration::from_millis(5));
        }
        wait_for_job(&mut app);
        let written = std::fs::read_dir(folder.join("saved")).unwrap().filter_map(Result::ok).find(|entry| entry.path().is_dir()).unwrap().path();
        let mut frames: Vec<std::path::PathBuf> = std::fs::read_dir(&written).unwrap().filter_map(Result::ok).map(|entry| entry.path()).collect();
        frames.sort();
        //(The pixels: the metadata names the video, which differs)
        let bytes: Vec<Vec<u8>> = frames.iter().map(|frame| {
            let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(frame).unwrap())).read_info().unwrap();
            let mut pixels = vec![0; reader.output_buffer_size()];
            reader.next_frame(&mut pixels).unwrap();
            pixels
        }).collect();
        let _ = std::fs::remove_dir_all(folder);
        bytes
    };
    let (plain, blended) = (render(false), render(true));
    assert_eq!(plain.len(), blended.len(), "every frame");
    assert_eq!(plain[0], blended[0], "the first frame has nothing to blend with");
    assert!(plain.iter().zip(&blended).skip(1).any(|(a, b)| a != b), "the frames after it are blended");
}

/// The measures of a configuration of the quality experiment against the reference: the mean error per color channel, the sharpness (the mean gradient
/// over the reference's), the shimmer (how much the change between two frames differs from the reference's change), and the change between frames
#[derive(Default)]
struct QualityMeasures {
    error: f64,
    sharpness: f64,
    shimmer: f64,
    change: f64,
    frames: usize,
}

fn channels(colors: &[crate::coloring::TrueColor]) -> impl Iterator<Item = f64> + '_ {
    colors.iter().flat_map(|color| [f64::from(color.red), f64::from(color.green), f64::from(color.blue)])
}

fn gradient(colors: &[crate::coloring::TrueColor], width: usize) -> f64 {
    let values: Vec<f64> = channels(colors).collect();
    let (rows, row) = (colors.len() / width, width * 3);
    let mut total = 0.0;
    for y in 0..rows - 1 {
        for x in 0..row - 3 {
            let at = y * row + x;
            total += (values[at + 3] - values[at]).abs() + (values[at + row] - values[at]).abs();
        }
    }
    total / ((rows - 1) * (row - 3)) as f64
}

#[test]
#[ignore = "a quality experiment (issues #186 and #187), run by hand: cargo test --release --lib quality_experiment -- --ignored --nocapture"]
#[allow(clippy::too_many_lines)] // One experiment: its configurations, its frames and its measures
fn quality_experiment_supersampling_blur_and_blend_in_motion() {
    use crate::coloring::TrueColor;
    use std::fmt::Write;
    let (width, height, frames) = (480, 270, 48);
    //(name, samples, blur sigma, blend, jitter: every frame another single sample, or with more samples other sample numbers (issue #188))
    let sigma = |samples: u8| super::blur::blur_sigma(1, samples);
    //(Blended configurations named "... zoom N" weigh new frames by the zoom with ZOOM_WEIGHT N, see `temporal::blend_with`)
    let configurations: Vec<(String, u8, Option<f32>, bool, bool)> = vec![
        (String::from("1x"), 1, None, false, false),
        (String::from("1x + blur"), 1, Some(sigma(1)), false, false),
        (String::from("1x jittered + blend + blur"), 1, Some(sigma(1)), true, true),
        (String::from("4x"), 4, None, false, false),
        (format!("4x + blur {:.2}", sigma(4) * 0.7), 4, Some(sigma(4) * 0.7), false, false),
        (format!("4x + blur {:.2}", sigma(4)), 4, Some(sigma(4)), false, false),
        (format!("4x + blur {:.2}", sigma(4) * 1.4), 4, Some(sigma(4) * 1.4), false, false),
        (String::from("8x"), 8, None, false, false),
        (format!("8x + blur {:.2}", sigma(8) * 0.7), 8, Some(sigma(8) * 0.7), false, false),
        (format!("8x + blur {:.2}", sigma(8)), 8, Some(sigma(8)), false, false),
        (format!("8x + blur {:.2}", sigma(8) * 1.4), 8, Some(sigma(8) * 1.4), false, false),
        (String::from("16x"), 16, None, false, false),
        (String::from("64x"), 64, None, false, false),
        (String::from("64x + blend"), 64, None, true, false),
        (String::from("4x + blend"), 4, None, true, false),
        (String::from("4x rotating + blend"), 4, None, true, true),
        (String::from("8x rotating + blend"), 8, None, true, true),
        (format!("4x rotating + blend + blur {:.2}", sigma(4) * 0.7), 4, Some(sigma(4) * 0.7), true, true),
        (String::from("64x + blend zoom 4"), 64, None, true, false),
        (String::from("64x + blend zoom 10"), 64, None, true, false),
        (String::from("64x + blend zoom 25"), 64, None, true, false),
        (String::from("1x jittered + blend + blur zoom 4"), 1, Some(sigma(1)), true, true),
        (String::from("1x jittered + blend + blur zoom 10"), 1, Some(sigma(1)), true, true),
    ];
    let only = std::env::var("MANDELBROT_EXPERIMENT_ONLY").ok(); //(A part of the names, to run only those configurations)
    let doublings: f64 = std::env::var("MANDELBROT_EXPERIMENT_DOUBLINGS").ok().and_then(|doublings| doublings.parse().ok()).unwrap_or(3.0); //(How far the zoom goes in the 48 frames)
    let gpu = rendering::gpu::GpuIterator::new().ok();
    let mut report = format!("{width} x {height}, {frames} frames of a zoom of {doublings} doublings into each place, adaptive supersampling, against 32x (the error per channel, 0 to 255)\n");
    for place in ["02-frozen-constellation.json", "06-seahorse-embers.json", "09-the-whirlpool.json"] {
        let (point, _) = crate::points_of_interest::parse(&std::fs::read_to_string(std::path::Path::new("points_of_interest").join(place)).unwrap(), place).unwrap();
        let m = crate::mandelbrot_set::MandelbrotSet::new(point.location.max_iterations, 2.0);
        let render = |frame: usize, samples: u8, jitter: Option<u8>| -> (Vec<TrueColor>, ComplexPlane, f64) {
            let mut plane = ComplexPlane::new(width, height);
            let scale = point.location.scale * 2.0_f64.powf(doublings * (1.0 - frame as f64 / (frames - 1) as f64));
            plane.set_view(&crate::complex_plane::View::new(point.location.real, point.location.imaginary, scale));
            let started = std::time::Instant::now();
            let mut colors = Vec::new();
            let run = rendering::Render::new(&plane, &m, width, height, TrueColor::new_from_bernstein_polynomials).supersampling(samples).adaptive(Some(rendering::AdaptiveSampling::default()))
                .gpu(gpu.as_ref().map(|gpu| (gpu, Duration::ZERO)));
            let run = match jitter { Some(sample) if samples <= 1 => run.colors_only_of_sample(sample), Some(sample) => run.sample_offset(sample.wrapping_mul(samples)), None => run };
            run.run(|layer, _| colors = layer.to_vec());
            (colors, plane, started.elapsed().as_secs_f64())
        };
        let reference: Vec<Vec<TrueColor>> = (0..frames).map(|frame| render(frame, 32, None).0).collect();
        let reference_gradient = reference.iter().map(|colors| gradient(colors, width)).sum::<f64>() / frames as f64;
        let _ = writeln!(report, "\n{} (zoom {:.0} to {:.0}):", point.name, 1.0 / (point.location.scale * 2.0_f64.powf(doublings)), 1.0 / point.location.scale);
        for (name, samples, blur, blend, jitter) in &configurations {
            if only.as_ref().is_some_and(|only| !name.contains(only.as_str())) {
                continue;
            }
            let zoom_weight = name.split(" zoom ").nth(1).and_then(|weight| weight.parse::<f32>().ok()).unwrap_or(super::temporal::ZOOM_WEIGHT);
            let (mut measures, mut seconds, mut history, mut previous) = (QualityMeasures::default(), 0.0, None, None::<Vec<TrueColor>>);
            for (frame, expected) in reference.iter().enumerate() {
                let (colors, plane, time) = render(frame, *samples, jitter.then_some((frame % 64) as u8));
                seconds += time;
                let colors = if *blend {
                    let mut shown = Vec::new();
                    super::temporal::blend_with(&mut history, &plane, &colors, width, height, &mut shown, zoom_weight);
                    shown
                } else {
                    colors
                };
                let colors = blur.map_or(colors.clone(), |sigma| super::blur::blurred(&colors, width, height, sigma));
                measures.error += channels(&colors).zip(channels(expected)).map(|(a, b)| (a - b).abs()).sum::<f64>() / (colors.len() * 3) as f64;
                measures.sharpness += gradient(&colors, width) / reference_gradient;
                if let Some(before) = &previous {
                    let expected_before = &reference[frame - 1];
                    let (mut shimmer, mut change) = (0.0, 0.0);
                    for (((now, then), expected_now), expected_then) in channels(&colors).zip(channels(before)).zip(channels(expected)).zip(channels(expected_before)) {
                        shimmer += ((now - then) - (expected_now - expected_then)).abs();
                        change += (now - then).abs();
                    }
                    measures.shimmer += shimmer / (colors.len() * 3) as f64;
                    measures.change += change / (colors.len() * 3) as f64;
                }
                previous = Some(colors);
                measures.frames += 1;
            }
            let n = measures.frames as f64;
            let _ = writeln!(report, "  {name:<28} error {:5.2}  sharpness {:4.2}  shimmer {:5.2}  change between frames {:5.2}  {:6.1} ms a frame",
                measures.error / n, measures.sharpness / n, measures.shimmer / (n - 1.0), measures.change / (n - 1.0), seconds / n * 1000.0);
        }
        let reference_change = reference.windows(2).map(|pair| channels(&pair[1]).zip(channels(&pair[0])).map(|(a, b)| (a - b).abs()).sum::<f64>() / (pair[0].len() * 3) as f64).sum::<f64>() / (frames - 1) as f64;
        let _ = writeln!(report, "  (32x itself: change between frames {reference_change:5.2})");
    }
    println!("{report}");
    std::fs::write(std::env::temp_dir().join("mandelbrot-quality-experiment.txt"), report).unwrap();
}

#[test]
#[ignore = "issue #192 with Jort's settings and the GPU, run by hand"]
fn after_a_cancelled_dive_with_jorts_settings_and_the_gpu_the_same_place_can_be_dived_to_again() {
    for attempt in 0..5 {
        let (mut app, _folder) = app_with_points(&[("test-place.json", HSV_BGR_PLACE)]);
        app.gpu = rendering::gpu::GpuIterator::new().ok().map(std::sync::Arc::new);
        (app.flight_rendering, app.pace, app.route, app.dive_zoom, app.auto_iterations, app.frames_ahead, app.f32_frames, app.fast_gpu, app.use_gpu) =
            (super::flight::FlightRendering::FULL_RESOLUTION, super::flight::Pace::Constant, super::flight::Route::Dive, 0.3, true, 2, true, true, true);
        app.go_to_point_of_interest(0);
        let cancel_at = 0.1 + 0.2 * f64::from(attempt);
        assert!(drive_flight(&mut app, 30, |app| app.flight.as_ref().is_some_and(|flight| flight.elapsed.as_secs_f64() > cancel_at * flight.flight.duration().as_secs_f64())), "attempt {attempt}: the first dive gets there");
        app.cancel_flight();
        app.go_to_point_of_interest(0);
        let moving = |app: &MandelbrotApp| app.flight.as_ref().is_none_or(|flight| !flight.elapsed.is_zero());
        assert!(drive_flight(&mut app, 30, moving), "attempt {attempt}: the second dive moves: {:?}", app.log.iter().rev().take(4).collect::<Vec<_>>());
        assert!(drive_flight(&mut app, 120, |app| app.flight.is_none()), "attempt {attempt}: the second dive arrives: {:?}", app.log.iter().rev().take(4).collect::<Vec<_>>());
    }
}

//! The benchmark tour (issue #101): a repeatable flight through the five deepest places, like the benchmark mode of a game, with frame statistics per scene.
//!
//! The tour renders the reset view (a warm-up), then flies at the cinematic pace with *Low res* frames to each of the five deepest places that the zoom limit
//! allows, shallow to deep, and waits for the full-quality render at each. Per scene it reports the frames, the frame times (minimum, average, maximum, 95th
//! percentile), the average and the 1% low frame rate, the flight time against the planned one (a slow scene stretches its flight), and the arrival: the time to
//! the first full layer and to the finished render. The settings of the window are restored afterwards; Stop, or using the controls during a flight, ends the
//! run. Every run is saved to `benchmarks/` (JSON with every frame time, and a CSV summary) and compared with the run before in the log.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use eframe::egui;

use super::flight::{FlightRendering, Pace, Route};
use super::MandelbrotApp;
use crate::complex_plane::View;

/// How many of the deepest places the tour visits
const SCENES: usize = 5;

/// The folder of the saved runs (see `MandelbrotApp::benchmark_folder`)
pub(super) const FOLDER: &str = "benchmarks";

/// The settings the tour changes, to restore them afterwards
struct Saved {
    pace: Pace,
    route: Route,
    flight_rendering: FlightRendering,
    auto_iterations: bool,
    view: View,
}

/// Where the tour is
enum Phase {
    /// Rendering the reset view (the warm-up), before the first flight
    WarmUp,
    /// Flying to scene `index` of `scenes`
    Flying { index: usize },
    /// At scene `index`: the flight arrived at `arrived`, and the first full layer was shown at `first_layer`
    Arriving { index: usize, arrived: Instant, first_layer: Option<Duration> },
}

/// The measurements of one scene
#[derive(Clone, Debug)]
pub(super) struct SceneResult {
    pub(super) name: String,
    pub(super) zoom: f64,
    /// The time between the frames shown (the first from the start of the flight)
    pub(super) frame_times: Vec<Duration>,
    pub(super) flight: Duration,
    pub(super) planned: Duration,
    pub(super) first_layer: Duration,
    pub(super) finished: Duration,
}

/// A run of the tour
pub(super) struct Benchmark {
    saved: Saved,
    scenes: Vec<usize>,
    phase: Phase,
    /// The frames of the flight under way and when it started, copied every frame of the window (the flight is gone once it arrives)
    frames: (Vec<Instant>, Instant, Duration),
    results: Vec<SceneResult>,
}

/// Statistics of frame times
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FrameStats {
    pub(super) frames: usize,
    pub(super) min: f64,
    pub(super) average: f64,
    pub(super) max: f64,
    pub(super) p95: f64,
    pub(super) average_fps: f64,
    pub(super) low_fps: f64,
}

/// The statistics of `frame_times` over `flight` (milliseconds; FPS): the 1% low is the frame rate of the average of the slowest 1% of the frames (at least one)
pub(super) fn frame_stats(frame_times: &[Duration], flight: Duration) -> FrameStats {
    if frame_times.is_empty() {
        return FrameStats::default();
    }
    let mut ms: Vec<f64> = frame_times.iter().map(|time| time.as_secs_f64() * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    let slowest = ms.len().div_ceil(100);
    let low = ms[ms.len() - slowest..].iter().sum::<f64>() / slowest as f64;
    let percentile = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
    FrameStats {
        frames: ms.len(),
        min: ms[0],
        average: ms.iter().sum::<f64>() / ms.len() as f64,
        max: ms[ms.len() - 1],
        p95: percentile(0.95),
        average_fps: ms.len() as f64 / flight.as_secs_f64().max(1e-9),
        low_fps: 1000.0 / low.max(1e-9),
    }
}

impl MandelbrotApp {
    /// Starts the tour, see the module
    pub(super) fn start_benchmark(&mut self) {
        //The deepest places the zoom limit allows, shallow to deep (the list is sorted by zoom, the deepest last)
        let mut scenes: Vec<usize> = (0..self.points.len()).filter(|index| {
            let location = &self.points[*index].location;
            let mut plane = self.c.clone();
            plane.set_view(&View::new(location.real, location.imaginary, location.scale));
            !self.zoom_limited(&plane)
        }).collect();
        scenes = scenes.split_off(scenes.len().saturating_sub(SCENES));
        if scenes.is_empty() {
            self.log("Benchmark: no places to fly to");
            return;
        }
        let saved = Saved { pace: self.pace, route: self.route, flight_rendering: self.flight_rendering, auto_iterations: self.auto_iterations, view: self.c.get_view() };
        (self.pace, self.route, self.flight_rendering, self.auto_iterations) = (Pace::Cinematic, Route::Shortest, FlightRendering::LowRes, true);
        self.log(format!("Benchmark: a tour of {} places, warming up at the reset view", scenes.len()));
        self.benchmark = Some(Benchmark { saved, scenes, phase: Phase::WarmUp, frames: (Vec::new(), Instant::now(), Duration::ZERO), results: Vec::new() });
        self.c.set_view(&View::new(-0.75, 0.0, 1.0));
        self.render("Benchmark: the reset view");
    }

    /// Ends the tour (finished or stopped) and restores the settings it changed
    pub(super) fn stop_benchmark(&mut self, finished: bool) {
        let Some(run) = self.benchmark.take() else { return };
        (self.pace, self.route, self.flight_rendering, self.auto_iterations) = (run.saved.pace, run.saved.route, run.saved.flight_rendering, run.saved.auto_iterations);
        if finished {
            self.report_benchmark(&run.results);
            self.benchmark_results = run.results;
        } else {
            self.log(format!("Benchmark stopped after {} of {} scenes: incomplete, not saved", run.results.len(), run.scenes.len()));
            self.cancel_flight();
        }
        self.c.set_view(&run.saved.view);
        self.render("Benchmark: back to the view before the tour");
    }

    /// One step of the tour, every frame of the window
    pub(super) fn step_benchmark(&mut self, ctx: &egui::Context) {
        let Some(phase) = self.benchmark.as_ref().map(|run| match run.phase {
            Phase::WarmUp => Phase::WarmUp,
            Phase::Flying { index } => Phase::Flying { index },
            Phase::Arriving { index, arrived, first_layer } => Phase::Arriving { index, arrived, first_layer },
        }) else { return };
        ctx.request_repaint();
        let now = Instant::now();
        let finished = !self.is_busy() && self.buffer_is_complete;
        match phase {
            Phase::WarmUp => {
                if finished {
                    self.fly_to_scene(0);
                }
            }
            Phase::Flying { index } => {
                if let Some(flight) = &self.flight {
                    let frames = (flight.frames.clone(), flight.started, flight.flight.duration());
                    if let Some(run) = &mut self.benchmark {
                        run.frames = frames;
                    }
                    return;
                }
                let Some(run) = &mut self.benchmark else { return };
                let place = &self.points[run.scenes[index]].location;
                let arrived = (self.c.get_scale() - place.scale).abs() <= 1e-12 * place.scale && (self.c.center().x - place.real).abs() <= 1e-9 * place.scale;
                if arrived {
                    run.phase = Phase::Arriving { index, arrived: now, first_layer: None };
                } else {
                    self.stop_benchmark(false); //The flight was stopped
                }
            }
            Phase::Arriving { index, arrived, first_layer } => {
                let first_layer = first_layer.or_else(|| self.layer_shown.then(|| now - arrived));
                let Some(run) = &mut self.benchmark else { return };
                if !finished {
                    run.phase = Phase::Arriving { index, arrived, first_layer };
                    return;
                }
                let point = &self.points[run.scenes[index]];
                let (frames, started, planned) = &run.frames;
                let mut last = *started;
                let frame_times = frames.iter().map(|shown| {
                    let time = shown.saturating_duration_since(last);
                    last = *shown;
                    time
                }).collect();
                let flight = last.saturating_duration_since(*started);
                run.results.push(SceneResult { name: point.name.clone(), zoom: 1.0 / point.location.scale, frame_times, flight, planned: *planned, first_layer: first_layer.unwrap_or(now - arrived), finished: now - arrived });
                if index + 1 < run.scenes.len() {
                    self.fly_to_scene(index + 1);
                } else {
                    self.stop_benchmark(true);
                }
            }
        }
    }

    fn fly_to_scene(&mut self, index: usize) {
        let Some(run) = &mut self.benchmark else { return };
        run.phase = Phase::Flying { index };
        let point = run.scenes[index];
        self.go_to_point_of_interest(point);
    }

    /// Logs the results, saves them to `benchmarks/` and compares them with the run before
    fn report_benchmark(&mut self, results: &[SceneResult]) {
        let all: Vec<Duration> = results.iter().flat_map(|scene| scene.frame_times.iter().copied()).collect();
        let total = frame_stats(&all, results.iter().map(|scene| scene.flight).sum());
        self.log(format!("Benchmark finished: {} frames, {:.1} FPS on average, 1% low {:.1} FPS, slowest scene {}", total.frames, total.average_fps, total.low_fps,
            results.iter().max_by(|a, b| frame_stats(&a.frame_times, a.flight).average.total_cmp(&frame_stats(&b.frame_times, b.flight).average)).map_or("-", |scene| scene.name.as_str())));
        let previous = previous_run(&self.benchmark_folder);
        for scene in results {
            let stats = frame_stats(&scene.frame_times, scene.flight);
            let before = previous.as_ref().and_then(|runs| runs.iter().find(|(name, _)| *name == scene.name)).map(|(_, average)| *average);
            let compared = before.map_or_else(String::new, |before| format!(" (before {before:.1} ms, {:+.0}%)", (stats.average / before - 1.0) * 100.0));
            self.log(format!("Benchmark {}: average frame {:.1} ms{compared}, {:.1} FPS, 1% low {:.1} FPS, flight {:.1} s of {:.1} s, finished {:.2} s after arriving",
                scene.name, stats.average, stats.average_fps, stats.low_fps, scene.flight.as_secs_f64(), scene.planned.as_secs_f64(), scene.finished.as_secs_f64()));
        }
        match self.save_benchmark(results) {
            Ok(path) => self.log(format!("Benchmark saved to {path}")),
            Err(err) => self.log(format!("Benchmark not saved: {err}")),
        }
    }

    /// Writes the run as `benchmarks/<date>.json` (with every frame time and the settings) and `.csv` (one row per scene)
    fn save_benchmark(&self, results: &[SceneResult]) -> std::io::Result<String> {
        let folder = self.benchmark_folder.display();
        std::fs::create_dir_all(&self.benchmark_folder)?;
        let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let commit = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().filter(|output| output.status.success())
            .map_or_else(|| String::from("unknown"), |output| String::from_utf8_lossy(&output.stdout).trim().to_string());
        let scenes: Vec<serde_json::Value> = results.iter().map(|scene| {
            let stats = frame_stats(&scene.frame_times, scene.flight);
            serde_json::json!({
                "name": scene.name, "zoom": scene.zoom, "frames": stats.frames, "min_ms": stats.min, "average_ms": stats.average, "max_ms": stats.max, "p95_ms": stats.p95,
                "average_fps": stats.average_fps, "low_1_percent_fps": stats.low_fps, "flight_s": scene.flight.as_secs_f64(), "planned_s": scene.planned.as_secs_f64(),
                "first_layer_s": scene.first_layer.as_secs_f64(), "finished_s": scene.finished.as_secs_f64(),
                "frame_times_ms": scene.frame_times.iter().map(|time| time.as_secs_f64() * 1000.0).collect::<Vec<f64>>(),
            })
        }).collect();
        let run = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"), "commit": commit, "date": stamp,
            "window": [self.p.pixel_plane.width, self.p.pixel_plane.height], "cpu_threads": crate::rendering::render_thread_count(),
            "gpu": self.gpu.as_ref().filter(|_| self.use_gpu).map_or_else(|| String::from("off"), |gpu| gpu.name().to_string()), "fast_gpu": self.fast_gpu,
            "flight_rendering": FlightRendering::LowRes.name(), "supersampling": self.supersampling_amount, "scenes": scenes,
        });
        let path = format!("{folder}/{stamp}-{commit}.json");
        std::fs::write(&path, serde_json::to_string_pretty(&run).map_err(std::io::Error::other)?)?;
        let mut csv = String::from("scene,zoom,frames,min_ms,average_ms,max_ms,p95_ms,average_fps,low_1_percent_fps,flight_s,planned_s,first_layer_s,finished_s\n");
        for scene in results {
            let stats = frame_stats(&scene.frame_times, scene.flight);
            let _ = writeln!(csv, "{},{:.0},{},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1},{:.2},{:.2},{:.3},{:.3}", scene.name, scene.zoom, stats.frames, stats.min, stats.average, stats.max, stats.p95,
                stats.average_fps, stats.low_fps, scene.flight.as_secs_f64(), scene.planned.as_secs_f64(), scene.first_layer.as_secs_f64(), scene.finished.as_secs_f64());
        }
        std::fs::write(format!("{folder}/{stamp}-{commit}.csv"), csv)?;
        Ok(path)
    }

    /// The Benchmark controls: the button, and the table of the last run
    pub(super) fn benchmark_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if self.benchmark.is_some() {
                if ui.button("Stop benchmark").clicked() {
                    self.stop_benchmark(false);
                }
                ui.label("Touring the deepest places...");
            } else if ui.add_enabled(self.flight.is_none(), egui::Button::new("Run benchmark")).on_hover_text("A cinematic tour through the five deepest places with Low res frames, like the benchmark mode of a game: frame statistics per place, saved to the benchmarks folder and compared with the run before. Your settings are restored afterwards.").clicked() {
                self.start_benchmark();
            }
        });
        if self.benchmark_results.is_empty() {
            return;
        }
        let slowest = self.benchmark_results.iter().enumerate().max_by(|(_, a), (_, b)| frame_stats(&a.frame_times, a.flight).average.total_cmp(&frame_stats(&b.frame_times, b.flight).average)).map(|(index, _)| index);
        egui::CollapsingHeader::new("Benchmark results").default_open(true).show(ui, |ui| {
            egui::Grid::new("benchmark").striped(true).show(ui, |ui| {
                for heading in ["Scene", "Frames", "Avg ms", "p95 ms", "Max ms", "FPS", "1% low", "Flight", "Arrival"] {
                    ui.strong(heading);
                }
                ui.end_row();
                for (index, scene) in self.benchmark_results.iter().enumerate() {
                    let stats = frame_stats(&scene.frame_times, scene.flight);
                    let name = if Some(index) == slowest { egui::RichText::new(&scene.name).color(egui::Color32::from_rgb(255, 140, 90)) } else { egui::RichText::new(&scene.name) };
                    ui.label(name);
                    for value in [format!("{}", stats.frames), format!("{:.1}", stats.average), format!("{:.1}", stats.p95), format!("{:.1}", stats.max), format!("{:.1}", stats.average_fps),
                        format!("{:.1}", stats.low_fps), format!("{:.1} / {:.1} s", scene.flight.as_secs_f64(), scene.planned.as_secs_f64()), format!("{:.2} s", scene.finished.as_secs_f64())] {
                        ui.label(value);
                    }
                    ui.end_row();
                }
            });
        });
    }
}

/// The average frame time of every scene of the latest saved run, if there is one
fn previous_run(folder: &std::path::Path) -> Option<Vec<(String, f64)>> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(folder).ok()?.filter_map(|entry| entry.ok().map(|entry| entry.path())).filter(|path| path.extension().is_some_and(|ext| ext == "json")).collect();
    files.sort();
    let text = std::fs::read_to_string(files.last()?).ok()?;
    let run: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(run["scenes"].as_array()?.iter().filter_map(|scene| Some((scene["name"].as_str()?.to_string(), scene["average_ms"].as_f64()?))).collect())
}

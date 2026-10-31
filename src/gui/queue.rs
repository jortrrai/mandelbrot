//! The render queue (issue #181): videos that render one after the other, unattended (overnight, for example), each with its own flight and settings.
//!
//! *Add to queue* takes the last flight with the current video settings (a `VideoSpec`: changing the controls later does not change it), *Add the saved
//! flights* every saved flight with them (with the length of a flight's file, when it has one). *Start queue* renders the videos in order; a video that is
//! cancelled or fails goes back to the front of the queue and the queue stops. The queue is saved next to the settings file (`video_queue.json`) whenever it
//! changes, so it is still there after the program was closed; it then waits for *Start queue*.

use eframe::egui;
use serde::{Deserialize, Serialize};

use super::flight::Flight;
use super::video::{speed_text, VideoSpec};
use super::MandelbrotApp;

/// The version of the queue file that this program writes
const FORMAT_VERSION: u32 = 1;

/// The contents of the queue file: the video under way when it was written first, then the waiting ones
#[derive(Serialize, Deserialize)]
struct QueueFile {
    format_version: u32,
    videos: Vec<VideoSpec>,
}

/// A short description of a queued video for its row
fn description(spec: &VideoSpec) -> String {
    format!("{}: {} x {}, {} FPS, {} speed, {}x{}, {}, {}", spec.flight.name, spec.width, spec.height, spec.fps, speed_text(spec.speed), spec.samples,
        if spec.blur { " + blur" } else { "" }, spec.quality.name(), spec.playback.name())
}

impl MandelbrotApp {
    /// The queue file: next to the settings file (none without one, as in the tests)
    fn queue_path(&self) -> Option<std::path::PathBuf> {
        self.settings_path.as_ref().map(|path| path.with_file_name("video_queue.json"))
    }

    /// Writes the queue file (the video under way first), or removes it when the queue is empty
    fn save_queue(&mut self) {
        let Some(path) = self.queue_path() else { return };
        let videos: Vec<VideoSpec> = self.queue_current.iter().chain(&self.video_queue).cloned().collect();
        let written = if videos.is_empty() {
            match std::fs::remove_file(&path) {
                Err(err) if err.kind() != std::io::ErrorKind::NotFound => Err(err),
                _ => Ok(()),
            }
        } else {
            let text = serde_json::to_string_pretty(&QueueFile { format_version: FORMAT_VERSION, videos }).expect("a queue is plain text and numbers") + "\n";
            path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&path, text))
        };
        if let Err(err) = written {
            self.log(format!("Could not save the queue to {}: {err}", path.display()));
        }
    }

    /// Reads the queue file of the last run, if there is one: its videos wait for *Start queue*
    pub(super) fn load_queue(&mut self) {
        let Some(path) = self.queue_path() else { return };
        let Ok(text) = std::fs::read_to_string(&path) else { return };
        match serde_json::from_str::<QueueFile>(&text) {
            Ok(file) if file.format_version <= FORMAT_VERSION => {
                self.video_queue = file.videos;
                if !self.video_queue.is_empty() {
                    self.log(format!("The queue of the last run has {} videos: Start queue renders them", self.video_queue.len()));
                }
            }
            Ok(file) => self.log(format!("The queue file {} has a newer format ({}) than this program reads", path.display(), file.format_version)),
            Err(err) => self.log(format!("Could not read the queue file {}: {err}", path.display())),
        }
    }

    /// Adds the last flight with the current video settings to the queue
    pub(super) fn add_to_queue(&mut self) {
        match self.video_spec() {
            Ok(spec) => {
                self.log(format!("Added to the queue: {}", description(&spec)));
                self.video_queue.push(spec);
                self.save_queue();
            }
            Err(reason) => self.log(format!("Could not add the video to the queue: {reason}")),
        }
    }

    /// Adds every saved flight with the current video settings to the queue: with the length of its file when it has one, else with the speed of the controls
    /// (or the length of *Fit to length*)
    pub(super) fn add_saved_flights_to_queue(&mut self) {
        let Ok(base) = self.video_spec().or_else(|_| {
            //Without a last flight, the settings with the first saved flight
            let first = self.saved_flights.first().map(|saved| saved.flight.clone()).ok_or_else(|| String::from("there are no saved flights"))?;
            let last = self.last_flight.replace(first);
            let spec = self.video_spec();
            self.last_flight = last;
            spec
        }).map_err(|reason| self.log(format!("Could not add the saved flights to the queue: {reason}"))) else { return };
        let mut added = 0;
        for saved in self.saved_flights.clone() {
            let flight = &saved.flight;
            let duration = Flight::with(flight.from, flight.to, flight.pace, flight.route, flight.constant_speed, flight.speed, flight.dive_zoom).duration();
            let speed = match saved.length {
                Some(length) => duration.as_secs_f64() * 100.0 / length.as_secs_f64(),
                None => match self.video_speed_for(duration) {
                    Ok(speed) => speed,
                    Err(reason) => {
                        self.log(format!("Not added to the queue: {} ({reason})", flight.name));
                        continue;
                    }
                },
            };
            self.video_queue.push(VideoSpec { flight: saved.flight.clone(), speed, ..base.clone() });
            added += 1;
        }
        self.log(format!("Added {added} saved flights to the queue"));
        self.save_queue();
    }

    /// Starts the next video of the queue when the queue runs and no video is under way; the queue stops when it is empty
    pub(super) fn run_queue(&mut self) {
        if !self.queue_running || self.video_job.is_some() {
            return;
        }
        if self.video_queue.is_empty() {
            self.queue_running = false;
            self.log("The queue is done");
            return;
        }
        let mut spec = self.video_queue.remove(0);
        //Named before it starts, so after a restart it resumes its own chunks (see `chunks`)
        spec.output = spec.output.or_else(|| Some(super::video::video_name(&spec.flight, &self.video_folder)));
        self.queue_current = Some(spec.clone());
        self.save_queue();
        self.start_video_of(spec);
    }

    /// A video has ended (`finished`: with all its frames): a video of the queue is done, or goes back to the front of the queue, which stops
    pub(super) fn queue_video_ended(&mut self, finished: bool) {
        let Some(spec) = self.queue_current.take() else { return };
        if !finished {
            self.log(format!("The queue stopped: {} did not finish and waits at the front of the queue", spec.flight.name));
            self.video_queue.insert(0, spec);
            self.queue_running = false;
        }
        self.save_queue();
    }

    /// The queue: its videos with buttons to move or remove them, and the buttons to add videos and to start or stop it
    pub(super) fn queue_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("Queue: {} videos", self.video_queue.len() + usize::from(self.queue_current.is_some())));
            if ui.add_enabled(self.last_flight.is_some(), egui::Button::new("Add to queue")).on_hover_text("Adds the video of the last flight, with the current video settings, to the queue (later changes of the settings do not change it)").clicked() {
                self.add_to_queue();
            }
            if ui.add_enabled(!self.saved_flights.is_empty(), egui::Button::new("Add the saved flights")).on_hover_text("Adds a video of every saved flight, with the current video settings (and the length of a flight's file, when it has one)").clicked() {
                self.add_saved_flights_to_queue();
            }
        });
        let mut action = None;
        for (index, spec) in self.video_queue.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.add_enabled(index > 0, egui::Button::new("Up").small()).clicked() {
                    action = Some((index, -1));
                }
                if ui.add_enabled(index + 1 < self.video_queue.len(), egui::Button::new("Down").small()).clicked() {
                    action = Some((index, 1));
                }
                if ui.small_button("Remove").clicked() {
                    action = Some((index, 0));
                }
                ui.label(format!("{}. {}", index + 1 + usize::from(self.queue_current.is_some()), description(spec)));
            });
        }
        if let Some((index, step)) = action {
            match step {
                0 => {
                    self.video_queue.remove(index);
                }
                step => self.video_queue.swap(index, index.saturating_add_signed(step)),
            }
            self.save_queue();
        }
        ui.horizontal(|ui| {
            if self.queue_running {
                if ui.button("Stop the queue after this video").on_hover_text("The video under way finishes; the next ones wait").clicked() {
                    self.queue_running = false;
                    self.log("The queue stops after this video");
                }
            } else if ui.add_enabled(!self.video_queue.is_empty(), egui::Button::new("Start queue")).on_hover_text("Renders the videos of the queue one after the other").clicked() {
                self.queue_running = true;
                self.log(format!("Started the queue: {} videos", self.video_queue.len()));
            }
        });
    }
}

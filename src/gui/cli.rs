//! The command line (issue #216): make, render, pause, resume and manage videos without the window, for scripts and long renders.
//!
//! `mandelbrot video <command> …` drives the same video job as the window (see `video`): the same renderer, place files, settings (the user's settings.json are the
//! defaults), chunks and manifests, so a paused video resumes like one paused in the window. A video is a *job*: a file `saved/jobs/<id>.json` with its
//! `VideoSpec`, named like its video. Every command prints one JSON object on stdout (`start` prints one per progress report); an error prints
//! `{"error": …}` on stderr and exits with 1 (2 for a command that is not understood).
//!
//! ```text
//! mandelbrot video new --to "Fractal Flower" [--from <place>] [--music song.mp3 [--strength 0.5]] [--length 3:20 | --speed 100]
//!                      [--size 4K | 3840x2160] [--fps 60] [--samples 32] [--shutter 180] [--pace Cinematic] [--route Dive] [--quality "Very High"]
//! mandelbrot video start <id>      render it here (in the foreground; with nohup or systemd for long renders), from its chunks if it has some
//! mandelbrot video pause <id>      ask the render to stop after its current chunk (start or resume continue it)
//! mandelbrot video resume <id>     the same as start
//! mandelbrot video cancel <id>     stop it (if it runs) and delete its chunks
//! mandelbrot video status <id>     its state, frames and progress
//! mandelbrot video list            every job and its state
//! mandelbrot video output <id>     the path of its video file
//! mandelbrot video delete <id> [--video]   delete the job and its chunks (and with --video its finished video)
//! mandelbrot youtube auth --client client_secret.json                          allow uploads, once (see `youtube`)
//! mandelbrot video credit <id> --kind bensound --text-file credit.txt          the exact credit of its music
//! mandelbrot video upload <id> --title "…" [--description-file d.txt]          upload the finished video, always private
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use eframe::egui;
use serde_json::{json, Value};

use super::video::VideoSpec;
use super::MandelbrotApp;

/// The version of the job files
const FORMAT_VERSION: u32 = 1;

/// How often `start` reports its progress and looks for a request to pause or cancel
const REPORT_EVERY: Duration = Duration::from_secs(5);

/// An error of a command: its message, and whether the command itself was not understood (exit code 2) or failed (1)
struct Failure {
    message: String,
    usage: bool,
}

fn failed(message: impl Into<String>) -> Failure {
    Failure { message: message.into(), usage: false }
}

fn usage(message: impl Into<String>) -> Failure {
    Failure { message: message.into(), usage: true }
}

/// Set while the command line runs: the log of the app goes to stderr then, so stdout has only the JSON of the commands
static QUIET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the log should stay off stdout (see `QUIET`)
pub(super) fn quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Runs the command line `arguments` (without the program's name) and returns the exit code
pub fn command_line(arguments: &[String]) -> i32 {
    QUIET.store(true, Ordering::Relaxed);
    match run(arguments) {
        Ok(()) => 0,
        Err(failure) => {
            eprintln!("{}", json!({ "error": failure.message }));
            if failure.usage { 2 } else { 1 }
        }
    }
}

fn run(arguments: &[String]) -> Result<(), Failure> {
    let (group, command, rest) = match arguments {
        [group, command, rest @ ..] => (group.as_str(), command.as_str(), rest),
        _ => return Err(usage("usage: mandelbrot video <new|start|pause|resume|cancel|status|list|output|delete|credit|upload> … (or mandelbrot youtube auth)")),
    };
    #[cfg(not(target_os = "android"))]
    if group == "youtube" {
        return match command {
            "auth" => {
                let options = options(rest)?;
                let client = option(&options, "client").ok_or_else(|| usage("youtube auth needs --client <client_secret.json> (a Desktop OAuth client of your Google Cloud project)"))?;
                print(&super::youtube::authorize(Path::new(client)).map_err(failed)?)
            }
            _ => Err(usage(format!("unknown youtube command {command:?}: the command is mandelbrot youtube auth"))),
        };
    }
    if group != "video" {
        return Err(usage(format!("unknown command {group:?}: the commands are mandelbrot video …")));
    }
    let jobs = jobs_folder();
    match command {
        "new" => print(&new_job(&jobs, &options(rest)?)?),
        "list" => print(&list(&jobs)),
        "start" | "resume" => start(&jobs, id_of(rest)?),
        "pause" => print(&request(&jobs, id_of(rest)?, "pause")?),
        "cancel" => print(&cancel(&jobs, id_of(rest)?)?),
        "status" => print(&status(&jobs, id_of(rest)?)?),
        "output" => print(&output(&jobs, id_of(rest)?)?),
        "delete" => print(&delete(&jobs, id_of(rest)?, rest.iter().any(|argument| argument == "--video"))?),
        #[cfg(not(target_os = "android"))]
        "credit" => print(&credit(&jobs, id_of(rest)?, &options(&rest[1..])?)?),
        #[cfg(not(target_os = "android"))]
        "upload" => print(&upload(&jobs, id_of(rest)?, &options(&rest[1..])?)?),
        _ => Err(usage(format!("unknown video command {command:?}"))),
    }
}

#[allow(clippy::unnecessary_wraps)] // (The result of a command, like the others)
fn print(value: &Value) -> Result<(), Failure> {
    println!("{value}");
    Ok(())
}

/// The folder of the jobs: `saved/jobs`, next to the videos (relative to where the program runs, like the window's `saved/`)
fn jobs_folder() -> PathBuf {
    PathBuf::from("saved").join("jobs")
}

fn videos_folder() -> PathBuf {
    PathBuf::from("saved")
}

/// The first argument: the id of a job
fn id_of(rest: &[String]) -> Result<&str, Failure> {
    rest.first().map(String::as_str).filter(|id| !id.starts_with("--")).ok_or_else(|| usage("which job? give its id (see mandelbrot video list)"))
}

/// The options `--name value` of `rest`, in order
fn options(rest: &[String]) -> Result<Vec<(String, String)>, Failure> {
    let mut options = Vec::new();
    let mut arguments = rest.iter();
    while let Some(argument) = arguments.next() {
        let name = argument.strip_prefix("--").ok_or_else(|| usage(format!("expected an option like --to, not {argument:?}")))?;
        let value = arguments.next().ok_or_else(|| usage(format!("--{name} needs a value")))?;
        options.push((name.to_string(), value.clone()));
    }
    Ok(options)
}

fn option<'a>(options: &'a [(String, String)], name: &str) -> Option<&'a str> {
    options.iter().rev().find(|(option, _)| option == name).map(|(_, value)| value.as_str())
}

fn number<T: std::str::FromStr>(options: &[(String, String)], name: &str) -> Result<Option<T>, Failure> {
    option(options, name).map(|text| text.trim().parse::<T>().map_err(|_| usage(format!("--{name} {text:?} is not a number")))).transpose()
}

/// A window-less app with the user's settings and places, as the window would start
fn headless_app() -> MandelbrotApp {
    let mut app = MandelbrotApp::new(&crate::Config::default());
    app.use_settings_file(super::settings::settings_path());
    app.video_folder = videos_folder();
    app
}

/// The index of the place `name` (its name, ignoring case, or its file name)
fn place_index(app: &MandelbrotApp, name: &str) -> Result<usize, Failure> {
    app.points.iter().position(|point| point.name.eq_ignore_ascii_case(name.trim()) || point.file_name == name.trim())
        .ok_or_else(|| failed(format!("there is no place {name:?}; the places are: {}", app.points.iter().map(|point| point.name.as_str()).collect::<Vec<_>>().join(", "))))
}

/// A resolution: 4K, 8K, 1080p, 1440p, 720p or `WIDTHxHEIGHT`
fn size_of(text: &str) -> Result<(usize, usize), Failure> {
    match text.trim().to_ascii_lowercase().as_str() {
        "4k" | "2160p" => Ok((3840, 2160)),
        "8k" | "4320p" => Ok((7680, 4320)),
        "1440p" => Ok((2560, 1440)),
        "1080p" => Ok((1920, 1080)),
        "720p" => Ok((1280, 720)),
        other => {
            let (width, height) = other.split_once('x').ok_or_else(|| usage(format!("--size {text:?}: use 4K, 1080p or WIDTHxHEIGHT")))?;
            Ok((width.trim().parse().map_err(|_| usage(format!("--size {text:?}: the width is not a number")))?, height.trim().parse().map_err(|_| usage(format!("--size {text:?}: the height is not a number")))?))
        }
    }
}

/// `video new`: makes a job of a flight to a place with the user's settings and these options, and writes its file
fn new_job(jobs: &Path, options: &[(String, String)]) -> Result<Value, Failure> {
    let to = option(options, "to").ok_or_else(|| usage("video new needs --to <place>"))?;
    let mut app = headless_app();
    let target = place_index(&app, to)?;
    //Where the flight starts: a place, or the starting view
    if let Some(from) = option(options, "from") {
        let index = place_index(&app, from)?;
        let pace = app.pace;
        app.pace = super::flight::Pace::NoFlight;
        app.go_to_point_of_interest(index);
        app.pace = pace;
    }
    if let Some(pace) = option(options, "pace") {
        app.pace = super::flight::Pace::ALL.into_iter().find(|known| known.name().eq_ignore_ascii_case(pace) && *known != super::flight::Pace::NoFlight)
            .ok_or_else(|| usage(format!("--pace {pace:?}: one of {}", super::flight::Pace::ALL.iter().filter(|pace| **pace != super::flight::Pace::NoFlight).map(|pace| pace.name()).collect::<Vec<_>>().join(", "))))?;
    } else if app.pace == super::flight::Pace::NoFlight {
        app.pace = super::flight::Pace::Cinematic;
    }
    if let Some(route) = option(options, "route") {
        app.route = super::flight::Route::ALL.into_iter().find(|known| known.name().eq_ignore_ascii_case(route))
            .ok_or_else(|| usage(format!("--route {route:?}: one of {}", super::flight::Route::ALL.iter().map(|route| route.name()).collect::<Vec<_>>().join(", "))))?;
    }
    app.go_to_point_of_interest(target);
    app.flight = None; //(Only its plan is wanted: the last flight)
    if let Some(size) = option(options, "size") {
        let (width, height) = size_of(size)?;
        app.save_size.choice = super::save_size::SaveSizeChoice::Custom;
        (app.save_size.custom_width, app.save_size.custom_height) = (width, height);
    }
    if let Some(fps) = number::<u32>(options, "fps")? {
        app.video_fps = fps;
    }
    if let Some(samples) = number::<u8>(options, "samples")? {
        app.image_supersampling_amount = samples.clamp(1, 64);
        app.image_quality_blur = false;
    }
    if let Some(shutter) = number::<f64>(options, "shutter")? {
        app.video_shutter = shutter.clamp(0.0, 360.0);
    }
    if let Some(quality) = option(options, "quality") {
        app.video_quality = super::video::VideoQuality::ALL.into_iter().find(|known| known.name().eq_ignore_ascii_case(quality))
            .ok_or_else(|| usage(format!("--quality {quality:?}: one of {}", super::video::VideoQuality::ALL.iter().map(|quality| quality.name()).collect::<Vec<_>>().join(", "))))?;
    }
    if let Some(speed) = number::<u16>(options, "speed")? {
        (app.video_speed, app.video_fit_length) = (speed, false);
    }
    if let Some(length) = option(options, "length") {
        //h:m:s, m:s or seconds
        let parts: Vec<&str> = length.split(':').collect();
        let mut boxes = [String::new(), String::new(), String::new()];
        for (slot, part) in boxes.iter_mut().rev().zip(parts.iter().rev()) {
            slot.clone_from(&(*part).to_string());
        }
        (app.video_length, app.video_fit_length) = (boxes, true);
        app.target_length().ok_or_else(|| usage(format!("--length {length:?}: use h:m:s, m:s or seconds, more than 0")))?;
    }
    if let Some(music) = option(options, "music") {
        if !Path::new(music).is_file() {
            return Err(failed(format!("the music {music:?} is not a file")));
        }
        (app.music_path, app.follow_music) = (music.to_string(), true);
        if let Some(strength) = number::<f64>(options, "strength")? {
            app.music_strength = strength.clamp(0.0, 1.0);
        }
    } else {
        app.follow_music = false;
    }
    let spec = app.video_spec().map_err(failed)?;
    let id = super::video::video_name(&spec.flight, &app.video_folder);
    let spec = VideoSpec { output: Some(id.clone()), ..spec };
    std::fs::create_dir_all(jobs).map_err(|err| failed(format!("{}: {err}", jobs.display())))?;
    let file = json!({ "format_version": FORMAT_VERSION, "id": id, "created": chrono::Local::now().to_rfc3339(), "spec": spec });
    std::fs::write(job_path(jobs, &id), serde_json::to_string_pretty(&file).unwrap_or_default()).map_err(|err| failed(format!("{}: {err}", job_path(jobs, &id).display())))?;
    let length = app.followed_music_length();
    let (frames, duration, _) = super::video::estimate(spec.flight.path().duration(), spec.fps, spec.speed, (spec.width, spec.height), spec.quality, spec.playback);
    let seconds = length.map_or(duration.as_secs_f64(), |length| length.as_secs_f64());
    let frames = length.map_or(frames, |length| (length.as_secs_f64() * f64::from(spec.fps)).ceil() as usize + 1);
    Ok(json!({ "id": id, "to": spec.flight.name, "frames": frames, "length_seconds": seconds, "width": spec.width, "height": spec.height, "fps": spec.fps,
        "samples": spec.samples, "music": spec.music.as_ref().map(|music| music.path.clone()), "output": output_path(&spec).display().to_string() }))
}

fn job_path(jobs: &Path, id: &str) -> PathBuf {
    jobs.join(format!("{id}.json"))
}

/// The spec of job `id`
fn read_job(jobs: &Path, id: &str) -> Result<VideoSpec, Failure> {
    let path = job_path(jobs, id);
    let text = std::fs::read_to_string(&path).map_err(|_| failed(format!("there is no job {id:?} (see mandelbrot video list)")))?;
    let file: Value = serde_json::from_str(&text).map_err(|err| failed(format!("{}: {err}", path.display())))?;
    serde_json::from_value(file["spec"].clone()).map_err(|err| failed(format!("{}: {err}", path.display())))
}

/// The file the video of `spec` is written to
fn output_path(spec: &VideoSpec) -> PathBuf {
    videos_folder().join(format!("{}.{}", spec.output.as_deref().unwrap_or_default(), super::video::extension(spec.quality)))
}

/// The files next to a job: its requests (pause, cancel) and the progress of its render
fn side_file(jobs: &Path, id: &str, what: &str) -> PathBuf {
    jobs.join(format!("{id}.{what}"))
}

/// The state of job `id` from its files: rendering (its progress file is fresh), finished, paused (chunks are kept), or new
fn state_of(jobs: &Path, id: &str, spec: &VideoSpec) -> (String, Value) {
    let progress: Value = std::fs::read_to_string(side_file(jobs, id, "progress.json")).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or(Value::Null);
    let fresh = std::fs::metadata(side_file(jobs, id, "progress.json")).and_then(|metadata| metadata.modified()).ok()
        .and_then(|modified| modified.elapsed().ok()).is_some_and(|age| age < REPORT_EVERY * 3) && progress["state"] == "rendering";
    let parts = super::chunks::parts_folder(&videos_folder(), id);
    let state = if fresh {
        "rendering"
    } else if output_path(spec).exists() && !parts.exists() {
        "finished"
    } else if parts.exists() {
        "paused"
    } else {
        "new"
    };
    (state.to_string(), progress)
}

/// `video start` (and `resume`): renders job `id` here, reporting its progress as JSON lines, until it is finished or paused
fn start(jobs: &Path, id: &str) -> Result<(), Failure> {
    let spec = read_job(jobs, id)?;
    if state_of(jobs, id, &spec).0 == "rendering" {
        return Err(failed(format!("the job {id} is being rendered already")));
    }
    let _ = (std::fs::remove_file(side_file(jobs, id, "pause")), std::fs::remove_file(side_file(jobs, id, "cancel")));
    let mut app = headless_app();
    if !app.start_video_of(spec) {
        return Err(failed(app.log.last().cloned().unwrap_or_else(|| String::from("the video could not start"))));
    }
    let (ctx, started) = (egui::Context::default(), Instant::now());
    let mut last_report = Instant::now().checked_sub(REPORT_EVERY).unwrap_or_else(Instant::now);
    let mut requested = None;
    while let Some(job) = &app.video_job {
        if last_report.elapsed() >= REPORT_EVERY {
            last_report = Instant::now();
            let (done, total) = (job.done.load(Ordering::Relaxed), job.total);
            let elapsed = started.elapsed().as_secs_f64();
            let left = if done > 0 { elapsed / done as f64 * (total.saturating_sub(done)) as f64 } else { f64::NAN };
            let report = json!({ "id": id, "state": "rendering", "frames_done": done, "frames": total, "elapsed_seconds": elapsed.round(), "seconds_left": if left.is_finite() { json!(left.round()) } else { Value::Null },
                "at_once": job.scheduler.allowed(), "cpu_percent": job.scheduler.cpu_percent.load(Ordering::Relaxed), "pid": std::process::id() });
            println!("{report}");
            let _ = std::fs::write(side_file(jobs, id, "progress.json"), report.to_string());
            //A request of video pause or video cancel from another command
            for what in ["cancel", "pause"] {
                if requested.is_none() && side_file(jobs, id, what).exists() {
                    requested = Some(what);
                    job.cancel.store(true, Ordering::Relaxed);
                }
            }
        }
        app.poll_video(&ctx);
        std::thread::sleep(Duration::from_millis(50));
    }
    let message = app.log.iter().rev().find(|line| line.starts_with("Video of the flight") || line.starts_with("Could not")).cloned().unwrap_or_default();
    let _ = (std::fs::remove_file(side_file(jobs, id, "pause")), std::fs::remove_file(side_file(jobs, id, "cancel")));
    if requested == Some("cancel") {
        let _ = std::fs::remove_dir_all(super::chunks::parts_folder(&videos_folder(), id));
    }
    let spec = read_job(jobs, id)?;
    //(The render's own progress file is done with: the state comes from the files of the video)
    let _ = std::fs::remove_file(side_file(jobs, id, "progress.json"));
    let state = if requested == Some("cancel") { String::from("cancelled") } else { state_of(jobs, id, &spec).0 };
    let report = json!({ "id": id, "state": state, "message": message, "output": output_path(&spec).display().to_string(), "elapsed_seconds": started.elapsed().as_secs_f64().round() });
    let _ = std::fs::write(side_file(jobs, id, "progress.json"), report.to_string());
    println!("{report}");
    if message.starts_with("Could not") { Err(failed(message)) } else { Ok(()) }
}

/// `video pause`: asks the render of job `id` to stop after its current chunk (it looks every few seconds)
fn request(jobs: &Path, id: &str, what: &str) -> Result<Value, Failure> {
    let spec = read_job(jobs, id)?;
    let (state, _) = state_of(jobs, id, &spec);
    if state != "rendering" {
        return Err(failed(format!("the job {id} is not being rendered (it is {state})")));
    }
    std::fs::write(side_file(jobs, id, what), "").map_err(|err| failed(err.to_string()))?;
    Ok(json!({ "id": id, "requested": what }))
}

/// `video cancel`: stops the render of job `id` if it runs and deletes its chunks
fn cancel(jobs: &Path, id: &str) -> Result<Value, Failure> {
    let spec = read_job(jobs, id)?;
    let (state, _) = state_of(jobs, id, &spec);
    if state == "rendering" {
        return request(jobs, id, "cancel");
    }
    let parts = super::chunks::parts_folder(&videos_folder(), id);
    let deleted = parts.exists();
    if deleted {
        std::fs::remove_dir_all(&parts).map_err(|err| failed(format!("{}: {err}", parts.display())))?;
    }
    Ok(json!({ "id": id, "state": if output_path(&spec).exists() { "finished" } else { "new" }, "deleted_chunks": deleted }))
}

/// `video status`: the state of job `id`, its frames and the last progress of its render
fn status(jobs: &Path, id: &str) -> Result<Value, Failure> {
    let spec = read_job(jobs, id)?;
    let (state, progress) = state_of(jobs, id, &spec);
    let manifest = super::chunks::read_manifest(&super::chunks::parts_folder(&videos_folder(), id));
    Ok(json!({ "id": id, "state": state, "to": spec.flight.name, "width": spec.width, "height": spec.height, "fps": spec.fps, "samples": spec.samples,
        "frames_kept": manifest.as_ref().map(super::chunks::Manifest::first_frame), "frames": manifest.as_ref().map(|manifest| manifest.total),
        "progress": progress, "output": output_path(&spec).display().to_string() }))
}

/// `video list`: every job, newest first, with its state
fn list(jobs: &Path) -> Value {
    let mut ids: Vec<String> = std::fs::read_dir(jobs).map(|entries| entries.filter_map(Result::ok).filter_map(|entry| entry.file_name().to_str()?.strip_suffix(".json").filter(|name| !name.ends_with(".progress")).map(str::to_string)).collect()).unwrap_or_default();
    ids.sort_by(|a, b| b.cmp(a));
    let jobs_list: Vec<Value> = ids.iter().filter_map(|id| read_job(jobs, id).ok().map(|spec| json!({ "id": id, "state": state_of(jobs, id, &spec).0, "to": spec.flight.name }))).collect();
    json!({ "jobs": jobs_list })
}

/// `video output`: the file of the video of job `id`, and whether it is there
fn output(jobs: &Path, id: &str) -> Result<Value, Failure> {
    let spec = read_job(jobs, id)?;
    let path = output_path(&spec);
    Ok(json!({ "id": id, "output": path.display().to_string(), "exists": path.exists() }))
}

/// `video delete`: removes job `id`, its chunks, and with `with_video` its video file. A job that is being rendered is refused
fn delete(jobs: &Path, id: &str, with_video: bool) -> Result<Value, Failure> {
    let spec = read_job(jobs, id)?;
    if state_of(jobs, id, &spec).0 == "rendering" {
        return Err(failed(format!("the job {id} is being rendered: pause or cancel it first")));
    }
    let _ = std::fs::remove_dir_all(super::chunks::parts_folder(&videos_folder(), id));
    let video = output_path(&spec);
    let deleted_video = with_video && video.exists() && std::fs::remove_file(&video).is_ok();
    for what in ["progress.json", "pause", "cancel"] {
        let _ = std::fs::remove_file(side_file(jobs, id, what));
    }
    std::fs::remove_file(job_path(jobs, id)).map_err(|err| failed(err.to_string()))?;
    Ok(json!({ "id": id, "deleted": true, "deleted_video": deleted_video }))
}

/// The whole file of job `id` (its spec, credit and uploads)
#[cfg(not(target_os = "android"))]
fn read_job_file(jobs: &Path, id: &str) -> Result<Value, Failure> {
    read_job(jobs, id)?;
    let path = job_path(jobs, id);
    std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str(&text).ok()).ok_or_else(|| failed(format!("{} cannot be read", path.display())))
}

#[cfg(not(target_os = "android"))]
fn write_job_file(jobs: &Path, id: &str, file: &Value) -> Result<(), Failure> {
    let path = job_path(jobs, id);
    std::fs::write(&path, serde_json::to_string_pretty(file).unwrap_or_default()).map_err(|err| failed(format!("{}: {err}", path.display())))
}

/// `video credit`: records the exact credit of the music of job `id` (see `youtube::credit_job`)
#[cfg(not(target_os = "android"))]
fn credit(jobs: &Path, id: &str, options: &[(String, String)]) -> Result<Value, Failure> {
    let kind = option(options, "kind").ok_or_else(|| usage(format!("video credit needs --kind ({})", super::youtube::CREDIT_KINDS.join(", "))))?;
    let text = match (option(options, "text-file"), option(options, "text")) {
        (Some(file), _) => std::fs::read_to_string(file).map_err(|err| failed(format!("{file}: {err}")))?,
        (None, Some(text)) => text.to_string(),
        (None, None) if kind == "own" => String::new(),
        (None, None) => return Err(usage("video credit needs --text-file <file> (or --text \"…\") with the exact credit the license asks for")),
    };
    let mut file = read_job_file(jobs, id)?;
    let code = super::youtube::credit_job(&mut file, id, kind, &text, &super::youtube::used_codes().map_err(failed)?).map_err(failed)?;
    write_job_file(jobs, id, &file)?;
    if let Some(code) = &code {
        super::youtube::record_code(code, id).map_err(failed)?;
    }
    Ok(json!({ "id": id, "credit": file["credit"] }))
}

/// `video upload`: uploads the finished video of job `id`, private, with its credit under the description; the upload is kept with the job
#[cfg(not(target_os = "android"))]
fn upload(jobs: &Path, id: &str, options: &[(String, String)]) -> Result<Value, Failure> {
    let spec = read_job(jobs, id)?;
    let mut file = read_job_file(jobs, id)?;
    let video = output_path(&spec);
    if state_of(jobs, id, &spec).0 != "finished" {
        return Err(failed(format!("the video of {id} is not finished (see mandelbrot video status {id})")));
    }
    let credit = file["credit"]["text"].as_str().map(str::to_string);
    if spec.music.is_some() && file["credit"].is_null() {
        return Err(failed(format!("the video {id} has music without a credit: record it first with mandelbrot video credit {id} --kind … --text-file …")));
    }
    let title = option(options, "title").map_or_else(|| spec.flight.name.clone(), str::to_string);
    let description = match option(options, "description-file") {
        Some(path) => std::fs::read_to_string(path).map_err(|err| failed(format!("{path}: {err}")))?,
        None => option(options, "description").unwrap_or_default().to_string(),
    };
    let description = super::youtube::description_with_credit(&description, credit.as_deref());
    let video_id = super::youtube::upload(&video, &title, &description).map_err(failed)?;
    let record = json!({ "video_id": video_id, "url": format!("https://youtu.be/{video_id}"), "privacy": "private", "title": title, "uploaded": chrono::Local::now().to_rfc3339() });
    if !file["uploads"].is_array() {
        file["uploads"] = json!([]);
    }
    if let Some(uploads) = file["uploads"].as_array_mut() {
        uploads.push(record.clone());
    }
    write_job_file(jobs, id, &file)?;
    Ok(json!({ "id": id, "upload": record, "kept": video.display().to_string() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(arguments: &[&str]) -> Vec<String> {
        arguments.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn options_sizes_and_numbers_are_read_and_mistakes_are_usage_errors() {
        let read = options(&strings(&["--to", "Fractal Flower", "--size", "4K", "--fps", "60"])).ok().unwrap();
        assert_eq!(option(&read, "to"), Some("Fractal Flower"));
        assert_eq!(number::<u32>(&read, "fps").ok().flatten(), Some(60));
        assert!(options(&strings(&["--to"])).err().is_some_and(|failure| failure.usage && failure.message.contains("--to needs a value")));
        assert!(options(&strings(&["Fractal"])).err().is_some_and(|failure| failure.usage));
        assert!(number::<u32>(&options(&strings(&["--fps", "lots"])).ok().unwrap(), "fps").err().is_some_and(|failure| failure.message.contains("not a number")));
        assert_eq!(size_of("4K").ok(), Some((3840, 2160)));
        assert_eq!(size_of("1080p").ok(), Some((1920, 1080)));
        assert_eq!(size_of("1080x2400").ok(), Some((1080, 2400)));
        assert!(size_of("big").is_err());
    }

    #[test]
    fn unknown_commands_and_missing_ids_exit_with_usage_errors() {
        assert_eq!(command_line(&strings(&["video", "dance"])), 2);
        assert_eq!(command_line(&strings(&["paint", "it"])), 2);
        assert_eq!(command_line(&strings(&["video"])), 2);
        assert_eq!(command_line(&strings(&["video", "status"])), 2, "which job?");
        assert_eq!(command_line(&strings(&["video", "status", "no-such-job-at-all"])), 1, "a job that does not exist fails");
        QUIET.store(false, Ordering::Relaxed);
    }
}

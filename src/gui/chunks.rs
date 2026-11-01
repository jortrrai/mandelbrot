//! Restartable videos (issue #182): a video is written as finished pieces (chunks) of `CHUNK_SECONDS` of video, each a complete file of its own, with a manifest
//! that says which are done, in a folder `saved/<video>.parts`. Whatever stops the render (*Cancel video*, closing the program, a crash, a reboot), the chunks
//! that were finished stay; *Resume* continues with the first chunk that is not, with the same settings and the same encoder. When the last chunk is done they
//! are joined into the video file without encoding them again (ffmpeg's concat demuxer), and the folder is removed.
//!
//! Every chunk starts with a keyframe and has its own closed groups of pictures, and all are encoded with the same arguments, so they join seamlessly. With
//! *Blend frames* a resumed render first renders `WARM_UP_FRAMES` frames before the chunk again (not written), so the blend continues as if it had not stopped.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::video::VideoSpec;

/// The length of a chunk, in seconds of video
pub(super) const CHUNK_SECONDS: usize = 10;

/// The container of the chunks: Matroska, which stores the timestamps of the frames as they are, so the chunks join without gaps. MP4 files with B-frames
/// have an edit list that shifts them, which joining did not take into account: a constant-rate decode of a joined H.264 video showed 2 frames too many
pub(super) const CHUNK_EXTENSION: &str = "mkv";

/// The frames a resumed render with *Blend frames* renders again before its first chunk: the blend follows the view within about 10 frames (a new frame weighs
/// at least 10%), so after 60 frames what came before weighs less than 0.2%
pub(super) const WARM_UP_FRAMES: usize = 60;

/// The version of the manifest that this program writes
const FORMAT_VERSION: u32 = 1;

/// The manifest of a video under way: what it is, how it is cut into chunks, the encoder, and how many chunks are done
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Manifest {
    pub(super) format_version: u32,
    pub(super) spec: VideoSpec,
    pub(super) total: usize,
    pub(super) chunk_frames: usize,
    pub(super) nvenc: bool,
    pub(super) chunks_done: usize,
}

impl Manifest {
    pub(super) fn new(spec: VideoSpec, total: usize, chunk_frames: usize, nvenc: bool) -> Manifest {
        Manifest { format_version: FORMAT_VERSION, spec, total, chunk_frames: chunk_frames.max(1), nvenc, chunks_done: 0 }
    }

    /// The chunks of the whole video
    pub(super) fn chunks(&self) -> usize {
        self.total.div_ceil(self.chunk_frames)
    }

    /// The first frame that is not in a finished chunk
    pub(super) fn first_frame(&self) -> usize {
        (self.chunks_done * self.chunk_frames).min(self.total)
    }
}

/// The folder of the chunks of video `name` in `saved`
pub(super) fn parts_folder(saved: &Path, name: &str) -> PathBuf {
    saved.join(format!("{name}.parts"))
}

/// The file of chunk `chunk` in `parts`
pub(super) fn chunk_path(parts: &Path, chunk: usize, extension: &str) -> PathBuf {
    parts.join(format!("part-{chunk:05}.{extension}"))
}

fn manifest_path(parts: &Path) -> PathBuf {
    parts.join("manifest.json")
}

/// The manifest in `parts`, if there is a readable one
pub(super) fn read_manifest(parts: &Path) -> Option<Manifest> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(manifest_path(parts)).ok()?).ok()?;
    (manifest.format_version <= FORMAT_VERSION).then_some(manifest)
}

/// Writes `manifest` to `parts`: to a new file that then replaces the old one, so a crash never leaves half a manifest
pub(super) fn write_manifest(parts: &Path, manifest: &Manifest) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(manifest).expect("a manifest is plain text and numbers") + "\n";
    let temporary = parts.join("manifest.json.new");
    std::fs::write(&temporary, text)?;
    std::fs::rename(temporary, manifest_path(parts))
}

/// Joins the chunks `0..chunks` of `parts` into `video` (an .mp4 or an .mkv) without encoding them again (`hevc`: they are HEVC); true if ffmpeg did
pub(super) fn join_chunks(parts: &Path, chunks: usize, video: &Path, hevc: bool) -> bool {
    let list = (0..chunks).fold(String::new(), |mut list, chunk| {
        //Only the name of the file: ffmpeg reads a relative path in the list from the folder of the list, which is `parts` itself, so the folder in front
        //of a relative `parts` (such as saved/<name>.parts) would be there twice
        let path = chunk_path(parts, chunk, CHUNK_EXTENSION);
        let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let _ = writeln!(list, "file '{}'", name.replace('\'', "'\\''"));
        list
    });
    let list_path = parts.join("chunks.txt");
    if std::fs::write(&list_path, list).is_err() {
        return false;
    }
    let mut arguments: Vec<String> = ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i"].map(String::from).to_vec();
    arguments.push(list_path.display().to_string());
    arguments.extend(["-map", "0", "-c", "copy"].map(String::from));
    if video.extension().is_some_and(|extension| extension == "mp4") {
        //The options of MP4 files that the chunks do not have: the tag of HEVC that Apple's players need, and the index at the start
        if hevc {
            arguments.extend(["-tag:v:0", "hvc1"].map(String::from));
        }
        arguments.extend(["-movflags", "+faststart"].map(String::from));
    }
    arguments.push(video.display().to_string());
    std::process::Command::new("ffmpeg").args(arguments).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .status().is_ok_and(|status| status.success())
}

/// The videos in `saved` that were stopped before they were finished (or whose chunks could not be joined): their names (without `.parts`) and manifests,
/// the newest first (a name starts with the time the video was started)
pub(super) fn unfinished(saved: &Path) -> Vec<(String, Manifest)> {
    let Ok(entries) = std::fs::read_dir(saved) else { return Vec::new() };
    let mut videos: Vec<(String, Manifest)> = entries.filter_map(|entry| {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_str()?.strip_suffix(".parts")?.to_string();
        Some((name, read_manifest(&path)?))
    }).collect();
    videos.sort_by(|a, b| b.0.cmp(&a.0));
    videos
}

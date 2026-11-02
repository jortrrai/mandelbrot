//! Uploading a finished video of the command line to YouTube, always private (issue #216), with the music credits it needs.
//!
//! * `mandelbrot youtube auth --client client_secret.json`: once, with the OAuth client of the user's own Google Cloud project (YouTube Data API v3, a
//!   *Desktop* client): opens the consent page, receives the answer on a local port, and keeps the refresh token in the configuration folder (readable only
//!   by the user, never in the repository).
//! * `mandelbrot video credit <id> --kind bensound|youtube-audio-library|own|other --text-file credit.txt`: records the exact credit text of the music of
//!   that video. A Bensound credit must have its license code (Bensound gives one per video through its own download flow; it is never made up here), and a
//!   code already recorded for another video is refused. The record stays with the job, so uploading it again uses the same credit.
//! * `mandelbrot video upload <id> --title "…" [--description-file d.txt]`: uploads the finished video with `privacyStatus: private` (there is no option for
//!   anything else: publishing is the user's choice, in YouTube Studio), the description followed by the credit. A video with music and no credit is refused.
//!   The local video is never touched; the upload's id and link are kept with the job.
//!
//! Not tried against YouTube yet: it needs the user's Google project and consent.

use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// The scope that allows uploads (and nothing else on the channel)
const SCOPE: &str = "https://www.googleapis.com/auth/youtube.upload";

/// The kinds of music credits
pub(super) const CREDIT_KINDS: [&str; 4] = ["bensound", "youtube-audio-library", "own", "other"];

/// The folder of the client and the token: next to the settings
fn config_folder() -> Result<PathBuf, String> {
    super::settings::settings_path().and_then(|path| path.parent().map(Path::to_path_buf)).ok_or_else(|| String::from("the configuration folder is not known (no $HOME)"))
}

/// Writes `text` to `path`, readable and writable only by the user (Unix)
fn write_private(path: &Path, text: &str) -> Result<(), String> {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder).map_err(|err| format!("{}: {err}", folder.display()))?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|err| format!("{}: {err}", path.display()))?;
    file.write_all(text.as_bytes()).map_err(|err| format!("{}: {err}", path.display()))
}

/// `value` percent-encoded for a URL or a form
pub(super) fn encoded(value: &str) -> String {
    value.bytes().map(|byte| if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) { char::from(byte).to_string() } else { format!("%{byte:02X}") }).collect()
}

/// The client id and secret of a Google OAuth client file (`installed` or `web`)
fn client_of(text: &str) -> Result<(String, String), String> {
    let file: Value = serde_json::from_str(text).map_err(|err| format!("the client file is not JSON: {err}"))?;
    let client = file.get("installed").or_else(|| file.get("web")).ok_or("the client file has no \"installed\" client (make a Desktop OAuth client)")?;
    let field = |name: &str| client[name].as_str().map(str::to_string).ok_or_else(|| format!("the client file has no {name}"));
    Ok((field("client_id")?, field("client_secret")?))
}

/// `youtube auth`: asks for consent in the browser and keeps the refresh token
pub(super) fn authorize(client_file: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(client_file).map_err(|err| format!("{}: {err}", client_file.display()))?;
    let (client_id, client_secret) = client_of(&text)?;
    let folder = config_folder()?;
    write_private(&folder.join("youtube-client.json"), &text)?;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|err| format!("no local port for the answer: {err}"))?;
    let redirect = format!("http://127.0.0.1:{}", listener.local_addr().map_err(|err| err.to_string())?.port());
    let url = format!("https://accounts.google.com/o/oauth2/v2/auth?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent",
        encoded(&client_id), encoded(&redirect), encoded(SCOPE));
    eprintln!("Open this page to allow the uploads (it should open by itself):\n{url}");
    let _ = std::process::Command::new("xdg-open").arg(&url).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn();
    let (stream, _) = listener.accept().map_err(|err| err.to_string())?;
    let mut request = String::new();
    std::io::BufReader::new(&stream).read_line(&mut request).map_err(|err| err.to_string())?;
    let mut stream = stream;
    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nDone: the Mandelbrot program may upload private videos now. You can close this page.");
    let code = query_value(&request, "code").ok_or_else(|| format!("the consent page answered without a code: {}", query_value(&request, "error").unwrap_or_default()))?;
    let answer = post_form("https://oauth2.googleapis.com/token", &[("code", &code), ("client_id", &client_id), ("client_secret", &client_secret), ("redirect_uri", &redirect), ("grant_type", "authorization_code")])?;
    let refresh = answer["refresh_token"].as_str().ok_or("Google gave no refresh token")?;
    write_private(&folder.join("youtube-token.json"), &json!({ "refresh_token": refresh }).to_string())?;
    Ok(json!({ "authorized": true, "scope": SCOPE }))
}

/// The value of `name` in the query of the first line of an HTTP request (`GET /?code=…&… HTTP/1.1`), decoded
pub(super) fn query_value(request: &str, name: &str) -> Option<String> {
    let target = request.split_whitespace().nth(1)?;
    let query = target.split_once('?')?.1;
    query.split('&').filter_map(|pair| pair.split_once('=')).find(|(key, _)| *key == name).map(|(_, value)| decoded(value))
}

fn decoded(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                if let Some(byte) = value.get(index + 1..index + 3).and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
                    out.push(byte);
                    index += 3;
                    continue;
                }
                out.push(b'%');
            }
            b'+' => out.push(b' '),
            byte => out.push(byte),
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Posts a form and reads its JSON answer
fn post_form(url: &str, fields: &[(&str, &str)]) -> Result<Value, String> {
    let response = ureq::post(url).send_form(fields).map_err(|err| format!("{url}: {err}"))?;
    serde_json::from_str(&response.into_string().map_err(|err| err.to_string())?).map_err(|err| err.to_string())
}

/// A fresh access token from the kept refresh token
fn access_token() -> Result<String, String> {
    let folder = config_folder()?;
    let client = std::fs::read_to_string(folder.join("youtube-client.json")).map_err(|_| String::from("not authorized yet: run mandelbrot youtube auth --client <client_secret.json>"))?;
    let (client_id, client_secret) = client_of(&client)?;
    let token: Value = serde_json::from_str(&std::fs::read_to_string(folder.join("youtube-token.json")).map_err(|_| String::from("not authorized yet: run mandelbrot youtube auth"))?).map_err(|err| err.to_string())?;
    let refresh = token["refresh_token"].as_str().ok_or("the kept token has no refresh token: run mandelbrot youtube auth again")?;
    let answer = post_form("https://oauth2.googleapis.com/token", &[("refresh_token", refresh), ("client_id", &client_id), ("client_secret", &client_secret), ("grant_type", "refresh_token")])?;
    answer["access_token"].as_str().map(str::to_string).ok_or_else(|| String::from("Google gave no access token"))
}

/// The license code in a Bensound credit ("License code: XXXXXXXX"), if it has one
pub(super) fn bensound_code(credit: &str) -> Option<String> {
    let lower = credit.to_lowercase();
    let at = lower.find("license code")?;
    let rest = &credit[at + "license code".len()..];
    let code: String = rest.trim_start_matches([':', ' ', '\t']).chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
    (code.len() >= 4).then_some(code)
}

/// The Bensound license codes recorded so far (code → job id), kept in the configuration folder so a deleted job does not free its code
fn codes_path() -> Result<PathBuf, String> {
    Ok(config_folder()?.join("bensound-codes.json"))
}

pub(super) fn used_codes() -> Result<Value, String> {
    let path = codes_path()?;
    Ok(std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_else(|| json!({})))
}

pub(super) fn record_code(code: &str, id: &str) -> Result<(), String> {
    let mut codes = used_codes()?;
    codes[code] = json!(id);
    write_private(&codes_path()?, &serde_json::to_string_pretty(&codes).map_err(|err| err.to_string())?)
}

/// `video credit`: records the credit of the music of `job` (its JSON) as `kind` with the exact `text`, checking a Bensound code against `used` (code → job id).
/// Returns the job with its credit, and the code to record as used
pub(super) fn credit_job(job: &mut Value, id: &str, kind: &str, text: &str, used: &Value) -> Result<Option<String>, String> {
    if !CREDIT_KINDS.contains(&kind) {
        return Err(format!("--kind {kind:?}: one of {}", CREDIT_KINDS.join(", ")));
    }
    let text = text.trim();
    if text.is_empty() && kind != "own" {
        return Err(String::from("the credit text is empty: give the exact credit the music's license asks for"));
    }
    let code = if kind == "bensound" {
        let code = bensound_code(text).ok_or("a Bensound credit needs its license code (\"License code: …\"), which Bensound gives for each video when the track is downloaded")?;
        if let Some(other) = used.get(&code).and_then(Value::as_str).filter(|other| *other != id) {
            return Err(format!("the Bensound license code {code} was used for the video {other} already: every video needs its own code from Bensound"));
        }
        Some(code)
    } else {
        None
    };
    job["credit"] = json!({ "kind": kind, "text": text, "code": code, "recorded": chrono::Local::now().to_rfc3339() });
    Ok(code)
}

/// The description of an upload: `description`, then the credit of the music (if any), separated by an empty line
pub(super) fn description_with_credit(description: &str, credit: Option<&str>) -> String {
    match credit.map(str::trim).filter(|credit| !credit.is_empty()) {
        Some(credit) if description.trim().is_empty() => credit.to_string(),
        Some(credit) => format!("{}\n\n{credit}", description.trim_end()),
        None => description.to_string(),
    }
}

/// `video upload`: uploads `video` privately with `title` and `description` (resumable: a session first, then the file in one stream). Returns the YouTube
/// video id
pub(super) fn upload(video: &Path, title: &str, description: &str) -> Result<String, String> {
    let token = access_token()?;
    let size = std::fs::metadata(video).map_err(|err| format!("{}: {err}", video.display()))?.len();
    //Always private: publishing is the user's own choice, in YouTube Studio
    let metadata = json!({ "snippet": { "title": title, "description": description, "categoryId": "28" }, "status": { "privacyStatus": "private", "selfDeclaredMadeForKids": false } });
    let session = ureq::post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status")
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/json; charset=UTF-8")
        .set("X-Upload-Content-Length", &size.to_string())
        .set("X-Upload-Content-Type", "video/*")
        .send_string(&metadata.to_string())
        .map_err(|err| format!("YouTube refused the upload: {err}"))?;
    let location = session.header("location").ok_or("YouTube gave no upload address")?.to_string();
    let file = std::fs::File::open(video).map_err(|err| format!("{}: {err}", video.display()))?;
    let answer = ureq::put(&location)
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Length", &size.to_string())
        .set("Content-Type", "video/*")
        .send(std::io::BufReader::new(file).take(size))
        .map_err(|err| format!("the upload failed (the video is kept as it is; run video upload again): {err}"))?;
    let answer: Value = serde_json::from_str(&answer.into_string().map_err(|err| err.to_string())?).map_err(|err| err.to_string())?;
    answer["id"].as_str().map(str::to_string).ok_or_else(|| String::from("YouTube answered without a video id"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credits_are_exact_never_reused_and_kept_for_the_same_video() {
        let snippet = "Music: \"Lofi Morning\" by Bensound.com\nLicense code: ABCD1234EFGH";
        assert_eq!(bensound_code(snippet).as_deref(), Some("ABCD1234EFGH"));
        assert_eq!(bensound_code("Music by Bensound.com"), None, "no code: refused below");
        let (mut job, used) = (json!({}), json!({}));
        assert_eq!(credit_job(&mut job, "video-1", "bensound", snippet, &used).unwrap().as_deref(), Some("ABCD1234EFGH"));
        assert_eq!(job["credit"]["text"], snippet, "the exact text");
        //Another video with the same code is refused; the same video again (a retried upload) keeps its record
        let used = json!({ "ABCD1234EFGH": "video-1" });
        assert!(credit_job(&mut json!({}), "video-2", "bensound", snippet, &used).unwrap_err().contains("was used for the video video-1"));
        assert!(credit_job(&mut job, "video-1", "bensound", snippet, &used).is_ok());
        assert!(credit_job(&mut json!({}), "video-3", "bensound", "Music by Bensound.com", &used).unwrap_err().contains("needs its license code"));
        assert!(credit_job(&mut json!({}), "video-3", "free-for-all", "x", &used).is_err());
        assert!(credit_job(&mut json!({}), "video-3", "youtube-audio-library", "", &used).is_err(), "an empty credit is refused");
        assert_eq!(description_with_credit("A dive into Fractal Flower.", Some(snippet)), format!("A dive into Fractal Flower.\n\n{snippet}"));
        assert_eq!(description_with_credit("", None), "");
    }

    #[test]
    fn the_answer_of_the_consent_page_and_urls_are_read_and_written() {
        assert_eq!(query_value("GET /?code=4%2F0Ab_x&scope=https%3A%2F%2Fwww.googleapis.com HTTP/1.1", "code").as_deref(), Some("4/0Ab_x"));
        assert_eq!(query_value("GET /?error=access_denied HTTP/1.1", "code"), None);
        assert_eq!(encoded("http://127.0.0.1:8080"), "http%3A%2F%2F127.0.0.1%3A8080");
        assert_eq!(client_of(r#"{"installed":{"client_id":"id","client_secret":"secret"}}"#), Ok((String::from("id"), String::from("secret"))));
        assert!(client_of("{}").is_err());
    }
}

//! The Mandelbrot app on Android (issue #219): the activity starts the same app as the desktop window (see `mandelbrot::run_on_android`). Only built for
//! Android; on other systems this crate is empty.

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    if let Err(err) = mandelbrot::run_on_android(app) {
        eprintln!("The Mandelbrot app could not run: {err}");
    }
}

//! The text printed to the terminal when the application starts (the terminal is only for developers).

/// The version printed in the banner: the version of the package, so there is one place to change it (`Cargo.toml`)
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The ASCII art of the application name, made with <https://patorjk.com/software/taag/#p=display&f=Big&t=Mandelbrot>
const APPLICATION_BANNER: &str = r"
__  __                 _      _ _               _   
|  \/  |               | |    | | |             | |  
| \  / | __ _ _ __   __| | ___| | |__  _ __ ___ | |_ 
| |\/| |/ _` | '_ \ / _` |/ _ \ | '_ \| '__/ _ \| __|
| |  | | (_| | | | | (_| |  __/ | |_) | | | (_) | |_ 
|_|  |_|\__,_|_| |_|\__,_|\___|_|_.__/|_|  \___/ \__|";

/// The credit: "by Jort" (<https://patorjk.com/software/taag/#p=display&f=Small%20Slant&t=by%20Jort>) and a little robot for the AI that helped to write the code
const AUTHOR_BANNER: &str = r"
   __             __         __      .---.
  / /  __ __  __ / /__  ____/ /_    [ o o ]   and his robotic friends:
 / _ \/ // / / // / _ \/ __/ __/     '-v-'    Claude Code, beep boop
/_.__/\_, /  \___/\___/_/  \__/       /|_|\
     /___/";

/// The banner: the application name, the credit and the version
pub fn banner_text() -> String {
    format!("{APPLICATION_BANNER}{AUTHOR_BANNER}\n                                 v{VERSION}\n\n")
}

/// Prints the banner
pub fn print_banner() {
    println!("{}", banner_text());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_banner_shows_the_version_of_the_package_and_credits_the_robot() {
        let banner = banner_text();
        assert!(banner.contains(&format!("v{}", env!("CARGO_PKG_VERSION"))));
        assert!(banner.contains("and his robotic friends") && banner.contains("Claude Code"));
        assert!(banner.contains("__  __") && banner.contains(".---."));
    }

    #[test]
    fn the_readme_has_the_same_logo_and_version_as_the_program() {
        let readme = include_str!("../README.md");
        assert!(readme.contains(&format!("v{}", env!("CARGO_PKG_VERSION"))), "update the version in the logo of README.md");
        for line in AUTHOR_BANNER.lines().skip(1) {
            assert!(readme.contains(line.trim_end()), "the README logo differs from the banner at: {line:?}");
        }
    }
}

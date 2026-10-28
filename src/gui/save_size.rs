//! The size of the saved image: the choice in the window (the command line size, the size of the view, a preset or a custom size) and the custom presets that the user keeps.

use std::path::PathBuf;

use crate::resolution::{self, Resolution, ResolutionPreset};

/// What the "Saved image size" list has selected
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SaveSizeChoice {
    /// The image size of the command line (the window size unless the window is scaled)
    CommandLine,
    /// The size of the view that is shown in the window
    View,
    /// A built-in or custom preset, by name
    Preset(String),
    /// The width and height that were typed in
    Custom,
}

/// The choice of the saved image size, the typed-in values and the custom presets
#[derive(Debug)]
pub(super) struct SaveSize {
    /// What is selected in the list
    pub choice: SaveSizeChoice,
    /// The typed-in width, used when the choice is `Custom`
    pub custom_width: usize,
    /// The typed-in height, used when the choice is `Custom`
    pub custom_height: usize,
    /// The typed-in name for a new preset
    pub preset_name: String,
    /// The presets of the user, see `resolution`
    pub custom_presets: Vec<ResolutionPreset>,
    /// The file the custom presets are kept in, if the configuration folder of the user is known
    pub presets_path: Option<PathBuf>,
}

impl SaveSize {
    /// The starting state: the command line size selected, and the custom presets loaded from `presets_path`.
    /// Returns the messages about presets that could not be used.
    pub fn new(command_line: Resolution, presets_path: Option<PathBuf>) -> (SaveSize, Vec<String>) {
        let (custom_presets, warnings) = presets_path.as_deref().map_or((Vec::new(), Vec::new()), resolution::load_custom_presets);
        let size = SaveSize { choice: SaveSizeChoice::CommandLine, custom_width: command_line.width, custom_height: command_line.height, preset_name: String::new(), custom_presets, presets_path };
        (size, warnings)
    }

    /// The built-in presets followed by the custom ones
    pub fn all_presets(&self) -> Vec<ResolutionPreset> {
        let mut presets = resolution::built_in_presets();
        presets.extend(self.custom_presets.iter().cloned());
        presets
    }

    /// Whether the preset `name` is one of the custom presets
    pub fn is_custom_preset(&self, name: &str) -> bool {
        self.custom_presets.iter().any(|preset| preset.name == name)
    }

    /// The resolution that the choice stands for, given the command line size and the size of the view. A preset that has disappeared, or a size that is not allowed, is an error.
    /// # Errors
    /// The reason, as a sentence for the status log
    pub fn resolve(&self, command_line: Resolution, view: Resolution) -> Result<Resolution, String> {
        let resolution = match &self.choice {
            SaveSizeChoice::CommandLine => command_line,
            SaveSizeChoice::View => view,
            SaveSizeChoice::Preset(name) => self.all_presets().into_iter().find(|preset| &preset.name == name).map(|preset| preset.resolution).ok_or_else(|| format!("The resolution preset \"{name}\" does not exist"))?,
            SaveSizeChoice::Custom => Resolution::new(self.custom_width, self.custom_height),
        };
        resolution.validate()
    }

    /// The text of the closed list: what is selected, with its size
    pub fn selected_text(&self, command_line: Resolution, view: Resolution) -> String {
        match &self.choice {
            SaveSizeChoice::CommandLine => format!("Command line ({command_line})"),
            SaveSizeChoice::View => format!("Same as the view ({view})"),
            SaveSizeChoice::Preset(name) => self.all_presets().iter().find(|preset| &preset.name == name).map_or_else(|| format!("{name} (missing)"), ResolutionPreset::label),
            SaveSizeChoice::Custom => format!("Custom ({})", Resolution::new(self.custom_width, self.custom_height)),
        }
    }

    /// Adds the typed-in width and height as a custom preset with the typed-in name (or replaces the size of the one with that name), selects it and keeps the presets in the file.
    /// Returns the message for the status log.
    /// # Errors
    /// The reason, if the name or the size is not allowed, or the file could not be written (the preset is then still there for this session)
    pub fn add_preset(&mut self) -> Result<String, String> {
        let name = self.preset_name.trim().to_string();
        resolution::add_custom_preset(&mut self.custom_presets, &name, Resolution::new(self.custom_width, self.custom_height))?;
        self.choice = SaveSizeChoice::Preset(name.clone());
        self.preset_name.clear();
        let stored = self.persist()?;
        Ok(format!("Saved the resolution preset \"{name}\" ({} x {}){stored}", self.custom_width, self.custom_height))
    }

    /// Removes the custom preset `name` and keeps the presets in the file; if it was selected, the command line size is selected again.
    /// Returns the message for the status log.
    /// # Errors
    /// The reason, if there is no such custom preset or the file could not be written
    pub fn delete_preset(&mut self, name: &str) -> Result<String, String> {
        let before = self.custom_presets.len();
        self.custom_presets.retain(|preset| preset.name != name);
        if self.custom_presets.len() == before {
            return Err(format!("There is no custom resolution preset \"{name}\""));
        }
        if self.choice == SaveSizeChoice::Preset(name.to_string()) {
            self.choice = SaveSizeChoice::CommandLine;
        }
        let stored = self.persist()?;
        Ok(format!("Deleted the resolution preset \"{name}\"{stored}"))
    }

    /// Writes the custom presets to the file. Returns the sentence to append to a message: where they were kept, or that they are only kept until the program ends.
    fn persist(&self) -> Result<String, String> {
        self.presets_path.as_ref().map_or_else(
            || Ok(String::from(" (only until the program ends: the configuration folder is not known)")),
            |path| resolution::save_custom_presets(path, &self.custom_presets).map(|()| format!(" in {}", path.display())).map_err(|err| format!("The custom resolution presets could not be written to {}: {err} (they stay until the program ends)", path.display())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMAND_LINE: Resolution = Resolution::new(1200, 800);
    const VIEW: Resolution = Resolution::new(900, 600);

    fn temporary_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mandelbrot-save-size-{name}-{}", std::process::id())).join(resolution::FILE_NAME)
    }

    #[test]
    fn the_choice_stands_for_a_resolution() {
        let (mut size, warnings) = SaveSize::new(COMMAND_LINE, None);
        assert_eq!(warnings, Vec::<String>::new());
        assert_eq!(size.resolve(COMMAND_LINE, VIEW), Ok(COMMAND_LINE), "the command line size is selected at the start");
        size.choice = SaveSizeChoice::View;
        assert_eq!(size.resolve(COMMAND_LINE, VIEW), Ok(VIEW));
        size.choice = SaveSizeChoice::Preset(String::from("4K UHD"));
        assert_eq!(size.resolve(COMMAND_LINE, VIEW), Ok(Resolution::new(3840, 2160)));
        assert_eq!(size.selected_text(COMMAND_LINE, VIEW), "4K UHD (3840 x 2160)");
        size.choice = SaveSizeChoice::Preset(String::from("Gone"));
        assert!(size.resolve(COMMAND_LINE, VIEW).unwrap_err().contains("does not exist"));
        size.choice = SaveSizeChoice::Custom;
        (size.custom_width, size.custom_height) = (3000, 2000);
        assert_eq!(size.resolve(COMMAND_LINE, VIEW), Ok(Resolution::new(3000, 2000)));
        assert_eq!(size.selected_text(COMMAND_LINE, VIEW), "Custom (3000 x 2000)");
        size.custom_width = 5;
        assert!(size.resolve(COMMAND_LINE, VIEW).is_err(), "a typed-in size that is not allowed is refused");
    }

    #[test]
    fn a_custom_preset_is_added_selected_kept_in_the_file_and_loaded_by_the_next_run() {
        let path = temporary_path("add");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let (mut size, _) = SaveSize::new(COMMAND_LINE, Some(path.clone()));
        (size.custom_width, size.custom_height, size.preset_name) = (6000, 4000, String::from("  Poster "));
        let message = size.add_preset().unwrap();
        assert!(message.starts_with("Saved the resolution preset \"Poster\" (6000 x 4000) in "), "{message}");
        assert_eq!(size.choice, SaveSizeChoice::Preset(String::from("Poster")));
        assert!(size.preset_name.is_empty() && size.is_custom_preset("Poster") && !size.is_custom_preset("4K UHD"));
        assert_eq!(size.all_presets().len(), 11);
        let (next_run, warnings) = SaveSize::new(COMMAND_LINE, Some(path.clone()));
        assert_eq!(warnings, Vec::<String>::new());
        assert_eq!(next_run.custom_presets, size.custom_presets);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_preset_that_is_refused_changes_nothing_and_a_deleted_one_goes_back_to_the_command_line_size() {
        let path = temporary_path("delete");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let (mut size, _) = SaveSize::new(COMMAND_LINE, Some(path.clone()));
        size.preset_name = String::from("4k uhd");
        assert!(size.add_preset().unwrap_err().contains("built-in"));
        size.preset_name = String::from("Poster");
        size.add_preset().unwrap();
        assert!(size.delete_preset("Nothing").is_err());
        assert!(size.delete_preset("Poster").unwrap().starts_with("Deleted the resolution preset \"Poster\""));
        assert_eq!((size.choice.clone(), size.custom_presets.len()), (SaveSizeChoice::CommandLine, 0));
        assert!(SaveSize::new(COMMAND_LINE, Some(path.clone())).0.custom_presets.is_empty(), "the file no longer has it");
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn without_a_known_configuration_folder_presets_last_until_the_program_ends() {
        let (mut size, _) = SaveSize::new(COMMAND_LINE, None);
        size.preset_name = String::from("Session");
        assert!(size.add_preset().unwrap().ends_with("(only until the program ends: the configuration folder is not known)"));
        assert!(size.is_custom_preset("Session"));
    }

    #[test]
    fn a_file_that_cannot_be_written_is_reported_and_the_preset_stays_for_the_session() {
        let blocker = temporary_path("blocked");
        std::fs::create_dir_all(blocker.parent().unwrap()).unwrap();
        std::fs::write(blocker.parent().unwrap().join("not-a-folder"), "x").unwrap();
        let path = blocker.parent().unwrap().join("not-a-folder").join(resolution::FILE_NAME);
        let (mut size, _) = SaveSize::new(COMMAND_LINE, Some(path));
        size.preset_name = String::from("Stuck");
        let reason = size.add_preset().unwrap_err();
        assert!(reason.contains("could not be written") && reason.contains("stay until the program ends"), "{reason}");
        assert!(size.is_custom_preset("Stuck"));
        std::fs::remove_dir_all(blocker.parent().unwrap()).unwrap();
    }
}

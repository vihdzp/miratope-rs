//! Reads and loads the configuration file for Miratope.

use std::{
    ffi::OsStr,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use miratope_core::lang::SelectedLanguage;

use bevy::{app::AppExit, prelude::*};
use bevy_egui::{egui, EguiContext};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

/// The default path in which we look for the Miratope library.
const DEFAULT_PATH: &str = "./lib";

/// The default name for the configuration file.
const CONF_FILE: &str = "miratope.conf";

/// The plugin that loads and saves the configuration from disk.
pub struct ConfigPlugin;

impl Plugin for ConfigPlugin {
    fn build(&self, app: &mut AppBuilder) {
        // The configuration directory.
        let config_dir = Config::config_dir();

        // The configuration path.
        let mut config_path = ConfigPath(config_dir.clone());
        config_path.0.push(CONF_FILE);

        // Reads the entire configuration from file.
        let config = Config::read(&config_dir, &config_path);

        // Makes resources from the configuration, which may or may not
        // correspond to the actual stored values themselves.
        app.insert_resource(config_path)
            .insert_resource(config.lib_path)
            .insert_resource(config.selected_language)
            .insert_resource(config.background_color.clear_color())
            .insert_resource(config.light_mode.visuals())
            .add_system(update_visuals.system())
            .add_system_to_stage(CoreStage::Last, save_config.system());
    }
}

/// Stores the file path to the configuration file in Miratope.
pub struct ConfigPath(PathBuf);

impl AsRef<Path> for ConfigPath {
    fn as_ref(&self) -> &Path {
        self.0.as_ref()
    }
}

/// The path to the Miratope library.
#[derive(Clone, Deserialize, Serialize)]
pub struct LibPath(String);

impl Default for LibPath {
    fn default() -> Self {
        Self(
            fs::canonicalize(DEFAULT_PATH)
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_else(|_| DEFAULT_PATH.to_string()),
        )
    }
}

impl AsRef<OsStr> for LibPath {
    fn as_ref(&self) -> &OsStr {
        self.0.as_ref()
    }
}

/// The background color of the application in sRGB. This exists since
/// `ClearColor` wasn't serializable.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct BgColor(f32, f32, f32);

impl BgColor {
    /// Makes a new `BgColor` from the given `ClearColor`.
    pub fn new(clear_color: &ClearColor) -> Self {
        let color = clear_color.0;
        Self(color.r(), color.g(), color.b())
    }

    /// Makes a new `ClearColor` from the given `BgColor`.
    pub fn clear_color(&self) -> ClearColor {
        ClearColor(Color::rgb(self.0, self.1, self.2))
    }
}

/// Whether light mode is turned on or off.
#[derive(Default, Serialize, Deserialize)]
pub struct LightMode(bool);

impl LightMode {
    pub fn visuals(&self) -> egui::Visuals {
        if self.0 {
            egui::Visuals::light()
        } else {
            egui::Visuals::dark()
        }
    }
}

/// Updates the application appearance whenever the visuals are changed. This
/// occurs at application startup and whenever the user toggles light/dark mode.
fn update_visuals(egui_ctx: Res<EguiContext>, visuals: Res<egui::Visuals>) {
    if visuals.is_changed() {
        egui_ctx.ctx().set_visuals(visuals.clone());
    }
}

/// A monolithic struct that contains all of the configuration data for
/// Miratope. This is used only to read and write to disk – throughout the rest
/// of the application, each of its attributes represents a separate resource.
#[derive(Default, Deserialize, Serialize)]
pub struct Config {
    /// The path to the Miratope library.
    pub lib_path: LibPath,

    /// The currently selected language.
    pub selected_language: SelectedLanguage,

    /// The background color of the application.
    pub background_color: BgColor,

    /// Whether light mode is enabled.
    pub light_mode: LightMode,
}

impl Config {
    /// Returns the path to the configuration directory in Miratope.
    pub fn config_dir() -> PathBuf {
        if let Some(proj_dir) = ProjectDirs::from("rs", "Miratope", "Miratope") {
            proj_dir.config_dir().to_owned()
        } else {
            PathBuf::new()
        }
    }

    /// Attempts to read the configuration from a given path.
    pub fn from_path<T: AsRef<OsStr>>(config_path: T) -> Option<Self> {
        ron::from_str(&fs::read_to_string(config_path.as_ref()).ok()?).ok()
    }

    pub fn save<T: AsRef<OsStr>>(&self, config_path: T) {
        match fs::File::create(config_path.as_ref()) {
            // If the file could be created, we write to it.
            Ok(mut file) => {
                if file
                    .write(ron::to_string(self).unwrap().as_bytes())
                    .is_err()
                {
                    eprintln!("Could not write to the configuration file!");
                } else {
                    println!("Saved new config!");
                }
            }

            // Otherwise, we print the error.
            Err(err) => {
                eprintln!("Could not create the configuration file: {}", err);
            }
        }
    }

    /// Attemps to read the configuration file from disk. If it succeeds, it
    /// returns the read configuration. Otherwise, it returns the default
    /// configuration.
    pub fn read<T: AsRef<Path>, U: AsRef<Path>>(config_dir: T, config_path: U) -> Self {
        let config_dir = config_dir.as_ref();

        // Creates the configuration folder if it doesn't exist.
        if !config_dir.exists() {
            println!("Could not find the configuration directory, creating it!");
            if let Err(err) = fs::create_dir_all(&config_dir) {
                eprintln!("Could not create the configuration directory: {}", err);
                return Default::default();
            }
        }

        let config_path = config_path.as_ref();

        // We read from the configuration file and return its contents.
        if config_path.exists() {
            Self::from_path(config_path).unwrap_or_default()
        }
        // Creates the configuration file if it doesn't exist.
        else {
            println!("Could not find the configuration file, creating it!");
            let config = Self::default();
            config.save(config_path);
            config
        }
    }
}

/// Saves the configuration at application exit.
fn save_config(
    mut exit: EventReader<AppExit>,
    config_path: Res<ConfigPath>,
    lib_path: Res<LibPath>,
    selected_language: Res<SelectedLanguage>,
    background_color: Res<ClearColor>,
    visuals: Res<egui::Visuals>,
) {
    // If the application is being exited:
    if exit.iter().next().is_some() {
        let config = Config {
            lib_path: lib_path.clone(),
            selected_language: *selected_language,
            background_color: BgColor::new(background_color.as_ref()),
            light_mode: LightMode(!visuals.dark_mode),
        };

        config.save(&config_path.0);
    }
}

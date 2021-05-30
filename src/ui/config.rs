use std::{
    ffi::{OsStr, OsString},
    fs,
    io::Write,
};

use bevy::prelude::{AppBuilder, Plugin};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

const DEFAULT_PATH: &str = "./lib";

pub struct ConfigPlugin;

impl Plugin for ConfigPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(Config::read());
    }
}

/// Contains all of the configuration data for Miratope.
#[derive(Default)]
pub struct Config {
    /// The filename of the configuration file.
    pub config_path: OsString,

    /// All of the actual configuration data.
    pub data: ConfigData,
}

#[derive(Deserialize, Serialize)]
pub struct ConfigData {
    /// The path to the Miratope library.
    pub lib_path: String,
}

impl Default for ConfigData {
    fn default() -> Self {
        // Attempts making the "./lib" path absolute and stores that. If it
        // fails, it simply stores "./lib" as the default path.

        Self {
            lib_path: fs::canonicalize(DEFAULT_PATH)
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or(DEFAULT_PATH.to_string()),
        }
    }
}

impl Config {
    /// Attempts to read the configuration from a given path.
    pub fn from_path<T: AsRef<OsStr>>(config_path: T) -> Option<Self> {
        Some(Self {
            data: ron::from_str(&fs::read_to_string(config_path.as_ref()).ok()?).ok()?,
            config_path: config_path.as_ref().to_os_string(),
        })
    }

    /// Attemps to read the configuration file from disk. If it succeeds, it
    /// returns the read configuration. Otherwise, it returns the default
    /// configuration.
    pub fn read() -> Self {
        // Attempts to read the configuration file from the standard place in
        // the OS to store application configuration files.
        if let Some(proj_dir) = ProjectDirs::from("rs", "Miratope", "Miratope") {
            let config_dir = proj_dir.config_dir();

            // Creates the configuration folder if it doesn't exist.
            if !proj_dir.config_dir().exists() {
                println!("Could not find the configuration directory, creating it!");
                if let Err(err) = fs::create_dir_all(dbg!(config_dir)) {
                    println!("Could not create the configuration directory: {}", err);
                    return Default::default();
                }
            }

            let config_path = config_dir.join("miratope.conf");

            // We read from the configuration file and return its contents.
            if config_path.exists() {
                Self::from_path(config_path).unwrap_or_default()
            }
            // Creates the configuration file if it doesn't exist.
            else {
                println!("Could not find the configuration file, creating it!");

                match fs::File::create(&config_path) {
                    // If the file could be created, we write to it.
                    Ok(mut file) => {
                        if file
                            .write(ron::to_string(&ConfigData::default()).unwrap().as_bytes())
                            .is_err()
                        {
                            println!("Could not write to the configuration file!");
                        }
                    }

                    // Otherwise, we print the error.
                    Err(err) => {
                        println!("Could not create the configuration file: {}", err);
                    }
                }

                Default::default()
            }
        }
        // If we couldn't retrieve a valid home directory, we just return the
        // default settings.
        else {
            Default::default()
        }
    }
}

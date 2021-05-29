use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::format;
use directories::ProjectDirs;

pub struct ConfigStruct {
    pub filename: String,
    pub data: ConfigData,
}

impl ConfigStruct {
    pub fn create() -> ConfigStruct {
        let _error_config_struct = ConfigStruct {
            filename: String::from(""),
            data: ConfigData {
                paths: ConfigDataPaths {
                    library_path: String::from("")
                }
            },
        };
        if let Some(proj_dir) = ProjectDirs::from("Rs", "OfficialURL", "Miratope-rs") {
            if !proj_dir.config_dir().exists() {
                println!("Could not find the configuration directory, creating it!");
                if fs::create_dir(proj_dir.config_dir()).is_err() {
                    println!("Could not create the configuration directory!");
                    return _error_config_struct;
                }
            }
            let dir_path = proj_dir.config_dir().to_str().unwrap();
            let config_path = format!("{}/miratope.conf", dir_path);
            let lib_path = format!("{}/lib/", dir_path);
            let data: ConfigData;
            if !fs::metadata(&config_path).is_ok() {
                println!("Could not find the configuration file, creating it!");
                let default_config_file = format!("\
                [paths]\n\
                library_path = \"{}\"\n\
                ", lib_path);

                let file = fs::File::create(&config_path);
                if file.is_err() {
                    println!("Could not create the configuration file!");
                    return _error_config_struct;
                }

                return if file.unwrap().write_all(default_config_file.as_ref()).is_err() {
                    println!("Could not write to the configuration file!");
                    _error_config_struct
                } else {
                    ConfigStruct {
                        filename: String::from(&config_path),
                        data: toml::from_str(&default_config_file).unwrap(),
                    }
                };
            }
            let contents = fs::read_to_string(&config_path).unwrap();
            data = toml::from_str(&contents).unwrap();
            return ConfigStruct {
                filename: String::from(&config_path),
                data,
            };
        }
        return _error_config_struct;
    }
}

#[derive(Deserialize)]
pub struct ConfigData {
    pub paths: ConfigDataPaths,
}

#[derive(Deserialize, Debug)]
pub struct ConfigDataPaths {
    pub library_path: String,
}


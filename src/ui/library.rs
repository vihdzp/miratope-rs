use std::{ffi::OsStr, fs, io, path::PathBuf};

use bevy_egui::egui::Ui;
use serde::{Deserialize, Serialize};

use crate::{
    lang::{name::Con, Name as LangName, SelectedLanguage},
    polytope::concrete::Concrete,
};

#[derive(Serialize, Deserialize)]
pub enum SpecialLibrary {
    Polygon,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Name {
    /// A name in its language-independent representation.
    Abstract(LangName<Con>),

    /// A literal string name.
    Literal(String),
}

impl Name {
    /// This is running at 60 FPS but name parsing isn't blazing fast.
    pub fn parse(&self, selected_language: SelectedLanguage) -> String {
        match self {
            Self::Abstract(name) => selected_language.parse_uppercase(name, Default::default()),
            Self::Literal(name) => name.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum Library {
    /// A folder whose contents have not yet been read.
    UnloadedFolder {
        path: PathBuf,
        name: Name,
    },

    /// A folder whose contents have been read.
    LoadedFolder {
        name: Name,
        contents: Vec<Library>,
    },

    /// A file that can be loaded into Miratope.
    File {
        path: PathBuf,
        name: Name,
    },

    Special(SpecialLibrary),
}

impl Library {
    /// Loads the data from a file at a given path.
    pub fn new_file(path: &impl AsRef<OsStr>) -> Self {
        let path = PathBuf::from(&path);
        let name = if let Some(name) = Concrete::name_from_off(&path) {
            Name::Abstract(name)
        } else {
            Name::Literal(String::from(
                path.file_stem().map(|f| f.to_str()).flatten().unwrap_or(""),
            ))
        };

        Self::File { path, name }
    }

    /// Creates a new unloaded folder from a given path.
    pub fn new_folder(path: &impl AsRef<OsStr>) -> Self {
        let path = PathBuf::from(&path);
        assert!(path.is_dir(), "Path {:?} not a directory!", path);

        // Attempts to read from the .name file.
        if let Ok(Ok(name)) = fs::read(path.join(".name"))
            .map(|file| ron::from_str(&String::from_utf8(file).unwrap()))
        {
            Self::UnloadedFolder { path, name }
        }
        // Else, takes the name from the folder itself.
        else {
            let name = Name::Literal(String::from(
                path.file_name()
                    .map(|name| name.to_str())
                    .flatten()
                    .unwrap_or(""),
            ));

            Self::UnloadedFolder { path, name }
        }
    }

    /// Reads a folder's data from the `.folder` file. If it doesn't exist, it
    /// defaults to loading the folder's name and its data in alphabetical order.
    /// If that also fails, it returns an `Err`.
    pub fn folder_contents(path: &impl AsRef<OsStr>) -> io::Result<Vec<Self>> {
        let path = PathBuf::from(&path);
        assert!(path.is_dir(), "Path {:?} not a directory!", path);

        // Attempts to read from the .folder file.
        Ok(
            if let Some(Ok(folder)) = fs::read(path.join(".folder"))
                .ok()
                .map(|file| ron::from_str(&String::from_utf8(file).unwrap()))
            {
                folder
            }
            // Otherwise, just manually goes through the files.
            else {
                let mut contents = Vec::new();

                for entry in fs::read_dir(path.clone())? {
                    let path = &entry?.path();

                    // Adds a new unloaded folder.
                    if path.is_dir() {
                        contents.push(Self::new_folder(path));
                    }
                    // Adds a new file.
                    else {
                        let ext = path.extension();

                        if ext == Some(OsStr::new("off")) || ext == Some(OsStr::new("ggb")) {
                            contents.push(Self::new_file(path));
                        }
                    }
                }

                // We cache these contents for future use.
                fs::write(path.join(".folder"), ron::to_string(&contents).unwrap()).unwrap();

                contents
            },
        )
    }

    /// Shows the library.
    pub fn show(&mut self, ui: &mut Ui, selected_language: SelectedLanguage) -> Option<PathBuf> {
        match self {
            // Shows a collapsing drop-down, and loads the folder in case it's clicked.
            Self::UnloadedFolder { path, name } => {
                // Clones so that the closure doesn't require unique access.
                let path = path.clone();
                let name = name.clone();

                let mut res = None;

                ui.collapsing(name.parse(selected_language), |ui| {
                    let mut contents = Self::folder_contents(&path).unwrap();

                    // Contents of drop down.
                    for lib in contents.iter_mut() {
                        if let Some(file) = lib.show(ui, selected_language) {
                            res = Some(file);
                        }
                    }

                    // Opens the folder.
                    *self = Self::LoadedFolder { name, contents };
                });

                res
            }
            // Shows a drop-down with all of the files and folders.
            Self::LoadedFolder { name, contents } => {
                let mut res = None;
                ui.collapsing(name.parse(selected_language), |ui| {
                    for lib in contents.iter_mut() {
                        if let Some(file) = lib.show(ui, selected_language) {
                            res = Some(file);
                        }
                    }
                });

                res
            }
            // Shows a button that loads the file if clicked.
            Self::File { path, name } => {
                if ui.button(name.parse(selected_language)).clicked() {
                    Some(path.clone())
                } else {
                    None
                }
            }
            Library::Special(_) => {
                todo!()
            }
        }
    }
}

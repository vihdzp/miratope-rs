//! Loads and displays the Miratope library.

use std::{
    ffi::{OsStr, OsString},
    fs, io,
    path::PathBuf,
};

use super::config::LibPath;
use crate::Concrete;
use miratope_core::file::FromFile;
use special::*;

use bevy::prelude::*;
use bevy_egui::{egui, egui::Ui, EguiContext};
use serde::{Deserialize, Serialize};

mod special;

/// The plugin that loads the library.
pub struct LibraryPlugin;

impl Plugin for LibraryPlugin {
    fn build(&self, app: &mut App) {
        // This must run after the Config resource has been added.
        let lib_path = app.world.get_resource::<LibPath>().unwrap();
        let library = Library::new_folder(lib_path);

        // The library must be shown after the top panel, to avoid incorrect
        // positioning.
        app.insert_resource(library).add_system(
            show_library
                .system()
                .label("show_library")
                .after("show_top_panel"),
        );
    }
}

/// The result of showing the Miratope library in a particular frame.
pub enum ShowResult {
    /// Nothing happened this frame.
    None,

    /// We asked to load a file.
    Load(OsString),

    /// We asked to load a special polytope.
    Special(SpecialLibrary),
}

impl Default for ShowResult {
    fn default() -> Self {
        Self::None
    }
}

impl ShowResult {
    /// Returns whether `self` matches `ShowResult::None`.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Implements the or operator, so that `a | b` is `a` if it isn't `None`, but
/// `b` otherwise.
impl std::ops::BitOr for ShowResult {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        if self.is_none() {
            rhs
        } else {
            self
        }
    }
}

/// Implements the or assignment operator, defined in the same way as the or
/// operator.
impl std::ops::BitOrAssign for ShowResult {
    fn bitor_assign(&mut self, rhs: Self) {
        if !rhs.is_none() {
            *self = rhs;
        }
    }
}

/// Represents any of the files or folders that make up the Miratope library.
///
/// The library is internally stored is a tree-like structure. Once a folder
/// loads, it's (currently) never unloaded.
#[derive(Serialize, Deserialize)]
pub enum Library {
    /// A folder whose contents have not yet been read.
    UnloadedFolder {
        /// The name of the folder.
        name: String,
    },

    /// A folder whose contents have been read.
    LoadedFolder {
        /// The name of the folder.
        name: String,

        /// The contents of the folder.
        contents: Vec<Library>,
    },

    /// A file that can be loaded into Miratope.
    File {
        /// The file name.
        name: String,
    },

    /// Any special file in the library.
    Special(SpecialLibrary),
}

impl Library {
    /// Returns either the file or folder name of a given component of the
    /// library. In case that this doesn't apply, returns the empty string.
    pub fn path_name(&self) -> &str {
        match self {
            Library::UnloadedFolder { name, .. }
            | Library::LoadedFolder { name, .. }
            | Library::File { name, .. } => name,
            Library::Special(_) => "",
        }
    }

    /// Loads the data from a file at a given path.
    pub fn new_file(path: &impl AsRef<OsStr>) -> Self {
        Self::File {
            name: PathBuf::from(path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
        }
    }

    /// Creates a new unloaded folder from a given path. If the path doesn't
    /// exist or doesn't refer to a folder, we return `None`.
    pub fn new_folder<U: AsRef<OsStr>>(path: &U) -> Option<Self> {
        let path = PathBuf::from(&path);
        if !(path.exists() && path.is_dir()) {
            return None;
        }

        // Takes the name from the folder itself.
        Some(Self::UnloadedFolder {
            name: String::from(path.file_name().map(OsStr::to_str).flatten().unwrap_or("")),
        })
    }

    /// Reads a folder's data from the `.folder` file. If it doesn't exist, it
    /// defaults to loading the folder's name and its data in alphabetical
    /// order. If that also fails, it returns an `Err`.
    pub fn folder_contents<U: AsRef<OsStr>>(path: U) -> io::Result<Vec<Self>> {
        let path = PathBuf::from(&path);
        if !path.is_dir() {
            return Ok(Vec::new());
        }

        // Attempts to read from the .folder file.
        if let Some(Ok(folder)) = fs::read(path.join(".folder"))
            .ok()
            .map(|file| ron::from_str(&String::from_utf8(file).unwrap()))
        {
            Ok(folder)
        }
        // Otherwise, just manually goes through the files.
        else {
            let mut contents = Vec::new();

            for entry in fs::read_dir(path.clone())? {
                let path = &entry?.path();

                // Adds a new unloaded folder.
                if let Some(unloaded_folder) = Self::new_folder(path) {
                    contents.push(unloaded_folder);
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
            if fs::write(path.join(".folder"), ron::to_string(&contents).unwrap()).is_ok() {
                println!(".folder file overwritten!");
            } else {
                println!(".folder file could not be overwritten!");
            }

            Ok(contents)
        }
    }

    /// Shows the library in a given `Ui`, starting from a given path.
    pub fn show(&mut self, ui: &mut Ui, path: PathBuf) -> ShowResult {
        match self {
            // Shows a collapsing drop-down, and loads the folder in case it's clicked.
            Self::UnloadedFolder { name, .. } => {
                *self = Self::LoadedFolder {
                    name: name.clone(),
                    contents: Self::folder_contents(&path).unwrap(),
                };

                self.show(ui, path)
            }

            // Shows a drop-down with all of the files and folders.
            Self::LoadedFolder { name, contents, .. } => ui
                .collapsing(name.clone(), |ui| {
                    let mut res = ShowResult::None;

                    for lib in contents.iter_mut() {
                        let mut new_path = path.clone();
                        new_path.push(lib.path_name());
                        res |= lib.show(ui, new_path);
                    }

                    res
                })
                .body_returned
                .unwrap_or_default(),

            // Shows a button that loads the file if clicked.
            Self::File { name, .. } => {
                let label = PathBuf::from(name as &_)
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();

                if ui.button(label).clicked() {
                    ShowResult::Load(path.into_os_string())
                } else {
                    ShowResult::None
                }
            }

            // Shows any of the special files.
            Self::Special(special) => special.show(ui),
        }
    }
}

/// The system that shows the Miratope library.
fn show_library(
    egui_ctx: Res<'_, EguiContext>,
    mut query: Query<'_, '_, &mut Concrete>,
    mut library: ResMut<'_, Option<Library>>,
    lib_path: Res<'_, LibPath>,
) {
    // Shows the polytope library.
    if let Some(library) = library.as_mut() {
        egui::SidePanel::left("left_panel")
            .default_width(300.0)
            .max_width(450.0)
            .show(egui_ctx.ctx(), |ui| {
                egui::containers::ScrollArea::auto_sized().show(ui, |ui| {
                    match library.show(ui, PathBuf::from(lib_path.as_ref())) {
                        // No action needs to be taken.
                        ShowResult::None => {}

                        // Loads a selected file.
                        ShowResult::Load(file) => match Concrete::from_path(&file) {
                            Ok(q) => *query.iter_mut().next().unwrap() = q,
                            Err(err) => eprintln!("File open failed: {}", err),
                        },

                        // Loads a special polytope.
                        ShowResult::Special(special) => {
                            *query.iter_mut().next().unwrap() = special.load()
                        }
                    }
                })
            });
    }
}

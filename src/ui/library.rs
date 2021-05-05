use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy_egui::egui::Ui;

pub enum Library {
    ClosedFolder {
        path: PathBuf,
        name: String,
    },
    OpenFolder {
        path: PathBuf,
        name: String,
        contents: Vec<Library>,
    },
    File {
        path: PathBuf,
        name: String,
    },
}

impl Library {
    pub fn path(&self) -> &Path {
        match self {
            Self::ClosedFolder { path, name: _ } => path,
            Self::OpenFolder {
                path,
                name: _,
                contents: _,
            } => path,
            Self::File { path, name: _ } => path,
        }
    }

    pub fn name(&self) -> &String {
        match self {
            Self::ClosedFolder { path: _, name } => name,
            Self::OpenFolder {
                path: _,
                name,
                contents: _,
            } => name,
            Self::File { path: _, name } => name,
        }
    }

    pub fn show(&mut self, ui: &mut Ui) -> Option<PathBuf> {
        match self {
            Self::ClosedFolder { path, name } => {
                // Clones so that the closure doesn't require unique access.
                let path = path.clone();
                let name = name.clone();

                let mut res = None;

                ui.collapsing(name.clone(), |ui| {
                    let mut contents = Vec::new();

                    // Reads through the entries of the folders.
                    match fs::read_dir(path.clone()) {
                        Ok(dir_entry) => {
                            // For every entry in the folder:
                            for entry in dir_entry {
                                match entry {
                                    Ok(entry) => {
                                        let path = entry.path();
                                        let name = String::from(
                                            path.file_name()
                                                .map(|s| s.to_str())
                                                .flatten()
                                                .unwrap_or("none"),
                                        );

                                        // Adds the file to the folder's contents.
                                        if path.is_dir() {
                                            contents.push(Self::ClosedFolder { path, name });
                                        } else {
                                            contents.push(Self::File { path, name });
                                        }
                                    }
                                    Err(err) => {
                                        println!("Folder read at {:?} failed! Error: {}", path, err)
                                    }
                                }
                            }

                            // Contents of drop down.
                            for lib in contents.iter_mut() {
                                if let Some(file) = lib.show(ui) {
                                    res = Some(PathBuf::from(file));
                                }
                            }

                            // Opens the folder.
                            *self = Self::OpenFolder {
                                path,
                                name,
                                contents,
                            };
                        }
                        Err(err) => {
                            println!("Folder read at {:?} failed! Error: {}", path, err);
                        }
                    }
                });

                res
            }
            Self::OpenFolder {
                path: _,
                name,
                contents,
            } => {
                let mut res = None;
                ui.collapsing(name.clone(), |ui| {
                    for lib in contents.iter_mut() {
                        if let Some(file) = lib.show(ui) {
                            res = Some(file);
                        }
                    }
                });

                res
            }
            Self::File { path, name } => {
                if ui.button(name.clone()).clicked() {
                    Some(path.clone())
                } else {
                    None
                }
            }
        }
    }
}

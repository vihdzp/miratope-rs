//! All of the code that configures the UI.

pub mod camera;
pub mod library;

use std::{marker::PhantomData, path::PathBuf};

use crate::{
    geometry::{Hyperplane, Point, Vector},
    lang::{Options, SelectedLanguage},
    polytope::{concrete::Concrete, Polytope},
    Consts, Float, OffOptions,
};
use approx::{abs_diff_eq, abs_diff_ne};
use camera::ProjectionType;
use library::{Library, ShowResult, SpecialLibrary};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContext, EguiSettings};
use rfd::FileDialog;
use strum::IntoEnumIterator;

/// Stores the state of the cross-section view.
pub enum SectionState {
    /// The view is active.
    Active {
        /// The polytope from which the cross-section originates.
        original_polytope: Concrete,

        /// The range of the slider.
        minmax: (Float, Float),

        /// The position of the slicing hyperplane.
        hyperplane_pos: Float,

        /// Whether the cross-section is flattened into a dimension lower.
        flatten: bool,

        /// Whether we're updating the cross-section.
        lock: bool,
    },

    /// The view is inactive.
    Inactive,
}

impl SectionState {
    /// Makes the view inactive.
    pub fn reset(&mut self) {
        *self = Self::Inactive;
    }

    /// Sets the position of the hyperplane.
    pub fn set_pos(&mut self, pos: Float) {
        if let Self::Active {
            original_polytope: _,
            minmax: _,
            hyperplane_pos,
            flatten: _,
            lock: _,
        } = self
        {
            *hyperplane_pos = pos;
        }
    }

    /// Flips the flattening setting.
    pub fn flip_flat(&mut self) {
        if let Self::Active {
            original_polytope: _,
            minmax: _,
            hyperplane_pos: _,
            flatten,
            lock: _,
        } = self
        {
            *flatten = !*flatten;
        }
    }

    /// Flips the lock setting.
    pub fn flip_lock(&mut self) {
        if let Self::Active {
            original_polytope: _,
            minmax: _,
            hyperplane_pos: _,
            flatten: _,
            lock,
        } = self
        {
            *lock = !*lock;
        }
    }
}

impl Default for SectionState {
    fn default() -> Self {
        Self::Inactive
    }
}

/// Stores the direction in which the cross-sections are taken.
pub struct SectionDirection(Vector);

impl Default for SectionDirection {
    fn default() -> Self {
        Self(Vector::zeros(0))
    }
}

/// The system in charge of the UI. Loads every single thing on screen save for
/// the polytope itself.
#[allow(clippy::too_many_arguments)]
pub fn ui(
    egui_ctx: ResMut<EguiContext>,
    mut query: Query<&mut Concrete>,
    mut section_state: ResMut<SectionState>,
    mut section_direction: ResMut<SectionDirection>,
    mut file_dialog_state: ResMut<FileDialogState>,
    mut projection_type: ResMut<ProjectionType>,
    mut library: ResMut<Library>,
    mut selected_language: ResMut<SelectedLanguage>,
) {
    let ctx = egui_ctx.ctx();

    // The top bar.
    egui::TopPanel::top("top_panel").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            // Operations on files.
            egui::menu::menu(ui, "File", |ui| {
                // Loads a file.
                if ui.button("Open").clicked() {
                    file_dialog_state.open();
                }

                // Saves a file.
                if ui.button("Save").clicked() {
                    if let Some(p) = query.iter_mut().next() {
                        file_dialog_state
                            .save(selected_language.parse_uppercase(p.name(), Default::default()));
                    }
                }

                ui.separator();

                // Quits the application.
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });

            // Configures the view.
            egui::menu::menu(ui, "View", |ui| {
                let mut checked = projection_type.is_orthogonal();

                if ui.checkbox(&mut checked, "Orthogonal projection").clicked() {
                    projection_type.flip();

                    // Forces an update on all polytopes. (This does have an effect!)
                    for mut p in query.iter_mut() {
                        // We'll use this code once Clippy Issue #7171 is fixed:
                        // #[allow(clippy::no_effect)]
                        // &mut *p;

                        // Workaround:
                        p.name_mut();
                    }
                }
            });

            // Operations on polytopes.
            egui::menu::menu(ui, "Polytope", |ui| {
                ui.collapsing("Operations", |ui| {
                    // Converts the active polytope into its dual.
                    if ui.button("Dual").clicked() {
                        for mut p in query.iter_mut() {
                            match p.try_dual_mut() {
                                Ok(_) => println!("Dual succeeded."),
                                Err(idx) => println!(
                                    "Dual failed: Facet {} passes through inversion center.",
                                    idx
                                ),
                            }
                        }
                    }

                    ui.separator();

                    macro_rules! operation {
                        ($name:expr, $operation:ident) => {
                            if ui.button($name).clicked() {
                                for mut p in query.iter_mut() {
                                    *p = p.$operation();
                                }
                            }
                        };
                    }

                    // Makes a pyramid out of the current polytope.
                    operation!("Pyramid", pyramid);

                    // Makes a prism out of the current polytope.
                    operation!("Prism", prism);

                    // Makes a tegum out of the current polytope.
                    operation!("Tegum", tegum);

                    // Converts the active polytope into its antiprism.
                    if ui.button("Antiprism").clicked() {
                        for mut p in query.iter_mut() {
                            match p.try_antiprism() {
                                Ok(q) => *p = q,
                                Err(idx) => println!(
                                    "Dual failed: Facet {} passes through inversion center.",
                                    idx
                                ),
                            }
                        }
                    }

                    // Converts the active polytope into its Petrial.
                    if ui.button("Petrial").clicked() {
                        for mut p in query.iter_mut() {
                            match p.petrial_mut() {
                                Ok(_) => println!("Petrial succeeded."),
                                Err(_) => println!("Petrial failed."),
                            }
                        }
                    }

                    // Converts the active polytope into its Petrie polygon.
                    if ui.button("Petrie polygon").clicked() {
                        for mut p in query.iter_mut() {
                            match p.petrie_polygon() {
                                Some(q) => {
                                    *p = q;
                                    println!("Petrie polygon succeeded.")
                                }
                                None => println!("Petrie polygon failed."),
                            }
                        }
                    }

                    ui.separator();

                    // Recenters a polytope.
                    if ui.button("Recenter").clicked() {
                        for mut p in query.iter_mut() {
                            p.recenter();
                        }
                    }

                    ui.separator();

                    // Toggles cross-section mode.
                    if ui.button("Cross-section").clicked() {
                        match &mut *section_state {
                            // The view is active, but will be inactivated.
                            SectionState::Active {
                                original_polytope,
                                minmax: _,
                                hyperplane_pos: _,
                                flatten: _,
                                lock: _,
                            } => {
                                *query.iter_mut().next().unwrap() = original_polytope.clone();
                                *section_state = SectionState::Inactive;
                            }

                            // The view is inactive, but will be activated.
                            SectionState::Inactive => {
                                let mut p = query.iter_mut().next().unwrap();
                                p.flatten();

                                // The default direction is in the last coordinate axis.
                                let dim = p.dim().unwrap_or(0);
                                let mut direction = Vector::zeros(dim);
                                if dim > 0 {
                                    direction[dim - 1] = 1.0;
                                }

                                let minmax = p.minmax(&direction).unwrap_or((-1.0, 1.0));
                                let original_polytope = p.clone();

                                *section_state = SectionState::Active {
                                    original_polytope,
                                    minmax,
                                    hyperplane_pos: (minmax.0 + minmax.1) / 2.0,
                                    flatten: true,
                                    lock: false,
                                };
                                section_direction.0 = direction;
                            }
                        };
                    }
                });

                // Operates on the elements of the loaded polytope.
                ui.collapsing("Elements", |ui| {
                    // Converts the active polytope into any of its facets.
                    if ui.button("Facet").clicked() {
                        for mut p in query.iter_mut() {
                            println!("Facet");

                            if let Some(mut facet) = p.facet(0) {
                                facet.flatten();
                                facet.recenter();
                                *p = facet;

                                println!("Facet succeeded.")
                            } else {
                                println!("Facet failed: no facets.")
                            }
                        }
                    }

                    // Converts the active polytope into any of its verfs.
                    if ui.button("Verf").clicked() {
                        for mut p in query.iter_mut() {
                            println!("Verf");

                            match p.verf(0) {
                                Ok(Some(mut verf)) => {
                                    verf.flatten();
                                    verf.recenter();
                                    *p = verf;

                                    println!("Verf succeeded.")
                                }
                                Ok(None) => {
                                    println!("Verf failed: no vertices.")
                                }
                                Err(idx) => println!(
                                    "Verf failed: facet {} passes through inversion center.",
                                    idx
                                ),
                            }
                        }
                    }

                    // Outputs the element types, currently just prints to console.
                    if ui.button("Print el. types").clicked() {
                        for p in query.iter_mut() {
                            println!("Print element types");

                            println!("{}", p.print_element_types());
                        }
                    }
                });

                // Prints out properties about the loaded polytope.
                ui.collapsing("Properties", |ui| {
                    // Determines whether the polytope is orientable.
                    if ui.button("Orientability").clicked() {
                        for p in query.iter_mut() {
                            if p.orientable() {
                                println!("The polytope is orientable.");
                            } else {
                                println!("The polytope is not orientable.");
                            }
                        }
                    }

                    // Gets the volume of the polytope.
                    if ui.button("Volume").clicked() {
                        for p in query.iter_mut() {
                            if let Some(vol) = p.volume() {
                                println!("The volume is {}.", vol);
                            } else {
                                println!("The polytope has no volume.");
                            }
                        }
                    }
                });
            });

            // Stuff related to the Polytope Wiki.
            egui::menu::menu(ui, "Wiki", |ui| {
                // Goes to the wiki main page.
                if ui.button("Main Page").clicked() && webbrowser::open(crate::WIKI_LINK).is_err() {
                    println!("Website opening failed!")
                }

                // Searches the current polytope on the wiki.
                if ui.button("Current").clicked() {
                    for p in query.iter_mut() {
                        if webbrowser::open(&p.wiki_link()).is_err() {
                            println!("Website opening failed!")
                        }
                    }
                }
            });

            // Switch language.
            egui::menu::menu(ui, "Language", |ui| {
                for lang in SelectedLanguage::iter() {
                    if ui.button(lang.to_string()).clicked() {
                        *selected_language = lang;
                    }
                }
            });
        });

        // The cross-section settings.
        if let SectionState::Active {
            original_polytope: _,
            minmax,
            hyperplane_pos,
            flatten,
            lock,
        } = *section_state
        {
            ui.label("Cross section settings:");
            ui.spacing_mut().slider_width = ui.available_width() / 3.0;

            // Sets the slider range to the range of x coordinates in the polytope.
            let mut new_hyperplane_pos = hyperplane_pos;
            ui.add(
                egui::Slider::new(
                    &mut new_hyperplane_pos,
                    (minmax.0 + 0.00001)..=(minmax.1 - 0.00001),
                )
                .text("Slice depth")
                .prefix("pos: "),
            );

            // Updates the slicing depth.
            if abs_diff_ne!(hyperplane_pos, new_hyperplane_pos, epsilon = Float::EPS) {
                section_state.set_pos(new_hyperplane_pos);
            }

            let mut new_direction = section_direction.0.clone();
            let mut modified_coord = 0;

            ui.horizontal(|ui| {
                ui.label("Slice direction:");
                for (idx, coord) in new_direction.iter_mut().enumerate() {
                    ui.add(egui::DragValue::new(coord).speed(0.01));

                    // The index of the modified coordinate.
                    if abs_diff_eq!(section_direction.0[idx], *coord, epsilon = Float::EPS) {
                        modified_coord = idx;
                    }

                    // Gets rid of floating point shenanigans.
                    if abs_diff_eq!(*coord, 0.0, epsilon = Float::EPS.sqrt()) {
                        *coord = 0.0;
                    } else if abs_diff_eq!(*coord, 1.0, epsilon = Float::EPS) {
                        *coord = 1.0;
                    } else if abs_diff_eq!(*coord, -1.0, epsilon = Float::EPS) {
                        *coord = -1.0;
                    }
                }
            });

            // Normalizes the slicing direction.
            if new_direction.try_normalize_mut(Float::EPS).is_none() {
                // If this fails, sets it to the axis direction corresponding
                // to the last modified coordinate.
                for coord in new_direction.iter_mut() {
                    *coord = 0.0;
                }
                new_direction[modified_coord] = 1.0;
            }

            // Updates the slicing direction.
            #[allow(clippy::float_cmp)]
            if section_direction.0 != new_direction {
                section_direction.0 = new_direction;
            }

            ui.horizontal(|ui| {
                // Makes the current cross-section into the main polytope.
                if ui.button("Make main").clicked() {
                    section_state.reset();
                }

                let mut new_flatten = flatten;
                ui.add(egui::Checkbox::new(&mut new_flatten, "Flatten"));

                // Updates the flattening setting.
                if flatten != new_flatten {
                    section_state.flip_flat();
                }

                let mut new_lock = lock;
                ui.add(egui::Checkbox::new(&mut new_lock, "Lock"));

                // Updates the flattening setting.
                if lock != new_lock {
                    section_state.flip_lock();
                }
            });
        }
    });

    // Shows the polytope library.
    egui::SidePanel::left("side_panel", 350.0).show(ctx, |ui| {
        egui::containers::ScrollArea::auto_sized().show(ui, |ui| {
            match library.show_root(ui, *selected_language) {
                // No action needs to be taken.
                ShowResult::None => {}

                // Loads a selected file.
                ShowResult::Load(file) => {
                    if let Some(mut p) = query.iter_mut().next() {
                        if let Ok(q) = Concrete::from_path(&file) {
                            *p = q;
                        } else {
                            println!("File open failed!");
                        }
                    }
                }

                // Loads a special polytope.
                ShowResult::Special(special) => match special {
                    SpecialLibrary::Polygon(n, d) => {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = Concrete::star_polygon(n, d);
                        }
                    }
                    SpecialLibrary::Prism(n, d) => {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = Concrete::uniform_prism(n, d);
                        }
                    }
                    SpecialLibrary::Antiprism(n, d) => {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = Concrete::uniform_antiprism(n, d);
                        }
                    }
                    SpecialLibrary::Simplex(rank) => {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = Concrete::simplex(rank);
                        }
                    }
                    SpecialLibrary::Hypercube(rank) => {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = Concrete::hypercube(rank);
                        }
                    }
                    SpecialLibrary::Orthoplex(rank) => {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = Concrete::orthoplex(rank);
                        }
                    }
                },
            }
        })
    });
}

/// Contains all operations that manipulate file dialogs concretely.
///
/// Guarantees that file dialogs will be opened on the main thread, so as to
/// circumvent a MacOS limitation that all GUI operations must be done on the
/// main thread.
#[derive(Default)]
pub struct MainThreadToken(PhantomData<*const ()>);

impl MainThreadToken {
    /// Auxiliary function to create a new file dialog.
    fn new_file_dialog() -> FileDialog {
        FileDialog::new()
            .add_filter("OFF File", &["off"])
            .add_filter("GGB file", &["ggb"])
    }

    /// Returns the path given by an open file dialog.
    fn pick_file(&self) -> Option<PathBuf> {
        Self::new_file_dialog().pick_file()
    }

    /// Returns the path given by a save file dialog.
    fn save_file(&self, name: &str) -> Option<PathBuf> {
        Self::new_file_dialog().set_file_name(name).save_file()
    }
}

/// The type of file dialog we're showing.
enum FileDialogMode {
    /// We're not currently showing any file dialog.
    Disabled,

    /// We're showing a file dialog to open a file.
    Open,

    /// We're showing a file dialog to save a file.
    Save,
}

/// The file dialog is disabled by default.
impl Default for FileDialogMode {
    fn default() -> Self {
        Self::Disabled
    }
}

/// The state the file dialog is in.
#[derive(Default)]
pub struct FileDialogState {
    /// The file dialog mode.
    mode: FileDialogMode,

    /// The name of the file to load or save, if any.
    name: Option<String>,
}

impl FileDialogState {
    /// Changes the file dialog mode to [`FileDialogMode::Open`].
    pub fn open(&mut self) {
        self.mode = FileDialogMode::Open;
    }

    /// Changes the file dialog mode to [`FileDialogMode::Save`], and loads the
    /// name of the file.
    pub fn save(&mut self, name: String) {
        self.mode = FileDialogMode::Save;
        self.name = Some(name);
    }
}

/// The system in charge of showing the file dialog.
pub fn file_dialog(
    mut query: Query<&mut Concrete>,
    file_dialog_state: ResMut<FileDialogState>,
    token: NonSend<MainThreadToken>,
) {
    if file_dialog_state.is_changed() {
        match file_dialog_state.mode {
            // We want to save a file.
            FileDialogMode::Save => {
                if let Some(path) = token.save_file(file_dialog_state.name.as_ref().unwrap()) {
                    for p in query.iter_mut() {
                        if p.to_path(&path, OffOptions::default()).is_err() {
                            println!("File saving failed!");
                        }
                    }
                }
            }

            // We want to open a file.
            FileDialogMode::Open => {
                if let Some(path) = token.pick_file() {
                    for mut p in query.iter_mut() {
                        *p = Concrete::from_path(&path).unwrap();
                        p.recenter();
                    }
                }
            }
            FileDialogMode::Disabled => {}
        }
    }
}

/// Resizes the UI when the screen is resized.
pub fn update_scale_factor(mut egui_settings: ResMut<EguiSettings>, windows: Res<Windows>) {
    if let Some(window) = windows.get_primary() {
        egui_settings.scale_factor = 1.0 / window.scale_factor();
    }
}

/// Updates polytopes after an operation.
pub fn update_changed_polytopes(
    mut meshes: ResMut<Assets<Mesh>>,
    polies: Query<(&Concrete, &Handle<Mesh>, &Children), Changed<Concrete>>,
    wfs: Query<&Handle<Mesh>, Without<Concrete>>,
    mut windows: ResMut<Windows>,
    mut section_state: ResMut<SectionState>,
    selected_language: Res<SelectedLanguage>,
    orthogonal: Res<ProjectionType>,
) {
    for (poly, _mesh_handle, children) in polies.iter() {
        // The mesh is currently hidden, so we don't bother updating it.
        // *meshes.get_mut(_mesh_handle).unwrap() = poly.get_mesh(*orthogonal);

        if cfg!(debug_assertions) {
            poly.abs.is_valid().unwrap();
        }

        // Sets the window's name to the polytope's name.
        windows
            .get_primary_mut()
            .unwrap()
            .set_title(selected_language.parse_uppercase(poly.name(), Options::default()));

        // Updates all wireframes.
        for child in children.iter() {
            if let Ok(wf_handle) = wfs.get_component::<Handle<Mesh>>(*child) {
                let wf: &mut Mesh = meshes.get_mut(wf_handle).unwrap();
                *wf = poly.get_wireframe(*orthogonal);

                break;
            }
        }

        // We reset the cross-section view if we didn't use it to change the polytope.
        if !section_state.is_changed() {
            section_state.reset();
        }
    }
}

/// Updates the cross-section shown.
///
/// TODO: separate direction, make update_slice_direction method.
pub fn update_cross_section(
    mut query: Query<&mut Concrete>,
    mut section_state: ResMut<SectionState>,
    section_direction: Res<SectionDirection>,
) {
    if section_direction.is_changed() {
        if let SectionState::Active {
            original_polytope,
            hyperplane_pos: _,
            minmax,
            flatten: _,
            lock: _,
        } = &mut *section_state
        {
            *minmax = original_polytope
                .minmax(&section_direction.0)
                .unwrap_or((-1.0, 1.0));
        }
    }

    if section_state.is_changed() {
        if let SectionState::Active {
            original_polytope,
            hyperplane_pos,
            minmax,
            flatten,
            lock,
        } = &mut *section_state
        {
            // We don't update the view if it's locked.
            if *lock {
                return;
            }

            for mut p in query.iter_mut() {
                let r = original_polytope.clone();
                let hyp_pos = *hyperplane_pos + 0.0000001; // Botch fix for degeneracies.

                if let Some(dim) = r.dim() {
                    let hyperplane = Hyperplane::from_normal(
                        original_polytope.dim().unwrap_or(0),
                        section_direction.0.clone(),
                        hyp_pos,
                    );
                    *minmax = original_polytope
                        .minmax(&section_direction.0)
                        .unwrap_or((-1.0, 1.0));

                    let mut slice = r.cross_section(&hyperplane);

                    if *flatten {
                        slice.flatten_into(&hyperplane.subspace);
                        slice.recenter_with(
                            &hyperplane.flatten(&hyperplane.project(&Point::zeros(dim))),
                        );
                    }

                    *p = slice;
                }
            }
        }
    }
}

/// Updates the selected language.
pub fn update_language(
    mut polies: Query<&Concrete>,
    mut windows: ResMut<Windows>,
    selected_language: Res<SelectedLanguage>,
) {
    if selected_language.is_changed() {
        if let Some(poly) = polies.iter_mut().next() {
            windows
                .get_primary_mut()
                .unwrap()
                .set_title(selected_language.parse_uppercase(poly.name(), Options::default()));
        }
    }
}

pub mod input;

use std::{marker::PhantomData, path::PathBuf};

use crate::{
    geometry::{Hyperplane, Point},
    lang::{self, Language, Options},
    polytope::concrete::Concrete,
    polytope::Polytope,
    Float, OffOptions,
};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContext, EguiSettings};
use rfd::FileDialog;

/// Guarantees that file dialogs will be opened on the main thread, used to
/// circumvent a MacOS limitation that all GUI operations must be done on the
/// main thread.
pub struct MainThreadToken(PhantomData<*const ()>);

impl MainThreadToken {
    /// Initializes a new token.
    pub fn new() -> Self {
        Self(Default::default())
    }

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

/// Stores whether the cross-section view is active.
pub struct CrossSectionActive(pub bool);

impl CrossSectionActive {
    /// Flips the state.
    pub fn flip(&mut self) {
        self.0 = !self.0;
    }
}

/// Stores the state of the cross-section view.
pub struct CrossSectionState {
    /// The polytope from which the cross-section originates.
    original_polytope: Option<Concrete>,

    /// The position of the slicing hyperplane.
    hyperplane_pos: Float,

    /// Whether the cross-section is flattened into a dimension lower.
    flatten: bool,
}

impl Default for CrossSectionState {
    fn default() -> Self {
        Self {
            original_polytope: None,
            hyperplane_pos: 0.0,
            flatten: true,
        }
    }
}

/// The system in charge of the UI.
pub fn ui(
    egui_ctx: ResMut<EguiContext>,
    mut query: Query<&mut Concrete>,
    mut section_state: ResMut<CrossSectionState>,
    mut section_active: ResMut<CrossSectionActive>,
    token: NonSend<MainThreadToken>,
) {
    let ctx = egui_ctx.ctx();

    egui::TopPanel::top("top_panel").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            egui::menu::menu(ui, "File", |ui| {
                // Loads a file.
                if ui.button("Load").clicked() {
                    let path = token.pick_file();

                    if let Some(path) = path {
                        for mut p in query.iter_mut() {
                            *p = Concrete::from_path(&path).unwrap();
                        }
                    }
                }

                // Saves a file.
                if ui.button("Save").clicked() {
                    for p in query.iter_mut() {
                        let path = token.save_file(&lang::En::parse(p.name(), Default::default()));

                        if let Some(path) = path {
                            std::fs::write(path.clone(), p.to_off(OffOptions::default())).unwrap();
                        }
                    }
                }

                // Quits the application.
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });
        });

        ui.columns(6, |columns| {
            // Converts the active polytope into its dual.
            if columns[0].button("Dual").clicked() {
                for mut p in query.iter_mut() {
                    match p.dual_mut() {
                        Ok(_) => println!("Dual succeeded"),
                        Err(_) => println!("Dual failed"),
                    }

                    // If we're currently viewing a cross-section, it gets "fixed"
                    // as the active polytope.
                    section_state.original_polytope = None;
                    section_active.0 = false;

                    // Crashes for some reason.
                    // println!("{}", &p.concrete.to_src(off::OffOptions { comments: true }));
                }
            }

            // Converts the active polytope into any of its facets.
            if columns[1].button("Facet").clicked() {
                for mut p in query.iter_mut() {
                    println!("Facet");

                    if let Some(mut facet) = p.facet(0) {
                        facet.flatten();
                        facet.recenter();
                        *p = facet;
                    };

                    // If we're currently viewing a cross-section, it gets "fixed"
                    // as the active polytope.
                    section_state.original_polytope = None;
                    section_active.0 = false;
                }
            }

            // Converts the active polytope into any of its verfs.
            if columns[2].button("Verf").clicked() {
                for mut p in query.iter_mut() {
                    println!("Verf");

                    if let Some(mut facet) = p.verf(0) {
                        facet.flatten();
                        facet.recenter();
                        *p = facet;
                    };

                    // If we're currently viewing a cross-section, it gets "fixed"
                    // as the active polytope.
                    section_state.original_polytope = None;
                    section_active.0 = false;
                }
            }

            // Exports the active polytope as an OFF file (not yet functional!)
            if columns[3].button("Print OFF").clicked() {
                for p in query.iter_mut() {
                    println!("{}", p.to_off(OffOptions::default()));
                }
            }

            // Gets the volume of the polytope.
            if columns[4].button("Volume").clicked() {
                for p in query.iter_mut() {
                    if let Some(vol) = p.volume() {
                        println!("The volume is {}.", vol);
                    } else {
                        println!("The polytope has no volume.");
                    }
                }
            }

            // Toggles cross-section mode.
            if columns[5].button("Cross-section").clicked() {
                section_active.flip();
            }
        });

        ui.spacing_mut().slider_width = 800.0;

        // Sets the slider range to the range of x coordinates in the polytope.
        let mut new_hyperplane_pos = section_state.hyperplane_pos;
        let x_min;
        let x_max;

        if let Some(original) = &section_state.original_polytope {
            x_min = original.x_min().unwrap() + 0.0001;
            x_max = original.x_max().unwrap() - 0.0001;
        } else {
            x_min = -1.0;
            x_max = 1.0;
        }
        ui.add(egui::Slider::new(&mut new_hyperplane_pos, x_min..=x_max).text("Slice depth"));

        #[allow(clippy::float_cmp)]
        // Updates the slicing depth for the polytope, but only when needed.
        if section_state.hyperplane_pos != new_hyperplane_pos {
            section_state.hyperplane_pos = new_hyperplane_pos;
        }

        // Updates the flattening setting.
        let mut new_flatten = section_state.flatten;
        ui.add(egui::Checkbox::new(&mut new_flatten, "Flatten"));

        if section_state.flatten != new_flatten {
            section_state.flatten = new_flatten;
        }
    });
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
) {
    for (poly, mesh_handle, children) in polies.iter() {
        let mesh: &mut Mesh = meshes.get_mut(mesh_handle).unwrap();
        *mesh = poly.get_mesh();

        windows
            .get_primary_mut()
            .unwrap()
            .set_title(lang::En::parse(poly.name(), Options::default()));

        for child in children.iter() {
            if let Ok(wf_handle) = wfs.get_component::<Handle<Mesh>>(*child) {
                let wf: &mut Mesh = meshes.get_mut(wf_handle).unwrap();
                *wf = poly.get_wireframe();

                break;
            }
        }
    }
}

/// Shows or hides the cross-section view.
pub fn update_cross_section_state(
    mut query: Query<&mut Concrete>,
    mut state: ResMut<CrossSectionState>,
    active: Res<CrossSectionActive>,
) {
    if active.is_changed() {
        if active.0 {
            state.original_polytope = Some(query.iter_mut().next().unwrap().clone());
        } else if let Some(p) = state.original_polytope.take() {
            *query.iter_mut().next().unwrap() = p;
        }
    }
}

/// Updates the cross-section shown.
pub fn update_cross_section(
    mut query: Query<&mut Concrete>,
    state: Res<CrossSectionState>,
    active: Res<CrossSectionActive>,
) {
    if state.is_changed() && active.0 {
        for mut p in query.iter_mut() {
            let r = state.original_polytope.clone().unwrap();
            let hyp_pos = state.hyperplane_pos + 0.0000001; // Botch fix for degeneracies.

            if let Some(dim) = r.dim() {
                let hyperplane = Hyperplane::x(dim, hyp_pos);
                let mut slice = r.slice(&hyperplane);

                if state.flatten {
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

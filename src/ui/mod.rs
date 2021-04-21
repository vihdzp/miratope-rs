use crate::polytope::geometry::{Hyperplane, Point};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext, EguiSettings};

use crate::polytope::{Polytope, Renderable};

pub mod input;

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
    original_polytope: Option<Renderable>,

    /// The position of the slicing hyperplane.
    hyperplane_pos: f64,

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
    mut egui_ctx: ResMut<EguiContext>,
    mut query: Query<&mut Renderable>,
    mut section_state: ResMut<CrossSectionState>,
    mut section_active: ResMut<CrossSectionActive>,
) {
    let ctx = &mut egui_ctx.ctx;

    egui::TopPanel::top("top_panel").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            egui::menu::menu(ui, "File", |ui| {
                if ui.button("Quit").clicked() {
                    std::process::exit(0);
                }
            });
        });

        ui.columns(6, |columns| {
            // Converts the active polytope into its dual.
            if columns[0].button("Dual").clicked() {
                for mut p in query.iter_mut() {
                    match p.concrete.dual_mut() {
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

                    if let Some(mut facet) = p.concrete.facet(0) {
                        facet.flatten();
                        facet.recenter();
                        *p = Renderable::new(facet);
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

                    if let Some(mut facet) = p.concrete.verf(0) {
                        facet.flatten();
                        facet.recenter();
                        *p = Renderable::new(facet);
                    };

                    // If we're currently viewing a cross-section, it gets "fixed"
                    // as the active polytope.
                    section_state.original_polytope = None;
                    section_active.0 = false;
                }
            }

            // Exports the active polytope as an OFF file (not yet functional!)
            if columns[3].button("Export OFF").clicked() {
                for _p in query.iter_mut() {
                    println!("Export OFF");
                }
            }

            // Gets the volume of the polytope.
            if columns[4].button("Volume").clicked() {
                for p in query.iter_mut() {
                    if let Some(vol) = p.concrete.volume() {
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
        let min_x;
        let max_x;

        if let Some(original) = &section_state.original_polytope {
            min_x = original.concrete.min_x().unwrap();
            max_x = original.concrete.max_x().unwrap();
        } else {
            min_x = -1.0;
            max_x = 1.0;
        }
        ui.add(
            egui::Slider::f64(&mut new_hyperplane_pos, min_x..=max_x - 0.000001).max_decimals(5).text("Slice depth"),
        );

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
    polies: Query<(&Renderable, &Handle<Mesh>, &Children), Changed<Renderable>>,
    wfs: Query<&Handle<Mesh>, Without<Renderable>>,
) {
    for (poly, mesh_handle, children) in polies.iter() {
        let mesh: &mut Mesh = meshes.get_mut(mesh_handle).unwrap();
        *mesh = poly.get_mesh();

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
    mut query: Query<&mut Renderable>,
    mut state: ResMut<CrossSectionState>,
    active: ChangedRes<CrossSectionActive>,
) {
    if dbg!(active.0) {
        state.original_polytope = Some(query.iter_mut().next().unwrap().clone());
    } else if let Some(p) = state.original_polytope.take() {
        *query.iter_mut().next().unwrap() = p;
    } else {
        println!("This should only happen on startup.");
    }
}

/// Updates the cross-section shown.
pub fn update_cross_section(
    mut query: Query<&mut Renderable>,
    state: ChangedRes<CrossSectionState>,
    active: Res<CrossSectionActive>,
) {
    if active.0 {
        for mut p in query.iter_mut() {
            let r = state.original_polytope.clone().unwrap();
            let hyp_pos = state.hyperplane_pos + 0.0000001; // Botch fix for degeneracies.

            if let Some(dim) = r.concrete.dim() {
                let hyperplane = Hyperplane::x(dim, hyp_pos);
                let mut slice = r.concrete.slice(&hyperplane);

                if state.flatten {
                    slice.flatten_into(&hyperplane.subspace);
                    slice.recenter_with(
                        &hyperplane.flatten(&hyperplane.project(&Point::zeros(dim))),
                    );
                }

                *p = Renderable::new(slice);
            }
        }
    }
}

//! All of the code that configures the UI.

pub mod camera;
pub mod library;
pub mod top_panel;
pub mod windows;

use crate::{
    lang::{Options, SelectedLanguage},
    polytope::{concrete::Concrete, Polytope},
};

use camera::ProjectionType;
use top_panel::SectionState;

use bevy::prelude::*;
use bevy_egui::EguiSettings;

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
                *meshes.get_mut(wf_handle).unwrap() = poly.get_wireframe(*orthogonal);
            }
        }

        // We reset the cross-section view if we didn't use it to change the polytope.
        if !section_state.is_changed() {
            section_state.reset();
        }
    }
}

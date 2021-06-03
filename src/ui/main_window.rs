//! The systems that update the main window.

use crate::{
    lang::{Options, SelectedLanguage},
    polytope::{concrete::Concrete, Polytope},
};

use super::{camera::ProjectionType, top_panel::SectionState};

use bevy::prelude::*;
use bevy_egui::EguiSettings;

/// The plugin in charge of the Miratope main window, and of drawing the
/// polytope onto it.
pub struct MainWindowPlugin;

impl Plugin for MainWindowPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_system(update_scale_factor.system())
            .add_system_to_stage(CoreStage::PostUpdate, update_changed_polytopes.system());
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
    for (poly, _, children) in polies.iter() {
        if cfg!(debug_assertions) {
            println!("Polytope updated");
            poly.abs.is_valid().unwrap();
        }

        // The mesh is currently hidden, so we don't bother updating it.
        // *meshes.get_mut(_mesh_handle).unwrap() = poly.get_mesh(*orthogonal);

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
            section_state.close();
        }
    }
}

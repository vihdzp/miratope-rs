//! The systems that update the main window.

use super::{camera::ProjectionType, top_panel::SectionState};
use crate::mesh::Renderable;
use crate::NamedConcrete;

use bevy::prelude::*;
use bevy_egui::EguiSettings;
use miratope_core::abs::Ranked;
use miratope_lang::SelectedLanguage;

/// The plugin in charge of the Miratope main window, and of drawing the
/// polytope onto it.
pub struct MainWindowPlugin;

impl Plugin for MainWindowPlugin {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(CoreStage::PreUpdate, update_visible.system())
            .add_system(update_scale_factor.system())
            .add_system_to_stage(CoreStage::PostUpdate, update_changed_polytopes.system());
    }
}

pub fn update_visible(
    keyboard: Res<'_, Input<KeyCode>>,
    mut polies_vis: Query<'_, '_, &mut Visible, With<NamedConcrete>>,
    mut wfs_vis: Query<'_, '_, &mut Visible, Without<NamedConcrete>>,
) {
    if keyboard.just_pressed(KeyCode::V) {
        if let Some(mut visible) = polies_vis.iter_mut().next() {
            let vis = visible.is_visible;
            visible.is_visible = !vis;
        }
    }

    if keyboard.just_pressed(KeyCode::B) {
        if let Some(mut visible) = wfs_vis.iter_mut().next() {
            let vis = visible.is_visible;
            visible.is_visible = !vis;
        }
    }
}

/// Resizes the UI when the screen is resized.
pub fn update_scale_factor(mut egui_settings: ResMut<'_, EguiSettings>, windows: Res<'_, Windows>) {
    if let Some(window) = windows.get_primary() {
        egui_settings.scale_factor = 1.0 / window.scale_factor();
    }
}

/// Updates polytopes after an operation.
pub fn update_changed_polytopes(
    mut meshes: ResMut<'_, Assets<Mesh>>,
    polies: Query<'_, '_, (&NamedConcrete, &Handle<Mesh>, &Children), Changed<NamedConcrete>>,
    wfs: Query<'_, '_, &Handle<Mesh>, Without<NamedConcrete>>,
    mut windows: ResMut<'_, Windows>,
    mut section_state: ResMut<'_, SectionState>,
    selected_language: Res<'_, SelectedLanguage>,
    orthogonal: Res<'_, ProjectionType>,
) {
    for (poly, mesh_handle, children) in polies.iter() {
        println!("Polytope updated");

        if cfg!(debug_assertions) {
            poly.assert_valid();
        }

        *meshes.get_mut(mesh_handle).unwrap() = poly.mesh(*orthogonal);

        // Sets the window's name to the polytope's name.
        windows
            .get_primary_mut()
            .unwrap()
            .set_title(selected_language.parse(&poly.name));

        // Updates all wireframes.
        for child in children.iter() {
            if let Ok(wf_handle) = wfs.get_component::<Handle<Mesh>>(*child) {
                *meshes.get_mut(wf_handle).unwrap() = poly.wireframe(*orthogonal);
            }
        }

        // We reset the cross-section view if we didn't use it to change the polytope.
        if !section_state.is_changed() {
            section_state.close();
        }
    }
}

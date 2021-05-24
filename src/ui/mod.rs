//! All of the code that configures the UI.

pub mod camera;
pub mod egui_windows;
pub mod library;
pub mod main_window;
pub mod top_panel;

pub struct MiratopePlugins;

impl bevy::prelude::PluginGroup for MiratopePlugins {
    fn build(&mut self, group: &mut bevy::app::PluginGroupBuilder) {
        group
            .add(main_window::MainWindowPlugin)
            .add(egui_windows::EguiWindowPlugin)
            .add(camera::InputPlugin)
            // TODO: separate these into stages so that they load in a predetermined order.
            .add(library::LibraryPlugin)
            .add(top_panel::TopPanelPlugin);
    }
}

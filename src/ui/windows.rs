//! Loads the windows that permit more advanced settings.

use crate::{
    geometry::{Hypersphere, Point},
    polytope::concrete::Concrete,
    Float,
};

use bevy::prelude::{CoreStage, IntoSystem, Plugin, Query, Res, ResMut};
use bevy_egui::{egui, egui::CtxRef, EguiContext};

pub struct EguiWindowPlugin;

impl Plugin for EguiWindowPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.add_system_to_stage(CoreStage::PostUpdate, show_windows.system());
    }
}

/// Represents any of the windows on screen and their settings.
#[derive(Clone)]
pub enum WindowType {
    Dual {
        center: Point,
        radius: Float,
        central_inversion: bool,
    },
}

/// Compares by discriminant.
impl std::cmp::PartialEq for WindowType {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl std::cmp::Eq for WindowType {}

/// The result of showing the windows every frame.
pub enum ShowResult {
    /// Nothing special happens.
    None,

    /// A window is closed.
    Close,

    /// A window runs some action.
    Action,
}

impl WindowType {
    pub fn show(&self, ctx: &CtxRef) -> ShowResult {
        let mut open = true;
        let mut action = false;

        match self {
            Self::Dual {
                center: _,
                radius: _,
                central_inversion: _,
            } => egui::Window::new("Dual").open(&mut open).show(ctx, |ui| {
                ui.label("bottom text");

                if ui.button("Dual").clicked() {
                    action = true;
                }
            }),
        };

        if !open {
            ShowResult::Close
        } else if action {
            ShowResult::Action
        } else {
            ShowResult::None
        }
    }
}

/// The list of all windows currently shown on screen.
pub struct EguiWindows(Vec<WindowType>);

impl std::default::Default for EguiWindows {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl EguiWindows {
    /// Adds a new window to the list.
    pub fn push(&mut self, value: WindowType) {
        if !self.0.contains(&value) {
            self.0.push(value);
        }
    }

    ///Removes a window with a given index.
    pub fn remove(&mut self, idx: usize) {
        self.0.swap_remove(idx);
    }

    /// The number of windows on the screen.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Shows all of the windows, and returns one if its action has to run.
    pub fn show(&mut self, ctx: &CtxRef) -> Option<WindowType> {
        let mut action_window = None;
        let window_count = self.len();

        for idx in 0..window_count {
            let window = &self.0[idx];

            match window.show(ctx) {
                ShowResult::Close => {
                    println!("Close");
                    self.0.swap_remove(idx);
                    break;
                }
                ShowResult::Action => {
                    action_window = Some(self.0.swap_remove(idx));
                    break;
                }
                ShowResult::None => {}
            }
        }

        action_window
    }
}

/// The system that shows the windows on screen.
fn show_windows(
    egui_ctx: Res<EguiContext>,
    mut query: Query<&mut Concrete>,
    mut egui_windows: ResMut<EguiWindows>,
) {
    if let Some(result) = egui_windows.show(egui_ctx.ctx()) {
        match result {
            WindowType::Dual {
                center,
                radius,
                central_inversion,
            } => {
                let mut squared_radius = radius * radius;
                if central_inversion {
                    squared_radius *= -1.0;
                }

                let sphere = Hypersphere::new(center, squared_radius);

                for mut p in query.iter_mut() {
                    p.dual_mut_with(&sphere);
                }
            }
        }
    }
}

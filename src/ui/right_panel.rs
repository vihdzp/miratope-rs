//! Contains all code related to the right side panel.

use crate::Concrete;

use bevy::prelude::*;
use bevy_egui::{
    egui,
    EguiContext,
};
use miratope_core::{conc::{element_types::{ElementType, EL_NAMES, EL_SUFFIXES}, ConcretePolytope}, Polytope, abs::Ranked};
use vec_like::VecLike;

#[derive(Clone)]
pub struct ElementTypes {
    poly: Concrete,
    types: Vec<Vec<ElementType>>,
}

impl Default for ElementTypes {
    fn default() -> ElementTypes {
        ElementTypes {
            poly: Concrete::nullitope(),
            types: Vec::new(),
        }
    }
}

fn types_from_poly(poly: Mut<'_, Concrete>) -> ElementTypes {
    ElementTypes {
        poly: poly.clone(),
        types: poly.element_types(),
    }
}

/// The plugin in charge of everything on the right panel.
pub struct RightPanelPlugin;

impl Plugin for RightPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ElementTypes>()
            // The top panel must be shown first.
            .add_system(
                show_right_panel
                    .system()
                    .label("show_right_panel")
                    .after("show_top_panel"),
            );
    }
}


/// The system that shows the right panel.
#[allow(clippy::too_many_arguments)]
pub fn show_right_panel(
    // Info about the application state.
    egui_ctx: Res<'_, EguiContext>,
    mut query: Query<'_, '_, &mut Concrete>,

    // The Miratope resources controlled by the right panel.
    mut element_types: ResMut<'_, ElementTypes>,
) {
    // The right panel.
    egui::SidePanel::right("right_panel")
        .default_width(300.0)
        .max_width(450.0)
        .show(egui_ctx.ctx(), |ui| {
            
            if ui.button("Generate").clicked() {
                if let Some(p) = query.iter_mut().next() {
                    *element_types = types_from_poly(p);
                }
            }

            if ui.button("Load").clicked() {
                if let Some(mut p) = query.iter_mut().next() {
                    *p = element_types.poly.clone();
                }
            }

            ui.separator();

            egui::containers::ScrollArea::auto_sized().show(ui, |ui| {
                for (r, types) in element_types.types.clone().into_iter().enumerate().skip(1) {
                    let rank = element_types.poly.rank();

                    if r == rank {
                        break;
                    }

                    ui.heading(format!("{}", EL_NAMES[r]));
                    for t in types {
                        let i = t.example;

                        ui.horizontal(|ui| {
                            if ui.button("e").clicked() {
                                if let Some(mut p) = query.iter_mut().next() {
                                    if let Some(mut element) = element_types.poly.element(r,i) {
                                        element.flatten();
                                        element.recenter();
                                        *p = element;
                                    } else {
                                        println!("Element failed: no element at given index.")
                                    }
                                }
                            }
                            if ui.button("f").clicked() {
                                if let Some(mut p) = query.iter_mut().next() {
                                    match element_types.poly.element_fig(r, i) {
                                        Ok(Some(mut figure)) => {
                                            figure.flatten();
                                            figure.recenter();
                                            *p = figure;
                                        }
                                        Ok(None) => eprintln!("Figure failed: no element at given index."),
                                        Err(err) => eprintln!("Figure failed: {}", err),
                                    }
                                }
                            }
                            ui.label(format!(
                                "{} × {}-{}, {}-{}",
                                t.count,
                                element_types.poly[(r, i)].subs.len(),
                                EL_SUFFIXES[r],
                                element_types.poly[(r, i)].sups.len(),
                                EL_SUFFIXES[rank - r],
                            ));
                        });
                    }

                    ui.separator();
                }
        });
    });
}
//! Manages the memory tab.

use bevy::prelude::{Query, Res};
use bevy_egui::{egui, EguiContext};

use crate::Concrete;

/// Represents the memory slots to store polytopes.
#[derive(Default)]
pub struct Memory(pub Vec<Option<(Concrete, Option<String>)>>);

impl std::ops::Index<usize> for Memory {
    type Output = Option<(Concrete, Option<String>)>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// The label for the `n`-th memory slot.
pub fn slot_label(n: usize) -> String {
    format!("polytope {}", n)
}

impl Memory {
    /// Returns the length of the memory vector.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over the memory slots.
    pub fn iter(&self) -> std::slice::Iter<'_, Option<(Concrete, Option<String>)>> {
        self.0.iter()
    }

    /// Appends an element.
    pub fn push(&mut self, a: (Concrete, Option<String>)) {
        self.0.push(Some(a));
    }

    /// Shows the memory menu in a specified Ui.
    pub fn show(&mut self, query: &mut Query<'_, '_, &mut Concrete>, egui_ctx: &Res<'_, EguiContext>, open: &mut bool) {
        egui::Window::new("Memory")
            .open(open)
            .scroll(true)
            .default_width(260.0)
            .show(egui_ctx.ctx(), |ui| {
            egui::containers::ScrollArea::auto_sized().show(ui, |ui| {
                
                ui.horizontal(|ui| {
                    if ui.button("Clear memory").clicked() {
                        self.0.clear();
                    }
        
                    if ui.button("Add slot").clicked() {
                        self.0.push(None);
                    }
                });
    
                ui.separator();
    
                for (idx, slot) in self.0.iter_mut().enumerate() {
                    match slot {
                        // Shows an empty slot.
                        None => {
                            ui.horizontal(|ui| {
                                ui.label(format!("{}:", idx));
                                ui.label("Empty");

                                if ui.button("Save").clicked() {
                                    if let Some(p) = query.iter_mut().next() {
                                        *slot = Some((p.clone(), None));
                                    }
                                }
                             });
                        }

                        // Shows a slot with a polytope on it.
                        Some((poly, label)) => {
                            let mut clear = false;

                            ui.horizontal(|ui| {
                                ui.label(format!("{}:", idx));
                                ui.label(
                                    match label {
                                    None => {
                                        slot_label(idx)
                                    }
                                    
                                    Some(name) => {
                                        name.to_string()
                                    }
                                });

                                // Clones a polytope from memory.
                                if ui.button("Load").clicked() {
                                    *query.iter_mut().next().unwrap() = poly.clone();
                                }

                                // Swaps the current polytope with the one on memory.
                                if ui.button("Swap").clicked() {
                                    std::mem::swap(query.iter_mut().next().unwrap().as_mut(), poly);
                                }

                                // Clones a polytope into memory.
                                if ui.button("Save").clicked() {
                                    *poly = query.iter_mut().next().unwrap().clone();
                                }

                                // Clears a polytope from memory.
                                if ui.button("Clear").clicked() {
                                    clear = true;
                                }
                            });

                            if clear {
                                *slot = None;
                            }
                        }
                    }
                }
            });
        });
    }
}

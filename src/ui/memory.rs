//! Manages the memory tab.

use bevy::prelude::Query;
use bevy_egui::egui;

use crate::Concrete;

/// Represents the memory slots to store polytopes.
#[derive(Default)]
pub struct Memory(Vec<Option<Concrete>>);

impl std::ops::Index<usize> for Memory {
    type Output = Option<Concrete>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// The label for the `n`-th memory slot.
pub fn slot_label(n: usize) -> String {
    format!("Slot {}", n + 1)
}

impl Memory {
    /// Returns the length of the memory vector.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over the memory slots.
    pub fn iter(&self) -> std::slice::Iter<'_, Option<Concrete>> {
        self.0.iter()
    }

    /// Appends an element.
    pub fn push(&mut self, a: Concrete) {
        self.0.push(Some(a));
    }

    /// Shows the memory menu in a specified Ui.
    pub fn show(&mut self, ui: &mut egui::Ui, query: &mut Query<'_, '_, &mut Concrete>) {
        egui::menu::menu(ui, "Memory", |ui| {

            if ui.button("Clear memory").clicked() {
                self.0.clear();
            }

            if ui.button("Add slot").clicked() {
                self.0.push(None);
            }

            ui.separator();

            for (idx, slot) in self.0.iter_mut().enumerate() {
                match slot {
                    // Shows an empty slot.
                    None => {
                        egui::CollapsingHeader::new("Empty")
                            .id_source(idx)
                            .show(ui, |ui| {
                                if ui.button("Save").clicked() {
                                    if let Some(p) = query.iter_mut().next() {
                                        *slot = Some(p.clone());
                                    }
                                }
                            });
                    }

                    // Shows a slot with a polytope on it.
                    Some(poly) => {
                        let clear = egui::CollapsingHeader::new(slot_label(idx))
                            .id_source(idx)
                            .show(ui, |ui| {
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
                                ui.button("Clear").clicked()
                            });

                        if clear.body_returned == Some(true) {
                            *slot = None;
                        }
                    }
                }
            }
        });
    }
}

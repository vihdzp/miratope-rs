use crate::NamedConcrete;

use bevy::prelude::Query;
use bevy_egui::egui;
use miratope_lang::lang::En;

pub const MEMORY_SLOTS: usize = 8;

/// Represents the memory slots to store polytopes.
#[derive(Default)]
pub struct Memory([Option<NamedConcrete>; MEMORY_SLOTS]);

impl std::ops::Index<usize> for Memory {
    type Output = Option<NamedConcrete>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Memory {
    /// Returns an iterator over the memory slots.
    pub fn iter(&self) -> std::slice::Iter<'_, Option<NamedConcrete>> {
        self.0.iter()
    }

    /// Shows the memory menu in a specified Ui.
    pub fn show(&mut self, ui: &mut egui::Ui, query: &mut Query<'_, &mut NamedConcrete>) {
        use miratope_lang::Language;

        egui::menu::menu(ui, "Memory", |ui| {
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
                        let mut clear = false;

                        egui::CollapsingHeader::new(En::parse_uppercase(&poly.name))
                            .id_source(idx)
                            .show(ui, |ui| {
                                // Clones a polytope from memory.
                                if ui.button("Load").clicked() {
                                    if let Some(mut p) = query.iter_mut().next() {
                                        *p = poly.clone();
                                    }
                                }

                                // Swaps the current polytope with the one on memory.
                                if ui.button("Swap").clicked() {
                                    if let Some(mut p) = query.iter_mut().next() {
                                        std::mem::swap(p.as_mut(), poly);
                                    }
                                }

                                // Clones a polytope into memory.
                                if ui.button("Save").clicked() {
                                    if let Some(p) = query.iter_mut().next() {
                                        *poly = p.clone();
                                    }
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
        })
    }
}

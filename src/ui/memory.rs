use bevy::prelude::Query;
use bevy_egui::egui::{self, Ui};

use crate::{lang::En, polytope::concrete::Concrete};

pub const MEMORY_SLOTS: usize = 8;

/// Represents the memory slots to store polytopes.
#[derive(Default)]
pub struct Memory([Option<Concrete>; MEMORY_SLOTS]);

impl std::ops::Index<usize> for Memory {
    type Output = Option<Concrete>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Memory {
    pub fn iter(&self) -> std::slice::Iter<Option<Concrete>> {
        self.0.iter()
    }

    pub fn show(&mut self, ui: &mut Ui, query: &mut Query<&mut Concrete>) {
        use crate::lang::Language;

        egui::menu::menu(ui, "Memory", |ui| {
            for (idx, slot) in self.0.iter_mut().enumerate() {
                match slot {
                    None => {
                        egui::CollapsingHeader::new("Empty")
                            .id_source(idx)
                            .show(ui, |ui| {
                                if ui.button("Save").clicked() {
                                    for p in query.iter_mut() {
                                        *slot = Some(p.clone());
                                    }
                                }
                            });
                    }
                    Some(poly) => {
                        let mut clear = false;

                        egui::CollapsingHeader::new(En::parse_uppercase(
                            &poly.name,
                            Default::default(),
                        ))
                        .id_source(idx)
                        .show(ui, |ui| {
                            // Clones a polytope from memory.
                            if ui.button("Load").clicked() {
                                for mut p in query.iter_mut() {
                                    *p = poly.clone();
                                }
                            }

                            // Swaps the current polytope with the one on memory.
                            if ui.button("Swap").clicked() {
                                for mut p in query.iter_mut() {
                                    std::mem::swap(p.as_mut(), poly);
                                }
                            }

                            // Clones a polytope into memory.
                            if ui.button("Save").clicked() {
                                for p in query.iter_mut() {
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

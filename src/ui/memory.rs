use bevy::prelude::Query;
use bevy_egui::egui::{self, Ui};

use crate::{lang::En, polytope::concrete::Concrete};

const MEMORY_SLOTS: usize = 9;

/// Represents the memory slots to store polytopes.
pub struct Memory([Option<Concrete>; MEMORY_SLOTS]);

impl std::default::Default for Memory {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl Memory {
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
                            if ui.button("Load").clicked() {
                                for mut p in query.iter_mut() {
                                    *p = poly.clone();
                                }
                            }

                            if ui.button("Swap").clicked() {
                                for mut p in query.iter_mut() {
                                    std::mem::swap(&mut *p, poly);
                                }
                            }

                            if ui.button("Save").clicked() {
                                for p in query.iter_mut() {
                                    *poly = p.clone();
                                }
                            }

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

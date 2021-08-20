//! Contains all code related to the top bar.

use std::{marker::PhantomData, path::PathBuf};

use super::{camera::ProjectionType, memory::Memory, operations::*, UnitPointWidget};
use crate::{Float, Hyperplane, NamedConcrete, Point, Vector};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, menu, Ui},
    EguiContext,
};
use miratope_core::{abs::Ranked, conc::ConcretePolytope, file::FromFile, Polytope};
use miratope_lang::SelectedLanguage;
use rfd::FileDialog;
use strum::IntoEnumIterator;

/// The plugin in charge of everything on the top panel.
pub struct TopPanelPlugin;

impl Plugin for TopPanelPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FileDialogState::default())
            .insert_resource(Memory::default())
            .insert_resource(SectionDirection::default())
            .insert_resource(SectionState::default())
            .insert_non_send_resource(FileDialogToken::default())
            .add_system(file_dialog.system())
            // Windows must be the first thing shown.
            .add_system(
                show_top_panel
                    .system()
                    .label("show_top_panel")
                    .after("show_windows"),
            );
    }
}

/// Stores the state of the cross-section view.
pub enum SectionState {
    /// The view is active.
    Active {
        /// The polytope from which the cross-section originates.
        original_polytope: NamedConcrete,

        /// The range of the slider.
        minmax: (Float, Float),

        /// The position of the slicing hyperplane.
        hyperplane_pos: Float,

        /// Whether the cross-section is flattened into a dimension lower.
        flatten: bool,

        /// Whether we're updating the cross-section.
        lock: bool,
    },

    /// The view is inactive.
    Inactive,
}

impl SectionState {
    /// Makes the view inactive.
    pub fn close(&mut self) {
        *self = Self::Inactive;
    }

    pub fn open(&mut self, original_polytope: NamedConcrete, minmax: (f32, f32)) {
        *self = SectionState::Active {
            original_polytope,
            minmax,
            hyperplane_pos: (minmax.0 + minmax.1) / 2.0,
            flatten: true,
            lock: false,
        }
    }
}

impl Default for SectionState {
    fn default() -> Self {
        Self::Inactive
    }
}

/// Stores the direction in which the cross-sections are taken.
pub struct SectionDirection(Vector);

impl Default for SectionDirection {
    fn default() -> Self {
        Self(Vector::zeros(0))
    }
}

/// Contains all operations that manipulate file dialogs concretely.
///
/// Guarantees that file dialogs will be opened on the main thread, so as to
/// circumvent a MacOS limitation that all GUI operations must be done on the
/// main thread.
#[derive(Default)]
pub struct FileDialogToken(PhantomData<*const ()>);

impl FileDialogToken {
    /// Auxiliary function to create a new file dialog.
    fn new_file_dialog() -> FileDialog {
        FileDialog::new()
            .add_filter("OFF File", &["off"])
            .add_filter("GGB file", &["ggb"])
    }

    /// Returns the path given by an open file dialog.
    fn pick_file(&self) -> Option<PathBuf> {
        Self::new_file_dialog().pick_file()
    }

    /// Returns the path given by a save file dialog.
    fn save_file(&self, name: &str) -> Option<PathBuf> {
        Self::new_file_dialog().set_file_name(name).save_file()
    }
}

/// The type of file dialog we're showing.
enum FileDialogMode {
    /// We're not currently showing any file dialog.
    Disabled,

    /// We're showing a file dialog to open a file.
    Open,

    /// We're showing a file dialog to save a file.
    Save,
}

/// The file dialog is disabled by default.
impl Default for FileDialogMode {
    fn default() -> Self {
        Self::Disabled
    }
}

/// The state the file dialog is in.
#[derive(Default)]
pub struct FileDialogState {
    /// The file dialog mode.
    mode: FileDialogMode,

    /// The name of the file to load or save, if any.
    name: Option<String>,
}

impl FileDialogState {
    /// Changes the file dialog mode to [`FileDialogMode::Open`].
    pub fn open(&mut self) {
        self.mode = FileDialogMode::Open;
    }

    /// Changes the file dialog mode to [`FileDialogMode::Save`], and loads the
    /// name of the file.
    pub fn save(&mut self, name: String) {
        self.mode = FileDialogMode::Save;
        self.name = Some(name);
    }

    /// Gets the name of the file dialog.
    pub fn unwrap_name(&self) -> &str {
        self.name.as_ref().unwrap()
    }
}

/// The system in charge of showing the file dialog.
pub fn file_dialog(
    mut query: Query<'_, '_, &mut NamedConcrete>,
    file_dialog_state: Res<'_, FileDialogState>,
    file_dialog: NonSend<'_, FileDialogToken>,
) {
    if file_dialog_state.is_changed() {
        match file_dialog_state.mode {
            // We want to save a file.
            FileDialogMode::Save => {
                if let Some(path) = file_dialog.save_file(file_dialog_state.unwrap_name()) {
                    if let Some(p) = query.iter_mut().next() {
                        if let Err(err) = p.con().to_path(&path, Default::default()) {
                            eprintln!("File saving failed: {}", err);
                        }
                    }
                }
            }

            // We want to open a file.
            FileDialogMode::Open => {
                if let Some(path) = file_dialog.pick_file() {
                    if let Some(mut p) = query.iter_mut().next() {
                        match NamedConcrete::from_path(&path) {
                            Ok(q) => {
                                *p = q;
                                p.recenter();
                            }
                            Err(err) => eprintln!("File open failed: {}", err),
                        }
                    }
                }
            }

            // There's nothing to do with the file dialog this frame.
            FileDialogMode::Disabled => {}
        }
    }
}

/// Whether the hotkey to enable "advanced" options is enabled.
pub fn advanced(keyboard: &Input<KeyCode>) -> bool {
    keyboard.pressed(KeyCode::LControl) || keyboard.pressed(KeyCode::RControl)
}

/// All of the windows that can be shown on screen, as mutable resources.
pub type EguiWindows<'a> = (
    ResMut<'a, DualWindow>,
    ResMut<'a, PyramidWindow>,
    ResMut<'a, PrismWindow>,
    ResMut<'a, TegumWindow>,
    ResMut<'a, AntiprismWindow>,
    ResMut<'a, DuopyramidWindow>,
    ResMut<'a, DuoprismWindow>,
    ResMut<'a, DuotegumWindow>,
    ResMut<'a, DuocombWindow>,
);

/// The system that shows the top panel.
#[allow(clippy::too_many_arguments)]
pub fn show_top_panel(
    // Info about the application state.
    egui_ctx: Res<'_, EguiContext>,
    mut query: Query<'_, '_, &mut NamedConcrete>,
    mut windows: ResMut<'_, Windows>,
    keyboard: Res<'_, Input<KeyCode>>,

    // The Miratope resources controlled by the top panel.
    mut section_state: ResMut<'_, SectionState>,
    mut section_direction: ResMut<'_, SectionDirection>,
    mut file_dialog_state: ResMut<'_, FileDialogState>,
    mut projection_type: ResMut<'_, ProjectionType>,
    mut memory: ResMut<'_, Memory>,
    mut background_color: ResMut<'_, ClearColor>,
    mut selected_language: ResMut<'_, SelectedLanguage>,
    mut visuals: ResMut<'_, egui::Visuals>,

    // The different windows that can be shown.
    (
        mut dual_window,
        mut pyramid_window,
        mut prism_window,
        mut tegum_window,
        mut antiprism_window,
        mut duopyramid_window,
        mut duoprism_window,
        mut duotegum_window,
        mut duocomb_window,
    ): EguiWindows<'_>,
) {
    // The top bar.
    egui::TopBottomPanel::top("top_panel").show(egui_ctx.ctx(), |ui| {
        menu::bar(ui, |ui| {
            // Operations on files.
            menu::menu(ui, "File", |ui| {
                // Loads a file.
                if ui.button("Open").clicked() {
                    file_dialog_state.open();
                }

                // Saves a file.
                if ui.button("Save").clicked() {
                    if let Some(p) = query.iter_mut().next() {
                        file_dialog_state.save(selected_language.parse(&p.name));
                    }
                }

                ui.separator();

                // Quits the application.
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });

            // Configures the view.
            menu::menu(ui, "View", |ui| {
                let mut checked = projection_type.is_orthogonal();

                if ui.checkbox(&mut checked, "Orthogonal projection").clicked() {
                    projection_type.flip();

                    // Forces an update on all polytopes.
                    if let Some(mut p) = query.iter_mut().next() {
                        p.set_changed();
                    }
                }
            });

            // Anything related to the polytope on screen.
            menu::menu(ui, "Polytope", |ui| {
                /// Sorts the elements of a polytope. This will only take the polytope as
                /// mutable if necessary, thus avoiding a potential reload.
                macro_rules! element_sort {
                    ($p:ident) => {
                        if !$p.abs().sorted {
                            $p.ranks_mut().element_sort();
                            $p.abs_mut().sorted = true;
                        }
                    };
                }

                // Operations on polytopes.
                ui.collapsing("Operations", |ui| {
                    // Operations that take a single polytope.
                    ui.collapsing("Single", |ui| {
                        // Converts the active polytope into its dual.
                        if ui.button("Dual").clicked() {
                            if advanced(&keyboard) {
                                dual_window.open();
                            } else if let Some(mut p) = query.iter_mut().next() {
                                match p.try_dual_mut() {
                                    Ok(_) => println!("Dual succeeded."),
                                    Err(err) => eprintln!("Dual failed: {}", err),
                                }
                            }
                        }

                        ui.separator();

                        // Makes a pyramid out of the current polytope.
                        if ui.button("Pyramid").clicked() {
                            if advanced(&keyboard) {
                                pyramid_window.open();
                            } else if let Some(mut p) = query.iter_mut().next() {
                                p.pyramid_mut();
                            }
                        }

                        // Makes a prism out of the current polytope.
                        if ui.button("Prism").clicked() {
                            if advanced(&keyboard) {
                                prism_window.open();
                            } else if let Some(mut p) = query.iter_mut().next() {
                                p.prism_mut();
                            }
                        }

                        // Makes a tegum out of the current polytope.
                        if ui.button("Tegum").clicked() {
                            if advanced(&keyboard) {
                                tegum_window.open();
                            } else if let Some(mut p) = query.iter_mut().next() {
                                p.tegum_mut();
                            }
                        }

                        // Converts the active polytope into its antiprism.
                        if ui.button("Antiprism").clicked() {
                            if advanced(&keyboard) {
                                antiprism_window.open();
                            } else if let Some(mut p) = query.iter_mut().next() {
                                match p.try_antiprism() {
                                    Ok(q) => *p = q,
                                    Err(err) => eprintln!("Antiprism failed: {}", err),
                                }
                            }
                        }

                        ui.separator();

                        // Converts the active polytope into its Petrial.
                        if ui.button("Petrial").clicked() {
                            if let Some(mut p) = query.iter_mut().next() {
                                if p.petrial_mut() {
                                    println!("Petrial succeeded.");
                                } else {
                                    eprintln!("Petrial failed.");
                                }
                            }
                        }

                        // Converts the active polytope into its Petrie polygon.
                        if ui.button("Petrie polygon").clicked() {
                            if let Some(mut p) = query.iter_mut().next() {
                                match p.petrie_polygon() {
                                    Some(q) => {
                                        *p = q;
                                        println!("Petrie polygon succeeded.")
                                    }
                                    None => eprintln!("Petrie polygon failed."),
                                }
                            }
                        }

                        ui.separator();

                        // Converts the active polytope into its ditope.
                        if ui.button("Ditope").clicked() {
                            if let Some(mut p) = query.iter_mut().next() {
                                p.ditope_mut();
                                println!("Ditope succeeded!");
                            }
                        }

                        // Converts the active polytope into its hosotope.
                        if ui.button("Hosotope").clicked() {
                            if let Some(mut p) = query.iter_mut().next() {
                                p.hosotope_mut();
                                println!("Hosotope succeeded!");
                            }
                        }
                    });

                    // Operations that take two polytopes an arguments.
                    ui.collapsing("Double", |ui| {
                        // Opens the window to make duopyramids.
                        if ui.button("Duopyramid").clicked() {
                            duopyramid_window.open();
                        }

                        // Opens the window to make duoprisms.
                        if ui.button("Duoprism").clicked() {
                            duoprism_window.open();
                        }

                        // Opens the window to make duotegums.
                        if ui.button("Duotegum").clicked() {
                            duotegum_window.open();
                        }

                        // Opens the window to make duocombs.
                        if ui.button("Duocomb").clicked() {
                            duocomb_window.open();
                        }
                    });

                    if ui.button("Truncate").clicked() {
                        let mut p = query.iter_mut().next().unwrap();
                        element_sort!(p);
                        *p = p.truncate(vec![0, 1], vec![0.5, 0.5]);
                    }

                    if ui.button("Omnitruncate").clicked() {
                        let mut p = query.iter_mut().next().unwrap();
                        element_sort!(p);
                        *p = p.omnitruncate();
                    }
                
                    ui.separator();

                    // Recenters a polytope.
                    if ui.button("Recenter").clicked() {
                        query.iter_mut().next().unwrap().recenter();
                    }

                    ui.separator();

                    // Toggles cross-section mode.
                    if ui.button("Cross-section").clicked() {
                        match section_state.as_mut() {
                            // The view is active, but will be inactivated.
                            SectionState::Active {
                                original_polytope, ..
                            } => {
                                *query.iter_mut().next().unwrap() = original_polytope.clone();
                                section_state.close();
                            }

                            // The view is inactive, but will be activated.
                            SectionState::Inactive => {
                                let mut p = query.iter_mut().next().unwrap();
                                p.flatten();

                                // The default direction is in the last coordinate axis.
                                let dim = p.dim_or();
                                let mut direction = Vector::zeros(dim);
                                if dim > 0 {
                                    direction[dim - 1] = 1.0;
                                }

                                let minmax = p.minmax(&direction).unwrap_or((-1.0, 1.0));
                                let original_polytope = p.clone();

                                section_state.open(original_polytope, minmax);
                                section_direction.0 = direction;
                            }
                        };
                    }
                });

                // Operates on the elements of the loaded polytope.
                ui.collapsing("Elements", |ui| {
                    // Converts the active polytope into any of its facets.
                    if ui.button("Facet").clicked() {
                        if let Some(mut p) = query.iter_mut().next() {
                            println!("Facet");

                            if let Some(mut facet) = p.facet(0) {
                                facet.flatten();
                                facet.recenter();
                                *p = facet;

                                println!("Facet succeeded.")
                            } else {
                                eprintln!("Facet failed: no facets.")
                            }
                        }
                    }

                    // Converts the active polytope into any of its verfs.
                    if ui.button("Verf").clicked() {
                        if let Some(mut p) = query.iter_mut().next() {
                            println!("Verf");

                            match p.verf(0) {
                                Ok(Some(mut verf)) => {
                                    verf.flatten();
                                    verf.recenter();
                                    *p = verf;

                                    println!("Verf succeeded.")
                                }
                                Ok(None) => eprintln!("Verf failed: no vertices."),
                                Err(err) => eprintln!("Verf failed: {}", err),
                            }
                        }
                    }

                    // Outputs the element types, currently just prints to console.
                    if ui.button("Counts").clicked() {
                        if let Some(p) = query.iter_mut().next() {
                            p.con().print_element_types();
                        }
                    }
                });

                // Prints out properties about the loaded polytope.
                ui.collapsing("Properties", |ui| {
                    // Determines the circumsphere of the polytope.
                    if ui.button("Circumsphere").clicked() {
                        if let Some(p) = query.iter_mut().next() {
                            match p.circumsphere() {
                                Some(sphere) => println!(
                                    "The circumradius is {} and the circumcenter is {}.",
                                    sphere.radius(),
                                    sphere.center
                                ),
                                None => println!("The polytope has no circumsphere."),
                            }
                        }
                    }

                    // Determines whether the polytope is orientable.
                    if ui.button("Orientability").clicked() {
                        if let Some(mut p) = query.iter_mut().next() {
                            element_sort!(p);

                            if p.orientable() {
                                println!("The polytope is orientable.");
                            } else {
                                println!("The polytope is not orientable.");
                            }
                        }
                    }

                    // Gets the volume of the polytope.
                    if ui.button("Volume").clicked() {
                        if let Some(mut p) = query.iter_mut().next() {
                            element_sort!(p);

                            if let Some(vol) = p.volume() {
                                println!("The volume is {}.", vol);
                            } else {
                                println!("The polytope has no volume.");
                            }
                        }
                    }

                    // Gets the number of flags of the polytope.
                    if ui.button("Flag count").clicked() {
                        if let Some(mut p) = query.iter_mut().next() {
                            element_sort!(p);
                            println!("The polytope has {} flags.", p.flags().count())
                        }
                    }
                });
            });

            memory.show(ui, &mut query);

            // Stuff related to the Polytope Wiki.
            menu::menu(ui, "Wiki", |ui| {
                // Goes to the wiki main page.
                if ui.button("Main Page").clicked() {
                    if let Err(err) = webbrowser::open(miratope_lang::WIKI_LINK) {
                        eprintln!("Website opening failed: {}", err);
                    }
                }

                // Searches the current polytope on the wiki.
                if ui.button("Current").clicked() {
                    if let Some(p) = query.iter_mut().next() {
                        if let Err(err) = webbrowser::open(&p.name.wiki_link()) {
                            eprintln!("Website opening failed: {}", err)
                        }
                    }
                }
            });

            // Switch language.
            menu::menu(ui, "Preferences", |ui| {
                ui.collapsing("Language", |ui| {
                    for lang in SelectedLanguage::iter() {
                        if ui.button(lang.to_string()).clicked() {
                            *selected_language = lang;
                        }
                    }

                    if selected_language.is_changed() {
                        if let Some(poly) = query.iter_mut().next() {
                            windows
                                .get_primary_mut()
                                .unwrap()
                                .set_title(selected_language.parse(&poly.name));
                        }
                    }
                });
            });

            // General help.
            menu::menu(ui, "Help", |ui| {
                if ui.button("File bug").clicked() {
                    if let Err(err) = webbrowser::open(crate::NEW_ISSUE) {
                        eprintln!("Website opening failed: {}", err);
                    }
                }
            });

            // Background color picker.

            // The current background color.
            let [r, g, b, a] = background_color.0.as_rgba_f32();
            let color = egui::Color32::from_rgba_premultiplied(
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
                (a * 255.0) as u8,
            );

            // The new background color.
            let mut new_color = color;
            egui::color_picker::color_edit_button_srgba(
                ui,
                &mut new_color,
                egui::color_picker::Alpha::Opaque,
            );

            // Updates the background color if necessary.
            if color != new_color {
                background_color.0 = Color::rgb(
                    new_color.r() as f32 / 255.0,
                    new_color.g() as f32 / 255.0,
                    new_color.b() as f32 / 255.0,
                );
            }

            // Light/dark mode toggle.
            if let Some(new_visuals) = visuals.light_dark_small_toggle_button(ui) {
                *visuals = new_visuals;
            }
        });

        // Shows secondary views below the menu bar.
        show_views(ui, query, section_state, section_direction);
    });
}

/// Shows any secondary views that are active. Currently, just shows the
/// cross-section view.
fn show_views(
    ui: &mut Ui,
    mut query: Query<'_, '_, &mut NamedConcrete>,
    mut section_state: ResMut<'_, SectionState>,
    mut section_direction: ResMut<'_, SectionDirection>,
) {
    // The cross-section settings.
    if let SectionState::Active {
        minmax,
        hyperplane_pos,
        flatten,
        lock,
        ..
    } = *section_state
    {
        ui.label("Cross section settings:");
        ui.spacing_mut().slider_width = ui.available_width() / 3.0;

        // Sets the slider range to the range of x coordinates in the polytope.
        let mut new_hyperplane_pos = hyperplane_pos;
        ui.add(
            egui::Slider::new(
                &mut new_hyperplane_pos,
                (minmax.0 + 0.0000001)..=(minmax.1 - 0.0000001), // We do this to avoid nullitopes.
            )
            .text("Slice depth")
            .prefix("pos: "),
        );

        // Updates the slicing depth.
        #[allow(clippy::float_cmp)]
        if hyperplane_pos != new_hyperplane_pos {
            if let SectionState::Active { hyperplane_pos, .. } = section_state.as_mut() {
                *hyperplane_pos = new_hyperplane_pos;
            } else {
                unreachable!()
            }
        }

        let mut new_direction = section_direction.0.clone();
        ui.add(UnitPointWidget::new(
            &mut new_direction,
            "Cross-section depth",
        ));

        // Updates the slicing direction.
        #[allow(clippy::float_cmp)]
        if section_direction.0 != new_direction {
            section_direction.0 = new_direction;
        }

        ui.horizontal(|ui| {
            // Makes the current cross-section into the main polytope.
            if ui.button("Make main").clicked() {
                section_state.close();
            }

            let mut new_flatten = flatten;
            ui.add(egui::Checkbox::new(&mut new_flatten, "Flatten"));

            // Updates the flattening setting.
            if flatten != new_flatten {
                if let SectionState::Active { flatten, .. } = section_state.as_mut() {
                    *flatten = new_flatten;
                } else {
                    unreachable!()
                }
            }

            let mut new_lock = lock;
            ui.add(egui::Checkbox::new(&mut new_lock, "Lock"));

            // Updates the flattening setting.
            if lock != new_lock {
                if let SectionState::Active { lock, .. } = section_state.as_mut() {
                    *lock = new_lock;
                } else {
                    unreachable!()
                }
            }
        });
    }

    if section_direction.is_changed() {
        if let SectionState::Active {
            original_polytope,
            minmax,
            ..
        } = section_state.as_mut()
        {
            *minmax = original_polytope
                .minmax(&section_direction.0)
                .unwrap_or((-1.0, 1.0));
        }
    }

    if section_state.is_changed() {
        if let SectionState::Active {
            original_polytope,
            hyperplane_pos,
            minmax,
            flatten,
            lock,
        } = section_state.as_mut()
        {
            // We don't update the view if it's locked.
            if *lock {
                return;
            }

            if let Some(mut p) = query.iter_mut().next() {
                let r = original_polytope.clone();
                let hyp_pos = *hyperplane_pos;

                if let Some(dim) = r.dim() {
                    let hyperplane = Hyperplane::new(section_direction.0.clone(), hyp_pos);
                    *minmax = original_polytope
                        .minmax(&section_direction.0)
                        .unwrap_or((-1.0, 1.0));

                    let mut slice = r.cross_section(&hyperplane);

                    if *flatten {
                        slice.flatten_into(&hyperplane.subspace);
                        slice.recenter_with(
                            &hyperplane.flatten(&hyperplane.project(&Point::zeros(dim))),
                        );
                    }

                    *p = slice;
                }
            }
        }
    }
}

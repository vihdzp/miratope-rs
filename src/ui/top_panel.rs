//! Contains all code related to the top bar.

use std::path::PathBuf;

use super::{camera::ProjectionType, memory::Memory, operations::*, UnitPointWidget};
use crate::{Concrete, Float, Hyperplane, Point, Vector};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, menu, Ui},
    EguiContext,
};
use miratope_core::{conc::{ConcretePolytope, faceting::GroupEnum}, file::FromFile, Polytope};

/// The plugin in charge of everything on the top panel.
pub struct TopPanelPlugin;

impl Plugin for TopPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FileDialogState>()
            .init_resource::<Memory>()
            .init_resource::<SectionDirection>()
            .init_resource::<SectionState>()
            .init_resource::<ExportMemory>()
            .init_non_send_resource::<FileDialogToken>()
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
        original_polytope: Concrete,

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

    pub fn open(&mut self, original_polytope: Concrete, minmax: (f64, f64)) {
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

/// Stores whether we're exporting the memory and the index of the memory slot.
pub struct ExportMemory(bool, usize);

impl Default for ExportMemory {
    fn default() -> Self {
        Self(false, 0)
    }
}

/// Contains all operations that manipulate file dialogs concretely.
///
/// Guarantees that file dialogs will be opened on the main thread, so as to
/// circumvent a MacOS limitation that all GUI operations must be done on the
/// main thread.
#[derive(Default)]
pub struct FileDialogToken(std::marker::PhantomData<*const ()>);

impl FileDialogToken {
    /// Auxiliary function to create a new file dialog.
    fn new_file_dialog() -> rfd::FileDialog {
        rfd::FileDialog::new()
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
    mut query: Query<'_, '_, &mut Concrete>,
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
                        match Concrete::from_path(&path) {
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
    ResMut<'a, CompoundWindow>,
    ResMut<'a, ScaleWindow>,
);

macro_rules! element_sort {
    ($p:ident) => {
        if !$p.abs().sorted() {
            $p.element_sort();
        }
    };
}

/// The system that shows the top panel.
#[allow(clippy::too_many_arguments)]
pub fn show_top_panel(
    // Info about the application state.
    egui_ctx: Res<'_, EguiContext>,
    mut query: Query<'_, '_, &mut Concrete>,
    keyboard: Res<'_, Input<KeyCode>>,

    // The Miratope resources controlled by the top panel.
    mut section_state: ResMut<'_, SectionState>,
    mut section_direction: ResMut<'_, SectionDirection>,
    mut file_dialog_state: ResMut<'_, FileDialogState>,
    mut projection_type: ResMut<'_, ProjectionType>,
    mut memory: ResMut<'_, Memory>,
    mut export_memory: ResMut<'_, ExportMemory>,
    mut background_color: ResMut<'_, ClearColor>,

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
        mut compound_window,
        mut scale_window,
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
                    file_dialog_state.save("polytope".to_string());
                }

                if ui.button("Export all memory slots").clicked() {
                    export_memory.0 = true;
                    export_memory.1 = 0;
                }

                ui.separator();

                // Quits the application.
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });

            if export_memory.0 {
                let idx = export_memory.1;
                if idx == memory.len() {
                    export_memory.1 = 0;
                    export_memory.0 = false;
                }
                else {
                    if let Some(poly) = &memory[idx] {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = poly.clone();
                            let mut name = "polytope ".to_owned();
                            name.push_str(&idx.to_string());
                            file_dialog_state.save(name.to_string());
                        }
                    }
                    export_memory.1 += 1;
                }
            }

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

            menu::menu(ui, "Scale", |ui| {
            
                if ui.button("Unit edge length").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    let e_l = (&p.vertices[p.abs[2][0].subs[0]] - &p.vertices[p.abs[2][0].subs[1]]).norm();
                    p.scale(1.0/e_l);
                }

                if ui.button("Unit circumradius").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    match p.circumsphere() {
                        Some(sphere) => {
                            p.scale(1.0/sphere.radius());
                        }
                        None => println!("The polytope has no circumsphere."),
                    }
                }

                // Opens a window to scale a polytope by some factor.
                if ui.button("Scale...").clicked() {
                    scale_window.open();
                }
                
                ui.separator();

                // Recenters a polytope.
                if ui.button("Recenter").clicked() {
                    query.iter_mut().next().unwrap().recenter();
                }
            });

            // Operations on polytopes.
            menu::menu(ui, "Operations", |ui| {
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
                        let flag = p.first_flag();
                        match p.petrie_polygon_with(flag) {
                            Some(q) => {
                                *p = q;
                                println!("Petrie polygon succeeded.")
                            }
                            None => eprintln!("Petrie polygon failed."),
                        }
                    }
                }

                ui.separator();

                // Makes a pyramid out of the current polytope.
                if ui.button("Pyramid").clicked() {
                    if advanced(&keyboard) {
                        pyramid_window.open();
                    } else if let Some(mut p) = query.iter_mut().next() {
                        *p = p.pyramid();
                    }
                }

                // Makes a prism out of the current polytope.
                if ui.button("Prism").clicked() {
                    if advanced(&keyboard) {
                        prism_window.open();
                    } else if let Some(mut p) = query.iter_mut().next() {
                        *p = p.prism();
                    }
                }

                // Makes a tegum out of the current polytope.
                if ui.button("Tegum").clicked() {
                    if advanced(&keyboard) {
                        tegum_window.open();
                    } else if let Some(mut p) = query.iter_mut().next() {
                        *p = p.tegum();
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
                
                ui.separator();

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

                // Opens the window to make compounds.
                if ui.button("Compound").clicked() {
                    compound_window.open();
                }

                ui.separator();

                if ui.button("Rectate").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    element_sort!(p);
                    *p = p.truncate_with(vec![1], vec![1.0]);
                }

                if ui.button("Truncate").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    element_sort!(p);
                    *p = p.truncate_with(vec![0, 1], vec![0.5, 0.5]);
                }

                if ui.button("Bitruncate").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    element_sort!(p);
                    *p = p.truncate_with(vec![1, 2], vec![0.5, 0.5]);
                }

                if ui.button("Cantellate").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    element_sort!(p);
                    *p = p.truncate_with(vec![0, 2], vec![0.5, 0.5]);
                }

                if ui.button("Runcinate").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    element_sort!(p);
                    *p = p.truncate_with(vec![0, 3], vec![0.5, 0.5]);
                }

                if ui.button("Omnitruncate").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    element_sort!(p);
                    *p = p.omnitruncate();
                }
            });

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

                        let minmax = p.minmax(direction.clone()).unwrap_or((-1.0, 1.0));
                        let original_polytope = p.clone();

                        section_state.open(original_polytope, minmax);
                        section_direction.0 = direction;
                    }
                };
            }

            // Operates on the elements of the loaded polytope.
            menu::menu(ui, "Elements", |ui| {
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
            menu::menu(ui, "Properties", |ui| {
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
                    if let Some(p) = query.iter_mut().next() {
                        println!("The polytope has {} flags.", p.flags().count())
                    }
                }

                // Gets the order of the symmetry group of the polytope
                if ui.button("Symmetry group").clicked() {
                    if let Some(mut p) = query.iter_mut().next() {
                        let group = p.get_symmetry_group().0;
                        println!("Symmetry order {}", group.count());
                    }
                }
            });

            menu::menu(ui, "Faceting", |ui| {
                if ui.button("Full faceting").clicked() {
                    if let Some(mut p) = query.iter_mut().next() {
                        let facetings = p.faceting(GroupEnum::None, None, None);
                        for faceting in facetings {
                            memory.push(faceting);
                        }
                    }
                }
                if ui.button("Superregiment faceting").clicked() {
                    if let Some(mut p) = query.iter_mut().next() {
                        let facetings = p.faceting(GroupEnum::None, Some(1.0), None);
                        for faceting in facetings {
                            memory.push(faceting);
                        }
                    }
                }
                if ui.button("Isotopic faceting").clicked() {
                    if let Some(mut p) = query.iter_mut().next() {
                        let facetings = p.faceting(GroupEnum::None, None, Some(1));
                        for faceting in facetings {
                            memory.push(faceting);
                        }
                    }
                }
            });

            memory.show(ui, &mut query);

            // Background color picker.

            // The current background color.
            let [r, g, b, a] = background_color.0.as_rgba_f32().map(|c| (c * 255.0) as u8);
            let color = egui::Color32::from_rgba_premultiplied(r, g, b, a);

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
    mut query: Query<'_, '_, &mut Concrete>,
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
                .minmax(section_direction.0.clone())
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
                        .minmax(section_direction.0.clone())
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

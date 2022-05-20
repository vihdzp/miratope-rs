//! Contains all code related to the top bar.

use std::path::PathBuf;

use super::{camera::ProjectionType, memory::Memory, window::{Window, *}, UnitPointWidget};
use crate::{Concrete, Float, Hyperplane, Point, Vector};

use bevy::prelude::*;
use bevy_egui::{egui::{self, menu, Ui}, EguiContext};
use miratope_core::{conc::{ConcretePolytope, faceting::GroupEnum, symmetry::Vertices}, file::FromFile, float::Float as Float2, Polytope, abs::Ranked};

/// The plugin in charge of everything on the top panel.
pub struct TopPanelPlugin;

impl Plugin for TopPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FileDialogState>()
            .init_resource::<SectionState>()
            .init_resource::<Vec<SectionDirection>>()
            .init_resource::<Memory>()
            .init_resource::<ShowMemory>()
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
        minmax: Vec<(Float, Float)>,

        /// The position of the slicing hyperplane.
        hyperplane_pos: Vec<Float>,

        /// Whether the cross-section is flattened into a dimension lower.
        flatten: bool,

        /// Whether we're not updating the cross-section.
        lock: bool,

        /// Whether to update the polytope. This is a bodge.
        update: bool,
    },

    /// The view is inactive.
    Inactive,
}

impl SectionState {
    /// Makes the view inactive.
    pub fn close(&mut self) {
        *self = Self::Inactive;
    }

	pub fn add(&mut self) {
		if let SectionState::Active {
            hyperplane_pos,
            minmax,
            ..
        } = self {
			minmax.push((0.0,0.0));
			hyperplane_pos.push(0.0);
		}
    }
	pub fn remove(&mut self) {
		if let SectionState::Active {
            hyperplane_pos,
            minmax,
            ..
        } = self {
			minmax.pop();
			hyperplane_pos.pop();
		}
    }

    pub fn open(&mut self, original_polytope: Concrete, minmax: Vec<(f64, f64)>) {
        *self = SectionState::Active {
            original_polytope,
            minmax: minmax.clone(),
            hyperplane_pos: minmax.clone().into_iter().map(|m| (m.0 + m.1) / 2.0).collect(),
            flatten: true,
            lock: false,
            update: false,
        }
    }
}

impl Clone for SectionState {
    fn clone(&self) -> Self {
		if let SectionState::Active{
				original_polytope,
				minmax,
				hyperplane_pos,
				flatten,
				lock,
                update,
			} = self{
				
			SectionState::Active{
				original_polytope: original_polytope.clone(),
				minmax: minmax.clone(),
				hyperplane_pos: hyperplane_pos.clone(),
				flatten: *flatten,
				lock: *lock,
                update: *update,
			}
		}
		else
		{
			SectionState::Inactive
		}
	}
}
impl Default for SectionState {
    fn default() -> Self {
        Self::Inactive
    }
}

/// Stores the direction in which the cross-sections are taken.
pub struct SectionDirection(pub Vector);

impl Default for SectionDirection {
    fn default() -> Self {
        Self(Vector::zeros(0))
    }
}

/// Stores whether the memory window is shown.
pub struct ShowMemory(bool);

impl Default for ShowMemory {
    fn default() -> Self {
        Self(false)
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
    ResMut<'a, TruncateWindow>,
    ResMut<'a, ScaleWindow>,
    ResMut<'a, FacetingSettings>,
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
    mut section_direction: ResMut<'_, Vec<SectionDirection>>,
    mut file_dialog_state: ResMut<'_, FileDialogState>,
    mut projection_type: ResMut<'_, ProjectionType>,
    mut memory: ResMut<'_, Memory>,
    mut show_memory: ResMut<'_, ShowMemory>,
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
        mut truncate_window,
        mut scale_window,
        mut faceting_settings,
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
                    if let Some((poly, label)) = &memory[idx] {
                        if let Some(mut p) = query.iter_mut().next() {
                            *p = poly.clone();
                            let name = match label {
                                None => {
                                    format!("polytope {}", idx)
                                }
                                Some(a) => a.to_string()
                            };
                            file_dialog_state.save(name);
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
				
                // Gets if it is a compound
                if ui.button("Is compound").clicked() {
                    if let Some(mut p) = query.iter_mut().next() {
						p.element_sort();
                        if p.abs.is_compound() {
							println!("The polytope is a compound.")
						} else {
							println!("The polytope is not a compound.")
						}
                    }
                }
				
                // Gets if it is fissary
                if ui.button("Is fissary").clicked() {
                    if let Some(mut p) = query.iter_mut().next() {
                        p.element_sort();
                        if p.is_fissary() {
							println!("The polytope is fissary.")
						} else {
							println!("The polytope is not fissary.")
						}
                    }
                }
            });

            menu::menu(ui, "Transform", |ui| {
            
                if ui.button("Scale to unit edge length").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    let e_l = (&p.vertices[p.abs[2][0].subs[0]] - &p.vertices[p.abs[2][0].subs[1]]).norm();
                    p.scale(1.0/e_l);
                }

                if ui.button("Scale to unit circumradius").clicked() {
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

                // Moves a polytope so that the circumcenter is at the origin.
                if ui.button("Recenter by circumcenter").clicked() {
                    let mut p = query.iter_mut().next().unwrap();
                    match p.circumsphere() {
                        Some(sphere) => {
                            p.recenter_with(&sphere.center);
                        }
                        None => println!("The polytope has no circumsphere."),
                    }
                }
                
                // Moves a polytope so that the gravicenter is at the origin.
                if ui.button("Recenter by gravicenter").clicked() {
                    query.iter_mut().next().unwrap().recenter();
                }
            });

            // Operations on polytopes.
            menu::menu(ui, "Operations", |ui| {
                // Converts the active polytope into its dual.
                if advanced(&keyboard) {
                    if ui.button("Dual...").clicked() {
                        dual_window.open();
                    }
                } else if let Some(mut p) = query.iter_mut().next() {
                    if ui.button("Dual").clicked() {
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
                        p.element_sort();
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
                if advanced(&keyboard) {
                    if ui.button("Pyramid...").clicked() {
                        pyramid_window.open();
                    }
                } else if let Some(mut p) = query.iter_mut().next() {
                    if ui.button("Pyramid").clicked() {
                        *p = p.pyramid();
                    }
                }

                // Makes a prism out of the current polytope.
                if advanced(&keyboard) {
                    if ui.button("Prism...").clicked() {
                        prism_window.open();
                    }
                } else if let Some(mut p) = query.iter_mut().next() {
                    if ui.button("Prism").clicked() {
                        *p = p.prism();
                    }
                }

                // Makes a tegum out of the current polytope.
                if advanced(&keyboard) {
                    if ui.button("Tegum...").clicked() {
                        tegum_window.open();
                    }
                } else if let Some(mut p) = query.iter_mut().next() {
                    if ui.button("Tegum").clicked() {
                        *p = p.tegum();
                    }
                }

                // Converts the active polytope into its antiprism.
                if advanced(&keyboard) {
                    if ui.button("Antiprism...").clicked() {
                        antiprism_window.open();
                    }
                } else if let Some(mut p) = query.iter_mut().next() {
                    if ui.button("Antiprism").clicked() {
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
                if ui.button("Duopyramid...").clicked() {
                    duopyramid_window.open();
                }

                // Opens the window to make duoprisms.
                if ui.button("Duoprism...").clicked() {
                    duoprism_window.open();
                }

                // Opens the window to make duotegums.
                if ui.button("Duotegum...").clicked() {
                    duotegum_window.open();
                }

                // Opens the window to make duocombs.
                if ui.button("Duocomb...").clicked() {
                    duocomb_window.open();
                }

                // Opens the window to make compounds.
                if ui.button("Compound...").clicked() {
                    compound_window.open();
                }

                ui.separator();

                if ui.button("Truncate...").clicked() {
                    truncate_window.open();
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

                        if p.rank() < 4 { // Cannot slice a polygon or lower.
                            println!("Slicing polytopes of rank less than 3 is not supported!");
                        } else {
                            p.flatten();

                            // The default direction is in the last coordinate axis.
                            let dim = p.dim_or();
                            let mut direction = Vector::zeros(dim);
                            if dim > 0 {
                                direction[dim - 1] = 1.0;
                            }
    
                            let minmax = p.minmax(direction.clone()).unwrap_or((-1.0, 1.0));
                            let original_polytope = p.clone();
    
                            section_state.open(original_polytope, vec![minmax]);
                            section_direction.clear();
                            section_direction.push(SectionDirection{0:direction});
                        }
                    }
                };
            }

            menu::menu(ui, "Faceting", |ui| {
                if ui.button("Enumerate facetings").clicked() {
                    if let Some(p) = query.iter_mut().next() {
                        let mut vertices_thing = (Vertices(vec![]), vec![]);
                        if let GroupEnum2::FromSlot(slot) = faceting_settings.group {
                            vertices_thing = Vertices(p.vertices.clone()).copy_by_symmetry(slot.to_poly(&mut memory, &p).unwrap().clone().get_symmetry_group().0);
                        }
                        let facetings = p.clone().faceting(
                            match faceting_settings.group {
                                GroupEnum2::Chiral(_) => p.vertices.clone(),
                                GroupEnum2::FromSlot(_) => vertices_thing.0.0
                            },
                            match faceting_settings.group {
                                GroupEnum2::Chiral(chiral) => GroupEnum::Chiral(chiral),
                                GroupEnum2::FromSlot(_) => GroupEnum::VertexMap(vertices_thing.1)
                            },
                            if faceting_settings.unit_edges {Some(1.0)} else {None}, 
                            if faceting_settings.max_facet_types == 0 {None} else {Some(faceting_settings.max_facet_types)},
                            if faceting_settings.max_per_hyperplane == 0 {None} else {Some(faceting_settings.max_per_hyperplane)},
                            faceting_settings.compounds,
                            faceting_settings.mark_fissary,
                            faceting_settings.save,
                            faceting_settings.save_facets,
			    faceting_settings.r
                        );
                        for faceting in facetings {
                            memory.push(faceting);
                        }
                    }
                }
                
                ui.separator();

                if ui.button("Settings...").clicked() {
                    faceting_settings.open();
                }
            });

            if ui.button("Memory").clicked() {
                show_memory.0 = !show_memory.0;
            }
            memory.show(&mut query, &egui_ctx, &mut show_memory.0);

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
    mut section_direction: ResMut<'_, Vec<SectionDirection>>
) {
    // The cross-section settings.
    if let SectionState::Active {
        minmax,
        hyperplane_pos,
        flatten,
        lock,
        ..
    } = (*section_state).clone()
    {
        ui.label("Cross section settings:");
        ui.spacing_mut().slider_width = ui.available_width() / 3.0;

        // Sets the slider range to the range of x coordinates in the polytope.
        let mut i = 0;

		while i < hyperplane_pos.len() {
			
			let mut new_hyperplane_pos = hyperplane_pos[i];
			ui.add(
				egui::Slider::new(
					&mut new_hyperplane_pos,
					(minmax[i].0 + 0.0000001)..=(minmax[i].1 - 0.0000001), // We do this to avoid empty slices.
				)
				.text("Slice depth")
				.prefix("pos: "),
			);

			// Updates the slicing depth.
			#[allow(clippy::float_cmp)]
			if hyperplane_pos[i] != new_hyperplane_pos {
				if let SectionState::Active { hyperplane_pos, .. } = section_state.as_mut() {
					hyperplane_pos[i] = new_hyperplane_pos;
				} else {
					unreachable!()
				}
			}

			let mut new_direction = section_direction[i].0.clone();

			ui.horizontal(|ui| {

				ui.add(UnitPointWidget::new(
					&mut new_direction,
					"Slice direction",
				));

				if ui.button("Diagonal").clicked() {
					new_direction = Point::from_element(new_direction.len(), 1.0/(new_direction.len() as f64).sqrt());
				}
			});
			
			// Updates the slicing direction.
			#[allow(clippy::float_cmp)]
			if section_direction[i].0 != new_direction {
				section_direction[i].0 = new_direction;
			}

			i = i + 1;
		}

        ui.horizontal(|ui| {
            // Makes the current cross-section into the main polytope.
            if ui.button("Make main").clicked() {
                section_state.close();
            }

            // Cross sections on a lower dimension
			if ui.add(egui::Button::new("+").enabled(
                section_direction.len() <
                    if let SectionState::Active {original_polytope, ..} = section_state.clone() {
                        original_polytope.rank()-3
                    } else {
                        0
                    }
                )).clicked() {
				let p = query.iter_mut().next().unwrap();
				let dim = p.dim_or();
				let mut direction = Vector::zeros(dim);
				if dim > 0 {
					direction[dim - 1] = 1.0;
				}
                section_state.add();
				section_direction.push(SectionDirection{0:direction});
            }
			// Cross sections on a higher dimension
			if ui.add(egui::Button::new("-").enabled(section_direction.len() > 1)).clicked() {
                section_state.remove();
                section_direction.pop();
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
            update,
            ..
        } = section_state.as_mut() {
            *update = true; // Force an update of the polytope.
        }
    }

    if section_state.is_changed() {
        if let SectionState::Active {
            original_polytope,
            hyperplane_pos,
            minmax,
            flatten,
            lock,
            update,
        } = section_state.as_mut() {
            *update = false;

            // We don't update the view if it's locked.
            if *lock {
                return;
            }

            if let Some(mut p) = query.iter_mut().next() {
                let mut r = original_polytope.clone();
				let mut i = 0;
                while i < hyperplane_pos.len() {
					let hyp_pos = hyperplane_pos[i];

					if let Some(dim) = r.dim() {
						let hyperplane = Hyperplane::new(section_direction[i].0.clone(), hyp_pos);
						minmax[i] = r
							.minmax(section_direction[i].0.clone())
							.unwrap_or((-1.0, 1.0));

						minmax[i].0 += f64::EPS;
						let mut slice = r.cross_section(&hyperplane);

						if *flatten {
							slice.flatten_into(&hyperplane.subspace);
							slice.recenter_with(
								&hyperplane.flatten(&hyperplane.project(&Point::zeros(dim))),
							);
						}

						r = slice;
					}
					i += 1;
				}
				*p = r;
            }
        }
    }
}

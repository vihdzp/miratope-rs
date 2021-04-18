#![allow(dead_code)]
//! A renderer for polytopes, spinned off from [Miratope JS](https://github.com/OfficialURL/miratope).
//! Still in alpha development.
//!
//! ## What can Miratope do now?
//! Miratope can already load some polytopes and find out various properties
//! about them, and it can operate on them via various methods. We're still in
//! the early stages of porting the original Miratope's functionality, though.
//!
//! ## What are Miratope's goals?
//! We plan to eventually support all of the original Miratope's features,
//! as well as the following:
//!
//! * Various families of polytopes to build and render
//!   * All [regular polytopes](https://polytope.miraheze.org/wiki/Regular_polytope)
//!   * All 3D and 4D known [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
//!   * Many of the known [CRFs](https://polytope.miraheze.org/wiki/Convex_regular-faced_polytope)
//! * Many operations to apply to these polytopes
//!   * [Duals](https://polytope.miraheze.org/wiki/Dual)
//!   * [Petrials](https://polytope.miraheze.org/wiki/Petrial)
//!   * [Prism products](https://polytope.miraheze.org/wiki/Prism_product)
//!   * [Tegum products](https://polytope.miraheze.org/wiki/Tegum_product)
//!   * [Pyramid products](https://polytope.miraheze.org/wiki/Pyramid_product)
//!   * [Convex hulls](https://polytope.miraheze.org/wiki/Convex_hull)
//! * Loading and saving into various formats
//!   * Support for the [Stella OFF format](https://www.software3d.com/StellaManual.php?prod=stella4D#import)
//!   * Support for the [GeoGebra GGB format](https://wiki.geogebra.org/en/Reference:File_Format)
//! * Localization
//!   * Automatic name generation in various languages for many shapes
//!
//! ## How do I use Miratope?
//! Miratope doesn't have a very good interface yet, so you'll have to download
//! the source code to do much of anything.
//!
//! ## Where do I get these "OFF files"?
//! The OFF file format is a format for storing certain kinds of geometric
//! shapes. Although not in widespread use, it has become the standard format
//! for those who investigate polyhedra and polytopes. It was initially meant
//! for the [Geomview software](https://people.sc.fsu.edu/~jburkardt/data/off/off.html),
//! and was later adapted for the [Stella software](https://www.software3d.com/StellaManual.php?prod=stella4D#import).
//! Miratope uses a further generalization of the Stella OFF format for any
//! amount of dimensions.
//!
//! Miratope does not yet include a library of OFF files. Nevertheless, many of
//! them can be downloaded from [OfficialURL's personal collection](https://drive.google.com/drive/u/0/folders/1nQZ-QVVBfgYSck4pkZ7he0djF82T9MVy).
//! Eventually, they'll be browsable from Miratope itself.
//!
//! ## Why is the rendering buggy?
//! Proper rendering, even in 3D, is a work in progress.

use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::{camera::PerspectiveProjection, pipeline::PipelineDescriptor};
use bevy_egui::{egui, EguiContext, EguiPlugin, EguiSettings};
use no_cull_pipeline::PbrNoBackfaceBundle;

use std::convert::TryInto;

#[allow(unused_imports)]
use polytope::{geometry::*, group::*, off::*, *};

mod input;
mod no_cull_pipeline;
mod polytope;

/// Standard constant used for floating point comparisons throughout the code.
const EPS: f64 = 1e-9;

struct CrossSectionActive(bool);

struct CrossSectionState {
    original_polytope: Option<Renderable>,
    hyperplane_pos: f32,
}

/// Loads all of the necessary systems for the application to run.
fn main() {
    App::build()
        // Sets the background color to black.
        .add_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        // Enables antialiasing.
        .add_resource(Msaa { samples: 4 })
        .add_resource(CrossSectionActive(false))
        .add_resource(CrossSectionState {
            original_polytope: None,
            hyperplane_pos: 0.0,
        })
        // Default Bevy plugins.
        .add_plugins(DefaultPlugins)
        // Enables egui to work in the first place.
        .add_plugin(EguiPlugin)
        // Enables camera controls.
        .add_plugin(input::InputPlugin)
        // Setups the initial state of the application.
        .add_startup_system(setup.system())
        // Scales the interface when the screen is resized.
        .add_system(update_ui_scale_factor.system())
        .add_system(update_cross_section.system())
        // Loads the User Interface.
        .add_system(ui.system())
        // Updates polytopes when operations are done to them.
        .add_system_to_stage(stage::POST_UPDATE, update_changed_polytopes.system())
        .add_system_to_stage(stage::POST_UPDATE, update_cross_section_state.system())
        .run();
}

const WIREFRAME_SELECTED_MATERIAL: HandleUntyped =
    HandleUntyped::weak_from_u64(StandardMaterial::TYPE_UUID, 0x82A3A5DD3A34CC21);
const WIREFRAME_UNSELECTED_MATERIAL: HandleUntyped =
    HandleUntyped::weak_from_u64(StandardMaterial::TYPE_UUID, 0x82A3A5DD3A34CC22);

/// Resizes the UI when the screen is resized.
fn update_ui_scale_factor(mut egui_settings: ResMut<EguiSettings>, windows: Res<Windows>) {
    if let Some(window) = windows.get_primary() {
        egui_settings.scale_factor = 1.0 / window.scale_factor();
    }
}

fn update_cross_section_state(
    mut query: Query<&mut Renderable>,
    mut state: ResMut<CrossSectionState>,
    active: ChangedRes<CrossSectionActive>,
) {
    if active.0 {
        state.original_polytope = Some(query.iter_mut().next().unwrap().clone());
    } else if let Some(t) = state.original_polytope.take() {
        *query.iter_mut().next().unwrap() = t;
    } else {
        println!("This should only happen on startup.");
    }
}

fn update_cross_section(
    mut query: Query<&mut Renderable>,
    state: Res<CrossSectionState>,
    active: Res<CrossSectionActive>,
) {
    if !active.0 {
        return;
    }

    for mut p in query.iter_mut() {
        let r = state.original_polytope.clone().unwrap();
        let mut hyp_pos = state.hyperplane_pos.into();
        hyp_pos += 0.00001;
        *p = Renderable::new(r.concrete.slice(Hyperplane::x(
            r.concrete.rank().try_into().unwrap(),
            hyp_pos,
        )));
    }
}

/// A system for a basic UI.
fn ui(
    mut egui_ctx: ResMut<EguiContext>,
    mut query: Query<&mut Renderable>,
    mut state: ResMut<CrossSectionState>,
    mut section_active: ResMut<CrossSectionActive>,
) {
    let ctx = &mut egui_ctx.ctx;

    egui::TopPanel::top("top_panel").show(ctx, |ui| {
        // The top panel is often a good place for a menu bar:
        egui::menu::bar(ui, |ui| {
            egui::menu::menu(ui, "File", |ui| {
                if ui.button("Quit").clicked() {
                    std::process::exit(0);
                }
            });
        });

        // Converts the active polytope into its dual.
        if ui.button("Dual").clicked() {
            for mut p in query.iter_mut() {
                match p.concrete.dual_mut() {
                    Ok(_) => println!("Dual succeeded"),
                    Err(_) => println!("Dual failed"),
                }
                // Crashes for some reason.
                // println!("{}", &p.concrete.to_src(off::OffOptions { comments: true }));
            }
        }
        // Converts the active polytope into any of its verfs.
        if ui.button("Verf").clicked() {
            for mut p in query.iter_mut() {
                println!("Verf");

                if let Some(verf) = p.concrete.verf(0) {
                    *p = Renderable::new(verf);
                };
            }
        }
        // Exports the active polytope as an OFF file (not yet functional!)
        if ui.button("Export OFF").clicked() {
            for _p in query.iter_mut() {
                println!("Export OFF");
            }
        }
        // Toggles cross-section mode.
        if ui.button("Cross-section").clicked() {
            section_active.0 = !section_active.0;
            println!("{}", section_active.0);
        }

        ui.add(egui::Slider::f32(&mut state.hyperplane_pos, -0.25..=0.25).text("slice"));
    });
}

/// Initializes the scene.
fn setup(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
) {
    let mut p = Concrete::hypercube(3);
    let q = Concrete::orthoplex(3);
    p.append(q).unwrap();
    let poly = Renderable::new(p);

    pipelines.set_untracked(
        no_cull_pipeline::NO_CULL_PIPELINE_HANDLE,
        no_cull_pipeline::build_no_cull_pipeline(&mut shaders),
    );

    materials.set_untracked(
        WIREFRAME_SELECTED_MATERIAL,
        Color::rgb_u8(126, 192, 236).into(),
    );

    let wf_unselected = materials.set(
        WIREFRAME_UNSELECTED_MATERIAL,
        Color::rgb_u8(56, 68, 236).into(),
    );

    commands
        .spawn(PbrNoBackfaceBundle {
            mesh: meshes.add(poly.get_mesh()),
            visible: Visible {
                is_visible: false,
                ..Default::default()
            },
            material: materials.add(Color::rgb(0.93, 0.5, 0.93).into()),
            ..Default::default()
        })
        .with_children(|cb| {
            cb.spawn(PbrNoBackfaceBundle {
                mesh: meshes.add(poly.get_wireframe()),
                material: wf_unselected,
                ..Default::default()
            });
        })
        .with(poly)
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 2.0)),
            ..Default::default()
        })
        // camera anchor
        .spawn((
            GlobalTransform::default(),
            Transform::from_translation(Vec3::new(0.02, -0.025, -0.05))
                * Transform::from_translation(Vec3::new(-0.02, 0.025, 0.05))
                    .looking_at(Vec3::default(), Vec3::unit_y()),
        ))
        .with_children(|cb| {
            // camera
            cb.spawn(Camera3dBundle {
                transform: Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
                perspective_projection: PerspectiveProjection {
                    near: 0.0001,
                    ..Default::default()
                },
                ..Default::default()
            });
        });
}

/// Updates polytopes after an operation.
fn update_changed_polytopes(
    mut meshes: ResMut<Assets<Mesh>>,
    polies: Query<(&Renderable, &Handle<Mesh>, &Children), Changed<Renderable>>,
    wfs: Query<&Handle<Mesh>, Without<Renderable>>,
) {
    for (poly, mesh_handle, children) in polies.iter() {
        let mesh: &mut Mesh = meshes.get_mut(mesh_handle).unwrap();
        *mesh = poly.get_mesh();

        for child in children.iter() {
            if let Ok(wf_handle) = wfs.get_component::<Handle<Mesh>>(*child) {
                let wf: &mut Mesh = meshes.get_mut(wf_handle).unwrap();
                *wf = poly.get_wireframe();

                break;
            }
        }
    }
}

/*
fn update_text(mut ctx: ResMut<EguiContext>) {
    todo!()
}
*/

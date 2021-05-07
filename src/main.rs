#![allow(dead_code)]
//! A renderer for [polytopes](https://polytope.miraheze.org/wiki/Polytope).
//! Still in alpha development.
//!
//! ## What are Miratope's goals?
//! Miratope can already load many polytopes, find out various properties about
//! them, and can operate on them via various methods. We plan to eventually
//! support all of the following:
//!
//! * An extensive library of polytopes to build and render, including
//!   * All [regular polytopes](https://polytope.miraheze.org/wiki/Regular_polytope)
//!   * All known 3D and 4D [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
//!   * Many of the known [CRFs](https://polytope.miraheze.org/wiki/Convex_regular-faced_polytope)
//! * Many operations to apply to these polytopes
//!   * [Duals](https://polytope.miraheze.org/wiki/Dual) ✅
//!   * [Petrials](https://polytope.miraheze.org/wiki/Petrial)
//!   * [Prism products](https://polytope.miraheze.org/wiki/Prism_product) ✅
//!   * [Tegum products](https://polytope.miraheze.org/wiki/Tegum_product) ✅
//!   * [Pyramid products](https://polytope.miraheze.org/wiki/Pyramid_product) ✅
//!   * [Convex hulls](https://polytope.miraheze.org/wiki/Convex_hull)
//! * Loading and saving into various formats
//!   * Support for the [Stella OFF format](https://www.software3d.com/StellaManual.php?prod=stella4D#import) ✅
//!   * Support for the [GeoGebra GGB format](https://wiki.geogebra.org/en/Reference:File_Format)
//! * Localization
//!   * Automatic name generation in various languages for many shapes
//!
//! ## How do I use Miratope?
//! Miratope's user interface is still a work in progress, so you'll have to
//! download the source code to do much of anything. Miratope is written in the
//! latest version of Rust. Download instructions can be found here:
//! [https://www.rust-lang.org/tools/install]. **You may have to restart your
//! computer for Rust to fully install**.
//!
//! 1. Once you have Rust set up, click the green button here on Github that
//!    says "Code".
//!    * If you already have Github Desktop, you can just click "Open with
//!      Github Desktop".
//!    * If you don't, click "Download ZIP", and once it's finished downloading,
//!      extract the `.zip` file.
//! 2. Next, open a command line. On Windows you can do this by opening Run with
//!    `Win+R` and typing `cmd` in the search box.
//! 3. In the command line, navigate to the Miratope folder you downloaded. In
//!    Windows, this can be done with the command `cd [FILE PATH]`. If you don't
//!    know how to get the file path, go in your files, open the unzipped
//!    Miratope file folder, and click on the address bar at the top. Copy the
//!    highlighted file path, paste it into the command line in place of `[FILE
//!    PATH]`, and press Enter. The last name in the command header should now
//!    be the name of the folder Miratope is in.
//! 4. Finally, type `cargo run --release` and hit Enter. It will take a while
//!    for the computer to open Miratope for the first time, but after that
//!    opening it should be a lot faster. Once you have completed all the steps
//!    you will only need to do step 4 to run Miratope from start up (but if the
//!    `[FILE PATH]` changes, you'll need to do step 3 again).
//!
//! These steps are in place because it would be too costly now to update an
//! executable file each time a bug is fixed or feature is added. Once Miratope
//! leaves the alpha stage, a simple executable for Version 1.0 will be provided.
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
//! Soon, they'll be browsable from Miratope itself.
//!
//! ## Why is the rendering buggy?
//! Proper rendering, even in 3D, is a work in progress.

use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::{camera::PerspectiveProjection, pipeline::PipelineDescriptor};
use bevy_egui::EguiPlugin;
use lang::SelectedLanguage;
use no_cull_pipeline::PbrNoBackfaceBundle;

#[allow(unused_imports)]
use lang::{name::Con, Language, Name, Options};
#[allow(unused_imports)]
use polytope::{
    concrete::{off::*, Concrete},
    Polytope, *,
};
use ui::{
    camera::{CameraInputEvent, ProjectionType},
    library::Library,
    *,
};

mod geometry;
mod lang;
mod no_cull_pipeline;
mod polytope;
mod ui;

/// The wiki link.
const WIKI_LINK: &str = "https://polytope.miraheze.org/wiki/";

/// A trait containing the constants associated to each floating point type.
trait Consts {
    type T;
    const EPS: Self::T;
    const PI: Self::T;
    const TAU: Self::T;
    const SQRT_2: Self::T;
}

/// Constants for `f64`.
impl Consts for f64 {
    type T = f64;
    const EPS: f64 = 1e-9;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const SQRT_2: f64 = std::f64::consts::SQRT_2;
}

/// Constants for `f32`.
impl Consts for f32 {
    type T = f32;
    const EPS: f32 = 1e-5;
    const PI: f32 = std::f32::consts::PI;
    const TAU: f32 = std::f32::consts::TAU;
    const SQRT_2: f32 = std::f32::consts::SQRT_2;
}

/// The floating point type used for all calculations.
type Float = f64;

/// Loads all of the necessary systems for the application to run.
fn main() {
    App::build()
        // Adds resources.
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(SectionState::default())
        .insert_resource(FileDialogState::default())
        .insert_resource(ProjectionType::Perspective)
        .insert_resource(Library::new_folder(&"./lib/"))
        .insert_resource(SelectedLanguage::default())
        .insert_non_send_resource(ui::MainThreadToken::new())
        // Adds plugins.
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(ui::camera::InputPlugin)
        // Adds systems
        .add_startup_system(setup.system())
        .add_system(ui::update_scale_factor.system())
        .add_system(ui::ui.system())
        .add_system(ui::file_dialog.system())
        .add_system(ui::update_language.system())
        .add_system_to_stage(CoreStage::PostUpdate, ui::update_cross_section.system())
        .add_system_to_stage(CoreStage::PostUpdate, ui::update_changed_polytopes.system())
        .run();
}

/// Initializes the scene.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
) {
    let poly = Concrete::uniform_antiprism(5);

    // Disables backface culling.
    pipelines.set_untracked(
        no_cull_pipeline::NO_CULL_PIPELINE_HANDLE,
        no_cull_pipeline::build_no_cull_pipeline(&mut shaders),
    );

    // Selected object (unused as of yet).
    materials.set_untracked(
        WIREFRAME_SELECTED_MATERIAL,
        Color::rgb_u8(126, 192, 255).into(),
    );

    // Unselected object (default material).
    let wf_unselected = materials.set(
        WIREFRAME_UNSELECTED_MATERIAL,
        Color::rgb_u8(240, 240, 240).into(),
    );

    // Camera configuration.
    let mut cam_anchor = Transform::default();
    let mut cam = Transform::default();
    CameraInputEvent::reset(&mut cam_anchor, &mut cam);

    commands
        .spawn()
        // Mesh
        .insert_bundle(PbrNoBackfaceBundle {
            mesh: meshes.add(poly.get_mesh(ProjectionType::Perspective)),
            material: materials.add(Color::rgb(0.93, 0.5, 0.93).into()),
            visible: Visible {
                is_visible: false,
                ..Default::default()
            },
            ..Default::default()
        })
        // Wireframe
        .with_children(|cb| {
            cb.spawn().insert_bundle(PbrNoBackfaceBundle {
                mesh: meshes.add(poly.get_wireframe(ProjectionType::Perspective)),
                material: wf_unselected,
                visible: Visible {
                    is_visible: true,
                    ..Default::default()
                },
                ..Default::default()
            });
        })
        // Polytope
        .insert(poly);

    // Light source
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 2.0)),
        ..Default::default()
    });

    // Camera anchor
    commands
        .spawn()
        .insert_bundle((GlobalTransform::default(), cam_anchor))
        .with_children(|cb| {
            // camera
            cb.spawn_bundle(PerspectiveCameraBundle {
                transform: cam,
                perspective_projection: PerspectiveProjection {
                    near: 0.0001,
                    ..Default::default()
                },
                ..Default::default()
            });
        });
}

const WIREFRAME_SELECTED_MATERIAL: HandleUntyped =
    HandleUntyped::weak_from_u64(StandardMaterial::TYPE_UUID, 0x82A3A5DD3A34CC21);
const WIREFRAME_UNSELECTED_MATERIAL: HandleUntyped =
    HandleUntyped::weak_from_u64(StandardMaterial::TYPE_UUID, 0x82A3A5DD3A34CC22);

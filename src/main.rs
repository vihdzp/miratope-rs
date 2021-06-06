#![allow(dead_code)]
//! A tool for building and visualizing polytopes. Still in alpha development.
//!
//! ## What can Miratope do now?
//! Miratope can already load polytopes from files and derive various properties from them, as well as do various operations on them. It can render wireframes in arbitrary dimension, though it can only rotate them in three of those dimensions.
//!
//! ## What are Miratope's goals?
//! We plan to eventually support all of the following:
//!
//! * Various families of polytopes to build and render
//!   * [ ] All [regular polytopes](https://polytope.miraheze.org/wiki/Regular_polytope)
//!   * [ ] All known 3D and 4D [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
//!   * [ ] Many of the known [CRFs](https://polytope.miraheze.org/wiki/Convex_regular-faced_polytope)
//! * Many operations to apply to these polytopes
//!   * [x] [Duals](https://polytope.miraheze.org/wiki/Dual)
//!   * [x] [Petrials](https://polytope.miraheze.org/wiki/Petrial)
//!   * [x] [Prism products](https://polytope.miraheze.org/wiki/Prism_product)
//!   * [x] [Tegum products](https://polytope.miraheze.org/wiki/Tegum_product)
//!   * [x] [Pyramid products](https://polytope.miraheze.org/wiki/Pyramid_product)
//!   * [x] [Antiprisms](https://polytope.miraheze.org/wiki/Antiprism)
//!   * [ ] [Convex hulls](https://polytope.miraheze.org/wiki/Convex_hull)
//! * Loading and saving into various formats
//!   * [x] Support for the [Stella `.off` format](https://www.software3d.com/StellaManual.php?prod=stella4D#import)
//!   * [ ] Support for the [GeoGebra `.ggb` format](https://wiki.geogebra.org/en/Reference:File_Format)
//! * Localization
//!   * Automatic name generation in various languages for many shapes
//!     * [x] English
//!     * [ ] Spanish
//!     * [ ] French
//!     * [ ] Japanese
//!     * [ ] Proto Indo-Iranian
//!
//! ## How do I use Miratope?
//! Miratope is in the alpha stage, and so doesn't have a completed interface yet. You'll have to download the source code to do much of anything.
//!
//! Miratope is written in Rust, so if you don't already have the latest version and its Visual Studio C++ Build tools downloaded then you should do that first. Instructions for downloading can be found here: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install). **You may have to restart your computer for Rust to fully install**.
//!
//! 1. Once you have Rust setup click the green button here on Github that says "Code".
//!    * If you already have Github Desktop, you can just click "Open with Github Desktop".
//!    * If you don't, click "Download ZIP" and once it's done downloading, extract the `.zip` file.
//! 2. Next, open a command line. On Windows you can do this by opening Run with `Win+R` and typing `cmd` in the search box.
//! 3. In the command line, first type `cd [FILE PATH]`. If you don't know how to get the file path, in your files go open the unzipped Miratope file folder, and click on the address bar at the top. Copy the highlighted file path and paste it into the command line in place of `[FILE PATH]`, and press Enter. The last name in the command header should now be the name of the folder Miratope is in.
//! 4. Finally, type `cargo run` and hit Enter. It will take a while for the computer to open Miratope for the first time, but after that, opening it should be a lot faster. A window should appear, if the version of Miratope you downloaded was a stable one. If it wasn't, you'll get an error, and you should wait until the devs have fixed whatever they broke.
//!
//! Once you have completed all the steps you will only need to do step 4 to run Miratope from startup (but if the `[FILE PATH]` changes, you'll need to do step 3 again).
//!
//! These steps are in place because it would be too cumbersome at this stage to update the executable each time a bug is fixed or feature is added. Once Miratope leaves the alpha stage, executable files for Version 1.0.0 will be provided.
//!
//! ## Where do I get these "`.off` files"?
//! The **O**bject **F**ile **F**ormat is a format for storing certain kinds of geometric shapes. Although not in widespread use, it has become the standard format for those interested in polyhedra and polytopes. It was initially meant for the [Geomview software](https://people.sc.fsu.edu/~jburkardt/data/off/off.html), and was later adapted for the [Stella software](https://www.software3d.com/StellaManual.php?prod=stella4D#import). Miratope uses a further generalization of the Stella `.off` format for any amount of dimensions.
//!
//! Miratope includes a small library simple or generatable polytopes at startup. More complicated polytopes can be downloaded from [OfficialURL's personal collection](https://drive.google.com/drive/u/0/folders/1nQZ-QVVBfgYSck4pkZ7he0djF82T9MVy). Eventually, most files here will be browsable from Miratope itself.
//!
//! ## Why is the rendering buggy?
//! Proper rendering, even in 3D, is a work in progress.

use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::{camera::PerspectiveProjection, pipeline::PipelineDescriptor};
use bevy_egui::EguiPlugin;
use no_cull_pipeline::PbrNoBackfaceBundle;

use polytope::concrete::Concrete;
use ui::{
    camera::{CameraInputEvent, ProjectionType},
    MiratopePlugins,
};

mod geometry;
mod lang;
mod no_cull_pipeline;
mod polytope;
mod ui;

/// The link to the [Polytope Wiki](https://polytope.miraheze.org/wiki/).
const WIKI_LINK: &str = "https://polytope.miraheze.org/wiki/";

/// The link to the GitHub issues.
const NEW_ISSUE: &str = "https://github.com/OfficialURL/miratope-rs/issues/new";

/// A trait containing the constants associated to each floating point type.
trait Consts {
    type T;
    const EPS: Self::T;
    const PI: Self::T;
    const TAU: Self::T;
    const SQRT_2: Self::T;
}

/// Constants for `f32`.
impl Consts for f32 {
    type T = f32;
    const EPS: f32 = 1e-5;
    const PI: f32 = std::f32::consts::PI;
    const TAU: f32 = std::f32::consts::TAU;
    const SQRT_2: f32 = std::f32::consts::SQRT_2;
}

/// Constants for `f64`.
impl Consts for f64 {
    type T = f64;
    const EPS: f64 = 1e-9;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const SQRT_2: f64 = std::f64::consts::SQRT_2;
}

/// The floating point type used for all calculations.
type Float = f64;

/// Loads all of the necessary systems for the application to run.
fn main() {
    App::build()
        // Adds resources.
        .insert_resource(Msaa { samples: 4 })
        // Adds plugins.
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugins(MiratopePlugins)
        // Adds systems.
        .add_startup_system(setup.system())
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
    // Default polytope.
    let poly = Concrete::from_off(include_str!("default.off")).unwrap();

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

    // Wireframe material.
    let wf_material = materials.set(WIREFRAME_UNSELECTED_MATERIAL, Color::rgb_u8(0, 0, 0).into());

    // Mesh material.
    let mesh_material = materials.add(StandardMaterial {
        base_color: Color::rgb_u8(255, 255, 255),
        metallic: 0.2,
        ..Default::default()
    });

    // Camera configuration.
    let mut cam_anchor = Default::default();
    let mut cam = Default::default();
    CameraInputEvent::reset(&mut cam_anchor, &mut cam);

    commands
        .spawn()
        // Mesh
        .insert_bundle(PbrNoBackfaceBundle {
            mesh: meshes.add(poly.mesh(ProjectionType::Perspective)),
            material: mesh_material,
            ..Default::default()
        })
        // Wireframe
        .with_children(|cb| {
            cb.spawn().insert_bundle(PbrNoBackfaceBundle {
                mesh: meshes.add(poly.wireframe(ProjectionType::Perspective)),
                material: wf_material,
                ..Default::default()
            });
        })
        // Polytope
        .insert(poly);

    // Camera anchor
    commands
        .spawn()
        .insert_bundle((GlobalTransform::default(), cam_anchor))
        .with_children(|cb| {
            // Camera
            cb.spawn_bundle(PerspectiveCameraBundle {
                transform: cam,
                perspective_projection: PerspectiveProjection {
                    near: 0.0001,
                    ..Default::default()
                },
                ..Default::default()
            });
            // Light source
            cb.spawn_bundle(PointLightBundle {
                transform: Transform::from_translation(Vec3::new(-50.0, 50.0, 50.0)),
                point_light: PointLight {
                    intensity: 10000.,
                    range: 100.,
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

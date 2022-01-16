#![deny(
    missing_docs,
    nonstandard_style,
    unused_parens,
    unused_qualifications,
    rust_2018_idioms,
    rust_2018_compatibility,
    future_incompatible,
    missing_copy_implementations
)]

//! A tool for building and visualizing polytopes. Still in alpha development.

use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::{camera::PerspectiveProjection, pipeline::PipelineDescriptor};
use bevy_egui::EguiPlugin;
use miratope_core::file::FromFile;
use no_cull_pipeline::PbrNoBackfaceBundle;

use ui::{
    camera::{CameraInputEvent, ProjectionType},
    MiratopePlugins,
};

use crate::mesh::Renderable;

mod mesh;
mod no_cull_pipeline;
mod ui;

/// The link to the [Polytope Wiki](https://polytope.miraheze.org/wiki/).
pub const WIKI_LINK: &str = "https://polytope.miraheze.org/wiki/";

/// The floating-point type for the entire application. Can be either `f32` or
/// `f64`, and it should compile the same.
type Float = f64;

/// A [`Concrete`](miratope_core::conc::Concrete) polytope with the floating
/// type for the application.
type Concrete = miratope_core::conc::Concrete;

/// A [`Point`](miratope_core::geometry::Point) polytope with the floating type
/// for the application.
type Point = miratope_core::geometry::Point<f64>;

/// A [`Vector`](miratope_core::geometry::Vector) polytope with the floating
/// type for the application.
type Vector = miratope_core::geometry::Vector<f64>;

/// A [`Hypersphere`](miratope_core::geometry::Hypersphere) polytope with the
/// floating type for the application.
type Hypersphere = miratope_core::geometry::Hypersphere<f64>;

/// A [`Hyperplane`](miratope_core::geometry::Hyperplane) polytope with the
/// floating type for the application.
type Hyperplane = miratope_core::geometry::Hyperplane<f64>;

/// The default epsilon value throughout the application.
const EPS: Float = <Float as miratope_core::float::Float>::EPS;

/// Loads all of the necessary systems for the application to run.
fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugins(MiratopePlugins)
        .add_startup_system(setup.system())
        .run();
}

/// Initializes the scene.
fn setup(
    mut commands: Commands<'_, '_>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
    mut shaders: ResMut<'_, Assets<Shader>>,
    mut pipelines: ResMut<'_, Assets<PipelineDescriptor>>,
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
                    far: 10000.0,
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

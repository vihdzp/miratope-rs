use bevy::prelude::*;
use bevy::render::camera::Camera;
use bevy::render::pipeline::{RenderPipeline ,PipelineDescriptor};
use polytope::shapes::*;
use polytope::*;

mod polytope;
mod no_cull_pipeline;

fn main() {
    App::build()
        .add_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .add_system(spin_camera.system())
        .run();
}

fn setup(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
) {
    //let poly = oct();
    /*
    let poly: Polytope = ron::from_str::<polytope::PolytopeSerde>(
        "(
            vertices: [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ],
            elements: [[
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 3],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ], [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 4, 8, 9],
                    [1, 5, 9, 10],
                    [2, 6, 10, 11],
                    [3, 7, 11, 8],
                ], [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            ],
        )",
    )
    .unwrap()
    .into();
    */
    let poly = antiprism(21, 3);

    pipelines.set_untracked(
        no_cull_pipeline::NO_CULL_PIPELINE_HANDLE,
        no_cull_pipeline::build_no_cull_pipeline(&mut shaders),
    );

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(poly.into()),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                no_cull_pipeline::NO_CULL_PIPELINE_HANDLE.typed(),
            )]),
            ..Default::default()
        })
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 2.0)),
            ..Default::default()
        })
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 5.0))
                .looking_at(Vec3::default(), Vec3::unit_y()),
            ..Default::default()
        });
}

fn spin_camera(mut query: Query<&mut Transform, With<Camera>>, time: Res<Time>) {
    const SPIN_RATE: f32 = std::f32::consts::PI * 2.0 / 3.0;

    for mut tf in query.iter_mut() {
        tf.translation = Quat::from_rotation_y(time.delta_seconds() * SPIN_RATE) * tf.translation;
        tf.look_at(Vec3::zero(), Vec3::unit_y());
    }
}

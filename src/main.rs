use bevy::prelude::*;
use bevy::render::camera::{Camera, PerspectiveProjection};
use polytope::PolytopeC;
use ultraviolet::DVec3;

mod polytope;

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
) {
    let vertices = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    ];
    let edges = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ];
    let faces = vec![
        vec![0, 1, 3],
        vec![0, 2, 4],
        vec![1, 2, 5],
        vec![3, 4, 5],
    ];

    let tet = PolytopeC::new(vertices, edges, faces);

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(tet.into()),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
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

fn spin_camera(mut query: Query<&mut Transform, With<Camera>>) {
    for mut tf in query.iter_mut() {
        tf.translation = Quat::from_rotation_y(0.01) * tf.translation;
        tf.look_at(Vec3::zero(), Vec3::unit_y());
    }
}

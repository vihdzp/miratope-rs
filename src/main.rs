use bevy::{prelude::*, render::camera::PerspectiveProjection};
use polytope::PolytopeC;
use ultraviolet::DVec3;

mod polytope;

fn main() {
    App::build()
        .add_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .run();
}

/// set up a simple 3D scene
fn setup(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    /* let tet = PolytopeC {
        vertices: vec![
            DVec3::new(3.0, 3.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        ],
        edges: vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        faces: vec![vec![0, 1, 3], vec![0, 2, 4], vec![1, 2, 5], vec![3, 4, 5]],

        triangles: vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    }; */

    let square = PolytopeC {
        vertices: vec![
            DVec3 {
                x: 0.1924022424786771,
                y: 0.6914529439214506,
                z: 0.022877420064978837,
            },
            DVec3 {
                x: 0.6574954634882417,
                y: 0.5316961553837378,
                z: 0.2749649445613721,
            },
            DVec3 {
                x: 0.8530329884018142,
                y: 0.3404575752385256,
                z: 0.3337797270193349,
            },
            DVec3 {
                x: 0.2842763894475524,
                y: 0.4825184215559576,
                z: 0.5003836319988735,
            },
        ],
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        faces: vec![vec![0, 1, 2, 3]],

        triangles: vec![[0, 1, 2], [1, 2, 3]],
    };

    // add entities to the world
    commands
        // cube
        .spawn(PbrBundle {
            mesh: meshes.add(square.into()),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            ..Default::default()
        })
        // light
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 2.0)),
            ..Default::default()
        })
        // camera
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 2.0))
                .looking_at(Vec3::default(), Vec3::unit_y()),
            perspective_projection: PerspectiveProjection {
                near: 0.001,
                far: 100.0,
                ..Default::default()
            },
            ..Default::default()
        });
}

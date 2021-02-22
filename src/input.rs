use std::borrow::BorrowMut;
use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel, MouseScrollUnit};
use bevy::render::camera::Camera;

/// The plugin handling all input.
///
/// This plugin handles just camera controls at the moment, but when we set up
/// a GUI at some point, this will also handle the input for that.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app
            .add_event::<CameraInputEvent>()
            .add_system_to_stage(stage::EVENT, add_cam_input_events.system())
            .add_system(update_cameras_and_anchors.system());
    }
}

/// An input event for the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraInputEvent {
    /// Rotate the camera about the anchor.
    RotateAnchor(Vec2),
    /// Translate the camera wrt. its perspective.
    ///
    /// The translation happens with respect to the perspective of the camera,
    /// so a translation of (1, 0, 0) is likely not going to change the global
    /// transform's translation by (1, 0, 0).
    Translate(Vec3),
    /// Roll the camera's view.
    Roll(f32),
    /// Zoom the camera.
    ///
    /// The zoom tapers with distance: closer in zooms slow, etc.
    Zoom(f32),
}

impl std::ops::Mul<f32> for CameraInputEvent {
    type Output = CameraInputEvent;

    fn mul(mut self, rhs: f32) -> CameraInputEvent {
        match &mut self {
            CameraInputEvent::RotateAnchor(r) => *r *= rhs,
            CameraInputEvent::Translate(p) => *p *= rhs,
            CameraInputEvent::Roll(r) | CameraInputEvent::Zoom(r) => *r *= rhs,
        }

        self
    }
}

impl std::ops::Mul<CameraInputEvent> for f32 {
    type Output = CameraInputEvent;

    fn mul(self, rhs: CameraInputEvent) -> CameraInputEvent {
        rhs * self
    }
}

impl CameraInputEvent {
    fn rotate(vec: Vec2, anchor_tf: &mut Transform) {
        anchor_tf.rotate(Quat::from_rotation_ypr(vec.x, vec.y, 0.0));
    }

    fn translate(vec: Vec3, anchor_tf: &mut Transform, cam_gtf: &GlobalTransform) {
        anchor_tf.translation += cam_gtf.rotation * vec;
    }

    fn roll(roll: f32, anchor_tf: &mut Transform) {
        anchor_tf.rotate(Quat::from_rotation_ypr(0.0, 0.0, roll));
    }

    fn zoom(zoom: f32, cam_tf: &mut Transform) {
        cam_tf.translation.z += zoom * cam_tf.translation.length();
        cam_tf.translation.z = cam_tf.translation.z.max(0.2);
    }

    fn update_camera_and_anchor(
        &self,
        anchor_tf: &mut Transform,
        cam_tf: &mut Transform,
        cam_gtf: &GlobalTransform,
    ) {
        match *self {
            CameraInputEvent::RotateAnchor(vec) =>
                CameraInputEvent::rotate(vec, anchor_tf),
            CameraInputEvent::Translate(vec) =>
                CameraInputEvent::translate(vec, anchor_tf, cam_gtf),
            CameraInputEvent::Roll(roll) =>
                CameraInputEvent::roll(roll, anchor_tf),
            CameraInputEvent::Zoom(zoom) =>
                CameraInputEvent::zoom(zoom, cam_tf),
        }
    }
}

fn cam_events_from_kb(
    time: Res<Time>,
    keyboard: Res<Input<KeyCode>>,
    cam_inputs: &mut Events<CameraInputEvent>,
) -> (f32, f32) {
    const SPIN_RATE: f32 = std::f32::consts::TAU / 3.0;
    let real_scale = time.delta_seconds();
    let scale = if keyboard.pressed(KeyCode::LShift) | keyboard.pressed(KeyCode::RShift) {
        real_scale * 3.0
    } else {
        real_scale
    };

    let fb = CameraInputEvent::Translate(Vec3::unit_z());
    let lr = CameraInputEvent::Translate(Vec3::unit_x());
    let ud = CameraInputEvent::Translate(Vec3::unit_y());
    const ROLL: CameraInputEvent = CameraInputEvent::Roll(SPIN_RATE);

    for keycode in keyboard.get_pressed() {
        cam_inputs.send(match keycode {
            KeyCode::W | KeyCode::Up | KeyCode::Numpad8 =>
                -scale * fb,
            KeyCode::S | KeyCode::Down | KeyCode::Numpad2 =>
                scale * fb,
            KeyCode::D | KeyCode::Right | KeyCode::Numpad6 =>
                scale * lr,
            KeyCode::A | KeyCode::Left | KeyCode::Numpad4 =>
                -scale * lr,
            KeyCode::Space =>
                scale * ud,
            KeyCode::LControl =>
                -scale * ud,
            KeyCode::Q | KeyCode::Numpad7 =>
                real_scale * ROLL,
            KeyCode::E | KeyCode::Numpad9 =>
                -real_scale * ROLL,
            _ => continue,
        })
    }

    (real_scale, scale)
}

fn cam_events_from_mouse(
    mouse_button: Res<Input<MouseButton>>,
    mouse_move: Res<Events<MouseMotion>>,
    width: f32,
    height: f32,
    real_scale: f32,
    cam_inputs: &mut Events<CameraInputEvent>,
) {
    if mouse_button.pressed(MouseButton::Right) {
        for &MouseMotion { mut delta } in mouse_move.get_reader().iter(&mouse_move) {
            delta.x /= width;
            delta.y /= height;
            cam_inputs.send(CameraInputEvent::RotateAnchor(70.0 * real_scale * delta))
        }
    }
}

fn cam_events_from_wheel(
    mouse_wheel: Res<Events<MouseWheel>>,
    scale: f32,
    cam_inputs: &mut Events<CameraInputEvent>,
) {
    for MouseWheel { unit, y, .. } in mouse_wheel.get_reader().iter(&mouse_wheel) {
        let unit_scale = match unit {
            MouseScrollUnit::Line => 12.0,
            MouseScrollUnit::Pixel => 1.0,
        };     

        cam_inputs.send(CameraInputEvent::Zoom(unit_scale * -scale * y))
    }
}

fn add_cam_input_events(
    time: Res<Time>,
    keyboard: Res<Input<KeyCode>>,
    mouse_button: Res<Input<MouseButton>>,
    mouse_move: Res<Events<MouseMotion>>,
    mouse_wheel: Res<Events<MouseWheel>>,
    windows: Res<Windows>,
    mut cam_inputs: ResMut<Events<CameraInputEvent>>,
) {
    let (width, height) = {
        let primary_win = windows.get_primary().expect("There is no primary window");
        (primary_win.physical_width() as f32, primary_win.physical_height() as f32)
    };

    let cam_inputs = &mut cam_inputs;
    let (real_scale, scale) = cam_events_from_kb(time, keyboard, cam_inputs);
    cam_events_from_mouse(mouse_button, mouse_move, width, height, real_scale, cam_inputs);
    cam_events_from_wheel(mouse_wheel, scale, cam_inputs);
}

fn update_cameras_and_anchors(
    events: Res<Events<CameraInputEvent>>,
    q: Query<(&mut Transform, &GlobalTransform, Option<&Parent>, Option<&Camera>)>,
) {
    for (mut cam_tf, cam_gtf, parent, cam) in unsafe { q.iter_unsafe() } {
        if cam.is_none() {
            continue;
        } else if let Some(parent) = parent {
            if let Ok(mut anchor_tf) = unsafe { q.get_component_unsafe::<Transform>(parent.0) } {
                for event in events.get_reader().iter(&events) {
                    event.update_camera_and_anchor(anchor_tf.borrow_mut(), cam_tf.borrow_mut(), cam_gtf);
                }
            }
        }
    }
}

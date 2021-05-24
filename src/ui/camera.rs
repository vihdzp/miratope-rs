//! Contains the methods to setup the camera.

use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    math::EulerRot,
    prelude::*,
    render::camera::Camera,
};
use bevy_egui::{egui::CtxRef, EguiContext};

#[derive(Clone, Copy)]
pub enum ProjectionType {
    /// We're projecting orthogonally.
    Orthogonal,
    Perspective,
}

impl ProjectionType {
    pub fn flip(&mut self) {
        match self {
            Self::Orthogonal => *self = Self::Perspective,
            Self::Perspective => *self = Self::Orthogonal,
        }
    }

    pub fn is_orthogonal(&self) -> bool {
        match self {
            Self::Orthogonal => true,
            Self::Perspective => false,
        }
    }
}

/// The plugin handling all input.
///
/// This plugin handles just camera controls at the moment, but when we set up
/// a GUI at some point, this will also handle the input for that.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_event::<CameraInputEvent>()
            .add_system_to_stage(CoreStage::PostUpdate, add_cam_input_events.system())
            .add_system(update_cameras_and_anchors.system());
    }
}

/// An input event for the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraInputEvent {
    /// Rotate the camera about the anchor.
    RotateAnchor(Vec2),

    /// Translate the camera with respect to its perspective.
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

    /// Resets the camera to its default state.
    Reset,
}

impl std::ops::Mul<f32> for CameraInputEvent {
    type Output = CameraInputEvent;

    /// Scales the effect of a camera input event by a certain factor.
    fn mul(mut self, rhs: f32) -> CameraInputEvent {
        match &mut self {
            CameraInputEvent::RotateAnchor(r) => *r *= rhs,
            CameraInputEvent::Translate(p) => *p *= rhs,
            CameraInputEvent::Roll(r) | CameraInputEvent::Zoom(r) => *r *= rhs,
            _ => {}
        }

        self
    }
}

impl std::ops::Mul<CameraInputEvent> for f32 {
    type Output = CameraInputEvent;

    /// Scales the effect of a camera input event by a certain factor.
    fn mul(self, rhs: CameraInputEvent) -> CameraInputEvent {
        rhs * self
    }
}

impl CameraInputEvent {
    fn rotate(vec: Vec2, anchor_tf: &mut Transform) {
        anchor_tf.rotate(Quat::from_euler(EulerRot::YXZ, vec.x, vec.y, 0.0));
    }

    fn translate(vec: Vec3, anchor_tf: &mut Transform, cam_gtf: &GlobalTransform) {
        anchor_tf.translation += cam_gtf.rotation * vec;
    }

    fn roll(roll: f32, anchor_tf: &mut Transform) {
        anchor_tf.rotate(Quat::from_euler(EulerRot::YXZ, 0.0, 0.0, roll));
    }

    /// Zooms into the camera.
    fn zoom(zoom: f32, cam_tf: &mut Transform) {
        cam_tf.translation.z += zoom * cam_tf.translation.length();
        cam_tf.translation.z = cam_tf.translation.z.max(0.2);
    }

    /// Resets the camera to the default position.
    pub fn reset(anchor_tf: &mut Transform, cam_tf: &mut Transform) {
        *cam_tf = Transform::from_translation(Vec3::new(0.0, 0.0, 5.0));
        *anchor_tf = Transform::from_translation(Vec3::new(0.02, -0.025, -0.05))
            * Transform::from_translation(Vec3::new(-0.02, 0.025, 0.05))
                .looking_at(Vec3::default(), Vec3::Y);
    }

    fn update_camera_and_anchor(
        &self,
        anchor_tf: &mut Transform,
        cam_tf: &mut Transform,
        cam_gtf: &GlobalTransform,
    ) {
        match *self {
            CameraInputEvent::RotateAnchor(vec) => CameraInputEvent::rotate(vec, anchor_tf),
            CameraInputEvent::Translate(vec) => {
                CameraInputEvent::translate(vec, anchor_tf, cam_gtf)
            }
            CameraInputEvent::Roll(roll) => CameraInputEvent::roll(roll, anchor_tf),
            CameraInputEvent::Zoom(zoom) => CameraInputEvent::zoom(zoom, cam_tf),
            CameraInputEvent::Reset => CameraInputEvent::reset(anchor_tf, cam_tf),
        }
    }
}

/// Processes camera events coming from the keyboard.
fn cam_events_from_kb(
    time: Res<Time>,
    keyboard: Res<Input<KeyCode>>,
    cam_inputs: &mut EventWriter<CameraInputEvent>,
    ctx: &CtxRef,
) -> (f32, f32) {
    const SPIN_RATE: f32 = std::f32::consts::TAU / 5.0;
    let real_scale = time.delta_seconds();
    let scale = if keyboard.pressed(KeyCode::LControl) | keyboard.pressed(KeyCode::RControl) {
        real_scale * 3.0
    } else {
        real_scale
    };

    let fb = CameraInputEvent::Translate(Vec3::Z);
    let lr = CameraInputEvent::Translate(Vec3::X);
    let ud = CameraInputEvent::Translate(Vec3::Y);
    const ROLL: CameraInputEvent = CameraInputEvent::Roll(SPIN_RATE);

    if !ctx.wants_keyboard_input() {
        for keycode in keyboard.get_pressed() {
            cam_inputs.send(match keycode {
                KeyCode::W | KeyCode::Up => -scale * fb,
                KeyCode::S | KeyCode::Down => scale * fb,
                KeyCode::D | KeyCode::Right => scale * lr,
                KeyCode::A | KeyCode::Left => -scale * lr,
                KeyCode::Space => scale * ud,
                KeyCode::LShift | KeyCode::RShift => -scale * ud,
                KeyCode::Q => real_scale * ROLL,
                KeyCode::E => -real_scale * ROLL,
                KeyCode::R => CameraInputEvent::Reset,
                _ => continue,
            })
        }
    }

    (real_scale, scale)
}

/// Processes camera events coming from the mouse buttons.
fn cam_events_from_mouse(
    mouse_button: Res<Input<MouseButton>>,
    mut mouse_move: EventReader<MouseMotion>,
    width: f32,
    height: f32,
    real_scale: f32,
    cam_inputs: &mut EventWriter<CameraInputEvent>,
) {
    if mouse_button.pressed(MouseButton::Right) {
        for &MouseMotion { mut delta } in mouse_move.iter() {
            delta.x /= width;
            delta.y /= height;
            cam_inputs.send(CameraInputEvent::RotateAnchor(-100.0 * real_scale * delta))
        }
    }
}

/// Processes camera events coming from the mouse wheel.
fn cam_events_from_wheel(
    mut mouse_wheel: EventReader<MouseWheel>,
    scale: f32,
    cam_inputs: &mut EventWriter<CameraInputEvent>,
) {
    for MouseWheel { unit, y, .. } in mouse_wheel.iter() {
        let unit_scale = match unit {
            MouseScrollUnit::Line => 12.0,
            MouseScrollUnit::Pixel => 1.0,
        };

        cam_inputs.send(CameraInputEvent::Zoom(unit_scale * -scale * y))
    }
}

#[allow(clippy::too_many_arguments)]
fn add_cam_input_events(
    time: Res<Time>,
    keyboard: Res<Input<KeyCode>>,
    mouse_button: Res<Input<MouseButton>>,
    mouse_move: EventReader<MouseMotion>,
    mouse_wheel: EventReader<MouseWheel>,
    windows: Res<Windows>,
    mut cam_inputs: EventWriter<CameraInputEvent>,
    egui_ctx: ResMut<EguiContext>,
) {
    let (width, height) = {
        let primary_win = windows.get_primary().expect("There is no primary window");
        (
            primary_win.physical_width() as f32,
            primary_win.physical_height() as f32,
        )
    };

    let ctx = egui_ctx.ctx();
    let cam_inputs = &mut cam_inputs;
    let (real_scale, scale) = cam_events_from_kb(time, keyboard, cam_inputs, ctx);

    // Omit any events if the UI will process them instead.
    if !ctx.wants_pointer_input() {
        cam_events_from_mouse(
            mouse_button,
            mouse_move,
            width,
            height,
            real_scale,
            cam_inputs,
        );
        cam_events_from_wheel(mouse_wheel, scale, cam_inputs);
    }
}

fn update_cameras_and_anchors(
    mut events: EventReader<CameraInputEvent>,
    q: Query<(
        &mut Transform,
        &GlobalTransform,
        Option<&Parent>,
        Option<&Camera>,
    )>,
) {
    for (mut cam_tf, cam_gtf, parent, cam) in unsafe { q.iter_unsafe() } {
        if cam.is_none() {
            continue;
        } else if let Some(parent) = parent {
            if let Ok(mut anchor_tf) =
                unsafe { q.get_component_unchecked_mut::<Transform>(parent.0) }
            {
                for event in events.iter() {
                    event.update_camera_and_anchor(&mut anchor_tf, &mut cam_tf, cam_gtf);
                }
            }
        }
    }
}

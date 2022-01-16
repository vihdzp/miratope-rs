//! Contains the methods to setup the camera.

use std::ops::Mul;

use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    math::EulerRot,
    prelude::*,
    render::camera::Camera,
};
use bevy_egui::{egui::CtxRef, EguiContext};

/// The plugin handling all camera input.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<CameraInputEvent>()
            .init_resource::<ProjectionType>()
            // We register inputs after the library has been shown, so that we
            // know whether mouse input should register.
            .add_system(add_cam_input_events.system().after("show_library"))
            .add_system(update_cameras_and_anchors.system());
    }
}

#[derive(Clone, Copy)]
pub enum ProjectionType {
    /// We're projecting orthogonally.
    Orthogonal,

    /// We're projecting from a point.
    Perspective,
}

impl Default for ProjectionType {
    fn default() -> Self {
        Self::Perspective
    }
}

impl ProjectionType {
    /// Flips the projection type.
    pub fn flip(&mut self) {
        match self {
            Self::Orthogonal => *self = Self::Perspective,
            Self::Perspective => *self = Self::Orthogonal,
        }
    }

    /// Returns whether the projection type is `Orthogonal`.
    pub fn is_orthogonal(&self) -> bool {
        matches!(self, Self::Orthogonal)
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

impl Mul<f32> for CameraInputEvent {
    type Output = Self;

    /// Scales the effect of a camera input event by a certain factor.
    fn mul(mut self, rhs: f32) -> Self {
        match &mut self {
            Self::RotateAnchor(r) => *r *= rhs,
            Self::Translate(p) => *p *= rhs,
            Self::Roll(r) | Self::Zoom(r) => *r *= rhs,
            _ => {}
        }

        self
    }
}

impl Mul<CameraInputEvent> for f32 {
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
            Self::RotateAnchor(vec) => Self::rotate(vec, anchor_tf),
            Self::Translate(vec) => Self::translate(vec, anchor_tf, cam_gtf),
            Self::Roll(roll) => Self::roll(roll, anchor_tf),
            Self::Zoom(zoom) => Self::zoom(zoom, cam_tf),
            Self::Reset => Self::reset(anchor_tf, cam_tf),
        }
    }

    /// Processes camera events coming from the keyboard.
    fn cam_events_from_kb(
        time: &Time,
        keyboard: &Input<KeyCode>,
        cam_inputs: &mut EventWriter<'_, '_, CameraInputEvent>,
        ctx: &CtxRef,
    ) -> (f32, f32) {
        // TODO: make the spin rate modifiable in preferences.
        const SPIN_RATE: f32 = std::f32::consts::TAU / 5.0;
        const ROLL: CameraInputEvent = CameraInputEvent::Roll(SPIN_RATE);

        let real_scale = time.delta_seconds();
        let scale = if keyboard.pressed(KeyCode::LControl) | keyboard.pressed(KeyCode::RControl) {
            real_scale * 3.0
        } else {
            real_scale
        };

        let fb = Self::Translate(Vec3::Z);
        let lr = Self::Translate(Vec3::X);
        let ud = Self::Translate(Vec3::Y);

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
                    KeyCode::R => Self::Reset,
                    _ => continue,
                })
            }
        }

        (real_scale, scale)
    }

    /// Processes camera events coming from the mouse buttons.
    fn cam_events_from_mouse(
        mouse_button: &Input<MouseButton>,
        mut mouse_move: EventReader<'_, '_, MouseMotion>,
        width: f32,
        height: f32,
        real_scale: f32,
        cam_inputs: &mut EventWriter<'_, '_, Self>,
    ) {
        if mouse_button.pressed(MouseButton::Right) {
            for MouseMotion { mut delta } in mouse_move.iter() {
                delta.x /= width;
                delta.y /= height;
                cam_inputs.send(Self::RotateAnchor(-400.0 * real_scale * delta))
            }
        }
    }

    /// Processes camera events coming from the mouse wheel.
    fn cam_events_from_wheel(
        mut mouse_wheel: EventReader<'_, '_, MouseWheel>,
        scale: f32,
        cam_inputs: &mut EventWriter<'_, '_, Self>,
    ) {
        for MouseWheel { unit, y, .. } in mouse_wheel.iter() {
            let unit_scale = match unit {
                MouseScrollUnit::Line => 12.0,
                MouseScrollUnit::Pixel => 1.0,
            };

            cam_inputs.send(Self::Zoom(unit_scale * -scale * y))
        }
    }
}

/// The system that processes all input from the mouse and keyboard.
#[allow(clippy::too_many_arguments)]
fn add_cam_input_events(
    time: Res<'_, Time>,
    keyboard: Res<'_, Input<KeyCode>>,
    mouse_button: Res<'_, Input<MouseButton>>,
    mouse_move: EventReader<'_, '_, MouseMotion>,
    mouse_wheel: EventReader<'_, '_, MouseWheel>,
    windows: Res<'_, Windows>,
    mut cam_inputs: EventWriter<'_, '_, CameraInputEvent>,
    egui_ctx: Res<'_, EguiContext>,
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
    let (real_scale, scale) =
        CameraInputEvent::cam_events_from_kb(&time, &keyboard, cam_inputs, ctx);

    // Omit any events if the UI will process them instead.
    if !ctx.wants_pointer_input() {
        CameraInputEvent::cam_events_from_mouse(
            &mouse_button,
            mouse_move,
            width,
            height,
            real_scale,
            cam_inputs,
        );
        CameraInputEvent::cam_events_from_wheel(mouse_wheel, scale, cam_inputs);
    }
}

fn update_cameras_and_anchors(
    mut events: EventReader<'_, '_, CameraInputEvent>,
    q: Query<
        '_,
        '_,
        (
            &mut Transform,
            &GlobalTransform,
            Option<&Parent>,
            Option<&Camera>,
        ),
    >,
) {
    // SAFETY: see the remark below.
    for (mut cam_tf, cam_gtf, parent, cam) in unsafe { q.iter_unsafe() } {
        if cam.is_some() {
            if let Some(parent) = parent {
                // SAFETY: we assume that a camera isn't its own parent (this
                // shouldn't ever happen on purpose)
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
}

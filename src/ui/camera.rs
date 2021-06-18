//! Contains the methods to setup the camera.

use std::ops::AddAssign;

use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    render::camera::{Camera, PerspectiveProjection, VisibleEntities},
};
use bevy_egui::{egui::CtxRef, EguiContext};

use crate::no_cull_pipeline::{GlobalTransform7, Trans7, Transform7};

/// The plugin handling all camera input.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_event::<CameraInputEvent>()
            .insert_resource(ProjectionType::Perspective)
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
        match self {
            Self::Orthogonal => true,
            Self::Perspective => false,
        }
    }
}

/// Component bundle for camera entities with perspective projection
///
/// Use this for 3D rendering.
#[derive(Bundle)]
pub struct PerspectiveCameraBundle7 {
    pub camera: Camera,
    pub perspective_projection: PerspectiveProjection,
    pub visible_entities: VisibleEntities,
    pub transform: Transform7,
    pub global_transform: GlobalTransform7,
}

impl Default for PerspectiveCameraBundle7 {
    fn default() -> Self {
        Self {
            camera: Camera {
                name: Some("Camera7d".to_string()),
                ..Default::default()
            },
            perspective_projection: Default::default(),
            visible_entities: Default::default(),
            transform: Default::default(),
            global_transform: Default::default(),
        }
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
    Translate(Vec7),

    /// Roll the camera's view.
    // Roll(f32),

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
            // CameraInputEvent::Roll(r) | CameraInputEvent::Zoom(r) => *r *= rhs,
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

/// The controls for nD rotation are essentially the same as those in 3D, except
/// that a subspace different from the XYZ one is affected. This struct stores
/// the indices of the axes defining the subspace that we rotate, in the order
/// that they were added.
#[derive(Clone, Copy)]
pub struct RotDirections([usize; 3]);

impl Default for RotDirections {
    fn default() -> Self {
        Self([0, 1, 2])
    }
}

impl std::ops::Index<usize> for RotDirections {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for RotDirections {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl RotDirections {
    /// Pushes an index at the front of the direction queue.
    pub fn push(&mut self, idx: usize) {
        if self[1] == idx {
            self.0.swap(0, 1);
        } else {
            self[2] = self[1];
            self[1] = self[0];
            self[0] = idx;
        }
    }

    pub fn iter(&self) -> std::slice::Iter<usize> {
        self.0.iter()
    }
}

type Vec7 = nalgebra::SVector<f32, 7>;

impl CameraInputEvent {
    /// Rotates the camera by the specified amount in the specified directions.
    fn rotate(vec: Vec2, directions: RotDirections, anchor_tf: &mut Transform7) {
        anchor_tf.rotate(Transform7::from_euler(directions, vec.x, vec.y));
    }

    fn translate(vec: Vec7, anchor_tf: &mut Transform7, cam_gtf: &GlobalTransform7) {
        anchor_tf
            .translation_mut()
            .add_assign((cam_gtf.rotation() * vec).transpose());
    }

    /* fn roll(roll: f32, directions: RotDirections, anchor_tf: &mut Transform7) {
        anchor_tf.rotate(Transform7::from_euler(directions, 0.0, 0.0, roll));
    } */

    /// Zooms into the camera.
    fn zoom(zoom: f32, cam_tf: &mut Transform7) {
        let norm = cam_tf.translation().norm();
        let z_trans = &mut cam_tf.translation_mut()[2];
        *z_trans += zoom * norm;
        *z_trans = (*z_trans).max(0.2);
    }

    /// Resets the camera to the default position.
    pub fn reset(anchor_tf: &mut Transform7, cam_tf: &mut Transform7) {
        /*   *cam_tf = Transform::from_translation(Vec3::new(0.0, 0.0, 5.0));
        *anchor_tf = Transform::from_translation(Vec3::new(0.02, -0.025, -0.05))
            * Transform::from_translation(Vec3::new(-0.02, 0.025, 0.05))
                .looking_at(Vec3::default(), Vec3::Y); */

        *anchor_tf = Default::default();
        *cam_tf = Default::default();
    }

    fn update_camera_and_anchor(
        &self,
        anchor_tf: &mut Transform7,
        cam_tf: &mut Transform7,
        cam_gtf: &GlobalTransform7,
        directions: RotDirections,
    ) {
        match *self {
            Self::RotateAnchor(vec) => Self::rotate(vec, directions, anchor_tf),
            Self::Translate(vec) => Self::translate(vec, anchor_tf, cam_gtf),
            // Self::Roll(roll) => Self::roll(roll, anchor_tf),
            Self::Zoom(zoom) => Self::zoom(zoom, cam_tf),
            Self::Reset => Self::reset(anchor_tf, cam_tf),
        }
    }
}

/// Processes camera events coming from the keyboard.
fn cam_events_from_kb(
    ctx: &CtxRef,

    time: Res<Time>,
    keyboard: Res<Input<KeyCode>>,
    cam_inputs: &mut EventWriter<CameraInputEvent>,
) -> (f32, f32) {
    // TODO: make the spin rate modifiable in preferences.
    const SPIN_RATE: f32 = std::f32::consts::TAU / 5.0;
    // const ROLL: CameraInputEvent = CameraInputEvent::Roll(SPIN_RATE);

    let real_scale = time.delta_seconds();
    let scale = if keyboard.pressed(KeyCode::LControl) | keyboard.pressed(KeyCode::RControl) {
        real_scale * 3.0
    } else {
        real_scale
    };

    let lr = CameraInputEvent::Translate(*Vec7::x_axis());
    let ud = CameraInputEvent::Translate(*Vec7::y_axis());
    let fb = CameraInputEvent::Translate(*Vec7::z_axis());

    if !ctx.wants_keyboard_input() {
        for keycode in keyboard.get_pressed() {
            cam_inputs.send(match keycode {
                KeyCode::W | KeyCode::Up => -scale * fb,
                KeyCode::S | KeyCode::Down => scale * fb,
                KeyCode::D | KeyCode::Right => scale * lr,
                KeyCode::A | KeyCode::Left => -scale * lr,
                KeyCode::Space => scale * ud,
                KeyCode::LShift | KeyCode::RShift => -scale * ud,
                /* KeyCode::Q => real_scale * ROLL,
                KeyCode::E => -real_scale * ROLL, */
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
        for MouseMotion { mut delta } in mouse_move.iter() {
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

/// The system that processes all input from the mouse and keyboard.
#[allow(clippy::too_many_arguments)]
fn add_cam_input_events(
    egui_ctx: Res<EguiContext>,

    time: Res<Time>,
    keyboard: Res<Input<KeyCode>>,
    mouse_button: Res<Input<MouseButton>>,
    mouse_move: EventReader<MouseMotion>,
    mouse_wheel: EventReader<MouseWheel>,
    windows: Res<Windows>,

    mut cam_inputs: EventWriter<CameraInputEvent>,
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
    let (real_scale, scale) = cam_events_from_kb(ctx, time, keyboard, cam_inputs);

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
        &mut Transform7,
        &GlobalTransform7,
        Option<&Parent>,
        Option<&Camera>,
    )>,
    directions: Res<RotDirections>,
) {
    // SAFETY: see the remark below.
    for (mut cam_tf, cam_gtf, parent, cam) in unsafe { q.iter_unsafe() } {
        if cam.is_none() {
            continue;
        } else if let Some(parent) = parent {
            // SAFETY: we assume that a camera isn't its own parent (this
            // shouldn't ever happen on purpose)
            if let Ok(mut anchor_tf) =
                unsafe { q.get_component_unchecked_mut::<Transform7>(parent.0) }
            {
                for event in events.iter() {
                    event.update_camera_and_anchor(
                        &mut anchor_tf,
                        &mut cam_tf,
                        cam_gtf,
                        *directions,
                    );
                }
            }
        }
    }
}

//! Sets up the windows that permit more advanced settings.

use super::PointWidget;
use crate::{
    geometry::{Hypersphere, Point},
    polytope::{concrete::Concrete, r#abstract::rank::Rank},
    Float,
};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, CtxRef, Layout, TextStyle, Ui, Widget},
    EguiContext,
};
pub struct EguiWindowPlugin;

impl Plugin for EguiWindowPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.insert_resource(EguiWindows::default())
            .add_system_to_stage(
                CoreStage::Update,
                show_windows.system().label("show_windows"),
            )
            .add_system_to_stage(CoreStage::PostUpdate, update_windows.system());
    }
}

pub struct OkReset<'a>(&'a mut ShowResult);

impl<'a> OkReset<'a> {
    pub fn new(result: &'a mut ShowResult) -> Self {
        Self(result)
    }
}

impl<'a> Widget for OkReset<'a> {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        ui.allocate_ui_with_layout(ui.min_size(), Layout::right_to_left(), |ui| {
            if ui.button("Ok").clicked() {
                *self.0 = ShowResult::Ok;
            } else if ui.button("Reset").clicked() {
                *self.0 = ShowResult::Reset;
            }
        })
        .response
    }
}

fn resize(point: &mut Point, rank: Rank) {
    *point = point
        .clone()
        .resize_vertically(rank.try_usize().unwrap_or(0), 0.0);
}

pub trait WindowType: Into<WindowTypeId> {
    /// The number of dimensions of the polytope on screen, used to set up the
    /// window.
    fn rank(&self) -> Rank;

    /// The default state of the window, when the polytope on the screen has a
    /// given rank.
    fn default_with(rank: Rank) -> Self;

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Self::default_with(self.rank())
    }

    /// Shows the window on screen.
    fn show(&mut self, ctx: &CtxRef) -> ShowResult;

    /// Updates the window's settings after the polytope's dimension is updated.
    fn update(&mut self, rank: Rank);
}

pub struct DualWindow {
    center: Point,
    radius: Float,
}

impl WindowType for DualWindow {
    fn rank(&self) -> Rank {
        Rank::from(self.center.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            center: Point::zeros(rank.try_usize().unwrap_or(0)),
            radius: 1.0,
        }
    }

    fn show(&mut self, ctx: &CtxRef) -> ShowResult {
        let mut open = true;
        let mut result = ShowResult::None;

        egui::Window::new("Dual")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add(PointWidget::new(&mut self.center, "Center:"));

                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(
                        egui::DragValue::new(&mut self.radius)
                            .speed(0.01)
                            .clamp_range(0.0..=Float::MAX),
                    );
                });

                ui.add(OkReset::new(&mut result));
            });

        if open {
            result
        } else {
            ShowResult::Close
        }
    }

    fn update(&mut self, rank: Rank) {
        resize(&mut self.center, rank);
    }
}

impl From<DualWindow> for WindowTypeId {
    fn from(dual: DualWindow) -> Self {
        WindowTypeId::Dual(dual)
    }
}

pub struct PyramidWindow {
    offset: Point,
    height: Float,
}

impl WindowType for PyramidWindow {
    fn rank(&self) -> Rank {
        Rank::from(self.offset.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            offset: Point::zeros(rank.try_usize().unwrap_or(0)),
            height: 1.0,
        }
    }

    fn show(&mut self, ctx: &CtxRef) -> ShowResult {
        let mut open = true;
        let mut result = ShowResult::None;

        egui::Window::new("Pyramid")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add(PointWidget::new(&mut self.offset, "Offset:"));

                ui.horizontal(|ui| {
                    ui.label("Height:");
                    ui.add(
                        egui::DragValue::new(&mut self.height)
                            .speed(0.01)
                            .clamp_range(0.0..=Float::MAX),
                    );
                });

                ui.add(OkReset::new(&mut result));
            });

        if open {
            result
        } else {
            ShowResult::Close
        }
    }

    fn update(&mut self, rank: Rank) {
        resize(&mut self.offset, rank);
    }
}

impl From<PyramidWindow> for WindowTypeId {
    fn from(pyramid: PyramidWindow) -> Self {
        WindowTypeId::Pyramid(pyramid)
    }
}

pub struct PrismWindow {
    height: Float,
}

impl WindowType for PrismWindow {
    fn rank(&self) -> Rank {
        Default::default()
    }

    fn default_with(_: Rank) -> Self {
        Default::default()
    }

    fn show(&mut self, ctx: &CtxRef) -> ShowResult {
        let mut open = true;
        let mut result = ShowResult::None;

        egui::Window::new("Prism")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Height:");
                    ui.add(
                        egui::DragValue::new(&mut self.height)
                            .speed(0.01)
                            .clamp_range(0.0..=Float::MAX),
                    );
                });

                ui.add(OkReset::new(&mut result));
            });

        if open {
            result
        } else {
            ShowResult::Close
        }
    }

    fn update(&mut self, _: Rank) {}
}

impl From<PrismWindow> for WindowTypeId {
    fn from(prism: PrismWindow) -> Self {
        WindowTypeId::Prism(prism)
    }
}

impl std::default::Default for PrismWindow {
    fn default() -> Self {
        Self { height: 1.0 }
    }
}

pub struct TegumWindow {
    offset: Point,
    height: Float,
    height_offset: Float,
}

impl WindowType for TegumWindow {
    fn rank(&self) -> Rank {
        Rank::from(self.offset.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            offset: Point::zeros(rank.try_usize().unwrap_or(0)),
            height: 1.0,
            height_offset: 0.0,
        }
    }

    fn show(&mut self, ctx: &CtxRef) -> ShowResult {
        let mut open = true;
        let mut result = ShowResult::None;

        egui::Window::new("Tegum")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add(PointWidget::new(&mut self.offset, "Offset:"));

                ui.horizontal(|ui| {
                    ui.label("Height:");
                    ui.add(
                        egui::DragValue::new(&mut self.height)
                            .speed(0.01)
                            .clamp_range(0.0..=Float::MAX),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Height offset:");
                    ui.add(egui::DragValue::new(&mut self.height_offset).speed(0.01));
                });

                ui.add(OkReset::new(&mut result));
            });

        if open {
            result
        } else {
            ShowResult::Close
        }
    }

    fn update(&mut self, rank: Rank) {
        resize(&mut self.offset, rank);
    }
}

impl From<TegumWindow> for WindowTypeId {
    fn from(tegum: TegumWindow) -> Self {
        Self::Tegum(tegum)
    }
}

pub struct AntiprismWindow {
    dual: DualWindow,
    height: Float,
    central_inversion: bool,
}

impl WindowType for AntiprismWindow {
    fn rank(&self) -> Rank {
        self.dual.rank()
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            dual: DualWindow::default_with(rank),
            height: 1.0,
            central_inversion: false,
        }
    }

    fn show(&mut self, ctx: &CtxRef) -> ShowResult {
        let mut open = true;
        let mut result = ShowResult::None;

        egui::Window::new("Antiprism")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add(PointWidget::new(&mut self.dual.center, "Center:"));

                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(
                        egui::DragValue::new(&mut self.dual.radius)
                            .speed(0.01)
                            .clamp_range(0.0..=Float::MAX),
                    );

                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut self.height).speed(0.01));

                    ui.add(
                        egui::Checkbox::new(&mut self.central_inversion, "Central inversion:")
                            .text_style(TextStyle::Body),
                    );
                });

                ui.add(OkReset::new(&mut result));
            });

        if open {
            result
        } else {
            ShowResult::Close
        }
    }

    fn update(&mut self, rank: Rank) {
        self.dual.update(rank);
    }
}

impl From<AntiprismWindow> for WindowTypeId {
    fn from(antiprism: AntiprismWindow) -> Self {
        WindowTypeId::Antiprism(antiprism)
    }
}

/// Makes sure that every window type is associated a unique ID (its enum
/// discriminant), which we can then use to test whether it's already in the
/// list of windows.
pub enum WindowTypeId {
    Dual(DualWindow),
    Pyramid(PyramidWindow),
    Prism(PrismWindow),
    Tegum(TegumWindow),
    Antiprism(AntiprismWindow),
}

/// Compares by discriminant.
impl std::cmp::PartialEq for WindowTypeId {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl std::cmp::Eq for WindowTypeId {}

/// The result of showing the windows every frame.
pub enum ShowResult {
    /// Nothing special happens.
    None,

    /// A window is closed.
    Close,

    /// A window is reset to its default state.
    Reset,

    /// A window runs some action.
    Ok,
}

impl WindowTypeId {
    /// Shows a given window on a given context.
    pub fn show(&mut self, ctx: &CtxRef) -> ShowResult {
        match self {
            Self::Dual(window) => window.show(ctx),
            Self::Pyramid(window) => window.show(ctx),
            Self::Prism(window) => window.show(ctx),
            Self::Tegum(window) => window.show(ctx),
            Self::Antiprism(window) => window.show(ctx),
        }
    }

    /// Updates the window after the amount of dimensions of the polytope on
    /// screen changes.
    pub fn update(&mut self, rank: Rank) {
        match self {
            Self::Dual(window) => window.update(rank),
            Self::Pyramid(window) => window.update(rank),
            Self::Prism(window) => window.update(rank),
            Self::Tegum(window) => window.update(rank),
            Self::Antiprism(window) => window.update(rank),
        }
    }

    /// Resets the window to its default state.
    pub fn reset(&mut self) {
        match self {
            Self::Dual(window) => window.reset(),
            Self::Prism(window) => window.reset(),
            Self::Pyramid(window) => window.reset(),
            Self::Tegum(window) => window.reset(),
            Self::Antiprism(window) => window.reset(),
        }
    }
}

/// The list of all windows currently shown on screen.
pub struct EguiWindows(Vec<WindowTypeId>);

impl std::default::Default for EguiWindows {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl EguiWindows {
    /// Adds a new window to the list.
    pub fn push<T: WindowType>(&mut self, value: T) {
        let value = value.into();
        if !self.0.contains(&value) {
            self.0.push(value);
        }
    }

    /// Removes a window with a given index and returns it.
    pub fn swap_remove(&mut self, idx: usize) -> WindowTypeId {
        self.0.swap_remove(idx)
    }

    /// The number of windows on the screen.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Mutably iterates over all windows.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<WindowTypeId> {
        self.0.iter_mut()
    }

    /// Shows all of the windows, and returns one if its action has to run.
    pub fn show(&mut self, ctx: &CtxRef) -> Option<WindowTypeId> {
        let mut action_window = None;
        let window_count = self.len();

        for idx in 0..window_count {
            let window = &mut self.0[idx];

            match window.show(ctx) {
                ShowResult::Close => {
                    println!("Close");
                    self.swap_remove(idx);
                    break;
                }
                ShowResult::Ok => {
                    action_window = Some(self.swap_remove(idx));
                    break;
                }
                ShowResult::Reset => window.reset(),
                ShowResult::None => {}
            }
        }

        action_window
    }

    /// Updates the window's settings whenever the rank of the polytope is
    /// updated.
    pub fn update(&mut self, rank: Rank) {
        for window in self.iter_mut() {
            window.update(rank);
        }
    }
}

/// The system that shows the windows on screen.
fn show_windows(
    egui_ctx: Res<EguiContext>,
    mut query: Query<&mut Concrete>,
    mut egui_windows: ResMut<EguiWindows>,
) {
    if let Some(result) = egui_windows.show(egui_ctx.ctx()) {
        match result {
            WindowTypeId::Dual(DualWindow { center, radius }) => {
                let sphere = Hypersphere::with_radius(center, radius);

                for mut p in query.iter_mut() {
                    if let Err(err) = p.try_dual_mut_with(&sphere) {
                        println!("{:?}", err);
                    }
                }
            }
            WindowTypeId::Pyramid(PyramidWindow { offset, height }) => {
                for mut p in query.iter_mut() {
                    *p = p.pyramid_with(offset.push(height));
                }
            }
            WindowTypeId::Prism(PrismWindow { height }) => {
                for mut p in query.iter_mut() {
                    *p = p.prism_with(height);
                }
            }
            WindowTypeId::Tegum(TegumWindow {
                offset,
                height,
                height_offset,
            }) => {
                for mut p in query.iter_mut() {
                    let half_height = height / 2.0;

                    *p = p.tegum_with(
                        offset.push(height_offset + half_height),
                        offset.push(height_offset - half_height),
                    );
                }
            }
            WindowTypeId::Antiprism(AntiprismWindow {
                dual: DualWindow { center, radius },
                height,
                central_inversion,
            }) => {
                let mut squared_radius = radius * radius;
                if central_inversion {
                    squared_radius *= -1.0;
                }

                let sphere = Hypersphere::with_squared_radius(center, squared_radius);

                for mut p in query.iter_mut() {
                    match p.try_antiprism_with(&sphere, height) {
                        Ok(q) => *p = q,
                        Err(err) => println!("{:?}", err),
                    }
                }
            }
        }
    }
}

/// Updates the windows after the polytopes change.
pub fn update_windows(
    polies: Query<(&Concrete, &Handle<Mesh>, &Children), Changed<Concrete>>,
    mut egui_windows: ResMut<EguiWindows>,
) {
    use crate::polytope::Polytope;

    if let Some((poly, _, _)) = polies.iter().next() {
        egui_windows.update(poly.rank());
    }
}

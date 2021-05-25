//! Sets up the windows that permit more advanced settings.

use super::{memory::Memory, PointWidget};
use crate::{
    geometry::{Hypersphere, Point},
    lang::{En, Language},
    polytope::{concrete::Concrete, r#abstract::rank::Rank},
    Float,
};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, CtxRef, Layout, TextStyle, Ui, Widget},
    EguiContext,
};

/// The plugin controlling these windows.
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

/// A widget consisting of a Reset button and an Ok button, right-aligned.
pub struct OkReset<'a>(&'a mut ShowResult);

impl<'a> OkReset<'a> {
    /// Initializes the buttons on screen.
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

/// Resizes a point so that it has a given number of coordinates.
fn resize(point: Point, rank: Rank) -> Point {
    point.resize_vertically(rank.try_usize().unwrap_or(0), 0.0)
}

/// Trait for a type defining a window.
pub trait WindowType: Into<WindowTypeId> {
    /// The unique name for the window, showed as a title.
    const NAME: &'static str;

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

    /// Builds the window to be shown on screen.
    fn build(&mut self, ui: &mut Ui, memory: &Res<Memory>);

    /// Shows the window on screen.
    fn show(&mut self, ctx: &CtxRef, memory: &Res<Memory>) -> ShowResult {
        let mut open = true;
        let mut result = ShowResult::None;

        egui::Window::new(Self::NAME)
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                self.build(ui, memory);
                ui.add(OkReset::new(&mut result));
            });

        if open {
            result
        } else {
            ShowResult::Close
        }
    }

    /// Updates the window's settings after the polytope's dimension is updated.
    fn update(&mut self, rank: Rank);
}

/// A window that allows the user to build a dual with a specified hypersphere.
pub struct DualWindow {
    /// The center of the sphere.
    center: Point,

    /// The radius of the sphere.
    radius: Float,
}

impl WindowType for DualWindow {
    const NAME: &'static str = "Dual";

    fn rank(&self) -> Rank {
        Rank::from(self.center.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            center: Point::zeros(rank.try_usize().unwrap_or(0)),
            radius: 1.0,
        }
    }

    fn build(&mut self, ui: &mut Ui, _: &Res<Memory>) {
        ui.add(PointWidget::new(&mut self.center, "Center:"));

        ui.horizontal(|ui| {
            ui.label("Radius:");
            ui.add(
                egui::DragValue::new(&mut self.radius)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );
        });
    }

    fn update(&mut self, rank: Rank) {
        self.center = resize(self.center.clone(), rank);
    }
}

impl From<DualWindow> for WindowTypeId {
    fn from(dual: DualWindow) -> Self {
        WindowTypeId::Dual(dual)
    }
}

/// A window that allows the user to build a pyramid with a specified apex.
pub struct PyramidWindow {
    /// How much the apex is offset from the origin.
    offset: Point,

    /// The height of the pyramid (signed distance from base to apex).
    height: Float,
}

impl WindowType for PyramidWindow {
    const NAME: &'static str = "Pyramid";

    fn rank(&self) -> Rank {
        Rank::from(self.offset.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            offset: Point::zeros(rank.try_usize().unwrap_or(0)),
            height: 1.0,
        }
    }

    fn build(&mut self, ui: &mut Ui, _: &Res<Memory>) {
        ui.add(PointWidget::new(&mut self.offset, "Offset:"));

        ui.horizontal(|ui| {
            ui.label("Height:");
            ui.add(
                egui::DragValue::new(&mut self.height)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );
        });
    }

    fn update(&mut self, rank: Rank) {
        self.offset = resize(self.offset.clone(), rank);
    }
}

impl From<PyramidWindow> for WindowTypeId {
    fn from(pyramid: PyramidWindow) -> Self {
        WindowTypeId::Pyramid(pyramid)
    }
}

/// Allows the user to build a prism with a given height.
pub struct PrismWindow {
    /// The height of the prism.
    height: Float,
}

impl WindowType for PrismWindow {
    const NAME: &'static str = "Prism";

    fn rank(&self) -> Rank {
        Default::default()
    }

    fn default_with(_: Rank) -> Self {
        Default::default()
    }

    fn build(&mut self, ui: &mut Ui, _: &Res<Memory>) {
        ui.horizontal(|ui| {
            ui.label("Height:");
            ui.add(
                egui::DragValue::new(&mut self.height)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );
        });
    }

    fn update(&mut self, _: Rank) {}
}

impl From<PrismWindow> for WindowTypeId {
    fn from(prism: PrismWindow) -> Self {
        WindowTypeId::Prism(prism)
    }
}

impl Default for PrismWindow {
    fn default() -> Self {
        Self { height: 1.0 }
    }
}

/// Allows the user to build a tegum with the specified apices and a height.
pub struct TegumWindow {
    /// The offset of the apices from the origin.
    offset: Point,

    /// The height of the tegum (the distance between both apices).
    height: Float,

    /// How much the apices are offset up or down.
    height_offset: Float,
}

impl WindowType for TegumWindow {
    const NAME: &'static str = "Tegum";

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

    fn build(&mut self, ui: &mut Ui, _: &Res<Memory>) {
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
    }

    fn update(&mut self, rank: Rank) {
        self.offset = resize(self.offset.clone(), rank);
    }
}

impl From<TegumWindow> for WindowTypeId {
    fn from(tegum: TegumWindow) -> Self {
        Self::Tegum(tegum)
    }
}

/// Allows the user to select an antiprism from a specified hypersphere and a
/// given height.
pub struct AntiprismWindow {
    dual: DualWindow,
    height: Float,
    retroprism: bool,
}

impl WindowType for AntiprismWindow {
    const NAME: &'static str = "Antiprism";

    fn rank(&self) -> Rank {
        self.dual.rank()
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            dual: DualWindow::default_with(rank),
            height: 1.0,
            retroprism: false,
        }
    }

    fn build(&mut self, ui: &mut Ui, _: &Res<Memory>) {
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
                egui::Checkbox::new(&mut self.retroprism, "Retroprism:")
                    .text_style(TextStyle::Body),
            );
        });
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

pub struct MultiprismWindow {
    selected_slots: Vec<Option<usize>>,
}

impl WindowType for MultiprismWindow {
    const NAME: &'static str = "Multiprism";

    fn rank(&self) -> Rank {
        Default::default()
    }

    fn default_with(_: Rank) -> Self {
        Default::default()
    }

    fn build(&mut self, ui: &mut Ui, memory: &Res<Memory>) {
        for (selected_slot_idx, selected_slot) in self.selected_slots.iter_mut().enumerate() {
            let selected_text = match selected_slot {
                None => "Select".to_string(),
                Some(selected_idx) => En::parse(
                    &memory[*selected_idx].as_ref().unwrap().name,
                    Default::default(),
                ),
            };

            egui::ComboBox::from_label(format!("#{}", selected_slot_idx))
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (slot_idx, poly) in memory
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, s)| s.as_ref().map(|s| (idx, s)))
                    {
                        let mut slot_inner = 69;

                        ui.selectable_value(
                            &mut slot_inner,
                            slot_idx,
                            En::parse(&poly.name, Default::default()),
                        );

                        if slot_inner != 69 {
                            *selected_slot = Some(slot_inner);
                        }
                    }
                });
        }
    }

    fn update(&mut self, _: Rank) {}
}

impl Default for MultiprismWindow {
    fn default() -> Self {
        Self {
            selected_slots: vec![None, None],
        }
    }
}

impl From<MultiprismWindow> for WindowTypeId {
    fn from(multiprism: MultiprismWindow) -> Self {
        Self::Multiprism(multiprism)
    }
}

/// Makes sure that every window type is associated a unique ID (its enum
/// discriminant), which we can then use to test whether it's already in the
/// list of windows.
///
/// `dyn WindowType` won't work here, so don't bother.
pub enum WindowTypeId {
    Dual(DualWindow),
    Pyramid(PyramidWindow),
    Prism(PrismWindow),
    Tegum(TegumWindow),
    Antiprism(AntiprismWindow),
    Multiprism(MultiprismWindow),
}

/// Compares by discriminant.
impl std::cmp::PartialEq for WindowTypeId {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl std::cmp::Eq for WindowTypeId {}

/// The result of showing a window, updated every frame.
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
    pub fn show(&mut self, ctx: &CtxRef, memory: &Res<Memory>) -> ShowResult {
        match self {
            Self::Dual(window) => window.show(ctx, memory),
            Self::Pyramid(window) => window.show(ctx, memory),
            Self::Prism(window) => window.show(ctx, memory),
            Self::Tegum(window) => window.show(ctx, memory),
            Self::Antiprism(window) => window.show(ctx, memory),
            Self::Multiprism(window) => window.show(ctx, memory),
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
            Self::Multiprism(window) => window.update(rank),
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
            Self::Multiprism(window) => window.reset(),
        }
    }
}

/// The list of all windows currently shown on screen.
#[derive(Default)]
pub struct EguiWindows(Vec<WindowTypeId>);

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
    pub fn show(&mut self, ctx: &CtxRef, memory: &Res<Memory>) -> Option<WindowTypeId> {
        let window_count = self.len();

        for idx in 0..window_count {
            let window = &mut self.0[idx];

            match window.show(ctx, memory) {
                // Closes the window.
                ShowResult::Close => {
                    self.swap_remove(idx);
                    return None;
                }

                // Runs the action from a given window.
                ShowResult::Ok => {
                    return Some(self.swap_remove(idx));
                }

                // Resets the window to its default state.
                ShowResult::Reset => {
                    window.reset();
                    return None;
                }

                // Does nothing.
                ShowResult::None => {}
            }
        }

        None
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
    memory: Res<Memory>,
    mut query: Query<&mut Concrete>,
    mut egui_windows: ResMut<EguiWindows>,
) {
    if let Some(result) = egui_windows.show(egui_ctx.ctx(), &memory) {
        match result {
            // Takes the dual of the polytope from a given hypersphere.
            WindowTypeId::Dual(DualWindow { center, radius }) => {
                let sphere = Hypersphere::with_radius(center, radius);

                for mut p in query.iter_mut() {
                    if let Err(err) = p.try_dual_mut_with(&sphere) {
                        println!("{:?}", err);
                    }
                }
            }

            // Builds a pyramid from the polytope with a given apex.
            WindowTypeId::Pyramid(PyramidWindow { offset, height }) => {
                for mut p in query.iter_mut() {
                    *p = p.pyramid_with(offset.push(height));
                }
            }

            // Builds a prism from a polytope with a given height.
            WindowTypeId::Prism(PrismWindow { height }) => {
                for mut p in query.iter_mut() {
                    *p = p.prism_with(height);
                }
            }

            // Builds a tegum from a polytope with two given apices.
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

            // Builds an antiprism from a polytope with a given height, from a
            // given hypersphere.
            WindowTypeId::Antiprism(AntiprismWindow {
                dual: DualWindow { center, radius },
                height,
                retroprism: central_inversion,
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

            WindowTypeId::Multiprism(_) => {}
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

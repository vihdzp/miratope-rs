//! Sets up the windows that permit more advanced settings.

use std::marker::PhantomData;

use super::{memory::Memory, PointWidget};
use crate::{
    geometry::{Hypersphere, Point},
    polytope::{concrete::Concrete, r#abstract::rank::Rank, Polytope},
    Float,
};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, CtxRef, Layout, TextStyle, Ui, Widget},
    EguiContext,
};

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

/// The plugin controlling these windows.
pub struct EguiWindowPlugin;

impl Plugin for EguiWindowPlugin {
    fn build(&self, app: &mut bevy::prelude::AppBuilder) {
        app.add_plugin(DualWindow::plugin())
            .add_plugin(PyramidWindow::plugin())
            .add_plugin(PrismWindow::plugin())
            .add_plugin(TegumWindow::plugin())
            .add_plugin(AntiprismWindow::plugin())
            .add_plugin(MultiprismWindow::plugin());
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

/// The base trait for a window, containing the common code. You probably don't
/// want to implement **only** this.
pub trait Window: Send + Sync + Sized + Default {
    const NAME: &'static str;

    /// Returns whether the window is open.
    fn is_open(&self) -> bool;

    /// Returns a mutable reference to the variable that determines whether the
    /// window is open.
    fn is_open_mut(&mut self) -> &mut bool;

    /// Opens a window.
    fn open(&mut self) {
        *self.is_open_mut() = true;
    }

    /// Closes a window.   
    fn close(&mut self) {
        *self.is_open_mut() = false;
    }
}

/// Implements the common methods of [`PlainWindow`] and [`UpdateWindow`]. Note
/// that this can't be put in a common trait since some of the methods here have
/// the same names but belong to different traits and have different defaults.
macro_rules! impl_show {
    () => {
        /// Shows the window on screen.
        fn show(&mut self, ctx: &CtxRef) -> ShowResult {
            let mut open = self.is_open();
            let mut result = ShowResult::None;

            egui::Window::new(Self::NAME)
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| {
                    self.build(ui);
                    ui.add(OkReset::new(&mut result));
                });

            if open {
                *self.is_open_mut() = true;
                result
            } else {
                ShowResult::Close
            }
        }

        /// The system that shows the window.
        fn show_system(
            mut self_: ResMut<Self>,
            egui_ctx: Res<EguiContext>,
            mut query: Query<&mut Concrete>,
        ) where
            Self: 'static,
        {
            match self_.show(egui_ctx.ctx()) {
                ShowResult::Ok => {
                    for mut polytope in query.iter_mut() {
                        self_.action(&mut *polytope);
                    }

                    self_.close()
                }
                ShowResult::Close => self_.close(),
                ShowResult::Reset => self_.reset(),
                ShowResult::None => {}
            }
        }
    };
}

/// A window that doesn't depend on any resources other than itself, and that
/// doesn't need to be updated when the polytope is changed.
pub trait PlainWindow: Window {
    /// Applies the action of the window to the polytope.
    fn action(&self, polytope: &mut Concrete);

    /// Builds the window to be shown on screen.
    fn build(&mut self, ui: &mut Ui);

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Default::default();
        self.open();
    }

    impl_show!();

    /// A plugin that adds a resource of type `Self` and the system to show it.
    fn plugin() -> PlainWindowPlugin<Self> {
        Default::default()
    }
}

/// A plugin that adds all of the necessary systems for a [`PlainWindow`].
#[derive(Default)]
pub struct PlainWindowPlugin<T: PlainWindow>(PhantomData<T>);

impl<T: PlainWindow + 'static> Plugin for PlainWindowPlugin<T> {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(T::default())
            .add_system(T::show_system.system().label("show_windows"));
    }
}

/// A window that doesn't depend on any resources other than itself, but needs
/// to be updated when the rank of the polytope is changed.
pub trait UpdateWindow: Window {
    /// Applies the action of the window to the polytope.
    fn action(&self, polytope: &mut Concrete);

    /// Builds the window to be shown on screen.
    fn build(&mut self, ui: &mut Ui);

    /// The default state of the window, when the polytope on the screen has a
    /// given rank.
    fn default_with(rank: Rank) -> Self;

    /// The rank of the polytope on the screen.
    fn rank(&self) -> Rank;

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Self::default_with(self.rank());
        self.open();
    }

    impl_show!();

    /// Updates the window when the rank of the polytope is updated.
    fn update(&mut self, rank: Rank);

    /// The system that updates the window when the rank of the polytope is
    /// updated.
    fn update_system(
        mut self_: ResMut<Self>,
        query: Query<(&Concrete, &Handle<Mesh>, &Children), Changed<Concrete>>,
    ) where
        Self: 'static,
    {
        if let Some((poly, _, _)) = query.iter().next() {
            self_.update(poly.rank());
        }
    }

    /// A plugin that adds a resource of type `Self` and the system to show it.
    fn plugin() -> UpdateWindowPlugin<Self> {
        Default::default()
    }
}

/// A plugin that adds all of the necessary systems for an [`UpdateWindow`].
#[derive(Default)]
pub struct UpdateWindowPlugin<T: UpdateWindow>(PhantomData<T>);

impl<T: UpdateWindow + 'static> Plugin for UpdateWindowPlugin<T> {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(T::default())
            .add_system(T::show_system.system().label("show_windows"))
            .add_system(T::update_system.system().label("show_windows"));
    }
}

/// A window that depends on the [`Memory`] but doesn't need to be updated when
/// the polytope changes.
pub trait MemoryWindow: Window {
    /// Applies the action of the window to the polytope.
    fn action(&self, polytope: &mut Concrete, memory: &Res<Memory>);

    /// Builds the window to be shown on screen.
    fn build(&mut self, ui: &mut Ui, memory: &Res<Memory>);

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Default::default();
        self.open();
    }

    /// Shows the window on screen.
    fn show(&mut self, ctx: &CtxRef, memory: &Res<Memory>) -> ShowResult {
        let mut open = self.is_open();
        let mut result = ShowResult::None;

        egui::Window::new(Self::NAME)
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                self.build(ui, &memory);
                ui.add(OkReset::new(&mut result));
            });

        if open {
            *self.is_open_mut() = true;
            result
        } else {
            ShowResult::Close
        }
    }

    /// The system that shows the window.
    fn show_system(
        mut self_: ResMut<Self>,
        egui_ctx: Res<EguiContext>,
        mut query: Query<&mut Concrete>,
        memory: Res<Memory>,
    ) where
        Self: 'static,
    {
        match self_.show(egui_ctx.ctx(), &memory) {
            ShowResult::Ok => {
                for mut polytope in query.iter_mut() {
                    self_.action(&mut *polytope, &memory);
                }

                self_.close()
            }
            ShowResult::Close => self_.close(),
            ShowResult::Reset => self_.reset(),
            ShowResult::None => {}
        }
    }

    /// A plugin that adds a resource of type `Self` and the system to show it.
    fn plugin() -> MemoryWindowPlugin<Self> {
        Default::default()
    }
}

/// A plugin that adds all of the necessary systems for an [`UpdateWindow`].
#[derive(Default)]
pub struct MemoryWindowPlugin<T: MemoryWindow>(PhantomData<T>);

impl<T: MemoryWindow + 'static> Plugin for MemoryWindowPlugin<T> {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(T::default())
            .add_system(T::show_system.system().label("show_windows"));
    }
}

/// A window that allows the user to build a dual with a specified hypersphere.
pub struct DualWindow {
    /// Whether the window is open.
    open: bool,

    /// The center of the sphere.
    center: Point,

    /// The radius of the sphere.
    radius: Float,
}

impl Default for DualWindow {
    fn default() -> Self {
        Self {
            open: false,
            center: Point::zeros(3),
            radius: 1.0,
        }
    }
}

impl Window for DualWindow {
    const NAME: &'static str = "Dual";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl UpdateWindow for DualWindow {
    fn action(&self, polytope: &mut Concrete) {
        let sphere = Hypersphere::with_radius(self.center.clone(), self.radius);

        if let Err(err) = polytope.try_dual_mut_with(&sphere) {
            println!("{:?}", err);
        }
    }

    fn build(&mut self, ui: &mut Ui) {
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

    fn rank(&self) -> Rank {
        Rank::from(self.center.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            center: Point::zeros(rank.try_usize().unwrap_or(0)),
            ..Default::default()
        }
    }

    fn update(&mut self, rank: Rank) {
        self.center = resize(self.center.clone(), rank);
    }
}

/// A window that allows the user to build a pyramid with a specified apex.
pub struct PyramidWindow {
    /// Whether the window is open.
    open: bool,

    /// How much the apex is offset from the origin.
    offset: Point,

    /// The height of the pyramid (signed distance from base to apex).
    height: Float,
}

impl Default for PyramidWindow {
    fn default() -> Self {
        Self {
            open: false,
            offset: Point::zeros(3),
            height: 1.0,
        }
    }
}

impl Window for PyramidWindow {
    const NAME: &'static str = "Pyramid";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl UpdateWindow for PyramidWindow {
    fn action(&self, polytope: &mut Concrete) {
        *polytope = polytope.pyramid_with(self.offset.push(self.height));
    }

    fn build(&mut self, ui: &mut Ui) {
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

    fn rank(&self) -> Rank {
        Rank::from(self.offset.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            offset: Point::zeros(rank.try_usize().unwrap_or(0)),
            ..Default::default()
        }
    }

    fn update(&mut self, rank: Rank) {
        self.offset = resize(self.offset.clone(), rank);
    }
}

/// Allows the user to build a prism with a given height.
pub struct PrismWindow {
    /// Whether the window is open.
    open: bool,

    /// The height of the prism.
    height: Float,
}

impl Window for PrismWindow {
    const NAME: &'static str = "Prism";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl PlainWindow for PrismWindow {
    fn action(&self, polytope: &mut Concrete) {
        *polytope = polytope.prism_with(self.height);
    }

    fn build(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Height:");
            ui.add(
                egui::DragValue::new(&mut self.height)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );
        });
    }
}

impl Default for PrismWindow {
    fn default() -> Self {
        Self {
            open: false,
            height: 1.0,
        }
    }
}

/// Allows the user to build a tegum with the specified apices and a height.
pub struct TegumWindow {
    /// Whether the window is open.
    open: bool,

    /// The offset of the apices from the origin.
    offset: Point,

    /// The height of the tegum (the distance between both apices).
    height: Float,

    /// How much the apices are offset up or down.
    height_offset: Float,
}

impl Default for TegumWindow {
    fn default() -> Self {
        Self {
            open: false,
            offset: Point::zeros(3),
            height: 1.0,
            height_offset: 0.0,
        }
    }
}

impl Window for TegumWindow {
    const NAME: &'static str = "Tegum";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl UpdateWindow for TegumWindow {
    fn action(&self, polytope: &mut Concrete) {
        let half_height = self.height / 2.0;

        *polytope = polytope.tegum_with(
            self.offset.push(self.height_offset + half_height),
            self.offset.push(self.height_offset - half_height),
        );
    }

    fn build(&mut self, ui: &mut Ui) {
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

    fn rank(&self) -> Rank {
        Rank::from(self.offset.len())
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            offset: Point::zeros(rank.try_usize().unwrap_or(0)),
            ..Default::default()
        }
    }

    fn update(&mut self, rank: Rank) {
        self.offset = resize(self.offset.clone(), rank);
    }
}

/// Allows the user to select an antiprism from a specified hypersphere and a
/// given height.
pub struct AntiprismWindow {
    dual: DualWindow,
    height: Float,
    retroprism: bool,
}

impl Default for AntiprismWindow {
    fn default() -> Self {
        Self {
            dual: Default::default(),
            height: 1.0,
            retroprism: false,
        }
    }
}

impl Window for AntiprismWindow {
    const NAME: &'static str = "Antiprism";

    fn is_open(&self) -> bool {
        self.dual.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.dual.open
    }
}

impl UpdateWindow for AntiprismWindow {
    fn action(&self, polytope: &mut Concrete) {
        let radius = self.dual.radius;
        let mut squared_radius = radius * radius;
        if self.retroprism {
            squared_radius *= -1.0;
        }

        let sphere = Hypersphere::with_squared_radius(self.dual.center.clone(), squared_radius);

        match polytope.try_antiprism_with(&sphere, self.height) {
            Ok(antiprism) => *polytope = antiprism,
            Err(err) => println!("{:?}", err),
        }
    }

    fn build(&mut self, ui: &mut Ui) {
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

    fn rank(&self) -> Rank {
        self.dual.rank()
    }

    fn default_with(rank: Rank) -> Self {
        Self {
            dual: DualWindow::default_with(rank),
            ..Default::default()
        }
    }

    fn update(&mut self, rank: Rank) {
        self.dual.update(rank);
    }
}

pub struct MultiprismWindow {
    open: bool,
    selected_slots: Vec<Option<usize>>,
}

impl Default for MultiprismWindow {
    fn default() -> Self {
        Self {
            open: false,
            selected_slots: vec![None, None],
        }
    }
}

impl Window for MultiprismWindow {
    const NAME: &'static str = "Duoprism";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl MemoryWindow for MultiprismWindow {
    fn action(&self, polytope: &mut Concrete, memory: &Res<Memory>) {
        let mut slots = Vec::with_capacity(self.selected_slots.len());

        for slot in &self.selected_slots {
            match slot {
                None => return,
                Some(idx) => slots.push(*idx),
            }
        }

        *polytope =
            Concrete::multiprism_iter(slots.into_iter().map(|idx| memory[idx].as_ref().unwrap()));
    }

    fn build(&mut self, ui: &mut Ui, memory: &Res<Memory>) {
        use crate::lang::{En, Language};

        for (selected_slot_idx, selected_slot) in self.selected_slots.iter_mut().enumerate() {
            let selected_text = match selected_slot {
                None => "Select".to_string(),
                Some(selected_idx) => En::parse_uppercase(
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
                            En::parse_uppercase(&poly.name, Default::default()),
                        );

                        if slot_inner != 69 {
                            *selected_slot = Some(slot_inner);
                        }
                    }
                });
        }
    }
}

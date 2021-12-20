//! Sets up the windows for operations on a polytope.
//!
//! All windows are loaded in parallel, before the top panel and the library are
//! shown on screen.

use std::marker::PhantomData;

use super::{
    memory::{slot_label, Memory},
    PointWidget,
};
use crate::{Concrete, Float, Hypersphere, Point};
use miratope_core::{conc::ConcretePolytope, Polytope};

use bevy::prelude::*;
use bevy_egui::{
    egui::{self, CtxRef, Layout, Ui, Widget},
    EguiContext,
};

/// The text on the loaded polytope slot.
const LOADED_LABEL: &str = "(Loaded polytope)";

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

/// The plugin controlling all of these windows.
pub struct OperationsPlugin;

impl Plugin for OperationsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(DualWindow::plugin())
            .add_plugin(PyramidWindow::plugin())
            .add_plugin(PrismWindow::plugin())
            .add_plugin(TegumWindow::plugin())
            .add_plugin(AntiprismWindow::plugin())
            .add_plugin(DuopyramidWindow::plugin())
            .add_plugin(DuoprismWindow::plugin())
            .add_plugin(DuotegumWindow::plugin())
            .add_plugin(DuocombWindow::plugin())
            .add_plugin(CompoundWindow::plugin());
    }
}

/// A widget consisting of a Reset button and an Ok button, right-aligned.
pub struct OkReset<'a> {
    result: &'a mut ShowResult,
}

impl<'a> OkReset<'a> {
    /// Initializes the buttons on screen.
    pub fn new(result: &'a mut ShowResult) -> Self {
        Self { result }
    }
}

impl<'a> Widget for OkReset<'a> {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        // We have to manually set the height of our control, for whatever reason.
        let size = egui::Vec2::new(ui.min_size().x, 30.0);

        ui.allocate_ui_with_layout(size, Layout::right_to_left(), |ui| {
            if ui.button("Ok").clicked() {
                *self.result = ShowResult::Ok;
            } else if ui.button("Reset").clicked() {
                *self.result = ShowResult::Reset;
            }
        })
        .response
    }
}

/// Resizes a point so that it has a given number of coordinates.
fn resize(point: &mut Point, dim: usize) {
    *point = point.clone().resize_vertically(dim, 0.0)
}

/// The base trait for a window, containing the common code. You probably don't
/// want to implement **only** this.
pub trait Window: Send + Sync + Default {
    /// The name on the window, shown on the upper left.
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
                self.open();
                result
            } else {
                ShowResult::Close
            }
        }

        /// The system that shows the window.
        fn show_system(
            mut self_: ResMut<'_, Self>,
            egui_ctx: Res<'_, EguiContext>,
            mut query: Query<'_, '_, &mut Concrete>,
        ) where
            Self: 'static,
        {
            match self_.show(egui_ctx.ctx()) {
                ShowResult::Ok => {
                    for mut polytope in query.iter_mut() {
                        self_.action(polytope.as_mut());
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
    fn build(&self, app: &mut App) {
        app.init_resource::<T>()
            .add_system(T::show_system.system().label("show_windows"));
    }
}

/// A window that doesn't depend on any resources other than itself, but needs
/// to be updated when the dimension of the polytope is changed.
pub trait UpdateWindow: Window {
    /// Applies the action of the window to the polytope.
    fn action(&self, polytope: &mut Concrete);

    /// Builds the window to be shown on screen.
    fn build(&mut self, ui: &mut Ui);

    /// The default state of the window, when the polytope on the screen has a
    /// given rank.
    fn default_with(dim: usize) -> Self;

    /// The rank of the polytope on the screen.
    fn dim(&self) -> usize;

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Self::default_with(self.dim());
        self.open();
    }

    impl_show!();

    /// Updates the window when the dimension of the polytope is updated.
    fn update(&mut self, dim: usize);

    /// The system that updates the window when the rank of the polytope is
    /// updated.
    fn update_system(
        mut self_: ResMut<'_, Self>,
        query: Query<'_, '_, (&Concrete, &Handle<Mesh>, &Children), Changed<Concrete>>,
    ) where
        Self: 'static,
    {
        if let Some((poly, _, _)) = query.iter().next() {
            self_.update(poly.dim_or());
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
    fn build(&self, app: &mut App) {
        app.insert_resource(T::default())
            .add_system(T::show_system.system().label("show_windows"))
            .add_system(T::update_system.system().label("show_windows"));
    }
}

/// A slot in the dropdown for duo-operations.
#[derive(Clone, Copy)]
pub enum Slot {
    /// No polytope in particular.
    None,

    /// The polytope currently on screen.
    Loaded,

    /// A polytope in a given position in memory.
    Memory(usize),
}

impl Default for Slot {
    fn default() -> Self {
        Slot::None
    }
}

impl Slot {
    pub fn to_poly<'a>(self, memory: &'a Memory, loaded: &'a Concrete) -> Option<&'a Concrete> {
        match self {
            Self::None => None,
            Self::Memory(idx) => memory[idx].as_ref(),
            Self::Loaded => Some(loaded),
        }
    }
}

/// A window for any duo-something. All of these depend on the [`Memory`] but
/// don't need to be updated when the polytope changes.
pub trait DuoWindow: Window {
    /// The duo-operation to apply.
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete;

    /// The slots in memory.
    fn slots(&self) -> [Slot; 2];

    /// A mutable reference to the slots in memory.
    fn slots_mut(&mut self) -> &mut [Slot; 2];

    /// Returns the references to the polytopes currently selected.
    fn polytopes<'a>(
        &'a self,
        loaded: &'a Concrete,
        memory: &'a Memory,
    ) -> [Option<&'a Concrete>; 2] {
        let [i, j] = self.slots();
        [i.to_poly(memory, loaded), j.to_poly(memory, loaded)]
    }

    /// Returns the dimensions of the polytopes currently selected, or 0 in case
    /// of the nullitope.
    fn dim_or(&self, polytope: &Concrete, memory: &Memory) -> [usize; 2] {
        self.polytopes(polytope, memory)
            .map(|p| p.map(|poly| poly.dim()).flatten().unwrap_or_default())
    }

    /// Applies the action of the window to the polytope.
    fn action(&self, polytope: &mut Concrete, memory: &Memory) {
        if let [Some(p), Some(q)] = self.polytopes(polytope, memory) {
            *polytope = self.operation(p, q);
        }
    }

    /// Builds the window to be shown on screen.
    fn build(&mut self, _: &mut Ui, _: &Concrete, _: &Memory) {}

    fn build_dropdowns(&mut self, ui: &mut Ui, memory: &Memory) {
        const SELECT: &str = "Select";

        // Iterates over both slots.
        for (slot_idx, selected) in self.slots_mut().iter_mut().enumerate() {
            // The text for the selected option.
            let selected_text = match selected {
                // Nothing has been selected.
                Slot::None => SELECT.to_string(),

                // The loaded polytope is selected.
                Slot::Loaded => LOADED_LABEL.to_string(),

                // Something is selected from the memory.
                Slot::Memory(selected_idx) => match memory[*selected_idx].as_ref() {
                    // Whatever was previously selected got deleted off the memory.
                    None => {
                        *selected = Slot::None;
                        SELECT.to_string()
                    }

                    // Shows the name of the selected polytope.
                    Some(_) => slot_label(*selected_idx),
                },
            };

            // The drop-down for selecting polytopes, either from memory or the
            // currently loaded one.
            egui::ComboBox::from_label(format!("#{}", slot_idx + 1))
                .selected_text(selected_text)
                .width(200.0)
                .show_ui(ui, |ui| {
                    // The currently loaded polytope.
                    let mut loaded_selected = false;

                    ui.selectable_value(&mut loaded_selected, true, LOADED_LABEL);

                    // If the value was changed, update it.
                    if loaded_selected {
                        *selected = Slot::Loaded;
                    }

                    // The polytopes in memory.
                    for (slot_idx, _) in memory
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, s)| s.as_ref().map(|s| (idx, s)))
                    {
                        // This value couldn't be selected by the user.
                        let mut slot_inner = None;

                        ui.selectable_value(&mut slot_inner, Some(slot_idx), slot_label(slot_idx));

                        // If the value was changed, update it.
                        if let Some(idx) = slot_inner {
                            *selected = Slot::Memory(idx);
                        }
                    }
                });
        }
    }

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Default::default();
        self.open();
    }

    /// Shows the window on screen.
    fn show(&mut self, ctx: &CtxRef, polytope: &Concrete, memory: &Memory) -> ShowResult {
        let mut open = self.is_open();
        let mut result = ShowResult::None;

        egui::Window::new(Self::NAME)
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                self.build_dropdowns(ui, memory);
                self.build(ui, polytope, memory);
                ui.add(OkReset::new(&mut result));
            });

        if open {
            self.open();
            result
        } else {
            ShowResult::Close
        }
    }

    /// The system that shows the window.
    fn show_system(
        mut self_: ResMut<'_, Self>,
        egui_ctx: Res<'_, EguiContext>,
        mut query: Query<'_, '_, &mut Concrete>,
        memory: Res<'_, Memory>,
    ) where
        Self: 'static,
    {
        for mut polytope in query.iter_mut() {
            match self_.show(egui_ctx.ctx(), &polytope, &memory) {
                ShowResult::Ok => {
                    self_.action(polytope.as_mut(), &memory);
                    self_.close()
                }
                ShowResult::Close => self_.close(),
                ShowResult::Reset => self_.reset(),
                ShowResult::None => {}
            }
        }
    }

    /// A plugin that adds a resource of type `Self` and the system to show it.
    fn plugin() -> DuoWindowPlugin<Self> {
        Default::default()
    }
}

/// A plugin that adds all of the necessary systems for a [`DuoWindow`].
#[derive(Default)]
pub struct DuoWindowPlugin<T: DuoWindow>(PhantomData<T>);

impl<T: DuoWindow + 'static> Plugin for DuoWindowPlugin<T> {
    fn build(&self, app: &mut App) {
        app.init_resource::<T>()
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
            center: Point::zeros(0),
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
            eprintln!("Dual failed: {}", err);
        }
    }

    fn build(&mut self, ui: &mut Ui) {
        ui.add(PointWidget::new(&mut self.center, "Center"));

        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.radius)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );

            ui.label("Radius");
        });
    }

    fn dim(&self) -> usize {
        self.center.len()
    }

    fn default_with(dim: usize) -> Self {
        Self {
            center: Point::zeros(dim),
            radius: 1.0,
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        resize(&mut self.center, dim);
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
            offset: Point::zeros(0),
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
        ui.add(PointWidget::new(&mut self.offset, "Offset"));

        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.height)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );

            ui.label("Height");
        });
    }

    fn dim(&self) -> usize {
        self.offset.len()
    }

    fn default_with(dim: usize) -> Self {
        Self {
            offset: Point::zeros(dim),
            height: 1.0,
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        resize(&mut self.offset, dim);
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
            offset: Point::zeros(0),
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
        ui.add(PointWidget::new(&mut self.offset, "Offset"));

        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.height)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );
            ui.label("Height");
        });

        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.height_offset).speed(0.01));
            ui.label("Height offset");
        });
    }

    fn dim(&self) -> usize {
        self.offset.len()
    }

    fn default_with(dim: usize) -> Self {
        Self {
            offset: Point::zeros(dim),
            height: 1.0,
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        resize(&mut self.offset, dim);
    }
}

/// Allows the user to select an antiprism from a specified hypersphere and a
/// given height.
pub struct AntiprismWindow {
    /// The info about the hypersphere we use to get from one base to another.
    dual: DualWindow,

    /// The height of the antiprism.
    height: Float,

    /// Whether the antiprism is a retroprism.
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
            Err(err) => eprintln!("Antiprism failed: {}", err),
        }
    }

    fn build(&mut self, ui: &mut Ui) {
        ui.add(PointWidget::new(&mut self.dual.center, "Center"));

        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.dual.radius)
                    .speed(0.01)
                    .clamp_range(0.0..=Float::MAX),
            );
            ui.label("Radius");
        });

        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.height).speed(0.01));
            ui.label("Height");
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::Checkbox::new(&mut self.retroprism, "Retroprism"), //.text_style(TextStyle::Body),
            );
        });
    }

    fn dim(&self) -> usize {
        self.dual.dim()
    }

    fn default_with(dim: usize) -> Self {
        Self {
            dual: DualWindow::default_with(dim),
            height: 1.0,
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        self.dual.update(dim);
    }
}

/// A window that allows a user to build a duopyramid, either using the
/// polytopes in memory or the currently loaded one.
pub struct DuopyramidWindow {
    /// Whether the window is currently open.
    open: bool,

    /// The slots corresponding to the selected polytopes.
    slots: [Slot; 2],

    /// The height of the duopyramid.
    height: Float,

    /// The offset of each base.
    offsets: [Point; 2],
}

impl Default for DuopyramidWindow {
    fn default() -> Self {
        Self {
            open: false,
            slots: Default::default(),
            height: 1.0,
            offsets: [Point::zeros(0), Point::zeros(0)],
        }
    }
}

impl Window for DuopyramidWindow {
    const NAME: &'static str = "Duopyramid";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl DuoWindow for DuopyramidWindow {
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete {
        let [p_offset, q_offset] = &self.offsets;
        Concrete::duopyramid_with(p, q, p_offset, q_offset, self.height)
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }

    fn build(&mut self, ui: &mut Ui, polytope: &Concrete, memory: &Memory) {
        let [p_dim, q_dim] = self.dim_or(polytope, memory);

        resize(&mut self.offsets[0], p_dim);
        resize(&mut self.offsets[1], q_dim);

        ui.add(PointWidget::new(&mut self.offsets[0], "Offset #1"));
        ui.add(PointWidget::new(&mut self.offsets[1], "Offset #2"));

        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.height).clamp_range(0.0..=Float::MAX));
            ui.label("Height");
        });
    }
}

/// A window that allows a user to build a duoprism, either using the polytopes
/// in memory or the currently loaded one.
#[derive(Default)]
pub struct DuoprismWindow {
    /// Whether the window is open.
    open: bool,

    /// The slots that are currently selected.
    slots: [Slot; 2],
}

impl Window for DuoprismWindow {
    const NAME: &'static str = "Duoprism";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl DuoWindow for DuoprismWindow {
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete {
        p.duoprism(q)
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }
}

/// A window that allows a user to build a duotegum, either using the polytopes
/// in memory or the currently loaded one.
pub struct DuotegumWindow {
    /// Whether the window is currently open.
    open: bool,

    /// The slots corresponding to the selected polytopes.
    slots: [Slot; 2],

    /// The offset of each base.
    offsets: [Point; 2],
}

impl Default for DuotegumWindow {
    fn default() -> Self {
        Self {
            open: false,
            slots: Default::default(),
            offsets: [Point::zeros(0), Point::zeros(0)],
        }
    }
}

impl Window for DuotegumWindow {
    const NAME: &'static str = "Duotegum";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl DuoWindow for DuotegumWindow {
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete {
        let [p_offset, q_offset] = &self.offsets;
        Concrete::duotegum_with(p, q, p_offset, q_offset)
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }

    fn build(&mut self, ui: &mut Ui, polytope: &Concrete, memory: &Memory) {
        let [p_dim, q_dim] = self.dim_or(polytope, memory);

        resize(&mut self.offsets[0], p_dim);
        resize(&mut self.offsets[1], q_dim);

        ui.add(PointWidget::new(&mut self.offsets[0], "Offset #1"));
        ui.add(PointWidget::new(&mut self.offsets[1], "Offset #2"));
    }
}

/// A window that allows a user to build a duocomb, either using the polytopes
/// in memory or the currently loaded one.
#[derive(Default)]
pub struct DuocombWindow {
    /// Whether the window is open.
    open: bool,

    /// The slots that are currently selected.
    slots: [Slot; 2],
}

impl Window for DuocombWindow {
    const NAME: &'static str = "Duocomb";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl DuoWindow for DuocombWindow {
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete {
        p.duocomb(q)
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }
}

/// A window that allows a user to build a compound, either using the polytopes
/// in memory or the currently loaded one.
#[derive(Default)]
pub struct CompoundWindow {
    /// Whether the window is open.
    open: bool,

    /// The slots that are currently selected.
    slots: [Slot; 2],
}

impl Window for CompoundWindow {
    const NAME: &'static str = "Compound";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl DuoWindow for CompoundWindow {
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete {
        let mut p2 = p.clone();
        p2.comp_append(q.clone());
        p2
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }
}

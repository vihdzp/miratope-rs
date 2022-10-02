//! Sets up the windows for operations on a polytope.
//!
//! All windows are l&mut &mut oaded in parallel, before the top panel and the library are
//! shown on screen.

use std::marker::PhantomData;

use super::{
    memory::{slot_label, Memory},
    PointWidget,
};
use crate::{Concrete, Float, Hypersphere, Point, ui::main_window::PolyName};

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
pub struct WindowPlugin;

impl Plugin for WindowPlugin {
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
            .add_plugin(StarWindow::plugin())
            .add_plugin(CompoundWindow::plugin())
            .add_plugin(TruncateWindow::plugin())
            .add_plugin(ScaleWindow::plugin())
            .add_plugin(FacetingSettings::plugin())
			.add_plugin(RotateWindow::plugin())
			.add_plugin(PlaneWindow::plugin());
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
            mut poly_name: ResMut<'_, PolyName>,
        ) where
            Self: 'static,
        {
            match self_.show(egui_ctx.ctx()) {
                ShowResult::Ok => {
                    for mut polytope in query.iter_mut() {
                        self_.action(polytope.as_mut());
                    }
                    self_.name_action(&mut poly_name.0);
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

    /// Applies an action to the polytope name.
    fn name_action(&self, name: &mut String);

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
    
    /// Applies an action to the polytope name.
    fn name_action(&self, name: &mut String);

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
#[derive(Clone, Copy, PartialEq)]
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
            Self::Memory(idx) => Some(&memory[idx].as_ref().unwrap().0),
            Self::Loaded => Some(loaded),
        }
    }
}

/// A window that depends on [`Memory`], and that
/// doesn't need to be updated when the polytope is changed.
pub trait MemoryWindow: Window {
    /// Applies the action of the window to the polytope.
    fn action(&self, polytope: &mut Concrete);

    /// Builds the window to be shown on screen.
    fn build(&mut self, ui: &mut Ui, memory: &Memory);

    /// Resets a window to its default state.
    fn reset(&mut self) {
        *self = Default::default();
        self.open();
    }

    /// Shows the window on screen.
    fn show(&mut self, ctx: &CtxRef, memory: &Memory) -> ShowResult {
        let mut open = self.is_open();
        let mut result = ShowResult::None;

        egui::Window::new(Self::NAME)
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                self.build(ui, memory);
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
        match self_.show(egui_ctx.ctx(), &memory) {
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

    /// A plugin that adds a resource of type `Self` and the system to show it.
    fn plugin() -> MemoryWindowPlugin<Self> {
        Default::default()
    }
}

/// A plugin that adds all of the necessary systems for a [`MemoryWindow`].
#[derive(Default)]
pub struct MemoryWindowPlugin<T: MemoryWindow>(PhantomData<T>);

impl<T: MemoryWindow + 'static> Plugin for MemoryWindowPlugin<T> {
    fn build(&self, app: &mut App) {
        app.init_resource::<T>()
            .add_system(T::show_system.system().label("show_windows"));
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

    /// Applies an action to the polytope name.
    fn name_action(&self, name: &mut String, memory: &Memory);

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
        mut poly_name: ResMut<'_, PolyName>,
    ) where
        Self: 'static,
    {
        for mut polytope in query.iter_mut() {
            match self_.show(egui_ctx.ctx(), &polytope, &memory) {
                ShowResult::Ok => {
                    self_.action(polytope.as_mut(), &memory);
                    self_.name_action(&mut poly_name.0, &memory);
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

    fn name_action(&self, name: &mut String) {
        *name = format!("Dual of {}", name);
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

    fn name_action(&self, name: &mut String) {
        *name = format!("Pyramid of {}", name);
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

    fn name_action(&self, name: &mut String) {
        *name = format!("Prism of {}", name);
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

    fn name_action(&self, name: &mut String) {
        *name = format!("Tegum of {}", name);
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

    fn name_action(&self, name: &mut String) {
        *name = format!("Dual of {}", name);
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

    fn name_action(&self, name: &mut String, memory: &Memory) {
        let name_a = match self.slots[0] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };
        let name_b = match self.slots[1] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };

        *name = format!("Duopyramid of ({}, {})", name_a, name_b);
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

        if ui.add(
            egui::Button::new("Try to make orbiform")
                .enabled(!matches!(self.slots[0], Slot::None) && !matches!(self.slots[1], Slot::None))
            ).clicked() {
                if let Some(circum0) = match self.slots[0] {
                    Slot::Loaded => polytope,
                    Slot::Memory(i) => &memory[i].as_ref().unwrap().0,
                    Slot::None => unreachable!(),
                }.circumsphere() {
                    if let Some(circum1) = match self.slots[1] {
                        Slot::Loaded => polytope,
                        Slot::Memory(i) => &memory[i].as_ref().unwrap().0,
                        Slot::None => unreachable!(),
                    }.circumsphere() {

                        let sq_height = 1. - circum0.squared_radius - circum1.squared_radius;
                        if sq_height >= 0. {
                            self.height = sq_height.sqrt();
                            self.offsets[0] = -circum0.center;
                            self.offsets[1] = -circum1.center;
                        } else {
                            println!("Orbiform failed: height is imaginary.");
                        }

                    } else {
                        println!("Orbiform failed: {} has no circumsphere.", match self.slots[1] {
                            Slot::Loaded => "Loaded polytope".to_string(),
                            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                                Some(label) => label.to_string(),
                                None => format!("polytope {}", i),
                            },
                            Slot::None => unreachable!(),
                        });
                    }
                } else {
                    println!("Orbiform failed: {} has no circumsphere.", match self.slots[0] {
                        Slot::Loaded => "Loaded polytope".to_string(),
                        Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                            Some(label) => label.to_string(),
                            None => format!("polytope {}", i),
                        },
                        Slot::None => unreachable!(),
                    });
                }
        }
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

    fn name_action(&self, name: &mut String, memory: &Memory) {
        let name_a = match self.slots[0] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };
        let name_b = match self.slots[1] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };

        *name = format!("Duoprism of ({}, {})", name_a, name_b);
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

    fn name_action(&self, name: &mut String, memory: &Memory) {
        let name_a = match self.slots[0] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };
        let name_b = match self.slots[1] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };

        *name = format!("Duotegum of ({}, {})", name_a, name_b);
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

    fn name_action(&self, name: &mut String, memory: &Memory) {
        let name_a = match self.slots[0] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };
        let name_b = match self.slots[1] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };

        *name = format!("Comb of ({}, {})", name_a, name_b);
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }
}

/// A window that allows a user to build a star product, either using the polytopes
/// in memory or the currently loaded one.
#[derive(Default)]
pub struct StarWindow {
    /// Whether the window is open.
    open: bool,

    /// The slots that are currently selected.
    slots: [Slot; 2],
}

impl Window for StarWindow {
    const NAME: &'static str = "Star product";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl DuoWindow for StarWindow {
    fn operation(&self, p: &Concrete, q: &Concrete) -> Concrete {
        p.star_product(q)
    }

    fn name_action(&self, name: &mut String, memory: &Memory) {
        let name_a = match self.slots[0] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };
        let name_b = match self.slots[1] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };

        *name = format!("Star of ({}, {})", name_a, name_b);
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

    fn name_action(&self, name: &mut String, memory: &Memory) {
        let name_a = match self.slots[0] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };
        let name_b = match self.slots[1] {
            Slot::Loaded => name.clone(),
            Slot::Memory(i) => match &memory[i].as_ref().unwrap().1 {
                Some(label) => label.to_string(),
                None => format!("polytope {}", i),
            },
            Slot::None => "".to_string(),
        };

        *name = format!("Compound of ({}, {})", name_a, name_b);
    }

    fn slots(&self) -> [Slot; 2] {
        self.slots
    }

    fn slots_mut(&mut self) -> &mut [Slot; 2] {
        &mut self.slots
    }
}

/// A window to configure a truncation of the polytope.
#[derive(Default)]
pub struct TruncateWindow {
    /// Whether the window is open.
    open: bool,

    /// The rank of the polytope.
    rank: usize,

    /// Which nodes are ringed.
    truncate_type: Vec<bool>,

    /// The weights applied to the coordinates. Intuitively, the truncation depths.
    depth: Vec<f64>,
}

impl Window for TruncateWindow {
    const NAME: &'static str = "Truncate";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl UpdateWindow for TruncateWindow {
    fn action(&self, polytope: &mut Concrete) {
        let mut rings = Vec::new();
        for (rank, ringed) in self.truncate_type.iter().enumerate() {
            if *ringed {
                rings.push(rank);
            }
        }
        polytope.element_sort();
        *polytope = polytope.truncate_with(rings, self.depth.clone());
    }

    fn name_action(&self, name: &mut String) {
        *name = format!("Truncated {}", name);
    }

    fn build(&mut self, ui: &mut Ui) {
        for r in 0..self.rank {
            ui.horizontal(|ui| {
                ui.add(egui::Checkbox::new(&mut self.truncate_type[r], ""));
                ui.add(egui::DragValue::new(&mut self.depth[r]).speed(0.01));
            });
        }
    }

    fn dim(&self) -> usize {
        self.rank
    }

    fn default_with(dim: usize) -> Self {
        Self {
            rank: dim,
            truncate_type: vec![false; dim],
            depth: vec![1.0; dim],
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        self.rank = dim;
        self.truncate_type = vec![false; dim];
        self.depth = vec![1.0; dim];
    }
}

/// A window that scales a polytope.
#[derive(Default)]
pub struct ScaleWindow {
    /// Whether the window is open.
    open: bool,

    /// The scale factor.
    scale: f64,
}

impl Window for ScaleWindow {
    const NAME: &'static str = "Scale";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl PlainWindow for ScaleWindow {
    fn action(&self, polytope: &mut Concrete) {
        polytope.scale(self.scale);
    }

    fn name_action(&self, _name: &mut String) {}

    fn build(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.scale)
                    .speed(0.01)
            );
        });
    }
}

/// Where to get the symmetry group for faceting
#[derive(PartialEq)]
pub enum GroupEnum2 {
    /// Group of matrices
    FromSlot(Slot),
    /// True: take chiral group
    /// False: take full group
    Chiral(bool),
}

/// A window that lets the user set settings for faceting.
pub struct FacetingSettings {
    /// Whether the window is open.
    open: bool,

    /// There's some niche settings that are hidden by default to reduce bloat.
    show_advanced_settings: bool,

    /// The slot for the dropdown menu.
    slot: Slot,

    /// The maximum number of facet types considered. 1 for isotopic, 0 for no limit.
    pub max_facet_types: usize,

    /// The maximum number of facets generated in each hyperplane, to prevent combinatorial explosion. 0 for no limit.
    pub max_per_hyperplane: usize,

    /// Where to get the symmetry group from.
    pub group: GroupEnum2,

    // These can't just be `Option`s because you need checkboxes and stuff.
    /// Whether to use a minimum edge length.
    pub do_min_edge_length: bool,

    /// The minimum edge length.
    pub min_edge_length: f64,

    /// Whether to use a maximum edge length.
    pub do_max_edge_length: bool,

    /// The maximum edge length.
    pub max_edge_length: f64,

    /// Whether to use a minimum inradius.
    pub do_min_inradius: bool,

    /// The minimum inradius.
    pub min_inradius: f64,

    /// Whether to use a maximum inradius.
    pub do_max_inradius: bool,

    /// The maximum inradius.
    pub max_inradius: f64,

    /// Whether to exclude planes passing through the origin.
    pub exclude_hemis: bool,

    /// Whether to only consider hyperplanes perpendicular to a vertex.
    pub only_below_vertex: bool,

    /// Whether to include trivial compounds (compounds of other full-symmetric facetings).
    pub compounds: bool,

    /// Whether to check if the faceting is compound or fissary and mark it.
    pub mark_fissary: bool,

    /// Only use uniform or semiuniform elements.
    pub uniform: bool,

    /// Whether to include the facet numbers in the name.
    pub label_facets: bool,

    /// Whether to save the facetings in memory.
    pub save: bool,

    /// Whether to save the facets in memory.
    pub save_facets: bool,

    /// Whether to save to file.
    pub save_to_file: bool,

    /// The path to save to, if saving to file.
    pub file_path: String,
}

impl Default for FacetingSettings {
    fn default() -> Self {
        Self {
            open: false,
            show_advanced_settings: false,
            slot: Slot::default(),
            max_facet_types: 0,
            max_per_hyperplane: 0,
            group: GroupEnum2::Chiral(false),
            do_min_edge_length: true,
            min_edge_length: 1.,
            do_max_edge_length: true,
            max_edge_length: 1.,
            do_min_inradius: false,
            min_inradius: 0.,
            do_max_inradius: false,
            max_inradius: 0.,
            exclude_hemis: false,
            only_below_vertex: false,
            compounds: false,
            mark_fissary: true,
            uniform: false,
            label_facets: true,
            save: true,
            save_facets: false,
            save_to_file: false,
            file_path: "".to_string(),
        }
    }
}

impl Window for FacetingSettings {
    const NAME: &'static str = "Faceting settings";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl MemoryWindow for FacetingSettings {
    fn action(&self, _polytope: &mut Concrete) {
    }

    fn build(&mut self, ui: &mut Ui, memory: &Memory) {
        ui.horizontal(|ui| {
            ui.label("Max facet types");
            ui.add(
                egui::DragValue::new(&mut self.max_facet_types)
                    .speed(0.02)
                    .clamp_range(0..=usize::MAX)
            );
        });
        if self.show_advanced_settings {
            ui.horizontal(|ui| {
                ui.label("Max facetings per hyperplane");
                ui.add(
                    egui::DragValue::new(&mut self.max_per_hyperplane)
                        .speed(200)
                        .clamp_range(0..=usize::MAX)
                );
            });
        }
        ui.separator();

        ui.label("Group:");

        ui.radio_value(&mut self.group, GroupEnum2::Chiral(false), "Full group");
        ui.radio_value(&mut self.group, GroupEnum2::Chiral(true), "Chiral subgroup");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.group, GroupEnum2::FromSlot(self.slot), "From other polytope:");
                
            const SELECT: &str = "Select";

            // The text for the selected option.
            let selected_text = match self.slot {
                // Nothing has been selected.
                Slot::None => SELECT.to_string(),

                // The loaded polytope is selected.
                Slot::Loaded => LOADED_LABEL.to_string(),

                // Something is selected from the memory.
                Slot::Memory(selected_idx) => match memory[selected_idx].as_ref() {
                    // Whatever was previously selected got deleted off the memory.
                    None => {
                        self.slot = Slot::None;
                        SELECT.to_string()
                    }

                    // Shows the name of the selected polytope.
                    Some(_) => slot_label(selected_idx),
                },
            };

            // The drop-down for selecting polytopes, either from memory or the
            // currently loaded one.
            egui::ComboBox::from_label("")
                .selected_text(selected_text)
                .width(200.0)
                .show_ui(ui, |ui| {
                    // The currently loaded polytope.
                    let mut loaded_selected = false;

                    ui.selectable_value(&mut loaded_selected, true, LOADED_LABEL);

                    // If the value was changed, update it.
                    if loaded_selected {
                        self.slot = Slot::Loaded;
                        self.group = GroupEnum2::FromSlot(self.slot);
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
                            self.slot = Slot::Memory(idx);
                            self.group = GroupEnum2::FromSlot(self.slot);
                        }
                    }
            });
        });

        ui.separator();

        ui.horizontal(|ui| {
            ui.add(
                egui::Checkbox::new(&mut self.do_min_edge_length, "")
            );
            ui.add(
                egui::DragValue::new(&mut self.min_edge_length).clamp_range(0.0..=Float::MAX).speed(0.01)
            );
            ui.label("Min edge length");
        });

        ui.horizontal(|ui| {
            ui.add(
                egui::Checkbox::new(&mut self.do_max_edge_length, "")
            );
            ui.add(
                egui::DragValue::new(&mut self.max_edge_length).clamp_range(0.0..=Float::MAX).speed(0.01)
            );
            ui.label("Max edge length");
        });

        if self.show_advanced_settings {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Checkbox::new(&mut self.do_min_inradius, "")
                );
                ui.add(
                    egui::DragValue::new(&mut self.min_inradius).clamp_range(0.0..=Float::MAX).speed(0.001)
                );
                ui.label("Min inradius");
            });
    
            ui.horizontal(|ui| {
                ui.add(
                    egui::Checkbox::new(&mut self.do_max_inradius, "")
                );
                ui.add(
                    egui::DragValue::new(&mut self.max_inradius).clamp_range(0.0..=Float::MAX).speed(0.001)
                );
                ui.label("Max inradius");
            });
    
            ui.add(
                egui::Checkbox::new(&mut self.exclude_hemis, "Exclude hemis")
            );
    
            ui.add(
                egui::Checkbox::new(&mut self.only_below_vertex, "Only hyperplanes perpendicular to a vertex")
            );
        }

        ui.separator();

        ui.add(
            egui::Checkbox::new(&mut self.uniform, "Only uniform/semiuniform facets")
        );

        if self.show_advanced_settings {
            ui.separator();
        
            ui.add(
                egui::Checkbox::new(&mut self.compounds, "Include trivial compounds")
            );
    
            ui.add(
                egui::Checkbox::new(&mut self.mark_fissary, "Mark compounds/fissaries")
            );
    
            ui.add(
                egui::Checkbox::new(&mut self.label_facets, "Label facets")
            );
        }

        ui.separator();

        ui.add(
            egui::Checkbox::new(&mut self.save, "Save facetings")
        );

        ui.add(
            egui::Checkbox::new(&mut self.save_facets, "Save facets")
        );

        ui.radio_value(&mut self.save_to_file, false, "Save to memory");

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.save_to_file, true, "Save to file");
            ui.label("Path:");
            ui.add(
                egui::TextEdit::singleline(&mut self.file_path).enabled(self.save_to_file)
            );
        });

        ui.separator();

        if ui.button(if self.show_advanced_settings {"Hide advanced settings"} else {"Show advanced settings"}).clicked() {
            self.show_advanced_settings = !self.show_advanced_settings;
        }
    }
}


/// Rotation window for Transform tab
#[derive(Default)]
pub struct RotateWindow {
    /// Whether the window is open.
    open: bool,

    /// The rank of the polytope.
    rank: usize,

    /// List of rotations (in radians). Rotates around xy plane, then yz plane, then zw plane, etc.
    rots: Vec<f64>,
	
	/// Determines if radians or degrees are used.
	degcheck: bool,
}

impl Window for RotateWindow {
    const NAME: &'static str = "Rotate";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

impl UpdateWindow for RotateWindow {
    fn action(&self, polytope: &mut Concrete) {
		if self.rank > 1 {
			polytope.element_sort();
			
			if self.degcheck { //Degrees
				for ind in 0..self.rank-1 {
					for v in polytope.vertices_mut() {
						let theta = self.rots[ind]*0.017453292519943295;
						
						let x = v[ind]*theta.cos() - v[ind+1]*theta.sin();
						let y = v[ind]*theta.sin() + v[ind+1]*theta.cos();
						v[ind] = x;
						v[(ind+1)%self.rank] = y;
					}
				}
			}
			else { //Radians
				for ind in 0..self.rank-1 {
					for v in polytope.vertices_mut() {
						let theta = self.rots[ind];
						
						let x = v[ind]*theta.cos() - v[ind+1]*theta.sin();
						let y = v[ind]*theta.sin() + v[ind+1]*theta.cos();
						v[ind] = x;
						v[(ind+1)%self.rank] = y;
					}
				}	
			}

			
			println!("Object rotated!");
		}
        else {
			println!("Objects with rank less than 2 cannot be rotated.")
		}
    }

    fn name_action(&self, name: &mut String) {
        *name = format!("Rotated {}", name);
    }

    fn build(&mut self, ui: &mut Ui) {
        ui.add(egui::Checkbox::new(&mut self.degcheck, "Use degrees instead of radians"));
		for r in 0..self.rank-1 {
            ui.horizontal(|ui| {
				if self.degcheck {
					ui.add(egui::DragValue::new(&mut self.rots[r]).speed(1.0).clamp_range(0.0..=360.0));
				}
                else{
					ui.add(egui::DragValue::new(&mut self.rots[r]).speed(0.01).clamp_range(0.0..=6.283185307179586));
				}
            });
        }
    }
	
    fn dim(&self) -> usize {
        self.rank
    }

    fn default_with(dim: usize) -> Self {
        Self {
            rank: dim,
            rots: vec![0.0001; dim], // if this is set to 0 the whole window becomes dark for some reason
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        self.rank = dim;
        self.rots = vec![0.0; dim];
    }
}

/// Plane rotation window (Rotate with plane... window)

pub struct PlaneWindow {
    /// Whether the window is open.
    open: bool,

    /// The rank of the polytope.
    rank: usize,

    /// Rotation amount (radians).
    rot: f64,
	
	/// Coordinates of points.
	p1: Point,
	p2: Point,
	
	/// Determines if radians or degrees are used.
	degcheck: bool,
	
	//Determines if a custom origin point should be used.
	origincheck: bool,
	po: Point,

}

impl Default for PlaneWindow {
    fn default() -> Self {
        Self {
            open: false,
			rank: Default::default(),
			
			rot: 0.0,
			
            p1: Point::zeros(0),
			p2: Point::zeros(0),
			
            degcheck: false,
			
			origincheck: false,
			po: Point::zeros(0),
        }
    }
}
 
impl Window for PlaneWindow {
    const NAME: &'static str = "Rotate with plane";

    fn is_open(&self) -> bool {
        self.open
    }

    fn is_open_mut(&mut self) -> &mut bool {
        &mut self.open
    }
}

fn dot(u: &Vec<f64>, v: &Vec<f64>) -> f64 {
	let mut sum = 0.0;
	for i in 0..u.len() {
		sum += u[i]*v[i];
	}
	return sum
}

impl UpdateWindow for PlaneWindow {
    fn action(&self, polytope: &mut Concrete) {
		if self.p1 == Point::zeros(self.rank) || self.p2 == Point::zeros(self.rank) {
			println!("Points within plane cannot be located at the origin.");
		}
		else if self.rot == 0.0 {
			println!("Rotated, but the rotation amount was set to 0 so there was no change.");
		}
		else {			
			//Step 0: Make plane of orthonormal basis based on input
			//Make points p1 and p2 into unit Vec<f64> objects.
			//Also subtract po from p1 and p2
			let ss1: f64 = self.p1.iter().map(|&x| x*x).sum();
			let ss2: f64 = self.p2.iter().map(|&x| x*x).sum();
			
			let mut v1: Vec<f64> = Vec::new();
			let mut v2: Vec<f64> = Vec::new();
			
			for i in 0..self.rank {
				v1.push( (self.p1[i]-self.po[i])/ss1.sqrt() );
				v2.push( (self.p2[i]-self.po[i])/ss2.sqrt() );
			}
			
			//Implement Gram-Schmidt process to make vectors orthonormal
			let prod = dot(&v1,&v2)/dot(&v2,&v2);
			let mut u2: Vec<f64> = Vec::new();
			for i in 0..self.rank {
				u2.push(v2[i] - v1[i] * prod);
			}
			let ss3: f64 = u2.iter().map(|&x| x*x).sum();
			
			for i in 0..self.rank {
				v2[i] = u2[i]/ss3.sqrt();
			}
			
			let theta: f64;
			if self.degcheck { //theta is the rotation amount in radians, which may or may not need conversion
				theta = self.rot * 0.017453292519943295;
			}
			else {
				theta = self.rot;
			}
			
			for v in polytope.vertices_mut() {
				
				//Step 1: Find perpendicular intersection of point and plane, in orthonormal basis
				//Equivalent to solving for the vector Q where (v-Q)·v1 = (v-Q)·v2 = 0, and Q is in the v1v2 plane.
				//From this we find Q in the v1v2 basis. It turns out to equal [v·v1/v1·v1,v·v2/v2·v2].
				let mut vvec = Vec::new();
				for i in 0..self.rank {
					vvec.push( v[i] );
				}
				let vf = vec![ dot(&vvec,&v1)/dot(&v1,&v1) , dot(&vvec,&v2)/dot(&v2,&v2) ];
				
				//Step 2: Rotate point around plane in basis
				let mut vr = Point::zeros(2);
				vr[0] = vf[0]*theta.cos() - vf[1]*theta.sin();
				vr[1] = vf[0]*theta.sin() + vf[1]*theta.cos();
				
				//Step 3: Determine non-basis coordinates of rotated point and intersection point
				let mut vc = Point::zeros(self.rank); //Intersection point
				let mut vrc = Point::zeros(self.rank); //Rotated point
				for i in 0..self.rank {
					vrc[i] = vr[0]*v1[i]+vr[1]*v2[i];
					vc[i] = vf[0]*v1[i]+vf[1]*v2[i];
				}
				
				//Step 4: Reverse vector transformation between original point and intersection point onto rotated point. This is our new point.
				//new v = vrc + v - vc
				for i in 0..self.rank {
					v[i] = vrc[i] + v[i] - vc[i];
				}
			}	
			
			println!("Rotated!");
		
		}
	
    }

    fn name_action(&self, name: &mut String) {
        *name = format!("Rotated {}", name);
    }

    fn build(&mut self, ui: &mut Ui) {
        ui.add(egui::Checkbox::new(&mut self.degcheck, "Use degrees instead of radians"));
		
		ui.horizontal(|ui| {
			
			if self.degcheck {
			   ui.add(egui::DragValue::new(&mut self.rot).speed(1.0).clamp_range::<f64>(0.0..=360.0));
			}
			else{
				ui.add(egui::DragValue::new(&mut self.rot).speed(0.01).clamp_range::<f64>(0.0..=6.283185307179586));
			}
			
			ui.label("Rotation"); 
        });
		
		
		ui.separator();
		
		ui.add(egui::Checkbox::new(&mut self.origincheck, "Use a third origin point"));
		
		ui.add(PointWidget::new(&mut self.p1, "First point"));
		ui.add(PointWidget::new(&mut self.p2, "Second point"));
		if self.origincheck {
			ui.add(PointWidget::new(&mut self.po, "Origin point"));
		}
		
    }
	
    fn dim(&self) -> usize {
        self.rank
    }

    fn default_with(dim: usize) -> Self {
        Self {
            rank: dim,
			rot: 0.0,
            p1: Point::zeros(dim),
			p2: Point::zeros(dim),
			po: Point::zeros(dim),
            ..Default::default()
        }
    }

    fn update(&mut self, dim: usize) {
        self.rank = dim;
        self.p1 = Point::zeros(dim);
		self.p2 = Point::zeros(dim);
		self.po = Point::zeros(dim);
    }
}
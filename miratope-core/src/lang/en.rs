//! Since the [`Language`](super::Language) trait already defaults to English,
//! we don't really have to do anything here.

use super::{GreekPrefix, Language};

/// The English language.
pub struct En;

impl GreekPrefix for En {}

impl Language for En {}

//! Since the [`Language`](super::Language) trait already defaults to English,
//! we don't really have to do anything here.

use super::super::{GreekPrefix, Language};

pub struct En;

impl GreekPrefix for En {}

impl Language for En {}

//! Since the [`Language`](super::Language) trait already defaults to English,
//! we don't really have to do anything here.

pub struct En;

impl super::GreekPrefix for En {}

impl super::Language for En {}

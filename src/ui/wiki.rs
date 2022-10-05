pub enum LinkType {
    Link,
    PTemp,
    Custom,
}

impl Default for LinkType {
    fn default() -> Self {
        Self::Link
    }
}

#[derive(Default)]
pub struct WikiElement {
    pub count: usize,
    pub name: String,
    pub link_type: LinkType,
}

#[derive(Clone, Default)]
pub struct InfoboxField {
    pub name: String,
    pub value: String,
}

impl InfoboxField {
    pub fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
}

#[derive(Default)]
pub struct Infobox {
    pub before_elements: Vec<InfoboxField>,
    pub elements: Vec<Vec<WikiElement>>,
    pub after_elements: Vec<InfoboxField>,
}

#[derive(Default)]
pub struct WikiArticle {
    pub title: String,
    pub infobox: Infobox,
    pub body: String,
    pub categories: Vec<String>,
}
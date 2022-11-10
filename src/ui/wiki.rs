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

pub struct WikiArticle {
    pub infobox: Infobox,
    pub body: String,
    pub categories: Vec<String>,
}

impl Default for WikiArticle {
    fn default() -> Self {
        Self {
            infobox: Infobox {
                before_elements: vec![
                    InfoboxField::new("rank", ""),
                    InfoboxField::new("type", ""),
                    InfoboxField::new("bsa", ""),
                ],

                elements: Vec::new(),

                after_elements: vec![
                        InfoboxField::new("verf", ""),
                        InfoboxField::new("army", ""),
                        InfoboxField::new("reg", ""),
                        InfoboxField::new("symmetry", ""),
                        InfoboxField::new("flags", ""),
                        InfoboxField::new("circum", ""),
                        InfoboxField::new("volume", ""),
                        InfoboxField::new("convex", ""),
                        InfoboxField::new("orient", ""),
                        InfoboxField::new("nature", ""),
                ]
                },
            body: String::default(),
            categories: Vec::new(),
        }
    }
}
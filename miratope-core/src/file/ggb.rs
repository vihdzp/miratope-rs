//! Contains the code that opens a GGB file and parses it into a polytope.

// This code is unfinished.
#![allow(dead_code)]
#![allow(clippy::collapsible_match)]

use std::io::Result as IoResult;

use crate::{conc::Concrete, geometry::Point, Float};

use nalgebra::dvector;
use xml::{
    attribute::OwnedAttribute,
    reader::{EventReader, XmlEvent},
};
use zip::result::ZipError;

type Events<'a> = xml::reader::Events<&'a [u8]>;

/// A wrapper around an iterator over events in an XML file.
pub struct XmlReader<'a>(Events<'a>);

impl<'a> Iterator for XmlReader<'a> {
    type Item = xml::reader::Result<XmlEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a> AsRef<Events<'a>> for XmlReader<'a> {
    fn as_ref(&self) -> &Events<'a> {
        &self.0
    }
}

impl<'a> AsMut<Events<'a>> for XmlReader<'a> {
    fn as_mut(&mut self) -> &mut Events<'a> {
        &mut self.0
    }
}

impl<'a> XmlReader<'a> {
    /// Initializes a new XML reader from a source XML string.
    pub fn new(xml: &'a str) -> Self {
        Self(EventReader::from_str(xml).into_iter())
    }

    /// Reads an XML file until an XML element with a given name is found.
    /// Returns its attributes.
    fn read_until(&mut self, search: &str) -> GgbResult<Vec<OwnedAttribute>> {
        for xml_result in self.as_mut() {
            match xml_result {
                // The next XML event to process:
                Ok(xml_event) => {
                    // If we have an element and it has the correct name, we return
                    // this XML event.
                    if let XmlEvent::StartElement {
                        ref name,
                        attributes,
                        ..
                    } = xml_event
                    {
                        if name.local_name == search {
                            return Ok(attributes);
                        }
                    } else {
                        todo!()
                    }
                }
                // Something went wrong while fetching the next XML event.
                Err(_) => return Err(GgbError::InvalidXml),
            }
        }

        // We didn't find the element we were looking for.
        Err(GgbError::MissingElement)
    }

    /// Reads a point from the GGB file, assuming that we're currently in an XML
    /// label of the form
    /// ```xml
    /// <element type="point3d" label="A">
    /// ```
    fn read_point<T: Float>(&mut self, attributes: &[OwnedAttribute]) -> GgbResult<Vertex<T>> {
        let label = attribute(attributes, "label").unwrap_or_default();
        let coord_attributes = self.read_until("coords")?;

        /// Reads any of the coordinates of a point, saves it in a variable with
        /// the same name.
        macro_rules! read_coord {
            ($x:ident) => {
                let $x: T;

                if let Some(c) = attribute(&coord_attributes, stringify!($x)) {
                    if let Ok(c) = c.parse() {
                        $x = c;
                    } else {
                        return Err(GgbError::ParseError);
                    };
                } else {
                    return Err(GgbError::MissingAttribute);
                }
            };
        }

        read_coord!(x);
        read_coord!(y);
        read_coord!(z);
        read_coord!(w);

        Ok(Vertex {
            coords: dvector![x / w, y / w, z / w],
            label: label.to_string(),
        })
    }

    fn read_edge(&self) -> IoResult<Edge> {
        todo!()
    }
}

/// Possible errors while reading a GGB file.
#[derive(Debug)]
pub enum GgbError {
    /// An attribute of an XML tag wasn't found.
    MissingAttribute,

    /// We couldn't find the next <element> tag.
    MissingElement,

    /// An error occured while reading the ZIP file.
    ZipError(ZipError),

    /// The XML file is not valid.
    InvalidXml,

    /// The GGB file is not valid.
    InvalidGgb,

    /// Some number could not be parsed.
    ParseError,
}

impl std::fmt::Display for GgbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAttribute => write!(f, "missing XML attribute"),
            Self::MissingElement => write!(f, "missing XML element"),
            Self::InvalidXml => write!(f, "invalid XML"),
            Self::InvalidGgb => write!(f, "invalid GGB"),
            Self::ZipError(err) => write!(f, "ZIP error: {}", err),
            Self::ParseError => write!(f, "parse error"),
        }
    }
}

impl From<ZipError> for GgbError {
    fn from(zip: ZipError) -> Self {
        Self::ZipError(zip)
    }
}

/// The result of trying to read a GGB file.
pub type GgbResult<T> = Result<T, GgbError>;

impl std::error::Error for GgbError {}

enum Element {
    Point3D { label: String },
}

/// Returns the value of an attribute with a given name in an XML element.
///
/// This method does a simple linear search over all attributes. This isn't
/// really an issue, as long as we never call this method on some element that
/// might have a large number of attributes.
fn attribute<'a>(attributes: &'a [OwnedAttribute], idx: &str) -> Option<&'a str> {
    for att in attributes {
        if att.name.local_name == idx {
            return Some(&att.value);
        }
    }

    None
}

/// A vertex in a GGB file.
#[derive(Debug)]
struct Vertex<T> {
    /// The coordinates of the vertex.
    coords: Point<T>,

    /// The name of the vertex.
    label: String,
}

struct Edge {
    label: String,
}

struct Face {
    vertices: Vec<String>,
    edges: Vec<String>,
}

fn read_face() -> Face {
    todo!()
}

/// Parses the `geogebra.xml` file to produce a polytope.
pub(super) fn parse_xml<T: Float>(xml: &str) -> GgbResult<Concrete<T>> {
    let mut vertices: Vec<Vertex<T>> = Vec::new();
    let mut edges = Vec::new();
    let mut xml = XmlReader::new(xml);

    loop {
        match xml.as_mut().next() {
            // If the document isn't yet over:
            Some(xml_result) => match xml_result {
                // The next XML event to process:
                Ok(xml_event) => {
                    if let XmlEvent::StartElement {
                        name, attributes, ..
                    } = xml_event
                    {
                        let name = name.local_name;

                        if name == "element" {
                            if let Some(el_type) = attribute(&attributes, "type") {
                                // We found a point.
                                if el_type == "point3d" {
                                    vertices.push(xml.read_point(&attributes)?);
                                }
                                // We found an edge.
                                else if el_type == "segment3d" {
                                    if let Ok(edge) = xml.read_edge() {
                                        edges.push(edge);
                                    }
                                }
                            }
                        }

                        /*  "element" => read_edge(&mut xml),
                        "command" => read_face(&mut xml), */
                    }
                }

                // Something went wrong while fetching the next XML event.
                Err(_) => return Err(GgbError::InvalidXml),
            },

            // The file has finished being read. Time for processing!
            None => todo!(),
        }
    }
}

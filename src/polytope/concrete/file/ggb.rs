use std::io::{self, Error, ErrorKind, Read};

use crate::{geometry::Point, Concrete};

use nalgebra::dvector;
use xml::{
    attribute::OwnedAttribute,
    reader::{EventReader, Events, XmlEvent},
};
use zip::read::ZipArchive;

/// A wrapper around an iterator over events in an XML file.
pub struct XmlReader<'a>(Events<&'a [u8]>);

impl<'a> Iterator for XmlReader<'a> {
    type Item = xml::reader::Result<XmlEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a> XmlReader<'a> {
    pub fn new(xml: &'a str) -> Self {
        Self(EventReader::from_str(xml).into_iter())
    }

    /// Returns a mutable reference to the internal iterator.
    pub fn as_iter_mut(&mut self) -> &mut Events<&'a [u8]> {
        &mut self.0
    }

    /// Reads an XML file until an XML element with a given name is found.
    /// Returns its attributes.
    fn read_until(&mut self, search: &str) -> io::Result<Vec<OwnedAttribute>> {
        for xml_result in self.as_iter_mut() {
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
                    }
                }
                // Something went wrong while fetching the next XML event.
                Err(_) => return GgbErrors::InvalidXml.to_err(),
            }
        }

        // We didn't find the element we were looking for.
        GgbErrors::MissingElement.to_err()
    }

    /// Reads a point from the GGB file, assuming that we're currently in an XML
    /// label of the form
    /// ```xml
    /// <element type="point3d" label="A">
    /// ```
    fn read_point(&mut self, attributes: &[OwnedAttribute]) -> io::Result<Vertex> {
        let label = attribute(&attributes, "label").unwrap_or_default();
        let coord_attributes = self.read_until("coords")?;

        /// Reads any of the coordinates of a point, saves it in a variable with
        /// the same name.
        macro_rules! read_coord {
            ($x:ident) => {
                let $x: crate::Float;

                if let Some(c) = attribute(&coord_attributes, stringify!($x)) {
                    if let Ok(c) = c.parse() {
                        $x = c;
                    } else {
                        return GgbErrors::ParseError.to_err();
                    };
                } else {
                    return GgbErrors::MissingAttribute.to_err();
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

    fn read_edge(&self) -> io::Result<Edge> {
        todo!()
    }
}

/// Possible errors while reading a GGB file.
enum GgbErrors {
    /// An attribute of an XML tag wasn't found.
    MissingAttribute,

    /// We couldn't find the next <element> tag.
    MissingElement,

    /// The XML file is not valid.
    InvalidXml,

    /// The GGB file is not valid.
    InvalidGgb,

    /// Some number could not be parsed.
    ParseError,
}

impl GgbErrors {
    /// Creates a new [`io::Result`] out of this error.
    pub fn to_err<T>(&self) -> io::Result<T> {
        use GgbErrors::*;

        Err(match self {
            MissingAttribute => Error::new(ErrorKind::InvalidData, "Attribute not found."),
            MissingElement => Error::new(ErrorKind::InvalidData, "Element not found."),
            InvalidXml => Error::new(ErrorKind::InvalidData, "Invalid XML data."),
            InvalidGgb => Error::new(ErrorKind::InvalidData, "File is not valid GGB file."),
            ParseError => Error::new(ErrorKind::InvalidData, "Data could not be parsed."),
        })
    }
}

enum Element {
    Point3D { label: String },
}

/// Returns the value of an attribute with a given name in an XML element.
///
/// This method does a simple linear search over all attributes. This isn't
/// really an issue, as long as we never call this method on some element that
/// might have an arbitrarily large number of attributes.
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
struct Vertex {
    /// The coordinates of the vertex.
    coords: Point,

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
fn parse_xml(xml: &str) -> io::Result<Concrete> {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    let mut xml = XmlReader::new(xml);

    loop {
        match xml.as_iter_mut().next() {
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
                Err(_) => return GgbErrors::InvalidXml.to_err(),
            },
            // The file has finished being read. Time for processing!
            None => todo!(),
        }
    }
}

impl Concrete {
    /// Attempts to read a GGB file. If succesful, outputs a polytope in at most
    /// 3D.
    pub fn from_ggb<R: io::Read + io::Seek>(mut zip: ZipArchive<R>) -> io::Result<Self> {
        if let Ok(xml) = String::from_utf8(
            zip.by_name("geogebra.xml")?
                .bytes()
                .map(|b| b.unwrap_or(0))
                .collect(),
        ) {
            parse_xml(&xml)
        } else {
            GgbErrors::InvalidGgb.to_err()
        }
    }
}

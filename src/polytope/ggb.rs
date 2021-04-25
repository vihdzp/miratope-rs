use std::io::{self, Read};
use xml::{
    attribute::OwnedAttribute,
    reader::{EventReader, Events, XmlEvent},
};
use zip::read::ZipArchive;

use crate::{polytope::Polytope, Concrete};

use super::geometry::Point;

enum GgbErrors {
    MissingAttribute,
    MissingElement,
    InvalidXml,
    InvalidGgb,
    ParseError,
}

impl GgbErrors {
    pub fn to_err<T>(&self) -> std::io::Result<T> {
        use GgbErrors::*;

        Err(match self {
            MissingAttribute => io::Error::new(io::ErrorKind::InvalidData, "Attribute not found."),
            MissingElement => io::Error::new(io::ErrorKind::InvalidData, "Element not found."),
            InvalidXml => io::Error::new(io::ErrorKind::InvalidData, "Invalid XML data."),
            InvalidGgb => io::Error::new(io::ErrorKind::InvalidData, "File is not valid GGB file."),
            ParseError => io::Error::new(io::ErrorKind::InvalidData, "Data could not be parsed."),
        })
    }
}

enum Element {
    Point3D { label: String },
}

fn get_attribute(attributes: &[OwnedAttribute], idx: &str) -> Option<String> {
    for att in attributes {
        if att.name.local_name == idx {
            return Some(att.value.clone());
        }
    }

    None
}

/// Reads an XML file until an element with a given name is found. Returns its
/// attributes.
fn read_until<R: io::Read>(xml: &mut Events<R>, search: &str) -> io::Result<Vec<OwnedAttribute>> {
    for xml_result in xml {
        match xml_result {
            // The next XML event to process:
            Ok(xml_event) => {
                // If we have an element and it has the correct name, we return
                // this XML event.
                if let XmlEvent::StartElement {
                    ref name,
                    attributes,
                    namespace: _,
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

/// A vertex in a GGB file.
#[derive(Debug)]
struct Vertex {
    /// The coordinates of the vertex.
    coords: Point,

    /// The name of the vertex.
    label: String,
}

fn read_point<R: std::io::Read>(xml: &mut Events<R>, label: String) -> io::Result<Option<Vertex>> {
    // Verifies that we're dealing with a point and not something else.
    let attributes = read_until(xml, "element")?;

    if get_attribute(&attributes, "type") != Some("point3d".to_string()) {
        return Ok(None);
    }

    // Reads the coordinates of the point.
    let attributes = read_until(xml, "coords")?;

    macro_rules! read_coord {
        ($x:ident) => {
            let $x: f64;

            if let Some(c) = get_attribute(&attributes, stringify!($x)) {
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

    Ok(Some(Vertex {
        coords: vec![x / w, y / w, z / w].into(),
        label,
    }))
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
fn parse_xml(xml: String) -> io::Result<Concrete> {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();

    let mut xml = EventReader::from_str(&xml).into_iter();

    loop {
        match xml.next() {
            // If the document isn't yet over:
            Some(xml_result) => match xml_result {
                // The next XML event to process:
                Ok(xml_event) => {
                    if let XmlEvent::StartElement {
                        name,
                        attributes,
                        namespace: _,
                    } = xml_event
                    {
                        let name = name.local_name;
                        if name == "expression" {
                            if let Some(label) = get_attribute(&attributes, "label") {
                                if let Ok(Some(vertex)) = read_point(&mut xml, label) {
                                    vertices.push(vertex);
                                }
                            }
                        } else if name == "element" {
                            if let Some(label) = get_attribute(&attributes, "label") {
                                edges.push(Edge { label });
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
            None => return Ok(Concrete::hypercube(3)),
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
            parse_xml(xml)
        } else {
            GgbErrors::InvalidGgb.to_err()
        }
    }
}

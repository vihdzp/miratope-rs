use std::io::{self, Read};
use xml::{
    attribute::OwnedAttribute,
    reader::{EventReader, Events, XmlEvent},
};
use zip::read::ZipArchive;

use crate::{polytope::Polytope, Concrete};

enum Element {
    Point3D { label: String },
}

fn get_attribute(attributes: &Vec<OwnedAttribute>, idx: &str) -> Option<String> {
    for att in attributes {
        if att.name.local_name == idx {
            return Some(att.value.clone());
        }
    }

    None
}

fn read_until<R: io::Read>(xml: &mut Events<R>, search: &str) -> io::Result<XmlEvent> {
    for xml_result in xml {
        match xml_result {
            Ok(xml_event) => match xml_event {
                XmlEvent::StartElement {
                    ref name,
                    attributes: _,
                    namespace: _,
                } => {
                    if name.local_name == search {
                        return Ok(xml_event);
                    }
                }
                _ => {}
            },
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid XML data.",
                ))
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "Mismatched XML tags.",
    ))
}

fn parse_xml(xml: String) -> io::Result<Concrete> {
    let mut xml = EventReader::from_str(&xml).into_iter();
    loop {
        match xml.next() {
            None => return Ok(Concrete::hypercube(3)),
            Some(xml_result) => match xml_result {
                Ok(xml_event) => match xml_event {
                    XmlEvent::StartElement {
                        name,
                        attributes,
                        namespace: _,
                    } => {
                        if name.local_name == "expression" {
                            if get_attribute(&attributes, "type") == Some("point".to_string()) {
                                let _label = get_attribute(&attributes, "label");
                                read_until(&mut xml, "element")?;
                            }
                            todo!()
                        }
                    }
                    _ => {}
                },
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid XML data.",
                    ))
                }
            },
        }
    }
}

impl Concrete {
    pub fn from_ggb<R: io::Read + io::Seek>(mut zip: ZipArchive<R>) -> io::Result<Self> {
        if let Ok(xml) = String::from_utf8(
            zip.by_name("geogebra.xml")?
                .bytes()
                .map(|b| b.unwrap_or(0))
                .collect(),
        ) {
            parse_xml(xml)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File is not valid GGB file.",
            ))
        }
    }
}

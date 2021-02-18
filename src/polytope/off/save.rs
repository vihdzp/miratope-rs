use super::super::Polytope;
use std::{fs, io::Result as IoResult, path::Path};

const EL_NAMES: &[&str] = &["Vertices", "Edges", "Faces"];

/// A set of options to be used when saving the OFF file.
#[derive(Clone, Copy)]
pub struct OFFOptions {
    /// Whether the OFF file should have comments specifying each face type.
    comments: bool,
    /// Whether all unnecessary whitespace should be removed.
    compress: bool,
    /// Whether the file should be compatible with Stella (3D and 4D only).
    stella_compat: bool,
}

impl Default for OFFOptions {
    fn default() -> Self {
        OFFOptions {
            comments: true,
            compress: false,
            stella_compat: true,
        }
    }
}

pub fn to_src(p: Polytope, opt: OFFOptions) -> String {
    let dim = p.rank();
    let mut off = String::new();
    let mut opt = opt;

    // Stella compatibility only matters in 3D and 4D.
    if !(dim == 3 || dim == 4) {
        opt.stella_compat = false;
    }

    // Newline character.
    let newline = if !opt.stella_compat && opt.compress {
        " "
    } else {
        "\n"
    };

    // Writes header.
    if dim != 3 {
        off += &dim.to_string();
    }
    off += "OFF";
    off += newline;

    off
}

pub fn to_path(fp: &Path, p: Polytope, opt: OFFOptions) -> IoResult<()> {
    fs::write(fp, to_src(p, opt))
}

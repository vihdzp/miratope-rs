//! Reading from and writing to files in various different formats.

pub mod ggb;
pub mod off;

use super::Concrete;
use off::OffError;

#[derive(Debug)]
/// Any error encountered while trying to load a polytope.
pub enum FileError {
    OffError(OffError),
    GgbError,
    InvalidFile,
    InvalidExtension,
}

pub type FileResult<T> = Result<T, FileError>;

impl Concrete {
    /// Loads a polytope from a file path.
    pub fn from_path(fp: &impl AsRef<std::path::Path>) -> std::io::Result<FileResult<Self>> {
        use std::{ffi::OsStr, fs};

        let ext = fp.as_ref().extension();

        Ok(if ext == Some(OsStr::new("off")) {
            String::from_utf8(fs::read(fp)?).map_or(Err(FileError::InvalidFile), |x| {
                Self::from_off(x).map_err(FileError::OffError)
            })
        } else if ext == Some(OsStr::new("ggb")) {
            Ok(Self::from_ggb(zip::read::ZipArchive::new(&mut fs::File::open(fp)?)?).unwrap())
        } else {
            Err(FileError::InvalidExtension)
        })
    }
}

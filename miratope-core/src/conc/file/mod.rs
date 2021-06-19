//! Reading from and writing to files in various different formats.

pub mod ggb;
pub mod off;

use super::Concrete;
use off::OffError;

pub use std::io::Result as IoResult;

/// Any error encountered while trying to load a polytope.
#[derive(Clone, Copy, Debug)]
pub enum FileError<'a> {
    /// An error while reading an OFF file.
    OffError(OffError),

    /// An error while reading a GGB file.
    GgbError,

    /// A non-supported file extension.
    InvalidExtension(&'a str),
}

impl<'a> std::fmt::Display for FileError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            FileError::OffError(err) => write!(f, "OFF error: {}", err),
            FileError::GgbError => write!(f, "GGB error"),
            FileError::InvalidExtension(ext) => write!(f, "invalid file extension \"{}\"", ext),
        }
    }
}

impl<'a> std::error::Error for FileError<'a> {}

impl<'a> From<OffError> for FileError<'a> {
    fn from(err: OffError) -> Self {
        FileError::OffError(err)
    }
}

/// The result of loading a polytope from a file.
pub type FileResult<'a, T> = Result<T, FileError<'a>>;

impl Concrete {
    /// Loads a polytope from a file path.
    pub fn from_path<U: AsRef<std::path::Path>>(fp: &U) -> IoResult<FileResult<Self>> {
        use std::{ffi::OsStr, fs};

        let ext = fp
            .as_ref()
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or_default();

        Ok(match ext {
            // Reads the file as an OFF file.
            "off" => String::from_utf8(fs::read(fp)?)
                .map_or(Err(OffError::InvalidFile.into()), |x| {
                    Ok(Self::from_off(&x)?)
                }),

            // Reads the file as a GGB file.
            "ggb" => {
                Ok(Self::from_ggb(zip::read::ZipArchive::new(&mut fs::File::open(fp)?)?).unwrap())
            }

            // Could not recognize the file extension.
            ext => Err(FileError::InvalidExtension(ext)),
        })
    }
}

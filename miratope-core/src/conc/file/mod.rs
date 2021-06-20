//! Reading from and writing to files in various different formats.

pub mod ggb;
pub mod off;

use self::ggb::GgbError;

use super::Concrete;
use off::OffError;
use zip::result::ZipError;

pub use std::io::Error as IoError;
use std::string::FromUtf8Error;

/// Any error encountered while trying to load a polytope.
#[derive(Debug)]
pub enum FileError<'a> {
    /// An error while reading an OFF file.
    OffError(OffError),

    /// An error while reading a GGB file.
    GgbError(GgbError),

    /// Some generic I/O error occured.
    IoError(IoError),

    /// The file couldn't be parsed as UTF-8.
    InvalidFile(FromUtf8Error),

    /// An error while opening the GGB file (which is really a ZIP file in
    /// disguise).
    ZipError(ZipError),

    /// A non-supported file extension.
    InvalidExtension(&'a str),
}

impl<'a> std::fmt::Display for FileError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileError::OffError(err) => write!(f, "OFF error: {}", err),
            FileError::GgbError(err) => write!(f, "GGB error: {}", err),
            FileError::IoError(err) => write!(f, "IO error: {}", err),
            FileError::ZipError(err) => {
                write!(f, "ZIP error encountered while opening GGB: {}", err)
            }
            FileError::InvalidFile(err) => write!(f, "invalid file: {}", err),
            FileError::InvalidExtension(ext) => write!(f, "invalid file extension \"{}\"", ext),
        }
    }
}

impl<'a> std::error::Error for FileError<'a> {}

impl<'a> From<OffError> for FileError<'a> {
    fn from(err: OffError) -> Self {
        Self::OffError(err)
    }
}

impl<'a> From<GgbError> for FileError<'a> {
    fn from(err: GgbError) -> Self {
        Self::GgbError(err)
    }
}

impl<'a> From<FromUtf8Error> for FileError<'a> {
    fn from(err: FromUtf8Error) -> Self {
        Self::InvalidFile(err)
    }
}

impl<'a> From<IoError> for FileError<'a> {
    fn from(err: IoError) -> Self {
        Self::IoError(err)
    }
}

impl<'a> From<ZipError> for FileError<'a> {
    fn from(err: ZipError) -> Self {
        Self::ZipError(err)
    }
}

/// The result of loading a polytope from a file.
pub type FileResult<'a, T> = Result<T, FileError<'a>>;

impl Concrete {
    /// Loads a polytope from a file path.
    ///
    /// # Todo
    /// Can we perhaps return a single error type?
    pub fn from_path<U: AsRef<std::path::Path>>(fp: &U) -> FileResult<Self> {
        use std::{ffi::OsStr, fs};

        let ext = fp
            .as_ref()
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or_default();

        match ext {
            // Reads the file as an OFF file.
            "off" => match String::from_utf8(fs::read(fp)?) {
                Ok(src) => Ok(Self::from_off(&src)?),
                Err(err) => Err(err.into()),
            },

            // Reads the file as a GGB file.
            "ggb" => Ok(Self::from_ggb(fs::File::open(fp)?)?),

            // Could not recognize the file extension.
            ext => Err(FileError::InvalidExtension(ext)),
        }
    }
}

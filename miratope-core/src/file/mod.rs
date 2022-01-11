//! Reading from and writing to files in various different formats.

pub mod ggb;
pub mod off;

use self::{
    ggb::{GgbError, GgbResult},
    off::{OffParseResult, OffReader},
};
use crate::conc::Concrete;

use off::OffParseError;
use zip::result::ZipError;

pub use std::io::Error as IoError;
use std::{fs::File, string::FromUtf8Error};

/// Any error encountered while trying to load a polytope.
#[derive(Debug)]
pub enum FileError<'a> {
    /// An error while reading an OFF file.
    OffError(OffParseError),

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
            Self::OffError(err) => write!(f, "OFF error: {}", err),
            Self::GgbError(err) => write!(f, "GGB error: {}", err),
            Self::IoError(err) => write!(f, "IO error: {}", err),
            Self::ZipError(err) => write!(f, "ZIP error while opening GGB: {}", err),
            Self::InvalidFile(err) => write!(f, "invalid file: {}", err),
            Self::InvalidExtension(ext) => write!(f, "invalid file extension \"{}\"", ext),
        }
    }
}

impl<'a> std::error::Error for FileError<'a> {}

/// [`OffParseError`] is a type of [`FileError`].
impl<'a> From<OffParseError> for FileError<'a> {
    fn from(err: OffParseError) -> Self {
        Self::OffError(err)
    }
}

/// [`GgbError`] is a type of [`FileError`].
impl<'a> From<GgbError> for FileError<'a> {
    fn from(err: GgbError) -> Self {
        Self::GgbError(err)
    }
}

/// [`FromUtf8Error`] is a type of [`FileError`].
impl<'a> From<FromUtf8Error> for FileError<'a> {
    fn from(err: FromUtf8Error) -> Self {
        Self::InvalidFile(err)
    }
}

/// [`IoError`] is a type of [`FileError`].
impl<'a> From<IoError> for FileError<'a> {
    fn from(err: IoError) -> Self {
        Self::IoError(err)
    }
}

/// [`ZipError`] is a type of [`FileError`].
impl<'a> From<ZipError> for FileError<'a> {
    fn from(err: ZipError) -> Self {
        Self::ZipError(err)
    }
}

/// The result of loading a polytope from a file.
pub type FileResult<'a, T> = Result<T, FileError<'a>>;

/// A trait for polytopes that can be read from an OFF file or a GGB file.
pub trait FromFile: Sized {
    /// Converts an OFF file into a new struct of type `Self`.
    ///
    /// # Todo
    /// Maybe don't load the entire file at once?
    fn from_off(src: &str) -> OffParseResult<Self>;

    /// Attempts to read a GGB file. If succesful, outputs a polytope in at most
    /// 3D.
    fn from_ggb(file: File) -> GgbResult<Self>;

    /// Loads a polytope from a file path.
    fn from_path<U: AsRef<std::path::Path>>(fp: &U) -> FileResult<'_, Self> {
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

impl FromFile for Concrete {
    fn from_off(src: &str) -> OffParseResult<Self> {
        OffReader::new(src).build()
    }

    /// Attempts to read a GGB file. If succesful, outputs a polytope in at most
    /// 3D.
    fn from_ggb(mut file: File) -> GgbResult<Self> {
        use std::io::Read;

        if let Ok(xml) = String::from_utf8(
            zip::read::ZipArchive::new(&mut file)?
                .by_name("geogebra.xml")?
                .bytes()
                .map(|b| b.unwrap_or(0))
                .collect(),
        ) {
            ggb::parse_xml(&xml)
        } else {
            Err(GgbError::InvalidGgb)
        }
    }
}

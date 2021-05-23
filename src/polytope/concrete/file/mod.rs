//! Reading from and writing to files in various different formats.

pub mod ggb;
pub mod off;

use super::Concrete;
use off::OffError;

/// Any error encountered while trying to load a polytope.
pub enum FileError<'a> {
    /// An error while reading an OFF file.
    OffError(OffError),

    /// An error while reading a GGB file.
    GgbError,

    /// A non-supported file extension.
    InvalidExtension(Option<&'a str>),
}

impl<'a> std::fmt::Debug for FileError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileError::OffError(err) => write!(f, "OFF error: {:?}", err),
            FileError::GgbError => write!(f, "GGB error"),
            FileError::InvalidExtension(ext) => {
                write!(f, "invalid file extension")?;
                if let Some(ext) = ext {
                    write!(f, " {}", ext)?;
                }
                Ok(())
            }
        }
    }
}

pub type FileResult<'a, T> = Result<T, FileError<'a>>;

impl Concrete {
    /// Loads a polytope from a file path.
    pub fn from_path<U: AsRef<std::path::Path>>(fp: &U) -> std::io::Result<FileResult<Self>> {
        use std::{ffi::OsStr, fs};

        let ext = fp.as_ref().extension();

        // Reads the file as an OFF file.
        Ok(if ext == Some(OsStr::new("off")) {
            String::from_utf8(fs::read(fp)?)
                .map_or(Err(FileError::OffError(OffError::InvalidFile)), |x| {
                    Self::from_off(&x).map_err(FileError::OffError)
                })
        }
        // Reads the file as a GGB file.
        else if ext == Some(OsStr::new("ggb")) {
            Ok(Self::from_ggb(zip::read::ZipArchive::new(&mut fs::File::open(fp)?)?).unwrap())
        }
        // Could not recognize the file extension.
        else {
            Err(FileError::InvalidExtension(
                ext.map(|ext| ext.to_str()).flatten(),
            ))
        })
    }
}

#[cfg(tests)]
mod tests {}

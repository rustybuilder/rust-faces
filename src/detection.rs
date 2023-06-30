use crate::{nms::Nms, Rect};
use ndarray::ArrayViewD;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustFacesError {
    #[error("IO error: {0}")]
    IoError(std::io::Error),
    #[error("Image error: {0}")]
    ImageError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<std::io::Error> for RustFacesError {
    fn from(err: std::io::Error) -> Self {
        RustFacesError::IoError(err)
    }
}

pub type RustFacesResult<R> = Result<R, RustFacesError>;

/// Face detection result.
#[derive(Debug, Clone)]
pub struct Face {
    /// Face's bounding rectangle.
    pub rect: Rect,
    /// Confidence of the detection.
    pub confidence: f32,
    /// Landmarks of the face.
    pub landmarks: Option<Vec<(f32, f32)>>,
}

/// Face detector trait.
pub trait FaceDetector: Sync + Send {
    /// Detects faces in the given image.
    ///
    /// # Arguments
    ///
    /// * `image` - Image to detect faces in. Should be in RGB format.
    fn detect(&self, image: ArrayViewD<u8>) -> RustFacesResult<Vec<Face>>;
}

/// Face detection common parameters.
#[derive(Debug, Copy, Clone)]
pub struct DetectionParams {
    pub score_threshold: f32,
    pub nms: Nms,
}

impl Default for DetectionParams {
    /// Default parameters.
    ///
    /// Sets the score threshold to 0.95 and uses the default 0.3 as NMS threshold.
    fn default() -> Self {
        Self {
            score_threshold: 0.95,
            nms: Nms::default(),
        }
    }
}

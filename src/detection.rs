use crate::Rect;
use ndarray::ArrayViewD;
use thiserror::Error;

/// Error type for RustFaces.
#[derive(Error, Debug)]
pub enum RustFacesError {
    /// IO errors.
    #[error("IO error: {0}")]
    IoError(std::io::Error),
    /// Errors related to image processing
    #[error("Image error: {0}")]
    ImageError(String),
    /// Errors related to inference engine (e.g. ONNX runtime)
    #[error("Inference error: {0}")]
    InferenceError(String),
    /// Other errors.
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

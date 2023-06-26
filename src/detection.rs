use crate::{nms::Nms, Rect};
use image::{ImageBuffer, Rgb};
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

#[derive(Debug, Clone)]
pub struct Face {
    pub rect: Rect,
    pub confidence: f32,
    pub landmarks: Option<Vec<(f32, f32)>>,
}

pub trait FaceDetector {
    fn detect(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> RustFacesResult<Vec<Face>>;
}

#[derive(Debug, Copy, Clone)]
pub struct DetectionParams {
    pub score_threshold: f32,
    pub nms: Nms,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            score_threshold: 0.95,
            nms: Nms::default(),
        }
    }
}

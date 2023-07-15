mod rect;
pub use rect::Rect;

mod detection;
pub use detection::{DetectionParams, Face, FaceDetector, RustFacesError, RustFacesResult};

mod ort;

mod nms;
pub use nms::Nms;

mod imaging;
pub use imaging::{ToArray3, ToRgb8};

#[cfg(test)]
pub mod testing;

mod blazeface;
pub use blazeface::BlazeFace;

mod mtcnn;
pub use mtcnn::{MtCnn, MtCnnParams};

mod builder;

#[cfg(feature = "viz")]
pub mod viz;

pub use builder::{FaceDetection, FaceDetectorBuilder, InferParams, Provider};

mod model_repository;

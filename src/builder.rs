use std::sync::Arc;

use crate::{
    detection::{DetectionParams, FaceDetector, RustFacesResult},
    model_repository::{GitHubRepository, ModelRepository},
    BlazeFace, Nms,
};

#[derive(Clone, Copy, Debug)]
pub enum FaceDetection {
    BlazeFace640 = 0,
    BlazeFace320 = 1,
}

#[derive(Clone, Debug)]
enum OpenMode {
    File(String),
    Download,
}

#[derive(Clone, Copy, Debug)]
pub enum Provider {
    /// Uses the, default, CPU inference
    OrtCpu,
    /// Uses the Cuda inference. May not be available depending of your Onnx runtime installation.
    OrtCuda,
    /// Uses Intel's OpenVINO inference. May not be available depending of your Onnx runtime installation.
    OrtVino,
}

/// Inference parameters.
pub struct InferParams {
    /// Chooses the ONNX runtime provider.
    pub provider: Provider,
    /// Sets the number of intra-op threads.
    pub intra_threads: Option<usize>,
    /// Sets the number of inter-op threads.
    pub inter_threads: Option<usize>,
}

impl Default for InferParams {
    fn default() -> Self {
        Self {
            provider: Provider::OrtCpu,
            intra_threads: None,
            inter_threads: None,
        }
    }
}

/// Builder for loading or downloading and creating face detectors.
pub struct FaceDetectorBuilder {
    detector: FaceDetection,
    open_mode: OpenMode,
    params: DetectionParams,
    infer_params: InferParams,
}

impl FaceDetectorBuilder {
    /// Create a new builder for the given face detector.
    ///
    /// # Arguments
    ///
    /// * `detector` - The face detector to build.
    pub fn new(detector: FaceDetection) -> Self {
        Self {
            detector,
            open_mode: OpenMode::Download,
            params: DetectionParams::default(),
            infer_params: InferParams::default(),
        }
    }

    /// Load the model from the given file path.
    pub fn from_file(mut self, path: String) -> Self {
        self.open_mode = OpenMode::File(path);
        self
    }

    /// Download the model from the model repository.
    pub fn download(mut self) -> Self {
        self.open_mode = OpenMode::Download;
        self
    }

    /// Set the detection parameters.
    pub fn detect_params(mut self, params: DetectionParams) -> Self {
        self.params = params;
        self
    }

    /// Set the non-maximum suppression.
    pub fn nms(mut self, nms: Nms) -> Self {
        self.params.nms = nms;
        self
    }

    /// Sets the inference parameters.
    pub fn infer_params(mut self, params: InferParams) -> Self {
        self.infer_params = params;
        self
    }

    /// Builds a new detector.
    pub fn build(&self) -> RustFacesResult<Box<dyn FaceDetector>> {
        let env = Arc::new(
            ort::Environment::builder()
                .with_name("RustFaces")
                .build()
                .unwrap(),
        );
        let repository = GitHubRepository::new();

        let model_path = match &self.open_mode {
            OpenMode::Download => repository
                .get_model(self.detector)?
                .to_str()
                .unwrap()
                .to_string(),
            OpenMode::File(path) => path.clone(),
        };

        Ok(Box::new(match self.detector {
            FaceDetection::BlazeFace640 => BlazeFace::from_file(env, &model_path, self.params),
            FaceDetection::BlazeFace320 => BlazeFace::from_file(env, &model_path, self.params),
        }))
    }
}

#[cfg(test)]
mod tests {}

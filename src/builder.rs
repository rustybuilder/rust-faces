use std::sync::Arc;

use crate::{
    detection::{DetectionParams, FaceDetector, RustFacesResult},
    model_repository::{GitHubRepository, ModelRepository},
    BlazeFace, Nms,
};

#[derive(Clone, Copy, Debug)]
pub enum FaceDetection {
    BlazeFace640,
}

#[derive(Clone, Debug)]
enum OpenMode {
    File(String),
    Download,
}

#[derive(Clone, Copy, Debug)]
pub enum Provider {
    OrtCpu,
    OrtCuda,
    OrtVino,
}
pub struct InferParams {
    pub provider: Provider,
    pub cpu_cores: usize,
    pub batch_size: usize,
}

impl Default for InferParams {
    fn default() -> Self {
        Self {
            provider: Provider::OrtCpu,
            cpu_cores: 1,
            batch_size: 1,
        }
    }
}

pub struct FaceDetectorBuilder {
    detector: FaceDetection,
    open_mode: OpenMode,
    params: DetectionParams,
    infer_params: InferParams,
}

impl FaceDetectorBuilder {
    pub fn new(detector: FaceDetection) -> Self {
        Self {
            detector,
            open_mode: OpenMode::Download,
            params: DetectionParams::default(),
            infer_params: InferParams::default(),
        }
    }

    pub fn from_file(mut self, path: String) -> Self {
        self.open_mode = OpenMode::File(path);
        self
    }

    pub fn download(mut self) -> Self {
        self.open_mode = OpenMode::Download;
        self
    }

    pub fn detect_params(mut self, params: DetectionParams) -> Self {
        self.params = params;
        self
    }

    pub fn nms(mut self, nms: Nms) -> Self {
        self.params.nms = nms;
        self
    }

    pub fn infer_params(mut self, params: InferParams) -> Self {
        self.infer_params = params;
        self
    }
    pub fn build(&self) -> RustFacesResult<Box<dyn FaceDetector>> {
        let env = Arc::new(
            ort::Environment::builder()
                .with_name("BlazeFace")
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
        }))
    }
}

#[cfg(test)]
mod tests {}

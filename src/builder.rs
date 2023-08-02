use std::sync::Arc;

use ort::ExecutionProvider;

use crate::{
    blazeface::BlazeFaceParams,
    detection::{FaceDetector, RustFacesResult},
    model_repository::{GitHubRepository, ModelRepository},
    BlazeFace, MtCnn, MtCnnParams,
};

pub enum FaceDetection {
    BlazeFace640(BlazeFaceParams),
    BlazeFace320(BlazeFaceParams),
    MtCnn(MtCnnParams),
}

#[derive(Clone, Debug)]
enum OpenMode {
    File(String),
    Download,
}

/// Runtime inference provider. Some may not be available depending of your Onnx runtime installation.
#[derive(Clone, Copy, Debug)]
pub enum Provider {
    /// Uses the, default, CPU inference
    OrtCpu,
    /// Uses the Cuda inference.
    OrtCuda(i32),
    /// Uses Intel's OpenVINO inference.
    OrtVino(i32),
    /// Apple's Core ML inference.
    OrtCoreMl,
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
    /// Default provider is `OrtCpu` (Onnx CPU).
    fn default() -> Self {
        Self {
            provider: Provider::OrtCpu,
            intra_threads: None,
            inter_threads: None,
        }
    }
}

/// Builder for loading or downloading, configuring, and creating face detectors.
pub struct FaceDetectorBuilder {
    detector: FaceDetection,
    open_mode: OpenMode,
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
            infer_params: InferParams::default(),
        }
    }

    /// Load the model from the given file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model file.
    pub fn from_file(mut self, path: String) -> Self {
        self.open_mode = OpenMode::File(path);
        self
    }

    /// Sets the model to be downloaded from the model repository.
    pub fn download(mut self) -> Self {
        self.open_mode = OpenMode::Download;
        self
    }

    /// Sets the inference parameters.
    pub fn infer_params(mut self, params: InferParams) -> Self {
        self.infer_params = params;
        self
    }

    /// Instantiates a new detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the model can't be loaded.
    ///
    /// # Returns
    ///
    /// A new face detector.
    pub fn build(&self) -> RustFacesResult<Box<dyn FaceDetector>> {
        let mut ort_builder = ort::Environment::builder().with_name("RustFaces");

        ort_builder = match self.infer_params.provider {
            Provider::OrtCuda(device_id) => ort_builder
                .with_execution_providers([ExecutionProvider::cuda().with_device_id(device_id)]),
            Provider::OrtVino(_device_id) => {
                return Err(crate::RustFacesError::Other(
                    "OpenVINO is not supported yet.".to_string(),
                ));
            }
            Provider::OrtCoreMl => {
                ort_builder.with_execution_providers([ExecutionProvider::coreml()])
            }
            _ => ort_builder,
        };

        let env = Arc::new(ort_builder.build()?);
        let repository = GitHubRepository::new();

        let model_paths = match &self.open_mode {
            OpenMode::Download => repository
                .get_model(&self.detector)?
                .iter()
                .map(|path| path.to_str().unwrap().to_string())
                .collect(),
            OpenMode::File(path) => vec![path.clone()],
        };

        match &self.detector {
            FaceDetection::BlazeFace640(params) => Ok(Box::new(BlazeFace::from_file(
                env,
                &model_paths[0],
                params.clone(),
            ))),
            FaceDetection::BlazeFace320(params) => Ok(Box::new(BlazeFace::from_file(
                env,
                &model_paths[0],
                params.clone(),
            ))),
            FaceDetection::MtCnn(params) => Ok(Box::new(
                MtCnn::from_file(
                    env,
                    &model_paths[0],
                    &model_paths[1],
                    &model_paths[2],
                    params.clone(),
                )
                .unwrap(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {}

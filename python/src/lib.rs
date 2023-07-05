use numpy::{
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;
use rust::{DetectionParams, Nms};
use rust_faces as rust;

#[pyclass]
#[derive(Copy, Clone)]
enum FaceDetection {
    BlazeFace640 = 0,
    BlazeFace320 = 1,
}

#[pyclass]
#[derive(Copy, Clone)]
#[allow(clippy::enum_variant_names)]
enum InferProvider {
    OrtCpu = 0,
    OrtCuda = 1,
    OrtVino = 2,
    OrtCoreMl = 3,
}

/// Face detector wrapper.
#[pyclass]
struct FaceDetector {
    detector: Box<dyn rust::FaceDetector>,
}

#[pymethods]
impl FaceDetector {
    /// Detects faces in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to detect faces in. Must be a 3D array of shape (height, width, channels) with type uint8.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///
    /// * `rects` - A 2D array of shape (num_faces, 4) containing the bounding boxes of the detected faces.
    /// * `scores` - A 1D array of shape (num_faces,) containing the confidence scores of the detected faces.
    /// * `landmarks` - A 3D array of shape (num_faces, num_landmarks, 2) containing the landmarks of the detected faces.
    fn detect<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<u8>,
    ) -> (&'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray3<f32>) {
        let array = image.as_array();
        let faces = self.detector.detect(array.view()).unwrap();

        let (rect_array, score_array, landmarks_array) = {
            let mut rect_array = Array2::zeros((faces.len(), 4));
            let mut score_array = Array1::zeros(faces.len());
            let num_landmarks = faces
                .first()
                .map(|f| f.landmarks.as_ref().map(|lms| lms.len()).unwrap_or(0))
                .unwrap_or(0);

            let mut landmarks_array = Array3::<f32>::zeros((faces.len(), num_landmarks, 2));
            for i in 0..faces.len() {
                let face = &faces[i];
                rect_array[(i, 0)] = face.rect.x;
                rect_array[(i, 1)] = face.rect.y;
                rect_array[(i, 2)] = face.rect.width;
                rect_array[(i, 3)] = face.rect.height;

                score_array[i] = face.confidence;

                if let Some(landmarks) = face.landmarks.as_ref() {
                    for j in 0..num_landmarks {
                        landmarks_array[(i, j, 0)] = landmarks[j].0;
                        landmarks_array[(i, j, 1)] = landmarks[j].1;
                    }
                }
            }
            (rect_array, score_array, landmarks_array)
        };

        (
            rect_array.into_pyarray(py),
            score_array.into_pyarray(py),
            landmarks_array.into_pyarray(py),
        )
    }
}

/// Builds a face detector.
///
/// # Arguments
///
/// * `detection` - The face detection method to use.
/// * `model_path` - The path to the model file. If not specified, the model will be downloaded.
/// * `score_threshold` - The minimum confidence score for a face to be detected.
/// * `nms_iou` - The IoU threshold for non-maximum suppression.
/// * `infer_provider` - The inference provider to use. If not specified, the cpu provider will be used.
/// * `device_id` - The device id to use. Only applicable for CUDA and OpenVINO providers.
///
/// # Returns
///
/// A `FaceDetector` instance.
///
/// # Raises
///
/// * `PyRuntimeError` - If the detector could not be built.
#[pyfunction]
#[pyo3(signature = (detection, model_path=None, score_threshold=0.95, nms_iou=0.3, infer_provider=None, device_id=0))]
fn build_detector(
    detection: FaceDetection,
    model_path: Option<&str>,
    score_threshold: f32,
    nms_iou: f32,
    infer_provider: Option<InferProvider>,
    device_id: i32,
) -> PyResult<FaceDetector> {
    let detection_method = match detection {
        FaceDetection::BlazeFace640 => rust::FaceDetection::BlazeFace640,
        FaceDetection::BlazeFace320 => rust::FaceDetection::BlazeFace320,
    };

    let mut builder = rust::FaceDetectorBuilder::new(detection_method);

    builder = if let Some(model_path) = model_path {
        builder.from_file(model_path.to_string())
    } else {
        builder.download()
    };

    let provider = match infer_provider {
        Some(InferProvider::OrtCpu) => rust::Provider::OrtCpu,
        Some(InferProvider::OrtCuda) => rust::Provider::OrtCuda(device_id),
        Some(InferProvider::OrtVino) => rust::Provider::OrtVino(device_id),
        Some(InferProvider::OrtCoreMl) => rust::Provider::OrtCoreMl,
        _ => rust::Provider::OrtCuda(0),
    };

    let detector_impl = builder
        .detect_params(DetectionParams {
            score_threshold,
            nms: Nms {
                iou_threshold: nms_iou,
            },
        })
        .infer_params(rust::InferParams {
            provider,
            ..Default::default()
        })
        .build();

    if let Err(err) = detector_impl {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to build detector: {}",
            err
        )))
    } else {
        Ok(FaceDetector {
            detector: detector_impl.unwrap(),
        })
    }
}

/// py-rust-faces is a Python binding to the rust-faces library.
/// (https://github.com/rustybuilder/rust-faces/)
#[pymodule]
fn py_rust_faces(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<FaceDetection>()?;
    m.add_class::<InferProvider>()?;
    m.add_class::<FaceDetector>()?;
    m.add_function(wrap_pyfunction!(build_detector, m)?)?;
    Ok(())
}

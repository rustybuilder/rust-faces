use numpy::{
    ndarray::{Array1, Array2},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;
use rust::{DetectionParams, Nms};
use rust_faces as rust;

#[pyclass]
#[derive(Copy, Clone)]
enum FaceDetection {
    BlazeFace640 = 0,
}

#[pyclass]
struct FaceDetector {
    detector: Box<dyn rust::FaceDetector>,
}

#[pymethods]
impl FaceDetector {
    fn detect<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<u8>,
    ) -> (&'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray2<f32>) {
        let array = image.as_array();
        let faces = self.detector.detect(array.view()).unwrap();

        let (rect_array, score_array, landmarks_array) = {
            let mut rect_array = Array2::zeros((faces.len(), 4));
            let mut score_array = Array1::zeros(faces.len());
            // TODO: fill with landmarks
            let landmarks_array = Array2::<f32>::zeros((faces.len(), 10));
            for i in 0..faces.len() {
                let face = &faces[i];
                rect_array[(i, 0)] = face.rect.x;
                rect_array[(i, 1)] = face.rect.y;
                rect_array[(i, 2)] = face.rect.width;
                rect_array[(i, 3)] = face.rect.height;

                score_array[i] = face.confidence;
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

#[pyfunction]
#[pyo3(signature = (detection, model_path=None, score_threshold=0.95, nms_iou=0.3))]
fn build_detector(
    detection: FaceDetection,
    model_path: Option<&str>,
    score_threshold: f32,
    nms_iou: f32,
) -> PyResult<FaceDetector> {
    let detection_method = match detection {
        FaceDetection::BlazeFace640 => rust::FaceDetection::BlazeFace640,
    };

    let mut builder = rust::FaceDetectorBuilder::new(detection_method);
    if let Some(model_path) = model_path {
        builder = builder.from_file(model_path.to_string());
    } else {
        builder = builder.download();
    }
    Ok(FaceDetector {
        detector: builder
            .detect_params(DetectionParams {
                score_threshold,
                nms: Nms {
                    iou_threshold: nms_iou,
                },
            })
            .build()
            .unwrap(),
    })
}

#[pymodule]
fn py_rust_faces(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<FaceDetection>()?;
    m.add_class::<FaceDetector>()?;
    m.add_function(wrap_pyfunction!(build_detector, m)?)?;
    Ok(())
}

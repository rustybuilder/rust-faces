use crate::detection::RustFacesError;

impl From<ort::Error> for RustFacesError {
    fn from(err: ort::Error) -> Self {
        RustFacesError::InferenceError(err.to_string())
    }
}

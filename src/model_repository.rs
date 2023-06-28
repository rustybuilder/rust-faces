use crate::builder::FaceDetection;
use crate::detection::{RustFacesError, RustFacesResult};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

pub trait ModelRepository {
    fn get_model(&self, face_detector: FaceDetection) -> RustFacesResult<PathBuf>;
}

fn download_file(url: &str, destination: &str) -> RustFacesResult<()> {
    fn get_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("*/*"));
        headers
    }

    let client = reqwest::blocking::Client::new();
    match client.get(url).headers(get_headers()).send() {
        Ok(mut response) => {
            if response.status().is_success() {
                let file = File::create(destination)?;
                let mut writer = BufWriter::new(file);
                while let Ok(bytes_read) = response.copy_to(&mut writer) {
                    if bytes_read == 0 {
                        break;
                    }
                }
                Ok(())
            } else {
                Err(RustFacesError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to download file.",
                )))
            }
        }
        Err(err) => Err(RustFacesError::Other(format!(
            "Failed to download file: {}",
            err
        ))),
    }
}

fn get_cache_dir() -> RustFacesResult<PathBuf> {
    let home_dir = home::home_dir();
    if home_dir.is_none() {
        return Err(RustFacesError::Other(
            "Failed to get home directory.".to_string(),
        ));
    }

    let cache_dir = home_dir.unwrap().join(".rust_faces/");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

pub struct GitHubRepository {}

impl GitHubRepository {
    pub fn new() -> GitHubRepository {
        GitHubRepository {}
    }
}

impl ModelRepository for GitHubRepository {
    fn get_model(&self, face_detector: FaceDetection) -> RustFacesResult<PathBuf> {
        let (url, model_filename) = match face_detector {
            FaceDetection::BlazeFace640 => (
                "https://github.com/rustybuilder/model-zoo/raw/main/face-detection/blazefaces-640.onnx",
                "blazeface-640.onnx",
            ),
            FaceDetection::BlazeFace320 => (
                "https://github.com/rustybuilder/model-zoo/raw/main/face-detection/blazeface-320.onnx",
                "blazeface-320.onnx",
            ),
        };

        let dest_filepath = get_cache_dir()?.join(model_filename);
        if !dest_filepath.exists() {
            download_file(url, dest_filepath.to_str().unwrap())?;
        }

        Ok(dest_filepath)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::FaceDetection,
        model_repository::{GitHubRepository, ModelRepository},
    };

    #[test]
    fn test_google_drive_repository() {
        let repo = GitHubRepository {};
        let model_path = repo.get_model(FaceDetection::BlazeFace640).unwrap();
        assert!(model_path.exists());
    }
}

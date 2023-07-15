use crate::builder::FaceDetection;
use crate::detection::{RustFacesError, RustFacesResult};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Duration;

pub trait ModelRepository {
    fn get_model(&self, face_detector: FaceDetection) -> RustFacesResult<Vec<PathBuf>>;
}

fn download_file(url: &str, destination: &str) -> RustFacesResult<()> {
    fn get_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("*/*"));
        headers
    }

    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .unwrap()
            .tick_strings(&[
                "▹▹▹▹▹",
                "▸▹▹▹▹",
                "▹▸▹▹▹",
                "▹▹▸▹▹",
                "▹▹▹▸▹",
                "▹▹▹▹▸",
                "▪▪▪▪▪",
            ]),
    );
    pb.set_message(format!("Downloading {url}"));
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
                pb.finish_with_message("Done");
                Ok(())
            } else {
                pb.finish_with_message("Failed to download file.");
                Err(RustFacesError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to download file.",
                )))
            }
        }
        Err(err) => {
            pb.finish_with_message("Failed to download file.");
            Err(RustFacesError::Other(format!(
                "Failed to download file: {}",
                err
            )))
        }
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
    fn get_model(&self, face_detector: FaceDetection) -> RustFacesResult<Vec<PathBuf>> {
        let (urls, filenames) = match face_detector {
            FaceDetection::BlazeFace640 => (
                ["https://github.com/rustybuilder/model-zoo/raw/main/face-detection/blazefaces-640.onnx"].as_slice(),
                ["blazeface-640.onnx"].as_slice(),
            ),
            FaceDetection::BlazeFace320 => (
                ["https://github.com/rustybuilder/model-zoo/raw/main/face-detection/blazeface-320.onnx"].as_slice(),
                ["blazeface-320.onnx"].as_slice(),
            ),
            FaceDetection::MtCnn => (
                ["https://github.com/rustybuilder/model-zoo/raw/main/face-detection/mtcnn-pnet.onnx",
                "https://github.com/rustybuilder/model-zoo/raw/main/face-detection/mtcnn-rnet.onnx",
                "https://github.com/rustybuilder/model-zoo/raw/main/face-detection/mtcnn-onet.onnx"].as_slice(),
                ["mtcnn-pnet.onnx", "mtcnn-rnet.onnx", "mtcnn-onet.onnx"].as_slice(),
            )
        };

        let mut result = Vec::new();
        for (url, filename) in urls.iter().zip(filenames) {
            let dest_filepath = get_cache_dir()?.join(filename);
            if !dest_filepath.exists() {
                download_file(url, dest_filepath.to_str().unwrap())?;
            }
            result.push(dest_filepath);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::FaceDetection,
        model_repository::{GitHubRepository, ModelRepository},
    };

    use super::download_file;

    #[test]
    #[ignore]
    fn test_download() {
        download_file(
            "https://github.com/rustybuilder/model-zoo/raw/main/face-detection/blazeface-320.onnx",
            "tests/output/sample_download",
        )
        .unwrap();
    }

    #[test]
    fn test_google_drive_repository() {
        let repo = GitHubRepository {};
        let model_path = repo.get_model(FaceDetection::BlazeFace640).expect("Failed to get model")[0].clone();
        assert!(model_path.exists());
    }
}

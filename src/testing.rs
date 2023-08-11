use std::path::PathBuf;

use image::RgbImage;
use ndarray::Array3;
use rstest::fixture;

use crate::imaging::ToArray3;

#[fixture]
pub fn sample_array_image() -> Array3<u8> {
    image::io::Reader::open("tests/data/images/faces.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8()
        .into_array3()
}

#[fixture]
pub fn sample_image() -> RgbImage {
    image::io::Reader::open("tests/data/images/kate_siegel.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8()
}

#[fixture]
pub fn output_dir() -> PathBuf {
    let output_path = PathBuf::from("tests/output");
    std::fs::create_dir_all(output_path.clone()).expect("Can't create output directory");
    output_path
}

use rust_faces::{
    viz, DetectionParams, FaceDetection, FaceDetectorBuilder, InferParams, Nms, Provider, ToArray3,
    ToRgb8,
};

pub fn main() {
    let face_detector = FaceDetectorBuilder::new(FaceDetection::BlazeFace640)
        .download()
        .detect_params(DetectionParams::default())
        .nms(Nms::default())
        .infer_params(InferParams {
            provider: Provider::OrtCpu,
            intra_threads: Some(5),
            ..Default::default()
        })
        .build()
        .expect("Fail to load the face detector.");

    let image = image::open("tests/data/images/faces.jpg")
        .expect("Can't open test image.")
        .into_rgb8()
        .into_array3();
    let faces = face_detector.detect(image.view().into_dyn()).unwrap();

    let mut image = image.to_rgb8();
    viz::draw_faces(&mut image, faces);
    std::fs::create_dir_all("tests/output").expect("Can't create test output dir.");
    image
        .save("tests/output/should_have_smooth_design.jpg")
        .expect("Can't save test image.");
}

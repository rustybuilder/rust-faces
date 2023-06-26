use rust_faces::{
    viz, DetectionParams, FaceDetection, FaceDetectorBuilder, InferParams, Nms, Provider,
};

pub fn main() {
    let face_detector = FaceDetectorBuilder::new(FaceDetection::BlazeFace640)
        .download()
        .detect_params(DetectionParams::default())
        .nms(Nms::default())
        .infer_params(InferParams {
            provider: Provider::OrtCpu,
            cpu_cores: 5,
            batch_size: 5,
        })
        .build()
        .expect("Fail to load the face detector.");

    let mut image = image::open("tests/data/images/faces.jpg")
        .expect("Can't open test image.")
        .into_rgb8();
    let faces = face_detector.detect(&image).unwrap();

    viz::draw_faces(&mut image, faces);
    std::fs::create_dir_all("tests/output").expect("Can't create test output dir.");
    image
        .save("tests/output/should_have_smooth_design.jpg")
        .expect("Can't save test image.");
}

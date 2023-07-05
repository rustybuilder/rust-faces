use criterion::{criterion_group, criterion_main, Criterion};
use rust_faces::{FaceDetection, FaceDetectorBuilder, InferParams, Provider, ToArray3};

fn criterion_benchmark(c: &mut Criterion) {
    let image = image::open("tests/data/images/faces.jpg")
        .expect("Can't open test image.")
        .into_rgb8()
        .into_array3();

    c.bench_function("blazeface_cpu", |b| {
        let face_detector = FaceDetectorBuilder::new(FaceDetection::BlazeFace640)
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCpu,
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");
        b.iter(|| face_detector.detect(image.view().into_dyn()).unwrap())
    });

    c.bench_function("blazeface_gpu", |b| {
        let face_detector = FaceDetectorBuilder::new(FaceDetection::BlazeFace640)
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCpu,
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");
        b.iter(|| face_detector.detect(image.view().into_dyn()).unwrap())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

use std::sync::Arc;

use image::{
    imageops::{self, FilterType},
    GenericImageView, ImageBuffer, Pixel, Rgb,
};
use itertools::iproduct;
use ndarray::{Array3, ArrayViewD, Axis};
use ort::tensor::{FromArray, OrtOwnedTensor};

use crate::{
    detection::{DetectionParams, FaceDetector, RustFacesResult},
    imaging::make_border,
    Face, Rect,
};

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

fn resize_and_border<I: GenericImageView>(
    image: &I,
    output_size: (u32, u32),
    border_color: I::Pixel,
) -> (Image<I::Pixel>, f32)
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let (input_width, input_height) = image.dimensions();
    let (output_width, output_height) = output_size;
    let ratio = (output_width as f32 / input_width as f32)
        .min(output_height as f32 / input_height as f32)
        .min(1.0); // avoid scaling up.

    let (resize_width, resize_height) = (
        (input_width as f32 * ratio).round() as i32,
        (input_height as f32 * ratio).round() as i32,
    );
    let resized = imageops::resize(
        image,
        resize_width as u32,
        resize_height as u32,
        FilterType::Nearest,
    );

    let (left, right, top, bottom) = {
        let (xpad, ypad) = (
            ((output_width as i32 - resize_width) % 16) as f32 / 2.0,
            ((output_height as i32 - resize_height) % 16) as f32 / 2.0,
        );
        (
            (xpad - 0.1).round() as u32,
            (xpad + 0.1).round() as u32,
            (ypad - 0.1).round() as u32,
            (ypad + 0.1).round() as u32,
        )
    };

    (
        make_border(&resized, top, bottom, left, right, border_color),
        ratio,
    )
}

struct PriorBoxesParams {
    min_sizes: Vec<Vec<usize>>,
    steps: Vec<usize>,
    variance: (f32, f32),
}

impl Default for PriorBoxesParams {
    fn default() -> Self {
        Self {
            min_sizes: vec![vec![8, 11], vec![14, 19, 26, 38, 64, 149]],
            steps: vec![8, 16],
            variance: (0.1, 0.2),
        }
    }
}

struct PriorBoxes {
    anchors: Vec<(f32, f32, f32, f32)>,
    variances: (f32, f32),
}

impl PriorBoxes {
    pub fn new(params: &PriorBoxesParams, image_size: (usize, usize)) -> Self {
        let feature_map_sizes: Vec<(usize, usize)> = params
            .steps
            .iter()
            .map(|&step| (image_size.0 / step, image_size.1 / step))
            .collect();

        let mut anchors = Vec::new();

        for ((f, min_sizes), step) in feature_map_sizes
            .iter()
            .zip(params.min_sizes.iter())
            .zip(params.steps.iter())
        {
            let step = *step;
            for (i, j) in iproduct!(0..f.1, 0..f.0) {
                for min_size in min_sizes {
                    let s_kx = *min_size as f32 / image_size.0 as f32;
                    let s_ky = *min_size as f32 / image_size.1 as f32;
                    let cx = (j as f32 + 0.5) * step as f32 / image_size.0 as f32;
                    let cy = (i as f32 + 0.5) * step as f32 / image_size.1 as f32;
                    anchors.push((cx, cy, s_kx, s_ky));
                }
            }
        }

        Self {
            anchors,
            variances: params.variance,
        }
    }

    pub fn decode_boxes(&self, anchors_pred_locs: &[(f32, f32, f32, f32)]) -> Vec<Rect> {
        anchors_pred_locs
            .iter()
            .zip(self.anchors.iter())
            .map(|(rect, prior)| {
                let (anchor_cx, anchor_cy, s_kx, s_ky) = prior;
                let (x1, y1, x2, y2) = rect;

                let cx = anchor_cx + x1 * self.variances.0 * s_kx;
                let cy = anchor_cy + y1 * self.variances.0 * s_ky;
                let width = s_kx * (x2 * self.variances.1).exp();
                let height = s_ky * (y2 * self.variances.1).exp();
                let x_start = cx - width / 2.0;
                let y_start = cy - height / 2.0;
                Rect::at(x_start, y_start).with_end(width + x_start, height + y_start)
            })
            .collect()
    }
}

pub struct BlazeFace {
    session: ort::Session,
    params: DetectionParams,
    target_size: usize,
    prior_boxes_params: PriorBoxesParams,
}

impl BlazeFace {
    pub fn from_file(
        env: Arc<ort::Environment>,
        model_path: &str,
        params: DetectionParams,
    ) -> Self {
        let session = ort::session::SessionBuilder::new(&env)
            .unwrap()
            .with_model_from_file(model_path)
            .unwrap();
        Self {
            session,
            params,
            target_size: 1280,
            prior_boxes_params: PriorBoxesParams::default(),
        }
    }
}

impl FaceDetector for BlazeFace {
    fn detect(&self, image: ArrayViewD<u8>) -> RustFacesResult<Vec<Face>> {
        let shape = image.shape().to_vec();
        let (width, height, _) = (shape[1], shape[0], shape[2]);

        let image = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(
            width as u32,
            height as u32,
            image.as_slice().unwrap(),
        )
        .unwrap();

        let (image, ratio) = resize_and_border(
            &image,
            (self.target_size as u32, self.target_size as u32),
            Rgb([104, 117, 123]),
        );
        let (input_width, input_height) = image.dimensions();
        let image = Array3::<f32>::from_shape_fn(
            (3, input_height as usize, input_width as usize),
            |(c, y, x)| {
                match c {
                    // https://github.com/zineos/blazeface/blob/main/tools/test.py seems to use OpenCV's BGR
                    0 => image.get_pixel(x as u32, y as u32)[2] as f32 - 104.0,
                    1 => image.get_pixel(x as u32, y as u32)[1] as f32 - 117.0,
                    2 => image.get_pixel(x as u32, y as u32)[0] as f32 - 123.0,
                    _ => unreachable!(),
                }
            },
        )
        .insert_axis(Axis(0));

        let output_tensors = self
            .session
            .run(vec![ort::tensor::InputTensor::from_array(image.into_dyn())])?;

        // Boxes regressions: N box with the format [start x, start y, end x, end y].
        let boxes: OrtOwnedTensor<f32, _> = output_tensors[0].try_extract()?;
        let scores: OrtOwnedTensor<f32, _> = output_tensors[1].try_extract()?;
        let landmarks: OrtOwnedTensor<f32, _> = output_tensors[2].try_extract()?;
        let num_boxes = boxes.view().shape()[1];

        let priors = PriorBoxes::new(
            &self.prior_boxes_params,
            (input_width as usize, input_height as usize),
        );

        let scale_ratios = (input_width as f32 / ratio, input_height as f32 / ratio);
        Ok(self.params.nms.suppress_non_maxima(
            priors
                .decode_boxes(
                    &boxes
                        .view()
                        .to_shape((num_boxes, 4))
                        .unwrap()
                        .axis_iter(Axis(0))
                        .map(|rect| (rect[0], rect[1], rect[2], rect[3]))
                        .collect::<Vec<_>>(),
                )
                .iter()
                .zip(
                    scores
                        .view()
                        .to_shape((num_boxes, 2))
                        .unwrap()
                        .axis_iter(Axis(0)),
                )
                .zip(
                    landmarks
                        .view()
                        .to_shape((num_boxes, 10))
                        .unwrap()
                        .axis_iter(Axis(0)),
                )
                .filter_map(|((rect, score), landmarks)| {
                    let score = score[1];

                    if score > self.params.score_threshold {
                        let rect = rect.scale(scale_ratios.0, scale_ratios.1);
                        Some((rect, score, landmarks))
                    } else {
                        None
                    }
                })
                .map(|(rect, confidence, landmarks)| Face {
                    rect,
                    confidence,
                    landmarks: Some(
                        landmarks
                            .to_vec()
                            .chunks(2)
                            .map(|chunk| (chunk[0] * scale_ratios.0, chunk[1] * scale_ratios.1))
                            .collect(),
                    ),
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        imaging::ToRgb8,
        model_repository::{GitHubRepository, ModelRepository},
        testing::{output_dir, sample_array_image, sample_image},
    };

    use super::*;
    use image::RgbImage;
    use rstest::rstest;

    use std::path::PathBuf;

    #[rstest]
    pub fn test_resize_and_border(sample_image: RgbImage, output_dir: PathBuf) {
        let (resized, _) = resize_and_border(&sample_image, (1280, 1280), Rgb([0, 255, 0]));

        resized.save(output_dir.join("test_resized.jpg")).unwrap();
        assert!(resized.width() == 896);
        assert!(resized.height() == 1280);
    }

    #[cfg(feature = "viz")]
    fn should_detect_impl(
        blaze_model: crate::FaceDetection,
        sample_array_image: Array3<u8>,
        output_dir: PathBuf,
    ) {
        use crate::viz;
        let environment = Arc::new(
            ort::Environment::builder()
                .with_name("BlazeFace")
                .build()
                .unwrap(),
        );

        let drive = GitHubRepository::new();
        let model_path = drive.get_model(blaze_model).expect("Can't download model");

        let face_detector = BlazeFace::from_file(
            environment,
            model_path.to_str().unwrap(),
            DetectionParams {
                score_threshold: 0.75,
                ..Default::default()
            },
        );
        let mut canvas = sample_array_image.to_rgb8();
        let faces = face_detector
            .detect(sample_array_image.into_dyn().view())
            .unwrap();

        viz::draw_faces(&mut canvas, faces);

        canvas
            .save(output_dir.join("blazefaces.png"))
            .expect("Can't save image");
    }

    #[rstest]
    #[cfg(feature = "viz")]
    fn should_detect_640(sample_array_image: Array3<u8>, output_dir: PathBuf) {
        should_detect_impl(
            crate::FaceDetection::BlazeFace640,
            sample_array_image,
            output_dir,
        );
    }

    #[rstest]
    #[cfg(feature = "viz")]
    fn should_detect_320(sample_array_image: Array3<u8>, output_dir: PathBuf) {
        should_detect_impl(
            crate::FaceDetection::BlazeFace320,
            sample_array_image,
            output_dir,
        );
    }
}

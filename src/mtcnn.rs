use std::sync::Arc;

use image::{
    imageops::{self, FilterType},
    ImageBuffer, Rgb, RgbImage,
};
use ndarray::{s, Array3, Array4, ArrayViewD, Axis, Zip};
use ort::tensor::{FromArray, OrtOwnedTensor};

use crate::{DetectionParams, Face, FaceDetector, Rect, RustFacesResult};

#[derive(Copy, Clone, Debug)]
pub struct MtCnnParams {
    pub min_face_size: usize,
    pub thresholds: [f32; 3],
    pub scale_factor: f32,
    pub common: DetectionParams,
}

impl Default for MtCnnParams {
    fn default() -> Self {
        Self {
            min_face_size: 24,
            thresholds: [0.6, 0.7, 0.7],
            scale_factor: 0.709,
            common: DetectionParams::default(),
        }
    }
}

pub struct MtCnn {
    pnet: ort::Session,
    rnet: ort::Session,
    onet: ort::Session,
    params: MtCnnParams,
}

impl MtCnn {
    pub fn from_file(
        env: Arc<ort::Environment>,
        pnet_path: &str,
        rnet_path: &str,
        onet_path: &str,
        params: MtCnnParams,
    ) -> RustFacesResult<Self> {
        let pnet = ort::session::SessionBuilder::new(&env)?.with_model_from_file(pnet_path)?;
        let rnet = ort::session::SessionBuilder::new(&env)?.with_model_from_file(rnet_path)?;
        let onet = ort::session::SessionBuilder::new(&env)?.with_model_from_file(onet_path)?;

        Ok(Self {
            pnet,
            rnet,
            onet,
            params,
        })
    }

    fn run_proposal_inference(
        &self,
        image: &ImageBuffer<Rgb<u8>, &[u8]>,
    ) -> Result<Vec<Face>, crate::RustFacesError> {
        const PNET_CELL_SIZE: usize = 12;
        const PNET_STRIDE: usize = 2;

        let (image_width, image_height) = (image.width() as usize, image.height() as usize);

        let scales = {
            // Make the first scale to match the minimum face size.
            // Example, if the minimum face size is the same as PNET_CELL_SIZE,
            // that means each cell in the output feature map will correspond
            // to a 12x12 pixel region in the input image, hence no resize (first_scale = 1.0).
            let first_scale = PNET_CELL_SIZE as f32 / self.params.min_face_size as f32;

            let mut curr_size = image_width.min(image_height) as f32 * first_scale;
            let mut scale = first_scale;
            let mut scales = Vec::new();

            while curr_size > PNET_CELL_SIZE as f32 {
                scales.push(scale);
                scale *= self.params.scale_factor;
                curr_size *= self.params.scale_factor;
            }
            scales
        };

        let mut face_proposals = Vec::new();
        for scale_factor in scales {
            let image = imageops::resize(
                image,
                (scale_factor * image_width as f32) as u32,
                (scale_factor * image_height as f32) as u32,
                FilterType::Gaussian,
            );

            let (in_width, in_height) = image.dimensions();
            let image = Array4::from_shape_fn(
                (1, 3, in_height as usize, in_width as usize),
                |(_n, c, h, w)| (image.get_pixel(w as u32, h as u32)[c] as f32 - 127.5) / 128.0,
            );

            let output_tensors = self
                .pnet
                .run(vec![ort::tensor::InputTensor::from_array(image.into_dyn())])?;

            let box_regressions: OrtOwnedTensor<f32, _> = output_tensors[0].try_extract()?;
            let scores: OrtOwnedTensor<f32, _> = output_tensors[1].try_extract()?;

            let (net_out_width, net_out_height) = {
                let shape = scores.view().dim();
                (shape[3], shape[2])
            };

            let rescale_factor = 1.0 / scale_factor;
            let mut faces = Vec::with_capacity(net_out_width * net_out_height);
            Zip::indexed(
                scores
                    .view()
                    .to_shape((2, net_out_height, net_out_width))
                    .unwrap()
                    .lanes(Axis(0)),
            )
            .and(
                box_regressions
                    .view()
                    .to_shape((4, net_out_height, net_out_width))
                    .unwrap()
                    .lanes(Axis(0)),
            )
            .for_each(|(row, col), score, regression| {
                let score = score[1];
                if score > self.params.thresholds[0] {
                    let x1 = col as f32 * PNET_STRIDE as f32 + regression[0];
                    let y1 = row as f32 * PNET_STRIDE as f32 + regression[1];
                    let x2 =
                        col as f32 * PNET_STRIDE as f32 + PNET_CELL_SIZE as f32 + regression[2];
                    let y2 =
                        row as f32 * PNET_STRIDE as f32 + PNET_CELL_SIZE as f32 + regression[3];

                    faces.push(Face {
                        rect: Rect::at(x1, y1)
                            .ending_at(x2, y2)
                            .scale(rescale_factor, rescale_factor),
                        confidence: score,
                        landmarks: None,
                    })
                }
            });

            face_proposals.extend(self.params.common.nms.suppress_non_maxima(faces));
        }
        let mut proposals = self.params.common.nms.suppress_non_maxima(face_proposals);
        let image_rect = Rect::at(0.0, 0.0).ending_at(image_width as f32, image_height as f32);
        proposals.iter_mut().for_each(|face| {
            face.rect = face.rect.intersection(&image_rect);
        });
        Ok(proposals)
    }

    fn run_face_inference(
        &self,
        network: &ort::Session,
        proposals: Vec<Face>,
        image: &ImageBuffer<Rgb<u8>, &[u8]>,
        threshold: f32,
        input_size: usize,
    ) -> Result<Vec<Face>, crate::RustFacesError> {
        const BATCH_SIZE: usize = 16;

        let mut rnet_faces = Vec::new();
        for proposal_batch in proposals.chunks(BATCH_SIZE) {
            let mut input_tensor = Array4::zeros((proposal_batch.len(), 3, input_size, input_size));
            for (n, face) in proposal_batch.iter().enumerate() {
                let face_image =
                    RgbImage::from_fn(face.rect.width as u32, face.rect.height as u32, |x, y| {
                        image.get_pixel(face.rect.x as u32 + x as u32, face.rect.y as u32 + y as u32).to_owned()
                    });
                let face_image = imageops::resize(&face_image, input_size as u32, input_size as u32, FilterType::Gaussian);
                input_tensor
                    .slice_mut(s![n, .., .., ..])
                    .assign(&Array3::from_shape_fn((3, input_size, input_size), |(c, h, w)| {
                        (face_image.get_pixel(w as u32, h as u32)[c] as f32 - 127.5) / 128.0
                    }));
            }

            let output_tensors = network.run(vec![ort::tensor::InputTensor::from_array(
                input_tensor.into_dyn(),
            )])?;
            let box_regressions: OrtOwnedTensor<f32, _> = output_tensors[0].try_extract()?;
            let scores: OrtOwnedTensor<f32, _> = output_tensors[1].try_extract()?;

            let batch_faces = itertools::izip!(
                proposal_batch.iter(),
                scores
                    .view()
                    .to_shape((proposal_batch.len(), 2))
                    .unwrap()
                    .lanes(Axis(1))
                    .into_iter(),
                box_regressions
                    .view()
                    .to_shape((proposal_batch.len(), 4))
                    .unwrap()
                    .lanes(Axis(1))
                    .into_iter()
            )
            .filter_map(|(face, score, regression)| {
                let score = score[1];
                if score >= threshold {
                    let width = face.rect.width;
                    let height = face.rect.height;
                    let regression = regression.to_vec();

                    let x1 = face.rect.x + regression[0] * width;
                    let y1 = face.rect.y + regression[1] * height;
                    let x2 = face.rect.right() + regression[2] * width;
                    let y2 = face.rect.bottom() + regression[3] * height;

                    Some(Face {
                        rect: Rect::at(x1, y1).ending_at(x2, y2),
                        confidence: score,
                        landmarks: None,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

            rnet_faces.extend(batch_faces);
        }
        let boxes = self.params.common.nms.suppress_non_maxima_min(rnet_faces);
        Ok(boxes)
    }
}

impl FaceDetector for MtCnn {
    fn detect(&self, image: ArrayViewD<u8>) -> RustFacesResult<Vec<Face>> {
        let shape = image.shape().to_vec();
        let (image_width, image_height) = (shape[1], shape[0]);
        let image = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(
            image_width as u32,
            image_height as u32,
            image.as_slice().unwrap(),
        )
        .unwrap();

        let proposals = self.run_proposal_inference(&image)?;
        let refined_faces = self.run_face_inference(&self.rnet, proposals.clone(), &image, self.params.thresholds[1], 24)?;
        // let optimized_faces = self.run_face_inference(&self.onet, refined_faces.clone(), &image, self.params.thresholds[2], 48)?;
        Ok(refined_faces)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        imaging::ToRgb8,
        model_repository::{GitHubRepository, ModelRepository},
        mtcnn::MtCnn,
        testing::{output_dir, sample_array_image},
        viz,
    };
    use ndarray::Array3;
    use rstest::rstest;
    use std::sync::Arc;

    #[cfg(feature = "viz")]
    #[rstest]
    fn should_detect(sample_array_image: Array3<u8>, output_dir: PathBuf) {
        let environment = Arc::new(
            ort::Environment::builder()
                .with_name("MtCnn")
                .build()
                .unwrap(),
        );

        let drive = GitHubRepository::new();
        let model_paths = drive
            .get_model(crate::FaceDetection::MtCnn)
            .expect("Can't download model");

        let face_detector = MtCnn::from_file(
            environment,
            model_paths[0].to_str().unwrap(),
            model_paths[1].to_str().unwrap(),
            model_paths[2].to_str().unwrap(),
            MtCnnParams::default(),
        )
        .expect("Failed to load MTCNN detector.");
        let mut canvas = sample_array_image.to_rgb8();
        let faces = face_detector
            .detect(sample_array_image.into_dyn().view())
            .expect("Can't detect faces");

        viz::draw_faces(&mut canvas, faces);

        canvas
            .save(output_dir.join("mtcnn.png"))
            .expect("Can't save image");
    }
}

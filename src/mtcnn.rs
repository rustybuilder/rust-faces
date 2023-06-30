use std::sync::Arc;

use image::{ImageBuffer, Rgb};
use ndarray::{Array3, ArrayViewD, Axis};
use ort::tensor::{FromArray, OrtOwnedTensor};

use crate::{DetectionParams, FaceDetector, RustFacesResult, Rect};

pub struct MtCnn {
    pnet: ort::Session,
    rnet: ort::Session,
    onet: ort::Session,
    params: DetectionParams,
}

impl MtCnn {
    pub fn from_file(
        env: Arc<ort::Environment>,
        pnet_path: &str,
        rnet_path: &str,
        onet_path: &str,
        params: DetectionParams,
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
}

impl FaceDetector for MtCnn {
    fn detect(&self, image: ArrayViewD<u8>) -> RustFacesResult<Vec<Face>> {
        let shape = image.shape().to_vec();
        let (width, height, _) = (shape[1], shape[0], shape[2]);

        let image = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(
            width as u32,
            height as u32,
            image.as_slice().unwrap(),
        )
        .unwrap();

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
            .pnet
            .run(vec![ort::tensor::InputTensor::from_array(image.into_dyn())])?;

        let boxes: OrtOwnedTensor<f32, _> = output_tensors[0].try_extract()?;
        let scores: OrtOwnedTensor<f32, _> = output_tensors[1].try_extract()?;
        boxes.view().to_shape((h, w, 2)).unwrap().indexed_iter().map(|
            y, x|{
                Rect {
                    y: stride*y as f32,
                    x: stride*x as f32,
                    width: x + cell_size as f32,
                    height: y + cell_size as f32,
                }
            })
        boxes
            .view()
            .to_shape((h, w, 2))
            .unwrap()
            .axis_iter(Axis(0))
            .zip(
                scores
                    .view()
                    .to_shape((num_boxes, 2))
                    .unwrap()
                    .axis_iter(Axis(0)),
            )
            .filter_map(|(((rect, landmarks), prior), score)| {
                let score = score[1];
                if score > self.params.score_threshold {
                    Some((
                        Rect {
                            x: rect[0] * width as f32,
                            y: rect[1] * height as f32,
                            width: (rect[2] - rect[0]) * width as f32,
                            height: (rect[3] - rect[1]) * height as f32,
                        },
                        score,
                    ))
                } else {
                    None
                }
            });
    }
}

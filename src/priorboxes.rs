use itertools::iproduct;

use crate::Rect;

#[derive(Debug, Clone)]
pub struct PriorBoxesParams {
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

pub struct PriorBoxes {
    pub anchors: Vec<(f32, f32, f32, f32)>,
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

    pub fn decode_box(&self, prior: &(f32, f32, f32, f32), pred: &(f32, f32, f32, f32)) -> Rect {
        let (anchor_cx, anchor_cy, s_kx, s_ky) = prior;
        let (x1, y1, x2, y2) = pred;

        let cx = anchor_cx + x1 * self.variances.0 * s_kx;
        let cy = anchor_cy + y1 * self.variances.0 * s_ky;
        let width = s_kx * (x2 * self.variances.1).exp();
        let height = s_ky * (y2 * self.variances.1).exp();
        let x_start = cx - width / 2.0;
        let y_start = cy - height / 2.0;
        Rect::at(x_start, y_start).ending_at(width + x_start, height + y_start)
    }

    pub fn decode_landmark(
        &self,
        prior: &(f32, f32, f32, f32),
        landmark: (f32, f32),
    ) -> (f32, f32) {
        let (anchor_cx, anchor_cy, s_kx, s_ky) = prior;
        let (x, y) = landmark;
        let x = anchor_cx + x * self.variances.0 * s_kx;
        let y = anchor_cy + y * self.variances.0 * s_ky;
        (x, y)
    }
}

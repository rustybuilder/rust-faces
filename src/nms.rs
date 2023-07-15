use std::collections::HashMap;

use crate::Face;

/// Non-maximum suppression.
#[derive(Copy, Clone, Debug)]
pub struct Nms {
    pub iou_threshold: f32,
}

impl Default for Nms {
    fn default() -> Self {
        Self { iou_threshold: 0.3 }
    }
}

impl Nms {
    /// Suppress non-maxima faces.
    ///
    /// # Arguments
    ///
    /// * `faces` - Faces to suppress.
    ///
    /// # Returns
    ///
    /// * `Vec<Face>` - Suppressed faces.
    pub fn suppress_non_maxima(&self, mut faces: Vec<Face>) -> Vec<Face> {
        faces.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

        let mut faces_map = HashMap::new();
        faces.iter().rev().enumerate().for_each(|(i, face)| {
            faces_map.insert(i, face);
        });

        let mut nms_faces = Vec::with_capacity(faces.len());
        let mut count = 0;
        while !faces_map.is_empty() {
            if let Some((_, face)) = faces_map.remove_entry(&count) {
                nms_faces.push(face.clone());
                //faces_map.retain(|_, face2| face.rect.iou(&face2.rect) < self.iou_threshold);
                faces_map.retain(|_, face2| face.rect.iou(&face2.rect) < self.iou_threshold);
            }
            count += 1;
        }

        nms_faces
    }

        /// Suppress non-maxima faces.
    ///
    /// # Arguments
    ///
    /// * `faces` - Faces to suppress.
    ///
    /// # Returns
    ///
    /// * `Vec<Face>` - Suppressed faces.
    pub fn suppress_non_maxima_min(&self, mut faces: Vec<Face>) -> Vec<Face> {
        faces.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

        let mut faces_map = HashMap::new();
        faces.iter().rev().enumerate().for_each(|(i, face)| {
            faces_map.insert(i, face);
        });

        let mut nms_faces = Vec::with_capacity(faces.len());
        let mut count = 0;
        while !faces_map.is_empty() {
            if let Some((_, face)) = faces_map.remove_entry(&count) {
                nms_faces.push(face.clone());
                //faces_map.retain(|_, face2| face.rect.iou(&face2.rect) < self.iou_threshold);
                faces_map.retain(|_, face2| face.rect.iou_min(&face2.rect) < self.iou_threshold);
            }
            count += 1;
        }

        nms_faces
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{Face, Rect};

    #[rstest]
    fn test_nms() {
        let nms = Nms::default();
        let faces = vec![
            Face {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
                confidence: 0.9,
                landmarks: None,
            },
            Face {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
                confidence: 0.8,
                landmarks: None,
            },
            Face {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
                confidence: 0.7,
                landmarks: None,
            },
        ];

        let faces = nms.suppress_non_maxima(faces);

        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].confidence, 0.9);
    }
}

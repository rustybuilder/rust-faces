use image::{GenericImage, Rgb};

use crate::{Face, Rect};

impl From<Rect> for imageproc::rect::Rect {
    fn from(rect: Rect) -> Self {
        imageproc::rect::Rect::at(rect.x as i32, rect.y as i32)
            .of_size(rect.width as u32, rect.height as u32)
    }
}

/// Draws faces on the image.
pub fn draw_faces<I>(image: &mut I, faces: Vec<Face>)
where
    I: GenericImage<Pixel = Rgb<u8>>,
{
    for face in faces {
        imageproc::drawing::draw_hollow_rect_mut(image, face.rect.into(), Rgb([0, 255, 0]));
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use image::ImageBuffer;

    #[test]
    fn test_draw_faces() {
        let mut image = ImageBuffer::new(100, 100);
        let faces = vec![Face {
            rect: Rect {
                x: 10.0,
                y: 10.0,
                width: 10.0,
                height: 10.0,
            },
            confidence: 0.9,
            landmarks: None,
        }];
        draw_faces(&mut image, faces);
        std::fs::create_dir_all(Path::new("tests/output")).expect("Failed to create output dir.");

        image.save("tests/output/test_draw_faces.png").unwrap();
    }
}

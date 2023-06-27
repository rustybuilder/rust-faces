use image::{flat::SampleLayout, GenericImageView, ImageBuffer, Pixel};
use ndarray::{Array3, ShapeBuilder};

pub fn make_border<I: GenericImageView>(
    image: &I,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    color: I::Pixel,
) -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let (width, height) = image.dimensions();

    let mut new_image = ImageBuffer::new(width + left + right, height + top + bottom);

    for (x, y, pixel) in new_image.enumerate_pixels_mut() {
        if x < left || x >= width + left || y < top || y >= height + top {
            *pixel = color;
        } else {
            *pixel = image.get_pixel(x - left, y - top);
        }
    }
    new_image
}

pub trait ToRgb8 {
    fn to_rgb8(&self) -> ImageBuffer<image::Rgb<u8>, Vec<u8>>;
}

impl ToRgb8 for Array3<u8> {
    fn to_rgb8(&self) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let (height, width, _) = self.dim();
        let mut image = ImageBuffer::new(width as u32, height as u32);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let r = self[[y as usize, x as usize, 0]];
            let g = self[[y as usize, x as usize, 1]];
            let b = self[[y as usize, x as usize, 2]];
            *pixel = image::Rgb([r, g, b]);
        }
        image
    }
}

pub trait ToArray3 {
    type Out;

    fn into_array3(self) -> Self::Out;
}

impl<P> ToArray3 for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out = Array3<P::Subpixel>;

    fn into_array3(self) -> Self::Out {
        let SampleLayout {
            channels,
            channel_stride,
            height,
            height_stride,
            width,
            width_stride,
        } = self.sample_layout();
        let shape = (height as usize, width as usize, channels as usize);
        let strides = (height_stride, width_stride, channel_stride);
        Array3::from_shape_vec(shape.strides(strides), self.into_raw()).unwrap()
    }
}

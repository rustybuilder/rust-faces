use image::{GenericImageView, ImageBuffer, Pixel};

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

use image::DynamicImage;

pub mod enhance;
pub trait ProcessFrame {
    fn process_image(&mut self, original_img: &DynamicImage) -> anyhow::Result<DynamicImage>;
}

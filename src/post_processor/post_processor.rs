use crate::face_processor::face_parsing::FaceParsing;
use crate::pre_processor::pre_processor::PreProcessResult;
use anyhow::Result;
use image::imageops::FilterType;
use image::{imageops, DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};

pub struct PostProcessor<'a> {
    #[allow(unused)]
    face_parser: &'a mut FaceParsing,
    pre_process_result: PreProcessResult,
}

impl<'a> PostProcessor<'a> {
    pub fn new(face_parser: &'a mut FaceParsing, pre_process_result: PreProcessResult) -> Self {
        Self {
            face_parser,
            pre_process_result,
        }
    }

    pub fn process(
        &mut self,
        background: &DynamicImage,
        face_image: &DynamicImage,
        mask: &GrayImage,
    ) -> Result<RgbaImage> {
        let (x1, x2, y1, y2) = self.pre_process_result.target_rect;
        let face_image = face_image.resize(x2 - x1, y2 - y1, FilterType::Lanczos3);
        let mask = imageops::resize(mask, x2 - x1, y2 - y1, FilterType::Lanczos3);
        // let mask = self.generate_face_mask(&face_image)?;

        let (x, _, y, _) = self.pre_process_result.target_rect;
        let mask = rotate_about_center(
            &mask,
            self.pre_process_result.target_rotation_angle,
            Interpolation::Bilinear,
            Luma([0]),
        );
        let face_image = rotate_about_center(
            &face_image.to_rgba8(),
            self.pre_process_result.target_rotation_angle,
            Interpolation::Bilinear,
            Rgba([0, 0, 0, 0]),
        );
        blend_face(background, &face_image.into(), &mask, x, y)
    }
}

// 将交换后的人脸与目标图像混合
fn blend_face(
    background: &DynamicImage,
    face: &DynamicImage,
    mask: &GrayImage,
    x: u32,
    y: u32,
) -> Result<RgbaImage> {
    let mut result = background.clone().to_rgba8();
    let face_rgb = face.to_rgb8();
    let (width, height) = (face_rgb.width(), face_rgb.height());

    // 确保掩码尺寸匹配
    assert_eq!(mask.width(), width, "掩码宽度不匹配");
    assert_eq!(mask.height(), height, "掩码高度不匹配");

    for fy in 0..height {
        for fx in 0..width {
            let alpha = mask.get_pixel(fx, fy)[0] as f32 / 255.0;
            let bg_x = x + fx;
            let bg_y = y + fy;

            // 边界检查
            if bg_x >= result.width() || bg_y >= result.height() {
                continue;
            }

            let face_pixel = face_rgb.get_pixel(fx, fy);
            let bg_pixel = result.get_pixel_mut(bg_x, bg_y);

            // 线性混合
            bg_pixel.0 = [
                (face_pixel[0] as f32 * alpha + bg_pixel[0] as f32 * (1.0 - alpha)) as u8,
                (face_pixel[1] as f32 * alpha + bg_pixel[1] as f32 * (1.0 - alpha)) as u8,
                (face_pixel[2] as f32 * alpha + bg_pixel[2] as f32 * (1.0 - alpha)) as u8,
                255,
            ];
        }
    }

    Ok(result)
}

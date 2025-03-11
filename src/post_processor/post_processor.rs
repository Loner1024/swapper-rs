use anyhow::Result;
use image::{imageops, DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use image::imageops::FilterType;
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use crate::face_processor::face_parsing::FaceParsing;
use crate::frame::enhance::{generate_dynamic_mask, FaceThresholds};
use crate::pre_processor::pre_processor::PreProcessResult;

pub struct PostProcessor<'a> {
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

    pub fn process(&mut self, background: &DynamicImage, face_image: &DynamicImage) -> Result<RgbaImage> {
        let (x1, x2, y1, y2) = self.pre_process_result.target_rect;
        let face_image = face_image.resize(x2-x1, y2-y1, FilterType::Lanczos3);
        let mask = self.generate_face_mask(&face_image)?;

        face_image.save("face.png")?;
        mask.save("mask.png")?;

        let (x, _, y, _)  = self.pre_process_result.target_rect;
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

    fn generate_face_mask(&mut self, face_img: &DynamicImage) -> Result<GrayImage> {
        let face_raw_data = self.face_parser.parse(face_img)?;
        let mask = generate_dynamic_mask(&face_raw_data, &FaceThresholds::default())?;

        let (height, width) = (mask.nrows(), mask.ncols());
        let mut img = GrayImage::new(width as u32, height as u32);

        mask.indexed_iter().for_each(|((y, x), &is_face)| {
            let alpha: u8 = if is_face { 255 } else { 0 };
            img.put_pixel(x as u32, y as u32, Luma([alpha]));
        });
        let resized = imageops::resize(
            &img,
            face_img.width(),
            face_img.height(),
            FilterType::Lanczos3,
        );
        Ok(resized)
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


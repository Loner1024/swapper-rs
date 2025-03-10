use crate::frame::ProcessFrame;
use anyhow::{Context, Result};
use image::{imageops, imageops::FilterType, DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use ndarray::{Array, ArrayBase, Dim, Ix, OwnedRepr};
use opencv::core::{Mat_, Vector};
use opencv::objdetect::CascadeClassifier;
use opencv::{
    core::{Mat, Point, Rect, Size},
    objdetect,
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;
use crate::face_detect::face_detect::FaceDetector;
use crate::face_parsing::face_parsing::{FaceParsing, FaceRawData};

const FACE_SIZE: u32 = 512;

pub struct FaceEnhancer {
    gfpgan: Session,
    face_parser: FaceParsing,
    face_detector: FaceDetector,
}

fn preprocess_image(img: &DynamicImage) -> Result<(Array<f32, Dim<[usize; 4]>>, (u32, u32))> {
    let original_size = (img.width(), img.height());

    // 转换为RGB并调整大小
    let resized = img.resize_exact(512, 512, image::imageops::FilterType::Lanczos3);
    let rgb_img = resized.to_rgb8();

    // 转换为CHW格式的数组并归一化
    let mut array = Array::zeros((1, 3, 512, 512));
    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
        array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
        array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
    }

    Ok((array, original_size))
}

fn postprocess_output(
    output: ndarray::ArrayViewD<f32>,
    original_size: (u32, u32),
) -> Result<DynamicImage> {
    let output_shape = output.shape();
    let (_, _, h, w) = (
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    );

    let mut img_buffer: RgbImage = ImageBuffer::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let r = (output[[0, 0, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
            let g = (output[[0, 1, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
            let b = (output[[0, 2, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r as u8, g as u8, b as u8]));
        }
    }

    // 使用 imageops::resize 代替直接方法调用
    let resized = imageops::resize(
        &img_buffer,
        original_size.0,
        original_size.1,
        FilterType::Lanczos3,
    );

    Ok(DynamicImage::ImageRgb8(resized))
}

impl ProcessFrame for FaceEnhancer {
    fn process_image(&mut self, original_img: &DynamicImage) -> Result<RgbImage> {
        // 人脸检测和对齐
        let faces = self.detect_and_align_faces(&original_img)?;

        // 处理每个人脸
        let mut final_img = original_img.to_rgb8();
        for (face_img, (x, y, w, h)) in faces {
            // 分块增强
            let restored = self.enhance_faces(&face_img)?;
            // 生成人脸mask
            let mask = self.generate_face_mask(&face_img)?;
            // 调整增强后的图像到实际尺寸
            let restored_resized = restored.resize_exact(w, h, FilterType::Lanczos3);

            // 调整掩码尺寸
            let mask_resized = imageops::resize(&mask, w, h, FilterType::Lanczos3);

            // 融合到原图
            self.blend_face(&mut final_img, &restored_resized, &mask_resized, x, y);

        }
        Ok(final_img)
    }
}

impl FaceEnhancer {
    pub fn new(model_path: &Path, parsing_model_path: &Path, cascade_path: &str) -> Result<Self> {
        // 初始化ONNX模型
        let gfpgan = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let face_parser = FaceParsing::new(parsing_model_path.into())?;
        let face_detector = FaceDetector::new(cascade_path)?;

        // 初始化OpenCV人脸检测器

        Ok(Self {
            gfpgan,
            face_parser,
            face_detector,
        })
    }

    pub fn enhance_faces(&mut self, img: &DynamicImage) -> Result<DynamicImage> {
        let (input_array, original_size) = preprocess_image(&img)?;

        // 准备输入张量 (shape: [1, 3, 512, 512])
        let input_tensor = TensorRef::from_array_view(input_array.view())?;

        // 执行推理
        let outputs = self.gfpgan.run(ort::inputs!["input" => input_tensor])?;

        let mut processed_image = img.clone();
        // 后处理输出
        if let Some(output_tensor) = outputs.get("output") {
            let output_array = output_tensor
                .try_extract_tensor::<f32>()
                .context("Failed to extract output tensor")?;

            processed_image = postprocess_output(output_array.view().into_dyn(), original_size)?;
        }
        Ok(processed_image)
    }

    fn detect_and_align_faces(&mut self, img: &DynamicImage) -> Result<Vec<(DynamicImage, (i32, i32, u32, u32))>> {
        self.face_detector.detect(img)
    }

    /// 生成人脸mask
    fn generate_face_mask(&mut self, face_img: &DynamicImage) -> Result<GrayImage> {
        let face_raw_data = self.face_parser.parse(face_img)?;
        let mask = generate_dynamic_mask(&face_raw_data, &FaceThresholds::default())?;

        let (height, width) = (mask.nrows(), mask.ncols());
        let mut img = GrayImage::new(width as u32, height as u32);

        mask.indexed_iter().for_each(|((y, x), &is_face)| {
            let alpha: u8 = if is_face { 255 } else { 0 };
            img.put_pixel(x as u32, y as u32, Luma([alpha]));
        });

        Ok(img)
    }

    /// 融合处理结果到原图
    fn blend_face(
        &self,
        background: &mut RgbImage,
        face: &DynamicImage,
        mask: &GrayImage,
        x: i32,
        y: i32,
    ) {
        let face_rgb = face.to_rgb8();
        let (width, height) = (face_rgb.width(), face_rgb.height());

        // 确保掩码尺寸匹配
        assert_eq!(mask.width(), width, "掩码宽度不匹配");
        assert_eq!(mask.height(), height, "掩码高度不匹配");

        for fy in 0..height {
            for fx in 0..width {
                let alpha = mask.get_pixel(fx, fy)[0] as f32 / 255.0;
                let bg_x = x as u32 + fx;
                let bg_y = y as u32 + fy;

                // 边界检查
                if bg_x >= background.width() || bg_y >= background.height() {
                    continue;
                }

                let face_pixel = face_rgb.get_pixel(fx, fy);
                let bg_pixel = background.get_pixel_mut(bg_x, bg_y);

                // 线性混合
                bg_pixel.0 = [
                    (face_pixel[0] as f32 * alpha + bg_pixel[0] as f32 * (1.0 - alpha)) as u8,
                    (face_pixel[1] as f32 * alpha + bg_pixel[1] as f32 * (1.0 - alpha)) as u8,
                    (face_pixel[2] as f32 * alpha + bg_pixel[2] as f32 * (1.0 - alpha)) as u8,
                ];
            }
        }
    }
}

/// 阈值配置结构体
#[derive(Debug, Clone)]
pub struct FaceThresholds {
    pub skin: f32,
    pub nose: f32,
    pub eyes: f32,
    pub eyebrows: f32,
    pub mouth: f32,
    pub lips: f32,
    pub ears: f32,
    pub neck: f32,
    // 其他通道可根据需要扩展...
}

impl Default for FaceThresholds {
    fn default() -> Self {
        Self {
            skin: 0.5,
            nose: 0.4,
            eyes: 0.3,
            eyebrows: 0.3,
            mouth: 0.4,
            lips: 0.4,
            ears: 0.4,
            neck: 0.5,
        }
    }
}

/// 动态生成组合掩码的工具方法
pub fn generate_dynamic_mask(
    data: &FaceRawData,
    thresholds: &FaceThresholds,
) -> Result<ndarray::Array2<bool>> {
    let shape = data.tensor.shape();
    let (height, width) = (shape[2], shape[3]);
    let mut mask = ndarray::Array2::<bool>::default((height, width));

    // 并行遍历每个像素
    ndarray::Zip::indexed(&mut mask).for_each(|(y, x), value| {
        let skin = data.tensor[[0, 1, y, x]] > thresholds.skin;
        let nose = data.tensor[[0, 2, y, x]] > thresholds.nose;
        let l_eye = data.tensor[[0, 4, y, x]] > thresholds.eyes;
        let r_eye = data.tensor[[0, 5, y, x]] > thresholds.eyes;
        let l_brow = data.tensor[[0, 6, y, x]] > thresholds.eyebrows;
        let r_brow = data.tensor[[0, 7, y, x]] > thresholds.eyebrows;
        let mouth = data.tensor[[0, 10, y, x]] > thresholds.mouth;
        let neck = data.tensor[[0, 17, y, x]] > thresholds.neck;

        *value = skin || nose || l_eye || r_eye || l_brow || r_brow || mouth || neck;
    });

    Ok(mask)
}
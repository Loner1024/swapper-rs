use crate::frame::ProcessFrame;
use anyhow::{Context, Result};
use image::{imageops, imageops::FilterType, DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array, Dim};

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

pub struct FaceEnhancer {
    model: Session,
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
            let r = ((output[[0, 0, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
            let g = ((output[[0, 1, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
            let b = ((output[[0, 2, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0) as u8;
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
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
    fn process_image(&mut self, face_img: &DynamicImage) -> Result<DynamicImage> {
        let restored = self.enhance_faces(&face_img)?;
        Ok(restored)
    }
}

impl FaceEnhancer {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;

        Ok(Self { model })
    }

    pub fn enhance_faces(&mut self, img: &DynamicImage) -> Result<DynamicImage> {
        let (input_array, original_size) = preprocess_image(&img)?;

        // 准备输入张量 (shape: [1, 3, 512, 512])
        let input_tensor = TensorRef::from_array_view(input_array.view())?;

        // 执行推理
        let outputs = self.model.run(ort::inputs!["input" => input_tensor])?;

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
    pub hair: f32,
}

impl Default for FaceThresholds {
    fn default() -> Self {
        Self {
            skin: 0.5,
            nose: 0.5,
            eyes: 0.5,
            eyebrows: 0.5,
            mouth: 0.5,
            lips: 0.5,
            ears: 0.5,
            neck: 0.5,
            hair: 0.5,
        }
    }
}

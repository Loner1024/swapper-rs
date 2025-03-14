use crate::utils::process_img::{preprocess_image, reinhard_color_transfer};
use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{imageops, DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array2, ArrayViewD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// 人脸替换器结构体
pub struct InSwapper {
    model: Session,
}

impl InSwapper {
    /// 创建新的人脸替换器实例
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {

        // 创建推理会话
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)
            .context("Failed to create session from model file")?;

        Ok(Self { model })
    }

    /// 执行人脸替换
    pub fn swap_face(&mut self, target_img: &DynamicImage, source_face_recognition: Array2<f32>,) -> Result<DynamicImage> {
        // 预处理目标图像
        let target_tensor = Tensor::from_array(preprocess_image(target_img, 128)?)?;

        // 从源图像提取人脸嵌入向量
        let source_tensor = Tensor::from_array(source_face_recognition)?;

        // 准备输入
        let inputs = ort::inputs![
            "target" => target_tensor,
            "source" => source_tensor,
        ];

        // 运行推理
        let outputs = self.model.run(inputs)
            .context("Failed to run inference")?;

        // 获取输出
        let output = outputs["output"].try_extract_tensor::<f32>()?;

        // 后处理输出
        postprocess_output(output, target_img, (target_img.height(), target_img.width()))
    }

}

/// 将模型输出转换回图像
fn postprocess_output(
    output: ArrayViewD<f32>,
    target_image: &DynamicImage, // 色彩参考图
    original_size: (u32, u32),
) -> Result<DynamicImage> {
    // 1. 计算动态范围（可缓存 min/max 避免重复计算）
    let min_val = output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    // 2. 安全处理除零错误（如果 range ≈ 0）
    let range = if range.abs() < 1e-6 { 1.0 } else { range };

    // 3. 创建图像缓冲区
    let (_, _, h, w) = match output.shape() {
        &[1, 3, h, w] => (1, 3, h, w),
        _ => panic!("Unexpected shape: {:?}", output.shape()),
    };
    let mut img_buffer: RgbImage = ImageBuffer::new(w as u32, h as u32);

    // 4. 动态反归一化到 [0, 255]
    for y in 0..h {
        for x in 0..w {
            let r = ((output[[0, 0, y, x]] - min_val) / range * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let g = ((output[[0, 1, y, x]] - min_val) / range * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let b = ((output[[0, 2, y, x]] - min_val) / range * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    let corrected_img = reinhard_color_transfer(&img_buffer.into(), target_image)?;
    let resized = imageops::resize(
        &corrected_img,
        original_size.0,
        original_size.1,
        FilterType::Triangle,
    );
    Ok(DynamicImage::ImageRgb8(resized))
}

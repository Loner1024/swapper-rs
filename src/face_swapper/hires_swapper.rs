use crate::log;
use crate::utils::process_img::{grayscale_tensor_as_image, preprocess_image_with_padding_square, reinhard_color_transfer};
use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{imageops, DynamicImage, GrayImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array2, ArrayViewD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, TensorRef};

pub struct HiresSwapper {
    model: Session,
}

impl HiresSwapper {
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;
        Ok(Self { model })
    }

    // 人脸交换
    pub fn swap_face(
        &mut self,
        target_img: &mut DynamicImage,
        source_face_recognition: Array2<f32>,
    ) -> Result<(DynamicImage, GrayImage)> {
        log!("执行人脸交换");
        // 准备网络输入
        let source_tensor = Tensor::from_array(source_face_recognition).context("张量转换失败")?;
        let (target_image_data, _) = preprocess_image_with_padding_square(&target_img, 256)?;
        let target_tensor: TensorRef<f32> = TensorRef::from_array_view(target_image_data.view())?;

        // 创建输入张量
        let inputs = ort::inputs! {
            "target" => target_tensor.view(),
            "vsid" => source_tensor.view(),
        };

        // 运行推理
        let outputs = self.model.run(inputs)?;
        // 获取输出结果
        let output = outputs.get("output").context("未找到换脸 output 输出")?;
        let mask = outputs.get("mask").context("未找到换脸 mask 输出")?;

        let output_tensor = output.try_extract_tensor::<f32>()?;
        let mask_tensor = mask.try_extract_tensor::<f32>()?;

        let out_image = postprocess_output(
            output_tensor,
            target_img,
            (target_img.height(), target_img.width()),
        )?;
        let mask_image =
            grayscale_tensor_as_image(mask_tensor, (target_img.height(), target_img.width()))?;
        Ok((out_image, mask_image))
    }
}

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


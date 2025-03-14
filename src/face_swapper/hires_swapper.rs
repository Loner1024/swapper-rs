use crate::log;
use crate::utils::process_img::{grayscale_tensor_as_image, preprocess_image_with_padding_square};
use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{imageops, DynamicImage, GrayImage, ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::Canvas;
use ndarray::{array, Array1, Array2, ArrayViewD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, TensorRef};
use palette::{FromColor, Lab, Srgb, Srgba};
use std::marker::PhantomData;

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

// Reinhard 颜色迁移核心逻辑
pub fn reinhard_color_transfer(
    src_img: &DynamicImage,
    target_img: &DynamicImage,
) -> Result<RgbImage> {
    // 统一图像尺寸 (假设模型输出需要对齐到目标尺寸)
    let target_size = target_img.dimensions();
    let resized_src = src_img.resize_exact(target_size.0, target_size.1, image::imageops::Triangle);

    // 转换到 LAB 颜色空间
    let (src_lab, target_lab) = convert_to_lab(&resized_src, target_img)?;

    // 计算统计量
    let src_stats = compute_lab_stats(&src_lab);
    let target_stats = compute_lab_stats(&target_lab);

    // 调整颜色分布
    let adjusted_lab = adjust_lab_channels(&src_lab, &src_stats, &target_stats);

    // 转回 RGB 并生成图像
    lab_to_rgb_image(adjusted_lab, target_size)
}

// RGB 转 LAB 颜色空间 (使用 palette 库)
fn convert_to_lab(src: &DynamicImage, target: &DynamicImage) -> Result<(Vec<Lab>, Vec<Lab>)> {
    let src_lab: Vec<Lab> = src
        .to_rgb32f()
        .pixels()
        .map(|p| {
            let rgb = Srgb::new(p.0[0], p.0[1], p.0[2]).into_linear();
            Lab::from_color(rgb)
        })
        .collect();

    let target_lab: Vec<Lab> = target
        .to_rgb32f()
        .pixels()
        .map(|p| {
            let rgb = Srgb::new(p.0[0], p.0[1], p.0[2]).into_linear();
            Lab::from_color(rgb)
        })
        .collect();

    Ok((src_lab, target_lab))
}

// 计算 LAB 各通道均值和标准差
fn compute_lab_stats(lab_pixels: &[Lab]) -> [Array1<f32>; 3] {
    let mut l_channel = Vec::new();
    let mut a_channel = Vec::new();
    let mut b_channel = Vec::new();

    for lab in lab_pixels {
        l_channel.push(lab.l);
        a_channel.push(lab.a);
        b_channel.push(lab.b);
    }

    [
        compute_mean_std(&l_channel),
        compute_mean_std(&a_channel),
        compute_mean_std(&b_channel),
    ]
}

// 均值和标准差计算
fn compute_mean_std(data: &[f32]) -> Array1<f32> {
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;

    let variance: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;

    array![mean, variance.sqrt()]
}

// 调整 LAB 通道
fn adjust_lab_channels(
    src_lab: &[Lab],
    src_stats: &[Array1<f32>; 3],
    target_stats: &[Array1<f32>; 3],
) -> Vec<Lab> {
    let data: PhantomData<palette::white_point::D65> = Default::default();
    src_lab
        .iter()
        .map(|lab| {
            let l = adjust_channel(lab.l, &src_stats[0], &target_stats[0]);
            let a = adjust_channel(lab.a, &src_stats[1], &target_stats[1]);
            let b = adjust_channel(lab.b, &src_stats[2], &target_stats[2]);
            Lab {
                l,
                a,
                b,
                white_point: data,
            }
        })
        .collect()
}

// 单通道调整公式
fn adjust_channel(val: f32, src_stat: &Array1<f32>, target_stat: &Array1<f32>) -> f32 {
    let eps = 1e-6; // 防止除零
    let src_mean = src_stat[0];
    let src_std = src_stat[1].max(eps);
    let target_mean = target_stat[0];
    let target_std = target_stat[1].max(eps);

    ((val - src_mean) / src_std) * target_std + target_mean
}

// LAB 转 RGB 并生成图像
fn lab_to_rgb_image(lab_pixels: Vec<Lab>, size: (u32, u32)) -> Result<RgbImage> {
    let mut img = RgbImage::new(size.0, size.1);

    for (i, lab) in lab_pixels.into_iter().enumerate() {
        let x = (i % size.0 as usize) as u32;
        let y = (i / size.0 as usize) as u32;

        // 转换到 sRGB 并限制范围
        let rgb: Srgba<f32> = Srgba::from_color(lab).into_format();
        let r = (rgb.red * 255.0).round().clamp(0.0, 255.0) as u8;
        let g = (rgb.green * 255.0).round().clamp(0.0, 255.0) as u8;
        let b = (rgb.blue * 255.0).round().clamp(0.0, 255.0) as u8;
        img.put_pixel(x, y, Rgb([r, g, b]));
    }

    Ok(img)
}

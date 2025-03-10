use crate::frame::ProcessFrame;
use anyhow::{Context, Result};
use frame::enhance::FaceEnhancer;
use ort::execution_providers::CoreMLExecutionProvider;
use std::path::Path;

mod frame;
mod face_parsing;
mod utils;

fn main() -> Result<()> {
    // 初始化ONNX Runtime
    ort::init()
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;

    let input_path = "input.png";
    let output_path = "output.png";
    // 加载原始图像
    let original_img = image::open(input_path).context("无法打开输入图片")?;

    // 初始化人脸修复器
    let mut restorer = FaceEnhancer::new(
        Path::new("./models/GFPGANv1.4.onnx"),
        Path::new("./models/face_parsing.onnx"),
        "./haarcascade_frontalface_alt.xml",
    )?;
    // 处理图像
    restorer.process_image(&original_img)?.save(output_path)?;

    Ok(())
}

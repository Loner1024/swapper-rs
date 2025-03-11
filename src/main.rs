use std::path::Path;
use anyhow::{Context, Result};
use ort::execution_providers::CoreMLExecutionProvider;
use swapper_rs::face_swapper::face_swapper::FaceSwapper;
use swapper_rs::frame::enhance::FaceEnhancer;
use swapper_rs::frame::ProcessFrame;
use swapper_rs::post_processor::post_processor::PostProcessor;
use swapper_rs::pre_processor::pre_processor::PreProcessor;

const FACE_SWAP_MODEL_PATH: &str = "./models/9O_865k.onnx";

fn main() -> Result<()> {
    // 初始化ONNX Runtime
    ort::init()
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;

    let source_path = "target.png";
    let target_path = "source.png";
    // 加载原始图像
    let mut source_img = image::open(source_path).context("无法打开输入图片")?;
    let mut target_img = image::open(target_path).context("无法打开输入图片")?;

    let mut pre_processor = PreProcessor::new()?;
    let pre_process_result = pre_processor.process(&mut source_img, &mut target_img)?;

    let mut face_swapper = FaceSwapper::new(FACE_SWAP_MODEL_PATH)?;

    let swaped_face =  face_swapper.swap_face(&mut pre_process_result.target_face.clone(), pre_process_result.face_recognition_source.clone())?;

    // 初始化人脸修复器
    let mut restorer = FaceEnhancer::new(
        Path::new("./models/GFPGANv1.4.onnx"),
    )?;
    // 处理图像
    let output_path = "result.png";
    let swaped_face = restorer.process_image(&swaped_face)?;

    let mut post_processor = PostProcessor::new(&mut pre_processor.parsing_session, pre_process_result.clone());
    post_processor.process(&target_img, &swaped_face)?.save(output_path)?;

    Ok(())
}

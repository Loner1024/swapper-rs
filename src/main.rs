use anyhow::{Context, Result};
use ort::execution_providers::CoreMLExecutionProvider;
use std::time::Instant;
use swapper_rs::face_processor::face_xseg::FaceXseger;
use swapper_rs::face_swapper::face_swapper::FaceSwapper;
use swapper_rs::frame::enhance::FaceEnhancer;
use swapper_rs::post_processor::post_processor::PostProcessor;
use swapper_rs::pre_processor::pre_processor::PreProcessor;

const FACE_SWAP_MODEL_PATH: &str = "./models/9O_865k.onnx";
const FACE_ENHANCE_MODEL_PATH: &str = "./models/GFPGANv1.4.onnx";
const FACE_XSEG_MODEL_PATH: &str = "./models/xseg.onnx";
fn main() -> Result<()> {
    let now = Instant::now();
    // 初始化ONNX Runtime
    ort::init()
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;

    let source_path = "./images/source.png";
    let target_path = "./images/target.png";
    // 加载原始图像
    let mut source_img = image::open(source_path).context("无法打开输入图片")?;
    let mut target_img = image::open(target_path).context("无法打开输入图片")?;

    let mut pre_processor = PreProcessor::new()?;
    let pre_process_result = pre_processor.process(&mut source_img, &mut target_img)?;
    println!("pre_processing in {:?}s", now.elapsed());

    let mut face_swapper = FaceSwapper::new(FACE_SWAP_MODEL_PATH)?;
    let (face, _) = face_swapper.swap_face(
        &mut pre_process_result.target_face.clone(),
        pre_process_result.face_recognition_source.clone(),
    )?;
    let mut xseger = FaceXseger::new(FACE_XSEG_MODEL_PATH)?;
    let xseg_mask = xseger.process_image(&face)?;
    println!("swap face in {:?}s", now.elapsed());

    // 初始化人脸修复器
    let mut restorer = FaceEnhancer::new(FACE_ENHANCE_MODEL_PATH)?;
    // 处理图像
    let swaped_face = restorer.enhance_faces(&face)?;
    println!("enhance face in {:?}s", now.elapsed());

    let mut post_processor = PostProcessor::new(
        &mut pre_processor.parsing_session,
        pre_process_result.clone(),
    );
    println!("post_processing face in {:?}s", now.elapsed());

    let output_path = "./images/result.png";
    post_processor
        .process(&target_img, &swaped_face, &xseg_mask)?
        .save(output_path)?;
    println!("Total time taken: {}s", now.elapsed().as_secs());

    Ok(())
}

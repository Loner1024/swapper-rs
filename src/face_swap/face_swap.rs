use std::time::Instant;
use anyhow::Context;
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array2, ArrayD};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, TensorRef};
use crate::log;
use crate::utils::process_img::preprocess_image;

pub struct FaceSwap {
    model: Session,
}

impl FaceSwap {
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self { model })
    }

    // 人脸交换
    pub fn swap_face(
        &mut self,
        target_img: &mut DynamicImage,
        source_face_recognition: Array2<f32>,
    ) -> anyhow::Result<(ArrayD<f32>, ArrayD<f32>)> {
        let start = Instant::now();
        log!("执行人脸交换");

        // 准备目标人脸图像
        let target_face =
            target_img
                .resize(256, 256, FilterType::Lanczos3);

        // 准备网络输入
        let source_tensor = Tensor::from_array(source_face_recognition)
            .context("张量转换失败")?;
        let target_image_data = preprocess_image(&target_face, (256, 256))?;
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

        log!("人脸交换完成，耗时: {:?}", start.elapsed());

        Ok((output_tensor.to_owned(), mask_tensor.to_owned()))
    }
}
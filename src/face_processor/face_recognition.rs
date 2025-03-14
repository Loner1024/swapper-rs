use crate::utils::process_img::{l2_normalize, normalize_image};
use anyhow::Result;
use image::DynamicImage;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::PathBuf;

pub struct FaceRecognition {
    model: Session,
}

impl FaceRecognition {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;
        Ok(Self { model })
    }

    pub fn recognition(&mut self, face_image: &DynamicImage) -> Result<Array2<f32>> {
        let face_image = normalize_image(face_image, (112, 112));
        let input_name = self.model.inputs[0].name.clone();
        let output_name = self.model.outputs[0].name.clone();

        // 准备输入张量 (shape: [1, 3, 112, 112])
        let input_tensor = TensorRef::from_array_view(face_image.view())?;

        // 执行推理
        let outputs = self.model.run(ort::inputs![input_name => input_tensor])?;
        let face_recognition_data = outputs[output_name].try_extract_tensor::<f32>()?;
        let normalized_feature = l2_normalize(face_recognition_data.into_owned());

        Ok(normalized_feature.into_dimensionality()?.to_owned())
    }
}

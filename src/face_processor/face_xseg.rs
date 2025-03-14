use crate::utils::process_img::grayscale_tensor_as_image;
use anyhow::Result;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use ndarray::{Array, ArrayBase, ArrayD, Dim, Ix, Ix4, OwnedRepr};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

pub struct FaceXseger {
    model: Session,
}

impl FaceXseger {
    /// 创建新的人脸分割器实例
    pub fn new(model_path: &str) -> Result<Self> {
        // 加载模型
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;

        Ok(FaceXseger { model })
    }

    /// 处理单张图像并返回掩码
    pub fn process_image(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        // 加载并预处理图像
        let img_array = self.preprocess_image(img)?;
        let input_tensor = TensorRef::from_array_view(img_array.view())?;

        // 获取输入名称
        let input_name = self.model.inputs[0].name.clone();
        let output_name = self.model.outputs[0].name.clone();

        // 运行推理
        let outputs = self.model.run(ort::inputs![input_name => input_tensor])?;
        let mask = outputs[output_name]
            .try_extract_tensor::<f32>()?
            .into_owned();
        let reshaped: ArrayD<f32> = mask
            .into_dimensionality::<Ix4>()? // 确保是4维数组
            .into_shape_with_order([1, 1, 256, 256])? // 重塑具体尺寸
            .into_dyn(); // 再转回动态维度 ArrayD
        let mask_img = grayscale_tensor_as_image(reshaped.view(), (img.width(), img.width()))?;
        Ok(mask_img)
    }

    // 预处理图像为模型输入格式
    fn preprocess_image(
        &self,
        img: &DynamicImage,
    ) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[Ix; 4]>>> {
        // 调整图像大小为256x256
        let img_resized = img.resize(256, 256, FilterType::Lanczos3);

        let mut array = Array::zeros((1, 256, 256, 3));

        // 假设模型需要归一化到[0,1]的RGB输入
        for (x, y, pixel) in img_resized.to_rgb8().enumerate_pixels() {
            array[[0, y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0; // R
            array[[0, y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0; // G
            array[[0, y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0; // B
        }

        Ok(array)
    }
}

use crate::frame::ProcessFrame;
use anyhow::{Context, Result};
use image::{
    imageops, imageops::FilterType, DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage,
};
use ndarray::{Array, ArrayBase, Dim, Ix, OwnedRepr};
use opencv::core::{Mat_, Vector};
use opencv::objdetect::CascadeClassifier;
use opencv::{
    core::{Mat, Point, Rect, Size},
    objdetect,
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;

const FACE_SIZE: u32 = 512;

pub struct FaceEnhancer {
    gfpgan: Session,
    face_parser: Session,
    face_detector: CascadeClassifier,
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
            let r = (output[[0, 0, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
            let g = (output[[0, 1, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
            let b = (output[[0, 2, y, x]].clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0;
            img_buffer.put_pixel(x as u32, y as u32, Rgb([r as u8, g as u8, b as u8]));
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
    fn process_image(&mut self, original_img: &DynamicImage) -> Result<RgbImage> {
        // 人脸检测和对齐
        let faces = self.detect_and_align_faces(&original_img)?;

        // 处理每个人脸
        let mut final_img = original_img.to_rgb8();
        for (face_img, (x, y, w, h)) in faces {
            // 分块增强
            let restored = self.enhance_faces(&face_img)?;
            // 生成人脸mask
            let mask = self.generate_face_mask(&face_img)?;
            // 调整增强后的图像到实际尺寸
            let restored_resized = restored.resize_exact(w, h, FilterType::Lanczos3);

            // 调整掩码尺寸
            let mask_resized = imageops::resize(&mask, w, h, FilterType::Lanczos3);

            // 融合到原图
            self.blend_face(&mut final_img, &restored_resized, &mask_resized, x, y);

        }
        Ok(final_img)
    }
}

impl FaceEnhancer {
    pub fn new(model_path: &Path, parsing_model_path: &Path, cascade_path: &str) -> Result<Self> {
        // 初始化ONNX模型
        let gfpgan = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let face_parser = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(parsing_model_path)?;

        // 初始化OpenCV人脸检测器
        let face_detector = CascadeClassifier::new(cascade_path).context("无法加载人脸检测模型")?;

        Ok(Self {
            gfpgan,
            face_parser,
            face_detector,
        })
    }

    pub fn enhance_faces(&mut self, img: &DynamicImage) -> Result<DynamicImage> {
        let (input_array, original_size) = preprocess_image(&img)?;

        // 准备输入张量 (shape: [1, 3, 512, 512])
        let input_tensor = TensorRef::from_array_view(input_array.view())?;

        // 执行推理
        let outputs = self.gfpgan.run(ort::inputs!["input" => input_tensor])?;

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

    fn detect_and_align_faces(
        &mut self,
        img: &DynamicImage,
    ) -> Result<Vec<(DynamicImage, (i32, i32, u32, u32))>> {
        // 1. 转换图像到OpenCV兼容格式
        let gray_img = img.to_luma8();
        let src_mat = Mat::from_slice(gray_img.as_raw())?;

        // 2. 重塑矩阵维度（返回BoxedRef<Mat>）
        let reshaped = src_mat.reshape(1, gray_img.height() as i32)?;

        // 3. 通过解引用获取底层Mat引用
        let mat_ref: &Mat = &reshaped.clone_pointee();

        // 4. 将Mat引用转换为拥有所有权的Mat
        let owned_mat = mat_ref.clone(); // 这里需要Mat实现Clone trait

        // 5. 转换为具体类型Mat_<u8>
        let ocv_mat: Mat_<u8> = owned_mat.try_into_typed()?;

        // 5. 执行人脸检测
        let mut faces = Vector::<Rect>::new();
        self.face_detector.detect_multi_scale(
            &ocv_mat,
            &mut faces,
            1.05, // scale factor
            3,    // min neighbors
            objdetect::CASCADE_SCALE_IMAGE,
            Size::new(30, 30), // min size
            Size::new(0, 0),   // max size (0表示无限制)
        )?;

        // 6. 处理检测结果
        let mut results = Vec::new();
        for face in faces {
            let expand_factor = 0.2;
            let expanded_width = (face.width as f32 * (1.0 + expand_factor)).round() as i32;
            let expanded_height = (face.height as f32 * (1.0 + expand_factor)).round() as i32;

            let center = Point::new(face.x + face.width / 2, face.y + face.height / 2);

            let x_start = (center.x - expanded_width / 2).max(0);
            let y_start = (center.y - expanded_height / 2).max(0);
            let width = expanded_width.min(ocv_mat.cols() - x_start);
            let height = expanded_height.min(ocv_mat.rows() - y_start);

            // 转换为图像坐标系
            let crop_x = x_start as u32;
            let crop_y = y_start as u32;
            let crop_width = width as u32;
            let crop_height = height as u32;

            let cropped = img.crop_imm(crop_x, crop_y, crop_width, crop_height);
            let aligned = cropped.resize_exact(512, 512, FilterType::Lanczos3);

            // 保存坐标和实际尺寸
            results.push((aligned, (x_start, y_start, width as u32, height as u32)));
        }

        // 调试输出
        println!("检测到 {} 个人脸", results.len());

        Ok(results)
    }

    /// 生成人脸mask
    fn generate_face_mask(&mut self, face_img: &DynamicImage) -> Result<GrayImage> {
        // 预处理（与人脸解析模型匹配）
        let input_tensor = preprocess_parser(&face_img); // 需要单独的预处理函数
        let input_tensor = TensorRef::from_array_view(input_tensor.view())?;

        // 推理
        let outputs = self
            .face_parser
            .run(ort::inputs!["input" => input_tensor])?;

        let mask_array = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("无法提取mask张量")?;

        // 后处理
        let mut mask = GrayImage::new(FACE_SIZE, FACE_SIZE);
        for y in 0..FACE_SIZE {
            for x in 0..FACE_SIZE {
                // 假设模型输出多通道分割结果：
                // 通道0: 背景, 通道1: 皮肤, 通道2: 眉毛, 通道3: 眼睛等...
                let is_face =
                    mask_array[[0, 1, y as usize, x as usize]] > 0.5 || // 皮肤
                        mask_array[[0, 2, y as usize, x as usize]] > 0.5 || // 鼻子
                        mask_array[[0, 4, y as usize, x as usize]] > 0.5 || // 左眼
                        mask_array[[0, 5, y as usize, x as usize]] > 0.5 || // 右眼
                        mask_array[[0, 6, y as usize, x as usize]] > 0.5 || // 左眉毛
                        mask_array[[0, 7, y as usize, x as usize]] > 0.5 || // 右眉毛
                        mask_array[[0, 10, y as usize, x as usize]] > 0.5 || // 嘴部
                        mask_array[[0, 11, y as usize, x as usize]] > 0.5 || // 上嘴唇
                        mask_array[[0, 12, y as usize, x as usize]] > 0.5;   // 下嘴唇
                let value = if is_face { 255 } else { 0 };
                mask.put_pixel(x, y, Luma([value]));
            }
        }
        mask.save("mask.png").unwrap();

        Ok(mask)
    }

    /// 融合处理结果到原图
    fn blend_face(
        &self,
        background: &mut RgbImage,
        face: &DynamicImage,
        mask: &GrayImage,
        x: i32,
        y: i32,
    ) {
        let face_rgb = face.to_rgb8();
        let (width, height) = (face_rgb.width(), face_rgb.height());

        // 确保掩码尺寸匹配
        assert_eq!(mask.width(), width, "掩码宽度不匹配");
        assert_eq!(mask.height(), height, "掩码高度不匹配");

        for fy in 0..height {
            for fx in 0..width {
                let alpha = mask.get_pixel(fx, fy)[0] as f32 / 255.0;
                let bg_x = x as u32 + fx;
                let bg_y = y as u32 + fy;

                // 边界检查
                if bg_x >= background.width() || bg_y >= background.height() {
                    continue;
                }

                let face_pixel = face_rgb.get_pixel(fx, fy);
                let bg_pixel = background.get_pixel_mut(bg_x, bg_y);

                // 线性混合
                bg_pixel.0 = [
                    (face_pixel[0] as f32 * alpha + bg_pixel[0] as f32 * (1.0 - alpha)) as u8,
                    (face_pixel[1] as f32 * alpha + bg_pixel[1] as f32 * (1.0 - alpha)) as u8,
                    (face_pixel[2] as f32 * alpha + bg_pixel[2] as f32 * (1.0 - alpha)) as u8,
                ];
            }
        }
    }
}

/// 人脸解析模型的专用预处理
fn preprocess_parser(img: &DynamicImage) -> ArrayBase<OwnedRepr<f32>, Dim<[Ix; 4]>> {
    let mut array = Array::zeros((1, 3, FACE_SIZE as usize, FACE_SIZE as usize));

    // 假设模型需要归一化到[0,1]的RGB输入
    for (x, y, pixel) in img.to_rgb8().enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0; // R
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; // G
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0; // B
    }
    array
}

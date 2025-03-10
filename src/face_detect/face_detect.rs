use anyhow::Context;
use opencv::objdetect::CascadeClassifier;
use anyhow::Result;
use image::DynamicImage;
use image::imageops::FilterType;
use opencv::core::{Mat, MatTraitConstManual, Mat_, Point, Rect, Size, Vector};
use opencv::objdetect;
use opencv::prelude::{CascadeClassifierTrait, MatTraitConst};

pub struct FaceDetector {
    dector: CascadeClassifier
}

impl FaceDetector {
    pub fn new(cascade_path: &str) -> Result<Self> {
        let face_detector = CascadeClassifier::new(cascade_path).context("无法加载人脸检测模型")?;
        Ok(Self { dector: face_detector })
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<(DynamicImage, (i32, i32, u32, u32))>> {
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
        self.dector.detect_multi_scale(
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
}
use crate::face_processor::face_align::align_and_normalize_face;
use crate::face_processor::face_parsing::{FaceParsing, FaceRawData};
use crate::face_processor::face_recognition::FaceRecognition;
use crate::face_processor::yolo_face_detector::YOLOFaceDetector;
use crate::utils::process_img::{
    blend_mask, generate_face_mask, pad_to_square_with_size, BoxDetection,
};
use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::Array2;

#[derive(Clone)]
pub struct PreProcessResult {
    pub face_recognition_source: Array2<f32>,
    pub source_face: DynamicImage,
    pub target_face: DynamicImage,
    pub source_rotation_angle: f32,
    pub target_rotation_angle: f32,
    pub target_rect: BoxDetection,
}

pub struct PreProcessor {
    recognition_session: FaceRecognition,
    pub parsing_session: FaceParsing,
    face_detector: YOLOFaceDetector,
}

// 模型路径常量
const FACE_RECOGNITION_MODEL_PATH: &str = "./models/curricularface.onnx";
const PARSING_MODEL_PATH: &str = "./models/face_parsing.onnx";

impl PreProcessor {
    pub fn new() -> anyhow::Result<Self> {
        // 构建模型会话
        let parsing_session = FaceParsing::new(PARSING_MODEL_PATH.into())?;
        let recognition_session = FaceRecognition::new(FACE_RECOGNITION_MODEL_PATH.into())?;
        let face_detector = YOLOFaceDetector::new("./models/yoloface_8n.onnx", 0.5, 0.4)?;
        Ok(Self {
            recognition_session,
            parsing_session,
            face_detector,
        })
    }

    pub fn process(
        &mut self,
        source_img: &mut DynamicImage,
        target_img: &mut DynamicImage,
    ) -> Result<PreProcessResult> {
        // 1. 在源图像中检测人脸
        let source_face_rect = self
            .detect_face(source_img)?
            .context("在源图像中未检测到人脸")?;

        // 2. 在目标图像中检测人脸
        let target_face_rect = self
            .detect_face(target_img)?
            .context("在目标图像中未检测到人脸")?;

        // 3. 获取源图像人脸解析结果
        let source_parsing_result = self.parse_face(source_img, source_face_rect)?;
        let target_parsing_result = self.parse_face(target_img, target_face_rect)?;

        // 4. 人脸对齐
        let (align_source, source_rotation_angle) =
            self.align_face(source_img, source_parsing_result, source_face_rect)?;
        let (align_target, target_rotation_angle) =
            self.align_face(target_img, target_parsing_result, target_face_rect)?;

        // 5. 人脸特征识别
        let face_recognition_source = self.recognition_session.recognition(&align_source)?;

        let result = PreProcessResult {
            face_recognition_source,
            source_face: align_source,
            target_face: align_target,
            source_rotation_angle,
            target_rotation_angle,
            target_rect: target_face_rect,
        };
        Ok(result)
    }

    // 人脸对齐
    fn align_face(
        &mut self,
        face_img: &mut DynamicImage,
        face_raw_data: FaceRawData,
        face_rect: BoxDetection,
    ) -> Result<(DynamicImage, f32)> {
        let face_img = face_img.crop(
            face_rect.x as u32,
            face_rect.y as u32,
            face_rect.width as u32,
            face_rect.height as u32,
        );
        let (face_img, _) = pad_to_square_with_size(&face_img, 512, 0.0, None);
        let face_mask = generate_face_mask(face_raw_data.clone(), &face_img)?;
        // let face_with_mask_blend = blend_mask(&face_img, &face_mask)?;

        align_and_normalize_face(&face_img, face_raw_data)
    }

    // 人脸解析（特征提取）
    fn parse_face(
        &mut self,
        img: &mut DynamicImage,
        face_rect: BoxDetection,
    ) -> Result<FaceRawData> {
        // 裁剪人脸区域
        let face_img = img.crop(
            face_rect.x as u32,
            face_rect.y as u32,
            face_rect.width as u32,
            face_rect.height as u32,
        );
        // 推理
        let face_raw_data = self.parsing_session.parse(&face_img)?;

        Ok(face_raw_data)
    }

    fn detect_face(&mut self, img: &DynamicImage) -> Result<Option<BoxDetection>> {
        let results = self.face_detector.detect(img)?;
        if results.is_empty() {
            Ok(None)
        } else {
            let result = results[0];
            Ok(Some(result))
        }
    }
}

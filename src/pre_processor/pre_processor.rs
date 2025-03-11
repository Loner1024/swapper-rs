use anyhow::{Context, Result};
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::Array2;
use crate::face_processor::face_align::align_and_normalize_face;
use crate::face_processor::face_detect::FaceDetector;
use crate::face_processor::face_parsing::{FaceParsing, FaceRawData};
use crate::face_processor::face_recognition::FaceRecognition;

#[derive(Clone)]
pub struct PreProcessResult {
    pub face_recognition_source: Array2<f32>,
    pub source_face: DynamicImage,
    pub target_face: DynamicImage,
    pub source_rotation_angle: f32,
    pub target_rotation_angle: f32,
    pub target_rect: (u32, u32, u32, u32) // x1, x2, y1 y2
}

pub struct PreProcessor {
    recognition_session: FaceRecognition,
    pub parsing_session: FaceParsing,
    face_detector: FaceDetector,
}

// 模型路径常量
const FACE_RECOGNITION_MODEL_PATH: &str = "./models/curricularface.onnx";
const PARSING_MODEL_PATH: &str = "./models/face_parsing.onnx";

impl PreProcessor {
    pub fn new() -> anyhow::Result<Self> {
        // 构建模型会话
        let parsing_session = FaceParsing::new(PARSING_MODEL_PATH.into())?;
        let recognition_session = FaceRecognition::new(FACE_RECOGNITION_MODEL_PATH.into())?;
        let face_detector = FaceDetector::new("./models/opencv/haarcascade_frontalface_alt2.xml")?;
        Ok(Self {
            recognition_session,
            parsing_session,
            face_detector,
        })
    }

    pub fn process(
        &mut self,
        source_img: &mut DynamicImage,
        target_img: &mut DynamicImage
    ) -> Result<PreProcessResult> {
        // 1. 在源图像中检测人脸
        let source_face_rect = self
            .detect_face(source_img)?
            .context("在源图像中未检测到人脸")?;

        // 2. 在目标图像中检测人脸
        let target_face_rect = self.
            detect_face(target_img)?
            .context("在目标图像中未检测到人脸")?;
        // let (target_face_height, target_face_width) = (target_face_rect.3-target_face_rect.2, target_face_rect.1-target_face_rect.0);

        // 3. 获取源图像人脸解析结果
        let source_parsing_result = self.parse_face(source_img, source_face_rect)?;
        let target_parsing_result = self.parse_face(target_img, target_face_rect)?;

        // 4. 人脸对齐
        let (align_source, source_rotation_angle) = self.align_face(source_img, source_parsing_result, source_face_rect)?;
        let (align_target, target_rotation_angle)= self.align_face(target_img, target_parsing_result, target_face_rect)?;

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
        face_rect: (u32, u32, u32, u32),
    ) -> Result<(DynamicImage, f32)> {
        let (x1, x2, y1, y2) = face_rect;
        let face_img = face_img.crop(x1, y1, x2 - x1, y2 - y1).resize(512, 512, FilterType::Lanczos3);
        align_and_normalize_face(&face_img, face_raw_data)
    }

    // 人脸解析（特征提取）
    fn parse_face(
        &mut self,
        img: &mut DynamicImage,
        face_rect: (u32, u32, u32, u32),
    ) -> Result<FaceRawData> {
        // 裁剪人脸区域
        let (x1, x2, y1, y2) = face_rect;
        let face_img = img.crop(x1, y1, x2 - x1, y2 - y1);
        // 推理
        let face_raw_data = self.parsing_session.parse(&face_img)?;

        Ok(face_raw_data)
    }

    fn detect_face(&mut self, img: &DynamicImage) -> anyhow::Result<Option<(u32, u32, u32, u32)>> {
        let results = self.face_detector.detect(img)?;
        if results.is_empty() {
            Ok(None)
        } else {
            let result = results[0].clone().1;
            Ok(Some((
                result.0,
                result.0 + result.2,
                result.1,
                result.1 + result.3,
            )))
        }
    }
}

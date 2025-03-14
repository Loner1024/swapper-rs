use crate::face_processor::face_align::align_and_normalize_face;
use crate::face_processor::face_parsing::{FaceParsing, FaceRawData};
use crate::face_processor::face_recognition::FaceRecognition;
use crate::face_processor::yolo_face_detector::YOLOFaceDetector;
use crate::utils::process_img::BoxDetection;
use anyhow::{Context, Result};
use image::imageops::FilterType;
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
        origin_img: &mut DynamicImage,
        face_raw_data: FaceRawData,
        face_rect: BoxDetection,
    ) -> Result<(DynamicImage, f32)> {
        let face_img = origin_img.crop(
            face_rect.x as u32,
            face_rect.y as u32,
            face_rect.width as u32,
            face_rect.height as u32,
        );
        let face_img = face_img.resize(512, 512, FilterType::Lanczos3);

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

    fn detect_face(&mut self, img: &mut DynamicImage) -> Result<Option<BoxDetection>> {
        let results = self.face_detector.detect(img)?;
        if results.is_empty() {
            Ok(None)
        } else {
            let face_rect = results[0];
            let size = face_rect.width.max(face_rect.height);
            let x = face_rect.x - (size - face_rect.width) / 2.0;
            let y = face_rect.y - (size - face_rect.height) / 2.0;
            Ok(Some(BoxDetection {
                x,
                y,
                width: size,
                height: size,
                conf: face_rect.conf,
            }))
        }
    }
}

use anyhow::{Context, Result};
use image::DynamicImage;
use imageproc::point::Point;
use ndarray::ArrayD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::PathBuf;

/// 模型输出通道元数据
#[derive(Debug, Clone)]
pub struct FaceChannelMeta {
    pub index: usize,
    pub name: &'static str,
    pub description: &'static str,
}

/// 原始人脸解析数据容器
#[derive(Debug, Clone)]
pub struct FaceRawData {
    /// 原始 4D 张量 [batch=0, channels, height, width]
    pub tensor: ArrayD<f32>,

    #[allow(unused)]
    /// 通道元数据列表（按模型输出顺序）
    channels: Vec<FaceChannelMeta>,
}

/// 对齐关键点
#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    pub left_eye: Point<f32>,
    pub right_eye: Point<f32>,
    pub nose: Point<f32>,
    pub mouth: Point<f32>,
}

impl FaceRawData {
    pub fn new(tensor: ArrayD<f32>) -> Result<Self> {
        // 验证张量维度
        let shape = tensor.shape();
        anyhow::ensure!(
            shape.len() == 4 && shape[0] == 1,
            "Invalid tensor shape: expected [1, 19, H, W], got {:?}",
            shape
        );

        Ok(Self {
            tensor,
            channels: vec![
                FaceChannelMeta {
                    index: 0,
                    name: "background",
                    description: "背景",
                },
                FaceChannelMeta {
                    index: 1,
                    name: "skin",
                    description: "皮肤区域",
                },
                FaceChannelMeta {
                    index: 2,
                    name: "nose",
                    description: "鼻子",
                },
                FaceChannelMeta {
                    index: 3,
                    name: "eye_g",
                    description: "眼镜",
                },
                FaceChannelMeta {
                    index: 4,
                    name: "l_eye",
                    description: "左眼",
                },
                FaceChannelMeta {
                    index: 5,
                    name: "r_eye",
                    description: "右眼",
                },
                FaceChannelMeta {
                    index: 6,
                    name: "l_brow",
                    description: "左眉",
                },
                FaceChannelMeta {
                    index: 7,
                    name: "r_brow",
                    description: "右眉",
                },
                FaceChannelMeta {
                    index: 8,
                    name: "l_ear",
                    description: "左耳",
                },
                FaceChannelMeta {
                    index: 9,
                    name: "r_ear",
                    description: "右耳",
                },
                FaceChannelMeta {
                    index: 10,
                    name: "mouth",
                    description: "嘴部区域",
                },
                FaceChannelMeta {
                    index: 11,
                    name: "u_lip",
                    description: "上唇",
                },
                FaceChannelMeta {
                    index: 12,
                    name: "l_lip",
                    description: "下唇",
                },
                FaceChannelMeta {
                    index: 13,
                    name: "hair",
                    description: "头发",
                },
                FaceChannelMeta {
                    index: 14,
                    name: "hat",
                    description: "帽子",
                },
                FaceChannelMeta {
                    index: 15,
                    name: "ear_r",
                    description: "耳环",
                },
                FaceChannelMeta {
                    index: 16,
                    name: "neck_l",
                    description: "项链",
                },
                FaceChannelMeta {
                    index: 17,
                    name: "neck",
                    description: "颈部",
                },
                FaceChannelMeta {
                    index: 18,
                    name: "cloth",
                    description: "衣物",
                },
            ],
        })
    }
}

pub struct FaceParsing {
    model: Session,
}

impl FaceParsing {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let face_parser = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;
        Ok(Self { model: face_parser })
    }

    pub fn parse(&mut self, face_img: &DynamicImage) -> Result<FaceRawData> {
        // 预处理（与人脸解析模型匹配）
        let (input_tensor, _) =
            crate::utils::process_img::preprocess_image_with_padding_square(&face_img, 512)?; // 需要单独的预处理函数
        let input_tensor = TensorRef::from_array_view(input_tensor.view())?;

        // 推理
        let outputs = self.model.run(ort::inputs!["input" => input_tensor])?;

        let mask_array = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("无法提取mask张量")?;

        FaceRawData::new(mask_array.to_owned())
    }
}

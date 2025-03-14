use crate::utils::process_img::{
    preprocess_image_with_padding_square, BoxDetection, TransformInfo,
};
use anyhow::Result;
use image::{DynamicImage, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::{Array4, ArrayD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

pub struct YOLOFaceDetector {
    model: Session,
    conf_threshold: f32,
    iou_threshold: f32,
}

impl YOLOFaceDetector {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Self> {
        // 创建会话
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(model_path)?;

        Ok(Self {
            model,
            conf_threshold,
            iou_threshold,
        })
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<BoxDetection>> {
        // 准备模型输入数据 (预处理)
        let (input_tensor, transform_info) = preprocess_image_with_padding_square(img, 640)?;

        // 运行推理
        let outputs = self.inference(&input_tensor)?;

        // 解析输出张量并应用NMS
        let detections = self.postprocess(&outputs)?;

        // 生成带有预测框的图像和位置信息
        let result = self.generate_results(&img, &detections, transform_info)?;

        Ok(result)
    }

    fn inference(&mut self, input_tensor: &Array4<f32>) -> Result<ArrayD<f32>> {
        let input_tensor = TensorRef::from_array_view(input_tensor.view())?;

        // 执行推理
        let outputs = self.model.run(ort::inputs!["input" => input_tensor])?;

        // 获取输出张量
        let face_recognition_data = outputs["output"].try_extract_tensor::<f32>()?;

        Ok(face_recognition_data.to_owned())
    }

    fn postprocess(&self, outputs: &ArrayD<f32>) -> Result<Vec<BoxDetection>> {
        let mut detections: Vec<BoxDetection> = Vec::new();
        let output_view = outputs.view();

        // 输出形状为[1, 20, 8400]
        // 对于YOLOFace，前4个值是边界框(x,y,w,h)，第5个是置信度，后面的是各类别置信度

        // 遍历所有8400个检测
        for i in 0..output_view.shape()[2] {
            // 获取置信度 (第5个值)
            let confidence = output_view[[0, 4, i]];

            // 如果置信度低于阈值，跳过
            if confidence < self.conf_threshold {
                continue;
            }

            // 寻找最高类别置信度及其索引
            let mut max_class_conf = 0.0;

            for c in 0..15 {
                // 假设有15个类别（剩余的20-5=15个值）
                let class_conf = output_view[[0, 5 + c, i]];
                if class_conf > max_class_conf {
                    max_class_conf = class_conf;
                }
            }

            // 计算最终置信度 = 对象置信度 * 类别置信度
            let final_confidence = confidence * max_class_conf;

            // 如果最终置信度低于阈值，跳过
            if final_confidence < self.conf_threshold {
                continue;
            }

            // 获取边界框坐标 (x, y, w, h) - 这是相对于640x640的缩放坐标
            let x = output_view[[0, 0, i]];
            let y = output_view[[0, 1, i]];
            let width = output_view[[0, 2, i]];
            let height = output_view[[0, 3, i]];

            // 添加检测结果
            detections.push(BoxDetection {
                x,
                y,
                width,
                height,
                conf: final_confidence,
            });
        }

        // 应用非极大值抑制(NMS)
        detections = self.nms(detections);

        Ok(detections)
    }

    fn nms(&self, mut boxes: Vec<BoxDetection>) -> Vec<BoxDetection> {
        // 按置信度降序排序
        boxes.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap());

        let mut result: Vec<BoxDetection> = Vec::new();

        while !boxes.is_empty() {
            // 取置信度最高的框
            let current = boxes.remove(0);
            result.push(current);

            // 移除所有与当前框IoU高于阈值的框
            boxes.retain(|b| self.calculate_iou(&current, &b) <= self.iou_threshold);
        }

        result
    }

    /// 计算两个边界框之间的交并比(IoU)
    ///
    /// IoU = 交集面积 / 并集面积
    ///
    /// 此函数接收两个BoxDetection结构体并返回IoU值
    /// 范围从0.0(无重叠)到1.0(完全重叠)
    pub fn calculate_iou(&self, box1: &BoxDetection, box2: &BoxDetection) -> f32 {
        // 将(x, y, width, height)格式转换为(x1, y1, x2, y2)格式
        let box1_x1 = box1.x;
        let box1_y1 = box1.y;
        let box1_x2 = box1.x + box1.width;
        let box1_y2 = box1.y + box1.height;

        let box2_x1 = box2.x;
        let box2_y1 = box2.y;
        let box2_x2 = box2.x + box2.width;
        let box2_y2 = box2.y + box2.height;

        // 计算交集坐标
        let intersection_x1 = box1_x1.max(box2_x1);
        let intersection_y1 = box1_y1.max(box2_y1);
        let intersection_x2 = box1_x2.min(box2_x2);
        let intersection_y2 = box1_y2.min(box2_y2);

        // 检查边界框是否重叠
        if intersection_x1 >= intersection_x2 || intersection_y1 >= intersection_y2 {
            return 0.0; // 无重叠
        }

        // 计算交集面积
        let intersection_area =
            (intersection_x2 - intersection_x1) as f32 * (intersection_y2 - intersection_y1) as f32;

        // 计算两个边界框的面积
        let box1_area = box1.width as f32 * box1.height as f32;
        let box2_area = box2.width as f32 * box2.height as f32;

        // 计算并集面积
        let union_area = box1_area + box2_area - intersection_area;

        // 返回IoU
        intersection_area / union_area
    }

    fn generate_results(
        &self,
        img: &DynamicImage,
        detections: &[BoxDetection],
        transform_info: TransformInfo,
    ) -> Result<Vec<BoxDetection>> {
        let mut results = Vec::new();

        // 创建一个可变的图像副本，用于绘制边界框
        let mut debug_img = img.clone();

        for detection in detections {
            let orig_detection = transform_info.convert_to_original_coordinates(detection);
            // 在调试图像上绘制边界框
            let rect = Rect::at(orig_detection.x as i32, orig_detection.y as i32)
                .of_size(orig_detection.width as u32, orig_detection.height as u32);
            draw_hollow_rect_mut(&mut debug_img, rect, Rgba([255, 0, 0, 255]));

            // 添加到结果中
            results.push(orig_detection);
        }
        // debug_img.save("detect.png")?;

        // 将带有框的图像添加到结果中，方便调试
        // if results.is_empty() {
        //     // 如果没有检测到人脸，也返回原图用于调试
        //     results.push((debug_img, (0, 0, img_width, img_height)));
        // } else {
        //     // 替换第一个结果为带框的图像
        //     let first_bbox = results[0].1;
        //     results[0] = (debug_img, first_bbox);
        // }

        Ok(results)
    }
}

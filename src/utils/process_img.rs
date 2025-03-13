use crate::face_processor::face_parsing::FaceRawData;
use crate::frame::enhance::FaceThresholds;
use anyhow::Result;
use image::imageops::FilterType;
use image::{
    imageops, DynamicImage, GenericImage, GenericImageView, GrayImage, Luma, Rgba, RgbaImage,
};
use ndarray::{Array, ArrayBase, ArrayD, Dim, Ix, IxDyn, OwnedRepr};

// 预处理图像为模型输入格式
pub fn preprocess_image_with_padding_square(
    img: &DynamicImage,
    target_size: u32,
) -> Result<(ArrayBase<OwnedRepr<f32>, Dim<[Ix; 4]>>, TransformInfo)> {
    // 调整图像大小
    let (img_resized, transform_info) = pad_to_square_with_size(img, target_size, 0.0, None);

    let mut array = Array::zeros((1, 3, target_size as usize, target_size as usize));

    // 假设模型需要归一化到[0,1]的RGB输入
    for (x, y, pixel) in img_resized.to_rgb8().enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0; // R
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; // G
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0; // B
    }

    Ok((array, transform_info))
}

pub fn normalize_image(img: &DynamicImage, target_size: (u32, u32)) -> ArrayD<f32> {
    // 1. 调整图像大小为 112x112
    let img = img.resize_exact(target_size.0, target_size.1, FilterType::Triangle);

    // 2. 转换图像为RGB并获取像素数据
    let rgb_img = img.to_rgb8();
    let (width, height) = (rgb_img.width() as usize, rgb_img.height() as usize);

    // 3. 创建一个形状为 [1, 3, 112, 112] 的数组，对应 NCHW 格式
    // (batch_size=1, channels=3, height=112, width=112)
    let mut tensor_data = Array::zeros((1, 3, height, width));

    // 4. 填充数据并同时归一化
    // 归一化使用 (pixel / 255.0 - 0.5) / 0.5，等价于 (pixel / 127.5 - 1.0)
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x as u32, y as u32);
            tensor_data[[0, 0, y, x]] = (pixel[0] as f32 / 127.5) - 1.0; // R
            tensor_data[[0, 1, y, x]] = (pixel[1] as f32 / 127.5) - 1.0; // G
            tensor_data[[0, 2, y, x]] = (pixel[2] as f32 / 127.5) - 1.0; // B
        }
    }

    // 5. 转换为动态维度数组
    let tensor_shape = IxDyn(&[1, 3, height, width]);
    ArrayD::from_shape_vec(tensor_shape, tensor_data.into_raw_vec_and_offset().0).unwrap()
}

pub fn l2_normalize(mut embedding: ArrayD<f32>) -> ArrayD<f32> {
    // 假设embedding形状为[1, 512]
    let shape = embedding.shape().to_vec();

    // 在维度1上计算L2范数
    let mut norm = Array::zeros(shape[0]);
    for i in 0..shape[0] {
        let mut sum_sq = 0.0;
        for j in 0..shape[1] {
            sum_sq += embedding[[i, j]].powi(2);
        }
        norm[i] = sum_sq.sqrt();
    }

    // 归一化
    for i in 0..shape[0] {
        if norm[i] > 1e-10 {
            // 避免除以零
            for j in 0..shape[1] {
                embedding[[i, j]] /= norm[i];
            }
        }
    }

    embedding
}

pub fn blend_mask(face: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage> {
    let mut result = DynamicImage::new_rgba8(face.width(), face.height());
    let (width, height) = (face.width(), face.height());

    // 确保掩码尺寸匹配
    assert_eq!(mask.width(), width, "掩码宽度不匹配");
    assert_eq!(mask.height(), height, "掩码高度不匹配");

    for fy in 0..height {
        for fx in 0..width {
            let alpha = (mask.get_pixel(fx, fy)[0] as f32 / 255.0) as u8;
            let face_pixel = face.get_pixel(fx, fy);
            let pixel = Rgba([
                (face_pixel[0] * alpha),
                (face_pixel[1] * alpha),
                (face_pixel[2] * alpha),
                alpha * 255,
            ]);
            // 线性混合
            result.put_pixel(fx, fy, pixel);
        }
    }

    Ok(result)
}

#[allow(unused)]
pub fn generate_face_mask(
    face_raw_data: FaceRawData,
    face_img: &DynamicImage,
) -> Result<GrayImage> {
    let mask = generate_dynamic_mask(&face_raw_data, &FaceThresholds::default())?;

    let (height, width) = (mask.nrows(), mask.ncols());
    let mut img = GrayImage::new(width as u32, height as u32);

    mask.indexed_iter().for_each(|((y, x), &is_face)| {
        let alpha: u8 = if is_face { 255 } else { 0 };
        img.put_pixel(x as u32, y as u32, Luma([alpha]));
    });
    let resized = imageops::resize(
        &img,
        face_img.width(),
        face_img.height(),
        FilterType::Lanczos3,
    );
    Ok(resized)
}

/// 动态生成组合掩码的工具方法
fn generate_dynamic_mask(
    data: &FaceRawData,
    thresholds: &FaceThresholds,
) -> Result<ndarray::Array2<bool>> {
    let shape = data.tensor.shape();
    let (height, width) = (shape[2], shape[3]);
    let mut mask = ndarray::Array2::<bool>::default((height, width));

    // 并行遍历每个像素
    ndarray::Zip::indexed(&mut mask).for_each(|(y, x), value| {
        let skin = data.tensor[[0, 1, y, x]] > thresholds.skin;
        let nose = data.tensor[[0, 2, y, x]] > thresholds.nose;
        let l_eye = data.tensor[[0, 4, y, x]] > thresholds.eyes;
        let r_eye = data.tensor[[0, 5, y, x]] > thresholds.eyes;
        let l_brow = data.tensor[[0, 6, y, x]] > thresholds.eyebrows;
        let r_brow = data.tensor[[0, 7, y, x]] > thresholds.eyebrows;
        let mouth = data.tensor[[0, 10, y, x]] > thresholds.mouth;
        let not_hair = data.tensor[[0, 13, y, x]] < thresholds.hair;

        *value = skin || nose || l_eye || r_eye || l_brow || r_brow || mouth && not_hair;
    });

    Ok(mask)
}

#[allow(unused)]
// 绘制人脸框
pub fn draw_rect(img: &mut DynamicImage, x1: u32, x2: u32, y1: u32, y2: u32) {
    let color = Rgba([255, 0, 0, 1]);
    for i in x1..x2 {
        img.put_pixel(i, y1, color);
        img.put_pixel(i, y2, color);
    }
    for i in y1..y2 {
        img.put_pixel(x1, i, color);
        img.put_pixel(x2, i, color);
    }
}

#[derive(Copy, Clone)]
pub struct BoxDetection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub conf: f32,
}

/// 图像变换信息结构体，存储预处理过程中的所有变换参数
#[derive(Debug, Clone, Copy)]
pub struct TransformInfo {
    // 原始图像信息
    pub orig_width: u32,
    pub orig_height: u32,

    // 裁剪信息
    pub crop_x: u32,
    pub crop_y: u32,
    pub crop_width: u32,
    pub crop_height: u32,

    // 填充信息
    pub pad_x_offset: u32,
    pub pad_y_offset: u32,

    // 缩放比例
    pub scale: f32,

    // 目标尺寸
    pub target_size: u32,
}

impl TransformInfo {
    /// 将YOLO检测坐标从预处理图像转换为原始图像坐标
    ///
    /// `detection` - YOLO的检测结果，假设格式为 (x, y, w, h, conf)
    ///               其中x,y是中心点坐标，w,h是宽高，都是相对于目标尺寸的像素值
    ///
    /// 返回：原始图像上的(x1, y1, x2, y2, conf)坐标和置信度
    pub fn convert_to_original_coordinates(&self, detection: &BoxDetection) -> BoxDetection {
        let (x_center, y_center, width, height) =
            (detection.x, detection.y, detection.width, detection.height);

        // 1. 将YOLO输出的中心点坐标转换为左上角坐标（在填充后的图像上）
        let padded_x1 = x_center - width / 2.0;
        let padded_y1 = y_center - height / 2.0;
        let padded_x2 = x_center + width / 2.0;
        let padded_y2 = y_center + height / 2.0;

        // 2. 去除填充偏移
        let unpadded_x1 = padded_x1 - self.pad_x_offset as f32;
        let unpadded_y1 = padded_y1 - self.pad_y_offset as f32;
        let unpadded_x2 = padded_x2 - self.pad_x_offset as f32;
        let unpadded_y2 = padded_y2 - self.pad_y_offset as f32;

        // 3. 反向缩放到裁剪尺寸
        let crop_x1 = unpadded_x1 / self.scale;
        let crop_y1 = unpadded_y1 / self.scale;
        let crop_x2 = unpadded_x2 / self.scale;
        let crop_y2 = unpadded_y2 / self.scale;

        // 4. 加上裁剪偏移，回到原始图像坐标
        let orig_x1 = crop_x1 + self.crop_x as f32;
        let orig_y1 = crop_y1 + self.crop_y as f32;
        let orig_x2 = crop_x2 + self.crop_x as f32;
        let orig_y2 = crop_y2 + self.crop_y as f32;

        // 5. 确保坐标在原图范围内
        let x1 = orig_x1.max(0.0).min(self.orig_width as f32) as u32;
        let y1 = orig_y1.max(0.0).min(self.orig_height as f32) as u32;
        let x2 = orig_x2.max(0.0).min(self.orig_width as f32) as u32;
        let y2 = orig_y2.max(0.0).min(self.orig_height as f32) as u32;

        BoxDetection {
            x: x1 as f32,
            y: y1 as f32,
            width: (x2 - x1) as f32,
            height: (y2 - y1) as f32,
            conf: detection.conf,
        }
    }
}

/// 将图像裁剪、调整大小并填充成指定尺寸的正方形，并返回变换信息
///
/// `img` - 输入图像
/// `target_size` - 目标正方形的边长
/// `crop_margin` - 裁剪时额外保留的边缘比例（0.0-1.0），增加可以保留更多上下文
///
/// 返回：(填充后的图像, 变换信息结构体)
pub fn pad_to_square_with_size(
    img: &DynamicImage,
    target_size: u32,
    crop_margin: f32,
    face_region: Option<(u32, u32, u32, u32)>, // 可选的人脸区域 (x, y, width, height)
) -> (DynamicImage, TransformInfo) {
    // 记录原始尺寸
    let (orig_width, orig_height) = (img.width(), img.height());

    // 1. 确定裁剪区域 - 基于人脸边界框或使用整个图像
    let (face_x, face_y, face_width, face_height) =
        face_region.unwrap_or((0, 0, orig_width, orig_height));

    // 计算扩展后的裁剪区域
    let crop_width = (face_width as f32 * (1.0 + crop_margin)).min(orig_width as f32) as u32;
    let crop_height = (face_height as f32 * (1.0 + crop_margin)).min(orig_height as f32) as u32;

    // 确保裁剪区域居中且不超出原图范围
    let mut crop_x =
        (face_x as f32 - (crop_width as f32 - face_width as f32) / 2.0).max(0.0) as u32;
    let mut crop_y =
        (face_y as f32 - (crop_height as f32 - face_height as f32) / 2.0).max(0.0) as u32;

    // 调整裁剪区域确保不超出图像边界
    if crop_x + crop_width > orig_width {
        crop_x = orig_width - crop_width;
    }
    if crop_y + crop_height > orig_height {
        crop_y = orig_height - crop_height;
    }

    // 2. 裁剪图像
    let cropped_img = img.crop_imm(crop_x, crop_y, crop_width, crop_height);

    // 3. 确定缩放因子，保持纵横比
    let crop_ratio = crop_width as f32 / crop_height as f32;
    let (scaled_width, scaled_height, scale) = if crop_ratio > 1.0 {
        // 横向图像，宽度适应目标尺寸
        let new_width = target_size;
        let new_height = (new_width as f32 / crop_ratio) as u32;
        (new_width, new_height, new_width as f32 / crop_width as f32)
    } else {
        // 纵向图像，高度适应目标尺寸
        let new_height = target_size;
        let new_width = (new_height as f32 * crop_ratio) as u32;
        (
            new_width,
            new_height,
            new_height as f32 / crop_height as f32,
        )
    };

    // 4. 调整大小
    let resized_img =
        cropped_img.resize_exact(scaled_width, scaled_height, imageops::FilterType::Lanczos3);

    // 5. 创建目标方形图像并填充
    let mut squared_img = RgbaImage::new(target_size, target_size);

    // 计算居中偏移
    let x_offset = (target_size - scaled_width) / 2;
    let y_offset = (target_size - scaled_height) / 2;

    // 复制调整大小后的图像到中心位置
    squared_img
        .copy_from(&resized_img, x_offset, y_offset)
        .expect("Failed to copy image");

    // 创建变换信息结构体
    let transform_info = TransformInfo {
        orig_width,
        orig_height,
        crop_x,
        crop_y,
        crop_width,
        crop_height,
        pad_x_offset: x_offset,
        pad_y_offset: y_offset,
        scale,
        target_size,
    };

    (DynamicImage::ImageRgba8(squared_img), transform_info)
}

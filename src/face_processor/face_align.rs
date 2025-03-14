use crate::face_processor::face_parsing::FaceRawData;
use anyhow::{anyhow, Result};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use ndarray::{Array3, Array4, ArrayView3, Axis};

/// 对模型输出的 logits（shape: [1, 19, 512, 512]）在通道维度上做 argmax，得到分割 mask（shape: [1, 512, 512]）
fn argmax_segmentation(seg_logits: &Array4<f32>) -> Array3<i64> {
    let (_, channels, height, width) = seg_logits.dim();
    let mut seg_mask = Array3::<i64>::zeros((1, height, width));
    for y in 0..height {
        for x in 0..width {
            let mut max_val = std::f32::MIN;
            let mut max_idx = 0;
            for c in 0..channels {
                let val = seg_logits[[0, c, y, x]];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }
            seg_mask[[0, y, x]] = max_idx as i64;
        }
    }
    seg_mask
}

#[allow(unused)]
/// 将分割 mask 可视化为 RGB 图像（颜色映射按类别顺序定义）
/// mask 的 shape 为 (1, height, width)
fn visualize_segmentation(seg: &ArrayView3<'_, i64>) -> RgbImage {
    // 定义每个类别的颜色（示例颜色，可根据需要调整）
    let colors = vec![
        (0, 0, 0),       // 0 background
        (255, 224, 189), // 1 skin
        (255, 0, 0),     // 2 nose
        (0, 0, 255),     // 3 eye_g / eyeglasses
        (0, 255, 0),     // 4 l_eye (left eye)
        (255, 0, 255),   // 5 r_eye (right eye)
        (255, 255, 0),   // 6 l_brow (left eyebrow)
        (0, 255, 255),   // 7 r_brow (right eyebrow)
        (128, 128, 128), // 8 l_ear (left ear)
        (64, 64, 64),    // 9 r_ear (right ear)
        (0, 128, 0),     // 10 mouth (area between lips)
        (128, 0, 0),     // 11 u_lip (upper lip)
        (128, 0, 128),   // 12 l_lip (lower lip)
        (0, 128, 128),   // 13 hair
        (128, 128, 0),   // 14 hat
        (255, 165, 0),   // 15 ear_r (earring)
        (100, 100, 100), // 16 neck_l (necklace)
        (50, 50, 50),    // 17 neck
        (200, 200, 200), // 18 cloth (clothing)
    ];

    // mask 的 shape 为 (1, height, width)
    let seg_2d = seg.index_axis(Axis(0), 0);
    let (height, width) = (seg_2d.dim().0, seg_2d.dim().1);
    let mut imgbuf = ImageBuffer::new(width as u32, height as u32);

    for ((y, x), &label) in seg_2d.indexed_iter() {
        let (r, g, b) = if let Some(&(r, g, b)) = colors.get(label as usize) {
            (r, g, b)
        } else {
            (255, 255, 255)
        };
        imgbuf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
    }
    imgbuf
}

/// 计算指定类别在分割 mask 中的质心（像素坐标），mask 的 shape 为 (1, height, width)
fn compute_centroid(seg: &ArrayView3<'_, i64>, target_label: i64) -> Option<(u32, u32)> {
    let seg_2d = seg.index_axis(Axis(0), 0);
    let (mut sum_x, mut sum_y, mut count) = (0u32, 0u32, 0u32);
    for ((y, x), &value) in seg_2d.indexed_iter() {
        if value == target_label {
            sum_x += x as u32;
            sum_y += y as u32;
            count += 1;
        }
    }
    if count > 0 {
        Some((sum_x / count, sum_y / count))
    } else {
        None
    }
}

pub fn align_and_normalize_face(
    img: &DynamicImage,
    face_raw_data: FaceRawData,
) -> Result<(DynamicImage, f32)> {
    // 输出 logits，形状为 (1, 19, 512, 512)
    // let seg_logits = outputs[0].try_extract::<Array4<f32>>()?;
    // 重塑为四维形状（需确保元素总数一致）
    let seg_logits: Array4<f32> = face_raw_data.tensor.into_dimensionality()?;

    // 对 logits 在通道维度上做 argmax，得到分割 mask（shape: [1, 512, 512]）
    let seg_mask = argmax_segmentation(&seg_logits);

    // 可视化分割结果，保存中间结果以便调试
    // let seg_img = visualize_segmentation(&seg_mask.view());
    // seg_img.save("segmentation.png")?;
    // log!("已保存分割图像：segmentation.png");

    // 计算关键点：左右眼的质心（根据模型输出，标签 4 = 左眼，标签 5 = 右眼）
    let left_eye = compute_centroid(&seg_mask.view(), 4);
    let right_eye = compute_centroid(&seg_mask.view(), 5);

    let (left_eye_center, right_eye_center) = (
        left_eye.ok_or(anyhow!("找不到左眼"))?,
        right_eye.ok_or(anyhow!("找不到右眼"))?,
    );
    let dx = right_eye_center.0 as f32 - left_eye_center.0 as f32;
    let dy = right_eye_center.1 as f32 - left_eye_center.1 as f32;
    let mut angle = dy.atan2(dx).to_degrees();

    // 对角度进行修正，确保旋转角度在合理范围内，避免 180 度的误差
    if angle > 90.0 {
        angle -= 180.0;
    } else if angle < -90.0 {
        angle += 180.0;
    }
    // 为对齐人脸，将图像旋转相反角度
    let img_rgb = img.to_rgb8();
    let aligned_img = rotate_about_center(
        &img_rgb,
        -angle.to_radians(),
        Interpolation::Bilinear,
        Rgb([0, 0, 0]),
    );
    Ok((aligned_img.into(), angle.to_radians()))
}

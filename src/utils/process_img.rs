use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array, ArrayBase, Dim, Ix, OwnedRepr};

// 预处理图像为模型输入格式
pub fn preprocess_image(img: &DynamicImage, target_size: (u32, u32)) -> anyhow::Result<ArrayBase<OwnedRepr<f32>, Dim<[Ix; 4]>>> {
    // 调整图像大小
    let img_resized = img.resize_exact(target_size.0, target_size.1, FilterType::Triangle);

    let mut array = Array::zeros((1, 3, target_size.1 as usize, target_size.0 as usize));

    // 假设模型需要归一化到[0,1]的RGB输入
    for (x, y, pixel) in img_resized.to_rgb8().enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0; // R
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; // G
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0; // B
    }

    Ok(array)
}
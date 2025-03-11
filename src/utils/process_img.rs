use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::{Array, ArrayBase, ArrayD, Dim, Ix, IxDyn, OwnedRepr};

// 预处理图像为模型输入格式
pub fn preprocess_image(
    img: &DynamicImage,
    target_size: (u32, u32),
) -> anyhow::Result<ArrayBase<OwnedRepr<f32>, Dim<[Ix; 4]>>> {
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


use ndarray::ArrayView1;

pub mod process_img;

pub mod log {
    // 日志宏，方便统一格式化和输出日志
    #[macro_export]
    macro_rules! log {
        ($($arg:tt)*) => {
            println!("[ReHiFace] {}", format!($($arg)*));
        }
    }
}


#[allow(unused)]
pub fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot_product = a.dot(&b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();
    dot_product / (norm_a * norm_b)
}


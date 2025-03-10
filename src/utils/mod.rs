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
use crate::utilities::{self, frame};

pub fn yin(y : Vec<f32>,
    fmin : f32,
    fmax : f32,
    sr : Option<i32>,
    frame_length : Option<i32>,
    hop_length : Option<i32>,
    trough_threshold : Option<f32>,
    center : Option<bool>,
    pad_mode : Option<&str>) {

    let sr : i32 = sr.unwrap_or(22050);
    let frame_length : i32 = frame_length.unwrap_or(2048);
    let hop_length : i32 = hop_length.unwrap_or(frame_length / 4);
    let trough_threshold : f32 = trough_threshold.unwrap_or(0.1);
    let center : bool = center.unwrap_or(true);
    let pad_mode : &str = pad_mode.unwrap_or("constant");

    check_yin_params(sr, fmin, fmax, frame_length);

    utilities::valid_audio(&y);

    let padded_y : Vec<f32> = pad(&y, (frame_length / 2) as usize);

    let y_frames : Vec<Vec<f32>> = frame(y, frame_length, hop_length);
    
    
}

fn check_yin_params(sr : i32, fmin : f32, fmax : f32, frame_length : i32) {
    use log::warn;
    if fmax > sr as f32 / 2.0 {
        panic!("fmax={} cannot exceed Nyquist frequency {}", fmax, sr as f32/2.0);
    }
    if fmin >= fmax {
        panic!("fmin={} must be less than fmax={}", fmin, fmax);
    }
    if fmin <= 0.0 {
        panic!("fmin={} must be strictly positive", fmin);
    }

    if sr as f32 / fmin >= (frame_length - 1) as f32 {
        let fmin_feasible : f32 = sr as f32 / (frame_length - 1) as f32;
        let frame_len_feasible : i32 = (sr as f32 / fmin).ceil() as i32 + 1;
        panic!("fmin={} is too small for frame_length={} and sr={}\n\
        Either increase to fmin={} or frame_length={}",
    fmin, frame_length, sr, fmin_feasible, frame_len_feasible)
    }

    if sr as f32 / fmin >= frame_length as f32 / 2.0 {
        let fmin_optimal : i32 = sr / (frame_length / 2);
        let frame_length_optimal : i32 = (sr as f32 / fmin as f32).ceil() as i32 * 2 + 1;
        warn!("With fmin={}, sr={}, and frame_length={}, less than two periods of fmin\n\
        fit into the frame, which can cause inaccurate pitch detection.\n\
        Consider increasing to fmin={}, or frame_length={}",
        fmin, sr, frame_length, fmin_optimal, frame_length_optimal);
    }
}


fn pad(y : &Vec<f32>, padding : usize) -> Vec<f32> {
    let mut result : Vec<f32> = Vec::with_capacity(y.len() + 2*padding);
    result.extend(vec![0.0; padding]);
    result.extend_from_slice(y);
    result.extend(vec![0.0; padding]);
    result    
}
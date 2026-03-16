use crate::core_io::*;

pub fn samples_like(x: i32, hop_length: Option<i32>, n_fft: Option<i32>) -> Vec<i32> {
    (0..x)
        .map(|f| frames_to_samples(f, hop_length, n_fft))
        .collect()
}

pub fn times_like(
    x: i32,
    sr: Option<i32>,
    hop_length: Option<i32>,
    n_fft: Option<i32>,
) -> Vec<f32> {
    let samples: Vec<i32> = samples_like(x, hop_length, n_fft);
    samples
        .into_iter()
        .map(|s| samples_to_time(s, sr))
        .collect()
}

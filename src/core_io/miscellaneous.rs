use crate::core_io::*;

/// Creates vector of sample indices that correspend to frames
///
/// Creates a vector of length x, where index i contains
/// the first sample of frame i
///
/// ### Arguments
/// x: Number of  samples to generate
///
/// hop_length: Size of each frame
///
/// n_fft: FFT window size. Defaults to 0
///
/// ### Returns
/// samples: vector of samples, length x
pub fn samples_like(x: i32, hop_length: Option<i32>, n_fft: Option<i32>) -> Vec<i32> {
    (0..x)
        .map(|f| frames_to_samples(f, hop_length, n_fft))
        .collect()
}

/// Creates vector of times that correspend to frames
///
/// Creates a vector of length x, where index i contains
/// the time that frame i begins
///
/// ### Arguments
/// x: Number of  samples to generate
///
/// sr: sample rate
///
/// hop_length: Size of each frame
///
/// n_fft: FFT window size. Defaults to 0
///
/// ### Returns
/// times: vector of times, length x
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

#[cfg(test)]
mod tests {
    use crate::core_io::{samples_like, times_like};

    #[test]
    fn test_samples_like() {
        let expected: Vec<i32> = vec![0, 512, 1024];
        assert_eq!(expected, samples_like(3, None, None));
    }

    #[test]
    fn test_times_like() {
        let expected: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5];
        assert_eq!(expected, times_like(4, Some(44100), Some(22050), None));
    }
}

/// Coverts frame index to sample index
///
/// ### Arguments:
/// frame: frame index
///
/// hop length: Distance between the start of each frame. Defaults to 512
///
/// n_fft: FFT window size. Defaults to 0
///
/// ### Returns
/// sample: sample index    
pub fn frames_to_samples(frame: i32, hop_length: Option<i32>, n_fft: Option<i32>) -> i32 {
    let offset: i32 = n_fft.unwrap_or(0) / 2;
    frame * hop_length.unwrap_or(512) + offset
}

/// Converts frame index to time in seconds
///
/// ### Arguments:
/// frame: frame index
///
/// sr: sample rate. Defaults to 22050
///
/// hop_length: distance between frames. Defaults to 512
///
/// n_fft: FFT window size. Defaults to 0
///
/// ### Returns:
/// time: time of the frame in seconds
pub fn frames_to_time(
    frame: i32,
    sr: Option<i32>,
    hop_length: Option<i32>,
    n_fft: Option<i32>,
) -> f32 {
    let sample: i32 = frames_to_samples(frame, hop_length, n_fft);
    samples_to_time(sample, sr)
}

/// Converts sample index to frame index
///
/// ### Arguments:
/// sample: sample index
///
/// hop_length: distance between frames. Defaults to 512
///
/// n_fft: FFT window size. Defaults to 0
///
/// ### Returns:
/// frame: index of frame
pub fn samples_to_frames(sample: i32, hop_length: Option<i32>, n_fft: Option<i32>) -> i32 {
    let offset: i32 = n_fft.unwrap_or(0) / 2;
    (sample - offset) / hop_length.unwrap_or(512)
}

/// Converts sample index to time in seconds
///
/// ### Arguments:
/// sample: sample index
///
/// sr: sample rate. Defaults to 22050
///
/// ### Returns:
/// time: time of the frame in seconds
pub fn samples_to_time(sample: i32, sr: Option<i32>) -> f32 {
    (sample as f32) / (sr.unwrap_or(22050) as f32)
}

/// Converts time in seconds to frame index
///
/// ### Arguments:
/// time: time, in seconds
///
/// sr: sample rate. Defaults to 22050
///
/// hop_length: distance between frames. Defaults to 512
///
/// n_fft: FFT window size. Defaults to 0
///
/// ### Returns:
/// frame: frame index of the time
pub fn time_to_frames(
    time: f32,
    sr: Option<i32>,
    hop_length: Option<i32>,
    n_fft: Option<i32>,
) -> i32 {
    let sample: i32 = time_to_samples(time, sr);
    samples_to_frames(sample, hop_length, n_fft)
}

/// Converts time in seconds to sample index
///
/// ### Arguments:
/// time: time, in seconds
///
/// sr: sample rate. Defaults to 22050
///
/// ### Returns:
/// sample: sample index of the time
pub fn time_to_samples(time: f32, sr: Option<i32>) -> i32 {
    (time * (sr.unwrap_or(22050) as f32)) as i32
}

/// Converts block index to frame index
///
/// ### Arguments:
/// block: block index
///
/// block_length: length of each block.
///
/// ### Returns:
/// frame: frame index corresponding to the beginning of the block
pub fn blocks_to_frames(block: i32, block_length: i32) -> i32 {
    block * block_length
}

/// Converts block index to sample index
///
/// ### Arguments:
/// block: block index
///
/// block_length: length of each block.
///
/// hop_length: distance between each frame. Defaults to 512
///
/// ### Returns:
/// sample: sample index corresponding to the beginning of the block
pub fn blocks_to_samples(block: i32, block_length: i32, hop_length: Option<i32>) -> i32 {
    let frame: i32 = blocks_to_frames(block, block_length);
    frames_to_samples(frame, hop_length, None)
}

/// Converts block index to time in seconds
///
/// ### Arguments:
/// block: block index
///
/// block_length: length of each block.
///
/// hop_length: distance between each frame. Defaults to 512
///
/// sr: sample rate. Defaults to 22050
///
/// ### Returns:
/// time: time in seconds corresponding to the beginning of the block
pub fn blocks_to_time(
    block: i32,
    block_length: i32,
    hop_length: Option<i32>,
    sr: Option<i32>,
) -> f32 {
    let sample: i32 = blocks_to_samples(block, block_length, hop_length);
    samples_to_time(sample, sr)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test frames_to_samples
    #[test]
    fn tst_frame_0_to_samples() {
        assert_eq!(
            0,
            frames_to_samples(0, Some(512), None),
            "Failed with n_fft = None"
        );
        assert_eq!(
            1,
            frames_to_samples(0, Some(512), Some(2)),
            "Failed with n_fft = 2"
        );
        assert_eq!(
            1,
            frames_to_samples(0, Some(512), Some(3)),
            "Failed with n_fft = 3"
        ); // Ensure rounding down
    }

    #[test]
    fn test_frames_to_samples_non_zero() {
        assert_eq!(
            1024,
            frames_to_samples(2, None, None),
            "Failed with n_fft = None"
        );
        assert_eq!(
            1025,
            frames_to_samples(2, None, Some(2)),
            "Failed with n_fft = 2"
        );
        assert_eq!(
            1025,
            frames_to_samples(2, None, Some(3)),
            "Failed with n_fft = 3"
        );
    }

    // Test samples_to_frames
    #[test]
    fn test_sample_0_to_frames() {
        assert_eq!(
            0,
            samples_to_frames(0, None, None),
            "Failed with n_fft = None"
        );
        assert_eq!(
            0,
            samples_to_frames(0, None, Some(2)),
            "Failed with n_fft = 2"
        );
        assert_eq!(
            0,
            samples_to_frames(0, None, Some(3)),
            "Failed with n_fft = 3"
        );
    }

    #[test]
    fn sample_to_frame_fft_boundary() {
        assert_eq!(
            3,
            samples_to_frames(4000, Some(1024), Some(1857)),
            "Failed with n_fft=1857"
        );
        assert_eq!(
            2,
            samples_to_frames(4000, Some(1024), Some(1858)),
            "Failed with n_fft=1858"
        );
    }

    #[test]
    fn sample_to_frame_sample_boundary() {
        assert_eq!(
            0,
            samples_to_frames(511, None, None),
            "Failed at boundary lower end"
        );
        assert_eq!(
            1,
            samples_to_frames(512, None, None),
            "Failed at boundary upper end"
        );
    }

    // Test samples_to_time
    #[test]
    fn test_sample_to_time_0() {
        assert_eq!(0.0, samples_to_time(0, None), "Failed with sr=None");
        assert_eq!(0.0, samples_to_time(0, Some(44100)), "Failed with sr!=None");
    }

    #[test]
    fn test_sample_to_time_not_0() {
        assert_eq!(1.0, samples_to_time(22050, None), "Failed with sr=None");
        assert_eq!(
            0.5,
            samples_to_time(22050, Some(44100)),
            "Failed with sr=44100"
        );
    }

    // Test time_to_samples
    #[test]
    fn test_time_to_samples_0() {
        assert_eq!(0, time_to_samples(0.0, None));
    }

    #[test]
    fn test_time_to_samples_not_0() {
        assert_eq!(22050, time_to_samples(1.0, None), "Failed with sr=None");
        assert_eq!(
            22050,
            time_to_samples(1.0, Some(22050)),
            "Failed with sr=22050"
        );
        assert_eq!(
            22050,
            time_to_samples(0.5, Some(44100)),
            "Failed with sr=44100"
        );
    }

    // Test blocks_to_frames
    #[test]
    fn test_block_to_frame_0() {
        assert_eq!(0, blocks_to_frames(0, 1), "Failed when block index=0");
        assert_eq!(0, blocks_to_frames(4, 0), "Failed when block size=0");
    }

    #[test]
    fn test_block_to_frame_not_0() {
        assert_eq!(20, blocks_to_frames(4, 5));
    }
}

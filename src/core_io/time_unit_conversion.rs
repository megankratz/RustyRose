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
pub fn frames_to_samples(frame : i32, hop_length : Option<i32>, n_fft : Option<i32>) -> i32 {
    let offset : i32 = n_fft.unwrap_or(0) / 2;
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
pub fn frames_to_time(frame : i32, sr : Option<i32>, hop_length : Option<i32>, n_fft : Option<i32>) -> f32 {
    let sample : i32 = frames_to_samples(frame, hop_length, n_fft);
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
pub fn samples_to_frames(sample : i32, hop_length : Option<i32>, n_fft : Option<i32>) -> i32 {
    let offset : i32 = n_fft.unwrap_or(0) / 2;
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
pub fn samples_to_time(sample : i32, sr : Option<i32>) -> f32 {
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
pub fn time_to_frames(time : f32, sr : Option<i32>, hop_length : Option<i32>, n_fft : Option<i32>) -> i32 {
    let sample : i32 = time_to_samples(time, sr);
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
pub fn time_to_samples(time : f32, sr : Option<i32>) -> i32 {
    (time * (sr.unwrap_or(22050) as f32) ) as i32
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
pub fn blocks_to_frames(block : i32, block_length : i32) -> i32 {
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
pub fn blocks_to_samples(block : i32, block_length : i32, hop_length : Option<i32>) -> i32 {
    let frame : i32 = blocks_to_frames(block, block_length);
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
pub fn blocks_to_time(block : i32, block_length : i32, hop_length : Option<i32>, sr : Option<i32>) -> f32 {
    let sample : i32 = blocks_to_samples(block, block_length, hop_length);
    samples_to_time(sample, sr)
}



// #[cfg(test)]
// mod tests {
//     use crate::core_io;

//     #[test]
// }
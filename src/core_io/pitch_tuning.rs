use crate::utilities::{self, frame};
use crate::core_io;
use ndarray::{Array1, Array2, array};
use std::cmp;

/// Fundamental pitch estimation for a single voice
/// 
/// ### Parameters
/// y : Input samples
/// 
/// fmin : minimum frequency
/// 
/// fmax : maximum frequency
/// 
/// sr : sample rate. Defaults to 22050
/// 
/// frame_length : size of each frame. Defaults to 2048
/// 
/// hop_length : number of samples between each frame. Defaults to 512
/// 
/// trough_threshold : absolute threshold for peak estimation. Defaults to 0.1
/// 
/// center : if True, then y is padded such that frame t is centered at y[t * hop_length]
/// 
/// ### Returns
/// f0 : Array of pitchs of each frame
pub fn yin(y : &Vec<f32>,
    fmin : f32,
    fmax : f32,
    sr : Option<i32>,
    frame_length : Option<i32>,
    hop_length : Option<i32>,
    trough_threshold : Option<f32>,
    center : Option<bool>) -> Vec<f32> {

    let sr : i32 = sr.unwrap_or(22050);
    let frame_length : i32 = frame_length.unwrap_or(2048);
    let hop_length : i32 = hop_length.unwrap_or(frame_length / 4);
    let trough_threshold : f32 = trough_threshold.unwrap_or(0.1);
    let center : bool = center.unwrap_or(true);

    check_yin_params(sr, fmin, fmax, frame_length);

    utilities::valid_audio(&y);

    // Center audio
    let padded_y : Vec<f32> = if center {
        pad(y, (frame_length / 2) as usize)
    } else {
        y.to_vec()
    };

    // Frame audio
    let y_frames_t : Vec<Vec<f32>> = frame(padded_y, frame_length, hop_length);
    let n_frames = y_frames_t[0].len();
    let frame_len = y_frames_t.len();
    let y_frames: Vec<Vec<f32>> = (0..n_frames)
        .map(|f| (0..frame_len).map(|i| y_frames_t[i][f]).collect())
        .collect();

    // Calculate min and max periods
    let min_period : usize = (sr as f32 / fmax).floor() as usize;
    let max_period : usize = cmp::min(
        (sr as f32 / fmin).ceil() as usize,
        (frame_length - 1) as usize
    );

    // Calculate cumulative mean normalized difference function
    let yin_frames = cumulative_mean_normalized_difference(
        y_frames, min_period, max_period
    );

    let parabolic_shifts : Vec<Vec<f32>> = parabolic_interpolation(&yin_frames);

    let mut is_trough = utilities::localmin(&yin_frames);

    let n_frames : usize = yin_frames.len();
    let n_tau : usize = yin_frames[0].len();

    // Check if any frame is a trough
    for f in 0..n_frames 
    {
        if n_tau > 1 
        {
            is_trough[f][0] = yin_frames[f][0] < yin_frames[f][1];
        }
    }


    let mut is_threshold_trough = vec![vec![false; n_tau]; n_frames];
    for f in 0..n_frames {
        for t in 0..n_tau {
            is_threshold_trough[f][t] =
                is_trough[f][t] && yin_frames[f][t] < trough_threshold;
        }
    }
    
    let mut f0 : Vec<f32> = vec![0.0; n_frames];

    for f in 0..n_frames 
    {
        let mut tau_index : Option<usize> = None;
        // Only search tau indices corresponding to fmin..fmax
        // let min_tau_idx = (sr as f32 / fmax).floor() as usize - min_period;
        // let max_tau_idx = cmp::min((sr as f32 / fmin).ceil() as usize - min_period, n_tau - 1);
        let min_tau_idx = 0;
        let max_tau_idx = n_tau - 1;

        for t in min_tau_idx..=max_tau_idx {
            if is_threshold_trough[f][t] {
                tau_index = Some(t);
                break;
            }
        }


        // argmin function
        let global_min = yin_frames[f]
            .iter()
            .enumerate()
            .min_by(|a,b| a.1.total_cmp(b.1))
            .unwrap().0 as i32;

        if tau_index.is_none() {
            tau_index = Some(
                (min_tau_idx..=max_tau_idx)
                    .min_by(|&a, &b| yin_frames[f][a].total_cmp(&yin_frames[f][b]))
                    .unwrap()
            );
        }
        let tau_index = tau_index.unwrap_or(global_min as usize);

        let r_tau : f32 = min_period as f32 + tau_index as f32 + parabolic_shifts[f][tau_index];

        f0[f] = sr as f32 / r_tau;
    }

    f0
}


/// Given a collection of pitches, estimate tuning offset (in fractions of a bin) relative 
/// to A4 = 440.0Hz
/// 
/// ### Parameters
/// frequencies : collection of frequencies
/// 
/// resolution : Resolution of tuning, to a fraction of a bin. Defaults to 0.01
/// 
/// bins_per_octave : Number of frequency bins per octave. Defaults to 12
/// 
/// ### Returns
/// tuning: float in range [-0.5, 0.5]
pub fn pitch_tuning(
    frequencies : &Array1<f32>,
    resolution : Option<f32>,
    bins_per_octave : Option<u32>
) -> f32 {

    let bins : i32 = bins_per_octave.unwrap_or(12) as i32;
    let res : f32 = resolution.unwrap_or(0.01);

    // Trim out DC components
    let freqs : Array1<f32> = frequencies
        .iter()
        .copied()
        .filter(|&f| f > 0.0)
        .collect();

    if freqs.is_empty() {
        panic!("Trying to estimate tuning from empty frequency set");
    }

    let residual : Array1<f32> = freqs
        .iter()
        .copied()
        .map(|f| core_io::hz_to_octs(f, 440., bins))
        .map(|oct| {
            let mut r = (bins as f32 * oct).fract();
            if r < 0.0 {
                r = r + 1.0
            } 
            if r >= 0.5 {
                r - 1.0
            }else {
                r
            }
        })
        .collect();

    let n_bins : usize = (1.0 / res).ceil() as usize;
    let bins : Vec<f32> = (0..=n_bins)
        .map(|i| -0.5 + i as f32 / n_bins as f32)
        .collect();

    let mut counts : Vec<usize> = vec![0; n_bins];
    for &r in residual.iter() {
        if r < bins[0] || r > bins[bins.len() - 1] {
            continue;
        }
        let index = bins.partition_point(|&b| b <= r)// Find first bin value less than r
            .saturating_sub(1) 
            .min(n_bins - 1);  // handle upper bound
        counts[index] += 1;
    }

    // Return the peak bin
    let peak_index : usize = counts 
        .iter()
        .enumerate()
        .max_by_key(|&(_, &val)| val)// Get tuple of index and max value
        .map(|(i, _)| i) // Remove max value to just get index
        .unwrap();

    bins[peak_index]
}


pub fn piptrack(
    y : &Array1<f32>, 
    sr : Option<i32>, 
    n_fft : Option<i32>, 
    hop_length : Option<i32>,
    threshold : Option<f32>,
    fmin : Option<f32>,
    fmax : Option<f32>,
    win_length : Option<i32>
) -> (Array2<f32>, Array2<f32>) {
    (array![[],[]], array![[],[]])
}

/// Estimate the tuning offset relative to A=440hz
/// 
/// ### Parameters
/// y : Input audio samples
/// 
/// sr : sample rate. Defaults to 22050
/// 
/// n_fft : number of fft bins to use. Defaults to 2048
/// 
/// resolution : Resolution of tuning, to a fraction of a bin. Defaults to 0.01
/// 
/// bins_per_octave : Number of frequency bins per octave. Defaults to 12
/// 
/// ### Returns
/// tuning: float in range [-0.5, 0.5]
pub fn estimate_tuning(
    y : &Array1<f32>,
    sr : Option<i32>,
    n_fft : Option<i32>,
    resolution : Option<f32>,
    bins_per_octave : Option<u32>
) -> f32 {


    let (pitch, mag) = piptrack(y, sr, n_fft, None, None, None, None, None);

    let pitch_mask : Array1<bool> = pitch.iter()
        .map(|&p| p > 0.0).collect();

    let threshold : f32 = if pitch_mask.iter().any(|&x| x) {
        let mut masked : Vec<f32> = mag.iter()
        .zip(pitch_mask.iter())
        .filter_map(|(&v, &mask)| if mask {Some(v)} else {None})
        .collect();

        let mid = masked.len() / 2;
        masked.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        if mid % 2 == 1 {
            masked[mid]
        } else {
            masked.select_nth_unstable_by(mid-1, |a, b| a.partial_cmp(b).unwrap());
            (masked[mid] + masked[mid+1]) / 2.0
        }
        } else {
            0.0
        };

    let filtered_pitches : Array1<f32> = pitch.iter()
        .zip(mag.iter())
        .zip(pitch_mask.iter())
        .filter_map(|((&p, &m), &mask)| {
            if mask && m >= threshold {Some(p)} else {None}
        })
        .collect();

    pitch_tuning(&filtered_pitches, resolution, bins_per_octave)
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


fn cumulative_mean_normalized_difference(y_frames : Vec<Vec<f32>>,
                                        min_period : usize,
                                        max_period : usize)
                                         -> Vec<Vec<f32>>
{
    let n_frames : usize = y_frames.len();
    // let frame_length : usize = y_frames[0].len();

    // let out_size : usize = max_period - min_period + 1;

    let tau_max = max_period.min(y_frames[1].len() / 2);
    let out_size : usize = tau_max - min_period + 1;
    let mut result : Vec<Vec<f32>> = vec![vec![0.0; out_size]; n_frames];


    for (frame_idx, frame) in y_frames.iter().enumerate()
    {
        let frame_length = frame.len();
        // let tau_max = max_period.min(frame_length - 1);


        // Difference function d(tau)
        let mut d = vec![0.0f32; max_period + 1];

        for tau in 1..=tau_max {
            let mut sum = 0.0;

            for i in 0..(frame_length - tau) {
                let diff = frame[i] - frame[i + tau];
                sum += diff * diff;
            }

            d[tau] = sum;
        }

        // CMND
        let mut running_sum = 0.0;

        for tau in 1..=tau_max {

            running_sum += d[tau];

            let cmnd = if tau == 0 {
                1.0
            } else {
                if running_sum == 0.0 {
                    1.0
                } else {
                    d[tau] * tau as f32 / running_sum
                }
            };

            if tau >= min_period {
                result[frame_idx][tau - min_period] = cmnd;
            }
        }
    }

    result
}


pub fn parabolic_interpolation(yin_frames: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {

    let n_frames : usize = yin_frames.len();
    let n_tau : usize = yin_frames[0].len();

    let mut shifts : Vec<Vec<f32>> = vec![vec![0.0; n_tau]; n_frames];

    for f in 0..n_frames {

        for t in 1..n_tau-1 {

            let ym1 : f32 = yin_frames[f][t-1];
            let y0 : f32  = yin_frames[f][t];
            let yp1 : f32 = yin_frames[f][t+1];

            let denom : f32 = ym1 - 2.0*y0 + yp1;

            if denom.abs() > 1e-12 {
                shifts[f][t] = 0.5 * (ym1 - yp1) / denom;
            }
        }
    }

    shifts
}


#[cfg(test)]
mod tests
{
    use crate::core_io::*;
    use ndarray::{Array1, array};

    #[test]
    fn test_tuning_known() {
        let freqs : Array1<f32> = array![55.0, 65.406, 77.782, 92.499, 110.0, 130.813];
        let result = pitch_tuning(&freqs, None, None);
        assert!((result - 0.25) < 0.01);
    }

    #[test]
    fn test_a440_octaves() {
        let freqs : Array1<f32> = array![55.0, 110.0, 220.0, 440.0, 880.0, 1760.0];
        let result : f32 = pitch_tuning(&freqs, None, None);
        assert_eq!(0.0, result);
    }

    #[test]
    fn test_below_a440() {
        let freqs : Array1<f32> = array![55.0, 110.0, 220.0];
        let result : f32 = pitch_tuning(&freqs, None, None);
        assert_eq!(0.0, result);
    }

}
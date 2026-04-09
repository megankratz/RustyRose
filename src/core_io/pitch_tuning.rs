use crate::utilities::{self, frame};
use std::cmp;

pub fn yin(y : &Vec<f32>,
    fmin : f32,
    fmax : f32,
    sr : Option<i32>,
    frame_length : Option<i32>,
    hop_length : Option<i32>,
    trough_threshold : Option<f32>,
    center : Option<bool>,
    pad_mode : Option<&str>) -> Vec<f32> {

    let sr : i32 = sr.unwrap_or(22050);
    let frame_length : i32 = frame_length.unwrap_or(2048);
    let hop_length : i32 = hop_length.unwrap_or(frame_length / 4);
    let trough_threshold : f32 = trough_threshold.unwrap_or(0.1);
    let center : bool = center.unwrap_or(true);
    let pad_mode : &str = pad_mode.unwrap_or("constant");

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
            is_trough[f][0] = (yin_frames[f][0] < yin_frames[f][1]);
        }
    }

    let mut debug_trough : i32 = 0;

    let mut is_threshold_trough = vec![vec![false; n_tau]; n_frames];
    for f in 0..n_frames {
        for t in 0..n_tau {
            is_threshold_trough[f][t] =
                is_trough[f][t] && yin_frames[f][t] < trough_threshold;
                if is_threshold_trough[f][t]{ debug_trough += 1;}
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


pub fn piptrack(y : Vec<f32>, 
    sr : Option<i32>, 
    n_fft : Option<i32>, 
    hop_length : Option<i32>,
    threshold : f32,
    fmin : f32,
    fmax : f32,
    win_length : Option<i32>)
    -> Vec<f32>
{
    


    vec![]
}


fn pad(y : &Vec<f32>, padding : usize) -> Vec<f32> {
    let mut result : Vec<f32> = Vec::with_capacity(y.len() + 2*padding);
    result.extend(vec![0.0; padding]);
    result.extend_from_slice(y);
    result.extend(vec![0.0; padding]);
    result    
}


pub fn cumulative_mean_normalized_difference(y_frames : Vec<Vec<f32>>,
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

    #[test]
    fn test_cmnd_shape() {
        let frames = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
        ];

        let min_period = 1;
        let max_period = 3;

        let result = cumulative_mean_normalized_difference(
            frames,
            min_period,
            max_period,
        );

        assert_eq!(result.len(), 2); // n_frames
        assert_eq!(result[0].len(), 3); // max_period - min_period + 1
    }

    #[test]
    fn test_cmnd_constant_signal() {

        let frames = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0]
        ];

        let result = cumulative_mean_normalized_difference(frames, 1, 3);

        for val in &result[0] {
            assert!((*val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cmnd_periodic_signal() 
    {
        let frames = vec![
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        ];

        let result = cumulative_mean_normalized_difference(frames, 1, 4);

        let cmnd = &result[0];

        let min_index = cmnd
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(min_index + 1, 2); // τ = 2
    }

    #[test]
    fn test_cmnd_multiple_frames() 
    {
        let frames = vec![
            vec![1.0, -1.0, 1.0, -1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];

        let result = cumulative_mean_normalized_difference(frames, 1, 3);

        assert_eq!(result.len(), 2);

        // second frame should be constant -> CMND ≈ 1
        for val in &result[1] 
        {
            assert!((*val - 1.0).abs() < 1e-6);
        }
    }

}
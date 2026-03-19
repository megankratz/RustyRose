#[derive(Clone, Debug, PartialEq)]
pub enum Units {
    Frames,
    Samples,
    Time,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OnsetOutput {
    Sparse(Vec<f64>),
    Dense(Vec<bool>),
}

pub fn peak_pick(
    x: &[f64],
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
    delta: f64,
    wait: usize,
) -> Vec<usize> {
    let n = x.len();
    let mut peaks = Vec::new();
    let mut last_onset: isize = -(wait as isize + 1);

    for i in 0..n {
        let max_start = i.saturating_sub(pre_max);
        let max_end = (i + post_max + 1).min(n);
        let local_max = x[max_start..max_end]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let avg_start = i.saturating_sub(pre_avg);
        let avg_end = (i + post_avg + 1).min(n);
        let local_avg = x[avg_start..avg_end].iter().sum::<f64>() / (avg_end - avg_start) as f64;

        let is_max = (x[i] - local_max).abs() < 1e-10;
        let above_threshold = x[i] >= local_avg + delta;
        let waited_long_enough = (i as isize - last_onset) >= wait as isize;

        if is_max && above_threshold && waited_long_enough {
            peaks.push(i);
            last_onset = i as isize;
        }
    }

    peaks
}

pub fn onset_strength(s: &[Vec<f64>], lag: usize) -> Vec<f64> {
    assert!(!s.is_empty(), "spectrogram must not be empty");
    assert!(lag >= 1, "lag must be >= 1");

    let n_frames = s[0].len();
    let n_bins = s.len();

    (0..n_frames)
        .map(|t| {
            let mut total = 0.0f64;
            for bin in 0..n_bins {
                let current = s[bin][t];
                let prev = if t >= lag { s[bin][t - lag] } else { s[bin][0] };
                total += (current - prev).max(0.0);
            }
            total / n_bins as f64
        })
        .collect()
}

pub fn onset_backtrack(events: &[usize], energy: &[f64]) -> Vec<usize> {
    events
        .iter()
        .map(|&onset| {
            let mut i = onset;
            while i > 0 && energy[i - 1] <= energy[i] {
                i -= 1;
            }
            i
        })
        .collect()
}

pub fn onset_detect(
    onset_envelope: &[f64],
    sr: u32,
    hop_length: usize,
    units: Units,
    normalize: bool,
    sparse: bool,
    backtrack: bool,
) -> OnsetOutput {
    assert!(!onset_envelope.is_empty(), "onset_envelope must not be empty");

    let n = onset_envelope.len();

    let envelope: Vec<f64> = if normalize {
        let min = onset_envelope.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = onset_envelope.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range < 1e-10 {
            vec![0.0; n]
        } else {
            onset_envelope.iter().map(|&v| (v - min) / range).collect()
        }
    } else {
        onset_envelope.to_vec()
    };

    let mut peaks = peak_pick(&envelope, 3, 3, 3, 5, 0.07, 10);

    if backtrack {
        peaks = onset_backtrack(&peaks, &envelope);
    }

    if !sparse {
        let mut dense = vec![false; n];
        for &p in &peaks {
            dense[p] = true;
        }
        return OnsetOutput::Dense(dense);
    }

    let converted: Vec<f64> = peaks
        .iter()
        .map(|&frame| match units {
            Units::Frames => frame as f64,
            Units::Samples => (frame * hop_length) as f64,
            Units::Time => (frame * hop_length) as f64 / sr as f64,
        })
        .collect();

    OnsetOutput::Sparse(converted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_pick_finds_clear_peak() {
        let x = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let peaks = peak_pick(&x, 3, 3, 3, 5, 0.07, 10);
        assert!(peaks.contains(&4));
    }

    #[test]
    fn test_peak_pick_flat_signal() {
        let peaks = peak_pick(&vec![0.5; 20], 3, 3, 3, 5, 0.07, 10);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_peak_pick_wait_enforced() {
        let mut x = vec![0.0; 20];
        x[5] = 1.0;
        x[8] = 1.0;
        let peaks = peak_pick(&x, 3, 3, 3, 5, 0.07, 10);
        assert_eq!(peaks, vec![5]);
    }

    #[test]
    fn test_onset_strength_silent() {
        let s = vec![vec![1.0f64; 10]; 16];
        let env = onset_strength(&s, 1);
        for v in &env {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_onset_strength_single_jump() {
        let n_frames = 10;
        let n_bins = 4;
        let mut s = vec![vec![0.0f64; n_frames]; n_bins];
        for bin in 0..n_bins {
            for t in 5..n_frames {
                s[bin][t] = 1.0;
            }
        }
        let env = onset_strength(&s, 1);
        assert!(env[5] > 0.5);
        assert!(env[6].abs() < 1e-10);
    }

    #[test]
    fn test_onset_detect_returns_frames() {
        let mut env = vec![0.0f64; 50];
        env[10] = 1.0;
        env[30] = 1.0;
        let result = onset_detect(&env, 22050, 512, Units::Frames, true, true, false);
        if let OnsetOutput::Sparse(frames) = result {
            assert!(frames.contains(&10.0));
            assert!(frames.contains(&30.0));
        } else {
            panic!("expected sparse output");
        }
    }

    #[test]
    fn test_onset_detect_time_units() {
        let mut env = vec![0.0f64; 50];
        env[10] = 1.0;
        let result = onset_detect(&env, 22050, 512, Units::Time, true, true, false);
        if let OnsetOutput::Sparse(times) = result {
            let expected = (10 * 512) as f64 / 22050.0;
            assert!((times[0] - expected).abs() < 1e-6);
        } else {
            panic!("expected sparse output");
        }
    }

    #[test]
    fn test_onset_detect_dense_output() {
        let mut env = vec![0.0f64; 50];
        env[10] = 1.0;
        let result = onset_detect(&env, 22050, 512, Units::Frames, true, false, false);
        if let OnsetOutput::Dense(bools) = result {
            assert!(bools[10]);
            assert!(!bools[9]);
        } else {
            panic!("expected dense output");
        }
    }

    #[test]
    fn test_onset_detect_flat_no_onsets() {
        let result = onset_detect(&vec![0.5; 50], 22050, 512, Units::Frames, true, true, false);
        if let OnsetOutput::Sparse(frames) = result {
            assert!(frames.is_empty());
        } else {
            panic!("expected sparse output");
        }
    }

    #[test]
    fn test_backtrack_moves_to_minimum() {
        let energy = vec![0.8, 0.6, 0.4, 0.2, 0.5, 1.0, 0.8];
        let bt = onset_backtrack(&[5], &energy);
        assert_eq!(bt[0], 3);
    }

    #[test]
    fn test_backtrack_already_at_minimum() {
        let energy = vec![1.0, 0.1, 0.9, 0.8];
        let bt = onset_backtrack(&[1], &energy);
        assert_eq!(bt[0], 1);
    }
}
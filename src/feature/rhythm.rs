use ndarray::{Array1, Array2, Axis};

/// Compute the tempogram: local autocorrelation of the onset strength envelope.
/// `onset_envelope` is expected to be a 1D sequence of onset strengths.
pub fn tempogram(onset_envelope: &Array1<f64>, win_length: usize, hop_length: usize) -> Array2<f64> {
    let n = onset_envelope.len();
    let pad_len = win_length / 2;
    
    let padded_len = n + 2 * pad_len;
    let mut padded = vec![0.0; padded_len];
    for i in 0..n { padded[pad_len + i] = onset_envelope[i]; }
    
    let t = if padded_len > win_length { (padded_len - win_length) / hop_length + 1 } else { 1 };
    
    let mut out = Array2::<f64>::zeros((win_length, t));
    
    for i in 0..t {
        let start = i * hop_length;
        let frame = &padded[start .. start + win_length];
        
        for lag in 0..win_length {
            let mut sum = 0.0;
            for j in 0..win_length - lag {
                sum += frame[j] * frame[j + lag];
            }
            out[[lag, i]] = sum;
        }
    }
    
    // Normalize each frame by its max
    for mut col in out.axis_iter_mut(Axis(1)) {
        let max_val = col.iter().fold(0.0f64, |m, &v| m.max(v));
        if max_val > 1e-10 {
            col.mapv_inplace(|x| x / max_val);
        }
    }
    
    out
}

/// Compute the Fourier tempogram: the short-time Fourier transform of the
/// onset strength envelope.
/// We use a basic Discrete Fourier Transform (DFT) implementation to avoid dragging heavy FFT libraries.
pub fn fourier_tempogram(onset_envelope: &Array1<f64>, win_length: usize, hop_length: usize) -> Array2<f64> {
    let n = onset_envelope.len();
    let pad_len = win_length / 2;
    
    let padded_len = n + 2 * pad_len;
    let mut padded = vec![0.0; padded_len];
    for i in 0..n { padded[pad_len + i] = onset_envelope[i]; }
    
    let t = if padded_len > win_length { (padded_len - win_length) / hop_length + 1 } else { 1 };
    let n_freqs = win_length / 2 + 1;
    
    // We store the magnitude spectrogram
    let mut out = Array2::<f64>::zeros((n_freqs, t));
    
    for i in 0..t {
        let start = i * hop_length;
        let frame = &padded[start .. start + win_length];
        
        // Basic DFT Magnitude
        for k in 0..n_freqs {
            let mut re = 0.0;
            let mut im = 0.0;
            for j in 0..win_length {
                // Hann window
                let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (j as f64) / ((win_length - 1) as f64)).cos());
                let val = frame[j] * w;
                
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (win_length as f64);
                re += val * angle.cos();
                im += val * angle.sin();
            }
            out[[k, i]] = (re*re + im*im).sqrt();
        }
    }
    
    out
}

/// Estimate the tempo (beats per minute)
/// `tempogram` is a 2D Array computed previously.
pub fn tempo(tempogram: &Array2<f64>, sr: f64, hop_length: usize) -> f64 {
    let mut global_onset = Array1::<f64>::zeros(tempogram.nrows());
    for i in 0..tempogram.nrows() {
        let mut sum = 0.0;
        for j in 0..tempogram.ncols() {
            sum += tempogram[[i, j]];
        }
        global_onset[i] = sum / (tempogram.ncols() as f64);
    }
    
    let mut best_bpm = 0.0;
    let mut max_score = -1.0;
    
    for lag in 1..tempogram.nrows() { 
        let bpm = 60.0 * sr / (hop_length as f64 * lag as f64);
        
        // Ignore physically unreasonable bpms
        if bpm < 20.0 || bpm > 300.0 { continue; }
        
        // Lognormal weighting, center around standard 120.0
        let weight = (-((bpm.ln() - 120.0f64.ln()).powi(2)) / (2.0 * 0.5f64.powi(2))).exp();
        let score = global_onset[lag] * weight;
        
        if score > max_score {
            max_score = score;
            best_bpm = bpm;
        }
    }
    
    // Fallback if no valid bpm found
    if best_bpm == 0.0 { return 120.0; }
    
    best_bpm
}

/// Tempogram ratio features.
pub fn tempogram_ratio(tempogram: &Array2<f64>) -> Array2<f64> {
    // Advanced scipy interpolation ratio summary.
    // For scaffolding, this returns the unmodified tempogram ratios against max peak
    let mut out = tempogram.clone();
    for mut col in out.axis_iter_mut(Axis(1)) {
        let max_val = col.iter().fold(0.0f64, |m, &v| m.max(v));
        if max_val > 1e-10 {
            col.mapv_inplace(|x| x / max_val);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_tempo_parametric() {
        // test_tempo port from test_beat.py:test_tempo
        let tempos: [f64; 4] = [60.0, 80.0, 110.0, 160.0];
        let srs: [f64; 2] = [22050.0, 44100.0];
        let hop_lengths: [usize; 2] = [512, 1024];

        for &target_tempo in &tempos {
            for &sr in &srs {
                for &hop_length in &hop_lengths {
                    let duration = 20.0;
                    let n_samples = (duration * sr) as usize;
                    let mut y = Array1::<f64>::zeros(n_samples);
                    
                    let delay = ((60.0_f64 / target_tempo) * sr).round() as usize;
                    if delay == 0 { continue; }
                    
                    for i in (0..n_samples).step_by(delay) {
                        y[i] = 1.0;
                    }
                    
                    let n_frames = n_samples / hop_length;
                    let mut oenv = Array1::<f64>::zeros(n_frames);
                    for i in 0..n_frames {
                        let start = i * hop_length;
                        let end = start + hop_length;
                        let mut sum = 0.0;
                        for j in start..end {
                            if j < n_samples {
                                sum += y[j];
                            }
                        }
                        oenv[i] = sum;
                    }
                    
                    let tgram = tempogram(&oenv, 384, 1);
                    let est_tempo = tempo(&tgram, sr, hop_length);
                    
                    // assert within ~5-10% bounds
                    assert!((est_tempo - target_tempo).abs() <= 0.10 * target_tempo, 
                            "Failed tempo estimation: target {}, got {}", target_tempo, est_tempo);
                }
            }
        }
    }

    #[test]
    fn test_tempogram_odf_peak() {
        // test_tempogram_odf_peak port from test_features.py
        let tempos: [f64; 3] = [60.0, 90.0, 200.0];
        let win_lengths: [usize; 2] = [192, 384];
        let sr = 22050.0;
        let hop_length = 512;
        let duration = 8.0;
        
        for &tempo in &tempos {
            for &win_length in &win_lengths {
                let n_frames = (duration * sr / hop_length as f64) as usize;
                let mut odf = Array1::<f64>::zeros(n_frames);
                let spacing = (sr * 60.0 / (hop_length as f64 * tempo)).round() as usize;
                
                for i in (0..n_frames).step_by(spacing) {
                    odf[i] = 1.0;
                }
                
                let tempogram_out = tempogram(&odf, win_length, 1);
                assert_eq!(tempogram_out.nrows(), win_length);
                
                // The peak lag should be exactly `spacing` or a non-zero integer multiple.
                // We check that the highest mean lag matches spacing
                let mut max_mean = -1.0;
                let mut best_lag = 0;
                for lag in 1..win_length {
                    let mut sum = 0.0;
                    for t in 0..tempogram_out.ncols() {
                        sum += tempogram_out[[lag, t]];
                    }
                    if sum > max_mean {
                        max_mean = sum;
                        best_lag = lag;
                    }
                }
                
                assert!(best_lag == spacing || best_lag % spacing == 0);
            }
        }
    }

    #[test]
    fn test_fourier_tempogram() {
        let mut env = Array1::<f64>::zeros(50);
        env[10] = 1.0;
        
        let ftgram = fourier_tempogram(&env, 16, 5);
        assert_eq!(ftgram.nrows(), 9); // 16/2 + 1
        assert!(ftgram.ncols() > 0);
    }
}

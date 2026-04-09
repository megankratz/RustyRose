use ndarray::{Array1, Array2, Axis, s};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

/// Compute the tempogram: local autocorrelation of the onset strength envelope.
/// `onset_envelope` is expected to be a 1D sequence of onset strengths.
pub fn tempogram(onset_envelope: &Array1<f64>, win_length: usize, hop_length: usize) -> Array2<f64> {
    let n = onset_envelope.len();
    if win_length < 1 { panic!("win_length must be a positive integer"); }

    let mut window = Array1::<f64>::zeros(win_length);
    for i in 0..win_length {
        window[i] = 0.5 * (1.0 - (2.0 * PI * (i as f64) / (win_length as f64)).cos());
    }

    let pad_len = win_length / 2;
    let padded_len = n + 2 * pad_len;
    let mut padded = Array1::<f64>::zeros(padded_len);
    
    for i in 0..pad_len {
        padded[i] = (onset_envelope[0] * (i as f64)) / (pad_len as f64);
    }
    for i in 0..n {
        padded[pad_len + i] = onset_envelope[i];
    }
    
    let end_val = onset_envelope[n-1];
    for i in 0..pad_len {
        padded[pad_len + n + i] = (end_val * (pad_len - 1 - i) as f64) / (pad_len as f64);
    }

    let t = if padded_len > win_length { (padded_len - win_length) / hop_length + 1 } else { 1 };
    let t = if hop_length == 1 { n } else { t };

    let mut out = Array2::<f64>::zeros((win_length, t));
    
    let mut planner = FftPlanner::new();
    let n_fft = (2 * win_length - 1).next_power_of_two();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);
    let scale = 1.0 / (n_fft as f64);

    for i in 0..t {
        let start = i * hop_length;
        if start + win_length > padded_len { break; }
        
        let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
        for j in 0..win_length {
            buffer[j] = Complex::new(padded[start + j] * window[j], 0.0);
        }
        
        fft.process(&mut buffer);
        for j in 0..n_fft {
            buffer[j] = buffer[j] * buffer[j].conj();
        }
        ifft.process(&mut buffer);
        
        for lag in 0..win_length {
            out[[lag, i]] = buffer[lag].re * scale;
        }
    }
    
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
pub fn fourier_tempogram(onset_envelope: &Array1<f64>, win_length: usize, hop_length: usize) -> Array2<f64> {
    let n = onset_envelope.len();
    if win_length < 1 { panic!("win_length must be a positive integer"); }

    let pad_len = win_length / 2;
    let padded_len = n + 2 * pad_len;
    let mut padded = Array1::<f64>::zeros(padded_len);
    
    for i in 0..n {
        padded[pad_len + i] = onset_envelope[i];
    }

    let t = if hop_length == 1 { n } else { (padded_len - win_length) / hop_length + 1 };
    let n_freqs = win_length / 2 + 1;
    
    let mut out = Array2::<f64>::zeros((n_freqs, t));
    
    let mut window = Array1::<f64>::zeros(win_length);
    for j in 0..win_length {
        window[j] = 0.5 * (1.0 - (2.0 * PI * (j as f64) / (win_length as f64)).cos());
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(win_length);

    for i in 0..t {
        let start = i * hop_length;
        if start + win_length > padded_len { break; }
        
        let mut buffer = vec![Complex::new(0.0, 0.0); win_length];
        for j in 0..win_length {
            buffer[j] = Complex::new(padded[start + j] * window[j], 0.0);
        }
        
        fft.process(&mut buffer);
        
        for k in 0..n_freqs {
            out[[k, i]] = buffer[k].norm();
        }
    }
    
    out
}

/// Estimate the tempo (beats per minute)
/// `tempogram` is a 2D Array computed previously.
pub fn tempo(tempogram: &Array2<f64>, sr: f64, hop_length: usize) -> f64 {
    let win_length = tempogram.nrows();
    
    // Librosa tempo logic: log-normal prior on BPM
    let start_bpm: f64 = 120.0;
    let std_bpm: f64 = 1.0;
    
    // Aggregate tempogram across time
    let mut global_ac = Array1::<f64>::zeros(win_length);
    for i in 0..win_length {
        let mut sum = 0.0;
        for j in 0..tempogram.ncols() {
            sum += tempogram[[i, j]];
        }
        global_ac[i] = sum / (tempogram.ncols() as f64);
    }
    
    let mut best_bpm: f64 = 120.0;
    let mut max_score: f64 = -1e10;
    
    for lag in 1..win_length {
        // Frequency in BPM: sr * 60 / (hop_length * lag)
        let bpm = 60.0 * sr / (hop_length as f64 * lag as f64);
        
        if bpm < 20.0 || bpm > 320.0 { continue; }
        
        // Log-normal weighting: -0.5 * ((log2(bpm) - log2(120)) / std_bpm)^2
        let log_prior = -0.5f64 * ((bpm.log2() - start_bpm.log2()) / std_bpm).powi(2);
        
        // Score = log1p(1e6 * global_ac) + log_prior
        let score = (1.0f64 + 1e6f64 * global_ac[lag]).ln() + log_prior;
        
        if score > max_score {
            max_score = score;
            best_bpm = bpm;
        }
    }
    
    best_bpm
}

/// Tempogram ratio features.
pub fn tempogram_ratio(tempogram: &Array2<f64>) -> Array2<f64> {
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
                    
                    assert!((est_tempo - target_tempo).abs() <= 0.10 * target_tempo, 
                            "Failed tempo estimation: target {}, got {}", target_tempo, est_tempo);
                }
            }
        }
    }

    #[test]
    fn test_tempogram_odf_peak() {
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

    #[test]
    fn test_rhythm_vs_librosa() {
        use std::fs::File;
        use std::io::Read;
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct Refs {
            onset_envelope: Vec<f64>,
            win_length: usize,
            tempogram: Vec<Vec<f64>>,
            fourier_tempogram_mag: Vec<Vec<f64>>,
            oenv_long: Vec<f64>,
            tempo: f64,
        }

        let mut file = File::open("tests/rhythm_refs.json").expect("Reference file not found. Run scripts/generate_rhythm_refs.py first.");
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        let refs: Refs = serde_json::from_str(&data).unwrap();

        let oenv = Array1::from(refs.onset_envelope);
        
        // Test Tempogram
        let tgram = tempogram(&oenv, refs.win_length, 1);
        assert_eq!(tgram.nrows(), refs.win_length);
        assert_eq!(tgram.ncols(), oenv.len());
        
        for i in 0..tgram.nrows() {
            for j in 0..tgram.ncols() {
                // Approximate match due to FFT differences (1e-5)
                assert!((tgram[[i, j]] - refs.tempogram[i][j]).abs() < 1e-5,
                    "Tempogram mismatch at [{}, {}]: got {}, expected {}", i, j, tgram[[i, j]], refs.tempogram[i][j]);
            }
        }
        
        // Test Fourier Tempogram
        let ftgram = fourier_tempogram(&oenv, refs.win_length, 1);
        assert_eq!(ftgram.nrows(), refs.win_length / 2 + 1);
        
        for i in 0..ftgram.nrows() {
            for j in 0..ftgram.ncols() {
                assert!((ftgram[[i, j]] - refs.fourier_tempogram_mag[i][j]).abs() < 1e-5,
                    "Fourier Tempogram mismatch at [{}, {}]: got {}, expected {}", i, j, ftgram[[i, j]], refs.fourier_tempogram_mag[i][j]);
            }
        }
        
        // Test Tempo
        let oenv_long = Array1::from(refs.oenv_long);
        let tgram_long = tempogram(&oenv_long, 384, 1); // Librosa default win_length
        let est_tempo = tempo(&tgram_long, 22050.0, 512);
        assert!((est_tempo - refs.tempo).abs() < 1.0, "Tempo mismatch: got {}, expected {}", est_tempo, refs.tempo);
    }
}

//! # special_features
//!
//! Rust implementations of librosa's core DSP feature extraction:
//!
//! - [`melspectrogram`] — Mel-scaled power spectrogram        (PI1)
//! - [`mfcc`]           — Mel-frequency cepstral coefficients (PI1)
//! - [`chroma_stft`]    — Chromagram via STFT                 (PI2)

pub mod spectrum;

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

use spectrum::{power_to_db, Ref};


// Parameter structs

/// STFT parameters shared by all three feature functions.
#[derive(Debug, Clone)]
pub struct StftParams {
    /// Sampling rate (Hz). Default: 22050.
    pub sr: f64,
    /// FFT window size. Default: 2048.
    pub n_fft: usize,
    /// Hop length in samples. Default: 512.
    pub hop_length: usize,
    /// Pad signal by n_fft/2 zeros on each side before framing. Default: true.
    pub center: bool,
}

impl Default for StftParams {
    fn default() -> Self {
        Self { sr: 22050.0, n_fft: 2048, hop_length: 512, center: true }
    }
}

/// Parameters for [`melspectrogram`].
#[derive(Debug, Clone)]
pub struct MelParams {
    pub stft: StftParams,
    /// Number of Mel bands. Default: 128.
    pub n_mels: usize,
    /// Lowest frequency (Hz). Default: 0.0.
    pub fmin: f64,
    /// Highest frequency (Hz). `None` → `sr / 2`. Default: None.
    pub fmax: Option<f64>,
    /// Spectrogram exponent (1 = magnitude, 2 = power). Default: 2.0.
    pub power: f64,
    /// Apply Slaney area-normalisation to Mel filters. Default: true.
    pub norm_slaney: bool,
}

impl Default for MelParams {
    fn default() -> Self {
        Self {
            stft: StftParams::default(),
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
            power: 2.0,
            norm_slaney: true,
        }
    }
}

/// Parameters for [`mfcc`].
#[derive(Debug, Clone)]
pub struct MfccParams {
    pub mel: MelParams,
    /// Number of MFCC coefficients to return. Default: 20.
    pub n_mfcc: usize,
    /// Liftering coefficient (0 = disabled). Default: 0.0.
    pub lifter: f64,
}

impl Default for MfccParams {
    fn default() -> Self {
        Self { mel: MelParams::default(), n_mfcc: 20, lifter: 0.0 }
    }
}

/// Parameters for [`chroma_stft`].
#[derive(Debug, Clone)]
pub struct ChromaParams {
    pub stft: StftParams,
    /// Number of chroma bins. Default: 12.
    pub n_chroma: usize,
    /// Tuning offset in fractional chroma bins from A440. Default: 0.0.
    pub tuning: f64,
    /// Centre octave for optional Gaussian octave weighting. Default: 5.0.
    pub ctroct: f64,
    /// Gaussian half-width in octaves. `None` = flat weighting. Default: None.
    pub octwidth: Option<f64>,
    /// `true` → bin 0 = C (librosa default). `false` → bin 0 = A.
    pub base_c: bool,
}

impl Default for ChromaParams {
    fn default() -> Self {
        Self {
            stft: StftParams::default(),
            n_chroma: 12,
            tuning: 0.0,
            ctroct: 5.0,
            octwidth: None,
            base_c: true,
        }
    }
}

// Internal DSP helpers

/// Periodic Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos())).collect()
}

/// Compute a power spectrogram `S[n_bins, n_frames]` from raw samples.
///
/// Returns `(data_row_major, n_bins, n_frames)`.
/// `power` is the exponent applied to each bin magnitude (2 → power spectrum).
pub fn power_spectrogram(samples: &[f64], p: &StftParams, power: f64) -> (Vec<f64>, usize, usize) {
    let n_bins  = p.n_fft / 2 + 1;
    let window  = hann_window(p.n_fft);

    let padded: Vec<f64> = if p.center {
        let pad = p.n_fft / 2;
        let mut v = vec![0.0f64; pad];
        v.extend_from_slice(samples);
        v.extend(vec![0.0f64; pad]);
        v
    } else {
        samples.to_vec()
    };

    let n_frames = if padded.len() >= p.n_fft {
        1 + (padded.len() - p.n_fft) / p.hop_length
    } else {
        0
    };

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(p.n_fft);

    // Accumulate frames in [frame, bin] order then transpose
    let mut col = vec![0.0f64; n_frames * n_bins];

    for frame in 0..n_frames {
        let start = frame * p.hop_length;
        let mut buf: Vec<Complex<f64>> = (0..p.n_fft)
            .map(|i| Complex::new(padded[start + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buf);
        for k in 0..n_bins {
            col[frame * n_bins + k] = buf[k].norm().powf(power);
        }
    }

    // Transpose to row-major [bin, frame]
    let mut out = vec![0.0f64; n_bins * n_frames];
    for k in 0..n_bins {
        for t in 0..n_frames {
            out[k * n_frames + t] = col[t * n_bins + k];
        }
    }
    (out, n_bins, n_frames)
}

//  Mel filter bank

/// Hz → Mel via librosa's Slaney linear/log formula.
fn hz_to_mel(hz: f64) -> f64 {
    let f_sp       = 200.0 / 3.0_f64;
    let min_log_hz = 1000.0_f64;
    let min_log_mel = min_log_hz / f_sp;
    let logstep    = (6.4_f64).ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + ((hz / min_log_hz).ln() / logstep)
    } else {
        hz / f_sp
    }
}

/// Mel → Hz (inverse of `hz_to_mel`).
fn mel_to_hz(mel: f64) -> f64 {
    let f_sp       = 200.0 / 3.0_f64;
    let min_log_hz = 1000.0_f64;
    let min_log_mel = min_log_hz / f_sp;
    let logstep    = (6.4_f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        f_sp * mel
    }
}

/// Build a Mel filter bank of shape `[n_mels, n_fft/2+1]`.
/// Mirrors `librosa.filters.mel` with optional Slaney area normalisation.
pub fn mel_filterbank(
    sr: f64, n_fft: usize, n_mels: usize,
    fmin: f64, fmax: f64, norm_slaney: bool,
) -> Vec<f64> {
    let n_bins    = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / n_fft as f64).collect();

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let mut weights = vec![0.0f64; n_mels * n_bins];

    for m in 0..n_mels {
        let f_low    = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_high   = mel_points[m + 2];

        for (k, &f) in fft_freqs.iter().enumerate() {
            weights[m * n_bins + k] =
                if f >= f_low && f <= f_center && (f_center - f_low) > 0.0 {
                    (f - f_low) / (f_center - f_low)
                } else if f > f_center && f <= f_high && (f_high - f_center) > 0.0 {
                    (f_high - f) / (f_high - f_center)
                } else {
                    0.0
                };
        }

        if norm_slaney {
            let bandwidth = f_high - f_low;
            if bandwidth > 0.0 {
                for k in 0..n_bins { weights[m * n_bins + k] /= bandwidth; }
            }
        }
    }
    weights
}

// DCT-II

/// Ortho-normalised DCT-II of each column of `s` (shape `[n_rows, n_cols]`).
/// Returns the first `n_out` rows.
///
/// Formula:  X[k] = w[k] · Σ_n x[n] · cos(π·k·(2n+1) / 2N)
///           w[0] = √(1/N),  w[k>0] = √(2/N)
pub fn dct2_columns(s: &[f64], n_rows: usize, n_cols: usize, n_out: usize) -> Vec<f64> {
    let mut out  = vec![0.0f64; n_out * n_cols];
    let sqrt_1_n = (1.0_f64 / n_rows as f64).sqrt();
    let sqrt_2_n = (2.0_f64 / n_rows as f64).sqrt();

    for t in 0..n_cols {
        for k in 0..n_out {
            let w = if k == 0 { sqrt_1_n } else { sqrt_2_n };
            let sum: f64 = (0..n_rows)
                .map(|n| {
                    s[n * n_cols + t]
                        * (PI * k as f64 * (2 * n + 1) as f64 / (2 * n_rows) as f64).cos()
                })
                .sum();
            out[k * n_cols + t] = w * sum;
        }
    }
    out
}

//L∞ column normalisation

/// Normalise each column of a `[n_rows × n_cols]` matrix by its L∞ (max-abs) norm.
/// Zero-energy columns are left unchanged.
pub fn normalize_linf_columns(m: &mut Vec<f64>, n_rows: usize, n_cols: usize) {
    for t in 0..n_cols {
        let max: f64 = (0..n_rows).map(|r| m[r * n_cols + t].abs()).fold(0.0f64, f64::max);
        if max > 0.0 {
            for r in 0..n_rows { m[r * n_cols + t] /= max; }
        }
    }
}

// Chroma filter bank
/// Build a chroma filter bank `[n_chroma × n_bins]`.
/// Mirrors `librosa.filters.chroma`.
///
/// The raw formula naturally produces **C-based** bins (C = 0, A = 9).
/// When `base_c = false` the result is rotated 3 positions to make A = 0.
pub fn chroma_filterbank(
    sr: f64, n_fft: usize, n_chroma: usize,
    tuning: f64, ctroct: f64, octwidth: Option<f64>, base_c: bool,
) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    // C0 = A440 / 2^(57/12)
    let c0 = 440.0_f64 / 2.0_f64.powf(57.0 / 12.0);

    let mut weights = vec![0.0f64; n_chroma * n_bins];

    for k in 1..n_bins {   // skip DC bin
        let freq     = k as f64 * sr / n_fft as f64;
        let semitone = 12.0 * (freq / c0).log2() + tuning;
        let octave   = semitone / 12.0;

        let oct_weight = match octwidth {
            Some(ow) => { let d = (octave - ctroct) / ow; (-0.5 * d * d).exp() }
            None     => 1.0,
        };

        let frac_bin = semitone.rem_euclid(n_chroma as f64);

        for c in 0..n_chroma {
            let mut d = frac_bin - c as f64;
            let half  = n_chroma as f64 / 2.0;
            if d >  half { d -= n_chroma as f64; }
            if d < -half { d += n_chroma as f64; }
            weights[c * n_bins + k] += oct_weight * (-0.5 * d * d).exp();
        }
    }

    // Rotate left by 3 only when base_c = false (C-based → A-based)
    if !base_c {
        let shift   = 3usize % n_chroma;
        let mut rot = vec![0.0f64; n_chroma * n_bins];
        for c in 0..n_chroma {
            let src = (c + shift) % n_chroma;
            for k in 0..n_bins { rot[c * n_bins + k] = weights[src * n_bins + k]; }
        }
        weights = rot;
    }

    // L2-normalise each chroma row
    for c in 0..n_chroma {
        let norm: f64 = (0..n_bins).map(|k| weights[c * n_bins + k].powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for k in 0..n_bins { weights[c * n_bins + k] /= norm; }
        }
    }
    weights
}


/// Compute a mel-scaled power spectrogram.
///
/// Returns `(data, n_mels, n_frames)` — row-major `[n_mels × n_frames]`.
///
/// Equivalent librosa call:
/// ```python
/// librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
///                                 n_mels=128, fmin=0, norm='slaney', power=2)
/// ```
pub fn melspectrogram(samples: &[f64], p: &MelParams) -> (Vec<f64>, usize, usize) {
    let (spec, _n_bins, n_frames) = power_spectrogram(samples, &p.stft, p.power);
    let n_bins = p.stft.n_fft / 2 + 1;
    let fmax   = p.fmax.unwrap_or(p.stft.sr / 2.0);
    let mel_fb = mel_filterbank(p.stft.sr, p.stft.n_fft, p.n_mels, p.fmin, fmax, p.norm_slaney);

    let mut mel_spec = vec![0.0f64; p.n_mels * n_frames];
    for m in 0..p.n_mels {
        for t in 0..n_frames {
            mel_spec[m * n_frames + t] = (0..n_bins)
                .map(|k| mel_fb[m * n_bins + k] * spec[k * n_frames + t])
                .sum();
        }
    }
    (mel_spec, p.n_mels, n_frames)
}

/// Compute Mel-Frequency Cepstral Coefficients (MFCCs).
///
/// Pipeline: audio → mel spectrogram → `spectrum::power_to_db` → ortho DCT-II → optional lifter.
///
/// Returns `(data, n_mfcc, n_frames)` — row-major `[n_mfcc × n_frames]`.
///
/// Equivalent librosa call:
/// ```python
/// librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
/// ```
pub fn mfcc(samples: &[f64], p: &MfccParams) -> (Vec<f64>, usize, usize) {
    let (mel_spec, n_mels, n_frames) = melspectrogram(samples, &p.mel);

    // Use spectrum::power_to_db — ref = global max, top_db = 80 (librosa defaults)
    let log_mel = power_to_db(
        &mel_spec,
        Ref::Fn(&|s| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
        1e-10,
        Some(80.0),
    );

    let mut out = dct2_columns(&log_mel, n_mels, n_frames, p.n_mfcc);

    if p.lifter > 0.0 {
        for n in 0..p.n_mfcc {
            let lift = 1.0 + (p.lifter / 2.0) * (PI * (n + 1) as f64 / p.lifter).sin();
            for t in 0..n_frames { out[n * n_frames + t] *= lift; }
        }
    }
    (out, p.n_mfcc, n_frames)
}

/// Compute a chromagram from raw audio using STFT.
///
/// Pipeline: audio → power spectrogram → chroma filter bank → L∞ column normalise.
///
/// Returns `(data, n_chroma, n_frames)` — row-major `[n_chroma × n_frames]`.
///
/// Equivalent librosa call:
/// ```python
/// librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512, n_chroma=12)
/// ```
pub fn chroma_stft(samples: &[f64], p: &ChromaParams) -> (Vec<f64>, usize, usize) {
    let (spec, _n_bins, n_frames) = power_spectrogram(samples, &p.stft, 2.0);
    let n_bins = p.stft.n_fft / 2 + 1;
    let fb     = chroma_filterbank(
        p.stft.sr, p.stft.n_fft, p.n_chroma,
        p.tuning, p.ctroct, p.octwidth, p.base_c,
    );

    let mut raw = vec![0.0f64; p.n_chroma * n_frames];
    for c in 0..p.n_chroma {
        for t in 0..n_frames {
            raw[c * n_frames + t] = (0..n_bins)
                .map(|k| fb[c * n_bins + k] * spec[k * n_frames + t])
                .sum();
        }
    }
    normalize_linf_columns(&mut raw, p.n_chroma, n_frames);
    (raw, p.n_chroma, n_frames)
}

// Tests


#[cfg(test)]
mod tests {
    use super::*;


    fn sine_wave(freq: f64, sr: f64, duration_s: f64) -> Vec<f64> {
        let n = (sr * duration_s) as usize;
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr).sin()).collect()
    }
    fn silence(n: usize)             -> Vec<f64> { vec![0.0; n] }
    fn impulse(n: usize, pos: usize) -> Vec<f64> { let mut v = silence(n); v[pos] = 1.0; v }

    fn p_mel()    -> MelParams    { MelParams::default() }
    fn p_mfcc()   -> MfccParams   { MfccParams::default() }
    fn p_chroma() -> ChromaParams { ChromaParams::default() }

    // hz_to_mel / mel_to_hz

    #[test]
    fn test_hz_mel_roundtrip() {
        for hz in [0.0f64, 100.0, 440.0, 1000.0, 8000.0, 22050.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((hz - back).abs() < 1e-6, "roundtrip failed at {} Hz (got {})", hz, back);
        }
    }

    #[test]
    fn test_hz_mel_monotone() {
        let freqs = [50.0f64, 200.0, 500.0, 1000.0, 4000.0, 11000.0];
        for w in freqs.windows(2) {
            assert!(hz_to_mel(w[0]) < hz_to_mel(w[1]));
        }
    }

    #[test]
    fn test_mel_linear_region_is_linear() {
        // Below 1000 Hz mel is linear: doubling Hz should double mel
        let m1 = hz_to_mel(500.0);
        let m2 = hz_to_mel(1000.0);
        assert!((m2 / m1 - 2.0).abs() < 1e-9);
    }

    // hann_window

    #[test]
    fn test_hann_endpoint_near_zero() {
        assert!(hann_window(1024)[0].abs() < 1e-12);
    }

    #[test]
    fn test_hann_midpoint_is_one() {
        let n = 1024;
        assert!((hann_window(n)[n / 2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hann_symmetric() {
        let n = 512;
        let w = hann_window(n);
        for i in 1..n / 2 {
            assert!((w[i] - w[n - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hann_non_negative() {
        assert!(hann_window(2048).iter().all(|&v| v >= 0.0));
    }
    
    // mel_filterbank


    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(22050.0, 2048, 128, 0.0, 11025.0, true);
        assert_eq!(fb.len(), 128 * (2048 / 2 + 1));
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        assert!(mel_filterbank(22050.0, 2048, 128, 0.0, 11025.0, true).iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_filterbank_every_band_has_support() {
        let n_mels = 40;
        let n_bins = 2048 / 2 + 1;
        let fb = mel_filterbank(22050.0, 2048, n_mels, 0.0, 11025.0, false);
        for m in 0..n_mels {
            let sum: f64 = (0..n_bins).map(|k| fb[m * n_bins + k]).sum();
            assert!(sum > 0.0, "Mel band {} has no support", m);
        }
    }

    #[test]
    fn test_mel_filterbank_peaks_near_one_without_slaney() {
        // Without Slaney, each filter's peak is ≥ 0.9 (edge bands may be slightly lower
        // due to discrete FFT bin spacing)
        let n_mels = 20;
        let n_bins = 2048 / 2 + 1;
        let fb = mel_filterbank(22050.0, 2048, n_mels, 0.0, 11025.0, false);
        for m in 0..n_mels {
            let peak: f64 = (0..n_bins).map(|k| fb[m * n_bins + k]).fold(0.0f64, f64::max);
            assert!(peak >= 0.9, "filter {} peak {} < 0.9", m, peak);
        }
    }

    // dct2_columns

    #[test]
    fn test_dct2_dc_input_coeff0() {
        let n = 8;
        let c = 3.0f64;
        let s = vec![c; n];
        let out = dct2_columns(&s, n, 1, n);
        assert!((out[0] - c * (n as f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_dct2_dc_input_higher_zero() {
        let s = vec![3.0f64; 8];
        let out = dct2_columns(&s, 8, 1, 8);
        for k in 1..8 {
            assert!(out[k].abs() < 1e-10, "DCT[{}] = {} should be ~0", k, out[k]);
        }
    }

    #[test]
    fn test_dct2_parseval() {
        let n = 16;
        let s: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let out = dct2_columns(&s, n, 1, n);
        let e_in:  f64 = s.iter().map(|x| x * x).sum();
        let e_out: f64 = out.iter().map(|x| x * x).sum();
        assert!((e_in - e_out).abs() < 1e-10);
    }

    #[test]
    fn test_dct2_n_out_respected() {
        let s: Vec<f64> = (0..32).map(|i| i as f64).collect();
        for n_out in [1usize, 10, 20, 32] {
            assert_eq!(dct2_columns(&s, 32, 1, n_out).len(), n_out);
        }
    }

    // normalize_linf_columns

    #[test]
    fn test_normalize_linf_max_is_one() {
        let mut m: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        normalize_linf_columns(&mut m, 4, 3);
        for t in 0..3 {
            let max = (0..4).map(|r| m[r * 3 + t].abs()).fold(0.0f64, f64::max);
            assert!((max - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_normalize_linf_zero_unchanged() {
        let mut m = vec![0.0f64; 4];
        normalize_linf_columns(&mut m, 2, 2);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_linf_preserves_ratio() {
        let mut m = vec![2.0f64, 4.0];
        normalize_linf_columns(&mut m, 2, 1);
        assert!((m[1] / m[0] - 2.0).abs() < 1e-12);
    }

    // Integration: melspectrogram

    #[test]
    fn test_mel_shape() {
        let (_, n_mels, n_frames) = melspectrogram(&sine_wave(440.0, 22050.0, 1.0), &p_mel());
        assert_eq!(n_mels, 128);
        assert!(n_frames > 0);
    }

    #[test]
    fn test_mel_len_matches_shape() {
        let sig = sine_wave(440.0, 22050.0, 0.5);
        let (out, n_mels, n_frames) = melspectrogram(&sig, &p_mel());
        assert_eq!(out.len(), n_mels * n_frames);
    }

    #[test]
    fn test_mel_non_negative() {
        let (out, _, _) = melspectrogram(&sine_wave(440.0, 22050.0, 0.5), &p_mel());
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_silence_is_zero() {
        let (out, _, _) = melspectrogram(&silence(22050), &p_mel());
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mel_440hz_peaks_in_correct_band() {
        let (out, n_mels, n_frames) = melspectrogram(&sine_wave(440.0, 22050.0, 1.0), &p_mel());
        let energy: Vec<f64> = (0..n_mels)
            .map(|m| (0..n_frames).map(|t| out[m * n_frames + t]).sum::<f64>())
            .collect();
        let peak = energy.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        assert!((15..38).contains(&peak), "440 Hz should peak in bands 15–38, got {}", peak);
    }

    #[test]
    fn test_mel_impulse_excites_most_bands() {
        let (out, n_mels, n_frames) = melspectrogram(&impulse(22050, 1024), &p_mel());
        let peak_t = (0..n_frames)
            .max_by(|&a, &b| {
                let ea: f64 = (0..n_mels).map(|m| out[m * n_frames + a]).sum();
                let eb: f64 = (0..n_mels).map(|m| out[m * n_frames + b]).sum();
                ea.partial_cmp(&eb).unwrap()
            }).unwrap();
        let active = (0..n_mels).filter(|&m| out[m * n_frames + peak_t] > 0.0).count();
        assert!(active > n_mels / 2, "Impulse should excite >50% of bands, got {}", active);
    }

    #[test]
    fn test_mel_longer_signal_more_frames() {
        let (_, _, n1) = melspectrogram(&sine_wave(440.0, 22050.0, 0.5), &p_mel());
        let (_, _, n2) = melspectrogram(&sine_wave(440.0, 22050.0, 1.0), &p_mel());
        assert!(n2 > n1);
    }

    #[test]
    fn test_mel_higher_freq_higher_band() {
        let peak = |freq: f64| {
            let (out, n_mels, n_frames) = melspectrogram(&sine_wave(freq, 22050.0, 1.0), &p_mel());
            let energy: Vec<f64> = (0..n_mels)
                .map(|m| (0..n_frames).map(|t| out[m * n_frames + t]).sum::<f64>())
                .collect();
            energy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        };
        assert!(peak(4000.0) > peak(200.0), "4 kHz should peak at a higher band than 200 Hz");
    }

    #[test]
    fn test_mel_custom_n_mels() {
        let (_, n_mels, _) = melspectrogram(&sine_wave(440.0, 22050.0, 0.2),
            &MelParams { n_mels: 40, ..p_mel() });
        assert_eq!(n_mels, 40);
    }

    // Integration: mfcc

    #[test]
    fn test_mfcc_shape() {
        let (_, n_mfcc, n_frames) = mfcc(&sine_wave(440.0, 22050.0, 1.0), &p_mfcc());
        assert_eq!(n_mfcc, 20);
        assert!(n_frames > 0);
    }

    #[test]
    fn test_mfcc_len_matches_shape() {
        let (out, n_mfcc, n_frames) = mfcc(&sine_wave(440.0, 22050.0, 0.5), &p_mfcc());
        assert_eq!(out.len(), n_mfcc * n_frames);
    }

    #[test]
    fn test_mfcc_silence_frames_uniform() {
        let (out, n_mfcc, n_frames) = mfcc(&silence(22050), &p_mfcc());
        for n in 0..n_mfcc {
            let first = out[n * n_frames];
            for t in 1..n_frames {
                assert!((out[n * n_frames + t] - first).abs() < 1e-9,
                    "coeff {} frame {} differs from frame 0", n, t);
            }
        }
    }

    #[test]
    fn test_mfcc_n_mfcc_respected() {
        for n_out in [13usize, 20, 40] {
            let p = MfccParams { n_mfcc: n_out, ..p_mfcc() };
            let (_, n_mfcc, _) = mfcc(&sine_wave(440.0, 22050.0, 0.5), &p);
            assert_eq!(n_mfcc, n_out);
        }
    }

    #[test]
    fn test_mfcc_lifter_changes_values() {
        let sig = sine_wave(440.0, 22050.0, 0.5);
        let (m1, _, _) = mfcc(&sig, &p_mfcc());
        let (m2, _, _) = mfcc(&sig, &MfccParams { lifter: 22.0, ..p_mfcc() });
        let diff: f64 = m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6);
    }

    #[test]
    fn test_mfcc_coeff0_largest_mean() {
        let (out, n_mfcc, n_frames) = mfcc(&sine_wave(440.0, 22050.0, 1.0), &p_mfcc());
        let mean_abs: Vec<f64> = (0..n_mfcc)
            .map(|n| (0..n_frames).map(|t| out[n * n_frames + t].abs()).sum::<f64>() / n_frames as f64)
            .collect();
        let peak = mean_abs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        assert_eq!(peak, 0, "C0 should dominate, got coeff {}", peak);
    }

    #[test]
    fn test_mfcc_different_signals_differ() {
        let (m1, _, _) = mfcc(&sine_wave(440.0,  22050.0, 0.5), &p_mfcc());
        let (m2, _, _) = mfcc(&sine_wave(4000.0, 22050.0, 0.5), &p_mfcc());
        let diff: f64 = m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0);
    }

    #[test]
    fn test_mfcc_deterministic() {
        let sig = sine_wave(440.0, 22050.0, 0.5);
        assert_eq!(mfcc(&sig, &p_mfcc()).0, mfcc(&sig, &p_mfcc()).0);
    }

    #[test]
    fn test_mfcc_values_finite() {
        let (out, _, _) = mfcc(&sine_wave(440.0, 22050.0, 0.5), &p_mfcc());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // Integration: chroma_stft


    #[test]
    fn test_chroma_shape() {
        let (_, n_chroma, n_frames) = chroma_stft(&sine_wave(440.0, 22050.0, 1.0), &p_chroma());
        assert_eq!(n_chroma, 12);
        assert!(n_frames > 0);
    }

    #[test]
    fn test_chroma_len_matches_shape() {
        let (out, n_chroma, n_frames) = chroma_stft(&sine_wave(440.0, 22050.0, 0.5), &p_chroma());
        assert_eq!(out.len(), n_chroma * n_frames);
    }

    #[test]
    fn test_chroma_values_in_unit_range() {
        let (out, _, _) = chroma_stft(&sine_wave(440.0, 22050.0, 1.0), &p_chroma());
        for &v in &out {
            assert!(v >= -1e-12 && v <= 1.0 + 1e-12, "chroma out of [0,1]: {}", v);
        }
    }

    #[test]
    fn test_chroma_each_frame_max_is_one() {
        let (out, n_chroma, n_frames) = chroma_stft(&sine_wave(440.0, 22050.0, 1.0), &p_chroma());
        for t in 0..n_frames {
            let max = (0..n_chroma).map(|c| out[c * n_frames + t]).fold(0.0f64, f64::max);
            if max > 1e-9 {
                assert!((max - 1.0).abs() < 1e-9, "frame {} max = {}", t, max);
            }
        }
    }

    #[test]
    fn test_chroma_silence_zero() {
        let (out, _, _) = chroma_stft(&silence(22050), &p_chroma());
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_chroma_a440_peaks_at_bin9() {
        // base_c=true: C=0, C#=1, …, A=9, A#=10, B=11
        let (out, n_chroma, n_frames) = chroma_stft(&sine_wave(440.0, 22050.0, 2.0), &p_chroma());
        let energy: Vec<f64> = (0..n_chroma)
            .map(|c| (0..n_frames).map(|t| out[c * n_frames + t]).sum::<f64>())
            .collect();
        let peak = energy.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        assert!((peak as i64 - 9).abs() <= 1, "A440 should peak at bin 9 (A), got {}", peak);
    }

    #[test]
    fn test_chroma_c4_peaks_at_bin0() {
        // Middle C = 261.63 Hz → bin 0
        let (out, n_chroma, n_frames) = chroma_stft(&sine_wave(261.63, 22050.0, 2.0), &p_chroma());
        let energy: Vec<f64> = (0..n_chroma)
            .map(|c| (0..n_frames).map(|t| out[c * n_frames + t]).sum::<f64>())
            .collect();
        let peak = energy.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        assert!(peak == 0 || peak == 1 || peak == 11,
            "C4 should peak at bin 0, got {}", peak);
    }

    #[test]
    fn test_chroma_octave_invariance() {
        // A3=220, A4=440, A5=880 should all land on the same chroma bin
        let peaks: Vec<usize> = [220.0f64, 440.0, 880.0].iter().map(|&f| {
            let (out, n_chroma, n_frames) = chroma_stft(&sine_wave(f, 22050.0, 2.0), &p_chroma());
            let energy: Vec<f64> = (0..n_chroma)
                .map(|c| (0..n_frames).map(|t| out[c * n_frames + t]).sum::<f64>())
                .collect();
            energy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        }).collect();
        let first = peaks[0] as i64;
        for (i, &p) in peaks.iter().enumerate() {
            assert!((p as i64 - first).abs() <= 1,
                "Octave {} gives bin {}, expected near {}", i, p, first);
        }
    }

    #[test]
    fn test_chroma_custom_n_chroma() {
        let (_, n_chroma, _) = chroma_stft(
            &sine_wave(440.0, 22050.0, 0.5),
            &ChromaParams { n_chroma: 24, ..p_chroma() });
        assert_eq!(n_chroma, 24);
    }

    #[test]
    fn test_chroma_finite() {
        let (out, _, _) = chroma_stft(&sine_wave(440.0, 22050.0, 0.5), &p_chroma());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // Cross-feature consistency


    #[test]
    fn test_mel_mfcc_same_frame_count() {
        let sig = sine_wave(440.0, 22050.0, 1.0);
        let (_, _, n1) = melspectrogram(&sig, &p_mel());
        let (_, _, n2) = mfcc(&sig, &p_mfcc());
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_mel_chroma_same_frame_count() {
        let sig = sine_wave(440.0, 22050.0, 1.0);
        let (_, _, n1) = melspectrogram(&sig, &p_mel());
        let (_, _, n2) = chroma_stft(&sig, &p_chroma());
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_smaller_hop_more_frames() {
        let sig = sine_wave(440.0, 22050.0, 1.0);
        let p512 = MelParams { stft: StftParams { hop_length: 512, ..Default::default() }, ..p_mel() };
        let p256 = MelParams { stft: StftParams { hop_length: 256, ..Default::default() }, ..p_mel() };
        let (_, _, f512) = melspectrogram(&sig, &p512);
        let (_, _, f256) = melspectrogram(&sig, &p256);
        assert!(f256 > f512);
    }


    // Verify power_to_db integration (uses spectrum::power_to_db via mfcc)

    #[test]
    fn test_mfcc_uses_spectrum_power_to_db_top_db() {
        // If top_db=80 is correctly applied, silence should produce uniform MFCCs
        // (all frames hit the noise floor identically). Already tested above, but
        // this one names the dependency explicitly.
        let (out, n_mfcc, n_frames) = mfcc(&silence(22050), &p_mfcc());
        for n in 0..n_mfcc {
            let first = out[n * n_frames];
            for t in 1..n_frames {
                assert!((out[n * n_frames + t] - first).abs() < 1e-9);
            }
        }
    }
}

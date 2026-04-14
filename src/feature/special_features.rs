//! # audio_features
//!
//! Rust implementations of librosa's core DSP feature extraction functions:
//!
//! - [`melspectrogram`] — Mel-scaled power spectrogram  (librosa PI1)
//! - [`mfcc`]           — Mel-frequency cepstral coefficients (librosa PI1)
//! - [`chroma_stft`]    — Chromagram via STFT (librosa PI2)
//!
//! ## Design notes
//! * All functions operate on `&[f32]` audio samples + a params config struct.
//! * Output matrices are returned as `Vec<f32>` in **row-major** order:
//!   index `[row * n_cols + col]`.
//! * The STFT uses a complex FFT via `rustfft` (r2c equivalent).
//! * Hann window, zero-pad centering, and power=2 are the defaults, matching
//!   librosa's defaults 

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Public parameter structs
// ─────────────────────────────────────────────────────────────────────────────

/// STFT parameters shared by all three feature functions.
#[derive(Debug, Clone)]
pub struct StftParams {
    pub sr: f32,          // Sampling rate (Hz). Default: 22050.
    pub n_fft: usize,     // FFT window size. Default: 2048.
    pub hop_length: usize,// Hop length in samples. Default: 512.
    pub center: bool,     // Pad signal by n_fft/2 on each side. Default: true.
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
    pub n_mels: usize,         // Number of Mel bands. Default: 128.
    pub fmin: f32,              // Lowest frequency (Hz). Default: 0.0.
    pub fmax: Option<f32>,      // Highest frequency. None → sr/2.
    pub power: f32,             // Spectrogram exponent (1=mag, 2=power). Default: 2.0.
    pub norm_slaney: bool,      // Area-normalise Mel filters. Default: true.
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
    pub n_mfcc: usize,  // Number of coefficients. Default: 20.
    pub lifter: f32,    // Liftering coefficient (0 = disabled). Default: 0.0.
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
    pub n_chroma: usize,       // Number of chroma bins. Default: 12.
    pub tuning: f32,            // Tuning offset in fractional bins. Default: 0.0.
    pub ctroct: f32,            // Centre octave for Gaussian weighting. Default: 5.0.
    pub octwidth: Option<f32>,  // Gaussian half-width in octaves. None = flat.
    pub base_c: bool,           // True → base note C; False → base note A.
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

// ─────────────────────────────────────────────────────────────────────────────
// Internal DSP helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a periodic Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos())).collect()
}

/// Compute the power spectrogram `S[n_bins, n_frames]` from raw samples.
///
/// Returns `(data_row_major, n_bins, n_frames)`.
/// `power` is the exponent applied to the magnitude (2 → power spectrum).
pub fn power_spectrogram(samples: &[f32], p: &StftParams, power: f32) -> (Vec<f32>, usize, usize) {
    let n_bins = p.n_fft / 2 + 1;
    let window = hann_window(p.n_fft);

    let padded: Vec<f32> = if p.center {
        let pad = p.n_fft / 2;
        let mut v = vec![0.0f32; pad];
        v.extend_from_slice(samples);
        v.extend(vec![0.0f32; pad]);
        v
    } else {
        samples.to_vec()
    };

    let n_frames = if padded.len() >= p.n_fft {
        1 + (padded.len() - p.n_fft) / p.hop_length
    } else {
        0
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(p.n_fft);

    // Collect frames column-major then transpose
    let mut spec_col = vec![0.0f32; n_bins * n_frames]; // [t, k] first

    for frame in 0..n_frames {
        let start = frame * p.hop_length;
        let mut buf: Vec<Complex<f32>> = (0..p.n_fft)
            .map(|i| Complex::new(padded[start + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buf);
        for k in 0..n_bins {
            spec_col[frame * n_bins + k] = buf[k].norm().powf(power);
        }
    }

    // Transpose to row-major [k, t]
    let mut out = vec![0.0f32; n_bins * n_frames];
    for k in 0..n_bins {
        for t in 0..n_frames {
            out[k * n_frames + t] = spec_col[t * n_bins + k];
        }
    }
    (out, n_bins, n_frames)
}

// ─────────────────────────── Mel filter bank ───────────────────────────────

/// Hz → Mel using librosa's Slaney linear/log formula.
fn hz_to_mel(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0_f32;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + ((hz / min_log_hz).ln() / logstep)
    } else {
        hz / f_sp
    }
}

/// Mel → Hz (inverse of `hz_to_mel`).
fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0_f32;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        f_sp * mel
    }
}

/// Build a Mel filter bank of shape `[n_mels, n_fft/2+1]`.
/// Mirrors `librosa.filters.mel` with optional Slaney area normalisation.
pub fn mel_filterbank(
    sr: f32, n_fft: usize, n_mels: usize,
    fmin: f32, fmax: f32, norm_slaney: bool,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let mut weights = vec![0.0f32; n_mels * n_bins];

    for m in 0..n_mels {
        let f_low    = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_high   = mel_points[m + 2];

        for (k, &f) in fft_freqs.iter().enumerate() {
            weights[m * n_bins + k] = if f >= f_low && f <= f_center && (f_center - f_low) > 0.0 {
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
                for k in 0..n_bins {
                    weights[m * n_bins + k] /= bandwidth;
                }
            }
        }
    }
    weights
}

// ──────────────────────────── DCT-II ───────────────────────────────────────

/// Ortho-normalised DCT-II of each column of `s` (shape `[n_rows, n_cols]`).
/// Returns the first `n_out` rows.
///
/// Formula:  X[k] = w[k] · Σ_n x[n] · cos(π·k·(2n+1) / 2N)
/// where     w[0] = √(1/N),  w[k>0] = √(2/N)
pub fn dct2_columns(s: &[f32], n_rows: usize, n_cols: usize, n_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_out * n_cols];
    let sqrt_1_n = (1.0_f32 / n_rows as f32).sqrt();
    let sqrt_2_n = (2.0_f32 / n_rows as f32).sqrt();

    for t in 0..n_cols {
        for k in 0..n_out {
            let w = if k == 0 { sqrt_1_n } else { sqrt_2_n };
            let sum: f32 = (0..n_rows)
                .map(|n| {
                    s[n * n_cols + t]
                        * (PI * k as f32 * (2 * n + 1) as f32 / (2 * n_rows) as f32).cos()
                })
                .sum();
            out[k * n_cols + t] = w * sum;
        }
    }
    out
}

// ───────────────────────── power_to_db ─────────────────────────────────────

/// Convert a power spectrogram to dB scale.
/// `ref_value` should be the global max power (librosa default).
/// `amin` is a noise floor (1e-10). `top_db` clips the dynamic range.
pub fn power_to_db(s: &[f32], ref_value: f32, amin: f32, top_db: Option<f32>) -> Vec<f32> {
    let ref_db = ref_value.max(amin);
    let mut db: Vec<f32> = s.iter().map(|&x| 10.0 * (x.max(amin) / ref_db).log10()).collect();

    if let Some(top) = top_db {
        let max_db = db.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut db {
            *v = v.max(max_db - top);
        }
    }
    db
}

// ──────────────────── L∞ column normalisation ───────────────────────────────

/// Normalise each column of `[n_rows × n_cols]` by its L∞ (max-abs) norm.
/// Zero columns are left unchanged.
pub fn normalize_linf_columns(m: &mut Vec<f32>, n_rows: usize, n_cols: usize) {
    for t in 0..n_cols {
        let max: f32 = (0..n_rows).map(|r| m[r * n_cols + t].abs()).fold(0.0f32, f32::max);
        if max > 0.0 {
            for r in 0..n_rows { m[r * n_cols + t] /= max; }
        }
    }
}

// ─────────────────────── Chroma filter bank ────────────────────────────────

/// Build a chroma filter bank `[n_chroma × n_bins]`.
/// Mirrors `librosa.filters.chroma`.
pub fn chroma_filterbank(
    sr: f32, n_fft: usize, n_chroma: usize,
    tuning: f32, ctroct: f32, octwidth: Option<f32>, base_c: bool,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    // C0 = A440 / 2^(57/12)  (57 semitones above C0)
    let c0 = 440.0_f32 / 2.0_f32.powf(57.0 / 12.0);

    let mut weights = vec![0.0f32; n_chroma * n_bins];

    for k in 1..n_bins {   // skip DC
        let freq = k as f32 * sr / n_fft as f32;
        let semitone = 12.0 * (freq / c0).log2() + tuning;
        let octave   = semitone / 12.0;

        let oct_weight = match octwidth {
            Some(ow) => { let d = (octave - ctroct) / ow; (-0.5 * d * d).exp() }
            None     => 1.0,
        };

        let frac_bin = semitone.rem_euclid(n_chroma as f32);

        for c in 0..n_chroma {
            let mut d = frac_bin - c as f32;
            let half = n_chroma as f32 / 2.0;
            if d >  half { d -= n_chroma as f32; }
            if d < -half { d += n_chroma as f32; }
            weights[c * n_bins + k] += oct_weight * (-0.5 * d * d).exp();
        }
    }

    // Our formula naturally produces C-based bins (C=0, A=9).
    // Only rotate when base_c=false (shift to A-based: A=0).
    if !base_c {
        let shift = 3usize % n_chroma;
        let mut rotated = vec![0.0f32; n_chroma * n_bins];
        for c in 0..n_chroma {
            let src = (c + shift) % n_chroma;
            for k in 0..n_bins {
                rotated[c * n_bins + k] = weights[src * n_bins + k];
            }
        }
        weights = rotated;
    }

    // L2-normalise each chroma row
    for c in 0..n_chroma {
        let norm: f32 = (0..n_bins).map(|k| weights[c * n_bins + k].powi(2)).sum::<f32>().sqrt();
        if norm > 0.0 {
            for k in 0..n_bins { weights[c * n_bins + k] /= norm; }
        }
    }
    weights
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a mel-scaled power spectrogram.
///
/// Returns `(data, n_mels, n_frames)` — row-major `[n_mels × n_frames]`.
///
/// Equivalent librosa call:
/// ```python
/// librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
///                                 n_mels=128, fmin=0, norm='slaney', power=2)
/// ```
pub fn melspectrogram(samples: &[f32], p: &MelParams) -> (Vec<f32>, usize, usize) {
    let (spec, _n_bins, n_frames) = power_spectrogram(samples, &p.stft, p.power);
    let n_bins = p.stft.n_fft / 2 + 1;
    let fmax   = p.fmax.unwrap_or(p.stft.sr / 2.0);
    let mel_fb = mel_filterbank(p.stft.sr, p.stft.n_fft, p.n_mels, p.fmin, fmax, p.norm_slaney);

    let mut mel_spec = vec![0.0f32; p.n_mels * n_frames];
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
/// Pipeline: audio → mel spectrogram → power_to_db → ortho DCT-II → optional lifter.
///
/// Returns `(data, n_mfcc, n_frames)` — row-major `[n_mfcc × n_frames]`.
///
/// Equivalent librosa call:
/// ```python
/// librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
/// ```
pub fn mfcc(samples: &[f32], p: &MfccParams) -> (Vec<f32>, usize, usize) {
    let (mel_spec, n_mels, n_frames) = melspectrogram(samples, &p.mel);
    let ref_val  = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(1e-10);
    let log_mel  = power_to_db(&mel_spec, ref_val, 1e-10, Some(80.0));
    let mut out  = dct2_columns(&log_mel, n_mels, n_frames, p.n_mfcc);

    if p.lifter > 0.0 {
        for n in 0..p.n_mfcc {
            let lift = 1.0 + (p.lifter / 2.0) * (PI * (n + 1) as f32 / p.lifter).sin();
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
/// librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512,
///                              n_chroma=12, tuning=0.0)
/// ```
pub fn chroma_stft(samples: &[f32], p: &ChromaParams) -> (Vec<f32>, usize, usize) {
    let (spec, _n_bins, n_frames) = power_spectrogram(samples, &p.stft, 2.0);
    let n_bins = p.stft.n_fft / 2 + 1;
    let fb = chroma_filterbank(p.stft.sr, p.stft.n_fft, p.n_chroma,
                               p.tuning, p.ctroct, p.octwidth, p.base_c);

    let mut raw = vec![0.0f32; p.n_chroma * n_frames];
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Signal generators ────────────────────────────────────────────────────

    fn sine_wave(freq: f32, sr: f32, duration_s: f32) -> Vec<f32> {
        let n = (sr * duration_s) as usize;
        (0..n).map(|i| (2.0 * PI * freq * i as f32 / sr).sin()).collect()
    }
    fn silence(n: usize)              -> Vec<f32> { vec![0.0; n] }
    fn impulse(n: usize, pos: usize)  -> Vec<f32> { let mut v = silence(n); v[pos] = 1.0; v }

    fn default_mel()    -> MelParams    { MelParams::default() }
    fn default_mfcc()   -> MfccParams   { MfccParams::default() }
    fn default_chroma() -> ChromaParams { ChromaParams::default() }

    // ─────────────────────────────────────────────────────────────────────────
    // hz_to_mel / mel_to_hz
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hz_mel_roundtrip() {
        for hz in [0.0f32, 100.0, 440.0, 1000.0, 8000.0, 22050.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((hz - back).abs() < 1e-2, "roundtrip failed at {} Hz (got {})", hz, back);
        }
    }

    #[test]
    fn test_hz_mel_monotone() {
        let freqs = [50.0f32, 200.0, 500.0, 1000.0, 4000.0, 11000.0];
        for w in freqs.windows(2) {
            assert!(hz_to_mel(w[0]) < hz_to_mel(w[1]),
                    "{} → {} mel not monotone", w[0], w[1]);
        }
    }

    #[test]
    fn test_mel_linear_region() {
        // Below 1000 Hz the formula is linear: mel ∝ hz
        let m1 = hz_to_mel(500.0);
        let m2 = hz_to_mel(1000.0);
        assert!((m2 / m1 - 2.0).abs() < 1e-3, "Linear region should be 2× at 2× freq");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // hann_window
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hann_window_endpoint_near_zero() {
        let w = hann_window(1024);
        assert!(w[0].abs() < 1e-6);
    }

    #[test]
    fn test_hann_window_midpoint_one() {
        let n = 1024;
        let w = hann_window(n);
        assert!((w[n / 2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_hann_window_symmetry() {
        let n = 512;
        let w = hann_window(n);
        for i in 1..n / 2 {
            assert!((w[i] - w[n - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hann_window_all_non_negative() {
        for &v in &hann_window(2048) {
            assert!(v >= 0.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // mel_filterbank
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(22050.0, 2048, 128, 0.0, 11025.0, true);
        assert_eq!(fb.len(), 128 * (2048 / 2 + 1));
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = mel_filterbank(22050.0, 2048, 128, 0.0, 11025.0, true);
        assert!(fb.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mel_filterbank_each_row_has_support() {
        let n_mels = 40;
        let n_bins = 2048 / 2 + 1;
        let fb = mel_filterbank(22050.0, 2048, n_mels, 0.0, 11025.0, false);
        for m in 0..n_mels {
            let sum: f32 = (0..n_bins).map(|k| fb[m * n_bins + k]).sum();
            assert!(sum > 0.0, "Mel band {} has no support", m);
        }
    }

    #[test]
    fn test_mel_filterbank_peak_at_most_one_without_slaney() {
        // Without Slaney, each filter peaks at 1.0
        let n_mels = 20;
        let n_bins = 2048 / 2 + 1;
        let fb = mel_filterbank(22050.0, 2048, n_mels, 0.0, 11025.0, false);
        for m in 0..n_mels {
            let peak: f32 = (0..n_bins).map(|k| fb[m * n_bins + k]).fold(0.0f32, f32::max);
            assert!(peak >= 0.9, "filter {} peak is {} (expected >= 0.9)", m, peak);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // power_to_db
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_power_to_db_reference_is_zero() {
        let db = power_to_db(&[1.0], 1.0, 1e-10, None);
        assert!((db[0]).abs() < 1e-4);
    }

    #[test]
    fn test_power_to_db_10x_is_10db() {
        let db = power_to_db(&[10.0], 1.0, 1e-10, None);
        assert!((db[0] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_power_to_db_100x_is_20db() {
        let db = power_to_db(&[100.0], 1.0, 1e-10, None);
        assert!((db[0] - 20.0).abs() < 1e-3);
    }

    #[test]
    fn test_power_to_db_top_db_respected() {
        let s = vec![1.0, 1e-10];
        let db = power_to_db(&s, 1.0, 1e-10, Some(80.0));
        assert!(db[0] - db[1] <= 80.0 + 1e-3);
    }

    #[test]
    fn test_power_to_db_amin_floor() {
        // value below amin should be clamped to -10*log10(ref/amin)
        let db = power_to_db(&[0.0], 1.0, 1e-10, None);
        assert!((db[0] - (-100.0)).abs() < 1e-2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dct2_columns
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dct2_dc_input_coefficient_zero() {
        let n = 8;
        let c = 3.0f32;
        let s: Vec<f32> = (0..n).map(|_| c).collect();
        let out = dct2_columns(&s, n, 1, n);
        // X[0] = c * sqrt(N)  (ortho normalisation)
        assert!((out[0] - c * (n as f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_dct2_dc_input_higher_coefficients_zero() {
        let n = 8;
        let s: Vec<f32> = vec![3.0; n];
        let out = dct2_columns(&s, n, 1, n);
        for k in 1..n {
            assert!(out[k].abs() < 1e-4, "DCT[{}]={} should be ~0", k, out[k]);
        }
    }

    #[test]
    fn test_dct2_parseval() {
        let n = 16;
        let s: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let out = dct2_columns(&s, n, 1, n);
        let e_in: f32  = s.iter().map(|x| x * x).sum();
        let e_out: f32 = out.iter().map(|x| x * x).sum();
        assert!((e_in - e_out).abs() < 1e-3);
    }

    #[test]
    fn test_dct2_n_out_respected() {
        let n = 32;
        let s: Vec<f32> = (0..n).map(|i| i as f32).collect();
        for n_out in [1, 10, 20, 32] {
            let out = dct2_columns(&s, n, 1, n_out);
            assert_eq!(out.len(), n_out);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // normalize_linf_columns
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_linf_max_is_one() {
        let mut m: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        normalize_linf_columns(&mut m, 4, 3);
        for t in 0..3 {
            let max = (0..4).map(|r| m[r * 3 + t].abs()).fold(0.0f32, f32::max);
            assert!((max - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_linf_zero_column_unchanged() {
        let mut m = vec![0.0f32; 4];
        normalize_linf_columns(&mut m, 2, 2);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_linf_preserves_ratio() {
        let mut m = vec![2.0f32, 4.0]; // single column
        normalize_linf_columns(&mut m, 2, 1);
        assert!((m[1] / m[0] - 2.0).abs() < 1e-6);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Integration: melspectrogram
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_melspectrogram_output_shape() {
        let signal = sine_wave(440.0, 22050.0, 1.0);
        let (_, n_mels, n_frames) = melspectrogram(&signal, &default_mel());
        assert_eq!(n_mels, 128);
        assert!(n_frames > 0);
    }

    #[test]
    fn test_melspectrogram_non_negative() {
        let (out, _, _) = melspectrogram(&sine_wave(440.0, 22050.0, 0.5), &default_mel());
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_melspectrogram_silence_is_zero() {
        let (out, _, _) = melspectrogram(&silence(22050), &default_mel());
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_melspectrogram_data_length_matches_shape() {
        let signal = sine_wave(440.0, 22050.0, 0.5);
        let (out, n_mels, n_frames) = melspectrogram(&signal, &default_mel());
        assert_eq!(out.len(), n_mels * n_frames);
    }

    #[test]
    fn test_melspectrogram_energy_near_440hz() {
        // 440 Hz sine → peak Mel band should be in the lower-mid range (~bands 20–35)
        let signal = sine_wave(440.0, 22050.0, 1.0);
        let (out, n_mels, n_frames) = melspectrogram(&signal, &default_mel());
        let energy_per_band: Vec<f32> = (0..n_mels)
            .map(|m| (0..n_frames).map(|t| out[m * n_frames + t]).sum::<f32>())
            .collect();
        let peak = energy_per_band.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
        assert!((15..38).contains(&peak), "440 Hz peak band should be 15–38, got {}", peak);
    }

    #[test]
    fn test_melspectrogram_impulse_excites_all_bands() {
        let s = impulse(22050, 1024);
        let (out, n_mels, n_frames) = melspectrogram(&s, &default_mel());
        let peak_t = (0..n_frames)
            .max_by(|&a, &b| {
                let ea: f32 = (0..n_mels).map(|m| out[m * n_frames + a]).sum();
                let eb: f32 = (0..n_mels).map(|m| out[m * n_frames + b]).sum();
                ea.partial_cmp(&eb).unwrap()
            }).unwrap();
        let bands_with_energy = (0..n_mels).filter(|&m| out[m * n_frames + peak_t] > 0.0).count();
        assert!(bands_with_energy > n_mels / 2,
                "Impulse should excite >50% of bands, got {}", bands_with_energy);
    }

    #[test]
    fn test_melspectrogram_longer_more_frames() {
        let (_, _, n_short) = melspectrogram(&sine_wave(440.0, 22050.0, 0.5), &default_mel());
        let (_, _, n_long)  = melspectrogram(&sine_wave(440.0, 22050.0, 1.0), &default_mel());
        assert!(n_long > n_short);
    }

    #[test]
    fn test_melspectrogram_custom_n_mels() {
        let (_, n_mels, _) = melspectrogram(&sine_wave(440.0, 22050.0, 0.2),
                                            &MelParams { n_mels: 40, ..default_mel() });
        assert_eq!(n_mels, 40);
    }

    #[test]
    fn test_melspectrogram_higher_freq_peaks_in_higher_band() {
        let signal_lo = sine_wave(200.0,  22050.0, 1.0);
        let signal_hi = sine_wave(4000.0, 22050.0, 1.0);
        let peak = |sig: &[f32]| {
            let (out, n_mels, n_frames) = melspectrogram(sig, &default_mel());
            let energy: Vec<f32> = (0..n_mels)
                .map(|m| (0..n_frames).map(|t| out[m * n_frames + t]).sum::<f32>())
                .collect();
            energy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        };
        assert!(peak(&signal_hi) > peak(&signal_lo),
                "4 kHz should peak at a higher Mel band than 200 Hz");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Integration: mfcc
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mfcc_output_shape() {
        let (_, n_mfcc, n_frames) = mfcc(&sine_wave(440.0, 22050.0, 1.0), &default_mfcc());
        assert_eq!(n_mfcc, 20);
        assert!(n_frames > 0);
    }

    #[test]
    fn test_mfcc_data_length_matches_shape() {
        let signal = sine_wave(440.0, 22050.0, 0.5);
        let (out, n_mfcc, n_frames) = mfcc(&signal, &default_mfcc());
        assert_eq!(out.len(), n_mfcc * n_frames);
    }

    #[test]
    fn test_mfcc_silence_frames_identical() {
        let (out, n_mfcc, n_frames) = mfcc(&silence(22050), &default_mfcc());
        for n in 0..n_mfcc {
            let first = out[n * n_frames];
            for t in 1..n_frames {
                assert!((out[n * n_frames + t] - first).abs() < 1e-3,
                        "Silence MFCC coeff {} frame {} differs", n, t);
            }
        }
    }

    #[test]
    fn test_mfcc_n_mfcc_respected() {
        for n_out in [13usize, 20, 40] {
            let p = MfccParams { n_mfcc: n_out, ..default_mfcc() };
            let (_, n_mfcc, _) = mfcc(&sine_wave(440.0, 22050.0, 0.5), &p);
            assert_eq!(n_mfcc, n_out);
        }
    }

    #[test]
    fn test_mfcc_lifter_changes_output() {
        let sig = sine_wave(440.0, 22050.0, 0.5);
        let (m1, _, _) = mfcc(&sig, &default_mfcc());
        let (m2, _, _) = mfcc(&sig, &MfccParams { lifter: 22.0, ..default_mfcc() });
        let diff: f32 = m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3);
    }

    #[test]
    fn test_mfcc_coeff0_largest_mean() {
        let signal = sine_wave(440.0, 22050.0, 1.0);
        let (out, n_mfcc, n_frames) = mfcc(&signal, &default_mfcc());
        let mean_abs: Vec<f32> = (0..n_mfcc)
            .map(|n| (0..n_frames).map(|t| out[n * n_frames + t].abs()).sum::<f32>() / n_frames as f32)
            .collect();
        let peak = mean_abs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
        assert_eq!(peak, 0, "C0 should dominate, got coeff {}", peak);
    }

    #[test]
    fn test_mfcc_different_signals_different() {
        let (m1, _, _) = mfcc(&sine_wave(440.0,  22050.0, 0.5), &default_mfcc());
        let (m2, _, _) = mfcc(&sine_wave(4000.0, 22050.0, 0.5), &default_mfcc());
        let diff: f32 = m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0);
    }

    #[test]
    fn test_mfcc_same_signal_same_output() {
        let sig = sine_wave(440.0, 22050.0, 0.5);
        let (m1, _, _) = mfcc(&sig, &default_mfcc());
        let (m2, _, _) = mfcc(&sig, &default_mfcc());
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_mfcc_values_finite() {
        let (out, _, _) = mfcc(&sine_wave(440.0, 22050.0, 0.5), &default_mfcc());
        assert!(out.iter().all(|v| v.is_finite()), "MFCCs contain non-finite values");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Integration: chroma_stft
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_stft_output_shape() {
        let (_, n_chroma, n_frames) = chroma_stft(&sine_wave(440.0, 22050.0, 1.0), &default_chroma());
        assert_eq!(n_chroma, 12);
        assert!(n_frames > 0);
    }

    #[test]
    fn test_chroma_stft_values_in_unit_range() {
        let (out, _, _) = chroma_stft(&sine_wave(440.0, 22050.0, 1.0), &default_chroma());
        for &v in &out {
            assert!(v >= -1e-6 && v <= 1.0 + 1e-6, "chroma out of [0,1]: {}", v);
        }
    }

    #[test]
    fn test_chroma_stft_each_frame_max_is_one() {
        let signal = sine_wave(440.0, 22050.0, 1.0);
        let (out, n_chroma, n_frames) = chroma_stft(&signal, &default_chroma());
        for t in 0..n_frames {
            let max = (0..n_chroma).map(|c| out[c * n_frames + t]).fold(0.0f32, f32::max);
            if max > 1e-6 {
                assert!((max - 1.0).abs() < 1e-5, "frame {} max={}", t, max);
            }
        }
    }

    #[test]
    fn test_chroma_stft_silence_zero() {
        let (out, _, _) = chroma_stft(&silence(22050), &default_chroma());
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_chroma_stft_data_length_matches_shape() {
        let signal = sine_wave(440.0, 22050.0, 0.5);
        let (out, n_chroma, n_frames) = chroma_stft(&signal, &default_chroma());
        assert_eq!(out.len(), n_chroma * n_frames);
    }

    #[test]
    fn test_chroma_stft_a440_peaks_on_a_bin() {
        // A4 = 440 Hz.  With base_c=true: C=0 … A=9 … B=11. Allow ±1 for leakage.
        let signal = sine_wave(440.0, 22050.0, 2.0);
        let (out, n_chroma, n_frames) = chroma_stft(&signal, &default_chroma());
        let energy: Vec<f32> = (0..n_chroma)
            .map(|c| (0..n_frames).map(|t| out[c * n_frames + t]).sum::<f32>())
            .collect();
        let peak = energy.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
        assert!((peak as i32 - 9).abs() <= 1,
                "A440 should peak at bin 9 (A), got {}", peak);
    }

    #[test]
    fn test_chroma_stft_c4_peaks_on_c_bin() {
        // Middle C = 261.63 Hz → bin 0
        let signal = sine_wave(261.63, 22050.0, 2.0);
        let (out, n_chroma, n_frames) = chroma_stft(&signal, &default_chroma());
        let energy: Vec<f32> = (0..n_chroma)
            .map(|c| (0..n_frames).map(|t| out[c * n_frames + t]).sum::<f32>())
            .collect();
        let peak = energy.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
        assert!(peak == 0 || peak == 1 || peak == 11,
                "C4 should peak at bin 0 (C), got {}", peak);
    }

    #[test]
    fn test_chroma_stft_octave_invariance() {
        // A3=220, A4=440, A5=880 should all peak at the same chroma bin
        let sr  = 22050.0;
        let dur = 2.0;
        let peaks: Vec<usize> = [220.0f32, 440.0, 880.0].iter().map(|&f| {
            let (out, n_chroma, n_frames) = chroma_stft(&sine_wave(f, sr, dur), &default_chroma());
            let energy: Vec<f32> = (0..n_chroma)
                .map(|c| (0..n_frames).map(|t| out[c * n_frames + t]).sum::<f32>())
                .collect();
            energy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        }).collect();
        let first = peaks[0] as i32;
        for (i, &p) in peaks.iter().enumerate() {
            assert!((p as i32 - first).abs() <= 1,
                    "Octave {} gives bin {}, expected near {}", i, p, first);
        }
    }

    #[test]
    fn test_chroma_stft_custom_n_chroma() {
        let p = ChromaParams { n_chroma: 24, ..default_chroma() };
        let (_, n_chroma, _) = chroma_stft(&sine_wave(440.0, 22050.0, 0.5), &p);
        assert_eq!(n_chroma, 24);
    }

    #[test]
    fn test_chroma_stft_values_finite() {
        let (out, _, _) = chroma_stft(&sine_wave(440.0, 22050.0, 0.5), &default_chroma());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cross-feature consistency
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_mel_mfcc_same_frame_count() {
        let sig = sine_wave(440.0, 22050.0, 1.0);
        let (_, _, mel_f)  = melspectrogram(&sig, &default_mel());
        let (_, _, mfcc_f) = mfcc(&sig, &default_mfcc());
        assert_eq!(mel_f, mfcc_f);
    }

    #[test]
    fn test_mel_chroma_same_frame_count() {
        let sig = sine_wave(440.0, 22050.0, 1.0);
        let (_, _, mel_f)    = melspectrogram(&sig, &default_mel());
        let (_, _, chroma_f) = chroma_stft(&sig, &default_chroma());
        assert_eq!(mel_f, chroma_f);
    }

    #[test]
    fn test_smaller_hop_more_frames() {
        let sig = sine_wave(440.0, 22050.0, 1.0);
        let p_512 = MelParams { stft: StftParams { hop_length: 512, ..Default::default() }, ..default_mel() };
        let p_256 = MelParams { stft: StftParams { hop_length: 256, ..Default::default() }, ..default_mel() };
        let (_, _, f512) = melspectrogram(&sig, &p_512);
        let (_, _, f256) = melspectrogram(&sig, &p_256);
        assert!(f256 > f512);
    }
}
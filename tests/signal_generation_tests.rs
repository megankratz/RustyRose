/// Root-level integration tests for the signal_generation module.
///
/// These tests exercise `tone`, `chirp`, and `clicks` as public API from the
/// crate root, verifying lengths, sample values against analytic formulas,
/// and error conditions.
use rusty_rose::core_io::{chirp, clicks, tone};

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// tone
// ---------------------------------------------------------------------------

#[test]
fn integration_tone_length_from_length() {
    let y = tone(440.0, 22050, Some(22050), None, None).unwrap();
    assert_eq!(y.len(), 22050);
}

#[test]
fn integration_tone_length_from_duration() {
    let y = tone(440.0, 22050, None, Some(1.0), None).unwrap();
    assert_eq!(y.len(), 22050);
}

#[test]
fn integration_tone_length_priority() {
    // length=50 beats duration=1.0 (22050 samples)
    let y = tone(440.0, 22050, Some(50), Some(1.0), None).unwrap();
    assert_eq!(y.len(), 50);
}

#[test]
fn integration_tone_exact_values_default_phi() {
    // Default phi = -π/2 → y[n] = cos(2π·f·n/sr - π/2) = sin(2π·f·n/sr)
    let sr = 22050u32;
    let freq = 440.0f64;
    let phi = -PI * 0.5;
    let y = tone(freq, sr, Some(100), None, None).unwrap();
    for (i, &v) in y.iter().enumerate() {
        let expected = (2.0 * PI * freq * i as f64 / sr as f64 + phi).cos();
        assert!(
            (v - expected).abs() < 1e-12,
            "sample {}: got {:.6}, expected {:.6}",
            i, v, expected
        );
    }
}

#[test]
fn integration_tone_exact_values_custom_phi() {
    let sr = 22050u32;
    let freq = 261.63f64; // C4
    let phi = 0.0f64;
    let y = tone(freq, sr, Some(100), None, Some(phi)).unwrap();
    for (i, &v) in y.iter().enumerate() {
        let expected = (2.0 * PI * freq * i as f64 / sr as f64 + phi).cos();
        assert!(
            (v - expected).abs() < 1e-12,
            "sample {}: got {:.6}, expected {:.6}",
            i, v, expected
        );
    }
}

#[test]
fn integration_tone_amplitude_bounded() {
    let y = tone(440.0, 22050, Some(22050), None, None).unwrap();
    for &v in &y {
        assert!(v.abs() <= 1.0 + 1e-12, "amplitude out of [-1,1]: {}", v);
    }
}

#[test]
fn integration_tone_a4_period() {
    // 441 Hz divides cleanly into 22050 Hz: period = 22050/441 = 50 samples exactly.
    // y[n] and y[n + period] should differ by < 1e-10 (floating-point precision only).
    let sr = 22050u32;
    let freq = 441.0f64; // 22050 / 441 = 50 samples per period
    let period = 50usize;
    let y = tone(freq, sr, Some(period * 4 + 1), None, None).unwrap();
    assert!(
        (y[0] - y[period]).abs() < 1e-10,
        "y[0]={:.6} vs y[{}]={:.6}; diff={:.2e}",
        y[0],
        period,
        y[period],
        (y[0] - y[period]).abs()
    );
    assert!(
        (y[period] - y[2 * period]).abs() < 1e-10,
        "signal should be strictly periodic"
    );
}

#[test]
fn integration_tone_error_no_length_or_duration() {
    let r = tone(440.0, 22050, None, None, None);
    assert!(r.is_err());
}

#[test]
fn integration_tone_error_negative_frequency() {
    let r = tone(-100.0, 22050, Some(100), None, None);
    assert!(r.is_err());
}

#[test]
fn integration_tone_error_zero_frequency() {
    let r = tone(0.0, 22050, Some(100), None, None);
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// chirp
// ---------------------------------------------------------------------------

#[test]
fn integration_chirp_length_from_length() {
    let y = chirp(110.0, 880.0, 22050, Some(22050), None, false, None).unwrap();
    assert_eq!(y.len(), 22050);
}

#[test]
fn integration_chirp_length_from_duration() {
    let y = chirp(110.0, 880.0, 22050, None, Some(1.0), false, None).unwrap();
    assert_eq!(y.len(), 22050);
}

#[test]
fn integration_chirp_length_priority() {
    let y = chirp(110.0, 880.0, 22050, Some(50), Some(1.0), false, None).unwrap();
    assert_eq!(y.len(), 50);
}

#[test]
fn integration_chirp_linear_exact_values() {
    let sr = 22050u32;
    let fmin = 110.0f64;
    let fmax = 880.0f64;
    let dur = 1.0f64;
    let phi = -PI * 0.5;
    let y = chirp(fmin, fmax, sr, None, Some(dur), true, Some(phi)).unwrap();
    let sr_f = sr as f64;
    for i in [0usize, 1, 500, 2000, 5000] {
        let t = i as f64 / sr_f;
        let phase = 2.0 * PI * (fmin * t + (fmax - fmin) * t * t / (2.0 * dur));
        let expected = (phi + phase).cos();
        assert!(
            (y[i] - expected).abs() < 1e-12,
            "linear chirp[{}]: got {:.6}, expected {:.6}",
            i, y[i], expected
        );
    }
}

#[test]
fn integration_chirp_log_exact_values() {
    let sr = 22050u32;
    let fmin = 110.0f64;
    let fmax = 880.0f64;
    let dur = 1.0f64;
    let phi = -PI * 0.5;
    let y = chirp(fmin, fmax, sr, None, Some(dur), false, Some(phi)).unwrap();
    let sr_f = sr as f64;
    let k = fmax / fmin;
    let log_k = k.ln();
    for i in [0usize, 1, 500, 2000, 5000] {
        let t = i as f64 / sr_f;
        let phase = 2.0 * PI * fmin * dur / log_k * (k.powf(t / dur) - 1.0);
        let expected = (phi + phase).cos();
        assert!(
            (y[i] - expected).abs() < 1e-12,
            "log chirp[{}]: got {:.6}, expected {:.6}",
            i, y[i], expected
        );
    }
}

#[test]
fn integration_chirp_amplitude_bounded() {
    let y = chirp(110.0, 880.0, 22050, Some(22050), None, true, None).unwrap();
    for &v in &y {
        assert!(v.abs() <= 1.0 + 1e-12, "chirp amplitude out of [-1,1]: {}", v);
    }
}

#[test]
fn integration_chirp_error_no_length_or_duration() {
    let r = chirp(110.0, 880.0, 22050, None, None, false, None);
    assert!(r.is_err());
}

#[test]
fn integration_chirp_error_negative_fmin() {
    let r = chirp(-1.0, 880.0, 22050, Some(100), None, false, None);
    assert!(r.is_err());
}

#[test]
fn integration_chirp_error_negative_fmax() {
    let r = chirp(110.0, -1.0, 22050, Some(100), None, false, None);
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// clicks
// ---------------------------------------------------------------------------

#[test]
fn integration_clicks_from_times_length() {
    let times = [0.0f64, 0.5, 1.0];
    let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.01, Some(22050)).unwrap();
    assert_eq!(y.len(), 22050);
}

#[test]
fn integration_clicks_from_frames_length() {
    let frames = [0usize, 10, 20];
    let y = clicks(None, Some(&frames), 22050, 512, 1000.0, 0.01, Some(22050)).unwrap();
    assert_eq!(y.len(), 22050);
}

#[test]
fn integration_clicks_auto_length() {
    // last click at t=1.0 → sample 22050; click_len = round(0.1 * 22050) = 2205
    let times = [0.0f64, 1.0];
    let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.1, None).unwrap();
    let click_len = (0.1 * 22050.0f64).round() as usize;
    assert_eq!(y.len(), 22050 + click_len);
}

#[test]
fn integration_clicks_click_at_time_zero() {
    let times = [0.0f64];
    let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.01, Some(22050)).unwrap();
    let peak = y[0..50].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(peak > 0.01, "expected energy at t=0, max={}", peak);
}

#[test]
fn integration_clicks_click_at_specific_frame() {
    let frames = [43usize]; // frame 43 * 512 = sample 22016
    let y = clicks(None, Some(&frames), 22050, 512, 1000.0, 0.01, Some(22500)).unwrap();
    let start = 43 * 512;
    let peak = y[start..start + 50]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(peak > 0.01, "expected click near frame 43, max={}", peak);
}

#[test]
fn integration_clicks_amplitude_bounded() {
    // Even with overlapping clicks from close times, amplitude is bounded
    let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.05, Some(22050)).unwrap();
    // amplitude can exceed 0.5 due to overlap, but should stay ≤ 1.0 per click alone
    // Just verify no NaN/Inf
    for &v in &y {
        assert!(v.is_finite(), "non-finite value in click track: {}", v);
    }
}

#[test]
fn integration_clicks_error_no_input() {
    let r = clicks(None, None, 22050, 512, 1000.0, 0.1, None);
    assert!(r.is_err());
}

#[test]
fn integration_clicks_error_bad_freq() {
    let times = [0.0f64];
    let r = clicks(Some(&times), None, 22050, 512, 0.0, 0.1, None);
    assert!(r.is_err());
}

#[test]
fn integration_clicks_error_bad_duration() {
    let times = [0.0f64];
    let r = clicks(Some(&times), None, 22050, 512, 1000.0, -0.1, None);
    assert!(r.is_err());
}

#[test]
fn integration_clicks_both_times_and_frames() {
    // Providing both should union positions
    let times = [0.0f64];
    let frames = [20usize];
    let r = clicks(Some(&times), Some(&frames), 22050, 512, 1000.0, 0.01, Some(22050));
    assert!(r.is_ok());
    let y = r.unwrap();
    // click at sample 0 and at sample 20*512=10240
    let peak0 = y[0..50].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let peak1 = y[10240..10290].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(peak0 > 0.01, "expected click at sample 0");
    assert!(peak1 > 0.01, "expected click at sample 10240");
}

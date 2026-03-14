use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// tone
// ---------------------------------------------------------------------------
// [docs] librosa.tone(frequency, *, sr=22050, length=None, duration=None, phi=None)
// Construct a pure tone (cosine) signal at a given frequency.
//
// Python source:
//   if phi is None:
//       phi = -np.pi * 0.5
//   y = np.cos(2 * np.pi * frequency * np.arange(length) / sr + phi)
//
// Note: cos(x - π/2) = sin(x), so the default phi yields a pure sine wave.
/// Construct a pure tone (cosine) signal at a given frequency.
///
/// # Parameters
/// - `frequency` : frequency in Hz (must be > 0)
/// - `sr`        : sample rate in Hz
/// - `length`    : exact number of output samples (takes priority over `duration`)
/// - `duration`  : duration in seconds (used if `length` is `None`)
/// - `phi`       : phase offset in radians (default: `−π/2`, which gives a pure sine)
///
/// # Returns
/// `Vec<f64>` of length `length` samples.
pub fn tone(
    frequency: f64,
    sr: u32,
    length: Option<usize>,
    duration: Option<f64>,
    phi: Option<f64>,
) -> Result<Vec<f64>, String> {
    if frequency <= 0.0 {
        return Err(format!(
            "\"frequency\" must be strictly positive, got {}",
            frequency
        ));
    }

    let n = match length {
        Some(l) => l,
        None => match duration {
            Some(d) => (d * sr as f64).round() as usize,
            None => {
                return Err(
                    "either \"length\" or \"duration\" must be provided".to_string()
                )
            }
        },
    };

    let phi = phi.unwrap_or(-PI * 0.5);
    let y = (0..n)
        .map(|i| (2.0 * PI * frequency * i as f64 / sr as f64 + phi).cos())
        .collect();

    Ok(y)
}

// ---------------------------------------------------------------------------
// chirp
// ---------------------------------------------------------------------------
// [docs] librosa.chirp(*, fmin, fmax, sr=22050, length=None, duration=None,
//                       linear=False, phi=None)
// Construct a "chirp" or "sine-sweep" signal.
//
// Python source (via scipy.signal.chirp):
//   method = "linear" if linear else "logarithmic"
//   y = scipy.signal.chirp(t, fmin, duration, fmax, method=method,
//                           phi=phi / np.pi * 180)
//
// Linear chirp  : y[t] = cos(φ + 2π · (f0·t + (f1-f0)·t²/(2T)))
// Log chirp     : y[t] = cos(φ + 2π · f0·T/ln(f1/f0) · (k^(t/T) − 1))
//   where k = f1/f0, T = duration
/// Construct a chirp (sine-sweep) signal from `fmin` to `fmax`.
///
/// # Parameters
/// - `fmin`     : start frequency in Hz
/// - `fmax`     : end frequency in Hz
/// - `sr`       : sample rate
/// - `length`   : number of output samples (priority over `duration`)
/// - `duration` : duration in seconds
/// - `linear`   : if `true` use linear sweep; if `false` use logarithmic (default)
/// - `phi`      : phase offset in radians (default `−π/2`)
pub fn chirp(
    fmin: f64,
    fmax: f64,
    sr: u32,
    length: Option<usize>,
    duration: Option<f64>,
    linear: bool,
    phi: Option<f64>,
) -> Result<Vec<f64>, String> {
    if fmin <= 0.0 || fmax <= 0.0 {
        return Err(format!(
            "both \"fmin\" ({}) and \"fmax\" ({}) must be strictly positive",
            fmin, fmax
        ));
    }

    // Resolve length and duration (same logic as librosa: length takes priority)
    let (n, dur) = match length {
        Some(l) => (l, l as f64 / sr as f64),
        None => match duration {
            Some(d) => ((d * sr as f64).round() as usize, d),
            None => {
                return Err(
                    "either \"length\" or \"duration\" must be provided".to_string()
                )
            }
        },
    };

    let phi = phi.unwrap_or(-PI * 0.5);

    // Convert phi from radians to degrees (matching scipy.signal.chirp convention)
    // We implement the formula directly so no conversion needed.
    let sr_f = sr as f64;

    let y: Vec<f64> = if linear {
        // Linear chirp: instantaneous frequency f(t) = fmin + (fmax-fmin)*t/T
        // Phase: integral of 2π·f(t) dt = 2π·(fmin·t + (fmax-fmin)·t²/(2T))
        (0..n)
            .map(|i| {
                let t = i as f64 / sr_f;
                let phase = 2.0 * PI * (fmin * t + (fmax - fmin) * t * t / (2.0 * dur));
                (phi + phase).cos()
            })
            .collect()
    } else {
        // Logarithmic (exponential) chirp: f(t) = fmin * (fmax/fmin)^(t/T)
        // Phase: integral of 2π·f(t) dt = 2π·fmin·T/ln(k) · (k^(t/T) − 1)
        // where k = fmax/fmin
        let k = fmax / fmin;
        let log_k = k.ln();
        (0..n)
            .map(|i| {
                let t = i as f64 / sr_f;
                let phase = 2.0 * PI * fmin * dur / log_k * (k.powf(t / dur) - 1.0);
                (phi + phase).cos()
            })
            .collect()
    };

    Ok(y)
}

// ---------------------------------------------------------------------------
// clicks
// ---------------------------------------------------------------------------
// [docs] librosa.clicks(*, times=None, frames=None, sr=22050, hop_length=512,
//                        click_freq=1000.0, click_duration=0.1, click=None, length=None)
// Construct a "click track".
//
// Python source (simplified):
//   # Build a click sample (cosine tone with half-cosine window)
//   click_length = int(np.round(click_duration * sr))
//   t = np.arange(click_length, dtype=float) / sr
//   click_sample = np.sin(2 * np.pi * click_freq * t)
//   click_sample *= np.cos(np.linspace(0, np.pi, click_length))   # window
//   click_sample /= 2 * click_sample.max()                         # normalize
//
//   # Convert times → sample positions
//   positions = time_to_samples(times, sr=sr)
//   # or positions = frames * hop_length
//
//   # Place click at each position
//   total = length or (max(positions) + click_length)
//   y = np.zeros(total)
//   for start in positions:
//       end = min(start + click_length, total)
//       y[start:end] += click_sample[:end-start]
//
/// Construct a click track by placing a short cosine click at given positions.
///
/// At least one of `times` or `frames` must be provided.
///
/// # Parameters
/// - `times`          : click positions in seconds
/// - `frames`         : click positions as frame indices
/// - `sr`             : sample rate
/// - `hop_length`     : frames → samples conversion factor (used only with `frames`)
/// - `click_freq`     : frequency of the click tone in Hz (default 1000 Hz)
/// - `click_duration` : duration of each click in seconds (default 0.1 s)
/// - `length`         : total output length in samples (`None` → auto)
///
/// # Returns
/// `Vec<f64>` of the synthesised click track.
pub fn clicks(
    times: Option<&[f64]>,
    frames: Option<&[usize]>,
    sr: u32,
    hop_length: usize,
    click_freq: f64,
    click_duration: f64,
    length: Option<usize>,
) -> Result<Vec<f64>, String> {
    if times.is_none() && frames.is_none() {
        return Err(
            "at least one of \"times\" or \"frames\" must be provided".to_string()
        );
    }
    if click_freq <= 0.0 {
        return Err(format!("\"click_freq\" must be positive, got {}", click_freq));
    }
    if click_duration <= 0.0 {
        return Err(format!(
            "\"click_duration\" must be positive, got {}",
            click_duration
        ));
    }

    let sr_f = sr as f64;

    // Build the click sample (cosine tone * half-cosine window)
    let click_len = (click_duration * sr_f).round() as usize;
    if click_len == 0 {
        return Err("click_duration is too short (results in 0 samples)".to_string());
    }

    let click_sample: Vec<f64> = (0..click_len)
        .map(|i| {
            let t = i as f64 / sr_f;
            // tone
            let tone_val = (2.0 * PI * click_freq * t).sin();
            // half-cosine window: cos(linspace(0, π, click_len))
            let window = (PI * i as f64 / (click_len - 1).max(1) as f64).cos();
            tone_val * window
        })
        .collect();

    // Normalise to peak = 0.5 (matching librosa: divide by 2*max)
    let max_abs = click_sample
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let click_sample: Vec<f64> = if max_abs > 0.0 {
        click_sample.iter().map(|&v| v / (2.0 * max_abs)).collect()
    } else {
        click_sample
    };

    // Convert times / frames → sample positions
    let mut positions: Vec<usize> = Vec::new();
    if let Some(ts) = times {
        for &t in ts {
            positions.push((t * sr_f).round() as usize);
        }
    }
    if let Some(fs) = frames {
        for &f in fs {
            positions.push(f * hop_length);
        }
    }

    // Determine output length
    let auto_len = positions.iter().cloned().max().unwrap_or(0) + click_len;
    let total = length.unwrap_or(auto_len).max(1);

    // Place clicks
    let mut y = vec![0.0f64; total];
    for start in positions {
        if start >= total {
            continue;
        }
        let end = (start + click_len).min(total);
        let count = end - start;
        for j in 0..count {
            y[start + j] += click_sample[j];
        }
    }

    Ok(y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ---- tone ----

    #[test]
    fn test_tone_length() {
        let y = tone(440.0, 22050, Some(22050), None, None).unwrap();
        assert_eq!(y.len(), 22050);
    }

    #[test]
    fn test_tone_duration() {
        let y = tone(440.0, 22050, None, Some(1.0), None).unwrap();
        assert_eq!(y.len(), 22050);
    }

    #[test]
    fn test_tone_length_priority_over_duration() {
        // length=100 should take priority, not duration=1.0 (22050 samples)
        let y = tone(440.0, 22050, Some(100), Some(1.0), None).unwrap();
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_tone_values() {
        let sr = 22050u32;
        let freq = 440.0f64;
        let phi = -PI * 0.5;
        let y = tone(freq, sr, Some(10), None, Some(phi)).unwrap();
        for (i, &v) in y.iter().enumerate() {
            let expected = (2.0 * PI * freq * i as f64 / sr as f64 + phi).cos();
            assert!(
                (v - expected).abs() < 1e-12,
                "sample {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_tone_default_phi_is_sine() {
        // cos(x - π/2) = sin(x), so sample[0] should be ~0.0 and sample[1] > 0
        let y = tone(440.0, 22050, Some(100), None, None).unwrap();
        assert!(y[0].abs() < 1e-10, "first sample should be ~0 (sine start)");
    }

    #[test]
    fn test_tone_error_no_length_or_duration() {
        let result = tone(440.0, 22050, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_tone_error_bad_frequency() {
        let result = tone(-1.0, 22050, Some(100), None, None);
        assert!(result.is_err());
        let result2 = tone(0.0, 22050, Some(100), None, None);
        assert!(result2.is_err());
    }

    #[test]
    fn test_tone_all_values_in_range() {
        let y = tone(440.0, 22050, Some(22050), None, None).unwrap();
        for &v in &y {
            assert!(v >= -1.0 && v <= 1.0, "sample out of range: {}", v);
        }
    }

    // ---- chirp ----

    #[test]
    fn test_chirp_length() {
        let y = chirp(110.0, 880.0, 22050, Some(22050), None, false, None).unwrap();
        assert_eq!(y.len(), 22050);
    }

    #[test]
    fn test_chirp_duration() {
        let y = chirp(110.0, 880.0, 22050, None, Some(1.0), false, None).unwrap();
        assert_eq!(y.len(), 22050);
    }

    #[test]
    fn test_chirp_length_priority_over_duration() {
        let y = chirp(110.0, 880.0, 22050, Some(50), Some(1.0), false, None).unwrap();
        assert_eq!(y.len(), 50);
    }

    #[test]
    fn test_chirp_linear_values() {
        let sr = 22050u32;
        let fmin = 110.0f64;
        let fmax = 880.0f64;
        let dur = 1.0f64;
        let phi = -PI * 0.5;
        let y = chirp(fmin, fmax, sr, None, Some(dur), true, Some(phi)).unwrap();
        // spot-check a few samples
        let sr_f = sr as f64;
        for i in [0usize, 1, 100, 1000] {
            let t = i as f64 / sr_f;
            let phase = 2.0 * PI * (fmin * t + (fmax - fmin) * t * t / (2.0 * dur));
            let expected = (phi + phase).cos();
            assert!(
                (y[i] - expected).abs() < 1e-12,
                "linear chirp sample {}: got {}, expected {}",
                i,
                y[i],
                expected
            );
        }
    }

    #[test]
    fn test_chirp_log_values() {
        let sr = 22050u32;
        let fmin = 110.0f64;
        let fmax = 880.0f64;
        let dur = 1.0f64;
        let phi = -PI * 0.5;
        let y = chirp(fmin, fmax, sr, None, Some(dur), false, Some(phi)).unwrap();
        let sr_f = sr as f64;
        let k = fmax / fmin;
        let log_k = k.ln();
        for i in [0usize, 1, 100, 1000] {
            let t = i as f64 / sr_f;
            let phase = 2.0 * PI * fmin * dur / log_k * (k.powf(t / dur) - 1.0);
            let expected = (phi + phase).cos();
            assert!(
                (y[i] - expected).abs() < 1e-12,
                "log chirp sample {}: got {}, expected {}",
                i,
                y[i],
                expected
            );
        }
    }

    #[test]
    fn test_chirp_all_in_range() {
        let y = chirp(110.0, 880.0, 22050, Some(22050), None, false, None).unwrap();
        for &v in &y {
            assert!(v >= -1.0 && v <= 1.0, "chirp sample out of range: {}", v);
        }
    }

    #[test]
    fn test_chirp_error_no_length_or_duration() {
        let result = chirp(110.0, 880.0, 22050, None, None, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_chirp_error_bad_fmin() {
        let result = chirp(-1.0, 880.0, 22050, Some(100), None, false, None);
        assert!(result.is_err());
    }

    // ---- clicks ----

    #[test]
    fn test_clicks_from_times() {
        let times = [0.0f64, 0.5, 1.0];
        let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.01, Some(22051)).unwrap();
        assert_eq!(y.len(), 22051);
        // Peak near sample 0
        let peak0 = y[0..10].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(peak0 > 0.01, "expected a click near sample 0, max={}", peak0);
    }

    #[test]
    fn test_clicks_from_frames() {
        let frames = [0usize, 10, 20];
        let hop = 512;
        let y = clicks(None, Some(&frames), 22050, hop, 1000.0, 0.01, Some(22050)).unwrap();
        assert_eq!(y.len(), 22050);
        // Click at frame 10 starts at sample 5120
        let start = 10 * hop;
        let peak = y[start..start + 50].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(peak > 0.01, "expected click near sample {}", start);
    }

    #[test]
    fn test_clicks_auto_length() {
        let times = [0.0f64, 1.0];
        let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.1, None).unwrap();
        // last click at sample 22050, click_len = 2205 → auto length = 22050 + 2205
        let click_len = (0.1 * 22050.0f64).round() as usize;
        assert_eq!(y.len(), 22050 + click_len);
    }

    #[test]
    fn test_clicks_error_no_times_or_frames() {
        let result = clicks(None, None, 22050, 512, 1000.0, 0.1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_clicks_both_times_and_frames() {
        // Providing both should work and union the positions
        let times = [0.0f64];
        let frames = [10usize];
        let y = clicks(Some(&times), Some(&frames), 22050, 512, 1000.0, 0.01, Some(22050));
        assert!(y.is_ok());
    }

    #[test]
    fn test_clicks_amplitude_bounded() {
        let times = [0.0f64, 0.25, 0.5, 0.75];
        let y = clicks(Some(&times), None, 22050, 512, 1000.0, 0.1, Some(22050)).unwrap();
        for &v in &y {
            assert!(v.abs() <= 1.0, "clicks amplitude out of range: {}", v);
        }
    }

    #[test]
    fn test_clicks_error_bad_click_freq() {
        let times = [0.0f64];
        let result = clicks(Some(&times), None, 22050, 512, -1.0, 0.1, None);
        assert!(result.is_err());
    }
}

use hound;

// ---------------------------------------------------------------------------
// to_mono
// ---------------------------------------------------------------------------
// [docs] librosa.to_mono(y)
// Convert an audio signal to mono by averaging samples across channels.
//
// Python source:
//   def to_mono(y):
//       if y.ndim > 1:
//           y = np.mean(y, axis=-2)   # average over channels (axis 0 for (C, N))
//       return y
//
// Here we accept interleaved multi-channel samples and the channel count.
// For mono input (channels == 1) the input is returned as-is.
/// Convert interleaved multi-channel audio samples to mono.
///
/// `y` contains interleaved samples (L₀ R₀ L₁ R₁ … for stereo).
/// Returns a `Vec<f32>` where each frame is the average across all channels.
pub fn to_mono(y: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return y.to_vec();
    }
    let n_frames = y.len() / channels;
    (0..n_frames)
        .map(|frame| {
            let sum: f32 = (0..channels).map(|ch| y[frame * channels + ch]).sum();
            sum / channels as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// get_duration
// ---------------------------------------------------------------------------
// [docs] librosa.get_duration(*, y=None, sr=22050, ...)
//
// Python:
//   if y is not None:
//       return float(len(y)) / sr
//
// We always take n_samples and sr directly.
/// Compute the duration (in seconds) of an audio signal.
///
/// `n_samples` is the total number of (mono) samples, `sr` is the sample rate.
pub fn get_duration(n_samples: usize, sr: u32) -> f64 {
    n_samples as f64 / sr as f64
}

// ---------------------------------------------------------------------------
// get_samplerate
// ---------------------------------------------------------------------------
// [docs] librosa.get_samplerate(path)
// Get the sampling rate for a given file.
//
// Python: sf.info(path).samplerate
//
// We read the WAV header via `hound` and return spec.sample_rate.
/// Get the sampling rate of a WAV file without loading its audio data.
pub fn get_samplerate(path: &str) -> Result<u32, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open '{}': {}", path, e))?;
    Ok(reader.spec().sample_rate)
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
// [docs] librosa.load(path, *, sr=22050, mono=True, offset=0.0, duration=None, ...)
// Load an audio file as a floating point time series.
//
// Python (simplified):
//   y, sr_native = __soundfile_load(path, offset, duration, dtype)
//   if mono:
//       y = to_mono(y)
//   if sr is not None:
//       y = resample(y, orig_sr=sr_native, target_sr=sr)
//   return y, sr
//
// Rust implementation supports WAV files only (via hound).
// Resampling is not yet implemented; if `sr` differs from the native rate
// the function returns an error advising to pass sr=None.
/// Load a WAV audio file as a `Vec<f32>` time series.
///
/// # Parameters
/// - `path`     : path to the WAV file
/// - `sr`       : target sample rate (`None` keeps native rate)
/// - `mono`     : if `true`, mix down to mono
/// - `offset`   : start reading after this many seconds
/// - `duration` : maximum duration to read in seconds (`None` = whole file)
///
/// # Returns
/// `(samples, sample_rate)` where `samples` is a flat `Vec<f32>`.
pub fn load(
    path: &str,
    sr: Option<u32>,
    mono: bool,
    offset: f32,
    duration: Option<f32>,
) -> Result<(Vec<f32>, u32), String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("Failed to open '{}': {}", path, e))?;

    let spec = reader.spec();
    let sr_native = spec.sample_rate;
    let channels = spec.channels as usize;
    let bits_per_sample = spec.bits_per_sample;
    let sample_fmt = spec.sample_format;

    // Validate sr argument
    if let Some(target_sr) = sr {
        if target_sr != sr_native {
            return Err(format!(
                "Resampling from {} Hz to {} Hz is not yet implemented. \
                 Pass sr=None to use the native sample rate.",
                sr_native, target_sr
            ));
        }
    }

    let out_sr = sr.unwrap_or(sr_native);

    // Compute frame-level offset and max length
    let offset_samples = (offset * sr_native as f32).round() as usize * channels;
    let max_samples: Option<usize> =
        duration.map(|d| (d * sr_native as f32).round() as usize * channels);

    // Read all raw samples as f32
    let all_samples: Vec<f32> = match sample_fmt {
        hound::SampleFormat::Float => {
            // 32-bit float WAV
            reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| format!("Read error: {}", e)))
                .collect::<Result<Vec<f32>, String>>()?
        }
        hound::SampleFormat::Int => {
            let max_val = (1_i64 << (bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val).map_err(|e| format!("Read error: {}", e)))
                .collect::<Result<Vec<f32>, String>>()?
        }
    };

    // Apply offset slice
    let sliced = if offset_samples < all_samples.len() {
        &all_samples[offset_samples..]
    } else {
        &[]
    };

    // Apply duration cap
    let sliced = match max_samples {
        Some(n) => &sliced[..n.min(sliced.len())],
        None => sliced,
    };

    // Convert to mono if requested
    let samples = if mono {
        to_mono(sliced, channels)
    } else {
        sliced.to_vec()
    };

    Ok((samples, out_sr))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;


    /// Helper: write a minimal WAV file to a temp file and return its path.
    fn write_test_wav(
        samples: &[i16],
        channels: u16,
        sample_rate: u32,
    ) -> tempfile::NamedTempFile {
        use hound::{SampleFormat, WavSpec, WavWriter};
        let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(tmp.path(), spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        tmp
    }

    // ---- to_mono ----

    #[test]
    fn test_to_mono_already_mono() {
        let y = vec![0.5_f32, -0.5, 1.0];
        assert_eq!(to_mono(&y, 1), y);
    }

    #[test]
    fn test_to_mono_stereo() {
        // L=1.0, R=0.0  →  0.5 per frame
        let y = vec![1.0_f32, 0.0, 1.0, 0.0];
        let mono = to_mono(&y, 2);
        assert_eq!(mono.len(), 2);
        for v in &mono {
            assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {}", v);
        }
    }

    #[test]
    fn test_to_mono_four_channel() {
        // All frames: 1.0, 2.0, 3.0, 4.0 → average 2.5
        let y = vec![1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let mono = to_mono(&y, 4);
        assert_eq!(mono.len(), 2);
        for v in &mono {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    // ---- get_duration ----

    #[test]
    fn test_get_duration_exact() {
        // 44100 samples at 44100 Hz = 1.0 second
        assert!((get_duration(44100, 44100) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_duration_half_second() {
        assert!((get_duration(11025, 22050) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_get_duration_zero() {
        assert_eq!(get_duration(0, 22050), 0.0);
    }

    // ---- get_samplerate ----

    #[test]
    fn test_get_samplerate() {
        let samples: Vec<i16> = vec![0; 100];
        let tmp = write_test_wav(&samples, 1, 44100);
        let sr = get_samplerate(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(sr, 44100);
    }

    #[test]
    fn test_get_samplerate_22050() {
        let samples: Vec<i16> = vec![0; 100];
        let tmp = write_test_wav(&samples, 1, 22050);
        let sr = get_samplerate(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(sr, 22050);
    }

    // ---- load ----

    #[test]
    fn test_load_mono_wav() {
        let samples: Vec<i16> = (0..100).map(|i| (i * 100) as i16).collect();
        let tmp = write_test_wav(&samples, 1, 22050);
        let (y, sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, None).unwrap();
        assert_eq!(sr, 22050);
        assert_eq!(y.len(), 100);
        // first sample: 0 / 32768 ≈ 0.0
        assert!((y[0]).abs() < 1e-5);
    }

    #[test]
    fn test_load_stereo_to_mono() {
        // stereo: L=16384, R=-16384 every frame → mono = 0
        let samples: Vec<i16> = (0..50).flat_map(|_| [16384_i16, -16384]).collect();
        let tmp = write_test_wav(&samples, 2, 22050);
        let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, None).unwrap();
        assert_eq!(y.len(), 50);
        for v in &y {
            assert!(v.abs() < 1e-5, "expected 0.0, got {}", v);
        }
    }

    #[test]
    fn test_load_duration_trim() {
        let samples: Vec<i16> = vec![1000_i16; 22050]; // 1 second at 22050 Hz
        let tmp = write_test_wav(&samples, 1, 22050);
        let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, Some(0.5)).unwrap();
        // 0.5 s * 22050 Hz = 11025 samples
        assert_eq!(y.len(), 11025);
    }

    #[test]
    fn test_load_offset() {
        let mut samples: Vec<i16> = vec![0i16; 22050]; // silence for 1s
        for s in samples[11025..].iter_mut() {
            *s = 10000; // loud second half
        }
        let tmp = write_test_wav(&samples, 1, 22050);
        // offset 0.5s → should start in the loud region
        let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 0.5, None).unwrap();
        assert_eq!(y.len(), 11025);
        assert!(y[0] > 0.1, "expected loud samples after offset");
    }

    #[test]
    fn test_load_sr_mismatch_error() {
        let samples: Vec<i16> = vec![0i16; 100];
        let tmp = write_test_wav(&samples, 1, 22050);
        let result = load(tmp.path().to_str().unwrap(), Some(44100), true, 0.0, None);
        assert!(result.is_err(), "expected error for mismatched sr");
    }

    #[test]
    fn test_load_bad_path() {
        let result = load("nonexistent_file.wav", None, true, 0.0, None);
        assert!(result.is_err());
    }
}

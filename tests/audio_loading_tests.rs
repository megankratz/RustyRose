/// Root-level integration tests for the audio_loading module.
///
/// These tests exercise `load`, `to_mono`, `get_duration`, and `get_samplerate`
/// as public API from the crate root, using real (temporary) WAV files on disk.
use rusty_rose::core_io::{get_duration, get_samplerate, load, to_mono};

use hound::{SampleFormat, WavSpec, WavWriter};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_wav(samples: &[i16], channels: u16, sample_rate: u32) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("failed to create temp file");
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

// ---------------------------------------------------------------------------
// to_mono
// ---------------------------------------------------------------------------

#[test]
fn integration_to_mono_mono_passthrough() {
    let y = vec![0.25_f32, -0.25, 0.5, -0.5];
    assert_eq!(to_mono(&y, 1), y);
}

#[test]
fn integration_to_mono_stereo_average() {
    // L = 0.8, R = 0.4 → expected mono = 0.6 per frame
    let y = vec![0.8_f32, 0.4, 0.8, 0.4];
    let m = to_mono(&y, 2);
    assert_eq!(m.len(), 2);
    assert!((m[0] - 0.6).abs() < 1e-6);
    assert!((m[1] - 0.6).abs() < 1e-6);
}

#[test]
fn integration_to_mono_zero_signal() {
    let y = vec![0.0_f32; 8];
    let m = to_mono(&y, 2);
    assert!(m.iter().all(|&v| v == 0.0));
}

#[test]
fn integration_to_mono_four_channel() {
    // each quad-frame sums to 8.0, avg = 2.0
    let frame = [1.0_f32, 2.0, 3.0, 4.0];
    let y: Vec<f32> = frame.iter().cycle().take(16).cloned().collect();
    let m = to_mono(&y, 4);
    assert_eq!(m.len(), 4);
    for v in &m {
        assert!((v - 2.5).abs() < 1e-6, "expected 2.5, got {}", v);
    }
}

// ---------------------------------------------------------------------------
// get_duration
// ---------------------------------------------------------------------------

#[test]
fn integration_get_duration_one_second() {
    assert!((get_duration(44100, 44100) - 1.0).abs() < 1e-9);
}

#[test]
fn integration_get_duration_half_second() {
    assert!((get_duration(11025, 22050) - 0.5).abs() < 1e-9);
}

#[test]
fn integration_get_duration_zero() {
    assert_eq!(get_duration(0, 16000), 0.0);
}

#[test]
fn integration_get_duration_fractional() {
    // 100 samples @ 16000 Hz = 0.00625 s
    let expected = 100.0 / 16000.0;
    assert!((get_duration(100, 16000) - expected).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// get_samplerate
// ---------------------------------------------------------------------------

#[test]
fn integration_get_samplerate_44100() {
    let tmp = make_wav(&[0i16; 10], 1, 44100);
    assert_eq!(get_samplerate(tmp.path().to_str().unwrap()).unwrap(), 44100);
}

#[test]
fn integration_get_samplerate_22050() {
    let tmp = make_wav(&[0i16; 10], 1, 22050);
    assert_eq!(get_samplerate(tmp.path().to_str().unwrap()).unwrap(), 22050);
}

#[test]
fn integration_get_samplerate_16000() {
    let tmp = make_wav(&[0i16; 10], 1, 16000);
    assert_eq!(get_samplerate(tmp.path().to_str().unwrap()).unwrap(), 16000);
}

#[test]
fn integration_get_samplerate_bad_path() {
    assert!(get_samplerate("no_such_file.wav").is_err());
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

#[test]
fn integration_load_mono_full() {
    // 22050 samples → 1 second at 22050 Hz
    let samples: Vec<i16> = (0..22050_i16).map(|i| i % 1000).collect();
    let tmp = make_wav(&samples, 1, 22050);
    let (y, sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, None).unwrap();
    assert_eq!(sr, 22050);
    assert_eq!(y.len(), 22050);
    // Verify normalisation: i16::MAX maps to 1.0 (approximately)
    let max_sample = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_sample <= 1.0, "samples should not exceed 1.0");
}

#[test]
fn integration_load_stereo_to_mono() {
    // L = i16::MAX/2, R = -(i16::MAX/2) → mono should be ≈ 0
    let samples: Vec<i16> = (0..100).flat_map(|_| [16384_i16, -16384]).collect();
    let tmp = make_wav(&samples, 2, 22050);
    let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, None).unwrap();
    assert_eq!(y.len(), 100, "expected 100 mono frames");
    for &v in &y {
        assert!(v.abs() < 1e-4, "expected near-zero mono, got {}", v);
    }
}

#[test]
fn integration_load_stereo_raw() {
    // Without mono mixing, interleaved samples are returned
    let samples: Vec<i16> = (0..50).flat_map(|_| [10000_i16, -10000i16]).collect(); // 50 stereo frames
    let tmp = make_wav(&samples, 2, 22050);
    let (y, sr) = load(tmp.path().to_str().unwrap(), None, false, 0.0, None).unwrap();
    assert_eq!(sr, 22050);
    assert_eq!(y.len(), 100, "expected 100 interleaved samples (50 frames × 2 ch)");
}

#[test]
fn integration_load_duration_clip() {
    let samples: Vec<i16> = vec![5000_i16; 22050]; // 1 second
    let tmp = make_wav(&samples, 1, 22050);
    let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, Some(0.5)).unwrap();
    assert_eq!(y.len(), 11025, "expected half-second of samples");
}

#[test]
fn integration_load_offset() {
    // First half: silence; second half: loud
    let mut samples = vec![0_i16; 22050];
    for s in samples[11025..].iter_mut() {
        *s = 20000;
    }
    let tmp = make_wav(&samples, 1, 22050);
    let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 0.5, None).unwrap();
    assert_eq!(y.len(), 11025);
    // All loaded samples should be loud (>0.3)
    for &v in &y {
        assert!(v > 0.3, "expected loud samples, got {}", v);
    }
}

#[test]
fn integration_load_offset_and_duration() {
    // 3 seconds of varying content; extract middle second
    let mut samples = vec![0_i16; 66150]; // 3 s @ 22050
    // middle second: samples 22050..44100 = 10000
    for s in samples[22050..44100].iter_mut() {
        *s = 10000;
    }
    let tmp = make_wav(&samples, 1, 22050);
    let (y, _sr) = load(tmp.path().to_str().unwrap(), None, true, 1.0, Some(1.0)).unwrap();
    assert_eq!(y.len(), 22050);
    for &v in &y {
        assert!(v > 0.1, "expected loud middle segment, got {}", v);
    }
}

#[test]
fn integration_load_sr_none_native() {
    let samples: Vec<i16> = vec![0i16; 100];
    let tmp = make_wav(&samples, 1, 44100);
    let (_, sr) = load(tmp.path().to_str().unwrap(), None, true, 0.0, None).unwrap();
    assert_eq!(sr, 44100);
}

#[test]
fn integration_load_sr_mismatch_returns_error() {
    let samples: Vec<i16> = vec![0i16; 100];
    let tmp = make_wav(&samples, 1, 22050);
    let result = load(tmp.path().to_str().unwrap(), Some(44100), true, 0.0, None);
    assert!(result.is_err(), "expected error for unsupported resampling");
}

#[test]
fn integration_load_sr_same_as_native_ok() {
    let samples: Vec<i16> = vec![0i16; 100];
    let tmp = make_wav(&samples, 1, 22050);
    let result = load(tmp.path().to_str().unwrap(), Some(22050), true, 0.0, None);
    assert!(result.is_ok());
}

#[test]
fn integration_load_file_not_found() {
    assert!(load("does_not_exist.wav", None, true, 0.0, None).is_err());
}

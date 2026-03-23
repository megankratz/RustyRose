// https://librosa.org/doc/latest/core.html#frequency-range-generation
use crate::core_io::frequency_unit_conversion::*;

// -------------------------------------------------------------------
// FFT_FREQUENCIES -> 
// parameters: sr (sample rate), n_fft (FFT window size)
// returns: array of frequencies corresponding to the FFT bins
// --------------------------------------------------------------------
pub fn fft_frequencies(sr: f32, n_fft: i32) -> Vec<f32> {
    return rfftfreq(n_fft, 1.0 / sr);
}

// rfftfreq - helper for fft_frequencies, calculates the frequencies for the positive FFT bins
pub fn rfftfreq(n: i32, d: f32) -> Vec<f32> {
    let val = 1.0 / (n as f32 * d);
    let N = n / 2 + 1;
    (0..N).map(|i| i as f32 * val).collect()
}

// ---------------------------------------------------------------------
// CQT_FREQUENCIES -> comppute the center frequencies of the CQT bins
// parameters: n_bins (number of frequency bins), fmin (minimum frequency), bins_per_octave, tuning
// returns: array of frequencies corresponding to the CQT bins
// -----------------------------------------------------------------------
pub fn cqt_frequencies(n_bins: i32, fmin: f32, bins_per_octave: i32, tuning: f32) -> Vec<f32> {
    let correction = 2.0_f32.powf(tuning / bins_per_octave as f32);
    let frequencies: Vec<f32> = (0..n_bins)
        .map(|i| 2.0_f32.powf(i as f32 / bins_per_octave as f32))
        .collect();
    return frequencies.iter().map(|&f| correction * fmin * f).collect();
}

// ---------------------------------------------------------------------
// MEL_FREQUENCIES -> compute the center frequencies of the Mel bins
// parameters: n_mels (number of Mel bins), fmin (minimum frequency), fmax (maximum frequency), htk (use HTK formula for Mel scale)
// returns: array of frequencies corresponding to the Mel bins
// ----------------------------------------------------------------------
pub fn mel_frequencies(n_mels: i32, fmin: f32, fmax: f32, htk: bool) -> Vec<f32> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);
    let mels: Vec<f32> = (0..n_mels)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels - 1) as f32)
        .collect();
    return mels.iter().map(|&m| mel_to_hz(m, htk)).collect();
}

// ---------------------------------------------------------------------
// TEMPO_FREQUENCIES -> compute the center frequencies of the tempo bins
// parameters: n_bins (number of tempo bins), sr (sample rate), hop_length (hop length in samples)
// returns: array of frequencies corresponding to the tempo bins
// ----------------------------------------------------------------------
pub fn tempo_frequencies(n_bins: i32, sr: f32, hop_length: i32) -> Vec<f32> {
    let mut bin_frequencies = vec![0.0; n_bins as usize];
    bin_frequencies[0] = std::f32::INFINITY;
    for i in 1..n_bins {
        bin_frequencies[i as usize] = 60.0 * sr / (hop_length as f32 * i as f32);
    }
    return bin_frequencies;
}

// ---------------------------------------------------------------------
// FOURIER_TEMPO_FREQUENCIES -> compute the center frequencies of the tempo bins using Fourier analysis
// parameters: sr (sample rate), win_length (window length in samples), hop_length (hop length in samples)
// returns: array of frequencies corresponding to the tempo bins
// ----------------------------------------------------------------------
pub fn fourier_tempo_frequencies(sr: f32, win_length: i32, hop_length: i32) -> Vec<f32> {
    return fft_frequencies(sr * 60.0 / hop_length as f32, win_length);
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;

    // Test rfftfreq
    #[test]
    fn test_rfftfreq() {
        let n = 8;
        let d = 0.5;
        let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(rfftfreq(n, d), expected);
        assert_eq!(rfftfreq(4, 1.0), vec![0.0, 0.25, 0.5]);
        assert_eq!(rfftfreq(5, 1.0), vec![0.0, 0.2, 0.4]);
    }

    // Test fft_frequencies
    #[test]
    fn test_fft_frequencies() {
        let sr = 22050.0;
        let n_fft = 2048;
        let expected = rfftfreq(n_fft, 1.0 / sr);
        assert_eq!(fft_frequencies(sr, n_fft), expected);
        assert_eq!(
            fft_frequencies(44100.0, 4096),
            rfftfreq(4096, 1.0 / 44100.0)
        );
        assert_eq!(
            fft_frequencies(16000.0, 1024),
            rfftfreq(1024, 1.0 / 16000.0)
        );
    }

    // Test cqt_frequencies\
    #[test]
    fn test_cqt_frequencies() {
        let n_bins = 12;
        let fmin = 32.7032; // C1
        let bins_per_octave = 12;
        let tuning = 0.0;
        let expected: Vec<f32> = (0..n_bins)
            .map(|i| 2.0_f32.powf(i as f32 / bins_per_octave as f32) * fmin)
            .collect();
        assert_eq!(
            cqt_frequencies(n_bins, fmin, bins_per_octave, tuning),
            expected
        );
    }

    // Test mel_frequencies
    #[test]
    fn test_mel_frequencies() {
        let n_mels = 10;
        let fmin = 20.0;
        let fmax = 20000.0;
        let htk = false;
        let expected: Vec<f32> = (0..n_mels)
            .map(|i| {
                let mel = hz_to_mel(fmin, htk)
                    + (hz_to_mel(fmax, htk) - hz_to_mel(fmin, htk)) * i as f32
                        / (n_mels - 1) as f32;
                mel_to_hz(mel, htk)
            })
            .collect();
        assert_eq!(mel_frequencies(n_mels, fmin, fmax, htk), expected);
    }
    // Test tempo_frequencies
    #[test]
    fn test_tempo_frequencies() {
        let n_bins = 5;
        let sr = 22050.0;
        let hop_length = 512;
        let expected = vec![
            std::f32::INFINITY,
            60.0 * sr / (hop_length as f32 * 1.0),
            60.0 * sr / (hop_length as f32 * 2.0),
            60.0 * sr / (hop_length as f32 * 3.0),
            60.0 * sr / (hop_length as f32 * 4.0),
        ];
        assert_eq!(tempo_frequencies(n_bins, sr, hop_length), expected);
    }

    // Test fourier_tempo_frequencies
    #[test]
    fn test_fourier_tempo_frequencies() {
        let sr = 22050.0;
        let win_length = 384;
        let hop_length = 512;
        let expected = fft_frequencies(sr * 60.0 / hop_length as f32, win_length);
        assert_eq!(
            fourier_tempo_frequencies(sr, win_length, hop_length),
            expected
        );
    }
}

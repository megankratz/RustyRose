use crate::core_io::frequency_unit_conversion::*;

// fft_frequencies
/*def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

fft.rfftfreq function 
if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = arange(0, N, dtype=int, device=device)
    return results * val
*/

pub fn rfftfreq(n: i32, d: f32) -> Vec<f32> {
    let val = 1.0 / (n as f32 * d);
    let N = n / 2 + 1;
    (0..N).map(|i| i as f32 * val).collect()
}

pub fn fft_frequencies(sr: f32, n_fft: i32) -> Vec<f32> {
    return rfftfreq(n_fft, 1.0 / sr);
}

// cqt_frequencies
/* 
[docs]def cqt_frequencies(
    n_bins: int, *, fmin: float, bins_per_octave: int = 12, tuning: float = 0.0
) -> np.ndarray:
    correction: float = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies: np.ndarray = 2.0 ** (
        np.arange(0, n_bins, dtype=float) / bins_per_octave
    )

    return correction * fmin * frequencies
*/
pub fn cqt_frequencies(n_bins: i32, fmin: f32, bins_per_octave: i32, tuning: f32) -> Vec<f32> {
    let correction = 2.0_f32.powf(tuning / bins_per_octave as f32);
    let frequencies: Vec<f32> = (0..n_bins)
        .map(|i| 2.0_f32.powf(i as f32 / bins_per_octave as f32))
        .collect();
    return frequencies.iter().map(|&f| correction * fmin * f).collect();
}

// mel_frequencies
/* 
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz
*/
pub fn mel_frequencies(n_mels: i32, fmin: f32, fmax: f32, htk: bool) -> Vec<f32> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);
    let mels: Vec<f32> = (0..n_mels)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels - 1) as f32)
        .collect();
    return mels.iter().map(|&m| mel_to_hz(m, htk)).collect();
}

// tempo_frequencies
/* 
    bin_frequencies = np.zeros(int(n_bins), dtype=np.float64)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies
*/
pub fn tempo_frequencies(n_bins: i32, sr: f32, hop_length: i32) -> Vec<f32> {
    let mut bin_frequencies = vec![0.0; n_bins as usize];
    bin_frequencies[0] = std::f32::INFINITY;
    for i in 1..n_bins {
        bin_frequencies[i as usize] = 60.0 * sr / (hop_length as f32 * i as f32);
    }
    return bin_frequencies;
}

// fourier_tempo_frequencies
/*
[docs]def fourier_tempo_frequencies(
    *, sr: float = 22050, win_length: int = 384, hop_length: int = 512
) -> np.ndarray:
    # sr / hop_length gets the frame rate
    # multiplying by 60 turns frames / sec into frames / minute
    return fft_frequencies(sr=sr * 60 / float(hop_length), n_fft=win_length)
*/
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
        assert_eq!(fft_frequencies(44100.0, 4096), rfftfreq(4096, 1.0 / 44100.0));
        assert_eq!(fft_frequencies(16000.0, 1024), rfftfreq(1024, 1.0 / 16000.0));
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
        assert_eq!(cqt_frequencies(n_bins, fmin, bins_per_octave, tuning), expected);
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
                let mel = hz_to_mel(fmin, htk) + (hz_to_mel(fmax, htk) - hz_to_mel(fmin, htk)) * i as f32 / (n_mels - 1) as f32;
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
        assert_eq!(fourier_tempo_frequencies(sr, win_length, hop_length), expected);
    }

}
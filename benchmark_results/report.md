# Performance Comparison: RustyRose vs Librosa


## Time-Domain Processing

| Function | RustyRose (us) | Librosa (us) | Speedup |
| :--- | :--- | :--- | :--- |
| autocorrelate | 42.29 | 37.76 | 0.89x |
| lpc | 13.85 | 53.31 | 3.85x |
| mu_compress | 8.82 | 24.03 | 2.72x |
| zero_crossing | 1.09 | 4.90 | 4.50x |

## Spectral Scaling & RMS

| Function | RustyRose (us) | Librosa (us) | Speedup |
| :--- | :--- | :--- | :--- |
| power_to_db | 4.30 | 6.24 | 1.45x |
| rms | 4.01 | 26.63 | 6.65x |

## Spectral Features (Tonnetz/Poly)

| Function | RustyRose (us) | Librosa (us) | Speedup |
| :--- | :--- | :--- | :--- |
| poly_features/10 | 3.20 | 63.50 | 19.84x |
| poly_features/100 | 16.36 | 105.29 | 6.44x |
| poly_features/1000 | 160.38 | 1247.73 | 7.78x |
| tonnetz/10 | 0.90 | 44.51 | 49.28x |
| tonnetz/100 | 2.85 | 43.89 | 15.39x |
| tonnetz/1000 | 22.25 | 84.57 | 3.80x |

## Utilities & Audio Effects

| Function | RustyRose (us) | Librosa (us) | Speedup |
| :--- | :--- | :--- | :--- |
| preemphasis | 0.96 | 152.35 | 159.27x |
| time_to_samples | 0.00 | 2.62 | 4523.12x |
| trim | 4.23 | 51.55 | 12.20x |

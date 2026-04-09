import librosa
import numpy as np
import timeit
import json
import sys
import os

def benchmark_tonnetz(n_frames_list):
    results = {}
    for t in n_frames_list:
        chroma = np.ones((12, t))
        # Warmup
        librosa.feature.tonnetz(chroma=chroma)
        
        timer = timeit.Timer(lambda: librosa.feature.tonnetz(chroma=chroma))
        n_iters = 100 if t < 1000 else 10
        t_exec = timer.timeit(number=n_iters) / n_iters
        results[str(t)] = t_exec * 1e9  # Convert to nanoseconds
        print(f"Tonnetz ({t} frames): {t_exec*1e3:.3f} ms")
    return results

def benchmark_poly_features(n_frames_list):
    results = {}
    n_freqs = 100
    freq = np.linspace(0.0, 5000.0, n_freqs)
    order = 2
    for t in n_frames_list:
        s = np.ones((n_freqs, t))
        # Warmup
        librosa.feature.poly_features(S=s, freq=freq, order=order)
        
        timer = timeit.Timer(lambda: librosa.feature.poly_features(S=s, freq=freq, order=order))
        n_iters = 100 if t < 1000 else 10
        t_exec = timer.timeit(number=n_iters) / n_iters
        results[str(t)] = t_exec * 1e9  # Convert to nanoseconds
        print(f"Poly Features ({t} frames): {t_exec*1e3:.3f} ms")
    return results

def benchmark_tempogram(n_frames_list):
    results = {}
    win_length = 384
    hop_length = 1
    for t in n_frames_list:
        onset_envelope = np.ones(t)
        # Warmup
        librosa.feature.tempogram(y=None, onset_envelope=onset_envelope, win_length=win_length, hop_length=hop_length)
        
        timer = timeit.Timer(lambda: librosa.feature.tempogram(y=None, onset_envelope=onset_envelope, win_length=win_length, hop_length=hop_length))
        n_iters = 100 if t < 1000 else 10
        t_exec = timer.timeit(number=n_iters) / n_iters
        results[str(t)] = t_exec * 1e9
        print(f"Tempogram ({t} frames): {t_exec*1e3:.3f} ms")
    return results

# (Removed frequency conversions)

def benchmark_time_domain():
    results = {}
    n = 1024
    y = np.zeros(n)
    
    # autocorrelate
    timer = timeit.Timer(lambda: librosa.autocorrelate(y, axis=0))
    results["autocorrelate"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    # lpc
    timer = timeit.Timer(lambda: librosa.lpc(y, order=12, axis=0))
    results["lpc"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    # zero_crossings
    timer = timeit.Timer(lambda: librosa.zero_crossings(y, threshold=0.0, ref_magnitude=None, pad=True, zero_pos=True, axis=0))
    results["zero_crossing"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    # mu_compress
    timer = timeit.Timer(lambda: librosa.mu_compress(y, mu=255.0, quantize=True))
    results["mu_compress"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    print("Time-domain features benchmarked")
    return results

def benchmark_spectral_scaling():
    results = {}
    n = 1024
    y = np.zeros(n)
    s = np.ones(n)
    
    # power_to_db
    timer = timeit.Timer(lambda: librosa.power_to_db(s, ref=1.0, amin=1e-10, top_db=None))
    results["power_to_db"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    # rms
    timer = timeit.Timer(lambda: librosa.feature.rms(y=y, frame_length=2048, hop_length=512, center=True))
    results["rms"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    print("Spectral scaling features benchmarked")
    return results

def benchmark_utils_effects():
    results = {}
    n = 1024
    y = np.zeros(n)
    
    # time_to_samples
    timer = timeit.Timer(lambda: librosa.time_to_samples(1.0, sr=22050))
    results["time_to_samples"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    # preemphasis
    timer = timeit.Timer(lambda: librosa.effects.preemphasis(y, coef=0.97, zi=None))
    results["preemphasis"] = (timer.timeit(number=1000) / 1000) * 1e9
    
    # trim
    timer = timeit.Timer(lambda: librosa.effects.trim(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512))
    results["trim"] = (timer.timeit(number=100) / 100) * 1e9
    
    print("Utils/effects benchmarked")
    return results

def benchmark_spectral_features(n_frames_list):
    results = {}
    
    # Tonnetz
    for t in n_frames_list:
        chroma = np.ones((12, t))
        # Warmup
        librosa.feature.tonnetz(chroma=chroma)
        timer = timeit.Timer(lambda: librosa.feature.tonnetz(chroma=chroma))
        n_iters = 50 if t < 1000 else 10
        t_exec = timer.timeit(number=n_iters) / n_iters
        results[f"tonnetz/{t}"] = t_exec * 1e9
        print(f"Tonnetz ({t} frames) benchmarked")

    # Poly Features
    n_freqs = 100
    freq = np.linspace(0.0, 5000.0, n_freqs)
    for t in n_frames_list:
        s = np.ones((n_freqs, t))
        # Warmup
        librosa.feature.poly_features(S=s, freq=freq, order=2)
        timer = timeit.Timer(lambda: librosa.feature.poly_features(S=s, freq=freq, order=2))
        n_iters = 50 if t < 1000 else 10
        t_exec = timer.timeit(number=n_iters) / n_iters
        results[f"poly_features/{t}"] = t_exec * 1e9
        print(f"Poly Features ({t} frames) benchmarked")
        
    return results

if __name__ == "__main__":
    n_frames = [10, 100, 1000]
    all_results = {
        "spectral_features": benchmark_spectral_features(n_frames),
        "time_domain": benchmark_time_domain(),
        "spectral_scaling": benchmark_spectral_scaling(),
        "utils_effects": benchmark_utils_effects()
    }
    
    output_path = "benchmark_results/librosa_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {output_path}")

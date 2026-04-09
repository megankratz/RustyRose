import librosa
import numpy as np
import json
import os

def generate_refs():
    # Simple deterministic onset envelope
    # Use 100 frames
    n_frames = 100
    onset_envelope = np.zeros(n_frames)
    onset_envelope[10] = 1.0
    onset_envelope[30] = 0.5
    onset_envelope[50] = 0.8
    onset_envelope[70] = 1.0
    onset_envelope[90] = 0.3
    
    sr = 22050
    hop_length = 512
    win_length = 32 # Small for testing
    
    # 1. Tempogram
    tgram = librosa.feature.tempogram(onset_envelope=onset_envelope, sr=sr, 
                                     hop_length=hop_length, win_length=win_length, 
                                     center=True, window='hann', norm=np.inf)
    
    # 2. Fourier Tempogram
    ftgram = librosa.feature.fourier_tempogram(onset_envelope=onset_envelope, sr=sr,
                                              hop_length=hop_length, win_length=win_length,
                                              center=True, window='hann')
    ftgram_mag = np.abs(ftgram)
    
    # 3. Tempo
    # We need a longer envelope for realistic tempo
    n_frames_long = 1000
    oenv_long = np.zeros(n_frames_long)
    spacing = 20 # ~ 60 * 22050 / (512 * 20) = 129 BPM
    for i in range(0, n_frames_long, spacing):
        oenv_long[i] = 1.0
        
    tempo_val = librosa.feature.tempo(onset_envelope=oenv_long, sr=sr, hop_length=hop_length)[0]
    
    results = {
        "onset_envelope": onset_envelope.tolist(),
        "win_length": win_length,
        "hop_length": 1, # relative to oenv
        "tempogram": tgram.tolist(),
        "fourier_tempogram_mag": ftgram_mag.tolist(),
        "oenv_long": oenv_long.tolist(),
        "tempo": float(tempo_val)
    }
    
    # Output relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, "tests", "rhythm_refs.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"References saved to {output_path}")

if __name__ == "__main__":
    generate_refs()

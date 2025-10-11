import numpy as np
import pandas as pd
import librosa
# converting audio to melspecs
sr = 22050
n_fft = 2048
win_length = int(0.050*sr)
hop_length = int(0.0125*sr)
n_mels = 80
fmin = 125
fmax = 7600
power = 1.0

def preprocess_audio(path):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=25)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0: y = y / rms
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax, power=power
    )
    mel = np.maximum(mel, 1e-6)
    return np.log(mel).T.astype(np.float32)
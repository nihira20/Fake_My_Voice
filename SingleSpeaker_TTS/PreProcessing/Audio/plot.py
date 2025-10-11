import matplotlib.pyplot as plt
import numpy as np
import librosa
mels_padded = np.load("/kaggle/input/tacotron2-inputs-outputs/mels_padded.npy")
mel = mels_padded[0]
mel_T = mel.T
sr = 22050
hop_length = 256
n_mels = 80
fmin = 125
fmax = 7600

n_frames = mel_T.shape[1]
time = np.arange(n_frames) * hop_length / sr
mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
plt.figure(figsize=(10, 4))
plt.imshow(mel_T, aspect="auto", origin="lower",
           extent=[time[0], time[-1], mel_freqs[0], mel_freqs[-1]],
           cmap="inferno")
plt.colorbar(label="amplitude(dB)")
plt.title("mel spec")
plt.xlabel("time in sec")
plt.ylabel("freq(Hz)")
plt.tight_layout()
plt.show()
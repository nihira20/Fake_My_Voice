import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio
import random
from train import model, device

model.eval()

idx = random.randint(0, len(np.load("/kaggle/input/tacotron2-inputs-outputs/phoneme_id_padded.npy")) - 1)
text = np.load("/kaggle/input/tacotron2-inputs-outputs/phoneme_id_padded.npy")[0]
text = torch.LongTensor(text).unsqueeze(0).to(device)
text_len = torch.LongTensor([text.size(1)]).to(device)

with torch.no_grad():
    mel_pred, mel_postnet, stop_pred, alignments = model(
        text, 
        text_len, 
        mel_targets=None, 
        teacher_forcing=False
    )
    mel = mel_postnet.squeeze(0).cpu().numpy()

sr = 22050
hop_length = 256
n_mels = 80
fmin = 125
fmax = 7600

mel_T = mel.T
n_frames = mel_T.shape[1]
time = np.arange(n_frames) * hop_length / sr
mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

plt.figure(figsize=(10, 4))
plt.imshow(mel_T, aspect="auto", origin="lower",
           extent=[time[0], time[-1], mel_freqs[0], mel_freqs[-1]], cmap="inferno")
plt.colorbar(label="Amplitude")
plt.title(f"Predicted Mel Spectrogram (idx={0})")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
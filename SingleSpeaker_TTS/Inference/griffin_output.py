import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
from IPython.display import Audio
from PreProcessing import mels_padded, mel
# from plot import mel

sr = 22050
n_fft = 1024
hoplength = 256
winlength = 1024
n_mels = 80
fmin = 125
fmax = 7600

def mel_to_audio(mel, n_iter=60):
    mel = librosa.db_to_power(mel.T)  
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    inv_mel = np.linalg.pinv(mel_basis)@mel
    audio = librosa.griffinlim(inv_mel, n_iter=n_iter, hop_length=hoplength, win_length=winlength)

    return audio

waveform = mel_to_audio(mel)
Audio(waveform, rate=sr)
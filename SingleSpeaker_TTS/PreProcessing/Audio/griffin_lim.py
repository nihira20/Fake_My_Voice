import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
from mels_padding import mels_padded

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

audio = mel_to_audio(mels_padded[0])
sf.write("/kaggle/working/audio_griffinlim.wav", audio, sr)

import IPython.display as ipd
ipd.Audio(audio, rate=sr)
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import torch
from Text import metadata


datapath ="your_dataset_path"

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 125
fmax = 7600

mel_list = []
lengths = []

for idx, row in metadata.iterrows():
    wavpath = os.path.join(datapath, "wavs", row["id"] + ".wav")
    y, _ = librosa.load(wavpath, sr=sr)

    #Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,             
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0) 
    mel = log_mel_spec.T.astype(np.float32)

    mel_list.append(mel)
    lengths.append(mel.shape[0])  
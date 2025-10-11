from mel import lengths
from mel import mel_list
import numpy as np

max_len = max(lengths)
print("Max mel length:", max_len)

#Pad all to same length
def pad_mel(mel, max_len):
    pad_width = max_len - mel.shape[0]
    if pad_width > 0:
        return np.pad(mel, ((0, pad_width), (0, 0)), mode="constant")
    else:
        return mel

mels_padded = np.array([pad_mel(m, max_len) for m in mel_list], dtype=np.float32)
print("Final padded mels shape:", mels_padded.shape)  
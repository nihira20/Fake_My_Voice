from mel_generation import preprocess_audio
from tqdm import tqdm
from metadata_creation import spk2idx, df
import numpy as np

#create chunks nd speaker indices
chunk_size = 160  #no of frames per training chunk
stride = 80 #overlap/steps

all_chunks = [] #small slices of audio in log mel form
chunk_speaker_indices = [] #corresponding speaker labels

for path in tqdm(df["wav_path"], desc="Processing files"):
    mel = preprocess_audio(path)
    T = mel.shape[0] #T is the no of time frames
    spk_idx = spk2idx[df.loc[df['wav_path']==path, 'speaker'].values[0]] #finding speaker index of this wav file

    #if the audio is shorter than the chunk then it is padded with zeros 
    if T < chunk_size:
        pad_width = chunk_size - T
        mel_chunk = np.pad(mel, ((0,pad_width),(0,0)), mode="constant")
        all_chunks.append(mel_chunk)
        chunk_speaker_indices.append(spk_idx)
    else:
        for start in range(0, T - chunk_size + 1, stride):
            all_chunks.append(mel[start:start+chunk_size])
            chunk_speaker_indices.append(spk_idx)
    #else slide across the audio with the window of sze chunk_size


all_chunks = np.array(all_chunks, dtype=np.float32)
chunk_speaker_indices = np.array(chunk_speaker_indices, dtype=np.int64)
print("Chunks:", all_chunks.shape, "Labels:", chunk_speaker_indices.shape)

#normalize chunks
mean = np.mean(all_chunks, axis=(0,1), keepdims=True)
std = np.std(all_chunks, axis=(0,1), keepdims=True)
all_chunks = (all_chunks - mean) / (std + 1e-9)

np.save("/kaggle/working/voxceleb_mels_chunks.npy", all_chunks)
np.save("/kaggle/working/voxceleb_chunk_speaker_indices.npy", chunk_speaker_indices)
print("Saved chunks and labels")

#wav files broken into 32k parts or chunks, each chunk is a 160*80 mel spec segment,
#each having their own label hence 32k labels
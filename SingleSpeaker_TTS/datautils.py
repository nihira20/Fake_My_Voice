import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ttsdataset(Dataset):
    def __init__(self, texts, mels, pad_id=0):
        self.texts = torch.from_numpy(texts).long()
        self.mels = torch.from_numpy(mels).float()
        self.pad_id = pad_id

        assert len(self.texts) == len(self.mels), "Text and Mel counts must match"
        self.text_lengths = (self.texts != pad_id).sum(dim=1)  #no of pad token
        self.mel_lengths = (self.mels.abs().sum(dim=2) != 0).sum(dim=1) 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        mel = self.mels[index]
        text_len = self.text_lengths[index]
        mel_len = self.mel_lengths[index]

        #stop token target: 0 for valid frames, 1 from last frame onward
        stop_target = torch.zeros(mel.size(0))
        stop_target[mel_len-1:] = 1.0

        return {
            "text": text,
            "mel": mel,
            "text_len": text_len,
            "mel_len": mel_len,
            "stop_target": stop_target
        }
    
texts = np.load("/kaggle/input/tacotron2-inputs-outputs/phoneme_id_padded.npy")
mels = np.load("/kaggle/input/tacotron2-inputs-outputs/mels_padded.npy")

batch_size = 32
dataset = ttsdataset(texts, mels, pad_id=0)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(len(dataset))
sample = dataset[0]
print("One sample keys:", sample.keys())

for batch in loader:
    print("Text batch shape:", batch["text"].shape)
    print("Mel batch shape:", batch["mel"].shape)
    print("Text lens shape:", batch["text_len"].shape)
    print("Mel lens shape:", batch["mel_len"].shape)
    print("Stop target shape:", batch["stop_target"].shape)
    break
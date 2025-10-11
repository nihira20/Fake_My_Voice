import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocessing import all_chunks, chunk_speaker_indices
# dataset and dataloader
class GE2EDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = torch.from_numpy(chunks).float()
        self.labels = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return self.chunks[idx], self.labels[idx]

dataset = GE2EDataset(all_chunks, chunk_speaker_indices)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
print("DataLoader ready. Total batches:", len(dataloader))
# each of those 32k chunks are loaded in the batch of 16, hence total 2040 batches
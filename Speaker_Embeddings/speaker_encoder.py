import torch.nn as nn
# specifying thr ge2e loss encoder 
class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels=80, hidden_size=256, n_layers=3, embedding_size=256):
        super().__init__()
        self.lstm = nn.LSTM(n_mels, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_size*2, embedding_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm(x)         
        out = out.mean(dim=1)         
        out = self.proj(out)
        out = self.relu(out)
        out = nn.functional.normalize(out, p=2, dim=1)
        return out
# result is a 256dim vector that represent that person's voice 
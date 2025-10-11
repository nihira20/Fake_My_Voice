import librosa
import numpy as np
import torch
import torch.nn as nn
# inference
class GE2EEncoder(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=256, embedding_dim=256, num_layers=3):
        super(GE2EEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.proj = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, mel):
        outputs, _ = self.lstm(mel) # mel dim is (1,T,n_mels)
        last = outputs[:, -1, :]  # take last frame
        embed = self.proj(last)
        embed = embed / torch.norm(embed, dim=1, keepdim=True)  #normalize
        return embed

# loading trained model/weightd
encoder = GE2EEncoder()
encoder.load_state_dict(torch.load("/kaggle/working/speaker_encoder_ge2e.pth", map_location="cpu"))
encoder.eval()
print("ge2e encoder loaded")

def preprocess_audio(path, sr=22050, n_fft=2048, hop_length=256, win_length=1024, 
                     n_mels=80, fmin=125, fmax=7600):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=25)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel = np.log(np.maximum(mel, 1e-6))
    mel = mel.T.astype(np.float32)  
    return mel

#takes one wav file, converts it to mel, runs through encoder and gives a 256dim speaker embedding vector
def get_speaker_embedding(wav_path, encoder):
    mel = preprocess_audio(wav_path)
    mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, T, n_mels)
    with torch.no_grad():
        embed = encoder(mel_tensor)
    return embed.squeeze(0).numpy()

wav_path = "/kaggle/input/voxceleb/vox1_test_wav/wav/id10270/5r0dWxy17C8/00001.wav"
embedding = get_speaker_embedding(wav_path, encoder)
print("Speaker embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])
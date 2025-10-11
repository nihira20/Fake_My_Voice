import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader   # for `loader` batches
from hparams_init import Tacotron2
from datautils import loader, texts

max_id = texts.max()  
vocab_size = int(max_id) + 1
print("Vocab size (number of phonemes):", vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Tacotron2(vocab_size=vocab_size).to(device)
checkpoint_path = "/kaggle/input/tacotron2model25epochs/tacotron2_modeltrain30ep.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model = Tacotron2(vocab_size=vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

learning_rate = 1e-3
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Tacotron2(vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
mel_loss_hist = []
stop_loss_hist = []
for epoch in range(num_epochs):
    model.train()
    epoch_mel_loss = 0.0
    epoch_stop_loss = 0.0

    for batch in loader:
        text = batch["text"].to(device)
        mel = batch["mel"].to(device)
        text_len = batch["text_len"].to(device)
        stop_target = batch["stop_target"].to(device)

        optimizer.zero_grad()

        #forw pass
        mel_pred, mel_postnet, gate_pred, alignments = model(text, text_len, mel_targets=mel, teacher_forcing=True)

        #losses
        mel_loss = mse_loss(mel_postnet, mel)
        stop_loss = bce_loss(gate_pred.squeeze(1), stop_target)
        total_loss = mel_loss + stop_loss

        #backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_mel_loss += mel_loss.item()
        epoch_stop_loss += stop_loss.item()
    mel_loss_hist.append(epoch_mel_loss / len(loader))
    stop_loss_hist.append(epoch_stop_loss / len(loader))
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Mel Loss: {epoch_mel_loss/len(loader):.4f} | "
          f"Stop Loss: {epoch_stop_loss/len(loader):.4f}")
    

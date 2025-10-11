from speaker_encoder import SpeakerEncoder
import torch
from preprocessing import n_mels
import torch.optim as optim
from GE2E_Loss import GE2ELoss, dataloader
# training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256
n_layers = 3
embedding_size = 256
encoder = SpeakerEncoder(n_mels=n_mels, hidden_size=hidden_size,
                         n_layers=n_layers, embedding_size=embedding_size).to(device)
ge2e_loss = GE2ELoss().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(ge2e_loss.parameters()), lr=1e-4)

num_epochs = 10
encoder.train()
for epoch in range(num_epochs):
    total_loss = 0
    for x, labels in dataloader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        emb = encoder(x)
        loss = ge2e_loss(emb, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(encoder.state_dict(), "/kaggle/working/speaker_encoder_ge2e.pth")
print("ge2e model saved.")
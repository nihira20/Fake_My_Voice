from train import mel_loss_hist, stop_loss_hist

import matplotlib.pyplot as plt

epochs = range(1, len(mel_loss_hist) + 1)

fig, ax1 = plt.subplots(figsize=(10,5))

color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mel Loss', color=color)
ax1.plot(epochs, mel_loss_hist, marker='o', color=color, label="Mel Loss")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Stop Loss', color=color)
ax2.plot(epochs, stop_loss_hist, marker='o', color=color, label="Stop Loss")
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Training Loss Curves (Mel Loss & Stop Token Loss)")
fig.tight_layout()
plt.show()
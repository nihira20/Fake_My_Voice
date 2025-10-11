import torch
import torch.nn as nn  
import torch.optim as optim
from model import Encoder
from model import Decoder
from model import Postnet
#complete tacotron2
class Tacotron2(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_mel_channels=80,
                 embed_dim=512,
                 enc_conv_channels=512,
                 enc_conv_kernel=5,
                 enc_conv_layers=3,
                 enc_lstm_dim=256,
                 prenet_dims=(256, 256),
                 attention_rnn_dim=1024,
                 decoder_rnn_dim=1024,
                 attention_dim=128,
                 attention_n_filters=32,
                 attention_kernel_size=31,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convolutions=5,
                 frames_per_step=1,
                 pad_id=0):
        super().__init__()

        self.n_mel_channels = n_mel_channels
        self.pad_id = pad_id

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            enc_conv_channels=enc_conv_channels,
            enc_conv_kernel=enc_conv_kernel,
            enc_conv_layers=enc_conv_layers,
            enc_lstm_dim=enc_lstm_dim,
            embed_padding_idx=pad_id,
        )

        self.decoder = Decoder(
            n_mel_channels=n_mel_channels,
            frames_per_step=frames_per_step,
            prenet_dims=prenet_dims,
            attention_rnn_dim=attention_rnn_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            encoder_out_dim=enc_lstm_dim*2,
            attention_dim=attention_dim,
            attention_n_filters=attention_n_filters,
            attention_kernel_size=attention_kernel_size,
        )
        self.postnet = Postnet(
            n_mel_channels=n_mel_channels,
            postnet_embedding_dim=postnet_embedding_dim,
            postnet_kernel_size=postnet_kernel_size,
            postnet_n_convolutions=postnet_n_convolutions,
        )

    def forward(self, text_ids, text_lengths, mel_targets=None, teacher_forcing=True):
        """
        text_ids: [B, T_text]
        text_lengths: [B]
        mel_targets (optional during training): [B, T_out, n_mel]
        returns:
          mel_out: [B, T_out, n_mel]
          mel_out_postnet:[B, T_out, n_mel]
          gate_out:[B, T_out]
          alignments:  [B, T_out, T_text]
        """
        memory = self.encoder(text_ids, text_lengths) 
        mel_out, gate_out, alignments = self.decoder(
            memory=memory,
            memory_lengths=text_lengths,
            mel_targets=mel_targets,
            teacher_forcing=teacher_forcing
        )
        residual = self.postnet(mel_out)#[B,T_out, n_mel]
        mel_out_postnet = mel_out + residual
        return mel_out, mel_out_postnet, gate_out, alignments
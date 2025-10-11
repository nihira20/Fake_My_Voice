import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import LinearNorm
from layers import get_mask_from_lengths


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=(256,256)):
        super(Prenet, self).__init__()
        in_sizes=[in_dim] + list(sizes[:-1])
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)]
        )

    def forward(self, x):
        for linear in self.layers:
            #dropout always applied
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
    

class Encoder(nn.Module):
    def __init__(self,vocab_size, embed_dim=512,enc_conv_channels=512, enc_conv_kernel=5,enc_conv_layers=3,enc_lstm_dim=256,embed_padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=embed_padding_idx)
        convs=[]
        in_ch=embed_dim
        for _ in range(enc_conv_layers):
            convs.append(nn.Sequential(
                nn.Conv1d(in_ch, enc_conv_channels, kernel_size=enc_conv_kernel, padding=enc_conv_kernel//2),
                nn.BatchNorm1d(enc_conv_channels),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))
            in_ch = enc_conv_channels
        self.convolutions= nn.ModuleList(convs)

        self.lstm = nn.LSTM(enc_conv_channels, enc_lstm_dim, num_layers=1, batch_first=True,bidirectional = True)

    def forward(self,text_ids,text_lengths):
        """
        text_ids: [B, T_text]
        text_lengths: [B]
        returns:
          memory: encoder outputs [B, T_text, 2*enc_lstm_dim]
        """
        x=self.embedding(text_ids)
        x=x.transpose(1,2)
        for conv in self.convolutions:
            x=conv(x)
        x=x.transpose(1,2)

        lengths_cpu = text_lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #outputs are [B, T_text, 2*enc_lstm_dim]
        return outputs
    

#Location sensitive attention
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters=32, attention_kernel_size=31, attention_dim=128):
        super().__init__()
        self.location_conv = nn.Conv1d(2, attention_n_filters, kernel_size=attention_kernel_size,
                                       padding=(attention_kernel_size-1)//2, bias=False)
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attn_weights_cat):
        """
        attn_weights_cat: [B, 2, T_text]  (prev + cumulative attention)
        returns: [B, T_text, attention_dim]
        """
        processed = self.location_conv(attn_weights_cat)
        processed = processed.transpose(1, 2)
        processed = self.location_dense(processed)
        return processed


class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim, encoder_out_dim, attention_dim=128,
                 attention_n_filters=32, attention_kernel_size=31):
        super().__init__()
        self.query_layer=nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer= nn.Linear(encoder_out_dim, attention_dim, bias=False)
        self.location_layer =LocationLayer(attention_n_filters, attention_kernel_size, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=True)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attn_weights_cat):
        """
        query: [B, attention_rnn_dim] (decoder state)
        processed_memory: [B, T_text, A] (memory transformed)
        attn_weights_cat: [B, 2, T_text] (prev + cumulative)
        returns energies: [B, T_text]
        """
        processed_query =self.query_layer(query).unsqueeze(1)   
        processed_location= self.location_layer(attn_weights_cat) 
        energies=self.v(torch.tanh(processed_query + processed_location + processed_memory))  
        energies=energies.squeeze(-1)                         
        return energies

    def forward(self, query, memory, processed_memory, attn_weights, attn_weights_cum, memory_lengths):
        """
        query: [B, attention_rnn_dim]
        memory: [B, T_text, encoder_out_dim]
        processed_memory: [B, T_text, A]
        attn_weights: [B, T_text] previous step attention
        attn_weights_cum: [B, T_text] cumulative attention
        memory_lengths: [B]
        returns:
          context: [B, encoder_out_dim]
          attn_weights: [B, T_text] new alignment
        """
        #concat prev and cumulative weights for location features
        attn_weights_cat = torch.stack([attn_weights, attn_weights_cum], dim=1)  

        energies = self.get_alignment_energies(query, processed_memory, attn_weights_cat) 
        if memory_lengths is not None:
            mask = get_mask_from_lengths(memory_lengths, max_len=memory.size(1))
            energies = energies.masked_fill(mask, self.score_mask_value)

        attn_weights = F.softmax(energies, dim=1)  #[B, T_text]
        attn_weights_cum = attn_weights_cum + attn_weights

        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)  #[B, encoder_out_dim]
        return context, attn_weights, attn_weights_cum
    
#decoder
class Decoder(nn.Module):
    def __init__(self,
                 n_mel_channels=80,
                 frames_per_step=1,
                 prenet_dims=(256, 256),
                 attention_rnn_dim=1024,
                 decoder_rnn_dim=1024,
                 encoder_out_dim=512,#2 times enc_lstm_dim (e.g., 2 * 256 = 512)
                 attention_dim=128,
                 attention_n_filters=32,
                 attention_kernel_size=31,
                 p_decoder_dropout=0.1):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.frames_per_step = frames_per_step

        #prenet on last predicted frame
        self.prenet = Prenet(n_mel_channels * frames_per_step, prenet_dims)
        #RNNS
        
        self.attention_rnn = nn.LSTMCell(prenet_dims[-1] + encoder_out_dim, attention_rnn_dim)
        self.decoder_rnn= nn.LSTMCell(attention_rnn_dim + encoder_out_dim, decoder_rnn_dim)
        #attention
        self.attention_layer=LocationSensitiveAttention(
            attention_rnn_dim=attention_rnn_dim,
            encoder_out_dim=encoder_out_dim,
            attention_dim=attention_dim,
            attention_n_filters=attention_n_filters,
            attention_kernel_size=attention_kernel_size
        )

        self.linear_projection = nn.Linear(decoder_rnn_dim + encoder_out_dim,n_mel_channels * frames_per_step)
        self.gate_projection   = nn.Linear(decoder_rnn_dim + encoder_out_dim, 1)

        self.dropout = p_decoder_dropout

    def initialize_decoder_states(self, memory):
        B=memory.size(0)
        device = memory.device

        #RNN STATES
        self.attention_hidden = torch.zeros(B, self.attention_rnn.hidden_size, device=device)
        self.attention_cell = torch.zeros(B, self.attention_rnn.hidden_size, device=device)
        self.decoder_hidden  = torch.zeros(B, self.decoder_rnn.hidden_size, device=device)
        self.decoder_cell=torch.zeros(B, self.decoder_rnn.hidden_size, device=device)

        #attention weights
        T_text = memory.size(1)
        self.attention_weights = torch.zeros(B, T_text, device=device)
        self.attention_weights_cum = torch.zeros(B, T_text, device=device)
        self.attention_context= torch.zeros(B,memory.size(2), device=device)

    def parse_decoder_inputs(self, mel_targets):
        """
        mel_targets: [B, T_out, n_mel]
        returns: inputs to decoder time steps as [B, T_out, n_mel*frames_per_step]
        """
        return mel_targets

    def decode(self, prenet_out, processed_memory, memory, memory_lengths):
        cell_input = torch.cat([prenet_out, self.attention_context], dim=1)  # [B, prenet_out + ctx]
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, p=self.dropout, training=self.training)

        context, self.attention_weights, self.attention_weights_cum = self.attention_layer(
            self.attention_hidden, memory, processed_memory, self.attention_weights, self.attention_weights_cum, memory_lengths
        )
        self.attention_context = context

        decoder_input = torch.cat([self.attention_hidden, context], dim=1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, p=self.dropout, training=self.training)

        proj_input= torch.cat([self.decoder_hidden, context],dim=1)
        mel_out = self.linear_projection(proj_input) #[B, n_mel*fps]
        gate_out = self.gate_projection(proj_input)
        return mel_out, gate_out, self.attention_weights

    def forward(self, memory, memory_lengths, mel_targets=None, teacher_forcing=True,gate_threshold=0.5,max_decoder_steps=2000):
        """
        memory: encoder outputs [B, T_text, E]
        memory_lengths: [B]
        mel_targets: [B, T_out, n_mel] (required for training)
        returns:
          mel_outputs: [B, T_out, n_mel]
          gate_outputs: [B, T_out]
          alignments:  [B, T_out, T_text]
        """
        self.initialize_decoder_states(memory)
        B = memory.size(0)
        device = memory.device
        #precompute processed memory for attention
        processed_memory = self.attention_layer.memory_layer(memory) 
        go_frame = torch.zeros(B, self.n_mel_channels * self.frames_per_step, device=device)
        mel_outputs, gate_outputs, alignments=[],[],[]
        #teacher forcing(last o/p fed as next input)
        if teacher_forcing:
            assert mel_targets is not None, "mel_targets must be provided during teacher forcing."
            decoder_inputs = self.parse_decoder_inputs(mel_targets)  # [B, T_out, n_mel]
            T_out = decoder_inputs.size(1)

            prev_output = go_frame
            for t in range(T_out):
                prenet_out = self.prenet(prev_output.unsqueeze(1)).squeeze(1)  # [B, prenet_dim]
                mel_out, gate_out, attn = self.decode(prenet_out, processed_memory, memory, memory_lengths)
                mel_outputs.append(mel_out.unsqueeze(1))  # [B, 1, n_mel]
                gate_outputs.append(gate_out.unsqueeze(1))# [B, 1, 1] -> will squeeze later
                alignments.append(attn.unsqueeze(1))

                prev_output = decoder_inputs[:, t, :]
        else:
            prev_output = go_frame
            for _ in range(max_decoder_steps):
                prenet_out =self.prenet(prev_output.unsqueeze(1)).squeeze(1)
                mel_out, gate_out, attn=self.decode(prenet_out, processed_memory, memory, memory_lengths)
                mel_outputs.append(mel_out.unsqueeze(1))
                gate_outputs.append(gate_out.unsqueeze(1))
                alignments.append(attn.unsqueeze(1))        

                prev_output = mel_out
                if torch.sigmoid(gate_out).item()>gate_threshold:
                    break

        mel_outputs = torch.cat(mel_outputs, dim=1)#[B, T_out, n_mel]
        gate_outputs = torch.cat(gate_outputs, dim=1).squeeze(-1)#[B, T_out]
        alignments = torch.cat(alignments, dim=1) 
        return mel_outputs, gate_outputs, alignments
    

#postnet
class Postnet(nn.Module):
    def __init__(self, n_mel_channels=80,postnet_embedding_dim=512, postnet_kernel_size=5, postnet_n_convolutions=5):
        super().__init__()
        layers=[]
        #first layer
        layers.append(nn.Sequential(
            nn.Conv1d(n_mel_channels, postnet_embedding_dim, kernel_size=postnet_kernel_size,
                      padding=postnet_kernel_size // 2),
            nn.BatchNorm1d(postnet_embedding_dim),
            nn.Tanh(),
            nn.Dropout(0.5)
        ))
        #middle layer
        for _ in range(1,postnet_n_convolutions-1):
            layers.append(nn.Sequential(
                nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size=postnet_kernel_size,
                          padding=postnet_kernel_size // 2),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            ))
        #last layer
        layers.append(nn.Sequential(
            nn.Conv1d(postnet_embedding_dim, n_mel_channels, kernel_size=postnet_kernel_size,
                      padding=postnet_kernel_size // 2),
            nn.BatchNorm1d(n_mel_channels),
            nn.Dropout(0.5)
        ))
        self.convolutions=nn.ModuleList(layers)

    def forward(self, mel_outputs):
        """
        mel_outputs: [B, T_out, n_mel]
        returns residual: [B, T_out, n_mel]
        """
        x=mel_outputs.transpose(1,2)
        for layer in self.convolutions:
            x=layer(x)
        x=x.transpose(1,2)
        return x
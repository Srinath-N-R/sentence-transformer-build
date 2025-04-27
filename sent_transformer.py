import torch
import math
from torch import nn
from enums import PAD_ID, MAX_LEN

class SentenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, proj_dim=32, dropout=0.1):
        super().__init__()
        self.padding_idx = PAD_ID
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_idx)

        pe = torch.zeros(MAX_LEN, d_model) # fixed positional embedding
        pos = torch.arange(0, MAX_LEN).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.proj_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)

    def forward(self, x, attention_mask=None, return_all=False):
        if attention_mask is None:
            attention_mask = x.eq(PAD_ID)
        else:
            attention_mask = ~attention_mask.bool()

        pe = self.pe[:x.size(1)].to(x.device)
        h = self.embed(x) + pe
        h = self.encoder(h, src_key_padding_mask=attention_mask)

        
        cls = h[:, 0] # CLS pooling (position 0 token, CLS_ID)

        sent_emb = self.proj_norm(self.proj(self.proj_dropout(cls))) # projection head for sentence tasks

        if return_all:
            return sent_emb, h
        return sent_emb

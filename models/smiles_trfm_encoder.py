from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .attn_transformer import AttnTransformerBlock, AttnTransformerEncoder


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class SmilesTrfmEncoder(nn.Module):
    """
    forward(token_ids, need_attn) -> (x, attn)
    fingerprint(token_ids) -> (B,4D)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pe = SinusoidalPositionalEncoding(self.d_model, dropout=dropout, max_len=self.max_seq_len)

        enc_block = AttnTransformerBlock(
            d_model=self.d_model,
            n_heads=int(n_heads),
            dropout=float(dropout),
            dim_feedforward=4 * self.d_model,
            activation="gelu",
        )
        self.encoder = AttnTransformerEncoder(enc_block, num_layers=self.n_layers)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, token_ids: torch.Tensor, need_attn: bool = False):
        if token_ids.dim() != 2:
            raise ValueError(f"SmilesTrfmEncoder expects (B,L), got {token_ids.shape}")
        B, L = token_ids.shape
        if L > self.max_seq_len:
            raise ValueError(f"L={L} exceeds max_seq_len={self.max_seq_len}")

        x = self.tok_emb(token_ids)
        x = self.pe(x)
        pad_mask = (token_ids == self.pad_id)

        x, attn = self.encoder(x, key_padding_mask=pad_mask, need_weights=need_attn, return_all_layers=False)
        x = self.norm(x)
        return x, attn

    @torch.no_grad()
    def fingerprint(self, token_ids: torch.Tensor):
        h_last, _ = self.forward(token_ids, need_attn=False)  # (B,L,D)
        mask = (token_ids != self.pad_id).unsqueeze(-1)       # (B,L,1)

        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (h_last * mask).sum(dim=1) / denom

        h_masked = h_last.masked_fill(~mask.bool(), float("-inf"))
        maxv = torch.max(h_masked, dim=1).values
        maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))

        cls_last = h_last[:, 0, :]
        cls_penul = cls_last  # 간단 버전(원하면 penultimate도 복원 가능)

        return torch.cat([mean, maxv, cls_last, cls_penul], dim=-1)

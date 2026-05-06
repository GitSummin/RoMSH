from __future__ import annotations

import copy
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str):
    name = (name or "gelu").lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unknown activation: {name}")


class AttnTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with per-head attention weights support.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(self.dropout)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, int(dim_feedforward)),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(dim_feedforward), self.d_model),
        )
        self.drop2 = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,                        # (B,L,D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B,L) True for PAD
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # --- self-attn (pre-norm) ---
        h = self.norm1(x)
        attn_out, attn_w = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,  # -> (B,H,L,S)
        )
        x = x + self.drop1(attn_out)

        # --- FFN (pre-norm) ---
        h2 = self.norm2(x)
        x = x + self.drop2(self.ff(h2))

        return x, attn_w


class AttnTransformerEncoder(nn.Module):
    """
    Stack of AttnTransformerBlock.
    forward returns:
      - x: (B,L,D)
      - attn: (B,H,L,L) from last layer if need_weights else None
    """
    def __init__(self, layer: AttnTransformerBlock, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(int(num_layers))])

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        return_all_layers: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, List[torch.Tensor]]]]:
        attn_last = None
        attn_all: List[torch.Tensor] = []

        for blk in self.layers:
            x, a = blk(x, key_padding_mask=key_padding_mask, need_weights=need_weights)
            if need_weights and a is not None:
                attn_last = a
                if return_all_layers:
                    attn_all.append(a)

        if need_weights:
            if return_all_layers:
                return x, attn_all
            return x, attn_last
        return x, None

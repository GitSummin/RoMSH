# models/mlm_model.py
from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .smiles_trfm_encoder import SmilesTrfmEncoder


class SmilesMLMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_id: int = 0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.encoder = SmilesTrfmEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
        )

        # ✅ weight tying: decoder.weight == tok_emb.weight
        self.lm_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = self.encoder.tok_emb.weight

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=float(label_smoothing))

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        h, _ = self.encoder(input_ids, need_attn=False)  # (B,L,D)
        logits = self.decoder(self.lm_norm(h))           # (B,L,V)

        out: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            B, L, V = logits.shape
            loss = self.loss_fn(logits.view(B * L, V), labels.view(B * L))
            out["loss"] = loss
        return out

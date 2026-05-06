# models/triplet_interaction.py
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _neg_large(x: torch.Tensor) -> float:
    # fp16에서도 안전한 마스킹 값
    return -1e4 if x.dtype in (torch.float16, torch.bfloat16) else -1e9


class TripletInteractionLayer(nn.Module):
    """
    Methods 1.7: Higher-order (triplet) interaction modelling.

    score_{i,j,k} ≈ (q_i · k_j) * (q_i · v_k) / sqrt(d_head)
    alpha_{i,j,k} = softmax_{(j,k)} score
    u'_i = sum_{j,k} alpha_{i,j,k} * Wo(u_k)

    - N(max_mols)가 작을 때(O(N^3)) 실용적. (여기선 보통 5)
    - multi-head 지원.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_heads({n_heads})")
        self.d_head = self.d_model // self.n_heads

        self.norm = nn.LayerNorm(self.d_model)

        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.Wo = nn.Linear(self.d_model, self.d_model)

        self.drop = nn.Dropout(self.dropout)

    def forward(
        self,
        u: torch.Tensor,              # (B,N,D)
        keep: torch.Tensor,           # (B,N) bool
        need_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = u.shape
        h = self.norm(u)

        q = self.Wq(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,N,d)
        k = self.Wk(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,N,d)
        v = self.Wv(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,N,d)

        # (B,H,N,N): s1_{i,j} = q_i · k_j
        s1 = torch.einsum("bhin,bhjn->bhij", q, k)
        # (B,H,N,N): s2_{i,k} = q_i · v_k
        s2 = torch.einsum("bhin,bhkn->bhik", q, v)

        # score (B,H,N,N,N): score_{i,j,k}
        score = (s1.unsqueeze(-1) * s2.unsqueeze(-2)) / math.sqrt(float(self.d_head))

        # mask invalid j,k (PAD mols)
        keep_b = keep.to(torch.bool)  # (B,N)
        valid_j = keep_b[:, None, None, :, None]      # (B,1,1,N,1)
        valid_k = keep_b[:, None, None, None, :]      # (B,1,1,1,N)
        valid = valid_j & valid_k
        score = score.masked_fill(~valid, _neg_large(score))

        # softmax over (j,k)
        score_flat = score.view(B, self.n_heads, N, N * N)
        alpha_flat = torch.softmax(score_flat, dim=-1)
        alpha = alpha_flat.view(B, self.n_heads, N, N, N)  # (B,H,i,j,k)

        # u_k projection (B,H,N,d)
        ok = self.Wo(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # sum over j first -> weights for k only (B,H,N,N)
        w_k = alpha.sum(dim=-2)  # (B,H,i,k)

        out_h = torch.einsum("bhik,bhkn->bhin", w_k, ok)  # (B,H,N,d)
        out = out_h.transpose(1, 2).contiguous().view(B, N, D)  # (B,N,D)

        # residual
        u2 = u + self.drop(out)

        if need_attn:
            # 반환: head-mean attention over k (B,N,N)  (i->k)
            attn_ik = w_k.mean(dim=1)  # (B,N,N)
            return u2, attn_ik
        return u2, None


class TripletInteractionStack(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TripletInteractionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(int(n_layers))
        ])

    def forward(self, u: torch.Tensor, keep: torch.Tensor, need_attn: bool = False):
        attn_last = None
        for layer in self.layers:
            u, attn_last = layer(u, keep, need_attn=need_attn)
        return u, attn_last
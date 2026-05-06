from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import math

import torch
import torch.nn as nn
from torch.autograd import Function

from .attn_transformer import AttnTransformerBlock, AttnTransformerEncoder
from .smiles_trfm_encoder import SmilesTrfmEncoder
from .triplet_interaction import TripletInteractionStack


def neg_inf_like(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)


def masked_mean(x: torch.Tensor, mask_keep: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask_keep.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=dim).clamp(min=1.0)
    return (x * m).sum(dim=dim) / denom


def masked_max(x: torch.Tensor, mask_keep: torch.Tensor, dim: int) -> torch.Tensor:
    x2 = x.masked_fill(~mask_keep.unsqueeze(-1), neg_inf_like(x))
    out = x2.max(dim=dim).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def lengths_from_padded(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    b, l = token_ids.shape
    is_pad = token_ids == pad_id
    idx = is_pad.float().argmax(dim=1)
    no_pad = (~is_pad).all(dim=1)
    idx = torch.where(no_pad, torch.full_like(idx, l), idx)
    return idx


class GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverseFn.apply(x, float(lambd))


class TokenTransformerFPPlus(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        pad_id: int = 0,
        attn_pool: bool = True,
        exclude_cls_from_pool: bool = True,
        alpha_init: float = 0.0,
        pool_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pad_id = int(pad_id)
        self.attn_pool = bool(attn_pool)
        self.exclude_cls_from_pool = bool(exclude_cls_from_pool)

        self.enc = SmilesTrfmEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
        )

        self.fp_proj = nn.Linear(4 * d_model, d_model)
        self.q = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.q, std=0.02)
        self.pool_proj = nn.Linear(d_model, d_model)
        self.pool_drop = nn.Dropout(float(pool_dropout))
        self.pool_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.out_norm = nn.LayerNorm(d_model)

    def _pool_keep(self, token_ids: torch.Tensor) -> torch.Tensor:
        keep = token_ids != self.pad_id
        if self.exclude_cls_from_pool and keep.size(1) > 0:
            keep[:, 0] = False
        return keep

    def _fingerprint(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        keep = self._pool_keep(token_ids)
        cls = h[:, 0, :]
        mean = masked_mean(h, keep, dim=1)
        mx = masked_max(h, keep, dim=1)
        lens = lengths_from_padded(token_ids, pad_id=self.pad_id).clamp(min=1)
        last_idx = (lens - 1).to(torch.long)
        b = torch.arange(h.size(0), device=h.device)
        last = h[b, last_idx, :]
        return torch.cat([cls, mean, mx, last], dim=-1)

    def _attn_pool(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        keep = self._pool_keep(token_ids)
        scale = 1.0 / math.sqrt(float(self.d_model))
        scores = (h * self.q.view(1, 1, -1)).sum(-1) * scale
        scores = scores.masked_fill(~keep, neg_inf_like(scores))
        w = torch.softmax(scores, dim=1)
        return (h * w.unsqueeze(-1)).sum(1)

    def forward(self, token_ids: torch.Tensor, need_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, attn = self.enc(token_ids, need_attn=bool(need_attn))
        fp = self._fingerprint(h, token_ids)
        mol_fp = self.fp_proj(fp)

        if not self.attn_pool:
            return mol_fp, (attn if need_attn else None)

        pooled = self.pool_proj(self._attn_pool(h, token_ids))
        pooled = self.pool_drop(pooled)
        a = torch.tanh(self.pool_alpha)
        mol = self.out_norm(mol_fp + a * pooled)
        return mol, (attn if need_attn else None)


class FourierRTFeatures(nn.Module):
    def __init__(self, K: int = 8, rt_norm: str = "minmax"):
        super().__init__()
        self.K = int(K)
        self.rt_norm = str(rt_norm).lower()
        self.register_buffer("freq", (2.0 ** torch.arange(self.K)).view(1, 1, self.K), persistent=False)

    def _normalize(self, rt: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        if self.rt_norm == "none":
            return rt

        out = rt.clone()
        bsz, _ = out.shape
        for b in range(bsz):
            m = keep[b]
            if m.sum().item() < 1:
                continue
            v = out[b, m]
            if self.rt_norm == "zscore":
                mu = v.mean()
                sd = v.std(unbiased=False).clamp(min=1e-6)
                out[b, m] = (v - mu) / sd
            else:
                vmin = v.min()
                vmax = v.max()
                denom = (vmax - vmin).clamp(min=1e-6)
                out[b, m] = (v - vmin) / denom
        out = out.masked_fill(~keep, 0.0)
        return out

    def forward(self, rt: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        rt = rt.to(dtype=torch.float32)
        rt = self._normalize(rt, keep)
        x = math.pi * rt.unsqueeze(-1) * self.freq.to(rt.device)
        feat = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        feat = feat * keep.unsqueeze(-1).to(feat.dtype)
        return feat


class MixtureTransformerCLS(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_mols: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        use_pos_emb: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_mols = int(max_mols)
        self.use_pos_emb = bool(use_pos_emb)

        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pos_emb = nn.Embedding(self.max_mols + 1, self.d_model)

        blk = AttnTransformerBlock(
            d_model=self.d_model,
            n_heads=int(n_heads),
            dropout=float(dropout),
            dim_feedforward=4 * self.d_model,
            activation="gelu",
        )
        self.enc = AttnTransformerEncoder(blk, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, u: torch.Tensor, keep: torch.Tensor, need_attn: bool = False):
        b, n, d = u.shape
        cls = self.cls.expand(b, 1, d)
        x = torch.cat([cls, u], dim=1)

        if self.use_pos_emb:
            pos = torch.arange(1 + n, device=u.device).unsqueeze(0).expand(b, 1 + n)
            x = x + self.pos_emb(pos)

        pad_mask = torch.cat([torch.zeros((b, 1), device=u.device, dtype=torch.bool), ~keep], dim=1)
        x, attn = self.enc(x, key_padding_mask=pad_mask, need_weights=bool(need_attn), return_all_layers=False)
        x = self.norm(x)
        return x[:, 0, :], x[:, 1:, :], attn


@dataclass
class ChemSeqConfig:
    vocab_size: int
    max_mols: int
    max_seq_len: int

    d_model: int = 192
    token_layers: int = 4
    token_heads: int = 4
    token_dropout: float = 0.10

    mix_layers: int = 1
    mix_heads: int = 4
    mix_dropout: float = 0.15

    rt_fourier_K: int = 8
    rt_norm: str = "minmax"
    use_pos_emb: bool = True
    use_rt_features: bool = True
    use_rt_sort: bool = True

    d_sub: Optional[int] = None
    head_hidden: int = 128
    head_dropout: float = 0.30

    eps_abundance: float = 1e-8
    ab_log_eps: float = 1e-4

    triplet_layers: int = 0
    triplet_heads: int = 4
    triplet_dropout: float = 0.1

    num_envs: int = 0
    grl_lambda: float = 1.0

    use_hybrid_stats: bool = True
    stat_dropout: float = 0.10

    causal_topk: int = 1
    causal_min_ab: float = 0.0

    use_meta_fusion: bool = True
    meta_input_dim: int = 5
    env_emb_dim: int = 16
    meta_dropout: float = 0.10


class ChemSeqModel(nn.Module):
    def __init__(self, cfg: ChemSeqConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = int(pad_id)

        d_sub = int(cfg.d_sub) if cfg.d_sub is not None else (cfg.d_model // 2)
        self.d_sub = d_sub

        self.token_enc = TokenTransformerFPPlus(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.token_layers,
            n_heads=cfg.token_heads,
            dropout=cfg.token_dropout,
            max_seq_len=cfg.max_seq_len,
            pad_id=self.pad_id,
            attn_pool=True,
            exclude_cls_from_pool=True,
            alpha_init=0.0,
            pool_dropout=0.1,
        )

        self.rt_feat = FourierRTFeatures(K=cfg.rt_fourier_K, rt_norm=cfg.rt_norm)
        self.Wm = nn.Linear(cfg.d_model, cfg.d_model)
        self.Wr = nn.Linear(2 * cfg.rt_fourier_K, cfg.d_model)
        self.Wa = nn.Linear(1, cfg.d_model)
        self.comp_norm = nn.LayerNorm(cfg.d_model)
        self.imp_ab_scale = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

        self.mix_enc = MixtureTransformerCLS(
            d_model=cfg.d_model,
            max_mols=cfg.max_mols,
            n_layers=cfg.mix_layers,
            n_heads=cfg.mix_heads,
            dropout=cfg.mix_dropout,
            use_pos_emb=cfg.use_pos_emb,
        )

        self.imp_scorer = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        self.stat_proj = nn.Sequential(
            nn.LayerNorm(4 * cfg.d_model),
            nn.Linear(4 * cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.stat_dropout)),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.fuse_norm = nn.LayerNorm(cfg.d_model)

        self.triplet = None
        if int(cfg.triplet_layers) > 0:
            self.triplet = TripletInteractionStack(
                d_model=cfg.d_model,
                n_layers=int(cfg.triplet_layers),
                n_heads=int(cfg.triplet_heads),
                dropout=float(cfg.triplet_dropout),
            )

        self.use_meta_fusion = bool(cfg.use_meta_fusion)
        self.env_emb = None
        meta_in_dim = int(cfg.meta_input_dim)
        if int(cfg.num_envs) > 1:
            self.env_emb = nn.Embedding(int(cfg.num_envs), int(cfg.env_emb_dim))
            meta_in_dim += int(cfg.env_emb_dim)

        self.meta_proj = None
        self.meta_gate = None
        self.meta_norm = None
        self.meta_event_proj = None
        self.meta_logit_bias = None
        if self.use_meta_fusion:
            self.meta_proj = nn.Sequential(
                nn.LayerNorm(meta_in_dim),
                nn.Linear(meta_in_dim, cfg.d_model),
                nn.GELU(),
                nn.Dropout(float(cfg.meta_dropout)),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.meta_gate = nn.Sequential(
                nn.Linear(2 * cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.meta_norm = nn.LayerNorm(cfg.d_model)
            self.meta_event_proj = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, max(16, d_sub // 2)),
                nn.GELU(),
            )
            self.meta_logit_bias = nn.Sequential(
                nn.LayerNorm(meta_in_dim),
                nn.Linear(meta_in_dim, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.meta_dropout)),
                nn.Linear(cfg.head_hidden, 1),
            )
        self.meta_in_dim = meta_in_dim

        self.proj_chem = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, d_sub),
            nn.GELU(),
        )
        self.proj_ctx = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, d_sub),
            nn.GELU(),
        )

        self.fuel_head = nn.Sequential(
            nn.LayerNorm(d_sub),
            nn.Linear(d_sub, 1),
        )

        if int(cfg.num_envs) > 1:
            self.context_head = nn.Sequential(
                nn.LayerNorm(d_sub),
                nn.Linear(d_sub, d_sub),
                nn.GELU(),
                nn.Dropout(float(cfg.head_dropout)),
                nn.Linear(d_sub, int(cfg.num_envs)),
            )
            self.adv_context_head = nn.Sequential(
                nn.LayerNorm(d_sub),
                nn.Linear(d_sub, d_sub),
                nn.GELU(),
                nn.Dropout(float(cfg.head_dropout)),
                nn.Linear(d_sub, int(cfg.num_envs)),
            )
        else:
            self.context_head = None
            self.adv_context_head = None

        meta_evt_dim = max(16, d_sub // 2) if self.use_meta_fusion else 0
        event_in_dim = 4 * d_sub + 1 + meta_evt_dim
        self.head = nn.Sequential(
            nn.LayerNorm(event_in_dim),
            nn.Linear(event_in_dim, int(cfg.head_hidden)),
            nn.GELU(),
            nn.Dropout(float(cfg.head_dropout)),
            nn.Linear(int(cfg.head_hidden), 1),
        )


    def _align_event_feat_dim(self, event_feat: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible alignment between actual event feature dimension
        and the input dimension expected by self.head.
        - If current dim < expected dim: zero-pad.
        - If current dim == expected dim: return as-is.
        - If current dim > expected dim: raise explicit error.
        """
        target_dim = None

        if isinstance(self.head, nn.Sequential) and len(self.head) > 0:
            first = self.head[0]
            if isinstance(first, nn.LayerNorm):
                ns = first.normalized_shape
                if isinstance(ns, (tuple, list)):
                    target_dim = int(ns[0])
                else:
                    target_dim = int(ns)

        if target_dim is None:
            return event_feat

        cur_dim = int(event_feat.size(-1))

        if cur_dim == target_dim:
            return event_feat

        if cur_dim < target_dim:
            pad = torch.zeros(
                event_feat.size(0),
                target_dim - cur_dim,
                device=event_feat.device,
                dtype=event_feat.dtype,
            )
            return torch.cat([event_feat, pad], dim=-1)

        raise RuntimeError(
            f"event_feat dim ({cur_dim}) is larger than head expected dim ({target_dim}). "
            "Please re-check head construction and concatenated feature branches."
        )

    def encode_molecules(self, x: torch.Tensor, need_token_attn: bool = False) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        b, n, l = x.shape
        flat = x.reshape(b * n, l)
        keep = flat[:, 0] != self.pad_id
        idx = torch.where(keep)[0]

        token_infos: List[Optional[torch.Tensor]] = [None] * (b * n)
        mol = torch.zeros((b * n, self.cfg.d_model), device=x.device, dtype=torch.float32)

        if idx.numel() > 0:
            m, attn = self.token_enc(flat.index_select(0, idx), need_attn=bool(need_token_attn))
            mol = mol.index_copy(0, idx, m.float())
            if need_token_attn and attn is not None:
                idx_list = idx.tolist()
                for j, i in enumerate(idx_list):
                    token_infos[i] = attn[j].detach()

        return mol.reshape(b, n, self.cfg.d_model), token_infos

    def _has_signal(self, v: torch.Tensor, keep: torch.Tensor) -> bool:
        return bool(((v.abs() > 0).to(torch.bool) & keep).any().item())

    def _normalize_ab(self, ab: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        ab = ab.clamp_min(0.0) * keep.to(ab.dtype)
        s = ab.sum(dim=1, keepdim=True).clamp_min(float(self.cfg.eps_abundance))
        return ab / s

    def _build_remove_mask_by_abundance(self, ab_norm: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, n = ab_norm.shape
        k = max(0, int(self.cfg.causal_topk))
        if k <= 0:
            return torch.zeros_like(keep, dtype=torch.bool)
        ab_masked = ab_norm.masked_fill(~keep, -1.0)
        kk = min(k, n)
        topk_idx = torch.topk(ab_masked, k=kk, dim=1, largest=True).indices
        remove = torch.zeros_like(keep, dtype=torch.bool)
        for i in range(b):
            for j in topk_idx[i].tolist():
                if j >= 0 and ab_masked[i, j].item() >= float(self.cfg.causal_min_ab):
                    remove[i, j] = True
        return remove

    def _build_meta_inputs(self, meta_feat: Optional[torch.Tensor], env_id: Optional[torch.Tensor], device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_meta_fusion or meta_feat is None:
            return None, None
        meta_feat = meta_feat.to(device=device, dtype=torch.float32)
        meta_in = meta_feat
        if self.env_emb is not None and env_id is not None:
            env_idx = env_id.to(device=device, dtype=torch.long).clamp(min=0)
            env_vec = self.env_emb(env_idx)
            meta_in = torch.cat([meta_in, env_vec], dim=-1)
        z_meta = self.meta_proj(meta_in)
        return meta_in, z_meta

    def _forward_once(
        self,
        x: torch.Tensor,
        rt: Optional[torch.Tensor],
        ab: Optional[torch.Tensor],
        meta_feat: Optional[torch.Tensor],
        env_id: Optional[torch.Tensor],
        need_explain: bool,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        b, n, _ = x.shape
        mol_keep = x[:, :, 0] != self.pad_id

        mol_emb, token_infos = self.encode_molecules(x, need_token_attn=bool(need_explain))
        mol_emb = mol_emb * mol_keep.unsqueeze(-1).to(mol_emb.dtype)
        u = self.Wm(mol_emb)

        if rt is None:
            rt = torch.zeros((b, n), device=x.device, dtype=torch.float32)
        else:
            rt = rt.to(device=x.device, dtype=torch.float32)

        has_rt_signal = self._has_signal(rt, mol_keep)
        if bool(self.cfg.use_rt_features) and has_rt_signal:
            rt_gamma = self.rt_feat(rt, mol_keep)
            u = u + self.Wr(rt_gamma)

        if ab is None:
            ab = torch.zeros((b, n), device=x.device, dtype=torch.float32)
        else:
            ab = ab.to(device=x.device, dtype=torch.float32)

        has_ab_signal = self._has_signal(ab, mol_keep)
        ab_norm = None
        if has_ab_signal:
            ab_norm = self._normalize_ab(ab, mol_keep)
            ab_feat = torch.log(ab_norm.clamp_min(float(self.cfg.ab_log_eps))).unsqueeze(-1)
            ab_add = self.Wa(ab_feat) * mol_keep.unsqueeze(-1).to(torch.float32)
            u = u + ab_add

        u = self.comp_norm(u) * mol_keep.unsqueeze(-1).to(torch.float32)

        mol_perm = None
        if bool(self.cfg.use_rt_sort) and has_rt_signal:
            rt_sort = rt.masked_fill(~mol_keep, float("inf"))
            mol_perm = torch.argsort(rt_sort, dim=1)
            d = u.size(-1)
            u = u.gather(1, mol_perm.unsqueeze(-1).expand(-1, -1, d))
            mol_keep = mol_keep.gather(1, mol_perm)
            if has_ab_signal and ab_norm is not None:
                ab_norm = ab_norm.gather(1, mol_perm)

        triplet_attn = None
        if self.triplet is not None:
            u, triplet_attn = self.triplet(u, mol_keep, need_attn=bool(need_explain))

        z_cls, h_mols, mix_attn = self.mix_enc(u, mol_keep, need_attn=bool(need_explain))

        imp_logits = self.imp_scorer(h_mols).squeeze(-1)
        if ab_norm is not None:
            imp_logits = imp_logits + self.imp_ab_scale * torch.log(ab_norm.clamp_min(float(self.cfg.ab_log_eps)))
        imp_logits = imp_logits.masked_fill(~mol_keep, -1e9)
        imp_w = torch.softmax(imp_logits, dim=1)
        z_imp = (h_mols * imp_w.unsqueeze(-1)).sum(dim=1)
        z_mean = masked_mean(h_mols, mol_keep, dim=1)
        z_max = masked_max(h_mols, mol_keep, dim=1)

        if bool(self.cfg.use_hybrid_stats):
            stat_cat = torch.cat([z_cls, z_imp, z_mean, z_max], dim=-1)
            z_stat = self.stat_proj(stat_cat)
            gamma = torch.sigmoid(self.fuse_gate(torch.cat([z_cls, z_stat], dim=-1)))
            z_mix = self.fuse_norm(z_cls + gamma * z_stat)
        else:
            z_stat = z_imp
            z_mix = z_cls

        meta_in, z_meta = self._build_meta_inputs(meta_feat=meta_feat, env_id=env_id, device=x.device)
        if z_meta is not None:
            meta_gate = torch.sigmoid(self.meta_gate(torch.cat([z_mix, z_meta], dim=-1)))
            z_mix = self.meta_norm(z_mix + meta_gate * z_meta)
            meta_evt = self.meta_event_proj(z_meta)
        else:
            meta_evt = None

        z_chem = self.proj_chem(z_mix)
        z_ctx = self.proj_ctx(z_mix)
        fuel_logits = self.fuel_head(z_chem).squeeze(-1)

        context_logits = None
        adv_context_logits = None
        if self.context_head is not None:
            context_logits = self.context_head(z_ctx)
        if self.adv_context_head is not None:
            adv_context_logits = self.adv_context_head(grad_reverse(z_chem, self.cfg.grl_lambda))

        fuel_prob = torch.sigmoid(fuel_logits).unsqueeze(-1)
        event_feat = torch.cat(
            [z_chem, z_ctx, z_chem * z_ctx, torch.abs(z_chem - z_ctx), fuel_prob],
            dim=-1,
        )
        event_feat = self._align_event_feat_dim(event_feat)
        logits = self.head(event_feat).squeeze(-1)

        out["logits"] = logits
        out["fuel_logits"] = fuel_logits
        out["context_logits"] = context_logits
        out["adv_context_logits"] = adv_context_logits
        out["z_mix"] = z_mix
        out["z_chem"] = z_chem
        out["z_context"] = z_ctx
        out["z_meta"] = z_meta
        out["imp_w"] = imp_w.detach()

        if need_explain:
            out["mix_attn"] = mix_attn
            out["token_infos"] = token_infos
            out["triplet_attn"] = triplet_attn
            if mol_perm is not None:
                out["mol_perm"] = mol_perm

        out["_mol_keep"] = mol_keep
        out["_ab_norm"] = ab_norm
        return out

    def forward(
        self,
        x: torch.Tensor,
        rt: Optional[torch.Tensor] = None,
        ab: Optional[torch.Tensor] = None,
        meta_feat: Optional[torch.Tensor] = None,
        env_id: Optional[torch.Tensor] = None,
        need_explain: bool = False,
        need_causal: bool = False,
    ) -> Dict[str, Any]:
        out = self._forward_once(
            x=x,
            rt=rt,
            ab=ab,
            meta_feat=meta_feat,
            env_id=env_id,
            need_explain=bool(need_explain),
        )

        if need_causal:
            mol_keep = out["_mol_keep"]
            ab_norm = out["_ab_norm"]
            if ab_norm is None:
                out["logits_masked"] = out["logits"].detach()
            else:
                remove = self._build_remove_mask_by_abundance(ab_norm, mol_keep)
                x_masked = x.clone()
                x_masked[remove] = self.pad_id

                rt_masked = None
                ab_masked = None
                if rt is not None:
                    rt_masked = rt.clone()
                    rt_masked[remove] = 0.0
                if ab is not None:
                    ab_masked = ab.clone()
                    ab_masked[remove] = 0.0

                out2 = self._forward_once(
                    x=x_masked,
                    rt=rt_masked,
                    ab=ab_masked,
                    meta_feat=meta_feat,
                    env_id=env_id,
                    need_explain=False,
                )
                out["logits_masked"] = out2["logits"]

        out.pop("_mol_keep", None)
        out.pop("_ab_norm", None)
        return out

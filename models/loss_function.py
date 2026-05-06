from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    pos_weight: float = 1.0

    event_loss: str = "bce"
    event_focal_gamma: float = 1.5
    event_label_smoothing: float = 0.02
    logit_l2_weight: float = 1e-4
    mixsize_weight_alpha: float = 0.0

    use_fuel_aux: bool = False
    fuel_loss_weight: float = 0.20
    fuel_pos_weight: float = 1.0
    fuel_pred_is_logits: bool = True

    use_contrastive: bool = True
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.07
    contrastive_hard_neg_k: int = 8
    contrastive_min_pos: int = 1

    use_context_aux: bool = True
    context_loss_weight: float = 0.20

    use_context_adv: bool = True
    context_adv_weight: float = 0.10

    use_irm: bool = False
    irm_weight: float = 0.1
    irm_reduce: str = "mean"

    use_causal: bool = False
    causal_weight: float = 0.1
    causal_use_logits: bool = False

    neg_penalty_weight: float = 0.08
    neg_penalty_power: float = 2.0

    soft_tversky_weight: float = 0.05
    soft_tversky_alpha: float = 0.40
    soft_tversky_beta: float = 0.60


def _supervised_hardneg_contrastive(
    z: torch.Tensor,
    y: torch.Tensor,
    tau: float,
    hard_neg_k: int,
    min_pos: int,
) -> Tuple[torch.Tensor, int]:
    if z.ndim != 2:
        raise ValueError(f"z must be (B,D), got {tuple(z.shape)}")
    b = z.size(0)
    if b < 2:
        return z.new_tensor(0.0), 0

    z = F.normalize(z, p=2, dim=-1)
    sim = (z @ z.t()) / max(float(tau), 1e-8)
    y_int = y.detach().long()

    losses = []
    used = 0
    for i in range(b):
        pos_idx = torch.where(y_int == y_int[i])[0]
        pos_idx = pos_idx[pos_idx != i]
        if pos_idx.numel() < min_pos:
            continue

        neg_idx = torch.where(y_int != y_int[i])[0]
        if neg_idx.numel() == 0:
            continue

        pos_score = sim[i, pos_idx].max()
        neg_scores = sim[i, neg_idx]
        if hard_neg_k > 0 and neg_scores.numel() > hard_neg_k:
            neg_scores = torch.topk(neg_scores, k=hard_neg_k, largest=True).values

        denom = torch.logsumexp(torch.cat([pos_score.view(1), neg_scores], dim=0), dim=0)
        losses.append(-(pos_score - denom))
        used += 1

    if used == 0:
        return z.new_tensor(0.0), 0
    return torch.stack(losses).mean(), used


def _irm_penalty(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    env_id: torch.Tensor,
    pos_weight_t: torch.Tensor,
    reduce: str = "mean",
) -> Tuple[torch.Tensor, int]:
    if logits.ndim != 1:
        raise ValueError(f"logits must be (B,), got {tuple(logits.shape)}")
    if env_id.ndim != 1:
        raise ValueError(f"env_id must be (B,), got {tuple(env_id.shape)}")

    scale = logits.new_tensor(1.0, requires_grad=True)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    penalty = logits.new_tensor(0.0)
    used_envs = 0
    uniq_envs = torch.unique(env_id.detach())
    for e in uniq_envs:
        idx = env_id == e
        if idx.sum().item() < 2:
            continue
        loss_e = bce(scale * logits[idx], y_true[idx])
        grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
        penalty = penalty + (grad ** 2)
        used_envs += 1

    if used_envs > 0 and str(reduce).lower() == "mean":
        penalty = penalty / float(used_envs)
    return penalty, used_envs


def _event_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight_t: torch.Tensor,
    mode: str,
    focal_gamma: float,
    label_smoothing: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if label_smoothing > 0:
        eps = float(label_smoothing)
        y = y_true * (1.0 - eps) + 0.5 * eps
    else:
        y = y_true

    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t, reduction="none")
    mode = str(mode).lower()
    if mode == "focal":
        p = torch.sigmoid(logits)
        pt = p * y + (1.0 - p) * (1.0 - y)
        loss_vec = ((1.0 - pt).clamp_min(1e-6).pow(float(focal_gamma))) * bce
    else:
        loss_vec = bce

    if sample_weight is not None:
        w = sample_weight.to(loss_vec.device, dtype=loss_vec.dtype).clamp_min(1e-8)
        return (loss_vec * w).sum() / w.sum()
    return loss_vec.mean()


def _soft_tversky_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    y = y_true.to(dtype=p.dtype)
    tp = (p * y).sum()
    fp = (p * (1.0 - y)).sum()
    fn = ((1.0 - p) * y).sum()
    score = (tp + 1e-6) / (tp + float(alpha) * fp + float(beta) * fn + 1e-6)
    return 1.0 - score


class ChemSeqLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("pos_weight_t", torch.tensor([float(cfg.pos_weight)], dtype=torch.float32), persistent=False)
        self.register_buffer("fuel_pos_weight_t", torch.tensor([float(cfg.fuel_pos_weight)], dtype=torch.float32), persistent=False)
        self.bce_fuel = nn.BCEWithLogitsLoss(pos_weight=self.fuel_pos_weight_t)
        self.ce_context = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        fuel_pred: Optional[torch.Tensor] = None,
        fuel_proxy: Optional[torch.Tensor] = None,
        fuel_mask: Optional[torch.Tensor] = None,
        z_chem: Optional[torch.Tensor] = None,
        env_id: Optional[torch.Tensor] = None,
        logits_masked: Optional[torch.Tensor] = None,
        mix_size: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        context_logits: Optional[torch.Tensor] = None,
        adv_context_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, float] = {
            "loss_total": 0.0,
            "loss_event": 0.0,
            "loss_logit_l2": 0.0,
            "loss_fuel": 0.0,
            "loss_contrast": 0.0,
            "loss_context": 0.0,
            "loss_context_adv": 0.0,
            "loss_inv": 0.0,
            "loss_causal": 0.0,
            "loss_neg": 0.0,
            "loss_tversky": 0.0,
            "contrast_used_anchors": 0.0,
            "irm_used_envs": 0.0,
            "fuel_used": 0.0,
            "context_used": 0.0,
        }

        w = sample_weight
        if w is None and self.cfg.mixsize_weight_alpha > 0 and mix_size is not None:
            k = mix_size.to(dtype=torch.float32).clamp_min(1.0)
            denom = (k.max().detach() - 1.0).clamp_min(1.0)
            w = 1.0 + float(self.cfg.mixsize_weight_alpha) * (k - 1.0) / denom

        loss_event = _event_loss(
            logits=logits,
            y_true=y_true,
            pos_weight_t=self.pos_weight_t,
            mode=self.cfg.event_loss,
            focal_gamma=float(self.cfg.event_focal_gamma),
            label_smoothing=float(self.cfg.event_label_smoothing),
            sample_weight=w,
        )
        total = loss_event
        losses["loss_event"] = float(loss_event.detach().cpu().item())

        if float(self.cfg.soft_tversky_weight) > 0:
            loss_tversky = _soft_tversky_loss(
                logits=logits,
                y_true=y_true,
                alpha=float(self.cfg.soft_tversky_alpha),
                beta=float(self.cfg.soft_tversky_beta),
            )
            total = total + float(self.cfg.soft_tversky_weight) * loss_tversky
            losses["loss_tversky"] = float(loss_tversky.detach().cpu().item())

        if float(self.cfg.logit_l2_weight) > 0:
            loss_l2 = (logits ** 2).mean()
            total = total + float(self.cfg.logit_l2_weight) * loss_l2
            losses["loss_logit_l2"] = float(loss_l2.detach().cpu().item())

        if float(self.cfg.neg_penalty_weight) > 0:
            neg_mask = (y_true < 0.5).to(logits.dtype)
            if neg_mask.sum().item() > 0:
                p = torch.sigmoid(logits)
                neg_pen = ((p.pow(float(self.cfg.neg_penalty_power))) * neg_mask).sum() / neg_mask.sum().clamp_min(1.0)
                total = total + float(self.cfg.neg_penalty_weight) * neg_pen
                losses["loss_neg"] = float(neg_pen.detach().cpu().item())

        if self.cfg.use_fuel_aux:
            if fuel_pred is None or fuel_proxy is None:
                raise ValueError("use_fuel_aux=True requires fuel_pred and fuel_proxy")
            if fuel_mask is None:
                fuel_mask = fuel_proxy >= 0
            fuel_mask = fuel_mask.to(torch.bool)
            used = int(fuel_mask.sum().item())
            losses["fuel_used"] = float(used)
            if used > 0:
                fp = fuel_pred[fuel_mask]
                fx = fuel_proxy[fuel_mask].float().clamp(0.0, 1.0)
                if self.cfg.fuel_pred_is_logits:
                    loss_fuel = self.bce_fuel(fp, fx)
                else:
                    loss_fuel = F.binary_cross_entropy(fp, fx)
                total = total + float(self.cfg.fuel_loss_weight) * loss_fuel
                losses["loss_fuel"] = float(loss_fuel.detach().cpu().item())

        if self.cfg.use_contrastive:
            if z_chem is None:
                raise ValueError("use_contrastive=True requires z_chem")
            loss_contrast, used = _supervised_hardneg_contrastive(
                z=z_chem,
                y=y_true,
                tau=float(self.cfg.contrastive_temperature),
                hard_neg_k=int(self.cfg.contrastive_hard_neg_k),
                min_pos=int(self.cfg.contrastive_min_pos),
            )
            total = total + float(self.cfg.contrastive_weight) * loss_contrast
            losses["loss_contrast"] = float(loss_contrast.detach().cpu().item())
            losses["contrast_used_anchors"] = float(used)

        if (self.cfg.use_context_aux or self.cfg.use_context_adv or self.cfg.use_irm) and env_id is None:
            raise ValueError("context/IRM losses require env_id")

        valid_env = None
        if env_id is not None:
            valid_env = env_id >= 0
            losses["context_used"] = float(valid_env.sum().item())

        if self.cfg.use_context_aux and context_logits is not None and valid_env is not None and valid_env.any():
            loss_ctx = self.ce_context(context_logits[valid_env], env_id[valid_env].long())
            total = total + float(self.cfg.context_loss_weight) * loss_ctx
            losses["loss_context"] = float(loss_ctx.detach().cpu().item())

        if self.cfg.use_context_adv and adv_context_logits is not None and valid_env is not None and valid_env.any():
            loss_ctx_adv = self.ce_context(adv_context_logits[valid_env], env_id[valid_env].long())
            total = total + float(self.cfg.context_adv_weight) * loss_ctx_adv
            losses["loss_context_adv"] = float(loss_ctx_adv.detach().cpu().item())

        if self.cfg.use_irm:
            irm_pen, used_envs = _irm_penalty(
                logits=logits,
                y_true=y_true,
                env_id=env_id.long(),
                pos_weight_t=self.pos_weight_t,
                reduce=str(self.cfg.irm_reduce),
            )
            total = total + float(self.cfg.irm_weight) * irm_pen
            losses["loss_inv"] = float(irm_pen.detach().cpu().item())
            losses["irm_used_envs"] = float(used_envs)

        if self.cfg.use_causal:
            if logits_masked is None:
                raise ValueError("use_causal=True requires logits_masked")
            if self.cfg.causal_use_logits:
                diff = logits - logits_masked
            else:
                diff = torch.sigmoid(logits) - torch.sigmoid(logits_masked)
            loss_causal = (diff ** 2).mean()
            total = total + float(self.cfg.causal_weight) * loss_causal
            losses["loss_causal"] = float(loss_causal.detach().cpu().item())

        losses["loss_total"] = float(total.detach().cpu().item())
        return total, losses

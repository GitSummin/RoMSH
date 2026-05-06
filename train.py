from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.smiles_tokenizer import build_vocab, load_vocab_json
from data.mixture_dataset import MixtureCSVDataset
from models.chemseq_model import ChemSeqModel, ChemSeqConfig
from models.loss_function import ChemSeqLoss, LossConfig
from metrics import sigmoid_np, compute_binary_metrics, tune_threshold


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", s)
    return s or "run"


def append_summary_row(csv_path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(row.keys())
    exists = os.path.exists(csv_path)
    if exists:
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header:
                fieldnames = list(dict.fromkeys(existing_header + fieldnames))
        except Exception:
            pass

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def extract_smiles_from_csv(csv_path: str, col: str = "mixture") -> List[str]:
    df = pd.read_csv(csv_path, usecols=[col])
    out: List[str] = []
    for mix in df[col].astype(str):
        mix = mix.strip()
        if not mix:
            continue
        out.extend(mix.split())
    return out


def compute_pos_weight_from_csv(train_csv: str) -> float:
    df = pd.read_csv(train_csv, usecols=["label"])
    y = df["label"].astype(int).values
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / max(1, pos))


def choose_context_column(df: pd.DataFrame) -> str:
    for c in ["env_name", "category", "env_id", "env_source"]:
        if c in df.columns:
            return c
    return ""


def parse_float_sequence(s: Any) -> List[float]:
    if s is None:
        return []
    txt = str(s).strip()
    if not txt:
        return []
    vals = []
    for t in txt.split():
        try:
            vals.append(float(t))
        except Exception:
            pass
    return vals


def entropy_norm_from_abundance_str(s: Any) -> float:
    vals = np.array(parse_float_sequence(s), dtype=np.float32)
    if vals.size <= 1:
        return 0.0
    vals = np.clip(vals, 1e-8, None)
    vals = vals / vals.sum()
    ent = float(-(vals * np.log(vals)).sum())
    return float(ent / math.log(len(vals))) if len(vals) > 1 else 0.0


def dominant_from_abundance_str(s: Any) -> float:
    vals = parse_float_sequence(s)
    if not vals:
        return 0.0
    vals = np.array(vals, dtype=np.float32)
    vals = np.clip(vals, 1e-8, None)
    vals = vals / vals.sum()
    return float(vals.max())


def rt_span_from_rts_str(s: Any) -> float:
    vals = parse_float_sequence(s)
    if len(vals) <= 1:
        return 0.0
    return float(max(vals) - min(vals))


def ensure_meta_columns(df: pd.DataFrame, fuel_col: str, rt_col: str, abundance_col: str) -> pd.DataFrame:
    df = df.copy()
    if "gid" not in df.columns:
        df["gid"] = [str(i) for i in range(len(df))]

    if fuel_col not in df.columns:
        df[fuel_col] = 1.0
    df[fuel_col] = pd.to_numeric(df[fuel_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    if "mix_size" not in df.columns:
        df["mix_size"] = df["mixture"].astype(str).str.split().str.len().fillna(1).astype(int)
    if "ab_entropy_norm" not in df.columns:
        df["ab_entropy_norm"] = df[abundance_col].apply(entropy_norm_from_abundance_str)
    if "dominant_abundance" not in df.columns:
        df["dominant_abundance"] = df[abundance_col].apply(dominant_from_abundance_str)
    if "rt_span" not in df.columns:
        df["rt_span"] = df[rt_col].apply(rt_span_from_rts_str)
    if "env_name" not in df.columns:
        ctx_col = choose_context_column(df)
        if ctx_col:
            df["env_name"] = df[ctx_col].astype(str)
        else:
            df["env_name"] = "NA"

    for c in ["mix_size", "ab_entropy_norm", "dominant_abundance", "rt_span"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def build_env_map(csv_paths: List[str]) -> Dict[str, int]:
    vals = []
    for p in csv_paths:
        df = ensure_meta_columns(pd.read_csv(p), fuel_col="fuel_proxy", rt_col="rts", abundance_col="abundance")
        vals.extend(df["env_name"].astype(str).fillna("NA").tolist())
    uniq = sorted(set(str(v) for v in vals))
    return {v: i for i, v in enumerate(uniq)}


def build_meta_lookup(csv_path: str, fuel_col: str, rt_col: str, abundance_col: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    df = ensure_meta_columns(df, fuel_col=fuel_col, rt_col=rt_col, abundance_col=abundance_col)
    out: Dict[str, Dict[str, float]] = {}
    for i, r in df.iterrows():
        gid = str(r.get("gid", i))
        out[gid] = {
            "fuel_proxy": float(r[fuel_col]),
            "mix_size": float(r["mix_size"]),
            "ab_entropy_norm": float(r["ab_entropy_norm"]),
            "dominant_abundance": float(r["dominant_abundance"]),
            "rt_span": float(r["rt_span"]),
            "env_name": str(r["env_name"]),
        }
        out[str(i)] = out[gid]
    return out


def compute_train_meta_stats(meta_lookup: Dict[str, Dict[str, float]], max_mols: int) -> Dict[str, float]:
    vals = defaultdict(list)
    for v in meta_lookup.values():
        if not isinstance(v, dict):
            continue
        vals["fuel_proxy"].append(float(v.get("fuel_proxy", 1.0)))
        vals["mix_size"].append(float(v.get("mix_size", 1.0)))
        vals["ab_entropy_norm"].append(float(v.get("ab_entropy_norm", 0.0)))
        vals["dominant_abundance"].append(float(v.get("dominant_abundance", 1.0)))
        vals["rt_span"].append(float(v.get("rt_span", 0.0)))
    rt_max = float(np.percentile(vals["rt_span"], 95)) if vals["rt_span"] else 1.0
    return {
        "fuel_proxy_mean": float(np.mean(vals["fuel_proxy"])) if vals["fuel_proxy"] else 1.0,
        "mix_size_max": float(max(max_mols, int(max(vals["mix_size"])) if vals["mix_size"] else max_mols)),
        "rt_span_p95": max(1e-6, rt_max),
    }


def build_meta_tensor(gids: List[Any], meta_lookup: Dict[str, Dict[str, float]], stats: Dict[str, float], env_map: Dict[str, int], max_mols: int, device: torch.device):
    meta_rows = []
    env_ids = []
    hardness = []
    for g in gids:
        rec = meta_lookup.get(str(g), None)
        if rec is None:
            rec = {
                "fuel_proxy": 1.0,
                "mix_size": 1.0,
                "ab_entropy_norm": 0.0,
                "dominant_abundance": 1.0,
                "rt_span": 0.0,
                "env_name": "NA",
            }
        fuel_proxy = float(rec.get("fuel_proxy", 1.0))
        mix_size = float(rec.get("mix_size", 1.0))
        ab_entropy_norm = float(rec.get("ab_entropy_norm", 0.0))
        dominant_abundance = float(rec.get("dominant_abundance", 1.0))
        rt_span = float(rec.get("rt_span", 0.0))
        env_name = str(rec.get("env_name", "NA"))

        mix_norm = min(max((mix_size - 1.0) / max(1.0, float(max_mols - 1)), 0.0), 1.0)
        rt_span_norm = min(max(rt_span / max(1e-6, float(stats["rt_span_p95"])), 0.0), 1.5)
        rt_span_norm = min(rt_span_norm, 1.0)
        boundary = 1.0 - abs(2.0 * fuel_proxy - 1.0)
        hard = 0.30 * mix_norm + 0.30 * ab_entropy_norm + 0.25 * boundary + 0.15 * rt_span_norm

        meta_rows.append([fuel_proxy, mix_norm, ab_entropy_norm, dominant_abundance, rt_span_norm])
        env_ids.append(int(env_map.get(env_name, -1)))
        hardness.append(float(hard))

    meta_feat = torch.tensor(meta_rows, dtype=torch.float32, device=device)
    env_id = torch.tensor(env_ids, dtype=torch.long, device=device)
    hard_t = torch.tensor(hardness, dtype=torch.float32, device=device)
    return meta_feat, env_id, hard_t


def make_curriculum_weights(y: torch.Tensor, hardness: torch.Tensor, fuel_proxy: torch.Tensor, epoch: int, curriculum_epochs: int, difficulty_alpha: float, pos_alpha: float, boundary_alpha: float) -> torch.Tensor:
    progress = min(1.0, float(epoch) / max(1.0, float(curriculum_epochs)))
    boundary = 1.0 - torch.abs(2.0 * fuel_proxy - 1.0)
    weights = 1.0 + progress * float(difficulty_alpha) * hardness
    weights = weights + progress * float(pos_alpha) * hardness * y
    weights = weights + progress * float(boundary_alpha) * boundary * (1.0 - y)
    return weights.clamp_min(1.0)


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device, temperature: float, meta_lookup: Dict[str, Dict[str, float]], meta_stats: Dict[str, float], env_map: Dict[str, int], max_mols: int):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        x, y, gid, cat, k, rt, ab = batch
        x = x.to(device)
        y = y.to(device).float()
        rt = rt.to(device)
        ab = ab.to(device)
        meta_feat, env_id, _ = build_meta_tensor(gid, meta_lookup, meta_stats, env_map, max_mols, device)
        out = model(x, rt=rt, ab=ab, meta_feat=meta_feat, env_id=env_id, need_explain=False)
        logits = out["logits"]
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
    logits_all = np.concatenate(logits_all).astype(np.float32)
    y_all = np.concatenate(y_all).astype(np.int64)
    prob = sigmoid_np(logits_all / max(float(temperature), 1e-6))
    return logits_all, prob, y_all


def fit_temperature_from_logits(logits_np: np.ndarray, y_np: np.ndarray, max_iter: int = 60) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.tensor(logits_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np.astype(np.float32), dtype=torch.float32, device=device)
    log_t = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=int(max_iter), line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp(1e-3, 100.0)
        loss = bce(logits / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return max(1e-3, float(torch.exp(log_t).detach().cpu().item()))


def renorm_ab_after_mask(ab: torch.Tensor, x: torch.Tensor, pad_id: int, eps: float = 1e-8) -> torch.Tensor:
    keep = (x[:, :, 0] != pad_id).to(ab.dtype)
    ab2 = ab.clamp_min(0.0) * keep
    s = ab2.sum(dim=1, keepdim=True).clamp_min(eps)
    return ab2 / s


def maybe_apply_moldrop(x: torch.Tensor, rt: torch.Tensor, ab: torch.Tensor, pad_id: int, mol_drop_p: float, min_keep: int):
    if float(mol_drop_p) <= 0.0:
        return x, rt, ab

    keep_m = x[:, :, 0] != pad_id
    drop = (torch.rand_like(rt) < float(mol_drop_p)) & keep_m
    min_keep = int(min_keep)
    if min_keep > 0:
        bsz, _ = drop.shape
        kept_after = (keep_m & (~drop)).sum(dim=1)
        for b in range(bsz):
            need = int(min_keep - int(kept_after[b].item()))
            if need <= 0:
                continue
            candidates = torch.where(drop[b])[0]
            if candidates.numel() == 0:
                continue
            if ab[b].abs().sum().item() > 0:
                cand_ab = ab[b, candidates]
                pick = candidates[torch.topk(cand_ab, k=min(need, candidates.numel()), largest=True).indices]
            else:
                pick = candidates[:min(need, candidates.numel())]
            drop[b, pick] = False

    if drop.any():
        x = x.clone()
        rt = rt.clone()
        ab = ab.clone()
        x[drop] = pad_id
        rt[drop] = 0.0
        ab[drop] = 0.0
        ab = renorm_ab_after_mask(ab, x, pad_id)
    return x, rt, ab


def maybe_load_pretrained_encoder(model: ChemSeqModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_sd = ckpt.get("encoder_state_dict", None) or ckpt.get("state_dict", None) or ckpt
    missing, unexpected = model.token_enc.enc.load_state_dict(enc_sd, strict=False)
    print(f"[MLM init] loaded: {ckpt_path}")
    if missing:
        print(f"  missing={len(missing)}")
    if unexpected:
        print(f"  unexpected={len(unexpected)}")


def configure_token_encoder_ft(model: ChemSeqModel, encoder_frozen: bool, ft_top_n_token_layers: int):
    for p in model.parameters():
        p.requires_grad = True

    if bool(encoder_frozen):
        for p in model.token_enc.enc.parameters():
            p.requires_grad = False
        return

    n = int(ft_top_n_token_layers)
    if n <= 0:
        return

    for p in model.token_enc.enc.parameters():
        p.requires_grad = False

    layers = model.token_enc.enc.encoder.layers
    n = min(n, len(layers))
    for blk in layers[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.token_enc.enc.norm.parameters():
        p.requires_grad = True


def save_predictions_csv(path: str, y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= float(thr)).astype(np.int64)
    df = pd.DataFrame({"y_true": y_true.astype(int), "y_prob": y_prob.astype(float), "y_pred": y_pred.astype(int)})
    df.to_csv(path, index=False)


def build_summary_row(args, out_dir: str, best_epoch: int, temperature: float, thr_final: float, val_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Dict[str, Any]:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ablation_group": args.ablation_group,
        "experiment_name": args.experiment_name,
        "dataset_tag": args.dataset_tag,
        "split_name": args.split_name,
        "run_dir": out_dir,
        "best_epoch": int(best_epoch),
        "temperature": float(temperature),
        "thr_final": float(thr_final),
        "stop_metric": args.stop_metric,
        "thr_metric": args.tune_thr_metric,
        "use_hybrid_stats": int(bool(args.use_hybrid_stats)),
        "use_fuel_aux": int(bool(args.use_fuel_aux)),
        "use_contrastive": int(bool(args.use_contrastive)),
        "use_context_aux": int(bool(args.use_context_aux)),
        "use_context_adv": int(bool(args.use_context_adv)),
        "use_causal": int(bool(args.use_causal)),
        "use_meta_fusion": int(bool(args.use_meta_fusion)),
    }
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--dataset_tag", type=str, default="Scaffold")
    p.add_argument("--split_name", type=str, default="C3_material_group")

    p.add_argument("--rt_col", type=str, default="rts")
    p.add_argument("--abundance_col", type=str, default="abundance")
    p.add_argument("--fuel_col", type=str, default="fuel_proxy")

    p.add_argument("--out_dir", type=str, default="results_chemseq")
    p.add_argument("--summary_csv", type=str, default="")
    p.add_argument("--ablation_group", type=str, default="proposed_upgrade")
    p.add_argument("--experiment_name", type=str, default="Proposed_MaterialGroup_V2")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--pretrained_vocab", type=str, default="")
    p.add_argument("--pretrained_mlm_encoder", type=str, default="")
    p.add_argument("--max_tokens", type=int, default=1200)
    p.add_argument("--encoder_frozen", action="store_true")
    p.add_argument("--ft_top_n_token_layers", type=int, default=0)

    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--max_mols", type=int, default=5)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--token_layers", type=int, default=4)
    p.add_argument("--token_heads", type=int, default=4)
    p.add_argument("--token_dropout", type=float, default=0.10)
    p.add_argument("--mix_layers", type=int, default=2)
    p.add_argument("--mix_heads", type=int, default=4)
    p.add_argument("--mix_dropout", type=float, default=0.15)
    p.add_argument("--rt_fourier_K", type=int, default=8)
    p.add_argument("--rt_norm", type=str, default="minmax", choices=["none", "minmax", "zscore"])
    p.add_argument("--use_pos_emb", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_rt_features", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_rt_sort", type=int, default=1, choices=[0, 1])
    p.add_argument("--head_hidden", type=int, default=160)
    p.add_argument("--head_dropout", type=float, default=0.25)
    p.add_argument("--triplet_layers", type=int, default=0)
    p.add_argument("--triplet_heads", type=int, default=4)
    p.add_argument("--triplet_dropout", type=float, default=0.10)
    p.add_argument("--grl_lambda", type=float, default=1.0)
    p.add_argument("--use_hybrid_stats", dest="use_hybrid_stats", action="store_true")
    p.add_argument("--no_use_hybrid_stats", dest="use_hybrid_stats", action="store_false")
    p.set_defaults(use_hybrid_stats=True)
    p.add_argument("--use_meta_fusion", dest="use_meta_fusion", action="store_true")
    p.add_argument("--no_use_meta_fusion", dest="use_meta_fusion", action="store_false")
    p.set_defaults(use_meta_fusion=True)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop", type=int, default=14)
    p.add_argument("--stop_metric", type=str, default="jaccard", choices=["bal_acc", "f1", "acc", "jaccard", "prec", "rec", "precision", "recall"])

    p.add_argument("--token_mask_p_train", type=float, default=0.08)
    p.add_argument("--token_mask_p_eval", type=float, default=0.0)
    p.add_argument("--mol_drop_p", type=float, default=0.08)
    p.add_argument("--mol_drop_min_keep", type=int, default=2)

    p.add_argument("--thr_mode", type=str, default="tune_on_val", choices=["fixed", "tune_on_val"])
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--tune_thr_metric", type=str, default="jaccard", choices=["bal_acc", "f1", "acc", "jaccard", "prec", "rec", "precision", "recall"])
    p.add_argument("--thr_strategy", type=str, default="quantile", choices=["linspace", "quantile"])
    p.add_argument("--thr_grid", type=int, default=199)
    p.add_argument("--temp_scale_on_val", action="store_true")
    p.add_argument("--temp_scale_max_iter", type=int, default=60)

    p.add_argument("--event_loss", type=str, default="focal", choices=["bce", "focal"])
    p.add_argument("--event_focal_gamma", type=float, default=1.5)
    p.add_argument("--event_label_smoothing", type=float, default=0.01)
    p.add_argument("--logit_l2_weight", type=float, default=1e-4)
    p.add_argument("--mixsize_weight_alpha", type=float, default=0.0)
    p.add_argument("--neg_penalty_weight", type=float, default=0.06)
    p.add_argument("--neg_penalty_power", type=float, default=2.0)
    p.add_argument("--soft_tversky_weight", type=float, default=0.05)
    p.add_argument("--soft_tversky_alpha", type=float, default=0.40)
    p.add_argument("--soft_tversky_beta", type=float, default=0.60)

    p.add_argument("--use_fuel_aux", action="store_true")
    p.add_argument("--fuel_loss_weight", type=float, default=0.15)
    p.add_argument("--fuel_pos_weight", type=float, default=1.0)

    p.add_argument("--use_contrastive", action="store_true")
    p.add_argument("--contrastive_weight", type=float, default=0.05)
    p.add_argument("--contrastive_temperature", type=float, default=0.07)
    p.add_argument("--contrastive_hard_neg_k", type=int, default=8)
    p.add_argument("--contrastive_min_pos", type=int, default=1)

    p.add_argument("--use_context_aux", action="store_true")
    p.add_argument("--context_loss_weight", type=float, default=0.15)
    p.add_argument("--use_context_adv", action="store_true")
    p.add_argument("--context_adv_weight", type=float, default=0.08)

    p.add_argument("--use_causal", action="store_true")
    p.add_argument("--causal_weight", type=float, default=0.03)
    p.add_argument("--causal_use_logits", action="store_true")
    p.add_argument("--causal_topk", type=int, default=1)
    p.add_argument("--causal_min_ab", type=float, default=0.0)

    p.add_argument("--curriculum_epochs", type=int, default=18)
    p.add_argument("--difficulty_alpha", type=float, default=0.50)
    p.add_argument("--difficulty_pos_alpha", type=float, default=0.55)
    p.add_argument("--difficulty_boundary_alpha", type=float, default=0.20)

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{sanitize_name(args.experiment_name)}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    if args.pretrained_vocab:
        vocab = load_vocab_json(args.pretrained_vocab)
        vocab_src = "pretrained"
    else:
        smiles_for_vocab = extract_smiles_from_csv(args.train_csv, col="mixture")
        vocab = build_vocab(smiles_for_vocab, max_tokens=args.max_tokens)
        vocab_src = "train_build"

    ds_tr = MixtureCSVDataset(
        args.train_csv, vocab, split_name="train",
        max_seq_len=args.max_seq_len, max_mols=args.max_mols,
        token_mask_p=float(args.token_mask_p_train), seed=args.seed,
        rt_col=args.rt_col, abundance_col=args.abundance_col,
    )
    ds_va = MixtureCSVDataset(
        args.val_csv, vocab, split_name="val",
        max_seq_len=args.max_seq_len, max_mols=args.max_mols,
        token_mask_p=float(args.token_mask_p_eval), seed=args.seed + 7,
        rt_col=args.rt_col, abundance_col=args.abundance_col,
    )
    ds_te = MixtureCSVDataset(
        args.test_csv, vocab, split_name="test",
        max_seq_len=args.max_seq_len, max_mols=args.max_mols,
        token_mask_p=float(args.token_mask_p_eval), seed=args.seed + 11,
        rt_col=args.rt_col, abundance_col=args.abundance_col,
    )

    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    env_map = build_env_map([args.train_csv, args.val_csv, args.test_csv])
    num_envs = len(env_map)
    if num_envs <= 1:
        print("[WARN] only one context domain detected; context losses will be disabled.")
        args.use_context_aux = False
        args.use_context_adv = False

    meta_lookup: Dict[str, Dict[str, float]] = {}
    meta_lookup.update(build_meta_lookup(args.train_csv, args.fuel_col, args.rt_col, args.abundance_col))
    meta_lookup.update(build_meta_lookup(args.val_csv, args.fuel_col, args.rt_col, args.abundance_col))
    meta_lookup.update(build_meta_lookup(args.test_csv, args.fuel_col, args.rt_col, args.abundance_col))
    train_meta_lookup = build_meta_lookup(args.train_csv, args.fuel_col, args.rt_col, args.abundance_col)
    meta_stats = compute_train_meta_stats(train_meta_lookup, max_mols=args.max_mols)

    cfg = ChemSeqConfig(
        vocab_size=len(vocab.itos),
        max_mols=args.max_mols,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        token_dropout=args.token_dropout,
        mix_layers=args.mix_layers,
        mix_heads=args.mix_heads,
        mix_dropout=args.mix_dropout,
        rt_fourier_K=args.rt_fourier_K,
        rt_norm=args.rt_norm,
        use_pos_emb=bool(int(args.use_pos_emb)),
        use_rt_features=bool(int(args.use_rt_features)),
        use_rt_sort=bool(int(args.use_rt_sort)),
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        triplet_layers=args.triplet_layers,
        triplet_heads=args.triplet_heads,
        triplet_dropout=args.triplet_dropout,
        num_envs=num_envs,
        grl_lambda=args.grl_lambda,
        use_hybrid_stats=bool(args.use_hybrid_stats),
        causal_topk=args.causal_topk,
        causal_min_ab=args.causal_min_ab,
        use_meta_fusion=bool(args.use_meta_fusion),
        meta_input_dim=5,
        env_emb_dim=16,
        meta_dropout=0.10,
    )
    model = ChemSeqModel(cfg, pad_id=vocab.pad_id).to(device)

    if args.pretrained_mlm_encoder:
        maybe_load_pretrained_encoder(model, args.pretrained_mlm_encoder)
    configure_token_encoder_ft(model, args.encoder_frozen, args.ft_top_n_token_layers)

    pos_w = compute_pos_weight_from_csv(args.train_csv)
    loss_cfg = LossConfig(
        pos_weight=float(pos_w),
        event_loss=args.event_loss,
        event_focal_gamma=args.event_focal_gamma,
        event_label_smoothing=args.event_label_smoothing,
        logit_l2_weight=args.logit_l2_weight,
        mixsize_weight_alpha=args.mixsize_weight_alpha,
        use_fuel_aux=bool(args.use_fuel_aux),
        fuel_loss_weight=args.fuel_loss_weight,
        fuel_pos_weight=args.fuel_pos_weight,
        use_contrastive=bool(args.use_contrastive),
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_hard_neg_k=args.contrastive_hard_neg_k,
        contrastive_min_pos=args.contrastive_min_pos,
        use_context_aux=bool(args.use_context_aux),
        context_loss_weight=args.context_loss_weight,
        use_context_adv=bool(args.use_context_adv),
        context_adv_weight=args.context_adv_weight,
        use_causal=bool(args.use_causal),
        causal_weight=args.causal_weight,
        causal_use_logits=bool(args.causal_use_logits),
        neg_penalty_weight=args.neg_penalty_weight,
        neg_penalty_power=args.neg_penalty_power,
        soft_tversky_weight=args.soft_tversky_weight,
        soft_tversky_alpha=args.soft_tversky_alpha,
        soft_tversky_beta=args.soft_tversky_beta,
    )
    loss_fn = ChemSeqLoss(loss_cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1e18
    best_state = None
    best_epoch = 0
    patience = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for batch in dl_tr:
            x, y, gid, cat, k, rt, ab = batch
            x = x.to(device)
            y = y.to(device).float()
            rt = rt.to(device)
            ab = ab.to(device)

            meta_feat, env_id, hardness = build_meta_tensor(gid, meta_lookup, meta_stats, env_map, args.max_mols, device)
            fuel_proxy = meta_feat[:, 0]
            mix_size = (x[:, :, 0] != vocab.pad_id).sum(dim=1)
            sample_weight = make_curriculum_weights(
                y=y,
                hardness=hardness,
                fuel_proxy=fuel_proxy,
                epoch=ep,
                curriculum_epochs=args.curriculum_epochs,
                difficulty_alpha=args.difficulty_alpha,
                pos_alpha=args.difficulty_pos_alpha,
                boundary_alpha=args.difficulty_boundary_alpha,
            )
            fuel_mask = fuel_proxy >= 0

            x, rt, ab = maybe_apply_moldrop(x, rt, ab, vocab.pad_id, args.mol_drop_p, args.mol_drop_min_keep)
            opt.zero_grad(set_to_none=True)
            out = model(x, rt=rt, ab=ab, meta_feat=meta_feat, env_id=env_id, need_causal=bool(args.use_causal))
            logits = out["logits"]
            if logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

            loss, parts = loss_fn(
                logits=logits,
                y_true=y,
                fuel_pred=out.get("fuel_logits", None),
                fuel_proxy=fuel_proxy,
                fuel_mask=fuel_mask,
                z_chem=out.get("z_chem", None),
                env_id=env_id,
                logits_masked=out.get("logits_masked", None),
                mix_size=mix_size,
                sample_weight=sample_weight,
                context_logits=out.get("context_logits", None),
                adv_context_logits=out.get("adv_context_logits", None),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
        _, v_prob_ep, v_y_ep = eval_loader(model, dl_va, device, temperature=1.0, meta_lookup=meta_lookup, meta_stats=meta_stats, env_map=env_map, max_mols=args.max_mols)
        thr_ep, _ = tune_threshold(v_y_ep, v_prob_ep, metric=args.stop_metric, grid=args.thr_grid, strategy=args.thr_strategy)
        v_metrics_ep = compute_binary_metrics(v_y_ep, v_prob_ep, thr_ep)
        stop_key = {"precision": "prec", "recall": "rec"}.get(args.stop_metric, args.stop_metric)
        stop_score = float(v_metrics_ep[stop_key])
        history.append({"epoch": ep, "train_loss": tr_loss, "val_metric": stop_score, "val_thr": thr_ep, "val_logloss": float(v_metrics_ep["logloss"])})

        improved = stop_score > best_score + 1e-6
        if improved:
            best_score = stop_score
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        print(f"[ep {ep:03d}] train_loss={tr_loss:.4f} val_{stop_key}={stop_score:.4f} thr={thr_ep:.4f} best={best_score:.4f} patience={patience}")
        if patience >= int(args.early_stop):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = 1.0
    if args.temp_scale_on_val:
        v_logits, _, v_y = eval_loader(model, dl_va, device, temperature=1.0, meta_lookup=meta_lookup, meta_stats=meta_stats, env_map=env_map, max_mols=args.max_mols)
        temperature = fit_temperature_from_logits(v_logits, v_y, max_iter=args.temp_scale_max_iter)
        print(f"[INFO] temperature fitted on val: T={temperature:.4f}")

    thr_final = float(args.thr)
    if args.thr_mode == "tune_on_val":
        _, v_prob, v_y = eval_loader(model, dl_va, device, temperature=temperature, meta_lookup=meta_lookup, meta_stats=meta_stats, env_map=env_map, max_mols=args.max_mols)
        thr_final, best_thr_score = tune_threshold(v_y, v_prob, metric=args.tune_thr_metric, grid=args.thr_grid, strategy=args.thr_strategy)
        print(f"[INFO] tuned threshold on val: thr={thr_final:.4f} score={best_thr_score:.6f} metric={args.tune_thr_metric}")

    _, v_prob, v_y = eval_loader(model, dl_va, device, temperature=temperature, meta_lookup=meta_lookup, meta_stats=meta_stats, env_map=env_map, max_mols=args.max_mols)
    _, t_prob, t_y = eval_loader(model, dl_te, device, temperature=temperature, meta_lookup=meta_lookup, meta_stats=meta_stats, env_map=env_map, max_mols=args.max_mols)
    v_metrics = compute_binary_metrics(v_y, v_prob, thr_final)
    t_metrics = compute_binary_metrics(t_y, t_prob, thr_final)

    save_predictions_csv(os.path.join(out_dir, "val_predictions.csv"), v_y, v_prob, thr_final)
    save_predictions_csv(os.path.join(out_dir, "test_predictions.csv"), t_y, t_prob, thr_final)

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    results: Dict[str, Any] = {
        "run_id": run_id,
        "best_epoch": int(best_epoch),
        "best_stop_metric": float(best_score),
        "thr_final": float(thr_final),
        "temperature": float(temperature),
        "vocab_source": vocab_src,
        "vocab_size": int(len(vocab.itos)),
        "pos_weight": float(pos_w),
        "num_envs": int(num_envs),
        "meta_stats": meta_stats,
        "val_metrics": v_metrics,
        "test_metrics": t_metrics,
        "config": vars(args),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "vocab_source": vocab_src}, os.path.join(out_dir, "model.pt"))

    if args.summary_csv:
        row = build_summary_row(args, out_dir, best_epoch, temperature, thr_final, v_metrics, t_metrics)
        append_summary_row(args.summary_csv, row)
        print(f"[SUMMARY] appended to: {args.summary_csv}")

    print("\n===== DONE =====")
    print("out_dir:", out_dir)
    print("val:", json.dumps(v_metrics, ensure_ascii=False))
    print("test:", json.dumps(t_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

# pretrain_mlm.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data.smiles_tokenizer import build_vocab, save_vocab_json, load_vocab_json
from data.smiles_mlm_dataset import SmilesMLMDataset
from models.mlm_model import SmilesMLMModel


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_warmup_cosine_lr_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + np.cos(np.pi * t))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cos)

    return lr_lambda


@torch.no_grad()
def eval_mlm(model: SmilesMLMModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, labels=labels)
        losses.append(float(out["loss"].detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def try_save_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train")
        if val_losses:
            plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("mlm_loss")
        plt.title("MLM Loss Curve")
        plt.legend()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--corpus_csv", type=str, default="data/smiles_corpus.csv")
    p.add_argument("--out_dir", type=str, default="results_mlm")
    p.add_argument("--seed", type=int, default=123)

    # vocab
    p.add_argument("--max_tokens", type=int, default=1200)
    p.add_argument("--vocab_in", type=str, default="", help="load existing vocab.json")
    p.add_argument("--vocab_out", type=str, default="", help="save vocab.json (default: run_dir/vocab.json)")

    # dataset
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--mask_strategy", type=str, default="span", choices=["bert","span"])
    p.add_argument("--span_len", type=int, default=3)

    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # train
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # schedule/amp
    p.add_argument("--amp", action="store_true")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)

    # saving
    p.add_argument("--save_best_only", action="store_true")

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Load corpus
    df = pd.read_csv(args.corpus_csv)
    if "smiles" not in df.columns:
        raise ValueError("corpus_csv must have column: smiles")
    smiles_list = df["smiles"].astype(str).tolist()

    # Vocab
    if args.vocab_in:
        vocab = load_vocab_json(args.vocab_in)
        print(f"[Vocab] loaded: {args.vocab_in} (size={len(vocab.itos)})")
    else:
        vocab = build_vocab(smiles_list, max_tokens=args.max_tokens)
        print(f"[Vocab] built from corpus (size={len(vocab.itos)})")

    vocab_out = args.vocab_out or os.path.join(output_dir, "vocab.json")
    save_vocab_json(vocab, vocab_out)

    # Dataset split
    full_ds = SmilesMLMDataset(
        args.corpus_csv,
        vocab,
        max_seq_len=args.max_seq_len,
        mask_prob=args.mask_prob,
        seed=args.seed,
        mask_strategy=args.mask_strategy,
        span_len=args.span_len,
    )
    n = len(full_ds)
    if n == 0:
        raise ValueError("Empty MLM dataset. Check corpus_csv.")

    val_n = int(round(n * float(args.val_ratio)))
    val_n = max(0, min(val_n, n - 1))

    idx = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)

    val_idx = idx[:val_n].tolist()
    tr_idx = idx[val_n:].tolist()

    ds_tr = Subset(full_ds, tr_idx)
    ds_va = Subset(full_ds, val_idx) if val_n > 0 else None

    pin = (device.type == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    dl_va = None
    if ds_va is not None:
        dl_va = DataLoader(ds_va, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Model
    model = SmilesMLMModel(
        vocab_size=len(vocab.itos),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_id=vocab.pad_id,
        label_smoothing=args.label_smoothing,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(dl_tr))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=make_warmup_cosine_lr_lambda(total_steps, warmup_steps, args.min_lr_ratio),
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in dl_tr:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_ids=input_ids, labels=labels)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(losses)) if losses else float("inf")
        train_losses.append(tr_loss)

        if dl_va is not None:
            va_loss = eval_mlm(model, dl_va, device)
            val_losses.append(va_loss)
            improved = va_loss < best_val - 1e-6
        else:
            va_loss = float("nan")
            improved = tr_loss < best_val - 1e-6

        if improved:
            best_val = float(va_loss if dl_va is not None else tr_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[ep {epoch:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} best={best_val:.4f}")

        if not args.save_best_only:
            ckpt_path = os.path.join(output_dir, f"mlm_epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "config": vars(args),
                    "vocab_path": vocab_out,
                },
                ckpt_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save artifacts
    model_path = os.path.join(output_dir, "mlm_model.pt")
    enc_path = os.path.join(output_dir, "mlm_encoder.pt")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "config": vars(args),
            "vocab_path": vocab_out,
        },
        model_path,
    )
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "encoder_config": {
                "vocab_size": len(vocab.itos),
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "dropout": args.dropout,
                "max_seq_len": args.max_seq_len,
                "pad_id": vocab.pad_id,
            },
            "vocab_path": vocab_out,
        },
        enc_path,
    )

    curve_path = os.path.join(output_dir, "mlm_loss_curve.png")
    saved_curve = try_save_loss_curve(train_losses, val_losses, curve_path)

    results: Dict[str, Any] = {
        "run_id": run_id,
        "best_val_loss": float(best_val),
        "paths": {
            "output_dir": output_dir,
            "vocab": vocab_out,
            "mlm_model": model_path,
            "mlm_encoder": enc_path,
            "loss_curve": curve_path if saved_curve else None,
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "mask_strategy": args.mask_strategy,
        "span_len": int(args.span_len),
        "label_smoothing": float(args.label_smoothing),
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print("  vocab:", vocab_out)
    print("  mlm_model:", model_path)
    print("  mlm_encoder:", enc_path)
    if saved_curve:
        print("  loss_curve:", curve_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import random
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .smiles_tokenizer import Vocab, encode_smiles


@dataclass(frozen=True)
class MixtureItem:
    group_id: str
    mixture: str
    y: int
    category: str
    rt_raw: Optional[str] = None
    ab_raw: Optional[str] = None


class MixtureCSVDataset(Dataset):
    """
    Required columns:
      - mixture
      - label

    Optional context columns (priority order):
      - category
      - env_name
      - env_id
      - env_source

    Optional signal columns:
      - rt_col (default: rts)
      - abundance_col (default: abundance)
    """
    def __init__(
        self,
        csv_path,
        vocab: Vocab,
        split_name: str = "train",
        max_seq_len: int = 128,
        max_mols: int = 5,
        token_mask_p: float = 0.0,
        seed: int = 123,
        rt_order_mode: str = "as_is",
        rt_shuffle_seed: int = 0,
        rt_value_mode: str = "as_is",
        ab_value_mode: str = "as_is",
        rt_col: str = "rts",
        abundance_col: str = "abundance",
        rt_noise_std: float = 0.0,
        single_mol_mode: str = "none",
        return_env_id: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.split_name = str(split_name)
        self.max_seq_len = int(max_seq_len)
        self.max_mols = int(max_mols)
        self.token_mask_p = float(token_mask_p)
        self.seed = int(seed)
        self.rt_order_mode = str(rt_order_mode)
        self.rt_shuffle_seed = int(rt_shuffle_seed)
        self.rt_value_mode = str(rt_value_mode).lower()
        self.ab_value_mode = str(ab_value_mode).lower()
        self.rt_col = str(rt_col)
        self.abundance_col = str(abundance_col)
        self.rt_noise_std = float(rt_noise_std)
        self.single_mol_mode = str(single_mol_mode)
        self.epoch = 0
        self.return_env_id = bool(return_env_id)

        if self.rt_value_mode not in ("as_is", "shuffle_fixed", "shuffle_each_epoch", "drop"):
            raise ValueError(f"invalid rt_value_mode: {rt_value_mode}")
        if self.ab_value_mode not in ("as_is", "shuffle_fixed", "shuffle_each_epoch", "drop"):
            raise ValueError(f"invalid ab_value_mode: {ab_value_mode}")
        if "mixture" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain columns: mixture, label")

        if "category" not in self.df.columns:
            if "env_name" in self.df.columns:
                self.df["category"] = self.df["env_name"].astype(str)
            elif "env_id" in self.df.columns:
                self.df["category"] = self.df["env_id"].astype(str)
            elif "env_source" in self.df.columns:
                self.df["category"] = self.df["env_source"].astype(str)
            else:
                self.df["category"] = "NA"

        uniq_cats = sorted(set(self.df["category"].astype(str).fillna("NA").tolist()))
        self.env_map: Dict[str, int] = {c: i for i, c in enumerate(uniq_cats)}

        has_rt = (self.rt_col in self.df.columns)
        has_ab = (self.abundance_col in self.df.columns)

        self.items: List[MixtureItem] = []
        for i, row in self.df.iterrows():
            mix = str(row["mixture"]).strip()
            if not mix:
                continue
            y = int(row["label"])
            cat = str(row["category"]).strip() if str(row["category"]).strip() else "NA"
            gid = str(row["gid"]) if "gid" in self.df.columns and str(row["gid"]).strip() else f"{self.split_name}:{i}"
            rt_raw = str(row[self.rt_col]) if has_rt else None
            ab_raw = str(row[self.abundance_col]) if has_ab else None
            self.items.append(MixtureItem(gid, mix, y, cat, rt_raw, ab_raw))

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _parse_floats(self, s):
        if s is None:
            return None
        ss = str(s).strip()
        if not ss or ss.lower() == "nan":
            return None
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ss)
        if not nums:
            return None
        try:
            return [float(x) for x in nums]
        except Exception:
            return None

    def _parse_row_to_lists(self, mixture_str, rt_str, ab_str):
        smiles = [s for s in str(mixture_str).strip().split() if s]
        rts = self._parse_floats(rt_str)
        abs_ = self._parse_floats(ab_str)
        if rts is not None and len(rts) != len(smiles):
            rts = None
        if abs_ is not None and len(abs_) != len(smiles):
            abs_ = None
        return smiles, rts, abs_

    def _apply_single_mol(self, smiles, rts, abs_, rng: np.random.RandomState):
        if self.single_mol_mode == "none" or len(smiles) == 0:
            return smiles, rts, abs_
        if self.single_mol_mode == "first":
            idx = 0
        elif self.single_mol_mode == "random":
            idx = int(rng.randint(0, len(smiles)))
        else:
            idx = 0
        smiles2 = [smiles[idx]]
        rts2 = [rts[idx]] if rts is not None else None
        abs2 = [abs_[idx]] if abs_ is not None else None
        return smiles2, rts2, abs2

    def _apply_order(self, smiles, rts, abs_, rng: np.random.RandomState):
        mode = self.rt_order_mode
        if mode == "as_is":
            return smiles, rts, abs_
        if mode == "lex_sort":
            order = sorted(range(len(smiles)), key=lambda i: smiles[i])
        elif mode in ("shuffle_fixed", "shuffle_each_epoch"):
            order = list(range(len(smiles)))
            rng.shuffle(order)
        else:
            return smiles, rts, abs_
        smiles = [smiles[i] for i in order]
        if rts is not None:
            rts = [rts[i] for i in order]
        if abs_ is not None:
            abs_ = [abs_[i] for i in order]
        return smiles, rts, abs_

    def _apply_value_mode(self, vals: Optional[List[float]], rng: np.random.RandomState, mode: str) -> Optional[List[float]]:
        if vals is None:
            return None
        if mode == "as_is":
            return vals
        if mode == "drop":
            return None
        out = list(vals)
        idx = list(range(len(out)))
        rng.shuffle(idx)
        return [out[i] for i in idx]

    def __getitem__(self, idx: int):
        it = self.items[idx]
        base = self.seed + 10007 * (0 if self.split_name == "train" else 1) + 97 * self.rt_shuffle_seed
        if (self.rt_order_mode == "shuffle_each_epoch") or (self.rt_value_mode == "shuffle_each_epoch") or (self.ab_value_mode == "shuffle_each_epoch"):
            base = base + 1000003 * self.epoch
        rng = np.random.RandomState(base + idx)

        smiles, rts, abs_ = self._parse_row_to_lists(it.mixture, it.rt_raw, it.ab_raw)
        smiles, rts, abs_ = self._apply_single_mol(smiles, rts, abs_, rng)
        smiles, rts, abs_ = self._apply_order(smiles, rts, abs_, rng)
        rts = self._apply_value_mode(rts, rng, self.rt_value_mode)
        abs_ = self._apply_value_mode(abs_, rng, self.ab_value_mode)

        n_mols = min(len(smiles), self.max_mols)
        smiles = smiles[:n_mols]
        if rts is not None:
            rts = rts[:n_mols]
        if abs_ is not None:
            abs_ = abs_[:n_mols]

        rt_vals = None
        if rts is not None:
            rt_vals = np.asarray(rts, dtype=np.float32)
            if self.rt_noise_std > 0:
                rt_vals = rt_vals + rng.normal(0.0, self.rt_noise_std, size=rt_vals.shape).astype(np.float32)

        ab_vals = None
        if abs_ is not None:
            ab_vals = np.asarray(abs_, dtype=np.float32)
            ab_vals = np.clip(ab_vals, 0.0, None)
            s = float(ab_vals.sum())
            if s > 0.0:
                ab_vals = ab_vals / s

        x_np = np.full((self.max_mols, self.max_seq_len), int(self.vocab.pad_id), dtype=np.int64)
        for mi in range(n_mols):
            py_rng = random.Random(int(rng.randint(0, 2**31 - 1)))
            ids = encode_smiles(smiles[mi], self.vocab, self.max_seq_len, rng=py_rng, token_mask_p=self.token_mask_p)
            x_np[mi, :] = np.asarray(ids, dtype=np.int64)
        x = torch.tensor(x_np, dtype=torch.long)

        y = torch.tensor(int(it.y), dtype=torch.long)
        gid = it.group_id
        cat = it.category
        k_out = torch.tensor(int(n_mols), dtype=torch.long)

        rt = np.zeros((self.max_mols,), dtype=np.float32)
        if rt_vals is not None:
            n = min(len(rt_vals), self.max_mols)
            rt[:n] = rt_vals[:n]
        rt = torch.tensor(rt, dtype=torch.float32)

        ab = np.zeros((self.max_mols,), dtype=np.float32)
        if ab_vals is not None:
            n = min(len(ab_vals), self.max_mols)
            ab[:n] = ab_vals[:n]
        ab = torch.tensor(ab, dtype=torch.float32)

        if self.return_env_id:
            env_id = torch.tensor(int(self.env_map.get(cat, 0)), dtype=torch.long)
            return x, y, gid, cat, k_out, rt, ab, env_id
        return x, y, gid, cat, k_out, rt, ab

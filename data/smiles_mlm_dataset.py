# data/smiles_mlm_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from .smiles_tokenizer import Vocab, encode_smiles, random_non_special_token_id


@dataclass(frozen=True)
class MLMItem:
    smiles: str


class SmilesMLMDataset(Dataset):
    """
    CSV must have column: 'smiles'
    Returns dict:
      - input_ids: (L,) LongTensor
      - labels:    (L,) LongTensor, with -100 for non-masked positions

    mask_strategy:
      - "bert": token-level MLM (기존)
      - "span": 연속 span 마스킹 (SMILES 구조 학습에 유리)
    """
    def __init__(
        self,
        csv_path: str,
        vocab: Vocab,
        max_seq_len: int = 256,
        mask_prob: float = 0.15,
        seed: int = 0,
        mask_strategy: str = "span",  # ✅ default span
        span_len: int = 3,
    ):
        self.df = pd.read_csv(csv_path)
        if "smiles" not in self.df.columns:
            raise ValueError("CSV must contain column: smiles")

        self.vocab = vocab
        self.max_seq_len = int(max_seq_len)
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)

        self.mask_strategy = str(mask_strategy).lower()
        if self.mask_strategy not in ("bert", "span"):
            raise ValueError(f"mask_strategy must be bert/span, got: {mask_strategy}")
        self.span_len = int(span_len)

        smiles_list = self.df["smiles"].astype(str).tolist()
        self.items: List[MLMItem] = []
        for s in smiles_list:
            s = str(s).strip()
            if not s:
                continue
            self.items.append(MLMItem(smiles=s))

    def __len__(self) -> int:
        return len(self.items)

    def _apply_bert_mask(self, ids: List[int], rng: random.Random) -> (List[int], List[int]):
        input_ids = list(ids)
        labels = [-100] * len(ids)

        for i in range(1, len(ids) - 1):
            tok = ids[i]
            if tok == self.vocab.pad_id:
                continue
            if tok in (self.vocab.cls_id, self.vocab.sep_id):
                continue

            if rng.random() < self.mask_prob:
                labels[i] = tok
                r = rng.random()
                if r < 0.8:
                    input_ids[i] = self.vocab.mask_id
                elif r < 0.9:
                    input_ids[i] = random_non_special_token_id(rng, self.vocab)
                else:
                    input_ids[i] = tok

        return input_ids, labels

    def _apply_span_mask(self, ids: List[int], rng: random.Random) -> (List[int], List[int]):
        input_ids = list(ids)
        labels = [-100] * len(ids)

        L = len(ids)
        i = 1
        while i < L - 1:
            tok = ids[i]
            if tok == self.vocab.pad_id or tok in (self.vocab.cls_id, self.vocab.sep_id):
                i += 1
                continue

            if rng.random() < self.mask_prob:
                span = max(1, self.span_len)
                for j in range(i, min(L - 1, i + span)):
                    tj = ids[j]
                    if tj == self.vocab.pad_id or tj in (self.vocab.cls_id, self.vocab.sep_id):
                        break
                    labels[j] = tj
                    r = rng.random()
                    if r < 0.8:
                        input_ids[j] = self.vocab.mask_id
                    elif r < 0.9:
                        input_ids[j] = random_non_special_token_id(rng, self.vocab)
                    else:
                        input_ids[j] = tj
                i += span
            else:
                i += 1

        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        rng = random.Random(self.seed + idx * 10007)

        ids = encode_smiles(it.smiles, self.vocab, self.max_seq_len, rng=None, token_mask_p=0.0)

        if self.mask_strategy == "bert":
            input_ids, labels = self._apply_bert_mask(ids, rng)
        else:
            input_ids, labels = self._apply_span_mask(ids, rng)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

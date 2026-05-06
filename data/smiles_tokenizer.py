from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import random

_TWO_CHAR_ELEMENTS = {
    "Cl", "Br", "Si", "Na", "Ca", "Li", "Al", "Mg", "Zn", "Fe", "Cu", "Sn",
    "Ag", "Au", "Hg", "Pb", "Bi", "Se", "As", "Pt", "Pd", "Ni", "Co", "Mn",
    "Cr", "Sr", "Ba", "Cs"
}

_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def normalize_smiles(s: str) -> str:
    s = (s or "").replace("\u00A0", " ").strip()
    s = re.sub(r"\s+", "", s)
    return s


def tokenize_smiles(smiles: str) -> List[str]:
    s = normalize_smiles(smiles)
    tokens: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]

        if ch == "[":
            j = s.find("]", i)
            if j == -1:
                tokens.append(ch)
                i += 1
            else:
                tokens.append(s[i:j + 1])
                i = j + 1
            continue

        if i + 1 < len(s) and s[i:i + 2] in _TWO_CHAR_ELEMENTS:
            tokens.append(s[i:i + 2])
            i += 2
            continue

        if ch == "@" and i + 1 < len(s) and s[i + 1] == "@":
            tokens.append("@@")
            i += 2
            continue

        if ch == "%" and i + 2 < len(s) and s[i + 1].isdigit() and s[i + 2].isdigit():
            tokens.append(s[i:i + 3])
            i += 3
            continue

        tokens.append(ch)
        i += 1

    return tokens


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    cls: str = "[CLS]"
    sep: str = "[SEP]"
    mask: str = "[MASK]"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk]

    @property
    def cls_id(self) -> int:
        return self.stoi[self.cls]

    @property
    def sep_id(self) -> int:
        return self.stoi[self.sep]

    @property
    def mask_id(self) -> int:
        return self.stoi[self.mask]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def build_vocab(smiles_list: Sequence[str], max_tokens: int = 800) -> Vocab:
    from collections import Counter
    c = Counter()
    for s in smiles_list:
        toks = tokenize_smiles(s)
        c.update(toks)

    max_tokens = max(len(_SPECIALS), int(max_tokens))
    itos = list(_SPECIALS)
    for tok, _ in c.most_common(max_tokens - len(itos)):
        if tok not in itos:
            itos.append(tok)

    stoi = {t: i for i, t in enumerate(itos)}
    if stoi["[PAD]"] != 0:
        pad_idx = stoi["[PAD]"]
        itos[0], itos[pad_idx] = itos[pad_idx], itos[0]
        stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def save_vocab_json(vocab: Vocab, path: str) -> None:
    obj = {"itos": vocab.itos}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_vocab_json(path: str) -> Vocab:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    itos = list(obj["itos"])
    stoi = {t: i for i, t in enumerate(itos)}

    if stoi.get("[PAD]", None) != 0:
        raise ValueError("Loaded vocab must have [PAD] at index 0.")
    return Vocab(stoi=stoi, itos=itos)


def encode_smiles(
    smiles: str,
    vocab: Vocab,
    max_len: int,
    rng: Optional[random.Random] = None,
    token_mask_p: float = 0.0,
) -> List[int]:
    toks = tokenize_smiles(smiles)
    ids = [vocab.cls_id] + [vocab.stoi.get(t, vocab.unk_id) for t in toks] + [vocab.sep_id]

    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = vocab.sep_id

    if rng is not None and token_mask_p > 0:
        for i in range(1, len(ids) - 1):
            if rng.random() < token_mask_p and ids[i] not in (vocab.pad_id, vocab.cls_id, vocab.sep_id):
                ids[i] = vocab.mask_id

    if len(ids) < max_len:
        ids += [vocab.pad_id] * (max_len - len(ids))
    return ids


def random_non_special_token_id(rng: random.Random, vocab: Vocab) -> int:
    if vocab.vocab_size <= len(_SPECIALS):
        return vocab.unk_id
    lo = len(_SPECIALS)
    return rng.randrange(lo, vocab.vocab_size)

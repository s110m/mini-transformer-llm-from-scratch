from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CharTokenizer:
    """
    A tiny tokenizer from scratch.

    Real LLMs usually use BPE/SentencePiece. For this laptop project we use
    character-level tokenization because it is transparent and easy to debug.

    Important LLM concepts:
    - Vocabulary size: len(stoi)
    - Text -> token ids
    - Token ids -> embeddings inside the model
    """
    stoi: dict
    itos: dict
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

    @classmethod
    def train_from_texts(cls, texts: List[str]) -> "CharTokenizer":
        special = ["<PAD>", "<BOS>", "<EOS>"]
        chars = sorted(set("".join(texts)))
        tokens = special + chars
        stoi = {ch: i for i, ch in enumerate(tokens)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi[" "]))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for idx in ids:
            tok = self.itos[int(idx)]
            if tok in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            out.append(tok)
        return "".join(out)

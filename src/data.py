from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch


def load_texts(data_dir: str) -> List[str]:
    data_dir = Path(data_dir)
    texts = []
    texts.append((data_dir / "raw_corpus.txt").read_text(encoding="utf-8"))
    for file in ["sft.jsonl", "preferences.jsonl"]:
        with open(data_dir / file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts.extend(str(v) for v in row.values())
    return texts


def make_pretrain_tokens(tokenizer, data_dir: str):
    text = Path(data_dir, "raw_corpus.txt").read_text(encoding="utf-8")
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_lm_batch(tokens, batch_size: int, block_size: int, device: str):
    """
    Pretraining batch.

    x: tokens from position i to i+block_size-1
    y: next tokens from i+1 to i+block_size

    This is the next-token prediction objective.
    """
    max_start = len(tokens) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def load_sft_examples(data_dir: str):
    rows = []
    with open(Path(data_dir, "sft.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_sft(instruction: str, answer: str) -> str:
    return f"Question: {instruction}\nAnswer: {answer}"


def get_sft_batch(tokenizer, examples, batch_size: int, block_size: int, device: str):
    """
    SFT batch.

    We train on input-output examples using the same next-token objective.
    In larger projects, you may mask the question part so loss is mainly on answer.
    """
    xs, ys = [], []
    for _ in range(batch_size):
        row = random.choice(examples)
        text = format_sft(row["instruction"], row["answer"])
        ids = tokenizer.encode(text)
        ids = ids[: block_size + 1]
        if len(ids) < block_size + 1:
            ids = ids + [tokenizer.pad_id] * (block_size + 1 - len(ids))
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        y[y == tokenizer.pad_id] = -100
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def load_preferences(data_dir: str):
    rows = []
    with open(Path(data_dir, "preferences.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_pref_batch(tokenizer, examples, batch_size: int, block_size: int, device: str):
    """
    Preference batch for DPO.

    For each prompt we have:
    - chosen answer: preferred
    - rejected answer: worse
    """
    chosen, rejected = [], []
    for _ in range(batch_size):
        row = random.choice(examples)
        c = f"Question: {row['prompt']}\nAnswer: {row['chosen']}"
        r = f"Question: {row['prompt']}\nAnswer: {row['rejected']}"
        cids = tokenizer.encode(c)[:block_size]
        rids = tokenizer.encode(r)[:block_size]
        cids += [tokenizer.pad_id] * (block_size - len(cids))
        rids += [tokenizer.pad_id] * (block_size - len(rids))
        chosen.append(torch.tensor(cids))
        rejected.append(torch.tensor(rids))
    return torch.stack(chosen).to(device), torch.stack(rejected).to(device)

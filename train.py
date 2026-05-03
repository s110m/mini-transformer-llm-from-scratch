from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from src.tokenizer import CharTokenizer
from src.model import GPTConfig, MiniGPT, freeze_base_for_lora
from src.data import (
    load_texts, make_pretrain_tokens, get_lm_batch,
    load_sft_examples, get_sft_batch,
    load_preferences, get_pref_batch,
)
from src.train_utils import generate, dpo_loss


def save_checkpoint(model, tokenizer, cfg, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "tokenizer": tokenizer}, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MiniGPT(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt["tokenizer"], ckpt["cfg"]


def train_pretrain(args, device):
    texts = load_texts(args.data_dir)
    tokenizer = CharTokenizer.train_from_texts(texts)

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_query_heads=args.n_query_heads,
        n_kv_heads=args.n_kv_heads,
        use_moe=not args.no_moe,
        moe_dense=args.dense_moe,
        use_latent_kv=not args.no_latent_kv,
    )

    model = MiniGPT(cfg).to(device)
    tokens = make_pretrain_tokens(tokenizer, args.data_dir)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens in tiny corpus: {len(tokens)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in tqdm(range(args.max_steps), desc="pretraining"):
        x, y = get_lm_batch(tokens, args.batch_size, args.block_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    save_checkpoint(model, tokenizer, cfg, "checkpoints/pretrained.pt")
    print(generate(model, tokenizer, "A transformer", device=device))
    return model, tokenizer, cfg


def train_sft(args, device, model=None, tokenizer=None, cfg=None):
    if model is None:
        model, tokenizer, cfg = load_checkpoint("checkpoints/pretrained.pt", device)

    examples = load_sft_examples(args.data_dir)

    # Demonstrate LoRA in finetuning if requested.
    # For simplicity, LoRA must be enabled before model creation in a real large model.
    # Here we keep standard full SFT by default.
    if args.use_lora_sft:
        print("Note: LoRA flag is educational here. For a clean LoRA-only run, start with --use_lora_sft from scratch.")
        freeze_base_for_lora(model)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    for step in tqdm(range(args.max_steps), desc="SFT"):
        x, y = get_sft_batch(tokenizer, examples, args.batch_size, args.block_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    save_checkpoint(model, tokenizer, cfg, "checkpoints/sft.pt")
    print(generate(model, tokenizer, "Question: What is KV caching?\nAnswer:", device=device))
    return model, tokenizer, cfg


def train_dpo(args, device, model=None, tokenizer=None, cfg=None):
    if model is None:
        model, tokenizer, cfg = load_checkpoint("checkpoints/sft.pt", device)

    reference = copy.deepcopy(model).to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    prefs = load_preferences(args.data_dir)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in tqdm(range(args.max_steps), desc="preference tuning DPO"):
        chosen, rejected = get_pref_batch(tokenizer, prefs, args.batch_size, args.block_size, device)
        loss = dpo_loss(model, reference, chosen, rejected, tokenizer.pad_id, beta=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step {step}: dpo_loss={loss.item():.4f}")

    save_checkpoint(model, tokenizer, cfg, "checkpoints/dpo.pt")
    print(generate(model, tokenizer, "Question: What is temperature?\nAnswer:", temperature=0.7, device=device))
    return model, tokenizer, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "sft", "dpo", "all"], default="all")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_embd", type=int, default=96)
    parser.add_argument("--n_query_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_moe", action="store_true")
    parser.add_argument("--dense_moe", action="store_true")
    parser.add_argument("--no_latent_kv", action="store_true")
    parser.add_argument("--use_lora_sft", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if args.stage == "pretrain":
        train_pretrain(args, device)
    elif args.stage == "sft":
        train_sft(args, device)
    elif args.stage == "dpo":
        train_dpo(args, device)
    else:
        model, tokenizer, cfg = train_pretrain(args, device)
        model, tokenizer, cfg = train_sft(args, device, model, tokenizer, cfg)
        train_dpo(args, device, model, tokenizer, cfg)


if __name__ == "__main__":
    main()

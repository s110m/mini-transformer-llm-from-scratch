import torch
from src.tokenizer import CharTokenizer
from src.data import load_texts
from src.model import GPTConfig, MiniGPT


texts = load_texts("data")
tok = CharTokenizer.train_from_texts(texts)

cfg = GPTConfig(vocab_size=tok.vocab_size)
model = MiniGPT(cfg)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Mini Transformer LLM configuration")
print("----------------------------------")
print(f"vocabulary size: {tok.vocab_size}")
print(f"block/context size: {cfg.block_size}")
print(f"layers: {cfg.n_layer}")
print(f"embedding dimension: {cfg.n_embd}")
print(f"query heads: {cfg.n_query_heads}")
print(f"key/value heads: {cfg.n_kv_heads}")
print(f"uses GQA: {cfg.n_kv_heads < cfg.n_query_heads}")
print(f"uses MoE: {cfg.use_moe}")
print(f"MoE experts: {cfg.moe_num_experts}")
print(f"MoE top-k: {cfg.moe_top_k}")
print(f"uses latent KV: {cfg.use_latent_kv}")
print(f"total parameters: {total:,}")
print(f"trainable parameters: {trainable:,}")

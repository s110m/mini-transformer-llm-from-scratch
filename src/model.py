from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_embd: int = 128
    n_query_heads: int = 4

    # GQA: number of key/value heads can be smaller than query heads.
    # MHA: n_kv_heads == n_query_heads
    # GQA: n_kv_heads < n_query_heads
    # MQA: n_kv_heads == 1
    n_kv_heads: int = 2

    dropout: float = 0.1

    # MoE settings.
    use_moe: bool = True
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_dense: bool = False

    # Latent KV compression.
    # This stores compressed K/V and expands them again.
    # It is educational and simple, not an optimized production MLA implementation.
    use_latent_kv: bool = True
    latent_kv_dim: int = 32

    # LoRA settings.
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0


class RMSNorm(nn.Module):
    """
    RMSNorm: common in modern LLMs.

    It stabilizes activations. Lecture 2 discusses LayerNorm/RMSNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return self.weight * x


class LoRALinear(nn.Module):
    """
    LoRA: W = W0 + B A.

    During PEFT, the original weight W0 can be frozen and only A/B are trained.
    This class is used in attention projections when use_lora=True.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=8.0, bias=False):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + (self.alpha / self.rank) * self.B(self.A(x))


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE: Rotary Position Embedding.

    This injects position information by rotating pairs of dimensions in Q and K.
    Shape: [batch, heads, seq, head_dim]
    """
    b, h, t, d = x.shape
    assert d % 2 == 0, "head_dim must be even for RoPE"
    half = d // 2
    device = x.device

    pos = torch.arange(t, device=device).float()
    freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    angles = pos[:, None] * freq[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Decoder-only causal self-attention.

    Covers:
    - Q, K, V projections
    - attention equation softmax(QK^T / sqrt(d))V
    - causal mask
    - GQA: fewer K/V heads than Q heads
    - KV cache for inference
    - latent KV compression for cache memory reduction
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_query_heads % cfg.n_kv_heads == 0
        assert cfg.n_embd % cfg.n_query_heads == 0

        self.cfg = cfg
        self.nq = cfg.n_query_heads
        self.nkv = cfg.n_kv_heads
        self.head_dim = cfg.n_embd // cfg.n_query_heads
        self.group_size = self.nq // self.nkv

        Linear = LoRALinear if cfg.use_lora else nn.Linear
        linear_kwargs = {}
        if cfg.use_lora:
            linear_kwargs = {"rank": cfg.lora_rank, "alpha": cfg.lora_alpha, "bias": False}
        else:
            linear_kwargs = {"bias": False}

        self.q_proj = Linear(cfg.n_embd, cfg.n_embd, **linear_kwargs)
        self.k_proj = Linear(cfg.n_embd, self.nkv * self.head_dim, **linear_kwargs)
        self.v_proj = Linear(cfg.n_embd, self.nkv * self.head_dim, **linear_kwargs)
        self.out_proj = Linear(cfg.n_embd, cfg.n_embd, **linear_kwargs)

        if cfg.use_latent_kv:
            self.k_down = nn.Linear(self.head_dim, cfg.latent_kv_dim, bias=False)
            self.v_down = nn.Linear(self.head_dim, cfg.latent_kv_dim, bias=False)
            self.k_up = nn.Linear(cfg.latent_kv_dim, self.head_dim, bias=False)
            self.v_up = nn.Linear(cfg.latent_kv_dim, self.head_dim, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def _shape_qkv(self, q, k, v):
        b, t, _ = q.shape
        q = q.view(b, t, self.nq, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.nkv, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.nkv, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, t, c = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = self._shape_qkv(q, k, v)

        # Position information.
        q = apply_rope(q)
        k = apply_rope(k)

        # Latent KV cache compression.
        # We compress K/V before storing, then expand for attention.
        if self.cfg.use_latent_kv:
            k_latent = self.k_down(k)
            v_latent = self.v_down(v)

            if kv_cache is not None:
                old_k_latent, old_v_latent = kv_cache
                k_latent = torch.cat([old_k_latent, k_latent], dim=2)
                v_latent = torch.cat([old_v_latent, v_latent], dim=2)

            new_cache = (k_latent, v_latent) if use_cache else None
            k = self.k_up(k_latent)
            v = self.v_up(v_latent)
        else:
            if kv_cache is not None:
                old_k, old_v = kv_cache
                k = torch.cat([old_k, k], dim=2)
                v = torch.cat([old_v, v], dim=2)
            new_cache = (k, v) if use_cache else None

        # GQA: repeat K/V heads so each group of query heads shares them.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Attention scores.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask. During cached inference, q length can be 1 and k length is longer.
        q_len, k_len = att.shape[-2], att.shape[-1]
        if kv_cache is None:
            mask = torch.tril(torch.ones(q_len, k_len, device=x.device)).view(1, 1, q_len, k_len)
            att = att.masked_fill(mask == 0, float("-inf"))

        probs = F.softmax(att, dim=-1)
        probs = self.dropout(probs)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.out_proj(y)

        return y, new_cache


class FeedForward(nn.Module):
    """Standard transformer MLP block."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x), torch.tensor(0.0, device=x.device)


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts MLP.

    Dense MoE:
        all experts are used and weighted.

    Sparse MoE:
        top-k experts are used per token.

    Routing is done for each token independently.
    Routing collapse is discouraged by a load-balancing loss.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.n_embd, cfg.moe_num_experts)
        self.experts = nn.ModuleList([FeedForward(cfg).net for _ in range(cfg.moe_num_experts)])

    def forward(self, x):
        b, t, c = x.shape
        flat = x.view(b * t, c)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)

        if self.cfg.moe_dense:
            # Dense MoE: weighted average of all expert outputs.
            outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
            y = (probs.unsqueeze(-1) * outputs).sum(dim=1)
        else:
            # Sparse MoE: only top-k experts per token.
            top_p, top_i = torch.topk(probs, k=self.cfg.moe_top_k, dim=-1)
            top_p = top_p / top_p.sum(dim=-1, keepdim=True)

            y = torch.zeros_like(flat)
            for slot in range(self.cfg.moe_top_k):
                expert_ids = top_i[:, slot]
                weights = top_p[:, slot].unsqueeze(-1)
                for expert_id, expert in enumerate(self.experts):
                    mask = expert_ids == expert_id
                    if mask.any():
                        y[mask] += weights[mask] * expert(flat[mask])

        # Load-balancing auxiliary loss.
        # importance: average routing probability
        # load: fraction of tokens assigned to each expert
        importance = probs.mean(dim=0)
        assigned = F.one_hot(probs.argmax(dim=-1), num_classes=self.cfg.moe_num_experts).float()
        load = assigned.mean(dim=0)
        aux_loss = self.cfg.moe_num_experts * torch.sum(importance * load)

        return y.view(b, t, c), aux_loss


class Block(nn.Module):
    """One decoder-only Transformer block: RMSNorm -> causal attention -> RMSNorm -> MLP/MoE."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ff = MoEFeedForward(cfg) if cfg.use_moe else FeedForward(cfg)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        ff_out, aux_loss = self.ff(self.norm2(x))
        x = x + ff_out
        return x, new_cache, aux_loss


class MiniGPT(nn.Module):
    """
    Tiny GPT-like decoder-only model.

    Pipeline:
    token ids -> token embeddings -> transformer blocks -> final RMSNorm -> logits

    Logits are later converted into probabilities by softmax during generation.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: common in language models.
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None, kv_cache=None, use_cache=False):
        b, t = idx.shape
        assert t <= self.cfg.block_size or use_cache, "Sequence too long"

        x = self.token_emb(idx)
        x = self.drop(x)

        new_caches = []
        aux_losses = []
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for block, cache in zip(self.blocks, kv_cache):
            x, new_cache, aux_loss = block(x, kv_cache=cache, use_cache=use_cache)
            new_caches.append(new_cache)
            aux_losses.append(aux_loss)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross entropy objective for next-token prediction.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            # Add MoE load-balancing loss.
            if aux_losses:
                loss = loss + 0.01 * torch.stack(aux_losses).mean()

        return logits, loss, new_caches if use_cache else None


def freeze_base_for_lora(model: nn.Module):
    """
    Freeze all parameters except LoRA matrices and small heads.

    This demonstrates parameter-efficient finetuning.
    """
    for name, p in model.named_parameters():
        p.requires_grad = False
        if ".A." in name or ".B." in name:
            p.requires_grad = True
        if "lm_head" in name:
            p.requires_grad = True

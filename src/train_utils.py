from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=20, use_kv_cache=True, device="cpu"):
    """
    Autoregressive generation.

    Concepts:
    - Predict next token
    - Convert logits to probabilities
    - Apply temperature
    - Top-k sampling
    - Optional KV caching
    """
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    kv_cache = None

    for step in range(max_new_tokens):
        if use_kv_cache and step > 0:
            idx_cond = idx[:, -1:]
        else:
            idx_cond = idx[:, -model.cfg.block_size:]

        logits, _, kv_cache = model(idx_cond, kv_cache=kv_cache, use_cache=use_kv_cache)
        logits = logits[:, -1, :]

        # Temperature: lower -> sharper, higher -> flatter.
        logits = logits / max(temperature, 1e-6)

        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            cutoff = values[:, [-1]]
            logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(idx[0].tolist())


def sequence_logprob(model, seq, pad_id):
    """
    Average log-probability of a full sequence.

    Used for preference tuning. We score how likely the model thinks a chosen
    or rejected answer is.
    """
    x = seq[:, :-1]
    y = seq[:, 1:]

    logits, _, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)

    mask = (y != pad_id).float()
    token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def dpo_loss(policy, reference, chosen, rejected, pad_id, beta=0.1):
    """
    Simple DPO-style preference tuning.

    We want:
        policy(chosen) > policy(rejected)

    But we compare against a frozen reference model so we do not drift too far.
    """
    pi_c = sequence_logprob(policy, chosen, pad_id)
    pi_r = sequence_logprob(policy, rejected, pad_id)

    with torch.no_grad():
        ref_c = sequence_logprob(reference, chosen, pad_id)
        ref_r = sequence_logprob(reference, rejected, pad_id)

    advantage = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    return -F.logsigmoid(advantage).mean()

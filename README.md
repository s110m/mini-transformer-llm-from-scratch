# Mini Transformer LLM From Scratch in PyTorch

A small, readable educational project that implements a decoder-only Transformer from scratch in PyTorch.

It covers:

1. Decoder-only Transformer
2. Mixture of Experts (MoE)
3. Dense MoE vs Sparse MoE
4. Token-level routing
5. Routing collapse + load-balancing loss
6. Next-token prediction with sampling
7. Temperature
8. KV cache
9. Grouped-Query Attention (GQA)
10. Latent KV cache compression
11. Pretraining
12. Finetuning
13. Preference tuning
14. Objectives for pretraining / finetuning / preference tuning
15. Data mixtures
16. Number of parameters
17. Vocabulary size
18. Number of training tokens
19. Supervised finetuning (SFT)
20. LoRA finetuning

Also included:
- Tokenizer from scratch
- Attention equation
- Positional embeddings
- RoPE
- LayerNorm / RMSNorm
- Cross-entropy loss
- DPO-style preference tuning
- Simple generation
- Debug-friendly code

## Project idea

We build a tiny GPT-like model and train it in three stages:

```text
Stage 1: Pretraining
Tiny raw text -> next-token prediction

Stage 2: Supervised Fine-Tuning (SFT)
Instruction-answer examples -> assistant behavior

Stage 3: Preference tuning
Chosen vs rejected answers -> prefer better responses
```

This project is intentionally small so you can run it on a laptop.

## Install

```bash
pip install torch tqdm
```

CPU is enough. GPU is optional.

## Run everything

```bash
python train.py --stage all
```

## Run stages separately

```bash
python train.py --stage pretrain
python train.py --stage sft
python train.py --stage dpo
python generate.py --prompt "Question: What is a transformer?"
```

## Useful debug commands

Use a tiny run:

```bash
python train.py --stage all --max_steps 30 --batch_size 8 --block_size 64
```

Print model size:

```bash
python inspect_model.py
```

## Files

```text
src/tokenizer.py       Character tokenizer from scratch
src/model.py           Decoder-only Transformer, attention, GQA, MoE, LoRA, KV cache
src/data.py            Tiny datasets and batching
src/train_utils.py     Losses, training loop, generation helpers
train.py               Pretraining, SFT, DPO
generate.py            Generate text
inspect_model.py       Count parameters and explain config
```

## Why character tokenizer?

A real LLM uses BPE/SentencePiece tokenization. Here we use character-level tokenization because:
- it is fully from scratch
- it needs no external files
- it is easy to debug
- it runs on a laptop

The idea is the same:

```text
text -> token ids -> embeddings -> transformer -> logits -> probabilities
```

## Expected result

This is a tiny educational model, not ChatGPT. After a short run, it should learn simple patterns from the included dataset and produce rough instruction-style answers.

For GitHub/LinkedIn, the value of this project is not final performance. The value is that the code clearly shows the full LLM lifecycle:

```text
pretraining -> SFT -> preference tuning -> inference optimization
```

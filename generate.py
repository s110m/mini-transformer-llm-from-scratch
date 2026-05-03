import argparse
import torch

from src.model import MiniGPT
from src.train_utils import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dpo.pt")
    parser.add_argument("--prompt", default="Question: What is a transformer?\nAnswer:")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--no_kv_cache", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = MiniGPT(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    tokenizer = ckpt["tokenizer"]

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_kv_cache=not args.no_kv_cache,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()

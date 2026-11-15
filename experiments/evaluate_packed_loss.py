#!/usr/bin/env python3
"""Evaluate matched Conditional Attention checkpoints on both packed held-out surfaces."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from train_matched_cpt import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--adapter", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(model, dataset, batch_size):
    import torch
    import torch.nn.functional as F

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
            labels = input_ids[:, 1:]
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
            )
            loss_sum += float(losses.sum())
            token_count += int(labels.numel())
    nll = loss_sum / token_count
    return {"blocks": len(dataset), "tokens": token_count, "nll": nll, "ppl": math.exp(nll)}


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from packed_data import PackedTokenDataset

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    datasets = {
        surface: PackedTokenDataset(args.pack, "validation", surface, max_blocks=args.max_blocks)
        for surface in ("conditional", "random")
    }
    report = {"split": "validation", "models": {}}
    for name, adapter in args.adapter:
        model_args = argparse.Namespace(model=args.model, adapter=Path(adapter))
        model = load_model(model_args, tokenizer).cuda().eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        report["models"][name] = {
            surface: evaluate(model, dataset, args.batch_size)
            for surface, dataset in datasets.items()
        }
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

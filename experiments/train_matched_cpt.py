#!/usr/bin/env python3
"""Continue one shared adapter on matched Conditional Attention or random dialect surfaces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("conditional", "random"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--token-learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, tokenizer):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    manifest = json.loads((args.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    contract = {
        "surface": args.surface,
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "adapter": str(args.adapter),
        "pack": str(args.pack),
        "block_size": manifest["block_size"],
        "train_blocks": (manifest["splits"]["train"].get("surfaces") or manifest["splits"]["train"]["arms"])[args.surface]["blocks"],
        "validation_blocks": (manifest["splits"]["validation"].get("surfaces") or manifest["splits"]["validation"]["arms"])[args.surface]["blocks"],
        "max_steps": args.max_steps,
        "effective_batch": args.batch_size * args.gradient_accumulation,
        "training_token_budget": args.max_steps
        * args.batch_size
        * args.gradient_accumulation
        * int(manifest["block_size"]),
        "learning_rate": args.learning_rate,
        "token_learning_rate": args.token_learning_rate,
        "eval_steps": args.eval_steps,
        "seed": args.seed,
    }
    print(json.dumps(contract, indent=2), flush=True)
    if not args.launch:
        return
    if args.output.exists():
        raise FileExistsError(args.output)

    import sys
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from packed_data import PackedTokenDataset, stack_collator

    train = PackedTokenDataset(args.pack, "train", args.surface)
    validation = PackedTokenDataset(args.pack, "validation", args.surface, max_blocks=128)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(args, tokenizer).cuda()

    class MatchedTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            token_parameters = []
            other_parameters = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                target = token_parameters if "trainable_tokens_delta" in name else other_parameters
                target.append(parameter)
            if not other_parameters:
                raise RuntimeError("shared adapter exposes no trainable LoRA parameters")
            groups = [{"params": other_parameters, "lr": args.learning_rate}]
            if token_parameters:
                groups.append({"params": token_parameters, "lr": args.token_learning_rate})
            self.optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
            return self.optimizer

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = MatchedTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=stack_collator,
    )
    result = trainer.train()
    evaluation = trainer.evaluate()
    final = args.output / "final_adapter"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    eval_loss = float(evaluation.get("eval_loss", math.nan))
    report = {
        **contract,
        "train_metrics": result.metrics,
        "eval_metrics": evaluation,
        "log_history": trainer.state.log_history,
        "eval_perplexity": math.exp(eval_loss)
        if math.isfinite(eval_loss) and eval_loss < 80
        else None,
    }
    (args.output / "run_contract.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

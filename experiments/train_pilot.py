#!/usr/bin/env python3
"""RQ2 50-step pilot trainer with a mandatory new-token gradient probe."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ARMS = ("raw", "pass_a", "pass_ab")
DEFAULT_STEPS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--gradient-probe", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-eval-blocks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--token-learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def special_ids(tokenizer, init_map: dict) -> list[int]:
    ids = []
    for row in init_map["items"]:
        token = row["token"]
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token not in tokenizer.get_vocab() or token_id < int(init_map["base_vocab"]):
            raise RuntimeError(f"new special missing or aliases base vocabulary: {token}")
        ids.append(token_id)
    if len(ids) != int(init_map["n_specials"]) or len(ids) != len(set(ids)):
        raise RuntimeError("special token ID count/uniqueness check failed")
    return ids


def initialize_new_rows(model, init_map: dict, ids: list[int]) -> None:
    import torch

    embed = model.get_input_embeddings().weight
    with torch.no_grad():
        for row, new_id in zip(init_map["items"], ids, strict=True):
            source_ids = [int(value) for value in row["base_token_ids"]]
            embed[new_id].copy_(embed[source_ids].mean(dim=0))


def build_model(args, tokenizer, init_map, ids):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if ids:
        model.resize_token_embeddings(len(tokenizer))
        initialize_new_rows(model, init_map, ids)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_kwargs = dict(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
    )
    if ids:
        lora_kwargs.update(
            trainable_token_indices={"model.embed_tokens": ids, "lm_head": ids},
            ensure_weight_tying=True,
        )
    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    return model


def run_gradient_probe(model, dataset, collator, require_token_adapter: bool) -> dict:
    import torch

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_adapter = getattr(input_embeddings, "token_adapter", None)
    output_adapter = getattr(output_embeddings, "token_adapter", None)
    output_tied_adapter = getattr(output_adapter, "tied_adapter", None)
    if require_token_adapter and (
        input_adapter is None or output_adapter is None or output_tied_adapter is not input_adapter
    ):
        raise RuntimeError(
            "new-token input/output adapters are not tied; model could read but not learn to emit dialect tokens; "
            f"input={type(input_embeddings).__name__} input_adapter={type(input_adapter).__name__}; "
            f"output={type(output_embeddings).__name__} output_adapter={type(output_adapter).__name__}"
        )
    model.train()
    device = torch.device("cuda")
    batch = collator([dataset[0]])
    batch = {key: value.to(device) for key, value in batch.items()}
    loss = model(**batch).loss
    loss.backward()
    trainable = []
    token_adapters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        nonzero = int(torch.count_nonzero(grad).item()) if grad is not None else 0
        row = {"name": name, "shape": list(parameter.shape), "grad_nonzero": nonzero}
        trainable.append(row)
        if "trainable_tokens" in name:
            token_adapters.append(row)
    model.zero_grad(set_to_none=True)
    if require_token_adapter and (
        not token_adapters or not any(row["grad_nonzero"] > 0 for row in token_adapters)
    ):
        raise RuntimeError("new-token adapter received no nonzero gradient")
    if not any(row["grad_nonzero"] > 0 and "lora_" in row["name"] for row in trainable):
        raise RuntimeError("LoRA parameters received no nonzero gradient")
    return {
        "loss": float(loss.detach().cpu()),
        "trainable_parameter_tensors": len(trainable),
        "nonzero_gradient_tensors": sum(row["grad_nonzero"] > 0 for row in trainable),
        "token_adapter_gradients": token_adapters,
        "input_output_token_adapter_tied": True if require_token_adapter else None,
        "input_embedding_type": type(input_embeddings).__name__,
        "output_embedding_type": type(output_embeddings).__name__,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from packed_dataset import PackedTokenDataset, stack_collator
    from transformers import AutoTokenizer

    manifest = json.loads((args.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if args.arm == "raw":
        tokenizer_path = args.model
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
        init_map = {
            "base_vocab": len(tokenizer), "new_vocab": len(tokenizer), "n_specials": 0,
            "n_pass_a": 0, "n_native_fallback": 0, "n_pass_b": 0, "items": [],
        }
        ids = []
    else:
        suffix = "_pass_a" if args.arm == "pass_a" else ""
        tokenizer_path = args.assets / f"tokenizer{suffix}"
        init_map = json.loads(
            (args.assets / f"special_init_map{suffix}.json").read_text(encoding="utf-8")
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
        ids = special_ids(tokenizer, init_map)
    effective_batch = args.batch_size * args.gradient_accumulation
    if effective_batch != int(manifest["effective_batch"]):
        raise RuntimeError(f"effective batch must be {manifest['effective_batch']}, got {effective_batch}")
    train_dataset = PackedTokenDataset(args.pack, "train", args.arm)
    eval_dataset = PackedTokenDataset(
        args.pack, "validation", args.arm, max_blocks=args.max_eval_blocks
    )
    dry = {
        "launch": args.launch,
        "gradient_probe": args.gradient_probe,
        "arm": args.arm,
        "model": str(args.model),
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab": len(tokenizer),
        "base_vocab": init_map["base_vocab"],
        "n_specials": len(ids),
        "special_counts": {
            "pass_a": init_map["n_pass_a"],
            "native_fallback": init_map["n_native_fallback"],
            "pass_b": init_map["n_pass_b"],
        },
        "block_size": manifest["block_size"],
        "train_blocks": len(train_dataset),
        "eval_blocks": len(eval_dataset),
        "max_steps": args.max_steps,
        "effective_batch": effective_batch,
        "learning_rate": args.learning_rate,
        "token_learning_rate": args.token_learning_rate,
        "token_budget": args.max_steps * effective_batch * int(manifest["block_size"]),
        "output": str(args.output),
    }
    print(json.dumps(dry, indent=2, ensure_ascii=False), flush=True)
    if not args.launch and not args.gradient_probe:
        print("dry-run only", flush=True)
        return

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    class PilotTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            token_parameters = []
            backbone_parameters = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if "trainable_tokens_delta" in name:
                    token_parameters.append(parameter)
                else:
                    backbone_parameters.append(parameter)
            if not backbone_parameters:
                raise RuntimeError("no LoRA trainable parameter group")
            parameter_groups = [{"params": backbone_parameters, "lr": args.learning_rate}]
            if token_parameters:
                parameter_groups.append(
                    {"params": token_parameters, "lr": args.token_learning_rate}
                )
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            return self.optimizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"pilot requires exactly one GPU, found {torch.cuda.device_count()}")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args, tokenizer, init_map, ids).cuda()
    trainable, total = model.get_nb_trainable_parameters()
    probe = run_gradient_probe(
        model, train_dataset, stack_collator, require_token_adapter=bool(ids)
    )
    print(json.dumps({"gradient_probe": probe}, indent=2), flush=True)
    if not args.launch:
        return

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preflight.json").write_text(
        json.dumps({**dry, "gradient_probe": probe}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.04,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps" if args.save_checkpoints else "no",
        save_steps=max(1, args.max_steps // 2),
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        include_tokens_per_second=True,
    )
    trainer = PilotTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=stack_collator,
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    evaluation = trainer.evaluate()
    final = args.output / "final_adapter"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    eval_loss = float(evaluation.get("eval_loss", math.nan))
    contract = {
        **dry,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "train_metrics": result.metrics,
        "eval_metrics": evaluation,
        "eval_perplexity": math.exp(eval_loss) if math.isfinite(eval_loss) and eval_loss < 80 else None,
    }
    (args.output / "run_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()

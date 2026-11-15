#!/usr/bin/env python3
"""Short task-alignment SFT on top of the frozen one-epoch K443 adapter."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("raw", "pass_a", "pass_ab"), default="pass_ab")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path,
                   help="Explicit tokenizer shared with the CPT adapter.")
    p.add_argument("--init-map-path", type=Path,
                   help="Explicit special-token initialization map.")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-eval-rows", type=int, default=256)
    p.add_argument("--eval-steps", type=int, default=0,
                   help="Evaluate every N optimizer steps; 0 keeps the legacy schedule.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--token-learning-rate", type=float, default=2e-4)
    p.add_argument("--pass-b-loss-weight", type=float, default=3.0)
    p.add_argument("--pass-b-capture-loss-weight", type=float, default=1.0,
                   help="Weight labels serialized after a Pass-B token through <pat_end>.")
    p.add_argument("--testgen-repeats", type=int, default=8)
    p.add_argument(
        "--oversample-b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Duplicate rows whose target contains Pass B. Disable for matched gate ablations.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--resume-from-checkpoint", type=Path)
    p.add_argument("--teacher-adapter", type=Path,
                   help="Optional frozen adapter used to preserve the starting distribution.")
    p.add_argument("--kl-weight", type=float, default=0.0,
                   help="Teacher-to-student KL weight on supervised target positions.")
    p.add_argument("--launch", action="store_true")
    return p.parse_args()


class TaskDataset:
    def __init__(self, path, tokenizer, max_length, oversample_b=False,
                 testgen_repeats=1, max_rows=None):
        self.rows = [json.loads(x) for x in path.open(encoding="utf-8")]
        self.dropped_long_targets = sum(
            len(tokenizer.encode(x["target"], add_special_tokens=False)) > 768
            for x in self.rows
        )
        self.rows = [x for x in self.rows
                     if len(tokenizer.encode(x["target"], add_special_tokens=False)) <= 768]
        if max_rows is not None:
            # Preserve all task types in a deterministic strided sample.
            self.rows = self.rows[::max(1, len(self.rows) // max_rows)][:max_rows]
        testgen = [x for x in self.rows if x["task"] == "testgen"]
        self.rows += testgen * max(0, testgen_repeats - 1)
        if oversample_b:
            self.rows += [x for x in self.rows if x["target_has_pass_b"]]
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        target = self.tokenizer.encode(row["target"], add_special_tokens=False)
        eos = int(self.tokenizer.eos_token_id)
        # Keep as much target supervision as possible. A truncated target has no
        # EOS label, so the model is not taught to stop at an artificial cut.
        prompt_budget = max(32, self.max_length - min(len(target) + 1, 768))
        prompt = self._encode_prompt(row, prompt_budget)
        room = self.max_length - len(prompt)
        truncated = len(target) + 1 > room
        target = target[:room] if truncated else target + [eos]
        ids = prompt + target
        return {"input_ids": ids, "labels": [-100] * len(prompt) + target,
                "task": row["task"], "target_has_pass_b": row["target_has_pass_b"]}

    def _encode_prompt(self, row, budget):
        text = row["prompt"]
        if row["task"] == "completion":
            marker, footer = "### RECODE prefix\n", "\n### Continue\n"
            prefix, payload = text.split(marker, 1)
            payload = payload.rsplit(footer, 1)[0]
            fixed = self.tokenizer.encode(prefix + marker, add_special_tokens=False)
            ending = self.tokenizer.encode(footer, add_special_tokens=False)
            body = self.tokenizer.encode(payload, add_special_tokens=False)
            room = max(0, budget - len(fixed) - len(ending))
            tail = body[-room:] if room else []
            return (fixed + tail + ending)[:budget]
        prompt = self.tokenizer.encode(text, add_special_tokens=False)
        if len(prompt) <= budget:
            return prompt
        # Preserve both the task/problem opening and the output-mode marker at
        # the end. A one-sided cut can remove the instruction being trained.
        left = budget // 2
        return prompt[:left] + prompt[-(budget - left):]


def collate(tokenizer):
    import torch
    pad = int(tokenizer.pad_token_id)
    def fn(rows):
        width = max(len(x["input_ids"]) for x in rows)
        ids, labels, masks = [], [], []
        for x in rows:
            n = width - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad] * n)
            labels.append(x["labels"] + [-100] * n)
            masks.append([1] * len(x["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(masks)}
    return fn


def load_model(a, tokenizer, init_map):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True,
        trust_remote_code=True, torch_dtype=dtype, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    embed = model.get_input_embeddings().weight
    with torch.no_grad():
        for row in init_map["items"]:
            tid = int(tokenizer.convert_tokens_to_ids(row["token"]))
            embed[tid].copy_(embed[[int(x) for x in row["base_token_ids"]]].mean(dim=0))
    model = PeftModel.from_pretrained(model, str(a.adapter), is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    return model


def main():
    a = args()
    from transformers import AutoTokenizer
    if a.tokenizer_path:
        tokenizer_path = a.tokenizer_path
        init_map = (json.loads(a.init_map_path.read_text())
                    if a.init_map_path else {"items": []})
    elif a.arm == "raw":
        tokenizer_path = a.model
        init_map = {"items": []}
    elif a.arm == "pass_a":
        tokenizer_path = a.assets / "tokenizer_pass_a"
        init_map = json.loads((a.assets / "special_init_map_pass_a.json").read_text())
    else:
        tokenizer_path = a.assets / "tokenizer"
        init_map = json.loads((a.assets / "special_init_map.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    b_ids = {int(tokenizer.convert_tokens_to_ids(x["token"])) for x in init_map["items"] if x["kind"] == "pass_b"}
    train = TaskDataset(a.data / "train.jsonl", tokenizer, a.max_length, oversample_b=a.oversample_b,
                        testgen_repeats=a.testgen_repeats)
    valid = TaskDataset(a.data / "validation.jsonl", tokenizer, a.max_length,
                        max_rows=a.max_eval_rows)
    contract = {"arm": a.arm, "model": str(a.model), "adapter": str(a.adapter), "data": str(a.data),
        "train_rows_after_b_oversampling": len(train), "validation_rows": len(valid),
        "dropped_long_train_targets": train.dropped_long_targets,
        "dropped_long_validation_targets": valid.dropped_long_targets,
        "max_steps": a.max_steps, "max_length": a.max_length,
        "eval_steps": a.eval_steps or max(10, a.max_steps // 2),
        "effective_batch": a.batch_size * a.gradient_accumulation,
        "learning_rate": a.learning_rate, "token_learning_rate": a.token_learning_rate,
        "pass_b_loss_weight": a.pass_b_loss_weight, "pass_b_token_ids": len(b_ids),
        "pass_b_capture_loss_weight": a.pass_b_capture_loss_weight,
        "testgen_repeats": a.testgen_repeats,
        "oversample_b": a.oversample_b,
        "seed": a.seed,
        "save_steps": a.save_steps,
        "resume_from_checkpoint": str(a.resume_from_checkpoint) if a.resume_from_checkpoint else None,
        "teacher_adapter": str(a.teacher_adapter) if a.teacher_adapter else None,
        "kl_weight": a.kl_weight,
        "output": str(a.output)}
    print(json.dumps(contract, indent=2), flush=True)
    if not a.launch: return

    import torch
    import torch.nn.functional as F
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(a, tokenizer, init_map).cuda()
    teacher = None
    if a.teacher_adapter:
        if a.kl_weight <= 0:
            raise ValueError("--teacher-adapter requires a positive --kl-weight")
        teacher_args = argparse.Namespace(**vars(a))
        teacher_args.adapter = a.teacher_adapter
        teacher = load_model(teacher_args, tokenizer, init_map).cuda().eval()
        teacher.gradient_checkpointing_disable()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    b_ids_tensor = torch.tensor(sorted(b_ids), device="cuda")
    pat_end_id = int(tokenizer.convert_tokens_to_ids("<pat_end>"))

    def pass_b_capture_mask(input_ids, shifted_labels):
        """Mask labels after each Pass-B token through its closing marker."""
        mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            starts = torch.nonzero(
                torch.isin(input_ids[row], b_ids_tensor), as_tuple=False
            ).flatten().tolist()
            for index, start in enumerate(starts):
                # shifted_labels[:, i] is the input token at i + 1. Therefore
                # index `start` predicts the first capture after the pattern.
                endings = torch.nonzero(
                    input_ids[row, start + 1:] == pat_end_id, as_tuple=False
                ).flatten()
                if not len(endings):
                    continue
                end_input_position = start + 1 + int(endings[0])
                next_pattern_position = (
                    starts[index + 1] if index + 1 < len(starts) else input_ids.shape[1]
                )
                # Zero-capture patterns have no <pat_end>. Do not accidentally
                # consume the capture terminator belonging to a later pattern.
                if end_input_position >= next_pattern_position:
                    continue
                mask[row, start:end_input_position] = True
        return mask & shifted_labels.ne(-100)

    class WeightedTrainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            # PEFT exposes **kwargs in forward, which makes recent Transformers
            # assume custom loss already handles accumulated-item normalization.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            shifted = labels[:, 1:].contiguous()
            valid_mask = shifted.ne(-100)
            teacher_probabilities = None
            b_mask = torch.isin(shifted, b_ids_tensor) & valid_mask
            capture_mask = pass_b_capture_mask(inputs["input_ids"], shifted)
            preservation_mask = valid_mask & ~b_mask & ~capture_mask
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(**inputs).logits[:, :-1]
                    teacher_probabilities = F.softmax(
                        teacher_logits[preservation_mask].float(), dim=-1
                    )
                del teacher_logits
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1].contiguous()
            losses = F.cross_entropy(logits.view(-1, logits.size(-1)), shifted.view(-1),
                                     ignore_index=-100, reduction="none").view_as(shifted)
            if not model.training:
                # Validation curves compare policy surfaces with different
                # Pass-B densities, so report ordinary per-token NLL. The
                # weighted objective remains active only for optimization.
                loss = (losses * valid_mask).sum() / valid_mask.sum().clamp_min(1)
                return (loss, outputs) if return_outputs else loss
            weights = torch.ones_like(losses)
            weights[capture_mask] = a.pass_b_capture_loss_weight
            weights[b_mask] = a.pass_b_loss_weight
            loss = (losses * weights * valid_mask).sum() / (weights * valid_mask).sum().clamp_min(1)
            if teacher_probabilities is not None:
                student_log_probabilities = F.log_softmax(
                    logits[preservation_mask].float(), dim=-1
                )
                preservation_loss = F.kl_div(
                    student_log_probabilities,
                    teacher_probabilities,
                    reduction="batchmean",
                )
                loss = loss + a.kl_weight * preservation_loss
            return (loss, outputs) if return_outputs else loss

        def create_optimizer(self):
            if self.optimizer is not None: return self.optimizer
            token, other = [], []
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    (token if "trainable_tokens_delta" in name else other).append(parameter)
            groups = [{"params": other, "lr": a.learning_rate}]
            if token: groups.append({"params": token, "lr": a.token_learning_rate})
            self.optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
            return self.optimizer

    training_args = TrainingArguments(output_dir=str(a.output), per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=2, gradient_accumulation_steps=a.gradient_accumulation,
        max_steps=a.max_steps, learning_rate=a.learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=0.05, eval_strategy="steps",
        eval_steps=a.eval_steps or max(10, a.max_steps // 2),
        save_strategy="steps" if a.save_steps else "no",
        save_steps=a.save_steps if a.save_steps else 500,
        save_total_limit=2, logging_steps=5, bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), tf32=True, gradient_checkpointing=True,
        optim="adamw_torch_fused", report_to="none", seed=a.seed, data_seed=a.seed,
        remove_unused_columns=False)
    trainer = WeightedTrainer(model=model, args=training_args, train_dataset=train,
                              eval_dataset=valid, data_collator=collate(tokenizer))
    result = trainer.train(resume_from_checkpoint=(str(a.resume_from_checkpoint)
                                                    if a.resume_from_checkpoint else None))
    evaluation = trainer.evaluate()
    final = a.output / "final_adapter"; trainer.save_model(str(final)); tokenizer.save_pretrained(str(final))
    eval_loss = float(evaluation.get("eval_loss", math.nan))
    report = {**contract, "train_metrics": result.metrics, "eval_metrics": evaluation,
              "log_history": trainer.state.log_history,
              "eval_perplexity": math.exp(eval_loss) if math.isfinite(eval_loss) and eval_loss < 80 else None}
    (a.output / "run_contract.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__": main()

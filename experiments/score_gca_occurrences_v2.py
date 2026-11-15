#!/usr/bin/env python3
"""Score train occurrences with attention and counterfactual suffix NLL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


COMMON = "### Task: code completion\n### RECODE prefix\n"
FEATURES = (
    "raw:received_sum",
    "raw:received_normalized",
    "raw:future_to_span",
    "raw:outgoing_entropy",
    "delta:future_to_span",
    "delta:outgoing_entropy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--locked-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--suffix-tokens", type=int, default=48)
    return parser.parse_args()


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_documents(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source = row.get("source") or row.get("raw") or row.get("code")
            if not source:
                continue
            rows.append(
                {
                    "id": str(row.get("id") or digest(str(source))),
                    "problem_group": str(row.get("problem_group") or ""),
                    "source": str(source),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


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
        attn_implementation="eager",
    )
    base.resize_token_embeddings(len(tokenizer))
    return PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False).eval().cuda()


def codec(args: argparse.Namespace):
    sys.path.insert(0, str(args.core / "code/recode_method/src"))
    sys.path.insert(0, str(args.core / "code/recode_experiments/src"))
    from normcore_codec import CodecAssets, CodecConfig, PassAConfig, PassBConfig
    from recode_experiments.codec import aggressive_assets

    base = aggressive_assets()
    assets = CodecAssets(
        pattern_registry=args.registry,
        gca_policy=base.gca_policy,
        type_table=base.type_table,
        grammar_binary=base.grammar_binary,
        asset_lock=None,
    )
    pass_a = CodecConfig(
        pass_a=PassAConfig(mode="aggressive"),
        pass_b=PassBConfig(enabled=False),
        verify_assets=False,
    )
    pass_ab = CodecConfig(
        pass_a=PassAConfig(mode="aggressive"),
        pass_b=PassBConfig(enabled=True),
        verify_assets=False,
    )
    return assets, pass_a, pass_ab


def model_state(model, tokenizer, text: str, max_length: int):
    import torch

    encoded = tokenizer(
        COMMON + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = [(int(left), int(right)) for left, right in encoded.pop("offset_mapping")[0].tolist()]
    encoded = {key: value.cuda() for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, output_attentions=True, use_cache=False)
    layers = [layer[0].float().mean(dim=0) for layer in output.attentions[-4:]]
    return {
        "offsets": offsets,
        "attention": torch.stack(layers).mean(dim=0),
        "input_ids": encoded["input_ids"][0],
        "logits": output.logits[0],
    }


def span_features(state, start: int, end: int) -> dict | None:
    import torch

    offsets = state["offsets"]
    attention = state["attention"]
    shift = len(COMMON)
    indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start + shift and left < end + shift
    ]
    if not indices:
        return None
    span = torch.tensor(indices, device=attention.device)
    length = attention.shape[0]
    received_sum = attention[:, span].sum(dim=0).mean()
    denominators = torch.tensor(
        [max(1, length - index) for index in indices],
        dtype=attention.dtype,
        device=attention.device,
    )
    received_normalized = (attention[:, span].sum(dim=0) / denominators).mean()
    future_queries = list(range(indices[-1] + 1, min(length, indices[-1] + 33)))
    future_to_span = (
        attention[future_queries][:, span].sum(dim=-1).mean()
        if future_queries
        else torch.tensor(0.0, device=attention.device)
    )
    outgoing = attention[span]
    outgoing_entropy = -(outgoing * outgoing.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return {
        "received_sum": float(received_sum.cpu()),
        "received_normalized": float(received_normalized.cpu()),
        "future_to_span": float(future_to_span.cpu()),
        "outgoing_entropy": float(outgoing_entropy.cpu()),
    }


def suffix_losses(state, boundary: int, limit: int) -> tuple[list[int], list[float]] | None:
    """Return token ids and per-token NLL wholly to the right of a boundary."""
    import torch.nn.functional as functional

    absolute = len(COMMON) + boundary
    indices = [
        index
        for index, (left, right) in enumerate(state["offsets"])
        if left >= absolute and right > left
    ]
    if not indices:
        return None
    start = indices[0]
    if start == 0:
        return None
    end = min(start + limit, len(state["input_ids"]))
    if end - start < 8:
        return None
    ids = state["input_ids"][start:end]
    logits = state["logits"][start - 1 : end - 1].float()
    losses = functional.cross_entropy(logits, ids, reduction="none")
    return ids.tolist(), losses.tolist()


def aligned_suffix_nll(
    raw_suffix: tuple[list[int], list[float]] | None,
    compressed_suffix: tuple[list[int], list[float]] | None,
) -> tuple[int, float, float] | None:
    """Align the common suffix after boundary-sensitive BPE tokenization."""
    if raw_suffix is None or compressed_suffix is None:
        return None
    raw_ids, raw_losses = raw_suffix
    compressed_ids, compressed_losses = compressed_suffix
    best = None
    for raw_drop in range(min(5, len(raw_ids))):
        for compressed_drop in range(min(5, len(compressed_ids))):
            length = 0
            while (
                raw_drop + length < len(raw_ids)
                and compressed_drop + length < len(compressed_ids)
                and raw_ids[raw_drop + length] == compressed_ids[compressed_drop + length]
            ):
                length += 1
            if length >= 8:
                candidate = (length, -(raw_drop + compressed_drop), raw_drop, compressed_drop)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return None
    length, _negative_drops, raw_drop, compressed_drop = best
    raw_nll = sum(raw_losses[raw_drop : raw_drop + length]) / length
    compressed_nll = sum(compressed_losses[compressed_drop : compressed_drop + length]) / length
    return length, raw_nll, compressed_nll


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    missing = [row["goal"] for row in registry["patterns"] if row["goal"] not in tokenizer.get_vocab()]
    if missing:
        raise RuntimeError(f"{len(missing)} registry goals are not atomic tokenizer entries")
    locked = json.loads(args.locked_scorer.read_text(encoding="utf-8"))
    if tuple(locked["locked_before_internal_test"]["features"]) != FEATURES:
        raise RuntimeError("locked scorer feature contract differs from this implementation")
    means = locked["feature_means"]
    scales = locked["feature_scales"]
    weights = locked["weights"]

    model = load_model(args, tokenizer)
    assets, pass_a, pass_ab = codec(args)
    from normcore_codec import compress

    count = lambda value: len(tokenizer(value, add_special_tokens=False).input_ids)
    documents = load_documents(args.documents, args.max_documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decision_path = args.output.with_suffix(".jsonl")
    scored = direct_scored = suffix_mismatches = unscored = candidates_seen = errors = 0
    document_manifest = []
    with decision_path.open("w", encoding="utf-8") as handle:
        for ordinal, document in enumerate(documents):
            source = document["source"]
            surface = compress(source, config=pass_a, assets=assets).code
            candidates = []

            def collect(match: dict) -> bool:
                raw_span = str(match["_text"])[int(match["start"]) : int(match["end"])]
                saving = count(raw_span) - count(str(match["goal"]))
                if saving > 0:
                    candidates.append(
                        {
                            "pattern_id": str(match["id"]),
                            "start": int(match["start"]),
                            "end": int(match["end"]),
                            "goal": str(match["goal"]),
                            "token_saving": saving,
                        }
                    )
                return True

            compress(
                source,
                config=pass_ab,
                assets=assets,
                pass_b_skip_match=collect,
                pass_b_count_fn=count,
            )
            candidates_seen += len(candidates)
            if not candidates:
                document_manifest.append(
                    {"id": document["id"], "source_sha256": digest(source), "candidates": 0, "scored": 0}
                )
                continue
            try:
                raw_state = model_state(model, tokenizer, surface, args.max_length)
            except Exception:
                errors += len(candidates)
                continue
            document_scored = 0
            for candidate in candidates:
                raw_features = span_features(raw_state, candidate["start"], candidate["end"])
                if raw_features is None:
                    unscored += 1
                    continue
                compressed = surface[: candidate["start"]] + candidate["goal"] + surface[candidate["end"] :]
                try:
                    compressed_state = model_state(model, tokenizer, compressed, args.max_length)
                    compressed_features = span_features(
                        compressed_state,
                        candidate["start"],
                        candidate["start"] + len(candidate["goal"]),
                    )
                    if compressed_features is None:
                        unscored += 1
                        continue
                    feature_values = {
                        "raw:received_sum": raw_features["received_sum"],
                        "raw:received_normalized": raw_features["received_normalized"],
                        "raw:future_to_span": raw_features["future_to_span"],
                        "raw:outgoing_entropy": raw_features["outgoing_entropy"],
                        "delta:future_to_span": compressed_features["future_to_span"] - raw_features["future_to_span"],
                        "delta:outgoing_entropy": compressed_features["outgoing_entropy"] - raw_features["outgoing_entropy"],
                    }
                    prediction = float(weights["intercept"]) + sum(
                        float(weights[name]) * (feature_values[name] - float(means[name])) / float(scales[name])
                        for name in FEATURES
                    )
                    raw_suffix = suffix_losses(raw_state, candidate["end"], args.suffix_tokens)
                    compressed_suffix = suffix_losses(
                        compressed_state,
                        candidate["start"] + len(candidate["goal"]),
                        args.suffix_tokens,
                    )
                    direct = {}
                    aligned = aligned_suffix_nll(raw_suffix, compressed_suffix)
                    if aligned is not None:
                        suffix_tokens, raw_nll, compressed_nll = aligned
                        direct = {
                            "suffix_tokens": suffix_tokens,
                            "raw_suffix_nll": raw_nll,
                            "compressed_suffix_nll": compressed_nll,
                            "damage_delta_nll": compressed_nll - raw_nll,
                        }
                        direct_scored += 1
                    else:
                        suffix_mismatches += 1
                    row = {
                        "document_id": document["id"],
                        "problem_group": document["problem_group"],
                        "source_sha256": digest(source),
                        **candidate,
                        "predicted_damage": prediction,
                        "features": feature_values,
                        **direct,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    scored += 1
                    document_scored += 1
                except Exception:
                    errors += 1
            document_manifest.append(
                {
                    "id": document["id"],
                    "source_sha256": digest(source),
                    "candidates": len(candidates),
                    "scored": document_scored,
                }
            )
            if (ordinal + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "documents": ordinal + 1,
                            "candidates": candidates_seen,
                            "scored": scored,
                            "unscored": unscored,
                            "direct_scored": direct_scored,
                            "suffix_mismatches": suffix_mismatches,
                            "errors": errors,
                        }
                    ),
                    flush=True,
                )

    summary = {
        "schema_version": 1,
        "definition": (
            "locked attention prediction and direct counterfactual suffix-NLL damage "
            "per legal positive-saving occurrence"
        ),
        "documents": len(documents),
        "candidates": candidates_seen,
        "scored": scored,
        "direct_scored": direct_scored,
        "suffix_mismatches": suffix_mismatches,
        "unscored": unscored,
        "errors": errors,
        "features": list(FEATURES),
        "registry_sha256": sha256_bytes(args.registry),
        "documents_sha256": sha256_bytes(args.documents),
        "locked_scorer_sha256": sha256_bytes(args.locked_scorer),
        "decisions": str(decision_path),
        "decisions_sha256": sha256_bytes(decision_path),
        "document_manifest": document_manifest,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "document_manifest"}, indent=2))


if __name__ == "__main__":
    main()

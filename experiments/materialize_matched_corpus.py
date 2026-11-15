#!/usr/bin/env python3
"""Materialize Conditional-Attention and uniform occurrence controls at matched realized tokens."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.25)
    parser.add_argument(
        "--score-field",
        choices=("predicted_damage", "damage_delta_nll"),
        default="predicted_damage",
    )
    parser.add_argument("--random-budget-scale", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=314159)
    parser.add_argument("--max-documents", type=int)
    return parser.parse_args()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def key(document_id: str, match: dict) -> tuple:
    return (
        document_id,
        str(match.get("id", match.get("pattern_id"))),
        int(match["start"]),
        int(match["end"]),
        str(match["goal"]),
    )


def choose_to_saving(rows: list[dict], score, target: int) -> set[tuple]:
    selected = set()
    saving = 0
    for row in sorted(rows, key=score):
        candidate = int(row["token_saving"])
        if saving + candidate > target:
            continue
        selected.add(key(row["document_id"], row))
        saving += candidate
        if saving == target:
            break
    return selected


def make_codec(args: argparse.Namespace):
    sys.path.insert(0, str(args.core / "src"))
    from normcore_codec import CodecAssets, CodecConfig, NormalizationConfig, ShorthandingConfig
    base = CodecAssets.from_root(args.core / "assets")
    assets = CodecAssets(
        pattern_registry=args.registry,
        conditional_policy=base.conditional_policy,
        type_table=base.type_table,
        grammar_binary=base.grammar_binary,
        asset_lock=None,
    )
    normalization_config = CodecConfig(
        normalization=NormalizationConfig(mode="aggressive"),
        shorthanding=ShorthandingConfig(enabled=False),
        verify_assets=False,
    )
    normcore_config = CodecConfig(
        normalization=NormalizationConfig(mode="aggressive"),
        shorthanding=ShorthandingConfig(enabled=True),
        verify_assets=False,
    )
    return assets, normalization_config, normcore_config


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    assets, normalization_config, normcore_config = make_codec(args)
    from normcore_codec import compress, decompress

    documents = [json.loads(line) for line in args.documents.open(encoding="utf-8")]
    if args.max_documents is not None:
        documents = documents[: args.max_documents]
    decision_rows = [json.loads(line) for line in args.decisions.open(encoding="utf-8")]
    registry_rows = json.loads(args.registry.read_text(encoding="utf-8")).get("patterns", [])
    shorthand_goals = {
        str(row["goal"]).split("__H", 1)[0].rstrip(";")
        for row in registry_rows
        if row.get("goal")
    }
    document_ids = {str(row["id"]) for row in documents}
    decision_rows = [row for row in decision_rows if row["document_id"] in document_ids]
    decision_rows = [row for row in decision_rows if args.score_field in row]
    if {row["document_id"] for row in decision_rows} - document_ids:
        raise RuntimeError("decision manifest contains unknown documents")
    available_saving = sum(int(row["token_saving"]) for row in decision_rows)
    nominal_target = round(available_saving * args.budget_fraction)
    conditional_selected = choose_to_saving(
        decision_rows,
        lambda row: (
            float(row[args.score_field]),
            digest(json.dumps(row, sort_keys=True)),
        ),
        nominal_target,
    )
    rng = random.Random(args.random_seed)
    salts = {key(row["document_id"], row): rng.random() for row in decision_rows}
    random_selected = choose_to_saving(
        decision_rows,
        lambda row: (salts[key(row["document_id"], row)], key(row["document_id"], row)),
        round(nominal_target * args.random_budget_scale),
    )

    source_by_id = {str(row["id"]): str(row["source"]) for row in documents}

    def encoded(document_id: str, allowed: set[tuple]):
        def skip(match: dict) -> bool:
            return key(document_id, match) not in allowed

        return compress(
            source_by_id[document_id],
            config=normcore_config,
            assets=assets,
            shorthanding_skip_match=skip,
            shorthanding_count_fn=count,
        ).code

    normalization_tokens = {}
    conditional_tokens = {}
    conditional_text = {}
    raw_tokens = {}
    for index, document_id in enumerate(source_by_id):
        source = source_by_id[document_id]
        normalization_surface = compress(source, config=normalization_config, assets=assets).code
        conditional_surface = encoded(document_id, conditional_selected)
        raw_tokens[document_id] = count(source)
        normalization_tokens[document_id] = count(normalization_surface)
        conditional_tokens[document_id] = count(conditional_surface)
        conditional_text[document_id] = conditional_surface
        if (index + 1) % 500 == 0:
            print(json.dumps({"phase": "conditional", "documents": index + 1}), flush=True)
    outputs = {
        "conditional": (args.output / "train.conditional.jsonl").open("w", encoding="utf-8"),
        "random": (args.output / "train.random.jsonl").open("w", encoding="utf-8"),
    }
    audits = {
        "documents": len(documents),
        "raw_tokens": sum(raw_tokens.values()),
        "normalization_tokens": sum(normalization_tokens.values()),
        "conditional_tokens": 0,
        "random_tokens": 0,
        "conditional_shorthanding_documents": 0,
        "random_shorthanding_documents": 0,
        "conditional_ast_roundtrip": 0,
        "random_ast_roundtrip": 0,
    }
    try:
        for index, document in enumerate(documents):
            document_id = str(document["id"])
            source = source_by_id[document_id]
            surfaces = {
                "conditional": conditional_text[document_id],
                "random": encoded(document_id, random_selected),
            }
            source_ast = ast.dump(ast.parse(source), include_attributes=False)
            for representation, surface in surfaces.items():
                restored = decompress(surface, config=normcore_config, assets=assets)
                valid = ast.dump(ast.parse(restored), include_attributes=False) == source_ast
                if not valid:
                    raise RuntimeError(f"AST mismatch: {representation}/{document_id}")
                tokens = count(surface)
                audits[f"{representation}_tokens"] += tokens
                has_shorthands = any(goal in surface for goal in shorthand_goals)
                audits[f"{representation}_shorthanding_documents"] += int(has_shorthands)
                audits[f"{representation}_ast_roundtrip"] += 1
                outputs[representation].write(
                    json.dumps(
                        {
                            "id": document_id,
                            "problem_group": str(document.get("problem_group") or ""),
                            "source_sha256": digest(source),
                            "text": surface,
                            "tokens": tokens,
                            "has_shorthanding": has_shorthands,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if (index + 1) % 500 == 0:
                print(json.dumps({"phase": "write", "documents": index + 1}), flush=True)
    finally:
        for handle in outputs.values():
            handle.close()

    manifest = {
        "schema_version": 1,
        "definition": "Conditional Attention versus uniform occurrence selection with matched realized train tokens",
        "registry": str(args.registry),
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "documents": str(args.documents),
        "documents_sha256": hashlib.sha256(args.documents.read_bytes()).hexdigest(),
        "decisions": str(args.decisions),
        "decisions_sha256": hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
        "budget_fraction": args.budget_fraction,
        "score_field": args.score_field,
        "random_budget_scale": args.random_budget_scale,
        "available_candidate_saving": available_saving,
        "nominal_target_saving": nominal_target,
        "conditional_allowed_occurrences": len(conditional_selected),
        "random_seed": args.random_seed,
        "random_allowed_occurrences": len(random_selected),
        "matching_rule": "same summed positive tokenizer saving before non-overlap DP",
        "realized_token_difference_random_minus_conditional": audits["random_tokens"] - audits["conditional_tokens"],
        "audits": audits,
    }
    for representation in ("conditional", "random"):
        path = args.output / f"train.{representation}.jsonl"
        manifest[f"{representation}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

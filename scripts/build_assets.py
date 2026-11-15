#!/usr/bin/env python3
"""Build the canonical GCA policy and safe Pass B registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recode_codec.assets import sha256_file, validate_registry
from recode_codec.attention.gca import (
    build_gca_registry,
    build_policy_from_attention_cfg,
)

MASK_SENTINELS = (
    '"MASK"',
    "'MASK'",
    '"""MASK"""',
    "'''MASK'''",
    "#MASK",
    "__AARI_FMASK__",
    "__RECODE_FMASK__",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    config = load_json(root / "configs" / "canonical.json")
    assets = root / "assets"
    scores_path = assets / "attention_scores.json"
    base_registry_path = assets / "registry_base.json"
    scores = {k: float(v) for k, v in load_json(scores_path).items()}
    attention = config["attention"]

    policy = build_policy_from_attention_cfg(
        attention,
        scores=scores,
        scores_path="assets/attention_scores.json",
    )
    source = load_json(base_registry_path)
    source_rows = source.get("patterns") or []
    safe_rows = [
        row
        for row in source_rows
        if not any(sentinel in str(row.get("pattern", "")) for sentinel in MASK_SENTINELS)
    ]
    pass_b = attention["pass_b"]
    bands = (policy.meta.get("tiers") or {}).get("bands")
    rows, stats = build_gca_registry(
        safe_rows,
        scores,
        threshold=policy.threshold,
        hard_protect=bool(pass_b["hard_protect"]),
        formula=str(pass_b["formula"]),
        lambda_=float(pass_b["lambda"]),
        rank_by=str(pass_b["rank_by"]),
        top_k=int(pass_b["top_k"]),
        category_bands=bands,
    )

    policy_path = assets / "policy_canonical.json"
    registry_path = assets / "registry_canonical.json"
    aggressive_path = assets / "registry_aggressive_safe.json"
    policy_path.write_text(
        json.dumps(policy.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "patterns": rows,
                "meta": {
                    "method": "RECODE",
                    "source_registry": "assets/registry_base.json",
                    "filtered_literal_mask_patterns": len(source_rows) - len(safe_rows),
                    "build": stats,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    aggressive_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "patterns": safe_rows,
                "meta": {
                    "method": "model_blind_safe_baseline",
                    "source_registry": "assets/registry_base.json",
                    "filtered_literal_mask_patterns": len(source_rows) - len(safe_rows),
                    "candidate_count": len(safe_rows),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_registry(registry_path)
    validate_registry(aggressive_path)

    files = {}
    for name, path in {
        "pattern_registry": registry_path,
        "gca_policy": policy_path,
        "type_table": assets / "type_specials.json",
        "grammar_binary": assets / "spy_lang.so",
        "attention_scores": scores_path,
        "registry_base": base_registry_path,
        "aggressive_registry": aggressive_path,
    }.items():
        files[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
    lock = {
        "schema_version": 1,
        "method_contract": "docs/METHOD_CONTRACT.md",
        "provenance_status": {
            "attention_model_hash": "unresolved",
            "sample_ids": "unresolved",
            "aggregation": "layers-heads-query mean; category span mean",
        },
        "files": files,
    }
    (assets / "ASSET_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_patterns": len(source_rows),
                "safe_patterns": len(safe_rows),
                "canonical_patterns": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()

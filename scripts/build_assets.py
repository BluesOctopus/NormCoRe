#!/usr/bin/env python3
"""Build the canonical Conditional Attention policy and shorthand registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from normcore_codec.assets import sha256_file, validate_registry
from normcore_codec.attention.conditional import (
    build_conditional_registry,
    build_policy_from_attention_cfg,
)

MASK_SENTINELS = (
    '"MASK"',
    "'MASK'",
    '"""MASK"""',
    "'''MASK'''",
    "#MASK",
    "__AARI_FMASK__",
    "__NORMCORE_FMASK__",
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
    # The release schema names the two method stages directly.  The fallback
    # keeps the builder able to regenerate assets from an older local config.
    shorthanding_cfg = attention.get("shorthanding") or attention.get("pass_b")
    if shorthanding_cfg is None:
        raise KeyError("canonical attention config is missing shorthanding settings")
    bands = (policy.meta.get("tiers") or {}).get("bands")
    rows, stats = build_conditional_registry(
        safe_rows,
        scores,
        threshold=policy.threshold,
        hard_protect=bool(shorthanding_cfg["hard_protect"]),
        formula=str(shorthanding_cfg["formula"]),
        lambda_=float(shorthanding_cfg["lambda"]),
        rank_by=str(shorthanding_cfg["rank_by"]),
        top_k=int(shorthanding_cfg["top_k"]),
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
                    "method": "NormCoRe",
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
        "conditional_policy": policy_path,
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

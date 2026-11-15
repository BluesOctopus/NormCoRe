#!/usr/bin/env python3
"""Generate the canonical method manifest from verified local files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from normcore_codec.assets import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    core = root.parents[1]
    source_files = sorted((root / "src/normcore_codec").rglob("*"))
    tracked = [
        path
        for path in source_files
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    reports = {}
    for name in ("roundtrip_gate.json", "legacy_samesize_gate.json"):
        path = root / "reports" / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            reports[name] = {
                "sha256": sha256_file(path),
                "passed": payload.get("passed"),
                "humaneval_ast": (payload.get("humaneval") or {}).get("ast_roundtrip_rate"),
                "starcoder_ast": (payload.get("starcoder") or {}).get("ast_roundtrip_rate"),
            }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_root": str(root),
        "python_package": "normcore_codec",
        "method_contract": "code/recode_method/docs/METHOD_CONTRACT.md",
        "active_experiments": "code/recode_experiments",
        "source_baseline": (
            "code/legacy_evidence/aari_training_ready/codec "
            "(2026-07-14 working tree)"
        ),
        "source_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in tracked
        ],
        "assets_lock": {
            "path": "code/recode_method/assets/ASSET_LOCK.json",
            "sha256": sha256_file(root / "assets/ASSET_LOCK.json"),
        },
        "tests": reports,
        "legacy_policy": {
            "valuable_historical_trees_centralized": True,
            "archive_root": "code/legacy_evidence",
            "lineage_root": "lineage",
            "active_imports_from_legacy_forbidden": True,
            "duplicate_root_cleanup": "manifests/ROOT_CLEANUP_20260816.md",
        },
        "known_limits": [
            "StarCoder first-100 AST roundtrip is not 100%; see roundtrip_gate failures.",
            "Attention model hash and sample IDs remain unresolved.",
            "Tree-sitter plus repair decoder is not proven to be a formal GLR implementation.",
        ],
    }
    output = core / "manifests" / "CANONICAL_METHOD_MANIFEST.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

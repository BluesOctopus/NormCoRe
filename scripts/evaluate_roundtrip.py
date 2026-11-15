#!/usr/bin/env python3
"""Evaluate canonical AST roundtrip and physical token reduction."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from normcore_codec import CodecAssets, CodecConfig, compress, decompress
from normcore_codec.dynamic import load_registry
from normcore_codec.dynamic.wildcard import bare_goal_token
from normcore_codec.markers import all_special_tokens
from normcore_codec.structural.aggressive import idiom_special_tokens
from normcore_codec.structural.type_pack import load_type_table
from normcore_codec.tokens import get_count_fn


def ast_key(source: str) -> str:
    return ast.dump(ast.parse(source), annotate_fields=False)


def evaluate(path: Path, *, limit: int | None, assets: CodecAssets) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("samples") or []
    if limit is not None:
        rows = rows[:limit]
    registry = load_registry(assets.pattern_registry)
    pattern_tokens = {bare_goal_token(row.goal) for row in registry}
    type_tokens = {
        str(row["id"])
        for row in load_type_table(assets.type_table)
        if row.get("id")
    }
    count_fn = get_count_fn(
        specials=set(all_special_tokens())
        | set(idiom_special_tokens())
        | pattern_tokens
        | type_tokens
    )
    native_tokens = compressed_tokens = 0
    accepted = ast_ok = exec_ok = patterns = 0
    failures = []
    for index, row in enumerate(rows):
        source = row.get("source")
        sample_id = row.get("task_id") or row.get("id") or str(index)
        if not isinstance(source, str):
            continue
        try:
            source_key = ast_key(source)
            compile(source, f"<{sample_id}>", "exec")
        except Exception as exc:
            failures.append({"id": sample_id, "stage": "source_rejected", "error": type(exc).__name__})
            continue
        accepted += 1
        try:
            result = compress(source, config=CodecConfig.canonical(), assets=assets)
            restored = decompress(
                result.code,
                config=CodecConfig.canonical(),
                assets=assets,
            )
            native_tokens += count_fn(source)
            compressed_tokens += count_fn(result.code)
            patterns += int(result.stats["pass_b"].get("patterns_applied", 0))
            compile(restored, f"<{sample_id}:restored>", "exec")
            exec_ok += 1
            if ast_key(restored) == source_key:
                ast_ok += 1
            else:
                failures.append({"id": sample_id, "stage": "ast_mismatch"})
        except Exception as exc:
            failures.append(
                {"id": sample_id, "stage": "codec_error", "error": f"{type(exc).__name__}: {exc}"}
            )
    return {
        "dataset": str(path),
        "input_rows": len(rows),
        "accepted_python": accepted,
        "ast_ok": ast_ok,
        "ast_roundtrip_rate": ast_ok / accepted if accepted else 0.0,
        "exec_roundtrip_rate": exec_ok / accepted if accepted else 0.0,
        "native_tokens": native_tokens,
        "compressed_tokens": compressed_tokens,
        "token_reduction_pct": (
            (1 - compressed_tokens / native_tokens) * 100 if native_tokens else 0.0
        ),
        "patterns_applied": patterns,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--humaneval", type=Path, required=True,
                        help="JSON file containing a samples list with Python sources")
    parser.add_argument("--starcoder", type=Path, required=True,
                        help="JSON file containing a samples list with Python sources")
    parser.add_argument("--starcoder-limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=root / "reports" / "roundtrip_gate.json")
    args = parser.parse_args()

    assets = CodecAssets.from_root(root / "assets")
    results = {
        "contract": "AST equivalence; compile-only is reported separately",
        "humaneval": evaluate(args.humaneval, limit=None, assets=assets),
        "starcoder": evaluate(args.starcoder, limit=args.starcoder_limit, assets=assets),
        "thresholds": {"humaneval_ast": 1.0, "starcoder_ast_minimum": 0.68},
    }
    results["passed"] = (
        results["humaneval"]["ast_roundtrip_rate"] == 1.0
        and results["starcoder"]["ast_roundtrip_rate"] >= 0.68
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "humaneval" and k != "starcoder"}))
    print(
        json.dumps(
            {
                "humaneval_ast": results["humaneval"]["ast_roundtrip_rate"],
                "starcoder_ast": results["starcoder"]["ast_roundtrip_rate"],
            }
        )
    )
    raise SystemExit(0 if results["passed"] else 2)


if __name__ == "__main__":
    main()

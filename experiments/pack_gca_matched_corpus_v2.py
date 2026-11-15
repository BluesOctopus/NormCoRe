#!/usr/bin/env python3
"""Pack matched GCA/random dialect corpora with a problem-disjoint validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("gca", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--validation-group-fraction", type=float, default=0.1)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_validation_group(group: str, fraction: float) -> bool:
    value = int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) / float(16**16)
    return value < fraction


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    eos = int(tokenizer.eos_token_id)
    rows = {}
    identities = None
    for arm in ARMS:
        path = args.materialized / f"train.{arm}.jsonl"
        values = [json.loads(line) for line in path.open(encoding="utf-8")]
        current = [
            (row["id"], row["problem_group"], row["source_sha256"])
            for row in values
        ]
        if identities is None:
            identities = current
        elif identities != current:
            raise RuntimeError("GCA/random rows are not source-aligned")
        rows[arm] = values

    manifest = {
        "schema_version": 1,
        "definition": "same sources, problem-disjoint validation groups, fixed-length causal-LM blocks",
        "tokenizer": str(args.tokenizer),
        "tokenizer_vocab": len(tokenizer),
        "block_size": args.block_size,
        "validation_group_fraction": args.validation_group_fraction,
        "splits": {},
    }
    split_members = {
        split: [
            index
            for index, row in enumerate(rows["gca"])
            if is_validation_group(row["problem_group"], args.validation_group_fraction)
            == (split == "validation")
        ]
        for split in ("train", "validation")
    }
    train_groups = {rows["gca"][i]["problem_group"] for i in split_members["train"]}
    validation_groups = {rows["gca"][i]["problem_group"] for i in split_members["validation"]}
    if train_groups & validation_groups:
        raise RuntimeError("problem-group leakage")

    for split, members in split_members.items():
        manifest["splits"][split] = {
            "documents": len(members),
            "problem_groups": len(
                {rows["gca"][index]["problem_group"] for index in members}
            ),
            "arms": {},
        }
        for arm in ARMS:
            path = args.output / f"{split}.{arm}.i32"
            stream = []
            with path.open("wb") as handle:
                for index in members:
                    stream.extend(tokenizer.encode(rows[arm][index]["text"], add_special_tokens=False))
                    stream.append(eos)
                    if len(stream) >= 1_000_000:
                        np.asarray(stream, dtype=np.int32).tofile(handle)
                        stream.clear()
                if stream:
                    np.asarray(stream, dtype=np.int32).tofile(handle)
            tokens = path.stat().st_size // np.dtype(np.int32).itemsize
            manifest["splits"][split]["arms"][arm] = {
                "file": path.name,
                "sha256": digest(path),
                "tokens": tokens,
                "blocks": tokens // args.block_size,
                "remainder_tokens": tokens % args.block_size,
            }
    manifest["problem_group_overlap"] = len(train_groups & validation_groups)
    manifest["train_token_difference_random_minus_gca"] = (
        manifest["splits"]["train"]["arms"]["random"]["tokens"]
        - manifest["splits"]["train"]["arms"]["gca"]["tokens"]
    )
    (args.output / "pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

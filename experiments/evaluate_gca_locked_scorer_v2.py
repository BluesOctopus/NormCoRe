#!/usr/bin/env python3
"""Fit the locked GCA-v2 scorer on train+validation and evaluate internal test once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


FEATURES = (
    "raw:received_sum",
    "raw:received_normalized",
    "raw:future_to_span",
    "raw:outgoing_entropy",
    "delta:future_to_span",
    "delta:outgoing_entropy",
)
RIDGE = 20.0
BUDGET_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-trials", type=int, default=20000)
    return parser.parse_args()


def read(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload["records"]


def value(row: dict, name: str) -> float:
    namespace, key = name.split(":", 1)
    source = row["raw_attention"] if namespace == "raw" else row["attention_delta"]
    return float(source[key])


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[value(row, name) for name in FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([row["damage_delta_nll"] for row in rows], dtype=np.float64)
    return x, y


def choose(rows: list[dict], order: list[int], budget: int) -> dict:
    indices: list[int] = []
    saving = 0
    for index in order:
        candidate_saving = int(rows[index]["token_saving"])
        if saving + candidate_saving > budget:
            continue
        indices.append(index)
        saving += candidate_saving
        if saving == budget:
            break
    damage = [float(rows[index]["damage_delta_nll"]) for index in indices]
    return {
        "occurrences": len(indices),
        "token_saving": saving,
        "total_damage": float(sum(damage)),
        "mean_damage": float(np.mean(damage)),
        "positive_damage_fraction": float(np.mean(np.asarray(damage) > 0)),
        "selection_keys": [
            {
                "document_id": rows[index]["document_id"],
                "pattern_id": rows[index]["pattern_id"],
                "start": rows[index]["start"],
                "end": rows[index]["end"],
            }
            for index in indices
        ],
    }


def random_distribution(rows: list[dict], budget: int, trials: int) -> dict:
    rng = random.Random(20260915)
    indices = list(range(len(rows)))
    totals = []
    for _ in range(trials):
        rng.shuffle(indices)
        totals.append(choose(rows, indices, budget)["total_damage"])
    totals.sort()
    return {
        "trials": trials,
        "mean": float(np.mean(totals)),
        "std": float(np.std(totals)),
        "p025": totals[math.floor(0.025 * (trials - 1))],
        "p500": totals[math.floor(0.500 * (trials - 1))],
        "p975": totals[math.floor(0.975 * (trials - 1))],
    }


def main() -> None:
    args = parse_args()
    train_payload, train = read(args.train)
    validation_payload, validation = read(args.validation)
    test_payload, test = read(args.test)
    groups = {
        "train": {row["problem_group"] for row in train},
        "validation": {row["problem_group"] for row in validation},
        "test": {row["problem_group"] for row in test},
    }
    overlaps = {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_test": len(groups["train"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"problem-group leakage: {overlaps}")

    fit = train + validation
    fit_x, fit_y = arrays(fit)
    test_x, test_y = arrays(test)
    means = fit_x.mean(axis=0)
    scales = fit_x.std(axis=0)
    scales[scales == 0] = 1.0
    fit_z = (fit_x - means) / scales
    test_z = (test_x - means) / scales
    design = np.column_stack([np.ones(len(fit_z)), fit_z])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ fit_y)
    predictions = np.column_stack([np.ones(len(test_z)), test_z]) @ weights

    available = sum(int(row["token_saving"]) for row in test)
    budget = round(available * BUDGET_FRACTION)
    selected = choose(test, list(np.argsort(predictions, kind="mergesort")), budget)
    oracle = choose(test, list(np.argsort(test_y, kind="mergesort")), budget)
    random_result = random_distribution(test, budget, args.random_trials)
    payload = {
        "schema_version": 1,
        "locked_before_internal_test": {
            "features": list(FEATURES),
            "ridge": RIDGE,
            "budget_fraction": BUDGET_FRACTION,
            "selection_rule": "compress occurrences in ascending predicted NLL damage until token budget",
        },
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("train", args.train),
                ("validation", args.validation),
                ("test", args.test),
            )
        },
        "registry_sha256": test_payload["registry_sha256"],
        "problem_group_overlap": overlaps,
        "fit_records": len(fit),
        "test_records": len(test),
        "feature_means": dict(zip(FEATURES, means.tolist())),
        "feature_scales": dict(zip(FEATURES, scales.tolist())),
        "weights": {
            "intercept": float(weights[0]),
            **dict(zip(FEATURES, weights[1:].tolist())),
        },
        "test_available_token_saving": available,
        "test_budget": budget,
        "gca": selected,
        "random": random_result,
        "oracle": oracle,
        "gca_minus_random_mean_total_damage": selected["total_damage"] - random_result["mean"],
        "gca_random_z": (selected["total_damage"] - random_result["mean"]) / random_result["std"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "problem_group_overlap",
        "fit_records",
        "test_records",
        "test_available_token_saving",
        "test_budget",
        "gca",
        "random",
        "oracle",
        "gca_minus_random_mean_total_damage",
        "gca_random_z",
    ) if key in payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

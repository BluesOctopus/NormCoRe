#!/usr/bin/env python3
"""Summarize GCA-v2 convergence across the canonical matched seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def curve(path):
    contract = json.loads(path.read_text())
    points = {}
    for row in contract["log_history"]:
        if "eval_loss" in row:
            points[int(row["step"])] = float(row["eval_loss"])
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--multi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "2718": {
            arm: args.main_root / "runs" / arm / "run_contract.json"
            for arm in ("gca", "random")
        }
    }
    for seed in (7, 123):
        paths[str(seed)] = {
            arm: args.multi_root / f"seed_{seed}" / arm / "run_contract.json"
            for arm in ("gca", "random")
        }
    curves = {
        seed: {arm: curve(path) for arm, path in arms.items()}
        for seed, arms in paths.items()
    }
    loss_matrix = json.loads((args.main_root / "loss_matrix_validation.json").read_text())
    for seed in curves:
        for arm in ("gca", "random"):
            curves[seed][arm][0] = float(loss_matrix["models"]["initial"][arm]["nll"])
    steps = sorted(set.intersection(*[
        set(curves[seed][arm]) for seed in curves for arm in ("gca", "random")
    ]))
    summary = {
        "definition": "matched GCA-v2 versus Random convergence across seeds",
        "threshold_nll": 1.0,
        "steps": steps,
        "seeds": {},
        "mean_curves": {"gca": [], "random": []},
    }
    for seed, arms in curves.items():
        area = {
            arm: float(np.trapz([arms[arm][step] for step in steps], steps))
            for arm in ("gca", "random")
        }
        threshold_step = {
            arm: next((step for step in steps if arms[arm][step] < 1.0), None)
            for arm in ("gca", "random")
        }
        summary["seeds"][seed] = {
            "initial_gca_nll": arms["gca"][steps[0]],
            "initial_random_nll": arms["random"][steps[0]],
            "final_gca_nll": arms["gca"][steps[-1]],
            "final_random_nll": arms["random"][steps[-1]],
            "final_difference_gca_minus_random": arms["gca"][steps[-1]] - arms["random"][steps[-1]],
            "aulc_gca": area["gca"],
            "aulc_random": area["random"],
            "aulc_difference_gca_minus_random": area["gca"] - area["random"],
            "threshold_step_gca": threshold_step["gca"],
            "threshold_step_random": threshold_step["random"],
            "gca_better_at_all_checkpoints": all(arms["gca"][step] < arms["random"][step] for step in steps),
        }
    for arm in ("gca", "random"):
        for step in steps:
            values = np.asarray([curves[seed][arm][step] for seed in curves])
            summary["mean_curves"][arm].append({
                "step": step,
                "mean_nll": float(values.mean()),
                "std_nll": float(values.std(ddof=1)),
                "mean_ppl": math.exp(float(values.mean())),
            })
    final_differences = np.asarray([
        row["final_difference_gca_minus_random"] for row in summary["seeds"].values()
    ])
    area_differences = np.asarray([
        row["aulc_difference_gca_minus_random"] for row in summary["seeds"].values()
    ])
    summary["aggregate"] = {
        "seeds": len(final_differences),
        "mean_final_difference": float(final_differences.mean()),
        "std_final_difference": float(final_differences.std(ddof=1)),
        "gca_wins": int(np.count_nonzero(final_differences < 0)),
        "mean_aulc_difference": float(area_differences.mean()),
        "std_aulc_difference": float(area_differences.std(ddof=1)),
        "gca_aulc_wins": int(np.count_nonzero(area_differences < 0)),
        "gca_better_at_all_checkpoints_all_seeds": all(
            row["gca_better_at_all_checkpoints"] for row in summary["seeds"].values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm, label, color in (
        ("random", "Matched Random", "#c2410c"),
        ("gca", "Model-informed GCA", "#007f5f"),
    ):
        points = summary["mean_curves"][arm]
        x = np.asarray([point["step"] for point in points])
        mean = np.asarray([point["mean_nll"] for point in points])
        std = np.asarray([point["std_nll"] for point in points])
        ax.plot(x, mean, marker="o", markersize=3.5, linewidth=2, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax.axhline(1.0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Policy-matched validation NLL")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output.parent / "gca_v2_multiseed_convergence.pdf", bbox_inches="tight")
    fig.savefig(
        args.output.parent / "gca_v2_multiseed_convergence.png",
        dpi=220,
        bbox_inches="tight",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

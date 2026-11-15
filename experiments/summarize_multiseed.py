#!/usr/bin/env python3
"""Summarize Conditional-Attention convergence across the canonical matched seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
    import numpy as np
    paths = {
        "2718": {
            surface: args.main_root / "runs" / surface / "run_contract.json"
            for surface in ("conditional", "random")
        }
    }
    for seed in (7, 123):
        paths[str(seed)] = {
            surface: args.multi_root / f"seed_{seed}" / surface / "run_contract.json"
            for surface in ("conditional", "random")
        }
    curves = {
        seed: {surface: curve(path) for surface, path in surfaces.items()}
        for seed, surfaces in paths.items()
    }
    loss_matrix = json.loads((args.main_root / "loss_matrix_validation.json").read_text())
    for seed in curves:
        for surface in ("conditional", "random"):
            curves[seed][surface][0] = float(loss_matrix["models"]["initial"][surface]["nll"])
    steps = sorted(set.intersection(*[
        set(curves[seed][surface]) for seed in curves for surface in ("conditional", "random")
    ]))
    summary = {
        "definition": "matched Conditional-Attention versus Random convergence across seeds",
        "threshold_nll": 1.0,
        "steps": steps,
        "seeds": {},
        "mean_curves": {"conditional": [], "random": []},
    }
    for seed, surfaces in curves.items():
        area = {
            surface: float(np.trapz([surfaces[surface][step] for step in steps], steps))
            for surface in ("conditional", "random")
        }
        threshold_step = {
            surface: next((step for step in steps if surfaces[surface][step] < 1.0), None)
            for surface in ("conditional", "random")
        }
        summary["seeds"][seed] = {
            "initial_conditional_nll": surfaces["conditional"][steps[0]],
            "initial_random_nll": surfaces["random"][steps[0]],
            "final_conditional_nll": surfaces["conditional"][steps[-1]],
            "final_random_nll": surfaces["random"][steps[-1]],
            "final_difference_conditional_minus_random": surfaces["conditional"][steps[-1]] - surfaces["random"][steps[-1]],
            "aulc_conditional": area["conditional"],
            "aulc_random": area["random"],
            "aulc_difference_conditional_minus_random": area["conditional"] - area["random"],
            "threshold_step_conditional": threshold_step["conditional"],
            "threshold_step_random": threshold_step["random"],
            "conditional_better_at_all_checkpoints": all(surfaces["conditional"][step] < surfaces["random"][step] for step in steps),
        }
    for surface in ("conditional", "random"):
        for step in steps:
            values = np.asarray([curves[seed][surface][step] for seed in curves])
            summary["mean_curves"][surface].append({
                "step": step,
                "mean_nll": float(values.mean()),
                "std_nll": float(values.std(ddof=1)),
                "mean_ppl": math.exp(float(values.mean())),
            })
    final_differences = np.asarray([
        row["final_difference_conditional_minus_random"] for row in summary["seeds"].values()
    ])
    area_differences = np.asarray([
        row["aulc_difference_conditional_minus_random"] for row in summary["seeds"].values()
    ])
    summary["aggregate"] = {
        "seeds": len(final_differences),
        "mean_final_difference": float(final_differences.mean()),
        "std_final_difference": float(final_differences.std(ddof=1)),
        "conditional_wins": int(np.count_nonzero(final_differences < 0)),
        "mean_aulc_difference": float(area_differences.mean()),
        "std_aulc_difference": float(area_differences.std(ddof=1)),
        "conditional_aulc_wins": int(np.count_nonzero(area_differences < 0)),
        "conditional_better_at_all_checkpoints_all_seeds": all(
            row["conditional_better_at_all_checkpoints"] for row in summary["seeds"].values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for surface, label, color in (
        ("random", "Matched Random", "#c2410c"),
        ("conditional", "Model-informed Conditional Attention", "#007f5f"),
    ):
        points = summary["mean_curves"][surface]
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
    fig.savefig(args.output.parent / "conditional_multiseed_convergence.pdf", bbox_inches="tight")
    fig.savefig(
        args.output.parent / "conditional_multiseed_convergence.png",
        dpi=220,
        bbox_inches="tight",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

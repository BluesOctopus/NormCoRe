#!/usr/bin/env python3
"""Summarize matched GCA-v2 and Random CPT convergence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def curve(contract):
    by_step = {}
    for row in contract["log_history"]:
        if "eval_loss" in row:
            step = int(row["step"])
            by_step[step] = {
                "step": step,
                "nll": float(row["eval_loss"]),
                "ppl": math.exp(float(row["eval_loss"])),
            }
    return [by_step[step] for step in sorted(by_step)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = {"curves": {}}
    for arm in ("gca", "random"):
        contract = json.loads((args.root / "runs" / arm / "run_contract.json").read_text())
        report["curves"][arm] = curve(contract)
    report["loss_matrix"] = json.loads((args.root / "loss_matrix_validation.json").read_text())
    initial = report["loss_matrix"]["models"].get("initial")
    if initial:
        for arm in ("gca", "random"):
            nll = float(initial[arm]["nll"])
            report["curves"][arm].insert(
                0, {"step": 0, "nll": nll, "ppl": math.exp(nll)}
            )
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm, label, color in (
        ("random", "Matched Random", "#c2410c"),
        ("gca", "Model-informed GCA", "#007f5f"),
    ):
        points = report["curves"][arm]
        ax.plot(
            [point["step"] for point in points],
            [point["nll"] for point in points],
            marker="o", markersize=3.5, linewidth=2, color=color, label=label,
        )
    ax.set_xlabel("Optimization steps")
    ax.set_ylabel("Policy-matched validation NLL (per token)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.root / "gca_v2_convergence.pdf", bbox_inches="tight")
    fig.savefig(args.root / "gca_v2_convergence.png", dpi=220, bbox_inches="tight")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

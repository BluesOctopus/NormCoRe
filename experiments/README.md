# Optional experiment entry points

These scripts are the compact research toolchain used for the training and
held-out scoring studies. They deliberately do not include datasets, model
weights, tokenizer caches, or run outputs. Supply those locations explicitly on
each script's command line and keep generated artifacts outside this release.

The codec itself is independent of these scripts. For a local smoke test, use
`scripts/evaluate_roundtrip.py` and the tests in the repository root.

The training scripts cover new-token initialization, task alignment, and
matched continual pretraining. The evaluation scripts cover packed loss
matrices, locked Conditional Attention scoring, occurrence scoring, and
convergence summaries. They are research utilities rather than a turnkey
benchmark distribution.

The canonical training comparison uses the two representation surfaces named
`conditional` and `random`; these names describe the selection policy, not
separate codec implementations. The task adapter uses `raw`, `normalized`,
and `normcore` as its representation names.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    def __init__(self, pack_dir: Path, split: str, surface: str, max_blocks: int | None = None):
        manifest = json.loads((pack_dir / "pack_manifest.json").read_text(encoding="utf-8"))
        split_meta = manifest["splits"][split]
        surfaces = split_meta.get("surfaces") or split_meta["arms"]
        meta = surfaces[surface]
        self.block_size = int(manifest["block_size"])
        self.n_blocks = int(meta["blocks"])
        if max_blocks is not None:
            self.n_blocks = min(self.n_blocks, int(max_blocks))
        if self.n_blocks < 1:
            raise ValueError(f"no complete blocks for {split}.{surface}")
        self.tokens = np.memmap(pack_dir / meta["file"], mode="r", dtype=np.int32)

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = (int(index) % self.n_blocks) * self.block_size
        ids = torch.from_numpy(np.array(self.tokens[start:start + self.block_size], dtype=np.int64))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def stack_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in features]) for key in features[0]}

"""Corpus loading (StarCoderData) for mining / benchmarks."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_CANDIDATES = (
    _ROOT / "entropy_tokenizer_v3/data/starcoder_1m_tokens.txt",
    _ROOT / "data/starcoder_1m_tokens.txt",
)
DEFAULT_CORPUS = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0])


def load_starcoder_samples(path: Path, n: int, offset: int = 0) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = [s for s in re.split(r"<\|sample_\d+\|>\n", text) if s.strip()]
    out: list[str] = []
    for s in parts[offset : offset + n]:
        lines = s.splitlines()
        if lines and lines[0].startswith("<"):
            lines = lines[1:]
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        out.append("\n".join(lines).strip())
    return out

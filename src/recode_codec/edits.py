from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SpanEdit:
    start: int
    end: int
    replacement: str
    priority: float = 0.0

    @classmethod
    def replace(cls, start: int, end: int, text: str, *, priority: float = 0.0) -> SpanEdit:
        return cls(start, end, text, priority)

    @classmethod
    def insert(cls, offset: int, text: str, *, priority: float = 0.0) -> SpanEdit:
        return cls(offset, offset, text, priority)

    @classmethod
    def delete(cls, start: int, end: int, *, priority: float = 0.0) -> SpanEdit:
        return cls(start, end, "", priority)


def apply_edits(source: str, edits: list[SpanEdit]) -> str:
    """Apply edits left-to-right on original byte offsets (non-overlapping)."""
    if not edits:
        return source
    # ABSOLUTE ORIGINAL SORT: only by start, then by end. No priority here.
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    parts: list[str] = []
    cursor = 0
    for edit in ordered:
        if edit.start < cursor:
            continue
        parts.append(source[cursor : edit.start])
        parts.append(edit.replacement)
        cursor = edit.end
    parts.append(source[cursor:])
    return "".join(parts)

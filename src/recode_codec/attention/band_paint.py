"""Parallel H/M/L band paint that tracks buffer edits for Pass-B gating.

Paint is a ``list[str]`` the same length as the current text buffer, each entry
in ``{"H","M","L","?"}``. Finer CST nodes overwrite coarser spans when painting.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from recode_codec.edits import SpanEdit


def paint_from_scores(source: str, scores: dict | None) -> list[str]:
    """Paint ``source`` from ``NodeScore`` spans (byte or char — use as given).

    NodeScore.start/end from ``score_nodes`` are LibCST ByteSpan offsets on the
    same ``source`` string used for scoring. For ASCII-heavy code they match
    char indices; we clamp to ``len(source)``.
    """
    n = len(source)
    paint = ["?"] * n
    if not scores:
        return paint
    # Coarse → fine: sort by span length descending so smaller nodes win
    items = sorted(
        scores.values(),
        key=lambda ns: (ns.end - ns.start),
        reverse=True,
    )
    for ns in items:
        s = max(0, min(n, int(ns.start)))
        e = max(0, min(n, int(ns.end)))
        if e <= s:
            continue
        band = ns.band if ns.band in ("H", "M", "L") else "?"
        for i in range(s, e):
            paint[i] = band
    return paint


def apply_edits_to_paint(paint: list[str], edits: list[SpanEdit]) -> list[str]:
    """Mirror ``apply_edits`` on the paint lane."""
    if not edits:
        return paint
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    parts: list[str] = []
    cursor = 0
    n = len(paint)
    for edit in ordered:
        if edit.start < cursor:
            continue
        s = max(0, min(n, edit.start))
        e = max(0, min(n, edit.end))
        parts.extend(paint[cursor:s])
        # replacement length in *text* equals len(edit.replacement)
        if edit.start == edit.end:
            # insert: inherit band from left neighbor, else right, else M
            inherit = paint[s - 1] if s > 0 else (paint[s] if s < n else "M")
            if inherit == "?":
                inherit = "M"
            parts.extend([inherit] * len(edit.replacement))
        else:
            # replace/delete: use majority of removed span for new chars
            span = paint[s:e] or ["M"]
            maj = Counter(span).most_common(1)[0][0]
            if maj == "?":
                maj = "M"
            parts.extend([maj] * len(edit.replacement))
        cursor = e
    parts.extend(paint[cursor:])
    return parts


def align_paint(old: str, new: str, paint: list[str]) -> list[str]:
    """Best-effort paint sync when ``old`` → ``new`` without explicit edits.

    Preserves order of non-trivial content; whitespace-only drops remove paint;
    inserted runs inherit the last emitted band.
    """
    if len(paint) != len(old):
        # length desync — fall back to all M
        return ["M"] * len(new)
    if old == new:
        return list(paint)
    out: list[str] = []
    i = j = 0
    last = "M"
    while i < len(old) or j < len(new):
        if i < len(old) and j < len(new) and old[i] == new[j]:
            b = paint[i] if paint[i] != "?" else last
            out.append(b)
            if b != "?":
                last = b
            i += 1
            j += 1
            continue
        # skip whitespace-only in old
        if i < len(old) and old[i].isspace() and (j >= len(new) or old[i] != new[j]):
            i += 1
            continue
        # inserted char(s) in new
        if j < len(new) and (i >= len(old) or old[i] != new[j]):
            # consume a run of inserts until sync or end
            while j < len(new) and (i >= len(old) or old[i] != new[j]):
                # if old has a different non-space char, advance old (substitution)
                if i < len(old) and not old[i].isspace() and not new[j].isspace() and old[i] != new[j]:
                    b = paint[i] if paint[i] != "?" else last
                    out.append(b)
                    last = b if b != "?" else last
                    i += 1
                    j += 1
                    break
                out.append(last)
                j += 1
            continue
        # fallback advance
        if i < len(old):
            i += 1
        elif j < len(new):
            out.append(last)
            j += 1
    # length fix
    if len(out) < len(new):
        out.extend([last] * (len(new) - len(out)))
    elif len(out) > len(new):
        out = out[: len(new)]
    return out


def majority_band(paint: list[str], start: int, end: int) -> str:
    """Majority H/M/L over ``[start, end)``; ``?`` ignored; default ``M``."""
    if end <= start or start >= len(paint):
        return "M"
    s = max(0, start)
    e = min(len(paint), end)
    counts: Counter[str] = Counter()
    for b in paint[s:e]:
        if b in ("H", "M", "L"):
            counts[b] += 1
    if not counts:
        return "M"
    return counts.most_common(1)[0][0]


def paint_replace_spans(
    paint: list[str],
    replacements: Iterable[tuple[int, int, int]],
) -> list[str]:
    """Apply ``(start, end, new_len)`` replacements right-to-left on paint."""
    out = list(paint)
    for start, end, new_len in sorted(replacements, key=lambda t: t[0], reverse=True):
        s = max(0, min(len(out), start))
        e = max(0, min(len(out), end))
        span = out[s:e] or ["M"]
        maj = Counter(b for b in span if b in ("H", "M", "L")).most_common(1)
        band = maj[0][0] if maj else "M"
        out = out[:s] + [band] * new_len + out[e:]
    return out

"""Opt-in reversible dictionary for repeated final Pass-B subtrees."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, ParentNodeProvider, PositionProvider

from normcore_codec.dynamic.miner import (
    _collect_expr_spans,
    _map_span_through_edits,
    _patched_and_edits,
    _slice_to_dialect,
)
from normcore_codec.markers import PATTERN_END
from normcore_codec.structural.rules import RuleContext


SUBTREE_BIND = "<subtree_bind>"
SUBTREE_REF_PREFIX = "<subtree_ref_"
SUBTREE_END = "<subtree_end>"


def subtree_reference_tokens(slots: int) -> set[str]:
    if slots <= 0:
        return set()
    return {SUBTREE_END} | {
        f"{SUBTREE_REF_PREFIX}{index}>" for index in range(slots)
    }


@dataclass(frozen=True)
class _Group:
    payload: str
    occurrences: tuple[tuple[int, int], ...]
    local_reward: int


def _all_occurrences(text: str, needle: str) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        out.append((index, index + len(needle)))
        start = index + len(needle)
    return tuple(out)


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in occupied)


def _collect_broad_spans(masked_source: str) -> set[tuple[int, int]]:
    """All non-trivial expressions and statements on the masked CST."""
    try:
        wrapper = cst.metadata.MetadataWrapper(cst.parse_module(masked_source))
    except Exception:
        return set()
    spans: set[tuple[int, int]] = set()

    class Collector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (
            ByteSpanPositionProvider,
            PositionProvider,
            ParentNodeProvider,
        )

        def __init__(self) -> None:
            super().__init__()
            self.ctx = RuleContext(masked_source, self)

        def on_visit(self, node: cst.CSTNode) -> bool:
            if isinstance(node, (cst.BaseExpression, cst.BaseStatement)):
                try:
                    start, end = self.ctx.span(node)
                except Exception:
                    return True
                if end - start >= 4:
                    spans.add((start, end))
            return True

    try:
        wrapper.visit(Collector())
    except Exception:
        return spans
    return spans


def _candidate_payloads(masked_source: str, mid: str, type_table_path) -> set[str]:
    try:
        patched, edits = _patched_and_edits(masked_source)
    except Exception:
        return set()
    payloads: set[str] = set()
    spans = set(_collect_expr_spans(masked_source)) | _collect_broad_spans(masked_source)
    for start, end in spans:
        try:
            mapped_start, mapped_end = _map_span_through_edits(start, end, edits)
            payload = _slice_to_dialect(
                patched[mapped_start:mapped_end],
                str(type_table_path) if type_table_path else None,
            )
        except Exception:
            continue
        if not payload or payload not in mid:
            continue
        if "MASK" in payload or PATTERN_END in payload:
            continue
        if (
            SUBTREE_BIND in payload
            or SUBTREE_REF_PREFIX in payload
            or SUBTREE_END in payload
        ):
            continue
        payloads.add(payload)
    return payloads


def _candidate_wire_windows(mid: str, *, max_units: int = 4) -> set[str]:
    """Repeated statement windows on the final Pass-B wire stream.

    Pass B can replace a complete statement with a goal token, or a wildcard
    goal plus ``<pat_end>``. Splitting only at layout/pattern boundaries keeps
    this arm structural while allowing a local reference to contain ordinary
    Pass-B shorthands. The older CST arm cannot see those post-match payloads.
    """
    if "_pat>" not in mid and PATTERN_END not in mid:
        return set()

    cuts = {0, len(mid)}
    for match in re.finditer(r"<[^<>\n]+>", mid):
        token = match.group(0)
        if token == "<block_start>":
            cuts.add(match.end())
        if token in {"<line_sep>", "<block_end>", PATTERN_END} or token.endswith("_pat>"):
            cuts.add(match.end())

    ordered = sorted(cuts)
    payloads: set[str] = set()
    for start_index in range(len(ordered) - 1):
        for end_index in range(start_index + 1, min(len(ordered), start_index + max_units + 1)):
            payload = mid[ordered[start_index] : ordered[end_index]]
            if len(payload) < 4:
                continue
            if "_pat>" not in payload and PATTERN_END not in payload:
                continue
            if "MASK" in payload:
                continue
            if (
                SUBTREE_BIND in payload
                or SUBTREE_REF_PREFIX in payload
                or SUBTREE_END in payload
            ):
                continue
            if mid.find(payload, ordered[end_index]) >= 0:
                payloads.add(payload)
    return payloads


def apply_subtree_pass(
    masked_source: str,
    mid: str,
    *,
    slots: int,
    count_fn: Callable[[str], int],
    type_table_path=None,
) -> tuple[str, dict]:
    """Bind profitable repeated structural spans and reference later copies.

    The wire format is self-describing and does not require the decoder to
    recover AST boundaries. The first ``<subtree_ref_i>payload<subtree_end>``
    defines an entry; later ``<subtree_ref_i>`` occurrences copy it. The
    dedicated end marker lets payloads contain ordinary ``<pat_end>`` tokens.
    """
    slots = max(0, int(slots))
    if slots == 0 or not mid:
        return mid, {"skipped": "disabled", "slots": slots}

    groups: list[_Group] = []
    candidate_payloads = _candidate_payloads(masked_source, mid, type_table_path)
    candidate_payloads.update(_candidate_wire_windows(mid))
    for payload in candidate_payloads:
        payload_cost = count_fn(payload)
        if payload_cost <= 1:
            continue
        occurrences = _all_occurrences(mid, payload)
        if len(occurrences) < 2:
            continue
        # The definition adds ref + end; each later copy costs one ref token.
        reward = (len(occurrences) - 1) * (payload_cost - 1) - 2
        if reward > 0:
            groups.append(_Group(payload, occurrences, reward))

    groups.sort(
        key=lambda group: (
            -group.local_reward,
            -len(group.occurrences),
            -len(group.payload),
            group.payload,
        )
    )

    selected: list[tuple[str, list[tuple[int, int]]]] = []
    occupied: list[tuple[int, int]] = []
    for group in groups:
        available = [span for span in group.occurrences if not _overlaps(span, occupied)]
        if len(available) < 2:
            continue
        payload_cost = count_fn(group.payload)
        realized_reward = (len(available) - 1) * (payload_cost - 1) - 2
        if realized_reward <= 0:
            continue
        selected.append((group.payload, available))
        occupied.extend(available)
        if len(selected) >= slots:
            break

    # Reference indices are definition order for deterministic output. The
    # index is explicit on both definitions and later uses.
    selected.sort(key=lambda row: row[1][0][0])
    edits: list[tuple[int, int, str]] = []
    reference_uses = 0
    for index, (payload, occurrences) in enumerate(selected):
        first = occurrences[0]
        ref = f"{SUBTREE_REF_PREFIX}{index}>"
        edits.append((first[0], first[1], ref + payload + SUBTREE_END))
        for start, end in occurrences[1:]:
            edits.append((start, end, ref))
            reference_uses += 1

    out = mid
    for start, end, replacement in sorted(edits, reverse=True):
        out = out[:start] + replacement + out[end:]

    original_tokens = count_fn(mid)
    compressed_tokens = count_fn(out)
    if compressed_tokens >= original_tokens:
        return mid, {
            "slots": slots,
            "candidate_groups": len(groups),
            "groups_bound": 0,
            "reference_uses": 0,
            "global_original_tokens": original_tokens,
            "global_compressed_tokens": original_tokens,
            "global_reward": 0,
            "global_fallback": True,
        }
    return out, {
        "slots": slots,
        "candidate_groups": len(groups),
        "groups_bound": len(selected),
        "reference_uses": reference_uses,
        "global_original_tokens": original_tokens,
        "global_compressed_tokens": compressed_tokens,
        "global_reward": original_tokens - compressed_tokens,
        "global_fallback": False,
    }


def invert_subtree_pass(text: str) -> str:
    if SUBTREE_BIND not in text and SUBTREE_REF_PREFIX not in text:
        return text
    table: dict[int, str] = {}
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        bind_at = text.find(SUBTREE_BIND, cursor)
        ref_at = text.find(SUBTREE_REF_PREFIX, cursor)
        positions = [position for position in (bind_at, ref_at) if position >= 0]
        if not positions:
            out.append(text[cursor:])
            break
        start = min(positions)
        out.append(text[cursor:start])
        if start == bind_at:
            payload_start = start + len(SUBTREE_BIND)
            payload_end = text.find(PATTERN_END, payload_start)
            if payload_end < 0:
                raise ValueError("unterminated Pass B subtree binding")
            payload = text[payload_start:payload_end]
            index = len(table)
            while index in table:
                index += 1
            table[index] = payload
            out.append(payload)
            cursor = payload_end + len(PATTERN_END)
            continue
        index_start = start + len(SUBTREE_REF_PREFIX)
        index_end = text.find(">", index_start)
        if index_end < 0:
            raise ValueError("unterminated Pass B subtree reference")
        try:
            index = int(text[index_start:index_end])
        except ValueError as exc:
            raise ValueError("invalid Pass B subtree reference") from exc
        if index in table:
            out.append(table[index])
            cursor = index_end + 1
            continue

        payload_start = index_end + 1
        payload_end = text.find(SUBTREE_END, payload_start)
        if payload_end < 0:
            raise ValueError("Pass B subtree reference precedes its binding")
        payload = text[payload_start:payload_end]
        if SUBTREE_REF_PREFIX in payload or SUBTREE_BIND in payload:
            raise ValueError("nested Pass B subtree binding is not supported")
        table[index] = payload
        out.append(payload)
        cursor = payload_end + len(SUBTREE_END)
    return "".join(out)

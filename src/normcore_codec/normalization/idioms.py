"""Aggressive reversible variants applied only on low-attention CST nodes.

True structural / idiom elisions (not type-dictionary packing):
- Builtin call heads: ``len(`` → ``<id_len>`` (expanded before spy decode)

Decode: ``unpack_aggressive_idioms`` before spy.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import libcst as cst

from normcore_codec.attention.node_score import NodeKey, NodeScore, node_key
from normcore_codec.reversible.edits import SpanEdit

if TYPE_CHECKING:
    from normcore_codec.normalization.rules import RuleContext

_IDIOM_FUNCS = frozenset(
    {
        "len",
        "range",
        "enumerate",
        "abs",
        "sorted",
        "sum",
        "min",
        "max",
        "zip",
        "map",
        "filter",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "list",
        "dict",
        "set",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "print",
        "open",
        "iter",
        "next",
        "any",
        "all",
        "reversed",
    }
)

IDIOM_MARKERS = {name: f"<id_{name}>" for name in sorted(_IDIOM_FUNCS)}


def idiom_special_tokens() -> list[str]:
    return list(IDIOM_MARKERS.values())


def _call_head_edit(ctx: "RuleContext", node: cst.Call) -> SpanEdit | None:
    func = node.func
    if not isinstance(func, cst.Name):
        return None
    name = func.value
    if name not in _IDIOM_FUNCS:
        return None
    fs, fe = ctx.span(func)
    idx = ctx.source.find("(", fe)
    if idx < 0 or idx > fe + 3:
        chunk = ctx.source[fe : fe + 4]
        m = re.match(r"\s*\(", chunk)
        if not m:
            return None
        idx = fe + m.end() - 1
    return SpanEdit.replace(fs, idx + 1, IDIOM_MARKERS[name])


def collect_aggressive_edits(
    ctx: "RuleContext",
    node: cst.CSTNode,
    scores: dict[NodeKey, NodeScore],
    *,
    apply_on: str = "L",
) -> list[SpanEdit]:
    try:
        start, end = ctx.span(node)
    except Exception:
        return []
    key = node_key(node, start, end)
    ns = scores.get(key)
    if ns is None or ns.band not in apply_on:
        return []
    edits: list[SpanEdit] = []
    if isinstance(node, cst.Call):
        e = _call_head_edit(ctx, node)
        if e is not None:
            edits.append(e)
    return edits


def collect_aggressive_edits_global(ctx: "RuleContext", node: cst.CSTNode) -> list[SpanEdit]:
    """Conditional Attention booster: apply idiom head edits on every matching Call (no instance gate)."""
    if not isinstance(node, cst.Call):
        return []
    e = _call_head_edit(ctx, node)
    return [e] if e is not None else []


def pack_residual_idioms(
    dialect: str,
    scores: dict[NodeKey, NodeScore] | None,
    *,
    apply_on: str = "L",
) -> tuple[str, int]:
    """Replace leftover ``name(`` in dialect, gated by how many L-band Calls exist.

    Runs **after** shorthanding so pattern matches on ``range(``/``len(`` are preserved.
    Quota per builtin = # of Call nodes with that name whose band ∈ apply_on.
    """
    if not scores:
        return dialect, 0
    quota: dict[str, int] = {}
    for ns in scores.values():
        if ns.kind != "Call" or ns.band not in apply_on:
            continue
        # recover func name from... we only have kind/span; use a soft global quota:
        # count L-band Calls as budget for *any* idiom replacement (conservative).
        quota["_any"] = quota.get("_any", 0) + 1

    budget = quota.get("_any", 0)
    if budget <= 0:
        return dialect, 0

    out = dialect
    n = 0
    # longest names first
    for name in sorted(_IDIOM_FUNCS, key=len, reverse=True):
        needle = f"{name}("
        marker = IDIOM_MARKERS[name]
        while budget > 0 and needle in out:
            out = out.replace(needle, marker, 1)
            budget -= 1
            n += 1
    return out, n


def unpack_aggressive_idioms(text: str) -> str:
    out = text
    for name in sorted(_IDIOM_FUNCS, key=len, reverse=True):
        out = out.replace(IDIOM_MARKERS[name], f"{name}(")
    return out

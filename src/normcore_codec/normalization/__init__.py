from __future__ import annotations

import libcst as cst
from pathlib import Path
from typing import Any, Callable

from libcst.metadata import ByteSpanPositionProvider, ParentNodeProvider, PositionProvider
from normcore_codec.reversible.edits import SpanEdit, apply_edits
from normcore_codec.normalization.rules import RuleContext, TRANSFORM_RULES, SPINE_FALLBACK_RULES
from normcore_codec.reversible.postprocess import postprocess_spy
from .type_pack import pack_types, load_type_table

def apply_normalization(
    source: str,
    band: str = "aggressive",
    type_table_path: str | Path | None = None,
    veto_fn: Callable[[int, int], bool] | None = None,
    **kwargs,
) -> tuple[str, dict[str, Any]]:
    """Apply the reversible normalization rules to one source buffer."""
    mask_records = kwargs.get("mask_records")

    # 建立源码映射 (i_masked -> i_original) 用于对齐 veto_fn
    if mask_records and veto_fn:
        from normcore_codec.reversible.literals import _UNMASK_RE
        pos_map = []
        last_m = 0
        last_o = 0
        it = iter(mask_records)
        for m in _UNMASK_RE.finditer(source):
            pre_len = m.start() - last_m
            pos_map.extend(range(last_o, last_o + pre_len))
            try:
                rec = next(it)
                m_len = m.end() - m.start()
                o_len = len(rec)
                pos_map.extend([last_o + pre_len] * m_len)
                last_o += (pre_len + o_len)
                last_m = m.end()
            except StopIteration: break
        pos_map.extend(range(last_o, last_o + (len(source) - last_m)))

        def aligned_veto(s, e):
            if s >= len(pos_map): return False
            # 精确查询：只对这一处换行的原始位置问 veto
            orig_i = pos_map[s]
            return veto_fn(orig_i, orig_i + 1)
        actual_veto = aligned_veto
    else:
        actual_veto = veto_fn

    try:
        wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
    except Exception:
        return source, {"error": "libcst_parse_failed"}

    active_types = TRANSFORM_RULES.keys()
    edits: list[SpanEdit] = []

    class StructuralVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider, PositionProvider, ParentNodeProvider)
        def __init__(self):
            super().__init__()
            self.ctx = RuleContext(source, self)
        def on_visit(self, node: cst.CSTNode) -> bool:
            rule = TRANSFORM_RULES.get(type(node))
            if rule:
                try:
                    res = rule(self.ctx, node)
                    if res: edits.extend(res)
                except Exception: pass
            return True

    wrapper.visit(StructuralVisitor())

    edit_veto_fn = kwargs.get("edit_veto_fn")
    n_edits = len(edits)
    n_edit_vetoed = 0
    if edit_veto_fn is not None:
        kept: list[SpanEdit] = []
        for edit in edits:
            drop = False
            try:
                drop = bool(edit_veto_fn(edit.start, edit.end))
            except TypeError:
                drop = False
            if drop:
                n_edit_vetoed += 1
            else:
                kept.append(edit)
        edits = kept

    elif actual_veto:
        # Conditional Attention protects high-contribution occurrences by
        # vetoing only edits whose source span overlaps a protected position.
        kept: list[SpanEdit] = []
        for edit in edits:
            protected = any(
                actual_veto(pos, pos + 1)
                for pos in range(edit.start, max(edit.start, edit.end))
            )
            if protected:
                n_edit_vetoed += 1
            else:
                kept.append(edit)
        edits = kept
        patched = apply_edits(source, edits)
        mid = postprocess_spy(patched, aggressive=True).rstrip("\n")
        hot_zones = []
        dialect_hotzones = {
            pos for pos in range(len(source)) if actual_veto(pos, pos + 1)
        }
        mode = "conditional"
    else:
        patched = apply_edits(source, edits)
        mid = postprocess_spy(patched, aggressive=True).rstrip("\n")
        hot_zones = []
        dialect_hotzones = set()
        mode = "frozen"

    # Final Type Packing
    type_table = load_type_table(type_table_path)
    mid, t_stats = pack_types(mid, table=type_table)
    if actual_veto:
        dialect_hotzones = {i for i, c in enumerate(mid) if c == "\n"}

    stats = {
        "band": band,
        "dialect_hotzones": dialect_hotzones,
        "anchors_protected": len([hz for hz in hot_zones if hz.get("handled")]),
        "anchor_offsets": [hz["start"] for hz in hot_zones if hz.get("handled")],
        "n_edits": n_edits,
        "n_edit_vetoed": n_edit_vetoed,
        "mode": mode,
    }
    stats.update(t_stats)
    return mid, stats

def invert_type_pack(text: str, type_table_path: str | Path | None = None) -> str:
    table = load_type_table(type_table_path)
    from .type_pack import unpack_types
    return unpack_types(text, table=table)

import libcst as cst
from pathlib import Path
from typing import Any, Callable, Union, Dict, List, Set, Optional
import re as std_re
import binascii

from libcst.metadata import ByteSpanPositionProvider, ParentNodeProvider, PositionProvider
from recode_codec.edits import SpanEdit, apply_edits
from recode_codec.structural.rules import RuleContext, TRANSFORM_RULES, SPINE_FALLBACK_RULES
from recode_codec.postprocess import postprocess_spy
from .type_pack import pack_types, load_type_table

def apply_structural_pass(
    source: str,
    band: str = "aggressive",
    type_table_path: str | Path | None = None,
    veto_fn: Callable[[int, int], bool] | None = None,
    **kwargs,
) -> tuple[str, dict[str, Any]]:
    """Pass A with GCA 2.0: Sovereign Scaffolding (Option B: Layout Anchors Only)."""
    mask_records = kwargs.get("mask_records")

    # 建立源码映射 (i_masked -> i_original) 用于对齐 veto_fn
    if mask_records and veto_fn:
        from recode_codec.mask import _UNMASK_RE
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

    if edit_veto_fn is not None:
        # DietCode: skip hot edits, then the frozen aggressive path.
        # Do not hang newlines (Option B).
        patched = apply_edits(source, edits)
        mid = postprocess_spy(patched, aggressive=True).rstrip("\n")
        hot_zones = []
        # DietCode: skip hot edits, then the frozen aggressive path.
        patched = apply_edits(source, edits)
        mid = postprocess_spy(patched, aggressive=True).rstrip("\n")
        hot_zones = []
        dialect_hotzones = set()
        mode = "dietcode_edit_veto"
    elif not actual_veto:
        if band == "gca":
            raise RuntimeError(
                "GCA Pass A requires veto_fn (attention hotzones). "
                "Missing hotzones must not silently fall back to aggressive. "
                "Use PassAConfig(mode='aggressive') for frozen dialect."
            )
        # PURE FROZEN PATH: leftover newlines are not GCA anchors.
        patched = apply_edits(source, edits)
        mid = postprocess_spy(patched, aggressive=True).rstrip("\n")
        hot_zones = []
        dialect_hotzones: set[int] = set()
        mode = "frozen"
    elif not kwargs.get("allow_legacy_option_b"):
        raise RuntimeError(
            "Option B newline GCA is closed. "
            "Use compress(..., keep_indices=) for GCA1, "
            "or allow_legacy_option_b=True to reproduce the failed arm."
        )
    else:
        # LEGACY Option B only. Do not use for new experiments.
        hot_zones = []
        i = 0
        while i < len(source):
            if source[i] == '\n' and actual_veto(i, i+1):
                # 跳过续行符防止 SyntaxError
                if i > 0 and source[i-1] == '\\':
                    i += 1
                    continue
                j = i + 1
                while j < len(source) and source[j] in ' \t': j += 1
                hot_zones.append({'start': i, 'end': j, 'text': source[i:j], 'handled': False})
                i = j
                continue
            i += 1

        # Collision & Merge: 只允许挂载到布局标记
        final_edits = []
        for e in sorted(edits, key=lambda e: (e.start, e.end)):
            replacement = e.replacement
            # 只对包含布局标记的规则进行热区挂载
            if any(m in replacement for m in ("<block_start>", "<block_end>", "<line_sep>")):
                covered_hz = [hz for hz in hot_zones if not hz['handled'] and not (e.end < hz['start'] or e.start > hz['end'])]
                if covered_hz:
                    markers = []
                    for hz in covered_hz:
                        h = binascii.hexlify(hz['text'].encode('utf-8')).decode('ascii')
                        markers.append(f"<gca_br>\x00GCA_{h}_SC\x00</gca_br>")
                        hz['handled'] = True
                    m_str = "".join(markers)
                    if "<block_start>" in replacement:
                        replacement = replacement.replace("<block_start>", f"<block_start>{m_str}")
                    elif "<block_end>" in replacement:
                        replacement = replacement.replace("<block_end>", f"<block_end>{m_str}")
                    elif "<line_sep>" in replacement:
                        replacement = replacement.replace("<line_sep>", f"<line_sep>{m_str}")

            final_edits.append(SpanEdit(e.start, e.end, replacement))

        # 孤儿丢掉：不再执行第二遍 micro-edit 循环
        patched = apply_edits(source, final_edits)
        mid = postprocess_spy(patched, aggressive=True).rstrip("\n")

        def restore_fn(m):
            h = m.group(1)
            return binascii.unhexlify(h).decode('utf-8')
        mid = std_re.sub(r'\x00GCA_([0-9a-f]+)_SC\x00', restore_fn, mid)
        mode = "sovereign_gca_v4_option_b"

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

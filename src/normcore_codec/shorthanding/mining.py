"""The only shorthanding miner: typed holes on the normalized surface.

Candidates are sliding windows over normalized ``mid_full`` units (compound heads
such as ``<for_stmt>…<block_start>``, adjacent payloads, ``<block_end>`` as a
crossable unit). Windows are substrings of the file, so they never need an
isolated re-projection filter. String/comment payloads become ``__S{k}__``
holes instead of fixed ``MASK`` text, which keeps ``unmask_literals``
positionally aligned.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from normcore_codec.tokens import get_count_fn
from normcore_codec.reversible.literals import mask_literals
from normcore_codec.normalization import apply_normalization
from normcore_codec.reversible.edits import SpanEdit, apply_edits
from normcore_codec.reversible.postprocess import postprocess_spy
from normcore_codec.normalization.rules import TRANSFORM_RULES, RuleContext
from normcore_codec.normalization.type_pack import load_type_table, pack_types

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, ParentNodeProvider, PositionProvider

PINNED = frozenset(
    {
        "self",
        "cls",
        "None",
        "True",
        "False",
        "len",
        "range",
        "print",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "bool",
        "super",
        "open",
        "enumerate",
        "zip",
        "sorted",
        "sum",
        "min",
        "max",
        "abs",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "pass",
        "return",
        "raise",
        "yield",
    }
)

MASK_PAT = re.compile(
    r'"""MASK"""|\'\'\'MASK\'\'\'|"MASK"|\'MASK\'|#MASK|__NORMCORE_FMASK__|__RECODE_FMASK__|__AARI_FMASK__'
)
IDENT_PAT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HOLE_PAT = re.compile(r"__[HS]\d+__")
_MASK_SENTINELS = (
    '"MASK"',
    "'MASK'",
    '"""MASK"""',
    "'''MASK'''",
    "#MASK",
    "__AARI_FMASK__",
    "__NORMCORE_FMASK__",
    "__RECODE_FMASK__",
)

# Compound heads on the normalized surface. TokenSugar's stmt_head is the span
# from the keyword through the colon; here that is `<for_stmt>…<block_start>`.
_HEAD_OPEN_RE = re.compile(
    r"(?:<async_keyword>)?"
    r"<(?:for|while|if|elif|else|def|class|try|except|finally|with|match|case|except_group)_stmt>"
)
_BOUNDARY_RE = re.compile(
    r"(?:<async_keyword>)?"
    r"<(?:for|while|if|elif|else|def|class|try|except|finally|with|match|case|except_group)_stmt>"
    r"|<block_end>|<line_sep>|<anchor_sep>"
)
_TRAIL_LAYOUT_RE = re.compile(r"(?:<line_sep>|<block_end>|<anchor_sep>)+$")
_LAYOUT_ONLY = frozenset({"<line_sep>", "<block_end>", "<anchor_sep>"})
_BLOCK_START = "<block_start>"
_SUFFIX_SEPS = frozenset({"<line_sep>", "<anchor_sep>"})
_MIN_CORE_LEN = 8
_MIN_EXPR_LEN = 6
_MAX_CORE_LEN = 400
_LEAF_EXPR = (cst.Name, cst.Integer, cst.Float, cst.Imaginary, cst.SimpleString)
_SKIP_CALL_NAMES = PINNED | {"Solution"}
_TYPE_PACK_ONLY = re.compile(r"^<t_[0-9a-f]+>$")


def generalize(code_masked: str) -> tuple[str, int]:
    """Turn mask placeholders and identifiers into typed holes.

    CRITICAL: S-holes (literals) are NEVER merged because different MASK
    instances represent different underlying literals. H-holes (identifiers)
    are merged by name.
    """
    cuts: list[tuple[int, int, str]] = []
    for match in MASK_PAT.finditer(code_masked):
        cuts.append((match.start(), match.end(), "S"))
    for match in IDENT_PAT.finditer(code_masked):
        start, end, name = match.start(), match.end(), match.group(0)
        if name in PINNED or name.startswith("__"):
            continue
        # Protect dialect boundaries
        if code_masked[max(0, start - 1) : start] == "<" or code_masked[end : end + 1] == ">":
            continue
        if code_masked[end : end + 1] == "(" or code_masked[max(0, start - 1) : start] == ".":
            continue
        cuts.append((start, end, "H"))

    if not cuts:
        return code_masked, 0

    cuts.sort()
    h_names: dict[str, int] = {} # H-holes share index by name
    n_holes = 0
    out: list[str] = []
    last = 0

    for start, end, kind in cuts:
        if start < last:
            continue

        out.append(code_masked[last:start])
        if kind == "H":
            name = code_masked[start:end]
            idx = h_names.setdefault(name, n_holes)
            if idx == n_holes:
                n_holes += 1
            out.append(f"__H{idx}__")
        else: # S-hole: Always unique index to prevent literal folding
            out.append(f"__S{n_holes}__")
            n_holes += 1
        last = end

    out.append(code_masked[last:])
    return "".join(out), n_holes


def strip_trailing_layout(text: str) -> str:
    """Drop trailing line/block seps so `x=0<line_sep>` and `x=0<block_end>` share a template.

    `<block_start>` stays: it terminates a compound head, not a layout suffix.
    """
    return _TRAIL_LAYOUT_RE.sub("", text)


def split_dialect_units(mid: str) -> list[str]:
    """Split a normalized buffer into units that concatenate back to `mid`.

    Delimiters stay attached: `<line_sep>` / `<anchor_sep>` as a suffix,
    compound heads as `<for_stmt>…<block_start>`, `<block_end>` as its own
    unit so adjacent windows can cross a block boundary. The dialect is not
    valid Python; this does not go through libcst.
    """
    units: list[str] = []
    i = 0
    n = len(mid)
    while i < n:
        head = _HEAD_OPEN_RE.match(mid, i)
        if head is not None:
            after = head.end()
            block_start = mid.find(_BLOCK_START, after)
            nxt = _BOUNDARY_RE.search(mid, after)
            if block_start != -1 and (nxt is None or block_start < nxt.start()):
                end = block_start + len(_BLOCK_START)
                units.append(mid[i:end])
                i = end
                continue
            if nxt is None:
                units.append(mid[i:])
                break
            token = nxt.group(0)
            if token in _SUFFIX_SEPS:
                units.append(mid[i : nxt.end()])
                i = nxt.end()
            else:
                if nxt.start() > i:
                    units.append(mid[i : nxt.start()])
                i = nxt.start()
            continue

        if mid.startswith("<block_end>", i):
            units.append("<block_end>")
            i += len("<block_end>")
            continue

        nxt = _BOUNDARY_RE.search(mid, i)
        if nxt is None:
            units.append(mid[i:])
            break
        token = nxt.group(0)
        if token in _SUFFIX_SEPS:
            units.append(mid[i : nxt.end()])
            i = nxt.end()
        else:
            if nxt.start() > i:
                units.append(mid[i : nxt.start()])
            i = nxt.start()
    return units


def _map_span_through_edits(start: int, end: int, edits: list[SpanEdit]) -> tuple[int, int]:
    """Map an original ``[start, end)`` span to the buffer after ``apply_edits``."""
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    out_pos = 0
    cursor = 0
    mapped_start: int | None = None
    mapped_end: int | None = None
    for edit in ordered:
        if edit.start < cursor:
            continue
        copied = edit.start - cursor
        if mapped_start is None and cursor <= start <= edit.start:
            mapped_start = out_pos + (start - cursor)
        if mapped_end is None and cursor <= end <= edit.start:
            mapped_end = out_pos + (end - cursor)
        out_pos += copied
        repl_len = len(edit.replacement)
        if mapped_start is None and edit.start <= start < edit.end:
            mapped_start = out_pos
        if mapped_end is None and edit.start < end <= edit.end:
            mapped_end = out_pos + repl_len
        out_pos += repl_len
        cursor = edit.end
    if mapped_start is None:
        mapped_start = out_pos + (start - cursor)
    if mapped_end is None:
        mapped_end = out_pos + (end - cursor)
    if mapped_end < mapped_start:
        mapped_end = mapped_start
    return mapped_start, mapped_end


def _patched_and_edits(masked: str) -> tuple[str, list[SpanEdit]]:
    wrapper = cst.metadata.MetadataWrapper(cst.parse_module(masked))
    edits: list[SpanEdit] = []

    class StructuralVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider, PositionProvider, ParentNodeProvider)

        def __init__(self) -> None:
            super().__init__()
            self.ctx = RuleContext(masked, self)

        def on_visit(self, node: cst.CSTNode) -> bool:
            rule = TRANSFORM_RULES.get(type(node))
            if rule:
                try:
                    res = rule(self.ctx, node)
                    if res:
                        edits.extend(res)
                except Exception:
                    pass
            return True

    wrapper.visit(StructuralVisitor())
    return apply_edits(masked, edits), edits


def _collect_expr_spans(masked: str) -> list[tuple[int, int]]:
    """Call / attribute / subscript / comprehension / assignment-RHS spans on masked source."""
    try:
        wrapper = cst.metadata.MetadataWrapper(cst.parse_module(masked))
    except Exception:
        return []
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    class Collector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider, PositionProvider, ParentNodeProvider)

        def __init__(self) -> None:
            super().__init__()
            self.ctx = RuleContext(masked, self)

        def _in_signature(self, node: cst.CSTNode) -> bool:
            cur: object = node
            for _ in range(32):
                try:
                    parent = self.get_metadata(ParentNodeProvider, cur)
                except Exception:
                    return False
                if parent is None:
                    return False
                if isinstance(parent, (cst.Annotation, cst.Param, cst.Parameters)):
                    return True
                if isinstance(parent, (cst.FunctionDef, cst.Lambda, cst.ClassDef, cst.Module)):
                    return False
                cur = parent
            return False

        def _add(self, node: cst.CSTNode) -> None:
            if self._in_signature(node):
                return
            try:
                start, end = self.ctx.span(node)
            except Exception:
                return
            if end - start < 4 or (start, end) in seen:
                return
            seen.add((start, end))
            spans.append((start, end))

        def visit_Call(self, node: cst.Call) -> bool:
            func = node.func
            if isinstance(func, cst.Name) and func.value in _SKIP_CALL_NAMES:
                return True
            self._add(node)
            return True

        def visit_Attribute(self, node: cst.Attribute) -> bool:
            self._add(node)
            return True

        def visit_Subscript(self, node: cst.Subscript) -> bool:
            head = node.value
            if isinstance(head, cst.Name) and head.value in {
                "List",
                "Dict",
                "Set",
                "Tuple",
                "Optional",
                "Union",
                "Callable",
                "Iterable",
                "Sequence",
                "Mapping",
            }:
                return True
            self._add(node)
            return True

        def visit_ListComp(self, node: cst.ListComp) -> bool:
            self._add(node)
            return True

        def visit_SetComp(self, node: cst.SetComp) -> bool:
            self._add(node)
            return True

        def visit_DictComp(self, node: cst.DictComp) -> bool:
            self._add(node)
            return True

        def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
            self._add(node)
            return True

        def visit_Assign(self, node: cst.Assign) -> bool:
            if not isinstance(node.value, _LEAF_EXPR):
                self._add(node.value)
            return True

        def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
            if node.value is not None and not isinstance(node.value, _LEAF_EXPR):
                self._add(node.value)
            return True

        def visit_AugAssign(self, node: cst.AugAssign) -> bool:
            if not isinstance(node.value, _LEAF_EXPR):
                self._add(node.value)
            return True

    try:
        wrapper.visit(Collector())
    except Exception:
        return []
    return spans


def _slice_to_dialect(patched_slice: str, types: str | None) -> str:
    local = postprocess_spy(patched_slice, aggressive=True).strip()
    local, _ = pack_types(local, table=load_type_table(types))
    return strip_trailing_layout(local.strip())


def _emit_candidate(
    cand: str,
    *,
    max_holes: int,
    min_len: int,
    template_to_regex,
) -> tuple[str, str, tuple[str, ...], int] | None:
    if "<anchor_sep>" in cand:
        return None
    core = strip_trailing_layout(cand)
    if not (min_len <= len(core) < _MAX_CORE_LEN):
        return None
    template, n_holes = generalize(core)
    if n_holes > max_holes:
        return None
    if any(sentinel in template for sentinel in _MASK_SENTINELS):
        return None
    match = template_to_regex(template).search(cand)
    if not match:
        return None
    dens = cand[match.start() : match.end()]
    return template, dens, match.groups(), n_holes


def instance_reward(
    count_fn: Callable[[str], int],
    dens: str,
    payloads: tuple[str, ...],
    n_holes: int,
) -> int:
    from normcore_codec.shorthanding.wildcard import PATTERN_END

    cost_orig = count_fn(dens)
    cost_comp = 1 + (1 if PATTERN_END else 0)
    for payload in payloads:
        cost_comp += count_fn(payload)
    if n_holes > 1:
        cost_comp += n_holes - 1
    return cost_orig - cost_comp


def iter_template_instances(
    source: str,
    *,
    band: str,
    policy: str | None,
    types: str | None,
    max_window: int = 3,
    max_holes: int = 4,
) -> list[tuple[str, str, tuple[str, ...], int]]:
    """Return normalized (template, dens, payloads, n_holes) instances from one file.

    Statement windows are taken on ``mid_full`` units. Expression candidates
    (Call / attribute / subscript / comprehension / assignment RHS) are
    collected on the masked source CST, mapped through the same normalization edits,
    and kept only when the densified slice is a substring of ``mid_full``.
    ``foo(`` / ``.attr`` stay skeleton via ``generalize``.
    """
    from normcore_codec.shorthanding.wildcard import _template_to_regex

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    try:
        masked_full, _ = mask_literals(text, fstring_mode="segment")
        mid_full, _ = apply_normalization(
            masked_full,
            band=band,
            conditional_policy_path=policy,
            type_table_path=types,
        )
        mid_full = (mid_full or "").strip()
        if not mid_full:
            return []
        units = split_dialect_units(mid_full)
        if not units or "".join(units) != mid_full:
            return []
        results: list[tuple[str, str, tuple[str, ...], int]] = []
        n_units = len(units)
        max_w = min(max_window, n_units)
        for width in range(1, max_w + 1):
            for i in range(n_units - width + 1):
                if units[i] in _LAYOUT_ONLY:
                    continue
                cand = "".join(units[i : i + width])
                row = _emit_candidate(
                    cand,
                    max_holes=max_holes,
                    min_len=_MIN_CORE_LEN,
                    template_to_regex=_template_to_regex,
                )
                if row:
                    results.append(row)
        try:
            results.extend(
                _iter_expr_instances(
                    masked_full,
                    mid_full,
                    types=types,
                    max_holes=max_holes,
                    template_to_regex=_template_to_regex,
                )
            )
        except Exception:
            pass
        return results
    except Exception:
        return []


def _iter_expr_instances(
    masked_full: str,
    mid_full: str,
    *,
    types: str | None,
    max_holes: int,
    template_to_regex,
) -> list[tuple[str, str, tuple[str, ...], int]]:
    try:
        patched, edits = _patched_and_edits(masked_full)
    except Exception:
        return []
    out: list[tuple[str, str, tuple[str, ...], int]] = []
    seen: set[str] = set()
    for start, end in _collect_expr_spans(masked_full):
        try:
            ps, pe = _map_span_through_edits(start, end, edits)
            snippet = patched[ps:pe]
            dens = _slice_to_dialect(snippet, types)
        except Exception:
            continue
        if not dens or dens not in mid_full:
            continue
        if dens in seen:
            continue
        if dens.startswith(("<class_stmt>", "<def_stmt>")):
            continue
        if _TYPE_PACK_ONLY.match(dens):
            continue
        if dens.startswith("self ") and ":" in dens:
            continue
        seen.add(dens)
        row = _emit_candidate(
            dens,
            max_holes=max_holes,
            min_len=_MIN_EXPR_LEN,
            template_to_regex=template_to_regex,
        )
        if row:
            out.append(row)
    return out


def _templates_from_source(
    source: str,
    *,
    band: str,
    policy: str | None,
    types: str | None,
    count_fn: Callable[[str], int],
    max_window: int = 3,
    max_holes: int = 4,
) -> list[tuple[str, int]]:
    return [
        (template, instance_reward(count_fn, dens, payloads, n_holes))
        for template, dens, payloads, n_holes in iter_template_instances(
            source,
            band=band,
            policy=policy,
            types=types,
            max_window=max_window,
            max_holes=max_holes,
        )
    ]


def mine_from_sources(
    sources: list[str],
    *,
    min_freq: int = 3,
    top_k: int = 300,
    count_tokens_fn: Callable[[str], int] | None = None,
    band: str = "conditional",
    conditional_policy_path: str | Path | None = None,
    type_table_path: str | Path | None = None,
    compress_fn=None,
    max_window: int = 3,
    max_holes: int = 4,
) -> list[dict]:
    """Return wildcard/exact registry rows mined from Python sources."""
    del compress_fn
    count = count_tokens_fn or get_count_fn()
    policy = str(conditional_policy_path) if conditional_policy_path else None
    types = str(type_table_path) if type_table_path else None

    stats: dict[str, list[int]] = {}

    from tqdm import tqdm

    for source in tqdm(sources, desc="Mining sources"):
        if not source or not str(source).strip():
            continue
        for template, reward in _templates_from_source(
            source,
            band=band,
            policy=policy,
            types=types,
            count_fn=count,
            max_window=max_window,
            max_holes=max_holes,
        ):
            stats.setdefault(template, []).append(reward)

    rows: list[dict] = []
    for template, rewards in stats.items():
        count_hit = len(rewards)
        if count_hit < min_freq:
            continue
        avg_reward = sum(rewards) / count_hit
        if avg_reward < 1.0:
            continue
        holes = len(HOLE_PAT.findall(template))
        pid = "nc_" + hashlib.md5(template.encode("utf-8")).hexdigest()[:8]
        rows.append(
            {
                "id": pid,
                "pattern": template,
                "goal": f"<{pid}_pat>",
                "reward": round(avg_reward, 3),
                "freq": count_hit,
                "type": "wildcard" if holes else "exact",
                "saved_est": int(avg_reward * count_hit),
            }
        )
    rows.sort(key=lambda row: (-row["saved_est"], -row["reward"], -row["freq"]))
    return rows[:top_k]

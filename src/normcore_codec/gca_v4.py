"""GCA-V4: every statement is an island. Pure attention scores. No KIND_W.

Units: SimpleStatementLine (Assign/Return/Expr/Import/...) plus leaf compounds
(If/For/While/With/Try/FunctionDef/ClassDef) after inner-preferring.
Call is not a separate island: it lives inside Expr/Assign lines, otherwise
the partition overlaps. Uncovered leftover bytes are recorded as Gap for
coverage accounting; they are not independently stubbed.
"""
from __future__ import annotations

import ast
from typing import Any

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper

from normcore_codec.gca1 import _byte_span, _frozen_cfg
from normcore_codec.gca_stmt import _PASS

STUB_PREFIX = "_RcdcV4"

COMPOUND = (
    cst.FunctionDef,
    cst.ClassDef,
    cst.If,
    cst.For,
    cst.While,
    cst.With,
    cst.Try,
)
ISLAND_TYPES = (cst.SimpleStatementLine, *COMPOUND)


def _kind(node: cst.CSTNode) -> str:
    if isinstance(node, cst.SimpleStatementLine) and node.body:
        return type(node.body[0]).__name__
    return type(node).__name__


def collect_v4(source: str) -> list[dict[str, Any]]:
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = cst.parse_module(text)
        wrapper = MetadataWrapper(tree)
    except Exception:
        return []

    raw: list[dict[str, Any]] = []

    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

        def on_visit(self, node: cst.CSTNode) -> bool:
            if not isinstance(node, ISLAND_TYPES):
                return True
            try:
                start, end = _byte_span(self.get_metadata(ByteSpanPositionProvider, node))
            except Exception:
                start, end = -1, -1
            if start < 0 or end <= start:
                return True
            raw.append(
                {
                    "node": node,
                    "kind": _kind(node),
                    "code": tree.code_for_node(node),
                    "start": start,
                    "end": end,
                }
            )
            return True

    wrapper.visit(V())
    leaves: list[dict[str, Any]] = []
    for i, a in enumerate(raw):
        contained = False
        for j, b in enumerate(raw):
            if i == j:
                continue
            if b["start"] >= a["start"] and b["end"] <= a["end"] and (b["start"], b["end"]) != (
                a["start"],
                a["end"],
            ):
                contained = True
                break
        if not contained:
            leaves.append(a)

    n = len(text)
    covered = [False] * n
    for row in leaves:
        for k in range(max(0, row["start"]), min(n, row["end"])):
            covered[k] = True
    gaps = []
    i = 0
    while i < n:
        if covered[i] or text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not covered[j] and not text[j].isspace():
            j += 1
        # extend over following whitespace until next covered or nonspace
        gaps.append({"kind": "Gap", "code": text[i:j], "start": i, "end": j, "node": None})
        i = j

    units = leaves + gaps
    units.sort(key=lambda r: (r["start"], r["end"]))
    for idx, row in enumerate(units):
        row["name"] = f"{STUB_PREFIX}{idx:03d}"
        row["n_chars"] = len(row["code"])
        row["stubbable"] = row["node"] is not None
    return units


def coverage_frac(source: str, units: list[dict[str, Any]]) -> float:
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return 1.0
    n = len(text)
    hit = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            hit += 1
            continue
        if any(r["start"] <= i < r["end"] for r in units):
            hit += 1
    return hit / n


def make_stub(node: cst.CSTNode, name: str) -> cst.CSTNode:
    ident = cst.Name(name)
    if isinstance(node, cst.SimpleStatementLine):
        return cst.SimpleStatementLine([cst.Expr(ident)])
    if isinstance(node, cst.FunctionDef):
        return node.with_changes(
            name=ident,
            params=cst.Parameters(),
            decorators=(),
            asynchronous=None,
            returns=None,
            body=_PASS,
        )
    if isinstance(node, cst.ClassDef):
        return node.with_changes(
            name=ident, bases=(), keywords=(), decorators=(), body=_PASS
        )
    if isinstance(node, cst.If):
        return cst.If(test=ident, body=_PASS, orelse=None)
    if isinstance(node, cst.While):
        return cst.While(test=ident, body=_PASS, orelse=None)
    if isinstance(node, cst.For):
        return cst.For(
            target=ident, iter=cst.Name("None"), body=_PASS, orelse=None, asynchronous=None
        )
    if isinstance(node, cst.With):
        return cst.With(items=[cst.WithItem(item=ident)], body=_PASS, asynchronous=None)
    if isinstance(node, cst.Try):
        return cst.Try(
            body=cst.IndentedBlock([cst.SimpleStatementLine([cst.Expr(ident)])]),
            handlers=[cst.ExceptHandler(body=_PASS, type=None, name=None)],
            orelse=None,
            finalbody=None,
        )
    raise TypeError(type(node))


def stub_name(node: cst.CSTNode) -> str | None:
    if isinstance(node, cst.SimpleStatementLine) and node.body:
        expr = node.body[0]
        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Name):
            return expr.value.value
    if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
        return node.name.value
    if isinstance(node, (cst.If, cst.While)) and isinstance(node.test, cst.Name):
        return node.test.value
    if isinstance(node, cst.For) and isinstance(node.target, cst.Name):
        return node.target.value
    if isinstance(node, cst.With) and node.items and isinstance(node.items[0].item, cst.Name):
        return node.items[0].item.value
    if isinstance(node, cst.Try) and node.body.body:
        first = node.body.body[0]
        if isinstance(first, cst.SimpleStatementLine) and first.body:
            expr = first.body[0]
            if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Name):
                return expr.value.value
    return None


class V4Stubber(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

    def __init__(self, keep_spans: dict[tuple[int, int], str]) -> None:
        super().__init__()
        self.keep_spans = keep_spans

    def on_leave(self, original: cst.CSTNode, updated: cst.CSTNode):
        if not isinstance(original, ISLAND_TYPES):
            return updated
        try:
            start, end = _byte_span(self.get_metadata(ByteSpanPositionProvider, original))
        except Exception:
            return updated
        name = self.keep_spans.get((start, end))
        if not name:
            return updated
        return make_stub(updated, name)


class V4Splicer(cst.CSTTransformer):
    def __init__(self, mapping: dict[str, cst.CSTNode]) -> None:
        super().__init__()
        self.mapping = mapping
        self.found: set[str] = set()

    def on_leave(self, original: cst.CSTNode, updated: cst.CSTNode):
        if not isinstance(original, ISLAND_TYPES):
            return updated
        name = stub_name(original)
        if name in self.mapping:
            self.found.add(name)
            return self.mapping[name]
        return updated


def compress_islands(
    source: str,
    *,
    keep_indices: list[int] | None = None,
    keep_names: list[str] | None = None,
    config=None,
    assets=None,
    **kwargs,
):
    from normcore_codec.pipeline import CompressResult, compress

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    units = collect_v4(text)
    stubbable = [u for u in units if u["stubbable"]]
    if not stubbable:
        raise ValueError("GCA v4: no stubbable statement islands")
    names = [s["name"] for s in stubbable]
    if keep_names is not None:
        keep = set(keep_names)
    elif keep_indices is not None:
        keep = {names[i] for i in keep_indices if 0 <= i < len(names)}
    else:
        raise ValueError("GCA v4 requires keep_indices or keep_names")
    keep &= set(names)
    if not keep:
        raise ValueError("GCA v4 keep set is empty after dropping Gap units")
    keep_spans = {(s["start"], s["end"]): s["name"] for s in stubbable if s["name"] in keep}
    wrapper = MetadataWrapper(cst.parse_module(text))
    stubbed = wrapper.visit(V4Stubber(keep_spans)).code
    ast.parse(stubbed)
    frozen = _frozen_cfg(config)
    encoded = compress(stubbed, config=frozen, assets=assets, surface="codec")
    missing = [name for name in keep if name not in encoded.code]
    if missing:
        raise RuntimeError(f"GCA v4 Pass B ate stubs: {missing[:3]}")
    train = encoded.code
    islands = {}
    for row in stubbable:
        if row["name"] in keep:
            islands[row["name"]] = row["code"]
            train = train.replace(row["name"], row["code"], 1)
    stats = dict(encoded.stats)
    stats["pass_a"] = {**(stats.get("pass_a") or {}), "mode": "gca_v4_statements"}
    stats["gca_v4"] = {
        "dialect": encoded.code,
        "islands": islands,
        "keep_names": sorted(keep),
        "n_units": len(units),
        "n_stubbable": len(stubbable),
        "n_keep": len(keep),
        "kinds": [s["kind"] for s in stubbable if s["name"] in keep],
    }
    # V10 Attention Coloring: replace ALL units with ranked markers
    # This ensures 100% differentiation based on attention scores
    units_with_scores = []
    for row in stubbable:
        units_with_scores.append(row)

    # We need the scores to do the ranking.
    # Since scores aren't passed into compress_islands directly,
    # we have to rely on the calling script to pass them or
    # we use a temporary hack for preview.
    # In the real 16k dump, we will pass a dict.

    # For now, let's just make sure the infrastructure exists.
    # The actual ranking logic will be in the compress_islands caller or a new helper.

    v10_markers = kwargs.get("v10_markers", {})
    for name, marker in v10_markers.items():
        if name in islands:
            # Inject marker BEFORE the code island
            train = train.replace(name, f"{marker}{islands[name]}", 1)

    return CompressResult(code=train, stats=stats)


def restore_islands(result: Any, *, config=None, assets=None) -> str:
    from normcore_codec.pipeline import decompress

    meta = (result.stats or {}).get("gca_v4") or {}
    dialect = meta.get("dialect")
    islands = meta.get("islands") or {}
    if not dialect or not islands:
        raise ValueError("GCA v4 restore needs dialect and islands")
    restored_stub = decompress(dialect, config=_frozen_cfg(config), assets=assets)
    mapping: dict[str, cst.CSTNode] = {}
    for name, code in islands.items():
        blob = code if code.endswith("\n") else code + "\n"
        parsed = cst.parse_module(blob)
        if not parsed.body:
            raise RuntimeError(f"GCA v4 {name} empty")
        mapping[name] = parsed.body[0]
    splicer = V4Splicer(mapping)
    spliced = cst.parse_module(restored_stub).visit(splicer)
    lost = [name for name in islands if name not in splicer.found]
    if lost:
        raise RuntimeError(f"GCA v4 stub missing after decompress: {lost[:3]}")
    return spliced.code

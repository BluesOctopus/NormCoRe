"""GCA statement islands: whole If/For/def/class subtrees stay native.

Inner-preferring: drop a span that contains another candidate so units stay small.
Never veto a single Pass A edit; the whole CST node is stubbed then spliced.
"""
from __future__ import annotations

import ast
from typing import Any

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper

from recode_codec.config import CodecAssets, CodecConfig, PassAConfig, PassBConfig
from recode_codec.gca1 import _byte_span, _frozen_cfg

STUB_PREFIX = "_RcdcStmt"

ISLAND_TYPES = (
    cst.FunctionDef,
    cst.ClassDef,
    cst.If,
    cst.For,
    cst.While,
    cst.With,
    cst.Try,
)

_PASS = cst.IndentedBlock([cst.SimpleStatementLine([cst.Pass()])])


def collect_statements(source: str) -> list[dict[str, Any]]:
    from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper

    try:
        tree = cst.parse_module(source)
        wrapper = MetadataWrapper(tree)
    except Exception:
        return []

    raw: list[dict[str, Any]] = []

    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

        def on_visit(self, node: cst.CSTNode) -> bool:
            if isinstance(node, ISLAND_TYPES):
                try:
                    start, end = _byte_span(self.get_metadata(ByteSpanPositionProvider, node))
                except Exception:
                    start, end = -1, -1
                raw.append(
                    {
                        "node": node,
                        "kind": type(node).__name__,
                        "code": tree.code_for_node(node),
                        "n_chars": 0,
                        "start": start,
                        "end": end,
                    }
                )
            return True

    wrapper.visit(V())
    kept: list[dict[str, Any]] = []
    for i, a in enumerate(raw):
        contained = False
        for j, b in enumerate(raw):
            if i == j or a["start"] < 0 or b["start"] < 0:
                continue
            if b["start"] >= a["start"] and b["end"] <= a["end"] and (b["start"], b["end"]) != (a["start"], a["end"]):
                contained = True
                break
        if not contained:
            kept.append(a)
    for i, row in enumerate(kept):
        row["name"] = f"{STUB_PREFIX}{i:03d}"
        row["n_chars"] = len(row["code"])
    return kept


def make_stub(node: cst.CSTNode, name: str) -> cst.CSTNode:
    ident = cst.Name(name)
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
            name=ident,
            bases=(),
            keywords=(),
            decorators=(),
            body=_PASS,
        )
    if isinstance(node, cst.If):
        return cst.If(test=ident, body=_PASS, orelse=None)
    if isinstance(node, cst.While):
        return cst.While(test=ident, body=_PASS, orelse=None)
    if isinstance(node, cst.For):
        return cst.For(target=ident, iter=cst.Name("None"), body=_PASS, orelse=None, asynchronous=None)
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


class StmtStubber(cst.CSTTransformer):
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


class StmtSplicer(cst.CSTTransformer):
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
    config: CodecConfig | None = None,
    assets: CodecAssets | None = None,
) -> Any:
    from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper
    from recode_codec.pipeline import CompressResult, compress

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    stmts = collect_statements(text)
    if not stmts:
        raise ValueError("GCA stmt: no statement islands")
    names = [s["name"] for s in stmts]
    if keep_names is not None:
        keep = set(keep_names)
    elif keep_indices is not None:
        keep = {names[i] for i in keep_indices if 0 <= i < len(names)}
    else:
        raise ValueError("GCA stmt requires keep_indices or keep_names")
    if not keep:
        raise ValueError("GCA stmt keep set is empty")
    unknown = keep - set(names)
    if unknown:
        raise ValueError(f"GCA stmt unknown names: {sorted(unknown)[:5]}")
    keep_spans = {(s["start"], s["end"]): s["name"] for s in stmts if s["name"] in keep}
    wrapper = MetadataWrapper(cst.parse_module(text))
    stubbed = wrapper.visit(StmtStubber(keep_spans)).code
    ast.parse(stubbed)
    frozen = _frozen_cfg(config)
    encoded = compress(stubbed, config=frozen, assets=assets, surface="codec")
    missing = [name for name in keep if name not in encoded.code]
    if missing:
        raise RuntimeError(f"GCA stmt Pass B ate stubs: {missing[:3]}")
    train = encoded.code
    islands = {}
    for row in stmts:
        if row["name"] in keep:
            islands[row["name"]] = row["code"]
            train = train.replace(row["name"], row["code"], 1)
    stats = dict(encoded.stats)
    stats["pass_a"] = {**(stats.get("pass_a") or {}), "mode": "gca_statement_islands"}
    stats["gca_stmt"] = {
        "dialect": encoded.code,
        "islands": islands,
        "keep_names": sorted(keep),
        "n_stmts": len(stmts),
        "n_keep": len(keep),
        "kinds": [s["kind"] for s in stmts if s["name"] in keep],
    }
    return CompressResult(code=train, stats=stats)


def restore_islands(result: Any, *, config: CodecConfig | None = None, assets: CodecAssets | None = None) -> str:
    from recode_codec.pipeline import decompress

    meta = (result.stats or {}).get("gca_stmt") or {}
    dialect = meta.get("dialect")
    islands = meta.get("islands") or {}
    if not dialect or not islands:
        raise ValueError("GCA stmt restore needs dialect and islands")
    restored_stub = decompress(dialect, config=_frozen_cfg(config), assets=assets)
    mapping: dict[str, cst.CSTNode] = {}
    for name, code in islands.items():
        blob = code if code.endswith("\n") else code + "\n"
        parsed = cst.parse_module(blob)
        if len(parsed.body) != 1:
            raise RuntimeError(f"GCA stmt {name} is not a single module statement")
        mapping[name] = parsed.body[0]
    splicer = StmtSplicer(mapping)
    spliced = cst.parse_module(restored_stub).visit(splicer)
    lost = [name for name in islands if name not in splicer.found]
    if lost:
        raise RuntimeError(f"GCA stmt stub missing after decompress: {lost[:3]}")
    return spliced.code

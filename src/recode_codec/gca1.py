"""GCA1: keep selected top-level FunctionDefs as native islands.

Frozen aggressive A-then-B compresses the rest. No newline hanging, no new specials.
"""
from __future__ import annotations

import ast
from typing import Any

import libcst as cst

from recode_codec.config import CodecAssets, CodecConfig, PassAConfig, PassBConfig

STUB_PREFIX = "_RcdcIsland"


def _byte_span(span: Any) -> tuple[int, int]:
    """libcst CodeSpan is (start, length), not (start, end)."""
    start = int(span.start)
    length = int(getattr(span, "length", 0) or 0)
    if length <= 0 and getattr(span, "end", None) is not None:
        return start, int(span.end)
    return start, start + length


def collect_functions(source: str) -> list[dict[str, Any]]:
    from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper

    try:
        tree = cst.parse_module(source)
        wrapper = MetadataWrapper(tree)
    except Exception:
        return []

    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

        def __init__(self) -> None:
            super().__init__()
            self.fn_depth = 0
            self.rows: list[dict[str, Any]] = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            self.fn_depth += 1
            if self.fn_depth == 1:
                name = f"{STUB_PREFIX}{len(self.rows):03d}"
                try:
                    start, end = _byte_span(self.get_metadata(ByteSpanPositionProvider, node))
                except Exception:
                    start, end = -1, -1
                # Island text must parse as a module statement. code_for_node
                # dedents class methods; source[start:end] keeps body indent
                # and breaks restore_islands. Byte spans are for scoring only.
                code = tree.code_for_node(node)
                self.rows.append(
                    {
                        "name": name,
                        "node": node,
                        "code": code,
                        "n_chars": len(code),
                        "start": start,
                        "end": end,
                    }
                )
            return True

        def leave_FunctionDef(self, original: cst.FunctionDef) -> None:
            self.fn_depth -= 1

    visitor = V()
    wrapper.visit(visitor)
    return visitor.rows


def make_stub(original: cst.FunctionDef, name: str) -> cst.FunctionDef:
    return original.with_changes(
        name=cst.Name(name),
        params=cst.Parameters(),
        decorators=(),
        asynchronous=None,
        returns=None,
        body=cst.IndentedBlock([cst.SimpleStatementLine([cst.Pass()])]),
    )


class SelectiveStubber(cst.CSTTransformer):
    def __init__(self, names: list[str], keep: set[str]) -> None:
        super().__init__()
        self.names = names
        self.keep = keep
        self.i = 0
        self.fn_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.fn_depth += 1
        return True

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.fn_depth -= 1
        if self.fn_depth == 0:
            name = self.names[self.i]
            self.i += 1
            if name in self.keep:
                return make_stub(updated, name)
        return updated


class Splicer(cst.CSTTransformer):
    def __init__(self, mapping: dict[str, cst.CSTNode]) -> None:
        super().__init__()
        self.mapping = mapping
        self.fn_depth = 0
        self.found: set[str] = set()
        self.errors: list[str] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.fn_depth += 1
        return True

    def leave_FunctionDef(self, original: cst.FunctionDef, updated: cst.FunctionDef):
        self.fn_depth -= 1
        name = original.name.value
        if name in self.mapping and self.fn_depth == 0:
            self.found.add(name)
            return self.mapping[name]
        return updated


def _frozen_cfg(config: CodecConfig | None) -> CodecConfig:
    base = config or CodecConfig.canonical()
    return CodecConfig(
        pass_a=PassAConfig(mode="aggressive"),
        pass_b=base.pass_b,
        verify_assets=False,
    )


def compress_islands(
    source: str,
    *,
    keep_indices: list[int] | None = None,
    keep_names: list[str] | None = None,
    config: CodecConfig | None = None,
    assets: CodecAssets | None = None,
) -> Any:
    from recode_codec.pipeline import CompressResult, compress

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    funcs = collect_functions(text)
    if not funcs:
        raise ValueError("GCA1: no top-level FunctionDef to keep")
    names = [f["name"] for f in funcs]
    if keep_names is not None:
        keep = set(keep_names)
    elif keep_indices is not None:
        keep = {names[i] for i in keep_indices if 0 <= i < len(names)}
    else:
        raise ValueError("GCA1 requires keep_indices or keep_names")
    if not keep:
        raise ValueError("GCA1 keep set is empty")
    unknown = keep - set(names)
    if unknown:
        raise ValueError(f"GCA1 unknown island names: {sorted(unknown)[:5]}")
    mapping = {f["name"]: f["node"] for f in funcs if f["name"] in keep}
    tree = cst.parse_module(text)
    stubbed = tree.visit(SelectiveStubber(names, keep)).code
    ast.parse(stubbed)
    frozen = _frozen_cfg(config)
    encoded = compress(stubbed, config=frozen, assets=assets, surface="codec")
    missing = [name for name in keep if name not in encoded.code]
    if missing:
        raise RuntimeError(f"GCA1 Pass B ate stubs: {missing[:3]}")
    train = encoded.code
    islands = {}
    for fn in funcs:
        if fn["name"] in keep:
            islands[fn["name"]] = fn["code"]
            train = train.replace(fn["name"], fn["code"], 1)
    stats = dict(encoded.stats)
    stats["pass_a"] = {**(stats.get("pass_a") or {}), "mode": "gca1_function_islands"}
    stats["gca1"] = {
        "dialect": encoded.code,
        "islands": islands,
        "keep_names": sorted(keep),
        "n_funcs": len(funcs),
        "n_keep": len(keep),
    }
    return CompressResult(code=train, stats=stats)


def restore_islands(result: Any, *, config: CodecConfig | None = None, assets: CodecAssets | None = None) -> str:
    from recode_codec.pipeline import decompress

    meta = (result.stats or {}).get("gca1") or {}
    dialect = meta.get("dialect")
    islands = meta.get("islands") or {}
    if not dialect or not islands:
        raise ValueError("GCA1 restore needs stats['gca1'] dialect and islands")
    restored_stub = decompress(dialect, config=_frozen_cfg(config), assets=assets)
    restored_tree = cst.parse_module(restored_stub)
    mapping: dict[str, cst.CSTNode] = {}
    for name, code in islands.items():
        blob = code if code.endswith("\n") else code + "\n"
        parsed = cst.parse_module(blob)
        if len(parsed.body) != 1:
            raise RuntimeError(f"GCA1 island {name} is not a single module statement")
        mapping[name] = parsed.body[0]
    splicer = Splicer(mapping)
    spliced = restored_tree.visit(splicer)
    lost = [name for name in islands if name not in splicer.found]
    if lost:
        raise RuntimeError(f"GCA1 stub missing after decompress: {lost[:3]}")
    if splicer.errors:
        raise RuntimeError(splicer.errors[0])
    return spliced.code

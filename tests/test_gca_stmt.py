"""Statement-island encode/restore. CPU."""
from __future__ import annotations

import ast
from pathlib import Path

from normcore_codec import CodecAssets, CodecConfig, PassAConfig, PassBConfig, compress, restore
from normcore_codec.gca_stmt import collect_statements

ASSETS = CodecAssets.from_root(Path(__file__).resolve().parents[1] / "assets")
CFG = CodecConfig(
    pass_a=PassAConfig(mode="aggressive"),
    pass_b=PassBConfig(enabled=True),
    verify_assets=False,
)

SRC = """
def outer(x):
    if x:
        return x + 1
    return 0

class A:
    def meth(self, y):
        return y
"""


def _dump(text: str) -> str:
    return ast.dump(ast.parse(text), annotate_fields=False)


def test_collect_prefers_inner_if_over_function():
    rows = collect_statements(SRC)
    kinds = [r["kind"] for r in rows]
    assert "If" in kinds
    assert "FunctionDef" in kinds
    assert kinds.count("FunctionDef") >= 1


def test_stmt_island_if_roundtrip():
    rows = collect_statements(SRC)
    if_names = [r["name"] for r in rows if r["kind"] == "If"]
    assert if_names
    result = compress(SRC, config=CFG, assets=ASSETS, stmt_keep_names=if_names)
    assert result.stats["pass_a"]["mode"] == "gca_statement_islands"
    restored = restore(result, config=CFG, assets=ASSETS)
    assert _dump(SRC) == _dump(restored)


def test_stmt_island_method_roundtrip():
    rows = collect_statements(SRC)
    meth = [r["name"] for r in rows if r["kind"] == "FunctionDef" and "meth" in r["code"]]
    assert meth
    result = compress(SRC, config=CFG, assets=ASSETS, stmt_keep_names=meth)
    restored = restore(result, config=CFG, assets=ASSETS)
    assert _dump(SRC) == _dump(restored)

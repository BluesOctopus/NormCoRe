"""GCA1 function-island encode/restore. CPU only."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from recode_codec import CodecAssets, CodecConfig, PassAConfig, PassBConfig, compress, restore

ASSETS = CodecAssets.from_root(Path(__file__).resolve().parents[1] / "assets")
CFG = CodecConfig(
    pass_a=PassAConfig(mode="aggressive"),
    pass_b=PassBConfig(enabled=True),
    verify_assets=False,
)

SRC = """
def keep_me(x):
    return x + 1

def cold_me(y):
    if y:
        return y
    return 0
"""


def _dump(text: str) -> str:
    return ast.dump(ast.parse(text), annotate_fields=False)


def test_gca1_roundtrip_keeps_selected_function():
    result = compress(SRC, config=CFG, assets=ASSETS, keep_indices=[0])
    assert result.stats["pass_a"]["mode"] == "gca1_function_islands"
    assert result.stats["gca1"]["n_keep"] == 1
    assert "def keep_me" in result.code
    restored = restore(result, config=CFG, assets=ASSETS)
    assert _dump(SRC) == _dump(restored)


def test_veto_without_legacy_flag_is_closed():
    with pytest.raises(RuntimeError, match="Option B newline GCA is closed"):
        compress(SRC, config=CFG, assets=ASSETS, veto_fn=lambda s, e: True)


CLASS_SRC = """
class A:
    def keep_me(self, x):
        return x + 1

    def cold_me(self, y):
        if y:
            return y
        return 0
"""


def test_collect_functions_byte_spans_include_class_methods():
    from recode_codec.gca1 import collect_functions

    funcs = collect_functions(CLASS_SRC)
    assert len(funcs) == 2
    for fn in funcs:
        assert fn["start"] >= 0
        assert fn["end"] > fn["start"]
        slice_ = CLASS_SRC[fn["start"] : fn["end"]]
        assert slice_.lstrip().startswith("def ")
        ast.parse(fn["code"])
    assert "keep_me" in CLASS_SRC[funcs[0]["start"] : funcs[0]["end"]]
    assert "cold_me" in CLASS_SRC[funcs[1]["start"] : funcs[1]["end"]]


def test_class_method_island_roundtrip():
    result = compress(CLASS_SRC, config=CFG, assets=ASSETS, keep_indices=[0])
    assert "def keep_me" in result.code
    restored = restore(result, config=CFG, assets=ASSETS)
    assert _dump(CLASS_SRC) == _dump(restored)


def test_edit_veto_none_matches_frozen_roundtrip():
    called = []

    def veto_none(start: int, end: int) -> bool:
        called.append((start, end))
        return False

    result = compress(SRC, config=CFG, assets=ASSETS, edit_veto_fn=veto_none)
    assert called
    assert result.stats["pass_a"]["mode"] == "dietcode_edit_veto"
    assert result.stats["pass_a"]["n_edit_vetoed"] == 0
    restored = restore(result, config=CFG, assets=ASSETS)
    assert _dump(SRC) == _dump(restored)


def test_gca_mode_without_keep_asks_for_islands():
    cfg = CodecConfig(
        pass_a=PassAConfig(mode="gca"),
        pass_b=PassBConfig(enabled=False),
        verify_assets=False,
    )
    with pytest.raises(RuntimeError, match="keep_indices"):
        compress("def f():\n    return 1\n", config=cfg, assets=ASSETS)

from __future__ import annotations

import ast
import ctypes
import json
from pathlib import Path

import pytest

from normcore_codec import (
    CodecAssets,
    CodecConfig,
    NormalizationConfig,
    ShorthandingConfig,
    compress,
    decompress,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = CodecAssets.from_root(ROOT / "assets")
FROZEN = CodecConfig(
    normalization=NormalizationConfig(mode="aggressive"),
    shorthanding=ShorthandingConfig(enabled=True),
    verify_assets=False,
)


def require_native_grammar() -> None:
    """Skip decoder tests when the checked-in grammar targets another OS."""
    try:
        ctypes.CDLL(str(ASSETS.grammar_binary))
    except OSError:
        pytest.skip("spy_lang.so is not loadable on this platform")


def ast_key(source: str) -> str:
    return ast.dump(ast.parse(source), annotate_fields=False)


@pytest.mark.parametrize(
    "source",
    [
        "def add(a, b):\n    return a + b\n",
        "for i in range(3):\n    if i % 2:\n        print(i)\n",
        "class Box:\n    def __init__(self, value):\n        self.value = value\n",
        "async def fetch(xs):\n    return [x async for x in xs]\n",
        "def classify(x):\n    match x:\n        case [a, b]:\n            return a + b\n        case _:\n            return None\n",
    ],
)
def test_frozen_ast_roundtrip(source: str):
    require_native_grammar()
    encoded = compress(source, config=FROZEN, assets=ASSETS)
    restored = decompress(encoded.code, config=FROZEN, assets=ASSETS)
    assert ast_key(restored) == ast_key(source)


def test_pipeline_declares_contract_order():
    result = compress(
        "def f():\n    return 1\n",
        config=FROZEN,
        assets=ASSETS,
    )
    assert result.stats["stage_order"] == ["mask", "normalization", "shorthanding", "unmask"]
    assert result.stats["normalization"]["mode"] == "frozen"


def test_marker_like_literal_and_comment_are_not_semantically_expanded():
    require_native_grammar()
    registry = json.loads(ASSETS.pattern_registry.read_text(encoding="utf-8"))
    goal = registry["patterns"][0]["goal"].split("__H", 1)[0].rstrip(";")
    source = f'def f():\n    value = "{goal}"  # {goal}\n    return value\n'
    encoded = compress(source, config=FROZEN, assets=ASSETS)
    restored = decompress(encoded.code, config=FROZEN, assets=ASSETS)
    assert goal in restored
    assert ast_key(restored) == ast_key(source)


def test_fstring_literal_and_expression_roundtrip():
    require_native_grammar()
    source = 'def f(user, n):\n    return f"<tag>{user}:{n + 1}</tag>"\n'
    encoded = compress(source, config=FROZEN, assets=ASSETS)
    restored = decompress(encoded.code, config=FROZEN, assets=ASSETS)
    assert ast_key(restored) == ast_key(source)


def test_structural_only_uses_same_public_api():
    require_native_grammar()
    config = CodecConfig.normalization_only()
    source = "while ready:\n    ready = False\n"
    encoded = compress(source, config=config, assets=ASSETS)
    restored = decompress(encoded.code, config=config, assets=ASSETS)
    assert ast_key(restored) == ast_key(source)


def test_pipeline_forwards_explicit_shorthanding_count_fn(monkeypatch):
    seen = {}

    def fake_dynamic(text, registry, **kwargs):
        seen["count_fn"] = kwargs.get("count_fn")
        return text, {"patterns_applied": 0, "reward_mode": "runtime_exact"}

    def fake_shorthanding(masked_source, text, registry, **kwargs):
        out, stats = fake_dynamic(text, registry, **kwargs)
        return out, stats, {"skipped": "disabled"}

    monkeypatch.setattr("normcore_codec.codec.apply_shorthanding", fake_shorthanding)
    count_fn = lambda value: len(value)
    result = compress(
        "def f():\n    return 1\n",
        config=FROZEN,
        assets=ASSETS,
        shorthanding_count_fn=count_fn,
    )

    assert seen["count_fn"] is count_fn
    assert result.stats["shorthanding"]["reward_mode"] == "runtime_exact"

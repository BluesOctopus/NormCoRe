from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from normcore_codec import (
    CodecAssets,
    CodecConfig,
    PassAConfig,
    PassBConfig,
    compress,
    decompress,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = CodecAssets.from_root(ROOT / "assets")
FROZEN = CodecConfig(
    pass_a=PassAConfig(mode="aggressive"),
    pass_b=PassBConfig(enabled=True),
    verify_assets=False,
)


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
    encoded = compress(source, config=FROZEN, assets=ASSETS)
    restored = decompress(encoded.code, config=FROZEN, assets=ASSETS)
    assert ast_key(restored) == ast_key(source)


def test_pipeline_declares_contract_order():
    result = compress(
        "def f():\n    return 1\n",
        config=FROZEN,
        assets=ASSETS,
    )
    assert result.stats["stage_order"] == ["mask", "pass_a", "pass_b", "unmask"]
    assert result.stats["pass_a"]["mode"] == "frozen"


def test_marker_like_literal_and_comment_are_not_semantically_expanded():
    registry = json.loads(ASSETS.pattern_registry.read_text(encoding="utf-8"))
    goal = registry["patterns"][0]["goal"].split("__H", 1)[0].rstrip(";")
    source = f'def f():\n    value = "{goal}"  # {goal}\n    return value\n'
    encoded = compress(source, config=FROZEN, assets=ASSETS)
    restored = decompress(encoded.code, config=FROZEN, assets=ASSETS)
    assert goal in restored
    assert ast_key(restored) == ast_key(source)


def test_fstring_literal_and_expression_roundtrip():
    source = 'def f(user, n):\n    return f"<tag>{user}:{n + 1}</tag>"\n'
    encoded = compress(source, config=FROZEN, assets=ASSETS)
    restored = decompress(encoded.code, config=FROZEN, assets=ASSETS)
    assert ast_key(restored) == ast_key(source)


def test_structural_only_uses_same_public_api():
    config = CodecConfig.structural_only()
    source = "while ready:\n    ready = False\n"
    encoded = compress(source, config=config, assets=ASSETS)
    restored = decompress(encoded.code, config=config, assets=ASSETS)
    assert ast_key(restored) == ast_key(source)


def test_pipeline_forwards_explicit_pass_b_count_fn(monkeypatch):
    seen = {}

    def fake_dynamic(text, registry, **kwargs):
        seen["count_fn"] = kwargs.get("count_fn")
        return text, {"patterns_applied": 0, "reward_mode": "runtime_exact"}

    def fake_pass_b(masked_source, text, registry, **kwargs):
        out, stats = fake_dynamic(text, registry, **kwargs)
        return out, stats, {"skipped": "disabled"}

    monkeypatch.setattr("normcore_codec.pipeline.apply_pass_b", fake_pass_b)
    count_fn = lambda value: len(value)
    result = compress(
        "def f():\n    return 1\n",
        config=FROZEN,
        assets=ASSETS,
        pass_b_count_fn=count_fn,
    )

    assert seen["count_fn"] is count_fn
    assert result.stats["pass_b"]["reward_mode"] == "runtime_exact"

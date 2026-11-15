"""Fail-loud GCA entry and Pass B hard veto. CPU only."""
from __future__ import annotations

import pytest

from recode_codec import CodecAssets, CodecConfig, PassAConfig, PassBConfig, compress
from recode_codec.dynamic import PatternEntry, apply_dynamic_pass

from pathlib import Path

ASSETS = CodecAssets.from_root(Path(__file__).resolve().parents[1] / "assets")


def test_canonical_compress_without_keep_raises():
    with pytest.raises(RuntimeError, match="keep_indices"):
        compress(
            "def f():\n    return 1\n",
            config=CodecConfig.canonical(),
            assets=ASSETS,
        )


def test_gca_mode_without_keep_raises():
    cfg = CodecConfig(
        pass_a=PassAConfig(mode="gca"),
        pass_b=PassBConfig(enabled=False),
        verify_assets=False,
    )
    with pytest.raises(RuntimeError, match="keep_indices"):
        compress("def f():\n    return 1\n", config=cfg, assets=ASSETS)


def test_aggressive_without_veto_still_runs():
    cfg = CodecConfig(
        pass_a=PassAConfig(mode="aggressive"),
        pass_b=PassBConfig(enabled=False),
        verify_assets=False,
    )
    result = compress("def f():\n    return 1\n", config=cfg, assets=ASSETS)
    assert result.stats["pass_a"]["mode"] == "frozen"
    assert result.stats["pass_a"]["dialect_hotzones"] == set()


def test_pass_b_hard_veto_drops_overlapping_match():
    text = "foo\nbar"
    nl = text.index("\n")
    registry = [PatternEntry("p", 4.0, "<p_pat>", "foo\nbar", "exact")]
    kept, stats_kept = apply_dynamic_pass(text, registry)
    assert kept == "<p_pat>"
    assert stats_kept["n_vetoed"] == 0
    blocked, stats_blocked = apply_dynamic_pass(text, registry, dialect_hotzones={nl})
    assert blocked == text
    assert stats_blocked["n_vetoed"] == 1
    assert stats_blocked["patterns_applied"] == 0

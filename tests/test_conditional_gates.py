"""Fail-loud Conditional Attention entry and shorthanding veto. CPU only."""
from __future__ import annotations

from normcore_codec import CodecAssets, CodecConfig, NormalizationConfig, ShorthandingConfig, compress
from normcore_codec.shorthanding import PatternEntry, apply_registry_shorthanding

from pathlib import Path

ASSETS = CodecAssets.from_root(Path(__file__).resolve().parents[1] / "assets")


def test_aggressive_without_veto_still_runs():
    cfg = CodecConfig(
        normalization=NormalizationConfig(mode="aggressive"),
        shorthanding=ShorthandingConfig(enabled=False),
        verify_assets=False,
    )
    result = compress("def f():\n    return 1\n", config=cfg, assets=ASSETS)
    assert result.stats["normalization"]["mode"] == "frozen"
    assert result.stats["normalization"]["dialect_hotzones"] == set()


def test_shorthanding_hard_veto_drops_overlapping_match():
    text = "foo\nbar"
    nl = text.index("\n")
    registry = [PatternEntry("p", 4.0, "<p_pat>", "foo\nbar", "exact")]
    kept, stats_kept = apply_registry_shorthanding(text, registry)
    assert kept == "<p_pat>"
    assert stats_kept["n_vetoed"] == 0
    blocked, stats_blocked = apply_registry_shorthanding(text, registry, dialect_hotzones={nl})
    assert blocked == text
    assert stats_blocked["n_vetoed"] == 1
    assert stats_blocked["patterns_applied"] == 0

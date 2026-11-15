from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from normcore_codec.assets import AssetValidationError, validate_assets, validate_registry
from normcore_codec.attention.conditional import load_policy
from normcore_codec.attention.scoring import exponential_reward
from normcore_codec.config import CodecAssets
from normcore_codec.shorthanding import PatternEntry, _dp_select, apply_registry_shorthanding, restore_registry_shorthanding
from normcore_codec.shorthanding.wildcard import bare_goal_token
from normcore_codec.reversible.markers import all_special_tokens
from normcore_codec.normalization.idioms import idiom_special_tokens
from normcore_codec.normalization.type_pack import load_type_table
from normcore_codec.tokens import get_count_fn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = CodecAssets.from_root(ROOT / "assets")


def test_figure_exponential_reward_exact():
    assert exponential_reward(10, 0.25, lambda_=2) == pytest.approx(
        10 * math.exp(-0.5)
    )
    with pytest.raises(ValueError):
        exponential_reward(1, 1.1, lambda_=1)


def test_canonical_policy_gates_high_attention_rules():
    policy = load_policy(ASSETS.conditional_policy)
    assert policy.per_rule_gate is True
    assert {"import", "class_def"} <= policy.protected
    assert "ImportFrom" in policy.disabled_optional_rules
    assert "Call" in policy.enabled_optional_rules
    assert policy.meta["formula"] == "exp_neg_lambda_i"


def test_locked_assets_and_registry_are_valid():
    result = validate_assets(ASSETS)
    assert result["registry"]["patterns"] > 0
    registry = json.loads(ASSETS.pattern_registry.read_text(encoding="utf-8"))
    assert all(row["reweight_mode"] == "exp_neg_lambda_i" for row in registry["patterns"])
    assert not any("MASK" in row["pattern"] for row in registry["patterns"])


def test_model_blind_baseline_uses_same_safe_candidate_pool():
    aggressive = ROOT / "assets/registry_aggressive_safe.json"
    canonical_stats = validate_registry(ASSETS.pattern_registry)
    aggressive_stats = validate_registry(aggressive)
    assert canonical_stats["patterns"] == aggressive_stats["patterns"] == 181
    baseline = json.loads(aggressive.read_text(encoding="utf-8"))
    assert baseline["meta"]["method"] == "model_blind_safe_baseline"


def test_registry_goals_are_atomic_for_locked_gpt4_tokenizer():
    registry = json.loads(ASSETS.pattern_registry.read_text(encoding="utf-8"))
    pattern_tokens = {bare_goal_token(row["goal"]) for row in registry["patterns"]}
    type_tokens = {
        str(row["id"])
        for row in load_type_table(ASSETS.type_table)
        if row.get("id")
    }
    count_fn = get_count_fn(
        specials=set(all_special_tokens())
        | set(idiom_special_tokens())
        | pattern_tokens
        | type_tokens
    )
    for row in registry["patterns"]:
        assert count_fn(bare_goal_token(row["goal"])) == 1


def test_registry_rejects_duplicate_goals(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "patterns": [
                    {"id": "a", "pattern": "alpha", "goal": "<x_pat>", "reward": 1},
                    {"id": "b", "pattern": "beta", "goal": "<x_pat>", "reward": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssetValidationError):
        validate_registry(path)


def test_dp_selects_optimal_non_overlapping_matches():
    matches = [
        {"start": 0, "end": 4, "reward": 4.0, "id": "wide"},
        {"start": 0, "end": 2, "reward": 3.0, "id": "left"},
        {"start": 2, "end": 4, "reward": 3.0, "id": "right"},
    ]
    assert {row["id"] for row in _dp_select(matches)} == {"left", "right"}


def test_exact_patterns_respect_both_identifier_boundaries():
    registry = [PatternEntry("p", 3.0, "<p_pat>", "name", "exact")]
    out, stats = apply_registry_shorthanding("name rename names", registry)
    assert out == "<p_pat> rename names"
    assert stats["patterns_applied"] == 1
    assert restore_registry_shorthanding(out, registry) == "name rename names"


def test_runtime_exact_reward_filters_nonpositive_matches_and_reports_funnel():
    registry = [
        PatternEntry("long", 99.0, "<long_pat>", "long", "exact"),
        PatternEntry("short", 99.0, "<short_pat>", "x", "exact"),
    ]
    costs = {
        "long": 3,
        "<long_pat>": 1,
        "x": 1,
        "<short_pat>": 2,
        "long x": 4,
        "<long_pat> x": 2,
    }

    out, stats = apply_registry_shorthanding(
        "long x",
        registry,
        count_fn=lambda value: costs.get(value, len(value)),
    )

    assert out == "<long_pat> x"
    assert stats["reward_mode"] == "runtime_exact"
    assert stats["candidate_matches_raw"] == 2
    assert stats["nonpositive_reward"] == 1
    assert stats["matches_after_filters"] == 1
    assert stats["patterns_applied"] == 1
    assert stats["selected_reward"] == 2
    assert stats["selected_original_tokens"] == 3
    assert stats["selected_compressed_tokens"] == 1
    assert stats["global_reward"] > 0
    assert stats["global_fallback"] is False
    assert restore_registry_shorthanding(out, registry) == "long x"


def test_runtime_exact_reward_falls_back_on_nonadditive_global_loss():
    registry = [PatternEntry("long", 99.0, "<long_pat>", "long", "exact")]
    costs = {
        "long": 3,
        "<long_pat>": 1,
        "long tail": 2,
        "<long_pat> tail": 4,
    }

    out, stats = apply_registry_shorthanding(
        "long tail",
        registry,
        count_fn=lambda value: costs.get(value, len(value)),
    )

    assert out == "long tail"
    assert stats["patterns_applied"] == 0
    assert stats["selected_reward"] == 2
    assert stats["global_reward"] == -2
    assert stats["reward_calibration_error"] == -4
    assert stats["global_fallback"] is True

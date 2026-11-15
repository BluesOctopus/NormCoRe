"""Unified facade for the reversible Pass B compression stages."""

from __future__ import annotations

from typing import Callable

from normcore_codec.dynamic import PatternEntry, apply_dynamic_pass, invert_dynamic_pass
from normcore_codec.dynamic.subtree import apply_subtree_pass, invert_subtree_pass


def apply_pass_b(
    masked_source: str,
    text: str,
    registry: list[PatternEntry],
    *,
    band_paint: list[str] | None = None,
    count_fn: Callable[[str], int] | None = None,
    dialect_hotzones: set[int] | None = None,
    skip_match: Callable[[dict], bool] | None = None,
    scope_capture_slots: int = 0,
    subtree_slots: int = 0,
    type_table_path=None,
    joint_best_of: bool = False,
) -> tuple[str, dict, dict]:
    """Apply registry/capture compression followed by local subtree refs."""
    out, dynamic_stats = apply_dynamic_pass(
        text,
        registry,
        band_paint=band_paint,
        count_fn=count_fn,
        dialect_hotzones=dialect_hotzones,
        skip_match=skip_match,
        scope_capture_slots=scope_capture_slots,
    )
    if subtree_slots <= 0:
        return out, dynamic_stats, {"skipped": "disabled"}
    if count_fn is None:
        raise ValueError("Pass B subtree references require pass_b_count_fn")
    out_with_subtree, subtree_stats = apply_subtree_pass(
        masked_source,
        out,
        slots=subtree_slots,
        count_fn=count_fn,
        type_table_path=type_table_path,
    )
    if not joint_best_of or scope_capture_slots <= 0:
        return out_with_subtree, dynamic_stats, subtree_stats

    plain, plain_stats = apply_dynamic_pass(
        text,
        registry,
        band_paint=band_paint,
        count_fn=count_fn,
        dialect_hotzones=dialect_hotzones,
        skip_match=skip_match,
        scope_capture_slots=0,
    )
    plain_with_subtree, plain_subtree_stats = apply_subtree_pass(
        masked_source,
        plain,
        slots=subtree_slots,
        count_fn=count_fn,
        type_table_path=type_table_path,
    )
    candidates = [
        ("patterns_capture_subtree", out_with_subtree, dynamic_stats, subtree_stats),
        ("patterns_subtree", plain_with_subtree, plain_stats, plain_subtree_stats),
        ("patterns_capture", out, dynamic_stats, {"skipped": "joint_best_of"}),
        ("patterns", plain, plain_stats, {"skipped": "joint_best_of"}),
    ]
    arm_tokens = {name: count_fn(candidate) for name, candidate, _dynamic, _subtree in candidates}
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (arm_tokens[candidates[index][0]], index),
    )
    selected = candidates[selected_index]
    name, selected_text, selected_dynamic, selected_subtree = selected
    selected_dynamic = {
        **selected_dynamic,
        "joint_best_of": True,
        "joint_selected_arm": name,
        "joint_arm_tokens": arm_tokens,
    }
    return selected_text, selected_dynamic, selected_subtree


def invert_pass_b(text: str, registry: list[PatternEntry]) -> str:
    """Invert the local dictionary before expanding registry patterns."""
    return invert_dynamic_pass(invert_subtree_pass(text), registry)

"""Conditional Attention policy and scoring helpers.

The model is used offline to produce a frozen policy. The codec hot path only
consumes that policy; it never loads a model.
"""

from __future__ import annotations

from normcore_codec.attention.occurrence import (
    align_paint,
    apply_edits_to_paint,
    majority_band,
    paint_from_scores,
)
from normcore_codec.attention.conditional import (
    ConditionalPolicy,
    build_conditional_registry,
    build_policy_from_attention_cfg,
    load_category_scores,
    load_policy,
    save_policy,
)
from normcore_codec.attention.scoring import (
    annotate_registry,
    build_attention_tables,
    exponential_reward,
    g_one_minus_a,
    load_attention_profile,
    pattern_attention,
    reweight_reward,
)

__all__ = [
    "ConditionalPolicy",
    "align_paint",
    "annotate_registry",
    "apply_edits_to_paint",
    "build_attention_tables",
    "build_conditional_registry",
    "build_policy_from_attention_cfg",
    "exponential_reward",
    "g_one_minus_a",
    "load_attention_profile",
    "load_category_scores",
    "load_policy",
    "majority_band",
    "paint_from_scores",
    "pattern_attention",
    "reweight_reward",
    "save_policy",
]

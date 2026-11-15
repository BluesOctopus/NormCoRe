"""Global Context-aware Attention; instance gating is legacy ablation only.

- ``gca``: offline category map + Pass A gating + Pass B exponential reward
- ``weight.*`` / ``band_paint``: used by GCA build and legacy ``gated`` ablation
"""

from __future__ import annotations

from normcore_codec.attention.band_paint import (
    align_paint,
    apply_edits_to_paint,
    majority_band,
    paint_from_scores,
)
from normcore_codec.attention.gca import (
    GCAPolicy,
    build_gca_registry,
    build_policy_from_attention_cfg,
    load_category_scores,
    load_policy,
    save_policy,
)
from normcore_codec.attention.weight import (
    annotate_registry,
    build_attention_tables,
    exponential_reward,
    g_one_minus_a,
    load_attention_profile,
    pattern_attention,
    reweight_reward,
)

__all__ = [
    "GCAPolicy",
    "align_paint",
    "annotate_registry",
    "apply_edits_to_paint",
    "build_attention_tables",
    "build_gca_registry",
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

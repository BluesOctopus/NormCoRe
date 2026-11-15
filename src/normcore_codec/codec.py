from __future__ import annotations

import ast
import re as std_re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Callable

from normcore_codec.assets import validate_assets
from normcore_codec.config import CodecAssets, CodecConfig
from normcore_codec.shorthanding import load_registry
from normcore_codec.reversible.literals import (
    count_literal_placeholders,
    mask_literals,
    unmask_literals,
)
from normcore_codec.shorthanding.pipeline import apply_shorthanding, restore_shorthanding
from normcore_codec.reversible.grammar import decode_spy
from normcore_codec.normalization import apply_normalization, invert_type_pack

class PlaceholderLeak(ValueError):
    """semantic shorthanding failed to conserve the literal placeholders handed over by normalization."""

@dataclass
class CompressResult:
    code: str
    stats: dict = field(default_factory=dict)

def compress(
    source: str,
    *,
    config: CodecConfig | None = None,
    assets: CodecAssets | None = None,
    surface: Literal["codec", "training"] = "codec",
    veto_fn: Callable[[int, int], bool] | None = None,
    shorthanding_count_fn: Callable[[str], int] | None = None,
    **kwargs,
) -> CompressResult:
    """Compress Python source with the canonical NormCoRe pipeline.

    ``shorthanding_count_fn`` enables per-instance token accounting for semantic shorthanding. It is
    deliberately supplied by the experiment runner because an already-loaded
    tokenizer cannot be represented portably by ``CodecAssets``.
    """
    cfg = config or CodecConfig.canonical()
    shorthanding_scope_capture_slots = int(kwargs.pop("shorthanding_scope_capture_slots", 0) or 0)
    shorthanding_subtree_slots = int(kwargs.pop("shorthanding_subtree_slots", 0) or 0)
    shorthanding_joint_best_of = bool(kwargs.pop("shorthanding_joint_best_of", False))
    resolved = assets or CodecAssets()
    needs_assets = cfg.shorthanding.enabled
    asset_stats = validate_assets(resolved) if cfg.verify_assets and needs_assets else None

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    normalization_kwargs = dict(
        band=cfg.normalization.mode,
        type_table_path=resolved.type_table,
        conditional_policy_path=resolved.conditional_policy,
        veto_fn=veto_fn,
        **kwargs,
    )
    masked, mask_records = mask_literals(text, fstring_mode="segment")
    mid, st_a = apply_normalization(masked, mask_records=mask_records, **normalization_kwargs)
    if st_a.get("error"):
        masked, mask_records = mask_literals(text, fstring_mode="whole")
        mid, st_a = apply_normalization(masked, mask_records=mask_records, **normalization_kwargs)
        st_a = {**st_a, "fstring_mask": "whole"}
    else:
        st_a = {**st_a, "fstring_mask": "segment"}

    dialect_hotzones = st_a.get("dialect_hotzones")
    stats: dict = {
        "contract": "NormCoRe-normalization-then-shorthanding/v1",
        "stage_order": ["mask", "normalization", "shorthanding", "unmask"],
        "assets": asset_stats,
        "normalization": st_a,
    }

    paint = st_a.get("band_paint") if cfg.normalization.mode != "conservative" else None

    if cfg.shorthanding.enabled:
        if resolved.pattern_registry is None:
            raise ValueError("semantic shorthanding requires registry")
        reg = load_registry(resolved.pattern_registry)
        count_fn = shorthanding_count_fn
        if count_fn is None and resolved.tokenizer_path:
            from normcore_codec.tokens import get_count_fn
            from normcore_codec.reversible.markers import all_special_tokens
            from normcore_codec.shorthanding.wildcard import bare_goal_token
            from normcore_codec.normalization.idioms import idiom_special_tokens
            from normcore_codec.normalization.type_pack import load_type_table
            b_specials = {bare_goal_token(r.goal) for r in reg}
            type_ids = {
                str(row["id"])
                for row in load_type_table(resolved.type_table)
                if row.get("id")
            }
            specials = (
                set(all_special_tokens())
                | set(idiom_special_tokens())
                | b_specials
                | type_ids
            )
            count_fn = get_count_fn(resolved.tokenizer_path, specials)

        before = count_literal_placeholders(mid)
        mid, st_b, subtree_stats = apply_shorthanding(
            masked, mid, reg, band_paint=paint, count_fn=count_fn,
            dialect_hotzones=dialect_hotzones if (dialect_hotzones and len(dialect_hotzones) > 0) else None,
            skip_match=kwargs.get("shorthanding_skip_match"),
            scope_capture_slots=shorthanding_scope_capture_slots,
            subtree_slots=shorthanding_subtree_slots,
            type_table_path=resolved.type_table,
            joint_best_of=shorthanding_joint_best_of,
        )
        after = count_literal_placeholders(mid)
        if after != before:
            raise PlaceholderLeak(f"semantic shorthanding leak: {before} -> {after}")
        stats["shorthanding"] = {**st_b, "literal_placeholders": before}
        if shorthanding_subtree_slots > 0:
            if count_literal_placeholders(mid) != before:
                raise PlaceholderLeak("semantic shorthanding subtree dictionary changed literal placeholders")
            stats["shorthanding_subtree"] = subtree_stats
            stats["stage_order"] = ["mask", "normalization", "shorthanding", "shorthanding_subtree", "unmask"]
        else:
            stats["shorthanding_subtree"] = {"skipped": "disabled"}
    else:
        stats["shorthanding"] = {"skipped": "disabled"}
        stats["shorthanding_subtree"] = {"skipped": "shorthanding_disabled"}

    code = unmask_literals(mid, mask_records)
    if surface == "training":
        code = std_re.sub(r"<attention_br>(.*?)</attention_br>", r"\1", code, flags=std_re.DOTALL)

    return CompressResult(code=code, stats=stats)

def decompress(
    code: str,
    *,
    config: CodecConfig | None = None,
    assets: CodecAssets | None = None,
) -> str:
    from normcore_codec.normalization.idioms import unpack_aggressive_idioms
    cfg = config or CodecConfig.canonical()
    resolved = assets or CodecAssets()
    # Remove optional attention annotations from training surfaces.
    mid = std_re.sub(r"<attn_q[1-4]>", "", code)
    mid = std_re.sub(r"<logic>|</logic>", "", mid)
    mid = std_re.sub(r"<attention_br>.*?</attention_br>", "", mid, flags=std_re.DOTALL)
    masked, records = mask_literals(mid, fstring_mode="segment_nested")
    if cfg.shorthanding.enabled:
        reg = load_registry(resolved.pattern_registry)
        mid = restore_shorthanding(masked, reg)
    else:
        mid = masked
    mid = unpack_aggressive_idioms(mid)
    mid = invert_type_pack(mid, type_table_path=resolved.type_table)
    return unmask_literals(decode_spy(mid, grammar_binary=resolved.grammar_binary), records)


def restore(result: CompressResult, *, config: CodecConfig | None = None, assets: CodecAssets | None = None) -> str:
    return decompress(result.code, config=config, assets=assets)

def verify_ast_roundtrip(source: str, *, config: CodecConfig | None = None, assets: CodecAssets | None = None) -> bool:
    try:
        cr = compress(source, config=config, assets=assets, surface="codec")
        restored = decompress(cr.code, config=config, assets=assets)
        return ast.dump(ast.parse(source), annotate_fields=False) == ast.dump(ast.parse(restored), annotate_fields=False)
    except Exception: return False

def verify_exec_roundtrip(source: str, *, config: CodecConfig | None = None, assets: CodecAssets | None = None) -> bool:
    try:
        cr = compress(source, config=config, assets=assets, surface="codec")
        restored = decompress(cr.code, config=config, assets=assets)
        compile(restored, "<normcore>", "exec")
        return True
    except Exception: return False

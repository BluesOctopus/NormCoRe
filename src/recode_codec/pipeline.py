import ast
import re as std_re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Callable

from recode_codec.assets import validate_assets
from recode_codec.config import CodecAssets, CodecConfig
from recode_codec.dynamic import load_registry
from recode_codec.mask import (
    count_literal_placeholders,
    mask_literals,
    unmask_literals,
)
from recode_codec.passb import apply_pass_b, invert_pass_b
from recode_codec.spy_decode import decode_spy
from recode_codec.structural import apply_structural_pass, invert_type_pack

class PlaceholderLeak(ValueError):
    """Pass B failed to conserve the literal placeholders handed over by Pass A."""

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
    keep_indices: list[int] | None = None,
    keep_names: list[str] | None = None,
    allow_legacy_option_b: bool = False,
    pass_b_count_fn: Callable[[str], int] | None = None,
    **kwargs,
) -> CompressResult:
    """RECODE compression.

    ``pass_b_count_fn`` enables per-instance token accounting for Pass B. It is
    deliberately supplied by the experiment runner because an already-loaded
    tokenizer cannot be represented portably by ``CodecAssets``.
    """
    cfg = config or CodecConfig.canonical()
    stmt_keep_names = kwargs.pop("stmt_keep_names", None)
    stmt_keep_indices = kwargs.pop("stmt_keep_indices", None)
    v4_keep_names = kwargs.pop("v4_keep_names", None)
    v4_keep_indices = kwargs.pop("v4_keep_indices", None)
    v5_keep_names = kwargs.pop("v5_keep_names", None)
    v5_keep_indices = kwargs.pop("v5_keep_indices", None)
    pass_b_scope_capture_slots = int(kwargs.pop("pass_b_scope_capture_slots", 0) or 0)
    pass_b_subtree_slots = int(kwargs.pop("pass_b_subtree_slots", 0) or 0)
    pass_b_joint_best_of = bool(kwargs.pop("pass_b_joint_best_of", False))
    if v5_keep_names is not None or v5_keep_indices is not None:
        from recode_codec.gca_v5 import compress_islands as compress_v5

        return compress_v5(
            source,
            keep_indices=v5_keep_indices,
            keep_names=v5_keep_names,
            config=cfg,
            assets=assets,
        )
    if v4_keep_names is not None or v4_keep_indices is not None:
        from recode_codec.gca_v4 import compress_islands as compress_v4

        return compress_v4(
            source,
            keep_indices=v4_keep_indices,
            keep_names=v4_keep_names,
            config=cfg,
            assets=assets,
        )
    if stmt_keep_names is not None or stmt_keep_indices is not None:
        from recode_codec.gca_stmt import compress_islands as compress_stmts

        return compress_stmts(
            source,
            keep_indices=stmt_keep_indices,
            keep_names=stmt_keep_names,
            config=cfg,
            assets=assets,
        )
    if keep_indices is not None or keep_names is not None:
        from recode_codec.gca1 import compress_islands

        return compress_islands(
            source,
            keep_indices=keep_indices,
            keep_names=keep_names,
            config=cfg,
            assets=assets,
        )
    if veto_fn is not None and not allow_legacy_option_b:
        raise RuntimeError(
            "Option B newline GCA is closed. "
            "Use keep_indices=/keep_names= for GCA1 function islands, "
            "or allow_legacy_option_b=True only to reproduce the failed arm."
        )
    if cfg.pass_a.mode in {"gca", "gca1"} and veto_fn is None:
        raise RuntimeError(
            "GCA1 requires keep_indices or keep_names (which FunctionDefs stay native). "
            "Frozen dialect uses PassAConfig(mode='aggressive')."
        )
    resolved = assets or CodecAssets()
    needs_assets = cfg.pass_b.enabled or cfg.pass_a.mode == "gca"
    asset_stats = validate_assets(resolved) if cfg.verify_assets and needs_assets else None

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    pass_a_kwargs = dict(
        band=cfg.pass_a.mode,
        type_table_path=resolved.type_table,
        gca_policy_path=resolved.gca_policy,
        veto_fn=veto_fn,
        allow_legacy_option_b=allow_legacy_option_b,
        **kwargs,
    )
    masked, mask_records = mask_literals(text, fstring_mode="segment")
    mid, st_a = apply_structural_pass(masked, mask_records=mask_records, **pass_a_kwargs)
    if st_a.get("error"):
        masked, mask_records = mask_literals(text, fstring_mode="whole")
        mid, st_a = apply_structural_pass(masked, mask_records=mask_records, **pass_a_kwargs)
        st_a = {**st_a, "fstring_mask": "whole"}
    else:
        st_a = {**st_a, "fstring_mask": "segment"}

    dialect_hotzones = st_a.get("dialect_hotzones")
    stats: dict = {
        "contract": "RECODE-A-then-B/v1",
        "stage_order": ["mask", "pass_a", "pass_b", "unmask"],
        "assets": asset_stats,
        "pass_a": st_a,
    }

    paint = st_a.get("band_paint") if cfg.pass_a.mode not in ("gca", "conservative") else None

    if cfg.pass_b.enabled:
        if resolved.pattern_registry is None:
            raise ValueError("Pass B requires registry")
        reg = load_registry(resolved.pattern_registry)
        count_fn = pass_b_count_fn
        if count_fn is None and resolved.tokenizer_path:
            from recode_codec.tokens import get_count_fn
            from recode_codec.markers import all_special_tokens
            from recode_codec.dynamic.wildcard import bare_goal_token
            from recode_codec.structural.aggressive import idiom_special_tokens
            from recode_codec.structural.type_pack import load_type_table
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
        mid, st_b, subtree_stats = apply_pass_b(
            masked, mid, reg, band_paint=paint, count_fn=count_fn,
            dialect_hotzones=dialect_hotzones if (dialect_hotzones and len(dialect_hotzones) > 0) else None,
            skip_match=kwargs.get("pass_b_skip_match"),
            scope_capture_slots=pass_b_scope_capture_slots,
            subtree_slots=pass_b_subtree_slots,
            type_table_path=resolved.type_table,
            joint_best_of=pass_b_joint_best_of,
        )
        after = count_literal_placeholders(mid)
        if after != before:
            raise PlaceholderLeak(f"Pass B leak: {before} -> {after}")
        stats["pass_b"] = {**st_b, "literal_placeholders": before}
        if pass_b_subtree_slots > 0:
            if count_literal_placeholders(mid) != before:
                raise PlaceholderLeak("Pass B subtree dictionary changed literal placeholders")
            stats["pass_b_subtree"] = subtree_stats
            stats["stage_order"] = ["mask", "pass_a", "pass_b", "pass_b_subtree", "unmask"]
        else:
            stats["pass_b_subtree"] = {"skipped": "disabled"}
    else:
        stats["pass_b"] = {"skipped": "disabled"}
        stats["pass_b_subtree"] = {"skipped": "pass_b_disabled"}

    code = unmask_literals(mid, mask_records)
    if surface == "training":
        code = std_re.sub(r"<gca_br>(.*?)</gca_br>", r"\1", code, flags=std_re.DOTALL)

    return CompressResult(code=code, stats=stats)

def decompress(
    code: str,
    *,
    config: CodecConfig | None = None,
    assets: CodecAssets | None = None,
) -> str:
    from recode_codec.structural.aggressive import unpack_aggressive_idioms
    cfg = config or CodecConfig.canonical()
    resolved = assets or CodecAssets()
    # Remove GCA Attention Quantile Markers
    mid = std_re.sub(r"<attn_q[1-4]>", "", code)
    mid = std_re.sub(r"<logic>|</logic>", "", mid)
    mid = std_re.sub(r"<gca_br>.*?</gca_br>", "", mid, flags=std_re.DOTALL)
    masked, records = mask_literals(mid, fstring_mode="segment_nested")
    if cfg.pass_b.enabled:
        reg = load_registry(resolved.pattern_registry)
        mid = invert_pass_b(masked, reg)
    else:
        mid = masked
    mid = unpack_aggressive_idioms(mid)
    mid = invert_type_pack(mid, type_table_path=resolved.type_table)
    return unmask_literals(decode_spy(mid, grammar_binary=resolved.grammar_binary), records)


def restore(result: CompressResult, *, config: CodecConfig | None = None, assets: CodecAssets | None = None) -> str:
    if result.stats.get("gca_v5"):
        from recode_codec.gca_v5 import restore_islands as restore_v5

        return restore_v5(result, config=config, assets=assets)
    if result.stats.get("gca_v4"):
        from recode_codec.gca_v4 import restore_islands as restore_v4

        return restore_v4(result, config=config, assets=assets)
    if result.stats.get("gca_stmt"):
        from recode_codec.gca_stmt import restore_islands as restore_stmts

        return restore_stmts(result, config=config, assets=assets)
    if result.stats.get("gca1"):
        from recode_codec.gca1 import restore_islands

        return restore_islands(result, config=config, assets=assets)
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
        compile(restored, "<recode>", "exec")
        return True
    except Exception: return False

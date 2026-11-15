"""Pass B typed holes and the A->B placeholder conservation contract."""

from __future__ import annotations

import json

import pytest

from recode_codec import CodecAssets, CodecConfig, PassAConfig, PassBConfig, compress
from recode_codec.assets import AssetValidationError, validate_registry
from recode_codec.dynamic import apply_dynamic_pass, invert_dynamic_pass, load_registry
from recode_codec.mask import count_literal_placeholders, mask_literals, unmask_literals
from recode_codec.markers import PATTERN_END
from recode_codec.pipeline import PlaceholderLeak


def write_registry(tmp_path, rows, name="registry.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "patterns": rows}), encoding="utf-8")
    return path


IDENT_ROW = {
    "id": "t_for_range_len",
    "pattern": "<for_stmt>__H0__ range(len(__H1__))",
    "goal": "<t_for_range_len_pat>",
    "reward": 3,
    "freq": 10,
}
STRING_ROW = {
    "id": "t_raise_value",
    "pattern": "<raise>ValueError(__S0__)",
    "goal": "<t_raise_value_pat>",
    "reward": 3,
    "freq": 10,
}


def test_identifier_holes_round_trip(tmp_path):
    registry = load_registry(write_registry(tmp_path, [IDENT_ROW]))
    text = "<for_stmt>i range(len(items))<block_start>pass<block_end>"
    encoded, stats = apply_dynamic_pass(text, registry, band_paint=None)
    assert stats["patterns_applied"] == 1
    assert encoded.startswith("<t_for_range_len_pat>i;items" + PATTERN_END)
    assert invert_dynamic_pass(encoded, registry) == text


def test_string_hole_carries_the_mask_placeholder(tmp_path):
    registry = load_registry(write_registry(tmp_path, [STRING_ROW]))
    text = '<raise>ValueError("MASK")<line_sep>'
    encoded, stats = apply_dynamic_pass(text, registry, band_paint=None)
    assert stats["patterns_applied"] == 1
    # The placeholder rides inside the shorthand instead of being deleted.
    assert count_literal_placeholders(encoded) == count_literal_placeholders(text) == 1
    assert invert_dynamic_pass(encoded, registry) == text


def test_terminator_keeps_adjacent_statements_apart(tmp_path):
    """Without <pat_end> the trailing capture would swallow the next token."""
    row = {
        "id": "t_assign_self",
        "pattern": "self.__H0__=__H1__",
        "goal": "<t_assign_self_pat>",
        "reward": 2,
        "freq": 10,
    }
    registry = load_registry(write_registry(tmp_path, [row]))
    text = "self.name=value<line_sep>other=thing"
    encoded, _ = apply_dynamic_pass(text, registry, band_paint=None)
    assert invert_dynamic_pass(encoded, registry) == text


def test_repeated_hole_index_must_match_the_same_text(tmp_path):
    row = {
        "id": "t_swap",
        "pattern": "__H0__=__H0__+1",
        "goal": "<t_swap_pat>",
        "reward": 2,
        "freq": 10,
    }
    registry = load_registry(write_registry(tmp_path, [row]))
    hit, stats = apply_dynamic_pass("count=count+1", registry, band_paint=None)
    assert stats["patterns_applied"] == 1
    assert invert_dynamic_pass(hit, registry) == "count=count+1"
    miss, stats = apply_dynamic_pass("count=total+1", registry, band_paint=None)
    assert stats["patterns_applied"] == 0
    assert miss == "count=total+1"


def test_mask_eating_pattern_is_rejected_at_build_time(tmp_path):
    row = {**STRING_ROW, "pattern": '<raise>ValueError("MASK")'}
    with pytest.raises(AssetValidationError, match="consumes a literal mask"):
        validate_registry(write_registry(tmp_path, [row]))


def test_hole_index_gap_is_rejected_at_build_time(tmp_path):
    row = {**IDENT_ROW, "pattern": "<for_stmt>__H0__ range(len(__H2__))"}
    with pytest.raises(AssetValidationError, match="malformed holes"):
        validate_registry(write_registry(tmp_path, [row]))


def test_compress_rejects_a_pattern_that_absorbs_a_placeholder(tmp_path):
    """The pipeline guard is the last line of defence behind validation."""
    source = 'def f():\n    raise ValueError("boom")\n'
    masked, records = mask_literals(source, fstring_mode="segment")
    assert count_literal_placeholders(masked) == len(records)

    base = CodecAssets.from_root(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "assets"
    )
    row = {
        "id": "t_eat",
        "pattern": 'ValueError("MASK")',
        "goal": "<t_eat_pat>",
        "reward": 4,
        "freq": 10,
    }
    assets = CodecAssets(
        pattern_registry=write_registry(tmp_path, [row], "eater.json"),
        gca_policy=base.gca_policy,
        type_table=base.type_table,
        grammar_binary=base.grammar_binary,
        asset_lock=None,
    )
    config = CodecConfig(
        pass_a=PassAConfig(mode="aggressive"), pass_b=PassBConfig(enabled=True), verify_assets=False
    )
    with pytest.raises(PlaceholderLeak):
        compress(source, config=config, assets=assets)


def test_unmask_consumes_exactly_the_counted_placeholders():
    """The guard counts what unmask will consume, comments included."""
    source = 'x = "a"\ny = f"{x} b"  # note\nz = """doc"""\n'
    masked, records = mask_literals(source, fstring_mode="segment")
    assert count_literal_placeholders(masked) == len(records) == 4
    # Comments are restored with a trailing newline by design, so compare the
    # byte-exact round trip on a comment-free sample only.
    plain = 'x = "a"\ny = f"{x} b"\nz = """doc"""\n'
    plain_masked, plain_records = mask_literals(plain, fstring_mode="segment")
    assert count_literal_placeholders(plain_masked) == len(plain_records)
    assert unmask_literals(plain_masked, plain_records) == plain

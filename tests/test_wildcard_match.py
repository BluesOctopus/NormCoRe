from recode_codec.dynamic.wildcard import (
    WildcardPattern,
    find_wildcard_matches,
    invert_wildcard,
    scope_reference_tokens,
)
from recode_codec.dynamic import apply_dynamic_pass, invert_dynamic_pass, PatternEntry


def test_wildcard_match_on_densified_pass_a_surface():
    row = {
        "id": "v2_test",
        "pattern": "<if_stmt>__H0__ is None<block_start><return>__H1__<block_end>",
        "reward": 10,
        "type": "wildcard",
        "goal": "<v2_test_pat>",
    }
    wp = WildcardPattern.from_row(row)
    assert wp is not None
    text = "<if_stmt>data is None<block_start><return>result<block_end>"
    matches = find_wildcard_matches(text, [wp])
    assert len(matches) == 1
    assert "data" in matches[0]["goal"]
    assert "result" in matches[0]["goal"]
    encoded = text[: matches[0]["start"]] + matches[0]["goal"] + text[matches[0]["end"] :]
    assert invert_wildcard(encoded, [wp]) == text


def test_wildcard_allows_flexible_spaces_between_fixed_tokens():
    row = {
        "id": "v2_test",
        "pattern": "<if_stmt>__H0__ is None<block_start><return>__H1__<block_end>",
        "reward": 10,
        "type": "wildcard",
        "goal": "<v2_test_pat>",
    }
    wp = WildcardPattern.from_row(row)
    text = "<if_stmt>data  is None<block_start><return>result<block_end>"
    matches = find_wildcard_matches(text, [wp])
    assert len(matches) == 1


def test_scope_capture_reuses_prior_identifier_without_dictionary_header():
    entry = PatternEntry(
        "scope_test",
        10,
        "<scope_test_pat>",
        "<return>normalize_and_validate(__H0__)<line_sep>",
        "wildcard",
    )
    specials = {
        "<scope_test_pat>",
        "<return>",
        "<line_sep>",
        "<pat_end>",
    } | scope_reference_tokens(4)

    def count_fn(value: str) -> int:
        total = 0
        cursor = 0
        ordered = sorted(specials, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in ordered if value.startswith(item, cursor)), None)
            if token is not None:
                total += 1
                cursor += len(token)
            else:
                total += 1
                cursor += 1
        return total

    text = (
        "<return>normalize_and_validate(long_variable)<line_sep>"
        "<return>normalize_and_validate(long_variable)<line_sep>"
    )
    encoded, stats = apply_dynamic_pass(
        text,
        [entry],
        count_fn=count_fn,
        scope_capture_slots=4,
    )
    assert "<cap_ref_0>" in encoded
    assert stats["scope_compact_matches"] == 1
    assert stats["scope_reference_uses"] == 1
    assert invert_dynamic_pass(encoded, [entry]) == text


def test_scope_capture_never_references_string_holes():
    entry = PatternEntry(
        "literal_test",
        10,
        "<literal_test_pat>",
        "<return>__S0__<line_sep>",
        "wildcard",
    )
    text = '<return>"MASK"<line_sep><return>"MASK"<line_sep>'
    encoded, stats = apply_dynamic_pass(
        text,
        [entry],
        scope_capture_slots=4,
    )
    assert "<cap_ref_" not in encoded
    assert stats["scope_compact_matches"] == 0
    assert invert_dynamic_pass(encoded, [entry]) == text


def test_scope_capture_can_mix_identifier_reference_with_raw_literal():
    entry = PatternEntry(
        "mixed_test",
        10,
        "<mixed_test_pat>",
        "<return>normalize_and_validate(__H0__,__S1__)<line_sep>",
        "wildcard",
    )
    specials = {
        "<mixed_test_pat>",
        "<return>",
        "<line_sep>",
        "<pat_end>",
    } | scope_reference_tokens(4)

    def count_fn(value: str) -> int:
        total = 0
        cursor = 0
        ordered = sorted(specials, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in ordered if value.startswith(item, cursor)), None)
            if token is not None:
                total += 1
                cursor += len(token)
            else:
                total += 1
                cursor += 1
        return total

    text = (
        '<return>normalize_and_validate(long_variable,"MASK")<line_sep>'
        '<return>normalize_and_validate(long_variable,"MASK")<line_sep>'
    )
    encoded, stats = apply_dynamic_pass(
        text,
        [entry],
        count_fn=count_fn,
        scope_capture_slots=4,
    )
    assert "<cap_ref_0>" in encoded
    assert encoded.count('"MASK"') == 2
    assert stats["scope_compact_matches"] == 1
    assert invert_dynamic_pass(encoded, [entry]) == text


def test_scope_capture_uses_identifier_from_reconstructed_plain_prefix():
    entry = PatternEntry(
        "prefix_test",
        10,
        "<prefix_test_pat>",
        "<return>normalize_and_validate(__H0__)<line_sep>",
        "wildcard",
    )
    specials = {
        "<prefix_test_pat>",
        "<return>",
        "<line_sep>",
        "<pat_end>",
    } | scope_reference_tokens(8)

    def count_fn(value: str) -> int:
        total = 0
        cursor = 0
        ordered = sorted(specials, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in ordered if value.startswith(item, cursor)), None)
            if token is not None:
                total += 1
                cursor += len(token)
            else:
                total += 1
                cursor += 1
        return total

    text = (
        "long_variable<line_sep>"
        "<return>normalize_and_validate(long_variable)<line_sep>"
    )
    encoded, stats = apply_dynamic_pass(
        text,
        [entry],
        count_fn=count_fn,
        scope_capture_slots=8,
    )
    assert "<prefix_test_pat><cap_ref_0>" in encoded
    assert stats["scope_compact_matches"] == 1
    assert invert_dynamic_pass(encoded, [entry]) == text


def test_scope_reference_compacts_plain_identifier_after_exact_goal():
    entry = PatternEntry(
        "plain_prefix_test",
        10,
        "<plain_prefix_test_pat>",
        "long_identifier<line_sep>",
        "exact",
    )
    specials = {
        "<plain_prefix_test_pat>",
        "<line_sep>",
    } | scope_reference_tokens(32)

    def count_fn(value: str) -> int:
        total = 0
        cursor = 0
        ordered = sorted(specials, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in ordered if value.startswith(item, cursor)), None)
            total += 1
            cursor += len(token) if token is not None else 1
        return total

    text = "long_identifier<line_sep>long_identifier+1<line_sep>"
    encoded, stats = apply_dynamic_pass(
        text,
        [entry],
        count_fn=count_fn,
        scope_capture_slots=32,
    )
    assert "<plain_prefix_test_pat>" in encoded
    assert "<cap_ref_0>" in encoded
    assert stats["scope_plain_reference_uses"] == 1
    assert invert_dynamic_pass(encoded, [entry]) == text


def test_scope_reference_never_compacts_composite_mask_placeholder():
    entry = PatternEntry(
        "unused_test",
        1,
        "<unused_test_pat>",
        "unmatched<line_sep>",
        "exact",
    )
    specials = {"<line_sep>"} | scope_reference_tokens(32)

    def count_fn(value: str) -> int:
        total = 0
        cursor = 0
        ordered = sorted(specials, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in ordered if value.startswith(item, cursor)), None)
            total += 1
            cursor += len(token) if token is not None else 1
        return total

    text = "table['MASK']<line_sep>table['MASK']<line_sep>"
    encoded, stats = apply_dynamic_pass(
        text,
        [entry],
        count_fn=count_fn,
        scope_capture_slots=32,
    )
    assert encoded == text
    assert stats["scope_plain_reference_uses"] == 0
    assert invert_dynamic_pass(encoded, [entry]) == text

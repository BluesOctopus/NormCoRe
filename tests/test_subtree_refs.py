from normcore_codec.shorthanding.subtree import (
    SUBTREE_BIND,
    SUBTREE_END,
    apply_subtree_pass,
    invert_subtree_pass,
    subtree_reference_tokens,
)


def count_atomic(text: str) -> int:
    specials = subtree_reference_tokens(8) | {"<pat_end>"}
    count = 0
    cursor = 0
    while cursor < len(text):
        token = next((value for value in specials if text.startswith(value, cursor)), None)
        if token is not None:
            count += 1
            cursor += len(token)
        else:
            count += 1
            cursor += 1
    return count


def test_subtree_wire_roundtrip():
    encoded = (
        "x="
        + SUBTREE_BIND
        + "values[index]"
        + "<pat_end>"
        + ";y=<subtree_ref_0>"
    )
    assert invert_subtree_pass(encoded) == "x=values[index];y=values[index]"


def test_subtree_wire_roundtrip_can_contain_pattern_end():
    payload = "<repeat_pat>long_identifier<pat_end>"
    encoded = (
        "x="
        + "<subtree_ref_0>"
        + payload
        + SUBTREE_END
        + ";y=<subtree_ref_0>"
    )
    assert invert_subtree_pass(encoded) == "x=" + payload + ";y=" + payload


def test_subtree_rejects_forward_reference():
    try:
        invert_subtree_pass("x=<subtree_ref_0>")
    except ValueError as exc:
        assert "precedes" in str(exc)
    else:
        raise AssertionError("forward subtree reference was accepted")


def test_subtree_special_budget_includes_one_control_token():
    assert len(subtree_reference_tokens(8)) == 9
    assert SUBTREE_BIND not in subtree_reference_tokens(8)
    assert SUBTREE_END in subtree_reference_tokens(8)


def test_disabled_subtree_pass_is_identity():
    text = "x=values[index]<line_sep>y=values[index]"
    encoded, stats = apply_subtree_pass(
        "x=values[index]\ny=values[index]\n",
        text,
        slots=0,
        count_fn=count_atomic,
    )
    assert encoded == text
    assert stats["skipped"] == "disabled"


def test_subtree_binds_repeated_final_wire_window():
    payload = "<repeat_pat>long_identifier<pat_end>"
    text = payload + payload
    specials = subtree_reference_tokens(8) | {"<repeat_pat>", "<pat_end>"}

    def count_wire(value: str) -> int:
        total = 0
        cursor = 0
        ordered = sorted(specials, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in ordered if value.startswith(item, cursor)), None)
            total += 1
            cursor += len(token) if token is not None else 1
        return total

    encoded, stats = apply_subtree_pass(
        "first_statement\nsecond_statement\n",
        text,
        slots=8,
        count_fn=count_wire,
    )
    assert SUBTREE_BIND not in encoded
    assert SUBTREE_END in encoded
    assert "<pat_end>" in encoded
    assert stats["groups_bound"] == 1
    assert invert_subtree_pass(encoded) == text

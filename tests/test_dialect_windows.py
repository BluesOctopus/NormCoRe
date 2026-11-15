"""Mine shorthands on normalized units, not isolated source windows."""

from __future__ import annotations

from normcore_codec.reversible.literals import mask_literals
from normcore_codec.normalization import apply_normalization
from normcore_codec.shorthanding.mining import (
    generalize,
    iter_template_instances,
    split_dialect_units,
    strip_trailing_layout,
)

FOR_RANGE = """\
class Solution:
    def foo(self, nums):
        n = len(nums)
        ans = 0
        for i in range(n):
            ans += nums[i]
        return ans
"""

WHILE_ASSIGN = """\
class Solution:
    def foo(self, lists):
        length = len(lists)
        factor = 2
        while factor < length:
            factor *= 2
        return factor
"""

ADJACENT_FOR_LEN = """\
class Solution:
    def foo(self, x):
        if not x:
            return 0
        n = len(x)
        for i in range(len(x)):
            x[i] += 1
        return n
"""


def _mid(source: str) -> str:
    masked, _ = mask_literals(source, fstring_mode="segment")
    mid, _ = apply_normalization(masked, band="aggressive")
    return (mid or "").strip()


def _templates(source: str, *, max_window: int = 8, max_holes: int = 6) -> set[str]:
    return {
        template
        for template, dens, _, _ in iter_template_instances(
            source,
            band="aggressive",
            policy=None,
            types=None,
            max_window=max_window,
            max_holes=max_holes,
        )
        if dens in _mid(source)
    }


def test_units_concatenate_to_mid_full():
    for source in (FOR_RANGE, WHILE_ASSIGN, ADJACENT_FOR_LEN):
        mid = _mid(source)
        units = split_dialect_units(mid)
        assert "".join(units) == mid
        assert any(u.startswith("<for_stmt>") or u.startswith("<while_stmt>") for u in units)
        heads = [u for u in units if u.startswith("<for_stmt>") or u.startswith("<while_stmt>")]
        assert all(h.endswith("<block_start>") for h in heads)


def test_compound_head_is_its_own_unit():
    mid = _mid(FOR_RANGE)
    units = split_dialect_units(mid)
    assert "<for_stmt>i range(n)<block_start>" in units
    assert "n=len(nums)<line_sep>" in units
    # Neighboring assign has no line_sep before the for-head (postprocess ate it).
    assert "ans=0" in units


def test_windows_cross_block_end_to_next_stmt():
    mid = _mid(FOR_RANGE)
    units = split_dialect_units(mid)
    joined = "".join(units)
    assert "ans<augadd>nums[i]<block_end><line_sep><return>ans" in joined
    i = units.index("ans<augadd>nums[i]")
    assert units[i + 1] == "<block_end>"
    assert units[i + 2] == "<line_sep>"
    assert units[i + 3] == "<return>ans"


def test_trailing_layout_merges_assign_shapes():
    assert strip_trailing_layout("x=0<line_sep>") == "x=0"
    assert strip_trailing_layout("x=0<block_end>") == "x=0"
    assert strip_trailing_layout("<for_stmt>i range(n)<block_start>") == (
        "<for_stmt>i range(n)<block_start>"
    )
    a, _ = generalize(strip_trailing_layout("ans=0<line_sep>"))
    b, _ = generalize(strip_trailing_layout("ans=0<block_end>"))
    c, _ = generalize("ans=0")
    assert a == b == c == "__H0__=0"


def test_for_head_and_adjacent_patterns_are_mined():
    templates = _templates(ADJACENT_FOR_LEN)
    assert "<for_stmt>__H0__ range(len(__H1__))<block_start>" in templates
    assert "__H0__=len(__H1__)<for_stmt>__H2__ range(len(__H1__))<block_start>" in templates
    assert "<if_stmt><not>__H0__<block_start><return>0" in templates


def test_consecutive_assign_survives_while_neighbor():
    """Isolated projection used to drop this because the window's trailing
    <line_sep> is eaten in the full file before <while_stmt>."""
    templates = _templates(WHILE_ASSIGN)
    assert any(
        t.startswith("__H0__=len(__H1__)<line_sep>__H2__=2")
        or t == "__H0__=len(__H1__)<line_sep>__H2__=2"
        for t in templates
    )
    assert any(t.startswith("<while_stmt>") and t.endswith("<block_start>") for t in templates)


def test_every_mined_dens_is_a_mid_full_substring():
    for source in (FOR_RANGE, WHILE_ASSIGN, ADJACENT_FOR_LEN):
        mid = _mid(source)
        rows = iter_template_instances(
            source, band="aggressive", policy=None, types=None, max_window=8, max_holes=6
        )
        assert rows
        for template, dens, _, _ in rows:
            assert dens in mid
            assert "<anchor_sep>" not in dens
            assert not template.endswith("<line_sep>")
            assert not template.endswith("<block_end>")


EXPR_SURFACE = """\
class Solution:
    def foo(self, nums, x):
        ans = []
        ans.append(x)
        node = node.next
        node.next = node.next.next
        v = nums[i]
        w = nums[i][j]
        return [k for k in nums]
"""


def test_expression_mine_skips_param_annotations():
    source = '''\
from typing import List
class Solution:
    def foo(self, nums: List[int]) -> int:
        return nums[0][1]
'''
    templates = _templates(source)
    assert any("[0][1]" in t or t == "__H0__[__H1__][__H2__]" for t in templates)
    assert not any(t.startswith("<t_") and t.count("<") == 1 for t in templates)
    assert not any(t.startswith("self ") and ":<t_" in t for t in templates)


def test_expression_mine_keeps_call_attr_skeleton():
    templates = _templates(EXPR_SURFACE)
    assert "__H0__.append(__H1__)" in templates
    assert "__H0__.next" in templates
    assert "__H0__[__H1__]" in templates
    assert "__H0__[__H1__][__H2__]" in templates
    assert any("<for>" in t for t in templates)
    assert all(".append(" not in t or ".append(__" in t for t in templates)
    assert not any(t.startswith("Solution(") for t in templates)

"""Pass A rule catalog: base (always on) vs optional (GCA global per-rule gate).

Paper framing
-------------
- **Base**: minimal untouchable skeleton (block/module/import-alias glue, match, …).
- **Optional**: attn-gateable rules, including former structural sugar promoted for
  GCA narrative closure (see ``PASS_A_PROMOTION`` / ``CONFIG/pass_a_promote_v1.yaml``).

Attention maps each optional rule → a category in ``rule_attention_scores``;
if that category is protected (a > τ or ``protect_categories_extra``), the rule is
disabled corpus-wide.

Promotion batches (base → optional), ranked by HE leave-one-out rate contribution:
  - batch1 (lower risk): Else, While, ImportFrom, Return
  - batch2 (narrative core): If, For, FunctionDef, Parameters
Kept base on purpose: IndentedBlock, Module, ClassDef, Import, Match*, Try glue, …
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from recode_codec.structural.rules import TRANSFORM_RULES


@dataclass(frozen=True)
class RuleSpec:
    node: str  # rule id (LibCST class name or booster id)
    tier: str  # "base" | "optional"
    category: str  # primary attention category
    note: str = ""
    kind: str = "transform"  # "transform" | "post" (after densify)
    default_off: bool = False
    also_protect: tuple[str, ...] = field(default_factory=tuple)


# Explicit catalog (must cover every key in TRANSFORM_RULES).
_RULE_SPECS: list[RuleSpec] = [
    # ---- base ----
    # ---- base (untouchable skeleton / glue) ----
    RuleSpec("Module", "base", "expr", "file wrapper"),
    RuleSpec("IndentedBlock", "base", "function_def", "block markers — keep always"),
    RuleSpec("SimpleStatementSuite", "base", "expr", "one-line suite"),
    RuleSpec("ClassDef", "base", "class_def", "class spine — keep always"),
    RuleSpec("With", "base", "expr"),
    RuleSpec("Try", "base", "expr"),
    RuleSpec("ExceptHandler", "base", "expr"),
    RuleSpec("Finally", "base", "expr"),
    RuleSpec("Match", "base", "expr"),
    RuleSpec("MatchCase", "base", "expr"),
    RuleSpec("MatchList", "base", "expr"),
    RuleSpec("MatchTuple", "base", "expr"),
    RuleSpec("MatchClass", "base", "expr"),
    RuleSpec("MatchMapping", "base", "expr"),
    RuleSpec("MatchOr", "base", "expr"),
    RuleSpec("Import", "base", "import", "import spine — keep always"),
    RuleSpec("ImportAlias", "base", "import"),
    RuleSpec("AsName", "base", "import"),
    RuleSpec("Raise", "base", "expr"),
    RuleSpec("Pass", "base", "expr"),
    RuleSpec("Break", "base", "loop"),
    RuleSpec("Continue", "base", "loop"),
    RuleSpec("Semicolon", "base", "expr"),
    # ---- base: keep narrative-core structural until RT-safe spine/decode ----
    RuleSpec("FunctionDef", "base", "function_def", "batch2 pending RT — spine alone insufficient"),
    RuleSpec("If", "base", "if_stmt", "batch2 pending RT — needs _if_spine + decode"),
    RuleSpec("For", "base", "loop", "batch2 pending RT — needs _for_spine + in-keyword contract"),
    # promoted / gateable
    RuleSpec("Parameters", "optional", "parameters", "PROMOTE batch2 — paren/comma sugar; RT-safe alone"),
    RuleSpec("Else", "optional", "if_stmt", "PROMOTE batch1"),
    RuleSpec("While", "base", "loop", "decode couples with prior <block_start>; not spine-safe yet"),
    RuleSpec("ImportFrom", "optional", "import", "PROMOTE batch1"),
    RuleSpec("Return", "base", "function_def", "decode needs <return>; not safely gateable yet"),
    # ---- optional: legacy expression sugar (often GPT-4 rate-negative) ----
    RuleSpec("Comparison", "optional", "expr", "cmp ops → markers"),
    RuleSpec("AugAssign", "optional", "assign", "+= etc"),
    RuleSpec("BooleanOperation", "optional", "expr", "and/or markers"),
    RuleSpec("BinaryOperation", "optional", "expr", "arith/bit ops"),
    RuleSpec("UnaryOperation", "optional", "expr"),
    RuleSpec("Ellipsis", "optional", "expr"),
    RuleSpec("Name", "optional", "expr", "True/False/None markers"),
    RuleSpec("Call", "optional", "expr", "call punctuation sugar"),
    RuleSpec("List", "optional", "expr"),
    RuleSpec("Tuple", "optional", "expr"),
    RuleSpec("Set", "optional", "expr"),
    RuleSpec("Dict", "optional", "expr"),
    RuleSpec("Subscript", "optional", "expr"),
    RuleSpec("CompFor", "optional", "expr", "comprehension for — NOT loop stmt (avoid gating with While/For)"),
    RuleSpec("IfExp", "optional", "expr", "ternary — NOT if_stmt (avoid gating with Else/If)"),
    RuleSpec("Lambda", "optional", "function_def"),
    RuleSpec("Await", "optional", "expr"),
    RuleSpec("Yield", "optional", "expr"),
    RuleSpec("ConcatenatedString", "optional", "expr"),
    RuleSpec("Assert", "optional", "expr"),
    RuleSpec("Del", "optional", "expr"),
    RuleSpec("Global", "optional", "expr"),
    RuleSpec("Nonlocal", "optional", "expr"),
    # ---- optional post-densify ----
    RuleSpec(
        "type_pack",
        "optional",
        "function_def",
        "List[int]→<t_…> after densify",
        kind="post",
        also_protect=("class_def",),
    ),
]

POST_OPTIONAL_IDS = frozenset(s.node for s in _RULE_SPECS if s.kind == "post")
TRANSFORM_OPTIONAL_IDS = frozenset(
    s.node for s in _RULE_SPECS if s.tier == "optional" and s.kind == "transform"
)


def _index() -> dict[str, RuleSpec]:
    return {s.node: s for s in _RULE_SPECS}


RULE_INDEX: dict[str, RuleSpec] = _index()


def validate_catalog_covers_transform_rules() -> list[str]:
    live = {t.__name__ for t in TRANSFORM_RULES}
    catalog_transform = {s.node for s in _RULE_SPECS if s.kind == "transform"}
    missing = sorted(live - catalog_transform)
    extra = sorted(catalog_transform - live)
    return [f"missing:{m}" for m in missing] + [f"extra:{e}" for e in extra]


def iter_specs(tier: str | None = None, *, kind: str | None = None) -> list[RuleSpec]:
    rows = list(_RULE_SPECS)
    if tier is not None:
        rows = [s for s in rows if s.tier == tier]
    if kind is not None:
        rows = [s for s in rows if s.kind == kind]
    return rows


def count_tiers() -> dict[str, int]:
    base = sum(1 for s in _RULE_SPECS if s.tier == "base")
    opt = sum(1 for s in _RULE_SPECS if s.tier == "optional")
    opt_tf = sum(1 for s in _RULE_SPECS if s.tier == "optional" and s.kind == "transform")
    opt_post = sum(1 for s in _RULE_SPECS if s.tier == "optional" and s.kind == "post")
    return {
        "base": base,
        "optional": opt,  # unified: transform + post
        "optional_transform": opt_tf,
        "optional_post": opt_post,
        "transform_total": base + opt_tf,  # LibCST rows only
        "gateable_total": opt,
    }


def decide_optional_rules(
    protected: Iterable[str],
    *,
    force_disable: Iterable[str] | None = None,
    force_enable: Iterable[str] | None = None,
    category_bands: dict[str, str] | None = None,
    low_unlock_default_off: bool = True,
) -> dict[str, Any]:
    """Global per-rule gate for all optional rules (LibCST + post).

    Two modes:
      - legacy: ``protected`` only (H = protected, else M-like)
      - tiers: ``category_bands`` maps category → H|M|L
          H: disable
          M: enable unless default_off
          L: enable; unlock default_off on primary category (more aggressive)
    """
    prot = set(protected)
    force_off = set(force_disable or [])
    force_on = set(force_enable or [])
    bands = category_bands

    enabled: list[str] = []
    disabled: list[str] = []
    decisions: dict[str, dict] = {}

    for spec in iter_specs("optional"):
        primary_band = (bands or {}).get(spec.category, "M") if bands else None
        also_bands = [(bands or {}).get(c, "M") for c in spec.also_protect] if bands else []

        if spec.node in force_off:
            on, reason = False, "force_disable"
        elif spec.node in force_on:
            on, reason = True, "force_enable"
        elif bands is not None:
            # any H among primary+also → protect skeleton
            if primary_band == "H" or "H" in also_bands:
                on, reason = False, f"tier_H:{spec.category}/{primary_band}"
            elif spec.default_off:
                if low_unlock_default_off and primary_band == "L":
                    on, reason = True, "tier_L_unlock_default_off"
                else:
                    on, reason = False, f"default_off(tier={primary_band})"
            else:
                # M or L without default_off → on
                on, reason = True, f"tier_{primary_band}"
        elif spec.default_off:
            on, reason = False, "default_off"
        elif spec.category in prot or any(c in prot for c in spec.also_protect):
            hit = [spec.category] if spec.category in prot else []
            hit += [c for c in spec.also_protect if c in prot]
            on, reason = False, f"protected_category:{','.join(hit)}"
        else:
            on, reason = True, "category_ok"

        decisions[spec.node] = {
            "enabled": on,
            "category": spec.category,
            "also_protect": list(spec.also_protect),
            "kind": spec.kind,
            "category_band": primary_band,
            "reason": reason,
            "note": spec.note,
        }
        (enabled if on else disabled).append(spec.node)

    return {
        "n_optional": len(enabled) + len(disabled),
        "n_enabled": len(enabled),
        "n_disabled": len(disabled),
        "enabled": sorted(enabled),
        "disabled": sorted(disabled),
        "enabled_transform": sorted(x for x in enabled if x in TRANSFORM_OPTIONAL_IDS),
        "disabled_transform": sorted(x for x in disabled if x in TRANSFORM_OPTIONAL_IDS),
        "enabled_post": sorted(x for x in enabled if x in POST_OPTIONAL_IDS),
        "disabled_post": sorted(x for x in disabled if x in POST_OPTIONAL_IDS),
        "decisions": decisions,
    }



def active_transform_types(
    *,
    band: str,
    enabled_optional: set[str] | None = None,
) -> set[type]:
    """LibCST types whose TRANSFORM_RULES should fire for this band."""
    if band != "gca" or enabled_optional is None:
        return set(TRANSFORM_RULES.keys())

    active: set[type] = set()
    for node_t in TRANSFORM_RULES:
        name = node_t.__name__
        spec = RULE_INDEX.get(name)
        if spec is None or spec.tier == "base":
            active.add(node_t)
        elif name in enabled_optional:
            active.add(node_t)
    return active

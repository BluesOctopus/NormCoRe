"""Scheme A: map patterns → attention a, reweight Pass-B rewards by g(1−a)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_MARKER_TO_KIND: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<if_stmt>|<elif_stmt>|<else_stmt>"), "if"),
    (re.compile(r"<for_stmt>|<for>"), "for"),
    (re.compile(r"<while_stmt>"), "while"),
    (re.compile(r"<with_stmt>"), "with"),
    (re.compile(r"<try_stmt>|<except|<finally"), "try"),
    (re.compile(r"<def_stmt>|<async_def"), "function_def"),
    (re.compile(r"<class_stmt>"), "class_def"),
    (re.compile(r"<return>"), "return"),
    (re.compile(r"<raise>"), "raise"),
    (re.compile(r"<assert>"), "assert"),
    (re.compile(r"<pass>"), "pass"),
    (re.compile(r"<continue>"), "continue"),
    (re.compile(r"<break>"), "break"),
    (re.compile(r"<import_"), "import"),
    (re.compile(r"<aug"), "augassign"),
]

_KIND_FIELD_FALLBACK = {
    "stmt": "assign",
    "stmt_head": "for",
    "expr": "call_expr",
}

_HOLE_RE = re.compile(r"__H\d+__")
_NUM_RE = re.compile(r"\b\d+\b")


def load_attention_profile(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"attention profile not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    # prefer StarCoder aggregate (main calibration corpus)
    block = data.get("starcoder") or data.get("humaneval_nodoc_diag") or data
    agg = block.get("aggregate") if isinstance(block, dict) else None
    if not isinstance(agg, dict):
        raise ValueError(f"no aggregate in profile: {p}")
    return {
        "path": str(p),
        "model": block.get("model") if isinstance(block, dict) else None,
        "stmt_kind": agg.get("stmt_kind") or {},
        "token_bucket": agg.get("token_bucket") or {},
        "stmt_band": agg.get("stmt_band") or {},
        "token_band": agg.get("token_band") or {},
        "n_ok": agg.get("n_ok", 0),
    }


def _means(table: dict[str, dict]) -> dict[str, float]:
    return {k: float(v["mean"]) for k, v in table.items() if v.get("n", 0) > 0}


def _minmax_norm(means: dict[str, float]) -> dict[str, float]:
    if not means:
        return {}
    lo, hi = min(means.values()), max(means.values())
    if hi <= lo:
        return {k: 0.5 for k in means}
    return {k: (v - lo) / (hi - lo) for k, v in means.items()}


def build_attention_tables(profile: dict[str, Any]) -> dict[str, Any]:
    stmt_a = _minmax_norm(_means(profile["stmt_kind"]))
    tok_a = _minmax_norm(_means(profile["token_bucket"]))
    return {"stmt_a": stmt_a, "token_a": tok_a, "profile": profile}


def infer_stmt_kinds(pattern: str, kind_field: str | None = None) -> list[str]:
    kinds: list[str] = []
    for rx, kind in _MARKER_TO_KIND:
        if rx.search(pattern):
            kinds.append(kind)
    if kinds:
        return kinds
    if kind_field and kind_field in _KIND_FIELD_FALLBACK:
        return [_KIND_FIELD_FALLBACK[kind_field]]
    return ["_outside_stmt"]


def infer_token_mix(pattern: str) -> dict[str, float]:
    """Rough token-bucket weights from dialect template text."""
    holes = len(_HOLE_RE.findall(pattern))
    nums = len(_NUM_RE.findall(pattern))
    # strip markers / holes for leftover literals
    stripped = _HOLE_RE.sub(" ", pattern)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    idents = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped))
    punct = len(re.findall(r"[()\[\]{},.:;+\-*/%@]", stripped))
    weights = {
        "identifier": float(holes + idents),
        "number": float(nums),
        "punct": float(punct),
        "keyword": 1.0 if re.search(r"<(if|for|while|return|raise|pass|with)_", pattern) else 0.0,
        "other": 1.0,
    }
    s = sum(weights.values()) or 1.0
    return {k: v / s for k, v in weights.items()}


def pattern_attention(
    pattern: str,
    tables: dict[str, Any],
    *,
    kind_field: str | None = None,
    stmt_weight: float = 0.7,
) -> float:
    """Return a ∈ [0,1]; higher = more attended → less aggressive compression."""
    stmt_a: dict[str, float] = tables["stmt_a"]
    tok_a: dict[str, float] = tables["token_a"]
    kinds = infer_stmt_kinds(pattern, kind_field)
    stmt_vals = [stmt_a.get(k, 0.5) for k in kinds]
    a_stmt = sum(stmt_vals) / len(stmt_vals)

    mix = infer_token_mix(pattern)
    a_tok = sum(mix.get(b, 0.0) * tok_a.get(b, 0.5) for b in mix)
    # if no token table, fall back
    if not tok_a:
        a_tok = 0.5
    a = stmt_weight * a_stmt + (1.0 - stmt_weight) * a_tok
    return max(0.0, min(1.0, float(a)))


def g_one_minus_a(a: float, *, gamma: float = 1.0, floor: float = 0.15) -> float:
    """g(1−a): low attention → larger multiplier. floor keeps high-a patterns alive."""
    return max(floor, (1.0 - a) ** gamma)


def exponential_reward(reward: int | float, importance: float, *, lambda_: float) -> float:
    """Canonical Figure-GCA reward: R_hat = R * exp(-lambda * I).

    ``importance`` is the persisted category importance normalized to [0, 1].
    The raw attention average is retained separately for provenance.
    """

    if not 0.0 <= importance <= 1.0:
        raise ValueError("normalized importance must be in [0, 1]")
    if lambda_ < 0:
        raise ValueError("lambda must be non-negative")
    return float(reward) * math.exp(-float(lambda_) * float(importance))


def reweight_reward(
    reward: int | float,
    a: float,
    *,
    mode: str = "reweight",
    gamma: float = 1.0,
    floor: float = 0.15,
    beta: float = 1.0,
    lambda_: float = 1.0,
    ban_above: float | None = None,
) -> int:
    """Map base reward → attention-aware reward.

    - ``exponential``: reward × exp(−λa) — canonical NormCoRe policy
    - ``reweight``: reward × g(1−a) — historical profile only
    - ``boost``: reward × (1 + β(1−a))  — low-attn rises above base (stronger DP tilt)
    """
    if ban_above is not None and a >= ban_above:
        return 0
    if mode == "exponential":
        return max(0, int(round(exponential_reward(reward, a, lambda_=lambda_))))
    if mode == "boost":
        w = 1.0 + beta * (1.0 - a)
    else:
        w = g_one_minus_a(a, gamma=gamma, floor=floor)
    return max(0, int(round(float(reward) * w)))


def annotate_registry(
    patterns: list[dict],
    tables: dict[str, Any],
    *,
    mode: str = "reweight",
    gamma: float = 1.0,
    floor: float = 0.15,
    beta: float = 1.0,
    ban_above: float | None = None,
) -> list[dict]:
    """Return new pattern rows with attention / reweighted reward (original kept)."""
    out: list[dict] = []
    for row in patterns:
        pattern = str(row.get("pattern", ""))
        base = int(row.get("reward", 1))
        a = pattern_attention(pattern, tables, kind_field=row.get("kind"))
        new_r = reweight_reward(
            base, a, mode=mode, gamma=gamma, floor=floor, beta=beta, ban_above=ban_above
        )
        nr = dict(row)
        nr["attention"] = round(a, 6)
        nr["reward_base"] = base
        if ban_above is not None and a >= ban_above:
            nr["reward"] = 0
            nr["banned"] = True
        else:
            nr["reward"] = new_r if new_r > 0 else max(1, base)
        out.append(nr)
    return out

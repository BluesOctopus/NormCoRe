"""GCA: Global Context-aware Attention for Pass A and Pass B.

Hot-path compress must NOT call a model. Build assets via ``scripts/build_assets.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from normcore_codec.attention.weight import exponential_reward, g_one_minus_a, reweight_reward


@dataclass
class GCAPolicy:
    """Frozen decisions consumed by ``structural_band="gca"`` and registry build."""

    protected: set[str]
    scores: dict[str, float]
    threshold: float
    enable_type_pack: bool
    enable_aggressive_idioms: bool = False  # removed mechanism; kept for JSON compat (always False)
    enabled_optional_rules: set[str] = field(default_factory=set)
    disabled_optional_rules: set[str] = field(default_factory=set)
    per_rule_gate: bool = True
    scores_path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        tiers = (self.meta or {}).get("tiers") or {}
        return {
            "mode": "gca",
            "protected": sorted(self.protected),
            "threshold": self.threshold,
            "scores_path": self.scores_path,
            "scores": self.scores,
            "per_rule_gate": self.per_rule_gate,
            "tiers": {
                "enabled": bool(tiers.get("enabled")),
                "high_threshold": tiers.get("high_threshold"),
                "low_threshold": tiers.get("low_threshold"),
                "band_sets": tiers.get("band_sets"),
            },
            "pass_a_rules": {
                "enabled_optional": sorted(self.enabled_optional_rules),
                "disabled_optional": sorted(self.disabled_optional_rules),
                "n_enabled_optional": len(self.enabled_optional_rules),
                "n_disabled_optional": len(self.disabled_optional_rules),
                "type_pack": self.enable_type_pack,
            },
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GCAPolicy":
        rules = data.get("pass_a_rules") or {}
        boost = data.get("boosters") or {}
        enabled = set(rules.get("enabled_optional") or [])
        disabled = set(rules.get("disabled_optional") or [])
        # idioms removed from catalog — strip if present in old policies
        enabled.discard("aggressive_idioms")
        disabled.discard("aggressive_idioms")
        if "type_pack" in rules:
            enable_tp = bool(rules["type_pack"])
        elif boost:
            enable_tp = bool(boost.get("type_pack", True))
        else:
            enable_tp = "type_pack" in enabled
        if enable_tp:
            enabled.add("type_pack")
            disabled.discard("type_pack")
        else:
            disabled.add("type_pack")
            enabled.discard("type_pack")
        return cls(
            protected=set(data.get("protected") or []),
            scores={k: float(v) for k, v in (data.get("scores") or {}).items()},
            threshold=float(data.get("threshold", 0.0008)),
            enable_type_pack=enable_tp,
            enable_aggressive_idioms=False,
            enabled_optional_rules=enabled,
            disabled_optional_rules=disabled,
            per_rule_gate=bool(data.get("per_rule_gate", True)),
            scores_path=str(data.get("scores_path") or ""),
            meta=dict(data.get("meta") or {}),
        )


def load_category_scores(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"category attention scores not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    # allow {"scores": {...}} or flat dict
    if isinstance(raw, dict) and "scores" in raw and isinstance(raw["scores"], dict):
        raw = raw["scores"]
    return {str(k): float(v) for k, v in raw.items()}


def protected_categories(scores: dict[str, float], threshold: float) -> set[str]:
    return {k for k, v in scores.items() if v > threshold}


def assign_category_bands(
    scores: dict[str, float],
    *,
    high_threshold: float,
    low_threshold: float,
) -> dict[str, str]:
    """Map each category → H / M / L by raw attention score.

    H: score >= high_threshold  (do not compress / drop rules)
    L: score <  low_threshold   (more aggressive)
    M: otherwise                (original optional logic)
    """
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must be <= high_threshold")
    out: dict[str, str] = {}
    for k, v in scores.items():
        fv = float(v)
        if fv >= high_threshold:
            out[k] = "H"
        elif fv < low_threshold:
            out[k] = "L"
        else:
            out[k] = "M"
    return out


def band_sets(bands: dict[str, str]) -> dict[str, list[str]]:
    return {
        "H": sorted(k for k, b in bands.items() if b == "H"),
        "M": sorted(k for k, b in bands.items() if b == "M"),
        "L": sorted(k for k, b in bands.items() if b == "L"),
    }


def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def decide_booster(
    protected: set[str],
    *,
    disable_if_protected: list[str] | None,
    force_enabled: bool | None = None,
) -> bool:
    """Return True if booster should be ON globally."""
    if force_enabled is False:
        return False
    if force_enabled is True and not disable_if_protected:
        return True
    blocked = set(disable_if_protected or [])
    if protected & blocked:
        return False
    if force_enabled is True:
        return True
    return True


def build_policy_from_attention_cfg(
    attn_cfg: dict[str, Any],
    *,
    scores: dict[str, float] | None = None,
    scores_path: str | Path | None = None,
) -> GCAPolicy:
    """Build policy from CONFIG ``attention:`` block.

    Pass A uses one gate: optional rule ids (LibCST + type_pack + idioms).
    With ``tiers.enabled``, categories split into H/M/L:
      H — disable optional Pass A rules; strongly downweight Pass B candidates
      M — retain ordinary optional rules and shorthand candidates
      L — retain optional rules and weakly penalize Pass B candidates
    """
    from normcore_codec.structural.rule_catalog import (
        count_tiers,
        decide_optional_rules,
        iter_specs,
        validate_catalog_covers_transform_rules,
    )

    sp = scores_path or attn_cfg.get("scores_path")
    sc = scores if scores is not None else load_category_scores(sp)
    thr = float(attn_cfg.get("protect_threshold", attn_cfg.get("category_threshold", 0.0008)))
    prot = protected_categories(sc, thr)
    pass_a = attn_cfg.get("pass_a") or {}
    # Explicit extras: promote high-value rules whose raw scores sit below τ.
    extra = attn_cfg.get("protect_categories_extra") or pass_a.get("protect_categories_extra") or []
    prot |= {str(x) for x in extra}

    tiers_cfg = attn_cfg.get("tiers") or {}
    tiers_on = bool(tiers_cfg.get("enabled", False))
    high_thr = float(tiers_cfg.get("high_threshold", thr))
    low_thr = float(tiers_cfg.get("low_threshold", 0.00035))
    bands = assign_category_bands(sc, high_threshold=high_thr, low_threshold=low_thr) if tiers_on else None
    if bands is not None:
        # H categories are the protected set under three-tier mode
        prot = {k for k, b in bands.items() if b == "H"}
        prot |= {str(x) for x in extra}

    per_rule = bool(pass_a.get("per_rule_gate", True))

    if per_rule:
        opt = decide_optional_rules(
            prot,
            force_disable=pass_a.get("force_disable_rules"),
            force_enable=pass_a.get("force_enable_rules"),
            category_bands=bands,
            low_unlock_default_off=bool(tiers_cfg.get("low_unlock_default_off", True)),
        )
        enabled_opt = set(opt["enabled"])
        disabled_opt = set(opt["disabled"])
    else:
        enabled_opt = set()
        disabled_opt = set()
        for s in iter_specs("optional"):
            if s.default_off:
                disabled_opt.add(s.node)
            else:
                enabled_opt.add(s.node)
        opt = {
            "n_enabled": len(enabled_opt),
            "n_disabled": len(disabled_opt),
            "enabled": sorted(enabled_opt),
            "disabled": sorted(disabled_opt),
            "enabled_post": sorted(x for x in enabled_opt if x == "type_pack"),
            "disabled_post": sorted(x for x in disabled_opt if x == "type_pack"),
        }

    enable_tp = "type_pack" in enabled_opt

    pb = attn_cfg.get("pass_b") or {}
    return GCAPolicy(
        protected=prot,
        scores=sc,
        threshold=thr if not tiers_on else high_thr,
        enable_type_pack=enable_tp,
        enable_aggressive_idioms=False,
        enabled_optional_rules=enabled_opt,
        disabled_optional_rules=disabled_opt,
        per_rule_gate=per_rule,
        scores_path=str(sp or ""),
        meta={
            "pass_a_cfg": pass_a,
            "tiers": {
                "enabled": tiers_on,
                "high_threshold": high_thr,
                "low_threshold": low_thr,
                "bands": bands,
                "band_sets": band_sets(bands) if bands else None,
                "low_unlock_default_off": bool(tiers_cfg.get("low_unlock_default_off", True)),
                "pass_b_l_mode": str(pb.get("l_mode", "boost")),
                "pass_b_boost_beta": float(pb.get("boost_beta", 1.0)),
            },
            "pass_a_rule_counts": count_tiers(),
            "pass_a_optional_gate": {
                "n_enabled": opt.get("n_enabled"),
                "n_disabled": opt.get("n_disabled"),
                "disabled": opt.get("disabled"),
                "enabled_post": opt.get("enabled_post"),
                "disabled_post": opt.get("disabled_post"),
            },
            "catalog_coverage": validate_catalog_covers_transform_rules(),
            "pass_b_reweight": bool(pb.get("reweight", True)),
            "formula": str(pb.get("formula", "exp_neg_lambda_i")),
            "lambda": float(pb.get("lambda", 2.0)),
        },
    )


def load_policy(path: str | Path) -> GCAPolicy:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GCA policy not found: {p}. Run: python scripts/build_assets.py"
        )
    return GCAPolicy.from_json(json.loads(p.read_text(encoding="utf-8")))


def save_policy(policy: GCAPolicy, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(policy.to_json(), indent=2), encoding="utf-8")
    return p


def pattern_category(row: dict[str, Any]) -> str:
    return str(row.get("node_type") or row.get("kind") or "expr")


def build_gca_registry(
    patterns: list[dict[str, Any]],
    scores: dict[str, float],
    *,
    threshold: float,
    hard_protect: bool = False,
    reweight: bool = True,
    formula: str = "exp_neg_lambda_i",
    lambda_: float = 2.0,
    gamma: float = 1.0,
    floor: float = 0.15,
    rank_by: str = "selection_utility",
    top_k: int | None = 300,
    default_a: float = 0.0005,
    category_bands: dict[str, str] | None = None,
    l_mode: str = "boost",
    boost_beta: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic GCA registry.

    Canonical mode uses ``R_hat = R * exp(-lambda * I_norm)`` for runtime DP
    and applies the same factor to the corpus-level selection utility. Historical
    ``g_one_minus_a`` and hard-protect behavior remain available only for an
    explicitly selected legacy profile.
    """
    if category_bands is not None:
        protected = {k for k, b in category_bands.items() if b == "H"} if hard_protect else set()
        soft_demote = (
            {k for k, b in category_bands.items() if b == "H"} if not hard_protect else set()
        )
    else:
        protected = protected_categories(scores, threshold) if hard_protect else set()
        soft_demote = protected_categories(scores, threshold) if not hard_protect else set()
    a_norm = _minmax_norm(scores)
    out: list[dict[str, Any]] = []
    dropped = 0
    n_boost = n_reweight = n_soft_demote = 0

    for row in patterns:
        cat = pattern_category(row)
        band = (category_bands or {}).get(cat, "M")
        # Hard drop only when explicitly requested
        if hard_protect and (cat in protected or band == "H"):
            dropped += 1
            continue
        base = int(row.get("reward", 1))
        a_raw = float(scores.get(cat, default_a))
        a = float(a_norm.get(cat, 0.5))
        nr = dict(row)
        nr["reward_base"] = base
        nr["attention"] = round(a, 6)
        nr["importance_normalized"] = round(a, 6)
        nr["attention_raw"] = a_raw
        nr["importance_raw"] = a_raw
        nr["category"] = cat
        nr["attn_band"] = band
        if reweight:
            if formula == "exp_neg_lambda_i":
                factor = exponential_reward(1.0, a, lambda_=lambda_)
                nr["reward"] = max(1e-9, exponential_reward(base, a, lambda_=lambda_))
                nr["reweight_mode"] = "exp_neg_lambda_i"
                nr["cognitive_factor"] = round(factor, 9)
                n_reweight += 1
            elif band == "L" and l_mode == "boost" and cat not in soft_demote:
                nr["reward"] = reweight_reward(base, a, mode="boost", beta=boost_beta)
                nr["reweight_mode"] = "boost"
                n_boost += 1
            else:
                # Soft-demote H/protected: still in pool, g(1−a) → near floor
                nr["reward"] = reweight_reward(base, a, mode="reweight", gamma=gamma, floor=floor)
                nr["g_one_minus_a"] = round(g_one_minus_a(a, gamma=gamma, floor=floor), 6)
                if cat in soft_demote or band == "H":
                    nr["reweight_mode"] = "soft_demote_H"
                    n_soft_demote += 1
                else:
                    nr["reweight_mode"] = "g_one_minus_a"
                    n_reweight += 1
        else:
            nr["reward"] = base
            nr["reweight_mode"] = "none"
        utility_base = float(row.get("saved_est", base * int(row.get("freq", 1))))
        nr["selection_utility_base"] = utility_base
        nr["selection_utility"] = (
            utility_base * float(nr.get("cognitive_factor", 1.0))
            if formula == "exp_neg_lambda_i" and reweight
            else utility_base
        )
        # Historical field retained only so legacy profiles remain inspectable.
        nr["balanced_score"] = float(nr["reward"]) / max(a_raw, 1e-6)
        out.append(nr)

    if rank_by == "selection_utility":
        out.sort(key=lambda x: x.get("selection_utility", 0), reverse=True)
    elif rank_by == "balanced_score":
        out.sort(key=lambda x: x.get("balanced_score", 0), reverse=True)
    elif rank_by == "reward":
        out.sort(key=lambda x: x.get("reward", 0), reverse=True)

    if top_k is not None and top_k > 0:
        out = out[: int(top_k)]

    stats = {
        "input": len(patterns),
        "dropped_protected": dropped,
        "protected": sorted(protected),
        "soft_demote": sorted(soft_demote),
        "after_filter": len(patterns) - dropped,
        "output": len(out),
        "threshold": threshold,
        "gamma": gamma,
        "floor": floor,
        "formula": formula,
        "lambda": lambda_,
        "rank_by": rank_by,
        "top_k": top_k,
        "reweight": reweight,
        "hard_protect": hard_protect,
        "tiers": bool(category_bands),
        "n_boost_L": n_boost,
        "n_reweight_M": n_reweight,
        "n_soft_demote_H": n_soft_demote,
        "l_mode": l_mode,
        "boost_beta": boost_beta,
        "band_sets": band_sets(category_bands) if category_bands else None,
    }
    return out, stats

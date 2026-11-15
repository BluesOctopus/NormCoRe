"""Validation and provenance checks for NormCoRe codec assets."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from normcore_codec.config import CodecAssets
from normcore_codec.shorthanding.wildcard import _HOLE_RE, WildcardPattern, bare_goal_token
from normcore_codec.reversible.markers import PATTERN_END

_GOAL_RE = re.compile(r"^<[A-Za-z0-9_]+_pat>$")
# A pattern may never consume a literal mask as fixed text: collapsing the span
# would delete the placeholder while ``mask_records`` keeps its entry, and the
# positional ``unmask_literals`` walk would then misalign every later literal in
# the file. Varying literals belong in an ``__S{k}__`` hole, whose capture is
# carried inside the shorthand.
_MASK_SENTINELS = (
    '"MASK"',
    "'MASK'",
    '"""MASK"""',
    "'''MASK'''",
    "#MASK",
    "__AARI_FMASK__",
    "__NORMCORE_FMASK__",
    "__RECODE_FMASK__",
)


class AssetValidationError(ValueError):
    """Raised when assets cannot satisfy the reversible codec contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetValidationError(f"invalid JSON asset: {path}") from exc


def validate_registry(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    rows = data if isinstance(data, list) else data.get("patterns")
    if not isinstance(rows, list):
        raise AssetValidationError(f"registry has no patterns list: {path}")

    ids: set[str] = set()
    goals: set[str] = set()
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: not an object")
            continue
        pid = str(row.get("id", ""))
        pattern = str(row.get("pattern", ""))
        goal = str(row.get("goal") or f"<{pid}_pat>")
        bare_goal = bare_goal_token(goal)
        reward = row.get("reward")
        if not pid or not pattern:
            errors.append(f"row {index}: empty id/pattern")
        if pid in ids:
            errors.append(f"row {index}: duplicate id {pid}")
        if bare_goal in goals:
            errors.append(f"row {index}: duplicate goal {bare_goal}")
        if not _GOAL_RE.fullmatch(bare_goal):
            errors.append(f"row {index}: non-atomic goal {bare_goal}")
        if bare_goal in pattern:
            errors.append(f"row {index}: goal collides with its pattern")
        if any(sentinel in pattern for sentinel in _MASK_SENTINELS):
            errors.append(f"row {index}: pattern consumes a literal mask")
        if PATTERN_END and PATTERN_END in pattern:
            errors.append(f"row {index}: pattern contains the shorthand terminator")
        if _HOLE_RE.search(pattern) and WildcardPattern.from_row(row) is None:
            errors.append(f"row {index}: malformed holes (index gap or mixed kinds)")
        if not isinstance(reward, (int, float)) or reward <= 0:
            errors.append(f"row {index}: non-positive reward")
        ids.add(pid)
        goals.add(bare_goal)

    if errors:
        raise AssetValidationError("; ".join(errors[:20]))
    return {"patterns": len(rows), "goals": sorted(goals)}


@lru_cache(maxsize=8)
def validate_assets(assets: CodecAssets) -> dict[str, Any]:
    """Validate presence, hashes, policy shape, and registry injectivity."""

    required = {
        "pattern_registry": assets.pattern_registry,
        "conditional_policy": assets.conditional_policy,
        "type_table": assets.type_table,
        "grammar_binary": assets.grammar_binary,
    }
    missing = [name for name, path in required.items() if path is None or not path.is_file()]
    if missing:
        raise AssetValidationError(f"missing required assets: {', '.join(missing)}")

    lock: dict[str, Any] = {}
    if assets.asset_lock is not None:
        lock = _load_json(assets.asset_lock)
        files = lock.get("files") or {}
        for name, path in required.items():
            expected = (files.get(name) or {}).get("sha256")
            if expected and sha256_file(path) != expected:
                raise AssetValidationError(f"SHA-256 mismatch for {name}: {path}")

    policy = _load_json(assets.conditional_policy)
    if policy.get("mode") != "conditional" or not policy.get("per_rule_gate", False):
        raise AssetValidationError("canonical policy must enable per-rule Conditional Attention gating")
    if (policy.get("meta") or {}).get("formula") != "exp_neg_lambda_i":
        raise AssetValidationError("canonical policy must declare exp(-lambda*I) weighting")

    registry = validate_registry(assets.pattern_registry)
    return {
        "lock_version": lock.get("schema_version"),
        "registry": registry,
        "policy_mode": policy.get("mode"),
    }

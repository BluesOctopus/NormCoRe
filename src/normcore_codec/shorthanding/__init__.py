"""shorthanding: exact + dialect wildcard patterns + non-overlapping DP (no TokenSugar)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Callable, Iterable

from normcore_codec.shorthanding.wildcard import (
    _HOLE_RE,
    WildcardPattern,
    compact_plain_scope_refs,
    find_wildcard_matches,
    expand_scope_capture_refs,
    prepare_scope_capture_goals,
    invert_wildcard,
)

@dataclass
class PatternEntry:
    pattern_id: str
    reward: float
    goal: str
    pattern: str
    pattern_type: str = "exact"  # exact | wildcard


@lru_cache(maxsize=4)
def load_registry(path: str | Path) -> list[PatternEntry]:
    """Load shorthanding patterns."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"shorthanding registry not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("patterns", [])
    out: list[PatternEntry] = []
    for row in items:
        pattern = str(row.get("pattern", ""))
        pid = str(row.get("id", ""))
        reward = float(row.get("reward", 1))
        if reward <= 0 or not pattern or not pid:
            continue
        has_hole = bool(_HOLE_RE.search(pattern))
        ptype = "wildcard" if (has_hole or row.get("type") == "wildcard") else "exact"
        goal = str(row.get("goal") or f"<{pid}_pat>")
        out.append(PatternEntry(pid, reward, goal, pattern, ptype))
    return out


def _split_registry(
    registry: list[PatternEntry],
) -> tuple[list[PatternEntry], list[WildcardPattern]]:
    exact: list[PatternEntry] = []
    wild: list[WildcardPattern] = []
    for e in registry:
        if e.pattern_type == "wildcard":
            wp = WildcardPattern.from_row(
                {"id": e.pattern_id, "pattern": e.pattern, "goal": e.goal, "reward": e.reward}
            )
            if wp:
                wild.append(wp)
        else:
            exact.append(e)
    return exact, wild


def apply_registry_shorthanding(
    text: str,
    registry: list[PatternEntry],
    *,
    band_paint: list[str] | None = None,
    count_fn: Callable[[str], int] | None = None,
    dialect_hotzones: set[int] | None = None,
    skip_match: Callable[[dict], bool] | None = None,
    scope_capture_slots: int = 0,
) -> tuple[str, dict]:
    """Apply shorthanding DP."""
    if not registry:
        return text, {
            "patterns_applied": 0,
            "dropped_h": 0,
            "reward_mode": "runtime_exact" if count_fn is not None else "registry_average",
            "skipped": "no_registry",
        }
    reg = registry

    exact, wild = _split_registry(reg)
    matches: list[dict] = []

    # Exact substring matches
    for entry in sorted(exact, key=lambda e: len(e.pattern), reverse=True):
        start = 0
        pat = entry.pattern
        need_left_bound = bool(pat) and (pat[0].isalnum() or pat[0] == "_")
        need_right_bound = bool(pat) and (pat[-1].isalnum() or pat[-1] == "_")
        while True:
            i = text.find(pat, start)
            if i < 0:
                break
            if need_left_bound and i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_"):
                start = i + 1
                continue
            right = i + len(pat)
            if need_right_bound and right < len(text) and (
                text[right].isalnum() or text[right] == "_"
            ):
                start = i + 1
                continue
            matches.append(
                {
                    "start": i,
                    "end": i + len(pat),
                    "reward": entry.reward,
                    "goal": entry.goal,
                    "id": entry.pattern_id,
                    "kind": "exact",
                }
            )
            start = i + 1

    n_exact_candidates = len(matches)
    wildcard_matches = find_wildcard_matches(text, wild)
    matches.extend(wildcard_matches)
    n_wildcard_candidates = len(wildcard_matches)
    n_candidates_raw = len(matches)

    scope_stats = {
        "scope_capture_slots": max(0, scope_capture_slots),
        "scope_values_registered": 0,
        "scope_candidate_compact_matches": 0,
        "scope_candidate_reference_uses": 0,
        "scope_candidate_local_tokens_saved": 0,
        "scope_compact_matches": 0,
        "scope_reference_uses": 0,
        "scope_local_tokens_saved": 0,
        "scope_plain_reference_uses": 0,
        "scope_plain_local_tokens_saved": 0,
    }
    if scope_capture_slots > 0 and count_fn is not None:
        scope_stats.update(
            prepare_scope_capture_goals(
                text,
                matches,
                slots=scope_capture_slots,
                count_fn=count_fn,
            )
        )

    n_vetoed = 0
    n_skip_vetoed = 0
    if skip_match is not None:
        valid = []
        for m in matches:
            # Expose the exact normalized surface used for matching so an offline
            # Conditional Attention gate can score the same offsets as the real codec pipeline.
            # The private key keeps the public match schema unchanged for
            # callers that only inspect start/end/goal/id.
            if skip_match({**m, "_text": text}):
                n_vetoed += 1
                n_skip_vetoed += 1
            else:
                valid.append(m)
        matches = valid
    n_hotzone_vetoed = 0
    n_nonpositive_reward = 0
    if count_fn is not None or dialect_hotzones:
        valid_matches = []
        for m in matches:
            if dialect_hotzones and any(
                i in dialect_hotzones for i in range(m["start"], m["end"])
            ):
                n_vetoed += 1
                n_hotzone_vetoed += 1
                continue
            if count_fn is not None:
                original_span = text[m["start"]:m["end"]]
                c_orig = count_fn(original_span)
                c_comp = count_fn(m["goal"])
                real_reward = c_orig - c_comp
                if real_reward > 0:
                    m["registry_reward"] = m["reward"]
                    m["reward"] = real_reward
                    m["tokens_original"] = c_orig
                    m["tokens_compressed"] = c_comp
                    valid_matches.append(m)
                else:
                    n_nonpositive_reward += 1
            else:
                valid_matches.append(m)
        matches = valid_matches

    dropped_h = 0
    if band_paint is not None and matches:
        from normcore_codec.attention.occurrence import majority_band
        kept: list[dict] = []
        for m in matches:
            if majority_band(band_paint, m["start"], m["end"]) == "H":
                dropped_h += 1
                continue
            kept.append(m)
        matches = kept

    funnel = {
        "reward_mode": "runtime_exact" if count_fn is not None else "registry_average",
        "registry_patterns": len(reg),
        "registry_exact": len(exact),
        "registry_wildcard": len(wild),
        "candidate_matches_raw": n_candidates_raw,
        "candidate_exact": n_exact_candidates,
        "candidate_wildcard": n_wildcard_candidates,
        "skip_vetoed": n_skip_vetoed,
        "hotzone_vetoed": n_hotzone_vetoed,
        "nonpositive_reward": n_nonpositive_reward,
        "attention_dropped": dropped_h,
        "matches_after_filters": len(matches),
        # Compatibility keys used by the existing RQ1 runners.
        "n_matches": len(matches),
        "n_vetoed": n_vetoed,
        "dropped_h": dropped_h,
    }

    if not matches:
        out = text
        if scope_capture_slots > 0 and count_fn is not None:
            out, plain_stats = compact_plain_scope_refs(
                text,
                wild,
                exact,
                slots=scope_capture_slots,
                count_fn=count_fn,
            )
            scope_stats.update(plain_stats)
        if count_fn is not None and count_fn(out) >= count_fn(text):
            out = text
            scope_stats.update(
                {
                    "scope_plain_reference_uses": 0,
                    "scope_plain_local_tokens_saved": 0,
                }
            )
        return out, {
            **funnel,
            "patterns_applied": 0,
            "dp_rejected": 0,
            "selected_reward": 0,
            "exact": 0,
            "wildcard": 0,
            "ids": [],
            **scope_stats,
        }

    selected = _dp_select(matches)
    scope_stats.update(
        {
            "scope_compact_matches": sum(
                1 for match in selected if match.get("_scope_refs_used")
            ),
            "scope_reference_uses": sum(
                int(match.get("_scope_refs_used", 0)) for match in selected
            ),
            "scope_local_tokens_saved": sum(
                int(match.get("_scope_tokens_saved", 0)) for match in selected
            ),
        }
    )
    selected.sort(key=lambda m: m["start"], reverse=True)

    out = text
    applied_ids = []
    for m in selected:
        out = out[: m["start"]] + m["goal"] + out[m["end"] :]
        applied_ids.append(m["id"])

    if scope_capture_slots > 0 and count_fn is not None:
        out, plain_stats = compact_plain_scope_refs(
            out,
            wild,
            exact,
            slots=scope_capture_slots,
            count_fn=count_fn,
        )
        scope_stats.update(plain_stats)

    selected_reward = sum(m["reward"] for m in selected)
    global_original_tokens = count_fn(text) if count_fn is not None else None
    global_compressed_tokens = count_fn(out) if count_fn is not None else None
    global_reward = (
        global_original_tokens - global_compressed_tokens
        if global_original_tokens is not None and global_compressed_tokens is not None
        else None
    )
    # BPE token costs are not additive across span boundaries. The DP uses
    # local rewards, so runtime-exact mode needs a whole-buffer safety gate.
    if global_reward is not None and global_reward <= 0:
        return text, {
            **funnel,
            "patterns_applied": 0,
            "dp_rejected": len(matches),
            "selected_reward": selected_reward,
            "selected_original_tokens": sum(m["tokens_original"] for m in selected),
            "selected_compressed_tokens": sum(m["tokens_compressed"] for m in selected),
            "global_original_tokens": global_original_tokens,
            "global_compressed_tokens": global_compressed_tokens,
            "global_reward": global_reward,
            "reward_calibration_error": global_reward - selected_reward,
            "global_fallback": True,
            **scope_stats,
            "ids": [],
            "exact": 0,
            "wildcard": 0,
        }

    return out, {
        **funnel,
        "patterns_applied": len(applied_ids),
        "dp_rejected": len(matches) - len(selected),
        "selected_reward": selected_reward,
        "selected_original_tokens": (
            sum(m["tokens_original"] for m in selected) if count_fn is not None else None
        ),
        "selected_compressed_tokens": (
            sum(m["tokens_compressed"] for m in selected) if count_fn is not None else None
        ),
        "global_original_tokens": global_original_tokens,
        "global_compressed_tokens": global_compressed_tokens,
        "global_reward": global_reward,
        "reward_calibration_error": (
            global_reward - selected_reward if global_reward is not None else None
        ),
        "global_fallback": False,
        **scope_stats,
        "ids": applied_ids,
        "exact": sum(1 for m in selected if m.get("kind") == "exact"),
        "wildcard": sum(1 for m in selected if m.get("kind") == "wildcard"),
    }


def restore_registry_shorthanding(text: str, registry: list[PatternEntry]) -> str:
    if not registry:
        return text
    exact, wild = _split_registry(registry)
    out = expand_scope_capture_refs(text, wild, exact)
    out = invert_wildcard(out, wild)
    ordered = sorted(exact, key=lambda e: len(e.goal), reverse=True)
    for entry in ordered:
        if entry.goal:
            out = out.replace(entry.goal, entry.pattern)
    return out


def _dp_select(matches: list[dict]) -> list[dict]:
    intervals = sorted(matches, key=lambda m: m["end"])
    n = len(intervals)
    if n == 0:
        return []

    def prev(i: int) -> int:
        lo, hi = 0, i - 1
        best = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if intervals[mid]["end"] <= intervals[i]["start"]:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    dp = [0] * n
    take = [False] * n
    for i in range(n):
        skip = dp[i - 1] if i else 0
        p = prev(i)
        use = intervals[i]["reward"] + (dp[p] if p >= 0 else 0)
        if use > skip:
            dp[i] = use
            take[i] = True
        else:
            dp[i] = skip
            take[i] = False

    selected: list[dict] = []
    i = n - 1
    while i >= 0:
        if take[i]:
            selected.append(intervals[i])
            i = prev(i)
        else:
            i -= 1
    return selected

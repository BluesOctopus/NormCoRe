#!/usr/bin/env python3
"""One-shot migration: rewrite a legacy MASK-fixed registry into typed holes.

This is not a miner or a second shorthanding implementation. New registries must be
produced by ``normcore_codec.shorthanding.mining`` from a user-provided source corpus.
Use this script only when converting a historical table so that
``validate_registry``
accepts it.

Legacy registries were mined on masked text, so the templates embed ``MASK``
sentinels as *fixed* text. Applying such a pattern deletes the placeholder and
desynchronises the positional ``unmask_literals`` walk, which is why the mined
registries look strong on paper and fail the AST round trip. Here every mask
sentinel becomes an ``__S{k}__`` hole whose capture rides inside the shorthand,
and identifiers can optionally be generalised into ``__H{k}__`` holes.


Rewards are repriced honestly. A hole's payload is paid for in both encodings,
so the only real saving is the fixed skeleton, minus the goal token, minus the
``;`` separators, minus the ``<pat_end>`` terminator::

    reward = tokens(fixed text) - 1 - (holes - 1) - 1

Anything that does not clear one token is dropped rather than shipped at a
flattering price.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normcore_codec.assets import validate_registry  # noqa: E402
from normcore_codec.reversible.markers import PATTERN_END, all_special_tokens  # noqa: E402

MASK_SENTINEL = re.compile(
    r'"""MASK"""|\'\'\'MASK\'\'\'|"MASK"|\'MASK\'|#MASK|__NORMCORE_FMASK__|__RECODE_FMASK__|__AARI_FMASK__'
)
MARKER = re.compile(r"<[^<>]*>")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HOLE = re.compile(r"__([HS])(\d+)__")

# Names whose identity carries the meaning of the idiom; generalising them would
# turn a recognisable pattern into a shape that matches anything.
PINNED = frozenset(
    {
        "self", "cls", "None", "True", "False", "len", "range", "print", "str", "int",
        "float", "list", "dict", "set", "tuple", "bool", "super", "open", "enumerate",
        "zip", "sorted", "sum", "min", "max", "abs", "type", "isinstance", "hasattr",
        "getattr", "setattr", "pass", "return", "raise", "yield", "format", "join",
        "append", "items", "keys", "values", "get", "split", "strip", "lower", "upper",
    }
)


def token_counter():
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    specials = set(all_special_tokens())

    def count(text: str) -> int:
        markers = [m for m in MARKER.findall(text) if m in specials]
        plain = text
        for marker in set(markers):
            plain = plain.replace(marker, " ")
        return len(encoding.encode(plain)) + len(markers)

    return count


def _marker_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in MARKER.finditer(text)]


def rewrite(pattern: str, *, generalize_identifiers: bool) -> tuple[str, int]:
    """Return the hole template and its hole count, or ``("", 0)`` if unusable."""
    cuts: list[tuple[int, int, str]] = [
        (m.start(), m.end(), "S") for m in MASK_SENTINEL.finditer(pattern)
    ]
    if generalize_identifiers:
        spans = _marker_spans(pattern)
        taken = {(a, b) for a, b, _ in cuts}
        for match in IDENTIFIER.finditer(pattern):
            start, end, name = match.start(), match.end(), match.group(0)
            if name in PINNED or name.startswith("__"):
                continue
            if any(a <= start < b for a, b in spans):
                continue
            if any(a <= start < b for a, b in taken):
                continue
            # Skip callee names and attribute tails: they identify the idiom.
            if pattern[end : end + 1] == "(" or pattern[max(0, start - 1) : start] == ".":
                continue
            cuts.append((start, end, "H"))

    if not cuts:
        return pattern, 0

    cuts.sort()
    names: dict[str, int] = {}
    pieces: list[str] = []
    cursor = 0
    for start, end, kind in cuts:
        if start < cursor:
            continue
        text = pattern[start:end]
        key = f"{kind}:{text}"
        index = names.setdefault(key, len(names))
        pieces.append(pattern[cursor:start])
        pieces.append(f"__{kind}{index}__")
        cursor = end
    pieces.append(pattern[cursor:])
    template = "".join(pieces)
    return template, len(names)


def fixed_text(template: str) -> str:
    return HOLE.sub("", template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--generalize-identifiers", action="store_true")
    parser.add_argument("--min-reward", type=int, default=1)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload["patterns"]
    count = token_counter()

    kept: list[dict] = []
    dropped = {"unusable": 0, "no_gain": 0, "hollow": 0}
    for row in rows:
        pattern = str(row.get("pattern", ""))
        if not pattern:
            dropped["unusable"] += 1
            continue
        template, holes = rewrite(
            pattern, generalize_identifiers=args.generalize_identifiers
        )
        if not template or MASK_SENTINEL.search(template) or PATTERN_END in template:
            dropped["unusable"] += 1
            continue
        skeleton = fixed_text(template)
        if not skeleton.strip():
            dropped["hollow"] += 1
            continue

        overhead = 1 + (max(0, holes - 1) + 1 if holes else 0)
        reward = count(skeleton) - overhead
        if reward < args.min_reward:
            dropped["no_gain"] += 1
            continue

        new = dict(row)
        new["pattern"] = template
        new["reward"] = reward
        new["type"] = "wildcard" if holes else "exact"
        new["holes"] = holes
        new["saved_est"] = reward * int(row.get("freq", 0) or 0)
        kept.append(new)

    kept.sort(key=lambda r: r["saved_est"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "patterns": kept,
                "meta": {
                    "method": "typed_hole_rewrite",
                    "source": str(args.source),
                    "generalize_identifiers": args.generalize_identifiers,
                    "pricing": "tokens(fixed) - goal - separators - pat_end",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = validate_registry(args.out)
    wildcards = sum(1 for r in kept if r["holes"])
    print(f"输入 {len(rows)} 条 → 保留 {len(kept)} 条 (含洞 {wildcards} 条)")
    print(f"丢弃: 无法重写 {dropped['unusable']}, 无骨架 {dropped['hollow']}, "
          f"收益不足 {dropped['no_gain']}")
    print(f"校验通过: {summary['patterns']} 条模式, {len(summary['goals'])} 个目标符号")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

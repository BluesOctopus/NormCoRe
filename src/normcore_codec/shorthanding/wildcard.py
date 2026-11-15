"""Dialect-level template patterns for shorthanding, with typed holes.

shorthanding runs *after* the static layer and *inside* the literal-mask window, so
templates match NormCoRe dialect text in which string and comment literals have
already been replaced by ``MASK`` placeholders. A template abstracts the parts
that vary between instances into holes, and every hole capture rides along
inside the emitted shorthand so the transformation stays bijective.

Hole syntax
-----------
``__H{k}__``
    Identifier, dotted name or simple subscript.
``__S{k}__``
    String literal, including the ``MASK`` placeholder that stands in for one.

Indices are global and contiguous from zero in order of first appearance; a
repeated index must capture the same text. Typing the holes matters because the
two bodies have different boundary behaviour: a quoted string is
self-delimiting, an identifier is not.

Shorthand layout
----------------
``<id_pat>`` ``capture_0;capture_1;…`` ``<pat_end>``

The trailing ``<pat_end>`` is what makes inversion unambiguous. Without it the
final capture has no right boundary: collapsing ``data[__H0__]<line_sep>`` over
two adjacent lines would emit ``<goal>CONF_LATITUDE`` immediately followed by
the next statement, and inversion would re-split the identifier in the wrong
place.

Example
-------
template: ``<for_stmt>__H0__ range(len(__H1__))``
goal:     ``<w_for_range_len_pat>``
match:    ``<for_stmt>i range(len(xs))`` → ``<w_for_range_len_pat>i;xs<pat_end>``
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from normcore_codec.reversible.markers import PATTERN_END

_HOLE_RE = re.compile(r"__([HS])(\d+)__")

# Aggressive Identifier hole: Allows identifiers, attribute chains, and simple subscripts/calls.
# This covers cases like 'data[0]', 'self.x()', or '[]' which are common in normalization surface.
_IDENT_BODY = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[^\]]*\]|\(\))*"
_IDENT_BODY_INVERT = _IDENT_BODY

# String-literal hole. Triple quotes are only ever seen as mask placeholders on
# this surface; single-line forms stay general so inversion also works on a
# buffer that ``tokenize`` refused to re-mask.
_STRING_BODY = r'"""MASK"""|\'\'\'MASK\'\'\'|"[^"\n]*"|\'[^\'\n]*\''

_BODIES = {"H": _IDENT_BODY, "S": _STRING_BODY}
_BODIES_INVERT = {"H": _IDENT_BODY_INVERT, "S": _STRING_BODY}

SEPARATOR = ";"
SCOPE_REF_PREFIX = "<cap_ref_"
_SCOPE_REF_RE = re.compile(r"<cap_ref_(\d+)>")
_VISIBLE_CAPTURE_RE = re.compile(_IDENT_BODY_INVERT)


def scope_reference_tokens(slots: int) -> set[str]:
    """Atomic local-reference vocabulary for optional shorthand captures."""
    return {f"{SCOPE_REF_PREFIX}{index}>" for index in range(max(0, slots))}


def bare_goal_token(goal: str) -> str:
    """The atomic token a goal emits, stripped of any hole skeleton."""
    return _HOLE_RE.split(goal)[0].split(SEPARATOR)[0].strip()


@dataclass(frozen=True)
class WildcardPattern:
    pattern_id: str
    template: str
    goal_token: str  # e.g. <w_for_range_len_pat>
    reward: float
    holes: tuple[tuple[str, int], ...]  # (kind, index) by first appearance

    @property
    def hole_count(self) -> int:
        return len(self.holes)

    @staticmethod
    def from_row(row: dict) -> WildcardPattern | None:
        pid = str(row.get("id", ""))
        template = str(row.get("pattern", row.get("template", "")))
        if not pid or not template or "__" not in template:
            return None

        ordered: list[tuple[str, int]] = []
        kind_of: dict[int, str] = {}
        for match in _HOLE_RE.finditer(template):
            kind, index = match.group(1), int(match.group(2))
            if index in kind_of:
                if kind_of[index] != kind:
                    return None
                continue
            kind_of[index] = kind
            ordered.append((kind, index))
        if not ordered:
            return None
        if sorted(index for _, index in ordered) != list(range(len(ordered))):
            return None

        goal = str(row.get("goal") or f"<{pid}_pat>")
        goal_token = bare_goal_token(goal)
        if not goal_token.startswith("<") or not goal_token.endswith(">"):
            goal_token = f"<{pid}_pat>"
        return WildcardPattern(
            pattern_id=pid,
            template=template,
            goal_token=goal_token,
            reward=max(1e-9, float(row.get("reward", 1))),
            holes=tuple(ordered),
        )


def _needs_word_boundary(template: str) -> bool:
    """A leading ``\\b`` helps anchor identifiers but never a leading quote."""
    if not template or template.startswith("__S"):
        return False
    return template[0].isalnum() or template[0] == "_"


def _template_to_regex(template: str) -> re.Pattern[str]:
    """Compile a template; a repeated hole index becomes a backreference."""
    parts: list[str] = []
    group_of: dict[int, int] = {}
    last = 0
    # Replace multiple spaces with a flexible regex space
    normalized_template = re.sub(r" +", " ", template)

    for match in _HOLE_RE.finditer(normalized_template):
        # Escape the text between holes, but then allow flexible spacing
        text_segment = normalized_template[last : match.start()]
        escaped = re.escape(text_segment).replace(r"\ ", r"\ +")
        parts.append(escaped)

        kind, index = match.group(1), int(match.group(2))
        if index in group_of:
            parts.append(f"\\{group_of[index]}")
        else:
            group_of[index] = len(group_of) + 1
            parts.append(f"({_BODIES[kind]})")
        last = match.end()

    parts.append(re.escape(normalized_template[last:]).replace(r"\ ", r"\ +"))
    body = "".join(parts)
    if _needs_word_boundary(normalized_template):
        body = r"\b" + body
    return re.compile(body)


def render_goal(pattern: WildcardPattern, captures: list[str]) -> str:
    """Shorthand carrying its captures, terminated so inversion is unambiguous."""
    return pattern.goal_token + SEPARATOR.join(captures) + PATTERN_END


def parse_raw_goal_at(
    text: str, start: int, pattern: WildcardPattern
) -> tuple[int, list[str]] | None:
    """Parse one legacy capture-carrying shorthand at ``start``."""
    if not text.startswith(pattern.goal_token, start):
        return None
    bodies = SEPARATOR.join(f"({_BODIES_INVERT[kind]})" for kind, _ in pattern.holes)
    rx = re.compile(
        re.escape(pattern.goal_token) + bodies + re.escape(PATTERN_END)
    )
    match = rx.match(text, start)
    if match is None:
        return None
    return match.end(), list(match.groups())


def parse_wire_goal_at(
    text: str, start: int, pattern: WildcardPattern
) -> tuple[int, list[str | int]] | None:
    """Parse a legacy or hybrid shorthand, returning raw strings or ref indices."""
    if not text.startswith(pattern.goal_token, start):
        return None
    parts = [re.escape(pattern.goal_token)]
    for field_index, (kind, _hole_index) in enumerate(pattern.holes):
        if field_index:
            parts.append(re.escape(SEPARATOR))
        parts.append(
            rf"(?:(?P<r{field_index}><cap_ref_(?P<i{field_index}>\d+)>)"
            rf"|(?P<v{field_index}>{_BODIES_INVERT[kind]}))"
        )
    parts.append(re.escape(PATTERN_END))
    match = re.compile("".join(parts)).match(text, start)
    if match is None:
        return None
    fields: list[str | int] = []
    for field_index in range(len(pattern.holes)):
        ref_index = match.group(f"i{field_index}")
        fields.append(int(ref_index) if ref_index is not None else match.group(f"v{field_index}"))
    return match.end(), fields


def compact_selected_goals(
    selected: list[dict],
    *,
    slots: int,
    count_fn,
) -> dict[str, int]:
    """Replace repeated H-only capture payloads with stream-local references.

    The first raw occurrence registers captures in deterministic left-to-right
    order. A later match can use the compact schema only when every capture is
    an identifier capture already present in the table. This keeps the decoder
    synchronized without a dictionary preamble.
    """
    table: dict[str, int] = {}
    compact_matches = 0
    reference_uses = 0
    local_tokens_saved = 0
    for match in sorted(selected, key=lambda item: item["start"]):
        if match.get("kind") != "wildcard":
            continue
        captures = list(match.get("captures") or ())
        kinds = tuple(match.get("capture_kinds") or ())
        if captures and len(captures) == len(kinds):
            fields: list[str] = []
            refs_used = 0
            all_fields_are_refs = True
            for kind, value in zip(kinds, captures):
                ref_index = table.get(value) if kind == "H" else None
                if ref_index is None:
                    fields.append(value)
                    all_fields_are_refs = False
                else:
                    fields.append(f"{SCOPE_REF_PREFIX}{ref_index}>")
                    refs_used += 1
            if refs_used:
                if all_fields_are_refs:
                    compact = str(match["goal_token"]) + "".join(fields)
                else:
                    compact = str(match["goal_token"]) + SEPARATOR.join(fields) + PATTERN_END
                raw_cost = count_fn(str(match["goal"]))
                compact_cost = count_fn(compact)
                if compact_cost < raw_cost:
                    match["goal"] = compact
                    match["tokens_compressed"] = compact_cost
                    match["reward"] = int(match.get("tokens_original", raw_cost)) - compact_cost
                    compact_matches += 1
                    reference_uses += refs_used
                    local_tokens_saved += raw_cost - compact_cost
        for kind, value in zip(kinds, captures):
            if kind == "H" and value not in table and len(table) < slots:
                table[value] = len(table)
    return {
        "scope_values_registered": len(table),
        "scope_compact_matches": compact_matches,
        "scope_reference_uses": reference_uses,
        "scope_local_tokens_saved": local_tokens_saved,
    }


def _visible_capture_values(text: str) -> list[tuple[int, int, str]]:
    """H-shaped values visible to both encoder and the streaming decoder."""
    values: list[tuple[int, int, str]] = []
    for match in _VISIBLE_CAPTURE_RE.finditer(text):
        start, end = match.span()
        if start > 0 and end < len(text) and text[start - 1] == "<" and text[end] == ">":
            continue
        if end < len(text) and text[end] == "(":
            continue
        values.append((start, end, match.group(0)))
    return values


def prepare_scope_capture_goals(
    text: str,
    matches: list[dict],
    *,
    slots: int,
    count_fn,
) -> dict[str, int]:
    """Make candidate costs reference-aware using the reconstructed prefix.

    Any H-shaped value already visible before a candidate is available to the
    decoder as well, even when its earlier occurrence was hidden by another
    selected pattern: the inverse scanner reconstructs that pattern before it
    advances. This makes each candidate's reference cost independent of the
    final overlap selection.
    """
    table: dict[str, int] = {}
    first_end: dict[str, int] = {}
    for _start, end, value in _visible_capture_values(text):
        if "MASK" in value:
            continue
        if value in table:
            continue
        if len(table) >= slots:
            break
        table[value] = len(table)
        first_end[value] = end

    candidate_compact = 0
    candidate_reference_uses = 0
    candidate_local_saved = 0
    for match in matches:
        if match.get("kind") != "wildcard":
            continue
        captures = list(match.get("captures") or ())
        kinds = tuple(match.get("capture_kinds") or ())
        if not captures or len(captures) != len(kinds):
            continue
        fields: list[str] = []
        refs_used = 0
        all_fields_are_refs = True
        for kind, value in zip(kinds, captures):
            ref_index = table.get(value) if kind == "H" else None
            if ref_index is None or first_end[value] > match["start"]:
                fields.append(value)
                all_fields_are_refs = False
            else:
                fields.append(f"{SCOPE_REF_PREFIX}{ref_index}>")
                refs_used += 1
        if not refs_used:
            continue
        if all_fields_are_refs:
            compact = str(match["goal_token"]) + "".join(fields)
        else:
            compact = str(match["goal_token"]) + SEPARATOR.join(fields) + PATTERN_END
        raw_cost = count_fn(str(match["goal"]))
        compact_cost = count_fn(compact)
        if compact_cost >= raw_cost:
            continue
        match["goal"] = compact
        match["_scope_refs_used"] = refs_used
        match["_scope_tokens_saved"] = raw_cost - compact_cost
        candidate_compact += 1
        candidate_reference_uses += refs_used
        candidate_local_saved += raw_cost - compact_cost
    return {
        "scope_values_registered": len(table),
        "scope_candidate_compact_matches": candidate_compact,
        "scope_candidate_reference_uses": candidate_reference_uses,
        "scope_candidate_local_tokens_saved": candidate_local_saved,
    }


def _register_visible_values(text: str, table: list[str]) -> None:
    for _start, _end, value in _visible_capture_values(text):
        if "MASK" not in value and value not in table:
            table.append(value)


def _table_from_prefix(text: str) -> list[str]:
    table: list[str] = []
    _register_visible_values(text, table)
    return table


def _compact_plain_scope_refs(
    text: str,
    dialect_prefix: str,
    *,
    slots: int,
    count_fn,
) -> tuple[str, str, int, int]:
    """Compact later visible identifiers while preserving decoder table order."""
    out: list[str] = []
    cursor = 0
    reference_uses = 0
    local_tokens_saved = 0
    for start, end, value in _visible_capture_values(text):
        out.append(text[cursor:start])
        dialect_prefix += text[cursor:start]
        table = _table_from_prefix(dialect_prefix)
        ref_index = table.index(value) if value in table else None
        if "MASK" not in value and ref_index is not None and ref_index < slots:
            ref = f"{SCOPE_REF_PREFIX}{ref_index}>"
            raw_cost = count_fn(value)
            ref_cost = count_fn(ref)
            if ref_cost < raw_cost:
                out.append(ref)
                reference_uses += 1
                local_tokens_saved += raw_cost - ref_cost
            else:
                out.append(value)
        else:
            out.append(value)
        dialect_prefix += value
        cursor = end
    out.append(text[cursor:])
    dialect_prefix += text[cursor:]
    return "".join(out), dialect_prefix, reference_uses, local_tokens_saved


def compact_plain_scope_refs(
    text: str,
    patterns: list[WildcardPattern],
    exact_patterns=(),
    *,
    slots: int,
    count_fn,
) -> tuple[str, dict[str, int]]:
    """Use the existing scope-reference vocabulary outside pattern captures."""
    if slots <= 0 or not text:
        return text, {
            "scope_plain_reference_uses": 0,
            "scope_plain_local_tokens_saved": 0,
        }
    goals: list[tuple[str, str, object]] = [
        (pattern.goal_token, "wildcard", pattern) for pattern in patterns
    ] + [
        (entry.goal, "exact", entry) for entry in exact_patterns if entry.goal
    ]
    goals.sort(key=lambda item: len(item[0]), reverse=True)
    dialect_prefix = ""
    out: list[str] = []
    cursor = 0
    reference_uses = 0
    local_tokens_saved = 0
    while cursor < len(text):
        found: tuple[int, str, str, object] | None = None
        for goal, kind, item in goals:
            index = text.find(goal, cursor)
            if index < 0:
                continue
            if found is None or index < found[0] or (
                index == found[0] and len(goal) > len(found[1])
            ):
                found = (index, goal, kind, item)
        plain_end = found[0] if found is not None else len(text)
        compact, dialect_prefix, uses, saved = _compact_plain_scope_refs(
            text[cursor:plain_end],
            dialect_prefix,
            slots=slots,
            count_fn=count_fn,
        )
        out.append(compact)
        reference_uses += uses
        local_tokens_saved += saved
        if found is None:
            break

        start, goal, kind, item = found
        if kind == "exact":
            out.append(goal)
            dialect_prefix += str(item.pattern)
            cursor = start + len(goal)
            continue

        pattern = item
        table = _table_from_prefix(dialect_prefix)
        pos = start + len(pattern.goal_token)
        refs: list[int] = []
        for _kind, _index in pattern.holes:
            ref_match = _SCOPE_REF_RE.match(text, pos)
            if ref_match is None:
                refs = []
                break
            refs.append(int(ref_match.group(1)))
            pos = ref_match.end()
        if refs:
            if any(index >= len(table) for index in refs):
                raise ValueError("shorthanding scope reference precedes its capture definition")
            captures = [table[index] for index in refs]
            out.append(text[start:pos])
            filled = pattern.template
            for (capture_kind, capture_index), value in zip(pattern.holes, captures):
                filled = filled.replace(f"__{capture_kind}{capture_index}__", value)
            dialect_prefix += filled
            cursor = pos
            continue

        wire = parse_wire_goal_at(text, start, pattern)
        if wire is None:
            out.append(pattern.goal_token)
            cursor = start + len(pattern.goal_token)
            continue
        end, fields = wire
        captures: list[str] = []
        for field in fields:
            if isinstance(field, int):
                if field >= len(table):
                    raise ValueError("shorthanding scope reference precedes its capture definition")
                captures.append(table[field])
            else:
                captures.append(field)
        out.append(text[start:end])
        filled = pattern.template
        for (capture_kind, capture_index), value in zip(pattern.holes, captures):
            filled = filled.replace(f"__{capture_kind}{capture_index}__", value)
        dialect_prefix += filled
        cursor = end
    return "".join(out), {
        "scope_plain_reference_uses": reference_uses,
        "scope_plain_local_tokens_saved": local_tokens_saved,
    }


def _expand_plain_scope_refs(
    text: str,
    dialect_prefix: str,
) -> tuple[str, str]:
    """Expand standalone references and update the reconstructed prefix."""
    out: list[str] = []
    cursor = 0
    for match in _SCOPE_REF_RE.finditer(text):
        plain = text[cursor:match.start()]
        out.append(plain)
        dialect_prefix += plain
        table = _table_from_prefix(dialect_prefix)
        index = int(match.group(1))
        if index >= len(table):
            raise ValueError("shorthanding scope reference precedes its capture definition")
        value = table[index]
        out.append(value)
        dialect_prefix += value
        cursor = match.end()
    out.append(text[cursor:])
    dialect_prefix += text[cursor:]
    return "".join(out), dialect_prefix


def expand_scope_capture_refs(
    text: str,
    patterns: list[WildcardPattern],
    exact_patterns=(),
) -> str:
    """Expand compact reference shorthands back to the legacy wire format."""
    if SCOPE_REF_PREFIX not in text:
        return text
    goals: list[tuple[str, str, object]] = [
        (pattern.goal_token, "wildcard", pattern) for pattern in patterns
    ] + [
        (entry.goal, "exact", entry) for entry in exact_patterns if entry.goal
    ]
    goals.sort(key=lambda item: len(item[0]), reverse=True)
    table: list[str] = []
    dialect_prefix = ""
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        found: tuple[int, str, str, object] | None = None
        for goal, kind, item in goals:
            index = text.find(goal, cursor)
            if index < 0:
                continue
            if found is None or index < found[0] or (
                index == found[0] and len(goal) > len(found[1])
            ):
                found = (index, goal, kind, item)
        if found is None:
            plain, dialect_prefix = _expand_plain_scope_refs(
                text[cursor:], dialect_prefix
            )
            out.append(plain)
            break
        start, goal, kind, item = found
        plain, dialect_prefix = _expand_plain_scope_refs(
            text[cursor:start], dialect_prefix
        )
        out.append(plain)
        table = _table_from_prefix(dialect_prefix)
        if kind == "exact":
            out.append(goal)
            dialect_prefix += str(item.pattern)
            table = _table_from_prefix(dialect_prefix)
            cursor = start + len(goal)
            continue
        pattern = item
        pos = start + len(pattern.goal_token)
        refs: list[int] = []
        for _kind, _index in pattern.holes:
            ref_match = _SCOPE_REF_RE.match(text, pos)
            if ref_match is None:
                refs = []
                break
            refs.append(int(ref_match.group(1)))
            pos = ref_match.end()
        if refs:
            if any(index >= len(table) for index in refs):
                raise ValueError("shorthanding scope reference precedes its capture definition")
            captures = [table[index] for index in refs]
            out.append(render_goal(pattern, captures))
            filled = pattern.template
            for (capture_kind, capture_index), value in zip(pattern.holes, captures):
                filled = filled.replace(f"__{capture_kind}{capture_index}__", value)
            dialect_prefix += filled
            table = _table_from_prefix(dialect_prefix)
            cursor = pos
            continue
        wire = parse_wire_goal_at(text, start, pattern)
        if wire is None:
            out.append(pattern.goal_token)
            cursor = start + len(pattern.goal_token)
            continue
        end, fields = wire
        captures: list[str] = []
        for field in fields:
            if isinstance(field, int):
                if field >= len(table):
                    raise ValueError("shorthanding scope reference precedes its capture definition")
                captures.append(table[field])
            else:
                captures.append(field)
        out.append(render_goal(pattern, captures))
        filled = pattern.template
        for (capture_kind, capture_index), value in zip(pattern.holes, captures):
            filled = filled.replace(f"__{capture_kind}{capture_index}__", value)
        dialect_prefix += filled
        table = _table_from_prefix(dialect_prefix)
        cursor = end
    return "".join(out)


def find_wildcard_matches(text: str, patterns: list[WildcardPattern]) -> list[dict]:
    """Return match dicts compatible with DP: start/end/reward/goal/id.

    Prefer longer templates when scanning; DP selects by reward. Do **not**
    filter by character length: goal tokens are long as text but count as one
    special once injected into the tokenizer vocabulary.
    """
    out: list[dict] = []
    for wp in sorted(patterns, key=lambda p: len(p.template), reverse=True):
        rx = _template_to_regex(wp.template)
        for match in rx.finditer(text):
            captures = list(match.groups())
            if len(captures) != wp.hole_count or any(c is None for c in captures):
                continue
            out.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "reward": wp.reward,
                    "goal": render_goal(wp, captures),
                    "id": wp.pattern_id,
                    "kind": "wildcard",
                    "captures": tuple(captures),
                    "capture_kinds": tuple(kind for kind, _ in wp.holes),
                    "goal_token": wp.goal_token,
                }
            )
    return out


def invert_wildcard(text: str, patterns: list[WildcardPattern]) -> str:
    """Replace terminated shorthands with their templates (longest goal first)."""
    ordered = sorted(patterns, key=lambda p: len(p.goal_token), reverse=True)
    out = text
    for wp in ordered:
        bodies = SEPARATOR.join(f"({_BODIES_INVERT[kind]})" for kind, _ in wp.holes)
        rx = re.compile(re.escape(wp.goal_token) + bodies + re.escape(PATTERN_END))

        def repl(match: re.Match[str], _wp: WildcardPattern = wp) -> str:
            filled = _wp.template
            for (kind, index), value in zip(_wp.holes, match.groups()):
                filled = filled.replace(f"__{kind}{index}__", value)
            return filled

        out = rx.sub(repl, out)
    return out

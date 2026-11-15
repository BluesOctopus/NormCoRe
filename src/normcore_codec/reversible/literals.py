"""Mask string/comment literals so structural edits do not touch them.

Contract
--------
- Plain strings / comments: whole-span MASK (SimPy).
- f-strings (``fstring_mode="segment"``, compress default): only literal text
  (``FSTRING_MIDDLE``) is replaced; ``{…}`` interpolations stay visible so
  normalization can compress inside them. Placeholders must keep the buffer **valid
  Python** for LibCST.
- f-strings (``fstring_mode="segment_nested"``): like ``segment``, but also
  whole-mask nested ``STRING`` tokens inside ``{…}``.
- f-strings (``fstring_mode="safe_whole"``, decompress remask): whole-mask
  f-strings that contain no dialect ``<…>`` markers; otherwise segment+nested.
  Avoids spy scanner breakage on nested quotes (case 43) without swallowing
  undecoded markers into string literals.
- f-strings (``fstring_mode="whole"``, compress fallback): replace the entire
  f-string with ``f"MASK"`` / ``f'''MASK'''`` when segment masks yield invalid
  Python for LibCST (case 156 nested/escaped-brace f-strings).
"""

from __future__ import annotations

import io
import re
import tokenize
from typing import Literal

# f-string middle placeholders (must be valid f-string text, unique enough)
_FMASK = "__NORMCORE_FMASK__"
_LEGACY_FMASK = "__RECODE_FMASK__"
_FMASK_RE = re.compile(re.escape(_FMASK))
_WHOLE_MASK_RE = re.compile(r'("""MASK"""|\'\'\'MASK\'\'\'|"MASK"|\'MASK\'|#MASK)')

# Exactly the placeholders ``unmask_literals`` consumes, in the same order. Any
# stage running between mask and unmask must conserve this multiset, otherwise
# the positional restore walk drifts.
_UNMASK_RE = re.compile(
    r"(?:[fF](?:r|R)?|[rR][fF]?)?(?:\"\"\"MASK\"\"\"|'''MASK'''|\"MASK\"|'MASK')"
    r"|#MASK|" + re.escape(_FMASK) + r"|" + re.escape(_LEGACY_FMASK)
)


def count_literal_placeholders(text: str) -> int:
    """Number of masked literals currently present in ``text``."""
    return len(_UNMASK_RE.findall(text))


def _fstring_placeholder(opener: str) -> str:
    """Build ``f"MASK"`` / ``rf'''MASK'''`` matching the f-string opener."""
    prefix = ""
    body = opener
    for ch in opener:
        if ch in "\"'":
            break
        prefix += ch
        body = body[1:]
    if body.startswith("'''") or body.startswith('"""'):
        q = body[:3]
    else:
        q = body[:1] if body else '"'
    return f"{prefix}{q}MASK{q}"


def _has_dialect_marker(text: str) -> bool:
    """True if ``text`` still contains normalization/B markers (not HTML tags)."""
    # Explicit dialect vocabulary only — never match HTML like ``<footer>``.
    return bool(
        re.search(
            r"<(?:"
            r"line_sep|block_start|block_end|arrow|concat|SPACE|"
            r"none|true|false|ne|eq|lt|gt|le|ge|lgne|"
            r"is|in|isnot|notin|and|or|not|as|from|await|yield|lambda|ellipsis|"
            r"[a-z]+_stmt|[a-z]+_end|[a-z]+_start|[a-z]+_keyword|"
            r"m_[A-Za-z0-9_]+_pat|a_stmt_[A-Za-z0-9_]+_pat"
            r")>",
            text,
        )
    )


def mask_literals(
    source: str,
    *,
    fstring_mode: Literal["segment", "segment_nested", "safe_whole", "whole"] = "segment",
) -> tuple[str, list[str]]:
    """Replace STRING/COMMENT/f-string literal spans with MASK placeholders."""
    records: list[str] = []
    out: list[str] = []
    last = 0
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source, records

    f_start = getattr(tokenize, "FSTRING_START", None)
    f_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    f_end = getattr(tokenize, "FSTRING_END", None)

    def _append_string_mask(original: str) -> None:
        records.append(original)
        if original.startswith(("'''", '"""')):
            q = original[:3]
            out.append(f"{q}MASK{q}")
        elif original.startswith(("'", '"')):
            q = original[0]
            out.append(f"{q}MASK{q}")
        else:
            out.append('"MASK"')

    def _segment_mask_fstring(i: int, j: int, start: int, end: int, nested: bool) -> None:
        cursor = start
        k = i
        while k <= j:
            t = tokens[k]
            if f_middle is not None and t.type == f_middle and t.string:
                ms, me = _offset(source, t.start), _offset(source, t.end)
                out.append(source[cursor:ms])
                records.append(source[ms:me])
                out.append(_FMASK)
                cursor = me
            elif nested and t.type == tokenize.STRING:
                ms, me = _offset(source, t.start), _offset(source, t.end)
                if ms >= cursor:
                    out.append(source[cursor:ms])
                    _append_string_mask(source[ms:me])
                    cursor = me
            k += 1
        out.append(source[cursor:end])

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # f-string span [i..j]
        if f_start is not None and tok.type == f_start:
            j = i
            depth = 0
            while j < len(tokens):
                t = tokens[j]
                if t.type == f_start:
                    depth += 1
                elif t.type == f_end:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= len(tokens):
                break
            start = _offset(source, tok.start)
            end = _offset(source, tokens[j].end)
            if start < last:
                i += 1
                continue
            out.append(source[last:start])
            original = source[start:end]
            if fstring_mode == "whole" or (
                fstring_mode == "safe_whole" and not _has_dialect_marker(original)
            ):
                records.append(original)
                out.append(_fstring_placeholder(tok.string))
            elif fstring_mode in ("segment_nested", "safe_whole"):
                _segment_mask_fstring(i, j, start, end, nested=True)
            else:
                _segment_mask_fstring(i, j, start, end, nested=False)
            last = end
            i = j + 1
            continue

        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            start = _offset(source, tok.start)
            end = _offset(source, tok.end)
            if start >= last:
                out.append(source[last:start])
                original = source[start:end]
                if tok.type == tokenize.COMMENT:
                    records.append(original)
                    out.append("#MASK")
                else:
                    _append_string_mask(original)
                last = end
        i += 1

    out.append(source[last:])
    return "".join(out), records


def unmask_literals(text: str, records: list[str]) -> str:
    """Restore masked literals in order (whole-string MASK and f-string middles).

    Comments get a trailing physical ``\\n`` (SimPy decode contract): spy grammar
    ``comment: #.*`` otherwise greedily eats ``<line_sep>`` / markers on the
    same logical line after postprocess removed real newlines.
    """
    if not records:
        return text

    it = iter(records)

    def next_record(raw: str) -> str:
        try:
            record = next(it)
        except StopIteration:
            return raw
        if record.startswith("#") and not record.endswith("\n"):
            record = record + "\n"
        return record

    # Combined scan: whole masks (incl. ``f"MASK"`` / ``rf'MASK'``) and middles.
    return _UNMASK_RE.sub(lambda m: next_record(m.group(0)), text)


def _offset(source: str, pos: tuple[int, int]) -> int:
    line, col = pos
    lines = source.splitlines(keepends=True)
    off = sum(len(lines[i]) for i in range(line - 1))
    return off + col

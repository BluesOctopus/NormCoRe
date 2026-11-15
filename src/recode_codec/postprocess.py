"""Post-process dialect after Pass A span edits.

Contract
--------
After structural edits the buffer may still contain *physical* newlines from the
original source (between statements, after comments). Those are **not** part of
the dialect layout vocabulary — layout is only ``<line_sep>`` /
``<block_start>`` / ``<block_end>``.

Collapsing all whitespace (SimPy-style) without first normalizing newlines
glues ``#MASK`` onto the next statement (``#MASK<import_stmt>…``), which is the
root cause of comment-eats-code restore failures.

Inside ``()[]{}``, newlines become ordinary spaces (spy ``spaceSep1`` lists),
not ``<line_sep>`` — otherwise soft breaks eat the comma-placeholder spaces
when ``space_next_to_symbol`` runs against held markers.
"""

from __future__ import annotations

try:
    import regex as re
except ImportError:
    import re  # type: ignore[no-redef]

_LINE_CONTINUE = re.compile(r"\\\r?\n")
_SPACE_NEXT_TO_SYMBOL = re.compile(
    r'(?<=[^a-zA-Z0-9_\*\s"]+)[ \t]+|[ \t]+(?=[^a-zA-Z0-9_\*\s"]+)'
)

_HOLD = {
    "<line_sep>": "\x00LS\x00",
    "<block_start>": "\x00BS\x00",
    "<block_end>": "\x00BE\x00",
}


def _newlines_to_layout(code: str, ls_hold: str) -> str:
    """Outside brackets → ``<line_sep>``; inside ``()[]{}`` → space."""
    out: list[str] = []
    depth = 0
    for ch in code:
        if ch in "([{":
            depth += 1
            out.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            out.append(ch)
        elif ch == "\n":
            out.append(" " if depth > 0 else ls_hold)
        else:
            out.append(ch)
    return "".join(out)


def postprocess_spy(code: str, *, aggressive: bool = False) -> str:
    # Join physical line continuations before newline → <line_sep>.
    code = _LINE_CONTINUE.sub("", code)

    for m, h in _HOLD.items():
        code = code.replace(m, h)

    # Physical newlines → dialect layout (held form); flatten inside brackets.
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = _newlines_to_layout(code, _HOLD["<line_sep>"])

    # Soft-wrap flattening
    code = re.sub(r"\\(?=[ \t]*[A-Za-z_])", "", code)

    if aggressive:
        try:
            code = _SPACE_NEXT_TO_SYMBOL.sub("", code)
        except re.error:
            pass

    code = code.replace("<SPACE>", " ")
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r" ?= ?", "=", code)
    code = re.sub(r" ?<as> ?", "<as>", code)
    code = re.sub(r" ?<arrow> ?", "<arrow>", code)

    # Drop redundant line_sep next to block markers / duplicate seps
    ls = _HOLD["<line_sep>"]
    bs, be = _HOLD["<block_start>"], _HOLD["<block_end>"]
    code = re.sub(rf"(?:{re.escape(ls)})+", ls, code)
    code = code.replace(ls + bs, bs)
    code = code.replace(bs + ls, bs)
    code = code.replace(ls + be, be)
    code = code.replace(be + ls + be, be + be)

    for m, h in _HOLD.items():
        code = code.replace(h, m)

    code = re.sub(
        r"<line_sep>\s*(?=<[a-z_]+_(?:stmt|end|start|keyword)>)",
        "",
        code,
    )
    # GCA Markers are already restored by the _HOLD loop above
    code = re.sub(r"(<block_end>)+ +", r"\1", code)

    return code.strip() + "\n"

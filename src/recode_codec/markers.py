"""RECODE dialect markers (encode vocabulary + tokenizer specials)."""

from __future__ import annotations

STMT_KEYWORDS = (
    "with",
    "def",
    "class",
    "try",
    "except",
    "finally",
    "match",
    "while",
    "assert",
    "del",
    "case",
    "elif",
)

EXPR_KEYWORDS = (
    "and",
    "or",
    "not",
    "is",
    "return",
    "raise",
    "break",
    "continue",
    "pass",
    "global",
    "nonlocal",
    "from",
    "as",
    "print",
    "exec",
    "in",
    "lambda",
    "await",
    "yield",
)

# Mid-expression keywords (decode: wrap with spaces)
EXPR_MID = frozenset({"and", "or", "not", "is", "from", "as", "in"})

OPERATORS: dict[str, str] = {
    "//": "<floordiv>",
    "<": "<l>",
    ">": "<g>",
    "<<": "<lshift>",
    ">>": "<rshift>",
    "==": "<eq>",
    "!=": "<ne>",
    "<=": "<le>",
    ">=": "<ge>",
    "not in": "<notin>",
    "is not": "<isnot>",
    "+=": "<augadd>",
    "-=": "<augsub>",
    "*=": "<augmul>",
    "/=": "<augdiv>",
    "@=": "<augmatmul>",
    "//=": "<augfloordiv>",
    "%=": "<augmod>",
    "**=": "<augpow>",
    ">>=": "<augrshift>",
    "<<=": "<auglshift>",
    "&=": "<augand>",
    "^=": "<augxor>",
    "|=": "<augor>",
    "->": "<arrow>",
    "*": "<times>",
    "**": "<power>",
}

LAYOUT = ("<line_sep>", "<block_start>", "<block_end>", "<SPACE>", "<async_keyword>", "<anchor_sep>")

# Terminates a Pass B shorthand that carries hole captures. Without it the last
# capture has no right boundary and inversion re-splits it against whatever
# follows in the stream.
PATTERN_END = "<pat_end>"

EXTRA_MARKERS = (
    PATTERN_END,
    "<ellipsis>",
    "<except_group_stmt>",
    "<import_from_future_stmt>",
    "<import_stmt>",
    "<import_from_stmt>",
    "<concat>",
    "<lgne>",
    "<true>",
    "<false>",
    "<none>",
    "<for>",
    "<if>",
    "<else>",
    "<anchor_sep>",
)

STMT_MARKERS = {kw: f"<{kw}_stmt>" for kw in STMT_KEYWORDS}
STMT_MARKERS.update({"if": "<if_stmt>", "for": "<for_stmt>", "else": "<else_stmt>"})

EXPR_MARKERS = {kw: f"<{kw}>" for kw in EXPR_KEYWORDS}
EXPR_MARKERS.update({"True": "<true>", "False": "<false>", "None": "<none>"})


def all_special_tokens() -> list[str]:
    tokens = set(LAYOUT)
    tokens.update(EXTRA_MARKERS)
    tokens.update(STMT_MARKERS.values())
    tokens.update(EXPR_MARKERS.values())
    tokens.update(OPERATORS.values())
    return sorted(tokens, key=len, reverse=True)

"""Spy→Python node transform rules for RECODE Pass A⁻¹.

Based on SimPy's spy→py surface, but RECODE's own node-rule set: finer CST
edits (indent/dedent, ``no_comma``, leaf ops, stmt/expr markers) tuned for
LibCST-encoded dialect. Does not import Simpy-master.
"""

from __future__ import annotations

from recode_codec.markers import EXPR_KEYWORDS as _EXPR_ALL
from recode_codec.markers import EXPR_MID, OPERATORS, STMT_KEYWORDS as _STMT_KW
from recode_codec.spy_edit import Edit, FakeNode, search_edits

# Leaf ``<kw_stmt>`` markers. ``try`` is handled by ``try_statement`` → ``try:``.
STMT_KEYWORDS = [k for k in _STMT_KW if k != "try"] + ["else"]
EXPR_KEYWORDS = list(EXPR_MID)
LINE_EXPR_KEYWORDS = [k for k in _EXPR_ALL if k not in EXPR_MID]
# Decode leaf ops; keep ``is not`` / ``not in`` on ``notin_expr`` (needs spaces).
OPER_KEYWORDS = {
    **{k: v for k, v in OPERATORS.items() if k not in ("is not", "not in")},
    "<>": "<lgne>",
    "async ": "<async_keyword>",
}


def scope_indent(node):
    return [
        Edit(node=node, action="indent"),
        Edit(node=node, action="replace", content=""),
    ]


def scope_dedent(node):
    return [
        Edit(node=node, action="dedent"),
        Edit(node=node, action="replace", content=""),
        Edit(
            node=FakeNode(start_byte=node.end_byte, end_byte=node.end_byte, text=node.text),
            action="newline",
        ),
    ]


def line_sep(node):
    return [
        Edit(node=node, action="newline"),
        Edit(node=node, action="replace", content=""),
    ]


def comment(node):
    return [
        Edit(
            node=FakeNode(start_byte=node.end_byte + 1, end_byte=node.end_byte, text=node.text),
            action="newline",
        )
    ]


def _is_single_space(node) -> bool:
    """True only for a one-character space (spaceSep1), not indent runs."""
    t = node.text
    if isinstance(t, bytes):
        return t == b" "
    return t == " "


def import_stmt(node):
    edits = [Edit(node=node.children[0], action="replace", content="import ")]
    # Do not turn spaces around ``<as>`` into commas (``import x, as, y``).
    # Only single-space separators — never indent runs.
    for c in node.children:
        if c.type != " " or not _is_single_space(c):
            continue
        edits.append(Edit(node=c, action="replace", content=", "))
    as_nodes = [c for c in node.children if c.type in ("<as>", "as")]
    if as_nodes:
        as_starts = {c.start_byte for c in as_nodes}
        as_ends = {c.end_byte for c in as_nodes}
        filtered = []
        for e in edits:
            if e.action == "replace" and e.content == ", " and e.node.type == " ":
                if e.node.end_byte in as_starts or e.node.start_byte in as_ends:
                    continue
            filtered.append(e)
        edits = filtered
    edits.extend(search_edits(node.children, "replace", "<as>", " as "))
    return edits


def import_from_stmt(node):
    edits = [
        Edit(node=node.children[0], action="replace", content="from "),
        Edit(node=node.children[1], action="append", content=" import"),
    ]
    # Skip first space (module ↔ import keyword); only single-space separators.
    space_edits = [
        Edit(node=c, action="replace", content=", ")
        for c in node.children
        if c.type == " " and _is_single_space(c)
    ]
    edits.extend(space_edits[1:])
    as_nodes = [c for c in node.children if c.type in ("<as>", "as")]
    if as_nodes:
        as_starts = {c.start_byte for c in as_nodes}
        as_ends = {c.end_byte for c in as_nodes}
        filtered = []
        for e in edits:
            if e.action == "replace" and e.content == ", " and getattr(e.node, "type", None) == " ":
                if e.node.end_byte in as_starts or e.node.start_byte in as_ends:
                    continue
            filtered.append(e)
        edits = filtered
    edits.extend(search_edits(node.children, "replace", "<as>", " as "))
    return edits


def import_from_future_stmt(node):
    edits = [Edit(node=node.children[0], action="replace", content="from __future__ import")]
    edits.extend(
        Edit(node=c, action="replace", content=", ")
        for c in node.children
        if c.type == " " and _is_single_space(c)
    )
    return edits


def def_stmt(node):
    non_comment_children = [n for n in node.children if n.type != "comment"]
    edits = [Edit(node=non_comment_children[-2], action="append", content=":", priority=0.6)]
    params = node.child_by_field_name("parameters")
    if not params:
        name = node.child_by_field_name("name")
        edits.append(Edit(node=name, action="append", content="()", priority=0.7))
    return edits


def parameters(node):
    # When Pass A skips param compression (inline comments), the dialect
    # already has ``(… )``. Re-wrapping would yield ``def f((…))``.
    text = node.text.decode("utf-8") if isinstance(node.text, bytes) else (node.text or "")
    already_paren = text.lstrip().startswith("(")
    edits = []
    if not already_paren:
        edits.append(Edit(node=node, action="insert", content="("))
        edits.append(Edit(node=node, action="append", content=")", priority=0.7))
    edits.extend(
        Edit(node=c, action="replace", content=", ")
        for c in node.children
        if c.type == " " and _is_single_space(c)
    )
    return edits


def class_stmt(node):
    non_comment_children = [n for n in node.children if n.type != "comment"]
    return [Edit(node=non_comment_children[-2], action="append", content=":", priority=0.6)]


def for_stmt(node):
    # Colon must attach to the header (node before *body* block), not
    # ``children[-2]`` — with ``for…else`` the alternative is last and [-2]
    # is the body, which produced a stray ``:`` after the loop.
    edits = search_edits(node.children, "replace", "<for_stmt>", "for ")
    # Only the separator between target and iterable (first direct `` ``).
    # If Pass A left a literal ``in`` (stale span skipped ``<SPACE>``), do not
    # also rewrite the following space into `` in `` (``for x in in y``).
    count = 0
    for c in node.children:
        if c.type == " ":
            if count == 0:
                # Peek: if previous sibling text already ends with ``in``, keep space.
                prev = None
                for p in node.children:
                    if p is c:
                        break
                    if p.type != "comment":
                        prev = p
                prev_txt = ""
                if prev is not None:
                    t = prev.text
                    prev_txt = (
                        t.decode("utf-8", errors="replace")
                        if isinstance(t, bytes)
                        else (t or "")
                    ).rstrip()
                if prev_txt.endswith(" in") or prev_txt == "in":
                    pass  # leave the space as-is
                else:
                    edits.append(Edit(node=c, action="replace", content=" in "))
            count += 1
    previous_node = None
    for child in node.children:
        if child.type != "comment":
            if child.type == "block":
                edits.append(Edit(node=previous_node, action="append", content=":", priority=0.6))
                break
            previous_node = child
    return edits


def for_in_clause_stmt(node):
    # Match SimPy: first space → `` in ``, later spaces → ``, ``.
    edits = search_edits(node.children, "replace", "<for>", " for ")
    count = 0
    for c in node.children:
        if c.type == " ":
            if count == 0:
                edits.append(Edit(node=c, action="replace", content=" in "))
            else:
                edits.append(Edit(node=c, action="replace", content=", "))
            count += 1  # SimPy omits this; keep correct multi-target commas
    return edits


def if_stmt(node):
    edits = search_edits(node.children, "replace", "<if_stmt>", "if ")
    previous_node = None
    for child in node.children:
        if child.type != "comment":
            if child.type == "block":
                edits.append(Edit(node=previous_node, action="append", content=":", priority=0.6))
                break
            previous_node = child
    return edits


def if_clause_stmt(node):
    return search_edits(node.children, "replace", "<if>", " if ")


def newline_clause_stmt(node):
    edits = [Edit(node=node, action="newline")]
    previous_node = None
    for child in node.children:
        if child.type != "comment":
            if child.type == "block":
                edits.append(Edit(node=previous_node, action="append", content=":", priority=0.6))
                break
            previous_node = child
    return edits


def match_stmt(node):
    edits = no_comma(node)
    edits.extend(rest_compound_stmt(node))
    return edits


def try_stmt(node):
    return search_edits(node.children, "replace", "<try_stmt>", "try:")


def rest_compound_stmt(node):
    edits = []
    previous_node = None
    for child in node.children:
        if child.type != "comment":
            if child.type == "block":
                edits.append(Edit(node=previous_node, action="append", content=":", priority=0.6))
                break
            previous_node = child
    return edits


def except_clause(node):
    edits = [Edit(node=node, action="newline")]
    previous_node = None
    for child in node.children:
        if child.type != "comment":
            if child.type == "block":
                edits.append(Edit(node=previous_node, action="append", content=":", priority=0.6))
                break
            previous_node = child
    return edits


def clause_stmt(node):
    non_comment_children = [n for n in node.children if n.type != "comment"]
    return [Edit(node=non_comment_children[-2], action="append", content=":", priority=0.6)]


def notin_expr(node):
    text = node.text.decode("utf-8") if isinstance(node.text, bytes) else node.text
    return [Edit(node, action="replace", content=text.replace("<", " ").replace(">", " "))]


def conditional_expr(node):
    edits = search_edits(node.children, "replace", "<if>", " if ")
    edits.extend(search_edits(node.children, "replace", "<else>", " else "))
    return edits


def concatenated_string(node):
    return search_edits(node.children, "replace", "<concat>", " ")


# When remask leaves plain Python (Pass A parse failed), spy may wrap a
# whole region as ``expression_statement`` and ``no_comma`` would turn
# ``if not`` / ``for x`` / ``import x as y`` header spaces into commas
# (case 156: ``if, not`` / ``import x, as, y``).
_NO_COMMA_SKIP_WORDS = frozenset(
    {
        "if",
        "elif",
        "else",
        "for",
        "while",
        "def",
        "class",
        "with",
        "try",
        "except",
        "finally",
        "async",
        "return",
        "raise",
        "assert",
        "import",
        "from",
        "global",
        "nonlocal",
        "match",
        "case",
        "not",
        "and",
        "or",
        "in",
        "is",
        "lambda",
        "yield",
        "await",
        "as",
        "del",
        "pass",
        "break",
        "continue",
    }
)


def _node_text(node) -> str:
    t = node.text
    if isinstance(t, bytes):
        return t.decode("utf-8", errors="replace")
    return t or ""


def no_comma(node):
    # Only single-space separators (spaceSep1). Multi-space nodes are
    # indentation left in ERROR / unparsed regions — replacing them with
    # ``, `` strips indent (seen as ``from …\\n if`` with a leading space).
    edits = []
    children = list(node.children)
    for i, n in enumerate(children):
        if n.type != " " or not _is_single_space(n):
            continue
        prev = children[i - 1] if i > 0 else None
        nxt = children[i + 1] if i + 1 < len(children) else None
        if prev is not None and prev.type in ("identifier", "keyword"):
            if _node_text(prev) in _NO_COMMA_SKIP_WORDS:
                continue
        if nxt is not None and nxt.type in ("identifier", "keyword", "<as>", "as"):
            if _node_text(nxt) in _NO_COMMA_SKIP_WORDS or nxt.type in ("<as>", "as"):
                continue
        edits.append(Edit(node=n, action="replace", content=", "))
    return edits


TRANSFORM_RULES = {
    "<line_sep>": line_sep,
    "<block_start>": scope_indent,
    "<block_end>": scope_dedent,
    "comment": comment,
    "function_definition": def_stmt,
    "class_definition": class_stmt,
    "for_statement": for_stmt,
    "with_statement": rest_compound_stmt,
    "try_statement": try_stmt,
    "if_statement": if_stmt,
    "if_clause": if_clause_stmt,
    "while_statement": rest_compound_stmt,
    "elif_clause": newline_clause_stmt,
    "else_clause": newline_clause_stmt,
    "except_clause": except_clause,
    "conditional_expression": conditional_expr,
    "<notin>": notin_expr,
    "<isnot>": notin_expr,
    "finally_clause": clause_stmt,
    "case_clause": newline_clause_stmt,
    "with_clause": no_comma,
    "expression_statement": no_comma,
    "assert_statement": no_comma,
    "subscript": no_comma,
    "class_pattern": no_comma,
    "global_statement": no_comma,
    "nonlocal_statement": no_comma,
    "import_from_statement": import_from_stmt,
    "import_statement": import_stmt,
    "future_import_statement": import_from_future_stmt,
    "match_statement": match_stmt,
    "for_in_clause": for_in_clause_stmt,
    "list": no_comma,
    "expression_list": no_comma,
    "tuple": no_comma,
    "set": no_comma,
    "dictionary": no_comma,
    "list_pattern": no_comma,
    "tuple_pattern": no_comma,
    "parameters": parameters,
    "lambda_parameters": no_comma,
    "type_parameter": no_comma,
    "argument_list": no_comma,
    "concatenated_string": concatenated_string,
    "none": lambda node: [Edit(node=node, action="replace", content=" None ")],
    "true": lambda node: [Edit(node=node, action="replace", content=" True ")],
    "false": lambda node: [Edit(node=node, action="replace", content=" False ")],
}

TRANSFORM_RULES.update(
    {
        f"<{keyword}_stmt>": (lambda node, c=keyword: [Edit(node=node, action="replace", content=f"{c} ")])
        for keyword in STMT_KEYWORDS
    }
)
TRANSFORM_RULES.update(
    {
        f"<{keyword}>": (lambda node, c=keyword: [Edit(node=node, action="replace", content=f" {c} ")])
        for keyword in EXPR_KEYWORDS
    }
)
TRANSFORM_RULES.update(
    {
        f"<{keyword}>": (lambda node, c=keyword: [Edit(node=node, action="replace", content=f"{c} ")])
        for keyword in LINE_EXPR_KEYWORDS
    }
)
TRANSFORM_RULES.update(
    {v: (lambda node, c=k: [Edit(node=node, action="replace", content=c)]) for k, v in OPER_KEYWORDS.items()}
)
# Grammar uses node type ``ellipsis`` for ``<ellipsis>`` marker content in some builds;
# also map the marker token type if present as a leaf.
TRANSFORM_RULES["ellipsis"] = lambda node: [Edit(node=node, action="replace", content="...")]
TRANSFORM_RULES["<ellipsis>"] = lambda node: [Edit(node=node, action="replace", content="...")]

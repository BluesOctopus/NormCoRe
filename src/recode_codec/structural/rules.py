"""LibCST port of SimPy rules_py2spy (source-preserving span edits)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, ParentNodeProvider, PositionProvider

from recode_codec.edits import SpanEdit
from recode_codec.markers import EXPR_KEYWORDS, OPERATORS, STMT_KEYWORDS

STMT_KW_MARKERS = {kw: f"<{kw}_stmt>" for kw in STMT_KEYWORDS}
STMT_KW_MARKERS.update({"if": "<if_stmt>", "for": "<for_stmt>", "else": "<else_stmt>"})
EXPR_KW_MARKERS = {kw: f"<{kw}>" for kw in EXPR_KEYWORDS}
EXPR_KW_MARKERS.update({"True": "<true>", "False": "<false>", "None": "<none>"})


@dataclass
class RuleContext:
    source: str
    metadata: cst.CSTVisitor
    _byte_to_char: list[int] | None = None
    scores: dict[tuple[str, int, int], object] | None = None
    hotzones: set[int] | None = None

    def _ensure_byte_map(self) -> list[int]:
        """LibCST ByteSpan is UTF-8 *byte* offsets; edits index Python ``str``.

        On ASCII-only files the two coincide. Unicode-heavy sources (case 43
        box-drawing) desync spans so ``is``/``!=`` land on neighboring chars.
        Cache the map on the visitor so per-node ``RuleContext`` rebuilds are free.
        """
        if self._byte_to_char is not None:
            return self._byte_to_char
        cached = getattr(self.metadata, "_aari_byte_to_char", None)
        if cached is not None:
            self._byte_to_char = cached
            return cached
        enc_len = len(self.source.encode("utf-8"))
        mapping = [0] * (enc_len + 1)
        char_i = 0
        byte_i = 0
        for ch in self.source:
            n = len(ch.encode("utf-8"))
            for _ in range(n):
                mapping[byte_i] = char_i
                byte_i += 1
            char_i += 1
        mapping[byte_i] = char_i
        setattr(self.metadata, "_aari_byte_to_char", mapping)
        self._byte_to_char = mapping
        return mapping

    def span(self, node: object) -> tuple[int, int]:
        cs = self.metadata.get_metadata(ByteSpanPositionProvider, node)  # type: ignore[attr-defined]
        b2c = self._ensure_byte_map()
        b0 = cs.start
        b1 = cs.start + cs.length
        n = len(b2c) - 1
        b0 = 0 if b0 < 0 else (n if b0 > n else b0)
        b1 = 0 if b1 < 0 else (n if b1 > n else b1)
        return b2c[b0], b2c[b1]

    def line_of(self, node: object) -> int:
        rng = self.metadata.get_metadata(PositionProvider, node)  # type: ignore[attr-defined]
        return rng.start.line


def _inside_fstring(ctx: RuleContext, node: cst.CSTNode) -> bool:
    """True if ``node`` sits under a ``FormattedString`` (f-string).

    Commas inside f-string interpolations must stay as ``,`` — ``safe_whole``
    remask restores the whole f-string from the compressed buffer, so
    ``no_comma`` would permanently drop them (cases 21/24/58/149).
    """
    cur: object = node
    for _ in range(32):
        try:
            parent = ctx.metadata.get_metadata(ParentNodeProvider, cur)  # type: ignore[attr-defined]
        except Exception:
            return False
        if parent is None:
            return False
        if isinstance(parent, cst.FormattedString):
            return True
        cur = parent
    return False


def _visit_commas(node: cst.CSTNode, ctx: RuleContext) -> list[SpanEdit]:
    if _inside_fstring(ctx, node):
        return []
    edits: list[SpanEdit] = []

    class V(cst.CSTVisitor):
        def visit_Comma(self, n: cst.Comma) -> None:
            s, e = ctx.span(n)
            if ctx.source[s:e].strip() != ",":
                return
            edits.append(SpanEdit.replace(s, e, "<SPACE>"))

    node.visit(V())
    return edits


def _direct_commas(elements: object, ctx: RuleContext) -> list[SpanEdit]:
    """Replace only commas between direct sequence elements (not nested subtrees).

    Guard: ByteSpanPositionProvider occasionally returns a stale/wrong span for
    ``Comma`` inside nested ``Subscript`` (seen as replacing ``s`` in ``List``).
    Only emit an edit when the spanned text is actually a comma.
    """
    edits: list[SpanEdit] = []
    if elements is None or isinstance(elements, cst.ImportStar):
        return edits
    try:
        iterable = list(elements)  # type: ignore[arg-type]
    except TypeError:
        return edits
    if iterable and isinstance(iterable[0], cst.CSTNode) and _inside_fstring(ctx, iterable[0]):
        return []
    for el in iterable:
        comma = getattr(el, "comma", None)
        if isinstance(comma, cst.Comma):
            s, e = ctx.span(comma)
            if ctx.source[s:e].strip() != ",":
                continue
            edits.append(SpanEdit.replace(s, e, "<SPACE>"))
    return edits


def _visit_colons_delete(node: cst.CSTNode, ctx: RuleContext) -> list[SpanEdit]:
    edits: list[SpanEdit] = []

    class V(cst.CSTVisitor):
        def visit_Colon(self, n: cst.Colon) -> None:
            s, e = ctx.span(n)
            edits.append(SpanEdit.delete(s, e))

    node.visit(V())
    return edits


def _replace_commas_depth0(text: str) -> str:
    """Replace commas only at nesting depth 0 (preserve List[a, b])."""
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
            out.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            out.append(ch)
        elif ch == "," and depth == 0:
            out.append("<SPACE>")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _find_matching_paren(source: str, open_pos: int) -> int:
    if open_pos < 0 or open_pos >= len(source) or source[open_pos] != "(":
        return -1
    depth = 0
    for i in range(open_pos, len(source)):
        ch = source[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _suite_colon_to_block_start(ctx: RuleContext, node: cst.CSTNode, body: cst.CSTNode) -> list[SpanEdit]:
    """Replace the suite ``:`` with ``<block_start>``.

    Contract: ``<block_start>`` *is* the colon. It must sit immediately after the
    statement header. Inserting at the first body statement leaves leading
    comments/decorators between the header and ``<block_start>``, so decode
    cannot close the header and the colon attaches to the comment.
    """
    body_s, _ = ctx.span(body)
    idx = ctx.source[:body_s].rfind(":")
    if idx < 0:
        return [SpanEdit.insert(body_s, "<block_start>")]
    return [SpanEdit.replace(idx, idx + 1, "<block_start>")]


def _replace_leading_keyword(
    ctx: RuleContext,
    node: cst.CSTNode,
    keyword: str,
    marker: str,
    *,
    allow_async: bool = False,
) -> list[SpanEdit]:
    s, _ = ctx.span(node)
    chunk = ctx.source[s : s + 80]
    lead = len(chunk) - len(chunk.lstrip())
    body = chunk.lstrip()
    if allow_async:
        m = re.match(r"async\s+" + re.escape(keyword) + r"\b", body)
        if m:
            # replace only the keyword, keep ``async`` / ``<async_keyword>``
            kw_start = s + lead + m.end() - len(keyword)
            return [SpanEdit.replace(kw_start, kw_start + len(keyword), marker)]
    m = re.match(re.escape(keyword) + r"\b", body)
    if not m:
        return []
    kw_start = s + lead + m.start()
    return [SpanEdit.replace(kw_start, kw_start + len(keyword), marker)]


def _replace_token_text(
    ctx: RuleContext,
    node: cst.CSTNode,
    old: str,
    new: str,
) -> list[SpanEdit]:
    """Replace a whole-word token (not a substring of an identifier).

    Critical for ``import`` inside module paths like ``utilities.imports``.
    """
    s, e = ctx.span(node)
    chunk = ctx.source[s:e]
    m = re.search(r"\b" + re.escape(old) + r"\b", chunk)
    if not m:
        return []
    return [SpanEdit.replace(s + m.start(), s + m.end(), new)]


# --- rules (mirror rules_py2spy.py) ---


def _meaningful_body_lines(body: object) -> list[cst.CSTNode]:
    """Statements plus comment-only empty lines (for <line_sep> placement)."""
    return [
        s
        for s in body  # type: ignore[union-attr]
        if not (isinstance(s, cst.EmptyLine) and s.comment is None)
    ]


def _is_anchor(ctx: RuleContext, node: cst.CSTNode) -> bool:
    """True if the node sits in a high-attention 'Hotzone' (GCA 2.0)."""
    if ctx.hotzones is not None:
        s, e = ctx.span(node)
        # GCA 2.0: Look slightly beyond the node boundaries to capture adjacent structural chars
        check_s = max(0, s - 2)
        check_e = min(len(ctx.source), e + 2)
        for i in range(check_s, check_e):
            if i in ctx.hotzones:
                return True
        return False

    if ctx.scores is None:
        return False
    # Check current node score
    s, e = ctx.span(node)
    from recode_codec.attention.node_score import node_key
    key = node_key(node, s, e)
    ns = ctx.scores.get(key)
    if ns and getattr(ns, "band", "L") == "H":
        return True
    return False


def module_rule(ctx: RuleContext, node: cst.Module) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    stmts = _meaningful_body_lines(node.body)
    future_line = -1
    for stmt in reversed(stmts):
        line = ctx.line_of(stmt)
        if line != future_line:
            _, end = ctx.span(stmt)
            end = _snap_stmt_end(ctx.source, end)
            marker = "<anchor_sep>" if _is_anchor(ctx, stmt) else "<line_sep>"
            edits.append(SpanEdit.insert(end, marker))
        future_line = line
    edits.extend(_apply_operator_token_rules(ctx, node))
    return edits


def block_rule(ctx: RuleContext, node: cst.IndentedBlock) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    body = _meaningful_body_lines(node.body)
    stmts_for_end = [s for s in node.body if not isinstance(s, cst.EmptyLine)]
    if stmts_for_end:
        _, end = ctx.span(stmts_for_end[-1])
        end = _snap_stmt_end(ctx.source, end)
        edits.append(SpanEdit.insert(end, "<block_end>"))
    future_line = 0
    for stmt in reversed(body):
        line = ctx.line_of(stmt)
        if line != future_line and future_line != 0:
            _, end = ctx.span(stmt)
            end = _snap_stmt_end(ctx.source, end)
            marker = "<anchor_sep>" if _is_anchor(ctx, stmt) else "<line_sep>"
            edits.append(SpanEdit.insert(end, marker))
        future_line = line
    return edits


def _snap_past_mask_comment(source: str, pos: int) -> int:
    """If ``pos`` lands inside ``#MASK`` / ``#…``, move to after that comment.

    LibCST ByteSpanPositionProvider occasionally ends a statement mid-``#MASK``
    (seen as ``#M`` + ``ASK`` desyncing the mask FIFO). Snap to EOL.
    """
    line_start = source.rfind("\n", 0, pos) + 1
    nl = source.find("\n", pos)
    line_end = nl if nl >= 0 else len(source)
    line = source[line_start:line_end]
    rel = pos - line_start
    hash_at = line.find("#")
    if hash_at >= 0 and hash_at <= rel <= len(line.rstrip("\r\n")):
        return line_start + len(line.rstrip("\r\n"))
    return pos


def _snap_stmt_end(source: str, pos: int) -> int:
    """Statement end for ``<line_sep>`` / ``<block_end>`` inserts.

    1. Snap out of mid-``#MASK``.
    2. If ``pos`` sits in the *leading indent* of the next line (AnnAssign
       spans sometimes include ``\\n\\n  `` before the next ``def``), rewind
       to the end of the previous non-empty line's content. Narrower than
       stripping all trailing newlines (that regressed list/assign layouts).
    """
    pos = _snap_past_mask_comment(source, pos)
    if pos <= 0:
        return pos
    line_start = source.rfind("\n", 0, pos) + 1
    # Mid-indent of a subsequent line: only whitespace between line_start and pos.
    if line_start < pos and source[line_start:pos].strip() == "":
        j = line_start - 1
        while j >= 0 and source[j] in "\r\n":
            j -= 1
        while j >= 0 and source[j] in " \t":
            j -= 1
        return j + 1
    return pos


def simple_suite_rule(ctx: RuleContext, node: cst.SimpleStatementSuite) -> list[SpanEdit]:
    _, e = ctx.span(node)
    return [SpanEdit.insert(e, "<block_end>")]


def def_rule(ctx: RuleContext, node: cst.FunctionDef) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    if getattr(node, "asynchronous", False):
        edits.extend(async_rule(ctx, node))
    edits.extend(
        _replace_leading_keyword(
            ctx, node, "def", STMT_KW_MARKERS["def"], allow_async=True
        )
    )
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def class_rule(ctx: RuleContext, node: cst.ClassDef) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "class", STMT_KW_MARKERS["class"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def for_rule(ctx: RuleContext, node: cst.For) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    if getattr(node, "asynchronous", False):
        edits.extend(async_rule(ctx, node))
    edits.extend(
        _replace_leading_keyword(
            ctx, node, "for", STMT_KW_MARKERS["for"], allow_async=True
        )
    )
    edits.extend(_replace_for_in_keyword(ctx, node.target, node.iter))
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def if_rule(ctx: RuleContext, node: cst.If) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "if", STMT_KW_MARKERS["if"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def elif_rule(ctx: RuleContext, node: cst.If) -> list[SpanEdit]:
    """Elif branch: nested If in orelse with leading elif keyword in source."""
    edits = _replace_leading_keyword(ctx, node, "elif", STMT_KW_MARKERS["elif"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    s, _ = ctx.span(node)
    edits.append(SpanEdit.insert(s, "<line_sep>"))
    return edits


def else_rule(ctx: RuleContext, node: cst.Else) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "else", STMT_KW_MARKERS["else"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def while_rule(ctx: RuleContext, node: cst.While) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "while", STMT_KW_MARKERS["while"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def with_rule(ctx: RuleContext, node: cst.With) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    if getattr(node, "asynchronous", False):
        edits.extend(async_rule(ctx, node))
    edits.extend(
        _replace_leading_keyword(
            ctx, node, "with", STMT_KW_MARKERS["with"], allow_async=True
        )
    )
    edits.extend(_with_clause_rule(ctx, node))
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def _with_clause_rule(ctx: RuleContext, node: cst.With) -> list[SpanEdit]:
    """Only touch the with-items clause — never the suite body."""
    edits: list[SpanEdit] = []
    for item in node.items:
        # Optional wrapping parens around with items
        for attr in ("lpar", "rpar"):
            pass

        class V(cst.CSTVisitor):
            def visit_LeftParen(self, n: cst.LeftParen) -> None:
                s, e = ctx.span(n)
                edits.append(SpanEdit.replace(s, e, "<SPACE>"))

            def visit_RightParen(self, n: cst.RightParen) -> None:
                s, e = ctx.span(n)
                edits.append(SpanEdit.replace(s, e, "<SPACE>"))

        item.visit(V())
    # Commas between with-items only (not nested call/dict commas in body).
    edits.extend(_direct_commas(node.items, ctx))
    return edits


def try_rule(ctx: RuleContext, node: cst.Try) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "try", STMT_KW_MARKERS["try"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    return edits


def except_rule(ctx: RuleContext, node: cst.ExceptHandler) -> list[SpanEdit]:
    """Do not walk the suite — nested dict/call commas belong to their own rules."""
    edits = _replace_leading_keyword(ctx, node, "except", STMT_KW_MARKERS["except"])
    # Tuple of exception types: ``except (A, B):`` — only those commas.
    if node.type is not None:
        edits.extend(_direct_commas_if_tuple(node.type, ctx))
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    s, _ = ctx.span(node)
    edits.append(SpanEdit.insert(s, "<line_sep>"))
    return edits


def _direct_commas_if_tuple(node: cst.CSTNode, ctx: RuleContext) -> list[SpanEdit]:
    if isinstance(node, cst.Tuple):
        return _direct_commas(node.elements, ctx)
    return []


def finally_rule(ctx: RuleContext, node: cst.Finally) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "finally", STMT_KW_MARKERS["finally"])
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    s, _ = ctx.span(node)
    edits.append(SpanEdit.insert(s, "<line_sep>"))
    return edits


def match_rule(ctx: RuleContext, node: cst.Match) -> list[SpanEdit]:
    """SimPy ``match_statement``: subject commas + subject ``:`` → ``<block_start>``.

    Case colons are handled by ``case_rule`` (each case has its own suite).
    Do **not** walk into cases here — that would delete case ``:`` markers.
    """
    edits = _replace_leading_keyword(ctx, node, "match", STMT_KW_MARKERS["match"])
    # Multi-subject: ``match a, b:`` (rare) — SimPy spaceSep1 on subjects.
    if isinstance(node.subject, cst.Tuple):
        edits.extend(_direct_commas(node.subject.elements, ctx))
    # Subject colon only (between subject and first case / end of header).
    _, subj_e = ctx.span(node.subject)
    if node.cases:
        case_s, _ = ctx.span(node.cases[0])
        region_end = case_s
    else:
        _, region_end = ctx.span(node)
    chunk = ctx.source[subj_e:region_end]
    idx = chunk.find(":")
    if idx >= 0:
        edits.append(SpanEdit.replace(subj_e + idx, subj_e + idx + 1, "<block_start>"))
    _, end = ctx.span(node)
    edits.append(SpanEdit.insert(end, "<block_end>"))
    return edits


def case_rule(ctx: RuleContext, node: cst.MatchCase) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "case", STMT_KW_MARKERS["case"])
    # Pattern commas handled by MatchList / MatchTuple / MatchClass / … rules.
    edits.extend(_suite_colon_to_block_start(ctx, node, node.body))
    s, _ = ctx.span(node)
    edits.append(SpanEdit.insert(s, "<line_sep>"))
    return edits


def match_list_rule(ctx: RuleContext, node: cst.MatchList) -> list[SpanEdit]:
    """SimPy ``list_pattern``: ``no_comma``."""
    return _direct_commas(node.patterns, ctx)


def match_tuple_rule(ctx: RuleContext, node: cst.MatchTuple) -> list[SpanEdit]:
    """SimPy ``tuple_pattern``: ``no_comma``."""
    return _direct_commas(node.patterns, ctx)


def match_class_rule(ctx: RuleContext, node: cst.MatchClass) -> list[SpanEdit]:
    """SimPy ``class_pattern``: ``no_comma`` on positional + keyword patterns."""
    edits = _direct_commas(node.patterns, ctx)
    edits.extend(_direct_commas(node.kwds, ctx))
    return edits


def match_mapping_rule(ctx: RuleContext, node: cst.MatchMapping) -> list[SpanEdit]:
    return _direct_commas(node.elements, ctx)


def match_or_rule(ctx: RuleContext, node: cst.MatchOr) -> list[SpanEdit]:
    # Alternatives use ``|``, not commas.
    return []


def _direct_commas_shallow(node: cst.CSTNode, ctx: RuleContext) -> list[SpanEdit]:
    """Replace commas that are *direct* children of ``node``, not nested deeper."""
    edits: list[SpanEdit] = []
    for child in node.children:
        if isinstance(child, cst.Comma):
            s, e = ctx.span(child)
            edits.append(SpanEdit.replace(s, e, "<SPACE>"))
    return edits


def as_name_rule(ctx: RuleContext, node: cst.AsName) -> list[SpanEdit]:
    """SimPy EXPR ``as`` → ``<as>`` (with/except/import aliases)."""
    s, e = ctx.span(node)
    # AsName span covers `` as name``; locate the keyword itself.
    chunk = ctx.source[s:e]
    m = re.search(r"\bas\b", chunk)
    if not m:
        return []
    start = s + m.start()
    return [SpanEdit.replace(start, start + 2, "<as>")]


def import_alias_rule(ctx: RuleContext, node: cst.ImportAlias) -> list[SpanEdit]:
    # ``as`` handled by ``as_name_rule`` on nested AsName.
    return []


def _import_list_commas(names: object, ctx: RuleContext) -> list[SpanEdit]:
    """Replace inter-name commas with space; drop trailing commas (no paren restore).

    Spy ``_import_list`` ends with ``optional(' ')``; a trailing SPACE would decode
    to a trailing ``,`` which is illegal without surrounding parentheses.
    """
    edits: list[SpanEdit] = []
    if names is None or isinstance(names, cst.ImportStar):
        return edits
    seq = list(names)  # type: ignore[arg-type]
    for i, el in enumerate(seq):
        comma = getattr(el, "comma", None)
        if comma is None or isinstance(comma, cst.MaybeSentinel):
            continue
        cs, ce = ctx.span(comma)
        if i == len(seq) - 1:
            edits.append(SpanEdit.replace(cs, ce, ""))
        else:
            edits.append(SpanEdit.replace(cs, ce, "<SPACE>"))
    return edits


def _flatten_import_newlines(node: cst.CSTNode, ctx: RuleContext) -> list[SpanEdit]:
    """Spy import lists are space-separated on one line; drop paren-style newlines."""
    edits: list[SpanEdit] = []
    s, e = ctx.span(node)
    chunk = ctx.source[s:e]
    for m in re.finditer(r"[\r\n]+", chunk):
        edits.append(SpanEdit.replace(s + m.start(), s + m.end(), " "))
    return edits


def import_rule(ctx: RuleContext, node: cst.Import) -> list[SpanEdit]:
    edits = _replace_leading_keyword(ctx, node, "import", "<import_stmt>")
    edits.extend(_import_list_commas(node.names, ctx))
    edits.extend(_flatten_import_newlines(node, ctx))

    class V(cst.CSTVisitor):
        def visit_LeftParen(self, n: cst.LeftParen) -> None:
            s, e = ctx.span(n)
            edits.append(SpanEdit.replace(s, e, " "))

        def visit_RightParen(self, n: cst.RightParen) -> None:
            s, e = ctx.span(n)
            # Empty — a trailing SPACE would decode to a trailing comma.
            edits.append(SpanEdit.replace(s, e, ""))

    node.visit(V())
    return edits


def import_from_rule(ctx: RuleContext, node: cst.ImportFrom) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    is_future = (
        node.module is not None
        and isinstance(node.module, cst.Name)
        and node.module.value == "__future__"
    )
    marker = "<import_from_future_stmt>" if is_future else "<import_from_stmt>"
    s, _ = ctx.span(node)
    chunk = ctx.source[s : s + 40]
    m = re.search(r"\bfrom\b", chunk)
    if m:
        fs = s + m.start()
        edits.append(SpanEdit.replace(fs, fs + 4, marker))
    if is_future:
        edits.extend(_replace_token_text(ctx, node, "__future__", ""))
    edits.extend(_replace_token_text(ctx, node, "import", "<SPACE>"))
    if node.names:
        edits.extend(_import_list_commas(node.names, ctx))
    edits.extend(_flatten_import_newlines(node, ctx))

    class V(cst.CSTVisitor):
        def visit_LeftParen(self, n: cst.LeftParen) -> None:
            a, b = ctx.span(n)
            edits.append(SpanEdit.replace(a, b, " "))

        def visit_RightParen(self, n: cst.RightParen) -> None:
            a, b = ctx.span(n)
            edits.append(SpanEdit.replace(a, b, ""))

    node.visit(V())
    return edits


def _param_inner(text: str) -> str:
    """Depth-0 commas → spaces; drop trailing comma/SPACE placeholders."""
    inner = _replace_commas_depth0(text).strip()
    inner = re.sub(r"(?:<SPACE>|\s|,)+$", "", inner)
    return inner.strip()


def _strip_line_comments(text: str) -> str:
    """Drop ``#...`` tails (param lists); join leftover fragments with space."""
    parts: list[str] = []
    for line in text.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        frag = line.strip()
        if frag:
            parts.append(frag)
    return " ".join(parts)


def parameters_rule(ctx: RuleContext, node: cst.Parameters) -> list[SpanEdit]:
    """SimPy ``parameters`` / ``lambda_parameters``.

    - FunctionDef: strip ``()`` and depth-0 commas (restore reinserts parens).
    - Lambda: only depth-0 commas (no surrounding parens in source; SimPy
      ``lambda_parameters: no_comma``).
    Drop trailing commas — a trailing SPACE would decode to ``, )``.
    """
    s, e = ctx.span(node)
    try:
        parent = ctx.metadata.get_metadata(ParentNodeProvider, node)  # type: ignore[attr-defined]
    except Exception:
        parent = None
    if isinstance(parent, cst.Lambda):
        inner = _param_inner(ctx.source[s:e])
        if inner == ctx.source[s:e].strip().rstrip(",") and "," not in ctx.source[s:e]:
            return []
        return [SpanEdit.replace(s, e, inner if inner else "")]
    if isinstance(parent, cst.FunctionDef):
        _, name_e = ctx.span(parent.name)
        left = ctx.source.find("(", name_e)
        if left >= 0:
            right = _find_matching_paren(ctx.source, left)
            if right >= 0 and right >= e - 1:
                inside = ctx.source[left + 1 : right]
                # Skip param compression when comments are present.
                # Spy ``comment: #.*`` + spaceSep1 params cannot round-trip
                # inline ``#`` tails. Stripping ``#MASK`` is forbidden — it
                # desyncs the mask FIFO (f-string / string restore corruption).
                # Do NOT skip on type annotations alone — that regressed SC RT
                # from 98%→86% (many defs keep source parens but lose colon
                # append in ERROR regions for unrelated reasons).
                if "#" in inside or "#" in ctx.source[name_e:left]:
                    return []
                inner = _param_inner(inside)
                text = f" {inner}" if inner else " "
                return [SpanEdit.replace(left, right + 1, text)]
    inner = _param_inner(ctx.source[s:e])
    return [SpanEdit.replace(s, e, f" {inner}" if inner else " ")]


def _replace_for_in_keyword(ctx: RuleContext, target: cst.CSTNode, iter_node: cst.CSTNode) -> list[SpanEdit]:
    """Replace only the 'in' between for-target and iterable (not guard 'in')."""
    _, target_e = ctx.span(target)
    iter_s, _ = ctx.span(iter_node)
    chunk = ctx.source[target_e:iter_s]
    m = re.search(r"\bin\b", chunk)
    if not m:
        return []
    pos = target_e + m.start()
    return [SpanEdit.replace(pos, pos + 2, "<SPACE>")]


def _replace_comp_if_keywords(ctx: RuleContext, node: cst.CompFor) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    for guard in node.ifs:
        s, e = ctx.span(guard)
        chunk = ctx.source[s:e].lstrip()
        if not chunk:
            continue
        offset = len(ctx.source[s:e]) - len(chunk)
        m = re.match(r"\bif\b", chunk)
        if m:
            start = s + offset + m.start()
            edits.append(SpanEdit.replace(start, start + 2, "<if>"))
    return edits


def no_comma_rule(ctx: RuleContext, node: cst.CSTNode) -> list[SpanEdit]:
    return _visit_commas(node, ctx)


def call_rule(ctx: RuleContext, node: cst.Call) -> list[SpanEdit]:
    """SimPy ``argument_list``: drop all direct arg commas (``spaceSep1``)."""
    return _direct_commas(node.args, ctx)


def list_rule(ctx: RuleContext, node: cst.List) -> list[SpanEdit]:
    """SimPy ``list``: ``no_comma`` on direct elements."""
    return _direct_commas(node.elements, ctx)


def set_rule(ctx: RuleContext, node: cst.Set) -> list[SpanEdit]:
    """SimPy ``set``: ``no_comma``."""
    return _direct_commas(node.elements, ctx)


def dict_rule(ctx: RuleContext, node: cst.Dict) -> list[SpanEdit]:
    """SimPy ``dictionary``: ``no_comma``."""
    return _direct_commas(node.elements, ctx)


def subscript_rule(ctx: RuleContext, node: cst.Subscript) -> list[SpanEdit]:
    """SimPy ``subscript``: ``no_comma`` (slice ``:`` is not a comma)."""
    return _direct_commas(node.slice, ctx)


def tuple_rule(ctx: RuleContext, node: cst.Tuple) -> list[SpanEdit]:
    """SimPy ``tuple`` / ``expression_list``: drop commas.

    Keep commas only for assignment / for / comp *targets* — SimPy grammar
    ``pattern_list`` retains ``,`` there (not compressed).
    """
    try:
        parent = ctx.metadata.get_metadata(ParentNodeProvider, node)  # type: ignore[attr-defined]
    except Exception:
        parent = None
    if isinstance(parent, (cst.CompFor, cst.For)) and parent.target is node:
        return []
    if isinstance(parent, cst.AssignTarget):
        return []
    return _direct_commas(node.elements, ctx)


def comp_for_rule(ctx: RuleContext, node: cst.CompFor) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    # Mid-comprehension ``#`` (e.g. ``# type: ignore``) breaks spy
    # ``for_in_clause`` restore (``in`` space → ``, `` via parent no_comma).
    # Skip keyword packing for this clause; keep source ``for``/``in``/``if``.
    s, e = ctx.span(node)
    if "#" in ctx.source[s:e]:
        if node.inner_for_in:
            edits.extend(comp_for_rule(ctx, node.inner_for_in))
        return edits
    # ``async for`` in comprehensions must mark async, else decode no_comma
    # turns ``async for`` into ``, async, for,``.
    if getattr(node, "asynchronous", None):
        edits.extend(async_rule(ctx, node))
    edits.extend(
        _replace_leading_keyword(ctx, node, "for", "<for>", allow_async=True)
    )
    edits.extend(_replace_for_in_keyword(ctx, node.target, node.iter))
    edits.extend(_replace_comp_if_keywords(ctx, node))
    if node.inner_for_in:
        edits.extend(comp_for_rule(ctx, node.inner_for_in))
    return edits


def name_rule(ctx: RuleContext, node: cst.Name) -> list[SpanEdit]:
    if node.value in ("True", "False", "None"):
        s, e = ctx.span(node)
        # Guard stale ByteSpan (seen replacing ``:\n  `` with ``<none>``).
        if ctx.source[s:e] != node.value:
            return []
        return [SpanEdit.replace(s, e, EXPR_KW_MARKERS[node.value])]
    return []


def conditional_rule(ctx: RuleContext, node: cst.IfExp) -> list[SpanEdit]:
    edits: list[SpanEdit] = []
    s, e = ctx.span(node)
    chunk = ctx.source[s:e]
    for kw, marker in (("if", "<if>"), ("else", "<else>")):
        for m in re.finditer(r"\b" + kw + r"\b", chunk):
            edits.append(SpanEdit.replace(s + m.start(), s + m.end(), marker))
    return edits


def binary_op_rule(ctx: RuleContext, node: cst.BinaryOperation) -> list[SpanEdit]:
    _, left_e = ctx.span(node.left)
    right_s, _ = ctx.span(node.right)
    op_slice = ctx.source[left_e:right_s]
    m = re.search(r"\*\*", op_slice)
    if m:
        return [SpanEdit.replace(left_e + m.start(), left_e + m.end(), OPERATORS["**"])]
    m = re.search(r"\*", op_slice)
    if m:
        return [SpanEdit.replace(left_e + m.start(), left_e + m.end(), OPERATORS["*"])]
    return []


def lambda_rule(ctx: RuleContext, node: cst.Lambda) -> list[SpanEdit]:
    s, e = ctx.span(node)
    m = re.search(r"\blambda\b", ctx.source[s:e])
    if not m:
        return []
    start = s + m.start()
    return [SpanEdit.replace(start, start + 6, "<lambda>")]


def await_rule(ctx: RuleContext, node: cst.Await) -> list[SpanEdit]:
    s, e = ctx.span(node)
    m = re.search(r"\bawait\b", ctx.source[s:e])
    if not m:
        return []
    start = s + m.start()
    return [SpanEdit.replace(start, start + 5, EXPR_KW_MARKERS["await"])]


def yield_rule(ctx: RuleContext, node: cst.Yield) -> list[SpanEdit]:
    s, e = ctx.span(node)
    m = re.search(r"\byield\b", ctx.source[s:e])
    if not m:
        return []
    start = s + m.start()
    edits = [SpanEdit.replace(start, start + 5, EXPR_KW_MARKERS["yield"])]
    # ``yield from xs`` — LibCST wraps iterable in ``From``; mark ``from``.
    if isinstance(node.value, cst.From):
        chunk = ctx.source[s:e]
        fm = re.search(r"\bfrom\b", chunk)
        if fm:
            pos = s + fm.start()
            edits.append(SpanEdit.replace(pos, pos + 4, EXPR_KW_MARKERS["from"]))
    return edits


def unary_op_rule(ctx: RuleContext, node: cst.UnaryOperation) -> list[SpanEdit]:
    if not isinstance(node.operator, cst.Not):
        return []
    s, e = ctx.span(node.operator)
    return [SpanEdit.replace(s, e, EXPR_KW_MARKERS["not"])]


def ellipsis_rule(ctx: RuleContext, node: cst.Ellipsis) -> list[SpanEdit]:
    s, e = ctx.span(node)
    return [SpanEdit.replace(s, e, "<ellipsis>")]


def concat_string_rule(ctx: RuleContext, node: cst.ConcatenatedString) -> list[SpanEdit]:
    """SimPy: insert ``<concat>`` before every string after the first."""
    edits: list[SpanEdit] = []
    seen = False

    class V(cst.CSTVisitor):
        def visit_SimpleString(self, n: cst.SimpleString) -> bool:
            nonlocal seen
            if seen:
                s, _ = ctx.span(n)
                edits.append(SpanEdit.insert(s, "<concat>"))
            seen = True
            return False

        def visit_FormattedString(self, n: cst.FormattedString) -> bool:
            nonlocal seen
            if seen:
                s, _ = ctx.span(n)
                edits.append(SpanEdit.insert(s, "<concat>"))
            seen = True
            return False

    node.visit(V())
    return edits


def async_rule(ctx: RuleContext, node: cst.CSTNode) -> list[SpanEdit]:
    """Mark the ``async`` keyword on FunctionDef / For / With / CompFor."""
    s, e = ctx.span(node)
    chunk = ctx.source[s:e]
    m = re.search(r"\basync\b", chunk)
    if not m:
        return []
    start = s + m.start()
    return [SpanEdit.replace(start, start + 5, "<async_keyword>")]


def semicolon_rule(ctx: RuleContext, node: cst.Semicolon) -> list[SpanEdit]:
    s, e = ctx.span(node)
    return [SpanEdit.replace(s, e, "<line_sep>")]


def _stmt_keyword_rule(keyword: str, marker: str) -> Callable[[RuleContext, cst.CSTNode], list[SpanEdit]]:
    def rule(ctx: RuleContext, node: cst.CSTNode) -> list[SpanEdit]:
        return _replace_leading_keyword(ctx, node, keyword, marker)

    return rule


def _small_stmt_rules(ctx: RuleContext, node: cst.CSTNode, keyword: str, marker: str) -> list[SpanEdit]:
    """Keyword marker + SimPy ``no_comma`` on the statement node (direct commas)."""
    edits = _replace_leading_keyword(ctx, node, keyword, marker)
    edits.extend(_direct_commas_from_node(node, ctx))
    return edits


def raise_rule(ctx: RuleContext, node: cst.Raise) -> list[SpanEdit]:
    """``raise`` / ``raise Exc from cause`` — mark ``from`` as ``<from>``.

    Bare ``from`` confuses spy decode into an import-like split
    (``raise …\\nfrom, e``).
    """
    edits = _replace_leading_keyword(ctx, node, "raise", EXPR_KW_MARKERS["raise"])
    if node.cause is not None:
        # Search the whole raise stmt after ``raise`` — LibCST cause spans may
        # start at ``from`` itself, so exc_e:cause_s can miss the keyword.
        s, e = ctx.span(node)
        chunk = ctx.source[s:e]
        m = re.search(r"\bfrom\b", chunk)
        if m:
            pos = s + m.start()
            edits.append(SpanEdit.replace(pos, pos + 4, EXPR_KW_MARKERS["from"]))
    edits.extend(_direct_commas_from_node(node, ctx))
    return edits


def _direct_commas_from_node(node: cst.CSTNode, ctx: RuleContext) -> list[SpanEdit]:
    """Drop commas that are direct children attributes (names / args / …)."""
    edits: list[SpanEdit] = []
    for attr in ("names", "args", "elements", "targets"):
        seq = getattr(node, attr, None)
        if seq is not None:
            edits.extend(_direct_commas(seq, ctx))
    # Assert: msg is separate; comma between test and msg
    if isinstance(node, cst.Assert) and node.msg is not None:
        # Find comma between test and msg in source
        _, te = ctx.span(node.test)
        ms, _ = ctx.span(node.msg)
        chunk = ctx.source[te:ms]
        idx = chunk.find(",")
        if idx >= 0:
            edits.append(SpanEdit.replace(te + idx, te + idx + 1, "<SPACE>"))
    return edits


# LibCST type → rule (mirrors TRANSFORM_RULES)
TRANSFORM_RULES: dict[type, Callable[[RuleContext, cst.CSTNode], list[SpanEdit]]] = {
    cst.Module: module_rule,
    cst.IndentedBlock: block_rule,
    cst.SimpleStatementSuite: simple_suite_rule,
    cst.FunctionDef: def_rule,
    cst.ClassDef: class_rule,
    cst.For: for_rule,
    cst.If: if_rule,
    cst.Else: else_rule,
    cst.While: while_rule,
    cst.With: with_rule,
    cst.Try: try_rule,
    cst.ExceptHandler: except_rule,
    cst.Finally: finally_rule,
    cst.Match: match_rule,
    cst.MatchCase: case_rule,
    cst.MatchList: match_list_rule,
    cst.MatchTuple: match_tuple_rule,
    cst.MatchClass: match_class_rule,
    cst.MatchMapping: match_mapping_rule,
    cst.MatchOr: match_or_rule,
    cst.Import: import_rule,
    cst.ImportFrom: import_from_rule,
    cst.ImportAlias: import_alias_rule,
    cst.AsName: as_name_rule,
    cst.Parameters: parameters_rule,
    cst.CompFor: comp_for_rule,
    cst.IfExp: conditional_rule,
    cst.Lambda: lambda_rule,
    cst.BinaryOperation: binary_op_rule,
    cst.ConcatenatedString: concat_string_rule,
    cst.List: list_rule,
    cst.Tuple: tuple_rule,
    cst.Set: set_rule,
    cst.Dict: dict_rule,
    cst.Call: call_rule,
    cst.Subscript: subscript_rule,
    cst.Await: await_rule,
    cst.Yield: yield_rule,
    cst.UnaryOperation: unary_op_rule,
    cst.Ellipsis: ellipsis_rule,
    cst.Name: name_rule,
    cst.Assert: lambda ctx, n: _small_stmt_rules(ctx, n, "assert", STMT_KW_MARKERS["assert"]),
    cst.Del: lambda ctx, n: _small_stmt_rules(ctx, n, "del", STMT_KW_MARKERS["del"]),
    cst.Global: lambda ctx, n: _small_stmt_rules(ctx, n, "global", EXPR_KW_MARKERS["global"]),
    cst.Nonlocal: lambda ctx, n: _small_stmt_rules(ctx, n, "nonlocal", EXPR_KW_MARKERS["nonlocal"]),
    cst.Return: lambda ctx, n: _small_stmt_rules(ctx, n, "return", EXPR_KW_MARKERS["return"]),
    cst.Raise: raise_rule,
    cst.Pass: lambda ctx, n: _small_stmt_rules(ctx, n, "pass", EXPR_KW_MARKERS["pass"]),
    cst.Break: lambda ctx, n: _small_stmt_rules(ctx, n, "break", EXPR_KW_MARKERS["break"]),
    cst.Continue: lambda ctx, n: _small_stmt_rules(ctx, n, "continue", EXPR_KW_MARKERS["continue"]),
    cst.Semicolon: semicolon_rule,
}

# Comparison / aug / unary operators — patch operator text in span
_COMP_OP_TEXT = {
    cst.Equal: OPERATORS["=="],
    cst.NotEqual: OPERATORS["!="],
    cst.LessThan: OPERATORS["<"],
    cst.GreaterThan: OPERATORS[">"],
    cst.LessThanEqual: OPERATORS["<="],
    cst.GreaterThanEqual: OPERATORS[">="],
    cst.Is: EXPR_KW_MARKERS["is"],
    cst.In: EXPR_KW_MARKERS["in"],
    cst.IsNot: OPERATORS["is not"],
    cst.NotIn: OPERATORS["not in"],
}

_AUG_OP_TEXT = {v: k for k, v in OPERATORS.items() if k.endswith("=")}


def comparison_rule(ctx: RuleContext, node: cst.Comparison) -> list[SpanEdit]:
    """Map comparison ops; ``is not`` / ``not in`` are single markers (SimPy)."""
    edits: list[SpanEdit] = []
    for comp in node.comparisons:
        op = comp.operator
        marker = _COMP_OP_TEXT.get(type(op))
        if marker:
            s, e = ctx.span(op)
            edits.append(SpanEdit.replace(s, e, marker))
    return edits


def augassign_rule(ctx: RuleContext, node: cst.AugAssign) -> list[SpanEdit]:
    s, e = ctx.span(node.operator)
    text = ctx.source[s:e]
    for op_text, marker in OPERATORS.items():
        if op_text.endswith("=") and text.strip() == op_text:
            return [SpanEdit.replace(s, e, marker)]
    return []


def boolean_op_rule(ctx: RuleContext, node: cst.BooleanOperation) -> list[SpanEdit]:
    s, e = ctx.span(node.operator)
    if isinstance(node.operator, cst.And):
        return [SpanEdit.replace(s, e, EXPR_KW_MARKERS["and"])]
    if isinstance(node.operator, cst.Or):
        return [SpanEdit.replace(s, e, EXPR_KW_MARKERS["or"])]
    return []


TRANSFORM_RULES.update(
    {
        cst.Comparison: comparison_rule,
        cst.AugAssign: augassign_rule,
        cst.BooleanOperation: boolean_op_rule,
    }
)

# Elif detection: If whose source starts with elif
_orig_if = TRANSFORM_RULES[cst.If]


def _if_or_elif(ctx: RuleContext, node: cst.If) -> list[SpanEdit]:
    s, _ = ctx.span(node)
    if re.match(r"\s*elif\b", ctx.source[s : s + 20]):
        return elif_rule(ctx, node)
    return _orig_if(ctx, node)


TRANSFORM_RULES[cst.If] = _if_or_elif


# When GCA disables optional Else/While (keyword sugar gated off), still emit
# ``:`` → ``<block_start>`` so spy densify/decode stays reversible. Keyword stays
# raw ``else`` / ``while`` (less compression, RT-safe).
def _else_spine(ctx: RuleContext, node: cst.Else) -> list[SpanEdit]:
    return _suite_colon_to_block_start(ctx, node, node.body)


def _while_spine(ctx: RuleContext, node: cst.While) -> list[SpanEdit]:
    return _suite_colon_to_block_start(ctx, node, node.body)


SPINE_FALLBACK_RULES: dict[type, Callable[[RuleContext, cst.CSTNode], list[SpanEdit]]] = {
    cst.Else: _else_spine,
    cst.While: _while_spine,
}


def _apply_operator_token_rules(ctx: RuleContext, node: cst.CSTNode) -> list[SpanEdit]:
    """Floor div, shifts, arrow — patch operator tokens in source spans."""
    _OP_TOKEN_RULES = {
        "->": OPERATORS["->"],
        "//": OPERATORS["//"],
        "<<": OPERATORS["<<"],
        ">>": OPERATORS[">>"],
    }
    edits: list[SpanEdit] = []
    s, e = ctx.span(node)
    chunk = ctx.source[s:e]
    for op_text, marker in sorted(_OP_TOKEN_RULES.items(), key=lambda x: -len(x[0])):
        start = 0
        while True:
            idx = chunk.find(op_text, start)
            if idx < 0:
                break
            abs_start = s + idx
            edits.append(SpanEdit.replace(abs_start, abs_start + len(op_text), marker))
            start = idx + len(op_text)
    return edits

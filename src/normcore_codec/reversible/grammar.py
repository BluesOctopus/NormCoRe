"""NormCoRe grammar recovery: parse + node rules + post-fix fallbacks.

Pipeline: ensure layout markers → tree-sitter (``spy_lang.so``) →
``grammar_rules.TRANSFORM_RULES`` node edits → regex/heuristic repairs for
unparsed regions and edge layouts. Grammar lineage is SimPy's dialect;
decode orchestration and fallbacks are NormCoRe's.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re  # type: ignore[no-redef]

from tree_sitter import Language, Parser

from normcore_codec.reversible.grammar_edits import apply_spy_edits, traverse_all_children
from normcore_codec.reversible.grammar_rules import TRANSFORM_RULES

_TOKENS_WITHOUT_LINE_SEP = re.compile(r"(<[a-z_]+_[a-z]+>)+")

_KEEP_AS_IS = frozenset(
    {
        "<line_sep>",
        "<block_start>",
        "<case_stmt>",
        "<elif_stmt>",
        "<else_stmt>",
        "<finally_stmt>",
        "<except_stmt>",
    }
)

_parsers: dict[Path, Parser] = {}


def _load_language(path: Path):
    """Load the bundled grammar across tree-sitter Python API generations."""
    try:
        # tree-sitter <= 0.20 accepts a shared-library path and language name.
        return Language(str(path), "python")
    except TypeError:
        # Newer releases accept the raw pointer returned by the grammar's
        # exported tree_sitter_python symbol instead.
        library = ctypes.CDLL(str(path))
        factory = library.tree_sitter_python
        factory.restype = ctypes.c_void_p
        pointer = factory()
        if not pointer:
            raise RuntimeError(f"grammar did not expose tree_sitter_python: {path}")
        return Language(pointer)


def _get_parser(grammar_binary: str | Path) -> Parser:
    path = Path(grammar_binary).resolve()
    if path not in _parsers:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; provide the locked NormCoRe grammar binary"
            )
        lang = _load_language(path)
        p = Parser()
        if hasattr(p, "set_language"):
            p.set_language(lang)
        else:
            p.language = lang
        _parsers[path] = p
    return _parsers[path]


def _ensure_line_seps(spy_code: str) -> str:
    """SimPy ``decode`` preamble: insert ``<line_sep>`` before glued stmt markers."""
    matches = list(_TOKENS_WITHOUT_LINE_SEP.finditer(spy_code))
    if not matches:
        return spy_code
    py_code = ""
    prev_end = 0
    for match in matches:
        # regex ``allcaptures`` is regex-module specific; fall back to group(0).
        token = match.group(0)
        first = token  # whole run; SimPy checks first capture group of nested groups
        # Mimic SimPy: if the match starts with a keep-as-is marker, don't prepend.
        keep = any(token.startswith(k) or token == k for k in _KEEP_AS_IS)
        # More faithful: SimPy uses match.allcaptures()[1][0] — first inner group.
        if hasattr(match, "allcaptures"):
            caps = match.allcaptures()
            if len(caps) > 1 and caps[1]:
                first = caps[1][0]
                keep = first in _KEEP_AS_IS
        if keep:
            py_code += spy_code[prev_end : match.end()]
        else:
            py_code += spy_code[prev_end : match.start()] + "<line_sep>" + match.group(0)
        prev_end = match.end()
    py_code += spy_code[prev_end:]
    return py_code


def decode_spy(spy_code: str, *, grammar_binary: str | Path) -> str:
    """Dialect → Python using spy grammar node edits (``no_comma`` etc.)."""
    text = spy_code.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    py_code = _ensure_line_seps(text)
    tree = _get_parser(grammar_binary).parse(bytes(py_code, "utf8"))
    nodes: list = []
    traverse_all_children(tree.root_node, nodes)
    edits = []
    for node in nodes:
        edit_func = TRANSFORM_RULES.get(node.type)
        if edit_func:
            edits.extend(edit_func(node))
    out = apply_spy_edits(py_code, edits)
    # Safety net: markers the grammar missed (broken trees / edge layouts)
    out = out.replace("<concat>", " ")
    out = out.replace("<async_keyword>", "async ")
    out = out.replace("<SPACE>", " ")
    out = out.replace("<arrow>", "->")
    # Layout markers left in unparsed regions
    out = out.replace("<block_start>", ":\n")
    out = out.replace("<block_end>", "\n")
    out = out.replace("<line_sep>", "\n")
    # Leftover stmt markers (unparsed regions) — better than leaving raw tags
    for marker, repl in (
        ("<for_stmt>", "for "),
        ("<if_stmt>", "if "),
        ("<while_stmt>", "while "),
        ("<with_stmt>", "with "),
        ("<try_stmt>", "try:"),
        ("<def_stmt>", "def "),
        ("<class_stmt>", "class "),
        ("<else_stmt>", "else:"),
        ("<elif_stmt>", "elif "),
        ("<except_stmt>", "except "),
        ("<finally_stmt>", "finally:"),
        ("<import_stmt>", "import "),
        ("<import_from_stmt>", "from "),
        ("<assert_stmt>", "assert "),
        ("<return>", "return "),
        ("<raise>", "raise "),
        ("<yield>", "yield "),
        ("<from>", " from "),
        ("<for>", " for "),
        ("<if>", " if "),
        ("<else>", " else "),
        ("<notin>", " not in "),
        ("<isnot>", " is not "),
        ("<in>", " in "),
        ("<as>", " as "),
        ("<eq>", "=="),
        ("<ne>", "!="),
        ("<le>", "<="),
        ("<ge>", ">="),
        ("<l>", "<"),
        ("<g>", ">"),
        ("<true>", "True"),
        ("<false>", "False"),
        ("<none>", "None"),
        ("<await>", "await "),
        ("<async_keyword>", "async "),
    ):
        if marker in out:
            out = out.replace(marker, repl)
    # Unparsed ``for target iterable`` (missing ``in``).
    _FOR_TGT = r"(?:[A-Za-z_][\w\.]*|\([^)\n]+\)|\[[^\]\n]+\])"
    _FOR_ITER = r"(?:[A-Za-z_][\w\.]*(?:\[[^\]]+\])?(?:\([^)]*\))?)"
    out = re.sub(
        rf"\bfor\s+({_FOR_TGT})\s+(?!in\b)({_FOR_ITER})([ \t]*:)",
        r"for \1 in \2\3",
        out,
    )
    out = re.sub(
        rf"(?m)\bfor\s+({_FOR_TGT})\s+(?!in\b)({_FOR_ITER})[ \t]*$",
        r"for \1 in \2:",
        out,
    )
    out = re.sub(
        rf"\bfor\s+({_FOR_TGT})\s*,\s+(?!in\b)({_FOR_ITER})(?=\s*[\]\}}\)])",
        r"for \1 in \2",
        out,
    )
    out = re.sub(
        rf"\bfor\s+({_FOR_TGT})\s+(?!in\b)({_FOR_ITER})(?=\s*(?:if\b|[\]\}}\)\,]))",
        r"for \1 in \2",
        out,
    )
    # ``else`` / ``finally`` / ``try`` line without colon
    out = re.sub(r"(?m)^([ \t]*)(else|finally|try)[ \t]*$", r"\1\2:", out)
    # Trailing commas before suite ``:`` only (with/for/def heads).
    # NEVER strip ``, :`` inside ``[]`` — that is valid multi-slice syntax
    # (``x[:, :, :3]``). Blind ``,\s*:`` → ``:`` was collapsing slices to ``:::``.
    out = re.sub(
        r"(?m)^([ \t]*(?:with|for|async for|def|class|while|if|elif|except)[^:\n]*),\s*:",
        r"\1:",
        out,
    )
    # Comp ``if`` / ``for`` / ``in`` mis-decoded as ``, if,`` / ``, for,`` /
    # ``, in,`` when a mid-comprehension comment broke ``for_in_clause``.
    out = re.sub(r",\s*if\s*,", " if ", out)
    out = re.sub(r",\s*for\s*,", " for ", out)
    out = re.sub(r",\s*in\s*,", " in ", out)
    # Empty call / index tails left by comment-eaten args.
    out = re.sub(r",\s*\)", ")", out)
    out = re.sub(r",\s*\]", "]", out)
    # Soft line-continuation backslash left before an identifier
    out = re.sub(r"\\(?=[ \t]*[A-Za-z_])", "", out)
    # ``import, name`` / ``from, x, import, y`` from total no_comma fallback
    out = re.sub(r"(?m)^import,\s+", "import ", out)
    out = re.sub(r"(?m)^from,\s+", "from ", out)
    out = re.sub(r"\bfrom\s+([A-Za-z_][\w\.]*)\s*,\s*import\s*,\s*", r"from \1 import ", out)
    out = re.sub(r"\bimport\s*,\s*", "import ", out)
    # ``import x, as, y`` / ``except, E`` / ``def, name`` / ``class, Name`` leftovers
    out = re.sub(r",\s*as\s*,", " as ", out)
    out = re.sub(r"(?m)^([ \t]*)except,\s*", r"\1except ", out)
    out = re.sub(r"(?m)^([ \t]*)def,\s+", r"\1def ", out)
    out = re.sub(r"(?m)^([ \t]*)class,\s+", r"\1class ", out)
    # ``import\n module`` — import_statement missed in ERROR region.
    out = re.sub(
        r"(?m)^([ \t]*)import\s*\n[ \t]*([A-Za-z_])",
        r"\1import \2",
        out,
    )
    # Compound headers that lost ``:`` in ERROR regions (leaf ``<block_start>``
    # still indented the body). Only fire when the next non-empty line is more
    # indented — avoids the wide ``if …:`` nets that regressed SC RT to ~86%.
    def _add_missing_suite_colons(text: str) -> str:
        lines = text.splitlines(keepends=True)
        hdr = re.compile(
            r"^([ \t]*)(async[ \t]+(?:for|with)|(?:if|elif|while|for|with|except))\b"
        )
        out_lines: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            bare = line.rstrip("\r\n")
            m = hdr.match(bare)
            if m and not bare.rstrip().endswith(":"):
                indent = m.group(1)
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines):
                    next_bare = lines[j].rstrip("\r\n")
                    next_indent = len(next_bare) - len(next_bare.lstrip(" \t"))
                    if next_indent > len(indent):
                        nl = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
                        out_lines.append(bare + ":" + nl)
                        i += 1
                        continue
            out_lines.append(line)
            i += 1
        return "".join(out_lines)

    out = _add_missing_suite_colons(out)
    # ``param), ->`` trailing comma before close-paren (comment/no_comma noise).
    out = re.sub(r"\),\s*->", ") ->", out)
    # Leading ``, NAME=`` / ``, # comment`` from no_comma after comment split.
    out = re.sub(r"(?m)^([ \t]*),\s*([A-Za-z_]\w*\s*=)", r"\1\2", out)
    out = re.sub(r"(?m)^[ \t]*,\s*(#[^\n]*)$", r"\1", out)
    # Comp leading ``, if`` / ``, for`` after comment-split.
    out = re.sub(r"(?m)^([ \t]*),\s*(if|for)\b", r"\1\2", out)
    # ``from, mod, import, name`` / ``from mod, import name`` leftovers.
    out = re.sub(
        r"(?m)^([ \t]*)from,\s+([A-Za-z_][\w\.]*)\s*,\s*import\s*,\s*",
        r"\1from \2 import ",
        out,
    )
    out = re.sub(
        r"(?m)^([ \t]*)from,\s+([A-Za-z_][\w\.]*)\s*,\s*import\s+",
        r"\1from \2 import ",
        out,
    )
    out = re.sub(
        r"(?m)^([ \t]*)from\s+([A-Za-z_][\w\.]*)\s*,\s*import\s+",
        r"\1from \2 import ",
        out,
    )
    # ``from mod, import name`` — spy import_from / ERROR-region no_comma.
    # Use a permissive module path (dots/underscores); apply late so it
    # catches commas introduced by ``import_from_stmt`` space→comma edits.
    out = re.sub(
        r"(?m)^([ \t]*)from\s+([A-Za-z_][\w.]*)\s*,\s*import\s+",
        r"\1from \2 import ",
        out,
    )
    out = re.sub(
        r"\bfrom\s+([A-Za-z_][\w.]*)\s*,\s*import\s+",
        r"from \1 import ",
        out,
    )
    out = re.sub(r",\s+import\s+", " import ", out)
    out = re.sub(r"(?m)^([ \t]*),\s*from,\s+", r"\1from ", out)
    # ``raise, E`` / ``if, not`` / ``if  not, x`` leftovers from ERROR-region
    # no_comma (remask left plain Python; expression_statement ate headers).
    out = re.sub(r"(?m)^([ \t]*)raise,\s+", r"\1raise ", out)
    out = re.sub(r"(?m)^([ \t]*)del,\s+", r"\1del ", out)
    out = re.sub(r"(?m)^([ \t]*)if,\s+", r"\1if ", out)
    out = re.sub(r"(?m)^([ \t]*)while,\s+", r"\1while ", out)
    out = re.sub(r"(?m)^([ \t]*)with,\s+", r"\1with ", out)
    out = re.sub(r"(?m)^([ \t]*)for,\s+", r"\1for ", out)
    out = re.sub(r"\bnot,\s+", "not ", out)
    # ``for x in in y`` when normalization left literal ``in`` and decode also inserted it.
    out = re.sub(r"\bin\s+in\b", "in", out)
    out = re.sub(r",,", ",", out)
    # Leading ``, "string"`` / ``, f"…"`` after comment-split.
    out = re.sub(r"(?m)^[ \t]*,\s*(?=[fF]?[\"'])", "", out)
    # ``def name args`` without ``(`` — ERROR region dropped parameters node.
    # Handles typed params (``a: T b: U -> R``) and plain spaceSep1 args.
    def _fix_bare_def(m: re.Match) -> str:
        indent, name, rest = m.group(1), m.group(2), (m.group(3) or "").strip()
        if not rest:
            return f"{indent}def {name}():"
        ret = ""
        arrow = re.search(r"(?:->|<arrow>)\s*(.+)$", rest)
        if arrow:
            ret = " -> " + arrow.group(1).strip()
            rest = rest[: arrow.start()].rstrip()
        parts: list[str] = []
        buf = ""
        depth = 0
        for ch in rest:
            if ch in "([{":
                depth += 1
                buf += ch
            elif ch in ")]}":
                depth = max(0, depth - 1)
                buf += ch
            elif ch in " \t" and depth == 0:
                if buf:
                    parts.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            parts.append(buf)
        return f"{indent}def {name}({', '.join(parts)}){ret}:"

    out = re.sub(
        r"(?m)^([ \t]*)def[ \t]+([A-Za-z_]\w*)(?:[ \t]+(?!\()([^\n]+?))?\s*$",
        _fix_bare_def,
        out,
    )
    return out

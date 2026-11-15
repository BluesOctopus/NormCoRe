"""Mutually-exclusive GPT-4 token mass by codec category.

Categories (each source character → one bucket; each GPT-4 token → majority
bucket over its char span):

  L_str / L_cmt / L_num   literal walls
  K_kw / K_op / D_delim   syntax glue (Pass A surface)
  S_sig / S_import / S_call / S_assign / S_ctrl   structural payload
  I_name                  leftover identifiers
  X_other                 whitespace / uncategorized
"""

from __future__ import annotations

import io
import keyword
import tokenize
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper

BUCKETS = (
    "L_str",
    "L_cmt",
    "L_num",
    "K_kw",
    "K_op",
    "D_delim",
    "S_sig",
    "S_import",
    "S_call",
    "S_assign",
    "S_ctrl",
    "I_name",
    "X_other",
)

# tokenize OP strings treated as delimiters (Pass A comma/paren surface)
_DELIMS = frozenset({"(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "...", "@"})


@dataclass(frozen=True)
class MassResult:
    n_files: int
    n_parse_ok: int
    n_parse_fail: int
    total_tokens: int
    by_bucket: dict[str, int]
    pct: dict[str, float]
    # lexical vs structural rollups
    rollup: dict[str, float]


def _byte_to_char(source: str) -> list[int]:
    enc_len = len(source.encode("utf-8"))
    mapping = [0] * (enc_len + 1)
    char_i = byte_i = 0
    for ch in source:
        n = len(ch.encode("utf-8"))
        for _ in range(n):
            mapping[byte_i] = char_i
            byte_i += 1
        char_i += 1
    mapping[byte_i] = char_i
    return mapping


def _paint(arr: list[str | None], start: int, end: int, bucket: str) -> None:
    n = len(arr)
    s = 0 if start < 0 else (n if start > n else start)
    e = 0 if end < 0 else (n if end > n else end)
    for i in range(s, e):
        if arr[i] is None:
            arr[i] = bucket


def _paint_overwrite(arr: list[str | None], start: int, end: int, bucket: str) -> None:
    """Force-paint (used for literals that must win)."""
    n = len(arr)
    s = 0 if start < 0 else (n if start > n else start)
    e = 0 if end < 0 else (n if end > n else end)
    for i in range(s, e):
        arr[i] = bucket


def _op_bucket(op: str) -> str:
    return "D_delim" if op in _DELIMS else "K_op"


def paint_categories(source: str) -> list[str] | None:
    """Return per-character bucket labels, or None if unparsable as module."""
    src = source.replace("\r\n", "\n").replace("\r", "\n")
    if not src.strip():
        return None
    try:
        module = cst.parse_module(src)
    except Exception:
        return None

    n = len(src)
    arr: list[str | None] = [None] * n

    # --- Lexical layer (literals + keywords + ops) via tokenize ---
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return None

    _fstring_types = {
        getattr(tokenize, name)
        for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
        if hasattr(tokenize, name)
    }
    _soft = set(getattr(keyword, "softkwlist", ()))

    for tok in toks:
        ttype, string, (srow, scol), (erow, ecol), _ = tok
        if ttype in (
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            continue
        start = _pos_to_offset(src, srow, scol)
        end = _pos_to_offset(src, erow, ecol)
        if end <= start:
            continue

        if ttype == tokenize.COMMENT:
            _paint_overwrite(arr, start, end, "L_cmt")
        elif ttype == tokenize.STRING or ttype in _fstring_types:
            _paint_overwrite(arr, start, end, "L_str")
        elif ttype == tokenize.NUMBER:
            _paint_overwrite(arr, start, end, "L_num")
        elif ttype == tokenize.NAME and (keyword.iskeyword(string) or string in _soft):
            _paint_overwrite(arr, start, end, "K_kw")
        elif ttype == tokenize.OP:
            _paint_overwrite(arr, start, end, _op_bucket(string))

    # --- Structural layer on unpainted chars (LibCST spans) ---
    # Paint on leave so nested Call/Assign win over outer If/For spans.
    b2c = _byte_to_char(src)
    wrapper = MetadataWrapper(module)

    class Painter(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

        def _char_span(self, node: cst.CSTNode) -> tuple[int, int] | None:
            try:
                cs = self.get_metadata(ByteSpanPositionProvider, node)
            except Exception:
                return None
            nbyte = len(b2c) - 1
            b0 = max(0, min(nbyte, cs.start))
            b1 = max(0, min(nbyte, cs.start + cs.length))
            return b2c[b0], b2c[b1]

        def _paint_node(self, node: cst.CSTNode, bucket: str) -> None:
            sp = self._char_span(node)
            if sp:
                _paint(arr, sp[0], sp[1], bucket)

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            sp = self._char_span(node)
            body_sp = self._char_span(node.body)
            if sp and body_sp:
                _paint(arr, sp[0], body_sp[0], "S_sig")
            elif sp:
                _paint(arr, sp[0], sp[1], "S_sig")
            return True

        def visit_ClassDef(self, node: cst.ClassDef) -> bool:
            sp = self._char_span(node)
            body_sp = self._char_span(node.body)
            if sp and body_sp:
                _paint(arr, sp[0], body_sp[0], "S_sig")
            elif sp:
                _paint(arr, sp[0], sp[1], "S_sig")
            return True

        def visit_Import(self, node: cst.Import) -> bool:
            self._paint_node(node, "S_import")
            return False

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
            self._paint_node(node, "S_import")
            return False

        def leave_Call(self, original_node: cst.Call) -> None:
            self._paint_node(original_node, "S_call")

        def leave_Assign(self, original_node: cst.Assign) -> None:
            self._paint_node(original_node, "S_assign")

        def leave_AnnAssign(self, original_node: cst.AnnAssign) -> None:
            self._paint_node(original_node, "S_assign")

        def leave_AugAssign(self, original_node: cst.AugAssign) -> None:
            self._paint_node(original_node, "S_assign")

        def leave_NamedExpr(self, original_node: cst.NamedExpr) -> None:
            self._paint_node(original_node, "S_assign")

        def _paint_header(self, node: cst.CSTNode, body: cst.CSTNode | None, bucket: str) -> None:
            """Paint statement head only (keyword + test/iter), not the suite."""
            sp = self._char_span(node)
            if not sp:
                return
            if body is not None:
                body_sp = self._char_span(body)
                if body_sp:
                    _paint(arr, sp[0], body_sp[0], bucket)
                    return
            _paint(arr, sp[0], sp[1], bucket)

        def leave_If(self, original_node: cst.If) -> None:
            self._paint_header(original_node, original_node.body, "S_ctrl")

        def leave_For(self, original_node: cst.For) -> None:
            self._paint_header(original_node, original_node.body, "S_ctrl")

        def leave_While(self, original_node: cst.While) -> None:
            self._paint_header(original_node, original_node.body, "S_ctrl")

        def leave_With(self, original_node: cst.With) -> None:
            self._paint_header(original_node, original_node.body, "S_ctrl")

        def leave_Try(self, original_node: cst.Try) -> None:
            # try: header is tiny; paint whole try-line keywords via node start→first body
            self._paint_header(original_node, original_node.body, "S_ctrl")

        def leave_Match(self, original_node: cst.Match) -> None:
            self._paint_header(original_node, original_node.cases[0] if original_node.cases else None, "S_ctrl")

        def leave_Raise(self, original_node: cst.Raise) -> None:
            self._paint_node(original_node, "S_ctrl")

        def leave_Assert(self, original_node: cst.Assert) -> None:
            self._paint_node(original_node, "S_ctrl")

        def leave_Return(self, original_node: cst.Return) -> None:
            self._paint_node(original_node, "S_ctrl")
    wrapper.visit(Painter())

    # Remaining identifiers → I_name; else X_other
    for tok in toks:
        if tok.type != tokenize.NAME:
            continue
        if keyword.iskeyword(tok.string) or tok.string in _soft:
            continue
        start = _pos_to_offset(src, tok.start[0], tok.start[1])
        end = _pos_to_offset(src, tok.end[0], tok.end[1])
        _paint(arr, start, end, "I_name")

    return [b if b is not None else "X_other" for b in arr]


def _pos_to_offset(src: str, row: int, col: int) -> int:
    """1-based row / 0-based col → char offset."""
    if row < 1:
        return 0
    lines = src.splitlines(keepends=True)
    if row > len(lines) + 1:
        return len(src)
    off = sum(len(lines[i]) for i in range(row - 1))
    if row == len(lines) + 1:
        return len(src)
    line = lines[row - 1] if row <= len(lines) else ""
    return off + min(col, len(line))


def _gpt4_token_spans(text: str, enc) -> list[tuple[int, int]]:
    """Char spans for each GPT-4 token (best-effort via incremental decode)."""
    ids = enc.encode(text, allowed_special="all")
    spans: list[tuple[int, int]] = []
    # Walk by decoding prefixes — stable for cl100k on BMP-heavy code
    pos = 0
    for i, tid in enumerate(ids):
        # decode single token; may be a replacement for incomplete bytes
        piece = enc.decode([tid])
        if not piece:
            # zero-width / special: attach to current pos
            spans.append((pos, pos))
            continue
        # Find piece at pos (tiktoken decode is exact for ordinary tokens)
        if text.startswith(piece, pos):
            end = pos + len(piece)
        else:
            # Fallback: search forward a little (rare whitespace normalize)
            j = text.find(piece, pos)
            if j < 0 or j > pos + 8:
                end = min(len(text), pos + max(1, len(piece)))
                spans.append((pos, end))
                pos = end
                continue
            # gap → X_other mass already; skip gap into this token
            pos = j
            end = pos + len(piece)
        spans.append((pos, end))
        pos = end
    return spans


def token_bucket_counts(source: str, enc) -> Counter[str] | None:
    labels = paint_categories(source)
    if labels is None:
        return None
    spans = _gpt4_token_spans(source.replace("\r\n", "\n").replace("\r", "\n"), enc)
    # paint used same newline normalize
    src = source.replace("\r\n", "\n").replace("\r", "\n")
    if len(labels) != len(src):
        return None
    c: Counter[str] = Counter()
    for s, e in spans:
        if e <= s:
            c["X_other"] += 1
            continue
        s = max(0, min(len(labels), s))
        e = max(0, min(len(labels), e))
        sub = labels[s:e]
        if not sub:
            c["X_other"] += 1
            continue
        # majority vote; tie → first char
        cnt = Counter(sub)
        bucket, _ = cnt.most_common(1)[0]
        c[bucket] += 1
    return c


def _rollup(pct: dict[str, float]) -> dict[str, float]:
    lit = pct.get("L_str", 0) + pct.get("L_cmt", 0) + pct.get("L_num", 0)
    glue = pct.get("K_kw", 0) + pct.get("K_op", 0) + pct.get("D_delim", 0)
    struct = (
        pct.get("S_sig", 0)
        + pct.get("S_import", 0)
        + pct.get("S_call", 0)
        + pct.get("S_assign", 0)
        + pct.get("S_ctrl", 0)
    )
    return {
        "literal_wall": round(lit, 2),
        "syntax_glue": round(glue, 2),
        "structural_payload": round(struct, 2),
        "identifiers": round(pct.get("I_name", 0), 2),
        "other_ws": round(pct.get("X_other", 0), 2),
    }


def profile_sources(sources: Iterable[str], *, enc=None) -> MassResult:
    from normcore_codec.tokens import load_gpt4_tokenizer

    enc = enc or load_gpt4_tokenizer()
    total: Counter[str] = Counter()
    n_ok = n_fail = n_files = 0
    for src in sources:
        n_files += 1
        if not src or not src.strip():
            n_fail += 1
            continue
        counts = token_bucket_counts(src, enc)
        if counts is None:
            n_fail += 1
            continue
        n_ok += 1
        total.update(counts)

    grand = sum(total.values())
    by_bucket = {b: int(total.get(b, 0)) for b in BUCKETS}
    # include any unexpected keys
    for k, v in total.items():
        if k not in by_bucket:
            by_bucket[k] = int(v)
    pct = {
        b: round(100.0 * by_bucket.get(b, 0) / grand, 2) if grand else 0.0 for b in BUCKETS
    }
    return MassResult(
        n_files=n_files,
        n_parse_ok=n_ok,
        n_parse_fail=n_fail,
        total_tokens=grand,
        by_bucket=by_bucket,
        pct=pct,
        rollup=_rollup(pct),
    )

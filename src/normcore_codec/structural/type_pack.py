"""S1 aggressive static: pack multi-token type expressions into reversible specials.

Compress: after Pass A, replace ``List[int]``-style spans with ``<t_…>``.
Decompress: expand specials back **before** spy decode (no grammar change).

Only transferable typing idioms (List/Dict/Optional/…) — mined from StarCoder,
not HE. Single-token types (``int``/``str``) are skipped (no GPT-4 gain).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TYPE_TABLE = ROOT / "results" / "type_specials.json"

# Transferable typing surface (SlimCode-style category prior)
_TYPING_HEAD = frozenset(
    {
        "List",
        "Dict",
        "Set",
        "Tuple",
        "Optional",
        "Union",
        "Callable",
        "Iterable",
        "Iterator",
        "Sequence",
        "Mapping",
        "FrozenSet",
        "DefaultDict",
        "Type",
        "Any",
        "list",
        "dict",
        "set",
        "tuple",
    }
)


def _type_id(expr: str) -> str:
    h = hashlib.sha1(expr.encode("utf-8")).hexdigest()[:8]
    return f"<t_{h}>"


def is_transferable_type(expr: str) -> bool:
    s = expr.strip().replace(" ", "")
    if "[" not in s and "|" not in s:
        return False  # skip bare int/str/bool
    head = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", s)
    if not head:
        return False
    return head.group(1) in _TYPING_HEAD


def collect_annotations(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []

    def add(node: ast.AST | None) -> None:
        if node is None:
            return
        try:
            out.append(ast.unparse(node))
        except Exception:
            return

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in list(n.args.args) + list(n.args.posonlyargs) + list(n.args.kwonlyargs):
                add(a.annotation)
            if n.args.vararg:
                add(n.args.vararg.annotation)
            if n.args.kwarg:
                add(n.args.kwarg.annotation)
            add(n.returns)
        elif isinstance(n, ast.AnnAssign):
            add(n.annotation)
    return out


def mine_type_specials(
    sources: list[str],
    *,
    count_tokens_fn,
    min_freq: int = 2,
    min_tok: int = 2,
    top_k: int = 40,
) -> list[dict]:
    freq: Counter[str] = Counter()
    for src in sources:
        for expr in collect_annotations(src):
            if is_transferable_type(expr):
                freq[expr.replace(" ", "")] += 1
                # also keep spaced form key normalized
    rows: list[dict] = []
    for expr, f in freq.items():
        if f < min_freq:
            continue
        ntok = int(count_tokens_fn(expr))
        if ntok < min_tok:
            continue
        rows.append(
            {
                "id": _type_id(expr),
                "type": expr,
                "freq": f,
                "n_tok": ntok,
                "saved_est": f * (ntok - 1),
            }
        )
    rows.sort(key=lambda r: (-r["saved_est"], -r["freq"], -r["n_tok"]))
    return rows[:top_k]


# Seed transferable idioms common across Python corpora (incl. HE-style)
_SEED_TYPES = (
    "List[int]",
    "List[str]",
    "List[float]",
    "List[Any]",
    "Dict[str,Any]",
    "Dict[str,str]",
    "Dict[str,int]",
    "Optional[str]",
    "Optional[int]",
    "Optional[bool]",
    "Tuple[int,int]",
    "Tuple[str,str]",
    "Tuple[float,float]",
    "Set[str]",
    "Set[int]",
    "Union[str,int]",
    "Callable[...,Any]",
    "Iterable[str]",
    "Sequence[int]",
)


def load_type_table(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else DEFAULT_TYPE_TABLE
    rows: list[dict] = []
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = list(data.get("types", data if isinstance(data, list) else []))
    have = {r["type"].replace(" ", "") for r in rows}
    for t in _SEED_TYPES:
        key = t.replace(" ", "")
        if key in have:
            continue
        rows.append({"id": _type_id(key), "type": key, "freq": 0, "n_tok": 0, "saved_est": 0, "seed": True})
        have.add(key)
    # longest first for replace
    rows.sort(key=lambda r: -len(r["type"]))
    return rows


def save_type_table(rows: list[dict], path: str | Path | None = None) -> Path:
    p = Path(path) if path else DEFAULT_TYPE_TABLE
    p.parent.mkdir(parents=True, exist_ok=True)
    # merge seeds
    have = {r["type"].replace(" ", "") for r in rows}
    merged = list(rows)
    for t in _SEED_TYPES:
        key = t.replace(" ", "")
        if key not in have:
            merged.append({"id": _type_id(key), "type": key, "freq": 0, "seed": True})
    p.write_text(json.dumps({"types": merged, "scheme": "S1"}, indent=2), encoding="utf-8")
    return p


def pack_types(text: str, table: list[dict] | None = None) -> tuple[str, dict]:
    """Replace type expressions with specials (longest first)."""
    rows = table if table is not None else load_type_table()
    out = text
    n = 0
    ids: list[str] = []
    for row in sorted(rows, key=lambda r: -len(r["type"])):
        typ = str(row["type"])
        tok = str(row["id"])
        if not typ or typ not in out:
            continue
        c = out.count(typ)
        if c:
            out = out.replace(typ, tok)
            n += c
            ids.extend([tok] * c)
    return out, {"types_packed": n, "ids": ids}


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _abs_off(line_starts: list[int], lineno: int, col: int, n: int) -> int:
    if lineno < 1:
        return 0
    if lineno >= len(line_starts):
        return n
    return min(n, line_starts[lineno - 1] + col)


def collect_type_quotas(
    source: str,
    scores: dict | None,
    *,
    table: list[dict] | None = None,
    apply_on: str = "L",
) -> dict[str, int]:
    """Count packable type occurrences inside FunctionDefs with band ∈ apply_on.

    Must run on the **same** source used for ``score_nodes`` (pre–Pass-A).
    Keys are whitespace-stripped type strings matching the type table.
    """
    rows = table if table is not None else load_type_table()
    known = {str(r["type"]).replace(" ", "") for r in rows}
    if not scores or not known:
        return {}

    allow_all = apply_on in ("*", "HML", "LMH", "all")
    fn_scores = [ns for ns in scores.values() if getattr(ns, "kind", None) == "FunctionDef"]
    fn_scores.sort(key=lambda ns: (ns.start, ns.end))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fns.sort(key=lambda n: (n.lineno, n.col_offset))
    starts = _line_starts(source)
    nsrc = len(source)

    # Map each function AST node to a band via overlapping CST FunctionDef span
    def band_for(fn: ast.AST) -> str:
        a0 = _abs_off(starts, fn.lineno, fn.col_offset, nsrc)
        a1 = _abs_off(
            starts,
            getattr(fn, "end_lineno", fn.lineno) or fn.lineno,
            getattr(fn, "end_col_offset", 0) or 0,
            nsrc,
        )
        best = None
        best_ov = -1
        for ns in fn_scores:
            ov = min(a1, ns.end) - max(a0, ns.start)
            if ov > best_ov:
                best_ov = ov
                best = ns
        return best.band if best is not None and best_ov > 0 else "H"

    quota: Counter[str] = Counter()

    def add_ann(node: ast.AST | None, band: str) -> None:
        if node is None:
            return
        if not allow_all and band not in apply_on:
            return
        try:
            expr = ast.unparse(node).replace(" ", "")
        except Exception:
            return
        if expr in known and is_transferable_type(expr):
            quota[expr] += 1

    for fn in fns:
        band = band_for(fn)
        for a in list(fn.args.args) + list(fn.args.posonlyargs) + list(fn.args.kwonlyargs):
            add_ann(a.annotation, band)
        if fn.args.vararg:
            add_ann(fn.args.vararg.annotation, band)
        if fn.args.kwarg:
            add_ann(fn.args.kwarg.annotation, band)
        add_ann(fn.returns, band)
        for n in ast.walk(fn):
            if isinstance(n, ast.AnnAssign):
                add_ann(n.annotation, band)

    return dict(quota)


def pack_types_gated(
    text: str,
    quotas: dict[str, int] | None,
    *,
    table: list[dict] | None = None,
) -> tuple[str, dict]:
    """AD-THP: replace types only up to per-type quotas from L-band functions."""
    rows = table if table is not None else load_type_table()
    if not quotas:
        return text, {"types_packed": 0, "ids": [], "gated": True, "quota_total": 0}

    budget = {k: int(v) for k, v in quotas.items() if v > 0}
    out = text
    n = 0
    ids: list[str] = []
    for row in sorted(rows, key=lambda r: -len(r["type"])):
        key = str(row["type"]).replace(" ", "")
        left = budget.get(key, 0)
        if left <= 0:
            continue
        tok = str(row["id"])
        for typ in (key, str(row["type"])):
            if typ not in out:
                continue
            while left > 0 and typ in out:
                out = out.replace(typ, tok, 1)
                left -= 1
                n += 1
                ids.append(tok)
            break
        budget[key] = left

    return out, {
        "types_packed": n,
        "ids": ids,
        "gated": True,
        "quota_total": sum(int(v) for v in quotas.values()),
    }


def unpack_types(text: str, table: list[dict] | None = None) -> str:
    rows = table if table is not None else load_type_table()
    out = text
    # longest special id first (ids are fixed length-ish; still sort)
    for row in sorted(rows, key=lambda r: -len(r["id"])):
        out = out.replace(str(row["id"]), str(row["type"]).replace(" ", ""))
    return out


def type_special_tokens(table: list[dict] | None = None) -> list[str]:
    rows = table if table is not None else load_type_table()
    return [str(r["id"]) for r in rows]

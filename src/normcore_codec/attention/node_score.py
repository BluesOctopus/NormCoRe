"""Per-CST-node attention scores for S1 (instance-level throttle).

Pipeline
--------
1. CodeBERT token attentions + char offsets
2. Map onto LibCST node spans → mean score per node
3. Within-file percentile bands: H / M / L

Keys are ``(kind, start, end)`` so scores survive a second parse in normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import libcst as cst
from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper, PositionProvider

# LibCST folds async into FunctionDef/For/With via ``asynchronous`` field.
SCORE_TYPES = (
    cst.FunctionDef,
    cst.ClassDef,
    cst.For,
    cst.While,
    cst.If,
    cst.With,
    cst.Call,
    cst.Parameters,
    cst.Import,
    cst.ImportFrom,
    cst.Return,
    cst.Assign,
    cst.AnnAssign,
    cst.AugAssign,
    cst.List,
    cst.Tuple,
    cst.Dict,
    cst.Set,
)

NodeKey = tuple[str, int, int]


@dataclass(frozen=True)
class NodeScore:
    key: NodeKey
    kind: str
    start: int
    end: int
    score: float
    band: str  # H | M | L


def _kind(node: cst.CSTNode) -> str:
    return type(node).__name__


def node_key(node: cst.CSTNode, start: int, end: int) -> NodeKey:
    return (_kind(node), start, end)


def collect_node_spans(source: str) -> list[tuple[cst.CSTNode, int, int, str]]:
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    positions: list[tuple[cst.CSTNode, int, int, str]] = []

    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ByteSpanPositionProvider, PositionProvider)

        def on_visit(self, node: cst.CSTNode) -> bool:
            if isinstance(node, SCORE_TYPES):
                try:
                    cs = self.get_metadata(ByteSpanPositionProvider, node)
                    positions.append((node, cs.start, cs.start + cs.length, _kind(node)))
                except Exception:
                    pass
            return True

    wrapper.visit(V())
    return positions


def score_nodes(
    source: str,
    *,
    model=None,
    tokenizer=None,
    device: str | None = None,
    max_length: int = 512,
    low_pct: float = 0.40,
    high_pct: float = 0.70,
) -> tuple[dict[NodeKey, NodeScore], dict[str, Any]]:
    """Score CST nodes; return map (kind,start,end)→NodeScore plus meta."""
    try:
        from normcore_codec.attention.profile import extract_token_attentions, load_codebert
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from normcore_codec.attention.profile import extract_token_attentions, load_codebert

    src = source.replace("\r\n", "\n").replace("\r", "\n")
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_codebert()

    try:
        token_rows, tok_meta = extract_token_attentions(
            src, model=model, tokenizer=tokenizer, device=device, max_length=max_length
        )
    except Exception as e:
        return {}, {"error": f"{type(e).__name__}: {e}"}

    spans = collect_node_spans(src)
    if not spans:
        return {}, {"n_nodes": 0, **tok_meta}

    raw: list[tuple[NodeKey, str, int, int, float]] = []
    for node, start, end, kind in spans:
        vals = [sc for ts, te, sc, _ in token_rows if te > start and ts < end]
        if not vals:
            mid = (start + end) / 2
            if token_rows:
                best = min(token_rows, key=lambda r: abs((r[0] + r[1]) / 2 - mid))
                vals = [best[2]]
            else:
                vals = [0.0]
        key = node_key(node, start, end)
        raw.append((key, kind, start, end, sum(vals) / len(vals)))

    scores_only = sorted(r[4] for r in raw)
    n = len(scores_only)

    def pct_val(p: float) -> float:
        if n == 1:
            return scores_only[0]
        idx = min(n - 1, max(0, int(p * (n - 1))))
        return scores_only[idx]

    lo = pct_val(low_pct)
    hi = pct_val(high_pct)

    def band(s: float) -> str:
        if s <= lo:
            return "L"
        if s >= hi:
            return "H"
        return "M"

    out: dict[NodeKey, NodeScore] = {}
    for key, kind, start, end, sc in raw:
        out[key] = NodeScore(key, kind, start, end, sc, band(sc))

    meta = {
        "n_nodes": n,
        "p_low": lo,
        "p_high": hi,
        "n_L": sum(1 for s in out.values() if s.band == "L"),
        "n_M": sum(1 for s in out.values() if s.band == "M"),
        "n_H": sum(1 for s in out.values() if s.band == "H"),
        **tok_meta,
    }
    return out, meta

def get_hotzones(
    source: str,
    *,
    model=None,
    tokenizer=None,
    device: str | None = None,
    max_length: int = 512,
    top_pct: float = 0.20,
) -> set[int]:
    """EXPERIMENTAL: Conditional Attention 2.0 hotzone detection."""
    import torch
    inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True).to(device)
    with torch.no_grad():
        outs = model(**inputs, output_attentions=True)
    n_tokens = inputs.input_ids.size(1)
    recv = torch.zeros(n_tokens, device=device)
    for layer_attn in outs.attentions:
        head_avg = layer_attn[0].mean(dim=0)
        recv += head_avg.sum(dim=0)
    recv = (recv / len(outs.attentions)).float().cpu().tolist()
    offsets = inputs.offset_mapping[0].cpu().tolist()
    indexed_vals = sorted([v for i, v in enumerate(recv) if offsets[i][0] != offsets[i][1]], reverse=True)
    if not indexed_vals: return set()
    thresh = indexed_vals[int(len(indexed_vals) * top_pct)] if len(indexed_vals) > 0 else 0
    hotzones = set()
    for i, v in enumerate(recv):
        if v >= thresh:
            s, e = offsets[i]
            for char_idx in range(s, e):
                hotzones.add(char_idx)
    return hotzones

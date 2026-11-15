import torch
from transformers import AutoModel, AutoTokenizer
from typing import Any, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class StatementSpan:
    start: int
    end: int
    kind: str
    score: float = 0.0

def load_codebert(model_name: str) -> Tuple[Any, Any, str]:
    """Load an explicitly selected calibration model; no machine-local default."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_safetensors=True
    ).to(device)
    model.eval()
    return model, tokenizer, device

def extract_token_attentions(
    source: str,
    model: Any,
    tokenizer: Any,
    device: str,
    max_length: int = 1024
) -> Tuple[List[Tuple[int, int, float, str]], dict]:
    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    ).to(device)

    offsets = inputs.pop("offset_mapping")[0].cpu().numpy()

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = torch.stack(outputs.attentions)
    avg_attn = attentions.mean(dim=(0, 2))[0] # (seq, seq)
    token_scores = avg_attn.mean(dim=0).cpu().to(torch.float32).numpy()

    token_data = []
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    for i, (start, end) in enumerate(offsets):
        if start == end == 0: continue
        token_data.append((int(start), int(end), float(token_scores[i]), tokens[i]))

    meta = {"n_tokens": len(token_data)}
    return token_data, meta

from .taxonomy import classify_statement

def spans_from_source(source: str) -> List[StatementSpan]:
    from recode_codec.attention.node_score import collect_node_spans
    spans = []
    for node, start, end, kind in collect_node_spans(source):
        spans.append(StatementSpan(start=start, end=end, kind=kind))
    return spans

def aggregate_profile(token_rows: List[Tuple[int, int, float, str]], spans: List[StatementSpan]) -> List[StatementSpan]:
    for span in spans:
        vals = [sc for ts, te, sc, _ in token_rows if te > span.start and ts < span.end]
        if vals:
            span.score = float(np.mean(vals))
    return spans

def profile_sources(sources: List[str], model=None, tokenizer=None, device=None) -> dict:
    if model is None:
        model, tokenizer, device = load_codebert()

    category_totals = {}
    category_counts = {}

    for src in sources:
        try:
            token_rows, _ = extract_token_attentions(src, model, tokenizer, device)
            spans = spans_from_source(src)
            print(f"DEBUG: Found {len(spans)} spans in sample")
            spans = aggregate_profile(token_rows, spans)

            for s in spans:
                cat = classify_statement(s.kind)
                category_totals[cat] = category_totals.get(cat, 0.0) + s.score
                category_counts[cat] = category_counts.get(cat, 0) + 1
        except Exception as e:
            print(f"DEBUG: Error profiling sample: {e}")
            continue

    print(f"DEBUG: category_counts: {category_counts}")
    return {k: category_totals[k] / category_counts[k] for k in category_totals if category_counts[k] > 0}

def profile_categories(samples: List[str], model=None, tokenizer=None, device=None) -> dict:
    return profile_sources(samples, model, tokenizer, device)

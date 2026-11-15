import torch
import numpy as np
from typing import Set, Dict, Any, Tuple
from pathlib import Path

def get_attention_hotzones(
    source: str,
    model,
    tokenizer,
    device: str = "cuda",
    top_pct: float = 0.20,
    max_length: int = 512,
) -> Set[int]:
    """Extract hotzones using sliding window to handle long files and reduce sink effect."""
    # Split into 512-token chunks with 256-token overlap
    full_inputs = tokenizer(source, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = full_inputs["input_ids"]
    offsets = full_inputs["offset_mapping"]

    seq_len = len(input_ids)
    if seq_len == 0: return set()

    # Received attention buffer
    total_recv = np.zeros(seq_len)
    counts = np.zeros(seq_len)

    stride = 256
    for start_idx in range(0, seq_len, stride):
        end_idx = min(start_idx + max_length, seq_len)
        chunk_ids = input_ids[start_idx:end_idx]

        # Prepare batch
        batch = torch.tensor([chunk_ids]).to(device)
        with torch.no_grad():
            outputs = model(batch, output_attentions=True)

        # Take last 4 layers
        relevant_layers = outputs.attentions[-4:]
        chunk_recv = torch.zeros(len(chunk_ids), device=device)
        for layer_attn in relevant_layers:
            # Average heads, then sum received attention (columns)
            # Avoid the very first token in each chunk if it's not the file start (sink reduction)
            head_avg = layer_attn[0].mean(dim=0)
            chunk_recv += head_avg.sum(dim=0)

        chunk_recv = chunk_recv.float().cpu().numpy()

        # Weighting: reduce weight for the first 16 tokens of each chunk to suppress sink
        if start_idx > 0:
            chunk_recv[:16] *= 0.1

        total_recv[start_idx:end_idx] += chunk_recv
        counts[start_idx:end_idx] += 1

        if end_idx == seq_len: break

    # Normalized score
    final_scores = total_recv / np.maximum(counts, 1)

    # Thresholding
    threshold = np.percentile(final_scores, 100 - (top_pct * 100))

    hotzones = set()
    for i, score in enumerate(final_scores):
        if score >= threshold:
            s, e = offsets[i]
            for char_idx in range(s, e):
                hotzones.add(char_idx)

    # Structural Propagation
    enhanced = set(hotzones)
    for i, char in enumerate(source):
        if char == '\n':
            if (i > 0 and (i - 1) in hotzones) or (i < len(source) - 1 and (i + 1) in hotzones):
                enhanced.add(i)

    return enhanced

class ConditionalScaffold:
    def __init__(self, source_hotzones: Set[int], source_len: int):
        self.hotzones = source_hotzones
        self.source_len = source_len

    def should_veto_source(self, start: int, end: int) -> bool:
        """Query if a span in ORIGINAL source hits a hotzone."""
        # Use range check for efficiency
        for i in range(max(0, start), min(self.source_len, end)):
            if i in self.hotzones:
                return True
        return False

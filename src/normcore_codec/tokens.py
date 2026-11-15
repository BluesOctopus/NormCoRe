from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Callable


@lru_cache(maxsize=1)
def get_tokenizer(path: str | Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)

@lru_cache(maxsize=1)
def get_tiktoken_encoding(name: str = "cl100k_base"):
    import tiktoken
    return tiktoken.get_encoding(name)


@lru_cache(maxsize=16)
def _specials_re(specials: frozenset[str]) -> re.Pattern[str] | None:
    if not specials:
        return None
    # First alternative wins; longest first so ``<if_stmt>`` beats ``<if>``.
    ordered = sorted(specials, key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in ordered))


def count_with_specials(
    text: str, encode_plain: Callable[[str], int], specials: set[str] | frozenset[str]
) -> int:
    """Each known marker is one token. Gaps are encoded on their own.

    Do not splice markers into spaces (the space is billed again) or delete
    them (adjacent identifiers glue). Unknown ``<...>`` spans stay in the gap.
    """
    if not text:
        return 0
    cre = _specials_re(frozenset(specials))
    if cre is None:
        return encode_plain(text)
    total = 0
    pos = 0
    for match in cre.finditer(text):
        if match.start() > pos:
            total += encode_plain(text[pos : match.start()])
        total += 1
        pos = match.end()
    if pos < len(text):
        total += encode_plain(text[pos:])
    return total


def get_count_fn(tokenizer_path: str | Path | None = None, specials: set[str] | None = None) -> Callable[[str], int]:
    """Universal counter. If path is None, falls back to cl100k_base."""
    known = set(specials or ())

    if tokenizer_path:
        tokenizer = get_tokenizer(tokenizer_path)

        def count(text: str) -> int:
            return count_with_specials(
                text, lambda t: len(tokenizer.encode(t, add_special_tokens=False)), known
            )
    else:
        encoding = get_tiktoken_encoding("cl100k_base")

        def count(text: str) -> int:
            return count_with_specials(text, lambda t: len(encoding.encode(t)), known)

    return count

# Legacy API Aliases
def load_gpt4_tokenizer(): return get_tiktoken_encoding("cl100k_base")
def count_tokens(text: str) -> int: return len(get_tiktoken_encoding("cl100k_base").encode(text))

"""NormCoRe: normalized and conditional token-reduction codec."""

from .config import CodecAssets, CodecConfig, NormalizationConfig, ShorthandingConfig
from .codec import (
    CompressResult,
    PlaceholderLeak,
    compress,
    decompress,
    restore,
    verify_ast_roundtrip,
    verify_exec_roundtrip,
)

__all__ = [
    "CodecAssets",
    "CodecConfig",
    "CompressResult",
    "NormalizationConfig",
    "ShorthandingConfig",
    "PlaceholderLeak",
    "compress",
    "decompress",
    "restore",
    "verify_ast_roundtrip",
    "verify_exec_roundtrip",
]

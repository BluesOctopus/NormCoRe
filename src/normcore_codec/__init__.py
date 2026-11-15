"""NormCoRe: normalized and conditional token-reduction codec."""

from .config import CodecAssets, CodecConfig, PassAConfig, PassBConfig
from .pipeline import (
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
    "PassAConfig",
    "PassBConfig",
    "PlaceholderLeak",
    "compress",
    "decompress",
    "restore",
    "verify_ast_roundtrip",
    "verify_exec_roundtrip",
]

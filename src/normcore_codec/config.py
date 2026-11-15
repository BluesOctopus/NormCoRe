"""Typed configuration for the canonical NormCoRe codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CodecAssets:
    """Explicit, portable assets consumed by the codec.

    The codec never falls back to an implicit working-directory path.
    """

    pattern_registry: Path | None = None
    conditional_policy: Path | None = None
    type_table: Path | None = None
    grammar_binary: Path | None = None
    asset_lock: Path | None = None
    tokenizer_path: str | Path | None = None

    @classmethod
    def from_root(cls, root: str | Path) -> "CodecAssets":
        base = Path(root)
        return cls(
            pattern_registry=base / "registry_canonical.json",
            conditional_policy=base / "policy_canonical.json",
            type_table=base / "type_specials.json",
            grammar_binary=base / "spy_lang.so",
            asset_lock=base / "ASSET_LOCK.json",
        )


@dataclass(frozen=True)
class NormalizationConfig:
    """Structural condensation controls."""

    mode: str = "aggressive"

    def __post_init__(self) -> None:
        if self.mode not in {"conservative", "aggressive"}:
            raise ValueError(
                "canonical API allows only conservative/aggressive normalization"
            )


@dataclass(frozen=True)
class ShorthandingConfig:
    """Semantic shorthanding controls."""

    enabled: bool = True


@dataclass(frozen=True)
class CodecConfig:
    """One immutable configuration for an encode/decode run."""

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    shorthanding: ShorthandingConfig = field(default_factory=ShorthandingConfig)
    verify_assets: bool = True

    @classmethod
    def canonical(cls) -> "CodecConfig":
        return cls()

    @classmethod
    def normalization_only(cls, mode: str = "conservative") -> "CodecConfig":
        return cls(
            normalization=NormalizationConfig(mode=mode),
            shorthanding=ShorthandingConfig(enabled=False),
        )

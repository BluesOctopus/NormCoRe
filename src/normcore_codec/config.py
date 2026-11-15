"""Typed configuration for the canonical NormCoRe codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CodecAssets:
    """Explicit, portable assets consumed by the codec.

    Pass B and GCA never fall back to an implicit working-directory path.
    """

    pattern_registry: Path | None = None
    gca_policy: Path | None = None
    type_table: Path | None = None
    grammar_binary: Path | None = None
    asset_lock: Path | None = None
    tokenizer_path: str | Path | None = None

    @classmethod
    def from_root(cls, root: str | Path) -> "CodecAssets":
        base = Path(root)
        return cls(
            pattern_registry=base / "registry_canonical.json",
            gca_policy=base / "policy_canonical.json",
            type_table=base / "type_specials.json",
            grammar_binary=base / "spy_lang.so",
            asset_lock=base / "ASSET_LOCK.json",
        )


@dataclass(frozen=True)
class PassAConfig:
    """Structural condensation controls."""

    mode: str = "gca"

    def __post_init__(self) -> None:
        if self.mode not in {"gca", "gca1", "conservative", "aggressive"}:
            raise ValueError(
                "canonical API allows only gca/gca1/conservative/aggressive; "
                "instance-gated modes are historical"
            )


@dataclass(frozen=True)
class PassBConfig:
    """Semantic shorthanding controls."""

    enabled: bool = True


@dataclass(frozen=True)
class CodecConfig:
    """One immutable configuration for an encode/decode run."""

    pass_a: PassAConfig = field(default_factory=PassAConfig)
    pass_b: PassBConfig = field(default_factory=PassBConfig)
    verify_assets: bool = True

    @classmethod
    def canonical(cls) -> "CodecConfig":
        return cls()

    @classmethod
    def structural_only(cls, mode: str = "conservative") -> "CodecConfig":
        return cls(pass_a=PassAConfig(mode=mode), pass_b=PassBConfig(enabled=False))

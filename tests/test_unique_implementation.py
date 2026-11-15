"""Guard that normalization and shorthanding each have one implementation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from normcore_codec.shorthanding import apply_registry_shorthanding, load_registry

METHOD_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = METHOD_ROOT / "experiments"
LIVE_TREES = (
    METHOD_ROOT / "src",
    METHOD_ROOT / "scripts",
    EXPERIMENTS_ROOT / "src",
    EXPERIMENTS_ROOT / "scripts",
)

UNIQUE_DEFS = {
    "apply_normalization": METHOD_ROOT / "src/normcore_codec/normalization/__init__.py",
    "apply_registry_shorthanding": METHOD_ROOT / "src/normcore_codec/shorthanding/__init__.py",
    "restore_registry_shorthanding": METHOD_ROOT / "src/normcore_codec/shorthanding/__init__.py",
    "compress": METHOD_ROOT / "src/normcore_codec/codec.py",
    "decompress": METHOD_ROOT / "src/normcore_codec/codec.py",
    "mine_from_sources": METHOD_ROOT / "src/normcore_codec/shorthanding/mining.py",
}


def _live_python_files() -> list[Path]:
    files: list[Path] = []
    for tree in LIVE_TREES:
        if not tree.is_dir():
            continue
        files.extend(
            path
            for path in tree.rglob("*.py")
            if "__pycache__" not in path.parts and path.name != "conftest.py"
        )
    return files


def test_normalization_and_shorthanding_have_one_definition_each():
    found: dict[str, list[Path]] = {name: [] for name in UNIQUE_DEFS}
    for path in _live_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in found:
                found[node.name].append(path)
    for name, expected in UNIQUE_DEFS.items():
        locations = found[name]
        assert locations == [expected], f"{name} must live only in {expected}, found {locations}"


def test_live_trees_do_not_import_legacy_codec_or_parallel_miners():
    banned_names = {"mine_legacy.py", "evaluate_legacy.py", "ast_mine.py"}
    live_names = {path.name for path in _live_python_files()}
    assert not (banned_names & live_names)
    forbidden = ("from codec.", "from codec import", "import codec.")
    for path in _live_python_files():
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still references legacy codec via {needle!r}"
        assert "AARI_SUCCESS" not in text
        assert "legacy_evidence" not in text or "sys.path.insert" not in text


def test_release_configuration_has_one_paper_facing_stage_schema():
    config = json.loads(
        (METHOD_ROOT / "configs" / "canonical.json").read_text(encoding="utf-8")
    )
    assert set(config["attention"]) >= {"normalization", "shorthanding"}
    assert "pass_a" not in config["attention"]
    assert "pass_b" not in config["attention"]
    assert not any(
        path.name.lower().startswith(("gca", "recode"))
        for path in (METHOD_ROOT / "src").rglob("*")
        if path.is_file()
    )



def test_load_registry_rejects_a_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="shorthanding registry not found"):
        load_registry(tmp_path / "missing.json")


def test_apply_registry_shorthanding_does_not_load_an_implicit_registry():
    encoded, stats = apply_registry_shorthanding("name = 1", [])
    assert encoded == "name = 1"
    assert stats["skipped"] == "no_registry"

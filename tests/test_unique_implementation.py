"""Guard: Pass A and Pass B each have exactly one live implementation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from recode_codec.dynamic import apply_dynamic_pass, load_registry

METHOD_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = METHOD_ROOT.parent / "recode_experiments"
LIVE_TREES = (
    METHOD_ROOT / "src",
    METHOD_ROOT / "scripts",
    EXPERIMENTS_ROOT / "src",
    EXPERIMENTS_ROOT / "scripts",
)

UNIQUE_DEFS = {
    "apply_structural_pass": METHOD_ROOT / "src/recode_codec/structural/__init__.py",
    "apply_dynamic_pass": METHOD_ROOT / "src/recode_codec/dynamic/__init__.py",
    "invert_dynamic_pass": METHOD_ROOT / "src/recode_codec/dynamic/__init__.py",
    "compress": METHOD_ROOT / "src/recode_codec/pipeline.py",
    "decompress": METHOD_ROOT / "src/recode_codec/pipeline.py",
    "mine_from_sources": METHOD_ROOT / "src/recode_codec/dynamic/miner.py",
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


def test_pass_a_and_pass_b_have_exactly_one_definition_each():
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
    banned_names = {"mine_v2.py", "mine_v3.py", "evaluate_legacy_samesize.py", "ast_mine.py"}
    live_names = {path.name for path in _live_python_files()}
    assert not (banned_names & live_names)
    forbidden = ("from codec.", "from codec import", "import codec.")
    for path in _live_python_files():
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path} still references legacy codec via {needle!r}"
        assert "AARI_SUCCESS" not in text
        assert "legacy_evidence" not in text or "sys.path.insert" not in text



def test_load_registry_rejects_a_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Pass B registry not found"):
        load_registry(tmp_path / "missing.json")


def test_apply_dynamic_pass_does_not_load_an_implicit_registry():
    encoded, stats = apply_dynamic_pass("name = 1", [])
    assert encoded == "name = 1"
    assert stats["skipped"] == "no_registry"

# NormCoRe

NormCoRe is a reversible code representation for reducing the token cost of
LLM-based code processing. This release contains the codec used in the paper,
its locked assets and tests, lightweight experiment entry points, and the
paper source.

## Included

- `src/recode_codec/`: reference implementation. The historical import path is
  retained for compatibility with encoded assets and experiment scripts.
- `assets/`: registry, structural policy, type table, grammar extension, and
  provenance lock used by the canonical configuration.
- `scripts/`: asset validation, round-trip evaluation, and manifest helpers.
- `experiments/`: optional training and held-out scoring entry points. Models,
  datasets, and checkpoints are intentionally not redistributed.
- `paper/`: source and compiled PDF for the accompanying paper.

The codec combines structural condensation with hybrid idiom shorthanding.
Conditional Attention (GCA) is represented by the frozen policy and registry;
the codec hot path does not load a model. Decoding restores a Python program
that is AST-equivalent to the input. Comments and other non-semantic layout
details are not promised to be byte-identical.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Compress and restore

```bash
PYTHONPATH=src python -m recode_codec.cli compress example.py \
  --assets assets --stats > example.dialect.py
PYTHONPATH=src python -m recode_codec.cli decompress example.dialect.py \
  --assets assets > example.restored.py
```

The Python API is:

```python
from recode_codec import CodecAssets, CodecConfig, compress, decompress

assets = CodecAssets.from_root("assets")
config = CodecConfig.canonical()
encoded = compress(source, config=config, assets=assets)
restored = decompress(encoded.code, config=config, assets=assets)
```

## Verification

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python scripts/evaluate_roundtrip.py \
  --sources path/to/python/files --assets assets
```

The experiment scripts require user-supplied models and datasets. They are
research entry points, not bundled benchmark data.

## Paper

```bash
cd paper
latexmk -pdf main.tex
```

See `paper/src/ch08-data-availability.tex` for the artifact scope and link.

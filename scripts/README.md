# Codec utilities

These utilities operate on the public codec and explicit asset paths:

- `build_assets.py` validates or rebuilds registry assets;
- `evaluate_roundtrip.py` checks AST-equivalent decoding and token counts;
- `generate_method_manifest.py` records source and asset hashes;
- `rebuild_registry_holes.py` is an optional registry-construction utility.

They do not download data or model weights.

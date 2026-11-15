"""Command-line interface for the canonical NormCoRe codec."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from normcore_codec.config import CodecAssets, CodecConfig
from normcore_codec.codec import compress, decompress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("compress", "decompress"))
    parser.add_argument("input", nargs="?", type=Path, help="UTF-8 input; defaults to stdin")
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets",
        help="directory containing locked policy, registry, grammar, and type table",
    )
    parser.add_argument("--normalization-only", action="store_true")
    parser.add_argument("--stats", action="store_true", help="write compression stats to stderr")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    assets = CodecAssets.from_root(args.assets)
    config = CodecConfig.normalization_only() if args.normalization_only else CodecConfig.canonical()
    if args.operation == "compress":
        result = compress(text, config=config, assets=assets)
        sys.stdout.write(result.code)
        if args.stats:
            sys.stderr.write(json.dumps(result.stats, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(decompress(text, config=config, assets=assets))


if __name__ == "__main__":
    main()

# Release consistency notes

The public release has one codec implementation under `src/normcore_codec/` and
one canonical asset set under `assets/`. `CodecConfig.canonical()` enables the
same structural policy, registry, type table, and grammar extension used by
the paper-facing round-trip checks.

The package import path is retained for compatibility; the distribution and
command line tool are named `normcore-codec`. Marker strings in encoded files
are part of the wire format and are intentionally unchanged.

The codec contract is AST equivalence after decoding. It does not claim to
preserve comments, whitespace, or other non-semantic source formatting.

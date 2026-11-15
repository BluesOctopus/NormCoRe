# Method contract

1. Callers provide an explicit `CodecAssets` root; no server-specific paths are
   consulted.
2. The canonical pipeline applies structural condensation, then semantic
   shorthanding, and restores them in reverse order.
3. Literal regions are masked before rewriting and restored after decoding.
4. The decoder must produce an AST-equivalent Python program for every accepted
   input. A failed verification is not counted as token savings.
5. Conditional Attention policy and registry files are validated against `ASSET_LOCK.json` before
   a canonical run.

Formatting and comments may change; program structure and literal values remain
recoverable under the codec contract.

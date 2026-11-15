# NormCoRe Codebase Audit

This document records the release refactor before any implementation is moved.
The purpose is to make the public artifact follow the paper's method story and
to keep one reproducible codec baseline. It is an engineering inventory, not a
second implementation of the method.

## Canonical method path

The paper's executable path is:

1. mask literals and preserve typed placeholders;
2. apply normalization (the structural condensation pass);
3. apply hybrid idiom shorthanding from the locked registry;
4. optionally use the frozen Conditional Attention policy when selecting
   occurrence-level candidates;
5. restore the source through the reversible grammar and typed unpacking.

The public API should expose one codec pipeline for this path. Historical GCA
island variants are not part of the canonical method.

## File inventory and target ownership

| Current path | Current role | Status | Target |
|---|---|---|---|
| `pipeline.py` | compress/decompress orchestration | canonical, needs cleanup | `codec.py` |
| `config.py` | typed configuration and asset paths | canonical | `config.py` |
| `assets.py` | asset and registry validation | canonical | `assets.py` |
| `tokens.py` | tokenizer counting helpers | canonical utility | `tokens.py` |
| `markers.py` | dialect markers and special tokens | canonical | `reversible/markers.py` |
| `mask.py` | literal masking/unmasking | canonical | `reversible/literals.py` |
| `spy_decode.py` | grammar decode orchestration | canonical | `reversible/grammar.py` |
| `spy_edit.py` | decoder edit primitives | canonical | `reversible/edits.py` |
| `spy_rules.py` | Python recovery rules | canonical | `reversible/grammar_rules.py` |
| `postprocess.py` | decode post-processing | canonical | `reversible/postprocess.py` |
| `edits.py` | generic span edits | canonical utility | `reversible/edits.py` or shared utility |
| `structural/__init__.py` | structural pass facade | canonical | `normalization/engine.py` |
| `structural/rules.py` | normalization rules | canonical | `normalization/rules.py` |
| `structural/rule_catalog.py` | rule metadata/catalogue | canonical utility | `normalization/catalog.py` |
| `structural/type_pack.py` | typed structural packing | canonical | `normalization/type_pack.py` |
| `structural/aggressive.py` | canonical idiom packing/unpacking | canonical | `normalization/idioms.py` |
| `dynamic/__init__.py` | registry facade and dynamic pass | canonical | `shorthanding/engine.py` |
| `dynamic/miner.py` | pattern mining | canonical research utility | `shorthanding/mining.py` |
| `dynamic/subtree.py` | subtree candidate pass | canonical helper | `shorthanding/subtree.py` |
| `dynamic/wildcard.py` | registry wildcard matching | canonical | `shorthanding/wildcard.py` |
| `passb.py` | historical Pass B facade | merge into canonical shorthanding engine | remove after migration |
| `attention/gca.py` | policy loading and occurrence selection | canonical | `attention/conditional.py` |
| `attention/weight.py` | attention/reward weighting | canonical | `attention/scoring.py` |
| `attention/node_score.py` | node-level score extraction | canonical analysis helper | `attention/node_score.py` |
| `attention/profile.py` | model attention profiling | experiment utility | `attention/profile.py` |
| `attention/band_paint.py` | occurrence band annotations | canonical helper | `attention/occurrence.py` |
| `attention/scaffold.py` | scaffold analysis | experiment utility | `attention/scaffold.py` |
| `attention/taxonomy.py` | feature taxonomy | experiment utility | `attention/taxonomy.py` |
| `corpus.py` | corpus loading helpers | experiment utility | `corpus.py` or `experiments/` |
| `mass_profile.py` | large-scale profiling | experiment utility | `experiments/` |
| `gca1.py` | function-island historical arm | legacy | remove from public package |
| `gca_stmt.py` | statement-island historical arm | legacy | remove from public package |
| `gca_v4.py` | v4 island/high-contrast arm | legacy | remove from public package |
| `gca_v5.py` | v5 island/high-contrast arm | legacy | remove from public package |

## Naming rules for the refactor

- The import package is `normcore_codec`.
- Public symbols and file names use `normcore` terminology; no new code may
  introduce `RECODE`, `Rcdc`, `gca_v4`, or `gca_v5` names.
- `Conditional Attention` is the paper-facing name. `gca` may remain only in
  compatibility parsing of old experiment artifacts, never as the primary
  public module name.
- The stable codec contract is versioned as
  `NormCoRe-normalization-then-shorthanding/v1`.
- Historical marker spellings are treated as serialized-format compatibility
  data. They are migrated only with a coordinated encoder, decoder, asset, and
  test change; blind text replacement is prohibited.

## Migration gates

Each stage must satisfy all of the following before the next stage starts:

1. `python -m pytest` passes for the available platform tests;
2. import smoke tests use only the new package name;
3. no canonical module imports a legacy module;
4. the round-trip script still validates the locked assets where the grammar
   binary is available;
5. the git diff contains no unrelated paper, model, dataset, or checkpoint
   files.

The first implementation stage changes only package/import names. Directory
and historical-branch removal happen in later commits so that failures can be
localized and reviewed.

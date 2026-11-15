# NormCoRe Codebase Audit

This document records the completed release refactor and its verification gates.
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

The public API exposes one codec pipeline for this path. Historical island
variants were reviewed and removed from the release tree.

## File inventory and target ownership

| Current path | Current role | Status | Target |
|---|---|---|---|
| `codec.py` | compress/decompress orchestration | canonical | `codec.py` |
| `config.py` | typed configuration and asset paths | canonical | `config.py` |
| `assets.py` | asset and registry validation | canonical | `assets.py` |
| `tokens.py` | tokenizer counting helpers | canonical utility | `tokens.py` |
| `reversible/markers.py` | dialect markers and special tokens | canonical | `reversible/markers.py` |
| `reversible/literals.py` | literal masking/unmasking | canonical | `reversible/literals.py` |
| `reversible/grammar.py` | grammar decode orchestration | canonical | `reversible/grammar.py` |
| `reversible/grammar_edits.py` | decoder edit primitives | canonical | `reversible/grammar_edits.py` |
| `reversible/grammar_rules.py` | Python recovery rules | canonical | `reversible/grammar_rules.py` |
| `reversible/postprocess.py` | decode post-processing | canonical | `reversible/postprocess.py` |
| `reversible/edits.py` | generic span edits | canonical utility | `reversible/edits.py` |
| `normalization/__init__.py` | normalization facade | canonical | `normalization/__init__.py` |
| `normalization/rules.py` | normalization rules | canonical | `normalization/rules.py` |
| `normalization/catalog.py` | rule metadata/catalogue | canonical utility | `normalization/catalog.py` |
| `normalization/type_pack.py` | typed structural packing | canonical | `normalization/type_pack.py` |
| `normalization/idioms.py` | canonical idiom packing/unpacking | canonical | `normalization/idioms.py` |
| `shorthanding/__init__.py` | registry matching and DP selection | canonical | `shorthanding/__init__.py` |
| `shorthanding/mining.py` | pattern mining | canonical research utility | `shorthanding/mining.py` |
| `shorthanding/subtree.py` | subtree candidate pass | canonical helper | `shorthanding/subtree.py` |
| `shorthanding/wildcard.py` | registry wildcard matching | canonical | `shorthanding/wildcard.py` |
| `shorthanding/pipeline.py` | registry + subtree orchestration | canonical | `shorthanding/pipeline.py` |
| `attention/conditional.py` | frozen Conditional Attention policy | canonical | `attention/conditional.py` |
| `attention/scoring.py` | attention/reward weighting | canonical | `attention/scoring.py` |
| `attention/node_score.py` | node-level score extraction | canonical analysis helper | `attention/node_score.py` |
| `attention/profile.py` | model attention profiling | experiment utility | `attention/profile.py` |
| `attention/occurrence.py` | occurrence band annotations | canonical helper | `attention/occurrence.py` |
| `attention/scaffold.py` | scaffold analysis | experiment utility | `attention/scaffold.py` |
| `attention/taxonomy.py` | feature taxonomy | experiment utility | `attention/taxonomy.py` |
| `corpus.py` | corpus loading helpers | experiment utility | `corpus.py` or `experiments/` |
| `mass_profile.py` | large-scale profiling | experiment utility | `experiments/` |
| historical island modules | superseded parallel branches | audited legacy | removed; available only in Git history |

## Naming rules for the refactor

- The import package is `normcore_codec`.
- Public symbols and file names use `NormCoRe` terminology; no canonical code
  introduces the former project name or versioned island names.
- `Conditional Attention` is the paper-facing name. The lowercase `conditional` token
  is used only as a machine-readable mode value.
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

Package naming, directory migration, API terminology, and historical-branch
removal are kept in separate commits so failures remain localized and each
stage can be reviewed independently.

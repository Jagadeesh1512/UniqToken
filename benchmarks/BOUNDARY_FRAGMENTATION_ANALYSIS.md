# Whitespace and Boundary Fragmentation Analysis (Issue #89)

This tokenizer-only comparison measures 36 controlled fixtures and bounded frozen
source probes. The [retained report](boundary_analysis/issue89/REPORT.md) contains
measured tables, with no fixed percentage improvements or universal winner.

## Shared Inputs and Matched Budgets

All models learn from identical generated synthetic documents, seed 42 and
40 documents per language/domain. This generator is not a real corpus.
The existing research training factory checks exactly 1024 IDs in every model,
including the same four controls and all 256 byte fallback IDs.

Boundary-BPE isolates each individual whitespace character. SentencePiece-Unigram
uses identity normalization after shared preprocessing, preserves extra whitespace,
adds no dummy prefix, uses complete character coverage and one thread without
input shuffling. UT-SuperBPE uses a 922-ID Unigram seed and 102 CEM merges,
its existing pre-tokenizer and left-to-right SuperBPE inference, minimum frequency
1, and non-mergeable EOS separators between training documents.
Actual configurations and vocabulary/score/merge hashes are recorded.
These differing configurations do not isolate an algorithmic cause.

## Frozen Source Probes

The retained run uses the immutable phase-a-madlad-stack-flores-v1 manifest.
Declared train/validation hashes and normalized overlap are checked. Split
aliases and directory escapes are rejected before access; the held-out test file
is never opened or hashed. Its hash is copied from manifest metadata only.

Selection takes at most five documents per stratum in frozen file order and their
first 256 raw Unicode characters. Original FLORES validation documents are labeled
`validation/`. Original MADLAD English and The Stack code training documents are
labeled `train_probe/`: unseen by these newly synthetic-trained models, but not
original Phase A held-out validation. Dataset assignments and Phase A models are
unchanged. Synthetic training/probe normalized overlap is rejected before training.
IDs, full source document hashes, excerpt spans and normalized excerpt hashes
are published. Prefix sampling and synthetic training limit generalizability.
Without `--dataset`, only synthetic evidence is emitted.

## Metrics and Source Alignment

Shared preprocessing applies NFKC and Unicode space mapping; repeated spaces,
indentation, tabs and newlines are preserved. NFKC is not raw-text reversible.
Reserved control/metaspace text is rejected. Every piece contributes nonempty
bytes and their concatenation must reconstruct normalized UTF-8 exactly.
Deleted whitespace fails validation rather than becoming a compression gain.

Normalized byte spans tile UTF-8 without gaps. Normalized character spans cover
all characters touched by each piece. Raw character and raw byte spans are source
envelopes from normalization alignment. Byte fallback pieces within one multibyte
character share a nonempty raw span; NFKC expansions may also overlap.
Character envelopes are not a disjoint tiling.

Fragmentation counts every token intersecting a maximal whitespace
(`str.isspace`) or Unicode category P punctuation run. Three fallback bytes in
one punctuation character count as three intersections, one split run and two
excess fragments. Code symbols outside category P are covered by fixtures but
are not counted as punctuation. Cross-field token emissions intersect two
non-whitespace fields separated by whitespace, not binary merge applications.
Bytes/token and tokens/Unicode character pool normalized source counts.
JSON and JSONL include all fixtures and frozen excerpts with every span kind.

## Reproduction and Integrity

Run from a clean committed checkout with the native extension installed:

```powershell
python -m benchmarks.boundary_fragmentation_analysis --vocab-budget 1024 --docs-per-lang 40 --seed 42 --dataset artifacts/data/phase-a-madlad-stack-flores-v1/manifest.json --output artifacts/issue89-new-run
```

Output must be a new directory outside frozen input. Source/runtime identity and
frozen hashes are rechecked before publishing a complete SHA-256 artifact receipt.
The receipt test verifies retained bytes; provenance may vary across machines.
Compare measurements and model hashes separately from runtime identity.

The separation of token boundaries from linguistic morphology is explicit:
no claim is made that tokens preserve morphemes, roots, affixes or clitics.
Whitespace-word fertility is invalid as a universal cross-script metric.
No language model training or held-out test access occurs.

# UniqToken Compatibility Exceptions Matrix

## Overview

UniqToken's Compatibility Engine (`uniqtoken.compat`) provides drop-in tokenization
adapters that produce **identical token IDs** to reference implementations. This
document records the formal compatibility status and any known intentional divergences
between UniqToken and each supported reference tokenizer.

The differential test suite in `tests/test_differential_compat.py` enforces the matrix
on every CI run (Python 3.9-3.12 on Ubuntu, macOS, and Windows). Tests that require
optional reference packages (`tiktoken`, `tokenizers`, `transformers`) skip gracefully
when the package is missing or the reference model cannot be downloaded.

## Compatibility Matrix

| Reference Tokenizer | Preset / Model | Adapter Class | Parity Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| OpenAI `tiktoken` | `cl100k_base` | `TiktokenEncoding` | 100% | Bit-for-bit ID parity across 50k samples |
| OpenAI `tiktoken` | `o200k_base` | `TiktokenEncoding` | 100% | Bit-for-bit ID parity |
| OpenAI `tiktoken` | `gpt2` | `TiktokenEncoding` | 100% | Bit-for-bit ID parity across 10k samples |
| HuggingFace `tokenizers` | ByteLevel BPE (GPT-2) | `HFByteLevelBPE` | 100% | Bit-for-bit ID parity across 50k samples |
| HuggingFace `tokenizers` | LLaMA-3 BPE | `import_hf_tokenizer` | 100% | Bit-for-bit ID parity across 50k samples |
| HuggingFace `tokenizers` | Unigram (LLaMA) | `import_hf_unigram` | 100% | Via Unigram lattice |
| Google `sentencepiece` | Unigram `.model` | `SentencePieceModel` | Conditional | See exception #1 below |

## Known Exceptions & Intentional Divergences

### 1. SentencePiece Leading Dummy Whitespace

SentencePiece prepends a synthetic metaspace (`▁`) to the first token of a sequence by
default (`add_dummy_prefix=True`). UniqToken's normalizer requires explicit configuration
via `prepend_scheme="always"` to replicate this behavior. When importing via
`SentencePieceModel`, the `add_dummy_prefix` flag is respected automatically.

**Impact**: Encode output differs by one leading `▁` token when the
`add_dummy_prefix` configuration is mismatched. The differential test suite accounts
for this by comparing with the flag explicitly set.

### 2. `regex` Package Requirement

Tiktoken and LLaMA-3 pre-tokenization patterns use Unicode property classes
(`\p{L}`, `\p{N}`) and possessive quantifiers (`?+`, `++`) that Python's standard
`re` module cannot parse. The third-party `regex` package is strictly required for
exact parity. Install with:

```bash
pip install regex
```

**Impact**: `TiktokenEncoding` and `HFByteLevelBPE` raise `ImportError` at construction
time if `regex` is not available.

### 3. Special Token Escaping Policy

`TiktokenEncoding.encode()` raises `ValueError` for unescaped special tokens (e.g.,
`<|endoftext|>`) when `allowed_special` does not include them, matching
`tiktoken`'s default behavior exactly.

**Impact**: No divergence - behavior is identical to the reference.

### 4. Digit Chunking in Research Engine

UniqToken's Research Engine defaults to `block3` digit chunking (`\d{1,3}`) for
arithmetic reasoning. The Compatibility Engine preserves each reference tokenizer's
exact digit regex pattern, so no divergence occurs in compatibility mode.

**Impact**: No divergence in compat mode. Research-engine digit chunking is documented
separately in the Research Engine configuration.

### 5. LLaMA-3 Pre-tokenization Pattern

LLaMA-3 uses a different pre-tokenization regex pattern than GPT-2 or `cl100k_base`.
UniqToken's `import_hf_tokenizer()` auto-detects and applies the correct pattern from
the HuggingFace `tokenizer.json` configuration.

**Impact**: No divergence - the pattern is extracted from the source JSON and applied
verbatim.

## How to Run the Differential Test Suite

```bash
# Install required dependencies
pip install regex tiktoken tokenizers transformers sentencepiece

# Run the differential compatibility tests
python -m unittest tests/test_differential_compat.py -v

# Run the full test suite
python -m unittest discover -s tests -p "test_*.py"
```

## Automated CI

The differential test suite runs automatically on every push and pull request via
GitHub Actions (`.github/workflows/ci.yml`). Tests that require optional packages
(`tiktoken`, `tokenizers`, `transformers`) are skipped gracefully when unavailable.

## Updating This Document

When adding a new adapter or discovering a divergence:

1. Add a row to the Compatibility Matrix table above.
2. If the divergence is intentional, add a numbered section under "Known Exceptions".
3. Add a corresponding test case in `tests/test_differential_compat.py`.
4. Run the suite locally to confirm parity before opening a PR.
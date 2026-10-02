# Whitespace and Boundary Fragmentation Analysis (Issue #89)

Three tokenizer conditions have exactly 1024 vocabulary IDs, including
the same four controls and all 256 byte fallback IDs. Each learns from the same
controlled synthetic training documents. These are small transfer diagnostics,
not models trained on the frozen real corpus and not a ranking of architectures.

NFKC and Unicode space mapping are shared. Leading spaces, repeated spaces,
tabs and newlines are preserved. Every scored encoding reconstructs normalized
UTF-8 exactly; deleted whitespace is an error, not a compression improvement.
Boundary-BPE isolates each whitespace character; SPM uses identity normalization
after shared preprocessing, and UT uses its default pre-tokenizer and SuperBPE pass.

## Synthetic Fixtures

| Category | Case | Tokens: BPE / SPM / UT | Bytes/token: BPE / SPM / UT |
| --- | --- | --- | --- |
| whitespace_runs | single_space | 9 / 9 / 9 | 1.222 / 1.222 / 1.222 |
| whitespace_runs | double_space | 10 / 10 / 11 | 1.200 / 1.200 / 1.091 |
| whitespace_runs | four_space_indent | 10 / 8 / 7 | 1.000 / 1.250 / 1.429 |
| whitespace_runs | eight_space_indent | 21 / 20 / 17 | 1.238 / 1.300 / 1.529 |
| whitespace_runs | tab_indentation | 16 / 15 / 21 | 1.500 / 1.600 / 1.143 |
| whitespace_runs | mixed_whitespace_run | 21 / 18 / 18 | 1.190 / 1.389 / 1.389 |
| whitespace_runs | padded_surrounding_whitespace | 16 / 18 / 17 | 1.250 / 1.111 / 1.176 |
| punctuation_sequences | ellipsis_repeats | 27 / 29 / 30 | 1.370 / 1.276 / 1.233 |
| punctuation_sequences | markdown_dividers | 17 / 18 / 17 | 1.353 / 1.278 / 1.353 |
| punctuation_sequences | nested_brackets | 23 / 23 / 23 | 1.087 / 1.087 / 1.087 |
| punctuation_sequences | multi_punctuation_emotive | 28 / 28 / 29 | 1.143 / 1.143 / 1.103 |
| punctuation_sequences | operator_arrows | 22 / 19 / 25 | 1.273 / 1.474 / 1.120 |
| mixed_alphanumeric | semver_version | 23 / 23 / 25 | 1.087 / 1.087 / 1.000 |
| mixed_alphanumeric | hex_memory_address | 48 / 49 / 55 | 1.271 / 1.245 / 1.109 |
| mixed_alphanumeric | system_identifiers | 42 / 43 / 45 | 1.238 / 1.209 / 1.156 |
| mixed_alphanumeric | camel_case_identifiers | 31 / 31 / 31 | 1.258 / 1.258 / 1.258 |
| mixed_alphanumeric | snake_case_identifiers | 25 / 30 / 32 | 1.560 / 1.300 / 1.219 |
| mixed_alphanumeric | kebab_case_headers | 28 / 27 / 30 | 1.536 / 1.593 / 1.433 |
| code_symbols | compound_assignment_operators | 39 / 36 / 35 | 1.000 / 1.083 / 1.114 |
| code_symbols | strict_equality_and_logical | 36 / 32 / 32 | 1.000 / 1.125 / 1.125 |
| code_symbols | bitwise_shift_operators | 38 / 37 / 39 | 1.079 / 1.108 / 1.051 |
| code_symbols | cxx_scope_and_pointers | 39 / 36 / 43 | 1.256 / 1.361 / 1.140 |
| code_symbols | rust_generics_and_returns | 45 / 46 / 44 | 1.200 / 1.174 / 1.227 |
| code_symbols | comment_delimiters | 37 / 35 / 35 | 1.378 / 1.457 / 1.457 |
| cross_word_merges | common_prepositional_phrases | 45 / 42 / 37 | 1.200 / 1.286 / 1.459 |
| cross_word_merges | frequent_determiner_bigrams | 33 / 31 / 25 | 1.273 / 1.355 / 1.680 |
| cross_word_merges | sentence_integration | 62 / 62 / 57 | 1.226 / 1.226 / 1.333 |
| script_boundaries | latin_finnish_compounds | 45 / 45 / 52 | 1.356 / 1.356 / 1.173 |
| script_boundaries | latin_spanish_inverted_punct | 42 / 41 / 42 | 1.190 / 1.220 / 1.190 |
| script_boundaries | cjk_mandarin_unsegmented | 75 / 75 / 75 | 1.293 / 1.293 / 1.293 |
| script_boundaries | cjk_japanese_mixed_scripts | 75 / 75 / 75 | 1.080 / 1.080 / 1.080 |
| script_boundaries | indic_hindi_virama_conjuncts | 54 / 70 / 70 | 3.481 / 2.686 / 2.686 |
| script_boundaries | indic_telugu_combining_vowels | 38 / 51 / 61 | 5.026 / 3.745 / 3.131 |
| script_boundaries | arabic_cursive_and_tatweel | 47 / 50 / 54 | 2.340 / 2.200 / 2.037 |
| script_boundaries | code_python_signature | 53 / 50 / 59 | 1.472 / 1.560 / 1.322 |
| script_boundaries | code_javascript_destructuring | 57 / 54 / 55 | 1.263 / 1.333 / 1.309 |

## Observed Whitespace Fragmentation

| Tokenizer | Fixture whitespace runs | Split runs | Excess fragments | Cross-field token emissions |
| --- | --- | --- | --- | --- |
| Boundary-BPE | 181 | 7 | 24 | 0 |
| SentencePiece-Unigram | 181 | 7 | 24 | 0 |
| UT-SuperBPE | 181 | 5 | 10 | 1 |

## Frozen Real Source Probes

The role is part of every stratum name. `validation/` uses original frozen
validation documents. `train_probe/` uses original source training English
and code documents, unseen by these newly synthetic-trained models. These
are not Phase A held-out validation. No frozen assignments were changed.
The retained run uses at most five documents per stratum, truncated to the
first 256 raw Unicode characters. IDs, source document hashes, excerpt spans
and input hashes are in results.json. This bounded prefix sample is not representative.

| Source role / domain / language | Documents | Bytes/token: BPE / SPM / UT | Tokens/char: BPE / SPM / UT |
| --- | --- | --- | --- |
| train_probe/code/c | 5 | 1.173 / 1.166 / 1.144 | 0.852 / 0.858 / 0.874 |
| train_probe/code/cpp | 5 | 1.185 / 1.199 / 1.154 | 0.847 / 0.837 / 0.870 |
| train_probe/code/go | 5 | 1.203 / 1.234 / 1.187 | 0.831 / 0.810 / 0.842 |
| train_probe/code/java | 5 | 1.203 / 1.240 / 1.214 | 0.831 / 0.806 / 0.823 |
| train_probe/code/javascript | 5 | 1.235 / 1.239 / 1.264 | 0.820 / 0.817 / 0.801 |
| train_probe/code/python | 5 | 1.265 / 1.290 / 1.230 | 0.791 / 0.775 / 0.813 |
| train_probe/code/rust | 5 | 1.210 / 1.233 / 1.223 | 0.826 / 0.811 / 0.817 |
| train_probe/code/sql | 5 | 1.074 / 1.093 / 1.154 | 0.931 / 0.915 / 0.866 |
| train_probe/code/typescript | 5 | 1.258 / 1.284 / 1.160 | 0.795 / 0.779 / 0.862 |
| train_probe/latin_english/en | 5 | 1.231 / 1.241 / 1.209 | 0.825 / 0.818 / 0.840 |
| validation/flores200/am | 5 | 1.001 / 1.001 / 1.001 | 2.512 / 2.514 / 2.512 |
| validation/flores200/ar | 5 | 1.942 / 1.983 / 1.956 | 0.942 / 0.922 / 0.935 |
| validation/flores200/bg | 5 | 1.872 / 1.828 / 2.053 | 0.963 / 0.986 / 0.878 |
| validation/flores200/bn | 5 | 1.000 / 1.000 / 1.003 | 2.697 / 2.697 / 2.689 |
| validation/flores200/fa | 5 | 1.651 / 1.688 / 1.708 | 1.092 / 1.069 / 1.056 |
| validation/flores200/gu | 5 | 1.000 / 1.000 / 1.003 | 2.625 / 2.625 / 2.618 |
| validation/flores200/hi | 5 | 2.349 / 2.285 / 1.867 | 1.082 / 1.113 / 1.362 |
| validation/flores200/ja | 5 | 1.058 / 1.058 / 1.058 | 2.700 / 2.700 / 2.700 |
| validation/flores200/kn | 5 | 1.000 / 1.000 / 1.003 | 2.652 / 2.652 / 2.644 |
| validation/flores200/ko | 5 | 1.000 / 1.000 / 1.004 | 2.384 / 2.384 / 2.373 |
| validation/flores200/ml | 5 | 1.001 / 1.000 / 1.003 | 2.722 / 2.723 / 2.715 |
| validation/flores200/mr | 5 | 2.506 / 2.453 / 1.783 | 1.074 / 1.097 / 1.509 |
| validation/flores200/ru | 5 | 1.885 / 1.835 / 2.029 | 0.963 / 0.989 / 0.895 |
| validation/flores200/sw | 5 | 1.244 / 1.298 / 1.228 | 0.804 / 0.771 / 0.814 |
| validation/flores200/ta | 5 | 1.000 / 1.000 / 1.004 | 2.704 / 2.704 / 2.694 |
| validation/flores200/te | 5 | 2.426 / 2.394 / 1.688 | 1.109 / 1.124 / 1.594 |
| validation/flores200/uk | 5 | 1.747 / 1.743 / 1.905 | 1.039 / 1.042 / 0.953 |
| validation/flores200/ur | 5 | 1.341 / 1.354 / 1.379 | 1.316 / 1.303 / 1.280 |
| validation/flores200/yo | 5 | 1.061 / 1.099 / 1.130 | 1.308 / 1.262 / 1.227 |
| validation/flores200/zh | 5 | 1.347 / 1.350 / 1.340 | 1.931 / 1.927 / 1.941 |

## Audit and Interpretation

- Normalized byte spans tile reconstructed UTF-8 exactly. Raw character and raw
  byte spans are source envelopes. Byte fallback pieces within one multibyte
  character share its nonempty raw character span; NFKC expansions may also overlap.
- Fragmentation counts every token intersecting a maximal whitespace or Unicode
  category P punctuation run, including every byte fallback fragment.
- Cross-field tokens intersect two non-whitespace fields separated by whitespace;
  this is an emission count, not a count of binary merge applications.
- Bytes/token and tokens/Unicode character use pooled normalized source counts.
  The JSONL audit includes synthetic fixtures and every frozen source excerpt.
- Separation of token boundaries from linguistic morphology is explicit:
  no claim is made about preserving morphemes, roots, affixes or clitics.
  Whitespace-word fertility is invalid as a universal cross-script metric.
- Differences describe these configurations, budgets, training data and probes.
  They do not isolate an algorithmic cause or establish general superiority.
- No language model was trained. Only frozen train/validation files were read;
  the declared test path was checked for aliases but never opened or hashed.

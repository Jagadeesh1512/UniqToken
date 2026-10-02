"""Research Benchmark: Whitespace and Boundary Fragmentation Analysis (Issue #89).

Compares UT-SuperBPE, Boundary-BPE, and SentencePiece-Unigram on:
- Whitespace runs (indentation, repeated spaces, tabs, mixed whitespace)
- Punctuation sequences (repeats, bracket nesting, operator combinations)
- Mixed alphanumeric strings (camelCase, snake_case, version strings, hex IDs)
- Code symbols (compound operators, pointers, delimiters, comments)
- Cross-word merges (multi-word tokens bridging whitespace boundaries)
- Script-specific boundary behaviors (Latin, CJK, Indic, Arabic, Code)

Research Integrity Constraints:
- Matched vocabulary budgets across all conditions.
- Strict separation of token-boundary observations from linguistic morphology.
- Avoids whitespace-word fertility as a universal cross-script measure (uses bytes/token
  and tokens/character instead).
- Zero language model training and zero access to held-out test sets.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import unicodedata

from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as research
from benchmarks.analyze_tokenizer_failures import guard_split_paths
from benchmarks.run_matched_budget_eval import generate_balanced_multilingual_corpus
from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.pre_tokenizer import Normalizer


# -----------------------------------------------------------------------------
# Metric Data Structures
# -----------------------------------------------------------------------------


@dataclass
class RunFragmentation:
    run_count: int = 0
    total_characters: int = 0
    token_intersections: int = 0
    split_runs: int = 0
    excess_fragments: int = 0

    @property
    def tokens_per_run(self) -> float:
        return self.token_intersections / max(self.run_count, 1)

    @property
    def split_run_percent(self) -> float:
        return (self.split_runs / max(self.run_count, 1)) * 100.0


@dataclass
class TokenAuditItem:
    token: str
    char_span: Tuple[int, int]
    byte_span: Tuple[int, int]
    byte_length: int
    is_cross_word: bool
    is_byte_fallback: bool
    normalized_char_span: Tuple[int, int]
    raw_byte_span: Tuple[int, int]


@dataclass
class DiagnosticCaseResult:
    category: str
    case_name: str
    text: str
    tokenizer_name: str
    token_count: int
    char_count: int
    byte_count: int
    bytes_per_token: float
    tokens_per_char: float
    cross_word_tokens: int
    cross_word_rate_percent: float
    whitespace_fragmentation: RunFragmentation
    punctuation_fragmentation: RunFragmentation
    audit_tokens: List[TokenAuditItem] = field(default_factory=list)
    normalized_text: str = ""


@dataclass
class AggregateDomainMetrics:
    domain: str
    tokenizer_name: str
    document_count: int
    total_tokens: int
    total_characters: int
    total_bytes: int
    bytes_per_token: float
    tokens_per_char: float
    cross_word_tokens: int
    cross_word_rate_percent: float
    whitespace_fragmentation: RunFragmentation
    punctuation_fragmentation: RunFragmentation


# -----------------------------------------------------------------------------
# Tokenizer Wrapper Protocol
# -----------------------------------------------------------------------------


class TokenizerAdapter:
    def __init__(
        self,
        name: str,
        vocab_size: int,
        encode_pieces_fn: Callable[[str], List[str]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.encode_pieces = encode_pieces_fn
        self.metadata = metadata or {}


# -----------------------------------------------------------------------------
# Diagnostic Measurement & Alignment Engine
# -----------------------------------------------------------------------------


def character_runs(text: str, predicate: Callable[[str], bool]) -> List[Tuple[int, int, str]]:
    """Identifies maximal contiguous character runs satisfying predicate."""
    runs: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    for i, c in enumerate(text):
        if predicate(c):
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i, text[start:i]))
                start = None
    if start is not None:
        runs.append((start, len(text), text[start : len(text)]))
    return runs


def evaluate_run_fragmentation(
    runs: List[Tuple[int, int, str]],
    char_spans: List[Tuple[int, int]],
) -> RunFragmentation:
    """Calculates fragmentation metrics for contiguous character runs."""
    if not runs:
        return RunFragmentation()

    total_chars = sum(end - start for start, end, _ in runs)
    intersections = 0
    split_runs = 0
    excess = 0

    for r_start, r_end, _ in runs:
        # Count tokens intersecting this run
        intersecting = [(s, e) for s, e in char_spans if max(s, r_start) < min(e, r_end)]
        cnt = len(intersecting)
        intersections += cnt
        if cnt > 1:
            split_runs += 1
            excess += cnt - 1

    return RunFragmentation(
        run_count=len(runs),
        total_characters=total_chars,
        token_intersections=intersections,
        split_runs=split_runs,
        excess_fragments=excess,
    )


def decode_piece_bytes(piece: str) -> bytes:
    """Decodes a tokenizer piece into raw bytes."""
    if ByteFallbackEngine.is_byte_token(piece):
        return bytes([ByteFallbackEngine.token_to_byte(piece)])
    # In SentencePiece / CustomTokenizer, \u2581 denotes an explicit space
    normalized_piece = piece.replace("\u2581", " ")
    return normalized_piece.encode("utf-8")


def audit_tokenize(
    tokenizer: TokenizerAdapter, text: str, category: str = "", case_name: str = ""
) -> DiagnosticCaseResult:
    """Encodes text, aligns tokens to exact raw character and byte spans, and records fragmentation."""
    if research.RESERVED_CORPUS_TEXT.search(text):
        raise ValueError("reserved control/metaspace text is not supported by this diagnostic")
    marked, alignment = Normalizer().normalize_with_alignment(text)
    normalized_text = marked.replace("\u2581", " ")
    pieces = tokenizer.encode_pieces(normalized_text)
    decoded_bytes_list = [decode_piece_bytes(p) for p in pieces]
    if any(not part for part in decoded_bytes_list) or b"".join(decoded_bytes_list) != normalized_text.encode("utf-8"):
        raise ValueError(f"{tokenizer.name}: token pieces must reconstruct normalized source exactly")

    # Build byte to character offset map for exact character span alignment
    byte_to_char: List[int] = []
    for c_idx, char in enumerate(normalized_text):
        char_len = len(char.encode("utf-8"))
        for _ in range(char_len):
            byte_to_char.append(c_idx)
    raw_byte_offsets = [0]
    for char in text:
        raw_byte_offsets.append(raw_byte_offsets[-1] + len(char.encode("utf-8")))

    audit_items: List[TokenAuditItem] = []
    char_spans: List[Tuple[int, int]] = []
    b_offset = 0

    for piece, b_part in zip(pieces, decoded_bytes_list):
        b_start = b_offset
        b_end = b_offset + len(b_part)
        b_offset = b_end

        c_start = byte_to_char[b_start]
        c_end = byte_to_char[b_end - 1] + 1
        normalized_span = (c_start, c_end)
        char_spans.append(normalized_span)
        source_spans = alignment[c_start:c_end]
        char_span = (min(start for start, _ in source_spans), max(end for _, end in source_spans))

        span_text = normalized_text[c_start:c_end]
        is_cross = bool(re.search(r"\S\s+\S", span_text))
        is_byte = ByteFallbackEngine.is_byte_token(piece)

        audit_items.append(
            TokenAuditItem(
                token=piece,
                char_span=char_span,
                byte_span=(b_start, b_end),
                byte_length=len(b_part),
                is_cross_word=is_cross,
                is_byte_fallback=is_byte,
                normalized_char_span=normalized_span,
                raw_byte_span=(raw_byte_offsets[char_span[0]], raw_byte_offsets[char_span[1]]),
            )
        )

    # Calculate fragmentation
    ws_runs = character_runs(normalized_text, str.isspace)
    punct_runs = character_runs(normalized_text, lambda c: unicodedata.category(c).startswith("P"))

    ws_frag = evaluate_run_fragmentation(ws_runs, char_spans)
    punct_frag = evaluate_run_fragmentation(punct_runs, char_spans)

    token_count = len(pieces)
    char_count = len(normalized_text)
    byte_count = len(normalized_text.encode("utf-8"))
    bpt = byte_count / max(token_count, 1)
    tpc = token_count / max(char_count, 1)
    cross_word_cnt = sum(1 for it in audit_items if it.is_cross_word)
    cross_word_rate = (cross_word_cnt / max(token_count, 1)) * 100.0

    return DiagnosticCaseResult(
        category=category,
        case_name=case_name,
        text=text,
        tokenizer_name=tokenizer.name,
        token_count=token_count,
        char_count=char_count,
        byte_count=byte_count,
        bytes_per_token=round(bpt, 3),
        tokens_per_char=round(tpc, 3),
        cross_word_tokens=cross_word_cnt,
        cross_word_rate_percent=round(cross_word_rate, 2),
        whitespace_fragmentation=ws_frag,
        punctuation_fragmentation=punct_frag,
        audit_tokens=audit_items,
        normalized_text=normalized_text,
    )


# -----------------------------------------------------------------------------
# Controlled Synthetic Fixtures
# -----------------------------------------------------------------------------


def build_synthetic_fixtures() -> List[Dict[str, str]]:
    """Builds controlled diagnostic test cases covering every required boundary class."""
    return [
        # 1. Whitespace Runs
        {
            "category": "whitespace_runs",
            "name": "single_space",
            "text": "hello world",
        },
        {
            "category": "whitespace_runs",
            "name": "double_space",
            "text": "hello  world",
        },
        {
            "category": "whitespace_runs",
            "name": "four_space_indent",
            "text": "    x = 10",
        },
        {
            "category": "whitespace_runs",
            "name": "eight_space_indent",
            "text": "        return total_value",
        },
        {
            "category": "whitespace_runs",
            "name": "tab_indentation",
            "text": "\t\tdef compute_metrics():",
        },
        {
            "category": "whitespace_runs",
            "name": "mixed_whitespace_run",
            "text": " \t  \n    data_item = None",
        },
        {
            "category": "whitespace_runs",
            "name": "padded_surrounding_whitespace",
            "text": "   isolated block   ",
        },
        # 2. Punctuation Sequences
        {
            "category": "punctuation_sequences",
            "name": "ellipsis_repeats",
            "text": "processing... please wait...... done!",
        },
        {
            "category": "punctuation_sequences",
            "name": "markdown_dividers",
            "text": "--- section divider ---",
        },
        {
            "category": "punctuation_sequences",
            "name": "nested_brackets",
            "text": "result = ([{a: (b + c)}])",
        },
        {
            "category": "punctuation_sequences",
            "name": "multi_punctuation_emotive",
            "text": "Are you sure?!?!?! Absolutely!?!",
        },
        {
            "category": "punctuation_sequences",
            "name": "operator_arrows",
            "text": "source -> target => handler;",
        },
        # 3. Mixed Alphanumeric Strings
        {
            "category": "mixed_alphanumeric",
            "name": "semver_version",
            "text": "v1.2.3-alpha.4+build.2026",
        },
        {
            "category": "mixed_alphanumeric",
            "name": "hex_memory_address",
            "text": "Memory address 0xDEADBEEF loaded into register 0x7FFF5FBFF8A0",
        },
        {
            "category": "mixed_alphanumeric",
            "name": "system_identifiers",
            "text": "SYS_49201 and usr_98a7f failed verification RFC-9110",
        },
        {
            "category": "mixed_alphanumeric",
            "name": "camel_case_identifiers",
            "text": "parseXMLDocument and getUserProfileByID",
        },
        {
            "category": "mixed_alphanumeric",
            "name": "snake_case_identifiers",
            "text": "calculate_total_cross_entropy_reduction",
        },
        {
            "category": "mixed_alphanumeric",
            "name": "kebab_case_headers",
            "text": "content-type-utf-8 and x-forwarded-for-host",
        },
        # 4. Code Symbols
        {
            "category": "code_symbols",
            "name": "compound_assignment_operators",
            "text": "x += 1; y -= 2; z *= 3; w /= 4; p %= 5;",
        },
        {
            "category": "code_symbols",
            "name": "strict_equality_and_logical",
            "text": "if (a === b && c !== d || !(x <= y))",
        },
        {
            "category": "code_symbols",
            "name": "bitwise_shift_operators",
            "text": "flags = (mask >> 4) ^ (val << 2) | ~mask;",
        },
        {
            "category": "code_symbols",
            "name": "cxx_scope_and_pointers",
            "text": "std::vector<int>::iterator ptr = obj->get_iter();",
        },
        {
            "category": "code_symbols",
            "name": "rust_generics_and_returns",
            "text": "fn transform<T: Clone>(item: &T) -> Result<T, Error> {",
        },
        {
            "category": "code_symbols",
            "name": "comment_delimiters",
            "text": "// single line\n/* block comment */\n# python comment",
        },
        # 5. Cross-Word Merges
        {
            "category": "cross_word_merges",
            "name": "common_prepositional_phrases",
            "text": "in the beginning of the project to the end for the win",
        },
        {
            "category": "cross_word_merges",
            "name": "frequent_determiner_bigrams",
            "text": "on the table with the users at the station",
        },
        {
            "category": "cross_word_merges",
            "name": "sentence_integration",
            "text": "In the analysis of the data, we observed that it was of the highest quality.",
        },
        # 6. Language- and Script-Specific Boundaries
        {
            "category": "script_boundaries",
            "name": "latin_finnish_compounds",
            "text": "lentokonesuihkuturbiinimoottoriapumekaanikkoaliupseerioppilas",
        },
        {
            "category": "script_boundaries",
            "name": "latin_spanish_inverted_punct",
            "text": "¿Cómo estás? ¡Muy bien, gracias por preguntar!",
        },
        {
            "category": "script_boundaries",
            "name": "cjk_mandarin_unsegmented",
            "text": "人工智能和自然语言处理技术发展迅速，多语言分词器边界研究十分关键。",
        },
        {
            "category": "script_boundaries",
            "name": "cjk_japanese_mixed_scripts",
            "text": "形態素解析ツールを用いてテキストを適切に分割する処理。",
        },
        {
            "category": "script_boundaries",
            "name": "indic_hindi_virama_conjuncts",
            "text": "प्रणाली और विश्वविद्यालय में अनुसंधान कार्य योजना के अनुसार चल रहा है।",
        },
        {
            "category": "script_boundaries",
            "name": "indic_telugu_combining_vowels",
            "text": "పరిశోధన మరియు అభివృద్ధి నిర్వహణ వ్యవస్థ విధానము ద్వారా సాధ్యమవుతుంది.",
        },
        {
            "category": "script_boundaries",
            "name": "arabic_cursive_and_tatweel",
            "text": "المعـــلوماتية والاســـتراتيجية في معالجة اللغات الطبيعية.",
        },
        {
            "category": "script_boundaries",
            "name": "code_python_signature",
            "text": "def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor) -> float:",
        },
        {
            "category": "script_boundaries",
            "name": "code_javascript_destructuring",
            "text": "const { data: responseData, status } = await api.fetchUser({ id: 101 });",
        },
    ]


# -----------------------------------------------------------------------------
# Tokenizer Training Factory
# -----------------------------------------------------------------------------


def train_all_tokenizers(train_docs: List[str], target_vocab: int = 1024) -> Dict[str, TokenizerAdapter]:
    """Trains UT-SuperBPE, Boundary-BPE, and SentencePiece-Unigram to the exact same matched budget."""
    adapters: Dict[str, TokenizerAdapter] = {}
    normalized = [research.normalize(text) for text in train_docs]
    research.require(all(normalized), "nonempty training documents required")
    research.require(
        not any(research.RESERVED_CORPUS_TEXT.search(text) for text in normalized), "reserved training text"
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name, label in (
            ("boundary_bpe", "Boundary-BPE"),
            ("sp_unigram", "SentencePiece-Unigram"),
            ("uniq_superbpe", "UT-SuperBPE"),
        ):
            tok = research.train_tokenizer(name, normalized, target_vocab, Path(temporary) / name)[0]
            research.validate_tokenizer(tok, target_vocab)

            # The shared factory validates four control IDs and all 256 byte IDs,
            # preserves whitespace, and separates CEM training documents with EOS.
            def encode(text: str, model: Any = tok) -> List[str]:
                return [model.piece_for_id(index) for index in model.encode(text)]

            if name == "sp_unigram":
                scores = [tok.model.get_score(index) for index in range(target_vocab)]
            elif name == "uniq_superbpe":
                scores = sorted(tok.model.model.vocab.items())
            else:
                scores = sorted((left, right, rank) for (left, right), rank in tok.model.merges.items())
            adapters[label] = TokenizerAdapter(
                label,
                len(tok.vocab),
                encode,
                metadata={
                    "configuration": research.tokenizer_configuration(name, target_vocab),
                    "vocabulary_sha256": research.digest(tok.vocab),
                    "scores_or_merges_sha256": research.digest(scores),
                    "actual_vocab_size": len(tok.vocab),
                    "learned_merges": len(tok.model.merges) if name == "boundary_bpe" else tok.merges,
                },
            )
    return adapters


# -----------------------------------------------------------------------------
# Benchmark Runner & Analysis Aggregator
# -----------------------------------------------------------------------------


def load_frozen_probes(dataset_path: Path, per_stratum: int = 5, excerpt_chars: int = 256):
    """Read frozen train/validation only; source training probes are explicitly labeled."""
    research.require(per_stratum > 0 and excerpt_chars > 0, "positive probe caps required")
    manifest = research.read_json(dataset_path)
    guard_split_paths(dataset_path, manifest)
    train_rows, _, validation_rows, _, provenance = stages.load_stage_source(dataset_path)
    research.require(
        not {row["id"] for row in train_rows}.intersection(row["id"] for row in validation_rows),
        "train/validation document ID leakage",
    )
    probes: Dict[str, List[Dict[str, Any]]] = {}
    for role, rows in (("validation", validation_rows), ("train_probe", train_rows)):
        counts: Counter[str] = Counter()
        for row in rows:
            if role == "train_probe" and row["domain"] not in ("code", "latin_english"):
                continue
            key = f"{role}/{row['domain']}/{row['language']}"
            if counts[key] >= per_stratum:
                continue
            text = row["text"][:excerpt_chars]
            research.require(text, "empty source excerpt")
            probes.setdefault(key, []).append(
                {
                    "id": row["id"],
                    "source_role": role,
                    "domain": row["domain"],
                    "language": row["language"],
                    "source_document_sha256": research.digest(row["text"]),
                    "source": row.get("source", {}),
                    "excerpt_raw_char_span": [0, len(text)],
                    "normalized_excerpt_sha256": research.digest(research.normalize(text)),
                    "text": text,
                }
            )
            counts[key] += 1
    research.require(probes, "no frozen source probes selected")
    return probes, {
        **provenance,
        "probe_policy": "first_documents_in_frozen_order_prefix_unicode_characters_v1",
        "documents_per_stratum_cap": per_stratum,
        "excerpt_characters_cap": excerpt_chars,
        "probe_assignment_sha256": research.digest(probes),
        "strata": {key: len(rows) for key, rows in sorted(probes.items())},
        "source_training_probes_are_original_phase_a_validation": False,
    }


class BoundaryFragmentationBenchmark:
    def __init__(self, target_vocab: int = 1024, num_docs_per_lang: int = 40, seed: int = 42):
        research.require(target_vocab > 260 and num_docs_per_lang > 0, "invalid budget or training document count")
        self.target_vocab = target_vocab
        self.num_docs_per_lang = num_docs_per_lang
        self.seed = seed
        self.real_audits: Dict[str, List[DiagnosticCaseResult]] = {}
        self.metadata: Dict[str, Any] = {}

    def run_benchmark(self, probes=None):
        train_docs, _ = generate_balanced_multilingual_corpus(num_docs_per_lang=self.num_docs_per_lang, seed=self.seed)
        training_hashes = {research.digest(research.normalize(text)) for text in train_docs}
        probes = probes or {}
        for rows in probes.values():
            research.require(
                not any(research.digest(research.normalize(row["text"])) in training_hashes for row in rows),
                "synthetic training/source probe leakage",
            )
        tokenizers = train_all_tokenizers(train_docs, target_vocab=self.target_vocab)
        self.metadata = {
            "target_vocab": self.target_vocab,
            "training": {
                "kind": "controlled_generated_synthetic",
                "docs_per_lang": self.num_docs_per_lang,
                "seed": self.seed,
                "documents": len(train_docs),
                "normalized_assignment_sha256": research.digest([research.normalize(text) for text in train_docs]),
            },
            "normalization": research.NORMALIZATION,
            "tokenizers": {name: tok.metadata for name, tok in tokenizers.items()},
            "test_access": "forbidden_not_opened",
            "language_model_training": False,
        }
        synthetic_results: Dict[str, List[DiagnosticCaseResult]] = {name: [] for name in tokenizers}
        for fixture in build_synthetic_fixtures():
            for name, tok in tokenizers.items():
                synthetic_results[name].append(
                    audit_tokenize(tok, fixture["text"], fixture["category"], fixture["name"])
                )
        domain_results: Dict[str, Dict[str, AggregateDomainMetrics]] = {name: {} for name in tokenizers}
        self.real_audits = {name: [] for name in tokenizers}
        for domain, rows in sorted(probes.items()):
            for name, tok in tokenizers.items():
                results = [audit_tokenize(tok, row["text"], domain, row["id"]) for row in rows]
                self.real_audits[name].extend(results)
                counts = {}
                for field_name in ("whitespace_fragmentation", "punctuation_fragmentation"):
                    counts[field_name] = RunFragmentation(
                        **{
                            field: sum(getattr(getattr(result, field_name), field) for result in results)
                            for field in RunFragmentation.__dataclass_fields__
                        }
                    )
                tokens = sum(result.token_count for result in results)
                chars = sum(result.char_count for result in results)
                byte_count = sum(result.byte_count for result in results)
                cross = sum(result.cross_word_tokens for result in results)
                domain_results[name][domain] = AggregateDomainMetrics(
                    domain=domain,
                    tokenizer_name=name,
                    document_count=len(results),
                    total_tokens=tokens,
                    total_characters=chars,
                    total_bytes=byte_count,
                    bytes_per_token=byte_count / tokens,
                    tokens_per_char=tokens / chars,
                    cross_word_tokens=cross,
                    cross_word_rate_percent=100 * cross / tokens,
                    **counts,
                )
        return synthetic_results, domain_results


# -----------------------------------------------------------------------------
# Exporters & Report Formatter
# -----------------------------------------------------------------------------


def format_markdown_report(
    synthetic_results: Dict[str, List[DiagnosticCaseResult]],
    domain_results: Dict[str, Dict[str, AggregateDomainMetrics]],
    vocab_size: int,
) -> str:
    names = ("Boundary-BPE", "SentencePiece-Unigram", "UT-SuperBPE")
    lines = [
        "# Whitespace and Boundary Fragmentation Analysis (Issue #89)",
        "",
        f"Three tokenizer conditions have exactly {vocab_size} vocabulary IDs, including",
        "the same four controls and all 256 byte fallback IDs. Each learns from the same",
        "controlled synthetic training documents. These are small transfer diagnostics,",
        "not models trained on the frozen real corpus and not a ranking of architectures.",
        "",
        "NFKC and Unicode space mapping are shared. Leading spaces, repeated spaces,",
        "tabs and newlines are preserved. Every scored encoding reconstructs normalized",
        "UTF-8 exactly; deleted whitespace is an error, not a compression improvement.",
        "Boundary-BPE isolates each whitespace character; SPM uses identity normalization",
        "after shared preprocessing, and UT uses its default pre-tokenizer and SuperBPE pass.",
        "",
        "## Synthetic Fixtures",
        "",
        "| Category | Case | Tokens: BPE / SPM / UT | Bytes/token: BPE / SPM / UT |",
        "| --- | --- | --- | --- |",
    ]
    indexed = {name: {r.case_name: r for r in synthetic_results[name]} for name in names}
    for result in synthetic_results[names[0]]:
        rows = [indexed[name][result.case_name] for name in names]
        tokens = " / ".join(str(row.token_count) for row in rows)
        ratios = " / ".join(f"{row.bytes_per_token:.3f}" for row in rows)
        lines.append(f"| {result.category} | {result.case_name} | {tokens} | {ratios} |")
    lines.extend(
        [
            "",
            "## Observed Whitespace Fragmentation",
            "",
            "| Tokenizer | Fixture whitespace runs | Split runs | Excess fragments | Cross-field token emissions |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name in names:
        rows = synthetic_results[name]
        runs = sum(r.whitespace_fragmentation.run_count for r in rows)
        splits = sum(r.whitespace_fragmentation.split_runs for r in rows)
        excess = sum(r.whitespace_fragmentation.excess_fragments for r in rows)
        cross = sum(r.cross_word_tokens for r in rows)
        lines.append(f"| {name} | {runs} | {splits} | {excess} | {cross} |")
    lines.extend(
        [
            "",
            "## Frozen Real Source Probes",
            "",
            "The role is part of every stratum name. `validation/` uses original frozen",
            "validation documents. `train_probe/` uses original source training English",
            "and code documents, unseen by these newly synthetic-trained models. These",
            "are not Phase A held-out validation. No frozen assignments were changed.",
            "The retained run uses at most five documents per stratum, truncated to the",
            "first 256 raw Unicode characters. IDs, source document hashes, excerpt spans",
            "and input hashes are in results.json. This bounded prefix sample is not representative.",
            "",
            "| Source role / domain / language | Documents | Bytes/token: BPE / SPM / UT | Tokens/char: BPE / SPM / UT |",
            "| --- | --- | --- | --- |",
        ]
    )
    for domain in sorted(domain_results[names[0]]):
        domain_rows = [domain_results[name][domain] for name in names]
        ratios = " / ".join(f"{row.bytes_per_token:.3f}" for row in domain_rows)
        chars = " / ".join(f"{row.tokens_per_char:.3f}" for row in domain_rows)
        lines.append(f"| {domain} | {domain_rows[0].document_count} | {ratios} | {chars} |")
    if not domain_results[names[0]]:
        lines.append("No frozen corpus was supplied; this run contains synthetic evidence only.")
    lines.extend(
        [
            "",
            "## Audit and Interpretation",
            "",
            "- Normalized byte spans tile reconstructed UTF-8 exactly. Raw character and raw",
            "  byte spans are source envelopes. Byte fallback pieces within one multibyte",
            "  character share its nonempty raw character span; NFKC expansions may also overlap.",
            "- Fragmentation counts every token intersecting a maximal whitespace or Unicode",
            "  category P punctuation run, including every byte fallback fragment.",
            "- Cross-field tokens intersect two non-whitespace fields separated by whitespace;",
            "  this is an emission count, not a count of binary merge applications.",
            "- Bytes/token and tokens/Unicode character use pooled normalized source counts.",
            "  The JSONL audit includes synthetic fixtures and every frozen source excerpt.",
            "- Separation of token boundaries from linguistic morphology is explicit:",
            "  no claim is made about preserving morphemes, roots, affixes or clitics.",
            "  Whitespace-word fertility is invalid as a universal cross-script metric.",
            "- Differences describe these configurations, budgets, training data and probes.",
            "  They do not isolate an algorithmic cause or establish general superiority.",
            "- No language model was trained. Only frozen train/validation files were read;",
            "  the declared test path was checked for aliases but never opened or hashed.",
            "",
        ]
    )
    return "\n".join(lines)


def export_all_artifacts(
    synthetic_results: Dict[str, List[DiagnosticCaseResult]],
    domain_results: Dict[str, Dict[str, AggregateDomainMetrics]],
    vocab_size: int,
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    real_audits: Optional[Dict[str, List[DiagnosticCaseResult]]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)

    # 1. REPORT.md
    report_text = format_markdown_report(synthetic_results, domain_results, vocab_size)
    (output_dir / "REPORT.md").write_text(report_text, encoding="utf-8")

    # 2. synthetic_diagnostics.csv
    synth_csv = output_dir / "synthetic_diagnostics.csv"
    with synth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "category",
                "case_name",
                "tokenizer",
                "token_count",
                "char_count",
                "byte_count",
                "bytes_per_token",
                "tokens_per_char",
                "cross_word_tokens",
                "cross_word_rate_percent",
                "ws_runs",
                "ws_split_runs",
                "ws_tokens_per_run",
                "punct_runs",
                "punct_split_runs",
                "punct_tokens_per_run",
            ]
        )
        for tok_name, r_list in synthetic_results.items():
            for r in r_list:
                writer.writerow(
                    [
                        r.category,
                        r.case_name,
                        tok_name,
                        r.token_count,
                        r.char_count,
                        r.byte_count,
                        r.bytes_per_token,
                        r.tokens_per_char,
                        r.cross_word_tokens,
                        r.cross_word_rate_percent,
                        r.whitespace_fragmentation.run_count,
                        r.whitespace_fragmentation.split_runs,
                        f"{r.whitespace_fragmentation.tokens_per_run:.2f}",
                        r.punctuation_fragmentation.run_count,
                        r.punctuation_fragmentation.split_runs,
                        f"{r.punctuation_fragmentation.tokens_per_run:.2f}",
                    ]
                )

    # 3. real_corpus_summary.csv
    real_csv = output_dir / "real_corpus_summary.csv"
    with real_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "domain",
                "tokenizer",
                "documents",
                "total_tokens",
                "total_characters",
                "total_bytes",
                "bytes_per_token",
                "tokens_per_char",
                "cross_word_tokens",
                "cross_word_rate_percent",
                "ws_split_run_percent",
                "punct_split_run_percent",
            ]
        )
        for tok_name, d_map in domain_results.items():
            for d_name, m in d_map.items():
                writer.writerow(
                    [
                        d_name,
                        tok_name,
                        m.document_count,
                        m.total_tokens,
                        m.total_characters,
                        m.total_bytes,
                        m.bytes_per_token,
                        m.tokens_per_char,
                        m.cross_word_tokens,
                        m.cross_word_rate_percent,
                        f"{m.whitespace_fragmentation.split_run_percent:.2f}",
                        f"{m.punctuation_fragmentation.split_run_percent:.2f}",
                    ]
                )

    # 4. span_audit_examples.jsonl
    span_jsonl = output_dir / "span_audit_examples.jsonl"
    with span_jsonl.open("w", encoding="utf-8") as f:
        for tok_name, r_list in synthetic_results.items():
            for r in r_list + (real_audits or {}).get(tok_name, []):
                payload = {
                    "tokenizer": tok_name,
                    "category": r.category,
                    "case_name": r.case_name,
                    "text": r.text,
                    "normalized_text": r.normalized_text,
                    "tokens": [it.token for it in r.audit_tokens],
                    "char_spans": [list(it.char_span) for it in r.audit_tokens],
                    "normalized_char_spans": [list(it.normalized_char_span) for it in r.audit_tokens],
                    "raw_byte_spans": [list(it.raw_byte_span) for it in r.audit_tokens],
                    "byte_spans": [list(it.byte_span) for it in r.audit_tokens],
                    "byte_lengths": [it.byte_length for it in r.audit_tokens],
                    "is_cross_word": [it.is_cross_word for it in r.audit_tokens],
                    "token_count": r.token_count,
                    "bytes_per_token": r.bytes_per_token,
                    "tokens_per_char": r.tokens_per_char,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 5. results.json
    def serialize_res(r: DiagnosticCaseResult) -> Dict[str, Any]:
        d = asdict(r)
        d["audit_tokens"] = [asdict(t) for t in r.audit_tokens]
        return d

    res_json = {
        "metadata": {
            **(metadata or {}),
            "target_vocab": vocab_size,
            "tokenizer_names": list(synthetic_results.keys()),
        },
        "synthetic_diagnostics": {k: [serialize_res(r) for r in v] for k, v in synthetic_results.items()},
        "real_corpus_summary": {k: {d: asdict(m) for d, m in v.items()} for k, v in domain_results.items()},
        "real_corpus_span_audit_artifact": "span_audit_examples.jsonl",
        "real_corpus_diagnostics": {
            name: [
                {
                    key: value
                    for key, value in asdict(result).items()
                    if key not in ("audit_tokens", "text", "normalized_text")
                }
                for result in rows
            ]
            for name, rows in (real_audits or {}).items()
        },
    }
    (output_dir / "results.json").write_text(json.dumps(res_json, indent=2, ensure_ascii=False), encoding="utf-8")
    research.write_new_json(
        output_dir / "manifest.json",
        {
            "status": "complete",
            "artifacts": {
                name: research.file_hash(output_dir / name)
                for name in (
                    "REPORT.md",
                    "synthetic_diagnostics.csv",
                    "real_corpus_summary.csv",
                    "span_audit_examples.jsonl",
                    "results.json",
                )
            },
        },
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Whitespace and Boundary Fragmentation Analysis Benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/boundary-fragmentation-issue89",
        help="Output path for benchmark reports and JSON/CSV artifacts",
    )
    parser.add_argument(
        "--vocab-budget",
        type=int,
        default=1024,
        help="Matched vocabulary budget for all tokenizers (default 1024)",
    )
    parser.add_argument(
        "--docs-per-lang",
        type=int,
        default=40,
        help="Number of training documents per language/domain (default 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible corpus generation",
    )
    parser.add_argument("--dataset", type=Path, help="Frozen manifest for train/validation source probes only")
    args = parser.parse_args()
    out_dir = Path(args.output)
    research.require(not out_dir.exists(), "output must be a new directory; existing evidence cannot be overwritten")
    if args.dataset:
        research.require(
            not out_dir.resolve().is_relative_to(args.dataset.resolve().parent), "output is inside frozen input"
        )
    identity = research.runtime_identity()
    research.require(not identity["working_tree_dirty"], "commit the experiment before recording research evidence")
    probes, provenance = load_frozen_probes(args.dataset) if args.dataset else ({}, {"kind": "not_supplied"})

    bench = BoundaryFragmentationBenchmark(
        target_vocab=args.vocab_budget,
        num_docs_per_lang=args.docs_per_lang,
        seed=args.seed,
    )
    synth_res, domain_res = bench.run_benchmark(probes)
    research.require(research.runtime_identity() == identity, "source/runtime changed during analysis")
    if args.dataset:
        research.require(
            research.file_hash(args.dataset) == provenance["manifest_sha256"], "manifest changed during analysis"
        )
        paths = guard_split_paths(args.dataset, research.read_json(args.dataset))
        for split in ("train", "validation"):
            research.require(research.file_hash(paths[split]) == provenance[f"{split}_file_sha256"], f"{split} changed")
    bench.metadata.update({"identity": identity, "frozen_source": provenance, "source_probes": probes})
    export_all_artifacts(synth_res, domain_res, args.vocab_budget, out_dir, bench.metadata, bench.real_audits)

    print("\n" + "=" * 80)
    print("BOUNDARY FRAGMENTATION BENCHMARK COMPLETE")
    print("=" * 80)
    report_text = format_markdown_report(synth_res, domain_res, args.vocab_budget)
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nArtifacts successfully written to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()

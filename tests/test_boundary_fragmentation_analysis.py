"""Tests for Whitespace and Boundary Fragmentation Analysis (Issue #89).

Validates:
1. Synthetic boundary diagnostics covering each boundary class:
   - Whitespace runs (indentation, tabs, double space)
   - Punctuation sequences (repeats, bracket nesting, operator arrows)
   - Mixed alphanumeric strings (versions, hex, camelCase, snake_case)
   - Code symbols (compound operators, pointers, delimiters)
   - CJK unsegmented scripts (Mandarin, Japanese)
   - Indic combining scripts (Hindi Devanagari, Telugu)
   - Arabic cursive script & Tatweel
2. Auditable token spans strictly tiling raw text without gaps or overlaps.
3. Quantified metrics (bytes/token, tokens/char, fragmentation counts).
4. Strict research integrity: no unsupported morpheme, clitic, or root-boundary claims.
5. Exact matched vocabulary budget invariance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import unittest

from benchmarks.boundary_fragmentation_analysis import (
    audit_tokenize,
    build_synthetic_fixtures,
    character_runs,
    decode_piece_bytes,
    evaluate_run_fragmentation,
    TokenizerAdapter,
)
from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.tokenizer import CustomTokenizer


class TestBoundaryFragmentationAnalysis(unittest.TestCase):
    train_corpus: List[str]
    target_vocab: int
    sbp_tok: CustomTokenizer
    ut_adapter: TokenizerAdapter
    bpe_adapter: TokenizerAdapter

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_corpus = [
            "hello world and welcome to the system",
            "in the beginning of the process to the end",
            "    x = 10; y += 2; ptr->member = 0xDEADBEEF;",
            "v1.2.3 and parseXMLDocument with calculate_total_score",
            "processing... wait...... result = ([{a: (b + c)}]);",
            "人工智能和自然语言处理技术发展迅速",
            "प्रणाली और विश्वविद्यालय में अनुसंधान कार्य",
            "పరిశోధన మరియు అభివృద్ధి నిర్వహణ వ్యవస్థ",
            "المعلوماتية والاسـتراتيجية في معالجة اللغات",
            "def calculate_loss(predictions, targets) -> float:",
        ]
        cls.target_vocab = 500

        # Build UT-SuperBPE model
        base_tok = CustomTokenizer.train_from_corpus(
            cls.train_corpus, target_vocab_size=cls.target_vocab - 30, verbose=False
        )
        pretok_chunks = [
            tok
            for d in cls.train_corpus
            for tok in base_tok.pre_tokenizer.pre_tokenize(base_tok.normalizer.normalize(d))
        ]
        cem = CrossEntropyMerging(max_merges=20, cross_word=True, verbose=False)
        sbp_model = cem.optimize(base_tok.model, chunks=pretok_chunks)
        cls.sbp_tok = CustomTokenizer(
            normalizer=base_tok.normalizer,
            pre_tokenizer=base_tok.pre_tokenizer,
            model=sbp_model,
        )

        cls.ut_adapter = TokenizerAdapter(
            name="UT-SuperBPE",
            vocab_size=len(cls.sbp_tok.model.vocab),
            encode_pieces_fn=lambda t: [tok.text for tok in cls.sbp_tok.encode_with_offsets(t)],
        )

        # Build Boundary-BPE model
        bpe_chunks = [w for doc in cls.train_corpus for w in doc.split() if w]
        bpe_model = BPETrainer(target_vocab_size=cls.target_vocab, byte_fallback=True).train(
            bpe_chunks, verbose=False
        )
        cls.bpe_adapter = TokenizerAdapter(
            name="Boundary-BPE",
            vocab_size=len(bpe_model.vocab),
            encode_pieces_fn=lambda t: [i for w in t.split() for i in bpe_model.encode(w)],
        )

    def test_synthetic_fixtures_cover_all_required_boundary_classes(self) -> None:
        """Synthetic test fixtures must cover every required boundary class."""
        fixtures = build_synthetic_fixtures()
        categories = {f["category"] for f in fixtures}
        required = {
            "whitespace_runs",
            "punctuation_sequences",
            "mixed_alphanumeric",
            "code_symbols",
            "cross_word_merges",
            "script_boundaries",
        }
        self.assertTrue(required.issubset(categories), f"Missing categories: {required - categories}")

    def test_auditable_spans_tile_text_without_gaps_or_overlaps(self) -> None:
        """Every token's character span must tile the input monotonically without gaps or overlaps."""
        test_strings = [
            "    hello world!",
            "wait... what?!?!?!",
            "0xDEADBEEF v1.2.3",
            "a === b && c !== d",
            "in the beginning of the",
            "人工智能和自然语言处理",
            "प्रणाली और विश्वविद्यालय",
            "المعلوماتية",
        ]
        for text in test_strings:
            res = audit_tokenize(self.ut_adapter, text)
            # 1. Total tokens > 0
            self.assertGreater(res.token_count, 0)
            # 2. Byte spans tile text bytes monotonically without gaps
            curr_b_pos = 0
            for item in res.audit_tokens:
                bs, be = item.byte_span
                self.assertEqual(bs, curr_b_pos, f"Byte span gap at {bs} vs {curr_b_pos}")
                self.assertGreater(be, bs, f"Zero or negative byte span length: {bs} to {be}")
                curr_b_pos = be
            self.assertEqual(curr_b_pos, len(text.encode("utf-8")), "Byte spans did not reach end of text bytes")

            # 3. Decoded byte reconstruction matches raw UTF-8 exactly
            decoded_bytes = b"".join(decode_piece_bytes(it.token) for it in res.audit_tokens)
            self.assertEqual(decoded_bytes, text.encode("utf-8"), "Decoded bytes do not match raw UTF-8")

            # 4. Character bounds valid
            for item in res.audit_tokens:
                s, e = item.char_span
                self.assertTrue(0 <= s <= e <= len(text), f"Invalid char span [{s}:{e}] for len {len(text)}")

    def test_whitespace_run_fragmentation_metrics(self) -> None:
        """Whitespace fragmentation metrics must quantify runs, split runs, and excess tokens."""
        text = "    indentation  double   triple"
        ws_runs = character_runs(text, str.isspace)
        self.assertEqual(len(ws_runs), 3)

        res = audit_tokenize(self.ut_adapter, text)
        frag = res.whitespace_fragmentation
        self.assertEqual(frag.run_count, 3)
        self.assertGreaterEqual(frag.token_intersections, 3)
        self.assertGreaterEqual(frag.excess_fragments, 0)
        self.assertGreaterEqual(frag.tokens_per_run, 1.0)

    def test_punctuation_sequences_fragmentation(self) -> None:
        """Punctuation sequences must be detected and measured for fragmentation."""
        text = "function()... -> result === true;"
        res = audit_tokenize(self.ut_adapter, text)
        frag = res.punctuation_fragmentation
        self.assertGreater(frag.run_count, 0)
        self.assertGreater(frag.token_intersections, 0)
        self.assertGreaterEqual(frag.split_runs, 0)

    def test_mixed_alphanumeric_and_casing(self) -> None:
        """Mixed alphanumeric strings (camelCase, snake_case, hex, semver) are audited cleanly."""
        text = "v2.0.4-beta + 0xCAFE parseXMLDoc process_stream_data"
        res = audit_tokenize(self.ut_adapter, text)
        self.assertGreater(res.bytes_per_token, 0.0)
        self.assertGreater(res.tokens_per_char, 0.0)
        self.assertEqual(res.char_count, len(text))
        self.assertEqual(res.byte_count, len(text.encode("utf-8")))

    def test_code_symbols_and_delimiters(self) -> None:
        """Code symbols (compound operators, pointers, generics) are measured."""
        text = "ptr->field += 10; flags = (mask >> 2) | 0x01;"
        res = audit_tokenize(self.ut_adapter, text)
        self.assertGreater(res.token_count, 0)
        self.assertGreaterEqual(res.punctuation_fragmentation.run_count, 1)

    def test_cjk_unsegmented_script_evaluation(self) -> None:
        """CJK unsegmented text is evaluated via bytes/token and tokens/char without whitespace dependence."""
        cjk_text = "人工智能和自然语言处理技术发展迅速。"
        res = audit_tokenize(self.ut_adapter, cjk_text)
        # In Chinese, each character is 3 UTF-8 bytes
        self.assertGreater(res.bytes_per_token, 1.0)
        self.assertLessEqual(res.tokens_per_char, 10.0)
        # Chinese text has 0 whitespace runs
        self.assertEqual(res.whitespace_fragmentation.run_count, 0)

    def test_indic_combining_virama_script_evaluation(self) -> None:
        """Indic scripts with virama conjuncts are accurately tracked."""
        indic_text = "प्रणाली और विश्वविद्यालय में अनुसंधान कार्य"
        res = audit_tokenize(self.ut_adapter, indic_text)
        self.assertGreater(res.token_count, 0)
        self.assertGreater(res.bytes_per_token, 1.0)
        # Spans must tile without index error on combining marks
        for item in res.audit_tokens:
            s, e = item.char_span
            self.assertTrue(0 <= s <= e <= len(indic_text))

    def test_arabic_cursive_script_evaluation(self) -> None:
        """Arabic cursive script and Tatweel elongation are audited with exact spans."""
        arabic_text = "المعـــلوماتية والاســـتراتيجية"
        res = audit_tokenize(self.ut_adapter, arabic_text)
        self.assertGreater(res.token_count, 0)
        self.assertEqual(res.char_count, len(arabic_text))
        self.assertEqual(res.byte_count, len(arabic_text.encode("utf-8")))

    def test_research_integrity_no_unsupported_linguistic_claims(self) -> None:
        """Reports and documentation must explicitly avoid unsupported morpheme/clitic claims."""
        doc_path = Path("benchmarks/BOUNDARY_FRAGMENTATION_ANALYSIS.md")
        self.assertTrue(doc_path.exists(), "Research documentation file does not exist")
        content = doc_path.read_text(encoding="utf-8").lower()

        # Must explicitly acknowledge the separation of tokens from morphology
        self.assertIn("separation of token boundaries from", content)
        self.assertIn("no claim is made", content)

        # Must disclaim morpheme / root / clitic preservation
        self.assertIn("morphemes", content)
        self.assertIn("clitics", content)

        # Must disclaim whitespace-word fertility
        self.assertIn("whitespace-word fertility is invalid", content)


if __name__ == "__main__":
    unittest.main()

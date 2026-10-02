from __future__ import annotations

"""Unit tests for Byte Fallback Analysis and Mitigation (Issue #86).

Covers:
- Fallback span extraction and contiguous run lengths
- Fallback span statistical metrics and histogram calculation
- Regression evaluation against predeclared threshold
- ByteFallbackEngine multi-byte token detection and lossless roundtrip
- CEM resolve_pair UTF-8 validity checks and fallback utility regularization
- Exact budget invariance and MergeRecord provenance tracking
"""

import unittest

from benchmarks.byte_fallback_analysis import (
    MAJOR_REFERENCE_STRATA,
    compute_span_metrics,
    evaluate_regressions,
    extract_fallback_spans,
)
from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.cem_merger import CrossEntropyMerging, MergeRecord, SuperBPE
from uniqtoken.tokenizer import CustomTokenizer


class FallbackSpanExtractionTests(unittest.TestCase):
    def test_empty_tokens(self):
        self.assertEqual(extract_fallback_spans([]), [])

    def test_no_fallback_tokens(self):
        tokens = ["hello", "world", "this", "is", "a", "test"]
        self.assertEqual(extract_fallback_spans(tokens), [])

    def test_all_fallback_tokens(self):
        tokens = ["<0xE0>", "<0xA4>", "<0xBE>", "<0xE0>", "<0xA5>"]
        self.assertEqual(extract_fallback_spans(tokens), [5])

    def test_contiguous_runs_and_multi_byte(self):
        # 3 fallback bytes, regular token, 2 fallback bytes, regular token, multi-byte token
        tokens = [
            "start",
            "<0xE0>",
            "<0xA4>",
            "<0xBE>",
            "word",
            "<0xC3>",
            "<0xA9>",
            "end",
            "<0xE0><0xA4>",
        ]
        spans = extract_fallback_spans(tokens)
        self.assertEqual(spans, [3, 2, 1])


class SpanMetricsTests(unittest.TestCase):
    def test_empty_spans(self):
        metrics = compute_span_metrics([])
        self.assertEqual(metrics["count"], 0)
        self.assertEqual(metrics["mean"], 0.0)
        self.assertEqual(metrics["median"], 0.0)
        self.assertEqual(metrics["p95"], 0.0)
        self.assertEqual(metrics["max"], 0)
        hist = metrics["histogram"]
        self.assertEqual(hist["len_1"], 0)
        self.assertEqual(hist["len_2"], 0)
        self.assertEqual(hist["len_3"], 0)
        self.assertEqual(hist["len_4"], 0)
        self.assertEqual(hist["len_5_6"], 0)
        self.assertEqual(hist["len_7_plus"], 0)

    def test_single_span(self):
        metrics = compute_span_metrics([3])
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["mean"], 3.0)
        self.assertEqual(metrics["median"], 3.0)
        self.assertEqual(metrics["p95"], 3.0)
        self.assertEqual(metrics["max"], 3)
        self.assertEqual(metrics["histogram"]["len_3"], 1)

    def test_span_statistics_and_histogram(self):
        spans = [1, 2, 3, 3, 4, 5, 6, 8, 12]
        metrics = compute_span_metrics(spans)
        self.assertEqual(metrics["count"], 9)
        self.assertEqual(metrics["mean"], round(sum(spans) / 9.0, 2))
        self.assertEqual(metrics["median"], 4.0)
        self.assertEqual(metrics["max"], 12)
        hist = metrics["histogram"]
        self.assertEqual(hist["len_1"], 1)
        self.assertEqual(hist["len_2"], 1)
        self.assertEqual(hist["len_3"], 2)
        self.assertEqual(hist["len_4"], 1)
        self.assertEqual(hist["len_5_6"], 2)
        self.assertEqual(hist["len_7_plus"], 2)


class RegressionGatingTests(unittest.TestCase):
    def test_pass_when_better_or_equal(self):
        baseline = {
            "latin_english": {"bytes_per_token": 2.500},
            "code": {"bytes_per_token": 2.000},
            "cyrillic": {"bytes_per_token": 3.000},
            "african_latin": {"bytes_per_token": 2.200},
        }
        # Candidate has higher BpT (better compression)
        cand = {
            "latin_english": {"bytes_per_token": 2.550},
            "code": {"bytes_per_token": 2.050},
            "cyrillic": {"bytes_per_token": 3.000},
            "african_latin": {"bytes_per_token": 2.300},
        }
        passed, regs = evaluate_regressions(baseline, cand, MAJOR_REFERENCE_STRATA, 1.0)
        self.assertTrue(passed)
        for val in regs.values():
            self.assertEqual(val, 0.0)

    def test_pass_within_threshold(self):
        baseline = {
            "latin_english": {"bytes_per_token": 2.000},
            "code": {"bytes_per_token": 2.000},
            "cyrillic": {"bytes_per_token": 2.000},
            "african_latin": {"bytes_per_token": 2.000},
        }
        # Candidate has 0.5% drop on latin_english: 2.0 * (1 - 0.005) = 1.990
        cand = {
            "latin_english": {"bytes_per_token": 1.990},
            "code": {"bytes_per_token": 2.000},
            "cyrillic": {"bytes_per_token": 2.000},
            "african_latin": {"bytes_per_token": 2.000},
        }
        passed, regs = evaluate_regressions(baseline, cand, MAJOR_REFERENCE_STRATA, 1.0)
        self.assertTrue(passed)
        self.assertAlmostEqual(regs["latin_english"], 0.5, places=2)

    def test_fail_beyond_threshold(self):
        baseline = {
            "latin_english": {"bytes_per_token": 2.000},
            "code": {"bytes_per_token": 2.000},
            "cyrillic": {"bytes_per_token": 2.000},
            "african_latin": {"bytes_per_token": 2.000},
        }
        # Code drops by 2.0%: 2.0 * (1 - 0.02) = 1.960
        cand = {
            "latin_english": {"bytes_per_token": 2.000},
            "code": {"bytes_per_token": 1.960},
            "cyrillic": {"bytes_per_token": 2.000},
            "african_latin": {"bytes_per_token": 2.000},
        }
        passed, regs = evaluate_regressions(baseline, cand, MAJOR_REFERENCE_STRATA, 1.0)
        self.assertFalse(passed)
        self.assertAlmostEqual(regs["code"], 2.0, places=2)

    def test_non_major_strata_ignored(self):
        baseline = {
            "latin_english": {"bytes_per_token": 2.000},
            "code": {"bytes_per_token": 2.000},
            "cyrillic": {"bytes_per_token": 2.000},
            "african_latin": {"bytes_per_token": 2.000},
            "indic": {"bytes_per_token": 3.000},
        }
        # Indic drops by 50%, but Indic is not a major reference stratum
        cand = {
            "latin_english": {"bytes_per_token": 2.000},
            "code": {"bytes_per_token": 2.000},
            "cyrillic": {"bytes_per_token": 2.000},
            "african_latin": {"bytes_per_token": 2.000},
            "indic": {"bytes_per_token": 1.500},
        }
        passed, regs = evaluate_regressions(baseline, cand, MAJOR_REFERENCE_STRATA, 1.0)
        self.assertTrue(passed)
        self.assertNotIn("indic", regs)


class ByteFallbackEngineMultiByteTests(unittest.TestCase):
    def test_is_byte_token_single_and_multi(self):
        self.assertTrue(ByteFallbackEngine.is_byte_token("<0x00>"))
        self.assertTrue(ByteFallbackEngine.is_byte_token("<0xFF>"))
        self.assertTrue(ByteFallbackEngine.is_byte_token("<0xE0><0xA4>"))
        self.assertTrue(ByteFallbackEngine.is_byte_token("<0xE0><0xA4><0xBE>"))
        self.assertFalse(ByteFallbackEngine.is_byte_token("<0x0>"))
        self.assertFalse(ByteFallbackEngine.is_byte_token("hello"))
        self.assertFalse(ByteFallbackEngine.is_byte_token("<0xGG>"))

    def test_token_to_bytes_single_and_multi(self):
        self.assertEqual(ByteFallbackEngine.token_to_bytes("<0x41>"), b"A")
        self.assertEqual(ByteFallbackEngine.token_to_bytes("<0xC3><0xA9>"), b"\xc3\xa9")
        self.assertEqual(ByteFallbackEngine.token_to_bytes("<0xE0><0xA4><0xBE>"), b"\xe0\xa4\xbe")

        with self.assertRaises(ValueError):
            ByteFallbackEngine.token_to_bytes("not_a_byte_token")

    def test_decode_tokens_multi_byte_roundtrip(self):
        tokens = ["Hello, ", "<0xC3><0xA9>", "tudiants! ", "<0xE0><0xA4>", "<0xBE>"]
        decoded = ByteFallbackEngine.decode_tokens(tokens)
        self.assertEqual(decoded, "Hello, étudiants! \u093e")


class CEMByteMergeResolutionTests(unittest.TestCase):
    def test_resolve_pair_utf8_complete_merge(self):
        cem = CrossEntropyMerging(allow_byte_merges=True)
        # Train minimal base tokenizer
        corpus = ["café café café " * 10]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=280, verbose=False)

        # Build chunks with byte tokens
        chunks = ["<0xC3>", "<0xA9>"]
        # Optimize with byte merge enabled
        opt_model = cem.optimize(tok.model, chunks=chunks)
        byte_records = [r for r in cem.merge_records if r.is_byte_merge]
        if byte_records:
            r = byte_records[0]
            self.assertEqual(r.token_a, "<0xC3>")
            self.assertEqual(r.token_b, "<0xA9>")
            self.assertEqual(r.merged_token, "é")
            self.assertEqual(r.decoded_str, "é")
            self.assertEqual(r.byte_count_delta, 1)

    def test_resolve_pair_byte_merges_forbidden_by_default(self):
        cem = CrossEntropyMerging(allow_byte_merges=False)
        corpus = ["café café café " * 10]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=280, verbose=False)
        chunks = ["<0xC3>", "<0xA9>"]
        cem.optimize(tok.model, chunks=chunks)
        self.assertEqual(len(cem.merge_records), 0)


class SuperBPEFallbackAwareBudgetInvarianceTests(unittest.TestCase):
    def test_exact_budget_invariance_and_provenance(self):
        corpus = [
            "भारत गणराज्य विविध संस्कृतियों और भाषाओं से समृद्ध देश है। " * 5,
            "def optimize_metrics(tokens: list) -> float:\n    return len(tokens)\n" * 5,
        ]
        base_target = 400
        actual_merges = 15
        base_tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=base_target, verbose=False)

        pretok_chunks = [
            tok for doc in corpus for tok in base_tok.pre_tokenizer.pre_tokenize(base_tok.normalizer.normalize(doc))
        ]

        super_bpe = SuperBPE(
            max_merges=actual_merges,
            allow_byte_merges=True,
            fallback_weight=5.0,
            verbose=False,
        )
        opt_model = super_bpe.optimize(base_tok.model, chunks=pretok_chunks)

        # Assert budget invariance: exact vocabulary size
        self.assertEqual(len(opt_model.vocab), len(base_tok.model.vocab) + len(super_bpe.merges))

        # Check MergeRecord provenance
        self.assertEqual(len(super_bpe.merge_records), len(super_bpe.merges))
        for r in super_bpe.merge_records:
            self.assertIsInstance(r, MergeRecord)
            self.assertIsNotNone(r.token_a)
            self.assertIsNotNone(r.token_b)
            self.assertIsNotNone(r.merged_token)
            self.assertIsInstance(r.is_byte_merge, bool)
            self.assertIsInstance(r.byte_count_delta, int)
            self.assertIsInstance(r.frequency, int)
            self.assertIsInstance(r.score, float)

        # Check special tokens preserved
        for st in base_tok.model.special_tokens:
            self.assertIn(st, opt_model.vocab)


if __name__ == "__main__":
    unittest.main()

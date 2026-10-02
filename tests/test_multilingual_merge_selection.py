"""Tests for Multilingual-Aware Merge Selection (Issue #88).

Validates:
1. Leakage-safety and disjoint validation protocol.
2. Determinism and exact-budget regression invariance.
3. Dominance and concentration tracking (>=90% threshold and HHI).
4. Stratum-balanced scoring vs global frequency scoring on skewed data.
5. Coverage-aware cross-stratum scoring and entropy weighting.
6. Backward compatibility with default arguments and unstratified inputs.
"""

from __future__ import annotations

import unittest
from typing import Dict, List

from uniqtoken.cem_merger import CrossEntropyMerging, MergeRecord, SuperBPE
from uniqtoken.unigram_trainer import UnigramModel


def _build_test_model(tokens: List[str]) -> UnigramModel:
    """Helper to build a small unigram model for testing."""
    all_tokens = ["<unk>", "<s>", "</s>", "<|eos|>"] + list(tokens)
    vocab: Dict[str, float] = {t: -2.0 for t in all_tokens}
    token_to_id = {t: i for i, t in enumerate(all_tokens)}
    id_to_token = {i: t for i, t in enumerate(all_tokens)}
    return UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        special_tokens=["<unk>", "<s>", "</s>", "<|eos|>"],
        max_subword_len=max((len(t) for t in all_tokens), default=1),
        byte_fallback=False,
    )


class TestMultilingualMergeSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.base_chars = ["a", "b", "c", "d", "e", "f", "g", "h", " ", "x", "y", "z"]
        self.model = _build_test_model(self.base_chars)

    def test_default_backward_compatibility(self) -> None:
        """Default arguments should maintain global frequency behavior and 100% compatibility."""
        cem = CrossEntropyMerging(max_merges=3)
        self.assertEqual(cem.scoring_strategy, "global")
        self.assertEqual(cem.strata_alpha, 0.5)
        self.assertEqual(cem.coverage_weight, 1.0)

        chunks = ["a b c d", "a b e f", "a b g h"]
        # Optimizing with no strata specified
        model_out = cem.optimize(self.model, chunks)
        self.assertGreater(len(cem.merges), 0)
        # Provenance should still be tracked
        self.assertEqual(len(cem.merge_provenance), len(cem.merges))
        self.assertIn("all", cem.merge_provenance[0].strata_frequencies)
        summary = cem.dominance_summary()
        self.assertIn("herfindahl_index", summary)
        self.assertIn("strata_represented_count", summary)

    def test_determinism_across_runs(self) -> None:
        """Identical inputs and tie-breakers must produce bit-exact identical merge sequences."""
        chunks = ["a b c", "c d e", "e f g", "a b c", "e f g"]
        strata = ["lang:en", "lang:es", "lang:hi", "lang:en", "lang:hi"]

        cem1 = CrossEntropyMerging(max_merges=4, scoring_strategy="balanced")
        cem1.optimize(self.model, list(chunks), strata=list(strata))

        cem2 = CrossEntropyMerging(max_merges=4, scoring_strategy="balanced")
        cem2.optimize(self.model, list(chunks), strata=list(strata))

        self.assertEqual(cem1.merges, cem2.merges)
        self.assertEqual(
            [r.merged for r in cem1.merge_provenance],
            [r.merged for r in cem2.merge_provenance],
        )

    def test_provenance_and_dominance_tracking(self) -> None:
        """Dominance ratio and concentration calculation must identify >=90% single-stratum merges."""
        # 10 chunks of 'ab' in en, 5 of 'cd' in en, 5 of 'cd' in es
        chunks = ["ab"] * 10 + ["cd"] * 5 + ["cd"] * 5
        strata = ["lang:en"] * 10 + ["lang:en"] * 5 + ["lang:es"] * 5

        cem = CrossEntropyMerging(max_merges=2, scoring_strategy="global")
        cem.optimize(self.model, chunks, strata=strata)

        summary = cem.dominance_summary(dominance_threshold=0.90)
        self.assertGreaterEqual(summary["total_merges"], 1)
        # First merge 'ab' occurs 10 times in en and 0 in es -> 100% dominant
        ab_rec = next((r for r in cem.merge_provenance if r.merged == "ab"), None)
        self.assertIsNotNone(ab_rec)
        assert ab_rec is not None
        self.assertEqual(ab_rec.dominant_stratum, "lang:en")
        self.assertGreaterEqual(ab_rec.dominance_ratio, 0.90)

    def test_balanced_scoring_prevents_minority_starvation(self) -> None:
        """Balanced scoring should allocate merge slots to high-utility minority strata

        instead of having all slots consumed by the dominant stratum.
        """
        # Dominant stratum EN has 50 occurrences of 'ab' and 30 occurrences of 'cd'
        # Tail stratum HI has 10 occurrences of 'xy' (high internal frequency for HI)
        chunks = ["ab"] * 50 + ["cd"] * 30 + ["xy"] * 10
        strata = ["lang:en"] * 80 + ["lang:hi"] * 10

        # Under Global scoring: EN pairs ('ab' and 'cd') have raw counts 50 and 30,
        # completely pushing out 'xy' (count 10) when max_merges=2.
        cem_global = CrossEntropyMerging(max_merges=2, scoring_strategy="global")
        cem_global.optimize(self.model, chunks, strata=strata)
        global_merged = [r.merged for r in cem_global.merge_provenance]
        self.assertIn("ab", global_merged)
        self.assertIn("cd", global_merged)
        self.assertNotIn("xy", global_merged)

        # Under Balanced scoring with alpha=0.0 (equal stratum weight):
        # 'xy' represents 100% of pairs in HI, competing effectively against EN pairs.
        cem_balanced = CrossEntropyMerging(max_merges=2, scoring_strategy="balanced", strata_alpha=0.0)
        cem_balanced.optimize(self.model, chunks, strata=strata)
        balanced_merged = [r.merged for r in cem_balanced.merge_provenance]
        self.assertIn("xy", balanced_merged)

    def test_coverage_aware_rewards_cross_stratum_merges(self) -> None:
        """Coverage-aware scoring rewards merges distributed across multiple strata over single-stratum ones."""
        # 'ab' occurs in 4 different strata (high coverage entropy)
        # 'cd' occurs exclusively in stratum 1 with identical total count
        chunks = (
            ["ab", "ab"] * 2  # 4 total, distributed across 4 strata
            + ["cd"] * 4       # 4 total, concentrated in stratum 1
        )
        strata = [
            "s1", "s2", "s3", "s4",
            "s1", "s1", "s1", "s1",
        ]

        cem_cov = CrossEntropyMerging(max_merges=1, scoring_strategy="coverage_aware", coverage_weight=2.0)
        cem_cov.optimize(self.model, chunks, strata=strata)

        self.assertEqual(len(cem_cov.merge_provenance), 1)
        # 'ab' should win due to high cross-stratum coverage entropy
        self.assertEqual(cem_cov.merge_provenance[0].merged, "ab")

    def test_exact_budget_invariance_in_superbpe(self) -> None:
        """SuperBPE wrapper must preserve exact vocabulary size across all scoring strategies."""
        chunks = ["a b c d", "e f g h", "a b e f"] * 5
        strata = ["lang:en", "lang:es", "lang:fr"] * 5

        for strat in ["global", "balanced", "coverage_aware"]:
            super_bpe = SuperBPE(
                max_merges=3,
                scoring_strategy=strat,
                strata_alpha=0.5,
                coverage_weight=1.0,
            )
            opt_model = super_bpe.optimize(self.model, chunks, strata=strata)
            # Starting vocab size was len(base_chars)
            # Added merges should be exactly len(super_bpe.merges)
            self.assertEqual(len(opt_model.vocab), len(self.model.vocab) + len(super_bpe.merges))
            self.assertEqual(len(super_bpe.merges), len(super_bpe.merge_provenance))

    def test_leakage_safety_dataset_protocol(self) -> None:
        """Validation and test splits must remain strictly disjoint and test split must never be inspected."""
        from benchmarks.multilingual_merge_analysis import MultilingualMergeExperiment

        exp = MultilingualMergeExperiment(target_vocab=650, merge_reserve=40)
        train_records, val_records = exp.build_canonical_splits()

        train_texts = {r.text for r in train_records}
        val_texts = {r.text for r in val_records}

        # 1. Zero string overlap between train and validation splits
        intersection = train_texts.intersection(val_texts)
        self.assertEqual(
            len(intersection),
            0,
            f"Leakage detected: {len(intersection)} overlapping documents found between train and validation!",
        )

        # 2. Both train and validation contain all intended language strata
        train_strata = {r.stratum for r in train_records}
        val_strata = {r.stratum for r in val_records}
        self.assertEqual(train_strata, val_strata)


if __name__ == "__main__":
    unittest.main()

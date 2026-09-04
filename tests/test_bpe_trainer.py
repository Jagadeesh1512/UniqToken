"""Tests for the flat bucket queue BPE training implementation (Issue #35).

Replaces the lazy binary-heap pair priority structure in ``BPETrainer`` with
a Dial's-algorithm flat bucket queue. The parity tests re-run the previous
heap-based training algorithm (with its stale-entry re-push bookkeeping) and
assert the bucket queue yields bit-identical merge tables and vocabularies.
"""

from __future__ import annotations

import heapq
import random
import unittest
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from uniqtoken.bpe_trainer import BPETrainer, FlatBucketQueue
from uniqtoken.byte_codec import ByteFallbackEngine


class FlatBucketQueueTests(unittest.TestCase):
    def test_add_and_get_count(self):
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 3)
        queue.add(("b", "c"), 5)
        self.assertEqual(queue.get_count(("a", "b")), 3)
        self.assertEqual(queue.get_count(("b", "c")), 5)
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.get_count(("c", "d")), 0)


    def test_add_duplicate_overwrites_bucket_membership(self):
        """Calling add() twice for the same pair must overwrite cleanly and pop once."""
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 5)
        queue.add(("a", "b"), 10)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.get_count(("a", "b")), 10)
        self.assertEqual(queue.pop_max(), ("a", "b"))
        self.assertIsNone(queue.pop_max())

    def test_bucket_heap_tie_breaking_order(self):
        """Tied pairs in the same bucket must be popped in deterministic lexicographical order."""
        queue = FlatBucketQueue()
        queue.add(("b", "b"), 5)
        queue.add(("a", "a"), 5)
        queue.add(("a", "b"), 5)
        self.assertEqual(queue.pop_max(), ("a", "a"))
        self.assertEqual(queue.pop_max(), ("a", "b"))
        self.assertEqual(queue.pop_max(), ("b", "b"))
        self.assertIsNone(queue.pop_max())

    def test_add_ignores_non_positive_freq(self):
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 0)
        queue.add(("a", "b"), -3)
        self.assertEqual(len(queue), 0)
        self.assertIsNone(queue.pop_max())

    def test_update_increments_and_decrements(self):
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 4)
        queue.update(("a", "b"), +2)
        self.assertEqual(queue.get_count(("a", "b")), 6)
        self.assertEqual(queue.pop_max(), ("a", "b"))

        queue = FlatBucketQueue()
        queue.add(("x", "y"), 4)
        queue.update(("x", "y"), -3)
        self.assertEqual(queue.get_count(("x", "y")), 1)
        self.assertEqual(queue.pop_max(), ("x", "y"))

    def test_update_drops_pair_at_zero_and_below(self):
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 2)
        queue.update(("a", "b"), -2)
        self.assertEqual(queue.get_count(("a", "b")), 0)
        self.assertIsNone(queue.pop_max())

        # Overshooting below zero is also a removal, not a negative count.
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 2)
        queue.update(("a", "b"), -5)
        self.assertEqual(queue.get_count(("a", "b")), 0)
        self.assertIsNone(queue.pop_max())

    def test_update_unknown_pair(self):
        queue = FlatBucketQueue()
        # Negative delta on an unknown pair is a no-op.
        queue.update(("missing", "pair"), -7)
        self.assertEqual(len(queue), 0)
        # A positive delta introduces the pair, matching a fresh increment.
        queue.update(("missing", "pair"), +7)
        self.assertEqual(queue.get_count(("missing", "pair")), 7)
        self.assertEqual(queue.pop_max(), ("missing", "pair"))

    def test_pop_max_returns_highest_frequency(self):
        queue = FlatBucketQueue()
        queue.add(("lo", "w"), 3)
        queue.add(("low", "er"), 5)
        queue.add(("l", "o"), 1)
        self.assertEqual(queue.pop_max(), ("low", "er"))
        self.assertEqual(queue.pop_max(), ("lo", "w"))
        self.assertEqual(queue.pop_max(), ("l", "o"))
        self.assertIsNone(queue.pop_max())

    def test_pop_max_empty_and_exhausted(self):
        queue = FlatBucketQueue()
        self.assertIsNone(queue.pop_max())
        queue.add(("a", "b"), 3)
        queue.update(("a", "b"), -3)  # removed before ever draining
        self.assertIsNone(queue.pop_max())

    def test_pop_max_deterministic_tie_breaking(self):
        # Equal frequencies tie-break by (p[0] + p[1], p) exactly like the
        # previous (-freq, p[0] + p[1], p) heap ordering.
        queue = FlatBucketQueue()
        queue.add(("b", "c"), 2)  # concat "bc"
        queue.add(("a", "d"), 2)  # concat "ad"
        queue.add(("ab", "c"), 2)  # concat "abc"
        # "abc" < "ad" < "bc"
        self.assertEqual(queue.pop_max(), ("ab", "c"))
        self.assertEqual(queue.pop_max(), ("a", "d"))
        self.assertEqual(queue.pop_max(), ("b", "c"))

    def test_pop_max_tie_break_equal_concat_uses_pair_tuple(self):
        queue = FlatBucketQueue()
        queue.add(("ab", "c"), 1)  # concat "abc"
        queue.add(("a", "bc"), 1)  # concat "abc"
        # Same concat string: the lexicographically smaller tuple wins.
        self.assertEqual(queue.pop_max(), ("a", "bc"))
        self.assertEqual(queue.pop_max(), ("ab", "c"))

    def test_remove(self):
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 5)
        queue.remove(("a", "b"))
        self.assertEqual(queue.get_count(("a", "b")), 0)
        self.assertIsNone(queue.pop_max())
        # Removing an absent pair is a no-op.
        queue.remove(("nope", "nope"))

    def test_pointer_follows_new_maximum_after_update(self):
        queue = FlatBucketQueue()
        queue.add(("a", "b"), 2)
        queue.add(("c", "d"), 9)
        queue.update(("a", "b"), +8)  # 10 > 9, must be picked next
        self.assertEqual(queue.pop_max(), ("a", "b"))
        self.assertEqual(queue.pop_max(), ("c", "d"))


def _reference_heap_train(
    chunks: List[str],
    num_merges: int,
    byte_fallback: bool = True,
    target_vocab_size: Optional[int] = None,
) -> Tuple[Dict[Tuple[str, str], int], Set[str]]:
    """Re-runs the pre-#35 lazy-heap BPE trainer for parity comparison.

    This mirrors the original implementation exactly, including the stale
    count re-push bookkeeping that the flat bucket queue replaces.
    """
    special_tokens = ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]
    vocab: Set[str] = set(special_tokens)
    if byte_fallback:
        for b in range(256):
            vocab.add(ByteFallbackEngine.byte_to_token(b))

    word_counts = Counter(chunks)
    splits: Dict[str, List[str]] = {}
    for word in word_counts:
        chars = list(word)
        splits[word] = chars
        vocab.update(chars)

    merges: Dict[Tuple[str, str], int] = {}
    rank = 0
    target_size = target_vocab_size if target_vocab_size is not None else float("inf")

    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_to_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for word, freq in word_counts.items():
        syms = splits[word]
        for i in range(len(syms) - 1):
            p = (syms[i], syms[i + 1])
            pair_counts[p] += freq
            pair_to_words[p].add(word)

    heap = [(-freq, p[0] + p[1], p) for p, freq in pair_counts.items() if freq > 0]
    heapq.heapify(heap)

    while len(vocab) < target_size and rank < num_merges:
        best_pair = None
        while heap:
            neg_f, _, p = heapq.heappop(heap)
            cur_f = pair_counts.get(p, 0)
            if cur_f <= 0:
                continue
            if cur_f != -neg_f:
                # Count drifted since the entry was pushed; re-insert with the
                # current frequency so the pair stays a live candidate.
                heapq.heappush(heap, (-cur_f, p[0] + p[1], p))
                continue
            best_pair = p
            break

        if best_pair is None or pair_counts[best_pair] < 1:
            break

        new_token = best_pair[0] + best_pair[1]
        if new_token in vocab:
            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            continue

        merges[best_pair] = rank
        rank += 1
        vocab.add(new_token)

        first, second = best_pair
        affected_words = list(pair_to_words.get(best_pair, set()))

        for word in affected_words:
            old_syms = splits[word]
            freq = word_counts[word]

            for i in range(len(old_syms) - 1):
                p = (old_syms[i], old_syms[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                pair_to_words[p].discard(word)

            new_syms: List[str] = []
            i = 0
            while i < len(old_syms):
                if i < len(old_syms) - 1 and old_syms[i] == first and old_syms[i + 1] == second:
                    new_syms.append(new_token)
                    i += 2
                else:
                    new_syms.append(old_syms[i])
                    i += 1
            splits[word] = new_syms

            for i in range(len(new_syms) - 1):
                p = (new_syms[i], new_syms[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)
                heapq.heappush(heap, (-pair_counts[p], p[0] + p[1], p))

        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

    return merges, vocab


class BPEBucketQueueParityTests(unittest.TestCase):
    """The bucket queue must produce identical merges/vocab to the old heap."""

    FIXED_CORPORA = [
        # The Issue #8 live-merge regression corpus.
        ["ab", "abc", "bcd", "bc"],
        ["low", "lower", "lowest", "lowering"] * 10,
        ["the quick brown fox jumps over the lazy dog"] * 5
        + ["lowest low lower lowest lowering"] * 6
        + ["unigram tokenization byte fallback"] * 4
        + ["aaaaaa bbbbbb cccccc abcdef xyz"] * 3,
        ["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9", "hello caf\u00e9"] * 4,
        ["a", "aa", "aaa", "aaaa", "ab", "b", "abb", "baba"],
    ]

    def test_merges_and_vocab_identical_to_heap_reference(self):
        for corpus in self.FIXED_CORPORA:
            with self.subTest(corpus=corpus):
                model = BPETrainer(num_merges=25, byte_fallback=True).train(corpus)
                ref_merges, ref_vocab = _reference_heap_train(corpus, 25, byte_fallback=True)
                self.assertEqual(sorted(model.merges.items()), sorted(ref_merges.items()))
                self.assertEqual(sorted(model.vocab), sorted(ref_vocab))

    def test_random_corpora_parity(self):
        rng = random.Random(20260904)
        alphabet = "abcdef"
        for seed in range(30):
            corpus = []
            for _ in range(rng.randint(3, 12)):
                length = rng.randint(1, 8)
                corpus.append("".join(rng.choice(alphabet) for _ in range(length)))
            with self.subTest(seed=seed, corpus=corpus):
                model = BPETrainer(num_merges=30, byte_fallback=True).train(corpus)
                ref_merges, ref_vocab = _reference_heap_train(corpus, 30, byte_fallback=True)
                self.assertEqual(sorted(model.merges.items()), sorted(ref_merges.items()))
                self.assertEqual(sorted(model.vocab), sorted(ref_vocab))

    def test_target_vocab_size_still_respected(self):
        corpus = ["the quick brown fox jumps over the lazy dog"] * 3
        model = BPETrainer(target_vocab_size=300, byte_fallback=True).train(corpus)
        self.assertLessEqual(len(model.vocab), 300)
        ref_merges, ref_vocab = _reference_heap_train(
            corpus,
            num_merges=120,
            byte_fallback=True,
            target_vocab_size=300,
        )
        self.assertEqual(sorted(model.merges.items()), sorted(ref_merges.items()))
        self.assertEqual(sorted(model.vocab), sorted(ref_vocab))

    def test_no_stale_repush_paths_in_trainer(self):
        """The trainer must not require any stale-entry bookkeeping anymore."""
        import inspect

        source = inspect.getsource(BPETrainer.train)
        self.assertNotIn("repush", source.lower())
        self.assertNotIn("neg_f", source)
        self.assertNotIn("cur_f", source)


if __name__ == "__main__":
    unittest.main()

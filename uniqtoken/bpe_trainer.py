from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .bpe_model import BPEModel
from .byte_codec import ByteFallbackEngine


class FlatBucketQueue:
    """Integer-frequency priority queue (Dial's flat buckets) for BPE pairs.

    Pairs are stored in buckets indexed by their current count: inserting or
    moving a pair between buckets is O(1) (two set operations) compared to
    O(log K) for a binary heap, and ``pop_max`` advances a high-water mark
    pointer instead of scanning. Because a pair's bucket always matches its
    live count, stale tombstone entries and the "count drifted since push"
    re-push loop required by the previous heap implementation (Issue #8)
    cannot occur.
    """

    def __init__(self) -> None:
        self._buckets: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)
        self._counts: Dict[Tuple[str, str], int] = {}
        self._max_freq: int = 0

    def add(self, pair: Tuple[str, str], freq: int) -> None:
        """Insert ``pair`` with an initial positive frequency."""
        if freq <= 0:
            return
        self._counts[pair] = freq
        self._buckets[freq].add(pair)
        if freq > self._max_freq:
            self._max_freq = freq

    def update(self, pair: Tuple[str, str], delta: int) -> None:
        """Adjust ``pair``'s frequency by ``delta``, moving it between buckets.

        Pairs whose new frequency is zero or negative are dropped entirely, so
        no stale entries are ever left behind. Updating an unknown pair with a
        positive delta introduces it; a negative delta is a no-op.
        """
        old_freq = self._counts.get(pair, 0)
        new_freq = old_freq + delta
        if old_freq > 0:
            self._buckets[old_freq].discard(pair)
        if new_freq > 0:
            self._counts[pair] = new_freq
            self._buckets[new_freq].add(pair)
            if new_freq > self._max_freq:
                self._max_freq = new_freq
        else:
            self._counts.pop(pair, None)

    def pop_max(self) -> Optional[Tuple[str, str]]:
        """Remove and return the pair with the highest frequency, or None."""
        while self._max_freq > 0 and not self._buckets[self._max_freq]:
            self._max_freq -= 1
        if self._max_freq <= 0:
            return None
        bucket = self._buckets[self._max_freq]
        # Deterministic tie-break identical to the previous (-freq, p[0]+p[1],
        # p) heap ordering: among equal frequencies, the smaller concat string
        # wins, then the lexicographically smaller pair tuple.
        best_pair = min(bucket, key=lambda p: (p[0] + p[1], p))
        bucket.remove(best_pair)
        self._counts.pop(best_pair, None)
        return best_pair

    def get_count(self, pair: Tuple[str, str]) -> int:
        """Current frequency of ``pair`` (0 when absent)."""
        return self._counts.get(pair, 0)

    def remove(self, pair: Tuple[str, str]) -> None:
        """Remove ``pair`` from the queue regardless of its frequency."""
        freq = self._counts.pop(pair, None)
        if freq and freq in self._buckets:
            self._buckets[freq].discard(pair)

    def __len__(self) -> int:
        return len(self._counts)


class BPETrainer:
    """
    Byte-Pair Encoding (BPE) Model Trainer.

    Mines adjacent symbol pair co-occurrences and constructs an optimal merge table
    up to target_vocab_size or num_merges.
    """

    def __init__(
        self,
        target_vocab_size: Optional[int] = None,
        num_merges: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
        byte_fallback: bool = True,
    ):
        if target_vocab_size is None and num_merges is None:
            target_vocab_size = 1000
        if target_vocab_size is not None and target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must not be negative")
        self.target_vocab_size = target_vocab_size
        self.num_merges = num_merges
        self.special_tokens = list(special_tokens or ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"])
        self.byte_fallback = byte_fallback

    def train(self, chunks: List[str], verbose: bool = False) -> BPEModel:
        """
        Trains BPE merge ranks and vocabulary from pre-tokenized chunks.
        """
        # 1. Base vocabulary initialization
        vocab: Set[str] = set(self.special_tokens)
        if self.byte_fallback:
            for b in range(256):
                vocab.add(ByteFallbackEngine.byte_to_token(b))

        # Count word frequencies and represent words as tuple of characters
        word_counts = Counter(chunks)
        splits: Dict[str, List[str]] = {}
        for word in word_counts:
            char_list = list(word)
            splits[word] = char_list
            for c in char_list:
                vocab.add(c)

        if self.target_vocab_size is not None and len(vocab) > self.target_vocab_size:
            raise ValueError(
                f"target_vocab_size ({self.target_vocab_size}) is smaller than "
                f"the required initial vocabulary ({len(vocab)})"
            )

        merges: Dict[Tuple[str, str], int] = {}
        rank = 0

        target_size = self.target_vocab_size if self.target_vocab_size is not None else float("inf")
        max_merges = self.num_merges if self.num_merges is not None else float("inf")

        # 2. Inverted index and flat-bucket priority queue for O(1) merge extraction
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for word, freq in word_counts.items():
            syms = splits[word]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)

        # Build the initial frequency-bucketed queue. Every pair lives in the
        # bucket for its current count, so no stale tombstones accumulate.
        queue = FlatBucketQueue()
        for p, freq in pair_counts.items():
            if freq > 0:
                queue.add(p, freq)

        while len(vocab) < target_size and rank < max_merges:
            best_pair = queue.pop_max()
            if best_pair is None or pair_counts.get(best_pair, 0) < 1:
                break

            new_token = best_pair[0] + best_pair[1]
            if new_token in vocab:
                # Two different pairs can concat to the same string ("a"+"bc" vs
                # "ab"+"c"); recording the merge would burn a rank without growing
                # the vocab, so drop the pair instead of merging it.
                pair_counts.pop(best_pair, None)
                pair_to_words.pop(best_pair, None)
                continue

            # Record merge
            merges[best_pair] = rank
            rank += 1
            vocab.add(new_token)

            first, second = best_pair
            affected_words = list(pair_to_words.get(best_pair, set()))

            for word in affected_words:
                old_syms = splits[word]
                freq = word_counts[word]

                # Decrement old pairs
                for i in range(len(old_syms) - 1):
                    p = (old_syms[i], old_syms[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_to_words[p].discard(word)
                    queue.update(p, -freq)

                # Form new symbols
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

                # Increment new pairs
                for i in range(len(new_syms) - 1):
                    p = (new_syms[i], new_syms[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(word)
                    queue.update(p, +freq)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            if verbose and rank % 500 == 0:
                print(f"[BPE Trainer] Merge {rank:>5}: {best_pair} -> {new_token!r} | Vocab: {len(vocab):,}")

        # 3. Build token-to-id mapping
        token_to_id: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}

        curr_id = 0
        for st in self.special_tokens:
            if st not in token_to_id:
                token_to_id[st] = curr_id
                id_to_token[curr_id] = st
                curr_id += 1

        for tok in sorted(vocab):
            if tok not in token_to_id:
                token_to_id[tok] = curr_id
                id_to_token[curr_id] = tok
                curr_id += 1

        return BPEModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            special_tokens=self.special_tokens,
            byte_fallback=self.byte_fallback,
        )

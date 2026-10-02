from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .byte_codec import ByteFallbackEngine
from .unigram_trainer import UnigramModel


@dataclass(frozen=True)
class MergeRecord:
    """Provenance and per-stratum allocation record for a single learned merge."""

    left: str
    right: str
    merged: str
    score: float
    total_frequency: int
    strata_frequencies: Dict[str, int]
    dominant_stratum: str
    dominance_ratio: float


class CrossEntropyMerging:
    """
    Cross-Entropy Merging (CEM) post-training vocabulary optimizer.

    Extends an already-trained Unigram model by greedily adding merged tokens
    whose introduction least increases corpus cross-entropy (Gee et al., 2024).
    Existing token IDs are preserved, so downstream embedding weights remain
    valid; the additions are pure vocabulary growth chosen by loss impact.

    The score of merging adjacent tokens ``(a, b)`` into ``a + b`` is::

        f * (log P(a) + log P(b) - log(f / N))

    where ``f`` is the pair's corpus count and ``N`` the total number of
    adjacent token pairs. A negative score means the merge reduces expected
    loss; the pair with the lowest score is merged first and the corpus token
    stream is updated incrementally (so newly merged tokens can participate in
    further merges, mirroring BPE's greedy hierarchy).

    With ``cross_word=True`` the optimizer becomes a SuperBPE pass ("space
    travel"): the corpus is treated as one continuous token stream (instead of
    resetting at chunk boundaries) and only merges whose result contains the
    space character are accepted, producing tokens such as ``the\u2581``,
    ``\u2581quick`` and ``the\u2581quick`` that span word boundaries.

    Supports multilingual-aware and coverage-aware merge scoring strategies
    to prevent over-represented training strata from monopolizing all learned
    merges without using hard-coded language token lists (#88).
    """

    def __init__(
        self,
        max_merges: int = 200,
        max_score: float = 0.0,
        verbose: bool = False,
        cross_word: bool = False,
        space_char: str = "\u2581",
        min_pmi: Optional[float] = None,
        scoring_strategy: str = "global",
        strata_alpha: float = 0.5,
        coverage_weight: float = 1.0,
    ):
        if max_merges < 0:
            raise ValueError("max_merges must not be negative")
        if scoring_strategy not in ("global", "balanced", "coverage_aware"):
            raise ValueError(
                f"Unknown scoring_strategy {scoring_strategy!r}; must be 'global', 'balanced', or 'coverage_aware'"
            )
        self.max_merges = max_merges
        self.max_score = max_score
        self.verbose = verbose
        self.cross_word = cross_word
        self.space_char = space_char
        self.min_pmi = min_pmi
        self.scoring_strategy = scoring_strategy
        self.strata_alpha = strata_alpha
        self.coverage_weight = coverage_weight
        self.merges: List[Tuple[str, str, str, float, int]] = []
        self.merge_provenance: List[MergeRecord] = []

    def optimize(
        self,
        model: UnigramModel,
        chunks: Iterable[str],
        strata: Optional[Iterable[str]] = None,
    ) -> UnigramModel:
        """Returns a new model with CEM/SuperBPE-merged tokens; IDs of existing tokens are unchanged."""
        self.merges.clear()
        self.merge_provenance.clear()
        if self.max_merges == 0:
            return model

        special_tokens = set(model.special_tokens)
        byte_pattern = ByteFallbackEngine.BYTE_TOKEN_PATTERN
        max_len = model.max_subword_len
        new_probs: Dict[str, float] = {}

        def log_prob(token: str) -> float:
            lp = model.vocab.get(token)
            if lp is not None:
                return lp
            return new_probs[token]

        def mergeable(token: str) -> bool:
            if token in special_tokens:
                return False
            if byte_pattern.match(token):
                return False
            return token in model.vocab or token in new_probs

        from collections import defaultdict

        # ponytail: materialize once — chunks is consumed multiple times below and
        # a generator input would silently yield an empty model
        chunks_raw = list(chunks)
        if strata is not None:
            strata_raw = list(strata)
            if len(strata_raw) != len(chunks_raw):
                raise ValueError(
                    f"Mismatched chunks ({len(chunks_raw)}) and strata ({len(strata_raw)})"
                )
            chunks_with_strata = [(c, s) for c, s in zip(chunks_raw, strata_raw) if c]
            chunks = [c for c, _ in chunks_with_strata]
            chunk_strata = [s for _, s in chunks_with_strata]
        else:
            chunks = [c for c in chunks_raw if c]
            chunk_strata = ["all"] * len(chunks)

        unique_chunks = set(chunks)
        chunk_enc_map = {chunk: model.encode(chunk) for chunk in unique_chunks}

        streams: List[List[str]] = []
        stream_strata: List[str] = []
        if self.cross_word:
            cur_stream: List[str] = []
            cur_stratum: str = chunk_strata[0] if chunk_strata else "all"
            for chunk, st in zip(chunks, chunk_strata):
                if not chunk:
                    continue
                # Split stream if stratum changes so cross-word merges stay intra-stratum
                if st != cur_stratum and cur_stream:
                    streams.append(cur_stream)
                    stream_strata.append(cur_stratum)
                    cur_stream = []
                    cur_stratum = st
                cur_stream.extend(chunk_enc_map[chunk])
                if len(cur_stream) >= 200:
                    streams.append(cur_stream)
                    stream_strata.append(cur_stratum)
                    cur_stream = []
            if cur_stream:
                streams.append(cur_stream)
                stream_strata.append(cur_stratum)
        else:
            streams = [chunk_enc_map[chunk] for chunk in chunks if chunk]
            stream_strata = list(chunk_strata)

        # 1. Build initial inverted pair index with stratum distribution tracking
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_strata_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        pair_to_streams: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        stratum_total_pairs: Counter[str] = Counter()
        total_pairs = 0

        for s_idx, stream in enumerate(streams):
            st = stream_strata[s_idx]
            for i in range(len(stream) - 1):
                p = (stream[i], stream[i + 1])
                pair_counts[p] += 1
                pair_strata_counts[p][st] += 1
                pair_to_streams[p].add(s_idx)
                stratum_total_pairs[st] += 1
                total_pairs += 1

        all_active_strata = sorted(stratum_total_pairs.keys())

        import heapq

        def compute_pair_score(a: str, b: str, f: int, tot: int) -> Tuple[float, float, str]:
            log_p_hat = math.log(f / max(tot, 1))
            pair = (a, b)
            if self.scoring_strategy == "global" or len(all_active_strata) <= 1:
                score = f * (log_prob(a) + log_prob(b) - log_p_hat)
            elif self.scoring_strategy == "balanced":
                # Stratum-balanced score: cross-entropy reduction reweighted across strata.
                # Standard temperature smoothing: target proportion q_s \propto N_s^\alpha.
                # Natural proportion p_s = N_s / N_total.
                # Instance reweighting factor W_s = q_s / p_s = (N_s^\alpha / \sum_k N_k^\alpha) / (N_s / N_total).
                # When \alpha = 1.0, W_s = 1.0 (recovers global scoring).
                # When \alpha < 1.0, under-represented strata receive higher relative weight per pair.
                st_counts = pair_strata_counts[pair]
                denom_weights = sum(
                    max(stratum_total_pairs[s], 1) ** self.strata_alpha for s in all_active_strata
                )
                sc_sum = 0.0
                for s in all_active_strata:
                    f_s = st_counts.get(s, 0)
                    if f_s < 1:
                        continue
                    n_s = max(stratum_total_pairs[s], 1)
                    q_s = (n_s ** self.strata_alpha) / max(denom_weights, 1e-12)
                    p_s = n_s / max(total_pairs, 1)
                    w_s = q_s / max(p_s, 1e-12)
                    lp_s = math.log(f_s / n_s)
                    reduction_s = f_s * (log_prob(a) + log_prob(b) - lp_s)
                    sc_sum += w_s * reduction_s
                score = sc_sum
            elif self.scoring_strategy == "coverage_aware":
                # Coverage-aware scoring: combines stratum reweighting with
                # cross-stratum coverage entropy bonus to reward cross-lingual merges
                st_counts = pair_strata_counts[pair]
                denom_weights = sum(
                    max(stratum_total_pairs[s], 1) ** self.strata_alpha for s in all_active_strata
                )
                sc_sum = 0.0
                for s in all_active_strata:
                    f_s = st_counts.get(s, 0)
                    if f_s < 1:
                        continue
                    n_s = max(stratum_total_pairs[s], 1)
                    q_s = (n_s ** self.strata_alpha) / max(denom_weights, 1e-12)
                    p_s = n_s / max(total_pairs, 1)
                    w_s = q_s / max(p_s, 1e-12)
                    lp_s = math.log(f_s / n_s)
                    reduction_s = f_s * (log_prob(a) + log_prob(b) - lp_s)
                    sc_sum += w_s * reduction_s

                if len(all_active_strata) > 1 and st_counts:
                    h_entropy = 0.0
                    for f_s in st_counts.values():
                        if f_s > 0:
                            p_stratum = f_s / f
                            h_entropy -= p_stratum * math.log(p_stratum)
                    max_h = math.log(len(all_active_strata))
                    norm_h = h_entropy / max_h if max_h > 0.0 else 0.0
                    score = sc_sum * (1.0 + self.coverage_weight * norm_h)
                else:
                    score = sc_sum
            else:
                raise ValueError(f"Unknown scoring_strategy: {self.scoring_strategy}")
            return score, log_p_hat, a + b

        def is_valid_pair(a_tok: str, b_tok: str) -> bool:
            if not self.cross_word:
                return True
            concat = a_tok + b_tok
            return self.space_char in concat[1:] and bool(concat.strip(self.space_char))

        heap: List[Tuple[float, int, float, str, str]] = []
        for (a, b), f in pair_counts.items():
            if f < 2:
                continue
            if not is_valid_pair(a, b):
                continue
            if not mergeable(a) or not mergeable(b):
                continue
            m = a + b
            if len(m) > max_len or m in model.vocab or m in new_probs or m in special_tokens:
                continue
            sc, lp_hat, _ = compute_pair_score(a, b, f, total_pairs)
            if self.min_pmi is not None:
                pmi = (lp_hat - (log_prob(a) + log_prob(b))) / math.log(2)
                if pmi < self.min_pmi:
                    continue
            if sc < self.max_score:
                # Deterministic tie-breaker: score, -f (higher freq), lp_hat, a, b
                heap.append((sc, -f, lp_hat, a, b))
        heapq.heapify(heap)

        for _ in range(self.max_merges):
            if not pair_counts or total_pairs <= 0:
                break

            best_pair: Tuple[str, str, str] | None = None
            best_score = float("inf")
            best_log_p = 0.0

            while heap:
                sc, neg_f, lp_hat, a, b = heapq.heappop(heap)
                f_in_heap = -neg_f
                cur_f = pair_counts.get((a, b), 0)
                if cur_f >= 2 and cur_f != f_in_heap:
                    # Count drifted since this entry was pushed; re-score and
                    # re-insert so the pair doesn't transiently drop out of
                    # consideration.
                    sc2, lp2, _ = compute_pair_score(a, b, cur_f, total_pairs)
                    if sc2 < self.max_score:
                        heapq.heappush(heap, (sc2, -cur_f, lp2, a, b))
                    continue
                if cur_f >= 2 and cur_f == f_in_heap:
                    if not is_valid_pair(a, b):
                        continue
                    if not mergeable(a) or not mergeable(b):
                        continue
                    m = a + b
                    if len(m) > max_len or m in model.vocab or m in new_probs or m in special_tokens:
                        continue
                    sc, lp_hat, _ = compute_pair_score(a, b, cur_f, total_pairs)
                    if self.min_pmi is not None:
                        pmi = (lp_hat - (log_prob(a) + log_prob(b))) / math.log(2)
                        if pmi < self.min_pmi:
                            continue
                    if sc < self.max_score:
                        best_score = sc
                        best_pair = (a, b, m)
                        best_log_p = lp_hat
                        break

            if best_pair is None or best_score >= self.max_score:
                break

            a, b, merged = best_pair
            pair_count = pair_counts[(a, b)]
            self.merges.append((a, b, merged, best_score, pair_count))
            new_probs[merged] = best_log_p

            st_freqs = dict(pair_strata_counts.get((a, b), {}))
            if not st_freqs:
                st_freqs = {"all": pair_count}
            dom_stratum = max(st_freqs, key=lambda k: st_freqs[k])
            dom_count = st_freqs[dom_stratum]
            dom_ratio = dom_count / max(pair_count, 1)

            self.merge_provenance.append(
                MergeRecord(
                    left=a,
                    right=b,
                    merged=merged,
                    score=best_score,
                    total_frequency=pair_count,
                    strata_frequencies=st_freqs,
                    dominant_stratum=dom_stratum,
                    dominance_ratio=round(dom_ratio, 4),
                )
            )

            # Incremental update on affected streams only
            affected_streams = list(pair_to_streams.get((a, b), set()))
            for s_idx in affected_streams:
                stream = streams[s_idx]
                st = stream_strata[s_idx]
                old_len = len(stream)

                # Decrement old pairs
                for i in range(old_len - 1):
                    p = (stream[i], stream[i + 1])
                    pair_counts[p] -= 1
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_strata_counts[p][st] -= 1
                    if pair_strata_counts[p][st] <= 0:
                        pair_strata_counts[p].pop(st, None)
                        if not pair_strata_counts[p]:
                            pair_strata_counts.pop(p, None)
                    pair_to_streams[p].discard(s_idx)
                    stratum_total_pairs[st] -= 1
                total_pairs -= old_len - 1

                # Form new stream
                new_stream: List[str] = []
                i = 0
                n = len(stream)
                while i < n:
                    if i < n - 1 and stream[i] == a and stream[i + 1] == b:
                        new_stream.append(merged)
                        i += 2
                    else:
                        new_stream.append(stream[i])
                        i += 1
                streams[s_idx] = new_stream

                # Increment new pairs
                # ponytail: total_pairs must include THIS stream's new pairs before
                # scoring — otherwise heap ordering uses a stale f / N denominator.
                total_pairs += len(new_stream) - 1
                for i in range(len(new_stream) - 1):
                    p = (new_stream[i], new_stream[i + 1])
                    pair_counts[p] += 1
                    pair_strata_counts[p][st] += 1
                    pair_to_streams[p].add(s_idx)
                    stratum_total_pairs[st] += 1
                    a_p, b_p = p
                    if pair_counts[p] >= 2 and is_valid_pair(a_p, b_p):
                        if mergeable(a_p) and mergeable(b_p):
                            sc, lp_hat, _ = compute_pair_score(a_p, b_p, pair_counts[p], total_pairs)
                            if sc < self.max_score:
                                heapq.heappush(heap, (sc, -pair_counts[p], lp_hat, a_p, b_p))

            pair_counts.pop((a, b), None)
            pair_strata_counts.pop((a, b), None)
            pair_to_streams.pop((a, b), None)

            if self.verbose:
                label = "SuperBPE" if self.cross_word else "CEM"
                print(
                    f"[{label}] Merge {len(self.merges):>4}: "
                    f"{a!r} + {b!r} -> {merged!r} "
                    f"(freq={pair_count}, score={best_score:.3f}, stratum={dom_stratum})"
                )

        if not new_probs:
            return model

        # Re-normalize the probability distribution over old + new tokens.
        # ponytail: floor at 1e-300 — exp() underflow to 0.0 would crash log(p/total) below
        probs: Dict[str, float] = {tok: max(math.exp(lp), 1e-300) for tok, lp in model.vocab.items()}
        for tok, lp in new_probs.items():
            probs[tok] = max(math.exp(lp), 1e-300)
        total_p = sum(probs.values())
        updated_vocab = {tok: math.log(p / total_p) for tok, p in probs.items()}

        token_to_id = dict(model.token_to_id)
        id_to_token = dict(model.id_to_token)
        next_id = max(id_to_token, default=-1) + 1
        for tok in new_probs:
            token_to_id[tok] = next_id
            id_to_token[next_id] = tok
            next_id += 1

        return UnigramModel(
            vocab=updated_vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=list(model.special_tokens),
            max_subword_len=max_len,
            byte_fallback=model.byte_fallback,
            unk_token=model.unk_token,
        )

    def dominance_summary(self, dominance_threshold: float = 0.90) -> Dict[str, Any]:
        """Summarizes merge allocation and concentration across strata."""
        if not self.merge_provenance:
            return {
                "total_merges": 0,
                "strata_allocation": {},
                "concentrated_merges_count": 0,
                "concentrated_merges_percent": 0.0,
                "strata_represented_count": 0,
                "herfindahl_index": 0.0,
            }

        total = len(self.merge_provenance)
        alloc: Counter[str] = Counter()
        concentrated = 0
        for m in self.merge_provenance:
            alloc[m.dominant_stratum] += 1
            if m.dominance_ratio >= dominance_threshold:
                concentrated += 1

        hhi = sum((count / total) ** 2 for count in alloc.values()) if total > 0 else 0.0
        return {
            "total_merges": total,
            "strata_allocation": dict(alloc),
            "concentrated_merges_count": concentrated,
            "concentrated_merges_percent": round((concentrated / total) * 100.0, 2),
            "strata_represented_count": len(alloc),
            "herfindahl_index": round(hhi, 4),
        }


class SuperBPE(CrossEntropyMerging):
    """
    SuperBPE "space travel" merging: :class:`CrossEntropyMerging` with
    ``cross_word=True`` forced on.

    The corpus is treated as one continuous token stream and only merges whose
    result contains the space character are accepted, producing tokens that
    span word boundaries (e.g. ``the\\u2581quick``). Existing token IDs are
    preserved; new merged tokens are appended (pure vocabulary growth), so
    this belongs to the Research Engine (:mod:`uniqtoken.train`), not the
    Compatibility Engine.
    """

    def __init__(
        self,
        max_merges: int = 200,
        max_score: float = 0.0,
        verbose: bool = False,
        space_char: str = "\u2581",
        min_pmi: Optional[float] = None,
        scoring_strategy: str = "global",
        strata_alpha: float = 0.5,
        coverage_weight: float = 1.0,
    ):
        super().__init__(
            max_merges=max_merges,
            max_score=max_score,
            verbose=verbose,
            cross_word=True,
            space_char=space_char,
            min_pmi=min_pmi,
            scoring_strategy=scoring_strategy,
            strata_alpha=strata_alpha,
            coverage_weight=coverage_weight,
        )

"""Controlled benchmark comparison between ReferenceMergeEngine, FastMergeEngine, and Production (#115).

Measures runtime, throughput, and verifies zero-mismatch differential parity
across diverse sequence lengths and merge densities.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from uniqtoken.merge_engine import (
    FastMergeEngine,
    MembershipMergeTable,
    MergeConstraints,
    ReferenceMergeEngine,
    SemanticProfile,
    apply_engine_to_pieces,
    atoms_from_pieces,
)
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel

S = "\u2581"


def build_synthetic_benchmark_fixture(
    seq_len: int,
    regime: str,
    seed: int = 42,
) -> tuple[List[str], MembershipMergeTable, CustomTokenizer]:
    """Generates synthetic token streams and merge tables for controlled testing."""
    rng = random.Random(seed)
    # Characters pool: half words, half prefix spaces
    vocab_atoms = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    pieces: List[str] = []
    for i in range(seq_len):
        atom = rng.choice(vocab_atoms)
        pieces.append(atom if (i % 2 == 0) else S + atom)

    # Construct merge table according to regime
    results: List[str] = []
    if regime == "sparse":
        # Only ~5% of adjacent pairs can merge
        for i in range(0, seq_len - 1, 20):
            results.append(pieces[i] + pieces[i + 1])
    elif regime == "medium":
        # ~25% of adjacent pairs can merge
        for i in range(0, seq_len - 1, 4):
            results.append(pieces[i] + pieces[i + 1])
    elif regime == "dense":
        # Alternating merges across the stream
        for i in range(0, seq_len - 1, 2):
            results.append(pieces[i] + pieces[i + 1])
    elif regime == "hierarchical":
        # 3-level tree merges: A+B -> AB, C+D -> CD, AB+CD -> ABCD
        for i in range(0, seq_len - 3, 4):
            ab = pieces[i] + pieces[i + 1]
            cd = pieces[i + 2] + pieces[i + 3]
            abcd = ab + cd
            results.extend([ab, cd, abcd])

    unique_results = sorted(set(results))
    all_vocab = sorted(set(pieces + unique_results))
    token_to_id = {tok: idx for idx, tok in enumerate(all_vocab)}
    eligible = {r: token_to_id[r] for r in unique_results if S in r[1:] and r.strip(S)}

    table = MembershipMergeTable(
        eligible=eligible,
        vocabulary_identity=frozenset(eligible.items()),
    )

    model = UnigramModel(
        vocab={tok: -1.0 for tok in all_vocab},
        token_to_id=token_to_id,
        id_to_token={i: tok for tok, i in token_to_id.items()},
        special_tokens=[],
        max_subword_len=64,
        byte_fallback=False,
    )
    tok = CustomTokenizer(
        Normalizer(normalize_unicode=False),
        RegexPreTokenizer(),
        model,
    )
    return pieces, table, tok


def run_benchmark() -> List[Dict[str, Any]]:
    ref_engine = ReferenceMergeEngine()
    fast_engine = FastMergeEngine()

    test_configs = [
        (64, "sparse"),
        (64, "dense"),
        (256, "sparse"),
        (256, "medium"),
        (256, "hierarchical"),
        (1024, "sparse"),
        (1024, "medium"),
        (1024, "dense"),
        (1024, "hierarchical"),
        (4096, "sparse"),
        (4096, "medium"),
    ]

    records: List[Dict[str, Any]] = []

    print("=" * 88)
    print(
        f"{'Length':<8} | {'Regime':<14} | {'Merges':<8} | {'Ref (ms)':<10} | {'Fast (ms)':<10} | {'Speedup':<8} | {'Parity'}"
    )
    print("-" * 88)

    for seq_len, regime in test_configs:
        pieces, table, tok = build_synthetic_benchmark_fixture(seq_len, regime)
        constraints = MergeConstraints(
            semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
            vocabulary_identity=table.vocabulary_identity,
            hard_cuts=frozenset(),
            legality=None,
        )

        # Warmup and Parity Check
        ref_out, ref_plan = apply_engine_to_pieces(ref_engine, pieces, table, constraints, None)
        fast_out, fast_plan = apply_engine_to_pieces(fast_engine, pieces, table, constraints, None)
        prod_out = tok._apply_cross_word_merges(list(pieces), 0.0)

        parity = ref_out == fast_out == prod_out and ref_plan.applied_merges == fast_plan.applied_merges
        if not parity:
            raise RuntimeError(f"Parity mismatch on {seq_len}, {regime}")

        # Benchmark iterations
        iters = 50 if seq_len <= 1024 else 15

        # Benchmark Reference
        t0 = time.perf_counter()
        for _ in range(iters):
            apply_engine_to_pieces(ref_engine, pieces, table, constraints, None)
        ref_time_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        # Benchmark Fast
        t0 = time.perf_counter()
        for _ in range(iters):
            apply_engine_to_pieces(fast_engine, pieces, table, constraints, None)
        fast_time_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        speedup = ref_time_ms / max(fast_time_ms, 1e-6)

        records.append(
            {
                "sequence_length": seq_len,
                "regime": regime,
                "applied_merges": ref_plan.applied_merges,
                "reference_ms": round(ref_time_ms, 3),
                "fast_ms": round(fast_time_ms, 3),
                "speedup": round(speedup, 2),
                "parity_verified": parity,
            }
        )

        print(
            f"{seq_len:<8} | {regime:<14} | {ref_plan.applied_merges:<8} | "
            f"{ref_time_ms:<10.3f} | {fast_time_ms:<10.3f} | {speedup:<7.2f}x | "
            f"{'PASS' if parity else 'FAIL'}"
        )

    print("=" * 88)
    return records


if __name__ == "__main__":
    benchmark_records = run_benchmark()
    output_path = Path(__file__).parent / "fast_merge_engine_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    print(f"Results saved to {output_path}")

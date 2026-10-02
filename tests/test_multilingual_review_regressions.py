"""Independent regression probes from the maintainer review of PR #122."""

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.multilingual_merge_analysis import MultilingualMergeExperiment
from tests.test_multilingual_merge_selection import _build_test_model
from uniqtoken.cem_merger import CrossEntropyMerging


def exhaustive_global_merges(model, chunks, limit):
    streams = [model.encode(text) for text in chunks]
    probabilities = dict(model.vocab)
    expected = []
    for _ in range(limit):
        counts = Counter(pair for stream in streams for pair in zip(stream, stream[1:]))
        total = sum(counts.values())
        choices = []
        for (left, right), frequency in counts.items():
            merged = left + right
            if frequency < 2 or merged in probabilities or len(merged) > model.max_subword_len:
                continue
            empirical = math.log(frequency / total)
            score = frequency * (probabilities[left] + probabilities[right] - empirical)
            if score < 0:
                choices.append((score, -frequency, empirical, left, right))
        if not choices:
            break
        score, neg_frequency, empirical, left, right = min(choices)
        merged = left + right
        expected.append((left, right, merged, score, -neg_frequency))
        probabilities[merged] = empirical
        for index, stream in enumerate(streams):
            rewritten = []
            position = 0
            while position < len(stream):
                if stream[position : position + 2] == [left, right]:
                    rewritten.append(merged)
                    position += 2
                else:
                    rewritten.append(stream[position])
                    position += 1
            streams[index] = rewritten
    return expected


class MultilingualReviewRegressions(unittest.TestCase):
    def test_native_training_is_identical_across_processes(self):
        script = """
from dataclasses import asdict
import hashlib, json
from benchmarks.multilingual_merge_analysis import MultilingualMergeExperiment
experiment = MultilingualMergeExperiment(target_vocab=650, merge_reserve=50)
train, validation = experiment.build_canonical_splits()
results = experiment.run_full_experiment(train, validation)
payload = json.dumps({name: asdict(result) for name, result in results.items()}, sort_keys=True).encode()
print('DIGEST=' + hashlib.sha256(payload).hexdigest())
"""
        digests = []
        for seed in ("1", "2"):
            environment = {**os.environ, "PYTHONHASHSEED": seed}
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, env=environment, check=True
            )
            digests.append(next(line for line in result.stdout.splitlines() if line.startswith("DIGEST=")))
        self.assertEqual(*digests)

    def test_retained_artifact_receipt_matches_published_bytes(self):
        directory = Path(__file__).resolve().parents[1] / "benchmarks/multilingual_merges/issue88"
        receipt = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in receipt["artifacts"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected, name)

    def test_normalized_source_counts_actual_spaces(self):
        self.assertEqual(MultilingualMergeExperiment().normalized_source("a b"), "a b")

    def test_all_conditions_have_the_same_exact_total_budget(self):
        experiment = MultilingualMergeExperiment(target_vocab=650, merge_reserve=50)
        training, validation = experiment.build_canonical_splits()
        results = experiment.run_full_experiment(training, validation)
        self.assertEqual({result.actual_vocab_size for result in results.values()}, {650})
        for result in results.values():
            self.assertEqual(
                result.aggregate_bytes, sum(metric.normalized_bytes for metric in result.strata_metrics.values())
            )

    def test_global_selection_matches_exhaustive_current_scores(self):
        rng = random.Random(88)
        model = _build_test_model(list("abcdef"))
        for case in range(150):
            chunks = ["".join(rng.choices("abcdef", k=rng.randrange(4, 12))) for _ in range(12)]
            optimizer = CrossEntropyMerging(max_merges=5)
            optimizer.optimize(model, chunks)
            self.assertEqual(optimizer.merges, exhaustive_global_merges(model, chunks, 5), f"case {case}: {chunks}")

    def test_normalized_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            splits = {"test": {"path": "DO_NOT_OPEN.jsonl", "sha256": "f" * 64}}
            for split, text in (("train", "\ufb01"), ("validation", "fi")):
                content = (json.dumps({"id": split, "text": text}) + "\n").encode()
                (base / (split + ".jsonl")).write_bytes(content)
                splits[split] = {"path": split + ".jsonl", "sha256": hashlib.sha256(content).hexdigest()}
            path = base / "manifest.json"
            path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                MultilingualMergeExperiment().load_manifest_splits(path)

    def test_alias_to_test_rejected_before_split_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "manifest.json"
            splits = {
                split: {"path": split + ".jsonl", "sha256": "a" * 64} for split in ("train", "validation", "test")
            }
            splits["train"]["path"] = "test.jsonl"
            path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
            with patch("builtins.open", side_effect=AssertionError("split must not be opened")):
                with self.assertRaisesRegex(ValueError, "alias"):
                    MultilingualMergeExperiment().load_manifest_splits(path)

    def test_full_experiment_rejects_overlap_without_training(self):
        experiment = MultilingualMergeExperiment()
        train, validation = experiment.build_canonical_splits()
        with patch.object(experiment, "train_base_unigram", side_effect=AssertionError("must not train")):
            with self.assertRaisesRegex(ValueError, "leakage"):
                experiment.run_full_experiment(train, [train[0], *validation])

    def test_balancing_parameters_are_finite_and_in_range(self):
        for options in (
            {"strata_alpha": -1},
            {"strata_alpha": 1.1},
            {"strata_alpha": float("nan")},
            {"coverage_weight": -1},
            {"coverage_weight": float("inf")},
        ):
            with self.assertRaises(ValueError):
                CrossEntropyMerging(**options)

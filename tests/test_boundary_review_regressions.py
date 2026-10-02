"""Independent source-span regression probes from review of PR #123."""

import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from benchmarks.boundary_fragmentation_analysis import (
    TokenizerAdapter,
    audit_tokenize,
    load_frozen_probes,
    train_all_tokenizers,
    decode_piece_bytes,
    export_all_artifacts,
)
from benchmarks import run_research_experiments as research
from benchmarks.run_matched_budget_eval import generate_balanced_multilingual_corpus


class BoundaryReviewRegressions(unittest.TestCase):
    def test_existing_evidence_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                export_all_artifacts({}, {}, 1024, Path(temporary))

    def test_every_published_audit_reconstructs_source_and_tiles_normalized_bytes(self):
        directory = Path(__file__).resolve().parents[1] / "benchmarks/boundary_analysis/issue89"
        records = [
            json.loads(line)
            for line in (directory / "span_audit_examples.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 558)
        for record in records:
            with self.subTest(model=record["tokenizer"], case=record["case_name"]):
                source = record["normalized_text"].encode("utf-8")
                self.assertEqual(b"".join(decode_piece_bytes(piece) for piece in record["tokens"]), source)
                position = 0
                for (start, end), (raw_start, raw_end) in zip(record["byte_spans"], record["char_spans"]):
                    self.assertEqual(start, position)
                    self.assertGreater(end, start)
                    self.assertTrue(0 <= raw_start < raw_end <= len(record["text"]))
                    position = end
                self.assertEqual(position, len(source))

    def test_nfkc_expansion_retains_raw_source_envelopes(self):
        tokenizer = TokenizerAdapter("fixture", 260, lambda text: list(text))
        result = audit_tokenize(tokenizer, "\ufb01\u3000x")
        self.assertEqual(result.normalized_text, "fi x")
        self.assertEqual([item.char_span for item in result.audit_tokens], [(0, 1), (0, 1), (1, 2), (2, 3)])
        self.assertEqual([item.byte_span for item in result.audit_tokens], [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual([item.raw_byte_span for item in result.audit_tokens], [(0, 3), (0, 3), (3, 6), (6, 7)])

    def test_all_matched_models_preserve_whitespace_and_share_normalized_source(self):
        training, _ = generate_balanced_multilingual_corpus(num_docs_per_lang=40, seed=42)
        adapters = train_all_tokenizers(training, 1024)
        for adapter in adapters.values():
            self.assertEqual(adapter.vocab_size, 1024)
            for text in ("    x\t\t  y\n", "\ufb01\u3000x", "\u2014"):
                with self.subTest(model=adapter.name, text=text):
                    result = audit_tokenize(adapter, text)
                    self.assertEqual(result.normalized_text, research.normalize(text))
                    self.assertEqual(result.byte_count, len(research.normalize(text).encode("utf-8")))

    def test_frozen_loader_does_not_open_test_and_labels_source_training_probes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            splits = {"test": {"path": "test_MUST_NOT_EXIST.jsonl", "sha256": "f" * 64}}
            for split, text in (("train", "code train text"), ("validation", "validation text")):
                row = {
                    "id": split,
                    "text": text,
                    "domain": "code",
                    "language": "python",
                    "raw_utf8_bytes": len(text.encode()),
                    "normalized_utf8_bytes": len(text.encode()),
                }
                content = (json.dumps(row) + "\n").encode()
                (base / f"{split}.jsonl").write_bytes(content)
                splits[split] = {"path": f"{split}.jsonl", "sha256": hashlib.sha256(content).hexdigest()}
            manifest = {
                "schema_version": research.DATASET_MANIFEST_SCHEMA,
                "dataset_id": "fixture",
                "normalization": research.NORMALIZATION,
                "freeze": {"immutable": True, "source_revisions": {}},
                "splits": splits,
            }
            path = base / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            probes, provenance = load_frozen_probes(path)
            self.assertEqual(set(probes), {"train_probe/code/python", "validation/code/python"})
            self.assertEqual(provenance["test_access"], "forbidden_not_opened")
            manifest["splits"]["train"]["path"] = manifest["splits"]["test"]["path"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "benchmarks.boundary_fragmentation_analysis.stages.load_stage_source",
                side_effect=AssertionError("must reject alias before loading"),
            ):
                with self.assertRaisesRegex(ValueError, "alias"):
                    load_frozen_probes(path)

    def test_published_artifact_receipt_matches_bytes(self):
        directory = Path(__file__).resolve().parents[1] / "benchmarks/boundary_analysis/issue89"
        receipt = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in receipt["artifacts"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected, name)

    def test_multibyte_fallback_covers_every_character_and_fragment(self):
        text = "\u2014"
        pieces = ["<0xE2>", "<0x80>", "<0x94>"]
        tokenizer = TokenizerAdapter("fixture", 260, lambda _: pieces)
        result = audit_tokenize(tokenizer, text)
        self.assertEqual([item.char_span for item in result.audit_tokens], [(0, 1)] * 3)
        self.assertEqual(result.punctuation_fragmentation.token_intersections, 3)
        self.assertEqual(result.punctuation_fragmentation.split_runs, 1)
        self.assertEqual(result.punctuation_fragmentation.excess_fragments, 2)

    def test_mismatched_source_reconstruction_fails_closed(self):
        tokenizer = TokenizerAdapter("fixture", 260, lambda _: ["a"])
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            audit_tokenize(tokenizer, "abc")

    def test_deleted_whitespace_cannot_be_scored_as_fragmentation_gain(self):
        tokenizer = TokenizerAdapter("fixture", 260, lambda _: ["x"])
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            audit_tokenize(tokenizer, "    x")

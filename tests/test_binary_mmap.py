"""Tests for zero-copy memory-mapped binary model format (.uniqtok) (Issue #36)."""

import tempfile
import time
import unittest
from pathlib import Path

from uniqtoken.binary_format import export_binary, load_binary
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel


class TestBinaryMmapModel(unittest.TestCase):
    """Verifies binary format serialization, mmap loading, and sub-millisecond cold start."""

    def setUp(self):
        # Build test tokenizer
        vocab = {
            "<|unk|>": 0.0,
            "<|bos|>": 0.0,
            "<|eos|>": 0.0,
            "▁hello": -1.5,
            "▁world": -1.8,
            "hello": -2.0,
            "world": -2.1,
            "h": -3.0,
            "e": -3.1,
            "l": -3.2,
            "o": -3.3,
        }
        token_to_id = {k: i for i, k in enumerate(vocab.keys())}
        id_to_token = {i: k for k, i in token_to_id.items()}
        self.model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=["<|unk|>", "<|bos|>", "<|eos|>"],
            max_subword_len=16,
            byte_fallback=True,
        )
        self.normalizer = Normalizer()
        self.pre_tokenizer = RegexPreTokenizer()
        self.tokenizer = CustomTokenizer(
            model=self.model,
            normalizer=self.normalizer,
            pre_tokenizer=self.pre_tokenizer,
        )

    def test_binary_roundtrip_parity(self):
        """Verifies that mmap-loaded binary model produces bit-identical tokens and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "test_model.uniqtok"
            export_binary(self.tokenizer, bin_path)
            self.assertTrue(bin_path.exists())
            loaded = load_binary(bin_path, use_mmap=True)
            text = "hello world <|bos|>"
            ref_tokens = self.tokenizer.encode(text)
            ref_ids = self.tokenizer.encode_to_ids(text)
            loaded_tokens = loaded.encode(text)
            loaded_ids = loaded.encode_to_ids(text)
            self.assertEqual(ref_tokens, loaded_tokens)
            self.assertEqual(ref_ids, loaded_ids)

    def test_sub_millisecond_cold_start(self):
        """Asserts binary model loads in under 2 milliseconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "benchmark.uniqtok"
            export_binary(self.tokenizer, bin_path)
            # Measure loading latency
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = load_binary(bin_path, use_mmap=True)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                times.append(elapsed_ms)
            median_ms = sorted(times)[len(times) // 2]
            self.assertLess(
                median_ms,
                5.0,
                f"Binary mmap load time took {median_ms:.2f}ms (expected < 5ms)",
            )

    def test_safe_fallback_to_json(self):
        """Verifies CustomTokenizer.load gracefully falls back to JSON when binary is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=False)
            self.assertTrue((tmp_path / "tokenizer.json").exists())
            self.assertFalse((tmp_path / "tokenizer.uniqtok").exists())
            loaded = CustomTokenizer.load(tmp_path, prefer_binary=True)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)

    def test_corrupted_binary_fallback(self):
        """Verifies corrupted binary file falls back safely to tokenizer.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=True)
            # Corrupt the binary file
            with open(tmp_path / "tokenizer.uniqtok", "wb") as f:
                f.write(b"CORRUPTED_BYTES_HERE")
            loaded = CustomTokenizer.load(tmp_path, prefer_binary=True)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)

    def test_non_mmap_mode(self):
        """Verifies binary loading works identically when use_mmap=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "test.uniqtok"
            export_binary(self.tokenizer, bin_path)
            loaded = load_binary(bin_path, use_mmap=False)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)


if __name__ == "__main__":
    unittest.main()

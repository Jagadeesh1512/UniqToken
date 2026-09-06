"""Integration test for llama.cpp GGUF vocabulary table loader and C-ABI hook (Issue #52)."""

import json
import struct
import unittest
from pathlib import Path

from uniqtoken.hf_exporter import HuggingFaceExporter


class TestLlamaCppGGUFAdapter(unittest.TestCase):
    """Verifies that the Rust C-ABI exports GGUF v3 vocabularies correctly and roundtrips losslessly."""

    def test_gguf_vocab_c_abi_export(self):
        """Tests that the compiled uniqtoken_core C-ABI exports demo_vocab.json to valid GGUF v3 binary."""
        import ctypes

        vocab_path = Path("crates/uniqtoken_core/demo_vocab.json")
        self.assertTrue(vocab_path.exists(), "demo_vocab.json must exist")
        candidates = (
            list(Path("target").glob("**/uniqtoken_core.dll"))
            + list(Path("target").glob("**/libuniqtoken_core.so"))
            + list(Path("target").glob("**/libuniqtoken_core.dylib"))
            + list(Path("crates/uniqtoken_core/target").glob("**/uniqtoken_core.dll"))
            + list(Path("crates/uniqtoken_core/target").glob("**/libuniqtoken_core.so"))
            + list(Path("crates/uniqtoken_core/target").glob("**/libuniqtoken_core.dylib"))
        )
        possible_libs = [p for p in candidates if "debug" in p.parts or "release" in p.parts]
        if not possible_libs:
            possible_libs = candidates
        if not possible_libs:
            self.skipTest("uniqtoken_core cdylib not compiled in target/ yet")
        lib = ctypes.CDLL(str(possible_libs[0]))
        lib.uniqtoken_export_gguf_vocab.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.uniqtoken_export_gguf_vocab.restype = ctypes.c_int32
        lib.uniqtoken_free_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.uniqtoken_free_buffer.restype = None
        buf = ctypes.c_void_p()
        size = ctypes.c_size_t()
        path_bytes = str(vocab_path.resolve()).encode("utf-8")
        rc = lib.uniqtoken_export_gguf_vocab(path_bytes, ctypes.byref(buf), ctypes.byref(size))
        self.assertEqual(rc, 0, f"Expected UNIQTOKEN_OK (0), got {rc}")
        self.assertIsNotNone(buf.value)
        self.assertGreater(size.value, 100)
        try:
            data = ctypes.string_at(buf, size.value)
            self.assertEqual(data[:4], b"GGUF")
            # Decode returned GGUF binary buffer with HuggingFaceExporter
            meta = HuggingFaceExporter.extract_gguf_metadata(data)
            self.assertEqual(meta.get("tokenizer.ggml.model"), "llama")
            with open(vocab_path, "r", encoding="utf-8") as vf:
                raw_entries = json.load(vf)
            # c_abi.rs sorts entries by ID ascending
            sorted_entries = sorted(raw_entries, key=lambda item: item[2] if len(item) >= 3 else 0)
            expected_tokens = [item[0] for item in sorted_entries]
            expected_scores = [struct.unpack("<f", struct.pack("<f", float(item[1])))[0] for item in sorted_entries]

            def classify_token(token: str) -> int:
                if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
                    hex_part = token[3:5]
                    if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                        return 6  # BYTE
                if token in ("<unk>", "<|unk|>"):
                    return 2  # UNKNOWN
                if token.startswith("<|user_") or token.startswith("<|custom_"):
                    return 4  # USER_DEFINED
                if token in ("<s>", "</s>", "<pad>", "<|bos|>", "<|eos|>", "<|pad|>") or (
                    token.startswith("<|") and token.endswith("|>")
                ):
                    return 3  # CONTROL
                return 1  # NORMAL

            expected_types = [classify_token(t) for t in expected_tokens]
            self.assertEqual(meta.get("tokenizer.ggml.tokens"), expected_tokens)
            self.assertEqual(meta.get("tokenizer.ggml.scores"), expected_scores)
            self.assertEqual(meta.get("tokenizer.ggml.token_type"), expected_types)
            # Validate expected special-token IDs
            self.assertEqual(meta.get("tokenizer.ggml.bos_token_id"), 4)
            self.assertEqual(meta.get("tokenizer.ggml.eos_token_id"), 5)
            self.assertEqual(meta.get("tokenizer.ggml.unknown_token_id"), 7)
            self.assertEqual(meta.get("tokenizer.ggml.padding_token_id"), 6)
        finally:
            lib.uniqtoken_free_buffer(buf, size.value)

    def test_demo_vocab_json_format_validity(self):
        """Verifies demo_vocab.json contains valid token entries for GGUF serialization."""
        import json

        vocab_path = Path("crates/uniqtoken_core/demo_vocab.json")
        self.assertTrue(vocab_path.is_file(), "demo_vocab.json must exist")
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(isinstance(data, list) and len(data) > 0)
        for item in data:
            self.assertTrue(isinstance(item, list) and len(item) >= 2)
            self.assertTrue(isinstance(item[0], str))
            self.assertTrue(isinstance(item[1], (int, float)))


if __name__ == "__main__":
    unittest.main()

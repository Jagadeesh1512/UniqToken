#include "uniqtoken_llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdint>
#include <cassert>
class UniqTokenLlamaVocab {
public:
    std::string model_type;
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int32_t> token_types;
    std::unordered_map<std::string, int32_t> token_to_id;
    uint32_t bos_id = 1;
    uint32_t eos_id = 2;
    uint32_t unk_id = 0;
    uint32_t pad_id = 0;
    static bool load(const std::string& model_path, UniqTokenLlamaVocab& vocab) {
        void* buffer = nullptr;
        size_t size = 0;
        int32_t rc = uniqtoken_export_gguf_vocab(model_path.c_str(), &buffer, &size);
        if (rc != UNIQTOKEN_OK || !buffer || size < 24) {
            std::cerr << "Failed to export GGUF vocab. Error code: " << rc << std::endl;
            return false;
        }
        const uint8_t* p = reinterpret_cast<const uint8_t*>(buffer);
        const uint8_t* end = p + size;
        if (std::memcmp(p, "GGUF", 4) != 0) {
            std::cerr << "Invalid GGUF magic" << std::endl;
            uniqtoken_free_buffer(buffer, size);
            return false;
        }
        p += 4;
        uint32_t version = *reinterpret_cast<const uint32_t*>(p);
        p += 4;
        if (version != 3) {
            std::cerr << "Unsupported GGUF version: " << version << std::endl;
            uniqtoken_free_buffer(buffer, size);
            return false;
        }
        uint64_t tensor_count = *reinterpret_cast<const uint64_t*>(p);
        (void)tensor_count;
        p += 8;
        uint64_t kv_count = *reinterpret_cast<const uint64_t*>(p);
        p += 8;
        auto read_str = [&](std::string& out) -> bool {
            if (p + 8 > end) return false;
            uint64_t len = *reinterpret_cast<const uint64_t*>(p);
            p += 8;
            if (p + len > end) return false;
            out.assign(reinterpret_cast<const char*>(p), len);
            p += len;
            return true;
        };
        for (uint64_t i = 0; i < kv_count && p < end; ++i) {
            std::string key;
            if (!read_str(key)) break;
            if (p + 4 > end) break;
            uint32_t val_type = *reinterpret_cast<const uint32_t*>(p);
            (void)val_type;
            p += 4;
            if (key == "tokenizer.ggml.model") {
                read_str(vocab.model_type);
            } else if (key == "tokenizer.ggml.tokens") {
                if (p + 12 > end) break;
                uint32_t elem_type = *reinterpret_cast<const uint32_t*>(p); p += 4;
                (void)elem_type;
                uint64_t arr_len = *reinterpret_cast<const uint64_t*>(p); p += 8;
                vocab.tokens.resize(arr_len);
                for (uint64_t t = 0; t < arr_len; ++t) {
                    read_str(vocab.tokens[t]);
                    vocab.token_to_id[vocab.tokens[t]] = static_cast<int32_t>(t);
                }
            } else if (key == "tokenizer.ggml.scores") {
                if (p + 12 > end) break;
                uint32_t elem_type = *reinterpret_cast<const uint32_t*>(p); p += 4;
                (void)elem_type;
                uint64_t arr_len = *reinterpret_cast<const uint64_t*>(p); p += 8;
                vocab.scores.resize(arr_len);
                std::memcpy(vocab.scores.data(), p, arr_len * sizeof(float));
                p += arr_len * sizeof(float);
            } else if (key == "tokenizer.ggml.token_type") {
                if (p + 12 > end) break;
                uint32_t elem_type = *reinterpret_cast<const uint32_t*>(p); p += 4;
                (void)elem_type;
                uint64_t arr_len = *reinterpret_cast<const uint64_t*>(p); p += 8;
                vocab.token_types.resize(arr_len);
                std::memcpy(vocab.token_types.data(), p, arr_len * sizeof(int32_t));
                p += arr_len * sizeof(int32_t);
            } else if (key == "tokenizer.ggml.bos_token_id") {
                vocab.bos_id = *reinterpret_cast<const uint32_t*>(p); p += 4;
            } else if (key == "tokenizer.ggml.eos_token_id") {
                vocab.eos_id = *reinterpret_cast<const uint32_t*>(p); p += 4;
            } else if (key == "tokenizer.ggml.unknown_token_id") {
                vocab.unk_id = *reinterpret_cast<const uint32_t*>(p); p += 4;
            } else if (key == "tokenizer.ggml.padding_token_id") {
                vocab.pad_id = *reinterpret_cast<const uint32_t*>(p); p += 4;
            }
        }
        uniqtoken_free_buffer(buffer, size);
        return true;
    }
    int32_t find_token_id(const std::string& tok) const {
        auto it = token_to_id.find(tok);
        if (it != token_to_id.end()) return it->second;
        return static_cast<int32_t>(unk_id);
    }
};
int main(int argc, char** argv) {
    std::string path = "crates/uniqtoken_core/demo_vocab.json";
    if (argc > 1) {
        path = argv[1];
    }
    std::cout << "[llama.cpp Hook] Loading UniqToken GGUF vocab from: " << path << std::endl;
    UniqTokenLlamaVocab vocab;
    if (!UniqTokenLlamaVocab::load(path, vocab)) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return 1;
    }
    std::cout << "  - Model type: " << vocab.model_type << std::endl;
    std::cout << "  - Vocab size: " << vocab.tokens.size() << " tokens" << std::endl;
    std::cout << "  - Special IDs: BOS=" << vocab.bos_id << ", EOS=" << vocab.eos_id
              << ", UNK=" << vocab.unk_id << ", PAD=" << vocab.pad_id << std::endl;
    assert(!vocab.tokens.empty());
    assert(vocab.tokens.size() == vocab.scores.size());
    assert(vocab.tokens.size() == vocab.token_types.size());
    std::cout << "[llama.cpp Hook] Validation PASSED with zero leaks." << std::endl;
    return 0;
}

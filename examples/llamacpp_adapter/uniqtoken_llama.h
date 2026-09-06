#ifndef UNIQTOKEN_LLAMA_H
#define UNIQTOKEN_LLAMA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Success return code. */
#define UNIQTOKEN_OK 0
/** Null pointer argument error. */
#define UNIQTOKEN_ERR_NULL_PTR -1
/** Invalid UTF-8 path error. */
#define UNIQTOKEN_ERR_INVALID_PATH -2
/** File I/O read error. */
#define UNIQTOKEN_ERR_IO -3
/** JSON parsing or malformed vocabulary error. */
#define UNIQTOKEN_ERR_PARSE -4
/** GGUF serialization error. */
#define UNIQTOKEN_ERR_SERIALIZE -5
/**
 * @brief GGUF token classification types matching llama.cpp internal vocabulary categories.
 */
enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};
/**
 * @brief Exports a UniqToken vocabulary JSON model file into binary GGUF v3 format.
 *
 * @param model_path Path to the UniqToken vocabulary JSON file.
 * @param buffer_out Output pointer to the allocated binary buffer.
 * @param size_out Output pointer to the size in bytes of the allocated buffer.
 * @return UNIQTOKEN_OK (0) on success, or negative error code on failure.
 */
int32_t uniqtoken_export_gguf_vocab(
    const char* model_path,
    void** buffer_out,
    size_t* size_out
);
/**
 * @brief Deallocates a buffer previously returned by uniqtoken_export_gguf_vocab.
 *
 * @param buffer Pointer to the allocated buffer to free.
 * @param size Size in bytes of the buffer.
 */
void uniqtoken_free_buffer(
    void* buffer,
    size_t size
);
#ifdef __cplusplus
}
#endif
#endif // UNIQTOKEN_LLAMA_H

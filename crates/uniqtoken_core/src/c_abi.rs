//! C-ABI exports for native GGUF vocabulary table loading and C++ integration (Issue #52).
use std::ffi::CStr;
use std::fs;
use std::os::raw::{c_char, c_void};
use std::path::Path;
pub const UNIQTOKEN_OK: i32 = 0;
pub const UNIQTOKEN_ERR_NULL_PTR: i32 = -1;
pub const UNIQTOKEN_ERR_INVALID_PATH: i32 = -2;
pub const UNIQTOKEN_ERR_IO: i32 = -3;
pub const UNIQTOKEN_ERR_PARSE: i32 = -4;
pub const UNIQTOKEN_ERR_SERIALIZE: i32 = -5;
// GGUF Token Types matching llama.cpp
pub const GGUF_TOKEN_TYPE_NORMAL: i32 = 1;
pub const GGUF_TOKEN_TYPE_UNKNOWN: i32 = 2;
pub const GGUF_TOKEN_TYPE_CONTROL: i32 = 3;
pub const GGUF_TOKEN_TYPE_USER_DEFINED: i32 = 4;
pub const GGUF_TOKEN_TYPE_BYTE: i32 = 6;
// GGUF Value Types
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub token: String,
    pub score: f32,
    pub id: u32,
    pub token_type: i32,
}
pub fn classify_token(token: &str) -> i32 {
    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
        let hex = &token[3..5];
        if hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return GGUF_TOKEN_TYPE_BYTE;
        }
    }
    if token == "<unk>" || token == "<|unk|>" {
        return GGUF_TOKEN_TYPE_UNKNOWN;
    }
    if token.starts_with("<|user_") || token.starts_with("<|custom_") {
        return GGUF_TOKEN_TYPE_USER_DEFINED;
    }
    if token == "<s>"
        || token == "</s>"
        || token == "<pad>"
        || token == "<|bos|>"
        || token == "<|eos|>"
        || token == "<|pad|>"
        || (token.starts_with("<|") && token.ends_with("|>"))
    {
        return GGUF_TOKEN_TYPE_CONTROL;
    }
    GGUF_TOKEN_TYPE_NORMAL
}
fn push_entry(entries: &mut Vec<TokenEntry>, i: usize, item: &serde_json::Value) {
    let Some(triple) = item.as_array() else { return };
    if triple.len() < 2 {
        return;
    }
    let tok = triple[0].as_str().unwrap_or("").to_string();
    let score = triple[1].as_f64().unwrap_or(0.0) as f32;
    let id = triple.get(2).and_then(|v| v.as_u64()).unwrap_or(i as u64) as u32;
    entries.push(TokenEntry { token: tok, score, id, token_type: classify_token(&tok) });
}
pub fn parse_vocab_json(content: &str) -> Result<Vec<TokenEntry>, i32> {
    let value: serde_json::Value = serde_json::from_str(content).map_err(|_| UNIQTOKEN_ERR_PARSE)?;
    let mut entries = Vec::new();
    if let Some(arr) = value.as_array() {
        for (i, item) in arr.iter().enumerate() {
            push_entry(&mut entries, i, item);
        }
    } else if let Some(arr) = value
        .as_object()
        .and_then(|obj| obj.get("vocab"))
        .and_then(|v| v.as_array())
    {
        for (i, item) in arr.iter().enumerate() {
            push_entry(&mut entries, i, item);
        }
    } else {
        return Err(UNIQTOKEN_ERR_PARSE);
    }
    if entries.is_empty() {
        return Err(UNIQTOKEN_ERR_PARSE);
    }
    entries.sort_by_key(|e| e.id);
    Ok(entries)
}
fn pack_str(data: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(bytes);
}
pub fn serialize_gguf_vocab(entries: &[TokenEntry]) -> Result<Vec<u8>, i32> {
    let mut data = Vec::with_capacity(1024 + entries.len() * 32);
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&8u64.to_le_bytes());
    pack_str(&mut data, "tokenizer.ggml.model");
    data.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    pack_str(&mut data, "llama");
    pack_str(&mut data, "tokenizer.ggml.tokens");
    data.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    data.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    data.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        pack_str(&mut data, &e.token);
    }
    pack_str(&mut data, "tokenizer.ggml.scores");
    data.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    data.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
    data.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        data.extend_from_slice(&e.score.to_le_bytes());
    }
    pack_str(&mut data, "tokenizer.ggml.token_type");
    data.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    data.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
    data.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        data.extend_from_slice(&e.token_type.to_le_bytes());
    }
    let mut bos_id = 1u32;
    let mut eos_id = 2u32;
    let mut unk_id = 0u32;
    let mut pad_id = 0u32;
    for e in entries {
        if e.token == "<|bos|>" || e.token == "<s>" {
            bos_id = e.id;
        } else if e.token == "<|eos|>" || e.token == "</s>" {
            eos_id = e.id;
        } else if e.token == "<|unk|>" || e.token == "<unk>" {
            unk_id = e.id;
        } else if e.token == "<|pad|>" || e.token == "<pad>" {
            pad_id = e.id;
        }
    }
    pack_str(&mut data, "tokenizer.ggml.bos_token_id");
    data.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    data.extend_from_slice(&bos_id.to_le_bytes());
    pack_str(&mut data, "tokenizer.ggml.eos_token_id");
    data.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    data.extend_from_slice(&eos_id.to_le_bytes());
    pack_str(&mut data, "tokenizer.ggml.unknown_token_id");
    data.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    data.extend_from_slice(&unk_id.to_le_bytes());
    pack_str(&mut data, "tokenizer.ggml.padding_token_id");
    data.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    data.extend_from_slice(&pad_id.to_le_bytes());
    Ok(data)
}
/// Pure C-ABI function exporting a UniqToken model/vocab into a binary GGUF v3 buffer.
#[no_mangle]
pub unsafe extern "C" fn uniqtoken_export_gguf_vocab(
    model_path: *const c_char,
    buffer_out: *mut *mut c_void,
    size_out: *mut usize,
) -> i32 {
    if model_path.is_null() || buffer_out.is_null() || size_out.is_null() {
        return UNIQTOKEN_ERR_NULL_PTR;
    }
    let c_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return UNIQTOKEN_ERR_INVALID_PATH,
    };
    let path = Path::new(c_str);
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return UNIQTOKEN_ERR_IO,
    };
    let entries = match parse_vocab_json(&content) {
        Ok(e) => e,
        Err(err) => return err,
    };
    let binary = match serialize_gguf_vocab(&entries) {
        Ok(b) => b,
        Err(err) => return err,
    };
    let mut boxed = binary.into_boxed_slice();
    *buffer_out = boxed.as_mut_ptr() as *mut c_void;
    *size_out = boxed.len();
    std::mem::forget(boxed);
    UNIQTOKEN_OK
}
/// Deallocates the buffer returned by `uniqtoken_export_gguf_vocab` to prevent leaks.
#[no_mangle]
pub unsafe extern "C" fn uniqtoken_free_buffer(buffer: *mut c_void, size: usize) {
    if !buffer.is_null() && size > 0 {
        let _ = Vec::from_raw_parts(buffer as *mut u8, size, size);
    }
}

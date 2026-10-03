//! Native Viterbi decoding and forward-backward expectation algorithms.

use crate::trie::{CachedSegmentation, RustPrefixTrie};
#[cfg(feature = "python")]
use crate::error::{core_error, CoreError, CoreResult};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "python")]
use rayon::prelude::*;

const DEFAULT_BYTE_LOG_P: f64 = -10.0;

/// Precomputed byte fallback tokens <0x00> through <0xFF> eliminating repeated string formatting and allocations.
pub(crate) const BYTE_FALLBACK_TOKENS: [&str; 256] = [
    "<0x00>", "<0x01>", "<0x02>", "<0x03>", "<0x04>", "<0x05>", "<0x06>", "<0x07>",
    "<0x08>", "<0x09>", "<0x0A>", "<0x0B>", "<0x0C>", "<0x0D>", "<0x0E>", "<0x0F>",
    "<0x10>", "<0x11>", "<0x12>", "<0x13>", "<0x14>", "<0x15>", "<0x16>", "<0x17>",
    "<0x18>", "<0x19>", "<0x1A>", "<0x1B>", "<0x1C>", "<0x1D>", "<0x1E>", "<0x1F>",
    "<0x20>", "<0x21>", "<0x22>", "<0x23>", "<0x24>", "<0x25>", "<0x26>", "<0x27>",
    "<0x28>", "<0x29>", "<0x2A>", "<0x2B>", "<0x2C>", "<0x2D>", "<0x2E>", "<0x2F>",
    "<0x30>", "<0x31>", "<0x32>", "<0x33>", "<0x34>", "<0x35>", "<0x36>", "<0x37>",
    "<0x38>", "<0x39>", "<0x3A>", "<0x3B>", "<0x3C>", "<0x3D>", "<0x3E>", "<0x3F>",
    "<0x40>", "<0x41>", "<0x42>", "<0x43>", "<0x44>", "<0x45>", "<0x46>", "<0x47>",
    "<0x48>", "<0x49>", "<0x4A>", "<0x4B>", "<0x4C>", "<0x4D>", "<0x4E>", "<0x4F>",
    "<0x50>", "<0x51>", "<0x52>", "<0x53>", "<0x54>", "<0x55>", "<0x56>", "<0x57>",
    "<0x58>", "<0x59>", "<0x5A>", "<0x5B>", "<0x5C>", "<0x5D>", "<0x5E>", "<0x5F>",
    "<0x60>", "<0x61>", "<0x62>", "<0x63>", "<0x64>", "<0x65>", "<0x66>", "<0x67>",
    "<0x68>", "<0x69>", "<0x6A>", "<0x6B>", "<0x6C>", "<0x6D>", "<0x6E>", "<0x6F>",
    "<0x70>", "<0x71>", "<0x72>", "<0x73>", "<0x74>", "<0x75>", "<0x76>", "<0x77>",
    "<0x78>", "<0x79>", "<0x7A>", "<0x7B>", "<0x7C>", "<0x7D>", "<0x7E>", "<0x7F>",
    "<0x80>", "<0x81>", "<0x82>", "<0x83>", "<0x84>", "<0x85>", "<0x86>", "<0x87>",
    "<0x88>", "<0x89>", "<0x8A>", "<0x8B>", "<0x8C>", "<0x8D>", "<0x8E>", "<0x8F>",
    "<0x90>", "<0x91>", "<0x92>", "<0x93>", "<0x94>", "<0x95>", "<0x96>", "<0x97>",
    "<0x98>", "<0x99>", "<0x9A>", "<0x9B>", "<0x9C>", "<0x9D>", "<0x9E>", "<0x9F>",
    "<0xA0>", "<0xA1>", "<0xA2>", "<0xA3>", "<0xA4>", "<0xA5>", "<0xA6>", "<0xA7>",
    "<0xA8>", "<0xA9>", "<0xAA>", "<0xAB>", "<0xAC>", "<0xAD>", "<0xAE>", "<0xAF>",
    "<0xB0>", "<0xB1>", "<0xB2>", "<0xB3>", "<0xB4>", "<0xB5>", "<0xB6>", "<0xB7>",
    "<0xB8>", "<0xB9>", "<0xBA>", "<0xBB>", "<0xBC>", "<0xBD>", "<0xBE>", "<0xBF>",
    "<0xC0>", "<0xC1>", "<0xC2>", "<0xC3>", "<0xC4>", "<0xC5>", "<0xC6>", "<0xC7>",
    "<0xC8>", "<0xC9>", "<0xCA>", "<0xCB>", "<0xCC>", "<0xCD>", "<0xCE>", "<0xCF>",
    "<0xD0>", "<0xD1>", "<0xD2>", "<0xD3>", "<0xD4>", "<0xD5>", "<0xD6>", "<0xD7>",
    "<0xD8>", "<0xD9>", "<0xDA>", "<0xDB>", "<0xDC>", "<0xDD>", "<0xDE>", "<0xDF>",
    "<0xE0>", "<0xE1>", "<0xE2>", "<0xE3>", "<0xE4>", "<0xE5>", "<0xE6>", "<0xE7>",
    "<0xE8>", "<0xE9>", "<0xEA>", "<0xEB>", "<0xEC>", "<0xED>", "<0xEE>", "<0xEF>",
    "<0xF0>", "<0xF1>", "<0xF2>", "<0xF3>", "<0xF4>", "<0xF5>", "<0xF6>", "<0xF7>",
    "<0xF8>", "<0xF9>", "<0xFA>", "<0xFB>", "<0xFC>", "<0xFD>", "<0xFE>", "<0xFF>",
];


/// Chunks longer than this skip the segmentation cache: long chunks dominate
/// cache memory while repeating far less often than short words.
const SEG_CACHE_MAX_CHUNK_BYTES: usize = 1024;

/// Word-level memoization wrapper around `viterbi_decode_chars`.
///
/// Real corpora are Zipfian — a handful of distinct words make up most chunks —
/// so a cache hit (hash lookup + Arc clone) replaces the whole trie walk + DP.
/// `max_edges_per_node` pruning is NOT cacheable; callers pass `None` here.
#[cfg(any(test, feature = "fuzzing"))]
pub fn decode_cached(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> Result<CachedSegmentation, String> {
    decode_cached_inner(text, trie, byte_fallback)
}

#[cfg(not(any(test, feature = "fuzzing")))]
pub(crate) fn decode_cached(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> Result<CachedSegmentation, String> {
    decode_cached_inner(text, trie, byte_fallback)
}

fn decode_cached_inner(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> Result<CachedSegmentation, String> {
    if text.is_empty() {
        return Ok(Arc::new(Vec::new()));
    }
    if text.len() > SEG_CACHE_MAX_CHUNK_BYTES {
        // ASCII fast-path: skip Vec<char> allocation entirely.
        let spans = if text.is_ascii() {
            viterbi_decode_ascii(text.as_bytes(), trie, byte_fallback, None)?
        } else {
            let chars: Vec<char> = text.chars().collect();
            viterbi_decode_chars(&chars, trie, byte_fallback, None)?
        };
        return Ok(Arc::new(
            spans
                .into_iter()
                .map(|s| (s.token, s.token_id, s.start, s.end))
                .collect(),
        ));
    }
    if let Some(hit) = trie.seg_cache_get(byte_fallback, text) {
        return Ok(hit);
    }
    // ASCII fast-path: skip Vec<char> allocation entirely.
    let spans = if text.is_ascii() {
        viterbi_decode_ascii(text.as_bytes(), trie, byte_fallback, None)?
    } else {
        let chars: Vec<char> = text.chars().collect();
        viterbi_decode_chars(&chars, trie, byte_fallback, None)?
    };
    let seg: CachedSegmentation = Arc::new(
        spans
            .into_iter()
            .map(|s| (s.token, s.token_id, s.start, s.end))
            .collect(),
    );
    trie.seg_cache_put(byte_fallback, text, seg.clone());
    Ok(seg)
}

#[cfg(feature = "python")]
fn spans_from_cached(seg: &CachedSegmentation) -> Vec<ViterbiSpan> {
    seg.iter()
        .map(|(token, token_id, start, end)| ViterbiSpan {
            token: token.clone(),
            token_id: *token_id,
            start: *start,
            end: *end,
        })
        .collect()
}

#[cfg(feature = "python")]
fn tokens_from_cached(seg: &CachedSegmentation) -> Vec<String> {
    seg.iter().map(|(token, ..)| token.clone()).collect()
}

#[cfg(feature = "python")]
fn ids_from_cached(seg: &CachedSegmentation) -> Result<Vec<u32>, String> {
    seg.iter()
        .map(|(token, token_id, ..)| {
            token_id
                .ok_or_else(|| format!("decoded token {token:?} has no integer ID"))
        })
        .collect()
}

#[derive(Clone, Debug)]
struct TokenPiece {
    token: String,
    token_id: Option<u32>,
}

#[derive(Clone, Debug)]
struct Edge {
    prev_node: usize,
    pieces: Vec<TokenPiece>,
    log_p: f64,
    // Consumed by the Python-gated forward-backward path; retained (not read)
    // on WebAssembly builds.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    length: usize,
}

#[derive(Clone, Debug)]
struct Node {
    best_score: f64,
    best_edge: Option<Edge>,
}

#[cfg(feature = "python")]
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct ViterbiSpan {
    #[pyo3(get)]
    pub token: String,
    #[pyo3(get)]
    pub token_id: Option<u32>,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

#[cfg(not(feature = "python"))]
#[derive(Clone, Debug)]
pub struct ViterbiSpan {
    pub token: String,
    pub token_id: Option<u32>,
    pub start: usize,
    pub end: usize,
}

#[cfg(feature = "python")]
pub(crate) fn diagnostic_viterbi_inner(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> (f64, f64, usize, usize) {
    use std::time::Instant;
    let n = chars.len();
    let mut t_trie = 0.0;
    let mut edges = 0usize;
    let mut incoming: Vec<Vec<Edge>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        let t = Instant::now();
        let matches = trie.common_prefix_search_chars(chars, i);
        t_trie += t.elapsed().as_secs_f64();
        if matches.is_empty() && byte_fallback {
            let mut pieces = Vec::new();
            let mut edge_log_p = 0.0;
            let mut encoded_buf = [0_u8; 4];
            for byte in chars[i].encode_utf8(&mut encoded_buf).as_bytes() {
                let token = BYTE_FALLBACK_TOKENS[*byte as usize].to_string();
                let (token_id, log_p) = trie.exact_metadata(&token).unwrap_or((None, DEFAULT_BYTE_LOG_P));
                pieces.push(TokenPiece { token, token_id });
                edge_log_p += log_p;
            }
            incoming[i + 1].push(Edge { prev_node: i, pieces, log_p: edge_log_p, length: 1 });
            edges += 1;
        } else {
            for (token, token_id, log_p, char_len) in matches {
                let end = i + char_len;
                if end <= n {
                    incoming[end].push(Edge { prev_node: i, pieces: vec![TokenPiece { token, token_id }], log_p, length: char_len });
                    edges += 1;
                }
            }
        }
    }
    let mut nodes: Vec<Node> = (0..=n).map(|_| Node { best_score: f64::NEG_INFINITY, best_edge: None }).collect();
    nodes[0].best_score = 0.0;
    let t = Instant::now();
    for end in 1..=n {
        for edge in &incoming[end] {
            let prev = nodes[edge.prev_node].best_score;
            if prev == f64::NEG_INFINITY { continue; }
            let score = prev + edge.log_p;
            if score > nodes[end].best_score {
                nodes[end].best_score = score;
                nodes[end].best_edge = Some(edge.clone());
            }
        }
    }
    let t_dp = t.elapsed().as_secs_f64();
    (t_trie, t_dp, edges, n + 1)
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn rust_diagnostic_viterbi(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
) -> CoreResult<(f64, f64, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let (t_trie, t_dp, edges, states) = diagnostic_viterbi_inner(&chars, trie, byte_fallback);
    Ok((t_trie, t_dp, edges, states))
}

#[cfg(any(test, feature = "fuzzing"))]
pub fn viterbi_decode_chars(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    viterbi_decode_chars_inner(chars, trie, byte_fallback, max_edges_per_node)
}

#[cfg(not(any(test, feature = "fuzzing")))]
pub(crate) fn viterbi_decode_chars(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    viterbi_decode_chars_inner(chars, trie, byte_fallback, max_edges_per_node)
}

#[derive(Clone, Copy)]
enum CharBackpointer<'a> {
    Single {
        prev_node: usize,
        token: &'a str,
        token_id: Option<u32>,
    },
    MultiByte {
        prev_node: usize,
        count: u8,
        bytes: [u8; 4],
        token_ids: [Option<u32>; 4],
    },
}

#[derive(Clone, Copy)]
struct CharNode<'a> {
    best_score: f64,
    backpointer: Option<CharBackpointer<'a>>,
}

fn viterbi_decode_chars_inner(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    let n = chars.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Zero-allocation forward relaxation for the unpruned hot path.
    if max_edges_per_node.is_none() {
        let mut nodes: Vec<CharNode> = vec![
            CharNode {
                best_score: f64::NEG_INFINITY,
                backpointer: None,
            };
            n + 1
        ];
        nodes[0].best_score = 0.0;

        for i in 0..n {
            let prev_score = nodes[i].best_score;
            if prev_score == f64::NEG_INFINITY {
                continue;
            }
            let mut has_match = false;
            trie.for_each_prefix_chars(chars, i, |token, token_id, log_p, char_len| {
                let end = i + char_len;
                if end <= n {
                    has_match = true;
                    let score = prev_score + log_p;
                    if score > nodes[end].best_score {
                        nodes[end].best_score = score;
                        nodes[end].backpointer = Some(CharBackpointer::Single {
                            prev_node: i,
                            token,
                            token_id,
                        });
                    }
                }
            });

            if !has_match && byte_fallback {
                let mut encoded_buf = [0_u8; 4];
                let utf8_str = chars[i].encode_utf8(&mut encoded_buf);
                let utf8_bytes = utf8_str.as_bytes();
                let count = utf8_bytes.len() as u8;
                let mut edge_log_p = 0.0;
                let mut byte_arr = [0_u8; 4];
                let mut token_ids = [None; 4];
                for (k, &byte) in utf8_bytes.iter().enumerate() {
                    byte_arr[k] = byte;
                    let token = BYTE_FALLBACK_TOKENS[byte as usize];
                    let (tok_id, log_p) = trie
                        .exact_metadata(token)
                        .unwrap_or((None, DEFAULT_BYTE_LOG_P));
                    token_ids[k] = tok_id;
                    edge_log_p += log_p;
                }
                let end = i + 1;
                let score = prev_score + edge_log_p;
                if score > nodes[end].best_score {
                    nodes[end].best_score = score;
                    nodes[end].backpointer = Some(CharBackpointer::MultiByte {
                        prev_node: i,
                        count,
                        bytes: byte_arr,
                        token_ids,
                    });
                }
            }
        }

        if nodes[n].backpointer.is_none() {
            return Err(format!(
                "lattice disconnected at character index {n}; enable byte fallback or provide complete vocabulary coverage"
            ));
        }

        let mut spans = Vec::with_capacity((n / 3).clamp(4, 128));
        let mut end = n;
        while end > 0 {
            let bp = nodes[end].backpointer.ok_or_else(|| {
                format!("lattice backpointer missing at character index {end}")
            })?;
            match bp {
                CharBackpointer::Single {
                    prev_node,
                    token,
                    token_id,
                } => {
                    spans.push(ViterbiSpan {
                        token: token.to_string(),
                        token_id,
                        start: prev_node,
                        end,
                    });
                    end = prev_node;
                }
                CharBackpointer::MultiByte {
                    prev_node,
                    count,
                    bytes,
                    token_ids,
                } => {
                    for k in (0..count as usize).rev() {
                        let token = BYTE_FALLBACK_TOKENS[bytes[k] as usize];
                        spans.push(ViterbiSpan {
                            token: token.to_string(),
                            token_id: token_ids[k],
                            start: prev_node,
                            end,
                        });
                    }
                    end = prev_node;
                }
            }
        }

        spans.reverse();
        return Ok(spans);
    }

    // Pruned beam fallback (retained for max_edges_per_node != None).
    let mut incoming: Vec<Vec<Edge>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        let mut has_match = false;
        trie.for_each_prefix_chars(chars, i, |token, token_id, log_p, char_len| {
            let end = i + char_len;
            if end <= n {
                has_match = true;
                incoming[end].push(Edge {
                    prev_node: i,
                    pieces: vec![TokenPiece {
                        token: token.to_string(),
                        token_id,
                    }],
                    log_p,
                    length: char_len,
                });
            }
        });
        if !has_match && byte_fallback {
            let mut pieces = Vec::new();
            let mut edge_log_p = 0.0;
            let mut encoded_buf = [0_u8; 4];
            for byte in chars[i].encode_utf8(&mut encoded_buf).as_bytes() {
                let token = BYTE_FALLBACK_TOKENS[*byte as usize].to_string();
                let (token_id, log_p) = trie
                    .exact_metadata(&token)
                    .unwrap_or((None, DEFAULT_BYTE_LOG_P));
                pieces.push(TokenPiece { token, token_id });
                edge_log_p += log_p;
            }
            incoming[i + 1].push(Edge {
                prev_node: i,
                pieces,
                log_p: edge_log_p,
                length: 1,
            });
        }
    }

    if let Some(limit) = max_edges_per_node {
        for edges in incoming.iter_mut().skip(1) {
            if edges.len() > limit {
                edges.sort_by(|left, right| right.log_p.total_cmp(&left.log_p));
                edges.truncate(limit);
            }
        }
    }

    let mut nodes: Vec<Node> = (0..=n)
        .map(|_| Node {
            best_score: f64::NEG_INFINITY,
            best_edge: None,
        })
        .collect();
    nodes[0].best_score = 0.0;

    for end in 1..=n {
        for edge in &incoming[end] {
            let previous_score = nodes[edge.prev_node].best_score;
            if previous_score == f64::NEG_INFINITY {
                continue;
            }
            let score = previous_score + edge.log_p;
            if score > nodes[end].best_score {
                nodes[end].best_score = score;
                nodes[end].best_edge = Some(edge.clone());
            }
        }
    }

    if nodes[n].best_edge.is_none() {
        return Err(format!(
            "lattice disconnected at character index {n}; enable byte fallback or provide complete vocabulary coverage"
        ));
    }

    let mut spans = Vec::new();
    let mut end = n;
    while end > 0 {
        let edge = nodes[end].best_edge.as_ref().ok_or_else(|| {
            format!("lattice backpointer missing at character index {end}")
        })?;
        for piece in edge.pieces.iter().rev() {
            spans.push(ViterbiSpan {
                token: piece.token.clone(),
                token_id: piece.token_id,
                start: edge.prev_node,
                end,
            });
        }
        end = edge.prev_node;
    }

    spans.reverse();
    Ok(spans)
}

#[derive(Clone, Copy)]
struct AsciiBackpointer<'a> {
    prev_node: usize,
    token: &'a str,
    token_id: Option<u32>,
}

#[derive(Clone, Copy)]
struct AsciiNode<'a> {
    best_score: f64,
    backpointer: Option<AsciiBackpointer<'a>>,
}

/// ASCII-specialized Viterbi decode operating on raw `&[u8]` byte slices.
///
/// For pure-ASCII text (`str::is_ascii()`), `byte_offset == char_offset`,
/// so the span positions produced by this function are **identical** to those
/// from [`viterbi_decode_chars`].  The key advantage is that we skip the
/// `Vec<char>` heap allocation (4 bytes per codepoint) and all UTF-8
/// boundary checks.
///
/// Callers must ensure every byte in `bytes` satisfies `b < 0x80`.
#[cfg(any(test, feature = "fuzzing"))]
pub fn viterbi_decode_ascii(
    bytes: &[u8],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    viterbi_decode_ascii_inner(bytes, trie, byte_fallback, max_edges_per_node)
}

#[cfg(not(any(test, feature = "fuzzing")))]
pub(crate) fn viterbi_decode_ascii(
    bytes: &[u8],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    viterbi_decode_ascii_inner(bytes, trie, byte_fallback, max_edges_per_node)
}

fn viterbi_decode_ascii_inner(
    bytes: &[u8],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<ViterbiSpan>, String> {
    let n = bytes.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Zero-allocation forward relaxation for the unpruned hot path.
    if max_edges_per_node.is_none() {
        let mut nodes: Vec<AsciiNode> = vec![
            AsciiNode {
                best_score: f64::NEG_INFINITY,
                backpointer: None,
            };
            n + 1
        ];
        nodes[0].best_score = 0.0;

        for i in 0..n {
            let prev_score = nodes[i].best_score;
            if prev_score == f64::NEG_INFINITY {
                continue;
            }
            let mut has_match = false;
            trie.for_each_prefix_ascii(bytes, i, |token, token_id, log_p, char_len| {
                let end = i + char_len;
                if end <= n {
                    has_match = true;
                    let score = prev_score + log_p;
                    if score > nodes[end].best_score {
                        nodes[end].best_score = score;
                        nodes[end].backpointer = Some(AsciiBackpointer {
                            prev_node: i,
                            token,
                            token_id,
                        });
                    }
                }
            });

            if !has_match && byte_fallback {
                let b = bytes[i];
                let token = BYTE_FALLBACK_TOKENS[b as usize];
                let (token_id, log_p) = trie
                    .exact_metadata(token)
                    .unwrap_or((None, DEFAULT_BYTE_LOG_P));
                let end = i + 1;
                let score = prev_score + log_p;
                if score > nodes[end].best_score {
                    nodes[end].best_score = score;
                    nodes[end].backpointer = Some(AsciiBackpointer {
                        prev_node: i,
                        token,
                        token_id,
                    });
                }
            }
        }

        if nodes[n].backpointer.is_none() {
            return Err(format!(
                "lattice disconnected at character index {n}; enable byte fallback or provide complete vocabulary coverage"
            ));
        }

        let mut spans = Vec::with_capacity((n / 3).clamp(4, 128));
        let mut end = n;
        while end > 0 {
            let bp = nodes[end].backpointer.ok_or_else(|| {
                format!("lattice backpointer missing at character index {end}")
            })?;
            spans.push(ViterbiSpan {
                token: bp.token.to_string(),
                token_id: bp.token_id,
                start: bp.prev_node,
                end,
            });
            end = bp.prev_node;
        }

        spans.reverse();
        return Ok(spans);
    }

    // Pruned beam fallback (retained for max_edges_per_node != None).
    let mut incoming: Vec<Vec<Edge>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        let mut has_match = false;
        trie.for_each_prefix_ascii(bytes, i, |token, token_id, log_p, char_len| {
            let end = i + char_len;
            if end <= n {
                has_match = true;
                incoming[end].push(Edge {
                    prev_node: i,
                    pieces: vec![TokenPiece {
                        token: token.to_string(),
                        token_id,
                    }],
                    log_p,
                    length: char_len,
                });
            }
        });
        if !has_match && byte_fallback {
            let b = bytes[i];
            let token = BYTE_FALLBACK_TOKENS[b as usize].to_string();
            let (token_id, log_p) = trie
                .exact_metadata(&token)
                .unwrap_or((None, DEFAULT_BYTE_LOG_P));
            incoming[i + 1].push(Edge {
                prev_node: i,
                pieces: vec![TokenPiece { token, token_id }],
                log_p,
                length: 1,
            });
        }
    }

    if let Some(limit) = max_edges_per_node {
        for edges in incoming.iter_mut().skip(1) {
            if edges.len() > limit {
                edges.sort_by(|left, right| right.log_p.total_cmp(&left.log_p));
                edges.truncate(limit);
            }
        }
    }

    let mut nodes: Vec<Node> = (0..=n)
        .map(|_| Node {
            best_score: f64::NEG_INFINITY,
            best_edge: None,
        })
        .collect();
    nodes[0].best_score = 0.0;

    for end in 1..=n {
        for edge in &incoming[end] {
            let previous_score = nodes[edge.prev_node].best_score;
            if previous_score == f64::NEG_INFINITY {
                continue;
            }
            let score = previous_score + edge.log_p;
            if score > nodes[end].best_score {
                nodes[end].best_score = score;
                nodes[end].best_edge = Some(edge.clone());
            }
        }
    }

    if nodes[n].best_edge.is_none() {
        return Err(format!(
            "lattice disconnected at character index {n}; enable byte fallback or provide complete vocabulary coverage"
        ));
    }

    let mut spans = Vec::new();
    let mut end = n;
    while end > 0 {
        let edge = nodes[end].best_edge.as_ref().ok_or_else(|| {
            format!("lattice backpointer missing at character index {end}")
        })?;
        for piece in edge.pieces.iter().rev() {
            spans.push(ViterbiSpan {
                token: piece.token.clone(),
                token_id: piece.token_id,
                start: edge.prev_node,
                end,
            });
        }
        end = edge.prev_node;
    }

    spans.reverse();
    Ok(spans)
}

#[cfg(feature = "python")]
fn viterbi_ids_chars(
    chars: &[char],
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> Result<Vec<u32>, String> {
    let spans = viterbi_decode_chars(chars, trie, byte_fallback, max_edges_per_node)?;
    let mut ids = Vec::with_capacity(spans.len());
    for s in spans {
        let id = s
            .token_id
            .ok_or_else(|| format!("decoded token {:?} has no integer ID", s.token))?;
        ids.push(id);
    }
    Ok(ids)
}

/// Computes the most probable segmentation using Python character offsets.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (text, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_viterbi_decode(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> CoreResult<Vec<ViterbiSpan>> {
    if matches!(max_edges_per_node, Some(0)) {
        return core_error("max_edges_per_node must be greater than zero");
    }
    if max_edges_per_node.is_none() {
        let seg = decode_cached(text, trie, byte_fallback).map_err(CoreError)?;
        return Ok(spans_from_cached(&seg));
    }
    // ASCII fast-path: skip Vec<char> allocation when possible.
    if text.is_ascii() {
        return viterbi_decode_ascii(text.as_bytes(), trie, byte_fallback, max_edges_per_node)
            .map_err(CoreError);
    }
    let chars: Vec<char> = text.chars().collect();
    viterbi_decode_chars(&chars, trie, byte_fallback, max_edges_per_node).map_err(CoreError)
}

/// Computes most probable segmentations for a batch of strings concurrently using Rayon (releases GIL).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_viterbi_decode_batch<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> CoreResult<Vec<Vec<ViterbiSpan>>> {
    if matches!(max_edges_per_node, Some(0)) {
        return core_error("max_edges_per_node must be greater than zero");
    }
    let borrowed = crate::pipeline::extract_borrowed_strings(texts)?;
    let decode_item = |text: &str| -> Result<Vec<ViterbiSpan>, String> {
        if max_edges_per_node.is_none() {
            decode_cached(text, trie, byte_fallback).map(|seg| spans_from_cached(&seg))
        } else if text.is_ascii() {
            viterbi_decode_ascii(text.as_bytes(), trie, byte_fallback, max_edges_per_node)
        } else {
            let chars: Vec<char> = text.chars().collect();
            viterbi_decode_chars(&chars, trie, byte_fallback, max_edges_per_node)
        }
    };
    // ponytail: rayon par_iter costs ~200us/call on Windows thread-pool wakeup;
    // below ~32 items sequential beats it ~4x. Upgrade path: work-estimate
    // (total bytes) instead of item count.
    if borrowed.len() < 32 {
        return borrowed
            .iter()
            .map(|text| decode_item(text.as_ref()).map_err(CoreError))
            .collect();
    }
    py.detach(|| {
        borrowed
            .par_iter()
            .map(|text| decode_item(text.as_ref()).map_err(CoreError))
            .collect()
    })
}

/// Batch encodes strings to token strings (no ViterbiSpan wrapper) — single FFI, minimal conversion.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_encode_tokens_batch<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> CoreResult<Vec<Vec<String>>> {
    if matches!(max_edges_per_node, Some(0)) {
        return core_error("max_edges_per_node must be greater than zero");
    }
    let borrowed = crate::pipeline::extract_borrowed_strings(texts)?;
    let decode_item = |text: &str| -> Result<Vec<String>, String> {
        if max_edges_per_node.is_none() {
            decode_cached(text, trie, byte_fallback).map(|seg| tokens_from_cached(&seg))
        } else if text.is_ascii() {
            viterbi_decode_ascii(text.as_bytes(), trie, byte_fallback, max_edges_per_node)
                .map(|spans| spans.into_iter().map(|s| s.token).collect())
        } else {
            let chars: Vec<char> = text.chars().collect();
            viterbi_decode_chars(&chars, trie, byte_fallback, max_edges_per_node)
                .map(|spans| spans.into_iter().map(|s| s.token).collect())
        }
    };
    // ponytail: rayon par_iter costs ~200us/call on Windows thread-pool wakeup;
    // below ~32 items sequential beats it ~4x. Upgrade path: work-estimate
    // (total bytes) instead of item count.
    if borrowed.len() < 32 {
        return borrowed
            .iter()
            .map(|text| decode_item(text.as_ref()).map_err(CoreError))
            .collect();
    }
    py.detach(|| {
        borrowed
            .par_iter()
            .map(|text| decode_item(text.as_ref()).map_err(CoreError))
            .collect()
    })
}

/// Batch encodes strings directly to token integer IDs using parallel Rayon workers (releases GIL).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback, max_edges_per_node=None))]
pub fn rust_encode_ids_batch<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    max_edges_per_node: Option<usize>,
) -> CoreResult<Vec<Vec<u32>>> {
    if matches!(max_edges_per_node, Some(0)) {
        return core_error("max_edges_per_node must be greater than zero");
    }
    let borrowed = crate::pipeline::extract_borrowed_strings(texts)?;
    let decode_item = |text: &str| -> Result<Vec<u32>, String> {
        if max_edges_per_node.is_none() {
            decode_cached(text, trie, byte_fallback).and_then(|seg| ids_from_cached(&seg))
        } else if text.is_ascii() {
            let spans = viterbi_decode_ascii(text.as_bytes(), trie, byte_fallback, max_edges_per_node)?;
            spans.into_iter()
                .map(|s| s.token_id.ok_or_else(|| format!("decoded token {:?} has no integer ID", s.token)))
                .collect()
        } else {
            let chars: Vec<char> = text.chars().collect();
            viterbi_ids_chars(&chars, trie, byte_fallback, max_edges_per_node)
        }
    };
    // ponytail: rayon par_iter costs ~200us/call on Windows thread-pool wakeup;
    // below ~32 items sequential beats it ~4x. Upgrade path: work-estimate
    // (total bytes) instead of item count.
    if borrowed.len() < 32 {
        return borrowed
            .iter()
            .map(|text| decode_item(text.as_ref()).map_err(CoreError))
            .collect();
    }
    py.detach(|| {
        borrowed
            .par_iter()
            .map(|text| decode_item(text.as_ref()).map_err(CoreError))
            .collect()
    })
}


/// Forward-backward statistics for text fully covered by the supplied trie.
#[cfg(feature = "python")]
#[pyfunction]
pub fn rust_forward_backward_expectations(
    text: &str,
    trie: &RustPrefixTrie,
    freq: f64,
) -> CoreResult<(HashMap<String, f64>, f64)> {
    if !freq.is_finite() || freq < 0.0 {
        return core_error("freq must be finite and non-negative");
    }

    let is_ascii = text.is_ascii();
    let n = if is_ascii { text.len() } else { text.chars().count() };
    if n == 0 || freq == 0.0 {
        return Ok((HashMap::new(), 0.0));
    }

    let mut all_edges: Vec<Edge> = Vec::new();
    if is_ascii {
        let bytes = text.as_bytes();
        for i in 0..n {
            for (token, token_id, log_p, char_len) in trie.common_prefix_search_ascii(bytes, i) {
                if i + char_len <= n {
                    all_edges.push(Edge {
                        prev_node: i,
                        pieces: vec![TokenPiece { token, token_id }],
                        log_p,
                        length: char_len,
                    });
                }
            }
        }
    } else {
        let chars: Vec<char> = text.chars().collect();
        for i in 0..n {
            for (token, token_id, log_p, char_len) in trie.common_prefix_search_chars(&chars, i) {
                if i + char_len <= n {
                    all_edges.push(Edge {
                        prev_node: i,
                        pieces: vec![TokenPiece { token, token_id }],
                        log_p,
                        length: char_len,
                    });
                }
            }
        }
    }

    let mut alpha = vec![f64::NEG_INFINITY; n + 1];
    alpha[0] = 0.0;
    for edge in &all_edges {
        let previous = alpha[edge.prev_node];
        if previous != f64::NEG_INFINITY {
            let end = edge.prev_node + edge.length;
            alpha[end] = log_add(alpha[end], previous + edge.log_p);
        }
    }

    let total_log_z = alpha[n];
    if total_log_z == f64::NEG_INFINITY {
        return core_error("lattice is disconnected; forward-backward requires complete trie coverage");
    }

    let mut beta = vec![f64::NEG_INFINITY; n + 1];
    beta[n] = 0.0;
    for edge in all_edges.iter().rev() {
        let end = edge.prev_node + edge.length;
        if beta[end] != f64::NEG_INFINITY {
            beta[edge.prev_node] = log_add(beta[edge.prev_node], beta[end] + edge.log_p);
        }
    }

    let mut expected_counts = HashMap::new();
    for edge in &all_edges {
        let end = edge.prev_node + edge.length;
        if alpha[edge.prev_node] == f64::NEG_INFINITY || beta[end] == f64::NEG_INFINITY {
            continue;
        }
        let posterior = (alpha[edge.prev_node] + edge.log_p + beta[end] - total_log_z).exp();
        let token = &edge.pieces[0].token;
        *expected_counts.entry(token.clone()).or_insert(0.0) += posterior * freq;
    }

    Ok((expected_counts, total_log_z * freq))
}

#[inline]
#[cfg(feature = "python")]
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a == f64::INFINITY || b == f64::INFINITY {
        f64::INFINITY
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn log_add_handles_infinities() {
        assert_eq!(log_add(f64::NEG_INFINITY, -2.0), -2.0);
        assert_eq!(log_add(f64::INFINITY, f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn multibyte_fallback_emits_every_byte_with_one_character_span() {
        let trie = RustPrefixTrie::new(None);
        let spans = rust_viterbi_decode("éa", &trie, true, None).unwrap();
        let tokens: Vec<&str> = spans.iter().map(|span| span.token.as_str()).collect();
        assert_eq!(tokens, vec!["<0xC3>", "<0xA9>", "<0x61>"]);
        assert_eq!((spans[0].start, spans[0].end), (0, 1));
        assert_eq!((spans[1].start, spans[1].end), (0, 1));
        assert_eq!((spans[2].start, spans[2].end), (1, 2));
    }

    #[test]
    fn pruning_keeps_stable_backpointers() {
        let mut trie = RustPrefixTrie::new(None);
        trie.insert("a", -1.0, Some(1)).unwrap();
        trie.insert("ba", -0.1, Some(2)).unwrap();
        trie.insert("ab", -0.2, Some(3)).unwrap();
        trie.insert("b", -1.0, Some(4)).unwrap();
        let spans = rust_viterbi_decode("aba", &trie, false, Some(1)).unwrap();
        assert!(!spans.is_empty());
        assert_eq!(spans.last().unwrap().end, 3);
    }

    #[test]
    fn rejects_zero_beam_size() {
        let trie = RustPrefixTrie::new(None);
        assert!(rust_viterbi_decode("a", &trie, true, Some(0)).is_err());
    }

    #[test]
    fn ascii_fast_path_matches_char_decode_exactly() {
        let mut trie = RustPrefixTrie::new(None);
        trie.insert("the", -0.5, Some(1)).unwrap();
        trie.insert("quick", -1.2, Some(2)).unwrap();
        trie.insert("brown", -1.5, Some(3)).unwrap();
        trie.insert("fox", -0.8, Some(4)).unwrap();
        trie.insert(" ", -0.1, Some(5)).unwrap();
        trie.insert("o", -2.0, Some(6)).unwrap();
        trie.insert("x", -2.0, Some(7)).unwrap();

        let text = "the quick brown fox";
        let chars: Vec<char> = text.chars().collect();
        let spans_chars = viterbi_decode_chars(&chars, &trie, true, None).unwrap();
        let spans_ascii = viterbi_decode_ascii(text.as_bytes(), &trie, true, None).unwrap();

        assert_eq!(spans_chars.len(), spans_ascii.len());
        for (c, a) in spans_chars.iter().zip(spans_ascii.iter()) {
            assert_eq!(c.token, a.token);
            assert_eq!(c.token_id, a.token_id);
            assert_eq!(c.start, a.start);
            assert_eq!(c.end, a.end);
        }
    }

    #[test]
    fn ascii_fast_path_byte_fallback_matches_char_decode() {
        let trie = RustPrefixTrie::new(None);
        let text = "xyz123";
        let chars: Vec<char> = text.chars().collect();
        let spans_chars = viterbi_decode_chars(&chars, &trie, true, None).unwrap();
        let spans_ascii = viterbi_decode_ascii(text.as_bytes(), &trie, true, None).unwrap();

        assert_eq!(spans_chars.len(), spans_ascii.len());
        for (c, a) in spans_chars.iter().zip(spans_ascii.iter()) {
            assert_eq!(c.token, a.token);
            assert_eq!(c.token_id, a.token_id);
            assert_eq!(c.start, a.start);
            assert_eq!(c.end, a.end);
        }
    }

    #[test]
    fn ascii_fast_path_pruning_matches_char_decode() {
        // "aaa" over {a, aa, aaa} gives node 3 three incoming edges, so a
        // beam of 1 exercises the duplicated truncation logic in both paths.
        let mut trie = RustPrefixTrie::new(None);
        trie.insert("a", -0.5, Some(1)).unwrap();
        trie.insert("aa", -0.4, Some(2)).unwrap();
        trie.insert("aaa", -0.3, Some(3)).unwrap();
        let text = "aaa";
        let chars: Vec<char> = text.chars().collect();
        for limit in [Some(1), Some(2), None] {
            let spans_chars = viterbi_decode_chars(&chars, &trie, true, limit).unwrap();
            let spans_ascii = viterbi_decode_ascii(text.as_bytes(), &trie, true, limit).unwrap();
            assert_eq!(spans_chars.len(), spans_ascii.len(), "span count diverged at {:?}", limit);
            for (c, a) in spans_chars.iter().zip(spans_ascii.iter()) {
                assert_eq!(
                    (&c.token, c.token_id, c.start, c.end),
                    (&a.token, a.token_id, a.start, a.end),
                    "span diverged at {:?}",
                    limit
                );
            }
        }
    }

    #[test]
    #[ignore = "long-running benchmark intended for release profile"]
    fn perf_bench_viterbi_throughput() {
        use std::time::Instant;
        let mut trie = RustPrefixTrie::new(None);
        let common_subwords = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "function", "return", "const", "let", "var", "import", "export", "class",
            "def", "self", "async", "await", "print", "None", "True", "False",
            "Hello", "world", "UniqToken", "tokenizer", "fast", "engine", "allocation",
            " ", "  ", "   ", "    ", "\n", "\t", "!", "?", ".", ",", ":", ";",
        ];
        for (i, word) in common_subwords.iter().enumerate() {
            let log_p = -1.0 - (i as f64 * 0.05);
            trie.insert(word, log_p, Some(i as u32)).unwrap();
        }

        let text = "function benchmark_viterbi_fast_path(input_tokens, max_subwords) {\n    const result = [];\n    for (let i = 0; i < input_tokens.length; i++) {\n        result.push(input_tokens[i]);\n    }\n    return result;\n}\n";
        let bytes = text.as_bytes();
        let chars: Vec<char> = text.chars().collect();

        // Warmup
        for _ in 0..100 {
            let _ = viterbi_decode_ascii(bytes, &trie, true, None).unwrap();
            let _ = viterbi_decode_chars(&chars, &trie, true, None).unwrap();
        }

        let iterations = if cfg!(debug_assertions) { 1_000 } else { 10_000 };
        let start_ascii = Instant::now();
        for _ in 0..iterations {
            let _ = viterbi_decode_ascii(bytes, &trie, true, None).unwrap();
        }
        let elapsed_ascii = start_ascii.elapsed();

        let start_chars = Instant::now();
        for _ in 0..iterations {
            let _ = viterbi_decode_chars(&chars, &trie, true, None).unwrap();
        }
        let elapsed_chars = start_chars.elapsed();

        let total_bytes = (bytes.len() * iterations) as f64;
        let mb_ascii = (total_bytes / (1024.0 * 1024.0)) / elapsed_ascii.as_secs_f64();
        let mb_chars = (total_bytes / (1024.0 * 1024.0)) / elapsed_chars.as_secs_f64();
        let us_per_call_ascii = elapsed_ascii.as_micros() as f64 / iterations as f64;
        let us_per_call_chars = elapsed_chars.as_micros() as f64 / iterations as f64;

        println!("\n=== VITERBI PERFORMANCE BENCHMARK (Issue #96) ===");
        println!("Input size: {} bytes", bytes.len());
        println!("ASCII Viterbi (zero-allocation forward relaxation):");
        println!("  Throughput: {:.2} MB/s", mb_ascii);
        println!("  Latency:    {:.2} us/call", us_per_call_ascii);
        println!("Char Viterbi (zero-allocation forward relaxation):");
        println!("  Throughput: {:.2} MB/s", mb_chars);
        println!("  Latency:    {:.2} us/call", us_per_call_chars);
        println!("=================================================\n");

        let min_mb = if cfg!(debug_assertions) { 0.5 } else { 5.0 };
        assert!(mb_ascii > min_mb, "ASCII throughput should be high");
    }
}

//! ZMQ-based KV-cache event indexer for cache-aware routing.
//!
//! This module decodes the msgpack wire format emitted by SGLang's
//! `ZmqEventPublisher` (see `python/sglang/srt/disaggregation/kv_events.py`).
//!
//! Task 1 scope: pure decoding only. No ZMQ networking, no tokio tasks —
//! those live in later tasks. The wire types here are the contract between
//! the SGLang publisher (Python `msgspec.msgpack` with
//! `array_like=True, omit_defaults=True, gc=False, tag=True`) and the
//! gateway-side indexer.

pub mod hash;
pub mod tree;
pub mod wire;

pub use hash::{compute_block_hashes, sha256_to_i64};
pub use tree::{HashTree, KvWorkerId, MatchResult};
pub use wire::{
    decode_event_batch, BlockRemoved, BlockStored, DecodeError, KvCacheEvent, KvEventBatch,
};

//! ZMQ-based KV-cache event indexer for cache-aware routing.
//!
//! Decodes the msgpack wire format emitted by SGLang's `ZmqEventPublisher`
//! (see `python/sglang/srt/disaggregation/kv_events.py`) and maintains the
//! gateway-side index used for cache-aware request routing.
//!
//! # Submodules
//!
//! - [`wire`]: msgpack types and [`decode_event_batch`] — the contract
//!   between the SGLang publisher (`msgspec.msgpack` with
//!   `array_like=True, omit_defaults=True, gc=False, tag=True`) and this
//!   crate. Pure decoding; no I/O.
//! - [`hash`]: block-hash computation that mirrors SGLang's hashing so
//!   prefix matches line up across publisher and indexer.
//! - [`tree`]: the hash-keyed radix tree ([`HashTree`], [`KvWorkerId`],
//!   [`MatchResult`]) that stores per-worker cached prefixes.
//! - [`subscriber`]: the per-worker, per-DP-rank ZMQ subscriber
//!   ([`KvEventSubscriberRegistry`], [`WorkerEvent`]) — owns SUB sockets
//!   and tokio tasks, forwards decoded batches over an mpsc channel.

pub mod discovery;
pub mod hash;
pub mod subscriber;
pub mod tree;
pub mod wire;

pub use discovery::{fetch_event_config, EventConfig};
pub use hash::{compute_block_hashes, sha256_to_i64};
pub use subscriber::{KvEventSubscriberRegistry, WorkerEvent};
pub use tree::{HashTree, KvWorkerId, MatchResult};
pub use wire::{
    decode_event_batch, BlockRemoved, BlockStored, DecodeError, KvCacheEvent, KvEventBatch,
};

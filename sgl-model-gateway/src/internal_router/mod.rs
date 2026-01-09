//! Internal Router module for pre-tokenized request handling
//!
//! This module provides the gRPC service that accepts pre-tokenized requests
//! from the Tokenizer tier and routes them to workers.
//!
//! # Architecture
//!
//! ```text
//! Tokenizer → InternalRouter → Worker Selection → Worker
//! ```
//!
//! The InternalRouter receives requests with pre-computed token IDs and processed text,
//! allowing it to make routing decisions without performing tokenization.
//!
//! # Testing
//!
//! ## Unit Tests
//!
//! Run the internal router unit tests:
//! ```bash
//! cargo test internal_router::service::tests --lib
//! ```
//!
//! The unit tests cover:
//! - Request validation (empty tokens, empty text, length limits)
//! - Boundary value testing for max token count and text length
//! - Model ID and request ID validation
//!
//! ## Integration Testing (Future)
//!
//! Integration tests for the full Tokenizer → Router → Worker flow should be added
//! to `tests/` once the feature is production-ready. These would test:
//! - End-to-end request flow with mock workers
//! - Streaming response handling
//! - Error propagation between tiers
//! - Load balancing with pre-tokenized requests

pub mod proto;
pub mod service;

pub use proto::internal_router_proto;
pub use service::InternalRouterService;

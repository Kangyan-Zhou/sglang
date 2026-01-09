//! Tokenizer Service module - Entry point for Option 1 architecture
//!
//! This module provides a standalone tokenizer service that:
//! 1. Receives client requests (HTTP/gRPC)
//! 2. Handles authentication (first layer)
//! 3. Tokenizes text and applies chat templates
//! 4. Forwards pre-tokenized requests to the Router tier
//!
//! # Architecture
//!
//! ```text
//! Client → Tokenizer Service → Internal Router → Worker
//! ```
//!
//! # Testing
//!
//! ## Unit Tests
//!
//! Run the tokenizer service unit tests:
//! ```bash
//! cargo test tokenizer_service --lib
//! ```
//!
//! The unit tests cover:
//! - Configuration builder pattern and bind addresses
//! - Chat message to JSON conversion for all message types
//! - Header extraction for routing
//! - Router client connection handling
//!
//! ## Manual Testing
//!
//! To test the tokenizer service manually:
//!
//! 1. Start the internal router (requires workers to be registered)
//! 2. Start the tokenizer service:
//!    ```bash
//!    cargo run --bin tokenizer -- \
//!      --router-url grpc://localhost:50052 \
//!      --model-path /path/to/model \
//!      --http-port 8080
//!    ```
//! 3. Send a request:
//!    ```bash
//!    curl -X POST http://localhost:8080/v1/chat/completions \
//!      -H "Content-Type: application/json" \
//!      -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello"}]}'
//!    ```
//!
//! ## E2E Testing with Kubernetes
//!
//! For end-to-end testing with mock workers, see the K8s manifests and Docker Compose
//! files in `.claude/k8s-manual-testing/`. This includes:
//! - Mock worker deployments that simulate SGLang worker responses
//! - Router tier configuration
//! - Tokenizer tier configuration
//! - Automated test scripts

pub mod client;
pub mod config;
pub mod handlers;
pub mod server;

pub use client::RouterClient;
pub use config::TokenizerServiceConfig;
pub use server::TokenizerServer;

//! Configuration for the Tokenizer Service

use serde::{Deserialize, Serialize};

/// Configuration for the Tokenizer Service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerServiceConfig {
    /// Host to bind the tokenizer service
    #[serde(default = "default_host")]
    pub host: String,

    /// HTTP port for the tokenizer service
    #[serde(default = "default_http_port")]
    pub http_port: u16,

    /// gRPC port for the tokenizer service (optional)
    #[serde(default)]
    pub grpc_port: Option<u16>,

    /// Router service URL (e.g., "router-service:50052")
    pub router_url: String,

    /// Timeout for router RPC calls in milliseconds
    #[serde(default = "default_router_timeout_ms")]
    pub router_timeout_ms: u64,

    /// Connection pool size for router connections
    #[serde(default = "default_router_pool_size")]
    pub router_pool_size: usize,

    /// Model path for tokenizer (HuggingFace model ID or local path)
    pub model_path: String,

    /// Optional chat template path override
    #[serde(default)]
    pub chat_template_path: Option<String>,

    /// Enable tokenizer caching (L0 - exact match)
    #[serde(default)]
    pub enable_cache_l0: bool,

    /// Max entries in L0 cache
    #[serde(default = "default_cache_l0_max_entries")]
    pub cache_l0_max_entries: usize,

    /// Enable tokenizer caching (L1 - prefix match)
    #[serde(default)]
    pub enable_cache_l1: bool,

    /// Max memory for L1 cache in bytes
    #[serde(default = "default_cache_l1_max_memory")]
    pub cache_l1_max_memory: usize,

    /// API key for authentication (optional)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Enable request logging
    #[serde(default = "default_true")]
    pub log_requests: bool,

    /// Max concurrent requests
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_http_port() -> u16 {
    8080
}

fn default_router_timeout_ms() -> u64 {
    30000 // 30 seconds
}

fn default_router_pool_size() -> usize {
    10
}

fn default_cache_l0_max_entries() -> usize {
    10000
}

fn default_cache_l1_max_memory() -> usize {
    100 * 1024 * 1024 // 100 MB
}

fn default_true() -> bool {
    true
}

fn default_max_concurrent_requests() -> usize {
    1000
}

impl Default for TokenizerServiceConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            http_port: default_http_port(),
            grpc_port: None,
            router_url: "localhost:50052".to_string(),
            router_timeout_ms: default_router_timeout_ms(),
            router_pool_size: default_router_pool_size(),
            model_path: String::new(),
            chat_template_path: None,
            enable_cache_l0: false,
            cache_l0_max_entries: default_cache_l0_max_entries(),
            enable_cache_l1: false,
            cache_l1_max_memory: default_cache_l1_max_memory(),
            api_key: None,
            log_requests: default_true(),
            max_concurrent_requests: default_max_concurrent_requests(),
        }
    }
}

impl TokenizerServiceConfig {
    /// Create a new config with required fields
    pub fn new(router_url: String, model_path: String) -> Self {
        Self {
            router_url,
            model_path,
            ..Default::default()
        }
    }

    /// Builder pattern: set host
    pub fn with_host(mut self, host: String) -> Self {
        self.host = host;
        self
    }

    /// Builder pattern: set HTTP port
    pub fn with_http_port(mut self, port: u16) -> Self {
        self.http_port = port;
        self
    }

    /// Builder pattern: set gRPC port
    pub fn with_grpc_port(mut self, port: u16) -> Self {
        self.grpc_port = Some(port);
        self
    }

    /// Builder pattern: set router timeout
    pub fn with_router_timeout_ms(mut self, timeout: u64) -> Self {
        self.router_timeout_ms = timeout;
        self
    }

    /// Builder pattern: enable L0 cache
    pub fn with_cache_l0(mut self, max_entries: usize) -> Self {
        self.enable_cache_l0 = true;
        self.cache_l0_max_entries = max_entries;
        self
    }

    /// Builder pattern: enable L1 cache
    pub fn with_cache_l1(mut self, max_memory: usize) -> Self {
        self.enable_cache_l1 = true;
        self.cache_l1_max_memory = max_memory;
        self
    }

    /// Builder pattern: set API key
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Get the HTTP bind address
    pub fn http_bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.http_port)
    }

    /// Get the gRPC bind address (if configured)
    pub fn grpc_bind_addr(&self) -> Option<String> {
        self.grpc_port.map(|port| format!("{}:{}", self.host, port))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TokenizerServiceConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.http_port, 8080);
        assert!(config.grpc_port.is_none());
        assert_eq!(config.router_timeout_ms, 30000);
    }

    #[test]
    fn test_builder_pattern() {
        let config = TokenizerServiceConfig::new(
            "router-service:50052".to_string(),
            "meta-llama/Llama-2-7b-chat-hf".to_string(),
        )
        .with_host("127.0.0.1".to_string())
        .with_http_port(9000)
        .with_grpc_port(50051)
        .with_cache_l0(5000)
        .with_api_key("secret".to_string());

        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.http_port, 9000);
        assert_eq!(config.grpc_port, Some(50051));
        assert!(config.enable_cache_l0);
        assert_eq!(config.cache_l0_max_entries, 5000);
        assert_eq!(config.api_key, Some("secret".to_string()));
    }

    #[test]
    fn test_bind_addresses() {
        let config = TokenizerServiceConfig::new("router:50052".to_string(), "model".to_string())
            .with_host("0.0.0.0".to_string())
            .with_http_port(8080)
            .with_grpc_port(50051);

        assert_eq!(config.http_bind_addr(), "0.0.0.0:8080");
        assert_eq!(config.grpc_bind_addr(), Some("0.0.0.0:50051".to_string()));
    }
}

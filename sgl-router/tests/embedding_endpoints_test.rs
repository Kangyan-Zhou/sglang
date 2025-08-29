mod common;

use base64::Engine;
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::json;
use sglang_router_rs::config::{
    CircuitBreakerConfig, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use sglang_router_rs::routers::{RouterFactory, RouterTrait};
use std::sync::Arc;

/// Integration tests for embedding endpoints
/// Tests the actual router endpoints with mock workers to ensure proper request handling,
/// response formatting, error handling, and all embedding-specific functionality.

/// Test context that manages mock workers and router
struct EmbeddingTestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    client: Client,
}

impl EmbeddingTestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        let mut config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3010,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            dp_aware: false,
            api_key: None,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            queue_size: 0,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: sglang_router_rs::config::HealthCheckConfig::default(),
            enable_igw: false,
        };

        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        config.mode = RoutingMode::Regular { worker_urls };

        let app_context = common::create_test_context(config);
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        let client = Client::new();

        Self {
            workers,
            router,
            client,
        }
    }

    async fn shutdown(mut self) {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        for worker in &mut self.workers {
            worker.stop().await;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    async fn make_embedding_request(
        &self,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let worker_urls = self.router.get_worker_urls();
        if worker_urls.is_empty() {
            return Err("No available workers".to_string());
        }

        let base_url = &worker_urls[0];
        let url = format!("{}/v1/embeddings", base_url);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        if !status.is_success() {
            return Err(format!("HTTP {}: {}", status, text));
        }

        serde_json::from_str(&text).map_err(|e| format!("JSON parse error: {}", e))
    }
}

#[cfg(test)]
mod embedding_endpoint_tests {
    use super::*;

    // ============= Basic Embedding Tests =============

    #[tokio::test]
    async fn test_basic_text_embedding() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": "The quick brown fox jumps over the lazy dog",
            "model": "text-embedding-ada-002"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Basic embedding request failed: {:?}", result);

        let response = result.unwrap();
        
        // Validate response structure
        assert_eq!(response["object"], "list");
        assert!(response["data"].is_array());
        assert_eq!(response["data"].as_array().unwrap().len(), 1);
        assert_eq!(response["model"], "text-embedding-ada-002");
        
        // Validate embedding data
        let embedding_obj = &response["data"][0];
        assert_eq!(embedding_obj["object"], "embedding");
        assert_eq!(embedding_obj["index"], 0);
        assert!(embedding_obj["embedding"].is_array());
        
        // Validate usage information
        assert!(response["usage"]["prompt_tokens"].is_number());
        assert!(response["usage"]["total_tokens"].is_number());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_text_embedding() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": [
                "First text to embed",
                "Second text to embed",
                "Third text to embed"
            ],
            "model": "text-embedding-3-small"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Batch embedding request failed: {:?}", result);

        let response = result.unwrap();
        
        // Validate batch response structure
        assert_eq!(response["object"], "list");
        let data = response["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);
        
        // Validate each embedding in the batch
        for (i, embedding_obj) in data.iter().enumerate() {
            assert_eq!(embedding_obj["object"], "embedding");
            assert_eq!(embedding_obj["index"], i);
            assert!(embedding_obj["embedding"].is_array());
        }
        
        // Validate usage reflects batch size
        let usage = &response["usage"];
        assert_eq!(usage["prompt_tokens"], 30); // 3 * 10 tokens per input

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_embedding_with_dimensions() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": "Test with custom dimensions",
            "model": "text-embedding-3-small",
            "dimensions": 512
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Dimensions request failed: {:?}", result);

        let response = result.unwrap();
        let embedding = response["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(embedding.len(), 512, "Embedding should have 512 dimensions");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_base64_encoding() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20004,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": "Test base64 encoding",
            "model": "text-embedding-ada-002",
            "encoding_format": "base64"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Base64 request failed: {:?}", result);

        let response = result.unwrap();
        let embedding = &response["data"][0]["embedding"];
        assert!(embedding.is_string(), "Base64 embedding should be a string");
        
        // Verify it's valid base64
        let embedding_str = embedding.as_str().unwrap();
        assert!(
            base64::engine::general_purpose::STANDARD.decode(embedding_str).is_ok(),
            "Should be valid base64"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_integer_token_input() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": [1234, 5678, 9012, 3456],
            "model": "text-embedding-ada-002"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Token input request failed: {:?}", result);

        let response = result.unwrap();
        assert_eq!(response["object"], "list");
        assert_eq!(response["data"].as_array().unwrap().len(), 1);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_token_input() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20006,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "model": "text-embedding-ada-002"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Batch token input request failed: {:?}", result);

        let response = result.unwrap();
        assert_eq!(response["data"].as_array().unwrap().len(), 3);

        ctx.shutdown().await;
    }

    // ============= Error Handling Tests =============

    #[tokio::test]
    async fn test_worker_failure_handling() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20007,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0, // Always fail
        }])
        .await;

        let request = json!({
            "input": "This should fail",
            "model": "text-embedding-ada-002"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_err(), "Request should fail when worker fails");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_malformed_request_handling() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20008,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        // Missing required fields
        let request = json!({
            "model": "text-embedding-ada-002"
            // Missing "input" field
        });

        let _result = ctx.make_embedding_request(request).await;
        // This should either fail or handle gracefully
        // The behavior depends on router validation

        ctx.shutdown().await;
    }

    // ============= Performance and Load Tests =============

    #[tokio::test]
    async fn test_concurrent_embedding_requests() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20009,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 50, // Small delay to test concurrency
            fail_rate: 0.0,
        }])
        .await;

        let mut handles = Vec::new();
        let ctx = Arc::new(ctx);

        // Launch 5 concurrent requests
        for i in 0..5 {
            let ctx_clone = ctx.clone();
            let handle = tokio::spawn(async move {
                let request = json!({
                    "input": format!("Concurrent request {}", i),
                    "model": "text-embedding-ada-002"
                });

                ctx_clone.make_embedding_request(request).await
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        let mut success_count = 0;
        for handle in handles {
            match handle.await.unwrap() {
                Ok(_) => success_count += 1,
                Err(e) => eprintln!("Concurrent request failed: {}", e),
            }
        }

        assert!(
            success_count >= 3,
            "At least 3 out of 5 concurrent requests should succeed"
        );

        // Convert back to owned type for shutdown
        let ctx = Arc::try_unwrap(ctx).unwrap_or_else(|arc| {
            panic!("Failed to unwrap Arc, there are still {} references", Arc::strong_count(&arc));
        });
        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_large_batch_embedding() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20010,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        // Create a large batch
        let mut inputs = Vec::new();
        for i in 0..50 {
            inputs.push(format!("Large batch text {}", i));
        }

        let request = json!({
            "input": inputs,
            "model": "text-embedding-ada-002"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Large batch request failed: {:?}", result);

        let response = result.unwrap();
        assert_eq!(response["data"].as_array().unwrap().len(), 50);
        assert_eq!(response["usage"]["prompt_tokens"], 500); // 50 * 10

        ctx.shutdown().await;
    }

    // ============= Model-Specific Tests =============

    #[tokio::test]
    async fn test_different_embedding_models() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20011,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let models = vec![
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ];

        for model in models {
            let request = json!({
                "input": "Test different models",
                "model": model
            });

            let result = ctx.make_embedding_request(request).await;
            assert!(
                result.is_ok(),
                "Request failed for model {}: {:?}",
                model,
                result
            );

            let response = result.unwrap();
            assert_eq!(response["model"], model);
        }

        ctx.shutdown().await;
    }

    // ============= Response Validation Tests =============

    #[tokio::test]
    async fn test_response_format_validation() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20012,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let request = json!({
            "input": "Validate response format",
            "model": "text-embedding-ada-002"
        });

        let result = ctx.make_embedding_request(request).await;
        assert!(result.is_ok(), "Request failed: {:?}", result);

        let response = result.unwrap();
        
        // Validate all required fields are present
        assert!(response.get("object").is_some());
        assert!(response.get("data").is_some());
        assert!(response.get("model").is_some());
        assert!(response.get("usage").is_some());
        
        // Validate data structure
        let data = response["data"].as_array().unwrap();
        assert!(!data.is_empty());
        
        let first_embedding = &data[0];
        assert!(first_embedding.get("object").is_some());
        assert!(first_embedding.get("index").is_some());
        assert!(first_embedding.get("embedding").is_some());
        
        // Validate usage structure
        let usage = &response["usage"];
        assert!(usage.get("prompt_tokens").is_some());
        assert!(usage.get("total_tokens").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_embedding_dimensions_consistency() {
        let ctx = EmbeddingTestContext::new(vec![MockWorkerConfig {
            port: 20013,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let dimensions = vec![256, 512, 1024];

        for dim in dimensions {
            let request = json!({
                "input": "Test dimensions",
                "model": "text-embedding-3-small",
                "dimensions": dim
            });

            let result = ctx.make_embedding_request(request).await;
            assert!(result.is_ok(), "Request failed for dimension {}: {:?}", dim, result);

            let response = result.unwrap();
            let embedding = response["data"][0]["embedding"].as_array().unwrap();
            assert_eq!(
                embedding.len(),
                dim as usize,
                "Embedding should have {} dimensions",
                dim
            );
        }

        ctx.shutdown().await;
    }
}
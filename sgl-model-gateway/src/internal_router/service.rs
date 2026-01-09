//! Internal Router gRPC service implementation
//!
//! This service accepts pre-tokenized requests from the Tokenizer tier
//! and routes them to workers using the existing routing policies.

use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

/// Maximum allowed token array size (128K tokens)
const MAX_TOKEN_COUNT: usize = 131_072;

/// Maximum allowed string length for processed text (4MB)
const MAX_TEXT_LENGTH: usize = 4 * 1024 * 1024;

/// Maximum request ID length
const MAX_REQUEST_ID_LENGTH: usize = 256;

/// Maximum model ID length
const MAX_MODEL_ID_LENGTH: usize = 256;

/// Default HTTP client timeout for worker requests (in seconds)
const DEFAULT_WORKER_HTTP_TIMEOUT_SECS: u64 = 300;

/// Shared HTTP client for worker requests
static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
        .build()
        .expect("Failed to create worker HTTP client")
});

use super::proto::internal_router_proto::{
    self, internal_router_server::InternalRouter, HealthRequest, HealthResponse,
    PreTokenizedChatRequest, PreTokenizedClassifyRequest, PreTokenizedEmbeddingRequest,
    PreTokenizedGenerateRequest, RouteError, RouteResponse, StreamChunk,
};
use crate::{
    core::{Worker, WorkerRegistry},
    policies::{PolicyRegistry, SelectWorkerInfo},
};

/// Internal Router service for handling pre-tokenized requests
pub struct InternalRouterService {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
}

impl InternalRouterService {
    /// Create a new InternalRouterService
    pub fn new(worker_registry: Arc<WorkerRegistry>, policy_registry: Arc<PolicyRegistry>) -> Self {
        info!("Creating InternalRouterService");
        Self {
            worker_registry,
            policy_registry,
        }
    }

    /// Select a worker based on pre-tokenized request data
    fn select_worker(
        &self,
        model_id: &str,
        token_ids: &[u32],
        processed_text: &str,
        _headers: &std::collections::HashMap<String, String>,
    ) -> Result<Arc<dyn Worker>, Status> {
        // Get workers for the model
        let workers = self.worker_registry.get_by_model(model_id);

        if workers.is_empty() {
            return Err(Status::not_found(format!(
                "No workers available for model: {}",
                model_id
            )));
        }

        // Filter to available workers
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available.is_empty() {
            return Err(Status::unavailable(format!(
                "No available workers for model: {}",
                model_id
            )));
        }

        // Get routing policy for this model
        let policy = self.policy_registry.get_policy_or_default(model_id);

        // Get hash ring for consistent hashing
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Build SelectWorkerInfo with pre-tokenized data
        let info = SelectWorkerInfo {
            request_text: Some(processed_text),
            tokens: Some(token_ids),
            headers: None, // TODO: Convert headers if needed
            hash_ring,
        };

        // Select worker using policy
        match policy.select_worker(&available, &info) {
            Some(idx) => Ok(available[idx].clone()),
            None => {
                error!(
                    model_id = %model_id,
                    available_workers = %available.len(),
                    token_count = %token_ids.len(),
                    policy_type = ?std::any::type_name_of_val(&*policy),
                    "Policy failed to select any worker from available pool"
                );
                Err(Status::internal(format!(
                    "Failed to select worker: policy returned None for model {} with {} available workers",
                    model_id,
                    available.len()
                )))
            }
        }
    }

    /// Forward a request to a worker and return a streaming response
    async fn forward_to_worker_streaming(
        worker: Arc<dyn Worker>,
        route: &str,
        request_body: Vec<u8>,
        request_id: &str,
        is_streaming: bool,
    ) -> Result<ResponseStream, Status> {
        let worker_url = worker.url();
        let full_url = format!("{}{}", worker_url, route);

        debug!(
            request_id = %request_id,
            worker_url = %worker_url,
            route = %route,
            is_streaming = %is_streaming,
            "Forwarding request to worker"
        );

        // Increment worker load
        worker.increment_load();

        // Build request
        let mut request_builder = WORKER_CLIENT
            .post(&full_url)
            .header("Content-Type", "application/json")
            .body(request_body);

        // Add API key if present
        if let Some(api_key) = worker.api_key() {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        // Send request
        let response = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                worker.decrement_load();
                worker.record_outcome(false);
                error!(
                    request_id = %request_id,
                    worker_url = %worker_url,
                    error = %e,
                    "Failed to send request to worker"
                );
                return Err(Status::unavailable(format!(
                    "Failed to connect to worker: {}",
                    e
                )));
            }
        };

        let status = response.status();
        if !status.is_success() {
            worker.decrement_load();
            worker.record_outcome(false);
            let body = response.text().await.unwrap_or_default();
            error!(
                request_id = %request_id,
                worker_url = %worker_url,
                status = %status,
                body = %body,
                "Worker returned error status"
            );
            return Err(Status::internal(format!(
                "Worker returned error: {} - {}",
                status, body
            )));
        }

        // Record success
        worker.record_outcome(true);

        if is_streaming {
            // Return streaming response
            let worker_clone = worker.clone();
            let request_id_owned = request_id.to_string();
            let byte_stream = response.bytes_stream();

            let stream = byte_stream.map(move |result| match result {
                Ok(bytes) => Ok(RouteResponse {
                    response: Some(internal_router_proto::route_response::Response::Chunk(
                        StreamChunk {
                            data: bytes.to_vec(),
                            is_final: false,
                        },
                    )),
                }),
                Err(e) => {
                    error!(
                        request_id = %request_id_owned,
                        error = %e,
                        "Error reading stream from worker"
                    );
                    Ok(RouteResponse {
                        response: Some(internal_router_proto::route_response::Response::Error(
                            RouteError {
                                code: "stream_error".to_string(),
                                message: format!("Error reading stream: {}", e),
                                http_status: 500,
                                details: String::new(),
                            },
                        )),
                    })
                }
            });

            // Add final chunk marker and decrement load when stream ends
            let final_stream = stream.chain(futures::stream::once(async move {
                worker_clone.decrement_load();
                Ok(RouteResponse {
                    response: Some(internal_router_proto::route_response::Response::Chunk(
                        StreamChunk {
                            data: Vec::new(),
                            is_final: true,
                        },
                    )),
                })
            }));

            Ok(Box::pin(final_stream))
        } else {
            // Non-streaming: collect full response
            let body = match response.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    worker.decrement_load();
                    error!(
                        request_id = %request_id,
                        error = %e,
                        "Failed to read response body from worker"
                    );
                    return Err(Status::internal(format!("Failed to read response: {}", e)));
                }
            };

            worker.decrement_load();

            // Return as complete response
            let stream = futures::stream::once(async move {
                Ok(RouteResponse {
                    response: Some(
                        internal_router_proto::route_response::Response::CompleteResponse(
                            body.to_vec(),
                        ),
                    ),
                })
            });

            Ok(Box::pin(stream))
        }
    }

    /// Forward a non-streaming request to a worker
    async fn forward_to_worker_unary(
        worker: Arc<dyn Worker>,
        route: &str,
        request_body: Vec<u8>,
        request_id: &str,
    ) -> Result<Bytes, Status> {
        let worker_url = worker.url();
        let full_url = format!("{}{}", worker_url, route);

        debug!(
            request_id = %request_id,
            worker_url = %worker_url,
            route = %route,
            "Forwarding unary request to worker"
        );

        // Increment worker load
        worker.increment_load();

        // Build request
        let mut request_builder = WORKER_CLIENT
            .post(&full_url)
            .header("Content-Type", "application/json")
            .body(request_body);

        // Add API key if present
        if let Some(api_key) = worker.api_key() {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        // Send request
        let response = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                worker.decrement_load();
                worker.record_outcome(false);
                error!(
                    request_id = %request_id,
                    worker_url = %worker_url,
                    error = %e,
                    "Failed to send request to worker"
                );
                return Err(Status::unavailable(format!(
                    "Failed to connect to worker: {}",
                    e
                )));
            }
        };

        let status = response.status();
        if !status.is_success() {
            worker.decrement_load();
            worker.record_outcome(false);
            let body = response.text().await.unwrap_or_default();
            error!(
                request_id = %request_id,
                worker_url = %worker_url,
                status = %status,
                body = %body,
                "Worker returned error status"
            );
            return Err(Status::internal(format!(
                "Worker returned error: {} - {}",
                status, body
            )));
        }

        // Record success
        worker.record_outcome(true);

        // Read response body
        let body = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                worker.decrement_load();
                error!(
                    request_id = %request_id,
                    error = %e,
                    "Failed to read response body from worker"
                );
                return Err(Status::internal(format!("Failed to read response: {}", e)));
            }
        };

        worker.decrement_load();
        Ok(body)
    }

    /// Create an error response
    #[allow(dead_code)]
    fn error_response(code: &str, message: &str, http_status: i32) -> RouteResponse {
        RouteResponse {
            response: Some(internal_router_proto::route_response::Response::Error(
                RouteError {
                    code: code.to_string(),
                    message: message.to_string(),
                    http_status,
                    details: String::new(),
                },
            )),
        }
    }

    /// Validate common request fields
    fn validate_request(
        request_id: &str,
        model_id: &str,
        tokenization: &internal_router_proto::TokenizationResult,
    ) -> Result<(), Status> {
        // Validate request ID
        if request_id.len() > MAX_REQUEST_ID_LENGTH {
            warn!(
                "Request ID too long: {} > {}",
                request_id.len(),
                MAX_REQUEST_ID_LENGTH
            );
            return Err(Status::invalid_argument(format!(
                "Request ID exceeds maximum length of {} characters",
                MAX_REQUEST_ID_LENGTH
            )));
        }

        // Validate model ID
        if model_id.is_empty() {
            return Err(Status::invalid_argument("Model ID cannot be empty"));
        }
        if model_id.len() > MAX_MODEL_ID_LENGTH {
            warn!(
                "Model ID too long: {} > {}",
                model_id.len(),
                MAX_MODEL_ID_LENGTH
            );
            return Err(Status::invalid_argument(format!(
                "Model ID exceeds maximum length of {} characters",
                MAX_MODEL_ID_LENGTH
            )));
        }

        // Validate token_ids is not empty
        if tokenization.token_ids.is_empty() {
            warn!(
                request_id = %request_id,
                model_id = %model_id,
                "Received request with empty token_ids"
            );
            return Err(Status::invalid_argument(
                "Token IDs cannot be empty - request must have at least one token",
            ));
        }

        // Validate token count
        if tokenization.token_ids.len() > MAX_TOKEN_COUNT {
            warn!(
                "Token count too high: {} > {}",
                tokenization.token_ids.len(),
                MAX_TOKEN_COUNT
            );
            return Err(Status::invalid_argument(format!(
                "Token count exceeds maximum of {} tokens",
                MAX_TOKEN_COUNT
            )));
        }

        // Validate processed_text is not empty
        if tokenization.processed_text.is_empty() {
            warn!(
                request_id = %request_id,
                model_id = %model_id,
                "Received request with empty processed_text"
            );
            return Err(Status::invalid_argument("Processed text cannot be empty"));
        }

        // Validate processed text length
        if tokenization.processed_text.len() > MAX_TEXT_LENGTH {
            warn!(
                "Processed text too long: {} > {}",
                tokenization.processed_text.len(),
                MAX_TEXT_LENGTH
            );
            return Err(Status::invalid_argument(format!(
                "Processed text exceeds maximum length of {} bytes",
                MAX_TEXT_LENGTH
            )));
        }

        Ok(())
    }
}

type ResponseStream = Pin<Box<dyn Stream<Item = Result<RouteResponse, Status>> + Send>>;

#[tonic::async_trait]
impl InternalRouter for InternalRouterService {
    type RouteChatCompletionStream = ResponseStream;
    type RouteGenerateStream = ResponseStream;

    async fn route_chat_completion(
        &self,
        request: Request<PreTokenizedChatRequest>,
    ) -> Result<Response<Self::RouteChatCompletionStream>, Status> {
        let req = request.into_inner();
        debug!(
            "Received pre-tokenized chat request: id={}, model={}",
            req.request_id, req.model_id
        );

        // Extract tokenization result
        let tokenization = req
            .tokenization
            .ok_or_else(|| Status::invalid_argument("Missing tokenization result in request"))?;

        // Validate request
        Self::validate_request(&req.request_id, &req.model_id, &tokenization)?;

        // Select worker based on pre-tokenized data
        let worker = self.select_worker(
            &req.model_id,
            &tokenization.token_ids,
            &tokenization.processed_text,
            &req.headers,
        )?;

        debug!(
            "Selected worker {} for request {}",
            worker.url(),
            req.request_id
        );

        // Forward request to worker
        let stream = Self::forward_to_worker_streaming(
            worker,
            "/v1/chat/completions",
            req.original_request_body,
            &req.request_id,
            req.stream,
        )
        .await?;

        Ok(Response::new(stream))
    }

    async fn route_generate(
        &self,
        request: Request<PreTokenizedGenerateRequest>,
    ) -> Result<Response<Self::RouteGenerateStream>, Status> {
        let req = request.into_inner();
        debug!(
            "Received pre-tokenized generate request: id={}, model={}",
            req.request_id, req.model_id
        );

        // Extract tokenization result
        let tokenization = req
            .tokenization
            .ok_or_else(|| Status::invalid_argument("Missing tokenization result in request"))?;

        // Validate request
        Self::validate_request(&req.request_id, &req.model_id, &tokenization)?;

        // Select worker based on pre-tokenized data
        let worker = self.select_worker(
            &req.model_id,
            &tokenization.token_ids,
            &tokenization.processed_text,
            &req.headers,
        )?;

        debug!(
            "Selected worker {} for request {}",
            worker.url(),
            req.request_id
        );

        // Forward request to worker
        let stream = Self::forward_to_worker_streaming(
            worker,
            "/generate",
            req.original_request_body,
            &req.request_id,
            req.stream,
        )
        .await?;

        Ok(Response::new(stream))
    }

    async fn route_embedding(
        &self,
        request: Request<PreTokenizedEmbeddingRequest>,
    ) -> Result<Response<RouteResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received pre-tokenized embedding request: id={}, model={}",
            req.request_id, req.model_id
        );

        // Extract tokenization result
        let tokenization = req
            .tokenization
            .ok_or_else(|| Status::invalid_argument("Missing tokenization result in request"))?;

        // Validate request
        Self::validate_request(&req.request_id, &req.model_id, &tokenization)?;

        // Select worker based on pre-tokenized data
        let worker = self.select_worker(
            &req.model_id,
            &tokenization.token_ids,
            &tokenization.processed_text,
            &req.headers,
        )?;

        debug!(
            "Selected worker {} for request {}",
            worker.url(),
            req.request_id
        );

        // Forward request to worker (embeddings are non-streaming)
        let body = Self::forward_to_worker_unary(
            worker,
            "/v1/embeddings",
            req.original_request_body,
            &req.request_id,
        )
        .await?;

        Ok(Response::new(RouteResponse {
            response: Some(
                internal_router_proto::route_response::Response::CompleteResponse(body.to_vec()),
            ),
        }))
    }

    async fn route_classify(
        &self,
        request: Request<PreTokenizedClassifyRequest>,
    ) -> Result<Response<RouteResponse>, Status> {
        let req = request.into_inner();
        debug!(
            "Received pre-tokenized classify request: id={}, model={}",
            req.request_id, req.model_id
        );

        // Extract tokenization result
        let tokenization = req
            .tokenization
            .ok_or_else(|| Status::invalid_argument("Missing tokenization result in request"))?;

        // Validate request
        Self::validate_request(&req.request_id, &req.model_id, &tokenization)?;

        // Select worker based on pre-tokenized data
        let worker = self.select_worker(
            &req.model_id,
            &tokenization.token_ids,
            &tokenization.processed_text,
            &req.headers,
        )?;

        debug!(
            "Selected worker {} for request {}",
            worker.url(),
            req.request_id
        );

        // Forward request to worker (classify is non-streaming)
        let body = Self::forward_to_worker_unary(
            worker,
            "/classify",
            req.original_request_body,
            &req.request_id,
        )
        .await?;

        Ok(Response::new(RouteResponse {
            response: Some(
                internal_router_proto::route_response::Response::CompleteResponse(body.to_vec()),
            ),
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let stats = self.worker_registry.stats();

        Ok(Response::new(HealthResponse {
            healthy: stats.healthy_workers > 0,
            message: format!(
                "{}/{} workers healthy",
                stats.healthy_workers, stats.total_workers
            ),
            available_workers: stats.healthy_workers as i32,
            total_workers: stats.total_workers as i32,
            mesh_connected: false, // TODO: Add mesh status when integrated
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_request_valid() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1, 2, 3, 4, 5],
            processed_text: "Hello world".to_string(),
            token_count: 5,
            original_text: "Hello world".to_string(),
        };

        let result = InternalRouterService::validate_request(
            "req-123",
            "meta-llama/Llama-2-7b",
            &tokenization,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_empty_model_id() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1, 2, 3],
            processed_text: "test".to_string(),
            token_count: 3,
            original_text: "test".to_string(),
        };

        let result = InternalRouterService::validate_request("req-123", "", &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("Model ID cannot be empty"));
    }

    #[test]
    fn test_validate_request_model_id_too_long() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1, 2, 3],
            processed_text: "test".to_string(),
            token_count: 3,
            original_text: "test".to_string(),
        };

        let long_model_id = "a".repeat(MAX_MODEL_ID_LENGTH + 1);
        let result =
            InternalRouterService::validate_request("req-123", &long_model_id, &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("Model ID exceeds maximum length"));
    }

    #[test]
    fn test_validate_request_request_id_too_long() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1, 2, 3],
            processed_text: "test".to_string(),
            token_count: 3,
            original_text: "test".to_string(),
        };

        let long_request_id = "r".repeat(MAX_REQUEST_ID_LENGTH + 1);
        let result =
            InternalRouterService::validate_request(&long_request_id, "model-id", &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("Request ID exceeds maximum length"));
    }

    #[test]
    fn test_validate_request_too_many_tokens() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![0u32; MAX_TOKEN_COUNT + 1],
            processed_text: "test".to_string(),
            token_count: (MAX_TOKEN_COUNT + 1) as i32,
            original_text: "test".to_string(),
        };

        let result = InternalRouterService::validate_request("req-123", "model-id", &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("Token count exceeds maximum"));
    }

    #[test]
    fn test_validate_request_text_too_long() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1, 2, 3],
            processed_text: "x".repeat(MAX_TEXT_LENGTH + 1),
            token_count: 3,
            original_text: "test".to_string(),
        };

        let result = InternalRouterService::validate_request("req-123", "model-id", &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .message()
            .contains("Processed text exceeds maximum length"));
    }

    #[test]
    fn test_validate_request_boundary_values() {
        // Test with maximum allowed values
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![0u32; MAX_TOKEN_COUNT],
            processed_text: "x".repeat(MAX_TEXT_LENGTH),
            token_count: MAX_TOKEN_COUNT as i32,
            original_text: "test".to_string(),
        };

        let max_request_id = "r".repeat(MAX_REQUEST_ID_LENGTH);
        let max_model_id = "m".repeat(MAX_MODEL_ID_LENGTH);
        let result =
            InternalRouterService::validate_request(&max_request_id, &max_model_id, &tokenization);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_request_empty_token_ids() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![],
            processed_text: "test".to_string(),
            token_count: 0,
            original_text: "test".to_string(),
        };

        let result = InternalRouterService::validate_request("req-123", "model-id", &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("Token IDs cannot be empty"));
    }

    #[test]
    fn test_validate_request_empty_processed_text() {
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1, 2, 3],
            processed_text: "".to_string(),
            token_count: 3,
            original_text: "test".to_string(),
        };

        let result = InternalRouterService::validate_request("req-123", "model-id", &tokenization);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message().contains("Processed text cannot be empty"));
    }

    #[test]
    fn test_validate_request_single_token() {
        // Minimum valid case: single token with non-empty text
        let tokenization = internal_router_proto::TokenizationResult {
            token_ids: vec![1],
            processed_text: "x".to_string(),
            token_count: 1,
            original_text: "x".to_string(),
        };

        let result = InternalRouterService::validate_request("req-123", "model-id", &tokenization);
        assert!(result.is_ok());
    }
}

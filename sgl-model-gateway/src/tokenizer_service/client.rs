//! Router Client for forwarding pre-tokenized requests to the Router tier

use std::time::{Duration, Instant};

use futures::StreamExt;
use tokio::time::timeout;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};

use crate::internal_router::proto::internal_router_proto::{
    self, internal_router_client::InternalRouterClient, HealthRequest, PreTokenizedChatRequest,
    PreTokenizedClassifyRequest, PreTokenizedEmbeddingRequest, PreTokenizedGenerateRequest,
    RouteResponse, TokenizationResult,
};

/// Client for communicating with the Internal Router service
#[derive(Clone)]
pub struct RouterClient {
    client: InternalRouterClient<Channel>,
    timeout: Duration,
}

impl RouterClient {
    /// Connect to the router service
    pub async fn connect(
        endpoint: &str,
        timeout_ms: u64,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("Connecting to router service at {}", endpoint);

        // Convert grpc:// to http:// for tonic
        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{}", addr)
        } else if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            format!("http://{}", endpoint)
        } else {
            endpoint.to_string()
        };

        let channel = Channel::from_shared(http_endpoint)?
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .http2_adaptive_window(true)
            .connect()
            .await?;

        let client = InternalRouterClient::new(channel);

        Ok(Self {
            client,
            timeout: Duration::from_millis(timeout_ms),
        })
    }

    /// Forward a pre-tokenized chat completion request to the router
    pub async fn route_chat_completion(
        &self,
        request_id: String,
        model_id: String,
        token_ids: Vec<u32>,
        processed_text: String,
        original_text: String,
        original_request_body: Vec<u8>,
        headers: std::collections::HashMap<String, String>,
        stream: bool,
    ) -> Result<RouterResponseStream, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            "Forwarding chat completion to router: id={}, model={}, tokens={}",
            request_id,
            model_id,
            token_ids.len()
        );

        let token_count = token_ids.len() as i32;
        let request = PreTokenizedChatRequest {
            request_id: request_id.clone(),
            model_id: model_id.clone(),
            tokenization: Some(TokenizationResult {
                token_ids,
                processed_text,
                token_count,
                original_text,
            }),
            original_request_body,
            headers,
            stream,
        };

        let mut client = self.client.clone();
        let start = Instant::now();
        let response = timeout(self.timeout, client.route_chat_completion(request))
            .await
            .map_err(|_| {
                let elapsed = start.elapsed();
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    timeout_ms = %self.timeout.as_millis(),
                    elapsed_ms = %elapsed.as_millis(),
                    "Router request timeout"
                );
                format!(
                    "Router request timeout after {}ms (request_id={}, model={})",
                    elapsed.as_millis(),
                    request_id,
                    model_id
                )
            })?
            .map_err(|e| {
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    grpc_code = ?e.code(),
                    grpc_message = %e.message(),
                    "gRPC error from router"
                );
                format!(
                    "gRPC error: code={:?}, message={}, request_id={}, model={}",
                    e.code(),
                    e.message(),
                    request_id,
                    model_id
                )
            })?;

        Ok(RouterResponseStream::new(response.into_inner()))
    }

    /// Forward a pre-tokenized generate request to the router
    pub async fn route_generate(
        &self,
        request_id: String,
        model_id: String,
        token_ids: Vec<u32>,
        processed_text: String,
        original_text: String,
        original_request_body: Vec<u8>,
        headers: std::collections::HashMap<String, String>,
        stream: bool,
    ) -> Result<RouterResponseStream, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            "Forwarding generate to router: id={}, model={}, tokens={}",
            request_id,
            model_id,
            token_ids.len()
        );

        let token_count = token_ids.len() as i32;
        let request = PreTokenizedGenerateRequest {
            request_id: request_id.clone(),
            model_id: model_id.clone(),
            tokenization: Some(TokenizationResult {
                token_ids,
                processed_text,
                token_count,
                original_text,
            }),
            original_request_body,
            headers,
            stream,
        };

        let mut client = self.client.clone();
        let start = Instant::now();
        let response = timeout(self.timeout, client.route_generate(request))
            .await
            .map_err(|_| {
                let elapsed = start.elapsed();
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    timeout_ms = %self.timeout.as_millis(),
                    elapsed_ms = %elapsed.as_millis(),
                    "Router request timeout"
                );
                format!(
                    "Router request timeout after {}ms (request_id={}, model={})",
                    elapsed.as_millis(),
                    request_id,
                    model_id
                )
            })?
            .map_err(|e| {
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    grpc_code = ?e.code(),
                    grpc_message = %e.message(),
                    "gRPC error from router"
                );
                format!(
                    "gRPC error: code={:?}, message={}, request_id={}, model={}",
                    e.code(),
                    e.message(),
                    request_id,
                    model_id
                )
            })?;

        Ok(RouterResponseStream::new(response.into_inner()))
    }

    /// Forward a pre-tokenized embedding request to the router
    pub async fn route_embedding(
        &self,
        request_id: String,
        model_id: String,
        token_ids: Vec<u32>,
        processed_text: String,
        original_text: String,
        original_request_body: Vec<u8>,
        headers: std::collections::HashMap<String, String>,
    ) -> Result<RouteResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            "Forwarding embedding to router: id={}, model={}, tokens={}",
            request_id,
            model_id,
            token_ids.len()
        );

        let token_count = token_ids.len() as i32;
        let request = PreTokenizedEmbeddingRequest {
            request_id: request_id.clone(),
            model_id: model_id.clone(),
            tokenization: Some(TokenizationResult {
                token_ids,
                processed_text,
                token_count,
                original_text,
            }),
            original_request_body,
            headers,
        };

        let mut client = self.client.clone();
        let start = Instant::now();
        let response = timeout(self.timeout, client.route_embedding(request))
            .await
            .map_err(|_| {
                let elapsed = start.elapsed();
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    timeout_ms = %self.timeout.as_millis(),
                    elapsed_ms = %elapsed.as_millis(),
                    "Router request timeout"
                );
                format!(
                    "Router request timeout after {}ms (request_id={}, model={})",
                    elapsed.as_millis(),
                    request_id,
                    model_id
                )
            })?
            .map_err(|e| {
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    grpc_code = ?e.code(),
                    grpc_message = %e.message(),
                    "gRPC error from router"
                );
                format!(
                    "gRPC error: code={:?}, message={}, request_id={}, model={}",
                    e.code(),
                    e.message(),
                    request_id,
                    model_id
                )
            })?;

        Ok(response.into_inner())
    }

    /// Forward a pre-tokenized classify request to the router
    pub async fn route_classify(
        &self,
        request_id: String,
        model_id: String,
        token_ids: Vec<u32>,
        processed_text: String,
        original_text: String,
        original_request_body: Vec<u8>,
        headers: std::collections::HashMap<String, String>,
    ) -> Result<RouteResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            "Forwarding classify to router: id={}, model={}, tokens={}",
            request_id,
            model_id,
            token_ids.len()
        );

        let token_count = token_ids.len() as i32;
        let request = PreTokenizedClassifyRequest {
            request_id: request_id.clone(),
            model_id: model_id.clone(),
            tokenization: Some(TokenizationResult {
                token_ids,
                processed_text,
                token_count,
                original_text,
            }),
            original_request_body,
            headers,
        };

        let mut client = self.client.clone();
        let start = Instant::now();
        let response = timeout(self.timeout, client.route_classify(request))
            .await
            .map_err(|_| {
                let elapsed = start.elapsed();
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    timeout_ms = %self.timeout.as_millis(),
                    elapsed_ms = %elapsed.as_millis(),
                    "Router request timeout"
                );
                format!(
                    "Router request timeout after {}ms (request_id={}, model={})",
                    elapsed.as_millis(),
                    request_id,
                    model_id
                )
            })?
            .map_err(|e| {
                error!(
                    request_id = %request_id,
                    model_id = %model_id,
                    grpc_code = ?e.code(),
                    grpc_message = %e.message(),
                    "gRPC error from router"
                );
                format!(
                    "gRPC error: code={:?}, message={}, request_id={}, model={}",
                    e.code(),
                    e.message(),
                    request_id,
                    model_id
                )
            })?;

        Ok(response.into_inner())
    }

    /// Check router health
    pub async fn health_check(
        &self,
    ) -> Result<internal_router_proto::HealthResponse, Box<dyn std::error::Error + Send + Sync>>
    {
        debug!("Checking router health");

        let mut client = self.client.clone();
        let start = Instant::now();
        let response = timeout(self.timeout, client.health(HealthRequest {}))
            .await
            .map_err(|_| {
                let elapsed = start.elapsed();
                error!(
                    timeout_ms = %self.timeout.as_millis(),
                    elapsed_ms = %elapsed.as_millis(),
                    "Router health check timeout"
                );
                format!(
                    "Router health check timeout after {}ms",
                    elapsed.as_millis()
                )
            })?
            .map_err(|e| {
                error!(
                    grpc_code = ?e.code(),
                    grpc_message = %e.message(),
                    "gRPC error from router health check"
                );
                format!(
                    "gRPC health check error: code={:?}, message={}",
                    e.code(),
                    e.message()
                )
            })?;

        Ok(response.into_inner())
    }
}

/// Wrapper for streaming responses from the router
pub struct RouterResponseStream {
    inner: tonic::Streaming<RouteResponse>,
}

impl RouterResponseStream {
    fn new(stream: tonic::Streaming<RouteResponse>) -> Self {
        Self { inner: stream }
    }

    /// Get the next response chunk
    pub async fn next(&mut self) -> Option<Result<RouteResponse, tonic::Status>> {
        self.inner.next().await
    }

    /// Collect all responses into a single response
    pub async fn collect(mut self) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let mut data = Vec::new();

        while let Some(result) = self.next().await {
            match result {
                Ok(response) => {
                    use internal_router_proto::route_response::Response;
                    match response.response {
                        Some(Response::Chunk(chunk)) => {
                            data.extend_from_slice(&chunk.data);
                            if chunk.is_final {
                                break;
                            }
                        }
                        Some(Response::CompleteResponse(complete)) => {
                            return Ok(complete);
                        }
                        Some(Response::Error(err)) => {
                            return Err(
                                format!("Router error: {} - {}", err.code, err.message).into()
                            );
                        }
                        None => {
                            warn!("Received RouteResponse with empty response variant");
                        }
                    }
                }
                Err(status) => {
                    return Err(format!("gRPC error: {}", status).into());
                }
            }
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_invalid_endpoint() {
        let result = RouterClient::connect("invalid://endpoint", 1000).await;
        assert!(result.is_err());
    }
}

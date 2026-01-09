//! Tokenizer Service Server
//!
//! HTTP server that exposes OpenAI-compatible endpoints for the tokenizer service.

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

use super::{
    client::RouterClient,
    config::TokenizerServiceConfig,
    handlers::{self, HandlerState},
};
use crate::tokenizer::{factory::create_tokenizer_with_chat_template, TokenizerRegistry};

/// Tokenizer Service Server
pub struct TokenizerServer {
    config: TokenizerServiceConfig,
    router_client: RouterClient,
    tokenizer_registry: Arc<TokenizerRegistry>,
}

impl TokenizerServer {
    /// Create a new TokenizerServer
    pub async fn new(
        config: TokenizerServiceConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing TokenizerServer");

        // Connect to router
        let router_client = RouterClient::connect(&config.router_url, config.router_timeout_ms)
            .await
            .map_err(|e| {
                error!(
                    router_url = %config.router_url,
                    timeout_ms = %config.router_timeout_ms,
                    error = %e,
                    "Failed to connect to router service"
                );
                format!(
                    "Failed to connect to router at {}: {}",
                    config.router_url, e
                )
            })?;

        // Create tokenizer registry
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());

        // Load tokenizer for the configured model
        if !config.model_path.is_empty() {
            info!("Loading tokenizer from: {}", config.model_path);
            let model_path = config.model_path.clone();
            let model_path_for_closure = model_path.clone();
            let chat_template_path = config.chat_template_path.clone();
            let id = TokenizerRegistry::generate_id();

            tokenizer_registry
                .load(&id, &model_path, &model_path, || async move {
                    create_tokenizer_with_chat_template(
                        &model_path_for_closure,
                        chat_template_path.as_deref(),
                    )
                    .map_err(|e| e.to_string())
                })
                .await
                .map_err(|e| {
                    error!(
                        model_path = %model_path,
                        chat_template_path = ?config.chat_template_path,
                        error = %e,
                        "Failed to load tokenizer"
                    );
                    format!("Failed to load tokenizer from {}: {}", model_path, e)
                })?;
        }

        Ok(Self {
            config,
            router_client,
            tokenizer_registry,
        })
    }

    /// Create with pre-initialized components (for testing or custom setup)
    pub fn with_components(
        config: TokenizerServiceConfig,
        router_client: RouterClient,
        tokenizer_registry: Arc<TokenizerRegistry>,
    ) -> Self {
        Self {
            config,
            router_client,
            tokenizer_registry,
        }
    }

    /// Build the Axum router
    fn build_router(&self) -> Router {
        let state = Arc::new(HandlerState::new(
            self.router_client.clone(),
            self.tokenizer_registry.clone(),
            Some(self.config.model_path.clone()),
        ));

        // CORS configuration
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        Router::new()
            // Health endpoints
            .route("/health", get(handlers::health_handler))
            .route("/liveness", get(handlers::liveness_handler))
            .route("/readiness", get(handlers::health_handler))
            // OpenAI-compatible endpoints
            .route(
                "/v1/chat/completions",
                post(handlers::chat_completion_handler),
            )
            // TODO: Add more endpoints
            // .route("/v1/completions", post(handlers::completion_handler))
            // .route("/v1/embeddings", post(handlers::embedding_handler))
            .with_state(state)
            .layer(cors)
    }

    /// Start the HTTP server
    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr = self.config.http_bind_addr();
        info!("Starting TokenizerServer on {}", addr);

        let router = self.build_router();

        let listener = TcpListener::bind(&addr).await?;
        info!("TokenizerServer listening on {}", addr);

        axum::serve(listener, router)
            .await
            .map_err(|e| format!("Server error: {}", e))?;

        Ok(())
    }

    /// Start both HTTP and gRPC servers (if gRPC port configured)
    pub async fn serve_all(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let http_addr = self.config.http_bind_addr();
        let grpc_addr = self.config.grpc_bind_addr();

        info!("Starting TokenizerServer HTTP on {}", http_addr);
        if let Some(ref grpc) = grpc_addr {
            info!("Starting TokenizerServer gRPC on {}", grpc);
        }

        let router = self.build_router();

        // Start HTTP server
        let http_listener = TcpListener::bind(&http_addr).await?;

        // For now, just serve HTTP
        // TODO: Add gRPC server when needed
        axum::serve(http_listener, router)
            .await
            .map_err(|e| format!("Server error: {}", e))?;

        Ok(())
    }

    /// Get the router client for testing
    pub fn router_client(&self) -> &RouterClient {
        &self.router_client
    }

    /// Get the tokenizer registry
    pub fn tokenizer_registry(&self) -> &Arc<TokenizerRegistry> {
        &self.tokenizer_registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_bind_addr() {
        let config = TokenizerServiceConfig::new("router:50052".to_string(), "model".to_string())
            .with_http_port(8080);

        assert_eq!(config.http_bind_addr(), "0.0.0.0:8080");
    }
}

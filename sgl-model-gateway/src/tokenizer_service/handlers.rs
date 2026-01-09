//! Request handlers for the Tokenizer Service
//!
//! These handlers receive client requests, perform tokenization,
//! and forward to the Router tier.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use serde_json::Value;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::client::RouterClient;
use crate::internal_router::proto::internal_router_proto::route_response;
use crate::{
    protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    tokenizer::{registry::TokenizerEntry, TokenizerRegistry},
};

/// Shared state for handlers
pub struct HandlerState {
    pub router_client: RouterClient,
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub default_model: Option<String>,
}

impl HandlerState {
    pub fn new(
        router_client: RouterClient,
        tokenizer_registry: Arc<TokenizerRegistry>,
        default_model: Option<String>,
    ) -> Self {
        Self {
            router_client,
            tokenizer_registry,
            default_model,
        }
    }

    /// Get tokenizer entry for a model
    fn get_tokenizer_entry(&self, model_id: &str) -> Result<TokenizerEntry, String> {
        self.tokenizer_registry
            .get_by_name(model_id)
            .ok_or_else(|| format!("Tokenizer not found for model: {}", model_id))
    }
}

/// Health check endpoint
pub async fn health_handler(State(state): State<Arc<HandlerState>>) -> impl IntoResponse {
    match state.router_client.health_check().await {
        Ok(health) => {
            let status = if health.healthy {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (
                status,
                Json(serde_json::json!({
                    "status": if health.healthy { "healthy" } else { "unhealthy" },
                    "router": {
                        "connected": true,
                        "healthy": health.healthy,
                        "message": health.message,
                        "available_workers": health.available_workers,
                        "total_workers": health.total_workers,
                    }
                })),
            )
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "unhealthy",
                "router": {
                    "connected": false,
                    "error": e.to_string(),
                }
            })),
        ),
    }
}

/// Liveness probe
pub async fn liveness_handler() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Chat completion endpoint
pub async fn chat_completion_handler(
    State(state): State<Arc<HandlerState>>,
    headers: HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let model_id = body.model.clone();

    debug!(
        "Processing chat completion: id={}, model={}",
        request_id, model_id
    );

    // Get tokenizer entry for this model
    let tokenizer_entry = match state.get_tokenizer_entry(&model_id) {
        Ok(t) => t,
        Err(e) => {
            error!("Tokenizer not found: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": e,
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })),
            )
                .into_response();
        }
    };

    // Apply chat template and tokenize
    let (processed_text, token_ids) = match process_chat_messages(&body, &tokenizer_entry) {
        Ok(result) => result,
        Err(e) => {
            error!("Tokenization failed: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Tokenization failed: {}", e),
                        "type": "server_error",
                        "code": "tokenization_error"
                    }
                })),
            )
                .into_response();
        }
    };

    debug!(
        "Tokenized request: id={}, tokens={}",
        request_id,
        token_ids.len()
    );

    // Serialize original request body
    let original_body = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize request: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to serialize request",
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Extract headers to forward
    let forward_headers = extract_forward_headers(&headers);

    let is_streaming = body.stream;

    // Forward to router
    match state
        .router_client
        .route_chat_completion(
            request_id.clone(),
            model_id.clone(),
            token_ids,
            processed_text.clone(),
            processed_text, // original_text same as processed for now
            original_body,
            forward_headers,
            body.stream,
        )
        .await
    {
        Ok(mut router_stream) => {
            if is_streaming {
                // Create a channel to forward streaming responses
                let (tx, rx) =
                    tokio::sync::mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
                let request_id_clone = request_id.clone();

                // Spawn task to read from router stream and forward to client
                tokio::spawn(async move {
                    while let Some(result) = router_stream.next().await {
                        match result {
                            Ok(response) => {
                                match response.response {
                                    Some(route_response::Response::Chunk(chunk)) => {
                                        // Forward chunk data as SSE event
                                        if !chunk.data.is_empty() {
                                            if tx.send(Ok(Bytes::from(chunk.data))).is_err() {
                                                debug!(
                                                    request_id = %request_id_clone,
                                                    "Client disconnected, stopping stream"
                                                );
                                                break;
                                            }
                                        }
                                        if chunk.is_final {
                                            break;
                                        }
                                    }
                                    Some(route_response::Response::CompleteResponse(data)) => {
                                        // Non-streaming response received during streaming mode
                                        let _ = tx.send(Ok(Bytes::from(data)));
                                        break;
                                    }
                                    Some(route_response::Response::Error(err)) => {
                                        error!(
                                            request_id = %request_id_clone,
                                            error_code = %err.code,
                                            error_message = %err.message,
                                            "Router returned error during streaming"
                                        );
                                        // Send error as SSE event
                                        let error_event = format!(
                                            "data: {{\"error\": {{\"message\": \"{}\", \"type\": \"{}\"}}}}\n\n",
                                            err.message.replace('"', "\\\""),
                                            err.code
                                        );
                                        let _ = tx.send(Ok(Bytes::from(error_event)));
                                        break;
                                    }
                                    None => {
                                        warn!(
                                            request_id = %request_id_clone,
                                            "Received empty response variant in stream"
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                error!(
                                    request_id = %request_id_clone,
                                    error = %e,
                                    "gRPC stream error"
                                );
                                // Send error as SSE event
                                let error_event = format!(
                                    "data: {{\"error\": {{\"message\": \"Stream error: {}\", \"type\": \"server_error\"}}}}\n\n",
                                    e.message().replace('"', "\\\"")
                                );
                                let _ = tx.send(Ok(Bytes::from(error_event)));
                                break;
                            }
                        }
                    }
                });

                // Build streaming response
                let stream = UnboundedReceiverStream::new(rx);
                let body = Body::from_stream(stream);
                let mut response = Response::new(body);
                response
                    .headers_mut()
                    .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                response
            } else {
                // Non-streaming: collect all chunks
                match router_stream.collect().await {
                    Ok(data) => match serde_json::from_slice::<Value>(&data) {
                        Ok(json) => Json(json).into_response(),
                        Err(_) => (StatusCode::OK, data).into_response(),
                    },
                    Err(e) => {
                        error!(
                            request_id = %request_id,
                            model_id = %model_id,
                            error = %e,
                            "Router response error"
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": {
                                    "message": format!("Router error: {}", e),
                                    "type": "server_error"
                                }
                            })),
                        )
                            .into_response()
                    }
                }
            }
        }
        Err(e) => {
            error!(
                request_id = %request_id,
                model_id = %model_id,
                error = %e,
                "Failed to forward to router"
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Router unavailable: {}", e),
                        "type": "server_error",
                        "code": "router_unavailable"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Process chat messages: apply chat template and tokenize
fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer_entry: &TokenizerEntry,
) -> Result<(String, Vec<u32>), String> {
    // Convert messages to the format expected by chat template
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|m| chat_message_to_json(m))
        .collect();

    // For now, use a simple approach: serialize messages as JSON string
    // TODO: Integrate with ChatTemplateProcessor when chat template is available
    let processed_text = serde_json::to_string(&messages)
        .map_err(|e| format!("Failed to serialize messages: {}", e))?;

    // Tokenize using the encoder
    let encoding = tokenizer_entry
        .tokenizer
        .encode(&processed_text, false) // add_special_tokens = false (chat template handles them)
        .map_err(|e| format!("Tokenization error: {}", e))?;

    // Convert Encoding to Vec<u32>
    let token_ids = encoding.token_ids().to_vec();

    Ok((processed_text, token_ids))
}

/// Convert ChatMessage to JSON value for chat template
fn chat_message_to_json(message: &ChatMessage) -> Value {
    match message {
        ChatMessage::System { content, name } => {
            let mut obj = serde_json::json!({
                "role": "system",
                "content": content_to_string(content)
            });
            if let Some(n) = name {
                obj["name"] = Value::String(n.clone());
            }
            obj
        }
        ChatMessage::User { content, name } => {
            let mut obj = serde_json::json!({
                "role": "user",
                "content": content_to_string(content)
            });
            if let Some(n) = name {
                obj["name"] = Value::String(n.clone());
            }
            obj
        }
        ChatMessage::Assistant {
            content,
            name,
            tool_calls,
            reasoning_content,
        } => {
            let mut obj = serde_json::json!({
                "role": "assistant",
            });
            if let Some(c) = content {
                obj["content"] = Value::String(content_to_string(c));
            }
            if let Some(n) = name {
                obj["name"] = Value::String(n.clone());
            }
            if let Some(tc) = tool_calls {
                match serde_json::to_value(tc) {
                    Ok(v) => obj["tool_calls"] = v,
                    Err(e) => {
                        warn!(error = %e, "Failed to serialize tool_calls, omitting from message");
                    }
                }
            }
            if let Some(rc) = reasoning_content {
                obj["reasoning_content"] = Value::String(rc.clone());
            }
            obj
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => {
            serde_json::json!({
                "role": "tool",
                "content": content_to_string(content),
                "tool_call_id": tool_call_id
            })
        }
        ChatMessage::Function { content, name } => {
            serde_json::json!({
                "role": "function",
                "content": content,
                "name": name
            })
        }
        ChatMessage::Developer {
            content,
            tools,
            name,
        } => {
            let mut obj = serde_json::json!({
                "role": "developer",
                "content": content_to_string(content)
            });
            if let Some(t) = tools {
                match serde_json::to_value(t) {
                    Ok(v) => obj["tools"] = v,
                    Err(e) => {
                        warn!(error = %e, "Failed to serialize tools, omitting from message");
                    }
                }
            }
            if let Some(n) = name {
                obj["name"] = Value::String(n.clone());
            }
            obj
        }
    }
}

/// Convert MessageContent to string
fn content_to_string(content: &MessageContent) -> String {
    content.to_simple_string()
}

/// Extract headers to forward to router
fn extract_forward_headers(headers: &HeaderMap) -> std::collections::HashMap<String, String> {
    let mut forward = std::collections::HashMap::new();

    // Forward routing-related headers
    let headers_to_forward = [
        "x-smg-routing-key",
        "x-smg-target-worker",
        "x-request-id",
        "traceparent",
        "tracestate",
    ];

    for name in headers_to_forward {
        if let Some(value) = headers.get(name) {
            if let Ok(v) = value.to_str() {
                forward.insert(name.to_string(), v.to_string());
            }
        }
    }

    forward
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_forward_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "test-key".parse().unwrap());
        headers.insert("x-request-id", "req-123".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap()); // Should not forward

        let forward = extract_forward_headers(&headers);

        assert_eq!(
            forward.get("x-smg-routing-key"),
            Some(&"test-key".to_string())
        );
        assert_eq!(forward.get("x-request-id"), Some(&"req-123".to_string()));
        assert!(!forward.contains_key("content-type"));
    }

    #[test]
    fn test_extract_forward_headers_all_routing_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "routing-key".parse().unwrap());
        headers.insert("x-smg-target-worker", "worker-1".parse().unwrap());
        headers.insert("x-request-id", "req-456".parse().unwrap());
        headers.insert("traceparent", "00-trace-span-01".parse().unwrap());
        headers.insert("tracestate", "vendor=value".parse().unwrap());

        let forward = extract_forward_headers(&headers);

        assert_eq!(forward.len(), 5);
        assert_eq!(
            forward.get("x-smg-routing-key"),
            Some(&"routing-key".to_string())
        );
        assert_eq!(
            forward.get("x-smg-target-worker"),
            Some(&"worker-1".to_string())
        );
        assert_eq!(forward.get("x-request-id"), Some(&"req-456".to_string()));
        assert_eq!(
            forward.get("traceparent"),
            Some(&"00-trace-span-01".to_string())
        );
        assert_eq!(forward.get("tracestate"), Some(&"vendor=value".to_string()));
    }

    #[test]
    fn test_extract_forward_headers_empty() {
        let headers = HeaderMap::new();
        let forward = extract_forward_headers(&headers);
        assert!(forward.is_empty());
    }

    #[test]
    fn test_chat_message_to_json_system() {
        let message = ChatMessage::System {
            content: MessageContent::Text("You are a helpful assistant.".to_string()),
            name: None,
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "You are a helpful assistant.");
        assert!(json.get("name").is_none());
    }

    #[test]
    fn test_chat_message_to_json_system_with_name() {
        let message = ChatMessage::System {
            content: MessageContent::Text("Instructions".to_string()),
            name: Some("instructor".to_string()),
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "Instructions");
        assert_eq!(json["name"], "instructor");
    }

    #[test]
    fn test_chat_message_to_json_user() {
        let message = ChatMessage::User {
            content: MessageContent::Text("Hello!".to_string()),
            name: None,
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello!");
    }

    #[test]
    fn test_chat_message_to_json_assistant() {
        let message = ChatMessage::Assistant {
            content: Some(MessageContent::Text("Hi there!".to_string())),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"], "Hi there!");
    }

    #[test]
    fn test_chat_message_to_json_assistant_with_reasoning() {
        let message = ChatMessage::Assistant {
            content: Some(MessageContent::Text("The answer is 42.".to_string())),
            name: None,
            tool_calls: None,
            reasoning_content: Some("Let me think about this...".to_string()),
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"], "The answer is 42.");
        assert_eq!(json["reasoning_content"], "Let me think about this...");
    }

    #[test]
    fn test_chat_message_to_json_tool() {
        let message = ChatMessage::Tool {
            content: MessageContent::Text("Tool result here".to_string()),
            tool_call_id: "call_123".to_string(),
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "tool");
        assert_eq!(json["content"], "Tool result here");
        assert_eq!(json["tool_call_id"], "call_123");
    }

    #[test]
    fn test_chat_message_to_json_function() {
        let message = ChatMessage::Function {
            content: "Function output".to_string(),
            name: "get_weather".to_string(),
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "function");
        assert_eq!(json["content"], "Function output");
        assert_eq!(json["name"], "get_weather");
    }

    #[test]
    fn test_chat_message_to_json_developer() {
        let message = ChatMessage::Developer {
            content: MessageContent::Text("Developer instructions".to_string()),
            tools: None,
            name: None,
        };

        let json = chat_message_to_json(&message);

        assert_eq!(json["role"], "developer");
        assert_eq!(json["content"], "Developer instructions");
    }

    #[test]
    fn test_content_to_string_text() {
        let content = MessageContent::Text("Simple text".to_string());
        assert_eq!(content_to_string(&content), "Simple text");
    }

    #[test]
    fn test_content_to_string_parts() {
        use crate::protocols::common::ContentPart;

        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Hello".to_string(),
            },
            ContentPart::Text {
                text: "World".to_string(),
            },
        ]);
        assert_eq!(content_to_string(&content), "Hello World");
    }
}

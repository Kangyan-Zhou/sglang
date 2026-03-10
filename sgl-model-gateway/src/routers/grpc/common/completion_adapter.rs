//! Adapter for routing `/v1/completions` requests through the gRPC chat pipeline.
//!
//! Converts CompletionRequest → ChatCompletionRequest, executes via the chat pipeline,
//! then converts the response back to CompletionResponse / CompletionStreamResponse format.

use axum::{
    body::{to_bytes, Body},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use tracing::{debug, error};

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
    common::StringOrArray,
    completion::{CompletionChoice, CompletionRequest, CompletionResponse},
};

/// Convert a CompletionRequest into a ChatCompletionRequest.
///
/// Maps the prompt to a single user message and copies shared sampling parameters.
#[allow(deprecated)]
pub(crate) fn completion_to_chat(req: &CompletionRequest) -> ChatCompletionRequest {
    // Convert prompt to a single user message
    let prompt_text = match &req.prompt {
        StringOrArray::String(s) => s.clone(),
        StringOrArray::Array(v) => v.join(""),
    };

    let messages = vec![ChatMessage::User {
        content: MessageContent::Text(prompt_text),
        name: None,
    }];

    ChatCompletionRequest {
        messages,
        model: req.model.clone(),
        frequency_penalty: req.frequency_penalty,
        function_call: None,
        functions: None,
        logit_bias: req.logit_bias.clone(),
        logprobs: false,
        max_tokens: req.max_tokens,
        max_completion_tokens: None,
        metadata: None,
        modalities: None,
        n: req.n,
        parallel_tool_calls: None,
        presence_penalty: req.presence_penalty,
        prompt_cache_key: None,
        reasoning_effort: None,
        response_format: None,
        safety_identifier: None,
        seed: req.seed,
        service_tier: None,
        stop: req.stop.clone(),
        stream: req.stream,
        stream_options: req.stream_options.clone(),
        temperature: req.temperature,
        tool_choice: None,
        tools: None,
        top_logprobs: None,
        top_p: req.top_p,
        verbosity: None,
        top_k: req.top_k,
        min_p: req.min_p,
        min_tokens: req.min_tokens,
        repetition_penalty: req.repetition_penalty,
        regex: req.regex.clone(),
        ebnf: req.ebnf.clone(),
        stop_token_ids: req.stop_token_ids.clone(),
        no_stop_trim: req.no_stop_trim,
        ignore_eos: req.ignore_eos,
        continue_final_message: false,
        skip_special_tokens: req.skip_special_tokens,
        lora_path: req.lora_path.clone(),
        session_params: req.session_params.clone(),
        separate_reasoning: true,
        stream_reasoning: true,
        chat_template_kwargs: None,
        return_hidden_states: req.return_hidden_states,
        sampling_seed: req.sampling_seed,
    }
}

/// Convert a ChatCompletionResponse JSON into CompletionResponse JSON.
fn chat_response_to_completion(chat_resp: &ChatCompletionResponse) -> CompletionResponse {
    let choices: Vec<CompletionChoice> = chat_resp
        .choices
        .iter()
        .map(|c| CompletionChoice {
            text: c.message.content.clone().unwrap_or_default(),
            index: c.index,
            logprobs: None,
            finish_reason: c.finish_reason.clone(),
            matched_stop: c.matched_stop.clone(),
        })
        .collect();

    CompletionResponse {
        id: chat_resp.id.clone(),
        object: "text_completion".to_string(),
        created: chat_resp.created,
        model: chat_resp.model.clone(),
        choices,
        usage: chat_resp.usage.clone(),
        system_fingerprint: chat_resp.system_fingerprint.clone(),
    }
}

/// Transform a non-streaming chat response into a completion response.
pub(crate) async fn transform_non_streaming_response(response: Response) -> Response {
    let status = response.status();
    if !status.is_success() {
        return response;
    }

    // Read the response body
    let body_bytes = match to_bytes(response.into_body(), usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("Failed to read chat response body: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to read response body",
            )
                .into_response();
        }
    };

    // Parse as ChatCompletionResponse
    let chat_resp: ChatCompletionResponse = match serde_json::from_slice(&body_bytes) {
        Ok(resp) => resp,
        Err(e) => {
            error!("Failed to parse chat response: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to parse chat response",
            )
                .into_response();
        }
    };

    // Convert to CompletionResponse
    let completion_resp = chat_response_to_completion(&chat_resp);

    axum::Json(completion_resp).into_response()
}

/// Transform a streaming chat response into a streaming completion response.
///
/// Each SSE line of the form `data: {...}` is converted from chat.completion.chunk
/// to text_completion format.
pub(crate) async fn transform_streaming_response(response: Response) -> Response {
    let (parts, body) = response.into_parts();

    // Create a stream that transforms each SSE chunk
    let stream = body.into_data_stream().map(move |result| {
        match result {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                let mut output = String::new();

                for line in text.split('\n') {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        output.push('\n');
                        continue;
                    }

                    if trimmed == "data: [DONE]" {
                        output.push_str("data: [DONE]\n");
                        continue;
                    }

                    if let Some(json_str) = trimmed.strip_prefix("data: ") {
                        match serde_json::from_str::<Value>(json_str) {
                            Ok(mut chunk) => {
                                // Convert object type
                                chunk["object"] = Value::String("text_completion".to_string());

                                // Convert choices: ChatStreamChoice → CompletionStreamChoice
                                if let Some(choices) = chunk.get_mut("choices") {
                                    if let Some(arr) = choices.as_array_mut() {
                                        for choice in arr.iter_mut() {
                                            // Extract text from delta.content → text
                                            let text = choice
                                                .get("delta")
                                                .and_then(|d| d.get("content"))
                                                .and_then(|c| c.as_str())
                                                .unwrap_or("")
                                                .to_string();
                                            choice["text"] = Value::String(text);

                                            // Remove chat-specific fields
                                            if let Some(obj) = choice.as_object_mut() {
                                                obj.remove("delta");
                                            }
                                        }
                                    }
                                }

                                output.push_str(&format!(
                                    "data: {}\n",
                                    serde_json::to_string(&chunk).unwrap_or_default()
                                ));
                            }
                            Err(e) => {
                                debug!("Failed to parse SSE chunk, passing through: {}", e);
                                output.push_str(line);
                                output.push('\n');
                            }
                        }
                    } else {
                        output.push_str(line);
                        output.push('\n');
                    }
                }

                Ok::<Bytes, std::io::Error>(Bytes::from(output))
            }
            Err(e) => {
                error!("Stream error during completion transform: {}", e);
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            }
        }
    });

    let body = Body::from_stream(stream);
    Response::from_parts(parts, body)
}

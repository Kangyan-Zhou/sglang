//! Shared response collection logic
//!
//! This module contains common logic for collecting responses from execution results.
//! Both regular and harmony processors use these functions to avoid duplication.

use axum::response::Response;

use crate::routers::{
    error,
    grpc::{context::ExecutionResult, proto_wrapper::ProtoGenerateComplete, utils},
};

/// Collect and merge responses from execution result
///
/// Handles both Single and Dual (prefill-decode) execution modes.
/// For Dual mode, merges prefill input_logprobs into decode responses if requested.
///
/// # Arguments
/// * `execution_result` - The execution result containing stream(s)
/// * `merge_logprobs` - Whether to merge prefill input_logprobs (for chat with logprobs=true)
///
/// # Returns
/// Vector of GenerateComplete responses, one per index (n parameter)
pub(crate) async fn collect_responses(
    execution_result: ExecutionResult,
    merge_logprobs: bool,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let all_responses = match execution_result {
        ExecutionResult::Single { mut stream } => {
            let responses = utils::collect_stream_responses(&mut stream, "Single").await?;
            stream.mark_completed();
            responses
        }
        ExecutionResult::Dual {
            mut prefill,
            decode,
        } => {
            // Spawn both stream collections as independent tokio tasks so the
            // tokio scheduler can multiplex them across worker threads. Without
            // spawning, each request handler task blocks on stream.next().await
            // and with N concurrent PD requests (2N streams), all worker threads
            // end up blocked in a circular deadlock (threads wait on data from
            // streams, but HTTP/2 receive buffers can't be drained because no
            // free thread is available).
            //
            // By spawning, each stream collection becomes a lightweight task
            // that yields back to the scheduler between polls, allowing the
            // runtime to interleave I/O processing across many streams with
            // only a few worker threads.
            let mut decode_stream = *decode;

            let prefill_handle = tokio::spawn(async move {
                let result = utils::collect_stream_responses(&mut prefill, "Prefill").await;
                (result, prefill)
            });

            let decode_handle = tokio::spawn(async move {
                let result =
                    utils::collect_stream_responses(&mut decode_stream, "Decode").await;
                (result, decode_stream)
            });

            // Await both spawned tasks. JoinError only occurs if the task
            // panics, which should not happen in normal operation.
            let (prefill_join, decode_join) = tokio::join!(prefill_handle, decode_handle);

            let (prefill_result, mut prefill) = prefill_join.map_err(|e| {
                error::internal_error("prefill_task_failed", format!("Prefill task panicked: {}", e))
            })?;
            let (decode_result, mut decode_stream) = decode_join.map_err(|e| {
                error::internal_error("decode_task_failed", format!("Decode task panicked: {}", e))
            })?;

            let prefill_responses = prefill_result?;
            let mut decode_responses = decode_result?;

            // Mark both streams as completed now that both succeeded
            prefill.mark_completed();
            decode_stream.mark_completed();

            // Merge prefill input_logprobs if requested
            if merge_logprobs {
                merge_prefill_logprobs(&prefill_responses, &mut decode_responses);
            }

            decode_responses
        }
        ExecutionResult::Embedding { .. } => {
            // Embeddings do not support this path (no generate complete response)
            return Err(error::internal_error(
                "invalid_execution_mode",
                "Embedding result encountered in response collection",
            ));
        }
    };

    if all_responses.is_empty() {
        return Err(error::internal_error(
            "no_responses_from_server",
            "No responses from server",
        ));
    }

    Ok(all_responses)
}

/// Merge prefill input_logprobs into decode responses
///
/// Takes input_logprobs from the first prefill response and copies them
/// into all decode responses. This is used in PD mode when logprobs are requested.
/// Only works with SGLang (vLLM doesn't support PD mode).
fn merge_prefill_logprobs(
    prefill_responses: &[ProtoGenerateComplete],
    decode_responses: &mut [ProtoGenerateComplete],
) {
    // Only SGLang supports PD mode and has input_logprobs
    if let Some(ProtoGenerateComplete::Sglang(prefill_first)) = prefill_responses.first() {
        // Use ref to borrow input_logprobs instead of cloning upfront
        // This avoids one allocation when the Option is Some
        if let Some(ref prefill_input_logprobs) = prefill_first.input_logprobs {
            for response in decode_responses.iter_mut() {
                if let ProtoGenerateComplete::Sglang(decode_resp) = response {
                    decode_resp.input_logprobs = Some(prefill_input_logprobs.clone());
                }
            }
        }
    }
}

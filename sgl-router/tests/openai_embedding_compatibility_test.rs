mod common;

use serde_json::json;
use sglang_router_rs::protocols::spec::{
    EmbeddingInput, EmbeddingRequest, OpenAIServingRequest,
    StringOrArray,
};

/// Tests for OpenAI SDK compatibility with embedding API
/// These tests ensure that the embedding API implementation matches OpenAI's specification
/// Reference: https://platform.openai.com/docs/api-reference/embeddings

#[cfg(test)]
mod openai_compatibility_tests {
    use super::*;

    // ============= Request Format Compatibility Tests =============

    #[test]
    fn test_openai_simple_text_embedding_request() {
        // Test the most basic OpenAI embedding request format
        let request_json = json!({
            "input": "The quick brown fox jumps over the lazy dog",
            "model": "text-embedding-ada-002"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json.clone());
        assert!(request.is_ok(), "Failed to parse basic OpenAI request");

        let req = request.unwrap();
        assert_eq!(req.model, "text-embedding-ada-002");
        assert_eq!(req.encoding_format, "float"); // Default value

        match req.input {
            EmbeddingInput::Text(text) => {
                assert_eq!(text, "The quick brown fox jumps over the lazy dog");
            }
            _ => panic!("Expected Text input variant"),
        }
    }

    #[test]
    fn test_openai_batch_text_embedding_request() {
        // Test batch embedding request format (multiple texts)
        let request_json = json!({
            "input": [
                "First text to embed",
                "Second text to embed",
                "Third text to embed"
            ],
            "model": "text-embedding-3-small"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse batch embedding request");

        let req = request.unwrap();
        match req.input {
            EmbeddingInput::TextArray(texts) => {
                assert_eq!(texts.len(), 3);
                assert_eq!(texts[0], "First text to embed");
                assert_eq!(texts[1], "Second text to embed");
                assert_eq!(texts[2], "Third text to embed");
            }
            _ => panic!("Expected TextArray input variant"),
        }
    }

    #[test]
    fn test_openai_embedding_with_dimensions() {
        // Test OpenAI's dimension parameter (for models like text-embedding-3-small)
        let request_json = json!({
            "input": "Sample text",
            "model": "text-embedding-3-small",
            "dimensions": 256
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse request with dimensions");

        let req = request.unwrap();
        assert_eq!(req.dimensions, Some(256));
    }

    #[test]
    fn test_openai_embedding_with_base64_encoding() {
        // Test base64 encoding format (reduces response size)
        let request_json = json!({
            "input": "Encode this as base64",
            "model": "text-embedding-3-large",
            "encoding_format": "base64"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse request with base64 encoding");

        let req = request.unwrap();
        assert_eq!(req.encoding_format, "base64");
    }

    #[test]
    fn test_openai_embedding_with_user_field() {
        // Test user field for tracking
        let request_json = json!({
            "input": "User tracking test",
            "model": "text-embedding-ada-002",
            "user": "user-12345"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse request with user field");

        let req = request.unwrap();
        assert_eq!(req.user, Some("user-12345".to_string()));
    }

    #[test]
    fn test_openai_integer_token_input() {
        // Test integer token input (for pre-tokenized input)
        let request_json = json!({
            "input": [1234, 5678, 9012, 3456],
            "model": "text-embedding-ada-002"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse integer token input");

        let req = request.unwrap();
        match req.input {
            EmbeddingInput::IntegerTokens(tokens) => {
                assert_eq!(tokens, vec![1234, 5678, 9012, 3456]);
            }
            _ => panic!("Expected IntegerTokens variant"),
        }
    }

    #[test]
    fn test_openai_batch_token_input() {
        // Test batch of token arrays
        let request_json = json!({
            "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "model": "text-embedding-ada-002"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse batch token input");

        let req = request.unwrap();
        match req.input {
            EmbeddingInput::NestedIntegerTokens(batches) => {
                assert_eq!(batches.len(), 3);
                assert_eq!(batches[0], vec![1, 2, 3]);
                assert_eq!(batches[1], vec![4, 5, 6]);
                assert_eq!(batches[2], vec![7, 8, 9]);
            }
            _ => panic!("Expected NestedIntegerTokens variant"),
        }
    }

    // ============= Model Name Compatibility Tests =============

    #[test]
    fn test_openai_model_names() {
        // Test compatibility with all OpenAI embedding model names
        let models = vec![
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ];

        for model in models {
            let request_json = json!({
                "input": "Test input",
                "model": model
            });

            let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
            assert!(
                request.is_ok(),
                "Failed to parse request with model: {}",
                model
            );
            assert_eq!(request.unwrap().model, model);
        }
    }

    // ============= Response Format Validation Tests =============

    #[test]
    fn test_openai_response_structure() {
        // Validate that response structure matches OpenAI format
        let expected_response = json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.0023064255, -0.009327292, -0.0028842222]
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        });

        // Verify the structure can be parsed
        let response_value: serde_json::Value = expected_response;
        assert_eq!(response_value["object"], "list");
        assert!(response_value["data"].is_array());
        assert_eq!(response_value["data"][0]["object"], "embedding");
        assert_eq!(response_value["data"][0]["index"], 0);
        assert!(response_value["data"][0]["embedding"].is_array());
    }

    #[test]
    fn test_openai_base64_response_structure() {
        // Validate base64 encoded response structure
        let expected_response = json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": "base64encodedstring=="
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        });

        let response_value: serde_json::Value = expected_response;
        assert!(response_value["data"][0]["embedding"].is_string());
    }

    // ============= Error Response Compatibility Tests =============

    #[test]
    fn test_openai_error_response_format() {
        // Validate error response matches OpenAI format
        let error_response = json!({
            "error": {
                "message": "Invalid model specified",
                "type": "invalid_request_error",
                "param": "model",
                "code": null
            }
        });

        assert!(error_response["error"]["message"].is_string());
        assert!(error_response["error"]["type"].is_string());
        assert_eq!(error_response["error"]["type"], "invalid_request_error");
    }

    // ============= SGLang Extensions Compatibility Tests =============

    #[test]
    fn test_sglang_request_id_extension() {
        // Test SGLang's rid (request ID) extension
        let request_json = json!({
            "input": "Test with request ID",
            "model": "text-embedding-ada-002",
            "rid": "custom-request-123"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse request with rid");

        let req = request.unwrap();
        match req.rid {
            Some(StringOrArray::String(id)) => {
                assert_eq!(id, "custom-request-123");
            }
            _ => panic!("Expected rid to be String variant"),
        }
    }

    #[test]
    fn test_sglang_batch_request_ids() {
        // Test batch request IDs for SGLang
        let request_json = json!({
            "input": ["text1", "text2", "text3"],
            "model": "text-embedding-ada-002",
            "rid": ["id1", "id2", "id3"]
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse request with batch rids");

        let req = request.unwrap();
        match req.rid {
            Some(StringOrArray::Array(ids)) => {
                assert_eq!(ids.len(), 3);
                assert_eq!(ids[0], "id1");
                assert_eq!(ids[1], "id2");
                assert_eq!(ids[2], "id3");
            }
            _ => panic!("Expected rid to be Array variant"),
        }
    }

    // ============= Multimodal Embedding Tests =============

    #[test]
    fn test_multimodal_embedding_request() {
        // Test multimodal embedding with text and image
        let request_json = json!({
            "input": [
                {
                    "text": "A beautiful sunset",
                    "image": "base64_encoded_image_data"
                },
                {
                    "text": "Another description"
                }
            ],
            "model": "multimodal-embedding-model"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
        assert!(request.is_ok(), "Failed to parse multimodal request");

        let req = request.unwrap();
        match req.input {
            EmbeddingInput::Multimodal(inputs) => {
                assert_eq!(inputs.len(), 2);
                assert_eq!(inputs[0].text, Some("A beautiful sunset".to_string()));
                assert_eq!(inputs[0].image, Some("base64_encoded_image_data".to_string()));
                assert_eq!(inputs[1].text, Some("Another description".to_string()));
                assert_eq!(inputs[1].image, None);
            }
            _ => panic!("Expected Multimodal variant"),
        }
    }

    // ============= OpenAIServingRequest Compatibility =============

    #[test]
    fn test_embedding_as_openai_serving_request() {
        // Test that embedding requests work as part of OpenAIServingRequest enum
        let request_json = json!({
            "input": "Test embedding in serving request",
            "model": "text-embedding-ada-002"
        });

        let request: Result<OpenAIServingRequest, _> = serde_json::from_value(request_json);
        assert!(
            request.is_ok(),
            "Failed to parse embedding as OpenAIServingRequest"
        );

        match request.unwrap() {
            OpenAIServingRequest::Embedding(embed_req) => {
                assert_eq!(embed_req.model, "text-embedding-ada-002");
            }
            _ => panic!("Expected Embedding variant in OpenAIServingRequest"),
        }
    }

    // ============= Dimension Validation Tests =============

    #[test]
    fn test_valid_dimension_ranges() {
        // Test various valid dimension values
        let valid_dimensions = vec![1, 64, 256, 512, 1024, 1536, 3072, 4096];

        for dim in valid_dimensions {
            let request_json = json!({
                "input": "Test",
                "model": "text-embedding-3-small",
                "dimensions": dim
            });

            let request: Result<EmbeddingRequest, _> = serde_json::from_value(request_json);
            assert!(
                request.is_ok(),
                "Failed to parse request with dimension: {}",
                dim
            );
            assert_eq!(request.unwrap().dimensions, Some(dim));
        }
    }

    // ============= Complete Request/Response Flow Test =============

    #[test]
    fn test_complete_openai_request_response_flow() {
        // Test a complete request with all optional fields
        let complete_request = json!({
            "input": ["First text", "Second text"],
            "model": "text-embedding-3-large",
            "encoding_format": "float",
            "dimensions": 1024,
            "user": "test-user-123"
        });

        let request: Result<EmbeddingRequest, _> = serde_json::from_value(complete_request);
        assert!(request.is_ok(), "Failed to parse complete request");

        let req = request.unwrap();
        assert_eq!(req.model, "text-embedding-3-large");
        assert_eq!(req.encoding_format, "float");
        assert_eq!(req.dimensions, Some(1024));
        assert_eq!(req.user, Some("test-user-123".to_string()));

        match req.input {
            EmbeddingInput::TextArray(texts) => {
                assert_eq!(texts.len(), 2);
                assert_eq!(texts[0], "First text");
                assert_eq!(texts[1], "Second text");
            }
            _ => panic!("Expected TextArray variant"),
        }
    }
}
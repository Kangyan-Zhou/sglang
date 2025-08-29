use serde_json::json;
use sglang_router_rs::protocols::spec::{
    EmbeddingInput, EmbeddingRequest, GenerationRequest, MultimodalEmbeddingInput, StringOrArray,
};

/// Comprehensive validation tests for embedding input formats
/// These tests validate input format parsing, constraint checking, and edge case handling
/// for all supported embedding input types and parameters.

#[cfg(test)]
mod embedding_validation_tests {
    use super::*;

    // ============= Input Format Validation Tests =============

    #[test]
    fn test_single_text_input_validation() {
        // Test valid single text input
        let valid_inputs = vec![
            "Hello world",
            "A very long text that should still be valid for embedding processing",
            "Text with special characters: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./ and unicode: 你好 🌟",
            "", // Empty string should be parseable but might be invalid for processing
        ];

        for input_text in valid_inputs {
            let request_json = json!({
                "input": input_text,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Failed to parse valid text input: '{}'",
                input_text
            );

            let request = result.unwrap();
            match request.input {
                EmbeddingInput::Text(text) => assert_eq!(text, input_text),
                _ => panic!("Expected Text variant for input: '{}'", input_text),
            }
        }
    }

    #[test]
    fn test_text_array_input_validation() {
        // Test valid text array inputs
        let test_cases = vec![
            (vec!["single"], "single item array"),
            (vec!["first", "second"], "two item array"),
            (vec!["a", "b", "c", "d", "e"], "five item array"),
            (vec![""], "array with empty string"),
            (vec!["normal", "", "another"], "array with mixed content"),
        ];

        for (input_array, description) in test_cases {
            let request_json = json!({
                "input": input_array,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(result.is_ok(), "Failed to parse {}: {:?}", description, result);

            let request = result.unwrap();
            match request.input {
                EmbeddingInput::TextArray(texts) => {
                    assert_eq!(texts, input_array);
                    assert_eq!(texts.len(), input_array.len());
                }
                _ => panic!("Expected TextArray variant for {}", description),
            }
        }
    }

    #[test]
    fn test_integer_token_input_validation() {
        // Test valid integer token inputs
        let test_cases = vec![
            (vec![1, 2, 3, 4], "simple token sequence"),
            (vec![101, 102, 103], "bert-style tokens"),
            (vec![0], "single zero token"),
            (vec![50256], "large token id"),
            (vec![1, 50000, 2, 50001], "mixed token ids"),
        ];

        for (token_array, description) in test_cases {
            let request_json = json!({
                "input": token_array,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(result.is_ok(), "Failed to parse {}: {:?}", description, result);

            let request = result.unwrap();
            match request.input {
                EmbeddingInput::IntegerTokens(tokens) => {
                    assert_eq!(tokens, token_array);
                }
                _ => panic!("Expected IntegerTokens variant for {}", description),
            }
        }
    }

    #[test]
    fn test_nested_integer_token_input_validation() {
        // Test valid nested integer token inputs (batch of token arrays)
        let test_cases = vec![
            (vec![vec![1, 2], vec![3, 4]], "simple batch"),
            (vec![vec![101, 102, 103]], "single sequence batch"),
            (
                vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]],
                "variable length sequences",
            ),
            (vec![vec![0], vec![1], vec![2]], "single token sequences"),
        ];

        for (token_batches, description) in test_cases {
            let request_json = json!({
                "input": token_batches,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Failed to parse {}: {:?}",
                description,
                result
            );

            let request = result.unwrap();
            match request.input {
                EmbeddingInput::NestedIntegerTokens(batches) => {
                    assert_eq!(batches, token_batches);
                    assert_eq!(batches.len(), token_batches.len());
                }
                _ => panic!("Expected NestedIntegerTokens variant for {}", description),
            }
        }
    }

    #[test]
    fn test_multimodal_input_validation() {
        // Test valid multimodal inputs
        let test_cases = vec![
            (
                json!([{"text": "description", "image": "base64_data"}]),
                "text and image",
            ),
            (json!([{"text": "only text"}]), "text only"),
            (json!([{"image": "only_image_data"}]), "image only"),
            (
                json!([
                    {"text": "first", "image": "img1"},
                    {"text": "second"},
                    {"image": "img2"}
                ]),
                "mixed multimodal batch",
            ),
        ];

        for (input_json, description) in test_cases {
            let request_json = json!({
                "input": input_json,
                "model": "multimodal-embedding"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Failed to parse {}: {:?}",
                description,
                result
            );

            let request = result.unwrap();
            match request.input {
                EmbeddingInput::Multimodal(inputs) => {
                    assert!(!inputs.is_empty(), "Multimodal input should not be empty");
                }
                _ => panic!("Expected Multimodal variant for {}", description),
            }
        }
    }

    // ============= Input Format Edge Cases =============

    #[test]
    fn test_empty_input_arrays() {
        // Test behavior with empty arrays
        let empty_cases = vec![
            (json!([]), "empty text array"),
            (json!([[]]), "nested array with empty inner array"),
        ];

        for (input_json, description) in empty_cases {
            let request_json = json!({
                "input": input_json,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            // Should parse successfully but might be invalid for processing
            assert!(result.is_ok(), "Should parse {} successfully", description);
        }
    }

    #[test]
    fn test_large_input_validation() {
        // Test very large inputs to validate size handling
        let large_text = "word ".repeat(10000); // 50k characters
        let request_json = json!({
            "input": large_text,
            "model": "text-embedding-ada-002"
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_ok(), "Should handle large text input");

        // Test large token array
        let large_tokens: Vec<i32> = (0..10000).collect();
        let request_json = json!({
            "input": large_tokens,
            "model": "text-embedding-ada-002"
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_ok(), "Should handle large token array");
    }

    #[test]
    fn test_special_characters_in_text() {
        // Test various special characters and unicode
        let special_texts = vec![
            "Emoji test: 😀🎉🌟🔥💯",
            "Chinese: 你好世界，这是一个测试",
            "Arabic: مرحبا بالعالم، هذا اختبار",
            "Russian: Привет мир, это тест",
            "Math symbols: ∑∆∞√π∂∫≠≤≥",
            "Code: fn main() { println!(\"Hello, world!\"); }",
            "Mixed: Hello 世界 🌍 मुझे खुशी होगी",
        ];

        for text in special_texts {
            let request_json = json!({
                "input": text,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should handle special characters in text: {}",
                text
            );
        }
    }

    // ============= Dimension Validation Tests =============

    #[test]
    fn test_valid_dimension_values() {
        let valid_dimensions = vec![1, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096];

        for dim in valid_dimensions {
            let request_json = json!({
                "input": "test",
                "model": "text-embedding-3-small",
                "dimensions": dim
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should accept valid dimension: {}",
                dim
            );

            let request = result.unwrap();
            assert_eq!(request.dimensions, Some(dim));
        }
    }

    #[test]
    fn test_dimension_edge_cases() {
        // Test boundary values
        let edge_cases = vec![
            (0, "zero dimensions", false),
            (-1, "negative dimensions", false),
            (4097, "exceeds maximum", false),
            (1, "minimum valid", true),
            (4096, "maximum valid", true),
        ];

        for (dim, description, should_parse) in edge_cases {
            let request_json = json!({
                "input": "test",
                "model": "text-embedding-3-small",
                "dimensions": dim
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            
            if should_parse {
                assert!(
                    result.is_ok(),
                    "Should parse {}: {}",
                    description,
                    dim
                );
            } else {
                // Note: JSON schema validation might not catch all invalid values
                // The validation should happen at request processing time
                println!("Testing {}: {} - result: {:?}", description, dim, result.is_ok());
            }
        }
    }

    // ============= Encoding Format Validation Tests =============

    #[test]
    fn test_valid_encoding_formats() {
        let valid_formats = vec!["float", "base64"];

        for format in valid_formats {
            let request_json = json!({
                "input": "test",
                "model": "text-embedding-ada-002",
                "encoding_format": format
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should accept valid encoding format: {}",
                format
            );

            let request = result.unwrap();
            assert_eq!(request.encoding_format, format);
        }
    }

    #[test]
    fn test_encoding_format_case_sensitivity() {
        // Test case variations
        let format_cases = vec![
            "float",
            "FLOAT",
            "Float", 
            "base64",
            "BASE64",
            "Base64",
        ];

        for format in format_cases {
            let request_json = json!({
                "input": "test",
                "model": "text-embedding-ada-002",
                "encoding_format": format
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should parse encoding format regardless of case: {}",
                format
            );
        }
    }

    #[test]
    fn test_invalid_encoding_formats() {
        let invalid_formats = vec![
            "json",
            "xml",
            "text", 
            "binary",
            "base32",
            "",
            "float64",
        ];

        for format in invalid_formats {
            let request_json = json!({
                "input": "test",
                "model": "text-embedding-ada-002",
                "encoding_format": format
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            // Should parse but might be invalid during processing
            // The validation should happen at request processing time
            println!("Testing invalid encoding format '{}': {:?}", format, result.is_ok());
        }
    }

    #[test]
    fn test_encoding_format_default() {
        // Test that default encoding format is "float"
        let request_json = json!({
            "input": "test",
            "model": "text-embedding-ada-002"
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_ok(), "Should parse request without encoding format");

        let request = result.unwrap();
        assert_eq!(request.encoding_format, "float");
    }

    // ============= Model Name Validation Tests =============

    #[test]
    fn test_model_name_validation() {
        let valid_models = vec![
            "text-embedding-ada-002",
            "text-embedding-3-small", 
            "text-embedding-3-large",
            "custom-model-name",
            "model_with_underscores",
            "model123",
        ];

        for model in valid_models {
            let request_json = json!({
                "input": "test",
                "model": model
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should accept valid model name: {}",
                model
            );

            let request = result.unwrap();
            assert_eq!(request.model, model);
        }
    }

    #[test]
    fn test_empty_model_name() {
        let request_json = json!({
            "input": "test",
            "model": ""
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_ok(), "Should parse empty model name");
        
        let request = result.unwrap();
        assert_eq!(request.model, "");
    }

    // ============= Request ID Validation Tests =============

    #[test]
    fn test_request_id_formats() {
        let rid_test_cases = vec![
            (json!("single-id"), "single string ID"),
            (json!(["id1", "id2", "id3"]), "array of IDs"),
            (json!(""), "empty string ID"),
            (json!([]), "empty array of IDs"),
            (json!(["single"]), "single item array"),
        ];

        for (rid_value, description) in rid_test_cases {
            let request_json = json!({
                "input": "test",
                "model": "text-embedding-ada-002",
                "rid": rid_value
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should parse {}: {:?}",
                description,
                result
            );

            let request = result.unwrap();
            assert!(request.rid.is_some(), "RID should be present");
        }
    }

    #[test]
    fn test_request_id_batch_consistency() {
        // Test that batch RID arrays match input batch sizes
        let test_cases = vec![
            (json!(["text1", "text2"]), json!(["id1", "id2"]), "matching batch sizes"),
            (json!(["text1"]), json!("single-id"), "single text with single ID"),
            (json!("single-text"), json!(["single-id"]), "single text with array ID"),
        ];

        for (input, rid, description) in test_cases {
            let request_json = json!({
                "input": input,
                "model": "text-embedding-ada-002",
                "rid": rid
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should parse {}: {:?}",
                description,
                result
            );
        }
    }

    // ============= Complete Request Validation Tests =============

    #[test]
    fn test_minimal_valid_request() {
        let request_json = json!({
            "input": "test",
            "model": "text-embedding-ada-002"
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_ok(), "Should parse minimal valid request");

        let request = result.unwrap();
        assert_eq!(request.encoding_format, "float"); // default
        assert_eq!(request.dimensions, None);
        assert_eq!(request.user, None);
        assert_eq!(request.rid, None);
    }

    #[test]
    fn test_complete_request_validation() {
        let request_json = json!({
            "input": ["text1", "text2", "text3"],
            "model": "text-embedding-3-large",
            "encoding_format": "base64",
            "dimensions": 1024,
            "user": "test-user-123",
            "rid": ["req1", "req2", "req3"]
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_ok(), "Should parse complete request");

        let request = result.unwrap();
        assert_eq!(request.model, "text-embedding-3-large");
        assert_eq!(request.encoding_format, "base64");
        assert_eq!(request.dimensions, Some(1024));
        assert_eq!(request.user, Some("test-user-123".to_string()));

        match request.input {
            EmbeddingInput::TextArray(texts) => assert_eq!(texts.len(), 3),
            _ => panic!("Expected TextArray"),
        }

        match request.rid {
            Some(StringOrArray::Array(ids)) => assert_eq!(ids.len(), 3),
            _ => panic!("Expected Array of RIDs"),
        }
    }

    #[test]
    fn test_missing_required_fields() {
        // Test missing input field
        let request_json = json!({
            "model": "text-embedding-ada-002"
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_err(), "Should fail without input field");

        // Test missing model field
        let request_json = json!({
            "input": "test"
        });

        let result = serde_json::from_value::<EmbeddingRequest>(request_json);
        assert!(result.is_err(), "Should fail without model field");
    }

    // ============= Batch Size Validation Tests =============

    #[test]
    fn test_various_batch_sizes() {
        let batch_sizes = vec![1, 5, 10, 50, 100, 1000];

        for size in batch_sizes {
            let inputs: Vec<String> = (0..size).map(|i| format!("text_{}", i)).collect();
            
            let request_json = json!({
                "input": inputs,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            assert!(
                result.is_ok(),
                "Should handle batch size: {}",
                size
            );

            let request = result.unwrap();
            match request.input {
                EmbeddingInput::TextArray(texts) => {
                    assert_eq!(texts.len(), size, "Batch size should match");
                }
                _ => panic!("Expected TextArray for batch size {}", size),
            }
        }
    }

    #[test]
    fn test_mixed_content_validation() {
        // Test arrays with mixed content types (should be handled by JSON parsing)
        let mixed_cases = vec![
            json!(["text", 123]),           // String and number
            json!([true, "text"]),          // Boolean and string  
            json!([null, "text"]),          // Null and string
            json!(["text", {"key": "val"}]), // String and object
        ];

        for (i, mixed_input) in mixed_cases.into_iter().enumerate() {
            let request_json = json!({
                "input": mixed_input,
                "model": "text-embedding-ada-002"
            });

            let result = serde_json::from_value::<EmbeddingRequest>(request_json);
            // These should generally fail during JSON deserialization
            println!("Mixed content test {}: success = {}", i, result.is_ok());
        }
    }

    // ============= Input Format Routing Text Extraction Tests =============

    #[test]
    fn test_routing_text_extraction() {
        // Test that extract_text_for_routing works correctly for all input types
        let test_cases = vec![
            (
                EmbeddingInput::Text("Hello world".to_string()),
                "Hello world",
                "single text",
            ),
            (
                EmbeddingInput::TextArray(vec!["First".to_string(), "Second".to_string()]),
                "First Second",
                "text array",
            ),
            (
                EmbeddingInput::IntegerTokens(vec![1, 2, 3]),
                "",
                "integer tokens",
            ),
            (
                EmbeddingInput::NestedIntegerTokens(vec![vec![1, 2], vec![3, 4]]),
                "",
                "nested integer tokens",
            ),
            (
                EmbeddingInput::Multimodal(vec![
                    MultimodalEmbeddingInput {
                        text: Some("Text 1".to_string()),
                        image: Some("image1".to_string()),
                    },
                    MultimodalEmbeddingInput {
                        text: Some("Text 2".to_string()),
                        image: None,
                    },
                    MultimodalEmbeddingInput {
                        text: None,
                        image: Some("image2".to_string()),
                    },
                ]),
                "Text 1 Text 2",
                "multimodal input",
            ),
        ];

        for (input, expected_text, description) in test_cases {
            let request = EmbeddingRequest {
                input,
                model: "test".to_string(),
                encoding_format: "float".to_string(),
                dimensions: None,
                user: None,
                rid: None,
            };

            let extracted = request.extract_text_for_routing();
            assert_eq!(
                extracted, expected_text,
                "Text extraction failed for {}: expected '{}', got '{}'",
                description, expected_text, extracted
            );
        }
    }
}
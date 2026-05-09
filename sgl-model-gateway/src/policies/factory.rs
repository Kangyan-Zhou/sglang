//! Factory for creating load balancing policies

use std::sync::Arc;

use tracing::warn;

use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, CacheAwareZmqPolicy,
    ConsistentHashingPolicy, LoadBalancingPolicy, ManualConfig, ManualPolicy, PowerOfTwoPolicy,
    PrefixHashConfig, PrefixHashPolicy, RandomPolicy, RoundRobinPolicy,
};
use crate::config::types::CacheAwareSyncMode;
use crate::config::PolicyConfig;
use crate::tokenizer::TokenizerRegistry;

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration.
    ///
    /// `tokenizer_registry` is `None` from legacy call sites that have not
    /// been threaded through yet (e.g., older test fixtures). When the
    /// configured `sync_mode` is [`CacheAwareSyncMode::Zmq`] but the
    /// tokenizer registry is missing, the factory logs and silently falls
    /// back to the legacy mesh-style [`CacheAwarePolicy`]. This keeps
    /// existing tests green while the rest of the wiring is migrated.
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        Self::create_from_config_with_registry(config, None)
    }

    /// Create a policy from configuration, optionally given access to the
    /// shared `TokenizerRegistry`. Required for the
    /// [`CacheAwareSyncMode::Zmq`] branch — without it the factory falls
    /// back to mesh.
    pub fn create_from_config_with_registry(
        config: &PolicyConfig,
        tokenizer_registry: Option<Arc<TokenizerRegistry>>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                sync_mode,
                block_size,
                event_port,
                event_channel_capacity,
            } => {
                let cache_config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                    block_size: *block_size,
                    event_port: *event_port,
                    event_channel_capacity: *event_channel_capacity,
                };
                match (*sync_mode, tokenizer_registry) {
                    (CacheAwareSyncMode::Zmq, Some(registry)) => {
                        Arc::new(CacheAwareZmqPolicy::new(cache_config, registry))
                    }
                    (CacheAwareSyncMode::Zmq, None) => {
                        warn!(
                            "cache_aware sync_mode=zmq requested but no TokenizerRegistry was \
                             provided to PolicyFactory; falling back to mesh policy. \
                             Construct via PolicyFactory::create_from_config_with_registry to \
                             enable the ZMQ KV-event indexer."
                        );
                        Arc::new(CacheAwarePolicy::with_config(cache_config))
                    }
                    (CacheAwareSyncMode::Mesh, _) => {
                        Arc::new(CacheAwarePolicy::with_config(cache_config))
                    }
                }
            }
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                let config = BucketConfig {
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    bucket_adjust_interval_secs: *bucket_adjust_interval_secs,
                };
                Arc::new(BucketPolicy::with_config(config))
            }
            PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode,
            } => {
                let config = ManualConfig {
                    eviction_interval_secs: *eviction_interval_secs,
                    max_idle_secs: *max_idle_secs,
                    assignment_mode: *assignment_mode,
                };
                Arc::new(ManualPolicy::with_config(config))
            }
            PolicyConfig::ConsistentHashing => Arc::new(ConsistentHashingPolicy::new()),
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                let config = PrefixHashConfig {
                    prefix_token_count: *prefix_token_count,
                    load_factor: *load_factor,
                };
                Arc::new(PrefixHashPolicy::new(config))
            }
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "bucket" => Some(Arc::new(BucketPolicy::new())),
            "manual" => Some(Arc::new(ManualPolicy::new())),
            "consistent_hashing" | "consistenthashing" => {
                Some(Arc::new(ConsistentHashingPolicy::new()))
            }
            "prefix_hash" | "prefixhash" => Some(Arc::new(PrefixHashPolicy::with_defaults())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random);
        assert_eq!(policy.name(), "random");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        assert_eq!(policy.name(), "round_robin");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        });
        assert_eq!(policy.name(), "power_of_two");

        // Mesh sync_mode keeps the existing CacheAwarePolicy semantics for
        // this test; the Zmq path is exercised by the e2e test in
        // `tests/routing/cache_aware_zmq_test.rs`.
        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
            sync_mode: CacheAwareSyncMode::Mesh,
            block_size: 64,
            event_port: 5557,
            event_channel_capacity: 1024,
        });
        assert_eq!(policy.name(), "cache_aware");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Bucket {
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        assert_eq!(policy.name(), "bucket");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Default::default(),
        });
        assert_eq!(policy.name(), "manual");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing);
        assert_eq!(policy.name(), "consistent_hashing");
    }

    #[tokio::test]
    async fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("bucket").is_some());
        assert!(PolicyFactory::create_by_name("Bucket").is_some());
        assert!(PolicyFactory::create_by_name("manual").is_some());
        assert!(PolicyFactory::create_by_name("Manual").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hashing").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHashing").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }
}

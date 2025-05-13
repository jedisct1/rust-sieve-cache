[![dependency status](https://deps.rs/repo/github/jedisct1/rust-sieve-cache/status.svg)](https://deps.rs/repo/github/jedisct1/rust-sieve-cache)

# SIEVE cache

A high-performance implementation of the [SIEVE](http://sievecache.com) cache replacement algorithm for Rust, with both single-threaded and thread-safe variants.

- [API documentation](https://docs.rs/sieve-cache)
- [`crates.io` page](https://crates.io/crates/sieve-cache)

## Overview

SIEVE is an eviction algorithm that is simpler than LRU but achieves state-of-the-art efficiency on skewed workloads. It works by maintaining a single bit per entry that tracks whether an item has been "visited" since it was last considered for eviction.

### Key Features

- **Simple and efficient**: SIEVE requires less state than LRU or LFU algorithms
- **Good performance on skewed workloads**: Particularly effective for real-world access patterns
- **Multiple implementations**:
  - `SieveCache`: Core single-threaded implementation
  - `SyncSieveCache`: Thread-safe wrapper using a single lock
  - `ShardedSieveCache`: High-concurrency implementation using multiple locks

This implementation exposes the same API as the `clock-pro` and `arc-cache` crates, so it can be used as a drop-in replacement for them in existing applications.

## Basic Usage

The library provides three cache implementations with different threading characteristics.

### `SieveCache` - Single-Threaded Implementation

The core `SieveCache` implementation is designed for single-threaded use. It's the most efficient option when thread safety isn't required.

```rust
use sieve_cache::SieveCache;

// Create a new cache with a specific capacity
let mut cache: SieveCache<String, String> = SieveCache::new(100000).unwrap();

// Insert key/value pairs into the cache
cache.insert("foo".to_string(), "foocontent".to_string());
cache.insert("bar".to_string(), "barcontent".to_string());

// Retrieve a value from the cache (returns a reference to the value)
assert_eq!(cache.get("foo"), Some(&"foocontent".to_string()));

// Check if a key exists in the cache
assert!(cache.contains_key("foo"));
assert!(!cache.contains_key("missing_key"));

// Get a mutable reference to a value
if let Some(value) = cache.get_mut("foo") {
   *value = "updated_content".to_string();
}

// Remove an entry from the cache
let removed_value = cache.remove("bar");
assert_eq!(removed_value, Some("barcontent".to_string()));

// Query cache information
assert_eq!(cache.len(), 1);       // Number of entries
assert_eq!(cache.capacity(), 100000);  // Maximum capacity
assert!(!cache.is_empty());       // Check if empty

// Manual eviction (normally handled automatically when capacity is reached)
let evicted = cache.evict();  // Returns and removes a value that wasn't recently accessed
```

## Thread-Safe Implementations

These implementations are available when using the appropriate feature flags:
- `SyncSieveCache` is available with the `sync` feature (enabled by default)
- `ShardedSieveCache` is available with the `sharded` feature (enabled by default)

### `SyncSieveCache` - Basic Thread-Safe Cache

For concurrent access from multiple threads, you can use the `SyncSieveCache` wrapper, which provides thread safety with a single global lock:

```rust
use sieve_cache::SyncSieveCache;
use std::thread;

// Create a thread-safe cache
let cache = SyncSieveCache::new(100000).unwrap();

// The cache can be safely cloned and shared between threads
let cache_clone = cache.clone();

// Insert from the main thread
cache.insert("foo".to_string(), "foocontent".to_string());

// Access from another thread
let handle = thread::spawn(move || {
    // Insert a new key
    cache_clone.insert("bar".to_string(), "barcontent".to_string());

    // Get returns a clone of the value, not a reference (unlike non-thread-safe version)
    assert_eq!(cache_clone.get(&"foo".to_string()), Some("foocontent".to_string()));
});

// Wait for the thread to complete
handle.join().unwrap();

// Check if keys exist
assert!(cache.contains_key(&"foo".to_string()));
assert!(cache.contains_key(&"bar".to_string()));

// Remove an entry
let removed = cache.remove(&"bar".to_string());
assert_eq!(removed, Some("barcontent".to_string()));

// Perform multiple operations atomically with exclusive access
cache.with_lock(|inner_cache| {
    // Operations inside this closure have exclusive access to the cache
    inner_cache.insert("atomic1".to_string(), "value1".to_string());
    inner_cache.insert("atomic2".to_string(), "value2".to_string());

    // We can check internal state as part of the transaction
    assert_eq!(inner_cache.len(), 3);
});
```

Key differences from the non-thread-safe version:
- Methods take `&self` instead of `&mut self`
- `get()` returns a clone of the value instead of a reference
- `with_lock()` method provides atomic multi-operation transactions

### `ShardedSieveCache` - High-Performance Thread-Safe Cache

For applications with high concurrency requirements, the `ShardedSieveCache` implementation uses multiple internal locks (sharding) to reduce contention and improve throughput:

```rust
use sieve_cache::ShardedSieveCache;
use std::thread;
use std::sync::Arc;

// Create a sharded cache with default shard count (16)
// We use Arc for sharing between threads
let cache = Arc::new(ShardedSieveCache::new(100000).unwrap());

// Alternatively, specify a custom number of shards
// let cache = Arc::new(ShardedSieveCache::with_shards(100000, 32).unwrap());

// Insert data from the main thread
cache.insert("foo".to_string(), "foocontent".to_string());

// Use multiple worker threads to insert data concurrently
let mut handles = vec![];
for i in 0..8 {
    let cache_clone = Arc::clone(&cache);
    let handle = thread::spawn(move || {
        // Each thread inserts multiple values
        for j in 0..100 {
            let key = format!("key_thread{}_item{}", i, j);
            let value = format!("value_{}", j);
            cache_clone.insert(key, value);
        }
    });
    handles.push(handle);
}

// Wait for all threads to complete
for handle in handles {
    handle.join().unwrap();
}

// Shard-specific atomic operations
// with_key_lock locks only the shard containing the key
cache.with_key_lock(&"foo", |shard| {
    // Operations inside this closure have exclusive access to the specific shard
    shard.insert("related_key1".to_string(), "value1".to_string());
    shard.insert("related_key2".to_string(), "value2".to_string());

    // We can check internal state within the transaction
    assert!(shard.contains_key(&"related_key1".to_string()));
});

// Get the number of entries across all shards
// Note: This acquires all shard locks sequentially
let total_entries = cache.len();
assert_eq!(total_entries, 803); // 800 from threads + 1 "foo" + 2 related keys

// Access shard information
println!("Cache has {} shards with total capacity {}",
         cache.num_shards(), cache.capacity());
```

#### How Sharding Works

The `ShardedSieveCache` divides the cache into multiple independent segments (shards), each protected by its own mutex. When an operation is performed on a key:

1. The key is hashed to determine which shard it belongs to
2. Only that shard's lock is acquired for the operation
3. Operations on keys in different shards can proceed in parallel

This design significantly reduces lock contention when operations are distributed across different keys, making it ideal for high-concurrency workloads.

## Feature Flags

This crate provides the following feature flags to control which implementations are available:

- `sync`: Enables the thread-safe `SyncSieveCache` implementation (enabled by default)
- `sharded`: Enables the sharded `ShardedSieveCache` implementation (enabled by default)

If you only need specific implementations, you can select just the features you need:

```toml
# Only use the core implementation
sieve-cache = { version = "1.0.0", default-features = false }

# Only use the core and sync implementations
sieve-cache = { version = "1.0.0", default-features = false, features = ["sync"] }

# Only use the core and sharded implementations
sieve-cache = { version = "1.0.0", default-features = false, features = ["sharded"] }

# For documentation tests to work correctly
sieve-cache = { version = "1.0.0", features = ["doctest"] }
```

## Performance Considerations

Choosing the right cache implementation depends on your workload:

- **Single-threaded usage**: Use `SieveCache` - it's the most efficient with the lowest overhead
- **Moderate concurrency**: Use `SyncSieveCache` - simple and effective with moderate thread count
- **High concurrency**: Use `ShardedSieveCache` - best performance with many threads accessing different keys
  - Sharding is most effective when operations are distributed across many keys
  - If most operations target the same few keys (which map to the same shards), the benefits may be limited
  - Generally, 16-32 shards provide a good balance of concurrency and overhead for most applications


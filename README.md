[![dependency status](https://deps.rs/repo/github/jedisct1/rust-sieve-cache/status.svg)](https://deps.rs/repo/github/jedisct1/rust-sieve-cache)

# SIEVE cache

This is an implementation of the [SIEVE](http://sievecache.com) cache replacement algorithm for Rust.

- [API documentation](https://docs.rs/sieve-cache/0.1.2/sieve_cache)
- [`crates.io` page](https://crates.io/crates/sieve-cache)

SIEVE is an eviction algorithm simpler than LRU that achieves state-of-the-art efficiency on skewed workloads.

This implementation exposes the same API as the `clock-pro` and `arc-cache` crates, so it can be used as a drop-in replacement for them in existing applications.

## Usage example

```rust
use sieve_cache::SieveCache;

// Create a new cache with a capacity of 100000.
let mut cache: SieveCache<String, String> = SieveCache::new(100000).unwrap();

// Insert key/value pairs into the cache.
cache.insert("foo".to_string(), "foocontent".to_string());
cache.insert("bar".to_string(), "barcontent".to_string());

// Remove an entry from the cache.
cache.remove("bar");

// Retrieve a value from the cache, returning `None` or a reference to the value.
assert_eq!(cache.get("foo"), Some(&"foocontent".to_string()));

// Check if a key is in the cache.
assert_eq!(cache.contains_key("bar"), false);

// Get a mutable reference to a value in the cache.
if let Some(value) = cache.get_mut("foo") {
   *value = "newfoocontent".to_string();
}

// Return the number of cached values.
assert_eq!(cache.len(), 1);

// Return the capacity of the cache.
assert_eq!(cache.capacity(), 100000);
```

## Thread-safe Cache Example

You can also use the thread-safe wrapper `SyncSieveCache` for concurrent access:

```rust
use sieve_cache::SyncSieveCache;
use std::thread;

// Create a thread-safe cache with a capacity of 100000.
let cache = SyncSieveCache::new(100000).unwrap();
let cache_clone = cache.clone();

// Insert key/value pairs into the cache from the main thread.
cache.insert("foo".to_string(), "foocontent".to_string());

// Access the cache from another thread.
let handle = thread::spawn(move || {
    cache_clone.insert("bar".to_string(), "barcontent".to_string());
    
    // Retrieve a value from the cache. Returns a clone of the value.
    assert_eq!(cache_clone.get(&"foo".to_string()), Some("foocontent".to_string()));
});

// Wait for the thread to complete.
handle.join().unwrap();

// Check if keys exist in the cache.
assert!(cache.contains_key(&"foo".to_string()));
assert!(cache.contains_key(&"bar".to_string()));

// Remove an entry from the cache.
cache.remove(&"bar".to_string());

// Return the number of cached values.
assert_eq!(cache.len(), 1);

// Perform multiple operations atomically.
cache.with_lock(|inner_cache| {
    inner_cache.insert("atomic1".to_string(), "value1".to_string());
    inner_cache.insert("atomic2".to_string(), "value2".to_string());
});
```


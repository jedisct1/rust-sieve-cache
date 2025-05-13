use crate::SieveCache;
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Default number of shards to use if not specified explicitly.
/// This value was chosen as a good default that balances memory overhead
/// with concurrency in most practical scenarios.
const DEFAULT_SHARDS: usize = 16;

/// A thread-safe implementation of `SieveCache` that uses multiple shards to reduce contention.
///
/// This provides better concurrency than `SyncSieveCache` by splitting the cache into multiple
/// independent shards, each protected by its own mutex. Operations on different shards can
/// proceed in parallel, which can significantly improve throughput in multi-threaded environments.
///
/// # How Sharding Works
///
/// The cache is partitioned into multiple independent segments (shards) based on the hash of the key.
/// Each shard has its own mutex, allowing operations on different shards to proceed concurrently.
/// This reduces lock contention compared to a single-mutex approach, especially under high
/// concurrency with access patterns distributed across different keys.
///
/// # Performance Considerations
///
/// - For workloads with high concurrency across different keys, `ShardedSieveCache` typically offers
///   better performance than `SyncSieveCache`
/// - The benefits increase with the number of concurrent threads and the distribution of keys
/// - More shards reduce contention but increase memory overhead
/// - If most operations target the same few keys (which map to the same shards), the benefits of
///   sharding may be limited
///
/// # Examples
///
/// ```
/// # use sieve_cache::ShardedSieveCache;
/// // Create a cache with default number of shards (16)
/// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(1000).unwrap();
///
/// // Or specify a custom number of shards
/// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::with_shards(1000, 32).unwrap();
///
/// cache.insert("key1".to_string(), "value1".to_string());
/// assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
/// ```
#[derive(Clone)]
pub struct ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    /// Array of shard mutexes, each containing a separate SieveCache instance
    shards: Vec<Arc<Mutex<SieveCache<K, V>>>>,
    /// Number of shards in the cache - kept as a separate field for quick access
    num_shards: usize,
}

impl<K, V> ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    /// Creates a new sharded cache with the specified capacity, using the default number of shards.
    ///
    /// The capacity will be divided evenly among the shards. The default shard count (16)
    /// provides a good balance between concurrency and memory overhead for most workloads.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(1000).unwrap();
    /// assert_eq!(cache.num_shards(), 16); // Default shard count
    /// ```
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        Self::with_shards(capacity, DEFAULT_SHARDS)
    }

    /// Creates a new sharded cache with the specified capacity and number of shards.
    ///
    /// The capacity will be divided among the shards, distributing any remainder to ensure
    /// the total capacity is at least the requested amount.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The total capacity of the cache
    /// * `num_shards` - The number of shards to divide the cache into
    ///
    /// # Errors
    ///
    /// Returns an error if either the capacity or number of shards is zero.
    ///
    /// # Performance Impact
    ///
    /// - More shards can reduce contention in highly concurrent environments
    /// - However, each shard has memory overhead, so very high shard counts may
    ///   increase memory usage without providing additional performance benefits
    /// - For most workloads, a value between 8 and 32 shards is optimal
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// // Create a cache with 8 shards
    /// let cache: ShardedSieveCache<String, u32> = ShardedSieveCache::with_shards(1000, 8).unwrap();
    /// assert_eq!(cache.num_shards(), 8);
    /// assert!(cache.capacity() >= 1000);
    /// ```
    pub fn with_shards(capacity: usize, num_shards: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("capacity must be greater than 0");
        }
        if num_shards == 0 {
            return Err("number of shards must be greater than 0");
        }

        // Calculate per-shard capacity
        let base_capacity_per_shard = capacity / num_shards;
        let remaining = capacity % num_shards;

        let mut shards = Vec::with_capacity(num_shards);
        for i in 0..num_shards {
            // Distribute the remaining capacity to the first 'remaining' shards
            let shard_capacity = if i < remaining {
                base_capacity_per_shard + 1
            } else {
                base_capacity_per_shard
            };

            // Ensure at least capacity 1 per shard
            let shard_capacity = std::cmp::max(1, shard_capacity);
            shards.push(Arc::new(Mutex::new(SieveCache::new(shard_capacity)?)));
        }

        Ok(Self { shards, num_shards })
    }

    /// Returns the shard index for a given key.
    ///
    /// This function computes a hash of the key and uses it to determine which shard
    /// the key belongs to.
    #[inline]
    fn get_shard_index<Q>(&self, key: &Q) -> usize
    where
        Q: Hash + ?Sized,
    {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        hash % self.num_shards
    }

    /// Returns a reference to the shard for a given key.
    ///
    /// This is an internal helper method that maps a key to its corresponding shard.
    #[inline]
    fn get_shard<Q>(&self, key: &Q) -> &Arc<Mutex<SieveCache<K, V>>>
    where
        Q: Hash + ?Sized,
    {
        let index = self.get_shard_index(key);
        &self.shards[index]
    }

    /// Returns a locked reference to the shard for a given key.
    ///
    /// This is an internal helper method to abstract away the lock handling and error recovery.
    /// If the mutex is poisoned due to a panic in another thread, the poison error is
    /// recovered from by calling `into_inner()` to access the underlying data.
    #[inline]
    fn locked_shard<Q>(&self, key: &Q) -> MutexGuard<'_, SieveCache<K, V>>
    where
        Q: Hash + ?Sized,
    {
        self.get_shard(key)
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns the total capacity of the cache (sum of all shard capacities).
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, u32> = ShardedSieveCache::new(1000).unwrap();
    /// assert!(cache.capacity() >= 1000);
    /// ```
    pub fn capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .capacity()
            })
            .sum()
    }

    /// Returns the total number of entries in the cache (sum of all shard lengths).
    ///
    /// Note that this operation requires acquiring a lock on each shard, so it may
    /// cause temporary contention if called frequently in a high-concurrency environment.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    ///
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock().unwrap_or_else(PoisonError::into_inner).len())
            .sum()
    }

    /// Returns `true` when no values are currently cached in any shard.
    ///
    /// Note that this operation requires acquiring a lock on each shard, so it may
    /// cause temporary contention if called frequently in a high-concurrency environment.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// assert!(cache.is_empty());
    ///
    /// cache.insert("key".to_string(), "value".to_string());
    /// assert!(!cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|shard| {
            shard
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        })
    }

    /// Returns `true` if there is a value in the cache mapped to by `key`.
    ///
    /// This operation only locks the specific shard containing the key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// assert!(cache.contains_key(&"key".to_string()));
    /// assert!(!cache.contains_key(&"missing".to_string()));
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        let mut guard = self.locked_shard(key);
        guard.contains_key(key)
    }

    /// Gets a clone of the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`. This operation only locks
    /// the specific shard containing the key.
    ///
    /// # Note
    ///
    /// This method returns a clone of the value rather than a reference, since the
    /// mutex guard would be dropped after this method returns. This means that
    /// `V` must implement `Clone`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// assert_eq!(cache.get(&"key".to_string()), Some("value".to_string()));
    /// assert_eq!(cache.get(&"missing".to_string()), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        V: Clone,
    {
        let mut guard = self.locked_shard(key);
        guard.get(key).cloned()
    }

    /// Maps `key` to `value` in the cache, possibly evicting old entries from the appropriate shard.
    ///
    /// This method returns `true` when this is a new entry, and `false` if an existing entry was
    /// updated. This operation only locks the specific shard containing the key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    ///
    /// // Insert a new key
    /// assert!(cache.insert("key1".to_string(), "value1".to_string()));
    ///
    /// // Update an existing key
    /// assert!(!cache.insert("key1".to_string(), "updated_value".to_string()));
    /// ```
    pub fn insert(&self, key: K, value: V) -> bool {
        let mut guard = self.locked_shard(&key);
        guard.insert(key, value)
    }

    /// Removes the cache entry mapped to by `key`.
    ///
    /// This method returns the value removed from the cache. If `key` did not map to any value,
    /// then this returns `None`. This operation only locks the specific shard containing the key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// // Remove an existing key
    /// assert_eq!(cache.remove(&"key".to_string()), Some("value".to_string()));
    ///
    /// // Attempt to remove a missing key
    /// assert_eq!(cache.remove(&"key".to_string()), None);
    /// ```
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let mut guard = self.locked_shard(key);
        guard.remove(key)
    }

    /// Removes and returns a value from the cache that was not recently accessed.
    ///
    /// This method tries to evict from each shard in turn until it finds a value to evict.
    /// If no suitable value exists in any shard, this returns `None`.
    ///
    /// # Note
    ///
    /// This implementation differs from the non-sharded version in that it checks each shard
    /// in sequence until it finds a suitable entry to evict. This may not provide the globally
    /// optimal eviction decision across all shards, but it avoids the need to lock all shards
    /// simultaneously.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::with_shards(10, 2).unwrap();
    ///
    /// // Fill the cache
    /// for i in 0..15 {
    ///     cache.insert(format!("key{}", i), format!("value{}", i));
    /// }
    ///
    /// // Should be able to evict something
    /// assert!(cache.evict().is_some());
    /// ```
    pub fn evict(&self) -> Option<V> {
        // Try each shard in turn
        for shard in &self.shards {
            let result = shard.lock().unwrap_or_else(PoisonError::into_inner).evict();
            if result.is_some() {
                return result;
            }
        }
        None
    }

    /// Gets exclusive access to a specific shard based on the key.
    ///
    /// This can be useful for performing multiple operations atomically on entries
    /// that share the same shard. Note that only keys that hash to the same shard
    /// can be manipulated within a single transaction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    ///
    /// // Perform multiple operations atomically
    /// cache.with_key_lock(&"foo", |shard| {
    ///     // All operations within this closure have exclusive access to the shard
    ///     shard.insert("key1".to_string(), "value1".to_string());
    ///     shard.insert("key2".to_string(), "value2".to_string());
    ///     
    ///     // We can check state mid-transaction
    ///     assert!(shard.contains_key(&"key1".to_string()));
    /// });
    /// ```
    pub fn with_key_lock<Q, F, T>(&self, key: &Q, f: F) -> T
    where
        Q: Hash + ?Sized,
        F: FnOnce(&mut SieveCache<K, V>) -> T,
    {
        let mut guard = self.locked_shard(key);
        f(&mut guard)
    }

    /// Returns the number of shards in this cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::with_shards(1000, 32).unwrap();
    /// assert_eq!(cache.num_shards(), 32);
    /// ```
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Gets a specific shard by index.
    ///
    /// This is mainly useful for advanced use cases and maintenance operations.
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::with_shards(1000, 8).unwrap();
    ///
    /// // Access a valid shard
    /// assert!(cache.get_shard_by_index(0).is_some());
    ///
    /// // Out of bounds index
    /// assert!(cache.get_shard_by_index(100).is_none());
    /// ```
    pub fn get_shard_by_index(&self, index: usize) -> Option<&Arc<Mutex<SieveCache<K, V>>>> {
        self.shards.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_sharded_cache_basics() {
        let cache = ShardedSieveCache::new(100).unwrap();

        // Insert a value
        assert!(cache.insert("key1".to_string(), "value1".to_string()));

        // Read back the value
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        // Check contains_key
        assert!(cache.contains_key(&"key1".to_string()));

        // Check capacity and length
        assert!(cache.capacity() >= 100); // May be slightly higher due to rounding up per shard
        assert_eq!(cache.len(), 1);

        // Remove a value
        assert_eq!(
            cache.remove(&"key1".to_string()),
            Some("value1".to_string())
        );
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_custom_shard_count() {
        let cache = ShardedSieveCache::with_shards(100, 4).unwrap();
        assert_eq!(cache.num_shards(), 4);

        for i in 0..10 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            cache.insert(key, value);
        }

        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_parallel_access() {
        let cache = Arc::new(ShardedSieveCache::with_shards(1000, 16).unwrap());
        let mut handles = vec![];

        // Spawn 8 threads that each insert 100 items
        for t in 0..8 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("thread{}key{}", t, i);
                    let value = format!("value{}_{}", t, i);
                    cache_clone.insert(key, value);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify total item count
        assert_eq!(cache.len(), 800);

        // Check a few random keys
        assert_eq!(
            cache.get(&"thread0key50".to_string()),
            Some("value0_50".to_string())
        );
        assert_eq!(
            cache.get(&"thread7key99".to_string()),
            Some("value7_99".to_string())
        );
    }

    #[test]
    fn test_with_key_lock() {
        let cache = ShardedSieveCache::new(100).unwrap();

        // Perform multiple operations atomically on keys that map to the same shard
        cache.with_key_lock(&"test_key", |shard| {
            shard.insert("key1".to_string(), "value1".to_string());
            shard.insert("key2".to_string(), "value2".to_string());
            shard.insert("key3".to_string(), "value3".to_string());
        });

        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_eviction() {
        let cache = ShardedSieveCache::with_shards(10, 2).unwrap();

        // Fill the cache
        for i in 0..15 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            cache.insert(key, value);
        }

        // The cache should not exceed its capacity
        assert!(cache.len() <= 10);

        // We should be able to evict items
        let evicted = cache.evict();
        assert!(evicted.is_some());
    }

    #[test]
    fn test_contention() {
        let cache = Arc::new(ShardedSieveCache::with_shards(1000, 16).unwrap());
        let mut handles = vec![];

        // Create keys that we know will hash to different shards
        let keys: Vec<String> = (0..16).map(|i| format!("shard_key_{}", i)).collect();

        // Spawn 16 threads, each hammering a different key
        for i in 0..16 {
            let cache_clone = Arc::clone(&cache);
            let key = keys[i].clone();

            let handle = thread::spawn(move || {
                for j in 0..1000 {
                    cache_clone.insert(key.clone(), format!("value_{}", j));
                    let _ = cache_clone.get(&key);

                    // Small sleep to make contention more likely
                    if j % 100 == 0 {
                        thread::sleep(Duration::from_micros(1));
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // All keys should still be present
        for key in keys {
            assert!(cache.contains_key(&key));
        }
    }
}

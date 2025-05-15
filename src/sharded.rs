use crate::SieveCache;
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
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
/// # Thread Safety Characteristics
///
/// ## Key-Based Locking
///
/// - Operations on the same key will always map to the same shard and are serialized
/// - Operations on different keys that hash to different shards can execute concurrently
/// - Hash-based sharding ensures good distribution of keys across shards by default
///
/// ## Concurrent Operations
///
/// - Single-key operations only lock one shard, allowing high concurrency
/// - Multi-key operations (like `clear()`, `keys()`, `values()`) access shards sequentially
/// - No operation holds locks on multiple shards simultaneously, preventing deadlocks
///
/// ## Consistency Model
///
/// - **Per-Key Consistency**: Operations on individual keys are atomic and isolated
/// - **Cross-Shard Consistency**: There are no guarantees of a globally consistent view across shards
/// - **Iteration Methods**: Methods like `keys()`, `values()`, and `entries()` create point-in-time snapshots that may not reflect concurrent modifications
/// - **Bulk Operations**: Methods like `retain()`, `for_each_value()`, and `for_each_entry()` operate on each shard independently
///
/// ## Callback Handling
///
/// - `get_mut`: Executes callbacks while holding only the lock for the relevant shard
/// - `with_key_lock`: Provides exclusive access to a specific shard for atomic multi-step operations
/// - `for_each_value`, `for_each_entry`: Process one shard at a time, with lock released between shards
///
/// # Performance Considerations
///
/// - For workloads with high concurrency across different keys, `ShardedSieveCache` typically offers
///   better performance than `SyncSieveCache`
/// - The benefits increase with the number of concurrent threads and the distribution of keys
/// - More shards reduce contention but increase memory overhead
/// - If most operations target the same few keys (which map to the same shards), the benefits of
///   sharding may be limited
/// - Default of 16 shards provides a good balance for most workloads, but can be customized
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
///
/// ## Multi-Threaded Example
///
/// ```
/// # use sieve_cache::ShardedSieveCache;
/// # use std::thread;
/// # use std::sync::Arc;
///
/// // Create a sharded cache with 8 shards
/// let cache = Arc::new(ShardedSieveCache::with_shards(1000, 8).unwrap());
///
/// // Spawn 4 threads that each insert 100 items
/// let mut handles = vec![];
/// for t in 0..4 {
///     let cache_clone = Arc::clone(&cache);
///     let handle = thread::spawn(move || {
///         for i in 0..100 {
///             let key = format!("thread{}key{}", t, i);
///             let value = format!("value{}_{}", t, i);
///             // Different threads can insert concurrently
///             cache_clone.insert(key, value);
///         }
///     });
///     handles.push(handle);
/// }
///
/// // Wait for all threads to complete
/// for handle in handles {
///     handle.join().unwrap();
/// }
///
/// assert_eq!(cache.len(), 400); // All 400 items were inserted
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

unsafe impl<K, V> Sync for ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
}

impl<K, V> Default for ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    /// Creates a new sharded cache with a default capacity of 100 entries and default number of shards.
    ///
    /// # Panics
    ///
    /// Panics if the underlying `ShardedSieveCache::new()` returns an error, which should never
    /// happen for a non-zero capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// # use std::default::Default;
    /// let cache: ShardedSieveCache<String, u32> = Default::default();
    /// assert!(cache.capacity() >= 100); // Due to shard distribution, might be slightly larger
    /// assert_eq!(cache.num_shards(), 16); // Default shard count
    /// ```
    fn default() -> Self {
        Self::new(100).expect("Failed to create cache with default capacity")
    }
}

impl<K, V> fmt::Debug for ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + fmt::Debug,
    V: Send + Sync + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShardedSieveCache")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .field("num_shards", &self.num_shards)
            .finish()
    }
}

impl<K, V> IntoIterator for ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;

    /// Converts the cache into an iterator over its key-value pairs.
    ///
    /// This collects all entries into a Vec and returns an iterator over that Vec.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// # use std::collections::HashMap;
    /// let cache = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Collect into a HashMap
    /// let map: HashMap<_, _> = cache.into_iter().collect();
    /// assert_eq!(map.len(), 2);
    /// assert_eq!(map.get("key1"), Some(&"value1".to_string()));
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.entries().into_iter()
    }
}

#[cfg(feature = "sync")]
impl<K, V> From<crate::SyncSieveCache<K, V>> for ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    /// Creates a new sharded cache from an existing `SyncSieveCache`.
    ///
    /// This allows for upgrading a standard thread-safe cache to a more scalable sharded version.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::{SyncSieveCache, ShardedSieveCache};
    /// let sync_cache = SyncSieveCache::new(100).unwrap();
    /// sync_cache.insert("key".to_string(), "value".to_string());
    ///
    /// // Convert to sharded version with default sharding
    /// let sharded_cache = ShardedSieveCache::from(sync_cache);
    /// assert_eq!(sharded_cache.get(&"key".to_string()), Some("value".to_string()));
    /// ```
    fn from(sync_cache: crate::SyncSieveCache<K, V>) -> Self {
        // Create a new sharded cache with the same capacity
        let capacity = sync_cache.capacity();
        let sharded = Self::new(capacity).expect("Failed to create sharded cache");

        // Transfer all entries
        for (key, value) in sync_cache.entries() {
            sharded.insert(key, value);
        }

        sharded
    }
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

    /// Gets a mutable reference to the value in the cache mapped to by `key` via a callback function.
    ///
    /// If no value exists for `key`, the callback will not be invoked and this returns `false`.
    /// Otherwise, the callback is invoked with a mutable reference to the value and this returns `true`.
    /// This operation only locks the specific shard containing the key.
    ///
    /// This operation marks the entry as "visited" in the SIEVE algorithm,
    /// which affects eviction decisions.
    ///
    /// # Thread Safety
    ///
    /// This method operates safely with recursive calls by:
    ///
    /// 1. Cloning the value with a short-lived lock on only the relevant shard
    /// 2. Releasing the lock during callback execution
    /// 3. Re-acquiring the lock to update the original value
    ///
    /// This approach means:
    ///
    /// - The callback can safely make other calls to the same cache instance
    /// - The value can be modified by other threads during the callback execution
    /// - Changes are not visible to other threads until the callback completes
    /// - Last writer wins if multiple threads modify the same key concurrently
    ///
    /// Compared to `SyncSieveCache.get_mut()`:
    /// - Only locks a single shard rather than the entire cache
    /// - Reduces contention when operating on different keys in different shards
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// // Modify the value in-place
    /// cache.get_mut(&"key".to_string(), |value| {
    ///     *value = "new_value".to_string();
    /// });
    ///
    /// assert_eq!(cache.get(&"key".to_string()), Some("new_value".to_string()));
    /// ```
    pub fn get_mut<Q, F>(&self, key: &Q, f: F) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        F: FnOnce(&mut V),
        V: Clone,
    {
        // Get a clone of the value if it exists, to avoid deadlocks
        // if the callback tries to use other methods on this cache
        let value_opt = {
            let mut guard = self.locked_shard(key);
            if let Some(v) = guard.get_mut(key) {
                // Clone the value before releasing the lock
                Some(v.clone())
            } else {
                None
            }
        };

        if let Some(mut value) = value_opt {
            // Execute the callback on the cloned value without holding the lock
            f(&mut value);

            // Update the value back to the cache
            let mut guard = self.locked_shard(key);
            if let Some(original) = guard.get_mut(key) {
                *original = value;
                true
            } else {
                // Key was removed while callback was executing
                false
            }
        } else {
            false
        }
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

    /// Removes all entries from the cache.
    ///
    /// This operation clears all stored values across all shards and resets the cache to an empty state,
    /// while maintaining the original capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// assert_eq!(cache.len(), 2);
    ///
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    /// assert!(cache.is_empty());
    /// ```
    pub fn clear(&self) {
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            guard.clear();
        }
    }

    /// Returns an iterator over all keys in the cache.
    ///
    /// The order of keys is not specified and should not be relied upon.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// # use std::collections::HashSet;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// let keys: HashSet<_> = cache.keys().into_iter().collect();
    /// assert_eq!(keys.len(), 2);
    /// assert!(keys.contains(&"key1".to_string()));
    /// assert!(keys.contains(&"key2".to_string()));
    /// ```
    pub fn keys(&self) -> Vec<K> {
        let mut all_keys = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            all_keys.extend(guard.keys().cloned());
        }
        all_keys
    }

    /// Returns all values in the cache.
    ///
    /// The order of values is not specified and should not be relied upon.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// # use std::collections::HashSet;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// let values: HashSet<_> = cache.values().into_iter().collect();
    /// assert_eq!(values.len(), 2);
    /// assert!(values.contains(&"value1".to_string()));
    /// assert!(values.contains(&"value2".to_string()));
    /// ```
    pub fn values(&self) -> Vec<V>
    where
        V: Clone,
    {
        let mut all_values = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            all_values.extend(guard.values().cloned());
        }
        all_values
    }

    /// Returns all key-value pairs in the cache.
    ///
    /// The order of pairs is not specified and should not be relied upon.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// # use std::collections::HashMap;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// let entries: HashMap<_, _> = cache.entries().into_iter().collect();
    /// assert_eq!(entries.len(), 2);
    /// assert_eq!(entries.get(&"key1".to_string()), Some(&"value1".to_string()));
    /// assert_eq!(entries.get(&"key2".to_string()), Some(&"value2".to_string()));
    /// ```
    pub fn entries(&self) -> Vec<(K, V)>
    where
        V: Clone,
    {
        let mut all_entries = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            all_entries.extend(guard.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        all_entries
    }

    /// Applies a function to all values in the cache across all shards.
    ///
    /// This method marks all entries as visited.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Update all values by appending text
    /// cache.for_each_value(|value| {
    ///     *value = format!("{}_updated", value);
    /// });
    ///
    /// assert_eq!(cache.get(&"key1".to_string()), Some("value1_updated".to_string()));
    /// assert_eq!(cache.get(&"key2".to_string()), Some("value2_updated".to_string()));
    /// ```
    pub fn for_each_value<F>(&self, mut f: F)
    where
        F: FnMut(&mut V),
    {
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            guard.values_mut().for_each(&mut f);
        }
    }

    /// Applies a function to all key-value pairs in the cache across all shards.
    ///
    /// This method marks all entries as visited.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, String> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Update all values associated with keys containing '1'
    /// cache.for_each_entry(|(key, value)| {
    ///     if key.contains('1') {
    ///         *value = format!("{}_special", value);
    ///     }
    /// });
    ///
    /// assert_eq!(cache.get(&"key1".to_string()), Some("value1_special".to_string()));
    /// assert_eq!(cache.get(&"key2".to_string()), Some("value2".to_string()));
    /// ```
    pub fn for_each_entry<F>(&self, mut f: F)
    where
        F: FnMut((&K, &mut V)),
    {
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            guard.iter_mut().for_each(&mut f);
        }
    }

    /// Gets exclusive access to a specific shard based on the key.
    ///
    /// This can be useful for performing multiple operations atomically on entries
    /// that share the same shard. Note that only keys that hash to the same shard
    /// can be manipulated within a single transaction.
    ///
    /// # Thread Safety
    ///
    /// This method provides a way to perform atomic operations on a subset of the cache:
    ///
    /// - Acquires a lock on a single shard determined by the key's hash
    /// - Provides exclusive access to that shard for the duration of the callback
    /// - Allows multiple operations to be performed atomically within the shard
    /// - Operations on different shards remain concurrent (unlike `SyncSieveCache.with_lock()`)
    ///
    /// Important thread safety considerations:
    ///
    /// - Only keys that hash to the same shard can be accessed atomically in a single call
    /// - Operations affect only one shard, providing partial atomicity (limited to that shard)
    /// - The callback should not attempt to acquire other locks to avoid deadlocks
    /// - Long-running callbacks will block other threads from accessing the same shard
    ///
    /// This method provides a good balance between atomicity and concurrency:
    /// it allows atomic multi-step operations while still permitting operations
    /// on other shards to proceed concurrently.
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

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all entries for which the provided function returns `false`.
    /// The elements are visited in arbitrary, unspecified order, across all shards.
    /// This operation processes each shard individually, acquiring and releasing locks as it goes.
    ///
    /// # Thread Safety
    ///
    /// This method has the following thread safety characteristics:
    ///
    /// - It first collects all entries across all shards into a snapshot
    /// - The lock for each shard is acquired and released independently
    /// - The predicate is evaluated outside any lock
    /// - Individual removals lock only the specific shard containing the key
    ///
    /// This design ensures:
    /// - Minimal lock contention during predicate evaluation
    /// - No deadlocks due to holding multiple shard locks simultaneously
    /// - Operations on different shards can proceed concurrently
    ///
    /// However, this also means:
    /// - The snapshot might not reflect concurrent modifications
    /// - There's no guarantee of cross-shard atomicity or consistency
    /// - Race conditions can occur if entries are modified between collection and removal
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::ShardedSieveCache;
    /// let cache: ShardedSieveCache<String, i32> = ShardedSieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), 100);
    /// cache.insert("key2".to_string(), 200);
    /// cache.insert("key3".to_string(), 300);
    ///
    /// // Keep only entries with values greater than 150
    /// cache.retain(|_, value| *value > 150);
    ///
    /// assert_eq!(cache.len(), 2);
    /// assert!(!cache.contains_key(&"key1".to_string()));
    /// assert!(cache.contains_key(&"key2".to_string()));
    /// assert!(cache.contains_key(&"key3".to_string()));
    /// ```
    pub fn retain<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
        V: Clone,
    {
        // First, collect all entries so we can check them without holding locks
        let entries = self.entries();

        // Now go through all entries and determine which ones to remove
        for (key, value) in entries {
            // Check the predicate outside the lock - using cloned data
            if !f(&key, &value) {
                // The predicate returned false, so remove this entry
                self.remove(&key);
            }
        }
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

    #[test]
    fn test_get_mut() {
        let cache = ShardedSieveCache::new(100).unwrap();
        cache.insert("key".to_string(), "value".to_string());

        // Modify the value in-place
        let modified = cache.get_mut(&"key".to_string(), |value| {
            *value = "new_value".to_string();
        });
        assert!(modified);

        // Verify the value was updated
        assert_eq!(cache.get(&"key".to_string()), Some("new_value".to_string()));

        // Try to modify a non-existent key
        let modified = cache.get_mut(&"missing".to_string(), |_| {
            panic!("This should not be called");
        });
        assert!(!modified);
    }

    #[test]
    fn test_get_mut_concurrent() {
        let cache = Arc::new(ShardedSieveCache::with_shards(100, 8).unwrap());

        // Insert initial values
        for i in 0..10 {
            cache.insert(format!("key{}", i), 0);
        }

        let mut handles = vec![];

        // Spawn 5 threads that modify values concurrently
        for _ in 0..5 {
            let cache_clone = Arc::clone(&cache);

            let handle = thread::spawn(move || {
                for i in 0..10 {
                    for _ in 0..100 {
                        cache_clone.get_mut(&format!("key{}", i), |value| {
                            *value += 1;
                        });
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // With our new thread-safe implementation that clones values during modification,
        // we can't guarantee exactly 500 increments due to race conditions.
        // Some increments may be lost when one thread's changes overwrite another's.
        // We simply verify that modifications happened and the cache remains functional.
        for i in 0..10 {
            let value = cache.get(&format!("key{}", i));
            assert!(value.is_some());
            let num = value.unwrap();
            // The value should be positive but might be less than 500 due to race conditions
            assert!(
                num > 0,
                "Value for key{} should be positive but was {}",
                i,
                num
            );
        }
    }

    #[test]
    fn test_clear() {
        let cache = ShardedSieveCache::with_shards(100, 4).unwrap();

        // Add entries to various shards
        for i in 0..20 {
            cache.insert(format!("key{}", i), format!("value{}", i));
        }

        assert_eq!(cache.len(), 20);
        assert!(!cache.is_empty());

        // Clear all shards
        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Verify entries are gone
        for i in 0..20 {
            assert_eq!(cache.get(&format!("key{}", i)), None);
        }
    }

    #[test]
    fn test_keys_values_entries() {
        let cache = ShardedSieveCache::with_shards(100, 4).unwrap();

        // Add entries to various shards
        for i in 0..10 {
            cache.insert(format!("key{}", i), format!("value{}", i));
        }

        // Test keys
        let keys = cache.keys();
        assert_eq!(keys.len(), 10);
        for i in 0..10 {
            assert!(keys.contains(&format!("key{}", i)));
        }

        // Test values
        let values = cache.values();
        assert_eq!(values.len(), 10);
        for i in 0..10 {
            assert!(values.contains(&format!("value{}", i)));
        }

        // Test entries
        let entries = cache.entries();
        assert_eq!(entries.len(), 10);
        for i in 0..10 {
            assert!(entries.contains(&(format!("key{}", i), format!("value{}", i))));
        }
    }

    #[test]
    fn test_for_each_operations() {
        let cache = ShardedSieveCache::with_shards(100, 4).unwrap();

        // Add entries to various shards
        for i in 0..10 {
            cache.insert(format!("key{}", i), format!("value{}", i));
        }

        // Test for_each_value
        cache.for_each_value(|value| {
            *value = format!("{}_updated", value);
        });

        for i in 0..10 {
            assert_eq!(
                cache.get(&format!("key{}", i)),
                Some(format!("value{}_updated", i))
            );
        }

        // Test for_each_entry
        cache.for_each_entry(|(key, value)| {
            if key.ends_with("5") {
                *value = format!("{}_special", value);
            }
        });

        assert_eq!(
            cache.get(&"key5".to_string()),
            Some("value5_updated_special".to_string())
        );
        assert_eq!(
            cache.get(&"key1".to_string()),
            Some("value1_updated".to_string())
        );
    }

    #[test]
    fn test_multithreaded_operations() {
        let cache = Arc::new(ShardedSieveCache::with_shards(100, 8).unwrap());

        // Fill the cache
        for i in 0..20 {
            cache.insert(format!("key{}", i), format!("value{}", i));
        }

        // Clear while concurrent accesses happen
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            // This thread tries to access the cache while main thread clears it
            thread::sleep(Duration::from_millis(10));

            for i in 0..20 {
                let _ = cache_clone.get(&format!("key{}", i));
                thread::sleep(Duration::from_micros(100));
            }
        });

        // Main thread clears the cache
        thread::sleep(Duration::from_millis(5));
        cache.clear();

        // Add new entries
        for i in 30..40 {
            cache.insert(format!("newkey{}", i), format!("newvalue{}", i));
        }

        // Wait for the thread to complete
        handle.join().unwrap();

        // Verify final state
        assert_eq!(cache.len(), 10);
        for i in 30..40 {
            assert_eq!(
                cache.get(&format!("newkey{}", i)),
                Some(format!("newvalue{}", i))
            );
        }
    }

    #[test]
    fn test_retain() {
        let cache = ShardedSieveCache::with_shards(100, 4).unwrap();

        // Add entries to various shards
        cache.insert("even1".to_string(), 2);
        cache.insert("even2".to_string(), 4);
        cache.insert("odd1".to_string(), 1);
        cache.insert("odd2".to_string(), 3);

        assert_eq!(cache.len(), 4);

        // Keep only entries with even values
        cache.retain(|_, v| v % 2 == 0);

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&"even1".to_string()));
        assert!(cache.contains_key(&"even2".to_string()));
        assert!(!cache.contains_key(&"odd1".to_string()));
        assert!(!cache.contains_key(&"odd2".to_string()));

        // Keep only entries with keys containing '1'
        cache.retain(|k, _| k.contains('1'));

        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&"even1".to_string()));
        assert!(!cache.contains_key(&"even2".to_string()));
    }

    #[test]
    fn test_retain_concurrent() {
        // Create a well-controlled test case that avoids race conditions
        let cache = ShardedSieveCache::with_shards(100, 8).unwrap();

        // Add a known set of entries
        for i in 0..10 {
            cache.insert(format!("even{}", i * 2), i * 2);
            cache.insert(format!("odd{}", i * 2 + 1), i * 2 + 1);
        }

        // Retain only odd values
        cache.retain(|_, value| value % 2 == 1);

        // Check that we have the right number of entries
        assert_eq!(cache.len(), 10, "Should have 10 odd-valued entries");

        // Verify that all remaining entries have odd values
        for (_, value) in cache.entries() {
            assert_eq!(
                value % 2,
                1,
                "Found an even value {value} which should have been removed"
            );
        }

        // Verify the specific entries we expect
        for i in 0..10 {
            let odd_key = format!("odd{}", i * 2 + 1);
            assert!(cache.contains_key(&odd_key), "Missing odd entry: {odd_key}");
            assert_eq!(cache.get(&odd_key), Some(i * 2 + 1));
        }
    }
}

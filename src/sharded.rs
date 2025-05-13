use crate::SieveCache;
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

const DEFAULT_SHARDS: usize = 16;

/// A thread-safe implementation of `SieveCache` that uses multiple shards to reduce contention.
///
/// This provides better concurrency than `SyncSieveCache` by splitting the cache into multiple
/// independent shards, each protected by its own mutex. Operations on different shards can
/// proceed in parallel.
#[derive(Clone)]
pub struct ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    shards: Vec<Arc<Mutex<SieveCache<K, V>>>>,
    num_shards: usize,
}

impl<K, V> ShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    /// Create a new sharded cache with the specified capacity.
    ///
    /// The capacity will be divided evenly among the shards.
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        Self::with_shards(capacity, DEFAULT_SHARDS)
    }

    /// Create a new sharded cache with the specified capacity and number of shards.
    ///
    /// The capacity will be divided among the shards, potentially allocating
    /// extra capacity to ensure the total is at least the requested amount.
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

    /// Get the shard index for a given key.
    fn get_shard_index<Q>(&self, key: &Q) -> usize
    where
        Q: Hash + ?Sized,
    {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        hash % self.num_shards
    }

    /// Get a reference to the shard for a given key.
    fn get_shard<Q>(&self, key: &Q) -> &Arc<Mutex<SieveCache<K, V>>>
    where
        Q: Hash + ?Sized,
    {
        let index = self.get_shard_index(key);
        &self.shards[index]
    }

    /// Return the total capacity of the cache (sum of all shard capacities).
    pub fn capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock().unwrap().capacity())
            .sum()
    }

    /// Return the total number of entries in the cache (sum of all shard lengths).
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock().unwrap().len())
            .sum()
    }

    /// Return `true` when no values are currently cached in any shard.
    pub fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|shard| shard.lock().unwrap().is_empty())
    }

    /// Return `true` if there is a value in the cache mapped to by `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        let shard = self.get_shard(key);
        shard.lock().unwrap().contains_key(key)
    }

    /// Get a clone of the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        V: Clone,
    {
        let shard = self.get_shard(key);
        let mut guard = shard.lock().unwrap();
        guard.get(key).cloned()
    }

    /// Map `key` to `value` in the cache, possibly evicting old entries from the appropriate shard.
    ///
    /// This method returns `true` when this is a new entry, and `false` if an existing entry was
    /// updated.
    pub fn insert(&self, key: K, value: V) -> bool {
        let shard = self.get_shard(&key);
        shard.lock().unwrap().insert(key, value)
    }

    /// Remove the cache entry mapped to by `key`.
    ///
    /// This method returns the value removed from the cache. If `key` did not map to any value,
    /// then this returns `None`.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let shard = self.get_shard(key);
        shard.lock().unwrap().remove(key)
    }

    /// Remove and return a value from the cache that was not recently accessed.
    ///
    /// This method tries to evict from each shard in turn until it finds a value to evict.
    /// If no suitable value exists in any shard, this returns `None`.
    pub fn evict(&self) -> Option<V> {
        // Try each shard in turn
        for shard in &self.shards {
            let result = shard.lock().unwrap().evict();
            if result.is_some() {
                return result;
            }
        }
        None
    }

    /// Get exclusive access to a specific shard based on the key.
    ///
    /// This can be useful for performing multiple operations atomically on entries
    /// that share the same shard.
    pub fn with_key_lock<Q, F, T>(&self, key: &Q, f: F) -> T
    where
        Q: Hash + ?Sized,
        F: FnOnce(&mut SieveCache<K, V>) -> T,
    {
        let shard = self.get_shard(key);
        let mut guard = shard.lock().unwrap();
        f(&mut guard)
    }

    /// Get the number of shards in this cache.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Get a specific shard by index.
    ///
    /// This is mainly useful for more advanced use cases and maintenance operations.
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

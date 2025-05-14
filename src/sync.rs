use crate::SieveCache;
use std::borrow::Borrow;
use std::hash::Hash;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A thread-safe wrapper around `SieveCache`.
///
/// This provides a thread-safe implementation of the SIEVE cache algorithm by wrapping
/// the standard `SieveCache` in an `Arc<Mutex<>>`. It offers the same functionality as
/// the underlying cache but with thread safety guarantees.
///
/// # Thread Safety
///
/// All operations acquire a lock on the entire cache, which provides strong consistency
/// but may become a bottleneck under high contention. For workloads with high concurrency,
/// consider using [`ShardedSieveCache`](crate::ShardedSieveCache) instead, which partitions
/// the cache into multiple independently-locked segments.
///
/// # Examples
///
/// ```
/// # use sieve_cache::SyncSieveCache;
/// let cache = SyncSieveCache::new(100).unwrap();
///
/// // Multiple threads can safely access the cache
/// cache.insert("key1".to_string(), "value1".to_string());
/// assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
/// ```
#[derive(Clone)]
pub struct SyncSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    inner: Arc<Mutex<SieveCache<K, V>>>,
}

impl<K, V> SyncSieveCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync,
{
    /// Creates a new thread-safe cache with the given capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::<String, String>::new(100).unwrap();
    /// ```
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        let cache = SieveCache::new(capacity)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(cache)),
        })
    }

    /// Returns a locked reference to the underlying cache.
    ///
    /// This is an internal helper method to abstract away the lock handling.
    /// If the mutex is poisoned due to a panic in another thread, the poison
    /// error is recovered from by calling `into_inner()` to access the underlying data.
    #[inline]
    fn locked_cache(&self) -> MutexGuard<'_, SieveCache<K, V>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns the capacity of the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::<String, u32>::new(100).unwrap();
    /// assert_eq!(cache.capacity(), 100);
    /// ```
    pub fn capacity(&self) -> usize {
        self.locked_cache().capacity()
    }

    /// Returns the number of cached values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    /// assert_eq!(cache.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.locked_cache().len()
    }

    /// Returns `true` when no values are currently cached.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::<String, String>::new(100).unwrap();
    /// assert!(cache.is_empty());
    ///
    /// cache.insert("key".to_string(), "value".to_string());
    /// assert!(!cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.locked_cache().is_empty()
    }

    /// Returns `true` if there is a value in the cache mapped to by `key`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
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
        let mut guard = self.locked_cache();
        guard.contains_key(key)
    }

    /// Gets a clone of the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    ///
    /// # Note
    ///
    /// Unlike the unwrapped `SieveCache`, this returns a clone of the value
    /// rather than a reference, since the mutex guard would be dropped after
    /// this method returns. This means that `V` must implement `Clone`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
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
        let mut guard = self.locked_cache();
        guard.get(key).cloned()
    }

    /// Gets a mutable reference to the value in the cache mapped to by `key` via a callback function.
    ///
    /// If no value exists for `key`, the callback will not be invoked and this returns `false`.
    /// Otherwise, the callback is invoked with a mutable reference to the value and this returns `true`.
    ///
    /// This operation marks the entry as "visited" in the SIEVE algorithm,
    /// which affects eviction decisions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
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
    {
        let mut guard = self.locked_cache();
        if let Some(value) = guard.get_mut(key) {
            f(value);
            true
        } else {
            false
        }
    }

    /// Maps `key` to `value` in the cache, possibly evicting old entries.
    ///
    /// This method returns `true` when this is a new entry, and `false` if an existing entry was
    /// updated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
    ///
    /// // Insert a new key
    /// assert!(cache.insert("key1".to_string(), "value1".to_string()));
    ///
    /// // Update an existing key
    /// assert!(!cache.insert("key1".to_string(), "updated_value".to_string()));
    /// ```
    pub fn insert(&self, key: K, value: V) -> bool {
        let mut guard = self.locked_cache();
        guard.insert(key, value)
    }

    /// Removes the cache entry mapped to by `key`.
    ///
    /// This method returns the value removed from the cache. If `key` did not map to any value,
    /// then this returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
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
        let mut guard = self.locked_cache();
        guard.remove(key)
    }

    /// Removes and returns a value from the cache that was not recently accessed.
    ///
    /// This implements the SIEVE eviction algorithm to select an entry for removal.
    /// If no suitable value exists, this returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(2).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Accessing key1 marks it as recently used
    /// assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
    ///
    /// // Insert a new key, which should evict key2
    /// cache.insert("key3".to_string(), "value3".to_string());
    ///
    /// // key2 should have been evicted
    /// assert_eq!(cache.get(&"key2".to_string()), None);
    /// ```
    pub fn evict(&self) -> Option<V> {
        let mut guard = self.locked_cache();
        guard.evict()
    }

    /// Gets exclusive access to the underlying cache to perform multiple operations atomically.
    ///
    /// This is useful when you need to perform a series of operations that depend on each other
    /// and you want to ensure that no other thread can access the cache between operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sieve_cache::SyncSieveCache;
    /// let cache = SyncSieveCache::new(100).unwrap();
    ///
    /// cache.with_lock(|inner_cache| {
    ///     // Perform multiple operations atomically
    ///     inner_cache.insert("key1".to_string(), "value1".to_string());
    ///     inner_cache.insert("key2".to_string(), "value2".to_string());
    ///
    ///     // We can check internal state mid-transaction
    ///     assert_eq!(inner_cache.len(), 2);
    /// });
    /// ```
    pub fn with_lock<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut SieveCache<K, V>) -> T,
    {
        let mut guard = self.locked_cache();
        f(&mut guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_sync_cache() {
        let cache = SyncSieveCache::new(100).unwrap();

        // Insert a value
        assert!(cache.insert("key1".to_string(), "value1".to_string()));

        // Read back the value
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));

        // Check contains_key
        assert!(cache.contains_key(&"key1".to_string()));

        // Check capacity and length
        assert_eq!(cache.capacity(), 100);
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
    fn test_multithreaded_access() {
        let cache = SyncSieveCache::new(100).unwrap();
        let cache_clone = cache.clone();

        // Add some initial data
        cache.insert("shared".to_string(), "initial".to_string());

        // Spawn a thread that updates the cache
        let thread = thread::spawn(move || {
            cache_clone.insert("shared".to_string(), "updated".to_string());
            cache_clone.insert("thread_only".to_string(), "thread_value".to_string());
        });

        // Main thread operations
        cache.insert("main_only".to_string(), "main_value".to_string());

        // Wait for thread to complete
        thread.join().unwrap();

        // Verify results
        assert_eq!(
            cache.get(&"shared".to_string()),
            Some("updated".to_string())
        );
        assert_eq!(
            cache.get(&"thread_only".to_string()),
            Some("thread_value".to_string())
        );
        assert_eq!(
            cache.get(&"main_only".to_string()),
            Some("main_value".to_string())
        );
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_with_lock() {
        let cache = SyncSieveCache::new(100).unwrap();

        // Perform multiple operations atomically
        cache.with_lock(|inner_cache| {
            inner_cache.insert("key1".to_string(), "value1".to_string());
            inner_cache.insert("key2".to_string(), "value2".to_string());
            inner_cache.insert("key3".to_string(), "value3".to_string());
        });

        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_get_mut() {
        let cache = SyncSieveCache::new(100).unwrap();
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
}

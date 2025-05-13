use crate::SieveCache;
use std::borrow::Borrow;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// A thread-safe wrapper around `SieveCache`.
///
/// This provides a thread-safe implementation of the SIEVE cache
/// algorithm by wrapping the standard `SieveCache` in an `Arc<Mutex<>>`.
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
    /// Create a new thread-safe cache with the given capacity.
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        let cache = SieveCache::new(capacity)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(cache)),
        })
    }

    /// Return the capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.inner.lock().unwrap().capacity()
    }

    /// Return the number of cached values.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Return `true` when no values are currently cached.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// Return `true` if there is a value in the cache mapped to by `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        self.inner.lock().unwrap().contains_key(key)
    }

    /// Get a clone of the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    ///
    /// Note: Unlike the unwrapped SieveCache, this returns a clone of the value
    /// rather than a reference, since the mutex guard would be dropped.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        V: Clone,
    {
        let mut guard = self.inner.lock().unwrap();
        guard.get(key).cloned()
    }

    /// Map `key` to `value` in the cache, possibly evicting old entries.
    ///
    /// This method returns `true` when this is a new entry, and `false` if an existing entry was
    /// updated.
    pub fn insert(&self, key: K, value: V) -> bool {
        self.inner.lock().unwrap().insert(key, value)
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
        self.inner.lock().unwrap().remove(key)
    }

    /// Remove and return a value from the cache that was not recently accessed.
    /// If no such value exists, this returns `None`.
    pub fn evict(&self) -> Option<V> {
        self.inner.lock().unwrap().evict()
    }

    /// Get exclusive access to the underlying cache.
    ///
    /// This can be useful for performing multiple operations atomically.
    pub fn with_lock<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut SieveCache<K, V>) -> T,
    {
        let mut guard = self.inner.lock().unwrap();
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
}

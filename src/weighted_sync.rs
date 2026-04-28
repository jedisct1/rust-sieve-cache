use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, RandomState};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::weighted::{Weigh, WeightedSieveCache};

/// A thread-safe, memory-bounded cache that tracks the total weight of
/// stored entries and evicts when a budget is exceeded.
///
/// This wraps [`WeightedSieveCache`] in an `Arc<Mutex<>>` to provide
/// thread safety. The weight bookkeeping lives inside the mutex alongside
/// the cache data so that every mutation is atomic with respect to weight.
///
/// # Type Parameters
///
/// * `K` - The key type, which must implement `Eq`, `Hash`, `Clone`, `Weigh`, `Send`, and `Sync`
/// * `V` - The value type, must implement `Weigh`, `Send`, and `Sync`
/// * `S` - The hash builder type for the underlying `HashMap` (default: `RandomState`). Must implement `BuildHasher`
///
/// For higher concurrency, see
/// [`WeightedShardedSieveCache`](crate::WeightedShardedSieveCache).
#[derive(Clone)]
pub struct WeightedSyncSieveCache<K, V, S = RandomState>
where
    K: Eq + Hash + Clone + Weigh + Send + Sync,
    V: Weigh + Send + Sync,
    S: BuildHasher,
{
    inner: Arc<Mutex<WeightedSieveCache<K, V, S>>>,
}

impl<K, V> WeightedSyncSieveCache<K, V>
where
    K: Eq + Hash + Clone + Weigh + Send + Sync,
    V: Weigh + Send + Sync,
{
    /// Creates a new thread-safe weighted cache.
    ///
    /// Returns `Err` if `capacity` or `max_weight` is 0.
    pub fn new(capacity: usize, max_weight: usize) -> Result<Self, &'static str> {
        Self::new_with_hasher(capacity, max_weight, Default::default())
    }
}

impl<K, V, S> WeightedSyncSieveCache<K, V, S>
where
    K: Eq + Hash + Clone + Weigh + Send + Sync,
    V: Weigh + Send + Sync,
    S: BuildHasher + Clone,
{
    /// Creates a new thread-safe weighted cache with a custom hash builder.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of entries in the cache
    /// * `max_weight` - The memory budget in bytes
    /// * `hasher` - A hash builder instance (e.g., from `ahash::AHasher` or `std::collections::hash_map::RandomState`)
    ///
    /// Returns `Err` if `capacity` or `max_weight` is 0.
    pub fn new_with_hasher(
        capacity: usize,
        max_weight: usize,
        hasher: S,
    ) -> Result<Self, &'static str> {
        let cache = WeightedSieveCache::new_with_hasher(capacity, max_weight, hasher)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(cache)),
        })
    }

    #[inline]
    fn locked_cache(&self) -> MutexGuard<'_, WeightedSieveCache<K, V, S>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns the maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.locked_cache().capacity()
    }

    /// Returns the current number of entries.
    pub fn len(&self) -> usize {
        self.locked_cache().len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.locked_cache().is_empty()
    }

    /// Returns `true` if the cache contains `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        self.locked_cache().contains_key(key)
    }

    /// Returns a clone of the value for `key`, marking it as visited.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        V: Clone,
    {
        self.locked_cache().get(key).cloned()
    }

    /// Mutates the value for `key` via a callback (clone-modify-update).
    pub fn get_mut<Q, F>(&self, key: &Q, f: F) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        F: FnOnce(&mut V),
        V: Clone,
    {
        let value_opt = {
            let mut guard = self.locked_cache();
            guard.get_mut(key).map(|v| v.clone())
        };

        if let Some(mut value) = value_opt {
            f(&mut value);
            let mut guard = self.locked_cache();
            if let Some(original) = guard.get_mut(key) {
                *original = value;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Inserts a key-value pair, enforcing the weight budget.
    pub fn insert(&self, key: K, value: V) -> bool {
        self.locked_cache().insert(key, value)
    }

    /// Removes the entry for `key`.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.locked_cache().remove(key)
    }

    /// Evicts one entry, returning the key-value pair.
    pub fn evict(&self) -> Option<(K, V)> {
        self.locked_cache().evict()
    }

    /// Evicts one entry from the underlying `SieveCache`, returning the
    /// key-value pair. Guarantees `None` only when the cache is truly empty.
    pub fn evict_pair(&self) -> Option<(K, V)> {
        self.locked_cache().evict()
    }

    /// Returns the current total weight.
    pub fn current_weight(&self) -> usize {
        self.locked_cache().current_weight()
    }

    /// Returns the maximum weight budget.
    pub fn max_weight(&self) -> usize {
        self.locked_cache().max_weight()
    }

    /// Clears all entries, resetting weight to 0.
    pub fn clear(&self) {
        self.locked_cache().clear();
    }

    /// Returns cloned keys.
    pub fn keys(&self) -> Vec<K> {
        self.locked_cache().keys().cloned().collect()
    }

    /// Returns cloned values.
    pub fn values(&self) -> Vec<V>
    where
        V: Clone,
    {
        self.locked_cache().values().cloned().collect()
    }

    /// Returns cloned key-value pairs.
    pub fn entries(&self) -> Vec<(K, V)>
    where
        V: Clone,
    {
        self.locked_cache()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Retains entries matching the predicate, adjusting weight.
    pub fn retain<F>(&self, f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        self.locked_cache().retain(f);
    }

    /// Gets exclusive access to the underlying weighted cache.
    pub fn with_lock<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut WeightedSieveCache<K, V, S>) -> T,
    {
        let mut guard = self.locked_cache();
        f(&mut guard)
    }
}

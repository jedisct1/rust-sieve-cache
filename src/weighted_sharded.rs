use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::weighted::{Weigh, WeightedSieveCache};

const DEFAULT_SHARDS: usize = 16;

/// A thread-safe, sharded, memory-bounded cache.
///
/// Each shard independently tracks and enforces its own weight budget,
/// avoiding cross-shard synchronization. The total `max_weight` is
/// distributed across shards: `max_weight / num_shards` per shard, with
/// the remainder given to the first N shards (mirroring the capacity
/// distribution).
///
/// # Type Parameters
///
/// * `K` - The key type, which must implement `Eq`, `Hash`, `Clone`, `Weigh`, `Send`, and `Sync`
/// * `V` - The value type, must implement `Weigh`, `Send`, and `Sync`
/// * `S` - The hash builder type for the underlying `HashMap` (default: `RandomState`). Must implement `BuildHasher`
///
/// # Weight overshoot
///
/// Because each shard enforces independently, the worst-case total
/// overshoot is `num_shards * max_single_entry_weight`.
#[derive(Clone)]
pub struct WeightedShardedSieveCache<K, V, S = RandomState>
where
    K: Eq + Hash + Clone + Weigh + Send + Sync,
    V: Weigh + Send + Sync,
    S: BuildHasher + Clone
{
    shards: Vec<Arc<Mutex<WeightedSieveCache<K, V, S>>>>,
    num_shards: usize,
    max_weight: usize,
    hasher: S
}

impl<K, V> WeightedShardedSieveCache<K, V>
where
    K: Eq + Hash + Clone + Weigh + Send + Sync,
    V: Weigh + Send + Sync,
{
    /// Creates a new sharded weighted cache with the default number of shards (16).
    pub fn new(capacity: usize, max_weight: usize) -> Result<Self, &'static str> {
        Self::with_shards(capacity, max_weight, DEFAULT_SHARDS)
    }

    /// Creates a new sharded weighted cache with a custom number of shards.
    ///
    /// Returns `Err` if `capacity`, `max_weight`, or `num_shards` is 0,
    /// or if `max_weight < num_shards` (each shard needs at least 1 byte
    /// of budget).
    pub fn with_shards(
        capacity: usize,
        max_weight: usize,
        num_shards: usize,
    ) -> Result<Self, &'static str> {
        Self::with_shards_and_hasher(capacity, max_weight, num_shards, Default::default())
    }
}

impl<K, V, S> WeightedShardedSieveCache<K, V, S>
where
    K: Eq + Hash + Clone + Weigh + Send + Sync,
    V: Weigh + Send + Sync,
    S: BuildHasher + Clone
{
    /// Creates a new sharded weighted cache with the default number of shards (16) and a custom hash builder.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The total capacity of the cache
    /// * `max_weight` - The memory budget in bytes
    /// * `hasher` - A hash builder instance (e.g., from `ahash::AHasher` or `std::collections::hash_map::RandomState`)
    ///
    /// Returns `Err` if `capacity` or `max_weight` is 0.
    pub fn new_with_hasher(capacity: usize, max_weight: usize, hasher: S) -> Result<Self, &'static str> {
        Self::with_shards_and_hasher(capacity, max_weight, DEFAULT_SHARDS, hasher)
    }

    /// Creates a new sharded weighted cache with a custom number of shards and hash builder.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The total capacity of the cache
    /// * `max_weight` - The memory budget in bytes
    /// * `num_shards` - The number of shards to divide the cache into
    /// * `hasher` - A hash builder instance (e.g., from `ahash::AHasher` or `std::collections::hash_map::RandomState`)
    ///
    /// Returns `Err` if `capacity`, `max_weight`, or `num_shards` is 0,
    /// or if `max_weight < num_shards` (each shard needs at least 1 byte
    /// of budget).
    pub fn with_shards_and_hasher(
        capacity: usize,
        max_weight: usize,
        num_shards: usize,
        hasher: S
    ) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("capacity must be greater than 0");
        }
        if max_weight == 0 {
            return Err("max_weight must be greater than zero");
        }
        if num_shards == 0 {
            return Err("number of shards must be greater than 0");
        }
        if max_weight < num_shards {
            return Err("max_weight must be >= num_shards (each shard needs at least 1 byte)");
        }

        let base_capacity = capacity / num_shards;
        let cap_remaining = capacity % num_shards;

        let base_weight = max_weight / num_shards;
        let weight_remaining = max_weight % num_shards;

        let mut shards = Vec::with_capacity(num_shards);
        for i in 0..num_shards {
            let shard_capacity = if i < cap_remaining {
                base_capacity + 1
            } else {
                base_capacity
            };
            let shard_capacity = std::cmp::max(1, shard_capacity);

            let shard_weight = if i < weight_remaining {
                base_weight + 1
            } else {
                base_weight
            };
            // base_weight >= 1 is guaranteed by the max_weight >= num_shards check above

            let cache = WeightedSieveCache::new_with_hasher(shard_capacity, shard_weight, hasher.clone())?;
            shards.push(Arc::new(Mutex::new(cache)));
        }

        Ok(Self {
            shards,
            num_shards,
            max_weight,
            hasher
        })
    }

    #[inline]
    fn get_shard_index<Q>(&self, key: &Q) -> usize
    where
        Q: Hash + ?Sized,
    {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        hash % self.num_shards
    }

    #[inline]
    fn locked_shard<Q>(&self, key: &Q) -> MutexGuard<'_, WeightedSieveCache<K, V, S>>
    where
        Q: Hash + ?Sized,
    {
        let index = self.get_shard_index(key);
        self.shards[index]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns the total capacity across all shards.
    pub fn capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().unwrap_or_else(PoisonError::into_inner).capacity())
            .sum()
    }

    /// Returns the total number of entries across all shards.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().unwrap_or_else(PoisonError::into_inner).len())
            .sum()
    }

    /// Returns `true` if all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|s| s.lock().unwrap_or_else(PoisonError::into_inner).is_empty())
    }

    /// Returns `true` if the cache contains `key`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        self.locked_shard(key).contains_key(key)
    }

    /// Returns a clone of the value for `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        V: Clone,
    {
        self.locked_shard(key).get(key).cloned()
    }

    /// Mutates the value for `key` via a callback.
    pub fn get_mut<Q, F>(&self, key: &Q, f: F) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
        F: FnOnce(&mut V),
        V: Clone,
    {
        let value_opt = {
            let mut guard = self.locked_shard(key);
            guard.get_mut(key).map(|v| v.clone())
        };

        if let Some(mut value) = value_opt {
            f(&mut value);
            let mut guard = self.locked_shard(key);
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

    /// Inserts a key-value pair, enforcing the shard's weight budget.
    pub fn insert(&self, key: K, value: V) -> bool {
        self.locked_shard(&key).insert(key, value)
    }

    /// Removes the entry for `key`.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.locked_shard(key).remove(key)
    }

    /// Evicts one entry from the first non-empty shard, returning the
    /// key-value pair.
    pub fn evict(&self) -> Option<(K, V)> {
        for shard in &self.shards {
            let result = shard.lock().unwrap_or_else(PoisonError::into_inner).evict();
            if result.is_some() {
                return result;
            }
        }
        None
    }

    /// Evicts one entry, same as [`evict`](Self::evict).
    pub fn evict_pair(&self) -> Option<(K, V)> {
        self.evict()
    }

    /// Returns the total current weight across all shards.
    pub fn current_weight(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                s.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .current_weight()
            })
            .fold(0usize, usize::saturating_add)
    }

    /// Returns the maximum weight budget (the constructor argument).
    pub fn max_weight(&self) -> usize {
        self.max_weight
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Clears all shards, resetting all weights to 0.
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.lock().unwrap_or_else(PoisonError::into_inner).clear();
        }
    }

    /// Returns all keys across all shards.
    pub fn keys(&self) -> Vec<K> {
        let mut result = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            result.extend(guard.keys().cloned());
        }
        result
    }

    /// Returns all values across all shards.
    pub fn values(&self) -> Vec<V>
    where
        V: Clone,
    {
        let mut result = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            result.extend(guard.values().cloned());
        }
        result
    }

    /// Returns all key-value pairs across all shards.
    pub fn entries(&self) -> Vec<(K, V)>
    where
        V: Clone,
    {
        let mut result = Vec::new();
        for shard in &self.shards {
            let guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            result.extend(guard.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        result
    }

    /// Retains entries matching the predicate across all shards.
    pub fn retain<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            guard.retain(&mut f);
        }
    }
}

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, RandomState};
use std::mem;

use crate::SieveCache;

/// A trait for types that can report their memory weight.
///
/// Implementations should return an estimate of the heap memory used by the
/// value, plus the stack size of the value itself. The returned value is in
/// bytes and is used by [`WeightedSieveCache`] to enforce a memory budget.
pub trait Weigh {
    fn weigh(&self) -> usize;
}

macro_rules! impl_weigh_fixed {
    ($($t:ty),*) => {
        $(
            impl Weigh for $t {
                #[inline]
                fn weigh(&self) -> usize {
                    mem::size_of::<Self>()
                }
            }
        )*
    };
}

impl_weigh_fixed!(
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    bool,
    char,
    ()
);

impl Weigh for String {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<String>() + self.capacity()
    }
}

impl<T> Weigh for Vec<T> {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<Vec<T>>() + self.len() * mem::size_of::<T>()
    }
}

impl Weigh for Box<[u8]> {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<Box<[u8]>>() + self.len()
    }
}

impl<T: Weigh> Weigh for Box<T> {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<Box<T>>() + (**self).weigh()
    }
}

impl<T: Weigh> Weigh for Option<T> {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<Option<T>>()
            + match self {
                Some(v) => v.weigh().saturating_sub(mem::size_of::<T>()),
                None => 0,
            }
    }
}

impl Weigh for &str {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<&str>()
    }
}

impl Weigh for &[u8] {
    #[inline]
    fn weigh(&self) -> usize {
        mem::size_of::<&[u8]>()
    }
}

/// A memory-bounded wrapper around [`SieveCache`] that tracks the total
/// weight of stored entries and evicts when a budget is exceeded.
///
/// Weight is computed at insertion time via the [`Weigh`] trait. Mutating
/// a value through [`get_mut`](Self::get_mut) does **not** update the
/// tracked weight — this matches the behavior of other weight-aware caches
/// like moka and quick_cache. Users who need accurate tracking after
/// mutation can `remove` + `insert`.
///
/// # Type Parameters
///
/// * `K` - The key type, which must implement `Eq`, `Hash`, `Clone`, and `Weigh`
/// * `V` - The value type, must implement `Weigh`
/// * `S` - The hash builder type for the underlying `HashMap` (default: `RandomState`). Must implement `BuildHasher`
///
/// # Weight overshoot
///
/// When a single entry's weight exceeds `max_weight`, the eviction loop
/// empties the cache and then inserts the oversized entry as the sole
/// occupant. This means `current_weight` can temporarily exceed
/// `max_weight` by at most the weight of one entry.
pub struct WeightedSieveCache<K: Eq + Hash + Clone + Weigh, V: Weigh, S: BuildHasher = RandomState> {
    inner: SieveCache<K, V, S>,
    /// Per-entry charged weight, snapshotted at insert time. This is the
    /// weight that will be subtracted when the entry is removed or evicted,
    /// regardless of any mutations via `get_mut`.
    charged: HashMap<K, usize, S>,
    max_weight: usize,
    current_weight: usize,
}

impl<K: Eq + Hash + Clone + Weigh, V: Weigh> WeightedSieveCache<K, V> {
    /// Creates a new weighted cache.
    ///
    /// `capacity` is the maximum number of entries (passed to the inner
    /// `SieveCache`). `max_weight` is the memory budget in bytes — the
    /// cache will evict entries to stay at or below this limit.
    ///
    /// Returns `Err` if `capacity` is 0 or `max_weight` is 0.
    pub fn new(capacity: usize, max_weight: usize) -> Result<Self, &'static str> {
        Self::new_with_hasher(capacity, max_weight, Default::default())
    }
}

impl<K: Eq + Hash + Clone + Weigh, V: Weigh, S: BuildHasher + Clone> WeightedSieveCache<K, V, S> {
    /// Creates a new weighted cache using a custom hash builder.
    ///
    /// `capacity` is the maximum number of entries (passed to the inner
    /// `SieveCache`). `max_weight` is the memory budget in bytes — the
    /// cache will evict entries to stay at or below this limit.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of entries in the cache
    /// * `max_weight` - The memory budget in bytes
    /// * `hasher` - A hash builder instance (e.g., from `ahash::AHasher` or `std::collections::hash_map::RandomState`)
    ///
    /// Returns `Err` if `capacity` is 0 or `max_weight` is 0.
    pub fn new_with_hasher(capacity: usize, max_weight: usize, hasher: S) -> Result<Self, &'static str> {
        if max_weight == 0 {
            return Err("max_weight must be greater than zero");
        }
        let inner = SieveCache::new_with_hasher(capacity, hasher.clone())?;
        Ok(Self {
            inner,
            charged: HashMap::with_hasher(hasher),
            max_weight,
            current_weight: 0,
        })
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the key already exists, its value is updated and the weight is
    /// adjusted. After insertion, entries are evicted until `current_weight
    /// <= max_weight` or the cache contains only the newly inserted entry.
    ///
    /// Returns `true` if the key was newly inserted, `false` if it was
    /// updated.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        if let Some(&old_charged) = self.charged.get(&key) {
            self.current_weight = self.current_weight.saturating_sub(old_charged);
        }

        // Evict for capacity before inner.insert, which would silently
        // evict without giving us the key to update charged weights.
        if !self.inner.contains_key(&key) && self.inner.len() >= self.inner.capacity() {
            if let Some((k, _v)) = self.inner.evict_pair() {
                if let Some(w) = self.charged.remove(&k) {
                    self.current_weight = self.current_weight.saturating_sub(w);
                }
            }
        }

        let key_clone = key.clone();
        let is_new = self.inner.insert(key, value);

        // Snapshot weight from the stored entry — the node's cloned key
        // may have different capacity than the original.
        let new_weight = self
            .inner
            .get_key_value(&key_clone)
            .map(|(k, v)| k.weigh().saturating_add(v.weigh()))
            .unwrap_or(0);
        self.current_weight = self.current_weight.saturating_add(new_weight);
        self.charged.insert(key_clone.clone(), new_weight);

        if is_new {
            self.inner.get(&key_clone);
        }

        while self.current_weight > self.max_weight {
            if self.inner.len() <= 1 {
                break;
            }
            match self.inner.evict_pair() {
                Some((k, v)) => {
                    if k == key_clone {
                        self.inner.insert(k, v);
                        self.inner.get(&key_clone);
                    } else {
                        debug_assert!(
                            self.charged.contains_key(&k),
                            "evicted key missing from charged weight index"
                        );
                        let w = self
                            .charged
                            .remove(&k)
                            .unwrap_or_else(|| k.weigh().saturating_add(v.weigh()));
                        self.current_weight = self.current_weight.saturating_sub(w);
                    }
                }
                None => break,
            }
        }

        is_new
    }

    /// Removes the entry for `key`, returning its value if it existed.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let removed = self.inner.remove(key);
        if removed.is_some() {
            if let Some(w) = self.charged.remove(key) {
                self.current_weight = self.current_weight.saturating_sub(w);
            }
        }
        removed
    }

    /// Evicts one entry from the cache, returning the key-value pair.
    ///
    /// Returns `None` only when the cache is empty. Subtracts the evicted
    /// pair's weight from `current_weight`.
    pub fn evict(&mut self) -> Option<(K, V)> {
        let (k, v) = self.inner.evict_pair()?;
        debug_assert!(
            self.charged.contains_key(&k),
            "evicted key missing from charged weight index"
        );
        let w = self
            .charged
            .remove(&k)
            .unwrap_or_else(|| k.weigh().saturating_add(v.weigh()));
        self.current_weight = self.current_weight.saturating_sub(w);
        Some((k, v))
    }

    /// Returns the current total weight of all entries.
    #[inline]
    pub fn current_weight(&self) -> usize {
        self.current_weight
    }

    /// Returns the maximum weight budget.
    #[inline]
    pub fn max_weight(&self) -> usize {
        self.max_weight
    }

    /// Returns a reference to the value for `key`, marking it as visited.
    #[inline]
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get(key)
    }

    /// Returns a mutable reference to the value for `key`, marking it as
    /// visited.
    ///
    /// **Note:** mutating the value does not update the tracked weight.
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get_mut(key)
    }

    /// Returns `true` if the cache contains the given key.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.contains_key(key)
    }

    /// Returns the number of entries in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the maximum number of entries the cache can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns `true` if the cache contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns an iterator over the keys.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.inner.keys()
    }

    /// Returns an iterator over the values.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.inner.values()
    }

    /// Returns an iterator over key-value pairs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter()
    }

    /// Removes all entries, resetting `current_weight` to 0.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.charged.clear();
        self.current_weight = 0;
    }

    /// Retains only the entries for which the predicate returns `true`.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        self.inner.retain(f);
        self.charged.retain(|k, _| self.inner.contains_key(k));
        self.current_weight = self
            .charged
            .values()
            .copied()
            .fold(0usize, usize::saturating_add);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weigh_primitives() {
        assert_eq!(42u64.weigh(), 8);
        assert_eq!(true.weigh(), 1);
        assert_eq!('x'.weigh(), 4);
    }

    #[test]
    fn test_weigh_string() {
        let s = String::from("hello");
        assert_eq!(s.weigh(), mem::size_of::<String>() + s.capacity());
    }

    #[test]
    fn test_weigh_vec() {
        let v: Vec<u32> = vec![1, 2, 3];
        assert_eq!(v.weigh(), mem::size_of::<Vec<u32>>() + 3 * 4);
    }

    #[test]
    fn test_weigh_box_slice() {
        let b: Box<[u8]> = vec![0u8; 10].into_boxed_slice();
        assert_eq!(b.weigh(), mem::size_of::<Box<[u8]>>() + 10);
    }

    #[test]
    fn test_basic_weight_tracking() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 1000).unwrap();

        let k = "key1".to_string();
        let v = "value1".to_string();
        let expected_weight = k.weigh() + v.weigh();

        cache.insert(k, v);
        assert_eq!(cache.current_weight(), expected_weight);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_weight_on_update() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10000).unwrap();

        let k = "key1".to_string();
        let v1 = "short".to_string();
        let v2 = "a much longer string value".to_string();

        cache.insert(k.clone(), v1);
        let w1 = cache.current_weight();

        cache.insert(k.clone(), v2.clone());
        let w2 = cache.current_weight();

        let expected = k.weigh() + v2.weigh();
        assert_eq!(w2, expected);
        assert_ne!(w1, w2);
    }

    #[test]
    fn test_weight_on_remove() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10000).unwrap();

        cache.insert("k1".to_string(), "v1".to_string());
        cache.insert("k2".to_string(), "v2".to_string());
        let w_before = cache.current_weight();

        let k = "k1".to_string();
        let expected_removed = k.weigh() + "v1".to_string().weigh();
        cache.remove("k1");

        assert_eq!(cache.current_weight(), w_before - expected_removed);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_eviction_enforces_weight() {
        // Use a tight weight budget
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(100, 200).unwrap();

        // Insert entries until weight is enforced
        for i in 0..20 {
            cache.insert(format!("key{}", i), format!("value{}", i));
        }

        // Weight should be at or below max
        assert!(cache.current_weight() <= cache.max_weight());
        assert!(cache.len() < 20);
    }

    #[test]
    fn test_oversized_single_entry() {
        // A single entry bigger than max_weight
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10).unwrap();

        let big_value = "x".repeat(1000);
        cache.insert("k".to_string(), big_value.clone());

        // The entry stays as the sole occupant
        assert_eq!(cache.len(), 1);
        assert!(cache.current_weight() > cache.max_weight());
        assert_eq!(cache.get("k"), Some(&big_value));
    }

    #[test]
    fn test_eviction_loop_terminates() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(5, 100).unwrap();

        // Fill and mark all visited
        for i in 0..5 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }
        for i in 0..5 {
            cache.get(&format!("k{}", i));
        }

        // Insert something that pushes over budget — should not hang
        let big = "x".repeat(200);
        cache.insert("overflow".to_string(), big);
        // Should have evicted everything except the new entry
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key("overflow"));
    }

    #[test]
    fn test_clear_resets_weight() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10000).unwrap();

        cache.insert("k".to_string(), "v".to_string());
        assert!(cache.current_weight() > 0);

        cache.clear();
        assert_eq!(cache.current_weight(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_retain_adjusts_weight() {
        let mut cache: WeightedSieveCache<String, u64> =
            WeightedSieveCache::new(10, 10000).unwrap();

        cache.insert("keep".to_string(), 1);
        cache.insert("drop".to_string(), 2);
        let keep_weight = "keep".to_string().weigh() + 1u64.weigh();

        cache.retain(|_, v| *v == 1);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.current_weight(), keep_weight);
    }

    #[test]
    fn test_max_weight_zero_rejected() {
        let result: Result<WeightedSieveCache<String, String>, _> = WeightedSieveCache::new(10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_evict_returns_pair() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(5, 10000).unwrap();

        cache.insert("a".to_string(), "1".to_string());
        cache.insert("b".to_string(), "2".to_string());

        let pair = cache.evict();
        assert!(pair.is_some());
        let (k, _v) = pair.unwrap();
        assert!(!cache.contains_key(&k));
    }
}

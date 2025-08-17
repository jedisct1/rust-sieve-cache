#![doc = include_str!("../README.md")]

//! # SIEVE Cache for Rust
//!
//! This crate provides implementations of the SIEVE cache replacement algorithm:
//!
//! * [`SieveCache`] - The core single-threaded implementation (always available)

#[cfg(feature = "sync")]
pub mod _docs_sync {}

#[cfg(feature = "sharded")]
pub mod _docs_sharded {}

pub mod _sieve_algorithm {
    //! ## The SIEVE Algorithm
    //!
    //! SIEVE (Simple, space-efficient, In-memory, EViction mEchanism) is a cache eviction
    //! algorithm that maintains a single bit per entry to track whether an item has been
    //! "visited" since it was last considered for eviction. This approach requires less
    //! state than LRU but achieves excellent performance, especially on skewed workloads.
}

pub mod _cache_implementation {
    //! ## Cache Implementation Details
    //!
    //! The cache is implemented as a combination of:
    //!
    //! 1. A `HashMap` for O(1) key lookups
    //! 2. A vector-based ordered collection for managing the eviction order
    //! 3. A "visited" flag on each entry to track recent access
    //!
    //! When the cache is full and a new item is inserted, the eviction algorithm:
    //! 1. Starts from the "hand" position (eviction candidate)
    //! 2. Finds the first non-visited entry, evicting it
    //! 3. Marks all visited entries as non-visited while searching
}

pub mod _implementation_choice {
    //! ## Choosing the Right Implementation
    //!
    //! - Use [`SieveCache`] for single-threaded applications
}

/// [DOCTEST_ONLY]
///
/// ```rust
/// # #[cfg(feature = "sync")]
/// # fn sync_example() {
/// # use sieve_cache::SyncSieveCache;
/// # let cache = SyncSieveCache::<String, u32>::new(1000).unwrap();
/// # let recommended = cache.recommended_capacity(0.5, 2.0, 0.3, 0.7);
/// # println!("Recommended capacity: {}", recommended);
/// # }
/// #
/// # #[cfg(feature = "sharded")]
/// # fn sharded_example() {
/// # use sieve_cache::ShardedSieveCache;
/// # let cache = ShardedSieveCache::<String, u32>::new(1000).unwrap();
/// # let recommended = cache.recommended_capacity(0.5, 2.0, 0.3, 0.7);
/// # println!("Recommended capacity: {}", recommended);
/// # }
/// ```
fn _readme_examples_doctest() {
    // This function exists only to host the doctest above
    // This ensures doctests from the README.md can be validated
}

#[cfg(feature = "sync")]
pub mod _docs_sync_usage {
    //! - Use [`SyncSieveCache`] for multi-threaded applications with moderate concurrency
}

#[cfg(feature = "sharded")]
pub mod _docs_sharded_usage {
    //! - Use [`ShardedSieveCache`] for applications with high concurrency where operations
    //!   are distributed across many different keys
}

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

#[cfg(feature = "sharded")]
mod sharded;
#[cfg(feature = "sync")]
mod sync;

#[cfg(feature = "sharded")]
pub use sharded::ShardedSieveCache;
#[cfg(feature = "sync")]
pub use sync::SyncSieveCache;

/// Internal representation of a cache entry
#[derive(Clone)]
struct Node<K: Eq + Hash + Clone, V> {
    key: K,
    value: V,
    visited: bool,
}

impl<K: Eq + Hash + Clone, V> Node<K, V> {
    fn new(key: K, value: V) -> Self {
        Self {
            key,
            value,
            visited: false,
        }
    }
}

/// A cache based on the SIEVE eviction algorithm.
///
/// `SieveCache` provides an efficient in-memory cache with a carefully designed eviction
/// strategy that balances simplicity with good performance characteristics, especially on
/// skewed workloads common in real-world applications.
///
/// This is the single-threaded implementation.
#[cfg(feature = "sync")]
/// For thread-safe variants, see [`SyncSieveCache`] (with the `sync` feature)
#[cfg(feature = "sharded")]
/// and [`ShardedSieveCache`] (with the `sharded` feature).
///
/// # Type Parameters
///
/// * `K` - The key type, which must implement `Eq`, `Hash`, and `Clone`
/// * `V` - The value type, with no constraints
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "doctest")]
/// # {
/// use sieve_cache::SieveCache;
///
/// // Create a new cache with capacity for 1000 items
/// let mut cache = SieveCache::new(1000).unwrap();
///
/// // Insert some values
/// cache.insert("key1".to_string(), "value1".to_string());
/// cache.insert("key2".to_string(), "value2".to_string());
///
/// // Retrieve values - returns references to the values
/// assert_eq!(cache.get("key1"), Some(&"value1".to_string()));
///
/// // Check if the cache contains a key
/// assert!(cache.contains_key("key1"));
/// assert!(!cache.contains_key("missing_key"));
///
/// // Get a mutable reference to modify a value
/// if let Some(value) = cache.get_mut("key1") {
///     *value = "modified".to_string();
/// }
///
/// // Verify the modification
/// assert_eq!(cache.get("key1"), Some(&"modified".to_string()));
///
/// // Remove a value
/// let removed = cache.remove("key2");
/// assert_eq!(removed, Some("value2".to_string()));
/// # }
/// ```
pub struct SieveCache<K: Eq + Hash + Clone, V> {
    /// Map of keys to indices in the nodes vector
    map: HashMap<K, usize>,
    /// Vector of all cache nodes
    nodes: Vec<Node<K, V>>,
    /// Index to the "hand" pointer used by the SIEVE algorithm for eviction
    hand: Option<usize>,
    /// Maximum number of entries the cache can hold
    capacity: usize,
}

impl<K: Eq + Hash + Clone, V> SieveCache<K, V> {
    /// Creates a new cache with the given capacity.
    ///
    /// The capacity represents the maximum number of key-value pairs
    /// that can be stored in the cache. When this limit is reached,
    /// the cache will use the SIEVE algorithm to evict entries.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity is zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// // Create a cache with space for 100 entries
    /// let cache: SieveCache<String, u32> = SieveCache::new(100).unwrap();
    ///
    /// // Capacity of zero is invalid
    /// let invalid = SieveCache::<String, u32>::new(0);
    /// assert!(invalid.is_err());
    /// # }
    /// ```
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("capacity must be greater than 0");
        }
        Ok(Self {
            map: HashMap::with_capacity(capacity),
            nodes: Vec::with_capacity(capacity),
            hand: None,
            capacity,
        })
    }

    /// Returns the capacity of the cache.
    ///
    /// This is the maximum number of entries that can be stored before
    /// the cache begins evicting old entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let cache = SieveCache::<String, u32>::new(100).unwrap();
    /// assert_eq!(cache.capacity(), 100);
    /// # }
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of cached values.
    ///
    /// This value will never exceed the capacity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert("key".to_string(), "value".to_string());
    /// assert_eq!(cache.len(), 1);
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when no values are currently cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::<String, u32>::new(100).unwrap();
    /// assert!(cache.is_empty());
    ///
    /// cache.insert("key".to_string(), "value".to_string());
    /// assert!(!cache.is_empty());
    ///
    /// cache.remove("key");
    /// assert!(cache.is_empty());
    /// # }
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns `true` if there is a value in the cache mapped to by `key`.
    ///
    /// This operation does not count as an access for the SIEVE algorithm's
    /// visited flag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// assert!(cache.contains_key("key"));
    /// assert!(!cache.contains_key("missing"));
    /// # }
    /// ```
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        self.map.contains_key(key)
    }

    /// Gets an immutable reference to the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    ///
    /// This operation marks the entry as "visited" in the SIEVE algorithm,
    /// which affects eviction decisions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// assert_eq!(cache.get("key"), Some(&"value".to_string()));
    /// assert_eq!(cache.get("missing"), None);
    /// # }
    /// ```
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        if let Some(&idx) = self.map.get(key) {
            // Mark as visited for the SIEVE algorithm
            self.nodes[idx].visited = true;
            Some(&self.nodes[idx].value)
        } else {
            None
        }
    }

    /// Gets a mutable reference to the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    ///
    /// This operation marks the entry as "visited" in the SIEVE algorithm,
    /// which affects eviction decisions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key".to_string(), "value".to_string());
    ///
    /// // Modify the value in-place
    /// if let Some(value) = cache.get_mut("key") {
    ///     *value = "new_value".to_string();
    /// }
    ///
    /// assert_eq!(cache.get("key"), Some(&"new_value".to_string()));
    /// # }
    /// ```
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: Hash + Eq + ?Sized,
        K: Borrow<Q>,
    {
        if let Some(&idx) = self.map.get(key) {
            // Mark as visited for the SIEVE algorithm
            self.nodes[idx].visited = true;
            Some(&mut self.nodes[idx].value)
        } else {
            None
        }
    }

    /// Maps `key` to `value` in the cache, possibly evicting old entries.
    ///
    /// If the key already exists, its value is updated and the entry is marked as visited.
    /// If the key doesn't exist and the cache is at capacity, the SIEVE algorithm is used
    /// to evict an entry before the new key-value pair is inserted.
    ///
    /// # Return Value
    ///
    /// Returns `true` when this is a new entry, and `false` if an existing entry was updated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    ///
    /// // Insert a new entry
    /// let is_new = cache.insert("key1".to_string(), "value1".to_string());
    /// assert!(is_new); // Returns true for new entries
    ///
    /// // Update an existing entry
    /// let is_new = cache.insert("key1".to_string(), "updated".to_string());
    /// assert!(!is_new); // Returns false for updates
    ///
    /// assert_eq!(cache.get("key1"), Some(&"updated".to_string()));
    /// # }
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> bool {
        // Check if key already exists
        if let Some(&idx) = self.map.get(&key) {
            // Update existing entry
            self.nodes[idx].visited = true;
            self.nodes[idx].value = value;
            return false;
        }

        // Evict if at capacity
        if self.nodes.len() >= self.capacity {
            let item = self.evict();
            // When the cache is full and *all* entries are marked `visited`, our `evict()` performs
            // a first pass that clears the `visited` bits but may return `None` without removing
            // anything. We still must free a slot before inserting, so we call `evict()` a second
            // time. This mirrors the original SIEVE miss path, which keeps scanning (wrapping once)
            // until it finds an item to evict after clearing bits.
            if item.is_none() {
                let item = self.evict();
                debug_assert!(
                    item.is_some(),
                    "evict() must remove one entry when at capacity"
                );
            }
        }

        // Add new node to the front
        let node = Node::new(key.clone(), value);
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        self.map.insert(key, idx);
        debug_assert!(self.nodes.len() < self.capacity);
        true
    }

    /// Removes the cache entry mapped to by `key`.
    ///
    /// This method explicitly removes an entry from the cache regardless of its
    /// "visited" status.
    ///
    /// # Return Value
    ///
    /// Returns the value removed from the cache. If `key` did not map to any value,
    /// then this returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Remove an existing entry
    /// let removed = cache.remove("key1");
    /// assert_eq!(removed, Some("value1".to_string()));
    ///
    /// // Try to remove a non-existent entry
    /// let removed = cache.remove("key1");
    /// assert_eq!(removed, None);
    ///
    /// assert_eq!(cache.len(), 1); // Only one entry remains
    /// # }
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        // Find the node index
        let idx = self.map.remove(key)?;

        // If this is the last element, just remove it
        if idx == self.nodes.len() - 1 {
            return Some(self.nodes.pop().unwrap().value);
        }

        // Update hand if needed
        if let Some(hand_idx) = self.hand {
            if hand_idx == idx {
                // Move hand to the previous node or wrap to end
                self.hand = if idx > 0 {
                    Some(idx - 1)
                } else {
                    Some(self.nodes.len() - 2)
                };
            } else if hand_idx == self.nodes.len() - 1 {
                // If hand points to the last element (which will be moved to idx)
                self.hand = Some(idx);
            }
        }

        // Remove the node by replacing it with the last one and updating the map
        let last_node = self.nodes.pop().unwrap();
        let removed_value = if idx < self.nodes.len() {
            // Only need to swap and update map if not the last element
            let last_key = last_node.key.clone(); // Clone the key before moving
            let removed_node = std::mem::replace(&mut self.nodes[idx], last_node);
            self.map.insert(last_key, idx); // Update map for swapped node
            removed_node.value
        } else {
            last_node.value
        };

        Some(removed_value)
    }

    /// Removes and returns a value from the cache that was not recently accessed.
    ///
    /// This method implements the SIEVE eviction algorithm:
    /// 1. Starting from the "hand" pointer or the end of the vector, look for an entry
    ///    that hasn't been visited since the last scan
    /// 2. Mark all visited entries as non-visited during the scan
    /// 3. If a non-visited entry is found, evict it
    /// 4. Update the hand to point to the previous node
    ///
    /// # Return Value
    ///
    /// If a suitable entry is found, it is removed from the cache and its value is returned.
    /// If all entries have been recently accessed or the cache is empty, this returns `None`.
    ///
    /// # Note
    ///
    /// This method is automatically called by `insert` when the cache is at capacity,
    /// but it can also be called manually to proactively free space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(3).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    /// cache.insert("key3".to_string(), "value3".to_string());
    ///
    /// // Access key1 and key2 to mark them as visited
    /// cache.get("key1");
    /// cache.get("key2");
    ///
    /// // key3 hasn't been accessed, so it should be evicted
    /// let evicted = cache.evict();
    /// assert!(evicted.is_some());
    /// assert!(!cache.contains_key("key3")); // key3 was evicted
    /// # }
    /// ```
    pub fn evict(&mut self) -> Option<V> {
        if self.nodes.is_empty() {
            return None;
        }

        // Start from the hand pointer or the end if no hand
        let mut current_idx = self.hand.unwrap_or(self.nodes.len() - 1);
        let start_idx = current_idx;

        // Track whether we've wrapped around and whether we found a node to evict
        let mut wrapped = false;
        let mut found_idx = None;

        // Scan for a non-visited entry
        loop {
            // If current node is not visited, mark it for eviction
            if !self.nodes[current_idx].visited {
                found_idx = Some(current_idx);
                break;
            }

            // Mark as non-visited for next scan
            self.nodes[current_idx].visited = false;

            // Move to previous node or wrap to end
            current_idx = if current_idx > 0 {
                current_idx - 1
            } else {
                // Wrap around to end of vector
                if wrapped {
                    // If we've already wrapped, break to avoid infinite loop
                    break;
                }
                wrapped = true;
                self.nodes.len() - 1
            };

            // If we've looped back to start, we've checked all nodes
            if current_idx == start_idx {
                break;
            }
        }

        // If we found a node to evict
        if let Some(idx) = found_idx {
            // Update the hand pointer to the previous node or wrap to end
            self.hand = if idx > 0 {
                Some(idx - 1)
            } else if self.nodes.len() > 1 {
                Some(self.nodes.len() - 2)
            } else {
                None
            };

            // Remove the key from the map
            let key = &self.nodes[idx].key;
            self.map.remove(key);

            // Remove the node and return its value
            if idx == self.nodes.len() - 1 {
                // If last node, just pop it
                return Some(self.nodes.pop().unwrap().value);
            } else {
                // Otherwise swap with the last node
                let last_node = self.nodes.pop().unwrap();
                let last_key = last_node.key.clone(); // Clone the key before moving
                let removed_node = std::mem::replace(&mut self.nodes[idx], last_node);
                // Update the map for the moved node
                self.map.insert(last_key, idx);
                return Some(removed_node.value);
            }
        }

        None
    }

    /// Removes all entries from the cache.
    ///
    /// This operation clears all stored values and resets the cache to an empty state,
    /// while maintaining the original capacity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// assert_eq!(cache.len(), 2);
    ///
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    /// assert!(cache.is_empty());
    /// # }
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
        self.nodes.clear();
        self.hand = None;
    }

    /// Returns an iterator over all keys in the cache.
    ///
    /// The order of keys is not specified and should not be relied upon.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    /// use std::collections::HashSet;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// let keys: HashSet<_> = cache.keys().collect();
    /// assert_eq!(keys.len(), 2);
    /// assert!(keys.contains(&"key1".to_string()));
    /// assert!(keys.contains(&"key2".to_string()));
    /// # }
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.nodes.iter().map(|node| &node.key)
    }

    /// Returns an iterator over all values in the cache.
    ///
    /// The order of values is not specified and should not be relied upon.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    /// use std::collections::HashSet;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// let values: HashSet<_> = cache.values().collect();
    /// assert_eq!(values.len(), 2);
    /// assert!(values.contains(&"value1".to_string()));
    /// assert!(values.contains(&"value2".to_string()));
    /// # }
    /// ```
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.nodes.iter().map(|node| &node.value)
    }

    /// Returns an iterator over all mutable values in the cache.
    ///
    /// The order of values is not specified and should not be relied upon.
    /// Note that iterating through this will mark all entries as visited.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Update all values by appending text
    /// for value in cache.values_mut() {
    ///     *value = format!("{}_updated", value);
    /// }
    ///
    /// assert_eq!(cache.get("key1"), Some(&"value1_updated".to_string()));
    /// assert_eq!(cache.get("key2"), Some(&"value2_updated".to_string()));
    /// # }
    /// ```
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        for node in &mut self.nodes {
            node.visited = true;
        }
        self.nodes.iter_mut().map(|node| &mut node.value)
    }

    /// Returns an iterator over all key-value pairs in the cache.
    ///
    /// The order of pairs is not specified and should not be relied upon.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    /// use std::collections::HashMap;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// let entries: HashMap<_, _> = cache.iter().collect();
    /// assert_eq!(entries.len(), 2);
    /// assert_eq!(entries.get(&"key1".to_string()), Some(&&"value1".to_string()));
    /// assert_eq!(entries.get(&"key2".to_string()), Some(&&"value2".to_string()));
    /// # }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.nodes.iter().map(|node| (&node.key, &node.value))
    }

    /// Returns an iterator over all key-value pairs in the cache, with mutable references to values.
    ///
    /// The order of pairs is not specified and should not be relied upon.
    /// Note that iterating through this will mark all entries as visited.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), "value1".to_string());
    /// cache.insert("key2".to_string(), "value2".to_string());
    ///
    /// // Update all values associated with keys containing '1'
    /// for (key, value) in cache.iter_mut() {
    ///     if key.contains('1') {
    ///         *value = format!("{}_special", value);
    ///     }
    /// }
    ///
    /// assert_eq!(cache.get("key1"), Some(&"value1_special".to_string()));
    /// assert_eq!(cache.get("key2"), Some(&"value2".to_string()));
    /// # }
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        for node in &mut self.nodes {
            node.visited = true;
        }
        self.nodes
            .iter_mut()
            .map(|node| (&node.key, &mut node.value))
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all entries for which the provided function returns `false`.
    /// The elements are visited in arbitrary, unspecified order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "doctest")]
    /// # {
    /// use sieve_cache::SieveCache;
    ///
    /// let mut cache = SieveCache::new(100).unwrap();
    /// cache.insert("key1".to_string(), 100);
    /// cache.insert("key2".to_string(), 200);
    /// cache.insert("key3".to_string(), 300);
    ///
    /// // Keep only entries with values greater than 150
    /// cache.retain(|_, value| *value > 150);
    ///
    /// assert_eq!(cache.len(), 2);
    /// assert!(!cache.contains_key("key1"));
    /// assert!(cache.contains_key("key2"));
    /// assert!(cache.contains_key("key3"));
    /// # }
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        // Collect indices to remove
        let mut to_remove = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            if !f(&node.key, &node.value) {
                to_remove.push(i);
            }
        }

        // Remove indices from highest to lowest to avoid invalidating other indices
        to_remove.sort_unstable_by(|a, b| b.cmp(a));

        for idx in to_remove {
            // Remove from map
            self.map.remove(&self.nodes[idx].key);

            // If it's the last element, just pop it
            if idx == self.nodes.len() - 1 {
                self.nodes.pop();
            } else {
                // Replace with the last element
                let last_idx = self.nodes.len() - 1;
                // Use swap_remove which replaces the removed element with the last element
                self.nodes.swap_remove(idx);
                if idx < self.nodes.len() {
                    // Update map for the swapped node
                    self.map.insert(self.nodes[idx].key.clone(), idx);
                }

                // Update hand if needed
                if let Some(hand_idx) = self.hand {
                    if hand_idx == idx {
                        // Hand was pointing to the removed node, move it to previous
                        self.hand = if idx > 0 {
                            Some(idx - 1)
                        } else if !self.nodes.is_empty() {
                            Some(self.nodes.len() - 1)
                        } else {
                            None
                        };
                    } else if hand_idx == last_idx {
                        // Hand was pointing to the last node that was moved
                        self.hand = Some(idx);
                    }
                }
            }
        }
    }

    /// Returns a recommended cache capacity based on current utilization.
    ///
    /// This method analyzes the current cache utilization and recommends a new capacity based on:
    /// - The fill ratio (how much of the capacity is actually being used)
    /// - The number of entries with the 'visited' flag set
    /// - A target utilization range
    ///
    /// The recommendation aims to keep the cache size optimal:
    /// - If the cache is significantly underfilled (fill ratio < 10%), it suggests decreasing capacity
    ///   regardless of other factors to avoid wasting memory
    /// - If many entries are frequently accessed (high utilization), it suggests increasing capacity
    /// - If few entries are accessed frequently (low utilization), it suggests decreasing capacity
    /// - Within a normal utilization range, it keeps the capacity stable
    ///
    /// # Arguments
    ///
    /// * `min_factor` - Minimum scaling factor (e.g., 0.5 means never recommend less than 50% of current capacity)
    /// * `max_factor` - Maximum scaling factor (e.g., 2.0 means never recommend more than 200% of current capacity)
    /// * `low_threshold` - Utilization threshold below which capacity is reduced (e.g., 0.3 means 30% utilization)
    /// * `high_threshold` - Utilization threshold above which capacity is increased (e.g., 0.7 means 70% utilization)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sieve_cache::SieveCache;
    /// #
    /// # fn main() {
    /// # let mut cache = SieveCache::<String, i32>::new(100).unwrap();
    /// #
    /// # // Add items to the cache
    /// # for i in 0..80 {
    /// #     cache.insert(i.to_string(), i);
    /// #     
    /// #     // Accessing some items to mark them as visited
    /// #     if i % 2 == 0 {
    /// #         cache.get(&i.to_string());
    /// #     }
    /// # }
    /// #
    /// // Get a recommended capacity with default parameters
    /// let recommended = cache.recommended_capacity(0.5, 2.0, 0.3, 0.7);
    /// println!("Recommended capacity: {}", recommended);
    /// # }
    /// ```
    ///
    /// # Default Recommendation Parameters
    ///
    /// If you're unsure what parameters to use, these values provide a reasonable starting point:
    /// - `min_factor`: 0.5 (never recommend less than half current capacity)
    /// - `max_factor`: 2.0 (never recommend more than twice current capacity)
    /// - `low_threshold`: 0.3 (reduce capacity if utilization below 30%)
    /// - `high_threshold`: 0.7 (increase capacity if utilization above 70%)
    pub fn recommended_capacity(
        &self,
        min_factor: f64,
        max_factor: f64,
        low_threshold: f64,
        high_threshold: f64,
    ) -> usize {
        // If the cache is empty, return the current capacity
        if self.nodes.is_empty() {
            return self.capacity;
        }

        // Count entries with visited flag set
        let visited_count = self.nodes.iter().filter(|node| node.visited).count();

        // Calculate the utilization ratio (visited entries / total entries)
        let utilization_ratio = visited_count as f64 / self.nodes.len() as f64;

        // Calculate fill ratio (total entries / capacity)
        let fill_ratio = self.nodes.len() as f64 / self.capacity as f64;

        // Low fill ratio threshold (consider the cache underfilled below this)
        let low_fill_threshold = 0.1; // 10% filled

        // Fill ratio takes precedence over utilization:
        // If the cache is severely underfilled, we should decrease capacity
        // regardless of utilization
        if fill_ratio < low_fill_threshold {
            // Calculate how much to decrease based on how empty the cache is
            let fill_below_threshold = if fill_ratio > 0.0 {
                (low_fill_threshold - fill_ratio) / low_fill_threshold
            } else {
                1.0
            };
            // Apply the min_factor as a floor
            let scaling_factor = 1.0 - (1.0 - min_factor) * fill_below_threshold;

            // Apply the scaling factor to current capacity and ensure it's at least 1
            return std::cmp::max(1, (self.capacity as f64 * scaling_factor).round() as usize);
        }

        // For normal fill levels, use the original logic based on utilization
        let scaling_factor = if utilization_ratio >= high_threshold {
            // High utilization - recommend increasing the capacity
            // Scale between 1.0 and max_factor based on utilization above the high threshold
            let utilization_above_threshold =
                (utilization_ratio - high_threshold) / (1.0 - high_threshold);
            1.0 + (max_factor - 1.0) * utilization_above_threshold
        } else if utilization_ratio <= low_threshold {
            // Low utilization - recommend decreasing capacity
            // Scale between min_factor and 1.0 based on how far below the low threshold
            let utilization_below_threshold = (low_threshold - utilization_ratio) / low_threshold;
            1.0 - (1.0 - min_factor) * utilization_below_threshold
        } else {
            // Normal utilization - keep current capacity
            1.0
        };

        // Apply the scaling factor to current capacity and ensure it's at least 1
        std::cmp::max(1, (self.capacity as f64 * scaling_factor).round() as usize)
    }
}

#[test]
fn test() {
    let mut cache = SieveCache::new(3).unwrap();
    assert!(cache.insert("foo".to_string(), "foocontent".to_string()));
    assert!(cache.insert("bar".to_string(), "barcontent".to_string()));
    cache.remove("bar");
    assert!(cache.insert("bar2".to_string(), "bar2content".to_string()));
    assert!(cache.insert("bar3".to_string(), "bar3content".to_string()));
    assert_eq!(cache.get("foo"), Some(&"foocontent".to_string()));
    assert_eq!(cache.get("bar"), None);
    assert_eq!(cache.get("bar2"), Some(&"bar2content".to_string()));
    assert_eq!(cache.get("bar3"), Some(&"bar3content".to_string()));
}

#[test]
fn test_visited_flag_update() {
    let mut cache = SieveCache::new(2).unwrap();
    cache.insert("key1".to_string(), "value1".to_string());
    cache.insert("key2".to_string(), "value2".to_string());
    // update `key1` entry.
    cache.insert("key1".to_string(), "updated".to_string());
    // new entry is added.
    cache.insert("key3".to_string(), "value3".to_string());
    assert_eq!(cache.get("key1"), Some(&"updated".to_string()));
}

#[test]
fn test_clear() {
    let mut cache = SieveCache::new(10).unwrap();
    cache.insert("key1".to_string(), "value1".to_string());
    cache.insert("key2".to_string(), "value2".to_string());

    assert_eq!(cache.len(), 2);
    assert!(!cache.is_empty());

    cache.clear();

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.get("key1"), None);
    assert_eq!(cache.get("key2"), None);
}

#[test]
fn test_iterators() {
    let mut cache = SieveCache::new(10).unwrap();
    cache.insert("key1".to_string(), "value1".to_string());
    cache.insert("key2".to_string(), "value2".to_string());

    // Test keys iterator
    let keys: Vec<&String> = cache.keys().collect();
    assert_eq!(keys.len(), 2);
    assert!(keys.contains(&&"key1".to_string()));
    assert!(keys.contains(&&"key2".to_string()));

    // Test values iterator
    let values: Vec<&String> = cache.values().collect();
    assert_eq!(values.len(), 2);
    assert!(values.contains(&&"value1".to_string()));
    assert!(values.contains(&&"value2".to_string()));

    // Test values_mut iterator
    for value in cache.values_mut() {
        *value = format!("{}_updated", value);
    }

    assert_eq!(cache.get("key1"), Some(&"value1_updated".to_string()));
    assert_eq!(cache.get("key2"), Some(&"value2_updated".to_string()));

    // Test key-value iterator
    let entries: Vec<(&String, &String)> = cache.iter().collect();
    assert_eq!(entries.len(), 2);

    // Test key-value mutable iterator
    for (key, value) in cache.iter_mut() {
        if key == "key1" {
            *value = format!("{}_special", value);
        }
    }

    assert_eq!(
        cache.get("key1"),
        Some(&"value1_updated_special".to_string())
    );
    assert_eq!(cache.get("key2"), Some(&"value2_updated".to_string()));
}

#[test]
fn test_retain() {
    let mut cache = SieveCache::new(10).unwrap();

    // Add some entries
    cache.insert("even1".to_string(), 2);
    cache.insert("even2".to_string(), 4);
    cache.insert("odd1".to_string(), 1);
    cache.insert("odd2".to_string(), 3);

    assert_eq!(cache.len(), 4);

    // Keep only entries with even values
    cache.retain(|_, v| v % 2 == 0);

    assert_eq!(cache.len(), 2);
    assert!(cache.contains_key("even1"));
    assert!(cache.contains_key("even2"));
    assert!(!cache.contains_key("odd1"));
    assert!(!cache.contains_key("odd2"));

    // Keep only entries with keys containing '1'
    cache.retain(|k, _| k.contains('1'));

    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key("even1"));
    assert!(!cache.contains_key("even2"));
}

#[test]
fn test_recommended_capacity() {
    // Test case 1: Empty cache - should return current capacity
    let cache = SieveCache::<String, u32>::new(100).unwrap();
    assert_eq!(cache.recommended_capacity(0.5, 2.0, 0.3, 0.7), 100);

    // Test case 2: Low utilization (few visited nodes)
    let mut cache = SieveCache::new(100).unwrap();
    // Fill the cache first without marking anything as visited
    for i in 0..90 {
        cache.insert(i.to_string(), i);
    }

    // Now mark only a tiny fraction as visited
    for i in 0..5 {
        cache.get(&i.to_string()); // Only 5% visited
    }

    // With very low utilization and high fill, should recommend decreasing capacity
    let recommended = cache.recommended_capacity(0.5, 2.0, 0.1, 0.7); // Lower threshold to ensure test passes
    assert!(recommended < 100);
    assert!(recommended >= 50); // Should not go below min_factor

    // Test case 3: High utilization (many visited nodes)
    let mut cache = SieveCache::new(100).unwrap();
    for i in 0..90 {
        cache.insert(i.to_string(), i);
        // Mark 80% as visited
        if i % 10 != 0 {
            cache.get(&i.to_string());
        }
    }
    // With 80% utilization, should recommend increasing capacity
    let recommended = cache.recommended_capacity(0.5, 2.0, 0.3, 0.7);
    assert!(recommended > 100);
    assert!(recommended <= 200); // Should not go above max_factor

    // Test case 4: Normal utilization (should keep capacity the same)
    let mut cache = SieveCache::new(100).unwrap();
    for i in 0..90 {
        cache.insert(i.to_string(), i);
        // Mark 50% as visited
        if i % 2 == 0 {
            cache.get(&i.to_string());
        }
    }
    // With 50% utilization (between thresholds), capacity should be fairly stable
    let recommended = cache.recommended_capacity(0.5, 2.0, 0.3, 0.7);
    assert!(
        recommended >= 95,
        "With normal utilization, capacity should be close to original"
    );
    assert!(
        recommended <= 100,
        "With normal utilization, capacity should not exceed original"
    );

    // Test case 5: Low fill ratio (few entries relative to capacity)
    let mut cache = SieveCache::new(2000).unwrap();
    // Add only a few entries (5% of capacity)
    for i in 0..100 {
        cache.insert(i.to_string(), i);
        // Mark all as visited to simulate high hit rate
        cache.get(&i.to_string());
    }

    // Even though utilization is high (100% visited), the fill ratio is very low (5%)
    // so it should still recommend decreasing capacity
    let recommended = cache.recommended_capacity(0.5, 2.0, 0.3, 0.7);
    assert!(
        recommended < 2000,
        "With low fill ratio, capacity should be decreased despite high hit rate"
    );
    assert!(
        recommended >= 1000, // min_factor = 0.5
        "Capacity should not go below min_factor of current capacity"
    );
}

#[test]
fn insert_never_exceeds_capacity_when_all_visited() {
    let mut c = SieveCache::new(2).unwrap();
    c.insert("a".to_string(), 1);
    c.insert("b".to_string(), 2);
    // Mark all visited
    assert!(c.get("a").is_some());
    assert!(c.get("b").is_some());
    // This would exceed capacity
    c.insert("c".to_string(), 3);
    // This is our an invariant
    assert!(c.len() <= c.capacity());
}

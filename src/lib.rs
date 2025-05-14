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
    //! 2. A doubly-linked list for managing the eviction order
    //! 3. A "visited" flag on each entry to track recent access
    //!
    //! When the cache is full and a new item is inserted, the eviction algorithm:
    //! 1. Starts from the "hand" position (or tail if no hand)
    //! 2. Finds the first non-visited entry, evicting it
    //! 3. Marks all visited entries as non-visited while searching
}

pub mod _implementation_choice {
    //! ## Choosing the Right Implementation
    //!
    //! - Use [`SieveCache`] for single-threaded applications
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
use std::hash::Hash;
use std::{collections::HashMap, ptr::NonNull};

#[cfg(feature = "sharded")]
mod sharded;
#[cfg(feature = "sync")]
mod sync;

#[cfg(feature = "sharded")]
pub use sharded::ShardedSieveCache;
#[cfg(feature = "sync")]
pub use sync::SyncSieveCache;

struct Node<K: Eq + Hash + Clone, V> {
    key: K,
    value: V,
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    visited: bool,
}

impl<K: Eq + Hash + Clone, V> Node<K, V> {
    fn new(key: K, value: V) -> Self {
        Self {
            key,
            value,
            prev: None,
            next: None,
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
    /// Map of keys to cache entries
    map: HashMap<K, Box<Node<K, V>>>,
    /// Pointer to the head (most recently added) node
    head: Option<NonNull<Node<K, V>>>,
    /// Pointer to the tail (least recently added) node
    tail: Option<NonNull<Node<K, V>>>,
    /// The "hand" pointer used by the SIEVE algorithm for eviction
    hand: Option<NonNull<Node<K, V>>>,
    /// Maximum number of entries the cache can hold
    capacity: usize,
    /// Current number of entries in the cache
    len: usize,
}

unsafe impl<K: Eq + Hash + Clone, V> Send for SieveCache<K, V> {}

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
            head: None,
            tail: None,
            hand: None,
            capacity,
            len: 0,
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

    /// Returns the number of entries currently in the cache.
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
    /// cache.insert("key".to_string(), 42);
    /// assert_eq!(cache.len(), 1);
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
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
    /// cache.insert("key".to_string(), 42);
    /// assert!(!cache.is_empty());
    ///
    /// cache.remove("key");
    /// assert!(cache.is_empty());
    /// # }
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    pub fn contains_key<Q>(&mut self, key: &Q) -> bool
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
        let node_ = self.map.get_mut(key)?;
        // Mark as visited for the SIEVE algorithm
        node_.visited = true;
        Some(&node_.value)
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
        let node_ = self.map.get_mut(key)?;
        // Mark as visited for the SIEVE algorithm
        node_.visited = true;
        Some(&mut node_.value)
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
        let node = self.map.get_mut(&key);
        if let Some(node_) = node {
            // Update existing entry
            node_.visited = true;
            node_.value = value;
            return false;
        }

        // Evict if at capacity
        if self.len >= self.capacity {
            self.evict();
        }

        // Create new node
        let node = Box::new(Node::new(key.clone(), value));
        self.add_node(NonNull::from(node.as_ref()));
        debug_assert!(!node.visited);
        self.map.insert(key, node);
        debug_assert!(self.len < self.capacity);
        self.len += 1;
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
        // Find the node
        let node_ = self.map.get_mut(key)?;
        let node__ = NonNull::from(node_.as_ref());

        // Update hand pointer if it points to the node being removed
        if self.hand == Some(node__) {
            self.hand = node_.as_ref().prev;
        }

        // Remove from the map and extract the value
        let value = self.map.remove(key).map(|node| node.value);

        // Remove from the linked list
        self.remove_node(node__);
        debug_assert!(self.len > 0);
        self.len -= 1;
        value
    }

    /// Removes and returns a value from the cache that was not recently accessed.
    ///
    /// This method implements the SIEVE eviction algorithm:
    /// 1. Starting from the "hand" pointer (or tail if no hand), look for an entry
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
        // Start from the hand pointer or the tail if hand is None
        let mut node = self.hand.or(self.tail);

        // Scan for a non-visited entry
        while node.is_some() {
            let mut node_ = node.unwrap();
            unsafe {
                // If we found a non-visited entry, evict it
                if !node_.as_ref().visited {
                    break;
                }

                // Mark as non-visited for the next scan
                node_.as_mut().visited = false;

                // Move to the previous node, or wrap around to the tail
                if node_.as_ref().prev.is_some() {
                    node = node_.as_ref().prev;
                } else {
                    node = self.tail;
                }
            }
        }

        // If we found a node to evict
        if let Some(node_) = node {
            let value = unsafe {
                // Update the hand pointer
                self.hand = node_.as_ref().prev;

                // Remove from the map and get the value
                self.map.remove(&node_.as_ref().key).map(|node| node.value)
            };

            // Remove from the linked list
            self.remove_node(node_);
            debug_assert!(self.len > 0);
            self.len -= 1;
            value
        } else {
            None
        }
    }

    /// Adds a node to the front of the linked list (making it the new head).
    ///
    /// This is an internal helper method used during insertion.
    fn add_node(&mut self, mut node: NonNull<Node<K, V>>) {
        unsafe {
            // Link the new node to the current head
            node.as_mut().next = self.head;
            node.as_mut().prev = None;

            // Update the current head's prev pointer to the new node
            if let Some(mut head) = self.head {
                head.as_mut().prev = Some(node);
            }
        }

        // Set the new node as the head
        self.head = Some(node);

        // If this is the first node, it's also the tail
        if self.tail.is_none() {
            self.tail = self.head;
        }
    }

    /// Removes a node from the linked list.
    ///
    /// This is an internal helper method used during removal and eviction.
    fn remove_node(&mut self, node: NonNull<Node<K, V>>) {
        unsafe {
            // Update the previous node's next pointer
            if let Some(mut prev) = node.as_ref().prev {
                prev.as_mut().next = node.as_ref().next;
            } else {
                // If no previous node, this was the head
                self.head = node.as_ref().next;
            }

            // Update the next node's prev pointer
            if let Some(mut next) = node.as_ref().next {
                next.as_mut().prev = node.as_ref().prev;
            } else {
                // If no next node, this was the tail
                self.tail = node.as_ref().prev;
            }
        }
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

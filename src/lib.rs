#![doc = include_str!("../README.md")]

use std::borrow::Borrow;
use std::hash::Hash;
use std::{collections::HashMap, ptr::NonNull};

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
pub struct SieveCache<K: Eq + Hash + Clone, V> {
    map: HashMap<K, Box<Node<K, V>>>,
    head: Option<NonNull<Node<K, V>>>,
    tail: Option<NonNull<Node<K, V>>>,
    hand: Option<NonNull<Node<K, V>>>,
    capacity: usize,
    len: usize,
}

unsafe impl<K: Eq + Hash + Clone, V> Send for SieveCache<K, V> {}

impl<K: Eq + Hash + Clone, V> SieveCache<K, V> {
    /// Create a new cache with the given capacity.
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

    /// Return the capacity of the cache.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the number of cached values.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` when no values are currently cached.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return `true` if there is a value in the cache mapped to by `key`.
    #[inline]
    pub fn contains_key<Q: ?Sized>(&mut self, key: &Q) -> bool
    where
        Q: Hash + Eq,
        K: Borrow<Q>,
    {
        self.map.contains_key(key)
    }

    /// Get an immutable reference to the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    pub fn get<Q: ?Sized>(&mut self, key: &Q) -> Option<&V>
    where
        Q: Hash + Eq,
        K: Borrow<Q>,
    {
        let node_ = self.map.get_mut(key)?;
        node_.visited = true;
        Some(&node_.value)
    }

    /// Get a mutable reference to the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: Hash + Eq,
        K: Borrow<Q>,
    {
        let node_ = self.map.get_mut(key)?;
        node_.visited = true;
        Some(&mut node_.value)
    }

    /// Map `key` to `value` in the cache, possibly evicting old entries.
    ///
    /// This method returns `true` when this is a new entry, and `false` if an existing entry was
    /// updated.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        let node = self.map.get_mut(&key);
        if let Some(node_) = node {
            node_.value = value;
            return false;
        }
        if self.len >= self.capacity {
            self.evict();
        }
        let node = Box::new(Node::new(key.clone(), value));
        self.add_node(NonNull::from(node.as_ref()));
        self.map.insert(key, node);
        debug_assert!(self.len < self.capacity);
        self.len += 1;
        true
    }

    /// Remove the cache entry mapped to by `key`.
    ///
    /// This method returns the value removed from the cache. If `key` did not map to any value,
    /// then this returns `None`.
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        let node_ = self.map.get_mut(key)?;
        let node__ = NonNull::from(node_.as_ref());
        let value = self.map.remove(key).map(|node| node.value);
        self.remove_node(node__);
        debug_assert!(self.len > 0);
        self.len -= 1;
        value
    }

    fn add_node(&mut self, mut node: NonNull<Node<K, V>>) {
        unsafe {
            node.as_mut().next = self.head;
            node.as_mut().prev = None;
            if let Some(mut head) = self.head {
                head.as_mut().prev = Some(node);
            }
        }
        self.head = Some(node);
        if self.tail.is_none() {
            self.tail = self.head;
        }
    }

    fn remove_node(&mut self, node: NonNull<Node<K, V>>) {
        unsafe {
            if let Some(mut prev) = node.as_ref().prev {
                prev.as_mut().next = node.as_ref().next;
            } else {
                self.head = node.as_ref().next;
            }
            if let Some(mut next) = node.as_ref().next {
                next.as_mut().prev = node.as_ref().prev;
            } else {
                self.tail = node.as_ref().prev;
            }
        }
    }

    fn evict(&mut self) {
        let mut node = self.hand.or(self.tail);
        while node.is_some() {
            let mut node_ = node.unwrap();
            unsafe {
                if !node_.as_ref().visited {
                    break;
                }
                node_.as_mut().visited = false;
                if node_.as_ref().prev.is_some() {
                    node = node_.as_ref().prev;
                } else {
                    node = self.tail;
                }
            }
        }
        if let Some(node_) = node {
            unsafe {
                self.hand = node_.as_ref().prev;
                self.map.remove(&node_.as_ref().key);
            }
            self.remove_node(node_);
            debug_assert!(self.len > 0);
            self.len -= 1;
        }
    }
}

#[test]
fn test() {
    let mut cache = SieveCache::new(3).unwrap();
    assert_eq!(
        cache.insert("foo".to_string(), "foocontent".to_string()),
        true
    );
    assert_eq!(
        cache.insert("bar".to_string(), "barcontent".to_string()),
        true
    );
    cache.remove("bar");
    assert_eq!(
        cache.insert("bar2".to_string(), "bar2content".to_string()),
        true
    );
    assert_eq!(
        cache.insert("bar3".to_string(), "bar3content".to_string()),
        true
    );
    assert_eq!(cache.get("foo"), Some(&"foocontent".to_string()));
    assert_eq!(cache.get("bar"), None);
    assert_eq!(cache.get("bar2"), Some(&"bar2content".to_string()));
    assert_eq!(cache.get("bar3"), Some(&"bar3content".to_string()));
}

use std::borrow::Borrow;
use std::hash::Hash;
use std::{collections::HashMap, mem::MaybeUninit, ptr::NonNull};

struct Node<K: PartialEq + Eq + Hash + Clone, V> {
    key: MaybeUninit<K>,
    value: V,
    visited: bool,
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
}

impl<K: PartialEq + Eq + Hash + Clone, V> Node<K, V> {
    fn new(key: K, value: V) -> Self {
        Self {
            key: MaybeUninit::new(key),
            value,
            visited: false,
            prev: None,
            next: None,
        }
    }
}

pub struct SieveCache<K: PartialEq + Eq + Hash + Clone, V> {
    capacity: usize,
    len: usize,
    head: Option<NonNull<Node<K, V>>>,
    tail: Option<NonNull<Node<K, V>>>,
    hand: Option<NonNull<Node<K, V>>>,
    map: HashMap<K, Box<Node<K, V>>>,
}

impl<K: PartialEq + Eq + Hash + Clone, V> SieveCache<K, V> {
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("capacity must be greater than 0");
        }
        Ok(Self {
            capacity,
            len: 0,
            head: None,
            tail: None,
            hand: None,
            map: HashMap::with_capacity(capacity),
        })
    }

    /// Returns the capacity of the cache.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of cached values.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no values are currently cached.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn add_node(&mut self, mut node: NonNull<Node<K, V>>) {
        unsafe {
            node.as_mut().next = self.head;
            node.as_mut().prev = None;
            if let Some(mut head) = self.head {
                head.as_mut().prev = Some(node);
            }
            self.head = Some(node);
            if self.tail.is_none() {
                self.tail = self.head;
            }
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
        unsafe {
            let mut node = match (self.hand) {
                Some(hand) => Some(hand),
                None => self.tail,
            };
            loop {
                if node == None {
                    break;
                }
                let mut node_ = node.unwrap();
                if node_.as_ref().visited == false {
                    break;
                }
                node_.as_mut().visited = false;
                if node_.as_ref().prev != None {
                    node = node_.as_ref().prev;
                } else {
                    node = self.tail;
                }
            }
            if let Some(node_) = node {
                if node_.as_ref().prev != None {
                    self.hand = node_.as_ref().prev;
                } else {
                    self.hand = None;
                }
                self.map.remove(node_.as_ref().key.assume_init_ref());
                self.remove_node(node_);
                self.len -= 1;
            }
        }
    }

    /// Get an immutable reference to the value in the cache mapped to by `key`.
    ///
    /// If no value exists for `key`, this returns `None`.
    pub fn get<Q: ?Sized>(&mut self, key: &Q) -> Option<&V>
    where
        Q: Hash + Eq,
        K: Borrow<Q>,
    {
        let node = self.map.get_mut(key);
        if node.is_none() {
            return None;
        }
        let node_ = node.unwrap();
        node_.visited = true;
        Some(node_.value.borrow())
    }

    /// Map `key` to `value` in the cache, possibly evicting old entries.
    ///
    /// This method returns `true` when this is a new entry, and `false` if an existing entry was
    /// updated.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        let node = self.map.get_mut(&key);
        if let Some(node_) = node {
            node_.value = value;
            return true;
        }
        if self.len >= self.capacity {
            self.evict();
        }
        let node = Box::new(Node::new(key.clone(), value));
        self.add_node(NonNull::from(node.as_ref()));
        self.map.insert(key, node);
        self.len += 1;
        false
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
        let node = self.map.get_mut(key);
        if node.is_none() {
            return None;
        }
        let node_ = node.unwrap();
        let node__ = NonNull::from(node_.as_ref());
        let value = self.map.remove(key).map(|node| node.value);
        self.remove_node(node__);
        self.len -= 1;
        value
    }
}

fn main() {
    let mut cache: SieveCache<String, String> = SieveCache::new(3).unwrap();
    cache.insert("foo".to_string(), "foocontent".to_string());
    cache.insert("bar".to_string(), "barcontent".to_string());
    cache.remove("bar");
    cache.insert("bar2".to_string(), "bar2content".to_string());
    cache.insert("bar3".to_string(), "bar3content".to_string());
    println!("{:?}", cache.get("foo"));
    println!("{:?}", cache.get("bar"));
    println!("{:?}", cache.get("bar2"));
    println!("{:?}", cache.get("bar3"));
}

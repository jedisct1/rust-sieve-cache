mod custom_hasher_test {
    use std::{cell::RefCell, hash::{BuildHasher, Hasher}, rc::Rc};

    use sieve_cache::{SieveCache, SyncSieveCache};

    #[derive(Clone)]
    struct CustomHasher {
        hash: u64,
        invocations: Rc<RefCell<usize>>,
    }

    impl CustomHasher {
        fn new() -> Self {
            CustomHasher { hash: 0, invocations: Rc::new(RefCell::new(0)) }
        }
    }

    impl Hasher for CustomHasher {
        fn finish(&self) -> u64 {
            self.hash
        }

        fn write(&mut self, bytes: &[u8]) {
            //convert the byte array to a u64. Just for testing.
            //for u8, this is a single byte. 
            //The tests below should use a u8 key, as introducing keys longer than a single byte 
            //would cause issues with endianness and could lead to evictions.
            for b in bytes {
                self.hash = self.hash << 8 | (*b as u64);
            }

            *self.invocations.borrow_mut() += 1;
        }
    }

    impl BuildHasher for CustomHasher {
        type Hasher = Self;

        fn build_hasher(&self) -> Self::Hasher {
            CustomHasher { hash: 0, invocations: Rc::clone(&self.invocations) }
        }
    }

    #[test]
    fn test_custom_hasher() {
        let hasher = CustomHasher::new();
        let mut cache = SieveCache::<u8, u64, CustomHasher>::new_with_hasher(10, hasher.clone()).unwrap();
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.get(&1), Some(&100u64));
        assert_eq!(cache.get(&2), Some(&200u64));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), None);

        cache.remove(&1);
        cache.remove(&2);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.len(), 0);

        //at least one invocation per get and remove, plus some for the insertions. 
        //The exact number depends on the implementation details of the sharding and eviction logic,
        //as well as the hashing logic used in the standard library, which may call the hasher multiple times for certain operations.
        assert!(*hasher.invocations.borrow() >= 9);
    }

    #[cfg(feature = "sync")]
    #[test]
    fn test_custom_hasher_sync() {
        let hasher = CustomHasher::new();
        let cache = SyncSieveCache::<u8, u64, CustomHasher>::new_with_hasher(10, hasher.clone()).unwrap();
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.get(&1), Some(100u64));
        assert_eq!(cache.get(&2), Some(200u64));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), None);

        cache.remove(&1);
        cache.remove(&2);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.len(), 0);

        //at least one invocation per get and remove, plus some for the insertions. 
        //The exact number depends on the implementation details of the sharding and eviction logic,
        //as well as the hashing logic used in the standard library, which may call the hasher multiple times for certain operations.
        assert!(*hasher.invocations.borrow() >= 9);
    }

    #[cfg(feature = "sharded")]
    #[test]
    fn test_custom_hasher_sharded() {
        let hasher = CustomHasher::new();
        let cache = sieve_cache::ShardedSieveCache::<u8, u64, CustomHasher>::new_with_hasher(100, hasher.clone()).unwrap();
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.get(&1), Some(100u64));
        assert_eq!(cache.get(&2), Some(200u64));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), None);

        cache.remove(&1);
        cache.remove(&2);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.len(), 0);

        //at least one invocation per get and remove, plus some for the insertions. 
        //The exact number depends on the implementation details of the sharding and eviction logic,
        //as well as the hashing logic used in the standard library, which may call the hasher multiple times for certain operations.
        assert!(*hasher.invocations.borrow() >= 9);
    }

    #[cfg(feature = "weighted")]
    #[test]
    fn test_custom_hasher_weighted() {
        let hasher = CustomHasher::new();
        let mut cache = sieve_cache::WeightedSieveCache::<u8, u64, CustomHasher>::new_with_hasher(10, 100, hasher.clone()).unwrap();
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.get(&1), Some(&100u64));
        assert_eq!(cache.get(&2), Some(&200u64));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), None);

        cache.remove(&1);
        cache.remove(&2);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.len(), 0);

        //at least one invocation per get and remove, plus some for the insertions. 
        //The exact number depends on the implementation details of the sharding and eviction logic,
        //as well as the hashing logic used in the standard library, which may call the hasher multiple times for certain operations.
        assert!(*hasher.invocations.borrow() >= 9);
    }

    #[cfg(all(feature = "weighted", feature = "sync"))]
    #[test]
    fn test_custom_hasher_weighted_sync() {
        let hasher = CustomHasher::new();
        let cache = sieve_cache::WeightedSyncSieveCache::<u8, u64, CustomHasher>::new_with_hasher(10, 100, hasher.clone()).unwrap();
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.get(&1), Some(100u64));
        assert_eq!(cache.get(&2), Some(200u64));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), None);

        cache.remove(&1);
        cache.remove(&2);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.len(), 0);

        //at least one invocation per get and remove, plus some for the insertions. 
        //The exact number depends on the implementation details of the sharding and eviction logic,
        //as well as the hashing logic used in the standard library, which may call the hasher multiple times for certain operations.
        assert!(*hasher.invocations.borrow() >= 9);
    }

    #[cfg(all(feature = "weighted", feature = "sharded"))]
    #[test]
    fn test_custom_hasher_weighted_sharded() {
        let hasher = CustomHasher::new();
        let cache = sieve_cache::WeightedShardedSieveCache::<u8, u64, CustomHasher>::new_with_hasher(10, 100, hasher.clone()).unwrap();
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.get(&1), Some(100u64));
        assert_eq!(cache.get(&2), Some(200u64));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3), None);

        cache.remove(&1);
        cache.remove(&2);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.len(), 0);

        //at least one invocation per get and remove, plus some for the insertions. 
        //The exact number depends on the implementation details of the sharding and eviction logic,
        //as well as the hashing logic used in the standard library, which may call the hasher multiple times for certain operations.
        assert!(*hasher.invocations.borrow() >= 9);
    }
}

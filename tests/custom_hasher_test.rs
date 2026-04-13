mod custom_hasher_test {
    use std::hash::{BuildHasher, Hasher};

    use sieve_cache::{SieveCache, SyncSieveCache};


    struct CustomHasher {
        hash: u64,
    }

    impl Hasher for CustomHasher {
        fn finish(&self) -> u64 {
            // Implement a custom hash function here
            0
        }

        fn write(&mut self, bytes: &[u8]) {
            for b in bytes {
                self.hash = self.hash << 8 | (*b as u64);
            }
        }
    }

    impl BuildHasher for CustomHasher {
        type Hasher = Self;

        fn build_hasher(&self) -> Self::Hasher {
            CustomHasher { hash: 0}
        }
    }

    impl Clone for CustomHasher {
        fn clone(&self) -> Self {
            CustomHasher { hash: self.hash }
        }
    }

    #[test]
    fn test_custom_hasher() {
        let mut cache = SieveCache::<u64, u64, CustomHasher>::new_with_hasher(10, CustomHasher { hash: 0 }).unwrap();
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
    }

    #[test]
    fn test_custom_hasher_sync() {
        let cache = SyncSieveCache::<u64, u64, CustomHasher>::new_with_hasher(10, CustomHasher { hash: 0 }).unwrap();
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
    }

    #[test]
    fn test_custom_hasher_sharded() {
        let cache = sieve_cache::ShardedSieveCache::<u64, u64, CustomHasher>::new_with_hasher(100, CustomHasher { hash: 0 }).unwrap();
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
    }

    #[test]
    fn test_custom_hasher_weighted() {
        let mut cache = sieve_cache::WeightedSieveCache::<u64, u64, CustomHasher>::new_with_hasher(10, 100, CustomHasher { hash: 0 }).unwrap();
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
    }

    #[test]
    fn test_custom_hasher_weighted_sync() {
        let cache = sieve_cache::WeightedSyncSieveCache::<u64, u64, CustomHasher>::new_with_hasher(10, 100, CustomHasher { hash: 0 }).unwrap();
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
    }

    #[test]
    fn test_custom_hasher_weighted_sharded() {
        let cache = sieve_cache::WeightedShardedSieveCache::<u64, u64, CustomHasher>::new_with_hasher(10, 100, CustomHasher { hash: 0 }).unwrap();
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
    }
}

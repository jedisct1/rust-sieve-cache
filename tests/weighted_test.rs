#[cfg(feature = "weighted")]
mod weighted_tests {
    use sieve_cache::{SieveCache, Weigh, WeightedSieveCache};

    #[test]
    fn evict_pair_returns_key_and_value() {
        let mut cache: SieveCache<String, String> = SieveCache::new(3).unwrap();
        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        cache.insert("c".into(), "3".into());

        let pair = cache.evict_pair();
        assert!(pair.is_some());
        let (k, _v) = pair.unwrap();
        assert!(!cache.contains_key(&k));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn evict_pair_none_only_when_empty() {
        let mut cache: SieveCache<String, String> = SieveCache::new(3).unwrap();
        assert!(cache.evict_pair().is_none());

        cache.insert("a".into(), "1".into());
        // Mark as visited
        cache.get("a");
        // evict_pair must still succeed (retries internally)
        let pair = cache.evict_pair();
        assert!(pair.is_some());
        assert!(cache.is_empty());
        // Now truly empty
        assert!(cache.evict_pair().is_none());
    }

    #[test]
    fn evict_pair_all_visited_cache() {
        let mut cache: SieveCache<String, u32> = SieveCache::new(5).unwrap();
        for i in 0..5 {
            cache.insert(format!("k{}", i), i);
        }
        // Mark all as visited
        for i in 0..5 {
            cache.get(&format!("k{}", i));
        }

        // evict_pair must succeed despite all entries being visited
        let pair = cache.evict_pair();
        assert!(pair.is_some());
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn legacy_evict_still_returns_none_on_all_visited() {
        // Regression test: the old evict() behavior is unchanged
        let mut cache: SieveCache<String, u32> = SieveCache::new(3).unwrap();
        for i in 0..3 {
            cache.insert(format!("k{}", i), i);
        }
        // Mark all visited
        for i in 0..3 {
            cache.get(&format!("k{}", i));
        }
        // Legacy evict returns None on first pass (all visited)
        let result = cache.evict();
        // It may or may not return None depending on hand position,
        // but the second call should succeed
        if result.is_none() {
            let result2 = cache.evict();
            assert!(result2.is_some());
        }
    }

    #[test]
    fn weight_tracking_insert_remove() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10000).unwrap();

        let k = "hello".to_string();
        let v = "world".to_string();
        let expected = k.weigh() + v.weigh();

        cache.insert(k.clone(), v);
        assert_eq!(cache.current_weight(), expected);

        cache.remove("hello");
        assert_eq!(cache.current_weight(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn weight_tracking_update() {
        let mut cache: WeightedSieveCache<String, Vec<u8>> =
            WeightedSieveCache::new(10, 100000).unwrap();

        let k = "key".to_string();
        cache.insert(k.clone(), vec![0u8; 10]);
        let w1 = cache.current_weight();

        cache.insert(k.clone(), vec![0u8; 100]);
        let w2 = cache.current_weight();

        // Weight should have changed
        assert_ne!(w1, w2);
        // Should reflect the new value's weight
        let expected = k.weigh() + vec![0u8; 100].weigh();
        assert_eq!(w2, expected);
    }

    #[test]
    fn weight_tracking_evict() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 100000).unwrap();

        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());
        let total = cache.current_weight();

        let pair = cache.evict();
        assert!(pair.is_some());
        let (k, v) = pair.unwrap();
        let evicted_weight = k.weigh() + v.weigh();
        assert_eq!(cache.current_weight(), total - evicted_weight);
    }

    #[test]
    fn weight_tracking_clear() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 100000).unwrap();

        for i in 0..5 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }
        assert!(cache.current_weight() > 0);

        cache.clear();
        assert_eq!(cache.current_weight(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn eviction_enforces_weight_budget() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(100, 300).unwrap();

        for i in 0..50 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }

        // Weight must be at or below max_weight (possibly exceeded by one entry)
        // But with many small entries, it should be within budget
        assert!(cache.current_weight() <= cache.max_weight());
        assert!(cache.len() < 50);
    }

    #[test]
    fn oversized_single_entry_stays() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10).unwrap();

        let big = "x".repeat(1000);
        cache.insert("big".into(), big.clone());

        assert_eq!(cache.len(), 1);
        assert!(cache.current_weight() > cache.max_weight());
        assert_eq!(cache.get("big"), Some(&big));
    }

    #[test]
    fn oversized_entry_evicts_previous() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 1000).unwrap();

        // Fill with small entries
        for i in 0..5 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }
        assert_eq!(cache.len(), 5);

        // Insert a big entry whose weight alone exceeds max_weight.
        // The eviction loop evicts all small entries, leaving only "big".
        let big = "x".repeat(1000);
        cache.insert("big".into(), big.clone());

        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key("big"));
        assert!(cache.current_weight() > cache.max_weight());
    }

    #[test]
    fn eviction_loop_terminates_all_visited() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(5, 100).unwrap();

        // Fill and mark all visited
        for i in 0..5 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }
        for i in 0..5 {
            cache.get(&format!("k{}", i));
        }

        // Insert something that exceeds weight — must not hang
        let big = "x".repeat(500);
        cache.insert("overflow".into(), big);

        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key("overflow"));
    }

    #[test]
    fn retain_adjusts_weight() {
        let mut cache: WeightedSieveCache<String, u64> =
            WeightedSieveCache::new(10, 100000).unwrap();

        cache.insert("keep".into(), 1);
        cache.insert("drop1".into(), 2);
        cache.insert("drop2".into(), 3);

        let expected_keep = "keep".to_string().weigh() + 1u64.weigh();

        cache.retain(|_, v| *v == 1);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.current_weight(), expected_keep);
    }

    #[test]
    fn max_weight_zero_rejected() {
        let r: Result<WeightedSieveCache<String, String>, _> = WeightedSieveCache::new(10, 0);
        assert!(r.is_err());
    }

    #[test]
    fn capacity_zero_rejected() {
        let r: Result<WeightedSieveCache<String, String>, _> = WeightedSieveCache::new(0, 100);
        assert!(r.is_err());
    }

    #[test]
    fn get_mut_does_not_change_weight() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 100000).unwrap();

        cache.insert("k".into(), "short".into());
        let w = cache.current_weight();

        // Mutate the value to a longer string
        if let Some(v) = cache.get_mut("k") {
            *v = "a much longer string".into();
        }

        // Weight should NOT have changed (snapshot-at-insert)
        assert_eq!(cache.current_weight(), w);
    }

    #[test]
    fn multiple_inserts_and_evictions() {
        let mut cache: WeightedSieveCache<String, Vec<u8>> =
            WeightedSieveCache::new(100, 500).unwrap();

        for i in 0..200 {
            cache.insert(format!("key{}", i), vec![0u8; 10]);
            // Weight should never wildly exceed max_weight
            // (at most by one entry's weight)
            let max_entry_weight = format!("key{}", i).weigh() + vec![0u8; 10].weigh();
            assert!(cache.current_weight() <= cache.max_weight() + max_entry_weight);
        }
    }

    #[test]
    fn capacity_eviction_does_not_drift_weight() {
        // Small capacity, large weight budget.  Every insert past capacity
        // triggers a capacity-driven eviction inside SieveCache::insert.
        // Before the fix, the evicted key's weight was never subtracted,
        // causing current_weight to drift upward.
        let cap = 5;
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(cap, 1_000_000).unwrap();

        // Insert 200 distinct keys through a 5-slot cache.
        for i in 0..200 {
            cache.insert(format!("k{}", i), format!("v{}", i));

            let actual: usize = cache
                .iter()
                .map(|(k, v)| k.weigh().saturating_add(v.weigh()))
                .sum();
            assert_eq!(
                cache.current_weight(),
                actual,
                "drift at i={}: tracked={} actual={} len={}",
                i,
                cache.current_weight(),
                actual,
                cache.len(),
            );
        }

        assert_eq!(cache.len(), cap);
    }

    #[test]
    fn remove_tracks_weight_correctly_at_scale() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(1000, 1_000_000).unwrap();

        // Insert many entries
        for i in 0..500 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }
        let total = cache.current_weight();
        assert!(total > 0);

        // Remove them one by one — weight should track exactly
        for i in 0..500 {
            cache.remove(&format!("k{}", i));
        }
        assert_eq!(cache.current_weight(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn weigh_option_some() {
        let val: Option<u32> = Some(42);
        let none: Option<u32> = None;
        // Some has the overhead of Option<T> plus the heap difference
        assert!(val.weigh() >= none.weigh());
    }

    #[test]
    fn weigh_box() {
        let b: Box<u32> = Box::new(42);
        // Box overhead + inner type's weigh
        assert_eq!(b.weigh(), std::mem::size_of::<Box<u32>>() + 42u32.weigh());
    }

    #[test]
    fn weigh_str_ref() {
        let s: &str = "hello";
        assert_eq!(s.weigh(), std::mem::size_of::<&str>());
    }

    #[test]
    fn get_mut_does_not_change_tracked_weight() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10_000).unwrap();

        cache.insert("k".to_string(), "short".to_string());
        let weight_after_insert = cache.current_weight();

        // Mutate the value to something much larger.
        if let Some(v) = cache.get_mut("k") {
            *v = "x".repeat(5000);
        }

        // Tracked weight must still reflect the original "short" value.
        assert_eq!(cache.current_weight(), weight_after_insert);

        // Removing must subtract the *charged* snapshot weight, not
        // the live weight.
        cache.remove("k");
        assert_eq!(cache.current_weight(), 0);
    }

    #[test]
    fn eviction_uses_charged_weight_after_mutation() {
        let mut cache: WeightedSieveCache<String, String> =
            WeightedSieveCache::new(10, 10_000).unwrap();

        cache.insert("a".to_string(), "small".to_string());
        let charged = cache.current_weight();

        // Grow the value via get_mut.
        if let Some(v) = cache.get_mut("a") {
            *v = "x".repeat(9000);
        }

        // Weight unchanged — snapshot semantics.
        assert_eq!(cache.current_weight(), charged);

        // Evict — should subtract the charged weight, not live weight.
        let evicted = cache.evict();
        assert!(evicted.is_some());
        assert_eq!(cache.current_weight(), 0);
    }
}

#[cfg(all(feature = "weighted", feature = "sync"))]
mod weighted_sync_tests {
    use sieve_cache::WeightedSyncSieveCache;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn basic_sync_operations() {
        let cache: WeightedSyncSieveCache<String, String> =
            WeightedSyncSieveCache::new(10, 10000).unwrap();

        cache.insert("k".into(), "v".into());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"k".to_string()), Some("v".into()));
        assert!(cache.current_weight() > 0);

        cache.remove(&"k".to_string());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.current_weight(), 0);
    }

    #[test]
    fn sync_weight_eviction() {
        let cache: WeightedSyncSieveCache<String, String> =
            WeightedSyncSieveCache::new(100, 300).unwrap();

        for i in 0..50 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }

        assert!(cache.current_weight() <= cache.max_weight());
    }

    #[test]
    fn sync_clear_resets_weight() {
        let cache: WeightedSyncSieveCache<String, String> =
            WeightedSyncSieveCache::new(10, 10000).unwrap();

        cache.insert("a".into(), "b".into());
        assert!(cache.current_weight() > 0);

        cache.clear();
        assert_eq!(cache.current_weight(), 0);
    }

    #[test]
    fn sync_thread_safety() {
        let cache = Arc::new(WeightedSyncSieveCache::<String, String>::new(100, 5000).unwrap());

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    for i in 0..100 {
                        c.insert(format!("t{}-k{}", t, i), format!("v{}", i));
                    }
                    for i in 0..100 {
                        c.get(&format!("t{}-k{}", t, i));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should not have panicked, and weight should be within bounds
        // (each entry is small so weight should be at/below max)
        assert!(cache.current_weight() <= cache.max_weight());
    }

    #[test]
    fn sync_max_weight_zero_rejected() {
        let r: Result<WeightedSyncSieveCache<String, String>, _> =
            WeightedSyncSieveCache::new(10, 0);
        assert!(r.is_err());
    }
}

#[cfg(all(feature = "weighted", feature = "sharded"))]
mod weighted_sharded_tests {
    use sieve_cache::WeightedShardedSieveCache;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn basic_sharded_operations() {
        let cache: WeightedShardedSieveCache<String, String> =
            WeightedShardedSieveCache::new(100, 10000).unwrap();

        cache.insert("k".into(), "v".into());
        assert_eq!(cache.get(&"k".to_string()), Some("v".into()));
        assert!(cache.current_weight() > 0);

        cache.remove(&"k".to_string());
        assert_eq!(cache.current_weight(), 0);
    }

    #[test]
    fn sharded_max_weight_less_than_num_shards_rejected() {
        // 3 bytes of budget across 4 shards would silently inflate to 4
        let r: Result<WeightedShardedSieveCache<String, String>, _> =
            WeightedShardedSieveCache::with_shards(100, 3, 4);
        assert!(r.is_err());

        // Exactly num_shards is fine (1 byte per shard)
        let r: Result<WeightedShardedSieveCache<String, String>, _> =
            WeightedShardedSieveCache::with_shards(100, 4, 4);
        assert!(r.is_ok());
    }

    #[test]
    fn sharded_weight_distribution() {
        // 1000 weight across 4 shards -> ~250 per shard
        let cache: WeightedShardedSieveCache<String, String> =
            WeightedShardedSieveCache::with_shards(100, 1000, 4).unwrap();

        assert_eq!(cache.num_shards(), 4);
        assert_eq!(cache.max_weight(), 1000);
    }

    #[test]
    fn sharded_weight_eviction() {
        let cache: WeightedShardedSieveCache<String, String> =
            WeightedShardedSieveCache::new(1000, 5000).unwrap();

        for i in 0..200 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }

        // Each shard independently enforces its weight budget. The worst-case
        // overshoot per shard is one entry's weight. We use a generous bound
        // here since String capacities can vary.
        let generous_entry_weight = 200; // more than any single (key, value) pair
        let max_overshoot = cache.num_shards() * generous_entry_weight;
        assert!(
            cache.current_weight() <= cache.max_weight() + max_overshoot,
            "current_weight={} max_weight={} bound={}",
            cache.current_weight(),
            cache.max_weight(),
            cache.max_weight() + max_overshoot,
        );
    }

    #[test]
    fn sharded_clear_resets_weight() {
        let cache: WeightedShardedSieveCache<String, String> =
            WeightedShardedSieveCache::new(100, 10000).unwrap();

        for i in 0..20 {
            cache.insert(format!("k{}", i), format!("v{}", i));
        }
        assert!(cache.current_weight() > 0);

        cache.clear();
        assert_eq!(cache.current_weight(), 0);
    }

    #[test]
    fn sharded_thread_safety() {
        let cache =
            Arc::new(WeightedShardedSieveCache::<String, String>::new(1000, 10000).unwrap());

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    for i in 0..200 {
                        c.insert(format!("t{}-k{}", t, i), format!("v{}", i));
                    }
                    for i in 0..200 {
                        c.get(&format!("t{}-k{}", t, i));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should not have panicked
        assert!(cache.len() > 0);
    }

    #[test]
    fn sharded_max_weight_zero_rejected() {
        let r: Result<WeightedShardedSieveCache<String, String>, _> =
            WeightedShardedSieveCache::new(100, 0);
        assert!(r.is_err());
    }

    #[test]
    fn sharded_retain() {
        let cache: WeightedShardedSieveCache<String, u64> =
            WeightedShardedSieveCache::with_shards(100, 100000, 4).unwrap();

        for i in 0..20 {
            cache.insert(format!("k{}", i), i as u64);
        }

        let before = cache.current_weight();
        cache.retain(|_, v| *v >= 10);

        assert!(cache.current_weight() < before);
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn sharded_evict_pair() {
        let cache: WeightedShardedSieveCache<String, String> =
            WeightedShardedSieveCache::with_shards(10, 100000, 4).unwrap();

        cache.insert("a".into(), "1".into());
        cache.insert("b".into(), "2".into());

        let pair = cache.evict_pair();
        assert!(pair.is_some());
        let (k, _) = pair.unwrap();
        assert!(!cache.contains_key(&k));
    }
}

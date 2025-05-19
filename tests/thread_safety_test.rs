//! Thread safety stress test for SieveCache implementations.
//!
//! This test verifies that the thread-safe implementations of SieveCache
//! properly handle concurrent operations without data races or other
//! thread safety issues.

use sieve_cache::{ShardedSieveCache, SyncSieveCache};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

const NUM_THREADS: usize = 8;
const OPERATIONS_PER_THREAD: usize = 5000;
const CACHE_CAPACITY: usize = 1000;
const SHARDED_CACHE_SHARDS: usize = 16;

/// Test concurrent operations on SyncSieveCache
#[test]
fn test_sync_cache_concurrent_operations() {
    let cache = Arc::new(SyncSieveCache::new(CACHE_CAPACITY).unwrap());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Pre-populate the cache with some values
    for i in 0..100 {
        cache.insert(format!("init_key{}", i), i);
    }

    let mut handles = Vec::with_capacity(NUM_THREADS);

    for thread_id in 0..NUM_THREADS {
        let cache_clone = Arc::clone(&cache);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            barrier_clone.wait();

            for i in 0..OPERATIONS_PER_THREAD {
                let op = i % 5;
                let key = format!("key{}_{}", thread_id, i % 200);

                match op {
                    0 => {
                        // Insert operation
                        cache_clone.insert(key, i);
                    }
                    1 => {
                        // Get operation
                        let _ = cache_clone.get(&key);
                    }
                    2 => {
                        // Remove operation
                        let _ = cache_clone.remove(&key);
                    }
                    3 => {
                        // get_mut operation
                        cache_clone.get_mut(&key, |value| {
                            *value += 1;
                        });
                    }
                    4 => {
                        // Every 1000 operations, use one of our more complex operations
                        if i % 1000 == 0 {
                            match i % 3 {
                                0 => {
                                    // for_each_value
                                    cache_clone.for_each_value(|value| {
                                        *value += 1;
                                    });
                                }
                                1 => {
                                    // for_each_entry
                                    cache_clone.for_each_entry(|(key, value)| {
                                        if key.contains("_50") {
                                            *value *= 2;
                                        }
                                    });
                                }
                                2 => {
                                    // Use the new batch retain operation occasionally
                                    cache_clone.retain_batch(|key, _| {
                                        !key.contains(&format!("_{}", thread_id))
                                    });
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify the cache is still functional
    let len = cache.len();
    println!("SyncSieveCache final size: {}", len);

    // Insert and retrieve a final value to confirm functionality
    cache.insert("final_key".to_string(), 999);
    assert_eq!(cache.get(&"final_key".to_string()), Some(999));
}

/// Test concurrent operations on ShardedSieveCache
#[test]
fn test_sharded_cache_concurrent_operations() {
    let cache =
        Arc::new(ShardedSieveCache::with_shards(CACHE_CAPACITY, SHARDED_CACHE_SHARDS).unwrap());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Pre-populate the cache with some values
    for i in 0..100 {
        cache.insert(format!("init_key{}", i), i);
    }

    let mut handles = Vec::with_capacity(NUM_THREADS);

    for thread_id in 0..NUM_THREADS {
        let cache_clone = Arc::clone(&cache);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            barrier_clone.wait();

            for i in 0..OPERATIONS_PER_THREAD {
                let op = i % 5;
                let key = format!("key{}_{}", thread_id, i % 200);

                match op {
                    0 => {
                        // Insert operation
                        cache_clone.insert(key, i);
                    }
                    1 => {
                        // Get operation
                        let _ = cache_clone.get(&key);
                    }
                    2 => {
                        // Remove operation
                        let _ = cache_clone.remove(&key);
                    }
                    3 => {
                        // get_mut operation
                        cache_clone.get_mut(&key, |value| {
                            *value += 1;
                        });
                    }
                    4 => {
                        // Every 1000 operations, use one of our more complex operations
                        if i % 1000 == 0 {
                            match i % 3 {
                                0 => {
                                    // for_each_value
                                    cache_clone.for_each_value(|value| {
                                        *value += 1;
                                    });
                                }
                                1 => {
                                    // for_each_entry
                                    cache_clone.for_each_entry(|(key, value)| {
                                        if key.contains("_50") {
                                            *value *= 2;
                                        }
                                    });
                                }
                                2 => {
                                    // retain
                                    cache_clone
                                        .retain(|key, _| !key.contains(&format!("_{}", thread_id)));
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify the cache is still functional
    let len = cache.len();
    println!("ShardedSieveCache final size: {}", len);

    // Insert and retrieve a final value to confirm functionality
    cache.insert("final_key".to_string(), 999);
    assert_eq!(cache.get(&"final_key".to_string()), Some(999));
}

/// Test specific race conditions that could occur in the retain operation
#[test]
fn test_retain_race_conditions() {
    let cache = Arc::new(SyncSieveCache::new(CACHE_CAPACITY).unwrap());

    // Pre-populate the cache with values
    for i in 0..500 {
        cache.insert(format!("key{}", i), i);
    }

    // Thread 1: Continuously modifies values
    let cache_clone1 = Arc::clone(&cache);
    let modifier = thread::spawn(move || {
        for i in 0..50 {
            // Modify existing values
            for j in 0..500 {
                let key = format!("key{}", j);
                cache_clone1.get_mut(&key, |value| {
                    *value += 1;
                });
            }
            // Add new values
            for j in 0..10 {
                let new_key = format!("new_key{}_{}", i, j);
                cache_clone1.insert(new_key, j);
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Thread 2: Performs retain operations while Thread 1 is modifying
    let cache_clone2 = Arc::clone(&cache);
    let retainer = thread::spawn(move || {
        for i in 0..10 {
            // Use standard retain
            if i % 2 == 0 {
                cache_clone2.retain(|key, value| {
                    // Keep only even-valued entries and keys not containing "new"
                    (*value % 2 == 0) || !key.contains("new")
                });
            } else {
                // Use batch retain
                cache_clone2.retain_batch(|key, value| {
                    // Keep only odd-valued entries and keys not containing "new"
                    (*value % 2 == 1) || !key.contains("new")
                });
            }
            thread::sleep(Duration::from_millis(5));
        }
    });

    // Wait for both threads to finish
    modifier.join().unwrap();
    retainer.join().unwrap();

    // Verify the cache is still functional
    let len = cache.len();
    println!("Cache size after retain race test: {}", len);

    // Insert and retrieve one more value to confirm functionality
    cache.insert("after_race_test".to_string(), 1000);
    assert_eq!(cache.get(&"after_race_test".to_string()), Some(1000));
}

/// Test with_lock operations for deadlock prevention
#[test]
fn test_with_lock_operation() {
    let cache = Arc::new(SyncSieveCache::new(100).unwrap());
    cache.insert("key1".to_string(), 1);

    // Test basic with_lock operation
    cache.with_lock(|inner_cache| {
        inner_cache.insert("key2".to_string(), 2);
    });

    assert!(!cache.is_empty());
}

/// Test nested get inside get_mut for deadlock prevention
#[test]
fn test_nested_get_during_mut() {
    let cache = Arc::new(SyncSieveCache::new(100).unwrap());
    cache.insert("key1".to_string(), 1);
    cache.insert("key2".to_string(), 2);

    // Simple nested get operation from within get_mut
    cache.get_mut(&"key1".to_string(), |val| {
        *val += 1;
        // This would deadlock if our implementation were flawed
        let _ = cache.get(&"key2".to_string());
    });

    assert!(!cache.is_empty());
}

/// Test simple get_mut operation for deadlock prevention
#[test]
fn test_get_mut_operation() {
    let cache = Arc::new(SyncSieveCache::new(100).unwrap());
    cache.insert("key1".to_string(), 1);

    // Simple get_mut operation
    cache.get_mut(&"key1".to_string(), |val| {
        *val += 1;
    });

    assert!(!cache.is_empty());
}

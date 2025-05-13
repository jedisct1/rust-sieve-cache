#[macro_use]
extern crate criterion;

use criterion::{black_box, BatchSize, Criterion};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use sieve_cache::SieveCache;
use std::sync::Arc;
use std::thread;

#[cfg(feature = "sync")]
use sieve_cache::SyncSieveCache;

#[cfg(feature = "sharded")]
use sieve_cache::ShardedSieveCache;

/// Benchmark sequential access patterns with the base SieveCache implementation.
///
/// This benchmark measures the performance of sequential insert and get operations
/// without any concurrency concerns, using a small cache size of 68 entries.
fn bench_sequence(c: &mut Criterion) {
    c.bench_function("bench_sequence", |b| {
        let mut cache: SieveCache<u64, u64> = SieveCache::new(68).unwrap();

        // Benchmark sequential inserts
        b.iter_batched(
            || (), // No setup needed for this benchmark
            |_| {
                for i in 1..1000 {
                    let n = i % 100;
                    black_box(cache.insert(n, n));
                }
            },
            BatchSize::SmallInput,
        );

        // Benchmark sequential gets
        b.iter_batched(
            || (), // No setup needed
            |_| {
                for i in 1..1000 {
                    let n = i % 100;
                    black_box(cache.get(&n));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_composite(c: &mut Criterion) {
    c.bench_function("bench_composite", |b| {
        let mut cache: SieveCache<u64, (Vec<u8>, u64)> = SieveCache::new(68).unwrap();

        // Use a new RNG for each iteration to ensure consistent results
        b.iter_batched(
            || StdRng::seed_from_u64(0), // Use deterministic seed
            |mut rng| {
                for _ in 1..1000 {
                    let n = rng.random_range(0..100);
                    black_box(cache.insert(n, (vec![0u8; 12], n)));
                }
            },
            BatchSize::SmallInput,
        );

        // Use a new RNG for each iteration to ensure consistent results
        b.iter_batched(
            || StdRng::seed_from_u64(0), // Use deterministic seed
            |mut rng| {
                for _ in 1..1000 {
                    let n = rng.random_range(0..100);
                    black_box(cache.get(&n));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_composite_normal(c: &mut Criterion) {
    // The cache size is ~ 1x sigma (stddev) to retain roughly >68% of records
    const SIGMA: f64 = 50.0 / 3.0;

    c.bench_function("bench_composite_normal", |b| {
        let mut cache: SieveCache<u64, (Vec<u8>, u64)> = SieveCache::new(SIGMA as usize).unwrap();
        let normal_dist = Normal::new(50.0, SIGMA).unwrap();

        // Use a new RNG for each iteration to ensure consistent results
        b.iter_batched(
            || StdRng::seed_from_u64(0), // Use deterministic seed
            |mut rng| {
                for _ in 1..1000 {
                    let sample = normal_dist.sample(&mut rng);
                    let n = (sample as u64) % 100;
                    black_box(cache.insert(n, (vec![0u8; 12], n)));
                }
            },
            BatchSize::SmallInput,
        );

        // Use a new RNG for each iteration to ensure consistent results
        b.iter_batched(
            || StdRng::seed_from_u64(0), // Use deterministic seed
            |mut rng| {
                for _ in 1..1000 {
                    let sample = normal_dist.sample(&mut rng);
                    let n = (sample as u64) % 100;
                    black_box(cache.get(&n));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

// Interface trait to allow treating both cache implementations uniformly
trait CacheInterface<K, V>: Send + Sync {
    fn insert(&self, key: K, value: V) -> bool;
    fn get(&self, key: &K) -> Option<V>;
}

// Only compile concurrent benchmark when both thread-safe implementations are available
#[cfg(all(feature = "sync", feature = "sharded"))]
mod concurrent_benchmarks {
    use super::*;

    /// Benchmark comparing different thread-safe cache implementations in a high-concurrency scenario.
    ///
    /// This benchmark measures the performance difference between:
    /// 1. SyncSieveCache - using a single mutex for the entire cache
    /// 2. ShardedSieveCache - using multiple mutexes (default 16 shards)
    /// 3. ShardedSieveCache with 32 shards - higher shard count
    ///
    /// The test simulates multiple threads performing random operations (inserts and lookups)
    /// concurrently, which should highlight the benefits of the sharded approach in
    /// reducing lock contention.
    pub fn bench_concurrent_access(c: &mut Criterion) {
        let mut group = c.benchmark_group("concurrent_access");
        group.sample_size(10); // Reduce sample size for these expensive benchmarks

        // Set up benchmark parameters
        const CACHE_SIZE: usize = 10000;
        const NUM_THREADS: usize = 8;
        const OPS_PER_THREAD: usize = 1000;

        // Generic benchmark function to reduce code duplication
        let run_concurrent_benchmark = |cache: Arc<dyn CacheInterface<u64, u64>>| {
            let mut handles = Vec::with_capacity(NUM_THREADS);

            for thread_id in 0..NUM_THREADS {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    // Use a seeded RNG for reproducibility, with different seeds per thread
                    let mut rng = StdRng::seed_from_u64(thread_id as u64);

                    for i in 0..OPS_PER_THREAD {
                        // Use a key range that creates some contention but also some distribution
                        let key = rng.random_range(0..1000);

                        // Mix operations: 40% inserts, 60% reads
                        if i % 10 < 4 {
                            black_box(cache_clone.insert(key, key));
                        } else {
                            black_box(cache_clone.get(&key));
                        }
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        };

        // Benchmark with SyncSieveCache (single mutex)
        group.bench_function("sync_cache", |b| {
            b.iter_batched(
                || {
                    // Setup for each iteration
                    Arc::new(SyncSieveCacheAdapter(
                        SyncSieveCache::new(CACHE_SIZE).unwrap(),
                    ))
                },
                |cache| run_concurrent_benchmark(cache),
                BatchSize::SmallInput,
            );
        });

        // Benchmark with ShardedSieveCache (default: 16 mutexes)
        group.bench_function("sharded_cache_16_shards", |b| {
            b.iter_batched(
                || {
                    // Setup for each iteration
                    Arc::new(ShardedSieveCacheAdapter(
                        ShardedSieveCache::new(CACHE_SIZE).unwrap(),
                    ))
                },
                |cache| run_concurrent_benchmark(cache),
                BatchSize::SmallInput,
            );
        });

        // Benchmark with different shard counts
        group.bench_function("sharded_cache_32_shards", |b| {
            b.iter_batched(
                || {
                    // Setup for each iteration
                    Arc::new(ShardedSieveCacheAdapter(
                        ShardedSieveCache::with_shards(CACHE_SIZE, 32).unwrap(),
                    ))
                },
                |cache| run_concurrent_benchmark(cache),
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }

    // Adapter for SyncSieveCache
    struct SyncSieveCacheAdapter<K: Eq + std::hash::Hash + Clone + Send + Sync, V: Clone + Send + Sync>(
        SyncSieveCache<K, V>,
    );

    impl<K: Eq + std::hash::Hash + Clone + Send + Sync, V: Clone + Send + Sync> CacheInterface<K, V>
        for SyncSieveCacheAdapter<K, V>
    {
        fn insert(&self, key: K, value: V) -> bool {
            self.0.insert(key, value)
        }

        fn get(&self, key: &K) -> Option<V> {
            self.0.get(key)
        }
    }

    // Adapter for ShardedSieveCache
    struct ShardedSieveCacheAdapter<
        K: Eq + std::hash::Hash + Clone + Send + Sync,
        V: Clone + Send + Sync,
    >(ShardedSieveCache<K, V>);

    impl<K: Eq + std::hash::Hash + Clone + Send + Sync, V: Clone + Send + Sync> CacheInterface<K, V>
        for ShardedSieveCacheAdapter<K, V>
    {
        fn insert(&self, key: K, value: V) -> bool {
            self.0.insert(key, value)
        }

        fn get(&self, key: &K) -> Option<V> {
            self.0.get(key)
        }
    }
}

#[cfg(all(feature = "sync", feature = "sharded"))]
criterion_group!(
    benches,
    bench_sequence,
    bench_composite,
    bench_composite_normal,
    concurrent_benchmarks::bench_concurrent_access
);

#[cfg(not(all(feature = "sync", feature = "sharded")))]
criterion_group!(
    benches,
    bench_sequence,
    bench_composite,
    bench_composite_normal
);

criterion_main!(benches);

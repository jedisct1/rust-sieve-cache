#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use rand::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use sieve_cache::{ShardedSieveCache, SieveCache, SyncSieveCache};
use std::sync::Arc;
use std::thread;

fn bench_sequence(c: &mut Criterion) {
    c.bench_function("bench_sequence", |b| {
        let mut cache: SieveCache<u64, u64> = SieveCache::new(68).unwrap();
        b.iter(|| {
            for i in 1..1000 {
                let n = i % 100;
                black_box(cache.insert(n, n));
            }
        });
        b.iter(|| {
            for i in 1..1000 {
                let n = i % 100;
                black_box(cache.get(&n));
            }
        });
    });
}

fn bench_composite(c: &mut Criterion) {
    c.bench_function("bench_composite", |b| {
        let mut cache: SieveCache<u64, (Vec<u8>, u64)> = SieveCache::new(68).unwrap();
        let mut rng = thread_rng();

        b.iter(|| {
            for _ in 1..1000 {
                let n = rng.gen_range(0..100);
                black_box(cache.insert(n, (vec![0u8; 12], n)));
            }
        });

        b.iter(|| {
            for _ in 1..1000 {
                let n = rng.gen_range(0..100);
                black_box(cache.get(&n));
            }
        });
    });
}

fn bench_composite_normal(c: &mut Criterion) {
    // The cache size is ~ 1x sigma (stddev) to retain roughly >68% of records
    const SIGMA: f64 = 50.0 / 3.0;

    c.bench_function("bench_composite_normal", |b| {
        let mut cache: SieveCache<u64, (Vec<u8>, u64)> = SieveCache::new(SIGMA as usize).unwrap();

        // This should roughly cover all elements (within 3-sigma)
        let mut rng = thread_rng();
        let normal = Normal::new(50.0, SIGMA).unwrap();

        b.iter(|| {
            for _ in 1..1000 {
                let sample = normal.sample(&mut rng);
                let n = (sample as u64) % 100;
                black_box(cache.insert(n, (vec![0u8; 12], n)));
            }
        });

        b.iter(|| {
            for _ in 1..1000 {
                let sample = normal.sample(&mut rng);
                let n = (sample as u64) % 100;
                black_box(cache.get(&n));
            }
        });
    });
}

// Benchmark to compare thread-safe implementations in high-concurrency scenario
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");

    // Set up benchmark parameters
    const CACHE_SIZE: usize = 10000;
    const NUM_THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 1000;

    // Benchmark with SyncSieveCache (single mutex)
    group.bench_function("sync_cache", |b| {
        b.iter(|| {
            let cache = Arc::new(SyncSieveCache::new(CACHE_SIZE).unwrap());
            let mut handles = Vec::with_capacity(NUM_THREADS);

            for _ in 0..NUM_THREADS {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    let mut rng = thread_rng();

                    for _ in 0..OPS_PER_THREAD {
                        let key = rng.gen_range(0..1000);
                        if rng.gen::<bool>() {
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
        });
    });

    // Benchmark with ShardedSieveCache (multiple mutexes)
    group.bench_function("sharded_cache", |b| {
        b.iter(|| {
            let cache = Arc::new(ShardedSieveCache::new(CACHE_SIZE).unwrap());
            let mut handles = Vec::with_capacity(NUM_THREADS);

            for _ in 0..NUM_THREADS {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    let mut rng = thread_rng();

                    for _ in 0..OPS_PER_THREAD {
                        let key = rng.gen_range(0..1000);
                        if rng.gen::<bool>() {
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
        });
    });

    // Benchmark with different shard counts
    group.bench_function("sharded_cache_32_shards", |b| {
        b.iter(|| {
            let cache = Arc::new(ShardedSieveCache::with_shards(CACHE_SIZE, 32).unwrap());
            let mut handles = Vec::with_capacity(NUM_THREADS);

            for _ in 0..NUM_THREADS {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    let mut rng = thread_rng();

                    for _ in 0..OPS_PER_THREAD {
                        let key = rng.gen_range(0..1000);
                        if rng.gen::<bool>() {
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
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sequence,
    bench_composite,
    bench_composite_normal,
    bench_concurrent_access
);
criterion_main!(benches);

#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};
use sieve_cache::SieveCache;

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
        let mut rng = rng();
        let uniform = Uniform::new(0, 100).unwrap();
        let mut rand_iter = uniform.sample_iter(&mut rng);
        b.iter(|| {
            for _ in 1..1000 {
                let n = rand_iter.next().unwrap();
                black_box(cache.insert(n, (vec![0u8; 12], n)));
            }
        });
        b.iter(|| {
            for _ in 1..1000 {
                let n = rand_iter.next().unwrap();
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
        let mut rng = rng();
        let normal = Normal::new(50.0, SIGMA).unwrap();
        let mut rand_iter = normal.sample_iter(&mut rng).map(|x| (x as u64) % 100);
        b.iter(|| {
            for _ in 1..1000 {
                let n = rand_iter.next().unwrap();
                black_box(cache.insert(n, (vec![0u8; 12], n)));
            }
        });
        b.iter(|| {
            for _ in 1..1000 {
                let n = rand_iter.next().unwrap();
                black_box(cache.get(&n));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_sequence,
    bench_composite,
    bench_composite_normal
);
criterion_main!(benches);

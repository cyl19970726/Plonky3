use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::extension::Complex;
use p3_field::TwoAdicField;
use p3_fri::yu_fold_poly;
use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_yu<F: TwoAdicField>(c: &mut Criterion, log_sizes: &[usize], folding_factors: &[usize])
where
    Standard: Distribution<F>,
{
    let name = format!("yu_fold_poly::<{}>", type_name::<F>(),);
    let mut group = c.benchmark_group(&name);
    group.sample_size(10);

    for log_size in log_sizes {
        for folding_factor in folding_factors {
            let n = 1 << log_size;

            let mut rng = thread_rng();
            let beta = rng.sample(Standard);
            let poly = rng.sample_iter(Standard).take(n).collect_vec();

            let benchmark_id: String = format!("n: {}, folding_factor: {}", n, folding_factor);
            group.bench_function(BenchmarkId::new(&benchmark_id, n), |b| {
                b.iter(|| {
                    yu_fold_poly(poly.clone(), beta, *folding_factor);
                })
            });
        }
    }
}

fn bench_yu_fold_poly(c: &mut Criterion) {
    let log_sizes = [12, 14, 16, 18, 20, 22];
    let folding_factors = [2, 4, 8, 16];

    bench_yu::<BabyBear>(c, &log_sizes, &folding_factors);
    bench_yu::<Goldilocks>(c, &log_sizes, &folding_factors);
    bench_yu::<Complex<Mersenne31>>(c, &log_sizes, &folding_factors);
}

criterion_group!(benches, bench_yu_fold_poly);
criterion_main!(benches);
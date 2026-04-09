use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use rusty_rose::feature::spectral::{tonnetz, poly_features, rms};
use rusty_rose::core_io::time_unit_conversion::time_to_samples;
use rusty_rose::core_io::time_domain_processing::{autocorrelate, lpc, zero_crossing, mu_compress};
use rusty_rose::core_io::spectrum::{power_to_db, Ref};
use rusty_rose::effects::misc::{preemphasis, trim};
use ndarray::IxDyn;

fn bench_spectral_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_features");
    
    // Tonnetz
    for t in [10, 100, 1000].iter() {
        group.bench_with_input(criterion::BenchmarkId::new("tonnetz", t), t, |b, &t| {
            let chroma = Array2::<f64>::ones((12, t));
            b.iter(|| tonnetz(black_box(&chroma)));
        });
    }

    // Poly Features
    let n_freqs = 100;
    let freq = Array1::linspace(0.0, 5000.0, n_freqs);
    let order = 2;
    for t in [10, 100, 1000].iter() {
        group.bench_with_input(criterion::BenchmarkId::new("poly_features", t), t, |b, &t| {
            let s = Array2::<f64>::ones((n_freqs, t));
            b.iter(|| poly_features(black_box(&s), black_box(&freq), order));
        });
    }
    group.finish();
}

fn bench_time_domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_domain");
    let n = 1024;
    let y = ndarray::ArrayD::<f32>::zeros(IxDyn(&[n]));
    
    group.bench_function("autocorrelate", |b| b.iter(|| autocorrelate(black_box(&y), None, 0)));
    group.bench_function("lpc", |b| b.iter(|| lpc(black_box(&y), 12, 0)));
    group.bench_function("zero_crossing", |b| b.iter(|| zero_crossing(black_box(&y), 0.0, None, true, true, 0)));
    group.bench_function("mu_compress", |b| b.iter(|| mu_compress(black_box(&y), 255.0, true)));
    group.finish();
}

fn bench_spectral_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_scaling");
    let n = 1024;
    let s = vec![1.0; n];
    let y = Array1::<f64>::zeros(n);

    group.bench_function("power_to_db", |b| b.iter(|| power_to_db(black_box(&s), Ref::Scalar(1.0), 1e-10, None)));
    group.bench_function("rms", |b| b.iter(|| rms(black_box(&y), None, None, true)));
    group.finish();
}

fn bench_utils_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils_effects");
    let n = 1024;
    let y = Array1::<f64>::zeros(n);
    
    group.bench_function("time_to_samples", |b| b.iter(|| time_to_samples(black_box(1.0), None)));
    group.bench_function("preemphasis", |b| b.iter(|| preemphasis(black_box(&y), None, None)));
    group.bench_function("trim", |b| b.iter(|| trim(black_box(&y), None, 1.0, None, None)));
    group.finish();
}

criterion_group!(benches, 
    bench_spectral_features,
    bench_time_domain,
    bench_spectral_scaling,
    bench_utils_effects
);
criterion_main!(benches);

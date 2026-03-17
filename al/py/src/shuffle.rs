// PyO3 bindings for ml::shuffle — cross-language deterministic RNG.

use pyo3::prelude::*;

/// Deterministic shuffle of [0, 1, ..., n-1] using PCG-XSH-RR.
/// Same (n, seed) → same permutation on all platforms, all languages.
#[pyfunction]
pub fn shuffle(n: usize, seed: u64) -> Vec<usize> {
    ml::shuffle::shuffle(n, seed)
}

/// Canonical partition sizes: (n_train, n_valid, n_test).
/// Uses round(n * ratio) — matches R and Julia.
#[pyfunction]
pub fn partition_sizes(n: usize, train: f64, valid: f64, test: f64) -> (usize, usize, usize) {
    ml::shuffle::partition_sizes(n, [train, valid, test])
}

//! K-Nearest Neighbors via KD-tree + vectorized brute-force.
//!
//! - d <= 24: balanced KD-tree, O(k log n) per query.
//! - d > 24: batch brute-force via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
//!   using BLAS GEMM (Accelerate on macOS, matrixmultiply fallback).
//!
//! All predict paths are rayon-parallel across queries.

use crate::error::MlError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Distance weighting
// ---------------------------------------------------------------------------

/// Weighting scheme for KNN predictions.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum KnnWeights {
    /// All neighbors contribute equally (default, backward-compatible).
    Uniform,
    /// Neighbors are weighted by inverse distance: w_i = 1/(d_i + eps).
    Distance,
}

impl Default for KnnWeights {
    fn default() -> Self {
        KnnWeights::Uniform
    }
}

// ---------------------------------------------------------------------------
// KD-tree node
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
enum KdNode {
    /// Internal node: splits on a dimension.
    Split {
        /// Index into the original training data (the pivot point).
        idx: usize,
        /// Split dimension.
        dim: usize,
        /// Split value (the point's coordinate at `dim`).
        split: f64,
        left: Option<Box<KdNode>>,
        right: Option<Box<KdNode>>,
    },
    /// Leaf node: brute-force scan over a small set of points.
    Leaf {
        /// Indices into the original training data.
        indices: Vec<usize>,
    },
}

/// Build a balanced KD-tree with leaf_size.  `data` is flat row-major (n×d).
fn build_kdtree(indices: &mut [usize], data: &[f64], d: usize, depth: usize) -> Option<Box<KdNode>> {
    if indices.is_empty() {
        return None;
    }
    // Leaf node: brute-force when few points remain
    if indices.len() <= KDTREE_LEAF_SIZE {
        return Some(Box::new(KdNode::Leaf {
            indices: indices.to_vec(),
        }));
    }
    let dim = depth % d;
    indices.sort_unstable_by(|&a, &b| {
        data[a * d + dim]
            .partial_cmp(&data[b * d + dim])
            .unwrap()
    });
    let mid = indices.len() / 2;
    let idx = indices[mid];
    let (left_slice, right_slice) = {
        let (left, rest) = indices.split_at_mut(mid);
        let right = &mut rest[1..];
        (left, right)
    };
    Some(Box::new(KdNode::Split {
        idx,
        dim,
        split: data[idx * d + dim],
        left: build_kdtree(left_slice, data, d, depth + 1),
        right: build_kdtree(right_slice, data, d, depth + 1),
    }))
}

// ---------------------------------------------------------------------------
// Bounded max-heap for k nearest neighbors
// ---------------------------------------------------------------------------

struct KnnHeap {
    k: usize,
    items: Vec<(f64, usize)>,
}

impl KnnHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    fn worst_dist(&self) -> f64 {
        if self.items.len() < self.k {
            f64::INFINITY
        } else {
            self.items[0].0
        }
    }

    fn push(&mut self, dist: f64, idx: usize) {
        if self.items.len() < self.k {
            self.items.push((dist, idx));
            self.sift_up(self.items.len() - 1);
        } else if dist < self.items[0].0 {
            self.items[0] = (dist, idx);
            self.sift_down(0);
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.items[i].0 > self.items[parent].0 {
                self.items.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.items.len();
        loop {
            let mut largest = i;
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < n && self.items[left].0 > self.items[largest].0 {
                largest = left;
            }
            if right < n && self.items[right].0 > self.items[largest].0 {
                largest = right;
            }
            if largest != i {
                self.items.swap(i, largest);
                i = largest;
            } else {
                break;
            }
        }
    }

    fn into_sorted(mut self) -> Vec<usize> {
        self.items
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items.into_iter().map(|(_, idx)| idx).collect()
    }

    /// Return sorted (squared_distance, training_index) pairs.
    fn into_sorted_with_dists(mut self) -> Vec<(f64, usize)> {
        self.items
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items
    }
}

// ---------------------------------------------------------------------------
// KD-tree query (flat data)
// ---------------------------------------------------------------------------

#[inline]
fn sq_dist_flat(data: &[f64], i: usize, query: &[f64], d: usize) -> f64 {
    let row = &data[i * d..i * d + d];
    let mut sum = 0.0_f64;
    // Manual loop for autovectorization (LLVM vectorizes counted loops better than iterators)
    for j in 0..d {
        let diff = unsafe { *row.get_unchecked(j) - *query.get_unchecked(j) };
        sum += diff * diff;
    }
    sum
}

fn query_kdtree(
    node: &Option<Box<KdNode>>,
    query: &[f64],
    data: &[f64],
    d: usize,
    heap: &mut KnnHeap,
) {
    let node = match node {
        Some(n) => n,
        None => return,
    };

    match node.as_ref() {
        KdNode::Leaf { indices } => {
            // Brute-force scan over leaf points
            for &idx in indices {
                let dist = sq_dist_flat(data, idx, query, d);
                heap.push(dist, idx);
            }
        }
        KdNode::Split { idx, dim, split, left, right } => {
            let dist = sq_dist_flat(data, *idx, query, d);
            heap.push(dist, *idx);

            let diff = query[*dim] - split;

            let (first, second) = if diff <= 0.0 {
                (left, right)
            } else {
                (right, left)
            };

            query_kdtree(first, query, data, d, heap);

            if diff * diff < heap.worst_dist() {
                query_kdtree(second, query, data, d, heap);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batch brute-force: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b  (BLAS GEMM)
// ---------------------------------------------------------------------------

use crate::blas;

/// Squared norm of a row-major slice.
#[inline]
fn sq_norm(row: &[f64]) -> f64 {
    row.iter().map(|x| x * x).sum()
}

/// GEMM brute-force threshold: use GEMM when m*n fits in ~50MB.
/// Reduced from 25M (200MB) to 6.25M (50MB) for better cache reuse
/// and lower peak memory. The chunked-heap path handles larger cases
/// with bounded memory via per-query heaps.
const GEMM_MAX_ELEMENTS: usize = 6_250_000;

/// Compute k nearest neighbor indices for each of m queries against n
/// training points, all in d dimensions.  Returns Vec of m Vec<usize>.
///
/// Strategy:
/// - Small m*n (fits in ~200MB): BLAS GEMM for all dot products at once.
/// - Large m*n: per-query scan with bounded heap (O(k) memory per query).
fn batch_brute_knn(
    train: &[f64],   // n×d row-major
    queries: &[f64], // m×d row-major
    n: usize,
    m: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<usize>> {
    // Pre-compute ||train_i||^2
    let train_sq: Vec<f64> = (0..n)
        .map(|i| sq_norm(&train[i * d..(i + 1) * d]))
        .collect();

    if m * n <= GEMM_MAX_ELEMENTS {
        // Small enough for one GEMM call
        batch_brute_gemm(train, queries, &train_sq, n, m, d, k)
    } else {
        // Large n: per-query heap scan (O(k) memory per query)
        batch_brute_heap(train, queries, &train_sq, n, m, d, k)
    }
}

/// Like `batch_brute_knn` but returns `(squared_distance, index)` pairs.
fn batch_brute_knn_with_dists(
    train: &[f64],
    queries: &[f64],
    n: usize,
    m: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<(f64, usize)>> {
    let train_sq: Vec<f64> = (0..n)
        .map(|i| sq_norm(&train[i * d..(i + 1) * d]))
        .collect();

    if m * n <= GEMM_MAX_ELEMENTS {
        batch_brute_gemm_with_dists(train, queries, &train_sq, n, m, d, k)
    } else {
        batch_brute_heap_with_dists(train, queries, &train_sq, n, m, d, k)
    }
}

/// GEMM path: compute all dot products at once, then partial sort.
fn batch_brute_gemm(
    train: &[f64],
    queries: &[f64],
    train_sq: &[f64],
    n: usize,
    m: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<usize>> {
    let query_sq: Vec<f64> = (0..m)
        .map(|i| sq_norm(&queries[i * d..(i + 1) * d]))
        .collect();

    let mut dots = vec![0.0f64; m * n];
    blas::gemm_ab_t(queries, train, &mut dots, m, n, d);

    (0..m)
        .into_par_iter()
        .map(|qi| {
            let q_sq = query_sq[qi];
            let dot_row = &dots[qi * n..(qi + 1) * n];

            let mut dists: Vec<(f64, usize)> = (0..n)
                .map(|j| {
                    let d2 = (q_sq + train_sq[j] - 2.0 * dot_row[j]).max(0.0);
                    (d2, j)
                })
                .collect();

            if k >= n {
                dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                return dists.iter().map(|(_, i)| *i).collect();
            }

            dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.truncate(k);
            dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.iter().map(|(_, i)| *i).collect()
        })
        .collect()
}

/// GEMM path returning `(squared_distance, index)` pairs.
fn batch_brute_gemm_with_dists(
    train: &[f64],
    queries: &[f64],
    train_sq: &[f64],
    n: usize,
    m: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<(f64, usize)>> {
    let query_sq: Vec<f64> = (0..m)
        .map(|i| sq_norm(&queries[i * d..(i + 1) * d]))
        .collect();

    let mut dots = vec![0.0f64; m * n];
    blas::gemm_ab_t(queries, train, &mut dots, m, n, d);

    (0..m)
        .into_par_iter()
        .map(|qi| {
            let q_sq = query_sq[qi];
            let dot_row = &dots[qi * n..(qi + 1) * n];

            let mut dists: Vec<(f64, usize)> = (0..n)
                .map(|j| {
                    let d2 = (q_sq + train_sq[j] - 2.0 * dot_row[j]).max(0.0);
                    (d2, j)
                })
                .collect();

            if k >= n {
                dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                return dists;
            }

            dists.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.truncate(k);
            dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists
        })
        .collect()
}

/// Chunked-training path: split training data into GEMM-sized chunks,
/// compute dot products per chunk, and feed into bounded heaps.
/// Uses BLAS GEMM without allocating an m×n matrix.
fn batch_brute_heap(
    train: &[f64],
    queries: &[f64],
    train_sq: &[f64],
    n: usize,
    m: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<usize>> {
    // Chunk training data so each GEMM is m × chunk_n
    let chunk_n = (GEMM_MAX_ELEMENTS / m).max(1).min(n);

    // Pre-compute query squared norms
    let query_sq: Vec<f64> = (0..m)
        .map(|i| sq_norm(&queries[i * d..(i + 1) * d]))
        .collect();

    // Initialize heaps for all queries
    let mut heaps: Vec<KnnHeap> = (0..m).map(|_| KnnHeap::new(k)).collect();

    // Process training data in chunks
    for chunk_start in (0..n).step_by(chunk_n) {
        let chunk_end = (chunk_start + chunk_n).min(n);
        let cn = chunk_end - chunk_start;
        let t_slice = &train[chunk_start * d..chunk_end * d];
        let tsq_slice = &train_sq[chunk_start..chunk_end];

        // GEMM: dots = queries(m×d) × chunk^T(d×cn)
        let mut dots = vec![0.0f64; m * cn];
        blas::gemm_ab_t(queries, t_slice, &mut dots, m, cn, d);

        // Update heaps with distances from this chunk (parallel across queries)
        heaps.par_iter_mut().enumerate().for_each(|(qi, heap)| {
            let q_sq = query_sq[qi];
            let dot_row = &dots[qi * cn..(qi + 1) * cn];
            for j in 0..cn {
                let d2 = (q_sq + tsq_slice[j] - 2.0 * dot_row[j]).max(0.0);
                heap.push(d2, chunk_start + j);
            }
        });
    }

    heaps.into_iter().map(|h| h.into_sorted()).collect()
}

/// Chunked-training path returning `(squared_distance, index)` pairs.
fn batch_brute_heap_with_dists(
    train: &[f64],
    queries: &[f64],
    train_sq: &[f64],
    n: usize,
    m: usize,
    d: usize,
    k: usize,
) -> Vec<Vec<(f64, usize)>> {
    let chunk_n = (GEMM_MAX_ELEMENTS / m).max(1).min(n);
    let query_sq: Vec<f64> = (0..m)
        .map(|i| sq_norm(&queries[i * d..(i + 1) * d]))
        .collect();
    let mut heaps: Vec<KnnHeap> = (0..m).map(|_| KnnHeap::new(k)).collect();

    for chunk_start in (0..n).step_by(chunk_n) {
        let chunk_end = (chunk_start + chunk_n).min(n);
        let cn = chunk_end - chunk_start;
        let t_slice = &train[chunk_start * d..chunk_end * d];
        let tsq_slice = &train_sq[chunk_start..chunk_end];

        let mut dots = vec![0.0f64; m * cn];
        blas::gemm_ab_t(queries, t_slice, &mut dots, m, cn, d);

        heaps.par_iter_mut().enumerate().for_each(|(qi, heap)| {
            let q_sq = query_sq[qi];
            let dot_row = &dots[qi * cn..(qi + 1) * cn];
            for j in 0..cn {
                let d2 = (q_sq + tsq_slice[j] - 2.0 * dot_row[j]).max(0.0);
                heap.push(d2, chunk_start + j);
            }
        });
    }

    heaps
        .into_iter()
        .map(|h| h.into_sorted_with_dists())
        .collect()
}

// ---------------------------------------------------------------------------
// KNN Model
// ---------------------------------------------------------------------------

/// KD-tree / brute-force crossover dimension.
///
/// d <= 24: KD-tree with leaf_size (O(k log n) per query, fast at moderate dims).
/// d > 24: BLAS GEMM brute-force (chunked for memory safety).
///
/// sklearn uses KD-tree up to ~30 dims. We use 24 as a balanced crossover:
/// KD-tree wins on real 20-30D datasets (breast cancer 30D, housing 13D),
/// while brute-force GEMM is better above ~30D where the curse of
/// dimensionality degrades tree pruning.
const KDTREE_DIM_LIMIT: usize = 24;

/// Number of points in a KD-tree leaf node below which we brute-force scan.
/// Avoids deep recursion overhead; leaf scan is cache-friendly.
/// sklearn default: 30. We use 32 (cache line aligned).
const KDTREE_LEAF_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Distance weighting helper
// ---------------------------------------------------------------------------

/// Epsilon to prevent division by zero for exact matches.
const DIST_EPS: f64 = 1e-10;

/// Compute normalized inverse-distance weights from squared distances.
///
/// `w_i = 1 / (sqrt(d_i^2) + eps)`, then normalize so `sum(w_i) = 1`.
/// Input distances are squared Euclidean; we take sqrt first.
#[inline]
fn inverse_distance_weights(neighbors: &[(f64, usize)]) -> Vec<f64> {
    let mut weights: Vec<f64> = neighbors
        .iter()
        .map(|&(sq_dist, _)| 1.0 / (sq_dist.sqrt() + DIST_EPS))
        .collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        for w in &mut weights {
            *w /= total;
        }
    }
    weights
}

/// K-Nearest Neighbors model (classification + regression).
///
/// Uses KD-tree for low-d or large-n, BLAS GEMM brute-force otherwise.
#[derive(Serialize, Deserialize)]
pub struct KnnModel {
    pub k: usize,
    pub n_samples: usize,
    pub n_features: usize,
    /// Flat row-major training data, length = n_samples * n_features.
    data: Vec<f64>,
    /// Classification labels (0-based i64). Empty for regression.
    labels_clf: Vec<i64>,
    /// Regression targets. Empty for classification.
    targets_reg: Vec<f64>,
    /// Number of classes (classification only).
    pub n_classes: usize,
    /// KD-tree root. None if brute-force mode.
    tree: Option<Box<KdNode>>,
    /// Whether we're using brute-force (d > KDTREE_DIM_LIMIT).
    use_brute: bool,
    /// Weighting scheme for predictions (Uniform or Distance).
    #[serde(default)]
    pub weights: KnnWeights,
}

impl KnnModel {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            n_samples: 0,
            n_features: 0,
            data: Vec::new(),
            labels_clf: Vec::new(),
            targets_reg: Vec::new(),
            n_classes: 0,
            tree: None,
            use_brute: false,
            weights: KnnWeights::Uniform,
        }
    }

    /// Fit for classification.  `y` must be 0-based integer labels.
    pub fn fit_clf(
        &mut self,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[i64],
    ) -> Result<(), MlError> {
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }

        self.n_samples = n;
        self.n_features = d;
        self.data = x.to_vec();
        self.labels_clf = y.to_vec();
        self.n_classes = (*y.iter().max().unwrap_or(&0) + 1) as usize;
        self.targets_reg.clear();

        self.build_tree();
        Ok(())
    }

    /// Fit for regression.
    pub fn fit_reg(
        &mut self,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
    ) -> Result<(), MlError> {
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }

        self.n_samples = n;
        self.n_features = d;
        self.data = x.to_vec();
        self.targets_reg = y.to_vec();
        self.labels_clf.clear();
        self.n_classes = 0;

        self.build_tree();
        Ok(())
    }

    fn build_tree(&mut self) {
        if self.n_features > KDTREE_DIM_LIMIT {
            self.use_brute = true;
            self.tree = None;
            return;
        }
        self.use_brute = false;
        let mut indices: Vec<usize> = (0..self.n_samples).collect();
        self.tree = build_kdtree(&mut indices, &self.data, self.n_features, 0);
    }

    /// Find k nearest neighbor indices for a single query point (KD-tree path).
    fn query_one(&self, query: &[f64]) -> Vec<usize> {
        let mut heap = KnnHeap::new(self.k);
        query_kdtree(&self.tree, query, &self.data, self.n_features, &mut heap);
        heap.into_sorted()
    }

    /// Find k nearest neighbors with squared distances (KD-tree path).
    fn query_one_with_dists(&self, query: &[f64]) -> Vec<(f64, usize)> {
        let mut heap = KnnHeap::new(self.k);
        query_kdtree(&self.tree, query, &self.data, self.n_features, &mut heap);
        heap.into_sorted_with_dists()
    }

    /// Predict class labels for multiple query points.
    pub fn predict_clf(&self, x: &[f64], n: usize, d: usize) -> Vec<i64> {
        match self.weights {
            KnnWeights::Uniform => {
                let neighbors = self.find_all_neighbors(x, n, d);
                neighbors
                    .into_par_iter()
                    .map(|nn| {
                        let mut counts = vec![0usize; self.n_classes];
                        for &idx in &nn {
                            counts[self.labels_clf[idx] as usize] += 1;
                        }
                        counts
                            .iter()
                            .enumerate()
                            .max_by_key(|&(_, c)| c)
                            .map(|(cls, _)| cls as i64)
                            .unwrap_or(0)
                    })
                    .collect()
            }
            KnnWeights::Distance => {
                let neighbors = self.find_all_neighbors_with_dists(x, n, d);
                let nc = self.n_classes;
                neighbors
                    .into_par_iter()
                    .map(|nn| {
                        let weights = inverse_distance_weights(&nn);
                        let mut wt_counts = vec![0.0f64; nc];
                        for (i, &(_, idx)) in nn.iter().enumerate() {
                            wt_counts[self.labels_clf[idx] as usize] += weights[i];
                        }
                        wt_counts
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(cls, _)| cls as i64)
                            .unwrap_or(0)
                    })
                    .collect()
            }
        }
    }

    /// Predict class probabilities.  Returns flat Vec of shape (n, n_classes).
    pub fn predict_proba(&self, x: &[f64], n: usize, d: usize) -> Vec<f64> {
        let nc = self.n_classes;
        match self.weights {
            KnnWeights::Uniform => {
                let neighbors = self.find_all_neighbors(x, n, d);
                let k = self.k;
                neighbors
                    .into_par_iter()
                    .flat_map(|nn| {
                        let mut counts = vec![0usize; nc];
                        for &idx in &nn {
                            counts[self.labels_clf[idx] as usize] += 1;
                        }
                        let kf = k.max(nn.len()) as f64;
                        counts
                            .iter()
                            .map(|&c| c as f64 / kf)
                            .collect::<Vec<_>>()
                    })
                    .collect()
            }
            KnnWeights::Distance => {
                let neighbors = self.find_all_neighbors_with_dists(x, n, d);
                neighbors
                    .into_par_iter()
                    .flat_map(|nn| {
                        let weights = inverse_distance_weights(&nn);
                        let mut proba = vec![0.0f64; nc];
                        for (i, &(_, idx)) in nn.iter().enumerate() {
                            proba[self.labels_clf[idx] as usize] += weights[i];
                        }
                        // weights already sum to 1.0 via inverse_distance_weights
                        proba
                    })
                    .collect()
            }
        }
    }

    /// Predict regression targets.
    pub fn predict_reg(&self, x: &[f64], n: usize, d: usize) -> Vec<f64> {
        match self.weights {
            KnnWeights::Uniform => {
                let neighbors = self.find_all_neighbors(x, n, d);
                neighbors
                    .into_par_iter()
                    .map(|nn| {
                        let sum: f64 = nn.iter().map(|&idx| self.targets_reg[idx]).sum();
                        sum / nn.len() as f64
                    })
                    .collect()
            }
            KnnWeights::Distance => {
                let neighbors = self.find_all_neighbors_with_dists(x, n, d);
                neighbors
                    .into_par_iter()
                    .map(|nn| {
                        let weights = inverse_distance_weights(&nn);
                        nn.iter()
                            .enumerate()
                            .map(|(i, &(_, idx))| weights[i] * self.targets_reg[idx])
                            .sum()
                    })
                    .collect()
            }
        }
    }

    /// Dispatch: KD-tree per-query or batch GEMM brute-force.
    fn find_all_neighbors(&self, x: &[f64], n: usize, d: usize) -> Vec<Vec<usize>> {
        if self.use_brute {
            batch_brute_knn(&self.data, x, self.n_samples, n, d, self.k)
        } else {
            let queries: Vec<&[f64]> = (0..n).map(|i| &x[i * d..(i + 1) * d]).collect();
            queries
                .par_iter()
                .map(|q| self.query_one(q))
                .collect()
        }
    }

    /// Dispatch returning (squared_distance, index) pairs for distance weighting.
    fn find_all_neighbors_with_dists(
        &self,
        x: &[f64],
        n: usize,
        d: usize,
    ) -> Vec<Vec<(f64, usize)>> {
        if self.use_brute {
            batch_brute_knn_with_dists(&self.data, x, self.n_samples, n, d, self.k)
        } else {
            let queries: Vec<&[f64]> = (0..n).map(|i| &x[i * d..(i + 1) * d]).collect();
            queries
                .par_iter()
                .map(|q| self.query_one_with_dists(q))
                .collect()
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 2-class dataset in 2D:
    ///   Class 0: (0,0), (1,0), (0,1)
    ///   Class 1: (10,10), (11,10), (10,11)
    fn clf_data() -> (Vec<f64>, Vec<i64>, usize, usize) {
        let x = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // class 0
            10.0, 10.0, 11.0, 10.0, 10.0, 11.0, // class 1
        ];
        let y = vec![0, 0, 0, 1, 1, 1];
        (x, y, 6, 2)
    }

    /// Simple regression dataset in 2D.
    fn reg_data() -> (Vec<f64>, Vec<f64>, usize, usize) {
        let x = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0, 11.0, 10.0, 10.0, 11.0,
        ];
        let y = vec![1.0, 2.0, 3.0, 100.0, 110.0, 120.0];
        (x, y, 6, 2)
    }

    #[test]
    fn test_knn_distance_weighted_basic() {
        let (x, y, n, d) = clf_data();
        let mut model = KnnModel::new(3);
        model.weights = KnnWeights::Distance;
        model.fit_clf(&x, n, d, &y).unwrap();

        // Query near class 0 cluster
        let q = vec![0.5, 0.5];
        let preds = model.predict_clf(&q, 1, d);
        assert_eq!(preds[0], 0, "query near class 0 should predict class 0");

        // Query near class 1 cluster
        let q = vec![10.5, 10.5];
        let preds = model.predict_clf(&q, 1, d);
        assert_eq!(preds[0], 1, "query near class 1 should predict class 1");
    }

    #[test]
    fn test_knn_distance_weighted_reg() {
        let (x, y, n, d) = reg_data();
        let mut model = KnnModel::new(3);
        model.weights = KnnWeights::Distance;
        model.fit_reg(&x, n, d, &y).unwrap();

        // Query near low-value cluster
        let q = vec![0.1, 0.1];
        let preds = model.predict_reg(&q, 1, d);
        assert!(preds[0] > 0.5 && preds[0] < 10.0,
            "prediction near low-value cluster should be low, got {}", preds[0]);

        // Query near high-value cluster
        let q = vec![10.1, 10.1];
        let preds = model.predict_reg(&q, 1, d);
        assert!(preds[0] > 90.0 && preds[0] < 130.0,
            "prediction near high-value cluster should be high, got {}", preds[0]);
    }

    #[test]
    fn test_knn_uniform_backward_compat() {
        let (x, y, n, d) = clf_data();

        // Fit with explicit Uniform weights
        let mut model_uniform = KnnModel::new(3);
        model_uniform.weights = KnnWeights::Uniform;
        model_uniform.fit_clf(&x, n, d, &y).unwrap();

        // Fit with default (should be Uniform)
        let mut model_default = KnnModel::new(3);
        model_default.fit_clf(&x, n, d, &y).unwrap();

        // Both must produce identical predictions
        let queries = vec![0.5, 0.5, 10.5, 10.5, 5.0, 5.0];
        let preds_u = model_uniform.predict_clf(&queries, 3, d);
        let preds_d = model_default.predict_clf(&queries, 3, d);
        assert_eq!(preds_u, preds_d, "Uniform weights must match default behavior");

        // Also check probabilities
        let proba_u = model_uniform.predict_proba(&queries, 3, d);
        let proba_d = model_default.predict_proba(&queries, 3, d);
        for (pu, pd) in proba_u.iter().zip(proba_d.iter()) {
            assert!(
                (pu - pd).abs() < 1e-12,
                "uniform proba mismatch: {} vs {}", pu, pd
            );
        }

        // Regression backward compat
        let (xr, yr, nr, dr) = reg_data();
        let mut model_ru = KnnModel::new(3);
        model_ru.weights = KnnWeights::Uniform;
        model_ru.fit_reg(&xr, nr, dr, &yr).unwrap();
        let mut model_rd = KnnModel::new(3);
        model_rd.fit_reg(&xr, nr, dr, &yr).unwrap();
        let q = vec![0.5, 0.5, 10.5, 10.5];
        let pr_u = model_ru.predict_reg(&q, 2, dr);
        let pr_d = model_rd.predict_reg(&q, 2, dr);
        for (pu, pd) in pr_u.iter().zip(pr_d.iter()) {
            assert!(
                (pu - pd).abs() < 1e-12,
                "uniform regression mismatch: {} vs {}", pu, pd
            );
        }
    }

    #[test]
    fn test_knn_distance_proba_sums_to_one() {
        let (x, y, n, d) = clf_data();
        let mut model = KnnModel::new(3);
        model.weights = KnnWeights::Distance;
        model.fit_clf(&x, n, d, &y).unwrap();

        // Multiple queries
        let queries = vec![0.5, 0.5, 10.5, 10.5, 5.0, 5.0, 0.0, 0.0];
        let nq = 4;
        let nc = model.n_classes;
        let proba = model.predict_proba(&queries, nq, d);

        assert_eq!(proba.len(), nq * nc, "proba shape mismatch");
        for qi in 0..nq {
            let row = &proba[qi * nc..(qi + 1) * nc];
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "proba for query {} sums to {}, expected 1.0", qi, sum
            );
            for &p in row {
                assert!(p >= 0.0, "negative probability {} for query {}", p, qi);
            }
        }
    }

    #[test]
    fn test_knn_distance_serialization() {
        let (x, y, n, d) = clf_data();
        let mut model = KnnModel::new(3);
        model.weights = KnnWeights::Distance;
        model.fit_clf(&x, n, d, &y).unwrap();

        // Serialize → deserialize
        let json = model.to_json().unwrap();
        let restored = KnnModel::from_json(&json).unwrap();

        // weights field survives roundtrip
        assert_eq!(restored.weights, KnnWeights::Distance, "weights not preserved");
        assert_eq!(restored.k, 3);
        assert_eq!(restored.n_classes, 2);

        // Predictions match
        let q = vec![0.5, 0.5, 10.5, 10.5];
        let p1 = model.predict_clf(&q, 2, d);
        let p2 = restored.predict_clf(&q, 2, d);
        assert_eq!(p1, p2, "predictions differ after roundtrip");

        // Also test that deserializing old JSON (without weights field) defaults to Uniform
        let json_old = r#"{"k":3,"n_samples":6,"n_features":2,"data":[0.0,0.0,1.0,0.0,0.0,1.0,10.0,10.0,11.0,10.0,10.0,11.0],"labels_clf":[0,0,0,1,1,1],"targets_reg":[],"n_classes":2,"tree":null,"use_brute":false}"#;
        let old_model = KnnModel::from_json(json_old).unwrap();
        assert_eq!(old_model.weights, KnnWeights::Uniform,
            "missing weights field should default to Uniform");
    }
}

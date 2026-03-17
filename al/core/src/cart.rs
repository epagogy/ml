//! CART decision tree (classification + regression) — v2 performance rewrite.
//!
//! Port targeting sklearn semantics:
//! - Classification: Gini impurity split criterion
//! - Regression:     MSE split criterion
//! - Feature importance: Mean Decrease in Impurity (MDI), normalized to sum = 1.0
//!
//! # v2 performance internals
//! - `ColMajorMatrix`: contiguous column slices, unchecked indexing
//! - `TreeConfig` + `BuildContext`: config struct replaces 17-param build_tree, 4-param build_node
//! - In-place two-pointer partition: zero heap allocation during tree build
//! - Flat `proba_pool`: single contiguous allocation for all leaf probabilities
//! - `(f64, usize)` pair sort: pdqsort on contiguous data, no bounds checks
//!
//! # v1 contract (updated)
//! - `predict_clf()` returns 0-based class indices (NOT original labels).
//! - `predict_reg()` returns raw floating-point values.
//! - NaN in `x` is routed by learned direction at each split node (NaN-aware splitting).

use crate::error::MlError;
use crate::histogram::{ClfHistogram, QuantizedMatrix, RegHistogram, MAX_BINS};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Split criterion for CART decision trees and Random Forests.
///
/// Classification: `Gini` (default) or `Entropy`.
/// Regression: `MSE` (default) or `Poisson` (count/rate targets, requires y ≥ 0).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Criterion {
    /// Gini impurity — classification default.
    #[default]
    Gini,
    /// Shannon entropy (information gain) — classification alternative.
    Entropy,
    /// Mean squared error (variance) — regression default.
    MSE,
    /// Poisson deviance for count/rate targets — requires y ≥ 0.
    Poisson,
}

/// Node count threshold: use histogram splitting above this, sort-based below.
const HISTOGRAM_THRESHOLD: usize = 4096;

/// Precomputed per-feature histograms for a node.
///
/// Used by the histogram subtraction optimisation: after finding the best split for a
/// parent node, the smaller child's histograms are built by scanning its samples (O(n_small
/// × p)), and the larger child's histograms are computed as parent − smaller in O(bins × p).
/// This roughly halves histogram-fill work at every level of a deep tree.
///
/// Only active when `max_features` is `None` (all features evaluated per node, i.e. single
/// decision trees). For RF the random feature subset changes per node, so passing precomputed
/// histograms would either over-allocate (all features) or under-allocate (random subset).
enum NodeHists {
    Clf(Vec<ClfHistogram>),
    Reg(Vec<RegHistogram>),
}

// ---------------------------------------------------------------------------
// Column-major matrix (contiguous column slices, unchecked access)
// ---------------------------------------------------------------------------

pub struct ColMajorMatrix {
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
}

// SAFETY: ColMajorMatrix is read-only after construction — safe to share across threads.
unsafe impl Send for ColMajorMatrix {}
unsafe impl Sync for ColMajorMatrix {}

impl ColMajorMatrix {
    pub fn from_dmatrix(x: &DMatrix<f64>) -> Self {
        let nrows = x.nrows();
        let ncols = x.ncols();
        // nalgebra DMatrix is column-major — data layout is identical
        let data = x.as_slice().to_vec();
        Self { data, nrows, ncols }
    }

    /// Build from row-major slice (e.g. numpy C-contiguous) — single transpose, no DMatrix.
    /// Eliminates the double copy: numpy → DMatrix → ColMajorMatrix → just numpy → ColMajorMatrix.
    pub fn from_row_major_slice(rm: &[f64], nrows: usize, ncols: usize) -> Self {
        let mut data = vec![0.0_f64; nrows * ncols];
        // Transpose: rm[i*ncols + j] → data[j*nrows + i]
        // Process column-by-column for write-sequential cache behavior.
        for j in 0..ncols {
            let col_start = j * nrows;
            for i in 0..nrows {
                unsafe {
                    *data.get_unchecked_mut(col_start + i) = *rm.get_unchecked(i * ncols + j);
                }
            }
        }
        Self { data, nrows, ncols }
    }

    #[inline(always)]
    fn col_slice(&self, c: usize) -> &[f64] {
        let start = c * self.nrows;
        &self.data[start..start + self.nrows]
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, r: usize, c: usize) -> f64 {
        // SAFETY: caller guarantees r < nrows and c < ncols
        unsafe { *self.data.get_unchecked(c * self.nrows + r) }
    }
}

// ---------------------------------------------------------------------------
// Node representation
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum NodeKind {
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    Leaf {
        value: f64,
        proba_offset: usize, // index into proba_pool (clf) or 0 sentinel (reg)
    },
    Placeholder,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Node {
    pub(crate) kind: NodeKind,
    /// For Split nodes: if true, NaN feature values route to the left child;
    /// if false, they route to the right child. Defaults to true for backward
    /// compatibility with models serialized before NaN-aware splitting.
    #[serde(default = "default_true")]
    pub(crate) nan_goes_left: bool,
    /// Number of training samples that reached this node. Used by CCP pruning.
    /// Old .mlw files without this field deserialize as 0.
    #[serde(default)]
    pub(crate) n_samples: usize,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Tree config (collects all hyperparameters in one place)
// ---------------------------------------------------------------------------

/// Configuration for tree building. Collects all hyperparameters in one place.
#[derive(Debug, Clone)]
pub(crate) struct TreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub histogram_threshold: usize,
    pub n_classes: usize,
    pub is_clf: bool,
    pub max_features: Option<usize>,
    pub rng_seed: u64,
    pub criterion: Criterion,
    pub extra_trees: bool,
    pub monotone_cst: Option<Vec<i8>>,
    pub min_impurity_decrease: f64,
    /// L2 regularization on node weight sums for histogram MSE splits.
    /// Used by GBT Newton-gain: gain = G²/(H+λ) instead of G²/H.
    /// 0.0 for standard CART/RF (no regularization in split selection).
    pub split_lambda: f64,
}

// ---------------------------------------------------------------------------
// Build context (replaces 13-parameter recursion)
// ---------------------------------------------------------------------------

struct BuildContext<'a> {
    x: &'a ColMajorMatrix,
    y: &'a [f64],
    weights: Vec<f64>,
    nodes: Vec<Node>,
    proba_pool: Vec<f64>,
    importance_raw: Vec<f64>,
    pair_buf: Vec<(f64, usize)>,
    qm: Option<&'a QuantizedMatrix>,
    // Hyperparams (from TreeConfig)
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    histogram_threshold: usize,
    n_classes: usize,
    is_clf: bool,
    total_w: f64,
    /// Per-node random feature subsampling (None = all features).
    max_features: Option<usize>,
    /// xorshift64 state for random feature selection. Evolves across nodes.
    rng_state: u64,
    /// Split quality criterion.
    criterion: Criterion,
    /// Extra-Trees mode: random threshold per feature instead of optimal scan.
    extra_trees: bool,
    /// Monotone constraints per feature: +1 = increasing, -1 = decreasing, 0 = unconstrained.
    /// Regression only. None = no constraints.
    monotone_cst: Option<Vec<i8>>,
    /// Minimum impurity decrease required for a split. If the weighted impurity decrease
    /// (parent_impurity - weighted_child_impurity) is below this threshold, the node
    /// becomes a leaf. Default 0.0 (no threshold).
    min_impurity_decrease: f64,
    /// L2 regularization for histogram MSE splits (GBT Newton-gain).
    split_lambda: f64,
    /// True when all sample weights are 1.0 (the common case). Enables fast paths that
    /// skip weight lookups and use integer counts directly.
    uniform_weights: bool,
}

// ---------------------------------------------------------------------------
// Public model
// ---------------------------------------------------------------------------

/// CART decision tree for classification and regression.
#[derive(Serialize, Deserialize)]
pub struct DecisionTreeModel {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    /// Use histogram-based splitting when node has >= this many samples.
    /// Default 1024. Set to `usize::MAX` to force sort-based, `0` to force histogram.
    pub histogram_threshold: usize,
    /// Per-node random feature subsampling. None = all features (default for single tree).
    /// Set to Some(sqrt(p)) for classification RF, Some(p/3) for regression RF.
    pub max_features: Option<usize>,
    /// Seed for random feature selection. Only used when `max_features` is Some.
    pub rng_seed: u64,
    pub n_features: usize,
    pub n_classes: usize,
    pub feature_importances: Vec<f64>,
    nodes: Vec<Node>,
    proba_pool: Vec<f64>,
    is_clf: bool,
    /// Split criterion used during training (stored for introspection).
    /// Old .mlw files without this field deserialize as Gini — harmless, criterion is
    /// only read during training, never during prediction.
    #[serde(default)]
    pub criterion: Criterion,
    /// Extra-Trees mode: one random threshold per feature per node.
    /// Old .mlw files without this field deserialize as false (standard CART).
    #[serde(default)]
    pub extra_trees: bool,
    /// Monotone constraints: +1 increasing, -1 decreasing, 0 unconstrained.
    /// Length must equal n_features. Regression only; None = no constraints.
    /// Old .mlw files without this field deserialize as None.
    #[serde(default)]
    pub monotone_cst: Option<Vec<i8>>,
    /// Minimum impurity decrease required for a split. A node will become a leaf
    /// if the weighted impurity decrease is below this threshold.
    /// Default 0.0 (no pruning beyond standard stopping criteria).
    /// Old .mlw files without this field deserialize as 0.0.
    #[serde(default)]
    pub min_impurity_decrease: f64,
    /// Cost-complexity pruning parameter (Breiman et al. 1984, Chapter 3).
    /// The full tree is built first, then subtrees with effective alpha <= ccp_alpha
    /// are pruned (weakest-link pruning). Default 0.0 (no pruning).
    /// Old .mlw files without this field deserialize as 0.0.
    #[serde(default)]
    pub ccp_alpha: f64,
}

impl DecisionTreeModel {
    pub fn new(max_depth: usize, min_samples_split: usize, min_samples_leaf: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            histogram_threshold: HISTOGRAM_THRESHOLD,
            max_features: None,
            rng_seed: 0,
            n_features: 0,
            n_classes: 0,
            feature_importances: Vec::new(),
            nodes: Vec::new(),
            proba_pool: Vec::new(),
            is_clf: false,
            criterion: Criterion::default(),
            extra_trees: false,
            monotone_cst: None,
            min_impurity_decrease: 0.0,
            ccp_alpha: 0.0,
        }
    }

    /// Fit classification tree.
    pub fn fit_clf(
        &mut self,
        x: &DMatrix<f64>,
        y: &[i64],
        sample_weight: Option<&DVector<f64>>,
        criterion: Criterion,
    ) -> Result<&mut Self, MlError> {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }

        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let k = classes.len();
        if k < 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: k,
            });
        }

        let y_f64: Vec<f64> = y
            .iter()
            .map(|&lbl| classes.binary_search(&lbl).unwrap() as f64)
            .collect();

        let weights = normalize_weights(n, sample_weight)?;

        self.is_clf = true;
        self.n_features = p;
        self.n_classes = k;
        self.criterion = criterion;

        let cm = ColMajorMatrix::from_dmatrix(x);
        let qm = if n >= self.histogram_threshold {
            Some(QuantizedMatrix::from_col_data(&cm.data, n, p, MAX_BINS))
        } else {
            None
        };

        let config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            histogram_threshold: self.histogram_threshold,
            n_classes: k,
            is_clf: true,
            max_features: self.max_features,
            rng_seed: self.rng_seed,
            criterion,
            extra_trees: self.extra_trees,
            monotone_cst: None, // monotone_cst not applied to classification
            min_impurity_decrease: self.min_impurity_decrease,
            split_lambda: 0.0,
        };

        let (nodes, proba_pool, importance_raw) = build_tree(
            &cm,
            &y_f64,
            weights,
            qm.as_ref(),
            &config,
        );

        self.feature_importances = normalize_importances(importance_raw, p);
        self.nodes = nodes;
        self.proba_pool = proba_pool;

        // Post-hoc cost-complexity pruning (ccp_alpha > 0).
        if self.ccp_alpha > 0.0 {
            prune_ccp(&mut self.nodes, &mut self.proba_pool, self.ccp_alpha, k, true);
        }

        debug_assert!(
            !self
                .nodes
                .iter()
                .any(|n| matches!(n.kind, NodeKind::Placeholder)),
            "placeholder node detected after tree build"
        );
        Ok(self)
    }

    /// Fit regression tree.
    pub fn fit_reg(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        sample_weight: Option<&DVector<f64>>,
        criterion: Criterion,
    ) -> Result<&mut Self, MlError> {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }

        let y_f64: Vec<f64> = y.iter().copied().collect();
        let weights = normalize_weights(n, sample_weight)?;

        self.is_clf = false;
        self.n_features = p;
        self.n_classes = 0;
        self.criterion = criterion;

        let cm = ColMajorMatrix::from_dmatrix(x);
        let qm = if n >= self.histogram_threshold {
            Some(QuantizedMatrix::from_col_data(&cm.data, n, p, MAX_BINS))
        } else {
            None
        };

        let config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            histogram_threshold: self.histogram_threshold,
            n_classes: 0,
            is_clf: false,
            max_features: self.max_features,
            rng_seed: self.rng_seed,
            criterion,
            extra_trees: self.extra_trees,
            monotone_cst: self.monotone_cst.clone(),
            min_impurity_decrease: self.min_impurity_decrease,
            split_lambda: 0.0,
        };

        let (nodes, proba_pool, importance_raw) = build_tree(
            &cm,
            &y_f64,
            weights,
            qm.as_ref(),
            &config,
        );

        self.feature_importances = normalize_importances(importance_raw, p);
        self.nodes = nodes;
        self.proba_pool = proba_pool;

        // Post-hoc cost-complexity pruning (ccp_alpha > 0).
        if self.ccp_alpha > 0.0 {
            prune_ccp(&mut self.nodes, &mut self.proba_pool, self.ccp_alpha, 0, false);
        }

        debug_assert!(
            !self
                .nodes
                .iter()
                .any(|n| matches!(n.kind, NodeKind::Placeholder)),
            "placeholder node detected after tree build"
        );
        Ok(self)
    }

    /// Predict class indices (0-based) for each row of `x`.
    pub fn predict_clf(&self, x: &DMatrix<f64>) -> Vec<i64> {
        assert!(
            self.is_clf,
            "predict_clf called on unfitted or regression model"
        );
        let cm = ColMajorMatrix::from_dmatrix(x);
        (0..x.nrows())
            .into_par_iter()
            .map(|i| traverse_cm(&self.nodes, &cm, i) as i64)
            .collect()
    }

    /// Predict continuous values for each row of `x`.
    pub fn predict_reg(&self, x: &DMatrix<f64>) -> DVector<f64> {
        assert!(!self.is_clf, "predict_reg called on classification model");
        let cm = ColMajorMatrix::from_dmatrix(x);
        let vals: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| traverse_cm(&self.nodes, &cm, i))
            .collect();
        DVector::from_vec(vals)
    }

    /// Class probability matrix, shape `(n_samples, n_classes)`.
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        assert!(
            self.is_clf,
            "predict_proba called on unfitted or regression model"
        );
        let k = self.n_classes;
        let n = x.nrows();
        let cm = ColMajorMatrix::from_dmatrix(x);
        // Parallel: each sample produces a row of k probabilities.
        let flat: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let leaf = leaf_index_cm(&self.nodes, &cm, i);
                match &self.nodes[leaf].kind {
                    NodeKind::Leaf { proba_offset, .. } => {
                        (0..k).map(move |j| self.proba_pool[proba_offset + j])
                    }
                    _ => unreachable!(),
                }
            })
            .collect();
        DMatrix::from_row_slice(n, k, &flat)
    }

    /// Serialize the fitted tree to JSON for Python `__getstate__`.
    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    /// Deserialize a tree from JSON for Python `__setstate__`.
    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

impl Default for DecisionTreeModel {
    fn default() -> Self {
        Self::new(500, 2, 1)
    }
}

// ---------------------------------------------------------------------------
// Tree traversal
// ---------------------------------------------------------------------------

/// Traverse tree using ColMajorMatrix (unchecked access). Returns leaf value.
/// NaN feature values are routed by the learned `nan_goes_left` direction.
#[inline]
fn traverse_cm(nodes: &[Node], x: &ColMajorMatrix, row: usize) -> f64 {
    let mut idx = 0;
    loop {
        match &nodes[idx].kind {
            NodeKind::Leaf { value, .. } => return *value,
            NodeKind::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                // SAFETY: row < x.nrows and feature < x.ncols guaranteed by caller
                let val = unsafe { x.get_unchecked(row, *feature) };
                idx = if val.is_nan() {
                    if nodes[idx].nan_goes_left { *left } else { *right }
                } else if val <= *threshold {
                    *left
                } else {
                    *right
                };
            }
            NodeKind::Placeholder => {
                panic!("placeholder node during prediction")
            }
        }
    }
}

/// Traverse tree to find leaf node index.
/// NaN feature values are routed by the learned `nan_goes_left` direction.
#[inline]
fn leaf_index_cm(nodes: &[Node], x: &ColMajorMatrix, row: usize) -> usize {
    let mut idx = 0;
    loop {
        match &nodes[idx].kind {
            NodeKind::Leaf { .. } => return idx,
            NodeKind::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let val = unsafe { x.get_unchecked(row, *feature) };
                idx = if val.is_nan() {
                    if nodes[idx].nan_goes_left { *left } else { *right }
                } else if val <= *threshold {
                    *left
                } else {
                    *right
                };
            }
            NodeKind::Placeholder => {
                panic!("placeholder node during prediction")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Histogram subtraction helpers
// ---------------------------------------------------------------------------

/// Build per-feature histograms for ALL p features over the given sample indices.
/// Returns a `NodeHists` that can be passed to the children or used for subtraction.
fn build_all_hists(
    qm: &QuantizedMatrix,
    y: &[f64],
    weights: &[f64],
    indices: &[usize],
    n_classes: usize,
    is_clf: bool,
) -> NodeHists {
    if is_clf {
        let hists: Vec<ClfHistogram> = (0..qm.ncols)
            .map(|feat| {
                let nb = qm.n_bins_for(feat);
                let mut h = ClfHistogram::new(nb, n_classes);
                for &i in indices {
                    // SAFETY: i < qm.nrows and feat < qm.ncols — validated at tree entry
                    let bin = unsafe { qm.get_unchecked(i, feat) };
                    h.add(bin, y[i] as usize, weights[i]);
                }
                h
            })
            .collect();
        NodeHists::Clf(hists)
    } else {
        let hists: Vec<RegHistogram> = (0..qm.ncols)
            .map(|feat| {
                let nb = qm.n_bins_for(feat);
                let mut h = RegHistogram::new(nb);
                for &i in indices {
                    let bin = unsafe { qm.get_unchecked(i, feat) };
                    h.add(bin, y[i], weights[i]);
                }
                h
            })
            .collect();
        NodeHists::Reg(hists)
    }
}

/// Compute right-child histograms = parent − left-child (O(bins × p)).
fn subtract_hists(parent: &NodeHists, left: &NodeHists) -> NodeHists {
    match (parent, left) {
        (NodeHists::Clf(p), NodeHists::Clf(l)) => NodeHists::Clf(
            p.iter()
                .zip(l.iter())
                .map(|(ph, lh)| ClfHistogram::subtract(ph, lh))
                .collect(),
        ),
        (NodeHists::Reg(p), NodeHists::Reg(l)) => NodeHists::Reg(
            p.iter()
                .zip(l.iter())
                .map(|(ph, lh)| RegHistogram::subtract(ph, lh))
                .collect(),
        ),
        _ => panic!("subtract_hists: mismatched histogram types"),
    }
}

/// Find the best split from precomputed per-feature histograms (evaluates ALL features).
/// Only used on the `max_features = None` path.
fn best_split_from_hists(
    hists: &NodeHists,
    qm: &QuantizedMatrix,
    node_w: f64,
    min_samples_leaf: usize,
    criterion: Criterion,
    split_lambda: f64,
) -> Option<SplitResult> {
    let p = qm.ncols;
    match hists {
        NodeHists::Clf(clf_hists) => {
            let mut best_feat = 0_usize;
            let mut best_bin = 0_u8;
            let mut best_cost = f64::INFINITY; // unified lower = better
            let mut best_nan_left = true;
            let mut found = false;
            for feat in 0..p {
                // Convert gini proxy to cost (node_w - proxy) for unified comparison
                let result = match criterion {
                    Criterion::Entropy => clf_hists[feat].best_entropy_split(min_samples_leaf),
                    _ => clf_hists[feat]
                        .best_gini_split(min_samples_leaf)
                        .map(|(bin, proxy, nan_left)| (bin, node_w - proxy, nan_left)),
                };
                if let Some((bin, cost, nan_left)) = result {
                    if cost < best_cost {
                        best_cost = cost;
                        best_feat = feat;
                        best_bin = bin;
                        best_nan_left = nan_left;
                        found = true;
                    }
                }
            }
            if found {
                Some(SplitResult {
                    feature: best_feat,
                    threshold: qm.threshold_for(best_feat, best_bin),
                    child_impurity_w: best_cost,
                    nan_goes_left: best_nan_left,
                    left_impurity: None,
                    right_impurity: None,
                })
            } else {
                None
            }
        }
        NodeHists::Reg(reg_hists) => {
            let mut best_feat = 0_usize;
            let mut best_bin = 0_u8;
            let mut best_child_iw = f64::INFINITY;
            let mut best_nan_left = true;
            let mut found = false;
            for feat in 0..p {
                let result = match criterion {
                    Criterion::Poisson => reg_hists[feat].best_poisson_split(min_samples_leaf),
                    _ => reg_hists[feat].best_mse_split(min_samples_leaf, split_lambda),
                };
                if let Some((bin, child_iw, nan_left)) = result {
                    if child_iw < best_child_iw {
                        best_child_iw = child_iw;
                        best_feat = feat;
                        best_bin = bin;
                        best_nan_left = nan_left;
                        found = true;
                    }
                }
            }
            if found {
                Some(SplitResult {
                    feature: best_feat,
                    threshold: qm.threshold_for(best_feat, best_bin),
                    child_impurity_w: best_child_iw,
                    nan_goes_left: best_nan_left,
                    left_impurity: None,
                    right_impurity: None,
                })
            } else {
                None
            }
        }
    }
}

/// Given a parent's histograms, compute child histograms via subtraction.
///
/// Scans the SMALLER child (O(n_small × p)), subtracts from parent to get the LARGER
/// child's histograms (O(bins × p)). Returns `None` for a child that is below the
/// histogram threshold (it will use sort-based splitting, so precomputed hists are useless).
fn compute_child_hists_via_subtraction(
    parent: NodeHists,
    qm: &QuantizedMatrix,
    y: &[f64],
    weights: &[f64],
    n_classes: usize,
    is_clf: bool,
    histogram_threshold: usize,
    left_indices: &[usize],
    right_indices: &[usize],
) -> (Option<NodeHists>, Option<NodeHists>) {
    let left_large = left_indices.len() >= histogram_threshold;
    let right_large = right_indices.len() >= histogram_threshold;

    // If the larger child is below threshold it will use sort-based splitting anyway.
    let left_is_smaller = left_indices.len() <= right_indices.len();
    let larger_len = if left_is_smaller { right_indices.len() } else { left_indices.len() };
    if larger_len < histogram_threshold {
        return (None, None);
    }

    let (smaller_idx, larger_is_right) = if left_is_smaller {
        (left_indices, true)
    } else {
        (right_indices, false)
    };

    let smaller_hists = build_all_hists(qm, y, weights, smaller_idx, n_classes, is_clf);
    let larger_hists = subtract_hists(&parent, &smaller_hists);

    if larger_is_right {
        // smaller = left, larger = right
        (
            if left_large { Some(smaller_hists) } else { None },
            Some(larger_hists),
        )
    } else {
        // smaller = right, larger = left
        (
            Some(larger_hists),
            if right_large { Some(smaller_hists) } else { None },
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone tree builder (shared-data path for forest)
// ---------------------------------------------------------------------------

/// Build a tree from borrowed data. Returns (nodes, proba_pool, importance_raw).
/// Used by both `DecisionTreeModel::fit_*` and `RandomForestModel` (via `pub(crate)`).
pub(crate) fn build_tree(
    x: &ColMajorMatrix,
    y: &[f64],
    weights: Vec<f64>,
    qm: Option<&QuantizedMatrix>,
    config: &TreeConfig,
) -> (Vec<Node>, Vec<f64>, Vec<f64>) {
    let n = y.len();
    let p = x.ncols;
    let total_w: f64 = weights.iter().sum();

    let uniform_weights = weights.iter().all(|&w| w == 1.0);
    let mut ctx = BuildContext {
        x,
        y,
        weights,
        nodes: Vec::with_capacity(64),
        proba_pool: Vec::new(),
        importance_raw: vec![0.0; p],
        pair_buf: Vec::with_capacity(n),
        qm,
        max_depth: config.max_depth,
        min_samples_split: config.min_samples_split,
        min_samples_leaf: config.min_samples_leaf,
        histogram_threshold: config.histogram_threshold,
        n_classes: config.n_classes,
        is_clf: config.is_clf,
        total_w,
        max_features: config.max_features,
        rng_state: if config.rng_seed == 0 { 1 } else { config.rng_seed },
        criterion: config.criterion,
        extra_trees: config.extra_trees,
        monotone_cst: config.monotone_cst.clone(),
        min_impurity_decrease: config.min_impurity_decrease,
        split_lambda: config.split_lambda,
        uniform_weights,
    };

    let mut indices: Vec<usize> = (0..n).collect();
    let root_impurity = compute_impurity(&ctx, &indices);
    build_node(&mut ctx, &mut indices, 0, root_impurity, None, (f64::NEG_INFINITY, f64::INFINITY));

    (ctx.nodes, ctx.proba_pool, ctx.importance_raw)
}

// ---------------------------------------------------------------------------
// Recursive tree builder
// ---------------------------------------------------------------------------

/// Recursive tree builder. Returns index of newly created node.
/// `indices` is a mutable slice — partitioned in-place for children.
/// `precomputed` carries histograms built by the parent via subtraction (None at root).
/// `bounds` is the (lb, ub) value range for leaves in this subtree (monotone constraints).
fn build_node(
    ctx: &mut BuildContext,
    indices: &mut [usize],
    depth: usize,
    node_impurity: f64,
    precomputed: Option<NodeHists>,
    bounds: (f64, f64),
) -> usize {
    let idx = ctx.nodes.len();
    let n = indices.len();
    ctx.nodes.push(Node {
        kind: NodeKind::Placeholder,
        nan_goes_left: true,
        n_samples: n,
    });
    let n_w: f64 = indices.iter().map(|&i| ctx.weights[i]).sum();

    // Leaf conditions
    let is_leaf = depth >= ctx.max_depth || n < ctx.min_samples_split || node_impurity < 1e-15;

    if !is_leaf {
        // Histogram subtraction path: only when max_features=None (all features per node).
        // RF uses random subsets per node, so we cannot safely reuse parent histograms.
        let mut parent_hists: Option<NodeHists> = None;

        let split_opt = if ctx.extra_trees {
            // Extra Trees: one random threshold per feature, no histogram scan.
            best_split_extra_trees(ctx, indices, n_w)
        } else if indices.len() >= ctx.histogram_threshold && ctx.qm.is_some() {
            let qm = ctx.qm.unwrap();
            if ctx.max_features.is_none() {
                // Build or reuse parent histograms, find best split from them.
                let hists = precomputed.unwrap_or_else(|| {
                    build_all_hists(qm, ctx.y, &ctx.weights, indices, ctx.n_classes, ctx.is_clf)
                });
                let split = best_split_from_hists(&hists, qm, n_w, ctx.min_samples_leaf, ctx.criterion, ctx.split_lambda);
                parent_hists = Some(hists);
                split
            } else {
                // RF path: random feature subset per node — standard histogram function.
                best_split_histogram(
                    qm, ctx.y, &ctx.weights, indices, n_w,
                    ctx.n_classes, ctx.min_samples_leaf, ctx.is_clf,
                    ctx.max_features, &mut ctx.rng_state, ctx.criterion,
                    ctx.split_lambda,
                )
            }
        } else {
            best_split(ctx, indices, n_w)
        };

        if let Some(split) = split_opt {
            let gain = (n_w / ctx.total_w) * (node_impurity - split.child_impurity_w / n_w);

            // min_impurity_decrease check: if the weighted impurity decrease is below
            // the threshold, skip this split and make a leaf instead.
            if gain < ctx.min_impurity_decrease {
                ctx.nodes[idx] = Node {
                    kind: make_leaf(ctx, indices, n_w, bounds),
                    nan_goes_left: true,
                    n_samples: n,
                };
                return idx;
            }

            ctx.importance_raw[split.feature] += gain;

            // Partition indices in-place (NaN samples routed by learned direction)
            let sp = partition(indices, &ctx.x, split.feature, split.threshold, split.nan_goes_left);

            // If partition is degenerate (all left or all right), make a leaf
            if sp == 0 || sp == n {
                ctx.nodes[idx] = Node {
                    kind: make_leaf(ctx, indices, n_w, bounds),
                    nan_goes_left: true,
                    n_samples: n,
                };
                return idx;
            }

            // Split indices into two non-overlapping mutable slices
            let (left_indices, right_indices) = indices.split_at_mut(sp);

            // Compute child histograms via subtraction (only when we have parent hists).
            // Scans the smaller child; subtracts for the larger. O(n_small × p + bins × p).
            let (left_child_hists, right_child_hists) = if let Some(ph) = parent_hists {
                compute_child_hists_via_subtraction(
                    ph,
                    ctx.qm.unwrap(),
                    ctx.y,
                    &ctx.weights,
                    ctx.n_classes,
                    ctx.is_clf,
                    ctx.histogram_threshold,
                    left_indices,
                    right_indices,
                )
            } else {
                (None, None)
            };

            // Propagate monotone bounds to children (regression only).
            let (left_bounds, right_bounds) = compute_child_bounds(
                ctx, left_indices, right_indices, split.feature, bounds,
            );

            // Use pre-computed child impurities when available (sort-based path),
            // fall back to scanning indices (histogram path).
            let left_impurity = split.left_impurity
                .unwrap_or_else(|| compute_impurity(ctx, left_indices));
            let right_impurity = split.right_impurity
                .unwrap_or_else(|| compute_impurity(ctx, right_indices));

            let left = build_node(ctx, left_indices, depth + 1, left_impurity, left_child_hists, left_bounds);
            let right = build_node(ctx, right_indices, depth + 1, right_impurity, right_child_hists, right_bounds);

            ctx.nodes[idx] = Node {
                kind: NodeKind::Split {
                    feature: split.feature,
                    threshold: split.threshold,
                    left,
                    right,
                },
                nan_goes_left: split.nan_goes_left,
                n_samples: n,
            };
            return idx;
        }
    }

    // Build leaf
    ctx.nodes[idx] = Node {
        kind: make_leaf(ctx, indices, n_w, bounds),
        nan_goes_left: true,
        n_samples: n,
    };
    idx
}

/// Compute impurity for a subset of indices.
fn compute_impurity(ctx: &BuildContext, indices: &[usize]) -> f64 {
    let n = indices.len();
    let n_w = if ctx.uniform_weights { n as f64 } else {
        indices.iter().map(|&i| ctx.weights[i]).sum()
    };
    match ctx.criterion {
        Criterion::Gini => {
            let k = ctx.n_classes;
            if ctx.uniform_weights {
                let mut counts = vec![0_u32; k];
                for &i in indices { counts[ctx.y[i] as usize] += 1; }
                let sq: u64 = counts.iter().map(|&c| (c as u64) * (c as u64)).sum();
                1.0 - sq as f64 / (n_w * n_w)
            } else {
                let mut counts = vec![0.0_f64; k];
                for &i in indices { counts[ctx.y[i] as usize] += ctx.weights[i]; }
                gini_impurity_from_counts(&counts, n_w)
            }
        }
        Criterion::Entropy => {
            let k = ctx.n_classes;
            let mut counts = vec![0.0_f64; k];
            for &i in indices {
                let w = if ctx.uniform_weights { 1.0 } else { ctx.weights[i] };
                counts[ctx.y[i] as usize] += w;
            }
            entropy_impurity_from_counts(&counts, n_w)
        }
        Criterion::MSE | Criterion::Poisson => {
            let mut sy = 0.0_f64;
            let mut sy2 = 0.0_f64;
            if ctx.uniform_weights {
                for &i in indices {
                    let v = ctx.y[i];
                    sy += v;
                    sy2 += v * v;
                }
            } else {
                for &i in indices {
                    let w = ctx.weights[i];
                    let v = ctx.y[i];
                    sy += w * v;
                    sy2 += w * v * v;
                }
            }
            mse_impurity_from_stats(sy, sy2, n_w)
        }
    }
}

// ---------------------------------------------------------------------------
// Best split search
// ---------------------------------------------------------------------------

struct SplitResult {
    feature: usize,
    threshold: f64,
    child_impurity_w: f64,
    /// Learned NaN routing direction for this split.
    nan_goes_left: bool,
    /// Pre-computed child impurities (avoids redundant O(n) scan after partition).
    /// `None` means caller must compute via `compute_impurity` (histogram path).
    left_impurity: Option<f64>,
    right_impurity: Option<f64>,
}

/// Inline xorshift64 step. Advances state and returns next random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Select random feature indices when max_features is set.
/// Returns a Vec of feature indices to evaluate.
fn select_features(p: usize, max_features: Option<usize>, rng_state: &mut u64) -> Vec<usize> {
    match max_features {
        None => (0..p).collect(),
        Some(m) if m >= p => (0..p).collect(),
        Some(m) => {
            // Fisher-Yates partial shuffle
            let mut indices: Vec<usize> = (0..p).collect();
            for i in 0..m {
                let j = i + (xorshift64(rng_state) % (p - i) as u64) as usize;
                indices.swap(i, j);
            }
            indices.truncate(m);
            indices
        }
    }
}

/// Find the best split across all features. Uses (value, index) pair sort
/// on contiguous column slices — zero bounds checks in the hot loop.
///
/// Classification uses proxy Gini: maximize `sq_l/n_left_w + sq_r/n_right_w`
/// instead of minimizing full `n_left_w * gini_l + n_right_w * gini_r`.
/// Equivalent for ranking (constant `node_w` cancels), saves 4 FP ops/candidate.
///
/// NaN values sort to the end via `total_cmp`. The NaN block is separated before
/// the scan; for each candidate split, NaN is tried in both children and the
/// direction with better impurity wins.
fn best_split(ctx: &mut BuildContext, indices: &[usize], node_w: f64) -> Option<SplitResult> {
    let p = ctx.x.ncols;
    let n = indices.len();

    // Unified lower = better for all criteria.
    // For gini we compute node_w - proxy directly (= actual weighted child gini).
    let mut best_cost = f64::INFINITY;
    let mut best_feat = 0;
    let mut best_thresh = 0.0_f64;
    let mut best_nan_left = true;
    let mut found = false;
    let mut best_left_imp = 0.0_f64;
    let mut best_right_imp = 0.0_f64;

    let feats_to_eval = select_features(p, ctx.max_features, &mut ctx.rng_state);

    for &feat in &feats_to_eval {
        // Build (value, index) pairs + track min/max for uniform feature skip
        let col = ctx.x.col_slice(feat);
        ctx.pair_buf.clear();
        ctx.pair_buf.reserve(n);
        let mut fmin = f64::INFINITY;
        let mut fmax = f64::NEG_INFINITY;
        for &i in indices {
            // SAFETY: i < nrows validated at tree entry
            let v = unsafe { *col.get_unchecked(i) };
            // Track min/max of non-NaN values only
            if !v.is_nan() {
                if v < fmin {
                    fmin = v;
                }
                if v > fmax {
                    fmax = v;
                }
            }
            ctx.pair_buf.push((v, i));
        }
        // Uniform feature skip: no valid split if all non-NaN values identical or all NaN.
        // fmin > fmax when all values are NaN (initial INFINITY > NEG_INFINITY).
        if fmin >= fmax {
            continue;
        }
        // pdqsort on contiguous (f64, usize) pairs. NaN sorts to the end via total_cmp.
        ctx.pair_buf.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        // Separate NaN block at the end. n_non_nan is the count of non-NaN values.
        let n_non_nan = ctx.pair_buf.iter().take_while(|(v, _)| !v.is_nan()).count();
        let has_nan = n_non_nan < ctx.pair_buf.len();

        if ctx.is_clf {
            let k = ctx.n_classes;

            // ── Fast path: uniform weights + no NaN + Gini (covers ~90% of splits) ──
            if ctx.uniform_weights && !has_nan && ctx.criterion == Criterion::Gini {
                // Integer counts — no weight loads, no f64 tracking
                let mut left_counts = vec![0_u32; k];
                let mut right_counts = vec![0_u32; k];
                for pos in 0..n_non_nan {
                    let (_, i) = ctx.pair_buf[pos];
                    right_counts[ctx.y[i] as usize] += 1;
                }
                let mut count_left = 0_usize;
                let mut count_right = n_non_nan;

                for pos in 0..(n_non_nan.saturating_sub(1)) {
                    let (val, i) = ctx.pair_buf[pos];
                    let cls = ctx.y[i] as usize;
                    left_counts[cls] += 1;
                    right_counts[cls] -= 1;
                    count_left += 1;
                    count_right -= 1;

                    let next_val = ctx.pair_buf[pos + 1].0;
                    if val == next_val { continue; }
                    if count_left < ctx.min_samples_leaf || count_right < ctx.min_samples_leaf {
                        continue;
                    }

                    // Gini proxy: node_w - sum(left^2)/n_left - sum(right^2)/n_right
                    let sq_l: u64 = left_counts.iter().map(|&c| (c as u64) * (c as u64)).sum();
                    let sq_r: u64 = right_counts.iter().map(|&c| (c as u64) * (c as u64)).sum();
                    let n_l = count_left as f64;
                    let n_r = count_right as f64;
                    let cost = node_w - sq_l as f64 / n_l - sq_r as f64 / n_r;
                    let li = 1.0 - sq_l as f64 / (n_l * n_l);
                    let ri = 1.0 - sq_r as f64 / (n_r * n_r);

                    if cost < best_cost {
                        best_cost = cost;
                        best_feat = feat;
                        best_thresh = (val + next_val) / 2.0;
                        best_nan_left = true;
                        best_left_imp = li;
                        best_right_imp = ri;
                        found = true;
                    }
                }
                continue; // next feature
            }

            // ── General path: weighted / NaN / Entropy ──

            // Accumulate NaN stats
            let mut nan_counts = vec![0.0_f64; k];
            let mut nan_w = 0.0_f64;
            let mut nan_n = 0_usize;
            if has_nan {
                for pos in n_non_nan..ctx.pair_buf.len() {
                    let (_, i) = ctx.pair_buf[pos];
                    let cls = ctx.y[i] as usize;
                    let w = ctx.weights[i];
                    nan_counts[cls] += w;
                    nan_w += w;
                    nan_n += 1;
                }
            }

            // Non-NaN node weight (for right-side init)
            let non_nan_w: f64 = node_w - nan_w;

            let mut left_counts = vec![0.0_f64; k];
            let mut right_counts = vec![0.0_f64; k];
            for pos in 0..n_non_nan {
                let (_, i) = ctx.pair_buf[pos];
                right_counts[ctx.y[i] as usize] += ctx.weights[i];
            }

            let mut n_left_w = 0.0_f64;
            let mut n_right_w = non_nan_w;
            let mut count_left = 0_usize;
            let mut count_right = n_non_nan;

            for pos in 0..(n_non_nan.saturating_sub(1)) {
                let (val, i) = ctx.pair_buf[pos];
                let cls = ctx.y[i] as usize;
                let w = ctx.weights[i];

                left_counts[cls] += w;
                right_counts[cls] -= w;
                n_left_w += w;
                n_right_w -= w;
                count_left += 1;
                count_right -= 1;

                let next_val = ctx.pair_buf[pos + 1].0;
                if val == next_val {
                    continue;
                }

                if !has_nan {
                    // Original path: no NaN
                    if count_left < ctx.min_samples_leaf || count_right < ctx.min_samples_leaf {
                        continue;
                    }
                    if n_left_w < 1e-15 || n_right_w < 1e-15 {
                        continue;
                    }
                    let (cost, li, ri) = match ctx.criterion {
                        Criterion::Entropy => {
                            let li = entropy_impurity_from_counts(&left_counts, n_left_w);
                            let ri = entropy_impurity_from_counts(&right_counts, n_right_w);
                            (n_left_w * li + n_right_w * ri, li, ri)
                        }
                        _ => {
                            let sq_l: f64 = left_counts.iter().map(|&c| c * c).sum();
                            let sq_r: f64 = right_counts.iter().map(|&c| c * c).sum();
                            let li = 1.0 - sq_l / (n_left_w * n_left_w);
                            let ri = 1.0 - sq_r / (n_right_w * n_right_w);
                            (node_w - sq_l / n_left_w - sq_r / n_right_w, li, ri)
                        }
                    };
                    if cost < best_cost {
                        best_cost = cost;
                        best_feat = feat;
                        best_thresh = (val + next_val) / 2.0;
                        best_nan_left = true;
                        best_left_imp = li;
                        best_right_imp = ri;
                        found = true;
                    }
                } else {
                    // Try NaN → left
                    let cl = count_left + nan_n;
                    let cr = count_right;
                    let nl_w = n_left_w + nan_w;
                    let nr_w = n_right_w;
                    if cl >= ctx.min_samples_leaf && cr >= ctx.min_samples_leaf
                        && nl_w > 1e-15 && nr_w > 1e-15
                    {
                        let (cost, li, ri) = match ctx.criterion {
                            Criterion::Entropy => {
                                let lc: Vec<f64> = (0..k).map(|c| left_counts[c] + nan_counts[c]).collect();
                                let li = entropy_impurity_from_counts(&lc, nl_w);
                                let ri = entropy_impurity_from_counts(&right_counts, nr_w);
                                (nl_w * li + nr_w * ri, li, ri)
                            }
                            _ => {
                                let sq_l: f64 = (0..k).map(|c| {
                                    let v = left_counts[c] + nan_counts[c];
                                    v * v
                                }).sum();
                                let sq_r: f64 = right_counts.iter().map(|&c| c * c).sum();
                                let li = 1.0 - sq_l / (nl_w * nl_w);
                                let ri = 1.0 - sq_r / (nr_w * nr_w);
                                (node_w - sq_l / nl_w - sq_r / nr_w, li, ri)
                            }
                        };
                        if cost < best_cost {
                            best_cost = cost;
                            best_feat = feat;
                            best_thresh = (val + next_val) / 2.0;
                            best_nan_left = true;
                            best_left_imp = li;
                            best_right_imp = ri;
                            found = true;
                        }
                    }

                    // Try NaN → right
                    let cl2 = count_left;
                    let cr2 = count_right + nan_n;
                    let nl_w2 = n_left_w;
                    let nr_w2 = n_right_w + nan_w;
                    if cl2 >= ctx.min_samples_leaf && cr2 >= ctx.min_samples_leaf
                        && nl_w2 > 1e-15 && nr_w2 > 1e-15
                    {
                        let (cost, li, ri) = match ctx.criterion {
                            Criterion::Entropy => {
                                let rc: Vec<f64> = (0..k).map(|c| right_counts[c] + nan_counts[c]).collect();
                                let li = entropy_impurity_from_counts(&left_counts, nl_w2);
                                let ri = entropy_impurity_from_counts(&rc, nr_w2);
                                (nl_w2 * li + nr_w2 * ri, li, ri)
                            }
                            _ => {
                                let sq_l: f64 = left_counts.iter().map(|&c| c * c).sum();
                                let sq_r: f64 = (0..k).map(|c| {
                                    let v = right_counts[c] + nan_counts[c];
                                    v * v
                                }).sum();
                                let li = 1.0 - sq_l / (nl_w2 * nl_w2);
                                let ri = 1.0 - sq_r / (nr_w2 * nr_w2);
                                (node_w - sq_l / nl_w2 - sq_r / nr_w2, li, ri)
                            }
                        };
                        if cost < best_cost {
                            best_cost = cost;
                            best_feat = feat;
                            best_thresh = (val + next_val) / 2.0;
                            best_nan_left = false;
                            best_left_imp = li;
                            best_right_imp = ri;
                            found = true;
                        }
                    }
                }
            }
        } else {
            // ── Fast path: uniform weights + no NaN + MSE (covers ~90% of regression splits) ──
            if ctx.uniform_weights && !has_nan && ctx.criterion == Criterion::MSE {
                let mut total_sy = 0.0_f64;
                let mut total_sy2 = 0.0_f64;
                for pos in 0..n_non_nan {
                    let (_, i) = ctx.pair_buf[pos];
                    let v = ctx.y[i];
                    total_sy += v;
                    total_sy2 += v * v;
                }
                let mut l_sy = 0.0_f64;
                let mut l_sy2 = 0.0_f64;
                let mut r_sy = total_sy;
                let mut r_sy2 = total_sy2;
                let mut count_left = 0_usize;
                let mut count_right = n_non_nan;

                for pos in 0..(n_non_nan.saturating_sub(1)) {
                    let (val, i) = ctx.pair_buf[pos];
                    let v = ctx.y[i];
                    l_sy += v;
                    l_sy2 += v * v;
                    r_sy -= v;
                    r_sy2 -= v * v;
                    count_left += 1;
                    count_right -= 1;

                    let next_val = ctx.pair_buf[pos + 1].0;
                    if val == next_val { continue; }
                    if count_left < ctx.min_samples_leaf || count_right < ctx.min_samples_leaf {
                        continue;
                    }
                    let n_l = count_left as f64;
                    let n_r = count_right as f64;
                    let lam = ctx.split_lambda;
                    let ld = n_l + lam;
                    let rd = n_r + lam;
                    let mse_l = (l_sy2 / ld - (l_sy / ld).powi(2)).max(0.0);
                    let mse_r = (r_sy2 / rd - (r_sy / rd).powi(2)).max(0.0);
                    let cost = ld * mse_l + rd * mse_r;
                    if cost < best_cost {
                        best_cost = cost;
                        best_feat = feat;
                        best_thresh = (val + next_val) / 2.0;
                        best_nan_left = true;
                        best_left_imp = mse_l;
                        best_right_imp = mse_r;
                        found = true;
                    }
                }
                continue; // next feature
            }

            // ── General path: weighted / NaN / Poisson ──

            // Regression: compute totals for non-NaN values
            let mut total_swy = 0.0_f64;
            let mut total_swy2 = 0.0_f64;
            let mut non_nan_w = 0.0_f64;
            for pos in 0..n_non_nan {
                let (_, i) = ctx.pair_buf[pos];
                let w = ctx.weights[i];
                let v = ctx.y[i];
                total_swy += w * v;
                total_swy2 += w * v * v;
                non_nan_w += w;
            }

            // NaN stats
            let mut nan_sw = 0.0_f64;
            let mut nan_swy = 0.0_f64;
            let mut nan_swy2 = 0.0_f64;
            let mut nan_n = 0_usize;
            if has_nan {
                for pos in n_non_nan..ctx.pair_buf.len() {
                    let (_, i) = ctx.pair_buf[pos];
                    let w = ctx.weights[i];
                    let v = ctx.y[i];
                    nan_sw += w;
                    nan_swy += w * v;
                    nan_swy2 += w * v * v;
                    nan_n += 1;
                }
            }

            let mut l_sw = 0.0_f64;
            let mut l_swy = 0.0_f64;
            let mut l_swy2 = 0.0_f64;
            let mut r_sw = non_nan_w;
            let mut r_swy = total_swy;
            let mut r_swy2 = total_swy2;
            let mut count_left = 0_usize;
            let mut count_right = n_non_nan;

            for pos in 0..(n_non_nan.saturating_sub(1)) {
                let (val, i) = ctx.pair_buf[pos];
                let w = ctx.weights[i];
                let v = ctx.y[i];

                l_sw += w;
                l_swy += w * v;
                l_swy2 += w * v * v;
                r_sw -= w;
                r_swy -= w * v;
                r_swy2 -= w * v * v;
                count_left += 1;
                count_right -= 1;

                let next_val = ctx.pair_buf[pos + 1].0;
                if val == next_val {
                    continue;
                }

                if !has_nan {
                    if count_left < ctx.min_samples_leaf || count_right < ctx.min_samples_leaf {
                        continue;
                    }
                    if l_sw < 1e-15 || r_sw < 1e-15 {
                        continue;
                    }
                    let (cost, li, ri) = match ctx.criterion {
                        Criterion::Poisson => {
                            if l_swy < 1e-15 || r_swy < 1e-15 {
                                continue;
                            }
                            let cost = -l_swy * (l_swy / l_sw).ln() - r_swy * (r_swy / r_sw).ln();
                            // For Poisson, use MSE for child impurity (non-negative, correct purity check)
                            let li = mse_impurity_from_stats(l_swy, l_swy2, l_sw);
                            let ri = mse_impurity_from_stats(r_swy, r_swy2, r_sw);
                            (cost, li, ri)
                        }
                        _ => {
                            let lam = ctx.split_lambda;
                            let ld = l_sw + lam;
                            let rd = r_sw + lam;
                            let mse_l = (l_swy2 / ld - (l_swy / ld).powi(2)).max(0.0);
                            let mse_r = (r_swy2 / rd - (r_swy / rd).powi(2)).max(0.0);
                            (ld * mse_l + rd * mse_r, mse_l, mse_r)
                        }
                    };
                    if cost < best_cost {
                        best_cost = cost;
                        best_feat = feat;
                        best_thresh = (val + next_val) / 2.0;
                        best_nan_left = true;
                        best_left_imp = li;
                        best_right_imp = ri;
                        found = true;
                    }
                } else {
                    // Try NaN → left
                    let nl_sw = l_sw + nan_sw;
                    let nl_swy = l_swy + nan_swy;
                    let nl_swy2 = l_swy2 + nan_swy2;
                    let cl = count_left + nan_n;
                    let cr = count_right;
                    if cl >= ctx.min_samples_leaf && cr >= ctx.min_samples_leaf
                        && nl_sw > 1e-15 && r_sw > 1e-15
                    {
                        let (cost, li, ri) = match ctx.criterion {
                            Criterion::Poisson => {
                                if nl_swy < 1e-15 || r_swy < 1e-15 { (f64::INFINITY, 0.0, 0.0) } else {
                                    let cost = -nl_swy * (nl_swy / nl_sw).ln() - r_swy * (r_swy / r_sw).ln();
                                    let li = mse_impurity_from_stats(nl_swy, nl_swy2, nl_sw);
                                    let ri = mse_impurity_from_stats(r_swy, r_swy2, r_sw);
                                    (cost, li, ri)
                                }
                            }
                            _ => {
                                let lam = ctx.split_lambda;
                                let ld = nl_sw + lam;
                                let rd = r_sw + lam;
                                let mse_l = (nl_swy2 / ld - (nl_swy / ld).powi(2)).max(0.0);
                                let mse_r = (r_swy2 / rd - (r_swy / rd).powi(2)).max(0.0);
                                (ld * mse_l + rd * mse_r, mse_l, mse_r)
                            }
                        };
                        if cost < best_cost {
                            best_cost = cost;
                            best_feat = feat;
                            best_thresh = (val + next_val) / 2.0;
                            best_nan_left = true;
                            best_left_imp = li;
                            best_right_imp = ri;
                            found = true;
                        }
                    }

                    // Try NaN → right
                    let nr_sw = r_sw + nan_sw;
                    let nr_swy = r_swy + nan_swy;
                    let nr_swy2 = r_swy2 + nan_swy2;
                    let cl2 = count_left;
                    let cr2 = count_right + nan_n;
                    if cl2 >= ctx.min_samples_leaf && cr2 >= ctx.min_samples_leaf
                        && l_sw > 1e-15 && nr_sw > 1e-15
                    {
                        let (cost, li, ri) = match ctx.criterion {
                            Criterion::Poisson => {
                                if l_swy < 1e-15 || nr_swy < 1e-15 { (f64::INFINITY, 0.0, 0.0) } else {
                                    let cost = -l_swy * (l_swy / l_sw).ln() - nr_swy * (nr_swy / nr_sw).ln();
                                    let li = mse_impurity_from_stats(l_swy, l_swy2, l_sw);
                                    let ri = mse_impurity_from_stats(nr_swy, nr_swy2, nr_sw);
                                    (cost, li, ri)
                                }
                            }
                            _ => {
                                let lam = ctx.split_lambda;
                                let ld = l_sw + lam;
                                let rd = nr_sw + lam;
                                let mse_l = (l_swy2 / ld - (l_swy / ld).powi(2)).max(0.0);
                                let mse_r = (nr_swy2 / rd - (nr_swy / rd).powi(2)).max(0.0);
                                (ld * mse_l + rd * mse_r, mse_l, mse_r)
                            }
                        };
                        if cost < best_cost {
                            best_cost = cost;
                            best_feat = feat;
                            best_thresh = (val + next_val) / 2.0;
                            best_nan_left = false;
                            best_left_imp = li;
                            best_right_imp = ri;
                            found = true;
                        }
                    }
                }
            }
        }
    }

    if !found {
        return None;
    }

    Some(SplitResult {
        feature: best_feat,
        threshold: best_thresh,
        child_impurity_w: best_cost,
        nan_goes_left: best_nan_left,
        left_impurity: Some(best_left_imp),
        right_impurity: Some(best_right_imp),
    })
}

/// Evaluate a single (feature, threshold) split. Returns (cost, is_valid, nan_goes_left).
/// Used by Extra Trees where one random threshold per feature replaces a full scan.
/// NaN values are tried in both directions; the direction with lower cost wins.
fn score_threshold(
    ctx: &BuildContext,
    indices: &[usize],
    feat: usize,
    threshold: f64,
    node_w: f64,
) -> (f64, bool, bool) {
    let col = ctx.x.col_slice(feat);

    if ctx.is_clf {
        let k = ctx.n_classes;
        let mut left_counts = vec![0.0_f64; k];
        let mut right_counts = vec![0.0_f64; k];
        let mut nan_counts = vec![0.0_f64; k];
        let mut n_left_w = 0.0_f64;
        let mut n_right_w = 0.0_f64;
        let mut nan_w = 0.0_f64;
        let mut count_left = 0_usize;
        let mut count_right = 0_usize;
        let mut nan_n = 0_usize;

        for &i in indices {
            let v = unsafe { *col.get_unchecked(i) };
            let cls = ctx.y[i] as usize;
            let w = ctx.weights[i];
            if v.is_nan() {
                nan_counts[cls] += w;
                nan_w += w;
                nan_n += 1;
            } else if v <= threshold {
                left_counts[cls] += w;
                n_left_w += w;
                count_left += 1;
            } else {
                right_counts[cls] += w;
                n_right_w += w;
                count_right += 1;
            }
        }

        if nan_n == 0 {
            // No NaN: original path
            if count_left < ctx.min_samples_leaf || count_right < ctx.min_samples_leaf {
                return (0.0, false, true);
            }
            if n_left_w < 1e-15 || n_right_w < 1e-15 {
                return (0.0, false, true);
            }
            let cost = match ctx.criterion {
                Criterion::Entropy => {
                    n_left_w * entropy_impurity_from_counts(&left_counts, n_left_w)
                        + n_right_w * entropy_impurity_from_counts(&right_counts, n_right_w)
                }
                _ => {
                    let sq_l: f64 = left_counts.iter().map(|&c| c * c).sum();
                    let sq_r: f64 = right_counts.iter().map(|&c| c * c).sum();
                    node_w - sq_l / n_left_w - sq_r / n_right_w
                }
            };
            return (cost, true, true);
        }

        // Try both NaN directions, pick better
        let mut best_cost = f64::INFINITY;
        let mut best_nan_left = true;
        let mut any_valid = false;

        // NaN → left
        let cl = count_left + nan_n;
        let cr = count_right;
        let nl_w = n_left_w + nan_w;
        if cl >= ctx.min_samples_leaf && cr >= ctx.min_samples_leaf && nl_w > 1e-15 && n_right_w > 1e-15 {
            let cost = match ctx.criterion {
                Criterion::Entropy => {
                    let lc: Vec<f64> = (0..k).map(|c| left_counts[c] + nan_counts[c]).collect();
                    nl_w * entropy_impurity_from_counts(&lc, nl_w)
                        + n_right_w * entropy_impurity_from_counts(&right_counts, n_right_w)
                }
                _ => {
                    let sq_l: f64 = (0..k).map(|c| { let v = left_counts[c] + nan_counts[c]; v * v }).sum();
                    let sq_r: f64 = right_counts.iter().map(|&c| c * c).sum();
                    node_w - sq_l / nl_w - sq_r / n_right_w
                }
            };
            if cost < best_cost { best_cost = cost; best_nan_left = true; any_valid = true; }
        }

        // NaN → right
        let cl2 = count_left;
        let cr2 = count_right + nan_n;
        let nr_w = n_right_w + nan_w;
        if cl2 >= ctx.min_samples_leaf && cr2 >= ctx.min_samples_leaf && n_left_w > 1e-15 && nr_w > 1e-15 {
            let cost = match ctx.criterion {
                Criterion::Entropy => {
                    let rc: Vec<f64> = (0..k).map(|c| right_counts[c] + nan_counts[c]).collect();
                    n_left_w * entropy_impurity_from_counts(&left_counts, n_left_w)
                        + nr_w * entropy_impurity_from_counts(&rc, nr_w)
                }
                _ => {
                    let sq_l: f64 = left_counts.iter().map(|&c| c * c).sum();
                    let sq_r: f64 = (0..k).map(|c| { let v = right_counts[c] + nan_counts[c]; v * v }).sum();
                    node_w - sq_l / n_left_w - sq_r / nr_w
                }
            };
            if cost < best_cost { best_cost = cost; best_nan_left = false; any_valid = true; }
        }

        if !any_valid { return (0.0, false, true); }
        (best_cost, true, best_nan_left)
    } else {
        let mut l_sw = 0.0_f64;
        let mut l_swy = 0.0_f64;
        let mut l_swy2 = 0.0_f64;
        let mut r_sw = 0.0_f64;
        let mut r_swy = 0.0_f64;
        let mut r_swy2 = 0.0_f64;
        let mut nan_sw = 0.0_f64;
        let mut nan_swy = 0.0_f64;
        let mut nan_swy2 = 0.0_f64;
        let mut count_left = 0_usize;
        let mut count_right = 0_usize;
        let mut nan_n = 0_usize;

        for &i in indices {
            let vx = unsafe { *col.get_unchecked(i) };
            let w = ctx.weights[i];
            let vy = ctx.y[i];
            if vx.is_nan() {
                nan_sw += w;
                nan_swy += w * vy;
                nan_swy2 += w * vy * vy;
                nan_n += 1;
            } else if vx <= threshold {
                l_sw += w;
                l_swy += w * vy;
                l_swy2 += w * vy * vy;
                count_left += 1;
            } else {
                r_sw += w;
                r_swy += w * vy;
                r_swy2 += w * vy * vy;
                count_right += 1;
            }
        }

        if nan_n == 0 {
            // No NaN: original path
            if count_left < ctx.min_samples_leaf || count_right < ctx.min_samples_leaf {
                return (0.0, false, true);
            }
            if l_sw < 1e-15 || r_sw < 1e-15 {
                return (0.0, false, true);
            }
            let cost = match ctx.criterion {
                Criterion::Poisson => {
                    if l_swy < 1e-15 || r_swy < 1e-15 {
                        return (0.0, false, true);
                    }
                    -l_swy * (l_swy / l_sw).ln() - r_swy * (r_swy / r_sw).ln()
                }
                _ => {
                    let lam = ctx.split_lambda;
                    let ld = l_sw + lam;
                    let rd = r_sw + lam;
                    let mse_l = (l_swy2 / ld - (l_swy / ld).powi(2)).max(0.0);
                    let mse_r = (r_swy2 / rd - (r_swy / rd).powi(2)).max(0.0);
                    ld * mse_l + rd * mse_r
                }
            };
            return (cost, true, true);
        }

        // Try both NaN directions, pick better
        let mut best_cost = f64::INFINITY;
        let mut best_nan_left = true;
        let mut any_valid = false;

        // NaN → left
        let nl_sw = l_sw + nan_sw;
        let nl_swy = l_swy + nan_swy;
        let nl_swy2 = l_swy2 + nan_swy2;
        let cl = count_left + nan_n;
        let cr = count_right;
        if cl >= ctx.min_samples_leaf && cr >= ctx.min_samples_leaf && nl_sw > 1e-15 && r_sw > 1e-15 {
            let cost = match ctx.criterion {
                Criterion::Poisson => {
                    if nl_swy > 1e-15 && r_swy > 1e-15 {
                        -nl_swy * (nl_swy / nl_sw).ln() - r_swy * (r_swy / r_sw).ln()
                    } else { f64::INFINITY }
                }
                _ => {
                    let lam = ctx.split_lambda;
                    let ld = nl_sw + lam;
                    let rd = r_sw + lam;
                    let mse_l = (nl_swy2 / ld - (nl_swy / ld).powi(2)).max(0.0);
                    let mse_r = (r_swy2 / rd - (r_swy / rd).powi(2)).max(0.0);
                    ld * mse_l + rd * mse_r
                }
            };
            if cost < best_cost { best_cost = cost; best_nan_left = true; any_valid = true; }
        }

        // NaN → right
        let nr_sw = r_sw + nan_sw;
        let nr_swy = r_swy + nan_swy;
        let nr_swy2 = r_swy2 + nan_swy2;
        let cl2 = count_left;
        let cr2 = count_right + nan_n;
        if cl2 >= ctx.min_samples_leaf && cr2 >= ctx.min_samples_leaf && l_sw > 1e-15 && nr_sw > 1e-15 {
            let cost = match ctx.criterion {
                Criterion::Poisson => {
                    if l_swy > 1e-15 && nr_swy > 1e-15 {
                        -l_swy * (l_swy / l_sw).ln() - nr_swy * (nr_swy / nr_sw).ln()
                    } else { f64::INFINITY }
                }
                _ => {
                    let lam = ctx.split_lambda;
                    let ld = l_sw + lam;
                    let rd = nr_sw + lam;
                    let mse_l = (l_swy2 / ld - (l_swy / ld).powi(2)).max(0.0);
                    let mse_r = (nr_swy2 / rd - (nr_swy / rd).powi(2)).max(0.0);
                    ld * mse_l + rd * mse_r
                }
            };
            if cost < best_cost { best_cost = cost; best_nan_left = false; any_valid = true; }
        }

        if !any_valid { return (0.0, false, true); }
        (best_cost, true, best_nan_left)
    }
}

/// Extra Trees split search (Geurts et al. 2006).
///
/// For each candidate feature, draw ONE random threshold uniformly in `[feat_min, feat_max]`
/// and score its impurity reduction. Return the best (feature, threshold).
/// Bypasses the histogram bin scan entirely — O(n·m) where m = max_features.
fn best_split_extra_trees(
    ctx: &mut BuildContext,
    indices: &[usize],
    node_w: f64,
) -> Option<SplitResult> {
    let p = ctx.x.ncols;

    let mut best_cost = f64::INFINITY;
    let mut best_feat = 0_usize;
    let mut best_thresh = 0.0_f64;
    let mut best_nan_left = true;
    let mut found = false;

    let feats_to_eval = select_features(p, ctx.max_features, &mut ctx.rng_state);

    for &feat in &feats_to_eval {
        let col = ctx.x.col_slice(feat);
        let mut fmin = f64::INFINITY;
        let mut fmax = f64::NEG_INFINITY;
        for &i in indices {
            let v = unsafe { *col.get_unchecked(i) };
            // Track min/max of non-NaN values only
            if !v.is_nan() {
                if v < fmin {
                    fmin = v;
                }
                if v > fmax {
                    fmax = v;
                }
            }
        }
        // Skip if all non-NaN values identical or all NaN.
        if fmin >= fmax {
            continue;
        }

        // Draw one random threshold uniformly in [fmin, fmax)
        let r = (xorshift64(&mut ctx.rng_state) as f64) / (u64::MAX as f64);
        let threshold = fmin + r * (fmax - fmin);

        let (cost, valid, nan_left) = score_threshold(ctx, indices, feat, threshold, node_w);
        if valid && cost < best_cost {
            best_cost = cost;
            best_feat = feat;
            best_thresh = threshold;
            best_nan_left = nan_left;
            found = true;
        }
    }

    if !found {
        return None;
    }

    Some(SplitResult {
        feature: best_feat,
        threshold: best_thresh,
        child_impurity_w: best_cost,
        nan_goes_left: best_nan_left,
        left_impurity: None,
        right_impurity: None,
    })
}

// ---------------------------------------------------------------------------
// Histogram-based split search (O(n·p) per node)
// ---------------------------------------------------------------------------

/// Find the best split using histogram binning.
///
/// Instead of sorting each feature (O(n log n)), we scan samples once per feature
/// to build a 256-bin histogram (O(n)), then scan bins for the best boundary (O(256)).
/// Net: O(n·p) per node instead of O(n·p·log n). 3-5x faster at 100K+ rows.
fn best_split_histogram(
    qm: &QuantizedMatrix,
    y: &[f64],
    weights: &[f64],
    indices: &[usize],
    node_w: f64,
    n_classes: usize,
    min_samples_leaf: usize,
    is_clf: bool,
    max_features: Option<usize>,
    rng_state: &mut u64,
    criterion: Criterion,
    split_lambda: f64,
) -> Option<SplitResult> {
    let p = qm.ncols;
    let mut best_feat = 0;
    let mut best_bin = 0_u8;
    let mut best_cost = f64::INFINITY; // unified lower = better for all criteria
    let mut best_nan_left = true;
    let mut found = false;

    let feats_to_eval = select_features(p, max_features, rng_state);

    if is_clf {
        let mut hist = ClfHistogram::new(MAX_BINS, n_classes);
        for &feat in &feats_to_eval {
            let nb = qm.n_bins_for(feat);
            hist.clear(nb);
            for &i in indices {
                // SAFETY: i < nrows, feat < ncols — validated at tree entry
                let bin = unsafe { qm.get_unchecked(i, feat) };
                hist.add(bin, y[i] as usize, weights[i]);
            }
            // Gini: convert proxy to cost (node_w - proxy) for unified comparison
            let result = match criterion {
                Criterion::Entropy => hist.best_entropy_split(min_samples_leaf),
                _ => hist
                    .best_gini_split(min_samples_leaf)
                    .map(|(bin, proxy, nan_left)| (bin, node_w - proxy, nan_left)),
            };
            if let Some((bin, cost, nan_left)) = result {
                if cost < best_cost {
                    best_cost = cost;
                    best_feat = feat;
                    best_bin = bin;
                    best_nan_left = nan_left;
                    found = true;
                }
            }
        }
    } else {
        let mut hist = RegHistogram::new(MAX_BINS);
        for &feat in &feats_to_eval {
            let nb = qm.n_bins_for(feat);
            hist.clear(nb);
            for &i in indices {
                let bin = unsafe { qm.get_unchecked(i, feat) };
                hist.add(bin, y[i], weights[i]);
            }
            let result = match criterion {
                Criterion::Poisson => hist.best_poisson_split(min_samples_leaf),
                _ => hist.best_mse_split(min_samples_leaf, split_lambda),
            };
            if let Some((bin, cost, nan_left)) = result {
                if cost < best_cost {
                    best_cost = cost;
                    best_feat = feat;
                    best_bin = bin;
                    best_nan_left = nan_left;
                    found = true;
                }
            }
        }
    }

    if found {
        Some(SplitResult {
            feature: best_feat,
            threshold: qm.threshold_for(best_feat, best_bin),
            child_impurity_w: best_cost,
            nan_goes_left: best_nan_left,
            left_impurity: None,
            right_impurity: None,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// In-place partition
// ---------------------------------------------------------------------------

/// In-place two-pointer partition. Returns split point: left = [..sp], right = [sp..].
/// NaN values are routed according to `nan_goes_left`. Zero allocation.
#[inline]
fn partition(indices: &mut [usize], x: &ColMajorMatrix, feat: usize, thresh: f64, nan_goes_left: bool) -> usize {
    let mut lo = 0;
    let mut hi = indices.len();
    while lo < hi {
        // SAFETY: indices[lo] < x.nrows, feat < x.ncols — validated at tree entry
        let val = unsafe { x.get_unchecked(indices[lo], feat) };
        let goes_left = if val.is_nan() { nan_goes_left } else { val <= thresh };
        if goes_left {
            lo += 1;
        } else {
            hi -= 1;
            indices.swap(lo, hi);
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// Leaf construction
// ---------------------------------------------------------------------------

/// Build a leaf node. For clf, pushes probabilities into proba_pool.
/// `bounds` is (lb, ub): regression leaf value is clamped to this range (monotone constraints).
fn make_leaf(ctx: &mut BuildContext, indices: &[usize], n_w: f64, bounds: (f64, f64)) -> NodeKind {
    if ctx.is_clf {
        let k = ctx.n_classes;
        let mut counts = vec![0.0_f64; k];
        for &i in indices {
            counts[ctx.y[i] as usize] += ctx.weights[i];
        }
        // Majority class (ties: lower index wins — sklearn argmax behaviour)
        let value = counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c as f64)
            .unwrap_or(0.0);
        let safe_w = if n_w < 1e-15 { 1.0 } else { n_w };
        let proba_offset = ctx.proba_pool.len();
        for &c in &counts {
            ctx.proba_pool.push(c / safe_w);
        }
        NodeKind::Leaf {
            value,
            proba_offset,
        }
    } else {
        let sum_wy: f64 = indices.iter().map(|&i| ctx.weights[i] * ctx.y[i]).sum();
        let safe_w = if n_w < 1e-15 { 1.0 } else { n_w };
        let raw_value = sum_wy / safe_w;
        // Clamp to monotone bounds (no-op when bounds == (-inf, +inf))
        let (lb, ub) = bounds;
        let value = raw_value.max(lb).min(ub);
        NodeKind::Leaf {
            value,
            proba_offset: 0,
        }
    }
}

/// Weighted mean of y for a set of sample indices (regression helper).
#[inline]
fn weighted_mean_of_indices(ctx: &BuildContext, indices: &[usize]) -> f64 {
    let n_w: f64 = indices.iter().map(|&i| ctx.weights[i]).sum();
    if n_w < 1e-15 {
        return 0.0;
    }
    let sum_wy: f64 = indices.iter().map(|&i| ctx.weights[i] * ctx.y[i]).sum();
    sum_wy / n_w
}

/// Compute monotone-constraint child bounds for a regression split.
///
/// For classification or no monotone_cst: returns `(bounds, bounds)` unchanged.
/// For regression with a constrained feature:
/// - `+1` (increasing): left_ub = min(ub, mu_R), right_lb = max(lb, mu_L)
/// - `-1` (decreasing): left_lb = max(lb, mu_R), right_ub = min(ub, mu_L)
///
/// Bounds are safe-clamped so lb ≤ ub at all times.
fn compute_child_bounds(
    ctx: &BuildContext,
    left_indices: &[usize],
    right_indices: &[usize],
    feature: usize,
    bounds: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
    if ctx.is_clf {
        return (bounds, bounds);
    }
    let cst = match &ctx.monotone_cst {
        Some(c) if feature < c.len() => c[feature],
        _ => return (bounds, bounds),
    };
    if cst == 0 {
        return (bounds, bounds);
    }
    let (lb, ub) = bounds;
    let mu_l = weighted_mean_of_indices(ctx, left_indices);
    let mu_r = weighted_mean_of_indices(ctx, right_indices);
    if cst > 0 {
        // Monotone increasing: use a shared midpoint so left_ub == right_lb.
        // By induction this guarantees max(left leaves) ≤ mid ≤ min(right leaves).
        let mid = ((mu_l + mu_r) / 2.0).max(lb).min(ub);
        ((lb, mid), (mid, ub))
    } else {
        // Monotone decreasing: shared midpoint, left ≥ mid ≥ right.
        let mid = ((mu_l + mu_r) / 2.0).max(lb).min(ub);
        ((mid, ub), (lb, mid))
    }
}

// ---------------------------------------------------------------------------
// Cost-complexity pruning (Breiman et al. 1984, Chapter 3)
// ---------------------------------------------------------------------------

/// Post-hoc cost-complexity pruning (CCP) — weakest-link pruning.
///
/// For each internal (split) node t with subtree T_t:
///   effective_alpha(t) = (R(t) - R(T_t)) / (|T_t| - 1)
/// where R(t) = node cost if t were a leaf, R(T_t) = sum of leaf costs,
/// |T_t| = number of leaves in subtree.
///
/// Repeatedly prune the subtree with smallest effective_alpha until
/// effective_alpha > ccp_alpha.
///
/// - Classification: R(t) = (1 - max(class_proportions)) * n_samples(t) / n_total
/// - Regression: R(t) = MSE(t) * n_samples(t) / n_total
fn prune_ccp(
    nodes: &mut Vec<Node>,
    proba_pool: &mut Vec<f64>,
    ccp_alpha: f64,
    n_classes: usize,
    is_clf: bool,
) {
    if ccp_alpha <= 0.0 || nodes.is_empty() {
        return;
    }

    // Total number of training samples (root node).
    let n_total = nodes[0].n_samples as f64;
    if n_total < 1e-15 {
        return;
    }

    loop {
        // Find the internal node with smallest effective alpha.
        let mut best_alpha = f64::INFINITY;
        let mut best_node = usize::MAX;

        for i in 0..nodes.len() {
            if !matches!(nodes[i].kind, NodeKind::Split { .. }) {
                continue;
            }

            // Compute R(t): cost if this node were a leaf.
            let r_t = node_leaf_cost(nodes, proba_pool, i, n_total, n_classes, is_clf);

            // Compute R(T_t): sum of leaf costs in subtree, and |T_t|.
            let (r_subtree, n_leaves) = subtree_cost(nodes, proba_pool, i, n_total, n_classes, is_clf);

            if n_leaves <= 1 {
                continue; // already a leaf or degenerate
            }

            let eff_alpha = (r_t - r_subtree) / (n_leaves as f64 - 1.0);

            if eff_alpha < best_alpha {
                best_alpha = eff_alpha;
                best_node = i;
            }
        }

        // Stop if no prunable node, or weakest link exceeds threshold.
        if best_node == usize::MAX || best_alpha > ccp_alpha {
            break;
        }

        // Prune: convert the subtree rooted at best_node to a leaf.
        convert_to_leaf(nodes, proba_pool, best_node, n_classes, is_clf);
    }
}

/// Cost of node i if it were treated as a leaf: R(t) = impurity(t) * n_samples(t) / n_total.
fn node_leaf_cost(
    nodes: &[Node],
    proba_pool: &[f64],
    i: usize,
    n_total: f64,
    n_classes: usize,
    is_clf: bool,
) -> f64 {
    let n_i = nodes[i].n_samples as f64;
    if is_clf {
        // Aggregate class proportions from descendant leaves.
        // Node's "leaf impurity" is 1 - max(class_proportion).
        let mut class_counts = vec![0.0_f64; n_classes];
        accumulate_leaf_probas(nodes, proba_pool, i, n_classes, &mut class_counts);
        let total: f64 = class_counts.iter().sum();
        if total < 1e-15 {
            return 0.0;
        }
        let max_prop = class_counts.iter().cloned().fold(0.0_f64, f64::max) / total;
        (1.0 - max_prop) * n_i / n_total
    } else {
        // Regression: MSE of the node as leaf = variance of leaf values weighted by samples.
        let (sum_val, sum_val_sq, total_n) = accumulate_leaf_reg_stats(nodes, i);
        if total_n < 1e-15 {
            return 0.0;
        }
        let mean = sum_val / total_n;
        let mse = (sum_val_sq / total_n - mean * mean).max(0.0);
        mse * n_i / n_total
    }
}

/// Accumulate class proportions from all leaves under node i.
/// Each leaf contributes its class probabilities weighted by its n_samples.
fn accumulate_leaf_probas(
    nodes: &[Node],
    proba_pool: &[f64],
    i: usize,
    n_classes: usize,
    counts: &mut [f64],
) {
    match &nodes[i].kind {
        NodeKind::Leaf { proba_offset, .. } => {
            let n_i = nodes[i].n_samples as f64;
            for c in 0..n_classes {
                counts[c] += proba_pool[proba_offset + c] * n_i;
            }
        }
        NodeKind::Split { left, right, .. } => {
            accumulate_leaf_probas(nodes, proba_pool, *left, n_classes, counts);
            accumulate_leaf_probas(nodes, proba_pool, *right, n_classes, counts);
        }
        NodeKind::Placeholder => {}
    }
}

/// Accumulate regression statistics from all leaves under node i.
/// Returns (sum of value*n_samples, sum of value^2*n_samples, sum of n_samples).
fn accumulate_leaf_reg_stats(nodes: &[Node], i: usize) -> (f64, f64, f64) {
    match &nodes[i].kind {
        NodeKind::Leaf { value, .. } => {
            let n_i = nodes[i].n_samples as f64;
            (value * n_i, value * value * n_i, n_i)
        }
        NodeKind::Split { left, right, .. } => {
            let (sv1, svs1, sn1) = accumulate_leaf_reg_stats(nodes, *left);
            let (sv2, svs2, sn2) = accumulate_leaf_reg_stats(nodes, *right);
            (sv1 + sv2, svs1 + svs2, sn1 + sn2)
        }
        NodeKind::Placeholder => (0.0, 0.0, 0.0),
    }
}

/// Sum of leaf costs R(T_t) and leaf count |T_t| for subtree rooted at i.
fn subtree_cost(
    nodes: &[Node],
    proba_pool: &[f64],
    i: usize,
    n_total: f64,
    n_classes: usize,
    is_clf: bool,
) -> (f64, usize) {
    match &nodes[i].kind {
        NodeKind::Leaf { .. } => {
            let cost = node_leaf_cost(nodes, proba_pool, i, n_total, n_classes, is_clf);
            (cost, 1)
        }
        NodeKind::Split { left, right, .. } => {
            let (cost_l, nl) = subtree_cost(nodes, proba_pool, *left, n_total, n_classes, is_clf);
            let (cost_r, nr) = subtree_cost(nodes, proba_pool, *right, n_total, n_classes, is_clf);
            (cost_l + cost_r, nl + nr)
        }
        NodeKind::Placeholder => (0.0, 0),
    }
}

/// Convert a split node (and its subtree) into a leaf.
/// For classification: majority class from aggregated leaf probabilities.
/// For regression: weighted mean of descendant leaf values.
fn convert_to_leaf(
    nodes: &mut Vec<Node>,
    proba_pool: &mut Vec<f64>,
    i: usize,
    n_classes: usize,
    is_clf: bool,
) {
    if is_clf {
        let mut class_counts = vec![0.0_f64; n_classes];
        accumulate_leaf_probas(nodes, proba_pool, i, n_classes, &mut class_counts);
        let total: f64 = class_counts.iter().sum();
        let safe_total = if total < 1e-15 { 1.0 } else { total };

        // Majority class
        let value = class_counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c as f64)
            .unwrap_or(0.0);

        // Write new probabilities to pool
        let proba_offset = proba_pool.len();
        for &c in &class_counts {
            proba_pool.push(c / safe_total);
        }

        nodes[i].kind = NodeKind::Leaf {
            value,
            proba_offset,
        };
    } else {
        // Regression: weighted mean of descendant leaf values.
        let (sum_val, _sum_val_sq, total_n) = accumulate_leaf_reg_stats(nodes, i);
        let value = if total_n < 1e-15 {
            0.0
        } else {
            sum_val / total_n
        };
        nodes[i].kind = NodeKind::Leaf {
            value,
            proba_offset: 0,
        };
    }
}

// ---------------------------------------------------------------------------
// Impurity
// ---------------------------------------------------------------------------

/// Shannon entropy from pre-computed weighted class counts (base-e log).
#[inline]
fn entropy_impurity_from_counts(counts: &[f64], n_w: f64) -> f64 {
    if n_w < 1e-15 {
        return 0.0;
    }
    let mut h = 0.0_f64;
    for &c in counts {
        if c > 1e-15 {
            let p = c / n_w;
            h -= p * p.ln();
        }
    }
    h
}

/// Gini impurity from pre-computed weighted class counts.
#[inline]
fn gini_impurity_from_counts(counts: &[f64], n_w: f64) -> f64 {
    if n_w < 1e-15 {
        return 0.0;
    }
    let sq_sum: f64 = counts.iter().map(|&c| c * c).sum();
    1.0 - sq_sum / (n_w * n_w)
}

/// MSE impurity from pre-computed weighted sums.
#[inline]
fn mse_impurity_from_stats(swy: f64, swy2: f64, sw: f64) -> f64 {
    if sw < 1e-15 {
        return 0.0;
    }
    (swy2 / sw - (swy / sw) * (swy / sw)).max(0.0)
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn normalize_weights(
    n: usize,
    sample_weight: Option<&DVector<f64>>,
) -> Result<Vec<f64>, MlError> {
    match sample_weight {
        None => Ok(vec![1.0_f64; n]),
        Some(sw) => {
            if sw.len() != n {
                return Err(MlError::DimensionMismatch {
                    expected: n,
                    got: sw.len(),
                });
            }
            let sw_sum: f64 = sw.iter().sum();
            if sw_sum <= 0.0 {
                return Err(MlError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
            Ok((0..n).map(|i| sw[i] * n as f64 / sw_sum).collect())
        }
    }
}

fn normalize_importances(raw: Vec<f64>, p: usize) -> Vec<f64> {
    let s: f64 = raw.iter().sum();
    if s < 1e-15 {
        vec![1.0 / p as f64; p]
    } else {
        raw.iter().map(|&v| v / s).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build deterministic classification data: 100 rows, 3 features, binary target.
    fn make_clf_data() -> (DMatrix<f64>, Vec<i64>) {
        let n = 100_usize;
        let data: Vec<f64> = (0..n * 3)
            .map(|i| (i as f64 * 1.618033) % 1.0)
            .collect();
        let x = DMatrix::from_row_slice(n, 3, &data);
        let y: Vec<i64> = (0..n)
            .map(|i| if data[i * 3] + data[i * 3 + 1] > 1.0 { 1 } else { 0 })
            .collect();
        (x, y)
    }

    /// Build deterministic regression data: 100 rows, 3 features.
    fn make_reg_data() -> (DMatrix<f64>, Vec<f64>) {
        let n = 100_usize;
        let data: Vec<f64> = (0..n * 3)
            .map(|i| (i as f64 * 1.618033) % 1.0)
            .collect();
        let x = DMatrix::from_row_slice(n, 3, &data);
        let y: Vec<f64> = (0..n)
            .map(|i| data[i * 3] + 2.0 * data[i * 3 + 1])
            .collect();
        (x, y)
    }

    #[test]
    fn test_cart_predict_clf_parallel_determinism() {
        let (x, y) = make_clf_data();
        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        // Run predict multiple times — must be bitwise identical.
        let baseline = tree.predict_clf(&x);
        for _ in 0..5 {
            let result = tree.predict_clf(&x);
            assert_eq!(baseline, result, "parallel predict_clf not deterministic");
        }
    }

    #[test]
    fn test_cart_predict_reg_parallel_determinism() {
        let (x, y_vec) = make_reg_data();
        let y = DVector::from_vec(y_vec);
        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let baseline = tree.predict_reg(&x);
        for _ in 0..5 {
            let result = tree.predict_reg(&x);
            assert_eq!(baseline.as_slice(), result.as_slice(),
                "parallel predict_reg not deterministic");
        }
    }

    #[test]
    fn test_cart_predict_proba_parallel_determinism() {
        let (x, y) = make_clf_data();
        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let baseline = tree.predict_proba(&x);
        for _ in 0..5 {
            let result = tree.predict_proba(&x);
            assert_eq!(baseline.as_slice(), result.as_slice(),
                "parallel predict_proba not deterministic");
        }
    }

    // -----------------------------------------------------------------------
    // min_impurity_decrease tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_impurity_decrease_backward_compat_clf() {
        // Default min_impurity_decrease=0.0 should produce identical tree to explicit 0.0
        let (x, y) = make_clf_data();

        let mut tree_default = DecisionTreeModel::new(10, 2, 1);
        tree_default.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let mut tree_explicit = DecisionTreeModel::new(10, 2, 1);
        tree_explicit.min_impurity_decrease = 0.0;
        tree_explicit.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        assert_eq!(
            tree_default.nodes.len(),
            tree_explicit.nodes.len(),
            "min_impurity_decrease=0.0 should produce identical tree as default"
        );
        let preds_d = tree_default.predict_clf(&x);
        let preds_e = tree_explicit.predict_clf(&x);
        assert_eq!(preds_d, preds_e);
    }

    #[test]
    fn test_min_impurity_decrease_backward_compat_reg() {
        let (x, y_vec) = make_reg_data();
        let y = DVector::from_vec(y_vec);

        let mut tree_default = DecisionTreeModel::new(10, 2, 1);
        tree_default.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let mut tree_explicit = DecisionTreeModel::new(10, 2, 1);
        tree_explicit.min_impurity_decrease = 0.0;
        tree_explicit.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        assert_eq!(
            tree_default.nodes.len(),
            tree_explicit.nodes.len(),
            "min_impurity_decrease=0.0 should produce identical tree as default"
        );
    }

    /// Build noisy classification data: 200 rows, 5 features, binary target with
    /// deterministic pseudo-random noise that forces a deep tree.
    fn make_noisy_clf_data() -> (DMatrix<f64>, Vec<i64>) {
        let n = 200_usize;
        let p = 5_usize;
        // Use golden ratio offsets for a pseudo-random spread, but multiply
        // by primes so columns are not correlated.
        let data: Vec<f64> = (0..n * p)
            .map(|i| ((i as f64 * 0.618033 + 0.123) * (1 + i % 7) as f64) % 1.0)
            .collect();
        let x = DMatrix::from_row_slice(n, p, &data);
        // Target depends on ALL features with nonlinear interaction + noise.
        // This prevents a clean single split.
        let y: Vec<i64> = (0..n)
            .map(|i| {
                let f0 = data[i * p];
                let f1 = data[i * p + 1];
                let f2 = data[i * p + 2];
                let f3 = data[i * p + 3];
                let score = f0 * f1 + f2 - 0.5 * f3;
                if score > 0.5 { 1 } else { 0 }
            })
            .collect();
        (x, y)
    }

    #[test]
    fn test_min_impurity_decrease_produces_shallower_tree_clf() {
        let (x, y) = make_noisy_clf_data();

        // Deep tree with min_samples_leaf=1 to get many marginal splits
        let mut tree_full = DecisionTreeModel::new(20, 2, 1);
        tree_full.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let mut tree_pruned = DecisionTreeModel::new(20, 2, 1);
        tree_pruned.min_impurity_decrease = 0.005;
        tree_pruned.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        // Verify full tree is deep enough that pruning matters
        assert!(
            tree_full.nodes.len() > 3,
            "full tree should have more than 3 nodes, got {}",
            tree_full.nodes.len()
        );
        assert!(
            tree_pruned.nodes.len() < tree_full.nodes.len(),
            "min_impurity_decrease=0.005 should produce fewer nodes: pruned={} vs full={}",
            tree_pruned.nodes.len(),
            tree_full.nodes.len()
        );
    }

    #[test]
    fn test_min_impurity_decrease_produces_shallower_tree_reg() {
        let (x, y_vec) = make_reg_data();
        let y = DVector::from_vec(y_vec);

        let mut tree_full = DecisionTreeModel::new(10, 2, 1);
        tree_full.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let mut tree_pruned = DecisionTreeModel::new(10, 2, 1);
        tree_pruned.min_impurity_decrease = 0.01;
        tree_pruned.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        assert!(
            tree_pruned.nodes.len() < tree_full.nodes.len(),
            "min_impurity_decrease=0.01 should produce fewer nodes: pruned={} vs full={}",
            tree_pruned.nodes.len(),
            tree_full.nodes.len()
        );
    }

    #[test]
    fn test_min_impurity_decrease_extreme_makes_stump() {
        // A very high threshold should produce only a root leaf (or root + 2 children at most)
        let (x, y) = make_clf_data();

        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.min_impurity_decrease = 10.0; // impossibly high threshold
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        // With threshold=10.0, no split's gain can be that high, so tree = single leaf
        assert_eq!(tree.nodes.len(), 1, "extreme threshold should produce single-leaf tree");
    }

    #[test]
    fn test_min_impurity_decrease_serialization_roundtrip() {
        let (x, y) = make_clf_data();

        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.min_impurity_decrease = 0.05;
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
        let n_nodes = tree.nodes.len();

        let json = tree.to_json().unwrap();
        let tree2 = DecisionTreeModel::from_json(&json).unwrap();

        assert_eq!(tree2.nodes.len(), n_nodes, "roundtrip should preserve structure");
        assert!((tree2.min_impurity_decrease - 0.05).abs() < 1e-15,
            "roundtrip should preserve min_impurity_decrease");

        let preds1 = tree.predict_clf(&x);
        let preds2 = tree2.predict_clf(&x);
        assert_eq!(preds1, preds2, "roundtrip should preserve predictions");
    }

    // -----------------------------------------------------------------------
    // ccp_alpha (cost-complexity pruning) tests
    // -----------------------------------------------------------------------

    /// Count how many nodes are Split (internal) nodes.
    fn count_splits(nodes: &[Node]) -> usize {
        nodes.iter().filter(|n| matches!(n.kind, NodeKind::Split { .. })).count()
    }

    #[test]
    fn test_ccp_alpha_backward_compat_clf() {
        // ccp_alpha=0.0 (default) must produce identical tree to explicit 0.0
        let (x, y) = make_clf_data();

        let mut tree_default = DecisionTreeModel::new(10, 2, 1);
        tree_default.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let mut tree_explicit = DecisionTreeModel::new(10, 2, 1);
        tree_explicit.ccp_alpha = 0.0;
        tree_explicit.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        assert_eq!(
            tree_default.nodes.len(),
            tree_explicit.nodes.len(),
            "ccp_alpha=0.0 should produce identical tree as default"
        );
        let preds_d = tree_default.predict_clf(&x);
        let preds_e = tree_explicit.predict_clf(&x);
        assert_eq!(preds_d, preds_e);
    }

    #[test]
    fn test_ccp_alpha_backward_compat_reg() {
        let (x, y_vec) = make_reg_data();
        let y = DVector::from_vec(y_vec);

        let mut tree_default = DecisionTreeModel::new(10, 2, 1);
        tree_default.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let mut tree_explicit = DecisionTreeModel::new(10, 2, 1);
        tree_explicit.ccp_alpha = 0.0;
        tree_explicit.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        assert_eq!(
            tree_default.nodes.len(),
            tree_explicit.nodes.len(),
            "ccp_alpha=0.0 should produce identical tree as default"
        );
    }

    #[test]
    fn test_ccp_alpha_produces_fewer_nodes_clf() {
        let (x, y) = make_noisy_clf_data();

        let mut tree_full = DecisionTreeModel::new(20, 2, 1);
        tree_full.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let mut tree_pruned = DecisionTreeModel::new(20, 2, 1);
        tree_pruned.ccp_alpha = 0.01;
        tree_pruned.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        // Full tree should be deep enough for pruning to matter
        assert!(
            count_splits(&tree_full.nodes) > 3,
            "full tree should have more than 3 split nodes, got {}",
            count_splits(&tree_full.nodes)
        );

        let full_splits = count_splits(&tree_full.nodes);
        let pruned_splits = count_splits(&tree_pruned.nodes);

        assert!(
            pruned_splits < full_splits,
            "ccp_alpha=0.01 should produce fewer split nodes: pruned={} vs full={}",
            pruned_splits, full_splits
        );
    }

    #[test]
    fn test_ccp_alpha_produces_fewer_nodes_reg() {
        let (x, y_vec) = make_reg_data();
        let y = DVector::from_vec(y_vec);

        let mut tree_full = DecisionTreeModel::new(10, 2, 1);
        tree_full.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let mut tree_pruned = DecisionTreeModel::new(10, 2, 1);
        tree_pruned.ccp_alpha = 0.01;
        tree_pruned.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let full_splits = count_splits(&tree_full.nodes);
        let pruned_splits = count_splits(&tree_pruned.nodes);

        assert!(
            pruned_splits < full_splits,
            "ccp_alpha=0.01 should produce fewer split nodes: pruned={} vs full={}",
            pruned_splits, full_splits
        );
    }

    #[test]
    fn test_ccp_alpha_extreme_makes_stump() {
        // Very high ccp_alpha should prune the root to a leaf.
        let (x, y) = make_clf_data();

        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.ccp_alpha = 10.0; // impossibly high
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        // Root should be a leaf after pruning (orphaned children still exist in Vec,
        // but the root itself must be a leaf — no reachable splits).
        assert!(
            matches!(tree.nodes[0].kind, NodeKind::Leaf { .. }),
            "extreme ccp_alpha should make root a leaf"
        );
        // Predictions should still work — all return the same majority class.
        let preds = tree.predict_clf(&x);
        assert_eq!(preds.len(), x.nrows());
        // All predictions should be the same class (the majority).
        let first = preds[0];
        assert!(preds.iter().all(|&p| p == first), "all predictions should be identical after full pruning");
    }

    #[test]
    fn test_ccp_alpha_monotonic_pruning() {
        // Larger ccp_alpha should produce fewer or equal splits.
        let (x, y) = make_noisy_clf_data();

        let alphas = [0.0, 0.001, 0.01, 0.05, 0.1, 1.0];
        let mut prev_splits = usize::MAX;
        for &alpha in &alphas {
            let mut tree = DecisionTreeModel::new(20, 2, 1);
            tree.ccp_alpha = alpha;
            tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
            let splits = count_splits(&tree.nodes);
            assert!(
                splits <= prev_splits,
                "larger ccp_alpha={} should not produce more splits ({}) than alpha=prev ({})",
                alpha, splits, prev_splits
            );
            prev_splits = splits;
        }
    }

    #[test]
    fn test_ccp_alpha_predictions_still_valid_clf() {
        // After pruning, all predictions must be valid class indices.
        let (x, y) = make_noisy_clf_data();

        let mut tree = DecisionTreeModel::new(20, 2, 1);
        tree.ccp_alpha = 0.02;
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let preds = tree.predict_clf(&x);
        for &p in &preds {
            assert!(p == 0 || p == 1, "prediction {} not a valid class index", p);
        }

        // predict_proba should still sum to ~1.0
        let proba = tree.predict_proba(&x);
        for i in 0..x.nrows() {
            let row_sum: f64 = (0..tree.n_classes).map(|j| proba[(i, j)]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "predict_proba row {} sums to {} not 1.0",
                i, row_sum
            );
        }
    }

    #[test]
    fn test_ccp_alpha_serialization_roundtrip() {
        let (x, y) = make_noisy_clf_data();

        let mut tree = DecisionTreeModel::new(20, 2, 1);
        tree.ccp_alpha = 0.02;
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        let n_nodes_orig = tree.nodes.len();
        let preds_orig = tree.predict_clf(&x);

        let json = tree.to_json().unwrap();
        let tree2 = DecisionTreeModel::from_json(&json).unwrap();

        assert_eq!(tree2.nodes.len(), n_nodes_orig, "roundtrip should preserve node count");
        assert!((tree2.ccp_alpha - 0.02).abs() < 1e-15, "roundtrip should preserve ccp_alpha");

        let preds2 = tree2.predict_clf(&x);
        assert_eq!(preds_orig, preds2, "roundtrip should preserve predictions");
    }

    #[test]
    fn test_ccp_alpha_serialization_roundtrip_reg() {
        let (x, y_vec) = make_reg_data();
        let y = DVector::from_vec(y_vec);

        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.ccp_alpha = 0.02;
        tree.fit_reg(&x, &y, None, Criterion::MSE).unwrap();

        let n_nodes_orig = tree.nodes.len();
        let preds_orig = tree.predict_reg(&x);

        let json = tree.to_json().unwrap();
        let tree2 = DecisionTreeModel::from_json(&json).unwrap();

        assert_eq!(tree2.nodes.len(), n_nodes_orig, "roundtrip should preserve node count");
        assert!((tree2.ccp_alpha - 0.02).abs() < 1e-15, "roundtrip should preserve ccp_alpha");

        let preds2 = tree2.predict_reg(&x);
        assert_eq!(preds_orig.as_slice(), preds2.as_slice(), "roundtrip should preserve predictions");
    }

    #[test]
    fn test_ccp_alpha_n_samples_populated() {
        // Verify that n_samples is correctly set on all nodes.
        let (x, y) = make_clf_data();

        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

        // Root should have n_samples == total training samples.
        assert_eq!(tree.nodes[0].n_samples, 100, "root n_samples should equal training size");

        // Every node should have n_samples > 0.
        for (i, node) in tree.nodes.iter().enumerate() {
            assert!(node.n_samples > 0, "node {} should have n_samples > 0", i);
        }

        // For split nodes, left + right n_samples should equal parent.
        for (i, node) in tree.nodes.iter().enumerate() {
            if let NodeKind::Split { left, right, .. } = &node.kind {
                let parent_n = node.n_samples;
                let child_sum = tree.nodes[*left].n_samples + tree.nodes[*right].n_samples;
                assert_eq!(
                    parent_n, child_sum,
                    "node {} n_samples={} but children sum to {}",
                    i, parent_n, child_sum
                );
            }
        }
    }
}

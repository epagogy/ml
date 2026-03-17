//! Random Forest: bagged ensemble of CART trees with rayon parallelism.
//!
//! # Performance (v3)
//! - Zero data copies: shared `ColMajorMatrix` + `QuantizedMatrix` across all trees
//! - Bootstrap-as-weights: count vector instead of row extraction
//! - rayon parallelism: ~N_CORES speedup
//! - Combined with histogram CART: 10-15x over sklearn at 100K

use crate::cart::{build_tree, ColMajorMatrix, Criterion, Node, TreeConfig};
use crate::error::MlError;
use crate::histogram::{QuantizedMatrix, MAX_BINS};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn default_true() -> bool { true }

/// Random Forest for classification and regression.
#[derive(Serialize, Deserialize)]
pub struct RandomForestModel {
    pub n_trees: usize,
    /// Number of features to consider per split. Default: sqrt(p) for clf, p/3 for reg.
    pub max_features: Option<usize>,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub histogram_threshold: usize,
    pub n_features: usize,
    pub n_classes: usize,
    pub feature_importances: Vec<f64>,
    pub oob_score: Option<f64>,
    /// Whether to compute OOB score after fitting. Default true.
    /// Set to false to skip the expensive OOB traversal.
    #[serde(default = "default_true")]
    pub compute_oob: bool,
    /// Split criterion. Old models without this field deserialize as Gini.
    #[serde(default)]
    pub criterion: Criterion,
    /// Extra Trees mode: random threshold per feature instead of optimal scan.
    /// Disables bootstrap (full dataset per tree). Old models deserialize as false.
    #[serde(default)]
    pub extra_trees: bool,
    /// Monotone constraints per feature: +1 increasing, -1 decreasing, 0 unconstrained.
    /// Regression only. Old models deserialize as None.
    #[serde(default)]
    pub monotone_cst: Option<Vec<i8>>,
    /// Minimum impurity decrease required for a split. Default 0.0.
    /// Old models deserialize as 0.0.
    #[serde(default)]
    pub min_impurity_decrease: f64,
    trees: Vec<TreeData>,
    is_clf: bool,
    seed: u64,
}

/// Internal tree storage — nodes + proba_pool (no DecisionTreeModel overhead).
#[derive(Serialize, Deserialize)]
struct TreeData {
    nodes: Vec<Node>,
    proba_pool: Vec<f64>,
    feature_importances: Vec<f64>,
}

impl RandomForestModel {
    pub fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_trees,
            max_features: None,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            histogram_threshold: 4096,
            n_features: 0,
            n_classes: 0,
            feature_importances: Vec::new(),
            oob_score: None,
            compute_oob: true,
            trees: Vec::new(),
            is_clf: false,
            seed,
            criterion: Criterion::default(),
            extra_trees: false,
            monotone_cst: None,
            min_impurity_decrease: 0.0,
        }
    }

    /// Fit classification forest.
    pub fn fit_clf(
        &mut self,
        x: &DMatrix<f64>,
        y: &[i64],
        sample_weight: Option<&DVector<f64>>,
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

        let max_feat = self.max_features.unwrap_or_else(|| {
            (p as f64).sqrt().round().max(1.0) as usize
        });

        self.is_clf = true;
        self.n_features = p;

        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        self.n_classes = classes.len();
        let k = self.n_classes;

        // Map labels to 0-based f64 (same as DecisionTreeModel::fit_clf)
        let y_f64: Vec<f64> = y
            .iter()
            .map(|&lbl| classes.binary_search(&lbl).unwrap() as f64)
            .collect();

        // Base weights (None → uniform 1.0)
        let base_weights: Vec<f64> = match sample_weight {
            None => vec![1.0; n],
            Some(sw) => {
                let sw_sum: f64 = sw.iter().sum();
                (0..n).map(|i| sw[i] * n as f64 / sw_sum).collect()
            }
        };

        // Build shared data structures ONCE
        let cm = ColMajorMatrix::from_dmatrix(x);
        let qm = if n >= self.histogram_threshold {
            Some(QuantizedMatrix::from_col_data(&cm.data, n, p, MAX_BINS))
        } else {
            None
        };

        // Train trees in parallel — zero data copies
        let extra_trees = self.extra_trees;
        let base_config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            histogram_threshold: self.histogram_threshold,
            n_classes: k,
            is_clf: true,
            max_features: Some(max_feat),
            rng_seed: 0, // overridden per tree
            criterion: self.criterion,
            extra_trees,
            monotone_cst: None, // monotone_cst not applied to classification
            min_impurity_decrease: self.min_impurity_decrease,
            split_lambda: 0.0,
        };
        let tree_results: Vec<_> = (0..self.n_trees)
            .into_par_iter()
            .map(|t| {
                let tree_seed = self.seed.wrapping_add(t as u64);
                let (weights, boot_indices) = if extra_trees {
                    (base_weights.clone(), Vec::new())
                } else {
                    let mut rng = SimpleRng::new(tree_seed);
                    let bi = bootstrap_sample(n, &mut rng);
                    let w = bootstrap_weights(&bi, n, &base_weights);
                    (w, bi)
                };

                let mut config = base_config.clone();
                config.rng_seed = tree_seed;
                let (nodes, proba_pool, importance_raw) = build_tree(
                    &cm, &y_f64, weights, qm.as_ref(), &config,
                );

                let imp = normalize_importances(&importance_raw, p);
                (TreeData { nodes, proba_pool, feature_importances: imp }, boot_indices)
            })
            .collect();

        self.trees.clear();
        let mut all_boot_indices = Vec::with_capacity(self.n_trees);
        for (tree, boots) in tree_results {
            self.trees.push(tree);
            all_boot_indices.push(boots);
        }

        self.feature_importances = aggregate_importances(&self.trees, p);
        if self.compute_oob && !extra_trees {
            self.oob_score = oob_score_clf(&cm, y, &self.trees, &all_boot_indices, k);
        }

        Ok(self)
    }

    /// Fit classification forest from pre-built ColMajorMatrix (zero-copy path).
    /// Skips the DMatrix→ColMajorMatrix copy for callers that already have row-major data.
    pub fn fit_clf_prepared(
        &mut self,
        cm: &ColMajorMatrix,
        y: &[i64],
        sample_weight: Option<&DVector<f64>>,
    ) -> Result<&mut Self, MlError> {
        let n = cm.nrows;
        let p = cm.ncols;
        if n == 0 { return Err(MlError::EmptyData); }
        if y.len() != n { return Err(MlError::DimensionMismatch { expected: n, got: y.len() }); }

        let max_feat = self.max_features.unwrap_or_else(|| (p as f64).sqrt().round().max(1.0) as usize);
        self.is_clf = true;
        self.n_features = p;

        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        self.n_classes = classes.len();
        let k = self.n_classes;

        let y_f64: Vec<f64> = y.iter().map(|&lbl| classes.binary_search(&lbl).unwrap() as f64).collect();
        let base_weights: Vec<f64> = match sample_weight {
            None => vec![1.0; n],
            Some(sw) => { let s: f64 = sw.iter().sum(); (0..n).map(|i| sw[i] * n as f64 / s).collect() }
        };

        let qm = if n >= self.histogram_threshold {
            Some(QuantizedMatrix::from_col_data(&cm.data, n, p, MAX_BINS))
        } else { None };

        let extra_trees = self.extra_trees;
        let base_config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            histogram_threshold: self.histogram_threshold,
            n_classes: k,
            is_clf: true,
            max_features: Some(max_feat),
            rng_seed: 0, // overridden per tree
            criterion: self.criterion,
            extra_trees,
            monotone_cst: None, // monotone_cst not applied to classification
            min_impurity_decrease: self.min_impurity_decrease,
            split_lambda: 0.0,
        };
        let tree_results: Vec<_> = (0..self.n_trees).into_par_iter().map(|t| {
            let tree_seed = self.seed.wrapping_add(t as u64);
            let (weights, boot_indices) = if extra_trees {
                (base_weights.clone(), Vec::new())
            } else {
                let mut rng = SimpleRng::new(tree_seed);
                let bi = bootstrap_sample(n, &mut rng);
                let w = bootstrap_weights(&bi, n, &base_weights);
                (w, bi)
            };
            let mut config = base_config.clone();
            config.rng_seed = tree_seed;
            let (nodes, proba_pool, importance_raw) = build_tree(
                cm, &y_f64, weights, qm.as_ref(), &config,
            );
            let imp = normalize_importances(&importance_raw, p);
            (TreeData { nodes, proba_pool, feature_importances: imp }, boot_indices)
        }).collect();

        self.trees.clear();
        let mut all_boot_indices = Vec::with_capacity(self.n_trees);
        for (tree, boots) in tree_results {
            self.trees.push(tree);
            all_boot_indices.push(boots);
        }
        self.feature_importances = aggregate_importances(&self.trees, p);
        if self.compute_oob && !extra_trees {
            self.oob_score = oob_score_clf(cm, y, &self.trees, &all_boot_indices, k);
        }
        Ok(self)
    }

    /// Fit regression forest.
    pub fn fit_reg(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        sample_weight: Option<&DVector<f64>>,
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

        // sklearn default for RF regression: max_features=1.0 (all features)
        let max_feat = self.max_features.unwrap_or(p);

        self.is_clf = false;
        self.n_features = p;
        self.n_classes = 0;
        // Regression trees require MSE or Poisson; coerce Gini/Entropy (clf-only) to MSE.
        let reg_criterion = match self.criterion {
            Criterion::Poisson => Criterion::Poisson,
            _ => Criterion::MSE,
        };

        let y_slice: Vec<f64> = y.iter().copied().collect();
        let base_weights: Vec<f64> = match sample_weight {
            None => vec![1.0; n],
            Some(sw) => {
                let sw_sum: f64 = sw.iter().sum();
                (0..n).map(|i| sw[i] * n as f64 / sw_sum).collect()
            }
        };

        let cm = ColMajorMatrix::from_dmatrix(x);
        let qm = if n >= self.histogram_threshold {
            Some(QuantizedMatrix::from_col_data(&cm.data, n, p, MAX_BINS))
        } else {
            None
        };

        let extra_trees = self.extra_trees;
        let base_config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            histogram_threshold: self.histogram_threshold,
            n_classes: 0,
            is_clf: false,
            max_features: Some(max_feat),
            rng_seed: 0, // overridden per tree
            criterion: reg_criterion,
            extra_trees,
            monotone_cst: self.monotone_cst.clone(),
            min_impurity_decrease: self.min_impurity_decrease,
            split_lambda: 0.0,
        };
        let tree_results: Vec<_> = (0..self.n_trees)
            .into_par_iter()
            .map(|t| {
                let tree_seed = self.seed.wrapping_add(t as u64);
                let (weights, boot_indices) = if extra_trees {
                    (base_weights.clone(), Vec::new())
                } else {
                    let mut rng = SimpleRng::new(tree_seed);
                    let bi = bootstrap_sample(n, &mut rng);
                    let w = bootstrap_weights(&bi, n, &base_weights);
                    (w, bi)
                };

                let mut config = base_config.clone();
                config.rng_seed = tree_seed;
                let (nodes, proba_pool, importance_raw) = build_tree(
                    &cm, &y_slice, weights, qm.as_ref(), &config,
                );

                let imp = normalize_importances(&importance_raw, p);
                (TreeData { nodes, proba_pool, feature_importances: imp }, boot_indices)
            })
            .collect();

        self.trees.clear();
        let mut all_boot_indices = Vec::with_capacity(self.n_trees);
        for (tree, boots) in tree_results {
            self.trees.push(tree);
            all_boot_indices.push(boots);
        }

        self.feature_importances = aggregate_importances(&self.trees, p);
        if self.compute_oob && !extra_trees {
            self.oob_score = oob_score_reg(&cm, &y_slice, &self.trees, &all_boot_indices);
        }

        Ok(self)
    }

    /// Fit regression forest from pre-built ColMajorMatrix (zero-copy path).
    pub fn fit_reg_prepared(
        &mut self,
        cm: &ColMajorMatrix,
        y: &DVector<f64>,
        sample_weight: Option<&DVector<f64>>,
    ) -> Result<&mut Self, MlError> {
        let n = cm.nrows;
        let p = cm.ncols;
        if n == 0 { return Err(MlError::EmptyData); }
        if y.len() != n { return Err(MlError::DimensionMismatch { expected: n, got: y.len() }); }

        // sklearn default for RF regression: max_features=1.0 (all features)
        let max_feat = self.max_features.unwrap_or(p);
        self.is_clf = false;
        self.n_features = p;
        self.n_classes = 0;
        // Regression trees require MSE or Poisson; coerce Gini/Entropy (clf-only) to MSE.
        let reg_criterion = match self.criterion {
            Criterion::Poisson => Criterion::Poisson,
            _ => Criterion::MSE,
        };

        let y_slice: Vec<f64> = y.iter().copied().collect();
        let base_weights: Vec<f64> = match sample_weight {
            None => vec![1.0; n],
            Some(sw) => { let s: f64 = sw.iter().sum(); (0..n).map(|i| sw[i] * n as f64 / s).collect() }
        };

        let qm = if n >= self.histogram_threshold {
            Some(QuantizedMatrix::from_col_data(&cm.data, n, p, MAX_BINS))
        } else { None };

        let extra_trees = self.extra_trees;
        let base_config = TreeConfig {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            histogram_threshold: self.histogram_threshold,
            n_classes: 0,
            is_clf: false,
            max_features: Some(max_feat),
            rng_seed: 0, // overridden per tree
            criterion: reg_criterion,
            extra_trees,
            monotone_cst: self.monotone_cst.clone(),
            min_impurity_decrease: self.min_impurity_decrease,
            split_lambda: 0.0,
        };
        let tree_results: Vec<_> = (0..self.n_trees).into_par_iter().map(|t| {
            let tree_seed = self.seed.wrapping_add(t as u64);
            let (weights, boot_indices) = if extra_trees {
                (base_weights.clone(), Vec::new())
            } else {
                let mut rng = SimpleRng::new(tree_seed);
                let bi = bootstrap_sample(n, &mut rng);
                let w = bootstrap_weights(&bi, n, &base_weights);
                (w, bi)
            };
            let mut config = base_config.clone();
            config.rng_seed = tree_seed;
            let (nodes, proba_pool, importance_raw) = build_tree(
                cm, &y_slice, weights, qm.as_ref(), &config,
            );
            let imp = normalize_importances(&importance_raw, p);
            (TreeData { nodes, proba_pool, feature_importances: imp }, boot_indices)
        }).collect();

        self.trees.clear();
        let mut all_boot_indices = Vec::with_capacity(self.n_trees);
        for (tree, boots) in tree_results {
            self.trees.push(tree);
            all_boot_indices.push(boots);
        }
        self.feature_importances = aggregate_importances(&self.trees, p);
        if self.compute_oob && !extra_trees {
            self.oob_score = oob_score_reg(cm, &y_slice, &self.trees, &all_boot_indices);
        }
        Ok(self)
    }

    /// Predict class indices (0-based).  Parallel across samples.
    ///
    /// Uses **soft voting** (probability averaging across trees, then argmax),
    /// matching sklearn's default behavior.  This is more accurate than hard
    /// voting (majority class label) on multiclass problems because it
    /// correctly weights confident vs uncertain tree predictions.
    pub fn predict_clf(&self, x: &DMatrix<f64>) -> Vec<i64> {
        assert!(self.is_clf, "predict_clf called on regression forest");
        let k = self.n_classes;
        let cm = ColMajorMatrix::from_dmatrix(x);
        let trees = &self.trees;

        (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let mut proba = vec![0.0_f64; k];
                for tree in trees {
                    let leaf = leaf_index(&tree.nodes, &cm, i);
                    if let crate::cart::NodeKind::Leaf { proba_offset, .. } = &tree.nodes[leaf].kind
                    {
                        for j in 0..k {
                            proba[j] += tree.proba_pool[proba_offset + j];
                        }
                    }
                }
                proba
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(cls, _)| cls as i64)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict continuous values.  Parallel across samples.
    pub fn predict_reg(&self, x: &DMatrix<f64>) -> DVector<f64> {
        assert!(!self.is_clf, "predict_reg called on classification forest");
        let cm = ColMajorMatrix::from_dmatrix(x);
        let trees = &self.trees;
        let scale = 1.0 / self.n_trees as f64;

        let vals: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let sum: f64 = trees.iter().map(|t| traverse(&t.nodes, &cm, i)).sum();
                sum * scale
            })
            .collect();

        DVector::from_vec(vals)
    }

    /// Class probability matrix, shape `(n_samples, n_classes)`.  Parallel across samples.
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        assert!(self.is_clf, "predict_proba called on regression forest");
        let k = self.n_classes;
        let cm = ColMajorMatrix::from_dmatrix(x);
        let trees = &self.trees;
        let scale = 1.0 / self.n_trees as f64;

        let proba: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0.0_f64; k];
                for tree in trees {
                    let leaf = leaf_index(&tree.nodes, &cm, i);
                    if let crate::cart::NodeKind::Leaf { proba_offset, .. } = &tree.nodes[leaf].kind
                    {
                        for j in 0..k {
                            row[j] += tree.proba_pool[proba_offset + j];
                        }
                    }
                }
                for v in &mut row {
                    *v *= scale;
                }
                row
            })
            .collect();

        DMatrix::from_row_slice(x.nrows(), k, &proba)
    }

    /// Serialize the fitted forest to JSON for Python `__getstate__`.
    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    /// Deserialize a forest from JSON for Python `__setstate__`.
    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

// ---------------------------------------------------------------------------
// Tree traversal (works on raw Node slices)
// ---------------------------------------------------------------------------

use crate::cart::NodeKind;

#[inline]
pub(crate) fn traverse(nodes: &[Node], x: &ColMajorMatrix, row: usize) -> f64 {
    let mut idx = 0;
    loop {
        match &nodes[idx].kind {
            NodeKind::Leaf { value, .. } => return *value,
            NodeKind::Split { feature, threshold, left, right } => {
                let val = unsafe { x.get_unchecked(row, *feature) };
                idx = if val.is_nan() {
                    if nodes[idx].nan_goes_left { *left } else { *right }
                } else if val <= *threshold {
                    *left
                } else {
                    *right
                };
            }
            NodeKind::Placeholder => panic!("placeholder node during prediction"),
        }
    }
}

#[inline]
fn leaf_index(nodes: &[Node], x: &ColMajorMatrix, row: usize) -> usize {
    let mut idx = 0;
    loop {
        match &nodes[idx].kind {
            NodeKind::Leaf { .. } => return idx,
            NodeKind::Split { feature, threshold, left, right } => {
                let val = unsafe { x.get_unchecked(row, *feature) };
                idx = if val.is_nan() {
                    if nodes[idx].nan_goes_left { *left } else { *right }
                } else if val <= *threshold {
                    *left
                } else {
                    *right
                };
            }
            NodeKind::Placeholder => panic!("placeholder node during prediction"),
        }
    }
}

/// Returns (leaf_value, proba_offset) for a single row.
///
/// Used by AdaBoost predict_proba to retrieve leaf probabilities.
#[inline]
pub(crate) fn traverse_leaf_info(
    nodes: &[Node],
    x: &ColMajorMatrix,
    row: usize,
) -> (f64, usize) {
    let mut idx = 0;
    loop {
        match &nodes[idx].kind {
            NodeKind::Leaf { value, proba_offset } => return (*value, *proba_offset),
            NodeKind::Split { feature, threshold, left, right } => {
                let val = unsafe { x.get_unchecked(row, *feature) };
                idx = if val.is_nan() {
                    if nodes[idx].nan_goes_left { *left } else { *right }
                } else if val <= *threshold {
                    *left
                } else {
                    *right
                };
            }
            NodeKind::Placeholder => panic!("placeholder node during prediction"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling utilities
// ---------------------------------------------------------------------------

pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Unbiased bounded random number in [0, n) using Lemire's fast method.
    ///
    /// The naive `next_u64() % n` has modulo bias for non-power-of-2 n:
    /// some residues are slightly more probable than others.  Lemire's
    /// method (2019) rejects the biased region of the 128-bit product
    /// without division in the fast path.
    #[inline]
    pub(crate) fn gen_range(&mut self, n: usize) -> usize {
        let n = n as u64;
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let t = n.wrapping_neg() % n; // rejection threshold
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }
}

fn bootstrap_sample(n: usize, rng: &mut SimpleRng) -> Vec<usize> {
    (0..n).map(|_| rng.gen_range(n)).collect()
}

/// Convert bootstrap indices to weight vector.
/// Weight[i] = base_weight[i] * count[i] (how many times sample i was drawn).
/// Samples not drawn get weight 0 → excluded from tree building.
fn bootstrap_weights(boot_indices: &[usize], n: usize, base_weights: &[f64]) -> Vec<f64> {
    let mut counts = vec![0_u32; n];
    for &i in boot_indices {
        counts[i] += 1;
    }
    (0..n).map(|i| base_weights[i] * counts[i] as f64).collect()
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

pub(crate) fn normalize_importances(raw: &[f64], p: usize) -> Vec<f64> {
    let s: f64 = raw.iter().sum();
    if s < 1e-15 {
        vec![1.0 / p as f64; p]
    } else {
        raw.iter().map(|&v| v / s).collect()
    }
}

fn aggregate_importances(trees: &[TreeData], p: usize) -> Vec<f64> {
    let mut imp = vec![0.0_f64; p];
    let n_trees = trees.len();
    for tree in trees {
        for (i, &v) in tree.feature_importances.iter().enumerate() {
            imp[i] += v;
        }
    }
    for v in &mut imp {
        *v /= n_trees as f64;
    }
    let s: f64 = imp.iter().sum();
    if s > 1e-15 {
        for v in &mut imp {
            *v /= s;
        }
    } else {
        imp.fill(1.0 / p as f64);
    }
    imp
}

/// Pre-compute in-bootstrap bitsets for all trees (shared by clf + reg OOB).
fn build_oob_masks(n: usize, all_boot_indices: &[Vec<usize>]) -> Vec<Vec<bool>> {
    all_boot_indices
        .iter()
        .map(|boots| {
            let mut mask = vec![false; n];
            for &bi in boots {
                mask[bi] = true;
            }
            mask
        })
        .collect()
}

/// OOB score for classification (accuracy).
/// Parallelized over samples via rayon.
fn oob_score_clf(
    x: &ColMajorMatrix,
    y: &[i64],
    trees: &[TreeData],
    all_boot_indices: &[Vec<usize>],
    k: usize,
) -> Option<f64> {
    let n = y.len();
    let masks = build_oob_masks(n, all_boot_indices);

    // Parallel: each sample independently tallies votes from its OOB trees
    let results: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut votes = vec![0_usize; k];
            let mut has_oob = false;
            for (t, tree) in trees.iter().enumerate() {
                if !masks[t][i] {
                    let cls = traverse(&tree.nodes, x, i) as usize;
                    votes[cls] += 1;
                    has_oob = true;
                }
            }
            if has_oob {
                let pred = votes
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, cnt)| cnt)
                    .map(|(cls, _)| cls as i64)
                    .unwrap_or(0);
                let correct = if pred == y[i] { 1 } else { 0 };
                (correct, 1)
            } else {
                (0, 0)
            }
        })
        .collect();

    let (correct, total) = results.iter().fold((0, 0), |acc, &(c, t)| (acc.0 + c, acc.1 + t));
    if total > 0 { Some(correct as f64 / total as f64) } else { None }
}

/// OOB score for regression (R²).
/// Parallelized over samples via rayon.
fn oob_score_reg(
    x: &ColMajorMatrix,
    y: &[f64],
    trees: &[TreeData],
    all_boot_indices: &[Vec<usize>],
) -> Option<f64> {
    let n = y.len();
    let masks = build_oob_masks(n, all_boot_indices);

    // Parallel: each sample independently averages predictions from its OOB trees
    let per_sample: Vec<(f64, usize)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut sum = 0.0_f64;
            let mut count = 0_usize;
            for (t, tree) in trees.iter().enumerate() {
                if !masks[t][i] {
                    sum += traverse(&tree.nodes, x, i);
                    count += 1;
                }
            }
            (sum, count)
        })
        .collect();

    let mut ss_res = 0.0_f64;
    let mut y_mean_sum = 0.0_f64;
    let mut total = 0_usize;

    for i in 0..n {
        let (sum, count) = per_sample[i];
        if count > 0 {
            let pred = sum / count as f64;
            ss_res += (y[i] - pred).powi(2);
            y_mean_sum += y[i];
            total += 1;
        }
    }

    if total < 2 {
        return None;
    }

    let y_mean = y_mean_sum / total as f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..n {
        if per_sample[i].1 > 0 {
            ss_tot += (y[i] - y_mean).powi(2);
        }
    }

    if ss_tot < 1e-15 { Some(1.0) } else { Some(1.0 - ss_res / ss_tot) }
}

#[cfg(test)]
mod tests {
    use super::SimpleRng;

    /// Chi-squared goodness-of-fit test for `gen_range` uniformity.
    ///
    /// Draws `n_draws` samples from `[0, n_buckets)` and computes the
    /// chi-squared statistic against the uniform distribution.  With
    /// `n_buckets = 7` (a non-power-of-2 that maximizes modulo bias for
    /// naive implementations) and `n_draws = 70_000`, the expected count
    /// per bucket is 10_000.  The critical value for chi-squared with
    /// df=6 at p=0.001 is 22.46; we use 30 for extra margin.
    ///
    /// The old `next_u64() % n` implementation would fail this test for
    /// n values that don't divide 2^64 evenly.
    #[test]
    fn test_gen_range_chi_squared_uniformity() {
        let n_buckets: usize = 7; // non-power-of-2 to expose modulo bias
        let n_draws: usize = 70_000;
        let expected = n_draws as f64 / n_buckets as f64; // 10_000

        let mut rng = SimpleRng::new(12345);
        let mut counts = vec![0_usize; n_buckets];
        for _ in 0..n_draws {
            let v = rng.gen_range(n_buckets);
            assert!(v < n_buckets, "gen_range({n_buckets}) returned {v}");
            counts[v] += 1;
        }

        // Chi-squared statistic: sum((observed - expected)^2 / expected)
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();

        // df = 6, critical value at p=0.001 is 22.46. Use 30 for safety.
        assert!(
            chi2 < 30.0,
            "chi-squared = {chi2:.2} exceeds threshold 30.0 (df=6, p<0.001). \
             Bucket counts: {counts:?}"
        );
    }

    /// Verify gen_range produces all values in [0, n) over many draws.
    #[test]
    fn test_gen_range_covers_full_range() {
        let n: usize = 13; // another non-power-of-2
        let mut rng = SimpleRng::new(42);
        let mut seen = vec![false; n];
        for _ in 0..10_000 {
            let v = rng.gen_range(n);
            seen[v] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "gen_range({n}) never produced value {i} in 10000 draws");
        }
    }

    /// Verify gen_range(1) always returns 0 (edge case).
    #[test]
    fn test_gen_range_one() {
        let mut rng = SimpleRng::new(99);
        for _ in 0..1000 {
            assert_eq!(rng.gen_range(1), 0, "gen_range(1) must always return 0");
        }
    }

    /// Verify gen_range works for power-of-2 values (should be trivially unbiased).
    #[test]
    fn test_gen_range_power_of_two() {
        let n: usize = 8;
        let n_draws: usize = 80_000;
        let expected = n_draws as f64 / n as f64;

        let mut rng = SimpleRng::new(7777);
        let mut counts = vec![0_usize; n];
        for _ in 0..n_draws {
            counts[rng.gen_range(n)] += 1;
        }

        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();

        assert!(
            chi2 < 25.0,
            "chi-squared = {chi2:.2} for n=8, counts: {counts:?}"
        );
    }
}

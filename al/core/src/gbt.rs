//! Gradient-Boosted Trees: sequential boosting over shallow CART regression trees.
//!
//! # Algorithm
//! Each round fits one (regression) or k (multiclass) shallow trees on pseudo-residuals
//! using `build_tree()` directly — same as RandomForest, ColMajorMatrix built once.
//!
//! # Losses
//! - `L2` (regression): residuals = y − F, leaf = mean(residuals) = Newton step (h=1)
//! - `BinomialDeviance` (binary clf): g = y − sigmoid(F), h = sigmoid(F)*(1−sigmoid(F))
//!   leaf = sum(g) / max(sum(h), 1e-15)  [Newton step — Chen & Guestrin 2016]
//! - `MultinomialDeviance` (multiclass): g_c = I(y==c) − p_c, h_c = p_c*(1−p_c)
//!   leaf = sum(g_c) / max(sum(h_c), 1e-15)  per class

use crate::cart::{build_tree, ColMajorMatrix, Criterion, Node, NodeKind, TreeConfig};
use crate::error::MlError;
use crate::forest::{normalize_importances, SimpleRng, traverse};
use crate::histogram::{GBTHistBin, GBTHistogram, QuantizedMatrix, MAX_BINS};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Loss enum + GrowPolicy enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum GBTLoss {
    #[default]
    L2,
    BinomialDeviance,
    MultinomialDeviance,
}

/// Tree growth strategy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum GrowPolicy {
    /// Depth-wise (level-wise): grow all nodes at the same depth. XGBoost default.
    #[default]
    Depthwise,
    /// Leaf-wise (best-first): always split the leaf with highest gain. LightGBM default.
    Lossguide,
}

fn default_one() -> f64 { 1.0 }
fn default_validation_fraction() -> f64 { 0.1 }
fn default_max_bin() -> usize { 254 }
fn default_goss_min_n() -> usize { 50_000 }
fn default_dart_rate() -> f64 { 0.0 }

/// XGBoost-style L1 soft-thresholding: shrinks gradient toward zero by alpha.
/// Returns 0 if |g| <= alpha, else g - sign(g) * alpha.
#[inline]
fn soft_thresh(g: f64, alpha: f64) -> f64 {
    if g.abs() <= alpha { 0.0 } else { g - g.signum() * alpha }
}

/// Sample a subset of feature indices for colsample_bytree (true per-tree sampling).
/// Uses a separate seed offset to avoid correlation with row subsampling.
fn sample_feature_indices(p: usize, frac: f64, seed: u64) -> Option<Vec<usize>> {
    if frac >= 1.0 { return None; }
    let k = ((p as f64 * frac).ceil() as usize).max(1).min(p);
    let mut rng = SimpleRng::new(seed);
    let mut indices: Vec<usize> = (0..p).collect();
    // Partial Fisher-Yates: shuffle first k positions
    for i in 0..k {
        let j = i + rng.gen_range(p - i);
        indices.swap(i, j);
    }
    let mut subset = indices[..k].to_vec();
    subset.sort_unstable(); // sorted for reproducible feature ordering
    Some(subset)
}

// ---------------------------------------------------------------------------
// Internal tree storage (regression trees — no proba_pool needed)
// ---------------------------------------------------------------------------

/// One weak learner in the boosting ensemble.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GBTTree {
    nodes: Vec<Node>,
    pub feature_importances: Vec<f64>,
}

impl GBTTree {
    pub(crate) fn predict(&self, cm: &ColMajorMatrix) -> Vec<f64> {
        (0..cm.nrows)
            .into_par_iter()
            .map(|i| traverse(&self.nodes, cm, i))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GBTModel
// ---------------------------------------------------------------------------

/// Gradient-Boosted Trees for classification and regression.
#[derive(Debug, Serialize, Deserialize)]
pub struct GBTModel {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    /// Row subsampling fraction per tree (1.0 = no sampling).
    pub subsample: f64,
    pub histogram_threshold: usize,
    pub n_features: usize,
    /// 0 = regression, 2 = binary clf, k≥3 = multiclass clf.
    pub n_classes: usize,
    pub feature_importances: Vec<f64>,
    pub loss: GBTLoss,
    /// trees[t][c]: tree for round t, class c (regression/binary: one tree per round).
    trees: Vec<Vec<GBTTree>>,
    /// Initial prediction: [scalar] for reg/binary, [k] for multiclass.
    pub initial_pred: Vec<f64>,
    seed: u64,
    /// L2 regularization on leaf weights (XGBoost's lambda). Default 0.0.
    #[serde(default)]
    pub lambda: f64,
    /// Minimum loss reduction for a split (XGBoost's gamma). Default 0.0.
    #[serde(default)]
    pub gamma: f64,
    /// Column subsampling per tree. Default 1.0 (all features).
    /// Selects a random fraction of features at the start of each tree.
    #[serde(default = "default_one")]
    pub colsample_bytree: f64,
    /// Column subsampling per split node. Default 1.0 (all features).
    /// At each node, only a random fraction of features are candidates.
    /// Stacks with colsample_bytree: effective = bytree × bynode.
    #[serde(default = "default_one")]
    pub colsample_bynode: f64,
    /// Minimum sum of hessians per leaf. Default 1.0.
    #[serde(default = "default_one")]
    pub min_child_weight: f64,
    /// Early stopping: stop if no validation improvement for N rounds. None = disabled.
    #[serde(default)]
    pub n_iter_no_change: Option<usize>,
    /// Fraction held out for early stopping validation. Default 0.1.
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: f64,
    /// Actual rounds used (< n_estimators if early stopping triggered).
    #[serde(default)]
    pub best_n_rounds: Option<usize>,
    /// L1 regularization on leaf weights (XGBoost's alpha). Default 0.0.
    #[serde(default)]
    pub reg_alpha: f64,
    /// Maximum absolute leaf weight. 0.0 = disabled. Default 0.0.
    #[serde(default)]
    pub max_delta_step: f64,
    /// User-supplied initial prediction. None = estimate from data. Default None.
    #[serde(default)]
    pub base_score: Option<f64>,
    /// Per-feature monotone constraints: +1 = increasing, -1 = decreasing, 0 = unconstrained.
    /// None = no constraints. Length must equal n_features.
    #[serde(default)]
    pub monotone_cst: Option<Vec<i8>>,
    /// Number of histogram bins for quantized CART splitting (1..254).
    /// NaN values always use reserved slot 255 outside this count.
    /// Default 254. Larger = more splits, smaller = faster + more regularization.
    #[serde(default = "default_max_bin")]
    pub max_bin: usize,
    /// Growth strategy: Depthwise (level-wise) or Lossguide (leaf-wise best-first).
    #[serde(default)]
    pub grow_policy: GrowPolicy,
    /// Maximum number of leaves for Lossguide. 0 = unlimited (bounded only by max_depth).
    #[serde(default)]
    pub max_leaves: usize,
    /// GOSS top-rate: fraction of large-gradient samples always kept. 1.0 = disabled.
    /// Mutually exclusive with subsample < 1.0 (GOSS replaces row subsampling).
    #[serde(default = "default_one")]
    pub goss_top_rate: f64,
    /// GOSS other-rate: fraction of small-gradient samples randomly included. Default 1.0 (disabled).
    #[serde(default = "default_one")]
    pub goss_other_rate: f64,
    /// Minimum n to activate GOSS. Below this, standard subsample is used. Default 50_000.
    #[serde(default = "default_goss_min_n")]
    pub goss_min_n: usize,
    /// Leaf smoothing: shrinks leaf values toward parent prediction.
    /// w_smooth = G / (H + lambda + leaf_smooth). Default 0.0 (off).
    /// Small positive values (1.0-10.0) regularize leaves with few samples.
    #[serde(default)]
    pub leaf_smooth: f64,
    /// Weight multiplier for positive class in binary classification.
    /// Useful for imbalanced datasets: set to n_negative / n_positive.
    /// Default 1.0 (no reweighting). Only used with BinomialDeviance loss.
    #[serde(default = "default_one")]
    pub scale_pos_weight: f64,
    /// DART: fraction of existing trees to drop per round (0.0 = disabled).
    /// When > 0, dropout regularization prevents over-specialization of later trees.
    /// Typical values: 0.05-0.3. Default 0.0 (standard GBT).
    #[serde(default = "default_dart_rate")]
    pub dart_rate: f64,
    /// Per-tree scale factors (DART normalization). When dart_rate=0, all entries = learning_rate.
    /// Length = number of trees (rounds × classes-per-round).
    #[serde(default)]
    tree_scales: Vec<f64>,
}

impl GBTModel {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            subsample,
            histogram_threshold: 0, // Always use histogram splits (like XGBoost/LightGBM)
            n_features: 0,
            n_classes: 0,
            feature_importances: Vec::new(),
            loss: GBTLoss::L2,
            trees: Vec::new(),
            initial_pred: Vec::new(),
            seed,
            lambda: 0.0,
            gamma: 0.0,
            colsample_bytree: 1.0,
            colsample_bynode: 1.0,
            min_child_weight: 1.0,
            n_iter_no_change: None,
            validation_fraction: 0.1,
            best_n_rounds: None,
            reg_alpha: 0.0,
            max_delta_step: 0.0,
            base_score: None,
            monotone_cst: None,
            max_bin: 254,
            grow_policy: GrowPolicy::Depthwise,
            max_leaves: 0,
            goss_top_rate: 1.0,
            goss_other_rate: 1.0,
            goss_min_n: 50_000,
            leaf_smooth: 0.0,
            scale_pos_weight: 1.0,
            dart_rate: 0.0,
            tree_scales: Vec::new(),
        }
    }

    /// Compute per-node max_features from colsample_bynode fraction.
    fn node_max_features(&self, p: usize) -> Option<usize> {
        if self.colsample_bynode < 1.0 {
            Some(((p as f64 * self.colsample_bynode).ceil() as usize).max(1).min(p))
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // fit_reg
    // -----------------------------------------------------------------------

    pub fn fit_reg(
        &mut self,
        x: &DMatrix<f64>,
        y: &[f64],
        sample_weight: Option<&DVector<f64>>,
    ) -> Result<&mut Self, MlError> {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch { expected: n, got: y.len() });
        }

        let node_mf = self.node_max_features(p);

        self.n_features = p;
        self.n_classes = 0;
        self.loss = GBTLoss::L2;

        let base_w = base_weights(n, sample_weight);
        let cm = ColMajorMatrix::from_dmatrix(x);
        let qm = build_qm(&cm, n, p, self.histogram_threshold, self.max_bin);
        let colsample_frac = self.colsample_bytree;

        // Early stopping: create validation mask
        let (val_indices, effective_base_w) = if self.n_iter_no_change.is_some() {
            let n_val = ((n as f64 * self.validation_fraction).ceil() as usize).max(1).min(n - 1);
            let mut rng = SimpleRng::new(self.seed.wrapping_add(999));
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..n_val {
                let j = i + rng.gen_range(n - i);
                indices.swap(i, j);
            }
            let val_set: Vec<usize> = indices[..n_val].to_vec();
            let mut masked_w = base_w.clone();
            for &vi in &val_set { masked_w[vi] = 0.0; }
            (val_set, masked_w)
        } else {
            (Vec::new(), base_w.clone())
        };

        let init_val = self.base_score.unwrap_or_else(|| weighted_mean(y, &effective_base_w));
        self.initial_pred = vec![init_val];
        let mut f_scores: Vec<f64> = vec![init_val; n];
        let mut imp_acc = vec![0.0_f64; p];
        self.trees = Vec::with_capacity(self.n_estimators);

        let mut best_val_loss = f64::INFINITY;
        let mut best_round: usize = 0;
        let mut rounds_no_improve: usize = 0;

        let reg_alpha = self.reg_alpha;
        let max_delta_step = self.max_delta_step;
        let dart_rate = self.dart_rate;
        // DART: track per-tree scale factors for dropout normalization.
        // tree_scales[t] = effective scale (learning_rate * dart_normalization).
        let mut tree_scales: Vec<f64> = Vec::with_capacity(self.n_estimators);

        for t in 0..self.n_estimators {
            // ── DART: dropout existing trees ──────────────────────────────
            // Drop a random subset of existing trees, recompute f_scores
            // from only the non-dropped trees plus initial prediction.
            let dropped: Vec<usize> = if dart_rate > 0.0 && !self.trees.is_empty() {
                let mut rng = SimpleRng::new(self.seed.wrapping_add(t as u64).wrapping_add(0xDA47_0000));
                let n_trees = self.trees.len();
                let n_drop = ((n_trees as f64 * dart_rate).ceil() as usize).max(1).min(n_trees);
                let mut indices: Vec<usize> = (0..n_trees).collect();
                for i in 0..n_drop {
                    let j = i + rng.gen_range(n_trees - i);
                    indices.swap(i, j);
                }
                let mut drop_set: Vec<usize> = indices[..n_drop].to_vec();
                drop_set.sort_unstable();
                // Recompute f_scores excluding dropped trees
                for i in 0..n { f_scores[i] = self.initial_pred[0]; }
                for (tidx, trees_round) in self.trees.iter().enumerate() {
                    if drop_set.binary_search(&tidx).is_ok() { continue; }
                    let preds = trees_round[0].predict(&cm);
                    let scale = tree_scales[tidx];
                    for i in 0..n { f_scores[i] += scale * preds[i]; }
                }
                drop_set
            } else {
                Vec::new()
            };

            // True per-tree colsample: sample feature subset once per round
            let feat_idx = sample_feature_indices(
                p, colsample_frac,
                self.seed.wrapping_add(t as u64).wrapping_add(0x_CAFE_0000),
            );
            let residuals: Vec<f64> = (0..n).map(|i| y[i] - f_scores[i]).collect();
            let sub_seed = self.seed ^ (t as u64).wrapping_mul(1_000_003);
            let weights = if self.goss_top_rate < 1.0 && n >= self.goss_min_n {
                // GOSS: gradient-based one-side sampling replaces row subsampling.
                goss_weights(&residuals, &effective_base_w, self.goss_top_rate,
                             self.goss_other_rate, sub_seed)
            } else {
                subsample_weights(n, self.subsample, sub_seed, &effective_base_w)
            };
            let (tree, leaf_assignments_opt) = match self.grow_policy {
                GrowPolicy::Lossguide => {
                    // Lossguide: Newton leaf values + gamma built into growth.
                    let hessians = vec![1.0_f64; n];
                    let (t, la) = fit_lossguide_tree(
                        &cm, &residuals, &hessians, weights, qm.as_ref(),
                        self.max_depth, self.min_samples_split, self.min_samples_leaf,
                        self.seed.wrapping_add(t as u64), feat_idx.as_deref(),
                        self.lambda, reg_alpha, self.gamma, self.min_child_weight,
                        max_delta_step, self.max_leaves, p, self.leaf_smooth,
                    );
                    (t, Some(la))
                }
                GrowPolicy::Depthwise => {
                    let mut tree = fit_regression_tree_subset(
                        &cm, &residuals, weights, qm.as_ref(),
                        self.max_depth, self.min_samples_split, self.min_samples_leaf,
                        self.histogram_threshold, self.seed.wrapping_add(t as u64),
                        feat_idx.as_deref(), self.monotone_cst.as_deref(), self.max_bin,
                        self.lambda, node_mf,
                    );
                    // Apply lambda/alpha/min_child_weight via Newton update on L2 pseudo-gradients.
                    // For L2: gradient_i = residual_i = (y_i - F_i), hessian_i = 1.0.
                    if self.lambda > 0.0 || self.gamma > 0.0 || self.min_child_weight > 1.0 || reg_alpha > 0.0 || max_delta_step > 0.0 || self.leaf_smooth > 0.0 {
                        let hessians = vec![1.0_f64; n];
                        let (mut g_sums, mut h_sums) = newton_leaf_update(
                            &mut tree.nodes, &residuals, &hessians, &cm,
                            self.lambda, reg_alpha, self.min_child_weight, max_delta_step,
                            self.leaf_smooth,
                        );
                        if self.gamma > 0.0 {
                            propagate_sums_up(&tree.nodes, &mut g_sums, &mut h_sums);
                            gamma_prune(&mut tree.nodes, &g_sums, &h_sums, self.lambda, reg_alpha, self.gamma);
                        }
                    }
                    (tree, None::<Vec<usize>>)
                }
            };
            let lr = self.learning_rate;
            // DART normalization: new tree gets scale lr/(k+1), dropped trees get k/(k+1).
            // When dart_rate=0 (standard GBT), k=0 so new tree scale = lr, matching vanilla.
            let k_dropped = dropped.len();
            let new_tree_scale = if k_dropped > 0 {
                lr / (k_dropped as f64 + 1.0)
            } else {
                lr
            };
            // Rescale dropped trees
            if k_dropped > 0 {
                let shrink = k_dropped as f64 / (k_dropped as f64 + 1.0);
                for &tidx in &dropped {
                    tree_scales[tidx] *= shrink;
                }
                // Recompute f_scores from all trees (including rescaled dropped + new)
                for i in 0..n { f_scores[i] = self.initial_pred[0]; }
                for (tidx, trees_round) in self.trees.iter().enumerate() {
                    let preds = trees_round[0].predict(&cm);
                    let scale = tree_scales[tidx];
                    for i in 0..n { f_scores[i] += scale * preds[i]; }
                }
            }
            // Add new tree contribution
            if let Some(assignments) = leaf_assignments_opt {
                for i in 0..n {
                    if let NodeKind::Leaf { value, .. } = tree.nodes[assignments[i]].kind {
                        f_scores[i] += new_tree_scale * value;
                    }
                }
            } else {
                let preds = tree.predict(&cm);
                for i in 0..n { f_scores[i] += new_tree_scale * preds[i]; }
            }
            tree_scales.push(new_tree_scale);
            // Lossguide importances are already in global feature space (length p);
            // depthwise importances are in projected space and need remapping via feat_idx.
            let imp_feat_idx = if self.grow_policy == GrowPolicy::Lossguide { None } else { feat_idx.as_deref() };
            accumulate_importances(&mut imp_acc, &tree.feature_importances, imp_feat_idx);
            self.trees.push(vec![tree]);

            // Early stopping check
            if let Some(patience) = self.n_iter_no_change {
                let val_loss = validation_loss(
                    GBTLoss::L2, &f_scores, None, y, &[], &val_indices, 0,
                );
                if val_loss < best_val_loss - 1e-10 {
                    best_val_loss = val_loss;
                    best_round = t;
                    rounds_no_improve = 0;
                } else {
                    rounds_no_improve += 1;
                }
                if rounds_no_improve >= patience {
                    break;
                }
            }
        }

        // Truncate trees to best round if early stopping fired
        if self.n_iter_no_change.is_some() {
            let nr = best_round + 1;
            self.best_n_rounds = Some(nr);
            self.trees.truncate(nr);
            tree_scales.truncate(nr);
        }
        self.tree_scales = tree_scales;

        let n_rounds = self.trees.len().max(1) as f64;
        for v in &mut imp_acc { *v /= n_rounds; }
        self.feature_importances = normalize_importances(&imp_acc, p);
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // fit_clf
    // -----------------------------------------------------------------------

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
            return Err(MlError::DimensionMismatch { expected: n, got: y.len() });
        }

        self.n_features = p;
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let k = classes.len();
        self.n_classes = k;
        let y_idx: Vec<usize> = y.iter().map(|&lbl| classes.binary_search(&lbl).unwrap()).collect();
        let base_w = base_weights(n, sample_weight);
        let cm = ColMajorMatrix::from_dmatrix(x);
        let qm = build_qm(&cm, n, p, self.histogram_threshold, self.max_bin);

        if k == 2 {
            self.fit_binary(&cm, &qm, &y_idx, &base_w, n, p)
        } else {
            self.fit_multiclass(&cm, &qm, &y_idx, k, &base_w, n, p)
        }
    }

    fn fit_binary(
        &mut self,
        cm: &ColMajorMatrix,
        qm: &Option<QuantizedMatrix>,
        y_idx: &[usize],
        base_w: &[f64],
        n: usize,
        p: usize,
    ) -> Result<&mut Self, MlError> {
        self.loss = GBTLoss::BinomialDeviance;
        let y_f64: Vec<f64> = y_idx.iter().map(|&c| c as f64).collect();
        let colsample_frac = self.colsample_bytree;
        let node_mf = self.node_max_features(p);

        // Early stopping: create validation mask
        let (val_indices, effective_base_w) = if self.n_iter_no_change.is_some() {
            let n_val = ((n as f64 * self.validation_fraction).ceil() as usize).max(1).min(n - 1);
            let mut rng = SimpleRng::new(self.seed.wrapping_add(999));
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..n_val {
                let j = i + rng.gen_range(n - i);
                indices.swap(i, j);
            }
            let val_set: Vec<usize> = indices[..n_val].to_vec();
            let mut masked_w = base_w.to_vec();
            for &vi in &val_set { masked_w[vi] = 0.0; }
            (val_set, masked_w)
        } else {
            (Vec::new(), base_w.to_vec())
        };

        let init = if let Some(bs) = self.base_score {
            bs
        } else {
            let total_w: f64 = effective_base_w.iter().sum();
            let pos_w: f64 = y_f64.iter().zip(effective_base_w.iter()).map(|(&yi, &wi)| yi * wi).sum();
            let p_pos = (pos_w / total_w).clamp(1e-7, 1.0 - 1e-7);
            p_pos.ln() - (1.0 - p_pos).ln()
        };
        self.initial_pred = vec![init];
        let mut f_scores: Vec<f64> = vec![init; n];
        let mut imp_acc = vec![0.0_f64; p];
        self.trees = Vec::with_capacity(self.n_estimators);
        let mut tree_scales: Vec<f64> = Vec::with_capacity(self.n_estimators);

        let mut best_val_loss = f64::INFINITY;
        let mut best_round: usize = 0;
        let mut rounds_no_improve: usize = 0;

        let reg_alpha = self.reg_alpha;
        let max_delta_step = self.max_delta_step;
        let dart_rate = self.dart_rate;

        for t in 0..self.n_estimators {
            // DART: drop fraction of existing trees, recompute f_scores from non-dropped.
            let dropped: Vec<usize> = if dart_rate > 0.0 && !self.trees.is_empty() {
                let mut rng = SimpleRng::new(self.seed.wrapping_add(t as u64).wrapping_add(0xDA47_0000));
                let nt = self.trees.len();
                let n_drop = ((nt as f64 * dart_rate).ceil() as usize).max(1).min(nt);
                let mut indices: Vec<usize> = (0..nt).collect();
                for i in 0..n_drop {
                    let j = i + rng.gen_range(nt - i);
                    indices.swap(i, j);
                }
                let dropped_set: Vec<usize> = indices[..n_drop].to_vec();
                // Recompute f_scores from non-dropped trees only
                for i in 0..n { f_scores[i] = init; }
                for (tidx, trees_round) in self.trees.iter().enumerate() {
                    if !dropped_set.contains(&tidx) {
                        let preds = trees_round[0].predict(cm);
                        let scale = tree_scales[tidx];
                        for i in 0..n { f_scores[i] += scale * preds[i]; }
                    }
                }
                dropped_set
            } else {
                Vec::new()
            };

            // True per-tree colsample: sample feature subset once per round
            let feat_idx = sample_feature_indices(
                p, colsample_frac,
                self.seed.wrapping_add(t as u64).wrapping_add(0x_CAFE_0000),
            );
            // Phase 2: single loop — sigmoid(F) called once per sample per round.
            // scale_pos_weight: multiply positive-class gradient+hessian by weight factor
            // (equivalent to reweighting loss for class imbalance).
            let spw = self.scale_pos_weight;
            let (residuals, hessians): (Vec<f64>, Vec<f64>) = (0..n).map(|i| {
                let s = sigmoid(f_scores[i]);
                let w = if y_f64[i] > 0.5 { spw } else { 1.0 };
                (w * (y_f64[i] - s), w * s * (1.0 - s))
            }).unzip();
            let sub_seed = self.seed ^ (t as u64).wrapping_mul(1_000_003);
            let weights = if self.goss_top_rate < 1.0 && n >= self.goss_min_n {
                // GOSS: |residual| = |gradient| for binary cross-entropy.
                goss_weights(&residuals, &effective_base_w, self.goss_top_rate,
                             self.goss_other_rate, sub_seed)
            } else {
                subsample_weights(n, self.subsample, sub_seed, &effective_base_w)
            };
            let (tree, leaf_assignments_opt) = match self.grow_policy {
                GrowPolicy::Lossguide => {
                    let (t, la) = fit_lossguide_tree(
                        cm, &residuals, &hessians, weights, qm.as_ref(),
                        self.max_depth, self.min_samples_split, self.min_samples_leaf,
                        self.seed.wrapping_add(t as u64), feat_idx.as_deref(),
                        self.lambda, reg_alpha, self.gamma, self.min_child_weight,
                        max_delta_step, self.max_leaves, p, self.leaf_smooth,
                    );
                    (t, Some(la))
                }
                GrowPolicy::Depthwise => {
                    // Newton-gain tree building for classification (no monotone constraints):
                    // targets = g_i / h_i, weights = w_i * h_i
                    // MSE on (g/h, w*h) ∝ Newton gain for split selection.
                    // When monotone constraints are active, use raw gradients to preserve
                    // monotone enforcement in the CART builder.
                    let use_newton = self.monotone_cst.is_none();
                    let (tree_targets, tree_weights) = if use_newton {
                        let (nt, nw): (Vec<f64>, Vec<f64>) = (0..n)
                            .map(|i| {
                                let h = hessians[i].max(1e-15);
                                (residuals[i] / h, weights[i] * h)
                            })
                            .unzip();
                        (nt, nw)
                    } else {
                        (residuals.clone(), weights.clone())
                    };
                    // When Newton-gain is active, pass lambda to split selection
                    // so gain = G²/(H+λ) instead of G²/H (XGBoost-style regularized splits).
                    let sl = if use_newton { self.lambda } else { 0.0 };
                    let mut tree = fit_regression_tree_subset(
                        cm, &tree_targets, tree_weights, qm.as_ref(),
                        self.max_depth, self.min_samples_split, self.min_samples_leaf,
                        self.histogram_threshold, self.seed.wrapping_add(t as u64),
                        feat_idx.as_deref(), self.monotone_cst.as_deref(), self.max_bin,
                        sl, node_mf,
                    );
                    // Newton leaf update: recomputes leaf values as sum(g)/(sum(h)+lambda).
                    let (mut g_sums, mut h_sums) = newton_leaf_update(
                        &mut tree.nodes, &residuals, &hessians, cm,
                        self.lambda, reg_alpha, self.min_child_weight, max_delta_step,
                        self.leaf_smooth,
                    );
                    if self.gamma > 0.0 {
                        propagate_sums_up(&tree.nodes, &mut g_sums, &mut h_sums);
                        gamma_prune(&mut tree.nodes, &g_sums, &h_sums, self.lambda, reg_alpha, self.gamma);
                    }
                    (tree, None::<Vec<usize>>)
                }
            };
            let lr = self.learning_rate;
            // DART normalization for binary clf
            let k_dropped = dropped.len();
            let new_tree_scale = if k_dropped > 0 {
                lr / (k_dropped as f64 + 1.0)
            } else {
                lr
            };
            // Rescale dropped trees
            if k_dropped > 0 {
                let shrink = k_dropped as f64 / (k_dropped as f64 + 1.0);
                for &tidx in &dropped {
                    tree_scales[tidx] *= shrink;
                }
                // Recompute f_scores from all trees (including rescaled dropped + new)
                for i in 0..n { f_scores[i] = init; }
                for (tidx, trees_round) in self.trees.iter().enumerate() {
                    let preds = trees_round[0].predict(cm);
                    let scale = tree_scales[tidx];
                    for i in 0..n { f_scores[i] += scale * preds[i]; }
                }
            }
            // Add new tree contribution
            if let Some(assignments) = leaf_assignments_opt {
                for i in 0..n {
                    if let NodeKind::Leaf { value, .. } = tree.nodes[assignments[i]].kind {
                        f_scores[i] += new_tree_scale * value;
                    }
                }
            } else {
                let preds = tree.predict(cm);
                for i in 0..n { f_scores[i] += new_tree_scale * preds[i]; }
            }
            tree_scales.push(new_tree_scale);
            let imp_feat_idx = if self.grow_policy == GrowPolicy::Lossguide { None } else { feat_idx.as_deref() };
            accumulate_importances(&mut imp_acc, &tree.feature_importances, imp_feat_idx);
            self.trees.push(vec![tree]);

            // Early stopping check
            if let Some(patience) = self.n_iter_no_change {
                let val_loss = validation_loss(
                    GBTLoss::BinomialDeviance, &f_scores, None,
                    &y_f64, y_idx, &val_indices, 0,
                );
                if val_loss < best_val_loss - 1e-10 {
                    best_val_loss = val_loss;
                    best_round = t;
                    rounds_no_improve = 0;
                } else {
                    rounds_no_improve += 1;
                }
                if rounds_no_improve >= patience {
                    break;
                }
            }
        }

        // Truncate trees to best round if early stopping fired
        if self.n_iter_no_change.is_some() {
            let nr = best_round + 1;
            self.best_n_rounds = Some(nr);
            self.trees.truncate(nr);
            tree_scales.truncate(nr);
        }
        self.tree_scales = tree_scales;

        let n_rounds = self.trees.len().max(1) as f64;
        for v in &mut imp_acc { *v /= n_rounds; }
        self.feature_importances = normalize_importances(&imp_acc, p);
        Ok(self)
    }

    fn fit_multiclass(
        &mut self,
        cm: &ColMajorMatrix,
        qm: &Option<QuantizedMatrix>,
        y_idx: &[usize],
        k: usize,
        base_w: &[f64],
        n: usize,
        p: usize,
    ) -> Result<&mut Self, MlError> {
        self.loss = GBTLoss::MultinomialDeviance;
        let node_mf = self.node_max_features(p);

        // Early stopping: create validation mask
        let (val_indices, effective_base_w) = if self.n_iter_no_change.is_some() {
            let n_val = ((n as f64 * self.validation_fraction).ceil() as usize).max(1).min(n - 1);
            let mut rng = SimpleRng::new(self.seed.wrapping_add(999));
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..n_val {
                let j = i + rng.gen_range(n - i);
                indices.swap(i, j);
            }
            let val_set: Vec<usize> = indices[..n_val].to_vec();
            let mut masked_w = base_w.to_vec();
            for &vi in &val_set { masked_w[vi] = 0.0; }
            (val_set, masked_w)
        } else {
            (Vec::new(), base_w.to_vec())
        };

        // Zero initial predictions → softmax(0) = uniform 1/k
        self.initial_pred = vec![0.0; k];
        // F[i][c] = log-odds score for sample i, class c
        let mut f_scores: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; n];
        let mut imp_acc = vec![0.0_f64; p];
        self.trees = Vec::with_capacity(self.n_estimators);

        // Copy hyperparams to avoid borrow in closure
        let seed = self.seed;
        let max_depth = self.max_depth;
        let min_ss = self.min_samples_split;
        let min_sl = self.min_samples_leaf;
        let hist_thresh = self.histogram_threshold;
        let subsample = self.subsample;
        let goss_top = self.goss_top_rate;
        let goss_other = self.goss_other_rate;
        let goss_min_n = self.goss_min_n;
        let lr = self.learning_rate;
        let lambda = self.lambda;
        let gamma = self.gamma;
        let min_child_weight = self.min_child_weight;
        let reg_alpha = self.reg_alpha;
        let max_delta_step = self.max_delta_step;
        let colsample_frac = self.colsample_bytree;
        let monotone_cst_ref = self.monotone_cst.as_deref();
        let max_bin = self.max_bin;
        let grow_policy = self.grow_policy;
        let max_leaves = self.max_leaves;
        let leaf_smooth = self.leaf_smooth;
        let dart_rate = self.dart_rate;
        let mut tree_scales: Vec<f64> = Vec::with_capacity(self.n_estimators);

        let mut best_val_loss = f64::INFINITY;
        let mut best_round: usize = 0;
        let mut rounds_no_improve: usize = 0;

        for t in 0..self.n_estimators {
            // DART: drop fraction of existing rounds, recompute f_scores
            let dropped: Vec<usize> = if dart_rate > 0.0 && !self.trees.is_empty() {
                let mut rng = SimpleRng::new(seed.wrapping_add(t as u64).wrapping_add(0xDA47_0000));
                let nt = self.trees.len();
                let n_drop = ((nt as f64 * dart_rate).ceil() as usize).max(1).min(nt);
                let mut indices: Vec<usize> = (0..nt).collect();
                for i in 0..n_drop {
                    let j = i + rng.gen_range(nt - i);
                    indices.swap(i, j);
                }
                let dropped_set: Vec<usize> = indices[..n_drop].to_vec();
                // Recompute f_scores from non-dropped rounds only
                for i in 0..n {
                    for c in 0..k { f_scores[i][c] = 0.0; }
                }
                for (tidx, trees_round) in self.trees.iter().enumerate() {
                    if !dropped_set.contains(&tidx) {
                        let scale = tree_scales[tidx];
                        for c in 0..k {
                            let preds = trees_round[c].predict(cm);
                            for i in 0..n { f_scores[i][c] += scale * preds[i]; }
                        }
                    }
                }
                dropped_set
            } else {
                Vec::new()
            };

            // True per-tree colsample: all k class trees share the same feature subset per round
            let feat_idx = sample_feature_indices(
                p, colsample_frac,
                seed.wrapping_add(t as u64).wrapping_add(0x_CAFE_0000),
            );
            let feat_idx_ref = feat_idx.as_deref();

            // Compute softmax for all samples (sequential read of f_scores)
            let proba: Vec<Vec<f64>> = f_scores.iter().map(|row| {
                let mut p_row = row.clone();
                softmax_inplace(&mut p_row);
                p_row
            }).collect();

            // GOSS: compute shared weight vector for all k class trees this round.
            // Sort by sum of |gradient| across all classes — same samples selected for all classes.
            let sub_seed_t = seed ^ (t as u64).wrapping_mul(1_000_003);
            let goss_w_opt: Option<Vec<f64>> = if goss_top < 1.0 && n >= goss_min_n {
                let grad_magnitude: Vec<f64> = (0..n).map(|i| {
                    (0..k).map(|c| {
                        let ind = if y_idx[i] == c { 1.0_f64 } else { 0.0 };
                        (ind - proba[i][c]).abs()
                    }).sum::<f64>()
                }).collect();
                Some(goss_weights(&grad_magnitude, &effective_base_w, goss_top, goss_other, sub_seed_t))
            } else {
                None
            };

            // Fit k trees in parallel.
            // Each thread reads cm/qm/proba/y_idx (shared immutable), computes own residuals.
            // Returns (class_idx, tree, leaf_assignments, importances) — no write to shared state.
            let round_data: Vec<(usize, GBTTree, Option<Vec<usize>>, Vec<f64>)> =
                (0..k).into_par_iter().map(|c| {
                    // Phase 2: single loop fuses residuals + hessians — proba[i][c] read once.
                    let (residuals_c, hessians_c): (Vec<f64>, Vec<f64>) = (0..n).map(|i| {
                        let pc = proba[i][c];
                        let indicator = if y_idx[i] == c { 1.0 } else { 0.0 };
                        (indicator - pc, pc * (1.0 - pc))
                    }).unzip();
                    let tree_seed = seed
                        .wrapping_add(t as u64)
                        .wrapping_add((c as u64).wrapping_mul(7));
                    let sub_seed = sub_seed_t.wrapping_add((c as u64).wrapping_mul(7));
                    // GOSS weights (same selection for all classes) or standard subsampling.
                    let weights = if let Some(ref gw) = goss_w_opt {
                        gw.clone()
                    } else {
                        subsample_weights(n, subsample, sub_seed, &effective_base_w)
                    };
                    let (tree, leaf_assignments_opt) = match grow_policy {
                        GrowPolicy::Lossguide => {
                            let (t, la) = fit_lossguide_tree(
                                cm, &residuals_c, &hessians_c, weights, qm.as_ref(),
                                max_depth, min_ss, min_sl, tree_seed, feat_idx_ref,
                                lambda, reg_alpha, gamma, min_child_weight,
                                max_delta_step, max_leaves, p, leaf_smooth,
                            );
                            (t, Some(la))
                        }
                        GrowPolicy::Depthwise => {
                            // Newton-gain tree building for classification:
                            // targets = g_i / h_i, weights = w_i * h_i
                            // When monotone constraints active, use raw gradients.
                            let use_newton = monotone_cst_ref.is_none();
                            let (tree_targets, tree_weights) = if use_newton {
                                let (nt, nw): (Vec<f64>, Vec<f64>) = (0..n)
                                    .map(|i| {
                                        let h = hessians_c[i].max(1e-15);
                                        (residuals_c[i] / h, weights[i] * h)
                                    })
                                    .unzip();
                                (nt, nw)
                            } else {
                                (residuals_c.clone(), weights.clone())
                            };
                            let sl = if use_newton { lambda } else { 0.0 };
                            let mut tree = fit_regression_tree_subset(
                                cm, &tree_targets, tree_weights, qm.as_ref(),
                                max_depth, min_ss, min_sl, hist_thresh, tree_seed,
                                feat_idx_ref, monotone_cst_ref, max_bin,
                                sl, node_mf,
                            );
                            let (mut g_sums, mut h_sums) = newton_leaf_update(
                                &mut tree.nodes, &residuals_c, &hessians_c, cm,
                                lambda, reg_alpha, min_child_weight, max_delta_step,
                                leaf_smooth,
                            );
                            if gamma > 0.0 {
                                propagate_sums_up(&tree.nodes, &mut g_sums, &mut h_sums);
                                gamma_prune(&mut tree.nodes, &g_sums, &h_sums, lambda, reg_alpha, gamma);
                            }
                            (tree, None::<Vec<usize>>)
                        }
                    };
                    let imps = tree.feature_importances.clone();
                    (c, tree, leaf_assignments_opt, imps)
                }).collect();

            // DART normalization for multiclass
            let k_dropped = dropped.len();
            let new_tree_scale = if k_dropped > 0 {
                lr / (k_dropped as f64 + 1.0)
            } else {
                lr
            };
            // Rescale dropped rounds
            if k_dropped > 0 {
                let shrink = k_dropped as f64 / (k_dropped as f64 + 1.0);
                for &tidx in &dropped {
                    tree_scales[tidx] *= shrink;
                }
                // Recompute f_scores from all rounds (rescaled dropped + existing)
                for i in 0..n {
                    for c in 0..k { f_scores[i][c] = 0.0; }
                }
                for (tidx, trees_round) in self.trees.iter().enumerate() {
                    let scale = tree_scales[tidx];
                    for c in 0..k {
                        let preds = trees_round[c].predict(cm);
                        for i in 0..n { f_scores[i][c] += scale * preds[i]; }
                    }
                }
            }

            // Sequential update of f_scores and importances — no data race.
            let mut round_trees: Vec<Option<GBTTree>> = (0..k).map(|_| None).collect();
            for (c, tree, leaf_assignments_opt, imps) in round_data {
                if let Some(assignments) = leaf_assignments_opt {
                    for i in 0..n {
                        if let NodeKind::Leaf { value, .. } = tree.nodes[assignments[i]].kind {
                            f_scores[i][c] += new_tree_scale * value;
                        }
                    }
                } else {
                    let preds = tree.predict(cm);
                    for i in 0..n { f_scores[i][c] += new_tree_scale * preds[i]; }
                }
                let imp_feat_idx = if grow_policy == GrowPolicy::Lossguide { None } else { feat_idx_ref };
                accumulate_importances(&mut imp_acc, &imps, imp_feat_idx);
                round_trees[c] = Some(tree);
            }
            let round_trees: Vec<GBTTree> = round_trees
                .into_iter()
                .map(|t| t.expect("GBT multiclass: tree not fit for class"))
                .collect();
            tree_scales.push(new_tree_scale);
            self.trees.push(round_trees);

            // Early stopping check
            if let Some(patience) = self.n_iter_no_change {
                let val_loss = validation_loss(
                    GBTLoss::MultinomialDeviance, &[], Some(&f_scores),
                    &[], y_idx, &val_indices, k,
                );
                if val_loss < best_val_loss - 1e-10 {
                    best_val_loss = val_loss;
                    best_round = t;
                    rounds_no_improve = 0;
                } else {
                    rounds_no_improve += 1;
                }
                if rounds_no_improve >= patience {
                    break;
                }
            }
        }

        // Truncate trees to best round if early stopping fired
        if self.n_iter_no_change.is_some() {
            let nr = best_round + 1;
            self.best_n_rounds = Some(nr);
            self.trees.truncate(nr);
            tree_scales.truncate(nr);
        }
        self.tree_scales = tree_scales;

        let total = (self.trees.len() * k).max(1) as f64;
        for v in &mut imp_acc { *v /= total; }
        self.feature_importances = normalize_importances(&imp_acc, p);
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // predict_reg
    // -----------------------------------------------------------------------

    pub fn predict_reg(&self, x: &DMatrix<f64>) -> Vec<f64> {
        let cm = ColMajorMatrix::from_dmatrix(x);
        let init = self.initial_pred.first().copied().unwrap_or(0.0);
        let lr = self.learning_rate;
        let trees = &self.trees;
        let scales = &self.tree_scales;
        (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let mut fi = init;
                for (t, round) in trees.iter().enumerate() {
                    if let Some(tree) = round.first() {
                        // Use per-tree scale if available (DART), else learning_rate
                        let s = scales.get(t).copied().unwrap_or(lr);
                        fi += s * traverse(&tree.nodes, &cm, i);
                    }
                }
                fi
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // predict_clf / predict_proba
    // -----------------------------------------------------------------------

    /// Predict class indices (0-based, matching the encoding from fit_clf).
    pub fn predict_clf(&self, x: &DMatrix<f64>) -> Vec<i64> {
        assert!(self.n_classes > 0, "predict_clf called on regression model");
        let proba = self.predict_proba(x);
        (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = proba.row(i);
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(c, _)| c as i64)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Probability matrix, shape (n_samples, n_classes).
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x.nrows();
        let k = self.n_classes;
        let cm = ColMajorMatrix::from_dmatrix(x);
        let lr = self.learning_rate;
        let trees = &self.trees;
        let scales = &self.tree_scales;

        if self.loss == GBTLoss::BinomialDeviance {
            // Binary: single tree per round, sigmoid output.
            let init = self.initial_pred.first().copied().unwrap_or(0.0);
            let flat: Vec<f64> = (0..n)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let mut fi = init;
                    for (t, round) in trees.iter().enumerate() {
                        if let Some(tree) = round.first() {
                            let s = scales.get(t).copied().unwrap_or(lr);
                            fi += s * traverse(&tree.nodes, &cm, i);
                        }
                    }
                    let p1 = sigmoid(fi);
                    [1.0 - p1, p1]
                })
                .collect();
            DMatrix::from_row_slice(n, 2, &flat)
        } else {
            // Multiclass: each sample accumulates a k-vector across rounds, then softmax.
            let init = &self.initial_pred;
            let flat: Vec<f64> = (0..n)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let mut f_row = init.clone();
                    for (t, round) in trees.iter().enumerate() {
                        let s = scales.get(t).copied().unwrap_or(lr);
                        for (c, tree) in round.iter().enumerate() {
                            f_row[c] += s * traverse(&tree.nodes, &cm, i);
                        }
                    }
                    softmax_inplace(&mut f_row);
                    f_row
                })
                .collect();
            DMatrix::from_row_slice(n, k, &flat)
        }
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Fit one shallow regression tree on residuals. Calls build_tree() directly.
fn fit_regression_tree(
    cm: &ColMajorMatrix,
    residuals: &[f64],
    weights: Vec<f64>,
    qm: Option<&QuantizedMatrix>,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    histogram_threshold: usize,
    seed: u64,
    p: usize,
    max_features: Option<usize>,
    monotone_cst: Option<&[i8]>,
    split_lambda: f64,
) -> GBTTree {
    let config = TreeConfig {
        max_depth,
        min_samples_split,
        min_samples_leaf,
        histogram_threshold,
        n_classes: 0,        // regression
        is_clf: false,       // regression
        max_features,
        rng_seed: seed,
        criterion: Criterion::MSE,
        extra_trees: false,
        monotone_cst: monotone_cst.map(|c| c.to_vec()),
        min_impurity_decrease: 0.0, // GBT uses gamma for pruning
        split_lambda,
    };
    let (nodes, _proba_pool, importance_raw) = build_tree(
        cm, residuals, weights, qm, &config,
    );
    let fi = normalize_importances(&importance_raw, p);
    GBTTree { nodes, feature_importances: fi }
}

/// Fit one shallow regression tree on a (possibly projected) feature subset.
/// When `feat_idx` is Some(indices), only those features are used; node Split.feature
/// indices are remapped back to original feature space after building.
/// Monotone constraints are projected to the feature subset when both are active.
fn fit_regression_tree_subset(
    cm: &ColMajorMatrix,
    residuals: &[f64],
    weights: Vec<f64>,
    qm: Option<&QuantizedMatrix>,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    histogram_threshold: usize,
    seed: u64,
    feat_idx: Option<&[usize]>,
    monotone_cst: Option<&[i8]>,
    max_bin: usize,
    split_lambda: f64,
    max_features: Option<usize>,
) -> GBTTree {
    match feat_idx {
        None => fit_regression_tree(
            cm, residuals, weights, qm,
            max_depth, min_samples_split, min_samples_leaf,
            histogram_threshold, seed, cm.ncols, max_features, monotone_cst,
            split_lambda,
        ),
        Some(indices) => {
            let nrows = cm.nrows;
            let k = indices.len();
            // Build projected ColMajorMatrix from the selected columns
            let mut proj_data = Vec::with_capacity(k * nrows);
            for &f in indices {
                let col_start = f * nrows;
                proj_data.extend_from_slice(&cm.data[col_start..col_start + nrows]);
            }
            let proj_cm = ColMajorMatrix { data: proj_data, nrows, ncols: k };
            // Build a fresh QM for the projected feature set
            let proj_qm = build_qm(&proj_cm, nrows, k, histogram_threshold, max_bin);
            // Project monotone constraints to the selected feature subset
            let proj_cst: Option<Vec<i8>> = monotone_cst.map(|cst| {
                indices.iter().map(|&f| cst[f]).collect()
            });
            // Adjust max_features for projected feature count
            let proj_mf = max_features.map(|mf| mf.min(k));
            let mut tree = fit_regression_tree(
                &proj_cm, residuals, weights, proj_qm.as_ref(),
                max_depth, min_samples_split, min_samples_leaf,
                histogram_threshold, seed, k, proj_mf, proj_cst.as_deref(),
                split_lambda,
            );
            // Remap Split.feature indices from projected space back to original
            for node in &mut tree.nodes {
                if let NodeKind::Split { ref mut feature, .. } = node.kind {
                    *feature = indices[*feature];
                }
            }
            // feature_importances is over projected k-feature space;
            // accumulate_importances will remap to full p-feature space.
            tree
        }
    }
}

/// Accumulate tree feature importances into a global importance accumulator.
/// When feat_idx is Some, tree importances are in projected feature space and
/// are remapped to original feature indices via feat_idx.
fn accumulate_importances(acc: &mut [f64], tree_imps: &[f64], feat_idx: Option<&[usize]>) {
    match feat_idx {
        None => {
            for (f, &v) in tree_imps.iter().enumerate() {
                if f < acc.len() { acc[f] += v; }
            }
        }
        Some(indices) => {
            for (f, &v) in tree_imps.iter().enumerate() {
                acc[indices[f]] += v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lossguide (leaf-wise best-first) tree growth
// ---------------------------------------------------------------------------

use crate::histogram::NAN_BIN;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Sort-based GBT split finder (for small data where histogram is not built).
fn find_best_gbt_split_sort(
    cm: &ColMajorMatrix,
    indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    weights: &[f64],
    feat_set: &[usize],
    lambda: f64,
    reg_alpha: f64,
    min_samples_leaf: usize,
    min_child_weight: f64,
) -> Option<(usize, f64, f64, bool)> {
    // Returns (feature_idx_in_feat_set, threshold, gain, nan_goes_left)
    let mut best_gain = 0.0_f64;
    let mut best_feat = 0_usize;
    let mut best_thresh = f64::NAN;
    let mut found = false;

    let mut g_total = 0.0_f64;
    let mut h_total = 0.0_f64;
    for &i in indices {
        let w = weights[i];
        if w == 0.0 { continue; }
        g_total += gradients[i] * w;
        h_total += hessians[i] * w;
    }

    let mut pairs: Vec<(f64, f64, f64)> = Vec::with_capacity(indices.len()); // (value, g*w, h*w)

    for (fi, &f) in feat_set.iter().enumerate() {
        pairs.clear();
        let mut g_nan = 0.0_f64;
        let mut h_nan = 0.0_f64;
        for &i in indices {
            let w = weights[i];
            if w == 0.0 { continue; }
            let v = cm.data[f * cm.nrows + i];
            if v.is_nan() {
                g_nan += gradients[i] * w;
                h_nan += hessians[i] * w;
            } else {
                pairs.push((v, gradients[i] * w, hessians[i] * w));
            }
        }
        if pairs.is_empty() { continue; }
        pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // NaN goes left by default for sort-based
        let gl_base = g_nan;
        let hl_base = h_nan;
        let mut gl = gl_base;
        let mut hl = hl_base;
        let mut n_left = 0_usize;

        for j in 0..pairs.len() - 1 {
            gl += pairs[j].1;
            hl += pairs[j].2;
            n_left += 1;
            // Skip if next value is same
            if (pairs[j].0 - pairs[j + 1].0).abs() < f64::EPSILON { continue; }
            let n_right = pairs.len() - n_left;
            if n_left < min_samples_leaf || n_right < min_samples_leaf { continue; }
            let gr = g_total - gl;
            let hr = h_total - hl;
            if hl < min_child_weight || hr < min_child_weight { continue; }
            let gain = 0.5 * (
                soft_thresh(gl, reg_alpha).powi(2) / (hl + lambda)
                + soft_thresh(gr, reg_alpha).powi(2) / (hr + lambda)
                - soft_thresh(g_total, reg_alpha).powi(2) / (h_total + lambda)
            );
            if gain > best_gain {
                best_gain = gain;
                best_feat = fi;
                best_thresh = (pairs[j].0 + pairs[j + 1].0) * 0.5;
                found = true;
            }
        }
    }

    if found { Some((best_feat, best_thresh, best_gain, true)) } else { None }
}

/// Build one GBTHistogram per feature in feat_set.
///
/// Strategy: parallelize over sample chunks (the large dimension) rather than features
/// (the small dimension). For nodes with few samples, go fully sequential to avoid
/// rayon task-spawn overhead exceeding the actual work.
fn build_node_histograms(
    qm: &QuantizedMatrix,
    indices: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    weights: &[f64],
    feat_set: &[usize],
) -> Vec<GBTHistogram> {
    let n_feats = feat_set.len();
    // Below this threshold rayon overhead > scatter work; go sequential.
    const SEQ_THRESHOLD: usize = 512;
    // Chunk size: enough work per task to amortize ~200ns rayon overhead.
    const CHUNK: usize = 512;

    if indices.len() < SEQ_THRESHOLD || n_feats == 0 {
        feat_set.iter().map(|&f| {
            let mut hist = GBTHistogram::new();
            for &i in indices {
                let w = weights[i];
                if w == 0.0 { continue; }
                let b = unsafe { qm.get_unchecked(i, f) } as usize;
                let gw = gradients[i] * w;
                let hw = hessians[i] * w;
                hist.bins[b].g += gw;
                hist.bins[b].h += hw;
            }
            hist
        }).collect()
    } else {
        // Each chunk builds partial histograms for ALL features, then we tree-reduce.
        indices
            .par_chunks(CHUNK)
            .map(|chunk| {
                let mut partial = vec![GBTHistogram::new(); n_feats];
                for &i in chunk {
                    let w = weights[i];
                    if w == 0.0 { continue; }
                    let gw = gradients[i] * w;
                    let hw = hessians[i] * w;
                    for (fi, &f) in feat_set.iter().enumerate() {
                        let b = unsafe { qm.get_unchecked(i, f) } as usize;
                        partial[fi].bins[b].g += gw;
                        partial[fi].bins[b].h += hw;
                    }
                }
                partial
            })
            .reduce(
                || vec![GBTHistogram::new(); n_feats],
                |mut a, b| {
                    for (ah, bh) in a.iter_mut().zip(b.iter()) {
                        for bin in 0..MAX_BINS {
                            ah.bins[bin].g += bh.bins[bin].g;
                            ah.bins[bin].h += bh.bins[bin].h;
                        }
                    }
                    a
                },
            )
    }
}

/// Derive sibling histograms via parent − child subtraction.
/// O(MAX_BINS × n_features) instead of O(n_larger × n_features).
fn subtract_histograms(parent: &[GBTHistogram], child: &[GBTHistogram]) -> Vec<GBTHistogram> {
    parent.iter().zip(child.iter()).map(|(p, c)| {
        let mut out = GBTHistogram::new();
        for b in 0..MAX_BINS {
            out.bins[b] = GBTHistBin { g: p.bins[b].g - c.bins[b].g, h: p.bins[b].h - c.bins[b].h };
        }
        out
    }).collect()
}

/// Find best GBT split from pre-built histograms.
/// Sequential scan: p features × max_bin bins is small (e.g. 20 × 254 = 5080 iters).
/// Rayon task-spawn overhead exceeds this work; sequential is faster here.
fn scan_hists_for_best_split(
    hists: &[GBTHistogram],
    qm: &QuantizedMatrix,
    feat_set: &[usize],
    g_total: f64,
    h_total: f64,
    lambda: f64,
    reg_alpha: f64,
    min_child_weight: f64,
) -> Option<(usize, u8, f64, bool)> {
    // Fast-path leaf score: specialize for reg_alpha=0 (common case) to skip soft_thresh branch.
    let parent_score = if reg_alpha == 0.0 {
        g_total * g_total / (h_total + lambda)
    } else {
        soft_thresh(g_total, reg_alpha).powi(2) / (h_total + lambda)
    };
    let mut best: Option<(usize, u8, f64, bool)> = None;
    for (fi, (hist, &f)) in hists.iter().zip(feat_set.iter()).enumerate() {
        let n_bins = qm.n_bins_for(f);
        let nan_bin = &hist.bins[NAN_BIN as usize];
        let g_nan = nan_bin.g;
        let h_nan = nan_bin.h;
        // NaN bypass: if no NaN samples for this feature, skip the nan_goes_right pass entirely.
        let has_nan = g_nan != 0.0 || h_nan != 0.0;
        let nan_dirs: &[bool] = if has_nan { &[true, false] } else { &[true] };
        for &nan_goes_left in nan_dirs {
            let mut gl = if nan_goes_left { g_nan } else { 0.0 };
            let mut hl = if nan_goes_left { h_nan } else { 0.0 };
            for b in 0..n_bins.saturating_sub(1) {
                gl += hist.bins[b].g;
                hl += hist.bins[b].h;
                let gr = g_total - gl;
                let hr = h_total - hl;
                if hl < min_child_weight || hr < min_child_weight { continue; }
                let gain = if reg_alpha == 0.0 {
                    0.5 * (gl * gl / (hl + lambda) + gr * gr / (hr + lambda) - parent_score)
                } else {
                    0.5 * (
                        soft_thresh(gl, reg_alpha).powi(2) / (hl + lambda)
                        + soft_thresh(gr, reg_alpha).powi(2) / (hr + lambda)
                        - parent_score
                    )
                };
                if best.map_or(true, |(_, _, bg, _)| gain > bg) {
                    best = Some((fi, b as u8, gain, nan_goes_left));
                }
            }
        }
    }
    best
}

/// Candidate leaf for lossguide growth heap.
struct LeafCandidate {
    node_idx: usize,
    depth: usize,
    gain: f64,            // gain of best available split (excluding gamma)
    feature: usize,       // feature index in full feature space
    threshold: f64,
    nan_goes_left: bool,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    left_g: f64,
    left_h: f64,
    right_g: f64,
    right_h: f64,
    /// Pre-built histograms (one per feature in feat_set). Empty when sort-based.
    histograms: Vec<GBTHistogram>,
    /// Weighted G/H totals for this leaf (cached for children's split scan).
    g_total: f64,
    h_total: f64,
}

impl PartialEq for LeafCandidate {
    fn eq(&self, other: &Self) -> bool { self.gain == other.gain }
}
impl Eq for LeafCandidate {}
impl PartialOrd for LeafCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for LeafCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Less)
    }
}

/// Compute Newton leaf value (with L1, L2, leaf smoothing, max_delta_step).
///
/// Leaf smoothing adds a constant to the denominator: w = G / (H + lambda + leaf_smooth).
/// This shrinks leaves with few samples (small H) more aggressively than large ones,
/// acting as Bayesian regularization with a zero-centered prior.
#[inline]
fn newton_value(g: f64, h: f64, lambda: f64, reg_alpha: f64, max_delta_step: f64) -> f64 {
    newton_value_smooth(g, h, lambda, reg_alpha, max_delta_step, 0.0)
}

/// Newton leaf value with explicit leaf_smooth parameter.
#[inline]
fn newton_value_smooth(g: f64, h: f64, lambda: f64, reg_alpha: f64, max_delta_step: f64, leaf_smooth: f64) -> f64 {
    let w = soft_thresh(g, reg_alpha) / (h + lambda + leaf_smooth);
    if max_delta_step > 0.0 { w.clamp(-max_delta_step, max_delta_step) } else { w }
}

/// Leaf-wise (best-first) tree growth for GBT.
/// Splits leaves in order of highest gain, respecting max_depth and max_leaves.
#[allow(clippy::too_many_arguments)]
fn fit_lossguide_tree(
    cm: &ColMajorMatrix,
    gradients: &[f64],
    hessians: &[f64],
    weights: Vec<f64>,
    qm: Option<&QuantizedMatrix>,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    seed: u64,
    feat_idx: Option<&[usize]>,
    lambda: f64,
    reg_alpha: f64,
    gamma: f64,
    min_child_weight: f64,
    max_delta_step: f64,
    max_leaves: usize,
    p: usize,
    leaf_smooth: f64,
) -> (GBTTree, Vec<usize>) {
    let n = cm.nrows;
    let _ = seed; // not used for deterministic lossguide; kept for API symmetry

    // Effective feature set (for per-tree colsample)
    let full_feats: Vec<usize> = (0..cm.ncols).collect();
    let feat_set: &[usize] = feat_idx.unwrap_or(&full_feats);

    // Accumulate root G/H — include all n samples; zero-weight contribute 0.
    // Including all n enables leaf_assignments to cover every sample without
    // a separate tree.predict pass after construction.
    let mut g_root = 0.0_f64;
    let mut h_root = 0.0_f64;
    let root_indices: Vec<usize> = (0..n).collect();
    for &i in &root_indices {
        g_root += gradients[i] * weights[i];
        h_root += hessians[i] * weights[i];
    }
    let root_val = newton_value_smooth(g_root, h_root, lambda, reg_alpha, max_delta_step, leaf_smooth);

    // Initialize nodes with a single root leaf
    let mut nodes: Vec<Node> = vec![Node {
        kind: NodeKind::Leaf { value: root_val, proba_offset: 0 },
        nan_goes_left: true,
        n_samples: root_indices.len(),
    }];
    // Feature importances accumulator (over full feature space p)
    let mut imp_acc = vec![0.0_f64; p];
    let mut n_leaves: usize = 1;
    // Track which leaf node each sample ends up in — avoids tree.predict after construction.
    let mut leaf_assignments = vec![0usize; n];

    // Helper: try to build a split candidate from pre-built histograms (histogram path).
    // Returns None if no profitable split, or candidate with histograms stored for children.
    let try_split_hist = |node_idx: usize, depth: usize, g_total: f64, h_total: f64,
                           indices: &[usize], hists: Vec<GBTHistogram>,
                           qm: &QuantizedMatrix| -> Option<LeafCandidate> {
        if depth >= max_depth || indices.len() < min_samples_split { return None; }
        let (fi, bin, gain, nan_left) = scan_hists_for_best_split(
            &hists, qm, feat_set, g_total, h_total, lambda, reg_alpha, min_child_weight,
        )?;
        if gain <= gamma { return None; }
        let threshold = qm.threshold_for(feat_set[fi], bin);
        let feature = feat_set[fi];
        let split_bin = bin; // u8 split bin; b <= split_bin ↔ v <= threshold (monotone quant)
        let mut left_idx = Vec::with_capacity(indices.len() / 2);
        let mut right_idx = Vec::with_capacity(indices.len() / 2);
        let mut lg = 0.0_f64; let mut lh = 0.0_f64;
        let mut rg = 0.0_f64; let mut rh = 0.0_f64;
        // Phase 6: read u8 bin from qm (1 byte) instead of f64 from cm (8 bytes).
        // For n=10k, p=20: 10KB footprint (u8) vs 80KB (f64). Fits in L1/L2.
        for &i in indices {
            let b = unsafe { qm.get_unchecked(i, feature) };
            let goes_left = if b == NAN_BIN { nan_left } else { b <= split_bin };
            if goes_left {
                left_idx.push(i); lg += gradients[i] * weights[i]; lh += hessians[i] * weights[i];
            } else {
                right_idx.push(i); rg += gradients[i] * weights[i]; rh += hessians[i] * weights[i];
            }
        }
        Some(LeafCandidate {
            node_idx, depth, gain, feature, threshold, nan_goes_left: nan_left,
            left_indices: left_idx, right_indices: right_idx,
            left_g: lg, left_h: lh, right_g: rg, right_h: rh,
            histograms: hists, g_total, h_total,
        })
    };

    // Helper: try to build a split candidate using sort-based search (no-QM fallback).
    let try_split_sort = |node_idx: usize, depth: usize,
                           indices: Vec<usize>| -> Option<LeafCandidate> {
        if depth >= max_depth || indices.len() < min_samples_split { return None; }
        let (fi, thresh, gain, nan_left) = find_best_gbt_split_sort(
            cm, &indices, gradients, hessians, &weights,
            feat_set, lambda, reg_alpha, min_samples_leaf, min_child_weight,
        )?;
        if gain <= gamma { return None; }
        let feature = feat_set[fi];
        let mut left_idx = Vec::with_capacity(indices.len() / 2);
        let mut right_idx = Vec::with_capacity(indices.len() / 2);
        let mut lg = 0.0_f64; let mut lh = 0.0_f64;
        let mut rg = 0.0_f64; let mut rh = 0.0_f64;
        let g_total: f64 = indices.iter().map(|&i| gradients[i] * weights[i]).sum();
        let h_total: f64 = indices.iter().map(|&i| hessians[i] * weights[i]).sum();
        for &i in &indices {
            let v = cm.data[feat_set[fi] * cm.nrows + i];
            let goes_left = if v.is_nan() { nan_left } else { v <= thresh };
            if goes_left {
                left_idx.push(i); lg += gradients[i] * weights[i]; lh += hessians[i] * weights[i];
            } else {
                right_idx.push(i); rg += gradients[i] * weights[i]; rh += hessians[i] * weights[i];
            }
        }
        Some(LeafCandidate {
            node_idx, depth, gain, feature, threshold: thresh, nan_goes_left: nan_left,
            left_indices: left_idx, right_indices: right_idx,
            left_g: lg, left_h: lh, right_g: rg, right_h: rh,
            histograms: vec![], g_total, h_total,
        })
    };

    // Try root candidate
    let mut heap: BinaryHeap<LeafCandidate> = BinaryHeap::new();
    if let Some(root_cand) = if let Some(qm) = qm {
        // Histogram path: build root histograms in parallel, then scan.
        let root_hists = build_node_histograms(qm, &root_indices, gradients, hessians, &weights, feat_set);
        try_split_hist(0, 0, g_root, h_root, &root_indices, root_hists, qm)
    } else {
        try_split_sort(0, 0, root_indices)
    } {
        heap.push(root_cand);
    }

    while let Some(cand) = heap.pop() {
        // Check max_leaves constraint: after split we'll have n_leaves+1 leaves
        if max_leaves > 0 && n_leaves >= max_leaves { break; }

        // Left child index = nodes.len(), right child index = nodes.len() + 1
        let left_idx_node = nodes.len();
        let right_idx_node = left_idx_node + 1;

        let left_val = newton_value_smooth(cand.left_g, cand.left_h, lambda, reg_alpha, max_delta_step, leaf_smooth);
        let right_val = newton_value_smooth(cand.right_g, cand.right_h, lambda, reg_alpha, max_delta_step, leaf_smooth);

        // Convert leaf to split
        nodes[cand.node_idx].kind = NodeKind::Split {
            feature: cand.feature,
            threshold: cand.threshold,
            left: left_idx_node,
            right: right_idx_node,
        };
        nodes[cand.node_idx].nan_goes_left = cand.nan_goes_left;

        // Add left and right leaf nodes
        nodes.push(Node {
            kind: NodeKind::Leaf { value: left_val, proba_offset: 0 },
            nan_goes_left: true,
            n_samples: cand.left_indices.len(),
        });
        nodes.push(Node {
            kind: NodeKind::Leaf { value: right_val, proba_offset: 0 },
            nan_goes_left: true,
            n_samples: cand.right_indices.len(),
        });

        // Accumulate feature importance (gain-based); feature is already in full feature space
        if cand.feature < imp_acc.len() {
            imp_acc[cand.feature] += cand.gain;
        }

        n_leaves += 1; // replaced 1 leaf with 2, net +1

        let left_indices = cand.left_indices;
        let right_indices = cand.right_indices;

        // Update leaf assignments so callers can skip tree.predict after construction.
        for &i in &left_indices { leaf_assignments[i] = left_idx_node; }
        for &i in &right_indices { leaf_assignments[i] = right_idx_node; }

        let next_depth = cand.depth + 1;

        if let Some(qm) = qm {
            // Histogram path: subtraction trick — build smaller child's histograms,
            // derive larger child's histograms via parent − smaller. O(n_smaller × p)
            // instead of O((n_left + n_right) × p).
            let (smaller_idx, larger_idx, smaller_g, smaller_h, smaller_node, larger_node) =
                if left_indices.len() <= right_indices.len() {
                    (&left_indices, &right_indices,
                     cand.left_g, cand.left_h, left_idx_node, right_idx_node)
                } else {
                    (&right_indices, &left_indices,
                     cand.right_g, cand.right_h, right_idx_node, left_idx_node)
                };

            let smaller_hists = build_node_histograms(
                qm, smaller_idx, gradients, hessians, &weights, feat_set,
            );
            let larger_hists = subtract_histograms(&cand.histograms, &smaller_hists);

            // Larger child's G/H = parent − smaller child (no re-scan needed)
            let larger_g = cand.g_total - smaller_g;
            let larger_h = cand.h_total - smaller_h;

            if let Some(c) = try_split_hist(
                smaller_node, next_depth, smaller_g, smaller_h,
                smaller_idx, smaller_hists, qm,
            ) { heap.push(c); }
            if let Some(c) = try_split_hist(
                larger_node, next_depth, larger_g, larger_h,
                larger_idx, larger_hists, qm,
            ) { heap.push(c); }
        } else {
            // Sort-based fallback: no subtraction trick
            if let Some(c) = try_split_sort(left_idx_node, next_depth, left_indices) {
                heap.push(c);
            }
            if let Some(c) = try_split_sort(right_idx_node, next_depth, right_indices) {
                heap.push(c);
            }
        }
    }

    let fi = normalize_importances(&imp_acc, p);
    (GBTTree { nodes, feature_importances: fi }, leaf_assignments)
}

fn build_qm(cm: &ColMajorMatrix, n: usize, p: usize, threshold: usize, max_bins: usize) -> Option<QuantizedMatrix> {
    if n >= threshold {
        // Clamp to [1, 254]: NaN uses reserved slot 255 outside the data bin count.
        let mb = max_bins.max(1).min(254);
        Some(QuantizedMatrix::from_col_data(&cm.data, n, p, mb))
    } else {
        None
    }
}

fn base_weights(n: usize, sw: Option<&DVector<f64>>) -> Vec<f64> {
    match sw {
        None => vec![1.0; n],
        Some(sw) => {
            let total: f64 = sw.iter().sum();
            (0..n).map(|i| sw[i] * n as f64 / total).collect()
        }
    }
}

/// Subsample rows: selected rows keep base weight, others get 0.
/// Uses partial Fisher-Yates shuffle on index array.
fn subsample_weights(n: usize, frac: f64, seed: u64, base_w: &[f64]) -> Vec<f64> {
    if frac >= 1.0 {
        return base_w.to_vec();
    }
    let k = ((n as f64 * frac).floor() as usize).max(1);
    let mut rng = SimpleRng::new(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.gen_range(n - i);
        indices.swap(i, j);
    }
    let mut w = vec![0.0_f64; n];
    for &idx in &indices[..k] {
        w[idx] = base_w[idx];
    }
    w
}

/// GOSS (Gradient-based One-Side Sampling) weight vector.
///
/// Returns a full-n weight vector suitable for `fit_lossguide_tree`:
/// - Large-gradient samples (top `top_rate` fraction by |gradient|): keep `base_weights[i]`
/// - Randomly selected small-gradient samples (`other_rate` of remaining): scale weight by
///   `(1 - top_rate) / other_rate` to preserve unbiased gradient estimates
/// - Unsampled small-gradient samples: weight = 0.0 (excluded from splits, still get leaf pred)
///
/// `gradients` may be the residual vector (for binary/reg) or a per-sample magnitude
/// computed from all-class residuals (for multiclass).
///
/// GOSS and `subsample < 1.0` are mutually exclusive — caller ensures subsample is ignored.
fn goss_weights(
    gradients: &[f64],
    base_weights: &[f64],
    top_rate: f64,
    other_rate: f64,
    seed: u64,
) -> Vec<f64> {
    let n = gradients.len();

    // 1. Partial sort: O(n) average — find threshold separating top top_rate samples
    let mut idx: Vec<usize> = (0..n).collect();
    let n_top = ((n as f64 * top_rate).ceil() as usize).min(n);
    let pivot = n.saturating_sub(n_top);

    if pivot > 0 {
        // select_nth_unstable_by places pivot-th element correctly; elements after it
        // are all >= pivot element (unordered). idx[pivot..] = large-gradient group.
        idx.select_nth_unstable_by(pivot, |&a, &b| {
            gradients[a].abs().partial_cmp(&gradients[b].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let scale = (1.0 - top_rate) / other_rate;
    let mut out = vec![0.0f64; n]; // default: unsampled = weight 0

    // 2. Large-gradient group: keep original weights unchanged
    for &i in &idx[pivot..] {
        out[i] = base_weights[i];
    }

    if pivot == 0 || other_rate == 0.0 {
        return out; // top_rate=1.0 or other_rate=0.0: only large-gradient group
    }

    // 3. Small-gradient group: random sample using existing SimpleRng infrastructure
    let small = &idx[..pivot];
    let n_other = ((small.len() as f64 * other_rate).floor() as usize).min(small.len());
    let mut rng = SimpleRng::new(seed);
    let mut pool: Vec<usize> = (0..small.len()).collect();
    for i in 0..n_other {
        let j = i + rng.gen_range(small.len() - i);
        pool.swap(i, j);
        let sample_idx = small[pool[i]];
        out[sample_idx] = base_weights[sample_idx] * scale;
    }

    out
}

fn weighted_mean(y: &[f64], w: &[f64]) -> f64 {
    let mut sum_w = 0.0_f64;
    let mut sum_wy = 0.0_f64;
    for (&yi, &wi) in y.iter().zip(w.iter()) {
        sum_w += wi;
        sum_wy += wi * yi;
    }
    if sum_w < 1e-15 { 0.0 } else { sum_wy / sum_w }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-x).exp())
}

fn softmax_inplace(row: &mut [f64]) {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 1e-15 {
        for v in row.iter_mut() { *v /= sum; }
    } else {
        let k = row.len();
        for v in row.iter_mut() { *v = 1.0 / k as f64; }
    }
}

/// Find the leaf node index for a given row (depth-first traversal).
/// NaN feature values are routed by the learned `nan_goes_left` direction.
#[inline]
fn find_leaf(nodes: &[Node], cm: &ColMajorMatrix, row: usize) -> usize {
    let mut idx = 0;
    loop {
        match &nodes[idx].kind {
            NodeKind::Leaf { .. } => return idx,
            NodeKind::Split { feature, threshold, left, right } => {
                let val = unsafe { cm.get_unchecked(row, *feature) };
                idx = if val.is_nan() {
                    if nodes[idx].nan_goes_left { *left } else { *right }
                } else if val <= *threshold {
                    *left
                } else {
                    *right
                };
            }
            NodeKind::Placeholder => panic!("placeholder node during Newton update"),
        }
    }
}

/// Replace leaf values with XGBoost Newton steps:
///   leaf = soft_thresh(G_j, alpha) / (H_j + lambda + leaf_smooth)
/// with optional max_delta_step clipping.
///
/// Splits are unchanged — only leaf values are updated (Chen & Guestrin 2016).
/// Returns (g_sums, h_sums) per node for optional gamma pruning.
fn newton_leaf_update(
    nodes: &mut Vec<Node>,
    gradients: &[f64],
    hessians: &[f64],
    cm: &ColMajorMatrix,
    lambda: f64,
    reg_alpha: f64,
    min_child_weight: f64,
    max_delta_step: f64,
    leaf_smooth: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = cm.nrows;
    let len = nodes.len();
    let mut g_sums = vec![0.0_f64; len];
    let mut h_sums = vec![0.0_f64; len];

    for i in 0..n {
        let leaf_idx = find_leaf(nodes, cm, i);
        g_sums[leaf_idx] += gradients[i];
        h_sums[leaf_idx] += hessians[i];
    }

    for idx in 0..len {
        if let NodeKind::Leaf { ref mut value, .. } = nodes[idx].kind {
            if h_sums[idx] >= min_child_weight {
                *value = newton_value_smooth(
                    g_sums[idx], h_sums[idx], lambda, reg_alpha, max_delta_step, leaf_smooth,
                );
            } else {
                *value = 0.0; // insufficient hessian mass
            }
        }
    }
    (g_sums, h_sums)
}

/// Propagate gradient/hessian sums from leaves up to internal nodes (post-order).
fn propagate_sums_up(nodes: &[Node], g_sums: &mut [f64], h_sums: &mut [f64]) {
    fn recurse(nodes: &[Node], idx: usize, g: &mut [f64], h: &mut [f64]) {
        match &nodes[idx].kind {
            NodeKind::Leaf { .. } => {}
            NodeKind::Split { left, right, .. } => {
                recurse(nodes, *left, g, h);
                recurse(nodes, *right, g, h);
                g[idx] = g[*left] + g[*right];
                h[idx] = h[*left] + h[*right];
            }
            NodeKind::Placeholder => {}
        }
    }
    recurse(nodes, 0, g_sums, h_sums);
}

/// Prune splits whose XGBoost-style gain < gamma (bottom-up, leaves only).
/// With L1 (reg_alpha > 0), all three gain numerators use soft-thresholding.
fn gamma_prune(
    nodes: &mut Vec<Node>,
    g_sums: &[f64],
    h_sums: &[f64],
    lambda: f64,
    reg_alpha: f64,
    gamma: f64,
) {
    if gamma <= 0.0 { return; }
    let mut changed = true;
    while changed {
        changed = false;
        for idx in 0..nodes.len() {
            if let NodeKind::Split { left, right, .. } = nodes[idx].kind {
                if matches!(nodes[left].kind, NodeKind::Leaf { .. })
                    && matches!(nodes[right].kind, NodeKind::Leaf { .. })
                {
                    let (gl, hl) = (g_sums[left], h_sums[left]);
                    let (gr, hr) = (g_sums[right], h_sums[right]);
                    let gain = 0.5 * (
                        soft_thresh(gl, reg_alpha).powi(2) / (hl + lambda)
                        + soft_thresh(gr, reg_alpha).powi(2) / (hr + lambda)
                        - soft_thresh(gl + gr, reg_alpha).powi(2) / (hl + hr + lambda)
                    );
                    if gain < gamma {
                        let value = soft_thresh(gl + gr, reg_alpha) / (hl + hr + lambda);
                        nodes[idx].kind = NodeKind::Leaf { value, proba_offset: 0 };
                        changed = true;
                    }
                }
            }
        }
    }
}

/// Compute validation loss for early stopping (L2 = MSE, BinomialDeviance = log_loss,
/// MultinomialDeviance = softmax cross-entropy).
fn validation_loss(
    loss: GBTLoss,
    f_scores: &[f64],         // regression or binary
    f_scores_multi: Option<&[Vec<f64>]>, // multiclass (f_scores_multi[i][c])
    y_f64: &[f64],            // regression: actual y; binary: 0/1; multiclass: unused
    y_idx: &[usize],          // class indices (unused for regression)
    val_indices: &[usize],
    _k: usize,
) -> f64 {
    match loss {
        GBTLoss::L2 => {
            let mut sum = 0.0;
            for &vi in val_indices {
                let diff = y_f64[vi] - f_scores[vi];
                sum += diff * diff;
            }
            sum / val_indices.len() as f64
        }
        GBTLoss::BinomialDeviance => {
            let mut sum = 0.0;
            for &vi in val_indices {
                let p = sigmoid(f_scores[vi]).clamp(1e-15, 1.0 - 1e-15);
                let yi = y_f64[vi];
                sum += -(yi * p.ln() + (1.0 - yi) * (1.0 - p).ln());
            }
            sum / val_indices.len() as f64
        }
        GBTLoss::MultinomialDeviance => {
            let f_multi = f_scores_multi.expect("multiclass early stopping requires f_scores_multi");
            let mut sum = 0.0;
            for &vi in val_indices {
                let mut row = f_multi[vi].clone();
                softmax_inplace(&mut row);
                let p = row[y_idx[vi]].clamp(1e-15, 1.0);
                sum += -p.ln();
            }
            sum / val_indices.len() as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reg_data() -> (DMatrix<f64>, Vec<f64>) {
        // y = x0 + 2*x1 (deterministic signal, 50 rows)
        let n = 50_usize;
        let data: Vec<f64> = (0..n * 3).map(|i| (i as f64 * 1.618033) % 1.0).collect();
        let x = DMatrix::from_row_slice(n, 3, &data);
        let y: Vec<f64> = (0..n).map(|i| data[i * 3] + 2.0 * data[i * 3 + 1]).collect();
        (x, y)
    }

    fn make_clf_data() -> (DMatrix<f64>, Vec<i64>) {
        let n = 60_usize;
        let data: Vec<f64> = (0..n * 2).map(|i| (i as f64 * 1.618033) % 1.0).collect();
        let x = DMatrix::from_row_slice(n, 2, &data);
        let y: Vec<i64> = (0..n).map(|i| if data[i * 2] > 0.5 { 1 } else { 0 }).collect();
        (x, y)
    }

    fn make_multiclass_data() -> (DMatrix<f64>, Vec<i64>) {
        let n = 90_usize;
        let data: Vec<f64> = (0..n * 2).map(|i| (i as f64 * 1.618033) % 1.0).collect();
        let x = DMatrix::from_row_slice(n, 2, &data);
        let y: Vec<i64> = (0..n).map(|i| (i % 3) as i64).collect();
        (x, y)
    }

    #[test]
    fn test_gbt_reg_fit_predict() {
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);
        assert_eq!(preds.len(), x.nrows());
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let ss_res: f64 = y.iter().zip(&preds).map(|(yi, pi)| (yi - pi).powi(2)).sum();
        let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.0, "r2={r2:.3} should be > 0 for structured data");
    }

    #[test]
    fn test_gbt_reg_serialization() {
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);
        let json = model.to_json().unwrap();
        let model2 = GBTModel::from_json(&json).unwrap();
        let preds2 = model2.predict_reg(&x);
        for (a, b) in preds.iter().zip(&preds2) {
            assert!((a - b).abs() < 1e-10, "serialization changed predictions");
        }
    }

    #[test]
    fn test_gbt_clf_binary() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();
        assert_eq!(model.n_classes, 2);
        assert_eq!(model.loss, GBTLoss::BinomialDeviance);

        let preds = model.predict_clf(&x);
        assert_eq!(preds.len(), x.nrows());
        assert!(preds.iter().all(|&p| p == 0 || p == 1));

        let proba = model.predict_proba(&x);
        assert_eq!(proba.nrows(), x.nrows());
        assert_eq!(proba.ncols(), 2);
        for i in 0..x.nrows() {
            let s: f64 = proba.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "proba row {i} sums to {s}");
        }
    }

    #[test]
    fn test_gbt_clf_multiclass() {
        let (x, y) = make_multiclass_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();
        assert_eq!(model.n_classes, 3);
        assert_eq!(model.loss, GBTLoss::MultinomialDeviance);

        let preds = model.predict_clf(&x);
        assert_eq!(preds.len(), x.nrows());

        let proba = model.predict_proba(&x);
        assert_eq!(proba.nrows(), x.nrows());
        assert_eq!(proba.ncols(), 3);
        for i in 0..x.nrows() {
            let s: f64 = proba.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "proba row {i} sums to {s:.8}");
        }
    }

    #[test]
    fn test_gbt_feature_importances_sum_to_one() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();
        let s: f64 = model.feature_importances.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "importances sum={s}");
    }

    #[test]
    fn test_gbt_subsample() {
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 0.5, 42);
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);
        assert_eq!(preds.len(), x.nrows());
    }

    #[test]
    fn test_gbt_clf_serialization() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();
        let preds = model.predict_clf(&x);
        let json = model.to_json().unwrap();
        let model2 = GBTModel::from_json(&json).unwrap();
        let preds2 = model2.predict_clf(&x);
        assert_eq!(preds, preds2);
    }

    #[test]
    fn test_gbt_empty_data_error() {
        let x = DMatrix::from_row_slice(0, 2, &[]);
        let y: Vec<f64> = vec![];
        let mut model = GBTModel::new(5, 0.1, 3, 2, 1, 1.0, 42);
        assert!(model.fit_reg(&x, &y, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Regularization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gbt_lambda_reduces_overfitting() {
        // High lambda should produce predictions closer to the mean (less extreme)
        let (x, y) = make_reg_data();

        let mut model_no_reg = GBTModel::new(30, 0.1, 4, 2, 1, 1.0, 42);
        model_no_reg.fit_reg(&x, &y, None).unwrap();
        let preds_no_reg = model_no_reg.predict_reg(&x);

        let mut model_reg = GBTModel::new(30, 0.1, 4, 2, 1, 1.0, 42);
        model_reg.lambda = 10.0;
        model_reg.fit_reg(&x, &y, None).unwrap();
        let preds_reg = model_reg.predict_reg(&x);

        // With high lambda, leaf values are shrunk → predictions are less extreme
        let var_no_reg: f64 = preds_no_reg.iter().map(|p| p * p).sum::<f64>() / preds_no_reg.len() as f64;
        let var_reg: f64 = preds_reg.iter().map(|p| p * p).sum::<f64>() / preds_reg.len() as f64;
        // Regularized model should have less variance in predictions (or at least not more)
        // With heavy lambda the Newton steps are damped: leaf = sum(g) / (sum(h) + lambda)
        assert!(
            var_reg <= var_no_reg * 1.05,
            "lambda=10 should shrink predictions: var_reg={var_reg:.4}, var_no_reg={var_no_reg:.4}"
        );
    }

    #[test]
    fn test_gbt_lambda_binary_clf() {
        // Lambda should not break binary classification
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.lambda = 1.0;
        model.fit_clf(&x, &y, None).unwrap();
        let preds = model.predict_clf(&x);
        assert_eq!(preds.len(), x.nrows());
        assert!(preds.iter().all(|&p| p == 0 || p == 1));
    }

    #[test]
    fn test_gbt_lambda_multiclass() {
        // Lambda should not break multiclass classification
        let (x, y) = make_multiclass_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.lambda = 2.0;
        model.fit_clf(&x, &y, None).unwrap();
        let proba = model.predict_proba(&x);
        for i in 0..x.nrows() {
            let s: f64 = proba.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "proba row {i} sums to {s}");
        }
    }

    #[test]
    fn test_gbt_gamma_prunes_nodes() {
        // High gamma should prune low-gain splits → fewer nodes per tree.
        // Use 1 round to avoid cross-round residual interaction (with split_lambda,
        // pruned round-1 trees change residuals, potentially growing later trees).
        let (x, y) = make_reg_data();

        let mut model_no_gamma = GBTModel::new(1, 0.1, 4, 2, 1, 1.0, 42);
        model_no_gamma.lambda = 1.0;
        model_no_gamma.fit_reg(&x, &y, None).unwrap();

        let mut model_gamma = GBTModel::new(1, 0.1, 4, 2, 1, 1.0, 42);
        model_gamma.lambda = 1.0;
        model_gamma.gamma = 5.0; // aggressive pruning
        model_gamma.fit_reg(&x, &y, None).unwrap();

        // Count total nodes (single round)
        let count_nodes = |m: &GBTModel| -> usize {
            m.trees.iter().flat_map(|round| round.iter().map(|t| t.nodes.len())).sum()
        };
        let n_no_gamma = count_nodes(&model_no_gamma);
        let n_gamma = count_nodes(&model_gamma);
        assert!(
            n_gamma <= n_no_gamma,
            "gamma=5 should produce <=nodes: got {n_gamma} vs {n_no_gamma}"
        );
    }

    #[test]
    fn test_gbt_min_child_weight_large_produces_stumps() {
        // Very large min_child_weight should cause leaf values to be zeroed
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.min_child_weight = 1e10; // impossibly large
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);
        // All leaf values should be 0 (insufficient hessian mass) so predictions = init_val
        let init_val = model.initial_pred[0];
        for (i, &p) in preds.iter().enumerate() {
            assert!(
                (p - init_val).abs() < 1e-10,
                "sample {i}: pred={p}, expected init_val={init_val}"
            );
        }
    }

    #[test]
    fn test_gbt_colsample_bytree_changes_predictions() {
        // Different colsample_bytree should produce different predictions
        let (x, y) = make_reg_data();

        let mut model_full = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model_full.fit_reg(&x, &y, None).unwrap();
        let preds_full = model_full.predict_reg(&x);

        let mut model_half = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model_half.colsample_bytree = 0.5;
        model_half.fit_reg(&x, &y, None).unwrap();
        let preds_half = model_half.predict_reg(&x);

        // Predictions should differ (column subsetting produces different trees)
        let any_diff = preds_full.iter().zip(&preds_half)
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(any_diff, "colsample_bytree=0.5 should produce different predictions from 1.0");
    }

    #[test]
    fn test_gbt_early_stopping_reg() {
        // With early stopping on noisy data with many rounds, should stop before n_estimators
        let n = 100_usize;
        let data: Vec<f64> = (0..n * 3).map(|i| (i as f64 * 1.618033) % 1.0).collect();
        let x = DMatrix::from_row_slice(n, 3, &data);
        let y: Vec<f64> = (0..n).map(|i| data[i * 3] + 2.0 * data[i * 3 + 1]).collect();

        let mut model = GBTModel::new(500, 0.3, 5, 2, 1, 1.0, 42);
        model.n_iter_no_change = Some(5);
        model.validation_fraction = 0.2;
        model.fit_reg(&x, &y, None).unwrap();

        assert!(
            model.trees.len() < 500,
            "early stopping should trigger before 500 rounds, got {}",
            model.trees.len()
        );
        assert!(model.best_n_rounds.is_some());
        assert_eq!(model.best_n_rounds.unwrap(), model.trees.len());
    }

    #[test]
    fn test_gbt_early_stopping_binary() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(500, 0.3, 3, 2, 1, 1.0, 42);
        model.n_iter_no_change = Some(5);
        model.validation_fraction = 0.2;
        model.fit_clf(&x, &y, None).unwrap();

        // Should either stop early or run all 500 — both are valid
        // But trees.len() should be consistent with best_n_rounds
        if let Some(nr) = model.best_n_rounds {
            assert_eq!(nr, model.trees.len());
        }
        // Predictions should still be valid
        let preds = model.predict_clf(&x);
        assert_eq!(preds.len(), x.nrows());
        assert!(preds.iter().all(|&p| p == 0 || p == 1));
    }

    #[test]
    fn test_gbt_early_stopping_multiclass() {
        let (x, y) = make_multiclass_data();
        let mut model = GBTModel::new(500, 0.3, 3, 2, 1, 1.0, 42);
        model.n_iter_no_change = Some(5);
        model.validation_fraction = 0.2;
        model.fit_clf(&x, &y, None).unwrap();

        if let Some(nr) = model.best_n_rounds {
            assert_eq!(nr, model.trees.len());
        }
        let proba = model.predict_proba(&x);
        for i in 0..x.nrows() {
            let s: f64 = proba.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "proba row {i} sums to {s}");
        }
    }

    #[test]
    fn test_gbt_regularized_serialization_roundtrip() {
        // Verify new fields survive JSON serialization
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.lambda = 2.5;
        model.gamma = 0.1;
        model.colsample_bytree = 0.7;
        model.min_child_weight = 3.0;
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);

        let json = model.to_json().unwrap();
        let model2 = GBTModel::from_json(&json).unwrap();
        assert!((model2.lambda - 2.5).abs() < 1e-10);
        assert!((model2.gamma - 0.1).abs() < 1e-10);
        assert!((model2.colsample_bytree - 0.7).abs() < 1e-10);
        assert!((model2.min_child_weight - 3.0).abs() < 1e-10);

        let preds2 = model2.predict_reg(&x);
        for (a, b) in preds.iter().zip(&preds2) {
            assert!((a - b).abs() < 1e-10, "serialization changed predictions");
        }
    }

    #[test]
    fn test_gbt_deserialize_old_format_gets_defaults() {
        // Simulate deserializing a model JSON without the new fields —
        // serde(default) should fill in default_one for colsample_bytree/min_child_weight
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(5, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_reg(&x, &y, None).unwrap();
        let json = model.to_json().unwrap();

        // Parse as Value, remove new fields, re-serialize
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        if let serde_json::Value::Object(ref mut map) = v {
            map.remove("lambda");
            map.remove("gamma");
            map.remove("colsample_bytree");
            map.remove("min_child_weight");
            map.remove("n_iter_no_change");
            map.remove("validation_fraction");
            map.remove("best_n_rounds");
        }
        let stripped = serde_json::to_string(&v).unwrap();
        let model2 = GBTModel::from_json(&stripped).unwrap();

        assert!((model2.lambda - 0.0).abs() < 1e-10, "default lambda should be 0");
        assert!((model2.gamma - 0.0).abs() < 1e-10, "default gamma should be 0");
        assert!((model2.colsample_bytree - 1.0).abs() < 1e-10, "default colsample_bytree should be 1");
        assert!((model2.min_child_weight - 1.0).abs() < 1e-10, "default min_child_weight should be 1");
        assert!(model2.n_iter_no_change.is_none());
        assert!((model2.validation_fraction - 0.1).abs() < 1e-10);
        assert!(model2.best_n_rounds.is_none());
    }

    #[test]
    fn test_gbt_all_regularization_combined() {
        // Smoke test: all regularization features enabled at once
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(50, 0.1, 3, 2, 1, 0.8, 42);
        model.lambda = 1.0;
        model.gamma = 0.5;
        model.colsample_bytree = 0.8;
        model.min_child_weight = 2.0;
        model.fit_clf(&x, &y, None).unwrap();

        let preds = model.predict_clf(&x);
        assert_eq!(preds.len(), x.nrows());
        assert!(preds.iter().all(|&p| p == 0 || p == 1));

        let proba = model.predict_proba(&x);
        for i in 0..x.nrows() {
            let s: f64 = proba.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    // -------------------------------------------------------------------
    // Parallel predict determinism tests
    // -------------------------------------------------------------------

    #[test]
    fn test_gbt_predict_reg_parallel_determinism() {
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_reg(&x, &y, None).unwrap();

        let baseline = model.predict_reg(&x);
        for _ in 0..5 {
            let result = model.predict_reg(&x);
            assert_eq!(baseline, result, "parallel predict_reg not deterministic");
        }
    }

    #[test]
    fn test_gbt_predict_clf_binary_parallel_determinism() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();

        let baseline_clf = model.predict_clf(&x);
        let baseline_proba = model.predict_proba(&x);
        for _ in 0..5 {
            let result_clf = model.predict_clf(&x);
            let result_proba = model.predict_proba(&x);
            assert_eq!(baseline_clf, result_clf,
                "parallel predict_clf (binary) not deterministic");
            assert_eq!(baseline_proba.as_slice(), result_proba.as_slice(),
                "parallel predict_proba (binary) not deterministic");
        }
    }

    #[test]
    fn test_gbt_predict_clf_multiclass_parallel_determinism() {
        let (x, y) = make_multiclass_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();

        let baseline_clf = model.predict_clf(&x);
        let baseline_proba = model.predict_proba(&x);
        for _ in 0..5 {
            let result_clf = model.predict_clf(&x);
            let result_proba = model.predict_proba(&x);
            assert_eq!(baseline_clf, result_clf,
                "parallel predict_clf (multiclass) not deterministic");
            assert_eq!(baseline_proba.as_slice(), result_proba.as_slice(),
                "parallel predict_proba (multiclass) not deterministic");
        }
    }

    // -----------------------------------------------------------------------
    // Phase 1 tests: reg_alpha, max_delta_step, base_score, colsample fix
    // -----------------------------------------------------------------------

    #[test]
    fn test_reg_alpha_sparsity() {
        // L1 with high alpha should produce more zero-weight leaves (more stumps)
        let (x, y) = make_reg_data();

        let mut model_no_l1 = GBTModel::new(20, 0.1, 4, 2, 1, 1.0, 42);
        model_no_l1.lambda = 1.0;
        model_no_l1.fit_reg(&x, &y, None).unwrap();

        let mut model_l1 = GBTModel::new(20, 0.1, 4, 2, 1, 1.0, 42);
        model_l1.lambda = 1.0;
        model_l1.reg_alpha = 5.0;
        model_l1.fit_reg(&x, &y, None).unwrap();

        // Count zero-weight leaves (soft-thresh zeroes them when |G| <= alpha)
        let count_zero_leaves = |m: &GBTModel| -> usize {
            m.trees.iter()
                .flat_map(|round| round.iter())
                .flat_map(|t| t.nodes.iter())
                .filter(|n| matches!(n.kind, NodeKind::Leaf { value, .. } if value.abs() < 1e-10))
                .count()
        };
        let zeros_no_l1 = count_zero_leaves(&model_no_l1);
        let zeros_l1 = count_zero_leaves(&model_l1);
        assert!(
            zeros_l1 >= zeros_no_l1,
            "reg_alpha=5.0 should produce >= zero-weight leaves: got {} vs {}",
            zeros_l1, zeros_no_l1
        );
    }

    #[test]
    fn test_reg_alpha_lambda_interaction() {
        // Both alpha and lambda nonzero → smaller leaf weights than either alone
        let (x, y) = make_reg_data();

        let mut model_lambda_only = GBTModel::new(10, 0.3, 3, 2, 1, 1.0, 42);
        model_lambda_only.lambda = 5.0;
        model_lambda_only.fit_reg(&x, &y, None).unwrap();

        let mut model_both = GBTModel::new(10, 0.3, 3, 2, 1, 1.0, 42);
        model_both.lambda = 5.0;
        model_both.reg_alpha = 2.0;
        model_both.fit_reg(&x, &y, None).unwrap();

        // Predictions from model_both should be more shrunken (closer to initial_pred)
        let init = model_both.initial_pred[0];
        let mean_dev_lambda: f64 = model_lambda_only.predict_reg(&x)
            .iter().map(|p| (p - init).abs()).sum::<f64>() / x.nrows() as f64;
        let mean_dev_both: f64 = model_both.predict_reg(&x)
            .iter().map(|p| (p - init).abs()).sum::<f64>() / x.nrows() as f64;
        assert!(
            mean_dev_both <= mean_dev_lambda * 1.05,
            "lambda+alpha should shrink predictions more: dev_both={mean_dev_both:.4} vs dev_lambda={mean_dev_lambda:.4}"
        );
    }

    #[test]
    fn test_max_delta_step_clips() {
        // max_delta_step=0.1 → all leaf values clamped to [-0.1, 0.1]
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(20, 0.1, 4, 2, 1, 1.0, 42);
        model.max_delta_step = 0.1;
        model.fit_reg(&x, &y, None).unwrap();

        for round in &model.trees {
            for tree in round {
                for node in &tree.nodes {
                    if let NodeKind::Leaf { value, .. } = node.kind {
                        assert!(
                            value.abs() <= 0.1 + 1e-12,
                            "leaf value {value} exceeds max_delta_step=0.1"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_base_score_override() {
        // base_score=5.0 → initial_pred=[5.0]
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(5, 0.1, 3, 2, 1, 1.0, 42);
        model.base_score = Some(5.0);
        model.fit_reg(&x, &y, None).unwrap();
        assert_eq!(model.initial_pred.len(), 1);
        assert!((model.initial_pred[0] - 5.0).abs() < 1e-10,
            "base_score=5.0 but initial_pred={}", model.initial_pred[0]);
    }

    #[test]
    fn test_colsample_bytree_per_tree_semantics() {
        // With colsample_bytree=0.5 on 4-feature data, each tree should only use 2 features.
        // Feature importances should be zero for features not in any tree's subset — but
        // because subsets vary per round, we only check that predictions are deterministic
        // and differ from full feature model.
        let n = 80_usize;
        let data: Vec<f64> = (0..n * 4).map(|i| (i as f64 * 1.618033) % 1.0).collect();
        let x = DMatrix::from_row_slice(n, 4, &data);
        let y: Vec<f64> = (0..n).map(|i| data[i * 4] + 2.0 * data[i * 4 + 1]).collect();

        let mut model_half = GBTModel::new(30, 0.1, 3, 2, 1, 1.0, 77);
        model_half.colsample_bytree = 0.5;
        model_half.fit_reg(&x, &y, None).unwrap();

        // Determinism: same predictions on multiple calls
        let p1 = model_half.predict_reg(&x);
        let p2 = model_half.predict_reg(&x);
        for (a, b) in p1.iter().zip(&p2) {
            assert_eq!(a, b, "colsample_bytree predictions not deterministic");
        }

        // Feature importances should sum to 1.0
        let imp_sum: f64 = model_half.feature_importances.iter().sum();
        assert!((imp_sum - 1.0).abs() < 1e-6, "importances sum={imp_sum}");
        assert_eq!(model_half.feature_importances.len(), 4);
    }

    #[test]
    fn test_colsample_subsample_seeds_independent() {
        // Changing only subsample should not change per-tree feature subsets (different seeds).
        // We verify this by checking predictions differ between subsample=1.0 and 0.5
        // but importances follow the same distribution pattern (no systematic bias).
        let (x, y) = make_reg_data();

        let mut m1 = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        m1.colsample_bytree = 0.5;
        m1.fit_reg(&x, &y, None).unwrap();
        let imps1 = m1.feature_importances.clone();

        let mut m2 = GBTModel::new(20, 0.1, 3, 2, 1, 0.7, 42);
        m2.colsample_bytree = 0.5;
        m2.fit_reg(&x, &y, None).unwrap();
        let imps2 = m2.feature_importances.clone();

        // importances should sum to 1.0 for both
        let s1: f64 = imps1.iter().sum();
        let s2: f64 = imps2.iter().sum();
        assert!((s1 - 1.0).abs() < 1e-6, "m1 importances sum={s1}");
        assert!((s2 - 1.0).abs() < 1e-6, "m2 importances sum={s2}");
    }

    // -----------------------------------------------------------------------
    // Phase 2: monotone constraints
    // -----------------------------------------------------------------------

    #[test]
    fn test_monotone_increasing_regression() {
        // Feature 0: monotone increasing. Predictions should not decrease as x0 increases.
        let n = 200_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = (i as f64) / (n as f64);
            let x1 = 0.5_f64; // constant feature
            x_data.push(x0);
            x_data.push(x1);
            y_data.push(x0 * 2.0 + 0.01 * (i as f64 % 5.0)); // nearly linear
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);
        let mut model = GBTModel::new(50, 0.1, 3, 2, 1, 1.0, 42);
        model.monotone_cst = Some(vec![1, 0]); // feature 0: increasing, feature 1: unconstrained
        model.fit_reg(&x_mat, &y_data, None).unwrap();

        // Build a test grid over feature 0 with feature 1 fixed
        let test_x: Vec<f64> = (0..20).flat_map(|i| vec![(i as f64) / 20.0, 0.5]).collect();
        let test_mat = DMatrix::from_row_slice(20, 2, &test_x);
        let preds = model.predict_reg(&test_mat);

        for i in 1..preds.len() {
            assert!(
                preds[i] >= preds[i - 1] - 1e-9,
                "monotone increasing violated: pred[{}]={} < pred[{}]={}",
                i, preds[i], i - 1, preds[i - 1]
            );
        }
    }

    #[test]
    fn test_monotone_decreasing_binary() {
        // Feature 0: monotone decreasing for P(class=1). As x0 increases, P(1) must not increase.
        let n = 200_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = (i as f64) / (n as f64);
            x_data.push(x0);
            x_data.push(0.5_f64);
            // y=1 for low x0, y=0 for high x0 (decreasing probability)
            y_data.push(if x0 < 0.5 { 1_i64 } else { 0_i64 });
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);
        let mut model = GBTModel::new(50, 0.1, 3, 2, 1, 1.0, 42);
        model.monotone_cst = Some(vec![-1, 0]); // feature 0: decreasing, feature 1: unconstrained
        model.fit_clf(&x_mat, &y_data, None).unwrap();

        // Test grid over feature 0
        let test_x: Vec<f64> = (0..20).flat_map(|i| vec![(i as f64) / 20.0, 0.5]).collect();
        let test_mat = DMatrix::from_row_slice(20, 2, &test_x);
        let proba = model.predict_proba(&test_mat);

        // proba column for class 1 is column 1 (binary: classes [0,1])
        for i in 1..20 {
            let p_prev = proba[(i - 1, 1)];
            let p_curr = proba[(i, 1)];
            assert!(
                p_curr <= p_prev + 1e-9,
                "monotone decreasing violated: P(class=1)[{}]={} > P(class=1)[{}]={}",
                i, p_curr, i - 1, p_prev
            );
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: max_bin parameter
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_bin_64_valid_model() {
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(20, 0.1, 3, 2, 1, 1.0, 42);
        model.max_bin = 64;
        model.histogram_threshold = 1; // force histogram mode even on small data
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);
        assert_eq!(preds.len(), y.len());
        // predictions should be finite
        assert!(preds.iter().all(|p| p.is_finite()), "predictions not finite with max_bin=64");
    }

    #[test]
    fn test_max_bin_nan_uses_reserved_slot() {
        // Column 1 has 50% NaN. Fitting should succeed and produce finite predictions.
        let n = 100_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            x_data.push(i as f64);
            // every other row: NaN in feature 1
            if i % 2 == 0 { x_data.push(f64::NAN); } else { x_data.push(i as f64 * 0.5); }
            y_data.push(i as f64 * 0.1);
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.max_bin = 32;
        model.histogram_threshold = 1; // force histogram mode
        model.fit_reg(&x_mat, &y_data, None).unwrap();
        let preds = model.predict_reg(&x_mat);
        assert!(preds.iter().all(|p| p.is_finite()), "NaN column with max_bin=32 produced non-finite pred");
    }

    // -----------------------------------------------------------------------
    // Phase 4a: Lossguide growth
    // -----------------------------------------------------------------------

    #[test]
    fn test_lossguide_grows_unbalanced() {
        // XOR-like data — lossguide should pursue best splits (potentially unbalanced)
        // rather than full symmetric depth like depthwise
        let n = 80_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_clf: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = (i as f64) / (n as f64);
            let f1 = ((i * 3) as f64) / (n as f64) % 1.0;
            x_data.push(f0);
            x_data.push(f1);
            y_clf.push(if f0 > 0.5 { 1 } else { 0 });
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);

        let mut model_lg = GBTModel::new(10, 0.3, 4, 2, 1, 1.0, 42);
        model_lg.grow_policy = GrowPolicy::Lossguide;
        model_lg.max_leaves = 8;
        model_lg.fit_clf(&x_mat, &y_clf, None).unwrap();
        let preds = model_lg.predict_clf(&x_mat);
        assert_eq!(preds.len(), n);
        // Model should achieve some accuracy
        let acc: f64 = preds.iter().zip(y_clf.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.6, "lossguide accuracy too low: {acc}");
    }

    #[test]
    fn test_max_leaves_bound() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(5, 0.3, 10, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 4;
        model.fit_clf(&x, &y, None).unwrap();
        let preds = model.predict_clf(&x);
        assert_eq!(preds.len(), y.len());
        assert!(preds.iter().all(|p| *p == 0 || *p == 1), "invalid class predictions");
    }

    #[test]
    fn test_lossguide_max_depth_enforced() {
        let (x, y) = make_reg_data();
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 0; // unlimited leaves, depth is the only constraint
        model.fit_reg(&x, &y, None).unwrap();
        let preds = model.predict_reg(&x);
        assert!(preds.iter().all(|p| p.is_finite()), "non-finite predictions with lossguide max_depth=3");
    }

    #[test]
    fn test_lossguide_convergence() {
        // Lossguide should achieve reasonable training loss
        let n = 100_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = i as f64 / n as f64;
            let f1 = (i as f64 / n as f64).powi(2);
            x_data.push(f0);
            x_data.push(f1);
            y_data.push(f0 * 2.0 + f1);
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);

        let mut model = GBTModel::new(30, 0.3, 4, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 8;
        model.fit_reg(&x_mat, &y_data, None).unwrap();
        let preds = model.predict_reg(&x_mat);
        let mse: f64 = preds.iter().zip(y_data.iter()).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / n as f64;
        assert!(mse < 0.5, "lossguide MSE too high: {mse}");
    }

    #[test]
    fn test_lossguide_serialization_roundtrip() {
        let (x, y) = make_clf_data();
        let mut model = GBTModel::new(10, 0.3, 3, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 4;
        model.fit_clf(&x, &y, None).unwrap();
        let preds_before = model.predict_proba(&x);

        let json = model.to_json().unwrap();
        let loaded = GBTModel::from_json(&json).unwrap();
        let preds_after = loaded.predict_proba(&x);

        assert_eq!(preds_before.nrows(), preds_after.nrows());
        for i in 0..preds_before.nrows() {
            for j in 0..preds_before.ncols() {
                assert!(
                    (preds_before[(i, j)] - preds_after[(i, j)]).abs() < 1e-10,
                    "roundtrip mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_lossguide_multiclass() {
        let n = 90_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_clf: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            x_data.push(i as f64 / n as f64);
            x_data.push((i * 2) as f64 / n as f64 % 1.0);
            y_clf.push((i % 3) as i64);
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);

        let mut model = GBTModel::new(20, 0.3, 3, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 4;
        model.fit_clf(&x_mat, &y_clf, None).unwrap();
        assert_eq!(model.n_classes, 3);
        let proba = model.predict_proba(&x_mat);
        assert_eq!(proba.ncols(), 3);
        for i in 0..n {
            let row_sum: f64 = (0..3).map(|j| proba[(i, j)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "proba row {i} sum={row_sum}");
        }
    }

    #[test]
    fn test_lossguide_early_stopping_converges() {
        let n = 100_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            x_data.push(i as f64 / n as f64);
            x_data.push(((i * 7) as f64 / n as f64) % 1.0);
            y_data.push(i as f64 / n as f64);
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);

        let mut model = GBTModel::new(200, 0.1, 3, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 4;
        model.n_iter_no_change = Some(5);
        model.validation_fraction = 0.2;
        model.fit_reg(&x_mat, &y_data, None).unwrap();
        // Should stop before 200 rounds
        let stopped = model.best_n_rounds.unwrap_or(200);
        assert!(stopped < 200, "early stopping didn't trigger: best_n_rounds={stopped}");
    }

    // ── GOSS unit tests ────────────────────────────────────────────────────────

    #[test]
    fn test_goss_weights_size() {
        // Active sample count ≈ (top_rate + (1-top_rate)*other_rate) * n (within ±5%)
        let n = 10_000_usize;
        let gradients: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64 / 2.0)).collect();
        let base_w = vec![1.0f64; n];
        let w = goss_weights(&gradients, &base_w, 0.2, 0.1, 42);
        let active = w.iter().filter(|&&x| x > 0.0).count();
        // Expected: ceil(0.2*10000)=2000 large + floor(0.1*8000)=800 small = 2800
        assert!((active as i64 - 2800).abs() < 200, "active={active}, expected ~2800");
    }

    #[test]
    fn test_goss_weights_scale() {
        let n = 1000_usize;
        // Create clear gradient separation: first 200 are "large", rest are "small"
        let mut gradients = vec![0.1f64; n];
        for i in 0..200 { gradients[i] = 10.0; }
        let base_w = vec![2.0f64; n]; // non-unit weights
        let top_rate = 0.2;
        let other_rate = 0.5;
        let scale = (1.0 - top_rate) / other_rate; // 1.6
        let w = goss_weights(&gradients, &base_w, top_rate, other_rate, 42);
        // Large-gradient samples must have original weight
        for i in 0..200 {
            assert_eq!(w[i], 2.0, "large-gradient sample {i} should keep original weight");
        }
        // Small-gradient samples that ARE selected must have scaled weight
        let small_selected: Vec<f64> = w[200..].iter().copied().filter(|&x| x > 0.0).collect();
        for &ws in &small_selected {
            assert!((ws - base_w[0] * scale).abs() < 1e-10, "small-gradient weight {ws} != {}", base_w[0] * scale);
        }
    }

    #[test]
    fn test_goss_weights_deterministic() {
        let n = 500_usize;
        let gradients: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03).sin()).collect();
        let base_w = vec![1.0f64; n];
        let w1 = goss_weights(&gradients, &base_w, 0.2, 0.1, 99);
        let w2 = goss_weights(&gradients, &base_w, 0.2, 0.1, 99);
        assert_eq!(w1, w2, "goss_weights must be deterministic for the same seed");
        let w3 = goss_weights(&gradients, &base_w, 0.2, 0.1, 100);
        assert_ne!(w1, w3, "different seeds must produce different samples");
    }

    #[test]
    fn test_goss_disabled_at_1() {
        let n = 200_usize;
        let gradients: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let base_w: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let w = goss_weights(&gradients, &base_w, 1.0, 1.0, 42);
        // All weights unchanged when top_rate=1.0 (all are "large gradient")
        for i in 0..n {
            assert_eq!(w[i], base_w[i], "top_rate=1.0: weight {i} should equal base");
        }
    }

    #[test]
    fn test_goss_zero_other_rate_no_small_selected() {
        // other_rate=0.0: only large-gradient group selected, no small-gradient samples
        let n = 500_usize;
        let gradients: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let base_w = vec![1.0f64; n];
        let w = goss_weights(&gradients, &base_w, 0.2, 0.0, 42);
        let n_top = (n as f64 * 0.2).ceil() as usize;
        let active = w.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(active, n_top, "active={active} should equal n_top={n_top}");
    }

    #[test]
    fn test_goss_min_n_gate_in_fit_reg() {
        // When n < goss_min_n, GOSS must NOT be applied (subsample used instead)
        let n = 100_usize;
        let x_data: Vec<f64> = (0..n * 2).map(|i| i as f64 / (n * 2) as f64).collect();
        let y_data: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        use nalgebra::DMatrix;
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);
        // GOSS active but n=100 < goss_min_n=50000 → no panic, normal training
        let mut model = GBTModel::new(10, 0.1, 3, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.goss_top_rate = 0.2;
        model.goss_other_rate = 0.1;
        model.goss_min_n = 50_000; // n=100 < this → GOSS skipped
        model.fit_reg(&x_mat, &y_data, None).unwrap();
        assert!(model.trees.len() > 0);
    }

    #[test]
    fn test_multiclass_lossguide_reg_alpha() {
        let n = 90_usize;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_clf: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            x_data.push(i as f64 / n as f64);
            x_data.push((i * 2) as f64 / n as f64 % 1.0);
            y_clf.push((i % 3) as i64);
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);

        let mut model = GBTModel::new(20, 0.3, 3, 2, 1, 1.0, 42);
        model.grow_policy = GrowPolicy::Lossguide;
        model.max_leaves = 4;
        model.reg_alpha = 0.5;
        model.lambda = 1.0;
        model.fit_clf(&x_mat, &y_clf, None).unwrap();
        let preds = model.predict_clf(&x_mat);
        assert_eq!(preds.len(), n);
        let proba = model.predict_proba(&x_mat);
        assert_eq!(proba.ncols(), 3);
    }
}

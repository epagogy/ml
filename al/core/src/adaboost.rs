//! AdaBoost classifier — SAMME (Hastie, Rosset, Zhu & Zou 2009).
//!
//! Algorithm 2 (SAMME — discrete multi-class AdaBoost) from:
//!   "Multi-class AdaBoost", Statistics and Its Interface 2:349–360.
//!
//! Uses max-depth-1 CART stumps as weak learners.
//! Reuses build_tree() from cart.rs and traverse_leaf_info() from forest.rs.
//!
//! Key formulas:
//!   err_t  = sum_i w_i * I(h_t(x_i) != y_i)          (w normalized to sum=1)
//!   alpha_t = lr * (ln((1-err_t)/err_t) + ln(K-1))
//!   w_i   *= exp(alpha_t * I(h_t(x_i) != y_i))
//!   predict = argmax_k sum_t alpha_t * I(h_t(x) == k)

use crate::cart::{build_tree, ColMajorMatrix, Criterion, Node, TreeConfig};
use crate::error::MlError;
use crate::forest::{normalize_importances, traverse, traverse_leaf_info, SimpleRng};
use serde::{Deserialize, Serialize};

/// Serializable decision stump (nodes + probability pool).
#[derive(Serialize, Deserialize)]
struct StumpData {
    nodes: Vec<Node>,
    proba_pool: Vec<f64>,
}

/// AdaBoost classifier (SAMME).
#[derive(Serialize, Deserialize)]
pub struct AdaBoostModel {
    /// Number of boosting rounds. Default 50.
    pub n_estimators: usize,
    /// Shrinkage applied to each alpha_t. Default 1.0.
    pub learning_rate: f64,
    /// Number of classes (set during fit).
    pub n_classes: usize,
    /// Number of features (set during fit).
    pub n_features: usize,
    /// Feature importances: alpha-weighted MDI, normalized to sum=1.
    pub feature_importances: Vec<f64>,
    stumps: Vec<StumpData>,
    stump_weights: Vec<f64>,
    /// RNG seed for stump splitting.
    pub seed: u64,
}

impl AdaBoostModel {
    /// Create a new, unfitted `AdaBoostModel`.
    pub fn new(n_estimators: usize, learning_rate: f64, seed: u64) -> Self {
        Self {
            n_estimators,
            learning_rate,
            n_classes: 0,
            n_features: 0,
            feature_importances: Vec::new(),
            stumps: Vec::new(),
            stump_weights: Vec::new(),
            seed,
        }
    }

    // -----------------------------------------------------------------------
    // Fit
    // -----------------------------------------------------------------------

    /// Fit on row-major X (n × p) and 0-based integer class labels y.
    ///
    /// `sample_weight`: optional per-sample weights (unnormalized).
    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[usize],
        n: usize,
        p: usize,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), MlError> {
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if x.len() != n * p || y.len() != n {
            return Err(MlError::DimensionMismatch { expected: n, got: y.len() });
        }
        let k = y.iter().copied().max().unwrap_or(0) + 1;
        if k < 2 {
            return Err(MlError::EmptyData);
        }
        self.n_classes = k;
        self.n_features = p;

        // Build column-major matrix once (shared across all stumps)
        let cm = ColMajorMatrix::from_row_major_slice(x, n, p);
        let y_f64: Vec<f64> = y.iter().map(|&c| c as f64).collect();

        // Initialize weights: uniform or provided, normalize to sum = 1.
        let mut weights: Vec<f64> = match sample_weight {
            Some(sw) => {
                let s: f64 = sw.iter().sum();
                let scale = 1.0 / s.max(1e-300);
                sw.iter().map(|&wi| wi * scale).collect()
            }
            None => vec![1.0 / n as f64; n],
        };

        let mut rng = SimpleRng::new(self.seed);
        let mut all_importances: Vec<Vec<f64>> = Vec::new();
        let mut all_alpha: Vec<f64> = Vec::new();

        // SAMME guard: stop if error >= (1 - 1/k)
        let random_rate = 1.0 - 1.0 / k as f64;

        for _ in 0..self.n_estimators {
            let stump_seed = rng.next_u64();

            // Build depth-1 stump with current weights.
            // Pass cloned weights (build_tree consumes them).
            let stump_config = TreeConfig {
                max_depth: 1,
                min_samples_split: 2,
                min_samples_leaf: 1,
                histogram_threshold: usize::MAX, // always exact for stumps
                n_classes: k,
                is_clf: true,
                max_features: None, // all features
                rng_seed: stump_seed,
                criterion: Criterion::Gini,
                extra_trees: false,
                monotone_cst: None,
                min_impurity_decrease: 0.0, // no pruning for stumps
                split_lambda: 0.0,
            };
            let (nodes, proba_pool, importance_raw) = build_tree(
                &cm,
                &y_f64,
                weights.clone(),
                None, // no histogram quantization for depth-1 stumps
                &stump_config,
            );

            // Compute weighted misclassification rate.
            // weights already sum to 1 (maintained by normalization below).
            let err: f64 = (0..n)
                .filter(|&i| traverse(&nodes, &cm, i) as usize != y[i])
                .map(|i| weights[i])
                .sum();

            // SAMME early stop: stump worse than random
            if err >= random_rate {
                break;
            }

            // alpha_t = lr * (ln((1-err)/err) + ln(K-1))
            let err_safe = err.clamp(1e-300, 1.0 - 1e-10);
            let mut alpha = self.learning_rate
                * ((1.0 - err_safe) / err_safe).ln()
                + self.learning_rate * (k as f64 - 1.0).max(1.0).ln();
            alpha = alpha.min(10.0); // cap to prevent explosion when err ≈ 0

            // Update sample weights: misclassified *= exp(alpha)
            for i in 0..n {
                let pred = traverse(&nodes, &cm, i) as usize;
                if pred != y[i] {
                    weights[i] *= alpha.exp();
                }
                weights[i] = weights[i].max(1e-300); // floor to prevent degenerate weights
            }

            // Normalize weights to sum = 1 for next round
            let w_sum: f64 = weights.iter().sum();
            let w_scale = 1.0 / w_sum.max(1e-300);
            for w in &mut weights {
                *w *= w_scale;
            }

            all_importances.push(importance_raw);
            all_alpha.push(alpha);
            self.stumps.push(StumpData { nodes, proba_pool });
        }

        // Aggregate feature importances: alpha-weighted sum of per-stump MDI.
        let n_actual = all_alpha.len();
        if n_actual > 0 {
            let alpha_sum: f64 = all_alpha.iter().sum::<f64>().max(1e-300);
            let mut imp = vec![0.0_f64; p];
            for (t, &alpha) in all_alpha.iter().enumerate() {
                let raw = normalize_importances(&all_importances[t], p);
                for j in 0..p {
                    imp[j] += (alpha / alpha_sum) * raw[j];
                }
            }
            // Final normalization
            let imp_sum: f64 = imp.iter().sum::<f64>().max(1e-300);
            for v in &mut imp {
                *v /= imp_sum;
            }
            self.feature_importances = imp;
        } else {
            self.feature_importances = vec![1.0 / p as f64; p];
        }

        self.stump_weights = all_alpha;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------

    /// Predict 0-based class index for each row in row-major X (n × p).
    pub fn predict(&self, x: &[f64], n: usize, p: usize) -> Vec<usize> {
        if self.stumps.is_empty() {
            return vec![0; n];
        }
        let cm = ColMajorMatrix::from_row_major_slice(x, n, p);
        let k = self.n_classes;
        (0..n)
            .map(|i| {
                let mut scores = vec![0.0_f64; k];
                for (stump, &alpha) in self.stumps.iter().zip(self.stump_weights.iter()) {
                    let pred = traverse(&stump.nodes, &cm, i) as usize;
                    if pred < k {
                        scores[pred] += alpha;
                    }
                }
                scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict class probabilities (n × k, row-major).
    ///
    /// Uses alpha-weighted average of stump leaf probabilities, then softmax.
    pub fn predict_proba(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        if self.stumps.is_empty() {
            let k = self.n_classes.max(1);
            return vec![1.0 / k as f64; n * k];
        }
        let cm = ColMajorMatrix::from_row_major_slice(x, n, p);
        let k = self.n_classes;
        let alpha_sum: f64 = self.stump_weights.iter().sum::<f64>().max(1e-300);

        let mut out = vec![0.0_f64; n * k];
        for i in 0..n {
            let row = &mut out[i * k..(i + 1) * k];
            for (stump, &alpha) in self.stumps.iter().zip(self.stump_weights.iter()) {
                let (_, proba_offset) = traverse_leaf_info(&stump.nodes, &cm, i);
                let end = proba_offset + k;
                if end <= stump.proba_pool.len() {
                    let proba = &stump.proba_pool[proba_offset..end];
                    for j in 0..k {
                        row[j] += (alpha / alpha_sum) * proba[j];
                    }
                }
            }
            // Softmax normalization
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
            let sum_safe = sum.max(1e-300);
            for v in row.iter_mut() {
                *v = (*v - max).exp() / sum_safe;
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize to JSON (internal).
    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    /// Deserialize from JSON (internal).
    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

impl Default for AdaBoostModel {
    fn default() -> Self {
        Self::new(50, 1.0, 42)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn iris_tiny() -> (Vec<f64>, Vec<usize>) {
        // 15 rows, 4 features — 5 per class (0, 1, 2)
        let x = vec![
            5.1, 3.5, 1.4, 0.2,  // class 0
            4.9, 3.0, 1.4, 0.2,
            4.7, 3.2, 1.3, 0.2,
            4.6, 3.1, 1.5, 0.2,
            5.0, 3.6, 1.4, 0.2,
            7.0, 3.2, 4.7, 1.4,  // class 1
            6.4, 3.2, 4.5, 1.5,
            6.9, 3.1, 4.9, 1.5,
            5.5, 2.3, 4.0, 1.3,
            6.5, 2.8, 4.6, 1.5,
            6.3, 3.3, 6.0, 2.5,  // class 2
            5.8, 2.7, 5.1, 1.9,
            7.1, 3.0, 5.9, 2.1,
            6.3, 2.9, 5.6, 1.8,
            6.5, 3.0, 5.8, 2.2,
        ];
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        (x, y)
    }

    #[test]
    fn test_fit_predict_basic() {
        let (x, y) = iris_tiny();
        let mut m = AdaBoostModel::default();
        m.fit(&x, &y, 15, 4, None).unwrap();
        assert_eq!(m.n_classes, 3);
        assert_eq!(m.n_features, 4);
        assert!(!m.stumps.is_empty());

        let preds = m.predict(&x, 15, 4);
        assert_eq!(preds.len(), 15);
        // At least 80% accuracy on training data
        let correct = preds.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(correct >= 12, "Expected >= 12/15 correct, got {correct}/15");
    }

    #[test]
    fn test_predict_proba_shape() {
        let (x, y) = iris_tiny();
        let mut m = AdaBoostModel::default();
        m.fit(&x, &y, 15, 4, None).unwrap();
        let proba = m.predict_proba(&x, 15, 4);
        assert_eq!(proba.len(), 15 * 3);
        // Each row sums to ~1
        for i in 0..15 {
            let row_sum: f64 = proba[i * 3..(i + 1) * 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_feature_importances_normalized() {
        let (x, y) = iris_tiny();
        let mut m = AdaBoostModel::default();
        m.fit(&x, &y, 15, 4, None).unwrap();
        let imp_sum: f64 = m.feature_importances.iter().sum();
        assert!(
            (imp_sum - 1.0).abs() < 1e-6,
            "Feature importances should sum to 1, got {imp_sum}"
        );
    }

    #[test]
    fn test_json_roundtrip() {
        let (x, y) = iris_tiny();
        let mut m = AdaBoostModel::default();
        m.fit(&x, &y, 15, 4, None).unwrap();

        let json = m.to_json().unwrap();
        let m2 = AdaBoostModel::from_json(&json).unwrap();

        let p1 = m.predict(&x, 15, 4);
        let p2 = m2.predict(&x, 15, 4);
        assert_eq!(p1, p2, "Roundtrip predictions must match");
    }

    #[test]
    fn test_weighted_fit() {
        let (x, y) = iris_tiny();
        let weights = vec![1.0_f64; 15];
        let mut m1 = AdaBoostModel::new(10, 1.0, 42);
        let mut m2 = AdaBoostModel::new(10, 1.0, 42);
        m1.fit(&x, &y, 15, 4, None).unwrap();
        m2.fit(&x, &y, 15, 4, Some(&weights)).unwrap();

        let p1 = m1.predict(&x, 15, 4);
        let p2 = m2.predict(&x, 15, 4);
        assert_eq!(p1, p2, "Uniform weights must match no-weight");
    }
}

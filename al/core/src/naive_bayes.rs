//! Gaussian Naive Bayes classifier.
//!
//! Matches the Python `_NaiveBayesModel` reference implementation exactly:
//!
//! - Weighted/unweighted per-class mean + population variance.
//! - `epsilon = var_smoothing * max(global_var_per_feature)` (sklearn convention).
//!   Global variance derived from per-class sufficient statistics (single pass).
//! - Log-posterior prediction with log-sum-exp normalization.
//! - Handles `sample_weight` via weighted mean/variance.

use crate::error::MlError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Gaussian Naive Bayes classifier.
///
/// Parameters match sklearn's `GaussianNB`:
/// - `var_smoothing=1e-9`: epsilon = var_smoothing * max(global feature variances).
#[derive(Serialize, Deserialize)]
pub struct NaiveBayesModel {
    /// Portion of the largest global variance added to per-class variances
    /// for numerical stability.  Default 1e-9 (matches sklearn).
    pub var_smoothing: f64,
    /// Number of classes.  Set after `fit`.
    pub n_classes: usize,
    /// Number of input features.  Set after `fit`.
    pub n_features: usize,
    /// Per-class means, flat row-major [k * p].  theta[c * p + j] = mean of feature j in class c.
    theta: Vec<f64>,
    /// Per-class smoothed variances, flat row-major [k * p].
    var: Vec<f64>,
    /// Log class priors, length k.
    log_prior: Vec<f64>,
}

impl NaiveBayesModel {
    /// Create a new, unfitted `NaiveBayesModel`.
    pub fn new(var_smoothing: f64) -> Self {
        Self {
            var_smoothing,
            n_classes: 0,
            n_features: 0,
            theta: Vec::new(),
            var: Vec::new(),
            log_prior: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Fit
    // -----------------------------------------------------------------------

    /// Fit on row-major X (n × p) and integer labels y (0-based, 0..k-1).
    ///
    /// `sample_weight`: optional per-sample weights (unnormalized).
    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[i64],
        n: usize,
        p: usize,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), MlError> {
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch { expected: n, got: y.len() });
        }
        if x.len() != n * p {
            return Err(MlError::DimensionMismatch { expected: n * p, got: x.len() });
        }

        let k = (*y.iter().max().unwrap() + 1) as usize;
        self.n_classes = k;
        self.n_features = p;

        // ── Per-class sufficient statistics ──────────────────────────────────
        // For each class c: weighted sum, weighted sum-of-squares, total weight.
        let mut sum_w = vec![0.0_f64; k];          // total weight per class
        let mut sum_wx = vec![0.0_f64; k * p];     // weighted feature sums  [k*p]
        let mut sum_wx2 = vec![0.0_f64; k * p];    // weighted feature sq sums [k*p]

        for i in 0..n {
            let c = y[i] as usize;
            let w = sample_weight.map_or(1.0, |sw| sw[i]);
            sum_w[c] += w;
            let row = &x[i * p..(i + 1) * p];
            for j in 0..p {
                let xij = row[j];
                sum_wx[c * p + j] += w * xij;
                sum_wx2[c * p + j] += w * xij * xij;
            }
        }

        // Mean + population variance per class
        let mut theta = vec![0.0_f64; k * p];
        let mut var = vec![0.0_f64; k * p];

        for c in 0..k {
            let w_total = sum_w[c].max(1e-300); // guard against empty class
            for j in 0..p {
                let mu = sum_wx[c * p + j] / w_total;
                // Population variance: E[X^2] - E[X]^2
                let v = sum_wx2[c * p + j] / w_total - mu * mu;
                theta[c * p + j] = mu;
                var[c * p + j] = v.max(0.0); // clamp numerical negatives
            }
        }

        // ── Global variance for epsilon ───────────────────────────────────────
        // epsilon = var_smoothing * max(global_var_per_feature)
        // Derived from per-class sufficient statistics — avoids a second O(n*p) pass.
        let total_w: f64 = sum_w.iter().sum::<f64>().max(1e-300);

        let mut max_global_var = 0.0_f64;
        for j in 0..p {
            let mut gm = 0.0_f64;
            let mut gsq = 0.0_f64;
            for c in 0..k {
                gm += sum_wx[c * p + j];
                gsq += sum_wx2[c * p + j];
            }
            let mu = gm / total_w;
            let gv = (gsq / total_w - mu * mu).max(0.0);
            if gv > max_global_var {
                max_global_var = gv;
            }
        }
        let epsilon = self.var_smoothing * max_global_var;

        // Add epsilon and apply safety floor
        for v in &mut var {
            *v = (*v + epsilon).max(1e-300);
        }

        // ── Log priors ────────────────────────────────────────────────────────
        let total_w_sum: f64 = sum_w.iter().sum();
        let log_prior: Vec<f64> = sum_w
            .iter()
            .map(|&w| (w / total_w_sum).max(1e-300).ln())
            .collect();

        self.theta = theta;
        self.var = var;
        self.log_prior = log_prior;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------

    /// Return class probabilities for row-major X (n × p).
    /// Output: flat row-major [n * k], probabilities sum to 1.
    ///
    /// Parallelized across samples via rayon for large datasets.
    pub fn predict_proba(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        let k = self.n_classes;

        // Pre-compute log(var) once — saves n*k*p ln() calls during prediction.
        let log_var: Vec<f64> = self.var.iter().map(|v| v.ln()).collect();

        let out: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row = &x[i * p..(i + 1) * p];
                let mut log_proba = vec![0.0_f64; k];

                for c in 0..k {
                    let theta_c = &self.theta[c * p..(c + 1) * p];
                    let var_c = &self.var[c * p..(c + 1) * p];
                    let log_var_c = &log_var[c * p..(c + 1) * p];

                    let mut ll = self.log_prior[c];
                    for j in 0..p {
                        let diff = row[j] - theta_c[j];
                        ll -= 0.5 * (log_var_c[j] + diff * diff / var_c[j]);
                    }
                    log_proba[c] = ll;
                }

                // Log-sum-exp normalization
                let max_ll = log_proba.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut z = 0.0_f64;
                for lp in log_proba.iter_mut() {
                    *lp = (*lp - max_ll).exp();
                    z += *lp;
                }
                let z = z.max(1e-300);
                for lp in log_proba.iter_mut() {
                    *lp /= z;
                }
                log_proba
            })
            .collect();

        out
    }

    /// Return class indices for row-major X (n × p).
    pub fn predict_clf(&self, x: &[f64], n: usize, p: usize) -> Vec<i64> {
        let k = self.n_classes;
        let proba = self.predict_proba(x, n, p);
        (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &proba[i * k..(i + 1) * k];
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize to JSON (internal — not public API).
    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    /// Deserialize from JSON (internal).
    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

impl Default for NaiveBayesModel {
    fn default() -> Self {
        Self::new(1e-9)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn iris_tiny() -> (Vec<f64>, Vec<i64>) {
        // 9 rows: 3 samples × 3 classes, 2 features each
        let x = vec![
            5.1, 3.5,  // class 0
            4.9, 3.0,
            4.7, 3.2,
            7.0, 3.2,  // class 1
            6.4, 3.2,
            6.9, 3.1,
            6.3, 3.3,  // class 2
            5.8, 2.7,
            7.1, 3.0,
        ];
        let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        (x, y)
    }

    #[test]
    fn test_fit_predict_basic() {
        let (x, y) = iris_tiny();
        let mut m = NaiveBayesModel::default();
        m.fit(&x, &y, 9, 2, None).unwrap();

        assert_eq!(m.n_classes, 3);
        assert_eq!(m.n_features, 2);

        let preds = m.predict_clf(&x, 9, 2);
        // Most training samples should be correctly classified
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| **p == **t).count();
        assert!(correct >= 7, "Expected >= 7 correct, got {correct}");
    }

    #[test]
    fn test_proba_sums_to_one() {
        let (x, y) = iris_tiny();
        let mut m = NaiveBayesModel::default();
        m.fit(&x, &y, 9, 2, None).unwrap();

        let proba = m.predict_proba(&x, 9, 2);
        for i in 0..9 {
            let row_sum: f64 = proba[i * 3..i * 3 + 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "row {i} sums to {row_sum}");
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let (x, y) = iris_tiny();
        let mut m = NaiveBayesModel::default();
        m.fit(&x, &y, 9, 2, None).unwrap();

        let json = m.to_json().unwrap();
        let m2 = NaiveBayesModel::from_json(&json).unwrap();

        let p1 = m.predict_clf(&x, 9, 2);
        let p2 = m2.predict_clf(&x, 9, 2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_weighted_fit() {
        let (x, y) = iris_tiny();
        // Uniform weights should give same result as no weights
        let weights = vec![1.0_f64; 9];
        let mut m1 = NaiveBayesModel::default();
        let mut m2 = NaiveBayesModel::default();
        m1.fit(&x, &y, 9, 2, None).unwrap();
        m2.fit(&x, &y, 9, 2, Some(&weights)).unwrap();

        let p1 = m1.predict_clf(&x, 9, 2);
        let p2 = m2.predict_clf(&x, 9, 2);
        assert_eq!(p1, p2, "Uniform weights must match no-weight result");
    }

    #[test]
    fn test_binary_classification() {
        let x = vec![
            1.0, 2.0,
            1.5, 2.5,
            5.0, 6.0,
            5.5, 6.5,
        ];
        let y = vec![0, 0, 1, 1];
        let mut m = NaiveBayesModel::default();
        m.fit(&x, &y, 4, 2, None).unwrap();
        let preds = m.predict_clf(&x, 4, 2);
        assert_eq!(preds, vec![0, 0, 1, 1]);
    }
}

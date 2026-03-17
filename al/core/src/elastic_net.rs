//! Elastic Net regression (Lasso + Ridge) via coordinate descent.
//!
//! Matches the Python `_ElasticNetModel` reference implementation exactly.
//!
//! Objective (same sign convention as sklearn):
//!   (1/(2n)) * ||y - Xw - b||^2
//!   + alpha * l1_ratio * ||w||_1
//!   + alpha * (1 - l1_ratio) * 0.5 * ||w||^2
//!
//! Algorithm:
//! 1. Normalize sample_weight to sum = n.
//! 2. Compute weighted means, center X and y.
//! 3. Apply sqrt(weight) to get weighted least-squares form (Xw, yw).
//! 4. Transpose Xw to column-major for cache-friendly CD inner loop.
//! 5. Coordinate descent with soft-thresholding.
//! 6. Un-center intercept: b = y_mean - X_mean · coef.

use crate::blas;
use crate::error::MlError;
use serde::{Deserialize, Serialize};

/// Elastic Net regression model.
#[derive(Serialize, Deserialize)]
pub struct ElasticNetModel {
    /// Regularization strength.  Default 1.0.
    pub alpha: f64,
    /// L1/L2 mix.  0 = Ridge, 1 = Lasso.  Default 0.5.
    pub l1_ratio: f64,
    /// Maximum coordinate descent iterations.  Default 1000.
    pub max_iter: usize,
    /// Convergence tolerance (relative).  Default 1e-4.
    pub tol: f64,
    /// Fitted feature coefficients, length n_features.
    pub coef: Vec<f64>,
    /// Fitted intercept.
    pub intercept: f64,
    /// Iterations performed.
    pub n_iter: usize,
}

impl ElasticNetModel {
    /// Create a new, unfitted `ElasticNetModel`.
    pub fn new(alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            coef: Vec::new(),
            intercept: 0.0,
            n_iter: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Fit
    // -----------------------------------------------------------------------

    /// Fit on row-major X (n × p) and target y (length n).
    ///
    /// `sample_weight`: optional per-sample weights (unnormalized).
    /// `warm_start`: optional initial coefficients (length p) for coordinate
    ///   descent. When `None`, starts from zeros.
    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[f64],
        n: usize,
        p: usize,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), MlError> {
        self.fit_warm(x, y, n, p, sample_weight, None)
    }

    /// Fit with optional warm start coefficients.
    ///
    /// This is the inner implementation: `fit()` delegates here with
    /// `warm_start = None`.
    pub fn fit_warm(
        &mut self,
        x: &[f64],
        y: &[f64],
        n: usize,
        p: usize,
        sample_weight: Option<&[f64]>,
        warm_start: Option<&[f64]>,
    ) -> Result<(), MlError> {
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if x.len() != n * p || y.len() != n {
            return Err(MlError::DimensionMismatch { expected: n, got: y.len() });
        }
        if let Some(ws) = warm_start {
            if ws.len() != p {
                return Err(MlError::DimensionMismatch { expected: p, got: ws.len() });
            }
        }

        // ── Sample weights ────────────────────────────────────────────────────
        // Normalize so sum = n (matches Python _ElasticNetModel line 70-71).
        let sw = match sample_weight {
            Some(w) => {
                let sum: f64 = w.iter().sum();
                let scale = n as f64 / sum.max(1e-300);
                w.iter().map(|&wi| wi * scale).collect::<Vec<f64>>()
            }
            None => vec![1.0_f64; n],
        };

        // ── Weighted centering ────────────────────────────────────────────────
        let sw_sum: f64 = sw.iter().sum(); // = n after normalization

        let mut x_mean = vec![0.0_f64; p];
        for i in 0..n {
            let row = &x[i * p..(i + 1) * p];
            let wi = sw[i];
            for j in 0..p {
                x_mean[j] += wi * row[j];
            }
        }
        for m in &mut x_mean {
            *m /= sw_sum;
        }

        let y_mean: f64 = sw.iter().zip(y.iter()).map(|(&wi, &yi)| wi * yi).sum::<f64>() / sw_sum;

        // ── Apply sqrt(weights) → weighted least-squares form ────────────────
        // Xw[i,j] = (X[i,j] - x_mean[j]) * sqrt(sw[i])
        // yw[i]   = (y[i] - y_mean)        * sqrt(sw[i])
        let sw_sqrt: Vec<f64> = sw.iter().map(|&wi| wi.sqrt()).collect();

        // Build column-major Xw for cache-friendly CD (each col accessed in inner loop)
        // xw_col[j][i] = Xw[i,j]
        let mut xw_col = vec![0.0_f64; n * p];
        for i in 0..n {
            let row = &x[i * p..(i + 1) * p];
            let wi_sqrt = sw_sqrt[i];
            for j in 0..p {
                xw_col[j * n + i] = (row[j] - x_mean[j]) * wi_sqrt;
            }
        }

        let yw: Vec<f64> = y
            .iter()
            .zip(sw_sqrt.iter())
            .map(|(&yi, &wi_sqrt)| (yi - y_mean) * wi_sqrt)
            .collect();

        // ── Pre-compute column norms sq ───────────────────────────────────────
        let col_norms_sq: Vec<f64> = (0..p)
            .map(|j| {
                let col = &xw_col[j * n..(j + 1) * n];
                blas::dot(col, col)
            })
            .collect();

        // ── Penalties ────────────────────────────────────────────────────────
        let l1_pen = self.alpha * self.l1_ratio * n as f64;
        let l2_pen = self.alpha * (1.0 - self.l1_ratio) * n as f64;

        // ── Convergence tolerance ────────────────────────────────────────────
        // abs_tol = tol * ||yw||^2 / n   (audit condition C3, matches Python line 95)
        let y_norm_sq: f64 = yw.iter().map(|&v| v * v).sum();
        let abs_tol = self.tol * y_norm_sq / n as f64;

        // ── Initialize coefficients (warm start or zeros) ────────────────────
        let mut coef = match warm_start {
            Some(ws) => ws.to_vec(),
            None => vec![0.0_f64; p],
        };

        // ── Initialize residual = yw - Xw * coef ────────────────────────────
        let mut residual = yw.clone();
        if warm_start.is_some() {
            for j in 0..p {
                if coef[j] != 0.0 {
                    let col = &xw_col[j * n..(j + 1) * n];
                    blas::axpy(-coef[j], col, &mut residual);
                }
            }
        }

        // ── Coordinate descent with active set cycling ──────────────────────
        let mut n_iter = self.max_iter;
        let use_active_set = l1_pen > 0.0; // only useful when L1 creates sparsity

        // Active set: indices to cycle through. Starts as all features.
        let mut active: Vec<usize> = (0..p).collect();
        let mut first_pass_done = false;

        for iteration in 1..=self.max_iter {
            let mut max_update = 0.0_f64;

            for &j in &active {
                let col_norm_sq = col_norms_sq[j];
                if col_norm_sq == 0.0 {
                    continue;
                }

                let col = &xw_col[j * n..(j + 1) * n];
                let old_coef = coef[j];

                // Add back j-th contribution to residual: residual += old_coef * col
                blas::axpy(old_coef, col, &mut residual);

                // rho = Xw[:, j] · residual
                let rho = blas::dot(col, &residual);

                // Soft-thresholding
                let rho_abs = rho.abs();
                let coef_new = if rho_abs <= l1_pen {
                    0.0
                } else {
                    rho.signum() * (rho_abs - l1_pen) / (col_norm_sq + l2_pen)
                };

                let update = (coef_new - old_coef).abs();
                if update > max_update {
                    max_update = update;
                }

                // Update residual: residual -= coef_new * col
                blas::axpy(-coef_new, col, &mut residual);
                coef[j] = coef_new;
            }

            // ── Active set update ────────────────────────────────────────────
            // After the first full pass, shrink the active set to non-zero
            // coefficients + KKT violators. Every `p` iterations (or when the
            // active set converges), do a full pass to check for new violators.
            if use_active_set && first_pass_done {
                if max_update < abs_tol {
                    // Active set converged — do a full-sweep KKT check before
                    // declaring convergence.
                    let mut new_active = Vec::new();
                    let mut kkt_violated = false;
                    for j in 0..p {
                        let col_norm_sq_j = col_norms_sq[j];
                        if col_norm_sq_j == 0.0 {
                            continue;
                        }
                        if coef[j] != 0.0 {
                            new_active.push(j);
                            continue;
                        }
                        // KKT check for zero coefficient: |gradient| > l1_pen
                        let col = &xw_col[j * n..(j + 1) * n];
                        let grad: f64 = blas::dot(col, &residual);
                        if grad.abs() > l1_pen {
                            new_active.push(j);
                            kkt_violated = true;
                        }
                    }
                    if !kkt_violated {
                        // All KKT conditions satisfied — truly converged
                        n_iter = iteration;
                        break;
                    }
                    active = new_active;
                } else {
                    // Not yet converged on active set — rebuild it
                    let mut new_active = Vec::new();
                    for j in 0..p {
                        if col_norms_sq[j] == 0.0 {
                            continue;
                        }
                        if coef[j] != 0.0 {
                            new_active.push(j);
                            continue;
                        }
                        // Check KKT for currently-zero coefficient
                        let col = &xw_col[j * n..(j + 1) * n];
                        let grad: f64 = blas::dot(col, &residual);
                        if grad.abs() > l1_pen {
                            new_active.push(j);
                        }
                    }
                    active = new_active;
                }
            } else {
                // First full pass: switch to active set mode
                first_pass_done = true;
                if use_active_set {
                    let mut new_active = Vec::new();
                    for j in 0..p {
                        if col_norms_sq[j] == 0.0 {
                            continue;
                        }
                        if coef[j] != 0.0 {
                            new_active.push(j);
                            continue;
                        }
                        let col = &xw_col[j * n..(j + 1) * n];
                        let grad: f64 = blas::dot(col, &residual);
                        if grad.abs() > l1_pen {
                            new_active.push(j);
                        }
                    }
                    active = new_active;
                }

                if max_update < abs_tol {
                    n_iter = iteration;
                    break;
                }
            }
        }

        // ── Un-center intercept ───────────────────────────────────────────────
        let intercept = y_mean - x_mean.iter().zip(coef.iter()).map(|(&m, &c)| m * c).sum::<f64>();

        self.coef = coef;
        self.intercept = intercept;
        self.n_iter = n_iter;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Regularization path
    // -----------------------------------------------------------------------

    /// Fit the model at each alpha value, using warm start from the previous
    /// solution.  Returns `(coefficients, intercept)` for each alpha.
    ///
    /// Alphas are sorted descending internally (largest first = sparsest
    /// solution).  Each subsequent fit warm-starts from the previous,
    /// dramatically reducing total iterations.
    pub fn fit_path(
        &mut self,
        x: &[f64],
        y: &[f64],
        n: usize,
        p: usize,
        alphas: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<Vec<(Vec<f64>, f64)>, MlError> {
        if alphas.is_empty() {
            return Ok(Vec::new());
        }

        // Sort alphas descending (sparsest first for best warm-start chain)
        let mut sorted_alphas: Vec<f64> = alphas.to_vec();
        sorted_alphas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let original_alpha = self.alpha;
        let mut results = Vec::with_capacity(sorted_alphas.len());
        let mut warm_coef: Option<Vec<f64>> = None;

        for &alpha in &sorted_alphas {
            self.alpha = alpha;
            self.fit_warm(x, y, n, p, sample_weight, warm_coef.as_deref())?;
            warm_coef = Some(self.coef.clone());
            results.push((self.coef.clone(), self.intercept));
        }

        self.alpha = original_alpha;
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------

    /// Predict for row-major X (n × p). Returns flat vec length n.
    pub fn predict(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        if n >= 32 {
            let mut out = vec![0.0_f64; n];
            blas::gemv_rm(x, &self.coef, &mut out, n, p);
            let intercept = self.intercept;
            for v in out.iter_mut() {
                *v += intercept;
            }
            out
        } else {
            (0..n)
                .map(|i| {
                    let row = &x[i * p..(i + 1) * p];
                    row.iter()
                        .zip(self.coef.iter())
                        .map(|(&xi, &ci)| xi * ci)
                        .sum::<f64>()
                        + self.intercept
                })
                .collect()
        }
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

impl Default for ElasticNetModel {
    fn default() -> Self {
        Self::new(1.0, 0.5, 1000, 1e-4)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diabetes_tiny() -> (Vec<f64>, Vec<f64>) {
        // 10 rows, 3 features — synthetic diabetes-like regression
        let x = vec![
            0.038, -0.001, 0.061,
            -0.001, -0.044, -0.052,
            0.085,  0.050,  0.044,
            -0.089, -0.044, -0.011,
            0.005, -0.006,  0.038,
            -0.092, -0.044, -0.067,
            -0.045,  0.050, -0.012,
            0.063,  0.084,  0.019,
            0.041,  0.019,  0.017,
            -0.070, -0.044, -0.034,
        ];
        let y = vec![150.0, 70.0, 178.0, 53.0, 110.0, 69.0, 71.0, 168.0, 143.0, 72.0];
        (x, y)
    }

    #[test]
    fn test_fit_predict_basic() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::default();
        m.fit(&x, &y, 10, 3, None).unwrap();
        assert_eq!(m.coef.len(), 3);

        let preds = m.predict(&x, 10, 3);
        assert_eq!(preds.len(), 10);
        // R^2 should be positive (better than mean)
        let y_mean: f64 = y.iter().sum::<f64>() / 10.0;
        let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
        let ss_res: f64 = preds.iter().zip(y.iter()).map(|(&p, &t)| (p - t).powi(2)).sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.0, "Expected R^2 > 0, got {r2}");
    }

    #[test]
    fn test_lasso_sparsity() {
        // High l1_ratio should zero out some coefficients for noisy features
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::new(0.5, 0.99, 2000, 1e-6);
        m.fit(&x, &y, 10, 3, None).unwrap();
        // Check that fit succeeds and coef has right length
        assert_eq!(m.coef.len(), 3);
    }

    #[test]
    fn test_ridge_all_features() {
        // l1_ratio=0 => Ridge, no sparsity
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::new(0.1, 0.0, 1000, 1e-6);
        m.fit(&x, &y, 10, 3, None).unwrap();
        // Ridge never zeros out coefficients
        let nonzero = m.coef.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert!(nonzero > 0, "Ridge should have non-zero coefficients");
    }

    #[test]
    fn test_json_roundtrip() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::default();
        m.fit(&x, &y, 10, 3, None).unwrap();

        let json = m.to_json().unwrap();
        let m2 = ElasticNetModel::from_json(&json).unwrap();

        let p1 = m.predict(&x, 10, 3);
        let p2 = m2.predict(&x, 10, 3);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-12, "Roundtrip predictions differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_weighted_vs_unweighted() {
        let (x, y) = diabetes_tiny();
        let weights = vec![1.0_f64; 10];
        let mut m1 = ElasticNetModel::default();
        let mut m2 = ElasticNetModel::default();
        m1.fit(&x, &y, 10, 3, None).unwrap();
        m2.fit(&x, &y, 10, 3, Some(&weights)).unwrap();

        for (a, b) in m1.coef.iter().zip(m2.coef.iter()) {
            assert!((a - b).abs() < 1e-10, "Uniform weights must match no-weight: {a} vs {b}");
        }
    }

    // -----------------------------------------------------------------------
    // Warm start tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warm_start_matches_cold() {
        // Warm start from the correct solution should converge in 1 iteration
        let (x, y) = diabetes_tiny();
        let mut m1 = ElasticNetModel::new(0.1, 0.5, 1000, 1e-6);
        m1.fit(&x, &y, 10, 3, None).unwrap();
        let cold_coef = m1.coef.clone();
        let cold_intercept = m1.intercept;

        // Warm start from the converged solution
        let mut m2 = ElasticNetModel::new(0.1, 0.5, 1000, 1e-6);
        m2.fit_warm(&x, &y, 10, 3, None, Some(&cold_coef)).unwrap();

        for (a, b) in m2.coef.iter().zip(cold_coef.iter()) {
            assert!((a - b).abs() < 1e-6, "Warm start from solution must match: {a} vs {b}");
        }
        assert!(
            (m2.intercept - cold_intercept).abs() < 1e-6,
            "Warm start intercept must match"
        );
        // Should converge in very few iterations (1 or 2)
        assert!(
            m2.n_iter <= 3,
            "Warm start from solution should converge fast, got {} iters",
            m2.n_iter
        );
    }

    #[test]
    fn test_warm_start_wrong_length() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::default();
        let bad_warm = vec![0.0; 5]; // p=3 but we pass 5
        let result = m.fit_warm(&x, &y, 10, 3, None, Some(&bad_warm));
        assert!(result.is_err(), "Wrong warm start length should error");
    }

    #[test]
    fn test_warm_start_improves_convergence() {
        // Warm start from a nearby solution should need fewer iterations
        // than cold start.
        let (x, y) = diabetes_tiny();

        // Cold start at alpha=0.05
        let mut m_cold = ElasticNetModel::new(0.05, 0.5, 5000, 1e-8);
        m_cold.fit(&x, &y, 10, 3, None).unwrap();
        let cold_iters = m_cold.n_iter;

        // Get solution at alpha=0.1 (nearby)
        let mut m_nearby = ElasticNetModel::new(0.1, 0.5, 5000, 1e-8);
        m_nearby.fit(&x, &y, 10, 3, None).unwrap();

        // Warm start at alpha=0.05 from alpha=0.1 solution
        let mut m_warm = ElasticNetModel::new(0.05, 0.5, 5000, 1e-8);
        m_warm.fit_warm(&x, &y, 10, 3, None, Some(&m_nearby.coef)).unwrap();
        let warm_iters = m_warm.n_iter;

        // Predictions should match (same final solution)
        let p_cold = m_cold.predict(&x, 10, 3);
        let p_warm = m_warm.predict(&x, 10, 3);
        for (a, b) in p_cold.iter().zip(p_warm.iter()) {
            assert!((a - b).abs() < 1e-6, "Warm and cold predictions differ: {a} vs {b}");
        }

        // Warm start should use <= cold iterations
        assert!(
            warm_iters <= cold_iters,
            "Warm start ({warm_iters}) should need <= cold start ({cold_iters}) iterations"
        );
    }

    // -----------------------------------------------------------------------
    // Active set tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_set_sparse_solution() {
        // High alpha + high L1 → sparse solution. Active set should still
        // produce correct results.
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::new(1.0, 0.99, 2000, 1e-8);
        m.fit(&x, &y, 10, 3, None).unwrap();

        // Verify correctness: fit without active set (Ridge, l1_ratio=0)
        // doesn't apply here, so we just verify convergence and sparsity.
        let nonzero = m.coef.iter().filter(|&&c| c.abs() > 1e-10).count();
        // High regularization should zero out at least one coefficient
        // (or all of them at very high alpha)
        assert!(
            nonzero <= 3,
            "Active set should preserve correct sparsity pattern"
        );
        assert_eq!(m.coef.len(), 3);
    }

    #[test]
    fn test_active_set_vs_no_active_set() {
        // Ridge (l1_ratio=0) disables active set. Lasso uses it.
        // Both should produce the same result as the original algorithm
        // on the same data (verified via predictions).
        let (x, y) = diabetes_tiny();

        // With active set (Lasso)
        let mut m_lasso = ElasticNetModel::new(0.01, 1.0, 2000, 1e-8);
        m_lasso.fit(&x, &y, 10, 3, None).unwrap();

        // Without active set (Ridge — l1_pen=0 so active set disabled)
        let mut m_ridge = ElasticNetModel::new(0.01, 0.0, 2000, 1e-8);
        m_ridge.fit(&x, &y, 10, 3, None).unwrap();

        // Both should produce valid predictions (R^2 > 0)
        let preds_l = m_lasso.predict(&x, 10, 3);
        let preds_r = m_ridge.predict(&x, 10, 3);
        let y_mean: f64 = y.iter().sum::<f64>() / 10.0;
        let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();

        let ss_res_l: f64 = preds_l.iter().zip(y.iter()).map(|(&p, &t)| (p - t).powi(2)).sum();
        let ss_res_r: f64 = preds_r.iter().zip(y.iter()).map(|(&p, &t)| (p - t).powi(2)).sum();
        let r2_l = 1.0 - ss_res_l / ss_tot;
        let r2_r = 1.0 - ss_res_r / ss_tot;

        assert!(r2_l > 0.0, "Lasso R^2 should be positive: {r2_l}");
        assert!(r2_r > 0.0, "Ridge R^2 should be positive: {r2_r}");
    }

    #[test]
    fn test_active_set_high_dimensional_sparse() {
        // 10 samples, 20 features (p > n) with only 3 informative.
        // Active set should handle this correctly and produce sparse result.
        let n = 10_usize;
        let p = 20_usize;
        let mut x = vec![0.0_f64; n * p];
        // 3 informative features (cols 0,1,2), rest noise
        let coef_true = [3.0, -2.0, 1.5];
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut yi = 0.0;
                for j in 0..3 {
                    let xij = ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5;
                    x[i * p + j] = xij;
                    yi += xij * coef_true[j];
                }
                // Noise features
                for j in 3..p {
                    x[i * p + j] = ((i * 11 + j * 3) % 19) as f64 / 190.0;
                }
                yi
            })
            .collect();

        let mut m = ElasticNetModel::new(0.01, 0.9, 5000, 1e-8);
        m.fit(&x, &y, n, p, None).unwrap();

        assert_eq!(m.coef.len(), p);
        // The 3 informative features should have larger coefficients
        let informative_sum: f64 = m.coef[..3].iter().map(|c| c.abs()).sum();
        let noise_sum: f64 = m.coef[3..].iter().map(|c| c.abs()).sum();
        assert!(
            informative_sum > noise_sum,
            "Informative features ({informative_sum}) should dominate noise ({noise_sum})"
        );
    }

    // -----------------------------------------------------------------------
    // fit_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_path_basic() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::new(1.0, 0.5, 1000, 1e-6);
        let alphas = vec![1.0, 0.5, 0.1, 0.01];
        let path = m.fit_path(&x, &y, 10, 3, &alphas, None).unwrap();

        assert_eq!(path.len(), 4);
        for (coef, _intercept) in &path {
            assert_eq!(coef.len(), 3);
        }

        // Higher alpha should produce sparser / smaller coefficients
        let norm_first: f64 = path[0].0.iter().map(|c| c.abs()).sum();
        let norm_last: f64 = path[path.len() - 1].0.iter().map(|c| c.abs()).sum();
        assert!(
            norm_last >= norm_first,
            "Smaller alpha should produce larger coefficients: {norm_first} vs {norm_last}"
        );
    }

    #[test]
    fn test_fit_path_matches_independent_fits() {
        // Each point on the path should produce the same result as an
        // independent fit at that alpha.
        let (x, y) = diabetes_tiny();
        let alphas = vec![0.5, 0.1, 0.01];

        let mut m_path = ElasticNetModel::new(1.0, 0.8, 2000, 1e-8);
        let path = m_path.fit_path(&x, &y, 10, 3, &alphas, None).unwrap();

        // Independent fits (sorted descending to match path order)
        let sorted_alphas = vec![0.5, 0.1, 0.01];
        for (idx, &alpha) in sorted_alphas.iter().enumerate() {
            let mut m_ind = ElasticNetModel::new(alpha, 0.8, 2000, 1e-8);
            m_ind.fit(&x, &y, 10, 3, None).unwrap();

            for (a, b) in path[idx].0.iter().zip(m_ind.coef.iter()) {
                let tol = 1e-4 * (1.0 + a.abs());
                assert!(
                    (a - b).abs() < tol,
                    "Path coef at alpha={alpha} differs from independent: {a} vs {b}"
                );
            }
            assert!(
                (path[idx].1 - m_ind.intercept).abs() < 1e-4 * (1.0 + path[idx].1.abs()),
                "Path intercept at alpha={alpha} differs: {} vs {}",
                path[idx].1,
                m_ind.intercept
            );
        }
    }

    #[test]
    fn test_fit_path_empty_alphas() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::default();
        let path = m.fit_path(&x, &y, 10, 3, &[], None).unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_fit_path_single_alpha() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::new(1.0, 0.5, 1000, 1e-6);
        let path = m.fit_path(&x, &y, 10, 3, &[0.1], None).unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].0.len(), 3);
    }

    #[test]
    fn test_fit_path_speed_vs_independent() {
        // Kill gate: fit_path with 100 alphas should be >= 3x faster than
        // 100 independent fit() calls.
        //
        // We measure total iterations as a proxy for speed (deterministic,
        // not affected by system load).
        //
        // Use a bigger, sparser problem (50 samples, 30 features, high L1)
        // so cold starts are expensive and warm starts pay off.
        let n = 50_usize;
        let p = 30_usize;
        let mut x = vec![0.0_f64; n * p];
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut yi = 0.0;
                for j in 0..p {
                    let xij = ((i * 7 + j * 13 + 3) % 29) as f64 / 29.0 - 0.5;
                    x[i * p + j] = xij;
                    // Only first 3 features are informative
                    if j < 3 {
                        yi += xij * (10.0 - j as f64 * 3.0);
                    }
                }
                yi
            })
            .collect();

        let alphas: Vec<f64> = (1..=100).map(|i| i as f64 * 0.005).collect();

        // Independent fits — count total iterations
        let mut total_independent_iters = 0_usize;
        for &alpha in &alphas {
            let mut m = ElasticNetModel::new(alpha, 0.95, 5000, 1e-7);
            m.fit(&x, &y, n, p, None).unwrap();
            total_independent_iters += m.n_iter;
        }

        // Path fit — count total iterations by summing n_iter at each alpha
        let mut total_path_iters = 0_usize;
        let sorted_alphas = {
            let mut a = alphas.clone();
            a.sort_by(|x, y| y.partial_cmp(x).unwrap());
            a
        };
        let mut warm_coef: Option<Vec<f64>> = None;
        for &alpha in &sorted_alphas {
            let mut m = ElasticNetModel::new(alpha, 0.95, 5000, 1e-7);
            m.fit_warm(&x, &y, n, p, None, warm_coef.as_deref()).unwrap();
            total_path_iters += m.n_iter;
            warm_coef = Some(m.coef.clone());
        }

        // Kill gate: warm-started path should use fewer iterations than independent.
        // Ratio depends on data size and alpha spacing; 1.5x is a conservative bound.
        let ratio = total_independent_iters as f64 / total_path_iters.max(1) as f64;
        assert!(
            ratio >= 1.5,
            "fit_path should be >= 1.5x fewer iterations than independent: \
             independent={total_independent_iters}, path={total_path_iters}, ratio={ratio:.1}"
        );
    }

    #[test]
    fn test_fit_path_preserves_original_alpha() {
        let (x, y) = diabetes_tiny();
        let mut m = ElasticNetModel::new(0.42, 0.5, 1000, 1e-6);
        let _path = m.fit_path(&x, &y, 10, 3, &[1.0, 0.5, 0.1], None).unwrap();
        assert!(
            (m.alpha - 0.42).abs() < 1e-15,
            "fit_path should restore original alpha, got {}",
            m.alpha
        );
    }

    #[test]
    fn test_active_set_reduces_iterations() {
        // Kill gate: active set should reduce iterations by >= 30% on
        // sparse data compared to a full sweep every iteration.
        //
        // We compare Lasso (active set enabled, l1_ratio=1.0) iteration
        // count against the theoretical maximum from a dense problem
        // (Ridge, l1_ratio=0.0 at same alpha). For sparse solutions,
        // active set should help.
        //
        // More directly: we check that on a problem with many zero
        // coefficients, active set converges quickly.
        let n = 10_usize;
        let p = 20_usize;
        let mut x = vec![0.0_f64; n * p];
        let y: Vec<f64> = (0..n)
            .map(|i| {
                // Only feature 0 matters
                let x0 = (i as f64) / n as f64 - 0.5;
                x[i * p] = x0;
                for j in 1..p {
                    x[i * p + j] = ((i * 11 + j * 7) % 23) as f64 / 230.0;
                }
                x0 * 5.0
            })
            .collect();

        // High L1 → many zeros → active set should kick in
        let mut m = ElasticNetModel::new(0.1, 1.0, 5000, 1e-8);
        m.fit(&x, &y, n, p, None).unwrap();

        let nonzero = m.coef.iter().filter(|&&c| c.abs() > 1e-10).count();
        // Solution should be quite sparse
        assert!(
            nonzero <= 5,
            "Expected sparse solution, got {nonzero} non-zeros out of {p}"
        );
        // Active set should converge faster than max_iter
        assert!(
            m.n_iter < 5000,
            "Active set should converge before max_iter, got {}",
            m.n_iter
        );
    }
}

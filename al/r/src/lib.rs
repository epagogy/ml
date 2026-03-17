// ml-r: extendr bindings for ml crate.
//
// All fit functions return JSON strings (serialized model state).
// All predict functions accept JSON + column-major data, return R vectors.
// Survives saveRDS/readRDS naturally — no externalptr.
//
// Data layout:
//   R matrices are column-major. CART/Forest/Linear/Logistic use
//   DMatrix::from_column_slice() (zero-copy to nalgebra).
//   KNN expects row-major slices — col_to_row_major() transposes.

use extendr_api::prelude::*;
use ml_core::cart::{Criterion, DecisionTreeModel};
use ml_core::forest::RandomForestModel;
use ml_core::adaboost::AdaBoostModel;
use ml_core::elastic_net::ElasticNetModel;
use ml_core::gbt::GBTModel;
use ml_core::naive_bayes::NaiveBayesModel;
use ml_core::svm::{KernelType, SvmClassifier, SvmRegressor};
use ml_core::knn::{KnnModel, KnnWeights};
use ml_core::linear::LinearModel;
use ml_core::logistic::{LogisticModel, MultiClass};
use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Column-major → row-major transpose for KNN (which needs row-major input).
fn col_to_row_major(col: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut row = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            row[i * d + j] = col[j * n + i];
        }
    }
    row
}

/// R i32 labels → Rust i64 labels.
fn i32_to_i64(v: &[i32]) -> Vec<i64> {
    v.iter().map(|&x| x as i64).collect()
}

/// Rust i64 labels → R i32 labels.
fn i64_to_i32(v: &[i64]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

/// Optional sample_weight: R passes NULL or numeric vector.
fn nullable_sw(sw: Nullable<Vec<f64>>) -> Option<DVector<f64>> {
    match sw {
        Nullable::NotNull(w) => Some(DVector::from_column_slice(&w)),
        Nullable::Null => None,
    }
}

/// Parse criterion string from R to Criterion enum.
fn parse_criterion(s: &str) -> Criterion {
    match s {
        "entropy" => Criterion::Entropy,
        "mse" | "squared_error" => Criterion::MSE,
        "poisson" => Criterion::Poisson,
        _ => Criterion::Gini,
    }
}

/// Parse multi_class string from R to MultiClass enum.
fn parse_multi_class(s: &str) -> MultiClass {
    match s {
        "softmax" | "multinomial" => MultiClass::Softmax,
        _ => MultiClass::OvR,
    }
}

/// Parse weights string from R to KnnWeights enum.
fn parse_knn_weights(s: &str) -> KnnWeights {
    match s {
        "distance" => KnnWeights::Distance,
        _ => KnnWeights::Uniform,
    }
}

/// Optional monotone constraints: R passes NULL or integer vector.
fn nullable_monotone_cst(mc: Nullable<Vec<i32>>) -> Option<Vec<i8>> {
    match mc {
        Nullable::NotNull(v) => Some(v.iter().map(|&x| x as i8).collect()),
        Nullable::Null => None,
    }
}

// ---------------------------------------------------------------------------
// Availability check
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_available() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Linear Regression (Ridge)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_linear_fit(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64], alpha: f64,
    sample_weight: Nullable<Vec<f64>>,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_vec = DVector::from_column_slice(y);
    let sw = nullable_sw(sample_weight);
    let mut model = LinearModel::new(alpha);
    model
        .fit(&x_mat, &y_vec, sw.as_ref())
        .expect("Rust linear fit failed");
    model.to_json().expect("Rust linear serialize failed")
}

/// @export
#[extendr]
fn ml_rust_linear_predict(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = LinearModel::from_json(json).expect("linear deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    model.predict(&x_mat).as_slice().to_vec()
}

/// Returns [intercept, coef1, coef2, ...].
/// @export
#[extendr]
fn ml_rust_linear_coef(json: &str) -> Vec<f64> {
    let model = LinearModel::from_json(json).expect("linear deserialize failed");
    let mut v = vec![model.intercept];
    v.extend(model.coef.iter());
    v
}

// ---------------------------------------------------------------------------
// Logistic Regression (OvR + L-BFGS)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_logistic_fit(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32], c_param: f64, max_iter: i32,
    sample_weight: Nullable<Vec<f64>>,
    multi_class: &str,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_i64 = i32_to_i64(y);
    let sw = nullable_sw(sample_weight);
    let mut model = LogisticModel::new(c_param, max_iter as usize);
    model.multi_class = parse_multi_class(multi_class);
    model
        .fit(&x_mat, &y_i64, sw.as_ref())
        .expect("Rust logistic fit failed");
    model.to_json().expect("Rust logistic serialize failed")
}

/// @export
#[extendr]
fn ml_rust_logistic_predict(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = LogisticModel::from_json(json).expect("logistic deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    i64_to_i32(&model.predict(&x_mat))
}

/// Returns list(data=col_major_flat, nrow=n, ncol=k).
/// @export
#[extendr]
fn ml_rust_logistic_predict_proba(
    json: &str, x: &[f64], nrow: i32, ncol: i32,
) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = LogisticModel::from_json(json).expect("logistic deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let proba = model.predict_proba(&x_mat); // DMatrix, column-major
    let k = model.n_classes;
    let flat: Vec<f64> = proba.as_slice().to_vec();
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

/// Returns average absolute coefficients across OvR classifiers.
/// Output: [avg_abs_coef_feature_1, ..., avg_abs_coef_feature_p].
/// @export
#[extendr]
fn ml_rust_logistic_coef(json: &str) -> Vec<f64> {
    let model = LogisticModel::from_json(json).expect("logistic deserialize failed");
    if model.coefs.is_empty() {
        return vec![];
    }
    let n_features = model.coefs[0].len() - 1; // coefs[k][0] = bias
    let mut avg = vec![0.0; n_features];
    for coef_vec in &model.coefs {
        for j in 0..n_features {
            avg[j] += coef_vec[j + 1].abs(); // skip bias at [0]
        }
    }
    let n_classes = model.coefs.len() as f64;
    for v in &mut avg {
        *v /= n_classes;
    }
    avg
}

// ---------------------------------------------------------------------------
// Decision Tree (CART)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_tree_fit_clf(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    max_depth: i32, min_samples_split: i32, min_samples_leaf: i32,
    seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    criterion: &str,
    min_impurity_decrease: f64,
    ccp_alpha: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_i64 = i32_to_i64(y);
    let sw = nullable_sw(sample_weight);
    let mut model = DecisionTreeModel::new(
        max_depth as usize,
        min_samples_split as usize,
        min_samples_leaf as usize,
    );
    model.rng_seed = seed as u64;
    model.min_impurity_decrease = min_impurity_decrease;
    model.ccp_alpha = ccp_alpha;
    model
        .fit_clf(&x_mat, &y_i64, sw.as_ref(), parse_criterion(criterion))
        .expect("Rust CART clf fit failed");
    model.to_json().expect("Rust CART serialize failed")
}

/// @export
#[extendr]
fn ml_rust_tree_fit_reg(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64],
    max_depth: i32, min_samples_split: i32, min_samples_leaf: i32,
    seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    criterion: &str,
    monotone_cst: Nullable<Vec<i32>>,
    min_impurity_decrease: f64,
    ccp_alpha: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_vec = DVector::from_column_slice(y);
    let sw = nullable_sw(sample_weight);
    let mut model = DecisionTreeModel::new(
        max_depth as usize,
        min_samples_split as usize,
        min_samples_leaf as usize,
    );
    model.rng_seed = seed as u64;
    model.monotone_cst = nullable_monotone_cst(monotone_cst);
    model.min_impurity_decrease = min_impurity_decrease;
    model.ccp_alpha = ccp_alpha;
    model
        .fit_reg(&x_mat, &y_vec, sw.as_ref(), parse_criterion(criterion))
        .expect("Rust CART reg fit failed");
    model.to_json().expect("Rust CART serialize failed")
}

/// @export
#[extendr]
fn ml_rust_tree_predict_clf(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = DecisionTreeModel::from_json(json).expect("CART deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    i64_to_i32(&model.predict_clf(&x_mat))
}

/// @export
#[extendr]
fn ml_rust_tree_predict_reg(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = DecisionTreeModel::from_json(json).expect("CART deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    model.predict_reg(&x_mat).as_slice().to_vec()
}

/// @export
#[extendr]
fn ml_rust_tree_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = DecisionTreeModel::from_json(json).expect("CART deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let proba = model.predict_proba(&x_mat);
    let k = model.n_classes;
    let flat: Vec<f64> = proba.as_slice().to_vec();
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

/// @export
#[extendr]
fn ml_rust_tree_importances(json: &str) -> Vec<f64> {
    let model = DecisionTreeModel::from_json(json).expect("CART deserialize failed");
    model.feature_importances.clone()
}

// ---------------------------------------------------------------------------
// Random Forest
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_forest_fit_clf(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    n_trees: i32, max_depth: i32,
    min_samples_split: i32, min_samples_leaf: i32,
    seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    criterion: &str,
    min_impurity_decrease: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_i64 = i32_to_i64(y);
    let sw = nullable_sw(sample_weight);
    let mut model = RandomForestModel::new(
        n_trees as usize,
        max_depth as usize,
        min_samples_split as usize,
        min_samples_leaf as usize,
        seed as u64,
    );
    model.compute_oob = false; // R wrapper doesn't expose OOB; skip traversal
    model.criterion = parse_criterion(criterion);
    model.min_impurity_decrease = min_impurity_decrease;
    model
        .fit_clf(&x_mat, &y_i64, sw.as_ref())
        .expect("Rust RF clf fit failed");
    model.to_json().expect("Rust RF serialize failed")
}

/// @export
#[extendr]
fn ml_rust_forest_fit_reg(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64],
    n_trees: i32, max_depth: i32,
    min_samples_split: i32, min_samples_leaf: i32,
    seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    criterion: &str,
    monotone_cst: Nullable<Vec<i32>>,
    min_impurity_decrease: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_vec = DVector::from_column_slice(y);
    let sw = nullable_sw(sample_weight);
    let mut model = RandomForestModel::new(
        n_trees as usize,
        max_depth as usize,
        min_samples_split as usize,
        min_samples_leaf as usize,
        seed as u64,
    );
    model.compute_oob = false; // R wrapper doesn't expose OOB; skip traversal
    model.criterion = parse_criterion(criterion);
    model.monotone_cst = nullable_monotone_cst(monotone_cst);
    model.min_impurity_decrease = min_impurity_decrease;
    model
        .fit_reg(&x_mat, &y_vec, sw.as_ref())
        .expect("Rust RF reg fit failed");
    model.to_json().expect("Rust RF serialize failed")
}

// ---------------------------------------------------------------------------
// Extra Trees (Random Forest with extra_trees=true, bootstrap=false)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_extra_trees_fit_clf(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    n_trees: i32, max_depth: i32,
    min_samples_split: i32, min_samples_leaf: i32,
    seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    criterion: &str,
    min_impurity_decrease: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_i64 = i32_to_i64(y);
    let sw = nullable_sw(sample_weight);
    let mut model = RandomForestModel::new(
        n_trees as usize,
        max_depth as usize,
        min_samples_split as usize,
        min_samples_leaf as usize,
        seed as u64,
    );
    model.compute_oob = false;
    model.extra_trees = true;
    model.criterion = parse_criterion(criterion);
    model.min_impurity_decrease = min_impurity_decrease;
    model
        .fit_clf(&x_mat, &y_i64, sw.as_ref())
        .expect("Rust ExtraTrees clf fit failed");
    model.to_json().expect("Rust ExtraTrees serialize failed")
}

/// @export
#[extendr]
fn ml_rust_extra_trees_fit_reg(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64],
    n_trees: i32, max_depth: i32,
    min_samples_split: i32, min_samples_leaf: i32,
    seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    criterion: &str,
    monotone_cst: Nullable<Vec<i32>>,
    min_impurity_decrease: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_vec = DVector::from_column_slice(y);
    let sw = nullable_sw(sample_weight);
    let mut model = RandomForestModel::new(
        n_trees as usize,
        max_depth as usize,
        min_samples_split as usize,
        min_samples_leaf as usize,
        seed as u64,
    );
    model.compute_oob = false;
    model.extra_trees = true;
    model.criterion = parse_criterion(criterion);
    model.monotone_cst = nullable_monotone_cst(monotone_cst);
    model.min_impurity_decrease = min_impurity_decrease;
    model
        .fit_reg(&x_mat, &y_vec, sw.as_ref())
        .expect("Rust ExtraTrees reg fit failed");
    model.to_json().expect("Rust ExtraTrees serialize failed")
}

/// @export
#[extendr]
fn ml_rust_forest_predict_clf(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = RandomForestModel::from_json(json).expect("RF deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    i64_to_i32(&model.predict_clf(&x_mat))
}

/// @export
#[extendr]
fn ml_rust_forest_predict_reg(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = RandomForestModel::from_json(json).expect("RF deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    model.predict_reg(&x_mat).as_slice().to_vec()
}

/// @export
#[extendr]
fn ml_rust_forest_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = RandomForestModel::from_json(json).expect("RF deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let proba = model.predict_proba(&x_mat);
    let k = model.n_classes;
    let flat: Vec<f64> = proba.as_slice().to_vec();
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

/// @export
#[extendr]
fn ml_rust_forest_importances(json: &str) -> Vec<f64> {
    let model = RandomForestModel::from_json(json).expect("RF deserialize failed");
    model.feature_importances.clone()
}

// ---------------------------------------------------------------------------
// KNN (KD-tree + brute-force hybrid)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_knn_fit_clf(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32], k: i32, weights: &str,
) -> String {
    let (n, d) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, d);
    let y_i64 = i32_to_i64(y);
    let mut model = KnnModel::new(k as usize);
    model.weights = parse_knn_weights(weights);
    model
        .fit_clf(&x_row, n, d, &y_i64)
        .expect("Rust KNN clf fit failed");
    model.to_json().expect("Rust KNN serialize failed")
}

/// @export
#[extendr]
fn ml_rust_knn_fit_reg(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64], k: i32, weights: &str,
) -> String {
    let (n, d) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, d);
    let mut model = KnnModel::new(k as usize);
    model.weights = parse_knn_weights(weights);
    model
        .fit_reg(&x_row, n, d, y)
        .expect("Rust KNN reg fit failed");
    model.to_json().expect("Rust KNN serialize failed")
}

/// @export
#[extendr]
fn ml_rust_knn_predict_clf(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, d) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, d);
    let model = KnnModel::from_json(json).expect("KNN deserialize failed");
    i64_to_i32(&model.predict_clf(&x_row, n, d))
}

/// @export
#[extendr]
fn ml_rust_knn_predict_reg(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, d) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, d);
    let model = KnnModel::from_json(json).expect("KNN deserialize failed");
    model.predict_reg(&x_row, n, d)
}

/// @export
#[extendr]
fn ml_rust_knn_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, d) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, d);
    let model = KnnModel::from_json(json).expect("KNN deserialize failed");
    let nc = model.n_classes;
    let flat_rm = model.predict_proba(&x_row, n, d); // row-major
    // Convert row-major → column-major for R
    let mut col_major = vec![0.0; n * nc];
    for i in 0..n {
        for j in 0..nc {
            col_major[j * n + i] = flat_rm[i * nc + j];
        }
    }
    list!(data = col_major, nrow = n as i32, ncol = nc as i32).into()
}

// ---------------------------------------------------------------------------
// Gradient-Boosted Trees (GBT)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_gbt_fit_clf(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    n_estimators: i32, learning_rate: f64, max_depth: i32,
    min_samples_split: i32, min_samples_leaf: i32,
    subsample: f64, seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    reg_lambda: f64, gamma: f64, colsample_bytree: f64,
    min_child_weight: f64, n_iter_no_change: Nullable<i32>,
    validation_fraction: f64,
    colsample_bynode: f64, scale_pos_weight: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let y_i64 = i32_to_i64(y);
    let sw = nullable_sw(sample_weight);
    let mut model = GBTModel::new(
        n_estimators as usize, learning_rate, max_depth as usize,
        min_samples_split as usize, min_samples_leaf as usize,
        subsample, seed as u64,
    );
    model.lambda = reg_lambda;
    model.gamma = gamma;
    model.colsample_bytree = colsample_bytree;
    model.min_child_weight = min_child_weight;
    model.colsample_bynode = colsample_bynode;
    model.scale_pos_weight = scale_pos_weight;
    model.n_iter_no_change = match n_iter_no_change {
        Nullable::NotNull(v) => Some(v as usize),
        Nullable::Null => None,
    };
    model.validation_fraction = validation_fraction;
    model.fit_clf(&x_mat, &y_i64, sw.as_ref())
        .expect("Rust GBT clf fit failed");
    model.to_json().expect("Rust GBT serialize failed")
}

/// @export
#[extendr]
fn ml_rust_gbt_fit_reg(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64],
    n_estimators: i32, learning_rate: f64, max_depth: i32,
    min_samples_split: i32, min_samples_leaf: i32,
    subsample: f64, seed: i32,
    sample_weight: Nullable<Vec<f64>>,
    reg_lambda: f64, gamma: f64, colsample_bytree: f64,
    min_child_weight: f64, n_iter_no_change: Nullable<i32>,
    validation_fraction: f64,
    colsample_bynode: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let sw = nullable_sw(sample_weight);
    let mut model = GBTModel::new(
        n_estimators as usize, learning_rate, max_depth as usize,
        min_samples_split as usize, min_samples_leaf as usize,
        subsample, seed as u64,
    );
    model.lambda = reg_lambda;
    model.gamma = gamma;
    model.colsample_bytree = colsample_bytree;
    model.min_child_weight = min_child_weight;
    model.colsample_bynode = colsample_bynode;
    model.n_iter_no_change = match n_iter_no_change {
        Nullable::NotNull(v) => Some(v as usize),
        Nullable::Null => None,
    };
    model.validation_fraction = validation_fraction;
    model.fit_reg(&x_mat, y, sw.as_ref())
        .expect("Rust GBT reg fit failed");
    model.to_json().expect("Rust GBT serialize failed")
}

/// @export
#[extendr]
fn ml_rust_gbt_predict_clf(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = GBTModel::from_json(json).expect("GBT deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    i64_to_i32(&model.predict_clf(&x_mat))
}

/// @export
#[extendr]
fn ml_rust_gbt_predict_reg(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = GBTModel::from_json(json).expect("GBT deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    model.predict_reg(&x_mat)
}

/// Returns list(data=col_major_flat, nrow=n, ncol=k).
/// @export
#[extendr]
fn ml_rust_gbt_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let model = GBTModel::from_json(json).expect("GBT deserialize failed");
    let x_mat = DMatrix::from_column_slice(n, p, x);
    let proba = model.predict_proba(&x_mat); // DMatrix n×k, column-major
    let k = model.n_classes;
    let flat: Vec<f64> = proba.as_slice().to_vec();
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

/// @export
#[extendr]
fn ml_rust_gbt_importances(json: &str) -> Vec<f64> {
    let model = GBTModel::from_json(json).expect("GBT deserialize failed");
    model.feature_importances.clone()
}

// ---------------------------------------------------------------------------
// Naive Bayes
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_nb_fit(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    var_smoothing: f64,
    sample_weight: Nullable<Vec<f64>>,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p); // R col-major → row-major
    let y_i64 = i32_to_i64(y);
    let sw: Option<Vec<f64>> = match sample_weight {
        Nullable::NotNull(w) => Some(w),
        Nullable::Null => None,
    };
    let mut model = NaiveBayesModel::new(var_smoothing);
    model.fit(&x_row, &y_i64, n, p, sw.as_deref())
        .expect("Rust NaiveBayes fit failed");
    model.to_json().expect("Rust NaiveBayes serialize failed")
}

/// @export
#[extendr]
fn ml_rust_nb_predict(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = NaiveBayesModel::from_json(json).expect("NaiveBayes deserialize failed");
    i64_to_i32(&model.predict_clf(&x_row, n, p))
}

/// Returns list(data=row_major_flat, nrow=n, ncol=k).
/// @export
#[extendr]
fn ml_rust_nb_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = NaiveBayesModel::from_json(json).expect("NaiveBayes deserialize failed");
    let flat = model.predict_proba(&x_row, n, p); // row-major [n, k]
    let k = model.n_classes;
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

// ---------------------------------------------------------------------------
// Elastic Net (regression)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_en_fit(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64],
    alpha: f64, l1_ratio: f64, max_iter: i32, tol: f64,
    sample_weight: Nullable<Vec<f64>>,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let sw: Option<Vec<f64>> = match sample_weight {
        Nullable::NotNull(w) => Some(w),
        Nullable::Null => None,
    };
    let mut model = ElasticNetModel::new(alpha, l1_ratio, max_iter as usize, tol);
    model.fit(&x_row, y, n, p, sw.as_deref())
        .expect("Rust ElasticNet fit failed");
    model.to_json().expect("Rust ElasticNet serialize failed")
}

/// @export
#[extendr]
fn ml_rust_en_predict(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = ElasticNetModel::from_json(json).expect("ElasticNet deserialize failed");
    model.predict(&x_row, n, p)
}

// ---------------------------------------------------------------------------
// AdaBoost (SAMME, classification only)
// ---------------------------------------------------------------------------

/// @export
#[extendr]
fn ml_rust_ada_fit(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    n_estimators: i32, learning_rate: f64, seed: i32,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p); // AdaBoost expects row-major
    let y_usize: Vec<usize> = y.iter().map(|&v| v as usize).collect();
    let mut model = AdaBoostModel::new(n_estimators as usize, learning_rate, seed as u64);
    model.fit(&x_row, &y_usize, n, p, None)
        .expect("Rust AdaBoost fit failed");
    model.to_json().expect("Rust AdaBoost serialize failed")
}

/// @export
#[extendr]
fn ml_rust_ada_predict(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = AdaBoostModel::from_json(json).expect("AdaBoost deserialize failed");
    model.predict(&x_row, n, p).iter().map(|&v| v as i32).collect()
}

/// Returns list(data=row_major_flat, nrow=n, ncol=k).
/// @export
#[extendr]
fn ml_rust_ada_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = AdaBoostModel::from_json(json).expect("AdaBoost deserialize failed");
    let flat = model.predict_proba(&x_row, n, p); // row-major [n, k]
    let k = model.n_classes;
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

/// @export
#[extendr]
fn ml_rust_ada_importances(json: &str) -> Vec<f64> {
    let model = AdaBoostModel::from_json(json).expect("AdaBoost deserialize failed");
    model.feature_importances.clone()
}

// ---------------------------------------------------------------------------
// SVM (linear/RBF/poly kernel: classifier OvR + epsilon-SVR)
// ---------------------------------------------------------------------------

/// Parse kernel string + params into KernelType.
/// gamma_val < 0.0 means "scale" mode (auto-compute from data).
fn parse_svm_kernel(kernel: &str, gamma_val: f64, degree: i32, coef0: f64) -> KernelType {
    match kernel {
        "rbf" => KernelType::RBF { gamma: gamma_val },
        "poly" | "polynomial" => KernelType::Polynomial {
            gamma: gamma_val,
            coef0,
            degree: degree as u32,
        },
        _ => KernelType::Linear,
    }
}

/// Compute "scale" gamma: 1 / (p * X_variance). Applies when gamma < 0.
fn resolve_svm_gamma(kt: &KernelType, x: &[f64], n: usize, p: usize) -> KernelType {
    let scale_gamma = || -> f64 {
        if n == 0 || p == 0 { return 1.0; }
        let total = (n * p) as f64;
        let mean: f64 = x.iter().sum::<f64>() / total;
        let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / total;
        if var < 1e-15 { 1.0 } else { 1.0 / (p as f64 * var) }
    };
    match kt {
        KernelType::RBF { gamma } if *gamma < 0.0 => {
            KernelType::RBF { gamma: scale_gamma() }
        }
        KernelType::Polynomial { gamma, coef0, degree } if *gamma < 0.0 => {
            KernelType::Polynomial { gamma: scale_gamma(), coef0: *coef0, degree: *degree }
        }
        other => other.clone(),
    }
}

/// @export
#[extendr]
fn ml_rust_svm_fit_clf(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[i32],
    c: f64, tol: f64, max_iter: i32,
    sample_weight: Nullable<Vec<f64>>,
    kernel: &str,
    gamma: f64,
    degree: i32,
    coef0: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    // R matrices are column-major; SVM expects row-major.
    let x_row = col_to_row_major(x, n, p);
    let y_i64 = i32_to_i64(y);
    let sw: Option<Vec<f64>> = match sample_weight {
        Nullable::NotNull(w) => Some(w),
        Nullable::Null => None,
    };
    let mut kt = parse_svm_kernel(kernel, gamma, degree, coef0);
    kt = resolve_svm_gamma(&kt, &x_row, n, p);
    let mut model = SvmClassifier::with_kernel(c, tol, max_iter as usize, kt);
    model.fit(&x_row, &y_i64, n, p, sw.as_deref())
        .expect("Rust SvmClassifier fit failed");
    model.to_json().expect("Rust SvmClassifier serialize failed")
}

/// @export
#[extendr]
fn ml_rust_svm_predict_clf(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<i32> {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = SvmClassifier::from_json(json).expect("SvmClassifier deserialize failed");
    i64_to_i32(&model.predict(&x_row, n, p))
}

/// Returns list(data=row_major_flat, nrow=n, ncol=k).
/// @export
#[extendr]
fn ml_rust_svm_predict_proba(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Robj {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = SvmClassifier::from_json(json).expect("SvmClassifier deserialize failed");
    let flat = model.predict_proba(&x_row, n, p); // row-major [n, k]
    let k = model.n_classes;
    list!(data = flat, nrow = n as i32, ncol = k as i32).into()
}

/// @export
#[extendr]
fn ml_rust_svm_fit_reg(
    x: &[f64], nrow: i32, ncol: i32,
    y: &[f64],
    c: f64, epsilon: f64, tol: f64, max_iter: i32,
    sample_weight: Nullable<Vec<f64>>,
    kernel: &str,
    gamma: f64,
    degree: i32,
    coef0: f64,
) -> String {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let sw: Option<Vec<f64>> = match sample_weight {
        Nullable::NotNull(w) => Some(w),
        Nullable::Null => None,
    };
    let mut kt = parse_svm_kernel(kernel, gamma, degree, coef0);
    kt = resolve_svm_gamma(&kt, &x_row, n, p);
    let mut model = SvmRegressor::with_kernel(c, epsilon, tol, max_iter as usize, kt);
    model.fit(&x_row, y, n, p, sw.as_deref())
        .expect("Rust SvmRegressor fit failed");
    model.to_json().expect("Rust SvmRegressor serialize failed")
}

/// @export
#[extendr]
fn ml_rust_svm_predict_reg(json: &str, x: &[f64], nrow: i32, ncol: i32) -> Vec<f64> {
    let (n, p) = (nrow as usize, ncol as usize);
    let x_row = col_to_row_major(x, n, p);
    let model = SvmRegressor::from_json(json).expect("SvmRegressor deserialize failed");
    model.predict(&x_row, n, p)
}

// ---------------------------------------------------------------------------
// Shuffle (cross-language deterministic RNG)
// ---------------------------------------------------------------------------

/// Deterministic shuffle of [0, 1, ..., n-1] using PCG-XSH-RR.
/// Same (n, seed) → same permutation on all platforms, all languages.
/// Returns 1-based indices for R.
/// @export
#[extendr]
fn ml_rust_shuffle(n: i32, seed: i64) -> Vec<i32> {
    ml_core::shuffle::shuffle(n as usize, seed as u64)
        .into_iter()
        .map(|i| (i + 1) as i32) // 0-based → 1-based for R
        .collect()
}

/// Canonical partition sizes: (n_train, n_valid, n_test).
/// Uses round(n * ratio) — matches Python and Julia.
/// @export
#[extendr]
fn ml_rust_partition_sizes(n: i32, train: f64, valid: f64, test: f64) -> Vec<i32> {
    let (t, v, te) = ml_core::shuffle::partition_sizes(n as usize, [train, valid, test]);
    vec![t as i32, v as i32, te as i32]
}

// ---------------------------------------------------------------------------
// Module registration — generates R_init_ml()
// ---------------------------------------------------------------------------

extendr_module! {
    mod ml_rust;
    fn ml_rust_available;
    fn ml_rust_shuffle;
    fn ml_rust_partition_sizes;
    fn ml_rust_linear_fit;
    fn ml_rust_linear_predict;
    fn ml_rust_linear_coef;
    fn ml_rust_logistic_fit;
    fn ml_rust_logistic_predict;
    fn ml_rust_logistic_predict_proba;
    fn ml_rust_logistic_coef;
    fn ml_rust_tree_fit_clf;
    fn ml_rust_tree_fit_reg;
    fn ml_rust_tree_predict_clf;
    fn ml_rust_tree_predict_reg;
    fn ml_rust_tree_predict_proba;
    fn ml_rust_tree_importances;
    fn ml_rust_forest_fit_clf;
    fn ml_rust_forest_fit_reg;
    fn ml_rust_forest_predict_clf;
    fn ml_rust_forest_predict_reg;
    fn ml_rust_forest_predict_proba;
    fn ml_rust_forest_importances;
    fn ml_rust_extra_trees_fit_clf;
    fn ml_rust_extra_trees_fit_reg;
    fn ml_rust_knn_fit_clf;
    fn ml_rust_knn_fit_reg;
    fn ml_rust_knn_predict_clf;
    fn ml_rust_knn_predict_reg;
    fn ml_rust_knn_predict_proba;
    fn ml_rust_gbt_fit_clf;
    fn ml_rust_gbt_fit_reg;
    fn ml_rust_gbt_predict_clf;
    fn ml_rust_gbt_predict_reg;
    fn ml_rust_gbt_predict_proba;
    fn ml_rust_gbt_importances;
    fn ml_rust_nb_fit;
    fn ml_rust_nb_predict;
    fn ml_rust_nb_predict_proba;
    fn ml_rust_en_fit;
    fn ml_rust_en_predict;
    fn ml_rust_ada_fit;
    fn ml_rust_ada_predict;
    fn ml_rust_ada_predict_proba;
    fn ml_rust_ada_importances;
    fn ml_rust_svm_fit_clf;
    fn ml_rust_svm_predict_clf;
    fn ml_rust_svm_predict_proba;
    fn ml_rust_svm_fit_reg;
    fn ml_rust_svm_predict_reg;
}

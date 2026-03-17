use crate::{dmat_to_flat, to_dvec};
use ml::gbt::{GBTModel, GrowPolicy};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Gradient-Boosted Trees
// ---------------------------------------------------------------------------

/// Gradient-Boosted Trees (classification + regression).
///
/// Parameters
/// ----------
/// n_estimators : int, default 100
/// learning_rate : float, default 0.1
/// max_depth : int, default 3
/// min_samples_split : int, default 2
/// min_samples_leaf : int, default 1
/// subsample : float, default 1.0
///     Row subsampling fraction per tree. 1.0 = no subsampling.
/// seed : int, default 42
/// reg_lambda : float, default 0.0
///     L2 regularization on leaf weights.
/// gamma : float, default 0.0
///     Minimum loss reduction required to make a further partition.
/// colsample_bytree : float, default 1.0
///     Column subsampling fraction per tree. 1.0 = no subsampling.
/// min_child_weight : float, default 1.0
///     Minimum sum of hessian (instance weight) needed in a child leaf.
/// histogram_threshold : int, default 0
///     Node count threshold: use histogram splitting above this, sort-based below.
///     Default 0 = always use histograms (like XGBoost/LightGBM).
/// n_iter_no_change : int or None, default None
///     Early stopping patience. None = no early stopping.
/// validation_fraction : float, default 0.1
///     Fraction of training data to use as validation for early stopping.
/// reg_alpha : float, default 0.0
///     L1 regularization on leaf weights (XGBoost alpha).
/// max_delta_step : float, default 0.0
///     Maximum absolute leaf weight. 0.0 = disabled.
/// base_score : float or None, default None
///     Initial prediction. None = estimated from data.
/// leaf_smooth : float, default 0.0
///     Bayesian shrinkage on leaf weights: w = G / (H + lambda + leaf_smooth).
///     Shrinks leaves with few samples toward zero. Good values: 1.0-10.0.
/// dart_rate : float, default 0.0
///     DART dropout rate. Fraction of existing trees dropped per round (regression only).
///     0.0 = standard boosting. Good values: 0.05-0.3.
#[pyclass(name = "GradientBoosting")]
pub(crate) struct PyGradientBoosting {
    model: GBTModel,
}

#[pymethods]
impl PyGradientBoosting {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=3,
                        min_samples_split=2, min_samples_leaf=1,
                        subsample=1.0, seed=42,
                        reg_lambda=0.0, gamma=0.0, colsample_bytree=1.0,
                        colsample_bynode=1.0,
                        min_child_weight=1.0, histogram_threshold=0,
                        n_iter_no_change=None, validation_fraction=0.1,
                        reg_alpha=0.0, max_delta_step=0.0,
                        base_score=None, monotone_cst=None, max_bin=254,
                        grow_policy="depthwise", max_leaves=0,
                        goss_top_rate=1.0, goss_other_rate=1.0, goss_min_n=50000,
                        leaf_smooth=0.0, scale_pos_weight=1.0, dart_rate=0.0))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        seed: u64,
        reg_lambda: f64,
        gamma: f64,
        colsample_bytree: f64,
        colsample_bynode: f64,
        min_child_weight: f64,
        histogram_threshold: usize,
        n_iter_no_change: Option<usize>,
        validation_fraction: f64,
        reg_alpha: f64,
        max_delta_step: f64,
        base_score: Option<f64>,
        monotone_cst: Option<Vec<i8>>,
        max_bin: usize,
        grow_policy: &str,
        max_leaves: usize,
        goss_top_rate: f64,
        goss_other_rate: f64,
        goss_min_n: usize,
        leaf_smooth: f64,
        scale_pos_weight: f64,
        dart_rate: f64,
    ) -> PyResult<Self> {
        let gp = match grow_policy {
            "depthwise" => GrowPolicy::Depthwise,
            "lossguide" => GrowPolicy::Lossguide,
            other => return Err(PyValueError::new_err(format!(
                "Unknown grow_policy '{other}'. Use 'depthwise' or 'lossguide'."
            ))),
        };
        let mut model = GBTModel::new(
            n_estimators, learning_rate, max_depth,
            min_samples_split, min_samples_leaf, subsample, seed,
        );
        model.lambda = reg_lambda;
        model.gamma = gamma;
        model.colsample_bytree = colsample_bytree;
        model.colsample_bynode = colsample_bynode;
        model.min_child_weight = min_child_weight;
        model.histogram_threshold = histogram_threshold;
        model.n_iter_no_change = n_iter_no_change;
        model.validation_fraction = validation_fraction;
        model.reg_alpha = reg_alpha;
        model.max_delta_step = max_delta_step;
        model.base_score = base_score;
        model.monotone_cst = monotone_cst;
        model.max_bin = max_bin.max(1).min(254);
        model.grow_policy = gp;
        model.max_leaves = max_leaves;
        model.goss_top_rate = goss_top_rate;
        model.goss_other_rate = goss_other_rate;
        model.goss_min_n = goss_min_n;
        model.leaf_smooth = leaf_smooth;
        model.scale_pos_weight = scale_pos_weight;
        model.dart_rate = dart_rate;
        Ok(PyGradientBoosting { model })
    }

    /// Fit classification. y must be 0-based integer labels (dtype int64).
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit_clf(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model
                .fit_clf(&x_mat, y_slice, sw.as_ref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Fit regression. y must be float64.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit_reg(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model
                .fit_reg(&x_mat, y_slice, sw.as_ref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Classification predict -> 1-D int64 array of class indices.
    fn predict_clf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let result = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_clf(&x_mat)
        });
        Ok(Array1::from(result).into_pyarray_bound(py))
    }

    /// Regression predict -> 1-D float64 array.
    fn predict_reg<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let result = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_reg(&x_mat)
        });
        Ok(Array1::from(result).into_pyarray_bound(py))
    }

    /// Predict probabilities -> 2-D float64 array, shape (n_samples, n_classes).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let (flat, rows, cols) = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            let proba = self.model.predict_proba(&x_mat);
            dmat_to_flat(&proba)
        });
        let arr = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// Serialize fitted model to JSON string (internal).
    #[pyo3(name = "_to_json")]
    fn to_json(&self) -> PyResult<String> {
        self.model
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reconstruct fitted model from JSON string (internal).
    #[staticmethod]
    #[pyo3(name = "_from_json")]
    fn from_json(json: &str) -> PyResult<Self> {
        let model = GBTModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyGradientBoosting { model })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.n_features
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.model.feature_importances.clone()).into_pyarray_bound(py)
    }

    #[getter]
    fn best_n_rounds(&self) -> Option<usize> {
        self.model.best_n_rounds
    }

    #[getter]
    fn grow_policy(&self) -> &str {
        match self.model.grow_policy {
            GrowPolicy::Depthwise => "depthwise",
            GrowPolicy::Lossguide => "lossguide",
        }
    }

    #[getter]
    fn max_leaves(&self) -> usize {
        self.model.max_leaves
    }

    #[getter]
    fn goss_top_rate(&self) -> f64 {
        self.model.goss_top_rate
    }

    #[getter]
    fn goss_other_rate(&self) -> f64 {
        self.model.goss_other_rate
    }

    #[getter]
    fn goss_min_n(&self) -> usize {
        self.model.goss_min_n
    }

    #[getter]
    fn leaf_smooth(&self) -> f64 {
        self.model.leaf_smooth
    }

    #[getter]
    fn dart_rate(&self) -> f64 {
        self.model.dart_rate
    }

    #[getter]
    fn colsample_bynode(&self) -> f64 {
        self.model.colsample_bynode
    }

    #[getter]
    fn scale_pos_weight(&self) -> f64 {
        self.model.scale_pos_weight
    }
}

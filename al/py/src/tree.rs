use crate::{dmat_to_flat, parse_criterion, to_dvec};
use ml::cart::{Criterion, DecisionTreeModel};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// DecisionTree
// ---------------------------------------------------------------------------

/// CART decision tree (classification + regression).
///
/// Parameters
/// ----------
/// max_depth : int, default 500
/// min_samples_split : int, default 2
/// min_samples_leaf : int, default 1
/// histogram_threshold : int, default 4096
/// max_features : int or None, default None
///     Number of features to consider per split. None = all features.
/// seed : int, default 0
///     Random seed for feature subsampling (only used when max_features is set).
#[pyclass(name = "DecisionTree")]
pub(crate) struct PyDecisionTree {
    model: DecisionTreeModel,
    criterion: String,
}

#[pymethods]
impl PyDecisionTree {
    #[new]
    #[pyo3(signature = (max_depth=500, min_samples_split=2, min_samples_leaf=1, histogram_threshold=4096, max_features=None, seed=0, criterion="gini", monotone_cst=None, min_impurity_decrease=0.0, ccp_alpha=0.0))]
    fn new(
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        histogram_threshold: usize,
        max_features: Option<usize>,
        seed: u64,
        criterion: &str,
        monotone_cst: Option<Vec<i64>>,
        min_impurity_decrease: f64,
        ccp_alpha: f64,
    ) -> PyResult<Self> {
        parse_criterion(criterion)?; // validate early
        let mut model = DecisionTreeModel::new(max_depth, min_samples_split, min_samples_leaf);
        model.histogram_threshold = histogram_threshold;
        model.max_features = max_features;
        model.rng_seed = seed;
        model.min_impurity_decrease = min_impurity_decrease;
        model.ccp_alpha = ccp_alpha;
        if let Some(cst) = monotone_cst {
            model.monotone_cst = Some(cst.iter().map(|&v| v.clamp(-1, 1) as i8).collect());
        }
        Ok(PyDecisionTree { model, criterion: criterion.to_string() })
    }

    // ----- fit -----

    /// Fit classification. y must be 0-based integer labels (dtype int64).
    /// Uses zero-copy path: numpy row-major → DMatrix directly.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit_clf(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let criterion = parse_criterion(&self.criterion)?;
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model
                .fit_clf(&x_mat, y_slice, sw.as_ref(), criterion)
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
        // Regression always uses MSE — ignore user criterion (gini/entropy are classification-only)
        let criterion = match self.criterion.as_str() {
            "poisson" => Criterion::Poisson,
            _ => Criterion::MSE,
        };
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            let y_vec = DVector::from_column_slice(y_slice);
            self.model
                .fit_reg(&x_mat, &y_vec, sw.as_ref(), criterion)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    // ----- predict -----

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
        let v: Vec<f64> = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_reg(&x_mat).data.into()
        });
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Classification predict_proba -> 2-D float64 array, shape (n_samples, n_classes).
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

    // ----- serialization (internal — not part of public Python API) -----

    /// Serialize fitted tree to JSON string.
    #[pyo3(name = "_to_json")]
    fn to_json(&self) -> PyResult<String> {
        self.model
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reconstruct fitted tree from JSON string.
    #[staticmethod]
    #[pyo3(name = "_from_json")]
    fn from_json(json: &str) -> PyResult<Self> {
        let model = DecisionTreeModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let criterion = model.criterion;
        let criterion_str = match criterion {
            Criterion::Gini => "gini",
            Criterion::Entropy => "entropy",
            Criterion::MSE => "mse",
            Criterion::Poisson => "poisson",
        };
        Ok(PyDecisionTree { model, criterion: criterion_str.to_string() })
    }

    // ----- properties -----

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
}

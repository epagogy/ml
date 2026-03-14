"""Data normalization and encoding.

Ownership: _normalize.py is the SINGLE source of truth for dtype handling.
- Detects column types
- Encodes string targets → int
- Encodes categorical features (ordinal for trees, one-hot for linear)
- Auto-scales for SVM/KNN/logistic
- Stores all mappings in NormState
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Algorithms that need one-hot encoding for nominal categoricals
LINEAR_ALGORITHMS = frozenset({"logistic", "linear", "svm", "knn", "elastic_net"})
# Algorithms that handle NaN natively — no imputation needed
TREE_ALGORITHMS = frozenset({
    "xgboost", "random_forest", "extra_trees", "decision_tree",
    "histgradient", "adaboost",
    "catboost", "lightgbm", "auto",
})
ONEHOT_MAX_CARDINALITY = 50
# Sentinel for NaN in one-hot encoding — collision-resistant (not a real category name)
_NAN_SENTINEL = "__mlw_nan_d4e5f6a1__"
# Pattern for characters that LightGBM rejects in column names (JSON special chars)
_LGBM_SPECIAL_CHARS = re.compile(r"[\[\]{}<>]")


def _decat(df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
    """Cast CategoricalDtype columns to object for safe fillna/map.

    pandas 3.0 CategoricalDtype rejects fillna() with values not in
    the category list. Converting to object avoids this.

    Args:
        df: Input DataFrame (mutated in-place unless copy=True).
        copy: If True, copy the DataFrame before mutating. If False,
            copies only when categorical columns are found.
    """
    has_cats = any(isinstance(df[col].dtype, pd.CategoricalDtype) for col in df.columns)
    if not has_cats:
        return df.copy() if copy else df
    if copy:
        df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(object)
    return df


@dataclass
class NormState:
    """Stores all encoding/scaling state from prepare().

    Attached to Model._feature_encoder for reuse during prediction.

    Attributes:
        label_encoder: LabelEncoder for string target → int (None if numeric)
        category_maps: dict[col, {value: code}] for ordinal-encoded categoricals
        onehot_encoder: OneHotEncoder for linear models (None if tree-based)
        onehot_columns: columns that are one-hot encoded
        scaler: StandardScaler state for SVM/KNN/logistic (None otherwise)
        feature_names: Ordered feature column names BEFORE encoding (original)
        target_name: Original target column name
    """
    label_encoder: Any  # sklearn LabelEncoder or None
    category_maps: dict[str, dict]
    onehot_encoder: Any  # sklearn OneHotEncoder or None
    onehot_columns: list[str]
    imputer: Any  # sklearn SimpleImputer or None
    scaler: Any  # sklearn StandardScaler or None
    feature_names: list[str]
    target_name: str
    _ohe_rename_map: dict[str, str] | None = None
    _col_rename_map: dict[str, str] | None = None  # original → sanitized for LightGBM
    _X_train_cache: Any = None  # fully transformed training data from prepare(); cleared after first use

    def pop_train_data(self) -> Any:
        """Return and clear cached training data from prepare().

        Returns None if no cached data (e.g. already consumed or prepare
        was called without caching). Caller falls back to transform().
        """
        result = self._X_train_cache
        self._X_train_cache = None
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored encoding + scaling to features.

        Used for BOTH train and predict paths.

        Steps:
        1. Ordinal-encode categoricals using stored category_maps
        2. One-hot encode columns via stored onehot_encoder (linear models)
        3. Map unseen categories to -1 (ordinal) or all-zeros (one-hot)
        4. Apply scaler if present (SVM/KNN/logistic)
        5. NaN handling: PASS THROUGH (tree-based handle natively)

        Args:
            X: DataFrame with feature columns (original column names)

        Returns:
            DataFrame with numeric columns only
        """
        # Apply column name sanitization (LightGBM rejects JSON-special chars)
        if self._col_rename_map:
            X = X.rename(columns=self._col_rename_map)

        # Only copy if we'll mutate (encoding, scaling, or imputation needed)
        needs_mutation = (
            bool(self.category_maps) or self.onehot_encoder is not None
            or self.scaler is not None or self.imputer is not None
        )
        if needs_mutation:
            X = _decat(X, copy=True)

        # Check for Inf early — always invalid (raises DataError)
        import numpy as np
        numeric_cols = X.select_dtypes(include="number")
        if numeric_cols.shape[1] > 0:
            # Convert to float64 to handle nullable integer dtypes (Int64, Int32)
            # which cause TypeError with np.isinf
            try:
                vals = numeric_cols.values.astype(np.float64)
            except (ValueError, TypeError):
                vals = numeric_cols.to_numpy(dtype=np.float64, na_value=np.nan)
            inf_mask = np.isinf(vals)
            if inf_mask.any():
                from ._types import DataError
                n_inf = int(inf_mask.sum())
                # Use same float64 conversion for per-column check
                inf_cols = []
                for c in numeric_cols.columns:
                    try:
                        col_vals = numeric_cols[c].to_numpy(dtype=np.float64, na_value=np.nan)
                    except (ValueError, TypeError):
                        continue
                    if np.isinf(col_vals).any():
                        inf_cols.append(c)
                raise DataError(
                    f"{n_inf} Inf values found in columns {inf_cols}. "
                    "Replace Inf with NaN or a finite value before fitting."
                )

        # Check for NaN early (warn for tree-based, impute later for others)
        has_nan = X.isna().any().any()
        if has_nan and self.imputer is None:
            n_nan_rows = int(X.isna().any(axis=1).sum())
            warnings.warn(
                f"{n_nan_rows} rows contain NaN features. "
                "NaN is passed through for tree-based models.",
                UserWarning,
                stacklevel=2,
            )

        # Ordinal-encode categoricals (category_maps only has ordinal cols)
        unseen_categories = {}
        for col, mapping in self.category_maps.items():
            if col in X.columns:
                # Vectorized .map() is 10-50x faster than .apply() per row
                nan_mask = X[col].isna()
                mapped = X[col].map(mapping)
                # Detect unseen categories: mapped is NaN where original was NOT NaN
                unseen_mask = mapped.isna() & ~nan_mask
                if unseen_mask.any():
                    unseen_vals = X[col][unseen_mask].unique()
                    unseen_categories[col] = set(unseen_vals)
                    mapped = mapped.fillna(-1)
                # Restore original NaN (not unseen, just missing)
                mapped[nan_mask] = np.nan
                X[col] = mapped.astype(float)

        # Warn about unseen categories (ordinal)
        if unseen_categories:
            for col, vals in unseen_categories.items():
                vals_str = ", ".join(map(str, sorted(vals)[:5]))
                if len(vals) > 5:
                    vals_str += f", ... ({len(vals)} total)"
                warnings.warn(
                    f"Column '{col}' has unseen categories: {vals_str}. Mapped to -1.",
                    UserWarning,
                    stacklevel=2
                )

        # One-hot encode columns (linear models)
        if self.onehot_encoder is not None and self.onehot_columns:
            ohe_input = X[self.onehot_columns]
            # Detect unseen categories before encoding (handle_unknown="ignore" silently zeros them)
            for col in self.onehot_columns:
                if col in self.category_maps:
                    known = set(self.category_maps[col].keys())
                else:
                    # Get known categories from the fitted encoder
                    col_idx = self.onehot_columns.index(col)
                    known = set(self.onehot_encoder.categories_[col_idx])
                actual = set(ohe_input[col].dropna().unique()) - {_NAN_SENTINEL}
                unseen = actual - known
                if unseen:
                    vals_str = ", ".join(str(v) for v in sorted(unseen)[:5])
                    if len(unseen) > 5:
                        vals_str += f", ... ({len(unseen)} total)"
                    warnings.warn(
                        f"Column '{col}' has unseen categories: {vals_str}. "
                        "Encoded as all-zeros (one-hot).",
                        UserWarning,
                        stacklevel=2,
                    )
            # Fill NaN with a sentinel for one-hot (will become all-zeros via handle_unknown)
            # _decat: pandas 3.0 CategoricalDtype rejects fillna with unknown value
            ohe_input = _decat(ohe_input, copy=True).fillna(_NAN_SENTINEL)
            ohe_data = self.onehot_encoder.transform(ohe_input)
            if hasattr(ohe_data, "toarray"):
                ohe_data = ohe_data.toarray()
            ohe_col_names = list(self.onehot_encoder.get_feature_names_out(self.onehot_columns))
            if self._ohe_rename_map:
                ohe_col_names = [self._ohe_rename_map.get(n, n) for n in ohe_col_names]
            ohe_df = pd.DataFrame(ohe_data, columns=ohe_col_names, index=X.index)
            X = pd.concat([X.drop(columns=self.onehot_columns), ohe_df], axis=1)

        # Impute NaN AFTER encoding (imputer fit on post-encoding columns)
        if self.imputer is not None and X.isna().any().any():
            # Guard: all-NaN columns at predict time would cause shape mismatch
            all_nan_cols = [c for c in X.columns if X[c].isna().all()]
            if all_nan_cols:
                from ._types import DataError
                raise DataError(
                    f"Columns {all_nan_cols} are entirely NaN and cannot be imputed. "
                    f"Drop them before predicting: data = data.drop(columns={all_nan_cols})"
                )
            n_nan_rows = int(X.isna().any(axis=1).sum())
            X = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index,
            )
            warnings.warn(
                f"{n_nan_rows} rows contained NaN features — "
                "auto-imputed with column medians.",
                UserWarning,
                stacklevel=2,
            )

        # Apply scaler if present (fit on post-encoding columns)
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        # Sanitize column names: XGBoost rejects [, ], <, >
        # Applied unconditionally — these chars carry no meaning for any algorithm
        if any(
            any(c in str(col) for c in "[]<>")
            for col in X.columns
        ):
            new_cols = [
                str(c).replace("[", "_").replace("]", "_")
                       .replace("<", "_").replace(">", "_")
                for c in X.columns
            ]
            X = X.copy()  # don't mutate caller's DataFrame
            X.columns = new_cols

        return X

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels using stored LabelEncoder."""
        if self.label_encoder is not None:
            return self.label_encoder.transform(y)
        return y.to_numpy()

    def decode(self, predictions: np.ndarray) -> pd.Series:
        """Inverse-transform predictions back to original labels."""
        predictions = np.asarray(predictions)
        if self.label_encoder is not None:
            return pd.Series(self.label_encoder.inverse_transform(predictions.astype(int)))
        return pd.Series(predictions)


def prepare(
    X: pd.DataFrame, y: pd.Series, *, algorithm: str = "auto", task: str = "auto",
) -> NormState:
    """Compute encoding + scaling state from training data.

    Called ONCE per training set (or once per CV fold).
    Returns NormState that can be reused for transform/encode/decode.

    Encoding strategy (algorithm-aware):
    - Tree-based (xgboost, random_forest): ordinal encoding for all categoricals
    - Linear models (logistic, linear, svm, knn):
        - cardinality ≤ 50: one-hot encoding
        - cardinality > 50: ordinal encoding + warning (nominal risk)

    Args:
        X: Feature DataFrame
        y: Target Series
        algorithm: Algorithm name (for encoding/scaling decision)
        task: "classification", "regression", or "auto" (for target encoding decision)

    Returns:
        NormState with all encoding/scaling state
    """
    from ._transforms import LabelEncoder, OneHotEncoder, SimpleImputer, StandardScaler

    # Sanitize column names: LightGBM rejects JSON-special chars ([]{}<>)
    # Rename before any encoding so all downstream maps use sanitized names.
    col_rename_map: dict[str, str] | None = None
    if any(_LGBM_SPECIAL_CHARS.search(str(col)) for col in X.columns):
        sanitized = [_LGBM_SPECIAL_CHARS.sub("_", str(col)) for col in X.columns]
        col_rename_map = {
            orig: san for orig, san in zip(X.columns, sanitized) if orig != san
        }
        X = X.rename(columns=col_rename_map)
        warnings.warn(
            f"Column names contain characters unsupported by LightGBM ([{{}}]<>): "
            f"{list(col_rename_map.keys())}. Renamed to: {list(col_rename_map.values())}. "
            "To avoid this warning, rename columns before calling ml.fit().",
            UserWarning,
            stacklevel=3,
        )

    # Detect categorical columns
    categorical_cols = []
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]) or \
           pd.api.types.is_string_dtype(X[col]) or \
           isinstance(X[col].dtype, pd.CategoricalDtype):
            categorical_cols.append(col)

    # Split categoricals by encoding strategy
    onehot_columns = []
    ordinal_columns = []
    onehot_encoder = None
    ohe_rename_map = None

    if algorithm in LINEAR_ALGORITHMS and categorical_cols:
        for col in categorical_cols:
            n_unique = X[col].dropna().nunique()
            if n_unique <= ONEHOT_MAX_CARDINALITY:
                onehot_columns.append(col)
            else:
                ordinal_columns.append(col)
                warnings.warn(
                    f"Column '{col}' has {n_unique} categories (>{ONEHOT_MAX_CARDINALITY}). "
                    f"Using ordinal encoding — one-hot would create {n_unique} columns. "
                    "Warning: ordinal encoding assumes category ordering. "
                    "For nominal categories (no natural order), consider reducing "
                    "cardinality with .map() or dropping the column.",
                    UserWarning,
                    stacklevel=2
                )

        # Fit OneHotEncoder on low-cardinality columns
        if onehot_columns:
            onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                drop=None,
            )
            # Fill NaN with sentinel for fitting (same as transform)
            # _decat: pandas 3.0 CategoricalDtype rejects fillna with unknown value
            # Use a view + fillna (no copy) — the full copy happens later at X_encoded
            ohe_fit_data = _decat(X[onehot_columns]).fillna(_NAN_SENTINEL)
            onehot_encoder.fit(ohe_fit_data)

            # Check for feature name collisions after one-hot expansion
            ohe_names = list(onehot_encoder.get_feature_names_out(onehot_columns))
            other_cols = [c for c in X.columns if c not in onehot_columns]
            collisions = set(ohe_names) & set(other_cols)
            ohe_rename_map = None
            if collisions:
                warnings.warn(
                    f"One-hot encoding creates columns that collide with existing "
                    f"feature names: {sorted(collisions)[:5]}. This may cause "
                    "unexpected behavior. Consider renaming columns.",
                    UserWarning,
                    stacklevel=2
                )
                # Rename colliding OHE columns to avoid duplicate names
                # Use incrementing suffix to avoid cascading collisions
                all_existing = set(other_cols) | set(ohe_names)
                ohe_rename_map = {}
                for name in ohe_names:
                    if name in collisions:
                        candidate = f"{name}_ohe"
                        counter = 2
                        while candidate in all_existing or candidate in ohe_rename_map.values():
                            candidate = f"{name}_ohe{counter}"
                            counter += 1
                        ohe_rename_map[name] = candidate
    else:
        # Tree-based or no categoricals: all ordinal
        ordinal_columns = list(categorical_cols)

    # Warn about very high cardinality string columns (likely IDs or free text)
    for col in categorical_cols:
        n_unique = X[col].dropna().nunique()
        if n_unique > 100 and col not in onehot_columns:
            warnings.warn(
                f"Column '{col}' has {n_unique} unique string values — "
                "this may be an ID or free-text column, not a useful feature. "
                "Consider dropping it or encoding text features before fitting.",
                UserWarning,
                stacklevel=2,
            )

    # Build ordinal encoding maps (only for ordinal columns)
    category_maps = {}
    for col in ordinal_columns:
        unique_vals = X[col].dropna().unique()
        # Sort with str() fallback for mixed-type columns (e.g., 1.0, "two", 3.0)
        try:
            sorted_vals = sorted(unique_vals)
        except TypeError:
            sorted_vals = sorted(unique_vals, key=str)
        category_maps[col] = {val: idx for idx, val in enumerate(sorted_vals)}

    # Encode ALL categoricals in X for scaling computation
    X_encoded = _decat(X, copy=True)

    # Check for Inf — always invalid. Must happen before encoding changes dtypes.
    numeric_cols = X_encoded.select_dtypes(include="number")
    if numeric_cols.shape[1] > 0:
        try:
            vals = numeric_cols.values.astype(np.float64)
        except (ValueError, TypeError):
            vals = numeric_cols.to_numpy(dtype=np.float64, na_value=np.nan)
        inf_mask = np.isinf(vals)
        if inf_mask.any():
            from ._types import DataError
            n_inf = int(inf_mask.sum())
            inf_cols = [
                c for c in numeric_cols.columns
                if np.isinf(numeric_cols[c].to_numpy(dtype=np.float64, na_value=np.nan)).any()
            ]
            raise DataError(
                f"{n_inf} Inf values found in columns {inf_cols}. "
                "Replace Inf with NaN or a finite value before fitting."
            )

    for col, mapping in category_maps.items():
        X_encoded[col] = X_encoded[col].map(mapping).astype(float)

    # One-hot expand for scaling if needed
    if onehot_encoder is not None and onehot_columns:
        # X_encoded is already _decat'd — no second _decat needed
        ohe_input = X_encoded[onehot_columns].fillna(_NAN_SENTINEL)
        ohe_data = onehot_encoder.transform(ohe_input)
        if hasattr(ohe_data, "toarray"):
            ohe_data = ohe_data.toarray()
        ohe_col_names = list(onehot_encoder.get_feature_names_out(onehot_columns))
        if ohe_rename_map:
            ohe_col_names = [ohe_rename_map.get(n, n) for n in ohe_col_names]
        ohe_df = pd.DataFrame(ohe_data, columns=ohe_col_names, index=X_encoded.index)
        X_encoded = pd.concat([X_encoded.drop(columns=onehot_columns), ohe_df], axis=1)

    # Fit LabelEncoder for string targets OR non-0-based integer targets
    label_encoder = None
    if pd.api.types.is_object_dtype(y) or \
       pd.api.types.is_string_dtype(y) or \
       isinstance(y.dtype, pd.CategoricalDtype) or \
       pd.api.types.is_bool_dtype(y):
        label_encoder = LabelEncoder()
        label_encoder.fit(y.dropna())
    elif pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
        # Remap non-sequential integer labels (e.g., [4,8,12,18] → [0,1,2,3])
        # Required by XGBoost which expects labels in range [0, num_class)
        # ONLY for classification — regression targets must stay as-is
        if task != "regression":
            unique_vals = sorted(y.dropna().unique())
            n_unique = len(unique_vals)
            if n_unique <= 30 and unique_vals != list(range(n_unique)):
                label_encoder = LabelEncoder()
                label_encoder.fit(y.dropna())

    # Fit SimpleImputer for algorithms that can't handle NaN
    imputer = None
    if algorithm not in TREE_ALGORITHMS:
        # Detect all-NaN columns — imputer silently drops them, causing shape mismatch
        # Use iloc to handle duplicate column names safely
        all_nan_cols = [
            X_encoded.columns[i]
            for i in range(len(X_encoded.columns))
            if X_encoded.iloc[:, i].isna().all()
        ]
        if all_nan_cols:
            from ._types import DataError
            raise DataError(
                f"Columns {all_nan_cols} are entirely NaN and cannot be imputed. "
                f"Drop them before fitting: data = data.drop(columns={all_nan_cols})"
            )
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_encoded)
        # Impute X_encoded so scaler sees clean data — assign in-place (avoid copy)
        if X_encoded.isna().any().any():
            X_encoded.iloc[:, :] = imputer.transform(X_encoded)

    # Fit StandardScaler for scale-sensitive algorithms
    scaler = None
    # NB excluded: GaussianNB uses raw feature distributions — scaling breaks it
    if algorithm in ("svm", "knn", "logistic", "elastic_net"):
        scaler = StandardScaler()
        scaler.fit(X_encoded)
        # Apply scaler to X_encoded so cached data is fully transformed.
        # Must construct new DataFrame — scaler changes int64→float64 dtypes
        # and pandas 3.0 iloc rejects lossy dtype assignment.
        X_encoded = pd.DataFrame(
            scaler.transform(X_encoded),
            columns=X_encoded.columns,
            index=X_encoded.index,
        )
        warnings.warn(
            f"Auto-scaling features for {algorithm} (StandardScaler). "
            "Inspect with model.preprocessing_['scaled'].",
            UserWarning,
            stacklevel=3,
        )

    # Sanitize column names: XGBoost rejects [, ], <, >
    if any(any(c in str(col) for c in "[]<>") for col in X_encoded.columns):
        new_cols = [
            str(c).replace("[", "_").replace("]", "_")
                   .replace("<", "_").replace(">", "_")
            for c in X_encoded.columns
        ]
        X_encoded.columns = new_cols

    ns = NormState(
        label_encoder=label_encoder,
        category_maps=category_maps,
        onehot_encoder=onehot_encoder,
        onehot_columns=onehot_columns,
        imputer=imputer,
        scaler=scaler,
        feature_names=list(X.columns),
        target_name=y.name or "target",
        _ohe_rename_map=ohe_rename_map if onehot_encoder is not None else None,
        _col_rename_map=col_rename_map,
    )
    # Cache fully transformed training data — avoids redundant transform() call.
    # Cleared after first pop_train_data() to avoid keeping training data in memory.
    ns._X_train_cache = X_encoded
    return ns

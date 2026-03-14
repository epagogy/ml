"""interact() — feature interaction detection.

Identifies which pairs of features have significant interactions in a model's
predictions. Uses a permutation-based H-statistic approximation: if shuffling
two features together has a larger effect than shuffling each individually,
those features interact.

Usage:
    >>> result = ml.interact(model, data=s.valid, seed=42)
    >>> result.top(5)        # DataFrame: feature_a, feature_b, score
    >>> result.pairs          # full DataFrame sorted by score
    >>> result.summary        # str description
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class InteractResult:
    """Result of feature interaction analysis.

    Attributes
    ----------
    pairs : pd.DataFrame
        All feature pairs with interaction scores, sorted descending.
        Columns: feature_a, feature_b, score.
    n_features : int
        Number of features evaluated.
    n_pairs : int
        Number of pairs evaluated.
    summary : str
        Human-readable description of the strongest interactions.
    """

    pairs: pd.DataFrame
    n_features: int
    n_pairs: int
    summary: str

    def top(self, n: int = 5) -> pd.DataFrame:
        """Return the top N pairs with the strongest interactions."""
        return self.pairs.head(n).reset_index(drop=True)

    def __repr__(self) -> str:
        if len(self.pairs) > 0 and self.pairs["score"].iloc[0] > 0:
            top = self.pairs.iloc[0]
            top_str = f"top=({top['feature_a']}, {top['feature_b']}, {top['score']:.3f})"
        else:
            top_str = "no interactions detected"
        return f"InteractResult(n_pairs={self.n_pairs}, {top_str})"


def interact(
    model,
    *,
    data: pd.DataFrame,
    n_top: int = 10,
    seed: int,
) -> InteractResult:
    """Detect feature interactions in a model.

    Identifies pairs of features where the combined effect exceeds the sum
    of individual effects (Friedman H-statistic approximation).

    For each pair (A, B) from the top N important features:
    - Shuffle A: measure prediction change (effect_A)
    - Shuffle B: measure prediction change (effect_B)
    - Shuffle both: measure prediction change (effect_AB)
    - interaction_score = std(effect_AB - effect_A - effect_B)

    High score → the features interact (model learned a joint relationship).
    Low score → the features contribute independently.

    Parameters
    ----------
    model : Model or TuningResult
        Trained model. Classification and regression both supported.
    data : pd.DataFrame
        Reference data for interaction evaluation. Should include feature
        columns (target column is automatically excluded if present).
        Recommended: held-out validation set (e.g., s.valid).
    n_top : int, default=10
        Number of top features to include in pairwise evaluation.
        n_top=10 evaluates 45 pairs. n_top=20 evaluates 190 pairs.
    seed : int
        Random seed for reproducible shuffling. Required — no default.

    Returns
    -------
    InteractResult
        - ``.pairs``: DataFrame of (feature_a, feature_b, score) sorted by score
        - ``.top(5)``: top N interacting pairs
        - ``.summary``: human-readable interpretation

    Raises
    ------
    DataError
        If data has fewer than 2 features or fewer than 20 rows.
    ModelError
        If model has no predict functionality.
    ConfigError
        If n_top < 2.

    Examples
    --------
    >>> result = ml.interact(model, data=s.valid, seed=42)
    >>> result.top(3)
       feature_a feature_b   score
    0  tenure    charges    0.142
    1  age       tenure     0.089
    2  contract  charges    0.071
    """
    from ._types import ConfigError, DataError, TuningResult
    from .predict import predict

    if isinstance(model, TuningResult):
        model = model.best_model

    if n_top < 2:
        raise ConfigError(
            f"n_top must be >= 2 to evaluate at least one pair, got {n_top}."
        )

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"data= must be a DataFrame, got {type(data).__name__}."
        )

    # Remove target column if present
    X = data.drop(columns=[model._target], errors="ignore")
    X = X[model._features] if model._features else X

    if len(X) < 20:
        raise DataError(
            f"interact() requires at least 20 rows, got {len(X)}. "
            "Use a larger dataset for reliable interaction estimates."
        )

    available_features = [f for f in model._features if f in X.columns]
    if len(available_features) < 2:
        raise DataError(
            f"interact() needs at least 2 features, found {len(available_features)}."
        )

    rng = np.random.RandomState(seed)

    # Get feature importance to select top features
    from .explain import explain
    imp_result = explain(model)
    if isinstance(imp_result, pd.DataFrame) and "feature" in imp_result.columns:
        imp_df = imp_result
    elif isinstance(imp_result, pd.DataFrame):
        imp_df = imp_result.reset_index()
        imp_df.columns = ["feature", "importance"]
    else:
        # Fallback: equal importance for all features
        imp_df = pd.DataFrame({
            "feature": available_features,
            "importance": [1.0] * len(available_features),
        })

    # Select top_n features by importance
    top_features = imp_df["feature"].tolist()
    top_features = [f for f in top_features if f in available_features][:n_top]
    if len(top_features) < 2:
        top_features = available_features[:n_top]

    # Use probabilities for classification (numeric, comparable); scores for regression
    use_proba = model._task == "classification" and hasattr(model._model, "predict_proba")

    def _to_numeric(pred) -> np.ndarray:
        """Convert prediction output to 1D numeric array for interaction scoring."""
        if isinstance(pred, pd.DataFrame):
            # proba DataFrame — use max probability column (1D summary)
            return pred.values.max(axis=1).astype(float)
        arr = pred.values if hasattr(pred, "values") else np.asarray(pred)
        if arr.dtype.kind in ("U", "O", "S"):
            # String labels — can't compare numerically; use rank encoding
            import warnings as _w2
            _w2.warn(
                "interact() received string predictions — using rank encoding "
                "for interaction scoring. Results are approximate.",
                UserWarning, stacklevel=4,
            )
            _classes = sorted(set(arr))
            _lmap = {c: i for i, c in enumerate(_classes)}
            return np.array([_lmap.get(v, -1) for v in arr], dtype=float)
        return arr.astype(float)

    baseline = predict(model, X, proba=use_proba)
    baseline_arr = _to_numeric(baseline)

    # Compute pairwise interaction scores
    records = []
    for i, feat_a in enumerate(top_features):
        for feat_b in top_features[i + 1:]:
            if feat_a not in X.columns or feat_b not in X.columns:
                continue

            # Shuffle feature_a
            X_a = X.copy()
            X_a[feat_a] = rng.permutation(X[feat_a].values)
            pred_a_arr = _to_numeric(predict(model, X_a, proba=use_proba))

            # Shuffle feature_b
            X_b = X.copy()
            X_b[feat_b] = rng.permutation(X[feat_b].values)
            pred_b_arr = _to_numeric(predict(model, X_b, proba=use_proba))

            # Shuffle both
            X_ab = X.copy()
            X_ab[feat_a] = rng.permutation(X[feat_a].values)
            X_ab[feat_b] = rng.permutation(X[feat_b].values)
            pred_ab_arr = _to_numeric(predict(model, X_ab, proba=use_proba))

            # Interaction = residual effect from joint perturbation
            effect_a = pred_a_arr - baseline_arr
            effect_b = pred_b_arr - baseline_arr
            effect_ab = pred_ab_arr - baseline_arr
            interaction = effect_ab - effect_a - effect_b
            score = float(np.std(interaction))

            records.append({
                "feature_a": feat_a,
                "feature_b": feat_b,
                "score": round(score, 6),
            })

    if not records:
        pairs_df = pd.DataFrame(columns=["feature_a", "feature_b", "score"])
        summary = "No feature pairs to evaluate."
    else:
        pairs_df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

        top3 = pairs_df.head(3)
        if top3["score"].iloc[0] < 0.01:
            summary = (
                "No strong feature interactions detected. Features contribute "
                "approximately independently to predictions."
            )
        else:
            top_pair = top3.iloc[0]
            summary = (
                f"Strongest interaction: ({top_pair['feature_a']}, {top_pair['feature_b']}) "
                f"score={top_pair['score']:.3f}. "
            )
            if len(top3) > 1 and top3["score"].iloc[1] > 0.01:
                summary += (
                    f"Also: ({top3.iloc[1]['feature_a']}, {top3.iloc[1]['feature_b']}) "
                    f"score={top3.iloc[1]['score']:.3f}."
                )

    return InteractResult(
        pairs=pairs_df,
        n_features=len(top_features),
        n_pairs=len(records),
        summary=summary,
    )

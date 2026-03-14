"""update() — incremental model update with new data.

Adds new observations to a trained model without full retraining.
Supports warm-start algorithms (gradient boosting, online learners)
and streaming/batch update workflows.

Usage:
    >>> updated = ml.update(model, new_data, target="churn")
    >>> metrics = ml.evaluate(updated, s.valid)

Status: stub — not yet implemented. Planned for v2.
"""

from __future__ import annotations


def update(model, data, target, *, seed=None):
    """Update a trained model with new data (incremental learning).

    Performs warm-start retraining for algorithms that support it.
    For algorithms without incremental support, raises NotImplementedError
    with guidance to use ml.fit() instead.

    Args:
        model: A trained ml.Model
        data: New observations (DataFrame with target column)
        target: Target column name
        seed: Random seed (keyword-only, optional)

    Returns:
        Model — updated model with new data incorporated

    Raises:
        NotImplementedError: Always (v2 feature)

    Example:
        >>> updated = ml.update(model, new_batch, target="churn")
    """
    raise NotImplementedError(
        "ml.update() is planned for v2. "
        "For now, retrain with ml.fit() on the combined dataset."
    )

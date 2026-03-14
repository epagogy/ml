"""pipe() — composable preprocessing pipeline.

Chains fitted transformers (Scaler, Encoder, Imputer, Tokenizer) into
a single serializable object. No new concepts — just composition.

Usage:
    >>> imp = ml.impute(s.train, columns=["age", "income"])
    >>> enc = ml.encode(s.train, columns=["city"])
    >>> scl = ml.scale(s.train, columns=["age", "income"])
    >>> p = ml.pipe([imp, enc, scl])
    >>> train_clean = p.process(s.train)
    >>> valid_clean = p.process(s.valid)
    >>> model = ml.fit(train_clean, "target", seed=42)
    >>> ml.save(p, "pipeline.pyml")
    >>> p2 = ml.load("pipeline.pyml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Pipeline:
    """Fitted preprocessing pipeline.

    An ordered sequence of transformers. Each step's ``.transform()`` is
    called in sequence. Preserves leakage-free preprocessing: steps must
    be fitted on training data before being added to the pipeline.

    Attributes
    ----------
    steps : list
        Ordered transformers (Scaler, Encoder, Imputer, Tokenizer, or
        any object with a ``.transform(df)`` method).

    Examples
    --------
    >>> imp = ml.impute(s.train, columns=["age"])
    >>> enc = ml.encode(s.train, columns=["city"])
    >>> scl = ml.scale(s.train, columns=["age"])
    >>> p = ml.pipe([imp, enc, scl])
    >>> p.process(s.valid)
    """

    steps: list[Any] = field(default_factory=list)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps in sequence.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame (train, valid, or new data).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame after all steps are applied.

        Raises
        ------
        DataError
            If data is not a DataFrame.
        """

        from ._types import DataError
        if not isinstance(data, pd.DataFrame):
            raise DataError(
                f"process() expects DataFrame, got {type(data).__name__}."
            )

        out = data
        for step in self.steps:
            out = step.transform(out)
        return out

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for process() — consistent with other transformer interface."""
        return self.process(data)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for process() — allows using Pipeline as a callable."""
        return self.process(data)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self.steps]
        return f"Pipeline(steps=[{', '.join(step_names)}])"


def pipe(steps: list[Any]) -> Pipeline:
    """Create a preprocessing pipeline from fitted transformers.

    Composes fitted transformers into a single serializable object that
    can be saved, loaded, and applied to any DataFrame. Steps are applied
    in the order they are provided.

    No leakage: each transformer must be fitted on training data before
    being passed to ``pipe()``.

    Parameters
    ----------
    steps : list
        Ordered list of fitted transformers. Applied left-to-right.
        Each step must implement ``.transform(pd.DataFrame) → pd.DataFrame``.
        Accepts: Scaler, Encoder, Imputer, Tokenizer, or any compatible object.

    Returns
    -------
    Pipeline
        Pipeline with ``.process(df)`` method and save/load support.

    Raises
    ------
    DataError
        If steps is empty or not a list.
    ConfigError
        If any step lacks a ``.transform()`` method.

    Examples
    --------
    >>> imp = ml.impute(s.train, columns=["age", "income"])
    >>> enc = ml.encode(s.train, columns=["city"])
    >>> scl = ml.scale(s.train, columns=["age", "income"])
    >>> p = ml.pipe([imp, enc, scl])
    >>> train_clean = p.process(s.train)
    >>> valid_clean = p.process(s.valid)
    >>> model = ml.fit(train_clean, "target", seed=42)
    >>> preds = ml.predict(model, valid_clean)

    Single-step pipeline (useful for save/load of a standalone transformer):
    >>> p = ml.pipe([ml.impute(s.train, columns=["age"])])

    Save and reload entire pipeline:
    >>> ml.save(p, "pipeline.pyml")
    >>> p2 = ml.load("pipeline.pyml")
    >>> valid_clean = p2.process(s.valid)
    """
    from ._types import ConfigError, DataError

    if not isinstance(steps, list):
        raise DataError(
            f"pipe() expects a list of steps, got {type(steps).__name__}. "
            "Example: ml.pipe([imputer, encoder, scaler])"
        )

    if not steps:
        raise DataError(
            "pipe() requires at least one step. "
            "Example: ml.pipe([ml.impute(train, columns=['age'])])"
        )

    invalid = [s for s in steps if not hasattr(s, "transform")]
    if invalid:
        names = [type(s).__name__ for s in invalid]
        raise ConfigError(
            f"Steps {names} do not have a .transform() method. "
            "Each step must implement transform(pd.DataFrame) → pd.DataFrame. "
            "Use ml.impute(), ml.scale(), ml.encode(), or ml.tokenize() to create steps."
        )

    return Pipeline(steps=list(steps))

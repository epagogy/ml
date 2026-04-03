"""Core types and dataclasses for ml library."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

if TYPE_CHECKING:
    pass


# ===== ERRORS =====


class MLError(Exception):
    """Base exception for all ml library errors."""
    pass


class ConfigError(MLError):
    """Configuration or parameter error."""
    pass


class DataError(MLError):
    """Data validation error."""
    pass


class PartitionError(DataError):
    """Partition guard violation (e.g., test data passed to fit())."""
    pass


class ModelError(MLError):
    """Model-related error."""
    pass


class VersionError(MLError):
    """Version compatibility error."""
    pass


class WorkflowStateError(ModelError):
    """Invalid workflow state transition (e.g., assess before fit)."""
    pass


# ===== WORKFLOW DFA =====
# The ML workflow language is a regular language recognizable by this DFA.
# States form a partial order: CREATED < FITTED < EVALUATED < ASSESSED.
# ASSESSED is a sink (terminal — no outbound transitions).


class WorkflowState(int):
    """Workflow state as int for dataclass compatibility.

    States:
        0 = CREATED  — Model object exists but not yet fitted
        1 = FITTED   — fit() returned successfully
        2 = EVALUATED — evaluate() called at least once
        3 = ASSESSED — assess() called (terminal, no further transitions)
    """
    CREATED = 0
    FITTED = 1
    EVALUATED = 2
    ASSESSED = 3

    _NAMES = {0: "CREATED", 1: "FITTED", 2: "EVALUATED", 3: "ASSESSED"}

    def __repr__(self) -> str:
        return self._NAMES.get(int(self), f"UNKNOWN({int(self)})")


# Valid transitions: (from_state, verb) → to_state
# Verbs not listed for a state are forbidden from that state.
_WORKFLOW_TRANSITIONS: dict[tuple[int, str], int] = {
    # From CREATED (only fit() constructor produces FITTED)
    # CREATED is never exposed to users — fit() returns FITTED directly.

    # From FITTED
    (WorkflowState.FITTED, "evaluate"): WorkflowState.EVALUATED,
    (WorkflowState.FITTED, "explain"): WorkflowState.FITTED,
    (WorkflowState.FITTED, "predict"): WorkflowState.FITTED,
    (WorkflowState.FITTED, "assess"): WorkflowState.ASSESSED,
    (WorkflowState.FITTED, "calibrate"): WorkflowState.FITTED,
    (WorkflowState.FITTED, "validate"): WorkflowState.FITTED,

    # From EVALUATED
    (WorkflowState.EVALUATED, "evaluate"): WorkflowState.EVALUATED,
    (WorkflowState.EVALUATED, "explain"): WorkflowState.EVALUATED,
    (WorkflowState.EVALUATED, "predict"): WorkflowState.EVALUATED,
    (WorkflowState.EVALUATED, "assess"): WorkflowState.ASSESSED,
    (WorkflowState.EVALUATED, "calibrate"): WorkflowState.EVALUATED,
    (WorkflowState.EVALUATED, "validate"): WorkflowState.EVALUATED,

    # From ASSESSED — terminal, no transitions out
}


def check_workflow_transition(state: int, verb: str) -> int:
    """Check if verb is valid from current state, return new state.

    Returns new state on success. Raises WorkflowStateError on invalid transition.
    """
    key = (state, verb)
    new_state = _WORKFLOW_TRANSITIONS.get(key)
    if new_state is None:
        if state == WorkflowState.ASSESSED:
            raise WorkflowStateError(
                f"{verb}() called on assessed model. "
                f"Assessment is terminal — the model's evidence is sealed. "
                f"To continue iterating, fit a new model."
            )
        state_name = WorkflowState._NAMES.get(state, f"state={state}")
        raise WorkflowStateError(
            f"{verb}() is not valid from workflow state {state_name}."
        )
    return new_state


# ===== DISPLAY TYPES =====


class Metrics(dict):
    """Dict subclass with clean terminal display for ML metrics."""

    def __init__(self, data: dict | None = None, *, task: str = "", time: float | None = None, source: str = "Metrics", **kwargs):
        super().__init__(data or {}, **kwargs)
        self._task = task
        self._time = time
        self._source = source

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if not self:
            return "{}"
        # Header
        label = self._source
        header = f"── {label} [{self._task}] " if self._task else f"── {label} "
        header += "─" * max(1, 40 - len(header))
        lines = [header]

        items = list(self.items())
        if len(items) > 6:
            # Two-column layout
            mid = (len(items) + 1) // 2
            left_items = items[:mid]
            right_items = items[mid:]
            left_max_k = max(len(k) for k, _ in left_items)
            right_max_k = max(len(k) for k, _ in right_items) if right_items else 0
            for i, (k, v) in enumerate(left_items):
                left = f"{k:<{left_max_k}}  {v:.4f}"
                if i < len(right_items):
                    rk, rv = right_items[i]
                    right = f"{rk:<{right_max_k}}  {rv:.4f}"
                    lines.append(f"{left}   {right}")
                else:
                    lines.append(left)
        else:
            # Single column — colon separator like Model output
            max_label = max(len(k) for k in self) + 1  # +1 for colon
            for k, v in items:
                label = f"{k}:"
                lines.append(f"{label:<{max_label}}  {v:.4f}")

        if self._time is not None:
            lines.append(f"({self._time:.2f}s)")

        return "\n".join(lines) + "\n"

    def __format__(self, format_spec: str) -> str:
        """Compact one-line repr for use inside f-strings.

        ``print(f"Final: {verdict}")`` → ``Final: {rmse=64.63, r2=0.28}``
        ``print(verdict)`` → full block display (via __str__)
        """
        if format_spec:
            return super().__format__(format_spec)
        parts = ", ".join(f"{k}={v:.4f}" for k, v in self.items())
        return "{" + parts + "}"

    def __getattr__(self, name: str):
        """Allow attribute-style access: met.accuracy instead of met['accuracy']."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"No metric '{name}'. Available: {list(self.keys())}"
            ) from None

    def _repr_html_(self) -> str:
        return f'<pre style="font-family: monospace;">{self.__str__()}</pre>'


class Evidence(dict):
    """Terminal assessment result. Sealed — not substitutable for Metrics.

    Returned exclusively by assess(). No primitive accepts Evidence as input;
    it flows to the human, not to the next step. isinstance(result, Metrics)
    returns False by design (Codd condition 7).
    """

    def __init__(self, data: dict | None = None, *, task: str = "", time: float | None = None, K: int = 0, **kwargs):
        super().__init__(data or {}, **kwargs)
        self._task = task
        self._time = time
        self._K = K  # number of evaluate() calls before assessment (selection pressure)
        self._source = "Evidence"

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if not self:
            return "{}"
        header = f"── Evidence [{self._task}] " if self._task else "── Evidence "
        header += "─" * max(1, 40 - len(header))
        lines = [header]
        max_label = max(len(k) for k in self) + 1
        for k, v in self.items():
            label = f"{k}:"
            lines.append(f"{label:<{max_label}}  {v:.4f}")
        footer_parts = []
        if self._K > 0:
            footer_parts.append(f"K={self._K}")
        if self._time is not None:
            footer_parts.append(f"{self._time:.2f}s")
        if footer_parts:
            lines.append(f"({', '.join(footer_parts)})")
        return "\n".join(lines) + "\n"

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return super().__format__(format_spec)
        parts = ", ".join(f"{k}={v:.4f}" for k, v in self.items())
        return "{" + parts + "}"

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No metric '{name}'. Available: {list(self.keys())}") from None

    def _repr_html_(self) -> str:
        return f'<pre style="font-family: monospace;">{self.__str__()}</pre>'


class ProfileResult(dict):
    """Dict subclass with clean terminal display for data profiling."""

    def __getattr__(self, name: str):
        """Allow attribute-style access: p.warnings instead of p['warnings']."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"ProfileResult has no attribute '{name}'. "
                f"Available keys: {list(self.keys())}"
            ) from None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        task = self.get("task", "?")
        header = f"── Profile [{task}] "
        header += "─" * max(1, 40 - len(header))
        lines = [header]

        shape = self.get("shape", (0, 0))
        lines.append(f"Rows:     {shape[0]:,}")
        lines.append(f"Columns:  {shape[1]}")

        if self.get("target"):
            lines.append(f"Target:   {self['target']}")

        if self.get("target_balance") is not None:
            lines.append(f"Balance:  {self['target_balance']:.0%} minority class")

        warnings_list = self.get("warnings", [])
        if warnings_list:
            lines.append("")
            for w in warnings_list:
                lines.append(f"  ! {w}")

        return "\n".join(lines) + "\n"


class Explanation:
    """Feature importance with visual bar chart display.

    Wraps a DataFrame — supports [] indexing, .columns, .shape, .head(), etc.
    """

    def __init__(self, df: pd.DataFrame, algorithm: str = "", method: str = "",
                 shap_values: pd.DataFrame | None = None):
        self._df = df
        self._algorithm = algorithm
        self._method = method
        self.shap_values = shap_values  # A7: per-sample SHAP values (n_samples × n_features)

    @property
    def method(self) -> str:
        """Method used to compute importances: 'tree_importance', 'abs_coefficients',
        'shap', or 'permutation'."""
        return self._method

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        method_str = f" ({self._method})" if self._method else ""
        header = f"── Explain [{self._algorithm}{method_str}] "
        header += "─" * max(1, 40 - len(header))
        lines = [header]

        if len(self._df) == 0:
            lines.append("(no features)")
            return "\n".join(lines) + "\n"

        max_name = max(len(str(f)) for f in self._df["feature"])
        max_imp = self._df["importance"].max()
        bar_width = 20

        has_direction = "direction" in self._df.columns
        for _, row in self._df.iterrows():
            name = str(row["feature"])
            imp = row["importance"]
            pct = imp * 100

            if max_imp > 0:
                filled = imp / max_imp * bar_width
            else:
                filled = 0
            full_blocks = int(filled)
            remainder = filled - full_blocks
            bar = "█" * full_blocks
            if remainder >= 0.5:
                bar += "▌"

            dir_str = f"  {'(+)' if row['direction'] >= 0 else '(-)'}" if has_direction else ""
            lines.append(f"  {name:<{max_name}}  {bar:<{bar_width}}  {pct:5.1f}%{dir_str}")

        return "\n".join(lines) + "\n"

    def _repr_html_(self) -> str:
        return f'<pre style="font-family: monospace;">{self.__str__()}</pre>'

    def items(self):
        """Yield (feature_name, importance_score) pairs, descending by importance."""
        for feat, score in zip(self._df["feature"], self._df["importance"]):
            yield str(feat), float(score)

    def keys(self):
        """Yield feature names, descending by importance."""
        yield from (str(f) for f in self._df["feature"])

    def values(self):
        """Yield importance scores, descending by importance."""
        for score in self._df["importance"]:
            yield float(score)

    def __iter__(self):
        """Iterate over feature names (matches dict/Series convention)."""
        yield from self._df["feature"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._df, name)

    def __getitem__(self, key: Any) -> Any:
        return self._df[key]

    def __len__(self) -> int:
        return len(self._df)


class Leaderboard:
    """Formatted algorithm comparison table.

    Wraps a DataFrame — supports [] indexing, .columns, .shape, .head(), etc.
    """

    def __init__(self, df: pd.DataFrame, title: str = "", models: list | None = None):
        self._df = df
        self._title = title
        self._models = models

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        header = f"── {self._title} "
        header += "─" * max(1, 40 - len(header))
        return header + "\n" + self._df.to_string(index=False) + "\n"

    def _repr_html_(self) -> str:
        return f'<pre style="font-family: monospace;">{self.__str__()}</pre>'

    @property
    def best(self) -> str:
        """Return the algorithm name of the top-ranked model."""
        if len(self._df) == 0:
            return ""
        return str(self._df["algorithm"].iloc[0])

    @property
    def best_model(self):
        """Return the top-ranked fitted Model object.

        Available for both ml.screen() and ml.compare() leaderboards.
        Returns the fitted Model for the top-ranked algorithm.

        Use ``ml.predict(lb.best_model, data)`` to predict directly.

        Returns
        -------
        Model or None
            None if no models were stored (e.g., all algorithms failed).
        """
        if self._models is None or len(self._models) == 0:
            return None
        for m in self._models:
            if m is not None:
                return m
        return None

    @property
    def ranking(self) -> pd.DataFrame:
        """Return the algorithm ranking DataFrame.

        Columns: algorithm, {metric}, cv_std, cv_min, cv_max, time_seconds, rank
        Sorted by primary metric (descending for higher-is-better metrics).

        Returns a copy — safe to modify without affecting the Leaderboard.
        """
        return self._df.copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._df, name)

    def __getitem__(self, key: Any) -> Any:
        return self._df[key]

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        return iter(self._df)


class CheckResult(NamedTuple):
    """Single leakage check result."""
    name: str        # display name, e.g. "Feature-target correlation"
    passed: bool     # True if no issues
    detail: str      # summary, e.g. "max |r|=0.42 (income)"
    severity: str    # "ok", "warn", "critical"


class OptimizeResult:
    """Result from ml.optimize() — threshold-tuned model with float comparison support.

    Supports both usage patterns:

    **Pattern A — use the calibrated model (recommended):**
        >>> optimized = ml.optimize(model, data=s.valid, metric='f1')
        >>> preds = ml.predict(optimized, s.test)  # threshold baked in

    **Pattern B — extract threshold for manual use:**
        >>> optimized = ml.optimize(model, data=s.valid, metric='f1')
        >>> proba > optimized          # comparison works via float()
        >>> proba > optimized.threshold  # explicit float access
        >>> float(optimized)             # convert to bare float
    """

    def __init__(self, threshold: float, model: Model) -> None:
        self.threshold = threshold
        self.model = model

    # ---------- numeric protocol ----------
    def __float__(self) -> float:
        return self.threshold

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.threshold < other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.threshold > other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.threshold <= other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.threshold >= other
        return NotImplemented

    # ---------- Model proxy — delegate predict/fit attributes to .model ----------
    def __getattr__(self, name: str):
        # Delegate Model attributes to self.model so OptimizeResult can be used
        # wherever Model is expected (e.g. ml.predict(optimized, data))
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Available: threshold, model, + all Model attributes."
            ) from None

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(threshold={self.threshold:.4f}, "
            f"algorithm='{self.model.algorithm}')"
        )


class SuspectFeature(NamedTuple):
    """Feature flagged as potential leakage source."""
    feature: str     # column name
    check: str       # which check flagged it
    value: float     # the metric value
    detail: str      # e.g. "|r|=0.98"
    action: str      # suggested action


@dataclass
class LeakReport:
    """Leakage detection report from leak().

    Attributes:
        clean: True if no warnings detected
        n_warnings: Number of checks that flagged issues
        checks: Per-check results
        suspects: Features flagged as potential leakage
        top_features: Top-3 most predictive features (name, check, value)
    """
    clean: bool
    n_warnings: int
    checks: list[CheckResult] = field(default_factory=list)
    suspects: list[SuspectFeature] = field(default_factory=list)
    top_features: list[tuple] = field(default_factory=list)

    @property
    def max_severity(self) -> str:
        """Worst severity across all checks: 'ok', 'warn', or 'critical'."""
        if any(c.severity == "critical" for c in self.checks):
            return "critical"
        if any(c.severity == "warn" for c in self.checks):
            return "warn"
        return "ok"

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        n_checks = len(self.checks)
        if self.clean:
            header = "── Leak [CLEAR] "
            header += "─" * max(1, 40 - len(header))
            lines = [header]
            lines.append(f"{n_checks} checks passed, 0 warnings")
        else:
            sev_label = "CRITICAL" if self.max_severity == "critical" else "WARNING"
            header = f"── Leak [{sev_label} — {self.n_warnings} issue{'s' if self.n_warnings != 1 else ''}] "
            header += "─" * max(1, 40 - len(header))
            lines = [header]
            lines.append(f"{n_checks} checks, {self.n_warnings} warning{'s' if self.n_warnings != 1 else ''}")

        # Show check details if there are warnings
        if not self.clean:
            lines.append("")
            name_width = max((len(c.name) for c in self.checks), default=20)
            for c in self.checks:
                if c.passed:
                    icon = "–" if "skipped" in c.detail else "✓"
                elif c.severity == "critical":
                    icon = "!!"
                else:
                    icon = "!"
                lines.append(f"{c.name:<{name_width}}  {icon}  {c.detail}")
                # Show action for flagged checks
                if not c.passed:
                    for s in self.suspects:
                        if s.check == c.name:
                            lines.append(f"{'':>{name_width}}     → {s.action}")
                            break

        # Top features line (deduplicate by feature name)
        if self.top_features:
            seen = set()
            deduped = []
            for name, check, detail in self.top_features:
                if name not in seen:
                    seen.add(name)
                    deduped.append((name, check, detail))
            parts = [f"{name} {detail}" for name, _, detail in deduped[:3]]
            lines.append(f"Top features: {', '.join(parts)}")

        return "\n".join(lines) + "\n"

    def _repr_html_(self) -> str:
        return f'<pre style="font-family: monospace;">{self.__str__()}</pre>'


# ===== DATA STRUCTURES =====


@dataclass
class SplitResult:
    """Result from three-way split.

    Attributes:
        train: Training data (60% default) - fit here, iterate
        valid: Validation data (20% default) - evaluate here, iterate freely
        test: Held-out test data (20% default) - assess here, do once

    Properties:
        dev: Development data (train + valid combined) - for final refit before assessment.
             Retrains the same algorithm on more data.

    Example:
        >>> import ml
        >>> data = ml.dataset("churn")
        >>> s = ml.split(data, "churn", seed=42)
        >>> s.train.shape, s.valid.shape, s.test.shape
        ((3000, 5), (1000, 5), (1000, 5))
        >>> s.dev.shape  # train + valid combined
        (4000, 5)
    """
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    _target: str | None = None
    _task: str | None = None
    _seed: int | None = None
    _temporal: bool = False

    @cached_property
    def dev(self) -> pd.DataFrame:
        """Development data = train + valid combined.

        Use for final refit before assessment (same hyperparameters, bigger data).
        Normalization statistics are recomputed on this larger set.
        """
        result = pd.concat([self.train, self.valid]).reset_index(drop=True)
        result.attrs["_ml_partition"] = "dev"
        # Layer 1: Register dev partition in provenance registry
        from ._provenance import get_split_id, register_partition
        train_sid = get_split_id(self.train)
        if train_sid:
            register_partition(result, "dev", train_sid)
        return result

    def __iter__(self):
        """Enable tuple unpacking. Adapts to 2-way vs 3-way split.

        Three-way (default): train, valid, test = ml.split(data, target, seed=42)
        Two-way (ratio=(0.8, 0.2)): train, test = ml.split(data, target, ratio=(0.8, 0.2), seed=42)
        """
        if len(self.valid) == 0:
            return iter((self.train, self.test))
        return iter((self.train, self.valid, self.test))

    def __repr__(self) -> str:
        return (
            f"SplitResult(train={self.train.shape}, "
            f"valid={self.valid.shape}, test={self.test.shape})"
        )

    def __str__(self) -> str:
        task_badge = f" · {self._task}" if self._task else ""
        temporal_badge = " · temporal" if self._temporal else ""
        header = f"── Split [{self._target or '?'}{task_badge}{temporal_badge}] "
        header += "─" * max(1, 40 - len(header))
        lines = [header]

        # Class balance for classification
        if self._task == "classification" and self._target and self._target in self.train.columns:
            all_data = pd.concat([self.train, self.valid, self.test])
            counts = all_data[self._target].value_counts()
            minority_pct = counts.iloc[-1] / len(all_data)
            lines.append(f"Balance: {minority_pct:.0%} minority class")

        total = len(self.train) + len(self.valid) + len(self.test)

        def _pct(n: int) -> str:
            return f"({n / total:.0%})" if total > 0 else ""

        n_tr, n_va, n_te = len(self.train), len(self.valid), len(self.test)
        n_dev = n_tr + n_va
        lines.append(f"Train:  {n_tr:>6,}  {_pct(n_tr)}")
        lines.append(f"Valid:  {n_va:>6,}  {_pct(n_va)}")
        lines.append(f"Dev:    {n_dev:>6,}  {_pct(n_dev)}")
        lines.append(f"Test:   {n_te:>6,}  {_pct(n_te)}")

        if self._seed is not None:
            lines.append(f"Seed:   {self._seed}")

        return "\n".join(lines) + "\n"


@dataclass
class CVResult:
    """Result from cross-validation split.

    Stores DataFrame slices (not raw indices) so folds can be iterated
    and passed directly to ml.fit().

    Attributes:
        _data: Reference to original data (NOT a copy)
        folds: List of (train_df, valid_df) DataFrame tuples for each fold
        k: Number of folds
        target: Target column name (for stratification info)

    Note:
        CVResult does NOT have .train, .valid, .test, or .dev attributes.
        Accessing them raises ConfigError. Use model.scores_ for CV metrics,
        or split without folds= for train/valid/test.

    Example:
        >>> import ml
        >>> data = ml.dataset("churn")
        >>> cv = ml.split(data, "churn", folds=5, seed=42)
        >>> cv.k
        5
        >>> for train_df, valid_df in cv.folds:
        ...     model = ml.fit(train_df, "churn", seed=42)
    """
    _data: pd.DataFrame
    folds: list[tuple[pd.DataFrame, pd.DataFrame]]
    k: int
    target: str | None
    _temporal: bool = False

    def __post_init__(self) -> None:
        # split.py passes raw index sequences (np.ndarray or list) for memory
        # efficiency during construction; normalise to (DataFrame, DataFrame)
        # here so callers can iterate folds and pass them directly to ml.fit().
        if self.folds and not isinstance(self.folds[0][0], pd.DataFrame):
            self.folds = [
                (self._data.iloc[train_idx], self._data.iloc[valid_idx])
                for train_idx, valid_idx in self.folds
            ]

    def __getattr__(self, name: str) -> Any:
        """Block access to .train, .valid, .test, .dev attributes."""
        if name in ("train", "valid", "test", "dev"):
            raise ConfigError(
                f"No {name} set — this is a CVResult. "
                "Use model.scores_ for CV metrics, or split without folds= for train/valid/test."
            )
        raise AttributeError(f"'CVResult' object has no attribute '{name}'")

    def __repr__(self) -> str:
        n_rows = len(self._data)
        return f"CVResult(folds={self.k}, rows={n_rows})"


@dataclass
class Model:
    """Fitted model.

    Attributes:
        _model: Underlying fitted estimator (internal)
        _task: "classification" or "regression"
        _algorithm: Algorithm name (e.g., "xgboost", "random_forest")
        _features: Feature column names
        _target: Target column name
        _seed: Random seed used
        _label_encoder: LabelEncoder for string targets (or None)
        _feature_encoder: Feature encoding state from _normalize (NormState)
        _n_train: Number of training rows
        scores_: CV scores dict (set by _fit_cv, None for holdout)
        _assess_count: Number of times assess() has been called (NOT persisted)

    Public properties:
        task, algorithm, features, target, seed (read-only access)

    Methods:
        predict(data): DataFrame → Series
        predict_proba(data): DataFrame → ndarray (classification only)

    Example:
        >>> import ml
        >>> data = ml.dataset("churn")
        >>> s = ml.split(data, "churn", seed=42)
        >>> model = ml.fit(s.train, "churn", seed=42)
        >>> model.task
        'classification'
        >>> model.algorithm
        'xgboost'
        >>> preds = model.predict(s.valid)
        >>> preds.shape
        (1000,)
    """
    _model: Any
    _task: str
    _algorithm: str
    _features: list[str]
    _target: str
    _seed: int
    _label_encoder: Any
    _feature_encoder: Any
    _preprocessor: Any = None  # Custom feature transform (Hook 4: Lego)
    _n_train: int = 0
    _n_train_actual: int | None = None  # Rows used for final fit (may be < _n_train when early stopping carves 10%)
    scores_: dict | None = None
    fold_scores_: list | None = None  # per-fold metrics from CV (list of dicts)
    _time: float | None = None  # training wall-clock seconds
    _balance: bool = False  # True if fit() used balance=True
    _calibrated: bool = False  # True if calibrate() was applied
    _assess_count: int = 0
    _workflow_state: int = 1  # WorkflowState.FITTED — fit() produces FITTED models
    # A5: OOF predictions from stack()
    _oof_predictions: Any = None  # pd.DataFrame | None — columns: {algo}_oof or {algo}_oof_class_{k}
    _base_models: Any = None      # list[(algo_name, fitted_estimator)] | None
    # A12: sample weight column name (stored for reference)
    _sample_weight_col: str | None = None
    # W35-F1: holdout estimate score (early-stopping eval or train/eval split)
    _holdout_score: float | None = None
    # A8: seed averaging — list of sub-Models, per-seed scores, score std
    _ensemble: Any = None        # list[Model] | None — sub-models when seed=list
    _seed_scores: Any = None     # list[float] | None — CV score per seed
    _seed_std: float | None = None  # std of seed scores (stability diagnostic)
    # A4: threshold optimisation — set by optimize()
    _threshold: float | None = None  # decision threshold; None means use default (0.5)

    # Public read-only properties
    @property
    def task(self) -> str:
        """Task type: 'classification' or 'regression'."""
        return self._task

    @property
    def algorithm(self) -> str:
        """Algorithm name (e.g., 'xgboost', 'random_forest')."""
        return self._algorithm

    @property
    def features(self) -> list[str]:
        """Feature column names."""
        return self._features

    @property
    def target(self) -> str:
        """Target column name."""
        return self._target

    @property
    def seed(self) -> int:
        """Random seed used for training."""
        return self._seed

    @property
    def calibrated(self) -> bool:
        """Whether this model has been probability-calibrated."""
        return self._calibrated

    @property
    def threshold(self) -> float | None:
        """Decision threshold set by optimize(), or None if using default (0.5)."""
        return self._threshold


    @property
    def time(self) -> float | None:
        """Training wall-clock time in seconds, or None if not recorded."""
        return self._time

    @property
    def n_train(self) -> int:
        """Number of training rows used to fit this model."""
        return self._n_train

    @property
    def hash(self) -> str:
        """Short 8-char hash identifying this model's full configuration.

        Derived from algorithm, task, target, features, seed, training size,
        and hyperparameters. Two models with identical config produce the same hash.

        Example:
            >>> model.hash
            'a3f7b2c1'
        """
        sig = f"{self._algorithm}|{self._task}|{self._target}|{self._seed}|{self._n_train}|{','.join(sorted(self._features))}"
        # Include hyperparameters so tuned models get different hashes
        if hasattr(self._model, 'get_params'):
            params = self._model.get_params()
        else:
            params = {k: v for k, v in self._model.__dict__.items()
                      if not k.startswith("_") and k != "classes_"}
        if params:
            sig += "|" + str(sorted(params.items()))
        return hashlib.sha256(sig.encode()).hexdigest()[:8]

    @property
    def oof_predictions_(self):
        """Out-of-fold predictions from stack() base models, or None.

        Returns a DataFrame with one column per base model:
        - Binary classification: ``{algo}_oof`` (positive-class probability)
        - Multiclass:            ``{algo}_oof_class_{k}`` per class k
        - Regression:            ``{algo}_oof`` (predicted value)
        - Non-stacked model:     None

        Use these for ensemble search, pseudo-labeling, and leakage checks.
        Rows are aligned with the training data passed to stack().
        """
        return self._oof_predictions

    @property
    def preprocessing_(self) -> dict:
        """Human-readable summary of preprocessing applied during fit.

        Returns:
            Dict with 'features' (column → encoding), 'scaled' (bool),
            and 'target' (label mapping or None).

        Example:
            >>> model.preprocessing_
            {'features': {'age': 'numeric', 'gender': 'onehot (2 categories)'},
             'scaled': True, 'target': {'no': 0, 'yes': 1}}
        """
        if self._feature_encoder is None:
            return {}
        enc = self._feature_encoder
        info: dict[str, Any] = {"features": {}, "scaled": False, "target": None}

        for col in enc.feature_names:
            if col in enc.category_maps:
                n = len(enc.category_maps[col])
                info["features"][col] = f"ordinal ({n} categories)"
            elif col in (enc.onehot_columns or []):
                if enc.onehot_encoder is not None:
                    idx = enc.onehot_columns.index(col)
                    n = len(enc.onehot_encoder.categories_[idx])
                    info["features"][col] = f"onehot ({n} categories)"
            else:
                info["features"][col] = "numeric"

        info["scaled"] = enc.scaler is not None

        if enc.label_encoder is not None:
            info["target"] = {
                str(c): int(i) for i, c in enumerate(enc.label_encoder.classes_)
            }

        return info

    @property
    def classes_(self):
        """Class labels for classification models.

        Returns the original class labels in the order used by predict_proba().
        None for regression models.
        """
        if self._task != "classification":
            return None
        if self._label_encoder is not None:
            return list(self._label_encoder.classes_)
        return list(self._model.classes_)

    @property
    def cv_score(self):
        """Best performance score from fitting.

        For CV fits (folds= was used): returns the mean CV score across folds.
        For holdout fits (no folds=): returns a holdout estimate computed during
        fit() — either the early-stopping eval score (gradient boosting) or a
        quick train/eval split estimate (other algorithms). This is labelled
        "holdout_estimate" rather than a true CV score.

        This attribute holds the score recorded during ml.fit() without
        requiring new data.
        """
        # CV path: return mean CV score
        if self.scores_ is not None:
            for k, v in self.scores_.items():
                if k.endswith("_mean"):
                    return v
            # fallback: first value
            if self.scores_:
                return next(iter(self.scores_.values()))
            return None
        # Holdout path: return pre-computed holdout estimate (may be None for
        # very small datasets or algorithms where estimation was skipped)
        return self._holdout_score

    def predict(
        self, data: pd.DataFrame, *, proba: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Predict on new data.

        Args:
            data: DataFrame with same features as training data
            proba: If True, return class probabilities instead of predictions.
                Classification only — raises ModelError for regression.

        Returns:
            Series of predictions (proba=False), or DataFrame of class
            probabilities (proba=True, classification only)

        Example:
            >>> preds = model.predict(s.valid)
            >>> probs = model.predict(s.valid, proba=True)
        """
        from .predict import _predict_impl  # late import — avoids circular _types ↔ predict
        return _predict_impl(self, data, proba=proba)

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities (classification only).

        Args:
            data: DataFrame with same features as training data

        Returns:
            DataFrame of shape (n_samples, n_classes). Columns are class
            labels matching model.classes_. All values float64.

        Raises:
            ModelError: If task is regression
            DataError: If features don't match training data

        Example:
            >>> probs = model.predict_proba(s.valid)
            >>> probs["yes"]  # probability of class "yes"
            0    0.87
            1    0.12
        """
        from .predict import _predict_proba  # late import — same pattern as predict()
        return _predict_proba(self, data)

    def __repr__(self) -> str:
        n_feat = len(self._features)
        cv = f", cv_metrics={len(self.scores_)}" if self.scores_ else ""
        cal = ", calibrated=True" if self._calibrated else ""
        n_train_str = str(self._n_train)
        if self._n_train_actual is not None and self._n_train_actual != self._n_train:
            n_train_str = f"{self._n_train_actual}/{self._n_train}"
        return (
            f"Model(algorithm='{self._algorithm}', task='{self._task}', "
            f"features={n_feat}, n_train={n_train_str}, seed={self._seed}{cv}{cal})"
        )

    def __str__(self) -> str:
        cal_badge = " · calibrated" if self._calibrated else ""
        badge = f"{self._algorithm} · {self._task}{cal_badge}"
        header = f"── Model [{badge}] "
        header += "─" * max(1, 40 - len(header))

        lines = [header]
        lines.append(f"Target:   {self._target}")
        lines.append(f"Features: {len(self._features)}")

        # Rows with optional timing and early stopping disclosure
        if self._n_train_actual is not None and self._n_train_actual != self._n_train:
            row_str = f"Rows:     {self._n_train_actual:,} (of {self._n_train:,} — early stopping held out {self._n_train - self._n_train_actual:,})"
        else:
            row_str = f"Rows:     {self._n_train:,}"
        if self._time is not None:
            if self._time < 0.1:
                row_str += f"  ({self._time:.2f}s)"
            elif self._time < 10:
                row_str += f"  ({self._time:.1f}s)"
            else:
                row_str += f"  ({self._time:.0f}s)"
        lines.append(row_str)

        # A8: seed averaging display
        if self._ensemble is not None:
            n_seeds = len(self._ensemble)
            lines.append(f"Seed:     {self._seed} (+{n_seeds - 1} averaged, n={n_seeds})")
            if self._seed_std is not None:
                stability = "stable" if self._seed_std < 0.01 else "unstable — consider more regularization"
                lines.append(f"SeedStd:  {self._seed_std:.4f} ({stability})")
        else:
            lines.append(f"Seed:     {self._seed}")
        lines.append(f"Hash:     {self.hash}")

        if self.scores_:
            lines.append("")
            lines.append("CV scores:")
            for k, v in self.scores_.items():
                if k.endswith("_mean"):
                    label = k[:-5]  # strip _mean suffix
                    std_key = f"{label}_std"
                    std = self.scores_.get(std_key)
                    if std is not None:
                        lines.append(f"  {label}: {v:.4f} ± {std:.4f}")
                    else:
                        lines.append(f"  {label}: {v:.4f}")

        return "\n".join(lines) + "\n"


@dataclass
class PreparedData:
    """Result of ml.prepare(). Grammar primitive #2: DataFrame -> PreparedData.

    Wraps transformed feature data and the preprocessing state used to produce it.
    The state can be applied to new data (e.g. validation/test sets) via state.transform().

    Attributes:
        data: Transformed DataFrame (all-numeric, ready for ml.fit())
        state: NormState with the fitted encoding/scaling (use state.transform(X) on new data)
        target: Name of the target column
        task: "classification" or "regression"
    """

    data: pd.DataFrame
    state: Any
    target: str
    task: str
    _target_values: pd.Series | None = None


@dataclass
class TuningResult:
    """Result from hyperparameter tuning.

    Wraps the best model with tuning metadata. Delegates predict/predict_proba
    to best_model for convenience.

    Attributes:
        best_model: Model fitted with best hyperparameters
        best_params_: dict of best hyperparameter values
        tuning_history_: DataFrame with all trial results (trial, score, params)

    Example:
        >>> tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42)
        >>> tuned.best_params_
        {'max_depth': 6, 'learning_rate': 0.12}
        >>> tuned.tuning_history_
           trial  score  max_depth  learning_rate
        0      3   0.87          6           0.12
        1      1   0.85          4           0.08
        >>> preds = tuned.predict(s.valid)  # delegates to best_model
    """
    best_model: Model
    best_params_: dict
    tuning_history_: pd.DataFrame
    metric_: str = "auto"

    # Convenience delegation to best_model
    @property
    def task(self) -> str:
        return self.best_model.task

    @property
    def algorithm(self) -> str:
        return self.best_model.algorithm

    @property
    def features(self) -> list[str]:
        return self.best_model.features

    @property
    def target(self) -> str:
        return self.best_model.target

    @property
    def seed(self) -> int:
        return self.best_model.seed

    @property
    def hash(self) -> str:
        """Configuration hash (delegated from best_model)."""
        return self.best_model.hash

    @property
    def time(self) -> float | None:
        """Training time (delegated from best_model)."""
        return self.best_model.time

    @property
    def metric(self) -> str:
        """Metric used during tuning (e.g. 'roc_auc', 'rmse', 'accuracy')."""
        return self.metric_

    @property
    def best_params(self) -> dict:
        """Best hyperparameter values (no trailing underscore — ml convention)."""
        return self.best_params_

    @property
    def cv_score(self) -> float | None:
        """Best score from tuning (alias for best_score — consistent with Model.cv_score)."""
        return self.best_score

    @property
    def best_score(self) -> float | None:
        """Best cross-validation score found during tuning."""
        if len(self.tuning_history_) > 0 and "score" in self.tuning_history_.columns:
            return float(self.tuning_history_["score"].iloc[0])
        return None

    @property
    def model(self) -> Model:
        """Alias for best_model (paper convention: ``tuned.model``)."""
        return self.best_model

    @property
    def best_score_(self) -> float | None:
        """Alias for best_score (convention: fitted attrs end in _)."""
        return self.best_score

    def predict(
        self, data: pd.DataFrame, *, proba: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Predict using best model."""
        return self.best_model.predict(data, proba=proba)

    def predict_proba(self, data: pd.DataFrame):
        """Predict probabilities using best model (classification only)."""
        return self.best_model.predict_proba(data)

    def __repr__(self) -> str:
        n_trials = len(self.tuning_history_)
        best_score = ""
        if "score" in self.tuning_history_.columns and len(self.tuning_history_) > 0:
            best_score = f", best_score={self.tuning_history_['score'].iloc[0]:.4f}"
        # Round float params for clean display
        display_params = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in self.best_params_.items()
        }
        return (
            f"TuningResult(algorithm='{self.best_model.algorithm}', "
            f"best_params={display_params}, trials={n_trials}{best_score})"
        )

    def __str__(self) -> str:
        m = self.best_model
        badge = f"{m.algorithm} · {m.task}"
        header = f"── Tuned [{badge}] "
        header += "─" * max(1, 40 - len(header))

        # Column 1: model info
        info = [
            f"Target:   {m.target}",
            f"Features: {len(m.features)}",
            f"Trials:   {len(self.tuning_history_)}",
        ]
        if "score" in self.tuning_history_.columns and len(self.tuning_history_) > 0:
            metric_label = "accuracy" if m.task == "classification" else "rmse"
            info.append(f"Best CV ({metric_label}):  {self.tuning_history_['score'].iloc[0]:.4f}")
        info.append(f"Seed:     {m.seed}")
        info.append(f"Hash:     {m.hash}")

        # Columns 2+3: params split in half
        if self.best_params_:
            items = list(self.best_params_.items())
            params = []
            for k, v in items:
                params.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            mid = (len(params) + 1) // 2
            col2 = params[:mid]
            col3 = params[mid:]

            info_w = max(len(s) for s in info) + 3
            param_w = max(len(s) for s in params) + 3
            n_rows = max(len(info), len(col2))

            lines = [header]
            for i in range(n_rows):
                c1 = info[i] if i < len(info) else ""
                c2 = col2[i] if i < len(col2) else ""
                c3 = col3[i] if i < len(col3) else ""
                row = f"{c1:<{info_w}}{c2:<{param_w}}{c3}"
                lines.append(row.rstrip())
        else:
            lines = [header] + info

        return "\n".join(lines) + "\n"

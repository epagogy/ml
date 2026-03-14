"""validate() — guarantee gate.

Check whether a model meets minimum thresholds (rules) and/or
doesn't regress vs. a baseline model.

"The code isn't the hard part — the guarantee that it works as it should is the hard part."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._types import Model


@dataclass
class ValidateResult:
    """Result from validate() gate check.

    Attributes:
        passed: True if all rules pass AND no regressions detected
        metrics: Current model's metrics on test data
        failures: List of failure descriptions (empty if passed)
        baseline_metrics: Baseline model's metrics (None if no baseline)
        improvements: List of metric improvements vs baseline (empty if no baseline)
        degradations: List of metric degradations vs baseline (empty if no baseline)
    """
    passed: bool
    metrics: dict[str, float]
    failures: list[str] = field(default_factory=list)
    baseline_metrics: dict[str, float] | None = None
    improvements: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    rules_checked: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        """Alias for failures (natural name for 'which rules failed')."""
        return self.failures

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        header = f"── Validate [{status}] "
        header += "─" * max(1, 40 - len(header))
        lines = [header]

        # Show metrics
        if self.metrics:
            max_key = max(len(k) for k in self.metrics)
            for k, v in self.metrics.items():
                lines.append(f"{k:<{max_key}}  {v:.4f}")

        if self.rules_checked:
            lines.append("")
            for r in self.rules_checked:
                lines.append(f"✓ {r}")

        if self.failures:
            lines.append("")
            for f in self.failures:
                lines.append(f"✗ {f}")

        if self.improvements:
            lines.append("")
            for imp in self.improvements:
                lines.append(f"✓ {imp}")

        if self.unchanged:
            lines.append("")
            for u in self.unchanged:
                lines.append(f"─ {u}")

        return "\n".join(lines) + "\n"


def _parse_rule(rule: str) -> tuple[str, float]:
    """Parse a rule string like '>0.85' into (operator, threshold).

    Supported operators: >, >=, <, <=

    Returns:
        (operator, threshold) tuple
    """
    from ._types import ConfigError

    if isinstance(rule, (int, float)):
        import warnings
        coerced = f">={rule}"
        warnings.warn(
            f"Rule value {rule!r} is a number — interpreting as '>={rule}'. "
            f"Use a string like '>={rule}' to be explicit and suppress this warning.",
            UserWarning,
            stacklevel=4,
        )
        return _parse_rule(coerced)

    if not isinstance(rule, str):
        raise ConfigError(
            f"Rule value must be a string, got {type(rule).__name__} ({rule!r}). "
            "Use string operators like: '>0.85', '>=0.90', '<5.0'. "
            "Example: rules={'accuracy': '>0.85', 'roc_auc': '>=0.90'}"
        )

    match = re.match(r"^\s*(>=|<=|>|<)\s*(-?[0-9]+\.?[0-9]*)\s*$", rule)
    if match is None:
        raise ConfigError(
            f"Invalid rule: '{rule}'. "
            "Use: '>0.85', '>=0.90', '<0.5', '<=0.3', '>-0.5'"
        )
    op = match.group(1)
    threshold = float(match.group(2))
    return op, threshold


def _check_rule(value: float, op: str, threshold: float) -> bool:
    """Check if value satisfies the rule."""
    if op == ">":
        return value > threshold
    elif op == ">=":
        return value >= threshold
    elif op == "<":
        return value < threshold
    elif op == "<=":
        return value <= threshold
    return False


# Metrics where lower is better
_LOWER_IS_BETTER = frozenset({"rmse", "mae", "mse", "log_loss"})


def validate(
    model: Model,
    *,
    test: pd.DataFrame,
    rules: dict[str, str] | None = None,
    baseline: Model | None = None,
    tolerance: float = 0.0,
) -> ValidateResult:
    """Validate a model against rules and/or a baseline.

    Two modes (can be combined):

    1. **Absolute gate** (rules): Does the model meet minimum thresholds?
       ``ml.validate(model, test=s.test, rules={"accuracy": ">0.85"})``

    2. **Regression check** (baseline): Is the new model at least as good?
       ``ml.validate(new_model, test=s.test, baseline=old_model)``

    Parameters
    ----------
    model : Model
        Model to validate
    test : pd.DataFrame
        Test data (keyword-only, same as assess)
    rules : dict[str, str], optional
        Metric thresholds. Keys = metric names, values = rule strings.
        Examples: ``{"accuracy": ">0.85", "roc_auc": ">=0.90", "rmse": "<5.0"}``
    baseline : Model, optional
        Previous model to compare against. If any metric regresses beyond
        tolerance, validation fails.
    tolerance : float, default=0.0
        Allowed degradation vs baseline, in **absolute units** (not relative).
        Tolerance is direction-aware:

        - **Higher-is-better** (accuracy, f1, r2, roc_auc): tolerance allows
          the new model to score *lower* by up to this amount.
        - **Lower-is-better** (rmse, mae): tolerance allows the new model to
          score *higher* (worse) by up to this amount.

        Example: ``tolerance=0.02`` with accuracy means 2 percentage points of
        slack. The same ``tolerance=0.02`` with RMSE means 0.02 raw units —
        which may be negligible or enormous depending on your target scale.
        Choose a value meaningful for each metric's range.

    Returns
    -------
    ValidateResult
        .passed (bool), .metrics (dict), .failures (list[str]),
        .baseline_metrics (dict or None), .improvements (list[str]),
        .degradations (list[str])

    Raises
    ------
    ConfigError
        If neither rules nor baseline is provided
    DataError
        If target column not found in test data

    Examples
    --------
    >>> # Absolute gate
    >>> result = ml.validate(model, test=s.test, rules={"accuracy": ">0.85"})
    >>> result.passed
    True
    >>> result.metrics
    {'accuracy': 0.88, 'f1': 0.84, ...}

    >>> # Regression check with tolerance
    >>> result = ml.validate(new_model, test=s.test, baseline=old_model,
    ...                      tolerance=0.02)
    >>> # For accuracy (higher=better): passes if new >= old - 0.02
    >>> # For RMSE (lower=better): passes if new <= old + 0.02
    >>> result.passed
    True
    >>> result.improvements
    ['accuracy: 0.85 → 0.88 (+0.03)']
    """
    from ._compat import to_pandas
    from ._types import ConfigError, TuningResult
    from .evaluate import evaluate

    # Auto-convert Polars/other DataFrames to pandas
    test = to_pandas(test)

    # DFA state transition: validate is idempotent (state unchanged)
    import contextlib

    from ._types import check_workflow_transition
    with contextlib.suppress(Exception):
        if hasattr(model, '_workflow_state'):
            model._workflow_state = check_workflow_transition(
                model._workflow_state, "validate"
            )

    # Partition guard — validate() uses test data like assess()
    from ._provenance import guard_validate
    guard_validate(test)

    # Must have at least one validation mode
    if rules is None and baseline is None:
        raise ConfigError(
            "validate() requires rules= and/or baseline=. "
            "Use: ml.validate(model, test=data, rules={'accuracy': '>0.85'}) "
            "or ml.validate(model, test=data, baseline=old_model)"
        )

    # Validate rules type
    if rules is not None and not isinstance(rules, dict):
        raise ConfigError(
            f"rules= must be a dict of {{metric: rule_str}}. "
            f"Got {type(rules).__name__}. "
            "Example: rules={'accuracy': '>0.85', 'roc_auc': '>=0.90'}"
        )

    # Validate tolerance
    if tolerance < 0:
        raise ConfigError(
            f"tolerance must be >= 0, got {tolerance}. "
            "Tolerance is the allowed degradation in absolute units. "
            "Example: tolerance=0.02"
        )

    # Unwrap TuningResult → Model
    if isinstance(model, TuningResult):
        model = model.best_model
    if baseline is not None and isinstance(baseline, TuningResult):
        baseline = baseline.best_model

    # Evaluate current model
    # NOTE: validate() does NOT increment _assess_count.
    # validate() is a gate check (pass/fail), not a final exam.
    # Only assess() counts as "peeking at test data".
    metrics = evaluate(model, test, _guard=False)

    failures = []
    improvements = []
    degradations = []
    unchanged = []
    rules_checked = []
    baseline_metrics = None

    # Mode 1: Absolute gate (rules)
    if rules is not None:
        for metric_name, rule_str in rules.items():
            if metric_name not in metrics:
                failures.append(
                    f"Rule '{metric_name}': metric not available. "
                    f"Available: {list(metrics.keys())}"
                )
                continue

            op, threshold = _parse_rule(rule_str)
            value = metrics[metric_name]

            if not _check_rule(value, op, threshold):
                failures.append(
                    f"Rule '{metric_name} {op} {threshold}' FAILED: "
                    f"actual={value:.4f}"
                )
            else:
                rules_checked.append(
                    f"Rule '{metric_name} {op} {threshold}' passed: "
                    f"actual={value:.4f}"
                )

    # Mode 2: Regression check (baseline)
    if baseline is not None:
        if hasattr(baseline, '_target') and hasattr(model, '_target'):
            if baseline._target != model._target:
                raise ConfigError(
                    f"model target={model._target!r} != baseline target="
                    f"{baseline._target!r}. Both must predict the same target."
                )
        baseline_metrics = evaluate(baseline, test, _guard=False)

        for metric_name in metrics:
            if metric_name not in baseline_metrics:
                continue

            new_val = metrics[metric_name]
            old_val = baseline_metrics[metric_name]
            diff = new_val - old_val

            lower_better = metric_name in _LOWER_IS_BETTER

            if lower_better:
                # For error metrics: lower is better, so negative diff = improvement
                if diff < -1e-10:
                    improvements.append(
                        f"{metric_name}: {old_val:.4f} → {new_val:.4f} ({diff:+.4f})"
                    )
                elif diff > tolerance + 1e-10:
                    degradations.append(
                        f"{metric_name}: {old_val:.4f} → {new_val:.4f} ({diff:+.4f})"
                    )
                    failures.append(
                        f"Degradation: {metric_name} degraded from "
                        f"{old_val:.4f} to {new_val:.4f} ({diff:+.4f})"
                    )
                else:
                    unchanged.append(
                        f"{metric_name}: {old_val:.4f} → {new_val:.4f} (within tolerance)"
                    )
            else:
                # For performance metrics: higher is better, so positive diff = improvement
                if diff > 1e-10:
                    improvements.append(
                        f"{metric_name}: {old_val:.4f} → {new_val:.4f} ({diff:+.4f})"
                    )
                elif diff < -(tolerance + 1e-10):
                    degradations.append(
                        f"{metric_name}: {old_val:.4f} → {new_val:.4f} ({diff:+.4f})"
                    )
                    failures.append(
                        f"Degradation: {metric_name} degraded from "
                        f"{old_val:.4f} to {new_val:.4f} ({diff:+.4f})"
                    )
                else:
                    unchanged.append(
                        f"{metric_name}: {old_val:.4f} → {new_val:.4f} (within tolerance)"
                    )

    passed = len(failures) == 0

    return ValidateResult(
        passed=passed,
        metrics=metrics,
        failures=failures,
        baseline_metrics=baseline_metrics,
        improvements=improvements,
        degradations=degradations,
        unchanged=unchanged,
        rules_checked=rules_checked,
    )
